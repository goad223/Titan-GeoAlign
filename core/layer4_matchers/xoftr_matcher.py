"""XoFTR SAR-Optical cross-modal transformer matcher."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class XoFTRMatcher(BaseMatcher):
    """XoFTR — cross-modal transformer matcher for SAR-Optical data.

    Primary:  XoFTR weights from ``models/matchers/xoftr.pth``
    Fallback: SIFT with contrast-limited equalisation for cross-modal robustness
    """

    name = "xoftr"

    def __init__(
        self,
        device: str = "auto",
        models_dir: str = "models/matchers",
        resize: tuple[int, int] = (640, 480),
        match_threshold: float = 0.2,
    ) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.resize = resize
        self.match_threshold = match_threshold
        self._model: Any = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        weights_path = Path(self.models_dir) / "xoftr.pth"

        # 1. Try XoFTR package or reference implementation
        try:
            from xoftr import XoFTR  # type: ignore[import]
            import torch

            cfg = {"match_threshold": self.match_threshold}
            self._model = XoFTR(config=cfg)
            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device)
                self._model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
            self._model.to(self.device).eval()
            self._backend = "xoftr"
            self._loaded = True
            self.logger.info("XoFTR loaded", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("xoftr package not available", error=str(exc))

        # 2. LoFTR via kornia as SAR-Optical-capable fallback
        try:
            import kornia.feature as KF  # type: ignore[import]
            import torch

            self._model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()
            self._backend = "loftr"
            self._loaded = True
            self.logger.info("XoFTR using LoFTR (kornia) fallback", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("kornia LoFTR not available", error=str(exc))

        # 3. SIFT + CLAHE scale-invariant fallback
        self._backend = "sift_clahe"
        self._loaded = True
        self.logger.warning("XoFTR using SIFT+CLAHE fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend == "xoftr":
            return self._match_xoftr(img0, img1)
        if self._backend == "loftr":
            return self._match_loftr(img0, img1)
        return self._sift_clahe(img0, img1)

    # ------------------------------------------------------------------
    def _prep_gray_tensor(self, img: np.ndarray) -> "torch.Tensor":
        import torch
        import cv2  # type: ignore[import]

        gray = (self._to_gray(img) * 255).astype(np.uint8)
        gray_resized = cv2.resize(gray, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(gray_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        return t

    def _match_xoftr(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch

        t0 = time.perf_counter()
        try:
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            t0_img = self._prep_gray_tensor(img0)
            t1_img = self._prep_gray_tensor(img1)
            batch = {"image0": t0_img, "image1": t1_img}
            with torch.no_grad():
                self._model(batch)
            kpts0 = batch["mkpts0_f"].cpu().numpy().astype(np.float32)
            kpts1 = batch["mkpts1_f"].cpu().numpy().astype(np.float32)
            conf = batch["mconf"].cpu().numpy().astype(np.float32)
            # Scale to original image size
            kpts0[:, 0] *= w0 / self.resize[0]
            kpts0[:, 1] *= h0 / self.resize[1]
            kpts1[:, 0] *= w1 / self.resize[0]
            kpts1[:, 1] *= h1 / self.resize[1]
            elapsed = time.perf_counter() - t0
            return MatchResult(kpts0=kpts0, kpts1=kpts1, confidence=conf, matcher_name=self.name, processing_time_s=elapsed)
        except Exception as exc:
            self.logger.warning("XoFTR inference failed; falling back", error=str(exc))
            return self._sift_clahe(img0, img1)

    def _match_loftr(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch

        t0 = time.perf_counter()
        try:
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            t0_img = self._prep_gray_tensor(img0)
            t1_img = self._prep_gray_tensor(img1)
            batch = {"image0": t0_img, "image1": t1_img}
            with torch.no_grad():
                correspondences = self._model(batch)
            kpts0 = correspondences["keypoints0"].cpu().numpy().astype(np.float32)
            kpts1 = correspondences["keypoints1"].cpu().numpy().astype(np.float32)
            conf = correspondences["confidence"].cpu().numpy().astype(np.float32)
            kpts0[:, 0] *= w0 / self.resize[0]
            kpts0[:, 1] *= h0 / self.resize[1]
            kpts1[:, 0] *= w1 / self.resize[0]
            kpts1[:, 1] *= h1 / self.resize[1]
            elapsed = time.perf_counter() - t0
            return MatchResult(kpts0=kpts0, kpts1=kpts1, confidence=conf, matcher_name=self.name, processing_time_s=elapsed)
        except Exception as exc:
            self.logger.warning("LoFTR inference failed; falling back", error=str(exc))
            return self._sift_clahe(img0, img1)

    # ------------------------------------------------------------------
    def _sift_clahe(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        """SIFT with CLAHE pre-processing for cross-modal robustness."""
        import cv2  # type: ignore[import]

        t0 = time.perf_counter()

        def _prep(img: np.ndarray) -> np.ndarray:
            gray = (self._to_gray(img) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)

        g0 = _prep(img0)
        g1 = _prep(img1)
        sift = cv2.SIFT_create(nfeatures=4000)
        kp0, des0 = sift.detectAndCompute(g0, None)
        kp1, des1 = sift.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 100})
        raw = flann.knnMatch(des0, des1, k=2)
        good = [(m, n) for m, n in raw if m.distance < 0.75 * n.distance]
        if not good:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        pts0 = np.array([kp0[m.queryIdx].pt for m, _ in good], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m, _ in good], dtype=np.float32)
        conf = np.clip([1.0 - m.distance / (n.distance + 1e-8) for m, n in good], 0.0, 1.0).astype(np.float32)
        elapsed = time.perf_counter() - t0
        return MatchResult(kpts0=pts0, kpts1=pts1, confidence=conf, matcher_name=self.name, processing_time_s=elapsed)
