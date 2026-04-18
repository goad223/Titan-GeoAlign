"""GIM-DKMv3 cross-domain generalist matcher with SIFT fallback."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class GIMMatcher(BaseMatcher):
    """GIM DKMv3 matcher — cross-domain generalist.

    Primary:  ``gim`` / ``DKM`` package
    Fallback: SIFT + Lowe ratio test
    """

    name = "gim_dkmv3"

    def __init__(self, device: str = "auto", models_dir: str = "models/matchers", num_keypoints: int = 5000) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.num_keypoints = num_keypoints
        self._model: Any = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        # 1. Try gim package
        try:
            from gim import GIM  # type: ignore[import]

            weights_path = Path(self.models_dir) / "gim_dkmv3.pth"
            self._model = GIM(weights=str(weights_path) if weights_path.exists() else None)
            self._model.to(self.device).eval()
            self._backend = "gim"
            self._loaded = True
            self.logger.info("GIM loaded from gim package", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("gim package not available", error=str(exc))

        # 2. Try DKM package
        try:
            from DKM import DKMv3  # type: ignore[import]

            weights_path = Path(self.models_dir) / "dkmv3_outdoor.pth"
            self._model = DKMv3(pretrained=True, version="outdoor")
            if weights_path.exists():
                import torch
                state = torch.load(weights_path, map_location=self.device)
                self._model.load_state_dict(state, strict=False)
            self._model.to(self.device).eval()
            self._backend = "dkmv3"
            self._loaded = True
            self.logger.info("GIM using DKMv3 backend", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("DKM package not available", error=str(exc))

        # 3. SIFT fallback
        self._backend = "sift"
        self._loaded = True
        self.logger.warning("GIM using SIFT fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend in ("gim", "dkmv3"):
            return self._match_dense(img0, img1)
        return self._sift_fallback(img0, img1)

    # ------------------------------------------------------------------
    def _match_dense(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch
        from PIL import Image  # type: ignore[import]

        t0 = time.perf_counter()
        try:
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]

            def _to_pil(img: np.ndarray) -> Image.Image:
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.dtype != np.uint8:
                    img = (np.clip(self._preprocess(img), 0, 1) * 255).astype(np.uint8)
                return Image.fromarray(img[:, :, :3])

            pil0, pil1 = _to_pil(img0), _to_pil(img1)

            with torch.no_grad():
                if self._backend == "gim":
                    corrs, conf = self._model.match(pil0, pil1)
                else:  # dkmv3
                    warp, certainty = self._model.match(pil0, pil1, device=self.device)
                    corrs, conf = self._model.sample(warp, certainty, num=self.num_keypoints)

            corrs_np = corrs.cpu().numpy() if hasattr(corrs, "cpu") else np.array(corrs)
            conf_np = conf.cpu().numpy().flatten() if hasattr(conf, "cpu") else np.array(conf).flatten()

            kpts0 = np.stack([(corrs_np[:, 0] + 1) / 2 * w0, (corrs_np[:, 1] + 1) / 2 * h0], axis=-1).astype(np.float32)
            kpts1 = np.stack([(corrs_np[:, 2] + 1) / 2 * w1, (corrs_np[:, 3] + 1) / 2 * h1], axis=-1).astype(np.float32)

            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=kpts0,
                kpts1=kpts1,
                confidence=conf_np.astype(np.float32),
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("Dense matching failed; using SIFT", error=str(exc))
            return self._sift_fallback(img0, img1)

    # ------------------------------------------------------------------
    def _sift_fallback(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import cv2  # type: ignore[import]

        t0 = time.perf_counter()
        g0 = (self._to_gray(img0) * 255).astype(np.uint8)
        g1 = (self._to_gray(img1) * 255).astype(np.uint8)

        sift = cv2.SIFT_create(nfeatures=self.num_keypoints)
        kp0, des0 = sift.detectAndCompute(g0, None)
        kp1, des1 = sift.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        raw_matches = flann.knnMatch(des0, des1, k=2)
        # Lowe ratio test
        good = [(m, n) for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if not good:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        pts0 = np.array([kp0[m.queryIdx].pt for m, _ in good], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m, _ in good], dtype=np.float32)
        conf = np.array([1.0 - m.distance / (n.distance + 1e-8) for m, n in good], dtype=np.float32)
        conf = np.clip(conf, 0.0, 1.0)

        elapsed = time.perf_counter() - t0
        return MatchResult(
            kpts0=pts0,
            kpts1=pts1,
            confidence=conf,
            matcher_name=self.name,
            processing_time_s=elapsed,
        )
