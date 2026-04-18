"""XFeat feature extractor + LightGlue matcher via kornia."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class XFeatLightGlueMatcher(BaseMatcher):
    """XFeat + LightGlue matcher.

    Uses kornia.feature.DISK (or XFeat-style descriptor) with kornia.feature.LightGlue.
    Fallback: SIFT + ratio test.
    """

    name = "xfeat_lightglue"

    def __init__(
        self,
        device: str = "auto",
        models_dir: str = "models/matchers",
        num_features: int = 2048,
        resize_max: int = 1024,
    ) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.num_features = num_features
        self.resize_max = resize_max
        self._extractor: Any = None
        self._matcher: Any = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        # 1. Try kornia DISK + LightGlue
        try:
            import kornia.feature as KF  # type: ignore[import]
            import torch  # type: ignore[import]

            self._extractor = KF.DISK.from_pretrained("depth").to(self.device).eval()
            self._matcher = KF.LightGlueMatcher("disk").to(self.device).eval()
            self._backend = "disk_lightglue"
            self._loaded = True
            self.logger.info("XFeat/LightGlue loaded via kornia DISK+LightGlue", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("kornia DISK+LightGlue not available", error=str(exc))

        # 2. Try kornia SuperPoint + LightGlue
        try:
            import kornia.feature as KF  # type: ignore[import]
            import torch  # type: ignore[import]

            self._extractor = KF.SuperPoint(pretrained=True).to(self.device).eval()
            self._matcher = KF.LightGlueMatcher("superpoint").to(self.device).eval()
            self._backend = "superpoint_lightglue"
            self._loaded = True
            self.logger.info("XFeat/LightGlue loaded via kornia SP+LightGlue", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("kornia SP+LightGlue not available", error=str(exc))

        # 3. SIFT fallback
        self._backend = "sift"
        self._loaded = True
        self.logger.warning("XFeatLightGlue using SIFT fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend in ("disk_lightglue", "superpoint_lightglue"):
            return self._match_kornia(img0, img1)
        return self._sift_match(img0, img1)

    # ------------------------------------------------------------------
    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.resize_max / max(h, w)
        if scale < 1.0:
            import cv2  # type: ignore[import]

            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return img

    def _match_kornia(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch
        import kornia.feature as KF  # type: ignore[import]

        t0 = time.perf_counter()
        try:
            img0r = self._resize_if_needed(img0)
            img1r = self._resize_if_needed(img1)
            h0r, w0r = img0r.shape[:2]
            h1r, w1r = img1r.shape[:2]
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]

            def _to_tensor_gray(img: np.ndarray) -> torch.Tensor:
                gray = self._to_gray(img)
                return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(self.device)

            def _to_tensor_rgb(img: np.ndarray) -> torch.Tensor:
                p = self._preprocess(img)
                if p.ndim == 2:
                    p = np.stack([p, p, p], axis=-1)
                p = p[:, :, :3]
                return torch.from_numpy(p.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                if self._backend == "disk_lightglue":
                    t0_img = _to_tensor_rgb(img0r)
                    t1_img = _to_tensor_rgb(img1r)
                    feats0 = self._extractor(t0_img, n=self.num_features)
                    feats1 = self._extractor(t1_img, n=self.num_features)
                    kps0 = feats0.keypoints[0]  # [N, 2]
                    kps1 = feats1.keypoints[0]
                    desc0 = feats0.descriptors[0]  # [N, D]
                    desc1 = feats1.descriptors[0]
                    # Build dicts for LightGlue
                    lg_in = {
                        "image0": {"keypoints": kps0.unsqueeze(0), "descriptors": desc0.unsqueeze(0), "image_size": torch.tensor([[w0r, h0r]], device=self.device)},
                        "image1": {"keypoints": kps1.unsqueeze(0), "descriptors": desc1.unsqueeze(0), "image_size": torch.tensor([[w1r, h1r]], device=self.device)},
                    }
                else:  # superpoint_lightglue
                    t0_img = _to_tensor_gray(img0r)
                    t1_img = _to_tensor_gray(img1r)
                    feats0 = self._extractor({"image": t0_img})
                    feats1 = self._extractor({"image": t1_img})
                    kps0 = feats0["keypoints"][0]
                    kps1 = feats1["keypoints"][0]
                    lg_in = {
                        "image0": {**feats0, "image": t0_img},
                        "image1": {**feats1, "image": t1_img},
                    }

                matches_out = self._matcher(lg_in)

            if "matches" in matches_out:
                idxs = matches_out["matches"][0].cpu().numpy()  # [M, 2]
                conf = matches_out.get("scores", [None])[0]
                if conf is not None:
                    conf = conf.cpu().numpy().astype(np.float32)
                else:
                    conf = np.ones(len(idxs), dtype=np.float32)
                kps0_np = kps0.cpu().numpy()
                kps1_np = kps1.cpu().numpy()
                pts0 = kps0_np[idxs[:, 0]].astype(np.float32)
                pts1 = kps1_np[idxs[:, 1]].astype(np.float32)
            else:
                m0 = matches_out["matches0"][0].cpu().numpy()
                valid = m0 > -1
                kps0_np = kps0.cpu().numpy()
                kps1_np = kps1.cpu().numpy()
                pts0 = kps0_np[valid].astype(np.float32)
                pts1 = kps1_np[m0[valid]].astype(np.float32)
                scores = matches_out.get("matching_scores0", [None])[0]
                conf = scores.cpu().numpy()[valid].astype(np.float32) if scores is not None else np.ones(len(pts0), dtype=np.float32)

            # Scale back to original image coordinates
            pts0[:, 0] *= w0 / w0r
            pts0[:, 1] *= h0 / h0r
            pts1[:, 0] *= w1 / w1r
            pts1[:, 1] *= h1 / h1r

            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=pts0,
                kpts1=pts1,
                confidence=conf,
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("Kornia matching failed; using SIFT", error=str(exc))
            return self._sift_match(img0, img1)

    # ------------------------------------------------------------------
    def _sift_match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import cv2  # type: ignore[import]

        t0 = time.perf_counter()
        g0 = (self._to_gray(img0) * 255).astype(np.uint8)
        g1 = (self._to_gray(img1) * 255).astype(np.uint8)
        sift = cv2.SIFT_create(nfeatures=self.num_features)
        kp0, des0 = sift.detectAndCompute(g0, None)
        kp1, des1 = sift.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
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
