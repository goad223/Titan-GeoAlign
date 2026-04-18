"""OmniGlue foundation-model-guided matcher with AKAZE+BF fallback."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class OmniGlueMatcher(BaseMatcher):
    """OmniGlue — DINO-guided feature matching.

    Primary:  ``omniglue`` package (Google OmniGlue)
    Secondary: DINO features via timm/transformers + cosine-similarity matching
    Fallback:  AKAZE + BFMatcher
    """

    name = "omniglue"

    def __init__(
        self,
        device: str = "auto",
        models_dir: str = "models/matchers",
        og_export: str = "models/matchers/omniglue",
        sp_export: str = "models/matchers/superpoint",
        dino_export: str = "models/matchers/dinov2",
        num_keypoints: int = 2048,
    ) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.og_export = og_export
        self.sp_export = sp_export
        self.dino_export = dino_export
        self.num_keypoints = num_keypoints
        self._model: Any = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        # 1. Try official omniglue package
        try:
            import omniglue  # type: ignore[import]

            self._model = omniglue.OmniGlue(
                og_export=self.og_export,
                sp_export=self.sp_export,
                dino_export=self.dino_export,
            )
            self._backend = "omniglue"
            self._loaded = True
            self.logger.info("OmniGlue loaded from omniglue package", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("omniglue package not available", error=str(exc))

        # 2. DINO + SuperPoint via transformers/timm
        try:
            import torch  # type: ignore[import]
            import kornia.feature as KF  # type: ignore[import]

            self._SP = KF.SuperPoint(pretrained=True).to(self.device).eval()
            # DINOv2 via torch hub
            self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(self.device).eval()
            self._backend = "dino_sp"
            self._loaded = True
            self.logger.info("OmniGlue using DINO+SuperPoint fallback", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("DINO+SuperPoint fallback not available", error=str(exc))

        # 3. AKAZE + BFMatcher
        self._backend = "akaze"
        self._loaded = True
        self.logger.warning("OmniGlue using AKAZE+BF fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend == "omniglue":
            return self._match_omniglue(img0, img1)
        if self._backend == "dino_sp":
            return self._match_dino_sp(img0, img1)
        return self._akaze_match(img0, img1)

    # ------------------------------------------------------------------
    def _to_uint8_rgb(self, img: np.ndarray) -> np.ndarray:
        img_f = self._preprocess(img)
        if img_f.ndim == 2:
            img_f = np.stack([img_f, img_f, img_f], axis=-1)
        img_f = img_f[:, :, :3]
        return (np.clip(img_f, 0, 1) * 255).astype(np.uint8)

    def _match_omniglue(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        t0 = time.perf_counter()
        try:
            rgb0 = self._to_uint8_rgb(img0)
            rgb1 = self._to_uint8_rgb(img1)
            kpts0_raw, kpts1_raw, conf_raw = self._model.FindMatches(rgb0, rgb1)
            kpts0 = np.array(kpts0_raw, dtype=np.float32)
            kpts1 = np.array(kpts1_raw, dtype=np.float32)
            conf = np.array(conf_raw, dtype=np.float32)
            # Filter by confidence
            valid = conf > 0.0
            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=kpts0[valid],
                kpts1=kpts1[valid],
                confidence=conf[valid],
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("OmniGlue inference failed; falling back", error=str(exc))
            return self._akaze_match(img0, img1)

    def _match_dino_sp(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        """SuperPoint keypoints + DINO features for semantic guidance."""
        import torch
        import torch.nn.functional as F
        import kornia.feature as KF  # type: ignore[import]

        t0 = time.perf_counter()
        try:
            def _gray_tensor(img: np.ndarray) -> torch.Tensor:
                g = self._to_gray(img)
                return torch.from_numpy(g).unsqueeze(0).unsqueeze(0).float().to(self.device)

            def _rgb_tensor(img: np.ndarray) -> torch.Tensor:
                p = self._preprocess(img)
                if p.ndim == 2:
                    p = np.stack([p, p, p], axis=-1)
                p = p[:, :, :3]
                return torch.from_numpy(p.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                sp_feats0 = self._SP({"image": _gray_tensor(img0)})
                sp_feats1 = self._SP({"image": _gray_tensor(img1)})

                # DINO patch features (resize to 224×224 multiple)
                dino_size = 224
                t0_dino = F.interpolate(_rgb_tensor(img0), size=(dino_size, dino_size), mode="bilinear", align_corners=False)
                t1_dino = F.interpolate(_rgb_tensor(img1), size=(dino_size, dino_size), mode="bilinear", align_corners=False)

                # Normalize for DINO
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                t0_dino = (t0_dino - mean) / std
                t1_dino = (t1_dino - mean) / std

                dino_out0 = self._dino.forward_features(t0_dino)
                dino_out1 = self._dino.forward_features(t1_dino)
                # Use patch tokens as global semantic descriptors
                dino_desc0 = dino_out0["x_norm_patchtokens"].mean(dim=1)  # [1, D]
                dino_desc1 = dino_out1["x_norm_patchtokens"].mean(dim=1)

            kps0 = sp_feats0["keypoints"][0].cpu().numpy()  # [N0, 2]
            kps1 = sp_feats1["keypoints"][0].cpu().numpy()
            desc0 = sp_feats0["descriptors"][0].cpu().numpy()  # [N0, 256]
            desc1 = sp_feats1["descriptors"][0].cpu().numpy()

            # Semantic similarity bias from DINO
            semantic_sim = float(F.cosine_similarity(dino_desc0, dino_desc1).item())

            # Match with SP descriptors, scaled by semantic similarity
            if len(kps0) == 0 or len(kps1) == 0:
                empty = np.zeros((0, 2), dtype=np.float32)
                return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

            # Cosine similarity between SP descriptors
            d0 = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
            d1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
            sim_mat = d0 @ d1.T  # [N0, N1]

            best1 = np.argmax(sim_mat, axis=1)
            best_score = sim_mat[np.arange(len(d0)), best1]
            # Cross-check
            best0_rev = np.argmax(sim_mat, axis=0)
            mutual = best0_rev[best1] == np.arange(len(d0))
            valid = mutual & (best_score > 0.3)

            pts0 = kps0[valid].astype(np.float32)
            pts1 = kps1[best1[valid]].astype(np.float32)
            conf = (best_score[valid] * max(0.5, semantic_sim)).astype(np.float32)

            elapsed = time.perf_counter() - t0
            return MatchResult(kpts0=pts0, kpts1=pts1, confidence=conf, matcher_name=self.name, processing_time_s=elapsed)
        except Exception as exc:
            self.logger.warning("DINO+SP matching failed; using AKAZE", error=str(exc))
            return self._akaze_match(img0, img1)

    # ------------------------------------------------------------------
    def _akaze_match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import cv2  # type: ignore[import]

        t0 = time.perf_counter()
        g0 = (self._to_gray(img0) * 255).astype(np.uint8)
        g1 = (self._to_gray(img1) * 255).astype(np.uint8)
        akaze = cv2.AKAZE_create()
        kp0, des0 = akaze.detectAndCompute(g0, None)
        kp1, des1 = akaze.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(des0, des1, k=2)
        good = [(m, n) for m, n in raw if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        if not good:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(kpts0=empty, kpts1=empty, confidence=np.zeros(0, dtype=np.float32), matcher_name=self.name)

        pts0 = np.array([kp0[m.queryIdx].pt for m, _ in good], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m, _ in good], dtype=np.float32)
        conf = np.clip([1.0 - m.distance / (n.distance + 1e-8) for m, n in good], 0.0, 1.0).astype(np.float32)
        elapsed = time.perf_counter() - t0
        return MatchResult(kpts0=pts0, kpts1=pts1, confidence=conf, matcher_name=self.name, processing_time_s=elapsed)
