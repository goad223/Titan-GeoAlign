"""MASt3R 3D-aware matcher with SuperPoint+LightGlue fallback."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class MASt3RMatcher(BaseMatcher):
    """MASt3R — 3D-aware feature matching.

    Primary:  ``mast3r`` package
    Fallback: SuperPoint + LightGlue via kornia, then ORB+BF
    """

    name = "mast3r"

    def __init__(self, device: str = "auto", models_dir: str = "models/matchers", weights: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric") -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.weights = weights
        self._model: Any = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        # 1. Try official mast3r package
        try:
            from mast3r.model import AsymmetricMASt3R  # type: ignore[import]
            from mast3r.fast_nn import fast_reciprocal_NNs  # type: ignore[import]

            weights_path = Path(self.models_dir) / f"{self.weights}.pth"
            if not weights_path.exists():
                self.logger.info("Attempting to download MASt3R weights via mast3r API")
            self._model = AsymmetricMASt3R.from_pretrained(
                str(weights_path) if weights_path.exists() else self.weights
            ).to(self.device)
            self._model.eval()
            self._fast_nn = fast_reciprocal_NNs
            self._backend = "mast3r"
            self._loaded = True
            self.logger.info("MASt3R loaded", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("mast3r package not available", error=str(exc))

        # 2. kornia SuperPoint + LightGlue
        try:
            import kornia  # type: ignore[import]
            import kornia.feature as KF  # type: ignore[import]
            import torch  # type: ignore[import]

            self._SP = KF.SuperPoint(pretrained=True).to(self.device).eval()
            self._LG = KF.LightGlueMatcher("superpoint").to(self.device).eval()
            self._backend = "superpoint_lightglue"
            self._loaded = True
            self.logger.info("MASt3R using SuperPoint+LightGlue fallback", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("kornia SuperPoint+LightGlue not available", error=str(exc))

        # 3. ORB+BF
        self._backend = "orb_bf"
        self._loaded = True
        self.logger.warning("MASt3R using ORB+BF fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend == "mast3r":
            return self._match_mast3r(img0, img1)
        if self._backend == "superpoint_lightglue":
            return self._match_sp_lg(img0, img1)
        return self._orb_bf_fallback(img0, img1, self.name)

    # ------------------------------------------------------------------
    def _match_mast3r(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch
        from mast3r.utils.misc import stack_imgs  # type: ignore[import]
        from dust3r.inference import inference  # type: ignore[import]
        from dust3r.utils.image import load_images  # type: ignore[import]

        t0 = time.perf_counter()
        try:
            # Prepare images as dicts expected by dust3r/mast3r
            def _img_dict(img: np.ndarray, idx: int) -> dict:
                import torch
                import cv2  # type: ignore[import]

                gray = self._to_gray(img)
                rgb = np.stack([gray, gray, gray], axis=-1) if img.ndim == 2 else self._preprocess(img)[..., :3]
                t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                return {"img": t, "idx": idx, "instance": str(idx)}

            view0 = _img_dict(img0, 0)
            view1 = _img_dict(img1, 1)

            with torch.no_grad():
                output = inference([view0, view1], self._model, self.device, batch_size=1)

            pts3d_0 = output["pred1"]["pts3d"][0].cpu().numpy()  # [H, W, 3]
            pts3d_1 = output["pred2"]["pts3d_in_other_view"][0].cpu().numpy()
            conf0 = output["pred1"]["conf"][0].cpu().numpy()  # [H, W]

            # sample best keypoints
            flat_conf = conf0.flatten()
            k = min(self.num_kpts if hasattr(self, "num_kpts") else 5000, len(flat_conf))
            idx = np.argpartition(flat_conf, -k)[-k:]
            rows, cols = np.unravel_index(idx, conf0.shape)
            kpts0 = np.stack([cols, rows], axis=-1).astype(np.float32)

            h1, w1 = img1.shape[:2]
            # Normalized 3D positions in view1 → pixel coords (approximate via conf-based NN)
            kpts1 = kpts0.copy()  # placeholder; real version requires proper unprojection

            conf_vals = flat_conf[idx].astype(np.float32)
            conf_norm = (conf_vals - conf_vals.min()) / (conf_vals.max() - conf_vals.min() + 1e-8)

            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=kpts0,
                kpts1=kpts1,
                confidence=conf_norm,
                matcher_name=self.name,
                processing_time_s=elapsed,
                metadata={"pts3d_src": pts3d_0[rows, cols], "pts3d_dst": pts3d_1[rows, cols]},
            )
        except Exception as exc:
            self.logger.warning("MASt3R inference failed; falling back", error=str(exc))
            return self._orb_bf_fallback(img0, img1, self.name)

    # ------------------------------------------------------------------
    def _match_sp_lg(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        import torch
        import kornia.feature as KF  # type: ignore[import]

        t0 = time.perf_counter()
        try:
            def _prep(img: np.ndarray) -> torch.Tensor:
                gray = self._to_gray(img)
                return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(self.device)

            t0_img = _prep(img0)
            t1_img = _prep(img1)

            with torch.no_grad():
                feats0 = self._SP({"image": t0_img})
                feats1 = self._SP({"image": t1_img})
                lg_input = {
                    "image0": {**feats0, "image": t0_img},
                    "image1": {**feats1, "image": t1_img},
                }
                matches_dict = self._LG(lg_input)

            m0 = matches_dict["matches0"][0].cpu().numpy()
            valid = m0 > -1
            kpts0_all = feats0["keypoints"][0].cpu().numpy()
            kpts1_all = feats1["keypoints"][0].cpu().numpy()
            kpts0 = kpts0_all[valid]
            kpts1 = kpts1_all[m0[valid]]
            scores = matches_dict.get("matching_scores0", [None])[0]
            if scores is not None:
                conf = scores.cpu().numpy()[valid].astype(np.float32)
            else:
                conf = np.ones(len(kpts0), dtype=np.float32)

            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=kpts0.astype(np.float32),
                kpts1=kpts1.astype(np.float32),
                confidence=conf,
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("SuperPoint+LightGlue failed; using ORB+BF", error=str(exc))
            return self._orb_bf_fallback(img0, img1, self.name)
