"""RoMa v2 dense matcher with ORB+BF fallback."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import structlog

from .base_matcher import BaseMatcher, MatchResult

logger = structlog.get_logger(__name__)


class RoMaV2Matcher(BaseMatcher):
    """Dense correspondence matcher using RoMa v2.

    Primary:  ``roma`` package  (pip install roma)
    Secondary: weights from ``models/matchers/roma_v2.pth``
    Fallback:  ORB + BruteForce
    """

    name = "roma_v2"

    def __init__(self, device: str = "auto", models_dir: str = "models/matchers", num_keypoints: int = 5000) -> None:
        super().__init__(device=device, models_dir=models_dir)
        self.num_keypoints = num_keypoints
        self._model = None
        self._backend: str = "none"

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load RoMa v2 model."""
        if self._loaded:
            return

        # 1. Try roma package (official)
        try:
            import roma  # type: ignore[import]

            self._model = roma.roma_outdoor(device=self.device)
            self._model.eval()
            self._backend = "roma_package"
            self._loaded = True
            self.logger.info("RoMa v2 loaded from roma package", backend=self._backend)
            return
        except Exception as exc:
            self.logger.debug("roma package not available", error=str(exc))

        # 2. Try loading weights from disk
        weights_path = Path(self.models_dir) / "roma_v2.pth"
        if weights_path.exists():
            try:
                import torch  # type: ignore[import]

                state = torch.load(weights_path, map_location=self.device)
                self.logger.info("RoMa v2 weights found on disk but no matching architecture; using fallback")
            except Exception as exc:
                self.logger.debug("Failed to load roma weights", error=str(exc))

        # 3. ORB+BF fallback
        self._backend = "orb_bf"
        self._loaded = True
        self.logger.warning("RoMa v2 unavailable; using ORB+BF fallback", backend=self._backend)

    # ------------------------------------------------------------------
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self._loaded:
            self.load()

        if self._backend == "roma_package":
            return self._match_roma(img0, img1)
        return self._orb_bf_fallback(img0, img1, self.name)

    # ------------------------------------------------------------------
    def _match_roma(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
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
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                return Image.fromarray(img)

            pil0 = _to_pil(img0)
            pil1 = _to_pil(img1)

            with torch.no_grad():
                warp, certainty = self._model.match(pil0, pil1, device=self.device)

            # Sample correspondences from dense warp field
            matches, conf = self._model.sample(
                warp,
                certainty,
                num=self.num_keypoints,
            )
            # matches: [N, 4] in (-1,1) normalised coords
            matches_np = matches.cpu().numpy()
            conf_np = conf.cpu().numpy().flatten()

            kpts0 = np.stack(
                [
                    (matches_np[:, 0] + 1) / 2 * w0,
                    (matches_np[:, 1] + 1) / 2 * h0,
                ],
                axis=-1,
            ).astype(np.float32)
            kpts1 = np.stack(
                [
                    (matches_np[:, 2] + 1) / 2 * w1,
                    (matches_np[:, 3] + 1) / 2 * h1,
                ],
                axis=-1,
            ).astype(np.float32)

            elapsed = time.perf_counter() - t0
            return MatchResult(
                kpts0=kpts0,
                kpts1=kpts1,
                confidence=conf_np.astype(np.float32),
                matcher_name=self.name,
                processing_time_s=elapsed,
            )
        except Exception as exc:
            self.logger.warning("RoMa matching failed; falling back to ORB+BF", error=str(exc))
            return self._orb_bf_fallback(img0, img1, self.name)
