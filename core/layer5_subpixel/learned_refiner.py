"""Neural network-based sub-pixel refiner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class LearnedSubpixelRefiner:
    """CNN-based sub-pixel offset estimator.

    Architecture: two-branch patch encoder → residual blocks → offset head.
    Estimates (dx, dy) sub-pixel offset for each match.

    If weights are unavailable, falls back to PhaseCorrelationRefiner.
    """

    def __init__(
        self,
        device: str = "auto",
        weights_path: str = "models/subpixel_refiner.pth",
        patch_size: int = 32,
    ) -> None:
        self._device_str = device
        self.weights_path = Path(weights_path)
        self.patch_size = patch_size
        self._model: Any = None
        self._loaded = False
        self._backend = "none"
        self.logger = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            import torch.nn as nn

            if self._device_str == "auto":
                import torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self._device_str)

            model = _SubpixelNet(patch_size=self.patch_size).to(self._device)

            if self.weights_path.exists():
                state = torch.load(self.weights_path, map_location=self._device)
                model.load_state_dict(state)
                self.logger.info("LearnedSubpixelRefiner weights loaded", path=str(self.weights_path))
                self._backend = "cnn"
            else:
                self.logger.warning(
                    "No weights found; LearnedSubpixelRefiner will use phase correlation fallback",
                    path=str(self.weights_path),
                )
                self._backend = "phase_correlation"

            model.eval()
            self._model = model

        except ImportError:
            self.logger.warning("PyTorch not available; using phase correlation fallback")
            self._backend = "phase_correlation"

        self._loaded = True

    # ------------------------------------------------------------------
    def refine(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine keypoints to sub-pixel accuracy.

        Returns
        -------
        refined_kpts0, refined_kpts1 : np.ndarray [N, 2]
        """
        if not self._loaded:
            self.load()

        if self._backend == "cnn":
            return self._refine_cnn(img0, img1, kpts0, kpts1)

        from .phase_correlation import PhaseCorrelationRefiner
        return PhaseCorrelationRefiner(patch_size=self.patch_size).refine(img0, img1, kpts0, kpts1)

    # ------------------------------------------------------------------
    def _refine_cnn(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        kpts0 = np.array(kpts0, dtype=np.float32)
        kpts1 = np.array(kpts1, dtype=np.float32)

        g0 = _to_gray(img0)
        g1 = _to_gray(img1)
        half = self.patch_size // 2
        H0, W0 = g0.shape
        H1, W1 = g1.shape

        refined1 = kpts1.copy()

        with torch.no_grad():
            for i in range(len(kpts0)):
                cx0, cy0 = kpts0[i, 0], kpts0[i, 1]
                cx1, cy1 = kpts1[i, 0], kpts1[i, 1]

                p0 = _extract_patch(g0, cx0, cy0, half, H0, W0)
                p1 = _extract_patch(g1, cx1, cy1, half, H1, W1)
                if p0 is None or p1 is None:
                    continue

                t0 = torch.from_numpy(p0[np.newaxis, np.newaxis]).float().to(self._device)
                t1 = torch.from_numpy(p1[np.newaxis, np.newaxis]).float().to(self._device)
                try:
                    delta = self._model(t0, t1)  # [1, 2]
                    dx = float(delta[0, 0].cpu())
                    dy = float(delta[0, 1].cpu())
                    refined1[i, 0] = cx1 + dx
                    refined1[i, 1] = cy1 + dy
                except Exception as exc:
                    self.logger.debug("CNN inference error for kpt", idx=i, error=str(exc))

        return kpts0, refined1


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype in (np.uint16, np.int16):
        mn, mx = img.min(), img.max()
        img = (img.astype(np.float32) - mn) / (mx - mn + 1e-8)
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    return img


def _extract_patch(img: np.ndarray, cx: float, cy: float, half: int, H: int, W: int) -> np.ndarray | None:
    ix, iy = int(round(cx)), int(round(cy))
    r1, r2 = iy - half, iy + half
    c1, c2 = ix - half, ix + half
    if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
        return None
    return img[r1:r2, c1:c2].copy()


try:
    import torch.nn as nn

    class _ResBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):  # type: ignore[override]
            return self.relu(x + self.net(x))

    class _SubpixelNet(nn.Module):
        """Two-branch CNN → sub-pixel offset regressor."""

        def __init__(self, patch_size: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                _ResBlock(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                _ResBlock(128),
            )
            reduced = patch_size // 4
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * reduced * reduced * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
                nn.Tanh(),
            )

        def forward(self, p0, p1):  # type: ignore[override]
            f0 = self.encoder(p0)
            f1 = self.encoder(p1)
            combined = torch.cat([f0, f1], dim=1)
            return self.head(combined)

    import torch  # noqa: F401 — needed for _SubpixelNet definition

except ImportError:
    # Torch not available; define dummy classes so imports don't fail
    class _ResBlock:  # type: ignore[no-redef]
        pass

    class _SubpixelNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            pass
