"""UncertaintyMap: dense spatial map of matching uncertainty."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


class UncertaintyMap:
    """Dense (H, W) uncertainty map interpolated from sparse keypoint data."""

    def __init__(self, image_shape: Tuple[int, int]) -> None:
        self.image_shape = image_shape  # (H, W)
        self.map: np.ndarray = np.zeros(image_shape, dtype=np.float32)

    # ------------------------------------------------------------------
    @classmethod
    def from_residuals(
        cls, residuals: np.ndarray, kpts: np.ndarray, image_shape: Tuple[int, int]
    ) -> "UncertaintyMap":
        """Interpolate residual magnitudes to a full image grid.

        Parameters
        ----------
        residuals:
            [N] residual magnitudes.
        kpts:
            [N, 2] keypoint coordinates (x, y).
        image_shape:
            (H, W) output map size.
        """
        instance = cls(image_shape)
        if len(kpts) == 0:
            return instance
        instance.map = _interpolate_to_grid(kpts, residuals, image_shape)
        return instance

    # ------------------------------------------------------------------
    @classmethod
    def from_ensemble(
        cls,
        ensemble_std: np.ndarray,
        kpts: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> "UncertaintyMap":
        """Build uncertainty map from ensemble std dev values.

        Parameters
        ----------
        ensemble_std:
            [N] per-match uncertainty values.
        kpts:
            [N, 2] keypoint coordinates.
        image_shape:
            (H, W) output map size.
        """
        instance = cls(image_shape)
        if len(kpts) == 0:
            return instance
        instance.map = _interpolate_to_grid(kpts, ensemble_std, image_shape)
        return instance

    # ------------------------------------------------------------------
    def visualize(self) -> np.ndarray:
        """Return a colormap-applied (H, W, 3) uint8 RGB image."""
        normalized = self.map.copy()
        max_val = normalized.max()
        if max_val > 0:
            normalized /= max_val
        uint8_map = (normalized * 255).astype(np.uint8)

        if _CV2_AVAILABLE:
            colored = _cv2.applyColorMap(uint8_map, _cv2.COLORMAP_JET)
            colored = _cv2.cvtColor(colored, _cv2.COLOR_BGR2RGB)
            return colored

        # Fallback: simple matplotlib-style jet colormap
        r = np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save the uncertainty map as a float32 numpy .npy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.map)
        logger.debug("uncertainty_map_saved", path=str(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_to_grid(
    kpts: np.ndarray,
    values: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Interpolate sparse values at kpts to a dense grid using RBFInterpolator."""
    h, w = image_shape
    try:
        from scipy.interpolate import RBFInterpolator
        rbf = RBFInterpolator(kpts[:, ::-1], values, kernel="linear", smoothing=1.0)
        gy, gx = np.mgrid[0:h, 0:w]
        grid_pts = np.column_stack([gy.ravel(), gx.ravel()])
        grid_values = rbf(grid_pts).reshape(h, w).astype(np.float32)
        return np.clip(grid_values, 0.0, None)
    except ImportError:
        logger.warning("scipy_not_available", fallback="nearest_neighbor_scatter")
        result = np.zeros((h, w), dtype=np.float32)
        xs = np.clip(kpts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(kpts[:, 1].astype(int), 0, h - 1)
        for xi, yi, v in zip(xs, ys, values):
            result[yi, xi] = max(result[yi, xi], float(v))
        return result
