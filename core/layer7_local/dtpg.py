"""Dense Tie-Point Grid for local correction."""
from __future__ import annotations
import numpy as np
import structlog
logger = structlog.get_logger(__name__)

class DenseTiePointGrid:
    def __init__(self) -> None:
        self._points: list[dict] = []

    def generate(self, img0: np.ndarray, img1: np.ndarray, grid_size: int = 64) -> list[dict]:
        import cv2  # type: ignore[import]
        H, W = img0.shape[:2]
        g0 = self._to_gray(img0)
        g1 = self._to_gray(img1)
        self._points = []
        rows = max(1, H // grid_size)
        cols = max(1, W // grid_size)
        cell_h = H // rows
        cell_w = W // cols
        for r in range(rows):
            for c in range(cols):
                cy = r * cell_h + cell_h // 2
                cx = c * cell_w + cell_w // 2
                half = min(cell_h, cell_w) // 2
                if cy - half < 0 or cy + half > H or cx - half < 0 or cx + half > W:
                    continue
                p0 = g0[cy-half:cy+half, cx-half:cx+half].astype(np.float64)
                p1 = g1[cy-half:cy+half, cx-half:cx+half].astype(np.float64)
                if p0.shape != p1.shape or p0.size == 0:
                    continue
                try:
                    shift, response = cv2.phaseCorrelate(p0, p1)
                    dx, dy = float(shift[0]), float(shift[1])
                    quality = float(np.clip(response, 0, 1))
                except Exception:
                    dx, dy, quality = 0.0, 0.0, 0.0
                self._points.append({"row": cy, "col": cx, "dx": dx, "dy": dy, "quality": quality})
        return self._points

    def to_array(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([[p["row"], p["col"], p["dx"], p["dy"], p["quality"]] for p in self._points], dtype=np.float32)

    def filter_outliers(self, sigma: float = 3.0) -> "DenseTiePointGrid":
        if not self._points:
            return self
        dxs = np.array([p["dx"] for p in self._points])
        dys = np.array([p["dy"] for p in self._points])
        mx, sx = dxs.mean(), dxs.std() + 1e-8
        my, sy = dys.mean(), dys.std() + 1e-8
        self._points = [p for p in self._points
                        if abs(p["dx"] - mx) < sigma * sx and abs(p["dy"] - my) < sigma * sy]
        return self

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            mn, mx = img.min(), img.max()
            img = (img.astype(np.float32) - mn) / (mx - mn + 1e-8)
        if img.ndim == 3:
            img = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
        return img
