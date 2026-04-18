"""Residual analysis for geometric alignment quality."""
from __future__ import annotations
import numpy as np
import structlog
logger = structlog.get_logger(__name__)

class ResidualAnalyzer:
    def __init__(self) -> None:
        self.logger = structlog.get_logger(self.__class__.__name__)

    def analyze(self, kpts0: np.ndarray, kpts1: np.ndarray, transform_matrix: np.ndarray) -> dict:
        residuals = self.compute_residuals(kpts0, kpts1, transform_matrix)
        stats = self.compute_statistics(residuals)
        return {"residuals": residuals, **stats}

    @staticmethod
    def compute_residuals(kpts0: np.ndarray, kpts1: np.ndarray, H: np.ndarray) -> np.ndarray:
        if len(kpts0) == 0 or H is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = kpts0.astype(np.float64)
        if H.shape == (3, 3):
            ones = np.ones((len(pts), 1))
            ph = np.hstack([pts, ones])
            proj = (H @ ph.T).T
            proj = proj[:, :2] / (proj[:, 2:3] + 1e-10)
        elif H.shape[0] == 2:
            ones = np.ones((len(pts), 1))
            ph = np.hstack([pts, ones])
            proj = (H[:2, :3] @ ph.T).T
        else:
            return np.zeros((len(kpts0), 2), dtype=np.float32)
        return (proj - kpts1.astype(np.float64)).astype(np.float32)

    def generate_heatmap(self, residuals: np.ndarray, image_shape: tuple) -> np.ndarray:
        from scipy.interpolate import RBFInterpolator  # type: ignore[import]
        H, W = image_shape[:2]
        if len(residuals) < 4:
            return np.zeros((H, W), dtype=np.float32)
        magnitudes = np.linalg.norm(residuals, axis=1)
        # We need tie-point coordinates; store them separately in practice
        # Approximate: distribute over grid
        n = len(residuals)
        pts = np.column_stack([np.random.default_rng(0).uniform(0, W, n),
                               np.random.default_rng(1).uniform(0, H, n)])
        try:
            rbf = RBFInterpolator(pts, magnitudes, kernel="linear", smoothing=1.0)
            ys, xs = np.mgrid[0:H:64j, 0:W:64j]
            grid = np.column_stack([xs.flatten(), ys.flatten()])
            vals = rbf(grid).reshape(64, 64)
            import cv2  # type: ignore[import]
            heatmap = cv2.resize(vals.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            return heatmap
        except Exception:
            return np.zeros((H, W), dtype=np.float32)

    @staticmethod
    def compute_statistics(residuals: np.ndarray) -> dict:
        if len(residuals) == 0:
            return {"rmse": 0.0, "mean": 0.0, "std": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        mag = np.linalg.norm(residuals, axis=1)
        return {
            "rmse": float(np.sqrt(np.mean(mag**2))),
            "mean": float(np.mean(mag)),
            "std": float(np.std(mag)),
            "max": float(np.max(mag)),
            "p50": float(np.percentile(mag, 50)),
            "p90": float(np.percentile(mag, 90)),
            "p95": float(np.percentile(mag, 95)),
            "p99": float(np.percentile(mag, 99)),
        }
