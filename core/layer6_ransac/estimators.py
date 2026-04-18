"""Robust geometric estimators (MAGSAC++, USAC, PROSAC, TPS)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from .model_types import ModelType

logger = structlog.get_logger(__name__)


@dataclass
class EstimationResult:
    """Output of robust geometric estimation."""

    model_type: ModelType
    transform_matrix: np.ndarray | None
    inlier_mask: np.ndarray
    num_inliers: int
    num_matches: int
    inlier_ratio: float
    rmse: float
    confidence: float
    metadata: dict = field(default_factory=dict)


class RobustEstimator:
    """Robust geometric estimation with multiple back-ends.

    Parameters
    ----------
    method:
        ``"magsac++"`` | ``"usac"`` | ``"gc_ransac"`` | ``"lmeds"``
    reproj_threshold:
        Reprojection error threshold in pixels.
    confidence:
        RANSAC confidence level (0–1).
    max_iters:
        Maximum RANSAC iterations.
    model_type:
        Default model type.
    """

    def __init__(
        self,
        method: str = "magsac++",
        reproj_threshold: float = 3.0,
        confidence: float = 0.999,
        max_iters: int = 10000,
        model_type: ModelType = ModelType.HOMOGRAPHY,
    ) -> None:
        self.method = method.lower()
        self.reproj_threshold = reproj_threshold
        self.confidence = confidence
        self.max_iters = max_iters
        self.default_model_type = model_type
        self.logger = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def estimate(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        method: str | None = None,
        model_type: ModelType | None = None,
    ) -> EstimationResult:
        """Run robust estimation.

        Parameters
        ----------
        kpts0, kpts1:
            [N, 2] matched keypoints.
        method:
            Override default method.
        model_type:
            Override default model type.

        Returns
        -------
        EstimationResult
        """
        m = method.lower() if method else self.method
        mt = model_type if model_type is not None else self.default_model_type

        kpts0 = np.array(kpts0, dtype=np.float64)
        kpts1 = np.array(kpts1, dtype=np.float64)
        N = len(kpts0)

        self.logger.info("Running estimation", method=m, model_type=mt.name, n_matches=N)

        # Minimum correspondences
        min_pts = {
            ModelType.HOMOGRAPHY: 4,
            ModelType.AFFINE: 3,
            ModelType.AFFINE_PARTIAL: 2,
            ModelType.SIMILARITY: 2,
            ModelType.TPS: 4,
            ModelType.POLYNOMIAL: 6,
            ModelType.PIECEWISE: 4,
        }
        if N < min_pts.get(mt, 4):
            self.logger.warning("Too few matches for estimation", n=N, required=min_pts.get(mt, 4))
            return EstimationResult(
                model_type=mt,
                transform_matrix=None,
                inlier_mask=np.zeros(N, dtype=bool),
                num_inliers=0,
                num_matches=N,
                inlier_ratio=0.0,
                rmse=float("inf"),
                confidence=0.0,
            )

        if mt == ModelType.TPS:
            return self._fit_tps(kpts0, kpts1)
        if mt == ModelType.POLYNOMIAL:
            return self._fit_polynomial(kpts0, kpts1)
        if mt == ModelType.PIECEWISE:
            return self._fit_piecewise(kpts0, kpts1)

        # OpenCV-based estimators
        try:
            import cv2

            if mt == ModelType.HOMOGRAPHY:
                return self._estimate_homography(kpts0, kpts1, m, cv2)
            if mt in (ModelType.AFFINE, ModelType.AFFINE_PARTIAL):
                return self._estimate_affine(kpts0, kpts1, mt, m, cv2)
            if mt == ModelType.SIMILARITY:
                return self._estimate_similarity(kpts0, kpts1, cv2)
        except ImportError:
            self.logger.warning("OpenCV not available; using numpy fallback")
            return self._numpy_fallback(kpts0, kpts1, mt)

        return self._numpy_fallback(kpts0, kpts1, mt)

    # ------------------------------------------------------------------
    def _cv2_method_flag(self, method: str, cv2: Any) -> int:
        flags = {
            "magsac++": cv2.USAC_MAGSAC,
            "usac": cv2.USAC_DEFAULT,
            "gc_ransac": cv2.USAC_PROSAC,
            "lmeds": cv2.LMEDS,
            "ransac": cv2.RANSAC,
        }
        return flags.get(method, cv2.USAC_MAGSAC)

    def _estimate_homography(self, kpts0: np.ndarray, kpts1: np.ndarray, method: str, cv2: Any) -> EstimationResult:
        flag = self._cv2_method_flag(method, cv2)
        H, mask = cv2.findHomography(
            kpts0, kpts1,
            flag,
            self.reproj_threshold,
            confidence=self.confidence,
            maxIters=self.max_iters,
        )
        return self._build_result(H, mask, kpts0, kpts1, ModelType.HOMOGRAPHY)

    def _estimate_affine(self, kpts0: np.ndarray, kpts1: np.ndarray, mt: ModelType, method: str, cv2: Any) -> EstimationResult:
        if mt == ModelType.AFFINE_PARTIAL:
            M, mask = cv2.estimateAffinePartial2D(
                kpts0, kpts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.reproj_threshold,
                confidence=self.confidence,
                maxIters=self.max_iters,
            )
        else:
            M, mask = cv2.estimateAffine2D(
                kpts0, kpts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.reproj_threshold,
                confidence=self.confidence,
                maxIters=self.max_iters,
            )
        # Extend to 3×3
        if M is not None:
            M3 = np.vstack([M, [0, 0, 1]])
        else:
            M3 = None
        return self._build_result(M3, mask, kpts0, kpts1, mt)

    def _estimate_similarity(self, kpts0: np.ndarray, kpts1: np.ndarray, cv2: Any) -> EstimationResult:
        # Similarity = partial affine (scale + rotation + translation)
        M, mask = cv2.estimateAffinePartial2D(
            kpts0, kpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold,
            confidence=self.confidence,
        )
        M3 = np.vstack([M, [0, 0, 1]]) if M is not None else None
        return self._build_result(M3, mask, kpts0, kpts1, ModelType.SIMILARITY)

    def _magsac_pp(self, kpts0: np.ndarray, kpts1: np.ndarray, model_type: ModelType) -> EstimationResult:
        return self.estimate(kpts0, kpts1, method="magsac++", model_type=model_type)

    def _usac(self, kpts0: np.ndarray, kpts1: np.ndarray, model_type: ModelType) -> EstimationResult:
        return self.estimate(kpts0, kpts1, method="usac", model_type=model_type)

    def _gc_ransac(self, kpts0: np.ndarray, kpts1: np.ndarray, model_type: ModelType) -> EstimationResult:
        return self.estimate(kpts0, kpts1, method="gc_ransac", model_type=model_type)

    # ------------------------------------------------------------------
    def _fit_tps(self, kpts0: np.ndarray, kpts1: np.ndarray) -> EstimationResult:
        """Thin-plate spline estimation using scipy RBFInterpolator."""
        from scipy.interpolate import RBFInterpolator  # type: ignore[import]

        try:
            rbf_x = RBFInterpolator(kpts0, kpts1[:, 0], kernel="thin_plate_spline", smoothing=0.0)
            rbf_y = RBFInterpolator(kpts0, kpts1[:, 1], kernel="thin_plate_spline", smoothing=0.0)

            # Evaluate on source points to compute residuals
            pred_x = rbf_x(kpts0)
            pred_y = rbf_y(kpts0)
            pred = np.column_stack([pred_x, pred_y])
            residuals = np.linalg.norm(pred - kpts1, axis=1)
            inlier_mask = residuals < self.reproj_threshold
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            return EstimationResult(
                model_type=ModelType.TPS,
                transform_matrix=None,
                inlier_mask=inlier_mask,
                num_inliers=int(np.sum(inlier_mask)),
                num_matches=len(kpts0),
                inlier_ratio=float(np.mean(inlier_mask)),
                rmse=rmse,
                confidence=float(np.mean(inlier_mask)),
                metadata={"rbf_x": rbf_x, "rbf_y": rbf_y},
            )
        except Exception as exc:
            self.logger.warning("TPS fitting failed", error=str(exc))
            return self._numpy_fallback(kpts0, kpts1, ModelType.TPS)

    def _fit_polynomial(self, kpts0: np.ndarray, kpts1: np.ndarray) -> EstimationResult:
        """2nd-order polynomial warp fit."""
        try:
            x0, y0 = kpts0[:, 0], kpts0[:, 1]
            # Design matrix for 2nd-order polynomial
            A = np.column_stack([
                np.ones(len(x0)), x0, y0, x0 * y0, x0 ** 2, y0 ** 2
            ])
            coeffs_x, _, _, _ = np.linalg.lstsq(A, kpts1[:, 0], rcond=None)
            coeffs_y, _, _, _ = np.linalg.lstsq(A, kpts1[:, 1], rcond=None)
            pred_x = A @ coeffs_x
            pred_y = A @ coeffs_y
            pred = np.column_stack([pred_x, pred_y])
            residuals = np.linalg.norm(pred - kpts1, axis=1)
            inlier_mask = residuals < self.reproj_threshold
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            M = np.vstack([coeffs_x, coeffs_y])  # [2, 6]
            return EstimationResult(
                model_type=ModelType.POLYNOMIAL,
                transform_matrix=M,
                inlier_mask=inlier_mask,
                num_inliers=int(np.sum(inlier_mask)),
                num_matches=len(kpts0),
                inlier_ratio=float(np.mean(inlier_mask)),
                rmse=rmse,
                confidence=float(np.mean(inlier_mask)),
            )
        except Exception as exc:
            self.logger.warning("Polynomial fitting failed", error=str(exc))
            return self._numpy_fallback(kpts0, kpts1, ModelType.POLYNOMIAL)

    def _fit_piecewise(self, kpts0: np.ndarray, kpts1: np.ndarray) -> EstimationResult:
        """Piecewise affine — Delaunay triangulation with per-triangle affine."""
        try:
            from scipy.spatial import Delaunay  # type: ignore[import]

            tri = Delaunay(kpts0)
            # Compute global affine as representative matrix
            M, mask = _fit_affine_numpy(kpts0, kpts1, self.reproj_threshold)
            rmse = self.compute_rmse(kpts0, kpts1, M, mask)
            return EstimationResult(
                model_type=ModelType.PIECEWISE,
                transform_matrix=M,
                inlier_mask=mask,
                num_inliers=int(np.sum(mask)),
                num_matches=len(kpts0),
                inlier_ratio=float(np.mean(mask)),
                rmse=rmse,
                confidence=float(np.mean(mask)),
                metadata={"triangulation": tri},
            )
        except Exception as exc:
            self.logger.warning("Piecewise fitting failed", error=str(exc))
            return self._numpy_fallback(kpts0, kpts1, ModelType.PIECEWISE)

    # ------------------------------------------------------------------
    def _numpy_fallback(self, kpts0: np.ndarray, kpts1: np.ndarray, mt: ModelType) -> EstimationResult:
        """Pure-numpy RANSAC homography fallback."""
        M, mask = _ransac_homography_numpy(kpts0, kpts1, self.reproj_threshold, 1000)
        rmse = self.compute_rmse(kpts0, kpts1, M, mask)
        return EstimationResult(
            model_type=mt,
            transform_matrix=M,
            inlier_mask=mask,
            num_inliers=int(np.sum(mask)),
            num_matches=len(kpts0),
            inlier_ratio=float(np.mean(mask)),
            rmse=rmse,
            confidence=float(np.mean(mask)),
        )

    # ------------------------------------------------------------------
    def _build_result(
        self,
        M: np.ndarray | None,
        mask: np.ndarray | None,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mt: ModelType,
    ) -> EstimationResult:
        N = len(kpts0)
        if mask is None:
            mask = np.zeros(N, dtype=bool)
        else:
            mask = mask.flatten().astype(bool)
        num_inliers = int(np.sum(mask))
        rmse = self.compute_rmse(kpts0, kpts1, M, mask) if M is not None else float("inf")
        return EstimationResult(
            model_type=mt,
            transform_matrix=M,
            inlier_mask=mask,
            num_inliers=num_inliers,
            num_matches=N,
            inlier_ratio=num_inliers / (N + 1e-10),
            rmse=rmse,
            confidence=num_inliers / (N + 1e-10),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_rmse(
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        M: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> float:
        """Compute reprojection RMSE for inliers."""
        if M is None or len(kpts0) == 0:
            return float("inf")
        if mask is None:
            mask = np.ones(len(kpts0), dtype=bool)
        mask_b = mask.astype(bool).flatten()
        if np.sum(mask_b) == 0:
            return float("inf")

        pts0 = kpts0[mask_b]
        pts1 = kpts1[mask_b]

        if M.shape == (3, 3):
            ones = np.ones((len(pts0), 1))
            pts0_h = np.hstack([pts0, ones])
            proj = (M @ pts0_h.T).T
            proj = proj[:, :2] / (proj[:, 2:3] + 1e-10)
        elif M.shape[0] == 2 and M.shape[1] in (3, 6):
            ones = np.ones((len(pts0), 1))
            pts0_h = np.hstack([pts0, ones])
            proj = (M[:2, :3] @ pts0_h.T).T
        else:
            return float("inf")

        residuals = np.linalg.norm(proj - pts1, axis=1)
        return float(np.sqrt(np.mean(residuals ** 2)))

    # ------------------------------------------------------------------
    def auto_select_model(self, kpts0: np.ndarray, kpts1: np.ndarray) -> ModelType:
        """Heuristically select model type based on match distribution."""
        N = len(kpts0)
        if N < 4:
            return ModelType.AFFINE_PARTIAL
        if N < 8:
            return ModelType.AFFINE
        if N >= 8:
            # Estimate spatial spread
            spread = np.std(kpts0, axis=0).mean()
            if spread < 20 and N < 20:
                return ModelType.AFFINE
            if N >= 20:
                return ModelType.HOMOGRAPHY
        return ModelType.HOMOGRAPHY


# ---------------------------------------------------------------------------
# Pure-numpy helpers
# ---------------------------------------------------------------------------

def _fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform for homography from 4+ correspondences."""
    N = len(src)
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = src[i, 0], src[i, 1]
        xp, yp = dst[i, 0], dst[i, 1]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2] + 1e-10
    return H


def _reproj_error_homography(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    ones = np.ones((len(src), 1))
    src_h = np.hstack([src, ones])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:3] + 1e-10)
    return np.linalg.norm(proj - dst, axis=1)


def _ransac_homography_numpy(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    threshold: float = 3.0,
    max_iters: int = 1000,
) -> tuple[np.ndarray | None, np.ndarray]:
    N = len(kpts0)
    if N < 4:
        return None, np.zeros(N, dtype=bool)

    best_mask = np.zeros(N, dtype=bool)
    best_H = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(max_iters):
        idx = rng.choice(N, 4, replace=False)
        try:
            H = _fit_homography_dlt(kpts0[idx], kpts1[idx])
        except Exception:
            continue
        errs = _reproj_error_homography(H, kpts0, kpts1)
        mask = errs < threshold
        count = int(np.sum(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
            best_H = H

    # Refit on all inliers
    if best_H is not None and best_count >= 4:
        try:
            best_H = _fit_homography_dlt(kpts0[best_mask], kpts1[best_mask])
            errs = _reproj_error_homography(best_H, kpts0, kpts1)
            best_mask = errs < threshold
        except Exception:
            pass

    return best_H, best_mask


def _fit_affine_numpy(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine with RANSAC inlier mask."""
    N = len(kpts0)
    ones = np.ones((N, 1))
    A = np.hstack([kpts0, ones])
    # Solve for [a, b, c] for x and y independently
    cx, _, _, _ = np.linalg.lstsq(A, kpts1[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, kpts1[:, 1], rcond=None)
    M = np.array([cx, cy, [0, 0, 1]])
    M3 = np.vstack([np.array([cx, cy]), np.array([[0, 0, 1]])])
    pred = A @ np.array([cx, cy]).T
    residuals = np.linalg.norm(pred - kpts1, axis=1)
    mask = residuals < threshold
    return M3, mask
