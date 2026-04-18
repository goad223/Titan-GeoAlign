"""Image warp / transform application functions."""
from __future__ import annotations

import numpy as np
import structlog

from .model_types import ModelType
from .estimators import EstimationResult

logger = structlog.get_logger(__name__)


def apply_homography(
    image: np.ndarray,
    H: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp *image* using homography *H*.

    Parameters
    ----------
    image:
        Source image (HxW or HxWxC).
    H:
        3×3 homography matrix (maps src → dst).
    output_shape:
        ``(height, width)`` of the output image.

    Returns
    -------
    np.ndarray
        Warped image, same dtype as input.
    """
    import cv2  # type: ignore[import]

    out_h, out_w = output_shape
    return cv2.warpPerspective(image, H.astype(np.float64), (out_w, out_h))


def apply_affine(
    image: np.ndarray,
    A: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp *image* using affine matrix *A*.

    Parameters
    ----------
    A:
        Either a 2×3 or 3×3 affine matrix.
    """
    import cv2  # type: ignore[import]

    out_h, out_w = output_shape
    if A.shape == (3, 3):
        M = A[:2, :]
    elif A.shape == (2, 3):
        M = A
    else:
        raise ValueError(f"Unexpected affine matrix shape {A.shape}")
    return cv2.warpAffine(image, M.astype(np.float64), (out_w, out_h))


def apply_tps(
    image: np.ndarray,
    control_pts_src: np.ndarray,
    control_pts_dst: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp *image* using a thin-plate spline defined by control points.

    Uses ``scipy.interpolate.RBFInterpolator`` with ``thin_plate_spline``
    kernel to build a pixel-level warp field.

    Parameters
    ----------
    control_pts_src:
        [N, 2] source control points (x, y).
    control_pts_dst:
        [N, 2] destination control points (x, y).
    output_shape:
        ``(height, width)`` of the output.
    """
    from scipy.interpolate import RBFInterpolator  # type: ignore[import]
    from scipy.ndimage import map_coordinates  # type: ignore[import]

    out_h, out_w = output_shape

    # Build forward-mapped TPS: dst → src (for inverse warping)
    rbf_x = RBFInterpolator(control_pts_dst, control_pts_src[:, 0], kernel="thin_plate_spline", smoothing=0.0)
    rbf_y = RBFInterpolator(control_pts_dst, control_pts_src[:, 1], kernel="thin_plate_spline", smoothing=0.0)

    # Build output grid (x, y)
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    coords_dst = np.column_stack([xs.flatten().astype(np.float64), ys.flatten().astype(np.float64)])

    src_x = rbf_x(coords_dst).reshape(out_h, out_w)
    src_y = rbf_y(coords_dst).reshape(out_h, out_w)

    if image.ndim == 2:
        warped = map_coordinates(image.astype(np.float64), [src_y, src_x], order=1, mode="constant", cval=0.0)
    else:
        channels = []
        for c in range(image.shape[2]):
            ch = map_coordinates(image[:, :, c].astype(np.float64), [src_y, src_x], order=1, mode="constant", cval=0.0)
            channels.append(ch)
        warped = np.stack(channels, axis=-1)

    # Cast back to original dtype
    if image.dtype == np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        warped = np.clip(warped, 0, 65535).astype(np.uint16)
    else:
        warped = warped.astype(image.dtype)

    return warped


def warp_image(
    image: np.ndarray,
    result: EstimationResult,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply the transform in *result* to *image*.

    Parameters
    ----------
    image:
        Source image.
    result:
        ``EstimationResult`` from the robust estimator.
    output_shape:
        ``(height, width)`` of the output. Defaults to same shape as input.

    Returns
    -------
    np.ndarray
        Warped image.
    """
    if output_shape is None:
        output_shape = image.shape[:2]

    mt = result.model_type
    M = result.transform_matrix

    if M is None:
        logger.warning("No transform matrix in EstimationResult; returning original image")
        return image.copy()

    if mt == ModelType.TPS:
        rbf_x = result.metadata.get("rbf_x")
        rbf_y = result.metadata.get("rbf_y")
        if rbf_x is None or rbf_y is None:
            logger.warning("TPS metadata missing; falling back to identity warp")
            return image.copy()
        # Reconstruct control points from metadata if available
        logger.warning("TPS warp via warp_image requires control_pts; using apply_tps wrapper is preferred")
        return image.copy()

    if mt == ModelType.HOMOGRAPHY:
        return apply_homography(image, M, output_shape)

    if mt in (ModelType.AFFINE, ModelType.AFFINE_PARTIAL, ModelType.SIMILARITY, ModelType.PIECEWISE):
        return apply_affine(image, M, output_shape)

    if mt == ModelType.POLYNOMIAL:
        # Polynomial warp via grid evaluation
        return _apply_polynomial(image, M, output_shape)

    logger.warning("Unhandled model type for warp_image", model_type=mt.name)
    return image.copy()


def _apply_polynomial(
    image: np.ndarray,
    coeffs: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Apply a 2nd-order polynomial warp using the coefficient matrix [2, 6]."""
    from scipy.ndimage import map_coordinates  # type: ignore[import]

    out_h, out_w = output_shape
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    x_f = xs.flatten().astype(np.float64)
    y_f = ys.flatten().astype(np.float64)

    # Design matrix for inverse (approximate: use same polynomial, maps dst→src)
    A = np.column_stack([np.ones_like(x_f), x_f, y_f, x_f * y_f, x_f ** 2, y_f ** 2])
    src_x = A @ coeffs[0]
    src_y = A @ coeffs[1]
    src_x = src_x.reshape(out_h, out_w)
    src_y = src_y.reshape(out_h, out_w)

    if image.ndim == 2:
        warped = map_coordinates(image.astype(np.float64), [src_y, src_x], order=1, mode="constant", cval=0.0)
    else:
        channels = [
            map_coordinates(image[:, :, c].astype(np.float64), [src_y, src_x], order=1, mode="constant", cval=0.0)
            for c in range(image.shape[2])
        ]
        warped = np.stack(channels, axis=-1)

    if image.dtype == np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        warped = np.clip(warped, 0, 65535).astype(np.uint16)
    else:
        warped = warped.astype(image.dtype)
    return warped
