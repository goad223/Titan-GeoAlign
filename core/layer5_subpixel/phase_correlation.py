"""Foroosh sub-pixel phase correlation refiner."""
from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PhaseCorrelationRefiner:
    """Sub-pixel refinement using Foroosh phase correlation.

    For each matched keypoint pair, extracts a small patch around each
    keypoint, computes the cross-power spectrum (phase correlation), and
    applies the Foroosh correction to obtain a sub-pixel displacement.
    """

    def __init__(self, patch_size: int = 32) -> None:
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    def refine(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        patch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine keypoint positions to sub-pixel accuracy.

        Parameters
        ----------
        img0, img1:
            Source images (HxW or HxWxC, any numeric dtype).
        kpts0, kpts1:
            [N, 2] arrays of (x, y) pixel coordinates.
        patch_size:
            Override instance patch_size if provided.

        Returns
        -------
        refined_kpts0, refined_kpts1 : np.ndarray [N, 2]
            Refined keypoint positions.
        """
        ps = patch_size if patch_size is not None else self.patch_size
        half = ps // 2

        kpts0 = np.array(kpts0, dtype=np.float32)
        kpts1 = np.array(kpts1, dtype=np.float32)

        g0 = self._to_gray(img0)
        g1 = self._to_gray(img1)
        H0, W0 = g0.shape
        H1, W1 = g1.shape

        refined0 = kpts0.copy()
        refined1 = kpts1.copy()

        for i in range(len(kpts0)):
            cx0, cy0 = kpts0[i, 0], kpts0[i, 1]
            cx1, cy1 = kpts1[i, 0], kpts1[i, 1]

            patch0 = self._extract_patch(g0, cx0, cy0, half, H0, W0)
            patch1 = self._extract_patch(g1, cx1, cy1, half, H1, W1)

            if patch0 is None or patch1 is None:
                continue

            dx, dy = self._foroosh_shift(patch0, patch1)
            # dx, dy represent the refinement of kpts1 relative to kpts0
            refined1[i, 0] = cx1 + dx
            refined1[i, 1] = cy1 + dy

        return refined0, refined1

    # ------------------------------------------------------------------
    @staticmethod
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

    @staticmethod
    def _extract_patch(
        img: np.ndarray,
        cx: float,
        cy: float,
        half: int,
        H: int,
        W: int,
    ) -> np.ndarray | None:
        ix, iy = int(round(cx)), int(round(cy))
        r1, r2 = iy - half, iy + half
        c1, c2 = ix - half, ix + half
        if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
            return None
        return img[r1:r2, c1:c2].copy()

    @staticmethod
    def _foroosh_shift(patch0: np.ndarray, patch1: np.ndarray) -> tuple[float, float]:
        """Compute sub-pixel shift between two patches using Foroosh correction.

        Foroosh et al. (2002) — extension of phase correlation to sub-pixel.
        """
        if patch0.shape != patch1.shape:
            return 0.0, 0.0

        # Apodisation window to reduce spectral leakage
        rows, cols = patch0.shape
        win_r = np.hanning(rows)[:, np.newaxis]
        win_c = np.hanning(cols)[np.newaxis, :]
        window = win_r * win_c

        F0 = np.fft.fft2(patch0 * window)
        F1 = np.fft.fft2(patch1 * window)

        cross_power = F0 * np.conj(F1)
        denom = np.abs(cross_power) + 1e-10
        cross_power_norm = cross_power / denom

        correlation = np.fft.ifft2(cross_power_norm).real

        # Find integer peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        pr, pc = peak_idx

        # Map to signed shift
        if pr > rows // 2:
            pr -= rows
        if pc > cols // 2:
            pc -= cols

        # Foroosh sub-pixel correction
        def _foroosh_1d(corr_1d: np.ndarray, shift_int: int, n: int) -> float:
            """1-D Foroosh sub-pixel correction."""
            idx = shift_int % n
            idx_p1 = (shift_int + 1) % n
            idx_m1 = (shift_int - 1) % n
            c0 = corr_1d[idx]
            c1 = corr_1d[idx_p1]
            cm1 = corr_1d[idx_m1]
            # Foroosh formula: δ = c1 / (c1 - c0) if c1 > cm1 else -cm1/(cm1 - c0)
            if abs(c1) > abs(cm1):
                delta = c1 / (c1 - c0 + 1e-10)
            else:
                delta = -cm1 / (cm1 - c0 + 1e-10)
            return float(np.clip(delta, -0.5, 0.5))

        corr_row = correlation[:, peak_idx[1] % cols]
        corr_col = correlation[peak_idx[0] % rows, :]
        sub_r = _foroosh_1d(corr_row, pr, rows)
        sub_c = _foroosh_1d(corr_col, pc, cols)

        # (dx, dy) → (col shift, row shift)
        dx = float(pc) + sub_c
        dy = float(pr) + sub_r
        return dx, dy
