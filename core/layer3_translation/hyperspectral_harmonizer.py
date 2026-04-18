"""Hyperspectral harmoniser — maps between arbitrary band configurations."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Known central wavelengths (nm) for common band designators
_WAVELENGTHS: dict[str, float] = {
    # Sentinel-2
    "B01": 443.0, "B02": 490.0, "B03": 560.0, "B04": 665.0,
    "B05": 705.0, "B06": 740.0, "B07": 783.0, "B08": 842.0,
    "B08A": 865.0, "B09": 940.0, "B10": 1375.0, "B11": 1610.0, "B12": 2190.0,
    # Landsat 8/9
    "SR_B1": 443.0, "SR_B2": 482.0, "SR_B3": 562.0, "SR_B4": 655.0,
    "SR_B5": 865.0, "SR_B6": 1610.0, "SR_B7": 2200.0,
    # Generic RGB / NIR
    "blue": 470.0, "green": 555.0, "red": 665.0, "nir": 842.0,
    "swir1": 1610.0, "swir2": 2190.0,
    "r": 665.0, "g": 555.0, "b": 470.0,
    # VV/VH SAR — assign placeholder wavelengths
    "VV": 56000.0, "VH": 56000.0,
}


def _band_to_wavelength(name: str) -> Optional[float]:
    """Return central wavelength in nm for a band name, or None."""
    return _WAVELENGTHS.get(name) or _WAVELENGTHS.get(name.upper()) or _WAVELENGTHS.get(name.lower())


def _linear_interpolate(
    src_arr: np.ndarray,
    src_wavelengths: list[float],
    tgt_wavelengths: list[float],
) -> np.ndarray:
    """Spectral linear interpolation from source to target wavelengths.

    Parameters
    ----------
    src_arr:
        (C_src, H, W) float32 source hypercube.
    src_wavelengths:
        Central wavelengths for each source band.
    tgt_wavelengths:
        Central wavelengths for each output band.

    Returns
    -------
    np.ndarray
        (C_tgt, H, W) float32 harmonised array.
    """
    C_src, H, W = src_arr.shape
    C_tgt = len(tgt_wavelengths)

    src_wl = np.array(src_wavelengths, dtype=np.float64)
    tgt_wl = np.array(tgt_wavelengths, dtype=np.float64)

    # Reshape to (C_src, H*W) for vectorised interp
    flat = src_arr.reshape(C_src, -1).T  # (H*W, C_src)
    out = np.zeros((H * W, C_tgt), dtype=np.float32)

    for j in range(C_tgt):
        wl = tgt_wl[j]
        # Nearest-neighbour if out of range
        if wl <= src_wl[0]:
            out[:, j] = flat[:, 0]
        elif wl >= src_wl[-1]:
            out[:, j] = flat[:, -1]
        else:
            # Linear interpolation between bracketing bands
            idx_hi = int(np.searchsorted(src_wl, wl))
            idx_lo = idx_hi - 1
            t = (wl - src_wl[idx_lo]) / (src_wl[idx_hi] - src_wl[idx_lo])
            out[:, j] = (1.0 - t) * flat[:, idx_lo] + t * flat[:, idx_hi]

    return out.T.reshape(C_tgt, H, W)


class HyperspectralHarmonizer:
    """Maps a hyperspectral/multispectral image from one band configuration
    to another.

    Uses DOFA embeddings for semantically-guided harmonisation when a GPU
    and model weights are available.  Falls back to linear spectral
    interpolation.

    Parameters
    ----------
    device:
        Target device for DOFA encoder.
    use_dofa:
        Whether to attempt DOFA-based harmonisation (default True).
    """

    def __init__(self, device: str = "auto", use_dofa: bool = True) -> None:
        self._device = device
        self._use_dofa = use_dofa
        self._dofa_adapter = None

    # ------------------------------------------------------------------

    def harmonize(
        self,
        image: np.ndarray,
        source_bands: list[str],
        target_bands: list[str],
    ) -> np.ndarray:
        """Harmonise *image* from *source_bands* to *target_bands*.

        Parameters
        ----------
        image:
            Input array ``(C, H, W)`` or ``(H, W, C)`` float32.
        source_bands:
            Band name list for the input (length must equal C).
        target_bands:
            Band names for the desired output.

        Returns
        -------
        np.ndarray
            Harmonised ``(len(target_bands), H, W)`` float32 array.
        """
        # Ensure (C, H, W)
        arr = image.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == len(source_bands):
            arr = arr.transpose(2, 0, 1)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        C, H, W = arr.shape
        if C != len(source_bands):
            raise ValueError(
                f"image has {C} bands but source_bands has {len(source_bands)} entries"
            )

        # Resolve wavelengths
        src_wl = [_band_to_wavelength(b) for b in source_bands]
        tgt_wl = [_band_to_wavelength(b) for b in target_bands]

        missing_src = [b for b, w in zip(source_bands, src_wl) if w is None]
        missing_tgt = [b for b, w in zip(target_bands, tgt_wl) if w is None]

        if missing_src or missing_tgt:
            logger.warning(
                "unknown band wavelengths; using linear fallback",
                unknown_src=missing_src,
                unknown_tgt=missing_tgt,
            )
            return self._fallback_linear(arr, source_bands, target_bands)

        src_wl_f: list[float] = src_wl  # type: ignore[assignment]
        tgt_wl_f: list[float] = tgt_wl  # type: ignore[assignment]

        # Try DOFA-based harmonisation
        if self._use_dofa and _HAS_TORCH:
            try:
                return self._harmonize_dofa(arr, src_wl_f, tgt_wl_f, target_bands)
            except Exception as exc:
                logger.warning("DOFA harmonisation failed; using linear fallback",
                               error=str(exc))

        return self._harmonize_linear(arr, src_wl_f, tgt_wl_f)

    # ------------------------------------------------------------------
    # DOFA-based harmonisation
    # ------------------------------------------------------------------

    def _harmonize_dofa(
        self,
        arr: np.ndarray,
        src_wl: list[float],
        tgt_wl: list[float],
        target_bands: list[str],
    ) -> np.ndarray:
        """Use DOFA encoder to produce harmonised output via embedding decode."""
        if self._dofa_adapter is None:
            from core.layer2_foundation.dofa_adapter import DOFAAdapter  # type: ignore[import]
            self._dofa_adapter = DOFAAdapter(device=self._device)
            self._dofa_adapter.load_model()

        # Encode source image
        embeddings = self._dofa_adapter.encode(arr, wavelengths=src_wl)  # (H, W, D)
        H, W, D = embeddings.shape

        # Decode each target band by projecting embeddings via a linear probe
        C_tgt = len(tgt_wl)
        output = np.zeros((C_tgt, H, W), dtype=np.float32)

        # Train per-band linear probe: embed → target band intensity
        flat_emb = embeddings.reshape(-1, D)  # (H*W, D)

        # First compute linear interpolation as supervision signal
        interp = _linear_interpolate(arr, src_wl, tgt_wl)  # (C_tgt, H, W)

        for i in range(C_tgt):
            y = interp[i].ravel()                # (H*W,)
            # Closed-form least squares: W = (X^T X)^{-1} X^T y
            XtX = flat_emb.T @ flat_emb           # (D, D)
            Xty = flat_emb.T @ y                  # (D,)
            reg = 1e-4 * np.eye(D)
            try:
                w = np.linalg.solve(XtX + reg, Xty)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(flat_emb, y, rcond=None)[0]
            pred = (flat_emb @ w).reshape(H, W)
            # Blend 50/50 with linear interpolation for stability
            output[i] = 0.5 * pred + 0.5 * interp[i]

        logger.debug("DOFA harmonisation complete", C_tgt=C_tgt)
        return output

    # ------------------------------------------------------------------
    # Linear interpolation
    # ------------------------------------------------------------------

    def _harmonize_linear(
        self, arr: np.ndarray, src_wl: list[float], tgt_wl: list[float]
    ) -> np.ndarray:
        # Sort source by wavelength
        order = np.argsort(src_wl)
        arr_sorted = arr[order]
        src_sorted = sorted(src_wl)
        return _linear_interpolate(arr_sorted, src_sorted, tgt_wl)

    def _fallback_linear(
        self, arr: np.ndarray, source_bands: list[str], target_bands: list[str]
    ) -> np.ndarray:
        """Fallback: assign evenly-spaced wavelengths and interpolate."""
        C_src, H, W = arr.shape
        C_tgt = len(target_bands)
        src_wl = list(np.linspace(400.0, 2500.0, C_src).tolist())
        tgt_wl = list(np.linspace(400.0, 2500.0, C_tgt).tolist())
        return _linear_interpolate(arr, src_wl, tgt_wl)
