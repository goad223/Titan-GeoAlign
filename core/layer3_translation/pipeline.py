"""TranslationPipeline — routes cross-modality translation between ImageData pairs."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .sar2optical import SAR2OpticalDiffusion
from .cyclegan_hd import CycleGANHD
from .hyperspectral_harmonizer import HyperspectralHarmonizer

try:
    from core.layer0_ingestion.image_data import ImageData
    from core.layer0_ingestion.sensor_types import SensorType
    from core.layer0_ingestion.image_data import _array_to_dataset
except ImportError:
    from ..layer0_ingestion.image_data import ImageData, _array_to_dataset  # type: ignore[no-redef]
    from ..layer0_ingestion.sensor_types import SensorType  # type: ignore[no-redef]

try:
    import xarray as xr
    _HAS_XR = True
except ImportError:
    _HAS_XR = False

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Modality groupings
# ---------------------------------------------------------------------------
_SAR_SENSORS = {SensorType.SENTINEL1_SAR}
_HYPERSPECTRAL_SENSORS = {SensorType.ENMAP, SensorType.PRISMA}
_OPTICAL_SENSORS = {
    SensorType.SENTINEL2, SensorType.LANDSAT8, SensorType.LANDSAT9,
    SensorType.PLANETSCOPE, SensorType.WORLDVIEW, SensorType.SKYSAT,
    SensorType.UAV_RGB, SensorType.UAV_MULTISPECTRAL, SensorType.MODIS,
}


def _is_sar(img: ImageData) -> bool:
    return img.sensor_type in _SAR_SENSORS


def _is_optical(img: ImageData) -> bool:
    return img.sensor_type in _OPTICAL_SENSORS


def _is_hyperspectral(img: ImageData) -> bool:
    return img.sensor_type in _HYPERSPECTRAL_SENSORS or len(img.band_names) > 20


def _is_thermal(img: ImageData) -> bool:
    """Heuristic: Landsat thermal band present."""
    return any(b.startswith("ST_B") for b in img.band_names)


class TranslationPipeline:
    """Detects modality mismatch and applies the appropriate translation.

    Parameters
    ----------
    sar2optical_model_path:
        Optional path to diffusion model for SAR→optical translation.
    cyclegan_checkpoint_dir:
        Optional directory with CycleGAN checkpoints.
    device:
        Target device for all translation models.
    """

    def __init__(
        self,
        sar2optical_model_path: Optional[Path] = None,
        cyclegan_checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
    ) -> None:
        self._device = device
        self._sar2optical = SAR2OpticalDiffusion(
            model_path=sar2optical_model_path, device=device
        )
        self._cyclegan = CycleGANHD(
            checkpoint_dir=cyclegan_checkpoint_dir, device=device
        )
        self._harmonizer = HyperspectralHarmonizer(device=device)

        if sar2optical_model_path and Path(sar2optical_model_path).exists():
            self._sar2optical.load_model()

    # ------------------------------------------------------------------

    def route(
        self, source: ImageData, target: ImageData
    ) -> tuple[ImageData, ImageData]:
        """Route source and target through the appropriate translator.

        Returns ``(translated_source, target)`` where *translated_source*
        has been converted to match *target*'s modality.  If no translation
        is required, the original source is returned unchanged.

        Parameters
        ----------
        source:
            Source image (may be SAR, hyperspectral, thermal, etc.).
        target:
            Target/reference image (typically optical multispectral).

        Returns
        -------
        tuple[ImageData, ImageData]
            ``(translated_source, target)`` pair.
        """
        src_type = source.sensor_type
        tgt_type = target.sensor_type

        # No translation needed if same modality
        if src_type == tgt_type:
            logger.debug("same modality; no translation needed", sensor=src_type)
            return source, target

        # SAR → Optical
        if _is_sar(source) and _is_optical(target):
            logger.info("translating SAR → Optical")
            translated = self._translate_sar_to_optical(source, target)
            return translated, target

        # Thermal → Optical (Landsat ST bands)
        if _is_thermal(source) and _is_optical(target):
            logger.info("translating Thermal → Optical")
            translated = self._translate_thermal_to_optical(source, target)
            return translated, target

        # Hyperspectral → Multispectral
        if _is_hyperspectral(source) and not _is_hyperspectral(target):
            logger.info("translating Hyperspectral → Multispectral")
            translated = self._translate_hyperspectral(source, target)
            return translated, target

        # Optical with different band counts → harmonise
        if _is_optical(source) and _is_optical(target):
            if set(source.band_names) != set(target.band_names):
                logger.info("harmonising optical bands")
                translated = self._harmonise_optical_bands(source, target)
                return translated, target

        logger.debug("no matching translation strategy; returning unchanged",
                     source_sensor=src_type, target_sensor=tgt_type)
        return source, target

    # ------------------------------------------------------------------
    # Translation strategies
    # ------------------------------------------------------------------

    def _translate_sar_to_optical(
        self, source: ImageData, target: ImageData
    ) -> ImageData:
        src_arr = source.to_numpy()   # (C, H, W)
        pseudo_rgb = self._sar2optical.translate(src_arr)  # (H, W, 3) uint8
        return self._wrap_as_imagedata(
            pseudo_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
            band_names=["pseudo_R", "pseudo_G", "pseudo_B"],
            template=source,
            sensor_type=SensorType.SENTINEL2,
        )

    def _translate_thermal_to_optical(
        self, source: ImageData, target: ImageData
    ) -> ImageData:
        src_arr = source.to_numpy()
        pseudo_rgb = self._cyclegan.translate(
            src_arr.transpose(1, 2, 0), "thermal", "day"
        )  # (H, W, 3) uint8
        return self._wrap_as_imagedata(
            pseudo_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
            band_names=["pseudo_R", "pseudo_G", "pseudo_B"],
            template=source,
            sensor_type=SensorType.LANDSAT8,
        )

    def _translate_hyperspectral(
        self, source: ImageData, target: ImageData
    ) -> ImageData:
        src_arr = source.to_numpy()   # (C_src, H, W)
        harmonised = self._harmonizer.harmonize(
            src_arr, source.band_names, target.band_names
        )  # (C_tgt, H, W)

        # Resample to target spatial size if different
        tgt_arr = target.to_numpy()  # (C_tgt, H_tgt, W_tgt)
        if harmonised.shape[1:] != tgt_arr.shape[1:]:
            harmonised = self._spatial_resample(
                harmonised, tgt_arr.shape[1], tgt_arr.shape[2]
            )

        return self._wrap_as_imagedata(
            harmonised,
            band_names=target.band_names,
            template=source,
            sensor_type=target.sensor_type,
        )

    def _harmonise_optical_bands(
        self, source: ImageData, target: ImageData
    ) -> ImageData:
        src_arr = source.to_numpy()
        harmonised = self._harmonizer.harmonize(
            src_arr, source.band_names, target.band_names
        )
        return self._wrap_as_imagedata(
            harmonised,
            band_names=target.band_names,
            template=source,
            sensor_type=target.sensor_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_resample(arr: np.ndarray, new_H: int, new_W: int) -> np.ndarray:
        """Bilinear resample (C, H, W) to (C, new_H, new_W)."""
        try:
            import torch
            import torch.nn.functional as F
            t = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
            out = F.interpolate(t, size=(new_H, new_W), mode="bilinear", align_corners=False)
            return out.squeeze(0).numpy()
        except ImportError:
            pass

        # Fallback: per-channel scipy zoom
        from scipy.ndimage import zoom  # type: ignore[import]
        C, H, W = arr.shape
        scale_H = new_H / H
        scale_W = new_W / W
        out = np.empty((C, new_H, new_W), dtype=arr.dtype)
        for i in range(C):
            out[i] = zoom(arr[i], (scale_H, scale_W), order=1)
        return out

    @staticmethod
    def _wrap_as_imagedata(
        arr: np.ndarray,
        band_names: list[str],
        template: ImageData,
        sensor_type: Optional[SensorType] = None,
    ) -> ImageData:
        """Create a new ImageData from array using template's spatial metadata."""
        result = copy.copy(template)
        result.data = _array_to_dataset(arr, band_names, template.data)
        result.band_names = band_names
        if sensor_type is not None:
            result.sensor_type = sensor_type
        result.bit_depth = 32
        result.nodata = None
        return result
