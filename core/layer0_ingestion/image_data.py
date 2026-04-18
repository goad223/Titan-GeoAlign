"""ImageData dataclass — central in-memory representation for all sensor data."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None  # type: ignore[assignment]

try:
    from affine import Affine
except ImportError:
    Affine = None  # type: ignore[assignment]

import structlog

from .sensor_types import SensorType

logger = structlog.get_logger(__name__)


@dataclass
class ImageData:
    """Central in-memory representation for all sensor imagery.

    Bands are stored as variables in an xarray.Dataset for lazy/chunked access.
    """

    data: "xr.Dataset"
    sensor_type: SensorType
    crs: str
    transform: "Affine"
    bounds: tuple[float, float, float, float]  # west, south, east, north
    resolution_m: float
    bit_depth: int
    band_names: list[str]
    metadata: dict = field(default_factory=dict)
    nodata: Optional[float] = None
    filepath: Optional[Path] = None

    # ------------------------------------------------------------------
    # numpy helpers
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return all bands as a (bands, height, width) float32 ndarray."""
        arrays = [
            self.data[b].values.astype(np.float32) for b in self.band_names
        ]
        return np.stack(arrays, axis=0)

    def get_band(self, name: str) -> np.ndarray:
        """Return a single band as a (height, width) float32 ndarray."""
        if name not in self.band_names:
            raise KeyError(f"Band '{name}' not found. Available: {self.band_names}")
        return self.data[name].values.astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, method: str = "minmax") -> "ImageData":
        """Return a normalised copy of this ImageData.

        Parameters
        ----------
        method:
            ``"minmax"`` — scale each band to [0, 1].
            ``"zscore"`` — zero-mean unit-variance per band.
            ``"percentile"`` — clip to 2nd/98th percentile then scale to [0, 1].
        """
        arr = self.to_numpy()
        out = np.empty_like(arr, dtype=np.float32)

        for i in range(arr.shape[0]):
            band = arr[i].copy()
            if self.nodata is not None:
                valid = band != self.nodata
            else:
                valid = np.isfinite(band)

            if method == "minmax":
                lo = float(band[valid].min()) if valid.any() else 0.0
                hi = float(band[valid].max()) if valid.any() else 1.0
                denom = (hi - lo) if hi != lo else 1.0
                out[i] = (band - lo) / denom
            elif method == "zscore":
                mu = float(band[valid].mean()) if valid.any() else 0.0
                sigma = float(band[valid].std()) if valid.any() else 1.0
                sigma = sigma if sigma > 0 else 1.0
                out[i] = (band - mu) / sigma
            elif method == "percentile":
                lo = float(np.percentile(band[valid], 2)) if valid.any() else 0.0
                hi = float(np.percentile(band[valid], 98)) if valid.any() else 1.0
                denom = (hi - lo) if hi != lo else 1.0
                out[i] = np.clip((band - lo) / denom, 0.0, 1.0)
            else:
                raise ValueError(f"Unknown normalisation method: {method!r}")

        new_ds = _array_to_dataset(out, self.band_names, self.data)
        result = copy.copy(self)
        result.data = new_ds
        result.bit_depth = 32
        result.nodata = None
        logger.debug("normalized", sensor=self.sensor_type, method=method)
        return result

    # ------------------------------------------------------------------
    # Reprojection
    # ------------------------------------------------------------------

    def reproject(self, target_crs: str) -> "ImageData":
        """Return a copy reprojected to *target_crs* (EPSG code or WKT)."""
        if rasterio is None:
            raise ImportError("rasterio is required for reproject()")

        src_crs = CRS.from_user_input(self.crs)
        dst_crs = CRS.from_user_input(target_crs)

        arr = self.to_numpy()  # (bands, H, W)
        nbands, height, width = arr.shape

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, *self.bounds
        )

        reprojected = np.empty(
            (nbands, dst_height, dst_width), dtype=arr.dtype
        )
        for i in range(nbands):
            reproject(
                source=arr[i],
                destination=reprojected[i],
                src_transform=self.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=self.nodata,
                dst_nodata=self.nodata,
            )

        new_ds = _array_to_dataset(reprojected, self.band_names, self.data)
        from rasterio.transform import array_bounds

        west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)

        result = copy.copy(self)
        result.data = new_ds
        result.crs = target_crs
        result.transform = dst_transform
        result.bounds = (west, south, east, north)
        logger.debug("reprojected", target_crs=target_crs)
        return result

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(self, target_resolution_m: float) -> "ImageData":
        """Return a copy resampled to *target_resolution_m* ground resolution."""
        if rasterio is None:
            raise ImportError("rasterio is required for resample()")

        arr = self.to_numpy()
        nbands, height, width = arr.shape

        scale = self.resolution_m / target_resolution_m
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))

        from rasterio.transform import from_bounds as _from_bounds

        new_transform = _from_bounds(*self.bounds, new_width, new_height)

        resampled = np.empty((nbands, new_height, new_width), dtype=arr.dtype)
        src_crs = CRS.from_user_input(self.crs)

        for i in range(nbands):
            reproject(
                source=arr[i],
                destination=resampled[i],
                src_transform=self.transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear,
                src_nodata=self.nodata,
                dst_nodata=self.nodata,
            )

        new_ds = _array_to_dataset(resampled, self.band_names, self.data)
        result = copy.copy(self)
        result.data = new_ds
        result.transform = new_transform
        result.resolution_m = target_resolution_m
        logger.debug("resampled", target_resolution_m=target_resolution_m)
        return result

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    def get_rgb(self) -> np.ndarray:
        """Return a (H, W, 3) uint8 array suitable for display.

        Attempts to select RGB bands intelligently based on band names or
        sensor type. Falls back to the first three bands.
        """
        rgb_candidates = [
            ["R", "G", "B"],
            ["red", "green", "blue"],
            ["B04", "B03", "B02"],   # Sentinel-2
            ["SR_B4", "SR_B3", "SR_B2"],  # Landsat 8/9
            ["band_4", "band_3", "band_2"],
        ]

        def _pick_rgb() -> list[str]:
            lower = [b.lower() for b in self.band_names]
            for triplet in rgb_candidates:
                t_lower = [t.lower() for t in triplet]
                if all(t in lower for t in t_lower):
                    idx = [lower.index(t) for t in t_lower]
                    return [self.band_names[i] for i in idx]
            # Fall back to first three available bands
            return self.band_names[:3]

        chosen = _pick_rgb()
        if len(chosen) < 3:
            raise ValueError("Need at least 3 bands to produce RGB output.")

        channels = []
        for name in chosen:
            band = self.data[name].values.astype(np.float32)
            lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
            denom = (hi - lo) if hi != lo else 1.0
            channels.append(np.clip((band - lo) / denom, 0.0, 1.0))

        rgb = np.stack(channels, axis=-1)
        return (rgb * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        h = self.data[self.band_names[0]].shape[0] if self.band_names else "?"
        w = self.data[self.band_names[0]].shape[1] if self.band_names else "?"
        return (
            f"ImageData(sensor={self.sensor_type.value}, crs={self.crs}, "
            f"bands={self.band_names}, shape=({h},{w}), res={self.resolution_m}m)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _array_to_dataset(
    arr: np.ndarray,
    band_names: list[str],
    template_ds: "xr.Dataset",
) -> "xr.Dataset":
    """Create an xarray.Dataset from a (bands, H, W) ndarray."""
    if xr is None:
        raise ImportError("xarray is required")

    # Reuse coordinate arrays from template if dimensions match
    nbands, height, width = arr.shape
    if (
        "y" in template_ds.coords
        and len(template_ds.coords["y"]) == height
        and "x" in template_ds.coords
        and len(template_ds.coords["x"]) == width
    ):
        coords: dict = {
            "y": template_ds.coords["y"],
            "x": template_ds.coords["x"],
        }
    else:
        coords = {}

    return xr.Dataset(
        {name: (["y", "x"], arr[i]) for i, name in enumerate(band_names)},
        coords=coords,
    )
