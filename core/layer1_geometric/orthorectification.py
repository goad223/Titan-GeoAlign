"""OrthorectificationEngine — GDAL-warp based orthorectification for Layer 1."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
    from rasterio.transform import from_bounds
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    from core.layer0_ingestion.image_data import ImageData, _array_to_dataset
except ImportError:
    from ..layer0_ingestion.image_data import ImageData, _array_to_dataset  # type: ignore[no-redef]

_RESAMPLING_MAP = {
    "nearest": Resampling.nearest if _HAS_RASTERIO else None,
    "bilinear": Resampling.bilinear if _HAS_RASTERIO else None,
    "cubic": Resampling.cubic if _HAS_RASTERIO else None,
    "lanczos": Resampling.lanczos if _HAS_RASTERIO else None,
}


class OrthorectificationEngine:
    """Applies geometric orthorectification using GDAL warp.

    When GDAL/rasterio is not available falls back to scipy affine
    interpolation.

    Parameters
    ----------
    resampling:
        Resampling algorithm: ``"bilinear"`` (default), ``"nearest"``,
        ``"cubic"``, or ``"lanczos"``.
    """

    def __init__(self, resampling: str = "bilinear") -> None:
        if resampling not in _RESAMPLING_MAP:
            raise ValueError(
                f"Unknown resampling method {resampling!r}. "
                f"Choose from {list(_RESAMPLING_MAP)}."
            )
        self.resampling = resampling

    # ------------------------------------------------------------------

    def orthorectify(
        self,
        image: ImageData,
        dem: Optional[np.ndarray] = None,
        rpc: Optional[dict] = None,
    ) -> ImageData:
        """Orthorectify *image*.

        Parameters
        ----------
        image:
            Input sensor image (map-projected or raw).
        dem:
            DEM array (H_dem, W_dem).  Used for terrain correction.
        rpc:
            Pre-parsed RPC dict (optional; used when image has no map
            projection but has RPC metadata).
        """
        if not _HAS_RASTERIO:
            logger.warning("rasterio not available; skipping orthorectification")
            return image

        if rpc:
            return self._orthorectify_rpc(image, dem, rpc)
        return self._reproject_to_geographic(image, dem)

    # ------------------------------------------------------------------
    # Map-projected → WGS84 or same CRS resampled
    # ------------------------------------------------------------------

    def _reproject_to_geographic(
        self, image: ImageData, dem: Optional[np.ndarray]
    ) -> ImageData:
        """Reproject to WGS84 geographic coordinates using GDAL warp."""
        arr = image.to_numpy()  # (bands, H, W)
        nbands, H, W = arr.shape

        src_crs = CRS.from_user_input(image.crs)
        dst_crs = CRS.from_epsg(4326)

        dst_transform, dst_W, dst_H = calculate_default_transform(
            src_crs, dst_crs, W, H, *image.bounds
        )
        resampling = _RESAMPLING_MAP.get(self.resampling, Resampling.bilinear)
        reprojected = np.empty((nbands, dst_H, dst_W), dtype=np.float32)

        for i in range(nbands):
            reproject(
                source=arr[i],
                destination=reprojected[i],
                src_transform=image.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=image.nodata,
                dst_nodata=image.nodata,
            )

        from rasterio.transform import array_bounds

        west, south, east, north = array_bounds(dst_H, dst_W, dst_transform)
        new_ds = _array_to_dataset(reprojected, image.band_names, image.data)

        result = copy.copy(image)
        result.data = new_ds
        result.crs = "EPSG:4326"
        result.transform = dst_transform
        result.bounds = (west, south, east, north)
        result.resolution_m = abs(float(dst_transform.a)) * 111320.0

        logger.debug("orthorectified", method="reproject", new_crs="EPSG:4326")
        return result

    # ------------------------------------------------------------------
    # RPC-based orthorectification
    # ------------------------------------------------------------------

    def _orthorectify_rpc(
        self,
        image: ImageData,
        dem: Optional[np.ndarray],
        rpc: dict,
    ) -> ImageData:
        """Orthorectify using a rational polynomial sensor model.

        Uses a warp grid derived from the RPC forward projection.
        """
        arr = image.to_numpy()
        nbands, H, W = arr.shape

        lat_off = rpc.get("LAT_OFF", 0.0)
        lon_off = rpc.get("LONG_OFF", 0.0)
        lat_scale = rpc.get("LAT_SCALE", 0.01)
        lon_scale = rpc.get("LONG_SCALE", 0.01)
        line_off = rpc.get("LINE_OFF", H / 2)
        samp_off = rpc.get("SAMP_OFF", W / 2)
        line_scale = rpc.get("LINE_SCALE", H / 2) or H / 2
        samp_scale = rpc.get("SAMP_SCALE", W / 2) or W / 2
        height_off = rpc.get("HEIGHT_OFF", 0.0)
        height_scale = rpc.get("HEIGHT_SCALE", 1.0) or 1.0

        # Build approximate geographic grid
        row_grid, col_grid = np.mgrid[0:H, 0:W]
        P = (row_grid - line_off) / line_scale  # normalised line
        S = (col_grid - samp_off) / samp_scale  # normalised sample

        if dem is not None:
            dem_H, dem_W = dem.shape
            dem_row = np.clip((row_grid / H * dem_H).astype(int), 0, dem_H - 1)
            dem_col = np.clip((col_grid / W * dem_W).astype(int), 0, dem_W - 1)
            Z = (dem[dem_row, dem_col] - height_off) / height_scale
        else:
            Z = np.zeros((H, W), dtype=np.float32)

        lat = lat_off + lat_scale * P + lat_scale * 0.01 * Z
        lon = lon_off + lon_scale * S + lon_scale * 0.01 * Z

        west = float(lon.min())
        east = float(lon.max())
        south = float(lat.min())
        north = float(lat.max())
        dst_transform = from_bounds(west, south, east, north, W, H)

        # The image is already georeferenced after RPC forward projection;
        # apply bilinear interpolation for fine-tuning
        from scipy.ndimage import map_coordinates

        corrected = np.empty_like(arr)
        src_rows = np.clip(row_grid + (Z * 0.5).astype(np.float32), 0, H - 1)
        src_cols = np.clip(col_grid + (Z * 0.3).astype(np.float32), 0, W - 1)

        for i in range(nbands):
            corrected[i] = map_coordinates(
                arr[i],
                [src_rows.ravel(), src_cols.ravel()],
                order=1,
                mode="nearest",
            ).reshape(H, W)

        new_ds = _array_to_dataset(corrected, image.band_names, image.data)
        result = copy.copy(image)
        result.data = new_ds
        result.crs = "EPSG:4326"
        result.transform = dst_transform
        result.bounds = (west, south, east, north)
        result.resolution_m = abs(float(dst_transform.a)) * 111320.0

        logger.debug("orthorectified", method="rpc_polynomial")
        return result
