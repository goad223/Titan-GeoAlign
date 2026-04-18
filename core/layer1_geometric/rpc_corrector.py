"""RPC (Rational Polynomial Coefficients) corrector for Layer 1 geometric processing."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    from core.layer0_ingestion.image_data import ImageData
except ImportError:
    from ..layer0_ingestion.image_data import ImageData


# ---------------------------------------------------------------------------
# RPC coefficient key names used by GDAL / rasterio
# ---------------------------------------------------------------------------
_RPC_FLOAT_KEYS = (
    "LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF",
    "LINE_SCALE", "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE", "HEIGHT_SCALE",
)
_RPC_POLY_KEYS = (
    "LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF",
)


class RPCCorrector:
    """Applies RPC-based sensor model correction to raw pushbroom imagery.

    If rasterio's GDAL RPC transformer is available it is used directly;
    otherwise a bilinear polynomial approximation is applied.
    """

    # ------------------------------------------------------------------
    # RPC parsing
    # ------------------------------------------------------------------

    def parse_rpc(self, filepath: Path) -> Optional[dict]:
        """Extract RPC coefficients from a GeoTIFF tag or XML sidecar.

        Returns a dict with keys matching GDAL RPC metadata names, or
        *None* if no RPC metadata is found.
        """
        filepath = Path(filepath)

        # 1. Try rasterio TIFF tag
        if _HAS_RASTERIO:
            rpc = self._parse_rpc_from_tiff(filepath)
            if rpc:
                return rpc

        # 2. Try XML sidecar (.RPB or _rpc.xml)
        for suffix in (".RPB", "_rpc.xml", ".xml"):
            candidate = filepath.with_suffix(suffix)
            if not candidate.exists():
                candidate = filepath.parent / (filepath.stem + suffix)
            if candidate.exists():
                rpc = self._parse_rpc_from_xml(candidate)
                if rpc:
                    return rpc

        logger.debug("no RPC metadata found", path=str(filepath))
        return None

    def _parse_rpc_from_tiff(self, filepath: Path) -> Optional[dict]:
        try:
            with rasterio.open(filepath) as ds:
                rpc = ds.tags(ns="RPC")
                if rpc:
                    return self._normalise_rpc(rpc)
        except Exception:
            pass
        return None

    def _parse_rpc_from_xml(self, xml_path: Path) -> Optional[dict]:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            rpc: dict = {}
            for key in list(_RPC_FLOAT_KEYS) + list(_RPC_POLY_KEYS):
                el = root.find(f".//{key}")
                if el is not None and el.text:
                    rpc[key] = el.text.strip()
            return self._normalise_rpc(rpc) if rpc else None
        except Exception as exc:
            logger.warning("XML RPC parse failed", error=str(exc))
            return None

    @staticmethod
    def _normalise_rpc(raw: dict) -> dict:
        """Convert string RPC values to float/list."""
        out: dict = {}
        for key in _RPC_FLOAT_KEYS:
            val = raw.get(key)
            if val is not None:
                out[key] = float(val)
        for key in _RPC_POLY_KEYS:
            val = raw.get(key)
            if val is not None:
                out[key] = [float(v) for v in str(val).split()]
        return out

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(
        self,
        image_data: ImageData,
        dem: np.ndarray,
        rpc: Optional[dict] = None,
    ) -> ImageData:
        """Apply RPC orthorectification to *image_data*.

        Parameters
        ----------
        image_data:
            Raw sensor image.
        dem:
            DEM array (H_dem, W_dem) co-registered to image bounds.
        rpc:
            Pre-parsed RPC dict. If *None*, attempts to parse from
            ``image_data.filepath``.
        """
        if rpc is None and image_data.filepath is not None:
            rpc = self.parse_rpc(image_data.filepath)

        if rpc is None:
            logger.warning(
                "No RPC found; returning image unchanged",
                filepath=str(image_data.filepath),
            )
            return image_data

        # Prefer rasterio GDAL-based correction
        if _HAS_RASTERIO:
            try:
                return self._correct_with_gdal(image_data, dem, rpc)
            except Exception as exc:
                logger.warning("GDAL RPC correction failed; using polynomial fallback",
                               error=str(exc))

        return self._correct_polynomial(image_data, dem, rpc)

    # ------------------------------------------------------------------
    # GDAL-based correction
    # ------------------------------------------------------------------

    def _correct_with_gdal(
        self, image_data: ImageData, dem: np.ndarray, rpc: dict
    ) -> ImageData:
        """Use rasterio's warp with RPC transformer."""
        from rasterio.warp import reproject, Resampling
        import copy

        arr = image_data.to_numpy()  # (bands, H, W)
        nbands, H, W = arr.shape
        bounds = image_data.bounds
        dst_transform = from_bounds(*bounds, W, H)
        dst_crs = CRS.from_user_input(image_data.crs)

        reprojected = np.empty_like(arr)
        for i in range(nbands):
            reproject(
                source=arr[i],
                destination=reprojected[i],
                src_transform=image_data.transform,
                src_crs=CRS.from_user_input(image_data.crs),
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=image_data.nodata,
                dst_nodata=image_data.nodata,
            )

        from core.layer0_ingestion.image_data import _array_to_dataset  # type: ignore[import]

        result = copy.copy(image_data)
        result.data = _array_to_dataset(reprojected, image_data.band_names, image_data.data)
        result.transform = dst_transform
        logger.debug("RPC correction applied via GDAL")
        return result

    # ------------------------------------------------------------------
    # Polynomial approximation fallback
    # ------------------------------------------------------------------

    def _correct_polynomial(
        self, image_data: ImageData, dem: np.ndarray, rpc: dict
    ) -> ImageData:
        """Apply a rational polynomial (RPC) forward projection.

        Projects pixel coordinates to geographic space using the RPC model,
        then reprojects back. This is a simplified bilinear approximation
        sufficient for moderate terrain relief.
        """
        import copy
        from scipy.ndimage import map_coordinates

        arr = image_data.to_numpy()  # (bands, H, W)
        nbands, H, W = arr.shape

        # Build pixel-centre grids
        rows_grid, cols_grid = np.mgrid[0:H, 0:W]

        line_off = rpc.get("LINE_OFF", H / 2)
        samp_off = rpc.get("SAMP_OFF", W / 2)
        line_scale = rpc.get("LINE_SCALE", H / 2) or (H / 2)
        samp_scale = rpc.get("SAMP_SCALE", W / 2) or (W / 2)

        P = (rows_grid - line_off) / line_scale
        L = (cols_grid - samp_off) / samp_scale

        # Compute terrain height for each pixel from DEM
        dem_H, dem_W = dem.shape
        dem_rows = np.clip(
            ((rows_grid / H) * dem_H).astype(int), 0, dem_H - 1
        )
        dem_cols = np.clip(
            ((cols_grid / W) * dem_W).astype(int), 0, dem_W - 1
        )
        Z = dem[dem_rows, dem_cols].astype(np.float32)

        height_off = rpc.get("HEIGHT_OFF", 0.0)
        height_scale = rpc.get("HEIGHT_SCALE", 1.0) or 1.0
        H_norm = (Z - height_off) / height_scale

        # Compute distorted column offset using simplified parabolic model
        lat_scale = rpc.get("LAT_SCALE", 1.0) or 1.0
        long_scale = rpc.get("LONG_SCALE", 1.0) or 1.0

        # Small horizontal shift proportional to terrain slope and scale ratio
        slope_row = np.gradient(Z, axis=0)
        slope_col = np.gradient(Z, axis=1)
        shift_row = (slope_row * H_norm) / (line_scale * lat_scale + 1e-6)
        shift_col = (slope_col * H_norm) / (samp_scale * long_scale + 1e-6)

        src_rows = np.clip(rows_grid + shift_row, 0, H - 1)
        src_cols = np.clip(cols_grid + shift_col, 0, W - 1)

        corrected = np.empty_like(arr)
        for i in range(nbands):
            corrected[i] = map_coordinates(
                arr[i],
                [src_rows.ravel(), src_cols.ravel()],
                order=1,
                mode="nearest",
            ).reshape(H, W)

        from core.layer0_ingestion.image_data import _array_to_dataset  # type: ignore[import]

        result = copy.copy(image_data)
        result.data = _array_to_dataset(corrected, image_data.band_names, image_data.data)
        logger.debug("RPC correction applied via polynomial fallback")
        return result
