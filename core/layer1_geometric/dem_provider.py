"""DEM provider — downloads or loads Digital Elevation Models for orthorectification."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    rasterio = None  # type: ignore[assignment]

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

# Supported DEM sources
_SOURCES = {"copernicus_glo30", "arcticdem", "srtm", "3dep"}


class DEMProvider:
    """Downloads and caches Digital Elevation Models.

    Parameters
    ----------
    source:
        DEM dataset to use: ``"copernicus_glo30"`` (default), ``"srtm"``,
        ``"arcticdem"``, or ``"3dep"``.
    cache_dir:
        Directory for cached DEM tiles.
    opentopo_api_key:
        OpenTopography API key (required for ``"3dep"`` and some Copernicus
        products). Falls back to the ``OPENTOPO_API_KEY`` environment variable.
    """

    _OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
    _SRTM_DEMTYPE = "SRTMGL1"
    _COP30_DEMTYPE = "COP30"

    def __init__(
        self,
        source: str = "copernicus_glo30",
        cache_dir: Path = Path("data/dem_cache"),
        opentopo_api_key: Optional[str] = None,
    ) -> None:
        if source not in _SOURCES:
            raise ValueError(f"Unknown DEM source {source!r}. Choose from {_SOURCES}.")
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = opentopo_api_key or os.environ.get("OPENTOPO_API_KEY", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dem(
        self,
        bounds: tuple[float, float, float, float],
        resolution_m: float = 30.0,
    ) -> np.ndarray:
        """Return a DEM array (H, W) for the given bounds.

        Parameters
        ----------
        bounds:
            ``(west, south, east, north)`` in WGS84 decimal degrees.
        resolution_m:
            Target resolution in metres.
        """
        cache_path = self.get_dem_filename(bounds)

        if cache_path.exists():
            logger.debug("loading DEM from cache", path=str(cache_path))
            return self._load_cached_dem(cache_path, bounds, resolution_m)

        # Try live download
        dem = self._try_download(bounds)
        if dem is None:
            logger.warning("DEM download failed; generating synthetic flat DEM")
            dem = self._synthetic_dem(bounds, resolution_m)

        # Cache result
        self._save_dem(dem, bounds, cache_path)
        return self._resample_to_resolution(dem, bounds, resolution_m)

    def get_dem_filename(self, bounds: tuple[float, float, float, float]) -> Path:
        """Return the cache file path for *bounds*."""
        key = f"{self.source}_{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"dem_{h}.tif"

    # ------------------------------------------------------------------
    # Download strategies
    # ------------------------------------------------------------------

    def _try_download(
        self, bounds: tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Attempt DEM download; return None on any failure."""
        if requests is None:
            logger.warning("requests library not installed; skipping DEM download")
            return None

        try:
            dem_type = self._source_to_demtype()
            return self._download_opentopography(bounds, dem_type)
        except Exception as exc:
            logger.warning("OpenTopography download failed", error=str(exc))

        # Try the elevation Python package (SRTM) as second option
        try:
            return self._download_via_elevation_pkg(bounds)
        except Exception as exc:
            logger.warning("elevation package download failed", error=str(exc))

        return None

    def _source_to_demtype(self) -> str:
        mapping = {
            "copernicus_glo30": self._COP30_DEMTYPE,
            "srtm": self._SRTM_DEMTYPE,
            "arcticdem": "ArcticDEM",
            "3dep": "3DEPEllipsoidal",
        }
        return mapping.get(self.source, self._SRTM_DEMTYPE)

    def _download_opentopography(
        self, bounds: tuple[float, float, float, float], dem_type: str
    ) -> np.ndarray:
        west, south, east, north = bounds
        params = {
            "demtype": dem_type,
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
        }
        if self.api_key:
            params["API_Key"] = self.api_key

        logger.info("downloading DEM from OpenTopography", dem_type=dem_type)
        resp = requests.get(self._OPENTOPO_URL, params=params, timeout=120, stream=True)
        resp.raise_for_status()

        raw_path = self.cache_dir / "_dem_raw.tif"
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        if rasterio is None:
            raise ImportError("rasterio required to read downloaded DEM")

        with rasterio.open(raw_path) as ds:
            arr = ds.read(1).astype(np.float32)

        raw_path.unlink(missing_ok=True)
        return arr

    def _download_via_elevation_pkg(
        self, bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        import elevation  # type: ignore[import]

        west, south, east, north = bounds
        output = self.cache_dir / "_elevation_tmp.tif"
        elevation.clip(bounds=(west, south, east, north), output=str(output))
        elevation.clean()

        with rasterio.open(output) as ds:
            arr = ds.read(1).astype(np.float32)
        output.unlink(missing_ok=True)
        return arr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _synthetic_dem(
        self, bounds: tuple, resolution_m: float
    ) -> np.ndarray:
        """Generate a flat (zero-elevation) DEM as fallback."""
        west, south, east, north = bounds
        deg_per_m = 1.0 / 111320.0
        cols = max(1, int((east - west) / (resolution_m * deg_per_m)))
        rows = max(1, int((north - south) / (resolution_m * deg_per_m)))
        return np.zeros((rows, cols), dtype=np.float32)

    def _save_dem(
        self, dem: np.ndarray, bounds: tuple, path: Path
    ) -> None:
        if rasterio is None:
            return
        rows, cols = dem.shape
        west, south, east, north = bounds
        transform = from_bounds(west, south, east, north, cols, rows)
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=rows, width=cols,
            count=1, dtype=dem.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as ds:
            ds.write(dem, 1)
        logger.debug("cached DEM", path=str(path))

    def _load_cached_dem(
        self, path: Path, bounds: tuple, resolution_m: float
    ) -> np.ndarray:
        if rasterio is None:
            raise ImportError("rasterio required to load cached DEM")
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
        return self._resample_to_resolution(arr, bounds, resolution_m)

    def _resample_to_resolution(
        self, dem: np.ndarray, bounds: tuple, resolution_m: float
    ) -> np.ndarray:
        if rasterio is None:
            return dem
        west, south, east, north = bounds
        current_res_deg = (east - west) / dem.shape[1] if dem.shape[1] > 0 else 1e-4
        current_res_m = current_res_deg * 111320.0
        if abs(current_res_m - resolution_m) < 1.0:
            return dem  # close enough

        scale = current_res_m / resolution_m
        new_rows = max(1, int(round(dem.shape[0] * scale)))
        new_cols = max(1, int(round(dem.shape[1] * scale)))
        new_transform = from_bounds(west, south, east, north, new_cols, new_rows)
        src_transform = from_bounds(west, south, east, north, dem.shape[1], dem.shape[0])

        resampled = np.empty((new_rows, new_cols), dtype=np.float32)
        reproject(
            source=dem,
            destination=resampled,
            src_transform=src_transform,
            src_crs=CRS.from_epsg(4326),
            dst_transform=new_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
        )
        return resampled
