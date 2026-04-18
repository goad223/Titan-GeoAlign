"""Sensor auto-detection from file paths and metadata."""
from __future__ import annotations

import re
from pathlib import Path

import structlog

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore[assignment]

from .sensor_types import SensorType

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Filename-based patterns
# ---------------------------------------------------------------------------
_FILENAME_PATTERNS: list[tuple[re.Pattern, SensorType]] = [
    # Sentinel-1 (S1A_IW_GRDH_, S1B_EW_SLC_, etc.)
    (re.compile(r"S1[AB]_(?:IW|EW|S[1-6])_(?:GRD|SLC|OCN)", re.IGNORECASE), SensorType.SENTINEL1_SAR),
    # Sentinel-2 (S2A_MSIL2A_, S2B_MSIL1C_, etc.)
    (re.compile(r"S2[AB]_MSI", re.IGNORECASE), SensorType.SENTINEL2),
    # Landsat 8
    (re.compile(r"LC08_", re.IGNORECASE), SensorType.LANDSAT8),
    (re.compile(r"LO08_", re.IGNORECASE), SensorType.LANDSAT8),
    # Landsat 9
    (re.compile(r"LC09_", re.IGNORECASE), SensorType.LANDSAT9),
    (re.compile(r"LO09_", re.IGNORECASE), SensorType.LANDSAT9),
    # PlanetScope (PSScene, PSOrthoTile)
    (re.compile(r"PS(?:Scene|OrthoTile|Analytic)", re.IGNORECASE), SensorType.PLANETSCOPE),
    # SkySat
    (re.compile(r"SKY(?:SAT|_)", re.IGNORECASE), SensorType.SKYSAT),
    # WorldView
    (re.compile(r"WV0[1-4]_|WORLDVIEW", re.IGNORECASE), SensorType.WORLDVIEW),
    # MODIS
    (re.compile(r"MOD(?:09|13|14|17)|MYD(?:09|13)", re.IGNORECASE), SensorType.MODIS),
    # EnMAP
    (re.compile(r"ENMAP\d{2}", re.IGNORECASE), SensorType.ENMAP),
    # PRISMA
    (re.compile(r"PRS_L[12]_", re.IGNORECASE), SensorType.PRISMA),
    # LiDAR extensions
    (re.compile(r"\.la[sz]$", re.IGNORECASE), SensorType.LIDAR),
]

# ---------------------------------------------------------------------------
# Band-count heuristics
# ---------------------------------------------------------------------------
_BAND_COUNT_HINTS: dict[tuple[int, ...], SensorType] = {
    (13,): SensorType.SENTINEL2,
    (12,): SensorType.SENTINEL2,
    (2,): SensorType.SENTINEL1_SAR,
    (11,): SensorType.LANDSAT8,
    (8,): SensorType.PLANETSCOPE,
    (4,): SensorType.PLANETSCOPE,
}


def detect_sensor(filepath: Path) -> SensorType:
    """Best-effort sensor type detection from filename, metadata and band count.

    Parameters
    ----------
    filepath:
        Path to the image file or product directory.

    Returns
    -------
    SensorType
        Detected sensor, or ``SensorType.UNKNOWN`` if detection fails.
    """
    filepath = Path(filepath)
    name = filepath.name

    # 1. Filename patterns
    for pattern, sensor in _FILENAME_PATTERNS:
        if pattern.search(name) or pattern.search(str(filepath)):
            logger.debug("sensor detected via filename", sensor=sensor, path=str(filepath))
            return sensor

    # 2. For .SAFE directory (Sentinel-2)
    if filepath.is_dir() and filepath.suffix.upper() == ".SAFE":
        logger.debug("sensor detected via .SAFE suffix", sensor=SensorType.SENTINEL2)
        return SensorType.SENTINEL2

    # 3. GeoTIFF metadata inspection
    if rasterio is not None and filepath.is_file():
        try:
            sensor = _detect_from_rasterio(filepath)
            if sensor != SensorType.UNKNOWN:
                return sensor
        except Exception as exc:
            logger.warning("rasterio metadata inspection failed", error=str(exc))

    # 4. Extension-based fallback
    if filepath.suffix.lower() in {".nc", ".h5", ".hdf5"}:
        logger.debug("sensor hint from extension", suffix=filepath.suffix)
        return SensorType.MODIS  # likely MODIS or hyperspectral netCDF/HDF

    logger.debug("sensor not detected, returning UNKNOWN", path=str(filepath))
    return SensorType.UNKNOWN


def _detect_from_rasterio(filepath: Path) -> SensorType:
    """Inspect rasterio metadata for sensor clues."""
    with rasterio.open(filepath) as ds:
        tags: dict = ds.tags()
        band_count: int = ds.count
        pixel_x = abs(ds.transform.a) if ds.transform else 0.0

        # Tags may contain sensor / satellite info
        for key in ("TIFFTAG_IMAGEDESCRIPTION", "SATELLITE", "SENSOR",
                    "SPACECRAFT_ID", "SENSOR_ID"):
            value = tags.get(key, "")
            for pattern, sensor in _FILENAME_PATTERNS:
                if pattern.search(value):
                    logger.debug("sensor detected via TIFF tag", key=key, sensor=sensor)
                    return sensor

        # Band count + resolution heuristics
        res_rounded = round(pixel_x)

        if band_count == 2 and res_rounded in {10, 20}:
            return SensorType.SENTINEL1_SAR

        if band_count in {12, 13} and res_rounded in {10, 20, 60}:
            return SensorType.SENTINEL2

        if band_count == 11 and res_rounded == 30:
            return SensorType.LANDSAT8

        if band_count >= 100:
            return SensorType.ENMAP  # hyperspectral

        hint = _BAND_COUNT_HINTS.get((band_count,))
        if hint:
            return hint

    return SensorType.UNKNOWN
