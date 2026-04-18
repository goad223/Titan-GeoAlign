"""IngestPipeline — sensor detection + reader dispatch for Layer 0."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from .sensor_types import SensorType
from .sensor_detector import detect_sensor
from .image_data import ImageData
from .readers import (
    BaseReader,
    GeoTiffReader,
    Sentinel1Reader,
    Sentinel2Reader,
    Landsat89Reader,
    NetCDFReader,
    HDF5Reader,
    LiDARReader,
    UAVReader,
)

logger = structlog.get_logger(__name__)

# Ordered list of readers to try (most-specific first)
_READERS: list[BaseReader] = [
    Sentinel1Reader(SensorType.SENTINEL1_SAR),
    Sentinel2Reader(SensorType.SENTINEL2),
    Landsat89Reader(SensorType.LANDSAT8),
    NetCDFReader(),
    HDF5Reader(),
    LiDARReader(),
    UAVReader(),
    GeoTiffReader(),  # generic fallback
]

_SENSOR_TO_READER: dict[SensorType, BaseReader] = {
    SensorType.SENTINEL1_SAR: Sentinel1Reader(SensorType.SENTINEL1_SAR),
    SensorType.SENTINEL2: Sentinel2Reader(SensorType.SENTINEL2),
    SensorType.LANDSAT8: Landsat89Reader(SensorType.LANDSAT8),
    SensorType.LANDSAT9: Landsat89Reader(SensorType.LANDSAT9),
    SensorType.PLANETSCOPE: GeoTiffReader(SensorType.PLANETSCOPE),
    SensorType.WORLDVIEW: GeoTiffReader(SensorType.WORLDVIEW),
    SensorType.SKYSAT: GeoTiffReader(SensorType.SKYSAT),
    SensorType.MODIS: NetCDFReader(),
    SensorType.ENMAP: HDF5Reader(),
    SensorType.PRISMA: HDF5Reader(),
    SensorType.UAV_RGB: UAVReader(SensorType.UAV_RGB),
    SensorType.UAV_MULTISPECTRAL: UAVReader(SensorType.UAV_MULTISPECTRAL),
    SensorType.LIDAR: LiDARReader(),
}


class IngestPipeline:
    """Orchestrates sensor detection and reader dispatch.

    Parameters
    ----------
    sensor_type:
        Override sensor type. If *None*, auto-detected from the file path.
    readers:
        Optional list of custom :class:`~readers.BaseReader` implementations to
        try before the built-in readers.
    """

    def __init__(
        self,
        sensor_type: Optional[SensorType] = None,
        readers: Optional[list[BaseReader]] = None,
    ) -> None:
        self._default_sensor = sensor_type
        self._extra_readers: list[BaseReader] = readers or []

    # ------------------------------------------------------------------

    def ingest(self, filepath: str | Path) -> ImageData:
        """Detect sensor and read *filepath* into an :class:`ImageData`.

        Parameters
        ----------
        filepath:
            Path to a single file or a product directory (SAFE, Landsat scene).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Path does not exist: {filepath}")

        sensor = self._default_sensor or detect_sensor(filepath)
        logger.info("ingesting", path=str(filepath), sensor=sensor)

        reader = self._pick_reader(filepath, sensor)
        logger.debug("using reader", reader=type(reader).__name__)
        image = reader.read(filepath)

        # Ensure sensor type is set correctly
        if self._default_sensor:
            image.sensor_type = self._default_sensor

        return image

    # ------------------------------------------------------------------

    def _pick_reader(self, filepath: Path, sensor: SensorType) -> BaseReader:
        """Select the best reader for (filepath, sensor)."""
        # 1. Check extra readers (user-supplied)
        for r in self._extra_readers:
            if r.can_read(filepath):
                return r

        # 2. Exact sensor → reader mapping
        if sensor in _SENSOR_TO_READER:
            return _SENSOR_TO_READER[sensor]

        # 3. Walk built-in readers by capability
        for r in _READERS:
            if r.can_read(filepath):
                return r

        # 4. Last resort — generic GeoTIFF
        return GeoTiffReader(sensor)


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def ingest(
    filepath: str | Path,
    sensor_type: Optional[SensorType] = None,
) -> ImageData:
    """Read a sensor image file into an :class:`ImageData`.

    Parameters
    ----------
    filepath:
        Path to a single file or product directory.
    sensor_type:
        Optional override; auto-detected from filename/metadata if *None*.
    """
    pipeline = IngestPipeline(sensor_type=sensor_type)
    return pipeline.ingest(filepath)
