"""ChangeMaster Ultimate — offline satellite-image change detection.

Phase 1 (Foundation): hardware adaptation, configuration, logging, a unified
multi-format I/O engine, tiled access for giant rasters, and sensor profiles.

برنامج ChangeMaster Ultimate — كشف التغيرات بين صور الأقمار الصناعية
يعمل أوفلاين بالكامل ويتكيف مع أي عتاد.
"""

from __future__ import annotations

__version__ = "0.1.0"

from changemaster.core import (
    AppConfig,
    ChangeMasterError,
    ConfigManager,
    HardwareProfile,
    HardwareTier,
    detect_hardware,
    get_logger,
    setup_logging,
)
from changemaster.io_engine import (
    BaseReader,
    ImageMetadata,
    ReaderRegistry,
    TiledReader,
    export_png,
    open_image,
    write_geotiff,
)
from changemaster.sensors import SensorProfile, SensorRegistry

__all__ = [
    "__version__",
    "AppConfig",
    "ChangeMasterError",
    "ConfigManager",
    "HardwareProfile",
    "HardwareTier",
    "detect_hardware",
    "get_logger",
    "setup_logging",
    "BaseReader",
    "ImageMetadata",
    "ReaderRegistry",
    "TiledReader",
    "export_png",
    "open_image",
    "write_geotiff",
    "SensorProfile",
    "SensorRegistry",
]
