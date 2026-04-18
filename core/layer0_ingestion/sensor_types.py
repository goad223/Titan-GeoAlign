"""Sensor type enumeration for Titan-GeoAlign."""
from enum import Enum, auto


class SensorType(str, Enum):
    """Enumeration of supported remote sensing sensor types."""

    SENTINEL1_SAR = "sentinel1_sar"
    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    LANDSAT9 = "landsat9"
    PLANETSCOPE = "planetscope"
    WORLDVIEW = "worldview"
    SKYSAT = "skysat"
    MODIS = "modis"
    ENMAP = "enmap"
    PRISMA = "prisma"
    UAV_RGB = "uav_rgb"
    UAV_MULTISPECTRAL = "uav_multispectral"
    LIDAR = "lidar"
    UNKNOWN = "unknown"
