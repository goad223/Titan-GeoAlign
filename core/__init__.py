"""Titan-GeoAlign core package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("titan-geoalign")
except PackageNotFoundError:
    __version__ = "0.0.0"
