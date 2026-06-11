"""I/O engine: unified readers, tiled access and writers for ChangeMaster.

Importing this package registers every reader with the registry. Readers
whose optional dependencies are missing remain registered (so they appear in
the formats table) but are skipped by the factory.
"""

from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.base_reader import (
    BaseReader,
    ReaderRegistry,
    open_image,
    iter_supported_extensions,
)
from changemaster.io_engine import simple_reader  # noqa: F401 - registers reader
from changemaster.io_engine import raster_reader  # noqa: F401 - registers reader
from changemaster.io_engine import hdf_reader  # noqa: F401 - registers reader
from changemaster.io_engine import netcdf_reader  # noqa: F401 - registers reader
from changemaster.io_engine import safe_reader  # noqa: F401 - registers reader
from changemaster.io_engine import landsat_reader  # noqa: F401 - registers reader
from changemaster.io_engine.tiled_access import TiledReader, Tile, compute_tile_grid
from changemaster.io_engine.writer import write_geotiff, export_png, normalize_to_uint8

__all__ = [
    "ImageMetadata",
    "BaseReader",
    "ReaderRegistry",
    "open_image",
    "iter_supported_extensions",
    "TiledReader",
    "Tile",
    "compute_tile_grid",
    "write_geotiff",
    "export_png",
    "normalize_to_uint8",
]
