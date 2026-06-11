"""Tests for the rasterio-backed raster reader (skipped without rasterio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from changemaster.core.exceptions import BandError, ReaderError
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.raster_reader import RasterReader


def test_reader_reports_available() -> None:
    assert RasterReader.is_available()
    assert RasterReader.required_package() == "rasterio"


def test_read_geotiff_metadata(geotiff_file: Path, gray_array: np.ndarray) -> None:
    with RasterReader(geotiff_file) as reader:
        meta = reader.metadata
        assert meta.width == gray_array.shape[1]
        assert meta.height == gray_array.shape[0]
        assert meta.band_count == 1
        assert meta.dtype == "uint16"
        assert meta.crs is not None and "32636" in meta.crs
        assert meta.transform is not None
        assert meta.transform[0] == 10.0  # pixel size x
        assert meta.nodata == 0.0
        assert meta.is_georeferenced


def test_read_full_and_band(geotiff_file: Path, gray_array: np.ndarray) -> None:
    with RasterReader(geotiff_file) as reader:
        data = reader.read()
        assert data.shape == (1, *gray_array.shape)
        np.testing.assert_array_equal(data[0], gray_array)
        np.testing.assert_array_equal(reader.read_band(1), gray_array)


def test_read_window(geotiff_file: Path, gray_array: np.ndarray) -> None:
    with RasterReader(geotiff_file) as reader:
        window = (10, 5, 20, 15)
        data = reader.read(window=window)
        assert data.shape == (1, 15, 20)
        np.testing.assert_array_equal(data[0], gray_array[5:20, 10:30])
        band = reader.read_band(1, window=window)
        np.testing.assert_array_equal(band, gray_array[5:20, 10:30])


def test_multiband(multiband_geotiff: Path) -> None:
    with RasterReader(multiband_geotiff) as reader:
        assert reader.metadata.band_count == 4
        assert reader.read().shape == (4, 60, 80)
        with pytest.raises(BandError):
            reader.read_band(5)


def test_factory_selects_raster_reader(geotiff_file: Path) -> None:
    reader = open_image(geotiff_file)
    try:
        assert isinstance(reader, RasterReader)
    finally:
        reader.close()


def test_corrupt_tif_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff at all")
    with pytest.raises(ReaderError):
        RasterReader(bad).open()


def test_close_idempotent(geotiff_file: Path) -> None:
    reader = RasterReader(geotiff_file).open()
    reader.close()
    reader.close()
