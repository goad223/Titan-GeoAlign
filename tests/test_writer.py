"""Tests for GeoTIFF writing and PNG export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import WriterError
from changemaster.io_engine.writer import export_png, normalize_to_uint8, write_geotiff


class TestNormalize:
    """Contrast normalization to uint8."""

    def test_basic_stretch(self) -> None:
        data = np.linspace(0, 1000, 100).reshape(10, 10)
        out = normalize_to_uint8(data, percentile_clip=0)
        assert out.dtype == np.uint8
        assert out.min() == 0
        assert out.max() == 255

    def test_constant_array_returns_zeros(self) -> None:
        out = normalize_to_uint8(np.full((5, 5), 42.0))
        assert out.dtype == np.uint8
        assert not out.any()

    def test_nan_handling(self) -> None:
        data = np.array([[np.nan, 0.0], [50.0, 100.0]])
        out = normalize_to_uint8(data, percentile_clip=0)
        assert out.dtype == np.uint8
        assert out[1, 1] == 255

    def test_all_nan_returns_zeros(self) -> None:
        out = normalize_to_uint8(np.full((3, 3), np.nan))
        assert not out.any()

    def test_invalid_clip(self) -> None:
        with pytest.raises(WriterError):
            normalize_to_uint8(np.zeros((2, 2)), percentile_clip=60)


class TestExportPng:
    """PNG quick-look export (always available via Pillow)."""

    def test_grayscale(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        from PIL import Image

        out = export_png(tmp_path / "out.png", gray_array)
        with Image.open(out) as img:
            assert img.mode == "L"
            assert img.size == (gray_array.shape[1], gray_array.shape[0])

    def test_rgb_from_multiband(self, tmp_path: Path, rgb_array: np.ndarray) -> None:
        from PIL import Image

        out = export_png(tmp_path / "out.png", rgb_array)
        with Image.open(out) as img:
            assert img.mode == "RGB"

    def test_creates_parent_dirs(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        out = export_png(tmp_path / "a" / "b" / "out.png", gray_array)
        assert out.exists()

    def test_invalid_shape(self, tmp_path: Path) -> None:
        with pytest.raises(WriterError):
            export_png(tmp_path / "x.png", np.zeros((2, 2, 2, 2)))


class TestWriteGeotiff:
    """GeoTIFF writing preserving CRS/transform (requires rasterio)."""

    @pytest.fixture(autouse=True)
    def _require_rasterio(self) -> None:
        pytest.importorskip("rasterio")

    def test_roundtrip_preserves_georeferencing(
        self, tmp_path: Path, geotiff_file: Path, gray_array: np.ndarray
    ) -> None:
        from changemaster.io_engine.raster_reader import RasterReader

        with RasterReader(geotiff_file) as reader:
            src_meta = reader.metadata
            data = reader.read()
        out = write_geotiff(tmp_path / "copy.tif", data, metadata=src_meta)
        with RasterReader(out) as reader:
            meta = reader.metadata
            assert meta.crs == src_meta.crs
            assert meta.transform == src_meta.transform
            assert meta.nodata == src_meta.nodata
            np.testing.assert_array_equal(reader.read()[0], gray_array)

    def test_write_2d_array(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.raster_reader import RasterReader

        out = write_geotiff(tmp_path / "plain.tif", gray_array)
        with RasterReader(out) as reader:
            assert reader.metadata.band_count == 1
            np.testing.assert_array_equal(reader.read_band(1), gray_array)

    def test_explicit_overrides(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.raster_reader import RasterReader

        transform = (30.0, 0.0, 600000.0, 0.0, -30.0, 3900000.0)
        out = write_geotiff(
            tmp_path / "geo.tif",
            gray_array,
            crs="EPSG:32637",
            transform=transform,
            nodata=65535,
        )
        with RasterReader(out) as reader:
            meta = reader.metadata
            assert meta.crs is not None and "32637" in meta.crs
            assert meta.transform == transform
            assert meta.nodata == 65535

    def test_invalid_shape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(WriterError):
            write_geotiff(tmp_path / "bad.tif", np.zeros((2, 2, 2, 2)))

    def test_write_failure_raises_writer_error(self, tmp_path: Path) -> None:
        target = tmp_path / "is_a_dir.tif"
        target.mkdir()
        with pytest.raises(WriterError):
            write_geotiff(target, np.zeros((2, 2), dtype=np.uint8))
