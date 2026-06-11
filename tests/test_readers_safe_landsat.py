"""Tests for the Sentinel SAFE and Landsat product readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from changemaster.core.exceptions import BandError, MetadataError
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.landsat_reader import LandsatReader, parse_mtl
from changemaster.io_engine.safe_reader import SafeReader, parse_manifest


class TestSafeReader:
    """Sentinel .SAFE directory reading."""

    def test_claims_safe_dir(self, safe_product: Path) -> None:
        assert SafeReader.claims(safe_product)
        assert not SafeReader.claims(safe_product.parent)

    def test_metadata(self, safe_product: Path, gray_array: np.ndarray) -> None:
        with SafeReader(safe_product) as reader:
            meta = reader.metadata
            assert meta.band_count == 3
            assert meta.band_names == ["B02", "B03", "B04"]
            assert meta.sensor == "SENTINEL-2"
            assert meta.acquisition_datetime == "2024-05-01T10:20:30"
            assert meta.width == gray_array.shape[1]
            assert meta.crs is not None
            assert meta.is_georeferenced

    def test_read_stack_and_bands(self, safe_product: Path, gray_array: np.ndarray) -> None:
        with SafeReader(safe_product) as reader:
            stack = reader.read()
            assert stack.shape == (3, *gray_array.shape)
            np.testing.assert_array_equal(reader.read_band("B03"), gray_array)
            np.testing.assert_array_equal(reader.read_band(1), stack[0])

    def test_windowed_band_read(self, safe_product: Path, gray_array: np.ndarray) -> None:
        with SafeReader(safe_product) as reader:
            data = reader.read_band("B02", window=(2, 1, 10, 6))
            np.testing.assert_array_equal(data, gray_array[1:7, 2:12])

    def test_invalid_band(self, safe_product: Path) -> None:
        with SafeReader(safe_product) as reader:
            with pytest.raises(BandError):
                reader.read_band("B99")
            with pytest.raises(BandError):
                reader.read_band(7)

    def test_factory_selects_safe_reader(self, safe_product: Path) -> None:
        reader = open_image(safe_product)
        try:
            assert isinstance(reader, SafeReader)
        finally:
            reader.close()

    def test_missing_manifest(self, tmp_path: Path) -> None:
        bare = tmp_path / "EMPTY.SAFE"
        bare.mkdir()
        assert not SafeReader.claims(bare)

    def test_manifest_without_bands(self, tmp_path: Path, safe_product: Path) -> None:
        clone = tmp_path / "NOBANDS.SAFE"
        clone.mkdir()
        (clone / "manifest.safe").write_text(
            (safe_product / "manifest.safe").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with pytest.raises(MetadataError):
            SafeReader(clone).open()

    def test_parse_manifest_invalid_xml(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.safe"
        bad.write_text("<broken", encoding="utf-8")
        with pytest.raises(MetadataError):
            parse_manifest(bad)


class TestLandsatReader:
    """Landsat directory and tar archive reading."""

    def test_parse_mtl(self, landsat_dir: Path) -> None:
        mtl_path = next(landsat_dir.glob("*_MTL.txt"))
        mtl = parse_mtl(mtl_path)
        assert mtl["SPACECRAFT_ID"] == "LANDSAT_8"
        assert mtl["DATE_ACQUIRED"] == "2024-05-01"
        assert mtl["CLOUD_COVER"] == "1.23"
        assert "GROUP" not in mtl

    def test_claims(self, landsat_dir: Path, landsat_tar: Path, tmp_path: Path) -> None:
        assert LandsatReader.claims(landsat_dir)
        assert LandsatReader.claims(landsat_tar)
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        assert not LandsatReader.claims(empty)

    def test_metadata_from_dir(self, landsat_dir: Path, gray_array: np.ndarray) -> None:
        with LandsatReader(landsat_dir) as reader:
            meta = reader.metadata
            assert meta.band_count == 3
            assert meta.band_names == ["B2", "B3", "B4"]
            assert meta.sensor == "LANDSAT_8"
            assert meta.acquisition_datetime == "2024-05-01T08:30:15"
            assert meta.width == gray_array.shape[1]
            assert reader.mtl["CLOUD_COVER"] == "1.23"

    def test_read_from_dir(self, landsat_dir: Path, gray_array: np.ndarray) -> None:
        with LandsatReader(landsat_dir) as reader:
            stack = reader.read()
            assert stack.shape == (3, *gray_array.shape)
            np.testing.assert_array_equal(reader.read_band(2), gray_array)

    def test_read_from_tar(self, landsat_tar: Path, gray_array: np.ndarray) -> None:
        with LandsatReader(landsat_tar) as reader:
            assert reader.metadata.band_count == 3
            np.testing.assert_array_equal(reader.read_band(1), gray_array)

    def test_windowed_read(self, landsat_dir: Path, gray_array: np.ndarray) -> None:
        with LandsatReader(landsat_dir) as reader:
            data = reader.read(window=(4, 2, 6, 5))
            assert data.shape == (3, 5, 6)
            np.testing.assert_array_equal(data[0], gray_array[2:7, 4:10])

    def test_invalid_band(self, landsat_dir: Path) -> None:
        with LandsatReader(landsat_dir) as reader:
            with pytest.raises(BandError):
                reader.read_band(99)

    def test_factory_selects_landsat_reader(self, landsat_dir: Path) -> None:
        reader = open_image(landsat_dir)
        try:
            assert isinstance(reader, LandsatReader)
        finally:
            reader.close()

    def test_dir_without_bands(self, tmp_path: Path) -> None:
        product = tmp_path / "LC08_no_bands"
        product.mkdir()
        (product / "LC08_no_bands_MTL.txt").write_text(
            'SPACECRAFT_ID = "LANDSAT_8"\nEND\n', encoding="utf-8"
        )
        with pytest.raises(MetadataError):
            LandsatReader(product).open()

    def test_tar_cleanup_on_close(self, landsat_tar: Path) -> None:
        reader = LandsatReader(landsat_tar).open()
        workdir = reader._workdir
        assert workdir is not None and workdir.exists()
        reader.close()
        assert not workdir.exists()
        reader.close()  # idempotent
