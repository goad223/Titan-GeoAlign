"""Tests for the Pillow-backed simple reader and the registry/factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import (
    BandError,
    FileAccessError,
    ReaderError,
    UnsupportedFormatError,
)
from changemaster.io_engine.base_reader import (
    ReaderRegistry,
    iter_supported_extensions,
    open_image,
)
from changemaster.io_engine.simple_reader import SimpleImageReader


def test_simple_reader_always_available() -> None:
    assert SimpleImageReader.is_available()
    assert SimpleImageReader.required_package() == "Pillow"


def test_read_png_full(png_file: Path, rgb_array: np.ndarray) -> None:
    with SimpleImageReader(png_file) as reader:
        meta = reader.metadata
        assert meta.width == rgb_array.shape[2]
        assert meta.height == rgb_array.shape[1]
        assert meta.band_count == 3
        assert meta.dtype == "uint8"
        assert meta.driver == "pillow"
        data = reader.read()
        assert data.shape == rgb_array.shape
        np.testing.assert_array_equal(data, rgb_array)


def test_read_window(png_file: Path, rgb_array: np.ndarray) -> None:
    with SimpleImageReader(png_file) as reader:
        window = (5, 3, 10, 8)
        data = reader.read(window=window)
        assert data.shape == (3, 8, 10)
        np.testing.assert_array_equal(data, rgb_array[:, 3:11, 5:15])


def test_read_band_and_invalid_band(png_file: Path, rgb_array: np.ndarray) -> None:
    with SimpleImageReader(png_file) as reader:
        band = reader.read_band(2)
        np.testing.assert_array_equal(band, rgb_array[1])
        with pytest.raises(BandError):
            reader.read_band(0)
        with pytest.raises(BandError):
            reader.read_band(4)


def test_read_jpeg(jpeg_file: Path) -> None:
    with SimpleImageReader(jpeg_file) as reader:
        assert reader.metadata.format_name == "JPEG"
        assert reader.read().shape[0] == 3


def test_grayscale_png(tmp_path: Path) -> None:
    from PIL import Image

    gray = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)
    path = tmp_path / "gray.png"
    Image.fromarray(gray, mode="L").save(path)
    with SimpleImageReader(path) as reader:
        assert reader.metadata.band_count == 1
        np.testing.assert_array_equal(reader.read_band(1), gray)


def test_missing_file_raises() -> None:
    with pytest.raises(FileAccessError):
        SimpleImageReader("/no/such/file.png")


def test_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(ReaderError):
        SimpleImageReader(bad).open()


def test_close_is_idempotent(png_file: Path) -> None:
    reader = SimpleImageReader(png_file).open()
    reader.close()
    reader.close()


def test_factory_selects_simple_reader(png_file: Path) -> None:
    reader = open_image(png_file)
    try:
        assert isinstance(reader, SimpleImageReader)
    finally:
        reader.close()


def test_factory_unknown_extension(tmp_path: Path) -> None:
    odd = tmp_path / "file.xyz"
    odd.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        open_image(odd)


def test_factory_missing_path() -> None:
    with pytest.raises(FileAccessError):
        open_image("/no/such/path.png")


def test_registry_table_and_extensions() -> None:
    rows = ReaderRegistry.format_table()
    formats = {row["format"] for row in rows}
    assert "PNG/JPEG/BMP" in formats
    for row in rows:
        assert set(row) == {"format", "extensions", "available", "requires"}
    assert ".png" in set(iter_supported_extensions())


def test_registry_priority_order() -> None:
    readers = ReaderRegistry.all_readers()
    priorities = [r.priority for r in readers]
    assert priorities == sorted(priorities, reverse=True)
