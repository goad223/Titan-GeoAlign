"""Tests for tiled access to large rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import TileAccessError
from changemaster.io_engine.simple_reader import SimpleImageReader
from changemaster.io_engine.tiled_access import (
    TiledReader,
    compute_tile_grid,
    estimate_tile_memory_mb,
)


def test_grid_exact_division() -> None:
    tiles = compute_tile_grid(100, 50, 25)
    assert len(tiles) == 4 * 2
    assert tiles[0].window == (0, 0, 25, 25)
    assert tiles[-1].window == (75, 25, 25, 25)
    assert all(t.width == 25 and t.height == 25 for t in tiles)


def test_grid_edge_tiles_smaller() -> None:
    tiles = compute_tile_grid(110, 55, 50)
    assert len(tiles) == 3 * 2
    last = tiles[-1]
    assert last.window == (100, 50, 10, 5)


def test_grid_single_tile_when_image_smaller() -> None:
    tiles = compute_tile_grid(30, 20, 100)
    assert len(tiles) == 1
    assert tiles[0].window == (0, 0, 30, 20)


def test_grid_huge_image_dimensions() -> None:
    # 100,000 x 100,000 px image: only the grid is computed, never pixels.
    tiles = compute_tile_grid(100_000, 100_000, 4096)
    expected = ((100_000 + 4095) // 4096) ** 2
    assert len(tiles) == expected
    assert tiles[0].window == (0, 0, 4096, 4096)
    total = sum(t.width * t.height for t in tiles)
    assert total == 100_000 * 100_000


def test_grid_with_overlap_clamped_at_edges() -> None:
    tiles = compute_tile_grid(100, 100, 50, overlap=10)
    first = tiles[0]
    assert first.window == (0, 0, 60, 60)  # clamped left/top, padded right/bottom
    inner = tiles[3]  # row 1, col 1
    assert inner.window == (40, 40, 60, 60)


def test_grid_row_major_indexing() -> None:
    tiles = compute_tile_grid(100, 100, 50)
    assert [(t.row, t.col) for t in tiles] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [t.index for t in tiles] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("width", "height", "tile_size", "overlap"),
    [(0, 100, 32, 0), (100, -1, 32, 0), (100, 100, 0, 0), (100, 100, 32, -1), (100, 100, 32, 32)],
)
def test_grid_invalid_arguments(width: int, height: int, tile_size: int, overlap: int) -> None:
    with pytest.raises(TileAccessError):
        compute_tile_grid(width, height, tile_size, overlap)


def test_tiled_reader_reassembles_image(png_file: Path, rgb_array: np.ndarray) -> None:
    with SimpleImageReader(png_file) as reader:
        tiled = TiledReader(reader, tile_size=16)
        assert tiled.tile_count == len(tiled)
        result = np.zeros_like(rgb_array)
        for tile, data in tiled:
            col, row, width, height = tile.window
            result[:, row : row + height, col : col + width] = data
        np.testing.assert_array_equal(result, rgb_array)


def test_tiled_reader_read_by_index(png_file: Path, rgb_array: np.ndarray) -> None:
    with SimpleImageReader(png_file) as reader:
        tiled = TiledReader(reader, tile_size=16)
        first = tiled.read_tile(0)
        np.testing.assert_array_equal(first, rgb_array[:, :16, :16])
        with pytest.raises(TileAccessError):
            tiled.read_tile(9999)
        with pytest.raises(TileAccessError):
            tiled.read_tile(-1)


def test_tiled_reader_with_rasterio(geotiff_file: Path, gray_array: np.ndarray) -> None:
    pytest.importorskip("rasterio")
    from changemaster.io_engine.raster_reader import RasterReader

    with RasterReader(geotiff_file) as reader:
        tiled = TiledReader(reader, tile_size=17)
        result = np.zeros((1, *gray_array.shape), dtype=gray_array.dtype)
        for tile, data in tiled:
            col, row, width, height = tile.window
            result[:, row : row + height, col : col + width] = data
        np.testing.assert_array_equal(result[0], gray_array)


def test_tiles_property_returns_copy(png_file: Path) -> None:
    with SimpleImageReader(png_file) as reader:
        tiled = TiledReader(reader, tile_size=16)
        tiles = tiled.tiles
        tiles.clear()
        assert tiled.tile_count > 0


def test_estimate_tile_memory() -> None:
    mb = estimate_tile_memory_mb(1024, 3, "uint16")
    assert mb == pytest.approx(1024 * 1024 * 3 * 2 / (1024 * 1024))
    padded = estimate_tile_memory_mb(1024, 1, "uint8", overlap=512)
    assert padded == pytest.approx(2048 * 2048 / (1024 * 1024))
