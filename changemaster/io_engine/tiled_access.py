"""Tiled (windowed) access for giant rasters (e.g. 100,000 x 100,000 px).

Provides a tile grid calculator and a :class:`TiledReader` that iterates any
:class:`~changemaster.io_engine.base_reader.BaseReader` in memory-bounded
chunks, enabling processing of images far larger than available RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from changemaster.core.exceptions import TileAccessError
from changemaster.io_engine.base_reader import BaseReader, Window


@dataclass(frozen=True)
class Tile:
    """A single tile within a tiling grid.

    Attributes:
        index: Sequential tile number (row-major, starting at 0).
        row: Tile row in the grid.
        col: Tile column in the grid.
        window: Pixel window ``(col_off, row_off, width, height)``.
    """

    index: int
    row: int
    col: int
    window: Window

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.window[2]

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.window[3]


def compute_tile_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int = 0,
) -> list[Tile]:
    """Compute a row-major tile grid covering a ``width x height`` image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Nominal tile edge in pixels (edge tiles may be smaller).
        overlap: Pixels of overlap added on each side (clamped to image).

    Returns:
        All tiles covering the image, in row-major order.

    Raises:
        TileAccessError: For non-positive sizes or invalid overlap.
    """
    if width <= 0 or height <= 0:
        raise TileAccessError(
            f"Image dimensions must be positive, got {width}x{height}.",
            f"أبعاد الصورة يجب أن تكون موجبة، الوارد {width}x{height}.",
        )
    if tile_size <= 0:
        raise TileAccessError(
            f"Tile size must be positive, got {tile_size}.",
            f"حجم البلاطة يجب أن يكون موجباً، الوارد {tile_size}.",
        )
    if overlap < 0 or overlap >= tile_size:
        raise TileAccessError(
            f"Overlap must satisfy 0 <= overlap < tile_size, got {overlap}.",
            f"التداخل يجب أن يحقق 0 <= التداخل < حجم البلاطة، الوارد {overlap}.",
        )
    tiles: list[Tile] = []
    index = 0
    n_rows = (height + tile_size - 1) // tile_size
    n_cols = (width + tile_size - 1) // tile_size
    for tile_row in range(n_rows):
        for tile_col in range(n_cols):
            col_off = tile_col * tile_size - overlap
            row_off = tile_row * tile_size - overlap
            col_start = max(0, col_off)
            row_start = max(0, row_off)
            col_end = min(width, tile_col * tile_size + tile_size + overlap)
            row_end = min(height, tile_row * tile_size + tile_size + overlap)
            tiles.append(
                Tile(
                    index=index,
                    row=tile_row,
                    col=tile_col,
                    window=(col_start, row_start, col_end - col_start, row_end - row_start),
                )
            )
            index += 1
    return tiles


class TiledReader:
    """Iterates a reader's pixels tile-by-tile within a fixed memory budget."""

    def __init__(
        self,
        reader: BaseReader,
        tile_size: int = 1024,
        overlap: int = 0,
    ) -> None:
        """Bind a tiling strategy to an opened reader.

        Args:
            reader: An opened :class:`BaseReader` instance.
            tile_size: Nominal tile edge in pixels.
            overlap: Pixels of overlap between adjacent tiles.

        Raises:
            TileAccessError: For invalid tile size or overlap.
        """
        self._reader = reader
        meta = reader.metadata
        self._tiles = compute_tile_grid(meta.width, meta.height, tile_size, overlap)
        self.tile_size = tile_size
        self.overlap = overlap

    @property
    def tiles(self) -> list[Tile]:
        """The full tile grid in row-major order."""
        return list(self._tiles)

    @property
    def tile_count(self) -> int:
        """Total number of tiles in the grid."""
        return len(self._tiles)

    def read_tile(self, tile: Tile | int) -> np.ndarray:
        """Read the pixels of one tile as ``(bands, rows, cols)``.

        Args:
            tile: A :class:`Tile` or its sequential index.

        Returns:
            Pixel data for the tile's window.

        Raises:
            TileAccessError: If the tile index is out of range.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= len(self._tiles):
                raise TileAccessError(
                    f"Tile index {tile} out of range (0..{len(self._tiles) - 1}).",
                    f"رقم البلاطة {tile} خارج النطاق (0..{len(self._tiles) - 1}).",
                )
            tile = self._tiles[tile]
        return self._reader.read(window=tile.window)

    def __iter__(self) -> Iterator[tuple[Tile, np.ndarray]]:
        """Yield ``(tile, pixels)`` pairs for every tile in row-major order."""
        for tile in self._tiles:
            yield tile, self.read_tile(tile)

    def __len__(self) -> int:
        """Number of tiles (so ``len(tiled_reader)`` works)."""
        return len(self._tiles)


def estimate_tile_memory_mb(
    tile_size: int,
    band_count: int,
    dtype: str,
    overlap: int = 0,
) -> float:
    """Estimate the memory one tile occupies, in megabytes.

    Args:
        tile_size: Nominal tile edge in pixels.
        band_count: Number of bands read per tile.
        dtype: NumPy dtype name of the pixel data.
        overlap: Overlap added on each side of the tile.

    Returns:
        Estimated megabytes for a fully padded tile.
    """
    edge = tile_size + 2 * overlap
    itemsize = np.dtype(dtype).itemsize
    return (edge * edge * band_count * itemsize) / (1024 * 1024)
