"""PNG/JPEG/BMP reader backed by Pillow.

This reader has no heavy dependencies and is always available, guaranteeing
the application can open common images even on a minimal installation.
"""

from __future__ import annotations

import numpy as np

from changemaster.core.exceptions import BandError, ReaderError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata


@ReaderRegistry.register
class SimpleImageReader(BaseReader):
    """Reads PNG, JPEG, BMP and TIFF-lite images through Pillow."""

    format_name = "PNG/JPEG/BMP"
    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    priority = 10

    @classmethod
    def is_available(cls) -> bool:
        """Pillow is a hard dependency, so this reader is always available."""
        try:
            import PIL  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    @classmethod
    def required_package(cls) -> str | None:
        """Pillow provides this reader's backend."""
        return "Pillow"

    def __init__(self, path: object) -> None:
        """Initialize the reader for ``path`` (validated by the base class)."""
        super().__init__(path)  # type: ignore[arg-type]
        self._image = None

    def open(self) -> "SimpleImageReader":
        """Open the image with Pillow and build unified metadata.

        Raises:
            ReaderError: When Pillow cannot decode the file.
        """
        if self._image is not None:
            return self
        from PIL import Image  # noqa: PLC0415

        try:
            self._image = Image.open(self.path)
            self._image.load()
        except Exception as exc:  # noqa: BLE001 - Pillow raises many types
            raise ReaderError(
                f"Pillow failed to open {self.path}: {exc}",
                f"فشل Pillow في فتح {self.path}: {exc}",
            ) from exc
        sample = np.asarray(self._image)
        bands = 1 if sample.ndim == 2 else int(sample.shape[2])
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=self._image.format or self.format_name,
            driver="pillow",
            width=int(self._image.width),
            height=int(self._image.height),
            band_count=bands,
            dtype=str(sample.dtype),
            band_names=list(self._image.getbands()),
            extra={"mode": self._image.mode},
        )
        return self

    def close(self) -> None:
        """Close the underlying Pillow image (safe to call twice)."""
        if self._image is not None:
            self._image.close()
            self._image = None

    def _array(self) -> np.ndarray:
        """Return the full image as a ``(bands, rows, cols)`` array."""
        if self._image is None:
            self.open()
        data = np.asarray(self._image)
        if data.ndim == 2:
            return data[np.newaxis, :, :]
        return np.transpose(data, (2, 0, 1))

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read all bands, optionally cropped to ``window``."""
        data = self._array()
        if window is None:
            return data
        col, row, width, height = window
        return data[:, row : row + height, col : col + width]

    def read_band(self, band: int, window: Window | None = None) -> np.ndarray:
        """Read one 1-based band, optionally cropped to ``window``.

        Raises:
            BandError: If the band index is out of range.
        """
        data = self.read(window)
        if band < 1 or band > data.shape[0]:
            raise BandError(band, data.shape[0])
        return data[band - 1]
