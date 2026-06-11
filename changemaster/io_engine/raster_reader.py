"""GeoTIFF/BigTIFF/JPEG2000/ENVI reader backed by rasterio (lazy import).

``rasterio`` is only imported when a file is actually opened, so the module
imports cleanly even without GDAL installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from changemaster.core.exceptions import BandError, ReaderError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata


def _try_import_rasterio() -> Any | None:
    """Return the rasterio module if installed, else ``None``."""
    try:
        import rasterio  # noqa: PLC0415

        return rasterio
    except ImportError:
        return None


@ReaderRegistry.register
class RasterReader(BaseReader):
    """Reads georeferenced rasters (GeoTIFF, BigTIFF, JP2, ENVI) via rasterio."""

    format_name = "GeoTIFF/JPEG2000/ENVI"
    extensions = (".tif", ".tiff", ".gtiff", ".jp2", ".img", ".dat", ".hdr", ".vrt")
    priority = 50

    @classmethod
    def is_available(cls) -> bool:
        """Available only when rasterio (and GDAL) are installed."""
        return _try_import_rasterio() is not None

    @classmethod
    def required_package(cls) -> str | None:
        """rasterio provides this reader's backend."""
        return "rasterio"

    def __init__(self, path: object) -> None:
        """Initialize the reader for ``path`` (validated by the base class)."""
        super().__init__(path)  # type: ignore[arg-type]
        self._dataset = None

    def open(self) -> "RasterReader":
        """Open the dataset with rasterio and build unified metadata.

        Raises:
            ReaderError: When rasterio/GDAL cannot open the file.
        """
        if self._dataset is not None:
            return self
        rasterio = _try_import_rasterio()
        assert rasterio is not None, "claims() guarantees availability"
        try:
            self._dataset = rasterio.open(self.path)
        except Exception as exc:  # noqa: BLE001 - rasterio raises many types
            raise ReaderError(
                f"rasterio failed to open {self.path}: {exc}",
                f"فشل rasterio في فتح {self.path}: {exc}",
            ) from exc
        ds = self._dataset
        transform = None
        if ds.transform is not None and not ds.transform.is_identity:
            t = ds.transform
            transform = (t.a, t.b, t.c, t.d, t.e, t.f)
        crs = str(ds.crs) if ds.crs else None
        descriptions = [d for d in (ds.descriptions or ()) if d]
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=ds.driver or self.format_name,
            driver="rasterio",
            width=int(ds.width),
            height=int(ds.height),
            band_count=int(ds.count),
            dtype=str(ds.dtypes[0]),
            crs=crs,
            transform=transform,
            nodata=ds.nodata,
            band_names=descriptions,
            extra={"tags": dict(ds.tags())},
        )
        return self

    def close(self) -> None:
        """Close the rasterio dataset (safe to call twice)."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def _window(self, window: Window | None) -> Any | None:
        """Convert our window tuple to a rasterio ``Window`` object."""
        if window is None:
            return None
        from rasterio.windows import Window as RioWindow  # noqa: PLC0415

        col, row, width, height = window
        return RioWindow(col, row, width, height)

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read all bands shaped ``(bands, rows, cols)``."""
        if self._dataset is None:
            self.open()
        assert self._dataset is not None
        return self._dataset.read(window=self._window(window))

    def read_band(self, band: int, window: Window | None = None) -> np.ndarray:
        """Read one 1-based band shaped ``(rows, cols)``.

        Raises:
            BandError: If the band index is out of range.
        """
        if self._dataset is None:
            self.open()
        assert self._dataset is not None
        if band < 1 or band > self._dataset.count:
            raise BandError(band, self._dataset.count)
        return self._dataset.read(band, window=self._window(window))
