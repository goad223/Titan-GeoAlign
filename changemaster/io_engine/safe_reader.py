"""Sentinel SAFE product reader: manifest parsing + JP2 band access.

A Sentinel-2 (or Sentinel-1) ``.SAFE`` product is a directory containing a
``manifest.safe`` XML index plus band image files. This reader parses the
manifest for product metadata, discovers the band files, and reads pixels
through the raster backend (rasterio) when JP2/TIFF decoding is required.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from changemaster.core.exceptions import BandError, MetadataError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.raster_reader import RasterReader, _try_import_rasterio

_BAND_PATTERN = re.compile(r"_(B[0-9]{1,2}[A-Z]?)(?:_\d+m)?\.(?:jp2|tiff?)$", re.IGNORECASE)
_DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def parse_manifest(manifest_path: Path) -> dict[str, str]:
    """Extract simple metadata fields from a ``manifest.safe`` XML file.

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        Dictionary with any of ``platform``, ``product_type`` and
        ``acquisition_datetime`` that could be discovered.

    Raises:
        MetadataError: If the XML is unreadable or malformed.
    """
    try:
        tree = ET.parse(manifest_path)
    except (ET.ParseError, OSError) as exc:
        raise MetadataError(
            f"Cannot parse manifest {manifest_path}: {exc}",
            f"تعذر تحليل ملف المانيفست {manifest_path}: {exc}",
        ) from exc
    info: dict[str, str] = {}
    root = tree.getroot()
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        text = (elem.text or "").strip()
        if not text:
            continue
        if tag in ("familyname", "platform") and "platform" not in info and len(text) < 64:
            info["platform"] = text
        elif tag in ("producttype", "product_type") and "product_type" not in info:
            info["product_type"] = text
        elif tag in ("startdatetime", "starttime", "acquisitionperiod"):
            match = _DATETIME_PATTERN.search(text)
            if match and "acquisition_datetime" not in info:
                info["acquisition_datetime"] = match.group(0)
        elif "acquisition_datetime" not in info and _DATETIME_PATTERN.fullmatch(text):
            info["acquisition_datetime"] = text
    return info


def find_band_files(safe_dir: Path) -> dict[str, Path]:
    """Locate band image files inside a SAFE directory.

    Args:
        safe_dir: Root of the ``.SAFE`` product.

    Returns:
        Mapping of band identifier (e.g. ``B04``, ``B8A``) to file path,
        sorted by band identifier. When the same band exists at several
        resolutions, the first discovered (highest resolution) is kept.
    """
    bands: dict[str, Path] = {}
    for candidate in sorted(safe_dir.rglob("*")):
        if not candidate.is_file():
            continue
        match = _BAND_PATTERN.search(candidate.name)
        if match:
            key = match.group(1).upper()
            bands.setdefault(key, candidate)
    return dict(sorted(bands.items()))


@ReaderRegistry.register
class SafeReader(BaseReader):
    """Reads Sentinel ``.SAFE`` product directories."""

    format_name = "Sentinel SAFE"
    extensions = (".safe",)
    priority = 60

    @classmethod
    def is_available(cls) -> bool:
        """Requires rasterio for JP2 band decoding."""
        return _try_import_rasterio() is not None

    @classmethod
    def required_package(cls) -> str | None:
        """rasterio is needed to decode JP2 band files."""
        return "rasterio"

    @classmethod
    def claims(cls, path: Path) -> bool:
        """Claim directories named ``*.SAFE`` that contain a manifest."""
        return (
            path.is_dir()
            and path.suffix.lower() == ".safe"
            and (path / "manifest.safe").exists()
        )

    def __init__(self, path: object) -> None:
        """Initialize the reader for a ``.SAFE`` directory."""
        super().__init__(path)  # type: ignore[arg-type]
        self._bands: dict[str, Path] = {}
        self._band_order: list[str] = []
        self._opened = False

    def open(self) -> "SafeReader":
        """Parse the manifest, discover bands, and build unified metadata.

        Raises:
            MetadataError: If the manifest is missing/invalid or no band
                files can be found.
        """
        if self._opened:
            return self
        manifest = self.path / "manifest.safe"
        if not manifest.exists():
            raise MetadataError(
                f"manifest.safe not found in {self.path}",
                f"ملف manifest.safe غير موجود في {self.path}",
            )
        info = parse_manifest(manifest)
        self._bands = find_band_files(self.path)
        if not self._bands:
            raise MetadataError(
                f"No band image files found in {self.path}",
                f"لا توجد ملفات نطاقات في {self.path}",
            )
        self._band_order = list(self._bands)
        first = self._bands[self._band_order[0]]
        with RasterReader(first) as reader:
            ref = reader.metadata
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=self.format_name,
            driver="safe+rasterio",
            width=ref.width,
            height=ref.height,
            band_count=len(self._band_order),
            dtype=ref.dtype,
            crs=ref.crs,
            transform=ref.transform,
            nodata=ref.nodata,
            band_names=list(self._band_order),
            sensor=info.get("platform"),
            acquisition_datetime=info.get("acquisition_datetime"),
            extra={
                "manifest": info,
                "band_files": {k: str(v) for k, v in self._bands.items()},
            },
        )
        self._opened = True
        return self

    def close(self) -> None:
        """Release internal state (band files are opened per read)."""
        self._opened = False

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read every discovered band, stacked as ``(bands, rows, cols)``.

        Bands with differing resolutions are read at their native size only
        when shapes match the reference band; otherwise a
        :class:`MetadataError` is raised so callers can use
        :meth:`read_band` per band instead.
        """
        if not self._opened:
            self.open()
        arrays = [self.read_band(i + 1, window) for i in range(len(self._band_order))]
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            raise MetadataError(
                f"Bands have mixed resolutions {sorted(shapes)}; read bands individually.",
                f"النطاقات بدقات مختلفة {sorted(shapes)}؛ اقرأ كل نطاق على حدة.",
            )
        return np.stack(arrays, axis=0)

    def read_band(self, band: int | str, window: Window | None = None) -> np.ndarray:
        """Read a single band by 1-based index or name (e.g. ``"B04"``).

        Raises:
            BandError: If the band index/name is invalid.
        """
        if not self._opened:
            self.open()
        if isinstance(band, str):
            key = band.upper()
            if key not in self._bands:
                raise BandError(band, self._band_order)
        else:
            if band < 1 or band > len(self._band_order):
                raise BandError(band, self._band_order)
            key = self._band_order[band - 1]
        with RasterReader(self._bands[key]) as reader:
            return reader.read_band(1, window)
