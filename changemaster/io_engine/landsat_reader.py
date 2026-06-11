"""Landsat product reader: MTL.txt parsing + band file access.

Supports Landsat Collection products delivered either as a directory of
``*_B{n}.TIF`` files with an ``*_MTL.txt`` metadata file, or as a ``.tar`` /
``.tar.gz`` archive of the same layout (extracted to a temp dir on open).
"""

from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from changemaster.core.exceptions import BandError, MetadataError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.raster_reader import RasterReader, _try_import_rasterio

_BAND_PATTERN = re.compile(r"_B(\d{1,2})\.TIF$", re.IGNORECASE)


def parse_mtl(mtl_path: Path) -> dict[str, str]:
    """Parse a Landsat ``MTL.txt`` metadata file into a flat dictionary.

    Args:
        mtl_path: Path of the ``*_MTL.txt`` file.

    Returns:
        Mapping of MTL keys to values (quotes stripped). Group markers
        (``GROUP``/``END_GROUP``/``END``) are skipped.

    Raises:
        MetadataError: If the file cannot be read.
    """
    try:
        text = mtl_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MetadataError(
            f"Cannot read MTL file {mtl_path}: {exc}",
            f"تعذر قراءة ملف MTL {mtl_path}: {exc}",
        ) from exc
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "END" or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"')
        if key in ("GROUP", "END_GROUP"):
            continue
        result[key] = value
    return result


def find_band_files(product_dir: Path) -> dict[int, Path]:
    """Locate ``*_B{n}.TIF`` band files inside a Landsat product directory.

    Args:
        product_dir: Directory containing the product files.

    Returns:
        Mapping of band number to file path, sorted by band number.
    """
    bands: dict[int, Path] = {}
    for candidate in sorted(product_dir.iterdir()):
        if not candidate.is_file():
            continue
        match = _BAND_PATTERN.search(candidate.name)
        if match:
            bands[int(match.group(1))] = candidate
    return dict(sorted(bands.items()))


def _find_mtl(product_dir: Path) -> Path | None:
    """Return the first ``*_MTL.txt`` file in ``product_dir``, if any."""
    matches = sorted(product_dir.glob("*_MTL.txt")) or sorted(product_dir.glob("*MTL.txt"))
    return matches[0] if matches else None


@ReaderRegistry.register
class LandsatReader(BaseReader):
    """Reads Landsat products from a directory or a tar archive."""

    format_name = "Landsat"
    extensions = (".tar",)
    priority = 60

    @classmethod
    def is_available(cls) -> bool:
        """Requires rasterio to decode GeoTIFF band files."""
        return _try_import_rasterio() is not None

    @classmethod
    def required_package(cls) -> str | None:
        """rasterio is needed to decode band TIFs."""
        return "rasterio"

    @classmethod
    def claims(cls, path: Path) -> bool:
        """Claim Landsat directories (with an MTL file) and tar archives."""
        if path.is_dir():
            return _find_mtl(path) is not None
        name = path.name.lower()
        if name.endswith((".tar", ".tar.gz", ".tgz")):
            return bool(re.match(r"^l[a-z]\d{2}_", name)) or "mtl" in name or _probe_tar(path)
        return False

    def __init__(self, path: object) -> None:
        """Initialize the reader for a product directory or tar archive."""
        super().__init__(path)  # type: ignore[arg-type]
        self._workdir: Path | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._bands: dict[int, Path] = {}
        self._band_order: list[int] = []
        self._mtl: dict[str, str] = {}
        self._opened = False

    def open(self) -> "LandsatReader":
        """Extract (if tar), parse MTL, discover bands, build metadata.

        Raises:
            MetadataError: If no MTL file or no band files are found.
        """
        if self._opened:
            return self
        if self.path.is_dir():
            self._workdir = self.path
        else:
            self._tempdir = tempfile.TemporaryDirectory(prefix="changemaster_landsat_")
            target = Path(self._tempdir.name)
            with tarfile.open(self.path) as archive:
                archive.extractall(target, filter="data")
            self._workdir = target
        mtl_path = _find_mtl(self._workdir)
        if mtl_path is None:
            self.close()
            raise MetadataError(
                f"No *_MTL.txt metadata file found in {self.path}",
                f"لا يوجد ملف بيانات وصفية *_MTL.txt في {self.path}",
            )
        self._mtl = parse_mtl(mtl_path)
        self._bands = find_band_files(self._workdir)
        if not self._bands:
            self.close()
            raise MetadataError(
                f"No band files (*_B*.TIF) found in {self.path}",
                f"لا توجد ملفات نطاقات (*_B*.TIF) في {self.path}",
            )
        self._band_order = list(self._bands)
        first = self._bands[self._band_order[0]]
        with RasterReader(first) as reader:
            ref = reader.metadata
        spacecraft = self._mtl.get("SPACECRAFT_ID", "")
        date = self._mtl.get("DATE_ACQUIRED", "")
        time = self._mtl.get("SCENE_CENTER_TIME", "").strip('"').split(".")[0]
        acquired = f"{date}T{time}" if date and time else (date or None)
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=self.format_name,
            driver="landsat+rasterio",
            width=ref.width,
            height=ref.height,
            band_count=len(self._band_order),
            dtype=ref.dtype,
            crs=ref.crs,
            transform=ref.transform,
            nodata=ref.nodata,
            band_names=[f"B{n}" for n in self._band_order],
            sensor=spacecraft or None,
            acquisition_datetime=acquired,
            extra={
                "mtl": self._mtl,
                "band_files": {str(k): str(v) for k, v in self._bands.items()},
            },
        )
        self._opened = True
        return self

    def close(self) -> None:
        """Remove any temporary extraction directory (safe to call twice)."""
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        self._workdir = None
        self._opened = False

    @property
    def mtl(self) -> dict[str, str]:
        """Parsed MTL metadata dictionary (opens the product if needed)."""
        if not self._opened:
            self.open()
        return dict(self._mtl)

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read every band, stacked as ``(bands, rows, cols)``.

        Raises:
            MetadataError: If bands have mismatched shapes (mixed resolution).
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

    def read_band(self, band: int, window: Window | None = None) -> np.ndarray:
        """Read one band by 1-based position in the discovered band list.

        Raises:
            BandError: If the index is out of range.
        """
        if not self._opened:
            self.open()
        if band < 1 or band > len(self._band_order):
            raise BandError(band, [f"B{n}" for n in self._band_order])
        number = self._band_order[band - 1]
        with RasterReader(self._bands[number]) as reader:
            return reader.read_band(1, window)


def _probe_tar(path: Path) -> bool:
    """Cheaply check whether a tar archive contains an MTL file."""
    try:
        with tarfile.open(path) as archive:
            for member in archive:
                if member.name.lower().endswith("_mtl.txt"):
                    return True
                # Avoid scanning enormous archives exhaustively.
                if archive.members and len(archive.members) > 200:
                    break
    except (tarfile.TarError, OSError):
        return False
    return False
