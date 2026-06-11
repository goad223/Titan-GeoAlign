"""HDF5 reader backed by h5py (lazy import).

Exposes 2-D and 3-D HDF5 datasets through the unified reader interface. When
the file contains several datasets the largest 2-D/3-D dataset is selected by
default; a specific dataset can be chosen with the ``dataset`` argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from changemaster.core.exceptions import BandError, MetadataError, ReaderError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata


def _try_import_h5py() -> Any | None:
    """Return the h5py module if installed, else ``None``."""
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError:
        return None


@ReaderRegistry.register
class HDFReader(BaseReader):
    """Reads 2-D/3-D datasets from HDF5 files via h5py."""

    format_name = "HDF5"
    extensions = (".h5", ".hdf", ".hdf5", ".he5")
    priority = 40

    @classmethod
    def is_available(cls) -> bool:
        """Available only when h5py is installed."""
        return _try_import_h5py() is not None

    @classmethod
    def required_package(cls) -> str | None:
        """h5py provides this reader's backend."""
        return "h5py"

    def __init__(self, path: object, dataset: str | None = None) -> None:
        """Initialize the reader.

        Args:
            path: HDF5 file path.
            dataset: Optional internal dataset name; auto-selected if omitted.
        """
        super().__init__(path)  # type: ignore[arg-type]
        self._file = None
        self._dataset_name: str | None = dataset
        self._dataset = None

    @staticmethod
    def list_datasets(path: str | Path) -> list[str]:
        """List the names of all 2-D/3-D datasets inside an HDF5 file.

        Args:
            path: HDF5 file path.

        Returns:
            Full internal paths of every dataset with 2 or 3 dimensions.
        """
        h5py = _try_import_h5py()
        if h5py is None:
            from changemaster.core.exceptions import MissingDependencyError

            raise MissingDependencyError("h5py", feature="reading HDF5 files")
        names: list[str] = []

        with h5py.File(str(path), "r") as handle:

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    names.append(name)

            handle.visititems(visitor)
        return names

    def open(self) -> "HDFReader":
        """Open the file, select a dataset and build unified metadata.

        Raises:
            ReaderError: When the file cannot be opened by h5py.
            MetadataError: When no usable 2-D/3-D dataset exists.
        """
        if self._file is not None:
            return self
        h5py = _try_import_h5py()
        assert h5py is not None, "claims() guarantees availability"
        try:
            self._file = h5py.File(str(self.path), "r")
        except OSError as exc:
            raise ReaderError(
                f"h5py failed to open {self.path}: {exc}",
                f"فشل h5py في فتح {self.path}: {exc}",
            ) from exc
        if self._dataset_name is None:
            candidates: list[tuple[int, str]] = []

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    candidates.append((int(np.prod(obj.shape)), name))

            self._file.visititems(visitor)
            if not candidates:
                self.close()
                raise MetadataError(
                    f"No 2-D/3-D dataset found in {self.path}",
                    f"لا توجد بيانات ثنائية/ثلاثية الأبعاد في {self.path}",
                )
            candidates.sort(reverse=True)
            self._dataset_name = candidates[0][1]
        if self._dataset_name not in self._file:
            available = self.list_datasets(self.path)
            self.close()
            raise MetadataError(
                f"Dataset '{self._dataset_name}' not in file. Available: {available}",
                f"البيانات '{self._dataset_name}' غير موجودة بالملف. المتاح: {available}",
            )
        self._dataset = self._file[self._dataset_name]
        shape = self._dataset.shape
        if len(shape) == 2:
            bands, height, width = 1, shape[0], shape[1]
        else:
            bands, height, width = shape[0], shape[1], shape[2]
        attrs = {k: _jsonable(v) for k, v in self._dataset.attrs.items()}
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=self.format_name,
            driver="h5py",
            width=int(width),
            height=int(height),
            band_count=int(bands),
            dtype=str(self._dataset.dtype),
            extra={"dataset": self._dataset_name, "attrs": attrs},
        )
        return self

    def close(self) -> None:
        """Close the HDF5 file (safe to call twice)."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read all bands shaped ``(bands, rows, cols)`` (windowed slices)."""
        if self._dataset is None:
            self.open()
        assert self._dataset is not None
        if window is None:
            data = np.asarray(self._dataset)
        else:
            col, row, width, height = window
            if self._dataset.ndim == 2:
                data = np.asarray(self._dataset[row : row + height, col : col + width])
            else:
                data = np.asarray(self._dataset[:, row : row + height, col : col + width])
        if data.ndim == 2:
            return data[np.newaxis, :, :]
        return data

    def read_band(self, band: int, window: Window | None = None) -> np.ndarray:
        """Read one 1-based band shaped ``(rows, cols)``.

        Raises:
            BandError: If the band index is out of range.
        """
        meta = self.metadata
        if band < 1 or band > meta.band_count:
            raise BandError(band, meta.band_count)
        return self.read(window)[band - 1]


def _jsonable(value: Any) -> Any:
    """Convert HDF attribute values (numpy/bytes) into JSON-safe Python."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
