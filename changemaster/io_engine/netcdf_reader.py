"""NetCDF reader backed by netCDF4 (lazy import).

Exposes 2-D and 3-D NetCDF variables through the unified reader interface.
The largest suitable variable is auto-selected when none is specified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from changemaster.core.exceptions import BandError, MetadataError, ReaderError
from changemaster.io_engine.base_reader import BaseReader, ReaderRegistry, Window
from changemaster.io_engine.metadata import ImageMetadata


def _try_import_netcdf4() -> Any | None:
    """Return the netCDF4 module if installed, else ``None``."""
    try:
        import netCDF4  # noqa: PLC0415

        return netCDF4
    except ImportError:
        return None


@ReaderRegistry.register
class NetCDFReader(BaseReader):
    """Reads 2-D/3-D variables from NetCDF files via netCDF4."""

    format_name = "NetCDF"
    extensions = (".nc", ".nc4", ".cdf")
    priority = 40

    @classmethod
    def is_available(cls) -> bool:
        """Available only when netCDF4 is installed."""
        return _try_import_netcdf4() is not None

    @classmethod
    def required_package(cls) -> str | None:
        """netCDF4 provides this reader's backend."""
        return "netCDF4"

    def __init__(self, path: object, variable: str | None = None) -> None:
        """Initialize the reader.

        Args:
            path: NetCDF file path.
            variable: Optional variable name; auto-selected if omitted.
        """
        super().__init__(path)  # type: ignore[arg-type]
        self._file = None
        self._variable_name: str | None = variable
        self._variable = None

    @staticmethod
    def list_variables(path: str | Path) -> list[str]:
        """List 2-D/3-D variable names inside a NetCDF file.

        Args:
            path: NetCDF file path.

        Returns:
            Names of variables with 2 or 3 dimensions.
        """
        netCDF4 = _try_import_netcdf4()
        if netCDF4 is None:
            from changemaster.core.exceptions import MissingDependencyError

            raise MissingDependencyError("netCDF4", feature="reading NetCDF files")
        with netCDF4.Dataset(str(path), "r") as handle:
            return [name for name, var in handle.variables.items() if var.ndim in (2, 3)]

    def open(self) -> "NetCDFReader":
        """Open the file, select a variable and build unified metadata.

        Raises:
            ReaderError: When the file cannot be opened by netCDF4.
            MetadataError: When no usable 2-D/3-D variable exists.
        """
        if self._file is not None:
            return self
        netCDF4 = _try_import_netcdf4()
        assert netCDF4 is not None, "claims() guarantees availability"
        try:
            self._file = netCDF4.Dataset(str(self.path), "r")
        except OSError as exc:
            raise ReaderError(
                f"netCDF4 failed to open {self.path}: {exc}",
                f"فشل netCDF4 في فتح {self.path}: {exc}",
            ) from exc
        if self._variable_name is None:
            candidates = [
                (int(np.prod(var.shape)), name)
                for name, var in self._file.variables.items()
                if var.ndim in (2, 3)
            ]
            if not candidates:
                self.close()
                raise MetadataError(
                    f"No 2-D/3-D variable found in {self.path}",
                    f"لا يوجد متغير ثنائي/ثلاثي الأبعاد في {self.path}",
                )
            candidates.sort(reverse=True)
            self._variable_name = candidates[0][1]
        if self._variable_name not in self._file.variables:
            available = list(self._file.variables)
            self.close()
            raise MetadataError(
                f"Variable '{self._variable_name}' not in file. Available: {available}",
                f"المتغير '{self._variable_name}' غير موجود بالملف. المتاح: {available}",
            )
        self._variable = self._file.variables[self._variable_name]
        shape = self._variable.shape
        if len(shape) == 2:
            bands, height, width = 1, shape[0], shape[1]
        else:
            bands, height, width = shape[0], shape[1], shape[2]
        attrs = {k: _jsonable(self._variable.getncattr(k)) for k in self._variable.ncattrs()}
        global_attrs = {k: _jsonable(self._file.getncattr(k)) for k in self._file.ncattrs()}
        self._metadata = ImageMetadata(
            path=self.path,
            format_name=self.format_name,
            driver="netCDF4",
            width=int(width),
            height=int(height),
            band_count=int(bands),
            dtype=str(self._variable.dtype),
            extra={
                "variable": self._variable_name,
                "attrs": attrs,
                "global_attrs": global_attrs,
            },
        )
        return self

    def close(self) -> None:
        """Close the NetCDF file (safe to call twice)."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._variable = None

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read all bands shaped ``(bands, rows, cols)`` (windowed slices)."""
        if self._variable is None:
            self.open()
        assert self._variable is not None
        if window is None:
            data = np.asarray(self._variable[:])
        else:
            col, row, width, height = window
            if self._variable.ndim == 2:
                data = np.asarray(self._variable[row : row + height, col : col + width])
            else:
                data = np.asarray(self._variable[:, row : row + height, col : col + width])
        data = np.ma.filled(data) if isinstance(data, np.ma.MaskedArray) else data
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
    """Convert NetCDF attribute values into JSON-safe Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
