"""Abstract reader interface plus the reader registry/factory.

Every concrete reader registers itself with :class:`ReaderRegistry`. The
factory function :func:`open_image` picks the best available reader for a
path, skipping readers whose optional dependencies are missing — so the
application keeps working with whatever formats the environment supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

import numpy as np

from changemaster.core.exceptions import (
    FileAccessError,
    UnsupportedFormatError,
)
from changemaster.io_engine.metadata import ImageMetadata

Window = tuple[int, int, int, int]
"""A pixel window expressed as ``(col_off, row_off, width, height)``."""


class BaseReader(ABC):
    """Abstract base class every image reader must implement.

    Readers are context managers::

        with SomeReader(path) as reader:
            meta = reader.metadata
            data = reader.read()

    Class attributes:
        format_name: Short identifier shown to users (e.g. ``GeoTIFF``).
        extensions: Lower-case file extensions this reader claims.
        priority: Larger numbers are tried first when extensions collide.
    """

    format_name: str = "unknown"
    extensions: tuple[str, ...] = ()
    priority: int = 0

    def __init__(self, path: str | Path) -> None:
        """Store and validate the source path.

        Args:
            path: File or product directory to read.

        Raises:
            FileAccessError: If the path does not exist.
        """
        self.path: Path = Path(path)
        if not self.path.exists():
            raise FileAccessError(str(self.path))
        self._metadata: ImageMetadata | None = None

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this reader's backend dependencies are installed."""

    @classmethod
    def required_package(cls) -> str | None:
        """pip package name needed by this reader (``None`` if stdlib-only)."""
        return None

    @classmethod
    def claims(cls, path: Path) -> bool:
        """Whether this reader recognizes ``path`` (default: by extension).

        Args:
            path: Candidate file or directory.

        Returns:
            ``True`` when the reader should be tried for this path.
        """
        return path.suffix.lower() in cls.extensions

    @abstractmethod
    def open(self) -> "BaseReader":
        """Open the underlying resource and populate metadata."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource (idempotent)."""

    @property
    def metadata(self) -> ImageMetadata:
        """Unified metadata; the reader must be opened first."""
        if self._metadata is None:
            self.open()
        assert self._metadata is not None
        return self._metadata

    @abstractmethod
    def read(self, window: Window | None = None) -> np.ndarray:
        """Read pixel data as an array shaped ``(bands, rows, cols)``.

        Args:
            window: Optional ``(col_off, row_off, width, height)`` sub-window;
                the full image is read when omitted.

        Returns:
            Pixel data for all bands within the window.
        """

    @abstractmethod
    def read_band(self, band: int, window: Window | None = None) -> np.ndarray:
        """Read a single band (1-based index) shaped ``(rows, cols)``.

        Args:
            band: 1-based band index.
            window: Optional sub-window, same convention as :meth:`read`.

        Returns:
            Pixel data for the requested band.
        """

    def __enter__(self) -> "BaseReader":
        """Open the reader when entering a ``with`` block."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the reader when leaving a ``with`` block."""
        self.close()


class ReaderRegistry:
    """Global registry mapping formats to reader classes."""

    _readers: list[type[BaseReader]] = []

    @classmethod
    def register(cls, reader_cls: type[BaseReader]) -> type[BaseReader]:
        """Class decorator that adds a reader to the registry.

        Args:
            reader_cls: Concrete :class:`BaseReader` subclass.

        Returns:
            The same class, unmodified (decorator-friendly).
        """
        if reader_cls not in cls._readers:
            cls._readers.append(reader_cls)
            cls._readers.sort(key=lambda r: -r.priority)
        return reader_cls

    @classmethod
    def all_readers(cls) -> list[type[BaseReader]]:
        """All registered readers ordered by descending priority."""
        return list(cls._readers)

    @classmethod
    def available_readers(cls) -> list[type[BaseReader]]:
        """Registered readers whose backend dependencies are installed."""
        return [r for r in cls._readers if r.is_available()]

    @classmethod
    def find_reader(cls, path: str | Path) -> type[BaseReader]:
        """Find the highest-priority available reader claiming ``path``.

        Args:
            path: File or product directory.

        Returns:
            A reader class ready to be instantiated.

        Raises:
            FileAccessError: If the path does not exist.
            UnsupportedFormatError: If no available reader claims the path.
            MissingDependencyError: If only unavailable readers claim it.
        """
        target = Path(path)
        if not target.exists():
            raise FileAccessError(str(target))
        claimed_unavailable: list[type[BaseReader]] = []
        for reader_cls in cls._readers:
            if reader_cls.claims(target):
                if reader_cls.is_available():
                    return reader_cls
                claimed_unavailable.append(reader_cls)
        if claimed_unavailable:
            from changemaster.core.exceptions import MissingDependencyError

            first = claimed_unavailable[0]
            raise MissingDependencyError(
                first.required_package() or "unknown",
                feature=f"reading {first.format_name} files",
            )
        raise UnsupportedFormatError(str(target))

    @classmethod
    def format_table(cls) -> list[dict[str, Any]]:
        """Tabular description of every registered format and its status.

        Returns:
            One dict per reader with keys ``format``, ``extensions``,
            ``available`` and ``requires``.
        """
        rows: list[dict[str, Any]] = []
        for reader_cls in cls._readers:
            rows.append(
                {
                    "format": reader_cls.format_name,
                    "extensions": ", ".join(reader_cls.extensions) or "(pattern-based)",
                    "available": reader_cls.is_available(),
                    "requires": reader_cls.required_package() or "built-in",
                }
            )
        return rows


def open_image(path: str | Path) -> BaseReader:
    """Open ``path`` with the best available reader (factory entry point).

    Args:
        path: Image file or product directory.

    Returns:
        An *opened* reader instance; callers should use it as a context
        manager or call :meth:`BaseReader.close` when done.
    """
    reader_cls = ReaderRegistry.find_reader(path)
    reader = reader_cls(path)
    return reader.open()


def iter_supported_extensions() -> Iterator[str]:
    """Yield every extension currently readable in this environment."""
    for reader_cls in ReaderRegistry.available_readers():
        yield from reader_cls.extensions
