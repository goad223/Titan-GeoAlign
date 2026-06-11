"""Unified image metadata model shared by every reader in the I/O engine."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ImageMetadata:
    """Format-agnostic description of a raster image.

    Attributes:
        path: Source file or product directory.
        format_name: Short format identifier (e.g. ``GeoTIFF``, ``PNG``).
        driver: Backend used to read the file (e.g. ``rasterio``, ``pillow``).
        width: Image width in pixels.
        height: Image height in pixels.
        band_count: Number of bands/channels.
        dtype: NumPy dtype name of pixel data (e.g. ``uint16``).
        crs: Coordinate reference system as WKT/EPSG string, if georeferenced.
        transform: Affine geotransform as a 6-tuple
            ``(a, b, c, d, e, f)`` matching rasterio's ``Affine``, if any.
        nodata: NoData value, if defined.
        band_names: Optional human-readable band names.
        sensor: Detected sensor/profile name (e.g. ``sentinel2``), if known.
        acquisition_datetime: ISO-8601 acquisition time, if known.
        extra: Free-form extra metadata from the source format.
    """

    path: Path
    format_name: str
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    nodata: float | None = None
    band_names: list[str] = field(default_factory=list)
    sensor: str | None = None
    acquisition_datetime: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` pixel dimensions."""
        return (self.height, self.width)

    @property
    def pixel_count(self) -> int:
        """Total number of pixels per band."""
        return self.width * self.height

    @property
    def is_georeferenced(self) -> bool:
        """Whether the image carries CRS and transform information."""
        return self.crs is not None and self.transform is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary (``path`` becomes str)."""
        data = asdict(self)
        data["path"] = str(self.path)
        if self.transform is not None:
            data["transform"] = list(self.transform)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageMetadata":
        """Rebuild metadata from :meth:`to_dict` output.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A reconstructed :class:`ImageMetadata` instance.
        """
        payload = dict(data)
        payload["path"] = Path(payload["path"])
        transform = payload.get("transform")
        if transform is not None:
            payload["transform"] = tuple(float(v) for v in transform)
        return cls(**payload)

    def summary(self) -> str:
        """Render a concise bilingual one-image summary for CLI/GUI display."""
        geo = "yes | نعم" if self.is_georeferenced else "no | لا"
        lines = [
            f"File | الملف: {self.path}",
            f"Format | الصيغة: {self.format_name} (driver: {self.driver})",
            f"Size | الأبعاد: {self.width} x {self.height} px, {self.band_count} band(s)",
            f"Data type | نوع البيانات: {self.dtype}",
            f"Georeferenced | مرجعية جغرافية: {geo}",
        ]
        if self.crs:
            lines.append(f"CRS | نظام الإحداثيات: {self.crs}")
        if self.nodata is not None:
            lines.append(f"NoData: {self.nodata}")
        if self.sensor:
            lines.append(f"Sensor | المستشعر: {self.sensor}")
        if self.acquisition_datetime:
            lines.append(f"Acquired | تاريخ الالتقاط: {self.acquisition_datetime}")
        return "\n".join(lines)
