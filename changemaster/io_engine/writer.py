"""Output writers: CRS-preserving GeoTIFF plus quick-look PNG export.

GeoTIFF writing uses rasterio (lazy import) and preserves CRS/transform from
the source metadata. PNG export normalizes any numeric array to 8-bit and
works with Pillow alone, so it is always available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from changemaster.core.exceptions import MissingDependencyError, WriterError
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.raster_reader import _try_import_rasterio


def _as_bands(data: np.ndarray) -> np.ndarray:
    """Normalize an array to ``(bands, rows, cols)`` shape.

    Raises:
        WriterError: For arrays that are not 2-D or 3-D.
    """
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3:
        return data
    raise WriterError(
        f"Expected a 2-D or 3-D array, got {data.ndim}-D.",
        f"متوقع مصفوفة ثنائية أو ثلاثية الأبعاد، الوارد {data.ndim} أبعاد.",
    )


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    metadata: ImageMetadata | None = None,
    crs: str | None = None,
    transform: tuple[float, float, float, float, float, float] | None = None,
    nodata: float | None = None,
    compress: str = "deflate",
) -> Path:
    """Write an array to GeoTIFF, preserving georeferencing.

    Args:
        path: Output file path (created/overwritten).
        data: Pixel array shaped ``(rows, cols)`` or ``(bands, rows, cols)``.
        metadata: Optional source metadata; its CRS/transform/nodata are used
            as defaults for the corresponding parameters.
        crs: CRS override (WKT or ``EPSG:xxxx`` string).
        transform: Affine transform override as a 6-tuple ``(a,b,c,d,e,f)``.
        nodata: NoData value override.
        compress: GeoTIFF compression (``deflate``, ``lzw``, ``none``).

    Returns:
        The path that was written.

    Raises:
        MissingDependencyError: If rasterio is not installed.
        WriterError: When the write fails or the array shape is invalid.
    """
    rasterio = _try_import_rasterio()
    if rasterio is None:
        raise MissingDependencyError("rasterio", feature="writing GeoTIFF files")
    from rasterio.transform import Affine  # noqa: PLC0415

    target = Path(path)
    bands = _as_bands(np.asarray(data))
    out_crs = crs or (metadata.crs if metadata else None)
    out_transform = transform or (metadata.transform if metadata else None)
    out_nodata = nodata if nodata is not None else (metadata.nodata if metadata else None)
    affine = Affine(*out_transform) if out_transform else None
    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": bands.dtype.name,
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }
    if out_crs:
        profile["crs"] = out_crs
    if affine is not None:
        profile["transform"] = affine
    if out_nodata is not None:
        profile["nodata"] = out_nodata
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(bands)
    except Exception as exc:  # noqa: BLE001 - rasterio raises many types
        raise WriterError(
            f"Failed to write GeoTIFF {target}: {exc}",
            f"فشل في كتابة GeoTIFF {target}: {exc}",
        ) from exc
    return target


def normalize_to_uint8(data: np.ndarray, percentile_clip: float = 2.0) -> np.ndarray:
    """Stretch a numeric array to the 0-255 range as ``uint8``.

    Uses percentile clipping for robust contrast on satellite imagery.

    Args:
        data: Numeric array of any shape/dtype.
        percentile_clip: Percentile clipped from each end (0 disables).

    Returns:
        ``uint8`` array of identical shape.

    Raises:
        WriterError: When ``percentile_clip`` is out of the ``[0, 50)`` range.
    """
    if not 0 <= percentile_clip < 50:
        raise WriterError(
            f"percentile_clip must be in [0, 50), got {percentile_clip}.",
            f"قيمة percentile_clip يجب أن تكون في [0, 50)، الوارد {percentile_clip}.",
        )
    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low = float(np.percentile(finite, percentile_clip))
    high = float(np.percentile(finite, 100 - percentile_clip))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip(arr, low, high)
    scaled = (arr - low) / (high - low) * 255.0
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.uint8)


def export_png(
    path: str | Path,
    data: np.ndarray,
    percentile_clip: float = 2.0,
) -> Path:
    """Export an array to an 8-bit PNG quick-look (always available).

    1-band input becomes grayscale; 3+ band input uses the first three
    bands as RGB.

    Args:
        path: Output PNG path.
        data: Array shaped ``(rows, cols)`` or ``(bands, rows, cols)``.
        percentile_clip: Contrast stretch clip percentage per band.

    Returns:
        The path that was written.

    Raises:
        WriterError: For invalid array shapes or write failures.
    """
    from PIL import Image  # noqa: PLC0415

    target = Path(path)
    bands = _as_bands(np.asarray(data))
    if bands.shape[0] >= 3:
        stacked = np.stack(
            [normalize_to_uint8(bands[i], percentile_clip) for i in range(3)],
            axis=-1,
        )
        image = Image.fromarray(stacked, mode="RGB")
    else:
        image = Image.fromarray(normalize_to_uint8(bands[0], percentile_clip), mode="L")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
    except OSError as exc:
        raise WriterError(
            f"Failed to write PNG {target}: {exc}",
            f"فشل في كتابة PNG {target}: {exc}",
        ) from exc
    return target
