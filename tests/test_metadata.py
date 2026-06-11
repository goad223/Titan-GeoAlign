"""Tests for the unified ImageMetadata model."""

from __future__ import annotations

from pathlib import Path

from changemaster.io_engine.metadata import ImageMetadata


def _meta(**overrides: object) -> ImageMetadata:
    base: dict[str, object] = {
        "path": Path("/data/scene.tif"),
        "format_name": "GeoTIFF",
        "driver": "rasterio",
        "width": 100,
        "height": 50,
        "band_count": 3,
        "dtype": "uint16",
    }
    base.update(overrides)
    return ImageMetadata(**base)  # type: ignore[arg-type]


def test_shape_and_pixel_count() -> None:
    meta = _meta()
    assert meta.shape == (50, 100)
    assert meta.pixel_count == 5000


def test_georeferenced_requires_crs_and_transform() -> None:
    assert not _meta().is_georeferenced
    assert not _meta(crs="EPSG:4326").is_georeferenced
    geo = _meta(crs="EPSG:4326", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0))
    assert geo.is_georeferenced


def test_to_dict_from_dict_roundtrip() -> None:
    meta = _meta(
        crs="EPSG:32636",
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0),
        nodata=0.0,
        band_names=["red", "green", "blue"],
        sensor="sentinel2",
        acquisition_datetime="2024-05-01T10:20:30",
        extra={"cloud_cover": 1.5},
    )
    data = meta.to_dict()
    assert isinstance(data["path"], str)
    assert data["transform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0]
    rebuilt = ImageMetadata.from_dict(data)
    assert rebuilt == meta


def test_from_dict_without_transform() -> None:
    rebuilt = ImageMetadata.from_dict(_meta().to_dict())
    assert rebuilt.transform is None
    assert rebuilt.path == Path("/data/scene.tif")


def test_summary_is_bilingual_and_complete() -> None:
    meta = _meta(
        crs="EPSG:32636",
        transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        nodata=0.0,
        sensor="sentinel2",
        acquisition_datetime="2024-05-01T10:20:30",
    )
    text = meta.summary()
    assert "الملف" in text
    assert "GeoTIFF" in text
    assert "100 x 50" in text
    assert "EPSG:32636" in text
    assert "sentinel2" in text
    assert "2024-05-01T10:20:30" in text
    assert "NoData: 0.0" in text


def test_summary_minimal() -> None:
    text = _meta().summary()
    assert "no | لا" in text
