"""Tests for sensor profiles, registry and auto-detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from changemaster.core.exceptions import SensorProfileError
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.sensors.profiles import SensorProfile, SensorRegistry

EXPECTED_PROFILES = {
    "sentinel1",
    "sentinel2",
    "landsat5",
    "landsat7",
    "landsat8",
    "landsat9",
    "modis",
    "worldview",
    "pleiades",
    "spot",
    "planetscope",
    "generic",
}


def test_all_twelve_profiles_load() -> None:
    names = {p.name for p in SensorRegistry.all_profiles()}
    assert names == EXPECTED_PROFILES


def test_profiles_have_bilingual_descriptions() -> None:
    for profile in SensorRegistry.all_profiles():
        assert profile.description_en, profile.name
        assert profile.description_ar, profile.name


def test_get_by_name_and_unknown() -> None:
    s2 = SensorRegistry.get("sentinel2")
    assert s2.display_name == "Sentinel-2 MSI"
    assert SensorRegistry.get("SENTINEL2") is s2  # case-insensitive
    with pytest.raises(SensorProfileError):
        SensorRegistry.get("hubble")


def test_sentinel2_band_lookup() -> None:
    s2 = SensorRegistry.get("sentinel2")
    red = s2.band_by_alias("red")
    assert red is not None and red.name == "B04"
    assert red.wavelength_nm == 665
    nir = s2.band_by_alias("NIR")
    assert nir is not None and nir.name == "B08"
    assert s2.band_index("B8A") == 9
    assert s2.band_index("B99") is None
    assert s2.band_by_alias("nothing") is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("S2A_MSIL2A_20240501T102030_N0510_R064_T36SXA.SAFE", "sentinel2"),
        ("S1B_IW_GRDH_1SDV_20240101T050505.SAFE", "sentinel1"),
        ("LC08_L2SP_174038_20240501_02_T1.tar", "landsat8"),
        ("LC09_L1TP_174038_20240501_02_T1", "landsat9"),
        ("LE07_L1TP_174038_20020501_02_T1", "landsat7"),
        ("LT05_L1TP_174038_19950501_02_T1", "landsat5"),
        ("MOD09GA.A2024122.h20v05.061.hdf", "modis"),
        ("WV03_20240501_PAN.tif", "worldview"),
        ("IMG_PHR1A_MS_001.tif", "pleiades"),
        ("SPOT7_MS_202405010830.tif", "spot"),
        ("20240501_083015_99_2451_3B_AnalyticMS_PSScene.tif", "planetscope"),
        ("random_image.tif", "generic"),
    ],
)
def test_detect_from_filename(filename: str, expected: str) -> None:
    assert SensorRegistry.detect_from_filename(filename).name == expected


def test_detect_from_full_path() -> None:
    path = Path("/archive/2024") / "S2A_MSIL2A_20240501T102030.SAFE"
    assert SensorRegistry.detect_from_filename(path).name == "sentinel2"


def _meta(path: str, sensor: str | None = None) -> ImageMetadata:
    return ImageMetadata(
        path=Path(path),
        format_name="GeoTIFF",
        driver="rasterio",
        width=10,
        height=10,
        band_count=1,
        dtype="uint16",
        sensor=sensor,
    )


def test_detect_from_metadata_sensor_field() -> None:
    meta = _meta("/data/scene.tif", sensor="LANDSAT_8")
    assert SensorRegistry.detect_from_metadata(meta).name == "landsat8"
    meta2 = _meta("/data/x.tif", sensor="SENTINEL-2")
    assert SensorRegistry.detect_from_metadata(meta2).name == "sentinel2"


def test_detect_from_metadata_falls_back_to_filename() -> None:
    meta = _meta("/data/LC09_L1TP_174038.tif")
    assert SensorRegistry.detect_from_metadata(meta).name == "landsat9"
    assert SensorRegistry.detect_from_metadata(_meta("/data/plain.tif")).name == "generic"


def test_from_dict_validates_required_keys() -> None:
    with pytest.raises(SensorProfileError):
        SensorProfile.from_dict({"name": "incomplete"})


def test_from_dict_full_roundtrip() -> None:
    profile = SensorProfile.from_dict(
        {
            "name": "testsat",
            "display_name": "TestSat",
            "platform": "Test",
            "sensor_type": "optical",
            "resolution_m": 5.0,
            "bands": [{"name": "B1", "wavelength_nm": 500, "alias": "blue", "resolution_m": 5}],
            "filename_patterns": ["^TEST_"],
        }
    )
    assert profile.matches_filename("TEST_20240101.tif")
    assert not profile.matches_filename("OTHER.tif")
    assert profile.bands[0].alias == "blue"


def test_registry_reload() -> None:
    before = SensorRegistry.all_profiles()
    SensorRegistry.reload()
    after = SensorRegistry.all_profiles()
    assert {p.name for p in before} == {p.name for p in after}
