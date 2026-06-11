"""Tests for the bilingual exception hierarchy."""

from __future__ import annotations

import pytest

from changemaster.core.exceptions import (
    BandError,
    ChangeMasterError,
    ConfigError,
    FileAccessError,
    MetadataError,
    MissingDependencyError,
    ReaderError,
    SensorProfileError,
    TileAccessError,
    UnsupportedFormatError,
    WriterError,
)


def test_base_error_default_messages() -> None:
    err = ChangeMasterError()
    assert err.message_en
    assert err.message_ar
    assert err.message_en in str(err)
    assert err.message_ar in str(err)


def test_base_error_custom_messages() -> None:
    err = ChangeMasterError("english", "عربي")
    assert err.message_en == "english"
    assert err.message_ar == "عربي"
    assert err.bilingual() == {"en": "english", "ar": "عربي"}


@pytest.mark.parametrize(
    "exc_cls",
    [ConfigError, ReaderError, WriterError, TileAccessError, SensorProfileError, MetadataError],
)
def test_subclasses_inherit_base(exc_cls: type[ChangeMasterError]) -> None:
    err = exc_cls()
    assert isinstance(err, ChangeMasterError)
    assert err.message_en and err.message_ar


def test_missing_dependency_error_names_package() -> None:
    err = MissingDependencyError("rasterio", feature="GeoTIFF reading")
    assert err.package == "rasterio"
    assert "pip install rasterio" in err.message_en
    assert "rasterio" in err.message_ar
    assert "GeoTIFF reading" in err.message_en


def test_unsupported_format_error_includes_path() -> None:
    err = UnsupportedFormatError("/data/file.xyz")
    assert "/data/file.xyz" in err.message_en
    assert "/data/file.xyz" in err.message_ar
    assert isinstance(err, ReaderError)


def test_file_access_error_includes_path() -> None:
    err = FileAccessError("/missing/file.tif")
    assert "/missing/file.tif" in str(err)
    assert isinstance(err, ReaderError)


def test_band_error_reports_available() -> None:
    err = BandError(9, 3)
    assert "9" in err.message_en
    assert "3" in err.message_en
    err2 = BandError("B99", ["B02", "B03"])
    assert "B99" in err2.message_en
    assert "B02" in err2.message_en


def test_exceptions_are_catchable_as_base() -> None:
    with pytest.raises(ChangeMasterError):
        raise FileAccessError("/nope")
