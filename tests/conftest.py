"""Shared pytest fixtures: every test asset is generated programmatically.

No binary test data is stored in the repository; each fixture builds the
smallest possible file (PNG, GeoTIFF, HDF5, NetCDF, SAFE dir, Landsat dir)
on demand inside pytest's ``tmp_path``.
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the app config dir away from the real user profile."""
    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv("CHANGEMASTER_CONFIG_DIR", str(cfg_dir))
    return cfg_dir


@pytest.fixture()
def rgb_array() -> np.ndarray:
    """Deterministic 3-band uint8 array shaped (3, 32, 48)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(3, 32, 48), dtype=np.uint8)


@pytest.fixture()
def gray_array() -> np.ndarray:
    """Deterministic single-band uint16 array shaped (40, 50)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 4096, size=(40, 50), dtype=np.uint16)


@pytest.fixture()
def png_file(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """A small RGB PNG file."""
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.fromarray(np.transpose(rgb_array, (1, 2, 0)), mode="RGB").save(path)
    return path


@pytest.fixture()
def jpeg_file(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """A small RGB JPEG file."""
    from PIL import Image

    path = tmp_path / "sample.jpg"
    Image.fromarray(np.transpose(rgb_array, (1, 2, 0)), mode="RGB").save(path, quality=95)
    return path


def _make_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:32636",
    nodata: float | None = 0.0,
) -> Path:
    """Write a georeferenced GeoTIFF for tests (requires rasterio)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    bands = data if data.ndim == 3 else data[np.newaxis]
    transform = from_origin(500000.0, 4100000.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": bands.dtype.name,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)
    return path


@pytest.fixture()
def geotiff_file(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """A small georeferenced single-band GeoTIFF."""
    return _make_geotiff(tmp_path / "sample.tif", gray_array)


@pytest.fixture()
def multiband_geotiff(tmp_path: Path) -> Path:
    """A 4-band georeferenced GeoTIFF (uint16, 60x80)."""
    rng = np.random.default_rng(3)
    data = rng.integers(0, 10000, size=(4, 60, 80), dtype=np.uint16)
    return _make_geotiff(tmp_path / "multi.tif", data)


@pytest.fixture()
def hdf5_file(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """An HDF5 file with one 2-D and one 3-D dataset."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as handle:
        dset = handle.create_dataset("grid/elevation", data=gray_array)
        dset.attrs["units"] = "metres"
        cube = np.stack([gray_array, gray_array + 1, gray_array + 2])
        handle.create_dataset("grid/cube", data=cube)
    return path


@pytest.fixture()
def netcdf_file(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """A NetCDF file with a 2-D temperature variable."""
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "sample.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4") as handle:
        handle.title = "test"
        handle.createDimension("y", gray_array.shape[0])
        handle.createDimension("x", gray_array.shape[1])
        var = handle.createVariable("temperature", "u2", ("y", "x"))
        var.units = "K"
        var[:] = gray_array
    return path


_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
           xmlns:safe="http://www.esa.int/safe/sentinel/1.1">
  <metadataSection>
    <metadataObject ID="platform">
      <metadataWrap><xmlData>
        <safe:platform><safe:familyName>SENTINEL-2</safe:familyName></safe:platform>
      </xmlData></metadataWrap>
    </metadataObject>
    <metadataObject ID="acquisitionPeriod">
      <metadataWrap><xmlData>
        <safe:acquisitionPeriod>
          <safe:startTime>2024-05-01T10:20:30</safe:startTime>
        </safe:acquisitionPeriod>
      </xmlData></metadataWrap>
    </metadataObject>
  </metadataSection>
</xfdu:XFDU>
"""


@pytest.fixture()
def safe_product(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """A minimal Sentinel-2 ``.SAFE`` directory with three TIFF bands.

    Real products use JP2, but the SAFE reader accepts TIFF band files too,
    which keeps the fixture free of an OpenJPEG encoder dependency.
    """
    pytest.importorskip("rasterio")
    safe_dir = tmp_path / "S2A_MSIL2A_20240501T102030_N0510_R064_T36SXA_20240501T134500.SAFE"
    granule = safe_dir / "GRANULE" / "L2A_T36SXA" / "IMG_DATA"
    granule.mkdir(parents=True)
    (safe_dir / "manifest.safe").write_text(_MANIFEST_XML, encoding="utf-8")
    for band in ("B02", "B03", "B04"):
        _make_geotiff(granule / f"T36SXA_20240501T102030_{band}_10m.tif", gray_array)
    return safe_dir


_MTL_TEXT = """GROUP = LANDSAT_METADATA_FILE
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    DATE_ACQUIRED = 2024-05-01
    SCENE_CENTER_TIME = "08:30:15.0000000Z"
    CLOUD_COVER = 1.23
  END_GROUP = IMAGE_ATTRIBUTES
END_GROUP = LANDSAT_METADATA_FILE
END
"""


@pytest.fixture()
def landsat_dir(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """A minimal Landsat 8 product directory (MTL + 3 bands)."""
    pytest.importorskip("rasterio")
    product = tmp_path / "LC08_L2SP_174038_20240501_20240512_02_T1"
    product.mkdir()
    stem = product.name
    (product / f"{stem}_MTL.txt").write_text(_MTL_TEXT, encoding="utf-8")
    for n in (2, 3, 4):
        _make_geotiff(product / f"{stem}_B{n}.TIF", gray_array)
    return product


@pytest.fixture()
def landsat_tar(tmp_path: Path, landsat_dir: Path) -> Path:
    """The Landsat product packed into a flat ``.tar`` archive."""
    tar_path = tmp_path / f"{landsat_dir.name}.tar"
    with tarfile.open(tar_path, "w") as archive:
        for item in sorted(landsat_dir.iterdir()):
            archive.add(item, arcname=item.name)
    return tar_path
