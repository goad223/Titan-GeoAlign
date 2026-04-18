"""Sensor-specific readers for Titan-GeoAlign Layer 0."""
from __future__ import annotations

import abc
import re
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    import rasterio.features
except ImportError:
    rasterio = None  # type: ignore[assignment]

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore[assignment]

try:
    import netCDF4 as nc4
except ImportError:
    nc4 = None  # type: ignore[assignment]

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

try:
    import laspy
except ImportError:
    laspy = None  # type: ignore[assignment]

from .sensor_types import SensorType
from .image_data import ImageData

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(arr: np.ndarray, band_names: list[str]) -> "xr.Dataset":
    if xr is None:
        raise ImportError("xarray is required")
    nbands = arr.shape[0]
    return xr.Dataset(
        {band_names[i]: (["y", "x"], arr[i].astype(np.float32)) for i in range(nbands)}
    )


def _pixel_size_from_transform(transform) -> float:
    """Return approximate ground resolution in metres from an Affine transform."""
    try:
        return float(abs(transform.a))
    except Exception:
        return 0.0


def _crs_to_epsg(crs_obj) -> str:
    """Convert a rasterio CRS to an EPSG string like 'EPSG:32633'."""
    try:
        epsg = crs_obj.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
    except Exception:
        pass
    try:
        return crs_obj.to_string()
    except Exception:
        return "EPSG:4326"


# ---------------------------------------------------------------------------
# Abstract base reader
# ---------------------------------------------------------------------------

class BaseReader(abc.ABC):
    """Abstract reader interface."""

    @abc.abstractmethod
    def read(self, filepath: Path) -> ImageData:
        """Read the file at *filepath* and return an :class:`ImageData`."""

    def can_read(self, filepath: Path) -> bool:  # noqa: D401
        """Return True if this reader can handle *filepath*."""
        return False


# ---------------------------------------------------------------------------
# GeoTIFF / generic rasterio reader
# ---------------------------------------------------------------------------

class GeoTiffReader(BaseReader):
    """Reads any file supported by rasterio/GDAL (GeoTIFF, COG, etc.)."""

    def __init__(self, sensor_type: SensorType = SensorType.UNKNOWN):
        self.sensor_type = sensor_type

    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in {".tif", ".tiff", ".img", ".vrt", ".jp2"}

    def read(self, filepath: Path) -> ImageData:
        if rasterio is None:
            raise ImportError("rasterio is required for GeoTiffReader")

        filepath = Path(filepath)
        logger.info("reading GeoTIFF", path=str(filepath))

        with rasterio.open(filepath) as ds:
            arr = ds.read().astype(np.float32)          # (bands, H, W)
            meta = dict(ds.meta)
            tags = ds.tags()
            nodata = ds.nodata
            transform = ds.transform
            crs = _crs_to_epsg(ds.crs) if ds.crs else "EPSG:4326"
            bounds = tuple(ds.bounds)                   # (west, south, east, north)
            bit_depth = _bit_depth(meta.get("dtype", "float32"))
            band_names = [
                ds.descriptions[i] or f"band_{i + 1}" for i in range(ds.count)
            ]

        dataset = _make_dataset(arr, band_names)
        return ImageData(
            data=dataset,
            sensor_type=self.sensor_type,
            crs=crs,
            transform=transform,
            bounds=bounds,
            resolution_m=_pixel_size_from_transform(transform),
            bit_depth=bit_depth,
            band_names=band_names,
            metadata=tags,
            nodata=nodata,
            filepath=filepath,
        )


# ---------------------------------------------------------------------------
# Sentinel-1 SAR reader
# ---------------------------------------------------------------------------

class Sentinel1Reader(GeoTiffReader):
    """Reads Sentinel-1 GRD/SLC products (SAFE directory or single .tif).

    Handles VV/VH polarisations and converts amplitude to dB.
    """

    # Common Sentinel-1 GRD sub-image suffixes
    _POLAR_RE = re.compile(r"_(vv|vh|hh|hv)_", re.IGNORECASE)

    def can_read(self, filepath: Path) -> bool:
        name = filepath.name.upper()
        return name.startswith("S1") or self._POLAR_RE.search(filepath.name) is not None

    def read(self, filepath: Path) -> ImageData:
        filepath = Path(filepath)

        # If given a SAFE directory, find measurement .tif files
        if filepath.is_dir():
            tifs = sorted(filepath.rglob("measurement/*.tif"))
            if not tifs:
                tifs = sorted(filepath.rglob("*.tif"))
            if not tifs:
                raise FileNotFoundError(f"No measurement GeoTIFFs in {filepath}")
            # Read and stack polarisations
            images = [self._read_single(t) for t in tifs]
            return self._stack_polarisations(images, tifs)

        return self._read_single(filepath)

    def _read_single(self, filepath: Path) -> ImageData:
        img = super().read(filepath)
        img.sensor_type = SensorType.SENTINEL1_SAR
        # Amplitude → power (linear) → dB
        arr = img.to_numpy()
        arr = np.where(arr > 0, 10.0 * np.log10(arr ** 2 + 1e-10), -30.0)
        img.data = _make_dataset(arr, img.band_names)
        return img

    def _stack_polarisations(
        self, images: list[ImageData], paths: list[Path]
    ) -> ImageData:
        if len(images) == 1:
            return images[0]
        # Build band names from polarisation suffix
        names = []
        for p in paths:
            m = self._POLAR_RE.search(p.name)
            names.append(m.group(1).upper() if m else p.stem)

        first = images[0]
        arrays = [img.to_numpy()[0] for img in images]  # each is single-band
        stacked = np.stack(arrays, axis=0)
        dataset = _make_dataset(stacked, names)
        first.data = dataset
        first.band_names = names
        first.sensor_type = SensorType.SENTINEL1_SAR
        return first


# ---------------------------------------------------------------------------
# Sentinel-2 reader
# ---------------------------------------------------------------------------

class Sentinel2Reader(GeoTiffReader):
    """Reads Sentinel-2 .SAFE directory or a pre-stacked .tif."""

    _S2_BAND_ORDER = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B08A", "B09", "B10", "B11", "B12",
    ]

    def can_read(self, filepath: Path) -> bool:
        return (
            (filepath.is_dir() and filepath.suffix.upper() == ".SAFE")
            or bool(re.match(r"S2[AB]_MSI", filepath.name, re.IGNORECASE))
        )

    def read(self, filepath: Path) -> ImageData:
        filepath = Path(filepath)

        if filepath.is_file():
            img = super().read(filepath)
            img.sensor_type = SensorType.SENTINEL2
            # Reassign generic band names to S2 convention if count matches
            if len(img.band_names) == len(self._S2_BAND_ORDER):
                img.band_names = list(self._S2_BAND_ORDER)
                ds_vars = {}
                for i, name in enumerate(img.band_names):
                    old_name = f"band_{i + 1}"
                    arr = img.data.get(old_name) if old_name in img.data else list(img.data.data_vars.values())[i]
                    ds_vars[name] = (["y", "x"], arr.values if hasattr(arr, "values") else arr)
                img.data = xr.Dataset(ds_vars)
            return img

        # .SAFE directory — find 10/20/60m jp2 files
        if filepath.is_dir():
            return self._read_safe(filepath)

        raise FileNotFoundError(f"Cannot read Sentinel-2 product: {filepath}")

    def _read_safe(self, safe_dir: Path) -> ImageData:
        """Read and stack all bands from a .SAFE directory at 10m."""
        granule_dirs = sorted((safe_dir / "GRANULE").iterdir()) if (safe_dir / "GRANULE").exists() else []
        if not granule_dirs:
            raise FileNotFoundError(f"No GRANULE directory in {safe_dir}")

        img_data_dir = granule_dirs[0] / "IMG_DATA"
        r10 = img_data_dir / "R10m"
        r20 = img_data_dir / "R20m"
        r60 = img_data_dir / "R60m"

        bands: dict[str, np.ndarray] = {}
        transform = None
        crs = "EPSG:32632"
        bounds = (0.0, 0.0, 1.0, 1.0)

        for res_dir in [r10, r20, r60, img_data_dir]:
            if not res_dir.exists():
                continue
            for jp2 in sorted(res_dir.glob("*.jp2")):
                m = re.search(r"_(B\d+[A-Z]?)_", jp2.name)
                if not m:
                    continue
                band_id = m.group(1)
                if band_id in bands:
                    continue
                with rasterio.open(jp2) as ds:
                    arr = ds.read(1).astype(np.float32)
                    if transform is None:
                        transform = ds.transform
                        crs = _crs_to_epsg(ds.crs)
                        bounds = tuple(ds.bounds)
                        ref_shape = arr.shape
                    # Resample to reference shape if needed
                    if arr.shape != ref_shape:
                        from rasterio.warp import reproject, Resampling
                        dest = np.empty(ref_shape, dtype=np.float32)
                        reproject(
                            source=arr,
                            destination=dest,
                            src_transform=ds.transform,
                            src_crs=ds.crs,
                            dst_transform=transform,
                            dst_crs=CRS.from_user_input(crs),
                            resampling=Resampling.bilinear,
                        )
                        arr = dest
                    bands[band_id] = arr

        if not bands:
            raise ValueError(f"No S2 bands found in {safe_dir}")

        # Order bands
        ordered_names = [b for b in self._S2_BAND_ORDER if b in bands]
        ordered_arrays = np.stack([bands[b] for b in ordered_names], axis=0)
        dataset = _make_dataset(ordered_arrays, ordered_names)
        resolution = abs(transform.a) if transform else 10.0

        return ImageData(
            data=dataset,
            sensor_type=SensorType.SENTINEL2,
            crs=crs,
            transform=transform,
            bounds=bounds,
            resolution_m=resolution,
            bit_depth=16,
            band_names=ordered_names,
            nodata=0.0,
            filepath=safe_dir,
        )


# ---------------------------------------------------------------------------
# Landsat 8/9 reader
# ---------------------------------------------------------------------------

class Landsat89Reader(GeoTiffReader):
    """Reads Landsat Collection 2 Level-2 products."""

    _BAND_MAP = {
        "B2": "SR_B2",  # Blue
        "B3": "SR_B3",  # Green
        "B4": "SR_B4",  # Red
        "B5": "SR_B5",  # NIR
        "B6": "SR_B6",  # SWIR 1
        "B7": "SR_B7",  # SWIR 2
        "B10": "ST_B10",  # Thermal
    }

    def can_read(self, filepath: Path) -> bool:
        return bool(re.match(r"L[CO]0[89]_", filepath.name, re.IGNORECASE))

    def read(self, filepath: Path) -> ImageData:
        filepath = Path(filepath)
        sensor = (
            SensorType.LANDSAT8
            if re.search(r"L[CO]08", filepath.name, re.IGNORECASE)
            else SensorType.LANDSAT9
        )

        # If a directory — stack individual band TIFFs
        if filepath.is_dir():
            return self._read_collection2_dir(filepath, sensor)

        img = super().read(filepath)
        img.sensor_type = sensor
        return img

    def _read_collection2_dir(self, scene_dir: Path, sensor: SensorType) -> ImageData:
        band_files: dict[str, Path] = {}
        for tif in sorted(scene_dir.glob("*_SR_B*.TIF")) + sorted(scene_dir.glob("*_ST_B*.TIF")):
            m = re.search(r"_(SR_B\d+|ST_B\d+)\.TIF", tif.name, re.IGNORECASE)
            if m:
                band_files[m.group(1).upper()] = tif

        if not band_files:
            raise FileNotFoundError(f"No Landsat band TIFFs in {scene_dir}")

        ordered_bands = sorted(band_files.keys())
        arrays = []
        transform = None
        crs = "EPSG:32632"
        bounds = (0.0, 0.0, 1.0, 1.0)
        ref_shape = None

        for band_name in ordered_bands:
            with rasterio.open(band_files[band_name]) as ds:
                arr = ds.read(1).astype(np.float32)
                if transform is None:
                    transform = ds.transform
                    crs = _crs_to_epsg(ds.crs)
                    bounds = tuple(ds.bounds)
                    ref_shape = arr.shape
                arrays.append(arr)

        stacked = np.stack(arrays, axis=0)
        dataset = _make_dataset(stacked, ordered_bands)

        return ImageData(
            data=dataset,
            sensor_type=sensor,
            crs=crs,
            transform=transform,
            bounds=bounds,
            resolution_m=30.0,
            bit_depth=16,
            band_names=ordered_bands,
            nodata=0.0,
            filepath=scene_dir,
        )


# ---------------------------------------------------------------------------
# NetCDF reader (MODIS, general)
# ---------------------------------------------------------------------------

class NetCDFReader(BaseReader):
    """Reads NetCDF4 files (MODIS, ERA5, general hyperspectral)."""

    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in {".nc", ".nc4"}

    def read(self, filepath: Path) -> ImageData:
        if xr is None:
            raise ImportError("xarray is required for NetCDFReader")

        filepath = Path(filepath)
        logger.info("reading NetCDF", path=str(filepath))

        ds = xr.open_dataset(filepath, engine="netcdf4")
        data_vars = list(ds.data_vars)

        # Build a consistent (H, W) representation
        bands: dict[str, np.ndarray] = {}
        ref_shape: Optional[tuple] = None

        for var in data_vars:
            arr = ds[var].values
            if arr.ndim == 2:
                if ref_shape is None:
                    ref_shape = arr.shape
                if arr.shape == ref_shape:
                    bands[var] = arr.astype(np.float32)
            elif arr.ndim == 3:
                for i in range(arr.shape[0]):
                    name = f"{var}_{i}"
                    if ref_shape is None:
                        ref_shape = arr.shape[1:]
                    if arr[i].shape == ref_shape:
                        bands[name] = arr[i].astype(np.float32)

        if not bands:
            raise ValueError(f"No 2D/3D variables found in {filepath}")

        band_names = list(bands.keys())
        stacked = np.stack([bands[n] for n in band_names], axis=0)
        dataset = _make_dataset(stacked, band_names)

        # Best-effort spatial metadata
        lat = ds.coords.get("lat") or ds.coords.get("latitude")
        lon = ds.coords.get("lon") or ds.coords.get("longitude")
        if lat is not None and lon is not None:
            west = float(lon.min())
            east = float(lon.max())
            south = float(lat.min())
            north = float(lat.max())
            h, w = ref_shape if ref_shape else (1, 1)
            from rasterio.transform import from_bounds as _fb

            transform = _fb(west, south, east, north, w, h)
            bounds = (west, south, east, north)
        else:
            from affine import Affine

            transform = Affine.identity()
            bounds = (0.0, 0.0, 1.0, 1.0)

        res = abs(float(transform.a)) if transform else 1.0

        return ImageData(
            data=dataset,
            sensor_type=SensorType.MODIS,
            crs="EPSG:4326",
            transform=transform,
            bounds=bounds,
            resolution_m=res * 111320.0 if res < 1.0 else res,
            bit_depth=16,
            band_names=band_names,
            metadata=dict(ds.attrs),
            nodata=None,
            filepath=filepath,
        )


# ---------------------------------------------------------------------------
# HDF5 reader (EnMAP, PRISMA)
# ---------------------------------------------------------------------------

class HDF5Reader(BaseReader):
    """Reads HDF5 files (EnMAP Level-2A, PRISMA)."""

    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in {".h5", ".he5", ".hdf", ".hdf5"}

    def read(self, filepath: Path) -> ImageData:
        if h5py is None:
            raise ImportError("h5py is required for HDF5Reader")
        if xr is None:
            raise ImportError("xarray is required for HDF5Reader")

        filepath = Path(filepath)
        logger.info("reading HDF5", path=str(filepath))

        bands: dict[str, np.ndarray] = {}
        metadata: dict = {}

        def _visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim in {2, 3}:
                arr = obj[()].astype(np.float32)
                safe_name = name.replace("/", "__")
                if arr.ndim == 2:
                    bands[safe_name] = arr
                elif arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        bands[f"{safe_name}_{i:04d}"] = arr[i]
            elif isinstance(obj, h5py.Group):
                metadata.update({k: str(v) for k, v in obj.attrs.items()})

        with h5py.File(filepath, "r") as f:
            metadata.update({k: str(v) for k, v in f.attrs.items()})
            f.visititems(_visit)

        if not bands:
            raise ValueError(f"No 2D/3D datasets in HDF5 file: {filepath}")

        # Normalise to same shape
        shapes = {v.shape for v in bands.values()}
        ref_shape = max(shapes, key=lambda s: s[0] * s[1])
        filtered = {k: v for k, v in bands.items() if v.shape == ref_shape}

        if not filtered:
            filtered = bands  # keep all anyway

        band_names = list(filtered.keys())
        stacked = np.stack([filtered[n] for n in band_names], axis=0)
        dataset = _make_dataset(stacked, band_names)

        # Detect sensor from filename
        sensor = SensorType.ENMAP
        if "PRS" in filepath.name.upper():
            sensor = SensorType.PRISMA

        from affine import Affine

        return ImageData(
            data=dataset,
            sensor_type=sensor,
            crs="EPSG:4326",
            transform=Affine.identity(),
            bounds=(0.0, 0.0, 1.0, 1.0),
            resolution_m=30.0,
            bit_depth=16,
            band_names=band_names,
            metadata=metadata,
            nodata=None,
            filepath=filepath,
        )


# ---------------------------------------------------------------------------
# LiDAR reader
# ---------------------------------------------------------------------------

class LiDARReader(BaseReader):
    """Reads LAS/LAZ point-cloud files and returns a rasterized DEM as ImageData."""

    def can_read(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in {".las", ".laz"}

    def read(self, filepath: Path, resolution_m: float = 1.0) -> ImageData:
        if laspy is None:
            raise ImportError("laspy is required for LiDARReader")
        if xr is None:
            raise ImportError("xarray is required for LiDARReader")

        filepath = Path(filepath)
        logger.info("reading LiDAR", path=str(filepath))

        las = laspy.read(filepath)
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)

        west, east = x.min(), x.max()
        south, north = y.min(), y.max()

        cols = max(1, int(np.ceil((east - west) / resolution_m)))
        rows = max(1, int(np.ceil((north - south) / resolution_m)))

        dem = np.full((rows, cols), np.nan, dtype=np.float32)
        col_idx = np.clip(((x - west) / resolution_m).astype(int), 0, cols - 1)
        row_idx = np.clip(((north - y) / resolution_m).astype(int), 0, rows - 1)

        # Simple max-z aggregation (last return)
        for ci, ri, zi in zip(col_idx, row_idx, z):
            if np.isnan(dem[ri, ci]) or zi > dem[ri, ci]:
                dem[ri, ci] = float(zi)

        # Interpolate NaN gaps with nearest-neighbour
        from scipy.ndimage import distance_transform_edt

        nan_mask = np.isnan(dem)
        if nan_mask.any():
            _, idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
            dem[nan_mask] = dem[idx[0][nan_mask], idx[1][nan_mask]]

        from rasterio.transform import from_bounds as _fb

        transform = _fb(west, south, east, north, cols, rows)
        dataset = _make_dataset(dem[np.newaxis], ["elevation"])

        return ImageData(
            data=dataset,
            sensor_type=SensorType.LIDAR,
            crs="EPSG:4326",
            transform=transform,
            bounds=(west, south, east, north),
            resolution_m=resolution_m,
            bit_depth=32,
            band_names=["elevation"],
            nodata=np.nan,
            filepath=filepath,
        )


# ---------------------------------------------------------------------------
# UAV reader
# ---------------------------------------------------------------------------

class UAVReader(GeoTiffReader):
    """Reads high-resolution UAV imagery (RGB or multi-spectral GeoTIFF)."""

    def can_read(self, filepath: Path) -> bool:
        if not super().can_read(filepath):
            return False
        return True  # catch-all for high-res geo-referenced imagery

    def read(self, filepath: Path) -> ImageData:
        img = super().read(filepath)
        # Heuristic: <1 m/px → UAV, and 3 bands → RGB else multispectral
        if img.resolution_m < 1.0:
            img.sensor_type = (
                SensorType.UAV_RGB if len(img.band_names) == 3
                else SensorType.UAV_MULTISPECTRAL
            )
        return img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bit_depth(dtype_str: str) -> int:
    mapping = {
        "uint8": 8,
        "uint16": 16,
        "int16": 16,
        "uint32": 32,
        "int32": 32,
        "float32": 32,
        "float64": 64,
    }
    return mapping.get(str(dtype_str), 16)
