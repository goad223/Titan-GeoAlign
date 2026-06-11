"""Tests for HDF5 and NetCDF readers (skipped if backends missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import BandError, MetadataError, ReaderError


class TestHDFReader:
    """HDF5 reader behavior (requires h5py)."""

    @pytest.fixture(autouse=True)
    def _require_h5py(self) -> None:
        pytest.importorskip("h5py")

    def test_metadata_auto_selects_largest_dataset(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file) as reader:
            meta = reader.metadata
            # The 3-D cube is larger than the 2-D elevation grid.
            assert meta.extra["dataset"] == "grid/cube"
            assert meta.band_count == 3
            assert meta.driver == "h5py"

    def test_explicit_dataset(self, hdf5_file: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file, dataset="grid/elevation") as reader:
            meta = reader.metadata
            assert meta.band_count == 1
            assert meta.extra["attrs"]["units"] == "metres"
            np.testing.assert_array_equal(reader.read_band(1), gray_array)

    def test_windowed_read(self, hdf5_file: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file, dataset="grid/cube") as reader:
            window = (5, 2, 12, 9)
            data = reader.read(window=window)
            assert data.shape == (3, 9, 12)
            np.testing.assert_array_equal(data[0], gray_array[2:11, 5:17])

    def test_2d_windowed_read(self, hdf5_file: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file, dataset="grid/elevation") as reader:
            data = reader.read(window=(0, 0, 10, 5))
            assert data.shape == (1, 5, 10)
            np.testing.assert_array_equal(data[0], gray_array[:5, :10])

    def test_invalid_dataset_name(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with pytest.raises(MetadataError):
            HDFReader(hdf5_file, dataset="missing/dataset").open()

    def test_invalid_band(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file, dataset="grid/cube") as reader:
            with pytest.raises(BandError):
                reader.read_band(4)

    def test_list_datasets(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        names = HDFReader.list_datasets(hdf5_file)
        assert "grid/elevation" in names
        assert "grid/cube" in names

    def test_no_suitable_dataset(self, tmp_path: Path) -> None:
        import h5py

        from changemaster.io_engine.hdf_reader import HDFReader

        path = tmp_path / "scalar.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("value", data=np.array([1, 2, 3]))
        with pytest.raises(MetadataError):
            HDFReader(path).open()

    def test_corrupt_file(self, tmp_path: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        bad = tmp_path / "bad.h5"
        bad.write_bytes(b"not hdf5 data")
        with pytest.raises(ReaderError):
            HDFReader(bad).open()


class TestNetCDFReader:
    """NetCDF reader behavior (requires netCDF4)."""

    @pytest.fixture(autouse=True)
    def _require_netcdf(self) -> None:
        pytest.importorskip("netCDF4")

    def test_metadata_and_read(self, netcdf_file: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file) as reader:
            meta = reader.metadata
            assert meta.extra["variable"] == "temperature"
            assert meta.extra["attrs"]["units"] == "K"
            assert meta.extra["global_attrs"]["title"] == "test"
            assert meta.band_count == 1
            np.testing.assert_array_equal(reader.read_band(1), gray_array)

    def test_windowed_read(self, netcdf_file: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file) as reader:
            data = reader.read(window=(3, 4, 8, 6))
            assert data.shape == (1, 6, 8)
            np.testing.assert_array_equal(data[0], gray_array[4:10, 3:11])

    def test_explicit_variable_invalid(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with pytest.raises(MetadataError):
            NetCDFReader(netcdf_file, variable="missing").open()

    def test_invalid_band(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file) as reader:
            with pytest.raises(BandError):
                reader.read_band(2)

    def test_list_variables(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        assert "temperature" in NetCDFReader.list_variables(netcdf_file)

    def test_no_suitable_variable(self, tmp_path: Path) -> None:
        import netCDF4

        from changemaster.io_engine.netcdf_reader import NetCDFReader

        path = tmp_path / "scalar.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as handle:
            handle.createDimension("t", 3)
            var = handle.createVariable("series", "f4", ("t",))
            var[:] = [1.0, 2.0, 3.0]
        with pytest.raises(MetadataError):
            NetCDFReader(path).open()

    def test_corrupt_file(self, tmp_path: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        bad = tmp_path / "bad.nc"
        bad.write_bytes(b"junk bytes")
        with pytest.raises(ReaderError):
            NetCDFReader(bad).open()
