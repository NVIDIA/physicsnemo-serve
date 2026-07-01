# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dtype parity between Earth2Studio's Python ZarrBackend and e2s_zarr_io."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")


DATA_DTYPES = [
    "bool",
    "int8",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

TEMPORAL_DATA_DTYPES = [
    "datetime64[ns]",
    "timedelta64[ns]",
]


def _coords(coord_dtype: str) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        {
            "time": np.asarray([0], dtype=np.int64),
            "lead_time": np.asarray([0], dtype=np.int64),
            "lat": np.asarray([-1.0, 1.0], dtype=coord_dtype),
            "lon": np.asarray([10.0, 20.0], dtype=coord_dtype),
        }
    )


def _payload(dtype: str) -> np.ndarray:
    if dtype == "bool":
        values = np.asarray([False, True, True, False], dtype=dtype)
    elif dtype.startswith("datetime64"):
        values = np.asarray([0, 1, 2, 3], dtype="int64").astype(dtype)
    elif dtype.startswith("timedelta64"):
        values = np.asarray([0, 1, 2, 3], dtype="int64").astype(dtype)
    elif dtype.startswith("u"):
        values = np.asarray([0, 1, 2, 3], dtype=dtype)
    elif dtype.startswith("i"):
        values = np.asarray([-2, -1, 1, 2], dtype=dtype)
    else:
        values = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=dtype)
    return values.reshape(1, 1, 2, 2)


def _open(path: Path):
    return zarr.open_group(str(path), mode="r")


def _dimension_names(array) -> list[str]:
    metadata = getattr(array, "metadata", None)
    dimension_names = getattr(metadata, "dimension_names", None)
    if dimension_names is not None:
        return list(dimension_names)
    attrs = dict(getattr(array, "attrs", {}))
    return list(attrs.get("_ARRAY_DIMENSIONS", []))


def _write_python_store(path: Path, *, data_dtype: str, coord_dtype: str) -> None:
    from earth2studio.io import ZarrBackend

    coords = _coords(coord_dtype)
    data = _payload(data_dtype)
    backend = ZarrBackend(
        str(path),
        chunks={"time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True, "zarr_format": 3},
    )
    backend.add_array(coords, "field", data=torch.as_tensor(data))


def _write_python_store_empty_then_write(
    path: Path, *, write_dtype: str, coord_dtype: str
) -> None:
    from earth2studio.io import ZarrBackend

    coords = _coords(coord_dtype)
    data = _payload(write_dtype)
    backend = ZarrBackend(
        str(path),
        chunks={"time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True, "zarr_format": 3},
    )
    backend.add_array(coords, "field")
    backend.write(torch.as_tensor(data), coords, "field")


def _write_rust_store(
    path: Path, e2s_zarr_io_module: ModuleType, *, data_dtype: str, coord_dtype: str
) -> None:
    coords = _coords(coord_dtype)
    data = np.ascontiguousarray(_payload(data_dtype))
    with e2s_zarr_io_module.E2sZarrIoBackend(
        file_name=str(path),
        zarr_format="v3",
        parallel_coords={
            "time": coords["time"],
            "lead_time": coords["lead_time"],
        },
    ) as backend:
        backend.add_array(coords, "field", data=data)


def _write_python_zarr_reference(
    path: Path, *, data_dtype: str, coord_dtype: str
) -> None:
    coords = _coords(coord_dtype)
    data = _payload(data_dtype)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    for coord_name, values in coords.items():
        root.create_array(
            coord_name,
            data=values,
            chunks=values.shape,
            compressors=None,
            dimension_names=[coord_name],
        )
    root.create_array(
        "field",
        data=data,
        chunks=(1, 1, 2, 2),
        compressors=None,
        dimension_names=list(coords),
    )


def _write_rust_store_empty_then_write(
    path: Path, e2s_zarr_io_module: ModuleType, *, write_dtype: str, coord_dtype: str
) -> None:
    coords = _coords(coord_dtype)
    data = np.ascontiguousarray(_payload(write_dtype))
    with e2s_zarr_io_module.E2sZarrIoBackend(
        file_name=str(path),
        zarr_format="v3",
        parallel_coords={
            "time": coords["time"],
            "lead_time": coords["lead_time"],
        },
    ) as backend:
        backend.add_array(coords, "field")
        backend.write(data, coords, "field")


@pytest.mark.parametrize("data_dtype", DATA_DTYPES)
@pytest.mark.parametrize("coord_dtype", ["float32", "float64"])
def test_earth2studio_zarr_backend_dtype_parity(
    tmp_path: Path,
    e2s_zarr_io_module: ModuleType,
    data_dtype: str,
    coord_dtype: str,
) -> None:
    py_path = tmp_path / f"python-{data_dtype}-{coord_dtype}.zarr"
    rust_path = tmp_path / f"rust-{data_dtype}-{coord_dtype}.zarr"

    _write_python_store(py_path, data_dtype=data_dtype, coord_dtype=coord_dtype)
    _write_rust_store(
        rust_path,
        e2s_zarr_io_module,
        data_dtype=data_dtype,
        coord_dtype=coord_dtype,
    )

    py_root = _open(py_path)
    rust_root = _open(rust_path)

    for coord_name in ("time", "lead_time", "lat", "lon"):
        py_coord = py_root[coord_name]
        rust_coord = rust_root[coord_name]
        assert rust_coord.dtype == py_coord.dtype
        assert _dimension_names(rust_coord) == _dimension_names(py_coord)
        np.testing.assert_array_equal(
            np.asarray(rust_coord[:]), np.asarray(py_coord[:])
        )

    py_array = py_root["field"]
    rust_array = rust_root["field"]
    assert rust_array.shape == py_array.shape
    assert rust_array.chunks == py_array.chunks
    assert rust_array.dtype == py_array.dtype
    assert _dimension_names(rust_array) == _dimension_names(py_array)
    np.testing.assert_array_equal(np.asarray(rust_array[:]), np.asarray(py_array[:]))


@pytest.mark.parametrize("data_dtype", TEMPORAL_DATA_DTYPES)
def test_direct_zarr_temporal_data_dtype_parity(
    tmp_path: Path,
    e2s_zarr_io_module: ModuleType,
    data_dtype: str,
) -> None:
    py_path = tmp_path / f"python-{data_dtype}.zarr"
    rust_path = tmp_path / f"rust-{data_dtype}.zarr"

    _write_python_zarr_reference(py_path, data_dtype=data_dtype, coord_dtype="float32")
    _write_rust_store(
        rust_path,
        e2s_zarr_io_module,
        data_dtype=data_dtype,
        coord_dtype="float32",
    )

    py_array = _open(py_path)["field"]
    rust_array = _open(rust_path)["field"]
    assert rust_array.shape == py_array.shape
    assert rust_array.chunks == py_array.chunks
    assert rust_array.dtype == py_array.dtype
    assert _dimension_names(rust_array) == _dimension_names(py_array)
    np.testing.assert_array_equal(np.asarray(rust_array[:]), np.asarray(py_array[:]))


@pytest.mark.parametrize("write_dtype", ["float16", "float64", "int16", "uint16"])
def test_empty_registration_write_casts_to_registered_dtype(
    tmp_path: Path,
    e2s_zarr_io_module: ModuleType,
    write_dtype: str,
) -> None:
    py_path = tmp_path / f"python-empty-then-{write_dtype}.zarr"
    rust_path = tmp_path / f"rust-empty-then-{write_dtype}.zarr"

    _write_python_store_empty_then_write(
        py_path, write_dtype=write_dtype, coord_dtype="float32"
    )
    _write_rust_store_empty_then_write(
        rust_path,
        e2s_zarr_io_module,
        write_dtype=write_dtype,
        coord_dtype="float32",
    )

    py_array = _open(py_path)["field"]
    rust_array = _open(rust_path)["field"]
    assert py_array.dtype == np.dtype("float32")
    assert rust_array.dtype == py_array.dtype
    np.testing.assert_array_equal(np.asarray(rust_array[:]), np.asarray(py_array[:]))
