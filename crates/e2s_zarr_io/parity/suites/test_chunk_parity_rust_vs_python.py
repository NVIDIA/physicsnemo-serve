# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity test: Rust e2s_zarr_io chunking vs Python zarr v2 reference.

This test creates the *exact same* dataset using both:
  1. The Rust backend (via e2s_zarr_io Python bindings)
  2. Pure Python zarr v2 (mimicking AsyncZarrBackend's chunking logic)

Then it reads both stores back and compares:
  - Array shapes, dtypes, dimension names
  - Chunk grid (shape / chunks metadata)
  - Coordinate arrays (bitwise identical)
  - Data arrays (bitwise identical, chunk-by-chunk)

Usage:
    .venv/bin/python -m pytest crates/e2s_zarr_io/parity/suites/test_chunk_parity_rust_vs_python.py -v
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

# ---------------------------------------------------------------------------
# Test data definition — identical to the Rust test
# `chunk_data_bytes_match_expected_payloads_for_single_parallel_combo_writes`
# ---------------------------------------------------------------------------

TOTAL_COORDS: OrderedDict[str, np.ndarray] = OrderedDict(
    {
        "time": np.array([0, 1], dtype=np.int64),
        "lead_time": np.array([0, 6, 12], dtype=np.int64),
        "lat": np.array([10.0, 20.0], dtype=np.float64),
        "lon": np.array([30.0, 40.0], dtype=np.float64),
    }
)

PARALLEL_COORD_NAMES = ("time", "lead_time")
ARRAY_NAMES = ["t2m", "tcwv"]

# 6 write steps: time(2) × lead_time(3)
# Each payload is 4 float32 values (lat=2 × lon=2)
WRITE_STEPS: list[dict] = [
    {
        "time": 0,
        "lead_time": 0,
        "t2m": [1.0, 2.0, 3.0, 4.0],
        "tcwv": [10.0, 20.0, 30.0, 40.0],
    },
    {
        "time": 0,
        "lead_time": 6,
        "t2m": [5.0, 6.0, 7.0, 8.0],
        "tcwv": [50.0, 60.0, 70.0, 80.0],
    },
    {
        "time": 0,
        "lead_time": 12,
        "t2m": [9.0, 10.0, 11.0, 12.0],
        "tcwv": [90.0, 100.0, 110.0, 120.0],
    },
    {
        "time": 1,
        "lead_time": 0,
        "t2m": [13.0, 14.0, 15.0, 16.0],
        "tcwv": [130.0, 140.0, 150.0, 160.0],
    },
    {
        "time": 1,
        "lead_time": 6,
        "t2m": [17.0, 18.0, 19.0, 20.0],
        "tcwv": [170.0, 180.0, 190.0, 200.0],
    },
    {
        "time": 1,
        "lead_time": 12,
        "t2m": [21.0, 22.0, 23.0, 24.0],
        "tcwv": [210.0, 220.0, 230.0, 240.0],
    },
]


# ---------------------------------------------------------------------------
# Helper: create dataset using pure Python zarr v2
# (mimics AsyncZarrBackend._initialize_arrays + _write logic)
# ---------------------------------------------------------------------------


def _create_python_zarr_v2_reference(dataset_path: Path) -> None:
    """Create a zarr v2 store using Python zarr with the same chunking as AsyncZarrBackend."""
    store = zarr.storage.LocalStore(root=str(dataset_path))
    root = zarr.open_group(store=store, mode="w", zarr_format=2)

    # Dimension order: time, lead_time, lat, lon (same as TOTAL_COORDS key order)
    shape = tuple(len(v) for v in TOTAL_COORDS.values())  # (2, 3, 2, 2)
    dim_names = list(TOTAL_COORDS.keys())

    # Chunking: parallel dims → chunk_size=1, non-parallel → full axis length
    chunks: dict[str, int] = {}
    for key, val in TOTAL_COORDS.items():
        if key in PARALLEL_COORD_NAMES:
            chunks[key] = 1
        else:
            chunks[key] = len(val)
    chunk_shape = tuple(chunks[k] for k in dim_names)  # (1, 1, 2, 2)

    # Create coordinate arrays (no compression to match Rust raw bytes)
    for key, val in TOTAL_COORDS.items():
        root.create_array(
            name=key,
            data=val,
            chunks=val.shape,
            compressors=None,
        )
        root[key].attrs["_ARRAY_DIMENSIONS"] = [key]

    # Create data arrays (no compression, initialize with zeros, then write step-by-step)
    for name in ARRAY_NAMES:
        root.create_array(
            name=name,
            data=np.zeros(shape, dtype=np.float32),
            chunks=chunk_shape,
            compressors=None,
        )
        root[name].attrs["_ARRAY_DIMENSIONS"] = dim_names

    # Write step-by-step (same pattern as AsyncZarrBackend._write)
    time_vals = TOTAL_COORDS["time"]
    lt_vals = TOTAL_COORDS["lead_time"]

    for step in WRITE_STEPS:
        time_idx = int(np.where(time_vals == step["time"])[0][0])
        lt_idx = int(np.where(lt_vals == step["lead_time"])[0][0])

        for arr_name in ARRAY_NAMES:
            data = np.array(step[arr_name], dtype=np.float32).reshape(1, 1, 2, 2)
            root[arr_name][time_idx : time_idx + 1, lt_idx : lt_idx + 1, :, :] = data


# ---------------------------------------------------------------------------
# Helper: create dataset using Rust backend
# ---------------------------------------------------------------------------


def _create_rust_zarr_store(dataset_path: Path, e2s_zarr_io_module: ModuleType) -> None:
    """Create a zarr v2 store using the Rust e2s_zarr_io backend."""
    # Build parallel_coords dict matching what Python bindings expect
    parallel_coords = {key: TOTAL_COORDS[key].tolist() for key in PARALLEL_COORD_NAMES}

    with e2s_zarr_io_module.E2sZarrIoBackend(
        file_name=str(dataset_path),
        parallel_coords=parallel_coords,
    ) as backend:
        # Convert coord arrays to plain lists for the Rust bindings
        total_coords_dict = {key: val.tolist() for key, val in TOTAL_COORDS.items()}

        backend.add_array(total_coords_dict, ARRAY_NAMES)

        for step in WRITE_STEPS:
            step_coords = {
                "time": [step["time"]],
                "lead_time": [step["lead_time"]],
                "lat": TOTAL_COORDS["lat"].tolist(),
                "lon": TOTAL_COORDS["lon"].tolist(),
            }
            arrays = []
            for arr_name in ARRAY_NAMES:
                data = np.array(step[arr_name], dtype=np.float32)
                arrays.append(data)

            backend.write(arrays, step_coords, ARRAY_NAMES)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _read_zarr_array_raw(store_path: Path, array_name: str) -> tuple[np.ndarray, dict]:
    """Read a zarr array and its attrs from a v2 store."""
    store = zarr.storage.LocalStore(root=str(store_path))
    root = zarr.open_group(store=store, mode="r")
    arr = root[array_name]
    data = np.asarray(arr[:])
    attrs = dict(arr.attrs)
    return data, attrs


def _read_chunk_bytes(store_path: Path, array_name: str, chunk_key: str) -> bytes:
    """Read raw chunk bytes from disk."""
    chunk_path = store_path / array_name / chunk_key
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")
    return chunk_path.read_bytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def parity_stores(tmp_path: Path, e2s_zarr_io_module: ModuleType):
    """Create both Python and Rust zarr stores, return their paths."""
    py_path = tmp_path / "python_ref.zarr"
    rust_path = tmp_path / "rust_output.zarr"

    _create_python_zarr_v2_reference(py_path)
    _create_rust_zarr_store(rust_path, e2s_zarr_io_module)

    return py_path, rust_path


class TestChunkParityRustVsPython:
    """Compare Rust e2s_zarr_io output against Python zarr v2 reference."""

    def test_data_arrays_are_bitwise_identical(
        self, parity_stores: tuple[Path, Path]
    ) -> None:
        """Data arrays should have identical values after reading back."""
        py_path, rust_path = parity_stores

        for arr_name in ARRAY_NAMES:
            py_data, _ = _read_zarr_array_raw(py_path, arr_name)
            rust_data, _ = _read_zarr_array_raw(rust_path, arr_name)

            assert py_data.shape == rust_data.shape, (
                f"{arr_name}: shape mismatch: python={py_data.shape} rust={rust_data.shape}"
            )
            assert py_data.dtype == rust_data.dtype, (
                f"{arr_name}: dtype mismatch: python={py_data.dtype} rust={rust_data.dtype}"
            )
            np.testing.assert_array_equal(
                py_data,
                rust_data,
                err_msg=f"{arr_name}: data values differ between Python and Rust stores",
            )

    def test_chunk_bytes_are_bitwise_identical(
        self, parity_stores: tuple[Path, Path]
    ) -> None:
        """Raw chunk file bytes should match between Python and Rust stores."""
        py_path, rust_path = parity_stores

        time_vals = TOTAL_COORDS["time"]
        lt_vals = TOTAL_COORDS["lead_time"]

        for arr_name in ARRAY_NAMES:
            for ti in range(len(time_vals)):
                for li in range(len(lt_vals)):
                    chunk_key = f"{ti}.{li}.0.0"

                    rust_bytes = _read_chunk_bytes(rust_path, arr_name, chunk_key)
                    py_bytes = _read_chunk_bytes(py_path, arr_name, chunk_key)

                    assert rust_bytes == py_bytes, (
                        f"{arr_name}/{chunk_key}: chunk bytes differ. "
                        f"Rust={len(rust_bytes)}B, Python={len(py_bytes)}B"
                    )

    def test_coordinate_arrays_match(self, parity_stores: tuple[Path, Path]) -> None:
        """Coordinate arrays should have identical values."""
        py_path, rust_path = parity_stores

        for coord_name, expected in TOTAL_COORDS.items():
            py_data, _ = _read_zarr_array_raw(py_path, coord_name)
            rust_data, _ = _read_zarr_array_raw(rust_path, coord_name)

            assert py_data.shape == rust_data.shape, (
                f"coord {coord_name}: shape mismatch"
            )
            np.testing.assert_array_equal(
                py_data,
                rust_data,
                err_msg=f"coord {coord_name}: values differ",
            )

    def test_metadata_shape_and_chunks(self, parity_stores: tuple[Path, Path]) -> None:
        """Array metadata (shape, chunks) should match."""
        py_path, rust_path = parity_stores

        for arr_name in ARRAY_NAMES:
            py_store = zarr.open_group(
                store=zarr.storage.LocalStore(root=str(py_path)), mode="r"
            )
            rust_store = zarr.open_group(
                store=zarr.storage.LocalStore(root=str(rust_path)), mode="r"
            )

            py_arr = py_store[arr_name]
            rust_arr = rust_store[arr_name]

            assert py_arr.shape == rust_arr.shape, (
                f"{arr_name}: shape mismatch: python={py_arr.shape} rust={rust_arr.shape}"
            )
            assert py_arr.chunks == rust_arr.chunks, (
                f"{arr_name}: chunks mismatch: python={py_arr.chunks} rust={rust_arr.chunks}"
            )
            assert py_arr.dtype == rust_arr.dtype, (
                f"{arr_name}: dtype mismatch: python={py_arr.dtype} rust={rust_arr.dtype}"
            )

    def test_dimension_names_match(self, parity_stores: tuple[Path, Path]) -> None:
        """_ARRAY_DIMENSIONS attr should match between stores."""
        py_path, rust_path = parity_stores

        for arr_name in ARRAY_NAMES:
            _, py_attrs = _read_zarr_array_raw(py_path, arr_name)
            _, rust_attrs = _read_zarr_array_raw(rust_path, arr_name)

            py_dims = py_attrs.get("_ARRAY_DIMENSIONS")
            rust_dims = rust_attrs.get("_ARRAY_DIMENSIONS")

            assert py_dims == rust_dims, (
                f"{arr_name}: _ARRAY_DIMENSIONS mismatch: python={py_dims} rust={rust_dims}"
            )

    def test_all_expected_data_chunks_exist_in_both_stores(
        self, parity_stores: tuple[Path, Path]
    ) -> None:
        """Both stores should have exactly the same set of data chunk files."""
        py_path, rust_path = parity_stores

        # Metadata file names to exclude from comparison
        metadata_names = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}

        for arr_name in ARRAY_NAMES:
            py_chunks = set()
            rust_chunks = set()

            py_arr_dir = py_path / arr_name
            rust_arr_dir = rust_path / arr_name

            if py_arr_dir.exists():
                py_chunks = {
                    f.name
                    for f in py_arr_dir.iterdir()
                    if f.is_file() and f.name not in metadata_names
                }
            if rust_arr_dir.exists():
                rust_chunks = {
                    f.name
                    for f in rust_arr_dir.iterdir()
                    if f.is_file() and f.name not in metadata_names
                }

            assert py_chunks == rust_chunks, (
                f"{arr_name}: chunk file sets differ.\n"
                f"  Only in Python: {py_chunks - rust_chunks}\n"
                f"  Only in Rust:   {rust_chunks - py_chunks}"
            )

    def test_read_accepts_torch_device_object(
        self, tmp_path: Path, e2s_zarr_io_module: ModuleType
    ) -> None:
        """Rust backend read() accepts torch.device and returns expected tensor slice."""
        torch = pytest.importorskip("torch")

        dataset_path = tmp_path / "read_device_object.zarr"
        _create_python_zarr_v2_reference(dataset_path)

        backend = e2s_zarr_io_module.E2sZarrIoBackend(file_name=str(dataset_path))
        request_coords = {
            "time": [0],
            "lead_time": [0],
            "lat": TOTAL_COORDS["lat"].tolist(),
            "lon": TOTAL_COORDS["lon"].tolist(),
        }

        tensor, returned_coords = backend.read(
            request_coords, "t2m", torch.device("cpu")
        )

        expected = np.array(WRITE_STEPS[0]["t2m"], dtype=np.float32).reshape(1, 1, 2, 2)
        assert tuple(tensor.shape) == (1, 1, 2, 2)
        np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
        assert returned_coords == request_coords

    def test_repr_includes_dataset_root_and_zarr_format(
        self, tmp_path: Path, e2s_zarr_io_module: ModuleType
    ) -> None:
        """repr() should expose stable identity fields for debugging multiple backends."""

        dataset_path_a = tmp_path / "repr_backend_a.zarr"
        dataset_path_b = tmp_path / "repr_backend_b.zarr"
        with (
            e2s_zarr_io_module.E2sZarrIoBackend(
                file_name=str(dataset_path_a),
                zarr_format="v2",
            ) as backend_a,
            e2s_zarr_io_module.E2sZarrIoBackend(
                file_name=str(dataset_path_b),
                zarr_format="v3",
            ) as backend_b,
        ):
            repr_a = repr(backend_a)
            repr_b = repr(backend_b)

            assert f"dataset_root='{dataset_path_a}'" in repr_a
            assert "zarr_format='v2'" in repr_a
            assert "closed=False" in repr_a

            assert f"dataset_root='{dataset_path_b}'" in repr_b
            assert "zarr_format='v3'" in repr_b
            assert "closed=False" in repr_b
            assert repr_a != repr_b

        assert "closed=True" in repr(backend_a)
        assert "closed=True" in repr(backend_b)

    def test_context_manager_closes_and_rethrows_exception(
        self, tmp_path: Path, e2s_zarr_io_module: ModuleType
    ) -> None:
        """with backend: should close on unwind and preserve the original exception."""

        dataset_path = tmp_path / "context_manager_exception.zarr"
        parallel_coords = {
            key: TOTAL_COORDS[key].tolist() for key in PARALLEL_COORD_NAMES
        }
        total_coords_dict = {key: val.tolist() for key, val in TOTAL_COORDS.items()}
        backend = e2s_zarr_io_module.E2sZarrIoBackend(
            file_name=str(dataset_path),
            parallel_coords=parallel_coords,
        )

        with pytest.raises(RuntimeError, match="context-manager boom"):
            with backend as managed:
                assert managed is backend
                managed.add_array(total_coords_dict, ARRAY_NAMES)
                raise RuntimeError("context-manager boom")

        assert backend.is_closed()

    def test_context_manager_ignores_object_closed_on_exit(
        self, tmp_path: Path, e2s_zarr_io_module: ModuleType
    ) -> None:
        """Explicit close inside with-block should not fail during __exit__."""

        dataset_path = tmp_path / "context_manager_already_closed.zarr"
        parallel_coords = {
            key: TOTAL_COORDS[key].tolist() for key in PARALLEL_COORD_NAMES
        }
        total_coords_dict = {key: val.tolist() for key, val in TOTAL_COORDS.items()}
        with e2s_zarr_io_module.E2sZarrIoBackend(
            file_name=str(dataset_path),
            parallel_coords=parallel_coords,
        ) as backend:
            backend.add_array(total_coords_dict, ARRAY_NAMES)
            backend.close()

        assert backend.is_closed()

    def test_v3_utf8_coordinate_array_is_materialized_and_readable(
        self, tmp_path: Path, e2s_zarr_io_module: ModuleType
    ) -> None:
        """Rust backend should persist V3 UTF-8 coords with fixed-length UTF-32 metadata."""
        dataset_path = tmp_path / "v3_utf8_coord_parity.zarr"
        with e2s_zarr_io_module.E2sZarrIoBackend(
            file_name=str(dataset_path),
            zarr_format="v3",
        ) as backend:
            backend.add_array(
                {
                    "time": [0],
                    "member": ["control", "pert01"],
                },
                ["t2m"],
            )

        member_meta_path = dataset_path / "member" / "zarr.json"
        assert member_meta_path.exists(), (
            "expected v3 coordinate metadata at member/zarr.json"
        )
        member_meta = json.loads(member_meta_path.read_text(encoding="utf-8"))
        assert member_meta.get("data_type") == {
            "name": "fixed_length_utf32",
            "configuration": {"length_bytes": 28},
        }
        assert member_meta.get("fill_value") == ""

        member_chunk_path = dataset_path / "member" / "c" / "0"
        assert member_chunk_path.exists(), "expected V3 coordinate chunk at member/c/0"

        group = zarr.open_group(
            store=zarr.storage.LocalStore(root=str(dataset_path)),
            mode="r",
        )
        np.testing.assert_array_equal(
            np.asarray(group["member"][:]),
            np.array(["control", "pert01"]),
        )
