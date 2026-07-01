# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused execution-path tests for parity workflow catalog helpers."""

from __future__ import annotations

from collections import OrderedDict
import builtins
from pathlib import Path
import sys
import types
from typing import Any
import warnings

import pytest

from parity.utils import workflow_catalog as workflow_catalog_module


def _base_case_spec() -> dict[str, Any]:
    return {
        "deterministic_seed": 3,
        "start_times": ["2026-01-01T00:00:00Z"],
        "num_steps": 2,
        "step_delta": "1h",
        "output_array_names": ["temperature", "pressure"],
        "parallel_coords": None,
        "zarr_format": "v2",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
    }


def _coords_and_arrays(np_module: Any) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    coords = OrderedDict(
        {
            "time": np_module.asarray([1], dtype=np_module.int64),
            "lead_time": np_module.asarray([0, 1], dtype=np_module.int64),
            "lat": np_module.asarray([10.0, 20.0], dtype=np_module.float32),
            "lon": np_module.asarray([30.0, 40.0], dtype=np_module.float32),
        }
    )
    arrays = {
        "temperature": np_module.zeros((1, 2, 2, 2), dtype=np_module.float32),
        "pressure": np_module.ones((1, 2, 2, 2), dtype=np_module.float32),
    }
    return coords, arrays


def test_parse_step_delta_parses_and_validates() -> None:
    np = pytest.importorskip("numpy")
    assert workflow_catalog_module._parse_step_delta("6h") == np.timedelta64(6, "h")
    with pytest.raises(ValueError, match="unsupported step_delta unit"):
        workflow_catalog_module._parse_step_delta("5x")
    with pytest.raises(ValueError, match="integer-prefixed"):
        workflow_catalog_module._parse_step_delta("xh")
    with pytest.raises(ValueError, match="must be > 0"):
        workflow_catalog_module._parse_step_delta("0h")


def test_discover_root_raises_when_markers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow_catalog_module.Path, "exists", lambda _self: False)
    with pytest.raises(
        RuntimeError, match="could not discover earth2studio repository root"
    ):
        workflow_catalog_module._discover_earth2studio_root()


def test_ensure_import_path_inserts_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "e2s_fake_root"
    monkeypatch.setattr(
        workflow_catalog_module, "_discover_earth2studio_root", lambda: fake_root
    )
    if str(fake_root) in sys.path:
        sys.path.remove(str(fake_root))
    workflow_catalog_module._ensure_earth2studio_import_path()
    assert str(fake_root) in sys.path


def test_resolve_parallel_coords_prefers_explicit_case_spec() -> None:
    np = pytest.importorskip("numpy")
    total_coords, _ = _coords_and_arrays(np)
    explicit = {"lead_time": [0], "ensemble": [7]}
    resolved = workflow_catalog_module._resolve_parallel_coords(
        {"parallel_coords": explicit},
        total_coords,
    )
    assert sorted(resolved.keys()) == ["ensemble", "lead_time"]


def test_build_workflow_inputs_constructs_expected_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    fake_torch = types.SimpleNamespace(from_numpy=lambda arr: arr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    total_coords, tensors = workflow_catalog_module._build_workflow_inputs(
        _base_case_spec()
    )
    assert sorted(total_coords.keys()) == ["lat", "lead_time", "lon", "time"]
    assert sorted(tensors.keys()) == ["pressure", "temperature"]
    assert tuple(tensors["temperature"].shape) == (1, 2, 8, 16)


def test_build_workflow_inputs_falls_back_to_numpy_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    original_import = builtins.__import__

    def failing_torch_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_torch_import)
    total_coords, tensors = workflow_catalog_module._build_workflow_inputs(
        _base_case_spec()
    )

    assert sorted(total_coords.keys()) == ["lat", "lead_time", "lon", "time"]
    assert isinstance(tensors["temperature"], np.ndarray)
    assert tuple(tensors["temperature"].shape) == (1, 2, 8, 16)


def test_build_workflow_inputs_handles_utc_z_without_timezone_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    fake_torch = types.SimpleNamespace(from_numpy=lambda arr: arr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    case_spec = _base_case_spec()
    case_spec["start_times"] = ["2026-01-01T00:00:00Z", "2026-01-01T06:00:00Z"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total_coords, tensors = workflow_catalog_module._build_workflow_inputs(
            case_spec
        )

    timezone_warnings = [
        warning
        for warning in caught
        if "no explicit representation of timezones available for np.datetime64"
        in str(warning.message)
    ]
    assert not timezone_warnings
    assert tuple(total_coords["time"].shape) == (2,)
    assert tuple(tensors["temperature"].shape) == (2, 2, 8, 16)


def test_import_earth2studio_backends_returns_imported_backend_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_catalog_module, "_ensure_earth2studio_import_path", lambda: None
    )
    earth2studio_module = types.ModuleType("earth2studio")
    io_module = types.ModuleType("earth2studio.io")
    async_module = types.ModuleType("earth2studio.io.async_zarr")
    zarr_module = types.ModuleType("earth2studio.io.zarr")

    class FakeAsyncBackend:
        pass

    class FakeSyncBackend:
        pass

    async_module.AsyncZarrBackend = FakeAsyncBackend
    zarr_module.ZarrBackend = FakeSyncBackend
    monkeypatch.setitem(sys.modules, "earth2studio", earth2studio_module)
    monkeypatch.setitem(sys.modules, "earth2studio.io", io_module)
    monkeypatch.setitem(sys.modules, "earth2studio.io.async_zarr", async_module)
    monkeypatch.setitem(sys.modules, "earth2studio.io.zarr", zarr_module)

    sync_backend, async_backend = (
        workflow_catalog_module._import_earth2studio_backends()
    )
    assert sync_backend is FakeSyncBackend
    assert async_backend is FakeAsyncBackend


def test_run_with_py_sync_backend_invokes_expected_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    coords, arrays = _coords_and_arrays(np)
    calls: dict[str, Any] = {"writes": 0}

    class FakeSyncBackend:
        def __init__(self, *, file_name: str, chunks: OrderedDict[str, int]) -> None:
            calls["file_name"] = file_name
            calls["chunks"] = chunks

        def add_array(self, total_coords: object, array_names: object) -> None:
            calls["add_array"] = (total_coords, array_names)

        def write(
            self, _step_tensors: object, _step_coords: object, _array_names: object
        ) -> None:
            calls["writes"] = int(calls["writes"]) + 1

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(
        workflow_catalog_module,
        "_import_earth2studio_backends",
        lambda: (FakeSyncBackend, object()),
    )
    monkeypatch.setattr(
        workflow_catalog_module,
        "_build_workflow_inputs",
        lambda _case_spec: (coords, arrays),
    )

    out = tmp_path / "rust_unused_sync.zarr"
    workflow_catalog_module._run_with_py_sync_backend(_base_case_spec(), out)
    assert calls["file_name"] == str(out)
    assert int(calls["writes"]) == 2


def test_run_with_py_async_backend_invokes_expected_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    coords, arrays = _coords_and_arrays(np)
    calls: dict[str, Any] = {"writes": 0, "closed": False}

    class FakeAsyncBackend:
        def __init__(
            self, file_name: str, parallel_coords: dict[str, Any], blocking: bool
        ) -> None:
            calls["file_name"] = file_name
            calls["parallel_coords"] = parallel_coords
            calls["blocking"] = blocking

        def write(
            self, _step_tensors: object, _step_coords: object, _array_names: object
        ) -> None:
            calls["writes"] = int(calls["writes"]) + 1

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(
        workflow_catalog_module,
        "_import_earth2studio_backends",
        lambda: (object(), FakeAsyncBackend),
    )
    monkeypatch.setattr(
        workflow_catalog_module,
        "_build_workflow_inputs",
        lambda _case_spec: (coords, arrays),
    )

    out = tmp_path / "rust_unused_async.zarr"
    workflow_catalog_module._run_with_py_async_backend(_base_case_spec(), out)
    assert calls["file_name"] == str(out)
    assert int(calls["writes"]) == 2
    assert calls["closed"] is True


def test_run_with_rust_backend_raises_when_extension_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        workflow_catalog_module, "_ensure_earth2studio_import_path", lambda: None
    )

    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "e2s_zarr_io":
            raise ImportError("module not found")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(
        RuntimeError, match="requires the e2s_zarr_io Python extension module"
    ):
        workflow_catalog_module._run_with_rust_backend(
            _base_case_spec(),
            tmp_path / "rust_missing.zarr",
        )


def test_run_with_rust_backend_raises_when_backend_class_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        workflow_catalog_module, "_ensure_earth2studio_import_path", lambda: None
    )
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", types.ModuleType("e2s_zarr_io"))
    with pytest.raises(RuntimeError, match="E2sZarrIoBackend is not exposed"):
        workflow_catalog_module._run_with_rust_backend(
            _base_case_spec(),
            tmp_path / "rust_missing_class.zarr",
        )


def test_run_default_workflow_rejects_unsupported_backend_kind(tmp_path: Path) -> None:
    case_spec = {"backend_kind": "unsupported"}
    with pytest.raises(ValueError, match="unsupported backend_kind"):
        workflow_catalog_module.run_default_workflow(
            case_spec, tmp_path / "unused.zarr"
        )


def test_run_with_rust_backend_emits_numpy_lead_time_subset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(
        workflow_catalog_module,
        "_ensure_earth2studio_import_path",
        lambda: None,
    )
    captured_step_coords: list[dict[str, object]] = []

    class FakeRustBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def add_array(self, _coords: object, _array_name: object) -> None:
            pass

        def write(self, _x: object, coords: object, _array_name: object) -> None:
            assert isinstance(coords, dict)
            captured_step_coords.append(dict(coords))

        def close(self) -> None:
            pass

    fake_module = types.SimpleNamespace(E2sZarrIoBackend=FakeRustBackend)
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", fake_module)
    total_coords = OrderedDict(
        {
            "time": np.asarray(
                [np.datetime64("2026-01-01T00:00:00", "ns")],
                dtype="datetime64[ns]",
            ),
            "lead_time": np.asarray(
                [np.timedelta64(0, "ns"), np.timedelta64(1, "h")],
                dtype="timedelta64[ns]",
            ),
            "lat": np.asarray([10.0, 20.0], dtype=np.float32),
            "lon": np.asarray([30.0, 40.0], dtype=np.float32),
        }
    )
    arrays = {
        "temperature": np.zeros((1, 2, 2, 2), dtype=np.float32),
    }
    monkeypatch.setattr(
        workflow_catalog_module,
        "_build_workflow_inputs",
        lambda _case_spec: (total_coords, arrays),
    )

    workflow_catalog_module._run_with_rust_backend(
        _base_case_spec(),
        tmp_path / "rust_coord_subset.zarr",
    )
    assert len(captured_step_coords) == 2
    for step_coords in captured_step_coords:
        lead_time = step_coords["lead_time"]
        assert isinstance(lead_time, np.ndarray)
        assert str(lead_time.dtype) == "timedelta64[ns]"
