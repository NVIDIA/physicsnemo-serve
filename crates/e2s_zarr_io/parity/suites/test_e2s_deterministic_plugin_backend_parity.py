# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity and IO timing for the e2s-deterministic plugin backend selector."""

from __future__ import annotations

from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from typing import Any
import uuid

import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_DIR = REPO_ROOT / "python"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PLUGIN_PATH = REPO_ROOT / "plugins" / "e2s-deterministic" / "workflow.py"

for path in (REPO_ROOT, PYTHON_DIR, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from plugin_sdk import ExecutionContext, OutputRegistry  # noqa: E402

ARRAY_NAMES = ["t2m", "tcwv"]
STEP_DELTA_NS = 6 * 60 * 60 * 1_000_000_000


def _load_plugin_module() -> ModuleType:
    module_name = f"e2s_deterministic_backend_parity_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load plugin module from {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _open_zarr_group(path: Path) -> Any:
    store = zarr.storage.LocalStore(root=str(path))
    return zarr.open_group(store=store, mode="r")


def _as_names(array_name: Any) -> list[str]:
    if isinstance(array_name, str):
        return [array_name]
    if hasattr(array_name, "tolist"):
        array_name = array_name.tolist()
    if isinstance(array_name, tuple):
        array_name = list(array_name)
    if isinstance(array_name, list):
        return [str(name) for name in array_name]
    return [str(array_name)]


def _slice_for_values(axis: np.ndarray, values: Any) -> slice | list[int]:
    requested = np.asarray(values)
    indices: list[int] = []
    for value in requested:
        matches = np.where(axis == value)[0]
        if len(matches) != 1:
            raise ValueError(
                f"coordinate value {value!r} is not registered exactly once"
            )
        indices.append(int(matches[0]))

    contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
    if contiguous:
        return slice(indices[0], indices[-1] + 1)
    return indices


class _PythonZarrBackend:
    """Small Earth2Studio ZarrBackend stand-in that writes real local Zarr v2."""

    def __init__(
        self,
        file_name: str | None = None,
        path: str | None = None,
        chunks: dict[str, int] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.file_name = file_name or path
        if self.file_name is None:
            raise ValueError("file_name or path is required")
        self.chunks = chunks or {
            "ensemble": 1,
            "time": 1,
            "lead_time": 1,
            "variable": 1,
        }
        overwrite = bool((backend_kwargs or {}).get("overwrite", False))
        mode = "w" if overwrite else "a"
        store = zarr.storage.LocalStore(root=str(self.file_name))
        self.root = zarr.open_group(store=store, mode=mode, zarr_format=2)
        self.coords: OrderedDict[str, np.ndarray] | None = None

    def add_array(
        self, coords: OrderedDict[str, Any], array_name: Any, data: Any = None
    ) -> None:
        self.coords = OrderedDict(
            (str(key), np.asarray(value)) for key, value in coords.items()
        )
        dim_names = list(self.coords)
        shape = tuple(len(values) for values in self.coords.values())
        chunk_shape = tuple(
            1 if self.chunks.get(dim_name) == 1 else len(self.coords[dim_name])
            for dim_name in dim_names
        )
        names = _as_names(array_name)

        for coord_name, values in self.coords.items():
            self.root.create_array(
                name=coord_name,
                data=values,
                chunks=values.shape,
                compressors=None,
            )
            self.root[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]

        for name in names:
            self.root.create_array(
                name=name,
                shape=shape,
                dtype=np.float32,
                chunks=chunk_shape,
                compressors=None,
                fill_value=0,
            )
            self.root[name].attrs["_ARRAY_DIMENSIONS"] = dim_names

        if data is not None:
            self.write(data, coords, names)

    def write(self, x: Any, coords: OrderedDict[str, Any], array_name: Any) -> None:
        if self.coords is None:
            raise RuntimeError("add_array must be called before write")
        arrays = list(x) if isinstance(x, list | tuple) else [x]
        names = _as_names(array_name)
        selection = tuple(
            _slice_for_values(axis, coords[dim_name])
            for dim_name, axis in self.coords.items()
        )
        selection_shape = tuple(len(np.asarray(coords[dim])) for dim in self.coords)

        for name, array in zip(names, arrays, strict=True):
            payload = np.asarray(array, dtype=np.float32).reshape(selection_shape)
            self.root[name][selection] = payload

    def close(self) -> None:
        return None

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def __iter__(self):
        return iter(self.root)


def _maybe_cuda_synchronize(write_device: str) -> None:
    if write_device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def _payload_arrays_for_device(
    base: np.ndarray, write_device: str, output_dtype: str
) -> list[Any]:
    dtype = np.dtype(output_dtype)
    arrays = [
        np.ascontiguousarray(base, dtype=dtype),
        np.ascontiguousarray(base + np.float32(100.0), dtype=dtype),
    ]
    if write_device != "cuda":
        return arrays

    import torch

    return [torch.as_tensor(array, device="cuda") for array in arrays]


def _install_fake_earth2studio(
    monkeypatch: pytest.MonkeyPatch,
    timings: dict[str, float | str],
    *,
    write_device: str,
    output_dtype: str,
) -> None:
    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = _PythonZarrBackend
    fake_io.IOBackend = object

    class _FakePackage:
        pass

    class _FakeModel:
        @staticmethod
        def load_default_package() -> _FakePackage:
            return _FakePackage()

        @staticmethod
        def load_model(_package: _FakePackage) -> object:
            return object()

    fake_models_px = ModuleType("earth2studio.models.px")
    fake_models_px.DLWP = _FakeModel
    fake_models_px.FCN = _FakeModel
    fake_models_px.FCN3 = _FakeModel

    class _FakeGFS:
        pass

    fake_data = ModuleType("earth2studio.data")
    fake_data.GFS = _FakeGFS

    fake_run = ModuleType("earth2studio.run")

    def deterministic(
        time: list[str],
        nsteps: int,
        _prognostic: object,
        _data: object,
        io: Any,
        **_kwargs: Any,
    ) -> Any:
        start_times = np.asarray(time, dtype="datetime64[ns]").astype(np.int64)
        lead_times = np.arange(nsteps, dtype=np.int64) * STEP_DELTA_NS
        lat = np.linspace(-90.0, 90.0, 16, dtype=np.float64)
        lon = np.linspace(0.0, 360.0, 32, endpoint=False, dtype=np.float64)
        coords = OrderedDict(
            {
                "time": start_times,
                "lead_time": lead_times,
                "lat": lat,
                "lon": lon,
            }
        )

        start = perf_counter()
        io.add_array(coords, ARRAY_NAMES)
        timings["add_array_sec"] = perf_counter() - start

        write_total = 0.0
        lat_grid = lat.reshape(1, 1, len(lat), 1).astype(np.float32)
        lon_grid = lon.reshape(1, 1, 1, len(lon)).astype(np.float32)
        time_grid = np.arange(len(start_times), dtype=np.float32).reshape(-1, 1, 1, 1)

        for step_index, lead_time in enumerate(lead_times):
            step_coords = OrderedDict(
                {
                    "time": start_times,
                    "lead_time": np.asarray([lead_time], dtype=np.int64),
                    "lat": lat,
                    "lon": lon,
                }
            )
            base = (
                time_grid + np.float32(step_index) + lat_grid * 0.01 + lon_grid * 0.001
            )
            arrays = _payload_arrays_for_device(base, write_device, output_dtype)
            _maybe_cuda_synchronize(write_device)
            start = perf_counter()
            io.write(arrays, step_coords, ARRAY_NAMES)
            _maybe_cuda_synchronize(write_device)
            write_total += perf_counter() - start

        timings["write_sec"] = write_total
        timings["write_count"] = float(nsteps)
        timings["write_device"] = write_device
        timings["output_dtype"] = output_dtype
        return io

    fake_run.deterministic = deterministic

    monkeypatch.setitem(sys.modules, "earth2studio", ModuleType("earth2studio"))
    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.models.px", fake_models_px)
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)
    monkeypatch.setitem(sys.modules, "earth2studio.run", fake_run)


def _run_e2s_deterministic_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    e2s_zarr_io_module: ModuleType,
    write_device: str = "cpu",
    output_dtype: str = "float32",
) -> tuple[Path, dict[str, float | str]]:
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", backend)
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", e2s_zarr_io_module)

    timings: dict[str, float | str] = {}
    _install_fake_earth2studio(
        monkeypatch,
        timings,
        write_device=write_device,
        output_dtype=output_dtype,
    )

    module = _load_plugin_module()
    run_dir = tmp_path / backend
    ctx = ExecutionContext(
        run_id=f"{backend}-run",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"device_kind": "cpu"},
        services={},
    )

    start = perf_counter()
    result = module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00", "2024-01-01T06:00:00"],
            "nsteps": 4,
            "model_type": "fcn",
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
        },
        ctx,
    )
    workflow_wall_sec = perf_counter() - start

    dataset_path = Path(result["dataset_path"])
    metrics: dict[str, float | str] = {
        "backend": backend,
        "workflow_wall_sec": workflow_wall_sec,
        "add_array_sec": timings["add_array_sec"],
        "write_sec": timings["write_sec"],
        "write_count": timings["write_count"],
        "write_device": timings["write_device"],
        "output_dtype": timings["output_dtype"],
        "dataset_path": str(dataset_path),
    }
    return dataset_path, metrics


def _assert_zarr_outputs_match(py_path: Path, rust_path: Path) -> None:
    py_root = _open_zarr_group(py_path)
    rust_root = _open_zarr_group(rust_path)

    expected_keys = {"time", "lead_time", "lat", "lon", *ARRAY_NAMES}
    assert set(py_root.keys()) == expected_keys
    assert set(rust_root.keys()) == expected_keys

    for key in ("time", "lead_time", "lat", "lon"):
        np.testing.assert_array_equal(
            np.asarray(py_root[key][:]), np.asarray(rust_root[key][:])
        )

    for name in ARRAY_NAMES:
        py_array = py_root[name]
        rust_array = rust_root[name]
        assert py_array.shape == rust_array.shape
        assert py_array.chunks == rust_array.chunks
        assert py_array.dtype == rust_array.dtype
        assert dict(py_array.attrs) == dict(rust_array.attrs)
        np.testing.assert_array_equal(
            np.asarray(py_array[:]), np.asarray(rust_array[:])
        )


@pytest.mark.parametrize("output_dtype", ["float16", "float32", "float64"])
def test_e2s_deterministic_plugin_python_rust_zarr_parity_and_io_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    e2s_zarr_io_module: ModuleType,
    output_dtype: str,
) -> None:
    py_path, py_metrics = _run_e2s_deterministic_plugin(
        tmp_path,
        monkeypatch,
        backend="python",
        e2s_zarr_io_module=e2s_zarr_io_module,
        output_dtype=output_dtype,
    )
    rust_path, rust_metrics = _run_e2s_deterministic_plugin(
        tmp_path,
        monkeypatch,
        backend="rust",
        e2s_zarr_io_module=e2s_zarr_io_module,
        output_dtype=output_dtype,
    )

    _assert_zarr_outputs_match(py_path, rust_path)

    performance_report = {
        "python": py_metrics,
        "rust": rust_metrics,
        "rust_vs_python_write_speedup": (
            float(py_metrics["write_sec"]) / float(rust_metrics["write_sec"])
            if float(rust_metrics["write_sec"]) > 0.0
            else None
        ),
        "rust_vs_python_workflow_speedup": (
            float(py_metrics["workflow_wall_sec"])
            / float(rust_metrics["workflow_wall_sec"])
            if float(rust_metrics["workflow_wall_sec"]) > 0.0
            else None
        ),
    }
    report_path = tmp_path / "e2s_deterministic_io_performance.json"
    report_path.write_text(json.dumps(performance_report, indent=2), encoding="utf-8")
    print(json.dumps(performance_report, indent=2))

    assert py_metrics["write_sec"] > 0.0
    assert rust_metrics["write_sec"] > 0.0
    assert py_metrics["workflow_wall_sec"] > 0.0
    assert rust_metrics["workflow_wall_sec"] > 0.0


def test_e2s_deterministic_plugin_python_cpu_rust_cuda_zarr_parity_and_io_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    e2s_zarr_io_module: ModuleType,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for Rust CUDA IO benchmark")

    py_path, py_metrics = _run_e2s_deterministic_plugin(
        tmp_path,
        monkeypatch,
        backend="python",
        e2s_zarr_io_module=e2s_zarr_io_module,
        write_device="cpu",
    )
    try:
        rust_path, rust_metrics = _run_e2s_deterministic_plugin(
            tmp_path,
            monkeypatch,
            backend="rust",
            e2s_zarr_io_module=e2s_zarr_io_module,
            write_device="cuda",
        )
    except RuntimeError as exc:
        if "cuda runtime path is unavailable" in str(exc):
            pytest.skip(
                "e2s_zarr_io CUDA runtime path is unavailable in this environment"
            )
        raise

    _assert_zarr_outputs_match(py_path, rust_path)

    performance_report = {
        "python": py_metrics,
        "rust": rust_metrics,
        "rust_vs_python_write_speedup": (
            float(py_metrics["write_sec"]) / float(rust_metrics["write_sec"])
            if float(rust_metrics["write_sec"]) > 0.0
            else None
        ),
        "rust_vs_python_workflow_speedup": (
            float(py_metrics["workflow_wall_sec"])
            / float(rust_metrics["workflow_wall_sec"])
            if float(rust_metrics["workflow_wall_sec"]) > 0.0
            else None
        ),
    }
    report_path = tmp_path / "e2s_deterministic_cuda_io_performance.json"
    report_path.write_text(json.dumps(performance_report, indent=2), encoding="utf-8")
    print(json.dumps(performance_report, indent=2))

    assert py_metrics["write_device"] == "cpu"
    assert rust_metrics["write_device"] == "cuda"
    assert py_metrics["write_sec"] > 0.0
    assert rust_metrics["write_sec"] > 0.0
