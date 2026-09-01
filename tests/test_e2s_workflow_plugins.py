# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the new e2s-* plugins that map to GitHub example_workflows."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PYTHON_DIR = REPO_ROOT / "python"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from plugin_sdk import (  # noqa: E402
    ExecutionContext,
    OutputRegistry,
    PrepareContext,
    RawRequest,
)


def _install_fake_rust_zarr_io(monkeypatch, *, rust_backend_calls=None):
    """Install a fake e2s_zarr_io module that records Rust backend usage."""

    class FakeRustZarrBackend:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.path = Path(kwargs["file_name"])
            self.closed = False
            self.record = {
                "kwargs": dict(kwargs),
                "add_array": [],
                "write": [],
                "close": [],
            }
            if rust_backend_calls is not None:
                rust_backend_calls.append(self.record)

        def add_array(self, coords, array_name, data=None):
            self.record["add_array"].append(
                {"coords": dict(coords), "array_name": array_name, "data": data}
            )
            if data is not None:
                self.write(data, coords, array_name)

        def write(self, x, coords, array_name):
            self.record["write"].append(
                {"x": x, "coords": dict(coords), "array_name": array_name}
            )

        def close(self, timeout_seconds=None):
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / ".written").write_text("ok", encoding="utf-8")
            self.closed = True
            self.record["close"].append(timeout_seconds)
            return {"total_close_ns": 1}

        def is_closed(self):
            return self.closed

    fake_module = ModuleType("e2s_zarr_io")
    fake_module.E2sZarrIoBackend = FakeRustZarrBackend
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", fake_module)


def test_rust_zarr_adapter_normalizes_host_arrays_to_c_contiguous():
    import numpy as np
    from e2s_workflow import _normalize_write_arrays_for_rust

    transposed = np.arange(6, dtype=np.float32).reshape(2, 3).T
    assert not transposed.flags.c_contiguous

    normalized = _normalize_write_arrays_for_rust([transposed])

    assert len(normalized) == 1
    assert normalized[0].flags.c_contiguous
    np.testing.assert_array_equal(normalized[0], transposed)


def test_rust_zarr_adapter_preserves_numpy_coordinate_dtypes(
    monkeypatch, tmp_path: Path
):
    import numpy as np
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    rust_backend_calls: list[dict[str, object]] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)

    io = create_zarr_backend(str(tmp_path / "coords.zarr"))
    io.add_array(
        {
            "lat": np.asarray([10.0, 20.0], dtype=np.float32),
            "lon": np.asarray([100, 110], dtype=np.int32),
        },
        "t2m",
    )

    captured_coords = rust_backend_calls[0]["add_array"][0]["coords"]
    assert captured_coords["lat"].dtype == np.dtype("float32")
    assert captured_coords["lon"].dtype == np.dtype("int32")
    assert rust_backend_calls[0]["kwargs"]["zarr_format"] == "v3"


def test_create_zarr_backend_rejects_v2_for_earth2studio(monkeypatch, tmp_path: Path):
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    _install_fake_rust_zarr_io(monkeypatch)

    with pytest.raises(ValueError, match="only support Zarr v3"):
        create_zarr_backend(str(tmp_path / "coords.zarr"), zarr_format="v2")


def test_create_zarr_backend_defaults_to_rust(monkeypatch, tmp_path: Path):
    from e2s_workflow import create_zarr_backend

    monkeypatch.delenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", raising=False)
    rust_backend_calls: list[dict[str, object]] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)

    io = create_zarr_backend(str(tmp_path / "coords.zarr"))
    io.add_array({}, "t2m")

    assert rust_backend_calls[0]["kwargs"]["file_name"] == str(tmp_path / "coords.zarr")
    assert rust_backend_calls[0]["kwargs"]["zarr_format"] == "v3"


def test_rust_zarr_adapter_sets_default_parallel_names_from_inferred_coords(
    monkeypatch, tmp_path: Path
):
    import numpy as np
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    rust_backend_calls: list[dict[str, object]] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)

    io = create_zarr_backend(
        str(tmp_path / "coords.zarr"),
        chunks={"ensemble": 1, "time": 1, "lead_time": 1},
    )
    io.add_array(
        {
            "ensemble": np.asarray([0, 1], dtype=np.int64),
            "time": np.asarray(["2026-01-01"], dtype="datetime64[D]"),
            "lead_time": np.asarray([0, 6], dtype="timedelta64[h]"),
            "lat": np.asarray([10.0, 20.0], dtype=np.float32),
        },
        "t2m",
    )

    backend_kwargs = rust_backend_calls[0]["kwargs"]
    assert list(backend_kwargs["parallel_coords"]) == [
        "ensemble",
        "time",
        "lead_time",
    ]
    assert backend_kwargs["default_parallel_coord_names"] == [
        "ensemble",
        "time",
        "lead_time",
    ]


def test_rust_zarr_adapter_rejects_unsupported_explicit_chunks(
    monkeypatch, tmp_path: Path
):
    import numpy as np
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    _install_fake_rust_zarr_io(monkeypatch)

    io = create_zarr_backend(
        str(tmp_path / "coords.zarr"),
        chunks={"time": 2, "lead_time": 2},
    )

    with pytest.raises(ValueError, match="supports explicit chunks only as 1"):
        io.add_array(
            {
                "time": np.asarray([0, 1, 2, 3], dtype=np.int64),
                "lead_time": np.asarray([0, 6, 12, 18], dtype=np.int64),
            },
            "t2m",
        )


def test_rust_zarr_adapter_rejects_explicit_chunks_without_parallel_axis(
    monkeypatch, tmp_path: Path
):
    import numpy as np
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    _install_fake_rust_zarr_io(monkeypatch)

    io = create_zarr_backend(
        str(tmp_path / "coords.zarr"),
        chunks={"time": 2, "lead_time": 2},
    )

    with pytest.raises(
        ValueError, match="must include at least one registered dimension"
    ):
        io.add_array(
            {
                "time": np.asarray([0, 1], dtype=np.int64),
                "lead_time": np.asarray([0, 6], dtype=np.int64),
            },
            "t2m",
        )


def test_rust_zarr_adapter_overwrite_true_removes_existing_path(
    monkeypatch, tmp_path: Path
):
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    stale_path = tmp_path / "stale.zarr"
    stale_path.mkdir()
    (stale_path / "old").write_text("stale", encoding="utf-8")

    rust_backend_calls: list[dict[str, object]] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)

    io = create_zarr_backend(
        str(stale_path),
        backend_kwargs={"overwrite": True},
    )
    io.add_array({}, "t2m")

    assert not (stale_path / "old").exists()
    assert rust_backend_calls[0]["kwargs"]["file_name"] == str(stale_path)


@pytest.mark.parametrize("format_key", ["zarr_format", "zarr_version"])
def test_rust_zarr_adapter_accepts_zarr_format_in_backend_kwargs(
    monkeypatch, tmp_path: Path, format_key: str
):
    from e2s_workflow import create_zarr_backend

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    rust_backend_calls: list[dict[str, object]] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)

    io = create_zarr_backend(
        str(tmp_path / "coords.zarr"),
        backend_kwargs={"overwrite": True, format_key: 3},
    )
    io.add_array({}, "t2m")

    assert rust_backend_calls[0]["kwargs"]["zarr_format"] == "v3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_subprocess_earth2_fakes(
    tmp_path: Path,
    monkeypatch,
    *,
    expected_processes: int,
) -> Path:
    """Install disk-backed fakes inherited by fresh item interpreters."""
    fake_root = tmp_path / "subprocess-fakes"
    sync_dir = tmp_path / "inference-sync"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "sitecustomize.py").write_text(
        """
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

signal.signal(signal.SIGUSR1, signal.SIG_IGN)

earth2studio = ModuleType("earth2studio")
earth2studio.__path__ = []
run = ModuleType("earth2studio.run")
data = ModuleType("earth2studio.data")
models = ModuleType("earth2studio.models")
models.__path__ = []
px = ModuleType("earth2studio.models.px")
io_module = ModuleType("earth2studio.io")


class FakeModelType:
    @classmethod
    def load_default_package(cls):
        return object()

    @classmethod
    def load_model(cls, _package):
        return object()


class GFS:
    def __init__(self, **_kwargs):
        pass


class ZarrBackend:
    def __init__(self, path, **_kwargs):
        self.path = Path(path)

    def finalize(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".written").write_text("ok", encoding="utf-8")
        return self


def deterministic(_times, _nsteps, _model, _data, io):
    sync_dir = Path(os.environ["E2S_TEST_INFERENCE_SYNC_DIR"])
    sync_dir.mkdir(parents=True, exist_ok=True)
    marker = sync_dir / f"inference-{os.getpid()}"
    marker.write_text(
        str(threading.current_thread() is threading.main_thread()),
        encoding="utf-8",
    )
    deadline = time.monotonic() + 10
    expected = int(os.environ["E2S_TEST_EXPECTED_PROCESSES"])
    while len(list(sync_dir.glob("inference-*"))) < expected:
        if time.monotonic() >= deadline:
            raise TimeoutError("Earth2 inference processes did not overlap")
        time.sleep(0.01)
    return io


run.deterministic = deterministic
data.GFS = GFS
px.DLWP = FakeModelType
px.FCN = FakeModelType
px.FCN3 = FakeModelType
io_module.ZarrBackend = ZarrBackend
earth2studio.run = run
earth2studio.data = data
earth2studio.models = models
earth2studio.io = io_module
models.px = px

sys.modules["earth2studio"] = earth2studio
sys.modules["earth2studio.run"] = run
sys.modules["earth2studio.data"] = data
sys.modules["earth2studio.models"] = models
sys.modules["earth2studio.models.px"] = px
sys.modules["earth2studio.io"] = io_module
""".strip(),
        encoding="utf-8",
    )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(fake_root)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("E2S_TEST_INFERENCE_SYNC_DIR", str(sync_dir))
    monkeypatch.setenv("E2S_TEST_EXPECTED_PROCESSES", str(expected_processes))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    return sync_dir


def _raw_request(**raw_fields) -> RawRequest:
    return RawRequest(
        content_type="application/json",
        operation="run",
        raw_fields=raw_fields,
        input_artifacts=[],
    )


def _prepare_context(
    tmp_path: Path, *, workflow_id: str, run_id: str
) -> PrepareContext:
    return PrepareContext(
        run_id=run_id,
        workflow_id=workflow_id,
        run_dir=tmp_path / run_id,
    )


def _execution_context(
    tmp_path: Path,
    *,
    run_id: str,
    resource_profile=None,
    services=None,
) -> ExecutionContext:
    run_dir = tmp_path / run_id
    return ExecutionContext(
        run_id=run_id,
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile=resource_profile or {"gpus_required": 1},
        services=services or {},
    )


def _install_fake_earth2_runtime(
    monkeypatch,
    *,
    interp_to_calls=None,
    model_to_calls=None,
    zarr_backend_calls=None,
    xarray_backend_calls=None,
    netcdf_backend_calls=None,
    consolidate_metadata_calls=None,
    cuda_empty_cache_calls=None,
    model_load_calls=None,
    gfs_close_calls=None,
    package_close_calls=None,
    pretrained_calls=None,
    data_source_init_calls=None,
    stormcast_model_instances=None,
):
    """Install comprehensive fakes for all earth2studio imports used by new plugins."""

    import numpy as np

    class FakeTensor:
        def __init__(self, data=None):
            self.data = np.asarray(data if data is not None else [[[[0.0]]]])

        def clone(self):
            return FakeTensor(self.data.copy())

        def unsqueeze(self, _dim):
            return self

        def squeeze(self, _dim=None):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.data)

    class FakeCoordSystem(OrderedDict):
        def copy(self):
            return FakeCoordSystem(self)

    class FakeModel:
        def to(self, _device=None, **_kwargs):
            device = _device if _device is not None else _kwargs.get("device")
            if model_to_calls is not None:
                model_to_calls.append(device)
            return self

        def eval(self):
            return self

        def input_coords(self):
            return {"lat": [0.0], "lon": [0.0]}

    class FakeFCN3Model(FakeModel):
        variables = np.asarray(["t2m", "u10m", "v10m", "z500"])

        def __init__(self):
            self.seed_calls: list[int] = []

        def input_coords(self):
            return {
                "variable": self.variables,
                "lat": np.asarray([0.0], dtype=np.float64),
                "lon": np.asarray([0.0], dtype=np.float64),
            }

        def set_rng(self, *, seed):
            self.seed_calls.append(seed)

        def create_iterator(self, _x, coords):
            base_time = np.asarray(coords["time"], dtype="datetime64[ns]")
            for step in range(4):
                yield (
                    FakeTensor(),
                    FakeCoordSystem(
                        {
                            "time": base_time,
                            "lead_time": np.asarray([step * 6], dtype="timedelta64[h]"),
                            "variable": self.variables,
                            "lat": np.asarray([0.0], dtype=np.float64),
                            "lon": np.asarray([0.0], dtype=np.float64),
                        }
                    ),
                )

    class FakeInterpModel(FakeModel):
        def to(self, _device=None, **_kwargs):
            device = _device if _device is not None else _kwargs.get("device")
            if interp_to_calls is not None:
                interp_to_calls.append(device)
            return self

    class _FakePackageFilesystem:
        def __init__(self, label: str) -> None:
            self.loop = f"{label}-package-loop"
            self._session = f"{label}-package-session"
            self._label = label

        def close_session(self, loop, session) -> None:
            if package_close_calls is not None:
                package_close_calls.append((loop, session, self._label))

    class _FakePackage:
        def __init__(self, label: str) -> None:
            self.fs = _FakePackageFilesystem(label)

        def resolve(self, _name):
            return "/dev/null"

    class _BaseFakeModelClass:
        model_name = "model"

        @staticmethod
        def load_default_package():
            return _FakePackage("model")

        @classmethod
        def load_model(cls, _package):
            if model_load_calls is not None:
                model_load_calls.append(cls.model_name)
            return FakeModel()

        @classmethod
        def from_pretrained(cls):
            if pretrained_calls is not None:
                pretrained_calls.append(cls.model_name)
            return FakeModel()

    class FakeDLWPClass(_BaseFakeModelClass):
        model_name = "DLWP"

        @staticmethod
        def load_default_package():
            return _FakePackage("DLWP")

    class FakeFCNClass(_BaseFakeModelClass):
        model_name = "FCN"

        @staticmethod
        def load_default_package():
            return _FakePackage("FCN")

    class FakeFCN3Class(_BaseFakeModelClass):
        model_name = "FCN3"

        @staticmethod
        def load_default_package():
            return _FakePackage("FCN3")

        @classmethod
        def load_model(cls, _package):
            if model_load_calls is not None:
                model_load_calls.append(cls.model_name)
            return FakeFCN3Model()

    class FakePrecipitationAFNOClass(_BaseFakeModelClass):
        model_name = "PrecipitationAFNO"

        @staticmethod
        def load_default_package():
            return _FakePackage("PrecipitationAFNO")

    class FakePerturbation:
        def __init__(self, **_kwargs):
            pass

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: (
            cuda_empty_cache_calls.append("empty_cache")
            if cuda_empty_cache_calls is not None
            else None
        ),
    )
    fake_torch.device = lambda name: name
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.as_tensor = lambda x: x
    fake_torch.tensor = lambda x: np.asarray(x)
    fake_torch.roll = lambda tensor, _shifts, _dims=None, **_kwargs: tensor

    fake_models = ModuleType("earth2studio.models")
    fake_models_px = ModuleType("earth2studio.models.px")
    fake_models_px.DLWP = FakeDLWPClass
    fake_models_px.FCN = FakeFCNClass
    fake_models_px.FCN3 = FakeFCN3Class
    fake_models_px.DiagnosticWrapper = lambda **_kwargs: FakeModel()
    fake_models_px.InterpModAFNO = type(
        "InterpModAFNO",
        (FakeInterpModel,),
        {
            "load_default_package": staticmethod(lambda: _FakePackage("InterpModAFNO")),
            "load_model": staticmethod(
                lambda _package, px_model=None: (
                    model_load_calls.append("InterpModAFNO")
                    if model_load_calls is not None
                    else None,
                    FakeInterpModel(),
                )[1]
            ),
            "from_pretrained": staticmethod(
                lambda: (
                    pretrained_calls.append("InterpModAFNO")
                    if pretrained_calls is not None
                    else None,
                    FakeInterpModel(),
                )[1]
            ),
            "px_model": None,
        },
    )

    class FakeStormCastModel(FakeModel):
        def __init__(self):
            self.conditioning_data_source = None

    class FakeStormCastClass:
        @staticmethod
        def load_default_package():
            return _FakePackage("StormCast")

        @staticmethod
        def load_model(_package, conditioning_data_source=None):
            if model_load_calls is not None:
                model_load_calls.append("StormCast")
            model = FakeStormCastModel()
            model.conditioning_data_source = conditioning_data_source
            if stormcast_model_instances is not None:
                stormcast_model_instances.append(model)
            return model

        @staticmethod
        def from_pretrained():
            if pretrained_calls is not None:
                pretrained_calls.append("StormCast")
            model = FakeStormCastModel()
            if stormcast_model_instances is not None:
                stormcast_model_instances.append(model)
            return model

    fake_models_px.StormCast = FakeStormCastClass

    fake_models_dx = ModuleType("earth2studio.models.dx")
    fake_models_dx.PrecipitationAFNO = FakePrecipitationAFNOClass
    fake_models_dx.DerivedSurfacePressure = lambda **_kwargs: FakeModel()

    fake_data = ModuleType("earth2studio.data")

    class FakeFilesystem:
        def __init__(self) -> None:
            self.loop = "fake-loop"
            self._s3creator = "fake-s3creator"

        def close_session(self, loop, s3creator) -> None:
            if gfs_close_calls is not None:
                gfs_close_calls.append((loop, s3creator))

    class FakeDataSource:
        def __init__(self, **_kwargs) -> None:
            if data_source_init_calls is not None:
                data_source_init_calls.append("GFS")
            self.fs = FakeFilesystem()

    class FakeHrrrDataSource:
        def __init__(self) -> None:
            if data_source_init_calls is not None:
                data_source_init_calls.append("HRRR")

    class FakePlanetaryComputerECMWFOpenDataIFS(FakeDataSource):
        def __init__(self, **_kwargs) -> None:
            if data_source_init_calls is not None:
                data_source_init_calls.append("PlanetaryComputerECMWFOpenDataIFS")
            self.fs = FakeFilesystem()

    def fake_fetch_data(_data, *, time, variable, device=None, **_kwargs):
        del variable, device
        return (
            FakeTensor(),
            FakeCoordSystem(
                {
                    "time": np.asarray(time, dtype="datetime64[ns]"),
                    "variable": FakeFCN3Model.variables,
                    "lat": np.asarray([0.0], dtype=np.float64),
                    "lon": np.asarray([0.0], dtype=np.float64),
                }
            ),
        )

    fake_data.GFS = FakeDataSource
    fake_data.HRRR = FakeHrrrDataSource
    fake_data.InferenceOutputSource = lambda _root: object()
    fake_data.PlanetaryComputerECMWFOpenDataIFS = FakePlanetaryComputerECMWFOpenDataIFS
    fake_data.fetch_data = fake_fetch_data

    fake_io = ModuleType("earth2studio.io")

    class FakeZarrBackendClass:
        def __new__(
            cls, file_name=None, path=None, chunks=None, backend_kwargs=None, **kw
        ):
            if zarr_backend_calls is not None:
                zarr_backend_calls.append(
                    {
                        "path": file_name or path or "backend",
                        "chunks": chunks,
                        "backend_kwargs": dict(backend_kwargs or {}),
                        "extra_kwargs": dict(kw),
                    }
                )
            return file_name or path or "backend"

    def fake_xarray_backend(**kw):
        if xarray_backend_calls is not None:
            xarray_backend_calls.append(dict(kw))
        return SimpleNamespace(root="memory")

    class FakeNetCDF4BackendClass:
        def __new__(cls, path=None, **kw):
            if netcdf_backend_calls is not None:
                netcdf_backend_calls.append({"path": path, "kwargs": dict(kw)})
            return object()

    fake_io.ZarrBackend = FakeZarrBackendClass
    fake_io.IOBackend = object
    fake_io.XarrayBackend = fake_xarray_backend
    fake_io.NetCDF4Backend = FakeNetCDF4BackendClass

    fake_run = ModuleType("earth2studio.run")

    def write_fake_zarr_output(io, *, ensemble_size: int | None = None):
        import numpy as np

        output_path = Path(io) if isinstance(io, str) else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / ".written").write_text("ok", encoding="utf-8")
            return io

        if not hasattr(io, "add_array"):
            return io

        coords = {
            "time": np.asarray(["2024-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.asarray([0], dtype="timedelta64[ns]"),
            "lat": np.asarray([0.0], dtype=np.float64),
            "lon": np.asarray([0.0], dtype=np.float64),
        }
        if ensemble_size is not None:
            coords = {
                "ensemble": np.arange(ensemble_size, dtype=np.int64),
                **coords,
            }
            data = np.zeros((ensemble_size, 1, 1, 1, 1), dtype=np.float32)
        else:
            data = np.zeros((1, 1, 1, 1), dtype=np.float32)

        io.add_array(coords, ["t2m"])
        io.write([data], coords, ["t2m"])
        return io

    def fake_deterministic(
        time=None, nsteps=None, prognostic=None, data=None, io=None, device=None, **_kw
    ):
        return write_fake_zarr_output(io)

    def fake_diagnostic(
        time=None,
        nsteps=None,
        prognostic=None,
        diagnostic=None,
        data=None,
        io=None,
        device=None,
        **_kw,
    ):
        return write_fake_zarr_output(io)

    def fake_ensemble(
        time=None,
        nsteps=None,
        nensemble=None,
        prognostic=None,
        data=None,
        io=None,
        perturbation=None,
        batch_size=None,
        output_coords=None,
        device=None,
        **_kw,
    ):
        return write_fake_zarr_output(io, ensemble_size=nensemble or 1)

    fake_run.deterministic = fake_deterministic
    fake_run.diagnostic = fake_diagnostic
    fake_run.ensemble = fake_ensemble

    fake_perturbation = ModuleType("earth2studio.perturbation")
    fake_perturbation.Gaussian = FakePerturbation
    fake_perturbation.Brown = FakePerturbation
    fake_perturbation.SphericalGaussian = FakePerturbation

    fake_utils = ModuleType("earth2studio.utils")
    fake_utils_coords = ModuleType("earth2studio.utils.coords")
    fake_utils_coords.CoordSystem = FakeCoordSystem

    def fake_map_coords(x, coords, target_coords):
        mapped = FakeCoordSystem(coords)
        if "variable" in target_coords:
            mapped["variable"] = np.asarray(target_coords["variable"])
        return x, mapped

    def fake_split_coords(x, coords):
        return x, coords, list(np.asarray(coords["variable"]))

    fake_utils_coords.map_coords = fake_map_coords
    fake_utils_coords.split_coords = fake_split_coords
    fake_utils_time = ModuleType("earth2studio.utils.time")
    fake_utils_time.to_time_array = lambda values: np.asarray(
        values, dtype="datetime64[ns]"
    )

    fake_zarr = ModuleType("zarr")
    fake_zarr.consolidate_metadata = lambda path: (
        consolidate_metadata_calls.append(path)
        if consolidate_metadata_calls is not None
        else None
    )

    class _FakeDataArray:
        def __init__(self, data):
            self._data = data
            self.values = data

        def __getitem__(self, idx):
            return _FakeDataArray(
                self._data[idx] if hasattr(self._data, "__getitem__") else self._data
            )

    class _FakeDataset:
        def __init__(self):
            self._vars = {"Z": _FakeDataArray(np.zeros((1, 4)))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def __getitem__(self, key):
            return self._vars[key]

    fake_xarray = ModuleType("xarray")
    fake_xarray.open_dataset = lambda _fn: _FakeDataset()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "earth2studio", ModuleType("earth2studio"))
    monkeypatch.setitem(sys.modules, "earth2studio.models", fake_models)
    monkeypatch.setitem(sys.modules, "earth2studio.models.px", fake_models_px)
    monkeypatch.setitem(sys.modules, "earth2studio.models.dx", fake_models_dx)
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)
    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.run", fake_run)
    monkeypatch.setitem(sys.modules, "earth2studio.perturbation", fake_perturbation)
    monkeypatch.setitem(sys.modules, "earth2studio.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_utils_coords)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_utils_time)
    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)
    if "e2s_zarr_io" not in sys.modules:
        _install_fake_rust_zarr_io(monkeypatch)


# ===========================================================================
# 1. e2s-deterministic-earth2
# ===========================================================================

PLUGIN_DETERMINISTIC_EARTH2 = REPO_ROOT / "plugins" / "e2s-deterministic-earth2"


def test_e2s_deterministic_earth2_prepare_coerces_input(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_earth2_prepare_dlwp",
        PLUGIN_DETERMINISTIC_EARTH2 / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(start_time=["2024-01-01T00:00:00"], num_steps=4),
        _prepare_context(tmp_path, workflow_id="e2s-deterministic-earth2", run_id="r1"),
    )
    assert prepared.inputs["start_time"] == ["2024-01-01T00:00:00"]
    assert prepared.inputs["num_steps"] == 4


def test_e2s_deterministic_earth2_run_creates_zarr_output(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_earth2_run",
        PLUGIN_DETERMINISTIC_EARTH2 / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="earth2-run")
    result = module.WORKFLOW().run(
        {"start_time": ["2024-01-01T00:00:00"], "num_steps": 4},
        ctx,
    )

    expected = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected)
    assert ctx.outputs.primary_output().path == str(expected)


def test_e2s_deterministic_earth2_gfs_cleanup_closes_filesystem_session(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    _install_fake_earth2_runtime(monkeypatch, gfs_close_calls=gfs_close_calls)
    module = _load_module(
        "e2s_deterministic_earth2_gfs_cleanup",
        PLUGIN_DETERMINISTIC_EARTH2 / "workflow.py",
    )

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="earth2-run-cleanup")
    workflow.run(
        {"start_time": ["2024-01-01T00:00:00"], "num_steps": 1},
        ctx,
    )

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_e2s_deterministic_earth2_cleanup_releases_package_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    package_close_calls: list[tuple[str, str, str]] = []
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        package_close_calls=package_close_calls,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_deterministic_earth2_cleanup_runtime_memory",
        PLUGIN_DETERMINISTIC_EARTH2 / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="earth2-run-cleanup-package")
    workflow.run(
        {"start_time": ["2024-01-01T00:00:00"], "num_steps": 1},
        ctx,
    )

    workflow.cleanup()

    assert package_close_calls == []
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


# ===========================================================================
# 2. e2s-deterministic
# ===========================================================================

PLUGIN_DETERMINISTIC = REPO_ROOT / "plugins" / "e2s-deterministic"


def test_e2s_deterministic_prepare_coerces_input(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_prepare_dlwp",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            forecast_times=["2024-01-01T00:00:00"],
            nsteps=6,
            model_type="dlwp",
            create_plots=False,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-deterministic", run_id="r1"),
    )
    assert prepared.inputs["model_type"] == "dlwp"
    assert prepared.inputs["nsteps"] == 6
    assert prepared.inputs["forecast_times"] == ["2024-01-01T00:00:00"]


def test_e2s_deterministic_prepare_rejects_invalid_model(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_prepare_bad_model",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )
    with pytest.raises(TypeError, match="model_type"):
        module.WORKFLOW().prepare(
            _raw_request(
                forecast_times=["2024-01-01T00:00:00"], nsteps=6, model_type="bad"
            ),
            _prepare_context(tmp_path, workflow_id="e2s-deterministic", run_id="r2"),
        )


def test_e2s_deterministic_run_creates_zarr_and_metadata(tmp_path: Path, monkeypatch):
    _install_fake_rust_zarr_io(monkeypatch)
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_run",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run")
    result = module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "model_type": "dlwp",
            "create_plots": False,
        },
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)

    metadata_path = ctx.run_dir / "forecast_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_type"] == "dlwp"
    assert metadata["nsteps"] == 4
    assert metadata["data_source"] == "gfs"
    assert metadata["output_format"] == "zarr"


def test_e2s_deterministic_run_stages_zarr_output_before_promoting_to_final_path(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    zarr_backend_calls = []
    consolidate_metadata_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        zarr_backend_calls=zarr_backend_calls,
        consolidate_metadata_calls=consolidate_metadata_calls,
    )
    module = _load_module(
        "e2s_deterministic_run_overwrite",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run-overwrite")
    module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "model_type": "dlwp",
            "create_plots": False,
        },
        ctx,
    )

    final_path = ctx.run_dir / "forecast.zarr"
    assert len(zarr_backend_calls) == 1

    backend_call = zarr_backend_calls[0]
    staged_path = Path(backend_call["path"])
    assert staged_path.parent == ctx.run_dir
    assert staged_path.name.startswith(".forecast.zarr.tmp-")
    assert staged_path != final_path
    assert backend_call["chunks"] is None
    assert backend_call["backend_kwargs"] == {"overwrite": True, "zarr_format": 3}
    assert backend_call["extra_kwargs"] == {}
    assert consolidate_metadata_calls == [str(staged_path)]
    assert final_path.exists()
    assert not staged_path.exists()


def test_e2s_deterministic_rust_zarr_backend_env_closes_before_promote(
    tmp_path: Path, monkeypatch
):
    import numpy as np

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    rust_backend_calls = []
    zarr_backend_calls = []
    consolidate_metadata_calls: list[str] = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)
    _install_fake_earth2_runtime(
        monkeypatch,
        zarr_backend_calls=zarr_backend_calls,
        consolidate_metadata_calls=consolidate_metadata_calls,
    )
    module = _load_module(
        "e2s_deterministic_run_rust_default",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run-rust-default")
    module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "model_type": "dlwp",
            "create_plots": False,
        },
        ctx,
    )

    final_path = ctx.run_dir / "forecast.zarr"
    assert zarr_backend_calls == []
    assert consolidate_metadata_calls == []
    assert len(rust_backend_calls) == 1

    backend_call = rust_backend_calls[0]
    staged_path = Path(backend_call["kwargs"]["file_name"])
    assert staged_path.parent == ctx.run_dir
    assert staged_path.name.startswith(".forecast.zarr.tmp-")
    assert staged_path != final_path
    assert backend_call["kwargs"]["fsync_policy"] == "never"
    assert set(backend_call["kwargs"]["parallel_coords"]) == {"time", "lead_time"}
    time_coord = backend_call["kwargs"]["parallel_coords"]["time"]
    lead_time_coord = backend_call["kwargs"]["parallel_coords"]["lead_time"]
    assert time_coord.dtype == np.dtype("datetime64[ns]")
    assert time_coord.tolist() == [1704067200000000000]
    assert lead_time_coord.dtype == np.dtype("timedelta64[ns]")
    assert lead_time_coord.tolist() == [0]
    add_array_coords = backend_call["add_array"][0]["coords"]
    assert add_array_coords["time"].dtype == np.dtype("datetime64[ns]")
    assert add_array_coords["lead_time"].dtype == np.dtype("timedelta64[ns]")
    assert backend_call["close"] == [None]
    assert final_path.exists()
    assert not staged_path.exists()


def test_e2s_deterministic_python_backend_env_bypasses_missing_rust_module(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", None)
    zarr_backend_calls = []
    _install_fake_earth2_runtime(monkeypatch, zarr_backend_calls=zarr_backend_calls)
    module = _load_module(
        "e2s_deterministic_run_python_backend_env",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run-python-env")
    module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "model_type": "dlwp",
            "create_plots": False,
        },
        ctx,
    )

    assert len(zarr_backend_calls) == 1


def test_e2s_deterministic_rust_backend_env_fails_when_extension_missing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", None)
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_run_missing_rust_backend",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run-missing-rust")
    with pytest.raises(RuntimeError, match="e2s_zarr_io"):
        module.WORKFLOW().run(
            {
                "forecast_times": ["2024-01-01T00:00:00"],
                "nsteps": 4,
                "model_type": "dlwp",
                "create_plots": False,
            },
            ctx,
        )


def test_e2s_deterministic_rejects_unknown_zarr_backend_env(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "pythno")
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_run_bad_backend_env",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="det-run-bad-backend-env")
    with pytest.raises(ValueError, match="PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND"):
        module.WORKFLOW().run(
            {
                "forecast_times": ["2024-01-01T00:00:00"],
                "nsteps": 4,
                "model_type": "dlwp",
                "create_plots": False,
            },
            ctx,
        )


def test_e2s_deterministic_cleanup_releases_runtime_resources_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    _install_fake_rust_zarr_io(monkeypatch)
    gfs_close_calls: list[tuple[str, str]] = []
    package_close_calls: list[tuple[str, str, str]] = []
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        gfs_close_calls=gfs_close_calls,
        package_close_calls=package_close_calls,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_deterministic_cleanup_runtime_memory",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="det-run-cleanup")
    workflow.run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 1,
            "model_type": "fcn3",
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
        },
        ctx,
    )

    assert gfs_close_calls == []

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]
    assert package_close_calls == []
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


def test_e2s_deterministic_gfs_cache_is_request_scoped(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_request_scoped_gfs",
        PLUGIN_DETERMINISTIC / "workflow.py",
    )

    first_workflow = module.WORKFLOW()
    second_workflow = module.WORKFLOW()
    first_gfs = first_workflow._data_for_source("gfs")
    second_gfs = second_workflow._data_for_source("gfs")
    first_cache = Path(first_gfs.cache)
    second_cache = Path(second_gfs.cache)

    assert first_gfs is not second_gfs
    assert first_cache != second_cache
    assert first_cache.is_dir()
    assert second_cache.is_dir()

    first_workflow.cleanup()
    second_workflow.cleanup()

    assert not first_cache.exists()
    assert not second_cache.exists()


def test_earth2_workflow_cleanup_removes_staged_zarr_output_after_failure(
    tmp_path: Path, monkeypatch
):
    _install_fake_earth2_runtime(monkeypatch)

    from e2s_workflow import Earth2Workflow

    class FailingWorkflow(Earth2Workflow):
        def __call__(self, io: object) -> None:
            del io
            Path(self.output_dataset_path).mkdir(parents=True, exist_ok=True)
            raise RuntimeError("boom")

    workflow = FailingWorkflow()
    ctx = _execution_context(tmp_path, run_id="det-run-staged-cleanup")

    with pytest.raises(RuntimeError, match="boom"):
        workflow.run({}, ctx)

    staged_paths = list(ctx.run_dir.glob(".forecast.zarr.tmp-*"))
    assert len(staged_paths) == 1
    assert staged_paths[0].exists()

    workflow.cleanup()

    assert not staged_paths[0].exists()


def test_earth2_workflow_cleanup_closes_http_filesystem_session():
    from e2s_workflow import Earth2Workflow

    http_close_calls: list[str] = []

    class FakeHTTPFilesystem:
        def __init__(self) -> None:
            self.loop = "http-loop"
            self._session = "http-session"

        def close_session(self, loop) -> None:
            http_close_calls.append(loop)

    class WorkflowWithHTTPPackage(Earth2Workflow):
        def __init__(self) -> None:
            super().__init__()
            self.package = SimpleNamespace(fs=FakeHTTPFilesystem())

        def __call__(self, io: object) -> None:
            del io

    workflow = WorkflowWithHTTPPackage()
    workflow.cleanup()

    assert http_close_calls == ["http-loop"]


def test_workflow_executor_env_parallelism_reaches_e2s_deterministic_inference(
    tmp_path: Path, monkeypatch
):
    item_count = 4
    sync_dir = _install_subprocess_earth2_fakes(
        tmp_path,
        monkeypatch,
        expected_processes=item_count,
    )
    monkeypatch.setenv("PLUGIN_DIR", str(REPO_ROOT / "plugins"))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "e2s-deterministic")
    monkeypatch.setenv("PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS", str(item_count))
    monkeypatch.delenv("REDIS_URL", raising=False)

    worker_module = _load_module(
        "inference_worker_e2s_parallel_integration",
        SCRIPTS_DIR / "inference_worker.py",
    )
    executor = worker_module.WorkflowExecutor(None)
    runtime = {"entrypoint": "workflow.py", "kind": "python"}
    items = []
    for index in range(item_count):
        run_id = f"parallel-batch:item:{index}"
        parameters = {
            "forecast_times": [f"2024-01-0{index + 1}T00:00:00"],
            "nsteps": 2,
            "model_type": "fcn",
            "create_plots": False,
        }
        items.append(
            {
                "run_id": run_id,
                "payload": {
                    "run_id": run_id,
                    "workflow_id": "e2s-deterministic",
                    "operation": "run",
                    "parameters": parameters,
                    "runtime": runtime,
                },
            }
        )

    try:
        assert executor._batch_coordinator.max_parallel_items == item_count
        result = executor.execute(
            "e2s-deterministic",
            "parallel-batch",
            {},
            payload={
                "workflow_id": "e2s-deterministic",
                "operation": "run",
                "runtime": runtime,
                "items": items,
            },
        )
    finally:
        executor.close()

    assert result["status"] == "succeeded"
    assert [entry["run_id"] for entry in result["batch_results"]] == [
        item["run_id"] for item in items
    ]
    item_results = [entry["result"] for entry in result["batch_results"]]
    assert all(item_result["status"] == "succeeded" for item_result in item_results)
    output_paths = [item_result["output_path"] for item_result in item_results]
    assert len(set(output_paths)) == item_count
    assert all(Path(path).exists() for path in output_paths)
    inference_markers = list(sync_dir.glob("inference-*"))
    assert len(inference_markers) == item_count
    assert all(
        marker.read_text(encoding="utf-8") == "True" for marker in inference_markers
    )


# ===========================================================================
# 3. e2s-deterministic-fcn
# ===========================================================================

PLUGIN_DETERMINISTIC_FCN = REPO_ROOT / "plugins" / "e2s-deterministic-fcn"


def test_e2s_deterministic_fcn_prepare_accepts_defaults(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_fcn_prepare",
        PLUGIN_DETERMINISTIC_FCN / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            forecast_times=["2024-01-01T00:00:00"],
            nsteps=6,
            data_source="gfs",
            output_format="zarr",
            create_plots=False,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-deterministic-fcn", run_id="r1"),
    )
    assert prepared.inputs["nsteps"] == 6
    assert prepared.inputs["forecast_times"] == ["2024-01-01T00:00:00"]
    assert prepared.inputs["data_source"] == "gfs"
    assert prepared.inputs["output_format"] == "zarr"


def test_e2s_deterministic_fcn_run_creates_zarr_output(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_deterministic_fcn_run",
        PLUGIN_DETERMINISTIC_FCN / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="fcn-run")
    result = module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
            "plot_variable": "t2m",
            "plot_step": 4,
        },
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)

    metadata_path = ctx.run_dir / "forecast_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_type"] == "FCN"
    assert metadata["data_source"] == "gfs"
    assert metadata["output_format"] == "zarr"


def test_e2s_deterministic_fcn_warmup_reuses_model_until_final_cleanup(
    tmp_path: Path, monkeypatch
):
    model_load_calls: list[str] = []
    model_to_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        model_load_calls=model_load_calls,
        model_to_calls=model_to_calls,
    )
    module = _load_module(
        "e2s_deterministic_fcn_warmup_cache",
        PLUGIN_DETERMINISTIC_FCN / "workflow.py",
    )

    workflow = module.WORKFLOW()
    assert workflow.cache_scope == "process"
    assert model_load_calls == ["FCN"]

    warmup_result = workflow.warmup(
        {"workflow_id": "e2s-deterministic-fcn", "device": "cuda"}
    )
    assert warmup_result["model_names"] == ["FCN"]
    assert model_load_calls == ["FCN"]
    assert model_to_calls == ["cuda"]

    ctx = _execution_context(tmp_path, run_id="fcn-warm-run")
    workflow.run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
        },
        ctx,
    )
    workflow.cleanup_request()

    assert model_load_calls == ["FCN"]
    assert workflow.model is not None

    workflow.cleanup()

    assert workflow.model is None


def test_e2s_deterministic_fcn_cleanup_releases_runtime_resources_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    package_close_calls: list[tuple[str, str, str]] = []
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        gfs_close_calls=gfs_close_calls,
        package_close_calls=package_close_calls,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_deterministic_fcn_cleanup_runtime_memory",
        PLUGIN_DETERMINISTIC_FCN / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="fcn-run-cleanup")
    workflow.run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
        },
        ctx,
    )

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]
    assert package_close_calls == []
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


# ===========================================================================
# 4. e2s-diagnostic
# ===========================================================================

PLUGIN_DIAGNOSTIC = REPO_ROOT / "plugins" / "e2s-diagnostic"


def test_e2s_diagnostic_prepare_defaults_to_fcn_prognostic(tmp_path: Path):
    module = _load_module(
        "e2s_diagnostic_prepare",
        PLUGIN_DIAGNOSTIC / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            forecast_times=["2024-01-01T00:00:00"],
            nsteps=6,
            diagnostic_model_type="precipitation_afno",
            data_source="gfs",
            output_format="zarr",
            create_plots=False,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-diagnostic", run_id="r1"),
    )
    assert prepared.inputs["prognostic_model_type"] == "fcn"
    assert prepared.inputs["diagnostic_model_type"] == "precipitation_afno"
    assert prepared.inputs["data_source"] == "gfs"
    assert prepared.inputs["output_format"] == "zarr"


def test_e2s_diagnostic_prepare_rejects_non_fcn_prognostic(tmp_path: Path):
    module = _load_module(
        "e2s_diagnostic_prepare_reject_non_fcn",
        PLUGIN_DIAGNOSTIC / "workflow.py",
    )
    with pytest.raises(TypeError, match="prognostic_model_type must be one of"):
        module.WORKFLOW().prepare(
            _raw_request(
                forecast_times=["2024-01-01T00:00:00"],
                nsteps=6,
                prognostic_model_type="dlwp",
                diagnostic_model_type="precipitation_afno",
                data_source="gfs",
                output_format="zarr",
                create_plots=False,
            ),
            _prepare_context(tmp_path, workflow_id="e2s-diagnostic", run_id="r1"),
        )


def test_e2s_diagnostic_run_creates_zarr_and_metadata(tmp_path: Path, monkeypatch):
    module = _load_module(
        "e2s_diagnostic_run",
        PLUGIN_DIAGNOSTIC / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)

    ctx = _execution_context(tmp_path, run_id="diag-run")
    result = module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 6,
            "prognostic_model_type": "fcn",
            "diagnostic_model_type": "precipitation_afno",
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
            "plot_variable": "tp",
            "plot_step": 4,
        },
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)

    metadata_path = ctx.run_dir / "forecast_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prognostic_model_type"] == "fcn"
    assert metadata["diagnostic_model_type"] == "precipitation_afno"
    assert metadata["data_source"] == "gfs"
    assert metadata["output_format"] == "zarr"


def test_e2s_diagnostic_cleanup_releases_runtime_resources_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    package_close_calls: list[tuple[str, str, str]] = []
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        gfs_close_calls=gfs_close_calls,
        package_close_calls=package_close_calls,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_diagnostic_cleanup_runtime_memory",
        PLUGIN_DIAGNOSTIC / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="diag-run-cleanup")
    workflow.run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 6,
            "diagnostic_model_type": "precipitation_afno",
            "data_source": "gfs",
            "output_format": "zarr",
            "create_plots": False,
        },
        ctx,
    )

    assert gfs_close_calls == []
    assert package_close_calls == []
    assert gc_collect_calls == []
    assert cuda_ipc_collect_calls == []
    assert cuda_empty_cache_calls == []

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]
    assert package_close_calls == []
    # workflow.cleanup() performs one torch cleanup pass.
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


# ===========================================================================
# 5. e2s-ensemble
# ===========================================================================

PLUGIN_ENSEMBLE = REPO_ROOT / "plugins" / "e2s-ensemble"


def test_e2s_ensemble_prepare_coerces_input(tmp_path: Path):
    module = _load_module(
        "e2s_ensemble_prepare",
        PLUGIN_ENSEMBLE / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            forecast_times=["2024-01-01T00:00:00"],
            nsteps=4,
            nensemble=2,
            batch_size=2,
            create_plots=False,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-ensemble", run_id="r1"),
    )
    assert prepared.inputs["nensemble"] == 2
    assert prepared.inputs["batch_size"] == 2
    assert prepared.inputs["model_type"] == "fcn"
    assert prepared.inputs["data_source"] == "gfs"
    assert prepared.inputs["output_format"] == "zarr"
    assert prepared.inputs["seed_base"] == 1000


def test_e2s_ensemble_run_creates_zarr_and_metadata(tmp_path: Path, monkeypatch):
    rust_backend_calls = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_ensemble_run",
        PLUGIN_ENSEMBLE / "workflow.py",
    )
    manual_seed_calls: list[int] = []
    monkeypatch.setattr(
        sys.modules["torch"],
        "manual_seed",
        lambda seed: manual_seed_calls.append(seed),
    )

    ctx = _execution_context(tmp_path, run_id="ens-run")
    result = module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "nensemble": 2,
            "batch_size": 2,
            "seed_base": 1234,
            "create_plots": False,
        },
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)

    metadata_path = ctx.run_dir / "forecast_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["nensemble"] == 2
    assert metadata["model_type"] == "fcn"
    assert metadata["noise_amplitude"] == 0.15
    assert metadata["seed_base"] == 1234
    assert metadata["data_source"] == "gfs"
    assert metadata["output_format"] == "zarr"
    assert manual_seed_calls == [1234]
    assert len(rust_backend_calls) == 1
    assert set(rust_backend_calls[0]["kwargs"]["parallel_coords"]) == {
        "ensemble",
        "time",
        "lead_time",
    }
    assert rust_backend_calls[0]["kwargs"]["default_parallel_coord_names"] == [
        "ensemble",
        "time",
        "lead_time",
    ]


def test_e2s_ensemble_python_backend_strips_rust_only_kwargs(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", None)
    zarr_backend_calls = []
    _install_fake_earth2_runtime(monkeypatch, zarr_backend_calls=zarr_backend_calls)
    module = _load_module(
        "e2s_ensemble_python_backend_kwargs",
        PLUGIN_ENSEMBLE / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="ens-run-python")
    module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "nensemble": 2,
            "batch_size": 2,
            "create_plots": False,
        },
        ctx,
    )

    assert len(zarr_backend_calls) == 1
    assert zarr_backend_calls[0]["chunks"] == {
        "ensemble": 1,
        "time": 1,
        "lead_time": 1,
    }
    assert zarr_backend_calls[0]["backend_kwargs"] == {
        "overwrite": True,
        "zarr_format": 3,
    }
    assert zarr_backend_calls[0]["extra_kwargs"] == {}


def test_e2s_ensemble_rust_preserves_ensemble_first_payload_with_matching_axis_order(
    tmp_path: Path, monkeypatch
):
    import numpy as np

    rust_backend_calls = []
    _install_fake_rust_zarr_io(monkeypatch, rust_backend_calls=rust_backend_calls)
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_ensemble_rust_payload_axis_order",
        PLUGIN_ENSEMBLE / "workflow.py",
    )

    def fake_ensemble(
        _forecast_times,
        _nsteps,
        nensemble,
        _model,
        _data,
        io,
        _perturbation,
        **_kwargs,
    ):
        coords = {
            "ensemble": np.arange(nensemble, dtype=np.int64),
            "time": np.asarray(
                ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                dtype="datetime64[ns]",
            ),
            "lead_time": np.asarray([0, 6, 12], dtype="timedelta64[h]"),
            "lat": np.asarray([0.0], dtype=np.float64),
            "lon": np.asarray([0.0], dtype=np.float64),
        }
        source = np.zeros((nensemble, 2, 3, 1, 1), dtype=np.float32)
        for ensemble_index in range(nensemble):
            for time_index in range(2):
                for lead_index in range(3):
                    source[ensemble_index, time_index, lead_index, 0, 0] = (
                        100 * ensemble_index + 10 * time_index + lead_index
                    )
        io.add_array(coords, ["t2m"])
        io.write([source], coords, ["t2m"])
        return io

    monkeypatch.setattr(sys.modules["earth2studio.run"], "ensemble", fake_ensemble)

    ctx = _execution_context(tmp_path, run_id="ens-run-axis-order")
    module.WORKFLOW().run(
        {
            "forecast_times": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            "nsteps": 3,
            "nensemble": 2,
            "batch_size": 2,
            "create_plots": False,
        },
        ctx,
    )

    backend_kwargs = rust_backend_calls[0]["kwargs"]
    assert backend_kwargs["default_parallel_coord_names"] == [
        "ensemble",
        "time",
        "lead_time",
    ]
    written = rust_backend_calls[0]["write"][0]["x"][0]
    assert written.shape == (2, 2, 3, 1, 1)
    assert written[0, 1, 2, 0, 0] == 12.0
    assert written[1, 1, 2, 0, 0] == 112.0


def test_e2s_ensemble_plot_selects_by_dimension_names_for_rust_axis_order(
    tmp_path: Path, monkeypatch
):
    import numpy as np

    module = _load_module(
        "e2s_ensemble_plot_rust_axis_order",
        PLUGIN_ENSEMBLE / "workflow.py",
    )
    plotted_means: list[float] = []

    class FakeAxes:
        def pcolormesh(self, _lon, _lat, data, **_kwargs):
            plotted_means.append(float(np.asarray(data).mean()))
            return object()

        def set_title(self, _title):
            return None

        def coastlines(self):
            return None

        def gridlines(self):
            return None

        def set_visible(self, _visible):
            return None

    fake_pyplot = ModuleType("matplotlib.pyplot")
    fake_pyplot.close = lambda *_args, **_kwargs: None
    fake_pyplot.subplots = lambda **_kwargs: (
        object(),
        (FakeAxes(), FakeAxes(), FakeAxes()),
    )
    fake_pyplot.colorbar = lambda *_args, **_kwargs: None
    fake_pyplot.savefig = lambda *_args, **_kwargs: None
    fake_matplotlib = ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    fake_cartopy_crs = ModuleType("cartopy.crs")
    fake_cartopy_crs.Robinson = lambda: object()
    fake_cartopy_crs.PlateCarree = lambda: object()
    fake_cartopy = ModuleType("cartopy")
    fake_cartopy.crs = fake_cartopy_crs
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(sys.modules, "cartopy", fake_cartopy)
    monkeypatch.setitem(sys.modules, "cartopy.crs", fake_cartopy_crs)

    class FakeArray:
        def __init__(self, data, dims):
            self.data = np.asarray(data)
            self.attrs = {"_ARRAY_DIMENSIONS": list(dims)}
            self.ndim = self.data.ndim

        def __getitem__(self, key):
            return self.data[key]

    values = np.zeros((1, 2, 2, 1, 1), dtype=np.float32)
    values[0, 1, 0, :, :] = 10.0
    values[0, 1, 1, :, :] = 20.0
    io = {
        "lon": FakeArray(np.asarray([0.0], dtype=np.float32), ["lon"]),
        "lat": FakeArray(np.asarray([0.0], dtype=np.float32), ["lat"]),
        "t2m": FakeArray(
            values,
            ["time", "lead_time", "ensemble", "lat", "lon"],
        ),
    }

    workflow = module.WORKFLOW()
    workflow.output_dir = tmp_path
    workflow.create_ensemble_plot(
        io,
        forecast_times=["2024-01-01T00:00:00"],
        nsteps=2,
        nensemble=2,
        plot_variable="t2m",
        plot_step=1,
    )

    assert plotted_means == [10.0, 20.0, 5.0]


def test_e2s_ensemble_cleanup_releases_runtime_resources_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    package_close_calls: list[tuple[str, str, str]] = []
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        gfs_close_calls=gfs_close_calls,
        package_close_calls=package_close_calls,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_ensemble_cleanup_runtime_memory",
        PLUGIN_ENSEMBLE / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="ens-run-cleanup")
    workflow.run(
        {
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 4,
            "nensemble": 2,
            "batch_size": 2,
            "create_plots": False,
        },
        ctx,
    )

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]
    assert package_close_calls == []
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


# ===========================================================================
# 6. e2s-example-user
# ===========================================================================

PLUGIN_EXAMPLE_USER = REPO_ROOT / "plugins" / "e2s-example-user"


def test_e2s_example_user_manifest_uses_scheduler_backed_gpu_pipeline():
    manifest = yaml.safe_load(
        (PLUGIN_EXAMPLE_USER / "plugin.yaml").read_text(encoding="utf-8")
    )

    assert manifest["pipeline"] == {
        "profile": "default",
    }
    assert manifest["runtime"] == {
        "executor_class": "earth2-gpu",
        "prepare_executor_class": "earth2-cpu",
        "readiness_executor_class": "earth2-cpu",
    }


def test_e2s_example_user_prepare_accepts_defaults(tmp_path: Path):
    module = _load_module(
        "e2s_example_user_prepare",
        PLUGIN_EXAMPLE_USER / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            task_name="example_task",
            num_iterations=5,
            delay_seconds=0.5,
            generate_output=True,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-example-user", run_id="r1"),
    )
    assert prepared.inputs.task_name == "example_task"
    assert prepared.inputs.num_iterations == 5
    assert prepared.inputs.delay_seconds == 0.5
    assert prepared.inputs.generate_output is True


def test_e2s_example_user_run_generates_output_files(tmp_path: Path):
    module = _load_module(
        "e2s_example_user_run",
        PLUGIN_EXAMPLE_USER / "workflow.py",
    )
    ctx = _execution_context(tmp_path, run_id="user-run")
    result = module.WORKFLOW().run(
        module.ExampleUserInput(
            task_name="test_task",
            num_iterations=2,
            delay_seconds=0.0,
            generate_output=True,
        ),
        ctx,
    )

    assert result.status == "success"
    results_path = ctx.run_dir / "results.json"
    assert results_path.exists()
    summary_path = ctx.run_dir / "summary.txt"
    assert summary_path.exists()


def test_e2s_example_user_run_skips_output_when_disabled(tmp_path: Path):
    module = _load_module(
        "e2s_example_user_run_no_output",
        PLUGIN_EXAMPLE_USER / "workflow.py",
    )
    ctx = _execution_context(tmp_path, run_id="user-run-no-out")
    result = module.WORKFLOW().run(
        module.ExampleUserInput(
            task_name="test_task",
            num_iterations=2,
            delay_seconds=0.0,
            generate_output=False,
        ),
        ctx,
    )

    assert result.status == "success"
    assert not (ctx.run_dir / "results.json").exists()


# ===========================================================================
# 7. e2s-stormcast-fcn3
# ===========================================================================

PLUGIN_STORMCAST = REPO_ROOT / "plugins" / "e2s-stormcast-fcn3"


def test_e2s_stormcast_fcn3_prepare_model_cache_downloads_shared_artifacts(
    monkeypatch,
):
    pretrained_calls: list[str] = []
    _install_fake_earth2_runtime(monkeypatch, pretrained_calls=pretrained_calls)
    module = _load_module(
        "e2s_stormcast_fcn3_prepare_model_cache",
        PLUGIN_STORMCAST / "workflow.py",
    )

    result = module.prepare_model_cache({"workflow_id": "e2s-stormcast-fcn3"})

    assert result["model_names"] == ["FCN3", "StormCast"]
    assert "InterpModAFNO" in pretrained_calls
    assert "StormCast" in pretrained_calls


def test_e2s_stormcast_fcn3_cleanup_clears_conditioning_source_and_tempdir(
    tmp_path: Path, monkeypatch
):
    netcdf_backend_calls: list[dict[str, object]] = []
    stormcast_model_instances: list[object] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        netcdf_backend_calls=netcdf_backend_calls,
        stormcast_model_instances=stormcast_model_instances,
    )
    module = _load_module(
        "e2s_stormcast_fcn3_cleanup",
        PLUGIN_STORMCAST / "workflow.py",
    )

    workflow = module.WORKFLOW(fcn3_result_storage="file")
    ctx = _execution_context(tmp_path, run_id="sc-run-cleanup")
    workflow.run(
        {
            "start_time": "2024-01-01T00:00:00",
            "num_hours": 10,
            "run_stormcast": True,
        },
        ctx,
    )

    assert len(stormcast_model_instances) == 1
    stormcast = stormcast_model_instances[0]
    tmp_dir_path = Path(str(netcdf_backend_calls[0]["path"])).parent
    assert getattr(stormcast, "conditioning_data_source") is None
    assert not tmp_dir_path.exists()

    workflow.cleanup()

    assert getattr(stormcast, "conditioning_data_source") is None
    assert not tmp_dir_path.exists()


def test_e2s_stormcast_fcn3_cleanup_closes_gfs_filesystem_session(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    _install_fake_earth2_runtime(monkeypatch, gfs_close_calls=gfs_close_calls)
    module = _load_module(
        "e2s_stormcast_fcn3_cleanup_gfs_session",
        PLUGIN_STORMCAST / "workflow.py",
    )

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="sc-run-cleanup-gfs-session")
    workflow.run(
        {
            "start_time": "2024-01-01T00:00:00",
            "num_hours": 10,
            "run_stormcast": True,
        },
        ctx,
    )

    assert gfs_close_calls == []

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_e2s_stormcast_fcn3_cleanup_keeps_fcn3_package_filesystem_session_open(
    tmp_path: Path, monkeypatch
):
    package_close_calls: list[tuple[str, str, str]] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        package_close_calls=package_close_calls,
    )
    module = _load_module(
        "e2s_stormcast_fcn3_cleanup_package_session",
        PLUGIN_STORMCAST / "workflow.py",
    )

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="sc-run-cleanup-package-sessions")
    workflow.run(
        {
            "start_time": "2024-01-01T00:00:00",
            "num_hours": 10,
            "run_stormcast": True,
        },
        ctx,
    )

    assert package_close_calls == []

    workflow.cleanup()

    assert package_close_calls == []


def test_e2s_stormcast_fcn3_cleanup_releases_request_scoped_refs_and_torch_memory(
    tmp_path: Path, monkeypatch
):
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        cuda_empty_cache_calls=cuda_empty_cache_calls,
    )
    module = _load_module(
        "e2s_stormcast_fcn3_cleanup_runtime_memory",
        PLUGIN_STORMCAST / "workflow.py",
    )

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["e2s_workflow"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)

    workflow = module.WORKFLOW(fcn3_result_storage="file")
    ctx = _execution_context(tmp_path, run_id="sc-run-runtime-cleanup")
    workflow.run(
        {
            "start_time": "2024-01-01T00:00:00",
            "num_hours": 10,
            "run_stormcast": True,
        },
        ctx,
    )

    assert gc_collect_calls == []
    assert cuda_ipc_collect_calls == []
    assert cuda_empty_cache_calls == []

    workflow.cleanup()

    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


def test_e2s_stormcast_fcn3_prepare_coerces_input(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_stormcast_fcn3_prepare",
        PLUGIN_STORMCAST / "workflow.py",
    )
    prepared = module.WORKFLOW().prepare(
        _raw_request(
            start_time="2024-01-01T00:00:00",
            num_hours=10,
            run_stormcast=True,
        ),
        _prepare_context(tmp_path, workflow_id="e2s-stormcast-fcn3", run_id="r1"),
    )
    assert prepared.inputs["start_time"] == "2024-01-01T00:00:00"
    assert prepared.inputs["num_hours"] == 10
    assert prepared.inputs["run_stormcast"] is True


def test_e2s_stormcast_fcn3_run_produces_output(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_stormcast_fcn3_run",
        PLUGIN_STORMCAST / "workflow.py",
    )

    ctx = _execution_context(tmp_path, run_id="sc-run")
    result = module.WORKFLOW().run(
        {"start_time": "2024-01-01T00:00:00", "num_hours": 10, "run_stormcast": False},
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)


# ===========================================================================
# 8. e2s-foundry-fcn3
# ===========================================================================

PLUGIN_FOUNDRY_FCN3 = REPO_ROOT / "plugins" / "e2s-foundry-fcn3"


class _FakeFoundryRootVar:
    def __init__(self):
        object.__setattr__(self, "attrs", {})

    def __setattr__(self, name, value):
        if name != "attrs":
            self.attrs[name] = value
        object.__setattr__(self, name, value)


class _FakeFoundryRoot(dict):
    def __missing__(self, key):
        value = _FakeFoundryRootVar()
        self[key] = value
        return value


class _FakeFoundryIO:
    def __init__(self, path: str):
        self.path = Path(path)
        self.root = _FakeFoundryRoot()
        self.store = f"store:{path}"
        self.add_array_calls: list[dict[str, object]] = []
        self.write_calls: list[dict[str, object]] = []

    @staticmethod
    def _array_names(array_name) -> list[str]:
        if isinstance(array_name, str):
            return [array_name]
        if hasattr(array_name, "tolist"):
            array_name = array_name.tolist()
        if isinstance(array_name, tuple):
            array_name = list(array_name)
        if isinstance(array_name, list):
            return [str(item) for item in array_name]
        return [str(array_name)]

    def add_array(self, coords, array_name, data=None):
        for coord_name in coords:
            self.root[coord_name]
        for name in self._array_names(array_name):
            self.root[name]
        self.add_array_calls.append(
            {"coords": dict(coords), "array_name": array_name, "data": data}
        )

    def write(self, x, coords, array_name):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".written").write_text("ok", encoding="utf-8")
        self.write_calls.append(
            {"x": x, "coords": dict(coords), "array_name": array_name}
        )


def test_e2s_foundry_fcn3_prepare_coerces_input(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_prepare",
        PLUGIN_FOUNDRY_FCN3 / "workflow.py",
    )

    prepared = module.WORKFLOW().prepare(
        _raw_request(
            start_time="2025-01-01T00:00:00",
            n_steps=1,
            n_samples=1,
            seeds=[123],
            variables=["t2m"],
            output_format="zarr",
        ),
        _prepare_context(tmp_path, workflow_id="e2s-foundry-fcn3", run_id="r1"),
    )

    assert prepared.inputs["start_time"] == "2025-01-01T00:00:00"
    assert prepared.inputs["n_steps"] == 1
    assert prepared.inputs["n_samples"] == 1
    assert prepared.inputs["seeds"] == [123]
    assert prepared.inputs["variables"] == ["t2m"]
    assert prepared.inputs["output_format"] == "zarr"


def test_e2s_foundry_fcn3_run_produces_zarr_output_and_metadata(
    tmp_path: Path, monkeypatch
):
    data_source_init_calls: list[str] = []
    gfs_close_calls: list[tuple[str, str]] = []
    consolidate_metadata_calls: list[str] = []
    _install_fake_earth2_runtime(
        monkeypatch,
        data_source_init_calls=data_source_init_calls,
        gfs_close_calls=gfs_close_calls,
        consolidate_metadata_calls=consolidate_metadata_calls,
    )
    module = _load_module(
        "e2s_foundry_fcn3_run",
        PLUGIN_FOUNDRY_FCN3 / "workflow.py",
    )

    created_ios: list[_FakeFoundryIO] = []

    def fake_create_io(_self, dataset_path: str):
        io = _FakeFoundryIO(dataset_path)
        created_ios.append(io)
        return io

    monkeypatch.setattr(module.FoundryFCN3Workflow, "create_io", fake_create_io)

    workflow = module.WORKFLOW()
    ctx = _execution_context(tmp_path, run_id="foundry-fcn3-run")
    result = workflow.run(
        {
            "start_time": "2025-01-01T00:00:00",
            "n_steps": 1,
            "n_samples": 1,
            "seeds": [123],
            "variables": ["t2m"],
            "output_format": "zarr",
        },
        ctx,
    )

    expected_zarr = ctx.run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_zarr)
    assert ctx.outputs.primary_output().path == str(expected_zarr)
    assert expected_zarr.is_dir()

    io = created_ios[0]
    assert len(io.write_calls) == 2
    assert io.root["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
    assert io.root["t2m"].attrs["grid_mapping"] == "crs"
    assert io.root["time"].attrs["axis"] == "T"
    assert "PlanetaryComputerECMWFOpenDataIFS" in data_source_init_calls
    assert gfs_close_calls == []

    workflow.cleanup()

    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_e2s_foundry_fcn3_rejects_invalid_step_count(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_invalid_steps",
        PLUGIN_FOUNDRY_FCN3 / "workflow.py",
    )

    with pytest.raises(ValueError, match="n_steps must be between"):
        module.WORKFLOW().run(
            {
                "start_time": "2025-01-01T00:00:00",
                "n_steps": 0,
                "n_samples": 1,
                "variables": ["t2m"],
                "output_format": "zarr",
            },
            _execution_context(tmp_path, run_id="foundry-fcn3-invalid"),
        )


# ===========================================================================
# 9. e2s-foundry-fcn3-stormscope-goes
# ===========================================================================

PLUGIN_FOUNDRY_FCN3_STORMSCOPE = (
    REPO_ROOT / "plugins" / "e2s-foundry-fcn3-stormscope-goes"
)


def test_e2s_foundry_fcn3_stormscope_prepare_coerces_input(tmp_path: Path, monkeypatch):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_stormscope_prepare",
        PLUGIN_FOUNDRY_FCN3_STORMSCOPE / "workflow.py",
    )

    prepared = module.WORKFLOW().prepare(
        _raw_request(
            start_time_fcn3="2025-01-01T00:00:00",
            start_time_stormscope="2025-01-01T01:00:00",
            n_steps=2,
            n_samples_fcn3=1,
            n_samples_stormscope=2,
            seeds_fcn3=[11],
            seeds_stormscope=[21, 22],
            variables=["abi01c"],
        ),
        _prepare_context(
            tmp_path, workflow_id="e2s-foundry-fcn3-stormscope-goes", run_id="r1"
        ),
    )

    assert prepared.inputs["start_time_fcn3"] == "2025-01-01T00:00:00"
    assert prepared.inputs["start_time_stormscope"] == "2025-01-01T01:00:00"
    assert prepared.inputs["n_steps"] == 2
    assert prepared.inputs["n_samples_fcn3"] == 1
    assert prepared.inputs["n_samples_stormscope"] == 2
    assert prepared.inputs["seeds_fcn3"] == [11]
    assert prepared.inputs["seeds_stormscope"] == [21, 22]
    assert prepared.inputs["variables"] == ["abi01c"]


def test_e2s_foundry_fcn3_stormscope_rejects_invalid_step_count(
    tmp_path: Path, monkeypatch
):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_stormscope_invalid_steps",
        PLUGIN_FOUNDRY_FCN3_STORMSCOPE / "workflow.py",
    )

    with pytest.raises(ValueError, match="n_steps must be between"):
        module.WORKFLOW().run(
            {
                "start_time_fcn3": "2025-01-01T00:00:00",
                "start_time_stormscope": "2025-01-01T01:00:00",
                "n_steps": 0,
                "n_samples_fcn3": 1,
                "n_samples_stormscope": 1,
                "variables": ["abi01c"],
            },
            _execution_context(
                tmp_path, run_id="foundry-fcn3-stormscope-invalid-steps"
            ),
        )


def test_e2s_foundry_fcn3_stormscope_rejects_misaligned_start_time(
    tmp_path: Path, monkeypatch
):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_stormscope_invalid_start",
        PLUGIN_FOUNDRY_FCN3_STORMSCOPE / "workflow.py",
    )

    with pytest.raises(
        ValueError, match="Start time for StormScope must be 1-hour interval"
    ):
        module.WORKFLOW().run(
            {
                "start_time_fcn3": "2025-01-01T00:00:00",
                "start_time_stormscope": "2025-01-01T01:30:00",
                "n_steps": 2,
                "n_samples_fcn3": 1,
                "n_samples_stormscope": 1,
                "variables": ["abi01c"],
            },
            _execution_context(
                tmp_path, run_id="foundry-fcn3-stormscope-invalid-start"
            ),
        )


def test_e2s_foundry_fcn3_stormscope_rejects_indivisible_sample_counts(
    tmp_path: Path, monkeypatch
):
    _install_fake_earth2_runtime(monkeypatch)
    module = _load_module(
        "e2s_foundry_fcn3_stormscope_indivisible",
        PLUGIN_FOUNDRY_FCN3_STORMSCOPE / "workflow.py",
    )

    # Divisibility check fires after _ensure_runtime_loaded but before any IO use.
    # Bypass the heavy model load and pin self.stormscope so the None-check passes.
    def fake_ensure_runtime_loaded(self):
        self.stormscope = object()

    monkeypatch.setattr(
        module.FoundryFCN3StormScopeGOESWorkflow,
        "_ensure_runtime_loaded",
        fake_ensure_runtime_loaded,
    )

    with pytest.raises(
        ValueError, match="'n_samples_stormscope' must be divisible by 'n_samples_fcn3'"
    ):
        module.WORKFLOW().run(
            {
                "start_time_fcn3": "2025-01-01T00:00:00",
                "start_time_stormscope": "2025-01-01T01:00:00",
                "n_steps": 2,
                "n_samples_fcn3": 2,
                "n_samples_stormscope": 3,
                "variables": ["abi01c"],
            },
            _execution_context(tmp_path, run_id="foundry-fcn3-stormscope-indivisible"),
        )
