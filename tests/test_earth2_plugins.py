# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PYTHON_DIR = REPO_ROOT / "python"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import plugin_sdk as plugin_sdk_module  # noqa: E402
from plugin_sdk import (  # noqa: E402
    BatchExecutionContext,
    BatchItem,
    ExecutionContext,
    ExecutionInfo,
    OutputRegistry,
    PostprocessContext,
    PrepareContext,
    PriorResult,
    RawRequest,
    cleanup_earth2_runtime_resources,
)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _fanout_postprocess_context(tmp_path: Path, *, run_id: str) -> PostprocessContext:
    run_dir = tmp_path / run_id
    return PostprocessContext(
        run_id=run_id,
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        request=_raw_request(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
        ),
        resource_profile={"gpus_required": 1},
    )


def _fanout_prior_result(
    *,
    run_id: str,
    child_results: list[dict],
    aggregation_summary: dict,
) -> PriorResult[dict]:
    return PriorResult(
        payload={
            "child_results": child_results,
            "aggregation_summary": aggregation_summary,
        },
        execution=ExecutionInfo(run_id=run_id, status="succeeded", outputs=[]),
    )


def _install_fake_earth2_runtime(
    monkeypatch, *, gfs_close_calls=None, package_close_calls=None
):
    class FakeModel:
        def to(self, _device):
            return self

    class FakePerturbation:
        def __init__(self, **_kwargs):
            pass

    class FakeDLWP:
        @staticmethod
        def load_default_package():
            class FakePackageFilesystem:
                def __init__(self) -> None:
                    self.loop = "package-loop"
                    self._session = "package-session"

                def close_session(self, loop, session) -> None:
                    if package_close_calls is not None:
                        package_close_calls.append((loop, session))

            class FakePackage:
                def __init__(self) -> None:
                    self.fs = FakePackageFilesystem()

            return FakePackage()

        @staticmethod
        def load_model(_package):
            return FakeModel()

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.manual_seed = lambda _seed: None

    fake_models = ModuleType("earth2studio.models")
    fake_models_px = ModuleType("earth2studio.models.px")
    fake_models_px.DLWP = FakeDLWP
    fake_data = ModuleType("earth2studio.data")

    class FakeFilesystem:
        def __init__(self) -> None:
            self.loop = "fake-loop"
            self._s3creator = "fake-s3creator"

        def close_session(self, loop, s3creator) -> None:
            if gfs_close_calls is not None:
                gfs_close_calls.append((loop, s3creator))

    class FakeDataSource:
        def __init__(self) -> None:
            self.fs = FakeFilesystem()

    fake_data.GFS = FakeDataSource
    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = lambda path: path
    fake_run = ModuleType("earth2studio.run")

    def fake_deterministic(*, io, **_kwargs):
        output_path = Path(io)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / ".written").write_text("ok", encoding="utf-8")

    fake_run.deterministic = fake_deterministic
    fake_run.ensemble = fake_deterministic
    fake_perturbation = ModuleType("earth2studio.perturbation")
    fake_perturbation.Gaussian = FakePerturbation
    fake_perturbation.Brown = FakePerturbation
    fake_perturbation.SphericalGaussian = FakePerturbation
    fake_utils = ModuleType("earth2studio.utils")
    fake_utils_time = ModuleType("earth2studio.utils.time")
    fake_utils_time.to_time_array = lambda values: values

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "earth2studio", ModuleType("earth2studio"))
    monkeypatch.setitem(sys.modules, "earth2studio.models", fake_models)
    monkeypatch.setitem(sys.modules, "earth2studio.models.px", fake_models_px)
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)
    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.run", fake_run)
    monkeypatch.setitem(sys.modules, "earth2studio.perturbation", fake_perturbation)
    monkeypatch.setitem(sys.modules, "earth2studio.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_utils_time)


def _install_fake_torch_gpu_cleanup(monkeypatch):
    cuda_empty_cache_calls: list[str] = []
    cuda_ipc_collect_calls: list[str] = []
    gc_collect_calls: list[str] = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            cuda_empty_cache_calls.append("empty_cache")

        def ipc_collect(self) -> None:
            cuda_ipc_collect_calls.append("ipc_collect")

    monkeypatch.setattr(
        sys.modules["plugin_sdk"].gc,
        "collect",
        lambda: gc_collect_calls.append("collect"),
    )
    monkeypatch.setattr(sys.modules["torch"], "cuda", FakeCuda(), raising=False)
    return gc_collect_calls, cuda_ipc_collect_calls, cuda_empty_cache_calls


def test_earth2_deterministic_prepare_returns_prepare_result_with_gpu_profile(
    tmp_path: Path,
):
    module = _load_module(
        "earth2_deterministic_workflow_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    workflow = module.DeterministicWorkflow()
    prepared = workflow.prepare(
        _raw_request(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=2,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-deterministic",
            run_id="deterministic-prepare",
        ),
    )

    assert prepared.resource_profile is None
    assert prepared.prefetch_plan == []


def test_earth2_deterministic_prepare_uses_manifest_defaults_for_cpu_profile(
    tmp_path: Path,
):
    module = _load_module(
        "earth2_deterministic_workflow_cpu_prepare_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    workflow = module.DeterministicWorkflow()
    prepared = workflow.prepare(
        _raw_request(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=2,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-deterministic",
            run_id="deterministic-cpu-prepare",
        ),
    )

    assert prepared.resource_profile is None
    assert prepared.prefetch_plan == []


def test_earth2_deterministic_batch_prepare_returns_batch_profile(tmp_path: Path):
    module = _load_module(
        "earth2_deterministic_batch_workflow_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    assert module.WORKFLOW is module.DeterministicBatchWorkflow
    workflow = module.WORKFLOW()
    prepared = workflow.prepare(
        _raw_request(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=2,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-deterministic-batch",
            run_id="deterministic-batch-prepare",
        ),
    )

    assert prepared.inputs.model == "dlwp"
    assert prepared.resource_profile["executor_class"] == "earth2-gpu"
    assert prepared.batch_profile["enabled"] is True
    assert prepared.batch_profile["batch_key"] == "dlwp"


def test_earth2_deterministic_batch_execute_registers_forecast_output(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_deterministic_batch_workflow_execute_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)

    run_dir = tmp_path / "deterministic-batch-run"
    outputs = OutputRegistry(run_dir)
    result = module.DeterministicBatchWorkflow().execute(
        {
            "run_id": "deterministic-batch-run",
            "parameters": {
                "model": "dlwp",
                "start_time": "2026-01-01T00:00:00Z",
                "nsteps": 1,
            },
            "outputs": outputs,
            "resource_profile": {"gpus_required": 1},
        }
    )

    expected_dataset = run_dir / "forecast.zarr"
    assert result["dataset_path"] == str(expected_dataset)
    assert outputs.primary_output().path == str(expected_dataset)


def test_earth2_deterministic_batch_run_batch_registers_forecast_outputs(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_deterministic_batch_workflow_run_batch_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)

    workflow = module.WORKFLOW()
    batch_ctx = BatchExecutionContext(
        batch_id="deterministic-batch-run",
        run_dir=tmp_path / "deterministic-batch-run",
        batch_info={"batch_id": "deterministic-batch-run", "batch_size": 2},
        resource_profile={"gpus_required": 1},
    )
    item_contexts = [
        ExecutionContext(
            run_id=f"deterministic-batch-run:item:{index}",
            run_dir=tmp_path / f"deterministic-batch-run:item:{index}",
            outputs=OutputRegistry(tmp_path / f"deterministic-batch-run:item:{index}"),
            resource_profile={"gpus_required": 1},
        )
        for index in range(2)
    ]
    items = [
        BatchItem(
            index=index,
            inputs=module.DeterministicBatchInput(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
            ),
            context=item_contexts[index],
        )
        for index in range(2)
    ]

    results = workflow.run_batch(items, batch_ctx)

    assert len(results) == 2
    for index, result in enumerate(results):
        expected_dataset = item_contexts[index].run_dir / "forecast.zarr"
        assert result.dataset_path == str(expected_dataset)
        assert item_contexts[index].outputs.primary_output().path == str(
            expected_dataset
        )


def test_earth2_deterministic_batch_run_batch_closes_gfs_filesystem_session(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_deterministic_batch_workflow_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch, gfs_close_calls=gfs_close_calls)

    workflow = module.WORKFLOW()
    batch_ctx = BatchExecutionContext(
        batch_id="deterministic-batch-cleanup",
        run_dir=tmp_path / "deterministic-batch-cleanup",
        batch_info={"batch_id": "deterministic-batch-cleanup", "batch_size": 2},
        resource_profile={"gpus_required": 1},
    )
    item_context = ExecutionContext(
        run_id="deterministic-batch-cleanup:item:0",
        run_dir=tmp_path / "deterministic-batch-cleanup:item:0",
        outputs=OutputRegistry(tmp_path / "deterministic-batch-cleanup:item:0"),
        resource_profile={"gpus_required": 1},
    )
    items = [
        BatchItem(
            index=0,
            inputs=module.DeterministicBatchInput(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
            ),
            context=item_context,
        )
    ]

    workflow.run_batch(items, batch_ctx)

    assert gfs_close_calls == []
    workflow.cleanup()
    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_earth2_deterministic_batch_run_batch_keeps_package_filesystem_session_open(
    tmp_path: Path, monkeypatch
):
    package_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_deterministic_batch_workflow_package_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    _install_fake_earth2_runtime(
        monkeypatch,
        package_close_calls=package_close_calls,
    )

    workflow = module.WORKFLOW()
    batch_ctx = BatchExecutionContext(
        batch_id="deterministic-batch-package-cleanup",
        run_dir=tmp_path / "deterministic-batch-package-cleanup",
        batch_info={
            "batch_id": "deterministic-batch-package-cleanup",
            "batch_size": 2,
        },
        resource_profile={"gpus_required": 1},
    )
    item_context = ExecutionContext(
        run_id="deterministic-batch-package-cleanup:item:0",
        run_dir=tmp_path / "deterministic-batch-package-cleanup:item:0",
        outputs=OutputRegistry(tmp_path / "deterministic-batch-package-cleanup:item:0"),
        resource_profile={"gpus_required": 1},
    )
    items = [
        BatchItem(
            index=0,
            inputs=module.DeterministicBatchInput(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
            ),
            context=item_context,
        )
    ]

    workflow.run_batch(items, batch_ctx)

    assert package_close_calls == []


def test_earth2_deterministic_batch_run_batch_releases_torch_gpu_memory(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_deterministic_batch_workflow_gpu_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic-batch" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)
    gc_collect_calls, cuda_ipc_collect_calls, cuda_empty_cache_calls = (
        _install_fake_torch_gpu_cleanup(monkeypatch)
    )

    workflow = module.WORKFLOW()
    batch_ctx = BatchExecutionContext(
        batch_id="deterministic-batch-gpu-cleanup",
        run_dir=tmp_path / "deterministic-batch-gpu-cleanup",
        batch_info={"batch_id": "deterministic-batch-gpu-cleanup", "batch_size": 1},
        resource_profile={"gpus_required": 1},
    )
    item_context = ExecutionContext(
        run_id="deterministic-batch-gpu-cleanup:item:0",
        run_dir=tmp_path / "deterministic-batch-gpu-cleanup:item:0",
        outputs=OutputRegistry(tmp_path / "deterministic-batch-gpu-cleanup:item:0"),
        resource_profile={"gpus_required": 1},
    )
    items = [
        BatchItem(
            index=0,
            inputs=module.DeterministicBatchInput(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
            ),
            context=item_context,
        )
    ]

    workflow.run_batch(items, batch_ctx)

    assert gc_collect_calls == []
    workflow.cleanup()
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


def test_earth2_deterministic_run_registers_forecast_output(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_deterministic_workflow_run_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)

    run_dir = tmp_path / "deterministic-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="deterministic-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    result = module.DeterministicWorkflow().run(
        module.DeterministicInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
        ),
        ctx,
    )

    expected_dataset = run_dir / "forecast.zarr"
    assert result.dataset_path == str(expected_dataset)
    assert outputs.primary_output().path == str(expected_dataset)


def test_earth2_deterministic_run_closes_gfs_filesystem_session(
    tmp_path: Path, monkeypatch
):
    gfs_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_deterministic_workflow_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch, gfs_close_calls=gfs_close_calls)

    run_dir = tmp_path / "deterministic-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="deterministic-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.DeterministicWorkflow()
    workflow.run(
        module.DeterministicInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
        ),
        ctx,
    )

    assert gfs_close_calls == []
    workflow.cleanup()
    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_earth2_deterministic_run_keeps_package_filesystem_session_open(
    tmp_path: Path, monkeypatch
):
    package_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_deterministic_workflow_package_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    _install_fake_earth2_runtime(
        monkeypatch,
        package_close_calls=package_close_calls,
    )

    run_dir = tmp_path / "deterministic-package-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="deterministic-package-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.DeterministicWorkflow()
    workflow.run(
        module.DeterministicInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
        ),
        ctx,
    )

    assert package_close_calls == []


def test_earth2_deterministic_run_releases_torch_gpu_memory(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_deterministic_workflow_gpu_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)
    gc_collect_calls, cuda_ipc_collect_calls, cuda_empty_cache_calls = (
        _install_fake_torch_gpu_cleanup(monkeypatch)
    )

    run_dir = tmp_path / "deterministic-gpu-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="deterministic-gpu-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.DeterministicWorkflow()
    workflow.run(
        module.DeterministicInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
        ),
        ctx,
    )

    assert gc_collect_calls == []
    workflow.cleanup()
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


def test_cleanup_earth2_runtime_resources_closes_http_filesystem_session():
    http_close_calls: list[str] = []

    class FakeHTTPFilesystem:
        def __init__(self) -> None:
            self.loop = "http-loop"
            self._session = "http-session"

        def close_session(self, loop) -> None:
            http_close_calls.append(loop)

    cleanup_earth2_runtime_resources(SimpleNamespace(fs=FakeHTTPFilesystem()))

    assert http_close_calls == ["http-loop"]


def test_cleanup_earth2_runtime_resources_closes_wrapped_http_session():
    http_close_calls: list[tuple[str, str]] = []

    class FakeInnerHTTPFilesystem:
        def __init__(self) -> None:
            self._session = "inner-http-session"

    def raw_close_session(loop, session) -> None:
        http_close_calls.append((loop, session))

    class FakeWrappedCloseSession:
        def __init__(self, inner: FakeInnerHTTPFilesystem) -> None:
            self.__self__ = inner
            self.__func__ = raw_close_session
            self.__signature__ = inspect.Signature(
                [
                    inspect.Parameter(
                        "session",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
            )

        def __call__(self, session) -> None:  # pragma: no cover - defensive
            raise AssertionError(
                "cleanup should call the underlying raw close_session function"
            )

    class FakeWrappedHTTPFilesystem:
        def __init__(self) -> None:
            self.loop = "wrapped-http-loop"
            self._session = None
            self.close_session = FakeWrappedCloseSession(FakeInnerHTTPFilesystem())

    cleanup_earth2_runtime_resources(SimpleNamespace(fs=FakeWrappedHTTPFilesystem()))

    assert http_close_calls == [("wrapped-http-loop", "inner-http-session")]


def test_cleanup_earth2_runtime_resources_prefers_async_session_owner():
    close_session_calls: list[tuple[str, object]] = []

    class FakeS3Creator:
        async def __aexit__(self, *_exc_info) -> None:
            return None

    class FakeAsyncS3Filesystem:
        def __init__(self) -> None:
            self.loop = None
            self._s3 = object()
            self._s3creator = FakeS3Creator()

        @property
        def s3(self) -> object:  # pragma: no cover - defensive
            raise AssertionError("cleanup should not inspect the s3 property")

        def close_session(self, loop, session_owner) -> None:
            close_session_calls.append((loop, session_owner))

    filesystem = FakeAsyncS3Filesystem()
    cleanup_earth2_runtime_resources(SimpleNamespace(fs=filesystem))

    assert close_session_calls == [(None, filesystem._s3creator)]


def test_enable_http_session_tracing_logs_session_lifecycle(monkeypatch, caplog):
    monkeypatch.setattr(
        plugin_sdk_module,
        "_HTTP_SESSION_TRACING_INSTALLED",
        False,
    )
    monkeypatch.delenv("PHYSICSNEMO_SERVE_TRACE_HTTP_SESSIONS", raising=False)

    fake_aiohttp = ModuleType("aiohttp")

    class FakeConnector:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeClientSession:
        def __init__(self) -> None:
            self._connector = FakeConnector()

        @property
        def connector(self):
            return self._connector

        @property
        def closed(self) -> bool:
            return self._connector is None or self._connector.closed

        async def close(self) -> None:
            if self._connector is not None:
                self._connector.close()
                self._connector = None

        def __del__(self) -> None:
            return None

    fake_aiohttp.ClientSession = FakeClientSession

    fake_aiobotocore = ModuleType("aiobotocore")
    fake_aiobotocore_httpsession = ModuleType("aiobotocore.httpsession")

    class FakeAIOHTTPSession:
        def __init__(self) -> None:
            self._sessions = {}

        async def _get_session(self, proxy_url):
            session = FakeClientSession()
            self._sessions[proxy_url] = session
            return session

        async def close(self) -> None:
            for session in list(self._sessions.values()):
                await session.close()
            self._sessions.clear()

    fake_aiobotocore.httpsession = fake_aiobotocore_httpsession
    fake_aiobotocore_httpsession.AIOHTTPSession = FakeAIOHTTPSession

    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "aiobotocore", fake_aiobotocore)
    monkeypatch.setitem(
        sys.modules, "aiobotocore.httpsession", fake_aiobotocore_httpsession
    )

    caplog.set_level(logging.DEBUG, logger="plugin_sdk")
    plugin_sdk_module._enable_http_session_tracing()

    http_session = FakeAIOHTTPSession()
    client_session = asyncio.run(http_session._get_session("https://proxy.internal"))
    asyncio.run(client_session.close())
    leaking_session = asyncio.run(http_session._get_session("https://proxy.gc"))

    leaking_session.__del__()

    assert "aiobotocore trace: get_session owner=" in caplog.text
    assert "aiohttp trace: closing session=" in caplog.text
    assert "aiohttp trace: closed session=" in caplog.text
    assert "recovered gc session via owner close" in caplog.text
    assert "session garbage collected without explicit close" not in caplog.text
    assert leaking_session.closed is True


def test_http_session_tracing_can_be_disabled_with_env(monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_TRACE_HTTP_SESSIONS", "0")

    assert plugin_sdk_module._http_session_tracing_requested() is False


def test_close_aiobotocore_http_sessions_closes_reachable_sessions():
    class FakeConnector:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeClientSession:
        def __init__(self) -> None:
            self._connector = FakeConnector()

        @property
        def connector(self):
            return self._connector

        @property
        def closed(self) -> bool:
            return self._connector is None or self._connector.closed

        async def close(self) -> None:
            if self._connector is not None:
                self._connector.close()
                self._connector = None

    class FakeAIOHTTPSession:
        def __init__(self, session) -> None:
            self._sessions = {None: session}

        async def close(self) -> None:
            for session in list(self._sessions.values()):
                await session.close()
            self._sessions.clear()

    session = FakeClientSession()
    http_session = FakeAIOHTTPSession(session)
    candidate = SimpleNamespace(
        _s3=SimpleNamespace(_endpoint=SimpleNamespace(http_session=http_session))
    )

    closed = plugin_sdk_module._close_aiobotocore_http_sessions(candidate)

    assert closed == 1
    assert session.closed is True
    assert http_session._sessions == {}


def test_close_live_aiobotocore_http_sessions_closes_orphaned_httpsessions(
    monkeypatch,
):
    class FakeConnector:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeClientSession:
        def __init__(self) -> None:
            self._connector = FakeConnector()

        @property
        def connector(self):
            return self._connector

        @property
        def closed(self) -> bool:
            return self._connector is None or self._connector.closed

        async def close(self) -> None:
            if self._connector is not None:
                self._connector.close()
                self._connector = None

    class FakeAIOHTTPSession:
        def __init__(self, session) -> None:
            self._sessions = {None: session}

        async def close(self) -> None:
            for session in list(self._sessions.values()):
                await session.close()
            self._sessions.clear()

    fake_aiobotocore = ModuleType("aiobotocore")
    fake_aiobotocore_httpsession = ModuleType("aiobotocore.httpsession")
    fake_aiobotocore.httpsession = fake_aiobotocore_httpsession
    fake_aiobotocore_httpsession.AIOHTTPSession = FakeAIOHTTPSession
    monkeypatch.setitem(sys.modules, "aiobotocore", fake_aiobotocore)
    monkeypatch.setitem(
        sys.modules, "aiobotocore.httpsession", fake_aiobotocore_httpsession
    )

    session = FakeClientSession()
    http_session = FakeAIOHTTPSession(session)
    monkeypatch.setattr(
        plugin_sdk_module.gc, "get_objects", lambda: [object(), http_session]
    )

    closed = plugin_sdk_module._close_live_aiobotocore_http_sessions()

    assert closed == 1
    assert session.closed is True
    assert http_session._sessions == {}


def test_earth2_deterministic_run_rejects_unknown_model(tmp_path: Path):
    module = _load_module(
        "earth2_deterministic_workflow_invalid_model_test",
        REPO_ROOT / "plugins" / "earth2-deterministic" / "workflow.py",
    )

    run_dir = tmp_path / "deterministic-invalid-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="deterministic-invalid-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    with pytest.raises(ValueError, match="supports only model='dlwp'"):
        module.DeterministicWorkflow().run(
            module.DeterministicInput(
                model="unknown-model",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
            ),
            ctx,
        )


def test_earth2_ensemble_execute_registers_forecast_output(tmp_path: Path, monkeypatch):
    module = _load_module(
        "earth2_ensemble_workflow_execute_test",
        REPO_ROOT / "plugins" / "earth2-ensemble" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)

    run_dir = tmp_path / "ensemble-run"
    outputs = OutputRegistry(run_dir)
    assert module.WORKFLOW is module.EnsembleWorkflow
    result = module.WORKFLOW().execute(
        {
            "run_id": "ensemble-run",
            "parameters": {
                "model": "dlwp",
                "start_time": "2026-01-01T00:00:00Z",
                "nsteps": 1,
                "nensemble": 2,
                "batch_size": 1,
                "perturbation": "gaussian",
                "noise_amplitude": 0.05,
                "seed_base": 1000,
            },
            "outputs": outputs,
            "resource_profile": {"gpus_required": 1},
        }
    )

    expected_dataset = run_dir / "forecast-ensemble.zarr"
    assert result["dataset_path"] == str(expected_dataset)
    assert "status" not in result
    assert "output_path" not in result
    assert outputs.primary_output().path == str(expected_dataset)


def test_earth2_ensemble_run_closes_gfs_filesystem_session(tmp_path: Path, monkeypatch):
    gfs_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_ensemble_workflow_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-ensemble" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch, gfs_close_calls=gfs_close_calls)

    run_dir = tmp_path / "ensemble-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="ensemble-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.EnsembleWorkflow()
    workflow.run(
        module.EnsembleInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
            perturbation="gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
        ),
        ctx,
    )

    assert gfs_close_calls == []
    workflow.cleanup()
    assert gfs_close_calls == [("fake-loop", "fake-s3creator")]


def test_earth2_ensemble_run_keeps_ensemble_package_filesystem_session_open(
    tmp_path: Path, monkeypatch
):
    package_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_ensemble_workflow_package_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-ensemble" / "workflow.py",
    )
    _install_fake_earth2_runtime(
        monkeypatch,
        package_close_calls=package_close_calls,
    )

    run_dir = tmp_path / "ensemble-package-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="ensemble-package-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.EnsembleWorkflow()
    workflow.run(
        module.EnsembleInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
            perturbation="gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
        ),
        ctx,
    )

    assert package_close_calls == []


def test_earth2_ensemble_run_releases_torch_gpu_memory(tmp_path: Path, monkeypatch):
    module = _load_module(
        "earth2_ensemble_workflow_gpu_cleanup_test",
        REPO_ROOT / "plugins" / "earth2-ensemble" / "workflow.py",
    )
    _install_fake_earth2_runtime(monkeypatch)
    gc_collect_calls, cuda_ipc_collect_calls, cuda_empty_cache_calls = (
        _install_fake_torch_gpu_cleanup(monkeypatch)
    )

    run_dir = tmp_path / "ensemble-gpu-cleanup-run"
    outputs = OutputRegistry(run_dir)
    ctx = ExecutionContext(
        run_id="ensemble-gpu-cleanup-run",
        run_dir=run_dir,
        outputs=outputs,
        resource_profile={"gpus_required": 1},
    )

    workflow = module.EnsembleWorkflow()
    workflow.run(
        module.EnsembleInput(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
            perturbation="gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
        ),
        ctx,
    )

    assert gc_collect_calls == []
    workflow.cleanup()
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]
