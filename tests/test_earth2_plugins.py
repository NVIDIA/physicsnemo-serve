# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
import importlib.util
import inspect
import json
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


def _spherical_gaussian_member_ids(
    batch_index: int, *, nensemble: int, batch_size: int
) -> list[int]:
    batch_start = batch_index * batch_size
    batch_end = min(batch_start + batch_size, nensemble)
    return list(range(batch_start, batch_end))


_FCN_VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
    "u100m",
    "v100m",
    "u250",
    "v250",
    "z250",
    "t250",
]


def _spherical_gaussian_coords(
    member_ids: list[int], *, nlat: int, nlon: int, variables: list[str]
):
    import numpy as np
    from collections import OrderedDict

    return OrderedDict(
        [
            ("ensemble", np.asarray(member_ids, dtype=np.int64)),
            ("time", np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]")),
            ("lead_time", np.asarray([0], dtype="timedelta64[h]")),
            ("variable", np.asarray(variables)),
            ("lat", np.linspace(90.0, -90.0, nlat, endpoint=False)),
            ("lon", np.linspace(0.0, 360.0, nlon, endpoint=False)),
        ]
    )


def _synthetic_spherical_gaussian_batch(
    member_ids: list[int], *, nlat: int, nlon: int, variables: list[str], device
):
    import torch

    shape = (len(member_ids), 1, 1, len(variables), nlat, nlon)
    return torch.zeros(shape, dtype=torch.float32, device=device)


def _apply_stock_spherical_gaussian_batch(
    perturbation,
    member_ids: list[int],
    *,
    nlat: int,
    nlon: int,
    variables: list[str],
    device,
):
    x = _synthetic_spherical_gaussian_batch(
        member_ids, nlat=nlat, nlon=nlon, variables=variables, device=device
    )
    coords = _spherical_gaussian_coords(
        member_ids, nlat=nlat, nlon=nlon, variables=variables
    )
    perturbed_x, _coords = perturbation(x, coords)
    return perturbed_x


def _reference_spherical_gaussian_batches(
    *,
    seed_base: int,
    nensemble: int,
    batch_size: int,
    nlat: int,
    nlon: int,
    variables: list[str],
    device,
):
    import torch
    from earth2studio.perturbation import SphericalGaussian

    torch.manual_seed(seed_base)
    perturbation = SphericalGaussian(noise_amplitude=0.15)
    batches = []
    for batch_index in range((nensemble + batch_size - 1) // batch_size):
        member_ids = _spherical_gaussian_member_ids(
            batch_index, nensemble=nensemble, batch_size=batch_size
        )
        batches.append(
            _apply_stock_spherical_gaussian_batch(
                perturbation,
                member_ids,
                nlat=nlat,
                nlon=nlon,
                variables=variables,
                device=device,
            )
        )
    return batches


def _skip_ahead_spherical_gaussian_batch(
    *,
    seed_base: int,
    batch_index: int,
    nensemble: int,
    batch_size: int,
    nlat: int,
    nlon: int,
    variables: list[str],
    device,
):
    import torch
    from earth2studio.perturbation import SphericalGaussian

    torch.manual_seed(seed_base)
    perturbation = SphericalGaussian(noise_amplitude=0.15)
    for prior_batch_index in range(batch_index):
        prior_member_ids = _spherical_gaussian_member_ids(
            prior_batch_index, nensemble=nensemble, batch_size=batch_size
        )
        _apply_stock_spherical_gaussian_batch(
            perturbation,
            prior_member_ids,
            nlat=nlat,
            nlon=nlon,
            variables=variables,
            device=device,
        )

    member_ids = _spherical_gaussian_member_ids(
        batch_index, nensemble=nensemble, batch_size=batch_size
    )
    return _apply_stock_spherical_gaussian_batch(
        perturbation,
        member_ids,
        nlat=nlat,
        nlon=nlon,
        variables=variables,
        device=device,
    )


def test_spherical_gaussian_skip_ahead_matches_sequential_gpu_batches():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_harmonics")
    pytest.importorskip("earth2studio.perturbation")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SphericalGaussian skip-ahead parity")

    device = torch.device("cuda")
    seed_base = 1000
    nensemble = 5
    batch_size = 2
    nlat = 8
    nlon = 16
    variables = ["t2m", "z500"]

    reference_batches = _reference_spherical_gaussian_batches(
        seed_base=seed_base,
        nensemble=nensemble,
        batch_size=batch_size,
        nlat=nlat,
        nlon=nlon,
        variables=variables,
        device=device,
    )

    for batch_index, reference_batch in enumerate(reference_batches):
        skip_ahead_batch = _skip_ahead_spherical_gaussian_batch(
            seed_base=seed_base,
            batch_index=batch_index,
            nensemble=nensemble,
            batch_size=batch_size,
            nlat=nlat,
            nlon=nlon,
            variables=variables,
            device=device,
        )
        torch.testing.assert_close(skip_ahead_batch, reference_batch, rtol=0, atol=0)


def test_spherical_gaussian_skip_ahead_matches_fcn_input_shape_on_gpu():
    import os

    if os.environ.get("PHYSICSNEMO_SERVE_RUN_FULL_FCN_PARITY") != "1":
        pytest.skip(
            "Set PHYSICSNEMO_SERVE_RUN_FULL_FCN_PARITY=1 to run full FCN-shape GPU parity"
        )

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_harmonics")
    pytest.importorskip("earth2studio.perturbation")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FCN-shape SphericalGaussian parity")

    device = torch.device("cuda")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    min_free_bytes = 60 * 1024 * 1024 * 1024
    if free_bytes < min_free_bytes:
        pytest.skip(
            "Full FCN-shape SphericalGaussian parity is resource-heavy; "
            f"requires at least 60 GiB free CUDA memory, found {free_bytes} bytes"
        )

    seed_base = 1000
    nensemble = 2
    batch_size = 1
    nlat = 720
    nlon = 1440

    try:
        reference_batches = _reference_spherical_gaussian_batches(
            seed_base=seed_base,
            nensemble=nensemble,
            batch_size=batch_size,
            nlat=nlat,
            nlon=nlon,
            variables=_FCN_VARIABLES,
            device=device,
        )
        skip_ahead_batch = _skip_ahead_spherical_gaussian_batch(
            seed_base=seed_base,
            batch_index=1,
            nensemble=nensemble,
            batch_size=batch_size,
            nlat=nlat,
            nlon=nlon,
            variables=_FCN_VARIABLES,
            device=device,
        )
        torch.testing.assert_close(
            skip_ahead_batch, reference_batches[1], rtol=0, atol=0
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        pytest.skip(f"Not enough GPU memory for FCN-shape parity test: {exc}")


def test_earth2_ensemble_fanout_prepare_returns_materialization_payload(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_prepare",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    prepared = module.prepare_ensemble_fanout_request(
        _raw_request(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=2,
            max_in_flight=1,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-ensemble-fanout",
            run_id="parent-run",
        ),
    )

    assert prepared["operation"] == "materialize_perturbations"
    assert prepared["resource_profile"]["executor_class"] == "earth2-gpu"
    assert prepared["parameters"]["model"] == "fcn"
    assert prepared["parameters"]["perturbation"] == "spherical_gaussian"
    assert (
        prepared["parameters"]["perturbation_materialization_mode"] == "scheduled_gpu"
    )
    assert prepared["parameters"]["batch_size"] == 2
    assert "fanout_items" not in prepared


@pytest.mark.parametrize(
    "perturbation",
    ["gaussian", "brown", "spherical_gaussian"],
)
def test_earth2_ensemble_fanout_prepare_accepts_supported_perturbations(
    tmp_path: Path, perturbation: str
):
    module = _load_module(
        f"earth2_ensemble_fanout_support_test_prepare_{perturbation}",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    prepared = module.prepare_ensemble_fanout_request(
        _raw_request(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
            max_in_flight=1,
            perturbation=perturbation,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-ensemble-fanout",
            run_id=f"parent-{perturbation}-run",
        ),
    )

    assert prepared["operation"] == "materialize_perturbations"
    assert prepared["parameters"]["perturbation"] == perturbation


@pytest.mark.parametrize(
    "perturbation,expected_class_name",
    [
        ("gaussian", "FakeGaussian"),
        ("brown", "FakeBrown"),
        ("spherical_gaussian", "FakeSphericalGaussian"),
    ],
)
def test_earth2_ensemble_fanout_builds_supported_perturbations(
    monkeypatch, perturbation: str, expected_class_name: str
):
    module = _load_module(
        f"earth2_ensemble_fanout_support_test_build_{perturbation}",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class FakePerturbation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGaussian(FakePerturbation):
        pass

    class FakeBrown(FakePerturbation):
        pass

    class FakeSphericalGaussian(FakePerturbation):
        pass

    fake_perturbation = ModuleType("earth2studio.perturbation")
    fake_perturbation.Gaussian = FakeGaussian
    fake_perturbation.Brown = FakeBrown
    fake_perturbation.SphericalGaussian = FakeSphericalGaussian
    monkeypatch.setitem(sys.modules, "earth2studio", ModuleType("earth2studio"))
    monkeypatch.setitem(sys.modules, "earth2studio.perturbation", fake_perturbation)

    built, normalized = module._build_perturbation(perturbation, 0.05)

    assert type(built).__name__ == expected_class_name
    assert built.kwargs == {"noise_amplitude": 0.05}
    assert normalized == perturbation


def test_earth2_ensemble_fanout_prepare_cpu_materializes_and_skips_to_fanout(
    tmp_path: Path, monkeypatch
):
    np = pytest.importorskip("numpy")
    module = _load_module(
        "earth2_ensemble_fanout_support_prepare_cpu_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    saved: list[tuple[Path, dict[str, object]]] = []
    selected_devices: list[object] = []
    perturbation_calls: list[list[int]] = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name

        def cpu(self):
            return FakeTensor(f"{self.name}:cpu")

    class FakeModel:
        def to(self, device):
            selected_devices.append(device)
            return self

        def input_coords(self):
            return {
                "variable": ["t2m"],
                "lead_time": [0],
            }

    class FakePerturbation:
        def __call__(self, batch_x, batch_coords):
            perturbation_calls.append(batch_coords["ensemble"].tolist())
            return batch_x, batch_coords

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda name: name
    fake_torch.manual_seed = lambda seed: saved.append((Path("seed"), {"seed": seed}))

    def fake_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("prepared", encoding="utf-8")
        saved.append((path, payload))

    fake_torch.save = fake_save
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_data = ModuleType("earth2studio.data")
    fake_data.GFS = lambda: "gfs"
    fake_data.fetch_data = lambda **_kwargs: (FakeTensor("x0"), {"coords": "coords0"})
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)

    fake_time = ModuleType("earth2studio.utils.time")
    fake_time.to_time_array = lambda value: value
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_time)

    monkeypatch.setattr(module, "_load_model", lambda _model: (object(), FakeModel()))
    monkeypatch.setattr(
        module,
        "_build_perturbation",
        lambda _name, _noise: (FakePerturbation(), "spherical_gaussian"),
    )
    monkeypatch.setattr(
        module,
        "_build_batch_initial_conditions",
        lambda _x0, _coords0, _prognostic_ic, member_ids: (
            FakeTensor(f"batch-{member_ids[0]}"),
            {"ensemble": np.asarray(member_ids)},
        ),
    )
    monkeypatch.setattr(
        module, "cleanup_earth2_runtime_resources", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "cleanup_python_and_torch_runtime", lambda *_args, **_kwargs: None
    )

    prepared = module.prepare_ensemble_fanout_request(
        _raw_request(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=3,
            batch_size=1,
            max_in_flight=2,
            perturbation_materialization_mode="prepare_cpu",
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-ensemble-fanout",
            run_id="parent-prepare-cpu-run",
        ),
    )

    assert prepared["operation"] == "run"
    assert prepared["next_stage_id"] == "fanout"
    assert "resource_profile" not in prepared
    assert prepared["fanout_profile"] == {"item_count": 3, "max_in_flight": 2}
    prepared_state_dir = (
        tmp_path / "parent-prepare-cpu-run" / "prepared-initial-conditions"
    )
    assert [
        item["parameters"]["prepared_state_path"] for item in prepared["fanout_items"]
    ] == [
        str(prepared_state_dir / "batch-0000.pt"),
        str(prepared_state_dir / "batch-0001.pt"),
        str(prepared_state_dir / "batch-0002.pt"),
    ]
    assert selected_devices == ["cpu"]
    assert perturbation_calls == [[0], [1], [2]]


def test_earth2_ensemble_fanout_prepare_cpu_captures_materialization_output(
    tmp_path: Path, monkeypatch, capsys
):
    module = _load_module(
        "earth2_ensemble_fanout_support_prepare_cpu_capture_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def noisy_materialize(*_args, **_kwargs):
        print("library progress on stdout")
        return {
            "fanout_profile": {"item_count": 1, "max_in_flight": 1},
            "fanout_items": [{"item_index": 0, "parameters": {}}],
        }

    monkeypatch.setattr(module, "_materialize_prepared_batch_states", noisy_materialize)

    prepared = module.prepare_ensemble_fanout_request(
        _raw_request(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=1,
            perturbation_materialization_mode="prepare_cpu",
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-ensemble-fanout",
            run_id="parent-prepare-cpu-capture-run",
        ),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert prepared["operation"] == "run"
    assert prepared["fanout_profile"] == {"item_count": 1, "max_in_flight": 1}


def test_earth2_ensemble_fanout_prepare_rejects_dlwp(tmp_path: Path):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_prepare_dlwp",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    with pytest.raises(ValueError, match="supports only model='fcn'"):
        module.prepare_ensemble_fanout_request(
            _raw_request(
                model="dlwp",
                device_kind="gpu",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=2,
                batch_size=1,
                max_in_flight=2,
            ),
            _prepare_context(
                tmp_path,
                workflow_id="earth2-ensemble-fanout",
                run_id="parent-parallel-dlwp-run",
            ),
        )


def test_earth2_ensemble_fanout_prepare_rejects_parallel_dlwp(tmp_path: Path):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_prepare_parallel_dlwp",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    with pytest.raises(ValueError, match="supports only model='fcn'"):
        module.prepare_ensemble_fanout_request(
            _raw_request(
                model="dlwp",
                device_kind="gpu",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=2,
                batch_size=1,
                max_in_flight=2,
            ),
            _prepare_context(
                tmp_path,
                workflow_id="earth2-ensemble-fanout",
                run_id="parent-parallel-dlwp-run",
            ),
        )


def test_earth2_ensemble_fanout_prepare_accepts_fcn_model(tmp_path: Path, monkeypatch):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_prepare_fcn",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    prepared = module.prepare_ensemble_fanout_request(
        _raw_request(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=2,
            batch_size=1,
            max_in_flight=2,
        ),
        _prepare_context(
            tmp_path,
            workflow_id="earth2-ensemble-fanout",
            run_id="parent-fcn-run",
        ),
    )

    assert prepared["operation"] == "materialize_perturbations"
    assert prepared["parameters"]["model"] == "fcn"
    assert prepared["parameters"]["max_in_flight"] == 2
    assert prepared["resource_profile"]["gpus_required"] == 1


def test_earth2_ensemble_fanout_load_model_supports_fcn(monkeypatch):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_load_fcn",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    calls: list[object] = []

    class FakeDLWP:
        @staticmethod
        def load_default_package():
            raise AssertionError("fcn should not load the DLWP package")

    class FakeFCN:
        @staticmethod
        def load_default_package():
            calls.append("package:fcn")
            return "fcn-package"

        @staticmethod
        def load_model(package):
            calls.append(("model:fcn", package))
            return "fcn-model"

    fake_earth2studio = ModuleType("earth2studio")
    fake_models = ModuleType("earth2studio.models")
    fake_models_px = ModuleType("earth2studio.models.px")
    fake_models_px.DLWP = FakeDLWP
    fake_models_px.FCN = FakeFCN
    monkeypatch.setitem(sys.modules, "earth2studio", fake_earth2studio)
    monkeypatch.setitem(sys.modules, "earth2studio.models", fake_models)
    monkeypatch.setitem(sys.modules, "earth2studio.models.px", fake_models_px)

    package, model = module._load_model("fcn")

    assert package == "fcn-package"
    assert model == "fcn-model"
    assert calls == ["package:fcn", ("model:fcn", "fcn-package")]


def test_earth2_ensemble_fanout_captured_output_preserves_exception():
    module = _load_module(
        "earth2_ensemble_fanout_support_error_details_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    with pytest.raises(RuntimeError) as exc_info:
        module._raise_with_captured_output(
            "earth2-ensemble-fanout parent precompute",
            "cleanup warning",
            ValueError("real failure"),
        )

    message = str(exc_info.value)
    assert "earth2-ensemble-fanout parent precompute failed" in message
    assert "ValueError: real failure" in message
    assert "cleanup warning" in message


def test_earth2_ensemble_fanout_creates_python_child_backend_with_batch_member_chunks(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    module = _load_module(
        "earth2_ensemble_fanout_support_child_backend_chunks_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    captured: dict[str, object] = {}

    class FakeZarrBackend:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = FakeZarrBackend
    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)

    dataset_path = tmp_path / "child.zarr"
    module._create_child_zarr_backend(dataset_path, ensemble_chunk_size=32)

    assert captured["args"] == (str(dataset_path),)
    assert captured["kwargs"] == {
        "chunks": {"ensemble": 32, "time": 1, "lead_time": 1},
        "backend_kwargs": {"overwrite": True, "zarr_format": 3},
    }


def test_earth2_ensemble_fanout_child_backend_rejects_unknown_zarr_backend_env(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "pythno")
    module = _load_module(
        "earth2_ensemble_fanout_support_child_backend_bad_env_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    with pytest.raises(ValueError, match="PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND"):
        module._create_child_zarr_backend(tmp_path / "child.zarr")


def test_earth2_ensemble_fanout_rust_child_backend_env_uses_rust_zarr(
    tmp_path: Path, monkeypatch
):
    import collections
    import numpy as np

    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    module = _load_module(
        "earth2_ensemble_fanout_support_child_backend_rust_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    rust_backend_calls: list[dict[str, object]] = []

    class FakeRustZarrBackend:
        def __init__(self, **kwargs):
            self.record = {
                "kwargs": dict(kwargs),
                "add_array": [],
            }
            rust_backend_calls.append(self.record)

        def add_array(self, coords, array_name, data=None):
            self.record["add_array"].append(
                {"coords": dict(coords), "array_name": array_name, "data": data}
            )

    class UnexpectedPythonZarrBackend:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("fanout should use Rust Zarr when requested")

    class FakeModel:
        def input_coords(self):
            return collections.OrderedDict(
                [
                    ("variable", ["t2m", "u10m"]),
                    ("lead_time", 6),
                    ("lat", np.asarray([0.0])),
                ]
            )

        def output_coords(self, _coords):
            return collections.OrderedDict(
                [
                    ("variable", ["t2m", "u10m"]),
                    ("lead_time", 6),
                    ("lat", np.asarray([0.0])),
                ]
            )

    fake_rust_module = ModuleType("e2s_zarr_io")
    fake_rust_module.E2sZarrIoBackend = FakeRustZarrBackend
    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = UnexpectedPythonZarrBackend
    fake_time = ModuleType("earth2studio.utils.time")
    fake_time.to_time_array = lambda values: values
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", fake_rust_module)
    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_time)

    dataset_path = tmp_path / "child.zarr"
    io_backend = module._create_child_zarr_backend(dataset_path)
    module._configure_ensemble_io(
        io_backend,
        FakeModel(),
        start_time="2026-01-01T00:00:00Z",
        nsteps=1,
        member_ids=[2, 3],
    )

    assert len(rust_backend_calls) == 1
    rust_kwargs = rust_backend_calls[0]["kwargs"]
    assert rust_kwargs["file_name"] == str(dataset_path)
    assert "chunks" not in rust_kwargs
    assert "backend_kwargs" not in rust_kwargs
    parallel_coords = rust_kwargs["parallel_coords"]
    assert list(parallel_coords) == ["ensemble", "time", "lead_time"]
    np.testing.assert_array_equal(parallel_coords["ensemble"], np.asarray([2, 3]))
    assert parallel_coords["time"] == ["2026-01-01T00:00:00Z"]
    np.testing.assert_array_equal(parallel_coords["lead_time"], np.asarray([0, 6]))
    assert parallel_coords["lead_time"].dtype == np.dtype("int64")
    assert rust_kwargs["zarr_format"] == "v3"
    assert rust_kwargs["max_pool_bytes"] == 2 * 1024 * 1024 * 1024
    assert rust_kwargs["max_inflight_transient_bytes"] == 4 * 1024 * 1024 * 1024
    assert len(rust_backend_calls[0]["add_array"]) == 1
    add_array_call = rust_backend_calls[0]["add_array"][0]
    assert add_array_call["array_name"] == ["t2m", "u10m"]
    assert add_array_call["data"] is None
    coords = add_array_call["coords"]
    assert coords["time"] == ["2026-01-01T00:00:00Z"]
    np.testing.assert_array_equal(coords["ensemble"], np.asarray([2, 3]))
    np.testing.assert_array_equal(coords["lead_time"], np.asarray([0, 6]))
    np.testing.assert_array_equal(coords["lat"], np.asarray([0.0]))


def test_earth2_ensemble_fanout_run_batch_finalizes_child_backend(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_finalize_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    events: list[str] = []

    class FakeTensor:
        def to(self, _device):
            return self

    class FakeModel:
        def to(self, _device):
            return self

        def create_iterator(self, _x, _coords):
            return iter([(FakeTensor(), {})])

    class FakeIOBackend:
        def write(self, *_args):
            events.append("write")

        def finalize(self):
            events.append("finalize")

    fake_coords = ModuleType("earth2studio.utils.coords")
    fake_coords.map_coords = lambda x, coords, _target: (x, coords)
    fake_coords.split_coords = lambda x, coords: (x, coords)

    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_coords)
    monkeypatch.setattr(
        module,
        "_load_and_perturb_batch_initial_conditions",
        lambda *_args, **_kwargs: (
            object(),
            FakeModel(),
            None,
            FakeTensor(),
            {},
            "cpu",
            "spherical_gaussian",
        ),
    )
    monkeypatch.setattr(
        module, "_create_child_zarr_backend", lambda _path, **_kwargs: FakeIOBackend()
    )
    monkeypatch.setattr(
        module, "_configure_ensemble_io", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "fanout-batch-finalize"
    ctx = ExecutionContext(
        run_id="fanout-batch-finalize",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"device_kind": "cpu"},
    )

    module.run_ensemble_fanout_batch(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=1,
            batch_size=1,
            max_in_flight=1,
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
            batch_index=0,
            batch_member_ids=[0],
        ),
        ctx,
    )

    assert events == ["write", "finalize"]


def test_earth2_ensemble_fanout_run_batch_loads_prepared_state(
    tmp_path: Path, monkeypatch
):
    np = pytest.importorskip("numpy")
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_prepared_state_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    events: list[str] = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name

        def to(self, device):
            events.append(f"{self.name}.to:{device}")
            return self

    class FakeModel:
        def to(self, device):
            events.append(f"model.to:{device}")
            return self

        def create_iterator(self, _x, _coords):
            return iter([(FakeTensor("step"), {})])

    class FakeIOBackend:
        def write(self, *_args):
            events.append("write")

        def finalize(self):
            events.append("finalize")

    fake_coords = ModuleType("earth2studio.utils.coords")
    fake_coords.map_coords = lambda x, coords, _target: (x, coords)
    fake_coords.split_coords = lambda x, coords: (x, coords)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_coords)

    prepared_state_path = tmp_path / "batch-0000.pt"
    prepared_state_path.write_text("state", encoding="utf-8")
    fake_torch = ModuleType("torch")
    fake_torch.load = lambda path, **_kwargs: {
        "x": FakeTensor(f"loaded:{Path(path).name}"),
        "coords": {"ensemble": np.asarray([4, 5])},
        "member_ids": [4, 5],
        "batch_index": 2,
        "perturbation": "spherical_gaussian",
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(module, "_select_device", lambda _ctx: ("cuda", "gpu"))
    monkeypatch.setattr(module, "_load_model", lambda _model: (object(), FakeModel()))
    monkeypatch.setattr(
        module,
        "_load_and_perturb_batch_initial_conditions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared state path should skip perturbation replay")
        ),
    )
    monkeypatch.setattr(
        module, "_create_child_zarr_backend", lambda _path, **_kwargs: FakeIOBackend()
    )
    monkeypatch.setattr(
        module, "_configure_ensemble_io", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "cleanup_earth2_runtime_resources", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "cleanup_python_and_torch_runtime", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "fanout-batch-prepared"
    ctx = ExecutionContext(
        run_id="fanout-batch-prepared",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"gpus_required": 1},
    )

    result = module.run_ensemble_fanout_batch(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=6,
            batch_size=2,
            max_in_flight=2,
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
            batch_index=2,
            batch_member_ids=[4, 5],
            prepared_state_path=str(prepared_state_path),
        ),
        ctx,
    )

    assert result.batch_index == 2
    assert result.batch_member_ids == [4, 5]
    assert result.prepared_state_path == str(prepared_state_path)
    assert events == [
        "model.to:cuda",
        "loaded:batch-0000.pt.to:cuda",
        "write",
        "finalize",
    ]


def test_earth2_ensemble_fanout_prepared_state_failure_does_not_cleanup_cached_model(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_prepared_state_error_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    load_model_calls: list[str] = []
    cleanup_resources: list[object] = []
    cleanup_devices: list[object] = []

    fake_torch = ModuleType("torch")

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("corrupt prepared state")

    fake_torch.load = fail_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(module, "_select_device", lambda _ctx: ("cuda", "gpu"))
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda _model: load_model_calls.append("load") or (object(), object()),
    )
    monkeypatch.setattr(
        module,
        "cleanup_earth2_runtime_resources",
        lambda *resources: cleanup_resources.extend(resources),
    )
    monkeypatch.setattr(
        module,
        "cleanup_python_and_torch_runtime",
        lambda *, device=None: cleanup_devices.append(device),
    )

    cached_model = object()
    with pytest.raises(RuntimeError, match="corrupt prepared state"):
        module._load_prepared_batch_initial_conditions(
            module.EnsembleFanoutInput(
                model="fcn",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=2,
                prepared_state_path=str(tmp_path / "missing.pt"),
            ),
            {"resource_profile": {"gpus_required": 1}},
            str(tmp_path / "missing.pt"),
            model_resource=(object(), cached_model),
        )

    assert load_model_calls == []
    assert cached_model not in cleanup_resources
    assert cleanup_resources == [None]
    assert cleanup_devices == ["gpu"]


def test_earth2_ensemble_fanout_run_batch_finalizes_child_backend_on_error(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_finalize_error_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    events: list[str] = []

    class FakeTensor:
        def to(self, _device):
            return self

    class FakeModel:
        def to(self, _device):
            return self

        def create_iterator(self, _x, _coords):
            return iter([(FakeTensor(), {})])

    class FakeIOBackend:
        def write(self, *_args):
            events.append("write")
            raise RuntimeError("write failed")

        def finalize(self):
            events.append("finalize")

    fake_coords = ModuleType("earth2studio.utils.coords")
    fake_coords.map_coords = lambda x, coords, _target: (x, coords)
    fake_coords.split_coords = lambda x, coords: (x, coords)

    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_coords)
    monkeypatch.setattr(
        module,
        "_load_and_perturb_batch_initial_conditions",
        lambda *_args, **_kwargs: (
            object(),
            FakeModel(),
            None,
            FakeTensor(),
            {},
            "cpu",
            "spherical_gaussian",
        ),
    )

    def fake_create_child_zarr_backend(path, **_kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "partial").write_text("stale", encoding="utf-8")
        return FakeIOBackend()

    monkeypatch.setattr(
        module, "_create_child_zarr_backend", fake_create_child_zarr_backend
    )
    monkeypatch.setattr(
        module, "_configure_ensemble_io", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "fanout-batch-finalize-error"
    ctx = ExecutionContext(
        run_id="fanout-batch-finalize-error",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"device_kind": "cpu"},
    )

    with pytest.raises(RuntimeError, match="write failed"):
        module.run_ensemble_fanout_batch(
            module.EnsembleFanoutInput(
                model="fcn",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=1,
                batch_size=1,
                max_in_flight=1,
                perturbation="spherical_gaussian",
                noise_amplitude=0.05,
                seed_base=1000,
                batch_index=0,
                batch_member_ids=[0],
            ),
            ctx,
        )

    assert events == ["write", "finalize"]
    assert not (run_dir / "forecast-batch-0000.zarr").exists()


def test_earth2_ensemble_fanout_load_and_perturb_cleans_selected_device_on_error(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_prepare_error_cleanup_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    cleanup_calls: list[str] = []

    class FakeModel:
        def to(self, _device):
            return self

        def input_coords(self):
            return {"variable": ["t2m"], "lead_time": [0]}

    fake_data = ModuleType("earth2studio.data")
    fake_data.GFS = lambda: "gfs"

    def fail_fetch_data(**_kwargs):
        raise RuntimeError("fetch failed")

    fake_data.fetch_data = fail_fetch_data
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)

    fake_time = ModuleType("earth2studio.utils.time")
    fake_time.to_time_array = lambda value: value
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_time)

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(module, "_select_device", lambda _ctx: ("cuda", "gpu"))
    monkeypatch.setattr(module, "_load_model", lambda _model: (object(), FakeModel()))
    monkeypatch.setattr(
        module, "cleanup_earth2_runtime_resources", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module,
        "cleanup_python_and_torch_runtime",
        lambda *, device=None, **_kwargs: cleanup_calls.append(device),
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        module._load_and_perturb_batch_initial_conditions(
            module.EnsembleFanoutInput(
                model="fcn",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=1,
                batch_size=1,
                max_in_flight=1,
                perturbation="spherical_gaussian",
                noise_amplitude=0.05,
                seed_base=1000,
            ),
            ExecutionContext(
                run_id="fanout-error-cleanup",
                run_dir=tmp_path / "fanout-error-cleanup",
                outputs=OutputRegistry(tmp_path / "fanout-error-cleanup"),
                resource_profile={"gpus_required": 1},
            ),
            batch_index=0,
            member_ids=[0],
        )

    assert cleanup_calls == ["gpu"]


def test_earth2_ensemble_fanout_prepare_batch_initial_conditions_is_metadata_only(
    tmp_path: Path,
):
    module = _load_module(
        "earth2_ensemble_fanout_support_metadata_prepare_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    prepared_batches = module._prepare_batch_initial_conditions(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=3,
            batch_size=2,
            max_in_flight=2,
            perturbation="brown",
            noise_amplitude=0.05,
            seed_base=1000,
        ),
        tmp_path / "fanout-cleanup-run",
    )

    assert prepared_batches == [
        {
            "batch_index": 0,
            "batch_member_ids": [0, 1],
            "perturbation": "brown",
        },
        {
            "batch_index": 1,
            "batch_member_ids": [2],
            "perturbation": "brown",
        },
    ]
    assert not (
        tmp_path / "fanout-cleanup-run" / "prepared-initial-conditions"
    ).exists()


def test_earth2_ensemble_fanout_materializer_writes_prepared_states(
    tmp_path: Path, monkeypatch
):
    np = pytest.importorskip("numpy")
    module = _load_module(
        "earth2_ensemble_fanout_support_materializer_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    saved: list[tuple[Path, dict[str, object]]] = []
    perturbation_calls: list[object] = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name

        def cpu(self):
            return FakeTensor(f"{self.name}:cpu")

    class FakeModel:
        def to(self, _device):
            return self

        def input_coords(self):
            return {
                "variable": ["t2m"],
                "lead_time": [0],
            }

    class FakePerturbation:
        def __call__(self, batch_x, batch_coords):
            perturbation_calls.append(batch_coords["ensemble"].tolist())
            return batch_x, batch_coords

    fake_torch = ModuleType("torch")
    fake_torch.manual_seed = lambda seed: saved.append((Path("seed"), {"seed": seed}))

    def fake_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("prepared", encoding="utf-8")
        saved.append((path, payload))

    fake_torch.save = fake_save
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_data = ModuleType("earth2studio.data")
    fake_data.GFS = lambda: "gfs"
    fake_data.fetch_data = lambda **_kwargs: (FakeTensor("x0"), {"coords": "coords0"})
    monkeypatch.setitem(sys.modules, "earth2studio.data", fake_data)

    fake_time = ModuleType("earth2studio.utils.time")
    fake_time.to_time_array = lambda value: value
    monkeypatch.setitem(sys.modules, "earth2studio.utils.time", fake_time)

    monkeypatch.setattr(module, "_select_device", lambda _ctx: ("cuda", "gpu"))
    monkeypatch.setattr(module, "_load_model", lambda _model: (object(), FakeModel()))
    monkeypatch.setattr(
        module,
        "_build_perturbation",
        lambda _name, _noise: (FakePerturbation(), "spherical_gaussian"),
    )
    monkeypatch.setattr(
        module,
        "_build_batch_initial_conditions",
        lambda _x0, _coords0, _prognostic_ic, member_ids: (
            FakeTensor(f"batch-{member_ids[0]}"),
            {"ensemble": np.asarray(member_ids)},
        ),
    )
    monkeypatch.setattr(
        module, "cleanup_earth2_runtime_resources", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "cleanup_python_and_torch_runtime", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "materializer-run"
    ctx = ExecutionContext(
        run_id="materializer-run",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"gpus_required": 1},
    )

    result = module.materialize_ensemble_fanout_perturbations(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=3,
            batch_size=1,
            max_in_flight=2,
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
        ),
        ctx,
    )

    assert result["status"] == "succeeded"
    updates = result["_pipeline_updates"]
    assert updates["operation"] == "run"
    assert updates["fanout_profile"] == {"item_count": 3, "max_in_flight": 2}
    assert [
        item["parameters"]["prepared_state_path"] for item in updates["fanout_items"]
    ] == [
        str(run_dir / "prepared-initial-conditions" / "batch-0000.pt"),
        str(run_dir / "prepared-initial-conditions" / "batch-0001.pt"),
        str(run_dir / "prepared-initial-conditions" / "batch-0002.pt"),
    ]
    assert perturbation_calls == [[0], [1], [2]]
    assert [entry[0].name for entry in saved if entry[0].name.startswith("batch-")] == [
        "batch-0000.pt",
        "batch-0001.pt",
        "batch-0002.pt",
    ]


def test_earth2_ensemble_fanout_materializer_captures_output_on_error(
    tmp_path: Path, monkeypatch, capsys
):
    module = _load_module(
        "earth2_ensemble_fanout_support_materializer_capture_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def noisy_materialize(*_args, **_kwargs):
        print("earth2studio diagnostic output")
        raise RuntimeError("model load failed")

    monkeypatch.setattr(module, "_materialize_prepared_batch_states", noisy_materialize)

    with pytest.raises(RuntimeError) as exc_info:
        module.materialize_ensemble_fanout_perturbations(
            module.EnsembleFanoutInput(
                model="fcn",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                nensemble=1,
            ),
            ExecutionContext(
                run_id="materializer-capture-run",
                run_dir=tmp_path / "materializer-capture-run",
                outputs=OutputRegistry(tmp_path / "materializer-capture-run"),
                resource_profile={"gpus_required": 1},
            ),
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    message = str(exc_info.value)
    assert "earth2-ensemble-fanout GPU materialization failed" in message
    assert "RuntimeError: model load failed" in message
    assert "earth2studio diagnostic output" in message


def test_earth2_ensemble_fanout_run_batch_keeps_package_session_open(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    package_close_calls: list[tuple[str, str]] = []
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_package_cleanup_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class FakePackageFilesystem:
        def __init__(self) -> None:
            self.loop = "package-loop"
            self._session = "package-session"

        def close_session(self, loop, session) -> None:
            package_close_calls.append((loop, session))

    class FakePackage:
        def __init__(self) -> None:
            self.fs = FakePackageFilesystem()

    class FakeTensor:
        def to(self, _device):
            return self

    class FakeModel:
        def to(self, _device):
            return self

        def create_iterator(self, _x, _coords):
            return iter([(FakeTensor(), {})])

    class FakeIOBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def write(self, *_args):
            return None

    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = FakeIOBackend

    fake_coords = ModuleType("earth2studio.utils.coords")
    fake_coords.map_coords = lambda x, coords, _target: (x, coords)
    fake_coords.split_coords = lambda x, coords: (x, coords)

    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_coords)
    monkeypatch.setattr(
        module,
        "_load_and_perturb_batch_initial_conditions",
        lambda *_args, **_kwargs: (
            FakePackage(),
            FakeModel(),
            None,
            FakeTensor(),
            {},
            "cpu",
            "spherical_gaussian",
        ),
    )
    monkeypatch.setattr(
        module, "_configure_ensemble_io", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "fanout-batch-package-cleanup"
    ctx = ExecutionContext(
        run_id="fanout-batch-package-cleanup",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"gpus_required": 1},
    )

    result = module.run_ensemble_fanout_batch(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=1,
            batch_size=1,
            max_in_flight=1,
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
            batch_index=0,
            batch_member_ids=[0],
        ),
        ctx,
    )

    assert result.dataset_path == str(run_dir / "forecast-batch-0000.zarr")
    assert package_close_calls == []


def test_earth2_ensemble_fanout_run_batch_releases_torch_gpu_memory(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "python")
    module = _load_module(
        "earth2_ensemble_fanout_support_batch_gpu_cleanup_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class FakePackageFilesystem:
        def __init__(self) -> None:
            self.loop = "package-loop"
            self._session = "package-session"

        def close_session(self, loop, session) -> None:
            return None

    class FakePackage:
        def __init__(self) -> None:
            self.fs = FakePackageFilesystem()

    class FakeTensor:
        def to(self, _device):
            return self

        def cpu(self):
            return self

    class FakeModel:
        def to(self, _device):
            return self

        def create_iterator(self, _x, _coords):
            return iter([(FakeTensor(), {})])

    class FakeIOBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def write(self, *_args):
            return None

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    gc_collect_calls, cuda_ipc_collect_calls, cuda_empty_cache_calls = (
        _install_fake_torch_gpu_cleanup(monkeypatch)
    )

    fake_io = ModuleType("earth2studio.io")
    fake_io.ZarrBackend = FakeIOBackend

    fake_coords = ModuleType("earth2studio.utils.coords")
    fake_coords.map_coords = lambda x, coords, _target: (x, coords)
    fake_coords.split_coords = lambda x, coords: (x, coords)

    monkeypatch.setitem(sys.modules, "earth2studio.io", fake_io)
    monkeypatch.setitem(sys.modules, "earth2studio.utils.coords", fake_coords)
    monkeypatch.setattr(
        module,
        "_load_and_perturb_batch_initial_conditions",
        lambda *_args, **_kwargs: (
            FakePackage(),
            FakeModel(),
            None,
            FakeTensor(),
            {},
            "gpu",
            "spherical_gaussian",
        ),
    )
    monkeypatch.setattr(
        module, "_configure_ensemble_io", lambda *_args, **_kwargs: None
    )

    run_dir = tmp_path / "fanout-batch-gpu-cleanup"
    ctx = ExecutionContext(
        run_id="fanout-batch-gpu-cleanup",
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"gpus_required": 1},
    )

    result = module.run_ensemble_fanout_batch(
        module.EnsembleFanoutInput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=1,
            batch_size=1,
            max_in_flight=1,
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            seed_base=1000,
            batch_index=0,
            batch_member_ids=[0],
        ),
        ctx,
    )

    assert result.dataset_path == str(run_dir / "forecast-batch-0000.zarr")
    assert gc_collect_calls == ["collect"] * 3
    assert cuda_ipc_collect_calls == ["ipc_collect"]
    assert cuda_empty_cache_calls == ["empty_cache"]


def test_earth2_ensemble_fanout_workflow_exports_class_style_workflow():
    module = _load_module(
        "earth2_ensemble_fanout_workflow_wrapper_test",
        REPO_ROOT / "plugins" / "earth2-ensemble-fanout" / "workflow.py",
    )

    assert module.WORKFLOW is module.EnsembleFanoutWorkflow
    workflow = module.WORKFLOW()
    assert workflow.input_model is module.EnsembleFanoutInput
    assert workflow.output_model is module.EnsembleFanoutBatchOutput
    assert workflow.cache_scope == "process"
    assert workflow.model_cache_names == ["FCN"]
    assert callable(module.prepare_model_cache)


def test_earth2_ensemble_fanout_workflow_reuses_loaded_model(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_workflow_cache_test",
        REPO_ROOT / "plugins" / "earth2-ensemble-fanout" / "workflow.py",
    )
    load_model_calls: list[str] = []
    model_resources: list[tuple[object, object]] = []
    materializer_resources: list[tuple[object, object]] = []
    model_to_calls: list[str] = []
    model_eval_calls: list[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: object):
            model_to_calls.append(f"{self.name}.to:{device}")
            return fake_gpu_model

        def eval(self):
            model_eval_calls.append(f"{self.name}.eval")
            return self

    fake_package = object()
    fake_model = FakeModel("cpu-model")
    fake_gpu_model = FakeModel("gpu-model")

    def fake_load_model(model_name: str):
        load_model_calls.append(model_name)
        return fake_package, fake_model

    def fake_run_batch(_inputs, _ctx, *, model_resource=None):
        model_resources.append(model_resource)
        return module.EnsembleFanoutBatchOutput(
            model="fcn",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            nensemble=1,
            batch_index=0,
            batch_member_ids=[0],
            perturbation="spherical_gaussian",
            noise_amplitude=0.05,
            dataset_path=str(tmp_path / "forecast.zarr"),
            prepared_state_path="",
            note="ok",
        )

    def fake_materialize(_inputs, _ctx, *, model_resource=None):
        materializer_resources.append(model_resource)
        return {"status": "succeeded"}

    monkeypatch.setattr(module, "_load_model", fake_load_model)
    monkeypatch.setattr(module, "_select_device", lambda _ctx: ("cuda", "gpu"))
    monkeypatch.setattr(module, "run_ensemble_fanout_batch", fake_run_batch)
    monkeypatch.setattr(
        module, "materialize_ensemble_fanout_perturbations", fake_materialize
    )

    workflow = module.EnsembleFanoutWorkflow()
    assert module.prepare_model_cache({}) == {"model_names": ["FCN"]}
    assert workflow.warmup({}) == {"model_names": ["FCN"]}
    ctx = ExecutionContext(
        run_id="fanout-workflow-cache",
        run_dir=tmp_path / "fanout-workflow-cache",
        outputs=OutputRegistry(tmp_path / "fanout-workflow-cache"),
        resource_profile={"gpus_required": 1},
    )
    inputs = module.EnsembleFanoutInput(
        model="fcn",
        start_time="2026-01-01T00:00:00Z",
        nsteps=1,
        nensemble=1,
    )

    workflow.run(inputs, ctx)
    workflow.run(inputs, ctx)
    workflow.execute(
        {
            "operation": "materialize_perturbations",
            "parameters": dict(inputs.__dict__),
            "run_id": "fanout-workflow-cache",
            "run_dir": str(tmp_path / "fanout-workflow-cache"),
            "outputs": OutputRegistry(tmp_path / "fanout-workflow-cache"),
            "resource_profile": {"gpus_required": 1},
        }
    )

    assert load_model_calls == ["fcn"]
    assert model_to_calls == ["cpu-model.to:cuda"]
    assert model_eval_calls == ["gpu-model.eval"]
    assert model_resources == [
        (fake_package, fake_gpu_model),
        (fake_package, fake_gpu_model),
    ]
    assert materializer_resources == [(fake_package, fake_gpu_model)]


def test_earth2_ensemble_fanout_postprocess_tolerates_failed_children(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_postprocess",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class FakeDataset:
        dims = {"ensemble": 1}

        def sortby(self, _dim):
            return self

        def to_zarr(self, path, mode="w"):
            output = Path(path)
            output.mkdir(parents=True, exist_ok=True)
            (output / ".written").write_text(mode, encoding="utf-8")

        def close(self):
            return None

    fake_xarray = SimpleNamespace(
        open_zarr=lambda _path, consolidated=False: FakeDataset(),
        concat=lambda _datasets, dim=None: FakeDataset(),
    )
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)

    successful_dataset = tmp_path / "child-success.zarr"
    successful_dataset.mkdir(parents=True)

    ctx = _fanout_postprocess_context(tmp_path, run_id="parent-1")
    outcome = module.postprocess_ensemble_fanout_result(
        _fanout_prior_result(
            run_id="parent-1",
            child_results=[
                {"item_index": 0, "result": {"status": "failed", "error": "boom"}},
                {
                    "item_index": 1,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(successful_dataset),
                        "batch_member_ids": [1],
                    },
                },
            ],
            aggregation_summary={
                "item_count": 2,
                "collected_count": 2,
                "succeeded_count": 1,
                "failed_count": 1,
            },
        ),
        ctx,
    )

    expected_dataset = ctx.run_dir / "forecast-ensemble.zarr"
    assert outcome.status == "failed"
    assert outcome.payload.dataset_path == str(expected_dataset)
    assert ctx.outputs.primary_output().path == str(expected_dataset)


def test_earth2_ensemble_fanout_postprocess_marks_partial_success_as_failed(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_partial",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class FakeDataset:
        dims = {"ensemble": 1}

        def sortby(self, _dim):
            return self

        def to_zarr(self, path, mode="w"):
            output = Path(path)
            output.mkdir(parents=True, exist_ok=True)
            (output / ".written").write_text(mode, encoding="utf-8")

        def close(self):
            return None

    fake_xarray = SimpleNamespace(
        open_zarr=lambda _path, consolidated=False: FakeDataset(),
        concat=lambda _datasets, dim=None: FakeDataset(),
    )
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)

    successful_dataset = tmp_path / "child-success.zarr"
    successful_dataset.mkdir(parents=True)

    ctx = _fanout_postprocess_context(tmp_path, run_id="parent-partial")
    outcome = module.postprocess_ensemble_fanout_result(
        _fanout_prior_result(
            run_id="parent-partial",
            child_results=[
                {
                    "item_index": 0,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(successful_dataset),
                        "batch_member_ids": [0],
                    },
                },
                {
                    "item_index": 1,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(tmp_path / "missing-child.zarr"),
                        "batch_member_ids": [1],
                    },
                },
            ],
            aggregation_summary={
                "item_count": 2,
                "collected_count": 2,
                "succeeded_count": 2,
                "failed_count": 0,
            },
        ),
        ctx,
    )

    assert outcome.status == "failed"
    assert outcome.payload.aggregation_summary["failed_count"] == 0
    assert outcome.payload.postprocess_summary.skipped_count == 1
    assert outcome.payload.postprocess_summary.partial_aggregation is True


def test_earth2_ensemble_fanout_lightweight_merge_remaps_child_chunks(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_lightweight_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def write_child_store(
        path: Path, member_ids: list[int], value_offset: float
    ) -> None:
        data = np.stack(
            [
                np.full((1, 2, 2, 2), value_offset + member_id, dtype=np.float32)
                for member_id in member_ids
            ],
            axis=0,
        )
        ds = xr.Dataset(
            {
                "t2m": (
                    ("ensemble", "time", "lead_time", "lat", "lon"),
                    data,
                )
            },
            coords={
                "ensemble": np.asarray(member_ids, dtype=np.int64),
                "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([0, 6], dtype="timedelta64[h]"),
                "lat": np.asarray([0.0, 1.0], dtype=np.float32),
                "lon": np.asarray([0.0, 1.0], dtype=np.float32),
            },
        )
        ds.to_zarr(
            path,
            mode="w",
            encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
        )

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    write_child_store(child_a, [0, 1], 10.0)
    write_child_store(child_b, [2], 10.0)

    module._merge_zarr_child_stores(
        [
            (child_a, {"batch_member_ids": [0, 1]}),
            (child_b, {"batch_member_ids": [2]}),
        ],
        final,
        nensemble=3,
    )

    metadata = json.loads((final / "t2m" / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["shape"][0] == 3
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 1

    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged.sizes["ensemble"] == 3
        assert merged["ensemble"].values.tolist() == [0, 1, 2]
        assert merged["lat"].values.tolist() == [0.0, 1.0]
        assert merged["lon"].values.tolist() == [0.0, 1.0]
        np.testing.assert_array_equal(
            merged["lead_time"].values,
            np.asarray([0, 6], dtype="timedelta64[h]"),
        )
        assert merged["t2m"].isel(ensemble=0).values.mean() == pytest.approx(10.0)
        assert merged["t2m"].isel(ensemble=1).values.mean() == pytest.approx(11.0)
        assert merged["t2m"].isel(ensemble=2).values.mean() == pytest.approx(12.0)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_lightweight_merge_creates_v3_destination(
    tmp_path: Path, monkeypatch
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_v3_destination_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    data = np.full((1, 1, 1, 1, 1), 7.0, dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), data)},
        coords={
            "ensemble": np.asarray([0], dtype=np.int64),
            "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.asarray([0], dtype="timedelta64[h]"),
            "lat": np.asarray([0.0], dtype=np.float32),
            "lon": np.asarray([0.0], dtype=np.float32),
        },
    )
    child = tmp_path / "child.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    ds.to_zarr(
        child,
        mode="w",
        zarr_format=3,
        encoding={"t2m": {"chunks": (1, 1, 1, 1, 1)}},
    )

    destination_to_zarr_kwargs = []
    original_to_zarr = xr.Dataset.to_zarr

    def record_to_zarr(self, store=None, *args, **kwargs):
        if store is not None and Path(store) == final:
            destination_to_zarr_kwargs.append(dict(kwargs))
        return original_to_zarr(self, store, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_zarr", record_to_zarr)

    module._merge_zarr_child_stores(
        [(child, {"batch_member_ids": [0]})],
        final,
        nensemble=1,
    )

    assert [kwargs.get("zarr_format") for kwargs in destination_to_zarr_kwargs] == [
        3,
        3,
    ]
    assert (final / "zarr.json").exists()
    assert (final / "t2m" / "zarr.json").exists()


def test_earth2_ensemble_fanout_lightweight_merge_reports_copy_fallback(
    tmp_path: Path, monkeypatch
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_copy_fallback_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    data = np.full((1, 1, 1, 1, 1), 7.0, dtype=np.float32)
    ds = xr.Dataset(
        {
            "t2m": (
                ("ensemble", "time", "lead_time", "lat", "lon"),
                data,
            )
        },
        coords={
            "ensemble": np.asarray([0], dtype=np.int64),
            "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.asarray([0], dtype="timedelta64[h]"),
            "lat": np.asarray([0.0], dtype=np.float32),
            "lon": np.asarray([0.0], dtype=np.float32),
        },
    )
    child = tmp_path / "child.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    ds.to_zarr(
        child,
        mode="w",
        encoding={"t2m": {"chunks": (1, 1, 1, 1, 1)}},
    )

    real_link = module.os.link
    link_attempts: list[Path] = []

    def fake_link(source, destination):
        if "t2m" in Path(destination).parts:
            link_attempts.append(Path(destination))
            raise OSError(18, "Invalid cross-device link")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", fake_link)

    module._merge_zarr_child_stores(
        [(child, {"batch_member_ids": [0]})],
        final,
        nensemble=1,
    )

    assert link_attempts

    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged["t2m"].values.mean() == pytest.approx(7.0)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_lightweight_merge_rejects_v2_children(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_reject_v2_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    data = np.full((1, 1, 1, 1, 1), 7.0, dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), data)},
        coords={
            "ensemble": np.asarray([0], dtype=np.int64),
            "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.asarray([0], dtype="timedelta64[h]"),
            "lat": np.asarray([0.0], dtype=np.float32),
            "lon": np.asarray([0.0], dtype=np.float32),
        },
    )
    child = tmp_path / "child-v2.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    ds.to_zarr(
        child,
        mode="w",
        zarr_format=2,
        encoding={"t2m": {"chunks": (1, 1, 1, 1, 1)}},
    )

    assert (child / "t2m" / ".zarray").exists()
    assert not (child / "t2m" / "zarr.json").exists()
    with pytest.raises(ValueError, match="only supports Zarr v3 child stores"):
        module._merge_zarr_child_stores(
            [(child, {"batch_member_ids": [0]})],
            final,
            nensemble=1,
        )


def test_earth2_ensemble_fanout_block_chunk_merge_remaps_child_chunks(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_block_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def write_child_store(
        path: Path, member_ids: list[int], value_offset: float
    ) -> None:
        data = np.stack(
            [
                np.full((1, 2, 2, 2), value_offset + member_id, dtype=np.float32)
                for member_id in member_ids
            ],
            axis=0,
        )
        ds = xr.Dataset(
            {
                "t2m": (
                    ("ensemble", "time", "lead_time", "lat", "lon"),
                    data,
                )
            },
            coords={
                "ensemble": np.asarray(member_ids, dtype=np.int64),
                "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([0, 6], dtype="timedelta64[h]"),
                "lat": np.asarray([0.0, 1.0], dtype=np.float32),
                "lon": np.asarray([0.0, 1.0], dtype=np.float32),
            },
        )
        ds.to_zarr(
            path,
            mode="w",
            encoding={"t2m": {"chunks": (len(member_ids), 1, 1, 2, 2)}},
        )

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    write_child_store(child_a, [0, 1], 20.0)
    write_child_store(child_b, [2, 3], 20.0)

    module._merge_zarr_child_stores(
        [
            (child_a, {"batch_member_ids": [0, 1]}),
            (child_b, {"batch_member_ids": [2, 3]}),
        ],
        final,
        nensemble=4,
    )

    metadata = json.loads((final / "t2m" / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["shape"][0] == 4
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 2

    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged.sizes["ensemble"] == 4
        assert merged["ensemble"].values.tolist() == [0, 1, 2, 3]
        for member_id in range(4):
            assert merged["t2m"].isel(
                ensemble=member_id
            ).values.mean() == pytest.approx(20.0 + member_id)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_block_chunk_merge_handles_partial_final_batch(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_partial_block_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def write_child(path: Path, member_ids: list[int]) -> None:
        data = np.stack(
            [np.full((1, 1, 1, 1), 30.0 + m, dtype=np.float32) for m in member_ids],
            axis=0,
        )
        xr.Dataset(
            {"t2m": (("ensemble", "time", "lead_time", "lat", "lon"), data)},
            coords={
                "ensemble": np.asarray(member_ids, dtype=np.int64),
                "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([0], dtype="timedelta64[h]"),
                "lat": np.asarray([0.0], dtype=np.float32),
                "lon": np.asarray([0.0], dtype=np.float32),
            },
        ).to_zarr(
            path, mode="w", encoding={"t2m": {"chunks": (len(member_ids), 1, 1, 1, 1)}}
        )

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    write_child(child_a, [0, 1])
    write_child(child_b, [2])

    module._merge_zarr_child_stores(
        [(child_a, {"batch_member_ids": [0, 1]}), (child_b, {"batch_member_ids": [2]})],
        final,
        nensemble=3,
    )

    metadata = json.loads((final / "t2m" / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["shape"][0] == 3
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 2
    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged["ensemble"].values.tolist() == [0, 1, 2]
        for member_id in range(3):
            assert merged["t2m"].isel(
                ensemble=member_id
            ).values.mean() == pytest.approx(30.0 + member_id)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_copy_fallback_streams_members(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_copy_streaming_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    child_path = tmp_path / "child.zarr"
    dataset_path = tmp_path / "forecast-ensemble.zarr"
    reads: list[tuple[tuple[int | None, int | None, int | None] | object, ...]] = []
    writes: list[tuple[tuple[int | None, int | None, int | None] | object, ...]] = []

    def normalize_key(key):
        return tuple(
            (part.start, part.stop, part.step) if isinstance(part, slice) else part
            for part in key
        )

    class FakeChildArray:
        ndim = 2

        def __getitem__(self, key):
            normalized = normalize_key(key)
            reads.append(normalized)
            assert normalized != ((None, None, None), (None, None, None))
            return object()

    class FakeDestinationArray:
        ndim = 2

        def __setitem__(self, key, _value):
            writes.append(normalize_key(key))

    def open_group(path, mode):
        if path == str(child_path):
            assert mode == "r"
            return {"t2m": FakeChildArray()}
        assert path == str(dataset_path)
        assert mode == "a"
        return {"t2m": FakeDestinationArray()}

    fake_zarr = ModuleType("zarr")
    fake_zarr.open_group = open_group
    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)

    module._copy_array_region(
        child_path,
        dataset_path,
        "t2m",
        [10, 11],
        {10: 3, 11: 4},
    )

    assert reads == [
        ((0, 1, None), (None, None, None)),
        ((1, 2, None), (None, None, None)),
    ]
    assert writes == [
        ((3, 4, None), (None, None, None)),
        ((4, 5, None), (None, None, None)),
    ]


def test_earth2_ensemble_fanout_copy_fallback_uses_nonzero_ensemble_axis(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_copy_nonzero_axis_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def write_child(path: Path, member_ids: list[int]) -> None:
        data = np.zeros((1, 1, len(member_ids), 1, 1), dtype=np.float32)
        for local_index, member_id in enumerate(member_ids):
            data[:, :, local_index, :, :] = 50.0 + member_id
        xr.Dataset(
            {"t2m": (("time", "lead_time", "ensemble", "lat", "lon"), data)},
            coords={
                "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([0], dtype="timedelta64[h]"),
                "ensemble": np.asarray(member_ids, dtype=np.int64),
                "lat": np.asarray([0.0], dtype=np.float32),
                "lon": np.asarray([0.0], dtype=np.float32),
            },
        ).to_zarr(
            path,
            mode="w",
            encoding={"t2m": {"chunks": (1, 1, len(member_ids), 1, 1)}},
        )

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    write_child(child_a, [0, 1])
    write_child(child_b, [2])

    module._merge_zarr_child_stores(
        [(child_a, {"batch_member_ids": [0, 1]}), (child_b, {"batch_member_ids": [2]})],
        final,
        nensemble=3,
    )

    metadata = json.loads((final / "t2m" / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["dimension_names"] == [
        "time",
        "lead_time",
        "ensemble",
        "lat",
        "lon",
    ]
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"][2] == 2
    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged["ensemble"].values.tolist() == [0, 1, 2]
        for member_id in range(3):
            assert merged["t2m"].isel(
                ensemble=member_id
            ).values.mean() == pytest.approx(50.0 + member_id)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_lightweight_merge_preserves_partial_children(
    tmp_path: Path,
):
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    module = _load_module(
        "earth2_ensemble_fanout_support_partial_lightweight_merge_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    def write_child_store(path: Path, member_id: int) -> None:
        data = np.full((1, 1, 1, 1, 1), 10.0 + member_id, dtype=np.float32)
        ds = xr.Dataset(
            {
                "t2m": (
                    ("ensemble", "time", "lead_time", "lat", "lon"),
                    data,
                )
            },
            coords={
                "ensemble": np.asarray([member_id], dtype=np.int64),
                "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([0], dtype="timedelta64[h]"),
                "lat": np.asarray([0.0], dtype=np.float32),
                "lon": np.asarray([0.0], dtype=np.float32),
            },
        )
        ds.to_zarr(
            path,
            mode="w",
            encoding={"t2m": {"chunks": (1, 1, 1, 1, 1)}},
        )

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    final = tmp_path / "forecast-ensemble.zarr"
    write_child_store(child_a, 0)
    write_child_store(child_b, 2)

    module._merge_zarr_child_stores(
        [
            (child_a, {"batch_member_ids": [0]}),
            (child_b, {"batch_member_ids": [2]}),
        ],
        final,
        nensemble=3,
    )

    metadata = json.loads((final / "t2m" / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["shape"][0] == 2

    merged = xr.open_zarr(final, consolidated=False)
    try:
        assert merged.sizes["ensemble"] == 2
        assert merged["ensemble"].values.tolist() == [0, 2]
        assert merged["t2m"].isel(ensemble=0).values.mean() == pytest.approx(10.0)
        assert merged["t2m"].isel(ensemble=1).values.mean() == pytest.approx(12.0)
    finally:
        merged.close()


def test_earth2_ensemble_fanout_postprocess_uses_lightweight_merge(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_lightweight_postprocess_test",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )
    merge_calls: list[tuple[list[tuple[Path, dict]], Path, int]] = []

    def fake_merge(candidate_children, dataset_path, *, nensemble):
        merge_calls.append((candidate_children, dataset_path, nensemble))
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "zarr.json").write_text("{}", encoding="utf-8")

    class UnexpectedXarray:
        def open_zarr(self, *_args, **_kwargs):
            raise AssertionError("postprocess should not call xarray.open_zarr")

        def concat(self, *_args, **_kwargs):
            raise AssertionError("postprocess should not call xarray.concat")

    monkeypatch.setattr(module, "_merge_zarr_child_stores", fake_merge)
    monkeypatch.setitem(sys.modules, "xarray", UnexpectedXarray())

    child_a = tmp_path / "child-a.zarr"
    child_b = tmp_path / "child-b.zarr"
    child_a.mkdir()
    child_b.mkdir()
    ctx = _fanout_postprocess_context(tmp_path, run_id="parent-merge")
    ctx.request.raw_fields["nensemble"] = 3
    outcome = module.postprocess_ensemble_fanout_result(
        _fanout_prior_result(
            run_id="parent-merge",
            child_results=[
                {
                    "item_index": 0,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(child_a),
                        "batch_member_ids": [0, 1],
                    },
                },
                {
                    "item_index": 1,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(child_b),
                        "batch_member_ids": [2],
                    },
                },
            ],
            aggregation_summary={
                "item_count": 2,
                "collected_count": 2,
                "succeeded_count": 2,
                "failed_count": 0,
            },
        ),
        ctx,
    )

    expected_dataset = ctx.run_dir / "forecast-ensemble.zarr"
    assert outcome.status == "succeeded"
    assert outcome.payload.dataset_path == str(expected_dataset)
    assert merge_calls == [
        (
            [
                (
                    child_a,
                    {
                        "status": "succeeded",
                        "dataset_path": str(child_a),
                        "batch_member_ids": [0, 1],
                    },
                ),
                (
                    child_b,
                    {
                        "status": "succeeded",
                        "dataset_path": str(child_b),
                        "batch_member_ids": [2],
                    },
                ),
            ],
            expected_dataset,
            3,
        )
    ]
    assert ctx.outputs.primary_output().path == str(expected_dataset)


def test_earth2_ensemble_fanout_postprocess_copies_single_child_store(
    tmp_path: Path, monkeypatch
):
    module = _load_module(
        "earth2_ensemble_fanout_support_test_single_child",
        REPO_ROOT
        / "plugins"
        / "earth2-ensemble-fanout"
        / "earth2_ensemble_fanout_support.py",
    )

    class UnexpectedXarray:
        def open_zarr(self, *_args, **_kwargs):
            raise AssertionError(
                "single-child postprocess should not call xarray.open_zarr"
            )

        def concat(self, *_args, **_kwargs):
            raise AssertionError(
                "single-child postprocess should not call xarray.concat"
            )

    monkeypatch.setitem(sys.modules, "xarray", UnexpectedXarray())

    successful_dataset = tmp_path / "child-success.zarr"
    successful_dataset.mkdir(parents=True)
    (successful_dataset / "zarr.json").write_text("{}", encoding="utf-8")
    (successful_dataset / "payload.bin").write_text("ok", encoding="utf-8")

    ctx = _fanout_postprocess_context(tmp_path, run_id="parent-single")
    outcome = module.postprocess_ensemble_fanout_result(
        _fanout_prior_result(
            run_id="parent-single",
            child_results=[
                {
                    "item_index": 0,
                    "result": {
                        "status": "succeeded",
                        "dataset_path": str(successful_dataset),
                        "batch_member_ids": [0, 1],
                    },
                }
            ],
            aggregation_summary={
                "item_count": 1,
                "collected_count": 1,
                "succeeded_count": 1,
                "failed_count": 0,
            },
        ),
        ctx,
    )

    expected_dataset = ctx.run_dir / "forecast-ensemble.zarr"
    assert outcome.status == "succeeded"
    assert outcome.payload.dataset_path == str(expected_dataset)
    assert (expected_dataset / "zarr.json").read_text(encoding="utf-8") == "{}"
    assert (expected_dataset / "payload.bin").read_text(encoding="utf-8") == "ok"
    assert ctx.outputs.primary_output().path == str(expected_dataset)
