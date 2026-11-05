# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework integration contracts; real model parity is a separate H100 gate."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from plugin_sdk import PrepareContext, RawRequest, workflow_request_schema
from plugin_runtime import expand_plugin_manifest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "earth2-aifs2ens"
sys.path.insert(0, str(PLUGIN))


@pytest.fixture
def workflow_module():
    spec = importlib.util.spec_from_file_location(
        "aifs2ens_plugin_workflow", PLUGIN / "workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def request_fields():
    return {
        "start_time": "2026-11-03T00:00:00Z",
        "nsteps": 4,
        "nensemble": 2,
        "ifs_member_id": 7,
        "seed_base": 0,
    }


def test_manifest_runs_one_gpu_task_for_the_complete_ensemble():
    manifest = expand_plugin_manifest(
        yaml.safe_load((PLUGIN / "plugin.yaml").read_text())
    )
    stages = manifest["pipeline"]["stages"]
    assert [s["phase"] for s in stages] == [
        "prepare",
        "prefetch",
        "schedule",
        "execute",
        "results",
    ]
    assert stages[0]["next"] == "prefetch"
    assert stages[3]["queue"] == "execute.earth2-gpu"
    assert stages[3]["next"] == "results"
    for key in (
        "executor_class",
        "prepare_executor_class",
        "postprocess_executor_class",
        "readiness_executor_class",
    ):
        assert manifest["runtime"][key] == "earth2-gpu"


def test_manifest_declares_the_result_contract_required_by_the_rest_server():
    from jsonschema import Draft202012Validator

    manifest = yaml.safe_load((PLUGIN / "plugin.yaml").read_text())
    schema = manifest["outputs"]["result_schema_inline"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(
        {
            "dataset_path": "/outputs/forecast.zarr",
            "complete": True,
            "nensemble": 2,
            "nsteps": 4,
            "ifs_member_id": 1,
            "seed_base": 0,
            "batch_size": 2,
            "rng_policy": "torch-seed-once-native-v1",
            "numerical_policy": "earth2studio-native-v1",
        }
    )
    validator.validate({"complete": False, "error": "incomplete ensemble"})
    assert schema["properties"]["dataset_path"]["type"] == "string"
    assert schema["properties"]["complete"]["type"] == "boolean"


def test_prepare_downloads_once_for_one_gpu_request_without_loading_a_model(
    tmp_path, workflow_module, monkeypatch
):
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
    monkeypatch.setattr(workflow, "_load_model", lambda *a: pytest.fail("loaded model"))
    staged = []

    def stage(inputs, ctx):
        staged.append((inputs.seed_base, ctx.run_dir))
        return {
            "manifest_path": str(ctx.run_dir / "manifest.json"),
            "manifest_sha256": "b" * 64,
        }

    monkeypatch.setattr(workflow, "_stage_inputs", stage)
    raw = RawRequest(
        content_type="application/json", operation="run", raw_fields=request_fields()
    )
    prepared = workflow.prepare(
        raw,
        PrepareContext(
            run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
        ),
    )
    assert staged == [(0, tmp_path)]
    assert prepared["operation"] == "forecast_ensemble"
    assert prepared["resource_profile"]["executor_class"] == "earth2-gpu"
    assert prepared["resource_profile"]["device_kind"] == "gpu"
    assert prepared["resource_profile"]["gpus_required"] == 1
    assert prepared["resource_profile"]["memory_mb"] == 65536
    assert set(prepared["resource_profile"]["tags"]) == {"earth2", "aifs2ens", "gpu"}
    assert prepared["batch_profile"]["enabled"] is False
    assert set(prepared["parameters"]) == {"manifest_path", "manifest_sha256"}
    assert "fanout_items" not in prepared and "fanout_profile" not in prepared
    schema = workflow_request_schema(workflow)
    assert schema["additionalProperties"] is False
    assert "manifest_path" not in schema["properties"]


def test_prepare_keeps_downloader_output_off_the_hook_json_channel(
    tmp_path, workflow_module, monkeypatch, capsys
):
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")

    def stage(*args):
        print("downloading IFS")
        return {"manifest_path": "/shared/manifest.json", "manifest_sha256": "b" * 64}

    monkeypatch.setattr(workflow, "_stage_inputs", stage)
    workflow.prepare(
        RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields=request_fields(),
        ),
        PrepareContext(
            run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
        ),
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "downloading IFS" in output.err


@pytest.mark.parametrize("changes", [{"nensemble": 3}, {"nensemble": 16}])
def test_default_batch_limit_rejects_before_downloading(
    tmp_path, workflow_module, monkeypatch, changes
):
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
    monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(
        workflow, "_stage_inputs", lambda *a: pytest.fail("downloaded oversized batch")
    )
    with pytest.raises(ValueError, match="batch_size.*qualified maximum 2"):
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields={**request_fields(), **changes},
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )


@pytest.mark.parametrize(
    "nensemble,batch_size,limit",
    [(3, 2, None), (16, 1, None), (16, 2, None), (1, 16, None), (3, 3, "3")],
)
def test_batch_limit_preserves_the_effective_upstream_layout(
    tmp_path, workflow_module, monkeypatch, nensemble, batch_size, limit
):
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
    monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
    if limit is not None:
        monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
    workflow = workflow_module.WORKFLOW()
    staged = []
    monkeypatch.setattr(
        workflow,
        "_stage_inputs",
        lambda inputs, ctx: (
            staged.append((inputs.nensemble, inputs.batch_size))
            or {"manifest_path": "/shared/manifest.json", "manifest_sha256": "b" * 64}
        ),
    )
    workflow.prepare(
        RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields={
                **request_fields(),
                "nensemble": nensemble,
                "batch_size": batch_size,
            },
        ),
        PrepareContext(
            run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
        ),
    )
    assert staged == [(nensemble, min(nensemble, batch_size))]


@pytest.mark.parametrize("limit,batch_size", [("1", 2), ("3", 4)])
def test_configured_batch_limit_rejects_before_downloading(
    tmp_path, workflow_module, monkeypatch, limit, batch_size
):
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
    monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(
        workflow, "_stage_inputs", lambda *a: pytest.fail("downloaded oversized batch")
    )
    with pytest.raises(ValueError, match=f"qualified maximum {limit}"):
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields={
                    **request_fields(),
                    "nensemble": 4,
                    "batch_size": batch_size,
                },
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )


@pytest.mark.parametrize("limit", ["", "0", "-1", "invalid", "1.5"])
def test_invalid_batch_limit_blocks_downloads(
    tmp_path, workflow_module, monkeypatch, limit
):
    monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
    monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(
        workflow,
        "_stage_inputs",
        lambda *a: pytest.fail("downloaded with invalid limit"),
    )
    with pytest.raises(RuntimeError, match="AIFS2ENS_MAX_BATCH_SIZE"):
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields=request_fields(),
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )


def test_staged_request_cannot_bypass_the_execution_batch_limit(
    tmp_path, workflow_module, monkeypatch
):
    from aifs2ens_contract import ForecastRequest

    monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
    loaded = SimpleNamespace(
        request=ForecastRequest.model_validate({**request_fields(), "nensemble": 3})
    )
    monkeypatch.setattr(workflow_module, "load_ensemble_input", lambda *a: loaded)
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(
        workflow, "_run_ensemble", lambda *a: pytest.fail("executed oversized batch")
    )
    with pytest.raises(ValueError, match="qualified maximum 2"):
        workflow.execute(
            {
                "operation": "forecast_ensemble",
                "parameters": {
                    "manifest_path": "/shared/manifest.json",
                    "manifest_sha256": "b" * 64,
                },
                "run_id": "oversized",
                "run_dir": str(tmp_path),
            }
        )


def test_cpu_warmup_does_not_load_weights(workflow_module, monkeypatch):
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(workflow, "_load_model", lambda *a: pytest.fail("loaded model"))
    assert workflow.warmup({"device": "cpu"}) == {"model_names": []}
    assert workflow._model is None


def test_gpu_warmup_reuses_the_cached_model(workflow_module, monkeypatch):
    workflow = workflow_module.WORKFLOW()
    calls = []
    model = SimpleNamespace(seed=None)

    def load(device):
        calls.append(device)
        return SimpleNamespace(field_hashes={"verified": "bytes"}), model

    monkeypatch.setattr(workflow, "_load_model", load)
    workflow.warmup({"device": "cuda"})
    workflow.warmup({"device": "cuda"})
    assert calls == ["cuda"]
    assert workflow._model is model


def test_unqualified_gpu_memory_profile_blocks_dispatch(
    tmp_path, workflow_module, monkeypatch
):
    workflow = workflow_module.WORKFLOW()
    monkeypatch.delenv("AIFS2ENS_GPU_MEMORY_MB", raising=False)
    monkeypatch.setattr(
        workflow,
        "_stage_inputs",
        lambda *a: pytest.fail("downloaded before deployment validation"),
    )
    with pytest.raises(RuntimeError, match="AIFS2ENS_GPU_MEMORY_MB"):
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields=request_fields(),
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )


def test_whole_ensemble_execution_passes_one_frozen_input_to_the_runner(
    tmp_path, workflow_module, monkeypatch
):
    workflow = workflow_module.WORKFLOW()
    calls = []
    loaded = SimpleNamespace(request=SimpleNamespace(nensemble=2, batch_size=2))
    monkeypatch.setattr(workflow_module, "load_ensemble_input", lambda *a: loaded)
    monkeypatch.setattr(
        workflow,
        "_run_ensemble",
        lambda item, ctx: (
            calls.append(item) or {"dataset_path": str(tmp_path / "ensemble.zarr")}
        ),
    )
    result = workflow.execute(
        {
            "operation": "forecast_ensemble",
            "parameters": {
                "manifest_path": "/shared/manifest.json",
                "manifest_sha256": "b" * 64,
            },
            "run_id": "child",
            "run_dir": str(tmp_path),
        }
    )
    assert calls == [loaded]
    assert result["dataset_path"].endswith("ensemble.zarr")


def test_example_is_a_valid_public_request(workflow_module):
    from aifs2ens_contract import ForecastRequest

    ForecastRequest.model_validate_json(
        (PLUGIN / "examples/default_request.json").read_text()
    )


def test_readiness_checks_require_qualified_deployment_assets(monkeypatch):
    import plugin_dev

    names = {
        "AIFS2ENS_INPUT_CONTRACT",
        "AIFS2ENS_INVARIANTS",
        "AIFS2ENS_CHECKPOINT_DIR",
        "AIFS2ENS_GPU_MEMORY_MB",
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        plugin_dev,
        "_probe_python_module_contract",
        lambda *a, **k: {"ok": True, "detail": "test module available"},
    )
    result = plugin_dev.check_env_plugin(PLUGIN)
    checks = {
        check["name"]: check for check in result["checks"] if check["name"] in names
    }
    assert set(checks) == names
    assert all(not check["ok"] for check in checks.values())


def test_local_service_plan_launches_only_gpu_workers_and_allows_prepare_downloads(
    tmp_path,
):
    import json
    import plugin_dev
    from runtime_env_launcher import build_worker_launch_specs

    workspace = tmp_path / "service"
    plugin_dev.build_run_local_plan(PLUGIN, workspace=workspace, port=0, redis_port=0)
    config = json.loads((workspace / "runtime_config.json").read_text())
    envs = config["python_runtime_envs"]
    assert set(envs) == {"earth2-gpu"}
    assert envs["earth2-gpu"]["launch"]["enabled"] is True
    assert envs["earth2-gpu"]["launch"]["device_kind"] == "gpu"
    assert set(envs["earth2-gpu"]["launch"]["tags"]) == {"earth2", "aifs2ens", "gpu"}
    assert config["roles"]["prepare"]["config"]["hook_timeout_secs"] == 7200
    specs = build_worker_launch_specs(
        envs,
        namespace="local",
        pod_name="aifs2ens",
        enabled_executor_classes={"earth2-gpu"},
        gpu_inventory=[
            {
                "device_index": 0,
                "device_name": "NVIDIA H100",
                "device_uuid": "test-gpu",
                "memory_mb": 81920,
            }
        ],
        worker_script="scripts/inference_worker.py",
    )
    assert len(specs) == 1
    worker_env = specs[0]["env"]
    assert worker_env["GPU_EXECUTOR_CLASS"] == "earth2-gpu"
    assert worker_env["GPU_DEVICE_KIND"] == "gpu"
    assert set(worker_env["GPU_TAGS"].split(",")) == {"earth2", "aifs2ens", "gpu"}
    assert worker_env["GPU_STREAM_NAME"] == "execute.earth2-gpu:gpu:local:aifs2ens:0"


@pytest.mark.parametrize("failure", [None, "prediction", "validation"])
def test_final_dataset_is_registered_only_after_full_ensemble_validation(
    tmp_path, workflow_module, monkeypatch, failure
):
    from aifs2ens_contract import ForecastRequest
    from plugin_sdk import build_execution_context

    request = ForecastRequest.model_validate(request_fields())
    contract = SimpleNamespace(checkpoint_sha256="a" * 64)
    loaded = SimpleNamespace(
        request=request,
        contract=contract,
        provenance={"mirror": "ecmwf"},
        manifest_sha256="b" * 64,
        input_sha256="c" * 64,
    )
    workflow = workflow_module.WORKFLOW()
    monkeypatch.setattr(workflow, "_contract", lambda: contract)
    monkeypatch.setattr(workflow, "_ensure_model", lambda device: None)
    monkeypatch.setattr(workflow, "_runtime_identity", lambda: {"torch": "worker"})
    written = []

    def sink(path, item):
        assert item is loaded
        path.mkdir()
        written.append(path)
        return SimpleNamespace(field_hashes={"verified": "bytes"})

    def rollout(model, item, writer, **kwargs):
        assert item is loaded
        if failure == "prediction":
            raise RuntimeError("prediction failed")

    def validate(path, item, *, expected_hashes):
        assert expected_hashes == {"verified": "bytes"}
        assert path == written[0] and item is loaded
        if failure == "validation":
            raise ValueError("incomplete ensemble")

    for name, value in (
        ("aifs2ens_storage", SimpleNamespace(ZarrEnsembleSink=sink)),
        ("aifs2ens_rollout", SimpleNamespace(run_ensemble_rollout=rollout)),
        ("aifs2ens_outputs", SimpleNamespace(validate_ensemble_output=validate)),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    ctx = build_execution_context({"run_id": "run", "run_dir": str(tmp_path)})
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            workflow._run_ensemble(loaded, ctx)
        assert ctx.outputs.registered_outputs() == []
        assert not written[0].exists()
    else:
        result = workflow._run_ensemble(loaded, ctx)
        assert result["complete"] is True
        assert result["nensemble"] == 2 and result["seed_base"] == 0
        assert result["batch_size"] == 2
        assert result["input_sha256"] == "c" * 64
        assert result["runtime"]["torch"] == "worker"
        assert result["rng_policy"] == "torch-seed-once-native-v1"
        assert ctx.outputs.primary_output().path == str(written[0])
        assert (tmp_path / "forecast_metadata.json").is_file()
