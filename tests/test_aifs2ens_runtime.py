# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
from aifs2ens_artifacts import InputContract, sha256_file  # noqa: E402
import aifs2ens_runtime as runtime  # noqa: E402


@pytest.fixture
def contract():
    return InputContract.model_validate(
        {
            "schema_version": 1,
            "checkpoint_sha256": "a" * 64,
            "input_variables": ["t2m", "lsm", "sdor", "slor", "z", "wmb"],
            "output_variables": ["t2m"],
            "lat": [90.0, -90.0],
            "lon": [0.0, 180.0],
        }
    )


@pytest.mark.parametrize("bad", [None, "variables", "grid", "lead"])
def test_loaded_model_must_match_the_qualified_coordinates(contract, bad):
    coords = {
        "variable": np.array(contract.input_variables),
        "lat": np.array(contract.lat),
        "lon": np.array(contract.lon),
        "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
    }
    outputs = {
        **coords,
        "variable": np.array(contract.output_variables),
        "lead_time": np.array([6], dtype="timedelta64[h]"),
    }
    if bad == "variables":
        coords["variable"] = coords["variable"][::-1]
    if bad == "grid":
        coords["lat"] = np.array([80.0, -80.0])
    if bad == "lead":
        outputs["lead_time"] = np.array([12], dtype="timedelta64[h]")
    model = SimpleNamespace(
        input_coords=lambda: coords, output_coords=lambda _: outputs
    )
    if bad:
        with pytest.raises(ValueError, match="contract"):
            runtime.verify_model_contract(model, contract)
    else:
        runtime.verify_model_contract(model, contract)


def test_changed_checkpoint_is_rejected_before_unpickling(
    tmp_path, monkeypatch, contract
):
    checkpoint = tmp_path / "aifs-ens-crps-2.0.ckpt"
    checkpoint.write_bytes(b"original pinned test bytes")
    expected = sha256_file(checkpoint)
    contract = contract.model_copy(update={"checkpoint_sha256": expected})
    monkeypatch.setenv("AIFS2ENS_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "CHECKPOINT_SHA256", expected)
    assert runtime.verified_checkpoint(contract) == checkpoint
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint.*digest"):
        runtime.verified_checkpoint(contract)


def test_a_different_checkpoint_cannot_be_selected_through_a_new_contract(
    tmp_path, monkeypatch, contract
):
    monkeypatch.setenv("AIFS2ENS_CHECKPOINT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="pinned checkpoint"):
        runtime.verified_checkpoint(contract)


@pytest.mark.parametrize("invalid_assets", [False, True])
def test_worker_warmup_verifies_assets_once_before_model_construction(
    tmp_path, monkeypatch, contract, invalid_assets
):
    import inference_worker

    plugin = Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"
    spec = importlib.util.spec_from_file_location(
        "aifs2ens_verified_warmup_workflow", plugin / "workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "cleanup_earth2_runtime_resources", lambda _: None)
    monkeypatch.setattr(module, "cleanup_python_and_torch_runtime", lambda: None)
    contract_path = tmp_path / "input-contract.json"
    contract_path.write_text(contract.model_dump_json())
    settings = {
        "PLUGIN_DIR": str(plugin.parent),
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": "earth2-aifs2ens",
        "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR": str(tmp_path / "locks"),
        "GPU_EXECUTOR_CLASS": "earth2-gpu",
        "GPU_DEVICE_KIND": "gpu",
        "AIFS2ENS_INPUT_CONTRACT": str(contract_path),
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    events = []

    def verify_assets(contract=None):
        events.append("verify")
        if invalid_assets:
            raise ValueError("asset digest mismatch")
        return tmp_path / "aifs-ens-crps-2.0.ckpt"

    monkeypatch.setattr(runtime, "prepare_assets", verify_assets)
    coords = {
        "variable": np.array(contract.input_variables),
        "lat": np.array(contract.lat),
        "lon": np.array(contract.lon),
        "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
    }
    model = SimpleNamespace(
        seed=None,
        input_coords=lambda: coords,
        output_coords=lambda _: {
            **coords,
            "variable": np.array(contract.output_variables),
            "lead_time": np.array([6], dtype="timedelta64[h]"),
        },
    )
    model.to = lambda _: model
    model.eval = lambda: model

    def construct(package, **kwargs):
        assert events[-1] == "verify"
        assert kwargs == {"preload_invariants": False}
        events.append("construct")
        return model

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.auto",
        SimpleNamespace(Package=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.px",
        SimpleNamespace(AIFS2ENS=SimpleNamespace(load_model=construct)),
    )
    executor = inference_worker.WorkflowExecutor(SimpleNamespace())
    monkeypatch.setattr(executor, "_load_plugin_module", lambda *args: module)
    try:
        if invalid_assets:
            with pytest.raises(ValueError, match="asset digest mismatch"):
                executor.warm_enabled_workflow()
            assert events == ["verify"]
        else:
            assert executor.warm_enabled_workflow()["status"] == "warmed"
            assert executor.warm_enabled_workflow()["status"] == "warmed"
            assert events == ["verify", "construct"]
            assert executor._workflow_cache["earth2-aifs2ens"]._model is model
    finally:
        executor.close()


def test_execution_runtime_records_the_actual_worker_math_settings(monkeypatch):
    torch = SimpleNamespace(
        __version__="worker-torch",
        version=SimpleNamespace(cuda="worker-cuda"),
        cuda=SimpleNamespace(get_device_name=lambda device: "worker H100"),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_tf32=True,
                    allow_bf16_reduced_precision_reduction=False,
                )
            ),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        runtime,
        "version",
        lambda name: {"flash-attn": "worker-flash", "anemoi-models": "0.11.2"}[name],
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    identity = runtime.runtime_identity(SimpleNamespace(seed=None), "cuda:0")
    assert identity["torch"] == "worker-torch"
    assert identity["flash_attn"] == "worker-flash"
    assert identity["allow_tf32_matmul"] is True
    assert identity["allow_tf32_cudnn"] is False
    assert identity["allow_bf16_reduced_precision_reduction"] is False
    assert identity["deterministic_algorithms"] is True
    assert identity["deterministic_warn_only"] is False
    assert identity["cublas_workspace_config"] == ":4096:8"
    assert identity["gpu_name"] == "worker H100" and identity["model_seed"] is None
