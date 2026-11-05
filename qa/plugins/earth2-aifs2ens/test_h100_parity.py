# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-ensemble parity on one configured H100, with frozen stock reference limits."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AIFS2ENS_QUALIFICATION_ROOT"),
    reason="requires the configured H100 and frozen AIFS2ENS qualification assets",
)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_parity import POLICY, RNG_POLICY, compare_field  # noqa: E402
from run_qualification import (  # noqa: E402
    LAYOUTS,
    verify_candidate_runtime,
    wait_for_parent_release,
    wait_for_structured_result,
)


@pytest.fixture
def assets(monkeypatch):
    if platform.system() != "Linux":
        raise RuntimeError("real-model tests run on the configured remote H100")
    root = Path(os.environ["AIFS2ENS_QUALIFICATION_ROOT"]).resolve()
    assert root.is_relative_to(Path("/mnt/my_lfs_data/Projects"))
    for path in (REPO / "scripts", REPO / "python", REPO / "plugins/earth2-aifs2ens"):
        monkeypatch.syspath_prepend(str(path))
    import torch

    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
    settings = {
        "AIFS2ENS_INPUT_CONTRACT": str(root / "qualified-input-contract.json"),
        "AIFS2ENS_INVARIANTS": str(root / "assets/invariants.npy"),
        "AIFS2ENS_CHECKPOINT_DIR": str(root / "assets/checkpoint-xet"),
        "AIFS2ENS_GPU_MEMORY_MB": "65536",
        "EARTH2STUDIO_MODEL_CACHE": str(root / "cache/models"),
        "EARTH2STUDIO_DATA_CACHE": str(root / "cache/weather"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "CPATH": "/usr/local/cuda/include",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    calibration_path = root / "qualification/calibration.json"
    sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    assert sha == os.environ["AIFS2ENS_PARITY_CALIBRATION_SHA256"]
    calibration = json.loads(calibration_path.read_text())
    assert (
        calibration["status"] == "passed" and calibration["numerical_policy"] == POLICY
    )
    assert calibration["rng_policy"] == RNG_POLICY
    assert all(
        row["holdout"]["status"] == "passed" for row in calibration["layouts"].values()
    )
    contract = json.loads((root / "qualified-input-contract.json").read_text())
    captured = json.loads((root / "weather-capture-report.json").read_text())
    assert captured["status"] == "passed" and captured["shared_input"] is True
    assert (
        captured["preprocessing_policy"] == "earth2studio-latlon-frozen-invariants-v1"
    )
    return {
        "root": root,
        "contract": contract,
        "captured": captured,
        "references": json.loads((root / "qualification/references.json").read_text()),
        "calibration": calibration,
        "calibration_sha256": sha,
    }


@pytest.fixture
def workflow(assets):
    spec = importlib.util.spec_from_file_location(
        "qualified_single_gpu_aifs2ens", REPO / "plugins/earth2-aifs2ens/workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    instance = module.WORKFLOW()
    try:
        yield instance
    finally:
        instance.cleanup()


def _request(assets, layout):
    return {
        **LAYOUTS[layout],
        "start_time": assets["captured"]["initialization"],
        "ifs_member_id": assets["captured"]["ifs_member_id"],
        "output_variables": assets["contract"]["output_variables"],
    }


def _math_flags(torch):
    return (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _compare(path, request, assets, layout, metrics_path, *, candidate_runtime):
    import numpy as np
    import xarray as xr

    reference = Path(assets["references"][layout]["path"])
    report = json.loads((reference / "report.json").read_text())
    assert report["status"] == "passed" and report["numerical_policy"] == POLICY
    assert report["rng_policy"] == RNG_POLICY and report["model_seed"] is None
    assert (
        report["noise_projection_device"] == "cuda" and report["model_adapters"] == []
    )
    assert report["input_sha256"] == assets["captured"]["input"]["sha256"]
    assert report["checkpoint_sha256"] == assets["contract"]["checkpoint_sha256"]
    for name in ("nensemble", "batch_size", "nsteps", "seed_base"):
        assert report[name] == request[name]
    verify_candidate_runtime(candidate_runtime, report)
    records = []
    outcome = {
        "status": "failed",
        "layout": layout,
        "calibration_sha256": assets["calibration_sha256"],
        "comparisons": records,
        "candidate_runtime": candidate_runtime,
    }
    try:
        with xr.open_zarr(path, consolidated=False) as dataset:
            assert set(dataset.data_vars) == set(request["output_variables"])
            expected_coords = {
                "ensemble": np.arange(request["nensemble"]),
                "time": np.array(
                    [request["start_time"].removesuffix("Z")], dtype="datetime64[ns]"
                ),
                "lead_time": np.arange(1, request["nsteps"] + 1)
                * np.timedelta64(6, "h"),
                "lat": assets["contract"]["lat"],
                "lon": assets["contract"]["lon"],
            }
            for name, values in expected_coords.items():
                np.testing.assert_array_equal(dataset[name].values, values)
            for name in request["output_variables"]:
                assert dataset[name].dims == (
                    "ensemble",
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                )
                assert dataset[name].dtype == np.float32
            for step in range(1, request["nsteps"] + 1):
                values = np.load(
                    reference / f"step-{step:04d}.npy",
                    allow_pickle=False,
                    mmap_mode="r",
                )
                assert values.shape[:4] == (
                    request["nensemble"],
                    1,
                    1,
                    len(report["output_variables"]),
                )
                for member in range(request["nensemble"]):
                    for name in request["output_variables"]:
                        expected = values[
                            member, :, 0, report["output_variables"].index(name)
                        ]
                        actual = (
                            dataset[name]
                            .isel(ensemble=member, lead_time=step - 1)
                            .values
                        )
                        limits = assets["calibration"]["layouts"][layout]["members"][
                            str(member)
                        ]["steps"][str(step)][name]
                        metrics = compare_field(actual, expected, limits)
                        record = {
                            "member_index": member,
                            "step": step,
                            "variable": name,
                            **metrics,
                        }
                        records.append(record)
                        assert metrics["passed"], record
                del values
        outcome["status"] = "passed"
    finally:
        metrics_path.write_text(json.dumps(outcome, indent=2) + "\n")
    return len(records)


def _run(workflow, assets, work, *, layout):
    import torch
    from plugin_sdk import PrepareContext, RawRequest
    from aifs2ens_artifacts import sha256_file

    work.mkdir()
    request = _request(assets, layout)
    raw = RawRequest(
        content_type="application/json", operation="run", raw_fields=request
    )
    prior_model = workflow._model
    prepared = workflow.prepare(
        raw,
        PrepareContext(run_id="ensemble", workflow_id="earth2-aifs2ens", run_dir=work),
    )
    assert workflow._model is prior_model
    assert prepared["operation"] == "forecast_ensemble"
    assert "fanout_items" not in prepared and "fanout_profile" not in prepared
    assert (
        prepared["resource_profile"]["gpus_required"] == 1
        and prepared["batch_profile"]["enabled"] is False
    )
    params = prepared["parameters"]
    assert set(params) == {"manifest_path", "manifest_sha256"}
    manifest_path = Path(params["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert (
        sha256_file(manifest_path.parent / manifest["input_path"])
        == assets["captured"]["input"]["sha256"]
    )
    assert "members" not in manifest
    workflow.warmup({"device": "cuda"})
    model = workflow._model
    cpu_rng, gpu_rng, flags = (
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
        _math_flags(torch),
    )
    torch.cuda.reset_peak_memory_stats()
    result = workflow.execute(
        {
            "operation": prepared["operation"],
            "parameters": params,
            "run_id": "ensemble",
            "run_dir": str(work),
        }
    )
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
        gpu_rng, torch.cuda.get_rng_state()
    )
    assert flags == _math_flags(torch)
    assert workflow._model is model and model.seed is None
    assert result["complete"] is True and result["rng_policy"] == RNG_POLICY
    assert (
        result["seed_base"] == request["seed_base"]
        and result["batch_size"] == request["batch_size"]
    )
    assert "member_seeds" not in result
    assert result["input_sha256"] == assets["captured"]["input"]["sha256"]
    report = {
        "status": "comparison_pending",
        "layout": layout,
        "request": request,
        "comparison_policy": "whole_ensemble_native_reference_envelope",
        "calibration_sha256": assets["calibration_sha256"],
        "input_bytes_identical": True,
        "rng_restored": True,
        "execution_tasks": 1,
        "fanout": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "result": result,
    }
    try:
        report["field_steps_compared"] = _compare(
            result["dataset_path"],
            request,
            assets,
            layout,
            work / "parity-metrics.json",
            candidate_runtime=result["runtime"],
        )
        report["status"] = "passed"
    except (AssertionError, ValueError) as exc:
        report.update(status="numerical_mismatch", error=str(exc))
        raise
    finally:
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return model


def test_reference_repeatability(assets):
    import numpy as np

    calibration = assets["calibration"]
    assert (
        calibration["status"] == "passed"
        and calibration["frozen_before_candidate"] is True
    )
    assert (
        calibration["calibration_runs_per_layout"] >= 4
        and calibration["holdout_runs_per_layout"] == 2
    )
    assert set(calibration["layouts"]) == set(LAYOUTS)
    for layout in LAYOUTS:
        assert calibration["layouts"][layout]["holdout"]["status"] == "passed"
        assert not calibration["layouts"][layout]["holdout"]["failures"]
    root = Path(assets["references"]["full"]["path"])
    first = np.load(root / "step-0001.npy", allow_pickle=False, mmap_mode="r")
    assert not np.array_equal(first[0], first[1], equal_nan=True), (
        "shared input must still produce distinct ensemble trajectories"
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_full_ensemble_four_steps(assets, workflow, tmp_path, monkeypatch, backend):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", backend)
    _run(workflow, assets, tmp_path / backend, layout="full")


def test_serial_batches_shared_rng(assets, workflow, tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    _run(workflow, assets, tmp_path / "serial", layout="serial")


def test_uneven_final_batch(assets, workflow, tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    _run(workflow, assets, tmp_path / "uneven", layout="uneven")


def test_cached_ensemble_A_B_A(assets, workflow, tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    model = _run(workflow, assets, tmp_path / "A", layout="full")
    assert _run(workflow, assets, tmp_path / "B", layout="serial") is model
    assert _run(workflow, assets, tmp_path / "A-repeat", layout="full") is model


def test_rest_single_gpu_no_fanout(assets, tmp_path, monkeypatch):
    """One complete ensemble task through the real Rust service, Redis and GPU worker."""
    import signal
    import subprocess
    import time
    import urllib.request
    import plugin_dev

    root = assets["root"]
    monkeypatch.setenv(
        "PATH",
        f"{root}/tools/bin:/root/.cargo/bin:/usr/local/cuda/bin:{os.environ['PATH']}",
    )
    monkeypatch.setenv("PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND", "rust")
    plugin_dev.ensure_run_local_prerequisites(skip_build=True)
    plan = plugin_dev.build_run_local_plan(
        REPO / "plugins/earth2-aifs2ens",
        workspace=tmp_path / "service",
        port=0,
        redis_port=0,
    )
    processes, logs = [], []

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            plan["server_url"] + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.load(response)

    try:
        for item in plan["processes"]:
            log = (Path(plan["workspace"]) / "logs" / f"{item['name']}.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                item["argv"],
                cwd=REPO,
                env={**os.environ, **item.get("env", {})},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)
            if item["name"] in {"redis", "inference_server"}:
                port = plan["redis_port"] if item["name"] == "redis" else plan["port"]
                plugin_dev._wait_for_tcp_port("127.0.0.1", port, timeout_secs=60)
        for stream in plan["execute_registration_streams"]:
            plugin_dev._wait_for_worker_registration(
                plan["redis_url"], stream, timeout_secs=180
            )
        payload = _request(assets, "full")
        route = f"/v1/infer/{plan['workflow_id']}"
        status, submitted = request(f"{route}/run", payload)
        assert status == 202
        run_id = submitted["run_id"]
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            assert all(proc.poll() is None for proc in processes), (
                "service process exited"
            )
            status, state = request(f"{route}/{run_id}/status")
            assert status == 200
            if state.get("status") in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(1)
        else:
            raise AssertionError(f"REST request timed out: {state}")
        assert state["status"] == "succeeded", state

        def read_result():
            status, result = request(f"{route}/{run_id}/results")
            assert status == 200
            return result

        result = wait_for_structured_result(read_result)
        assert result["execution"]["run_id"] == run_id
        assert result["payload"]["complete"] is True
        assert (
            result["payload"]["nensemble"] == 2 and result["payload"]["batch_size"] == 2
        )
        assert (
            result["payload"]["seed_base"] == 0
            and result["payload"]["rng_policy"] == RNG_POLICY
        )
        assert "member_seeds" not in result["payload"]
        assert (
            result["payload"]["input_sha256"] == assets["captured"]["input"]["sha256"]
        )
        assert any(
            a["name"] == "forecast_dataset" and a["media_type"] == "application/x-zarr"
            for a in result["execution"]["outputs"]
        )
        import redis

        with redis.Redis.from_url(plan["redis_url"], decode_responses=True) as client:
            releases = wait_for_parent_release(lambda: client.xrange("release"), run_id)
        assert len(releases) == 1 and releases[0][1]["run_id"] == run_id
        (tmp_path / "service-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        count = _compare(
            result["payload"]["dataset_path"],
            payload,
            assets,
            "full",
            tmp_path / "parity-metrics.json",
            candidate_runtime=result["payload"]["runtime"],
        )
        (tmp_path / "report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "layout": "full",
                    "run_id": run_id,
                    "execution_tasks": 1,
                    "fanout": False,
                    "field_steps_compared": count,
                    "comparison_policy": "whole_ensemble_native_reference_envelope",
                    "calibration_sha256": assets["calibration_sha256"],
                    "result": result,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for proc in reversed(processes):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in reversed(processes):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
        for log in logs:
            log.close()
