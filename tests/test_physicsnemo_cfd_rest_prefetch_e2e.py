# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in local REST-to-prefetch acceptance test for the CFD surface plugin.

This deliberately stops at the schedule stream.  It proves the real HTTP API,
Python prepare hook, verified HTTPS downloader, digest-addressed cache, and
Redis handoff without starting a scheduler or a GPU execution worker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ID = "physicsnemo-cfd-surface-benchmark"
PLUGIN_ROOT = REPO_ROOT / "plugins" / WORKFLOW_ID
REQUEST_PATH = PLUGIN_ROOT / "examples" / "prefetch_smoke_request.json"
FIXTURE_URL = (
    "https://raw.githubusercontent.com/Kitware/vtk-examples/"
    "5bdc0728cfe386fcba9df4250aa48fbd3ac05fab/"
    "src/SupplementaryData/Cxx/Visualization/Ring.vtp"
)
FIXTURE_SHA256 = "4996c7de165682f3bf35a0b4c1c019c366dde3ffafbc56d815a1a54a08cf0678"
FIXTURE_SIZE_BYTES = 5297
EXPECTED_PIPELINE = ["prepare", "prefetch", "schedule", "execute", "results"]

pytestmark = pytest.mark.skipif(
    os.environ.get("QA_CFD_PREFETCH_E2E_ENABLED", "").strip() != "1",
    reason="set QA_CFD_PREFETCH_E2E_ENABLED=1 to run the networked local acceptance test",
)


def _load_plugin_dev_module():
    script_path = REPO_ROOT / "scripts" / "plugin_dev.py"
    spec = importlib.util.spec_from_file_location(
        "cfd_prefetch_plugin_dev", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_readiness_stub(root: Path) -> None:
    """Satisfy only the import-light startup probe in a non-CFD test Python."""
    package = root / "physicsnemo" / "cfd" / "evaluation" / "benchmarks"
    package.mkdir(parents=True)
    for directory in [
        root / "physicsnemo",
        root / "physicsnemo" / "cfd",
        root / "physicsnemo" / "cfd" / "evaluation",
        package,
    ]:
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        'raise RuntimeError("readiness-only stub must never execute")\n',
        encoding="utf-8",
    )


def _request_json(
    url: str, *, method: str = "GET", payload: object | None = None
) -> tuple[int, dict]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _wait_for_schedule(
    server_url: str, run_id: str, *, timeout_seconds: float = 45
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    latest: dict = {}
    while time.monotonic() < deadline:
        status, latest = _request_json(
            f"{server_url}/v1/infer/{WORKFLOW_ID}/{run_id}/status"
        )
        assert status == 200, latest
        if latest.get("stage") == "schedule":
            return latest
        if latest.get("status") in {"failed", "cancelled"}:
            pytest.fail(f"run failed before schedule handoff: {latest}")
        time.sleep(0.2)
    pytest.fail(f"timed out waiting for schedule handoff: {latest}")


def _wait_for_failure(
    server_url: str, run_id: str, *, timeout_seconds: float = 45
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    latest: dict = {}
    while time.monotonic() < deadline:
        status, latest = _request_json(
            f"{server_url}/v1/infer/{WORKFLOW_ID}/{run_id}/status"
        )
        assert status == 200, latest
        if latest.get("status") == "failed":
            return latest
        if latest.get("stage") == "schedule":
            pytest.fail(f"invalid verified asset reached schedule: {latest}")
        time.sleep(0.2)
    pytest.fail(f"timed out waiting for prefetch failure: {latest}")


def _stream_run_ids(redis_client, stream: str) -> set[str]:
    run_ids: set[str] = set()
    for _entry_id, fields in redis_client.xrange(stream):
        run_id = fields.get(b"run_id", fields.get("run_id"))
        if isinstance(run_id, bytes):
            run_id = run_id.decode("utf-8")
        run_ids.add(run_id)
    return run_ids


def _schedule_payload(redis_client, stream: str, run_id: str) -> dict:
    for _entry_id, fields in redis_client.xrange(stream):
        decoded_run_id = fields.get(b"run_id", fields.get("run_id"))
        if isinstance(decoded_run_id, bytes):
            decoded_run_id = decoded_run_id.decode("utf-8")
        if decoded_run_id != run_id:
            continue
        raw_payload = fields.get(b"payload", fields.get("payload"))
        if isinstance(raw_payload, bytes):
            raw_payload = raw_payload.decode("utf-8")
        return json.loads(raw_payload)
    raise AssertionError(f"run {run_id} was not present in Redis stream {stream}")


def _assert_verified_artifact(payload: dict, cache_root: Path) -> Path:
    assert payload["stage_context"]["current_phase"] == "schedule"
    assert payload["prefetch_errors"] == 0
    assert payload["prefetch_required_errors"] == 0
    assert payload["prefetch_optional_errors"] == 0
    assert payload["prefetch_plan_count"] == 2
    assert payload["prefetch_plan"][0]["expected_sha256"] == FIXTURE_SHA256
    assert payload["prefetch_plan"][0]["expected_size_bytes"] == FIXTURE_SIZE_BYTES
    assert (
        payload["prefetch_plan"][1]["target_artifact_name"] == "surface-geometry-run_1"
    )
    assert payload["prefetch_plan"][1]["expected_sha256"] == FIXTURE_SHA256
    assert payload["prefetch_plan"][1]["expected_size_bytes"] == FIXTURE_SIZE_BYTES

    artifacts = payload["prefetch_artifacts"]
    assert len(artifacts) == 2
    artifact = artifacts[0]
    assert artifact["name"] == "surface-mesh-run_1"
    assert artifact["sha256"] == FIXTURE_SHA256
    assert artifact["size_bytes"] == FIXTURE_SIZE_BYTES
    geometry_artifact = artifacts[1]
    assert geometry_artifact["name"] == "surface-geometry-run_1"
    assert geometry_artifact["sha256"] == FIXTURE_SHA256
    assert geometry_artifact["size_bytes"] == FIXTURE_SIZE_BYTES

    cache_path = Path(artifact["storage_path"]).resolve()
    assert Path(geometry_artifact["storage_path"]).resolve() == cache_path
    expected_path = (cache_root / "prefetch" / "sha256" / FIXTURE_SHA256).resolve()
    assert cache_path == expected_path
    contents = cache_path.read_bytes()
    assert b'<VTKFile type="PolyData"' in contents[:256]
    assert cache_path.stat().st_size == FIXTURE_SIZE_BYTES
    assert hashlib.sha256(contents).hexdigest() == FIXTURE_SHA256

    metadata_path = Path(str(cache_path) + ".metadata.json")
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "sha256": FIXTURE_SHA256,
        "size_bytes": FIXTURE_SIZE_BYTES,
    }
    assert not list(cache_root.rglob("*.part-*"))
    return cache_path


def test_cfd_rest_prepare_and_verified_prefetch_reach_schedule_without_gpu(
    tmp_path: Path,
):
    redis = pytest.importorskip("redis")
    module = _load_plugin_dev_module()

    skip_build = os.environ.get("QA_CFD_PREFETCH_E2E_SKIP_BUILD", "").strip() == "1"
    try:
        module.ensure_run_local_prerequisites(skip_build=skip_build)
    except ValueError as error:
        if "redis-server" in str(error):
            pytest.skip(str(error))
        raise
    if not skip_build:
        module.build_run_local_binaries()

    workspace = tmp_path / "cfd-rest-prefetch"
    plan = module.build_run_local_plan(
        PLUGIN_ROOT,
        workspace=workspace,
        port=0,
        redis_port=0,
    )
    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    runtime_config["max_retries"] = 1
    Path(plan["runtime_config_path"]).write_text(
        json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8"
    )
    schedule_stream = f"{runtime_config['stream_prefix']}schedule"
    cache_root = workspace / "cache"

    readiness_stub = workspace / "readiness-stub"
    _write_readiness_stub(readiness_stub)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path
        for path in [
            str(readiness_stub),
            str(REPO_ROOT / "scripts"),
            str(REPO_ROOT / "python"),
            inherited_pythonpath,
        ]
        if path
    )

    selected_processes = [
        process
        for process in plan["processes"]
        if process["name"] in {"redis", "inference_server", "prepare", "prefetch"}
    ]
    assert [process["name"] for process in selected_processes] == [
        "redis",
        "inference_server",
        "prepare",
        "prefetch",
    ]
    assert all(
        process["name"] != "runtime_env_launcher" for process in selected_processes
    )

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    log_handles = []
    try:
        for process in selected_processes:
            log_path = workspace / "logs" / f"{process['name']}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)
            env = os.environ.copy()
            env.update(process.get("env", {}))
            env.update(
                {
                    "PYTHONPATH": pythonpath,
                    "E2S_EXT_CACHE": str(cache_root),
                    "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS": "raw.githubusercontent.com",
                    "E2S_PREFETCH_MAX_OBJECT_BYTES": str(1024 * 1024),
                    "E2S_PREFETCH_MAX_REQUEST_BYTES": str(1024 * 1024),
                    "E2S_DOWNLOAD_TIMEOUT_SECS": "60",
                }
            )
            child = subprocess.Popen(
                process["argv"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process["name"], child))
            if process["name"] == "redis":
                module._wait_for_tcp_port(
                    "127.0.0.1", int(plan["redis_port"]), timeout_secs=10
                )
            elif process["name"] == "inference_server":
                module._wait_for_tcp_port(
                    "127.0.0.1", int(plan["port"]), timeout_secs=30
                )

        ready_status, ready = _request_json(plan["server_url"] + "/readyz")
        assert ready_status == 200 and ready["status"] == "ready"

        readiness_status, readiness = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/readiness"
        )
        assert readiness_status == 200
        assert readiness["readiness"]["ready"] is True, readiness

        schema_status, schema = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/schema"
        )
        assert schema_status == 200
        assert schema["request_schemas"]["application/json"]["required"] == [
            "models",
            "cases",
        ]

        invalid_status, invalid = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/run",
            method="POST",
            payload={"models": [], "cases": []},
        )
        assert invalid_status == 422, invalid

        request_payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        assert request_payload["cases"] == [
            {
                "case_id": "run_1",
                "mesh_uri": FIXTURE_URL,
                "sha256": FIXTURE_SHA256,
                "size_bytes": FIXTURE_SIZE_BYTES,
                "geometry_uri": FIXTURE_URL,
                "geometry_sha256": FIXTURE_SHA256,
                "geometry_size_bytes": FIXTURE_SIZE_BYTES,
            }
        ]

        redis_client = redis.Redis.from_url(plan["redis_url"])
        invalid_digest_payload = json.loads(json.dumps(request_payload))
        invalid_digest_payload["cases"][0]["sha256"] = "0" * 64
        invalid_digest_payload["cases"][0]["geometry_sha256"] = "1" * 64
        failed_status, failed_submit = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/run",
            method="POST",
            payload=invalid_digest_payload,
        )
        assert failed_status == 202, failed_submit
        failed_run_id = failed_submit["run_id"]
        failed_run = _wait_for_failure(plan["server_url"], failed_run_id)
        assert failed_run["stage"] == "dlq"
        assert failed_run_id not in _stream_run_ids(redis_client, schedule_stream)
        rejected_cache_path = cache_root / "prefetch" / "sha256" / ("0" * 64)
        rejected_geometry_cache_path = cache_root / "prefetch" / "sha256" / ("1" * 64)
        assert not rejected_cache_path.exists()
        assert not rejected_geometry_cache_path.exists()
        assert not Path(str(rejected_cache_path) + ".metadata.json").exists()
        assert not Path(str(rejected_geometry_cache_path) + ".metadata.json").exists()
        assert not list(cache_root.rglob("*.part-*"))

        first_status, first_submit = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/run",
            method="POST",
            payload=request_payload,
        )
        assert first_status == 202, first_submit
        assert first_submit["pipeline"] == EXPECTED_PIPELINE
        first_run_id = first_submit["run_id"]
        _wait_for_schedule(plan["server_url"], first_run_id)

        first_payload = _schedule_payload(redis_client, schedule_stream, first_run_id)
        assert first_payload["prefetch_downloaded"] == 1
        assert first_payload["prefetch_cached"] == 0
        cache_path = _assert_verified_artifact(first_payload, cache_root)
        first_mtime_ns = cache_path.stat().st_mtime_ns

        second_status, second_submit = _request_json(
            plan["server_url"] + f"/v1/infer/{WORKFLOW_ID}/run",
            method="POST",
            payload=request_payload,
        )
        assert second_status == 202, second_submit
        second_run_id = second_submit["run_id"]
        _wait_for_schedule(plan["server_url"], second_run_id)
        second_payload = _schedule_payload(redis_client, schedule_stream, second_run_id)
        assert second_payload["prefetch_downloaded"] == 0
        assert second_payload["prefetch_cached"] == 1
        _assert_verified_artifact(second_payload, cache_root)
        assert cache_path.stat().st_mtime_ns == first_mtime_ns
    finally:
        module._terminate_processes(processes, suppress_interrupts=True)
        for log_handle in log_handles:
            log_handle.close()
