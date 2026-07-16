# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live E2E for the first PhysicsNeMo-CFD workflow plugin."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import pytest
import requests


WORKFLOW_ID = "physicsnemo-cfd-surface-benchmark"
REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = (
    REPO_ROOT / "plugins" / WORKFLOW_ID / "examples" / "public_run_1_request.json"
)
EXPECTED_PIPELINE = ["prepare", "prefetch", "schedule", "execute", "results"]
EXPECTED_PROVIDER = {
    "repository": "https://github.com/NVIDIA/physicsnemo-cfd.git",
    "tag": "v0.0.2",
    "version": "0.0.2",
    "commit": "921f14dc2ac14c04aabffaba3290deb792379dd8",
    "physicsnemo_version": "2.1.1",
    "python_version": "3.12",
}
EXPECTED_ARTIFACTS = {
    "benchmark_results.json",
    "benchmark_results.csv",
    "benchmark_results.html",
    "resolved_config.json",
    "benchmark_diagnostics.json",
    "benchmark.log",
}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_STAGES = {"completed", "failed"}

pytestmark = [
    pytest.mark.cfd_e2e,
    pytest.mark.rust_only,
    pytest.mark.skipif(
        os.environ.get("QA_CFD_E2E_ENABLED", "").strip() != "1",
        reason="set QA_CFD_E2E_ENABLED=1 to run the expensive live CFD E2E",
    ),
]


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _evidence_root(run_id: str) -> Path:
    configured = os.environ.get("QA_CFD_E2E_ARTIFACT_DIR", "").strip()
    root = Path(configured) if configured else Path("artifacts") / "cfd-e2e"
    target = root / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_report(report: object) -> None:
    assert isinstance(report, list) and len(report) == 1
    row = report[0]
    assert isinstance(row, dict)
    assert row.get("skipped") is not True
    assert row.get("model") == "domino_surface"
    assert row.get("dataset") == "drivaerml"
    assert row.get("cases") == ["run_1"]

    summary_metrics = row.get("metrics")
    assert isinstance(summary_metrics, dict)
    assert "l2_pressure" in summary_metrics
    assert math.isfinite(float(summary_metrics["l2_pressure"]))

    per_case = row.get("per_case")
    assert isinstance(per_case, list) and len(per_case) == 1
    assert per_case[0].get("case_id") == "run_1"
    case_metrics = per_case[0].get("metrics")
    assert isinstance(case_metrics, dict)
    assert math.isfinite(float(case_metrics["l2_pressure"]))


def test_physicsnemo_cfd_surface_public_run_1(client, user_token):
    request_payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))

    health = client.get(client.base_url + "/readyz")
    health.raise_for_status()
    assert health.json().get("status") == "ready"

    workflows_response = client.get(client.base_url + "/v1/infer/workflows")
    workflows_response.raise_for_status()
    workflows = workflows_response.json().get("workflows", [])
    workflow_ids = {
        item.get("name") if isinstance(item, dict) else item for item in workflows
    }
    assert WORKFLOW_ID in workflow_ids

    readiness_response = client.get(
        client.base_url + f"/v1/infer/{WORKFLOW_ID}/readiness"
    )
    readiness_response.raise_for_status()
    readiness = readiness_response.json()
    assert readiness.get("workflow_id") == WORKFLOW_ID
    assert readiness.get("readiness", {}).get("ready") is True, readiness

    # The shared QA client retries POST. A fresh default Session deliberately does
    # not: this API has no idempotency key, so retrying could launch two GPU jobs.
    submit_timeout = _positive_env_int("QA_CFD_E2E_SUBMIT_TIMEOUT_SECS", 300)
    with requests.Session() as submit_session:
        submit_response = submit_session.post(
            client.base_url + f"/v1/infer/{WORKFLOW_ID}/run",
            headers={
                "Authorization": f"Bearer {user_token}",
                "Content-Type": "application/json",
            },
            json={"parameters": request_payload},
            timeout=submit_timeout,
        )
    assert submit_response.status_code == 202, submit_response.text
    submitted = submit_response.json()
    assert submitted.get("workflow") == WORKFLOW_ID
    assert submitted.get("status") == "queued"
    assert submitted.get("pipeline") == EXPECTED_PIPELINE
    run_id = submitted["run_id"]

    evidence = _evidence_root(run_id)
    _write_json(evidence / "request.json", request_payload)
    _write_json(evidence / "submit.json", submitted)

    timeout_seconds = _positive_env_int("QA_CFD_E2E_TIMEOUT_SECS", 23_400)
    poll_seconds = _positive_env_int("QA_CFD_E2E_POLL_SECS", 20)
    deadline = time.monotonic() + timeout_seconds
    final_status = None
    while time.monotonic() < deadline:
        status_response = client.get(
            client.base_url + f"/v1/infer/{WORKFLOW_ID}/{run_id}/status"
        )
        status_response.raise_for_status()
        current = status_response.json()
        _write_json(evidence / "latest_status.json", current)
        if (
            str(current.get("status") or "") in TERMINAL_STATUSES
            or str(current.get("stage") or "") in TERMINAL_STAGES
        ):
            final_status = current
            break
        time.sleep(poll_seconds)

    assert final_status is not None, (
        f"run {run_id} did not reach a terminal state within {timeout_seconds}s; "
        f"last status is in {evidence / 'latest_status.json'}"
    )
    _write_json(evidence / "final_status.json", final_status)
    assert final_status.get("status") == "succeeded", final_status

    results_response = client.get(
        client.base_url + f"/v1/infer/{WORKFLOW_ID}/{run_id}/results"
    )
    results_response.raise_for_status()
    results = results_response.json()
    _write_json(evidence / "results.json", results)
    assert set(results) == {"request", "execution", "payload"}

    execution = results["execution"]
    assert execution.get("status") == "succeeded"
    outputs = execution.get("outputs")
    assert isinstance(outputs, list)
    output_names = {output.get("name") for output in outputs}
    assert EXPECTED_ARTIFACTS <= output_names
    primary_outputs = [output for output in outputs if output.get("primary") is True]
    assert len(primary_outputs) == 1
    assert primary_outputs[0].get("name") == "benchmark_results.json"

    payload = results["payload"]
    assert payload.get("model_names") == ["domino_surface"]
    assert payload.get("case_ids") == ["run_1"]
    assert payload.get("selected_metrics") == ["l2_pressure"]
    assert payload.get("provider") == EXPECTED_PROVIDER
    assert re.fullmatch(r"[0-9a-f]{64}", payload.get("preset_sha256", ""))
    assert payload.get("case_digests") == [
        {
            "case_id": "run_1",
            "sha256": "01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40",
            "size_bytes": 659606189,
            "geometry_sha256": "411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2",
            "geometry_size_bytes": 142385186,
        }
    ]
    assert EXPECTED_ARTIFACTS <= set(payload.get("registered_artifact_names", []))

    downloaded: dict[str, bytes] = {}
    for artifact_name in sorted(EXPECTED_ARTIFACTS):
        response = client.get(
            client.base_url + f"/v1/infer/{WORKFLOW_ID}/{run_id}/results",
            params={"artifact": artifact_name},
        )
        response.raise_for_status()
        downloaded[artifact_name] = response.content
        (evidence / artifact_name).write_bytes(response.content)

    primary_response = client.get(
        client.base_url + f"/v1/infer/{WORKFLOW_ID}/{run_id}/results",
        params={"artifact": "primary"},
    )
    primary_response.raise_for_status()
    assert primary_response.content == downloaded["benchmark_results.json"]

    report = json.loads(downloaded["benchmark_results.json"])
    _assert_report(report)
