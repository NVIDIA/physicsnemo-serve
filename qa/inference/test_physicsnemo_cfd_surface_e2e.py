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
DEFAULT_REQUEST_PATH = (
    REPO_ROOT / "plugins" / WORKFLOW_ID / "examples" / "public_run_1_request.json"
)
L2_PRESSURE_BASELINE = 0.16348028933450415
L2_PRESSURE_MAX_RELATIVE_REGRESSION = 0.01
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
EXPECTED_METRIC_OUTPUTS = {
    "l2_pressure": {"l2_pressure"},
    "l2_shear_stress": {
        "l2_shear_stress_wall_shear_stress_true_x_l2_error",
        "l2_shear_stress_wall_shear_stress_true_y_l2_error",
        "l2_shear_stress_wall_shear_stress_true_z_l2_error",
    },
    "l2_pressure_area_weighted": {"l2_pressure_area_weighted"},
    "drag": {"drag_error", "drag_true", "drag_pred"},
    "lift": {"lift_error", "lift_true", "lift_pred"},
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


def _request_path() -> Path:
    configured = os.environ.get("QA_CFD_E2E_REQUEST_PATH", "").strip()
    if not configured:
        return DEFAULT_REQUEST_PATH
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


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


def _assert_l2_pressure(value: object) -> None:
    actual = float(value)
    assert math.isfinite(actual)
    maximum = L2_PRESSURE_BASELINE * (1 + L2_PRESSURE_MAX_RELATIVE_REGRESSION)
    assert actual <= maximum, (
        f"l2_pressure regression: {actual} exceeds baseline "
        f"{L2_PRESSURE_BASELINE} by more than "
        f"{L2_PRESSURE_MAX_RELATIVE_REGRESSION:.0%}"
    )


def _assert_finite_metric(value: object) -> None:
    if isinstance(value, dict):
        assert value
        for component in value.values():
            _assert_finite_metric(component)
        return
    assert not isinstance(value, bool)
    assert math.isfinite(float(value))


def _assert_report(report: object, request_payload: dict[str, object]) -> None:
    models = request_payload["models"]
    metrics = request_payload["metrics"]
    cases = request_payload["cases"]
    assert isinstance(models, list)
    assert isinstance(metrics, list)
    assert isinstance(cases, list)
    case_ids = [case["case_id"] for case in cases]
    assert isinstance(report, list) and len(report) == len(models)
    expected_metric_outputs = set().union(
        *(EXPECTED_METRIC_OUTPUTS[name] for name in metrics)
    )
    rows_by_model = {row.get("model"): row for row in report if isinstance(row, dict)}
    assert set(rows_by_model) == set(models)

    for model in models:
        row = rows_by_model[model]
        assert row.get("skipped") is not True
        assert row.get("dataset") == "drivaerml"
        assert row.get("cases") == case_ids

        summary_metrics = row.get("metrics")
        assert isinstance(summary_metrics, dict)
        assert set(summary_metrics) == expected_metric_outputs
        for name, value in summary_metrics.items():
            _assert_finite_metric(value)
            if (
                model == "domino_surface"
                and name == "l2_pressure"
                and case_ids == ["run_1"]
            ):
                _assert_l2_pressure(value)

        per_case = row.get("per_case")
        assert isinstance(per_case, list) and len(per_case) == len(case_ids)
        cases_by_id = {
            case.get("case_id"): case for case in per_case if isinstance(case, dict)
        }
        assert set(cases_by_id) == set(case_ids)
        for case_id in case_ids:
            case_metrics = cases_by_id[case_id].get("metrics")
            assert isinstance(case_metrics, dict)
            assert set(case_metrics) == expected_metric_outputs
            for name, value in case_metrics.items():
                _assert_finite_metric(value)
                if (
                    model == "domino_surface"
                    and case_id == "run_1"
                    and name == "l2_pressure"
                ):
                    _assert_l2_pressure(value)


def _expected_case_digests(request_payload: dict[str, object]) -> list[dict]:
    cases = request_payload["cases"]
    assert isinstance(cases, list)
    keys = (
        "case_id",
        "sha256",
        "size_bytes",
        "geometry_sha256",
        "geometry_size_bytes",
    )
    return [{key: case[key] for key in keys if key in case} for case in cases]


def _submit_without_retry(
    *,
    base_url: str,
    user_token: str,
    request_payload: dict[str, object],
    timeout: int,
    warmup_attempts: int,
) -> requests.Response:
    try:
        with requests.Session() as session:
            session.headers.update(
                {
                    "Authorization": f"Bearer {user_token}",
                    "Content-Type": "application/json",
                }
            )
            warmup = None
            last_warmup_error = None
            for attempt in range(warmup_attempts):
                try:
                    warmup = session.get(
                        base_url + "/readyz",
                        timeout=min(timeout, 10),
                    )
                    break
                except (requests.ConnectionError, requests.Timeout) as exc:
                    last_warmup_error = exc
                    if attempt + 1 < warmup_attempts:
                        time.sleep(1)
            if warmup is None:
                assert last_warmup_error is not None
                raise last_warmup_error
            warmup.raise_for_status()
            return session.post(
                base_url + f"/v1/infer/{WORKFLOW_ID}/run",
                json={"parameters": request_payload},
                timeout=timeout,
            )
    except requests.RequestException as exc:
        pytest.fail(
            f"no-retry CFD submission transport failed: {type(exc).__name__}: {exc}",
            pytrace=False,
        )


def test_physicsnemo_cfd_surface_public_run_1(client, user_token):
    request_payload = json.loads(_request_path().read_text(encoding="utf-8"))

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

    # The shared QA client retries POST, so use a separate no-retry session: this
    # API has no idempotency key and retrying could launch two GPU jobs. Warm that
    # same session with a safe GET so the POST reuses an established TLS connection.
    submit_timeout = _positive_env_int("QA_CFD_E2E_SUBMIT_TIMEOUT_SECS", 300)
    warmup_attempts = _positive_env_int("QA_CFD_E2E_WARMUP_ATTEMPTS", 10)
    submit_response = _submit_without_retry(
        base_url=client.base_url,
        user_token=user_token,
        request_payload=request_payload,
        timeout=submit_timeout,
        warmup_attempts=warmup_attempts,
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
    assert payload.get("model_names") == request_payload["models"]
    assert payload.get("case_ids") == [
        case["case_id"] for case in request_payload["cases"]
    ]
    assert payload.get("selected_metrics") == request_payload["metrics"]
    assert payload.get("provider") == EXPECTED_PROVIDER
    assert re.fullmatch(r"[0-9a-f]{64}", payload.get("preset_sha256", ""))
    assert payload.get("case_digests") == _expected_case_digests(request_payload)
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
    _assert_report(report, request_payload)
