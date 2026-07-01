# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sustained load/stability tests for PhysicsNeMo Serve inference endpoints."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import pathlib
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from helpers import RequestStatus, skip_if_workflow_disabled


pytestmark = [pytest.mark.stress, pytest.mark.rust_only]

logger = logging.getLogger(__name__)

GPU_METRIC_RE = re.compile(
    r"^physicsnemo_serve_gpu_(?:compute_utilization_percent|"
    r"memory_(?:bus_utilization_percent|used_bytes|total_bytes))"
    r'\{[^}]*gpu_id="([^"]+)"[^}]*\}\s+'
)
GPU_STREAM_RE = re.compile(r":gpu:[^:]+:[^:]+:(\d+)$")

DEFAULT_STRESS_WORKFLOW = "earth2-deterministic"
DEFAULT_STRESS_PARAMS = {
    "earth2-deterministic": {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 1,
    },
}

TERMINAL_STATUSES = {
    RequestStatus.COMPLETED.value,
    RequestStatus.FAILED.value,
    RequestStatus.CANCELLED.value,
}


@dataclass
class PendingExecution:
    exec_id: str
    submitted_at: float
    sequence: int


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise pytest.UsageError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_optional_int(name: str, *, minimum: int = 1) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise pytest.UsageError(f"{name} must be >= {minimum}, got {value}")
    return value


def _sort_gpu_ids(gpu_ids: set[str]) -> list[str]:
    def key(gpu_id: str) -> tuple[int, int | str]:
        if gpu_id.isdigit():
            return (0, int(gpu_id))
        return (1, gpu_id)

    return sorted(gpu_ids, key=key)


def _gpu_ids_from_metrics(metrics_text: str) -> list[str]:
    gpu_ids = set()
    for line in metrics_text.splitlines():
        match = GPU_METRIC_RE.match(line)
        if match:
            gpu_ids.add(match.group(1))
    return _sort_gpu_ids(gpu_ids)


def _fetch_gpu_ids(client) -> list[str]:
    for path in ("/metrics", "/v1/metrics"):
        resp = client.get(client.base_url + path)
        if resp.status_code != 200:
            continue
        gpu_ids = _gpu_ids_from_metrics(resp.text)
        if gpu_ids:
            logger.info("Stress endpoint GPU ids from %s: %s", path, gpu_ids)
            return gpu_ids
    return []


def _wait_for_gpu_ids(client) -> list[str]:
    expected_count = _env_optional_int("QA_STRESS_GPU_COUNT")
    timeout = _env_int("QA_STRESS_GPU_READY_TIMEOUT_SECS", 600)
    poll_interval = _env_int("QA_STRESS_GPU_READY_POLL_SECS", 10)
    deadline = time.monotonic() + timeout
    last_gpu_ids: list[str] = []

    while time.monotonic() < deadline:
        gpu_ids = _fetch_gpu_ids(client)
        last_gpu_ids = gpu_ids
        if expected_count is None and gpu_ids:
            return gpu_ids
        if expected_count is not None and len(gpu_ids) == expected_count:
            return gpu_ids
        logger.info(
            "Waiting for stress GPU count: expected=%s observed=%s gpu_ids=%s",
            expected_count or "any",
            len(gpu_ids),
            gpu_ids,
        )
        time.sleep(poll_interval)

    if expected_count is None:
        pytest.skip("Stress test requires at least one visible GPU from /metrics")
    raise AssertionError(
        "Stress test expected "
        f"QA_STRESS_GPU_COUNT={expected_count}, observed {len(last_gpu_ids)} "
        f"GPU ids: {last_gpu_ids}"
    )


def _load_stress_params(workflow_name: str) -> dict[str, Any]:
    request_file = os.environ.get("QA_STRESS_REQUEST_FILE", "").strip()
    request_json = os.environ.get("QA_STRESS_REQUEST_JSON", "").strip()
    if request_file and request_json:
        raise pytest.UsageError(
            "Set only one of QA_STRESS_REQUEST_FILE or QA_STRESS_REQUEST_JSON"
        )
    if request_file:
        return json.loads(pathlib.Path(request_file).read_text(encoding="utf-8"))
    if request_json:
        return json.loads(request_json)
    if workflow_name in DEFAULT_STRESS_PARAMS:
        return copy.deepcopy(DEFAULT_STRESS_PARAMS[workflow_name])
    raise pytest.UsageError(
        f"No default stress request parameters for workflow {workflow_name!r}; "
        "set QA_STRESS_REQUEST_JSON or QA_STRESS_REQUEST_FILE"
    )


def _submit_request(client, adapter, workflow_name: str, params: dict[str, Any]) -> str:
    resp = client.post(
        client.base_url + adapter.submit_url(workflow_name),
        json=adapter.format_submit_body(params),
    )
    resp.raise_for_status()
    parsed = adapter.parse_submit_response(resp.json())
    status = parsed.get("status")
    if status not in {RequestStatus.ACCEPTED.value, RequestStatus.QUEUED.value}:
        raise AssertionError(f"Unexpected submit status for {workflow_name}: {status}")
    return str(parsed["execution_id"])


def _fetch_status(client, adapter, workflow_name: str, exec_id: str) -> dict[str, Any]:
    resp = client.get(client.base_url + adapter.status_url(workflow_name, exec_id))
    resp.raise_for_status()
    return adapter.parse_status_response(resp.json())


def _fetch_result(client, adapter, workflow_name: str, exec_id: str) -> dict[str, Any]:
    resp = client.get(client.base_url + adapter.results_url(workflow_name, exec_id))
    resp.raise_for_status()
    return resp.json()


def _gpu_id_from_stream(gpu_stream: str | None) -> str | None:
    if not gpu_stream:
        return None
    match = GPU_STREAM_RE.search(gpu_stream)
    if not match:
        return None
    return match.group(1)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _execution_result_record(
    *,
    pending: PendingExecution,
    status_details: dict[str, Any],
    result_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    completed_at = time.monotonic()
    result_payload = result_payload or {}
    execution = result_payload.get("execution")
    if not isinstance(execution, dict):
        execution = {}

    gpu_stream = execution.get("gpu_stream")
    execution_time_seconds = _float_or_none(
        execution.get("execution_time_seconds")
        or status_details.get("execution_time_seconds")
    )
    return {
        "exec_id": pending.exec_id,
        "sequence": pending.sequence,
        "status": status_details["status"],
        "submitted_at_monotonic": pending.submitted_at,
        "completed_at_monotonic": completed_at,
        "wall_time_seconds": completed_at - pending.submitted_at,
        "execution_time_seconds": execution_time_seconds,
        "gpu_stream": gpu_stream,
        "gpu_id": _gpu_id_from_stream(gpu_stream),
        "error": execution.get("error") or result_payload.get("error"),
    }


def _collect_terminal_executions(
    *,
    client,
    adapter,
    workflow_name: str,
    pending: dict[str, PendingExecution],
) -> list[dict[str, Any]]:
    terminal_records = []
    for exec_id, pending_execution in list(pending.items()):
        status_details = _fetch_status(client, adapter, workflow_name, exec_id)
        status = status_details["status"]
        if status not in TERMINAL_STATUSES:
            continue

        result_payload = None
        try:
            result_payload = _fetch_result(client, adapter, workflow_name, exec_id)
        except Exception as exc:  # Results may not be materialized for failed requests.
            result_payload = {"error": f"failed to fetch result payload: {exc}"}
        terminal_records.append(
            _execution_result_record(
                pending=pending_execution,
                status_details=status_details,
                result_payload=result_payload,
            )
        )
        pending.pop(exec_id, None)
    return terminal_records


def _wait_for_terminal_status(
    *,
    client,
    adapter,
    workflow_name: str,
    exec_id: str,
    timeout: int,
    poll_interval: int,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_details = _fetch_status(client, adapter, workflow_name, exec_id)
        if status_details["status"] in TERMINAL_STATUSES:
            return status_details
        time.sleep(poll_interval)
    return None


def _wait_for_schedulable_workflow(
    *,
    client,
    adapter,
    workflow_name: str,
    params: dict[str, Any],
) -> None:
    ready_timeout = _env_int("QA_STRESS_READY_TIMEOUT_SECS", 900)
    probe_timeout = _env_int("QA_STRESS_READY_PROBE_TIMEOUT_SECS", 180)
    poll_interval = _env_int("QA_STRESS_POLL_INTERVAL_SECS", 5)
    deadline = time.monotonic() + ready_timeout
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        exec_id = _submit_request(client, adapter, workflow_name, params)
        logger.info(
            "Submitted stress readiness probe attempt=%s execution_id=%s",
            attempt,
            exec_id,
        )
        remaining = max(1, int(deadline - time.monotonic()))
        status_details = _wait_for_terminal_status(
            client=client,
            adapter=adapter,
            workflow_name=workflow_name,
            exec_id=exec_id,
            timeout=min(probe_timeout, remaining),
            poll_interval=poll_interval,
        )
        if status_details is None:
            logger.warning(
                "Stress readiness probe execution_id=%s did not finish within %ss",
                exec_id,
                probe_timeout,
            )
            continue
        if status_details["status"] == RequestStatus.COMPLETED.value:
            logger.info("Stress readiness probe succeeded: execution_id=%s", exec_id)
            return

        result_payload = None
        try:
            result_payload = _fetch_result(client, adapter, workflow_name, exec_id)
        except Exception as exc:
            result_payload = {"error": f"failed to fetch result payload: {exc}"}
        raise AssertionError(
            "Stress readiness probe reached terminal non-success status: "
            f"execution_id={exec_id} status={status_details['status']} "
            f"result={json.dumps(result_payload, sort_keys=True)[:2000]}"
        )

    raise RuntimeError(
        f"Stress workflow {workflow_name!r} did not become schedulable within "
        f"{ready_timeout}s"
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize_stress_run(
    *,
    workflow_name: str,
    gpu_ids: list[str],
    target_inflight: int,
    concurrency_per_gpu: int,
    duration_secs: int,
    started_at: float,
    ended_at: float,
    records: list[dict[str, Any]],
    timed_out: list[dict[str, Any]],
    inflight_samples: list[int],
) -> dict[str, Any]:
    completed = [
        record
        for record in records
        if record["status"] == RequestStatus.COMPLETED.value
    ]
    failed = [
        record for record in records if record["status"] == RequestStatus.FAILED.value
    ]
    cancelled = [
        record
        for record in records
        if record["status"] == RequestStatus.CANCELLED.value
    ]
    execution_times = [
        record["execution_time_seconds"]
        for record in completed
        if record.get("execution_time_seconds") is not None
    ]
    wall_times = [
        record["wall_time_seconds"]
        for record in completed
        if record.get("wall_time_seconds") is not None
    ]
    per_gpu_completed = Counter(
        record["gpu_id"] for record in completed if record.get("gpu_id") is not None
    )
    total_wall = max(ended_at - started_at, 0.001)
    error_examples = [
        {
            "exec_id": record["exec_id"],
            "status": record["status"],
            "error": record.get("error"),
        }
        for record in [*failed, *cancelled, *timed_out][:5]
    ]
    return {
        "workflow": workflow_name,
        "gpu_ids": gpu_ids,
        "concurrency_per_gpu": concurrency_per_gpu,
        "target_inflight": target_inflight,
        "configured_duration_seconds": duration_secs,
        "wall_time_seconds": total_wall,
        "submitted": len(records) + len(timed_out),
        "completed": len(completed),
        "failed": len(failed),
        "cancelled": len(cancelled),
        "timed_out": len(timed_out),
        "throughput_per_minute": len(completed) / total_wall * 60.0,
        "max_inflight": max(inflight_samples) if inflight_samples else 0,
        "avg_inflight": (
            sum(inflight_samples) / len(inflight_samples) if inflight_samples else 0
        ),
        "completed_per_gpu": {
            gpu_id: per_gpu_completed.get(gpu_id, 0) for gpu_id in gpu_ids
        },
        "execution_time_seconds": {
            "p50": _percentile(execution_times, 0.50),
            "p90": _percentile(execution_times, 0.90),
            "p99": _percentile(execution_times, 0.99),
        },
        "wall_time_per_request_seconds": {
            "p50": _percentile(wall_times, 0.50),
            "p90": _percentile(wall_times, 0.90),
            "p99": _percentile(wall_times, 0.99),
        },
        "error_examples": error_examples,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_stress_summary(summary: dict[str, Any]) -> pathlib.Path:
    summary_path = os.environ.get("QA_STRESS_SUMMARY_PATH", "").strip()
    if summary_path:
        path = pathlib.Path(summary_path)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = pathlib.Path("reports") / f"stress_summary_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**summary, "summary_path": str(path)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _skip_unless_stress_marker_selected(request) -> None:
    markexpr = str(getattr(request.config.option, "markexpr", "") or "")
    if "stress" in markexpr or os.environ.get("QA_STRESS_FORCE", "").strip() == "1":
        return
    pytest.skip(
        "Stress tests run only when selected with -m stress or QA_STRESS_FORCE=1"
    )


def test_sustained_workflow_concurrency_stress(request, client, adapter):
    """Keep target workflow concurrency full for a configured stress interval."""
    _skip_unless_stress_marker_selected(request)

    workflow_name = os.environ.get(
        "QA_STRESS_WORKFLOW", DEFAULT_STRESS_WORKFLOW
    ).strip()
    if not workflow_name:
        raise pytest.UsageError("QA_STRESS_WORKFLOW must be non-empty")

    skip_if_workflow_disabled(adapter, workflow_name)

    params = _load_stress_params(workflow_name)
    gpu_ids = _wait_for_gpu_ids(client)
    concurrency_per_gpu = _env_int("QA_STRESS_CONCURRENCY_PER_GPU", 2)
    duration_secs = _env_int("QA_STRESS_DURATION_SECS", 600)
    drain_timeout_secs = _env_int("QA_STRESS_DRAIN_TIMEOUT_SECS", 1800)
    poll_interval = _env_int("QA_STRESS_POLL_INTERVAL_SECS", 5)
    target_inflight = len(gpu_ids) * concurrency_per_gpu

    logger.info(
        "Starting stress readiness probe: workflow=%s gpu_ids=%s target_inflight=%s",
        workflow_name,
        gpu_ids,
        target_inflight,
    )
    _wait_for_schedulable_workflow(
        client=client,
        adapter=adapter,
        workflow_name=workflow_name,
        params=params,
    )

    pending: dict[str, PendingExecution] = {}
    records: list[dict[str, Any]] = []
    timed_out: list[dict[str, Any]] = []
    inflight_samples: list[int] = []
    sequence = 0
    started_at = time.monotonic()
    active_deadline = started_at + duration_secs

    logger.info(
        "Starting stress run: workflow=%s duration=%ss target_inflight=%s",
        workflow_name,
        duration_secs,
        target_inflight,
    )

    while time.monotonic() < active_deadline:
        records.extend(
            _collect_terminal_executions(
                client=client,
                adapter=adapter,
                workflow_name=workflow_name,
                pending=pending,
            )
        )

        while len(pending) < target_inflight and time.monotonic() < active_deadline:
            sequence += 1
            exec_id = _submit_request(client, adapter, workflow_name, params)
            pending[exec_id] = PendingExecution(
                exec_id=exec_id,
                submitted_at=time.monotonic(),
                sequence=sequence,
            )
            logger.info(
                "Stress submitted execution_id=%s sequence=%s inflight=%s/%s",
                exec_id,
                sequence,
                len(pending),
                target_inflight,
            )

        inflight_samples.append(len(pending))
        if pending and time.monotonic() < active_deadline:
            time.sleep(poll_interval)

    logger.info(
        "Stress active interval complete; draining %s outstanding request(s)",
        len(pending),
    )
    drain_deadline = time.monotonic() + drain_timeout_secs
    while pending and time.monotonic() < drain_deadline:
        records.extend(
            _collect_terminal_executions(
                client=client,
                adapter=adapter,
                workflow_name=workflow_name,
                pending=pending,
            )
        )
        inflight_samples.append(len(pending))
        if pending:
            time.sleep(poll_interval)

    ended_at = time.monotonic()
    for exec_id, pending_execution in pending.items():
        timed_out.append(
            {
                "exec_id": exec_id,
                "sequence": pending_execution.sequence,
                "status": "timed_out",
                "submitted_at_monotonic": pending_execution.submitted_at,
                "completed_at_monotonic": ended_at,
                "wall_time_seconds": ended_at - pending_execution.submitted_at,
                "error": "request did not finish before QA_STRESS_DRAIN_TIMEOUT_SECS",
            }
        )
    pending.clear()

    summary = _summarize_stress_run(
        workflow_name=workflow_name,
        gpu_ids=gpu_ids,
        target_inflight=target_inflight,
        concurrency_per_gpu=concurrency_per_gpu,
        duration_secs=duration_secs,
        started_at=started_at,
        ended_at=ended_at,
        records=records,
        timed_out=timed_out,
        inflight_samples=inflight_samples,
    )
    summary_path = _write_stress_summary(summary)
    summary["summary_path"] = str(summary_path)
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    logger.info("Stress test summary:\n%s", summary_text)
    print(f"\nSTRESS_TEST_SUMMARY\n{summary_text}")

    assert summary["completed"] > 0, "Stress test did not complete any requests"
    assert summary["failed"] == 0, "Stress test had failed requests"
    assert summary["cancelled"] == 0, "Stress test had cancelled requests"
    assert summary["timed_out"] == 0, "Stress test had requests left after drain"
