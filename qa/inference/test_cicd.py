# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI/CD test suite — intended for automated pipelines.

Run with:  pytest -m cicd --service rust --urls <URL> --token <TOKEN>

Currently mirrors the smoke suite. Expand as the CI pipeline matures.
"""

import copy
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
import requests

from helpers import (
    RequestStatus,
    run_workflow_test,
    QUICK_VALIDATION,
    skip_if_workflow_disabled,
)


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.cicd

CICD_TIMEOUT = 300
MULTIGPU_TIMEOUT = 1800
MULTIGPU_POLL_INTERVAL = 10
API_LOG_BODY_LIMIT = 10000
PARALLEL_BATCH_REQUEST_COUNT = 4
PARALLEL_BATCH_DELAY_SECONDS = 15
PARALLEL_BATCH_POLL_INTERVAL = 1
PARALLEL_BATCH_MAX_WALL_TO_ITEM_RATIO = 0.45
BATCH_METADATA_TIMEOUT = 30
BATCH_METADATA_POLL_INTERVAL = 1
GPU_METRIC_RE = re.compile(
    r"^physicsnemo_serve_gpu_(?:compute_utilization_percent|"
    r"memory_(?:bus_utilization_percent|used_bytes|total_bytes))"
    r'\{[^}]*gpu_id="([^"]+)"[^}]*\}\s+'
)


@dataclass(frozen=True)
class MultiGpuWorkflowCase:
    workflow_name: str
    params: dict
    timeout: int = MULTIGPU_TIMEOUT
    require_overlap: bool = False


DEFAULT_MULTIGPU_WORKFLOWS = "earth2-deterministic"
MULTIGPU_WORKFLOW_CASES = {
    "earth2-deterministic": MultiGpuWorkflowCase(
        workflow_name="earth2-deterministic",
        params={
            "model": "dlwp",
            "start_time": "2024-01-01T00:00:00",
            "nsteps": 1,
        },
    ),
    "deterministic-fcn3": MultiGpuWorkflowCase(
        workflow_name="deterministic_workflow",
        params={
            "model_type": "fcn3",
            "data_source": "gfs",
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 1,
            "output_format": "zarr",
            "create_plots": False,
        },
        timeout=3600,
        require_overlap=True,
    ),
    "stormcast-fcn3": MultiGpuWorkflowCase(
        workflow_name="stormcast_fcn3_workflow",
        params={
            "start_time": "2024-01-01T00:00:00",
            "num_hours": 6,
            "run_stormcast": True,
        },
        timeout=3600,
        require_overlap=True,
    ),
    "e2s-foundry-fcn3": MultiGpuWorkflowCase(
        workflow_name="e2s-foundry-fcn3",
        params={
            "start_time": "2025-01-01T00:00:00",
            "n_steps": 1,
            "n_samples": 1,
            "variables": ["t2m"],
        },
        timeout=3600,
        require_overlap=True,
    ),
}


def test_health(client, adapter):
    """Verify the service is reachable and reports healthy."""
    resp = client.get(client.base_url + adapter.health_url())
    resp.raise_for_status()
    data = adapter.parse_health_response(resp)
    assert data["status"] == "healthy", f"Service not healthy: {data}"


def test_list_workflows(client, adapter):
    """Verify we can list workflows and fetch schemas."""
    resp = client.get(client.base_url + adapter.list_workflows_url())
    resp.raise_for_status()
    workflows = adapter.parse_list_workflows_response(resp.json())
    assert len(workflows) > 0, "No workflows returned"
    logger.info(f"Available workflows: {workflows}")

    for wf in workflows:
        resp = client.get(client.base_url + adapter.workflow_schema_url(wf))
        resp.raise_for_status()


def _parse_gpu_ids_from_metrics(metrics_text):
    gpu_ids = set()
    for line in metrics_text.splitlines():
        match = GPU_METRIC_RE.match(line)
        if match:
            gpu_ids.add(match.group(1))
    return sorted(gpu_ids)


def _env_value(*names):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _configured_multigpu_gpu_count():
    raw_value = _env_value("QA_MULTIGPU_GPU_COUNT", "MULTIGPU_GPU_COUNT")
    if raw_value is None:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "QA_MULTIGPU_GPU_COUNT/MULTIGPU_GPU_COUNT must be an integer"
        ) from exc
    if value < 2:
        raise ValueError("multi-GPU tests require a GPU count of at least 2")
    return value


def _selected_multigpu_workflow_cases():
    raw_value = (
        _env_value("QA_MULTIGPU_WORKFLOWS", "MULTIGPU_WORKFLOWS")
        or DEFAULT_MULTIGPU_WORKFLOWS
    )
    selected = [name.strip().lower() for name in raw_value.split(",") if name.strip()]
    if not selected:
        selected = [DEFAULT_MULTIGPU_WORKFLOWS]

    params = []
    for name in selected:
        case = MULTIGPU_WORKFLOW_CASES.get(name)
        if case is None:
            valid = ", ".join(sorted(MULTIGPU_WORKFLOW_CASES))
            raise ValueError(
                f"unknown multi-GPU workflow preset {name!r}; valid: {valid}"
            )
        params.append(pytest.param(name, case, id=name))
    return params


def _endpoint_gpu_ids(client, attempts=6, sleep_seconds=5):
    """Return GPU ids reported by endpoint metrics, or an empty list when unavailable."""
    last_status = None
    for attempt in range(attempts):
        for path in ("/metrics", "/v1/metrics"):
            try:
                resp = client.get(client.base_url + path)
            except requests.RequestException as exc:
                last_status = f"{type(exc).__name__}: {exc}"
                continue
            last_status = resp.status_code
            if resp.status_code != 200:
                continue
            gpu_ids = _parse_gpu_ids_from_metrics(resp.text)
            if gpu_ids:
                logger.info("Endpoint GPU ids from %s: %s", path, gpu_ids)
                return gpu_ids
        if attempt + 1 < attempts:
            time.sleep(sleep_seconds)

    logger.info(
        "No GPU ids found in endpoint metrics; last metrics status=%s", last_status
    )
    return []


def _format_api_payload(payload):
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, indent=2, sort_keys=True)
        except TypeError:
            text = repr(payload)
    if len(text) <= API_LOG_BODY_LIMIT:
        return text
    truncated_chars = len(text) - API_LOG_BODY_LIMIT
    return text[:API_LOG_BODY_LIMIT] + f"... <truncated {truncated_chars} chars>"


def _response_payload(response):
    try:
        return response.json()
    except ValueError:
        return response.text


def _log_api_response(label, response):
    request = response.request
    logger.info(
        "%s API response: %s %s -> %s\n%s",
        label,
        request.method,
        request.url,
        response.status_code,
        _format_api_payload(_response_payload(response)),
    )


def _submit_workflow_once(client, adapter, workflow_name, params):
    submit_url = client.base_url + adapter.submit_url(workflow_name)
    body = adapter.format_submit_body(copy.deepcopy(params))
    logger.info(
        "Submit API request for workflow=%s: %s",
        workflow_name,
        _format_api_payload(body),
    )
    resp = requests.post(
        submit_url,
        json=body,
        headers=dict(client.headers),
        timeout=60,
    )
    _log_api_response(f"Submit workflow={workflow_name}", resp)
    resp.raise_for_status()
    parsed = adapter.parse_submit_response(resp.json())
    assert parsed.get("workflow_name") == workflow_name
    assert parsed.get("status") in [
        RequestStatus.QUEUED.value,
        RequestStatus.ACCEPTED.value,
    ]
    return parsed["execution_id"]


def _submit_workflows_concurrently(client, adapter, workflow_name, params, count):
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                _submit_workflow_once, client, adapter, workflow_name, params
            )
            for _ in range(count)
        ]
        return [future.result() for future in as_completed(futures)]


def _get_execution_status_details(client, adapter, workflow_name, exec_id):
    resp = client.get(client.base_url + adapter.status_url(workflow_name, exec_id))
    _log_api_response(f"Status execution={exec_id}", resp)
    resp.raise_for_status()
    response_payload = resp.json()
    parsed = adapter.parse_status_response(response_payload)

    progress = parsed.get("progress")
    position = parsed.get("position")
    if progress:
        cur_step = progress.get("current_step", "-")
        total_steps = progress.get("total_steps", "-")
        progress_text = f"{cur_step}/{total_steps}"
    else:
        progress_text = "n/a"
    duration = parsed.get("execution_time_seconds")
    duration_text = f"{int(float(duration))}s" if duration else None
    queue_position = f"[{position}]" if position else ""
    logger.info(
        "EXECUTION[%s]: (status: %s%s, step: %s, duration: %s)",
        exec_id,
        parsed["status"],
        queue_position,
        progress_text,
        duration_text,
    )
    return {
        "execution_id": exec_id,
        "status": parsed["status"],
        "queue_position": queue_position,
        "step": progress_text,
        "duration": duration_text,
        "batch_info": response_payload.get("batch_info"),
    }


def _log_result_api_response(client, adapter, workflow_name, exec_id):
    try:
        resp = client.get(client.base_url + adapter.results_url(workflow_name, exec_id))
    except requests.RequestException as exc:
        logger.info(
            "Result API request failed for execution=%s: %s: %s",
            exec_id,
            type(exc).__name__,
            exc,
        )
        return
    _log_api_response(f"Result execution={exec_id}", resp)


def _wait_for_all_executions(
    client,
    adapter,
    workflow_name,
    exec_ids,
    timeout,
    poll_interval=MULTIGPU_POLL_INTERVAL,
):
    pending = set(exec_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for exec_id in list(pending):
            status_details = _get_execution_status_details(
                client, adapter, workflow_name, exec_id
            )
            status = status_details["status"]
            if status == RequestStatus.COMPLETED.value:
                pending.remove(exec_id)
            elif status in [
                RequestStatus.FAILED.value,
                RequestStatus.CANCELLED.value,
            ]:
                _log_result_api_response(client, adapter, workflow_name, exec_id)
                raise RuntimeError(f"Execution[{exec_id}] ended with status={status}")
        if pending:
            time.sleep(poll_interval)

    if pending:
        raise RuntimeError(
            f"Timed out waiting for {len(pending)} executions: {sorted(pending)}"
        )


def _result_execution(client, adapter, workflow_name, exec_id):
    resp = client.get(client.base_url + adapter.results_url(workflow_name, exec_id))
    _log_api_response(f"Result execution={exec_id}", resp)
    resp.raise_for_status()
    data = resp.json()
    execution = data.get("execution", {})
    if not isinstance(execution, dict):
        raise AssertionError(f"Result for {exec_id} did not include execution metadata")
    return execution


def _wait_for_scheduler_batch_infos(
    client,
    adapter,
    workflow_name,
    exec_ids,
    *,
    timeout=BATCH_METADATA_TIMEOUT,
    poll_interval=BATCH_METADATA_POLL_INTERVAL,
):
    """Wait for downstream results persistence to expose scheduler metadata."""
    deadline = time.monotonic() + timeout
    last_statuses = {}

    while True:
        batch_infos = []
        missing_exec_ids = []
        for exec_id in exec_ids:
            status_details = _get_execution_status_details(
                client, adapter, workflow_name, exec_id
            )
            last_statuses[exec_id] = status_details
            batch_info = status_details.get("batch_info")
            if isinstance(batch_info, dict):
                batch_infos.append(batch_info)
            else:
                missing_exec_ids.append(exec_id)

        if not missing_exec_ids:
            return batch_infos

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing_statuses = {
                exec_id: last_statuses[exec_id] for exec_id in missing_exec_ids
            }
            raise AssertionError(
                "Scheduler batch metadata did not become available within "
                f"{timeout} seconds: {missing_statuses}"
            )
        time.sleep(min(poll_interval, remaining))


def _assert_same_scheduler_batch(
    client,
    adapter,
    workflow_name,
    exec_ids,
    expected_size,
    *,
    metadata_timeout=BATCH_METADATA_TIMEOUT,
    metadata_poll_interval=BATCH_METADATA_POLL_INTERVAL,
):
    batch_infos = _wait_for_scheduler_batch_infos(
        client,
        adapter,
        workflow_name,
        exec_ids,
        timeout=metadata_timeout,
        poll_interval=metadata_poll_interval,
    )

    batch_ids = {
        str(batch_info.get("batch_id") or "").strip() for batch_info in batch_infos
    }
    assert "" not in batch_ids
    assert len(batch_ids) == 1, (
        f"Requests were not dispatched in one batch: batch_infos={batch_infos}"
    )
    assert {batch_info.get("batch_size") for batch_info in batch_infos} == {
        expected_size
    }
    assert {batch_info.get("flush_reason") for batch_info in batch_infos} == {
        "max_batch_size"
    }


def _positive_execution_duration(execution, exec_id):
    raw_duration = execution.get("execution_time_seconds")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"Execution[{exec_id}] has invalid execution_time_seconds: {raw_duration!r}"
        ) from exc
    assert duration > 0, (
        f"Execution[{exec_id}] execution_time_seconds must be positive: {duration}"
    )
    return duration


def _result_gpu_stream(client, adapter, workflow_name, exec_id):
    execution = _result_execution(client, adapter, workflow_name, exec_id)
    gpu_stream = execution.get("gpu_stream")
    if not gpu_stream:
        raise AssertionError(
            f"Result for {exec_id} did not include execution.gpu_stream"
        )
    return gpu_stream


def _gpu_id_from_stream(gpu_stream):
    for part in reversed(gpu_stream.split(":")):
        if part.isdigit():
            return part
    raise AssertionError(f"Could not extract GPU id from gpu_stream={gpu_stream!r}")


def _parse_completed_at(value):
    if not isinstance(value, str) or not value.strip():
        raise AssertionError("execution.completed_at is required for overlap checks")

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    completed_at = datetime.fromisoformat(normalized)
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    return completed_at


def _execution_interval(execution):
    gpu_stream = execution.get("gpu_stream")
    if not gpu_stream:
        raise AssertionError("execution.gpu_stream is required for overlap checks")

    duration = execution.get("execution_time_seconds")
    if duration is None:
        raise AssertionError(
            f"execution_time_seconds is required for overlap checks: {execution}"
        )

    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"execution_time_seconds must be numeric, got {duration!r}"
        ) from exc
    if duration_seconds <= 0:
        raise AssertionError(
            f"execution_time_seconds must be positive, got {duration_seconds}"
        )

    end = _parse_completed_at(execution.get("completed_at"))
    start = end - timedelta(seconds=duration_seconds)
    return gpu_stream, start, end


def _assert_common_execution_overlap(executions):
    intervals = [_execution_interval(execution) for execution in executions]
    latest_start = max(start for _stream, start, _end in intervals)
    earliest_end = min(end for _stream, _start, end in intervals)
    interval_summary = [
        (stream, start.isoformat(), end.isoformat()) for stream, start, end in intervals
    ]

    assert latest_start < earliest_end, (
        "Workflow executions did not have a common overlap window; "
        f"intervals={interval_summary}"
    )
    logger.info(
        "Common execution overlap window: %s -> %s",
        latest_start.isoformat(),
        earliest_end.isoformat(),
    )


def test_deterministic_workflow(client, adapter):
    """Run a 10-step deterministic forecast (FCN, no plots)."""
    # NOTE: The v0.1.0 Rust plugin path completes successfully but does not
    # expose forecast_metadata.json in execution.outputs. Validation is
    # completion-only for now.
    test_params = {
        "model_type": "fcn",
        "data_source": "gfs",
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 10,
        "output_format": "zarr",
        "create_plots": False,
    }
    run_workflow_test(
        client,
        "deterministic_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


def test_diagnostic_workflow(client, adapter):
    """Run a diagnostic forecast (FCN prognostic + precipitation_afno, 9 steps)."""
    # NOTE: The v0.1.0 Rust plugin path completes successfully but does not
    # expose forecast_metadata.json in execution.outputs. Validation is
    # completion-only for now.
    test_params = {
        "data_source": "gfs",
        "forecast_times": ["2024-01-01T00:00:00"],
        "prognostic_model_type": "fcn",
        "diagnostic_model_type": "precipitation_afno",
        "nsteps": 9,
        "output_format": "zarr",
        "create_plots": False,
    }
    run_workflow_test(
        client,
        "diagnostic_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


STORMCAST_TIMEOUT = 600


def test_deterministic_fcn_workflow(client, adapter):
    """Run a 6-step deterministic FCN-only forecast (no model_type param)."""
    test_params = {
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 6,
        "data_source": "gfs",
        "output_format": "zarr",
        "create_plots": False,
    }
    run_workflow_test(
        client,
        "deterministic_fcn_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


def test_ensemble_workflow(client, adapter):
    """Run a small ensemble forecast (2 members, 4 steps) for CI speed."""
    test_params = {
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 4,
        "nensemble": 2,
        "model_type": "fcn",
        "data_source": "gfs",
        "output_format": "zarr",
        "create_plots": False,
    }
    run_workflow_test(
        client,
        "ensemble_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


def test_deterministic_earth2_workflow(client, adapter):
    """Run a 6-step deterministic forecast via the Earth2Workflow interface."""
    # NOTE: This workflow uses the older Earth2Workflow base class which does not
    # produce forecast_metadata.json. Validation is completion-only for now.
    # TODO: Add metadata writing to the Earth2Workflow base class in earth2studio/physicsnemo-serve.
    # start_time is intentionally a list: the workflow signature is
    # start_time: list[datetime] (unlike stormcast which takes a scalar).
    test_params = {
        "start_time": ["2024-01-01T00:00:00"],
        "num_steps": 6,
    }
    run_workflow_test(
        client,
        "deterministic_earth2_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


def test_stormcast_fcn3_workflow(client, adapter):
    """Run a StormCast + FCN3 two-stage forecast (6 hours)."""
    # NOTE: This workflow uses the older Earth2Workflow base class which does not
    # produce forecast_metadata.json. Validation is completion-only for now.
    # TODO: Add metadata writing to the Earth2Workflow base class in earth2studio/physicsnemo-serve.
    test_params = {
        "start_time": "2024-01-01T00:00:00",
        "num_hours": 6,
        "run_stormcast": True,
    }
    run_workflow_test(
        client,
        "stormcast_fcn3_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=STORMCAST_TIMEOUT,
    )


def test_example_user_workflow(client, adapter):
    """Run the lightweight example/template workflow to validate pipeline machinery."""
    test_params = {
        "task_name": "cicd_test",
        "num_iterations": 3,
        "delay_seconds": 0.1,
        "generate_output": True,
    }
    run_workflow_test(
        client,
        "example_user_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Rust/physicsnemo-serve-only workflows (no Python/earth2studio equivalent)
# ---------------------------------------------------------------------------


@pytest.mark.rust_only
def test_earth2_deterministic(client, adapter):
    """Run native Rust deterministic forecast (DLWP, 3 steps)."""
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
    }
    run_workflow_test(
        client,
        "earth2-deterministic",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


@pytest.mark.rust_only
def test_earth2_deterministic_batch(client, adapter):
    """Run native Rust batched deterministic forecast (DLWP, 3 steps)."""
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
    }
    run_workflow_test(
        client,
        "earth2-deterministic-batch",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


@pytest.mark.rust_only
def test_earth2_ensemble(client, adapter):
    """Run native Rust ensemble forecast (DLWP, 2 members, 3 steps)."""
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
        "nensemble": 2,
        "perturbation": "gaussian",
        "noise_amplitude": 0.05,
    }
    run_workflow_test(
        client,
        "earth2-ensemble",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=CICD_TIMEOUT,
    )


@pytest.mark.rust_only
def test_earth2_ensemble_fanout(client, adapter):
    """Run native Rust ensemble fanout pipeline (FCN, 4 members, batch_size=2, 3 steps)."""
    test_params = {
        "model": "fcn",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
        "nensemble": 4,
        "batch_size": 2,
        "perturbation": "spherical_gaussian",
        "noise_amplitude": 0.05,
    }
    run_workflow_test(
        client,
        "earth2-ensemble-fanout",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=STORMCAST_TIMEOUT,
    )


@pytest.mark.rust_only
def test_scheduler_batches_four_compatible_requests(client, adapter):
    """Verify four compatible requests are dispatched as one scheduler batch."""
    workflow_name = "earth2-deterministic"
    request_count = 4
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 1,
    }
    skip_if_workflow_disabled(adapter, workflow_name)

    exec_ids = _submit_workflows_concurrently(
        client,
        adapter,
        workflow_name,
        test_params,
        count=request_count,
    )
    assert len(set(exec_ids)) == request_count

    _wait_for_all_executions(
        client,
        adapter,
        workflow_name,
        exec_ids,
        timeout=MULTIGPU_TIMEOUT,
    )

    _assert_same_scheduler_batch(
        client,
        adapter,
        workflow_name,
        exec_ids,
        expected_size=request_count,
    )


@pytest.mark.rust_only
def test_batch_coordinator_executes_four_items_in_parallel(client, adapter):
    """Verify one scheduler batch executes four lightweight items concurrently."""
    workflow_name = "example_user_workflow"
    test_params = {
        "task_name": "parallel_batch_cicd_test",
        "num_iterations": 1,
        "delay_seconds": PARALLEL_BATCH_DELAY_SECONDS,
        "generate_output": False,
    }
    skip_if_workflow_disabled(adapter, workflow_name)

    started_at = time.monotonic()
    exec_ids = _submit_workflows_concurrently(
        client,
        adapter,
        workflow_name,
        test_params,
        count=PARALLEL_BATCH_REQUEST_COUNT,
    )
    assert len(set(exec_ids)) == PARALLEL_BATCH_REQUEST_COUNT

    _wait_for_all_executions(
        client,
        adapter,
        workflow_name,
        exec_ids,
        timeout=CICD_TIMEOUT,
        poll_interval=PARALLEL_BATCH_POLL_INTERVAL,
    )
    wall_time_seconds = time.monotonic() - started_at

    _assert_same_scheduler_batch(
        client,
        adapter,
        workflow_name,
        exec_ids,
        expected_size=PARALLEL_BATCH_REQUEST_COUNT,
    )
    item_durations = [
        _positive_execution_duration(
            _result_execution(client, adapter, workflow_name, exec_id), exec_id
        )
        for exec_id in exec_ids
    ]
    summed_item_seconds = sum(item_durations)
    maximum_parallel_wall_time = (
        summed_item_seconds * PARALLEL_BATCH_MAX_WALL_TO_ITEM_RATIO
    )

    assert wall_time_seconds < maximum_parallel_wall_time, (
        "Batch items did not demonstrate four-way execution overlap: "
        f"wall_time_seconds={wall_time_seconds:.3f}, "
        f"item_durations={item_durations}, "
        f"maximum_parallel_wall_time={maximum_parallel_wall_time:.3f}"
    )


@pytest.mark.rust_only
@pytest.mark.multigpu
@pytest.mark.parametrize("case_name, case", _selected_multigpu_workflow_cases())
def test_multigpu_concurrent_requests_use_all_gpu_streams(
    client, adapter, case_name, case
):
    """Submit one GPU workflow per visible GPU and verify all GPU streams complete work."""
    skip_if_workflow_disabled(adapter, case.workflow_name)

    gpu_ids = _endpoint_gpu_ids(client)
    expected_gpu_count = _configured_multigpu_gpu_count()

    if expected_gpu_count is not None:
        assert len(gpu_ids) == expected_gpu_count, (
            "endpoint GPU count did not match requested multi-GPU test count: "
            f"expected={expected_gpu_count}, observed={len(gpu_ids)}, gpu_ids={gpu_ids}"
        )
    if len(gpu_ids) < 2:
        pytest.skip(
            "endpoint does not report multiple GPUs in /metrics; "
            f"reported gpu_ids={gpu_ids}"
        )

    logger.info(
        "Submitting %d concurrent %s requests for case=%s and GPU ids: %s",
        len(gpu_ids),
        case.workflow_name,
        case_name,
        gpu_ids,
    )
    exec_ids = _submit_workflows_concurrently(
        client,
        adapter,
        case.workflow_name,
        case.params,
        count=len(gpu_ids),
    )
    logger.info("Submitted concurrent executions: %s", exec_ids)

    _wait_for_all_executions(
        client,
        adapter,
        case.workflow_name,
        exec_ids,
        timeout=case.timeout,
    )

    executions = [
        _result_execution(client, adapter, case.workflow_name, exec_id)
        for exec_id in exec_ids
    ]
    gpu_streams = []
    for exec_id, execution in zip(exec_ids, executions, strict=True):
        gpu_stream = execution.get("gpu_stream")
        if not gpu_stream:
            raise AssertionError(
                f"Result for {exec_id} did not include execution.gpu_stream"
            )
        gpu_streams.append(gpu_stream)
    observed_gpu_ids = sorted({_gpu_id_from_stream(stream) for stream in gpu_streams})
    logger.info("Observed GPU streams: %s", gpu_streams)

    assert observed_gpu_ids == gpu_ids, (
        "Concurrent workflow executions did not cover every visible GPU: "
        f"expected gpu_ids={gpu_ids}, observed gpu_ids={observed_gpu_ids}, "
        f"gpu_streams={gpu_streams}"
    )

    if case.require_overlap:
        _assert_common_execution_overlap(executions)


# TODO: Add PhysicsNeMo Serve plugins for the two earth2studio-only workflows
# (foundry_fcn3 and foundry_fcn3_stormscope_goes), then add corresponding
# test cases here once available.
