# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Lepton CRPS comparison QA orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from service_adapter import ServiceAdapter


TERMINAL_STATUSES = {"completed", "pending_results", "failed", "cancelled"}
SUCCESS_STATUSES = {"completed", "pending_results"}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RESULTS_READY_TIMEOUT_SECONDS = 1800
RESULTS_READY_POLL_INTERVAL_SECONDS = 30


@dataclass(frozen=True)
class WorkflowRun:
    """Normalized endpoint execution details plus raw payload artifacts."""

    label: str
    workflow: str
    execution_id: str
    submit_payload: dict[str, Any]
    final_status_payload: dict[str, Any]
    results_payload: dict[str, Any]
    forecast_zarr_path: str


def load_request_payload(path: Path) -> dict[str, Any]:
    """Load one request JSON object to submit to both comparison endpoints."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"request payload must be a JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, reviewable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def make_client(endpoint_token: str) -> Any:
    """Create a retrying requests session for inference endpoints."""

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {endpoint_token}"
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def endpoint_url(base_url: str, path: str) -> str:
    """Join an endpoint base URL with an adapter path."""

    return base_url.rstrip("/") + path


def submit_workflow(
    *,
    client: Any,
    base_url: str,
    adapter: ServiceAdapter,
    workflow: str,
    request_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Submit one workflow request and return its execution id and raw response."""

    url = endpoint_url(base_url, adapter.submit_url(workflow))
    body = adapter.format_submit_body(request_payload)
    response = client.post(url, json=body, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    parsed = adapter.parse_submit_response(payload)
    execution_id = parsed.get("execution_id")
    if not execution_id:
        raise RuntimeError(
            f"submit response did not include an execution id: {payload}"
        )
    return execution_id, payload


def poll_workflow(
    *,
    client: Any,
    base_url: str,
    adapter: ServiceAdapter,
    workflow: str,
    execution_id: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> dict[str, Any]:
    """Poll workflow status until a terminal state or timeout."""

    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        url = endpoint_url(base_url, adapter.status_url(workflow, execution_id))
        response = client.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        parsed = adapter.parse_status_response(payload)
        status = parsed.get("status")
        print(f"    {workflow}/{execution_id}: {status}", flush=True)
        if status in TERMINAL_STATUSES:
            if status not in SUCCESS_STATUSES:
                raise RuntimeError(
                    f"execution {execution_id} ended with status {status}: {payload}"
                )
            return payload
        time.sleep(interval_seconds)

    raise TimeoutError(
        f"execution {execution_id} did not finish within {timeout_seconds}s; "
        f"last status payload: {last_payload}"
    )


def fetch_results(
    *,
    client: Any,
    base_url: str,
    adapter: ServiceAdapter,
    workflow: str,
    execution_id: str,
) -> dict[str, Any]:
    """Fetch one workflow results payload."""

    url = endpoint_url(base_url, adapter.results_url(workflow, execution_id))
    response = client.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_results_with_zarr_path(
    *,
    client: Any,
    base_url: str,
    adapter: ServiceAdapter,
    workflow: str,
    execution_id: str,
    artifact_dir: Path,
    label: str,
    mount_target: str,
    timeout_seconds: int = RESULTS_READY_TIMEOUT_SECONDS,
    interval_seconds: int = RESULTS_READY_POLL_INTERVAL_SECONDS,
) -> tuple[dict[str, Any], str]:
    """Fetch results until the Zarr manifest/path is available."""

    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            results_payload = fetch_results(
                client=client,
                base_url=base_url,
                adapter=adapter,
                workflow=workflow,
                execution_id=execution_id,
            )
        except Exception as exc:
            if _http_status_code(exc) != 404:
                raise
            last_error = exc
            print(
                f"    {workflow}/{execution_id}: results not ready "
                f"(attempt {attempt}): {exc}",
                flush=True,
            )
            time.sleep(interval_seconds)
            continue
        last_payload = results_payload
        write_json(artifact_dir / f"{label}-results.json", results_payload)
        try:
            forecast_zarr_path = extract_forecast_zarr_path(
                results_payload,
                mount_target=mount_target,
            )
        except RuntimeError as exc:
            last_error = exc
            print(
                f"    {workflow}/{execution_id}: results not ready "
                f"(attempt {attempt}): {exc}",
                flush=True,
            )
            time.sleep(interval_seconds)
            continue
        return results_payload, forecast_zarr_path

    write_json(
        artifact_dir / f"{label}-results-last.json",
        last_payload or {"error": "no results payload fetched"},
    )
    raise TimeoutError(
        f"results for {execution_id} did not expose a forecast Zarr path within "
        f"{timeout_seconds}s; last error: {last_error}; last payload: {last_payload}"
    )


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def run_workflow_and_fetch_results(
    *,
    label: str,
    client: Any,
    base_url: str,
    adapter: ServiceAdapter,
    workflow: str,
    request_payload: dict[str, Any],
    artifact_dir: Path,
    mount_target: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> WorkflowRun:
    """Submit, wait for, fetch, and artifact one endpoint workflow run."""

    print(f"==> Submitting {label} workflow: {workflow}", flush=True)
    execution_id, submit_payload = submit_workflow(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        request_payload=request_payload,
    )
    write_json(artifact_dir / f"{label}-submit.json", submit_payload)

    print(f"==> Waiting for {label} execution: {execution_id}", flush=True)
    final_status_payload = poll_workflow(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        execution_id=execution_id,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )
    write_json(artifact_dir / f"{label}-status-final.json", final_status_payload)

    print(f"==> Fetching {label} results", flush=True)
    results_payload, forecast_zarr_path = fetch_results_with_zarr_path(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        execution_id=execution_id,
        artifact_dir=artifact_dir,
        label=label,
        mount_target=mount_target,
    )

    return WorkflowRun(
        label=label,
        workflow=workflow,
        execution_id=execution_id,
        submit_payload=submit_payload,
        final_status_payload=final_status_payload,
        results_payload=results_payload,
        forecast_zarr_path=forecast_zarr_path,
    )


def extract_forecast_zarr_path(
    results_payload: dict[str, Any],
    *,
    mount_target: str,
) -> str:
    """Extract and validate the primary Lustre-backed forecast Zarr path."""

    candidates = list(_candidate_zarr_paths(results_payload, mount_target=mount_target))
    for path in candidates:
        if _valid_forecast_path(path, mount_target=mount_target):
            return path

    candidate_preview = ", ".join(candidates) if candidates else "<none>"
    raise RuntimeError(
        "could not find a valid forecast Zarr path in results payload; "
        f"candidates: {candidate_preview}"
    )


def _candidate_zarr_paths(
    results_payload: dict[str, Any], *, mount_target: str
) -> list[str]:
    execution = results_payload.get("execution")
    if isinstance(execution, dict):
        outputs = execution.get("outputs")
        if isinstance(outputs, list):
            primary_storage = [
                _normalize_zarr_path(output.get("storage_path"), mount_target)
                for output in outputs
                if isinstance(output, dict) and output.get("primary")
            ]
            storage_paths = [
                _normalize_zarr_path(output.get("storage_path"), mount_target)
                for output in outputs
                if isinstance(output, dict)
            ]
            paths = [
                _normalize_zarr_path(output.get("path"), mount_target)
                for output in outputs
                if isinstance(output, dict)
            ]
            return _dedupe_paths([*primary_storage, *storage_paths, *paths])

    output_files = results_payload.get("output_files")
    if isinstance(output_files, list):
        output_paths = []
        workflow_name = results_payload.get("workflow_name")
        for output in output_files:
            if not isinstance(output, dict):
                continue
            output_paths.append(output.get("storage_path"))
            output_path = output.get("path")
            if (
                isinstance(workflow_name, str)
                and isinstance(output_path, str)
                and not output_path.startswith("/")
            ):
                output_paths.append(f"{workflow_name}/{output_path}")
            output_paths.append(output_path)
        paths = _dedupe_paths(
            _normalize_zarr_path(path, mount_target) for path in output_paths
        )
        if paths:
            return paths

    payload = results_payload.get("payload")
    if isinstance(payload, dict):
        path = _normalize_zarr_path(payload.get("dataset_path"), mount_target)
        if path:
            return [path]

    path = _normalize_zarr_path(results_payload.get("dataset_path"), mount_target)
    if path:
        return [path]

    return []


def _dedupe_paths(paths: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not isinstance(path, str) or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _normalize_zarr_path(path: Any, mount_target: str) -> str | None:
    """Return an absolute Zarr store path from direct or nested result entries."""

    if not isinstance(path, str):
        return None

    stripped = path.rstrip("/")
    if stripped.endswith(".zarr"):
        zarr_path = stripped
    else:
        marker = ".zarr/"
        marker_idx = stripped.find(marker)
        if marker_idx < 0:
            return None
        zarr_path = stripped[: marker_idx + len(".zarr")]

    if zarr_path.startswith("/"):
        return zarr_path
    return f"{mount_target.rstrip('/')}/{zarr_path.lstrip('/')}"


def _is_zarr_path(path: Any) -> bool:
    return isinstance(path, str) and path.rstrip("/").endswith(".zarr")


def _valid_forecast_path(path: str, *, mount_target: str) -> bool:
    normalized_mount = mount_target.rstrip("/")
    return (
        path.startswith("/")
        and path.rstrip("/").endswith(".zarr")
        and (path == normalized_mount or path.startswith(normalized_mount + "/"))
    )
