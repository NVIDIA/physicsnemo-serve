# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import time
import enum
import xarray
import logging
import zipfile

import pytest


logger = logging.getLogger(__name__)


WORKFLOWS = [
    "deterministic_fcn_workflow",
    "deterministic_workflow",
    "diagnostic_workflow",
]

QUICK_VALIDATION = "quick"
FULL_VALIDATION = "full"
VALIDATE_OUTPUT_FILES = False
PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID_ENV = "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"


class RequestStatus(enum.Enum):
    """Status values for inference requests."""

    # Initial statuses
    ACCEPTED = "accepted"
    QUEUED = "queued"

    # Processing statuses
    RUNNING = "running"

    # Final statuses
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def enabled_physicsnemo_serve_plugin_id():
    value = os.environ.get(PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID_ENV, "").strip()
    return value or None


def physicsnemo_serve_plugin_id_for_workflow(adapter, workflow_name):
    resolver = getattr(adapter, "_resolve_workflow", None)
    if not callable(resolver):
        return None
    plugin_id = resolver(workflow_name)
    return str(plugin_id).strip() if plugin_id else None


def skip_if_workflow_disabled(adapter, workflow_name):
    enabled_plugin_id = enabled_physicsnemo_serve_plugin_id()
    if enabled_plugin_id is None:
        return

    plugin_id = physicsnemo_serve_plugin_id_for_workflow(adapter, workflow_name)
    if plugin_id is None:
        return
    if plugin_id == enabled_plugin_id:
        return

    pytest.skip(
        f"{PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID_ENV}={enabled_plugin_id!r} only enables "
        f"plugin {enabled_plugin_id!r}; workflow {workflow_name!r} resolves to "
        f"{plugin_id!r}"
    )


def get_execution_status(client, workflow, exec_id, adapter=None):
    if adapter is None:
        from service_adapter import PythonAdapter

        adapter = PythonAdapter()

    url = client.base_url + adapter.status_url(workflow, exec_id)
    resp = client.get(url)
    resp.raise_for_status()
    parsed = adapter.parse_status_response(resp.json())

    progress = parsed.get("progress")
    position = parsed.get("position")
    if progress:
        cur_step = progress.get("current_step", "-")
        total_steps = progress.get("total_steps", "-")
        progress = f"{cur_step}/{total_steps}"
    else:
        progress = "n/a"
    status = parsed["status"]
    duration = parsed.get("execution_time_seconds")
    duration = f"{int(float(duration))}s" if duration else None
    queue_position = f"[{position}]" if position else ""
    logger.info(
        f"EXECUTION[{exec_id}]: ("
        f"status: {status}{queue_position}, "
        f"step: {progress}, "
        f"duration: {duration})"
    )
    return {
        "execution_id": exec_id,
        "status": status,
        "queue_position": queue_position,
        "step": progress,
        "duration": duration,
    }


def watch_execution(
    client, workflow, exec_id, timeout=900, check_every=10, adapter=None
):
    wait_till = time.time() + timeout
    while time.time() < wait_till:
        status_details = get_execution_status(
            client, workflow, exec_id, adapter=adapter
        )
        if status_details["status"] in [
            RequestStatus.FAILED.value,
            RequestStatus.COMPLETED.value,
            RequestStatus.CANCELLED.value,
        ]:
            return status_details
        time.sleep(check_every)
    raise RuntimeError(f"Watch execution timed out after {timeout} seconds")


def validate_metadata_file(local_file, test_params):
    with open(local_file, "r") as f:
        used_params = json.load(f)

    common_keys = set(used_params).intersection(test_params)
    for key in common_keys:
        if isinstance(used_params[key], str):
            assert used_params[key].lower() == test_params[key].lower()
        else:
            assert used_params[key] == test_params[key]
    return used_params


def validate_zarr_file(local_file, test_params):
    dataset = xarray.load_dataset(local_file)
    # assert len(dataset["tp"].coords["time"]) == len(test_params["forecast_times"])
    if "forecast_times" in test_params:
        assert len(dataset["time"]) == len(test_params["forecast_times"])


def validate_results(
    client, workflow_name, exec_id, test_params, validation, adapter=None
):
    if adapter is None:
        from service_adapter import PythonAdapter

        adapter = PythonAdapter()

    logger.info(f"Validating results of {exec_id}")
    files_dir = pathlib.Path().cwd().joinpath("downloads")
    files_dir.mkdir(exist_ok=True, parents=True)

    results_url = client.base_url + adapter.results_url(workflow_name, exec_id)
    results_response = client.get(results_url)
    results_response.raise_for_status()
    parsed = adapter.parse_results_response(results_response.json())
    output_files = parsed.get("output_files", [])

    logger.info("Validating metadata file")
    target_file = "forecast_metadata.json"
    try:
        metadata_file = [f for f in output_files if f["path"].endswith(target_file)][0]
    except IndexError:
        raise RuntimeError(f"No {target_file} file found")
    metadata_file_path = metadata_file["path"]
    file_url = client.base_url + adapter.result_file_url(
        workflow_name, exec_id, metadata_file_path
    )
    file_response = client.get(file_url)
    local_file_name = f"{int(time.time())}-{metadata_file_path.split('/')[-1]}"
    local_file = files_dir.joinpath(local_file_name)
    logger.info(f"Saving '{file_response.url}' to '{local_file}'")
    with open(local_file, "wb+") as f:
        f.write(file_response.content)
    validate_metadata_file(local_file, test_params)

    if validation != FULL_VALIDATION:
        return

    # Find zarr: prefer primary output (Rust), fall back to path prefix (Python)
    zarr_candidates = [f for f in output_files if f.get("primary")]
    if not zarr_candidates:
        zarr_candidates = [
            f for f in output_files if f["path"].startswith(workflow_name)
        ]
    if not zarr_candidates:
        raise RuntimeError("No zarr file found")
    zarr_file = zarr_candidates[0]
    zarr_file_path = zarr_file["path"]
    logger.info("Downloading zarr file...")
    file_url = client.base_url + adapter.result_file_url(
        workflow_name, exec_id, zarr_file_path
    )
    file_response = client.get(file_url)
    local_file = files_dir.joinpath(f"{exec_id}-results.zip")
    logger.info(f"Saving '{file_response.url}' to '{local_file}'")
    with open(local_file, "wb+") as f:
        f.write(file_response.content)

    extract_to = local_file.parent.joinpath(f"{exec_id}-results")
    with zipfile.ZipFile(local_file, "r") as zip_file:
        zip_file.extractall(extract_to)
    results_file = extract_to.joinpath(f"{exec_id}/results.zarr")
    logger.info("Validating zarr file")
    validate_zarr_file(results_file, test_params)


def run_workflow_test(
    client,
    workflow_name,
    test_params,
    validate_output_files=None,
    adapter=None,
    timeout=900,
):
    if adapter is None:
        from service_adapter import PythonAdapter

        adapter = PythonAdapter()

    if validate_output_files is None:
        global VALIDATE_OUTPUT_FILES
        validate_output_files = VALIDATE_OUTPUT_FILES

    skip_if_workflow_disabled(adapter, workflow_name)

    logger.info(f"Submitting a job with params: {test_params}")
    submit_url = client.base_url + adapter.submit_url(workflow_name)
    body = adapter.format_submit_body(test_params)
    resp = client.post(submit_url, json=body)
    resp.raise_for_status()
    parsed = adapter.parse_submit_response(resp.json())
    assert parsed.get("workflow_name") == workflow_name
    assert parsed.get("status") in [
        RequestStatus.QUEUED.value,
        RequestStatus.ACCEPTED.value,
    ]
    exec_id = parsed["execution_id"]

    result = watch_execution(
        client, workflow_name, exec_id, timeout=timeout, adapter=adapter
    )
    if result["status"] != RequestStatus.COMPLETED.value:
        raise RuntimeError(f"Execution[{exec_id}] failed.")

    if validate_output_files:
        validate_results(
            client,
            workflow_name,
            exec_id,
            test_params,
            validate_output_files,
            adapter=adapter,
        )
