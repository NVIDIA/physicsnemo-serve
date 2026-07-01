# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import pytest

from helpers import (
    run_workflow_test,
    get_execution_status,
    RequestStatus,
    skip_if_workflow_disabled,
)


logger = logging.getLogger(__name__)


def test_stability(client, adapter):
    skip_if_workflow_disabled(adapter, "diagnostic_workflow")
    success = 0
    failed = 0
    for i in range(10):
        test_params = {
            "data_source": "gfs",
            "forecast_times": ["2024-01-01T00:00:00"],
            "nsteps": 1,
            "output_format": "zarr",
            "create_plots": True,
            "plot_step": 1,
            "plot_variable": "tp",
        }
        try:
            run_workflow_test(
                client, "diagnostic_workflow", test_params, adapter=adapter
            )
            logger.info(f"{i}: SUCCESS")
            success += 1
        except Exception as e:
            logger.exception(e)
            logger.info(f"{i}: FAILED")
            failed += 1
    logger.info(f"{success=}, {failed=}")
    assert failed == 0


@pytest.mark.parametrize("workflow_name", ["diagnostic_workflow"])
def test_many_executions(client, adapter, workflow_name):
    skip_if_workflow_disabled(adapter, workflow_name)
    params = {"forecast_times": ["2024-01-01T00:00:00"]}
    body = adapter.format_submit_body(params)

    executions = []
    for i in range(5):
        logger.info(f"Submitting a job #{i} with payload: {body}")
        submit_url = client.base_url + adapter.submit_url(workflow_name)
        resp = client.post(submit_url, json=body)
        resp.raise_for_status()
        parsed = adapter.parse_submit_response(resp.json())
        assert parsed.get("workflow_name") == workflow_name
        assert parsed.get("status") in [
            RequestStatus.QUEUED.value,
            RequestStatus.ACCEPTED.value,
        ]
        exec_id = parsed["execution_id"]
        executions.append(exec_id)

    timeout = 1200
    wait_till = time.time() + timeout
    while time.time() < wait_till:
        running = 0
        for exec_id in executions:
            status_details = get_execution_status(
                client, workflow_name, exec_id, adapter=adapter
            )
            if status_details["status"] != "completed":
                running += 1
        logger.info("=" * 80)
        if running == 0:
            break
        time.sleep(10)
