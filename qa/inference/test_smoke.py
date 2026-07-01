# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test — fast sanity check for the test harness and service connectivity.

Run with:  pytest -m smoke --service python
"""

import pytest
import logging

from helpers import run_workflow_test, QUICK_VALIDATION


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.smoke

SMOKE_TIMEOUT = 300


def test_health(client, adapter):
    """Verify the service is reachable and reports healthy."""
    resp = client.get(client.base_url + adapter.health_url())
    resp.raise_for_status()
    data = adapter.parse_health_response(resp)
    assert data["status"] == "healthy", f"Service not healthy: {data}"


def test_list_workflows(client, adapter):
    """Verify we can list workflows and fetch at least one schema."""
    resp = client.get(client.base_url + adapter.list_workflows_url())
    resp.raise_for_status()
    workflows = adapter.parse_list_workflows_response(resp.json())
    assert len(workflows) > 0, "No workflows returned"
    logger.info(f"Available workflows: {workflows}")

    resp = client.get(client.base_url + adapter.workflow_schema_url(workflows[0]))
    resp.raise_for_status()


def test_deterministic_workflow(client, adapter):
    """Submit a single deterministic workflow run and verify completion."""
    test_params = {
        "model_type": "fcn",
        "data_source": "gfs",
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 1,
        "output_format": "zarr",
        "create_plots": False,
    }
    run_workflow_test(
        client,
        "deterministic_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
        timeout=SMOKE_TIMEOUT,
    )
