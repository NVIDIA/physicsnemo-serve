# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from helpers import run_workflow_test, WORKFLOWS, QUICK_VALIDATION


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.basic


def test_health(client, adapter):
    resp = client.get(client.base_url + adapter.health_url())
    resp.raise_for_status()
    data = adapter.parse_health_response(resp)
    assert data.get("status") == "healthy"


def test_list_workflows(client, adapter):
    resp = client.get(client.base_url + adapter.list_workflows_url())
    resp.raise_for_status()
    workflows = adapter.parse_list_workflows_response(resp.json())
    for wf in workflows:
        resp = client.get(client.base_url + adapter.workflow_schema_url(wf))
        resp.raise_for_status()


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_run_workflow_defaults(client, adapter, workflow):
    run_workflow_test(
        client,
        workflow,
        {"forecast_times": ["2024-01-01T00:00:00"]},
        adapter=adapter,
    )


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_run_workflow_multiple_timestamps(client, adapter, workflow):
    run_workflow_test(
        client,
        workflow,
        {
            "forecast_times": [
                "2024-01-01T00:00:00",
                "2024-02-01T00:00:00",
                "2026-01-01T00:00:00",
            ]
        },
        adapter=adapter,
    )


@pytest.mark.parametrize(
    "forecast_times",
    [
        ["2024-01-01T00:00:00"],
    ],
)
@pytest.mark.parametrize("model_type", ["fcn", "dlwp"])
@pytest.mark.parametrize("data_source", ["gfs"])
@pytest.mark.parametrize("nsteps", [1, 10])
@pytest.mark.parametrize("output_format", ["zarr"])
@pytest.mark.parametrize("create_plots", [True, False])
@pytest.mark.parametrize("plot_step", [1, 999])
@pytest.mark.parametrize("plot_variable", ["t2m", "tcwv", "z500"])
def test_deterministic_workflow(
    client,
    adapter,
    forecast_times,
    model_type,
    data_source,
    nsteps,
    output_format,
    create_plots,
    plot_step,
    plot_variable,
):
    test_params = {
        "model_type": model_type,
        "data_source": data_source,
        "forecast_times": forecast_times,
        "nsteps": nsteps,
        "output_format": output_format,
        "create_plots": create_plots,
        "plot_step": plot_step,
        "plot_variable": plot_variable,
    }
    run_workflow_test(
        client,
        "deterministic_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
    )


@pytest.mark.parametrize(
    "forecast_times",
    [
        ["2024-01-01T00:00:00"],
    ],
)
@pytest.mark.parametrize("nsteps", [1, 10])
@pytest.mark.parametrize("data_source", ["gfs"])
@pytest.mark.parametrize("output_format", ["zarr"])
@pytest.mark.parametrize("create_plots", [True, False])
@pytest.mark.parametrize(
    "plot_variable", ["t2m", "msl", "u10m", "v10m", "tcwv", "z500"]
)
@pytest.mark.parametrize("plot_step", [1, 999])
def test_deterministic_fcn_workflow(
    client,
    adapter,
    forecast_times,
    nsteps,
    data_source,
    output_format,
    create_plots,
    plot_variable,
    plot_step,
):
    test_params = {
        "data_source": data_source,
        "forecast_times": forecast_times,
        "nsteps": nsteps,
        "output_format": output_format,
        "create_plots": create_plots,
        "plot_step": plot_step,
        "plot_variable": plot_variable,
    }
    run_workflow_test(
        client,
        "deterministic_fcn_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
    )


@pytest.mark.parametrize(
    "forecast_times",
    [
        ["2024-01-01T00:00:00"],
    ],
)
@pytest.mark.parametrize("nsteps", [1, 9])
@pytest.mark.parametrize("prognostic_model_type", ["fcn"])
@pytest.mark.parametrize("diagnostic_model_type", ["precipitation_afno"])
@pytest.mark.parametrize("data_source", ["gfs"])
@pytest.mark.parametrize("output_format", ["zarr"])
@pytest.mark.parametrize("create_plots", [True, False])
@pytest.mark.parametrize("plot_variable", ["tp"])
@pytest.mark.parametrize("plot_step", [1, 999])
def test_diagnostic_workflow(
    client,
    adapter,
    forecast_times,
    nsteps,
    prognostic_model_type,
    diagnostic_model_type,
    data_source,
    output_format,
    create_plots,
    plot_variable,
    plot_step,
):
    test_params = {
        "data_source": data_source,
        "forecast_times": forecast_times,
        "prognostic_model_type": prognostic_model_type,
        "diagnostic_model_type": diagnostic_model_type,
        "nsteps": nsteps,
        "output_format": output_format,
        "create_plots": create_plots,
        "plot_step": plot_step,
        "plot_variable": plot_variable,
    }
    run_workflow_test(
        client,
        "diagnostic_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
    )


# ---------------------------------------------------------------------------
# Ensemble workflow (e2s-ensemble)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forecast_times",
    [
        ["2024-01-01T00:00:00"],
        ["2024-06-15T00:00:00"],
    ],
)
@pytest.mark.parametrize("nsteps", [1, 4])
@pytest.mark.parametrize("nensemble", [2, 4])
@pytest.mark.parametrize("create_plots", [True, False])
def test_ensemble_workflow(
    client,
    adapter,
    forecast_times,
    nsteps,
    nensemble,
    create_plots,
):
    test_params = {
        "forecast_times": forecast_times,
        "nsteps": nsteps,
        "nensemble": nensemble,
        "noise_amplitude": 0.05,
        "model_type": "fcn",
        "data_source": "gfs",
        "output_format": "zarr",
        "create_plots": create_plots,
        "plot_variable": "t2m",
    }
    run_workflow_test(
        client,
        "ensemble_workflow",
        test_params,
        validate_output_files=QUICK_VALIDATION,
        adapter=adapter,
    )


# ---------------------------------------------------------------------------
# Deterministic Earth2 workflow (e2s-deterministic-earth2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start_time",
    [
        ["2024-01-01T00:00:00"],
        ["2024-03-20T00:00:00"],
        ["2024-06-15T12:00:00"],
    ],
)
@pytest.mark.parametrize("num_steps", [1, 6, 10])
def test_deterministic_earth2_workflow(
    client,
    adapter,
    start_time,
    num_steps,
):
    test_params = {
        "start_time": start_time,
        "num_steps": num_steps,
    }
    run_workflow_test(
        client,
        "deterministic_earth2_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
    )


# ---------------------------------------------------------------------------
# StormCast + FCN3 workflow (e2s-stormcast-fcn3)
# ---------------------------------------------------------------------------


STORMCAST_TIMEOUT = 600


@pytest.mark.parametrize("start_time", ["2024-01-01T00:00:00", "2024-07-01T00:00:00"])
@pytest.mark.parametrize("num_hours", [6, 12])
@pytest.mark.parametrize("run_stormcast", [True, False])
def test_stormcast_fcn3_workflow(
    client,
    adapter,
    start_time,
    num_hours,
    run_stormcast,
):
    test_params = {
        "start_time": start_time,
        "num_hours": num_hours,
        "run_stormcast": run_stormcast,
    }
    run_workflow_test(
        client,
        "stormcast_fcn3_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=STORMCAST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Example user workflow (e2s-example-user)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", ["qa_basic", "qa test with spaces"])
@pytest.mark.parametrize("num_iterations", [1, 5, 10])
@pytest.mark.parametrize("delay_seconds", [0.0, 0.1])
@pytest.mark.parametrize("generate_output", [True, False])
def test_example_user_workflow(
    client,
    adapter,
    task_name,
    num_iterations,
    delay_seconds,
    generate_output,
):
    test_params = {
        "task_name": task_name,
        "num_iterations": num_iterations,
        "delay_seconds": delay_seconds,
        "generate_output": generate_output,
    }
    run_workflow_test(
        client,
        "example_user_workflow",
        test_params,
        validate_output_files=False,
        adapter=adapter,
    )


# ---------------------------------------------------------------------------
# Rust/physicsnemo-serve-only workflows (no Python/earth2studio equivalent)
# ---------------------------------------------------------------------------


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "start_time",
    ["2024-01-01T00:00:00", "2024-03-20T00:00:00", "2024-06-15T00:00:00"],
)
@pytest.mark.parametrize("nsteps", [1, 3, 6, 10])
def test_earth2_deterministic(
    client,
    adapter,
    start_time,
    nsteps,
):
    test_params = {
        "model": "dlwp",
        "start_time": start_time,
        "nsteps": nsteps,
    }
    run_workflow_test(
        client,
        "earth2-deterministic",
        test_params,
        validate_output_files=False,
        adapter=adapter,
    )


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "start_time",
    ["2024-01-01T00:00:00", "2024-03-20T00:00:00", "2024-06-15T00:00:00"],
)
@pytest.mark.parametrize("nsteps", [1, 3, 10])
def test_earth2_deterministic_batch(
    client,
    adapter,
    start_time,
    nsteps,
):
    test_params = {
        "model": "dlwp",
        "start_time": start_time,
        "nsteps": nsteps,
    }
    run_workflow_test(
        client,
        "earth2-deterministic-batch",
        test_params,
        validate_output_files=False,
        adapter=adapter,
    )


@pytest.mark.rust_only
@pytest.mark.parametrize("nsteps", [1, 3])
@pytest.mark.parametrize("nensemble", [2, 4])
@pytest.mark.parametrize(
    "perturbation,noise_amplitude",
    [
        ("gaussian", 0.05),
        ("brown", 0.02),
    ],
)
def test_earth2_ensemble(
    client,
    adapter,
    nsteps,
    nensemble,
    perturbation,
    noise_amplitude,
):
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": nsteps,
        "nensemble": nensemble,
        "perturbation": perturbation,
        "noise_amplitude": noise_amplitude,
    }
    run_workflow_test(
        client,
        "earth2-ensemble",
        test_params,
        validate_output_files=False,
        adapter=adapter,
    )


@pytest.mark.rust_only
@pytest.mark.parametrize("nsteps", [1, 3])
@pytest.mark.parametrize("nensemble", [4, 6])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize(
    "perturbation,noise_amplitude",
    [
        ("spherical_gaussian", 0.05),
    ],
)
def test_earth2_ensemble_fanout(
    client,
    adapter,
    nsteps,
    nensemble,
    batch_size,
    perturbation,
    noise_amplitude,
):
    test_params = {
        "model": "fcn",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": nsteps,
        "nensemble": nensemble,
        "batch_size": batch_size,
        "perturbation": perturbation,
        "noise_amplitude": noise_amplitude,
    }
    run_workflow_test(
        client,
        "earth2-ensemble-fanout",
        test_params,
        validate_output_files=False,
        adapter=adapter,
        timeout=STORMCAST_TIMEOUT,
    )
