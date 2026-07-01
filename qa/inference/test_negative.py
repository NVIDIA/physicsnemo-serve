# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from helpers import run_workflow_test, WORKFLOWS

pytestmark = pytest.mark.negative


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize(
    "invalid_params",
    [
        {"forecast_times": []},
        {"forecast_times": 1},
        {"nsteps": -1},
        {"nsteps": 1000},
        {"nsteps": "some_steps"},
        {"plot_step": -1},
        {"create_plots": "x"},
    ],
)
def test_workflow_negative(client, adapter, invalid_params, workflow):
    test_params = {"forecast_times": ["2024-01-01T00:00:00"]}
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, workflow, test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"model_type": "no_such_model"},
        {"data_source": "no_such_data_source"},
        {"forecast_times": ["no_such_forecast_time"]},
        {"output_format": "no_such_output_format"},
        {"plot_variable": "no_such_variable"},
    ],
)
def test_deterministic_workflow_negative(client, adapter, invalid_params):
    test_params = {"forecast_times": ["2024-01-01T00:00:00"]}
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "deterministic_workflow", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"data_source": "no_such_data_source"},
        {"forecast_times": ["no_such_forecast_time"]},
        {"output_format": "no_such_output_format"},
        {"plot_variable": "no_such_variable"},
    ],
)
def test_deterministic_fcn_workflow_negative(client, adapter, invalid_params):
    test_params = {"forecast_times": ["2024-01-01T00:00:00"]}
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "deterministic_fcn_workflow", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"diagnostic_model_type": "no_such_model"},
        {"data_source": "no_such_data_source"},
        {"forecast_times": ["no_such_forecast_time"]},
        {"output_format": "no_such_output_format"},
        {"plot_variable": "no_such_variable"},
    ],
)
def test_diagnostic_workflow_negative(client, adapter, invalid_params):
    test_params = {"forecast_times": ["2024-01-01T00:00:00"]}
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, "diagnostic_workflow", test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"nensemble": 0},
        {"nensemble": -1},
        {"nensemble": "many"},
        {"model_type": "no_such_model"},
        {"noise_amplitude": 0},
        {"noise_amplitude": -0.5},
        {"noise_amplitude": "loud"},
    ],
)
def test_ensemble_workflow_negative(client, adapter, invalid_params):
    test_params = {
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 4,
        "nensemble": 2,
        "model_type": "fcn",
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, "ensemble_workflow", test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"start_time": "not-a-datetime"},
        {"start_time": []},
        {"num_steps": 0},
        {"num_steps": -1},
        {"num_steps": "six"},
    ],
)
def test_deterministic_earth2_workflow_negative(client, adapter, invalid_params):
    test_params = {
        "start_time": ["2024-01-01T00:00:00"],
        "num_steps": 6,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "deterministic_earth2_workflow", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"start_time": "not-a-datetime"},
        {"num_hours": 0},
        {"num_hours": -1},
        {"num_hours": "twelve"},
        {"run_stormcast": "maybe"},
    ],
)
def test_stormcast_fcn3_workflow_negative(client, adapter, invalid_params):
    test_params = {
        "start_time": "2024-01-01T00:00:00",
        "num_hours": 6,
        "run_stormcast": True,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "stormcast_fcn3_workflow", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.parametrize(
    "invalid_params",
    [
        {"num_iterations": 0},
        {"num_iterations": -1},
        {"num_iterations": "five"},
        {"delay_seconds": -1.0},
        {"delay_seconds": "slow"},
    ],
)
def test_example_user_workflow_negative(client, adapter, invalid_params):
    test_params = {
        "task_name": "negative_test",
        "num_iterations": 3,
        "delay_seconds": 0.1,
        "generate_output": True,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, "example_user_workflow", test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


# ---------------------------------------------------------------------------
# Rust/physicsnemo-serve-only negative tests
# ---------------------------------------------------------------------------


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "invalid_params",
    [
        {"model": "no_such_model"},
        {"start_time": "not-a-datetime"},
        {"start_time": ""},
        {"nsteps": 0},
        {"nsteps": -1},
    ],
)
def test_earth2_deterministic_negative(client, adapter, invalid_params):
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, "earth2-deterministic", test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "invalid_params",
    [
        {"model": "no_such_model"},
        {"start_time": "not-a-datetime"},
        {"start_time": ""},
        {"nsteps": 0},
        {"nsteps": -1},
    ],
)
def test_earth2_deterministic_batch_negative(client, adapter, invalid_params):
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "earth2-deterministic-batch", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "invalid_params",
    [
        {"model": "no_such_model"},
        {"nensemble": 0},
        {"nensemble": -1},
        {"perturbation": "invalid_method"},
        {"noise_amplitude": 0},
        {"noise_amplitude": -0.5},
    ],
)
def test_earth2_ensemble_negative(client, adapter, invalid_params):
    test_params = {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
        "nensemble": 2,
        "perturbation": "gaussian",
        "noise_amplitude": 0.05,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(client, "earth2-ensemble", test_params, adapter=adapter)

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)


@pytest.mark.rust_only
@pytest.mark.parametrize(
    "invalid_params",
    [
        {"model": "no_such_model"},
        {"start_time": "not-a-datetime"},
        {"start_time": ""},
        {"nensemble": 0},
        {"nensemble": -1},
        {"batch_size": 0},
        {"batch_size": -1},
        {"perturbation": "invalid_method"},
    ],
)
def test_earth2_ensemble_fanout_negative(client, adapter, invalid_params):
    test_params = {
        "model": "fcn",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 3,
        "nensemble": 4,
        "batch_size": 2,
        "perturbation": "spherical_gaussian",
        "noise_amplitude": 0.05,
    }
    test_params.update(invalid_params)
    with pytest.raises(requests.exceptions.HTTPError) as err:
        run_workflow_test(
            client, "earth2-ensemble-fanout", test_params, adapter=adapter
        )

    assert err.value.response.status_code == 422
    assert "Unprocessable Entity" in str(err.value)
