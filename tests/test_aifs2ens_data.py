# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasource preparation and invariant binding without remote data downloads."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "plugins" / "earth2-aifs2ens")
)

from aifs2ens_artifacts import INVARIANTS, InputContract, sha256_file  # noqa: E402
from aifs2ens_contract import ForecastRequest  # noqa: E402
from aifs2ens_data import WeatherSource  # noqa: E402


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fixed = np.ones((5, 2, 2), dtype=np.float32)
    fixed[0, 0, 0] = 0  # ocean in the frozen land mask
    path = tmp_path / "invariants.npy"
    np.save(path, fixed)
    monkeypatch.setenv("AIFS2ENS_INVARIANTS", str(path))
    contract = InputContract.model_validate(
        {
            "schema_version": 1,
            "checkpoint_sha256": "a" * 64,
            "invariants_sha256": sha256_file(path),
            "input_variables": ["t2m", "sd", "swvl1", "swvl2", *INVARIANTS],
            "output_variables": ["t2m"],
            "lat": [90.0, -90.0],
            "lon": [0.0, 180.0],
        }
    )
    req = ForecastRequest.model_validate(
        {
            "start_time": "2026-11-03T00:00:00Z",
            "nensemble": 1,
            "nsteps": 1,
            "ifs_member_id": 7,
            "output_variables": ["t2m"],
        }
    )
    return req, contract, path, fixed


def test_preserves_earth2studio_values_with_selected_member_and_frozen_invariants(
    setup,
):
    req, contract, path, fixed = setup
    calls = []

    def fetch(request, variables):
        calls.append((request.ifs_member_id, list(variables)))
        coords = contract.coords(request)
        coords["variable"] = np.array(variables)
        return np.full((1, 2, 4, 2, 2), 7, dtype=np.float32), coords

    result = WeatherSource(fetch=fetch)(req, contract)
    assert calls == [(7, ["t2m", "sd", "swvl1", "swvl2"])]
    for i, name in enumerate(INVARIANTS):
        actual = result.values[:, :, contract.input_variables.index(name)]
        np.testing.assert_array_equal(actual, np.broadcast_to(fixed[i], (1, 2, 2, 2)))
    assert result.values[0, 0, 1, 0, 0] == 7
    assert result.values[0, 0, 2, 0, 0] == 7
    assert result.values[0, 0, 0, 0, 0] == 7
    assert result.provenance["ifs_member_id"] == 7
    assert result.provenance["invariants_sha256"] == sha256_file(path)


def test_rejects_changed_invariants_before_requesting_weather(setup):
    req, contract, path, fixed = setup
    np.save(path, fixed + 1)
    with pytest.raises(ValueError, match="invariant.*digest"):
        WeatherSource(fetch=lambda *a: pytest.fail("downloaded weather"))(req, contract)


def test_data_coordinate_order_must_match_the_model_contract(setup):
    req, contract, path, fixed = setup

    def fetch(request, variables):
        coords = contract.coords(request)
        coords["variable"] = np.array(variables)
        coords["lead_time"] = coords["lead_time"][::-1]
        return np.zeros((1, 2, 4, 2, 2), dtype=np.float32), coords

    with pytest.raises(ValueError, match="lead_time"):
        WeatherSource(fetch=fetch)(req, contract)


def test_datasource_rejects_unpinned_runtime_before_loading_sources(monkeypatch):
    import aifs2ens_runtime

    def reject():
        raise RuntimeError("install the pinned Earth2Studio source revision")

    monkeypatch.setattr(aifs2ens_runtime, "_verify_runtime", reject)
    with pytest.raises(RuntimeError, match="pinned Earth2Studio"):
        WeatherSource._fetch(None, [])
