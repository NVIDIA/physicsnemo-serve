# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A frozen common input must bind the whole ensemble and exact numerical bytes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "plugins" / "earth2-aifs2ens")
)

from aifs2ens_artifacts import (  # noqa: E402
    InputContract,
    WeatherInput,
    load_ensemble_input,
    sha256_file,
    stage_inputs,
)
from aifs2ens_contract import ForecastRequest  # noqa: E402


@pytest.fixture
def contract():
    return InputContract.model_validate(
        {
            "schema_version": 1,
            "checkpoint_sha256": "a" * 64,
            "input_variables": ["t2m", "msl", "lsm", "sdor", "slor", "z", "wmb"],
            "output_variables": ["t2m", "msl", "u10m", "v10m"],
            "lat": [90.0, -90.0],
            "lon": [0.0, 120.0, 240.0],
        }
    )


def request(**changes):
    return ForecastRequest.model_validate(
        {
            "start_time": "2026-11-03T00:00:00Z",
            "ifs_member_id": 7,
            "nensemble": 2,
            "nsteps": 4,
            "seed_base": 0,
            **changes,
        }
    )


def weather(contract, member):
    values = np.zeros((1, 2, len(contract.input_variables), 2, 3), dtype=np.float32)
    values[:, :, :2] = member if member is not None else 42
    return WeatherInput(values=values, provenance={"source_member": member})


def test_common_input_is_fetched_once_for_the_entire_ensemble(tmp_path, contract):
    calls = []

    def fetch(req, shape):
        calls.append((req.start_time, req.ifs_member_id))
        return weather(shape, req.ifs_member_id)

    staged = stage_inputs(request(), contract, tmp_path, fetch, max_output_bytes=10**6)
    assert calls == [("2026-11-03T00:00:00Z", 7)]
    loaded = load_ensemble_input(staged["manifest_path"], staged["manifest_sha256"])
    assert loaded.input_sha256 == sha256_file(
        Path(staged["manifest_path"]).parent / "input.npy"
    )
    assert loaded.request.nensemble == 2
    assert loaded.request.batch_size == 2
    assert loaded.request.seed_base == 0
    np.testing.assert_array_equal(loaded.values, weather(contract, 7).values)
    assert list(loaded.coords) == ["time", "lead_time", "variable", "lat", "lon"]
    np.testing.assert_array_equal(
        loaded.coords["lead_time"], np.array([-6, 0], dtype="timedelta64[h]")
    )
    assert loaded.coords["time"][0] == np.datetime64("2026-11-03T00:00:00")
    manifest = json.loads(Path(staged["manifest_path"]).read_text())
    assert "members" not in manifest
    assert manifest["provenance"] == {"source_member": 7}


def test_storage_budget_rejects_before_weather_download(tmp_path, contract):
    def fetch(*args):
        pytest.fail("weather must not be downloaded for an over-budget request")

    with pytest.raises(ValueError, match="output.*budget"):
        stage_inputs(request(), contract, tmp_path, fetch, max_output_bytes=1)


def test_unknown_output_variable_rejects_before_weather_download(tmp_path, contract):
    with pytest.raises(ValueError, match="output variable"):
        stage_inputs(
            request(output_variables=["not_a_model_field"]),
            contract,
            tmp_path,
            lambda *args: pytest.fail("downloaded"),
            max_output_bytes=10**6,
        )


@pytest.mark.parametrize("change", ["bytes", "manifest", "contract"])
def test_tampering_blocks_ensemble_execution(tmp_path, contract, change):
    staged = stage_inputs(
        request(),
        contract,
        tmp_path,
        lambda req, shape: weather(shape, req.ifs_member_id),
        max_output_bytes=10**6,
    )
    manifest_path = Path(staged["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    target = {
        "bytes": manifest_path.parent / manifest["input_path"],
        "manifest": manifest_path,
        "contract": manifest_path.parent / manifest["contract_path"],
    }[change]
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        load_ensemble_input(str(manifest_path), staged["manifest_sha256"])


@pytest.mark.parametrize("bad", ["shape", "dtype", "invariants"])
def test_incompatible_inputs_cannot_emit_a_valid_manifest(tmp_path, contract, bad):
    fetched = weather(contract, 7)
    if bad == "shape":
        fetched.values = fetched.values[:, :1]
    elif bad == "dtype":
        fetched.values = fetched.values.astype(np.float64)
    else:
        fetched.values[0, 1, -1, 0, 0] = 1
    with pytest.raises(ValueError, match="shape|float32|invariant"):
        stage_inputs(
            request(), contract, tmp_path, lambda *args: fetched, max_output_bytes=10**6
        )
    assert not list(tmp_path.rglob("manifest.json"))


def test_two_parent_attempts_cannot_overwrite_staged_input_bytes(tmp_path, contract):
    def fetch(req, shape):
        return weather(shape, req.ifs_member_id)

    first = stage_inputs(request(), contract, tmp_path, fetch, max_output_bytes=10**6)
    second = stage_inputs(request(), contract, tmp_path, fetch, max_output_bytes=10**6)
    assert first["manifest_path"] != second["manifest_path"]
    assert (
        load_ensemble_input(
            first["manifest_path"], first["manifest_sha256"]
        ).request.seed_base
        == 0
    )
