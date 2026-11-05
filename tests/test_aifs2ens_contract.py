# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One shared initialization and one seed for a complete GPU ensemble."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
from aifs2ens_contract import ForecastRequest  # noqa: E402


def request(**changes):
    return {
        "start_time": "2026-11-03T00:00:00Z",
        "nsteps": 4,
        "nensemble": 2,
        "seed_base": 0,
        **changes,
    }


def test_default_batches_all_members_with_one_seed_and_one_common_source():
    parsed = ForecastRequest.model_validate(request())
    assert parsed.seed_base == 0
    assert parsed.batch_size == 2
    assert parsed.ifs_member_id is None
    assert not {
        "max_in_flight",
        "ifs_member_ids",
        "initial_condition_mode",
    }.intersection(ForecastRequest.model_fields)


def test_common_ifs_ensemble_source_is_shared_by_every_output_member():
    parsed = ForecastRequest.model_validate(request(ifs_member_id=7, nensemble=5))
    assert parsed.ifs_member_id == 7
    assert parsed.nensemble == parsed.batch_size == 5
    assert parsed.seed_base == 0


@pytest.mark.parametrize("size,expected", [(None, 2), (1, 1), (2, 2), (8, 2)])
def test_batch_size_follows_stock_ensemble_normalization(size, expected):
    assert (
        ForecastRequest.model_validate(request(batch_size=size)).batch_size == expected
    )


def test_same_seed_and_requested_layout_are_preserved_for_every_horizon():
    assert ForecastRequest.model_validate(request(nsteps=60)).seed_base == 0
    assert (
        ForecastRequest.model_validate(request(seed_base=2**63 - 1)).seed_base
        == 2**63 - 1
    )


def test_normalizes_utc_and_requires_six_hour_initialization():
    parsed = ForecastRequest.model_validate(
        request(start_time="2026-11-03T01:00:00+01:00")
    )
    assert parsed.start_time == "2026-11-03T00:00:00Z"


@pytest.mark.parametrize(
    "changes",
    [
        {"start_time": "2026-11-03T00:00:00"},
        {"start_time": "2026-11-03T01:00:00Z"},
        {"start_time": "2026-11-03T00:00:01Z"},
        {"start_time": "not-a-date"},
        {"nsteps": 0},
        {"nsteps": 61},
        {"nsteps": True},
        {"nsteps": "4"},
        {"nensemble": 0},
        {"nensemble": 17},
        {"nensemble": True},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": "2"},
        {"batch_size": 17},
        {"ifs_member_id": 0},
        {"ifs_member_id": 51},
        {"ifs_member_id": True},
        {"seed_base": -1},
        {"seed_base": 2**63},
        {"seed_base": True},
        {"output_variables": []},
        {"output_variables": ["t2m", "t2m"]},
        {"output_variables": ["../input"]},
        {"ifs_member_ids": [1, 2]},
        {"initial_condition_mode": "shared_ifs"},
        {"max_in_flight": 1},
        {"checkpoint": "/caller/checkpoint"},
    ],
)
def test_rejects_invalid_and_obsolete_fanout_requests(changes):
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(request(**changes))


def test_public_schema_rejects_unknown_fields():
    schema = ForecastRequest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "batch_size" in schema["properties"]
    assert "manifest_path" not in schema["properties"]
