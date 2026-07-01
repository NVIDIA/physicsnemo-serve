# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CaseSpec validation tests."""

from __future__ import annotations

import pytest

from parity.utils.case_spec import validate_case_spec


def _valid_case_spec() -> dict[str, object]:
    return {
        "schema_version": "case_spec.v1",
        "case_id": "wf_demo__fmt_v2__steps_2",
        "workflow_id": "demo_workflow",
        "deterministic_seed": 7,
        "start_time": "2026-01-01T00:00:00Z",
        "start_times": ["2026-01-01T00:00:00Z"],
        "num_steps": 2,
        "step_delta": "1h",
        "output_array_names": ["temperature"],
        "coords_policy": "default_parallel_coords",
        "parallel_coords": None,
        "zarr_format": "v2",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
    }


def test_valid_case_spec_passes_validation() -> None:
    case_spec = _valid_case_spec()
    validate_case_spec(case_spec)


def test_single_start_requires_start_times_match() -> None:
    case_spec = _valid_case_spec()
    case_spec["start_times"] = ["2026-01-01T00:00:00Z", "2026-01-01T06:00:00Z"]
    with pytest.raises(ValueError, match="single-start case"):
        validate_case_spec(case_spec)


def test_v2_requires_v2_encoding() -> None:
    case_spec = _valid_case_spec()
    case_spec["chunk_key_encoding"] = "default"
    with pytest.raises(ValueError, match="v2 format requires chunk_key_encoding"):
        validate_case_spec(case_spec)


def test_explicit_coords_policy_requires_parallel_coords() -> None:
    case_spec = _valid_case_spec()
    case_spec["coords_policy"] = "explicit_parallel_coords"
    case_spec["parallel_coords"] = None
    with pytest.raises(
        ValueError, match="explicit coords_policy requires parallel_coords"
    ):
        validate_case_spec(case_spec)


def test_v3_requires_null_separator_and_default_encoding() -> None:
    case_spec = _valid_case_spec()
    case_spec["zarr_format"] = "v3"
    case_spec["chunk_key_encoding"] = "default"
    case_spec["chunk_key_separator"] = "/"
    with pytest.raises(ValueError, match="v3 format requires chunk_key_separator=None"):
        validate_case_spec(case_spec)
