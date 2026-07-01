# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CaseSpec load and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CASE_SPEC_SCHEMA_VERSION = "case_spec.v1"
COORDS_POLICIES = {"default_parallel_coords", "explicit_parallel_coords"}
ZARR_FORMATS = {"v2", "v3"}
CHUNK_KEY_ENCODINGS = {"v2", "default"}
CHUNK_KEY_SEPARATORS = {".", "/", None}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_non_empty_string(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and value.strip() != "",
        f"{field} must be a non-empty string",
    )
    return value


def validate_case_spec(case_spec: dict[str, Any]) -> None:
    """Validate a CaseSpecV1 mapping."""
    required = {
        "schema_version",
        "case_id",
        "workflow_id",
        "deterministic_seed",
        "start_time",
        "start_times",
        "num_steps",
        "step_delta",
        "output_array_names",
        "coords_policy",
        "parallel_coords",
        "zarr_format",
        "chunk_key_encoding",
        "chunk_key_separator",
    }
    missing = sorted(required - set(case_spec))
    _require(not missing, f"missing CaseSpec fields: {missing}")
    _require(
        case_spec["schema_version"] == CASE_SPEC_SCHEMA_VERSION,
        "schema_version must be case_spec.v1",
    )
    _require_non_empty_string(case_spec["case_id"], "case_id")
    _require_non_empty_string(case_spec["workflow_id"], "workflow_id")
    _require(
        isinstance(case_spec["deterministic_seed"], int),
        "deterministic_seed must be an int",
    )
    _require(
        isinstance(case_spec["num_steps"], int) and case_spec["num_steps"] > 0,
        "num_steps must be > 0",
    )
    _require_non_empty_string(case_spec["step_delta"], "step_delta")

    start_time = case_spec["start_time"]
    _require(
        start_time is None or isinstance(start_time, str),
        "start_time must be str or None",
    )
    start_times = case_spec["start_times"]
    _require(
        isinstance(start_times, list) and start_times,
        "start_times must be a non-empty list",
    )
    _require(
        all(isinstance(item, str) and item for item in start_times),
        "start_times must contain non-empty strings",
    )
    if start_time is not None:
        _require(
            start_times == [start_time],
            "single-start case must set start_times to [start_time]",
        )

    output_array_names = case_spec["output_array_names"]
    _require(
        isinstance(output_array_names, list) and output_array_names,
        "output_array_names must be a non-empty list",
    )
    _require(
        all(isinstance(name, str) and name for name in output_array_names),
        "output_array_names must contain non-empty strings",
    )

    coords_policy = case_spec["coords_policy"]
    _require(
        coords_policy in COORDS_POLICIES,
        f"coords_policy must be one of {sorted(COORDS_POLICIES)}",
    )
    parallel_coords = case_spec["parallel_coords"]
    _require(
        parallel_coords is None or isinstance(parallel_coords, dict),
        "parallel_coords must be a dict or None",
    )
    if coords_policy == "explicit_parallel_coords":
        _require(
            isinstance(parallel_coords, dict) and parallel_coords,
            "explicit coords_policy requires parallel_coords",
        )

    zarr_format = case_spec["zarr_format"]
    _require(
        zarr_format in ZARR_FORMATS,
        f"zarr_format must be one of {sorted(ZARR_FORMATS)}",
    )
    chunk_key_encoding = case_spec["chunk_key_encoding"]
    _require(
        chunk_key_encoding in CHUNK_KEY_ENCODINGS,
        f"chunk_key_encoding must be one of {sorted(CHUNK_KEY_ENCODINGS)}",
    )
    chunk_key_separator = case_spec["chunk_key_separator"]
    _require(
        chunk_key_separator in CHUNK_KEY_SEPARATORS,
        "chunk_key_separator must be '.', '/', or None",
    )
    if zarr_format == "v2":
        _require(
            chunk_key_encoding == "v2", "v2 format requires chunk_key_encoding='v2'"
        )
        _require(
            chunk_key_separator in {".", "/"},
            "v2 format requires chunk_key_separator '.' or '/'",
        )
    else:
        _require(
            chunk_key_encoding == "default",
            "v3 format requires chunk_key_encoding='default'",
        )
        _require(
            chunk_key_separator is None, "v3 format requires chunk_key_separator=None"
        )


def load_case_spec(case_spec_path: str | Path) -> dict[str, Any]:
    """Load and validate a CaseSpecV1 JSON file."""
    path = Path(case_spec_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "CaseSpec must be a JSON object")
    validate_case_spec(data)
    return data
