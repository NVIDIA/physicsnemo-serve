# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compare_module():
    module_path = REPO_ROOT / "scripts" / "compare_zarr_outputs.py"
    spec = importlib.util.spec_from_file_location(
        "compare_zarr_outputs_test", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_numeric_arrays_sanitizes_infinite_relative_difference():
    module = _load_compare_module()
    array_a = SimpleNamespace(dims=["x"], values=np.asarray([0.0]))
    array_b = SimpleNamespace(dims=["x"], values=np.asarray([1.0]))

    with np.errstate(divide="ignore", invalid="ignore"):
        comparison = module._compare_numeric_arrays(
            "sample",
            array_a,
            array_b,
            rtol=0.0,
            atol=0.0,
        )

    assert comparison.max_abs_diff == 1.0
    assert comparison.max_rel_diff is None


def _write_dataset(path: Path, values: np.ndarray) -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    dataset = xr.Dataset(
        data_vars={
            "t2m": (
                ["ensemble", "time", "lead_time", "lat", "lon"],
                values.astype(np.float32),
            )
        },
        coords={
            "ensemble": np.asarray([0, 1], dtype=np.int64),
            "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.asarray([0], dtype="timedelta64[h]"),
            "lat": np.asarray([0.0], dtype=np.float32),
            "lon": np.asarray([0.0], dtype=np.float32),
        },
    )
    dataset.to_zarr(path, mode="w", zarr_format=3)


def test_compare_zarr_outputs_reports_pass_for_matching_datasets(tmp_path: Path):
    module = _load_compare_module()
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    values = np.arange(2, dtype=np.float32).reshape(2, 1, 1, 1, 1)
    _write_dataset(first, values)
    _write_dataset(second, values.copy())

    result_json = tmp_path / "report.json"
    exit_code = module.main(
        [
            str(first),
            str(second),
            "--include-coords",
            "--json",
            str(result_json),
        ]
    )

    assert exit_code == 0
    report = json.loads(result_json.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["summary"]["failure_count"] == 0
    assert report["data_variables"][0]["name"] == "t2m"
    assert report["data_variables"][0]["allclose"] is True


def test_compare_zarr_outputs_reports_failure_for_different_values(tmp_path: Path):
    module = _load_compare_module()
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    values = np.arange(2, dtype=np.float32).reshape(2, 1, 1, 1, 1)
    changed = values.copy()
    changed[1, 0, 0, 0, 0] += 10.0
    _write_dataset(first, values)
    _write_dataset(second, changed)

    result = module._compare_dataset(
        first,
        second,
        variables=["t2m"],
        include_coords=False,
        rtol=0.0,
        atol=0.0,
    )

    assert result["status"] == "failed"
    variable = result["data_variables"][0]
    assert variable["status"] == "different"
    assert variable["max_abs_diff"] == pytest.approx(10.0)
    assert result["summary"]["different"] == 1


def test_compare_zarr_outputs_reports_failure_for_nonfinite_mismatch(tmp_path: Path):
    module = _load_compare_module()
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    values = np.asarray([np.inf, np.nan], dtype=np.float32).reshape(2, 1, 1, 1, 1)
    changed = np.asarray([-np.inf, np.inf], dtype=np.float32).reshape(2, 1, 1, 1, 1)
    _write_dataset(first, values)
    _write_dataset(second, changed)

    result = module._compare_dataset(
        first,
        second,
        variables=["t2m"],
        include_coords=False,
        rtol=0.0,
        atol=0.0,
    )

    assert result["status"] == "failed"
    variable = result["data_variables"][0]
    assert variable["status"] == "different"
    assert variable["allclose"] is False
    assert result["summary"]["different"] == 1
