# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare untouched CUDA forecasts against a frozen reference-variability envelope."""

from itertools import combinations

import numpy as np

POLICY = "earth2studio-native-v1"
RNG_POLICY = "torch-seed-once-native-v1"
MARGIN = 2.0  # Declared before any reference or candidate execution.


def field_metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("forecast shape differs from reference")
    if actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError("forecast/reference dtype must be float32")
    for mask in (np.isnan, np.isposinf, np.isneginf):
        if not np.array_equal(mask(actual), mask(expected)):
            raise ValueError("forecast nonfinite mask differs from reference")
    finite = np.isfinite(expected)
    left = actual[finite].astype(np.float64)
    right = expected[finite].astype(np.float64)
    if not right.size:
        return {"max_abs": 0.0, "rmse": 0.0, "max_relative": 0.0, "finite_count": 0}
    error = np.abs(left - right)
    # Relative errors are diagnostic; near-zero denominators use a recorded
    # field-scale floor. Acceptance uses absolute error and RMSE in field units.
    floor = max(float(np.max(np.abs(right))) * 1e-6, float(np.finfo(np.float32).tiny))
    return {
        "max_abs": float(np.max(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_relative": float(np.max(error / np.maximum(np.abs(right), floor))),
        "relative_denominator_floor": floor,
        "finite_count": int(right.size),
    }


def calibrate_arrays(samples):
    if len(samples) < 4:
        raise ValueError("at least four independent reference runs are required")
    metrics = [field_metrics(a, b) for a, b in combinations(samples, 2)]
    maximum = max(item["max_abs"] for item in metrics)
    rmse = max(item["rmse"] for item in metrics)
    return {
        "reference_count": len(samples),
        "margin": MARGIN,
        "observed_max_abs": maximum,
        "observed_rmse": rmse,
        "max_abs_limit": MARGIN * maximum,
        "rmse_limit": MARGIN * rmse,
    }


def compare_field(actual, expected, limits):
    for name in ("max_abs_limit", "rmse_limit"):
        if not np.isfinite(limits[name]) or limits[name] < 0:
            raise ValueError("invalid frozen reference limits")
    metrics = field_metrics(actual, expected)
    return {
        **metrics,
        "max_abs_limit": limits["max_abs_limit"],
        "rmse_limit": limits["rmse_limit"],
        "passed": metrics["max_abs"] <= limits["max_abs_limit"]
        and metrics["rmse"] <= limits["rmse_limit"],
    }


def _identity(report):
    return {
        name: report.get(name)
        for name in (
            "nensemble",
            "nsteps",
            "batch_size",
            "seed_base",
            "model_seed",
            "rng_policy",
            "input_sha256",
            "checkpoint_sha256",
            "output_variables",
        )
    }


def _read_report(root):
    import json

    report = json.loads((root / "report.json").read_text())
    if report.get("rng_policy") != RNG_POLICY or report.get("model_seed") is not None:
        raise ValueError("reference RNG policy differs from shared ensemble stream")
    if len(report["steps"]) != report["nsteps"]:
        raise ValueError("reference forecast step coverage differs")
    return report


def _array(root, step, report):
    array = np.load(root / f"step-{step:04d}.npy", allow_pickle=False, mmap_mode="r")
    if (
        array.ndim != 6
        or array.shape[:4]
        != (report["nensemble"], 1, 1, len(report["output_variables"]))
        or array.dtype != np.float32
    ):
        raise ValueError("reference ensemble shape/dtype differs from layout")
    return array


def calibrate_ensemble(reference_paths):
    """Freeze limits for each member/field/step of one exact ensemble batch layout."""
    from pathlib import Path

    roots = [Path(path) for path in reference_paths]
    if len(roots) < 4 or len(set(roots)) != len(roots):
        raise ValueError("at least four independent reference paths are required")
    reports = [_read_report(root) for root in roots]
    first = reports[0]
    if any(_identity(report) != _identity(first) for report in reports):
        raise ValueError("reference ensemble batch layout or identity differs")
    result = {
        **_identity(first),
        "reference_count": len(roots),
        "members": {str(index): {"steps": {}} for index in range(first["nensemble"])},
    }
    for step in range(1, first["nsteps"] + 1):
        arrays = [_array(root, step, first) for root in roots]
        for member in range(first["nensemble"]):
            result["members"][str(member)]["steps"][str(step)] = {
                name: calibrate_arrays([array[member, :, 0, index] for array in arrays])
                for index, name in enumerate(first["output_variables"])
            }
        del arrays
    return result


def validate_holdouts(reference_path, holdout_paths, limits):
    """Whole-ensemble holdouts excluded from calibration must pass before candidates."""
    from pathlib import Path

    root = Path(reference_path)
    report = _read_report(root)
    if _identity(report) != _identity(limits):
        raise ValueError("frozen calibration identity differs from reference")
    records, failures = [], []
    for holdout in holdout_paths:
        holdout = Path(holdout)
        if _identity(_read_report(holdout)) != _identity(report):
            raise ValueError("held-out reference batch layout or identity differs")
        for step in range(1, report["nsteps"] + 1):
            expected = _array(root, step, report)
            actual = _array(holdout, step, report)
            for member in range(report["nensemble"]):
                for index, name in enumerate(report["output_variables"]):
                    metrics = compare_field(
                        actual[member, :, 0, index],
                        expected[member, :, 0, index],
                        limits["members"][str(member)]["steps"][str(step)][name],
                    )
                    item = {
                        "reference": str(holdout),
                        "member_index": member,
                        "step": step,
                        "variable": name,
                        **metrics,
                    }
                    records.append(item)
                    if not metrics["passed"]:
                        failures.append(item)
            del actual, expected
    return {
        "status": "passed" if not failures else "failed",
        "comparisons": records,
        "failures": failures,
    }
