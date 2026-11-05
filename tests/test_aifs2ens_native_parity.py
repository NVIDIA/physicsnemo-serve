# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CUDA variation is measured before candidate evaluation, never hidden."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = (
    Path(__file__).resolve().parents[1] / "qa/plugins/earth2-aifs2ens/native_parity.py"
)
spec = importlib.util.spec_from_file_location("aifs2ens_native_parity", path)
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_identical_nonfinite_masks_and_values_have_zero_error():
    values = np.array([1, np.nan, np.inf, -np.inf], dtype=np.float32)
    metrics = parity.field_metrics(values.copy(), values)
    assert metrics["max_abs"] == metrics["rmse"] == 0
    assert metrics["finite_count"] == 1


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "nan_mask", "infinity_sign"])
def test_structural_or_nonfinite_differences_cannot_be_tolerated(mismatch):
    expected = np.array([1, np.nan, np.inf], dtype=np.float32)
    actual = expected.copy()
    if mismatch == "shape":
        actual = actual[None]
    elif mismatch == "dtype":
        actual = actual.astype(np.float64)
    elif mismatch == "nan_mask":
        actual[1] = 0
    else:
        actual[2] = -np.inf
    with pytest.raises(ValueError):
        parity.field_metrics(actual, expected)


def test_reference_variation_sets_frozen_bounds_that_reject_a_changed_candidate():
    samples = [
        np.full((2, 2), value, dtype=np.float32) for value in (10, 10.01, 9.99, 10.005)
    ]
    limits = parity.calibrate_arrays(samples)
    original = copy.deepcopy(limits)
    assert limits["margin"] == 2
    assert limits["max_abs_limit"] == 2 * limits["observed_max_abs"]
    assert parity.compare_field(
        np.full((2, 2), 10.015, dtype=np.float32), samples[0], limits
    )["passed"]
    assert not parity.compare_field(
        np.full((2, 2), 10.25, dtype=np.float32), samples[0], limits
    )["passed"]
    assert limits == original  # A failed candidate cannot change its threshold.


def test_bitwise_repeatable_fields_retain_zero_tolerance():
    expected = np.ones((2, 2), dtype=np.float32)
    limits = parity.calibrate_arrays([expected.copy() for _ in range(4)])
    assert limits["max_abs_limit"] == limits["rmse_limit"] == 0
    changed = expected.copy()
    changed[0, 0] = np.nextafter(np.float32(1), np.float32(2))
    assert not parity.compare_field(changed, expected, limits)["passed"]


def test_insufficient_reference_calibration_is_rejected():
    with pytest.raises(ValueError, match="four"):
        parity.calibrate_arrays([np.ones(2, dtype=np.float32)] * 3)


def write_ensemble_reference(root, values, *, batch_size=2):
    import json

    root.mkdir()
    array = np.stack(
        [np.full((1, 1, 1, 2, 2), value, dtype=np.float32) for value in values]
    )
    np.save(root / "step-0001.npy", array)
    (root / "report.json").write_text(
        json.dumps(
            {
                "output_variables": ["t2m"],
                "steps": [{"step": 1}],
                "nensemble": len(values),
                "nsteps": 1,
                "batch_size": batch_size,
                "seed_base": 0,
                "model_seed": None,
                "rng_policy": "torch-seed-once-native-v1",
            }
        )
    )
    return root


def test_reference_limits_are_per_member_and_reject_swapped_member_slots(tmp_path):
    roots = [
        write_ensemble_reference(tmp_path / str(i), values)
        for i, values in enumerate(
            (
                (10, 20),
                (10.01, 20.01),
                (9.99, 19.99),
                (10.005, 20.005),
                (10.015, 20.015),
                (20, 10),
            )
        )
    ]
    limits = parity.calibrate_ensemble(roots[:4])
    assert limits["nensemble"] == 2
    assert set(limits["members"]) == {"0", "1"}
    assert limits["members"]["0"]["steps"]["1"]["t2m"]["max_abs_limit"] < 0.1
    assert parity.validate_holdouts(roots[0], [roots[4]], limits)["status"] == "passed"
    failed = parity.validate_holdouts(roots[0], [roots[5]], limits)
    assert failed["status"] == "failed"
    assert {row["member_index"] for row in failed["failures"]} == {0, 1}


def test_reference_calibration_cannot_mix_batch_layouts(tmp_path):
    roots = [
        write_ensemble_reference(
            tmp_path / str(i), (10, 20), batch_size=1 if i == 3 else 2
        )
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="layout|identity|batch"):
        parity.calibrate_ensemble(roots)
