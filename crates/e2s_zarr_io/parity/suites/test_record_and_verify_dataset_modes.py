# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-mode tests for record_truth and verify_parity tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from parity.tools.record_truth import build_blessed_truth_manifest_from_datasets
from parity.tools.verify_parity import verify_candidate_dataset_against_truth
from parity.utils.canonical_reader import build_manifest_from_dataset

from ._dataset_helpers import create_test_zarr_dataset


def _case_spec() -> dict[str, object]:
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
        "coord_names": ["time"],
    }


def test_record_truth_dataset_mode_succeeds_for_matching_py_baselines(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    py_sync_path = tmp_path / "py_sync.zarr"
    py_async_path = tmp_path / "py_async.zarr"
    for path in (py_sync_path, py_async_path):
        create_test_zarr_dataset(
            dataset_path=path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "baseline"},
        )
    blessed, report = build_blessed_truth_manifest_from_datasets(
        case_spec=_case_spec(),
        py_sync_dataset_path=py_sync_path,
        py_async_dataset_path=py_async_path,
        earth2studio_commit="abc1234",
        python_version="3.12.4",
        zarr_python_version="2.18.3",
        case_spec_sha256="0" * 64,
    )
    assert blessed["generated_by_backend"] == "py_sync"
    assert report["py_baseline_equal"] is True
    assert report["diff_count"] == 0


def test_record_truth_dataset_mode_fails_for_baseline_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    py_sync_path = tmp_path / "py_sync_bad.zarr"
    py_async_path = tmp_path / "py_async_bad.zarr"
    create_test_zarr_dataset(
        dataset_path=py_sync_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "baseline"},
    )
    create_test_zarr_dataset(
        dataset_path=py_async_path,
        arrays={"temperature": np.array([[9.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "baseline"},
    )
    with pytest.raises(AssertionError, match="manifest semantic parity failed"):
        build_blessed_truth_manifest_from_datasets(
            case_spec=_case_spec(),
            py_sync_dataset_path=py_sync_path,
            py_async_dataset_path=py_async_path,
            earth2studio_commit="abc1234",
            python_version="3.12.4",
            zarr_python_version="2.18.3",
            case_spec_sha256="0" * 64,
        )


def test_verify_parity_dataset_mode_passes_for_matching_candidate(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    truth_path = tmp_path / "truth.zarr"
    rust_path = tmp_path / "rust.zarr"
    for path in (truth_path, rust_path):
        create_test_zarr_dataset(
            dataset_path=path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "parity"},
        )
    case_spec = _case_spec()
    truth_manifest = build_manifest_from_dataset(
        dataset_path=truth_path,
        case_spec=case_spec,
        generated_by_backend="py_sync",
    )
    candidate_manifest = verify_candidate_dataset_against_truth(
        truth_manifest=truth_manifest,
        case_spec=case_spec,
        candidate_dataset_path=str(rust_path),
    )
    assert candidate_manifest["generated_by_backend"] == "rust"


def test_verify_parity_dataset_mode_fails_for_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    truth_path = tmp_path / "truth_bad.zarr"
    rust_path = tmp_path / "rust_bad.zarr"
    create_test_zarr_dataset(
        dataset_path=truth_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "parity"},
    )
    create_test_zarr_dataset(
        dataset_path=rust_path,
        arrays={"temperature": np.array([[8.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "parity"},
    )
    case_spec = _case_spec()
    truth_manifest = build_manifest_from_dataset(
        dataset_path=truth_path,
        case_spec=case_spec,
        generated_by_backend="py_sync",
    )
    with pytest.raises(AssertionError, match="manifest semantic parity failed"):
        verify_candidate_dataset_against_truth(
            truth_manifest=truth_manifest,
            case_spec=case_spec,
            candidate_dataset_path=str(rust_path),
        )
