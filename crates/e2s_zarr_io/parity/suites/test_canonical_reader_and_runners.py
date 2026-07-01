# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-driven canonical reader and backend runner tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from parity.utils.backend_runners import (
    BackendRunnerRegistry,
    run_backend_and_collect_manifest,
)
from parity.utils.canonical_reader import (
    build_manifest_from_dataset,
    read_dataset_content,
)
from parity.utils.case_spec import load_case_spec

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
        "output_array_names": ["temperature", "pressure"],
        "coords_policy": "default_parallel_coords",
        "parallel_coords": None,
        "zarr_format": "v2",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
        "coord_names": ["time"],
    }


def test_canonical_reader_reads_arrays_coords_and_attrs(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "reader_case.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={
            "temperature": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "pressure": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        },
        coords={"time": np.array([10, 11], dtype=np.int64)},
        attrs={"generated_at": "volatile", "source": "test"},
    )
    arrays, coords, attrs, zarr_info = read_dataset_content(dataset_path, _case_spec())
    assert sorted(arrays.keys()) == ["pressure", "temperature"]
    assert sorted(coords.keys()) == ["time"]
    assert attrs["source"] == "test"
    assert zarr_info["zarr_format"] == "v2"


def test_build_manifest_from_dataset_is_deterministic(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "manifest_case.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={
            "temperature": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "pressure": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        },
        coords={"time": np.array([10, 11], dtype=np.int64)},
        attrs={"source": "test"},
    )
    manifest_a = build_manifest_from_dataset(
        dataset_path=dataset_path,
        case_spec=_case_spec(),
        generated_by_backend="py_sync",
    )
    manifest_b = build_manifest_from_dataset(
        dataset_path=dataset_path,
        case_spec=_case_spec(),
        generated_by_backend="rust",
    )
    assert manifest_a["dataset_sha256"] == manifest_b["dataset_sha256"]
    assert manifest_a["attrs_canonical_sha256"] == manifest_b["attrs_canonical_sha256"]


def test_backend_runner_registry_runs_and_collects_manifest(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")

    def fake_py_sync_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={
                "temperature": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "pressure": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            },
            coords={"time": np.array([10, 11], dtype=np.int64)},
            attrs={"source": "runner"},
        )

    registry = BackendRunnerRegistry()
    registry.register("py_sync", fake_py_sync_runner)
    manifest = run_backend_and_collect_manifest(
        registry=registry,
        backend_kind="py_sync",
        case_spec=_case_spec(),
        dataset_path=tmp_path / "runner_case.zarr",
        generated_by_backend="py_sync",
    )
    assert manifest["generated_by_backend"] == "py_sync"
    assert manifest["zarr_info"]["zarr_format"] == "v2"


def test_backend_runner_registry_rejects_unregistered_backend(tmp_path: Path) -> None:
    registry = BackendRunnerRegistry()
    with pytest.raises(RuntimeError, match="no runner registered"):
        run_backend_and_collect_manifest(
            registry=registry,
            backend_kind="rust",
            case_spec=_case_spec(),
            dataset_path=tmp_path / "runner_missing.zarr",
            generated_by_backend="rust",
        )


def test_case_spec_loader_supports_dataset_mode_fixture(tmp_path: Path) -> None:
    case_spec_path = tmp_path / "case_spec.json"
    case_spec_path.write_text(
        """{
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
  "parallel_coords": null,
  "zarr_format": "v2",
  "chunk_key_encoding": "v2",
  "chunk_key_separator": "."
}""",
        encoding="utf-8",
    )
    loaded = load_case_spec(case_spec_path)
    assert loaded["case_id"] == "wf_demo__fmt_v2__steps_2"
