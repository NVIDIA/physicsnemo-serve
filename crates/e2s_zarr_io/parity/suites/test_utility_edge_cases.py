# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge-case coverage tests for parity utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from parity.utils.backend_runners import BackendRunnerRegistry
from parity.utils.canonical_reader import (
    build_manifest_from_dataset,
    read_dataset_content,
)
from parity.utils.manifest_builder import build_truth_manifest
from parity.utils.manifest_compare import compare_semantic_manifests
from parity.utils.workflow_catalog import WorkflowCatalog

from ._dataset_helpers import create_test_zarr_dataset


class BytesOnlyArray:
    """Array-like payload with no shape/dtype attributes."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise ValueError("only C-order is supported in this fake")
        return self._payload


def _base_case_spec() -> dict[str, object]:
    return {
        "schema_version": "case_spec.v1",
        "case_id": "edge_case",
        "workflow_id": "demo_workflow",
        "deterministic_seed": 7,
        "start_time": "2026-01-01T00:00:00Z",
        "start_times": ["2026-01-01T00:00:00Z"],
        "num_steps": 2,
        "step_delta": "1h",
        "output_array_names": ["temperature"],
        "coords_policy": "default_parallel_coords",
        "parallel_coords": None,
        "zarr_format": "invalid_format_for_inference",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
    }


def test_build_truth_manifest_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="generated_by_backend"):
        build_truth_manifest(
            case_id="edge_case",
            generated_by_backend="not_supported",
            zarr_info={
                "zarr_format": "v2",
                "chunk_key_encoding": "v2",
                "chunk_key_separator": ".",
                "store_kind": "local_fs",
            },
            arrays={"temperature": BytesOnlyArray(b"\x10\x11")},
            coords={"time": BytesOnlyArray(b"\x01")},
        )


def test_build_truth_manifest_rejects_empty_case_id() -> None:
    with pytest.raises(ValueError, match="case_id must be non-empty"):
        build_truth_manifest(
            case_id="",
            generated_by_backend="py_sync",
            zarr_info={
                "zarr_format": "v2",
                "chunk_key_encoding": "v2",
                "chunk_key_separator": ".",
                "store_kind": "local_fs",
            },
            arrays={"temperature": BytesOnlyArray(b"\x10\x11")},
            coords={"time": BytesOnlyArray(b"\x01")},
        )


def test_build_truth_manifest_supports_bytes_only_inputs() -> None:
    manifest = build_truth_manifest(
        case_id="edge_case",
        generated_by_backend="py_sync",
        zarr_info={
            "zarr_format": "v2",
            "chunk_key_encoding": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
        },
        arrays={"temperature": BytesOnlyArray(b"\x10\x11")},
        coords={"time": BytesOnlyArray(b"\x01")},
        attrs={"generated_at": "volatile", "source": "stable"},
    )
    assert manifest["arrays"][0]["shape"] == [2]
    assert manifest["arrays"][0]["dtype"] == "bytes"
    assert manifest["attrs_canonical_sha256"]


def test_compare_semantic_manifests_reports_missing_and_unexpected_nested_keys() -> (
    None
):
    expected = {
        "schema_version": "truth_manifest.v1",
        "case_id": "edge_case",
        "zarr_info": {
            "zarr_format": "v2",
            "chunk_key_encoding": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
        },
        "attrs_canonical_sha256": "1" * 64,
        "arrays": [],
        "coords": [],
        "dataset_sha256": "2" * 64,
    }
    candidate = {
        "schema_version": "truth_manifest.v1",
        "case_id": "edge_case",
        "zarr_info": {
            "zarr_format": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
            "extra_key": "unexpected",
        },
        "attrs_canonical_sha256": "1" * 64,
        "arrays": [],
        "coords": [],
        "dataset_sha256": "2" * 64,
    }
    diffs = compare_semantic_manifests(expected, candidate)
    joined = "\n".join(diffs)
    assert "zarr_info.chunk_key_encoding: missing key in candidate" in joined
    assert "zarr_info.extra_key: unexpected key in candidate" in joined


def test_backend_registry_rejects_unsupported_backend_kind() -> None:
    registry = BackendRunnerRegistry()
    with pytest.raises(ValueError, match="unsupported backend_kind"):
        registry.register("unsupported_backend", lambda _case_spec, _path: None)


def test_backend_registry_has_runner_reports_state() -> None:
    registry = BackendRunnerRegistry()
    assert registry.has_runner("py_sync") is False
    registry.register("py_sync", lambda _case_spec, _path: None)
    assert registry.has_runner("py_sync") is True


def test_workflow_catalog_rejects_empty_workflow_id() -> None:
    catalog = WorkflowCatalog()
    with pytest.raises(ValueError, match="workflow_id must be non-empty"):
        catalog.register("", lambda _case_spec, _path: None)


def test_workflow_catalog_rejects_invalid_case_spec_workflow_id(tmp_path: Path) -> None:
    catalog = WorkflowCatalog()
    with pytest.raises(
        ValueError, match="case_spec.workflow_id must be a non-empty string"
    ):
        catalog.run({"workflow_id": None}, tmp_path / "unused")


def test_read_dataset_content_raises_for_missing_dataset_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset path does not exist"):
        read_dataset_content(tmp_path / "missing.zarr", _base_case_spec())


def test_read_dataset_content_raises_for_missing_output_array_name(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "missing_output.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
    )
    bad_case_spec = _base_case_spec()
    bad_case_spec["output_array_names"] = ["not_present"]
    with pytest.raises(RuntimeError, match="failed to read output array"):
        read_dataset_content(dataset_path, bad_case_spec)


def test_read_dataset_content_raises_for_invalid_output_array_names(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "invalid_output_names.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
    )
    bad_case_spec = _base_case_spec()
    bad_case_spec["output_array_names"] = []
    with pytest.raises(ValueError, match="output_array_names"):
        read_dataset_content(dataset_path, bad_case_spec)


def test_read_dataset_content_infers_coords_and_format_when_case_spec_invalid(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "infer_coords.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "infer"},
    )
    case_spec = _base_case_spec()
    case_spec["coord_names"] = None
    arrays, coords, attrs, zarr_info = read_dataset_content(dataset_path, case_spec)
    assert "temperature" in arrays
    assert "time" in coords
    assert attrs["source"] == "infer"
    if (dataset_path / "zarr.json").exists():
        assert zarr_info["zarr_format"] == "v3"
    else:
        assert zarr_info["zarr_format"] == "v2"


def test_build_manifest_from_dataset_requires_case_id(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    dataset_path = tmp_path / "missing_case_id.zarr"
    create_test_zarr_dataset(
        dataset_path=dataset_path,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
    )
    case_spec = _base_case_spec()
    case_spec["case_id"] = ""
    with pytest.raises(
        ValueError, match="case_spec.case_id must be a non-empty string"
    ):
        build_manifest_from_dataset(
            dataset_path=dataset_path,
            case_spec=case_spec,
            generated_by_backend="rust",
        )
