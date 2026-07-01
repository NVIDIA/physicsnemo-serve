# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI mode tests for record_truth and verify_parity tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from parity.tools import record_truth, verify_parity
from parity.tools.record_truth import (
    build_blessed_truth_manifest_from_case_spec_runners,
)
from parity.tools.verify_parity import verify_candidate_from_case_spec_runner
from parity.utils.backend_runners import BackendRunnerRegistry
from parity.utils.canonical_reader import build_manifest_from_dataset
from parity.utils.manifest_builder import build_truth_manifest
from parity.utils.manifest_compare import load_manifest, write_manifest

from ._dataset_helpers import create_test_zarr_dataset


class FakeArray:
    """Simple array-like object for deterministic payloads."""

    def __init__(self, payload: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self._payload = payload
        self.shape = shape
        self.dtype = dtype

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise ValueError("only C-order is supported in this fake")
        return self._payload


def _manifest(backend: str, payload: bytes) -> dict[str, object]:
    return build_truth_manifest(
        case_id="cli_case",
        generated_by_backend=backend,
        zarr_info={
            "zarr_format": "v2",
            "chunk_key_encoding": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
        },
        arrays={"temperature": FakeArray(payload, (2, 2), "float32")},
        coords={"time": FakeArray(b"\x01\x02", (2,), "int64")},
        attrs={"source": "cli"},
    )


def _case_spec() -> dict[str, object]:
    return {
        "schema_version": "case_spec.v1",
        "case_id": "cli_case",
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


def _write_case_spec(path: Path) -> None:
    path.write_text(json.dumps(_case_spec()), encoding="utf-8")


def test_record_truth_cli_manifest_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    py_sync_path = tmp_path / "py_sync_manifest.json"
    py_async_path = tmp_path / "py_async_manifest.json"
    out_path = tmp_path / "blessed_manifest.json"
    report_path = tmp_path / "baseline_report.json"
    write_manifest(py_sync_path, _manifest("py_sync", b"\x10\x11\x12\x13"))
    write_manifest(py_async_path, _manifest("py_async", b"\x10\x11\x12\x13"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_truth",
            "--py-sync-manifest",
            str(py_sync_path),
            "--py-async-manifest",
            str(py_async_path),
            "--output",
            str(out_path),
            "--baseline-report-output",
            str(report_path),
            "--earth2studio-commit",
            "abc1234",
            "--python-version",
            "3.12.4",
            "--zarr-python-version",
            "2.18.3",
            "--case-spec-sha256",
            "0" * 64,
        ],
    )
    assert record_truth.main() == 0
    blessed = load_manifest(out_path)
    report = load_manifest(report_path)
    assert blessed["generated_by_backend"] == "py_sync"
    assert report["py_baseline_equal"] is True


def test_record_truth_cli_dataset_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    case_spec_path = tmp_path / "case_spec.json"
    _write_case_spec(case_spec_path)
    py_sync_dataset = tmp_path / "py_sync.zarr"
    py_async_dataset = tmp_path / "py_async.zarr"
    for path in (py_sync_dataset, py_async_dataset):
        create_test_zarr_dataset(
            dataset_path=path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "dataset"},
        )
    out_path = tmp_path / "blessed_dataset_mode.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_truth",
            "--case-spec",
            str(case_spec_path),
            "--py-sync-dataset",
            str(py_sync_dataset),
            "--py-async-dataset",
            str(py_async_dataset),
            "--output",
            str(out_path),
            "--earth2studio-commit",
            "abc1234",
            "--python-version",
            "3.12.4",
            "--zarr-python-version",
            "2.18.3",
            "--case-spec-sha256",
            "0" * 64,
        ],
    )
    assert record_truth.main() == 0
    blessed = load_manifest(out_path)
    assert blessed["case_id"] == "cli_case"


def test_record_truth_cli_rejects_invalid_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_path = tmp_path / "invalid_mode.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_truth",
            "--output",
            str(out_path),
            "--earth2studio-commit",
            "abc1234",
            "--python-version",
            "3.12.4",
            "--zarr-python-version",
            "2.18.3",
            "--case-spec-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(ValueError, match="no mode selected"):
        record_truth.main()


def test_verify_parity_cli_manifest_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    truth_path = tmp_path / "truth.json"
    candidate_path = tmp_path / "candidate.json"
    write_manifest(truth_path, _manifest("py_sync", b"\x10\x11\x12\x13"))
    write_manifest(candidate_path, _manifest("rust", b"\x10\x11\x12\x13"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_parity",
            "--truth-manifest",
            str(truth_path),
            "--candidate-manifest",
            str(candidate_path),
        ],
    )
    assert verify_parity.main() == 0


def test_verify_parity_cli_dataset_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    case_spec_path = tmp_path / "case_spec_verify.json"
    _write_case_spec(case_spec_path)
    truth_dataset = tmp_path / "truth.zarr"
    candidate_dataset = tmp_path / "candidate.zarr"
    for path in (truth_dataset, candidate_dataset):
        create_test_zarr_dataset(
            dataset_path=path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "verify"},
        )
    truth_manifest = build_manifest_from_dataset(
        dataset_path=truth_dataset,
        case_spec=_case_spec(),
        generated_by_backend="py_sync",
    )
    truth_path = tmp_path / "truth_for_dataset_mode.json"
    write_manifest(truth_path, truth_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_parity",
            "--truth-manifest",
            str(truth_path),
            "--case-spec",
            str(case_spec_path),
            "--candidate-dataset",
            str(candidate_dataset),
            "--generated-by-backend",
            "rust",
        ],
    )
    assert verify_parity.main() == 0


def test_verify_parity_cli_rejects_invalid_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    truth_path = tmp_path / "truth_invalid_mode.json"
    write_manifest(truth_path, _manifest("py_sync", b"\x10\x11\x12\x13"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_parity",
            "--truth-manifest",
            str(truth_path),
        ],
    )
    with pytest.raises(ValueError, match="choose exactly one mode"):
        verify_parity.main()


def test_record_truth_runner_mode_helper_uses_registry_outputs(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    registry = BackendRunnerRegistry()

    def py_sync_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner"},
        )

    def py_async_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner"},
        )

    registry.register("py_sync", py_sync_runner)
    registry.register("py_async", py_async_runner)
    registry.register("rust", py_sync_runner)

    blessed, report, outputs = build_blessed_truth_manifest_from_case_spec_runners(
        case_spec=_case_spec(),
        generated_datasets_dir=tmp_path / "runner_mode",
        earth2studio_commit="abc1234",
        python_version="3.12.4",
        zarr_python_version="2.18.3",
        case_spec_sha256="0" * 64,
        runner_registry=registry,
    )
    assert blessed["generated_by_backend"] == "py_sync"
    assert report["py_baseline_equal"] is True
    assert outputs["py_sync_dataset"].endswith(".py_sync.zarr")
    assert outputs["py_async_dataset"].endswith(".py_async.zarr")


def test_verify_parity_runner_mode_helper_uses_registry_outputs(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    registry = BackendRunnerRegistry()

    def rust_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner"},
        )

    registry.register("rust", rust_runner)
    registry.register("py_sync", rust_runner)
    registry.register("py_async", rust_runner)
    truth_dataset = tmp_path / "truth_runner_mode.zarr"
    create_test_zarr_dataset(
        dataset_path=truth_dataset,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "runner"},
    )
    truth_manifest = build_manifest_from_dataset(
        dataset_path=truth_dataset,
        case_spec=_case_spec(),
        generated_by_backend="py_sync",
    )
    candidate_manifest, outputs = verify_candidate_from_case_spec_runner(
        truth_manifest=truth_manifest,
        case_spec=_case_spec(),
        candidate_backend="rust",
        generated_datasets_dir=tmp_path / "runner_verify",
        runner_registry=registry,
    )
    assert candidate_manifest["generated_by_backend"] == "rust"
    assert outputs["candidate_dataset"].endswith(".rust.zarr")


def test_record_truth_cli_runner_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    registry = BackendRunnerRegistry()

    def py_sync_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner_cli"},
        )

    def py_async_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner_cli"},
        )

    registry.register("py_sync", py_sync_runner)
    registry.register("py_async", py_async_runner)
    registry.register("rust", py_sync_runner)
    monkeypatch.setattr(
        record_truth, "create_default_backend_runner_registry", lambda: registry
    )
    case_spec_path = tmp_path / "case_spec_runner_mode.json"
    _write_case_spec(case_spec_path)
    output_path = tmp_path / "runner_mode_blessed.json"
    runner_output_map = tmp_path / "runner_mode_paths.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_truth",
            "--case-spec",
            str(case_spec_path),
            "--generated-datasets-dir",
            str(tmp_path / "runner_datasets"),
            "--output",
            str(output_path),
            "--runner-output-map",
            str(runner_output_map),
            "--earth2studio-commit",
            "abc1234",
            "--python-version",
            "3.12.4",
            "--zarr-python-version",
            "2.18.3",
            "--case-spec-sha256",
            "0" * 64,
        ],
    )
    assert record_truth.main() == 0
    blessed = load_manifest(output_path)
    outputs = load_manifest(runner_output_map)
    assert blessed["case_id"] == "cli_case"
    assert "py_sync_dataset" in outputs


def test_verify_parity_cli_runner_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    registry = BackendRunnerRegistry()

    def rust_runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec
        create_test_zarr_dataset(
            dataset_path=dataset_path,
            arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
            coords={"time": np.array([10], dtype=np.int64)},
            attrs={"source": "runner_cli"},
        )

    registry.register("rust", rust_runner)
    registry.register("py_sync", rust_runner)
    registry.register("py_async", rust_runner)
    monkeypatch.setattr(
        verify_parity, "create_default_backend_runner_registry", lambda: registry
    )
    truth_dataset = tmp_path / "truth_runner_cli.zarr"
    create_test_zarr_dataset(
        dataset_path=truth_dataset,
        arrays={"temperature": np.array([[1.0, 2.0]], dtype=np.float32)},
        coords={"time": np.array([10], dtype=np.int64)},
        attrs={"source": "runner_cli"},
    )
    truth_manifest = build_manifest_from_dataset(
        dataset_path=truth_dataset,
        case_spec=_case_spec(),
        generated_by_backend="py_sync",
    )
    truth_manifest_path = tmp_path / "truth_runner_cli.json"
    write_manifest(truth_manifest_path, truth_manifest)
    case_spec_path = tmp_path / "case_spec_runner_verify.json"
    _write_case_spec(case_spec_path)
    output_map = tmp_path / "runner_verify_paths.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_parity",
            "--truth-manifest",
            str(truth_manifest_path),
            "--case-spec",
            str(case_spec_path),
            "--candidate-backend",
            "rust",
            "--generated-datasets-dir",
            str(tmp_path / "runner_verify_datasets"),
            "--runner-output-map",
            str(output_map),
        ],
    )
    assert verify_parity.main() == 0
    outputs = load_manifest(output_map)
    assert outputs["candidate_dataset"].endswith(".rust.zarr")
