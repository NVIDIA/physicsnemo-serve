# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for manifest compare I/O and reporting helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
import types

import pytest

from parity.utils.backend_runners import create_default_backend_runner_registry
from parity.utils.manifest_compare import (
    compare_semantic_manifests,
    load_manifest,
    write_manifest,
)
from parity.utils.report import build_baseline_report
from parity.utils.workflow_catalog import (
    WorkflowCatalog,
    create_default_workflow_catalog,
)


def _manifest(payload_sha: str, backend: str) -> dict[str, object]:
    return {
        "schema_version": "truth_manifest.v1",
        "case_id": "case_one",
        "generated_by_backend": backend,
        "zarr_info": {
            "zarr_format": "v2",
            "chunk_key_encoding": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
        },
        "truth_provenance": {
            "generation_mode": "manual_tier0",
            "earth2studio_commit": "abc1234",
            "python_version": "3.12.4",
            "zarr_python_version": "2.18.3",
            "case_spec_sha256": "0" * 64,
            "generated_at_utc": "2026-02-10T00:00:00Z",
        },
        "attrs_canonical_sha256": "1" * 64,
        "arrays": [
            {
                "name": "temperature",
                "dtype": "float32",
                "shape": [1, 2],
                "order": "C",
                "payload_sha256": payload_sha,
                "nan_count": None,
                "finite_min": None,
                "finite_max": None,
            }
        ],
        "coords": [
            {"name": "time", "dtype": "int64", "shape": [1], "payload_sha256": "2" * 64}
        ],
        "dataset_sha256": "3" * 64,
    }


def test_semantic_compare_ignores_non_semantic_backend_tag() -> None:
    left = _manifest("4" * 64, backend="py_sync")
    right = _manifest("4" * 64, backend="rust")
    assert compare_semantic_manifests(left, right) == []


def test_semantic_compare_reports_payload_difference() -> None:
    left = _manifest("4" * 64, backend="py_sync")
    right = _manifest("5" * 64, backend="rust")
    diffs = compare_semantic_manifests(left, right)
    assert diffs
    assert "payload_sha256" in diffs[0]


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = _manifest("4" * 64, backend="py_sync")
    out_path = tmp_path / "manifest.json"
    write_manifest(out_path, manifest)
    loaded = load_manifest(out_path)
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]


def test_manifest_zst_roundtrip_if_zstandard_available(tmp_path: Path) -> None:
    manifest = _manifest("4" * 64, backend="py_sync")
    out_path = tmp_path / "manifest.json.zst"
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        with pytest.raises(RuntimeError, match="zstandard is required"):
            write_manifest(out_path, manifest)
        return
    del zstandard
    write_manifest(out_path, manifest)
    loaded = load_manifest(out_path)
    assert loaded["case_id"] == "case_one"


def test_baseline_report_contains_diff_preview() -> None:
    py_sync = _manifest("4" * 64, backend="py_sync")
    py_async = _manifest("5" * 64, backend="py_async")
    report = build_baseline_report(
        case_id="case_one", py_sync_manifest=py_sync, py_async_manifest=py_async
    )
    assert report["case_id"] == "case_one"
    assert report["py_baseline_equal"] is False
    assert report["diff_count"] >= 1
    assert report["diff_preview"]


def test_workflow_catalog_register_resolve_and_run(tmp_path: Path) -> None:
    marker = {"called": False}

    def runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        marker["called"] = True
        assert case_spec["workflow_id"] == "demo_workflow"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("ok", encoding="utf-8")

    catalog = WorkflowCatalog()
    catalog.register("demo_workflow", runner)
    case_spec = {"workflow_id": "demo_workflow"}
    output = tmp_path / "wf" / "dataset.txt"
    catalog.run(case_spec, output)
    assert marker["called"] is True
    assert output.read_text(encoding="utf-8") == "ok"


def test_workflow_catalog_raises_for_unknown_workflow() -> None:
    catalog = WorkflowCatalog()
    with pytest.raises(KeyError, match="unknown workflow_id"):
        catalog.resolve("missing_workflow")


def test_workflow_catalog_run_with_backend_injects_backend_kind(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        captured["backend_kind"] = case_spec.get("backend_kind")
        dataset_path.write_text("ok", encoding="utf-8")

    catalog = WorkflowCatalog()
    catalog.register("demo_workflow", runner)
    case_spec = {"workflow_id": "demo_workflow"}
    output = tmp_path / "run_with_backend.txt"
    catalog.run_with_backend("py_sync", case_spec, output)
    assert captured["backend_kind"] == "py_sync"
    assert output.read_text(encoding="utf-8") == "ok"


def test_default_workflow_rust_backend_invokes_extension_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    from parity.utils import workflow_catalog as workflow_catalog_module

    calls: dict[str, object] = {"writes": 0}

    class FakeRustBackend:
        def __init__(self, **kwargs: object) -> None:
            calls["init_kwargs"] = kwargs
            calls["closed"] = False

        def add_array(self, coords: object, array_name: object) -> None:
            calls["add_array"] = (coords, array_name)

        def write(self, x: object, coords: object, array_name: object) -> None:
            calls["writes"] = int(calls["writes"]) + 1
            calls["last_write"] = (x, coords, array_name)

        def close(self) -> None:
            calls["closed"] = True

    fake_module = types.SimpleNamespace(E2sZarrIoBackend=FakeRustBackend)
    monkeypatch.setitem(sys.modules, "e2s_zarr_io", fake_module)
    monkeypatch.setattr(
        workflow_catalog_module, "_ensure_earth2studio_import_path", lambda: None
    )

    total_coords = OrderedDict(
        {
            "time": np.asarray([1], dtype=np.int64),
            "lead_time": np.asarray([0, 1], dtype=np.int64),
            "lat": np.asarray([10.0, 20.0], dtype=np.float32),
            "lon": np.asarray([30.0, 40.0], dtype=np.float32),
        }
    )
    arrays = {
        "temperature": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "pressure": np.ones((1, 2, 2, 2), dtype=np.float32),
    }
    monkeypatch.setattr(
        workflow_catalog_module,
        "_build_workflow_inputs",
        lambda _case_spec: (total_coords, arrays),
    )

    case_spec: dict[str, object] = {
        "workflow_id": "deterministic_io_small_v1",
        "backend_kind": "rust",
        "parallel_coords": None,
        "zarr_format": "v2",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
    }
    dataset_path = tmp_path / "rust_case.zarr"
    workflow_catalog_module.run_default_workflow(case_spec, dataset_path)

    init_kwargs = calls.get("init_kwargs")
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["file_name"] == str(dataset_path)
    assert int(calls["writes"]) == 2
    assert calls["closed"] is True


def test_create_default_workflow_catalog_contains_default_id() -> None:
    catalog = create_default_workflow_catalog()
    runner = catalog.resolve("deterministic_io_small_v1")
    assert callable(runner)


def test_create_default_backend_registry_registers_all_backends() -> None:
    catalog = WorkflowCatalog()

    def runner(case_spec: dict[str, object], dataset_path: Path) -> None:
        del case_spec, dataset_path

    catalog.register("deterministic_io_small_v1", runner)
    registry = create_default_backend_runner_registry(workflow_catalog=catalog)
    assert registry.has_runner("py_sync") is True
    assert registry.has_runner("py_async") is True
    assert registry.has_runner("rust") is True
