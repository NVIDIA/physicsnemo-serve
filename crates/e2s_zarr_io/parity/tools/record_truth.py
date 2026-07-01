# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual Tier0 baseline tool: bless truth manifest after Python baseline agreement."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parity.utils.manifest_compare import (
    assert_semantic_manifest_equal,
    load_manifest,
    write_manifest,
)
from parity.utils.backend_runners import (
    BackendRunnerRegistry,
    run_backend_and_collect_manifest,
    create_default_backend_runner_registry,
)
from parity.utils.canonical_reader import build_manifest_from_dataset
from parity.utils.case_spec import load_case_spec
from parity.utils.report import build_baseline_report


def build_blessed_truth_manifest(
    *,
    py_sync_manifest: dict[str, Any],
    py_async_manifest: dict[str, Any],
    earth2studio_commit: str,
    python_version: str,
    zarr_python_version: str,
    case_spec_sha256: str,
) -> dict[str, Any]:
    """Return blessed manifest if py_sync and py_async logical data are identical."""
    assert_semantic_manifest_equal(py_sync_manifest, py_async_manifest)
    blessed = deepcopy(py_sync_manifest)
    blessed["truth_provenance"] = {
        "generation_mode": "manual_tier0",
        "earth2studio_commit": earth2studio_commit,
        "python_version": python_version,
        "zarr_python_version": zarr_python_version,
        "case_spec_sha256": case_spec_sha256,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    blessed["generated_by_backend"] = "py_sync"
    return blessed


def build_blessed_truth_manifest_from_datasets(
    *,
    case_spec: dict[str, Any],
    py_sync_dataset_path: str | Path,
    py_async_dataset_path: str | Path,
    earth2studio_commit: str,
    python_version: str,
    zarr_python_version: str,
    case_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build blessed truth manifest and baseline report from dataset paths."""
    py_sync_manifest = build_manifest_from_dataset(
        dataset_path=py_sync_dataset_path,
        case_spec=case_spec,
        generated_by_backend="py_sync",
    )
    py_async_manifest = build_manifest_from_dataset(
        dataset_path=py_async_dataset_path,
        case_spec=case_spec,
        generated_by_backend="py_async",
    )
    blessed = build_blessed_truth_manifest(
        py_sync_manifest=py_sync_manifest,
        py_async_manifest=py_async_manifest,
        earth2studio_commit=earth2studio_commit,
        python_version=python_version,
        zarr_python_version=zarr_python_version,
        case_spec_sha256=case_spec_sha256,
    )
    case_id = case_spec.get("case_id", "")
    report = build_baseline_report(
        case_id=str(case_id),
        py_sync_manifest=py_sync_manifest,
        py_async_manifest=py_async_manifest,
    )
    return blessed, report


def build_blessed_truth_manifest_from_case_spec_runners(
    *,
    case_spec: dict[str, Any],
    generated_datasets_dir: str | Path,
    earth2studio_commit: str,
    python_version: str,
    zarr_python_version: str,
    case_spec_sha256: str,
    runner_registry: BackendRunnerRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Run py_sync/py_async runners from case spec and return blessed manifest + report."""
    case_id = str(case_spec.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("case_spec.case_id must be a non-empty string")
    output_dir = Path(generated_datasets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    py_sync_dataset = output_dir / f"{case_id}.py_sync.zarr"
    py_async_dataset = output_dir / f"{case_id}.py_async.zarr"
    registry = (
        runner_registry
        if runner_registry is not None
        else create_default_backend_runner_registry()
    )
    py_sync_manifest = run_backend_and_collect_manifest(
        registry=registry,
        backend_kind="py_sync",
        case_spec=case_spec,
        dataset_path=py_sync_dataset,
        generated_by_backend="py_sync",
    )
    py_async_manifest = run_backend_and_collect_manifest(
        registry=registry,
        backend_kind="py_async",
        case_spec=case_spec,
        dataset_path=py_async_dataset,
        generated_by_backend="py_async",
    )
    blessed = build_blessed_truth_manifest(
        py_sync_manifest=py_sync_manifest,
        py_async_manifest=py_async_manifest,
        earth2studio_commit=earth2studio_commit,
        python_version=python_version,
        zarr_python_version=zarr_python_version,
        case_spec_sha256=case_spec_sha256,
    )
    report = build_baseline_report(
        case_id=case_id,
        py_sync_manifest=py_sync_manifest,
        py_async_manifest=py_async_manifest,
    )
    outputs = {
        "py_sync_dataset": str(py_sync_dataset),
        "py_async_dataset": str(py_async_dataset),
    }
    return blessed, report, outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bless truth manifest from py_sync/py_async baseline."
    )
    parser.add_argument(
        "--py-sync-manifest", help="Path to py_sync manifest (.json or .json.zst)"
    )
    parser.add_argument(
        "--py-async-manifest", help="Path to py_async manifest (.json or .json.zst)"
    )
    parser.add_argument(
        "--case-spec", help="Path to case spec JSON for dataset-based mode"
    )
    parser.add_argument(
        "--py-sync-dataset", help="Path to py_sync produced Zarr dataset"
    )
    parser.add_argument(
        "--py-async-dataset", help="Path to py_async produced Zarr dataset"
    )
    parser.add_argument(
        "--generated-datasets-dir",
        help="Directory where runner mode writes py_sync/py_async datasets before baseline comparison",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output blessed truth manifest path (.json or .json.zst)",
    )
    parser.add_argument(
        "--baseline-report-output", help="Optional output path for baseline report JSON"
    )
    parser.add_argument(
        "--runner-output-map",
        help="Optional JSON output path recording generated dataset locations in runner mode",
    )
    parser.add_argument(
        "--earth2studio-commit",
        required=True,
        help="Earth2Studio git commit used for baseline run",
    )
    parser.add_argument(
        "--python-version",
        required=True,
        help="Python runtime version used for baseline run",
    )
    parser.add_argument(
        "--zarr-python-version",
        required=True,
        help="Python zarr package version used for baseline run",
    )
    parser.add_argument(
        "--case-spec-sha256", required=True, help="SHA256 of the CaseSpec JSON"
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args()
    manifest_mode = bool(args.py_sync_manifest and args.py_async_manifest)
    dataset_mode = bool(
        args.case_spec and args.py_sync_dataset and args.py_async_dataset
    )
    runner_mode = bool(args.case_spec and args.generated_datasets_dir)
    selected_modes = int(manifest_mode) + int(dataset_mode) + int(runner_mode)
    if selected_modes == 0:
        raise ValueError(
            "no mode selected; provide one of: "
            "(--py-sync-manifest + --py-async-manifest), "
            "(--case-spec + --py-sync-dataset + --py-async-dataset), or "
            "(--case-spec + --generated-datasets-dir)"
        )
    if selected_modes > 1:
        raise ValueError(
            f"{selected_modes} modes selected; provide exactly one of: "
            "(--py-sync-manifest + --py-async-manifest), "
            "(--case-spec + --py-sync-dataset + --py-async-dataset), or "
            "(--case-spec + --generated-datasets-dir). "
            "Overlapping arguments (e.g. --case-spec appears in both dataset "
            "and runner modes) can activate multiple modes at once."
        )

    if manifest_mode:
        py_sync_manifest = load_manifest(args.py_sync_manifest)
        py_async_manifest = load_manifest(args.py_async_manifest)
        blessed = build_blessed_truth_manifest(
            py_sync_manifest=py_sync_manifest,
            py_async_manifest=py_async_manifest,
            earth2studio_commit=args.earth2studio_commit,
            python_version=args.python_version,
            zarr_python_version=args.zarr_python_version,
            case_spec_sha256=args.case_spec_sha256,
        )
        report = build_baseline_report(
            case_id=str(blessed.get("case_id", "")),
            py_sync_manifest=py_sync_manifest,
            py_async_manifest=py_async_manifest,
        )
        outputs: dict[str, str] | None = None
    elif dataset_mode:
        case_spec = load_case_spec(args.case_spec)
        blessed, report = build_blessed_truth_manifest_from_datasets(
            case_spec=case_spec,
            py_sync_dataset_path=args.py_sync_dataset,
            py_async_dataset_path=args.py_async_dataset,
            earth2studio_commit=args.earth2studio_commit,
            python_version=args.python_version,
            zarr_python_version=args.zarr_python_version,
            case_spec_sha256=args.case_spec_sha256,
        )
        outputs = None
    else:
        case_spec = load_case_spec(args.case_spec)
        blessed, report, outputs = build_blessed_truth_manifest_from_case_spec_runners(
            case_spec=case_spec,
            generated_datasets_dir=args.generated_datasets_dir,
            earth2studio_commit=args.earth2studio_commit,
            python_version=args.python_version,
            zarr_python_version=args.zarr_python_version,
            case_spec_sha256=args.case_spec_sha256,
        )
    write_manifest(args.output, blessed)
    if args.baseline_report_output:
        write_manifest(args.baseline_report_output, report)
    if args.runner_output_map and outputs is not None:
        write_manifest(args.runner_output_map, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
