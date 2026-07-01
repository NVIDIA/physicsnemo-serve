# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier1+ parity verifier tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from parity.utils.backend_runners import (
    BackendRunnerRegistry,
    run_backend_and_collect_manifest,
    create_default_backend_runner_registry,
)
from parity.utils.canonical_reader import build_manifest_from_dataset
from parity.utils.case_spec import load_case_spec
from parity.utils.manifest_compare import assert_semantic_manifest_equal, load_manifest


def verify_candidate_against_truth(
    *, truth_manifest: dict[str, Any], candidate_manifest: dict[str, Any]
) -> None:
    """Assert candidate logical parity against committed truth manifest."""
    assert_semantic_manifest_equal(truth_manifest, candidate_manifest)


def verify_candidate_dataset_against_truth(
    *,
    truth_manifest: dict[str, Any],
    case_spec: dict[str, Any],
    candidate_dataset_path: str,
    generated_by_backend: str = "rust",
) -> dict[str, Any]:
    """Build candidate manifest from dataset path and verify parity against truth."""
    candidate_manifest = build_manifest_from_dataset(
        dataset_path=candidate_dataset_path,
        case_spec=case_spec,
        generated_by_backend=generated_by_backend,
    )
    verify_candidate_against_truth(
        truth_manifest=truth_manifest, candidate_manifest=candidate_manifest
    )
    return candidate_manifest


def verify_candidate_from_case_spec_runner(
    *,
    truth_manifest: dict[str, Any],
    case_spec: dict[str, Any],
    candidate_backend: str,
    generated_datasets_dir: str | Path,
    runner_registry: BackendRunnerRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run candidate backend from case spec, then verify against truth manifest."""
    case_id = str(case_spec.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("case_spec.case_id must be a non-empty string")
    output_dir = Path(generated_datasets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dataset = output_dir / f"{case_id}.{candidate_backend}.zarr"
    registry = (
        runner_registry
        if runner_registry is not None
        else create_default_backend_runner_registry()
    )
    candidate_manifest = run_backend_and_collect_manifest(
        registry=registry,
        backend_kind=candidate_backend,
        case_spec=case_spec,
        dataset_path=candidate_dataset,
        generated_by_backend=candidate_backend,
    )
    verify_candidate_against_truth(
        truth_manifest=truth_manifest, candidate_manifest=candidate_manifest
    )
    outputs = {"candidate_dataset": str(candidate_dataset)}
    return candidate_manifest, outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify candidate manifest against committed truth."
    )
    parser.add_argument(
        "--truth-manifest", required=True, help="Path to committed truth manifest"
    )
    parser.add_argument(
        "--candidate-manifest", help="Path to candidate backend manifest"
    )
    parser.add_argument("--case-spec", help="Path to case spec JSON for dataset mode")
    parser.add_argument("--candidate-dataset", help="Path to candidate backend dataset")
    parser.add_argument(
        "--candidate-backend",
        choices=["rust", "py_sync", "py_async"],
        help="Candidate backend kind to run in runner mode",
    )
    parser.add_argument(
        "--generated-datasets-dir",
        help="Directory where runner mode writes candidate dataset",
    )
    parser.add_argument(
        "--runner-output-map",
        help="Optional JSON output path recording generated dataset location in runner mode",
    )
    parser.add_argument(
        "--generated-by-backend",
        default="rust",
        choices=["rust", "py_sync", "py_async"],
        help="Backend tag used when building candidate manifest from dataset mode",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args()
    truth_manifest = load_manifest(args.truth_manifest)
    manifest_mode = bool(args.candidate_manifest)
    dataset_mode = bool(args.case_spec and args.candidate_dataset)
    runner_mode = bool(
        args.case_spec and args.candidate_backend and args.generated_datasets_dir
    )
    selected_modes = int(manifest_mode) + int(dataset_mode) + int(runner_mode)
    if selected_modes != 1:
        raise ValueError(
            "choose exactly one mode: "
            "--candidate-manifest or "
            "(--case-spec + --candidate-dataset) or "
            "(--case-spec + --candidate-backend + --generated-datasets-dir)"
        )
    if manifest_mode:
        candidate_manifest = load_manifest(args.candidate_manifest)
        verify_candidate_against_truth(
            truth_manifest=truth_manifest, candidate_manifest=candidate_manifest
        )
        outputs: dict[str, str] | None = None
    elif dataset_mode:
        case_spec = load_case_spec(args.case_spec)
        verify_candidate_dataset_against_truth(
            truth_manifest=truth_manifest,
            case_spec=case_spec,
            candidate_dataset_path=args.candidate_dataset,
            generated_by_backend=args.generated_by_backend,
        )
        outputs = None
    else:
        case_spec = load_case_spec(args.case_spec)
        _, outputs = verify_candidate_from_case_spec_runner(
            truth_manifest=truth_manifest,
            case_spec=case_spec,
            candidate_backend=args.candidate_backend,
            generated_datasets_dir=args.generated_datasets_dir,
        )
    if args.runner_output_map and outputs is not None:
        from parity.utils.manifest_compare import write_manifest

        write_manifest(args.runner_output_map, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
