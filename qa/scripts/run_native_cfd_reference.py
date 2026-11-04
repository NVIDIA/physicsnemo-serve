#!/usr/bin/env python3
"""Capture pinned Transolver inference, evaluate native fields, or compare both.

Run capture in the provider Python 3.12 environment, for example:
  python -m qa.scripts.run_native_cfd_reference capture --help

The capture and evaluator write fresh directories. Field comparisons need only
the repository's CPU QA dependencies; neither imports Torch or the provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from qa.native_inference.cfd_contract import (
    CASE_PINS,
    FieldContractError,
    compare_evidence,
    validate_evidence,
    validate_reference_request,
    validate_report,
    verify_file,
)
from qa.native_inference.cfd_reference import (
    ProviderCapture,
    build_capture_config,
    local_model_assets,
    verify_provider,
)
from qa.scripts.cfd_parity_contract import (
    compare_reports,
    read_json_object,
    sha256_file,
    write_json_atomic,
)


def _identity(image: str, source_sha: str) -> None:
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        raise FieldContractError(
            "--image must identify the actual immutable image digest"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise FieldContractError("--source-sha must be a full Git SHA")


def _verify_inputs(request, root: Path) -> None:
    for case in request["cases"]:
        case_number = case["case_id"].removeprefix("run_")
        case_root = root / case["case_id"]
        pin = CASE_PINS[case["case_id"]]
        verify_file(
            case_root / f"boundary_{case_number}.vtp",
            {"sha256": pin["sha256"], "size_bytes": pin["size_bytes"]},
        )
        verify_file(
            case_root / f"drivaer_{case_number}.stl",
            {
                "sha256": pin["geometry_sha256"],
                "size_bytes": pin["geometry_size_bytes"],
            },
        )


def _run_benchmark(profile, request, root, input_root, *, checkpoint=None, stats=None):
    from physicsnemo.cfd.evaluation.config import load_config
    from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark

    config = build_capture_config(
        profile,
        request,
        input_root,
        root / "benchmark-output",
        model_assets=local_model_assets(checkpoint, stats),
    )
    path = root / "resolved_config.json"
    write_json_atomic(path, config)
    return run_benchmark(load_config(str(path), {}))


def capture(args) -> None:
    profile, request = read_json_object(args.profile), read_json_object(args.request)
    validate_reference_request(profile, request)
    _identity(args.image, args.source_sha)
    provider = verify_provider(profile)
    _verify_inputs(request, args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    observer = ProviderCapture(
        profile=profile,
        request=request,
        root=args.output,
        image=args.image,
        source_sha=args.source_sha,
    )
    observer.manifest["provider"] = provider
    with observer.observe():
        report = _run_benchmark(
            profile,
            request,
            args.output,
            args.input_root,
            checkpoint=args.checkpoint,
            stats=args.stats,
        )
    print(observer.finish(report))


def evaluate(args) -> None:
    from qa.native_inference.cfd_evaluate import physical_field_predictions

    profile, request = read_json_object(args.profile), read_json_object(args.request)
    validate_reference_request(profile, request)
    _identity(args.image, args.source_sha)
    provider = verify_provider(profile)
    evidence = read_json_object(args.fields)
    fields = validate_evidence(profile, evidence, args.fields.parent)
    _verify_inputs(request, args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    with physical_field_predictions(fields):
        report = _run_benchmark(
            profile,
            request,
            args.output,
            args.input_root,
            checkpoint=args.checkpoint,
            stats=args.stats,
        )
    validate_report(profile, report)
    report_path = args.output / "benchmark-output" / "benchmark_results.json"
    if json.loads(report_path.read_text()) != report:
        raise FieldContractError(
            "saved evaluation report differs from provider return value"
        )
    write_json_atomic(
        args.output / "evaluation.json",
        {
            "schema_version": 1,
            "profile_id": profile["profile_id"],
            "mode": "external-shared-evaluator",
            "inference_executed": False,
            "fields_sha256": sha256_file(args.fields),
            "provider": provider,
            "image": args.image,
            "source_sha": args.source_sha,
            "report": {
                "path": "benchmark-output/benchmark_results.json",
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        },
    )
    print(args.output / "evaluation.json")


def compare(args) -> int:
    profile = read_json_object(args.profile)
    result = compare_evidence(profile, args.reference, args.candidate)
    if args.candidate_evaluation:
        evaluation = read_json_object(args.candidate_evaluation)
        if (
            evaluation.get("mode") != "external-shared-evaluator"
            or evaluation.get("inference_executed") is not False
        ):
            raise FieldContractError("candidate evaluator provenance is invalid")
        if evaluation.get("fields_sha256") != sha256_file(args.candidate):
            raise FieldContractError("evaluation does not belong to candidate fields")
        if (
            evaluation.get("profile_id") != profile["profile_id"]
            or evaluation.get("provider") != profile["provider"]
        ):
            raise FieldContractError("evaluation profile/provider mismatch")
        _identity(evaluation["image"], evaluation["source_sha"])
        reference = read_json_object(args.reference)
        from qa.native_inference.cfd_contract import _artifact_path

        ref_path = _artifact_path(args.reference.parent, reference["report"]["path"])
        value_path = _artifact_path(
            args.candidate_evaluation.parent, evaluation["report"]["path"]
        )
        verify_file(ref_path, reference["report"])
        verify_file(value_path, evaluation["report"])
        ref_report = json.loads(ref_path.read_text())
        value_report = json.loads(value_path.read_text())
        validate_report(profile, ref_report)
        validate_report(profile, value_report)
        comparison = compare_reports(
            rest_report=ref_report,
            direct_report=value_report,
            comparison=profile["comparison"],
        )
        result["scientific_reports_compared"] = True
        result["reports"] = comparison
        result["passed"] = result["passed"] and comparison["status"] == "passed"
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


def native(args) -> None:
    from qa.native_inference.cfd_native import package_native_fields

    _identity(args.image, args.source_sha)
    package_native_fields(
        profile=read_json_object(args.profile),
        metadata_path=args.metadata,
        receipt_path=args.receipt,
        input_dumps=args.input_dumps,
        output_path=args.output,
        image=args.image,
        source_sha=args.source_sha,
    )
    print(args.output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("capture", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--profile", required=True, type=Path)
        sub.add_argument("--request", required=True, type=Path)
        sub.add_argument("--input-root", required=True, type=Path)
        sub.add_argument("--checkpoint", type=Path)
        sub.add_argument("--stats", type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--image", required=True)
        sub.add_argument("--source-sha", required=True)
        if command == "evaluate":
            sub.add_argument("--fields", required=True, type=Path)
    sub = commands.add_parser("compare")
    sub.add_argument("--profile", required=True, type=Path)
    sub.add_argument("--reference", required=True, type=Path)
    sub.add_argument("--candidate", required=True, type=Path)
    sub.add_argument("--candidate-evaluation", type=Path)
    sub.add_argument("--output", required=True, type=Path)
    sub = commands.add_parser("native")
    for name in ("profile", "metadata", "receipt", "input-dumps", "output"):
        sub.add_argument(f"--{name}", required=True, type=Path)
    sub.add_argument("--image", required=True)
    sub.add_argument("--source-sha", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "compare":
        return compare(args)
    {"capture": capture, "evaluate": evaluate, "native": native}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
