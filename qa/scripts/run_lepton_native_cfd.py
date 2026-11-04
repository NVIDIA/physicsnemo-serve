#!/usr/bin/env python3
"""Run full native CFD compile/field QA or final-image HTTP QA on one Lepton H100.

Only this wrapper selects the CFD driver and gate contract. Job ownership,
finite deadlines, interruption handling and cleanup use the component QA
controller. It neither builds an image locally nor starts a second GPU job.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402


COMPILE_CASES = (
    "preflight",
    "assets",
    "python_reference",
    "model_import",
    "compile60000",
    "compile8095",
    "native_runner",
    "native_fields",
    "shared_evaluator",
    "parity",
)
HTTP_CASES = (
    "preflight",
    "start_stack",
    "readiness",
    "first_request",
    "second_request",
    "redis_completion",
    "artifact_downloads",
    "native_fields",
    "shared_evaluator",
    "parity",
    "shutdown",
)
MARKERS = {
    "compile": ("NATIVE_CFD_QA_SUMMARY_BEGIN", "NATIVE_CFD_QA_SUMMARY_END"),
    "http": ("NATIVE_CFD_HTTP_QA_SUMMARY_BEGIN", "NATIVE_CFD_HTTP_QA_SUMMARY_END"),
}
IMAGE_ROOT = "/opt/native-cfd-qa"


def build_parser():
    parser = lifecycle.build_parser(
        profile_flag="--mode", profile_choices=("compile", "http")
    )
    parser.description = __doc__
    parser.set_defaults(
        artifact_dir=lifecycle.REPO_ROOT / "qa/artifacts/native-cfd",
        job_timeout=7200,
    )
    parser.add_argument(
        "--reference",
        help="Qualified Python fields.json under the remote mount (HTTP mode)",
    )
    parser.add_argument(
        "--input-root",
        help="Verified run_1 dataset root under the remote mount (HTTP mode)",
    )
    parser.add_argument(
        "--asset-cache", help="Reusable compile assets directory under the remote mount"
    )
    parser.add_argument(
        "--runtime", default="/opt/physicsnemo-inference/bin/physicsnemo-infer"
    )
    return parser


def validate_args(args):
    lifecycle.validate_args(args)
    if not 60 < args.job_timeout <= 7200:
        raise ValueError(
            "full CFD qualification deadline must exceed 60 and be at most 7200 seconds"
        )
    if args.assets:
        raise ValueError("CFD uses --asset-cache, not the component --assets manifest")
    if args.profile not in MARKERS:
        raise ValueError("unsupported native CFD mode")
    if args.profile == "http" and (not args.reference or not args.input_root):
        raise ValueError("HTTP mode requires --reference and --input-root")
    if args.profile == "http" and args.mount_target != "/outputs":
        raise ValueError("HTTP release artifacts require the persistent /outputs mount")
    if args.profile == "compile" and (args.reference or args.input_root):
        raise ValueError("--reference and --input-root apply only to HTTP mode")
    for name in ("reference", "input_root", "asset_cache"):
        value = getattr(args, name)
        if value:
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or not path.is_relative_to(PurePosixPath(args.mount_target))
                or path == PurePosixPath(args.mount_target)
            ):
                raise ValueError(
                    f"--{name.replace('_', '-')} must be below --mount-target"
                )
    runtime = PurePosixPath(args.runtime)
    if not runtime.is_absolute() or ".." in runtime.parts:
        raise ValueError(
            "--runtime must be an absolute executable path inside the image"
        )


def extract_summary(output, mode):
    begin_marker, end_marker = MARKERS[mode]
    begin = output.rfind(begin_marker)
    end = output.find(end_marker, begin + len(begin_marker))
    if begin < 0 or end < 0:
        raise ValueError("job logs contain no complete native CFD summary")
    report = json.loads(output[begin + len(begin_marker) : end].strip())
    if not isinstance(report, dict):
        raise ValueError("native CFD summary must be an object")
    return report


def validate_summary(report, *, run_id, source_sha, image_digest, profile):
    if profile not in MARKERS:
        raise ValueError("unsupported native CFD mode")
    cases = COMPILE_CASES if profile == "compile" else HTTP_CASES
    expected = {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "status": "passed",
        "expected_cases": list(cases),
        "qualification": "standalone-persistent-runner"
        if profile == "compile"
        else "native-http-fields-and-metrics-v1",
    }
    if profile == "compile":
        expected.update(
            profile_id="physicsnemo-cfd-surface-transolver-run1-v1",
            serve_http_tested=False,
        )
    else:
        expected["compiled_models_rebuilt"] = False
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report.get(key) != value:
            raise ValueError(
                f"native CFD summary {key} does not match the requested job"
            )
    records = report.get("cases", [])
    if (
        not isinstance(records, list)
        or any(not isinstance(row, dict) for row in records)
        or [row.get("name") for row in records] != list(cases)
    ):
        raise ValueError(
            "native CFD case inventory must match every required gate in order"
        )
    for row in records:
        seconds = row.get("seconds")
        if (
            row.get("status") != "passed"
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("native CFD gates must pass with finite elapsed times")


def build_job_command(args, job_name):
    command = lifecycle.build_job_command(args, job_name)
    output = f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}"
    runner = [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{math.floor(args.job_timeout) - 60}s",
        "/opt/nvidia/nvidia_entrypoint.sh",
        "python3",
        "-m",
        "qa.scripts.run_native_cfd_job"
        if args.profile == "compile"
        else "qa.scripts.run_native_cfd_http_job",
        "--run-id",
        args.run_id,
        "--output",
        output,
        "--profile",
        f"{IMAGE_ROOT}/qa/inference/cfd_parity_surface_run1_transolver.json",
        "--request",
        f"{IMAGE_ROOT}/plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json",
        "--image",
        args.image,
        "--source-sha",
        args.expected_source_sha,
        "--command-timeout",
        str(math.floor(args.job_timeout)),
    ]
    if args.profile == "compile":
        runner += [
            "--asset-cache",
            args.asset_cache or f"{args.mount_target.rstrip('/')}/assets/transolver",
            "--runtime",
            args.runtime,
        ]
    else:
        runner += ["--reference", args.reference, "--input-root", args.input_root]
    command[command.index("--command") + 1] = shlex.join(runner)
    command += ["--shared-memory-size", "16384"]
    return command


def run(args):
    validate_args(args)
    return lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=lambda output: extract_summary(output, args.profile),
        summary_validator=validate_summary,
        historical_first=True,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = run(args)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
