#!/usr/bin/env python3
"""Qualify the normal Model Builder image payload without nested containers."""

from pathlib import Path, PurePosixPath
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import model_builder_image_job as job  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("builder-image",))
    parser.add_argument(
        "--runner",
        required=True,
        help="Staged standalone driver below --mount-target, with sibling contract.py",
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--sdk-prefix", default="/opt/physicsnemo-inference")
    return parser


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        "python3",
        "-B",
        args.runner,
        "--source-root",
        args.source_root,
        "--source-manifest",
        args.source_manifest,
        "--source-manifest-sha256",
        args.source_manifest_sha256,
        "--expected-runner-sha256",
        job.sha256(job.__file__),
        "--expected-contract-sha256",
        job.sha256(job.contract.__file__),
        "--sdk-prefix",
        args.sdk_prefix,
        "--run-id",
        args.run_id,
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--profile",
        args.profile,
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
    ]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def run(args):
    lifecycle.validate_args(args)
    root = PurePosixPath(args.mount_target)
    for field in ("runner", "source_root", "source_manifest"):
        path = PurePosixPath(getattr(args, field))
        job.require(
            path.is_absolute()
            and ".." not in path.parts
            and path.is_relative_to(root)
            and path != root,
            f"--{field.replace('_', '-')} must be below the shared mount",
        )
    job.require(
        re.fullmatch(r"[0-9a-f]{64}", args.source_manifest_sha256),
        "Source manifest SHA-256 must have 64 lowercase hex characters",
    )

    def validate(report, **identity):
        job.validate_summary(report, **identity)
        environment = next(
            case["details"] for case in report["cases"] if case["name"] == "environment"
        )
        for field, expected in (
            ("runner_sha256", job.sha256(job.__file__)),
            ("contract_sha256", job.sha256(job.contract.__file__)),
            ("source_manifest_sha256", args.source_manifest_sha256),
        ):
            job.require(
                environment.get(field) == expected,
                f"Builder image QA {field} differs from submitted bytes",
            )

    return lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=job.extract_summary,
        summary_validator=validate,
        historical_first=True,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
