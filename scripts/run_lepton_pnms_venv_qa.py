#!/usr/bin/env python3
"""Submit fresh isolated Python-venv/native-workflow QA to an ordinary Lepton job."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_venv_job as job  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("fresh-venv",))
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--source-metadata", default="/opt/physicsnemo-qa/source.json")
    parser.add_argument("--requirements", action="append", required=True)
    parser.add_argument("--requirements-sha256", action="append", required=True)
    parser.add_argument("--wheelhouse")
    parser.add_argument("--tensorrt-root", default="/usr")
    parser.add_argument("--tensorrt-cuda-major", choices=("12", "13"), default="13")
    parser.add_argument("--build-command-timeout", type=float, default=2400)
    return parser


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        args.python,
        "-B",
        str(Path(args.repository) / "qa/native_inference/pnms_venv_job.py"),
        "--repository",
        args.repository,
        "--python",
        args.python,
        "--source-metadata",
        args.source_metadata,
        "--profile",
        "fresh-venv",
        "--run-id",
        args.run_id,
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--tensorrt-root",
        args.tensorrt_root,
        "--tensorrt-cuda-major",
        args.tensorrt_cuda_major,
        "--command-timeout",
        str(args.build_command_timeout),
    ]
    for path in args.requirements:
        runner.extend(["--requirements", path])
    for digest in args.requirements_sha256:
        runner.extend(["--requirements-sha256", digest])
    if args.wheelhouse:
        runner.extend(["--wheelhouse", args.wheelhouse])
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def run(args):
    if len(args.requirements) != len(args.requirements_sha256) or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in args.requirements_sha256
    ):
        raise ValueError("Each requirements file needs one expected SHA-256 digest")
    for value in [*args.requirements, *([args.wheelhouse] if args.wheelhouse else [])]:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to(PurePosixPath(args.mount_target))
        ):
            raise ValueError(
                "Requirements/wheelhouse must be staged inside the job artifact mount"
            )
    return lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=job.extract_summary,
        summary_validator=job.validate_summary,
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
