#!/usr/bin/env python3
"""Submit public pnms local-workflow acceptance to a finite Lepton H100 job."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_local_job as job  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("local",))
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--sdk-prefix", default="/opt/physicsnemo-inference")
    parser.add_argument("--source-metadata", default="/opt/physicsnemo-qa/source.json")
    return parser


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        args.python,
        str(Path(args.repository) / "qa/native_inference/pnms_local_job.py"),
        "--repository",
        args.repository,
        "--sdk-prefix",
        args.sdk_prefix,
        "--source-metadata",
        args.source_metadata,
        "--profile",
        "local",
        "--run-id",
        args.run_id,
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
    ]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def run(args):
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
