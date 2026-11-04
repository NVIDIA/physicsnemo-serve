#!/usr/bin/env python3
"""Stage immutable QA-image source through one ordinary Lepton CPU job."""

from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_stage_job as job  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("stage",))
    parser.set_defaults(resource_shape="cpu.medium")
    return parser


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "python3",
        "/workspace/qa/native_inference/pnms_stage_job.py",
        "--run-id",
        args.run_id,
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--expected-runner-sha256",
        job.sha256(job.__file__),
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
    ]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def run(args):
    def validate(report, **identity):
        job.validate_summary(report, **identity)
        job.require(
            report["runner_sha256"] == job.sha256(job.__file__), "Stage runner differs"
        )
        job.require(
            report["source_root"]
            == f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}/source",
            "Stage source root differs from submitted path",
        )

    return lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=job.extract_summary,
        summary_validator=validate,
        historical_first=True,
    )


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
