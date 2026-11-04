#!/usr/bin/env python3
"""Run native CPU component tests in Lepton and collect sealed original reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qa.native_inference.components import (  # noqa: E402
    QUALIFICATION,
    SUMMARY_BEGIN,
    SUMMARY_END,
    validate_summary,
)
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402

sys.path.insert(0, str(lifecycle.REPO_ROOT / "native-inference/deploy"))
from native_release import ProcessRunner, read_json, require, write_json  # noqa: E402
from native_release_lepton import fetch_artifact  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("full",))
    parser.description = __doc__
    parser.set_defaults(
        resource_shape="cpu.medium",
        job_timeout=7200,
        artifact_dir=lifecycle.REPO_ROOT / "qa/artifacts/native-components",
    )
    parser.add_argument("--artifact-timeout", type=int, default=1200)
    return parser


def validate_args(args):
    lifecycle.validate_args(args)
    require(args.profile == "full", "unsupported component profile")
    require(
        args.resource_shape.startswith("cpu."),
        "component QA requires a CPU resource shape",
    )
    require(
        60 < args.job_timeout <= 7200,
        "component job deadline must exceed 60s and be at most 7200s",
    )
    require(
        0 < args.artifact_timeout <= 1800, "finite component artifact timeout required"
    )
    require(not args.assets, "component QA uses image-baked source, not model assets")
    require(not args.keep_job, "component qualification must remove its owned job")


def extract_summary(output):
    first = output.rfind(SUMMARY_BEGIN)
    last = output.find(SUMMARY_END, first + len(SUMMARY_BEGIN))
    require(first >= 0 and last >= 0, "job logs have no complete component summary")
    value = json.loads(output[first + len(SUMMARY_BEGIN) : last].strip())
    require(isinstance(value, dict), "component summary must be an object")
    return value


def build_job_command(args, job_name):
    command = lifecycle.build_job_command(args, job_name)
    seconds = math.floor(args.job_timeout) - 60
    runner = [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=60s",
        str(seconds) + "s",
        "python3",
        "-m",
        "qa.native_inference.components",
        "--profile",
        args.profile,
        "--run-id",
        args.run_id,
        "--output",
        args.mount_target.rstrip("/") + "/native-inference/" + args.run_id,
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--timeout-seconds",
        str(seconds),
        "--resource-shape",
        args.resource_shape,
    ]
    command[command.index("--command") + 1] = shlex.join(
        ["/bin/sh", "-c", "cd /workspace && " + shlex.join(runner)]
    )
    return command


def run(args):
    validate_args(args)
    root = args.artifact_dir.resolve() / args.run_id

    def validated(report, **identity):
        validate_summary(report, **identity)
        controller = read_json(root / "summary.json")
        env = {
            **os.environ,
            "LEPTON_WORKSPACE_ID": args.workspace_id,
            "NO_COLOR": "1",
            "COLUMNS": "1000000",
        }
        if args.workspace_url:
            env["LEPTON_WORKSPACE_URL"] = args.workspace_url
        code, output = lifecycle.lep(
            ["job", "nodes", "-i", controller["job_id"]],
            env=env,
            timeout=min(30, args.command_timeout),
        )
        first, last = output.find("["), output.rfind("]")
        nodes = (
            json.loads(output[first : last + 1])
            if code == 0 and first >= 0 and last >= first
            else []
        )
        require(
            isinstance(nodes, list)
            and len(nodes) == 1
            and isinstance(nodes[0], str)
            and nodes[0],
            "component job must identify one evidence storage node",
        )
        write_json(
            root / "job-placement.json",
            {"job_id": controller["job_id"], "node_id": nodes[0]},
        )

    code = lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=extract_summary,
        summary_validator=validated,
        historical_first=True,
    )
    if args.dry_run or code:
        return code
    controller = read_json(root / "summary.json")
    require(
        controller.get("status") == "passed" and controller.get("cleanup_errors") == [],
        "component job or owned cleanup failed",
    )
    placement = read_json(root / "job-placement.json")
    require(
        placement.get("job_id") == controller["job_id"],
        "component evidence placement differs",
    )
    controller["node_id"] = placement["node_id"]
    write_json(root / "summary.json", controller)
    transfer = fetch_artifact(
        image=args.image,
        job={"args": args, "controller": controller},
        expected={
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image,
            "qualification": QUALIFICATION,
        },
        output=root / "handoff",
        runner=ProcessRunner(root / "artifact-commands"),
        timeout=args.artifact_timeout,
    )
    report = read_json(Path(transfer["extracted"]) / "summary.json")
    validate_summary(
        report,
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image,
        profile=args.profile,
    )
    require(
        report == read_json(root / "job-summary.json"),
        "sealed component summary differs from job logs",
    )
    write_json(
        root / "component-receipt.json",
        {
            "schema_version": 1,
            "status": "passed",
            "qualification": QUALIFICATION,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image,
            "run_id": args.run_id,
            "execution": report["execution"],
            "job_id": controller["job_id"],
            "node_id": controller["node_id"],
            "summary": report,
            "transfer": transfer,
        },
    )
    return 0


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
