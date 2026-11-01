#!/usr/bin/env python3
"""Run one public pnms build on Lepton and retrieve compact verified inputs."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "native-inference/deploy"))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_release_build_job as job  # noqa: E402
from native_release import ProcessRunner, image_digest, read_json, require, write_json  # noqa: E402
from native_release_lepton import _object, fetch_artifact  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("release-build",))
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--sdk-prefix", default="/opt/physicsnemo-inference")
    parser.add_argument("--source-metadata", default="/opt/physicsnemo-qa/source.json")
    parser.add_argument(
        "--source-inventory", default="/opt/physicsnemo-qa/source-inventory.json"
    )
    parser.add_argument(
        "--project", help="Project directory relative to the frozen image snapshot"
    )
    parser.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    parser.add_argument("--template", choices=("tensor", "custom"))
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--build-command-timeout", type=float, default=2400)
    parser.add_argument("--export-timeout", type=int, default=1200)
    return parser


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{int(args.job_timeout) - 60}s",
        "/opt/nvidia/nvidia_entrypoint.sh",
        args.python,
        "-B",
        str(Path(args.repository) / "qa/native_inference/pnms_release_build_job.py"),
        "--repository",
        args.repository,
        "--python",
        args.python,
        "--sdk-prefix",
        args.sdk_prefix,
        "--source-metadata",
        args.source_metadata,
        "--source-inventory",
        args.source_inventory,
        "--run-id",
        args.run_id,
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--expected-runner-sha256",
        job.sha256(job.__file__),
        "--runtime-image",
        args.runtime_image,
        "--backend",
        args.backend,
        "--command-timeout",
        str(args.build_command_timeout),
    ]
    if args.project:
        runner += ["--project", args.project]
    if args.template:
        runner += ["--template", args.template]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def run(args):
    image_digest(args.runtime_image)
    require(not args.keep_job, "release-build handoff requires owned GPU job cleanup")
    require(
        math.isfinite(args.job_timeout) and 60 < args.job_timeout <= 7200,
        "release build job timeout must exceed 60 and be at most 7200 seconds",
    )
    require(
        math.isfinite(args.build_command_timeout)
        and 0 < args.build_command_timeout <= args.job_timeout,
        "build command timeout must be finite and within the job deadline",
    )
    require(
        0 < args.export_timeout <= 1800,
        "artifact export timeout must be at most 1800 seconds",
    )
    if args.project:
        project = Path(args.project)
        require(
            not project.is_absolute() and project.parts and ".." not in project.parts,
            "project must be relative to the frozen source snapshot",
        )
    root = args.artifact_dir.resolve() / args.run_id

    def validated(report, **identity):
        job.validate_summary(report, **identity)
        require(
            report["runtime_image"] == args.runtime_image,
            "runtime image differs from submitted build",
        )
        require(
            report["runner_sha256"] == job.sha256(job.__file__),
            "release-build runner differs",
        )
        runner_root = f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}/http-runner"
        require(
            report["http_runner"] == runner_root + "/pnms_http_job.py",
            "HTTP runner mount differs",
        )
        for key, name in (
            ("http_runner_sha256", "pnms_http_job.py"),
            ("contract_sha256", "contract.py"),
        ):
            require(
                report[key] == job.sha256(ROOT / "qa/native_inference" / name),
                "frozen HTTP runner differs: " + name,
            )
        controller = read_json(root / "summary.json")
        env = {
            **os.environ,
            "LEPTON_WORKSPACE_ID": args.workspace_id,
            "COLUMNS": "1000000",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
        if args.workspace_url:
            env["LEPTON_WORKSPACE_URL"] = args.workspace_url
        code, output = lifecycle.lep(
            ["job", "nodes", "-i", controller["job_id"]],
            env=env,
            timeout=min(30, args.command_timeout),
        )
        nodes = _object(output, "[", "]") if code == 0 else []
        require(
            len(set(nodes)) == 1 and isinstance(nodes[0], str) and nodes[0],
            "completed public build must identify one worker node before cleanup",
        )
        write_json(
            root / "job-placement.json",
            {"job_id": controller["job_id"], "node_id": nodes[0]},
        )

    code = lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=job.extract_summary,
        summary_validator=validated,
        historical_first=True,
    )
    if code or args.dry_run:
        return code
    controller = read_json(root / "summary.json")
    try:
        require(
            controller.get("status") == "passed"
            and controller.get("cleanup_errors") == [],
            "public build or owned GPU cleanup failed",
        )
        report = read_json(root / "job-summary.json")
        job.validate_summary(
            report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image,
            profile="release-build",
        )
        placement = read_json(root / "job-placement.json")
        require(
            placement["job_id"] == controller["job_id"],
            "artifact placement belongs to another job",
        )
        controller["node_id"] = placement["node_id"]
        write_json(root / "summary.json", controller)
        expected = {
            key: report[key]
            for key in (
                "run_id",
                "source_sha",
                "source_inventory_sha256",
                "image_digest",
                "runtime_image",
                "build_id",
                "bundle_digest",
                "archive",
                "build_receipt_identity",
                "source_inventory_identity",
            )
        }
        transfer = fetch_artifact(
            image=args.image,
            job={"args": args, "controller": controller},
            expected=expected,
            output=root / "transfer",
            runner=ProcessRunner(root / "transfer-commands"),
            timeout=args.export_timeout,
        )
        require(
            transfer.get("status") == "passed"
            and transfer.get("cleanup_errors") == []
            and transfer.get("deployment_deleted") is True,
            "artifact transfer or owned CPU exporter cleanup failed",
        )
        manifest = read_json(root / "transfer/manifest.json")
        inputs = job.restore_handoff(transfer["extracted"], manifest)
        inputs.update(
            {
                key: report[key]
                for key in ("http_runner", "http_runner_sha256", "contract_sha256")
            }
        )
        inputs.update(
            node_id=placement["node_id"],
            job_id=controller["job_id"],
            transport_manifest=str(root / "transfer/manifest.json"),
            transfer_summary=str(root / "transfer/transfer-summary.json"),
        )
        write_json(root / "release-inputs.json", inputs)
        controller["release_inputs"] = str(root / "release-inputs.json")
        write_json(root / "summary.json", controller)
        return 0
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        cancelled = isinstance(error, (KeyboardInterrupt, SystemExit))
        controller.update(
            status="cancelled" if cancelled else "failed",
            transfer_error=f"{type(error).__name__}: {error}",
        )
        write_json(root / "summary.json", controller)
        return 130 if cancelled else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
