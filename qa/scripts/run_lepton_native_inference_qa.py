#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit and verify one native inference QA job on Lepton."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "deploy"))
sys.path.insert(0, str(REPO_ROOT))
from config import load_deploy_config  # noqa: E402

from qa.native_inference.contract import (  # noqa: E402
    SUMMARY_BEGIN,
    SUMMARY_END,
    validate_summary,
)

SUCCESS_STATES = {"Completed", "Succeeded", "Success"}
FAILURE_STATES = {"Failed", "Cancelled", "Stopped", "Error", "Deleted", "Archived"}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def extract_summary(output: str) -> dict:
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    if begin < 0 or end < 0:
        raise ValueError("job logs contain no complete native QA summary")
    report = json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())
    if not isinstance(report, dict):
        raise ValueError("native QA summary must be an object")
    return report


def decode_historical_logs(output: str) -> str:
    """Undo the timestamp and JSON encoding applied by `lep log get`."""
    lines = []
    for line in output.splitlines():
        match = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\|(.*)$", line)
        if not match:
            continue
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        lines.append(value if isinstance(value, str) else json.dumps(value))
    return "\n".join(lines)


def lep(command: list[str], *, env: dict, timeout: float) -> tuple[int, str]:
    """Bound each CLI operation and never propagate token-bearing command text."""
    try:
        result = subprocess.run(
            ["lep", *command],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        code, output = result.returncode, result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        code, output = 124, exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
    except OSError:
        code, output = 127, "Could not execute the installed Lepton CLI.\n"
    for key in ("LEPTON_WORKSPACE_TOKEN", "LEPTON_API_TOKEN"):
        if env.get(key):
            output = output.replace(env[key], "<redacted>")
    return code, output


def build_parser() -> argparse.ArgumentParser:
    config = load_deploy_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Immutable repository@sha256:digest"
    )
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--assets",
        help="Use a mounted manifest offline; full profile otherwise downloads pinned assets",
    )
    parser.add_argument("--run-id", default=secrets.token_hex(8))
    parser.add_argument(
        "--artifact-dir", type=Path, default=REPO_ROOT / "qa/artifacts/native-inference"
    )
    parser.add_argument("--workspace-id", default=config.get("lepton_workspace_id", ""))
    parser.add_argument(
        "--workspace-url", default=os.environ.get("LEPTON_WORKSPACE_URL", "")
    )
    parser.add_argument("--node-group", default=config.get("lepton_node_group", ""))
    parser.add_argument(
        "--pull-secret",
        default=os.environ.get("LEPTON_PULL_SECRET") or config.get("pull_secret", ""),
    )
    parser.add_argument(
        "--resource-shape",
        default=os.environ.get("LEPTON_RESOURCE_SHAPE", "gpu.h100-sxm"),
    )
    parser.add_argument("--nfs-mount-base", default=config.get("nfs_mount_base", ""))
    parser.add_argument("--lustre-dir", default=os.environ.get("QA_LUSTRE_DIR", "qa"))
    parser.add_argument(
        "--lustre-storage", default=os.environ.get("LEPTON_LUSTRE_STORAGE", "lustre")
    )
    parser.add_argument("--mount-target", default="/outputs")
    parser.add_argument("--job-timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=20)
    parser.add_argument("--command-timeout", type=float, default=60)
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", args.image
    ):
        raise ValueError(
            "--image must name an immutable repository@sha256:<64 lowercase hex> image"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_source_sha):
        raise ValueError("--expected-source-sha must be a full 40-character Git SHA")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", args.run_id):
        raise ValueError(
            "--run-id must be 1-40 lowercase alphanumeric or hyphen characters"
        )
    for name in (
        "workspace_id",
        "node_group",
        "pull_secret",
        "resource_shape",
        "nfs_mount_base",
        "lustre_storage",
    ):
        if not getattr(args, name):
            raise ValueError(
                f"--{name.replace('_', '-')} or its configured value is required"
            )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.job_timeout, args.poll_interval, args.command_timeout)
    ):
        raise ValueError(
            "timeouts and polling interval must be finite positive numbers"
        )
    if args.poll_interval > 60 or args.command_timeout > 300:
        raise ValueError(
            "poll interval must be at most 60s and command timeout at most 300s"
        )
    for name in ("nfs_mount_base", "mount_target"):
        path = PurePosixPath(getattr(args, name))
        if (
            not path.is_absolute()
            or ".." in path.parts
            or ":" in str(path)
            or str(path) == "/"
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a non-root absolute mount path"
            )
    relative = PurePosixPath(args.lustre_dir)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or ":" in str(relative)
        or str(relative) == "."
    ):
        raise ValueError("--lustre-dir must be a relative directory without traversal")
    if args.assets:
        assets = PurePosixPath(args.assets)
        if ".." in assets.parts or not assets.is_relative_to(
            PurePosixPath(args.mount_target)
        ):
            raise ValueError("--assets must be an absolute path below --mount-target")


def build_job_command(args, job_name: str) -> list[str]:
    output = f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}"
    runner = [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        "python3",
        "/opt/physicsnemo-qa/run_job.py",
        "--profile",
        args.profile,
        "--run-id",
        args.run_id,
        "--output",
        output,
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
    ]
    if args.assets:
        runner += ["--assets", args.assets]
    mount = f"{args.nfs_mount_base.rstrip('/')}/{args.lustre_dir}:{args.mount_target}:node-nfs:{args.lustre_storage}"
    return [
        "job",
        "create",
        "--name",
        job_name,
        "--file",
        str(args.artifact_dir.resolve() / args.run_id / "job-spec.json"),
        "--container-image",
        args.image,
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.resource_shape,
        "--num-workers",
        "1",
        "--max-failure-retry",
        "0",
        "--max-job-failure-retry",
        "0",
        "--log-collection",
        "true",
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        mount,
        "--command",
        shlex.join(runner),
    ]


def run(args) -> int:
    validate_args(args)
    started = datetime.now(timezone.utc)
    artifacts = args.artifact_dir.resolve() / args.run_id
    artifacts.mkdir(parents=True, exist_ok=False)
    # Some CLI releases ignore explicit zero retry flags because they test
    # truthiness. The user-spec file preserves zero through SDK parsing.
    write_json(
        artifacts / "job-spec.json",
        {
            "max_failure_retry": 0,
            "max_job_failure_retry": 0,
        },
    )
    job_name = f"pn-native-{args.run_id[:16]}-{secrets.token_hex(3)}"
    command = build_job_command(args, job_name)
    summary = {
        "run_id": args.run_id,
        "source_sha": args.expected_source_sha,
        "image_digest": args.image,
        "profile": args.profile,
        "status": "failed",
        "job_name": job_name,
        "job_id": None,
        "started_at": started.isoformat(),
        "remote_artifacts": f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
    }
    write_json(artifacts / "job-command.json", {"argv": ["lep", *command]})
    write_json(artifacts / "summary.json", summary)
    if args.dry_run:
        summary["status"] = "dry-run"
        write_json(artifacts / "summary.json", summary)
        print(f"Dry run recorded at {artifacts}")
        return 0

    # The CLI prints logs through Rich; wrapping a long JSON string corrupts it.
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_ID": args.workspace_id,
        "COLUMNS": "1000000",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    if args.workspace_url:
        env["LEPTON_WORKSPACE_URL"] = args.workspace_url
    submitted = False
    terminal = False
    logs_captured = False
    old_handlers = {}
    exit_code = 1

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    def call(argv, filename=None, timeout=None):
        code, output = lep(argv, env=env, timeout=timeout or args.command_timeout)
        if filename:
            (artifacts / filename).write_text(output)
        return code, output

    def collect_logs(job_id):
        _, output = call(["job", "log", "-i", job_id], "job.log")
        try:
            extract_summary(output)
            return output
        except ValueError:
            pass
        # Completed pods can have no live log stream. Historical ingestion can
        # also lag job completion, so retry within a fixed, small budget.
        start = (started - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S.%f")
        for attempt in range(1, 4):
            _, historical = call(
                [
                    "log",
                    "get",
                    "--job",
                    job_id,
                    "--start",
                    start,
                    "--end",
                    "now",
                    "--limit",
                    "5000",
                ],
                f"job-history-{attempt}.log",
                min(args.command_timeout, 30),
            )
            decoded = decode_historical_logs(historical)
            (artifacts / "job-history.log").write_text(decoded)
            try:
                extract_summary(decoded)
                return decoded
            except ValueError:
                if attempt < 3:
                    time.sleep(5)
        return decoded or output

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        if env.get("LEPTON_WORKSPACE_TOKEN"):
            login = [
                "login",
                "-c",
                f"{args.workspace_id}:{env['LEPTON_WORKSPACE_TOKEN']}",
            ]
            if args.workspace_url:
                login += ["-u", args.workspace_url]
            code, _ = call(login, "login.log")
            if code:
                raise RuntimeError(f"Lepton login failed (exit {code})")

        submitted = True
        code, output = call(command, "job-create.log")
        match = re.search(r"(?m)^\s*ID:\s*(\S+)", output)
        if match:
            summary["job_id"] = match.group(1)
        write_json(artifacts / "summary.json", summary)
        if code or not summary["job_id"]:
            raise RuntimeError(
                f"Lepton job creation did not confirm a job ID (exit {code})"
            )
        job_id = summary["job_id"]
        deadline = time.monotonic() + args.job_timeout
        previous_state = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Lepton native QA job timed out")
            code, output = call(
                ["job", "get", "-i", job_id],
                "job-state.log",
                min(args.command_timeout, remaining),
            )
            match = re.search(r'"state"\s*:\s*"([^\"]+)"', output)
            state = match.group(1) if code == 0 and match else "Unknown"
            if state != previous_state:
                print(f"Native QA job {job_id}: {state}", flush=True)
                previous_state = state
            if state in SUCCESS_STATES | FAILURE_STATES:
                terminal = True
                summary["job_state"] = state
                break
            time.sleep(min(args.poll_interval, max(0, deadline - time.monotonic())))

        output = collect_logs(job_id)
        logs_captured = True
        report = extract_summary(output)
        write_json(artifacts / "job-summary.json", report)
        if state not in SUCCESS_STATES:
            raise RuntimeError(f"Lepton native QA job ended in {state}")
        validate_summary(
            report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image,
            profile=args.profile,
        )
        summary["status"] = "passed"
        exit_code = 0
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            summary["status"] = "cancelled"
            exit_code = (
                int(exc.code)
                if isinstance(exc, SystemExit) and isinstance(exc.code, int)
                else 130
            )
    finally:
        # Keep signal handling from interrupting the bounded cleanup sequence.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        cleanup_errors = []
        try:
            job_id = summary["job_id"]
            if submitted:
                if job_id and not terminal and not args.keep_job:
                    code, _ = call(["job", "stop", "-i", job_id], "job-stop.log")
                    if code:
                        cleanup_errors.append(f"stop exited {code}")
                if job_id and not logs_captured:
                    collect_logs(job_id)
                if not args.keep_job:
                    selector = ["-i", job_id] if job_id else ["-n", job_name]
                    code, _ = call(["job", "remove", *selector], "job-remove.log")
                    if code:
                        cleanup_errors.append(f"remove exited {code}")
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            summary["cleanup_errors"] = cleanup_errors
            if cleanup_errors:
                summary["status"] = "failed"
                exit_code = exit_code or 1
            write_json(artifacts / "summary.json", summary)
            print(f"Native QA {summary['status']}; evidence: {artifacts}", flush=True)
    return exit_code


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
