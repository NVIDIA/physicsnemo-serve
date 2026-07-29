# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run REST PhysicsNeMo-CFD QA, then compare with a direct Lepton batch job."""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cfd_parity_contract import (
    ParityContractError,
    build_handoff,
    read_json_object,
    validate_profile,
    write_json_atomic,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
QA_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = QA_ROOT.parent
RUN_QA_SCRIPT = SCRIPTS_DIR / "run_qa.py"
JOB_RUNNER = SCRIPTS_DIR / "run_cfd_parity_job.py"
CONTRACT_MODULE = SCRIPTS_DIR / "cfd_parity_contract.py"
DEFAULT_PROFILE = QA_ROOT / "inference" / "cfd_parity_surface_run1.json"
SUMMARY_BEGIN = "PHYSICSNEMO_CFD_PARITY_SUMMARY_BEGIN"
SUMMARY_END = "PHYSICSNEMO_CFD_PARITY_SUMMARY_END"

sys.path.insert(0, str(REPO_ROOT / "deploy"))
from config import load_deploy_config  # noqa: E402


_DEPLOY_CONFIG = load_deploy_config()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))


def validate_run_id(run_id: str) -> str:
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", run_id) is None:
        raise ValueError(
            "run ID must be 1-64 lowercase alphanumeric or hyphen characters"
        )
    return run_id


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def image_full_reference(image_tag: str, image_name: str) -> str:
    last_slash = image_tag.rfind("/")
    last_colon = image_tag.rfind(":")
    if last_colon > last_slash:
        return image_tag
    return f"{image_name}:{image_tag}"


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in args)


def run_streaming(
    cmd: list[str],
    *,
    env: dict[str, str],
    artifact_path: Path | None = None,
) -> tuple[int, str]:
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    artifact = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = artifact_path.open("w", encoding="utf-8", buffering=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
            if artifact is not None:
                artifact.write(line)
    finally:
        if artifact is not None:
            artifact.close()
    return process.wait(), "".join(output)


def parse_job_id(output: str) -> str | None:
    for line in output.splitlines():
        match = re.match(r"\s*ID:\s*(\S+)", line)
        if match:
            return match.group(1)
    return None


def _job_succeeded(output: str) -> bool:
    return bool(re.search(r'"state":\s*"(Completed|Succeeded|Success)"', output, re.I))


def _job_failed(output: str) -> bool:
    return bool(
        re.search(r'"state":\s*"(Failed|Cancelled|Stopped|Error)"', output, re.I)
    )


def _job_state(output: str) -> str | None:
    match = re.search(r'"state":\s*"([^"]+)"', output, re.I)
    return match.group(1) if match else None


def poll_job(
    *,
    job_id: str,
    env: dict[str, str],
    timeout_seconds: int,
    interval_seconds: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    previous_state: str | None = None
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["lep", "job", "get", "-i", job_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout or ""
        state = _job_state(output)
        if state != previous_state:
            print(f"==> Lepton job state: {state or 'unknown'}", flush=True)
            previous_state = state
        if _job_succeeded(output):
            return 0
        if _job_failed(output):
            return 1
        time.sleep(interval_seconds)
    print(f"Error: Lepton job timed out after {timeout_seconds}s", file=sys.stderr)
    return 1


def capture_job_logs(
    *,
    job_id: str,
    env: dict[str, str],
    artifact_path: Path,
    timeout_seconds: int = 120,
    print_output: bool = True,
) -> str:
    try:
        result = subprocess.run(
            ["lep", "job", "log", "-i", job_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        value = exc.stdout or ""
        output = value.decode(errors="replace") if isinstance(value, bytes) else value
        output += f"\nJob log capture timed out after {timeout_seconds}s.\n"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(output, encoding="utf-8")
    if output and print_output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    return output


def wait_for_loggable_job(
    *,
    job_id: str,
    env: dict[str, str],
    timeout_seconds: int = 120,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["lep", "job", "get", "-i", job_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        state = _job_state(result.stdout or "")
        if state in {
            "Running",
            "Completed",
            "Succeeded",
            "Success",
            "Failed",
            "Error",
        }:
            time.sleep(5)
            return
        time.sleep(5)


def extract_marked_summary(output: str) -> dict[str, Any] | None:
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.rfind(SUMMARY_END)
    if begin < 0 or end <= begin:
        return None
    text = output[begin + len(SUMMARY_BEGIN) : end].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(
                "".join(text.split()),
                validate=True,
            ).decode("utf-8")
            value = json.loads(decoded)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    return value if isinstance(value, dict) else None


def base64_write_command(path: str, text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"


def remote_root(mount_target: str, run_id: str) -> str:
    return f"{mount_target.rstrip('/')}/cfd-parity/{run_id}"


def _lepton_job_name(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    candidate = f"{prefix}-{slug}"
    if len(candidate) <= 36:
        return candidate
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:27].rstrip('-')}-{suffix}"


def job_name(profile: Mapping[str, Any], run_id: str) -> str:
    domain = re.sub(r"[^a-z0-9]+", "-", str(profile["domain"]).lower()).strip("-")
    return _lepton_job_name(f"pn-cfd-parity-{domain}", run_id)


def build_remote_command(
    *,
    profile: Mapping[str, Any],
    profile_text: str,
    handoff_text: str,
    mount_target: str,
    run_id: str,
) -> str:
    root = remote_root(mount_target, run_id)
    parent = str(Path(root).parent)
    contract_path = f"{root}/cfd_parity_contract.py"
    runner_path = f"{root}/run_cfd_parity_job.py"
    profile_path = f"{root}/profile.json"
    handoff_path = f"{root}/handoff.json"
    log_path = f"{root}/job-output.log"
    setup = [
        f"mkdir -p {shlex.quote(parent)}",
        f"test ! -e {shlex.quote(root)}",
        f"mkdir {shlex.quote(root)}",
        base64_write_command(
            contract_path, CONTRACT_MODULE.read_text(encoding="utf-8")
        ),
        base64_write_command(runner_path, JOB_RUNNER.read_text(encoding="utf-8")),
        base64_write_command(profile_path, profile_text),
        base64_write_command(handoff_path, handoff_text),
    ]
    command = [
        str(profile["runner"]["python"]),
        runner_path,
        "--profile",
        profile_path,
        "--handoff",
        handoff_path,
        "--mount-target",
        mount_target,
        "--work-dir",
        root,
    ]
    return (
        "set -o pipefail; "
        + "; ".join(setup)
        + "; "
        + f"{shell_join(command)} 2>&1 | tee {shlex.quote(log_path)}; "
        + "rc=${PIPESTATUS[0]}; exit $rc"
    )


def build_job_args(
    args: argparse.Namespace,
    *,
    profile: Mapping[str, Any],
    profile_text: str,
    handoff_text: str,
    image: str,
    nfs_path: str,
    run_id: str,
) -> list[str]:
    return [
        "--name",
        job_name(profile, run_id),
        "--container-image",
        image,
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.resource_shape,
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
        "--command",
        build_remote_command(
            profile=profile,
            profile_text=profile_text,
            handoff_text=handoff_text,
            mount_target=args.mount_target,
            run_id=run_id,
        ),
    ]


def discover_rest_evidence(root: Path) -> Path:
    required = {
        "request.json",
        "results.json",
        "resolved_config.json",
        "benchmark_results.json",
    }
    matches = [
        child
        for child in root.iterdir()
        if child.is_dir() and all((child / name).is_file() for name in required)
    ]
    if len(matches) != 1:
        raise ParityContractError(
            f"expected exactly one REST evidence directory below {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def login_if_needed(args: argparse.Namespace, env: dict[str, str]) -> None:
    if not args.workspace_token:
        print("==> Using existing Lepton login session", flush=True)
        return
    credentials = f"{args.workspace_id}:{args.workspace_token}"
    cmd = ["lep", "login", "-c", credentials]
    if args.workspace_url:
        cmd += ["-u", args.workspace_url]
    print(
        f"+ lep login -c {args.workspace_id}:<redacted>"
        f"{' -u ' + args.workspace_url if args.workspace_url else ''}",
        flush=True,
    )
    returncode, _ = run_streaming(cmd, env=env)
    if returncode != 0:
        raise RuntimeError(f"lep login exited with code {returncode}")


def run_rest_qa(
    args: argparse.Namespace,
    *,
    profile: Mapping[str, Any],
    image: str,
    artifact_dir: Path,
    env: dict[str, str],
) -> Path:
    rest = profile["rest"]
    request_path = Path(args.rest_request_path or str(rest["request_path"]))
    if not request_path.is_absolute():
        request_path = REPO_ROOT / request_path
    if not request_path.is_file():
        raise ParityContractError(f"REST parity request does not exist: {request_path}")
    command = [
        sys.executable,
        "-u",
        str(RUN_QA_SCRIPT),
        "--service",
        str(rest["service"]),
        "--image-tag",
        image,
        "--suite",
        str(rest["suite"]),
        "--num-proc",
        "1",
        "--lustre-dir",
        args.lustre_dir,
        "--artifact-dir",
        str(artifact_dir),
    ]
    print(f"+ {shell_join(command)}", flush=True)
    rest_env = {
        **env,
        "QA_CFD_E2E_REQUEST_PATH": str(request_path),
    }
    returncode, _ = run_streaming(
        command,
        env=rest_env,
        artifact_path=artifact_dir.parent / "rest-qa.log",
    )
    if returncode != 0:
        raise RuntimeError(f"REST QA exited with code {returncode}")
    return discover_rest_evidence(artifact_dir / str(rest["evidence_subdir"]))


def _load_remote_summary_if_available(
    nfs_path: str, run_id: str
) -> dict[str, Any] | None:
    path = Path(nfs_path) / "cfd-parity" / run_id / "summary.json"
    try:
        return read_json_object(path) if path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def fetch_summary_via_reader_job(
    args: argparse.Namespace,
    *,
    image: str,
    nfs_path: str,
    run_id: str,
    env: dict[str, str],
    artifact_dir: Path,
) -> dict[str, Any] | None:
    summary_path = f"{remote_root(args.mount_target, run_id)}/summary.json"
    reader_name = _lepton_job_name("pn-cfd-parity-read", run_id)
    reader_python = (
        "import base64\n"
        "from pathlib import Path\n"
        f"text = Path({summary_path!r}).read_text(encoding='utf-8')\n"
        f"print({SUMMARY_BEGIN!r}, flush=True)\n"
        "print(base64.b64encode(text.encode('utf-8')).decode('ascii'), flush=True)\n"
        f"print({SUMMARY_END!r}, flush=True)\n"
    )
    encoded = base64.b64encode(reader_python.encode("utf-8")).decode("ascii")
    reader_command = f"printf %s {shlex.quote(encoded)} | base64 -d | python3; sleep 30"
    reader_args = [
        "--name",
        reader_name,
        "--container-image",
        image,
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.reader_resource_shape,
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
        "--command",
        reader_command,
    ]
    print(f"==> Fetching remote parity summary with reader job: {reader_name}")
    returncode, output = run_streaming(
        ["lep", "job", "create", *reader_args],
        env=env,
        artifact_path=artifact_dir / "reader-create.log",
    )
    if returncode != 0:
        return None
    reader_job_id = parse_job_id(output)
    if reader_job_id is None:
        return None
    try:
        wait_for_loggable_job(job_id=reader_job_id, env=env)
        logs = capture_job_logs(
            job_id=reader_job_id,
            env=env,
            artifact_path=artifact_dir / "reader.log",
            timeout_seconds=75,
            print_output=False,
        )
        return extract_marked_summary(logs)
    finally:
        run_streaming(["lep", "job", "stop", "-i", reader_job_id], env=env)
        run_streaming(["lep", "job", "remove", "-i", reader_job_id], env=env)


def run(args: argparse.Namespace) -> int:
    profile_path = resolve_path(args.profile)
    profile_text = profile_path.read_text(encoding="utf-8")
    profile = json.loads(profile_text)
    if not isinstance(profile, dict):
        raise ParityContractError("parity profile must contain a JSON object")
    validate_profile(profile)

    run_id = validate_run_id(args.run_id or generate_run_id())
    image = image_full_reference(args.image_tag, args.image_name)
    nfs_path = f"{args.nfs_mount_base.rstrip('/')}/{args.lustre_dir}"
    artifact_root = resolve_path(args.artifact_dir)
    run_artifact_dir = artifact_root / "cfd-parity" / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_artifact_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "final_result": "failed",
        "run_id": run_id,
        "profile_id": profile["profile_id"],
        "image": image,
        "mount": {
            "nfs_path": nfs_path,
            "mount_target": args.mount_target,
            "lustre_storage": args.lustre_storage,
        },
        "artifacts": {
            "local_dir": str(run_artifact_dir),
            "remote_dir": remote_root(args.mount_target, run_id),
        },
    }

    env = dict(os.environ)
    if args.workspace_token:
        env["LEPTON_WORKSPACE_TOKEN"] = args.workspace_token
    if args.workspace_url:
        env["LEPTON_WORKSPACE_URL"] = args.workspace_url
    env.update(
        {
            "LEPTON_WORKSPACE_ID": args.workspace_id,
            "LEPTON_NODE_GROUP": args.node_group,
            "LEPTON_RESOURCE_SHAPE": args.resource_shape,
            "LEPTON_PULL_SECRET": args.pull_secret,
            "LEPTON_LUSTRE_STORAGE": args.lustre_storage,
            "NFS_MOUNT_BASE": args.nfs_mount_base,
        }
    )

    job_id: str | None = None
    job_finished = False
    cleanup_done = False

    def cleanup_job() -> None:
        nonlocal cleanup_done
        if cleanup_done or job_id is None or args.keep_job or args.dry_run:
            return
        cleanup_done = True
        if not job_finished:
            run_streaming(["lep", "job", "stop", "-i", job_id], env=env)
        run_streaming(["lep", "job", "remove", "-i", job_id], env=env)

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n==> Received signal {signum}; cleaning up parity job", flush=True)
        cleanup_job()
        raise SystemExit(128 + signum)

    atexit.register(cleanup_job)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.rest_evidence_dir:
            evidence_dir = resolve_path(args.rest_evidence_dir)
        else:
            if args.dry_run:
                raise ParityContractError("--dry-run requires --rest-evidence-dir")
            evidence_dir = run_rest_qa(
                args,
                profile=profile,
                image=image,
                artifact_dir=run_artifact_dir / "rest-qa",
                env=env,
            )
        summary["artifacts"]["rest_evidence_dir"] = str(evidence_dir)

        handoff = build_handoff(
            evidence_dir=evidence_dir,
            profile=profile,
            parity_run_id=run_id,
            image=image,
            mount_target=args.mount_target,
        )
        handoff_path = run_artifact_dir / "parity-handoff.json"
        write_json_atomic(handoff_path, handoff)
        summary["rest_run_id"] = handoff["rest_run_id"]
        summary["artifacts"]["handoff"] = str(handoff_path)
        handoff_text = json.dumps(handoff, indent=2, sort_keys=True) + "\n"
        job_args = build_job_args(
            args,
            profile=profile,
            profile_text=profile_text,
            handoff_text=handoff_text,
            image=image,
            nfs_path=nfs_path,
            run_id=run_id,
        )
        remote_command = job_args[-1]
        summary["job"] = {
            "name": job_name(profile, run_id),
            "image": image,
            "node_group": args.node_group,
            "resource_shape": args.resource_shape,
            "pull_secret": args.pull_secret,
            "mount": (f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}"),
            "remote_command_sha256": hashlib.sha256(
                remote_command.encode("utf-8")
            ).hexdigest(),
        }
        if args.dry_run:
            summary["final_result"] = "dry-run"
            write_json_atomic(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        login_if_needed(args, env)
        command = ["lep", "job", "create", *job_args]
        print(
            "+ lep job create "
            f"--name {shlex.quote(str(summary['job']['name']))} "
            f"--container-image {shlex.quote(image)} "
            f"--node-group {shlex.quote(args.node_group)} "
            f"--resource-shape {shlex.quote(args.resource_shape)} "
            f"--mount {shlex.quote(str(summary['job']['mount']))} "
            "--command <injected-cfd-parity-runner>",
            flush=True,
        )
        create_returncode, create_output = run_streaming(
            command,
            env=env,
            artifact_path=run_artifact_dir / "job-create.log",
        )
        if create_returncode != 0:
            raise RuntimeError(
                f"Lepton job create exited with code {create_returncode}"
            )
        job_id = parse_job_id(create_output)
        if job_id is None:
            raise RuntimeError("could not parse Lepton job ID from create output")
        summary["job"]["id"] = job_id

        job_returncode = poll_job(
            job_id=job_id,
            env=env,
            timeout_seconds=args.job_timeout,
            interval_seconds=args.job_poll_interval,
        )
        job_finished = True
        if job_returncode != 0:
            run_streaming(["lep", "job", "stop", "-i", job_id], env=env)
        job_output = capture_job_logs(
            job_id=job_id,
            env=env,
            artifact_path=run_artifact_dir / "job.log",
        )
        direct_summary = _load_remote_summary_if_available(nfs_path, run_id)
        if direct_summary is None:
            direct_summary = extract_marked_summary(job_output)
        if direct_summary is None:
            direct_summary = fetch_summary_via_reader_job(
                args,
                image=image,
                nfs_path=nfs_path,
                run_id=run_id,
                env=env,
                artifact_dir=run_artifact_dir,
            )
        if direct_summary is None:
            raise RuntimeError("direct parity job produced no readable summary")
        write_json_atomic(run_artifact_dir / "direct-summary.json", direct_summary)
        summary["job"]["exit_code"] = job_returncode
        summary["direct_summary"] = direct_summary
        summary["final_result"] = (
            "passed"
            if job_returncode == 0 and direct_summary.get("final_result") == "passed"
            else "failed"
        )
    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        cleanup_job()
        summary["finished_at"] = utc_now()
        write_json_atomic(summary_path, summary)
        print(f"==> Parity summary: {summary_path}", flush=True)
    return 0 if summary["final_result"] in {"passed", "dry-run"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tag", required=True)
    registry = _DEPLOY_CONFIG.get("docker_registry", "")
    default_image = (
        f"{registry}/{_DEPLOY_CONFIG.get('image_name', 'physicsnemo-serve')}"
        if registry
        else _DEPLOY_CONFIG.get("image_name", "physicsnemo-serve")
    )
    parser.add_argument("--image-name", default=default_image)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument(
        "--rest-request-path",
        help="Override the profile's REST request fixture without changing its direct config.",
    )
    parser.add_argument("--rest-evidence-dir")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("QA_ARTIFACT_DIR", str(QA_ROOT / "artifacts")),
    )
    parser.add_argument(
        "--workspace-id",
        default=os.environ.get("LEPTON_WORKSPACE_ID")
        or _DEPLOY_CONFIG.get("lepton_workspace_id", ""),
    )
    parser.add_argument(
        "--workspace-token", default=os.environ.get("LEPTON_WORKSPACE_TOKEN", "")
    )
    parser.add_argument(
        "--workspace-url", default=os.environ.get("LEPTON_WORKSPACE_URL", "")
    )
    parser.add_argument(
        "--node-group",
        default=os.environ.get("LEPTON_NODE_GROUP")
        or _DEPLOY_CONFIG.get("lepton_node_group", ""),
    )
    parser.add_argument(
        "--resource-shape",
        default=os.environ.get("LEPTON_RESOURCE_SHAPE", "gpu.h100-sxm"),
    )
    parser.add_argument(
        "--pull-secret",
        default=os.environ.get("LEPTON_PULL_SECRET")
        or _DEPLOY_CONFIG.get("pull_secret", ""),
    )
    parser.add_argument(
        "--nfs-mount-base",
        default=os.environ.get("NFS_MOUNT_BASE")
        or _DEPLOY_CONFIG.get("nfs_mount_base", "/mnt/shared"),
    )
    parser.add_argument(
        "--lustre-dir",
        default=os.environ.get("QA_LUSTRE_DIR", os.environ.get("USER", "shared")),
    )
    parser.add_argument(
        "--lustre-storage",
        default=os.environ.get("LEPTON_LUSTRE_STORAGE", "lustre"),
    )
    parser.add_argument("--mount-target", default="/outputs")
    parser.add_argument("--job-timeout", type=int, default=23_400)
    parser.add_argument("--job-poll-interval", type=int, default=20)
    parser.add_argument("--reader-resource-shape", default="cpu.small")
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("workspace_id", "node_group", "pull_secret", "nfs_mount_base"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    if not args.rest_evidence_dir and not args.workspace_token:
        raise ValueError(
            "--workspace-token is required when the REST QA phase is enabled"
        )
    if args.job_timeout <= 0 or args.job_poll_interval <= 0:
        raise ValueError("job timeout and poll interval must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
