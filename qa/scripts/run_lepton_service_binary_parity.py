# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare physicsnemo-serve service-mode (REST) vs binary-mode for the CFD surface plugin.

Runs the physicsnemo-cfd-surface-benchmark plugin twice:
  1. Service-mode: deployed as a Lepton endpoint via run_qa.py (REST interface).
  2. Binary-mode: a one-shot Lepton batch job using the scicomp-ferroflux-cmd image,
     which runs `physicsnemo-serve infer ...` directly with no HTTP layer.

Both runs use the same request JSON.  The benchmark_results.json outputs are then
compared using cfd_parity_contract.compare_reports with the tolerances defined in
the parity profile.

Typical usage (first run, both phases):

    python run_lepton_service_binary_parity.py \\
        --image-tag <service-image-tag> \\
        --binary-image nvcr.io/nvidia/scicomp-ferroflux-cmd:pr-14-1c397c1

Skip the slow REST QA phase on subsequent runs by reusing existing evidence:

    python run_lepton_service_binary_parity.py \\
        --image-tag <service-image-tag> \\
        --binary-image nvcr.io/nvidia/scicomp-ferroflux-cmd:pr-14-1c397c1 \\
        --rest-evidence-dir qa/artifacts/cfd-binary/<run-id>/rest-qa/cfd-e2e/<rest-run-id>
"""

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
    compare_reports,
    validate_profile,
    write_json_atomic,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
QA_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = QA_ROOT.parent
RUN_QA_SCRIPT = SCRIPTS_DIR / "run_qa.py"
DEFAULT_PROFILE = QA_ROOT / "inference" / "cfd_parity_surface_run1_full_matrix.json"

PLUGIN_EXAMPLES = (
    REPO_ROOT / "plugins" / "physicsnemo-cfd-surface-benchmark" / "examples"
)

# Maps --matrix choice to (profile, request) paths.
MATRIX_CONFIGS: dict[str, tuple[Path, Path]] = {
    "all": (
        QA_ROOT / "inference" / "cfd_parity_surface_run1_full_matrix.json",
        PLUGIN_EXAMPLES / "public_run_1_full_matrix_request.json",
    ),
    "three": (
        QA_ROOT / "inference" / "cfd_parity_surface_run1_3model.json",
        PLUGIN_EXAMPLES / "public_run_1_3model_request.json",
    ),
    "domino": (
        QA_ROOT / "inference" / "cfd_parity_surface_run1.json",
        PLUGIN_EXAMPLES / "public_run_1_request.json",
    ),
}

# Fixed paths baked into the scicomp-ferroflux-cmd image.
BINARY_PLUGIN_DIR = "/opt/physicsnemo-serve/plugins/physicsnemo-cfd-surface-benchmark"
BINARY_RUNTIME_DIR = "/opt/physicsnemo-serve/runtimes/cfd"

# run-id argument passed to `physicsnemo-serve infer --run-id`.
# Determines the subdirectory under --output-dir where artifacts land.
BINARY_RUN_SUBID = "binary"

# Env vars required by the binary's prefetch / download machinery.
_PREFETCH_ENV: dict[str, str] = {
    "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS": (
        "huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"
    ),
    "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS": (
        "us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"
    ),
    "E2S_DOWNLOAD_TIMEOUT_SECS": "1800",
}

# Sentinel strings printed by the reader job around the base64-encoded report JSON.
BINARY_REPORT_BEGIN = "PHYSICSNEMO_SERVE_BINARY_PARITY_REPORT_BEGIN"
BINARY_REPORT_END = "PHYSICSNEMO_SERVE_BINARY_PARITY_REPORT_END"

DEFAULT_BINARY_IMAGE_NAME = "physicsnemo-serve-cmd"

sys.path.insert(0, str(REPO_ROOT / "deploy"))
from config import load_deploy_config  # noqa: E402

_DEPLOY_CONFIG = load_deploy_config()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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
    """Return a full image reference, prepending image_name if the tag has no colon."""
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
        # Always wait for the subprocess so that on SIGINT the child process
        # (e.g. run_qa.py or a lep CLI call) has time to run its own teardown
        # before we exit.  The child also receives SIGINT (same process group)
        # so it should exit promptly; 120 s is a generous upper bound.
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode, "".join(output)


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
        if state is not None and state.lower() in {
            "running",
            "completed",
            "succeeded",
            "success",
            "failed",
            "error",
        }:
            time.sleep(5)
            return
        time.sleep(5)


def base64_write_command(path: str, text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"


def extract_marked_report(output: str) -> list[Any] | None:
    """Extract and decode a base64-wrapped benchmark_results.json list from job logs."""
    begin = output.rfind(BINARY_REPORT_BEGIN)
    end = output.rfind(BINARY_REPORT_END)
    if begin < 0 or end <= begin:
        return None
    text = output[begin + len(BINARY_REPORT_BEGIN) : end].strip()
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
    return value if isinstance(value, list) else None


# ---------------------------------------------------------------------------
# Remote path helpers
# ---------------------------------------------------------------------------


def remote_root(mount_target: str, run_id: str) -> str:
    """Root directory on the shared mount for this binary parity run."""
    return f"{mount_target.rstrip('/')}/cfd-binary/{run_id}"


def _binary_report_remote_path(mount_target: str, run_id: str) -> str:
    """Return the glob pattern for benchmark_results.json under the binary output dir.

    The binary creates an attempt subdirectory at runtime, so the layout is:
      output/<run-subid>/physicsnemo-cfd-surface-attempt-<id>/benchmark-output/benchmark_results.json
    """
    root = remote_root(mount_target, run_id)
    return f"{root}/output/{BINARY_RUN_SUBID}/*/benchmark-output/benchmark_results.json"


def _lepton_job_name(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    candidate = f"{prefix}-{slug}"
    if len(candidate) <= 36:
        return candidate
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:27].rstrip('-')}-{suffix}"


def job_name(profile: Mapping[str, Any], run_id: str) -> str:
    domain = re.sub(r"[^a-z0-9]+", "-", str(profile["domain"]).lower()).strip("-")
    return _lepton_job_name(f"pn-cfd-binary-{domain}", run_id)


# ---------------------------------------------------------------------------
# Binary job construction
# ---------------------------------------------------------------------------


def build_binary_command(
    *,
    request_text: str,
    mount_target: str,
    run_id: str,
) -> str:
    """Build the --command string for the binary-mode Lepton job.

    Writes the request JSON onto the shared mount, then invokes
    ``physicsnemo-serve infer`` with fixed plugin and runtime paths from the
    scicomp-ferroflux-cmd image.
    """
    root = remote_root(mount_target, run_id)
    request_path = f"{root}/request.json"
    output_dir = f"{root}/output"
    log_path = f"{root}/binary-job.log"

    infer_args = [
        "physicsnemo-serve",
        "infer",
        "--plugin",
        BINARY_PLUGIN_DIR,
        "--runtime-dir",
        BINARY_RUNTIME_DIR,
        "--request",
        request_path,
        "--output-dir",
        output_dir,
        "--run-id",
        BINARY_RUN_SUBID,
        "--device",
        "0",
    ]

    # mkdir -p is intentional: Lepton's --max-failure-retry reruns the same
    # command, so we must not reject an existing root directory. Results are
    # isolated in attempt-specific subdirectories created by the binary, and
    # the report reader always picks the newest by mtime.
    setup = [
        f"mkdir -p {shlex.quote(root)}",
        base64_write_command(request_path, request_text),
    ]
    return (
        "set -o pipefail; "
        + "; ".join(setup)
        + "; "
        + f"{shell_join(infer_args)} 2>&1 | tee {shlex.quote(log_path)}; "
        + "rc=${PIPESTATUS[0]}; exit $rc"
    )


def build_binary_job_args(
    args: argparse.Namespace,
    *,
    binary_image: str,
    nfs_path: str,
    run_id: str,
    profile: Mapping[str, Any],
    binary_command: str,
) -> list[str]:
    hf_cache = f"{args.mount_target}/.cache/physicsnemo-cfd/huggingface"
    model_cache = f"{args.mount_target}/.cache/physicsnemo-cfd/models"
    env_flags: list[str] = []
    for key, value in _PREFETCH_ENV.items():
        env_flags += ["--env", f"{key}={value}"]
    env_flags += ["--env", f"HF_HOME={hf_cache}"]
    env_flags += ["--env", f"PHYSICSNEMO_CFD_MODEL_CACHE={model_cache}"]
    return [
        "--name",
        job_name(profile, run_id),
        "--container-image",
        binary_image,
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.resource_shape,
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
        "--max-failure-retry",
        "1",
        *env_flags,
        "--command",
        binary_command,
    ]


# ---------------------------------------------------------------------------
# REST QA helpers
# ---------------------------------------------------------------------------


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
    rest_env = {**env, "QA_CFD_E2E_REQUEST_PATH": str(request_path)}
    returncode, _ = run_streaming(
        command,
        env=rest_env,
        artifact_path=artifact_dir.parent / "rest-qa.log",
    )
    if returncode != 0:
        raise RuntimeError(f"REST QA exited with code {returncode}")
    return discover_rest_evidence(artifact_dir / str(rest["evidence_subdir"]))


# ---------------------------------------------------------------------------
# Binary report retrieval
# ---------------------------------------------------------------------------


def _load_binary_report_if_available(
    nfs_path: str, run_id: str, mount_target: str
) -> list[Any] | None:
    """Try to read benchmark_results.json directly from the NFS mount.

    The binary nests the file under an attempt subdirectory, so we glob for it.
    """
    glob_pattern = _binary_report_remote_path(mount_target, run_id)
    rel_posix = glob_pattern.removeprefix(mount_target.rstrip("/") + "/")
    matches = sorted(Path(nfs_path).glob(rel_posix))
    if not matches:
        return None
    # Take the most recently modified attempt if multiple exist.
    path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def fetch_binary_report_via_reader_job(
    args: argparse.Namespace,
    *,
    service_image: str,
    nfs_path: str,
    run_id: str,
    env: dict[str, str],
    artifact_dir: Path,
) -> list[Any] | None:
    """Submit a small CPU reader job (using the service image) to read the
    benchmark_results.json from NFS and echo it back via Lepton job logs."""
    glob_pattern = _binary_report_remote_path(args.mount_target, run_id)
    reader_name = _lepton_job_name("pn-cfd-binary-read", run_id)
    reader_python = (
        "import base64, json\n"
        "from pathlib import Path\n"
        f"matches = sorted(Path('/').glob({glob_pattern.lstrip('/')!r}))\n"
        "path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None\n"
        "assert path and path.is_file(), f'benchmark_results.json not found: {glob_pattern}'\n"
        "text = path.read_text(encoding='utf-8')\n"
        f"print({BINARY_REPORT_BEGIN!r}, flush=True)\n"
        "print(base64.b64encode(text.encode('utf-8')).decode('ascii'), flush=True)\n"
        f"print({BINARY_REPORT_END!r}, flush=True)\n"
    )
    encoded = base64.b64encode(reader_python.encode("utf-8")).decode("ascii")
    reader_command = (
        f"printf %s {shlex.quote(encoded)} | base64 -d | python3 2>&1; sleep 30"
    )
    reader_args = [
        "--name",
        reader_name,
        "--container-image",
        service_image,
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
    print(f"==> Fetching binary report with reader job: {reader_name}", flush=True)
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
        return extract_marked_report(logs)
    finally:
        run_streaming(["lep", "job", "stop", "-i", reader_job_id], env=env)
        run_streaming(["lep", "job", "remove", "-i", reader_job_id], env=env)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915
    matrix_profile, matrix_request = MATRIX_CONFIGS[args.matrix]
    if args.profile is None:
        args.profile = str(matrix_profile)
    if args.rest_request_path is None:
        args.rest_request_path = str(matrix_request)

    profile_path = resolve_path(args.profile)
    profile_text = profile_path.read_text(encoding="utf-8")
    profile = json.loads(profile_text)
    if not isinstance(profile, dict):
        raise ParityContractError("parity profile must contain a JSON object")
    validate_profile(profile)

    run_id = validate_run_id(args.run_id or generate_run_id())
    service_image = image_full_reference(args.image_tag, args.image_name)
    binary_image = args.binary_image
    nfs_path = f"{args.nfs_mount_base.rstrip('/')}/{args.lustre_dir}"
    artifact_root = resolve_path(args.artifact_dir)
    run_artifact_dir = artifact_root / "cfd-binary" / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_artifact_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "final_result": "failed",
        "run_id": run_id,
        "profile_id": profile["profile_id"],
        "images": {
            "service": service_image,
            "binary": binary_image,
        },
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
        print(f"\n==> Received signal {signum}; cleaning up binary job", flush=True)
        cleanup_job()
        raise SystemExit(128 + signum)

    atexit.register(cleanup_job)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # -- Phase 1: REST QA (service-mode) ---------------------------------
        if args.dry_run and not args.rest_evidence_dir:
            # Config-check dry-run: validate setup and print what would be run
            # without needing real REST evidence or submitting anything.
            summary["job"] = {
                "name": job_name(profile, run_id),
                "binary_image": binary_image,
                "node_group": args.node_group,
                "resource_shape": args.resource_shape,
                "mount": (
                    f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}"
                ),
                "note": (
                    "command omitted — provide --rest-evidence-dir for a full dry-run"
                ),
            }
            summary["final_result"] = "dry-run"
            write_json_atomic(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        rest_qa_log = run_artifact_dir / "rest-qa.log"
        summary["artifacts"]["rest_job_log"] = str(rest_qa_log)

        if args.rest_evidence_dir:
            evidence_dir = resolve_path(args.rest_evidence_dir)
        else:
            try:
                evidence_dir = run_rest_qa(
                    args,
                    profile=profile,
                    image=service_image,
                    artifact_dir=run_artifact_dir / "rest-qa",
                    env=env,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc}  —  see {rest_qa_log} for full output"
                ) from exc
        summary["artifacts"]["rest_evidence_dir"] = str(evidence_dir)

        # -- Phase 2: Build handoff from REST evidence -----------------------
        handoff = build_handoff(
            evidence_dir=evidence_dir,
            profile=profile,
            parity_run_id=run_id,
            image=service_image,
            mount_target=args.mount_target,
        )
        handoff_path = run_artifact_dir / "parity-handoff.json"
        write_json_atomic(handoff_path, handoff)
        summary["rest_run_id"] = handoff["rest_run_id"]
        summary["artifacts"]["handoff"] = str(handoff_path)
        rest_output_nfs = f"{nfs_path}/{Path(handoff['rest']['report_relpath']).parent}"
        summary["artifacts"]["rest_output_dir"] = rest_output_nfs
        print(f"\n==> REST output dir:    {rest_output_nfs}", flush=True)

        # -- Phase 3: Prepare binary job -------------------------------------
        # Use the canonical matrix request file so the binary always runs the
        # same request regardless of how the REST evidence was captured.
        matrix_request_path = Path(args.rest_request_path)
        request_text = matrix_request_path.read_text(encoding="utf-8")

        binary_command = build_binary_command(
            request_text=request_text,
            mount_target=args.mount_target,
            run_id=run_id,
        )
        job_args = build_binary_job_args(
            args,
            binary_image=binary_image,
            nfs_path=nfs_path,
            run_id=run_id,
            profile=profile,
            binary_command=binary_command,
        )
        summary["job"] = {
            "name": job_name(profile, run_id),
            "binary_image": binary_image,
            "node_group": args.node_group,
            "resource_shape": args.resource_shape,
            "mount": f"{nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
            "binary_command_sha256": hashlib.sha256(
                binary_command.encode("utf-8")
            ).hexdigest(),
        }

        if args.dry_run:
            summary["final_result"] = "dry-run"
            write_json_atomic(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        # -- Phase 4: Submit binary-mode Lepton job --------------------------
        login_if_needed(args, env)
        print(
            "+ lep job create "
            f"--name {shlex.quote(str(summary['job']['name']))} "
            f"--container-image {shlex.quote(binary_image)} "
            f"--node-group {shlex.quote(args.node_group)} "
            f"--resource-shape {shlex.quote(args.resource_shape)} "
            f"--mount {shlex.quote(str(summary['job']['mount']))} "
            "--command <binary-infer-command>",
            flush=True,
        )
        create_returncode, create_output = run_streaming(
            ["lep", "job", "create", *job_args],
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

        # -- Phase 5: Poll until done ----------------------------------------
        job_returncode = poll_job(
            job_id=job_id,
            env=env,
            timeout_seconds=args.job_timeout,
            interval_seconds=args.job_poll_interval,
        )
        job_finished = True
        summary["job"]["exit_code"] = job_returncode
        if job_returncode != 0:
            run_streaming(["lep", "job", "stop", "-i", job_id], env=env)

        binary_output_nfs = f"{nfs_path}/cfd-binary/{run_id}/output/{BINARY_RUN_SUBID}/<attempt>/benchmark-output"
        summary["artifacts"]["binary_output_dir"] = binary_output_nfs
        summary["artifacts"]["binary_job_log"] = str(
            run_artifact_dir / "binary-job.log"
        )
        print(f"\n==> Binary output dir:  {binary_output_nfs}", flush=True)

        capture_job_logs(
            job_id=job_id,
            env=env,
            artifact_path=run_artifact_dir / "binary-job.log",
        )

        # -- Phase 6: Load binary benchmark_results.json ---------------------
        binary_report = _load_binary_report_if_available(
            nfs_path, run_id, args.mount_target
        )
        if binary_report is None:
            binary_report = fetch_binary_report_via_reader_job(
                args,
                service_image=service_image,
                nfs_path=nfs_path,
                run_id=run_id,
                env=env,
                artifact_dir=run_artifact_dir,
            )
        if binary_report is None:
            raise RuntimeError(
                "binary job produced no readable benchmark_results.json; "
                f"check {run_artifact_dir / 'binary-job.log'}"
            )
        write_json_atomic(
            run_artifact_dir / "binary-benchmark_results.json", binary_report
        )

        # -- Phase 7: Load REST benchmark_results.json (already local) -------
        rest_report = json.loads(
            (evidence_dir / "benchmark_results.json").read_text(encoding="utf-8")
        )
        write_json_atomic(run_artifact_dir / "rest-benchmark_results.json", rest_report)

        # -- Phase 8: Compare ------------------------------------------------
        if job_returncode != 0:
            raise RuntimeError(
                f"binary Lepton job exited with code {job_returncode}; "
                "skipping comparison"
            )

        comparison = compare_reports(
            rest_report=rest_report,
            direct_report=binary_report,
            comparison=profile["comparison"],
        )
        write_json_atomic(run_artifact_dir / "comparison.json", comparison)
        summary["comparison"] = comparison
        summary["final_result"] = (
            "passed" if comparison.get("status") == "passed" else "failed"
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
        _print_result(summary)

    return 0 if summary["final_result"] in {"passed", "dry-run"} else 1


_FAILURE_PATTERNS = re.compile(
    r"FAILED|ERROR|Error|Exception|assert|Traceback|timeout|exit code",
    re.IGNORECASE,
)


def _print_log_failures(log_path: Path, context: int = 3) -> None:
    """Print lines from log_path that look like failures, with surrounding context."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if not lines:
        return
    hit_indices: set[int] = set()
    for i, line in enumerate(lines):
        if _FAILURE_PATTERNS.search(line):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                hit_indices.add(j)
    if not hit_indices:
        return
    print(f"\n  Failure context from {log_path}:", flush=True)
    prev: int | None = None
    for i in sorted(hit_indices):
        if prev is not None and i > prev + 1:
            print("    ...", flush=True)
        print(f"    {lines[i]}", flush=True)
        prev = i


def _print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print per-model, per-metric pass/fail table from compare_reports output."""
    metrics: list[dict[str, Any]] = comparison.get("metrics", [])
    if not metrics:
        return

    # Group by model
    by_model: dict[str, list[dict[str, Any]]] = {}
    for m in metrics:
        by_model.setdefault(m["model"], []).append(m)

    print(flush=True)
    for model, rows in by_model.items():
        model_pass = all(r["matches"] for r in rows)
        status = "PASS" if model_pass else "FAIL"
        print(f"  [{status}] {model}", flush=True)
        for r in rows:
            metric_status = "PASS" if r["matches"] else "FAIL"
            rest_val = r["rest"]
            direct_val = r["direct"]
            rel_diff = r["relative_difference"]
            rel_str = f"{rel_diff:.2e}" if rel_diff is not None else "n/a"
            case = r.get("case_id") or "-"
            print(
                f"         [{metric_status}] {r['metric']:35s}"
                f"  rest={rest_val:.6f}  binary={direct_val:.6f}"
                f"  rel_diff={rel_str}  case={case}",
                flush=True,
            )


def _print_result(summary: dict[str, Any]) -> None:
    result = summary.get("final_result", "failed")
    run_id = summary.get("run_id", "?")
    artifacts = summary.get("artifacts", {})

    print(f"\n{'=' * 60}", flush=True)

    rest_out = artifacts.get("rest_output_dir")
    if rest_out:
        print(f"  REST outputs:      {rest_out}", flush=True)

    rest_log = artifacts.get("rest_job_log")
    if rest_log:
        print(f"  REST job log:      {rest_log}", flush=True)

    binary_out = artifacts.get("binary_output_dir")
    if binary_out:
        print(f"  Binary outputs:    {binary_out}", flush=True)

    binary_log = artifacts.get("binary_job_log")
    if binary_log:
        print(f"  Binary job log:    {binary_log}", flush=True)

    local_dir = artifacts.get("local_dir")
    if local_dir:
        print(f"  Local artifacts:   {local_dir}", flush=True)

    comparison = summary.get("comparison")
    if comparison:
        _print_comparison_table(comparison)
        errors = comparison.get("errors", [])
        if errors:
            print(f"\n  Comparison errors ({len(errors)}):", flush=True)
            for err in errors[:10]:
                print(f"    - {err}", flush=True)
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more", flush=True)

    error = summary.get("error")
    if error:
        print(
            f"\n  Error: {error.get('type')}: {error.get('message')}",
            flush=True,
        )
        rest_log_path = artifacts.get("rest_job_log")
        if rest_log_path and Path(rest_log_path).is_file():
            _print_log_failures(Path(rest_log_path))

    print(f"{'=' * 60}", flush=True)
    print(f"  Binary parity run {run_id}: {result.upper()}", flush=True)
    print(f"{'=' * 60}\n", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Service-mode image (for REST QA and reader jobs)
    parser.add_argument(
        "--image-tag",
        required=True,
        help="Tag for the service-mode image (used for the REST QA endpoint).",
    )
    registry = _DEPLOY_CONFIG.get("docker_registry", "")
    default_service_image = (
        f"{registry}/{_DEPLOY_CONFIG.get('image_name', 'physicsnemo-serve')}"
        if registry
        else _DEPLOY_CONFIG.get("image_name", "physicsnemo-serve")
    )
    parser.add_argument("--image-name", default=default_service_image)

    # Binary-mode image
    default_binary_registry = registry or "nvcr.io/nvidia"
    parser.add_argument(
        "--binary-image",
        default=os.environ.get("BINARY_IMAGE")
        or f"{default_binary_registry}/{DEFAULT_BINARY_IMAGE_NAME}:pr-14-1c397c1",
        help=(
            "Full image reference for the binary-mode container "
            "(scicomp-ferroflux-cmd). "
            "Defaults to BINARY_IMAGE env var or pr-14-1c397c1."
        ),
    )

    parser.add_argument(
        "--matrix",
        choices=["all", "three", "domino"],
        default="all",
        help=(
            "'three' runs domino_surface, fignet_surface, xmgn_surface "
            "(public_run_1_3model_request.json) — these have confirmed binary parity; "
            "'all' runs all 5 CFD models (public_run_1_full_matrix_request.json); "
            "'domino' runs domino_surface only (public_run_1_request.json). "
            "Sets the profile and request automatically; ignored if --profile or "
            "--rest-request-path are given explicitly."
        ),
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--rest-request-path",
        help=(
            "Override the profile's REST request fixture. "
            "The same request body is also injected into the binary job."
        ),
    )
    parser.add_argument(
        "--rest-evidence-dir",
        help=(
            "Skip the REST QA phase and reuse an existing evidence directory. "
            "Useful for re-running the binary phase after a successful REST run."
        ),
    )
    parser.add_argument("--run-id", help="Explicit run ID (default: random 10-char).")
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("QA_ARTIFACT_DIR", str(QA_ROOT / "artifacts")),
    )

    # Lepton workspace
    parser.add_argument(
        "--workspace-id",
        default=os.environ.get("LEPTON_WORKSPACE_ID")
        or _DEPLOY_CONFIG.get("lepton_workspace_id", ""),
    )
    parser.add_argument(
        "--workspace-token",
        default=os.environ.get("LEPTON_WORKSPACE_TOKEN", ""),
    )
    parser.add_argument(
        "--workspace-url",
        default=os.environ.get("LEPTON_WORKSPACE_URL", ""),
    )

    # Compute
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

    # Storage
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

    # Timing
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=23_400,
        help="Binary job timeout in seconds (default: 23400 = 6.5 h).",
    )
    parser.add_argument("--job-poll-interval", type=int, default=20)
    parser.add_argument("--reader-resource-shape", default="cpu.small")

    # Misc
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the job spec without submitting anything to Lepton.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("workspace_id", "node_group", "pull_secret", "nfs_mount_base"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    if not args.dry_run and not args.rest_evidence_dir and not args.workspace_token:
        raise ValueError(
            "--workspace-token is required when the REST QA phase is enabled"
        )
    if args.job_timeout <= 0 or args.job_poll_interval <= 0:
        raise ValueError("job timeout and poll interval must be positive")
    if not args.binary_image:
        raise ValueError("--binary-image is required")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
