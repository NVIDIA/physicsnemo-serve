#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Push-button QA runner: deploy → test → teardown (per workflow).

Each workflow plugin gets its own Lepton deployment. Tests for other
workflows are automatically skipped via the existing
skip_if_workflow_disabled() mechanism in the pytest helpers.

Deployments are strictly sequential (one at a time) to avoid exceeding
GPU resource limits.

Required environment variables:
    LEPTON_WORKSPACE_ID      Lepton workspace id (from deploy/config.yaml or env)
    LEPTON_WORKSPACE_TOKEN   Token to manage deployments (lep login)

Optional environment variables:
    LEPTON_ENDPOINT_TOKEN    Bearer token for the endpoint; generated when empty
    PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID
                             Rust-only: if set externally, only that single
                             workflow is deployed and tested (legacy mode)
    QA_POST_HEALTH_WAIT_SECS Explicit Rust multi-GPU cicd/full only: seconds to
                             wait after /health before pytest starts
                             (alias: QA_POST_HEALTH_SLEEP_SECS)

Usage:
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --suite smoke
    python scripts/run_qa.py --service python --image-tag v0.1.20260518.0 --suite cicd
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --suite stress
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --num-proc 1
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --suite cicd \
        --workflows e2s-deterministic,earth2-deterministic
"""

from __future__ import annotations

import argparse
import atexit
import io
import os
import re
import secrets
import signal
import string
import subprocess
import sys
import threading
import time
from pathlib import Path


def _ensure_line_buffered() -> None:
    """Rewrap stdout/stderr with line buffering when piped (block-buffered)."""
    if os.environ.get("PYTHONUNBUFFERED"):
        return
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, line_buffering=True
    )


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
INFERENCE_DIR = REPO_ROOT / "inference"


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
from config import load_deploy_config  # noqa: E402


DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy-to-lepton.sh"
TEARDOWN_SCRIPT = SCRIPTS_DIR / "teardown-lepton.sh"

HEALTH_POLL_INTERVAL = 15  # seconds
HEALTH_POLL_TIMEOUT = 600  # seconds
ENDPOINT_LOG_POLL_INTERVAL = 30  # seconds
ENDPOINT_LOG_COMMAND_TIMEOUT = 45  # seconds
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"
ENDPOINT_TOKEN_LENGTH = 32
MULTIGPU_ENV_NAMES = (
    "QA_MULTIGPU_GPU_COUNT",
    "MULTIGPU_GPU_COUNT",
    "QA_MULTIGPU_WORKFLOWS",
    "MULTIGPU_WORKFLOWS",
)

ALL_WORKFLOW_PLUGIN_IDS = [
    "e2s-deterministic",
    "e2s-deterministic-fcn",
    "e2s-diagnostic",
    "e2s-ensemble",
    "e2s-deterministic-earth2",
    "e2s-stormcast-fcn3",
    "e2s-example-user",
    "earth2-deterministic",
    "earth2-deterministic-batch",
    "earth2-ensemble",
    "earth2-ensemble-fanout",
    "e2s-foundry-fcn3",
    "e2s-foundry-fcn3-stormscope-goes",
]


WORKFLOW_ALL = "__all__"


def determine_workflows(
    *,
    service: str,
    workflows_arg: str | None,
) -> list[str]:
    """Determine which workflow plugin IDs to iterate over.

    Priority:
    1. PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID env var (single workflow, legacy mode)
    2. --workflows CLI arg (comma-separated list)
    3. For the Rust service: full list of all known plugin IDs (one deploy each)
       For the Python service: single deployment running all workflows at once
    """
    env_plugin_id = os.environ.get("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "").strip()
    if env_plugin_id:
        return [env_plugin_id]

    if workflows_arg:
        return [w.strip() for w in workflows_arg.split(",") if w.strip()]

    if service == "python":
        return [WORKFLOW_ALL]

    return list(ALL_WORKFLOW_PLUGIN_IDS)


def command_output_text(output: str | bytes | None) -> str:
    """Normalize captured subprocess output to text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Error: environment variable {name} is required but not set.")
    return value


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            sys.exit(f"Error: environment variable {name} must be an integer.")
    return default


def multigpu_env_requested() -> bool:
    """Return whether the caller explicitly selected multi-GPU QA behavior."""
    return any(os.environ.get(name, "").strip() for name in MULTIGPU_ENV_NAMES)


def post_health_wait_applies(
    *,
    service: str,
    suite: str,
    multigpu_requested: bool | None = None,
) -> bool:
    """Return whether this QA run explicitly targets multi-GPU CICD coverage."""
    if multigpu_requested is None:
        multigpu_requested = multigpu_env_requested()
    return service == "rust" and suite in {"cicd", "full"} and multigpu_requested


def post_health_wait(seconds: int, *, service: str, suite: str) -> None:
    """Optional grace period for multi-GPU-capable runs after HTTP health."""
    if seconds <= 0:
        return
    if not post_health_wait_applies(service=service, suite=suite):
        print(
            "==> Skipping post-health wait; it only applies to explicit "
            "Rust cicd/full multi-GPU runs",
            flush=True,
        )
        return
    print(
        f"==> Waiting {seconds}s after health check before running tests",
        flush=True,
    )
    time.sleep(seconds)


def get_or_generate_endpoint_token() -> str:
    """Return a provided endpoint token or generate a temporary bearer token."""
    value = os.environ.get("LEPTON_ENDPOINT_TOKEN", "").strip()
    if value:
        return value

    alphabet = string.ascii_letters + string.digits
    token = "".join(secrets.choice(alphabet) for _ in range(ENDPOINT_TOKEN_LENGTH))
    print("==> Generated temporary LEPTON_ENDPOINT_TOKEN for this QA run", flush=True)
    return token


def default_endpoint_name(source: str) -> str:
    """Generate the endpoint name here so cleanup knows it before deploy runs."""
    user = os.environ.get("USER") or os.environ.get("GITLAB_USER_LOGIN") or "ci"
    user = re.sub(r"[^a-z0-9-]+", "-", user.lower()).strip("-") or "ci"
    short = "pnserve" if source == "physicsnemo-serve" else source
    suffix = f"{secrets.randbelow(90_000_000) + 10_000_000:08d}"
    return f"{user}-{short}-ep-{suffix}"


def parse_endpoint_url(output: str) -> str | None:
    """Extract the endpoint URL from deploy-to-lepton.sh output."""
    for line in output.splitlines():
        match = re.match(r"\s*URL:\s*(https?://\S+)", line)
        if match:
            return match.group(1)
    for line in output.splitlines():
        if "xenon.lepton.run" in line or "dgxc-lepton" in line:
            match = re.search(r"(https?://\S+)", line)
            if match:
                return match.group(1)
    return None


def parse_endpoint_name(output: str) -> str | None:
    """Extract the endpoint name from deploy-to-lepton.sh output."""
    for line in output.splitlines():
        match = re.match(r"\s*endpoint-name\s*:\s*(\S+)", line)
        if match:
            return match.group(1)
    for line in output.splitlines():
        match = re.match(r"\s*Endpoint:\s*(\S+)", line)
        if match:
            return match.group(1)
    return None


def run_streaming(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
) -> tuple[int, str]:
    """Run a subprocess, stream its output live, and return captured output."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)

    return process.wait(), "".join(output)


def output_delta(previous: str, current: str) -> str:
    """Return the new part of a repeated CLI log snapshot."""
    if not current or current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]

    previous_lines = set(previous.splitlines())
    new_lines = [line for line in current.splitlines() if line not in previous_lines]
    if not new_lines:
        return ""
    return "\n".join(new_lines) + "\n"


def capture_endpoint_logs_once(
    endpoint_name: str,
    artifact_path: Path,
    *,
    timeout: int = ENDPOINT_LOG_COMMAND_TIMEOUT,
) -> bool:
    """Fetch endpoint logs once and append them to the artifact before teardown."""
    print(f"\n==> Capturing endpoint logs for {endpoint_name}", flush=True)
    print(f"==> Endpoint log artifact: {artifact_path}", flush=True)
    artifact_path.parent.mkdir(exist_ok=True, parents=True)

    with artifact_path.open("a", encoding="utf-8", buffering=1) as artifact:
        artifact.write(f"\nFinal endpoint log capture for {endpoint_name}\n")
        artifact.write(
            f"Captured at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
        )

        try:
            result = subprocess.run(
                ["lep", "endpoint", "log", "-n", endpoint_name],
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as err:
            output = command_output_text(err.output)
            warning = f"==> Final endpoint log capture timed out after {timeout}s"
            print(warning, flush=True)
            artifact.write(warning + "\n")
            if output.strip():
                print(output, end="" if output.endswith("\n") else "\n", flush=True)
                artifact.write(output)
                if not output.endswith("\n"):
                    artifact.write("\n")
            return False

        output = result.stdout or ""
        if output.strip():
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
            artifact.write(output)
            if not output.endswith("\n"):
                artifact.write("\n")

        if result.returncode != 0:
            warning = (
                f"==> Final endpoint log capture exited with code {result.returncode}"
            )
            print(warning, flush=True)
            artifact.write(warning + "\n")
            return False

    return True


def stream_endpoint_logs(
    endpoint_name: str,
    stop_event: threading.Event,
    interval: int,
    artifact_path: Path,
):
    """Poll endpoint logs in the background, print new output, and save it."""
    print(
        f"==> Streaming endpoint logs for {endpoint_name} every {interval}s", flush=True
    )
    print(f"==> Endpoint log artifact: {artifact_path}", flush=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    previous_output = ""
    previous_error = ""
    env = os.environ.copy()

    with artifact_path.open("a", encoding="utf-8", buffering=1) as artifact:
        artifact.write(f"Endpoint logs for {endpoint_name}\n")
        artifact.write(
            f"Started at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
        )

        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["lep", "endpoint", "log", "-n", endpoint_name],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=ENDPOINT_LOG_COMMAND_TIMEOUT,
                )
            except subprocess.TimeoutExpired as err:
                error = (
                    f"lep endpoint log timed out after {ENDPOINT_LOG_COMMAND_TIMEOUT}s"
                )
                if error != previous_error:
                    warning = f"==> Endpoint log polling warning: {error}\n"
                    print(warning, end="", flush=True)
                    artifact.write(warning)
                    previous_error = error
                output = command_output_text(err.output)
                new_output = output_delta(previous_output, output)
                previous_output = output
                if new_output.strip():
                    print("\n==> Endpoint logs", flush=True)
                    print(
                        new_output,
                        end="" if new_output.endswith("\n") else "\n",
                        flush=True,
                    )
                    artifact.write(new_output)
                    if not new_output.endswith("\n"):
                        artifact.write("\n")
                stop_event.wait(interval)
                continue

            output = result.stdout or ""
            if result.returncode != 0:
                error = (
                    output.strip()
                    or f"lep endpoint log exited with code {result.returncode}"
                )
                if error != previous_error:
                    warning = f"==> Endpoint log polling warning:\n{error}\n"
                    print(warning, end="", flush=True)
                    artifact.write(warning)
                    previous_error = error
                stop_event.wait(interval)
                continue

            previous_error = ""
            new_output = output_delta(previous_output, output)
            previous_output = output
            if new_output.strip():
                print("\n==> Endpoint logs", flush=True)
                print(
                    new_output,
                    end="" if new_output.endswith("\n") else "\n",
                    flush=True,
                )
                artifact.write(new_output)
                if not new_output.endswith("\n"):
                    artifact.write("\n")

            stop_event.wait(interval)


def health_check(url: str, token: str, timeout: int = HEALTH_POLL_TIMEOUT) -> bool:
    """Poll the service health endpoint until it responds 200."""
    import requests

    health_url = url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    print(f"==> Health-checking {health_url} (timeout {timeout}s)")

    while time.time() < deadline:
        try:
            resp = requests.get(
                health_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"    Health OK ({resp.status_code})")
                return True
            remaining = int(deadline - time.time())
            print(f"    Not ready (HTTP {resp.status_code}), {remaining}s remaining...")
        except (requests.ConnectionError, requests.Timeout, OSError) as e:
            remaining = int(deadline - time.time())
            print(f"    Not ready ({e}), {remaining}s remaining...")
        time.sleep(HEALTH_POLL_INTERVAL)

    print("Error: health check timed out")
    return False


def deploy(
    *,
    source: str,
    service: str,
    image_tag: str,
    workspace_id: str,
    workspace_token: str,
    endpoint_token: str,
    endpoint_name: str | None = None,
    nfs_path: str,
    container_envs: list[str] | None = None,
) -> tuple[str, str]:
    """Run deploy-to-lepton.sh --skip-build --skip-push.

    Returns (endpoint_url, endpoint_name).
    """
    cmd = [
        str(DEPLOY_SCRIPT),
        "--source",
        source,
        "--skip-build",
        "--skip-push",
        "--image-tag",
        image_tag,
        "--workspace-id",
        workspace_id,
        "--nfs-path",
        nfs_path,
    ]
    if endpoint_name:
        cmd += ["--endpoint-name", endpoint_name]
    for env_pair in container_envs or []:
        cmd += ["--env", env_pair]

    print(f"==> Deploying {source} (image tag: {image_tag})")
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_TOKEN": workspace_token,
        "LEPTON_ENDPOINT_TOKEN": endpoint_token,
    }
    returncode, output = run_streaming(cmd, env=env)
    if returncode != 0:
        sys.exit(f"Error: deploy-to-lepton.sh exited with code {returncode}")

    url = parse_endpoint_url(output)
    if not url:
        sys.exit("Error: could not parse endpoint URL from deploy output")

    name = endpoint_name or parse_endpoint_name(output)
    if not name:
        sys.exit("Error: could not determine endpoint name from deploy output")

    print(f"==> Endpoint: {name}")
    print(f"==> URL: {url}")
    return url, name


def teardown(
    *,
    workspace_id: str,
    workspace_token: str,
    endpoint_name: str,
):
    """Remove the Lepton deployment.

    TODO: On test failure, consider terminating (scale to 0) instead of
    deleting so that container logs remain accessible for debugging. Could
    add a --keep-on-failure flag that checks the pytest exit code.
    """
    print(f"\n==> Tearing down deployment: {endpoint_name}")
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_ID": workspace_id,
        "LEPTON_WORKSPACE_TOKEN": workspace_token,
        "LEPTON_ENDPOINT_NAME": endpoint_name,
    }
    returncode, _ = run_streaming([str(TEARDOWN_SCRIPT)], env=env)
    if returncode != 0:
        print(f"Warning: teardown exited with code {returncode}", file=sys.stderr)


def run_pytest(
    *,
    suite: str,
    service: str,
    endpoint_url: str,
    endpoint_token: str,
    num_proc: int,
    test_filter: str | None = None,
    artifact_dir: Path | None = None,
) -> int:
    """Run pytest with the appropriate marker and return the exit code."""
    urls = endpoint_url
    if num_proc > 1:
        urls = ",".join([endpoint_url] * num_proc)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--urls",
        urls,
        "--service",
        service,
        "-v",
    ]
    if artifact_dir is not None:
        results_dir = artifact_dir / "pytest-results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        workflow_slug = os.environ.get(
            "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", ""
        ).strip()
        if workflow_slug:
            junit_path = results_dir / f"results_{workflow_slug}_{timestamp}.xml"
        else:
            junit_path = results_dir / f"results_{timestamp}.xml"
        cmd += [f"--junitxml={junit_path}"]
        print(f"==> JUnit XML report: {junit_path}", flush=True)
    if num_proc > 1:
        cmd += ["-n", str(num_proc)]
    if suite:
        cmd += ["-m", suite]
    if test_filter:
        cmd += ["-k", test_filter]
    print(f"==> Running: {' '.join(cmd)}")
    env = {
        **os.environ,
        "LEPTON_ENDPOINT_TOKEN": endpoint_token,
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(cmd, cwd=str(INFERENCE_DIR), env=env)
    return result.returncode


_atexit_endpoint_name: dict[str, str | None] = {"name": None}


def run_one_workflow(
    *,
    workflow_id: str,
    source: str,
    service: str,
    image_tag: str,
    workspace_id: str,
    workspace_token: str,
    endpoint_token: str,
    nfs_path: str,
    suite: str,
    test_filter: str | None,
    num_proc: int,
    skip_teardown: bool,
    stream_logs: bool,
    log_interval: int,
    artifact_dir: Path,
    post_health_wait_secs: int,
    endpoint_name_override: str | None = None,
) -> int:
    """Deploy, test, and teardown a single workflow. Returns pytest exit code."""
    display_name = workflow_id if workflow_id != WORKFLOW_ALL else "ALL (single deploy)"
    print(
        f"\n{'=' * 76}\n  WORKFLOW: {display_name}\n{'=' * 76}",
        flush=True,
    )

    is_single_deploy = workflow_id == WORKFLOW_ALL
    container_envs = (
        []
        if is_single_deploy
        else [f"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={workflow_id}"]
    )
    endpoint_name = endpoint_name_override or default_endpoint_name(source)
    endpoint_log_artifact = artifact_dir / "endpoint-logs" / f"{endpoint_name}.log"

    log_stop_event = threading.Event()
    log_thread: threading.Thread | None = None

    def _stop_endpoint_logs():
        log_stop_event.set()
        if log_thread and log_thread.is_alive():
            log_thread.join(timeout=5)

    def _teardown_this():
        _stop_endpoint_logs()
        _atexit_endpoint_name["name"] = None
        if not skip_teardown:
            teardown(
                workspace_id=workspace_id,
                workspace_token=workspace_token,
                endpoint_name=endpoint_name,
            )

    _atexit_endpoint_name["name"] = endpoint_name
    try:
        try:
            url, _ = deploy(
                source=source,
                service=service,
                image_tag=image_tag,
                workspace_id=workspace_id,
                workspace_token=workspace_token,
                endpoint_token=endpoint_token,
                endpoint_name=endpoint_name,
                nfs_path=nfs_path,
                container_envs=container_envs,
            )
        except SystemExit:
            print(
                f"==> Deploy FAILED for workflow {workflow_id}",
                flush=True,
            )
            return 1

        if stream_logs:
            log_thread = threading.Thread(
                target=stream_endpoint_logs,
                args=(
                    endpoint_name,
                    log_stop_event,
                    log_interval,
                    endpoint_log_artifact,
                ),
                daemon=True,
            )
            log_thread.start()

        if not health_check(url, endpoint_token):
            _stop_endpoint_logs()
            capture_endpoint_logs_once(endpoint_name, endpoint_log_artifact)
            print(
                f"==> Health check FAILED for workflow {workflow_id}; skipping tests",
                flush=True,
            )
            return 1

        post_health_wait(post_health_wait_secs, service=service, suite=suite)

        suite_marker = suite if suite != "full" else ""
        pytest_env_override = (
            {}
            if is_single_deploy
            else {"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": workflow_id}
        )
        old_env = os.environ.copy()
        os.environ.update(pytest_env_override)
        exit_code = 1
        try:
            exit_code = run_pytest(
                suite=suite_marker,
                service=service,
                endpoint_url=url,
                endpoint_token=endpoint_token,
                num_proc=num_proc,
                test_filter=test_filter,
                artifact_dir=artifact_dir,
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        if exit_code != 0:
            _stop_endpoint_logs()
            capture_endpoint_logs_once(endpoint_name, endpoint_log_artifact)

        return exit_code
    finally:
        _teardown_this()


def main():
    _ensure_line_buffered()
    parser = argparse.ArgumentParser(
        description="Push-button QA: deploy → test → teardown (per workflow)",
    )
    parser.add_argument(
        "--service",
        choices=["python", "rust"],
        required=True,
        help="Service backend type (determines --source and pytest --service)",
    )
    parser.add_argument(
        "--image-tag",
        required=True,
        help="Docker image tag to deploy (image must already be pushed)",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "cicd", "basic", "negative", "stress", "full"],
        default="smoke",
        help="Test suite to run (pytest marker). 'full' runs without marker filter.",
    )
    parser.add_argument(
        "-k",
        dest="test_filter",
        default=None,
        help="Pytest -k expression to filter tests (e.g. 'test_ensemble_workflow').",
    )
    parser.add_argument(
        "--workflows",
        default=None,
        help=(
            "Comma-separated list of workflow plugin IDs to test. "
            "Each gets its own deployment. Default: all known workflows "
            "(or single workflow if PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID is set)."
        ),
    )
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help=(
            "Override endpoint name (only valid with a single workflow; "
            "ignored in multi-workflow mode)."
        ),
    )
    parser.add_argument(
        "--lustre-dir",
        default=os.environ.get("USER", "shared"),
        help="Subdirectory under NFS mount base to mount (default: $USER)",
    )
    parser.add_argument(
        "--skip-teardown",
        action="store_true",
        help=(
            "Leave the deployment running after tests (for debugging). "
            "Only applies when testing a single workflow; ignored in "
            "multi-workflow mode to avoid leaking GPU resources."
        ),
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=int(os.environ.get("PYTEST_NUM_PROC", "1")),
        help="Number of pytest processes to run. Values 0 or 1 run without xdist.",
    )
    parser.add_argument(
        "--stream-endpoint-logs",
        dest="stream_endpoint_logs",
        action="store_true",
        default=env_bool("STREAM_ENDPOINT_LOGS", True),
        help="Poll and print endpoint logs while tests run.",
    )
    parser.add_argument(
        "--no-endpoint-logs",
        dest="stream_endpoint_logs",
        action="store_false",
        help="Disable endpoint log polling.",
    )
    parser.add_argument(
        "--endpoint-log-interval",
        type=int,
        default=int(
            os.environ.get("ENDPOINT_LOG_INTERVAL", ENDPOINT_LOG_POLL_INTERVAL)
        ),
        help="Seconds between endpoint log polling attempts.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("QA_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)),
        help="Directory for QA artifacts such as endpoint logs.",
    )
    parser.add_argument(
        "--post-health-wait-secs",
        type=int,
        default=env_int_any(
            ("QA_POST_HEALTH_WAIT_SECS", "QA_POST_HEALTH_SLEEP_SECS"),
            0,
        ),
        help=(
            "Seconds to wait after /health passes before starting pytest for "
            "explicit Rust cicd/full multi-GPU runs "
            "(or QA_POST_HEALTH_WAIT_SECS / QA_POST_HEALTH_SLEEP_SECS)."
        ),
    )
    args = parser.parse_args()
    if args.num_proc < 0:
        sys.exit("Error: --num-proc must be 0 or greater")
    if args.endpoint_log_interval <= 0:
        sys.exit("Error: --endpoint-log-interval must be greater than 0")
    if args.post_health_wait_secs < 0:
        sys.exit("Error: --post-health-wait-secs must be 0 or greater")

    _deploy_cfg = load_deploy_config()
    workspace_id = os.environ.get("LEPTON_WORKSPACE_ID", "").strip() or _deploy_cfg.get(
        "lepton_workspace_id", ""
    )
    workspace_token = require_env("LEPTON_WORKSPACE_TOKEN")
    endpoint_token = get_or_generate_endpoint_token()

    source = "physicsnemo-serve" if args.service == "rust" else "earth2studio"
    nfs_path = f"{_deploy_cfg.get('nfs_mount_base', '/mnt/shared')}/{args.lustre_dir}"

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = Path.cwd() / artifact_dir

    workflows = determine_workflows(
        service=args.service,
        workflows_arg=args.workflows,
    )

    endpoint_name_override = None
    if len(workflows) == 1 and args.endpoint_name:
        endpoint_name_override = args.endpoint_name
    elif len(workflows) > 1 and args.endpoint_name:
        print(
            "Warning: --endpoint-name is ignored in multi-workflow mode "
            "(each workflow gets a unique endpoint name).",
            flush=True,
        )

    if len(workflows) > 1 and args.skip_teardown:
        print(
            "Warning: --skip-teardown only applies to single-workflow runs. "
            "Ignoring in multi-workflow mode to avoid leaking GPU resources.",
            flush=True,
        )

    skip_teardown = args.skip_teardown and len(workflows) == 1

    print(
        f"\n==> QA run: {len(workflows)} workflow(s) to test sequentially",
        flush=True,
    )
    for i, wf in enumerate(workflows, 1):
        print(f"    {i}. {wf}", flush=True)
    print(flush=True)

    # Install signal handler for clean teardown on interrupt.
    # The per-workflow function handles its own teardown via try/finally,
    # but signals need to be converted to SystemExit so the stack unwinds.
    def _handle_signal(signum, _frame):
        print(f"\n==> Received signal {signum}; exiting", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Safety net: if the process exits unexpectedly between deploy and
    # teardown (e.g. unhandled exception outside run_one_workflow), attempt
    # to tear down whatever endpoint might still be running. The per-workflow
    # try/finally handles normal cases; this covers truly unexpected exits.
    def _atexit_teardown():
        name = _atexit_endpoint_name.get("name")
        if name:
            teardown(
                workspace_id=workspace_id,
                workspace_token=workspace_token,
                endpoint_name=name,
            )

    atexit.register(_atexit_teardown)

    results: list[tuple[str, int]] = []
    for workflow_id in workflows:
        exit_code = run_one_workflow(
            workflow_id=workflow_id,
            source=source,
            service=args.service,
            image_tag=args.image_tag,
            workspace_id=workspace_id,
            workspace_token=workspace_token,
            endpoint_token=endpoint_token,
            nfs_path=nfs_path,
            suite=args.suite,
            test_filter=args.test_filter,
            num_proc=args.num_proc,
            skip_teardown=skip_teardown,
            stream_logs=args.stream_endpoint_logs,
            log_interval=args.endpoint_log_interval,
            artifact_dir=artifact_dir,
            post_health_wait_secs=args.post_health_wait_secs,
            endpoint_name_override=endpoint_name_override,
        )
        results.append((workflow_id, exit_code))

    # Print summary table
    print(
        f"\n{'=' * 76}\n  QA SUMMARY\n{'=' * 76}",
        flush=True,
    )
    any_failed = False
    for workflow_id, exit_code in results:
        status = "PASS" if exit_code == 0 else "FAIL"
        if exit_code != 0:
            any_failed = True
        print(f"  [{status}] {workflow_id} (exit code: {exit_code})", flush=True)

    total = len(results)
    passed = sum(1 for _, code in results if code == 0)
    failed = total - passed
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}", flush=True)
    print(f"{'=' * 76}\n", flush=True)

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
