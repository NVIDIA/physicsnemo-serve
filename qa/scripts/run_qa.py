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
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --suite cfd_e2e
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --num-proc 1
    python scripts/run_qa.py --service rust --image-tag v0.1.20260518.0 --suite cicd \
        --workflows e2s-deterministic,earth2-deterministic
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import re
import secrets
import signal
import string
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
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
sys.path.insert(0, str(INFERENCE_DIR))
from config import load_deploy_config  # noqa: E402


DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy-to-lepton.sh"
TEARDOWN_SCRIPT = SCRIPTS_DIR / "teardown-lepton.sh"

HEALTH_POLL_INTERVAL = 15  # seconds
HEALTH_POLL_TIMEOUT = 600  # seconds
ENDPOINT_LOG_POLL_INTERVAL = 30  # seconds
ENDPOINT_LOG_COMMAND_TIMEOUT = 45  # seconds
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"
ENDPOINT_TOKEN_LENGTH = 32
CFD_E2E_WORKFLOW_ID = "physicsnemo-cfd-surface-benchmark"
CFD_E2E_EXECUTOR_CLASS = "physicsnemo-cfd-gpu"
MULTIGPU_ENV_NAMES = (
    "QA_MULTIGPU_GPU_COUNT",
    "MULTIGPU_GPU_COUNT",
    "QA_MULTIGPU_WORKFLOWS",
    "MULTIGPU_WORKFLOWS",
)
PUBLICATION_CONTAINER_CONFIG_PATH = "/outputs/qa-publication"
PUBLICATION_ENV_CONFIG_SENTINEL = "__physicsnemo_serve_publication_env_config__"
DEFAULT_CONTAINER_RUNTIME_CONFIG = "/app/scripts/worker_runtime_config.json"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT.parent / "scripts" / "worker_runtime_config.json"
DEFAULT_PUBLICATION_COMPARE_IMAGE_NAME = (
    "your-registry.example.com/your-org/physicsnemo-serve"
)
DEFAULT_PUBLICATION_COMPARE_NODE_GROUP = "your-node-group"
DEFAULT_PUBLICATION_COMPARE_PULL_SECRET = "your-pull-secret"
PUBLICATION_CREDENTIAL_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_STORAGE_ACCESS_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
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


def determine_e2s_workflows(
    *,
    service: str,
    workflows_arg: str | None,
    suite: str = "",
) -> list[str]:
    """Determine which Earth2Studio workflow plugin IDs to iterate over.

    Priority:
    1. PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID env var (single workflow, legacy mode)
    2. --workflows CLI arg (comma-separated list)
    3. For Rust publication QA: plugins selected by the publication requests
    4. For Rust CFD-only QA (cfd_e2e): no E2S workflows
    5. For the Rust service: full list of all known E2S plugin IDs (one deploy each)
       For the Python service: single deployment running all workflows at once
    """
    env_plugin_id = os.environ.get("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "").strip()
    if env_plugin_id:
        return [env_plugin_id]

    if workflows_arg:
        return [w.strip() for w in workflows_arg.split(",") if w.strip()]

    if service == "rust" and suite == "publication":
        from output_publication_helpers import load_request_payloads
        from service_adapter import RustAdapter

        adapter = RustAdapter()
        return list(
            dict.fromkeys(
                adapter._resolve_workflow(workflow)
                for workflow, _payload in load_request_payloads()
            )
        )

    if service == "rust" and suite == "cfd_e2e":
        return []

    if service == "python":
        return [WORKFLOW_ALL]

    return list(ALL_WORKFLOW_PLUGIN_IDS)


def determine_cfd_workflows(*, service: str, suite: str) -> list[str]:
    """Return CFD workflow IDs to test.

    Returns the PhysicsNeMo-CFD surface benchmark plugin for dedicated CFD runs
    (--suite cfd_e2e) and as an appended step for full QA (--suite full).
    Returns an empty list for all other suites.
    """
    if service == "rust" and suite in ("cfd_e2e", "full"):
        return [CFD_E2E_WORKFLOW_ID]
    return []


def determine_workflows(
    *,
    service: str,
    workflows_arg: str | None,
    suite: str = "",
) -> list[str]:
    """Return all workflow IDs: E2S workflows followed by CFD workflows."""
    return determine_e2s_workflows(
        service=service, workflows_arg=workflows_arg, suite=suite
    ) + determine_cfd_workflows(service=service, suite=suite)


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


def _env_positive_int(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError:
            sys.exit(f"Error: environment variable {name} must be an integer.")
        if parsed <= 0:
            sys.exit(f"Error: environment variable {name} must be greater than zero.")
        return parsed
    return None


def cfd_e2e_container_envs() -> list[str]:
    """Return the non-secret, fail-closed deployment policy for the live CFD E2E."""
    download_timeout = _env_positive_int("QA_CFD_E2E_DOWNLOAD_TIMEOUT_SECS") or 1800
    return [
        f"PHYSICSNEMO_SERVE_EXECUTOR_CLASSES={CFD_E2E_EXECUTOR_CLASS}",
        "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS=huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS=us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_EXT_CACHE=/outputs/.cache/physicsnemo-cfd",
        f"E2S_DOWNLOAD_TIMEOUT_SECS={download_timeout}",
    ]


def multigpu_env_requested() -> bool:
    """Return whether the caller explicitly selected multi-GPU QA behavior."""
    return any(os.environ.get(name, "").strip() for name in MULTIGPU_ENV_NAMES)


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def build_publication_storage_config() -> dict[str, object]:
    """Build the simplified output_publication.storage block from QA env vars."""
    storage_type = (_env_value("QA_PUBLICATION_STORAGE_TYPE") or "").lower()
    prefix = _env_value("QA_PUBLICATION_PREFIX") or "outputs"
    if storage_type == "s3":
        bucket = _env_value("QA_PUBLICATION_S3_BUCKET")
        if not bucket:
            sys.exit(
                "Error: QA_PUBLICATION_S3_BUCKET is required for publication S3 QA."
            )
        storage: dict[str, object] = {
            "type": "s3",
            "bucket": bucket,
            "prefix": prefix,
        }
        region = _env_value(
            "QA_PUBLICATION_S3_REGION", "AWS_DEFAULT_REGION", "AWS_REGION"
        )
        if region:
            storage["region"] = region
        endpoint = _env_value("QA_PUBLICATION_S3_ENDPOINT", "S3_ENDPOINT_URL")
        if endpoint:
            storage["endpoint"] = endpoint
        return storage

    if storage_type == "azure":
        endpoint = _env_value("QA_PUBLICATION_AZURE_ENDPOINT")
        container = _env_value("QA_PUBLICATION_AZURE_CONTAINER")
        if not endpoint:
            sys.exit(
                "Error: QA_PUBLICATION_AZURE_ENDPOINT is required for publication Azure QA."
            )
        if not container:
            sys.exit(
                "Error: QA_PUBLICATION_AZURE_CONTAINER is required for publication Azure QA."
            )
        return {
            "type": "azure",
            "endpoint": endpoint.rstrip("/"),
            "container": container.strip("/"),
            "prefix": prefix,
        }

    sys.exit(
        "Error: QA_PUBLICATION_STORAGE_TYPE must be set to 's3' or 'azure' "
        "for publication QA."
    )


def build_publication_publish_role_config() -> dict[str, object] | None:
    """Build optional publish-role performance config from QA env vars."""
    config: dict[str, object] = {}
    scalar_fields = {
        "max_concurrent_files": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES"
        ),
        "multipart_threshold_bytes": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_MULTIPART_THRESHOLD_BYTES"
        ),
        "multipart_part_size_bytes": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_MULTIPART_PART_SIZE_BYTES"
        ),
        "multipart_max_concurrency": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_MULTIPART_MAX_CONCURRENCY"
        ),
    }
    for key, value in scalar_fields.items():
        if value is not None:
            config[key] = value

    client_options = {
        "timeout_secs": _env_positive_int("QA_PUBLICATION_UPLOAD_TIMEOUT_SECS"),
        "connect_timeout_secs": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_CONNECT_TIMEOUT_SECS"
        ),
        "pool_max_idle_per_host": _env_positive_int(
            "QA_PUBLICATION_UPLOAD_POOL_MAX_IDLE_PER_HOST"
        ),
    }
    client_options = {
        key: value for key, value in client_options.items() if value is not None
    }
    if client_options:
        config["client_options"] = client_options

    retry = {
        "max_retries": _env_positive_int("QA_PUBLICATION_UPLOAD_RETRY_MAX_RETRIES"),
        "timeout_secs": _env_positive_int("QA_PUBLICATION_UPLOAD_RETRY_TIMEOUT_SECS"),
    }
    retry = {key: value for key, value in retry.items() if value is not None}
    if retry:
        config["retry"] = retry

    return config or None


def write_publication_runtime_config(
    *, nfs_path: str, endpoint_name: str
) -> tuple[Path, str]:
    """Write config through an explicit local mapping of the deployed NFS path."""
    local_mount_path = _env_value("QA_PUBLICATION_LOCAL_MOUNT_PATH")
    if not local_mount_path:
        return Path(PUBLICATION_ENV_CONFIG_SENTINEL), PUBLICATION_ENV_CONFIG_SENTINEL

    host_dir = Path(local_mount_path) / "qa-publication" / endpoint_name
    try:
        host_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return Path(PUBLICATION_ENV_CONFIG_SENTINEL), PUBLICATION_ENV_CONFIG_SENTINEL
    host_path = host_dir / "worker_runtime_config.json"
    container_path = f"{PUBLICATION_CONTAINER_CONFIG_PATH}/{endpoint_name}/worker_runtime_config.json"
    config = json.loads(DEFAULT_RUNTIME_CONFIG.read_text(encoding="utf-8"))
    config["output_publication"] = {
        "enabled": True,
        "storage": build_publication_storage_config(),
    }
    publish_config = build_publication_publish_role_config()
    if publish_config:
        roles = config.setdefault("roles", {})
        publish_role = roles.setdefault("publish", {"inputs": [], "outputs": []})
        role_config = publish_role.setdefault("config", {})
        if not isinstance(role_config, dict):
            sys.exit(
                "Error: roles.publish.config in default runtime config must be an object."
            )
        role_config.update(publish_config)
    host_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return host_path, container_path


def publication_container_envs(config_path: str) -> list[str]:
    """Return endpoint env vars needed for publication without logging secrets."""
    if config_path == PUBLICATION_ENV_CONFIG_SENTINEL:
        envs = [
            f"PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG={DEFAULT_CONTAINER_RUNTIME_CONFIG}",
            f"WORKER_RUNTIME_CONFIG={DEFAULT_CONTAINER_RUNTIME_CONFIG}",
            "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON="
            + json.dumps(
                {
                    "enabled": True,
                    "storage": build_publication_storage_config(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        ]
        publish_config = build_publication_publish_role_config()
        if publish_config:
            envs.append(
                "PHYSICSNEMO_SERVE_PUBLISH_ROLE_CONFIG_JSON="
                + json.dumps(publish_config, separators=(",", ":"), sort_keys=True)
            )
        for name in PUBLICATION_CREDENTIAL_ENV_NAMES:
            value = os.environ.get(name, "").strip()
            if value:
                envs.append(f"{name}={value}")
        return envs

    envs = [
        f"PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG={config_path}",
        f"WORKER_RUNTIME_CONFIG={config_path}",
    ]
    for name in PUBLICATION_CREDENTIAL_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            envs.append(f"{name}={value}")
    return envs


def publication_compare_env(
    *,
    image_tag: str,
    workspace_id: str,
    workspace_token: str,
    nfs_path: str,
    deploy_config: dict[str, str],
) -> dict[str, str]:
    """Build compare-job env using QA, Lepton, deploy config, then defaults."""
    compare_image = _env_value("QA_PUBLICATION_COMPARE_IMAGE")
    if not compare_image:
        if ":" in image_tag.rsplit("/", 1)[-1]:
            compare_image = image_tag
        else:
            registry = (
                deploy_config.get("docker_registry", "").strip()
                or DEFAULT_PUBLICATION_COMPARE_IMAGE_NAME.rsplit("/", 1)[0]
            ).rstrip("/")
            image_name = (
                deploy_config.get("image_name", "").strip()
                or DEFAULT_PUBLICATION_COMPARE_IMAGE_NAME.rsplit("/", 1)[1]
            ).lstrip("/")
            compare_image = f"{registry}/{image_name}:{image_tag}"

    return {
        "LEPTON_WORKSPACE_ID": workspace_id,
        "LEPTON_WORKSPACE_TOKEN": workspace_token,
        "QA_PUBLICATION_NFS_PATH": nfs_path,
        "QA_PUBLICATION_MOUNT_TARGET": _env_value("QA_PUBLICATION_MOUNT_TARGET")
        or "/outputs",
        "QA_PUBLICATION_COMPARE_IMAGE": compare_image,
        "QA_PUBLICATION_NODE_GROUP": _env_value(
            "QA_PUBLICATION_NODE_GROUP", "LEPTON_NODE_GROUP"
        )
        or deploy_config.get("lepton_node_group", "").strip()
        or DEFAULT_PUBLICATION_COMPARE_NODE_GROUP,
        "QA_PUBLICATION_RESOURCE_SHAPE": _env_value(
            "QA_PUBLICATION_RESOURCE_SHAPE",
            "QA_PUBLICATION_COMPARE_RESOURCE_SHAPE",
        )
        or "cpu.large",
        "QA_PUBLICATION_PULL_SECRET": _env_value(
            "QA_PUBLICATION_PULL_SECRET", "LEPTON_PULL_SECRET"
        )
        or deploy_config.get("pull_secret", "").strip()
        or DEFAULT_PUBLICATION_COMPARE_PULL_SECRET,
        "QA_PUBLICATION_LUSTRE_STORAGE": _env_value(
            "QA_PUBLICATION_LUSTRE_STORAGE", "LEPTON_LUSTRE_STORAGE"
        )
        or deploy_config.get("lustre_storage", "").strip()
        or "lustre",
    }


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


def strip_ansi_codes(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def capture_endpoint_logs_text(
    endpoint_name: str,
    *,
    timeout: int = ENDPOINT_LOG_COMMAND_TIMEOUT,
) -> str:
    try:
        result = subprocess.run(
            ["lep", "endpoint", "log", "-n", endpoint_name],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout or ""
    except subprocess.TimeoutExpired as err:
        return command_output_text(err.output)


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
    extra_env: dict[str, str] | None = None,
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
        **(extra_env or {}),
        "LEPTON_ENDPOINT_TOKEN": endpoint_token,
        "PYTHONUNBUFFERED": "1",
    }
    if extra_env:
        env.update(extra_env)
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
    container_envs: list[str] | None = None,
    endpoint_name_override: str | None = None,
    deploy_config: dict[str, str] | None = None,
) -> int:
    """Deploy, test, and teardown a single workflow. Returns pytest exit code."""
    display_name = workflow_id if workflow_id != WORKFLOW_ALL else "ALL (single deploy)"
    print(
        f"\n{'=' * 76}\n  WORKFLOW: {display_name}\n{'=' * 76}",
        flush=True,
    )

    is_single_deploy = workflow_id == WORKFLOW_ALL
    deploy_container_envs = list(container_envs or [])
    if not is_single_deploy:
        deploy_container_envs.append(
            f"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={workflow_id}"
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
                container_envs=deploy_container_envs,
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
        pytest_extra_env = {}
        if suite == "publication":
            if not _env_value("QA_PUBLICATION_WORKFLOWS", "QA_PUBLICATION_WORKFLOW"):
                from service_adapter import RustAdapter

                pytest_extra_env["QA_PUBLICATION_WORKFLOW"] = (
                    RustAdapter()._reverse_workflow(workflow_id)
                )
            pytest_extra_env.update(
                publication_compare_env(
                    image_tag=image_tag,
                    workspace_id=workspace_id,
                    workspace_token=workspace_token,
                    nfs_path=nfs_path,
                    deploy_config=deploy_config or {},
                )
            )
        elif suite == "cfd_e2e":
            pytest_extra_env = {
                "QA_CFD_E2E_ENABLED": "1",
                "QA_CFD_E2E_ARTIFACT_DIR": str(artifact_dir / "cfd-e2e"),
                "QA_CFD_E2E_TIMEOUT_SECS": os.environ.get(
                    "QA_CFD_E2E_TIMEOUT_SECS", "23400"
                ),
                "QA_CFD_E2E_POLL_SECS": os.environ.get("QA_CFD_E2E_POLL_SECS", "20"),
                "QA_CFD_E2E_SUBMIT_TIMEOUT_SECS": os.environ.get(
                    "QA_CFD_E2E_SUBMIT_TIMEOUT_SECS", "300"
                ),
            }
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
                extra_env=pytest_extra_env,
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


def _pytest_counts(artifact_dir: Path, workflow_id: str) -> str:
    """Return 'N passed / M total' by parsing the newest JUnit XML for the workflow."""
    results_dir = artifact_dir / "pytest-results"
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", workflow_id)
    xmls = sorted(results_dir.glob(f"results_{slug}_*.xml"), key=lambda p: p.stat().st_mtime)
    if not xmls:
        return ""
    try:
        root = ET.parse(xmls[-1]).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            return ""
        total = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        passed = total - failures - errors - skipped
        ran = total - skipped
        return f"{passed} passed / {ran} ran"
    except Exception:
        return ""


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
        choices=[
            "smoke",
            "cicd",
            "basic",
            "negative",
            "stress",
            "publication",
            "cfd_e2e",
            "full",
        ],
        default="smoke",
        help=(
            "Test suite to run (pytest marker). 'publication' runs live "
            "object-store sync tests; 'cfd_e2e' runs the opt-in live "
            "PhysicsNeMo-CFD test; 'full' runs without a marker filter."
        ),
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
    if args.suite == "cfd_e2e":
        if args.service != "rust":
            sys.exit("Error: --suite cfd_e2e only supports --service rust.")
        if args.num_proc not in {0, 1}:
            sys.exit("Error: --suite cfd_e2e requires --num-proc 0 or 1.")

    _deploy_cfg = load_deploy_config()
    workspace_id = os.environ.get("LEPTON_WORKSPACE_ID", "").strip() or _deploy_cfg.get(
        "lepton_workspace_id", ""
    )
    if not workspace_id:
        sys.exit(
            "Error: LEPTON_WORKSPACE_ID is required but not set.\n"
            "Set it via environment variable or lepton_workspace_id in deploy/config.yaml."
        )
    workspace_token = require_env("LEPTON_WORKSPACE_TOKEN")
    endpoint_token = get_or_generate_endpoint_token()

    source = "physicsnemo-serve" if args.service == "rust" else "earth2studio"
    nfs_path = f"{_deploy_cfg.get('nfs_mount_base', '/mnt/shared')}/{args.lustre_dir}"

    container_envs = []
    enabled_plugin_id = os.environ.get(
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", ""
    ).strip()
    endpoint_name = args.endpoint_name or default_endpoint_name(source)
    if args.suite == "publication":
        if args.service != "rust":
            sys.exit("Error: --suite publication only supports --service rust.")
        host_config_path, container_config_path = write_publication_runtime_config(
            nfs_path=nfs_path,
            endpoint_name=endpoint_name,
        )
        container_envs.extend(publication_container_envs(container_config_path))
        print(
            f"==> Publication runtime config: {host_config_path} -> {container_config_path}",
            flush=True,
        )
    if enabled_plugin_id and args.service == "rust":
        print(
            f"==> Single-plugin mode: PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={enabled_plugin_id}",
            flush=True,
        )
    elif enabled_plugin_id:
        print(
            "==> Ignoring PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID because --service is not rust",
            flush=True,
        )
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = Path.cwd() / artifact_dir

    e2s_workflows = determine_e2s_workflows(
        service=args.service,
        workflows_arg=args.workflows,
        suite=args.suite,
    )
    cfd_workflows = determine_cfd_workflows(service=args.service, suite=args.suite)
    all_workflows = e2s_workflows + cfd_workflows

    if args.suite == "cfd_e2e" and e2s_workflows:
        sys.exit(
            "Error: --suite cfd_e2e must run only "
            f"{CFD_E2E_WORKFLOW_ID!r}; remove conflicting workflow selection."
        )

    endpoint_name_override = None
    if len(all_workflows) == 1 and args.endpoint_name:
        endpoint_name_override = args.endpoint_name
    elif len(all_workflows) > 1 and args.endpoint_name:
        print(
            "Warning: --endpoint-name is ignored in multi-workflow mode "
            "(each workflow gets a unique endpoint name).",
            flush=True,
        )

    if len(all_workflows) > 1 and args.skip_teardown:
        print(
            "Warning: --skip-teardown only applies to single-workflow runs. "
            "Ignoring in multi-workflow mode to avoid leaking GPU resources.",
            flush=True,
        )

    skip_teardown = args.skip_teardown and len(all_workflows) == 1

    print(
        f"\n==> QA run: {len(all_workflows)} workflow(s) to test sequentially",
        flush=True,
    )
    for i, wf in enumerate(all_workflows, 1):
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

    # Run E2S workflows with standard configuration.
    for workflow_id in e2s_workflows:
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
            container_envs=container_envs,
            deploy_config=_deploy_cfg,
        )
        results.append((workflow_id, exit_code))

    # Run CFD workflows with their dedicated configuration:
    # - CFD-specific container envs (executor class, prefetch hosts, ext cache)
    # - suite="cfd_e2e" so run_one_workflow applies QA_CFD_E2E_ENABLED=1
    # - public_run_1_full_matrix_request.json (all 5 models) unless overridden
    if cfd_workflows:
        _cfd_default_request = (
            SCRIPTS_DIR.parent.parent
            / "plugins"
            / "physicsnemo-cfd-surface-benchmark"
            / "examples"
            / "public_run_1_full_matrix_request.json"
        )
        _cfd_request_was_set = bool(os.environ.get("QA_CFD_E2E_REQUEST_PATH", "").strip())
        if not _cfd_request_was_set:
            os.environ["QA_CFD_E2E_REQUEST_PATH"] = str(_cfd_default_request)
        try:
            for workflow_id in cfd_workflows:
                exit_code = run_one_workflow(
                    workflow_id=workflow_id,
                    source=source,
                    service=args.service,
                    image_tag=args.image_tag,
                    workspace_id=workspace_id,
                    workspace_token=workspace_token,
                    endpoint_token=endpoint_token,
                    nfs_path=nfs_path,
                    suite="cfd_e2e",
                    test_filter=args.test_filter,
                    num_proc=min(args.num_proc, 1),
                    skip_teardown=False,
                    stream_logs=args.stream_endpoint_logs,
                    log_interval=args.endpoint_log_interval,
                    artifact_dir=artifact_dir,
                    post_health_wait_secs=args.post_health_wait_secs,
                    container_envs=list(container_envs) + cfd_e2e_container_envs(),
                    deploy_config=_deploy_cfg,
                )
                results.append((workflow_id, exit_code))
        finally:
            if not _cfd_request_was_set:
                os.environ.pop("QA_CFD_E2E_REQUEST_PATH", None)

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
        test_counts = _pytest_counts(artifact_dir, workflow_id)
        counts_str = f" ({test_counts})" if test_counts else ""
        print(f"  [{status}] {workflow_id}{counts_str}", flush=True)

    total = len(results)
    passed = sum(1 for _, code in results if code == 0)
    failed = total - passed
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}", flush=True)
    print(f"{'=' * 76}\n", flush=True)

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
