#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deploy two Lepton endpoints and compare their forecast Zarr outputs with CRPS."""

from __future__ import annotations

import argparse
import atexit
import datetime
import os
import re
import secrets
import signal
import string
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
QA_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = QA_ROOT.parent
INFERENCE_DIR = QA_ROOT / "inference"
DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy-to-lepton.sh"
TEARDOWN_SCRIPT = SCRIPTS_DIR / "teardown-lepton.sh"
CRPS_JOB_SCRIPT = SCRIPTS_DIR / "submit-lepton-crps-job.sh"


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
from config import load_deploy_config  # noqa: E402


sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from compare_helpers import (  # noqa: E402
    WorkflowRun,
    fetch_results_with_zarr_path,
    load_request_payload,
    make_client,
    poll_workflow,
    submit_workflow,
    write_json,
)
from run_qa import (  # noqa: E402
    ENDPOINT_LOG_POLL_INTERVAL,
    env_bool,
    get_or_generate_endpoint_token,
    health_check,
    parse_endpoint_name,
    parse_endpoint_url,
    run_streaming,
    stream_endpoint_logs,
)
from service_adapter import get_adapter  # noqa: E402


_DEPLOY_CFG = load_deploy_config()
_REGISTRY = _DEPLOY_CFG.get("docker_registry", "")
DEFAULT_BASELINE_IMAGE = f"{_REGISTRY}/{_DEPLOY_CFG.get('python_service_image', 'earth2studio')}:v0.15.0.20260515.0"
DEFAULT_CANDIDATE_IMAGE = (
    f"{_REGISTRY}/{_DEPLOY_CFG.get('image_name', 'scicomp-ferroflux')}:v0.1.0"
)
DEFAULT_BASELINE_REQUEST_JSON = INFERENCE_DIR / "requests/crps_baseline_request.json"
DEFAULT_CANDIDATE_REQUEST_JSON = (
    INFERENCE_DIR / "requests/crps_candidate_fanout_request.json"
)
DEFAULT_ARTIFACT_DIR = QA_ROOT / "artifacts"
DEFAULT_COMPARISON_SCRIPT = (
    "/workspace/earth2studio-project/serve/server/scripts/compare_crps.py"
)
DEFAULT_RUN_TIMEOUT_SECONDS = 3600
DEFAULT_RUN_POLL_INTERVAL_SECONDS = 30
DEFAULT_CANDIDATE_RESOURCE_SHAPE = "gpu.8xh100-sxm"
DEFAULT_CANDIDATE_MATERIALIZATION_MODES = ("scheduled_gpu", "prepare_cpu")


def default_lustre_dir() -> str:
    """Return the default shared Lustre subdirectory for CRPS QA runs."""

    return f"crps_tests_{datetime.datetime.now(datetime.UTC):%Y%m%d}"


@dataclass(frozen=True)
class EndpointConfig:
    label: str
    source: str
    service: str
    image: str
    workflow: str
    endpoint_name: str


@dataclass(frozen=True)
class ImageSpec:
    image_name: str | None
    image_tag: str


@dataclass(frozen=True)
class CandidateModeRequest:
    mode: str
    label: str
    payload: dict[str, Any]


def split_image_reference(image: str) -> ImageSpec:
    """Split either a tag or a full image reference into deploy script parts."""

    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return ImageSpec(
            image_name=image[:last_colon], image_tag=image[last_colon + 1 :]
        )
    return ImageSpec(image_name=None, image_tag=image)


def generate_run_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _sanitize_endpoint_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def candidate_mode_label(mode: str) -> str:
    return f"candidate-{mode.replace('_', '-')}"


def parse_candidate_materialization_modes(value: str) -> list[str]:
    modes = [mode.strip().lower() for mode in value.split(",") if mode.strip()]
    if not modes:
        raise ValueError(
            "--candidate-materialization-modes must include at least one mode"
        )

    supported = set(DEFAULT_CANDIDATE_MATERIALIZATION_MODES)
    unsupported = sorted(set(modes) - supported)
    if unsupported:
        supported_values = ", ".join(DEFAULT_CANDIDATE_MATERIALIZATION_MODES)
        raise ValueError(
            "--candidate-materialization-modes contains unsupported values "
            f"{unsupported}; supported values: {supported_values}"
        )

    unique_modes: list[str] = []
    seen: set[str] = set()
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        unique_modes.append(mode)
    return unique_modes


def candidate_mode_requests(
    base_payload: dict[str, Any],
    modes: list[str],
) -> list[CandidateModeRequest]:
    return [
        CandidateModeRequest(
            mode=mode,
            label=candidate_mode_label(mode),
            payload={**base_payload, "perturbation_materialization_mode": mode},
        )
        for mode in modes
    ]


def default_compare_endpoint_name(label: str, source: str, run_id: str) -> str:
    """Return a meaningful Lepton endpoint name within the 36-char limit."""

    if label == "base":
        prefix = "crps-e2s-python-base"
    elif source == "physicsnemo-serve":
        prefix = "crps-e2s-rust-ff"
    else:
        prefix = f"crps-e2s-rust-{source}"
    meaningful = _sanitize_endpoint_name(f"{prefix}-{run_id}")
    return meaningful[:36].rstrip("-")


def resolve_path(path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path.cwd() / value


def deploy_endpoint(
    *,
    config: EndpointConfig,
    workspace_id: str,
    workspace_token: str,
    workspace_url: str | None,
    endpoint_token: str,
    nfs_path: str,
    mount_target: str,
    lustre_storage: str,
    node_group: str | None,
    resource_shape: str | None,
    pull_secret: str | None,
    container_env: dict[str, str],
    dry_run: bool,
    artifact_dir: Path,
) -> tuple[str, str]:
    """Run deploy-to-lepton.sh and return endpoint URL and name."""

    image = split_image_reference(config.image)
    cmd = [
        str(DEPLOY_SCRIPT),
        "--source",
        config.source,
        "--skip-build",
        "--skip-push",
        "--image-tag",
        image.image_tag,
        "--workspace-id",
        workspace_id,
        "--nfs-path",
        nfs_path,
        "--mount-target",
        mount_target,
        "--lustre-storage",
        lustre_storage,
        "--endpoint-name",
        config.endpoint_name,
    ]
    if image.image_name:
        cmd += ["--image-name", image.image_name]
    if workspace_url:
        cmd += ["--workspace-url", workspace_url]
    if node_group:
        cmd += ["--node-group", node_group]
    if resource_shape:
        cmd += ["--resource-shape", resource_shape]
    if pull_secret:
        cmd += ["--pull-secret", pull_secret]
    for name, value in container_env.items():
        cmd += ["--env", f"{name}={value}"]
    if dry_run:
        cmd.append("--dry-run")

    print(f"==> Deploying {config.label} endpoint: {config.endpoint_name}")
    print(f"==> {config.label} image: {config.image}")
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_TOKEN": workspace_token,
        "LEPTON_ENDPOINT_TOKEN": endpoint_token,
    }
    returncode, output = run_streaming(cmd, env=env)
    write_text_artifact(artifact_dir / f"{config.label}-deploy.log", output)
    if returncode != 0:
        raise RuntimeError(
            f"{config.label} deploy-to-lepton.sh exited with code {returncode}"
        )

    url = parse_endpoint_url(output)
    if not url and dry_run:
        url = "(dry-run; not deployed)"
    if not url:
        raise RuntimeError(f"could not parse {config.label} endpoint URL")

    name = parse_endpoint_name(output) or config.endpoint_name
    return url, name


def write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def teardown_endpoint(
    *,
    workspace_id: str,
    workspace_token: str,
    workspace_url: str | None,
    endpoint_name: str,
) -> None:
    print(f"\n==> Tearing down deployment: {endpoint_name}")
    cmd = [
        str(TEARDOWN_SCRIPT),
        "--workspace-id",
        workspace_id,
        "--endpoint-name",
        endpoint_name,
    ]
    if workspace_url:
        cmd += ["--workspace-url", workspace_url]
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_TOKEN": workspace_token,
    }
    returncode, _ = run_streaming(cmd, env=env)
    if returncode != 0:
        print(f"Warning: teardown exited with code {returncode}", file=sys.stderr)


def submit_crps_job(
    *,
    run_id: str,
    comparison_label: str,
    comparison_image: str,
    baseline_zarr: str,
    candidate_zarr: str,
    threshold: str,
    lead_time_chunk_size: str,
    device: str,
    variables: str | None,
    workspace_id: str,
    workspace_token: str,
    workspace_url: str | None,
    nfs_path: str,
    mount_target: str,
    lustre_storage: str,
    node_group: str | None,
    resource_shape: str | None,
    pull_secret: str | None,
    artifact_dir: Path,
    comparison_script: str,
    keep_batch_job: bool,
    dry_run: bool,
) -> tuple[int, str]:
    job_name = f"ff-crps-{run_id}-{_sanitize_endpoint_name(comparison_label)}"
    job_log = artifact_dir / "crps-jobs" / f"{job_name}.log"
    cmd = [
        str(CRPS_JOB_SCRIPT),
        "--job-name",
        job_name,
        "--image-tag",
        comparison_image,
        "--forecast-a",
        baseline_zarr,
        "--forecast-b",
        candidate_zarr,
        "--threshold",
        threshold,
        "--lead-time-chunk-size",
        lead_time_chunk_size,
        "--device",
        device,
        "--workspace-id",
        workspace_id,
        "--nfs-path",
        nfs_path,
        "--mount-target",
        mount_target,
        "--lustre-storage",
        lustre_storage,
        "--artifact-log",
        str(job_log),
        "--comparison-script",
        comparison_script,
    ]
    if workspace_url:
        cmd += ["--workspace-url", workspace_url]
    if node_group:
        cmd += ["--node-group", node_group]
    if resource_shape:
        cmd += ["--resource-shape", resource_shape]
    if pull_secret:
        cmd += ["--pull-secret", pull_secret]
    if variables:
        cmd += ["--variables", variables]
    if keep_batch_job:
        cmd += ["--keep-job"]
    if dry_run:
        cmd += ["--dry-run"]

    env = {**os.environ, "LEPTON_WORKSPACE_TOKEN": workspace_token}
    returncode, output = run_streaming(cmd, env=env)
    return returncode, job_name


def parse_crps_report(report_log: Path) -> dict[str, Any]:
    """Extract the final compare_crps.py report from a CRPS job log artifact."""

    if not report_log.exists():
        return {
            "available": False,
            "log_path": str(report_log),
            "reason": "CRPS job log artifact does not exist",
        }

    text = report_log.read_text(encoding="utf-8", errors="replace")
    marker = "CRPS COMPARISON REPORT"
    marker_index = text.rfind(marker)
    if marker_index < 0:
        return {
            "available": False,
            "log_path": str(report_log),
            "reason": "CRPS comparison report was not found in the job log",
        }

    report_text = text[marker_index:]
    max_diff = _extract_report_percent(report_text, "Max relative CRPS difference")
    threshold = _extract_report_percent(report_text, "Threshold")
    result_match = re.search(r"^Result:\s*(\w+)", report_text, flags=re.MULTILINE)

    return {
        "available": True,
        "log_path": str(report_log),
        "result": result_match.group(1) if result_match else None,
        "max_relative_crps_difference": _format_percent(max_diff),
        "max_relative_diff_percent": max_diff,
        "threshold_percent": threshold,
        "report_text": report_text.strip(),
    }


def _extract_report_percent(report_text: str, label: str) -> float | None:
    match = re.search(
        rf"^{re.escape(label)}:\s*([0-9.]+)%",
        report_text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def _format_percent(value: float | None) -> str | None:
    return f"{value:.4f}%" if value is not None else None


def start_endpoint_log_thread(
    *,
    endpoint_name: str,
    artifact_dir: Path,
    interval: int,
    stop_event: threading.Event,
) -> threading.Thread:
    artifact_path = artifact_dir / "endpoint-logs" / f"{endpoint_name}.log"
    thread = threading.Thread(
        target=stream_endpoint_logs,
        args=(endpoint_name, stop_event, interval, artifact_path),
        daemon=True,
    )
    thread.start()
    return thread


def submit_endpoint_workflow(
    *,
    label: str,
    client: Any,
    base_url: str,
    adapter: Any,
    workflow: str,
    request_payload: dict[str, Any],
    artifact_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Submit one endpoint workflow and artifact the submit response."""

    print(f"==> Submitting {label} workflow: {workflow}", flush=True)
    execution_id, submit_payload = submit_workflow(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        request_payload=request_payload,
    )
    write_json(artifact_dir / f"{label}-submit.json", submit_payload)
    return execution_id, submit_payload


def finish_endpoint_workflow(
    *,
    label: str,
    client: Any,
    base_url: str,
    adapter: Any,
    workflow: str,
    execution_id: str,
    submit_payload: dict[str, Any],
    artifact_dir: Path,
    mount_target: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> WorkflowRun:
    """Wait for one submitted endpoint workflow and fetch its Zarr result."""

    print(f"==> Waiting for {label} execution: {execution_id}", flush=True)
    final_status_payload = poll_workflow(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        execution_id=execution_id,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )
    write_json(artifact_dir / f"{label}-status-final.json", final_status_payload)

    print(f"==> Fetching {label} results", flush=True)
    results_payload, forecast_zarr_path = fetch_results_with_zarr_path(
        client=client,
        base_url=base_url,
        adapter=adapter,
        workflow=workflow,
        execution_id=execution_id,
        artifact_dir=artifact_dir,
        label=label,
        mount_target=mount_target,
    )

    return WorkflowRun(
        label=label,
        workflow=workflow,
        execution_id=execution_id,
        submit_payload=submit_payload,
        final_status_payload=final_status_payload,
        results_payload=results_payload,
        forecast_zarr_path=forecast_zarr_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy baseline/candidate Lepton endpoints and run CRPS compare.",
    )
    parser.add_argument(
        "--baseline-service",
        choices=["python", "rust"],
        default="python",
        help="Baseline inference service API type.",
    )
    parser.add_argument(
        "--candidate-service",
        choices=["python", "rust"],
        default="rust",
        help="Candidate inference service API type.",
    )
    parser.add_argument(
        "--baseline-source",
        default="earth2studio",
        choices=["earth2studio", "physicsnemo-serve", "custom"],
        help="deploy-to-lepton.sh source preset for the baseline endpoint.",
    )
    parser.add_argument(
        "--candidate-source",
        default="physicsnemo-serve",
        choices=["earth2studio", "physicsnemo-serve", "custom"],
        help="deploy-to-lepton.sh source preset for the candidate endpoint.",
    )
    parser.add_argument("--baseline-image-tag", default=DEFAULT_BASELINE_IMAGE)
    parser.add_argument("--candidate-image-tag", default=DEFAULT_CANDIDATE_IMAGE)
    parser.add_argument(
        "--baseline-workflow",
        default="ensemble_workflow",
        help="Workflow name to run on the baseline endpoint.",
    )
    parser.add_argument(
        "--candidate-workflow",
        default="earth2-ensemble-fanout",
        help="Workflow name to run on the candidate endpoint.",
    )
    parser.add_argument(
        "--request-json",
        default=None,
        help=(
            "Legacy single request JSON payload submitted to both workflows. "
            "Prefer --baseline-request-json and --candidate-request-json."
        ),
    )
    parser.add_argument(
        "--baseline-request-json",
        default=None,
        help=(
            "Request JSON payload for the baseline workflow "
            f"(default: {DEFAULT_BASELINE_REQUEST_JSON})."
        ),
    )
    parser.add_argument(
        "--candidate-request-json",
        default=None,
        help=(
            "Request JSON payload for the candidate workflow "
            f"(default: {DEFAULT_CANDIDATE_REQUEST_JSON})."
        ),
    )
    parser.add_argument(
        "--candidate-materialization-modes",
        default=os.environ.get(
            "QA_CANDIDATE_MATERIALIZATION_MODES",
            ",".join(DEFAULT_CANDIDATE_MATERIALIZATION_MODES),
        ),
        help=(
            "Comma-separated candidate fanout perturbation materialization modes "
            "to compare against the Python baseline "
            f"(default: {','.join(DEFAULT_CANDIDATE_MATERIALIZATION_MODES)})."
        ),
    )
    parser.add_argument(
        "--comparison-image-tag",
        default=None,
        help="Image tag or full image reference for the CRPS batch job.",
    )
    parser.add_argument(
        "--comparison-script",
        default=DEFAULT_COMPARISON_SCRIPT,
        help="Path to compare_crps.py inside the comparison image.",
    )
    parser.add_argument("--threshold", default="0.01")
    parser.add_argument("--variables", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lead-time-chunk-size", default="1")
    parser.add_argument(
        "--workspace-id",
        default=os.environ.get(
            "LEPTON_WORKSPACE_ID", _DEPLOY_CFG.get("lepton_workspace_id", "")
        ),
    )
    parser.add_argument(
        "--workspace-url", default=os.environ.get("LEPTON_WORKSPACE_URL", "")
    )
    parser.add_argument(
        "--lustre-dir",
        default=os.environ.get("QA_LUSTRE_DIR", default_lustre_dir()),
        help=(
            "Subdirectory under NFS mount base to mount "
            "(default: crps_tests_<YYYYMMDD>)."
        ),
    )
    parser.add_argument(
        "--lustre-storage", default=os.environ.get("LEPTON_LUSTRE_STORAGE", "lustre")
    )
    parser.add_argument("--mount-target", default="/outputs")
    parser.add_argument("--node-group", default=os.environ.get("LEPTON_NODE_GROUP"))
    parser.add_argument(
        "--resource-shape", default=os.environ.get("LEPTON_RESOURCE_SHAPE")
    )
    parser.add_argument(
        "--baseline-resource-shape",
        default=os.environ.get("QA_BASELINE_RESOURCE_SHAPE"),
        help=(
            "Resource shape for the baseline endpoint. Falls back to "
            "--resource-shape or deploy-to-lepton.sh's default."
        ),
    )
    parser.add_argument(
        "--candidate-resource-shape",
        default=os.environ.get(
            "QA_CANDIDATE_RESOURCE_SHAPE", DEFAULT_CANDIDATE_RESOURCE_SHAPE
        ),
        help=(
            "Resource shape for the candidate endpoint "
            f"(default: {DEFAULT_CANDIDATE_RESOURCE_SHAPE})."
        ),
    )
    parser.add_argument("--pull-secret", default=os.environ.get("LEPTON_PULL_SECRET"))
    parser.add_argument("--baseline-endpoint-name", default=None)
    parser.add_argument("--candidate-endpoint-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-teardown", action="store_true")
    parser.add_argument("--keep-batch-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stream-endpoint-logs",
        dest="stream_endpoint_logs",
        action="store_true",
        default=env_bool("STREAM_ENDPOINT_LOGS", True),
    )
    parser.add_argument(
        "--no-endpoint-logs",
        dest="stream_endpoint_logs",
        action="store_false",
    )
    parser.add_argument(
        "--endpoint-log-interval",
        type=int,
        default=int(
            os.environ.get("ENDPOINT_LOG_INTERVAL", ENDPOINT_LOG_POLL_INTERVAL)
        ),
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        default=int(os.environ.get("QA_RUN_TIMEOUT", DEFAULT_RUN_TIMEOUT_SECONDS)),
    )
    parser.add_argument(
        "--run-poll-interval",
        type=int,
        default=int(
            os.environ.get("QA_RUN_POLL_INTERVAL", DEFAULT_RUN_POLL_INTERVAL_SECONDS)
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("QA_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.endpoint_log_interval <= 0:
        raise ValueError("--endpoint-log-interval must be greater than 0")
    if args.run_timeout <= 0:
        raise ValueError("--run-timeout must be greater than 0")
    if args.run_poll_interval <= 0:
        raise ValueError("--run-poll-interval must be greater than 0")
    parse_candidate_materialization_modes(args.candidate_materialization_modes)


def run(args: argparse.Namespace) -> int:
    validate_args(args)

    workspace_token = os.environ.get("LEPTON_WORKSPACE_TOKEN", "").strip()
    endpoint_token = get_or_generate_endpoint_token()
    workspace_url = args.workspace_url.strip() or None
    run_id = args.run_id or generate_run_id()
    nfs_path = f"{_DEPLOY_CFG.get('nfs_mount_base', '/mnt/shared')}/{args.lustre_dir}"
    artifact_root = resolve_path(args.artifact_dir)
    run_artifact_dir = artifact_root / "crps-compare" / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    baseline = EndpointConfig(
        label="baseline",
        source=args.baseline_source,
        service=args.baseline_service,
        image=args.baseline_image_tag,
        workflow=args.baseline_workflow,
        endpoint_name=args.baseline_endpoint_name
        or default_compare_endpoint_name("base", args.baseline_source, run_id),
    )
    candidate = EndpointConfig(
        label="candidate",
        source=args.candidate_source,
        service=args.candidate_service,
        image=args.candidate_image_tag,
        workflow=args.candidate_workflow,
        endpoint_name=args.candidate_endpoint_name
        or default_compare_endpoint_name("candidate", args.candidate_source, run_id),
    )
    comparison_image = args.comparison_image_tag or args.baseline_image_tag
    baseline_request_json_path = resolve_path(
        args.baseline_request_json
        or args.request_json
        or str(DEFAULT_BASELINE_REQUEST_JSON)
    )
    candidate_request_json_path = resolve_path(
        args.candidate_request_json
        or args.request_json
        or str(DEFAULT_CANDIDATE_REQUEST_JSON)
    )
    baseline_request_payload = load_request_payload(baseline_request_json_path)
    candidate_request_payload = load_request_payload(candidate_request_json_path)
    candidate_modes = parse_candidate_materialization_modes(
        args.candidate_materialization_modes
    )
    candidate_requests = candidate_mode_requests(
        candidate_request_payload, candidate_modes
    )
    shared_output_env = {
        "DEFAULT_OUTPUT_DIR": args.mount_target,
        "RESULTS_ZIP_DIR": args.mount_target,
    }

    config_payload: dict[str, Any] = {
        "run_id": run_id,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "comparison_image": comparison_image,
        "comparison_script": args.comparison_script,
        "baseline_request_json": str(baseline_request_json_path),
        "candidate_request_json": str(candidate_request_json_path),
        "candidate_materialization_modes": candidate_modes,
        "mount": {
            "nfs_path": nfs_path,
            "mount_target": args.mount_target,
            "lustre_storage": args.lustre_storage,
        },
        "resource_shapes": {
            "baseline": args.baseline_resource_shape or args.resource_shape,
            "candidate": args.candidate_resource_shape or args.resource_shape,
            "comparison_job": args.resource_shape,
        },
        "container_env": shared_output_env,
        "threshold": args.threshold,
        "variables": args.variables,
        "device": args.device,
        "lead_time_chunk_size": args.lead_time_chunk_size,
        "dry_run": args.dry_run,
    }
    write_json(run_artifact_dir / "config.json", config_payload)
    write_json(run_artifact_dir / "baseline-request.json", baseline_request_payload)
    write_json(run_artifact_dir / "candidate-request.json", candidate_request_payload)
    for candidate_request in candidate_requests:
        write_json(
            run_artifact_dir / f"{candidate_request.label}-request.json",
            candidate_request.payload,
        )

    endpoint_names = [baseline.endpoint_name, candidate.endpoint_name]
    log_stop_event = threading.Event()
    log_threads: list[threading.Thread] = []
    teardown_done = {"done": False}

    def stop_endpoint_logs() -> None:
        log_stop_event.set()
        for thread in log_threads:
            if thread.is_alive():
                thread.join(timeout=5)

    def teardown_once() -> None:
        if teardown_done["done"]:
            return
        teardown_done["done"] = True
        stop_endpoint_logs()
        for endpoint_name in endpoint_names:
            teardown_endpoint(
                workspace_id=args.workspace_id,
                workspace_token=workspace_token,
                workspace_url=workspace_url,
                endpoint_name=endpoint_name,
            )

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n==> Received signal {signum}; cleaning up before exit", flush=True)
        teardown_once()
        raise SystemExit(128 + signum)

    if args.skip_teardown or args.dry_run:
        atexit.register(stop_endpoint_logs)
    else:
        atexit.register(teardown_once)
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    baseline_url, baseline_name = deploy_endpoint(
        config=baseline,
        workspace_id=args.workspace_id,
        workspace_token=workspace_token,
        workspace_url=workspace_url,
        endpoint_token=endpoint_token,
        nfs_path=nfs_path,
        mount_target=args.mount_target,
        lustre_storage=args.lustre_storage,
        node_group=args.node_group,
        resource_shape=args.baseline_resource_shape or args.resource_shape,
        pull_secret=args.pull_secret,
        container_env=shared_output_env,
        dry_run=args.dry_run,
        artifact_dir=run_artifact_dir,
    )
    candidate_url, candidate_name = deploy_endpoint(
        config=candidate,
        workspace_id=args.workspace_id,
        workspace_token=workspace_token,
        workspace_url=workspace_url,
        endpoint_token=endpoint_token,
        nfs_path=nfs_path,
        mount_target=args.mount_target,
        lustre_storage=args.lustre_storage,
        node_group=args.node_group,
        resource_shape=args.candidate_resource_shape or args.resource_shape,
        pull_secret=args.pull_secret,
        container_env=shared_output_env,
        dry_run=args.dry_run,
        artifact_dir=run_artifact_dir,
    )

    if args.stream_endpoint_logs and not args.dry_run:
        for endpoint_name in [baseline_name, candidate_name]:
            log_threads.append(
                start_endpoint_log_thread(
                    endpoint_name=endpoint_name,
                    artifact_dir=artifact_root,
                    interval=args.endpoint_log_interval,
                    stop_event=log_stop_event,
                )
            )

    if args.dry_run:
        baseline_zarr = (
            f"{args.mount_target.rstrip('/')}/{run_id}/baseline/forecast.zarr"
        )
        dry_run_comparisons = {}
        for candidate_request in candidate_requests:
            candidate_zarr = (
                f"{args.mount_target.rstrip('/')}/{run_id}/"
                f"{candidate_request.label}/forecast.zarr"
            )
            job_returncode, job_name = submit_crps_job(
                run_id=run_id,
                comparison_label=candidate_request.mode,
                comparison_image=comparison_image,
                baseline_zarr=baseline_zarr,
                candidate_zarr=candidate_zarr,
                threshold=args.threshold,
                lead_time_chunk_size=args.lead_time_chunk_size,
                device=args.device,
                variables=args.variables,
                workspace_id=args.workspace_id,
                workspace_token=workspace_token,
                workspace_url=workspace_url,
                nfs_path=nfs_path,
                mount_target=args.mount_target,
                lustre_storage=args.lustre_storage,
                node_group=args.node_group,
                resource_shape=args.resource_shape,
                pull_secret=args.pull_secret,
                artifact_dir=artifact_root,
                comparison_script=args.comparison_script,
                keep_batch_job=args.keep_batch_job,
                dry_run=True,
            )
            dry_run_comparisons[candidate_request.mode] = {
                "candidate_zarr": candidate_zarr,
                "comparison_job_name": job_name,
                "comparison_job_exit_code": job_returncode,
            }
        write_json(
            run_artifact_dir / "summary.json",
            {
                **config_payload,
                "baseline_endpoint_url": baseline_url,
                "candidate_endpoint_url": candidate_url,
                "baseline_zarr": baseline_zarr,
                "comparisons": dry_run_comparisons,
                "final_result": "dry-run",
            },
        )
        print(f"==> Dry run complete. Artifacts: {run_artifact_dir}", flush=True)
        return 0

    if not health_check(baseline_url, endpoint_token):
        raise RuntimeError("baseline service never became healthy")
    if not health_check(candidate_url, endpoint_token):
        raise RuntimeError("candidate service never became healthy")

    client = make_client(endpoint_token)
    baseline_adapter = get_adapter(baseline.service)
    candidate_adapter = get_adapter(candidate.service)
    baseline_execution_id, baseline_submit_payload = submit_endpoint_workflow(
        label="baseline",
        client=client,
        base_url=baseline_url,
        adapter=baseline_adapter,
        workflow=baseline.workflow,
        request_payload=baseline_request_payload,
        artifact_dir=run_artifact_dir,
    )
    candidate_submissions = [
        (
            candidate_request,
            *submit_endpoint_workflow(
                label=candidate_request.label,
                client=client,
                base_url=candidate_url,
                adapter=candidate_adapter,
                workflow=candidate.workflow,
                request_payload=candidate_request.payload,
                artifact_dir=run_artifact_dir,
            ),
        )
        for candidate_request in candidate_requests
    ]

    baseline_run = finish_endpoint_workflow(
        label="baseline",
        client=client,
        base_url=baseline_url,
        adapter=baseline_adapter,
        workflow=baseline.workflow,
        execution_id=baseline_execution_id,
        submit_payload=baseline_submit_payload,
        artifact_dir=run_artifact_dir,
        mount_target=args.mount_target,
        timeout_seconds=args.run_timeout,
        interval_seconds=args.run_poll_interval,
    )
    candidate_runs: dict[str, WorkflowRun] = {}
    for candidate_request, execution_id, submit_payload in candidate_submissions:
        candidate_runs[candidate_request.mode] = finish_endpoint_workflow(
            label=candidate_request.label,
            client=client,
            base_url=candidate_url,
            adapter=candidate_adapter,
            workflow=candidate.workflow,
            execution_id=execution_id,
            submit_payload=submit_payload,
            artifact_dir=run_artifact_dir,
            mount_target=args.mount_target,
            timeout_seconds=args.run_timeout,
            interval_seconds=args.run_poll_interval,
        )

    if not args.skip_teardown:
        print(
            "==> Endpoint outputs captured; tearing down inference endpoints "
            "before CRPS comparison",
            flush=True,
        )
        teardown_once()

    comparisons: dict[str, dict[str, Any]] = {}
    final_result = "passed"
    for mode, candidate_run in candidate_runs.items():
        job_returncode, job_name = submit_crps_job(
            run_id=run_id,
            comparison_label=mode,
            comparison_image=comparison_image,
            baseline_zarr=baseline_run.forecast_zarr_path,
            candidate_zarr=candidate_run.forecast_zarr_path,
            threshold=args.threshold,
            lead_time_chunk_size=args.lead_time_chunk_size,
            device=args.device,
            variables=args.variables,
            workspace_id=args.workspace_id,
            workspace_token=workspace_token,
            workspace_url=workspace_url,
            nfs_path=nfs_path,
            mount_target=args.mount_target,
            lustre_storage=args.lustre_storage,
            node_group=args.node_group,
            resource_shape=args.resource_shape,
            pull_secret=args.pull_secret,
            artifact_dir=artifact_root,
            comparison_script=args.comparison_script,
            keep_batch_job=args.keep_batch_job,
            dry_run=False,
        )
        comparison_report = parse_crps_report(
            artifact_root / "crps-jobs" / f"{job_name}.log"
        )
        if job_returncode != 0:
            final_result = "failed"
        comparisons[mode] = {
            "candidate_run_id": candidate_run.execution_id,
            "candidate_zarr": candidate_run.forecast_zarr_path,
            "comparison_job_name": job_name,
            "comparison_job_exit_code": job_returncode,
            "comparison_report": comparison_report,
        }

    summary = {
        **config_payload,
        "baseline_endpoint_url": baseline_url,
        "candidate_endpoint_url": candidate_url,
        "baseline_endpoint_name": baseline_name,
        "candidate_endpoint_name": candidate_name,
        "baseline_run_id": baseline_run.execution_id,
        "baseline_zarr": baseline_run.forecast_zarr_path,
        "comparisons": comparisons,
        "final_result": final_result,
    }
    write_json(run_artifact_dir / "summary.json", summary)
    print(f"==> Summary artifact: {run_artifact_dir / 'summary.json'}", flush=True)
    return 0 if final_result == "passed" else 1


def main() -> None:
    if not _REGISTRY:
        sys.exit(
            "ERROR: docker_registry not set. "
            "Configure deploy/config.yaml (see deploy/config.example.yaml)."
        )
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
