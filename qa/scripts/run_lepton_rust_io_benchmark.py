#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit a Lepton batch job for the Rust I/O benchmark QA flow."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import secrets
import shlex
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
QA_ROOT = SCRIPTS_DIR.parent
DEFAULT_ARTIFACT_DIR = QA_ROOT / "artifacts"


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
from config import load_deploy_config  # noqa: E402


_DEPLOY_CFG = load_deploy_config()
_REGISTRY = _DEPLOY_CFG.get("docker_registry", "")
DEFAULT_IMAGE_NAME = f"{_REGISTRY}/{_DEPLOY_CFG.get('image_name', 'scicomp-ferroflux')}"
DEFAULT_LUSTRE_STORAGE = "lustre"
DEFAULT_MOUNT_TARGET = "/outputs"
DEFAULT_NODE_GROUP = _DEPLOY_CFG.get("lepton_node_group", "")
DEFAULT_PULL_SECRET = _DEPLOY_CFG.get("pull_secret", "")
DEFAULT_RESOURCE_SHAPE = "gpu.h100-sxm"
DEFAULT_MODELS = "fcn,dlwp,stormcast,sfno,fcn3"
DEFAULT_REPORT_MODELS = "fcn,dlwp,sfno,stormcast,fcn3"
DEFAULT_BACKENDS = "rust,py_async,zarr_sync"
DEFAULT_REMOTE_EARTH2STUDIO_ROOT = (
    "/opt/physicsnemo-serve-venv/lib/python3.12/site-packages"
)
JOB_TIMEOUT_SECONDS = 24 * 60 * 60
JOB_POLL_INTERVAL_SECONDS = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_lustre_dir() -> str:
    return f"rust_io_tests_{datetime.now(timezone.utc):%Y%m%d}"


def generate_run_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


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
    return " ".join(shlex.quote(arg) for arg in args)


def job_name_part(value: str, *, max_len: int = 20) -> str:
    part = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    part = re.sub(r"-+", "-", part)
    return (part or "run")[:max_len].strip("-") or "run"


def model_scope_label(args: argparse.Namespace) -> str:
    models = [
        item.strip() for item in str(args.models or "").split(",") if item.strip()
    ]
    if args.preset == "benchmark-report" and ",".join(models) == DEFAULT_REPORT_MODELS:
        return "report"
    if args.preset == "custom" and ",".join(models) == DEFAULT_MODELS:
        return "all"
    if len(models) == 1:
        return models[0]
    if 1 < len(models) <= 3:
        return "-".join(models)
    if models:
        return f"{models[0]}-{len(models)}m"
    return "models"


def lepton_job_name(
    args: argparse.Namespace, *, run_id: str, suffix: str | None = None
) -> str:
    preset_label = "report" if args.preset == "benchmark-report" else "custom"
    scope_label = model_scope_label(args)
    parts = ["ff-rio", preset_label]
    if scope_label != preset_label:
        parts.append(scope_label)
    parts.append(run_id)
    if suffix:
        parts.append(suffix)
    name = "-".join(job_name_part(part) for part in parts)
    return name[:63].rstrip("-")


def output_delta(previous: str, current: str) -> str:
    if not current or current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]
    previous_lines = set(previous.splitlines())
    new_lines = [line for line in current.splitlines() if line not in previous_lines]
    return "\n".join(new_lines) + ("\n" if new_lines else "")


def run_streaming(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        cmd,
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


def parse_job_id(output: str) -> str | None:
    for line in output.splitlines():
        match = re.match(r"\s*ID:\s*(\S+)", line)
        if match:
            return match.group(1)
    return None


def job_succeeded(status_output: str) -> bool:
    return bool(
        re.search(r'"state":\s*"(Completed|Succeeded|Success)"', status_output, re.I)
    )


def job_failed(status_output: str) -> bool:
    return bool(
        re.search(r'"state":\s*"(Failed|Cancelled|Stopped|Error)"', status_output, re.I)
    )


def remote_output_dir(mount_target: str, run_id: str) -> str:
    return f"{mount_target.rstrip('/')}/rust-io-benchmark/{run_id}"


def remote_job_runner_path(mount_target: str, run_id: str) -> str:
    return f"{remote_output_dir(mount_target, run_id)}/run_rust_io_benchmark_job.py"


def remote_benchmark_script_path(mount_target: str, run_id: str) -> str:
    return (
        f"{remote_output_dir(mount_target, run_id)}"
        "/compare_deterministic_rust_vs_py_async.py"
    )


def local_artifact_dir(artifact_root: Path, run_id: str) -> Path:
    return artifact_root / "rust-io-benchmark" / run_id


def base64_write_command(path: str, text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}\n"


def base64_python_command(text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"printf %s {shlex.quote(encoded)} | base64 -d | python\n"


def setup_scripts_command(mount_target: str, run_id: str) -> str:
    """Return shell that writes QA scripts into the shared mount."""

    runner_source = SCRIPTS_DIR / "run_rust_io_benchmark_job.py"
    benchmark_source = (
        QA_ROOT.parent / "scripts" / "compare_deterministic_rust_vs_py_async.py"
    )
    return base64_write_command(
        remote_job_runner_path(mount_target, run_id),
        runner_source.read_text(encoding="utf-8"),
    ) + base64_write_command(
        remote_benchmark_script_path(mount_target, run_id),
        benchmark_source.read_text(encoding="utf-8"),
    )


def build_remote_command(args: argparse.Namespace, run_id: str) -> str:
    output_dir = remote_output_dir(args.mount_target, run_id)
    log_path = f"{output_dir}/job-output.log"
    benchmark_script = args.benchmark_script or remote_benchmark_script_path(
        args.mount_target, run_id
    )
    job_cmd = [
        "python",
        remote_job_runner_path(args.mount_target, run_id),
        "--output-dir",
        output_dir,
        "--preset",
        args.preset,
        "--backends",
        args.backends,
        "--start-time",
        args.start_time,
        "--nsteps",
        str(args.nsteps),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--rust-profile",
        args.rust_profile,
        "--warmup-steps",
        str(args.warmup_steps),
        "--profile-top-n",
        str(args.profile_top_n),
        "--benchmark-script",
        benchmark_script,
    ]
    if args.models:
        job_cmd += ["--models", args.models]
    if args.earth2studio_root:
        job_cmd += ["--earth2studio-root", args.earth2studio_root]

    return (
        f"mkdir -p {shlex.quote(output_dir)}; "
        "{ "
        f"{setup_scripts_command(args.mount_target, run_id)}"
        f"{shell_join(job_cmd)}; "
        "} "
        f"2>&1 | tee {shlex.quote(log_path)}; "
        f"rc=${{PIPESTATUS[0]}}; exit $rc"
    )


def build_job_args(args: argparse.Namespace, *, run_id: str) -> list[str]:
    image = image_full_reference(args.image_tag, args.image_name)
    job_name = lepton_job_name(args, run_id=run_id)
    return [
        "--name",
        job_name,
        "--container-image",
        image,
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.resource_shape,
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        f"{args.nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
        "--command",
        build_remote_command(args, run_id),
    ]


def build_summary_payload(
    args: argparse.Namespace,
    *,
    run_id: str,
    job_args: list[str],
    final_result: str,
    job_id: str | None = None,
    job_exit_code: int | None = None,
    benchmark_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = resolve_path(args.artifact_dir)
    run_artifact_dir = local_artifact_dir(artifact_root, run_id)
    remote_dir = remote_output_dir(args.mount_target, run_id)
    return {
        "created_at": utc_now(),
        "run_id": run_id,
        "final_result": final_result,
        "image": image_full_reference(args.image_tag, args.image_name),
        "config": {
            "models": args.models,
            "preset": args.preset,
            "backends": args.backends,
            "start_time": args.start_time,
            "nsteps": args.nsteps,
            "device": args.device,
            "seed": args.seed,
            "rust_profile": args.rust_profile,
            "warmup_steps": args.warmup_steps,
            "profile_top_n": args.profile_top_n,
            "earth2studio_root": args.earth2studio_root,
            "dry_run": args.dry_run,
        },
        "mount": {
            "nfs_path": args.nfs_path,
            "mount_target": args.mount_target,
            "lustre_storage": args.lustre_storage,
        },
        "lepton": {
            "job_name": lepton_job_name(args, run_id=run_id),
            "job_id": job_id,
            "job_args": job_args,
            "workspace_id": args.workspace_id,
            "workspace_url": args.workspace_url or None,
            "node_group": args.node_group,
            "resource_shape": args.resource_shape,
            "pull_secret": args.pull_secret,
            "job_exit_code": job_exit_code,
        },
        "artifacts": {
            "remote_output_dir": remote_dir,
            "remote_summary_json": f"{remote_dir}/summary.json",
            "remote_perf_compare_report_md": f"{remote_dir}/perf_compare_report.md",
            "local_artifact_dir": str(run_artifact_dir),
            "local_summary_json": str(run_artifact_dir / "summary.json"),
            "local_perf_compare_report_md": str(
                run_artifact_dir / "perf_compare_report.md"
            ),
            "job_log": str(
                artifact_root
                / "rust-io-jobs"
                / f"{lepton_job_name(args, run_id=run_id)}.log"
            ),
        },
        "benchmark_report": benchmark_report,
    }


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
    returncode, _output = run_streaming(cmd, env=env)
    if returncode != 0:
        raise RuntimeError(f"lep login exited with code {returncode}")


def poll_job(args: argparse.Namespace, *, job_id: str, env: dict[str, str]) -> int:
    deadline = time.time() + args.job_timeout
    last_status = ""
    while time.time() < deadline:
        result = subprocess.run(
            ["lep", "job", "get", "-i", job_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        status_output = result.stdout or ""
        delta = output_delta(last_status, status_output)
        if delta:
            print(delta, end="" if delta.endswith("\n") else "\n", flush=True)
        last_status = status_output
        if job_succeeded(status_output):
            return 0
        if job_failed(status_output):
            return 1
        time.sleep(args.job_poll_interval)
    print(f"Error: Lepton job timed out after {args.job_timeout}s", file=sys.stderr)
    return 1


def stop_job_if_needed(*, job_id: str, env: dict[str, str], job_exit_code: int) -> None:
    if job_exit_code == 0:
        return
    print(f"==> Stopping Lepton job before cleanup: {job_id}", flush=True)
    run_streaming(["lep", "job", "stop", "-i", job_id], env=env)


def wait_for_loggable_job(
    job_id: str, *, env: dict[str, str], timeout: int = 120
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["lep", "job", "get", "-i", job_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        status_output = result.stdout or ""
        if re.search(
            r'"state":\s*"(Running|Completed|Succeeded|Success|Failed|Error)"',
            status_output,
            re.I,
        ):
            time.sleep(5)
            return
        time.sleep(5)


def local_nfs_summary_path(args: argparse.Namespace, run_id: str) -> Path:
    return Path(args.nfs_path) / "rust-io-benchmark" / run_id / "summary.json"


def mirror_summary_if_available(
    args: argparse.Namespace, run_id: str
) -> dict[str, Any] | None:
    source = local_nfs_summary_path(args, run_id)
    try:
        exists = source.exists()
    except OSError:
        exists = False
    if not exists:
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination = (
        local_artifact_dir(resolve_path(args.artifact_dir), run_id)
        / "benchmark-summary.json"
    )
    write_json(destination, payload)
    return payload


def write_perf_report_if_available(
    *,
    args: argparse.Namespace,
    run_id: str,
    benchmark_report: dict[str, Any] | None,
) -> str | None:
    if benchmark_report is None:
        return None

    report_module_path = SCRIPTS_DIR / "run_rust_io_benchmark_job.py"
    spec = importlib.util.spec_from_file_location(
        "_physicsnemo_serve_run_rust_io_benchmark_job", report_module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load report generator from {report_module_path}")
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)

    destination = (
        local_artifact_dir(resolve_path(args.artifact_dir), run_id)
        / "perf_compare_report.md"
    )
    destination.write_text(
        report_module.generate_perf_compare_report(benchmark_report), encoding="utf-8"
    )
    return str(destination)


def extract_json_object(text: str) -> dict[str, Any] | None:
    begin = "PHYSICSNEMO_SERVE_RUST_IO_SUMMARY_BEGIN"
    end = "PHYSICSNEMO_SERVE_RUST_IO_SUMMARY_END"
    begin_index = text.rfind(begin)
    end_index = text.rfind(end)
    if begin_index >= 0 and end_index > begin_index:
        marked = text[begin_index + len(begin) : end_index].strip()
        try:
            payload = json.loads(marked)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    for payload in reversed(candidates):
        if "final_result" in payload:
            return payload
    return candidates[-1] if candidates else None


def capture_job_logs(
    *,
    job_id: str,
    env: dict[str, str],
    artifact_path: Path,
    timeout_seconds: int = 120,
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
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += (
            f"\n==> Timed out after {timeout_seconds}s while capturing Lepton job logs; "
            "continuing summary and cleanup.\n"
        )
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        with artifact_path.open("a", encoding="utf-8") as log_file:
            log_file.write(output)
            if not output.endswith("\n"):
                log_file.write("\n")
    return output


def fetch_summary_via_reader_job(
    args: argparse.Namespace,
    *,
    run_id: str,
    env: dict[str, str],
) -> dict[str, Any] | None:
    remote_summary = f"{remote_output_dir(args.mount_target, run_id)}/summary.json"
    reader_python = f"""
import json
payload = json.load(open({json.dumps(remote_summary)}, encoding="utf-8"))
config = payload.get("config", {{}})
config = {{
    "models": config.get("models"),
    "backends": config.get("backends"),
    "preset": config.get("preset"),
    "model_configs": config.get("model_configs"),
    "nsteps": config.get("nsteps"),
    "device": config.get("device"),
    "rust_profile": config.get("rust_profile"),
    "warmup_steps": config.get("warmup_steps"),
}}
models = {{}}
for name, model in payload.get("models", {{}}).items():
    models[name] = {{
        "status": model.get("status"),
        "returncode": model.get("returncode"),
        "backends": model.get("backends"),
        "ratios": model.get("ratios"),
        "consistency": {{
            "all_consistent": model.get("consistency", {{}}).get("all_consistent"),
            "skipped": model.get("consistency", {{}}).get("skipped"),
            "max_abs_diff_global": model.get("consistency", {{}}).get("max_abs_diff_global"),
        }},
    }}
out = {{
    "final_result": payload.get("final_result"),
    "config": config,
    "models": models,
}}
print("PHYSICSNEMO_SERVE_RUST_IO_SUMMARY_BEGIN")
print(json.dumps(out, indent=2, sort_keys=True))
print("PHYSICSNEMO_SERVE_RUST_IO_SUMMARY_END")
""".strip()
    reader_name = lepton_job_name(args, run_id=run_id, suffix="fetch")
    reader_args = [
        "--name",
        reader_name,
        "--container-image",
        image_full_reference(args.image_tag, args.image_name),
        "--node-group",
        args.node_group,
        "--resource-shape",
        args.resource_shape,
        "--image-pull-secrets",
        args.pull_secret,
        "--mount",
        f"{args.nfs_path}:{args.mount_target}:node-nfs:{args.lustre_storage}",
        "--command",
        f"{base64_python_command(reader_python)}sleep 30",
    ]
    print(f"==> Fetching benchmark summary via reader job: {reader_name}", flush=True)
    returncode, create_output = run_streaming(
        ["lep", "job", "create", *reader_args], env=env
    )
    if returncode != 0:
        return None
    reader_job_id = parse_job_id(create_output)
    if not reader_job_id:
        return None
    wait_for_loggable_job(reader_job_id, env=env)
    try:
        output = capture_job_logs(
            job_id=reader_job_id,
            env=env,
            artifact_path=resolve_path(args.artifact_dir)
            / "rust-io-jobs"
            / f"{reader_name}.log",
            timeout_seconds=60,
        )
    finally:
        run_streaming(["lep", "job", "remove", "-i", reader_job_id], env=env)
    payload = extract_json_object(output)
    if payload is not None:
        destination = (
            local_artifact_dir(resolve_path(args.artifact_dir), run_id)
            / "benchmark-summary.json"
        )
        write_json(destination, payload)
    return payload


def run(args: argparse.Namespace) -> int:
    args = normalize_args(args)
    run_id = args.run_id or generate_run_id()
    artifact_root = resolve_path(args.artifact_dir)
    run_artifact_dir = local_artifact_dir(artifact_root, run_id)
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    job_args = build_job_args(args, run_id=run_id)
    job_log = (
        artifact_root / "rust-io-jobs" / f"{lepton_job_name(args, run_id=run_id)}.log"
    )

    if args.dry_run:
        summary = build_summary_payload(
            args,
            run_id=run_id,
            job_args=job_args,
            final_result="dry-run",
            job_exit_code=0,
        )
        write_json(run_artifact_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(
            f"==> Dry run complete. Summary artifact: {run_artifact_dir / 'summary.json'}"
        )
        return 0

    env = {**os.environ}
    if args.workspace_token:
        env["LEPTON_WORKSPACE_TOKEN"] = args.workspace_token
    if args.workspace_url:
        env["LEPTON_WORKSPACE_URL"] = args.workspace_url

    login_if_needed(args, env)
    job_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["lep", "job", "create", *job_args]
    print(f"+ {shell_join(cmd)}", flush=True)
    with job_log.open("w", encoding="utf-8") as log_file:
        returncode, create_output = run_streaming(cmd, env=env)
        log_file.write(create_output)

    if returncode != 0:
        summary = build_summary_payload(
            args,
            run_id=run_id,
            job_args=job_args,
            final_result="failed",
            job_exit_code=returncode,
        )
        write_json(run_artifact_dir / "summary.json", summary)
        return returncode

    job_id = parse_job_id(create_output)
    if not job_id:
        raise RuntimeError("could not parse Lepton job ID from create output")

    job_exit_code = poll_job(args, job_id=job_id, env=env)
    stop_job_if_needed(job_id=job_id, env=env, job_exit_code=job_exit_code)
    capture_job_logs(job_id=job_id, env=env, artifact_path=job_log)
    benchmark_report = mirror_summary_if_available(args, run_id)
    if benchmark_report is None:
        benchmark_report = fetch_summary_via_reader_job(args, run_id=run_id, env=env)
    perf_report_path = write_perf_report_if_available(
        args=args,
        run_id=run_id,
        benchmark_report=benchmark_report,
    )
    if not args.keep_job:
        run_streaming(["lep", "job", "remove", "-i", job_id], env=env)

    final_result = (
        "passed"
        if job_exit_code == 0
        and benchmark_report is not None
        and benchmark_report.get("final_result") == "passed"
        else "failed"
    )
    summary = build_summary_payload(
        args,
        run_id=run_id,
        job_args=job_args,
        final_result=final_result,
        job_id=job_id,
        job_exit_code=job_exit_code,
        benchmark_report=benchmark_report,
    )
    if perf_report_path is not None:
        summary["artifacts"]["local_perf_compare_report_md"] = perf_report_path
    write_json(run_artifact_dir / "summary.json", summary)
    print(f"==> Summary artifact: {run_artifact_dir / 'summary.json'}", flush=True)
    if perf_report_path is not None:
        print(f"==> Performance comparison report: {perf_report_path}", flush=True)
    return 0 if final_result == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a Lepton batch job for Rust I/O benchmark QA."
    )
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated models. Defaults to all smoke models for custom preset "
            f"or {DEFAULT_REPORT_MODELS} for benchmark-report preset."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["custom", "benchmark-report"],
        default="custom",
        help="Use benchmark-report to match docs/benchmark_report_e2s_zarr_io.md.",
    )
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--start-time", default="2024-01-01T00:00:00")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rust-profile", default="default")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-top-n", type=int, default=40)
    parser.add_argument("--benchmark-script", default=None)
    parser.add_argument("--earth2studio-root", default=DEFAULT_REMOTE_EARTH2STUDIO_ROOT)
    parser.add_argument("--run-id", default=None)
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
        "--workspace-token", default=os.environ.get("LEPTON_WORKSPACE_TOKEN", "")
    )
    parser.add_argument(
        "--lustre-dir",
        default=os.environ.get("QA_LUSTRE_DIR", default_lustre_dir()),
    )
    parser.add_argument("--lustre-storage", default=DEFAULT_LUSTRE_STORAGE)
    parser.add_argument("--mount-target", default=DEFAULT_MOUNT_TARGET)
    parser.add_argument(
        "--node-group", default=os.environ.get("LEPTON_NODE_GROUP", DEFAULT_NODE_GROUP)
    )
    parser.add_argument(
        "--resource-shape",
        default=os.environ.get("LEPTON_RESOURCE_SHAPE", DEFAULT_RESOURCE_SHAPE),
    )
    parser.add_argument(
        "--pull-secret",
        default=os.environ.get("LEPTON_PULL_SECRET", DEFAULT_PULL_SECRET),
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("QA_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)),
    )
    parser.add_argument("--job-timeout", type=int, default=JOB_TIMEOUT_SECONDS)
    parser.add_argument(
        "--job-poll-interval", type=int, default=JOB_POLL_INTERVAL_SECONDS
    )
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(nfs_path=None)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.nfs_path is None:
        args.nfs_path = (
            f"{_DEPLOY_CFG.get('nfs_mount_base', '/mnt/shared')}/{args.lustre_dir}"
        )
    if args.models is None:
        args.models = (
            DEFAULT_REPORT_MODELS
            if args.preset == "benchmark-report"
            else DEFAULT_MODELS
        )
    return args


def main() -> None:
    if not _REGISTRY:
        sys.exit(
            "ERROR: docker_registry not set. "
            "Configure deploy/config.yaml (see deploy/config.example.yaml)."
        )
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
