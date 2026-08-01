# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare ``physicsnemo-serve infer`` with the original PhysicsNeMo-CFD runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cfd_parity_contract import (
    ParityContractError,
    build_direct_config_for_request,
    compare_reports,
    output_by_name,
    read_json_object,
    resolved_dataset_root,
    sha256_file,
    validate_execution_result,
    validate_image_digest_reference,
    validate_profile,
    validate_request,
    validate_report_coverage,
    verify_staged_inputs_for_request,
    write_json_atomic,
)
from run_cfd_parity_job import (
    _emit_summary,
    _resolve_work_dir,
    _runner_environment,
    _verify_provider,
    utc_now,
)


ALLOWED_HTTPS_HOSTS = "huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"
ALLOWED_SIGNED_REDIRECT_HOSTS = "us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"


def _load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityContractError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityContractError(f"{label} must be a non-empty string")
    return value


def _resolve_infer_artifact(
    result: Mapping[str, Any],
    name: str,
    *,
    output_root: Path,
) -> Path:
    output = output_by_name(result, name)
    path = Path(
        _string(output.get("storage_path"), f"infer output {name} storage_path")
    ).resolve(strict=True)
    resolved_root = output_root.resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ParityContractError(
            f"infer output {name} escapes output root {resolved_root}"
        ) from exc
    if not path.is_file():
        raise ParityContractError(f"infer output {name} is not a regular file")
    return path


def _validate_infer_result(
    *,
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    output_root: Path,
    expected_run_id: str,
) -> dict[str, Any]:
    request_envelope = _mapping(result.get("request"), "infer result.request")
    if request_envelope.get("operation") != "run":
        raise ParityContractError("infer result operation must be 'run'")
    if request_envelope.get("raw_fields") != request:
        raise ParityContractError(
            "infer result request does not match the submitted request"
        )

    validated = validate_execution_result(
        profile,
        request,
        source_label="infer",
        run_id=result.get("run_id"),
        workflow=result.get("workflow"),
        status=result.get("status"),
        payload=result.get("payload"),
    )
    if validated["run_id"] != expected_run_id:
        raise ParityContractError(
            "infer result run_id does not match the requested run ID"
        )
    report_path = _resolve_infer_artifact(
        result,
        "benchmark_results.json",
        output_root=output_root,
    )
    config_path = _resolve_infer_artifact(
        result,
        "resolved_config.json",
        output_root=output_root,
    )

    execution = _mapping(result.get("execution"), "infer result.execution")
    primary_path = Path(
        _string(execution.get("output_path"), "infer result.execution.output_path")
    ).resolve(strict=True)
    if primary_path != report_path:
        raise ParityContractError("infer primary output is not benchmark_results.json")

    payload = _mapping(result.get("payload"), "infer result.payload")
    for field, expected in (
        ("report_path", report_path),
        ("resolved_config_path", config_path),
    ):
        payload_path = Path(
            _string(payload.get(field), f"infer result.payload.{field}")
        ).resolve(strict=True)
        if payload_path != expected:
            raise ParityContractError(
                f"infer payload {field} does not match its registered output"
            )

    resolved_config = read_json_object(config_path)
    config = _mapping(profile.get("config"), "profile.config")
    dataset = _mapping(config.get("dataset"), "profile.config.dataset")
    input_root = Path(
        resolved_dataset_root(
            resolved_config,
            _string(dataset.get("name"), "profile dataset name"),
        )
    ).resolve(strict=True)
    try:
        input_root.relative_to(output_root.resolve(strict=True))
    except ValueError as exc:
        raise ParityContractError(
            f"infer dataset root escapes output root: {input_root}"
        ) from exc
    if not input_root.is_dir():
        raise ParityContractError(
            f"infer dataset root is not a directory: {input_root}"
        )
    return {
        **validated,
        "report_path": report_path,
        "config_path": config_path,
        "input_root": input_root,
    }


def _runtime_environment(
    profile: Mapping[str, Any],
    *,
    mount_root: Path,
    runtime_dir: Path,
    device: str,
    download_timeout_seconds: int,
) -> dict[str, str]:
    environment = _runner_environment(profile, mount_root=mount_root)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": device,
            "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS": ALLOWED_HTTPS_HOSTS,
            "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS": (
                ALLOWED_SIGNED_REDIRECT_HOSTS
            ),
            "E2S_DOWNLOAD_TIMEOUT_SECS": str(download_timeout_seconds),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(runtime_dir / "scripts"),
                    str(runtime_dir / "python"),
                ]
            ),
        }
    )
    return environment


def _run_process_group(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: Any,
    stderr: Any,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(env),
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    try:
        process_stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 10
        group_alive = True
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except (ProcessLookupError, PermissionError):
                group_alive = False
                break
            time.sleep(0.1)
        if group_alive:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.wait()
        raise
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=process_stdout,
    )


def run(args: argparse.Namespace) -> int:
    started_at = utc_now()
    started = time.monotonic()
    work_dir = Path(args.work_dir)
    summary_path = work_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "final_result": "failed",
        "started_at": started_at,
        "profile_path": str(args.profile),
        "request_path": str(args.request),
        "work_dir": str(work_dir),
        "candidate": "physicsnemo-serve infer",
        "baseline": "physicsnemo-cfd",
    }
    try:
        if args.infer_timeout_seconds <= 0:
            raise ValueError("--infer-timeout-seconds must be positive")
        if args.download_timeout_seconds <= 0:
            raise ValueError("--download-timeout-seconds must be positive")

        profile = read_json_object(args.profile)
        request = read_json_object(args.request)
        validate_profile(profile)
        validate_request(profile, request)
        validate_image_digest_reference(args.image, "--image")
        mount_root, work_dir = _resolve_work_dir(
            args.mount_target,
            args.work_dir,
        )
        summary_path = work_dir / "summary.json"

        runtime_dir = Path(args.runtime_dir).resolve(strict=True)
        runtime_python = runtime_dir / "bin" / "python"
        if not runtime_python.is_file():
            raise ParityContractError(
                f"runtime Python does not exist: {runtime_python}"
            )
        infer_binary = Path(args.infer_binary).resolve(strict=True)
        if not infer_binary.is_file():
            raise ParityContractError(
                f"physicsnemo-serve binary does not exist: {infer_binary}"
            )
        plugin_root = Path(args.plugin).resolve(strict=True)
        if not plugin_root.is_dir() or not (plugin_root / "plugin.yaml").is_file():
            raise ParityContractError(
                f"PhysicsNeMo-CFD plugin is invalid: {plugin_root}"
            )

        environment = _runtime_environment(
            profile,
            mount_root=mount_root,
            runtime_dir=runtime_dir,
            device=args.device,
            download_timeout_seconds=args.download_timeout_seconds,
        )
        infer_root = work_dir / "infer"
        infer_root.mkdir(mode=0o750)
        infer_output_root = infer_root / "outputs"
        infer_output_root.mkdir(mode=0o750)
        infer_result_path = infer_root / "result.json"
        infer_log_path = infer_root / "infer.log"
        infer_command = [
            str(infer_binary),
            "infer",
            "--runtime-dir",
            str(runtime_dir),
            "--plugin",
            str(plugin_root),
            "--request",
            str(args.request),
            "--output-dir",
            str(infer_output_root),
            "--run-id",
            str(args.infer_run_id),
            "--device",
            str(args.device),
        ]
        with infer_log_path.open("w", encoding="utf-8") as log_file:
            try:
                completed = _run_process_group(
                    infer_command,
                    cwd=infer_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    timeout=args.infer_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "physicsnemo-serve infer timed out after "
                    f"{args.infer_timeout_seconds}s"
                ) from exc
        infer_stdout = completed.stdout or ""
        infer_result_path.write_text(infer_stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"physicsnemo-serve infer exited with code {completed.returncode}"
            )
        try:
            infer_result_value = json.loads(infer_stdout)
        except json.JSONDecodeError as exc:
            raise ParityContractError(
                "physicsnemo-serve infer did not emit valid JSON"
            ) from exc
        infer_result = _mapping(infer_result_value, "infer result")
        infer = _validate_infer_result(
            profile=profile,
            request=request,
            result=infer_result,
            output_root=infer_output_root,
            expected_run_id=str(args.infer_run_id),
        )
        infer_report_path = Path(infer["report_path"])
        infer_report_size = infer_report_path.stat().st_size
        infer_report_sha256 = sha256_file(infer_report_path)
        verified_inputs = verify_staged_inputs_for_request(
            profile,
            request,
            input_root=infer["input_root"],
        )

        provider = _verify_provider(profile["provider"])
        direct_root = work_dir / "physicsnemo-cfd"
        direct_root.mkdir(mode=0o750)
        direct_output_dir = direct_root / "benchmark-output"
        direct_output_dir.mkdir(mode=0o750)
        direct_config = build_direct_config_for_request(
            profile,
            request,
            input_root=infer["input_root"],
            output_dir=direct_output_dir,
        )
        direct_config_path = direct_root / "direct_config.json"
        write_json_atomic(direct_config_path, direct_config)
        runner = _mapping(profile.get("runner"), "profile.runner")
        direct_command = [
            str(runtime_python),
            "-m",
            _string(runner.get("module"), "profile.runner.module"),
            "--config",
            str(direct_config_path),
        ]
        direct_log_path = direct_root / "direct.log"
        with direct_log_path.open("w", encoding="utf-8") as log_file:
            try:
                completed = _run_process_group(
                    direct_command,
                    cwd=direct_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=int(runner["timeout_seconds"]),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "PhysicsNeMo-CFD direct run timed out after "
                    f"{runner['timeout_seconds']}s"
                ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"PhysicsNeMo-CFD direct run exited with code {completed.returncode}"
            )

        direct_report_path = direct_output_dir / "benchmark_results.json"
        if not direct_report_path.is_file():
            raise RuntimeError(
                f"PhysicsNeMo-CFD direct report was not created: {direct_report_path}"
            )
        if infer_report_path.stat().st_size != infer_report_size:
            raise ParityContractError(
                "infer report size changed during the direct provider run"
            )
        if sha256_file(infer_report_path) != infer_report_sha256:
            raise ParityContractError(
                "infer report digest changed during the direct provider run"
            )
        infer_report = _load_json(infer_report_path)
        direct_report = _load_json(direct_report_path)
        expected_metric_values = validate_report_coverage(
            profile,
            request,
            infer_report,
        )
        validate_report_coverage(
            profile,
            request,
            direct_report,
        )
        comparison = compare_reports(
            rest_report=infer_report,
            direct_report=direct_report,
            comparison=_mapping(profile.get("comparison"), "profile.comparison"),
            rest_label="infer",
            direct_label="physicsnemo_cfd",
            metric_outputs=_mapping(
                profile.get("report_metric_outputs"),
                "profile.report_metric_outputs",
            ),
        )
        comparison.update(
            {
                "infer_report": str(infer_report_path),
                "infer_report_sha256": infer_report_sha256,
                "physicsnemo_cfd_report": str(direct_report_path),
                "physicsnemo_cfd_report_sha256": sha256_file(direct_report_path),
                "metric_values_per_report": expected_metric_values,
            }
        )
        comparison_path = work_dir / "comparison.json"
        write_json_atomic(comparison_path, comparison)

        summary.update(
            {
                "final_result": comparison["status"],
                "parity_run_id": args.parity_run_id,
                "infer_run_id": infer["run_id"],
                "profile_id": profile["profile_id"],
                "domain": profile["domain"],
                "image": args.image,
                "provider": provider,
                "infer_provenance": {
                    "provider": infer["provider"],
                    "preset_sha256": infer["preset_sha256"],
                    "case_digests": infer["case_digests"],
                },
                "verified_inputs": verified_inputs,
                "commands": {
                    "infer": infer_command,
                    "physicsnemo_cfd": direct_command,
                },
                "artifacts": {
                    "infer_result": str(infer_result_path),
                    "infer_log": str(infer_log_path),
                    "infer_config": str(infer["config_path"]),
                    "infer_report": str(infer["report_path"]),
                    "physicsnemo_cfd_config": str(direct_config_path),
                    "physicsnemo_cfd_log": str(direct_log_path),
                    "physicsnemo_cfd_report": str(direct_report_path),
                    "comparison": str(comparison_path),
                    "summary": str(summary_path),
                },
                "comparison": comparison,
            }
        )
    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    summary["finished_at"] = utc_now()
    summary["duration_seconds"] = time.monotonic() - started
    write_json_atomic(summary_path, summary)
    _emit_summary(summary)
    return 0 if summary["final_result"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--mount-target", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--parity-run-id", required=True)
    parser.add_argument("--infer-run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--infer-binary",
        default="/usr/local/bin/physicsnemo-serve",
    )
    parser.add_argument(
        "--runtime-dir",
        default="/opt/physicsnemo-serve/runtimes/shared",
    )
    parser.add_argument(
        "--plugin",
        default=("/opt/physicsnemo-serve/plugins/physicsnemo-cfd-surface-benchmark"),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--infer-timeout-seconds", type=int, default=23_400)
    parser.add_argument("--download-timeout-seconds", type=int, default=1_800)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
