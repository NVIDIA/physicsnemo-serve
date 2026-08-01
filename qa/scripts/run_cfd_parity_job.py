# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run PhysicsNeMo-CFD directly and compare its report with a REST execution."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cfd_parity_contract import (
    ParityContractError,
    build_direct_config,
    compare_reports,
    read_json_object,
    resolve_existing_mount_path,
    sha256_file,
    validate_handoff,
    validate_profile,
    validate_report_coverage,
    verify_staged_inputs,
    write_json_atomic,
)


SUMMARY_BEGIN = "PHYSICSNEMO_CFD_PARITY_SUMMARY_BEGIN"
SUMMARY_END = "PHYSICSNEMO_CFD_PARITY_SUMMARY_END"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_provider(provider: Mapping[str, Any]) -> dict[str, str]:
    module_name = str(provider["module"])
    distribution_name = str(provider["distribution"])
    module = importlib.import_module(module_name)
    actual_version = str(getattr(module, "__version__", ""))
    expected_version = str(provider["version"])
    if actual_version != expected_version:
        raise RuntimeError(
            f"provider version mismatch: {actual_version!r} != {expected_version!r}"
        )

    distribution = importlib.metadata.distribution(distribution_name)
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{distribution_name} has no direct_url.json")
    direct_url = json.loads(direct_url_text)
    actual_commit = str((direct_url.get("vcs_info") or {}).get("commit_id") or "")
    expected_commit = str(provider["commit"])
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"provider commit mismatch: {actual_commit!r} != {expected_commit!r}"
        )

    expected_python = provider.get("python_version")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if expected_python is not None and actual_python != str(expected_python):
        raise RuntimeError(
            f"Python version mismatch: {actual_python!r} != {expected_python!r}"
        )
    expected_physicsnemo = provider.get("physicsnemo_version")
    actual_physicsnemo = importlib.metadata.version("nvidia-physicsnemo")
    if expected_physicsnemo is not None and actual_physicsnemo != str(
        expected_physicsnemo
    ):
        raise RuntimeError(
            "PhysicsNeMo version mismatch: "
            f"{actual_physicsnemo!r} != {expected_physicsnemo!r}"
        )
    return {
        "distribution": distribution_name,
        "module": module_name,
        "version": actual_version,
        "commit": actual_commit,
        "python_version": actual_python,
        "physicsnemo_version": actual_physicsnemo,
    }


def _resolve_work_dir(mount_target: str, work_dir: str) -> tuple[Path, Path]:
    mount_root = Path(mount_target).resolve(strict=True)
    work = Path(work_dir).resolve(strict=True)
    try:
        work.relative_to(mount_root)
    except ValueError as exc:
        raise ParityContractError(
            f"work directory {work} escapes mount target {mount_root}"
        ) from exc
    if not work.is_dir():
        raise ParityContractError(f"work directory is not a directory: {work}")
    return mount_root, work


def _runner_environment(
    profile: Mapping[str, Any], *, mount_root: Path
) -> dict[str, str]:
    environment = dict(os.environ)
    configured = profile["runner"].get("environment", {})
    for name, raw_value in configured.items():
        value = str(raw_value).replace("$MOUNT_TARGET", str(mount_root))
        if "$" in value:
            raise ParityContractError(
                f"unsupported environment placeholder in {name}: {raw_value!r}"
            )
        environment[str(name)] = value
        if name in {"HF_HOME", "PHYSICSNEMO_CFD_MODEL_CACHE"}:
            cache_path = Path(value)
            try:
                cache_path.relative_to(mount_root)
            except ValueError as exc:
                raise ParityContractError(
                    f"{name} must remain under the shared mount"
                ) from exc
            cache_path.mkdir(parents=True, exist_ok=True)
    return environment


def _emit_summary(summary: Mapping[str, Any]) -> None:
    print(SUMMARY_BEGIN, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(SUMMARY_END, flush=True)


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
        "handoff_path": str(args.handoff),
        "work_dir": str(work_dir),
    }
    try:
        profile = read_json_object(args.profile)
        handoff = read_json_object(args.handoff)
        validate_profile(profile)
        validate_handoff(profile, handoff)
        if args.mount_target != handoff["mount_target"]:
            raise ParityContractError(
                "--mount-target does not match the parity handoff"
            )

        mount_root, work_dir = _resolve_work_dir(args.mount_target, args.work_dir)
        summary_path = work_dir / "summary.json"
        rest = handoff["rest"]
        input_root = resolve_existing_mount_path(
            args.mount_target, rest["input_root_relpath"]
        )
        rest_report_path = resolve_existing_mount_path(
            args.mount_target, rest["report_relpath"]
        )
        if not rest_report_path.is_file():
            raise ParityContractError(
                f"REST report is not a regular file: {rest_report_path}"
            )
        if rest_report_path.stat().st_size != rest["report_size_bytes"]:
            raise ParityContractError("REST report size changed after handoff")
        if sha256_file(rest_report_path) != rest["report_sha256"]:
            raise ParityContractError("REST report digest changed after handoff")

        direct_root = work_dir / "direct"
        direct_root.mkdir(mode=0o750)
        output_dir = direct_root / "benchmark-output"
        output_dir.mkdir(mode=0o750)
        verified_inputs = verify_staged_inputs(
            profile,
            handoff,
            input_root=input_root,
        )
        provider = _verify_provider(profile["provider"])
        direct_config = build_direct_config(
            profile,
            handoff,
            input_root=input_root,
            output_dir=output_dir,
        )
        config_path = direct_root / "direct_config.json"
        write_json_atomic(config_path, direct_config)

        runner = profile["runner"]
        command = [
            str(runner["python"]),
            "-m",
            str(runner["module"]),
            "--config",
            str(config_path),
        ]
        log_path = direct_root / "direct.log"
        environment = _runner_environment(profile, mount_root=mount_root)
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                completed = subprocess.run(
                    command,
                    cwd=direct_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=int(runner["timeout_seconds"]),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"direct provider run timed out after {runner['timeout_seconds']}s"
                ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"direct provider run exited with code {completed.returncode}"
            )

        direct_report_path = output_dir / "benchmark_results.json"
        if not direct_report_path.is_file():
            raise RuntimeError(
                f"direct provider report was not created: {direct_report_path}"
            )
        rest_report = _load_json(rest_report_path)
        direct_report = _load_json(direct_report_path)
        expected_metric_values = validate_report_coverage(
            profile,
            handoff["request"],
            rest_report,
        )
        validate_report_coverage(
            profile,
            handoff["request"],
            direct_report,
        )
        comparison = compare_reports(
            rest_report=rest_report,
            direct_report=direct_report,
            comparison=profile["comparison"],
            metric_outputs=profile["report_metric_outputs"],
        )
        comparison.update(
            {
                "rest_report": str(rest_report_path),
                "rest_report_sha256": rest["report_sha256"],
                "direct_report": str(direct_report_path),
                "direct_report_sha256": sha256_file(direct_report_path),
                "metric_values_per_report": expected_metric_values,
            }
        )
        comparison_path = work_dir / "comparison.json"
        write_json_atomic(comparison_path, comparison)

        summary.update(
            {
                "final_result": comparison["status"],
                "parity_run_id": handoff["parity_run_id"],
                "rest_run_id": handoff["rest_run_id"],
                "profile_id": profile["profile_id"],
                "domain": profile["domain"],
                "image": handoff["image"],
                "provider": provider,
                "verified_inputs": verified_inputs,
                "command": command,
                "artifacts": {
                    "rest_report": str(rest_report_path),
                    "direct_config": str(config_path),
                    "direct_log": str(log_path),
                    "direct_report": str(direct_report_path),
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
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--mount-target", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
