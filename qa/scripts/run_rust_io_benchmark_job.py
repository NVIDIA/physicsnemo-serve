#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Rust/Python/Async Zarr I/O benchmarks across Earth2Studio models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_SCRIPT = (
    REPO_ROOT / "scripts" / "compare_deterministic_rust_vs_py_async.py"
)
DEFAULT_MODELS = ("fcn", "dlwp", "stormcast", "sfno", "fcn3")
DEFAULT_BACKENDS = ("rust", "py_async", "zarr_sync")
REPORT_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "fcn": {
        "nsteps": 10,
        "max_pool_buffers": 64,
        "hot_slab_buffers": 26,
        "warm_slab_buffers": 26,
        "queue_capacity": 128,
    },
    "dlwp": {
        "nsteps": 20,
        "max_pool_buffers": 128,
        "hot_slab_buffers": 52,
        "warm_slab_buffers": 26,
        "skip_consistency": True,
    },
    "sfno": {
        "nsteps": 10,
        "max_pool_buffers": 300,
        "hot_slab_buffers": 150,
        "warm_slab_buffers": 75,
        "max_pool_bytes": 2_147_483_648,
    },
    "stormcast": {
        "nsteps": 10,
        "max_pool_buffers": 200,
        "hot_slab_buffers": 100,
        "warm_slab_buffers": 50,
        "max_pool_bytes": 1_073_741_824,
    },
    "fcn3": {
        "nsteps": 10,
        "max_pool_buffers": 150,
        "hot_slab_buffers": 73,
        "warm_slab_buffers": 36,
        "max_pool_bytes": 1_073_741_824,
    },
}
Runner = Callable[..., Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_csv(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def format_seconds(value: Any) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    if value < 1.0:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f} s"


def format_ratio(value: Any) -> str:
    return f"{value:.1f}x" if isinstance(value, int | float) else "n/a"


def backend_metric(model_payload: dict[str, Any], backend: str, metric: str) -> Any:
    backends = model_payload.get("backends")
    if not isinstance(backends, dict):
        return None
    backend_payload = backends.get(backend)
    if not isinstance(backend_payload, dict):
        return None
    return backend_payload.get(metric)


def model_ratio(model_payload: dict[str, Any], pair: str, metric: str) -> Any:
    ratios = model_payload.get("ratios")
    if not isinstance(ratios, dict):
        ratios = {}
    pair_payload = ratios.get(pair)
    if isinstance(pair_payload, dict):
        return pair_payload.get(metric)
    left, separator, right = pair.partition("_vs_")
    if not separator:
        return None
    reversed_payload = ratios.get(f"{right}_vs_{left}")
    if isinstance(reversed_payload, dict):
        value = reversed_payload.get(metric)
        if isinstance(value, int | float) and value != 0:
            return 1 / value

    metric_by_ratio = {
        "total_wall_ratio": "total_wall_sec",
        "total_io_ratio": "total_io_sec",
        "io_write_ratio": "io_write_sec",
    }
    backend_metric_name = metric_by_ratio.get(metric)
    if backend_metric_name is None:
        return None
    left_value = backend_metric(model_payload, left, backend_metric_name)
    right_value = backend_metric(model_payload, right, backend_metric_name)
    if (
        not isinstance(left_value, int | float)
        or not isinstance(right_value, int | float)
        or left_value == 0
    ):
        return None
    return right_value / left_value


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, int | float) else None


def summarize_comparison(
    model: str,
    comparison_json: Path,
    *,
    returncode: int,
) -> dict[str, Any]:
    """Extract the compact model summary from one benchmark comparison.json."""

    payload = json.loads(comparison_json.read_text(encoding="utf-8"))
    performance = payload.get("performance", {})
    consistency_payload = payload.get("consistency", {})

    backends: dict[str, dict[str, float | None]] = {}
    if isinstance(performance, dict):
        for backend, backend_payload in performance.items():
            if backend == "comparisons" or not isinstance(backend_payload, dict):
                continue
            backends[backend] = {
                "total_wall_sec": _nested_float(
                    backend_payload, "timings_sec", "total_wall_sec"
                ),
                "total_io_sec": _nested_float(
                    backend_payload, "io_metrics", "total_io"
                ),
                "io_write_sec": _nested_float(
                    backend_payload, "io_metrics", "io_write"
                ),
                "total_compute_sec": _nested_float(
                    backend_payload, "compute_metrics", "total_compute"
                ),
            }

    ratios = performance.get("comparisons", {}) if isinstance(performance, dict) else {}
    if not isinstance(ratios, dict):
        ratios = {}

    config_payload = payload.get("config", {})
    config_payload = config_payload if isinstance(config_payload, dict) else {}
    consistency_pairs = (
        consistency_payload if isinstance(consistency_payload, dict) else {}
    )
    consistency_skipped = bool(config_payload.get("skip_consistency"))
    all_consistent = True if consistency_skipped else bool(consistency_pairs)
    max_abs_diff_global = 0.0
    for check in consistency_pairs.values():
        if not isinstance(check, dict):
            all_consistent = False
            continue
        all_consistent = all_consistent and bool(check.get("all_consistent"))
        diff = check.get("max_abs_diff_global")
        if isinstance(diff, int | float):
            max_abs_diff_global = max(max_abs_diff_global, float(diff))

    status = "passed" if returncode == 0 and all_consistent else "failed"
    return {
        "model": model,
        "status": status,
        "returncode": returncode,
        "comparison_json": str(comparison_json),
        "run_dir": payload.get("artifacts", {}).get("run_dir")
        if isinstance(payload.get("artifacts"), dict)
        else None,
        "backends": backends,
        "ratios": ratios,
        "consistency": {
            "all_consistent": all_consistent,
            "skipped": consistency_skipped,
            "max_abs_diff_global": max_abs_diff_global,
            "pairs": consistency_pairs,
        },
    }


def find_latest_comparison_json(model_output_dir: Path) -> Path | None:
    matches = list(model_output_dir.rglob("comparison.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def build_benchmark_command(
    *,
    benchmark_script: Path,
    model: str,
    model_output_dir: Path,
    start_time: str,
    nsteps: int,
    device: str,
    backends: list[str],
    seed: int,
    rust_profile: str,
    warmup_steps: int,
    profile_top_n: int,
    earth2studio_root: Path | None,
    rust_options: dict[str, int] | None = None,
    skip_consistency: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--start-time",
        start_time,
        "--nsteps",
        str(nsteps),
        "--model-type",
        model,
        "--device",
        device,
        "--backends",
        ",".join(backends),
        "--seed",
        str(seed),
        "--rust-profile",
        rust_profile,
        "--warmup-steps",
        str(warmup_steps),
        "--profile-top-n",
        str(profile_top_n),
        "--output-dir",
        str(model_output_dir),
    ]
    if skip_consistency:
        cmd.append("--skip-consistency")
    if earth2studio_root is not None:
        cmd.extend(["--earth2studio-root", str(earth2studio_root)])
    for option_name, value in (rust_options or {}).items():
        cmd.extend([f"--{option_name.replace('_', '-')}", str(value)])
    return cmd


def run_command_streaming(cmd: list[str], *, log_path: Path) -> int:
    """Run a command, stream output to stdout, and persist the same output."""

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def model_config_for(
    model: str,
    *,
    preset: str,
    default_nsteps: int,
) -> dict[str, Any]:
    if preset == "custom":
        return {"nsteps": default_nsteps, "rust_options": {}}
    if preset != "benchmark-report":
        raise ValueError(f"unsupported preset: {preset}")
    try:
        config = REPORT_MODEL_CONFIGS[model]
    except KeyError as exc:
        raise ValueError(
            f"model {model!r} is not part of preset {preset!r}; "
            f"available: {sorted(REPORT_MODEL_CONFIGS)}"
        ) from exc
    rust_options = {
        key: value
        for key, value in config.items()
        if key not in {"nsteps", "skip_consistency"}
    }
    return {
        "nsteps": config["nsteps"],
        "rust_options": rust_options,
        "skip_consistency": bool(config.get("skip_consistency", False)),
    }


def generate_perf_compare_report(summary: dict[str, Any]) -> str:
    """Build a markdown report comparing current benchmark results to the doc."""

    config = summary.get("config", {})
    models = summary.get("models", {})
    config = config if isinstance(config, dict) else {}
    models = models if isinstance(models, dict) else {}
    lines = [
        "# Rust I/O Performance Comparison Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Run Configuration",
        "",
        f"- Final result: `{summary.get('final_result', 'unknown')}`",
        f"- Preset: `{config.get('preset', 'custom')}`",
        f"- Device: `{config.get('device', 'unknown')}`",
        f"- Backends: `{', '.join(config.get('backends', []))}`",
        f"- Models: `{', '.join(config.get('models', []))}`",
        "",
        "## Status",
        "",
        "| Model | Status | Return Code | Consistent | Max Abs Diff | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for model_name, model_payload in models.items():
        if not isinstance(model_payload, dict):
            continue
        consistency = model_payload.get("consistency")
        consistency_skipped = (
            bool(consistency.get("skipped")) if isinstance(consistency, dict) else False
        )
        consistent = (
            consistency.get("all_consistent") if isinstance(consistency, dict) else None
        )
        max_diff = (
            consistency.get("max_abs_diff_global")
            if isinstance(consistency, dict)
            else None
        )
        notes = []
        if consistency_skipped:
            notes.append("Consistency skipped for DLWP CUDA nondeterminism")
        elif model_name == "dlwp" and consistent is False:
            notes.append("DLWP CUDA forecast nondeterminism across executions")
        if not notes and model_payload.get("status") == "passed":
            notes.append("OK")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model_name),
                    str(model_payload.get("status", "unknown")),
                    str(model_payload.get("returncode", "n/a")),
                    "skipped" if consistency_skipped else str(consistent),
                    "n/a"
                    if consistency_skipped
                    else f"{max_diff:.6g}"
                    if isinstance(max_diff, int | float)
                    else "n/a",
                    "; ".join(notes),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Total I/O",
        "",
        "| Model | Backend | Total I/O | I/O Write |",
        "|---|---|---:|---:|",
    ]
    for model_name, model_payload in models.items():
        if not isinstance(model_payload, dict):
            continue
        for backend in ("rust", "py_async", "zarr_sync"):
            total_io = backend_metric(model_payload, backend, "total_io_sec")
            io_write = backend_metric(model_payload, backend, "io_write_sec")
            lines.append(
                f"| {model_name} | {backend} | {format_seconds(total_io)} | "
                f"{format_seconds(io_write)} |"
            )

    lines += [
        "",
        "## Wall Time",
        "",
        "| Model | Backend | Wall Time | Compute Time |",
        "|---|---|---:|---:|",
    ]
    for model_name, model_payload in models.items():
        if not isinstance(model_payload, dict):
            continue
        for backend in ("rust", "py_async", "zarr_sync"):
            wall = backend_metric(model_payload, backend, "total_wall_sec")
            compute = backend_metric(model_payload, backend, "total_compute_sec")
            lines.append(
                f"| {model_name} | {backend} | {format_seconds(wall)} | "
                f"{format_seconds(compute)} |"
            )

    lines += [
        "",
        "## Rust Speedups",
        "",
        "| Model | Metric | Speedup |",
        "|---|---|---:|",
    ]
    for model_name, model_payload in models.items():
        if not isinstance(model_payload, dict):
            continue
        for pair, label, metric in (
            ("rust_vs_py_async", "Rust vs Async total I/O", "total_io_ratio"),
            ("rust_vs_zarr_sync", "Rust vs Sync total I/O", "total_io_ratio"),
            ("rust_vs_py_async", "Rust vs Async wall", "total_wall_ratio"),
            ("rust_vs_zarr_sync", "Rust vs Sync wall", "total_wall_ratio"),
        ):
            current = model_ratio(model_payload, pair, metric)
            lines.append(f"| {model_name} | {label} | {format_ratio(current)} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- `total_io_sec` and I/O ratios are the primary metrics for this report.",
        "- Wall time includes model/data runtime and can move independently of I/O.",
        "- A failed model can still have useful timing data if its backend timings completed.",
        "",
    ]
    return "\n".join(lines)


def run_one_model(
    *,
    model: str,
    output_dir: Path,
    benchmark_script: Path,
    start_time: str,
    nsteps: int,
    device: str,
    backends: list[str],
    seed: int,
    rust_profile: str,
    warmup_steps: int,
    profile_top_n: int,
    earth2studio_root: Path | None,
    preset: str = "custom",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    model_output_dir = output_dir / model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_config = model_config_for(model, preset=preset, default_nsteps=nsteps)
    cmd = build_benchmark_command(
        benchmark_script=benchmark_script,
        model=model,
        model_output_dir=model_output_dir,
        start_time=start_time,
        nsteps=int(model_config["nsteps"]),
        device=device,
        backends=backends,
        seed=seed,
        rust_profile=rust_profile,
        warmup_steps=warmup_steps,
        profile_top_n=profile_top_n,
        earth2studio_root=earth2studio_root,
        rust_options=model_config["rust_options"],
        skip_consistency=bool(model_config.get("skip_consistency")),
    )

    print(f"==> Running Rust I/O benchmark for model: {model}", flush=True)
    log_path = model_output_dir / "benchmark.log"
    if runner is subprocess.run:
        returncode = run_command_streaming(cmd, log_path=log_path)
    else:
        completed = runner(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout = getattr(completed, "stdout", "") or ""
        log_path.write_text(stdout, encoding="utf-8")
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        returncode = int(getattr(completed, "returncode", 1))
    comparison_json = find_latest_comparison_json(model_output_dir)
    if comparison_json is None:
        return {
            "model": model,
            "status": "failed",
            "returncode": returncode,
            "error": f"comparison.json not found under {model_output_dir}",
            "log_path": str(log_path),
            "command": cmd,
            "model_config": model_config,
        }

    summary = summarize_comparison(model, comparison_json, returncode=returncode)
    summary["log_path"] = str(log_path)
    summary["command"] = cmd
    if returncode != 0:
        summary["status"] = "failed"
        summary["error"] = f"benchmark exited with code {returncode}"
    summary["model_config"] = model_config
    return summary


def run_models(
    *,
    models: list[str],
    output_dir: Path,
    benchmark_script: Path,
    start_time: str,
    nsteps: int,
    device: str,
    backends: list[str],
    seed: int,
    rust_profile: str,
    warmup_steps: int,
    profile_top_n: int,
    earth2studio_root: Path | None,
    preset: str = "custom",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_summaries: dict[str, Any] = {}
    for model in models:
        model_summaries[model] = run_one_model(
            model=model,
            output_dir=output_dir,
            benchmark_script=benchmark_script,
            start_time=start_time,
            nsteps=nsteps,
            device=device,
            backends=backends,
            seed=seed,
            rust_profile=rust_profile,
            warmup_steps=warmup_steps,
            profile_top_n=profile_top_n,
            earth2studio_root=earth2studio_root,
            preset=preset,
            runner=runner,
        )

    final_result = (
        "passed"
        if all(item.get("status") == "passed" for item in model_summaries.values())
        else "failed"
    )
    return {
        "created_at": utc_now(),
        "final_result": final_result,
        "config": {
            "benchmark_script": str(benchmark_script),
            "output_dir": str(output_dir),
            "models": models,
            "backends": backends,
            "start_time": start_time,
            "nsteps": nsteps,
            "preset": preset,
            "model_configs": {
                model: model_config_for(model, preset=preset, default_nsteps=nsteps)
                for model in models
            },
            "device": device,
            "seed": seed,
            "rust_profile": rust_profile,
            "warmup_steps": warmup_steps,
            "profile_top_n": profile_top_n,
            "earth2studio_root": str(earth2studio_root)
            if earth2studio_root is not None
            else None,
        },
        "models": model_summaries,
        "artifacts": {
            "output_dir": str(output_dir),
            "summary_json": str(output_dir / "summary.json"),
            "perf_compare_report_md": str(output_dir / "perf_compare_report.md"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Rust I/O benchmark comparisons across E2S models."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", type=parse_csv, default=None)
    parser.add_argument(
        "--preset",
        choices=["custom", "benchmark-report"],
        default="custom",
        help="Use benchmark-report to match docs/benchmark_report_e2s_zarr_io.md.",
    )
    parser.add_argument("--backends", type=parse_csv, default=list(DEFAULT_BACKENDS))
    parser.add_argument("--start-time", default="2024-01-01T00:00:00")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rust-profile", default="default")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-top-n", type=int, default=40)
    parser.add_argument(
        "--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT
    )
    parser.add_argument("--earth2studio-root", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = args.models
    if models is None:
        models = (
            list(REPORT_MODEL_CONFIGS)
            if args.preset == "benchmark-report"
            else list(DEFAULT_MODELS)
        )
    summary = run_models(
        models=models,
        output_dir=args.output_dir,
        benchmark_script=args.benchmark_script,
        start_time=args.start_time,
        nsteps=args.nsteps,
        device=args.device,
        backends=args.backends,
        seed=args.seed,
        rust_profile=args.rust_profile,
        warmup_steps=args.warmup_steps,
        profile_top_n=args.profile_top_n,
        earth2studio_root=args.earth2studio_root,
        preset=args.preset,
    )
    write_json(args.output_dir / "summary.json", summary)
    write_text(
        args.output_dir / "perf_compare_report.md",
        generate_perf_compare_report(summary),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)
    print(f"perf_report={args.output_dir / 'perf_compare_report.md'}", flush=True)
    return 0 if summary["final_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
