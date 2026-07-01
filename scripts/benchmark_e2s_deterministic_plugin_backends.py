#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the integrated e2s-deterministic plugin Zarr backend selector.

This script loads ``plugins/e2s-deterministic/workflow.py`` from the current
workspace and runs it through the normal ``Earth2Workflow.run`` path with
``PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND`` set to ``rust`` or ``python``.

The first phase is untimed warmup. By default, both backends run a short warmup
so model downloads, GFS/cache population, CUDA kernel initialization, and backend
one-time setup do not dominate the timed comparison.

Example inside the local CUDA image:

    docker run --rm --entrypoint bash \
      --device /dev/nvidia0 --device /dev/nvidiactl \
      --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
      -v "$PWD:/workspace" -w /workspace \
      $DOCKER_REGISTRY/$IMAGE_NAME:v0.1.0 \
      -lc 'python scripts/benchmark_e2s_deterministic_plugin_backends.py --nsteps 20'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


PYTHON_DIR = REPO_ROOT / "python"
E2S_TOOLS_DIR = PYTHON_DIR / "e2s_tools"
PLUGIN_PATH = REPO_ROOT / "plugins" / "e2s-deterministic" / "workflow.py"

for path in (SCRIPTS_DIR, PYTHON_DIR, E2S_TOOLS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from plugin_sdk import ExecutionContext, OutputRegistry  # noqa: E402

VALID_BACKENDS = {"rust", "python"}


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_backend_list(raw: str, *, option_name: str) -> list[str]:
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not backends:
        raise ValueError(f"{option_name} must contain at least one backend")
    unknown = sorted(set(backends) - VALID_BACKENDS)
    if unknown:
        raise ValueError(
            f"{option_name} contains unsupported backend(s) {unknown}; "
            f"expected values from {sorted(VALID_BACKENDS)}"
        )
    return backends


def _load_workflow_module(label: str) -> Any:
    module_name = f"e2s_deterministic_benchmark_{label}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load plugin module from {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_inputs(args: argparse.Namespace, *, nsteps: int) -> dict[str, Any]:
    return {
        "forecast_times": [args.start_time],
        "nsteps": nsteps,
        "model_type": args.model_type,
        "data_source": args.data_source,
        "output_format": "zarr",
        "create_plots": False,
    }


def _execution_context(run_dir: Path, *, run_id: str, device: str) -> ExecutionContext:
    device_kind = "gpu" if device.startswith("cuda") else "cpu"
    return ExecutionContext(
        run_id=run_id,
        run_dir=run_dir,
        outputs=OutputRegistry(run_dir),
        resource_profile={"device_kind": device_kind, "device": device},
        services={},
    )


def _add_phase(phases: dict[str, float], key: str, elapsed_sec: float) -> None:
    phases[key] = phases.get(key, 0.0) + elapsed_sec


def _maybe_cuda_synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _wrap_backend_io_methods(io: Any, phases: dict[str, float], *, device: str) -> Any:
    for method_name in ("add_array", "write", "close"):
        method = getattr(io, method_name, None)
        if not callable(method):
            continue

        phase_key = f"io_{method_name}_sec"
        count_key = f"io_{method_name}_count"

        def timed_method(
            *args: Any,
            __method: Any = method,
            __method_name: str = method_name,
            __phase_key: str = phase_key,
            __count_key: str = count_key,
            **kwargs: Any,
        ) -> Any:
            if __method_name == "write":
                _maybe_cuda_synchronize(device)
            start = perf_counter()
            try:
                return __method(*args, **kwargs)
            finally:
                if __method_name == "write":
                    _maybe_cuda_synchronize(device)
                _add_phase(phases, __phase_key, perf_counter() - start)
                phases[__count_key] = phases.get(__count_key, 0.0) + 1.0

        setattr(io, method_name, timed_method)

    return io


def _install_timing_hooks(
    workflow: Any, phases: dict[str, float], *, device: str
) -> Any:
    import earth2studio.run as earth2_run

    original_create_io = workflow.create_io
    original_finalize = workflow.finalize_zarr_output
    original_deterministic = earth2_run.deterministic

    def timed_create_io(self: Any, dataset_path: str) -> Any:
        start = perf_counter()
        try:
            io = original_create_io(dataset_path)
        finally:
            _add_phase(phases, "create_io_sec", perf_counter() - start)
        return _wrap_backend_io_methods(io, phases, device=device)

    def timed_finalize(self: Any, io: Any) -> Any:
        start = perf_counter()
        try:
            return original_finalize(io)
        finally:
            _add_phase(phases, "finalize_zarr_output_sec", perf_counter() - start)

    def timed_deterministic(*args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        try:
            return original_deterministic(*args, **kwargs)
        finally:
            _add_phase(phases, "earth2_deterministic_sec", perf_counter() - start)

    workflow.create_io = types.MethodType(timed_create_io, workflow)
    workflow.finalize_zarr_output = types.MethodType(timed_finalize, workflow)
    earth2_run.deterministic = timed_deterministic

    def restore() -> None:
        workflow.create_io = original_create_io
        workflow.finalize_zarr_output = original_finalize
        earth2_run.deterministic = original_deterministic

    return restore


def _run_plugin_once(
    *,
    backend: str,
    run_dir: Path,
    run_id: str,
    inputs: dict[str, Any],
    device: str,
    timed: bool,
) -> dict[str, Any]:
    os.environ["PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND"] = backend
    module = _load_workflow_module(f"{backend}_{run_id}")
    workflow = module.WORKFLOW()
    ctx = _execution_context(run_dir, run_id=run_id, device=device)
    phases: dict[str, float] = {}
    restore_hooks = _install_timing_hooks(workflow, phases, device=device)

    start = perf_counter()
    try:
        result = workflow.run(inputs, ctx)
    finally:
        elapsed_sec = perf_counter() - start
        restore_hooks()

    cleanup_start = perf_counter()
    try:
        workflow.cleanup()
    except Exception as exc:  # pragma: no cover - cleanup diagnostics only
        print(f"cleanup warning for {backend}/{run_id}: {type(exc).__name__}: {exc}")
    finally:
        phases["cleanup_sec"] = perf_counter() - cleanup_start

    dataset_path = Path(result["dataset_path"])
    return {
        "backend": backend,
        "run_id": run_id,
        "timed": timed,
        "elapsed_sec": elapsed_sec,
        "phases_sec": phases,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
    }


def _speedup_summary(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "rust" not in results or "python" not in results:
        return summary

    rust_times = [float(item["elapsed_sec"]) for item in results["rust"]]
    python_times = [float(item["elapsed_sec"]) for item in results["python"]]
    rust_best = min(rust_times)
    python_best = min(python_times)
    rust_mean = sum(rust_times) / len(rust_times)
    python_mean = sum(python_times) / len(python_times)

    summary["rust_best_sec"] = rust_best
    summary["python_best_sec"] = python_best
    summary["rust_mean_sec"] = rust_mean
    summary["python_mean_sec"] = python_mean
    summary["rust_vs_python_best_speedup"] = (
        python_best / rust_best if rust_best > 0.0 else None
    )
    summary["rust_vs_python_mean_speedup"] = (
        python_mean / rust_mean if rust_mean > 0.0 else None
    )
    summary["python_vs_rust_best_speedup"] = (
        rust_best / python_best if python_best > 0.0 else None
    )
    summary["python_vs_rust_mean_speedup"] = (
        rust_mean / python_mean if python_mean > 0.0 else None
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark current e2s-deterministic plugin Rust/Python Zarr backends."
    )
    parser.add_argument("--start-time", default="2024-01-01T00:00:00")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--warmup-nsteps", type=int, default=1)
    parser.add_argument(
        "--model-type",
        choices=["dlwp", "fcn", "fcn3"],
        default="fcn",
    )
    parser.add_argument("--data-source", choices=["gfs"], default="gfs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backends",
        default="rust,python",
        help="Comma-separated timed backends. Values: rust, python.",
    )
    parser.add_argument(
        "--warmup-backends",
        default="python,rust",
        help=(
            "Comma-separated untimed warmup backends. Values: rust, python. "
            "Use an empty string to skip warmup."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="outputs/e2s_deterministic_plugin_backend_compare",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional output subdirectory name. Defaults to timestamped label.",
    )
    parser.add_argument(
        "--keep-warmup-outputs",
        action="store_true",
        help="Keep warmup forecast.zarr directories instead of removing them after warmup.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.nsteps <= 0:
        raise ValueError("--nsteps must be > 0")
    if args.warmup_nsteps <= 0:
        raise ValueError("--warmup-nsteps must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    timed_backends = _parse_backend_list(args.backends, option_name="--backends")
    warmup_backends = (
        _parse_backend_list(args.warmup_backends, option_name="--warmup-backends")
        if args.warmup_backends.strip()
        else []
    )

    run_label = args.run_label or (
        f"{args.model_type}_{args.start_time.replace(':', '').replace('-', '')}_"
        f"n{args.nsteps}_{_timestamp_label()}"
    )
    run_root = Path(args.output_dir).expanduser().resolve() / run_label
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    warmup_inputs = _build_inputs(args, nsteps=args.warmup_nsteps)
    warmup_results: list[dict[str, Any]] = []
    for backend in warmup_backends:
        print(f"Warmup {backend} backend (nsteps={args.warmup_nsteps}) ...")
        warmup_dir = run_root / "warmup" / backend
        item = _run_plugin_once(
            backend=backend,
            run_dir=warmup_dir,
            run_id=f"warmup-{backend}",
            inputs=warmup_inputs,
            device=args.device,
            timed=False,
        )
        warmup_results.append(item)
        if not args.keep_warmup_outputs:
            shutil.rmtree(warmup_dir, ignore_errors=True)

    timed_inputs = _build_inputs(args, nsteps=args.nsteps)
    timed_results: dict[str, list[dict[str, Any]]] = {
        backend: [] for backend in timed_backends
    }
    for repeat_idx in range(args.repeats):
        for backend in timed_backends:
            print(
                f"Timed {backend} backend repeat {repeat_idx + 1}/{args.repeats} "
                f"(nsteps={args.nsteps}) ..."
            )
            run_dir = run_root / "timed" / backend / f"repeat_{repeat_idx + 1}"
            item = _run_plugin_once(
                backend=backend,
                run_dir=run_dir,
                run_id=f"timed-{backend}-{repeat_idx + 1}",
                inputs=timed_inputs,
                device=args.device,
                timed=True,
            )
            timed_results[backend].append(item)

    report = {
        "config": {
            "plugin_path": str(PLUGIN_PATH),
            "start_time": args.start_time,
            "nsteps": args.nsteps,
            "warmup_nsteps": args.warmup_nsteps,
            "model_type": args.model_type,
            "data_source": args.data_source,
            "device": args.device,
            "backends": timed_backends,
            "warmup_backends": warmup_backends,
            "repeats": args.repeats,
        },
        "warmup": warmup_results,
        "timed": timed_results,
        "speedup": _speedup_summary(timed_results),
        "artifacts": {
            "run_dir": str(run_root),
            "comparison_json": str(run_root / "comparison.json"),
        },
    }

    report_path = run_root / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
