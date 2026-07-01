#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare deterministic performance and output consistency across IO backends.

This script runs the same deterministic Earth2Studio workflow with multiple backends:
1) rust: e2s_zarr_io.E2sZarrIoBackend (optimized Rust extension)
2) py_async: earth2studio.io.AsyncZarrBackend (blocking mode)
3) zarr_sync: earth2studio.io.ZarrBackend (synchronous Python)

It records performance metrics for each backend, writes per-backend profile artifacts,
then reads all output datasets in Python and verifies cross-backend data consistency.
"""

from __future__ import annotations

import argparse
import cProfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import pstats
import random
import re
import sys
from time import perf_counter
from typing import Any


LOGGER = logging.getLogger(__name__)


def _maybe_cuda_synchronize(device: Any) -> None:
    """Synchronize CUDA device for accurate phase timing when needed."""
    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _resolve_torch_device(device_name: str) -> Any:
    """Resolve and validate a torch device string."""
    import torch

    device = torch.device(device_name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device_name}) but torch.cuda.is_available() is False"
        )
    device_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


RUST_PROFILE_DEFAULT = "default"
RUST_PROFILE_FCN_HIGH_THROUGHPUT = "fcn_high_throughput"
RUST_BACKEND_PROFILES: dict[str, dict[str, Any]] = {
    RUST_PROFILE_DEFAULT: {},
    RUST_PROFILE_FCN_HIGH_THROUGHPUT: {
        "max_pool_buffers": 64,
        "hot_slab_buffers": 26,
        "warm_slab_buffers": 26,
        "queue_capacity": 128,
        "pin_pooled_slabs": False,
        "cuda_register_pool_if_available": False,
    },
}


@dataclass(frozen=True)
class StepTiming:
    """Per-step compute and write timings."""

    step: int
    compute_sec: float
    io_write_sec: float
    total_step_sec: float
    io_write_internal_ns: dict[str, Any] | None = None
    io_write_internal_sec: dict[str, float] | None = None


@dataclass(frozen=True)
class BackendOutputPaths:
    """Output paths for one backend run."""

    dataset_path: Path
    metrics_json: Path
    profile_stats: Path
    profile_summary: Path


@dataclass(frozen=True)
class CompareOutputPaths:
    """Output paths for a compare run."""

    run_dir: Path
    backends: dict[str, BackendOutputPaths]
    comparison_json: Path


def safe_percent(part: float, total: float) -> float:
    """Return percentage in [0, 100], returning 0.0 when total is 0."""
    if total <= 0.0:
        return 0.0
    return (part / total) * 100.0


def sanitize_start_time_label(start_time: str) -> str:
    """Convert start_time into a filesystem-safe label."""
    label = re.sub(r"[^0-9A-Za-z]+", "_", start_time).strip("_")
    return label or "start_time"


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_start_time(start_time: str) -> str:
    try:
        datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "start_time must be ISO8601-like, e.g. 2024-01-01T00:00:00"
        ) from exc
    return start_time


def _discover_earth2studio_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        pkg = root / "earth2studio" / "__init__.py"
        if not pkg.exists():
            raise ValueError(f"earth2studio package not found under: {root}")
        return root

    here = Path(__file__).resolve()
    for parent in here.parents:
        pkg = parent / "earth2studio" / "__init__.py"
        if pkg.exists() and (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not discover earth2studio root; pass --earth2studio-root")


def _configure_determinism(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_model(model_type: str) -> Any:
    if model_type == "fcn":
        from earth2studio.models.px import FCN

        package = FCN.load_default_package()
        return FCN.load_model(package)
    if model_type == "stormcast":
        from earth2studio.models.px import StormCast

        package = StormCast.load_default_package()
        return StormCast.load_model(package)
    if model_type == "pangu3":
        from earth2studio.models.px import Pangu3

        package = Pangu3.load_default_package()
        return Pangu3.load_model(package)
    if model_type in ("sfno", "fcn3"):
        # Monkey-patch makani ParamsBase for API compat (img_shape_x_resampled)
        from makani.utils.YParams import ParamsBase

        def _patched_getattr(self, name):
            if name == "img_shape_x_resampled":
                return self.img_shape_x
            if name == "img_shape_y_resampled":
                return self.img_shape_y
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        ParamsBase.__getattr__ = _patched_getattr

        if model_type == "sfno":
            from earth2studio.models.px import SFNO

            package = SFNO.load_default_package()
            return SFNO.load_model(package)
        else:
            from earth2studio.models.px import FCN3

            package = FCN3.load_default_package()
            return FCN3.load_model(package)
    from earth2studio.models.px import DLWP

    package = DLWP.load_default_package()
    return DLWP.load_model(package)


def _normalize_coord_values_for_rust(value: Any) -> Any:
    """Normalize datetime/timedelta coordinate values to int64(ns) for Rust bindings."""
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype(np.int64)
    if arr.dtype.kind == "m":
        return arr.astype("timedelta64[ns]").astype(np.int64)
    if arr.dtype.kind == "O" and arr.size > 0:
        first = arr.flat[0]
        if isinstance(first, datetime):
            return arr.astype("datetime64[ns]").astype(np.int64)
        if isinstance(first, timedelta):
            return arr.astype("timedelta64[ns]").astype(np.int64)
    return value


def _normalize_coord_map_for_rust(coords: dict[str, Any]) -> dict[str, Any]:
    """Normalize full coordinate map to Rust-friendly primitive coordinate values."""
    return {
        key: _normalize_coord_values_for_rust(value) for key, value in coords.items()
    }


def _normalize_array_names(array_names: Any) -> list[str]:
    """Normalize array names into list[str]."""
    if isinstance(array_names, str):
        return [array_names]
    if hasattr(array_names, "tolist"):
        array_names = array_names.tolist()
    if isinstance(array_names, tuple):
        array_names = list(array_names)
    if isinstance(array_names, list):
        return [str(name) for name in array_names]
    return [str(array_names)]


def _normalize_write_arrays_for_rust(
    arrays: Any,
    *,
    prefer_cuda_direct: bool = True,
) -> list[Any]:
    """Normalize write payload arrays for Rust bindings.

    For CUDA tensors, this can keep device-backed tensors on the fast path so
    Rust can consume `__cuda_array_interface__` directly (DMA path).
    """
    import numpy as np

    if isinstance(arrays, tuple):
        arrays = list(arrays)
    if not isinstance(arrays, list):
        arrays = [arrays]

    normalized: list[Any] = []
    for arr in arrays:
        if hasattr(arr, "detach"):
            tensor = arr.detach()
            if prefer_cuda_direct and bool(getattr(tensor, "is_cuda", False)):
                # Ensure pointer math in Rust matches contiguous C-order layout.
                if bool(getattr(tensor, "is_contiguous", lambda: True)()) is False:
                    tensor = tensor.contiguous()
                normalized.append(tensor)
            else:
                normalized.append(tensor.cpu().numpy())
        else:
            normalized.append(np.asarray(arr))
    return normalized


def _build_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build optional Rust-backend tuning kwargs from CLI args."""
    if args.rust_profile not in RUST_BACKEND_PROFILES:
        raise ValueError(
            f"unknown --rust-profile: {args.rust_profile} (expected one of {sorted(RUST_BACKEND_PROFILES)})"
        )
    backend_kwargs: dict[str, Any] = dict(RUST_BACKEND_PROFILES[args.rust_profile])
    optional_fields = (
        "max_pool_buffers",
        "max_pool_bytes",
        "hot_slab_buffers",
        "warm_slab_buffers",
        "queue_capacity",
        "pin_pooled_slabs",
        "cuda_register_pool_if_available",
    )
    for field_name in optional_fields:
        value = getattr(args, field_name, None)
        if value is not None:
            backend_kwargs[field_name] = value
    return backend_kwargs


def _extract_internal_write_timing(backend: Any) -> dict[str, Any] | None:
    """Return backend-reported internal timing dict, when available."""
    getter = getattr(backend, "last_write_timing", None)
    if getter is None:
        return None
    timing = getter()
    if timing is None:
        return None
    if not isinstance(timing, dict):
        return None
    return dict(timing)


def _internal_timing_ns_to_sec(
    timing_ns: dict[str, Any] | None,
) -> dict[str, float] | None:
    """Convert `*_ns` fields to seconds for easier metric comparison."""
    if not timing_ns:
        return None
    out: dict[str, float] = {}
    for key, value in timing_ns.items():
        if key.endswith("_ns") and isinstance(value, int | float):
            out[f"{key[:-3]}_sec"] = float(value) / 1_000_000_000.0
    return out or None


def _accumulate_internal_timing_totals(
    totals_sec: dict[str, float],
    timing_sec: dict[str, float] | None,
) -> None:
    """Accumulate per-step internal timing seconds into backend totals."""
    if timing_sec is None:
        return
    for key, value in timing_sec.items():
        totals_sec[key] = totals_sec.get(key, 0.0) + value


def summarize_metrics(timings: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build IO/compute metric views from timing values."""
    total_wall = timings["total_wall_sec"]
    io_setup = timings["io_setup_sec"]
    io_write = timings["io_write_sec"]
    io_close = timings["io_close_sec"]
    total_io = io_setup + io_write + io_close

    model_to_device = timings["model_to_device_sec"]
    data_fetch = timings["data_fetch_sec"]
    step_compute = timings["step_compute_sec"]
    total_compute = model_to_device + data_fetch + step_compute

    io_metrics: dict[str, float] = {
        "io_setup": io_setup,
        "io_write": io_write,
        "io_close": io_close,
        "total_io": total_io,
        "io_setup_percent": safe_percent(io_setup, total_wall),
        "io_write_percent": safe_percent(io_write, total_wall),
        "total_io_percent": safe_percent(total_io, total_wall),
    }
    if "close_internal" in timings:
        ci = timings["close_internal"]
        io_metrics["close_async_drain_sec"] = ci["async_drain"]
        io_metrics["close_consolidate_sec"] = ci["consolidate"]
        io_metrics["close_teardown_sec"] = ci["teardown"]
    compute_metrics = {
        "model_to_device": model_to_device,
        "data_fetch": data_fetch,
        "step_compute": step_compute,
        "total_compute": total_compute,
        "model_to_device_percent": safe_percent(model_to_device, total_wall),
        "total_compute_percent": safe_percent(total_compute, total_wall),
    }
    return {"io_metrics": io_metrics, "compute_metrics": compute_metrics}


def _resolve_output_paths(
    args: argparse.Namespace, backend_kinds: list[str]
) -> CompareOutputPaths:
    output_root = Path(args.output_dir).expanduser().resolve()
    run_label = f"{sanitize_start_time_label(args.start_time)}_{_timestamp_label()}"
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    backends: dict[str, BackendOutputPaths] = {}
    for kind in backend_kinds:
        backend_dir = run_dir / kind
        backend_dir.mkdir(parents=True, exist_ok=True)
        backends[kind] = BackendOutputPaths(
            dataset_path=backend_dir / "results.zarr",
            metrics_json=backend_dir / "metrics.json",
            profile_stats=backend_dir / "profile.pstats",
            profile_summary=backend_dir / "profile_summary.txt",
        )
    return CompareOutputPaths(
        run_dir=run_dir,
        backends=backends,
        comparison_json=run_dir / "comparison.json",
    )


def _build_total_coords(
    *,
    prognostic: Any,
    start_array: Any,
    nsteps: int,
) -> tuple[OrderedDict[str, Any], list[str]]:
    import numpy as np

    total_coords = prognostic.output_coords(prognostic.input_coords()).copy()
    for key, value in list(total_coords.items()):
        if getattr(value, "shape", None) == (0,):
            del total_coords[key]
    total_coords["time"] = start_array
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    var_names = _normalize_array_names(total_coords.pop("variable"))
    return total_coords, var_names


def _warmup_inference(
    *,
    model_type: str,
    device_name: str,
    prefetched: dict[str, Any],
    warmup_steps: int,
) -> dict[str, float]:
    """Load model, move to device, and run warmup_steps forward passes.

    Warms up CUDA kernels, cuDNN autotuning, and torch.compile so that
    neither backend pays first-run JIT/autotuning costs during the timed
    comparison.  Returns timing info for logging only.
    """
    import torch
    from earth2studio.utils.coords import map_coords

    LOGGER.info(
        "Warming up model (%s, %d steps) on %s …", model_type, warmup_steps, device_name
    )
    t0 = perf_counter()

    device = _resolve_torch_device(device_name)
    _maybe_cuda_synchronize(device)

    model_load_start = perf_counter()
    prognostic = _load_model(model_type)
    model_load_sec = perf_counter() - model_load_start

    to_device_start = perf_counter()
    prognostic = prognostic.to(device)
    _maybe_cuda_synchronize(device)
    to_device_sec = perf_counter() - to_device_start

    prefetched_x = prefetched["x"]
    x = (
        prefetched_x.detach().clone()
        if hasattr(prefetched_x, "detach")
        else prefetched_x
    )
    if hasattr(x, "to") and str(getattr(x, "device", device)) != str(device):
        x = x.to(device)
    coords = prefetched["coords"].copy()

    input_coords = prognostic.input_coords()
    x, coords = map_coords(x, coords, input_coords)
    model_iter = iter(prognostic.create_iterator(x, coords))

    inference_start = perf_counter()
    for _ in range(warmup_steps):
        try:
            x_step, coords_step = next(model_iter)
            _maybe_cuda_synchronize(device)
        except StopIteration:
            break
    inference_sec = perf_counter() - inference_start

    # Explicit cleanup so the warmup model doesn't hold GPU memory during the timed runs.
    del prognostic, model_iter, x, coords
    try:
        del x_step, coords_step
    except NameError:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_sec = perf_counter() - t0
    LOGGER.info(
        "Warmup complete: model_load=%.3fs, to_device=%.3fs, "
        "%d inference steps=%.3fs, total=%.3fs",
        model_load_sec,
        to_device_sec,
        warmup_steps,
        inference_sec,
        total_sec,
    )
    return {
        "warmup_steps": warmup_steps,
        "model_load_sec": model_load_sec,
        "to_device_sec": to_device_sec,
        "inference_sec": inference_sec,
        "total_sec": total_sec,
    }


def _prefetch_initial_data(
    *,
    start_time: str,
    model_type: str,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Fetch initial condition data once to avoid repeated data-source session bugs."""
    from earth2studio.data import GFS, fetch_data
    from earth2studio.utils.time import to_time_array

    _configure_determinism(seed)
    device = _resolve_torch_device(device_name)
    _maybe_cuda_synchronize(device)
    model = _load_model(model_type)
    input_coords = model.input_coords()
    start_array = to_time_array([start_time])
    interp_to = input_coords if hasattr(model, "interp_method") else None
    interp_method = getattr(model, "interp_method", "nearest")

    data_source_start = perf_counter()
    if model_type == "stormcast":
        from earth2studio.data import HRRR

        data = HRRR()
    else:
        data = GFS()
    data_source_init_sec = perf_counter() - data_source_start

    data_fetch_start = perf_counter()
    x, coords = fetch_data(
        source=data,
        time=start_array,
        variable=input_coords["variable"],
        lead_time=input_coords["lead_time"],
        device=device,
        interp_to=interp_to,
        interp_method=interp_method,
    )
    _maybe_cuda_synchronize(device)
    data_fetch_sec = perf_counter() - data_fetch_start

    return {
        "x": x.detach().clone() if hasattr(x, "detach") else x,
        "coords": coords.copy(),
        "device": str(device),
        "data_source_init_sec": data_source_init_sec,
        "data_fetch_sec": data_fetch_sec,
    }


def _run_backend_profiled(
    *,
    backend_kind: str,
    start_time: str,
    nsteps: int,
    model_type: str,
    seed: int,
    device_name: str,
    dataset_path: Path,
    prefetched: dict[str, Any] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one backend and return timing + artifact metadata."""
    from earth2studio.utils.coords import map_coords, split_coords
    from earth2studio.utils.time import to_time_array

    if backend_kind not in {"rust", "py_async", "zarr_sync"}:
        raise ValueError(f"unsupported backend_kind: {backend_kind}")

    _configure_determinism(seed)
    output_coords: OrderedDict[str, Any] = OrderedDict()
    timings: dict[str, float] = {}
    step_timings: list[StepTiming] = []
    rust_backend_kwargs = backend_kwargs or {}

    total_start = perf_counter()

    model_load_start = perf_counter()
    prognostic = _load_model(model_type)
    timings["model_load_sec"] = perf_counter() - model_load_start

    device = _resolve_torch_device(device_name)
    _maybe_cuda_synchronize(device)
    model_to_device_start = perf_counter()
    prognostic = prognostic.to(device)
    _maybe_cuda_synchronize(device)
    timings["model_to_device_sec"] = perf_counter() - model_to_device_start

    input_coords = prognostic.input_coords()
    start_array = to_time_array([start_time])
    if prefetched is None:
        from earth2studio.data import GFS, fetch_data

        data_source_start = perf_counter()
        if model_type == "stormcast":
            from earth2studio.data import HRRR

            data = HRRR()
        else:
            data = GFS()
        timings["data_source_init_sec"] = perf_counter() - data_source_start
        interp_to = input_coords if hasattr(prognostic, "interp_method") else None
        interp_method = getattr(prognostic, "interp_method", "nearest")
        data_fetch_start = perf_counter()
        x, coords = fetch_data(
            source=data,
            time=start_array,
            variable=input_coords["variable"],
            lead_time=input_coords["lead_time"],
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )
        _maybe_cuda_synchronize(device)
        timings["data_fetch_sec"] = perf_counter() - data_fetch_start
    else:
        prefetched_x = prefetched["x"]
        x = (
            prefetched_x.detach().clone()
            if hasattr(prefetched_x, "detach")
            else prefetched_x
        )
        if (
            hasattr(x, "to")
            and hasattr(x, "device")
            and str(getattr(x, "device")) != str(device)
        ):
            x = x.to(device)
        coords = prefetched["coords"].copy()
        timings["data_source_init_sec"] = float(prefetched["data_source_init_sec"])
        timings["data_fetch_sec"] = float(prefetched["data_fetch_sec"])

    total_coords, var_names = _build_total_coords(
        prognostic=prognostic,
        start_array=start_array,
        nsteps=nsteps,
    )
    for key, value in list(total_coords.items()):
        total_coords[key] = output_coords.get(key, value)
    coord_keys = list(total_coords.keys())

    backend: Any
    effective_rust_backend_kwargs: dict[str, Any] = {}
    io_setup_start = perf_counter()
    if backend_kind == "rust":
        import e2s_zarr_io

        rust_total_coords = _normalize_coord_map_for_rust(total_coords)
        effective_rust_backend_kwargs = dict(rust_backend_kwargs)
        if (
            device.type == "cuda"
            and "cuda_register_pool_if_available" not in effective_rust_backend_kwargs
        ):
            # Pinned/registered slabs significantly reduce D2H copy cost for GPU writes.
            effective_rust_backend_kwargs["cuda_register_pool_if_available"] = True
        if "fsync_policy" not in effective_rust_backend_kwargs:
            # Inference data is re-computable; skip fsync for lower I/O latency.
            effective_rust_backend_kwargs["fsync_policy"] = "never"
        backend = e2s_zarr_io.E2sZarrIoBackend(
            file_name=str(dataset_path),
            parallel_coords={
                "time": rust_total_coords["time"],
                "lead_time": rust_total_coords["lead_time"],
            },
            require_host_array_interface=True,
            **effective_rust_backend_kwargs,
        )
        backend.add_array(rust_total_coords, var_names)
    elif backend_kind == "py_async":
        from earth2studio.io import AsyncZarrBackend

        backend = AsyncZarrBackend(
            str(dataset_path),
            parallel_coords={
                "time": total_coords["time"],
                "lead_time": total_coords["lead_time"],
            },
            blocking=True,
        )
        backend.add_array(total_coords, var_names)
    elif backend_kind == "zarr_sync":
        from earth2studio.io import ZarrBackend

        backend = ZarrBackend(
            file_name=str(dataset_path),
            chunks={"time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )
        backend.add_array(total_coords, var_names)
    timings["io_setup_sec"] = perf_counter() - io_setup_start

    iterator_setup_start = perf_counter()
    x, coords = map_coords(x, coords, input_coords)
    model_iter = iter(prognostic.create_iterator(x, coords))
    timings["iterator_setup_sec"] = perf_counter() - iterator_setup_start

    step_compute_total = 0.0
    io_write_total = 0.0
    io_write_internal_totals_sec: dict[str, float] = {}
    run_error: Exception | None = None
    try:
        for step in range(nsteps + 1):
            compute_start = perf_counter()
            try:
                x_step, coords_step = next(model_iter)
            except StopIteration as exc:
                raise RuntimeError(f"model iterator ended before step {step}") from exc
            x_step, coords_step = map_coords(x_step, coords_step, output_coords)
            _maybe_cuda_synchronize(device)
            compute_end = perf_counter()

            io_start = perf_counter()
            step_arrays, step_coords, step_names = split_coords(x_step, coords_step)
            if backend_kind == "rust":
                backend.write(
                    _normalize_write_arrays_for_rust(
                        step_arrays,
                        prefer_cuda_direct=device.type == "cuda",
                    ),
                    _normalize_coord_map_for_rust(step_coords),
                    _normalize_array_names(step_names),
                )
                _maybe_cuda_synchronize(device)
                io_end = perf_counter()
                io_write_internal_ns = _extract_internal_write_timing(backend)
                io_write_internal_sec = _internal_timing_ns_to_sec(io_write_internal_ns)
                _accumulate_internal_timing_totals(
                    io_write_internal_totals_sec,
                    io_write_internal_sec,
                )
            else:
                backend.write(
                    step_arrays, step_coords, _normalize_array_names(step_names)
                )
                _maybe_cuda_synchronize(device)
                io_end = perf_counter()
                io_write_internal_ns = None
                io_write_internal_sec = None

            compute_sec = compute_end - compute_start
            io_write_sec = io_end - io_start
            step_compute_total += compute_sec
            io_write_total += io_write_sec
            step_timings.append(
                StepTiming(
                    step=step,
                    compute_sec=compute_sec,
                    io_write_sec=io_write_sec,
                    total_step_sec=io_end - compute_start,
                    io_write_internal_ns=io_write_internal_ns,
                    io_write_internal_sec=io_write_internal_sec,
                )
            )
    except Exception as exc:
        run_error = exc
        raise
    finally:
        timings["step_compute_sec"] = step_compute_total
        timings["io_write_sec"] = io_write_total
        io_close_start = perf_counter()
        close_timing_dict = None
        try:
            if hasattr(backend, "close"):
                close_timing_dict = backend.close()
        except Exception:
            if run_error is None:
                raise
            LOGGER.exception("backend.close() failed while handling prior run failure")
        finally:
            timings["io_close_sec"] = perf_counter() - io_close_start
            if close_timing_dict is not None:
                timings["close_internal"] = {
                    k.removesuffix("_ns"): v / 1e9 for k, v in close_timing_dict.items()
                }

    wall_sec = perf_counter() - total_start
    if prefetched is not None:
        # Attribute shared prefetch cost to each backend for fair end-to-end comparison.
        wall_sec += float(prefetched["data_source_init_sec"]) + float(
            prefetched["data_fetch_sec"]
        )
    timings["total_wall_sec"] = wall_sec
    summary = summarize_metrics(timings)
    return {
        "config": {
            "backend": backend_kind,
            "start_time": start_time,
            "nsteps": nsteps,
            "device": str(device),
            "model_type": model_type,
            "seed": seed,
            "backend_kwargs": effective_rust_backend_kwargs
            if backend_kind == "rust"
            else {},
            "profiler": "cProfile + phase_timers",
        },
        "dataset_path": str(dataset_path),
        "timings_sec": timings,
        **summary,
        "io_write_internal_totals_sec": io_write_internal_totals_sec,
        "step_metrics": [asdict(metric) for metric in step_timings],
        "array_names": var_names,
        "coord_keys": coord_keys,
    }


def _write_profile_summary(
    profiler: cProfile.Profile,
    profile_stats: Path,
    profile_summary: Path,
    top_n: int,
) -> None:
    profiler.dump_stats(str(profile_stats))
    with profile_summary.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.sort_stats("cumtime")
        stats.print_stats(top_n)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_backend_with_profiler(
    *,
    backend_kind: str,
    start_time: str,
    nsteps: int,
    model_type: str,
    seed: int,
    device_name: str,
    paths: BackendOutputPaths,
    profile_top_n: int,
    prefetched: dict[str, Any] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        payload = _run_backend_profiled(
            backend_kind=backend_kind,
            start_time=start_time,
            nsteps=nsteps,
            model_type=model_type,
            seed=seed,
            device_name=device_name,
            dataset_path=paths.dataset_path,
            prefetched=prefetched,
            backend_kwargs=backend_kwargs,
        )
    finally:
        profiler.disable()

    _write_profile_summary(
        profiler=profiler,
        profile_stats=paths.profile_stats,
        profile_summary=paths.profile_summary,
        top_n=profile_top_n,
    )
    payload["profile_stats"] = str(paths.profile_stats)
    payload["profile_summary"] = str(paths.profile_summary)
    _write_json(paths.metrics_json, payload)
    return payload


def _normalize_coord_array_for_compare(values: Any) -> Any:
    """Normalize coordinate arrays for backend-agnostic comparison."""
    import numpy as np

    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype(np.int64)
    if arr.dtype.kind == "m":
        return arr.astype("timedelta64[ns]").astype(np.int64)
    if arr.dtype.kind == "O" and arr.size > 0:
        first = arr.flat[0]
        if isinstance(first, datetime):
            return arr.astype("datetime64[ns]").astype(np.int64)
        if isinstance(first, timedelta):
            return arr.astype("timedelta64[ns]").astype(np.int64)
    return arr


def _compare_numeric_arrays(
    lhs: Any,
    rhs: Any,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float | None]:
    """Compare arrays and return (allclose, max_abs_diff)."""
    import numpy as np

    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if lhs_arr.shape != rhs_arr.shape:
        return False, None
    allclose = bool(np.allclose(lhs_arr, rhs_arr, rtol=rtol, atol=atol, equal_nan=True))
    if lhs_arr.size == 0:
        return allclose, 0.0
    max_abs_diff = float(
        np.max(np.abs(lhs_arr.astype(np.float64) - rhs_arr.astype(np.float64)))
    )
    return allclose, max_abs_diff


def _compare_datasets(
    *,
    ref_dataset_path: Path,
    ref_label: str,
    other_dataset_path: Path,
    other_label: str,
    array_names: list[str],
    coord_keys: list[str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Read both datasets in Python and verify variable/coord consistency."""
    import numpy as np
    import zarr

    ref_root = zarr.open_group(str(ref_dataset_path), mode="r")
    other_root = zarr.open_group(str(other_dataset_path), mode="r")

    variable_checks: list[dict[str, Any]] = []
    variable_mismatches: list[str] = []
    global_max_abs_diff = 0.0
    for name in array_names:
        if name not in ref_root:
            variable_mismatches.append(f"{name}: missing in {ref_label} dataset")
            variable_checks.append(
                {"name": name, "consistent": False, "reason": f"missing_in_{ref_label}"}
            )
            continue
        if name not in other_root:
            variable_mismatches.append(f"{name}: missing in {other_label} dataset")
            variable_checks.append(
                {
                    "name": name,
                    "consistent": False,
                    "reason": f"missing_in_{other_label}",
                }
            )
            continue

        ref_values = np.asarray(ref_root[name][:])
        other_values = np.asarray(other_root[name][:])
        allclose, max_abs_diff = _compare_numeric_arrays(
            ref_values, other_values, rtol=rtol, atol=atol
        )
        if max_abs_diff is not None:
            global_max_abs_diff = max(global_max_abs_diff, max_abs_diff)
        if not allclose:
            variable_mismatches.append(
                f"{name}: shape {ref_label}={ref_values.shape} {other_label}={other_values.shape}, max_abs_diff={max_abs_diff}"
            )
        variable_checks.append(
            {
                "name": name,
                "consistent": allclose,
                f"shape_{ref_label}": list(ref_values.shape),
                f"shape_{other_label}": list(other_values.shape),
                f"dtype_{ref_label}": str(ref_values.dtype),
                f"dtype_{other_label}": str(other_values.dtype),
                "max_abs_diff": max_abs_diff,
            }
        )

    coord_checks: list[dict[str, Any]] = []
    coord_mismatches: list[str] = []
    for key in coord_keys:
        if key not in ref_root:
            coord_mismatches.append(f"{key}: missing in {ref_label} dataset")
            coord_checks.append(
                {"name": key, "consistent": False, "reason": f"missing_in_{ref_label}"}
            )
            continue
        if key not in other_root:
            coord_mismatches.append(f"{key}: missing in {other_label} dataset")
            coord_checks.append(
                {
                    "name": key,
                    "consistent": False,
                    "reason": f"missing_in_{other_label}",
                }
            )
            continue

        ref_coord = _normalize_coord_array_for_compare(np.asarray(ref_root[key][:]))
        other_coord = _normalize_coord_array_for_compare(np.asarray(other_root[key][:]))
        equal = ref_coord.shape == other_coord.shape and bool(
            np.array_equal(ref_coord, other_coord)
        )
        if not equal:
            coord_mismatches.append(f"{key}: coordinate values differ")
        coord_checks.append(
            {
                "name": key,
                "consistent": equal,
                f"shape_{ref_label}": list(ref_coord.shape),
                f"shape_{other_label}": list(other_coord.shape),
                f"dtype_{ref_label}": str(np.asarray(ref_coord).dtype),
                f"dtype_{other_label}": str(np.asarray(other_coord).dtype),
            }
        )

    variables_consistent = len(variable_mismatches) == 0
    coords_consistent = len(coord_mismatches) == 0
    return {
        "variables_consistent": variables_consistent,
        "coords_consistent": coords_consistent,
        "all_consistent": variables_consistent and coords_consistent,
        "variables_checked": len(array_names),
        "coords_checked": len(coord_keys),
        "max_abs_diff_global": global_max_abs_diff,
        "variable_mismatches": variable_mismatches,
        "coord_mismatches": coord_mismatches,
        "variable_checks": variable_checks,
        "coord_checks": coord_checks,
    }


def _build_performance_comparison(
    payloads: dict[str, dict[str, Any]],
    reference: str = "rust",
) -> dict[str, Any]:
    """Build performance comparison: reference vs each other backend."""
    ref = payloads.get(reference)
    if ref is None:
        return {}
    ref_wall = ref["timings_sec"]["total_wall_sec"]
    ref_io = ref["io_metrics"]["total_io"]
    ref_write = ref["io_metrics"]["io_write"]

    comparisons: dict[str, Any] = {}
    for name, payload in payloads.items():
        if name == reference:
            continue
        other_wall = payload["timings_sec"]["total_wall_sec"]
        other_io = payload["io_metrics"]["total_io"]
        other_write = payload["io_metrics"]["io_write"]
        pair_key = f"{reference}_vs_{name}"
        comparisons[pair_key] = {
            "total_wall_ratio": other_wall / ref_wall if ref_wall > 0 else None,
            "total_io_ratio": other_io / ref_io if ref_io > 0 else None,
            "io_write_ratio": other_write / ref_write if ref_write > 0 else None,
            "total_wall_delta_sec": ref_wall - other_wall,
            "total_io_delta_sec": ref_io - other_io,
            "io_write_delta_sec": ref_write - other_write,
        }
    return comparisons


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic workflow with Rust and py_async backends, "
            "compare performance, and verify data consistency."
        )
    )
    parser.add_argument("--start-time", required=True, type=_validate_start_time)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument(
        "--model-type",
        choices=["fcn", "dlwp", "stormcast", "pangu3", "sfno", "fcn3"],
        default="fcn",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, e.g. 'cpu', 'cuda', or 'cuda:0' (default: cpu)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--rust-profile",
        choices=sorted(RUST_BACKEND_PROFILES.keys()),
        default=RUST_PROFILE_DEFAULT,
        help=(
            "Named Rust backend tuning preset. "
            "'fcn_high_throughput' expands pool/queue defaults for FCN-like CPU workloads. "
            "Explicit pool flags override this preset."
        ),
    )
    parser.add_argument(
        "--earth2studio-root",
        default=None,
        help="Optional Earth2Studio repo root path; auto-discovered when omitted.",
    )
    parser.add_argument("--output-dir", default="outputs/deterministic_backend_compare")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--profile-top-n", type=int, default=40)
    parser.add_argument(
        "--max-pool-buffers",
        type=int,
        default=None,
        help="Optional Rust-backend override for max pooled buffers.",
    )
    parser.add_argument(
        "--hot-slab-buffers",
        type=int,
        default=None,
        help="Optional Rust-backend override for hot slab pooled buffers.",
    )
    parser.add_argument(
        "--warm-slab-buffers",
        type=int,
        default=None,
        help="Optional Rust-backend override for warm slab pooled buffers.",
    )
    parser.add_argument(
        "--max-pool-bytes",
        type=int,
        default=None,
        help="Optional Rust-backend override for max pooled memory budget in bytes.",
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=None,
        help="Optional Rust-backend override for scheduler queue capacity.",
    )
    parser.add_argument(
        "--pin-pooled-slabs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional Rust-backend override to enable/disable pooled slab pinning.",
    )
    parser.add_argument(
        "--cuda-register-pool-if-available",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional Rust-backend override to enable/disable pooled slab CUDA registration.",
    )
    parser.add_argument(
        "--backends",
        default="rust,py_async,zarr_sync",
        help=(
            "Comma-separated list of backends to run. "
            "Available: rust, py_async, zarr_sync. (default: all three)"
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help=(
            "Number of model forward steps to run before the timed comparison. "
            "Warms CUDA kernels and cuDNN autotuning so neither backend pays "
            "first-run JIT costs. Set to 0 to disable. (default: 3)"
        ),
    )
    parser.add_argument(
        "--skip-consistency",
        action="store_true",
        help="Skip cross-backend dataset consistency checks and report timings only.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    if args.nsteps <= 0:
        raise ValueError("--nsteps must be > 0")
    if args.max_pool_buffers is not None and args.max_pool_buffers <= 0:
        raise ValueError("--max-pool-buffers must be > 0")
    if args.hot_slab_buffers is not None and args.hot_slab_buffers <= 0:
        raise ValueError("--hot-slab-buffers must be > 0")
    if args.warm_slab_buffers is not None and args.warm_slab_buffers <= 0:
        raise ValueError("--warm-slab-buffers must be > 0")
    if args.queue_capacity is not None and args.queue_capacity <= 0:
        raise ValueError("--queue-capacity must be > 0")

    valid_backends = {"rust", "py_async", "zarr_sync"}
    backend_kinds = [b.strip() for b in args.backends.split(",")]
    for kind in backend_kinds:
        if kind not in valid_backends:
            raise ValueError(
                f"unknown backend '{kind}'; available: {sorted(valid_backends)}"
            )

    earth2studio_root = _discover_earth2studio_root(args.earth2studio_root)
    root_str = str(earth2studio_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    paths = _resolve_output_paths(args, backend_kinds)
    backend_kwargs = _build_backend_kwargs(args)
    prefetched = _prefetch_initial_data(
        start_time=args.start_time,
        model_type=args.model_type,
        seed=args.seed,
        device_name=args.device,
    )
    warmup_timing: dict[str, Any] | None = None
    if args.warmup_steps > 0:
        warmup_timing = _warmup_inference(
            model_type=args.model_type,
            device_name=args.device,
            prefetched=prefetched,
            warmup_steps=args.warmup_steps,
        )

    payloads: dict[str, dict[str, Any]] = {}
    for kind in backend_kinds:
        payloads[kind] = _run_backend_with_profiler(
            backend_kind=kind,
            start_time=args.start_time,
            nsteps=args.nsteps,
            model_type=args.model_type,
            seed=args.seed,
            device_name=args.device,
            paths=paths.backends[kind],
            profile_top_n=args.profile_top_n,
            prefetched=prefetched,
            backend_kwargs=backend_kwargs if kind == "rust" else None,
        )

    # Use first backend as reference for comparisons.
    reference = backend_kinds[0]
    ref_payload = payloads[reference]

    consistency: dict[str, Any] = {}
    if args.skip_consistency:
        LOGGER.info("Skipping cross-backend dataset consistency checks")
    else:
        # Pairwise consistency checks against reference.
        for kind in backend_kinds:
            if kind == reference:
                continue
            pair_key = f"{reference}_vs_{kind}"
            consistency[pair_key] = _compare_datasets(
                ref_dataset_path=paths.backends[reference].dataset_path,
                ref_label=reference,
                other_dataset_path=paths.backends[kind].dataset_path,
                other_label=kind,
                array_names=ref_payload["array_names"],
                coord_keys=ref_payload["coord_keys"],
                rtol=args.rtol,
                atol=args.atol,
            )

    performance: dict[str, Any] = {}
    for kind in backend_kinds:
        performance[kind] = {
            "timings_sec": payloads[kind]["timings_sec"],
            "io_metrics": payloads[kind]["io_metrics"],
            "compute_metrics": payloads[kind]["compute_metrics"],
        }
    performance["comparisons"] = _build_performance_comparison(payloads, reference)

    # Build artifacts dict.
    artifacts: dict[str, str] = {"run_dir": str(paths.run_dir)}
    for kind in backend_kinds:
        bp = paths.backends[kind]
        artifacts[f"{kind}_dataset_path"] = str(bp.dataset_path)
        artifacts[f"{kind}_metrics_json"] = str(bp.metrics_json)
        artifacts[f"{kind}_profile_stats"] = str(bp.profile_stats)
        artifacts[f"{kind}_profile_summary"] = str(bp.profile_summary)

    rust_config = (
        payloads["rust"]["config"]["backend_kwargs"] if "rust" in payloads else {}
    )
    output_payload = {
        "config": {
            "start_time": args.start_time,
            "nsteps": args.nsteps,
            "model_type": args.model_type,
            "device": ref_payload["config"]["device"],
            "seed": args.seed,
            "backends": backend_kinds,
            "rust_profile": args.rust_profile,
            "rust_backend_kwargs": rust_config,
            "rtol": args.rtol,
            "atol": args.atol,
            "warmup_steps": args.warmup_steps,
            "skip_consistency": args.skip_consistency,
        },
        "warmup": warmup_timing,
        "artifacts": artifacts,
        "performance": performance,
        "consistency": consistency,
    }
    _write_json(paths.comparison_json, output_payload)

    LOGGER.info("Comparison complete")
    LOGGER.info("Output JSON: %s", paths.comparison_json)
    wall_parts = ", ".join(
        f"{k}={performance[k]['timings_sec']['total_wall_sec']:.6f}"
        for k in backend_kinds
    )
    io_parts = ", ".join(
        f"{k}={performance[k]['io_metrics']['total_io']:.6f}" for k in backend_kinds
    )
    LOGGER.info("Total wall (sec): %s", wall_parts)
    LOGGER.info("Total IO (sec): %s", io_parts)
    for pair_key, check in consistency.items():
        LOGGER.info(
            "Consistency [%s]: all_consistent=%s, max_abs_diff_global=%.9f",
            pair_key,
            check["all_consistent"],
            check["max_abs_diff_global"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
