# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Runtime injects the scripts directory into sys.path for plugin execution.
from plugin_sdk import (  # pyright: ignore[reportMissingImports]
    ExecutionContext,
    PostprocessContext,
    PostprocessOutcome,
    PrepareContext,
    PriorResult,
    RawRequest,
    cleanup_earth2_runtime_resources,
    cleanup_python_and_torch_runtime,
)

_FANOUT_RUST_ZARR_MAX_POOL_BYTES = 2 * 1024 * 1024 * 1024
_FANOUT_RUST_ZARR_MAX_INFLIGHT_TRANSIENT_BYTES = 4 * 1024 * 1024 * 1024
_FANOUT_STAGE_ID = "fanout"


@dataclass
class EnsembleFanoutInput:
    model: str
    start_time: str
    nsteps: int
    nensemble: int
    batch_size: int = 1
    max_in_flight: int = 1
    perturbation: str = "spherical_gaussian"
    noise_amplitude: float = 0.05
    seed_base: int = 1000
    perturbation_materialization_mode: str = "scheduled_gpu"
    batch_index: int = 0
    batch_member_ids: list[int] | None = None
    prepared_state_path: str | None = None


@dataclass
class EnsembleFanoutBatchOutput:
    model: str
    start_time: str
    nsteps: int
    nensemble: int
    batch_index: int
    batch_member_ids: list[int]
    perturbation: str
    noise_amplitude: float
    dataset_path: str
    prepared_state_path: str
    note: str


@dataclass
class EnsembleFanoutPostprocessSummary:
    skipped_count: int
    aggregated_count: int
    partial_aggregation: bool


@dataclass
class EnsembleFanoutOutput:
    model: Any
    start_time: Any
    nsteps: Any
    nensemble: Any
    batch_size: Any
    dataset_path: str
    batch_dataset_paths: list[str]
    aggregation_summary: dict[str, Any]
    postprocess_summary: EnsembleFanoutPostprocessSummary
    note: str


def _load_model(model_name: str) -> tuple[Any, Any]:
    from earth2studio.models.px import FCN

    normalized = str(model_name).strip().lower()
    if normalized == "fcn":
        package = FCN.load_default_package()
        return package, FCN.load_model(package)

    raise ValueError("earth2-ensemble-fanout supports only model='fcn'")


def _build_perturbation(name: str, noise_amplitude: float):
    normalized = _normalize_perturbation(name)
    if normalized == "gaussian":
        from earth2studio.perturbation import Gaussian

        return Gaussian(noise_amplitude=noise_amplitude), normalized
    if normalized == "brown":
        from earth2studio.perturbation import Brown

        return Brown(noise_amplitude=noise_amplitude), normalized
    if normalized == "spherical_gaussian":
        from earth2studio.perturbation import SphericalGaussian

        return SphericalGaussian(noise_amplitude=noise_amplitude), normalized
    raise ValueError(
        "perturbation must be 'gaussian', 'brown', or 'spherical_gaussian'"
    )


def _normalize_perturbation(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"", "spherical_gaussian", "spherical-gaussian"}:
        return "spherical_gaussian"
    if normalized in {"gaussian", "brown"}:
        return normalized
    raise ValueError(
        "perturbation must be 'gaussian', 'brown', or 'spherical_gaussian'"
    )


def _select_device(
    ctx: ExecutionContext | Mapping[str, Any], *, force_cpu: bool = False
):
    import torch

    if force_cpu:
        return torch.device("cpu"), "cpu"

    resource_profile = _context_resource_profile(ctx)
    runtime_device = (
        "gpu" if int(resource_profile.get("gpus_required") or 1) > 0 else "cpu"
    )
    if runtime_device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    return torch.device("cpu"), "cpu"


def _raise_with_captured_output(label: str, captured_output: str, exc: Exception):
    details = captured_output.strip()
    exc_details = f"{type(exc).__name__}: {exc}"
    if details:
        raise RuntimeError(f"{label} failed: {exc_details}\n{details}") from exc
    raise RuntimeError(f"{label} failed: {exc_details}") from exc


def _prepared_state_dir(run_dir: Path) -> Path:
    return run_dir / "prepared-initial-conditions"


def _context_run_id(
    ctx: PrepareContext | ExecutionContext | PostprocessContext | Mapping[str, Any],
) -> str:
    if isinstance(ctx, Mapping):
        return str(ctx.get("run_id") or "")
    return str(getattr(ctx, "run_id", "") or "")


def _context_run_dir(
    ctx: PrepareContext | ExecutionContext | PostprocessContext | Mapping[str, Any],
) -> Path:
    if isinstance(ctx, Mapping):
        run_dir = ctx.get("run_dir")
        if run_dir is not None:
            return Path(run_dir)
    else:
        run_dir = getattr(ctx, "run_dir", None)
        if run_dir is not None:
            return Path(run_dir)
    return Path(os.environ["DEFAULT_OUTPUT_DIR"]) / _context_run_id(ctx)


def _context_resource_profile(
    ctx: PrepareContext | ExecutionContext | PostprocessContext | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(ctx, Mapping):
        return dict(ctx.get("resource_profile") or {})
    return dict(getattr(ctx, "resource_profile", None) or {})


def _context_fanout_item(ctx: ExecutionContext | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(ctx, Mapping):
        return dict(ctx.get("fanout_item") or {})
    return dict(getattr(ctx, "fanout_item", None) or {})


def _copy_store(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _create_child_zarr_backend(
    dataset_path: Path, *, ensemble_chunk_size: int | None = None
):
    from e2s_workflow import (  # pyright: ignore[reportMissingImports]
        _selected_zarr_backend,
        create_zarr_backend,
    )

    selected_backend = _selected_zarr_backend()
    ensemble_chunk = 1
    if selected_backend == "python" and ensemble_chunk_size is not None:
        ensemble_chunk = max(1, int(ensemble_chunk_size))
    chunks = {"ensemble": ensemble_chunk, "time": 1, "lead_time": 1}
    rust_kwargs = (
        {}
        if selected_backend == "python"
        else {
            "zarr_format": "v3",
            "max_pool_bytes": _FANOUT_RUST_ZARR_MAX_POOL_BYTES,
            "max_inflight_transient_bytes": (
                _FANOUT_RUST_ZARR_MAX_INFLIGHT_TRANSIENT_BYTES
            ),
        }
    )
    return create_zarr_backend(
        str(dataset_path),
        chunks=chunks,
        backend_kwargs={"overwrite": True},
        **rust_kwargs,
    )


def _finalize_child_zarr_backend(io_backend) -> None:
    finalizer = getattr(io_backend, "finalize", None)
    if callable(finalizer):
        finalizer()
        return

    closer = getattr(io_backend, "close", None)
    if callable(closer):
        closer()


def _configure_ensemble_io(
    io_backend, model, *, start_time: str, nsteps: int, member_ids: list[int]
):
    import numpy as np
    from earth2studio.utils.time import to_time_array

    total_coords = model.output_coords(model.input_coords()).copy()
    if "batch" in total_coords:
        del total_coords["batch"]

    total_coords["time"] = to_time_array([start_time])
    lead_time = model.output_coords(model.input_coords())["lead_time"]
    total_coords["lead_time"] = np.asarray(
        [lead_time * i for i in range(nsteps + 1)]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    total_coords = {"ensemble": np.asarray(member_ids)} | total_coords
    variables_to_save = total_coords.pop("variable")
    io_backend.add_array(total_coords, variables_to_save)


def _normalize_parameters(params: Mapping[str, Any]) -> EnsembleFanoutInput:
    model = str(params.get("model") or "").strip().lower()
    if model != "fcn":
        raise ValueError("earth2-ensemble-fanout supports only model='fcn'")

    start_time = str(params.get("start_time") or "").strip()
    if not start_time:
        raise ValueError("start_time must be a non-empty string")

    nsteps = int(params.get("nsteps"))
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")

    nensemble = int(params.get("nensemble"))
    if nensemble < 1:
        raise ValueError("nensemble must be >= 1")

    batch_size = int(params.get("batch_size") or nensemble)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    max_in_flight = int(params.get("max_in_flight") or 1)
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")

    perturbation = _normalize_perturbation(
        str(params.get("perturbation") or "spherical_gaussian")
    )
    noise_amplitude = float(params.get("noise_amplitude") or 0.05)
    if noise_amplitude <= 0:
        raise ValueError("noise_amplitude must be > 0")

    seed_base = int(params.get("seed_base") or 1000)
    materialization = (
        str(params.get("perturbation_materialization_mode") or "scheduled_gpu")
        .strip()
        .lower()
    )
    if materialization not in {"scheduled_gpu", "prepare_cpu"}:
        raise ValueError(
            "perturbation_materialization_mode must be 'scheduled_gpu' or 'prepare_cpu'"
        )

    return EnsembleFanoutInput(
        model=model,
        start_time=start_time,
        nsteps=nsteps,
        nensemble=nensemble,
        batch_size=min(batch_size, nensemble),
        max_in_flight=min(max_in_flight, nensemble),
        perturbation=perturbation,
        noise_amplitude=noise_amplitude,
        seed_base=seed_base,
        perturbation_materialization_mode=materialization,
    )


def _resource_profile() -> dict[str, object]:
    return {
        "executor_class": "earth2-gpu",
        "gpus_required": 1,
        "memory_mb": 4096,
        "tags": ["earth2", "gpu"],
    }


def _batch_member_ids(
    batch_index: int, *, nensemble: int, batch_size: int
) -> list[int]:
    batch_offset = batch_index * batch_size
    return list(range(batch_offset, min(batch_offset + batch_size, nensemble)))


def _build_fanout_batches(parameters: EnsembleFanoutInput) -> list[dict[str, Any]]:
    nensemble = int(parameters.nensemble)
    batch_size = int(parameters.batch_size)
    batches: list[dict[str, Any]] = []
    for batch_index, batch_offset in enumerate(range(0, nensemble, batch_size)):
        member_ids = list(
            range(batch_offset, min(batch_offset + batch_size, nensemble))
        )
        batches.append(
            {
                "batch_index": batch_index,
                "batch_member_ids": member_ids,
                "perturbation": parameters.perturbation,
            }
        )
    return batches


def _build_batch_initial_conditions(
    x0,
    coords0,
    prognostic_ic,
    member_ids: list[int],
):
    import numpy as np
    from earth2studio.utils.coords import map_coords

    batch_x = x0.unsqueeze(0).repeat(len(member_ids), *([1] * x0.ndim))
    batch_coords = {"ensemble": np.asarray(member_ids)} | coords0.copy()
    return map_coords(batch_x, batch_coords, prognostic_ic)


def _load_and_perturb_batch_initial_conditions(
    inputs: EnsembleFanoutInput,
    ctx: ExecutionContext | Mapping[str, Any],
    *,
    batch_index: int,
    member_ids: list[int],
    model_resource: tuple[Any, Any] | None = None,
):
    import torch
    from earth2studio.data import GFS, fetch_data
    from earth2studio.utils.time import to_time_array

    package = None
    model = None
    data = None
    x0 = None
    coords0 = None
    batch_x = None
    batch_coords = None
    prior_x = None
    prior_coords = None
    perturbation = None
    actual_device_kind = "cpu"
    owns_model = model_resource is None
    try:
        device, actual_device_kind = _select_device(ctx)
        if model_resource is None:
            package, model = _load_model(inputs.model)
        else:
            package, model = model_resource
        model = model.to(device)
        prognostic_ic = model.input_coords()
        time = to_time_array([inputs.start_time])
        if hasattr(model, "interp_method"):
            interp_to = prognostic_ic
            interp_method = model.interp_method
        else:
            interp_to = None
            interp_method = "nearest"

        data = GFS()
        x0, coords0 = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic["variable"],
            lead_time=prognostic_ic["lead_time"],
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )

        perturbation, perturbation_name = _build_perturbation(
            inputs.perturbation,
            inputs.noise_amplitude,
        )

        torch.manual_seed(int(inputs.seed_base))
        for prior_batch_index in range(batch_index):
            prior_member_ids = _batch_member_ids(
                prior_batch_index,
                nensemble=int(inputs.nensemble),
                batch_size=int(inputs.batch_size),
            )
            prior_x, prior_coords = _build_batch_initial_conditions(
                x0,
                coords0,
                prognostic_ic,
                prior_member_ids,
            )
            perturbation(prior_x, prior_coords)

        batch_x, batch_coords = _build_batch_initial_conditions(
            x0,
            coords0,
            prognostic_ic,
            member_ids,
        )
        batch_x, batch_coords = perturbation(batch_x, batch_coords)

        return (
            package,
            model,
            data,
            batch_x,
            batch_coords,
            actual_device_kind,
            perturbation_name,
        )
    except Exception:
        cleanup_earth2_runtime_resources(data, model if owns_model else None)
        cleanup_python_and_torch_runtime(device=actual_device_kind)
        raise
    finally:
        x0 = None
        coords0 = None
        prior_x = None
        prior_coords = None
        perturbation = None


def _prepare_batch_initial_conditions(
    parameters: EnsembleFanoutInput, _run_dir: Path
) -> list[dict[str, Any]]:
    return _build_fanout_batches(parameters)


def prepare_ensemble_fanout_request(
    request: RawRequest, ctx: PrepareContext
) -> dict[str, Any]:
    normalized = _normalize_parameters(dict(request.raw_fields))
    if normalized.perturbation_materialization_mode == "prepare_cpu":
        captured_output = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured_output),
                contextlib.redirect_stderr(captured_output),
            ):
                materialized = _materialize_prepared_batch_states(
                    normalized,
                    ctx,
                    force_cpu=True,
                )
        except Exception as exc:
            _raise_with_captured_output(
                "earth2-ensemble-fanout prepare CPU materialization",
                captured_output.getvalue(),
                exc,
            )
        return {
            "operation": "run",
            "parameters": dict(normalized.__dict__),
            "fanout_profile": materialized["fanout_profile"],
            "fanout_items": materialized["fanout_items"],
            "next_stage_id": _FANOUT_STAGE_ID,
        }

    return {
        "operation": "materialize_perturbations",
        "parameters": dict(normalized.__dict__),
        "resource_profile": _resource_profile(),
    }


def _fanout_items_from_prepared_batches(
    inputs: EnsembleFanoutInput, prepared_batches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "item_index": int(batch["batch_index"]),
            "operation": "run",
            "parameters": {
                **inputs.__dict__,
                "batch_index": int(batch["batch_index"]),
                "batch_member_ids": [
                    int(member) for member in batch["batch_member_ids"]
                ],
                "prepared_state_path": str(batch["prepared_state_path"]),
            },
        }
        for batch in prepared_batches
    ]


def _materialize_prepared_batch_states(
    inputs: EnsembleFanoutInput,
    ctx: PrepareContext | ExecutionContext | Mapping[str, Any],
    *,
    force_cpu: bool = False,
    model_resource: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from earth2studio.data import GFS, fetch_data
    from earth2studio.utils.time import to_time_array

    run_dir = _context_run_dir(ctx)
    state_dir = _prepared_state_dir(run_dir)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    _package = None
    model = None
    data = None
    x0 = None
    coords0 = None
    batch_x = None
    batch_coords = None
    perturbation = None
    actual_device_kind = "cpu"
    prepared_batches: list[dict[str, Any]] = []
    owns_model = model_resource is None
    try:
        device, actual_device_kind = (
            _select_device(ctx, force_cpu=True) if force_cpu else _select_device(ctx)
        )
        if model_resource is None:
            _package, model = _load_model(inputs.model)
        else:
            _package, model = model_resource
        model = model.to(device)
        prognostic_ic = model.input_coords()
        time = to_time_array([inputs.start_time])
        if hasattr(model, "interp_method"):
            interp_to = prognostic_ic
            interp_method = model.interp_method
        else:
            interp_to = None
            interp_method = "nearest"

        data = GFS()
        x0, coords0 = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic["variable"],
            lead_time=prognostic_ic["lead_time"],
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )
        perturbation, perturbation_name = _build_perturbation(
            inputs.perturbation, inputs.noise_amplitude
        )

        torch.manual_seed(int(inputs.seed_base))
        for batch in _build_fanout_batches(inputs):
            batch_index = int(batch["batch_index"])
            member_ids = [int(member) for member in batch["batch_member_ids"]]
            batch_x, batch_coords = _build_batch_initial_conditions(
                x0,
                coords0,
                prognostic_ic,
                member_ids,
            )
            batch_x, batch_coords = perturbation(batch_x, batch_coords)
            state_path = state_dir / f"batch-{batch_index:04d}.pt"
            torch.save(
                {
                    "x": batch_x.cpu(),
                    "coords": batch_coords,
                    "member_ids": member_ids,
                    "batch_index": batch_index,
                    "perturbation": perturbation_name,
                },
                state_path,
            )
            prepared_batches.append(
                {
                    "batch_index": batch_index,
                    "batch_member_ids": member_ids,
                    "prepared_state_path": str(state_path),
                }
            )

        fanout_items = _fanout_items_from_prepared_batches(inputs, prepared_batches)
        return {
            "fanout_profile": {
                "item_count": len(fanout_items),
                "max_in_flight": min(int(inputs.max_in_flight), len(fanout_items)),
            },
            "fanout_items": fanout_items,
        }
    finally:
        cleanup_earth2_runtime_resources(data, model if owns_model else None)
        _package = None
        model = None
        data = None
        x0 = None
        coords0 = None
        batch_x = None
        batch_coords = None
        perturbation = None
        cleanup_python_and_torch_runtime(device=actual_device_kind)


def materialize_ensemble_fanout_perturbations(
    inputs: EnsembleFanoutInput,
    ctx: ExecutionContext | Mapping[str, Any],
    *,
    model_resource: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    captured_output = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(captured_output),
            contextlib.redirect_stderr(captured_output),
        ):
            materialized = _materialize_prepared_batch_states(
                inputs,
                ctx,
                model_resource=model_resource,
            )
    except Exception as exc:
        _raise_with_captured_output(
            "earth2-ensemble-fanout GPU materialization",
            captured_output.getvalue(),
            exc,
        )
    return {
        "status": "succeeded",
        "_pipeline_updates": {
            "operation": "run",
            "parameters": dict(inputs.__dict__),
            "fanout_profile": materialized["fanout_profile"],
            "fanout_items": materialized["fanout_items"],
        },
    }


def _load_prepared_batch_initial_conditions(
    inputs: EnsembleFanoutInput,
    ctx: ExecutionContext | Mapping[str, Any],
    prepared_state_path: str,
    *,
    model_resource: tuple[Any, Any] | None = None,
):
    import torch

    device, actual_device_kind = _select_device(ctx)
    package = None
    model = None
    owns_model = model_resource is None
    try:
        state = torch.load(prepared_state_path, map_location=device, weights_only=False)
        if model_resource is None:
            package, model = _load_model(inputs.model)
        else:
            package, model = model_resource
        model = model.to(device)
        batch_x = state["x"]
        if hasattr(batch_x, "to"):
            batch_x = batch_x.to(device)
        batch_coords = dict(state["coords"])
        member_ids = [int(member_id) for member_id in state["member_ids"]]
        batch_index = int(state.get("batch_index", inputs.batch_index))
        perturbation_name = str(state.get("perturbation") or inputs.perturbation)
        return (
            package,
            model,
            None,
            batch_x,
            batch_coords,
            actual_device_kind,
            perturbation_name,
            member_ids,
            batch_index,
        )
    except Exception:
        cleanup_earth2_runtime_resources(model if owns_model else None)
        cleanup_python_and_torch_runtime(device=actual_device_kind)
        raise


def run_ensemble_fanout_batch(
    inputs: EnsembleFanoutInput,
    ctx: ExecutionContext,
    *,
    model_resource: tuple[Any, Any] | None = None,
) -> EnsembleFanoutBatchOutput:
    from earth2studio.utils.coords import map_coords, split_coords

    fanout_item = _context_fanout_item(ctx)
    batch_index = int(fanout_item.get("item_index", inputs.batch_index))
    dataset_path = ctx.outputs.create(
        "forecast_batch_dataset",
        filename=f"forecast-batch-{batch_index:04d}.zarr",
        media_type="application/x-zarr",
        primary=True,
    )

    captured_output = io.StringIO()
    package = None
    model = None
    actual_device_kind = "cpu"
    member_ids = [int(member_id) for member_id in inputs.batch_member_ids or []]
    perturbation_name = str(inputs.perturbation).strip().lower()
    prepared_state_path = str(inputs.prepared_state_path or "").strip()
    data = None
    batch_x = None
    batch_coords = None
    io_backend = None
    model_iter = None
    step_x = None
    step_coords = None
    try:
        with (
            contextlib.redirect_stdout(captured_output),
            contextlib.redirect_stderr(captured_output),
        ):
            if not member_ids:
                member_ids = _batch_member_ids(
                    batch_index,
                    nensemble=int(inputs.nensemble),
                    batch_size=int(inputs.batch_size),
                )

            if prepared_state_path:
                (
                    package,
                    model,
                    data,
                    batch_x,
                    batch_coords,
                    actual_device_kind,
                    perturbation_name,
                    member_ids,
                    batch_index,
                ) = _load_prepared_batch_initial_conditions(
                    inputs,
                    ctx,
                    prepared_state_path,
                    model_resource=model_resource,
                )
            else:
                (
                    package,
                    model,
                    data,
                    batch_x,
                    batch_coords,
                    actual_device_kind,
                    perturbation_name,
                ) = _load_and_perturb_batch_initial_conditions(
                    inputs,
                    ctx,
                    batch_index=batch_index,
                    member_ids=member_ids,
                    model_resource=model_resource,
                )

            io_backend = _create_child_zarr_backend(
                dataset_path, ensemble_chunk_size=len(member_ids)
            )
            _configure_ensemble_io(
                io_backend,
                model,
                start_time=inputs.start_time,
                nsteps=inputs.nsteps,
                member_ids=member_ids,
            )

            model_iter = model.create_iterator(batch_x, batch_coords)
            try:
                for step, (step_x, step_coords) in enumerate(model_iter):
                    step_x, step_coords = map_coords(step_x, step_coords, {})
                    io_backend.write(*split_coords(step_x, step_coords))
                    if step == inputs.nsteps:
                        break
            except Exception:
                try:
                    _finalize_child_zarr_backend(io_backend)
                finally:
                    shutil.rmtree(dataset_path, ignore_errors=True)
                raise
            else:
                _finalize_child_zarr_backend(io_backend)
    except Exception as exc:
        _raise_with_captured_output(
            "earth2-ensemble-fanout batch forecast",
            captured_output.getvalue(),
            exc,
        )
    finally:
        cleanup_earth2_runtime_resources(
            data,
            model if model_resource is None else None,
        )
        model = None
        data = None
        batch_x = None
        batch_coords = None
        io_backend = None
        model_iter = None
        step_x = None
        step_coords = None
        cleanup_python_and_torch_runtime(device=actual_device_kind)

    return EnsembleFanoutBatchOutput(
        model=inputs.model,
        start_time=inputs.start_time,
        nsteps=inputs.nsteps,
        nensemble=len(member_ids),
        batch_index=batch_index,
        batch_member_ids=member_ids,
        perturbation=perturbation_name,
        noise_amplitude=inputs.noise_amplitude,
        dataset_path=str(dataset_path),
        prepared_state_path=prepared_state_path,
        note="earth2 ensemble fanout batch completed",
    )


def _link_or_copy_chunk(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _merge_chunk_tree(source_root: Path, destination_root: Path) -> None:
    if not source_root.exists():
        return
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        _link_or_copy_chunk(source, destination_root / relative)


def _array_dimension_names(array_root: Path, metadata: dict[str, Any]) -> list[str]:
    dimension_names = metadata.get("dimension_names")
    if isinstance(dimension_names, list) and all(
        isinstance(name, str) for name in dimension_names
    ):
        return [str(name) for name in dimension_names]

    return []


def _array_chunk_shape(metadata: dict[str, Any]) -> list[int]:
    chunk_grid = metadata.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        configuration = chunk_grid.get("configuration")
        if isinstance(configuration, dict):
            chunk_shape = configuration.get("chunk_shape")
            if isinstance(chunk_shape, list):
                return [int(chunk) for chunk in chunk_shape]
    return []


def _ensemble_axis_and_chunk_size(
    array_root: Path, metadata: dict[str, Any]
) -> tuple[int, int] | None:
    dimension_names = _array_dimension_names(array_root, metadata)
    chunk_shape = _array_chunk_shape(metadata)
    if "ensemble" not in dimension_names:
        return None
    axis = dimension_names.index("ensemble")
    if axis >= len(chunk_shape):
        return None
    return axis, max(1, int(chunk_shape[axis]))


def _destination_ensemble_chunk_index(
    *,
    local_chunk_index: int,
    child_ensemble_chunk_size: int,
    destination_ensemble_chunk_size: int,
    member_ids: list[int],
    member_index_by_id: dict[int, int],
) -> int | None:
    local_member_offset = local_chunk_index * child_ensemble_chunk_size
    if local_member_offset >= len(member_ids):
        return None
    global_member_id = member_ids[local_member_offset]
    destination_member_index = member_index_by_id[global_member_id]
    return destination_member_index // destination_ensemble_chunk_size


def _remap_chunk_tuple(
    parts: tuple[str, ...],
    *,
    child_ensemble_axis: int,
    child_ensemble_chunk_size: int,
    destination_ensemble_chunk_size: int,
    member_ids: list[int],
    member_index_by_id: dict[int, int],
) -> tuple[str, ...] | None:
    if child_ensemble_axis >= len(parts):
        return None
    try:
        local_chunk_index = int(parts[child_ensemble_axis])
    except ValueError:
        return None
    destination_chunk_index = _destination_ensemble_chunk_index(
        local_chunk_index=local_chunk_index,
        child_ensemble_chunk_size=child_ensemble_chunk_size,
        destination_ensemble_chunk_size=destination_ensemble_chunk_size,
        member_ids=member_ids,
        member_index_by_id=member_index_by_id,
    )
    if destination_chunk_index is None:
        return None
    remapped = list(parts)
    remapped[child_ensemble_axis] = str(destination_chunk_index)
    return tuple(remapped)


def _ensemble_axis_from_array_metadata(array_root: Path) -> int:
    metadata_path = array_root / "zarr.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ensemble = _ensemble_axis_and_chunk_size(array_root, metadata)
        if ensemble is not None:
            axis, _chunk_size = ensemble
            return axis
    return 0


def _copy_array_region(
    child_path: Path,
    dataset_path: Path,
    array_name: str,
    member_ids: list[int],
    member_index_by_id: dict[int, int],
) -> None:
    import zarr

    child_array = zarr.open_group(str(child_path), mode="r")[array_name]
    destination_array = zarr.open_group(str(dataset_path), mode="a")[array_name]
    child_ensemble_axis = _ensemble_axis_from_array_metadata(child_path / array_name)
    destination_ensemble_axis = _ensemble_axis_from_array_metadata(
        dataset_path / array_name
    )
    destination_indices = [
        member_index_by_id[int(member_id)] for member_id in member_ids
    ]
    if not destination_indices:
        return

    for local_index, destination_index in enumerate(destination_indices):
        child_selection = [slice(None)] * child_array.ndim
        child_selection[child_ensemble_axis] = slice(local_index, local_index + 1)
        destination_selection = [slice(None)] * destination_array.ndim
        destination_selection[destination_ensemble_axis] = slice(
            destination_index, destination_index + 1
        )
        child_data = child_array[tuple(child_selection)]
        if child_ensemble_axis != destination_ensemble_axis:
            import numpy as np

            child_data = np.moveaxis(
                child_data, child_ensemble_axis, destination_ensemble_axis
            )
        destination_array[tuple(destination_selection)] = child_data


def _merge_v3_array_chunks(
    child_path: Path,
    dataset_path: Path,
    array_name: str,
    member_ids: list[int],
    member_index_by_id: dict[int, int],
) -> None:
    child_array_root = child_path / array_name
    destination_array_root = dataset_path / array_name
    child_metadata = json.loads(
        (child_array_root / "zarr.json").read_text(encoding="utf-8")
    )
    destination_metadata = json.loads(
        (destination_array_root / "zarr.json").read_text(encoding="utf-8")
    )
    child_ensemble = _ensemble_axis_and_chunk_size(child_array_root, child_metadata)
    destination_ensemble = _ensemble_axis_and_chunk_size(
        destination_array_root, destination_metadata
    )
    if child_ensemble is not None and destination_ensemble is not None:
        child_axis, child_chunk_size = child_ensemble
        _destination_axis, destination_chunk_size = destination_ensemble
        if child_chunk_size != destination_chunk_size:
            _copy_array_region(
                child_path,
                dataset_path,
                array_name,
                member_ids,
                member_index_by_id,
            )
            return

        chunk_root = child_array_root / "c"
        for source in chunk_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(chunk_root)
            remapped = _remap_chunk_tuple(
                tuple(relative.parts),
                child_ensemble_axis=child_axis,
                child_ensemble_chunk_size=child_chunk_size,
                destination_ensemble_chunk_size=destination_chunk_size,
                member_ids=member_ids,
                member_index_by_id=member_index_by_id,
            )
            if remapped is None:
                continue
            _link_or_copy_chunk(source, destination_array_root / "c" / Path(*remapped))
        return

    chunk_root = child_path / array_name / "c"
    for local_member_index, global_member_id in enumerate(member_ids):
        destination_member_index = member_index_by_id[global_member_id]
        _merge_chunk_tree(
            chunk_root / str(local_member_index),
            dataset_path / array_name / "c" / str(destination_member_index),
        )


def _merge_child_data_chunks(
    candidate_children: list[tuple[Path, dict]],
    dataset_path: Path,
    data_array_names: list[str],
    member_index_by_id: dict[int, int],
) -> None:
    for child_path, child_result in candidate_children:
        member_ids = [int(member_id) for member_id in child_result["batch_member_ids"]]
        for array_name in data_array_names:
            v3_metadata = child_path / array_name / "zarr.json"
            if v3_metadata.exists():
                _merge_v3_array_chunks(
                    child_path,
                    dataset_path,
                    array_name,
                    member_ids,
                    member_index_by_id,
                )
                continue
            raise ValueError(
                "earth2 ensemble fanout merge only supports Zarr v3 child stores; "
                f"missing zarr.json for array {array_name!r} in {child_path}"
            )


def _ensure_v3_zarr_child_store(child_path: Path) -> None:
    if (child_path / "zarr.json").exists():
        return
    raise ValueError(
        "earth2 ensemble fanout merge only supports Zarr v3 child stores; "
        f"missing root zarr.json in {child_path}"
    )


def _merge_zarr_child_stores(
    candidate_children: list[tuple[Path, dict]],
    dataset_path: Path,
    *,
    nensemble: int,
) -> None:
    import xarray as xr

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    child_datasets = []
    try:
        for child_path, child_result in candidate_children:
            _ensure_v3_zarr_child_store(child_path)
            ds = xr.open_zarr(child_path, consolidated=False)
            if "ensemble" not in ds.dims:
                member_ids = child_result.get("batch_member_ids")
                ds = ds.expand_dims(
                    ensemble=[int(member_id) for member_id in member_ids]
                )
            child_datasets.append(ds)

        combined = xr.concat(child_datasets, dim="ensemble").sortby("ensemble")
        if int(combined.sizes.get("ensemble", 0)) > int(nensemble):
            raise ValueError("merged child datasets exceed the requested ensemble size")
        data_array_names = list(combined.data_vars)
        member_index_by_id = {
            int(member_id): index
            for index, member_id in enumerate(combined["ensemble"].values.tolist())
        }
        combined.drop_vars(data_array_names).to_zarr(
            dataset_path, mode="w", zarr_format=3
        )
        combined.to_zarr(dataset_path, mode="a", compute=False, zarr_format=3)
    finally:
        for ds in child_datasets:
            close = getattr(ds, "close", None)
            if callable(close):
                close()

    _merge_child_data_chunks(
        candidate_children, dataset_path, data_array_names, member_index_by_id
    )


def postprocess_ensemble_fanout_result(
    result: PriorResult[Any], ctx: PostprocessContext
) -> PostprocessOutcome[EnsembleFanoutOutput | dict[str, Any]]:
    result_payload = getattr(result, "payload", result)
    if not isinstance(result_payload, dict):
        return PostprocessOutcome(payload={}, status="succeeded")

    child_results = result_payload.get("child_results")
    if not isinstance(child_results, list):
        execution = getattr(result, "execution", None)
        status = getattr(execution, "status", "succeeded")
        return PostprocessOutcome(payload=dict(result_payload), status=status)

    output_root = _context_run_dir(ctx)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "forecast-ensemble.zarr"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    candidate_children: list[tuple[Path, dict]] = []
    child_paths: list[str] = []
    skipped_children = 0
    for entry in sorted(
        child_results,
        key=lambda item: (
            int(item.get("item_index", 0)) if isinstance(item, dict) else 0
        ),
    ):
        if not isinstance(entry, dict):
            skipped_children += 1
            continue

        child_result = entry.get("result")
        if not isinstance(child_result, dict):
            skipped_children += 1
            continue

        child_status = str(child_result.get("status") or "succeeded").strip().lower()
        if child_status in {"failed", "fail", "error", "cancelled", "canceled"}:
            skipped_children += 1
            continue

        child_path = Path(str(child_result.get("dataset_path") or "")).expanduser()
        if not child_path.exists():
            skipped_children += 1
            continue
        member_ids = child_result.get("batch_member_ids")
        if not isinstance(member_ids, list) or not member_ids:
            skipped_children += 1
            continue

        candidate_children.append((child_path, child_result))

    if len(candidate_children) == 1:
        child_path, _child_result = candidate_children[0]
        _copy_store(child_path, dataset_path)
        child_paths.append(str(child_path))
    elif candidate_children:
        params = dict(ctx.request.raw_fields)
        nensemble = int(params.get("nensemble") or 0)
        if nensemble < 1:
            nensemble = 1 + max(
                int(member_id)
                for _child_path, child_result in candidate_children
                for member_id in child_result.get("batch_member_ids", [])
            )
        _merge_zarr_child_stores(
            candidate_children,
            dataset_path,
            nensemble=nensemble,
        )
        child_paths.extend(str(child_path) for child_path, _ in candidate_children)

    aggregation_summary = result_payload.get("aggregation_summary", {})
    collect_failed_count = 0
    if isinstance(aggregation_summary, dict):
        collect_failed_count = int(aggregation_summary.get("failed_count") or 0)

    params = dict(ctx.request.raw_fields)
    has_aggregated_dataset = bool(child_paths)
    postprocess_failed = (not has_aggregated_dataset) or skipped_children > 0
    status_failed = collect_failed_count > 0 or postprocess_failed

    if has_aggregated_dataset:
        ctx.outputs.register(
            "forecast_dataset",
            dataset_path,
            media_type="application/x-zarr",
            primary=True,
        )

    return PostprocessOutcome(
        payload=EnsembleFanoutOutput(
            model=params.get("model"),
            start_time=params.get("start_time"),
            nsteps=params.get("nsteps"),
            nensemble=params.get("nensemble"),
            batch_size=params.get("batch_size"),
            dataset_path=str(dataset_path),
            batch_dataset_paths=child_paths,
            aggregation_summary=dict(aggregation_summary)
            if isinstance(aggregation_summary, dict)
            else {},
            postprocess_summary=EnsembleFanoutPostprocessSummary(
                skipped_count=skipped_children,
                aggregated_count=len(child_paths),
                partial_aggregation=postprocess_failed,
            ),
            note="earth2 ensemble fanout run completed",
        ),
        status="succeeded" if not status_failed else "failed",
    )
