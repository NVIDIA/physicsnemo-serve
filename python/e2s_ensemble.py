# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-scatter helpers for the FCN Earth2Studio ensemble workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from e2s_workflow import _selected_zarr_backend, create_zarr_backend
from plugin_sdk import (
    ExecutionContext,
    PluginCancelledError,
    ScatterChild,
    ScatterResult,
)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_device(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("e2s-ensemble requires a CUDA worker")
    return torch.device("cuda")


def _member_groups(nensemble: int, batch_size: int) -> list[list[int]]:
    return [
        list(range(offset, min(offset + batch_size, nensemble)))
        for offset in range(0, nensemble, batch_size)
    ]


def _normalize_perturbation(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"gaussian", "brown", "spherical_gaussian"}:
        raise ValueError(
            "perturbation must be 'gaussian', 'brown', or 'spherical_gaussian'"
        )
    return normalized


def _build_perturbation(name: str, noise_amplitude: float) -> tuple[Any, str]:
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
    raise AssertionError("unreachable")


def _validate_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    forecast_times = normalized.get("forecast_times")
    if (
        not isinstance(forecast_times, list)
        or not forecast_times
        or any(
            not isinstance(value, str) or not value.strip() for value in forecast_times
        )
    ):
        raise ValueError("forecast_times must be a non-empty list")
    nensemble = int(normalized.get("nensemble") or 0)
    batch_size = int(normalized.get("batch_size") or 0)
    raw_max_in_flight = normalized.get("max_in_flight")
    max_in_flight = 1 if raw_max_in_flight is None else int(raw_max_in_flight)
    nsteps = int(normalized.get("nsteps") or 0)
    if str(normalized.get("model_type") or "fcn").lower() != "fcn":
        raise ValueError("e2s-ensemble supports only model_type='fcn'")
    if str(normalized.get("data_source") or "gfs").lower() != "gfs":
        raise ValueError("e2s-ensemble supports only data_source='gfs'")
    if str(normalized.get("output_format") or "zarr").lower() != "zarr":
        raise ValueError("e2s-ensemble supports only output_format='zarr'")
    normalized["perturbation"] = _normalize_perturbation(
        str(normalized.get("perturbation") or "spherical_gaussian")
    )
    raw_noise_amplitude = normalized.get("noise_amplitude")
    noise_amplitude = (
        0.15 if raw_noise_amplitude is None else float(raw_noise_amplitude)
    )
    if noise_amplitude <= 0:
        raise ValueError("noise_amplitude must be > 0")
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")
    if nensemble < 1:
        raise ValueError("nensemble must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    normalized["nensemble"] = nensemble
    normalized["batch_size"] = min(batch_size, nensemble)
    normalized["max_in_flight"] = max_in_flight
    normalized["nsteps"] = nsteps
    normalized["noise_amplitude"] = noise_amplitude
    return normalized


def _load_committed_manifest(
    state_dir: Path, inputs_digest: str
) -> dict[str, Any] | None:
    manifest_path = state_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("inputs_digest") != inputs_digest:
        raise ValueError("committed ensemble inputs do not match this request")
    for batch in manifest.get("batches", []):
        state_path = state_dir / str(batch.get("filename") or "")
        if not state_path.is_file() or _file_digest(state_path) != batch.get("sha256"):
            raise ValueError(
                f"committed ensemble input is missing or corrupt: {state_path}"
            )
    return manifest


def _validate_committed_aggregate(
    dataset_path: Path, expected_members: list[int], xr: Any
) -> None:
    if not dataset_path.exists():
        raise ValueError("committed ensemble output is missing")
    dataset = xr.open_zarr(dataset_path, consolidated=False)
    try:
        if "ensemble" not in dataset.coords:
            raise ValueError("committed ensemble output has no ensemble coordinate")
        actual_members = [int(member) for member in dataset["ensemble"].values.tolist()]
        if actual_members != expected_members:
            raise ValueError(
                f"committed ensemble output requires members {expected_members}, "
                f"got {actual_members}"
            )
    finally:
        dataset.close()


def materialize_inputs(
    inputs: Mapping[str, Any],
    ctx: ExecutionContext,
    *,
    model: Any,
    data: Any,
) -> tuple[dict[str, Any], ScatterResult]:
    """Create immutable perturbations and return the scheduler scatter instruction."""
    import numpy as np
    import torch
    from earth2studio.data import fetch_data
    from earth2studio.utils.coords import map_coords
    from earth2studio.utils.time import to_time_array

    values = _validate_inputs(inputs)
    expected_groups = _member_groups(values["nensemble"], values["batch_size"])
    scientific_inputs = {
        key: value for key, value in values.items() if key != "max_in_flight"
    }
    inputs_digest = _canonical_digest(scientific_inputs)
    state_dir = ctx.run_dir / "prepared-initial-conditions"
    manifest = _load_committed_manifest(state_dir, inputs_digest)
    if (
        manifest is not None
        and [batch.get("member_ids") for batch in manifest.get("batches", [])]
        != expected_groups
    ):
        raise ValueError("committed ensemble manifest has invalid member coverage")

    if manifest is None:
        # A directory without a committed manifest is only a failed prior attempt.
        if state_dir.exists():
            shutil.rmtree(state_dir)
        temp_dir = state_dir.with_name(f".{state_dir.name}.tmp-{uuid.uuid4().hex}")
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            device = _cuda_device(torch)
            model = model.to(device)
            prognostic_coords = model.input_coords()
            times = to_time_array(values["forecast_times"])
            interpolation_coords = (
                prognostic_coords if hasattr(model, "interp_method") else None
            )
            interpolation_method = getattr(model, "interp_method", "nearest")
            x0, coords0 = fetch_data(
                source=data,
                time=times,
                variable=prognostic_coords["variable"],
                lead_time=prognostic_coords["lead_time"],
                device=device,
                interp_to=interpolation_coords,
                interp_method=interpolation_method,
            )
            perturbation, perturbation_name = _build_perturbation(
                str(values.get("perturbation") or "spherical_gaussian"),
                float(values.get("noise_amplitude") or 0.15),
            )
            seed_base = values.get("seed_base")
            torch.manual_seed(1000 if seed_base is None else int(seed_base))
            batches = []
            for batch_index, member_ids in enumerate(expected_groups):
                batch_x = x0.unsqueeze(0).repeat(len(member_ids), *([1] * x0.ndim))
                batch_coords = {"ensemble": np.asarray(member_ids)} | coords0.copy()
                batch_x, batch_coords = map_coords(
                    batch_x, batch_coords, prognostic_coords
                )
                batch_x, batch_coords = perturbation(batch_x, batch_coords)
                filename = f"batch-{batch_index:04d}.pt"
                state_path = temp_dir / filename
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
                batches.append(
                    {
                        "batch_index": batch_index,
                        "member_ids": member_ids,
                        "filename": filename,
                        "sha256": _file_digest(state_path),
                    }
                )
            manifest = {"inputs_digest": inputs_digest, "batches": batches}
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            try:
                temp_dir.replace(state_dir)
            except OSError:
                committed = _load_committed_manifest(state_dir, inputs_digest)
                if committed is None:
                    raise
                manifest = committed
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    children = []
    for batch in manifest["batches"]:
        child_parameters = {
            **values,
            "batch_index": int(batch["batch_index"]),
            "batch_member_ids": [int(member) for member in batch["member_ids"]],
            "prepared_state_path": str(state_dir / batch["filename"]),
            "prepared_state_sha256": str(batch["sha256"]),
        }
        children.append(
            ScatterChild(
                operation="run_ensemble_batch",
                parameters=child_parameters,
                resource_profile={
                    "executor_class": "earth2-gpu",
                    "gpus_required": 1,
                    "memory_mb": 4096,
                    "tags": ["earth2", "gpu"],
                },
                batch_profile={
                    "batch_key": f"e2s-ensemble:{ctx.run_id}",
                    "max_batch_size": min(
                        values["max_in_flight"], len(manifest["batches"])
                    ),
                },
            )
        )
    return values, ScatterResult(
        children=children,
        child_stage_id="execute",
        continuation_stage_id="execute",
        max_in_flight=min(values["max_in_flight"], len(children)),
    )


def run_batch(
    inputs: Mapping[str, Any],
    ctx: ExecutionContext,
    *,
    model: Any,
    attempt_token: str = "",
) -> dict[str, Any]:
    """Run one pre-materialized scientific ensemble batch."""
    import numpy as np
    import torch
    from earth2studio.utils.coords import map_coords, split_coords
    from earth2studio.utils.time import to_time_array

    values = _validate_inputs(inputs)
    state_path = Path(str(values.get("prepared_state_path") or ""))
    expected_digest = str(values.get("prepared_state_sha256") or "")
    if not state_path.is_file() or _file_digest(state_path) != expected_digest:
        raise ValueError(f"prepared ensemble input is missing or corrupt: {state_path}")

    member_ids = [int(member) for member in values.get("batch_member_ids") or []]
    if not member_ids:
        raise ValueError("ensemble child has no members")
    batch_index = int(values.get("batch_index") or 0)
    safe_attempt = "".join(
        character
        for character in attempt_token
        if character.isalnum() or character in "-_"
    )
    suffix = f"-attempt-{safe_attempt}" if safe_attempt else ""
    dataset_path = Path(
        ctx.outputs.create(
            "forecast_batch_dataset",
            filename=f"forecast-batch-{batch_index:04d}{suffix}.zarr",
            media_type="application/x-zarr",
            primary=True,
        )
    )
    active_path = dataset_path.with_name(f".{dataset_path.name}.tmp-{uuid.uuid4().hex}")
    device = _cuda_device(torch)
    state = torch.load(state_path, map_location=device, weights_only=False)
    if [int(member) for member in state["member_ids"]] != member_ids:
        raise ValueError("prepared ensemble member IDs do not match the child request")

    model = model.to(device)
    total_coords = model.output_coords(model.input_coords()).copy()
    total_coords.pop("batch", None)
    total_coords["time"] = to_time_array(values["forecast_times"])
    lead_time = model.output_coords(model.input_coords())["lead_time"]
    total_coords["lead_time"] = np.asarray(
        [lead_time * step for step in range(int(values["nsteps"]) + 1)]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    total_coords = {"ensemble": np.asarray(member_ids)} | total_coords
    requested_variables = values.get("output_variables")
    variables = (
        np.asarray(requested_variables)
        if requested_variables
        else total_coords["variable"]
    )
    total_coords["variable"] = variables
    output_coords = OrderedDict({"variable": variables})

    io_kwargs: dict[str, Any] = {
        "chunks": {"ensemble": 1, "time": 1, "lead_time": 1},
        "backend_kwargs": {"overwrite": True},
    }
    if _selected_zarr_backend() != "python":
        io_kwargs["default_parallel_coord_names"] = [
            "ensemble",
            "time",
            "lead_time",
        ]
    io_backend = create_zarr_backend(str(active_path), **io_kwargs)
    arrays = total_coords.pop("variable")
    io_backend.add_array(total_coords, arrays)
    try:
        iterator = model.create_iterator(state["x"].to(device), dict(state["coords"]))
        for step, (step_x, step_coords) in enumerate(iterator):
            if ctx.abort_requested():
                raise PluginCancelledError("ensemble parent was cancelled")
            step_x, step_coords = map_coords(step_x, step_coords, output_coords)
            io_backend.write(*split_coords(step_x, step_coords))
            if step == int(values["nsteps"]):
                break
        finalizer = getattr(io_backend, "finalize", None)
        if callable(finalizer):
            finalizer()
        else:
            closer = getattr(io_backend, "close", None)
            if callable(closer):
                closer()
        if dataset_path.exists():
            shutil.rmtree(active_path, ignore_errors=True)
        else:
            active_path.replace(dataset_path)
    except Exception:
        shutil.rmtree(active_path, ignore_errors=True)
        raise

    return {
        "dataset_path": str(dataset_path),
        "batch_index": batch_index,
        "batch_member_ids": member_ids,
        "prepared_state_path": str(state_path),
    }


def aggregate(
    inputs: Mapping[str, Any], child_results: list[Any], ctx: ExecutionContext
) -> dict[str, Any]:
    """Validate a complete round and assemble its child Zarr stores."""
    import xarray as xr

    values = _validate_inputs(inputs)
    expected_members = list(range(values["nensemble"]))
    children: list[tuple[list[int], Path]] = []
    for entry in sorted(
        child_results,
        key=lambda value: (
            int(value.get("item_index", 0)) if isinstance(value, dict) else 0
        ),
    ):
        if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
            raise ValueError("ensemble gather received an invalid child result")
        result = entry["result"]
        if str(result.get("status") or "succeeded") != "succeeded":
            raise ValueError("ensemble gather requires every child to succeed")
        member_ids = [int(member) for member in result.get("batch_member_ids") or []]
        dataset_path = Path(str(result.get("dataset_path") or ""))
        if not member_ids or not dataset_path.exists():
            raise ValueError("ensemble child output is missing or incomplete")
        children.append((member_ids, dataset_path))
    actual_members = [member for member_ids, _ in children for member in member_ids]
    if actual_members != expected_members:
        raise ValueError(
            f"ensemble gather requires members {expected_members}, got {actual_members}"
        )

    dataset_path = Path(
        ctx.outputs.create(
            "forecast_dataset",
            filename="forecast.zarr",
            media_type="application/x-zarr",
            primary=True,
        )
    )
    if not dataset_path.exists():
        active_path = dataset_path.with_name(
            f".{dataset_path.name}.tmp-{uuid.uuid4().hex}"
        )
        datasets = [xr.open_zarr(path, consolidated=False) for _, path in children]
        try:
            combined = xr.concat(datasets, dim="ensemble").sortby("ensemble")
            combined.to_zarr(active_path, mode="w", zarr_format=3)
            try:
                active_path.replace(dataset_path)
            except OSError:
                if not dataset_path.exists():
                    raise
        finally:
            shutil.rmtree(active_path, ignore_errors=True)
            for dataset in datasets:
                dataset.close()

    _validate_committed_aggregate(dataset_path, expected_members, xr)

    metadata = {
        key: values.get(key)
        for key in (
            "forecast_times",
            "nsteps",
            "nensemble",
            "batch_size",
            "max_in_flight",
            "model_type",
            "perturbation",
            "noise_amplitude",
            "seed_base",
            "data_source",
            "output_format",
        )
    }
    (ctx.run_dir / "forecast_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "dataset_path": str(dataset_path),
        "batch_dataset_paths": [str(path) for _, path in children],
        "nensemble": values["nensemble"],
    }
