# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Mapping

# Runtime injects the scripts directory into sys.path for plugin execution.
from plugin_sdk import (  # pyright: ignore[reportMissingImports]
    BatchExecutionContext,
    BatchItem,
    BatchItemResult,
    ExecutionContext,
    PluginWorkflow,
    PrepareContext,
    PrepareResult,
    RawRequest,
    ResourceProfile,
    cleanup_earth2_runtime_resources,
    cleanup_python_and_torch_runtime,
)


@dataclass
class DeterministicBatchInput:
    model: str
    start_time: str
    nsteps: int


@dataclass
class DeterministicBatchOutput:
    model: str
    start_time: str
    nsteps: int
    dataset_path: str
    note: str


def prepare_model_cache(_ctx: dict[str, object]) -> dict[str, list[str]]:
    from earth2studio.models.px import DLWP

    package = DLWP.load_default_package()
    DLWP.load_model(package)
    return {"model_names": ["DLWP"]}


def _load_model(model_name: str) -> tuple[Any, Any]:
    from earth2studio.models.px import DLWP

    if model_name != "dlwp":
        raise ValueError(
            "earth2-deterministic-batch currently supports only model='dlwp'"
        )

    package = DLWP.load_default_package()
    return package, DLWP.load_model(package)


def _normalize_parameters(raw: Mapping[str, Any]) -> DeterministicBatchInput:
    model = str(raw.get("model") or "").strip()
    if model != "dlwp":
        raise ValueError(
            "earth2-deterministic-batch currently supports only model='dlwp'"
        )

    start_time = str(raw.get("start_time") or "").strip()
    if not start_time:
        raise ValueError("start_time must be a non-empty string")

    nsteps = int(raw.get("nsteps"))
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")

    return DeterministicBatchInput(
        model=model,
        start_time=start_time,
        nsteps=nsteps,
    )


def _resource_profile() -> dict[str, object]:
    return {
        "executor_class": "earth2-gpu",
        "gpus_required": 1,
        "memory_mb": 2048,
        "tags": ["earth2", "gpu"],
    }


def _batch_profile(inputs: DeterministicBatchInput) -> dict[str, Any]:
    return {
        "enabled": True,
        "batch_key": inputs.model,
        "max_batch_size": 4,
        "max_wait_ms": 200,
        "shared_memory_mb": 8192,
        "incremental_memory_mb": 1024,
    }


def _select_device(
    ctx: BatchExecutionContext | ExecutionContext | Mapping[str, Any],
):
    import torch

    resource_profile: ResourceProfile | Mapping[str, Any] | None
    if isinstance(ctx, Mapping):
        resource_profile = ctx.get("resource_profile")
    else:
        resource_profile = ctx.resource_profile

    runtime_device = "gpu"
    if isinstance(resource_profile, ResourceProfile):
        runtime_device = "gpu" if resource_profile.gpus_required > 0 else "cpu"
    elif isinstance(resource_profile, Mapping):
        runtime_device = (
            "gpu" if int(resource_profile.get("gpus_required") or 1) > 0 else "cpu"
        )

    if runtime_device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    return torch.device("cpu"), "cpu"


def _raise_with_captured_output(label: str, captured_output: str, exc: Exception):
    details = captured_output.strip()
    if details:
        raise RuntimeError(f"{label} failed:\n{details}") from exc
    raise exc


def _load_runtime_resources(
    model_name: str,
    ctx: BatchExecutionContext | ExecutionContext | Mapping[str, Any],
) -> tuple[Any, Any, Any, str]:
    device, actual_device_kind = _select_device(ctx)
    package = None
    model = None
    data = None
    captured_output = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(captured_output),
            contextlib.redirect_stderr(captured_output),
        ):
            from earth2studio.data import GFS

            package, model = _load_model(model_name)
            model = model.to(device)
            data = GFS()
    except Exception as exc:
        cleanup_earth2_runtime_resources(data, model)
        package = None
        model = None
        data = None
        cleanup_python_and_torch_runtime(device=actual_device_kind)
        _raise_with_captured_output(
            "earth2-deterministic-batch setup", captured_output.getvalue(), exc
        )
    return package, model, data, actual_device_kind


class DeterministicBatchWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["DLWP"]
    input_model = DeterministicBatchInput
    output_model = DeterministicBatchOutput

    def __init__(self) -> None:
        self._package = None
        self._model = None
        self._data = None
        self._runtime_device = "cpu"

    def _ensure_runtime_loaded(
        self,
        model_name: str,
        ctx: BatchExecutionContext | ExecutionContext | Mapping[str, Any],
    ) -> tuple[Any, Any, str]:
        device, self._runtime_device = _select_device(ctx)
        if self._model is None:
            self._package, self._model = _load_model(model_name)
        self._model = self._model.to(device)
        if self._data is None:
            from earth2studio.data import GFS

            self._data = GFS()
        return self._model, self._data, self._runtime_device

    def warmup(self, ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded("dlwp", ctx)
        return {"model_names": ["DLWP"]}

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        inputs = _normalize_parameters(request.raw_fields)
        return PrepareResult(
            inputs=inputs,
            resource_profile=_resource_profile(),
            batch_profile=_batch_profile(inputs),
        )

    # Batch execution hook. Process compatible items together.
    def run_batch(
        self,
        items: list[BatchItem[DeterministicBatchInput]],
        ctx: BatchExecutionContext,
    ) -> list[DeterministicBatchOutput | BatchItemResult[DeterministicBatchOutput]]:
        if not items:
            return []

        model, data, actual_device_kind = self._ensure_runtime_loaded(
            items[0].inputs.model, ctx
        )
        try:
            results: list[
                DeterministicBatchOutput | BatchItemResult[DeterministicBatchOutput]
            ] = []
            for item in items:
                try:
                    results.append(
                        self._run_single(
                            item.inputs,
                            item.context,
                            model=model,
                            data=data,
                            actual_device_kind=actual_device_kind,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(BatchItemResult.failed(str(exc)))
            return results
        finally:
            pass

    def _run_single(
        self,
        inputs: DeterministicBatchInput,
        ctx: ExecutionContext,
        *,
        model,
        data,
        actual_device_kind: str,
    ) -> DeterministicBatchOutput:
        dataset_path = ctx.outputs.create(
            "forecast_dataset",
            filename="forecast.zarr",
            media_type="application/x-zarr",
            primary=True,
        )

        captured_output = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured_output),
                contextlib.redirect_stderr(captured_output),
            ):
                from earth2studio.io import ZarrBackend
                from earth2studio.run import deterministic
                from earth2studio.utils.time import to_time_array

                device, _ = _select_device(ctx)
                deterministic(
                    time=to_time_array([inputs.start_time]),
                    nsteps=inputs.nsteps,
                    prognostic=model,
                    data=data,
                    io=ZarrBackend(str(dataset_path)),
                    device=device,
                )
        except Exception as exc:
            _raise_with_captured_output(
                f"earth2-deterministic-batch forecast for run {ctx.run_id}",
                captured_output.getvalue(),
                exc,
            )

        return DeterministicBatchOutput(
            model=inputs.model,
            start_time=inputs.start_time,
            nsteps=inputs.nsteps,
            dataset_path=str(dataset_path),
            note="earth2 deterministic batch forecast completed",
        )

    def cleanup(self) -> None:
        cleanup_earth2_runtime_resources(self._data, self._model)
        self._package = None
        self._model = None
        self._data = None
        cleanup_python_and_torch_runtime(device=self._runtime_device)


WORKFLOW = DeterministicBatchWorkflow
