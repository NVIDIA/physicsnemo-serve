# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Mapping

from plugin_sdk import (
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
class EnsembleInput:
    model: str
    start_time: str
    nsteps: int
    nensemble: int
    batch_size: int = 1
    perturbation: str = "gaussian"
    noise_amplitude: float = 0.05
    seed_base: int = 1000


@dataclass
class EnsembleOutput:
    model: str
    start_time: str
    nsteps: int
    nensemble: int
    batch_size: int
    perturbation: str
    noise_amplitude: float
    seed_base: int
    dataset_path: str
    note: str


def prepare_model_cache(_ctx: dict[str, object]) -> dict[str, list[str]]:
    from earth2studio.models.px import DLWP

    package = DLWP.load_default_package()
    DLWP.load_model(package)
    return {"model_names": ["DLWP"]}


def _load_model(model_name: str):
    from earth2studio.models.px import DLWP

    if model_name != "dlwp":
        raise ValueError("earth2-ensemble currently supports only model='dlwp'")

    package = DLWP.load_default_package()
    return DLWP.load_model(package)


def _build_perturbation(name: str, noise_amplitude: float):
    normalized = str(name).strip().lower()
    if normalized == "gaussian":
        from earth2studio.perturbation import Gaussian

        return Gaussian(noise_amplitude=noise_amplitude), normalized
    if normalized == "brown":
        from earth2studio.perturbation import Brown

        return Brown(noise_amplitude=noise_amplitude), normalized
    raise ValueError("perturbation must be 'gaussian' or 'brown'")


def _normalize_parameters(raw: Mapping[str, Any]) -> EnsembleInput:
    model = str(raw.get("model") or "").strip()
    if model != "dlwp":
        raise ValueError("earth2-ensemble currently supports only model='dlwp'")

    start_time = str(raw.get("start_time") or "").strip()
    if not start_time:
        raise ValueError("start_time must be a non-empty string")

    nsteps = int(raw.get("nsteps"))
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")

    nensemble = int(raw.get("nensemble"))
    if nensemble < 1:
        raise ValueError("nensemble must be >= 1")

    batch_size = int(raw.get("batch_size") or nensemble)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    perturbation = str(raw.get("perturbation") or "gaussian").strip().lower()
    noise_amplitude = float(raw.get("noise_amplitude") or 0.05)
    if noise_amplitude <= 0:
        raise ValueError("noise_amplitude must be > 0")

    seed_base = int(raw.get("seed_base") or 1000)

    return EnsembleInput(
        model=model,
        start_time=start_time,
        nsteps=nsteps,
        nensemble=nensemble,
        batch_size=min(batch_size, nensemble),
        perturbation=perturbation,
        noise_amplitude=noise_amplitude,
        seed_base=seed_base,
    )


def _select_device(ctx: ExecutionContext | Mapping[str, Any]):
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


def _resource_profile() -> dict[str, object]:
    return {
        "executor_class": "earth2-gpu",
        "gpus_required": 1,
        "memory_mb": 2048,
        "tags": ["earth2", "gpu"],
    }


class EnsembleWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["DLWP"]
    input_model = EnsembleInput
    output_model = EnsembleOutput

    def __init__(self) -> None:
        self._package = None
        self._model = None
        self._data = None
        self._runtime_device = "cpu"

    def _ensure_runtime_loaded(
        self, model_name: str, ctx: ExecutionContext | Mapping[str, Any]
    ):
        from earth2studio.data import GFS

        device, self._runtime_device = _select_device(ctx)
        if self._model is None:
            self._package = None
            self._model = _load_model(model_name)
        self._model = self._model.to(device)
        if self._data is None:
            self._data = GFS()
        return self._model, self._data, device

    def warmup(self, ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded("dlwp", ctx)
        return {"model_names": ["DLWP"]}

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        inputs = _normalize_parameters(request.raw_fields)
        return PrepareResult(
            inputs=inputs,
            resource_profile=_resource_profile(),
            prefetch_plan=[],
        )

    # Main execution hook.
    def run(self, inputs: EnsembleInput, ctx: ExecutionContext) -> EnsembleOutput:
        dataset_path = ctx.outputs.create(
            "forecast_dataset",
            filename="forecast-ensemble.zarr",
            media_type="application/x-zarr",
            primary=True,
        )

        perturbation_name = str(inputs.perturbation).strip().lower()
        perturbation = None
        captured_output = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured_output),
                contextlib.redirect_stderr(captured_output),
            ):
                import torch
                from earth2studio.io import ZarrBackend
                from earth2studio.run import ensemble
                from earth2studio.utils.time import to_time_array

                model, data, device = self._ensure_runtime_loaded(inputs.model, ctx)
                torch.manual_seed(inputs.seed_base)
                perturbation, perturbation_name = _build_perturbation(
                    inputs.perturbation, inputs.noise_amplitude
                )

                ensemble(
                    time=to_time_array([inputs.start_time]),
                    nsteps=inputs.nsteps,
                    nensemble=inputs.nensemble,
                    prognostic=model,
                    data=data,
                    io=ZarrBackend(str(dataset_path)),
                    perturbation=perturbation,
                    batch_size=inputs.batch_size,
                    device=device,
                )
        except Exception as exc:
            _raise_with_captured_output(
                "earth2-ensemble forecast", captured_output.getvalue(), exc
            )
        finally:
            perturbation = None

        return EnsembleOutput(
            model=inputs.model,
            start_time=inputs.start_time,
            nsteps=inputs.nsteps,
            nensemble=inputs.nensemble,
            batch_size=inputs.batch_size,
            perturbation=perturbation_name,
            noise_amplitude=inputs.noise_amplitude,
            seed_base=inputs.seed_base,
            dataset_path=str(dataset_path),
            note="earth2 ensemble run completed",
        )

    def cleanup(self) -> None:
        cleanup_earth2_runtime_resources(self._data, self._model)
        self._package = None
        self._model = None
        self._data = None
        cleanup_python_and_torch_runtime(device=self._runtime_device)


WORKFLOW = EnsembleWorkflow
