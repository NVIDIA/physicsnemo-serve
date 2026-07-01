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
class DeterministicInput:
    model: str
    start_time: str
    nsteps: int


@dataclass
class DeterministicOutput:
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


class DeterministicWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["DLWP"]
    input_model = DeterministicInput
    output_model = DeterministicOutput

    def __init__(self) -> None:
        self._package = None
        self._model = None
        self._data = None
        self._runtime_device = "cpu"

    def _ensure_runtime_loaded(self, ctx: ExecutionContext | Mapping[str, Any]):
        import torch
        from earth2studio.data import GFS
        from earth2studio.models.px import DLWP

        gpus_required = 1
        resource_profile = (
            ctx.get("resource_profile")
            if isinstance(ctx, Mapping)
            else ctx.resource_profile
        )
        if isinstance(resource_profile, ResourceProfile):
            gpus_required = resource_profile.gpus_required
        elif isinstance(resource_profile, dict):
            gpus_required = int(resource_profile.get("gpus_required") or 0)

        if gpus_required > 0 and torch.cuda.is_available():
            device = torch.device("cuda")
            self._runtime_device = "gpu"
        else:
            device = torch.device("cpu")
            self._runtime_device = "cpu"

        if self._model is None:
            self._package = DLWP.load_default_package()
            self._model = DLWP.load_model(self._package)
        self._model = self._model.to(device)
        if self._data is None:
            self._data = GFS()
        return self._model, self._data, device

    def warmup(self, ctx: dict) -> dict[str, list[str]]:
        self._ensure_runtime_loaded(ctx)
        return {"model_names": ["DLWP"]}

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        params = dict(request.raw_fields)

        model = str(params.get("model") or "").strip()
        if model != "dlwp":
            raise ValueError(
                "earth2-deterministic currently supports only model='dlwp'"
            )

        start_time = str(params.get("start_time") or "").strip()
        if not start_time:
            raise ValueError("start_time must be a non-empty string")

        nsteps = int(params.get("nsteps"))
        if nsteps < 1:
            raise ValueError("nsteps must be >= 1")

        return PrepareResult(
            inputs=DeterministicInput(
                model=model,
                start_time=start_time,
                nsteps=nsteps,
            ),
            resource_profile=None,
            prefetch_plan=[],
        )

    # Main execution hook.
    def run(
        self, inputs: DeterministicInput, ctx: ExecutionContext
    ) -> DeterministicOutput:
        dataset_path = ctx.outputs.create(
            "forecast_dataset",
            filename="forecast.zarr",
            media_type="application/x-zarr",
            primary=True,
        )

        if inputs.model != "dlwp":
            raise ValueError(
                "earth2-deterministic currently supports only model='dlwp'"
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

                model, data, device = self._ensure_runtime_loaded(ctx)

                deterministic(
                    time=to_time_array([inputs.start_time]),
                    nsteps=inputs.nsteps,
                    prognostic=model,
                    data=data,
                    io=ZarrBackend(str(dataset_path)),
                    device=device,
                )
        except Exception as exc:
            details = captured_output.getvalue().strip()
            if details:
                raise RuntimeError(
                    f"earth2-deterministic forecast failed:\n{details}"
                ) from exc
            raise

        return DeterministicOutput(
            model=inputs.model,
            start_time=inputs.start_time,
            nsteps=inputs.nsteps,
            dataset_path=str(dataset_path),
            note="earth2 deterministic run completed",
        )

    def cleanup(self) -> None:
        cleanup_earth2_runtime_resources(self._data, self._model)
        self._package = None
        self._model = None
        self._data = None
        cleanup_python_and_torch_runtime(device=self._runtime_device)


WORKFLOW = DeterministicWorkflow
