# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Runtime injects the scripts directory into sys.path for plugin execution.
from plugin_sdk import (  # pyright: ignore[reportMissingImports]
    ExecutionContext,
    PluginWorkflow,
    PostprocessContext,
    PostprocessOutcome,
    PrepareContext,
    PriorResult,
    RawRequest,
    build_execution_context,
    cleanup_earth2_runtime_resources,
    cleanup_python_and_torch_runtime,
)

PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from earth2_ensemble_fanout_support import (  # noqa: E402
    EnsembleFanoutBatchOutput,
    EnsembleFanoutInput,
    EnsembleFanoutOutput,
    _load_model,
    _select_device,
    materialize_ensemble_fanout_perturbations,
    postprocess_ensemble_fanout_result,
    prepare_ensemble_fanout_request,
    run_ensemble_fanout_batch,
)

_MODEL_RESOURCE: tuple[Any, Any] | None = None


def _ensure_process_model_resource(model_name: str = "fcn") -> tuple[Any, Any]:
    global _MODEL_RESOURCE
    if _MODEL_RESOURCE is None:
        _MODEL_RESOURCE = _load_model(model_name)
    return _MODEL_RESOURCE


def _move_process_model_to_device(
    ctx: dict[str, Any], model_name: str = "fcn"
) -> tuple[Any, Any]:
    global _MODEL_RESOURCE
    device, _ = _select_device(ctx)
    package, model = _ensure_process_model_resource(model_name)
    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    _MODEL_RESOURCE = (package, model)
    return _MODEL_RESOURCE


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    _ensure_process_model_resource()
    return {"model_names": ["FCN"]}


class EnsembleFanoutWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["FCN"]
    input_model = EnsembleFanoutInput
    output_model = EnsembleFanoutBatchOutput

    def __init__(self) -> None:
        self._package: Any | None = None
        self._model: Any | None = None

    def _ensure_model_resource(self, model_name: str = "fcn") -> tuple[Any, Any]:
        if self._model is None:
            self._package, self._model = _ensure_process_model_resource(model_name)
        return self._package, self._model

    def warmup(self, ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._package, self._model = _move_process_model_to_device(ctx)
        return {"model_names": ["FCN"]}

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request: RawRequest, ctx: PrepareContext) -> dict[str, Any]:
        return prepare_ensemble_fanout_request(request, ctx)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        operation = str(ctx.get("operation") or "").strip()
        if operation == "materialize_perturbations":
            inputs = EnsembleFanoutInput(**dict(ctx.get("parameters") or {}))
            return materialize_ensemble_fanout_perturbations(
                inputs,
                build_execution_context(ctx),
                model_resource=self._ensure_model_resource(inputs.model),
            )
        return super().execute(ctx)

    # Main execution hook.
    def run(
        self, inputs: EnsembleFanoutInput, ctx: ExecutionContext
    ) -> EnsembleFanoutBatchOutput:
        return run_ensemble_fanout_batch(
            inputs,
            ctx,
            model_resource=self._ensure_model_resource(inputs.model),
        )

    def cleanup(self) -> None:
        global _MODEL_RESOURCE
        cached_model = (
            _MODEL_RESOURCE[1] if _MODEL_RESOURCE is not None else self._model
        )
        cleanup_earth2_runtime_resources(cached_model)
        self._package = None
        self._model = None
        _MODEL_RESOURCE = None
        cleanup_python_and_torch_runtime()

    # Optional finalization hook.
    def postprocess(
        self, result: PriorResult[Any], ctx: PostprocessContext
    ) -> PostprocessOutcome[EnsembleFanoutOutput | dict[str, Any]]:
        return postprocess_ensemble_fanout_result(result, ctx)


WORKFLOW = EnsembleFanoutWorkflow
