# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from plugin_sdk import (
    ExecutionInfo,
    InputArtifact,
    OutputRef,
    OutputRegistry,
    PostprocessContext,
    PostprocessOutcome,
    PrepareContext,
    PrepareResult,
    PriorResult,
    RawRequest,
    default_run_dir,
    model_to_jsonable,
)

DEFAULT_PLUGIN_MANIFEST_NAME = "plugin.yaml"
ENABLED_PLUGIN_ID_ENV = "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
DEFAULT_CONTENT_TYPE = "application/json"
DEFAULT_PIPELINE_PROFILE = "simple"
DEFAULT_RUNTIME_PROFILE = "python-test"
DEFAULT_EXECUTOR_CLASS = "default"
PIPELINE_PROFILE_ALIASES = {
    "default": "prefetch",
}
PARENT_TERMINAL_PREFIX = os.environ.get(
    "PHYSICSNEMO_SERVE_PARENT_TERMINAL_PREFIX", "parent_terminal"
)
PARENT_TERMINAL_POLL_INTERVAL_SECONDS = 0.25
PARENT_TERMINAL_POLL_WORKERS = 2
PARENT_TERMINAL_POLL_QUEUE_SIZE = 64
logger = logging.getLogger(__name__)
PHASE_EXECUTOR_FIELDS = {
    "prepare": "prepare_executor_class",
    "postprocess": "postprocess_executor_class",
    "readiness": "readiness_executor_class",
}

RUNTIME_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    DEFAULT_RUNTIME_PROFILE: {
        "runtime": {
            "kind": "python",
            "entrypoint": "workflow.py",
            "executor_class": "python.test",
            "hook_timeout_seconds": {"prepare": 15, "execute": 300, "postprocess": 300},
        },
        "resources": {
            "defaults": {
                "gpus_required": 0,
                "memory_mb": 1024,
            }
        },
        "outputs": {
            "primary_artifact": {
                "name": "primary",
                "media_type": "application/json",
            },
            "retention_hours": 24,
        },
        "developer": {
            "readiness": {
                "python_modules": [],
                "env": [],
                "paths": [],
            }
        },
    },
}

PIPELINE_RECOMMENDED_CHECK_PHASE = {
    "simple": "execute",
    "default": "prepare",
    "prefetch": "prepare",
    "postprocess": "prepare",
    "batch": "prepare",
    "ensemble": "prepare",
}
SUPPORTED_PIPELINE_STAGE_HANDLERS = {
    ("prepare", "plugin_phase"),
    ("prefetch", "prefetch"),
    ("fanout", "fanout"),
    ("schedule", "schedule"),
    ("execute", "plugin_phase"),
    ("collect", "collect"),
    ("postprocess", "plugin_phase"),
    ("publish", "plugin_phase"),
    ("publish", "publish_outputs"),
    ("results", "persist_results"),
}


def _supported_runtime_profiles() -> str:
    return ", ".join(sorted(RUNTIME_PROFILE_DEFAULTS))


def _canonical_pipeline_profile(profile_name: str) -> str:
    return PIPELINE_PROFILE_ALIASES.get(profile_name, profile_name)


def _execute_queue(executor_class: str) -> str:
    return f"execute.{executor_class}"


def _default_runtime_executor(runtime: dict[str, Any]) -> str:
    explicit = str(runtime.get("executor_class") or "").strip()
    if explicit:
        return explicit
    return DEFAULT_EXECUTOR_CLASS


def _runtime_profile_defaults(profile_name: str) -> dict[str, Any]:
    defaults = RUNTIME_PROFILE_DEFAULTS.get(profile_name)
    if defaults is None:
        raise ValueError(
            f"Plugin manifest runtime.profile '{profile_name}' is not supported. "
            f"Supported profiles: {_supported_runtime_profiles()}"
        )
    return copy.deepcopy(defaults)


def _resolve_runtime_profile_executor(profile_name: str) -> str:
    defaults = _runtime_profile_defaults(profile_name)
    return str(defaults["runtime"]["executor_class"])


def _build_pipeline_phases(profile_name: str, options: dict[str, Any]) -> list[str]:
    profile_name = _canonical_pipeline_profile(profile_name)
    postprocess_enabled = bool(options.get("postprocess"))

    if profile_name == "simple":
        phases = ["prepare", "execute"]
    elif profile_name == "prefetch":
        phases = ["prepare", "prefetch", "schedule", "execute"]
    elif profile_name == "postprocess":
        phases = ["prepare", "schedule", "execute", "postprocess"]
    elif profile_name == "batch":
        phases = ["prepare", "schedule", "execute"]
    elif profile_name == "ensemble":
        phases = ["prepare"]
        prefetch_scope = options.get("prefetch")
        if prefetch_scope in (True, "parent"):
            phases.append("prefetch")
        elif prefetch_scope not in (None, False):
            raise ValueError(
                "Plugin manifest pipeline.options.prefetch must be true, false, or 'parent'"
            )
        phases.extend(["fanout", "schedule", "execute", "collect"])
        if postprocess_enabled:
            phases.append("postprocess")
    else:
        supported = ", ".join(sorted(PIPELINE_RECOMMENDED_CHECK_PHASE))
        raise ValueError(
            f"Plugin manifest pipeline.profile '{profile_name}' is not supported. "
            f"Supported profiles: {supported}"
        )

    if postprocess_enabled and "postprocess" not in phases:
        phases.append("postprocess")

    phases.append("results")
    return phases


def _stage_definition(phase: str, execute_queue: str) -> dict[str, Any]:
    if phase == "prepare":
        return {
            "id": "prepare",
            "phase": "prepare",
            "handler": "plugin_phase",
            "queue": "prepare",
        }
    if phase == "prefetch":
        return {
            "id": "prefetch",
            "phase": "prefetch",
            "handler": "prefetch",
            "queue": "prefetch",
        }
    if phase == "fanout":
        return {
            "id": "fanout",
            "phase": "fanout",
            "handler": "fanout",
            "queue": "fanout",
        }
    if phase == "schedule":
        return {
            "id": "schedule",
            "phase": "schedule",
            "handler": "schedule",
            "queue": "schedule",
        }
    if phase == "execute":
        return {
            "id": "execute",
            "phase": "execute",
            "handler": "plugin_phase",
            "queue": execute_queue,
        }
    if phase == "collect":
        return {
            "id": "collect",
            "phase": "collect",
            "handler": "collect",
            "queue": "collect",
        }
    if phase == "postprocess":
        return {
            "id": "postprocess",
            "phase": "postprocess",
            "handler": "plugin_phase",
            "queue": "postprocess",
        }
    if phase == "results":
        return {
            "id": "results",
            "phase": "results",
            "handler": "persist_results",
            "queue": "results",
        }
    raise ValueError(f"Unsupported pipeline phase '{phase}'")


def _build_pipeline_stages(
    profile_name: str, options: dict[str, Any], execute_queue: str
) -> list[dict[str, Any]]:
    phases = _build_pipeline_phases(profile_name, options)
    stages = [_stage_definition(phase, execute_queue) for phase in phases]
    for index, stage in enumerate(stages):
        stage["next"] = stages[index + 1]["id"] if index + 1 < len(stages) else None
    return stages


def ensure_plugin_support_path() -> None:
    scripts_dir = str(SCRIPT_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def plugin_dirs() -> list[Path]:
    raw = os.environ.get("PLUGIN_DIR") or ""
    dirs: list[Path] = []
    for item in raw.replace(",", os.pathsep).split(os.pathsep):
        candidate = item.strip()
        if candidate:
            dirs.append(Path(candidate).expanduser().resolve())
    return dirs


def enabled_plugin_id() -> str | None:
    value = os.environ.get(ENABLED_PLUGIN_ID_ENV, "").strip()
    return value or None


def ensure_workflow_enabled(workflow_id: str) -> None:
    enabled_id = enabled_plugin_id()
    if enabled_id is not None and workflow_id != enabled_id:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' is disabled by {ENABLED_PLUGIN_ID_ENV}='{enabled_id}'"
        )


def load_plugin_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Plugin manifest must be a mapping: {manifest_path}")
    return expand_plugin_manifest(manifest)


def _validate_pipeline_stage_handlers(stages: Any) -> None:
    if not isinstance(stages, list):
        raise ValueError("Plugin manifest pipeline.stages must be an array")
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Plugin manifest pipeline stages must be objects")
        phase = str(stage.get("phase") or "").strip()
        handler = str(stage.get("handler") or "").strip()
        if (phase, handler) not in SUPPORTED_PIPELINE_STAGE_HANDLERS:
            stage_id = str(stage.get("id") or "").strip()
            raise ValueError(
                f"Plugin pipeline stage '{stage_id}' uses unsupported "
                f"phase/handler combination '{phase}/{handler}'"
            )


def expand_plugin_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expanded = copy.deepcopy(manifest)

    metadata = expanded.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Plugin manifest metadata must be an object")

    pipeline = expanded.setdefault("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("Plugin manifest pipeline must be an object")

    ingress = expanded.setdefault("ingress", {})
    if not isinstance(ingress, dict):
        raise ValueError("Plugin manifest ingress must be an object")

    runtime = expanded.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("Plugin manifest runtime must be an object")

    _normalize_ingress_aliases(ingress)

    runtime_profile = str(runtime.get("profile") or "").strip()
    if runtime_profile:
        _merge_missing(expanded, _runtime_profile_defaults(runtime_profile))

    runtime = expanded["runtime"]
    runtime.setdefault("executor_class", _default_runtime_executor(runtime))
    phase_profiles = runtime.get("phases")
    if phase_profiles is not None:
        if not isinstance(phase_profiles, dict):
            raise ValueError("Plugin manifest runtime.phases must be an object")
        for phase_name, field_name in PHASE_EXECUTOR_FIELDS.items():
            profile_name = str(phase_profiles.get(phase_name) or "").strip()
            if not profile_name or field_name in runtime:
                continue
            runtime[field_name] = _resolve_runtime_profile_executor(profile_name)

    pipeline_profile = str(pipeline.get("profile") or "").strip()
    if pipeline_profile and "stages" not in pipeline:
        options = pipeline.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("Plugin manifest pipeline.options must be an object")
        execute_queue = _execute_queue(_default_runtime_executor(runtime))
        pipeline["stages"] = _build_pipeline_stages(
            pipeline_profile, options, execute_queue
        )
    if "stages" in pipeline:
        _validate_pipeline_stage_handlers(pipeline["stages"])

    runtime.setdefault("kind", "python")
    runtime.setdefault("entrypoint", "workflow.py")

    outputs = expanded.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Plugin manifest outputs must be an object")
    outputs.setdefault(
        "primary_artifact",
        {
            "name": "primary",
            "media_type": "application/json",
        },
    )
    outputs.setdefault("retention_hours", 24)

    developer = expanded.setdefault("developer", {})
    if not isinstance(developer, dict):
        raise ValueError("Plugin manifest developer must be an object")
    readiness = developer.setdefault("readiness", {})
    if readiness is None:
        readiness = {}
        developer["readiness"] = readiness
    if not isinstance(readiness, dict):
        raise ValueError("Plugin manifest developer.readiness must be an object")

    configuration = expanded.get("configuration")
    if "configuration" in expanded and not isinstance(configuration, dict):
        raise ValueError("Plugin manifest configuration must be an object")

    if pipeline_profile:
        readiness.setdefault(
            "recommended_check_phase",
            PIPELINE_RECOMMENDED_CHECK_PHASE.get(
                _canonical_pipeline_profile(pipeline_profile),
                "prepare",
            ),
        )
    return expanded


def _normalize_ingress_aliases(ingress: dict[str, Any]) -> None:
    content_type = str(ingress.get("content_type") or "").strip()
    content_types = ingress.get("content_types")
    if content_types is not None:
        if not isinstance(content_types, list):
            raise ValueError("Plugin manifest ingress.content_types must be an array")
        if not all(
            isinstance(content_type, str) and content_type.strip()
            for content_type in content_types
        ):
            raise ValueError(
                "Plugin manifest ingress.content_types entries must be non-empty strings"
            )
    if not content_types:
        if content_type:
            ingress["content_types"] = [content_type]
        elif (
            ingress.get("form_schema")
            or ingress.get("form_schema_inline")
            or ingress.get("files")
        ):
            ingress["content_types"] = ["multipart/form-data"]
        else:
            ingress["content_types"] = [DEFAULT_CONTENT_TYPE]

    normalized_content_type = _normalized_ingress_content_type(ingress)
    request_schema = ingress.get("request_schema")
    if request_schema is not None:
        target_key = (
            "form_schema"
            if normalized_content_type == "multipart/form-data"
            else "json_schema"
        )
        ingress.setdefault(target_key, request_schema)

    request_schema_inline = ingress.get("request_schema_inline")
    if request_schema_inline is not None:
        target_key = (
            "form_schema_inline"
            if normalized_content_type == "multipart/form-data"
            else "json_schema_inline"
        )
        ingress.setdefault(target_key, request_schema_inline)

    if "operations" not in ingress and "operation" in ingress:
        operation = ingress["operation"]
        if isinstance(operation, str):
            ingress["operations"] = {"default": operation, "allowed": [operation]}
        else:
            ingress["operations"] = copy.deepcopy(operation)

    content_types = ingress.get("content_types")
    if isinstance(content_types, list) and content_types:
        ingress.setdefault("default_content_type", content_types[0])
    ingress.setdefault("operations", {"default": "run", "allowed": ["run"]})
    if isinstance(ingress.get("operations"), dict):
        operations = ingress["operations"]
        operations.setdefault("default", "run")
        operations.setdefault("allowed", [operations["default"]])


def _normalized_ingress_content_type(ingress: dict[str, Any]) -> str:
    content_types = ingress.get("content_types")
    if isinstance(content_types, list) and content_types:
        return str(content_types[0]).strip() or DEFAULT_CONTENT_TYPE
    return DEFAULT_CONTENT_TYPE


def _append_unique_readiness_modules(
    readiness: dict[str, Any], modules: list[str]
) -> None:
    current = readiness.setdefault("python_modules", [])
    if not isinstance(current, list):
        raise ValueError(
            "Plugin manifest developer.readiness.python_modules must be a list"
        )
    seen = {str(module) for module in current}
    for module in modules:
        if module not in seen:
            current.append(module)
            seen.add(module)


def _merge_missing(target: Any, defaults: Any) -> Any:
    if isinstance(target, dict) and isinstance(defaults, dict):
        for key, default_value in defaults.items():
            if key not in target:
                target[key] = copy.deepcopy(default_value)
                continue
            target[key] = _merge_missing(target[key], default_value)
        return target

    if target is None:
        return copy.deepcopy(defaults)

    return target


def resolve_plugin_manifest(workflow_id: str) -> tuple[Path, dict[str, Any]]:
    ensure_workflow_enabled(workflow_id)

    for plugin_dir in plugin_dirs():
        child_manifest = plugin_dir / workflow_id / DEFAULT_PLUGIN_MANIFEST_NAME
        if child_manifest.is_file():
            manifest = load_plugin_manifest(child_manifest)
            manifest_id = manifest.get("metadata", {}).get("id")
            if manifest_id and manifest_id != workflow_id:
                raise ValueError(
                    f"Plugin manifest id mismatch for '{workflow_id}': found '{manifest_id}'"
                )
            return child_manifest.parent, manifest

        direct_manifest = plugin_dir / DEFAULT_PLUGIN_MANIFEST_NAME
        if direct_manifest.is_file():
            manifest = load_plugin_manifest(direct_manifest)
            if manifest.get("metadata", {}).get("id") == workflow_id:
                return plugin_dir, manifest

    raise ValueError(
        f"Plugin workflow '{workflow_id}' not found in plugin directories: {plugin_dirs()}"
    )


def load_plugin_module(
    workflow_id: str,
    entrypoint_path: Path,
    module_prefix: str = "physicsnemo_serve_plugin",
) -> Any:
    if not entrypoint_path.is_file():
        raise ValueError(
            f"Plugin workflow '{workflow_id}' entrypoint does not exist: {entrypoint_path}"
        )

    ensure_plugin_support_path()
    module_name = f"{module_prefix}_{workflow_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint_path)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' entrypoint could not be imported: {entrypoint_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_context(payload: dict[str, Any]) -> dict[str, Any]:
    services = payload.get("services", {})
    if not isinstance(services, dict):
        services = {}
    if "redis_url" not in services:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            services = {**services, "redis_url": redis_url}
    if "default_output_dir" not in services:
        default_output_dir = os.environ.get("DEFAULT_OUTPUT_DIR")
        if default_output_dir:
            services = {**services, "default_output_dir": default_output_dir}

    service_objects = payload.get("service_objects", {})
    if not isinstance(service_objects, dict):
        service_objects = {}

    parent_run_id = payload.get("parent_run_id")
    run_dir = _resolve_run_dir(payload, services)
    outputs = payload.get("outputs")
    if not isinstance(outputs, OutputRegistry):
        outputs = OutputRegistry(run_dir)

    return {
        "run_id": payload.get("run_id"),
        "batch_id": payload.get("batch_id"),
        "parent_run_id": parent_run_id,
        "workflow_id": payload.get("workflow_id"),
        "operation": payload.get("operation"),
        "parameters": payload.get("parameters", {}),
        "request": payload.get("request", {}),
        "resource_profile": payload.get("resource_profile"),
        "batch_profile": payload.get("batch_profile"),
        "batch_info": payload.get("batch_info"),
        "fanout_profile": payload.get("fanout_profile"),
        "fanout_item": payload.get("fanout_item"),
        "child_results": payload.get("child_results", []),
        "aggregation_summary": payload.get("aggregation_summary"),
        "items": payload.get("items", []),
        "prefetch_plan": payload.get("prefetch_plan", []),
        "prefetch_artifacts": payload.get("prefetch_artifacts", []),
        "stage_context": payload.get("stage_context", {}),
        "result": payload.get("result"),
        "runtime": payload.get("runtime", {}),
        "run_dir": run_dir,
        "outputs": outputs,
        "services": services,
        "service_objects": service_objects,
        "abort_requested": _build_abort_requested(service_objects, parent_run_id),
        "payload": payload,
    }


def build_raw_request(payload: dict[str, Any]) -> RawRequest:
    request = payload.get("request", {})
    if not isinstance(request, dict):
        request = {}

    content_type = str(request.get("content_type") or DEFAULT_CONTENT_TYPE)
    raw_fields = request.get("raw_fields")
    if content_type == DEFAULT_CONTENT_TYPE and isinstance(raw_fields, dict):
        wrapped_parameters = raw_fields.get("parameters")
        wrapper_only = all(key in {"parameters", "operation"} for key in raw_fields)
        if isinstance(wrapped_parameters, dict) and wrapper_only:
            payload_parameters = payload.get("parameters")
            if isinstance(payload_parameters, dict):
                raw_fields = payload_parameters
            else:
                raw_fields = wrapped_parameters
    if not isinstance(raw_fields, dict):
        raw_fields = payload.get("parameters", {})
    if not isinstance(raw_fields, dict):
        raw_fields = {}

    return RawRequest(
        content_type=content_type,
        operation=str(payload.get("operation") or request.get("operation") or "run"),
        raw_fields=raw_fields,
        input_artifacts=_coerce_input_artifacts(request.get("input_artifacts")),
    )


def build_prepare_context(payload: dict[str, Any]) -> PrepareContext:
    ctx = build_context(payload)
    return PrepareContext(
        run_id=str(ctx.get("run_id") or ""),
        workflow_id=str(ctx.get("workflow_id") or ""),
        run_dir=ctx["run_dir"],
        parent_run_id=str(ctx.get("parent_run_id") or "") or None,
        batch_id=str(ctx.get("batch_id") or "") or None,
        default_resource_profile=ctx.get("resource_profile"),
        services=ctx.get("services", {}),
        stage_context=ctx.get("stage_context", {}),
    )


def build_postprocess_context(payload: dict[str, Any]) -> PostprocessContext:
    ctx = build_context(payload)
    return PostprocessContext(
        run_id=str(ctx.get("run_id") or ""),
        run_dir=ctx["run_dir"],
        outputs=ctx["outputs"],
        request=build_raw_request(payload),
        resource_profile=ctx.get("resource_profile"),
        services=ctx.get("services", {}),
    )


def build_prior_result(payload: dict[str, Any]) -> PriorResult[Any]:
    raw_result = payload.get("result", {})
    if not isinstance(raw_result, dict):
        raw_result = {}

    result_payload = dict(raw_result)
    status = str(result_payload.pop("status", "succeeded"))
    run_id = str(result_payload.pop("run_id", payload.get("run_id") or "") or "")
    raw_outputs = result_payload.pop("outputs", None)
    artifacts = result_payload.pop("artifacts", raw_outputs)
    published_outputs = result_payload.pop("published_outputs", None)
    if published_outputs is None:
        published_outputs = result_payload.pop("published_artifacts", [])
    output_path = result_payload.pop("output_path", None)
    result_payload.pop("output_archive", None)
    result_payload.pop("batch_info", None)
    result_payload.pop("error", None)
    result_payload.pop("gpu_stream", None)
    result_payload.pop("workflow", None)
    execution_time = result_payload.pop("execution_time_seconds", None)

    outputs = _output_refs_from_artifacts(artifacts)
    if not outputs and output_path:
        outputs = [
            OutputRef(
                name="primary",
                media_type="application/octet-stream",
                path=str(output_path),
                primary=True,
            )
        ]

    primary_output = next((output for output in outputs if output.primary), None)
    if primary_output is None and outputs:
        primary_output = outputs[0]

    return PriorResult(
        payload=result_payload,
        execution=ExecutionInfo(
            run_id=run_id,
            status=status
            if status in {"succeeded", "failed", "cancelled"}
            else "succeeded",
            outputs=outputs,
            primary_output=primary_output,
            execution_time_seconds=execution_time
            if isinstance(execution_time, (int, float))
            else None,
            published_outputs=published_outputs
            if isinstance(published_outputs, list)
            else [],
        ),
    )


def serialize_prepare_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, PrepareResult):
        response: dict[str, Any] = {
            "parameters": model_to_jsonable(result.inputs),
        }
        if result.resource_profile is not None:
            response["resource_profile"] = model_to_jsonable(result.resource_profile)
        if result.prefetch_plan:
            response["prefetch_plan"] = model_to_jsonable(result.prefetch_plan)
        if result.batch_profile is not None:
            response["batch_profile"] = model_to_jsonable(result.batch_profile)
        if result.fanout_profile is not None:
            response["fanout_profile"] = model_to_jsonable(result.fanout_profile)
        if result.fanout_items:
            response["fanout_items"] = model_to_jsonable(result.fanout_items)
        return response
    if isinstance(result, dict):
        return result
    raise TypeError(
        f"Plugin prepare hook returned {type(result).__name__}, expected PrepareResult or dict"
    )


def serialize_postprocess_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, PostprocessOutcome):
        payload = model_to_jsonable(result.payload)
        if not isinstance(payload, dict):
            raise TypeError(
                "Plugin postprocess hook payload must serialize to a dict for the legacy adapter"
            )
        response = dict(payload)
        response["status"] = result.status
        if result.result_ops:
            response["result_ops"] = model_to_jsonable(result.result_ops)
        return response
    if isinstance(result, dict):
        return result
    raise TypeError(
        f"Plugin postprocess hook returned {type(result).__name__}, expected PostprocessOutcome or dict"
    )


def legacy_artifacts_from_outputs(
    outputs: OutputRegistry | Any,
) -> list[dict[str, Any]]:
    if outputs is None or not hasattr(outputs, "registered_outputs"):
        return []

    artifacts: list[dict[str, Any]] = []
    for output in outputs.registered_outputs():
        artifact = {
            "name": str(output.name),
            "media_type": str(output.media_type),
            "storage_path": str(output.path),
        }
        if bool(getattr(output, "primary", False)):
            artifact["primary"] = True
        artifacts.append(artifact)
    return artifacts


def primary_output_path_from_outputs(outputs: OutputRegistry | Any) -> str | None:
    if outputs is None:
        return None

    primary_output = (
        outputs.primary_output() if hasattr(outputs, "primary_output") else None
    )
    if primary_output is not None:
        return str(primary_output.path)

    if not hasattr(outputs, "registered_outputs"):
        return None
    registered_outputs = outputs.registered_outputs()
    if not registered_outputs:
        return None
    return str(registered_outputs[0].path)


def merge_registered_outputs_into_result(
    result: dict[str, Any], outputs: OutputRegistry | Any
) -> dict[str, Any]:
    merged = dict(result)
    registered_outputs = legacy_artifacts_from_outputs(outputs)
    if registered_outputs and not merged.get("artifacts"):
        merged["artifacts"] = registered_outputs

    if merged.get("output_path") in (None, ""):
        primary_output_path = primary_output_path_from_outputs(outputs)
        if primary_output_path is not None:
            merged["output_path"] = primary_output_path

    return merged


def _resolve_run_dir(payload: dict[str, Any], services: dict[str, Any]) -> Path:
    explicit_run_dir = payload.get("run_dir")
    if explicit_run_dir:
        return Path(str(explicit_run_dir))

    default_output_dir = services.get("default_output_dir")
    if default_output_dir:
        root = Path(str(default_output_dir))
        run_id = payload.get("run_id") or payload.get("batch_id") or "unknown-run"
        return root / str(run_id)

    run_id = payload.get("run_id") or payload.get("batch_id") or "unknown-run"
    return default_run_dir(run_id)


def _coerce_input_artifacts(raw_artifacts: Any) -> list[InputArtifact]:
    if not isinstance(raw_artifacts, list):
        return []

    artifacts: list[InputArtifact] = []
    for raw_artifact in raw_artifacts:
        if isinstance(raw_artifact, InputArtifact):
            artifacts.append(raw_artifact)
            continue
        if not isinstance(raw_artifact, dict):
            continue
        artifacts.append(
            InputArtifact(
                name=str(raw_artifact.get("name") or "input"),
                media_type=str(
                    raw_artifact.get("media_type") or "application/octet-stream"
                ),
                storage_path=str(
                    raw_artifact.get("storage_path") or raw_artifact.get("path") or ""
                ),
                original_filename=str(raw_artifact["original_filename"])
                if raw_artifact.get("original_filename") is not None
                else None,
            )
        )
    return artifacts


def _output_refs_from_artifacts(raw_artifacts: Any) -> list[OutputRef]:
    if not isinstance(raw_artifacts, list):
        return []

    outputs: list[OutputRef] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if isinstance(raw_artifact, OutputRef):
            outputs.append(raw_artifact)
            continue
        if not isinstance(raw_artifact, dict):
            continue
        outputs.append(
            OutputRef(
                name=str(raw_artifact.get("name") or f"output-{index}"),
                media_type=str(
                    raw_artifact.get("media_type") or "application/octet-stream"
                ),
                path=str(
                    raw_artifact.get("storage_path") or raw_artifact.get("path") or ""
                ),
                primary=bool(raw_artifact.get("primary")) or index == 0,
            )
        )
    return outputs


def get_workflow_schema_source(module: Any, workflow_id: str) -> Any:
    workflow_obj = getattr(module, "WORKFLOW", None)
    if workflow_obj is not None:
        return workflow_obj

    factory = getattr(module, "build_workflow", None)
    if callable(factory):
        built = factory()
        if built is not None:
            return built

    return get_workflow_instance(module, workflow_id)


def get_prepare_workflow(module: Any, workflow_id: str) -> Any:
    workflow_obj = getattr(module, "WORKFLOW", None)
    if workflow_obj is not None:
        if inspect.isclass(workflow_obj):
            return object.__new__(workflow_obj)
        if _has_phase_method(workflow_obj):
            return _instantiate_exported_workflow(workflow_obj, workflow_id)
        if callable(workflow_obj):
            workflow_obj = workflow_obj()
            if workflow_obj is not None:
                return workflow_obj

    factory = getattr(module, "build_workflow", None)
    if callable(factory):
        built = factory()
        if built is not None:
            return built

    return get_workflow_instance(module, workflow_id)


def get_workflow_instance(module: Any, workflow_id: str) -> Any:
    workflow_obj = getattr(module, "WORKFLOW", None)
    if workflow_obj is not None:
        if inspect.isclass(workflow_obj):
            return workflow_obj()
        if _has_phase_method(workflow_obj):
            return _instantiate_exported_workflow(workflow_obj, workflow_id)
        if callable(workflow_obj):
            workflow_obj = workflow_obj()
            if workflow_obj is not None:
                return workflow_obj

    factory = getattr(module, "build_workflow", None)
    if callable(factory):
        workflow_obj = factory()
        if workflow_obj is not None:
            return workflow_obj

    if workflow_obj is None:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' entrypoint does not define module-level hooks, "
            "WORKFLOW, or build_workflow()"
        )

    return workflow_obj


def _instantiate_exported_workflow(workflow_obj: Any, workflow_id: str) -> Any:
    workflow_class = workflow_obj.__class__
    try:
        return workflow_class()
    except TypeError as exc:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' exports a pre-instantiated WORKFLOW that "
            "cannot be recreated per request. Export the workflow class or "
            "build_workflow() instead."
        ) from exc


def _wrap_workflow_hook(
    workflow: Any,
    hook: Callable[..., Any],
    workflow_id: str,
    cleanup_method: str = "cleanup",
) -> Callable[..., Any]:
    def invoke(*args: Any, **kwargs: Any) -> Any:
        try:
            return hook(*args, **kwargs)
        finally:
            cleanup = getattr(workflow, cleanup_method, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning(
                        "Plugin workflow cleanup failed for %s: %s",
                        workflow_id,
                        exc,
                    )
                    logger.debug(
                        "Cleanup traceback for %s follows",
                        workflow_id,
                        exc_info=True,
                    )

    return invoke


def workflow_is_cacheable(workflow: Any) -> bool:
    cache_scope = str(getattr(workflow, "cache_scope", "") or "").strip().lower()
    return cache_scope == "process" or bool(getattr(workflow, "cache_models", False))


def resolve_workflow_hook(
    workflow: Any,
    workflow_id: str,
    phase: str,
    *,
    cleanup_method: str = "cleanup",
) -> Callable[..., Any]:
    hook = getattr(workflow, phase, None)
    if callable(hook):
        if phase == "prepare":
            return hook
        return _wrap_workflow_hook(
            workflow,
            hook,
            workflow_id,
            cleanup_method=cleanup_method,
        )

    raise ValueError(
        f"Plugin workflow '{workflow_id}' workflow object does not expose {phase}(ctx)"
    )


def resolve_phase_hook(
    module: Any, workflow_id: str, phase: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    direct = getattr(module, phase, None)
    if callable(direct):
        return direct

    workflow = (
        get_prepare_workflow(module, workflow_id)
        if phase == "prepare"
        else get_workflow_instance(module, workflow_id)
    )
    hook = getattr(workflow, phase, None)
    if callable(hook):
        if phase == "prepare":
            return hook
        return _wrap_workflow_hook(workflow, hook, workflow_id)

    raise ValueError(
        f"Plugin workflow '{workflow_id}' entrypoint does not define {phase}(ctx), "
        "nor does its workflow object expose that method"
    )


def plugin_phases_from_manifest(manifest: dict[str, Any]) -> list[str]:
    phases: list[str] = []
    for stage in manifest.get("pipeline", {}).get("stages", []):
        if stage.get("handler") != "plugin_phase":
            continue
        phase = str(stage.get("phase") or "").strip()
        if phase and phase not in phases:
            phases.append(phase)
    return phases


def read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_phase_method(candidate: Any) -> bool:
    return any(
        callable(getattr(candidate, phase, None))
        for phase in ("prepare", "execute", "execute_batch", "postprocess", "collect")
    )


class _DaemonPollPool:
    """Small bounded daemon pool for Redis cancellation probes."""

    def __init__(self) -> None:
        self._tasks: queue.Queue[Callable[[], None]] = queue.Queue(
            maxsize=PARENT_TERMINAL_POLL_QUEUE_SIZE
        )
        for index in range(PARENT_TERMINAL_POLL_WORKERS):
            threading.Thread(
                target=self._run,
                name=f"plugin-parent-terminal-poll-{index}",
                daemon=True,
            ).start()

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except Exception:
                logger.exception("Unhandled parent-terminal poll failure")
            finally:
                self._tasks.task_done()

    def submit(self, task: Callable[[], None]) -> bool:
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            return False
        return True


_PARENT_TERMINAL_POLL_POOL: _DaemonPollPool | None = None
_PARENT_TERMINAL_POLL_POOL_LOCK = threading.Lock()


def _parent_terminal_poll_pool() -> _DaemonPollPool:
    global _PARENT_TERMINAL_POLL_POOL
    with _PARENT_TERMINAL_POLL_POOL_LOCK:
        if _PARENT_TERMINAL_POLL_POOL is None:
            _PARENT_TERMINAL_POLL_POOL = _DaemonPollPool()
        return _PARENT_TERMINAL_POLL_POOL


class _AbortRequested:
    """Nonblocking composite cancellation check for synchronous plugin code."""

    def __init__(self, service_objects: dict[str, Any], parent_run_id: Any) -> None:
        self._worker_shutdown_event = service_objects.get("worker_shutdown_event")
        self._parent_terminal_event = threading.Event()
        self._redis_client = service_objects.get("redis_client")
        parent = str(parent_run_id or "").strip()
        self._terminal_key = f"{PARENT_TERMINAL_PREFIX}:{parent}" if parent else None
        self._poll_lock = threading.Lock()
        self._poll_in_flight = False
        self._next_poll_at = 0.0
        self._initial_poll_complete = threading.Event()

    def _worker_shutdown_requested(self) -> bool:
        is_set = getattr(self._worker_shutdown_event, "is_set", None)
        if not callable(is_set):
            return False
        try:
            return bool(is_set())
        except Exception:
            return False

    def _poll_parent_terminal(self) -> None:
        try:
            if bool(self._redis_client.exists(self._terminal_key)):
                self._parent_terminal_event.set()
        except Exception as exc:
            logger.debug("Failed to poll parent terminal state: %s", exc)
        finally:
            self._initial_poll_complete.set()
            with self._poll_lock:
                self._poll_in_flight = False
                self._next_poll_at = (
                    time.monotonic() + PARENT_TERMINAL_POLL_INTERVAL_SECONDS
                )

    def _start_parent_poll_if_due(self) -> None:
        if (
            self._terminal_key is None
            or self._redis_client is None
            or not hasattr(self._redis_client, "exists")
        ):
            return

        now = time.monotonic()
        with self._poll_lock:
            if self._poll_in_flight or now < self._next_poll_at:
                return
            self._poll_in_flight = True

        if not _parent_terminal_poll_pool().submit(self._poll_parent_terminal):
            self._initial_poll_complete.set()
            with self._poll_lock:
                self._poll_in_flight = False
                self._next_poll_at = (
                    time.monotonic() + PARENT_TERMINAL_POLL_INTERVAL_SECONDS
                )

    def __call__(self) -> bool:
        if self._worker_shutdown_requested() or self._parent_terminal_event.is_set():
            return True
        self._start_parent_poll_if_due()
        return self._worker_shutdown_requested() or self._parent_terminal_event.is_set()

    def wait_for_initial_poll(self, timeout_seconds: float) -> bool:
        """Wait briefly for the first parent check while remaining shutdown-aware."""
        if self():
            return True
        if (
            self._terminal_key is None
            or self._redis_client is None
            or not hasattr(self._redis_client, "exists")
        ):
            return False

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while not self._initial_poll_complete.is_set():
            if self._worker_shutdown_requested():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self()
            self._initial_poll_complete.wait(min(remaining, 0.01))
        return self()


def _build_abort_requested(
    service_objects: dict[str, Any], parent_run_id: Any
) -> Callable[[], bool]:
    return _AbortRequested(service_objects, parent_run_id)
