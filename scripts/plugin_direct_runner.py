#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from plugin_runtime import (  # noqa: E402
    build_context,
    build_postprocess_context,
    build_prepare_context,
    build_prior_result,
    build_raw_request,
    legacy_artifacts_from_outputs,
    load_plugin_manifest,
    load_plugin_module,
    merge_registered_outputs_into_result,
    resolve_phase_hook,
    serialize_postprocess_result,
    serialize_prepare_result,
)
from plugin_sdk import OutputRegistry  # noqa: E402


SUPPORTED_STAGE_HANDLERS = {
    ("prepare", "plugin_phase"),
    ("prefetch", "prefetch"),
    ("batch", "batch"),
    ("schedule", "schedule"),
    ("execute", "plugin_phase"),
    ("postprocess", "plugin_phase"),
    ("results", "persist_results"),
}
PREPARE_FIELDS = (
    "operation",
    "parameters",
    "request",
    "resource_profile",
    "batch_profile",
    "prefetch_plan",
    "fanout_profile",
    "fanout_items",
)
RESULT_METADATA_FIELDS = {
    "artifacts",
    "error",
    "error_traceback",
    "execution_time_seconds",
    "output_path",
    "run_id",
    "status",
    "workflow",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one manifest-driven plugin inference without service processes."
    )
    parser.add_argument("--plugin-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    # Plugin imports and hooks may print progress or library diagnostics.
    # Keep fd 1 redirected through process shutdown so buffered native output
    # cannot be appended to the JSON protocol response while exiting.
    with _redirect_process_stdout_to_stderr() as protocol_stdout:
        try:
            result = run_plugin(
                Path(args.plugin_root).expanduser().resolve(),
                Path(args.request).expanduser().resolve(),
                Path(args.output_dir).expanduser().resolve(),
                args.run_id,
            )
            encoded = json.dumps(result)
            return_code = 0
        except Exception as exc:
            error_result = {
                "run_id": args.run_id,
                "status": "failed",
                "workflow": None,
                "request": None,
                "execution": {
                    "outputs": [],
                    "output_path": None,
                    "error": str(exc),
                },
                "payload": {},
            }
            encoded = json.dumps(error_result)
            print(str(exc), file=sys.stderr)
            return_code = 1

        _write_protocol_response(protocol_stdout, encoded)
        return return_code


@contextmanager
def _redirect_process_stdout_to_stderr():
    """Redirect fd 1 until process exit and yield its original descriptor."""
    sys.stdout.flush()
    saved_stdout: int | None = None
    try:
        saved_stdout = os.dup(1)
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield saved_stdout
    finally:
        if saved_stdout is not None:
            os.close(saved_stdout)


def _write_protocol_response(protocol_stdout: int, encoded: str) -> None:
    """Write the single JSON response through the saved original stdout."""
    remaining = memoryview(f"{encoded}\n".encode())
    while remaining:
        written = os.write(protocol_stdout, remaining)
        if written == 0:
            raise BrokenPipeError("failed to write plugin protocol response")
        remaining = remaining[written:]


def run_plugin(
    plugin_root: Path,
    request_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    manifest_path = plugin_root / "plugin.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Plugin manifest not found: {manifest_path}")
    if not request_path.is_file():
        raise ValueError(f"Request JSON not found: {request_path}")
    if not run_id.strip():
        raise ValueError("Run ID must be non-empty")

    manifest = load_plugin_manifest(manifest_path)
    workflow_id = str(manifest.get("metadata", {}).get("id") or "").strip()
    if not workflow_id:
        raise ValueError("Plugin manifest is missing metadata.id")

    runtime = manifest.get("runtime", {})
    if not isinstance(runtime, dict) or runtime.get("kind") != "python":
        raise ValueError("Direct inference supports only runtime.kind 'python'")
    entrypoint = str(runtime.get("entrypoint") or "").strip()
    if not entrypoint:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
        )

    stages = manifest.get("pipeline", {}).get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"Plugin workflow '{workflow_id}' has no pipeline stages")
    _validate_pipeline(stages)

    request_body = _read_request(request_path)
    operation, parameters = _normalize_request(manifest, request_body)
    _validate_request(plugin_root, manifest, parameters)

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_registry = OutputRegistry(run_dir)
    resource_profile = _resource_profile(manifest)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "operation": operation,
        "request": {
            "content_type": "application/json",
            "operation": operation,
            "raw_fields": parameters,
            "input_artifacts": [],
        },
        "parameters": parameters,
        "resource_profile": resource_profile,
        "prefetch_plan": [],
        "prefetch_artifacts": [],
        "stage_context": {
            "current_stage_id": stages[0].get("id"),
            "current_phase": stages[0].get("phase"),
            "pipeline": stages,
        },
        "result": None,
        "runtime": runtime,
        "run_dir": str(run_dir),
        "outputs": output_registry,
    }

    plugin_path = str(plugin_root)
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)

    os.environ["DEFAULT_OUTPUT_DIR"] = str(output_dir)
    os.environ["PLUGIN_DIR"] = str(plugin_root)
    os.environ["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"] = workflow_id
    module = load_plugin_module(
        workflow_id,
        plugin_root / entrypoint,
        module_prefix="physicsnemo_serve_direct_plugin",
    )
    prefetch_stats: dict[str, Any] | None = None
    stage_by_id = {
        str(stage.get("id")): stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("id")
    }
    current_stage = stages[0]
    visited: set[str] = set()

    while current_stage is not None:
        stage_id = str(current_stage.get("id") or "")
        phase = str(current_stage.get("phase") or "")
        if stage_id in visited:
            raise ValueError(f"Plugin pipeline contains a cycle at stage '{stage_id}'")
        visited.add(stage_id)
        payload["stage_context"]["current_stage_id"] = stage_id
        payload["stage_context"]["current_phase"] = phase

        next_stage_id = current_stage.get("next")
        if phase == "prepare":
            result = _invoke_phase(module, workflow_id, phase, payload)
            for field in PREPARE_FIELDS:
                if field in result:
                    payload[field] = result[field]
            next_stage_id = result.get("next_stage_id", next_stage_id)
        elif phase == "prefetch":
            materialized = _materialize_prefetch(
                payload.get("prefetch_plan", []),
                output_dir / ".prefetch-cache",
                run_id,
            )
            payload["prefetch_artifacts"] = materialized["artifacts"]
            prefetch_stats = materialized["stats"]
        elif phase in {"batch", "schedule"}:
            pass
        elif phase == "execute":
            payload["result"] = _invoke_phase(module, workflow_id, phase, payload)
        elif phase == "postprocess":
            payload["result"] = _invoke_phase(module, workflow_id, phase, payload)
        elif phase == "results":
            break
        else:
            raise ValueError(
                f"Direct inference does not support pipeline phase '{phase}'"
            )

        current_stage = (
            stage_by_id.get(str(next_stage_id)) if next_stage_id is not None else None
        )
        if next_stage_id is not None and current_stage is None:
            raise ValueError(
                f"Pipeline stage '{stage_id}' selects unknown next stage "
                f"'{next_stage_id}'"
            )

    final_result = payload.get("result")
    if not isinstance(final_result, dict):
        raise ValueError(
            f"Plugin workflow '{workflow_id}' did not produce a result object"
        )
    return _build_result_envelope(
        payload,
        request_body=request_body,
        prefetch_stats=prefetch_stats,
    )


def _validate_pipeline(stages: list[Any]) -> None:
    stage_ids: set[str] = set()
    validated_stages: list[dict[str, Any]] = []
    for raw_stage in stages:
        if not isinstance(raw_stage, dict):
            raise ValueError("Plugin pipeline stages must be objects")
        raw_stage_id = raw_stage.get("id")
        if not isinstance(raw_stage_id, str) or not raw_stage_id.strip():
            raise ValueError("Plugin pipeline stages must have non-empty ids")
        stage_id = raw_stage_id.strip()
        raw_stage["id"] = stage_id
        if stage_id in stage_ids:
            raise ValueError(
                f"Plugin pipeline contains duplicate stage id '{stage_id}'"
            )
        stage_ids.add(stage_id)
        phase = str(raw_stage.get("phase") or "")
        handler = str(raw_stage.get("handler") or "")
        if (phase, handler) not in SUPPORTED_STAGE_HANDLERS:
            raise ValueError(
                f"Direct inference does not support pipeline phase '{phase}'"
            )
        if phase == "results":
            if raw_stage.get("next") is not None:
                raise ValueError(
                    f"Plugin pipeline results stage '{stage_id}' must be terminal"
                )
        else:
            next_stage_id = raw_stage.get("next")
            if not isinstance(next_stage_id, str) or not next_stage_id.strip():
                raise ValueError(
                    f"Plugin pipeline stage '{stage_id}' must define a non-empty next stage"
                )
            raw_stage["next"] = next_stage_id.strip()
        validated_stages.append(raw_stage)

    for stage in validated_stages:
        if str(stage.get("phase") or "") == "results":
            continue
        next_stage_id = str(stage["next"]).strip()
        if next_stage_id not in stage_ids:
            raise ValueError(
                f"Plugin pipeline stage '{stage['id']}' references unknown next stage "
                f"'{next_stage_id}'"
            )

    stage_by_id = {str(stage["id"]): stage for stage in validated_stages}
    fully_visited: set[str] = set()
    for start_stage_id in stage_by_id:
        current_stage_id = start_stage_id
        current_path: set[str] = set()
        while current_stage_id not in fully_visited:
            if current_stage_id in current_path:
                raise ValueError(
                    f"Plugin pipeline contains a cycle at stage '{current_stage_id}'"
                )
            current_path.add(current_stage_id)
            current_stage = stage_by_id[current_stage_id]
            if str(current_stage.get("phase") or "") == "results":
                break
            current_stage_id = str(current_stage["next"]).strip()
        fully_visited.update(current_path)


def _read_request(request_path: Path) -> dict[str, Any]:
    with request_path.open("r", encoding="utf-8") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise ValueError("Direct inference request must be a JSON object")
    return request


def _normalize_request(
    manifest: dict[str, Any], request: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    ingress = manifest.get("ingress", {})
    content_types = ingress.get("content_types")
    if content_types is not None and not isinstance(content_types, list):
        raise ValueError("ingress.content_types must be an array")
    if isinstance(content_types, list) and not all(
        isinstance(content_type, str) and content_type.strip()
        for content_type in content_types
    ):
        raise ValueError("ingress.content_types entries must be non-empty strings")
    if not content_types:
        content_type = str(ingress.get("content_type") or "application/json").strip()
        content_types = [content_type]
    if "application/json" not in content_types:
        raise ValueError(
            "Direct inference currently supports only application/json ingress"
        )

    if "operations" in ingress:
        operations = ingress["operations"]
        operations_field = "ingress.operations"
    else:
        operations = ingress.get("operation", {})
        operations_field = "ingress.operation"
    if not isinstance(operations, (dict, str)):
        raise ValueError(f"{operations_field} must be a string or object")
    if isinstance(operations, str):
        if not operations.strip():
            raise ValueError(f"{operations_field} must be non-empty")
        default_operation = operations
        allowed_operations = [operations]
    else:
        default_operation = operations.get("default", "run")
        if not isinstance(default_operation, str) or not default_operation.strip():
            raise ValueError(f"{operations_field}.default must be a non-empty string")
        allowed = operations.get("allowed", [])
        if not isinstance(allowed, list) or not all(
            isinstance(value, str) and value.strip() for value in allowed
        ):
            raise ValueError(
                f"{operations_field}.allowed must be an array of non-empty strings"
            )
        allowed_operations = allowed
        if allowed_operations and default_operation not in allowed_operations:
            raise ValueError(
                f"{operations_field}.default must be included in "
                f"{operations_field}.allowed"
            )

    operation = request.get("operation", default_operation)
    if (
        not isinstance(operation, str)
        or not operation.strip()
        or (allowed_operations and operation not in allowed_operations)
    ):
        raise ValueError(
            f"Unsupported operation '{operation}'; allowed operations: "
            f"{allowed_operations}"
        )

    parameters = request.get("parameters")
    if parameters is None:
        parameters = dict(request)
        parameters.pop("operation", None)
    if not isinstance(parameters, dict):
        raise ValueError("Direct inference parameters must be a JSON object")
    return operation, parameters


def _validate_request(
    plugin_root: Path, manifest: dict[str, Any], parameters: dict[str, Any]
) -> None:
    ingress = manifest.get("ingress", {})
    schema = ingress.get("json_schema_inline")
    if schema is None and ingress.get("json_schema"):
        schema_path = (plugin_root / str(ingress["json_schema"])).resolve()
        if not schema_path.is_relative_to(plugin_root):
            raise ValueError("Plugin JSON schema path escapes the plugin directory")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(
            "Direct inference requires ingress.json_schema or ingress.json_schema_inline"
        )

    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    try:
        validator_cls(schema).validate(parameters)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.path)
        suffix = f" at '{path}'" if path else ""
        raise ValueError(
            f"Request does not conform to schema{suffix}: {exc.message}"
        ) from exc


def _resource_profile(manifest: dict[str, Any]) -> dict[str, Any] | None:
    defaults = manifest.get("resources", {}).get("defaults", {})
    if not isinstance(defaults, dict) or not defaults:
        return None
    return {
        "executor_class": manifest.get("runtime", {}).get("executor_class"),
        **defaults,
    }


def _invoke_phase(
    module: Any,
    workflow_id: str,
    phase: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    hook = resolve_phase_hook(module, workflow_id, phase)
    if phase == "prepare" and _supports_explicit_contract(hook):
        result = serialize_prepare_result(
            hook(build_raw_request(payload), build_prepare_context(payload))
        )
    elif phase == "postprocess" and _supports_explicit_contract(hook):
        result = serialize_postprocess_result(
            hook(build_prior_result(payload), build_postprocess_context(payload))
        )
    else:
        ctx = build_context(payload)
        result = hook(ctx)
        if result is None:
            result = {}
        if phase == "prepare":
            result = serialize_prepare_result(result)
        elif phase == "postprocess":
            result = serialize_postprocess_result(result)
        elif not isinstance(result, dict):
            raise TypeError(
                f"Plugin workflow '{workflow_id}' hook '{phase}' returned "
                f"{type(result).__name__}, expected dict"
            )

    if not isinstance(result, dict):
        raise TypeError(
            f"Plugin workflow '{workflow_id}' hook '{phase}' must return an object"
        )
    if phase == "execute":
        result = merge_registered_outputs_into_result(result, payload["outputs"])
        result.setdefault("status", "succeeded")
        result.setdefault("artifacts", [])
        result.setdefault("output_path", None)
    elif phase == "postprocess":
        result_ops = result.pop("result_ops", None)
        if result_ops:
            raise ValueError("Direct inference does not support postprocess result_ops")
        result = merge_registered_outputs_into_result(result, payload["outputs"])
        result["artifacts"] = _merge_artifact_lists(
            result.get("artifacts"),
            legacy_artifacts_from_outputs(payload["outputs"]),
        )
        prior_result = payload.get("result")
        if isinstance(prior_result, dict):
            result["artifacts"] = _merge_artifact_lists(
                prior_result.get("artifacts"), result.get("artifacts")
            )
            for field in (
                "status",
                "output_path",
                "error",
                "error_traceback",
                "execution_time_seconds",
            ):
                if field not in result and field in prior_result:
                    result[field] = prior_result[field]
    return result


def _merge_artifact_lists(prior: Any, current: Any) -> list[Any]:
    """Preserve execute artifacts and overlay matching postprocess outputs."""
    merged: list[Any] = []
    positions: dict[tuple[str, str], int] = {}
    for artifact in [
        *(prior if isinstance(prior, list) else []),
        *(current if isinstance(current, list) else []),
    ]:
        if not isinstance(artifact, dict):
            if artifact not in merged:
                merged.append(artifact)
            continue
        path = str(artifact.get("storage_path") or artifact.get("path") or "")
        name = str(artifact.get("name") or "")
        key = (path, name)
        if not any(key):
            if artifact not in merged:
                merged.append(artifact)
            continue
        if key in positions:
            merged[positions[key]] = artifact
        else:
            positions[key] = len(merged)
            merged.append(artifact)
    return merged


def _supports_explicit_contract(hook: Any) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return (
        any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        or len(positional) >= 2
    )


def _materialize_prefetch(
    plan: Any,
    cache_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(plan, list):
        raise ValueError("Plugin prepare hook returned a non-array prefetch_plan")
    if not plan:
        return {
            "artifacts": [],
            "stats": {"downloaded": 0, "cached": 0, "errors": 0},
        }

    helper = os.environ.get("PHYSICSNEMO_SERVE_PREFETCH_HELPER")
    if not helper:
        raise ValueError(
            "Plugin requires prefetch, but PHYSICSNEMO_SERVE_PREFETCH_HELPER is not set"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            helper,
            "__prefetch",
            "--cache-dir",
            str(cache_dir),
            "--run-id",
            run_id,
        ],
        input=json.dumps(plan),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"exit status {proc.returncode}"
        raise RuntimeError(f"Local prefetch failed: {detail}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Local prefetch helper returned invalid JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
        raise RuntimeError("Local prefetch helper returned an invalid result")
    stats = result.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    result["stats"] = stats
    return result


def _build_result_envelope(
    payload: dict[str, Any],
    *,
    request_body: dict[str, Any],
    prefetch_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    result = payload["result"]
    status = _normalize_status(result.get("status"))
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    execution: dict[str, Any] = {
        "outputs": artifacts,
        "output_path": result.get("output_path"),
    }
    for field in ("error", "error_traceback", "execution_time_seconds"):
        if field in result:
            execution[field] = result[field]
    if prefetch_stats is not None:
        execution["prefetch"] = prefetch_stats
    plugin_payload = {
        key: value for key, value in result.items() if key not in RESULT_METADATA_FIELDS
    }
    return {
        "run_id": payload["run_id"],
        "status": status,
        "workflow": payload["workflow_id"],
        "request": {
            "content_type": "application/json",
            "operation": payload["operation"],
            "raw_fields": request_body,
        },
        "execution": execution,
        "payload": plugin_payload,
    }


def _normalize_status(value: Any) -> str:
    status = str(value or "succeeded").strip().lower()
    return {
        "complete": "succeeded",
        "completed": "succeeded",
        "ok": "succeeded",
        "success": "succeeded",
        "error": "failed",
        "failure": "failed",
        "canceled": "cancelled",
    }.get(status, status)


if __name__ == "__main__":
    raise SystemExit(main())
