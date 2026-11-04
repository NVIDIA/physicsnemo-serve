# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process boundary around the unchanged repository Python plugin tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import uuid


def _capability_error(reason: str) -> ValueError:
    return ValueError(
        f"Direct Python execution does not support {reason}; use pnms serve --dev for the service pipeline"
    )


def _coverage(
    manifest: dict[str, Any], executed: list[str], requested: str | None
) -> dict[str, Any]:
    stages = manifest.get("pipeline", {}).get("stages", [])
    hooks = [
        str(stage.get("phase"))
        for stage in stages
        if stage.get("handler") == "plugin_phase"
    ]
    return {
        "requested_through_phase": requested,
        "executed_phases": executed,
        "service_stages_not_run": [
            str(stage.get("phase"))
            for stage in stages
            if stage.get("handler") != "plugin_phase"
        ],
        "complete_workflow": executed == hooks,
        "service_qa": False,
    }


def _request_payload(
    legacy: Any, plugin: Path, manifest: dict[str, Any], module: Any, request_path: Path
) -> dict[str, Any]:
    body = legacy.read_json_file(request_path)
    if not isinstance(body, dict):
        raise ValueError("Request file must contain a JSON object")
    content_type = legacy._default_content_type(manifest)
    artifacts: list[dict[str, Any]] = []
    request_schemas, _ = legacy._load_schema_documents(
        plugin,
        manifest,
        workflow_id=legacy._manifest_id(manifest),
        module=module,
    )
    if content_type == "application/json":
        legacy._validate_json_against_schema(
            body, request_schemas[content_type], "Request"
        )
        fields = body
    elif content_type == "multipart/form-data":
        fields = body.get("form_fields", {})
        files = body.get("files", {})
        if not isinstance(fields, dict) or not isinstance(files, dict):
            raise ValueError(
                "Multipart request requires object fields 'form_fields' and 'files'"
            )
        legacy._validate_json_against_schema(
            fields, request_schemas[content_type], "Request form_fields"
        )
        selected_files = {}
        for field, value in files.items():
            if not isinstance(value, str):
                raise ValueError("Multipart file paths must be strings")
            base = (
                request_path.parent
                if (request_path.parent / value).is_file()
                else plugin
            )
            selected_files[field] = str((base / value).resolve())
        legacy._validate_example_multipart_fixture_files(
            plugin, manifest, selected_files, request_path
        )
        rules = {item["name"]: item for item in manifest["ingress"].get("files", [])}
        for field, value in selected_files.items():
            artifact = legacy._artifact_from_fixture(plugin, field, value)
            rule = rules[field]
            if artifact["media_type"] not in rule["media_types"]:
                raise ValueError(f"Unsupported file media type for field '{field}'")
            # The legacy fixture helper treats zero as unlimited; HTTP applies
            # the declared size limit even when it is zero.
            if artifact["size_bytes"] > int(rule.get("max_size_mb", 0)) * 1024 * 1024:
                raise ValueError(f"File exceeds max_size_mb for field '{field}'")
            artifacts.append(artifact)
    else:
        raise _capability_error(f"content type {content_type}")
    operation = str(
        fields.get(
            "operation",
            manifest.get("ingress", {}).get("operations", {}).get("default", "run"),
        )
    )
    allowed = manifest.get("ingress", {}).get("operations", {}).get("allowed")
    if allowed and operation not in allowed:
        raise ValueError(f"Request operation {operation!r} is not allowed")
    resource_defaults = manifest.get("resources", {}).get("defaults", {})
    return {
        "run_id": f"pnms-{uuid.uuid4()}",
        "workflow_id": legacy._manifest_id(manifest),
        "operation": operation,
        "request": {
            "content_type": content_type,
            "raw_fields": fields,
            "input_artifacts": artifacts,
        },
        "parameters": {
            key: value for key, value in fields.items() if key != "operation"
        },
        "resource_profile": {
            "executor_class": manifest["runtime"].get("executor_class"),
            **resource_defaults,
        },
        "prefetch_plan": [],
        "result": None,
        "runtime": dict(manifest["runtime"]),
        "stage_context": {
            "pipeline": list(manifest.get("pipeline", {}).get("stages", []))
        },
    }


def _invoke_postprocess_hook(
    legacy: Any, hook: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    from plugin_direct_runner import _merge_artifact_lists
    from plugin_runtime import (
        legacy_artifacts_from_outputs,
        merge_registered_outputs_into_result,
    )

    outputs = payload["outputs"]
    prior_primary_output = outputs.primary_output()
    if legacy._supports_explicit_contract(hook):
        result = hook(
            legacy.build_prior_result(payload),
            legacy.build_postprocess_context(payload),
        )
    else:
        result = hook(legacy.build_context(payload))
    result = legacy.serialize_postprocess_result(result)
    prior_result = payload.get("result")
    primary_output = outputs.primary_output()
    if (
        result.get("output_path") in (None, "")
        and primary_output is not None
        and primary_output is not prior_primary_output
    ):
        result["output_path"] = str(primary_output.path)
    artifacts = _merge_artifact_lists(
        prior_result.get("artifacts") if isinstance(prior_result, dict) else None,
        result.get("artifacts"),
    )
    registered_artifacts = legacy_artifacts_from_outputs(outputs)
    for registered in registered_artifacts:
        explicit = next(
            (
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and str(artifact.get("storage_path") or artifact.get("path") or "")
                == registered["storage_path"]
                and str(artifact.get("name") or "") == registered["name"]
            ),
            None,
        )
        if explicit is not None:
            registered.update(explicit)
    result["artifacts"] = _merge_artifact_lists(artifacts, registered_artifacts)
    result = legacy._merge_legacy_result_metadata(result, prior_result)
    return merge_registered_outputs_into_result(result, outputs)


def _execute(
    legacy: Any,
    plugin: Path,
    manifest: dict[str, Any],
    module: Any,
    payload: dict[str, Any],
    through_phase: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stages = manifest.get("pipeline", {}).get("stages", [])
    phases = legacy.plugin_phases_from_manifest(manifest)
    if through_phase is not None and through_phase not in phases:
        raise ValueError(
            f"Unknown plugin phase {through_phase!r}; available phases: {phases}"
        )
    stage_by_id = {str(stage.get("id")): stage for stage in stages}
    stage = stages[0] if stages else None
    visited: set[str] = set()
    executed = []
    result: dict[str, Any] | None = None
    payload["outputs"] = legacy.build_context(payload)["outputs"]
    while stage is not None:
        stage_id = str(stage.get("id") or "")
        if stage_id in visited:
            raise ValueError(f"Plugin pipeline contains a cycle at stage '{stage_id}'")
        visited.add(stage_id)
        phase, handler = stage.get("phase"), stage.get("handler")
        if handler == "plugin_phase" and phase not in {
            "prepare",
            "execute",
            "postprocess",
        }:
            raise _capability_error(f"{phase} stages")
        if handler not in {"plugin_phase", "prefetch", "schedule", "persist_results"}:
            raise _capability_error(f"{phase} stages ({handler})")
        payload["stage_context"].update(current_stage_id=stage_id, current_phase=phase)
        next_stage_id = stage.get("next")
        if handler == "plugin_phase":
            phase = str(phase)
            hook = legacy.resolve_phase_hook(
                module, legacy._manifest_id(manifest), phase
            )
            result = (
                _invoke_postprocess_hook(legacy, hook, payload)
                if phase == "postprocess"
                else legacy._invoke_dev_hook(phase, hook, payload)
            )
            executed.append(phase)
            if phase == "prepare":
                legacy._merge_prepare_output(payload, result)
                next_stage_id = result.get("next_stage_id", next_stage_id)
                # A requested prepare-only check may inspect these plans; execution may not skip them.
                if through_phase != "prepare":
                    if payload.get("fanout_profile") or payload.get("fanout_items"):
                        raise _capability_error("fanout")
                    if payload.get("prefetch_plan"):
                        raise _capability_error("prefetch downloads")
                    batch = payload.get("batch_profile")
                    if batch and batch.get("enabled", True):
                        raise _capability_error("batching")
            else:
                if result.get("result_ops"):
                    raise _capability_error("postprocess result operations")
                payload["result"] = result
                if result.get("status") in {"failed", "cancelled"}:
                    break
            if through_phase == phase:
                break
        stage = (
            stage_by_id.get(str(next_stage_id)) if next_stage_id is not None else None
        )
        if next_stage_id is not None and stage is None:
            raise ValueError(
                f"Pipeline stage '{stage_id}' selects unknown next stage '{next_stage_id}'"
            )
    if result is None:
        raise ValueError("Python workflow did not produce a result")
    return result, _coverage(manifest, executed, through_phase)


def dispatch(options: dict[str, Any]) -> dict[str, Any]:
    import plugin_dev as legacy

    plugin = Path(options["plugin"])
    workspace = Path(options["workspace"]) if options.get("workspace") else None
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        os.environ["DEFAULT_OUTPUT_DIR"] = str(workspace / "artifacts")
    if options["operation"] == "init":
        return legacy.init_plugin(
            plugin,
            pipeline_profile=options["pipeline_profile"],
            runtime_profile=options["runtime_profile"],
            phase_runtime_overrides=options.get("phase_runtime_overrides"),
            executor_class=options.get("executor_class"),
            phase_executor_overrides=options.get("phase_executor_overrides"),
        )
    if options["operation"] == "check":
        manifest, _, module = legacy._load_plugin_contract(plugin)
        validation = legacy.validate_plugin(plugin)
        if options["config_only"]:
            return {
                **validation,
                "coverage": _coverage(manifest, [], options.get("through_phase")),
            }
        environment = legacy.check_env_plugin(plugin)
        requested = options.get("through_phase")
        selected = requested
        ready = environment["status"] == "ready"
        if selected is None and not ready:
            selected = environment.get("recommended_check_phase")
            if not selected:
                return {
                    "status": "not_ready",
                    "validation": validation,
                    "environment": environment,
                    "coverage": _coverage(manifest, [], requested),
                    "_exit_code": 1,
                }
        payload = legacy.build_example_payload(plugin, manifest)
        result, coverage = _execute(legacy, plugin, manifest, module, payload, selected)
        coverage["requested_through_phase"] = requested
        coverage["complete_workflow"] = coverage["complete_workflow"] and ready
        successful = result.get("status") not in {"failed", "cancelled"}
        return {
            "status": ("ready" if ready else "needs_setup")
            if successful
            else result["status"],
            "validation": validation,
            "environment": environment,
            "example_run": result,
            "example_phase": selected or "execute",
            "coverage": coverage,
            "_exit_code": 0 if ready and successful else 1,
        }
    if options["operation"] == "run":
        manifest, _, module = legacy._load_plugin_contract(plugin)
        payload = _request_payload(
            legacy, plugin, manifest, module, Path(options["input"])
        )
        result, coverage = _execute(
            legacy, plugin, manifest, module, payload, options.get("through_phase")
        )
        result_path = workspace / "result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return {
            "status": result.get("status", "succeeded"),
            "result": result,
            "result_path": str(result_path),
            "coverage": coverage,
            "_exit_code": 1 if result.get("status") in {"failed", "cancelled"} else 0,
        }
    if options["operation"] == "serve":
        return legacy.run_local_plugin(
            plugin,
            workspace=workspace,
            port=options["port"],
            redis_port=options["redis_port"],
            skip_build=options["skip_build"],
            bootstrap_python_bindings=options["bootstrap_python_bindings"],
            dry_run=options["dry_run"],
        )
    if options["operation"] == "validate":
        manifest = legacy.load_plugin_manifest(plugin / "plugin.yaml")
        return {
            "validation": legacy.validate_plugin(plugin),
            "resources": manifest.get("resources", {}),
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
            },
        }
    raise ValueError(f"Unknown Python operation: {options['operation']}")


def main() -> int:
    repository = Path(sys.argv[1]).resolve()
    report_path = Path(sys.argv[2])
    sys.path.insert(0, str(repository / "scripts"))
    try:
        report = dispatch(json.load(sys.stdin))
    except Exception as exc:
        report_path.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
