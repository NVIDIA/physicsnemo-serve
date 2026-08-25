#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import inspect
import json
import mimetypes
import os
import re
import socket
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
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
    DEFAULT_EXECUTOR_CLASS,
    build_context,
    build_postprocess_context,
    build_prepare_context,
    build_prior_result,
    build_raw_request,
    get_workflow_schema_source,
    load_plugin_manifest,
    load_plugin_module,
    plugin_phases_from_manifest,
    read_json_file,
    resolve_phase_hook,
    serialize_postprocess_result,
    serialize_prepare_result,
    validate_batch_execution_contract,
)
from plugin_sdk import (  # noqa: E402
    workflow_form_schema,
    workflow_request_schema,
    workflow_result_schema,
)

INIT_PIPELINE_PROFILES = (
    "simple",
    "default",
    "postprocess",
    "batch",
    "ensemble",
    "prefetch",
)
INIT_RUNTIME_PROFILES = (
    "python-test",
    "custom",
)
INIT_PHASE_NAMES = ("prepare", "postprocess", "readiness")
ENV_RUNTIME_CONFIG = "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"
ENV_OUTPUT_PUBLICATION_CONFIG_JSON = "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Developer tooling for PhysicsNeMo Serve plugins"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new plugin scaffold")
    init_parser.add_argument("plugin_root")
    init_parser.add_argument(
        "--content-type",
        default="application/json",
        choices=("application/json", "multipart/form-data"),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plugin directory",
    )
    init_parser.add_argument(
        "--pipeline",
        default="simple",
        choices=INIT_PIPELINE_PROFILES,
        help="Pipeline profile for the scaffolded plugin",
    )
    init_parser.add_argument(
        "--runtime",
        default="python-test",
        choices=INIT_RUNTIME_PROFILES,
        help="Runtime family for the scaffolded plugin",
    )
    init_parser.add_argument(
        "--phase-runtime",
        action="append",
        default=[],
        metavar="PHASE=PROFILE",
        help="Override a built-in runtime profile for a specific phase",
    )
    init_parser.add_argument(
        "--executor-class",
        default=None,
        help="Explicit execute executor class for custom runtime scaffolds",
    )
    init_parser.add_argument(
        "--phase-executor",
        action="append",
        default=[],
        metavar="PHASE=EXECUTOR_CLASS",
        help="Explicit phase executor override for custom runtime scaffolds",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a plugin directory"
    )
    validate_parser.add_argument("plugin_root")

    check_parser = subparsers.add_parser(
        "check",
        help="Run the recommended local plugin check flow",
    )
    check_parser.add_argument("plugin_root")
    check_parser.add_argument("--run-id", default=None)
    check_parser.add_argument(
        "--through-phase",
        default=None,
        help="Run the example request through the named plugin phase",
    )

    run_example_parser = subparsers.add_parser(
        "run-example",
        help="Run the plugin's example request locally",
    )
    run_example_parser.add_argument("plugin_root")
    run_example_parser.add_argument("--run-id", default=None)
    run_example_parser.add_argument(
        "--through-phase",
        default=None,
        help="Run plugin phases through the named plugin phase and return that phase output",
    )

    check_env_parser = subparsers.add_parser(
        "check-env",
        help="Check plugin-specific environment and dependency readiness",
    )
    check_env_parser.add_argument("plugin_root")

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-python",
        help="Install local Python bindings required by the dev kit",
    )
    bootstrap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the bootstrap plan without installing anything",
    )

    run_local_parser = subparsers.add_parser(
        "run-local",
        help="Stand up a local single-plugin server + worker stack",
    )
    run_local_parser.add_argument("plugin_root")
    run_local_parser.add_argument(
        "--workspace",
        default=None,
        help="Directory for generated runtime config, helper scripts, and artifacts",
    )
    run_local_parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="HTTP port for inference_server (0 picks a free port)",
    )
    run_local_parser.add_argument(
        "--redis-port",
        type=int,
        default=0,
        help="Redis port for the local stack (0 picks a free port)",
    )
    run_local_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing debug binaries instead of running cargo build",
    )
    run_local_parser.add_argument(
        "--bootstrap-python",
        action="store_true",
        help="Install or refresh local Python bindings required by the dev kit before launch",
    )
    run_local_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the workspace and print the launch plan without starting processes",
    )

    args = parser.parse_args()

    try:
        plugin_root = (
            Path(args.plugin_root).expanduser().resolve()
            if hasattr(args, "plugin_root")
            else None
        )
        if args.command == "init":
            result = init_plugin(
                plugin_root,
                content_type=args.content_type,
                pipeline_profile=args.pipeline,
                runtime_profile=args.runtime,
                phase_runtime_overrides=args.phase_runtime,
                executor_class=args.executor_class,
                phase_executor_overrides=args.phase_executor,
                force=args.force,
            )
        elif args.command == "validate":
            result = validate_plugin(plugin_root)
        elif args.command == "check":
            result = check_plugin(
                plugin_root,
                run_id=args.run_id,
                through_phase=args.through_phase,
            )
        elif args.command == "run-example":
            result = run_example_plugin(
                plugin_root,
                run_id=args.run_id,
                through_phase=args.through_phase,
            )
        elif args.command == "check-env":
            result = check_env_plugin(plugin_root)
        elif args.command == "bootstrap-python":
            result = bootstrap_python(dry_run=args.dry_run)
        elif args.command == "run-local":
            result = run_local_plugin(
                plugin_root,
                workspace=Path(args.workspace).expanduser().resolve()
                if args.workspace
                else None,
                port=args.port,
                redis_port=args.redis_port,
                skip_build=args.skip_build,
                bootstrap_python_bindings=args.bootstrap_python,
                dry_run=args.dry_run,
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")

        exit_code = int(result.pop("_exit_code", 0))
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return exit_code
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def init_plugin(
    plugin_root: Path,
    content_type: str = "application/json",
    pipeline_profile: str = "simple",
    runtime_profile: str = "python-test",
    phase_runtime_overrides: list[str] | None = None,
    executor_class: str | None = None,
    phase_executor_overrides: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workflow_id = _derive_workflow_id(plugin_root.name)
    display_name = _display_name(workflow_id)
    manifest = _scaffold_manifest(
        workflow_id=workflow_id,
        display_name=display_name,
        content_type=content_type,
        pipeline_profile=pipeline_profile,
        runtime_profile=runtime_profile,
        phase_runtime_overrides=phase_runtime_overrides or [],
        executor_class=executor_class,
        phase_executor_overrides=phase_executor_overrides or [],
    )

    if plugin_root.exists():
        if not force:
            raise ValueError(f"Plugin directory already exists: {plugin_root}")
        shutil.rmtree(plugin_root)

    plugin_root.mkdir(parents=True, exist_ok=False)
    (plugin_root / "examples").mkdir()

    (plugin_root / "plugin.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (plugin_root / "workflow.py").write_text(
        _workflow_template(
            content_type=content_type,
            pipeline_profile=pipeline_profile,
        ),
        encoding="utf-8",
    )
    (plugin_root / "README.md").write_text(
        _readme_template(
            plugin_root=plugin_root,
            workflow_id=workflow_id,
            content_type=content_type,
            pipeline_profile=pipeline_profile,
        ),
        encoding="utf-8",
    )

    if content_type != "application/json":
        (plugin_root / "examples" / "sample.txt").write_text(
            "fixture-data\n",
            encoding="utf-8",
        )
        (plugin_root / "examples" / "default_request.multipart.json").write_text(
            json.dumps(
                {
                    "form_fields": {"note": "hello"},
                    "files": {"sample_file": "examples/sample.txt"},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "status": "scaffolded",
        "workflow_id": workflow_id,
        "plugin_root": str(plugin_root),
        "content_type": content_type,
        "pipeline_profile": pipeline_profile,
        "runtime_profile": runtime_profile,
    }


def validate_plugin(plugin_root: Path) -> dict[str, Any]:
    manifest, workflow_id, module = _load_plugin_contract(plugin_root)
    request_schemas, result_schema = _load_schema_documents(
        plugin_root,
        manifest,
        workflow_id=workflow_id,
        module=module,
    )

    _validate_readiness_config(manifest)
    validate_batch_execution_contract(module, workflow_id)

    phases = plugin_phases_from_manifest(manifest)
    for phase in phases:
        resolve_phase_hook(module, workflow_id, phase)

    _validate_example_request_fixture(plugin_root, manifest, request_schemas)
    _validate_expected_result_fixture(plugin_root, result_schema)

    # Build the example payload as part of validation to catch missing fixture files
    # and malformed multipart fixture structure early.
    build_example_payload(plugin_root, manifest)

    return {
        "plugin_root": str(plugin_root),
        "workflow_id": workflow_id,
        "status": "valid",
        "phases": phases,
    }


def check_plugin(
    plugin_root: Path,
    run_id: str | None = None,
    through_phase: str | None = None,
) -> dict[str, Any]:
    validation = validate_plugin(plugin_root)
    validation.pop("_exit_code", None)

    environment = check_env_plugin(plugin_root)
    environment.pop("_exit_code", None)

    example_phase = through_phase
    status = "ready"

    if example_phase is None and environment["status"] != "ready":
        recommended = environment.get("recommended_check_phase")
        if isinstance(recommended, str) and recommended.strip():
            example_phase = recommended.strip()
            status = "needs_setup"
        else:
            return {
                "status": "not_ready",
                "validation": validation,
                "environment": environment,
                "_exit_code": 1,
            }

    example_run = run_example_plugin(
        plugin_root,
        run_id=run_id,
        through_phase=example_phase,
    )

    return {
        "status": status,
        "validation": validation,
        "environment": environment,
        "example_phase": example_phase or "execute",
        "example_run": example_run,
        "_exit_code": 0,
    }


def run_example_plugin(
    plugin_root: Path,
    run_id: str | None = None,
    through_phase: str | None = None,
) -> dict[str, Any]:
    manifest, workflow_id, module = _load_plugin_contract(plugin_root)

    plugin_phases = plugin_phases_from_manifest(manifest)
    if through_phase is not None and through_phase not in plugin_phases:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' does not define plugin phase '{through_phase}'. "
            f"Available phases: {plugin_phases}"
        )

    payload = build_example_payload(plugin_root, manifest, run_id=run_id)

    for stage in manifest.get("pipeline", {}).get("stages", []):
        if stage.get("handler") != "plugin_phase":
            continue

        phase = str(stage.get("phase") or "")
        hook = resolve_phase_hook(module, workflow_id, phase)
        result = _invoke_dev_hook(phase, hook, payload)

        if phase == "prepare":
            _merge_prepare_output(payload, result)
        elif phase in {"execute", "postprocess"}:
            payload["result"] = result
        else:
            raise ValueError(f"Unsupported plugin example-run phase: {phase}")

        if through_phase == phase:
            return result

    final_result = payload.get("result")
    if not isinstance(final_result, dict):
        raise ValueError(
            f"Plugin workflow '{workflow_id}' example run did not produce a result payload"
        )
    return final_result


def check_env_plugin(plugin_root: Path) -> dict[str, Any]:
    manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
    workflow_id = _manifest_id(manifest)
    readiness = _readiness_config(manifest)
    _validate_readiness_config(manifest)

    checks: list[dict[str, Any]] = []
    added_paths: list[str] = []
    try:
        for module_name in readiness.get("python_modules", []):
            probe = _probe_python_module_contract(
                module_name,
                [],
                extra_python_paths=added_paths,
            )
            checks.append(
                {
                    "type": "python_module",
                    "name": module_name,
                    "required": True,
                    "ok": probe["ok"],
                    "detail": str(probe["detail"]),
                }
            )
    finally:
        for value in reversed(added_paths):
            try:
                sys.path.remove(value)
            except ValueError:
                pass

    for env_spec in readiness.get("env", []):
        checks.append(_run_env_check(env_spec))

    for path_spec in readiness.get("paths", []):
        checks.append(_run_path_check(plugin_root, path_spec))

    failed_required = [
        check for check in checks if check["required"] and not check["ok"]
    ]
    status = "ready" if not failed_required else "not_ready"
    return {
        "workflow_id": workflow_id,
        "status": status,
        "recommended_check_phase": readiness.get("recommended_check_phase"),
        "checks": checks,
        "_exit_code": 0 if status == "ready" else 1,
    }


def run_local_plugin(
    plugin_root: Path,
    *,
    workspace: Path | None,
    port: int,
    redis_port: int,
    skip_build: bool,
    bootstrap_python_bindings: bool,
    dry_run: bool,
) -> dict[str, Any]:
    validate_plugin(plugin_root)

    environment = check_env_plugin(plugin_root)
    if environment["status"] != "ready":
        recommended = environment.get("recommended_check_phase")
        extra_hint = (
            f" If you only need a lightweight check, run `python scripts/plugin_dev.py check {plugin_root} --through-phase {recommended}`."
            if recommended
            else ""
        )
        raise ValueError(
            f"Plugin workflow '{environment['workflow_id']}' is not ready for local run. "
            f"Run `python scripts/plugin_dev.py check-env {plugin_root}` "
            f"to inspect readiness details.{extra_hint}"
        )

    plan = build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=port,
        redis_port=redis_port,
    )
    write_run_local_workspace(plan)

    if dry_run:
        return {
            "status": "planned",
            **plan_to_json(plan),
        }

    if bootstrap_python_bindings:
        bootstrap_python(dry_run=False)

    ensure_run_local_prerequisites(skip_build=skip_build)
    if not skip_build:
        build_run_local_binaries()

    run_local_stack(plan)
    return {"status": "completed", **plan_to_json(plan)}


def _request_fixture_candidates(plugin_root: Path, content_type: str) -> list[Path]:
    examples_root = plugin_root / "examples"
    fixtures_root = plugin_root / "fixtures"
    if content_type == "application/json":
        return [
            examples_root / "default_request.json",
            fixtures_root / "example_request.json",
        ]
    if content_type == "multipart/form-data":
        return [
            examples_root / "default_request.multipart.json",
            fixtures_root / "example_request.multipart.json",
        ]
    raise ValueError(
        f"Unsupported content type for example fixture lookup: {content_type}"
    )


def _request_fixture_path(plugin_root: Path, content_type: str) -> Path:
    candidates = _request_fixture_candidates(plugin_root, content_type)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[0])


def _generate_example_from_schema(schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return {}

    if "default" in schema:
        return schema["default"]
    if "example" in schema:
        return schema["example"]
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        return examples[0]
    if "const" in schema:
        return schema["const"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    for key in ("oneOf", "anyOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            preferred_variants = [
                item
                for item in variants
                if not (isinstance(item, dict) and item.get("type") == "null")
            ]
            if preferred_variants:
                return _generate_example_from_schema(preferred_variants[0])

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for item in all_of:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "object" or "properties" in item:
                merged["properties"].update(item.get("properties", {}))
                merged["required"] = list(
                    dict.fromkeys([*merged["required"], *item.get("required", [])])
                )
        if merged["properties"]:
            return _generate_example_from_schema(merged)
        return _generate_example_from_schema(all_of[0])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if non_null:
            schema_type = non_null[0]

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        result: dict[str, Any] = {}
        if isinstance(required, list) and isinstance(properties, dict):
            for field_name in required:
                property_schema = properties.get(field_name)
                if isinstance(property_schema, dict):
                    result[str(field_name)] = _generate_example_from_schema(
                        property_schema
                    )
        if result:
            return result
        if isinstance(properties, dict):
            preferred_fields = [
                name
                for name, property_schema in properties.items()
                if isinstance(property_schema, dict)
                and any(
                    key in property_schema
                    for key in ("default", "example", "examples", "const", "enum")
                )
            ]
            for field_name in preferred_fields[:1]:
                result[str(field_name)] = _generate_example_from_schema(
                    properties[field_name]
                )
        return result

    if schema_type == "array":
        item_schema = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        try:
            count = max(int(min_items), 1)
        except (TypeError, ValueError):
            count = 1
        return [_generate_example_from_schema(item_schema) for _ in range(count)]

    if schema_type == "string":
        schema_format = schema.get("format")
        if schema_format == "date-time":
            return "2026-01-01T00:00:00Z"
        if schema_format == "date":
            return "2026-01-01"
        min_length = schema.get("minLength", 0)
        try:
            min_length = int(min_length)
        except (TypeError, ValueError):
            min_length = 0
        return "x" * max(1, min_length)

    if schema_type == "integer":
        minimum = schema.get("minimum")
        if minimum is not None:
            return int(minimum)
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is not None:
            return int(exclusive_minimum) + 1
        return 1

    if schema_type == "number":
        minimum = schema.get("minimum")
        if minimum is not None:
            return float(minimum)
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is not None:
            return float(exclusive_minimum) + 1.0
        return 1.0

    if schema_type == "boolean":
        return True

    if schema_type == "null":
        return None

    return {}


def _load_or_generate_json_example_request(
    plugin_root: Path,
    manifest: dict[str, Any],
    request_schemas: dict[str, Any] | None = None,
    *,
    workflow_id: str | None = None,
    module: Any | None = None,
) -> tuple[dict[str, Any], Path | None]:
    try:
        fixture_path = _request_fixture_path(plugin_root, "application/json")
        fixture = read_json_file(fixture_path)
        if not isinstance(fixture, dict):
            raise ValueError(
                f"Example request file must be a JSON object: {fixture_path}"
            )
        return fixture, fixture_path
    except FileNotFoundError:
        pass

    if request_schemas is None:
        request_schemas, _result_schema = _load_schema_documents(
            plugin_root,
            manifest,
            workflow_id=workflow_id,
            module=module,
        )

    schema = request_schemas.get("application/json")
    if schema is None:
        raise ValueError(
            "JSON example request generation requires an application/json request schema"
        )

    generated = _generate_example_from_schema(schema)
    if not isinstance(generated, dict):
        raise ValueError("Generated example request must be a JSON object")

    _validate_json_against_schema(
        generated,
        schema,
        "Generated example request",
    )
    return generated, None


def _expected_result_fixture_path(plugin_root: Path) -> Path | None:
    candidates = [
        plugin_root / "examples" / "expected_result.json",
        plugin_root / "fixtures" / "expected_result.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_example_payload(
    plugin_root: Path,
    manifest: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    workflow_id = _manifest_id(manifest)
    content_type = _default_content_type(manifest)
    operation_default = (
        manifest.get("ingress", {}).get("operations", {}).get("default", "run")
    )

    if content_type == "application/json":
        request_body, _request_path = _load_or_generate_json_example_request(
            plugin_root,
            manifest,
            workflow_id=workflow_id,
        )
        raw_fields = dict(request_body)
        operation = str(raw_fields.get("operation", operation_default))
        parameters = dict(raw_fields)
        parameters.pop("operation", None)
        input_artifacts: list[dict[str, Any]] = []
    elif content_type == "multipart/form-data":
        fixture_path = _request_fixture_path(plugin_root, content_type)
        fixture = read_json_file(fixture_path)
        if not isinstance(fixture, dict):
            raise ValueError(
                f"Example request file must be a JSON object: {fixture_path}"
            )
        form_fields = fixture.get("form_fields", {})
        files = fixture.get("files", {})
        if not isinstance(form_fields, dict) or not isinstance(files, dict):
            raise ValueError(
                f"Example request file must contain object fields 'form_fields' and 'files': {fixture_path}"
            )
        raw_fields = dict(form_fields)
        operation = str(raw_fields.get("operation", operation_default))
        parameters = dict(raw_fields)
        parameters.pop("operation", None)
        input_artifacts = [
            _artifact_from_fixture(plugin_root, field_name, relative_path)
            for field_name, relative_path in files.items()
        ]
    else:
        raise ValueError(
            f"Unsupported content type for example execution: {content_type}"
        )

    run_id = run_id or f"example-{uuid.uuid4()}"
    resources = manifest.get("resources", {})
    resource_defaults = (
        resources.get("defaults", {}) if isinstance(resources, dict) else {}
    )
    resource_profile = (
        {
            "executor_class": manifest.get("runtime", {}).get("executor_class"),
            "device_kind": resource_defaults.get("device_kind"),
            "gpus_required": resource_defaults.get("gpus_required"),
            "memory_mb": resource_defaults.get("memory_mb"),
            "cpu_cores": resource_defaults.get("cpu_cores"),
        }
        if resource_defaults
        else None
    )
    payload = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "operation": operation,
        "request": {
            "content_type": content_type,
            "raw_fields": raw_fields,
            "input_artifacts": input_artifacts,
        },
        "parameters": parameters,
        "resource_profile": resource_profile,
        "prefetch_plan": [],
        "stage_context": {
            "current_stage_id": _first_stage(manifest).get("id"),
            "current_phase": _first_stage(manifest).get("phase"),
            "pipeline": list(manifest.get("pipeline", {}).get("stages", [])),
        },
        "result": None,
        "runtime": dict(manifest.get("runtime", {})),
    }

    os.environ.setdefault("DEFAULT_OUTPUT_DIR", str(plugin_root / ".example-output"))
    return payload


def _load_output_publication_override_from_env() -> dict[str, Any] | None:
    raw = str(os.environ.get(ENV_OUTPUT_PUBLICATION_CONFIG_JSON, "")).strip()
    if not raw:
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {ENV_OUTPUT_PUBLICATION_CONFIG_JSON} as output_publication JSON: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"{ENV_OUTPUT_PUBLICATION_CONFIG_JSON} must contain a JSON object"
        )
    return config


def _load_runtime_output_publication_from_env() -> dict[str, Any] | None:
    output_publication_override = _load_output_publication_override_from_env()
    config_path = str(os.environ.get(ENV_RUNTIME_CONFIG, "")).strip()
    if not config_path:
        return output_publication_override
    path = Path(config_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"failed to read {ENV_RUNTIME_CONFIG} file '{path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {ENV_RUNTIME_CONFIG} file '{path}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{ENV_RUNTIME_CONFIG} file '{path}' must contain a JSON object"
        )
    output_publication = data.get("output_publication")
    if output_publication_override is not None:
        return output_publication_override
    if output_publication is None:
        return None
    if not isinstance(output_publication, dict):
        raise ValueError(
            f"{ENV_RUNTIME_CONFIG} file '{path}' field output_publication must be an object"
        )
    return output_publication


def _load_runtime_publish_role_config_from_env() -> dict[str, Any] | None:
    config_path = str(os.environ.get(ENV_RUNTIME_CONFIG, "")).strip()
    if not config_path:
        return None
    path = Path(config_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"failed to read {ENV_RUNTIME_CONFIG} file '{path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {ENV_RUNTIME_CONFIG} file '{path}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{ENV_RUNTIME_CONFIG} file '{path}' must contain a JSON object"
        )
    roles = data.get("roles")
    if not isinstance(roles, dict):
        return None
    publish = roles.get("publish")
    if not isinstance(publish, dict):
        return None
    config = publish.get("config")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError(
            f"{ENV_RUNTIME_CONFIG} file '{path}' field roles.publish.config must be an object"
        )
    return copy.deepcopy(config)


def build_run_local_plan(
    plugin_root: Path,
    *,
    workspace: Path | None,
    port: int,
    redis_port: int,
) -> dict[str, Any]:
    manifest, workflow_id, _module = _load_plugin_contract(plugin_root)
    execute_profiles = _derive_run_local_execute_worker_profiles(plugin_root, manifest)
    workspace_root = workspace or (
        plugin_root
        / ".run-local"
        / f"session-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "logs").mkdir(exist_ok=True)
    (workspace_root / "artifacts").mkdir(exist_ok=True)
    (workspace_root / "outputs").mkdir(exist_ok=True)

    runtime_plugin_root = _materialize_run_local_plugin_root(
        plugin_root,
        workspace_root=workspace_root,
        manifest=manifest,
        execute_profiles=execute_profiles,
    )
    if runtime_plugin_root != plugin_root:
        manifest, workflow_id, _module = _load_plugin_contract(runtime_plugin_root)
    stages = _pipeline_stages(manifest)
    output_publication = _load_runtime_output_publication_from_env()
    include_publish = bool(output_publication and output_publication.get("enabled"))
    publish_role_config = _load_runtime_publish_role_config_from_env()

    selected_port = _pick_free_port(18080) if port in (None, 0) else port
    selected_redis_port = (
        _pick_free_port(16379) if redis_port in (None, 0) else redis_port
    )
    redis_url = f"redis://127.0.0.1:{selected_redis_port}/0"

    python_runtime_envs = _build_run_local_python_runtime_envs(
        manifest, execute_profiles
    )
    runtime_config = _build_runtime_config(
        stages,
        python_runtime_envs=python_runtime_envs,
        include_publish=include_publish,
        output_publication=output_publication,
        publish_role_config=publish_role_config,
    )
    runtime_config_path = workspace_root / "runtime_config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8"
    )

    example_request = _build_example_request_hint(
        plugin_root=runtime_plugin_root,
        workflow_id=workflow_id,
        port=selected_port,
        workspace_root=workspace_root,
    )
    submit_example_script = workspace_root / "submit_example.sh"
    submit_example_script.write_text(example_request["script"], encoding="utf-8")
    submit_example_script.chmod(0o755)

    shared_env = {
        "PYTHONUNBUFFERED": "1",
        "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE": sys.executable,
        ENV_RUNTIME_CONFIG: str(runtime_config_path),
        "PORT": str(selected_port),
        "REDIS_URL": redis_url,
        "POD_NAMESPACE": "local",
        "POD_NAME": f"{workflow_id}-run-local",
        "PLUGIN_DIR": str(runtime_plugin_root.parent),
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": workflow_id,
        "ARTIFACT_DIR": str(workspace_root / "artifacts"),
        "DEFAULT_OUTPUT_DIR": str(workspace_root / "outputs"),
    }

    processes = [
        {
            "name": "redis",
            "argv": [
                "redis-server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(selected_redis_port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--loglevel",
                "warning",
            ],
            "env": {},
        },
        {
            "name": "inference_server",
            "argv": [str(_inference_server_binary_path())],
            "env": {
                **shared_env,
                "RUST_LOG": "info",
            },
        },
    ]

    for role_name in _runtime_roles_for_pipeline(
        stages, include_publish=include_publish
    ):
        processes.append(
            {
                "name": role_name,
                "argv": [
                    str(_worker_runtime_binary_path()),
                    "--role",
                    role_name,
                    "--config-path",
                    str(runtime_config_path),
                ],
                "env": {
                    **shared_env,
                    "RUST_LOG": "info",
                    "WORKER_PIPELINE_CONFIG": str(runtime_config_path),
                    "WORKER_ROLE": role_name,
                },
            }
        )

    execute_executor_classes = ",".join(
        profile["executor_class"]
        for profile in execute_profiles
        if str(profile.get("executor_class") or "").strip()
    )
    if execute_executor_classes:
        processes.append(
            {
                "name": "runtime_env_launcher",
                "argv": [sys.executable, str(SCRIPT_DIR / "runtime_env_launcher.py")],
                "env": {
                    **shared_env,
                    "PHYSICSNEMO_SERVE_EXECUTOR_CLASSES": execute_executor_classes,
                    "WORKER_RUNTIME_CONFIG": str(runtime_config_path),
                    "WORKER_SCRIPT": str(SCRIPT_DIR / "inference_worker.py"),
                    "WORKERS_PER_GPU": "1",
                    "STREAM_PREFIX": "",
                },
            }
        )

    commands_path = workspace_root / "commands.json"
    commands_path.write_text(
        json.dumps({"processes": processes}, indent=2) + "\n", encoding="utf-8"
    )

    execute_registration_streams = _run_local_execute_registration_streams(
        execute_profiles,
        pod_namespace=shared_env["POD_NAMESPACE"],
        pod_name=shared_env["POD_NAME"],
    )

    return {
        "workflow_id": workflow_id,
        "workspace": workspace_root,
        "runtime_config_path": runtime_config_path,
        "submit_example_script": submit_example_script,
        "commands_path": commands_path,
        "server_url": f"http://127.0.0.1:{selected_port}",
        "port": selected_port,
        "redis_port": selected_redis_port,
        "redis_url": redis_url,
        "processes": processes,
        "example_request": example_request,
        "execute_registration_stream": execute_registration_streams[0]
        if execute_registration_streams
        else None,
        "execute_registration_streams": execute_registration_streams,
    }


def _materialize_run_local_plugin_root(
    plugin_root: Path,
    *,
    workspace_root: Path,
    manifest: dict[str, Any],
    execute_profiles: list[dict[str, Any]],
) -> Path:
    if not _run_local_needs_cpu_direct_pipeline(manifest, execute_profiles):
        return plugin_root

    runtime_plugins_root = workspace_root / "plugins"
    runtime_plugin_root = runtime_plugins_root / plugin_root.name
    if runtime_plugin_root.resolve() == plugin_root.resolve():
        runtime_plugins_root = workspace_root / ".run-local" / "plugins"
        runtime_plugin_root = runtime_plugins_root / plugin_root.name
    if runtime_plugin_root.exists():
        shutil.rmtree(runtime_plugin_root)

    shutil.copytree(
        plugin_root,
        runtime_plugin_root,
        ignore=shutil.ignore_patterns(".run-local", "__pycache__", ".pytest_cache"),
    )
    runtime_manifest = copy.deepcopy(manifest)
    runtime_manifest["pipeline"] = {
        "stages": _direct_run_local_execute_stages(_pipeline_stages(manifest))
    }
    (runtime_plugin_root / "plugin.yaml").write_text(
        yaml.safe_dump(runtime_manifest, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_plugin_root


def _run_local_needs_cpu_direct_pipeline(
    manifest: dict[str, Any],
    execute_profiles: list[dict[str, Any]],
) -> bool:
    if not execute_profiles or any(
        str(profile.get("device_kind") or "").strip().lower() != "cpu"
        for profile in execute_profiles
    ):
        return False
    if not _manifest_uses_compact_pipeline_profile(manifest):
        return False
    pipeline = manifest.get("pipeline", {})
    if str(pipeline.get("profile") or "").strip() == "batch":
        return False
    if any(
        str(stage.get("phase") or "").strip() == "fanout"
        for stage in _pipeline_stages(manifest)
    ):
        return False
    return any(
        str(stage.get("phase") or "").strip() == "schedule"
        and str(stage.get("handler") or "").strip() == "schedule"
        for stage in _pipeline_stages(manifest)
    )


def _manifest_uses_compact_pipeline_profile(manifest: dict[str, Any]) -> bool:
    pipeline = manifest.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return False
    return bool(str(pipeline.get("profile") or "").strip())


def _direct_run_local_execute_stages(
    stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_stages = [
        copy.deepcopy(stage)
        for stage in stages
        if not (
            str(stage.get("phase") or "").strip() == "schedule"
            and str(stage.get("handler") or "").strip() == "schedule"
        )
    ]
    for index, stage in enumerate(direct_stages):
        stage["next"] = (
            str(direct_stages[index + 1]["id"])
            if index + 1 < len(direct_stages)
            else None
        )
    return direct_stages


def _build_run_local_python_runtime_envs(
    manifest: dict[str, Any],
    execute_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime = manifest.get("runtime", {})
    executor_classes: set[str] = set()

    for key in (
        "executor_class",
        "prepare_executor_class",
        "postprocess_executor_class",
        "readiness_executor_class",
    ):
        value = str(runtime.get(key) or "").strip()
        if value:
            executor_classes.add(value)

    for execute_profile in execute_profiles:
        execute_executor_class = str(
            execute_profile.get("executor_class") or ""
        ).strip()
        if execute_executor_class:
            executor_classes.add(execute_executor_class)

    python_executable = os.environ.get(
        "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", sys.executable
    )
    runtime_envs = {
        executor_class: {
            "python_executable": python_executable,
            "env": {},
        }
        for executor_class in sorted(executor_classes)
    }

    for execute_profile in execute_profiles:
        execute_executor_class = str(
            execute_profile.get("executor_class") or ""
        ).strip()
        if not execute_executor_class:
            continue
        launch: dict[str, Any] = {
            "enabled": True,
            "device_kind": str(execute_profile.get("device_kind") or "gpu"),
            "tags": list(execute_profile.get("tags") or []),
        }
        if launch["device_kind"] == "cpu":
            launch["replicas"] = 1
            launch["memory_mb"] = int(execute_profile.get("memory_mb") or 4096)
        else:
            launch["workers_per_device"] = 1
        runtime_envs[execute_executor_class]["launch"] = launch

    return runtime_envs


def _run_local_execute_registration_streams(
    execute_profiles: list[dict[str, Any]],
    *,
    pod_namespace: str,
    pod_name: str,
) -> list[str]:
    streams: list[str] = []
    for execute_profile in execute_profiles:
        executor_class = str(execute_profile.get("executor_class") or "").strip()
        if not executor_class:
            continue

        device_kind = str(execute_profile.get("device_kind") or "gpu").strip().lower()
        if device_kind == "cpu":
            stream = f"execute.{executor_class}"
        else:
            stream = f"execute.{executor_class}:gpu:{pod_namespace}:{pod_name}:0"
        if stream not in streams:
            streams.append(stream)
    return streams


def plan_to_json(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_id": plan["workflow_id"],
        "workspace": str(plan["workspace"]),
        "runtime_config_path": str(plan["runtime_config_path"]),
        "submit_example_script": str(plan["submit_example_script"]),
        "commands_path": str(plan["commands_path"]),
        "server_url": plan["server_url"],
        "port": plan["port"],
        "redis_port": plan["redis_port"],
        "redis_url": plan["redis_url"],
        "processes": plan["processes"],
        "example_request": {
            "command": plan["example_request"]["command"],
            "content_type": plan["example_request"]["content_type"],
            "fixture_path": str(plan["example_request"]["fixture_path"]),
        },
        "execute_registration_stream": plan["execute_registration_stream"],
        "execute_registration_streams": list(
            plan.get("execute_registration_streams") or []
        ),
    }


def write_run_local_workspace(plan: dict[str, Any]) -> None:
    readme_path = Path(plan["workspace"]) / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                f"# Local run: {plan['workflow_id']}",
                "",
                f"- Server: {plan['server_url']}",
                f"- Redis: {plan['redis_url']}",
                f"- Runtime config: `{Path(plan['runtime_config_path']).name}`",
                f"- Example submit script: `{Path(plan['submit_example_script']).name}`",
                "",
                "## Dry-run launch plan",
                "",
                "```json",
                json.dumps({"processes": plan["processes"]}, indent=2),
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_run_local_prerequisites(*, skip_build: bool) -> None:
    missing: list[str] = []
    if shutil.which("redis-server") is None:
        missing.append("redis-server")
    if importlib.util.find_spec("redis") is None:
        missing.append("python package 'redis'")
    scicomp_rq = _probe_python_module_contract("scicomp_rq", ["QueueManager", "Output"])
    if not scicomp_rq["ok"]:
        missing.append(
            f"compatible python package 'scicomp_rq' ({scicomp_rq['detail']})"
        )
    if not skip_build and shutil.which("cargo") is None:
        missing.append("cargo")
    if missing:
        raise ValueError(
            "Local run prerequisites are missing: "
            + ", ".join(missing)
            + ". Install them, run `python scripts/plugin_dev.py bootstrap-python`, or rerun with --dry-run."
        )

    if skip_build:
        missing_binaries: list[str] = []
        if not _inference_server_binary_path().is_file():
            missing_binaries.append(str(_inference_server_binary_path()))
        if not _worker_runtime_binary_path().is_file():
            missing_binaries.append(str(_worker_runtime_binary_path()))
        if missing_binaries:
            raise ValueError(
                "Missing built binaries for --skip-build: "
                + ", ".join(missing_binaries)
            )


def build_run_local_binaries() -> None:
    proc = subprocess.run(
        ["cargo", "build", "-p", "inference_server", "-p", "worker-runtime"],
        cwd=REPO_ROOT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError("cargo build failed for inference_server/worker-runtime")


def bootstrap_python(*, dry_run: bool) -> dict[str, Any]:
    targets = [
        {
            "module": "redis",
            "required_attrs": ["Redis"],
            "install_path": "redis (PyPI)",
            "command": [
                sys.executable,
                "-m",
                "pip",
                "install",
                "redis",
            ],
        },
        {
            "module": "scicomp_rq",
            "required_attrs": ["QueueManager", "Output"],
            "install_path": str(REPO_ROOT / "crates" / "scicomp-rq"),
            "command": [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(REPO_ROOT / "crates" / "scicomp-rq"),
            ],
        },
    ]

    planned_targets: list[dict[str, Any]] = []
    for target in targets:
        probe = _probe_python_module_contract(
            target["module"], target["required_attrs"]
        )
        if probe["ok"]:
            planned_targets.append(
                {
                    "module": target["module"],
                    "status": "ready",
                    "detail": probe["detail"],
                    "command": " ".join(target["command"]),
                }
            )
            continue

        if dry_run:
            planned_targets.append(
                {
                    "module": target["module"],
                    "status": "planned",
                    "detail": probe["detail"],
                    "command": " ".join(target["command"]),
                }
            )
            continue

        proc = subprocess.run(
            target["command"],
            cwd=REPO_ROOT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise ValueError(
                f"Failed to bootstrap Python module '{target['module']}' from {target['install_path']}"
            )

        post_probe = _probe_python_module_contract(
            target["module"], target["required_attrs"]
        )
        if not post_probe["ok"]:
            raise ValueError(
                f"Bootstrapped Python module '{target['module']}', but it is still incompatible: {post_probe['detail']}"
            )

        planned_targets.append(
            {
                "module": target["module"],
                "status": "bootstrapped",
                "detail": post_probe["detail"],
                "command": " ".join(target["command"]),
            }
        )

    status = "ready"
    if any(target["status"] == "bootstrapped" for target in planned_targets):
        status = "bootstrapped"
    elif any(target["status"] == "planned" for target in planned_targets):
        status = "planned"

    return {
        "status": status,
        "targets": planned_targets,
    }


def run_local_stack(plan: dict[str, Any]) -> None:
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        for process in plan["processes"]:
            name = process["name"]
            env = os.environ.copy()
            env.update(process.get("env", {}))
            proc = subprocess.Popen(
                process["argv"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
            )
            processes.append((name, proc))
            if name == "redis":
                _wait_for_tcp_port(
                    "127.0.0.1", int(plan["redis_port"]), timeout_secs=10
                )
            elif name == "inference_server":
                _wait_for_tcp_port("127.0.0.1", int(plan["port"]), timeout_secs=30)
        execute_streams = plan.get("execute_registration_streams")
        if isinstance(execute_streams, list):
            streams_to_wait_for = [
                str(stream).strip() for stream in execute_streams if str(stream).strip()
            ]
        else:
            execute_stream = str(plan.get("execute_registration_stream") or "").strip()
            streams_to_wait_for = [execute_stream] if execute_stream else []
        for execute_stream in streams_to_wait_for:
            _wait_for_worker_registration(
                plan["redis_url"], execute_stream, timeout_secs=10
            )
        time.sleep(1.2)

        print(f"Local stack ready for workflow '{plan['workflow_id']}'.")
        print(f"Workspace: {plan['workspace']}")
        print(f"Server: {plan['server_url']}")
        print(f"Example request: {plan['example_request']['command']}")
        print("Press Ctrl-C to stop.")

        while True:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"Process '{name}' exited unexpectedly with code {code}"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down local stack...")
    finally:
        _terminate_processes(processes, suppress_interrupts=True)


def _terminate_processes(
    processes: list[tuple[str, subprocess.Popen[str]]],
    *,
    suppress_interrupts: bool = False,
) -> None:
    for _name, proc in reversed(processes):
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 5
    for _name, proc in reversed(processes):
        if proc.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except KeyboardInterrupt:
            if not suppress_interrupts:
                raise
            if proc.poll() is None:
                proc.kill()
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_tcp_port(host: str, port: int, *, timeout_secs: float) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise ValueError(f"Timed out waiting for {host}:{port}")


def _wait_for_worker_registration(
    redis_url: str, stream_name: str, *, timeout_secs: float
) -> None:
    import redis

    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            entries = client.hgetall("gpu:registry")
        finally:
            client.close()

        for raw_entry in entries.values():
            try:
                payload = json.loads(raw_entry)
            except json.JSONDecodeError:
                continue
            if (
                payload.get("stream") == stream_name
                and str(payload.get("status", "")).lower() == "available"
            ):
                return

        time.sleep(0.2)

    raise ValueError(
        f"Timed out waiting for worker registration on stream '{stream_name}' via {redis_url}"
    )


def _pick_free_port(fallback: int) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        return fallback


def _probe_python_module_contract(
    module_name: str,
    required_attrs: list[str] | tuple[str, ...],
    *,
    extra_python_paths: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    required = [str(attr) for attr in required_attrs]
    probe_script = (
        "import importlib, json, sys, os\n"
        f"module_name = {module_name!r}\n"
        f"required = {required!r}\n"
        "_real_stdout = sys.stdout\n"
        "sys.stdout = open(os.devnull, 'w')\n"
        "try:\n"
        "    module = importlib.import_module(module_name)\n"
        "except Exception as exc:\n"
        "    sys.stdout.close()\n"
        "    sys.stdout = _real_stdout\n"
        "    print(json.dumps({\n"
        "        'module': module_name,\n"
        "        'required_attrs': required,\n"
        "        'missing_attrs': required,\n"
        "        'ok': False,\n"
        "        'detail': f\"module '{module_name}' could not be imported: {exc}\",\n"
        "    }))\n"
        "    raise SystemExit(0)\n"
        "sys.stdout.close()\n"
        "sys.stdout = _real_stdout\n"
        "missing = [attr for attr in required if not hasattr(module, attr)]\n"
        "if missing:\n"
        "    print(json.dumps({\n"
        "        'module': module_name,\n"
        "        'required_attrs': required,\n"
        "        'missing_attrs': missing,\n"
        "        'ok': False,\n"
        "        'detail': f\"module '{module_name}' is missing required attributes: {', '.join(missing)}\",\n"
        "    }))\n"
        "else:\n"
        "    detail = (\n"
        "        f\"module '{module_name}' imported successfully\"\n"
        "        if not required\n"
        "        else f\"module '{module_name}' exposes required attributes: {', '.join(required)}\"\n"
        "    )\n"
        "    print(json.dumps({\n"
        "        'module': module_name,\n"
        "        'required_attrs': required,\n"
        "        'missing_attrs': [],\n"
        "        'ok': True,\n"
        "        'detail': detail,\n"
        "    }))\n"
    )

    probe_env = os.environ.copy()
    python_paths = [str(PYTHON_DIR), str(SCRIPT_DIR)]
    python_paths.extend(
        str(Path(path).expanduser())
        for path in (extra_python_paths or [])
        if str(path).strip()
    )
    existing_pythonpath = probe_env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    probe_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))

    proc = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=probe_env,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or f"exit status {proc.returncode}"
        return {
            "module": module_name,
            "required_attrs": required,
            "missing_attrs": list(required),
            "ok": False,
            "detail": f"module '{module_name}' probe failed: {stderr}",
        }

    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        return {
            "module": module_name,
            "required_attrs": required,
            "missing_attrs": list(required),
            "ok": False,
            "detail": f"module '{module_name}' probe returned invalid JSON: {exc}",
        }

    return result


def _build_runtime_config(
    stages: list[dict[str, Any]],
    *,
    python_runtime_envs: dict[str, Any] | None = None,
    include_publish: bool = False,
    output_publication: dict[str, Any] | None = None,
    publish_role_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    streams: list[str] = []
    roles: dict[str, Any] = {}

    def add_stream(name: str) -> None:
        if name not in streams:
            streams.append(name)

    for stage in stages:
        phase = str(stage.get("phase") or "").strip()
        handler = str(stage.get("handler") or "").strip()
        queue = str(stage.get("queue") or "").strip()
        if not queue:
            raise ValueError(f"Pipeline stage is missing queue: {stage}")

        if phase == "prepare" and handler == "plugin_phase":
            add_stream(queue)
            roles["prepare"] = {
                "inputs": [_input_stream_spec(queue)],
                "outputs": [],
            }
        elif phase == "collect" and handler == "collect":
            add_stream(queue)
            roles["collect"] = {
                "inputs": [_input_stream_spec(queue)],
                "outputs": [],
            }
        elif phase == "fanout" and handler == "fanout":
            add_stream(queue)
            roles["fanout"] = {
                "inputs": [_input_stream_spec(queue)],
                "outputs": [],
            }
        elif phase == "prefetch" and handler == "prefetch":
            add_stream(queue)
            roles["prefetch"] = {
                "inputs": [_input_stream_spec(queue)],
                "outputs": [],
            }
        elif phase == "schedule" and handler == "schedule":
            add_stream(queue)
            add_stream("release")
            roles["scheduler"] = {
                "inputs": [
                    _input_stream_spec(queue),
                    _input_stream_spec("release", poll_interval_ms=250, block_ms=500),
                ],
                "outputs": [],
                "config": {
                    # Local single-plugin run should schedule against the full advertised
                    # worker capacity instead of production safety headroom.
                    "memory_utilization_percent": 100,
                    # Keep local discovery responsive so users can submit example
                    # runs immediately after the local stack reports ready.
                    "gpu_discovery_interval_secs": 1,
                    "batching_enabled": True,
                    "max_batch_size": 4,
                    "max_batch_wait_ms": 200,
                },
            }
        elif phase == "postprocess" and handler == "plugin_phase":
            add_stream(queue)
            roles["postprocess"] = {
                "inputs": [_input_stream_spec(queue)],
                "outputs": [],
            }
        elif phase == "results" and handler == "persist_results":
            add_stream(queue)
            roles["results"] = {
                "inputs": [_input_stream_spec(queue, max_dequeue_items=8)],
                "outputs": [],
            }

    manifest_declares_publish = any(
        str(stage.get("phase") or "").strip() == "publish" for stage in stages
    )
    if include_publish or manifest_declares_publish:
        publish_queue = "publish"
        for stage in stages:
            if str(stage.get("phase") or "").strip() == "publish":
                publish_queue = str(stage.get("queue") or "").strip()
                break
        try:
            results_index = streams.index("results")
        except ValueError:
            results_index = len(streams)
        if publish_queue not in streams:
            streams.insert(results_index, publish_queue)
        roles["publish"] = {
            "inputs": [_input_stream_spec(publish_queue)],
            "outputs": [],
        }
        if publish_role_config is not None:
            roles["publish"]["config"] = copy.deepcopy(publish_role_config)

    if "results" not in roles:
        if not (include_publish or manifest_declares_publish):
            raise ValueError(
                "Plugin pipeline must include a results stage for local runs"
            )
        add_stream("results")
        roles["results"] = {
            "inputs": [_input_stream_spec("results", max_dequeue_items=8)],
            "outputs": [],
        }

    config = {
        "stream_prefix": "",
        "max_retries": 5,
        "shared_dlq_stream": "dlq",
        "python_runtime_envs": python_runtime_envs or {},
        "streams": streams,
        "roles": roles,
    }
    if output_publication is not None:
        config["output_publication"] = output_publication
    return config


def _runtime_roles_for_pipeline(
    stages: list[dict[str, Any]], *, include_publish: bool = False
) -> list[str]:
    roles: list[str] = []
    manifest_declares_publish = any(
        str(stage.get("phase") or "").strip() == "publish" for stage in stages
    )
    for candidate in (
        "prepare",
        "fanout",
        "prefetch",
        "scheduler",
        "collect",
        "postprocess",
        "publish",
        "results",
    ):
        if candidate == "publish":
            if include_publish or manifest_declares_publish:
                roles.append(candidate)
            continue
        if candidate == "scheduler":
            if any(
                str(stage.get("phase")) == "schedule"
                and str(stage.get("handler")) == "schedule"
                for stage in stages
            ):
                roles.append(candidate)
            continue
        if candidate == "results":
            if (
                include_publish
                or manifest_declares_publish
                or any(
                    str(stage.get("phase")) == "results"
                    and str(stage.get("handler")) == "persist_results"
                    for stage in stages
                )
            ):
                roles.append(candidate)
            continue

        if any(
            str(stage.get("phase")) == candidate
            and (
                str(stage.get("handler")) == "plugin_phase"
                if candidate in {"prepare", "postprocess"}
                else str(stage.get("handler")) == "fanout"
                if candidate == "fanout"
                else str(stage.get("handler")) == "collect"
                if candidate == "collect"
                else str(stage.get("handler")) == candidate
            )
            for stage in stages
        ):
            roles.append(candidate)
    return roles


def _input_stream_spec(
    stream: str,
    *,
    max_dequeue_items: int = 4,
    poll_interval_ms: int = 100,
    block_ms: int = 1000,
    reclaim_idle_ms: int = 60_000,
) -> dict[str, Any]:
    return {
        "stream": stream,
        "max_dequeue_items": max_dequeue_items,
        "poll_interval_ms": poll_interval_ms,
        "block_ms": block_ms,
        "reclaim_idle_ms": reclaim_idle_ms,
    }


def _build_example_request_hint(
    *,
    plugin_root: Path,
    workflow_id: str,
    port: int,
    workspace_root: Path,
) -> dict[str, Any]:
    manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
    content_type = _default_content_type(manifest)
    server_url = f"http://127.0.0.1:{port}"

    if content_type == "application/json":
        request_body, existing_path = _load_or_generate_json_example_request(
            plugin_root,
            manifest,
            workflow_id=workflow_id,
        )
        fixture_path = existing_path or (workspace_root / "generated_request.json")
        if existing_path is None:
            fixture_path.write_text(
                json.dumps(request_body, indent=2) + "\n",
                encoding="utf-8",
            )
        command = (
            f"curl -X POST {server_url}/v1/infer/{workflow_id}/run "
            f"-H 'Content-Type: application/json' "
            f"--data-binary @{fixture_path}"
        )
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                command,
                "",
            ]
        )
        return {
            "command": command,
            "content_type": content_type,
            "fixture_path": fixture_path,
            "script": script,
        }

    fixture_path = _request_fixture_path(plugin_root, content_type)
    fixture = read_json_file(fixture_path)
    if not isinstance(fixture, dict):
        raise ValueError(f"Example request file must be a JSON object: {fixture_path}")

    form_fields = fixture.get("form_fields", {})
    files = fixture.get("files", {})
    args = [
        "curl",
        "-X",
        "POST",
        f"{server_url}/v1/infer/{workflow_id}/run",
    ]
    shell_parts = [*args]
    for key, value in form_fields.items():
        shell_parts.append(f"-F {key}={value}")
    for field_name, relative_path in files.items():
        absolute_path = (plugin_root / relative_path).resolve()
        shell_parts.append(f"-F {field_name}=@{absolute_path}")
    command = " ".join(shell_parts)
    script = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", command, ""])
    return {
        "command": command,
        "content_type": content_type,
        "fixture_path": fixture_path,
        "script": script,
    }


def _pipeline_stages(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    stages = manifest.get("pipeline", {}).get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("Plugin manifest must define a non-empty pipeline.stages list")
    normalized: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Plugin pipeline stages must be objects")
        normalized.append(stage)
    return normalized


def _inference_server_binary_path() -> Path:
    return REPO_ROOT / "target" / "debug" / "inference_server"


def _worker_runtime_binary_path() -> Path:
    return REPO_ROOT / "target" / "debug" / "worker-runtime"


def _load_plugin_contract(plugin_root: Path) -> tuple[dict[str, Any], str, Any]:
    manifest_path = plugin_root / "plugin.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Plugin manifest not found: {manifest_path}")

    manifest = load_plugin_manifest(manifest_path)
    workflow_id = _manifest_id(manifest)
    runtime = manifest.get("runtime", {})
    entrypoint_name = runtime.get("entrypoint")
    if not entrypoint_name:
        raise ValueError(
            f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
        )

    entrypoint_path = plugin_root / entrypoint_name
    module = load_plugin_module(
        workflow_id, entrypoint_path, module_prefix="physicsnemo_serve_plugin_dev"
    )
    return manifest, workflow_id, module


def _derive_run_local_execute_worker_profiles(
    plugin_root: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    probe_payloads = _build_run_local_profile_probe_payloads(plugin_root, manifest)
    base_profile = probe_payloads[0].get("resource_profile")
    if not isinstance(base_profile, dict):
        fallback_profile = _run_local_scheduler_profile_fallback(
            manifest, probe_payloads[0]
        )
        if fallback_profile is None:
            raise ValueError(
                "Example payload resource_profile must be an object unless the workflow matches a scheduler profile"
            )
        return [fallback_profile]

    plugin_phases = plugin_phases_from_manifest(manifest)
    if "prepare" not in plugin_phases:
        return _merge_run_local_execute_worker_profiles(
            [_normalize_run_local_execute_worker_profile(base_profile)]
        )

    _, workflow_id, module = _load_plugin_contract(plugin_root)
    hook = resolve_phase_hook(module, workflow_id, "prepare")
    profiles: list[dict[str, Any]] = []
    for index, payload in enumerate(probe_payloads):
        try:
            result = _invoke_dev_hook("prepare", hook, payload)
            _merge_prepare_output(payload, result)
            profile = payload.get("resource_profile")
            if not isinstance(profile, dict):
                raise ValueError(
                    f"Plugin workflow '{workflow_id}' prepare() did not produce a valid resource_profile"
                )
            profiles.append(_normalize_run_local_execute_worker_profile(profile))
        except Exception:
            if index == 0:
                raise
    return _merge_run_local_execute_worker_profiles(profiles)


def _run_local_scheduler_profile_fallback(
    manifest: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any] | None:
    profile = _match_scheduler_profile_for_payload(manifest, payload)
    if profile is None:
        return None

    memory_mb = _parse_profile_memory_mib(
        ((profile.get("peak") or {}).get("memory.used") or "")
    )
    if memory_mb is None:
        return None

    gpus_required = int(profile.get("gpus.used") or 0)
    runtime = manifest.get("runtime", {})
    return {
        "executor_class": str(runtime.get("executor_class") or ""),
        "device_kind": "gpu" if gpus_required > 0 else "cpu",
        "memory_mb": memory_mb,
        "tags": [],
    }


def _match_scheduler_profile_for_payload(
    manifest: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any] | None:
    profiles_path = REPO_ROOT / "crates" / "worker-runtime" / "config" / "profiles.json"
    profiles_doc = read_json_file(profiles_path)
    profiles = profiles_doc.get("profiles") if isinstance(profiles_doc, dict) else None
    if not isinstance(profiles, list):
        return None

    workflow_id = _manifest_id(manifest)
    candidates = [
        profile
        for profile in profiles
        if isinstance(profile, dict)
        and str(profile.get("workflow") or "").strip() == workflow_id
    ]
    if not candidates and workflow_id.endswith("-fanout"):
        base_workflow = workflow_id.removesuffix("-fanout")
        candidates = [
            profile
            for profile in profiles
            if isinstance(profile, dict)
            and str(profile.get("workflow") or "").strip() == base_workflow
            and profile.get("type") == "ensemble"
        ]
    if not candidates:
        return None

    diagnostic = _payload_string_field(
        payload, "diagnostic_model"
    ) or _payload_string_field(payload, "diagnostic_model_type")
    prognostic = _payload_string_field(
        payload, "prognostic_model"
    ) or _payload_string_field(payload, "prognostic_model_type")
    model = _payload_string_field(payload, "model") or _payload_string_field(
        payload, "model_type"
    )

    if diagnostic and prognostic:
        matched = [
            profile
            for profile in candidates
            if _profile_string_field(profile, "diagnostic_model") == diagnostic
            and _profile_string_field(profile, "prognostic_model") == prognostic
        ]
        return matched[0] if len(matched) == 1 else None

    if model:
        matched = [
            profile
            for profile in candidates
            if _profile_string_field(profile, "model") == model
        ]
        if len(matched) == 1:
            return matched[0]

        defaults = [
            profile for profile in candidates if _profile_is_unconstrained(profile)
        ]
        return defaults[0] if len(defaults) == 1 else None

    defaults = [profile for profile in candidates if _profile_is_unconstrained(profile)]
    if len(defaults) == 1:
        return defaults[0]
    return candidates[0] if not defaults and len(candidates) == 1 else None


def _profile_string_field(profile: dict[str, Any], field_name: str) -> str | None:
    value = str(profile.get(field_name) or "").strip()
    return value or None


def _profile_is_unconstrained(profile: dict[str, Any]) -> bool:
    return not any(
        _profile_string_field(profile, field_name)
        for field_name in ("model", "diagnostic_model", "prognostic_model")
    )


def _payload_string_field(payload: dict[str, Any], field_name: str) -> str | None:
    for source in (
        payload,
        payload.get("parameters")
        if isinstance(payload.get("parameters"), dict)
        else None,
        payload.get("request", {}).get("raw_fields")
        if isinstance(payload.get("request"), dict)
        else None,
    ):
        if not isinstance(source, dict):
            continue
        value = str(source.get(field_name) or "").strip()
        if value:
            return value
    return None


def _parse_profile_memory_mib(raw: str) -> int | None:
    parts = str(raw).split()
    if not parts:
        return None
    try:
        value = int(parts[0])
    except ValueError:
        return None
    if len(parts) == 1 or parts[1].lower() == "mib":
        return value
    return None


def _build_run_local_profile_probe_payloads(
    plugin_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = build_example_payload(
        plugin_root, manifest, run_id="run-local-profile-probe"
    )
    payloads = [payload]
    request = payload.get("request")
    raw_fields = request.get("raw_fields") if isinstance(request, dict) else None
    parameters = payload.get("parameters")
    if not isinstance(raw_fields, dict) or not isinstance(parameters, dict):
        return payloads

    current_device_kind = (
        str(raw_fields.get("device_kind") or parameters.get("device_kind") or "")
        .strip()
        .lower()
    )
    if current_device_kind not in {"cpu", "gpu"}:
        return payloads

    alternate_payload = copy.deepcopy(payload)
    alternate_device_kind = "gpu" if current_device_kind == "cpu" else "cpu"
    alternate_payload["request"]["raw_fields"]["device_kind"] = alternate_device_kind
    alternate_payload["parameters"]["device_kind"] = alternate_device_kind
    payloads.append(alternate_payload)
    return payloads


def _normalize_run_local_execute_worker_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    tags_raw = profile.get("tags", [])
    if isinstance(tags_raw, (list, tuple)):
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
    elif tags_raw:
        tags = [str(tags_raw).strip()]
    else:
        tags = []

    return {
        "executor_class": str(profile.get("executor_class") or ""),
        "device_kind": str(profile.get("device_kind") or "gpu"),
        "memory_mb": int(profile.get("memory_mb") or 4096),
        "tags": tags,
    }


def _merge_run_local_execute_worker_profiles(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_executor_class: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        executor_class = str(profile.get("executor_class") or "").strip()
        if not executor_class:
            continue

        existing = by_executor_class.get(executor_class)
        if existing is None:
            normalized = {
                "executor_class": executor_class,
                "device_kind": str(profile.get("device_kind") or "gpu").strip().lower(),
                "memory_mb": int(profile.get("memory_mb") or 4096),
                "tags": list(profile.get("tags") or []),
            }
            by_executor_class[executor_class] = normalized
            merged.append(normalized)
            continue

        device_kind = str(profile.get("device_kind") or "gpu").strip().lower()
        if existing["device_kind"] != device_kind:
            raise ValueError(
                "run-local discovered conflicting device kinds for executor class "
                f"'{executor_class}': {existing['device_kind']} and {device_kind}"
            )
        existing["memory_mb"] = max(
            existing["memory_mb"], int(profile.get("memory_mb") or 4096)
        )
        existing["tags"] = list(
            dict.fromkeys([*existing["tags"], *list(profile.get("tags") or [])])
        )

    return merged


def _load_schema_documents(
    plugin_root: Path,
    manifest: dict[str, Any],
    *,
    workflow_id: str | None = None,
    module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    # Result schema is optional for Python plugins unless explicitly declared or
    # derived from an output model.
    ingress = manifest.get("ingress", {})
    outputs = manifest.get("outputs", {})
    request_schemas: dict[str, Any] = {}

    if ingress.get("json_schema") and ingress.get("json_schema_inline") is not None:
        raise ValueError(
            "Plugin manifest must not define both ingress.json_schema and ingress.json_schema_inline"
        )
    if ingress.get("form_schema") and ingress.get("form_schema_inline") is not None:
        raise ValueError(
            "Plugin manifest must not define both ingress.form_schema and ingress.form_schema_inline"
        )
    if outputs.get("result_schema") and outputs.get("result_schema_inline") is not None:
        raise ValueError(
            "Plugin manifest must not define both outputs.result_schema and outputs.result_schema_inline"
        )

    if ingress.get("json_schema"):
        request_schemas["application/json"] = _read_schema_document(
            plugin_root,
            str(ingress["json_schema"]),
            "JSON request schema",
        )
    elif ingress.get("json_schema_inline") is not None:
        request_schemas["application/json"] = _normalize_inline_schema(
            ingress["json_schema_inline"],
            "ingress.json_schema_inline",
        )
    if ingress.get("form_schema"):
        request_schemas["multipart/form-data"] = _read_schema_document(
            plugin_root,
            str(ingress["form_schema"]),
            "multipart request schema",
        )
    elif ingress.get("form_schema_inline") is not None:
        request_schemas["multipart/form-data"] = _normalize_inline_schema(
            ingress["form_schema_inline"],
            "ingress.form_schema_inline",
        )

    derived_request_schema = None
    derived_result_schema = None
    derived_form_schema = None
    if not request_schemas or (
        outputs.get("result_schema") is None
        and outputs.get("result_schema_inline") is None
    ):
        derived_request_schema, derived_result_schema, derived_form_schema = (
            _derive_workflow_model_schemas(
                plugin_root,
                manifest,
                workflow_id=workflow_id,
                module=module,
            )
        )

    if "application/json" not in request_schemas and derived_request_schema is not None:
        request_schemas["application/json"] = derived_request_schema
    if "multipart/form-data" not in request_schemas and derived_form_schema is not None:
        request_schemas["multipart/form-data"] = derived_form_schema

    result_schema_path = outputs.get("result_schema")
    if result_schema_path:
        result_schema = _read_schema_document(
            plugin_root,
            str(result_schema_path),
            "result schema",
        )
    elif outputs.get("result_schema_inline") is not None:
        result_schema = _normalize_inline_schema(
            outputs["result_schema_inline"],
            "outputs.result_schema_inline",
        )
    elif derived_result_schema is not None:
        result_schema = derived_result_schema
    else:
        result_schema = None
    return request_schemas, result_schema


def _derive_workflow_model_schemas(
    plugin_root: Path,
    manifest: dict[str, Any],
    *,
    workflow_id: str | None,
    module: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    runtime = manifest.get("runtime", {})
    if str(runtime.get("kind") or "").strip() != "python":
        return None, None, None

    if workflow_id is None:
        workflow_id = _manifest_id(manifest)
    if module is None:
        entrypoint_name = runtime.get("entrypoint")
        if not entrypoint_name:
            return None, None
        module = load_plugin_module(
            workflow_id,
            plugin_root / str(entrypoint_name),
            module_prefix="physicsnemo_serve_plugin_dev_schema",
        )

    workflow = get_workflow_schema_source(module, workflow_id)
    return (
        workflow_request_schema(workflow),
        workflow_result_schema(workflow),
        workflow_form_schema(workflow),
    )


def _read_schema_document(
    plugin_root: Path, relative_path: str, label: str
) -> dict[str, Any]:
    path = plugin_root / relative_path
    if not path.is_file():
        raise ValueError(f"Referenced {label} not found: {path}")

    document = read_json_file(path)
    if not isinstance(document, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object: {path}")
    return document


def _normalize_inline_schema(document: Any, label: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _validate_example_request_fixture(
    plugin_root: Path,
    manifest: dict[str, Any],
    request_schemas: dict[str, Any],
) -> None:
    content_type = _default_content_type(manifest)

    if content_type == "application/json":
        fixture, fixture_path = _load_or_generate_json_example_request(
            plugin_root,
            manifest,
            request_schemas,
        )
        schema = request_schemas.get("application/json")
        if schema is None:
            raise ValueError(
                "Plugin manifest is missing ingress.json_schema or ingress.json_schema_inline"
            )
        _validate_json_against_schema(
            fixture,
            schema,
            (
                f"Example request '{fixture_path}'"
                if fixture_path is not None
                else "Generated example request"
            ),
        )
        return

    if content_type == "multipart/form-data":
        fixture_path = _request_fixture_path(plugin_root, content_type)
        fixture = read_json_file(fixture_path)
        if not isinstance(fixture, dict):
            raise ValueError(
                f"Example request file must be a JSON object: {fixture_path}"
            )

        form_fields = fixture.get("form_fields", {})
        files = fixture.get("files", {})
        if not isinstance(form_fields, dict) or not isinstance(files, dict):
            raise ValueError(
                f"Example request file must contain object fields 'form_fields' and 'files': {fixture_path}"
            )

        schema = request_schemas.get("multipart/form-data")
        if schema is None:
            raise ValueError(
                "Plugin manifest is missing ingress.form_schema or ingress.form_schema_inline"
            )
        _validate_json_against_schema(
            form_fields,
            schema,
            f"Example request '{fixture_path}' form_fields",
        )
        _validate_example_multipart_fixture_files(
            plugin_root, manifest, files, fixture_path
        )
        return

    raise ValueError(f"Unsupported content type for validation: {content_type}")


def _validate_example_multipart_fixture_files(
    plugin_root: Path,
    manifest: dict[str, Any],
    files: dict[str, Any],
    fixture_path: Path,
) -> None:
    ingress_files = {
        str(file_spec.get("name")): file_spec
        for file_spec in manifest.get("ingress", {}).get("files", [])
        if isinstance(file_spec, dict)
    }

    for field_name in files:
        if field_name not in ingress_files:
            raise ValueError(
                f"Example request '{fixture_path}' references unknown file field '{field_name}'"
            )

    for field_name, file_spec in ingress_files.items():
        required = bool(file_spec.get("required", False))
        if required and field_name not in files:
            raise ValueError(
                f"Example request '{fixture_path}' is missing required file field '{field_name}'"
            )

    for field_name, relative_path in files.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(
                f"Example request '{fixture_path}' must map file fields to relative paths"
            )
        path = (plugin_root / relative_path).resolve()
        if not path.is_file():
            raise ValueError(f"Example request file not found: {path}")

        max_size_mb = int(ingress_files[field_name].get("max_size_mb", 0))
        if max_size_mb > 0 and path.stat().st_size > max_size_mb * 1024 * 1024:
            raise ValueError(
                f"Example request file '{path}' exceeds max_size_mb for field '{field_name}'"
            )


def _validate_expected_result_fixture(
    plugin_root: Path, result_schema: dict[str, Any] | None
) -> None:
    fixture_path = _expected_result_fixture_path(plugin_root)
    if fixture_path is None:
        return
    if result_schema is None:
        raise ValueError(
            f"Expected result fixture requires outputs.result_schema, outputs.result_schema_inline, "
            f"or a workflow output_model: {fixture_path}"
        )
    fixture = read_json_file(fixture_path)
    if not isinstance(fixture, dict):
        raise ValueError(
            f"Expected result fixture must be a JSON object: {fixture_path}"
        )
    _validate_json_against_schema(
        fixture,
        result_schema,
        f"Expected result fixture '{fixture_path}'",
    )


def _validate_readiness_config(manifest: dict[str, Any]) -> None:
    readiness = _readiness_config(manifest)
    if not isinstance(readiness, dict):
        raise ValueError("developer.readiness must be an object")

    recommended = readiness.get("recommended_check_phase")
    if recommended is not None and not isinstance(recommended, str):
        raise ValueError("developer.readiness.recommended_check_phase must be a string")

    python_modules = readiness.get("python_modules", [])
    if not isinstance(python_modules, list):
        raise ValueError("developer.readiness.python_modules must be an array")
    for module_name in python_modules:
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(
                "developer.readiness.python_modules entries must be non-empty strings"
            )

    env_specs = readiness.get("env", [])
    if not isinstance(env_specs, list):
        raise ValueError("developer.readiness.env must be an array")
    for spec in env_specs:
        if not isinstance(spec, dict):
            raise ValueError("developer.readiness.env entries must be objects")
        has_name = isinstance(spec.get("name"), str) and bool(str(spec["name"]).strip())
        any_of = spec.get("any_of")
        has_any_of = isinstance(any_of, list) and any(
            isinstance(name, str) and name.strip() for name in any_of
        )
        if not has_name and not has_any_of:
            raise ValueError(
                "developer.readiness.env entries must declare 'name' or non-empty 'any_of'"
            )
        _validate_readiness_kind(spec.get("kind"))

    path_specs = readiness.get("paths", [])
    if not isinstance(path_specs, list):
        raise ValueError("developer.readiness.paths must be an array")
    for spec in path_specs:
        if not isinstance(spec, dict):
            raise ValueError("developer.readiness.paths entries must be objects")
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "developer.readiness.paths entries must declare a non-empty 'path'"
            )
        _validate_readiness_kind(spec.get("kind"))


def _validate_readiness_kind(kind: Any) -> None:
    normalized = str(kind or "path")
    if normalized not in {"file", "dir", "path", "string"}:
        raise ValueError(
            "readiness check kind must be one of: 'file', 'dir', 'path', 'string'"
        )


def _readiness_config(manifest: dict[str, Any]) -> dict[str, Any]:
    developer = manifest.get("developer", {})
    if developer is None:
        return {}
    if not isinstance(developer, dict):
        raise ValueError("developer must be an object")
    readiness = developer.get("readiness", {})
    if readiness is None:
        return {}
    return readiness


def _run_env_check(spec: dict[str, Any]) -> dict[str, Any]:
    kind = str(spec.get("kind") or "string")
    required = bool(spec.get("required", True))
    any_of = spec.get("any_of")

    if isinstance(spec.get("name"), str) and spec["name"].strip():
        env_name = str(spec["name"]).strip()
        raw_value = os.environ.get(env_name)
        ok, detail = _evaluate_env_value(env_name, raw_value, kind, required)
        return {
            "type": "env",
            "name": env_name,
            "required": required,
            "ok": ok,
            "detail": detail,
        }

    names = [
        str(name).strip()
        for name in (any_of or [])
        if isinstance(name, str) and name.strip()
    ]
    for env_name in names:
        raw_value = os.environ.get(env_name)
        if raw_value:
            ok, detail = _evaluate_env_value(env_name, raw_value, kind, required)
            return {
                "type": "env",
                "name": " | ".join(names),
                "required": required,
                "ok": ok,
                "detail": detail,
            }

    if required:
        return {
            "type": "env",
            "name": " | ".join(names),
            "required": True,
            "ok": False,
            "detail": f"none of the env vars are set: {', '.join(names)}",
        }

    return {
        "type": "env",
        "name": " | ".join(names),
        "required": False,
        "ok": True,
        "detail": "optional env check not set",
    }


def _evaluate_env_value(
    env_name: str,
    raw_value: str | None,
    kind: str,
    required: bool,
) -> tuple[bool, str]:
    if not raw_value:
        if required:
            return False, f"required env var '{env_name}' is not set"
        return True, f"optional env var '{env_name}' is not set"

    if kind == "string":
        return True, f"env var '{env_name}' is set"

    path = Path(raw_value).expanduser()
    if kind == "file":
        if path.is_file():
            return True, f"env var '{env_name}' points to file '{path}'"
        return False, f"env var '{env_name}' does not point to a file: {path}"
    if kind == "dir":
        if path.is_dir():
            return True, f"env var '{env_name}' points to directory '{path}'"
        return False, f"env var '{env_name}' does not point to a directory: {path}"

    if path.exists():
        return True, f"env var '{env_name}' points to path '{path}'"
    return False, f"env var '{env_name}' points to a missing path: {path}"


def _run_path_check(plugin_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    kind = str(spec.get("kind") or "path")
    required = bool(spec.get("required", True))
    raw_path = str(spec.get("path") or "").strip()
    path = Path(raw_path)
    if not path.is_absolute():
        path = (plugin_root / path).resolve()

    if kind == "file":
        ok = path.is_file()
    elif kind == "dir":
        ok = path.is_dir()
    else:
        ok = path.exists()

    detail = (
        f"path check passed for '{path}'" if ok else f"path check failed for '{path}'"
    )
    return {
        "type": "path",
        "name": raw_path,
        "required": required,
        "ok": ok or not required,
        "detail": detail if ok else detail,
    }


def _validate_json_against_schema(
    instance: Any, schema: dict[str, Any], label: str
) -> None:
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)
    try:
        validator.validate(instance)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.path)
        suffix = f" at '{path}'" if path else ""
        raise ValueError(
            f"{label} does not conform to schema{suffix}: {exc.message}"
        ) from exc


def _artifact_from_fixture(
    plugin_root: Path, field_name: str, relative_path: str
) -> dict[str, Any]:
    path = (plugin_root / relative_path).resolve()
    if not path.is_file():
        raise ValueError(f"Example request file not found: {path}")

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "field_name": field_name,
        "name": path.name,
        "artifact_id": f"example-{field_name}",
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "storage_path": str(path),
        "original_filename": path.name,
    }


def _merge_prepare_output(payload: dict[str, Any], result: dict[str, Any]) -> None:
    for key in (
        "operation",
        "parameters",
        "request",
        "resource_profile",
        "batch_profile",
        "prefetch_plan",
        "fanout_profile",
        "fanout_items",
        "next_stage_id",
    ):
        if key in result:
            payload[key] = result[key]


def _invoke_dev_hook(phase: str, hook: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if phase == "prepare":
        return _invoke_prepare_hook(hook, payload)
    if phase == "postprocess":
        return _invoke_postprocess_hook(hook, payload)
    if phase == "execute":
        return _invoke_execute_hook(hook, payload)
    raise ValueError(f"Unsupported plugin dev phase invocation: {phase}")


def _invoke_prepare_hook(hook: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if _supports_explicit_contract(hook):
        return serialize_prepare_result(
            hook(build_raw_request(payload), build_prepare_context(payload))
        )

    result = hook(build_context(payload))
    if result is None:
        result = {}
    return serialize_prepare_result(result)


def _invoke_postprocess_hook(hook: Any, payload: dict[str, Any]) -> dict[str, Any]:
    prior_result = payload.get("result")
    if _supports_explicit_contract(hook):
        result = hook(build_prior_result(payload), build_postprocess_context(payload))
    else:
        result = hook(build_context(payload))
    serialized = serialize_postprocess_result(result)
    return _merge_legacy_result_metadata(serialized, prior_result)


def _invoke_execute_hook(hook: Any, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = build_context(payload)
    result = hook(ctx)
    if result is None:
        result = {}
    if not isinstance(result, dict):
        raise TypeError(
            f"Plugin execute hook returned {type(result).__name__}, expected dict"
        )
    return _normalize_legacy_example_result(result, ctx)


def _normalize_legacy_example_result(
    result: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    normalized = dict(result)

    artifacts = normalized.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        normalized["artifacts"] = _legacy_artifacts_from_context(ctx)

    if normalized.get("output_path") is None:
        normalized["output_path"] = _primary_registered_output_path(ctx)

    normalized.setdefault("status", "succeeded")
    normalized.setdefault("artifacts", [])
    normalized.setdefault("output_path", None)
    return normalized


def _legacy_artifacts_from_context(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = ctx.get("outputs")
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


def _primary_registered_output_path(ctx: dict[str, Any]) -> str | None:
    outputs = ctx.get("outputs")
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


def _merge_legacy_result_metadata(
    result: dict[str, Any],
    prior_result: Any,
) -> dict[str, Any]:
    merged = dict(result)
    if not isinstance(prior_result, dict):
        return merged

    for key in ("output_path", "artifacts", "execution_time_seconds"):
        if key not in merged and key in prior_result:
            merged[key] = prior_result[key]
    return merged


def _supports_explicit_contract(hook: Any) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return True
    return len(positional_params) >= 2


def _manifest_id(manifest: dict[str, Any]) -> str:
    workflow_id = str(manifest.get("metadata", {}).get("id") or "").strip()
    if not workflow_id:
        raise ValueError("Plugin manifest is missing metadata.id")
    return workflow_id


def _default_content_type(manifest: dict[str, Any]) -> str:
    content_type = str(
        manifest.get("ingress", {}).get("default_content_type") or ""
    ).strip()
    if not content_type:
        raise ValueError("Plugin manifest is missing ingress.default_content_type")
    return content_type


def _first_stage(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = manifest.get("pipeline", {}).get("stages", [])
    if not stages:
        raise ValueError("Plugin manifest must define at least one pipeline stage")
    first_stage = stages[0]
    if not isinstance(first_stage, dict):
        raise ValueError("Plugin pipeline stages must be objects")
    return first_stage


def _derive_workflow_id(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    if not normalized:
        raise ValueError(
            "Could not derive a workflow id from the plugin directory name"
        )
    return normalized


def _display_name(workflow_id: str) -> str:
    return " ".join(part.capitalize() for part in workflow_id.split("-"))


def _parse_phase_mapping(
    values: list[str],
    *,
    value_label: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_value in values:
        phase, sep, value = raw_value.partition("=")
        phase = phase.strip()
        value = value.strip()
        if sep != "=" or not phase or not value:
            raise ValueError(f"{value_label} overrides must use the form PHASE=VALUE")
        if phase not in INIT_PHASE_NAMES:
            supported = ", ".join(INIT_PHASE_NAMES)
            raise ValueError(f"{value_label} overrides must use one of: {supported}")
        mapping[phase] = value
    return mapping


def _scaffold_manifest(
    workflow_id: str,
    display_name: str,
    content_type: str,
    pipeline_profile: str,
    runtime_profile: str,
    phase_runtime_overrides: list[str],
    executor_class: str | None,
    phase_executor_overrides: list[str],
) -> dict[str, Any]:
    ingress: dict[str, Any] = {}
    if content_type == "application/json":
        if pipeline_profile != "simple":
            ingress["json_schema_inline"] = _json_request_schema()
    else:
        ingress["content_type"] = content_type
        ingress["files"] = [
            {
                "name": "sample_file",
                "required": True,
                "media_types": ["application/octet-stream", "text/plain"],
                "max_size_mb": 1,
            }
        ]

    manifest: dict[str, Any] = {
        "metadata": {
            "id": workflow_id,
            "display_name": display_name,
            "version": "1.0.0",
            "description": f"{display_name} plugin scaffold",
            "tags": ["example"],
        },
        **({"ingress": ingress} if ingress else {}),
        "pipeline": {"profile": pipeline_profile},
    }
    phase_runtime_map = _parse_phase_mapping(
        phase_runtime_overrides,
        value_label="phase runtime",
    )
    phase_executor_map = _parse_phase_mapping(
        phase_executor_overrides,
        value_label="phase executor",
    )

    if runtime_profile == "custom":
        explicit_executor_class = str(executor_class or "").strip()
        runtime: dict[str, Any] = {}
        if explicit_executor_class:
            runtime["executor_class"] = explicit_executor_class
        if phase_runtime_map:
            raise ValueError(
                "Custom runtime scaffolds do not accept --phase-runtime; use --phase-executor instead"
            )
        for phase_name, runtime_field in (
            ("prepare", "prepare_executor_class"),
            ("postprocess", "postprocess_executor_class"),
            ("readiness", "readiness_executor_class"),
        ):
            explicit_phase_executor = phase_executor_map.get(phase_name)
            if explicit_phase_executor:
                runtime[runtime_field] = explicit_phase_executor
        manifest["runtime"] = runtime
        manifest["resources"] = {
            "defaults": _default_resource_defaults(
                explicit_executor_class or DEFAULT_EXECUTOR_CLASS
            )
        }
        return manifest

    if executor_class:
        raise ValueError("--executor-class is only valid with --runtime custom")
    if phase_executor_map:
        raise ValueError("--phase-executor is only valid with --runtime custom")
    supported_phase_runtimes = set(INIT_RUNTIME_PROFILES) - {"custom"}
    for phase_name, phase_runtime in phase_runtime_map.items():
        if phase_runtime not in supported_phase_runtimes:
            supported = ", ".join(sorted(supported_phase_runtimes))
            raise ValueError(
                f"--phase-runtime {phase_name}={phase_runtime} is not supported. "
                f"Supported runtime profiles: {supported}"
            )

    runtime = {"profile": runtime_profile}
    if phase_runtime_map:
        runtime["phases"] = phase_runtime_map
    manifest["runtime"] = runtime
    manifest["resources"] = {
        "defaults": _default_resource_defaults_for_runtime_profile(runtime_profile)
    }
    return manifest


def _default_resource_defaults(executor_class: str) -> dict[str, Any]:
    normalized = executor_class.strip().lower()
    if "gpu" in normalized:
        return {
            "device_kind": "gpu",
            "gpus_required": 1,
            "memory_mb": 16384,
            "cpu_cores": 4,
        }
    return {
        "device_kind": "cpu",
        "gpus_required": 0,
        "memory_mb": 1024,
        "cpu_cores": 1,
    }


def _default_resource_defaults_for_runtime_profile(
    runtime_profile: str,
) -> dict[str, Any]:
    _ = runtime_profile
    return _default_resource_defaults(DEFAULT_EXECUTOR_CLASS)


def _workflow_template(content_type: str, pipeline_profile: str) -> str:
    if content_type == "application/json":
        if pipeline_profile == "simple":
            return """from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow


@dataclass
class ScaffoldInput:
    value: int


@dataclass
class ScaffoldOutput:
    value: int
    doubled: int


class ScaffoldWorkflow(PluginWorkflow):
    input_model = ScaffoldInput
    output_model = ScaffoldOutput

    # Main execution hook.
    def run(self, inputs: ScaffoldInput, ctx) -> ScaffoldOutput:
        return ScaffoldOutput(value=inputs.value, doubled=inputs.value * 2)


WORKFLOW = ScaffoldWorkflow
"""
        return _json_explicit_workflow_template(pipeline_profile)

    return _multipart_workflow_template()


def _json_explicit_workflow_template(pipeline_profile: str) -> str:
    prepare_extras = ""
    output_model = ""
    output_model_assignment = ""
    extra_result_fields = ""
    extra_postprocess = ""
    extra_batch_hook = ""
    sdk_imports = "PluginWorkflow, PostprocessOutcome, PrepareResult, RawRequest"
    main_hook = """
    # Main execution hook.
    def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:
        output_path = ctx.outputs.create(
            "primary",
            filename="result.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "value": inputs.value,
            "doubled": inputs.doubled,%s
        }"""

    if pipeline_profile in {"default", "prefetch"}:
        prepare_extras = "\n            prefetch_plan=[],"
    elif pipeline_profile == "postprocess":
        output_model = """

@dataclass
class ScaffoldOutput:
    value: int
    doubled: int
    postprocessed: bool = False
"""
        output_model_assignment = "\n    output_model = ScaffoldOutput"
        extra_postprocess = """

    # Optional finalization hook.
    def postprocess(self, result, ctx) -> PostprocessOutcome[dict[str, object]]:
        final_payload = dict(result.payload)
        final_payload["postprocessed"] = True
        return PostprocessOutcome(payload=final_payload, status="succeeded")
"""
        extra_result_fields = '\n            "postprocessed": False,'
    elif pipeline_profile == "batch":
        sdk_imports = "BatchExecutionContext, BatchItem, PluginWorkflow, PrepareResult, RawRequest"
        prepare_extras = """
            batch_profile={
                "enabled": True,
                "batch_key": "demo-batch",
                "max_batch_size": 4,
                "max_wait_ms": 50,
                "shared_memory_mb": 256,
                "incremental_memory_mb": 64,
            },"""
        extra_batch_hook = """

    # Batch execution hook. Process compatible items together.
    def run_batch(
        self,
        items: list[BatchItem[ScaffoldInput]],
        ctx: BatchExecutionContext,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for item in items:
            output_path = item.context.outputs.create(
                "primary",
                filename="result.json",
                media_type="application/json",
                primary=True,
            )
            output_path.write_text('{"ok": true}', encoding="utf-8")
            results.append(
                {
                    "value": item.inputs.value,
                    "doubled": item.inputs.doubled,
                }
            )
        return results
"""
        main_hook = ""
    elif pipeline_profile == "ensemble":
        prepare_extras = """
            fanout_profile={
                "item_count": 2,
                "max_in_flight": 1,
            },
            fanout_items=[
                {
                    "item_index": 0,
                    "parameters": {
                        "value": value,
                        "doubled": value * 2,
                        "item_index": 0,
                    },
                },
                {
                    "item_index": 1,
                    "parameters": {
                        "value": value,
                        "doubled": value * 2,
                        "item_index": 1,
                    },
                },
            ],"""
        extra_result_fields = '\n            "item_index": inputs.item_index,'

    return """from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import %s


@dataclass
class ScaffoldInput:
    value: int
    doubled: int
    item_index: int | None = None
%s

class ScaffoldWorkflow(PluginWorkflow):
    input_model = ScaffoldInput%s

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request, ctx) -> PrepareResult:
        value = int(request.raw_fields["value"])
        return PrepareResult(
            inputs={
                "value": value,
                "doubled": value * 2,
            },%s
        )
%s%s%s


WORKFLOW = ScaffoldWorkflow
""" % (
        sdk_imports,
        output_model,
        output_model_assignment,
        prepare_extras,
        main_hook % extra_result_fields if main_hook else "",
        extra_batch_hook,
        extra_postprocess,
    )


def _multipart_workflow_template() -> str:
    return """from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow, PrepareResult, RawRequest


@dataclass
class ScaffoldForm:
    note: str


@dataclass
class ScaffoldInput:
    note: str
    sample_path: str


class ScaffoldWorkflow(PluginWorkflow):
    form_model = ScaffoldForm
    input_model = ScaffoldInput

    # Prepare hook. Normalize inputs and declare framework-managed work.
    def prepare(self, request, ctx) -> PrepareResult:
        artifact = request.input_artifacts[0]
        return PrepareResult(
            inputs={
                "note": request.raw_fields["note"],
                "sample_path": artifact.storage_path,
            }
        )

    # Main execution hook.
    def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:
        output_path = ctx.outputs.create(
            "primary",
            filename="result.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "note": inputs.note,
            "sample_path": inputs.sample_path,
        }


WORKFLOW = ScaffoldWorkflow
"""


def _readme_template(
    plugin_root: Path,
    workflow_id: str,
    content_type: str,
    pipeline_profile: str,
) -> str:
    fixture_name = (
        "examples/default_request.json"
        if content_type == "application/json"
        else "examples/default_request.multipart.json"
    )
    plugin_root_arg = str(plugin_root)
    if content_type == "application/json" and pipeline_profile == "simple":
        model_note = "Request and result schemas are generated from the input/output models in `workflow.py`.\n\n"
    elif content_type == "application/json":
        model_note = (
            "Request schema is generated from the input model in `workflow.py`. "
            "Non-simple pipeline scaffolds use explicit `prepare()` / `run()` hooks so you can control resources and artifacts.\n\n"
        )
    else:
        model_note = "Multipart plugins usually keep their form schema in `plugin.yaml` and implement custom hooks in `workflow.py`.\n\n"
    return (
        f"# {workflow_id}\n\n"
        f"Scaffolded PhysicsNeMo Serve plugin.\n\n"
        f"{model_note}"
        f"## Local checks\n\n"
        f"```bash\n"
        f"python scripts/plugin_dev.py check {plugin_root_arg}\n"
        f"python scripts/plugin_dev.py check-env {plugin_root_arg}\n"
        f"python scripts/plugin_dev.py run-local {plugin_root_arg} --dry-run\n"
        f"```\n\n"
        f"## Authoring\n\n"
        f"- implement the workflow logic in `workflow.py`\n"
        f"- keep a small happy-path request in `{fixture_name}`\n"
        f"- `{fixture_name}` is optional for simple JSON plugins because the dev kit can generate one from `workflow.py`\n\n"
        f"## Examples\n\n"
        f"- `{fixture_name}`\n"
    )


def _json_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {
            "value": {"type": "integer", "minimum": 1},
        },
    }


def _multipart_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["note"],
        "properties": {
            "note": {"type": "string", "minLength": 1},
        },
    }


def _result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["status", "output_path", "artifacts"],
        "properties": {
            "status": {"type": "string"},
            "output_path": {"type": "string"},
            "artifacts": {"type": "array"},
        },
    }


def _expected_result_fixture(workflow_id: str) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "workflow": workflow_id,
        "operation": "run",
        "output_path": "/tmp/example-output.json",
        "artifacts": [
            {
                "name": "primary",
                "media_type": "application/json",
                "storage_path": "/tmp/example-output.json",
            }
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
