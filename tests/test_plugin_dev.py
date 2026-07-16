# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_plugin_dev_module():
    script_path = repo_root() / "scripts" / "plugin_dev.py"
    spec = importlib.util.spec_from_file_location("plugin_dev_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_inference_worker_module():
    script_path = repo_root() / "scripts" / "inference_worker.py"
    spec = importlib.util.spec_from_file_location("inference_worker", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_plugin_sdk_module():
    script_path = repo_root() / "scripts" / "plugin_sdk.py"
    spec = importlib.util.spec_from_file_location("plugin_sdk", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyRedis:
    def setex(self, *_args, **_kwargs):
        return None

    def hset(self, *_args, **_kwargs):
        return None

    def hdel(self, *_args, **_kwargs):
        return None

    def xack(self, *_args, **_kwargs):
        return None

    def xadd(self, *_args, **_kwargs):
        return "1-0"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def update_manifest(plugin_root: Path, updater) -> None:
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    updater(manifest)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def create_class_based_json_plugin(root: Path, plugin_id: str = "demo-json") -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo JSON Plugin
  version: 1.0.0
  description: Demo plugin for CLI testing
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: demo-output
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: execute
""".strip(),
    )
    write_file(
        plugin_root / "schemas" / "request.json",
        """
{
  "type": "object",
  "additionalProperties": false,
  "required": ["value"],
  "properties": {
    "value": { "type": "integer", "minimum": 1 }
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "schemas" / "result.json",
        """
{
  "type": "object",
  "additionalProperties": true,
  "required": ["status", "output_path", "artifacts"],
  "properties": {
    "status": { "type": "string" },
    "output_path": { "type": "string" },
    "artifacts": { "type": "array" }
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "fixtures" / "example_request.json",
        """
{
  "value": 7
}
""".strip(),
    )
    write_file(
        plugin_root / "fixtures" / "expected_result.json",
        """
{
  "status": "succeeded",
  "output_path": "/tmp/demo.json",
  "artifacts": []
}
""".strip(),
    )
    write_file(
        plugin_root / "README.md",
        "# Demo JSON Plugin\n",
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

import os
from pathlib import Path

from plugin_sdk import PluginWorkflow


class DemoWorkflow(PluginWorkflow):
    def prepare(self, ctx):
        value = int(ctx["parameters"]["value"])
        return {
            "parameters": {"value": value, "doubled": value * 2},
        }

    def execute(self, ctx):
        output_root = Path(os.environ["DEFAULT_OUTPUT_DIR"])
        output_path = output_root / ctx["run_id"] / "demo.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "artifacts": [
                {
                    "name": "demo-output",
                    "media_type": "application/json",
                    "storage_path": str(output_path),
                }
            ],
            "value": ctx["parameters"]["value"],
            "doubled": ctx["parameters"]["doubled"],
        }


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    write_file(
        plugin_root / "examples" / "default_request.json",
        """
{
  "value": 7
}
""".strip(),
    )
    return plugin_root


def test_runtime_roles_include_manifest_declared_publish_without_global_publication(
    tmp_path,
):
    plugin_dev = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(tmp_path)
    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    manifest["pipeline"]["stages"].insert(
        -1,
        {
            "id": "publish",
            "phase": "publish",
            "handler": "publish_outputs",
            "queue": "publish",
            "next": "results",
        },
    )
    stages = manifest["pipeline"]["stages"]

    roles = plugin_dev._runtime_roles_for_pipeline(stages, include_publish=False)

    assert "publish" in roles


def create_cleanup_tracking_json_plugin(
    root: Path, plugin_id: str = "demo-json-unload"
) -> Path:
    plugin_root = create_class_based_json_plugin(root, plugin_id=plugin_id)
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

import os
from pathlib import Path

from plugin_sdk import PluginWorkflow


class DemoWorkflow(PluginWorkflow):
    def execute(self, ctx):
        output_root = Path(os.environ["DEFAULT_OUTPUT_DIR"])
        output_path = output_root / ctx["run_id"] / "demo.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "artifacts": [
                {
                    "name": "demo-output",
                    "media_type": "application/json",
                    "storage_path": str(output_path),
                }
            ],
            "value": ctx["parameters"]["value"],
        }

    def cleanup(self) -> None:
        marker_path = Path(os.environ["DEFAULT_OUTPUT_DIR"]) / "cleanup-marker.txt"
        marker_path.write_text("cleaned", encoding="utf-8")


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    return plugin_root


def create_instance_tracking_json_plugin(
    root: Path, plugin_id: str = "demo-json-instance-tracking"
) -> Path:
    plugin_root = create_class_based_json_plugin(root, plugin_id=plugin_id)
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from plugin_sdk import PluginWorkflow

INSTANCE_COUNTER = 0


class DemoWorkflow(PluginWorkflow):
    def __init__(self):
        global INSTANCE_COUNTER
        INSTANCE_COUNTER += 1
        self.instance_id = INSTANCE_COUNTER

    def execute(self, ctx):
        return {
            "status": "succeeded",
            "instance_id": self.instance_id,
            "value": ctx["parameters"]["value"],
        }


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    return plugin_root


def create_cacheable_json_plugin(
    root: Path, plugin_id: str = "demo-json-cacheable"
) -> Path:
    plugin_root = create_class_based_json_plugin(root, plugin_id=plugin_id)
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from plugin_sdk import PluginWorkflow

INSTANCE_COUNTER = 0
LOAD_CALLS = 0
WARMUP_CALLS = 0
PREPARE_MODEL_CACHE_CALLS = 0
CLEANUP_REQUEST_CALLS = 0
CLEANUP_CALLS = 0


def prepare_model_cache(ctx):
    global PREPARE_MODEL_CACHE_CALLS
    PREPARE_MODEL_CACHE_CALLS += 1
    return {
        "model_names": ["demo-model"],
        "workflow_id": ctx["workflow_id"],
    }


class DemoWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["demo-model"]

    def __init__(self):
        global INSTANCE_COUNTER
        INSTANCE_COUNTER += 1
        self.instance_id = INSTANCE_COUNTER
        self.model_loaded = False

    def warmup(self, ctx):
        global LOAD_CALLS, WARMUP_CALLS
        WARMUP_CALLS += 1
        if not self.model_loaded:
            LOAD_CALLS += 1
            self.model_loaded = True
        return {
            "model_names": ["demo-model"],
            "workflow_id": ctx["workflow_id"],
        }

    def execute(self, ctx):
        global LOAD_CALLS
        if not self.model_loaded:
            LOAD_CALLS += 1
            self.model_loaded = True
        return {
            "status": "succeeded",
            "instance_id": self.instance_id,
            "load_calls": LOAD_CALLS,
            "warmup_calls": WARMUP_CALLS,
            "cleanup_request_calls": CLEANUP_REQUEST_CALLS,
            "cleanup_calls": CLEANUP_CALLS,
            "value": ctx["parameters"]["value"],
        }

    def cleanup_request(self) -> None:
        global CLEANUP_REQUEST_CALLS
        CLEANUP_REQUEST_CALLS += 1

    def cleanup(self) -> None:
        global CLEANUP_CALLS
        CLEANUP_CALLS += 1
        self.model_loaded = False


WORKFLOW = DemoWorkflow
""".strip(),
    )
    return plugin_root


def create_device_switching_json_plugin(
    root: Path, plugin_id: str = "demo-device-switch"
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Device Switch Plugin
  version: 1.0.0
  description: Demo plugin that switches execute workers by device_kind
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.cpu.demo
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.cpu.demo
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
    additionalProperties: true
  primary_artifact:
    name: demo-output
    media_type: application/json
developer:
  readiness:
    recommended_check_phase: prepare
""".strip(),
    )
    write_file(
        plugin_root / "schemas" / "request.json",
        """
{
  "type": "object",
  "additionalProperties": false,
  "required": ["value", "device_kind"],
  "properties": {
    "value": { "type": "integer", "minimum": 1 },
    "device_kind": { "type": "string", "enum": ["cpu", "gpu"] }
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

import os
from pathlib import Path

from plugin_sdk import PluginWorkflow


class DemoWorkflow(PluginWorkflow):
    def prepare(self, ctx):
        value = int(ctx["parameters"]["value"])
        device_kind = str(ctx["parameters"]["device_kind"]).strip().lower()
        if device_kind == "gpu":
            resource_profile = {
                "executor_class": "python.gpu.demo",
                "device_kind": "gpu",
                "memory_mb": 16384,
                "tags": ["demo", "gpu"],
            }
        else:
            resource_profile = {
                "executor_class": "python.cpu.demo",
                "device_kind": "cpu",
                "memory_mb": 2048,
                "tags": ["demo", "cpu"],
            }

        return {
            "parameters": {"value": value, "device_kind": device_kind},
            "resource_profile": resource_profile,
        }

    def execute(self, ctx):
        output_root = Path(os.environ["DEFAULT_OUTPUT_DIR"])
        output_path = output_root / ctx["run_id"] / "demo.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "artifacts": [
                {
                    "name": "demo-output",
                    "media_type": "application/json",
                    "storage_path": str(output_path),
                }
            ],
            "value": ctx["parameters"]["value"],
            "device_kind": ctx["parameters"]["device_kind"],
        }


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    write_file(
        plugin_root / "examples" / "default_request.json",
        """
{
  "value": 7,
  "device_kind": "cpu"
}
""".strip(),
    )
    return plugin_root


def create_new_contract_json_plugin(
    root: Path, plugin_id: str = "demo-new-contract"
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo New Contract Plugin
  version: 1.0.0
  description: Demo plugin for new hook contracts
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: postprocess
    - id: postprocess
      phase: postprocess
      handler: plugin_phase
      queue: postprocess
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
    additionalProperties: true
  primary_artifact:
    name: demo-output
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: execute
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import (
    PluginWorkflow,
    PostprocessOutcome,
    PrepareResult,
    PriorResult,
    RawRequest,
)


@dataclass
class DemoInput:
    value: int
    doubled: int


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput

    def prepare(self, request: RawRequest, ctx) -> PrepareResult:
        value = int(request.raw_fields["value"])
        return PrepareResult(inputs={"value": value, "doubled": value * 2})

    def run(self, inputs: DemoInput, ctx):
        output_path = ctx.outputs.create(
            "demo-output",
            filename="demo.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "value": inputs.value,
            "doubled": inputs.doubled,
        }

    def postprocess(self, result: PriorResult[dict], ctx) -> PostprocessOutcome[dict]:
        return PostprocessOutcome(
            payload={
                "value": result.payload["value"],
                "doubled": result.payload["doubled"],
                "postprocessed": True,
            },
            status="succeeded",
        )


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    write_file(
        plugin_root / "examples" / "default_request.json",
        """
{
  "value": 7
}
""".strip(),
    )
    return plugin_root


def create_minimal_inline_json_plugin(
    root: Path, plugin_id: str = "demo-inline"
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Inline Plugin
  version: 1.0.0
  description: Minimal inline-schema plugin for CLI testing
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
    additionalProperties: false
    required: [value]
    properties:
      value:
        type: integer
        minimum: 1
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
    additionalProperties: true
    required: [status, output_path, artifacts]
    properties:
      status:
        type: string
      output_path:
        type: string
      artifacts:
        type: array
  primary_artifact:
    name: demo-output
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: execute
""".strip(),
    )
    write_file(
        plugin_root / "examples" / "default_request.json",
        """
{
  "value": 7
}
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

import os
from pathlib import Path

from plugin_sdk import PluginWorkflow


class DemoWorkflow(PluginWorkflow):
    def prepare(self, ctx):
        value = int(ctx["parameters"]["value"])
        return {
            "parameters": {"value": value, "doubled": value * 2},
        }

    def execute(self, ctx):
        output_root = Path(os.environ["DEFAULT_OUTPUT_DIR"])
        output_path = output_root / ctx["run_id"] / "demo.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "artifacts": [
                {
                    "name": "demo-output",
                    "media_type": "application/json",
                    "storage_path": str(output_path),
                }
            ],
            "value": ctx["parameters"]["value"],
            "doubled": ctx["parameters"]["doubled"],
        }


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    return plugin_root


def create_model_driven_json_plugin(
    root: Path, plugin_id: str = "demo-model", *, with_example_request: bool = True
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Model Plugin
  version: 1.0.0
  description: Model-driven plugin with generated schemas
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: execute
""".strip(),
    )
    if with_example_request:
        write_file(
            plugin_root / "examples" / "default_request.json",
            """
{
  "value": 7
}
""".strip(),
        )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow


@dataclass
class DemoInput:
    value: int


@dataclass
class DemoOutput:
    value: int
    doubled: int


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput
    output_model = DemoOutput

    def run(self, inputs: DemoInput, ctx):
        return DemoOutput(value=inputs.value, doubled=inputs.value * 2)


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    return plugin_root


def create_compact_profile_json_plugin(
    root: Path,
    plugin_id: str = "demo-compact",
    *,
    with_example_request: bool = True,
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Compact Plugin
  version: 1.0.0
  description: Model-driven plugin using preset defaults
pipeline:
  profile: simple
runtime:
  profile: python-test
""".strip(),
    )
    if with_example_request:
        write_file(
            plugin_root / "examples" / "default_request.json",
            """
{
  "value": 7
}
""".strip(),
        )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow


@dataclass
class DemoInput:
    value: int


@dataclass
class DemoOutput:
    value: int
    doubled: int


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput
    output_model = DemoOutput

    def run(self, inputs: DemoInput, ctx):
        return DemoOutput(value=inputs.value, doubled=inputs.value * 2)


WORKFLOW = DemoWorkflow()
""".strip(),
    )
    return plugin_root


def create_class_based_multipart_plugin(
    root: Path, plugin_id: str = "demo-multipart"
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Multipart Plugin
  version: 1.0.0
  description: Demo multipart plugin for CLI testing
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
  form_schema: schemas/request.multipart.json
  files:
    - name: sample_file
      required: true
      media_types: [application/octet-stream, text/plain]
      max_size_mb: 1
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: multipart-output
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: execute
""".strip(),
    )
    write_file(
        plugin_root / "schemas" / "request.multipart.json",
        """
{
  "type": "object",
  "additionalProperties": false,
  "required": ["note"],
  "properties": {
    "note": { "type": "string", "minLength": 1 }
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "schemas" / "result.json",
        """
{
  "type": "object",
  "additionalProperties": true,
  "required": ["status", "output_path", "artifacts"],
  "properties": {
    "status": { "type": "string" },
    "output_path": { "type": "string" },
    "artifacts": { "type": "array" }
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "fixtures" / "sample.txt",
        "fixture-data",
    )
    write_file(
        plugin_root / "fixtures" / "example_request.multipart.json",
        """
{
  "form_fields": {
    "note": "hello"
  },
  "files": {
    "sample_file": "fixtures/sample.txt"
  }
}
""".strip(),
    )
    write_file(
        plugin_root / "fixtures" / "expected_result.json",
        """
{
  "status": "succeeded",
  "output_path": "/tmp/multipart.json",
  "artifacts": []
}
""".strip(),
    )
    write_file(
        plugin_root / "README.md",
        "# Demo Multipart Plugin\n",
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

import os
from pathlib import Path

from plugin_sdk import PluginWorkflow


class DemoMultipartWorkflow(PluginWorkflow):
    def prepare(self, ctx):
        artifact = ctx["request"]["input_artifacts"][0]
        return {
            "parameters": {
                "note": ctx["parameters"]["note"],
                "sample_path": artifact["storage_path"],
            }
        }

    def execute(self, ctx):
        output_root = Path(os.environ["DEFAULT_OUTPUT_DIR"])
        output_path = output_root / ctx["run_id"] / "multipart.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "artifacts": [
                {
                    "name": "multipart-output",
                    "media_type": "application/json",
                    "storage_path": str(output_path),
                }
            ],
            "sample_path": ctx["parameters"]["sample_path"],
        }


def build_workflow():
    return DemoMultipartWorkflow()
""".strip(),
    )
    return plugin_root


def create_class_based_batch_plugin(root: Path, plugin_id: str = "demo-batch") -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Batch Plugin
  version: 1.0.0
  description: Demo batch plugin for worker tests
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

BATCH_CALLS = []


def execute_batch(items, ctx):
    BATCH_CALLS.append([item["run_id"] for item in items])
    return [
        {
            "run_id": item["run_id"],
            "status": "succeeded",
            "output_path": None,
            "artifacts": [],
            "value": item["parameters"]["value"],
        }
        for item in items
    ]
""".strip(),
    )
    return plugin_root


def create_class_based_run_batch_plugin(
    root: Path, plugin_id: str = "demo-run-batch"
) -> Path:
    plugin_root = root / plugin_id
    write_file(
        plugin_root / "plugin.yaml",
        f"""
metadata:
  id: {plugin_id}
  display_name: Demo Run Batch Plugin
  version: 1.0.0
  description: Demo typed run_batch plugin for worker tests
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import BatchExecutionContext, BatchItem, PluginWorkflow

BATCH_CALLS = []


@dataclass
class DemoInput:
    value: int


@dataclass
class DemoOutput:
    value: int
    batch_id_seen: str


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput
    output_model = DemoOutput

    def run_batch(
        self,
        items: list[BatchItem[DemoInput]],
        ctx: BatchExecutionContext,
    ) -> list[DemoOutput]:
        BATCH_CALLS.append(
            {
                "batch_id": ctx.batch_id,
                "batch_size": int(ctx.batch_info.get("batch_size") or 0),
                "run_ids": [item.context.run_id for item in items],
            }
        )
        return [
            DemoOutput(
                value=item.inputs.value * 10,
                batch_id_seen=ctx.batch_id,
            )
            for item in items
        ]


WORKFLOW = DemoWorkflow
""".strip(),
    )
    return plugin_root


def test_plugin_hook_runner_supports_class_based_workflow_object(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_hook_runner.py"
    env = os.environ.copy()
    env["PLUGIN_DIR"] = str(plugin_root.parent)
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    payload = {
        "workflow_id": plugin_root.name,
        "run_id": "run-1",
        "operation": "run",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
    }

    proc = subprocess.run(
        [sys.executable, str(script), "--phase", "prepare"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["parameters"] == {"value": 3, "doubled": 6}


def test_plugin_hook_runner_dispatches_new_contract_prepare_hook(tmp_path: Path):
    plugin_root = create_new_contract_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_hook_runner.py"
    env = os.environ.copy()
    env["PLUGIN_DIR"] = str(plugin_root.parent)
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    payload = {
        "workflow_id": plugin_root.name,
        "run_id": "run-prepare-new",
        "operation": "run",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
    }

    proc = subprocess.run(
        [sys.executable, str(script), "--phase", "prepare"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["parameters"] == {"value": 3, "doubled": 6}


def test_plugin_hook_runner_dispatches_new_contract_postprocess_hook(tmp_path: Path):
    plugin_root = create_new_contract_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_hook_runner.py"
    env = os.environ.copy()
    env["PLUGIN_DIR"] = str(plugin_root.parent)
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    dataset_path = tmp_path / "outputs" / "run-postprocess-new" / "demo.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text('{"ok": true}', encoding="utf-8")

    payload = {
        "workflow_id": plugin_root.name,
        "run_id": "run-postprocess-new",
        "operation": "run",
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 4},
            "input_artifacts": [],
        },
        "result": {
            "status": "succeeded",
            "value": 4,
            "doubled": 8,
            "output_path": str(dataset_path),
            "artifacts": [
                {
                    "name": "demo-output",
                    "media_type": "application/json",
                    "storage_path": str(dataset_path),
                }
            ],
        },
    }

    proc = subprocess.run(
        [sys.executable, str(script), "--phase", "postprocess"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["value"] == 4
    assert data["doubled"] == 8
    assert data["postprocessed"] is True
    assert "run_id" not in data
    assert "artifacts" not in data
    assert "output_path" not in data


def test_plugin_hook_runner_postprocess_includes_registered_outputs(tmp_path: Path):
    plugin_root = create_new_contract_json_plugin(
        tmp_path, plugin_id="demo-postprocess-outputs"
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from plugin_sdk import PluginWorkflow, PostprocessOutcome, PrepareResult, RawRequest


class DemoWorkflow(PluginWorkflow):
    def prepare(self, request: RawRequest, ctx) -> PrepareResult:
        value = int(request.raw_fields["value"])
        return PrepareResult(inputs={"value": value})

    def run(self, inputs, ctx):
        return {"value": int(inputs["value"])}

    def postprocess(self, result, ctx) -> PostprocessOutcome[dict]:
        output_path = ctx.run_dir / "postprocessed.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"ok": true}', encoding="utf-8")
        ctx.outputs.register(
            "postprocessed-output",
            output_path,
            media_type="application/json",
            primary=True,
        )
        return PostprocessOutcome(
            payload={"value": int(result.payload["value"]), "postprocessed": True},
            status="succeeded",
        )


WORKFLOW = DemoWorkflow()
""".strip(),
    )

    script = repo_root() / "scripts" / "plugin_hook_runner.py"
    env = os.environ.copy()
    env["PLUGIN_DIR"] = str(plugin_root.parent)
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    payload = {
        "workflow_id": plugin_root.name,
        "run_id": "run-postprocess-outputs",
        "operation": "run",
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 4},
            "input_artifacts": [],
        },
        "result": {
            "status": "succeeded",
            "value": 4,
        },
    }

    proc = subprocess.run(
        [sys.executable, str(script), "--phase", "postprocess"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    expected_output = (
        tmp_path / "outputs" / "run-postprocess-outputs" / "postprocessed.json"
    )
    assert data["status"] == "succeeded"
    assert data["postprocessed"] is True
    assert data["output_path"] == str(expected_output)
    assert data["artifacts"] == [
        {
            "name": "postprocessed-output",
            "media_type": "application/json",
            "storage_path": str(expected_output),
            "primary": True,
        }
    ]


def test_inference_worker_supports_class_based_plugin_module(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_json_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    executor = module.WorkflowExecutor(DummyRedis())
    result = executor.execute(
        plugin_root.name,
        "run-2",
        {"value": 5, "doubled": 10},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 5, "doubled": 10},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 5},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )

    assert result["status"] == "succeeded"
    assert result["doubled"] == 10


def test_inference_worker_maps_plugin_cancellation_to_cancelled_status(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="demo-cancel")
    write_file(
        plugin_root / "workflow.py",
        """
from plugin_sdk import PluginCancelledError, PluginWorkflow


class CancelledWorkflow(PluginWorkflow):
    def execute(self, ctx):
        raise PluginCancelledError("operator requested shutdown")


WORKFLOW = CancelledWorkflow
""".strip(),
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    result = module.WorkflowExecutor(DummyRedis()).execute(
        plugin_root.name,
        "run-cancelled",
        {"value": 5},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 5},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 5},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )

    assert result["status"] == "cancelled"
    assert result["output_path"] is None
    assert result["error"] == "operator requested shutdown"


def test_inference_worker_execute_creates_fresh_workflow_object_per_request(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_instance_tracking_json_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    executor = module.WorkflowExecutor(DummyRedis())
    first_result = executor.execute(
        plugin_root.name,
        "run-instance-1",
        {"value": 5},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 5},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 5},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )
    second_result = executor.execute(
        plugin_root.name,
        "run-instance-2",
        {"value": 7},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 7},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 7},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )

    assert first_result["instance_id"] > 1
    assert second_result["instance_id"] == first_result["instance_id"] + 1
    assert plugin_root.name in executor._plugin_modules


def test_inference_worker_execute_reuses_cacheable_workflow_and_defers_final_cleanup(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cacheable_json_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_UUID", "gpu-test-0")

    runtime = {
        "entrypoint": "workflow.py",
        "kind": "python",
        "executor_class": "python.test",
    }
    executor = module.WorkflowExecutor(DummyRedis())
    first_result = executor.execute(
        plugin_root.name,
        "run-cache-1",
        {"value": 5},
        payload={
            "workflow_id": plugin_root.name,
            "manifest_version": "1.0.0",
            "operation": "run",
            "parameters": {"value": 5},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 5},
                "input_artifacts": [],
            },
            "runtime": runtime,
        },
    )
    second_result = executor.execute(
        plugin_root.name,
        "run-cache-2",
        {"value": 7},
        payload={
            "workflow_id": plugin_root.name,
            "manifest_version": "1.0.0",
            "operation": "run",
            "parameters": {"value": 7},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 7},
                "input_artifacts": [],
            },
            "runtime": runtime,
        },
    )

    plugin_module = executor._plugin_modules[plugin_root.name]
    assert first_result["instance_id"] == second_result["instance_id"]
    assert first_result["load_calls"] == 1
    assert second_result["load_calls"] == 1
    assert plugin_module.CLEANUP_REQUEST_CALLS == 2
    assert plugin_module.CLEANUP_CALLS == 0

    executor.close()

    assert plugin_module.CLEANUP_CALLS == 1


def test_inference_worker_build_worker_metadata_includes_empty_model_cache(
    monkeypatch,
):
    module = load_inference_worker_module()
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)

    metadata = module.build_worker_metadata(
        stream_name="execute.python.test",
        device_index=0,
        device_name="test-gpu",
        device_uuid="gpu-test-0",
        memory_mb=8192,
        worker_index=0,
        worker_pid=123,
        registry_field="execute.python.test:worker:0:pid:123",
    )

    assert metadata["status"] == "available"
    assert metadata["model_cache"]["schema_version"] == 1
    assert metadata["model_cache"]["scope"] == "process"
    assert metadata["model_cache"]["entries"] == []
    assert metadata["model_cache"]["total_entries"] == 0


def test_inference_worker_warms_enabled_workflow_and_publishes_registry_state(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cacheable_json_plugin(tmp_path, plugin_id="enabled-cacheable")
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", plugin_root.name)
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR", str(tmp_path / "locks")
    )
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_UUID", "gpu-test-0")

    class RecordingRedis(DummyRedis):
        def __init__(self) -> None:
            self.registry_payloads: list[dict] = []

        def hset(self, name, key=None, value=None, mapping=None):
            if name == "gpu:registry" and mapping is None:
                self.registry_payloads.append(json.loads(value))
            return None

    redis_client = RecordingRedis()
    metadata = module.build_worker_metadata(
        stream_name="execute.python.test",
        device_index=0,
        device_name="test-gpu",
        device_uuid="gpu-test-0",
        memory_mb=8192,
        worker_index=0,
        worker_pid=123,
        registry_field="execute.python.test:worker:0:pid:123",
    )
    publisher = module.WorkerRegistryPublisher(
        redis_client,
        stream_name="execute.python.test",
        registry_field="execute.python.test:worker:0:pid:123",
        metadata=metadata,
    )
    executor = module.WorkflowExecutor(redis_client, registry_publisher=publisher)

    warmup_result = executor.warm_enabled_workflow()

    plugin_module = executor._plugin_modules[plugin_root.name]
    assert warmup_result["status"] == "warmed"
    assert plugin_module.PREPARE_MODEL_CACHE_CALLS == 1
    assert plugin_module.WARMUP_CALLS == 1
    assert plugin_module.LOAD_CALLS == 1
    assert redis_client.registry_payloads[0]["status"] == "warming"
    assert redis_client.registry_payloads[-1]["status"] == "available"
    assert redis_client.registry_payloads[-1]["model_cache"]["total_entries"] == 1
    assert (
        redis_client.registry_payloads[-1]["model_cache"]["entries"][0]["workflow_id"]
        == plugin_root.name
    )
    assert (
        redis_client.registry_payloads[-1]["model_cache"]["entries"][0]["cache_key"]
        == plugin_root.name
    )
    assert (
        redis_client.registry_payloads[-1]["model_cache"]["warmup"]["status"]
        == "succeeded"
    )


def test_inference_worker_skips_enabled_workflow_warmup_for_non_matching_executor(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cacheable_json_plugin(tmp_path, plugin_id="enabled-cacheable")
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", plugin_root.name)
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR", str(tmp_path / "locks")
    )
    monkeypatch.setenv("GPU_EXECUTOR_CLASS", "earth2-cpu")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_UUID", "cpu-test")

    class RecordingRedis(DummyRedis):
        def __init__(self) -> None:
            self.registry_payloads: list[dict] = []

        def hset(self, name, key=None, value=None, mapping=None):
            if name == "gpu:registry" and mapping is None:
                self.registry_payloads.append(json.loads(value))
            return None

    redis_client = RecordingRedis()
    metadata = module.build_worker_metadata(
        stream_name="execute.python.test",
        device_index=0,
        device_name="cpu",
        device_uuid="cpu-test",
        memory_mb=8192,
        worker_index=0,
        worker_pid=123,
        registry_field="execute.python.test:worker:0:pid:123",
    )
    publisher = module.WorkerRegistryPublisher(
        redis_client,
        stream_name="execute.python.test",
        registry_field="execute.python.test:worker:0:pid:123",
        metadata=metadata,
    )
    executor = module.WorkflowExecutor(redis_client, registry_publisher=publisher)

    warmup_result = executor.warm_enabled_workflow()

    assert warmup_result == {
        "status": "skipped",
        "reason": "executor_class_mismatch",
        "workflow_id": plugin_root.name,
    }
    assert plugin_root.name not in executor._plugin_modules
    assert redis_client.registry_payloads[-1]["status"] == "available"
    assert (
        redis_client.registry_payloads[-1]["model_cache"]["warmup"]["status"]
        == "skipped"
    )


def test_inference_worker_warmup_uses_model_cache_lock(tmp_path: Path, monkeypatch):
    plugin_root = create_cacheable_json_plugin(tmp_path, plugin_id="enabled-cacheable")
    module = load_inference_worker_module()
    lock_events: list[tuple[str, str]] = []

    class FakeWarmupLock:
        def __init__(self, lock_name: str) -> None:
            self.lock_name = lock_name
            self.is_prepared = False

        def __enter__(self):
            lock_events.append(("enter", self.lock_name))
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            lock_events.append(("exit", self.lock_name))
            return False

        def prepared(self):
            return self.is_prepared

        def mark_prepared(self):
            self.is_prepared = True
            lock_events.append(("prepared", self.lock_name))

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", plugin_root.name)
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR", str(tmp_path / "locks")
    )
    monkeypatch.setenv("GPU_EXECUTOR_CLASS", "python.test")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_UUID", "gpu-test-0")
    monkeypatch.setattr(module, "ModelCacheWarmupLock", FakeWarmupLock)

    executor = module.WorkflowExecutor(DummyRedis())

    result = executor.warm_enabled_workflow()

    assert result["status"] == "warmed"
    assert lock_events == [("enter", plugin_root.name), ("exit", plugin_root.name)]
    assert plugin_root.name in executor._plugin_modules


def test_inference_worker_builds_legacy_envelope_from_registered_outputs(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_new_contract_json_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    executor = module.WorkflowExecutor(DummyRedis())
    result = executor.execute(
        plugin_root.name,
        "run-legacy-envelope",
        {"value": 5, "doubled": 10},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 5, "doubled": 10},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 5},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )

    expected_output_path = tmp_path / "outputs" / "run-legacy-envelope" / "demo.json"
    assert result["status"] == "succeeded"
    assert result["value"] == 5
    assert result["doubled"] == 10
    assert result["output_path"] == str(expected_output_path)
    assert result["artifacts"] == [
        {
            "name": "demo-output",
            "media_type": "application/json",
            "storage_path": str(expected_output_path),
            "primary": True,
        }
    ]


def test_inference_worker_batch_cancels_items_for_terminal_parent_without_running_hook(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_batch_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    class ParentTerminalRedis(DummyRedis):
        def exists(self, key: str) -> int:
            return 1 if key == "parent_terminal:parent-run" else 0

    executor = module.WorkflowExecutor(ParentTerminalRedis())
    result = executor.execute(
        plugin_root.name,
        "batch-1",
        {},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "items": [
                {
                    "run_id": "parent-run:item:0",
                    "payload": {
                        "run_id": "parent-run:item:0",
                        "workflow_id": plugin_root.name,
                        "parent_run_id": "parent-run",
                        "operation": "run",
                        "parameters": {"value": 1},
                    },
                },
                {
                    "run_id": "parent-run:item:1",
                    "payload": {
                        "run_id": "parent-run:item:1",
                        "workflow_id": plugin_root.name,
                        "parent_run_id": "parent-run",
                        "operation": "run",
                        "parameters": {"value": 2},
                    },
                },
            ],
        },
    )

    assert result["status"] == "cancelled"
    assert result["skipped_reason"] == "parent_run_terminal"
    assert [entry["run_id"] for entry in result["batch_results"]] == [
        "parent-run:item:0",
        "parent-run:item:1",
    ]
    for entry in result["batch_results"]:
        item_result = entry["result"]
        assert item_result["status"] == "cancelled"
        assert item_result["parent_run_id"] == "parent-run"
        assert item_result["skipped_reason"] == "parent_run_terminal"
        assert item_result["output_path"] is None
        assert item_result["artifacts"] == []

    assert executor._plugin_modules[plugin_root.name].BATCH_CALLS == []


def test_inference_worker_batch_only_cancels_terminal_parent_items(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_batch_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    class ParentTerminalRedis(DummyRedis):
        def exists(self, key: str) -> int:
            return 1 if key == "parent_terminal:parent-terminal" else 0

    executor = module.WorkflowExecutor(ParentTerminalRedis())
    result = executor.execute(
        plugin_root.name,
        "batch-2",
        {},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "items": [
                {
                    "run_id": "parent-terminal:item:0",
                    "payload": {
                        "run_id": "parent-terminal:item:0",
                        "workflow_id": plugin_root.name,
                        "parent_run_id": "parent-terminal",
                        "operation": "run",
                        "parameters": {"value": 1},
                    },
                },
                {
                    "run_id": "parent-active:item:0",
                    "payload": {
                        "run_id": "parent-active:item:0",
                        "workflow_id": plugin_root.name,
                        "parent_run_id": "parent-active",
                        "operation": "run",
                        "parameters": {"value": 2},
                    },
                },
            ],
        },
    )

    assert result["status"] == "cancelled"
    assert [entry["run_id"] for entry in result["batch_results"]] == [
        "parent-terminal:item:0",
        "parent-active:item:0",
    ]
    cancelled_result = result["batch_results"][0]["result"]
    active_result = result["batch_results"][1]["result"]
    assert cancelled_result["status"] == "cancelled"
    assert cancelled_result["parent_run_id"] == "parent-terminal"
    assert cancelled_result["skipped_reason"] == "parent_run_terminal"
    assert active_result["status"] == "succeeded"
    assert active_result["value"] == 2
    assert executor._plugin_modules[plugin_root.name].BATCH_CALLS == [
        ["parent-active:item:0"]
    ]


def test_inference_worker_supports_class_based_run_batch_hook(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_run_batch_plugin(tmp_path)
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    executor = module.WorkflowExecutor(DummyRedis())
    result = executor.execute(
        plugin_root.name,
        "batch-run-1",
        {},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "batch_info": {
                "batch_id": "batch-run-1",
                "batch_size": 2,
                "flush_reason": "max_batch_size",
            },
            "items": [
                {
                    "run_id": "batch-run-1:item:0",
                    "payload": {
                        "run_id": "batch-run-1:item:0",
                        "workflow_id": plugin_root.name,
                        "operation": "run",
                        "parameters": {"value": 2},
                    },
                },
                {
                    "run_id": "batch-run-1:item:1",
                    "payload": {
                        "run_id": "batch-run-1:item:1",
                        "workflow_id": plugin_root.name,
                        "operation": "run",
                        "parameters": {"value": 3},
                    },
                },
            ],
        },
    )

    assert result["status"] == "succeeded"
    assert [entry["run_id"] for entry in result["batch_results"]] == [
        "batch-run-1:item:0",
        "batch-run-1:item:1",
    ]
    assert result["batch_results"][0]["result"]["value"] == 20
    assert result["batch_results"][1]["result"]["value"] == 30
    assert result["batch_results"][0]["result"]["batch_id_seen"] == "batch-run-1"
    assert executor._plugin_modules[plugin_root.name].BATCH_CALLS == [
        {
            "batch_id": "batch-run-1",
            "batch_size": 2,
            "run_ids": ["batch-run-1:item:0", "batch-run-1:item:1"],
        }
    ]


def test_inference_worker_routes_batch_item_completion_from_execute_stage():
    module = load_inference_worker_module()
    item_payload = {
        "run_id": "batch-run-1:item:0",
        "workflow_id": "demo-batch",
        "operation": "run",
        "parameters": {"value": 2},
        "output_publication": {"enabled": True},
        "stage_context": {
            "current_stage_id": "batch",
            "current_phase": "batch",
            "pipeline": [
                {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "queue": "schedule",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.demo",
                    "next": "publish",
                },
                {
                    "id": "publish",
                    "phase": "publish",
                    "queue": "publish",
                    "next": "results",
                },
                {"id": "results", "phase": "results", "queue": "results", "next": None},
            ],
        },
    }
    batch_payload = {
        "batch_id": "batch-run-1",
        "batch_info": {"batch_id": "batch-run-1", "batch_size": 1},
        "output_publication": {"enabled": True},
        "stage_context": {
            **item_payload["stage_context"],
            "current_stage_id": "execute",
            "current_phase": "execute",
        },
    }
    batch_result = {
        "run_id": "batch-run-1",
        "status": "succeeded",
        "batch_results": [
            {
                "run_id": "batch-run-1:item:0",
                "payload": item_payload,
                "result": {
                    "run_id": "batch-run-1:item:0",
                    "status": "succeeded",
                    "artifacts": [
                        {
                            "name": "forecast_dataset",
                            "media_type": "application/x-zarr",
                            "storage_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                            "primary": True,
                        }
                    ],
                    "output_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                    "batch_info": {"batch_id": "batch-run-1", "batch_size": 1},
                },
            }
        ],
    }

    outputs = module._build_batch_primary_outputs(
        "execute.demo", batch_payload, batch_result
    )

    assert len(outputs) == 1
    stream_name, payload, stage, run_id = outputs[0]
    assert stream_name == "publish"
    assert stage == "publish"
    assert run_id == "batch-run-1:item:0"
    assert payload["stage_context"]["current_stage_id"] == "publish"
    assert payload["stage_context"]["current_phase"] == "publish"
    assert payload["result"]["run_id"] == "batch-run-1:item:0"
    assert not module._should_persist_run_status_after_execute(
        module._batch_item_completion_payload(batch_payload, item_payload),
        batch_result["batch_results"][0]["result"],
    )


def test_inference_worker_routes_batch_item_through_declared_intermediate_stage():
    module = load_inference_worker_module()
    item_payload = {
        "run_id": "batch-run-1:item:0",
        "workflow_id": "demo-batch",
        "operation": "run",
        "parameters": {"value": 2},
        "output_publication": {"enabled": True},
        "stage_context": {
            "current_stage_id": "batch",
            "current_phase": "batch",
            "pipeline": [
                {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "queue": "schedule",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.demo",
                    "next": "postprocess",
                },
                {
                    "id": "postprocess",
                    "phase": "postprocess",
                    "queue": "postprocess",
                    "next": "publish",
                },
                {
                    "id": "publish",
                    "phase": "publish",
                    "queue": "publish",
                    "next": "results",
                },
                {"id": "results", "phase": "results", "queue": "results", "next": None},
            ],
        },
    }
    batch_payload = {
        "batch_id": "batch-run-1",
        "batch_info": {"batch_id": "batch-run-1", "batch_size": 1},
        "output_publication": {"enabled": True},
        "stage_context": {
            **item_payload["stage_context"],
            "current_stage_id": "execute",
            "current_phase": "execute",
        },
    }
    batch_result = {
        "run_id": "batch-run-1",
        "status": "succeeded",
        "batch_results": [
            {
                "run_id": "batch-run-1:item:0",
                "payload": item_payload,
                "result": {
                    "run_id": "batch-run-1:item:0",
                    "status": "succeeded",
                    "artifacts": [
                        {
                            "name": "forecast_dataset",
                            "media_type": "application/x-zarr",
                            "storage_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                            "primary": True,
                        }
                    ],
                    "output_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                    "batch_info": {"batch_id": "batch-run-1", "batch_size": 1},
                },
            }
        ],
    }

    outputs = module._build_batch_primary_outputs(
        "execute.demo", batch_payload, batch_result
    )

    assert len(outputs) == 1
    stream_name, payload, stage, run_id = outputs[0]
    assert stream_name == "postprocess"
    assert stage == "postprocess"
    assert run_id == "batch-run-1:item:0"
    assert payload["stage_context"]["current_stage_id"] == "postprocess"
    assert payload["stage_context"]["current_phase"] == "postprocess"
    publish_stages = [
        pipeline_stage
        for pipeline_stage in payload["stage_context"]["pipeline"]
        if pipeline_stage.get("phase") == "publish"
    ]
    assert len(publish_stages) == 1, "no duplicate publish stage may be inserted"
    assert publish_stages[0]["next"] == "results"
    postprocess_stage = next(
        pipeline_stage
        for pipeline_stage in payload["stage_context"]["pipeline"]
        if pipeline_stage["id"] == "postprocess"
    )
    assert postprocess_stage["next"] == "publish"


def test_inference_worker_synthesizes_publish_stage_for_batch_publication(monkeypatch):
    module = load_inference_worker_module()
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON",
        json.dumps(
            {
                "enabled": True,
                "storage": {
                    "type": "s3",
                    "bucket": "bucket",
                    "prefix": "runs",
                    "region": "us-east-1",
                },
            }
        ),
    )
    item_payload = {
        "run_id": "batch-run-1:item:0",
        "workflow_id": "demo-batch",
        "operation": "run",
        "parameters": {"value": 2},
        "stage_context": {
            "current_stage_id": "batch",
            "current_phase": "batch",
            "pipeline": [
                {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "queue": "schedule",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.demo",
                    "next": "results",
                },
                {"id": "results", "phase": "results", "queue": "results", "next": None},
            ],
        },
    }
    batch_payload = {
        "batch_id": "batch-run-1",
        "batch_info": {"batch_id": "batch-run-1", "batch_size": 1},
        "stage_context": {
            **item_payload["stage_context"],
            "current_stage_id": "execute",
            "current_phase": "execute",
        },
    }
    batch_result = {
        "run_id": "batch-run-1",
        "status": "succeeded",
        "batch_results": [
            {
                "run_id": "batch-run-1:item:0",
                "payload": item_payload,
                "result": {
                    "run_id": "batch-run-1:item:0",
                    "status": "succeeded",
                    "artifacts": [
                        {
                            "name": "forecast_dataset",
                            "media_type": "application/x-zarr",
                            "storage_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                            "primary": True,
                        }
                    ],
                    "output_path": "/outputs/batch-run-1/item-0/forecast.zarr",
                },
            }
        ],
    }

    outputs = module._build_batch_primary_outputs(
        "execute.demo", batch_payload, batch_result
    )

    assert len(outputs) == 1
    stream_name, payload, stage, _run_id = outputs[0]
    assert stream_name == "publish"
    assert stage == "publish"
    publish_stage = next(
        stage
        for stage in payload["stage_context"]["pipeline"]
        if stage["id"] == "publish"
    )
    execute_stage = next(
        stage
        for stage in payload["stage_context"]["pipeline"]
        if stage["id"] == "execute"
    )
    assert execute_stage["next"] == "publish"
    assert publish_stage["next"] == "results"
    assert payload["output_publication"]["target"]["storage"]["prefix"] == (
        "runs/demo-batch/batch-run-1:item:0"
    )


def test_inference_worker_uses_run_batch_for_single_item_execution(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_run_batch_plugin(
        tmp_path, plugin_id="demo-run-batch-single"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    executor = module.WorkflowExecutor(DummyRedis())
    result = executor.execute(
        plugin_root.name,
        "single-run-1",
        {"value": 4},
        payload={
            "workflow_id": plugin_root.name,
            "operation": "run",
            "parameters": {"value": 4},
            "request": {
                "content_type": "application/json",
                "raw_fields": {"value": 4},
                "input_artifacts": [],
            },
            "runtime": {"entrypoint": "workflow.py", "kind": "python"},
        },
    )

    assert result["status"] == "succeeded"
    assert result["value"] == 40
    assert result["batch_id_seen"] == "single-run-1"
    assert executor._plugin_modules[plugin_root.name].BATCH_CALLS == [
        {
            "batch_id": "single-run-1",
            "batch_size": 1,
            "run_ids": ["single-run-1"],
        }
    ]


def test_inference_worker_builds_structured_results_envelope_for_direct_execute_completion():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-1",
        "workflow_id": "demo-json",
        "operation": "run",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-1",
            "status": "succeeded",
            "output_path": "/tmp/run-1/result.json",
            "artifacts": [
                {
                    "name": "primary",
                    "media_type": "application/json",
                    "storage_path": "/tmp/run-1/result.json",
                }
            ],
            "value": 3,
        },
    )

    assert stream_name == "results"
    assert stage == "results"
    assert forwarded["request"]["content_type"] == "application/json"
    assert forwarded["request"]["raw_fields"] == {"value": 3}
    assert forwarded["request"]["parameters"] == {"value": 3}
    assert forwarded["execution"]["run_id"] == "run-1"
    assert forwarded["execution"]["status"] == "succeeded"
    assert forwarded["execution"]["workflow"] == "demo-json"
    assert forwarded["execution"]["gpu_stream"] == "execute.python.test"
    assert forwarded["execution"]["output_path"] == "/tmp/run-1/result.json"
    assert forwarded["execution"]["outputs"][0]["name"] == "primary"
    assert forwarded["payload"] == {"value": 3}


def test_inference_worker_handoffs_successful_execute_to_generic_next_stage_with_updates():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-preprocess",
        "workflow_id": "demo-preprocess",
        "operation": "preprocess",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
        "stage_context": {
            "current_stage_id": "preprocess_execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "preprocess_execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "fanout",
                },
                {
                    "id": "fanout",
                    "phase": "fanout",
                    "queue": "fanout",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-preprocess",
            "status": "succeeded",
            "_pipeline_updates": {
                "operation": "run",
                "parameters": {"value": 6},
                "batch_profile": {"enabled": True, "max_batch_size": 4},
                "fanout_profile": {"item_count": 1, "max_in_flight": 1},
                "fanout_items": [{"item_index": 0, "parameters": {"value": 6}}],
                "stage_context": {"current_phase": "should-not-merge"},
            },
            "debug": "kept out of pipeline updates",
        },
    )

    assert stream_name == "fanout"
    assert stage == "fanout"
    assert forwarded["operation"] == "run"
    assert forwarded["parameters"] == {"value": 6}
    assert forwarded["batch_profile"] == {"enabled": True, "max_batch_size": 4}
    assert forwarded["fanout_profile"] == {"item_count": 1, "max_in_flight": 1}
    assert forwarded["fanout_items"] == [{"item_index": 0, "parameters": {"value": 6}}]
    assert forwarded["result"] == {
        "run_id": "run-preprocess",
        "status": "succeeded",
        "debug": "kept out of pipeline updates",
    }
    assert forwarded["stage_context"]["current_stage_id"] == "fanout"
    assert forwarded["stage_context"]["current_phase"] == "fanout"


def test_inference_worker_handoffs_success_alias_to_publish_stage():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-publish",
        "workflow_id": "demo-publish",
        "operation": "run",
        "parameters": {"value": 3},
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "publish",
                },
                {
                    "id": "publish",
                    "phase": "publish",
                    "queue": "publish",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-publish",
            "status": "success",
            "output_path": "/tmp/run-publish/result.json",
        },
    )

    assert stream_name == "publish"
    assert stage == "publish"
    assert forwarded["result"] == {
        "run_id": "run-publish",
        "status": "success",
        "output_path": "/tmp/run-publish/result.json",
    }
    assert forwarded["stage_context"]["current_stage_id"] == "publish"
    assert forwarded["stage_context"]["current_phase"] == "publish"


def test_inference_worker_handoffs_failed_execute_to_publish_stage():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-publish-failed",
        "workflow_id": "demo-publish",
        "operation": "run",
        "parameters": {"value": 3},
        "output_publication": {
            "target": {
                "artifact": "primary",
                "provider": "s3",
                "storage": {
                    "type": "s3",
                    "bucket": "bucket",
                    "prefix": "outputs/demo-publish/run-publish-failed",
                },
            }
        },
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "publish",
                },
                {
                    "id": "publish",
                    "phase": "publish",
                    "queue": "publish",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-publish-failed",
            "status": "failed",
            "output_path": None,
            "error": "model failed before producing an artifact",
        },
    )

    assert stream_name == "publish"
    assert stage == "publish"
    assert forwarded["result"] == {
        "run_id": "run-publish-failed",
        "status": "failed",
        "output_path": None,
        "error": "model failed before producing an artifact",
    }
    assert forwarded["output_publication"]["target"]["artifact"] == "primary"
    assert forwarded["stage_context"]["current_stage_id"] == "publish"
    assert forwarded["stage_context"]["current_phase"] == "publish"


def test_inference_worker_process_job_does_not_persist_successful_intermediate_handoff():
    module = load_inference_worker_module()
    redis_client = DummyRedis()
    redis_client.hset_calls = []
    redis_client.setex_calls = []

    def record_hset(*args, **kwargs):
        redis_client.hset_calls.append((args, kwargs))
        return None

    def record_setex(*args, **kwargs):
        redis_client.setex_calls.append((args, kwargs))
        return None

    redis_client.hset = record_hset
    redis_client.setex = record_setex

    class FakeExecutor:
        def __init__(self):
            self.redis_client = redis_client

        def execute(self, workflow_name, run_id, parameters, payload=None):
            assert workflow_name == "demo-preprocess"
            assert run_id == "run-preprocess"
            assert parameters == {"value": 3}
            assert payload is not None
            return {
                "run_id": run_id,
                "status": "succeeded",
                "_pipeline_updates": {
                    "operation": "run",
                    "parameters": {"value": 6},
                    "fanout_profile": {"item_count": 1, "max_in_flight": 1},
                    "fanout_items": [{"item_index": 0, "parameters": {"value": 6}}],
                },
            }

    job = {
        "run_id": "run-preprocess",
        "payload": json.dumps(
            {
                "run_id": "run-preprocess",
                "workflow_id": "demo-preprocess",
                "operation": "preprocess",
                "parameters": {"value": 3},
                "stage_context": {
                    "current_stage_id": "preprocess_execute",
                    "current_phase": "execute",
                    "pipeline": [
                        {
                            "id": "preprocess_execute",
                            "phase": "execute",
                            "queue": "execute.python.test",
                            "next": "fanout",
                        },
                        {
                            "id": "fanout",
                            "phase": "fanout",
                            "queue": "fanout",
                            "next": "results",
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": None,
                        },
                    ],
                },
            }
        ),
    }

    result = module.process_job(FakeExecutor(), job)

    assert result["status"] == "succeeded"
    assert redis_client.hset_calls == []
    assert redis_client.setex_calls == []


def test_inference_worker_process_job_does_not_persist_successful_postprocess_handoff():
    module = load_inference_worker_module()
    redis_client = DummyRedis()
    redis_client.hset_calls = []
    redis_client.setex_calls = []

    def record_hset(*args, **kwargs):
        redis_client.hset_calls.append((args, kwargs))
        return None

    def record_setex(*args, **kwargs):
        redis_client.setex_calls.append((args, kwargs))
        return None

    redis_client.hset = record_hset
    redis_client.setex = record_setex

    class FakeExecutor:
        def __init__(self):
            self.redis_client = redis_client

        def execute(self, workflow_name, run_id, parameters, payload=None):
            assert workflow_name == "demo-postprocess"
            assert run_id == "run-postprocess"
            assert parameters == {"value": 3}
            assert payload is not None
            return {
                "run_id": run_id,
                "status": "succeeded",
                "value": 6,
            }

    job = {
        "run_id": "run-postprocess",
        "payload": json.dumps(
            {
                "run_id": "run-postprocess",
                "workflow_id": "demo-postprocess",
                "operation": "run",
                "parameters": {"value": 3},
                "stage_context": {
                    "current_stage_id": "execute",
                    "current_phase": "execute",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute.python.test",
                            "next": "postprocess",
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
                            "next": "results",
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": None,
                        },
                    ],
                },
            }
        ),
    }

    result = module.process_job(FakeExecutor(), job)

    assert result["status"] == "succeeded"
    assert redis_client.hset_calls == []
    assert redis_client.setex_calls == []


def test_inference_worker_process_job_does_not_persist_successful_publish_handoff():
    module = load_inference_worker_module()
    redis_client = DummyRedis()
    redis_client.hset_calls = []
    redis_client.setex_calls = []

    def record_hset(*args, **kwargs):
        redis_client.hset_calls.append((args, kwargs))
        return None

    def record_setex(*args, **kwargs):
        redis_client.setex_calls.append((args, kwargs))
        return None

    redis_client.hset = record_hset
    redis_client.setex = record_setex

    class FakeExecutor:
        def __init__(self):
            self.redis_client = redis_client

        def execute(self, workflow_name, run_id, parameters, payload=None):
            assert workflow_name == "demo-publish"
            assert run_id == "run-publish"
            assert parameters == {"value": 3}
            assert payload is not None
            return {
                "run_id": run_id,
                "status": "succeeded",
                "output_path": "/tmp/output.json",
            }

    job = {
        "run_id": "run-publish",
        "payload": json.dumps(
            {
                "run_id": "run-publish",
                "workflow_id": "demo-publish",
                "operation": "run",
                "parameters": {"value": 3},
                "stage_context": {
                    "current_stage_id": "execute",
                    "current_phase": "execute",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute.python.test",
                            "next": "publish",
                        },
                        {
                            "id": "publish",
                            "phase": "publish",
                            "queue": "publish",
                            "next": "results",
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": None,
                        },
                    ],
                },
            }
        ),
    }

    result = module.process_job(FakeExecutor(), job)

    assert result["status"] == "succeeded"
    assert redis_client.hset_calls == []
    assert redis_client.setex_calls == []


def test_inference_worker_process_job_persists_successful_execute_collect_handoff():
    module = load_inference_worker_module()
    redis_client = DummyRedis()
    redis_client.hset_calls = []
    redis_client.setex_calls = []

    def record_hset(*args, **kwargs):
        redis_client.hset_calls.append((args, kwargs))
        return None

    def record_setex(*args, **kwargs):
        redis_client.setex_calls.append((args, kwargs))
        return None

    redis_client.hset = record_hset
    redis_client.setex = record_setex

    class FakeExecutor:
        def __init__(self):
            self.redis_client = redis_client

        def execute(self, workflow_name, run_id, parameters, payload=None):
            assert workflow_name == "demo-execute"
            assert run_id == "run-execute"
            assert parameters == {"value": 3}
            assert payload is not None
            return {
                "run_id": run_id,
                "status": "succeeded",
                "output_path": "/tmp/output.json",
            }

    job = {
        "run_id": "run-execute",
        "payload": json.dumps(
            {
                "run_id": "run-execute",
                "workflow_id": "demo-execute",
                "operation": "run",
                "parameters": {"value": 3},
                "stage_context": {
                    "current_stage_id": "execute",
                    "current_phase": "execute",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute.python.test",
                            "next": "collect",
                        },
                        {
                            "id": "collect",
                            "phase": "collect",
                            "queue": "collect",
                            "next": "results",
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": None,
                        },
                    ],
                },
            }
        ),
    }

    result = module.process_job(FakeExecutor(), job)

    assert result["status"] == "succeeded"
    assert len(redis_client.hset_calls) == 1
    assert len(redis_client.setex_calls) == 1


def test_inference_worker_failed_execute_without_collect_next_stage_goes_to_results():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-preprocess-failed",
        "workflow_id": "demo-preprocess",
        "operation": "preprocess",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
        "stage_context": {
            "current_stage_id": "preprocess_execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "preprocess_execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "fanout",
                },
                {
                    "id": "fanout",
                    "phase": "fanout",
                    "queue": "fanout",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-preprocess-failed",
            "status": "failed",
            "error": "preprocess failed",
        },
    )

    assert stream_name == "results"
    assert stage == "results"
    assert forwarded["status"] == "failed"
    assert forwarded["execution"]["error"] == "preprocess failed"
    assert forwarded["payload"] == {}


def test_inference_worker_failed_execute_to_postprocess_marks_publication_skipped():
    module = load_inference_worker_module()
    redis_client = DummyRedis()
    redis_client.hset_calls = []
    redis_client.hdel_calls = []
    redis_client.setex_calls = []

    def record_hset(*args, **kwargs):
        redis_client.hset_calls.append((args, kwargs))
        return None

    def record_hdel(*args, **kwargs):
        redis_client.hdel_calls.append((args, kwargs))
        return None

    def record_setex(*args, **kwargs):
        redis_client.setex_calls.append((args, kwargs))
        return None

    redis_client.hset = record_hset
    redis_client.hdel = record_hdel
    redis_client.setex = record_setex

    class FakeExecutor:
        def __init__(self):
            self.redis_client = redis_client

        def execute(self, workflow_name, run_id, parameters, payload=None):
            assert workflow_name == "demo-postprocess-publish"
            assert run_id == "run-postprocess-publish-failed"
            assert parameters == {"value": 3}
            assert payload is not None
            return {
                "run_id": run_id,
                "status": "failed",
                "error": "execute failed",
            }

    job = {
        "run_id": "run-postprocess-publish-failed",
        "payload": json.dumps(
            {
                "run_id": "run-postprocess-publish-failed",
                "workflow_id": "demo-postprocess-publish",
                "operation": "run",
                "parameters": {"value": 3},
                "output_publication": {
                    "target": {
                        "artifact": "primary",
                        "provider": "s3",
                        "storage": {
                            "type": "s3",
                            "bucket": "bucket",
                            "prefix": "outputs/demo/run-postprocess-publish-failed",
                        },
                    }
                },
                "stage_context": {
                    "current_stage_id": "execute",
                    "current_phase": "execute",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute.python.test",
                            "next": "postprocess",
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
                            "next": "publish",
                        },
                        {
                            "id": "publish",
                            "phase": "publish",
                            "queue": "publish",
                            "next": "results",
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": None,
                        },
                    ],
                },
            }
        ),
    }

    result = module.process_job(FakeExecutor(), job)

    assert result["status"] == "failed"
    assert len(redis_client.hset_calls) == 1
    (run_key,) = redis_client.hset_calls[0][0]
    mapping = redis_client.hset_calls[0][1]["mapping"]
    assert run_key == "run:run-postprocess-publish-failed"
    assert mapping["output_publication_status"] == "skipped"
    assert mapping["published_artifact_count"] == "0"
    assert mapping["publish_completed_at"] == mapping["updated_at"]
    assert redis_client.hdel_calls == [
        (("run:run-postprocess-publish-failed", "publish_error"), {})
    ]


def test_inference_worker_builds_structured_results_envelope_for_batch_item_completion():
    module = load_inference_worker_module()
    item_payload = {
        "run_id": "batch-item-1",
        "workflow_id": "demo-batch",
        "operation": "run",
        "parameters": {"value": 5},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 5},
            "input_artifacts": [],
        },
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    outputs = module._build_batch_primary_outputs(
        "execute.python.test",
        item_payload,
        {
            "batch_results": [
                {
                    "run_id": "batch-item-1",
                    "payload": item_payload,
                    "result": {
                        "run_id": "batch-item-1",
                        "status": "succeeded",
                        "artifacts": [
                            {
                                "name": "primary",
                                "media_type": "application/json",
                                "storage_path": "/tmp/batch-item-1/result.json",
                            }
                        ],
                        "batch_info": {
                            "batch_id": "batch-1",
                            "batch_size": 1,
                            "flush_reason": "max_wait_ms",
                        },
                        "value": 5,
                    },
                }
            ]
        },
    )

    assert len(outputs) == 1
    stream_name, forwarded, stage, run_id = outputs[0]
    assert stream_name == "results"
    assert stage == "results"
    assert run_id == "batch-item-1"
    assert forwarded["request"]["raw_fields"] == {"value": 5}
    assert forwarded["request"]["parameters"] == {"value": 5}
    assert forwarded["execution"]["run_id"] == "batch-item-1"
    assert forwarded["execution"]["status"] == "succeeded"
    assert "batch_info" not in forwarded["execution"]
    assert "batch_info" not in forwarded["payload"]
    assert forwarded["execution"]["output_path"] == "/tmp/batch-item-1/result.json"
    assert forwarded["payload"] == {"value": 5}


def test_inference_worker_handoffs_failed_execute_result_to_collect_with_failed_status():
    module = load_inference_worker_module()
    payload = {
        "run_id": "run-collect-failed",
        "workflow_id": "demo-collect",
        "operation": "run",
        "parameters": {"value": 3},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 3},
            "input_artifacts": [],
        },
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute.python.test",
                    "next": "collect",
                },
                {
                    "id": "collect",
                    "phase": "collect",
                    "queue": "collect",
                    "next": "postprocess",
                },
                {
                    "id": "postprocess",
                    "phase": "postprocess",
                    "queue": "postprocess",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": None,
                },
            ],
        },
    }

    stream_name, forwarded, stage = module._build_primary_completion(
        "execute.python.test",
        payload,
        {
            "run_id": "run-collect-failed",
            "status": "failed",
            "error": "CUDA out of memory",
        },
    )

    assert stream_name == "collect"
    assert stage == "collect"
    assert forwarded["result"]["status"] == "failed"
    assert forwarded["result"]["error"] == "CUDA out of memory"
    assert forwarded["stage_context"]["current_stage_id"] == "collect"
    assert forwarded["stage_context"]["current_phase"] == "collect"


def test_inference_worker_process_message_async_emits_results_and_release(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="demo-async-message-worker"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    completion_events: list[str] = []
    after_request_cleanup_called = False

    def fake_after_request_cleanup(*, run_id: str, workflow_id: str) -> None:
        nonlocal after_request_cleanup_called
        assert run_id == "async-run-1"
        assert workflow_id == plugin_root.name
        after_request_cleanup_called = True
        completion_events.append("cleanup")

    @dataclasses.dataclass
    class FakeOutput:
        stream: str
        payload: str
        stage: str

    @dataclasses.dataclass
    class FakeMessage:
        id: str
        run_id: str
        stream: str
        payload: str

    class FakeQueueManager:
        def __init__(self):
            self.forward_calls: list[tuple[FakeMessage, list[FakeOutput]]] = []

        async def forward_many(self, msg, output_targets):
            assert after_request_cleanup_called is True
            completion_events.append("forward_many")
            self.forward_calls.append((msg, list(output_targets)))
            return ["2-0", "2-1"]

    payload = {
        "workflow_id": plugin_root.name,
        "operation": "run",
        "parameters": {"value": 6, "doubled": 12},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 6},
            "input_artifacts": [],
        },
        "resource_id": 7,
        "memory_mb": 8192,
    }
    msg = FakeMessage(
        id="1-0",
        run_id="async-run-1",
        stream="execute.python.test",
        payload=json.dumps(payload),
    )
    qm = FakeQueueManager()
    executor = module.WorkflowExecutor(DummyRedis())
    monkeypatch.setattr(
        module, "_run_after_request_cleanup", fake_after_request_cleanup
    )
    monkeypatch.setattr(module, "Output", FakeOutput)

    asyncio.run(module.process_message_async(executor, qm, msg))

    assert completion_events == ["cleanup", "forward_many"]
    assert len(qm.forward_calls) == 1
    forwarded_msg, output_targets = qm.forward_calls[0]
    assert forwarded_msg == msg
    assert [target.stream for target in output_targets] == ["results", "release"]
    assert [target.stage for target in output_targets] == ["results", "release"]

    results_payload = json.loads(output_targets[0].payload)
    assert results_payload["workflow"] == plugin_root.name
    assert results_payload["run_id"] == "async-run-1"
    assert results_payload["request"]["content_type"] == "application/json"
    assert results_payload["request"]["raw_fields"] == {"value": 6}
    assert results_payload["request"]["parameters"] == {"value": 6, "doubled": 12}
    assert results_payload["execution"]["run_id"] == "async-run-1"
    assert results_payload["execution"]["status"] == "succeeded"
    assert results_payload["execution"]["gpu_stream"] == "execute.python.test"
    assert results_payload["execution"]["outputs"] == [
        {
            "name": "demo-output",
            "media_type": "application/json",
            "storage_path": str(tmp_path / "outputs" / "async-run-1" / "demo.json"),
        }
    ]
    assert results_payload["payload"] == {"value": 6, "doubled": 12}

    output_path = Path(results_payload["execution"]["output_path"])
    assert output_path == tmp_path / "outputs" / "async-run-1" / "demo.json"
    assert output_path.read_text(encoding="utf-8") == '{"ok": true}'

    release_payload = json.loads(output_targets[1].payload)
    assert release_payload["run_id"] == "async-run-1"
    assert release_payload["resource_id"] == 7
    assert release_payload["memory_mb"] == 8192
    assert "gpu_stream" not in release_payload
    assert release_payload["status"] == "succeeded"


def test_inference_worker_process_message_completes_after_request_cleanup(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="demo-sync-message-worker"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    completion_events: list[str] = []
    after_request_cleanup_called = False

    def fake_after_request_cleanup(*, run_id: str, workflow_id: str) -> None:
        nonlocal after_request_cleanup_called
        assert run_id == "sync-run-1"
        assert workflow_id == plugin_root.name
        after_request_cleanup_called = True
        completion_events.append("cleanup")

    class RecordingRedis(DummyRedis):
        def xadd(self, stream, fields):
            assert after_request_cleanup_called is True
            completion_events.append(f"xadd:{stream}:{fields['stage']}")
            return "2-0"

        def xack(self, stream, group, msg_id):
            assert after_request_cleanup_called is True
            completion_events.append(f"xack:{stream}:{group}:{msg_id}")
            return 1

    payload = {
        "workflow_id": plugin_root.name,
        "operation": "run",
        "parameters": {"value": 4},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 4},
            "input_artifacts": [],
        },
        "resource_id": 7,
        "memory_mb": 8192,
    }
    job = {
        "run_id": "sync-run-1",
        "msg_id": "1-0",
        "payload": json.dumps(payload),
    }
    redis_client = RecordingRedis()
    executor = module.WorkflowExecutor(redis_client)
    monkeypatch.setattr(
        module, "_run_after_request_cleanup", fake_after_request_cleanup
    )

    module.process_message(executor, redis_client, "execute.python.test", job)

    assert completion_events == [
        "cleanup",
        "xadd:results:results",
        "xadd:release:release",
        "xack:execute.python.test:workers:1-0",
    ]


def test_inference_worker_process_message_async_keeps_plugin_module_cached_after_request(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cleanup_tracking_json_plugin(
        tmp_path, plugin_id="demo-unload-default"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    @dataclasses.dataclass
    class FakeOutput:
        stream: str
        payload: str
        stage: str

    @dataclasses.dataclass
    class FakeMessage:
        id: str
        run_id: str
        stream: str
        payload: str

    class FakeQueueManager:
        async def forward_many(self, _msg, _output_targets):
            return ["2-0", "2-1"]

    payload = {
        "workflow_id": plugin_root.name,
        "operation": "run",
        "parameters": {"value": 6},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 6},
            "input_artifacts": [],
        },
        "resource_id": 7,
        "memory_mb": 8192,
    }
    msg = FakeMessage(
        id="1-0",
        run_id="async-unload-1",
        stream="execute.python.test",
        payload=json.dumps(payload),
    )
    executor = module.WorkflowExecutor(DummyRedis())
    monkeypatch.setattr(module, "Output", FakeOutput)

    asyncio.run(module.process_message_async(executor, FakeQueueManager(), msg))

    cleanup_marker = tmp_path / "outputs" / "cleanup-marker.txt"
    module_name = f"physicsnemo_serve_plugin_{plugin_root.name.replace('-', '_')}"
    assert cleanup_marker.exists()
    assert cleanup_marker.read_text(encoding="utf-8") == "cleaned"
    assert plugin_root.name in executor._plugin_modules
    assert module_name in sys.modules


def test_inference_worker_process_message_async_logs_cuda_memory_per_request(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cleanup_tracking_json_plugin(
        tmp_path, plugin_id="demo-unload-memory-logs"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    @dataclasses.dataclass
    class FakeOutput:
        stream: str
        payload: str
        stage: str

    @dataclasses.dataclass
    class FakeMessage:
        id: str
        run_id: str
        stream: str
        payload: str

    class FakeQueueManager:
        async def forward_many(self, _msg, _output_targets):
            return ["2-0", "2-1"]

    class FakeCuda:
        def __init__(self) -> None:
            self.reset_peak_calls = 0
            self.empty_cache_calls = 0
            self.ipc_collect_calls = 0

        def is_available(self) -> bool:
            return True

        def memory_allocated(self) -> int:
            return 128 * 1024 * 1024

        def memory_reserved(self) -> int:
            return 256 * 1024 * 1024

        def max_memory_allocated(self) -> int:
            return 512 * 1024 * 1024

        def reset_peak_memory_stats(self) -> None:
            self.reset_peak_calls += 1

        def empty_cache(self) -> None:
            self.empty_cache_calls += 1

        def ipc_collect(self) -> None:
            self.ipc_collect_calls += 1

    fake_cuda = FakeCuda()
    fake_torch = type(sys)("torch")
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info_logs: list[str] = []

    def capture_info(message: str, *args) -> None:
        info_logs.append(message % args if args else message)

    monkeypatch.setattr(module.logger, "info", capture_info)

    payload = {
        "workflow_id": plugin_root.name,
        "operation": "run",
        "parameters": {"value": 6},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 6},
            "input_artifacts": [],
        },
        "resource_id": 7,
        "memory_mb": 8192,
    }
    msg = FakeMessage(
        id="1-0",
        run_id="async-unload-memory-1",
        stream="execute.python.test",
        payload=json.dumps(payload),
    )
    executor = module.WorkflowExecutor(DummyRedis())
    monkeypatch.setattr(module, "Output", FakeOutput)

    asyncio.run(module.process_message_async(executor, FakeQueueManager(), msg))

    snapshot_logs = [entry for entry in info_logs if "CUDA memory snapshot" in entry]
    assert len(snapshot_logs) == 3
    assert "stage=before_execute" in snapshot_logs[0]
    assert "stage=after_execute" in snapshot_logs[1]
    assert "stage=after_request_cleanup" in snapshot_logs[2]
    assert "allocated=128.0MiB" in snapshot_logs[0]
    assert "reserved=256.0MiB" in snapshot_logs[0]
    assert "peak_allocated=512.0MiB" in snapshot_logs[1]
    assert "workflow_unloaded=" not in snapshot_logs[2]
    assert fake_cuda.reset_peak_calls == 1
    assert fake_cuda.ipc_collect_calls == 1
    assert fake_cuda.empty_cache_calls == 1


def test_inference_worker_process_message_async_ignores_removed_unload_flag(
    tmp_path: Path, monkeypatch
):
    plugin_root = create_cleanup_tracking_json_plugin(
        tmp_path, plugin_id="demo-unload-disabled"
    )
    module = load_inference_worker_module()

    monkeypatch.setenv("PLUGIN_DIR", str(plugin_root.parent))
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_UNLOAD_WORKFLOW_AFTER_REQUEST", "1")

    @dataclasses.dataclass
    class FakeOutput:
        stream: str
        payload: str
        stage: str

    @dataclasses.dataclass
    class FakeMessage:
        id: str
        run_id: str
        stream: str
        payload: str

    class FakeQueueManager:
        async def forward_many(self, _msg, _output_targets):
            return ["2-0", "2-1"]

    payload = {
        "workflow_id": plugin_root.name,
        "operation": "run",
        "parameters": {"value": 6},
        "request": {
            "content_type": "application/json",
            "raw_fields": {"value": 6},
            "input_artifacts": [],
        },
        "resource_id": 7,
        "memory_mb": 8192,
    }
    msg = FakeMessage(
        id="1-0",
        run_id="async-unload-2",
        stream="execute.python.test",
        payload=json.dumps(payload),
    )
    executor = module.WorkflowExecutor(DummyRedis())
    monkeypatch.setattr(module, "Output", FakeOutput)

    asyncio.run(module.process_message_async(executor, FakeQueueManager(), msg))

    cleanup_marker = tmp_path / "outputs" / "cleanup-marker.txt"
    module_name = f"physicsnemo_serve_plugin_{plugin_root.name.replace('-', '_')}"
    assert cleanup_marker.exists()
    assert cleanup_marker.read_text(encoding="utf-8") == "cleaned"
    assert plugin_root.name in executor._plugin_modules
    assert module_name in sys.modules
    sys.modules.pop(module_name, None)


def test_inference_worker_main_async_registers_signal_handlers_for_shutdown(
    monkeypatch,
):
    module = load_inference_worker_module()

    monkeypatch.setenv("GPU_STREAM_NAME", "execute.python.test")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_NAME", "cpu")
    monkeypatch.setenv("GPU_MEMORY_MB", "1024")
    monkeypatch.setenv("GPU_DEVICE_UUID", "cpu-0")
    monkeypatch.setenv("GPU_WORKER_INDEX", "0")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    class FakeRedisClient:
        def close(self):
            return None

    def fake_from_url(_url: str) -> FakeRedisClient:
        return FakeRedisClient()

    fake_redis_module = type(
        "FakeRedisModule",
        (),
        {"from_url": staticmethod(fake_from_url)},
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    deregister_calls: list[tuple[str, str | None]] = []
    signal_handlers: dict[int, tuple[object, tuple[object, ...]]] = {}

    async def fake_register_stream_async(
        _qm, _stream_name, _metadata, registry_field=None
    ):
        return None

    async def fake_deregister_stream_async(_qm, stream_name, registry_field=None):
        deregister_calls.append((stream_name, registry_field))

    async def scenario() -> None:
        read_started = asyncio.Event()

        class FakeQueueManagerInstance:
            async def claim_idle_messages(self, *_args, **_kwargs):
                await asyncio.sleep(3600)
                return "0-0", []

            async def read_messages(self, *_args, **_kwargs):
                read_started.set()
                await asyncio.Future()

        fake_qm = FakeQueueManagerInstance()

        class FakeQueueManager:
            @classmethod
            async def from_env(cls):
                return fake_qm

        monkeypatch.setattr(module, "QueueManager", FakeQueueManager)
        monkeypatch.setattr(module, "register_stream_async", fake_register_stream_async)
        monkeypatch.setattr(
            module, "deregister_stream_async", fake_deregister_stream_async
        )

        loop_type = type(asyncio.get_running_loop())

        def fake_add_signal_handler(self, signum, callback, *args):
            signal_handlers[signum] = (callback, args)

        def fake_remove_signal_handler(self, signum):
            signal_handlers.pop(signum, None)
            return True

        monkeypatch.setattr(loop_type, "add_signal_handler", fake_add_signal_handler)
        monkeypatch.setattr(
            loop_type, "remove_signal_handler", fake_remove_signal_handler
        )

        task = asyncio.create_task(module.main_async())
        try:
            await asyncio.wait_for(read_started.wait(), timeout=1)
            assert signal.SIGTERM in signal_handlers, (
                "main_async should register a SIGTERM handler for graceful shutdown"
            )
            assert signal.SIGINT in signal_handlers, (
                "main_async should register a SIGINT handler for graceful shutdown"
            )
            callback, args = signal_handlers[signal.SIGTERM]
            callback(*args)
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert deregister_calls
        assert deregister_calls[0][0] == "execute.python.test"

    asyncio.run(scenario())


def test_inference_worker_main_async_closes_redis_on_shutdown(monkeypatch):
    module = load_inference_worker_module()

    monkeypatch.setenv("GPU_STREAM_NAME", "execute.python.test")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_NAME", "cpu")
    monkeypatch.setenv("GPU_MEMORY_MB", "1024")
    monkeypatch.setenv("GPU_DEVICE_UUID", "cpu-0")
    monkeypatch.setenv("GPU_WORKER_INDEX", "0")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    events: list[str] = []

    class FakeRedisClient:
        def close(self):
            events.append("redis.close")
            return None

    def fake_from_url(_url: str) -> FakeRedisClient:
        return FakeRedisClient()

    fake_redis_module = type(
        "FakeRedisModule",
        (),
        {"from_url": staticmethod(fake_from_url)},
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    async def fake_register_stream_async(
        _qm, _stream_name, _metadata, registry_field=None
    ):
        return None

    async def fake_deregister_stream_async(_qm, _stream_name, registry_field=None):
        events.append("deregister")

    async def scenario() -> None:
        read_started = asyncio.Event()

        class FakeQueueManagerInstance:
            async def claim_idle_messages(self, *_args, **_kwargs):
                await asyncio.sleep(3600)
                return "0-0", []

            async def read_messages(self, *_args, **_kwargs):
                read_started.set()
                await asyncio.Future()

        fake_qm = FakeQueueManagerInstance()

        class FakeQueueManager:
            @classmethod
            async def from_env(cls):
                return fake_qm

        monkeypatch.setattr(module, "QueueManager", FakeQueueManager)
        monkeypatch.setattr(module, "register_stream_async", fake_register_stream_async)
        monkeypatch.setattr(
            module, "deregister_stream_async", fake_deregister_stream_async
        )

        task = asyncio.create_task(module.main_async())
        try:
            await asyncio.wait_for(read_started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(scenario())

    assert "deregister" in events
    assert "redis.close" in events, (
        "main_async should close the Redis client on shutdown"
    )


def test_inference_worker_main_async_accepts_future_returning_read_messages(
    monkeypatch,
):
    module = load_inference_worker_module()

    monkeypatch.setenv("GPU_STREAM_NAME", "execute.python.test")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_NAME", "cpu")
    monkeypatch.setenv("GPU_MEMORY_MB", "1024")
    monkeypatch.setenv("GPU_DEVICE_UUID", "cpu-0")
    monkeypatch.setenv("GPU_WORKER_INDEX", "0")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    class FakeRedisClient:
        def close(self):
            return None

    def fake_from_url(_url: str) -> FakeRedisClient:
        return FakeRedisClient()

    fake_redis_module = type(
        "FakeRedisModule",
        (),
        {"from_url": staticmethod(fake_from_url)},
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    async def fake_register_stream_async(
        _qm, _stream_name, _metadata, registry_field=None
    ):
        return None

    async def fake_deregister_stream_async(_qm, _stream_name, registry_field=None):
        return None

    async def scenario() -> None:
        read_started = asyncio.Event()

        class FakeQueueManagerInstance:
            async def claim_idle_messages(self, *_args, **_kwargs):
                await asyncio.sleep(3600)
                return "0-0", []

            def read_messages(self, *_args, **_kwargs):
                future = asyncio.get_running_loop().create_future()
                read_started.set()
                return future

        fake_qm = FakeQueueManagerInstance()

        class FakeQueueManager:
            @classmethod
            async def from_env(cls):
                return fake_qm

        monkeypatch.setattr(module, "QueueManager", FakeQueueManager)
        monkeypatch.setattr(module, "register_stream_async", fake_register_stream_async)
        monkeypatch.setattr(
            module, "deregister_stream_async", fake_deregister_stream_async
        )

        task = asyncio.create_task(module.main_async())
        try:
            await asyncio.wait_for(read_started.wait(), timeout=1)
            if task.done():
                exc = task.exception()
                raise AssertionError(
                    "main_async should keep running when read_messages returns a Future"
                ) from exc
            task.cancel()
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(scenario())


def test_inference_worker_main_async_tolerates_deregister_error_on_shutdown(
    monkeypatch,
):
    module = load_inference_worker_module()

    monkeypatch.setenv("GPU_STREAM_NAME", "execute.python.test")
    monkeypatch.setenv("GPU_DEVICE_INDEX", "0")
    monkeypatch.setenv("GPU_DEVICE_NAME", "cpu")
    monkeypatch.setenv("GPU_MEMORY_MB", "1024")
    monkeypatch.setenv("GPU_DEVICE_UUID", "cpu-0")
    monkeypatch.setenv("GPU_WORKER_INDEX", "0")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    events: list[str] = []

    class FakeRedisClient:
        def close(self):
            events.append("redis.close")
            return None

    def fake_from_url(_url: str) -> FakeRedisClient:
        return FakeRedisClient()

    fake_redis_module = type(
        "FakeRedisModule",
        (),
        {"from_url": staticmethod(fake_from_url)},
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    async def fake_register_stream_async(
        _qm, _stream_name, _metadata, registry_field=None
    ):
        return None

    async def fake_deregister_stream_async(_qm, _stream_name, registry_field=None):
        events.append("deregister")
        raise RuntimeError("Redis error: broken pipe")

    async def scenario() -> None:
        read_started = asyncio.Event()

        class FakeQueueManagerInstance:
            async def claim_idle_messages(self, *_args, **_kwargs):
                await asyncio.sleep(3600)
                return "0-0", []

            def read_messages(self, *_args, **_kwargs):
                future = asyncio.get_running_loop().create_future()
                read_started.set()
                return future

        fake_qm = FakeQueueManagerInstance()

        class FakeQueueManager:
            @classmethod
            async def from_env(cls):
                return fake_qm

        monkeypatch.setattr(module, "QueueManager", FakeQueueManager)
        monkeypatch.setattr(module, "register_stream_async", fake_register_stream_async)
        monkeypatch.setattr(
            module, "deregister_stream_async", fake_deregister_stream_async
        )

        task = asyncio.create_task(module.main_async())
        try:
            await asyncio.wait_for(read_started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(scenario())

    assert "deregister" in events
    assert "redis.close" in events


def test_plugin_dev_validate_accepts_class_based_plugin(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "valid" in proc.stdout.lower()


def test_plugin_dev_run_example_runs_json_plugin_end_to_end(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env = os.environ.copy()
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["doubled"] == 14


def test_plugin_dev_run_example_runs_multipart_plugin_end_to_end(tmp_path: Path):
    plugin_root = create_class_based_multipart_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env = os.environ.copy()
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["sample_path"].endswith("fixtures/sample.txt")


def _plugin_readiness_modules(plugin_root):
    """Return the list of python_modules from plugin readiness config, or []."""
    import yaml

    manifest_path = plugin_root / "plugin.yaml"
    if not manifest_path.is_file():
        return []
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    return manifest.get("developer", {}).get("readiness", {}).get("python_modules", [])


def _plugin_deps_available(plugin_root):
    """Check if all readiness python_modules are importable."""
    for module_name in _plugin_readiness_modules(plugin_root):
        try:
            __import__(module_name)
        except Exception:
            return False
    return True


def test_plugin_dev_validate_accepts_shipped_plugins():
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugins_root = repo_root() / "plugins"

    for plugin_root in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
        if not _plugin_deps_available(plugin_root):
            continue
        proc = subprocess.run(
            [sys.executable, str(script), "validate", str(plugin_root)],
            text=True,
            capture_output=True,
            cwd=repo_root(),
            check=False,
        )

        assert proc.returncode == 0, f"{plugin_root.name}: {proc.stderr}"


def test_plugin_dev_build_example_payload_omits_empty_resource_profile_for_scheduler_fallback():
    module = load_plugin_dev_module()
    plugin_root = repo_root() / "plugins" / "e2s-deterministic"
    manifest, _, _ = module._load_plugin_contract(plugin_root)

    payload = module.build_example_payload(
        plugin_root, manifest, run_id="scheduler-profile-fallback"
    )

    assert payload["workflow_id"] == "e2s-deterministic"
    assert payload["resource_profile"] is None


def test_plugin_dev_run_local_plan_uses_scheduler_profile_for_empty_resource_defaults(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    plugin_root = repo_root() / "plugins" / "e2s-deterministic"
    workspace = tmp_path / "run-local-e2s-deterministic"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    launcher = next(
        process
        for process in plan["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert launcher["env"]["PHYSICSNEMO_SERVE_EXECUTOR_CLASSES"] == "earth2-gpu"
    assert plan["execute_registration_streams"] == [
        "execute.earth2-gpu:gpu:local:e2s-deterministic-run-local:0"
    ]


@pytest.mark.parametrize("value", ["unsafe", "null"])
def test_plugin_manifest_configuration_must_be_an_object(tmp_path: Path, value: str):
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import load_plugin_manifest  # type: ignore

    manifest_path = tmp_path / "plugin.yaml"
    manifest_path.write_text(f"configuration: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configuration must be an object"):
        load_plugin_manifest(manifest_path)


def test_shipped_plugins_have_local_docs_and_example_fixtures():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import load_plugin_manifest  # type: ignore

    plugins_root = repo_root() / "plugins"

    for plugin_root in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
        readme_path = plugin_root / "README.md"
        assert readme_path.is_file(), f"{plugin_root.name} is missing README.md"

        manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
        content_types = manifest.get("ingress", {}).get("content_types", [])
        assert manifest.get("runtime", {}).get("entrypoint") == "workflow.py"
        assert (plugin_root / "workflow.py").is_file(), (
            f"{plugin_root.name} is missing workflow.py"
        )
        assert not (plugin_root / "plugin.py").exists(), (
            f"{plugin_root.name} still uses plugin.py"
        )

        if "application/json" in content_types:
            example_path = plugin_root / "examples" / "default_request.json"
            legacy_path = plugin_root / "fixtures" / "example_request.json"
            assert not legacy_path.exists(), (
                f"{plugin_root.name} still uses fixtures/example_request.json"
            )
            if example_path.is_file():
                fixture = json.loads(example_path.read_text(encoding="utf-8"))
                assert isinstance(fixture, dict), (
                    f"{plugin_root.name} JSON example must be an object"
                )

        if "multipart/form-data" in content_types:
            fixture_path = plugin_root / "examples" / "default_request.multipart.json"
            legacy_path = plugin_root / "fixtures" / "example_request.multipart.json"
            assert not legacy_path.exists(), (
                f"{plugin_root.name} still uses fixtures/example_request.multipart.json"
            )
            assert fixture_path.is_file(), (
                f"{plugin_root.name} is missing an example multipart request"
            )
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            for relative_path in fixture.get("files", {}).values():
                assert (plugin_root / relative_path).is_file(), (
                    f"{plugin_root.name} fixture file is missing: {relative_path}"
                )

        assert not (plugin_root / "fixtures" / "expected_result.json").exists(), (
            f"{plugin_root.name} still uses fixtures/expected_result.json"
        )


def test_shipped_plugin_readmes_prefer_simplified_commands():
    plugins_root = repo_root() / "plugins"

    for plugin_root in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
        content = (plugin_root / "README.md").read_text(encoding="utf-8")
        assert "python scripts/plugin_dev.py check " in content
        assert "python scripts/plugin_dev.py check-env " in content
        if "python scripts/plugin_dev.py run-local " in content:
            assert "python scripts/plugin_dev.py run-local " in content


def test_shipped_plugins_expose_workflow_objects():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import (  # type: ignore
        get_workflow_instance,
        load_plugin_manifest,
        load_plugin_module,
    )

    plugins_root = repo_root() / "plugins"
    for plugin_root in sorted(path for path in plugins_root.iterdir() if path.is_dir()):
        if not _plugin_deps_available(plugin_root):
            continue
        manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
        workflow_id = manifest["metadata"]["id"]
        entrypoint = plugin_root / manifest["runtime"]["entrypoint"]
        module = load_plugin_module(
            workflow_id,
            entrypoint,
            module_prefix="physicsnemo_serve_plugin_examples_test",
        )

        workflow = get_workflow_instance(module, workflow_id)
        assert workflow is not None
        assert callable(getattr(workflow, "execute", None))


def test_shipped_plugins_opt_into_process_cache():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import (  # type: ignore
        get_workflow_instance,
        load_plugin_manifest,
        load_plugin_module,
    )

    for plugin_root in sorted((repo_root() / "plugins").iterdir()):
        if not (plugin_root / "plugin.yaml").is_file():
            continue
        if not _plugin_deps_available(plugin_root):
            continue
        manifest = load_plugin_manifest(plugin_root / "plugin.yaml")
        workflow_id = manifest["metadata"]["id"]
        entrypoint = plugin_root / manifest["runtime"]["entrypoint"]
        module = load_plugin_module(
            workflow_id,
            entrypoint,
            module_prefix="physicsnemo_serve_plugin_cache_contract_test",
        )

        workflow = get_workflow_instance(module, workflow_id)
        assert getattr(workflow, "cache_scope", None) == "process", workflow_id
        assert callable(getattr(workflow, "cleanup_request", None)), workflow_id
        if getattr(workflow, "model_cache_names", None):
            assert callable(getattr(module, "prepare_model_cache", None)), workflow_id
            assert callable(getattr(workflow, "warmup", None)), workflow_id


def test_plugin_runtime_build_context_includes_batch_info():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import build_context  # type: ignore

    ctx = build_context(
        {
            "run_id": "batch-1",
            "workflow_id": "demo-batch",
            "batch_info": {
                "batch_id": "batch-1",
                "batch_size": 2,
                "flush_reason": "max_batch_size",
            },
        }
    )

    assert ctx["batch_info"]["batch_id"] == "batch-1"
    assert ctx["batch_info"]["batch_size"] == 2
    assert ctx["batch_info"]["flush_reason"] == "max_batch_size"


def test_plugin_runtime_build_context_exposes_abort_requested_for_parent_run():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import build_context  # type: ignore

    class FakeRedis:
        def exists(self, key: str) -> int:
            return 1 if key == "parent_terminal:parent-run" else 0

    ctx = build_context(
        {
            "run_id": "parent-run:item:0",
            "parent_run_id": "parent-run",
            "workflow_id": "fanout-demo",
            "service_objects": {"redis_client": FakeRedis()},
        }
    )

    assert callable(ctx["abort_requested"])
    deadline = time.monotonic() + 1
    while not ctx["abort_requested"]() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ctx["abort_requested"]() is True


def test_plugin_runtime_resolve_manifest_rejects_disabled_workflow(
    tmp_path: Path, monkeypatch
):
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import resolve_plugin_manifest  # type: ignore

    create_class_based_json_plugin(tmp_path, plugin_id="enabled-plugin")
    create_class_based_json_plugin(tmp_path, plugin_id="disabled-plugin")
    monkeypatch.setenv("PLUGIN_DIR", str(tmp_path))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "enabled-plugin")

    with pytest.raises(ValueError, match="disabled-plugin"):
        resolve_plugin_manifest("disabled-plugin")


def test_inference_worker_resolve_manifest_rejects_disabled_workflow(
    tmp_path: Path, monkeypatch
):
    module = load_inference_worker_module()
    create_class_based_json_plugin(tmp_path, plugin_id="enabled-plugin")
    create_class_based_json_plugin(tmp_path, plugin_id="disabled-plugin")
    monkeypatch.setenv("PLUGIN_DIR", str(tmp_path))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "enabled-plugin")

    with pytest.raises(ValueError, match="disabled-plugin"):
        module._resolve_plugin_manifest("disabled-plugin")


def test_plugin_runtime_serialize_prepare_result_preserves_raw_operation_dict():
    script_dir = repo_root() / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from plugin_runtime import serialize_prepare_result  # type: ignore

    result = serialize_prepare_result(
        {
            "operation": "preprocess",
            "parameters": {"value": 3},
            "resource_profile": {"gpus_required": 1, "memory_mb": 4096},
        }
    )

    assert result == {
        "operation": "preprocess",
        "parameters": {"value": 3},
        "resource_profile": {"gpus_required": 1, "memory_mb": 4096},
    }


def test_plugin_dev_merge_prepare_output_preserves_next_stage_id():
    module = load_plugin_dev_module()
    payload = {
        "operation": "materialize_perturbations",
        "parameters": {"value": 3},
    }

    module._merge_prepare_output(
        payload,
        {
            "operation": "run",
            "parameters": {"value": 6},
            "fanout_profile": {"item_count": 1, "max_in_flight": 1},
            "fanout_items": [{"item_index": 0, "parameters": {"value": 6}}],
            "next_stage_id": "fanout",
        },
    )

    assert payload == {
        "operation": "run",
        "parameters": {"value": 6},
        "fanout_profile": {"item_count": 1, "max_in_flight": 1},
        "fanout_items": [{"item_index": 0, "parameters": {"value": 6}}],
        "next_stage_id": "fanout",
    }


def test_plugin_authoring_guide_exists():
    guide = repo_root() / "docs" / "plugin-authoring-guide.md"
    assert guide.is_file()

    content = guide.read_text(encoding="utf-8")
    assert "python scripts/plugin_dev.py init" in content
    assert "python scripts/plugin_dev.py check" in content
    assert "python scripts/plugin_dev.py check-env" in content
    assert "python scripts/plugin_dev.py run-local" in content
    assert "run_batch(items, ctx)" in content
    assert "scheduler hint" in content


def test_plugin_dev_init_creates_minimal_json_scaffold(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-json-plugin"

    proc = subprocess.run(
        [sys.executable, str(script), "init", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["workflow_id"] == "my-json-plugin"
    assert (plugin_root / "plugin.yaml").is_file()
    assert (plugin_root / "workflow.py").is_file()
    assert (plugin_root / "examples").is_dir()
    assert (plugin_root / "README.md").is_file()
    assert not (plugin_root / "plugin.py").exists()
    assert not (plugin_root / "schemas").exists()
    assert not (plugin_root / "fixtures" / "expected_result.json").exists()
    assert not (plugin_root / "examples" / "default_request.json").exists()

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert "ingress" not in manifest
    assert manifest["pipeline"] == {"profile": "simple"}
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest
    assert "developer" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "@dataclass" in workflow_content
    assert "input_model = ScaffoldInput" in workflow_content
    assert "output_model = ScaffoldOutput" in workflow_content
    assert (
        "def run(self, inputs: ScaffoldInput, ctx) -> ScaffoldOutput:"
        in workflow_content
    )
    assert "WORKFLOW = ScaffoldWorkflow" in workflow_content
    assert "WORKFLOW = ScaffoldWorkflow()" not in workflow_content

    readme_content = (plugin_root / "README.md").read_text(encoding="utf-8")
    assert "workflow.py" in readme_content
    assert "generated from the input/output models" in readme_content
    assert "examples/default_request.json" in readme_content
    assert "optional for simple JSON plugins" in readme_content

    validate_proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert validate_proc.returncode == 0, validate_proc.stderr


def test_plugin_dev_init_creates_prefetch_json_scaffold_with_explicit_hooks(
    tmp_path: Path,
):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-prefetch-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "prefetch",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "prefetch"}
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "input_model = ScaffoldInput" in workflow_content
    assert "def prepare(self, request, ctx) -> PrepareResult:" in workflow_content
    assert "PrepareResult" in workflow_content
    assert (
        "def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:"
        in workflow_content
    )
    assert "ctx.outputs.create(" in workflow_content
    assert "prefetch_plan=[]" in workflow_content
    assert '"status": "succeeded"' not in workflow_content
    assert '"output_path": str(output_path)' not in workflow_content
    assert '"artifacts": [' not in workflow_content
    assert "def execute(self, ctx):" not in workflow_content
    assert "output_model = ScaffoldOutput" not in workflow_content
    assert "WORKFLOW = ScaffoldWorkflow" in workflow_content
    assert "WORKFLOW = ScaffoldWorkflow()" not in workflow_content

    run_proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert run_proc.returncode == 0, run_proc.stderr
    result = json.loads(run_proc.stdout)
    assert result["status"] == "succeeded"
    assert Path(result["output_path"]).is_file()
    assert result["artifacts"][0]["name"] == "primary"
    assert result["artifacts"][0]["storage_path"] == result["output_path"]


def test_plugin_dev_init_accepts_default_pipeline_alias(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-default-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "default",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "default"}
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "def prepare(self, request, ctx) -> PrepareResult:" in workflow_content
    assert "PrepareResult" in workflow_content
    assert (
        "def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:"
        in workflow_content
    )
    assert "ctx.outputs.create(" in workflow_content
    assert "prefetch_plan=[]" in workflow_content
    assert '"status": "succeeded"' not in workflow_content
    assert '"output_path": str(output_path)' not in workflow_content
    assert '"artifacts": [' not in workflow_content

    validate_proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert validate_proc.returncode == 0, validate_proc.stderr

    run_proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert run_proc.returncode == 0, run_proc.stderr
    result = json.loads(run_proc.stdout)
    assert result["status"] == "succeeded"
    assert Path(result["output_path"]).is_file()
    assert result["artifacts"][0]["name"] == "primary"
    assert result["artifacts"][0]["storage_path"] == result["output_path"]


def test_plugin_dev_init_creates_postprocess_scaffold_with_postprocess_hook(
    tmp_path: Path,
):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-postprocess-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "postprocess",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "postprocess"}
    assert manifest["ingress"]["json_schema_inline"] == {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {
            "value": {"type": "integer", "minimum": 1},
        },
    }
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "def prepare(self, request, ctx) -> PrepareResult:" in workflow_content
    assert "PrepareResult" in workflow_content
    assert (
        "def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:"
        in workflow_content
    )
    assert "ctx.outputs.create(" in workflow_content
    assert (
        "def postprocess(self, result, ctx) -> PostprocessOutcome[dict[str, object]]:"
        in workflow_content
    )
    assert "PostprocessOutcome" in workflow_content
    assert "output_model = ScaffoldOutput" in workflow_content
    assert "result.payload" in workflow_content

    module = load_plugin_dev_module()
    expanded_manifest, _, _ = module._load_plugin_contract(plugin_root)
    example_payload = module.build_example_payload(
        plugin_root,
        expanded_manifest,
        run_id="example-postprocess",
    )
    assert example_payload["request"]["raw_fields"] == {"value": 1}
    assert example_payload["parameters"] == {"value": 1}


def test_plugin_dev_init_creates_batch_scaffold_with_run_batch_hook(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-batch-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "batch",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "batch"}
    assert manifest["runtime"] == {"profile": "python-test"}

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "batch_profile={" in workflow_content
    assert (
        "# Batch execution hook. Process compatible items together." in workflow_content
    )
    assert (
        "from plugin_sdk import BatchExecutionContext, BatchItem, PluginWorkflow"
        in workflow_content
    )
    assert "def run_batch(" in workflow_content
    assert "items: list[BatchItem[ScaffoldInput]]" in workflow_content
    assert "ctx: BatchExecutionContext" in workflow_content
    assert "def run(self, inputs: ScaffoldInput, ctx)" not in workflow_content
    assert "item.context.outputs.create(" in workflow_content


def test_plugin_dev_init_creates_ensemble_scaffold_with_fanout_placeholders(
    tmp_path: Path,
):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-ensemble-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "ensemble",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "ensemble"}
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "fanout_profile={" in workflow_content
    assert "fanout_items=[" in workflow_content
    assert '"item_index": 0' in workflow_content
    assert '"item_index": inputs.item_index' in workflow_content


def test_plugin_dev_validate_accepts_minimal_inline_plugin_without_optional_files(
    tmp_path: Path,
):
    plugin_root = create_minimal_inline_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_plugin_dev_validate_accepts_model_driven_plugin_without_manifest_schemas(
    tmp_path: Path,
):
    plugin_root = create_model_driven_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_plugin_dev_validate_accepts_compact_profile_manifest(
    tmp_path: Path,
):
    plugin_root = create_compact_profile_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_plugin_dev_validate_generates_json_example_when_fixture_is_missing(
    tmp_path: Path,
):
    plugin_root = create_model_driven_json_plugin(tmp_path, with_example_request=False)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert not (plugin_root / "examples" / "default_request.json").exists()


def test_plugin_dev_run_example_runs_model_driven_plugin_end_to_end(tmp_path: Path):
    plugin_root = create_model_driven_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env = os.environ.copy()
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["value"] == 7
    assert data["doubled"] == 14
    assert data["output_path"] is None
    assert data["artifacts"] == []


def test_plugin_workflow_prepare_returns_prepare_result_for_input_model(tmp_path: Path):
    plugin_sdk = load_plugin_sdk_module()

    @dataclasses.dataclass
    class DemoInput:
        value: int

    class DemoWorkflow(plugin_sdk.PluginWorkflow):
        input_model = DemoInput

    workflow = DemoWorkflow()
    request = plugin_sdk.RawRequest(
        content_type="application/json",
        operation="run",
        raw_fields={"value": 7},
        input_artifacts=[],
    )
    ctx = plugin_sdk.PrepareContext(
        run_id="run-prepare",
        workflow_id="demo-workflow",
        run_dir=tmp_path / "run-prepare",
    )

    result = workflow.prepare(request, ctx)

    assert isinstance(result, plugin_sdk.PrepareResult)
    assert result.inputs == {"value": 7}


def test_plugin_workflow_execute_returns_payload_only_for_dict_output():
    plugin_sdk = load_plugin_sdk_module()

    class DemoWorkflow(plugin_sdk.PluginWorkflow):
        def run(self, inputs, ctx):
            return {
                "value": inputs["value"],
            }

    workflow = DemoWorkflow()
    result = workflow.execute({"run_id": "run-dict", "parameters": {"value": 7}})

    assert result == {"value": 7}
    assert "status" not in result
    assert "output_path" not in result
    assert "artifacts" not in result


def test_plugin_workflow_execute_returns_payload_only_for_output_model():
    plugin_sdk = load_plugin_sdk_module()

    @dataclasses.dataclass
    class DemoOutput:
        value: int
        doubled: int

    class DemoWorkflow(plugin_sdk.PluginWorkflow):
        output_model = DemoOutput

        def run(self, inputs, ctx):
            return DemoOutput(
                value=inputs["value"],
                doubled=inputs["value"] * 2,
            )

    workflow = DemoWorkflow()
    result = workflow.execute({"run_id": "run-model", "parameters": {"value": 11}})

    assert result == {"value": 11, "doubled": 22}
    assert "status" not in result
    assert "output_path" not in result
    assert "artifacts" not in result


def test_plugin_workflow_execute_falls_back_to_run_batch_for_single_item():
    plugin_sdk = load_plugin_sdk_module()
    batch_calls: list[dict[str, object]] = []

    class DemoWorkflow(plugin_sdk.PluginWorkflow):
        def run_batch(self, items, ctx):
            batch_calls.append(
                {
                    "batch_id": ctx.batch_id,
                    "batch_size": int(ctx.batch_info.get("batch_size") or 0),
                    "run_ids": [item.context.run_id for item in items],
                }
            )
            return [{"value": items[0].inputs["value"] * 2}]

    workflow = DemoWorkflow()
    result = workflow.execute(
        {"run_id": "run-batch-fallback", "parameters": {"value": 11}}
    )

    assert result == {"value": 22}
    assert batch_calls == [
        {
            "batch_id": "run-batch-fallback",
            "batch_size": 1,
            "run_ids": ["run-batch-fallback"],
        }
    ]


def test_output_registry_tracks_registered_outputs(tmp_path: Path):
    plugin_sdk = load_plugin_sdk_module()
    run_dir = tmp_path / "run-outputs"
    outputs = plugin_sdk.OutputRegistry(run_dir)

    primary_path = outputs.create(
        "forecast_dataset",
        filename="forecast.zarr",
        media_type="application/x-zarr",
        primary=True,
    )
    preview_path = run_dir / "preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("preview", encoding="utf-8")
    outputs.register(
        "preview_image",
        preview_path,
        media_type="image/png",
    )

    registered = outputs.registered_outputs()

    assert primary_path == run_dir / "forecast.zarr"
    assert registered[0].name == "forecast_dataset"
    assert registered[0].primary is True
    assert registered[0].path == str(primary_path)
    assert registered[1].name == "preview_image"
    assert registered[1].primary is False
    assert registered[1].path == str(preview_path)


def test_plugin_workflow_postprocess_wraps_prior_result_payload(tmp_path: Path):
    plugin_sdk = load_plugin_sdk_module()

    @dataclasses.dataclass
    class DemoPayload:
        value: int

    class DemoWorkflow(plugin_sdk.PluginWorkflow):
        pass

    workflow = DemoWorkflow()
    run_dir = tmp_path / "run-postprocess"
    prior_result = plugin_sdk.PriorResult(
        payload=DemoPayload(value=5),
        execution=plugin_sdk.ExecutionInfo(
            run_id="run-postprocess",
            status="succeeded",
            outputs=[],
        ),
    )
    ctx = plugin_sdk.PostprocessContext(
        run_id="run-postprocess",
        run_dir=run_dir,
        outputs=plugin_sdk.OutputRegistry(run_dir),
        request=plugin_sdk.RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields={"value": 5},
            input_artifacts=[],
        ),
    )

    outcome = workflow.postprocess(prior_result, ctx)

    assert isinstance(outcome, plugin_sdk.PostprocessOutcome)
    assert dataclasses.asdict(outcome.payload) == {"value": 5}
    assert outcome.status == "succeeded"
    assert outcome.result_ops == []


def test_plugin_dev_run_example_generates_request_when_fixture_is_missing(
    tmp_path: Path,
):
    plugin_root = create_model_driven_json_plugin(tmp_path, with_example_request=False)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env = os.environ.copy()
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["value"] == 1
    assert data["doubled"] == 2


def test_plugin_dev_run_example_uses_profile_defaults_for_compact_manifest(
    tmp_path: Path,
):
    plugin_root = create_compact_profile_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env = os.environ.copy()
    env["DEFAULT_OUTPUT_DIR"] = str(tmp_path / "outputs")

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        env=env,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["value"] == 7
    assert data["doubled"] == 14


def test_plugin_dev_run_local_plan_materializes_generated_example_request(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    plugin_root = create_model_driven_json_plugin(tmp_path, with_example_request=False)
    workspace = tmp_path / "run-local"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    fixture_path = Path(plan["example_request"]["fixture_path"])
    assert fixture_path.is_file()
    assert fixture_path.parent == workspace
    assert json.loads(fixture_path.read_text(encoding="utf-8")) == {"value": 1}


def test_plugin_dev_run_local_plan_adds_gpu_executor_from_device_probe(tmp_path: Path):
    module = load_plugin_dev_module()
    plugin_root = create_device_switching_json_plugin(tmp_path)
    workspace = tmp_path / "run-local-device-switch"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert runtime_config["python_runtime_envs"]["python.cpu.demo"]["launch"] == {
        "enabled": True,
        "device_kind": "cpu",
        "replicas": 1,
        "memory_mb": 2048,
        "tags": ["demo", "cpu"],
    }
    assert runtime_config["python_runtime_envs"]["python.gpu.demo"]["launch"] == {
        "enabled": True,
        "device_kind": "gpu",
        "workers_per_device": 1,
        "tags": ["demo", "gpu"],
    }

    launcher = next(
        process
        for process in plan["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert (
        launcher["env"]["PHYSICSNEMO_SERVE_EXECUTOR_CLASSES"]
        == "python.cpu.demo,python.gpu.demo"
    )
    assert plan["execute_registration_stream"] == "execute.python.cpu.demo"
    assert plan["execute_registration_streams"] == [
        "execute.python.cpu.demo",
        "execute.python.gpu.demo:gpu:local:demo-device-switch-run-local:0",
    ]


def test_plugin_dev_run_local_plan_uses_profile_defaults_for_compact_manifest(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    plugin_root = create_compact_profile_json_plugin(
        tmp_path, with_example_request=False
    )
    workspace = tmp_path / "run-local-compact"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    launcher = next(
        process
        for process in plan["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert launcher["argv"][0] == sys.executable
    assert launcher["argv"][1].endswith("runtime_env_launcher.py")
    assert launcher["env"]["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"] == str(
        plan["runtime_config_path"]
    )
    assert launcher["env"]["WORKER_SCRIPT"].endswith("inference_worker.py")


def test_plugin_dev_run_local_plan_rewrites_cpu_postprocess_pipeline_without_scheduler(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "cpu-postprocess-plugin"
    workspace = tmp_path / "run-local-cpu-postprocess"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "postprocess",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "scheduler" not in runtime_config["roles"]
    assert "schedule" not in runtime_config["streams"]
    assert "release" not in runtime_config["streams"]

    inference_server = next(
        process
        for process in plan["processes"]
        if process["name"] == "inference_server"
    )
    runtime_plugin_root = Path(inference_server["env"]["PLUGIN_DIR"]) / plugin_root.name
    runtime_manifest = yaml.safe_load(
        (runtime_plugin_root / "plugin.yaml").read_text(encoding="utf-8")
    )
    assert [stage["phase"] for stage in runtime_manifest["pipeline"]["stages"]] == [
        "prepare",
        "execute",
        "postprocess",
        "results",
    ]
    assert yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))[
        "pipeline"
    ] == {"profile": "postprocess"}


def test_plugin_dev_run_local_plan_does_not_delete_source_plugin_when_workspace_is_parent(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path
    plugin_root = workspace / "plugins" / "cpu-postprocess-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "postprocess",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    assert (plugin_root / "plugin.yaml").is_file()
    assert yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))[
        "pipeline"
    ] == {"profile": "postprocess"}

    inference_server = next(
        process
        for process in plan["processes"]
        if process["name"] == "inference_server"
    )
    runtime_plugin_root = Path(inference_server["env"]["PLUGIN_DIR"]) / plugin_root.name
    assert (
        runtime_plugin_root == workspace / ".run-local" / "plugins" / plugin_root.name
    )
    runtime_manifest = yaml.safe_load(
        (runtime_plugin_root / "plugin.yaml").read_text(encoding="utf-8")
    )
    assert [stage["phase"] for stage in runtime_manifest["pipeline"]["stages"]] == [
        "prepare",
        "execute",
        "postprocess",
        "results",
    ]


def test_plugin_dev_run_example_uses_examples_default_request_for_minimal_plugin(
    tmp_path: Path,
):
    plugin_root = create_minimal_inline_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "succeeded"
    assert data["doubled"] == 14


def test_plugin_dev_init_creates_multipart_scaffold(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "my-multipart-plugin"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--content-type",
            "multipart/form-data",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (plugin_root / "workflow.py").is_file()
    assert (plugin_root / "examples" / "default_request.multipart.json").is_file()
    assert (plugin_root / "examples" / "sample.txt").is_file()
    assert not (plugin_root / "plugin.py").exists()
    assert not (plugin_root / "schemas").exists()

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["ingress"]["content_type"] == "multipart/form-data"
    assert "content_types" not in manifest["ingress"]
    assert "default_content_type" not in manifest["ingress"]
    assert "operations" not in manifest["ingress"]
    assert "request_schema_inline" not in manifest["ingress"]
    assert "files" in manifest["ingress"]
    assert manifest["pipeline"] == {"profile": "simple"}
    assert manifest["runtime"] == {"profile": "python-test"}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "@dataclass" in workflow_content
    assert "form_model = ScaffoldForm" in workflow_content
    assert "input_model = ScaffoldInput" in workflow_content
    assert "def prepare(self, request, ctx) -> PrepareResult:" in workflow_content
    assert "PrepareResult" in workflow_content
    assert "ctx.outputs.create(" in workflow_content
    assert "WORKFLOW = ScaffoldWorkflow" in workflow_content
    assert "WORKFLOW = ScaffoldWorkflow()" not in workflow_content


def test_plugin_dev_multipart_plugin_can_derive_form_schema_from_form_model(
    tmp_path: Path,
):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "multipart-derived-schema"

    write_file(
        plugin_root / "plugin.yaml",
        """
metadata:
  id: multipart-derived-schema
  display_name: Multipart Derived Schema
  version: 1.0.0
  description: Multipart plugin using workflow form_model
ingress:
  content_type: multipart/form-data
  files:
    - name: sample_file
      required: true
      media_types:
        - text/plain
        - application/octet-stream
      max_size_mb: 1
pipeline:
  profile: simple
runtime:
  profile: python-test
""".strip(),
    )
    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow, PrepareResult


@dataclass
class UploadForm:
    note: str
    count: int = 1


@dataclass
class UploadInput:
    note: str
    count: int
    sample_path: str


class MultipartWorkflow(PluginWorkflow):
    form_model = UploadForm
    input_model = UploadInput

    def prepare(self, request, ctx):
        artifact = request.input_artifacts[0]
        return PrepareResult(
            inputs={
                "note": request.raw_fields["note"],
                "count": int(request.raw_fields["count"]),
                "sample_path": artifact.storage_path,
            }
        )

    def run(self, inputs: UploadInput, ctx):
        output_path = ctx.outputs.create(
            "primary",
            filename="result.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {
            "note": inputs.note,
            "count": inputs.count,
            "sample_path": inputs.sample_path,
        }


WORKFLOW = MultipartWorkflow()
""".strip(),
    )
    write_file(
        plugin_root / "examples" / "sample.txt",
        "hello\n",
    )
    write_file(
        plugin_root / "examples" / "default_request.multipart.json",
        json.dumps(
            {
                "form_fields": {
                    "note": "demo",
                    "count": 2,
                },
                "files": {
                    "sample_file": "examples/sample.txt",
                },
            },
            indent=2,
        )
        + "\n",
    )
    write_file(plugin_root / "README.md", "# Multipart Derived Schema\n")

    validate_proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert validate_proc.returncode == 0, validate_proc.stderr

    run_proc = subprocess.run(
        [sys.executable, str(script), "run-example", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert run_proc.returncode == 0, run_proc.stderr
    result = json.loads(run_proc.stdout)
    assert result["status"] == "succeeded"
    assert result["note"] == "demo"
    assert result["count"] == 2


def test_plugin_dev_init_accepts_custom_runtime_scaffold(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "biology-demo"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "postprocess",
            "--runtime",
            "custom",
            "--executor-class",
            "python.gpu.biology",
            "--phase-executor",
            "prepare=python.cpu.biology",
            "--phase-executor",
            "postprocess=python.cpu.biology",
            "--phase-executor",
            "readiness=python.cpu.biology",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["workflow_id"] == "biology-demo"
    assert data["pipeline_profile"] == "postprocess"
    assert data["runtime_profile"] == "custom"

    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "postprocess"}
    assert manifest["runtime"] == {
        "executor_class": "python.gpu.biology",
        "prepare_executor_class": "python.cpu.biology",
        "postprocess_executor_class": "python.cpu.biology",
        "readiness_executor_class": "python.cpu.biology",
    }
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "gpu",
            "gpus_required": 1,
            "memory_mb": 16384,
            "cpu_cores": 4,
        }
    }
    assert "outputs" not in manifest

    workflow_content = (plugin_root / "workflow.py").read_text(encoding="utf-8")
    assert "def prepare(self, request, ctx) -> PrepareResult:" in workflow_content
    assert (
        "def run(self, inputs: ScaffoldInput, ctx) -> dict[str, object]:"
        in workflow_content
    )
    assert (
        "def postprocess(self, result, ctx) -> PostprocessOutcome[dict[str, object]]:"
        in workflow_content
    )
    assert "def execute(self, ctx):" not in workflow_content


def test_plugin_dev_run_example_builds_legacy_metadata_for_new_contract_plugin(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_new_contract_json_plugin(tmp_path, plugin_id="dev-example-new")
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))

    result = module.run_example_plugin(plugin_root, run_id="example-new-contract")

    expected_output_path = tmp_path / "outputs" / "example-new-contract" / "demo.json"
    assert result["status"] == "succeeded"
    assert result["value"] == 7
    assert result["doubled"] == 14
    assert result["output_path"] == str(expected_output_path)
    assert result["artifacts"] == [
        {
            "name": "demo-output",
            "media_type": "application/json",
            "storage_path": str(expected_output_path),
            "primary": True,
        }
    ]


def test_plugin_dev_run_local_plan_supports_new_contract_prepare_hook(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_new_contract_json_plugin(
        tmp_path, plugin_id="run-local-new-contract"
    )
    monkeypatch.setenv("DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    workspace = tmp_path / "run-local-new-contract"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    launcher = next(
        process
        for process in plan["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert launcher["env"]["PHYSICSNEMO_SERVE_EXECUTOR_CLASSES"] == "python.test"


def test_plugin_dev_init_defaults_executor_class_for_custom_runtime(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "biology-default-executor"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--runtime",
            "custom",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = yaml.safe_load((plugin_root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == {"profile": "simple"}
    assert manifest["runtime"] == {}
    assert manifest["resources"] == {
        "defaults": {
            "device_kind": "cpu",
            "gpus_required": 0,
            "memory_mb": 1024,
            "cpu_cores": 1,
        }
    }
    assert "outputs" not in manifest

    validate_proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert validate_proc.returncode == 0, validate_proc.stderr


def test_plugin_dev_validate_accepts_example_fixture_names(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_plugin_dev_validate_generates_json_example_when_named_fixture_is_missing(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="missing-example-fixtures"
    )
    script = repo_root() / "scripts" / "plugin_dev.py"

    (plugin_root / "fixtures" / "example_request.json").unlink()
    (plugin_root / "fixtures" / "expected_result.json").unlink()
    (plugin_root / "fixtures" / "ignored_request.json").write_text(
        json.dumps({"value": 7}),
        encoding="utf-8",
    )
    (plugin_root / "fixtures" / "ignored_result.json").write_text(
        json.dumps({"status": "succeeded"}),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_plugin_dev_check_runs_validation_environment_and_example(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "check", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "ready"
    assert data["validation"]["status"] == "valid"
    assert data["environment"]["status"] == "ready"
    assert data["example_run"]["status"] == "succeeded"
    assert data["example_run"]["doubled"] == 14


def test_plugin_dev_check_falls_back_to_recommended_phase_when_not_ready(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="needs-setup")
    script = repo_root() / "scripts" / "plugin_dev.py"
    update_manifest(
        plugin_root,
        lambda manifest: manifest.update(
            {
                "developer": {
                    "readiness": {
                        "recommended_check_phase": "prepare",
                        "env": [{"name": "MISSING_PLUGIN_ROOT", "kind": "dir"}],
                        "python_modules": [],
                        "paths": [],
                    }
                }
            }
        ),
    )

    proc = subprocess.run(
        [sys.executable, str(script), "check", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "needs_setup"
    assert data["environment"]["status"] == "not_ready"
    assert data["example_run"]["parameters"]["doubled"] == 14
    assert data["example_phase"] == "prepare"


def test_plugin_dev_check_env_reports_readiness(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "check-env", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "ready"


def test_plugin_dev_rejects_unknown_command(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [sys.executable, str(script), "unknown-command", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr.lower()


def test_plugin_dev_run_local_dry_run_writes_example_submit_script(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="run-local-demo")
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path / "run-local-workspace"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--workspace",
            str(workspace),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert Path(data["submit_example_script"]).is_file()
    assert data["example_request"]["content_type"] == "application/json"
    assert (workspace / "submit_example.sh").is_file()


def test_plugin_dev_init_rejects_existing_directory_without_force(tmp_path: Path):
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "existing-plugin"
    plugin_root.mkdir()

    proc = subprocess.run(
        [sys.executable, str(script), "init", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_plugin_dev_validate_rejects_invalid_example_request_fixture(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    write_file(plugin_root / "examples" / "default_request.json", '{"value": 0}')

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode != 0
    assert "example request" in proc.stderr.lower()


def test_plugin_dev_validate_rejects_invalid_expected_result_fixture(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    write_file(
        plugin_root / "fixtures" / "expected_result.json",
        """
{
  "status": "succeeded",
  "artifacts": []
}
""".strip(),
    )

    proc = subprocess.run(
        [sys.executable, str(script), "validate", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode != 0
    assert "expected result fixture" in proc.stderr.lower()


def test_plugin_dev_check_env_reports_missing_requirements(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    update_manifest(
        plugin_root,
        lambda manifest: manifest.update(
            {
                "developer": {
                    "readiness": {
                        "env": [
                            {"name": "DEMO_PLUGIN_ROOT", "kind": "dir"},
                            {
                                "any_of": [
                                    "DEMO_PLUGIN_CHECKPOINT",
                                    "CUSTOM_CHECKPOINT_PATH",
                                ],
                                "kind": "file",
                            },
                        ],
                        "python_modules": ["json", "definitely_missing_plugin_module"],
                    }
                }
            }
        ),
    )

    proc = subprocess.run(
        [sys.executable, str(script), "check-env", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 1, proc.stdout
    data = json.loads(proc.stdout)
    assert data["status"] == "not_ready"
    failed = [check for check in data["checks"] if not check["ok"]]
    assert any(check["type"] == "env" for check in failed)
    assert any(check["type"] == "python_module" for check in failed)


def test_plugin_dev_check_env_reports_broken_python_module_imports(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    write_file(
        tmp_path / "brokenpkg" / "__init__.py",
        "import definitely_missing_dependency\n",
    )
    update_manifest(
        plugin_root,
        lambda manifest: manifest.update(
            {
                "developer": {
                    "readiness": {
                        "python_modules": ["brokenpkg"],
                    }
                }
            }
        ),
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(tmp_path)
        if not existing_pythonpath
        else os.pathsep.join([str(tmp_path), existing_pythonpath])
    )

    proc = subprocess.run(
        [sys.executable, str(script), "check-env", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        env=env,
        check=False,
    )

    assert proc.returncode == 1, proc.stdout
    data = json.loads(proc.stdout)
    assert data["status"] == "not_ready"
    broken_check = next(
        check for check in data["checks"] if check["name"] == "brokenpkg"
    )
    assert broken_check["type"] == "python_module"
    assert broken_check["ok"] is False
    assert "could not be imported" in broken_check["detail"]
    assert "definitely_missing_dependency" in broken_check["detail"]


def test_plugin_dev_check_env_passes_when_requirements_are_met(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    env_root = tmp_path / "runtime"
    env_root.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    update_manifest(
        plugin_root,
        lambda manifest: manifest.update(
            {
                "developer": {
                    "readiness": {
                        "recommended_check_phase": "prepare",
                        "env": [
                            {"name": "DEMO_PLUGIN_ROOT", "kind": "dir"},
                            {"any_of": ["DEMO_PLUGIN_CHECKPOINT"], "kind": "file"},
                        ],
                        "python_modules": ["json", "pathlib"],
                    }
                }
            }
        ),
    )

    env = os.environ.copy()
    env["DEMO_PLUGIN_ROOT"] = str(env_root)
    env["DEMO_PLUGIN_CHECKPOINT"] = str(checkpoint)

    proc = subprocess.run(
        [sys.executable, str(script), "check-env", str(plugin_root)],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "ready"
    assert data["recommended_check_phase"] == "prepare"
    assert all(check["ok"] for check in data["checks"])


def test_plugin_dev_run_local_dry_run_builds_workspace_and_launch_plan(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path)
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path / "run-local-workspace"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
            "--workspace",
            str(workspace),
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "planned"
    assert data["workflow_id"] == plugin_root.name
    assert Path(data["workspace"]).is_dir()
    assert Path(data["runtime_config_path"]).is_file()
    assert Path(data["submit_example_script"]).is_file()
    assert data["server_url"].startswith("http://127.0.0.1:")

    runtime_config = json.loads(
        Path(data["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert set(runtime_config["roles"]) == {"prepare", "results"}
    assert set(runtime_config["streams"]) == {"prepare", "results"}
    assert (
        runtime_config["python_runtime_envs"]["python.test"]["python_executable"]
        == sys.executable
    )
    assert runtime_config["python_runtime_envs"]["python.test"]["launch"] == {
        "enabled": True,
        "device_kind": "cpu",
        "memory_mb": 1024,
        "replicas": 1,
        "tags": [],
    }

    process_names = [process["name"] for process in data["processes"]]
    assert process_names == [
        "redis",
        "inference_server",
        "prepare",
        "results",
        "runtime_env_launcher",
    ]
    assert "curl" in data["example_request"]["command"]


def test_plugin_dev_run_local_dry_run_sets_enabled_plugin_id(tmp_path: Path):
    module = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="run-local-demo")
    workspace = tmp_path / "run-local-workspace"

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    inference_server = next(
        process
        for process in plan["processes"]
        if process["name"] == "inference_server"
    )

    assert (
        inference_server["env"]["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"]
        == "run-local-demo"
    )


def test_plugin_dev_run_local_dry_run_includes_publish_when_output_publication_configured(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="run-local-publish-demo"
    )
    workspace = tmp_path / "run-local-publish-workspace"
    publication_config = tmp_path / "output_publication.json"
    publication_config.write_text(
        json.dumps(
            {
                "output_publication": {
                    "enabled": True,
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "outputs",
                    },
                },
                "roles": {
                    "publish": {
                        "config": {
                            "max_concurrent_files": 12,
                            "multipart_max_concurrency": 3,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG", str(publication_config))

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "publish" in runtime_config["streams"]
    assert runtime_config["roles"]["publish"]["inputs"][0]["stream"] == "publish"
    assert runtime_config["roles"]["publish"]["config"] == {
        "max_concurrent_files": 12,
        "multipart_max_concurrency": 3,
    }
    assert runtime_config["output_publication"]["enabled"] is True
    assert runtime_config["output_publication"]["storage"]["bucket"] == "bucket"
    process_names = [process["name"] for process in plan["processes"]]
    assert "publish" in process_names

    inference_server = next(
        process
        for process in plan["processes"]
        if process["name"] == "inference_server"
    )
    assert inference_server["env"]["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"] == str(
        plan["runtime_config_path"]
    )


def test_plugin_dev_run_local_includes_publish_for_output_publication_json_override(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="run-local-publish-json-demo"
    )
    workspace = tmp_path / "run-local-publish-json-workspace"
    monkeypatch.delenv("PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG", raising=False)
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON",
        json.dumps(
            {
                "enabled": True,
                "storage": {
                    "type": "s3",
                    "bucket": "bucket",
                    "prefix": "outputs",
                },
            }
        ),
    )

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "publish" in runtime_config["streams"]
    assert runtime_config["roles"]["publish"]["inputs"][0]["stream"] == "publish"
    assert runtime_config["output_publication"]["enabled"] is True
    assert runtime_config["output_publication"]["storage"]["bucket"] == "bucket"
    process_names = [process["name"] for process in plan["processes"]]
    assert "publish" in process_names


def test_plugin_dev_run_local_uses_manifest_publish_queue_for_output_publication(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="run-local-custom-publish-demo"
    )
    workspace = tmp_path / "run-local-custom-publish-workspace"
    publication_config = tmp_path / "output_publication.json"
    publication_config.write_text(
        json.dumps(
            {
                "output_publication": {
                    "enabled": True,
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "outputs",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG", str(publication_config))

    def add_custom_publish_stage(manifest: dict) -> None:
        stages = manifest["pipeline"]["stages"]
        stages[1]["next"] = "publish"
        stages.insert(
            2,
            {
                "id": "publish",
                "phase": "publish",
                "handler": "plugin_phase",
                "queue": "publish.custom",
                "next": "results",
            },
        )

    update_manifest(plugin_root, add_custom_publish_stage)

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "publish.custom" in runtime_config["streams"]
    assert "publish" not in runtime_config["streams"]
    assert runtime_config["roles"]["publish"]["inputs"][0]["stream"] == "publish.custom"
    process_names = [process["name"] for process in plan["processes"]]
    assert "publish" in process_names


def test_plugin_dev_run_local_synthesizes_results_after_manifest_publish(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="run-local-publish-no-results-demo"
    )
    workspace = tmp_path / "run-local-publish-no-results-workspace"
    monkeypatch.delenv("PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG", raising=False)
    monkeypatch.setenv(
        "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON",
        json.dumps(
            {
                "enabled": True,
                "storage": {
                    "type": "s3",
                    "bucket": "bucket",
                    "prefix": "outputs",
                },
            }
        ),
    )

    def end_pipeline_with_publish(manifest: dict) -> None:
        stages = manifest["pipeline"]["stages"]
        stages.pop()
        stages[-1]["next"] = "publish"
        stages.append(
            {
                "id": "publish",
                "phase": "publish",
                "handler": "publish_outputs",
                "queue": "publish",
                "next": None,
            }
        )

    update_manifest(plugin_root, end_pipeline_with_publish)

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    runtime_config = json.loads(
        Path(plan["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert runtime_config["streams"][-2:] == ["publish", "results"]
    assert list(runtime_config["roles"])[-2:] == ["publish", "results"]
    process_names = [process["name"] for process in plan["processes"]]
    assert process_names.count("publish") == 1
    assert process_names.count("results") == 1
    assert process_names.index("publish") < process_names.index("results")


def test_plugin_dev_run_local_dry_run_includes_scheduler_when_pipeline_declares_schedule(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="demo-scheduled")
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path / "run-local-scheduled"

    def add_schedule_stage(manifest: dict) -> None:
        manifest["pipeline"]["stages"] = [
            {
                "id": "prepare",
                "phase": "prepare",
                "handler": "plugin_phase",
                "queue": "prepare",
                "next": "schedule",
            },
            {
                "id": "schedule",
                "phase": "schedule",
                "handler": "schedule",
                "queue": "schedule",
                "next": "execute",
            },
            {
                "id": "execute",
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": "results",
            },
            {
                "id": "results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]

    update_manifest(plugin_root, add_schedule_stage)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
            "--workspace",
            str(workspace),
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    runtime_config = json.loads(
        Path(data["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "scheduler" in runtime_config["roles"]
    assert runtime_config["roles"]["scheduler"]["inputs"][0]["stream"] == "schedule"
    assert runtime_config["roles"]["scheduler"]["inputs"][1]["stream"] == "release"
    assert (
        runtime_config["roles"]["scheduler"]["config"]["memory_utilization_percent"]
        == 100
    )
    assert (
        runtime_config["roles"]["scheduler"]["config"]["gpu_discovery_interval_secs"]
        == 1
    )
    assert "release" in runtime_config["streams"]
    assert (
        runtime_config["python_runtime_envs"]["python.test"]["python_executable"]
        == sys.executable
    )
    assert (
        runtime_config["python_runtime_envs"]["python.test"]["launch"]["device_kind"]
        == "cpu"
    )
    assert (
        runtime_config["python_runtime_envs"]["python.test"]["launch"]["replicas"] == 1
    )

    launcher = next(
        process
        for process in data["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert launcher["argv"][0] == sys.executable
    assert launcher["argv"][1].endswith("runtime_env_launcher.py")
    assert (
        launcher["env"]["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"]
        == data["runtime_config_path"]
    )


def test_plugin_dev_run_local_dry_run_includes_batch_role_when_pipeline_declares_batch(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="demo-batched")
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path / "run-local-batched"

    def add_batch_pipeline(manifest: dict) -> None:
        manifest["pipeline"]["stages"] = [
            {
                "id": "prepare",
                "phase": "prepare",
                "handler": "plugin_phase",
                "queue": "prepare",
                "next": "batch",
            },
            {
                "id": "batch",
                "phase": "batch",
                "handler": "batch",
                "queue": "batch",
                "next": "schedule",
            },
            {
                "id": "schedule",
                "phase": "schedule",
                "handler": "schedule",
                "queue": "schedule",
                "next": "execute",
            },
            {
                "id": "execute",
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": "results",
            },
            {
                "id": "results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]

    update_manifest(plugin_root, add_batch_pipeline)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
            "--workspace",
            str(workspace),
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    runtime_config = json.loads(
        Path(data["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "batch" in runtime_config["roles"]
    assert runtime_config["roles"]["batch"]["inputs"][0]["stream"] == "batch"
    assert "batch" in runtime_config["streams"]

    process_names = [process["name"] for process in data["processes"]]
    assert "batch" in process_names


def test_plugin_dev_run_local_dry_run_includes_fanout_and_collect_for_ensemble_pipeline(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="demo-ensemble")
    script = repo_root() / "scripts" / "plugin_dev.py"
    workspace = tmp_path / "run-local-ensemble"

    def add_ensemble_pipeline(manifest: dict) -> None:
        manifest["pipeline"]["stages"] = [
            {
                "id": "prepare",
                "phase": "prepare",
                "handler": "plugin_phase",
                "queue": "prepare",
                "next": "fanout",
            },
            {
                "id": "fanout",
                "phase": "fanout",
                "handler": "fanout",
                "queue": "fanout",
                "next": "schedule",
            },
            {
                "id": "schedule",
                "phase": "schedule",
                "handler": "schedule",
                "queue": "schedule",
                "next": "execute",
            },
            {
                "id": "execute",
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": "collect",
            },
            {
                "id": "collect",
                "phase": "collect",
                "handler": "collect",
                "queue": "collect",
                "next": "results",
            },
            {
                "id": "results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]

    update_manifest(plugin_root, add_ensemble_pipeline)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
            "--workspace",
            str(workspace),
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    runtime_config = json.loads(
        Path(data["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert "fanout" in runtime_config["roles"]
    assert "collect" in runtime_config["roles"]
    assert runtime_config["roles"]["fanout"]["inputs"][0]["stream"] == "fanout"
    assert runtime_config["roles"]["collect"]["inputs"][0]["stream"] == "collect"
    assert "fanout" in runtime_config["streams"]
    assert "collect" in runtime_config["streams"]

    process_names = [process["name"] for process in data["processes"]]
    assert "fanout" in process_names
    assert "collect" in process_names
    assert "scheduler" in process_names


def test_plugin_dev_run_local_uses_prepare_resource_profile_for_runtime_env_launch(
    tmp_path: Path,
):
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="demo-capabilities"
    )
    script = repo_root() / "scripts" / "plugin_dev.py"

    write_file(
        plugin_root / "workflow.py",
        """
from __future__ import annotations

from plugin_sdk import PluginWorkflow


class DemoCapabilitiesWorkflow(PluginWorkflow):
    def prepare(self, ctx):
        return {
            "parameters": {"value": int(ctx["parameters"]["value"])},
            "resource_profile": {
                "executor_class": "python.gpu.custom",
                "device_kind": "gpu",
                "gpus_required": 1,
                "memory_mb": 8192,
                "cpu_cores": 2,
                "tags": ["alpha", "beta"],
            },
        }

    def execute(self, ctx):
        return {
            "status": "succeeded",
            "output_path": "/tmp/demo.json",
            "artifacts": [],
        }


WORKFLOW = DemoCapabilitiesWorkflow()
""".strip(),
    )

    def add_schedule_stage(manifest: dict) -> None:
        manifest["pipeline"]["stages"] = [
            {
                "id": "prepare",
                "phase": "prepare",
                "handler": "plugin_phase",
                "queue": "prepare",
                "next": "schedule",
            },
            {
                "id": "schedule",
                "phase": "schedule",
                "handler": "schedule",
                "queue": "schedule",
                "next": "execute",
            },
            {
                "id": "execute",
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": "results",
            },
            {
                "id": "results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]

    update_manifest(plugin_root, add_schedule_stage)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    runtime_config = json.loads(
        Path(data["runtime_config_path"]).read_text(encoding="utf-8")
    )
    assert (
        runtime_config["python_runtime_envs"]["python.gpu.custom"]["python_executable"]
        == sys.executable
    )
    assert runtime_config["python_runtime_envs"]["python.gpu.custom"]["launch"] == {
        "enabled": True,
        "device_kind": "gpu",
        "workers_per_device": 1,
        "tags": ["alpha", "beta"],
    }
    launcher = next(
        process
        for process in data["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert (
        launcher["env"]["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"]
        == data["runtime_config_path"]
    )
    assert launcher["env"]["WORKER_SCRIPT"].endswith("inference_worker.py")


def test_plugin_dev_run_local_sets_explicit_python_interpreter(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(tmp_path, plugin_id="demo-python-env")
    script = repo_root() / "scripts" / "plugin_dev.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            str(plugin_root),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    server = next(
        process
        for process in data["processes"]
        if process["name"] == "inference_server"
    )
    launcher = next(
        process
        for process in data["processes"]
        if process["name"] == "runtime_env_launcher"
    )
    assert server["env"]["PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE"] == sys.executable
    assert (
        server["env"]["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"]
        == data["runtime_config_path"]
    )
    assert launcher["env"]["PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE"] == sys.executable


def test_plugin_dev_run_local_rejects_not_ready_plugin(tmp_path: Path):
    plugin_root = create_class_based_json_plugin(
        tmp_path, plugin_id="not-ready-run-local"
    )
    script = repo_root() / "scripts" / "plugin_dev.py"

    def add_missing_env_check(manifest: dict) -> None:
        manifest.setdefault("developer", {}).setdefault("readiness", {})["env"] = [
            {"name": "DEMO_REQUIRED_PATH", "kind": "file", "required": True}
        ]

    update_manifest(plugin_root, add_missing_env_check)

    proc = subprocess.run(
        [sys.executable, str(script), "run-local", str(plugin_root), "--dry-run"],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )

    assert proc.returncode != 0
    assert "not ready" in proc.stderr.lower()
    assert "plugin_dev.py check-env" in proc.stderr


def test_plugin_dev_run_local_stack_waits_for_server_port_and_worker_registration(
    monkeypatch,
):
    module = load_plugin_dev_module()
    events: list[tuple] = []

    class DummyProc:
        def __init__(self, name: str):
            self.name = name

        def poll(self):
            return None

        def terminate(self):
            events.append(("terminate", self.name))

        def wait(self, timeout=None):
            events.append(("wait", self.name, timeout))
            return 0

        def kill(self):
            events.append(("kill", self.name))

    def fake_popen(argv, cwd, env, text):
        return DummyProc(Path(argv[0]).name)

    def fake_wait_for_tcp_port(host, port, *, timeout_secs):
        events.append(("tcp", host, port, timeout_secs))

    def fake_wait_for_worker_registration(redis_url, stream_name, *, timeout_secs):
        events.append(("worker", redis_url, stream_name, timeout_secs))

    sleep_calls: list[float] = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if seconds == 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "_wait_for_tcp_port", fake_wait_for_tcp_port)
    monkeypatch.setattr(
        module, "_wait_for_worker_registration", fake_wait_for_worker_registration
    )
    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        module,
        "_terminate_processes",
        lambda processes, *, suppress_interrupts=False: events.append(
            ("terminate_all", len(processes), suppress_interrupts)
        ),
    )

    plan = {
        "processes": [
            {"name": "redis", "argv": ["redis-server"], "env": {}},
            {"name": "inference_server", "argv": ["inference_server"], "env": {}},
            {
                "name": "runtime_env_launcher",
                "argv": ["runtime_env_launcher"],
                "env": {},
            },
        ],
        "redis_port": 16379,
        "port": 18080,
        "redis_url": "redis://127.0.0.1:16379/0",
        "server_url": "http://127.0.0.1:18080",
        "workflow_id": "demo-json",
        "workspace": "/tmp/demo-json",
        "execute_registration_stream": "execute.python.test",
        "execute_registration_streams": [
            "execute.python.test",
            "execute.python.gpu.demo:gpu:local:demo-json:0",
        ],
        "example_request": {
            "command": "curl http://127.0.0.1:18080/v1/infer/demo-json/run"
        },
    }

    module.run_local_stack(plan)

    assert not hasattr(module, "_wait_for_http_ready")
    assert ("tcp", "127.0.0.1", 16379, 10) in events
    assert ("tcp", "127.0.0.1", 18080, 30) in events
    assert ("worker", "redis://127.0.0.1:16379/0", "execute.python.test", 10) in events
    assert (
        "worker",
        "redis://127.0.0.1:16379/0",
        "execute.python.gpu.demo:gpu:local:demo-json:0",
        10,
    ) in events
    assert ("terminate_all", 3, True) in events
    assert 1.2 in sleep_calls


def test_plugin_dev_terminate_processes_suppresses_interrupts_during_cleanup():
    module = load_plugin_dev_module()
    events: list[tuple] = []

    class DummyProc:
        def __init__(self, name: str):
            self.name = name
            self.killed = False

        def poll(self):
            return 0 if self.killed else None

        def terminate(self):
            events.append(("terminate", self.name))

        def wait(self, timeout=None):
            events.append(("wait", self.name, timeout))
            raise KeyboardInterrupt()

        def kill(self):
            self.killed = True
            events.append(("kill", self.name))

    processes = [("runtime_env_launcher", DummyProc("runtime_env_launcher"))]

    module._terminate_processes(processes, suppress_interrupts=True)

    assert ("terminate", "runtime_env_launcher") in events
    assert any(event[:2] == ("wait", "runtime_env_launcher") for event in events)
    assert ("kill", "runtime_env_launcher") in events


def test_plugin_dev_run_local_postprocess_stack_serves_structured_results(
    tmp_path: Path,
):
    module = load_plugin_dev_module()
    script = repo_root() / "scripts" / "plugin_dev.py"
    plugin_root = tmp_path / "local-e2e-postprocess-plugin"
    workspace = tmp_path / "run-local-e2e"

    init_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            str(plugin_root),
            "--pipeline",
            "postprocess",
        ],
        text=True,
        capture_output=True,
        cwd=repo_root(),
        check=False,
    )
    assert init_proc.returncode == 0, init_proc.stderr
    # This e2e checks postprocess/result serving; avoid scheduler admission here.
    update_manifest(
        plugin_root,
        lambda manifest: manifest["pipeline"].update(
            {"profile": "simple", "options": {"postprocess": True}}
        ),
    )

    try:
        module.ensure_run_local_prerequisites(skip_build=False)
    except ValueError as exc:
        if "redis-server" in str(exc):
            pytest.skip(str(exc))
        bootstrap = module.bootstrap_python(dry_run=False)
        assert bootstrap["status"] in {"ready", "bootstrapped"}
        module.ensure_run_local_prerequisites(skip_build=False)
    module.build_run_local_binaries()

    plan = module.build_run_local_plan(
        plugin_root,
        workspace=workspace,
        port=0,
        redis_port=0,
    )

    def request_json(url: str, *, method: str = "GET", payload: dict | None = None):
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
                response_headers,
            )

    def request_bytes(url: str):
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
            return response.status, response.read(), response_headers

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    log_handles = []
    try:
        for process in plan["processes"]:
            log_path = Path(plan["workspace"]) / "logs" / f"{process['name']}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)
            env = os.environ.copy()
            env.update(process.get("env", {}))
            proc = subprocess.Popen(
                process["argv"],
                cwd=repo_root(),
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process["name"], proc))
            if process["name"] == "redis":
                module._wait_for_tcp_port(
                    "127.0.0.1", int(plan["redis_port"]), timeout_secs=10
                )
            elif process["name"] == "inference_server":
                module._wait_for_tcp_port(
                    "127.0.0.1", int(plan["port"]), timeout_secs=30
                )

        execute_stream = plan.get("execute_registration_stream")
        if execute_stream:
            module._wait_for_worker_registration(
                plan["redis_url"], execute_stream, timeout_secs=10
            )
        time.sleep(1.2)

        fixture_path = Path(plan["example_request"]["fixture_path"])
        request_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        run_status, run_response, _ = request_json(
            f"{plan['server_url']}/v1/infer/{plan['workflow_id']}/run",
            method="POST",
            payload=request_payload,
        )
        assert run_status == 202
        run_id = run_response["run_id"]

        deadline = time.time() + 30
        final_status = None
        while time.time() < deadline:
            status_code, status_payload, _ = request_json(
                f"{plan['server_url']}/v1/infer/{plan['workflow_id']}/{run_id}/status"
            )
            assert status_code == 200
            final_status = status_payload
            stage = str(status_payload.get("stage") or "")
            status = str(status_payload.get("status") or "")
            if stage in {"completed", "failed"} or status in {
                "succeeded",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(0.5)
        else:
            raise AssertionError(
                f"Timed out waiting for run completion: {final_status}"
            )

        assert final_status is not None
        assert final_status.get("status") == "succeeded"

        result_status, result_payload, _ = request_json(
            f"{plan['server_url']}/v1/infer/{plan['workflow_id']}/{run_id}/results"
        )
        assert result_status == 200
        assert set(result_payload.keys()) == {"request", "execution", "payload"}
        assert (
            result_payload["request"]["raw_fields"]["value"] == request_payload["value"]
        )
        assert result_payload["payload"]["value"] == request_payload["value"]
        assert result_payload["payload"]["doubled"] == request_payload["value"] * 2
        assert result_payload["payload"]["postprocessed"] is True
        assert "run_id" not in result_payload["payload"]
        assert "status" not in result_payload["payload"]
        assert result_payload["execution"]["outputs"][0]["name"] == "primary"
        assert (
            result_payload["execution"]["outputs"][0]["media_type"]
            == "application/json"
        )
        assert result_payload["execution"]["outputs"][0]["storage_path"] == str(
            Path(plan["workspace"]) / "outputs" / run_id / "result.json"
        )

        output_path = Path(result_payload["execution"]["output_path"]).resolve()
        output_root = (Path(plan["workspace"]) / "outputs").resolve()
        assert output_path.is_file()
        assert output_path.is_relative_to(output_root)

        artifact_status, artifact_bytes, artifact_headers = request_bytes(
            f"{plan['server_url']}/v1/infer/{plan['workflow_id']}/{run_id}/results?artifact=primary"
        )
        assert artifact_status == 200
        assert artifact_headers.get("content-type") == "application/json"
        assert artifact_bytes == b'{"ok": true}'
    finally:
        module._terminate_processes(processes)
        for log_handle in log_handles:
            log_handle.close()


def test_plugin_dev_probe_python_module_contract_reports_missing_attrs():
    module = load_plugin_dev_module()
    report = module._probe_python_module_contract("json", ["QueueManager"])

    assert report["ok"] is False
    assert "missing required attributes" in report["detail"]
    assert report["missing_attrs"] == ["QueueManager"]


def test_plugin_dev_probe_python_module_contract_uses_fresh_interpreter(
    tmp_path: Path, monkeypatch
):
    module = load_plugin_dev_module()

    package_root = tmp_path / "pydeps"
    package_root.mkdir()
    module_path = package_root / "demo_reload.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(package_root))

    first = module._probe_python_module_contract("demo_reload", ["QueueManager"])
    assert first["ok"] is False

    module_path.write_text("VALUE = 1\nQueueManager = object()\n", encoding="utf-8")
    second = module._probe_python_module_contract("demo_reload", ["QueueManager"])

    assert second["ok"] is True
    assert "QueueManager" in second["detail"]


def test_plugin_dev_bootstrap_python_dry_run_reports_local_scicomp_rq_install(
    monkeypatch,
):
    module = load_plugin_dev_module()

    def fake_probe(module_name: str, required_attrs: list[str]):
        if module_name == "redis":
            return {
                "module": "redis",
                "required_attrs": required_attrs,
                "missing_attrs": [],
                "ok": True,
                "detail": "module 'redis' exposes required attributes: Redis",
            }
        return {
            "module": "scicomp_rq",
            "required_attrs": required_attrs,
            "missing_attrs": ["QueueManager", "Output"],
            "ok": False,
            "detail": "module 'scicomp_rq' is missing required attributes: QueueManager, Output",
        }

    monkeypatch.setattr(module, "_probe_python_module_contract", fake_probe)

    result = module.bootstrap_python(dry_run=True)
    scicomp_rq_target = next(
        target for target in result["targets"] if target["module"] == "scicomp_rq"
    )

    assert result["status"] == "planned"
    assert scicomp_rq_target["status"] == "planned"
    assert "pip install -e" in scicomp_rq_target["command"]


def test_plugin_dev_run_local_prereq_error_mentions_bootstrap_python(monkeypatch):
    module = load_plugin_dev_module()

    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/usr/bin/fake" if name in {"redis-server", "cargo"} else None,
    )
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: object() if name == "redis" else None,
    )
    monkeypatch.setattr(
        module,
        "_probe_python_module_contract",
        lambda *_args, **_kwargs: {
            "module": "scicomp_rq",
            "required_attrs": ["QueueManager", "Output"],
            "missing_attrs": ["QueueManager", "Output"],
            "ok": False,
            "detail": "module 'scicomp_rq' is missing required attributes: QueueManager, Output",
        },
    )

    try:
        module.ensure_run_local_prerequisites(skip_build=False)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("ensure_run_local_prerequisites should have failed")

    assert "Local run prerequisites are missing" in message
    assert "bootstrap-python" in message
    assert "scicomp_rq" in message
