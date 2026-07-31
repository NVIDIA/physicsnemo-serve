# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "plugin_direct_runner.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip(), encoding="utf-8")


def _write_plugin(
    root: Path,
    *,
    plugin_id: str,
    profile: str,
    workflow: str,
    request_schema: dict[str, object],
    options: dict[str, object] | None = None,
) -> Path:
    plugin_root = root / plugin_id
    manifest = {
        "metadata": {
            "id": plugin_id,
            "display_name": plugin_id,
            "version": "1.0.0",
            "description": "Direct runner test plugin",
        },
        "ingress": {
            "content_type": "application/json",
            "operation": {"default": "run", "allowed": ["run"]},
            "json_schema_inline": request_schema,
        },
        "pipeline": {"profile": profile, "options": options or {}},
        "runtime": {
            "kind": "python",
            "entrypoint": "workflow.py",
            "executor_class": "python.test",
        },
    }
    _write(plugin_root / "plugin.yaml", yaml.safe_dump(manifest, sort_keys=False))
    _write(plugin_root / "workflow.py", workflow)
    return plugin_root


def _run_direct(
    plugin_root: Path,
    request: dict[str, object],
    output_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    request_path = output_dir.parent / f"{plugin_root.name}-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plugin-root",
            str(plugin_root),
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "direct-test",
        ],
        cwd=REPO_ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_direct_runner_executes_postprocess_pipeline_without_services(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-postprocess",
        profile="postprocess",
        options={},
        request_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "integer", "minimum": 1}},
        },
        workflow="""
from plugin_sdk import PluginWorkflow, PostprocessOutcome, PrepareResult


class DirectWorkflow(PluginWorkflow):
    def prepare(self, request, ctx):
        return PrepareResult(inputs={"value": int(request.raw_fields["value"])})

    def execute(self, ctx):
        output_path = ctx["outputs"].create(
            "primary",
            filename="result.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {"value": ctx["parameters"]["value"], "postprocessed": False}

    def postprocess(self, result, ctx):
        payload = dict(result.payload)
        payload["postprocessed"] = True
        return PostprocessOutcome(payload=payload, status=result.execution.status)


WORKFLOW = DirectWorkflow
""",
    )

    proc = _run_direct(plugin_root, {"value": 7}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "succeeded"
    assert result["workflow"] == "direct-postprocess"
    assert result["payload"] == {"postprocessed": True, "value": 7}
    assert result["execution"]["output_path"].endswith("direct-test/result.json")
    assert result["execution"]["outputs"][0]["name"] == "primary"
    assert result["request"]["raw_fields"] == {"value": 7}


def test_direct_runner_preserves_execution_failure_through_postprocess(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-failed-postprocess",
        profile="postprocess",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class FailedPostprocessWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {
            "status": "failed",
            "error": "execution failed",
            "error_traceback": "execution traceback",
        }

    def postprocess(self, result, ctx):
        return {"postprocessed": True}


WORKFLOW = FailedPostprocessWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "failed"
    assert result["execution"]["error"] == "execution failed"
    assert result["execution"]["error_traceback"] == "execution traceback"
    assert result["payload"] == {"postprocessed": True}


def test_direct_runner_merges_outputs_registered_during_postprocess(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-postprocess-output",
        profile="postprocess",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow, PostprocessOutcome


class PostprocessOutputWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"value": 1}

    def postprocess(self, result, ctx):
        explicit_path = ctx.run_dir / "explicit.json"
        explicit_path.write_text('{"explicit": true}', encoding="utf-8")
        output_path = ctx.outputs.create(
            "postprocessed",
            filename="postprocessed.json",
            media_type="application/json",
            primary=True,
        )
        output_path.write_text('{"postprocessed": true}', encoding="utf-8")
        return PostprocessOutcome(payload={
            "value": 2,
            "artifacts": [{
                "name": "explicit",
                "media_type": "application/json",
                "storage_path": str(explicit_path),
            }],
        })


WORKFLOW = PostprocessOutputWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert [artifact["name"] for artifact in result["execution"]["outputs"]] == [
        "explicit",
        "postprocessed",
    ]
    assert result["execution"]["output_path"].endswith("postprocessed.json")
    assert result["payload"] == {"value": 2}


def test_direct_runner_preserves_execute_artifacts_when_postprocess_registers_output(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-merged-postprocess-outputs",
        profile="postprocess",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow, PostprocessOutcome


class MergedOutputWorkflow(PluginWorkflow):
    def execute(self, ctx):
        output_path = ctx["run_dir"] / "execute.json"
        output_path.write_text('{"execute": true}', encoding="utf-8")
        return {
            "artifacts": [{
                "name": "execute",
                "media_type": "application/json",
                "storage_path": str(output_path),
                "primary": True,
            }],
            "output_path": str(output_path),
        }

    def postprocess(self, result, ctx):
        output_path = ctx.outputs.create(
            "postprocess",
            filename="postprocess.json",
            media_type="application/json",
        )
        output_path.write_text('{"postprocess": true}', encoding="utf-8")
        return PostprocessOutcome(payload={"merged": True})


WORKFLOW = MergedOutputWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert [artifact["name"] for artifact in result["execution"]["outputs"]] == [
        "execute",
        "postprocess",
    ]
    assert result["execution"]["output_path"].endswith("postprocess.json")
    assert result["payload"] == {"merged": True}


def test_direct_runner_rejects_postprocess_result_operations(tmp_path: Path) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-postprocess-ops",
        profile="postprocess",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import DatasetExportNetcdfOp, PluginWorkflow, PostprocessOutcome


class PostprocessOpsWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"value": 1}

    def postprocess(self, result, ctx):
        return PostprocessOutcome(
            payload={"value": 2},
            result_ops=[DatasetExportNetcdfOp()],
        )


WORKFLOW = PostprocessOpsWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert "does not support postprocess result_ops" in proc.stderr
    assert (
        "does not support postprocess result_ops"
        in json.loads(proc.stdout)["execution"]["error"]
    )


def test_direct_runner_materializes_prefetch_and_skips_schedule(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"prefetched input")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-prefetch",
        profile="prefetch",
        options={},
        request_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "sha256", "size_bytes"],
            "properties": {
                "source": {"type": "string"},
                "sha256": {"type": "string"},
                "size_bytes": {"type": "integer"},
            },
        },
        workflow="""
from pathlib import Path

from plugin_sdk import PluginWorkflow, PrepareResult


class DirectPrefetchWorkflow(PluginWorkflow):
    def prepare(self, request, ctx):
        fields = request.raw_fields
        return PrepareResult(
            inputs=fields,
            prefetch_plan=[
                {
                    "kind": "file_copy",
                    "source_uri": fields["source"],
                    "target_artifact_name": "input",
                    "required": True,
                    "expected_sha256": fields["sha256"],
                    "expected_size_bytes": fields["size_bytes"],
                }
            ],
        )

    def execute(self, ctx):
        artifact = ctx["prefetch_artifacts"][0]
        return {
            "content": Path(artifact["storage_path"]).read_text(encoding="utf-8"),
            "verified_sha256": artifact["verified_sha256"],
        }


WORKFLOW = DirectPrefetchWorkflow
""",
    )
    helper = tmp_path / "fake-prefetch-helper"
    _write(
        helper,
        f"""
#!{sys.executable}
import hashlib
import json
import shutil
import sys
from pathlib import Path

plan = json.load(sys.stdin)
item = plan[0]
source = Path(item["source_uri"])
target = Path(sys.argv[sys.argv.index("--cache-dir") + 1]) / source.name
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
json.dump({{
    "artifacts": [{{
        "name": item["target_artifact_name"],
        "source_uri": item["source_uri"],
        "storage_path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": digest,
        "verified_sha256": digest,
    }}],
    "stats": {{"downloaded": 1, "cached": 0, "errors": 0}},
}}, sys.stdout)
""",
    )
    helper.chmod(0o755)

    proc = _run_direct(
        plugin_root,
        {
            "source": str(source),
            "sha256": digest,
            "size_bytes": source.stat().st_size,
        },
        tmp_path / "outputs",
        env={"PHYSICSNEMO_SERVE_PREFETCH_HELPER": str(helper)},
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["payload"]["content"] == "prefetched input"
    assert result["payload"]["verified_sha256"] == digest
    assert result["execution"]["prefetch"]["downloaded"] == 1


def test_direct_runner_rejects_invalid_request_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-validation",
        profile="simple",
        options={},
        request_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "integer", "minimum": 1}},
        },
        workflow=f"""
from pathlib import Path

from plugin_sdk import PluginWorkflow


class ValidationWorkflow(PluginWorkflow):
    def execute(self, ctx):
        Path({str(marker)!r}).touch()
        return {{"value": ctx["parameters"]["value"]}}


WORKFLOW = ValidationWorkflow
""",
    )

    proc = _run_direct(plugin_root, {"value": 0}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert "does not conform to schema" in proc.stderr
    error_result = json.loads(proc.stdout)
    assert error_result["status"] == "failed"
    assert "does not conform to schema" in error_result["execution"]["error"]
    assert not marker.exists()


@pytest.mark.parametrize(
    ("stage_ids", "message"),
    [
        ([None, None], "non-empty ids"),
        (["execute", "execute"], "duplicate stage id 'execute'"),
    ],
)
def test_direct_runner_rejects_missing_or_duplicate_stage_ids(
    tmp_path: Path,
    stage_ids: list[str | None],
    message: str,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-invalid-stages",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class InvalidStagesWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"ok": True}


WORKFLOW = InvalidStagesWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["pipeline"] = {
        "stages": [
            {
                "id": stage_ids[0],
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": stage_ids[1],
            },
            {
                "id": stage_ids[1],
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert message in proc.stderr
    assert message in json.loads(proc.stdout)["execution"]["error"]


@pytest.mark.parametrize(
    ("next_value", "message"),
    [
        (None, "must define a non-empty next stage"),
        ("missing-stage", "references unknown next stage 'missing-stage'"),
    ],
)
def test_direct_runner_rejects_invalid_pipeline_transition(
    tmp_path: Path,
    next_value: str | None,
    message: str,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-invalid-transition",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class InvalidTransitionWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"ok": True}


WORKFLOW = InvalidTransitionWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["pipeline"] = {
        "stages": [
            {
                "id": "execute",
                "phase": "execute",
                "handler": "plugin_phase",
                "queue": "execute.python.test",
                "next": next_value,
            },
            {
                "id": "results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert message in proc.stderr
    assert message in json.loads(proc.stdout)["execution"]["error"]


def test_direct_runner_rejects_transition_after_results(tmp_path: Path) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-results-transition",
        profile="postprocess",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class ResultsTransitionWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"ok": True}

    def postprocess(self, result, ctx):
        return {"postprocessed": True}


WORKFLOW = ResultsTransitionWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["pipeline"] = {
        "stages": [
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
                "next": "postprocess",
            },
            {
                "id": "postprocess",
                "phase": "postprocess",
                "handler": "plugin_phase",
                "queue": "postprocess",
                "next": "final-results",
            },
            {
                "id": "final-results",
                "phase": "results",
                "handler": "persist_results",
                "queue": "results",
                "next": None,
            },
        ]
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert "results stage 'results' must be terminal" in proc.stderr
    assert (
        "results stage 'results' must be terminal"
        in json.loads(proc.stdout)["execution"]["error"]
    )


def test_direct_runner_defaults_content_type_and_skips_empty_operation_allowlist(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-ingress-defaults",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class IngressDefaultsWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"operation": ctx["operation"]}


WORKFLOW = IngressDefaultsWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["ingress"].pop("content_type")
    manifest["ingress"]["operation"] = {"default": "run", "allowed": []}
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(
        plugin_root,
        {"operation": "custom-operation"},
        tmp_path / "outputs",
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["request"]["content_type"] == "application/json"
    assert result["request"]["operation"] == "custom-operation"
    assert result["payload"]["operation"] == "custom-operation"


@pytest.mark.parametrize("operation", ["", "   "])
def test_direct_runner_rejects_blank_operation_with_empty_allowlist(
    tmp_path: Path,
    operation: str,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-blank-operation",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class BlankOperationWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"operation": ctx["operation"]}


WORKFLOW = BlankOperationWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["ingress"]["operation"] = {"default": "run", "allowed": []}
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(
        plugin_root,
        {"operation": operation},
        tmp_path / "outputs",
    )

    assert proc.returncode == 1
    assert "Unsupported operation" in proc.stderr
    assert "Unsupported operation" in json.loads(proc.stdout)["execution"]["error"]


@pytest.mark.parametrize(
    ("content_types", "expected_returncode", "message"),
    [
        (["application/json"], 0, None),
        ("application/json", 1, "ingress.content_types must be an array"),
        (
            ["application/json", 42],
            1,
            "ingress.content_types entries must be non-empty strings",
        ),
    ],
)
def test_direct_runner_validates_plural_content_types(
    tmp_path: Path,
    content_types: object,
    expected_returncode: int,
    message: str | None,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-content-types",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class ContentTypesWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"ok": True}


WORKFLOW = ContentTypesWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["ingress"].pop("content_type")
    manifest["ingress"]["content_types"] = content_types
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == expected_returncode
    if message is not None:
        assert message in proc.stderr
        assert message in json.loads(proc.stdout)["execution"]["error"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("operations", ["run"], "ingress.operations must be a string or object"),
        (
            "operation",
            {"default": "run", "allowed": "run"},
            "ingress.operations.allowed must be an array of non-empty strings",
        ),
        (
            "operations",
            {"default": "run", "allowed": ["other"]},
            "ingress.operations.default must be included in ingress.operations.allowed",
        ),
    ],
)
def test_direct_runner_rejects_malformed_operation_declarations(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-invalid-operations",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class InvalidOperationsWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {"ok": True}


WORKFLOW = InvalidOperationsWorkflow
""",
    )
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["ingress"].pop("operation", None)
    manifest["ingress"][field] = value
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert message in proc.stderr
    assert message in json.loads(proc.stdout)["execution"]["error"]


def test_direct_runner_executes_batch_profile_as_one_item(tmp_path: Path) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-batch",
        profile="batch",
        options={},
        request_schema={
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
        },
        workflow="""
from dataclasses import dataclass

from plugin_sdk import BatchItem, BatchExecutionContext, PluginWorkflow


@dataclass
class BatchInput:
    value: int


class BatchWorkflow(PluginWorkflow):
    input_model = BatchInput

    def run_batch(
        self,
        items: list[BatchItem[BatchInput]],
        ctx: BatchExecutionContext,
    ):
        return [{"status": "success", "value": item.inputs.value * 3} for item in items]


WORKFLOW = BatchWorkflow
""",
    )

    proc = _run_direct(plugin_root, {"value": 4}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "succeeded"
    assert result["payload"]["value"] == 12


def test_direct_runner_sets_plugin_environment_before_import(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="import-environment",
        profile="batch",
        options={},
        request_schema={"type": "object"},
        workflow="""
import os

from plugin_sdk import PluginWorkflow


IMPORTED_ENVIRONMENT = {
    "default_output_dir": os.environ["DEFAULT_OUTPUT_DIR"],
    "plugin_dir": os.environ["PLUGIN_DIR"],
    "plugin_id": os.environ["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"],
}


class EnvironmentWorkflow(PluginWorkflow):
    def run_batch(self, items, ctx):
        return [IMPORTED_ENVIRONMENT]


WORKFLOW = EnvironmentWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, output_dir)

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["payload"] == {
        "default_output_dir": str(output_dir),
        "plugin_dir": str(plugin_root),
        "plugin_id": "import-environment",
    }


def test_direct_runner_preserves_batch_item_failure(tmp_path: Path) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-failed-batch",
        profile="batch",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import BatchItemResult, PluginWorkflow


class FailedBatchWorkflow(PluginWorkflow):
    def run_batch(self, items, ctx):
        return [BatchItemResult.failed("batch item failed")]


WORKFLOW = FailedBatchWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "failed"
    assert result["execution"]["error"] == "batch item failed"
    assert result["payload"] == {}


def test_direct_runner_keeps_plugin_output_out_of_json_protocol(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-noisy-plugin",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
import ctypes
import os
import subprocess
import sys

from plugin_sdk import PluginWorkflow

print("noise emitted while importing plugin")
os.write(1, b"native noise emitted while importing plugin\\n")


class NoisyWorkflow(PluginWorkflow):
    def execute(self, ctx):
        print("noise emitted while executing plugin")
        os.write(1, b"native noise emitted while executing plugin\\n")
        ctypes.CDLL(None).printf(b"buffered native noise emitted while exiting")
        subprocess.run(
            [sys.executable, "-c", "print('child process noise')"],
            check=True,
        )
        return {"ok": True}


WORKFLOW = NoisyWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["payload"] == {"ok": True}
    assert "noise emitted" not in proc.stdout
    assert "native noise emitted" not in proc.stdout
    assert "child process noise" not in proc.stdout
    assert "native noise emitted" in proc.stderr
    assert "buffered native noise emitted" in proc.stderr
    assert "child process noise" in proc.stderr


def test_direct_runner_moves_failure_and_timing_metadata_to_execution(
    tmp_path: Path,
) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-failed-result",
        profile="simple",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow


class FailedWorkflow(PluginWorkflow):
    def execute(self, ctx):
        return {
            "status": "failed",
            "error": "model execution failed",
            "error_traceback": "traceback details",
            "execution_time_seconds": 1.25,
            "diagnostic_code": "E_MODEL",
        }


WORKFLOW = FailedWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "failed"
    assert result["execution"]["error"] == "model execution failed"
    assert result["execution"]["error_traceback"] == "traceback details"
    assert result["execution"]["execution_time_seconds"] == 1.25
    assert result["payload"] == {"diagnostic_code": "E_MODEL"}


def test_direct_runner_rejects_ensemble_pipeline_explicitly(tmp_path: Path) -> None:
    plugin_root = _write_plugin(
        tmp_path,
        plugin_id="direct-ensemble",
        profile="ensemble",
        options={},
        request_schema={"type": "object"},
        workflow="""
from plugin_sdk import PluginWorkflow, PrepareResult


class EnsembleWorkflow(PluginWorkflow):
    def prepare(self, request, ctx):
        return PrepareResult(
            inputs={},
            fanout_profile={"item_count": 1},
            fanout_items=[{"item_index": 0, "parameters": {}}],
        )

    def execute(self, ctx):
        return {"ok": True}


WORKFLOW = EnsembleWorkflow
""",
    )

    proc = _run_direct(plugin_root, {}, tmp_path / "outputs")

    assert proc.returncode == 1
    assert "does not support pipeline phase 'fanout'" in proc.stderr
