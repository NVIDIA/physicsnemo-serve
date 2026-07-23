# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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
    assert not marker.exists()


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
