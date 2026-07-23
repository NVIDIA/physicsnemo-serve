# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip(), encoding="utf-8")


def _build_binary() -> Path:
    proc = subprocess.run(
        ["cargo", "build", "-p", "physicsnemo-serve-cmd", "--bin", "physicsnemo-serve"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return REPO_ROOT / "target" / "debug" / "physicsnemo-serve"


def _create_fixture_runtime(root: Path) -> Path:
    runtime = root / "runtime"
    _write(
        runtime / "bin" / "python",
        f"""
#!/bin/sh
exec {sys.executable!r} "$@"
""",
    )
    (runtime / "bin" / "python").chmod(0o755)
    (runtime / "scripts").mkdir()
    for script_name in (
        "plugin_direct_runner.py",
        "plugin_runtime.py",
        "plugin_sdk.py",
    ):
        shutil.copy2(
            REPO_ROOT / "scripts" / script_name, runtime / "scripts" / script_name
        )
    (runtime / "python").mkdir()
    return runtime


def _create_plugin(root: Path) -> tuple[Path, Path]:
    plugin_root = root / "external-plugin"
    manifest = {
        "metadata": {
            "id": "external-plugin",
            "display_name": "External Plugin",
            "version": "1.0.0",
            "description": "Packaged CLI smoke plugin",
        },
        "ingress": {
            "content_type": "application/json",
            "operation": {"default": "run", "allowed": ["run"]},
            "json_schema_inline": {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "integer"}},
            },
        },
        "pipeline": {"profile": "simple"},
        "runtime": {
            "kind": "python",
            "entrypoint": "workflow.py",
            "executor_class": "python.test",
        },
    }
    _write(plugin_root / "plugin.yaml", yaml.safe_dump(manifest, sort_keys=False))
    _write(
        plugin_root / "workflow.py",
        """
def prepare(ctx):
    return {"parameters": {"value": int(ctx["parameters"]["value"])}}


def execute(ctx):
    return {"value": ctx["parameters"]["value"] * 2}
""",
    )
    request_path = root / "request.json"
    request_path.write_text('{"value": 9}', encoding="utf-8")
    return plugin_root, request_path


def _run_infer(
    executable: Path,
    plugin_root: Path,
    request_path: Path,
    output_dir: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(executable),
            "infer",
            "--plugin",
            str(plugin_root),
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "binary-smoke",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_binary_runs_external_plugin_from_embedded_runtime(tmp_path: Path) -> None:
    binary = _build_binary()
    runtime = _create_fixture_runtime(tmp_path)
    plugin_root, request_path = _create_plugin(tmp_path)
    packaged = tmp_path / "physicsnemo-serve"

    package_proc = subprocess.run(
        [
            str(binary),
            "package",
            "--runtime-dir",
            str(runtime),
            "--output",
            str(packaged),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert package_proc.returncode == 0, package_proc.stderr

    env = os.environ.copy()
    env.pop("REDIS_URL", None)
    env["PHYSICSNEMO_SERVE_CLI_CACHE_DIR"] = str(tmp_path / "runtime-cache")
    proc = _run_infer(
        packaged,
        plugin_root,
        request_path,
        tmp_path / "outputs",
        env,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["payload"] == {"value": 18}
    assert result["status"] == "succeeded"
    assert list((tmp_path / "runtime-cache").iterdir())
