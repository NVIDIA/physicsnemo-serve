# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip(), encoding="utf-8")


def _build_binary() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip(
            "cargo is required for the embedded-runtime binary integration test"
        )
    proc = subprocess.run(
        [cargo, "build", "-p", "physicsnemo-serve-cmd", "--bin", "physicsnemo-serve"],
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
    _write(
        runtime / "python" / "bundled_runtime_fixture.py",
        "BUNDLED_MULTIPLIER = 2",
    )
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
from bundled_runtime_fixture import BUNDLED_MULTIPLIER


def prepare(ctx):
    return {"parameters": {"value": int(ctx["parameters"]["value"])}}


def execute(ctx):
    return {"value": ctx["parameters"]["value"] * BUNDLED_MULTIPLIER}
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
    runtime_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(executable), "infer"]
    if runtime_dir is not None:
        command.extend(["--runtime-dir", str(runtime_dir)])
    command.extend(
        [
            "--plugin",
            str(plugin_root),
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "binary-smoke",
        ]
    )
    return subprocess.run(
        command,
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


def test_binary_runs_external_plugin_from_explicit_runtime_dir(tmp_path: Path) -> None:
    binary = _build_binary()
    runtime = _create_fixture_runtime(tmp_path)
    plugin_root, request_path = _create_plugin(tmp_path)
    cache_dir = tmp_path / "unused-runtime-cache"
    hostile_python_path = tmp_path / "host-python-path"
    _write(
        hostile_python_path / "bundled_runtime_fixture.py",
        "BUNDLED_MULTIPLIER = 99",
    )
    env = os.environ.copy()
    env["PHYSICSNEMO_SERVE_CLI_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = str(hostile_python_path)
    env["PYTHONHOME"] = str(tmp_path / "invalid-python-home")

    proc = _run_infer(
        binary,
        plugin_root,
        request_path,
        tmp_path / "outputs",
        env,
        runtime_dir=runtime,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["payload"] == {"value": 18}
    assert not cache_dir.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix process signals")
def test_binary_termination_stops_python_runner_process_group(tmp_path: Path) -> None:
    binary = _build_binary()
    runtime = _create_fixture_runtime(tmp_path)
    plugin_root, request_path = _create_plugin(tmp_path)
    runner_pid_path = tmp_path / "runner.pid"
    manifest_path = plugin_root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["ingress"]["json_schema_inline"] = {
        "type": "object",
        "required": ["runner_pid_path"],
        "properties": {"runner_pid_path": {"type": "string"}},
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _write(
        plugin_root / "workflow.py",
        """
import os
import json
import subprocess
import sys
import time
from pathlib import Path


def prepare(ctx):
    return {"parameters": ctx["parameters"]}


def execute(ctx):
    ready_path = Path(ctx["parameters"]["runner_pid_path"] + ".descendant-ready")
    descendant = subprocess.Popen([
        sys.executable,
        "-c",
        (
            "import signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(ready_path)!r}).write_text('ready'); "
            "time.sleep(60)"
        ),
    ])
    while not ready_path.is_file():
        time.sleep(0.01)
    Path(ctx["parameters"]["runner_pid_path"]).write_text(
        json.dumps({"runner": os.getpid(), "descendant": descendant.pid})
    )
    time.sleep(60)
    return {"unexpected": True}
""",
    )
    request_path.write_text(
        json.dumps({"runner_pid_path": str(runner_pid_path)}), encoding="utf-8"
    )
    command = [
        str(binary),
        "infer",
        "--runtime-dir",
        str(runtime),
        "--plugin",
        str(plugin_root),
        "--request",
        str(request_path),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--run-id",
        "termination-test",
    ]
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process_ids: list[int] = []
    try:
        deadline = time.monotonic() + 15
        while not runner_pid_path.is_file() and time.monotonic() < deadline:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                pytest.fail(
                    f"CLI exited before starting its runner: {stdout=} {stderr=}"
                )
            time.sleep(0.05)
        assert runner_pid_path.is_file(), "Python runner did not publish its PID"
        recorded_processes = json.loads(runner_pid_path.read_text(encoding="utf-8"))
        process_ids = [
            int(recorded_processes["runner"]),
            int(recorded_processes["descendant"]),
        ]

        proc.terminate()
        proc.communicate(timeout=15)

        deadline = time.monotonic() + 5
        while (
            any(_process_exists(pid) for pid in process_ids)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert not [pid for pid in process_ids if _process_exists(pid)], (
            "Python process tree survived CLI termination"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        for pid in process_ids:
            if _process_exists(pid):
                os.kill(pid, signal.SIGKILL)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
