# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application adapter for the existing Python plugin development tools."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any
import uuid

from .common import is_dotenv_path, require, verify_inventory

_GENERATED_DIRECTORIES = {
    ".git",
    ".pnms",
    ".venv",
    ".run-local",
    ".example-output",
    ".pytest_cache",
    "__pycache__",
}


def init_project(
    root: Path,
    name: str,
    *,
    repository: Path,
    pipeline_profile: str = "simple",
    runtime_profile: str = "python-test",
    phase_runtime_overrides: list[str] | None = None,
    executor_class: str | None = None,
    phase_executor_overrides: list[str] | None = None,
) -> dict[str, Any]:
    return _invoke(
        {"plugin": root},
        repository,
        "init",
        pipeline_profile=pipeline_profile,
        runtime_profile=runtime_profile,
        phase_runtime_overrides=phase_runtime_overrides,
        executor_class=executor_class,
        phase_executor_overrides=phase_executor_overrides,
    )


def check_project(
    project: dict[str, Any],
    *,
    repository: Path,
    config_only: bool = False,
    through_phase: str | None = None,
) -> dict[str, Any]:
    workspace = Path(project["root"]) / ".pnms" / "dev" / f"check-{uuid.uuid4().hex}"
    return _invoke(
        project,
        repository,
        "check",
        config_only=config_only,
        through_phase=through_phase,
        workspace=str(workspace),
    )


def run_project(
    project: dict[str, Any],
    input: Path,
    output: Path | None = None,
    *,
    repository: Path,
    dev: bool = False,
    through_phase: str | None = None,
) -> dict[str, Any]:
    output = Path(
        output
        or Path(project["root"])
        / ".pnms"
        / ("dev" if dev else "runs")
        / uuid.uuid4().hex
    ).resolve()
    return _invoke(
        project,
        repository,
        "run",
        input=str(Path(input).resolve()),
        workspace=str(output),
        through_phase=through_phase,
    )


def serve_project(
    project: dict[str, Any],
    *,
    repository: Path,
    dev: bool = True,
    port: int = 8080,
    workspace: Path | None = None,
    redis_port: int = 0,
    skip_build: bool = False,
    bootstrap_python_bindings: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    workspace = Path(
        workspace
        or Path(project["root"]) / ".pnms" / "dev" / f"serve-{uuid.uuid4().hex}"
    ).resolve()
    source = Path(project["plugin"]).resolve()
    if workspace == source:
        raise ValueError("Python service workspace cannot be the plugin directory")
    if workspace.is_relative_to(source):
        # Legacy CPU profiles may copy a plugin into the service workspace.
        # Give that copier clean input so service output cannot recursively copy itself.
        staged = workspace / "source" / source.name
        if staged.exists() or staged.is_symlink():
            staged = workspace / "source" / uuid.uuid4().hex / source.name
        if not staged.resolve().is_relative_to(workspace):
            raise ValueError("Python service staging must remain inside the workspace")
        _copy_plugin(source, staged, excluded_directory=workspace, exclude_dotenv=False)
        project = {**project, "plugin": staged}
    return _invoke(
        project,
        repository,
        "serve",
        foreground=not dry_run,
        workspace=str(workspace),
        port=port,
        redis_port=redis_port,
        skip_build=skip_build,
        bootstrap_python_bindings=bootstrap_python_bindings,
        dry_run=dry_run,
    )


def _invoke(
    project: dict[str, Any],
    repository: Path,
    operation: str,
    *,
    foreground: bool = False,
    **options: Any,
) -> dict[str, Any]:
    """Keep user imports, selected venv, stdout and module caches isolated."""
    executable = str(
        project.get("python_executable")
        or project.get("build", {}).get("python")
        or sys.executable
    )
    if os.sep in executable:
        executable = str(Path(executable).expanduser().absolute())
    plugin = Path(project["plugin"]).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="pnms-python-") as temporary:
        report_path = Path(temporary) / "report.json"
        run = _run_foreground if foreground and os.name == "posix" else subprocess.run
        proc = run(
            [
                executable,
                str(Path(__file__).with_name("python_host.py")),
                str(Path(repository).resolve()),
                str(report_path),
            ],
            input=json.dumps(
                {"operation": operation, "plugin": str(plugin), **options}
            ),
            capture_output=not foreground,
            text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                **(
                    {"PHYSICSNEMO_SERVE_BIND_HOST": "127.0.0.1"}
                    if operation == "serve"
                    else {}
                ),
            },
            cwd=plugin if plugin.is_dir() else None,
        )
        if not report_path.is_file():
            raise ValueError(
                f"Python plugin {operation} failed: {(proc.stderr or '').strip() or (proc.stdout or '').strip() or proc.returncode}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if proc.returncode:
            raise ValueError(
                str(
                    report.get("error")
                    or proc.stderr
                    or "Python plugin operation failed"
                )
            )
        if proc.stdout or proc.stderr:
            report["diagnostics"] = {"stdout": proc.stdout, "stderr": proc.stderr}
        return report


def _run_foreground(argv, *, input, capture_output, **options):
    from .service import _cleanup_signals

    process = subprocess.Popen(
        argv, stdin=subprocess.PIPE, start_new_session=True, **options
    )
    try:
        process.communicate(input)
    except KeyboardInterrupt:
        with _cleanup_signals():
            try:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            finally:
                # The host first runs the legacy stack's owned-child cleanup.
                # Stop anything left in its isolated group after that deadline.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        raise
    return subprocess.CompletedProcess(argv, process.returncode)


def build_project(
    project: dict[str, Any], output: Path, *, repository: Path
) -> dict[str, Any]:
    """Capture source once; root seals the resulting application build receipt."""
    source = Path(project["plugin"]).resolve()
    output = Path(output).resolve()
    if output == source or (
        output.is_relative_to(source)
        and output.relative_to(source).parts[0] not in _GENERATED_DIRECTORIES
    ):
        raise ValueError("Python build output inside the plugin must be under .pnms")
    snapshot = output / "plugin"
    _copy_plugin(source, snapshot)
    try:
        validation = _invoke({**project, "plugin": snapshot}, repository, "validate")
        _require_no_dotenv_paths(snapshot)
    except Exception:
        shutil.rmtree(snapshot)
        raise
    inventory = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    return {
        "status": "complete",
        "runtime": {"kind": "python"},
        "resources": validation.get("resources", {}),
        "plugin_root": str(snapshot),
        "validation": validation["validation"],
        "environment": validation["environment"],
        "inventory": inventory,
    }


def _require_no_dotenv_paths(root: Path) -> None:
    require(
        not any(is_dotenv_path(path.relative_to(root)) for path in root.rglob("*")),
        "Python plugin snapshot contains .env or .env.* paths; run pnms build "
        "to create a new snapshot and supply credentials through runtime environment variables",
    )


def verify_plugin_snapshot(root: Path, records: list[dict[str, Any]]) -> None:
    # Artifact verification must include every file, including unsafe old files.
    verify_inventory(root, records)
    _require_no_dotenv_paths(root)


def _copy_plugin(
    source: Path,
    destination: Path,
    *,
    excluded_directory: Path | None = None,
    exclude_dotenv: bool = True,
) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in _GENERATED_DIRECTORIES
            or (exclude_dotenv and is_dotenv_path(name))
            or Path(directory) / name == excluded_directory
        }

    for directory, folders, files in os.walk(source, followlinks=False):
        ignored = ignore(directory, [*folders, *files])
        folders[:] = [name for name in folders if name not in ignored]
        for name in [*folders, *files]:
            if name not in ignored and (Path(directory) / name).is_symlink():
                raise ValueError(
                    f"Python snapshots cannot contain a symlink: {Path(directory) / name}"
                )
    shutil.copytree(source, destination, ignore=ignore)
