# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = REPO_ROOT / "packaging" / "physicsnemo-serve-cmd" / "assemble_runtime.py"


def _load_assembler():
    spec = importlib.util.spec_from_file_location(
        "physicsnemo_serve_runtime_assembler", ASSEMBLER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assemble_runtime_copies_interpreter_and_runner_support(tmp_path: Path) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f'#!/bin/sh\nexec {sys.executable!r} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    output = tmp_path / "runtime"

    module.assemble_runtime(
        repo_root=REPO_ROOT,
        python_prefix=python_prefix,
        output=output,
        requirements=None,
        wheelhouse=None,
    )

    assert (output / "bin" / "python").is_file()
    assert (output / "scripts" / "plugin_direct_runner.py").is_file()
    assert (output / "scripts" / "plugin_runtime.py").is_file()
    assert (output / "scripts" / "plugin_sdk.py").is_file()
    assert (output / "python").is_dir()
    manifest = json.loads(
        (output / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["requirements_sha256"] is None


def test_assemble_runtime_rejects_missing_interpreter(tmp_path: Path) -> None:
    module = _load_assembler()

    try:
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=tmp_path / "missing",
            output=tmp_path / "runtime",
            requirements=None,
            wheelhouse=None,
        )
    except ValueError as exc:
        assert "bin/python" in str(exc)
    else:
        raise AssertionError("missing interpreter should be rejected")


def test_assemble_runtime_rejects_venv_python_prefix(tmp_path: Path) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "venv-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    (python_prefix / "pyvenv.cfg").write_text("home = /test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="venvs and system Python prefixes"):
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=python_prefix,
            output=tmp_path / "runtime",
            requirements=None,
            wheelhouse=None,
        )


def test_assemble_runtime_rejects_absolute_interpreter_symlink(
    tmp_path: Path,
) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-relocatable absolute symlink"):
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=python_prefix,
            output=tmp_path / "runtime",
            requirements=None,
            wheelhouse=None,
        )


def test_assemble_runtime_does_not_publish_partial_output(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    output = tmp_path / "runtime"
    monkeypatch.setattr(module, "_require_uv", lambda: "uv")

    def fail_install(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated install failure")

    monkeypatch.setattr(module, "_install_requirements", fail_install)

    with pytest.raises(RuntimeError, match="simulated install failure"):
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=python_prefix,
            output=output,
            requirements=requirements,
            wheelhouse=None,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".physicsnemo-runtime-*"))


def test_assemble_runtime_rejects_output_inside_python_prefix(tmp_path: Path) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    output = python_prefix / "nested" / "runtime"

    with pytest.raises(ValueError, match="must not be inside the Python prefix"):
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=python_prefix,
            output=output,
            requirements=None,
            wheelhouse=None,
        )

    assert not (python_prefix / "nested").exists()


def test_assemble_runtime_rejects_output_inside_symlinked_python_prefix(
    tmp_path: Path,
) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    prefix_alias = tmp_path / "python-prefix-alias"
    prefix_alias.symlink_to(python_prefix, target_is_directory=True)
    output = prefix_alias / "nested" / ".." / "runtime"

    with pytest.raises(ValueError, match="must not be inside the Python prefix"):
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=python_prefix,
            output=output,
            requirements=None,
            wheelhouse=None,
        )

    assert not (python_prefix / "runtime").exists()


def test_assemble_runtime_verification_ignores_python_environment_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def capture_run(command, *, env, **_kwargs):
        calls.append((command, env))
        return subprocess.CompletedProcess(
            command, 0, stdout="Python 3.12.0\n", stderr=""
        )

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "host-modules"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "host-python"))
    monkeypatch.setattr(module.subprocess, "run", capture_run)

    module.assemble_runtime(
        repo_root=REPO_ROOT,
        python_prefix=python_prefix,
        output=tmp_path / "runtime",
        requirements=None,
        wheelhouse=None,
    )

    assert len(calls) == 2
    for command, environment in calls:
        assert command[1] == "-I"
        assert "PYTHONPATH" not in environment
        assert "PYTHONHOME" not in environment


def test_assemble_runtime_rewrites_staging_entrypoint_shebangs(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f'#!/bin/sh\nexec {sys.executable!r} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    (python_prefix / "BUILD").write_text("standalone\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    output = tmp_path / "runtime"
    monkeypatch.setattr(module, "_require_uv", lambda: "uv")

    def install_entrypoint(runtime_python, *_args, **_kwargs) -> None:
        entrypoint = runtime_python.parent / "fixture-command"
        entrypoint.write_text(
            f"#!{runtime_python}\nprint('relocatable-entrypoint')\n",
            encoding="utf-8",
        )
        entrypoint.chmod(0o755)

    monkeypatch.setattr(module, "_install_requirements", install_entrypoint)

    module.assemble_runtime(
        repo_root=REPO_ROOT,
        python_prefix=python_prefix,
        output=output,
        requirements=requirements,
        wheelhouse=None,
    )

    result = subprocess.run(
        [str(output / "bin" / "fixture-command")],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "relocatable-entrypoint"
    assert ".physicsnemo-runtime-" not in (
        output / "bin" / "fixture-command"
    ).read_text(encoding="utf-8")


def test_install_requirements_reports_missing_uv(monkeypatch, tmp_path: Path) -> None:
    module = _load_assembler()
    monkeypatch.setattr(shutil, "which", lambda executable: None)

    try:
        module._install_requirements(
            tmp_path / "python",
            tmp_path / "requirements.txt",
            None,
        )
    except RuntimeError as exc:
        assert "uv is required" in str(exc)
        assert "not found on PATH" in str(exc)
    else:
        raise AssertionError(
            "missing uv should be rejected before subprocess execution"
        )


@pytest.mark.parametrize(
    "target",
    ["build-serve-cmd-linux-amd64", "build-serve-installer-linux-amd64"],
)
def test_cross_build_targets_install_pinned_toolchain_before_target(
    target: str,
) -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split(f"{target}:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]

    install_position = recipe.index(
        "rustup toolchain install $(SERVE_CMD_RUST_TOOLCHAIN) --profile minimal"
    )
    cmake_position = recipe.index("command -v cmake")
    target_position = recipe.index(
        "rustup target add --toolchain $(SERVE_CMD_RUST_TOOLCHAIN)"
    )
    build_position = recipe.index("cargo +$(SERVE_CMD_RUST_TOOLCHAIN) zigbuild")

    assert cmake_position < install_position < target_position < build_position
    assert "rustup which" not in recipe


def test_cross_build_documentation_uses_the_pinned_toolchain() -> None:
    readme = (REPO_ROOT / "crates/physicsnemo-serve-cmd/README.md").read_text(
        encoding="utf-8"
    )

    assert "rustup toolchain install 1.94.1 --profile minimal" in readme
    assert "rustup target add --toolchain 1.94.1 x86_64-unknown-linux-gnu" in readme
    assert "brew install zig cmake" in readme


def test_cli_docker_rust_builder_installs_cmake() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.physicsnemo-serve-cmd").read_text(
        encoding="utf-8"
    )
    rust_builder = dockerfile.split(
        "FROM ${PHYSICSNEMO_SERVE_UBUNTU_IMAGE} AS rust-builder", maxsplit=1
    )[1].split("FROM ${PHYSICSNEMO_SERVE_UBUNTU_IMAGE} AS runtime-builder", maxsplit=1)[
        0
    ]

    assert "        cmake \\\n" in rust_builder
