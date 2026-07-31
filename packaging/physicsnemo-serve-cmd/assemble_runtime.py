#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


SUPPORT_SCRIPTS = (
    "plugin_direct_runner.py",
    "plugin_runtime.py",
    "plugin_sdk.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a filesystem runtime for physicsnemo-serve packaging."
    )
    parser.add_argument("--python-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--requirements")
    parser.add_argument("--wheelhouse")
    parser.add_argument(
        "--extra-python-package",
        action="append",
        default=[],
        help="Additional package directory to copy into runtime/python",
    )
    args = parser.parse_args()

    assemble_runtime(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        python_prefix=Path(args.python_prefix).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        requirements=(
            Path(args.requirements).expanduser().resolve()
            if args.requirements
            else None
        ),
        wheelhouse=(
            Path(args.wheelhouse).expanduser().resolve() if args.wheelhouse else None
        ),
        extra_python_packages=[
            Path(value).expanduser().resolve() for value in args.extra_python_package
        ],
    )
    return 0


def assemble_runtime(
    *,
    repo_root: Path,
    python_prefix: Path,
    output: Path,
    requirements: Path | None,
    wheelhouse: Path | None,
    extra_python_packages: list[Path] | None = None,
) -> None:
    python = python_prefix / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"Python prefix is missing bin/python: {python}")
    if not (python_prefix / "BUILD").is_file():
        raise ValueError(
            "Python prefix must be a standalone uv-managed Python installation "
            "containing BUILD; venvs and system Python prefixes are not relocatable"
        )
    _validate_relocatable_symlinks(python_prefix)
    if output.exists():
        raise ValueError(f"Runtime output already exists: {output}")
    if requirements is not None and not requirements.is_file():
        raise ValueError(f"Requirements lock does not exist: {requirements}")
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise ValueError(f"Wheelhouse does not exist: {wheelhouse}")
    for script_name in SUPPORT_SCRIPTS:
        source = repo_root / "scripts" / script_name
        if not source.is_file():
            raise ValueError(f"Required support script does not exist: {source}")
    for package in extra_python_packages or []:
        if not package.is_dir():
            raise ValueError(f"Extra Python package does not exist: {package}")
    uv = _require_uv() if requirements is not None else None

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".physicsnemo-runtime-", dir=output.parent)
    )
    staging = staging_root / "runtime"
    try:
        shutil.copytree(python_prefix, staging, symlinks=True)
        runtime_python = staging / "bin" / "python"
        if requirements is not None:
            _install_requirements(runtime_python, requirements, wheelhouse, uv=uv)

        _rewrite_python_entrypoints(staging)

        scripts_dir = staging / "scripts"
        scripts_dir.mkdir()
        for script_name in SUPPORT_SCRIPTS:
            source = repo_root / "scripts" / script_name
            shutil.copy2(source, scripts_dir / script_name)

        python_dir = staging / "python"
        shutil.copytree(repo_root / "python", python_dir)
        for package in extra_python_packages or []:
            destination = python_dir / package.name
            if destination.exists():
                raise ValueError(
                    f"Extra Python package destination already exists: {destination}"
                )
            shutil.copytree(package, destination, symlinks=True)

        _validate_relocatable_symlinks(staging)

        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        version = subprocess.run(
            [str(runtime_python), "--version"],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [str(runtime_python), "-c", "import jsonschema, yaml"],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        manifest = {
            "schema_version": 1,
            "python_version": (version.stdout or version.stderr).strip(),
            "requirements_sha256": (
                _sha256_file(requirements) if requirements is not None else None
            ),
            "support_scripts": list(SUPPORT_SCRIPTS),
            "extra_python_packages": [
                package.name for package in extra_python_packages or []
            ],
        }
        (staging / "runtime-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _validate_relocatable_symlinks(python_prefix: Path) -> None:
    prefix_root = python_prefix.resolve(strict=True)
    for path in python_prefix.rglob("*"):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        if target.is_absolute():
            raise ValueError(
                f"Python prefix contains a non-relocatable absolute symlink: {path}"
            )
        try:
            path.resolve(strict=True).relative_to(prefix_root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"Python prefix contains a symlink that escapes the runtime: {path}"
            ) from error


def _rewrite_python_entrypoints(runtime: Path) -> None:
    """Replace staging-path shebangs with wrappers resolved from each command."""
    bin_dir = runtime / "bin"
    for path in bin_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as entrypoint:
            first_line = entrypoint.readline(4096)
        if not first_line.startswith(b"#!") or not first_line.endswith(b"\n"):
            continue
        try:
            interpreter = Path(os.fsdecode(first_line[2:-1]))
        except UnicodeDecodeError:
            continue
        if interpreter.parent != bin_dir or not interpreter.name.startswith("python"):
            continue
        contents = path.read_bytes()
        _, separator, body = contents.partition(b"\n")
        if not separator:
            continue
        wrapper = (
            b"#!/bin/sh\n"
            b'\'\'\'exec\' "$(dirname -- "$(realpath -- "$0")")"/\'python\' "$0" "$@"\n'
            b"' '''\n"
        )
        path.write_bytes(wrapper + body)


def _install_requirements(
    python: Path,
    requirements: Path,
    wheelhouse: Path | None,
    *,
    uv: str | None = None,
) -> None:
    uv = uv or _require_uv()
    command = [
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--break-system-packages",
        "--require-hashes",
        "--requirements",
        str(requirements),
    ]
    if wheelhouse is not None:
        command.extend(["--no-index", "--find-links", str(wheelhouse)])
    subprocess.run(command, check=True)


def _require_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv is required to install runtime requirements but was not found on PATH; "
            "install it from https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
