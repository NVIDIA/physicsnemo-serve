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
    if output.exists():
        raise ValueError(f"Runtime output already exists: {output}")
    if requirements is not None and not requirements.is_file():
        raise ValueError(f"Requirements lock does not exist: {requirements}")
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise ValueError(f"Wheelhouse does not exist: {wheelhouse}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(python_prefix, output, symlinks=True)
    runtime_python = output / "bin" / "python"
    if requirements is not None:
        _install_requirements(runtime_python, requirements, wheelhouse)

    scripts_dir = output / "scripts"
    scripts_dir.mkdir()
    for script_name in SUPPORT_SCRIPTS:
        source = repo_root / "scripts" / script_name
        if not source.is_file():
            raise ValueError(f"Required support script does not exist: {source}")
        shutil.copy2(source, scripts_dir / script_name)

    python_dir = output / "python"
    shutil.copytree(repo_root / "python", python_dir)
    for package in extra_python_packages or []:
        if not package.is_dir():
            raise ValueError(f"Extra Python package does not exist: {package}")
        destination = python_dir / package.name
        if destination.exists():
            raise ValueError(
                f"Extra Python package destination already exists: {destination}"
            )
        shutil.copytree(package, destination, symlinks=True)

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
    (output / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _install_requirements(
    python: Path,
    requirements: Path,
    wheelhouse: Path | None,
) -> None:
    command = [
        "uv",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
