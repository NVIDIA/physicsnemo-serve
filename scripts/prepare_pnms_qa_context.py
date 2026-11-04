#!/usr/bin/env python3
"""Capture source-bound CPU Docker contexts for native Lepton CI.

This promotes the existing capture_dev_context.py snapshot recipe into a tested
controller utility. It does not compile models or run acceptance locally.
"""

from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile


def _git(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source must be a regular non-symlink file: " + str(path))
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "mode": stat.S_IMODE(path.stat().st_mode),
        "symlink": None,
    }


def _excluded(relative):
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError("Unsafe source inventory path")
    blocked = {
        ".git",
        ".aws",
        ".ssh",
        ".kube",
        ".codex",
        ".claude",
        ".venv",
        "venv",
        ".pnms",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "target",
        "dist",
        ".egg-info",
    }
    for index, part in enumerate(path.parts):
        if part in blocked or part.startswith(".env") or part.endswith(".egg-info"):
            return True
        if part == "build" and path.parts[: index + 1] != (
            "native-inference",
            "model-builder",
            "src",
            "model_builder",
            "build",
        ):
            return True
    return (
        relative == "deploy/config.yaml"
        or relative.startswith("qa/artifacts/")
        or path.suffix in {".pyc", ".pyo", ".pem", ".key"}
        or path.name in {"credentials", "credentials.json", "id_rsa", "id_ed25519"}
    )


def prepare(repository, output, *, expected_source_sha, allow_dirty=False):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    revision = _git(repository, "rev-parse", "HEAD").decode().strip()
    if (
        not re.fullmatch("[a-f0-9]{40}", expected_source_sha)
        or revision != expected_source_sha
    ):
        raise ValueError("Source revision differs from the current checkout")
    initial_status = _git(repository, "status", "--porcelain")
    dirty = bool(initial_status)
    if dirty and not allow_dirty:
        raise ValueError("Automatic QA requires a clean source checkout")
    raw = _git(
        repository, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    )
    paths = sorted(
        set(
            value
            for value in raw.decode().split("\0")
            if value and not _excluded(value)
        )
    )
    records = {}
    for relative in paths:
        path = repository / relative
        current = path
        while current != repository:
            if current.is_symlink():
                raise ValueError("Source path is a symlink: " + relative)
            current = current.parent
        records[relative] = _identity(path)
    mandatory = {
        "pnms",
        "qa/native_inference/Dockerfile.pnms",
        "qa/native_inference/venv-requirements.txt",
        "native-inference/model-builder/src/model_builder/build/cli.py",
        "native-inference/model-builder/images/Dockerfile.builder",
        "native-inference/model-builder/images/Dockerfile.builder.dockerignore",
    }
    if not mandatory <= records.keys():
        raise ValueError("Required native QA/build source is absent")
    output.mkdir(parents=True, exist_ok=False)
    dev, builder = output / "dev-context", output / "builder-context"
    dev.mkdir()
    with tarfile.open(dev / "source.tar", "w") as archive:
        for relative, record in records.items():
            data = (repository / relative).read_bytes()
            if (
                len(data) != record["size_bytes"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                raise ValueError("Source identity changed during capture: " + relative)
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = len(data), record["mode"], 0
            archive.addfile(info, io.BytesIO(data))
            if relative.startswith("native-inference/"):
                path = builder / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(record["mode"])
                if _identity(path) != record:
                    raise ValueError("Builder source identity changed during copy")
            elif relative == "qa/native_inference/Dockerfile.pnms":
                (dev / "Dockerfile.pnms").write_bytes(data)
    if any(
        _identity(repository / relative) != record
        for relative, record in records.items()
    ):
        raise ValueError("Source identity changed during capture")
    if (
        _git(repository, "rev-parse", "HEAD").decode().strip() != revision
        or _git(repository, "status", "--porcelain") != initial_status
    ):
        raise ValueError("Source checkout changed during capture")
    (dev / "source-inventory.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    metadata = {
        "source_sha": revision,
        "source_dirty": dirty,
        "file_count": len(records),
        "source_archive_sha256": _identity(dev / "source.tar")["sha256"],
        "source_inventory_sha256": _identity(dev / "source-inventory.json")["sha256"],
    }
    (dev / "source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Explicit local snapshot capture; never used by automatic CI",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(
                args.repository,
                args.output,
                expected_source_sha=args.expected_source_sha,
                allow_dirty=args.allow_dirty,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
