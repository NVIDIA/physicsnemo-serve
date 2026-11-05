# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and verify the source fingerprint used by QA and deployed images."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[3]
BLOCKED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    ".run-local",
    ".example-output",
}
ASSET_SUFFIXES = {".ckpt", ".npy", ".pt", ".pth", ".onnx", ".zarr", ".log"}
SECRET_SUFFIXES = {".key", ".pem"}
MAX_SOURCE_BYTES = 10 * 1024**2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_source(repo, expected):
    repo = Path(repo)
    manifest = json.loads((repo / "qa-source-inventory.json").read_text())
    encoded = json.dumps(
        manifest["files"], sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("image source inventory differs from the requested snapshot")
    for item in manifest["files"]:
        if digest(repo / item["path"]) != item["sha256"]:
            raise ValueError(f"image source bytes differ: {item['path']}")
    return manifest["base_commit"]


def excluded(path):
    return (
        path.as_posix() in {"deploy/config.yaml", "qa-source-inventory.json"}
        or path.as_posix().startswith("qa/artifacts/")
        or any(
            part in BLOCKED_PARTS
            or part.startswith(".env")
            or Path(part).suffix in ASSET_SUFFIXES | SECRET_SUFFIXES
            for part in path.parts
        )
    )


def prepare_source(repo, output):
    repo, output = Path(repo).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source context already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("source context must be outside the repository")
    actual_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True
    ).strip()
    if Path(actual_root).resolve() != repo:
        raise ValueError("--repo must name the Git repository root")
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        text=True,
    )
    candidates = []
    for name in sorted(set(names.split("\0")) - {""}):
        relative = Path(name)
        if excluded(relative):
            continue
        source = repo / relative
        if source.is_symlink() or not source.resolve().is_relative_to(repo):
            raise ValueError(f"source context does not accept symlinks: {relative}")
        if not source.is_file():
            continue  # Tracked deletions and submodule directories have no file bytes.
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"source file exceeds 10 MiB: {relative}")
        candidates.append(relative)

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for relative in candidates:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
        data = target.read_bytes()
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "base_commit": base_commit,
        "source_content_sha256": identity,
        "files": records,
    }
    write_json(output / "qa-source-inventory.json", manifest)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_source(args.repo, args.output))


if __name__ == "__main__":
    main()
