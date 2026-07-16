# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symlink-safe file primitives shared by PhysicsNeMo-CFD plugins."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


def validated_run_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {path}")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return validated_directory(path, label="run directory")


def create_attempt_directory(run_root: Path, *, prefix: str) -> Path:
    root = validated_directory(run_root, label="run directory")
    attempt = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    resolved = validated_directory(attempt, label="attempt directory")
    if not resolved.is_relative_to(root):
        raise ValueError(f"attempt directory escapes run directory: {attempt}")
    return resolved


def open_exclusive_file(path: Path, *, mode: int = 0o600) -> int:
    validated_directory(path.parent, label=f"{path.name} parent directory")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("secure file creation requires O_NOFOLLOW")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(path, flags, mode)
    except OSError as exc:
        raise ValueError(f"refusing existing or unsafe output path: {path}") from exc


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = open_exclusive_file(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    fsync_directory(path.parent)


def ensure_empty_file(path: Path) -> None:
    if os.path.lexists(path):
        return
    descriptor = open_exclusive_file(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def validated_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} is not a directory: {path}")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError(
            "secure directory validation requires O_NOFOLLOW/O_DIRECTORY"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow | directory)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        after = os.lstat(path)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or (
        after.st_dev,
        after.st_ino,
    ) != (opened.st_dev, opened.st_ino):
        raise ValueError(f"{label} changed during validation: {path}")
    return path.resolve(strict=True)


def create_fresh_child_directory(parent: Path, name: str) -> Path:
    parent_resolved = validated_directory(parent, label="managed parent directory")
    child = parent_resolved / name
    if os.path.lexists(child):
        raise ValueError(f"fresh directory already exists: {child}")
    child.mkdir(mode=0o700)
    resolved = validated_directory(child, label="fresh directory")
    if not resolved.is_relative_to(parent_resolved):
        raise ValueError(f"fresh directory escapes managed parent: {child}")
    return resolved


def copy_verified_file_nofollow(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    artifact_label: str,
    cancellation_check: Callable[[], None] | None = None,
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise RuntimeError(
            "secure artifact materialization requires O_NOFOLLOW/O_NONBLOCK"
        )
    if cancellation_check is not None:
        cancellation_check()
    source_flags = os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)
    try:
        source_fd = os.open(source, source_flags)
    except OSError as exc:
        raise ValueError(
            f"prefetched artifact cannot be opened safely: {source}"
        ) from exc

    temporary_path: Path | None = None
    temporary_fd = -1
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"prefetched artifact is not a regular file: {source}")
        if source_stat.st_size != expected_size:
            raise ValueError(f"prefetched artifact size mismatch for {artifact_label}")
        if os.path.lexists(destination):
            raise ValueError(
                f"fresh artifact destination already exists: {destination}"
            )

        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        source_handle = os.fdopen(source_fd, "rb", closefd=True)
        source_fd = -1
        try:
            destination_handle = os.fdopen(temporary_fd, "wb", closefd=True)
        except BaseException:
            source_handle.close()
            raise
        temporary_fd = -1
        with source_handle, destination_handle:
            while True:
                if cancellation_check is not None:
                    cancellation_check()
                chunk = source_handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > expected_size:
                    raise ValueError(
                        f"prefetched artifact size mismatch for {artifact_label}"
                    )
                digest.update(chunk)
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())

        if cancellation_check is not None:
            cancellation_check()
        if size != expected_size:
            raise ValueError(f"prefetched artifact size mismatch for {artifact_label}")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"prefetched artifact digest mismatch for {artifact_label}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        fsync_directory(destination.parent)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def fsync_directory(path: Path) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("secure directory fsync requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(path, os.O_RDONLY | nofollow | directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validated_output_file(root: Path, candidate: Path) -> Path:
    if candidate.is_symlink():
        raise ValueError(f"output must not be a symlink: {candidate}")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"output escapes managed root: {candidate}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"output is not a regular file: {candidate}")
    return resolved


def bounded_log_tail(path: Path, max_bytes: int = 16 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read(max_bytes).decode("utf-8", errors="replace").strip()
