# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-contained artifact helpers shared by PhysicsNeMo-CFD plugins."""

from __future__ import annotations

from pathlib import Path

from plugin_sdk import ExecutionContext

from .safe_files import validated_directory, validated_output_file


def collect_globbed_outputs(
    root: Path,
    pattern: str,
    *,
    allowed_suffixes: set[str],
) -> list[tuple[str, Path, str]]:
    outputs: list[tuple[str, Path, str]] = []
    for candidate in sorted(root.glob(pattern)):
        if candidate.suffix.lower() not in allowed_suffixes:
            continue
        path = validated_output_file(root, candidate)
        if path.exists():
            relative = path.relative_to(root).as_posix()
            outputs.append((relative, path, media_type_for_path(path)))
    return outputs


def media_type_for_path(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".vtp": "application/vnd.vtk.vtp",
    }.get(path.suffix.lower(), "application/octet-stream")


def validate_audit_outputs(
    ctx: ExecutionContext,
    *,
    resolved_config_path: str | Path,
    diagnostics_path: str | Path,
    log_path: str | Path,
) -> list[tuple[Path, str]]:
    managed_run_root = validated_directory(ctx.run_dir, label="run directory")
    validated: list[tuple[Path, str]] = []
    for candidate, media_type in (
        (Path(resolved_config_path), "application/json"),
        (Path(diagnostics_path), "application/json"),
        (Path(log_path), "text/plain"),
    ):
        path = validated_output_file(managed_run_root, candidate)
        if not path.exists():
            raise RuntimeError(f"required audit artifact is missing: {candidate.name}")
        validated.append((path, media_type))
    return validated


def register_artifact_once(ctx: ExecutionContext, path: Path, media_type: str) -> None:
    matches = [
        output
        for output in ctx.outputs.registered_outputs()
        if output.name == path.name
    ]
    if not matches:
        ctx.outputs.register(path.name, path, media_type=media_type)
        return
    resolved_path = path.resolve(strict=True)
    if any(
        Path(output.path).resolve(strict=False) != resolved_path
        or output.media_type != media_type
        or output.primary
        for output in matches
    ):
        raise ValueError(f"conflicting audit artifact registration: {path.name}")


def register_validated_artifacts_once(
    ctx: ExecutionContext, outputs: list[tuple[Path, str]]
) -> None:
    for path, media_type in outputs:
        register_artifact_once(ctx, path, media_type)


def register_audit_outputs(
    ctx: ExecutionContext,
    *,
    resolved_config_path: str | Path,
    diagnostics_path: str | Path,
    log_path: str | Path,
) -> list[str]:
    audit_outputs = validate_audit_outputs(
        ctx,
        resolved_config_path=resolved_config_path,
        diagnostics_path=diagnostics_path,
        log_path=log_path,
    )
    register_validated_artifacts_once(ctx, audit_outputs)
    return [path.name for path, _media_type in audit_outputs]
