# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical parity comparison for truth manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest_builder import canonical_json_bytes

SEMANTIC_KEYS = (
    "schema_version",
    "case_id",
    "zarr_info",
    "attrs_canonical_sha256",
    "arrays",
    "coords",
    "dataset_sha256",
)


def _sorted_named_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: str(item.get("name", "")))


def semantic_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only fields relevant for read/decompressed logical parity."""
    projected = {key: manifest.get(key) for key in SEMANTIC_KEYS}
    projected["arrays"] = _sorted_named_items(list(projected.get("arrays") or []))
    projected["coords"] = _sorted_named_items(list(projected.get("coords") or []))
    return projected


def _diff(expected: Any, candidate: Any, path: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(candidate, dict):
        diffs: list[str] = []
        all_keys = sorted(set(expected) | set(candidate))
        for key in all_keys:
            key_path = f"{path}.{key}" if path else key
            if key not in expected:
                diffs.append(f"{key_path}: unexpected key in candidate")
                continue
            if key not in candidate:
                diffs.append(f"{key_path}: missing key in candidate")
                continue
            diffs.extend(_diff(expected[key], candidate[key], key_path))
        return diffs
    if isinstance(expected, list) and isinstance(candidate, list):
        if len(expected) != len(candidate):
            return [
                f"{path}: list length differs expected={len(expected)} candidate={len(candidate)}"
            ]
        diffs: list[str] = []
        for index, (left, right) in enumerate(zip(expected, candidate, strict=True)):
            diffs.extend(_diff(left, right, f"{path}[{index}]"))
        return diffs
    if expected != candidate:
        return [f"{path}: expected={expected!r} candidate={candidate!r}"]
    return []


def compare_semantic_manifests(
    expected: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    """Return semantic mismatch descriptions; empty list means parity."""
    return _diff(semantic_projection(expected), semantic_projection(candidate), "")


def assert_semantic_manifest_equal(
    expected: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """Raise AssertionError if semantic parity check fails."""
    diffs = compare_semantic_manifests(expected, candidate)
    if diffs:
        preview = "\n".join(f"- {item}" for item in diffs[:20])
        raise AssertionError(
            f"manifest semantic parity failed ({len(diffs)} diffs)\n{preview}"
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest from .json or .json.zst path."""
    file_path = Path(path)
    if file_path.suffix == ".zst":
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("zstandard is required to read .zst manifests") from exc
        compressed = file_path.read_bytes()
        payload = zstandard.ZstdDecompressor().decompress(compressed)
        return json.loads(payload.decode("utf-8"))
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write manifest as deterministic JSON, optionally compressed when suffix is .zst."""
    file_path = Path(path)
    payload = canonical_json_bytes(manifest)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix == ".zst":
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("zstandard is required to write .zst manifests") from exc
        compressed = zstandard.ZstdCompressor(level=19).compress(payload)
        file_path.write_bytes(compressed)
        return
    file_path.write_text(payload.decode("utf-8"), encoding="utf-8")
