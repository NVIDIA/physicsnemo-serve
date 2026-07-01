# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical read/decompress utilities for Zarr datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest_builder import build_truth_manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_zarr_group(dataset_path: Path) -> Any:
    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "zarr package is required for canonical dataset reading"
        ) from exc
    return zarr.open_group(str(dataset_path), mode="r")


def _infer_coord_names(group: Any, output_array_names: set[str]) -> list[str]:
    inferred: list[str] = []
    member_names = sorted(getattr(group, "array_keys", lambda: [])())
    for name in member_names:
        if name in output_array_names:
            continue
        try:
            array = group[name]
            ndim = int(getattr(array, "ndim", -1))
        except Exception:
            continue
        if ndim == 1:
            inferred.append(name)
    return inferred


def _normalize_zarr_info(
    case_spec: dict[str, Any], dataset_path: Path
) -> dict[str, Any]:
    zarr_format = case_spec.get("zarr_format")
    if zarr_format not in {"v2", "v3"}:
        zarr_format = "v3" if (dataset_path / "zarr.json").exists() else "v2"
    chunk_key_encoding = case_spec.get("chunk_key_encoding")
    chunk_key_separator = case_spec.get("chunk_key_separator")
    return {
        "zarr_format": zarr_format,
        "chunk_key_encoding": chunk_key_encoding,
        "chunk_key_separator": chunk_key_separator,
        "store_kind": "local_fs",
    }


def _normalize_coord_values(name: str, values: Any) -> Any:
    """Normalize coord dtypes to backend-agnostic semantic representation.

    Dispatches on dtype kind rather than coordinate name so that any
    datetime64 or timedelta64 axis is normalized to int64[ns], matching the
    Rust backend's serialization behaviour.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return values
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]").view("int64")
    if np.issubdtype(array.dtype, np.timedelta64):
        return array.astype("timedelta64[ns]").view("int64")
    return values


def read_dataset_content(
    dataset_path: str | Path, case_spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read arrays/coords/attrs/zarr_info from dataset for manifest generation."""
    path = Path(dataset_path)
    _require(path.exists(), f"dataset path does not exist: {path}")
    group = _load_zarr_group(path)
    output_array_names = case_spec.get("output_array_names")
    _require(
        isinstance(output_array_names, list) and output_array_names,
        "case_spec.output_array_names must be a non-empty list",
    )
    _require(
        all(isinstance(name, str) and name for name in output_array_names),
        "case_spec.output_array_names must contain non-empty string names",
    )
    arrays: dict[str, Any] = {}
    for name in output_array_names:
        try:
            arrays[name] = group[name][...]
        except Exception as exc:
            raise RuntimeError(
                f"failed to read output array '{name}' from dataset"
            ) from exc

    case_coord_names = case_spec.get("coord_names")
    if isinstance(case_coord_names, list) and case_coord_names:
        coord_names = [str(name) for name in case_coord_names]
    else:
        coord_names = _infer_coord_names(group, set(output_array_names))
    coords: dict[str, Any] = {}
    for name in coord_names:
        try:
            coords[name] = _normalize_coord_values(name, group[name][...])
        except Exception:
            continue

    attrs_obj = getattr(group, "attrs", {})
    attrs = attrs_obj.asdict() if hasattr(attrs_obj, "asdict") else dict(attrs_obj)
    zarr_info = _normalize_zarr_info(case_spec, path)
    return arrays, coords, attrs, zarr_info


def build_manifest_from_dataset(
    *,
    dataset_path: str | Path,
    case_spec: dict[str, Any],
    generated_by_backend: str,
    truth_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical truth/candidate manifest from a dataset path."""
    arrays, coords, attrs, zarr_info = read_dataset_content(dataset_path, case_spec)
    case_id = case_spec.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_spec.case_id must be a non-empty string")
    return build_truth_manifest(
        case_id=case_id,
        generated_by_backend=generated_by_backend,
        zarr_info=zarr_info,
        arrays=arrays,
        coords=coords,
        attrs=attrs,
        truth_provenance=truth_provenance,
    )
