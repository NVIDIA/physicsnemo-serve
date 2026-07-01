# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest creation from read/decompressed logical dataset content."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

TRUTH_MANIFEST_SCHEMA_VERSION = "truth_manifest.v1"
GENERATED_BACKENDS = {"py_sync", "py_async", "rust"}
VOLATILE_ATTR_KEYS = frozenset(
    {"generated_at", "timestamp", "created_at", "updated_at", "run_id"}
)


def sha256_bytes(payload: bytes) -> str:
    """Return lowercase SHA-256 hex digest for bytes payload."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize value as deterministic JSON bytes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _normalize_shape(array: Any, payload: bytes) -> list[int]:
    shape = getattr(array, "shape", None)
    if shape is None:
        return [len(payload)]
    if isinstance(shape, (tuple, list)):
        return [int(dim) for dim in shape]
    return [int(shape)]


def _normalize_dtype(array: Any) -> str:
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return "bytes"
    return str(dtype)


def _to_c_order_bytes(array: Any) -> bytes:
    if isinstance(array, (bytes, bytearray, memoryview)):
        return bytes(array)
    if hasattr(array, "tobytes"):
        try:
            return bytes(array.tobytes(order="C"))
        except TypeError:
            return bytes(array.tobytes())
    raise TypeError("array value must be bytes-like or expose tobytes()")


def _normalize_attrs(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            key_str = str(key)
            if key_str.lower() in VOLATILE_ATTR_KEYS:
                continue
            normalized[key_str] = _normalize_attrs(value[key])
        return normalized
    if isinstance(value, list):
        return [_normalize_attrs(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_attrs(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_zarr_info(zarr_info: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "zarr_format": zarr_info.get("zarr_format"),
        "chunk_key_encoding": zarr_info.get("chunk_key_encoding"),
        "chunk_key_separator": zarr_info.get("chunk_key_separator"),
        "store_kind": zarr_info.get("store_kind", "local_fs"),
    }
    if normalized["zarr_format"] not in {"v2", "v3"}:
        raise ValueError("zarr_info.zarr_format must be 'v2' or 'v3'")
    return normalized


def _build_array_digest(name: str, array: Any) -> dict[str, Any]:
    payload = _to_c_order_bytes(array)
    return {
        "name": name,
        "dtype": _normalize_dtype(array),
        "shape": _normalize_shape(array, payload),
        "order": "C",
        "payload_sha256": sha256_bytes(payload),
        "nan_count": None,
        "finite_min": None,
        "finite_max": None,
    }


def _build_coord_digest(name: str, coord_values: Any) -> dict[str, Any]:
    payload = _to_c_order_bytes(coord_values)
    return {
        "name": name,
        "dtype": _normalize_dtype(coord_values),
        "shape": _normalize_shape(coord_values, payload),
        "payload_sha256": sha256_bytes(payload),
    }


def _build_dataset_hash_input(
    *,
    case_id: str,
    zarr_info: dict[str, Any],
    attrs_canonical_sha256: str,
    arrays: list[dict[str, Any]],
    coords: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "zarr_info": zarr_info,
        "attrs_canonical_sha256": attrs_canonical_sha256,
        "arrays": sorted(arrays, key=lambda item: item["name"]),
        "coords": sorted(coords, key=lambda item: item["name"]),
    }


def build_truth_manifest(
    *,
    case_id: str,
    generated_by_backend: str,
    zarr_info: Mapping[str, Any],
    arrays: Mapping[str, Any],
    coords: Mapping[str, Any],
    attrs: Mapping[str, Any] | None = None,
    truth_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a TruthManifestV1 from read/decompressed logical data."""
    if generated_by_backend not in GENERATED_BACKENDS:
        raise ValueError(
            "generated_by_backend must be one of {'py_sync', 'py_async', 'rust'}"
        )
    if not case_id:
        raise ValueError("case_id must be non-empty")

    normalized_zarr_info = _normalize_zarr_info(zarr_info)
    normalized_attrs = _normalize_attrs(attrs or {})
    attrs_canonical_sha256 = sha256_bytes(canonical_json_bytes(normalized_attrs))
    array_digests = [_build_array_digest(name, arrays[name]) for name in sorted(arrays)]
    coord_digests = [_build_coord_digest(name, coords[name]) for name in sorted(coords)]
    hash_input = _build_dataset_hash_input(
        case_id=case_id,
        zarr_info=normalized_zarr_info,
        attrs_canonical_sha256=attrs_canonical_sha256,
        arrays=array_digests,
        coords=coord_digests,
    )
    dataset_sha256 = sha256_bytes(canonical_json_bytes(hash_input))
    manifest = {
        "schema_version": TRUTH_MANIFEST_SCHEMA_VERSION,
        "case_id": case_id,
        "generated_by_backend": generated_by_backend,
        "zarr_info": normalized_zarr_info,
        "truth_provenance": deepcopy(dict(truth_provenance or {})),
        "attrs_canonical_sha256": attrs_canonical_sha256,
        "arrays": array_digests,
        "coords": coord_digests,
        "dataset_sha256": dataset_sha256,
    }
    return manifest
