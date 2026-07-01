# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest builder tests."""

from __future__ import annotations

import pytest

from parity.utils.manifest_builder import build_truth_manifest


class FakeArray:
    """Simple array-like object for deterministic test payloads."""

    def __init__(self, payload: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self._payload = payload
        self.shape = shape
        self.dtype = dtype

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise ValueError("only C-order is supported in this fake")
        return self._payload


def _build_input_arrays() -> dict[str, FakeArray]:
    return {
        "temperature": FakeArray(
            payload=b"\x01\x02\x03\x04", shape=(2, 2), dtype="float32"
        ),
        "pressure": FakeArray(
            payload=b"\x05\x06\x07\x08", shape=(2, 2), dtype="float32"
        ),
    }


def _build_input_coords() -> dict[str, FakeArray]:
    return {
        "time": FakeArray(payload=b"\x09\x0a", shape=(2,), dtype="int64"),
    }


def test_manifest_builder_is_deterministic_for_same_inputs() -> None:
    arrays = _build_input_arrays()
    coords = _build_input_coords()
    zarr_info = {
        "zarr_format": "v2",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
        "store_kind": "local_fs",
    }
    manifest_a = build_truth_manifest(
        case_id="case_one",
        generated_by_backend="py_sync",
        zarr_info=zarr_info,
        arrays=arrays,
        coords=coords,
        attrs={"source": "test"},
    )
    manifest_b = build_truth_manifest(
        case_id="case_one",
        generated_by_backend="rust",
        zarr_info=zarr_info,
        arrays=arrays,
        coords=coords,
        attrs={"source": "test"},
    )
    assert manifest_a["dataset_sha256"] == manifest_b["dataset_sha256"]
    assert manifest_a["attrs_canonical_sha256"] == manifest_b["attrs_canonical_sha256"]


def test_manifest_builder_rejects_invalid_zarr_format() -> None:
    arrays = _build_input_arrays()
    coords = _build_input_coords()
    bad_zarr_info = {
        "zarr_format": "v4",
        "chunk_key_encoding": "v2",
        "chunk_key_separator": ".",
        "store_kind": "local_fs",
    }
    with pytest.raises(ValueError, match="zarr_info.zarr_format"):
        build_truth_manifest(
            case_id="case_bad",
            generated_by_backend="py_sync",
            zarr_info=bad_zarr_info,
            arrays=arrays,
            coords=coords,
        )
