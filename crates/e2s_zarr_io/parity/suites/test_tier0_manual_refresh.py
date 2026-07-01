# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier0 manual refresh tests."""

from __future__ import annotations

import pytest

from parity.tools.record_truth import build_blessed_truth_manifest
from parity.utils.manifest_builder import build_truth_manifest


class FakeArray:
    """Simple array-like object for deterministic payloads."""

    def __init__(self, payload: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self._payload = payload
        self.shape = shape
        self.dtype = dtype

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise ValueError("only C-order is supported in this fake")
        return self._payload


def _manifest(backend: str, payload: bytes) -> dict[str, object]:
    return build_truth_manifest(
        case_id="tier0_case",
        generated_by_backend=backend,
        zarr_info={
            "zarr_format": "v2",
            "chunk_key_encoding": "v2",
            "chunk_key_separator": ".",
            "store_kind": "local_fs",
        },
        arrays={"temperature": FakeArray(payload, (2, 2), "float32")},
        coords={"time": FakeArray(b"\x01\x02", (2,), "int64")},
        attrs={"foo": "bar"},
    )


def test_manual_refresh_blesses_when_py_baseline_matches() -> None:
    py_sync = _manifest("py_sync", b"\x10\x11\x12\x13")
    py_async = _manifest("py_async", b"\x10\x11\x12\x13")
    blessed = build_blessed_truth_manifest(
        py_sync_manifest=py_sync,
        py_async_manifest=py_async,
        earth2studio_commit="abc1234",
        python_version="3.12.4",
        zarr_python_version="2.18.3",
        case_spec_sha256="0" * 64,
    )
    assert blessed["generated_by_backend"] == "py_sync"
    provenance = blessed["truth_provenance"]
    assert provenance["generation_mode"] == "manual_tier0"
    assert provenance["earth2studio_commit"] == "abc1234"
    assert "generated_at_utc" in provenance


def test_manual_refresh_rejects_py_baseline_mismatch() -> None:
    py_sync = _manifest("py_sync", b"\x10\x11\x12\x13")
    py_async = _manifest("py_async", b"\x99\x11\x12\x13")
    with pytest.raises(AssertionError, match="manifest semantic parity failed"):
        build_blessed_truth_manifest(
            py_sync_manifest=py_sync,
            py_async_manifest=py_async,
            earth2studio_commit="abc1234",
            python_version="3.12.4",
            zarr_python_version="2.18.3",
            case_spec_sha256="0" * 64,
        )
