# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier1 PR parity tests against committed truth semantics."""

from __future__ import annotations

import pytest

from parity.tools.verify_parity import verify_candidate_against_truth
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


def _manifest(
    backend: str, payload: bytes, zarr_format: str = "v2"
) -> dict[str, object]:
    return build_truth_manifest(
        case_id="tier1_case",
        generated_by_backend=backend,
        zarr_info={
            "zarr_format": zarr_format,
            "chunk_key_encoding": "v2" if zarr_format == "v2" else "default",
            "chunk_key_separator": "." if zarr_format == "v2" else None,
            "store_kind": "local_fs",
        },
        arrays={"temperature": FakeArray(payload, (2, 2), "float32")},
        coords={"time": FakeArray(b"\x01\x02", (2,), "int64")},
        attrs={"foo": "bar"},
    )


def test_tier1_parity_passes_for_logically_equal_data() -> None:
    truth = _manifest("py_sync", b"\x10\x11\x12\x13")
    candidate = _manifest("rust", b"\x10\x11\x12\x13")
    verify_candidate_against_truth(truth_manifest=truth, candidate_manifest=candidate)


def test_tier1_parity_fails_for_data_mismatch() -> None:
    truth = _manifest("py_sync", b"\x10\x11\x12\x13")
    candidate = _manifest("rust", b"\xff\x11\x12\x13")
    with pytest.raises(AssertionError, match="manifest semantic parity failed"):
        verify_candidate_against_truth(
            truth_manifest=truth, candidate_manifest=candidate
        )


def test_tier1_parity_fails_for_zarr_format_mismatch() -> None:
    truth = _manifest("py_sync", b"\x10\x11\x12\x13", zarr_format="v2")
    candidate = _manifest("rust", b"\x10\x11\x12\x13", zarr_format="v3")
    with pytest.raises(AssertionError, match="manifest semantic parity failed"):
        verify_candidate_against_truth(
            truth_manifest=truth, candidate_manifest=candidate
        )
