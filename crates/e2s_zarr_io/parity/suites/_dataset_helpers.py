# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for creating tiny Zarr datasets in parity tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_test_zarr_dataset(
    *,
    dataset_path: Path,
    arrays: dict[str, Any],
    coords: dict[str, Any],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Create dataset with arrays and coords using zarr package."""
    import zarr  # type: ignore[import-not-found]

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(dataset_path), mode="w")

    def _write_array(name: str, values: Any) -> None:
        try:
            array = group.create_array(
                name=name, shape=values.shape, dtype=values.dtype
            )
            array[...] = values
            return
        except Exception:
            pass
        try:
            group.array(name=name, data=values)
            return
        except Exception:
            pass
        group.create_dataset(
            name=name, shape=values.shape, dtype=values.dtype, data=values
        )

    for name, values in arrays.items():
        _write_array(name, values)
    for name, values in coords.items():
        _write_array(name, values)

    for key, value in (attrs or {}).items():
        group.attrs[key] = value
