# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A single final ensemble requires exact field, member and coordinate coverage."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
import aifs2ens_outputs as outputs  # noqa: E402


def dataset():
    return xr.Dataset(
        {
            "t2m": (
                ("ensemble", "time", "lead_time", "lat", "lon"),
                np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2),
            )
        },
        coords={
            "ensemble": [0, 1],
            "time": np.array(["2026-11-04T12:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array([6, 12], dtype="timedelta64[h]"),
            "lat": [90.0, -90.0],
            "lon": [0.0, 180.0],
        },
    )


def loaded():
    return SimpleNamespace(
        request=SimpleNamespace(
            nensemble=2,
            nsteps=2,
            start_time="2026-11-04T12:00:00Z",
            output_variables=["t2m"],
        ),
        contract=SimpleNamespace(lat=[90.0, -90.0], lon=[0.0, 180.0]),
    )


def test_complete_single_dataset_preserves_members_and_forecast_fields(tmp_path):
    path = tmp_path / "ensemble.zarr"
    ds = dataset()
    ds.to_zarr(path, zarr_format=3, consolidated=False)
    outputs.validate_ensemble_output(path, loaded())
    with xr.open_zarr(path, consolidated=False) as actual:
        np.testing.assert_array_equal(actual.t2m.values, ds.t2m.values)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_member",
        "reordered_members",
        "missing_step",
        "initial_step",
        "variable",
        "dtype",
        "dims",
        "lat",
        "time",
    ],
)
def test_invalid_final_dataset_cannot_be_registered(tmp_path, failure):
    ds = dataset()
    if failure == "missing_member":
        ds = ds.isel(ensemble=[0])
    elif failure == "reordered_members":
        ds = ds.isel(ensemble=[1, 0])
    elif failure == "missing_step":
        ds = ds.isel(lead_time=[0])
    elif failure == "initial_step":
        ds = ds.assign_coords(lead_time=np.array([0, 6], dtype="timedelta64[h]"))
    elif failure == "variable":
        ds = ds.rename(t2m="msl")
    elif failure == "dtype":
        ds["t2m"] = ds.t2m.astype(np.float64)
    elif failure == "dims":
        ds["t2m"] = ds.t2m.transpose("time", "ensemble", "lead_time", "lat", "lon")
    elif failure == "lat":
        ds = ds.assign_coords(lat=[80.0, -80.0])
    else:
        ds = ds.assign_coords(
            time=np.array(["2026-11-04T06:00:00"], dtype="datetime64[ns]")
        )
    path = tmp_path / "invalid.zarr"
    ds.to_zarr(path, zarr_format=3, consolidated=False)
    with pytest.raises(ValueError, match="coordinate|variables|dimensions|dtype"):
        outputs.validate_ensemble_output(path, loaded())


def field_hashes(ds):
    return {
        (member, step + 1, "t2m"): outputs.field_checksum(
            ds.t2m.values[member, 0, step]
        )
        for member in range(2)
        for step in range(2)
    }


def test_corrupt_forecast_chunk_is_read_before_publication(tmp_path):
    path = tmp_path / "corrupt.zarr"
    dataset().to_zarr(path, zarr_format=3, consolidated=False)
    chunk = next(p for p in (path / "t2m" / "c").rglob("*") if p.is_file())
    chunk.write_bytes(b"corrupt compressed forecast bytes")
    with pytest.raises(ValueError, match="read"):
        outputs.validate_ensemble_output(path, loaded())


@pytest.mark.parametrize(
    "failure", ["deleted_chunk", "miswritten_values", "missing_hash"]
)
def test_stored_values_must_match_successful_writer_hashes(tmp_path, failure):
    import zarr

    path = tmp_path / "miswritten.zarr"
    ds = dataset()
    hashes = field_hashes(ds)
    ds.to_zarr(
        path,
        zarr_format=3,
        consolidated=False,
        encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
    )
    if failure == "deleted_chunk":
        chunk = next(p for p in (path / "t2m" / "c").rglob("*") if p.is_file())
        chunk.unlink()
    elif failure == "miswritten_values":
        array = zarr.open_group(path, mode="a")["t2m"]
        array[1, 0, 1] = np.zeros((2, 2), dtype=np.float32)
    else:
        hashes.pop((1, 2, "t2m"))
    with pytest.raises(ValueError, match="checksum"):
        outputs.validate_ensemble_output(path, loaded(), expected_hashes=hashes)


def test_sparse_all_fill_forecasts_can_match_written_hashes_without_chunks(tmp_path):
    path = tmp_path / "all-fill.zarr"
    ds = dataset()
    ds["t2m"] = xr.full_like(ds.t2m, np.float32(np.nan))
    ds.to_zarr(
        path,
        zarr_format=3,
        consolidated=False,
        write_empty_chunks=False,
        encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
    )
    assert not any(p.is_file() for p in (path / "t2m" / "c").rglob("*"))
    outputs.validate_ensemble_output(path, loaded(), expected_hashes=field_hashes(ds))


def test_noncanonical_nan_fill_chunks_match_values_after_elision(tmp_path):
    path = tmp_path / "nan-payload-fill.zarr"
    ds = dataset()
    ds["t2m"].data = np.full(ds.t2m.shape, 0xFFC00001, dtype=np.uint32).view(np.float32)
    hashes = field_hashes(ds)
    ds.to_zarr(
        path,
        zarr_format=3,
        consolidated=False,
        write_empty_chunks=False,
        encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
    )
    assert not any(p.is_file() for p in (path / "t2m" / "c").rglob("*"))
    outputs.validate_ensemble_output(path, loaded(), expected_hashes=hashes)


@pytest.mark.parametrize(
    "left_bits,right_bits,equal",
    [
        ([0xFFC00001], [0x7FC00000], True),
        ([0x80000000, 0], [0, 0], True),
        ([0x7F800000], [0xFF800000], False),
        ([0x3F800000], [0x3F800001], False),
        ([0xFFC00001, 0], [0, 0xFFC00001], False),
    ],
)
def test_field_checksum_normalizes_only_nan_payloads_and_signed_zero(
    left_bits, right_bits, equal
):
    left = np.array(left_bits, dtype=np.uint32).view(np.float32)
    right = np.array(right_bits, dtype=np.uint32).view(np.float32)
    original = left.view(np.uint32).copy()
    assert (outputs.field_checksum(left) == outputs.field_checksum(right)) is equal
    np.testing.assert_array_equal(left.view(np.uint32), original)
