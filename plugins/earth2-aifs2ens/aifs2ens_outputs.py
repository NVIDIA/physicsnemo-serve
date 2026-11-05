# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the complete single-task ensemble before registering its artifact."""

import hashlib

import numpy as np


def field_checksum(values):
    """Hash float32 values with canonical NaN payloads and positive zero."""
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise ValueError("ensemble field checksum requires float32 values")
    canonical = np.array(values, copy=True, order="C")
    # Zarr may elide all-fill chunks and reconstruct their NaNs with a
    # canonical payload. NaN positions and infinity signs remain meaningful.
    canonical[np.isnan(canonical)] = np.float32(np.nan)
    canonical[canonical == 0] = np.float32(0)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_ensemble_output(path, loaded, *, expected_hashes=None):
    import xarray as xr

    request, contract = loaded.request, loaded.contract
    expected = {
        "ensemble": np.arange(request.nensemble),
        "time": np.array(
            [request.start_time.removesuffix("Z")], dtype="datetime64[ns]"
        ),
        "lead_time": np.arange(1, request.nsteps + 1) * np.timedelta64(6, "h"),
        "lat": np.array(contract.lat),
        "lon": np.array(contract.lon),
    }
    dims = tuple(expected)
    if expected_hashes is not None:
        keys = {
            (member, step + 1, name)
            for member in range(request.nensemble)
            for step in range(request.nsteps)
            for name in request.output_variables
        }
        if set(expected_hashes) != keys:
            raise ValueError("ensemble field checksum coverage differs from request")
    with xr.open_zarr(path, consolidated=False) as dataset:
        if set(dataset.data_vars) != set(request.output_variables):
            raise ValueError("ensemble output variables differ from request")
        for name, values in expected.items():
            if name not in dataset.coords or not np.array_equal(
                dataset[name].values, values
            ):
                raise ValueError(f"ensemble {name} coordinates differ from request")
        for name in request.output_variables:
            if dataset[name].dims != dims or dataset[name].dtype != np.float32:
                raise ValueError(
                    "ensemble field dimensions or dtype differ from request"
                )
            for member in range(request.nensemble):
                for step in range(request.nsteps):
                    try:
                        # Read one spatial field at a time. Metadata alone does
                        # not prove chunks are readable or contain written data.
                        values = np.ascontiguousarray(
                            dataset[name]
                            .isel(ensemble=member, time=0, lead_time=step)
                            .values
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"ensemble field data cannot be read: {name}, "
                            f"member {member}, step {step + 1}"
                        ) from exc
                    if expected_hashes is not None:
                        digest = field_checksum(values)
                        if digest != expected_hashes[(member, step + 1, name)]:
                            raise ValueError(
                                f"ensemble field checksum differs: {name}, "
                                f"member {member}, step {step + 1}"
                            )
