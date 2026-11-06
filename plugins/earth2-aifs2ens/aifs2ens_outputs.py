# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write and validate the complete ensemble before registering its artifact."""

from collections import OrderedDict
import hashlib

import numpy as np


def _forecast_coords(loaded):
    request, contract = loaded.request, loaded.contract
    return OrderedDict(
        ensemble=np.arange(request.nensemble),
        time=np.array([request.start_time.removesuffix("Z")], dtype="datetime64[ns]"),
        lead_time=np.arange(1, request.nsteps + 1) * np.timedelta64(6, "h"),
        lat=np.array(contract.lat),
        lon=np.array(contract.lon),
    )


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

    request = loaded.request
    expected = _forecast_coords(loaded)
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


class ZarrEnsembleSink:
    def __init__(self, path, loaded):
        from e2s_workflow import _selected_zarr_backend, create_zarr_backend

        request, contract = loaded.request, loaded.contract
        self.variables = np.array(request.output_variables)
        self.coords = _forecast_coords(loaded)
        self._written = np.zeros((request.nensemble, request.nsteps), dtype=bool)
        self.field_hashes = {}
        self._registered = False
        self.closed = False
        self.python_backend = _selected_zarr_backend() == "python"
        options = (
            {}
            if self.python_backend
            else {
                "zarr_format": "v3",
                "max_pool_bytes": 256 * 1024**2,
                # Native writes reject excessive transient leases rather than
                # waiting for slow flushes. Bound them by the accepted forecast.
                "max_inflight_transient_bytes": 256 * 1024**2
                + request.nensemble
                * request.nsteps
                * len(self.variables)
                * len(contract.lat)
                * len(contract.lon)
                * np.dtype(np.float32).itemsize,
            }
        )
        chunks = {"ensemble": 1, "time": 1, "lead_time": 1}
        if self.python_backend:
            chunks.update(lat=180, lon=360)
        self.backend = create_zarr_backend(
            str(path), chunks=chunks, backend_kwargs={"overwrite": False}, **options
        )

    def add_array(self, coords, array_name):
        if self.closed or self._registered:
            raise ValueError("forecast arrays are already registered or closed")
        if not np.array_equal(np.asarray(array_name), self.variables):
            raise ValueError("forecast output variables differ from request")
        expected = OrderedDict(self.coords)
        expected["lead_time"] = np.arange(self._written.shape[1] + 1) * np.timedelta64(
            6, "h"
        )
        for name, values in expected.items():
            if name not in coords or not np.array_equal(coords[name], values):
                raise ValueError(f"forecast {name} coordinates differ from request")
        self.backend.add_array(self.coords, self.variables)
        self._registered = True

    def write(self, data, coords, array_name):
        if self.closed or not self._registered:
            raise RuntimeError("forecast writer is closed or unregistered")
        if not np.array_equal(np.asarray(array_name), self.variables):
            raise ValueError("forecast output variables differ from request")
        for name in ("time", "lat", "lon"):
            if name not in coords or not np.array_equal(
                coords[name], self.coords[name]
            ):
                raise ValueError(f"forecast {name} coordinates differ from request")
        members = np.asarray(coords.get("ensemble", []))
        if (
            members.ndim != 1
            or not members.size
            or members.dtype.kind not in "iu"
            or len(np.unique(members)) != len(members)
            or np.any(members < 0)
            or np.any(members >= self._written.shape[0])
        ):
            raise ValueError("forecast ensemble coordinates differ from request")
        lead = np.asarray(coords.get("lead_time", []))
        if lead.shape != (1,):
            raise ValueError("forecast lead_time coordinates must identify one step")
        if lead[0] == np.timedelta64(0, "h"):
            return  # Stock ensemble() emits initialization before forecasts.
        step = np.flatnonzero(self.coords["lead_time"] == lead[0])
        if len(step) != 1:
            raise ValueError("forecast lead_time coordinates differ from request")
        step_index = int(step[0])
        if np.any(self._written[members, step_index]):
            raise ValueError("duplicate forecast member/step write")
        values = list(data) if isinstance(data, (list, tuple)) else [data]
        if len(values) != len(self.variables):
            raise ValueError("forecast field count differs from request")
        arrays = []
        expected_shape = (
            len(members),
            1,
            1,
            len(self.coords["lat"]),
            len(self.coords["lon"]),
        )
        hashes = {}
        for name, value in zip(self.variables, values):
            array = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )
            if array.dtype != np.float32:
                raise ValueError("forecast fields must be float32")
            if array.shape != expected_shape:
                raise ValueError("forecast field shape differs from request")
            arrays.append(np.ascontiguousarray(array))
            for offset, member in enumerate(members):
                hashes[(int(member), step_index + 1, str(name))] = field_checksum(
                    array[offset, 0, 0]
                )
        if self.python_backend:
            import torch

            arrays = [torch.from_numpy(array) for array in arrays]
        self.backend.write(arrays, coords, self.variables)
        self.field_hashes.update(hashes)
        self._written[members, step_index] = True

    def assert_complete(self):
        if not self._registered or not self._written.all():
            raise ValueError("forecast output coverage is incomplete")

    def close(self):
        if self.closed:
            return
        self.closed = True
        finalize = getattr(self.backend, "finalize", None)
        close = getattr(self.backend, "close", None)
        if callable(finalize):
            finalize()
        elif callable(close):
            close()
