# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IFS preparation for the pinned Earth2Studio latitude/longitude interface."""

import os
from pathlib import Path

import numpy as np

from aifs2ens_artifacts import (
    EARTH2STUDIO_REVISION,
    INVARIANTS,
    WeatherInput,
    sha256_file,
)

PREPROCESSING_POLICY = "earth2studio-latlon-frozen-invariants-v1"


class WeatherSource:
    def __init__(self, fetch=None):
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(request, variables):
        from aifs2ens_runtime import _verify_runtime

        _verify_runtime()
        from earth2studio.data import IFS, IFS_ENS, fetch_data
        from plugin_sdk import cleanup_earth2_runtime_resources

        mirror = os.environ.get("AIFS2ENS_SOURCE", "ecmwf")
        if mirror not in {"ecmwf", "aws", "google", "azure"}:
            raise ValueError("unsupported AIFS2ENS_SOURCE mirror")
        source = (
            IFS(source=mirror, verbose=False)
            if request.ifs_member_id is None
            else IFS_ENS(source=mirror, member=request.ifs_member_id, verbose=False)
        )
        try:
            values, coords = fetch_data(
                source=source,
                time=np.array(
                    [request.start_time.removesuffix("Z")], dtype="datetime64[ns]"
                ),
                variable=np.array(variables),
                lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
                device="cpu",
            )
            return values.detach().cpu().numpy(), coords
        finally:
            cleanup_earth2_runtime_resources(source)

    def __call__(self, request, contract):
        configured = os.environ.get("AIFS2ENS_INVARIANTS")
        if not configured or contract.invariants_sha256 is None:
            raise ValueError("qualified invariants path and digest are required")
        path = Path(configured)
        digest = sha256_file(path)
        if digest != contract.invariants_sha256:
            raise ValueError("invariant artifact digest differs from contract")
        fixed = np.load(path, allow_pickle=False)
        height, width = len(contract.lat), len(contract.lon)
        if fixed.shape != (5, height, width) or fixed.dtype != np.dtype("<f4"):
            raise ValueError("invariant shape/dtype differs from model contract")
        variables = [v for v in contract.input_variables if v not in INVARIANTS]
        values, coords = self.fetch(request, variables)
        expected = contract.coords(request)
        expected["variable"] = np.array(variables)
        for name, coordinate in expected.items():
            if name not in coords or not np.array_equal(coords[name], coordinate):
                raise ValueError(
                    f"weather {name} coordinates differ from model contract"
                )
        if values.shape != (
            1,
            2,
            len(variables),
            height,
            width,
        ) or values.dtype != np.dtype("<f4"):
            raise ValueError("weather input shape/dtype differs from model contract")
        fields = {name: values[:, :, i : i + 1] for i, name in enumerate(variables)}
        for index, name in enumerate(INVARIANTS):
            fields[name] = np.broadcast_to(
                fixed[index][None, None, None], (1, 2, 1, height, width)
            )
        combined = np.concatenate([fields[v] for v in contract.input_variables], axis=2)
        return WeatherInput(
            combined,
            {
                "earth2studio_revision": EARTH2STUDIO_REVISION,
                "initialization": request.start_time,
                "ifs_member_id": request.ifs_member_id,
                "mirror": os.environ.get("AIFS2ENS_SOURCE", "ecmwf"),
                "invariants_sha256": digest,
                "preprocessing_policy": PREPROCESSING_POLICY,
            },
        )
