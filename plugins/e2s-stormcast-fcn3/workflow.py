# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

from e2s_workflow import Earth2Workflow
from plugin_sdk import _close_cached_fsspec_sessions, cleanup_earth2_runtime_resources

if TYPE_CHECKING:
    from earth2studio.io import IOBackend


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.px import FCN3, InterpModAFNO, StormCast

    package = FCN3.load_default_package()
    FCN3.load_model(package)
    InterpModAFNO.from_pretrained()
    StormCast.from_pretrained()
    return {"model_names": ["FCN3", "StormCast"]}


class StormCastFCN3Workflow(Earth2Workflow):
    name = "stormcast_fcn3_workflow"
    description = "StormCast + FCN3 workflow"
    cache_scope = "process"
    model_cache_names = ["FCN3", "StormCast"]
    cache_preserve_attributes = (
        "fcn3_package",
        "fcn3_interp",
        "stormcast",
        "gfs_ic",
        "hrrr_ic",
    )

    def __init__(
        self,
        fcn3_result_storage: Literal["memory", "file"] = "memory",
        device: str = "cuda",
    ):
        super().__init__()

        self.fcn3_result_storage = fcn3_result_storage
        self.device = device
        self.fcn3_package: Any = None
        self.fcn3_interp: Any = None
        self.stormcast: Any = None
        self.gfs_ic: Any = None
        self.hrrr_ic: Any = None

    def _ensure_runtime_loaded(self) -> None:
        import torch
        import xarray as xr
        from earth2studio.data import GFS, HRRR
        from earth2studio.models.dx import DerivedSurfacePressure
        from earth2studio.models.px import (
            FCN3,
            DiagnosticWrapper,
            InterpModAFNO,
            StormCast,
        )

        if self.fcn3_interp is None:
            self.fcn3_package = FCN3.load_default_package()
            fcn3 = FCN3.load_model(self.fcn3_package)
            _close_cached_fsspec_sessions()

            orography_fn = self.fcn3_package.resolve("orography.nc")
            with xr.open_dataset(orography_fn) as ds:
                z_surface = torch.as_tensor(ds["Z"][0].values)
            z_surf_coords = OrderedDict(
                {
                    dimension: fcn3.input_coords()[dimension]
                    for dimension in ["lat", "lon"]
                }
            )
            sp_model = DerivedSurfacePressure(
                p_levels=[
                    50,
                    100,
                    150,
                    200,
                    250,
                    300,
                    400,
                    500,
                    600,
                    700,
                    850,
                    925,
                    1000,
                ],
                surface_geopotential=z_surface,
                surface_geopotential_coords=z_surf_coords,
            )

            fcn3_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)
            self.fcn3_interp = InterpModAFNO.from_pretrained()
            self.fcn3_interp.px_model = fcn3_sp
            self.fcn3_interp.to(device=self.device)

        if self.gfs_ic is None:
            self.gfs_ic = GFS()
        if self.stormcast is None:
            self.stormcast = StormCast.from_pretrained()
        if self.hrrr_ic is None:
            self.hrrr_ic = HRRR()

    def warmup(self, _ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded()
        return {"model_names": ["FCN3", "StormCast"]}

    def __call__(
        self,
        io: IOBackend,
        start_time: datetime = datetime(2024, 1, 1, 0),
        num_hours: int = 10,
        run_stormcast: bool = True,
    ) -> None:
        from earth2studio import run
        from earth2studio.data import InferenceOutputSource
        from earth2studio.io import NetCDF4Backend, XarrayBackend

        fcn3_results: Any = io
        tmp_dir: TemporaryDirectory[str] | None = None
        tmp_file: str | None = None

        try:
            self._ensure_runtime_loaded()
            if not run_stormcast:
                fcn3_results = io
            elif self.fcn3_result_storage == "memory":
                fcn3_results = XarrayBackend()
            else:
                tmp_dir = TemporaryDirectory()
                tmp_file = os.path.join(tmp_dir.name, "fcn3_output.nc")
                fcn3_results = NetCDF4Backend(  # type: ignore[assignment]
                    tmp_file, backend_kwargs={"mode": "w", "diskless": False}
                )

            # run surface-pressure interpolated FCN3
            run.deterministic(
                [start_time],
                num_hours,
                self.fcn3_interp,
                self.gfs_ic,
                fcn3_results,
                device=self.device,
            )

            if not run_stormcast:
                return

            if self.fcn3_result_storage == "memory":
                # XarrayBackend has a root attribute, but IOBackend doesn't.
                source = InferenceOutputSource(fcn3_results.root)  # type: ignore[attr-defined]
            else:
                source = InferenceOutputSource(tmp_file)
            self.stormcast.conditioning_data_source = source

            run.deterministic(
                [start_time],
                num_hours,
                self.stormcast,
                self.hrrr_ic,
                io,
                device=self.device,
            )
        finally:
            if self.stormcast is not None:
                self.stormcast.conditioning_data_source = None

            if fcn3_results is not io:
                close = getattr(fcn3_results, "close", None)
                if callable(close):
                    close()

            if tmp_dir is not None:
                tmp_dir.cleanup()

    def cleanup(self) -> None:
        self.fcn3_package = None
        try:
            super().cleanup()
        finally:
            if self.stormcast is not None:
                self.stormcast.conditioning_data_source = None
            cleanup_earth2_runtime_resources(
                self.fcn3_interp,
                self.stormcast,
            )
            self._clear_attributes(
                "fcn3_package",
                "fcn3_interp",
                "stormcast",
                "gfs_ic",
                "hrrr_ic",
            )
            self._cleanup_torch_runtime(device=self.device)

    def cleanup_request(self) -> None:
        if self.stormcast is not None:
            self.stormcast.conditioning_data_source = None
        super().cleanup_request()


WORKFLOW = StormCastFCN3Workflow
