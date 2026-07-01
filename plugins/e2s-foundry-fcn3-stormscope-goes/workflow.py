# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from e2s_workflow import Earth2Workflow

if TYPE_CHECKING:
    import torch
    from earth2studio.io import IOBackend
    from earth2studio.models.px import InterpModAFNO
    from earth2studio.models.px.stormscope import StormScopeGOES
    from earth2studio.utils.coords import CoordSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("foundry_fcn3_stormscope_goes_workflow")

GOES_MODEL_NAME = "6km_60min_natten_cos_zenith_input_eoe_v2"

_MAX_FORECAST_STEPS = 32
_MAX_ENSEMBLE_SAMPLES = 32


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.px import FCN3, InterpModAFNO
    from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES

    fcn3_package = FCN3.load_default_package()
    FCN3.load_model(fcn3_package)
    InterpModAFNO.from_pretrained()
    stormscope_package = StormScopeBase.load_default_package()
    StormScopeGOES.load_model(
        package=stormscope_package,
        conditioning_data_source=None,
        model_name=GOES_MODEL_NAME,
    )
    return {"model_names": ["FCN3", "StormScopeGOES"]}


@dataclass
class WorkflowProgress:
    """Small PhysicsNeMo Serve-local stand-in for Earth2Studio serve progress records."""

    progress: str
    current_step: int | None = None
    total_steps: int | None = None


class FoundryFCN3StormScopeGOESWorkflow(Earth2Workflow):
    """FCN3 plus StormScope GOES diagnostic ensemble for Foundry."""

    name = "foundry_fcn3_stormscope_goes_workflow"
    description = "FCN3+StormScopeGOES ensemble workflow for Foundry"
    cache_scope = "process"
    model_cache_names = ["FCN3", "StormScopeGOES"]
    cache_preserve_attributes = (
        "fcn3_interp",
        "stormscope",
        "data_fcn3",
        "data_stormscope",
    )

    def __init__(
        self,
        device: str = "cuda",
        init_seed: int = 1234,
    ):
        super().__init__()

        self.device = device

        # Keep model/data initialization lazy in PhysicsNeMo Serve so prepare/readiness
        # phases do not load GPU resources. The source workflow initializes
        # these objects in __init__.
        self.fcn3_interp: InterpModAFNO | None = None
        self.stormscope: StormScopeGOES | None = None
        self.rng = np.random.default_rng(init_seed)
        self.data_fcn3: Any = None
        self.data_stormscope: dict[str, Any] | None = None
        self._scan_mode = "C"

    def create_io(self, dataset_path: str):
        # The source workflow mutates io.root attrs directly. Use Earth2Studio's
        # Python Zarr backend for now because the PhysicsNeMo Serve Rust adapter does
        # not expose that mutable metadata surface.
        from earth2studio.io import ZarrBackend

        return ZarrBackend(
            dataset_path,
            backend_kwargs={"overwrite": True, "zarr_format": 3},
            chunks={"ensemble": 1, "time": 1},
        )

    def update_progress(self, progress: WorkflowProgress) -> None:
        """Log progress in PhysicsNeMo Serve where Earth2Studio serve would publish it."""
        logger.info(progress.progress)

    def _ensure_runtime_loaded(self) -> None:
        """Initialize request-time Earth2Studio objects lazily for PhysicsNeMo Serve."""
        if self.fcn3_interp is None:
            self.fcn3_interp = self.load_fcn3_interp()
        if self.stormscope is None:
            self.stormscope = self.load_stormscope()
        if self.data_fcn3 is None:
            from earth2studio.data import PlanetaryComputerECMWFOpenDataIFS

            self.data_fcn3 = PlanetaryComputerECMWFOpenDataIFS(
                verbose=False, cache=False
            )
        if self.data_stormscope is None:
            from earth2studio.data import GOES, PlanetaryComputerGOES

            self.data_stormscope = {
                satellite: PlanetaryComputerGOES(
                    satellite=satellite,
                    scan_mode=self._scan_mode,
                    verbose=False,
                    cache=False,
                )
                for satellite in ["goes16", "goes19"]
            }

            # GOES-16 and GOES-19 have the same grid.
            goes_lat, goes_lon = GOES.grid(
                satellite="goes16", scan_mode=self._scan_mode
            )
            coords_out = self.fcn3_interp.output_coords(self.fcn3_interp.input_coords())
            self.stormscope.build_input_interpolator(goes_lat, goes_lon)
            self.stormscope.build_conditioning_interpolator(
                coords_out["lat"], coords_out["lon"]
            )

    def warmup(self, _ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded()
        return {"model_names": ["FCN3", "StormScopeGOES"]}

    @staticmethod
    def _coerce_start_time(start_time: datetime | str) -> datetime:
        if isinstance(start_time, datetime):
            return start_time
        return datetime.fromisoformat(str(start_time))

    @staticmethod
    def _validate_plugin_parameters(
        n_steps: int,
        n_samples_fcn3: int,
        n_samples_stormscope: int,
        output_format: str,
        container_url: str | None,
        geo_catalog_url: str | None,
    ) -> None:
        """PhysicsNeMo Serve equivalent of the source validate_parameters limit checks."""
        if not 1 <= n_steps <= _MAX_FORECAST_STEPS:
            raise ValueError(
                f"n_steps must be between 1 and {_MAX_FORECAST_STEPS}, got {n_steps}"
            )
        if not 1 <= n_samples_fcn3 <= _MAX_ENSEMBLE_SAMPLES:
            raise ValueError(
                f"n_samples_fcn3 must be between 1 and {_MAX_ENSEMBLE_SAMPLES}, "
                f"got {n_samples_fcn3}"
            )
        if not 1 <= n_samples_stormscope <= _MAX_ENSEMBLE_SAMPLES:
            raise ValueError(
                f"n_samples_stormscope must be between 1 and {_MAX_ENSEMBLE_SAMPLES}, "
                f"got {n_samples_stormscope}"
            )
        if geo_catalog_url is not None:
            if container_url is None:
                raise ValueError(
                    "container_url is required when geo_catalog_url is set."
                )
            if output_format != "netcdf4":
                raise ValueError(
                    "output_format must be 'netcdf4' when geo_catalog_url is set."
                )

    def load_fcn3_interp(self) -> InterpModAFNO:
        """Load FCN3 with surface pressure diagnostics and hourly interpolation."""
        import torch
        import xarray as xr
        from earth2studio.models.dx import DerivedSurfacePressure
        from earth2studio.models.px import FCN3, DiagnosticWrapper, InterpModAFNO

        logger.info("Loading FCN3")
        package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(package)

        # Surface pressure interpolation
        orography_fn = package.resolve("orography.nc")
        with xr.open_dataset(orography_fn) as ds:
            z_surface = torch.as_tensor(ds["Z"][0].values)
        z_surf_coords = OrderedDict({d: fcn3.input_coords()[d] for d in ["lat", "lon"]})
        sp_model = DerivedSurfacePressure(
            p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            surface_geopotential=z_surface,
            surface_geopotential_coords=z_surf_coords,
        )

        # Bundle surface pressure with FCN3
        fcn3_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)

        # Add temporal interpolation to 1 hour
        fcn3_interp = InterpModAFNO.from_pretrained()
        fcn3_interp.px_model = fcn3_sp
        fcn3_interp.to(device=self.device)
        fcn3_interp.eval()
        return fcn3_interp

    def load_stormscope(self) -> StormScopeGOES:
        """Load the StormScope GOES model package and move it to the workflow device."""
        from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES

        logger.info("Loading StormScope")
        package = StormScopeBase.load_default_package()
        stormscope = StormScopeGOES.load_model(
            package=package,
            conditioning_data_source=None,
            model_name=GOES_MODEL_NAME,
        )
        stormscope.to(self.device)
        stormscope.eval()
        return stormscope

    def get_seeds(self, n_seeds: int) -> list[int]:
        """Sample ``n_seeds`` distinct integer RNG seeds for ensemble members."""
        seeds = self.rng.choice(2**32, size=n_seeds, replace=False)
        return [int(s) for s in seeds]

    def validate_start_times(
        self, time_stormscope: datetime, time_fcn3: datetime
    ) -> None:
        """Check StormScope and FCN3 start times and their relative ordering."""
        ref = datetime(1900, 1, 1)
        if (time_stormscope - ref).total_seconds() % (1 * 60 * 60) != 0:
            raise ValueError(
                f"Start time for StormScope must be 1-hour interval: {time_stormscope}"
            )
        if (time_fcn3 - ref).total_seconds() % (6 * 60 * 60) != 0:
            raise ValueError(
                f"Start time for FCN3 must be 6-hour interval: {time_fcn3}"
            )
        if time_stormscope < time_fcn3:
            raise ValueError(
                "Start time for StormScope cannot preceed start time for FCN3"
            )
        if time_stormscope - time_fcn3 > timedelta(hours=12):
            logger.warning(
                "Start times for StormScope and FCN3 should not be more than 12 "
                "hours apart but got '%s' and '%s'",
                time_stormscope,
                time_fcn3,
            )

    def validate_samples(
        self, n_samples: int, seeds: Sequence[int] | None
    ) -> list[int]:
        """Return ensemble seeds of length ``n_samples``, generating them if missing."""
        if not seeds:
            return self.get_seeds(n_samples)
        if len(seeds) != n_samples:
            logger.warning(
                "Ignoring requested number of samples because it does not match number of seeds"
            )
        return list(seeds)

    def validate_variables(self, variables: Sequence[str] | None) -> np.ndarray:
        """Resolve StormScope output variables, defaulting to the model's variables."""
        if self.stormscope is None:
            raise RuntimeError(
                "StormScope model must be loaded before validating variables"
            )

        if variables is None:
            variables = self.stormscope.variables
        else:
            unknown_variables = set(variables) - set(self.stormscope.variables)
            if len(unknown_variables):
                raise ValueError(f"Unknown variable(s) {', '.join(unknown_variables)}")
            variables = np.array(variables)
        return variables

    def setup_io(
        self,
        io: IOBackend,
        output_coords: CoordSystem,
        seeds_fcn3: Sequence[int],
        seeds_stormscope: Sequence[int],
    ) -> None:
        """Define IO arrays, CRS metadata, and per-model seeds for ensemble outputs."""
        import torch
        import zarr
        from earth2studio.io import NetCDF4Backend, ZarrBackend

        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"},
            output_coords["variable"],
        )

        # Storing seeds separately makes it easier to filter with Titiler
        e_coords = {"ensemble": output_coords["ensemble"]}
        n_stormscope_per_fcn3 = len(seeds_stormscope) // len(seeds_fcn3)
        tiled_seeds_fcn3 = np.repeat(seeds_fcn3, n_stormscope_per_fcn3)
        io.add_array(e_coords, "seed_fcn3", data=torch.tensor(tiled_seeds_fcn3))
        io.add_array(e_coords, "seed_stormscope", data=torch.tensor(seeds_stormscope))

        # Add CRS definition
        io.add_array({}, "crs")
        io.root["crs"].grid_mapping_name = "lambert_conformal_conic"
        io.root["crs"].standard_parallel = 38.5
        io.root["crs"].longitude_of_central_meridian = 262.5
        io.root["crs"].latitude_of_projection_origin = 38.5
        io.root["crs"].semi_major_axis = 6371229
        io.root["crs"].semi_minor_axis = 6371229

        for var in output_coords["variable"]:
            io.root[var].grid_mapping = "crs"

        # Set attributes for automatic parsing of dimensions
        io.root["ensemble"].standard_name = "realization"
        io.root["time"].standard_name = "time"
        io.root["time"].axis = "T"
        io.root["y"].standard_name = "projection_y_coordinate"
        io.root["y"].units = "m"
        io.root["y"].axis = "Y"
        io.root["x"].standard_name = "projection_x_coordinate"
        io.root["x"].units = "m"
        io.root["x"].axis = "X"

        # Unwrap BackendProgress (serve API)
        e2io = (
            io
            if isinstance(io, (NetCDF4Backend, ZarrBackend))
            else getattr(io, "io", None)
        )

        if isinstance(e2io, ZarrBackend):
            zarr.consolidate_metadata(e2io.store)

        if isinstance(e2io, NetCDF4Backend):
            from cftime import date2num
            from earth2studio.utils.time import timearray_to_datetime

            # Planetary Computer does not like the original time format.
            ref_time = np.datetime_as_string(output_coords["time"][0], unit="s")
            units = f"hours since {ref_time.replace('T', ' ')}"
            tv = e2io.root["time"]
            tv.units = units
            tv[:] = date2num(
                timearray_to_datetime(output_coords["time"]),
                units=units,
                calendar=tv.calendar,
            )
            e2io.root.sync()

        return io

    def get_fcn3_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch FCN3 branch input from Planetary Computer ECMWF IFS."""
        from earth2studio.data import fetch_data
        from earth2studio.utils.time import to_time_array

        if self.fcn3_interp is None:
            raise RuntimeError("FCN3 model must be loaded before fetching input")

        x, coords = fetch_data(
            self.data_fcn3,
            time=to_time_array([time]),
            variable=self.fcn3_interp.input_coords()["variable"],
            device=self.device,
        )
        return x, coords

    def get_stormscope_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch GOES inputs for StormScope and preprocess them."""
        import torch
        from earth2studio.data import fetch_data
        from earth2studio.utils.time import to_time_array

        if self.stormscope is None:
            raise RuntimeError("StormScope model must be loaded before fetching input")
        if self.data_stormscope is None:
            raise RuntimeError(
                "StormScope data sources must be loaded before fetching input"
            )

        coords_in = self.stormscope.input_coords()
        if time < datetime(2025, 4, 7):
            data = self.data_stormscope["goes16"]
        else:
            data = self.data_stormscope["goes19"]
        x, coords = fetch_data(
            data,
            time=to_time_array([time]),
            variable=coords_in["variable"],
            lead_time=coords_in["lead_time"],
            device=self.device,
        )

        batch_size = 1
        if x.dim() == 5:
            x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
            coords["batch"] = np.arange(batch_size)
            coords.move_to_end("batch", last=False)

        x, coords = self.stormscope.prep_input(x, coords)
        x = torch.where(self.stormscope.valid_mask, x, torch.nan)

        return x, coords

    def run_fcn3(
        self,
        io: IOBackend,
        x: torch.Tensor,
        coords_x: CoordSystem,
        seed_fcn3: int,
        start_time_stormscope: datetime,
        lead_times: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
        """Run FCN3 to produce conditioning fields for StormScope."""
        from earth2studio.utils.coords import map_coords, split_coords
        from earth2studio.utils.time import to_time_array

        if self.fcn3_interp is None:
            raise RuntimeError("FCN3 model must be loaded before inference")
        if self.stormscope is None:
            raise RuntimeError("StormScope model must be loaded before inference")

        # Create z500 conditioning with FCN3
        coords_in = self.stormscope.input_coords()
        start_time_stormscope = to_time_array([start_time_stormscope])
        variables = self.stormscope.conditioning_variables
        # Start time and lead times are shifted to StormScope start time
        output_coords = {
            "time": start_time_stormscope,
            "lead_time": lead_times,
            "variable": variables,
            "y": coords_in["y"],
            "x": coords_in["x"],
        }
        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"}, variables
        )

        model_gap = int(
            (start_time_stormscope - coords_x["time"]) / np.timedelta64(1, "h")
        )

        self.fcn3_interp.px_model.px_model.set_rng(seed=seed_fcn3)
        iterator = self.fcn3_interp.create_iterator(x.clone(), coords_x.copy())

        n_steps = model_gap + len(lead_times)
        for step, (x, coords_x) in enumerate(iterator):
            msg = (
                f"Processing FCN3 for sample {sample + 1}/{total_samples} "
                f"(seed_fcn3={seed_fcn3}) "
                f"step {step + 1}/{n_steps}"
            )
            progress = WorkflowProgress(
                progress=msg,
                current_step=step + 1,
                total_steps=n_steps,
            )
            self.update_progress(progress)
            logger.info(msg)

            if step < model_gap:
                # Skip initial steps leading up to StormScope start time
                continue

            x, coords_x = map_coords(x, coords_x, OrderedDict({"variable": variables}))
            x, coords_x = self.stormscope.prep_input(x, coords_x, conditioning=True)
            coords_x["time"] = start_time_stormscope
            coords_x["lead_time"] = coords_x["lead_time"] - np.timedelta64(
                model_gap, "h"
            )
            io.write(*split_coords(x, coords_x))

            if step == (n_steps - 1):
                break

    def run_stormscope(
        self,
        io: IOBackend,
        y: torch.Tensor,
        coords_y: CoordSystem,
        seed_fcn3: int,
        seed_stormscope: int,
        lead_times: np.ndarray,
        variables: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
        """Run StormScope autoregressively and write outputs to ``io``."""
        import torch
        from earth2studio.utils.coords import CoordSystem, map_coords, split_coords

        n_steps = len(lead_times)

        def log_progress(step: int) -> None:
            msg = (
                f"Processing sample {sample + 1}/{total_samples} "
                f"(seed_fcn3={seed_fcn3}, seed_stormscope={seed_stormscope}), "
                f"step {step + 1}/{n_steps}"
            )
            progress = WorkflowProgress(
                progress=msg,
                current_step=step + 1,
                total_steps=n_steps,
            )
            self.update_progress(progress)
            logger.info(msg)

        def prep_output(
            y_pred: torch.Tensor, coords_pred: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            y_out, coords_out = map_coords(
                y_pred, coords_pred, CoordSystem({"variable": variables})
            )
            del coords_out["batch"]
            # Reuse batch dimension as ensemble dimension (squeeze/unsqueeze)
            coords_out["ensemble"] = np.array([sample])
            coords_out.move_to_end("ensemble", last=False)
            # Combine time and lead_time
            lead_time_dim = list(coords_out).index("lead_time")
            y_out = y_out.squeeze(lead_time_dim)
            coords_out["time"] = coords_out["time"] + coords_out["lead_time"]
            del coords_out["lead_time"]
            return y_out, coords_out

        if self.stormscope is None:
            raise RuntimeError("StormScope model must be loaded before inference")

        # Update progress for step within sample
        log_progress(0)

        # Store initial GOES data (identical across seeds)
        y_out, coords_out = prep_output(y, coords_y)
        io.write(*split_coords(y_out, coords_out))

        # Cannot use seeded Generator before torch==2.10.
        # Use self.stormscope.sampler_args["randn_like"] once updated.
        torch.manual_seed(seed_stormscope)

        for step in range(1, n_steps):
            y_pred, coords_pred = self.stormscope(y, coords_y)

            # Update progress for step within sample
            log_progress(step)

            y_out, coords_out = prep_output(y_pred, coords_pred)
            io.write(*split_coords(y_out, coords_out))

            if step == (n_steps - 1):
                break

            y, coords_y = self.stormscope.next_input(y_pred, coords_pred, y, coords_y)

    def __call__(
        self,
        io: IOBackend,
        start_time_fcn3: datetime | str = datetime(2025, 1, 1, 18),
        start_time_stormscope: datetime | str = datetime(2025, 1, 1, 18),
        n_steps: int = 12,
        n_samples_fcn3: int = 1,
        n_samples_stormscope: int = 1,
        seeds_fcn3: Sequence[int] | None = None,
        seeds_stormscope: Sequence[int] | None = None,
        variables: Sequence[str] | None = ("abi01c", "abi02c", "abi03c"),
    ) -> None:
        from earth2studio.data import InferenceOutputSource
        from earth2studio.io import XarrayBackend
        from earth2studio.utils.time import to_time_array

        start_time_fcn3 = self._coerce_start_time(start_time_fcn3)
        start_time_stormscope = self._coerce_start_time(start_time_stormscope)
        self._validate_plugin_parameters(
            n_steps=n_steps,
            n_samples_fcn3=n_samples_fcn3,
            n_samples_stormscope=n_samples_stormscope,
            output_format="zarr",
            container_url=None,
            geo_catalog_url=None,
        )
        self.validate_start_times(start_time_stormscope, start_time_fcn3)

        self._ensure_runtime_loaded()
        if self.stormscope is None:
            raise RuntimeError("StormScope model must be loaded before inference")

        lead_times = np.array([np.timedelta64(i, "h") for i in range(n_steps + 1)])
        # Different StormScope seed for every trajectory
        if n_samples_stormscope % n_samples_fcn3 != 0:
            raise ValueError(
                "'n_samples_stormscope' must be divisible by 'n_samples_fcn3'"
            )
        seeds_fcn3 = self.validate_samples(n_samples_fcn3, seeds_fcn3)
        seeds_stormscope = self.validate_samples(n_samples_stormscope, seeds_stormscope)
        n_stormscope_per_fcn3 = len(seeds_stormscope) // len(seeds_fcn3)
        variables = self.validate_variables(variables)

        x_ori, coords_x_ori = self.get_fcn3_input(start_time_fcn3)
        y_ori, coords_y_ori = self.get_stormscope_input(start_time_stormscope)

        coords_out = self.stormscope.output_coords(self.stormscope.input_coords())
        output_coords = {
            "ensemble": np.arange(len(seeds_stormscope)),
            # Planetary Computer does not like separate lead_time
            "time": to_time_array([start_time_stormscope]) + lead_times,
            "variable": variables,
            "y": coords_out["y"],
            "x": coords_out["x"],
        }
        self.setup_io(io, output_coords, seeds_fcn3, seeds_stormscope)

        total_samples = len(seeds_stormscope)
        sample = 0
        for seed_fcn3 in seeds_fcn3:
            # Generate FCN3 conditioning (z500)
            logger.info("Starting FCN3 inference")
            io_fcn3 = XarrayBackend()
            self.run_fcn3(
                io=io_fcn3,
                x=x_ori.clone(),
                coords_x=coords_x_ori.copy(),
                seed_fcn3=seed_fcn3,
                start_time_stormscope=start_time_stormscope,
                lead_times=lead_times,
                sample=sample,
                total_samples=total_samples,
            )
            self.stormscope.conditioning_data_source = InferenceOutputSource(
                io_fcn3.root
            )

            # Run StormScope forecast conditioned on FCN3
            logger.info("Starting StormScope inference")
            for _ in range(n_stormscope_per_fcn3):
                self.run_stormscope(
                    io=io,
                    y=y_ori.clone(),
                    coords_y=coords_y_ori.copy(),
                    seed_fcn3=seed_fcn3,
                    seed_stormscope=seeds_stormscope[sample],
                    lead_times=lead_times,
                    variables=variables,
                    sample=sample,
                    total_samples=total_samples,
                )
                sample += 1

    def cleanup(self) -> None:
        try:
            super().cleanup()
        finally:
            if self.stormscope is not None:
                self.stormscope.conditioning_data_source = None
            self._release_models(self.fcn3_interp, self.stormscope)
            self._clear_attributes(
                "fcn3_interp",
                "stormscope",
                "data_fcn3",
                "data_stormscope",
            )
            self._cleanup_torch_runtime(device=str(self.device))

    def cleanup_request(self) -> None:
        if self.stormscope is not None:
            self.stormscope.conditioning_data_source = None
        super().cleanup_request()


WORKFLOW = FoundryFCN3StormScopeGOESWorkflow
