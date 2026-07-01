# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from e2s_workflow import Earth2Workflow

if TYPE_CHECKING:
    from earth2studio.io import IOBackend

logger = logging.getLogger(__name__)


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.px import FCN

    package = FCN.load_default_package()
    FCN.load_model(package)
    return {"model_names": ["FCN"]}


class DeterministicFCNWorkflow(Earth2Workflow):
    """
    Deterministic workflow that runs Earth2Studio deterministic forecasts.

    This workflow:
    1. Loads a prognostic model (FCN)
    2. Sets up a data source (GFS, ERA5)
    3. Runs deterministic forecast
    4. Saves results in specified format
    5. Optionally creates visualization plots
    """

    name = "deterministic_fcn_workflow"
    description = "Earth2Studio deterministic forecast workflow with FCN model"
    cache_scope = "process"
    model_cache_names = ["FCN"]
    cache_preserve_attributes = ("package", "model")

    def __init__(self) -> None:
        super().__init__()
        from earth2studio.models.px import FCN

        # load the model once and store it in the instance
        self.package = FCN.load_default_package()
        self.model = FCN.load_model(self.package)
        self._data: Any = None

    def warmup(self, ctx: dict[str, Any]) -> dict[str, list[str]]:
        device = str(ctx.get("device") or "cuda")
        self.model = self.model.to(device)
        self.model.eval()
        return {"model_names": ["FCN"]}

    def __call__(
        self,
        io: IOBackend,
        forecast_times: list[str] = ["2024-01-01T00:00:00"],
        nsteps: int = 6,
        data_source: Literal["gfs"] = "gfs",
        output_format: Literal["zarr"] = "zarr",
        create_plots: bool = True,
        plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = "t2m",
        plot_step: int = 4,
    ) -> None:
        from earth2studio import run
        from earth2studio.data import GFS

        # Set up data source
        if data_source.lower() == "gfs":
            data = GFS()
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        if output_format.lower() != "zarr":
            raise ValueError(f"Unsupported output format: {output_format}")

        self._data = data

        # Execute the workflow
        io_result = run.deterministic(  # type: ignore[assignment]
            forecast_times,
            nsteps,
            self.model,
            data,
            io,
        )
        if io_result is not None:
            io = io_result  # type: ignore[assignment]

        # Consolidate zarr metadata for faster remote access
        if output_format.lower() == "zarr":
            io = self.finalize_zarr_output(io)

        forecast_info = {
            "forecast_times": forecast_times,
            "nsteps": nsteps,
            "model_type": "FCN",
            "data_source": data_source,
            "output_format": output_format,
        }

        metadata_path = self.output_dir / "forecast_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(forecast_info, f, indent=2)

        if create_plots:
            self.create_forecast_plot(
                io,
                forecast_times,
                nsteps,
                plot_variable,
                plot_step,
            )

    def create_forecast_plot(
        self,
        io: Any,
        forecast_times: list[str],
        nsteps: int,
        plot_variable: str,
        plot_step: int,
    ) -> None:
        """Create a forecast visualization plot."""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt

            forecast_time = forecast_times[0]
            variable = plot_variable
            step = min(plot_step, nsteps - 1)

            plt.close("all")

            # Create a Robinson projection
            projection = ccrs.Robinson()

            # Create a figure and axes with the specified projection
            _, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 8))

            # Get data from the IO object
            lon = io["lon"][:]
            lat = io["lat"][:]

            # Handle the case where the variable might not exist
            coord_keys = {"lon", "lat", "time", "lead_time", "ensemble"}
            if variable in io and variable not in coord_keys:
                data = io[variable][0, step]
            else:
                available_vars = [
                    key for key in io if key not in coord_keys and io[key].ndim >= 2
                ]
                if available_vars:
                    variable = available_vars[0]
                    data = io[variable][0, step]
                else:
                    raise ValueError("No data variables found in forecast output")

            # Plot the field using pcolormesh
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=ccrs.PlateCarree(),
                cmap="Spectral_r",
            )

            # Add colorbar
            cbar = plt.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                pad=0.1,
                shrink=0.8,
            )
            cbar.set_label(f"{variable}")

            # Calculate lead time in hours (assuming 6-hour steps for FCN)
            lead_time_hours = step * 6

            # Set title
            ax.set_title(
                f"{forecast_time} + {lead_time_hours}hrs - {variable}",
                fontsize=14,
            )

            # Add coastlines and gridlines
            ax.coastlines()
            ax.gridlines(alpha=0.5)

            plot_path = self.output_dir / f"forecast_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create forecast plot")
            raise

    def cleanup(self) -> None:
        # Package HTTP sessions may be shared by Earth2 package caches.
        # Drop package refs before generic cleanup to avoid closing shared clients.
        self._clear_attributes("package")
        try:
            super().cleanup()
        finally:
            self._clear_attributes("model", "_data")
            self._cleanup_torch_runtime()

    def cleanup_request(self) -> None:
        package = self.package
        model = self.model
        self.package = None
        self.model = None
        try:
            super().cleanup()
        finally:
            self.package = package
            self.model = model
            self._clear_attributes("_data")


WORKFLOW = DeterministicFCNWorkflow
