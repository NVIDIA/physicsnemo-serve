# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from e2s_workflow import Earth2Workflow
from plugin_sdk import cleanup_earth2_runtime_resources

if TYPE_CHECKING:
    from earth2studio.io import IOBackend

logger = logging.getLogger(__name__)


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.dx import PrecipitationAFNO
    from earth2studio.models.px import FCN

    fcn_package = FCN.load_default_package()
    FCN.load_model(fcn_package)
    diagnostic_package = PrecipitationAFNO.load_default_package()
    PrecipitationAFNO.load_model(diagnostic_package)
    return {"model_names": ["FCN", "PrecipitationAFNO"]}


class DiagnosticWorkflow(Earth2Workflow):
    """
    Diagnostic workflow that runs Earth2Studio diagnostic forecasts.

    This workflow:
    1. Loads the FCN prognostic model
    2. Loads a diagnostic model (e.g. precipitation_afno)
    3. Sets up a data source (GFS, ERA5)
    4. Runs diagnostic forecast
    5. Saves results in specified format
    6. Optionally creates visualization plots
    """

    name = "diagnostic_workflow"
    description = "Earth2Studio diagnostic forecast workflow with visualization"
    cache_scope = "process"
    model_cache_names = ["FCN", "PrecipitationAFNO"]
    cache_preserve_attributes = (
        "prognostic_package",
        "diagnostic_package",
        "prognostic_model",
        "diagnostic_model",
        "data",
    )

    def __init__(self):
        super().__init__()
        self.prognostic_package: Any = None
        self.diagnostic_package: Any = None
        self.prognostic_model: Any = None
        self.diagnostic_model: Any = None
        self.data: Any = None

    def _ensure_runtime_loaded(self) -> tuple[Any, Any, Any]:
        from earth2studio.data import GFS
        from earth2studio.models.dx import PrecipitationAFNO
        from earth2studio.models.px import FCN

        if self.prognostic_model is None:
            self.prognostic_package = FCN.load_default_package()
            self.prognostic_model = FCN.load_model(self.prognostic_package)
        if self.diagnostic_model is None:
            self.diagnostic_package = PrecipitationAFNO.load_default_package()
            self.diagnostic_model = PrecipitationAFNO.load_model(
                self.diagnostic_package
            )
        if self.data is None:
            self.data = GFS()
        return self.prognostic_model, self.diagnostic_model, self.data

    def warmup(self, _ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded()
        return {"model_names": ["FCN", "PrecipitationAFNO"]}

    def __call__(
        self,
        io: IOBackend,
        forecast_times: list[str] = ["2024-01-01T00:00:00"],
        nsteps: int = 6,
        prognostic_model_type: Literal["fcn"] = "fcn",
        diagnostic_model_type: Literal["precipitation_afno"] = "precipitation_afno",
        data_source: Literal["gfs"] = "gfs",
        output_format: Literal["zarr"] = "zarr",
        create_plots: bool = True,
        plot_variable: Literal["tp"] = "tp",
        plot_step: int = 4,
    ) -> None:
        from earth2studio import run

        if prognostic_model_type.lower() != "fcn":
            raise ValueError("e2s-diagnostic supports only prognostic_model_type='fcn'")
        if diagnostic_model_type.lower() != "precipitation_afno":
            raise ValueError(
                f"Unsupported diagnostic model type: {diagnostic_model_type}"
            )
        if data_source.lower() != "gfs":
            raise ValueError(f"Unsupported data source: {data_source}")
        if output_format.lower() != "zarr":
            raise ValueError(f"Unsupported output format: {output_format}")

        prognostic_model, diagnostic_model, data = self._ensure_runtime_loaded()
        io = run.diagnostic(  # type: ignore[assignment]
            forecast_times,
            nsteps,
            prognostic_model,
            diagnostic_model,
            data,
            io,
        )

        if output_format.lower() == "zarr":
            io = self.finalize_zarr_output(io)

        forecast_info = {
            "forecast_times": forecast_times,
            "nsteps": nsteps,
            "prognostic_model_type": prognostic_model_type,
            "diagnostic_model_type": diagnostic_model_type,
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
            import numpy as np

            forecast_time = forecast_times[0]
            variable = plot_variable
            step = min(plot_step, nsteps - 1)

            plt.close("all")

            projection = ccrs.Orthographic(-100, 40)
            _, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 6))

            lon = io["lon"][:]
            lat = io["lat"][:]

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

            levels = np.arange(0.0, 0.01, 0.001)

            im = ax.contourf(
                lon,
                lat,
                data,
                levels,
                transform=ccrs.PlateCarree(),
                vmax=0.01,
                vmin=0.00,
                cmap="terrain",
            )

            plt.colorbar(
                im,
                ax=ax,
                ticks=levels,
                shrink=0.75,
                pad=0.04,
                label="Total precipitation (m)",
            )

            lead_time_hours = step * 6
            ax.set_title(
                f"{forecast_time} + {lead_time_hours}hrs - {variable}",
                fontsize=14,
            )

            ax.set_extent([220, 340, 20, 70])
            ax.coastlines()
            ax.gridlines()

            plot_path = self.output_dir / f"forecast_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create forecast plot")
            raise

    def cleanup(self) -> None:
        self._clear_attributes("prognostic_package", "diagnostic_package")
        try:
            super().cleanup()
        finally:
            cleanup_earth2_runtime_resources(
                self.prognostic_model,
                self.diagnostic_model,
            )
            self._clear_attributes("prognostic_model", "diagnostic_model", "data")
            self._cleanup_torch_runtime()


WORKFLOW = DiagnosticWorkflow
