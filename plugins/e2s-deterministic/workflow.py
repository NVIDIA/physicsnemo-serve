# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from e2s_workflow import Earth2Workflow

if TYPE_CHECKING:
    from earth2studio.io import IOBackend

logger = logging.getLogger(__name__)


def _new_request_scoped_gfs(cache_dir: Path) -> Any:
    """Create GFS lazily with a cache directory owned by one request."""
    from earth2studio.data import GFS

    class _RequestScopedGFS(GFS):
        def __init__(self, owned_cache_dir: Path) -> None:
            self._request_cache_dir = owned_cache_dir
            super().__init__(cache=False)

        @property
        def cache(self) -> str:
            return str(self._request_cache_dir)

    return _RequestScopedGFS(cache_dir)


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.px import DLWP, FCN, FCN3

    dlwp_package = DLWP.load_default_package()
    DLWP.load_model(dlwp_package)
    fcn_package = FCN.load_default_package()
    FCN.load_model(fcn_package)
    fcn3_package = FCN3.load_default_package()
    FCN3.load_model(fcn3_package)
    return {"model_names": ["DLWP", "FCN", "FCN3"]}


class DeterministicWorkflow(Earth2Workflow):
    """
    Deterministic workflow that runs Earth2Studio deterministic forecasts.

    This workflow:
    1. Loads a prognostic model (DLWP, FCN, etc.)
    2. Sets up a data source (GFS, ERA5)
    3. Runs deterministic forecast
    4. Saves results in specified format
    5. Optionally creates visualization plots
    """

    name = "deterministic_workflow"
    description = "Earth2Studio deterministic forecast workflow with visualization"
    model_cache_names = ["DLWP", "FCN", "FCN3"]

    def __init__(self) -> None:
        super().__init__()
        self._packages: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._data_sources: dict[str, Any] = {}
        self._data_cache_dirs: set[Path] = set()

    def _model_for_type(self, model_type: str) -> tuple[Any, Any]:
        from earth2studio.models.px import DLWP, FCN, FCN3

        normalized = model_type.lower()
        if normalized in self._models:
            return self._packages[normalized], self._models[normalized]

        if normalized == "dlwp":
            package = DLWP.load_default_package()
            model = DLWP.load_model(package)
        elif normalized == "fcn":
            package = FCN.load_default_package()
            model = FCN.load_model(package)
        elif normalized == "fcn3":
            package = FCN3.load_default_package()
            model = FCN3.load_model(package)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self._packages[normalized] = package
        self._models[normalized] = model
        return package, model

    def _data_for_source(self, data_source: str) -> Any:
        normalized = data_source.lower()
        if normalized in self._data_sources:
            return self._data_sources[normalized]
        if normalized != "gfs":
            raise ValueError(f"Unsupported data source: {data_source}")
        cache_dir = Path(tempfile.mkdtemp(prefix="physicsnemo-e2s-gfs-"))
        try:
            data = _new_request_scoped_gfs(cache_dir)
        except BaseException:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise
        self._data_cache_dirs.add(cache_dir)
        self._data_sources[normalized] = data
        return data

    def __call__(
        self,
        io: IOBackend,
        forecast_times: list[str] = ["2024-01-01T00:00:00"],
        nsteps: int = 6,
        model_type: Literal["dlwp", "fcn", "fcn3"] = "fcn",
        data_source: Literal["gfs"] = "gfs",
        output_format: Literal["zarr"] = "zarr",
        create_plots: bool = True,
        plot_variable: Literal["t2m", "tcwv", "z500"] = "t2m",
        plot_step: int = 4,
    ) -> None:
        from earth2studio import run

        _package, model = self._model_for_type(model_type)
        data = self._data_for_source(data_source)

        if output_format.lower() != "zarr":
            raise ValueError(f"Unsupported output format: {output_format}")

        # Run deterministic workflow
        io_result = run.deterministic(  # type: ignore[assignment]
            forecast_times, nsteps, model, data, io
        )
        if io_result is not None:
            io = io_result  # type: ignore[assignment]

        # Consolidate zarr metadata for faster remote access
        if output_format.lower() == "zarr":
            io = self.finalize_zarr_output(io)

        # Save metadata about the forecast
        forecast_info = {
            "forecast_times": forecast_times,
            "nsteps": nsteps,
            "model_type": model_type,
            "data_source": data_source,
            "output_format": output_format,
        }
        metadata_path = self.output_dir / "forecast_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(forecast_info, f, indent=2)

        # Create visualization plots if requested
        if create_plots:
            self.create_forecast_plot(
                io, forecast_times, nsteps, plot_variable, plot_step
            )

    def create_forecast_plot(
        self,
        io: Any,
        forecast_times: list[str],
        nsteps: int,
        plot_variable: str,
        plot_step: int,
    ) -> None:
        """Create a forecast visualization plot"""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt

            # Create plot
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
                im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8
            )
            cbar.set_label(f"{variable}")

            # Calculate lead time in hours (assuming 6-hour steps for DLWP)
            lead_time_hours = step * 6

            # Set title
            ax.set_title(
                f"{forecast_time} + {lead_time_hours}hrs - {variable}", fontsize=14
            )

            # Add coastlines and gridlines
            ax.coastlines()
            ax.gridlines(alpha=0.5)

            # Save plot
            plot_path = self.output_dir / f"forecast_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create forecast plot")
            raise

    def cleanup(self) -> None:
        try:
            super().cleanup()
        finally:
            from plugin_sdk import cleanup_earth2_runtime_resources

            try:
                cleanup_earth2_runtime_resources(
                    *list((self._models or {}).values()),
                    *list((self._data_sources or {}).values()),
                )
            finally:
                for cache_dir in self._data_cache_dirs or ():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                self._clear_attributes(
                    "_packages", "_models", "_data_sources", "_data_cache_dirs"
                )
                self._cleanup_torch_runtime()


WORKFLOW = DeterministicWorkflow
