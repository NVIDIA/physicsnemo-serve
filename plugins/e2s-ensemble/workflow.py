# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from e2s_workflow import Earth2Workflow, _selected_zarr_backend

if TYPE_CHECKING:
    from earth2studio.io import IOBackend

logger = logging.getLogger(__name__)


def prepare_model_cache(_ctx: dict[str, Any]) -> dict[str, list[str]]:
    from earth2studio.models.px import FCN

    package = FCN.load_default_package()
    FCN.load_model(package)
    return {"model_names": ["FCN"]}


def _build_perturbation(name: str, noise_amplitude: float) -> tuple[Any, str]:
    normalized = str(name).strip().lower()
    if normalized == "gaussian":
        from earth2studio.perturbation import Gaussian

        return Gaussian(noise_amplitude=noise_amplitude), normalized
    if normalized == "brown":
        from earth2studio.perturbation import Brown

        return Brown(noise_amplitude=noise_amplitude), normalized
    if normalized == "spherical_gaussian":
        from earth2studio.perturbation import SphericalGaussian

        return SphericalGaussian(noise_amplitude=noise_amplitude), normalized
    raise ValueError(
        "perturbation must be 'gaussian', 'brown', or 'spherical_gaussian'"
    )


class EnsembleWorkflow(Earth2Workflow):
    """
    Ensemble workflow that runs Earth2Studio ensemble forecasts.

    This workflow:
    1. Loads a prognostic model (FCN)
    2. Sets up the requested ensemble perturbation
    3. Sets up a data source (GFS)
    4. Runs ensemble forecast with multiple members
    5. Saves results in zarr format
    6. Optionally creates visualization plots (members + std)
    """

    name = "ensemble_workflow"
    description = (
        "Earth2Studio ensemble forecast workflow with perturbation and visualization"
    )
    cache_scope = "process"
    model_cache_names = ["FCN"]
    cache_preserve_attributes = ("_package", "_model", "_data")

    def __init__(self):
        super().__init__()
        self._package: Any = None
        self._model: Any = None
        self._data: Any = None

    def _ensure_runtime_loaded(self) -> tuple[Any, Any, Any]:
        from earth2studio.data import GFS
        from earth2studio.models.px import FCN

        if self._model is None:
            self._package = FCN.load_default_package()
            self._model = FCN.load_model(self._package)
        if self._data is None:
            self._data = GFS()
        return self._package, self._model, self._data

    def warmup(self, _ctx: dict[str, Any]) -> dict[str, list[str]]:
        self._ensure_runtime_loaded()
        return {"model_names": ["FCN"]}

    def create_io(self, dataset_path: str):
        zarr_kwargs = {
            "chunks": {"ensemble": 1, "time": 1, "lead_time": 1},
            "backend_kwargs": {"overwrite": True},
        }
        if _selected_zarr_backend() != "python":
            zarr_kwargs["default_parallel_coord_names"] = [
                "ensemble",
                "time",
                "lead_time",
            ]
        return self.create_zarr_io(dataset_path, **zarr_kwargs)

    def __call__(
        self,
        io: IOBackend,
        forecast_times: list[str] = ["2024-01-01T00:00:00"],
        nsteps: int = 10,
        nensemble: int = 8,
        batch_size: int = 2,
        perturbation: Literal[
            "gaussian", "brown", "spherical_gaussian"
        ] = "spherical_gaussian",
        model_type: Literal["fcn"] = "fcn",
        noise_amplitude: float = 0.15,
        seed_base: int = 1000,
        data_source: Literal["gfs"] = "gfs",
        output_format: Literal["zarr"] = "zarr",
        output_variables: list[str] | None = None,
        create_plots: bool = True,
        plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = "tcwv",
        plot_step: int = 4,
    ) -> None:
        import numpy as np
        import torch
        from earth2studio import run

        if model_type.lower() != "fcn":
            raise ValueError(f"Unsupported model type: {model_type}")
        if data_source.lower() != "gfs":
            raise ValueError(f"Unsupported data source: {data_source}")
        if output_format.lower() != "zarr":
            raise ValueError(f"Unsupported output format: {output_format}")

        _package, model, data = self._ensure_runtime_loaded()
        torch.manual_seed(int(seed_base))
        sg, perturbation = _build_perturbation(perturbation, noise_amplitude)

        # Optional output coordinate filter (variable subset)
        output_coords: OrderedDict = OrderedDict()
        if output_variables:
            output_coords = OrderedDict({"variable": np.array(output_variables)})

        # Run ensemble workflow
        io_result = run.ensemble(  # type: ignore[assignment]
            forecast_times,
            nsteps,
            nensemble,
            model,
            data,
            io,
            sg,
            batch_size=batch_size,
            output_coords=output_coords,
        )
        if io_result is not None:
            io = io_result  # type: ignore[assignment]

        # Consolidate zarr metadata
        io = self.finalize_zarr_output(io)

        forecast_info = {
            "forecast_times": forecast_times,
            "nsteps": nsteps,
            "nensemble": nensemble,
            "batch_size": batch_size,
            "model_type": model_type,
            "perturbation": perturbation,
            "noise_amplitude": noise_amplitude,
            "seed_base": seed_base,
            "data_source": data_source,
            "output_format": output_format,
        }

        metadata_path = self.output_dir / "forecast_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(forecast_info, f, indent=2)

        if create_plots:
            self.create_ensemble_plot(
                io,
                forecast_times,
                nsteps,
                nensemble,
                plot_variable,
                plot_step,
            )

    def create_ensemble_plot(
        self,
        io: Any,
        forecast_times: list[str],
        nsteps: int,
        nensemble: int,
        plot_variable: str,
        plot_step: int,
    ) -> None:
        """Create ensemble visualization (two members + std)."""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            import numpy as np

            variable = plot_variable
            step = min(plot_step, nsteps - 1)
            forecast_time = forecast_times[0]

            coord_keys = {"lon", "lat", "time", "lead_time", "ensemble"}
            if variable not in io or variable in coord_keys:
                available = [k for k in io if k not in coord_keys and io[k].ndim >= 2]
                variable = available[0] if available else "t2m"

            plt.close("all")
            projection = ccrs.Robinson()
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=1,
                ncols=3,
                subplot_kw={"projection": projection},
                figsize=(16, 3),
            )

            lon = io["lon"][:]
            lat = io["lat"][:]
            lead_hrs = 6 * step

            def plot_field(
                axi: Any,
                data: np.ndarray,
                title: str,
                cmap: str = "Blues",
            ) -> None:
                im = axi.pcolormesh(
                    lon,
                    lat,
                    data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                )
                plt.colorbar(im, ax=axi, shrink=0.6, pad=0.04)
                axi.set_title(title)
                axi.coastlines()
                axi.gridlines()

            variable_array = io[variable]

            def array_dimension_names(array: Any) -> list[str] | None:
                attrs = getattr(array, "attrs", None)
                if attrs is not None:
                    try:
                        dimensions = attrs.get("_ARRAY_DIMENSIONS")
                    except Exception:
                        dimensions = None
                    if isinstance(dimensions, list) and all(
                        isinstance(dim, str) for dim in dimensions
                    ):
                        return dimensions

                metadata = getattr(array, "metadata", None)
                dimensions = getattr(metadata, "dimension_names", None)
                if isinstance(dimensions, list) and all(
                    isinstance(dim, str) for dim in dimensions
                ):
                    return dimensions
                return None

            dimension_names = array_dimension_names(variable_array)

            def select_member(member_index: int) -> np.ndarray:
                if dimension_names is None or not all(
                    dim in dimension_names for dim in ["ensemble", "time", "lead_time"]
                ):
                    return variable_array[member_index, 0, step]

                selection: list[Any] = [slice(None)] * len(dimension_names)
                selection[dimension_names.index("ensemble")] = member_index
                selection[dimension_names.index("time")] = 0
                selection[dimension_names.index("lead_time")] = step
                return variable_array[tuple(selection)]

            def ensemble_std() -> np.ndarray:
                if dimension_names is None or not all(
                    dim in dimension_names for dim in ["ensemble", "time", "lead_time"]
                ):
                    return np.std(variable_array[:, 0, step], axis=0)

                selection: list[Any] = [slice(None)] * len(dimension_names)
                ensemble_axis = dimension_names.index("ensemble")
                selection[dimension_names.index("time")] = 0
                selection[dimension_names.index("lead_time")] = step
                subset = np.asarray(variable_array[tuple(selection)])
                remaining_axes = [
                    axis
                    for axis, selected in enumerate(selection)
                    if isinstance(selected, slice)
                ]
                return np.std(subset, axis=remaining_axes.index(ensemble_axis))

            # Member 0, Member 1, Std across members
            plot_field(
                ax1,
                select_member(0),
                f"{forecast_time} - Lead: {lead_hrs}hrs - Member 0",
            )
            if nensemble >= 2:
                plot_field(
                    ax2,
                    select_member(1),
                    f"{forecast_time} - Lead: {lead_hrs}hrs - Member 1",
                )
            else:
                ax2.set_visible(False)
            plot_field(
                ax3,
                ensemble_std(),
                f"{forecast_time} - Lead: {lead_hrs}hrs - Std",
            )

            plot_path = self.output_dir / f"ensemble_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create ensemble plot")
            raise

    def cleanup(self) -> None:
        # Package HTTP sessions may be shared by Earth2 package caches.
        # Drop package refs before generic cleanup to avoid closing shared clients.
        self._clear_attributes("_package")
        try:
            super().cleanup()
        finally:
            self._clear_attributes("_model", "_data")
            self._cleanup_torch_runtime()


WORKFLOW = EnsembleWorkflow
