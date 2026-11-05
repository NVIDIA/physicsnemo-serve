# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the complete ensemble through stock Earth2Studio on one CUDA device."""

from collections import OrderedDict

import numpy as np

from plugin_sdk import PluginCancelledError


class FrozenInputSource:
    """Expose the staged two-time initialization through Earth2Studio's data API."""

    def __init__(self, loaded):
        import xarray as xr

        self._data = xr.DataArray(
            loaded.values[0],
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": loaded.coords["time"][0] + loaded.coords["lead_time"],
                "variable": loaded.coords["variable"],
                "lat": loaded.coords["lat"],
                "lon": loaded.coords["lon"],
            },
        )

    def __call__(self, time, variable):
        return self._data.sel(
            time=np.asarray(time, dtype="datetime64[ns]"), variable=np.asarray(variable)
        ).copy(deep=True)


class _ForecastIterator:
    """Release native state at the final yield, before the next batch allocates."""

    def __init__(self, iterator, abort_requested, nsteps):
        self._iterator = iterator
        self._abort_requested = abort_requested
        self._remaining = nsteps + 1  # Initialization plus the forecast steps.

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            raise StopIteration
        try:
            if self._abort_requested():
                raise PluginCancelledError("forecast cancelled")
            result = next(self._iterator)
        except BaseException:
            self.close()
            raise
        self._remaining -= 1
        if self._remaining == 0:
            # A plain iterator has no suspended frame retaining the input or
            # final output while stock ensemble() allocates the next minibatch.
            self.close()
        return result

    def close(self):
        iterator, self._iterator = self._iterator, None
        if iterator is not None:
            iterator.close()


class _AbortableModel:
    """Keep stock ensemble orchestration while bounding each native iterator."""

    def __init__(self, model, abort_requested, nsteps):
        self._model = model
        self._abort_requested = abort_requested
        self._nsteps = nsteps
        self._iterators = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def to(self, device):
        self._model.to(device)
        return self

    def create_iterator(self, values, coords):
        self.close()
        if self._abort_requested():
            raise PluginCancelledError("forecast cancelled")
        iterator = _ForecastIterator(
            self._model.create_iterator(values, coords),
            self._abort_requested,
            self._nsteps,
        )
        self._iterators.append(iterator)
        return iterator

    def close(self):
        try:
            for iterator in self._iterators:
                iterator.close()
        finally:
            self._iterators.clear()


def run_ensemble_rollout(
    model,
    loaded,
    sink,
    *,
    device="cuda",
    abort_requested=lambda: False,
    torch_module=None,
    ensemble_runner=None,
):
    if torch_module is None:
        import torch as torch_module

    torch = torch_module
    previous_seed = model.seed
    adapted = None
    try:
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("AIFS2ENS ensemble execution requires a CUDA device")
        device_index = (
            target.index if target.index is not None else torch.cuda.current_device()
        )
        if ensemble_runner is None:
            from earth2studio.run import ensemble as ensemble_runner
        from earth2studio.perturbation import Zero

        source = FrozenInputSource(loaded)
        perturbation = Zero()
        request = loaded.request
        with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
            try:
                if abort_requested():
                    raise PluginCancelledError("forecast cancelled")
                # Default Earth2Studio AIFS2ENS leaves the shared RNG stream
                # advancing. A non-None model.seed would reseed every forward.
                model.seed = None
                adapted = _AbortableModel(model, abort_requested, request.nsteps)
                torch.manual_seed(request.seed_base)
                ensemble_runner(
                    time=loaded.coords["time"].copy(),
                    nsteps=request.nsteps,
                    nensemble=request.nensemble,
                    prognostic=adapted,
                    data=source,
                    io=sink,
                    perturbation=perturbation,
                    batch_size=request.batch_size,
                    output_coords=OrderedDict(
                        variable=np.array(request.output_variables)
                    ),
                    device=target,
                    verbose=False,
                )
                if abort_requested():
                    raise PluginCancelledError("forecast cancelled")
                sink.assert_complete()
            finally:
                try:
                    if adapted is not None:
                        adapted.close()
                finally:
                    model.seed = previous_seed
    finally:
        sink.close()
