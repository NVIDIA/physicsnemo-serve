# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single seeded ensemble execution and frozen datasource lifecycle contracts."""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
import aifs2ens_rollout as rollout  # noqa: E402


@pytest.fixture
def loaded():
    request = SimpleNamespace(
        start_time="2026-11-04T12:00:00Z",
        nensemble=3,
        batch_size=2,
        nsteps=2,
        seed_base=0,
        output_variables=["snowc", "t2m"],
    )
    coords = OrderedDict(
        time=np.array(["2026-11-04T12:00:00"], dtype="datetime64[ns]"),
        lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
        variable=np.array(["t2m", "msl"]),
        lat=np.array([90.0, -90.0]),
        lon=np.array([0.0, 180.0]),
    )
    return SimpleNamespace(
        request=request,
        coords=coords,
        values=np.arange(16, dtype=np.float32).reshape(1, 2, 2, 2, 2),
        contract=SimpleNamespace(
            input_variables=["t2m", "msl"], lat=[90.0, -90.0], lon=[0.0, 180.0]
        ),
    )


class Torch:
    def __init__(self):
        self.events = []
        self.random = SimpleNamespace(fork_rng=self.fork_rng)
        self.cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0)
        self.math_flags = (True, True, False)

    @contextmanager
    def fork_rng(self, devices):
        self.events.append(("save_rng", devices))
        try:
            yield
        finally:
            self.events.append(("restore_rng", devices))

    def inference_mode(self):
        return nullcontext()

    def manual_seed(self, seed):
        self.events.append(("seed", seed))

    def device(self, device):
        return SimpleNamespace(type="cuda", index=None if device == "cuda" else 2)


class Model:
    def __init__(self):
        self.seed = 999
        self.moves = []
        self.closed = False
        self.predictions = 0

    def to(self, device):
        self.moves.append(device)
        return self

    def create_iterator(self, tensor, coords):
        try:
            yield "initial", coords
            for step in (1, 2):
                self.predictions += 1
                yield step, coords
        finally:
            self.closed = True


class Sink:
    def __init__(self):
        self.closed = False
        self.complete = False

    def assert_complete(self):
        self.complete = True

    def close(self):
        self.closed = True


class Zero:
    def __call__(self, values, coords):
        return values, coords


@pytest.fixture(autouse=True)
def zero_perturbation(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "earth2studio.perturbation", SimpleNamespace(Zero=Zero)
    )


def test_frozen_source_selects_two_physical_times_and_requested_variable_order(loaded):
    source = rollout.FrozenInputSource(loaded)
    requested = np.array(
        ["2026-11-04T12:00:00", "2026-11-04T06:00:00"], dtype="datetime64[ns]"
    )
    data = source(requested, np.array(["msl", "t2m"]))
    assert data.dims == ("time", "variable", "lat", "lon")
    np.testing.assert_array_equal(data.time.values, requested)
    np.testing.assert_array_equal(data.coords["variable"].values, ["msl", "t2m"])
    np.testing.assert_array_equal(data.values, loaded.values[0, ::-1, ::-1])
    data.values[...] = -1
    assert loaded.values.min() == 0  # The staged input must remain immutable.
    with pytest.raises(KeyError):
        source(np.array(["2026-11-05"], dtype="datetime64[ns]"), ["t2m"])


def test_calls_upstream_ensemble_once_with_shared_seed_and_request_batching(loaded):
    model, sink, torch = Model(), Sink(), Torch()
    calls = []
    retained = []

    def ensemble(**kwargs):
        calls.append(kwargs)
        assert kwargs["prognostic"].seed is None
        assert torch.math_flags == (True, True, False)
        kwargs["prognostic"] = kwargs["prognostic"].to(kwargs["device"])
        iterator = kwargs["prognostic"].create_iterator(None, {})
        retained.append(iterator)
        assert next(iterator)[0] == "initial"
        assert next(iterator)[0] == 1
        # Upstream breaks at the requested horizon rather than closing iterators.

    rollout.run_ensemble_rollout(
        model, loaded, sink, torch_module=torch, ensemble_runner=ensemble
    )
    assert len(calls) == 1
    args = calls[0]
    assert args["nensemble"] == 3 and args["batch_size"] == 2 and args["nsteps"] == 2
    assert args["io"] is sink
    assert args["output_coords"]["variable"].tolist() == ["snowc", "t2m"]
    assert isinstance(args["perturbation"], Zero)
    np.testing.assert_array_equal(args["time"], loaded.coords["time"])
    assert torch.events == [("save_rng", [0]), ("seed", 0), ("restore_rng", [0])]
    assert model.seed == 999 and model.closed and sink.closed and sink.complete


@pytest.mark.parametrize("failure", ["runner", "cancel", "coverage"])
def test_failure_restores_seed_rng_and_closes_retained_iterator(loaded, failure):
    model, sink, torch = Model(), Sink(), Torch()
    cancelled = False
    retained = []

    def ensemble(**kwargs):
        nonlocal cancelled
        iterator = kwargs["prognostic"].create_iterator(None, {})
        retained.append(iterator)
        next(iterator)
        if failure == "cancel":
            cancelled = True
            next(iterator)
        elif failure == "runner":
            raise RuntimeError("inference failed")

    if failure == "coverage":
        sink.assert_complete = lambda: (_ for _ in ()).throw(
            ValueError("incomplete forecast coverage")
        )
    with pytest.raises(
        (RuntimeError, InterruptedError, ValueError), match="failed|cancelled|coverage"
    ):
        rollout.run_ensemble_rollout(
            model,
            loaded,
            sink,
            torch_module=torch,
            ensemble_runner=ensemble,
            abort_requested=lambda: cancelled,
        )
    assert model.closed and model.seed == 999 and sink.closed
    assert torch.events[-1] == ("restore_rng", [0])
    if failure == "cancel":
        assert model.predictions == 0


def test_cancelled_request_does_not_enter_upstream_runner(loaded):
    sink, model, torch = Sink(), Model(), Torch()
    with pytest.raises(InterruptedError, match="cancelled"):
        rollout.run_ensemble_rollout(
            model,
            loaded,
            sink,
            torch_module=torch,
            ensemble_runner=lambda **kwargs: pytest.fail(
                "cancelled request ran inference"
            ),
            abort_requested=lambda: True,
        )
    assert sink.closed and model.seed == 999
    assert not any(event[0] == "seed" for event in torch.events)


def test_explicit_gpu_index_is_used_for_rng_isolation(loaded):
    torch, sink, model = Torch(), Sink(), Model()
    rollout.run_ensemble_rollout(
        model,
        loaded,
        sink,
        device="cuda:2",
        torch_module=torch,
        ensemble_runner=lambda **kwargs: None,
    )
    assert torch.events[0] == ("save_rng", [2])
    assert torch.events[-1] == ("restore_rng", [2])


def test_final_forecast_releases_native_state_before_next_batch_allocation(loaded):
    model, sink, torch = Model(), Sink(), Torch()
    retained = []

    class BatchAllocation:
        pass

    def ensemble(**kwargs):
        previous_allocation = None
        # This preserves upstream run.ensemble's ordering: repeat/allocate the
        # new input batch first, then call prognostic.create_iterator().
        for _batch in range(2):
            if previous_allocation is not None:
                assert model.closed, "old native state survived until the next repeat"
                assert previous_allocation() is None, (
                    "old input is retained by the proxy"
                )
            x = BatchAllocation()  # Represents x0.unsqueeze(...).repeat(...).
            previous_allocation = weakref.ref(x)
            iterator = kwargs["prognostic"].create_iterator(x, {})
            retained.append(iterator)
            for step, (x, _coords) in enumerate(iterator):
                if step == loaded.request.nsteps:
                    break
            assert x == 2  # Closing the native iterator preserves its final output.
        assert previous_allocation() is None

    rollout.run_ensemble_rollout(
        model, loaded, sink, torch_module=torch, ensemble_runner=ensemble
    )
