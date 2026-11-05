# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A whole-ensemble sink must preserve native minibatch writes and coverage."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
import aifs2ens_storage as storage  # noqa: E402


class HostTensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return np.asarray(self)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    writes, registrations, options, finalizes = [], [], [], []
    backend = SimpleNamespace(
        add_array=lambda coords, names: registrations.append((coords, names)),
        write=lambda data, coords, names: writes.append((data, coords, names)),
        finalize=lambda: finalizes.append(True),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda value: value.view(HostTensor)),
    )
    monkeypatch.setitem(
        sys.modules,
        "e2s_workflow",
        SimpleNamespace(
            _selected_zarr_backend=lambda: "python",
            create_zarr_backend=lambda *args, **kwargs: (
                options.append(kwargs) or backend
            ),
        ),
    )
    loaded = SimpleNamespace(
        request=SimpleNamespace(
            start_time="2026-11-04T12:00:00Z",
            nensemble=3,
            nsteps=2,
            output_variables=["t2m", "msl"],
        ),
        contract=SimpleNamespace(lat=[90.0, -90.0], lon=[0.0, 180.0]),
    )
    coords = OrderedDict(
        ensemble=np.arange(3),
        time=np.array(["2026-11-04T12:00:00"], dtype="datetime64[ns]"),
        lead_time=np.array([0, 6, 12], dtype="timedelta64[h]"),
        lat=np.array([90.0, -90.0]),
        lon=np.array([0.0, 180.0]),
    )
    return tmp_path, loaded, coords, writes, registrations, options, finalizes


def write_batch(sink, coords, members, lead, *, dtype=np.float32):
    current = OrderedDict(coords)
    current["ensemble"] = np.array(members)
    current["lead_time"] = np.array([lead], dtype="timedelta64[h]")
    arrays = [
        np.full((len(members), 1, 1, 2, 2), index + lead, dtype=dtype)
        for index in (0, 1)
    ]
    sink.write(arrays, current, np.array(["t2m", "msl"]))


def test_filters_initial_yield_and_writes_complete_uneven_minibatches(setup):
    path, loaded, coords, writes, registrations, _, finalizes = setup
    sink = storage.ZarrEnsembleSink(path / "ensemble.zarr", loaded)
    sink.add_array(coords, np.array(["t2m", "msl"]))
    for members in ([0, 1], [2]):
        for lead in (0, 6, 12):
            write_batch(sink, coords, members, lead)
    sink.assert_complete()
    sink.close()
    sink.close()
    assert len(writes) == 4 and finalizes == [True]
    np.testing.assert_array_equal(
        registrations[0][0]["lead_time"], np.array([6, 12], dtype="timedelta64[h]")
    )
    assert writes[-1][1]["ensemble"].tolist() == [2]
    assert all(
        isinstance(value, HostTensor) for values, _, _ in writes for value in values
    )
    np.testing.assert_array_equal(
        writes[0][0][0].to("cpu", non_blocking=False).numpy(),
        np.full((2, 1, 1, 2, 2), 6, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_member",
        "missing_step",
        "duplicate",
        "dtype",
        "wrong_coordinates",
        "shape",
    ],
)
def test_incomplete_or_invalid_writes_cannot_publish(setup, failure):
    path, loaded, coords, writes, *_ = setup
    sink = storage.ZarrEnsembleSink(path / "ensemble.zarr", loaded)
    sink.add_array(coords, np.array(["t2m", "msl"]))
    write_batch(sink, coords, [0, 1], 6)
    with pytest.raises(ValueError, match="coverage|duplicate|float32|coordinate|shape"):
        if failure == "missing_member":
            write_batch(sink, coords, [0, 1], 12)
            sink.assert_complete()
        elif failure == "missing_step":
            write_batch(sink, coords, [2], 6)
            sink.assert_complete()
        elif failure == "duplicate":
            write_batch(sink, coords, [1], 6)
        elif failure == "dtype":
            write_batch(sink, coords, [2], 6, dtype=np.float64)
        elif failure == "wrong_coordinates":
            bad = OrderedDict(coords)
            bad["lat"] = np.array([80.0, -80.0])
            write_batch(sink, bad, [2], 6)
        else:
            bad = OrderedDict(coords)
            bad["ensemble"] = np.array([2])
            bad["lead_time"] = np.array([6], dtype="timedelta64[h]")
            sink.write(
                [np.ones((1, 1, 1, 1, 2), dtype=np.float32)] * 2, bad, ["t2m", "msl"]
            )
    sink.close()  # Cleanup must not hide the original error with a coverage error.


def test_rust_capacity_covers_the_entire_accepted_ensemble(setup, monkeypatch):
    path, loaded, coords, _, _, options, _ = setup
    monkeypatch.setattr(
        sys.modules["e2s_workflow"], "_selected_zarr_backend", lambda: "rust"
    )
    loaded.contract.lat, loaded.contract.lon = np.arange(721), np.arange(1440)
    sink = storage.ZarrEnsembleSink(path / "rust.zarr", loaded)
    sink.close()
    settings = options[0]
    expected = 3 * 2 * 2 * 721 * 1440 * 4
    assert settings["max_pool_bytes"] == 256 * 1024**2
    assert settings["max_inflight_transient_bytes"] == expected + 256 * 1024**2
    assert settings["chunks"] == {"ensemble": 1, "time": 1, "lead_time": 1}


def test_backend_failure_finalizes_without_registering_a_write(setup):
    path, loaded, coords, _, _, _, _ = setup
    sink = storage.ZarrEnsembleSink(path / "bad.zarr", loaded)
    sink.add_array(coords, np.array(["t2m", "msl"]))
    sink.backend.write = lambda *a: (_ for _ in ()).throw(
        RuntimeError("storage failed")
    )
    with pytest.raises(RuntimeError, match="storage failed"):
        write_batch(sink, coords, [0, 1, 2], 6)
    with pytest.raises(ValueError, match="coverage"):
        sink.assert_complete()
    assert sink.field_hashes == {}
    sink.close()


def test_hashes_bind_field_bytes_before_the_backend_can_mutate_them(setup):
    path, loaded, coords, *_ = setup
    sink = storage.ZarrEnsembleSink(path / "checksummed.zarr", loaded)
    sink.add_array(coords, np.array(["t2m", "msl"]))

    def miswrite(data, _coords, _names):
        data[0][...] = -99

    sink.backend.write = miswrite
    write_batch(sink, coords, [0, 1, 2], 6)
    expected = hashlib.sha256(
        np.full((2, 2), 6, dtype=np.float32).tobytes()
    ).hexdigest()
    assert sink.field_hashes[(2, 1, "t2m")] == expected
    assert len(sink.field_hashes) == 6
    sink.close()
