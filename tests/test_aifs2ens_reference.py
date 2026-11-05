# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent stock ensemble reference: one RNG seed and batched member identity."""

from contextlib import nullcontext
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_gpu_reference",
    ROOT / "qa/plugins/earth2-aifs2ens/reference_earth2studio.py",
)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.mark.parametrize("missing_tail", [False, True])
def test_reference_calls_stock_ensemble_with_one_seed_and_preserves_all_member_slots(
    tmp_path, monkeypatch, missing_tail
):
    __import__("xarray")  # Import optional-version checks before metadata mocks.
    checkpoint_hash = "e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb"
    contract = {
        "checkpoint_sha256": checkpoint_hash,
        "input_variables": ["t2m"],
        "output_variables": ["t2m"],
        "lat": [90.0, -90.0],
        "lon": [0.0, 180.0],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    values = np.arange(8, dtype=np.float32).reshape(1, 2, 1, 2, 2)
    input_path = tmp_path / "input.npy"
    np.save(input_path, values)
    original_digest = REFERENCE.digest
    input_hash = original_digest(input_path)
    monkeypatch.setattr(
        REFERENCE,
        "digest",
        lambda path: (
            checkpoint_hash if str(path).endswith(".ckpt") else original_digest(path)
        ),
    )
    coords = OrderedDict(
        variable=np.array(["t2m"]),
        lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
        lat=np.array(contract["lat"]),
        lon=np.array(contract["lon"]),
    )

    class Projector:
        projection_matrix = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    projector = Projector()
    model = SimpleNamespace(
        seed=None,
        modules=lambda: [projector],
        input_coords=lambda: coords,
        output_coords=lambda c: {"variable": np.array(["t2m"])},
    )
    model.to = lambda device: model
    model.eval = lambda: model
    calls, seeds = [], []

    def load(*args, **kwargs):
        calls.append(("load", kwargs))
        return model

    def ensemble(**kwargs):
        calls.append(("ensemble", kwargs))
        assert kwargs["prognostic"].seed is None
        assert kwargs["nensemble"] == 3 and kwargs["batch_size"] == 2
        source = kwargs["data"]
        times = np.array(
            ["2026-11-04T06:00:00", "2026-11-04T12:00:00"], dtype="datetime64[ns]"
        )
        np.testing.assert_array_equal(
            source(times, np.array(["t2m"])).values, values[0]
        )
        output_coords = OrderedDict(
            ensemble=np.arange(3),
            time=np.array(["2026-11-04T12:00:00"], dtype="datetime64[ns]"),
            lead_time=np.array([0, 6, 12], dtype="timedelta64[h]"),
            lat=coords["lat"],
            lon=coords["lon"],
        )
        kwargs["io"].add_array(output_coords, np.array(["t2m"]))
        for members in ([0, 1],) if missing_tail else ([0, 1], [2]):
            for step in range(3):
                current = OrderedDict(output_coords)
                current["ensemble"] = np.array(members)
                current["lead_time"] = np.array([6 * step], dtype="timedelta64[h]")
                arrays = np.stack(
                    [
                        np.full((1, 1, 2, 2), 10 * i + step, dtype=np.float32)
                        for i in members
                    ]
                )
                if step:
                    projector.projection_matrix.device.type = "cuda"
                kwargs["io"].write([Tensor(arrays)], current, np.array(["t2m"]))

    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.auto",
        SimpleNamespace(Package=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.px",
        SimpleNamespace(AIFS2ENS=SimpleNamespace(load_model=load)),
    )
    monkeypatch.setitem(
        sys.modules, "earth2studio.run", SimpleNamespace(ensemble=ensemble)
    )
    monkeypatch.setitem(
        sys.modules, "earth2studio.perturbation", SimpleNamespace(Zero=lambda: object())
    )
    monkeypatch.setitem(
        sys.modules,
        "anemoi.models.layers.sparse_projector",
        SimpleNamespace(SparseProjector=Projector),
    )
    original_distribution = REFERENCE.importlib.metadata.distribution
    original_version = REFERENCE.importlib.metadata.version
    monkeypatch.setattr(
        REFERENCE.importlib.metadata,
        "distribution",
        lambda name: (
            SimpleNamespace(
                read_text=lambda name: json.dumps(
                    {
                        "vcs_info": {
                            "commit_id": "f8af4030cfd43ec79bf4c0cf6c23463a87bfef97"
                        }
                    }
                )
            )
            if name == "earth2studio"
            else original_distribution(name)
        ),
    )
    monkeypatch.setattr(
        REFERENCE.importlib.metadata,
        "version",
        lambda name: (
            "0.11.2"
            if name == "anemoi-models"
            else "1.0.0"
            if name == "flash-attn"
            else original_version(name)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda seed: seeds.append(seed),
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda: None,
                synchronize=lambda: None,
                max_memory_allocated=lambda: 0,
                max_memory_reserved=lambda: 0,
                get_device_name=lambda: "test",
            ),
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(
                        allow_tf32=True, allow_bf16_reduced_precision_reduction=True
                    )
                ),
                cudnn=SimpleNamespace(allow_tf32=True),
            ),
            are_deterministic_algorithms_enabled=lambda: False,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
        ),
    )
    output = tmp_path / "reference"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SPEC.origin),
            "--checkpoint-dir",
            str(tmp_path),
            "--input",
            str(input_path),
            "--input-sha256",
            input_hash,
            "--contract",
            str(contract_path),
            "--start-time",
            "2026-11-04T12:00:00Z",
            "--seed",
            "5",
            "--nsteps",
            "2",
            "--nensemble",
            "3",
            "--batch-size",
            "2",
            "--output",
            str(output),
        ],
    )
    if missing_tail:
        with pytest.raises(ValueError, match="coverage|missing|incomplete"):
            REFERENCE.main()
        assert not (output / "report.json").exists()
    else:
        REFERENCE.main()
        assert calls[0] == ("load", {"preload_invariants": False, "seed": None})
        assert seeds == [5]
        report = json.loads((output / "report.json").read_text())
        assert (
            report["seed_base"] == 5
            and report["nensemble"] == 3
            and report["batch_size"] == 2
        )
        assert (
            report["model_seed"] is None
            and report["rng_policy"] == "torch-seed-once-native-v1"
        )
        assert (
            report["noise_projection_device"] == "cuda"
            and report["model_adapters"] == []
        )
        first = np.load(output / "step-0001.npy")
        assert first.shape == (3, 1, 1, 1, 2, 2)
        np.testing.assert_array_equal(first[:, 0, 0, 0, 0, 0], [1, 11, 21])
        assert [row["step"] for row in report["steps"]] == [1, 2]
        assert (
            report["steps"][0]["sha256"]
            == hashlib.sha256((output / "step-0001.npy").read_bytes()).hexdigest()
        )


def test_capture_accepts_relative_asset_and_output_paths(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "relative_path_input_capture",
        ROOT / "qa/plugins/earth2-aifs2ens/capture_earth2studio_inputs.py",
    )
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    monkeypatch.chdir(tmp_path)
    original = tmp_path / "old-assets"
    assets = original / "assets"
    checkpoint = assets / "checkpoint-xet"
    checkpoint.mkdir(parents=True)
    (checkpoint / "aifs-ens-crps-2.0.ckpt").write_bytes(b"checkpoint fixture")
    fixed = np.broadcast_to(
        np.arange(5, dtype=np.float32)[:, None, None], (5, 721, 1440)
    )
    np.save(assets / "invariants.npy", fixed, allow_pickle=False)
    contract = {
        "input_variables": ["lsm", "t2m", "sdor", "slor", "z", "wmb"],
        "lat": np.linspace(90, -90, 721).tolist(),
        "lon": (np.arange(1440) / 4).tolist(),
        "invariants_sha256": capture.digest(assets / "invariants.npy"),
    }
    (original / "qualified-input-contract.json").write_text(json.dumps(contract))
    (original / "weather-capture-report.json").write_text(
        json.dumps({"initialization": "2026-11-04T12:00:00Z"})
    )
    weather = np.broadcast_to(
        np.array([10, 20], dtype=np.float32)[None, :, None, None, None],
        (1, 2, 1, 721, 1440),
    )
    calls = []

    def fetch_data(**kwargs):
        calls.append(kwargs)
        coords = {key: kwargs[key] for key in ("time", "lead_time", "variable")}
        coords.update(lat=np.array(contract["lat"]), lon=np.array(contract["lon"]))
        return SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: weather)
        ), coords

    monkeypatch.setitem(
        sys.modules,
        "earth2studio.data",
        SimpleNamespace(IFS_ENS=lambda **kwargs: kwargs, fetch_data=fetch_data),
    )
    distribution = capture.importlib.metadata.distribution
    monkeypatch.setattr(
        capture.importlib.metadata,
        "distribution",
        lambda name: (
            SimpleNamespace(
                read_text=lambda _: json.dumps(
                    {
                        "vcs_info": {
                            "commit_id": "f8af4030cfd43ec79bf4c0cf6c23463a87bfef97"
                        }
                    }
                )
            )
            if name == "earth2studio"
            else distribution(name)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(spec.origin),
            "--original-assets",
            "old-assets",
            "--output",
            "new-fixture",
            "--ifs-member-id",
            "3",
        ],
    )
    capture.main()
    output = tmp_path / "new-fixture"
    for name in ("invariants.npy", "checkpoint-xet"):
        linked = output / "assets" / name
        assert linked.is_symlink() and linked.resolve() == assets / name
    assert (output / "assets/checkpoint-xet/aifs-ens-crps-2.0.ckpt").read_bytes() == (
        checkpoint / "aifs-ens-crps-2.0.ckpt"
    ).read_bytes()
    assert (
        json.loads((output / "qualified-input-contract.json").read_text()) == contract
    )
    assert len(calls) == 1
    assert calls[0]["source"] == {"member": 3, "source": "ecmwf", "verbose": False}
    assert calls[0]["device"] == "cpu"
    np.testing.assert_array_equal(calls[0]["variable"], ["t2m"])
    np.testing.assert_array_equal(
        calls[0]["lead_time"], np.array([-6, 0], "timedelta64[h]")
    )
    report = json.loads((output / "weather-capture-report.json").read_text())
    assert report["status"] == "passed" and report["ifs_member_id"] == 3
    assert report["shared_input"] is True
    assert report["preprocessing_policy"] == capture.PREPROCESSING_POLICY
    captured = np.load(report["input"]["path"], allow_pickle=False, mmap_mode="r")
    assert captured.shape == (1, 2, 6, 721, 1440) and captured.dtype == np.float32
    np.testing.assert_array_equal(captured[:, :, 1:2], weather)
    for index, name in enumerate(capture.INVARIANTS):
        np.testing.assert_array_equal(
            captured[0, :, contract["input_variables"].index(name)],
            np.broadcast_to(fixed[index], (2, 721, 1440)),
        )
    assert report["input"]["sha256"] == capture.digest(output / "assets/input.npy")
