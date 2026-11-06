# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable AIFS2ENS regressions, grouped by component with isolated fixtures."""

import json
import sys
import numpy as np
import pytest
import copy
import importlib.util
import xarray as xr
import yaml
import hashlib
import threading
import weakref
import signal
import subprocess
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from pydantic import ValidationError
from types import SimpleNamespace
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins/earth2-aifs2ens"))

import aifs2ens_outputs as outputs  # noqa: E402
import aifs2ens_runtime as runtime  # noqa: E402
import aifs2ens_rollout as rollout  # noqa: E402
from aifs2ens_inputs import (  # noqa: E402
    ForecastRequest,
    INVARIANTS,
    InputContract,
    WeatherInput,
    WeatherSource,
    load_ensemble_input,
    sha256_file,
    stage_inputs,
)
from plugin_runtime import expand_plugin_manifest  # noqa: E402
from plugin_sdk import (  # noqa: E402
    PluginCancelledError,
    PrepareContext,
    RawRequest,
    workflow_request_schema,
)


# Artifacts


def _artifact_request(**changes):
    return ForecastRequest.model_validate(
        {
            "start_time": "2026-11-03T00:00:00Z",
            "ifs_member_id": 7,
            "nensemble": 2,
            "nsteps": 4,
            "seed_base": 0,
            **changes,
        }
    )


def _artifact_weather(contract, member):
    values = np.zeros((1, 2, len(contract.input_variables), 2, 3), dtype=np.float32)
    values[:, :, :2] = member if member is not None else 42
    return WeatherInput(values=values, provenance={"source_member": member})


class TestArtifacts:
    "A frozen common input must bind the whole ensemble and exact numerical bytes."

    @pytest.fixture
    def contract(self):
        return InputContract.model_validate(
            {
                "schema_version": 1,
                "checkpoint_sha256": "a" * 64,
                "input_variables": ["t2m", "msl", "lsm", "sdor", "slor", "z", "wmb"],
                "output_variables": ["t2m", "msl", "u10m", "v10m"],
                "lat": [90.0, -90.0],
                "lon": [0.0, 120.0, 240.0],
            }
        )

    def test_common_input_is_fetched_once_for_the_entire_ensemble(
        self, tmp_path, contract
    ):
        calls = []

        def fetch(req, shape):
            calls.append((req.start_time, req.ifs_member_id))
            return _artifact_weather(shape, req.ifs_member_id)

        staged = stage_inputs(
            _artifact_request(), contract, tmp_path, fetch, max_output_bytes=10**6
        )
        assert calls == [("2026-11-03T00:00:00Z", 7)]
        loaded = load_ensemble_input(staged["manifest_path"], staged["manifest_sha256"])
        assert loaded.input_sha256 == sha256_file(
            Path(staged["manifest_path"]).parent / "input.npy"
        )
        assert loaded.request.nensemble == 2
        assert loaded.request.batch_size == 2
        assert loaded.request.seed_base == 0
        np.testing.assert_array_equal(
            loaded.values, _artifact_weather(contract, 7).values
        )
        assert list(loaded.coords) == ["time", "lead_time", "variable", "lat", "lon"]
        np.testing.assert_array_equal(
            loaded.coords["lead_time"], np.array([-6, 0], dtype="timedelta64[h]")
        )
        assert loaded.coords["time"][0] == np.datetime64("2026-11-03T00:00:00")
        manifest = json.loads(Path(staged["manifest_path"]).read_text())
        assert "members" not in manifest
        assert manifest["provenance"] == {"source_member": 7}

    def test_storage_budget_rejects_before_weather_download(self, tmp_path, contract):
        def fetch(*args):
            pytest.fail("weather must not be downloaded for an over-budget request")

        with pytest.raises(ValueError, match="output.*budget"):
            stage_inputs(
                _artifact_request(), contract, tmp_path, fetch, max_output_bytes=1
            )

    def test_unknown_output_variable_rejects_before_weather_download(
        self, tmp_path, contract
    ):
        with pytest.raises(ValueError, match="output variable"):
            stage_inputs(
                _artifact_request(output_variables=["not_a_model_field"]),
                contract,
                tmp_path,
                lambda *args: pytest.fail("downloaded"),
                max_output_bytes=10**6,
            )

    @pytest.mark.parametrize("change", ["bytes", "manifest", "contract"])
    def test_tampering_blocks_ensemble_execution(self, tmp_path, contract, change):
        staged = stage_inputs(
            _artifact_request(),
            contract,
            tmp_path,
            lambda req, shape: _artifact_weather(shape, req.ifs_member_id),
            max_output_bytes=10**6,
        )
        manifest_path = Path(staged["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        target = {
            "bytes": manifest_path.parent / manifest["input_path"],
            "manifest": manifest_path,
            "contract": manifest_path.parent / manifest["contract_path"],
        }[change]
        with target.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="digest"):
            load_ensemble_input(str(manifest_path), staged["manifest_sha256"])

    @pytest.mark.parametrize("bad", ["shape", "dtype", "invariants"])
    def test_incompatible_inputs_cannot_emit_a_valid_manifest(
        self, tmp_path, contract, bad
    ):
        fetched = _artifact_weather(contract, 7)
        if bad == "shape":
            fetched.values = fetched.values[:, :1]
        elif bad == "dtype":
            fetched.values = fetched.values.astype(np.float64)
        else:
            fetched.values[0, 1, -1, 0, 0] = 1
        with pytest.raises(ValueError, match="shape|float32|invariant"):
            stage_inputs(
                _artifact_request(),
                contract,
                tmp_path,
                lambda *args: fetched,
                max_output_bytes=10**6,
            )
        assert not list(tmp_path.rglob("manifest.json"))

    def test_two_parent_attempts_cannot_overwrite_staged_input_bytes(
        self, tmp_path, contract
    ):
        def fetch(req, shape):
            return _artifact_weather(shape, req.ifs_member_id)

        first = stage_inputs(
            _artifact_request(), contract, tmp_path, fetch, max_output_bytes=10**6
        )
        second = stage_inputs(
            _artifact_request(), contract, tmp_path, fetch, max_output_bytes=10**6
        )
        assert first["manifest_path"] != second["manifest_path"]
        assert (
            load_ensemble_input(
                first["manifest_path"], first["manifest_sha256"]
            ).request.seed_base
            == 0
        )


# Contract


def _contract_request(**changes):
    return {
        "start_time": "2026-11-03T00:00:00Z",
        "nsteps": 4,
        "nensemble": 2,
        "seed_base": 0,
        **changes,
    }


class TestContract:
    "One shared initialization and one seed for a complete GPU ensemble."

    def test_default_batches_all_members_with_one_seed_and_one_common_source(self):
        parsed = ForecastRequest.model_validate(_contract_request())
        assert parsed.seed_base == 0
        assert parsed.batch_size == 2
        assert parsed.ifs_member_id is None
        assert not {
            "max_in_flight",
            "ifs_member_ids",
            "initial_condition_mode",
        }.intersection(ForecastRequest.model_fields)

    def test_common_ifs_ensemble_source_is_shared_by_every_output_member(self):
        parsed = ForecastRequest.model_validate(
            _contract_request(ifs_member_id=7, nensemble=5)
        )
        assert parsed.ifs_member_id == 7
        assert parsed.nensemble == parsed.batch_size == 5
        assert parsed.seed_base == 0

    @pytest.mark.parametrize("size,expected", [(None, 2), (1, 1), (2, 2), (8, 2)])
    def test_batch_size_follows_stock_ensemble_normalization(self, size, expected):
        assert (
            ForecastRequest.model_validate(
                _contract_request(batch_size=size)
            ).batch_size
            == expected
        )

    def test_same_seed_and_requested_layout_are_preserved_for_every_horizon(self):
        assert (
            ForecastRequest.model_validate(_contract_request(nsteps=60)).seed_base == 0
        )
        assert (
            ForecastRequest.model_validate(
                _contract_request(seed_base=2**63 - 1)
            ).seed_base
            == 2**63 - 1
        )

    def test_normalizes_utc_and_requires_six_hour_initialization(self):
        parsed = ForecastRequest.model_validate(
            _contract_request(start_time="2026-11-03T01:00:00+01:00")
        )
        assert parsed.start_time == "2026-11-03T00:00:00Z"

    @pytest.mark.parametrize(
        "changes",
        [
            {"start_time": "2026-11-03T00:00:00"},
            {"start_time": "2026-11-03T01:00:00Z"},
            {"start_time": "2026-11-03T00:00:01Z"},
            {"start_time": "not-a-date"},
            {"nsteps": 0},
            {"nsteps": 61},
            {"nsteps": True},
            {"nsteps": "4"},
            {"nensemble": 0},
            {"nensemble": 17},
            {"nensemble": True},
            {"batch_size": 0},
            {"batch_size": True},
            {"batch_size": "2"},
            {"batch_size": 17},
            {"ifs_member_id": 0},
            {"ifs_member_id": 51},
            {"ifs_member_id": True},
            {"seed_base": -1},
            {"seed_base": 2**63},
            {"seed_base": True},
            {"output_variables": []},
            {"output_variables": ["t2m", "t2m"]},
            {"output_variables": ["../input"]},
            {"ifs_member_ids": [1, 2]},
            {"initial_condition_mode": "shared_ifs"},
            {"max_in_flight": 1},
            {"checkpoint": "/caller/checkpoint"},
        ],
    )
    def test_rejects_invalid_and_obsolete_fanout_requests(self, changes):
        with pytest.raises(ValidationError):
            ForecastRequest.model_validate(_contract_request(**changes))

    def test_public_schema_rejects_unknown_fields(self):
        schema = ForecastRequest.model_json_schema()
        assert schema["additionalProperties"] is False
        assert "batch_size" in schema["properties"]
        assert "manifest_path" not in schema["properties"]


# Data


class TestData:
    "Datasource preparation and invariant binding without remote data downloads."

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        fixed = np.ones((5, 2, 2), dtype=np.float32)
        fixed[0, 0, 0] = 0  # ocean in the frozen land mask
        path = tmp_path / "invariants.npy"
        np.save(path, fixed)
        monkeypatch.setenv("AIFS2ENS_INVARIANTS", str(path))
        contract = InputContract.model_validate(
            {
                "schema_version": 1,
                "checkpoint_sha256": "a" * 64,
                "invariants_sha256": sha256_file(path),
                "input_variables": ["t2m", "sd", "swvl1", "swvl2", *INVARIANTS],
                "output_variables": ["t2m"],
                "lat": [90.0, -90.0],
                "lon": [0.0, 180.0],
            }
        )
        req = ForecastRequest.model_validate(
            {
                "start_time": "2026-11-03T00:00:00Z",
                "nensemble": 1,
                "nsteps": 1,
                "ifs_member_id": 7,
                "output_variables": ["t2m"],
            }
        )
        return req, contract, path, fixed

    def test_preserves_earth2studio_values_with_selected_member_and_frozen_invariants(
        self,
        setup,
    ):
        req, contract, path, fixed = setup
        calls = []

        def fetch(request, variables):
            calls.append((request.ifs_member_id, list(variables)))
            coords = contract.coords(request)
            coords["variable"] = np.array(variables)
            return np.full((1, 2, 4, 2, 2), 7, dtype=np.float32), coords

        result = WeatherSource(fetch=fetch)(req, contract)
        assert calls == [(7, ["t2m", "sd", "swvl1", "swvl2"])]
        for i, name in enumerate(INVARIANTS):
            actual = result.values[:, :, contract.input_variables.index(name)]
            np.testing.assert_array_equal(
                actual, np.broadcast_to(fixed[i], (1, 2, 2, 2))
            )
        assert result.values[0, 0, 1, 0, 0] == 7
        assert result.values[0, 0, 2, 0, 0] == 7
        assert result.values[0, 0, 0, 0, 0] == 7
        assert result.provenance["ifs_member_id"] == 7
        assert result.provenance["invariants_sha256"] == sha256_file(path)

    def test_rejects_changed_invariants_before_requesting_weather(self, setup):
        req, contract, path, fixed = setup
        np.save(path, fixed + 1)
        with pytest.raises(ValueError, match="invariant.*digest"):
            WeatherSource(fetch=lambda *a: pytest.fail("downloaded weather"))(
                req, contract
            )

    def test_data_coordinate_order_must_match_the_model_contract(self, setup):
        req, contract, path, fixed = setup

        def fetch(request, variables):
            coords = contract.coords(request)
            coords["variable"] = np.array(variables)
            coords["lead_time"] = coords["lead_time"][::-1]
            return np.zeros((1, 2, 4, 2, 2), dtype=np.float32), coords

        with pytest.raises(ValueError, match="lead_time"):
            WeatherSource(fetch=fetch)(req, contract)

    def test_datasource_rejects_unpinned_runtime_before_loading_sources(
        self, monkeypatch
    ):
        import aifs2ens_runtime

        def reject():
            raise RuntimeError("install the pinned Earth2Studio source revision")

        monkeypatch.setattr(aifs2ens_runtime, "_verify_runtime", reject)
        with pytest.raises(RuntimeError, match="pinned Earth2Studio"):
            WeatherSource._fetch(None, [])


# Native Parity

_PARITY_PATH = (
    Path(__file__).resolve().parents[1] / "qa/plugins/earth2-aifs2ens/native_parity.py"
)


_PARITY_SPEC = importlib.util.spec_from_file_location(
    "aifs2ens_native_parity", _PARITY_PATH
)


_parity = importlib.util.module_from_spec(_PARITY_SPEC)


_PARITY_SPEC.loader.exec_module(_parity)


def _write_parity_reference(root, values, *, batch_size=2):
    import json

    root.mkdir()
    array = np.stack(
        [np.full((1, 1, 1, 2, 2), value, dtype=np.float32) for value in values]
    )
    np.save(root / "step-0001.npy", array)
    (root / "report.json").write_text(
        json.dumps(
            {
                "output_variables": ["t2m"],
                "steps": [{"step": 1}],
                "nensemble": len(values),
                "nsteps": 1,
                "batch_size": batch_size,
                "seed_base": 0,
                "model_seed": None,
                "rng_policy": "torch-seed-once-native-v1",
            }
        )
    )
    return root


class TestNativeParity:
    "Native CUDA variation is measured before candidate evaluation, never hidden."

    def test_identical_nonfinite_masks_and_values_have_zero_error(self):
        values = np.array([1, np.nan, np.inf, -np.inf], dtype=np.float32)
        metrics = _parity.field_metrics(values.copy(), values)
        assert metrics["max_abs"] == metrics["rmse"] == 0
        assert metrics["finite_count"] == 1

    @pytest.mark.parametrize(
        "mismatch", ["shape", "dtype", "nan_mask", "infinity_sign"]
    )
    def test_structural_or_nonfinite_differences_cannot_be_tolerated(self, mismatch):
        expected = np.array([1, np.nan, np.inf], dtype=np.float32)
        actual = expected.copy()
        if mismatch == "shape":
            actual = actual[None]
        elif mismatch == "dtype":
            actual = actual.astype(np.float64)
        elif mismatch == "nan_mask":
            actual[1] = 0
        else:
            actual[2] = -np.inf
        with pytest.raises(ValueError):
            _parity.field_metrics(actual, expected)

    def test_reference_variation_sets_frozen_bounds_that_reject_a_changed_candidate(
        self,
    ):
        samples = [
            np.full((2, 2), value, dtype=np.float32)
            for value in (10, 10.01, 9.99, 10.005)
        ]
        limits = _parity.calibrate_arrays(samples)
        original = copy.deepcopy(limits)
        assert limits["margin"] == 2
        assert limits["max_abs_limit"] == 2 * limits["observed_max_abs"]
        assert _parity.compare_field(
            np.full((2, 2), 10.015, dtype=np.float32), samples[0], limits
        )["passed"]
        assert not _parity.compare_field(
            np.full((2, 2), 10.25, dtype=np.float32), samples[0], limits
        )["passed"]
        assert limits == original  # A failed candidate cannot change its threshold.

    def test_bitwise_repeatable_fields_retain_zero_tolerance(self):
        expected = np.ones((2, 2), dtype=np.float32)
        limits = _parity.calibrate_arrays([expected.copy() for _ in range(4)])
        assert limits["max_abs_limit"] == limits["rmse_limit"] == 0
        changed = expected.copy()
        changed[0, 0] = np.nextafter(np.float32(1), np.float32(2))
        assert not _parity.compare_field(changed, expected, limits)["passed"]

    def test_insufficient_reference_calibration_is_rejected(self):
        with pytest.raises(ValueError, match="four"):
            _parity.calibrate_arrays([np.ones(2, dtype=np.float32)] * 3)

    def test_reference_limits_are_per_member_and_reject_swapped_member_slots(
        self, tmp_path
    ):
        roots = [
            _write_parity_reference(tmp_path / str(i), values)
            for i, values in enumerate(
                (
                    (10, 20),
                    (10.01, 20.01),
                    (9.99, 19.99),
                    (10.005, 20.005),
                    (10.015, 20.015),
                    (20, 10),
                )
            )
        ]
        limits = _parity.calibrate_ensemble(roots[:4])
        assert limits["nensemble"] == 2
        assert set(limits["members"]) == {"0", "1"}
        assert limits["members"]["0"]["steps"]["1"]["t2m"]["max_abs_limit"] < 0.1
        assert (
            _parity.validate_holdouts(roots[0], [roots[4]], limits)["status"]
            == "passed"
        )
        failed = _parity.validate_holdouts(roots[0], [roots[5]], limits)
        assert failed["status"] == "failed"
        assert {row["member_index"] for row in failed["failures"]} == {0, 1}

    def test_reference_calibration_cannot_mix_batch_layouts(self, tmp_path):
        roots = [
            _write_parity_reference(
                tmp_path / str(i), (10, 20), batch_size=1 if i == 3 else 2
            )
            for i in range(4)
        ]
        with pytest.raises(ValueError, match="layout|identity|batch"):
            _parity.calibrate_ensemble(roots)


# Outputs


def _output_dataset():
    return xr.Dataset(
        {
            "t2m": (
                ("ensemble", "time", "lead_time", "lat", "lon"),
                np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2),
            )
        },
        coords={
            "ensemble": [0, 1],
            "time": np.array(["2026-11-04T12:00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array([6, 12], dtype="timedelta64[h]"),
            "lat": [90.0, -90.0],
            "lon": [0.0, 180.0],
        },
    )


def _output_loaded():
    return SimpleNamespace(
        request=SimpleNamespace(
            nensemble=2,
            nsteps=2,
            start_time="2026-11-04T12:00:00Z",
            output_variables=["t2m"],
        ),
        contract=SimpleNamespace(lat=[90.0, -90.0], lon=[0.0, 180.0]),
    )


def _output_field_hashes(ds):
    return {
        (member, step + 1, "t2m"): outputs.field_checksum(
            ds.t2m.values[member, 0, step]
        )
        for member in range(2)
        for step in range(2)
    }


class TestOutputs:
    "A single final ensemble requires exact field, member and coordinate coverage."

    def test_complete_single_dataset_preserves_members_and_forecast_fields(
        self, tmp_path
    ):
        path = tmp_path / "ensemble.zarr"
        ds = _output_dataset()
        ds.to_zarr(path, zarr_format=3, consolidated=False)
        outputs.validate_ensemble_output(path, _output_loaded())
        with xr.open_zarr(path, consolidated=False) as actual:
            np.testing.assert_array_equal(actual.t2m.values, ds.t2m.values)

    @pytest.mark.parametrize(
        "failure",
        [
            "missing_member",
            "reordered_members",
            "missing_step",
            "initial_step",
            "variable",
            "dtype",
            "dims",
            "lat",
            "time",
        ],
    )
    def test_invalid_final_dataset_cannot_be_registered(self, tmp_path, failure):
        ds = _output_dataset()
        if failure == "missing_member":
            ds = ds.isel(ensemble=[0])
        elif failure == "reordered_members":
            ds = ds.isel(ensemble=[1, 0])
        elif failure == "missing_step":
            ds = ds.isel(lead_time=[0])
        elif failure == "initial_step":
            ds = ds.assign_coords(lead_time=np.array([0, 6], dtype="timedelta64[h]"))
        elif failure == "variable":
            ds = ds.rename(t2m="msl")
        elif failure == "dtype":
            ds["t2m"] = ds.t2m.astype(np.float64)
        elif failure == "dims":
            ds["t2m"] = ds.t2m.transpose("time", "ensemble", "lead_time", "lat", "lon")
        elif failure == "lat":
            ds = ds.assign_coords(lat=[80.0, -80.0])
        else:
            ds = ds.assign_coords(
                time=np.array(["2026-11-04T06:00:00"], dtype="datetime64[ns]")
            )
        path = tmp_path / "invalid.zarr"
        ds.to_zarr(path, zarr_format=3, consolidated=False)
        with pytest.raises(ValueError, match="coordinate|variables|dimensions|dtype"):
            outputs.validate_ensemble_output(path, _output_loaded())

    def test_corrupt_forecast_chunk_is_read_before_publication(self, tmp_path):
        path = tmp_path / "corrupt.zarr"
        _output_dataset().to_zarr(path, zarr_format=3, consolidated=False)
        chunk = next(p for p in (path / "t2m" / "c").rglob("*") if p.is_file())
        chunk.write_bytes(b"corrupt compressed forecast bytes")
        with pytest.raises(ValueError, match="read"):
            outputs.validate_ensemble_output(path, _output_loaded())

    @pytest.mark.parametrize(
        "failure", ["deleted_chunk", "miswritten_values", "missing_hash"]
    )
    def test_stored_values_must_match_successful_writer_hashes(self, tmp_path, failure):
        import zarr

        path = tmp_path / "miswritten.zarr"
        ds = _output_dataset()
        hashes = _output_field_hashes(ds)
        ds.to_zarr(
            path,
            zarr_format=3,
            consolidated=False,
            encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
        )
        if failure == "deleted_chunk":
            chunk = next(p for p in (path / "t2m" / "c").rglob("*") if p.is_file())
            chunk.unlink()
        elif failure == "miswritten_values":
            array = zarr.open_group(path, mode="a")["t2m"]
            array[1, 0, 1] = np.zeros((2, 2), dtype=np.float32)
        else:
            hashes.pop((1, 2, "t2m"))
        with pytest.raises(ValueError, match="checksum"):
            outputs.validate_ensemble_output(
                path, _output_loaded(), expected_hashes=hashes
            )

    def test_sparse_all_fill_forecasts_can_match_written_hashes_without_chunks(
        self, tmp_path
    ):
        path = tmp_path / "all-fill.zarr"
        ds = _output_dataset()
        ds["t2m"] = xr.full_like(ds.t2m, np.float32(np.nan))
        ds.to_zarr(
            path,
            zarr_format=3,
            consolidated=False,
            write_empty_chunks=False,
            encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
        )
        assert not any(p.is_file() for p in (path / "t2m" / "c").rglob("*"))
        outputs.validate_ensemble_output(
            path, _output_loaded(), expected_hashes=_output_field_hashes(ds)
        )

    def test_noncanonical_nan_fill_chunks_match_values_after_elision(self, tmp_path):
        path = tmp_path / "nan-payload-fill.zarr"
        ds = _output_dataset()
        ds["t2m"].data = np.full(ds.t2m.shape, 0xFFC00001, dtype=np.uint32).view(
            np.float32
        )
        hashes = _output_field_hashes(ds)
        ds.to_zarr(
            path,
            zarr_format=3,
            consolidated=False,
            write_empty_chunks=False,
            encoding={"t2m": {"chunks": (1, 1, 1, 2, 2)}},
        )
        assert not any(p.is_file() for p in (path / "t2m" / "c").rglob("*"))
        outputs.validate_ensemble_output(path, _output_loaded(), expected_hashes=hashes)

    @pytest.mark.parametrize(
        "left_bits,right_bits,equal",
        [
            ([0xFFC00001], [0x7FC00000], True),
            ([0x80000000, 0], [0, 0], True),
            ([0x7F800000], [0xFF800000], False),
            ([0x3F800000], [0x3F800001], False),
            ([0xFFC00001, 0], [0, 0xFFC00001], False),
        ],
    )
    def test_field_checksum_normalizes_only_nan_payloads_and_signed_zero(
        self, left_bits, right_bits, equal
    ):
        left = np.array(left_bits, dtype=np.uint32).view(np.float32)
        right = np.array(right_bits, dtype=np.uint32).view(np.float32)
        original = left.view(np.uint32).copy()
        assert (outputs.field_checksum(left) == outputs.field_checksum(right)) is equal
        np.testing.assert_array_equal(left.view(np.uint32), original)


# Plugin

_PLUGIN_DIR = ROOT / "plugins" / "earth2-aifs2ens"


def _plugin_request_fields():
    return {
        "start_time": "2026-11-03T00:00:00Z",
        "nsteps": 4,
        "nensemble": 2,
        "ifs_member_id": 7,
        "seed_base": 0,
    }


class TestPlugin:
    "Framework integration contracts; real model parity is a separate H100 gate."

    @pytest.fixture
    def workflow_module(self):
        spec = importlib.util.spec_from_file_location(
            "aifs2ens_plugin_workflow", _PLUGIN_DIR / "workflow.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_manifest_runs_one_gpu_task_for_the_complete_ensemble(self):
        manifest = expand_plugin_manifest(
            yaml.safe_load((_PLUGIN_DIR / "plugin.yaml").read_text())
        )
        stages = manifest["pipeline"]["stages"]
        assert [s["phase"] for s in stages] == [
            "prepare",
            "prefetch",
            "schedule",
            "execute",
            "results",
        ]
        assert stages[0]["next"] == "prefetch"
        assert stages[3]["queue"] == "execute.earth2-gpu"
        assert stages[3]["next"] == "results"
        for key in (
            "executor_class",
            "prepare_executor_class",
            "postprocess_executor_class",
            "readiness_executor_class",
        ):
            assert manifest["runtime"][key] == "earth2-gpu"

    def test_manifest_declares_the_result_contract_required_by_the_rest_server(self):
        from jsonschema import Draft202012Validator

        manifest = yaml.safe_load((_PLUGIN_DIR / "plugin.yaml").read_text())
        schema = manifest["outputs"]["result_schema_inline"]
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(
            {
                "dataset_path": "/outputs/forecast.zarr",
                "complete": True,
                "nensemble": 2,
                "nsteps": 4,
                "ifs_member_id": 1,
                "seed_base": 0,
                "batch_size": 2,
                "rng_policy": "torch-seed-once-native-v1",
                "numerical_policy": "earth2studio-native-v1",
            }
        )
        validator.validate({"complete": False, "error": "incomplete ensemble"})
        assert schema["properties"]["dataset_path"]["type"] == "string"
        assert schema["properties"]["complete"]["type"] == "boolean"

    def test_prepare_downloads_once_for_one_gpu_request_without_loading_a_model(
        self, tmp_path, workflow_module, monkeypatch
    ):
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
        monkeypatch.setattr(
            workflow, "_load_model", lambda *a: pytest.fail("loaded model")
        )
        staged = []

        def stage(inputs, ctx):
            staged.append((inputs.seed_base, ctx.run_dir))
            return {
                "manifest_path": str(ctx.run_dir / "manifest.json"),
                "manifest_sha256": "b" * 64,
            }

        monkeypatch.setattr(workflow, "_stage_inputs", stage)
        raw = RawRequest(
            content_type="application/json",
            operation="run",
            raw_fields=_plugin_request_fields(),
        )
        prepared = workflow.prepare(
            raw,
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )
        assert staged == [(0, tmp_path)]
        assert prepared["operation"] == "forecast_ensemble"
        assert prepared["resource_profile"]["executor_class"] == "earth2-gpu"
        assert prepared["resource_profile"]["device_kind"] == "gpu"
        assert prepared["resource_profile"]["gpus_required"] == 1
        assert prepared["resource_profile"]["memory_mb"] == 65536
        assert set(prepared["resource_profile"]["tags"]) == {
            "earth2",
            "aifs2ens",
            "gpu",
        }
        assert prepared["batch_profile"]["enabled"] is False
        assert set(prepared["parameters"]) == {"manifest_path", "manifest_sha256"}
        assert "fanout_items" not in prepared and "fanout_profile" not in prepared
        schema = workflow_request_schema(workflow)
        assert schema["additionalProperties"] is False
        assert "manifest_path" not in schema["properties"]

    def test_prepare_keeps_downloader_output_off_the_hook_json_channel(
        self, tmp_path, workflow_module, monkeypatch, capsys
    ):
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")

        def stage(*args):
            print("downloading IFS")
            return {
                "manifest_path": "/shared/manifest.json",
                "manifest_sha256": "b" * 64,
            }

        monkeypatch.setattr(workflow, "_stage_inputs", stage)
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields=_plugin_request_fields(),
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )
        output = capsys.readouterr()
        assert output.out == ""
        assert "downloading IFS" in output.err

    @pytest.mark.parametrize("changes", [{"nensemble": 3}, {"nensemble": 16}])
    def test_default_batch_limit_rejects_before_downloading(
        self, tmp_path, workflow_module, monkeypatch, changes
    ):
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
        monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(
            workflow,
            "_stage_inputs",
            lambda *a: pytest.fail("downloaded oversized batch"),
        )
        with pytest.raises(ValueError, match="batch_size.*qualified maximum 2"):
            workflow.prepare(
                RawRequest(
                    content_type="application/json",
                    operation="run",
                    raw_fields={**_plugin_request_fields(), **changes},
                ),
                PrepareContext(
                    run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
                ),
            )

    @pytest.mark.parametrize(
        "nensemble,batch_size,limit",
        [(3, 2, None), (16, 1, None), (16, 2, None), (1, 16, None), (3, 3, "3")],
    )
    def test_batch_limit_preserves_the_effective_upstream_layout(
        self, tmp_path, workflow_module, monkeypatch, nensemble, batch_size, limit
    ):
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
        monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
        if limit is not None:
            monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
        workflow = workflow_module.WORKFLOW()
        staged = []
        monkeypatch.setattr(
            workflow,
            "_stage_inputs",
            lambda inputs, ctx: (
                staged.append((inputs.nensemble, inputs.batch_size))
                or {
                    "manifest_path": "/shared/manifest.json",
                    "manifest_sha256": "b" * 64,
                }
            ),
        )
        workflow.prepare(
            RawRequest(
                content_type="application/json",
                operation="run",
                raw_fields={
                    **_plugin_request_fields(),
                    "nensemble": nensemble,
                    "batch_size": batch_size,
                },
            ),
            PrepareContext(
                run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
            ),
        )
        assert staged == [(nensemble, min(nensemble, batch_size))]

    @pytest.mark.parametrize("limit,batch_size", [("1", 2), ("3", 4)])
    def test_configured_batch_limit_rejects_before_downloading(
        self, tmp_path, workflow_module, monkeypatch, limit, batch_size
    ):
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
        monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(
            workflow,
            "_stage_inputs",
            lambda *a: pytest.fail("downloaded oversized batch"),
        )
        with pytest.raises(ValueError, match=f"qualified maximum {limit}"):
            workflow.prepare(
                RawRequest(
                    content_type="application/json",
                    operation="run",
                    raw_fields={
                        **_plugin_request_fields(),
                        "nensemble": 4,
                        "batch_size": batch_size,
                    },
                ),
                PrepareContext(
                    run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
                ),
            )

    @pytest.mark.parametrize("limit", ["", "0", "-1", "invalid", "1.5"])
    def test_invalid_batch_limit_blocks_downloads(
        self, tmp_path, workflow_module, monkeypatch, limit
    ):
        monkeypatch.setenv("AIFS2ENS_GPU_MEMORY_MB", "65536")
        monkeypatch.setenv("AIFS2ENS_MAX_BATCH_SIZE", limit)
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(
            workflow,
            "_stage_inputs",
            lambda *a: pytest.fail("downloaded with invalid limit"),
        )
        with pytest.raises(RuntimeError, match="AIFS2ENS_MAX_BATCH_SIZE"):
            workflow.prepare(
                RawRequest(
                    content_type="application/json",
                    operation="run",
                    raw_fields=_plugin_request_fields(),
                ),
                PrepareContext(
                    run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
                ),
            )

    def test_staged_request_cannot_bypass_the_execution_batch_limit(
        self, tmp_path, workflow_module, monkeypatch
    ):
        from aifs2ens_inputs import ForecastRequest

        monkeypatch.delenv("AIFS2ENS_MAX_BATCH_SIZE", raising=False)
        loaded = SimpleNamespace(
            request=ForecastRequest.model_validate(
                {**_plugin_request_fields(), "nensemble": 3}
            )
        )
        monkeypatch.setattr(workflow_module, "load_ensemble_input", lambda *a: loaded)
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(
            workflow,
            "_run_ensemble",
            lambda *a: pytest.fail("executed oversized batch"),
        )
        with pytest.raises(ValueError, match="qualified maximum 2"):
            workflow.execute(
                {
                    "operation": "forecast_ensemble",
                    "parameters": {
                        "manifest_path": "/shared/manifest.json",
                        "manifest_sha256": "b" * 64,
                    },
                    "run_id": "oversized",
                    "run_dir": str(tmp_path),
                }
            )

    def test_cpu_warmup_does_not_load_weights(self, workflow_module, monkeypatch):
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(
            workflow, "_load_model", lambda *a: pytest.fail("loaded model")
        )
        assert workflow.warmup({"device": "cpu"}) == {"model_names": []}
        assert workflow._model is None

    def test_gpu_warmup_reuses_the_cached_model(self, workflow_module, monkeypatch):
        workflow = workflow_module.WORKFLOW()
        calls = []
        model = SimpleNamespace(seed=None)

        def load(device):
            calls.append(device)
            return SimpleNamespace(field_hashes={"verified": "bytes"}), model

        monkeypatch.setattr(workflow, "_load_model", load)
        workflow.warmup({"device": "cuda"})
        workflow.warmup({"device": "cuda"})
        assert calls == ["cuda"]
        assert workflow._model is model

    def test_unqualified_gpu_memory_profile_blocks_dispatch(
        self, tmp_path, workflow_module, monkeypatch
    ):
        workflow = workflow_module.WORKFLOW()
        monkeypatch.delenv("AIFS2ENS_GPU_MEMORY_MB", raising=False)
        monkeypatch.setattr(
            workflow,
            "_stage_inputs",
            lambda *a: pytest.fail("downloaded before deployment validation"),
        )
        with pytest.raises(RuntimeError, match="AIFS2ENS_GPU_MEMORY_MB"):
            workflow.prepare(
                RawRequest(
                    content_type="application/json",
                    operation="run",
                    raw_fields=_plugin_request_fields(),
                ),
                PrepareContext(
                    run_id="parent", workflow_id="earth2-aifs2ens", run_dir=tmp_path
                ),
            )

    def test_whole_ensemble_execution_passes_one_frozen_input_to_the_runner(
        self, tmp_path, workflow_module, monkeypatch
    ):
        workflow = workflow_module.WORKFLOW()
        calls = []
        loaded = SimpleNamespace(request=SimpleNamespace(nensemble=2, batch_size=2))
        monkeypatch.setattr(workflow_module, "load_ensemble_input", lambda *a: loaded)
        monkeypatch.setattr(
            workflow,
            "_run_ensemble",
            lambda item, ctx: (
                calls.append(item) or {"dataset_path": str(tmp_path / "ensemble.zarr")}
            ),
        )
        result = workflow.execute(
            {
                "operation": "forecast_ensemble",
                "parameters": {
                    "manifest_path": "/shared/manifest.json",
                    "manifest_sha256": "b" * 64,
                },
                "run_id": "child",
                "run_dir": str(tmp_path),
            }
        )
        assert calls == [loaded]
        assert result["dataset_path"].endswith("ensemble.zarr")

    def test_example_is_a_valid_public_request(self, workflow_module):
        from aifs2ens_inputs import ForecastRequest

        ForecastRequest.model_validate_json(
            (_PLUGIN_DIR / "examples/default_request.json").read_text()
        )

    def test_readiness_checks_require_qualified_deployment_assets(self, monkeypatch):
        import plugin_dev

        names = {
            "AIFS2ENS_INPUT_CONTRACT",
            "AIFS2ENS_INVARIANTS",
            "AIFS2ENS_CHECKPOINT_DIR",
            "AIFS2ENS_GPU_MEMORY_MB",
        }
        for name in names:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            plugin_dev,
            "_probe_python_module_contract",
            lambda *a, **k: {"ok": True, "detail": "test module available"},
        )
        result = plugin_dev.check_env_plugin(_PLUGIN_DIR)
        checks = {
            check["name"]: check for check in result["checks"] if check["name"] in names
        }
        assert set(checks) == names
        assert all(not check["ok"] for check in checks.values())

    def test_local_service_plan_launches_only_gpu_workers_and_allows_prepare_downloads(
        self,
        tmp_path,
    ):
        import json
        import plugin_dev
        from runtime_env_launcher import build_worker_launch_specs

        workspace = tmp_path / "service"
        plugin_dev.build_run_local_plan(
            _PLUGIN_DIR, workspace=workspace, port=0, redis_port=0
        )
        config = json.loads((workspace / "runtime_config.json").read_text())
        envs = config["python_runtime_envs"]
        assert set(envs) == {"earth2-gpu"}
        assert envs["earth2-gpu"]["launch"]["enabled"] is True
        assert envs["earth2-gpu"]["launch"]["device_kind"] == "gpu"
        assert set(envs["earth2-gpu"]["launch"]["tags"]) == {
            "earth2",
            "aifs2ens",
            "gpu",
        }
        assert config["roles"]["prepare"]["config"]["hook_timeout_secs"] == 7200
        specs = build_worker_launch_specs(
            envs,
            namespace="local",
            pod_name="aifs2ens",
            enabled_executor_classes={"earth2-gpu"},
            gpu_inventory=[
                {
                    "device_index": 0,
                    "device_name": "NVIDIA H100",
                    "device_uuid": "test-gpu",
                    "memory_mb": 81920,
                }
            ],
            worker_script="scripts/inference_worker.py",
        )
        assert len(specs) == 1
        worker_env = specs[0]["env"]
        assert worker_env["GPU_EXECUTOR_CLASS"] == "earth2-gpu"
        assert worker_env["GPU_DEVICE_KIND"] == "gpu"
        assert set(worker_env["GPU_TAGS"].split(",")) == {"earth2", "aifs2ens", "gpu"}
        assert (
            worker_env["GPU_STREAM_NAME"] == "execute.earth2-gpu:gpu:local:aifs2ens:0"
        )

    @pytest.mark.parametrize("failure", [None, "prediction", "validation"])
    def test_final_dataset_is_registered_only_after_full_ensemble_validation(
        self, tmp_path, workflow_module, monkeypatch, failure
    ):
        from aifs2ens_inputs import ForecastRequest
        from plugin_sdk import build_execution_context

        request = ForecastRequest.model_validate(_plugin_request_fields())
        contract = SimpleNamespace(checkpoint_sha256="a" * 64)
        loaded = SimpleNamespace(
            request=request,
            contract=contract,
            provenance={"mirror": "ecmwf"},
            manifest_sha256="b" * 64,
            input_sha256="c" * 64,
        )
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(workflow, "_contract", lambda: contract)
        monkeypatch.setattr(workflow, "_ensure_model", lambda device: None)
        monkeypatch.setattr(workflow, "_runtime_identity", lambda: {"torch": "worker"})
        written = []

        def sink(path, item):
            assert item is loaded
            path.mkdir()
            written.append(path)
            return SimpleNamespace(field_hashes={"verified": "bytes"})

        def rollout(model, item, writer, **kwargs):
            assert item is loaded
            if failure == "prediction":
                raise RuntimeError("prediction failed")

        def validate(path, item, *, expected_hashes):
            assert expected_hashes == {"verified": "bytes"}
            assert path == written[0] and item is loaded
            if failure == "validation":
                raise ValueError("incomplete ensemble")

        for name, value in (
            ("aifs2ens_rollout", SimpleNamespace(run_ensemble_rollout=rollout)),
            (
                "aifs2ens_outputs",
                SimpleNamespace(
                    ZarrEnsembleSink=sink, validate_ensemble_output=validate
                ),
            ),
        ):
            monkeypatch.setitem(sys.modules, name, value)
        ctx = build_execution_context({"run_id": "run", "run_dir": str(tmp_path)})
        if failure:
            with pytest.raises((RuntimeError, ValueError)):
                workflow._run_ensemble(loaded, ctx)
            assert ctx.outputs.registered_outputs() == []
            assert not written[0].exists()
        else:
            result = workflow._run_ensemble(loaded, ctx)
            assert result["complete"] is True
            assert result["nensemble"] == 2 and result["seed_base"] == 0
            assert result["batch_size"] == 2
            assert result["input_sha256"] == "c" * 64
            assert result["runtime"]["torch"] == "worker"
            assert result["rng_policy"] == "torch-seed-once-native-v1"
            assert ctx.outputs.primary_output().path == str(written[0])
            assert (tmp_path / "forecast_metadata.json").is_file()

    @pytest.mark.parametrize("cancel_during", ["flush", "validation"])
    def test_late_cancellation_discards_validated_output_before_publication(
        self, tmp_path, workflow_module, monkeypatch, cancel_during
    ):
        from plugin_sdk import build_execution_context

        request = ForecastRequest.model_validate(_plugin_request_fields())
        contract = SimpleNamespace(checkpoint_sha256="a" * 64)
        loaded = SimpleNamespace(
            request=request,
            contract=contract,
            provenance={"mirror": "ecmwf"},
            manifest_sha256="b" * 64,
            input_sha256="c" * 64,
        )
        workflow = workflow_module.WORKFLOW()
        monkeypatch.setattr(workflow, "_contract", lambda: contract)
        monkeypatch.setattr(workflow, "_ensure_model", lambda device: None)
        monkeypatch.setattr(workflow, "_runtime_identity", lambda: {})
        shutdown = threading.Event()
        stores = []
        events = []

        def sink(path, item):
            assert item is loaded
            path.mkdir()
            stores.append(path)

            def close():
                events.append("flush")
                if cancel_during == "flush":
                    shutdown.set()

            return SimpleNamespace(field_hashes={"verified": "bytes"}, close=close)

        def rollout(model, item, writer, **kwargs):
            assert not kwargs["abort_requested"]()
            writer.close()

        def validate(path, item, *, expected_hashes):
            assert path == stores[0] and item is loaded
            assert expected_hashes == {"verified": "bytes"}
            events.append("validation")
            if cancel_during == "validation":
                shutdown.set()

        for name, value in (
            ("aifs2ens_rollout", SimpleNamespace(run_ensemble_rollout=rollout)),
            (
                "aifs2ens_outputs",
                SimpleNamespace(
                    ZarrEnsembleSink=sink, validate_ensemble_output=validate
                ),
            ),
        ):
            monkeypatch.setitem(sys.modules, name, value)
        ctx = build_execution_context(
            {
                "run_id": "late-cancel",
                "run_dir": str(tmp_path),
                "abort_requested": shutdown.is_set,
            }
        )

        with pytest.raises(PluginCancelledError, match="cancelled"):
            workflow._run_ensemble(loaded, ctx)

        assert events == ["flush", "validation"] and shutdown.is_set()
        assert ctx.outputs.registered_outputs() == []
        assert not stores[0].exists()
        assert not (tmp_path / "forecast_metadata.json").exists()


# Projection


class TestProjection:
    "Loading a model must preserve Earth2Studio's native noise implementation."

    @pytest.mark.parametrize("workspace", [None, ":4096:8"])
    def test_loading_preserves_the_native_gpu_noise_projector(
        self, tmp_path, monkeypatch, workspace
    ):
        if workspace is None:
            monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        else:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
        monkeypatch.setattr(
            runtime, "prepare_assets", lambda contract: tmp_path / "model.ckpt"
        )
        verified = []
        monkeypatch.setattr(
            runtime,
            "verify_model_contract",
            lambda model, contract: verified.append((model, contract)),
        )

        class Matrix:
            dtype = np.dtype("float32")
            device = "cpu"  # Native forward lazily moves this unregistered matrix.

            def to(self, device):
                assert device == "cuda", (
                    "the plugin moved native noise projection to CPU"
                )
                return self

        class Projector:
            autocast = False
            projection_matrix = Matrix()

            def forward(self, values):
                return values

        projector = Projector()
        original_forward = projector.forward.__func__
        moves = []
        model = SimpleNamespace(seed=None, modules=lambda: [projector])
        model.to = lambda device: moves.append(device) or model
        model.eval = lambda: model
        calls = []

        def load(package, **kwargs):
            calls.append(kwargs)
            return model

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True),
                float32=np.dtype("float32"),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "earth2studio.models.auto",
            SimpleNamespace(Package=lambda path, **kwargs: SimpleNamespace(path=path)),
        )
        monkeypatch.setitem(
            sys.modules,
            "earth2studio.models.px",
            SimpleNamespace(AIFS2ENS=SimpleNamespace(load_model=load)),
        )
        monkeypatch.setitem(
            sys.modules,
            "anemoi.models.layers.sparse_projector",
            SimpleNamespace(SparseProjector=Projector),
        )
        contract = object()
        _, loaded = runtime.load_model(contract, "cuda")
        assert loaded is model and moves == ["cuda"]
        assert calls == [{"preload_invariants": False}]
        assert verified == [(model, contract)]
        assert projector.projection_matrix.device == "cpu"
        assert projector.forward.__func__ is original_forward


# Qualification

_QUALIFICATION_SPEC = importlib.util.spec_from_file_location(
    "aifs2ens_qualification", ROOT / "qa/plugins/earth2-aifs2ens/run_qualification.py"
)


_qualification = importlib.util.module_from_spec(_QUALIFICATION_SPEC)


sys.modules[_QUALIFICATION_SPEC.name] = _qualification


_QUALIFICATION_SPEC.loader.exec_module(_qualification)


def _qualification_reference(path, *, seed=0):
    path.mkdir()
    steps = []
    for step in range(1, 5):
        target = path / f"step-{step:04d}.npy"
        target.write_bytes(f"frozen reference step {step}".encode())
        steps.append(
            {
                "step": step,
                "path": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    report = {
        "status": "passed",
        "seed_base": seed,
        "nensemble": 2,
        "batch_size": 2,
        "nsteps": 4,
        "rng_policy": "torch-seed-once-native-v1",
        "model_seed": None,
        "steps": steps,
        "numerical_policy": "earth2studio-native-v1",
        "checkpoint_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "reference_script_sha256": "c" * 64,
        "output_variables": ["t2m"],
        "noise_projection_device": "cuda",
        "model_adapters": [],
        "torch": "native-stack",
        "cuda": "native-cuda",
        "flash_attn": "native-flash",
        "anemoi_models": "0.11.2",
        "deterministic_warn_only": False,
        "gpu_name": "H100",
        "allow_tf32_cudnn": True,
        "deterministic_algorithms": False,
        "allow_bf16_reduced_precision_reduction": True,
        "allow_tf32_matmul": False,
        "cublas_workspace_config": ":4096:8",
    }
    (path / "report.json").write_text(json.dumps(report))
    return report


def _qualification_reference_expected():
    return {
        "seed_base": 0,
        "nensemble": 2,
        "batch_size": 2,
        "nsteps": 4,
        "checkpoint_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "reference_script_sha256": "c" * 64,
        "output_variables": ["t2m"],
    }


class TestQualification:
    "A successful job must contain matching references and executed GPU checks."

    def test_relative_output_keeps_parent_and_capture_paths_together(
        self, tmp_path, monkeypatch
    ):
        caller, repo, work = (tmp_path / name for name in ("caller", "repo", "work"))
        for path in (caller, repo, work):
            path.mkdir()
        assets = caller / "assets"
        (assets / "cache").mkdir(parents=True)
        capture_script = repo / "capture_earth2studio_inputs.py"
        capture_script.write_text(
            "import argparse, hashlib, json\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--original-assets')\n"
            "parser.add_argument('--output', type=Path)\n"
            "args = parser.parse_args()\n"
            "(args.output / 'assets').mkdir(parents=True)\n"
            "for name in ('input', 'invariants'):\n"
            "    (args.output / 'assets' / (name + '.npy')).write_bytes(name.encode())\n"
            "contract = {'invariants_sha256': hashlib.sha256(b'invariants').hexdigest(),\n"
            "            'checkpoint_sha256': 'a' * 64, 'output_variables': ['t2m']}\n"
            "capture = {'status': 'passed', 'shared_input': True,\n"
            "           'preprocessing_policy': 'earth2studio-latlon-frozen-invariants-v1',\n"
            "           'initialization': '2026-11-03T00:00:00Z',\n"
            "           'input': {'sha256': hashlib.sha256(b'input').hexdigest()}}\n"
            "for filename, value in [('qualified-input-contract.json', contract),\n"
            "                        ('weather-capture-report.json', capture)]:\n"
            "    (args.output / filename).write_text(json.dumps(value))\n"
            "print(Path.cwd())\n"
        )
        reference_script = repo / "reference_earth2studio.py"
        reference_script.write_text("# Reference boundary; no model is executed.\n")
        monkeypatch.chdir(caller)
        monkeypatch.setattr(_qualification, "REPO", repo)
        monkeypatch.setattr(
            _qualification, "__file__", str(repo / "run_qualification.py")
        )
        monkeypatch.setattr(_qualification, "verify_source", lambda *_: "base")
        monkeypatch.setattr(_qualification.tempfile, "mkdtemp", lambda **_: str(work))
        real_run = subprocess.run
        references = []

        def run_child(argv, **kwargs):
            if argv[0] == "nvidia-smi":
                return subprocess.CompletedProcess(argv, 0, stdout="mock GPU inventory")
            if argv[1] == str(reference_script):
                references.append((argv, kwargs))
                raise RuntimeError("reference boundary reached")
            return real_run(argv, **kwargs)

        monkeypatch.setattr(_qualification.subprocess, "run", run_child)
        args = SimpleNamespace(
            output=Path("evidence/qualification"),
            assets=Path("assets"),
            image="test-image",
            source_content_sha256="source",
        )
        assert _qualification.run(args) == 1
        output = caller / "evidence/qualification"
        summary = json.loads((output / "summary.json").read_text())
        assert summary["error"] == {
            "type": "RuntimeError",
            "message": "reference boundary reached",
        }
        fixture = output / "fixture"
        assert (fixture / "cache").resolve() == assets / "cache"
        assert (fixture / "qualification").is_dir()
        assert summary["input_capture"]["status"] == "passed"
        assert not (repo / "evidence").exists()
        assert (output / "native-input-capture.log").read_text().strip() == str(repo)
        command = json.loads((output / "native-input-capture.command.json").read_text())
        assert command["argv"][-1] == str(fixture)
        assert len(references) == 1
        argv, options = references[0]
        assert argv[argv.index("--input") + 1] == str(fixture / "assets/input.npy")
        assert options["env"]["AIFS2ENS_QUALIFICATION_ROOT"] == str(fixture)
        assert options["cwd"] == repo
        assert (output / "reference-full-1.log").exists()

    @pytest.mark.parametrize(
        "failure",
        [
            None,
            "different_values",
            "tampered_file",
            "missing_step",
            "wrong_seed",
            "wrong_policy",
            "wrong_batch",
            "wrong_rng",
            "wrong_model_seed",
        ],
    )
    def test_reference_gate_checks_bytes_coverage_and_numerical_identity(
        self, tmp_path, failure
    ):
        first, second = tmp_path / "first", tmp_path / "second"
        _qualification_reference(first)
        report = _qualification_reference(second)
        if failure in {"different_values", "tampered_file"}:
            target = second / "step-0004.npy"
            target.write_bytes(b"different values")
            if failure == "different_values":
                report["steps"][-1]["sha256"] = hashlib.sha256(
                    target.read_bytes()
                ).hexdigest()
        elif failure == "missing_step":
            report["steps"].pop()
        elif failure == "wrong_seed":
            report["seed_base"] = 1024
        elif failure == "wrong_batch":
            report["batch_size"] = 1
        elif failure == "wrong_rng":
            report["rng_policy"] = "per-member-seed-offsets"
        elif failure == "wrong_model_seed":
            report["model_seed"] = 0
        elif failure == "wrong_policy":
            report["numerical_policy"] = "cpu-noise-projection-bf16-v1"
        (second / "report.json").write_text(json.dumps(report))
        expected = {
            key: value for key, value in _qualification_reference_expected().items()
        }
        if failure and failure != "different_values":
            with pytest.raises(ValueError):
                _qualification.verify_reference_pair(first, second, expected, nsteps=4)
        else:
            result = _qualification.verify_reference_pair(
                first, second, expected, nsteps=4
            )
            assert len(result["steps"]) == 4
            assert result["bitwise_identical"] is (failure != "different_values")

    @pytest.mark.parametrize("failure", [None, "skipped", "failed", "missing", "empty"])
    def test_job_gate_rejects_skipped_failed_or_absent_h100_cases(
        self, tmp_path, failure
    ):
        suite = ET.Element("testsuite")
        names = sorted(_qualification.H100_CASES)
        if failure == "missing":
            names.pop()
        if failure == "empty":
            names = []
        for index, name in enumerate(names):
            case = ET.SubElement(
                suite, "testcase", classname="test_h100_parity", name=name
            )
            if index == 0 and failure in {"skipped", "failed"}:
                ET.SubElement(case, "skipped" if failure == "skipped" else "failure")
        path = tmp_path / "junit.xml"
        ET.ElementTree(suite).write(path)
        if failure:
            with pytest.raises(ValueError):
                _qualification.verify_junit(path)
        else:
            assert _qualification.verify_junit(path)["h100_passed"] == 7

    def test_combined_evidence_includes_portable_and_gpu_environment_results(
        self, tmp_path
    ):
        portable = tmp_path / "portable.xml"
        portable.write_text(
            '<testsuites><testsuite><testcase classname="portable" name="contract"/></testsuite></testsuites>'
        )
        gpu = tmp_path / "gpu.xml"
        suite = ET.Element("testsuite")
        for name in _qualification.H100_CASES:
            ET.SubElement(suite, "testcase", classname="test_h100_parity", name=name)
        ET.ElementTree(suite).write(gpu)
        combined = tmp_path / "combined.xml"
        _qualification.merge_junit([portable, gpu], combined)
        assert _qualification.verify_junit(combined) == {
            "passed": 8,
            "h100_passed": 7,
            "skipped": 0,
        }

    def test_parent_release_waits_for_worker_after_completion_without_deduplicating(
        self,
    ):
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, duration):
                self.now += duration

        clock = Clock()
        observations = iter([[], [("1-0", {"run_id": "parent"})]])
        result = _qualification.wait_for_parent_release(
            lambda: next(observations),
            "parent",
            timeout=1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        assert result == [("1-0", {"run_id": "parent"})]
        assert clock.now > 0

    @pytest.mark.parametrize(
        "run_ids", [["parent:item:0"], ["other"], ["parent", "parent"]]
    )
    def test_parent_release_rejects_unexpected_child_or_duplicate_entries(
        self, run_ids
    ):
        entries = [
            (str(index), {"run_id": run_id}) for index, run_id in enumerate(run_ids)
        ]
        with pytest.raises(AssertionError, match="release|allocation"):
            _qualification.wait_for_parent_release(lambda: entries, "parent", timeout=1)

    def test_parent_release_times_out_when_completion_has_no_allocation_release(self):
        now = [0.0]

        def sleep(duration):
            now[0] += duration

        with pytest.raises(TimeoutError, match="release"):
            _qualification.wait_for_parent_release(
                lambda: [], "parent", timeout=0.2, monotonic=lambda: now[0], sleep=sleep
            )

    def test_structured_result_waits_for_finalization_of_the_same_accepted_parent(self):
        provisional = {"status": "succeeded", "execution": {"run_id": "parent"}}
        final = {
            "request": {},
            "execution": {"run_id": "parent"},
            "payload": {"complete": True},
        }
        snapshots = json.loads(json.dumps([provisional, final]))
        responses = iter([provisional, final])
        reads = []
        now = [0.0]

        def read():
            reads.append("/v1/infer/earth2-aifs2ens/parent/results")
            return next(responses)

        def sleep(duration):
            now[0] += duration

        result = _qualification.wait_for_structured_result(
            read, timeout=1, monotonic=lambda: now[0], sleep=sleep
        )
        assert result is final and result["execution"]["run_id"] == "parent"
        assert reads == ["/v1/infer/earth2-aifs2ens/parent/results"] * 2
        assert now[0] > 0
        assert [provisional, final] == snapshots

    @pytest.mark.parametrize(
        "provisional",
        [
            None,
            {},
            {"status": "succeeded", "execution": {}},
            {"request": {}, "execution": {}, "payload": None},
            {"request": [], "execution": {}, "payload": {}},
        ],
    )
    def test_structured_result_times_out_without_all_result_sections(self, provisional):
        now = [0.0]

        def sleep(duration):
            now[0] += duration

        with pytest.raises(TimeoutError, match="structured result"):
            _qualification.wait_for_structured_result(
                lambda: provisional,
                timeout=0.2,
                monotonic=lambda: now[0],
                sleep=sleep,
            )
        assert now[0] == pytest.approx(0.2)

    def test_structured_result_does_not_relax_the_callers_payload_assertions(self):
        final = {"request": {}, "execution": {}, "payload": {"complete": False}}
        result = _qualification.wait_for_structured_result(lambda: final)
        assert result is final
        with pytest.raises(AssertionError):
            assert result["payload"]["complete"] is True

    @pytest.mark.parametrize(
        "failure",
        [None, "wrong_math", "wrong_model_seed", "missing_model_seed", "wrong_anemoi"],
    )
    def test_candidate_runtime_is_checked_from_actual_worker_record(self, failure):
        reference = {
            "torch": "native-torch",
            "cuda": "native-cuda",
            "flash_attn": "native-flash",
            "anemoi_models": "0.11.2",
            "allow_tf32_matmul": True,
            "allow_tf32_cudnn": True,
            "allow_bf16_reduced_precision_reduction": True,
            "deterministic_algorithms": False,
            "deterministic_warn_only": False,
            "cublas_workspace_config": None,
            "gpu_name": "H100",
            "model_seed": None,
        }
        candidate = dict(reference)
        if failure == "wrong_math":
            candidate["allow_tf32_matmul"] = False
        elif failure == "wrong_model_seed":
            candidate["model_seed"] = 0
        elif failure == "missing_model_seed":
            candidate.pop("model_seed")
        elif failure == "wrong_anemoi":
            candidate["anemoi_models"] = "0.11.1"
        if failure:
            with pytest.raises(AssertionError, match="runtime|worker|model_seed"):
                _qualification.verify_candidate_runtime(candidate, reference)
        else:
            _qualification.verify_candidate_runtime(candidate, reference)


# Reference

_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "single_gpu_reference",
    ROOT / "qa/plugins/earth2-aifs2ens/reference_earth2studio.py",
)


_reference = importlib.util.module_from_spec(_REFERENCE_SPEC)


_REFERENCE_SPEC.loader.exec_module(_reference)


class _ReferenceTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class TestReference:
    "Independent stock ensemble reference: one RNG seed and batched member identity."

    @pytest.mark.parametrize("missing_tail", [False, True])
    def test_reference_calls_stock_ensemble_with_one_seed_and_preserves_all_member_slots(
        self, tmp_path, monkeypatch, missing_tail
    ):
        __import__("xarray")  # Import optional-version checks before metadata mocks.
        checkpoint_hash = (
            "e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb"
        )
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
        original_digest = _reference.digest
        input_hash = original_digest(input_path)
        monkeypatch.setattr(
            _reference,
            "digest",
            lambda path: (
                checkpoint_hash
                if str(path).endswith(".ckpt")
                else original_digest(path)
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
                    kwargs["io"].write(
                        [_ReferenceTensor(arrays)], current, np.array(["t2m"])
                    )

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
            sys.modules,
            "earth2studio.perturbation",
            SimpleNamespace(Zero=lambda: object()),
        )
        monkeypatch.setitem(
            sys.modules,
            "anemoi.models.layers.sparse_projector",
            SimpleNamespace(SparseProjector=Projector),
        )
        original_distribution = _reference.importlib.metadata.distribution
        original_version = _reference.importlib.metadata.version
        monkeypatch.setattr(
            _reference.importlib.metadata,
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
            _reference.importlib.metadata,
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
                str(_REFERENCE_SPEC.origin),
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
                _reference.main()
            assert not (output / "report.json").exists()
        else:
            _reference.main()
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

    def test_capture_accepts_relative_asset_and_output_paths(
        self, tmp_path, monkeypatch
    ):
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
        assert (
            output / "assets/checkpoint-xet/aifs-ens-crps-2.0.ckpt"
        ).read_bytes() == (checkpoint / "aifs-ens-crps-2.0.ckpt").read_bytes()
        assert (
            json.loads((output / "qualified-input-contract.json").read_text())
            == contract
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


# Rollout


class _RolloutTorch:
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


class _RolloutModel:
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


class _RolloutSink:
    def __init__(self):
        self.closed = False
        self.complete = False

    def assert_complete(self):
        self.complete = True

    def close(self):
        self.closed = True


class _RolloutZero:
    def __call__(self, values, coords):
        return values, coords


class TestRollout:
    "Single seeded ensemble execution and frozen datasource lifecycle contracts."

    @pytest.fixture
    def loaded(self):
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

    @pytest.fixture(autouse=True)
    def zero_perturbation(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "earth2studio.perturbation", SimpleNamespace(Zero=_RolloutZero)
        )

    def test_frozen_source_selects_two_physical_times_and_requested_variable_order(
        self, loaded
    ):
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

    def test_calls_upstream_ensemble_once_with_shared_seed_and_request_batching(
        self, loaded
    ):
        model, sink, torch = _RolloutModel(), _RolloutSink(), _RolloutTorch()
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
        assert (
            args["nensemble"] == 3 and args["batch_size"] == 2 and args["nsteps"] == 2
        )
        assert args["io"] is sink
        assert args["output_coords"]["variable"].tolist() == ["snowc", "t2m"]
        assert isinstance(args["perturbation"], _RolloutZero)
        np.testing.assert_array_equal(args["time"], loaded.coords["time"])
        assert torch.events == [("save_rng", [0]), ("seed", 0), ("restore_rng", [0])]
        assert model.seed == 999 and model.closed and sink.closed and sink.complete

    @pytest.mark.parametrize("failure", ["runner", "cancel", "coverage"])
    def test_failure_restores_seed_rng_and_closes_retained_iterator(
        self, loaded, failure
    ):
        model, sink, torch = _RolloutModel(), _RolloutSink(), _RolloutTorch()
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
        error = {
            "runner": RuntimeError,
            "cancel": PluginCancelledError,
            "coverage": ValueError,
        }[failure]
        with pytest.raises(error, match="failed|cancelled|coverage"):
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

    def test_cancelled_request_does_not_enter_upstream_runner(self, loaded):
        sink, model, torch = _RolloutSink(), _RolloutModel(), _RolloutTorch()
        with pytest.raises(PluginCancelledError, match="cancelled"):
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

    @pytest.mark.parametrize(
        "phase", ["before_run", "iterator_creation", "iteration", "after_run"]
    )
    def test_worker_classifies_rollout_shutdown_as_cancelled_and_restores_state(
        self, loaded, tmp_path, monkeypatch, phase
    ):
        import inference_worker

        model, sink, torch = _RolloutModel(), _RolloutSink(), _RolloutTorch()
        shutdown = threading.Event()
        if phase == "before_run":
            shutdown.set()

        def ensemble(**kwargs):
            if phase == "iterator_creation":
                shutdown.set()
            iterator = kwargs["prognostic"].create_iterator(None, {})
            assert next(iterator)[0] == "initial"
            shutdown.set()
            if phase == "iteration":
                next(iterator)

        def execute(ctx):
            rollout.run_ensemble_rollout(
                model,
                loaded,
                sink,
                torch_module=torch,
                ensemble_runner=ensemble,
                abort_requested=ctx["abort_requested"],
            )

        class Redis:
            def __init__(self):
                self.values = {}

            def setex(self, key, ttl, value):
                self.values[key] = json.loads(value)

        manifest = {"runtime": {"entrypoint": "workflow.py"}}
        monkeypatch.setattr(
            inference_worker,
            "_resolve_plugin_manifest",
            lambda _: (tmp_path, manifest),
        )
        redis = Redis()
        worker = inference_worker.WorkflowExecutor(
            redis, worker_shutdown_event=shutdown
        )
        monkeypatch.setattr(
            worker,
            "_load_plugin_module",
            lambda *args: SimpleNamespace(execute=execute),
        )
        result = worker.execute(
            "earth2-aifs2ens",
            "cancelled-ensemble",
            {},
            payload={"workflow_id": "earth2-aifs2ens", "run_dir": str(tmp_path)},
        )

        assert result["status"] == "cancelled"
        assert result["error"] == "forecast cancelled"
        assert result["output_path"] is None and result["artifacts"] == []
        assert (
            redis.values["workflow_execution:earth2-aifs2ens:cancelled-ensemble"][
                "status"
            ]
            == "cancelled"
        )
        assert model.seed == 999 and sink.closed and not sink.complete
        assert model.closed is (phase in {"iteration", "after_run"})
        assert model.predictions == 0
        assert torch.events[-1] == ("restore_rng", [0])
        assert (("seed", 0) in torch.events) is (phase != "before_run")

    def test_explicit_gpu_index_is_used_for_rng_isolation(self, loaded):
        torch, sink, model = _RolloutTorch(), _RolloutSink(), _RolloutModel()
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

    def test_new_iterator_closes_previous_before_native_creation(self):
        events = []

        def create_iterator(values, coords):
            events.append(("create", values))
            return SimpleNamespace(close=lambda: events.append(("close", values)))

        adapted = rollout._AbortableModel(
            SimpleNamespace(create_iterator=create_iterator), lambda: False, 2
        )
        first = adapted.create_iterator("first", {})
        second = adapted.create_iterator("second", {})
        assert events == [
            ("create", "first"),
            ("close", "first"),
            ("create", "second"),
        ]
        first.close()
        adapted.close()
        adapted.close()
        second.close()
        assert events[-1] == ("close", "second")
        assert len(events) == 4

    def test_native_close_failure_is_propagated_without_retrying_cleanup(self):
        closes = []

        def close():
            closes.append("close")
            raise RuntimeError("native close failed")

        adapted = rollout._AbortableModel(
            SimpleNamespace(
                create_iterator=lambda values, coords: SimpleNamespace(close=close)
            ),
            lambda: False,
            2,
        )
        iterator = adapted.create_iterator(None, {})
        with pytest.raises(RuntimeError, match="native close failed"):
            adapted.close()
        adapted.close()
        iterator.close()
        assert closes == ["close"]

    def test_final_forecast_releases_native_state_before_next_batch_allocation(
        self, loaded
    ):
        model, sink, torch = _RolloutModel(), _RolloutSink(), _RolloutTorch()
        retained = []

        class BatchAllocation:
            pass

        def ensemble(**kwargs):
            previous_allocation = None
            # This preserves upstream run.ensemble's ordering: repeat/allocate the
            # new input batch first, then call prognostic.create_iterator().
            for _batch in range(2):
                if previous_allocation is not None:
                    assert model.closed, (
                        "old native state survived until the next repeat"
                    )
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


# Runtime


class TestRuntime:
    "Runtime regressions."

    @pytest.fixture
    def contract(self):
        return InputContract.model_validate(
            {
                "schema_version": 1,
                "checkpoint_sha256": "a" * 64,
                "input_variables": ["t2m", "lsm", "sdor", "slor", "z", "wmb"],
                "output_variables": ["t2m"],
                "lat": [90.0, -90.0],
                "lon": [0.0, 180.0],
            }
        )

    @pytest.mark.parametrize("bad", [None, "variables", "grid", "lead"])
    def test_loaded_model_must_match_the_qualified_coordinates(self, contract, bad):
        coords = {
            "variable": np.array(contract.input_variables),
            "lat": np.array(contract.lat),
            "lon": np.array(contract.lon),
            "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
        }
        outputs = {
            **coords,
            "variable": np.array(contract.output_variables),
            "lead_time": np.array([6], dtype="timedelta64[h]"),
        }
        if bad == "variables":
            coords["variable"] = coords["variable"][::-1]
        if bad == "grid":
            coords["lat"] = np.array([80.0, -80.0])
        if bad == "lead":
            outputs["lead_time"] = np.array([12], dtype="timedelta64[h]")
        model = SimpleNamespace(
            input_coords=lambda: coords, output_coords=lambda _: outputs
        )
        if bad:
            with pytest.raises(ValueError, match="contract"):
                runtime.verify_model_contract(model, contract)
        else:
            runtime.verify_model_contract(model, contract)

    def test_changed_checkpoint_is_rejected_before_unpickling(
        self, tmp_path, monkeypatch, contract
    ):
        checkpoint = tmp_path / "aifs-ens-crps-2.0.ckpt"
        checkpoint.write_bytes(b"original pinned test bytes")
        expected = sha256_file(checkpoint)
        contract = contract.model_copy(update={"checkpoint_sha256": expected})
        monkeypatch.setenv("AIFS2ENS_CHECKPOINT_DIR", str(tmp_path))
        monkeypatch.setattr(runtime, "CHECKPOINT_SHA256", expected)
        assert runtime.verified_checkpoint(contract) == checkpoint
        checkpoint.write_bytes(b"changed")
        with pytest.raises(ValueError, match="checkpoint.*digest"):
            runtime.verified_checkpoint(contract)

    def test_a_different_checkpoint_cannot_be_selected_through_a_new_contract(
        self, tmp_path, monkeypatch, contract
    ):
        monkeypatch.setenv("AIFS2ENS_CHECKPOINT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="pinned checkpoint"):
            runtime.verified_checkpoint(contract)

    @pytest.mark.parametrize("invalid_assets", [False, True])
    def test_worker_warmup_verifies_assets_once_before_model_construction(
        self, tmp_path, monkeypatch, contract, invalid_assets
    ):
        import inference_worker

        plugin = Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"
        spec = importlib.util.spec_from_file_location(
            "aifs2ens_verified_warmup_workflow", plugin / "workflow.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, "cleanup_earth2_runtime_resources", lambda _: None)
        monkeypatch.setattr(module, "cleanup_python_and_torch_runtime", lambda: None)
        contract_path = tmp_path / "input-contract.json"
        contract_path.write_text(contract.model_dump_json())
        settings = {
            "PLUGIN_DIR": str(plugin.parent),
            "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": "earth2-aifs2ens",
            "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR": str(tmp_path / "locks"),
            "GPU_EXECUTOR_CLASS": "earth2-gpu",
            "GPU_DEVICE_KIND": "gpu",
            "AIFS2ENS_INPUT_CONTRACT": str(contract_path),
        }
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        events = []

        def verify_assets(contract=None):
            events.append("verify")
            if invalid_assets:
                raise ValueError("asset digest mismatch")
            return tmp_path / "aifs-ens-crps-2.0.ckpt"

        monkeypatch.setattr(runtime, "prepare_assets", verify_assets)
        coords = {
            "variable": np.array(contract.input_variables),
            "lat": np.array(contract.lat),
            "lon": np.array(contract.lon),
            "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
        }
        model = SimpleNamespace(
            seed=None,
            input_coords=lambda: coords,
            output_coords=lambda _: {
                **coords,
                "variable": np.array(contract.output_variables),
                "lead_time": np.array([6], dtype="timedelta64[h]"),
            },
        )
        model.to = lambda _: model
        model.eval = lambda: model

        def construct(package, **kwargs):
            assert events[-1] == "verify"
            assert kwargs == {"preload_invariants": False}
            events.append("construct")
            return model

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        )
        monkeypatch.setitem(
            sys.modules,
            "earth2studio.models.auto",
            SimpleNamespace(Package=lambda *args, **kwargs: object()),
        )
        monkeypatch.setitem(
            sys.modules,
            "earth2studio.models.px",
            SimpleNamespace(AIFS2ENS=SimpleNamespace(load_model=construct)),
        )
        executor = inference_worker.WorkflowExecutor(SimpleNamespace())
        monkeypatch.setattr(executor, "_load_plugin_module", lambda *args: module)
        try:
            if invalid_assets:
                with pytest.raises(ValueError, match="asset digest mismatch"):
                    executor.warm_enabled_workflow()
                assert events == ["verify"]
            else:
                assert executor.warm_enabled_workflow()["status"] == "warmed"
                assert executor.warm_enabled_workflow()["status"] == "warmed"
                assert events == ["verify", "construct"]
                assert executor._workflow_cache["earth2-aifs2ens"]._model is model
        finally:
            executor.close()

    def test_execution_runtime_records_the_actual_worker_math_settings(
        self, monkeypatch
    ):
        torch = SimpleNamespace(
            __version__="worker-torch",
            version=SimpleNamespace(cuda="worker-cuda"),
            cuda=SimpleNamespace(get_device_name=lambda device: "worker H100"),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(
                        allow_tf32=True,
                        allow_bf16_reduced_precision_reduction=False,
                    )
                ),
                cudnn=SimpleNamespace(allow_tf32=False),
            ),
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setattr(
            runtime,
            "version",
            lambda name: {"flash-attn": "worker-flash", "anemoi-models": "0.11.2"}[
                name
            ],
        )
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        identity = runtime.runtime_identity(SimpleNamespace(seed=None), "cuda:0")
        assert identity["torch"] == "worker-torch"
        assert identity["flash_attn"] == "worker-flash"
        assert identity["allow_tf32_matmul"] is True
        assert identity["allow_tf32_cudnn"] is False
        assert identity["allow_bf16_reduced_precision_reduction"] is False
        assert identity["deterministic_algorithms"] is True
        assert identity["deterministic_warn_only"] is False
        assert identity["cublas_workspace_config"] == ":4096:8"
        assert identity["gpu_name"] == "worker H100" and identity["model_seed"] is None


# Serve


class _ServiceProcess:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.code = 0
        return 0


def _fake_service(serving, monkeypatch, tmp_path, *, available=True, failure=None):
    stream = "execute.earth2-gpu:gpu:local:earth2-aifs2ens-run-local:0"
    plan = {
        "workspace": tmp_path,
        "port": 8001,
        "redis_port": 6379,
        "redis_url": "redis://127.0.0.1:6379/0",
        "execute_registration_streams": [stream],
        "processes": [
            {"name": name, "argv": [name], "env": {}}
            for name in ("redis", "inference_server", "runtime_env_launcher")
        ],
    }
    (tmp_path / "logs").mkdir()
    processes, killed, spawned, ports = [], [], [], []
    stop = threading.Event()
    clock = SimpleNamespace(value=0, reads=0, ready=None)

    def popen(argv, **kwargs):
        if failure == "spawn" and len(processes) == 1:
            raise OSError("failed to start server")
        proc = _ServiceProcess(100 + len(processes), 7 if failure == "child" else None)
        processes.append(proc)
        spawned.append(kwargs)
        return proc

    class Redis:
        def hgetall(self, key):
            assert key == "gpu:registry"
            clock.reads += 1
            return {
                "worker": json.dumps(
                    {
                        "stream": stream,
                        "executor_class": "earth2-gpu",
                        "device_kind": "gpu",
                        "status": "available"
                        if available and clock.reads > 1
                        else "warming",
                    }
                )
            }

        def xrevrange(self, key, *, count):
            assert key == "release"
            assert count == 100
            return [] if clock.reads < 3 else [("1-0", {"run_id": "parent"})]

        def close(self):
            pass

    def sleep(seconds):
        clock.value += seconds
        ready = tmp_path / "service-ready.json"
        if ready.exists():
            clock.ready = json.loads(ready.read_text())
        if available and clock.reads >= 3:
            stop.set()

    monkeypatch.setattr(serving.subprocess, "Popen", popen)
    monkeypatch.setattr(
        serving.os, "killpg", lambda pid, signum: killed.append((pid, signum))
    )
    monkeypatch.setattr(
        serving.plugin_dev, "_wait_for_tcp_port", lambda *a, **kw: ports.append((a, kw))
    )
    monkeypatch.setattr(serving.redis.Redis, "from_url", lambda *a, **kw: Redis())
    monkeypatch.setattr(serving.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(serving.time, "sleep", sleep)
    return plan, stop, clock, processes, killed, spawned, ports


class TestServe:
    "Portable lifecycle checks for the externally exposed AIFS service."

    @pytest.fixture
    def serving(self):
        path = ROOT / "qa/plugins/earth2-aifs2ens/serve.py"
        spec = importlib.util.spec_from_file_location("aifs2ens_service", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_service_import_does_not_load_the_qualification_runner(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def isolated_import(name, *args, **kwargs):
            assert name not in {"run_qualification", "native_parity"}, name
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", isolated_import)
        spec = importlib.util.spec_from_file_location(
            "standalone_aifs2ens_service", ROOT / "qa/plugins/earth2-aifs2ens/serve.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def test_waits_for_available_worker_then_persists_releases_and_cleans_groups(
        self, serving, monkeypatch, tmp_path
    ):
        plan, stop, clock, processes, killed, spawned, ports = _fake_service(
            serving, monkeypatch, tmp_path
        )
        serving.run_service(plan, stop_event=stop, startup_timeout=600)
        assert clock.reads == 3
        assert clock.ready["executor_class"] == "earth2-gpu"
        assert clock.ready["workspace"] == str(tmp_path)
        assert clock.ready["registry_path"] == str(
            tmp_path / "worker-observations.json"
        )
        observations = json.loads((tmp_path / "worker-observations.json").read_text())
        assert observations["registry"]["worker"]["status"] == "available"
        assert observations["releases"] == [["1-0", {"run_id": "parent"}]]
        assert (
            json.loads((tmp_path / "service-state.json").read_text())["status"]
            == "stopped"
        )
        assert not (tmp_path / "service-ready.json").exists()
        assert all(item["start_new_session"] is True for item in spawned)
        assert len(ports) == 2
        assert all(
            (proc.pid, signal.SIGTERM) in killed
            and (proc.pid, signal.SIGKILL) in killed
            for proc in processes
        )

    def test_release_observations_keep_only_recent_history_in_stream_order(
        self, serving
    ):
        entries = [(f"{index}-0", {"run_id": str(index)}) for index in range(1000)]
        calls = []

        class Redis:
            def hgetall(self, key):
                assert key == "gpu:registry"
                return {}

            def xrevrange(self, key, *, count):
                calls.append((key, count))
                return list(reversed(entries))[:count]

        result = serving._observations(Redis())
        assert calls == [("release", 100)]
        assert result["releases"] == entries[-100:]

    def test_stale_readiness_is_removed_before_starting_processes(
        self, serving, monkeypatch, tmp_path
    ):
        plan, stop, _, _, _, _, _ = _fake_service(serving, monkeypatch, tmp_path)
        ready = tmp_path / "service-ready.json"
        ready.write_text('{"status": "ready"}')
        popen = serving.subprocess.Popen

        def spawn(*args, **kwargs):
            assert not ready.exists(), "stale GPU readiness leaked into startup"
            return popen(*args, **kwargs)

        monkeypatch.setattr(serving.subprocess, "Popen", spawn)
        serving.run_service(plan, stop_event=stop)

    @pytest.mark.parametrize("failure", ["timeout", "child", "spawn", "port"])
    def test_startup_failure_never_signals_ready_and_cleans_started_groups(
        self, serving, monkeypatch, tmp_path, failure
    ):
        plan, stop, _, processes, killed, spawned, _ = _fake_service(
            serving, monkeypatch, tmp_path, available=False, failure=failure
        )
        if failure == "port":

            def no_port(*args, **kwargs):
                raise TimeoutError("service port did not open")

            monkeypatch.setattr(serving.plugin_dev, "_wait_for_tcp_port", no_port)
        error = {
            "timeout": TimeoutError,
            "child": RuntimeError,
            "spawn": OSError,
            "port": TimeoutError,
        }[failure]
        with pytest.raises(error):
            serving.run_service(plan, stop_event=stop, startup_timeout=2)
        assert not (tmp_path / "service-ready.json").exists()
        assert (
            json.loads((tmp_path / "service-state.json").read_text())["status"]
            == "failed"
        )
        assert all(
            (proc.pid, signal.SIGTERM) in killed
            and (proc.pid, signal.SIGKILL) in killed
            for proc in processes
        )
        assert all(item["stdout"].closed for item in spawned)

    @pytest.mark.parametrize("already_stopped", [False, True])
    def test_stop_during_startup_skips_remaining_processes_and_closes_logs(
        self, serving, monkeypatch, tmp_path, already_stopped
    ):
        plan, stop, clock, processes, killed, spawned, _ = _fake_service(
            serving, monkeypatch, tmp_path
        )
        if already_stopped:
            stop.set()
        else:
            popen = serving.subprocess.Popen

            def spawn(*args, **kwargs):
                process = popen(*args, **kwargs)
                stop.set()
                return process

            monkeypatch.setattr(serving.subprocess, "Popen", spawn)
        serving.run_service(plan, stop_event=stop)
        assert len(processes) == (0 if already_stopped else 1)
        assert clock.reads == 0
        assert not (tmp_path / "service-ready.json").exists()
        assert json.loads((tmp_path / "service-state.json").read_text()) == {
            "status": "stopped",
            "error": None,
        }
        assert all(item["stdout"].closed for item in spawned)
        assert all((proc.pid, signal.SIGKILL) in killed for proc in processes)

    def test_process_context_cleans_exited_leaders_and_logs_after_body_failure(
        self, serving, monkeypatch, tmp_path
    ):
        plan, _, _, processes, killed, spawned, ports = _fake_service(
            serving, monkeypatch, tmp_path
        )
        with pytest.raises(ValueError, match="request failed"):
            with serving.service_processes(plan, port_timeout=60):
                for process in processes:
                    process.code = 0
                raise ValueError("request failed")
        assert all(kwargs["timeout_secs"] == 60 for _, kwargs in ports)
        assert all(
            (proc.pid, signal.SIGTERM) in killed
            and (proc.pid, signal.SIGKILL) in killed
            for proc in processes
        )
        assert all(item["stdout"].closed for item in spawned)

    @pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
    def test_main_verifies_source_and_handles_signals_in_unique_sessions(
        self, serving, monkeypatch, tmp_path, signum
    ):
        calls, handlers, sessions = [], {}, []
        monkeypatch.setenv("AIFS2ENS_SOURCE_CONTENT_SHA256", "a" * 64)
        monkeypatch.setattr(
            serving, "verify_source", lambda repo, value: calls.append((repo, value))
        )

        def install(signum, handler):
            handlers[signum] = handler

        monkeypatch.setattr(serving.signal, "signal", install)

        def plan(plugin, *, workspace, port, redis_port):
            calls.append("plan")
            assert port == 8001 and redis_port == 6379
            sessions.append(workspace)
            return {"workspace": workspace}

        def run(plan, *, stop_event, startup_timeout):
            assert startup_timeout == 600
            handlers[signum](signum, None)
            assert stop_event.is_set()

        monkeypatch.setattr(serving.plugin_dev, "build_run_local_plan", plan)
        monkeypatch.setattr(serving, "run_service", run)
        assert serving.main(["--workspace", str(tmp_path)]) == 0
        assert serving.main(["--workspace", str(tmp_path)]) == 0
        assert calls == [(serving.REPO, "a" * 64), "plan"] * 2
        assert sessions[0] != sessions[1]
        pointer = json.loads((tmp_path / "current-service.json").read_text())
        assert pointer["workspace"] == str(sessions[1])
        assert pointer["source_content_sha256"] == "a" * 64

    def test_source_mismatch_blocks_service_creation(
        self, serving, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("AIFS2ENS_SOURCE_CONTENT_SHA256", "a" * 64)

        def reject(repo, expected):
            raise ValueError("image source bytes differ")

        monkeypatch.setattr(serving, "verify_source", reject)
        with pytest.raises(ValueError, match="source bytes"):
            serving.main(["--workspace", str(tmp_path)])
        assert not (tmp_path / "current-service.json").exists()

    def test_readiness_tracks_worker_restarts_and_preserves_healthy_busy_workers(
        self, serving, monkeypatch, tmp_path
    ):
        plan, stop, clock, _, _, _, _ = _fake_service(serving, monkeypatch, tmp_path)
        statuses = ["available", "busy", "warming", None, "failed", "busy", "available"]
        observed = []

        def read(_):
            status = statuses[clock.reads]
            clock.reads += 1
            return {
                "registry": {}
                if status is None
                else {
                    "worker": {
                        "stream": plan["execute_registration_streams"][0],
                        "executor_class": "earth2-gpu",
                        "device_kind": "gpu",
                        "status": status,
                    }
                },
                "releases": [],
            }

        def sleep(seconds):
            clock.value += seconds
            observed.append(
                (
                    (tmp_path / "service-ready.json").exists(),
                    json.loads((tmp_path / "service-state.json").read_text())["status"],
                )
            )
            if clock.reads == len(statuses):
                stop.set()

        monkeypatch.setattr(serving, "_observations", read)
        monkeypatch.setattr(serving.time, "sleep", sleep)
        serving.run_service(plan, stop_event=stop, startup_timeout=600)
        assert observed == [
            (True, "ready"),
            (True, "ready"),
            (False, "recovering"),
            (False, "recovering"),
            (False, "recovering"),
            (True, "ready"),
            (True, "ready"),
        ]

    def test_worker_recovery_has_a_bounded_deadline(
        self, serving, monkeypatch, tmp_path
    ):
        plan, stop, clock, _, _, _, _ = _fake_service(serving, monkeypatch, tmp_path)

        def read(_):
            clock.reads += 1
            return {
                "registry": {
                    "worker": {
                        "stream": plan["execute_registration_streams"][0],
                        "executor_class": "earth2-gpu",
                        "device_kind": "gpu",
                        "status": "available",
                    }
                }
                if clock.reads == 1
                else {},
                "releases": [],
            }

        monkeypatch.setattr(serving, "_observations", read)

        def sleep(seconds):
            clock.value += seconds
            assert clock.value <= 4, "recovery deadline was never enforced"

        monkeypatch.setattr(serving.time, "sleep", sleep)
        with pytest.raises(TimeoutError):
            serving.run_service(plan, stop_event=stop, startup_timeout=2)
        assert clock.value == 3
        assert not (tmp_path / "service-ready.json").exists()


# Source

_SOURCE_SCRIPT = ROOT / "qa/plugins/earth2-aifs2ens/prepare_source.py"


class TestSource:
    "A clean image context binds the actual source bytes and excludes local assets."

    @pytest.fixture
    def source_builder(self):
        spec = importlib.util.spec_from_file_location("aifs2ens_source", _SOURCE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def repository(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "tracked.py").write_text("print('initial')\n")
        (repo / ".gitignore").write_text("ignored/\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Source Test",
                "-c",
                "user.email=source-test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=repo,
            check=True,
        )
        (repo / "tracked.py").write_text("print('working tree')\n")
        (repo / "new-plugin.py").write_text("print('untracked feature')\n")
        (repo / "new-plugin.py").chmod(0o755)
        return repo

    def test_context_has_modified_and_untracked_source_with_deterministic_inventory(
        self, source_builder, repository, tmp_path
    ):
        first, second = tmp_path / "first", tmp_path / "second"
        identity = source_builder.prepare_source(repository, first)
        assert source_builder.prepare_source(repository, second) == identity
        manifest = json.loads((first / "qa-source-inventory.json").read_text())
        assert (first / "qa-source-inventory.json").read_bytes() == (
            second / "qa-source-inventory.json"
        ).read_bytes()
        assert manifest["source_content_sha256"] == identity
        assert (
            manifest["base_commit"]
            == subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()
        )
        assert [item["path"] for item in manifest["files"]] == [
            ".gitignore",
            "new-plugin.py",
            "tracked.py",
        ]
        assert (
            hashlib.sha256(
                json.dumps(
                    manifest["files"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            == identity
        )
        for item in manifest["files"]:
            data = (first / item["path"]).read_bytes()
            assert data == (repository / item["path"]).read_bytes()
            assert item["sha256"] == hashlib.sha256(data).hexdigest()
            assert item["bytes"] == len(data)
        assert (first / "new-plugin.py").stat().st_mode & 0o111

    def test_context_works_with_image_verifier_and_detects_tampering(
        self, source_builder, repository, tmp_path
    ):
        context = tmp_path / "context"
        identity = source_builder.prepare_source(repository, context)
        assert (
            source_builder.verify_source(context, identity)
            == subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()
        )
        (context / "tracked.py").write_text("tampered\n")
        with pytest.raises(ValueError, match="source bytes differ"):
            source_builder.verify_source(context, identity)

    def test_verifier_rejects_inventory_identity_changes(
        self, source_builder, repository, tmp_path
    ):
        context = tmp_path / "context"
        identity = source_builder.prepare_source(repository, context)
        inventory = context / "qa-source-inventory.json"
        manifest = json.loads(inventory.read_text())
        manifest["files"][0]["sha256"] = "0" * 64
        source_builder.write_json(inventory, manifest)
        with pytest.raises(ValueError, match="source inventory differs"):
            source_builder.verify_source(context, identity)

    def test_context_excludes_credentials_caches_assets_and_old_inventories(
        self, source_builder, repository, tmp_path
    ):
        blocked = [
            ".env",
            "nested/.env.token",
            "nested/private.pem",
            "nested/private.key",
            ".venv/package.py",
            "nested/__pycache__/code.pyc",
            "target/generated.rs",
            ".run-local/worker.json",
            "qa/artifacts/report.json",
            "deploy/config.yaml",
            "ignored/unrelated.py",
            "assets/model.ckpt",
            "assets/frozen.npy",
            "assets/forecast.zarr/chunk.json",
            "qa-source-inventory.json",
        ]
        for name in blocked:
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("excluded fixture\n")
        # Explicit exclusions still apply if a sensitive file was accidentally tracked.
        subprocess.run(["git", "add", "-f", ".env"], cwd=repository, check=True)
        context = tmp_path / "context"
        source_builder.prepare_source(repository, context)
        manifest = json.loads((context / "qa-source-inventory.json").read_text())
        assert [item["path"] for item in manifest["files"]] == [
            ".gitignore",
            "new-plugin.py",
            "tracked.py",
        ]
        assert not any((context / name).exists() for name in blocked[:-1])

    @pytest.mark.parametrize("existing", [False, True])
    def test_context_refuses_to_replace_existing_destination(
        self, source_builder, repository, tmp_path, existing
    ):
        context = tmp_path / "context"
        if existing:
            context.write_text("preserve this file")
        else:
            context.mkdir()
            (context / "keep.txt").write_text("preserve this directory")
        with pytest.raises(FileExistsError):
            source_builder.prepare_source(repository, context)
        assert (
            context.read_text() if existing else (context / "keep.txt").read_text()
        ).startswith("preserve this")

    def test_context_must_be_outside_the_repository(self, source_builder, repository):
        with pytest.raises(ValueError, match="outside"):
            source_builder.prepare_source(repository, repository / "context")
        assert not (repository / "context").exists()

    def test_context_refuses_existing_dangling_destination_symlink(
        self, source_builder, repository, tmp_path
    ):
        context, target = tmp_path / "context", tmp_path / "uncreated-target"
        context.symlink_to(target)
        with pytest.raises(FileExistsError):
            source_builder.prepare_source(repository, context)
        assert context.is_symlink()
        assert not target.exists()

    def test_context_rejects_external_symlinks_without_copying_target(
        self, source_builder, repository, tmp_path
    ):
        secret = tmp_path / "private.txt"
        secret.write_text("private fixture")
        (repository / "linked.py").symlink_to(secret)
        context = tmp_path / "context"
        with pytest.raises(ValueError, match="symlink"):
            source_builder.prepare_source(repository, context)
        assert not context.exists()

    def test_context_fingerprint_changes_with_working_tree_bytes(
        self, source_builder, repository, tmp_path
    ):
        before = source_builder.prepare_source(repository, tmp_path / "before")
        (repository / "new-plugin.py").write_text("updated implementation\n")
        after = source_builder.prepare_source(repository, tmp_path / "after")
        assert before != after

    def test_cli_prints_fingerprint_used_by_build_argument(self, repository, tmp_path):
        context = tmp_path / "context"
        result = subprocess.run(
            [
                sys.executable,
                str(_SOURCE_SCRIPT),
                "--repo",
                str(repository),
                "--output",
                str(context),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((context / "qa-source-inventory.json").read_text())
        assert result.stdout == manifest["source_content_sha256"] + "\n"


# Storage


class _HostTensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return np.asarray(self)


def _write_storage_batch(sink, coords, members, lead, *, dtype=np.float32):
    current = OrderedDict(coords)
    current["ensemble"] = np.array(members)
    current["lead_time"] = np.array([lead], dtype="timedelta64[h]")
    arrays = [
        np.full((len(members), 1, 1, 2, 2), index + lead, dtype=dtype)
        for index in (0, 1)
    ]
    sink.write(arrays, current, np.array(["t2m", "msl"]))


class TestStorage:
    "A whole-ensemble sink must preserve native minibatch writes and coverage."

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        writes, registrations, options, finalizes = [], [], [], []
        backend = SimpleNamespace(
            add_array=lambda coords, names: registrations.append((coords, names)),
            write=lambda data, coords, names: writes.append((data, coords, names)),
            finalize=lambda: finalizes.append(True),
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(from_numpy=lambda value: value.view(_HostTensor)),
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

    def test_filters_initial_yield_and_writes_complete_uneven_minibatches(self, setup):
        path, loaded, coords, writes, registrations, _, finalizes = setup
        sink = outputs.ZarrEnsembleSink(path / "ensemble.zarr", loaded)
        sink.add_array(coords, np.array(["t2m", "msl"]))
        for members in ([0, 1], [2]):
            for lead in (0, 6, 12):
                _write_storage_batch(sink, coords, members, lead)
        sink.assert_complete()
        sink.close()
        sink.close()
        assert len(writes) == 4 and finalizes == [True]
        np.testing.assert_array_equal(
            registrations[0][0]["lead_time"], np.array([6, 12], dtype="timedelta64[h]")
        )
        assert writes[-1][1]["ensemble"].tolist() == [2]
        assert all(
            isinstance(value, _HostTensor)
            for values, _, _ in writes
            for value in values
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
    def test_incomplete_or_invalid_writes_cannot_publish(self, setup, failure):
        path, loaded, coords, writes, *_ = setup
        sink = outputs.ZarrEnsembleSink(path / "ensemble.zarr", loaded)
        sink.add_array(coords, np.array(["t2m", "msl"]))
        _write_storage_batch(sink, coords, [0, 1], 6)
        with pytest.raises(
            ValueError, match="coverage|duplicate|float32|coordinate|shape"
        ):
            if failure == "missing_member":
                _write_storage_batch(sink, coords, [0, 1], 12)
                sink.assert_complete()
            elif failure == "missing_step":
                _write_storage_batch(sink, coords, [2], 6)
                sink.assert_complete()
            elif failure == "duplicate":
                _write_storage_batch(sink, coords, [1], 6)
            elif failure == "dtype":
                _write_storage_batch(sink, coords, [2], 6, dtype=np.float64)
            elif failure == "wrong_coordinates":
                bad = OrderedDict(coords)
                bad["lat"] = np.array([80.0, -80.0])
                _write_storage_batch(sink, bad, [2], 6)
            else:
                bad = OrderedDict(coords)
                bad["ensemble"] = np.array([2])
                bad["lead_time"] = np.array([6], dtype="timedelta64[h]")
                sink.write(
                    [np.ones((1, 1, 1, 1, 2), dtype=np.float32)] * 2,
                    bad,
                    ["t2m", "msl"],
                )
        sink.close()  # Cleanup must not hide the original error with a coverage error.

    def test_rust_capacity_covers_the_entire_accepted_ensemble(
        self, setup, monkeypatch
    ):
        path, loaded, coords, _, _, options, _ = setup
        monkeypatch.setattr(
            sys.modules["e2s_workflow"], "_selected_zarr_backend", lambda: "rust"
        )
        loaded.contract.lat, loaded.contract.lon = np.arange(721), np.arange(1440)
        sink = outputs.ZarrEnsembleSink(path / "rust.zarr", loaded)
        sink.close()
        settings = options[0]
        expected = 3 * 2 * 2 * 721 * 1440 * 4
        assert settings["max_pool_bytes"] == 256 * 1024**2
        assert settings["max_inflight_transient_bytes"] == expected + 256 * 1024**2
        assert settings["chunks"] == {"ensemble": 1, "time": 1, "lead_time": 1}

    def test_backend_failure_finalizes_without_registering_a_write(self, setup):
        path, loaded, coords, _, _, _, _ = setup
        sink = outputs.ZarrEnsembleSink(path / "bad.zarr", loaded)
        sink.add_array(coords, np.array(["t2m", "msl"]))
        sink.backend.write = lambda *a: (_ for _ in ()).throw(
            RuntimeError("storage failed")
        )
        with pytest.raises(RuntimeError, match="storage failed"):
            _write_storage_batch(sink, coords, [0, 1, 2], 6)
        with pytest.raises(ValueError, match="coverage"):
            sink.assert_complete()
        assert sink.field_hashes == {}
        sink.close()

    def test_hashes_bind_field_bytes_before_the_backend_can_mutate_them(self, setup):
        path, loaded, coords, *_ = setup
        sink = outputs.ZarrEnsembleSink(path / "checksummed.zarr", loaded)
        sink.add_array(coords, np.array(["t2m", "msl"]))

        def miswrite(data, _coords, _names):
            data[0][...] = -99

        sink.backend.write = miswrite
        _write_storage_batch(sink, coords, [0, 1, 2], 6)
        expected = hashlib.sha256(
            np.full((2, 2), 6, dtype=np.float32).tobytes()
        ).hexdigest()
        assert sink.field_hashes[(2, 1, "t2m")] == expected
        assert len(sink.field_hashes) == 6
        sink.close()
