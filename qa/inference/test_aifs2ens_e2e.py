# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in HTTPS inference and forecast-download coverage for AIFS2ENS."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse
import zipfile

import pytest
import requests

from helpers import skip_if_workflow_disabled


WORKFLOW_ID = "earth2-aifs2ens"
OUTPUT_VARIABLES = ["t2m", "msl", "u10m", "v10m"]
CHECKPOINT_SHA256 = "e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb"
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.cicd,
    pytest.mark.rust_only,
    pytest.mark.skipif(
        os.environ.get("QA_AIFS2ENS_E2E_ENABLED", "").strip() != "1",
        reason="set QA_AIFS2ENS_E2E_ENABLED=1 to run live AIFS2ENS inference",
    ),
]


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _verify_result(result, request, run_id):
    execution, payload = result["execution"], result["payload"]
    assert execution["run_id"] == run_id and execution["status"] == "succeeded"
    assert execution["workflow"] == WORKFLOW_ID and execution.get("error") is None
    assert execution.get("parent_run_id") is None
    assert payload["complete"] is True
    for name in ("nensemble", "nsteps", "batch_size", "seed_base", "ifs_member_id"):
        assert payload[name] == request[name], name
    assert payload["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert payload["numerical_policy"] == "earth2studio-native-v1"
    assert payload["rng_policy"] == "torch-seed-once-native-v1"
    for name in ("input_sha256", "manifest_sha256"):
        assert re.fullmatch("[0-9a-f]{64}", payload[name]), name
    runtime = payload["runtime"]
    assert "H100" in runtime["gpu_name"]
    assert runtime["model_seed"] is None
    assert runtime["anemoi_models"] == "0.11.2"
    assert result["request"]["raw_fields"] == request
    assert any(
        item["name"] == "forecast_dataset"
        and item["media_type"] == "application/x-zarr"
        and item["primary"] is True
        and item["storage_path"] == payload["dataset_path"]
        for item in execution["outputs"]
    )


def _verify_archive(path, request):
    import numpy as np
    import xarray as xr
    from zarr.storage import ZipStore

    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert sum(item.file_size for item in archive.infolist()) <= 256 * 1024**2
    with ZipStore(path, mode="r") as store:
        with xr.open_zarr(
            store, chunks=None, consolidated=False, decode_timedelta=True
        ) as dataset:
            assert set(dataset.data_vars) == set(OUTPUT_VARIABLES)
            assert dict(dataset.sizes) == {
                "ensemble": 2,
                "time": 1,
                "lead_time": 2,
                "lat": 721,
                "lon": 1440,
            }
            np.testing.assert_array_equal(dataset.ensemble, [0, 1])
            np.testing.assert_array_equal(
                dataset.time,
                np.array(
                    [request["start_time"].removesuffix("Z")], dtype="datetime64[ns]"
                ),
            )
            np.testing.assert_array_equal(
                dataset.lead_time, np.array([6, 12], dtype="timedelta64[h]")
            )
            np.testing.assert_array_equal(dataset.lat, np.linspace(90, -90, 721))
            np.testing.assert_array_equal(dataset.lon, np.arange(1440) * 0.25)
            fields = {}
            for name in OUTPUT_VARIABLES:
                field = dataset[name]
                assert field.dims == ("ensemble", "time", "lead_time", "lat", "lon")
                assert field.dtype == np.float32
                values = np.ascontiguousarray(field.values)
                assert np.isfinite(values).all(), name
                fields[name] = {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                }
    return fields


def test_aifs2ens_https_forecast_and_download(base_url, user_token, adapter):
    """Run one two-member forecast and read all four fields from its HTTPS ZIP."""
    skip_if_workflow_disabled(adapter, WORKFLOW_ID)
    parsed = urlparse(base_url)
    assert parsed.scheme == "https" and (parsed.hostname or "").endswith(".lepton.run")
    assert not (parsed.username or parsed.password or parsed.query or parsed.fragment)
    raw_time = os.environ.get("QA_AIFS2ENS_START_TIME", "").strip()
    assert raw_time, (
        "QA_AIFS2ENS_START_TIME must select an available six-hour UTC cycle"
    )
    initialization = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
    assert initialization.tzinfo is not None
    initialization = initialization.astimezone(timezone.utc)
    assert initialization.hour % 6 == 0
    assert (
        initialization.minute
        == initialization.second
        == initialization.microsecond
        == 0
    )
    configured_output = os.environ.get("QA_AIFS2ENS_ARTIFACT_DIR", "").strip()
    assert configured_output, (
        "QA_AIFS2ENS_ARTIFACT_DIR must name a task-owned evidence directory"
    )
    output = Path(configured_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    timeout = int(os.environ.get("QA_AIFS2ENS_TIMEOUT_SECS", "1800"))
    assert timeout > 0
    request = {
        "start_time": initialization.isoformat().replace("+00:00", "Z"),
        "ifs_member_id": 1,
        "nsteps": 2,
        "nensemble": 2,
        "batch_size": 2,
        "seed_base": 0,
        "output_variables": OUTPUT_VARIABLES,
    }
    endpoint = base_url.rstrip("/")
    submission_path = output / "submission.json"
    with requests.Session() as session:
        # This session has no retry adapter, so an uncertain POST is never repeated.
        session.headers["Authorization"] = "Bearer " + user_token
        if submission_path.exists():
            submission = json.loads(submission_path.read_text())
            assert (
                submission["endpoint_url"] == endpoint
                and submission["request"] == request
            )
            assert submission.get("run_id"), (
                "submission outcome is uncertain; inspect saved evidence before another POST"
            )
        else:
            submission = {
                "endpoint_url": endpoint,
                "request": request,
                "phase": "submitting",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            # Exclusive intent prevents concurrent runs from submitting twice.
            with submission_path.open("x") as stream:
                json.dump(submission, stream, indent=2)
            response = session.post(
                endpoint + adapter.submit_url(WORKFLOW_ID),
                json=request,
                timeout=(10, 45),
                allow_redirects=False,
            )
            submission["http_status"] = response.status_code
            submission["response"] = response.json()
            _write_json(submission_path, submission)
            assert response.status_code == 202, submission["response"]
            submission["run_id"] = submission["response"]["run_id"]
            submission["phase"] = "accepted"
            _write_json(submission_path, submission)
        run_id = submission["run_id"]
        assert isinstance(run_id, str) and re.fullmatch("[A-Za-z0-9-]+", run_id)
        logger.info("AIFS2ENS accepted run %s", run_id)
        deadline = time.monotonic() + timeout
        result = None
        while time.monotonic() < deadline:
            try:
                response = session.get(
                    endpoint + adapter.status_url(WORKFLOW_ID, run_id),
                    timeout=(8, 20),
                    allow_redirects=False,
                )
                response.raise_for_status()
                status = response.json()
                _write_json(output / "status.json", status)
                assert status.get("status") not in {
                    "failed",
                    "cancelled",
                    "canceled",
                }, status
                if status.get("status") == "succeeded":
                    response = session.get(
                        endpoint + adapter.results_url(WORKFLOW_ID, run_id),
                        timeout=(8, 20),
                        allow_redirects=False,
                    )
                    response.raise_for_status()
                    candidate = response.json()
                    if all(
                        isinstance(candidate.get(name), dict)
                        for name in ("request", "execution", "payload")
                    ):
                        result = candidate
                        _write_json(output / "result.json", result)
                        break
                    _write_json(output / "provisional-result.json", candidate)
            except (requests.RequestException, ValueError) as error:
                logger.info(
                    "AIFS2ENS polling run %s encountered %s",
                    run_id,
                    type(error).__name__,
                )
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        assert result is not None, (
            f"AIFS2ENS run {run_id} did not produce a final result"
        )
        _verify_result(result, request, run_id)
        target = output / "forecast_dataset.zarr.zip"
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            download_deadline = time.monotonic() + 180
            with session.get(
                endpoint + adapter.results_url(WORKFLOW_ID, run_id),
                params={"artifact": "forecast_dataset", "format": "zarr_zip"},
                timeout=(8, 90),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                assert response.headers.get("Content-Type", "").startswith(
                    "application/zip"
                )
                size = 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024**2):
                        assert time.monotonic() < download_deadline, (
                            "forecast download timed out"
                        )
                        size += len(chunk)
                        assert size <= 256 * 1024**2
                        stream.write(chunk)
            partial.replace(target)
    fields = _verify_archive(target, request)
    _write_json(
        output / "summary.json",
        {
            "status": "passed",
            "endpoint_url": endpoint,
            "run_id": run_id,
            "request": request,
            "archive": str(target),
            "archive_bytes": target.stat().st_size,
            "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "fields": fields,
        },
    )
