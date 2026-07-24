# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_PATH = REPO_ROOT / "qa" / "inference" / "test_physicsnemo_cfd_surface_e2e.py"
FULL_REQUEST_PATH = (
    REPO_ROOT
    / "plugins"
    / "physicsnemo-cfd-surface-benchmark"
    / "examples"
    / "public_run_1_full_matrix_request.json"
)


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location("cfd_surface_e2e_contract", E2E_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unknown pytest\.mark\.cfd_e2e",
        )
        spec.loader.exec_module(module)
    return module


def _full_report(request: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for model in request["models"]:
        metrics = {
            "l2_pressure": 0.16,
            "l2_shear_stress_wall_shear_stress_true_x_l2_error": 0.2,
            "l2_shear_stress_wall_shear_stress_true_y_l2_error": 0.3,
            "l2_shear_stress_wall_shear_stress_true_z_l2_error": 0.4,
            "l2_pressure_area_weighted": 0.17,
            "drag_error": 0.01,
            "drag_true": 510.0,
            "drag_pred": 511.0,
            "lift_error": 0.02,
            "lift_true": 113.0,
            "lift_pred": 114.0,
        }
        rows.append(
            {
                "model": model,
                "dataset": "drivaerml",
                "cases": ["run_1"],
                "metrics": metrics,
                "per_case": [
                    {
                        "case_id": "run_1",
                        "metrics": metrics,
                        "metric_dtype": "cell",
                    }
                ],
            }
        )
    return rows


def test_full_matrix_report_contract_accepts_all_25_tuples() -> None:
    module = _load_e2e_module()
    request = json.loads(FULL_REQUEST_PATH.read_text(encoding="utf-8"))

    module._assert_report(_full_report(request), request)


def test_full_matrix_report_contract_rejects_missing_metric() -> None:
    module = _load_e2e_module()
    request = json.loads(FULL_REQUEST_PATH.read_text(encoding="utf-8"))
    report = _full_report(request)
    report[0]["metrics"].pop("lift_error")

    with pytest.raises(AssertionError):
        module._assert_report(report, request)


def test_e2e_request_path_can_be_selected_by_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_e2e_module()
    request = tmp_path / "matrix.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("QA_CFD_E2E_REQUEST_PATH", str(request))

    assert module._request_path() == request


def test_submission_warms_no_retry_session_before_post(monkeypatch) -> None:
    module = _load_e2e_module()
    events: list[str] = []
    response = object()

    class WarmupResponse:
        def raise_for_status(self) -> None:
            events.append("warmup-ok")

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, _url, *, timeout):
            assert timeout == 10
            events.append("get")
            return WarmupResponse()

        def post(self, _url, *, json, timeout):
            assert json == {"parameters": {"models": []}}
            assert timeout == 30
            events.append("post")
            return response

    monkeypatch.setattr(module.requests, "Session", Session)

    assert (
        module._submit_without_retry(
            base_url="https://endpoint",
            user_token="secret-token",
            request_payload={"models": []},
            timeout=30,
            warmup_attempts=3,
        )
        is response
    )
    assert events == ["get", "warmup-ok", "post"]


def test_submission_transport_failure_does_not_expose_token(monkeypatch) -> None:
    module = _load_e2e_module()

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, _url, *, timeout):
            raise module.requests.ReadTimeout(f"timed out after {timeout}")

    monkeypatch.setattr(module.requests, "Session", Session)

    with pytest.raises(pytest.fail.Exception) as error:
        module._submit_without_retry(
            base_url="https://endpoint",
            user_token="secret-token",
            request_payload={"models": []},
            timeout=30,
            warmup_attempts=1,
        )
    assert "secret-token" not in str(error.value)


def test_submission_retries_only_safe_warmup_get(monkeypatch) -> None:
    module = _load_e2e_module()
    attempts = 0
    posts = 0

    class WarmupResponse:
        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, _url, *, timeout):
            nonlocal attempts
            assert timeout == 10
            attempts += 1
            if attempts < 3:
                raise module.requests.ReadTimeout("TLS handshake timeout")
            return WarmupResponse()

        def post(self, _url, *, json, timeout):
            nonlocal posts
            posts += 1
            return json, timeout

    monkeypatch.setattr(module.requests, "Session", Session)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._submit_without_retry(
        base_url="https://endpoint",
        user_token="secret-token",
        request_payload={"models": []},
        timeout=30,
        warmup_attempts=4,
    )

    assert result == ({"parameters": {"models": []}}, 30)
    assert attempts == 3
    assert posts == 1
