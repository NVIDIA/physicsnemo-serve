# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "scripts"))

import run_qa  # noqa: E402


class FakeResponse:
    def __init__(self, body: str, *, status_code: int = 200):
        self.status_code = status_code
        self.text = body

    def json(self):
        import json

        return json.loads(self.text)


def test_health_check_waits_for_backend_after_loading_stub(monkeypatch):
    responses = iter(
        [
            FakeResponse('{"status":"loading"}'),
            FakeResponse("ok"),
        ]
    )
    requests = []
    sleeps = []

    def fake_get(url, *, headers, timeout):
        requests.append((url, headers, timeout))
        return next(responses)

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=fake_get,
            ConnectionError=ConnectionError,
            Timeout=TimeoutError,
        ),
    )
    monkeypatch.setattr(run_qa.time, "sleep", sleeps.append)

    assert run_qa.health_check("https://endpoint.example.test", "secret")
    assert len(requests) == 2
    assert sleeps == [run_qa.HEALTH_POLL_INTERVAL]


def test_health_readiness_accepts_python_healthy_response():
    ready, detail = run_qa._health_response_ready(
        FakeResponse('{"status":"healthy","timestamp":"2026-08-27T00:00:00Z"}')
    )

    assert ready
    assert detail == "healthy"


def test_health_readiness_rejects_unexpected_200_response():
    ready, detail = run_qa._health_response_ready(FakeResponse("starting"))

    assert not ready
    assert "unexpected body" in detail
