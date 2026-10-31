"""Qualification uses sealed deployment deadlines before starting GPU services."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release_cfd_http_job as cfd_job  # noqa: E402
import native_release_http_job as http_job  # noqa: E402


@pytest.mark.parametrize(
    "deployment, overrides, expected",
    [
        (
            {"startup_timeout_seconds": 900, "request_timeout_seconds": 1800},
            [],
            (900, 1800),
        ),
        (
            {"startup_timeout_seconds": 7200, "request_timeout_seconds": 7200},
            [],
            (7200, 7200),
        ),
        (
            {"startup_timeout_seconds": 1, "request_timeout_seconds": 2},
            [],
            (1, 2),
        ),
        (
            {"startup_timeout_seconds": 900, "request_timeout_seconds": 1800},
            ["--startup-timeout", "60", "--request-timeout", "90"],
            (60, 90),
        ),
        (
            {"startup_timeout_seconds": 900, "request_timeout_seconds": 1800},
            ["--startup-timeout", "1200"],
            (1200, 1800),
        ),
        (
            {"startup_timeout_seconds": 900, "request_timeout_seconds": 1800},
            ["--request-timeout", "2400"],
            (900, 2400),
        ),
        ({"request_timeout_seconds": 600}, [], (300, 600)),
        ({}, [], (300, 300)),
    ],
)
def test_http_cli_uses_deployment_deadlines_and_preserves_explicit_overrides(
    tmp_path, monkeypatch, deployment, overrides, expected
):
    (tmp_path / "deployment.json").write_text(json.dumps(deployment))
    monkeypatch.setattr(http_job, "verify_bundle", lambda root: {})
    captured = []

    class BeforeServiceStart(Exception):
        pass

    def service(args, release, config):
        captured.append((args.startup_timeout, args.request_timeout))
        raise BeforeServiceStart()

    monkeypatch.setattr(http_job, "ServiceJob", service)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_release_http_job.py",
            "--release-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--run-id",
            "deadline-regression",
            "--image",
            "registry/native@sha256:" + "a" * 64,
            *overrides,
        ],
    )
    with pytest.raises(BeforeServiceStart):
        http_job.main()
    assert captured == [expected]


@pytest.mark.parametrize(
    "deployment, expected",
    [
        (
            {"startup_timeout_seconds": 900, "request_timeout_seconds": 1800},
            (900, 1800),
        ),
        (
            {"startup_timeout_seconds": 7200, "request_timeout_seconds": 7200},
            (7200, 7200),
        ),
        ({"request_timeout_seconds": 600}, (300, 600)),
        ({}, (300, 1200)),
    ],
)
def test_cfd_qualification_uses_deployment_deadlines(
    tmp_path, monkeypatch, deployment, expected
):
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text(json.dumps(deployment))
    (tmp_path / "orchestration.json").write_text(
        json.dumps({"reference": {}, "input_root": str(tmp_path / "inputs")})
    )
    monkeypatch.setenv("NATIVE_DEPLOYMENT_CONFIG", str(deployment_path))
    monkeypatch.setattr(
        cfd_job,
        "verify_bundle",
        lambda root: {"source_sha": "a" * 40, "workflow": cfd_job.scientific.WORKFLOW},
    )
    monkeypatch.setattr(
        cfd_job, "verify_reference", lambda record: tmp_path / "reference.json"
    )
    captured = []
    monkeypatch.setattr(
        cfd_job.scientific,
        "run",
        lambda cfg, **kwargs: captured.append(
            (cfg.startup_timeout, cfg.request_timeout)
        ),
    )
    cfd_job.run(
        SimpleNamespace(
            output=tmp_path / "output",
            run_id="deadline-regression",
            image="registry/native@sha256:" + "a" * 64,
        )
    )
    assert captured == [expected]
