# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "qa" / "scripts"))

import compare_helpers  # noqa: E402
import run_lepton_crps_compare as crps_runner  # noqa: E402
from compare_helpers import (  # noqa: E402
    WorkflowRun,
    extract_forecast_zarr_path,
    fetch_results_with_zarr_path,
)
from run_lepton_crps_compare import (  # noqa: E402
    default_compare_endpoint_name,
    parse_crps_report,
    split_image_reference,
)


def test_extract_forecast_zarr_path_prefers_primary_storage_path():
    payload = {
        "execution": {
            "outputs": [
                {
                    "name": "secondary",
                    "storage_path": "/outputs/run-1/secondary.zarr",
                    "primary": False,
                },
                {
                    "name": "forecast",
                    "storage_path": "/outputs/run-1/forecast.zarr",
                    "primary": True,
                },
            ]
        }
    }

    assert (
        extract_forecast_zarr_path(payload, mount_target="/outputs")
        == "/outputs/run-1/forecast.zarr"
    )


def test_extract_forecast_zarr_path_falls_back_to_payload_dataset_path():
    payload = {"payload": {"dataset_path": "/outputs/run-2/forecast.zarr"}}

    assert (
        extract_forecast_zarr_path(payload, mount_target="/outputs")
        == "/outputs/run-2/forecast.zarr"
    )


def test_extract_forecast_zarr_path_infers_relative_zarr_root_from_output_files():
    payload = {
        "request_id": "exec-1",
        "output_files": [
            {"path": "exec-1/forecast_metadata.json", "size": 123},
            {"path": "exec-1/results.zarr/zarr.json", "size": 456},
            {"path": "exec-1/results.zarr/t2m/c/0/0/0", "size": 789},
        ],
    }

    assert (
        extract_forecast_zarr_path(payload, mount_target="/outputs")
        == "/outputs/exec-1/results.zarr"
    )


def test_extract_forecast_zarr_path_adds_workflow_prefix_for_python_results():
    payload = {
        "request_id": "exec-1",
        "workflow_name": "ensemble_workflow",
        "output_files": [
            {"path": "exec-1/forecast_metadata.json", "size": 123},
            {"path": "exec-1/results.zarr/zarr.json", "size": 456},
        ],
    }

    assert (
        extract_forecast_zarr_path(payload, mount_target="/outputs")
        == "/outputs/ensemble_workflow/exec-1/results.zarr"
    )


def test_extract_forecast_zarr_path_infers_absolute_zarr_root_from_output_files():
    payload = {
        "output_files": [
            {"path": "/outputs/exec-2/results.zarr/zarr.json", "size": 456},
        ],
    }

    assert (
        extract_forecast_zarr_path(payload, mount_target="/outputs")
        == "/outputs/exec-2/results.zarr"
    )


def test_extract_forecast_zarr_path_rejects_paths_outside_mount():
    payload = {
        "execution": {
            "outputs": [
                {
                    "name": "forecast",
                    "storage_path": "/tmp/run-1/forecast.zarr",
                    "primary": True,
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="valid forecast Zarr path"):
        extract_forecast_zarr_path(payload, mount_target="/outputs")


def test_split_image_reference_accepts_full_image_or_tag_only():
    full = split_image_reference("nvcr.io/test-org/earth2studio:v0.15.0.20260515.0")
    assert full.image_name == "nvcr.io/test-org/earth2studio"
    assert full.image_tag == "v0.15.0.20260515.0"

    tag_only = split_image_reference("v0.1.0")
    assert tag_only.image_name is None
    assert tag_only.image_tag == "v0.1.0"


def test_default_compare_endpoint_name_is_meaningful_and_valid():
    baseline = default_compare_endpoint_name("base", "earth2studio", "20260527abcdef")
    candidate = default_compare_endpoint_name(
        "candidate", "physicsnemo-serve", "20260527abcdef"
    )

    assert baseline == "crps-e2s-python-base-20260527abcdef"
    assert candidate == "crps-e2s-rust-ff-20260527abcdef"
    assert len(baseline) <= 36
    assert len(candidate) <= 36
    assert baseline[-1].isalnum()
    assert candidate[-1].isalnum()


def test_crps_report_reader_job_name_fits_lepton_limit(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_lep = fake_bin / "lep"
    fake_lep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_lep.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "IMAGE_NAME": "nvcr.io/test/earth2studio",
        "LEPTON_NFS_PATH": "/shared/crps",
    }
    result = subprocess.run(
        [
            str(REPO_ROOT / "qa" / "scripts" / "submit-lepton-crps-job.sh"),
            "--job-name",
            "ff-crps-rogcl016-scheduled-gpu",
            "--image-tag",
            "nvcr.io/test/earth2studio:test",
            "--forecast-a",
            "/outputs/baseline.zarr",
            "--forecast-b",
            "/outputs/candidate.zarr",
            "--workspace-id",
            "test-workspace",
            "--node-group",
            "test-node-group",
            "--pull-secret",
            "test-pull-secret",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report_job_line = next(
        line for line in result.stdout.splitlines() if "report-job" in line
    )
    report_job_name = report_job_line.split(":", 1)[1].strip()
    assert report_job_name == "ff-crps-rogcl016-scheduled-gp-report"
    assert len(report_job_name) <= 36


def test_parse_crps_report_extracts_final_summary(tmp_path):
    log_path = tmp_path / "crps.log"
    log_path.write_text(
        """
noise before report
================================================================================
CRPS COMPARISON REPORT
================================================================================
Variable     Lead Time          CRPS A     CRPS B  Rel Diff%   Status
--------------------------------------------------------------------------------
t2m              24h           0.9726     0.9724    0.0119%     PASS
--------------------------------------------------------------------------------
Max relative CRPS difference: 0.0434%
Threshold:                    1.0000%
Result:                       PASS
================================================================================

PASSED: Both systems produce equivalent CRPS scores.
EXIT=0
""",
        encoding="utf-8",
    )

    report = parse_crps_report(log_path)

    assert report["available"] is True
    assert report["result"] == "PASS"
    assert report["max_relative_crps_difference"] == "0.0434%"
    assert report["max_relative_diff_percent"] == 0.0434
    assert report["threshold_percent"] == 1.0
    assert "CRPS COMPARISON REPORT" in report["report_text"]


def test_fetch_results_with_zarr_path_retries_until_manifest_ready(
    monkeypatch, tmp_path
):
    payloads = [
        {"status": "pending_results", "output_files": []},
        {
            "workflow_name": "ensemble_workflow",
            "output_files": [
                {"path": "exec-1/results.zarr/zarr.json", "size": 123},
            ],
        },
    ]

    def fake_fetch_results(**_kwargs):
        return payloads.pop(0)

    monkeypatch.setattr(compare_helpers, "fetch_results", fake_fetch_results)
    monkeypatch.setattr(compare_helpers.time, "sleep", lambda _seconds: None)

    results_payload, zarr_path = fetch_results_with_zarr_path(
        client=object(),
        base_url="https://example.invalid",
        adapter=object(),
        workflow="ensemble_workflow",
        execution_id="exec-1",
        artifact_dir=tmp_path,
        label="baseline",
        mount_target="/outputs",
        timeout_seconds=60,
        interval_seconds=1,
    )

    assert results_payload["workflow_name"] == "ensemble_workflow"
    assert zarr_path == "/outputs/ensemble_workflow/exec-1/results.zarr"
    assert (tmp_path / "baseline-results.json").is_file()


def test_run_tears_down_endpoints_after_outputs_before_crps_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_request = tmp_path / "baseline.json"
    candidate_request = tmp_path / "candidate.json"
    baseline_request.write_text(json.dumps({"input": "baseline"}), encoding="utf-8")
    candidate_request.write_text(json.dumps({"input": "candidate"}), encoding="utf-8")
    events: list[str] = []
    deployment_envs: dict[str, dict[str, str]] = {}

    def fake_deploy_endpoint(**kwargs):
        config = kwargs["config"]
        deployment_envs[config.label] = kwargs["container_env"]
        return f"https://{config.label}.example.test", config.endpoint_name

    def fake_submit_endpoint_workflow(**kwargs):
        label = kwargs["label"]
        events.append(f"{label}-submitted")
        return f"{label}-execution", {"run_id": f"{label}-execution"}

    def fake_finish_endpoint_workflow(**kwargs):
        label = kwargs["label"]
        events.append(f"{label}-outputs")
        return WorkflowRun(
            label=label,
            workflow=kwargs["workflow"],
            execution_id=kwargs["execution_id"],
            submit_payload=kwargs["submit_payload"],
            final_status_payload={"status": "completed"},
            results_payload={"execution": {"outputs": []}},
            forecast_zarr_path=f"/outputs/{label}/forecast.zarr",
        )

    def fake_teardown_endpoint(**kwargs):
        events.append(f"teardown:{kwargs['endpoint_name']}")

    def fake_submit_crps_job(**_kwargs):
        events.append("crps-job")
        return 0, "ff-crps-test-run"

    monkeypatch.setenv("LEPTON_WORKSPACE_TOKEN", "workspace-token")
    monkeypatch.setattr(crps_runner, "get_or_generate_endpoint_token", lambda: "token")
    monkeypatch.setattr(crps_runner, "deploy_endpoint", fake_deploy_endpoint)
    monkeypatch.setattr(crps_runner, "health_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(crps_runner, "make_client", lambda _token: object())
    monkeypatch.setattr(crps_runner, "get_adapter", lambda _service: object())
    monkeypatch.setattr(
        crps_runner,
        "submit_endpoint_workflow",
        fake_submit_endpoint_workflow,
    )
    monkeypatch.setattr(
        crps_runner,
        "finish_endpoint_workflow",
        fake_finish_endpoint_workflow,
    )
    monkeypatch.setattr(crps_runner, "teardown_endpoint", fake_teardown_endpoint)
    monkeypatch.setattr(crps_runner, "submit_crps_job", fake_submit_crps_job)
    monkeypatch.setattr(
        crps_runner,
        "parse_crps_report",
        lambda _path: {"available": False},
    )

    args = SimpleNamespace(
        baseline_service="python",
        candidate_service="rust",
        baseline_source="earth2studio",
        candidate_source="physicsnemo-serve",
        baseline_image_tag="baseline:image",
        candidate_image_tag="candidate:image",
        baseline_workflow="ensemble_workflow",
        candidate_workflow="earth2-ensemble-fanout",
        request_json=None,
        baseline_request_json=str(baseline_request),
        candidate_request_json=str(candidate_request),
        candidate_materialization_modes="scheduled_gpu",
        comparison_image_tag=None,
        comparison_script="/compare_crps.py",
        threshold="0.01",
        variables=None,
        device="cuda",
        lead_time_chunk_size="1",
        workspace_id="workspace",
        workspace_url="",
        lustre_dir="crps-test",
        lustre_storage="lustre",
        mount_target="/outputs",
        node_group=None,
        resource_shape=None,
        baseline_resource_shape=None,
        candidate_resource_shape="gpu.h100-sxm",
        pull_secret=None,
        baseline_endpoint_name="baseline-endpoint",
        candidate_endpoint_name="candidate-endpoint",
        run_id="test-run",
        skip_teardown=False,
        keep_batch_job=False,
        dry_run=False,
        stream_endpoint_logs=False,
        endpoint_log_interval=30,
        run_timeout=60,
        run_poll_interval=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert crps_runner.run(args) == 0

    assert deployment_envs == {
        "baseline": {
            "DEFAULT_OUTPUT_DIR": "/outputs",
            "RESULTS_ZIP_DIR": "/outputs",
        },
        "candidate": {
            "DEFAULT_OUTPUT_DIR": "/outputs",
            "RESULTS_ZIP_DIR": "/outputs",
            "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": "earth2-ensemble-fanout",
        },
    }
    assert events == [
        "baseline-submitted",
        "candidate-scheduled-gpu-submitted",
        "baseline-outputs",
        "candidate-scheduled-gpu-outputs",
        "teardown:baseline-endpoint",
        "teardown:candidate-endpoint",
        "crps-job",
    ]
