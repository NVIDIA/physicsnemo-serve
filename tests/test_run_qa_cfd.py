# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "scripts"))

import run_qa  # noqa: E402


def test_cfd_e2e_defaults_to_only_surface_workflow(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)

    assert run_qa.determine_workflows(
        service="rust",
        workflows_arg=None,
        suite="cfd_e2e",
    ) == [run_qa.CFD_E2E_WORKFLOW_ID]


def test_cfd_e2e_container_environment_is_exact_and_timeout_is_validated(monkeypatch):
    monkeypatch.delenv("QA_CFD_E2E_DOWNLOAD_TIMEOUT_SECS", raising=False)
    assert run_qa.cfd_e2e_container_envs() == [
        "PHYSICSNEMO_SERVE_EXECUTOR_CLASSES=physicsnemo-cfd-gpu",
        "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS=huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS=us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_EXT_CACHE=/outputs/.cache/physicsnemo-cfd",
        "E2S_DOWNLOAD_TIMEOUT_SECS=1800",
    ]

    monkeypatch.setenv("QA_CFD_E2E_DOWNLOAD_TIMEOUT_SECS", "2400")
    assert run_qa.cfd_e2e_container_envs()[-1] == "E2S_DOWNLOAD_TIMEOUT_SECS=2400"

    monkeypatch.setenv("QA_CFD_E2E_DOWNLOAD_TIMEOUT_SECS", "0")
    try:
        run_qa.cfd_e2e_container_envs()
    except SystemExit as exc:
        assert "must be greater than zero" in str(exc)
    else:
        raise AssertionError("zero download timeout must be rejected")


def test_cfd_e2e_deployment_and_pytest_receive_required_environment(
    monkeypatch, tmp_path
):
    deploy_calls = []
    pytest_calls = []

    def fake_deploy(**kwargs):
        deploy_calls.append(kwargs)
        return "https://endpoint.example.test", "cfd-test-endpoint"

    monkeypatch.setattr(run_qa, "deploy", fake_deploy)
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    def fake_run_pytest(**kwargs):
        pytest_calls.append(kwargs)
        return 0

    monkeypatch.setattr(run_qa, "run_pytest", fake_run_pytest)
    container_envs = [
        "PHYSICSNEMO_SERVE_EXECUTOR_CLASSES=physicsnemo-cfd-gpu",
        "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS=huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS=us.aws.cdn.hf.co,cas-bridge.xethub.hf.co",
        "E2S_EXT_CACHE=/outputs/.cache/physicsnemo-cfd",
        "E2S_DOWNLOAD_TIMEOUT_SECS=1800",
    ]

    result = run_qa.run_one_workflow(
        workflow_id=run_qa.CFD_E2E_WORKFLOW_ID,
        source="physicsnemo-serve",
        service="rust",
        image_tag="test-image",
        workspace_id="test-workspace",
        workspace_token="test-token",
        endpoint_token="endpoint-token",
        nfs_path="/mnt/shared/test",
        suite="cfd_e2e",
        test_filter=None,
        num_proc=1,
        skip_teardown=False,
        stream_logs=False,
        log_interval=30,
        artifact_dir=tmp_path,
        post_health_wait_secs=0,
        container_envs=container_envs,
    )

    assert result == 0
    assert len(deploy_calls) == 1
    deployed_envs = deploy_calls[0]["container_envs"]
    assert set(container_envs) <= set(deployed_envs)
    assert (
        f"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={run_qa.CFD_E2E_WORKFLOW_ID}"
        in deployed_envs
    )

    assert len(pytest_calls) == 1
    assert pytest_calls[0]["suite"] == "cfd_e2e"
    assert pytest_calls[0]["num_proc"] == 1
    assert pytest_calls[0]["extra_env"] == {
        "QA_CFD_E2E_ENABLED": "1",
        "QA_CFD_E2E_ARTIFACT_DIR": str(tmp_path / "cfd-e2e"),
        "QA_CFD_E2E_TIMEOUT_SECS": "23400",
        "QA_CFD_E2E_POLL_SECS": "20",
        "QA_CFD_E2E_SUBMIT_TIMEOUT_SECS": "300",
    }


def test_cfd_e2e_main_wires_single_workflow_and_container_policy(monkeypatch, tmp_path):
    workflow_calls = []
    monkeypatch.setattr(run_qa, "_ensure_line_buffered", lambda: None)
    monkeypatch.setattr(
        run_qa,
        "load_deploy_config",
        lambda: {
            "lepton_workspace_id": "test-workspace",
            "nfs_mount_base": "/mnt/shared",
        },
    )
    monkeypatch.setattr(run_qa.signal, "signal", lambda _signal, _handler: None)
    monkeypatch.setattr(
        run_qa,
        "run_one_workflow",
        lambda **kwargs: workflow_calls.append(kwargs) or 0,
    )
    monkeypatch.setenv("LEPTON_WORKSPACE_TOKEN", "workspace-token")
    monkeypatch.setenv("LEPTON_ENDPOINT_TOKEN", "endpoint-token")
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qa.py",
            "--service",
            "rust",
            "--image-tag",
            "test-image",
            "--suite",
            "cfd_e2e",
            "--artifact-dir",
            str(tmp_path),
            "--no-endpoint-logs",
        ],
    )

    try:
        run_qa.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("main() must exit with the QA result code")

    assert len(workflow_calls) == 1
    assert workflow_calls[0]["workflow_id"] == run_qa.CFD_E2E_WORKFLOW_ID
    assert workflow_calls[0]["suite"] == "cfd_e2e"
    assert workflow_calls[0]["num_proc"] == 1
    assert workflow_calls[0]["container_envs"] == run_qa.cfd_e2e_container_envs()
