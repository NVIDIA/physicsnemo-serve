# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "scripts"))

import run_qa  # noqa: E402


@pytest.mark.parametrize(
    ("service", "workflow_id", "suite", "expected"),
    [
        (
            "rust",
            "e2s-example-user",
            "cicd",
            ["PHYSICSNEMO_SERVE_MAX_BATCH_PARALLEL_ITEMS=4"],
        ),
        (
            "rust",
            "e2s-example-user",
            "full",
            ["PHYSICSNEMO_SERVE_MAX_BATCH_PARALLEL_ITEMS=4"],
        ),
        ("rust", "e2s-example-user", "smoke", []),
        ("rust", "earth2-deterministic", "cicd", []),
        ("python", "e2s-example-user", "cicd", []),
    ],
)
def test_parallel_batch_qa_container_envs_are_scoped(
    service: str,
    workflow_id: str,
    suite: str,
    expected: list[str],
) -> None:
    assert (
        run_qa.parallel_batch_qa_container_envs(
            service=service,
            workflow_id=workflow_id,
            suite=suite,
        )
        == expected
    )


def test_parallel_batch_env_is_passed_to_example_user_deployment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deploy_calls = []

    def fake_deploy(**kwargs):
        deploy_calls.append(kwargs)
        return "https://endpoint.example.test", "batch-test-endpoint"

    monkeypatch.setattr(run_qa, "deploy", fake_deploy)
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "run_pytest", lambda **_kwargs: 0)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    result = run_qa.run_one_workflow(
        workflow_id="e2s-example-user",
        source="physicsnemo-serve",
        service="rust",
        image_tag="test-image",
        workspace_id="test-workspace",
        workspace_token="test-token",
        endpoint_token="endpoint-token",
        nfs_path="/mnt/shared/test",
        suite="cicd",
        test_filter="test_batch_coordinator_executes_four_items_in_parallel",
        num_proc=1,
        skip_teardown=False,
        stream_logs=False,
        log_interval=30,
        artifact_dir=tmp_path,
        post_health_wait_secs=0,
        container_envs=["EXISTING_SETTING=value"],
    )

    assert result == 0
    assert len(deploy_calls) == 1
    assert deploy_calls[0]["container_envs"] == [
        "EXISTING_SETTING=value",
        "PHYSICSNEMO_SERVE_MAX_BATCH_PARALLEL_ITEMS=4",
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=e2s-example-user",
    ]
