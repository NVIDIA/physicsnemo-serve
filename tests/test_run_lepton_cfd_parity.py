# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "qa" / "scripts"
PROFILE_PATH = REPO_ROOT / "qa" / "inference" / "cfd_parity_surface_run1.json"
IMAGE_DIGEST = "registry/image@sha256:" + ("d" * 64)
INFER_IMAGE_DIGEST = "registry/physicsnemo-serve-cmd@sha256:" + ("e" * 64)
sys.path.insert(0, str(SCRIPTS_DIR))

import cfd_parity_contract as contract  # noqa: E402
import run_lepton_cfd_parity as orchestrator  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(tmp_path: Path) -> Path:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    evidence = tmp_path / "evidence"
    request = {
        "models": ["domino_surface"],
        "cases": [
            {
                "case_id": "run_1",
                "sha256": "a" * 64,
                "size_bytes": 10,
                "geometry_sha256": "b" * 64,
                "geometry_size_bytes": 20,
            }
        ],
        "metrics": ["l2_pressure"],
        "seed": 42,
        "save_inference_mesh": False,
        "visual_case_ids": [],
    }
    report = [
        {
            "model": "domino_surface",
            "dataset": "drivaerml",
            "cases": ["run_1"],
            "metrics": {"l2_pressure": 0.16},
            "per_case": [
                {
                    "case_id": "run_1",
                    "metrics": {"l2_pressure": 0.16},
                    "metric_dtype": "cell",
                }
            ],
        }
    ]
    _write_json(evidence / "request.json", request)
    _write_json(evidence / "benchmark_results.json", report)
    _write_json(
        evidence / "resolved_config.json",
        {
            "benchmark": {
                "datasets": [
                    {
                        "name": "drivaerml",
                        "root": "/outputs/rest-run/attempt/inputs/drivaerml",
                    }
                ]
            }
        },
    )
    _write_json(
        evidence / "results.json",
        {
            "execution": {
                "run_id": "rest-run",
                "workflow": profile["workflow_id"],
                "status": "succeeded",
                "outputs": [
                    {
                        "name": "benchmark_results.json",
                        "storage_path": (
                            "/outputs/rest-run/attempt/benchmark-output/"
                            "benchmark_results.json"
                        ),
                    },
                    {
                        "name": "resolved_config.json",
                        "storage_path": (
                            "/outputs/rest-run/attempt/resolved_config.json"
                        ),
                    },
                ],
            },
            "payload": {
                "provider": {
                    key: profile["provider"][key]
                    for key in (
                        "repository",
                        "tag",
                        "version",
                        "commit",
                        "physicsnemo_version",
                        "python_version",
                    )
                },
                "model_names": request["models"],
                "case_ids": ["run_1"],
                "selected_metrics": request["metrics"],
                "preset_sha256": profile["preset_sha256"],
                "case_digests": [
                    {
                        key: request["cases"][0][key]
                        for key in (
                            "case_id",
                            "sha256",
                            "size_bytes",
                            "geometry_sha256",
                            "geometry_size_bytes",
                        )
                    }
                ],
            },
        },
    )
    return evidence


def _args(tmp_path: Path, evidence: Path) -> argparse.Namespace:
    return argparse.Namespace(
        image_tag=IMAGE_DIGEST,
        image_name="ignored",
        infer_image_name="registry/physicsnemo-serve-cmd",
        candidate="rest",
        profile=str(PROFILE_PATH),
        rest_request_path=None,
        rest_evidence_dir=str(evidence),
        infer_binary="/usr/local/bin/physicsnemo-serve",
        infer_runtime_dir="/opt/physicsnemo-serve/runtimes/shared",
        infer_plugin=(
            "/opt/physicsnemo-serve/plugins/physicsnemo-cfd-surface-benchmark"
        ),
        device="0",
        infer_timeout=60,
        download_timeout=60,
        run_id="parity-test",
        artifact_dir=str(tmp_path / "artifacts"),
        workspace_id="workspace",
        workspace_token="",
        workspace_url="",
        node_group="node-group",
        resource_shape="gpu.h100-sxm",
        pull_secret="pull-secret",
        nfs_mount_base=str(tmp_path / "shared"),
        lustre_dir="user",
        lustre_storage="lustre",
        mount_target="/outputs",
        job_timeout=60,
        job_poll_interval=1,
        reader_resource_shape="cpu.small",
        keep_job=False,
        dry_run=True,
    )


def test_dry_run_persists_handoff_and_extensible_job_spec(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)

    assert orchestrator.run(args) == 0

    run_dir = tmp_path / "artifacts" / "cfd-parity" / "parity-test"
    handoff = contract.read_json_object(run_dir / "parity-handoff.json")
    summary = contract.read_json_object(run_dir / "summary.json")
    assert handoff["rest"]["input_root_relpath"].endswith("inputs/drivaerml")
    assert handoff["rest"]["report_relpath"].endswith("benchmark_results.json")
    assert summary["final_result"] == "dry-run"
    assert summary["job"]["mount"] == (
        f"{tmp_path / 'shared' / 'user'}:/outputs:node-nfs:lustre"
    )
    assert len(summary["job"]["remote_command_sha256"]) == 64
    assert "physicsnemo-cfd-surface-domino-run1-v1" in json.dumps(summary)


def test_rest_qa_uses_request_path_override(tmp_path: Path, monkeypatch) -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    args = _args(tmp_path, _evidence(tmp_path))
    request = tmp_path / "run-1-11.json"
    request.write_text("{}", encoding="utf-8")
    args.rest_request_path = str(request)
    artifact_dir = tmp_path / "rest-qa"
    evidence = artifact_dir / "cfd-e2e" / "rest-run"
    for name in (
        "request.json",
        "results.json",
        "resolved_config.json",
        "benchmark_results.json",
    ):
        _write_json(evidence / name, {})

    def fake_streaming(_command, *, env, artifact_path):
        assert env["QA_CFD_E2E_REQUEST_PATH"] == str(request)
        assert artifact_path == artifact_dir.parent / "rest-qa.log"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)

    assert (
        orchestrator.run_rest_qa(
            args,
            profile=profile,
            image=IMAGE_DIGEST,
            artifact_dir=artifact_dir,
            env={},
        )
        == evidence
    )


def test_remote_command_creates_fresh_root_and_uses_profile_runner() -> None:
    profile_text = PROFILE_PATH.read_text(encoding="utf-8")
    profile = json.loads(profile_text)

    command = orchestrator.build_remote_command(
        profile=profile,
        profile_text=profile_text,
        handoff_text="{}\n",
        mount_target="/outputs",
        run_id="abc123",
    )

    assert "test ! -e /outputs/cfd-parity/abc123" in command
    assert "/opt/physicsnemo-cfd-venv/bin/python" in command
    assert "run_cfd_parity_job.py" in command
    assert command.startswith("set -euo pipefail;")
    assert "PIPESTATUS[0]" in command


def test_infer_dry_run_injects_cli_candidate_job(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.candidate = "infer"
    args.rest_evidence_dir = None
    args.image_tag = INFER_IMAGE_DIGEST

    assert orchestrator.run(args) == 0

    run_dir = tmp_path / "artifacts" / "cfd-parity" / "parity-test"
    summary = contract.read_json_object(run_dir / "summary.json")
    assert summary["candidate"] == "infer"
    assert summary["final_result"] == "dry-run"
    assert summary["job"]["name"].startswith("pn-cfd-infer-parity")
    assert summary["job"]["timeout_seconds"] == 60
    assert (run_dir / "request.json").is_file()

    profile_text = PROFILE_PATH.read_text(encoding="utf-8")
    command = orchestrator.build_infer_remote_command(
        profile_text=profile_text,
        request_text=(run_dir / "request.json").read_text(encoding="utf-8"),
        mount_target="/outputs",
        run_id="abc123",
        image=INFER_IMAGE_DIGEST,
        infer_binary="/usr/local/bin/physicsnemo-serve",
        infer_runtime_dir="/opt/physicsnemo-serve/runtimes/shared",
        infer_plugin=(
            "/opt/physicsnemo-serve/plugins/physicsnemo-cfd-surface-benchmark"
        ),
        device="0",
        infer_timeout_seconds=60,
        download_timeout_seconds=60,
    )
    assert "run_cfd_infer_parity_job.py" in command
    assert command.startswith("set -euo pipefail;")
    assert "/opt/physicsnemo-serve/runtimes/shared/bin/python" in command
    assert "/usr/local/bin/physicsnemo-serve" in command
    assert "--infer-run-id infer-abc123" in command


def test_run_ids_are_path_safe_and_long_job_names_keep_unique_suffixes() -> None:
    with pytest.raises(ValueError, match="run ID"):
        orchestrator.validate_run_id("../escape")

    first = orchestrator._lepton_job_name(
        "pn-cfd-parity-surface",
        "long-shared-prefix-that-would-otherwise-collide-one",
    )
    second = orchestrator._lepton_job_name(
        "pn-cfd-parity-surface",
        "long-shared-prefix-that-would-otherwise-collide-two",
    )
    assert len(first) <= 36
    assert len(second) <= 36
    assert first != second


def test_parity_requires_an_immutable_image_reference() -> None:
    bare_digest = "sha256:" + ("f" * 64)
    assert (
        orchestrator.image_full_reference(bare_digest, "registry/image")
        == f"registry/image@{bare_digest}"
    )
    assert orchestrator.image_full_reference(IMAGE_DIGEST, "ignored") == IMAGE_DIGEST

    for mutable in ("latest", "registry/image:latest"):
        with pytest.raises(contract.ParityContractError, match="immutable image"):
            orchestrator.image_full_reference(mutable, "registry/image")


def test_discover_rest_evidence_requires_exactly_one_run(tmp_path: Path) -> None:
    root = tmp_path / "cfd-e2e"
    root.mkdir()
    with pytest.raises(contract.ParityContractError, match="exactly one"):
        orchestrator.discover_rest_evidence(root)

    evidence = root / "run-1"
    for name in (
        "request.json",
        "results.json",
        "resolved_config.json",
        "benchmark_results.json",
    ):
        _write_json(evidence / name, {})
    assert orchestrator.discover_rest_evidence(root) == evidence

    second = root / "run-2"
    for name in (
        "request.json",
        "results.json",
        "resolved_config.json",
        "benchmark_results.json",
    ):
        _write_json(second / name, {})
    with pytest.raises(contract.ParityContractError, match="found 2"):
        orchestrator.discover_rest_evidence(root)


def test_extract_marked_summary_ignores_unmarked_logs() -> None:
    payload = {"final_result": "passed", "comparison": {"status": "passed"}}
    output = (
        "noise\n"
        f"{orchestrator.SUMMARY_BEGIN}\n"
        f"{json.dumps(payload)}\n"
        f"{orchestrator.SUMMARY_END}\n"
    )
    assert orchestrator.extract_marked_summary(output) == payload
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    wrapped = "\n".join(
        encoded[index : index + 20] for index in range(0, len(encoded), 20)
    )
    assert (
        orchestrator.extract_marked_summary(
            f"{orchestrator.SUMMARY_BEGIN}\n{wrapped}\n{orchestrator.SUMMARY_END}\n"
        )
        == payload
    )
    assert orchestrator.extract_marked_summary("noise only") is None


def test_reader_job_fetches_remote_summary_and_cleans_up(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, _evidence(tmp_path))
    calls: list[list[str]] = []
    payload = {"final_result": "passed", "comparison": {"status": "passed"}}

    def fake_streaming(command, **_kwargs):
        calls.append(command)
        if command[:3] == ["lep", "job", "create"]:
            return 0, "ID: reader-job-id\n"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)
    monkeypatch.setattr(orchestrator, "wait_for_loggable_job", lambda **_kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "capture_job_logs",
        lambda **_kwargs: (
            f"{orchestrator.SUMMARY_BEGIN}\n"
            f"{json.dumps(payload)}\n"
            f"{orchestrator.SUMMARY_END}\n"
        ),
    )

    result = orchestrator.fetch_summary_via_reader_job(
        args,
        image=IMAGE_DIGEST,
        nfs_path="/shared/user",
        run_id="run-1",
        env={},
        artifact_dir=tmp_path / "artifacts",
    )

    assert result == payload
    assert any(command[:3] == ["lep", "job", "stop"] for command in calls)
    assert any(command[:3] == ["lep", "job", "remove"] for command in calls)
