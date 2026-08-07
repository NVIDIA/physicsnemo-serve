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
PROFILE_PATH = REPO_ROOT / "qa" / "inference" / "cfd_parity_surface_run1_full_matrix.json"
sys.path.insert(0, str(SCRIPTS_DIR))

import cfd_parity_contract as contract  # noqa: E402
import run_lepton_service_binary_parity as orchestrator  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


_ALL_MODELS = [
    "domino_surface",
    "geotransolver_surface",
    "transolver_surface",
    "xmgn_surface",
    "fignet_surface",
]
_ALL_METRICS = [
    "l2_pressure",
    "l2_shear_stress",
    "l2_pressure_area_weighted",
    "drag",
    "lift",
]


def _make_report(models: list[str] | None = None) -> list[dict]:
    if models is None:
        models = _ALL_MODELS
    return [
        {
            "model": model,
            "dataset": "drivaerml",
            "cases": ["run_1"],
            "metrics": {m: 0.16 for m in _ALL_METRICS},
            "per_case": [
                {
                    "case_id": "run_1",
                    "metrics": {m: 0.16 for m in _ALL_METRICS},
                    "metric_dtype": "cell",
                }
            ],
        }
        for model in models
    ]


def _evidence(tmp_path: Path, profile: dict | None = None) -> Path:
    if profile is None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    evidence = tmp_path / "evidence"
    # Request must match the full-matrix profile (all 5 models, all 5 metrics).
    request = {
        "models": _ALL_MODELS,
        "cases": [
            {
                "case_id": "run_1",
                "sha256": "a" * 64,
                "size_bytes": 10,
                "geometry_sha256": "b" * 64,
                "geometry_size_bytes": 20,
            }
        ],
        "metrics": _ALL_METRICS,
        "seed": 42,
        "save_inference_mesh": False,
        "visual_case_ids": [],
    }
    _write_json(evidence / "request.json", request)
    _write_json(evidence / "benchmark_results.json", _make_report())
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
                "model_names": _ALL_MODELS,
                "case_ids": ["run_1"],
                "selected_metrics": _ALL_METRICS,
                "preset_sha256": "c" * 64,
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
        image_tag="registry/service-image:latest",
        image_name="ignored",
        binary_image="nvcr.io/nvidia/physicsnemo-serve-cmd:pr-14-1c397c1",
        matrix="three",
        profile=str(PROFILE_PATH),
        rest_request_path=str(REPO_ROOT / "plugins" / "physicsnemo-cfd-surface-benchmark" / "examples" / "public_run_1_3model_request.json"),
        rest_evidence_dir=str(evidence),
        run_id="binary-test",
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


# ---------------------------------------------------------------------------
# Dry-run: structure validation
# ---------------------------------------------------------------------------


def test_dry_run_persists_handoff_and_summary(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)

    assert orchestrator.run(args) == 0

    run_dir = tmp_path / "artifacts" / "cfd-binary" / "binary-test"
    handoff = contract.read_json_object(run_dir / "parity-handoff.json")
    summary = contract.read_json_object(run_dir / "summary.json")

    assert handoff["rest"]["input_root_relpath"].endswith("inputs/drivaerml")
    assert handoff["rest"]["report_relpath"].endswith("benchmark_results.json")
    assert summary["final_result"] == "dry-run"
    assert summary["images"]["binary"] == args.binary_image
    assert summary["images"]["service"] == "registry/service-image:latest"
    assert summary["job"]["mount"] == (
        f"{tmp_path / 'shared' / 'user'}:/outputs:node-nfs:lustre"
    )
    assert len(summary["job"]["binary_command_sha256"]) == 64
    assert "physicsnemo-cfd-surface-full-v1" in json.dumps(summary)


def test_dry_run_without_rest_evidence_dir_succeeds_as_config_check(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.rest_evidence_dir = None  # no evidence — config-check mode
    assert orchestrator.run(args) == 0
    run_dir = tmp_path / "artifacts" / "cfd-binary" / "binary-test"
    summary = contract.read_json_object(run_dir / "summary.json")
    assert summary["final_result"] == "dry-run"
    assert summary["images"]["binary"] == args.binary_image
    assert summary["job"]["node_group"] == "node-group"
    assert "command omitted" in summary["job"]["note"]


# ---------------------------------------------------------------------------
# Binary command construction
# ---------------------------------------------------------------------------


def test_binary_command_contains_expected_paths_and_pipestatus() -> None:
    request_text = json.dumps({"models": ["domino_surface"]}) + "\n"
    command = orchestrator.build_binary_command(
        request_text=request_text,
        mount_target="/outputs",
        run_id="abc123",
    )

    assert "set -o pipefail" in command
    assert "test ! -e" not in command  # binary command doesn't guard against re-use
    assert "/outputs/cfd-binary/abc123/request.json" in command
    assert orchestrator.BINARY_PLUGIN_DIR in command
    assert orchestrator.BINARY_RUNTIME_DIR in command
    assert f"--run-id {orchestrator.BINARY_RUN_SUBID}" in command
    assert "--device 0" in command
    assert "PIPESTATUS[0]" in command
    assert "/outputs/cfd-binary/abc123/binary-job.log" in command


def test_binary_command_injects_request_json_via_base64() -> None:
    request = {"models": ["domino_surface"], "seed": 42}
    request_text = json.dumps(request) + "\n"

    # Verify base64_write_command round-trips correctly, and that the combined
    # binary command contains the write sub-command for the request file.
    write_cmd = orchestrator.base64_write_command(
        "/outputs/cfd-binary/testr/request.json", request_text
    )
    # Extract base64 payload from "printf %s <payload>" in the write command.
    import re
    match = re.search(r"printf %s (\S+)", write_cmd)
    assert match, "base64 write command has unexpected format"
    raw = match.group(1).strip("'\"")
    decoded = base64.b64decode(raw).decode("utf-8")
    assert json.loads(decoded) == request

    # Confirm the combined binary command embeds the same write sub-command.
    full_command = orchestrator.build_binary_command(
        request_text=request_text,
        mount_target="/outputs",
        run_id="testr",
    )
    assert "/outputs/cfd-binary/testr/request.json" in full_command


def test_binary_job_args_include_all_prefetch_env_vars(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.dry_run = False  # doesn't matter, we're testing build_binary_job_args directly

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    job_args = orchestrator.build_binary_job_args(
        args,
        binary_image=args.binary_image,
        nfs_path="/shared/user",
        run_id="testr",
        profile=profile,
        binary_command="echo hi",
    )
    args_str = " ".join(job_args)

    for key, value in orchestrator._PREFETCH_ENV.items():
        assert f"{key}={value}" in args_str, f"missing prefetch env var {key}"
    assert "HF_HOME=" in args_str
    assert "PHYSICSNEMO_CFD_MODEL_CACHE=" in args_str
    assert "huggingface" in args_str
    assert "models" in args_str


# ---------------------------------------------------------------------------
# Run ID and job naming
# ---------------------------------------------------------------------------


def test_run_ids_are_path_safe_and_long_job_names_keep_unique_suffixes() -> None:
    with pytest.raises(ValueError, match="run ID"):
        orchestrator.validate_run_id("../escape")

    first = orchestrator._lepton_job_name(
        "pn-cfd-binary-surface",
        "long-shared-prefix-that-would-otherwise-collide-one",
    )
    second = orchestrator._lepton_job_name(
        "pn-cfd-binary-surface",
        "long-shared-prefix-that-would-otherwise-collide-two",
    )
    assert len(first) <= 36
    assert len(second) <= 36
    assert first != second


def test_job_name_uses_binary_prefix() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    name = orchestrator.job_name(profile, "abc123")
    assert name.startswith("pn-cfd-binary-")
    assert len(name) <= 36


# ---------------------------------------------------------------------------
# REST evidence discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Marked-report extraction
# ---------------------------------------------------------------------------


def test_extract_marked_report_handles_plain_json_and_base64() -> None:
    payload = _make_report()
    output = (
        "noise\n"
        f"{orchestrator.BINARY_REPORT_BEGIN}\n"
        f"{json.dumps(payload)}\n"
        f"{orchestrator.BINARY_REPORT_END}\n"
    )
    assert orchestrator.extract_marked_report(output) == payload

    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    wrapped = "\n".join(encoded[i : i + 20] for i in range(0, len(encoded), 20))
    assert (
        orchestrator.extract_marked_report(
            f"{orchestrator.BINARY_REPORT_BEGIN}\n{wrapped}\n"
            f"{orchestrator.BINARY_REPORT_END}\n"
        )
        == payload
    )
    assert orchestrator.extract_marked_report("noise only") is None


def test_extract_marked_report_rejects_non_list_json() -> None:
    output = (
        f"{orchestrator.BINARY_REPORT_BEGIN}\n"
        '{"status": "this is a dict, not a list"}\n'
        f"{orchestrator.BINARY_REPORT_END}\n"
    )
    assert orchestrator.extract_marked_report(output) is None


# ---------------------------------------------------------------------------
# Reader job
# ---------------------------------------------------------------------------


def test_reader_job_fetches_binary_report_and_cleans_up(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, _evidence(tmp_path))
    calls: list[list[str]] = []
    payload = _make_report()

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
            f"{orchestrator.BINARY_REPORT_BEGIN}\n"
            f"{json.dumps(payload)}\n"
            f"{orchestrator.BINARY_REPORT_END}\n"
        ),
    )

    result = orchestrator.fetch_binary_report_via_reader_job(
        args,
        service_image="registry/service:latest",
        nfs_path="/shared/user",
        run_id="run-1",
        env={},
        artifact_dir=tmp_path / "artifacts",
    )

    assert result == payload
    assert any(command[:3] == ["lep", "job", "stop"] for command in calls)
    assert any(command[:3] == ["lep", "job", "remove"] for command in calls)


def test_reader_job_uses_service_image_not_binary_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, _evidence(tmp_path))
    captured_image: list[str] = []

    def fake_streaming(command, **_kwargs):
        if command[:3] == ["lep", "job", "create"]:
            for i, token in enumerate(command):
                if token == "--container-image" and i + 1 < len(command):
                    captured_image.append(command[i + 1])
            return 0, "ID: reader-id\n"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)
    monkeypatch.setattr(orchestrator, "wait_for_loggable_job", lambda **_kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "capture_job_logs",
        lambda **_kwargs: "",
    )

    orchestrator.fetch_binary_report_via_reader_job(
        args,
        service_image="registry/service:v1",
        nfs_path="/shared/user",
        run_id="run-1",
        env={},
        artifact_dir=tmp_path / "artifacts",
    )

    assert captured_image == ["registry/service:v1"], (
        "reader job must use the service image (which has Python), not the binary image"
    )


# ---------------------------------------------------------------------------
# Full run with monkeypatched Lepton calls
# ---------------------------------------------------------------------------


def test_full_run_passes_when_reports_match(tmp_path: Path, monkeypatch) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.dry_run = False
    args.workspace_token = "tok"

    report = _make_report()

    def fake_streaming(command, **_kwargs):
        if command[:3] == ["lep", "job", "create"]:
            return 0, "ID: binary-job-id\n"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)
    monkeypatch.setattr(orchestrator, "login_if_needed", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator,
        "poll_job",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        orchestrator,
        "capture_job_logs",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_binary_report_if_available",
        lambda *_a, **_k: report,
    )

    rc = orchestrator.run(args)
    assert rc == 0

    run_dir = tmp_path / "artifacts" / "cfd-binary" / "binary-test"
    summary = contract.read_json_object(run_dir / "summary.json")
    assert summary["final_result"] == "passed"
    assert summary["comparison"]["status"] == "passed"


def test_full_run_fails_when_binary_job_exits_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.dry_run = False
    args.workspace_token = "tok"

    def fake_streaming(command, **_kwargs):
        if command[:3] == ["lep", "job", "create"]:
            return 0, "ID: binary-job-id\n"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)
    monkeypatch.setattr(orchestrator, "login_if_needed", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "poll_job", lambda **_kwargs: 1)
    monkeypatch.setattr(orchestrator, "capture_job_logs", lambda **_kwargs: "")
    monkeypatch.setattr(
        orchestrator,
        "_load_binary_report_if_available",
        lambda *_a, **_k: _make_report(),
    )

    rc = orchestrator.run(args)
    assert rc == 1

    run_dir = tmp_path / "artifacts" / "cfd-binary" / "binary-test"
    summary = contract.read_json_object(run_dir / "summary.json")
    assert summary["final_result"] == "failed"
    assert "exit_code" in summary["job"]


def test_full_run_fails_when_no_binary_report_available(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = _evidence(tmp_path)
    args = _args(tmp_path, evidence)
    args.dry_run = False
    args.workspace_token = "tok"

    def fake_streaming(command, **_kwargs):
        if command[:3] == ["lep", "job", "create"]:
            return 0, "ID: binary-job-id\n"
        return 0, ""

    monkeypatch.setattr(orchestrator, "run_streaming", fake_streaming)
    monkeypatch.setattr(orchestrator, "login_if_needed", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "poll_job", lambda **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "capture_job_logs", lambda **_kwargs: "")
    monkeypatch.setattr(
        orchestrator,
        "_load_binary_report_if_available",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_binary_report_via_reader_job",
        lambda *_a, **_k: None,
    )

    rc = orchestrator.run(args)
    assert rc == 1

    run_dir = tmp_path / "artifacts" / "cfd-binary" / "binary-test"
    summary = contract.read_json_object(run_dir / "summary.json")
    assert summary["final_result"] == "failed"
    assert "benchmark_results.json" in summary["error"]["message"]
