# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "qa" / "scripts"
PROFILE_PATH = REPO_ROOT / "qa" / "inference" / "cfd_parity_surface_run1.json"
sys.path.insert(0, str(SCRIPTS_DIR))

import cfd_parity_contract as contract  # noqa: E402
import run_cfd_parity_job as job_runner  # noqa: E402


def _report() -> list[dict[str, object]]:
    return [
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_job_runner_executes_independent_config_and_compares_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    mount = tmp_path / "outputs"
    work = mount / "cfd-parity" / "run-1"
    input_root = mount / "rest" / "inputs"
    rest_report = mount / "rest" / "benchmark_results.json"
    work.mkdir(parents=True)
    input_root.mkdir(parents=True)
    _write_json(rest_report, _report())
    report_digest = hashlib.sha256(rest_report.read_bytes()).hexdigest()
    handoff = {
        "schema_version": 2,
        "parity_run_id": "run-1",
        "profile_id": profile["profile_id"],
        "workflow_id": profile["workflow_id"],
        "domain": profile["domain"],
        "image": "registry/image@sha256:" + ("d" * 64),
        "mount_target": str(mount),
        "rest_run_id": "rest-run",
        "request": {
            "models": ["domino_surface"],
            "cases": [
                {
                    "case_id": "run_1",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "geometry_sha256": "b" * 64,
                    "geometry_size_bytes": 1,
                }
            ],
            "metrics": ["l2_pressure"],
            "seed": 42,
            "save_inference_mesh": False,
            "visual_case_ids": [],
        },
        "provenance": {
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
            "preset_sha256": profile["preset_sha256"],
            "case_digests": [
                {
                    "case_id": "run_1",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "geometry_sha256": "b" * 64,
                    "geometry_size_bytes": 1,
                }
            ],
        },
        "rest": {
            "input_root_relpath": "rest/inputs",
            "report_relpath": "rest/benchmark_results.json",
            "report_sha256": report_digest,
            "report_size_bytes": rest_report.stat().st_size,
            "resolved_config_relpath": "rest/resolved_config.json",
        },
    }
    profile_path = work / "profile.json"
    handoff_path = work / "handoff.json"
    _write_json(profile_path, profile)
    _write_json(handoff_path, handoff)

    monkeypatch.setattr(
        job_runner,
        "_verify_provider",
        lambda _provider: {
            "distribution": "nvidia-physicsnemo-cfd",
            "module": "physicsnemo.cfd",
            "version": "0.0.2",
            "commit": profile["provider"]["commit"],
            "python_version": "3.12",
        },
    )
    monkeypatch.setattr(
        job_runner, "verify_staged_inputs", lambda *_args, **_kwargs: []
    )

    def fake_run(command, **_kwargs):
        config = contract.read_json_object(command[-1])
        output_dir = Path(config["run"]["output_dir"])
        _write_json(output_dir / "benchmark_results.json", _report())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(job_runner.subprocess, "run", fake_run)
    args = argparse.Namespace(
        profile=str(profile_path),
        handoff=str(handoff_path),
        mount_target=str(mount),
        work_dir=str(work),
    )

    assert job_runner.run(args) == 0
    summary = contract.read_json_object(work / "summary.json")
    comparison = contract.read_json_object(work / "comparison.json")
    direct_config = contract.read_json_object(work / "direct" / "direct_config.json")
    assert summary["final_result"] == "passed"
    assert comparison["status"] == "passed"
    assert direct_config["benchmark"]["models"][0]["name"] == "domino_surface"
    assert "package" not in direct_config["benchmark"]["models"][0]
