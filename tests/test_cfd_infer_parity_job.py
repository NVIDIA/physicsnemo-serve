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
import run_cfd_infer_parity_job as infer_job  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_infer_job_compares_cli_report_with_original_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    mount = tmp_path / "outputs"
    work = mount / "cfd-parity" / "parity-test"
    work.mkdir(parents=True)

    mesh = b"mesh"
    geometry = b"geometry"
    request = {
        "models": ["domino_surface"],
        "cases": [
            {
                "case_id": "run_1",
                "mesh_uri": "https://example.test/boundary_1.vtp",
                "sha256": hashlib.sha256(mesh).hexdigest(),
                "size_bytes": len(mesh),
                "geometry_uri": "https://example.test/drivaer_1.stl",
                "geometry_sha256": hashlib.sha256(geometry).hexdigest(),
                "geometry_size_bytes": len(geometry),
            }
        ],
        "metrics": ["l2_pressure"],
        "seed": 42,
        "save_inference_mesh": False,
        "visual_case_ids": [],
    }
    profile_path = work / "profile.json"
    request_path = work / "request.json"
    _write_json(profile_path, profile)
    _write_json(request_path, request)

    runtime_dir = tmp_path / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("", encoding="utf-8")
    (runtime_dir / "scripts").mkdir()
    (runtime_dir / "python").mkdir()
    infer_binary = tmp_path / "physicsnemo-serve"
    infer_binary.write_text("", encoding="utf-8")
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    (plugin_root / "plugin.yaml").write_text("metadata: {}", encoding="utf-8")

    monkeypatch.setattr(
        infer_job,
        "_verify_provider",
        lambda _provider: {
            "distribution": "nvidia-physicsnemo-cfd",
            "module": "physicsnemo.cfd",
            "version": "0.0.2",
            "commit": profile["provider"]["commit"],
            "python_version": "3.12",
        },
    )

    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == str(infer_binary):
            output_root = Path(command[command.index("--output-dir") + 1])
            run_id = command[command.index("--run-id") + 1]
            attempt = output_root / run_id / "attempt"
            input_root = attempt / "inputs" / "drivaerml"
            case_root = input_root / "run_1"
            case_root.mkdir(parents=True)
            (case_root / "boundary_1.vtp").write_bytes(mesh)
            (case_root / "drivaer_1.stl").write_bytes(geometry)
            report_path = attempt / "benchmark-output" / "benchmark_results.json"
            config_path = attempt / "resolved_config.json"
            _write_json(report_path, _report())
            _write_json(
                config_path,
                {
                    "benchmark": {
                        "datasets": [
                            {
                                "name": "drivaerml",
                                "root": str(input_root),
                            }
                        ]
                    }
                },
            )
            provider = {
                key: profile["provider"][key]
                for key in (
                    "repository",
                    "tag",
                    "version",
                    "commit",
                    "physicsnemo_version",
                    "python_version",
                )
            }
            result = {
                "run_id": run_id,
                "status": "succeeded",
                "workflow": profile["workflow_id"],
                "request": {
                    "content_type": "application/json",
                    "operation": "run",
                    "raw_fields": request,
                },
                "execution": {
                    "outputs": [
                        {
                            "name": "benchmark_results.json",
                            "storage_path": str(report_path),
                            "primary": True,
                        },
                        {
                            "name": "resolved_config.json",
                            "storage_path": str(config_path),
                        },
                    ],
                    "output_path": str(report_path),
                },
                "payload": {
                    "report_path": str(report_path),
                    "resolved_config_path": str(config_path),
                    "provider": provider,
                    "preset_sha256": profile["preset_sha256"],
                    "model_names": request["models"],
                    "case_ids": ["run_1"],
                    "selected_metrics": request["metrics"],
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
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(result))

        direct_config = contract.read_json_object(command[-1])
        direct_output = Path(direct_config["run"]["output_dir"])
        _write_json(direct_output / "benchmark_results.json", _report())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(infer_job, "_run_process_group", fake_run)
    args = argparse.Namespace(
        profile=str(profile_path),
        request=str(request_path),
        mount_target=str(mount),
        work_dir=str(work),
        parity_run_id="parity-test",
        infer_run_id="infer-test",
        image="registry/physicsnemo-serve-cmd@sha256:" + ("d" * 64),
        infer_binary=str(infer_binary),
        runtime_dir=str(runtime_dir),
        plugin=str(plugin_root),
        device="0",
        infer_timeout_seconds=60,
        download_timeout_seconds=60,
    )

    assert infer_job.run(args) == 0
    summary = contract.read_json_object(work / "summary.json")
    comparison = contract.read_json_object(work / "comparison.json")
    direct_config = contract.read_json_object(
        work / "physicsnemo-cfd" / "direct_config.json"
    )
    assert summary["final_result"] == "passed"
    assert summary["infer_run_id"] == "infer-test"
    assert len(summary["verified_inputs"]) == 2
    assert comparison["status"] == "passed"
    assert (
        comparison["metrics"][0]["infer"] == comparison["metrics"][0]["physicsnemo_cfd"]
    )
    assert direct_config["benchmark"]["models"][0]["name"] == "domino_surface"
    assert "package" not in direct_config["benchmark"]["models"][0]
    assert len(calls) == 2
