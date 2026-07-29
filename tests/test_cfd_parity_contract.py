# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "qa" / "scripts"
PROFILE_PATH = REPO_ROOT / "qa" / "inference" / "cfd_parity_surface_run1.json"
FULL_PROFILE_PATH = (
    REPO_ROOT / "qa" / "inference" / "cfd_parity_surface_run1_full_matrix.json"
)
FULL_REQUEST_PATH = (
    REPO_ROOT
    / "plugins"
    / "physicsnemo-cfd-surface-benchmark"
    / "examples"
    / "public_run_1_full_matrix_request.json"
)
TWO_CASE_REQUEST_PATH = (
    REPO_ROOT
    / "plugins"
    / "physicsnemo-cfd-surface-benchmark"
    / "examples"
    / "public_run_1_11_full_matrix_request.json"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import cfd_parity_contract as contract  # noqa: E402


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _report(value: float = 0.16348028933450415) -> list[dict[str, object]]:
    return [
        {
            "model": "domino_surface",
            "dataset": "drivaerml",
            "cases": ["run_1"],
            "metrics": {"l2_pressure": value},
            "per_case": [
                {
                    "case_id": "run_1",
                    "metrics": {"l2_pressure": value},
                    "metric_dtype": "cell",
                }
            ],
        }
    ]


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_evidence(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    profile = _profile()
    mount = tmp_path / "outputs"
    input_root = mount / "rest-run" / "attempt" / "inputs" / "drivaerml"
    output_root = mount / "rest-run" / "attempt" / "benchmark-output"
    mesh = b"mesh"
    geometry = b"geometry"
    mesh_path = input_root / "run_1" / "boundary_1.vtp"
    geometry_path = input_root / "run_1" / "drivaer_1.stl"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(mesh)
    geometry_path.write_bytes(geometry)
    output_root.mkdir(parents=True)

    request = {
        "models": ["domino_surface"],
        "cases": [
            {
                "case_id": "run_1",
                "mesh_uri": "https://example.test/boundary_1.vtp",
                "sha256": _digest(mesh),
                "size_bytes": len(mesh),
                "geometry_uri": "https://example.test/drivaer_1.stl",
                "geometry_sha256": _digest(geometry),
                "geometry_size_bytes": len(geometry),
            }
        ],
        "metrics": ["l2_pressure"],
        "seed": 42,
        "save_inference_mesh": False,
        "visual_case_ids": [],
    }
    report = _report()
    remote_report = output_root / "benchmark_results.json"
    _write_json(remote_report, report)
    remote_config = mount / "rest-run" / "attempt" / "resolved_config.json"
    resolved_config = {
        "run": {"seed": -999},
        "benchmark": {
            "mode": "matrix",
            "models": [{"name": "poisoned-plugin-config"}],
            "datasets": [{"name": "drivaerml", "root": str(input_root)}],
        },
        "metrics": ["poisoned_metric"],
    }
    _write_json(remote_config, resolved_config)

    evidence = tmp_path / "evidence"
    _write_json(evidence / "request.json", request)
    _write_json(evidence / "benchmark_results.json", report)
    _write_json(evidence / "resolved_config.json", resolved_config)
    _write_json(
        evidence / "results.json",
        {
            "request": {},
            "execution": {
                "run_id": "rest-run",
                "workflow": profile["workflow_id"],
                "status": "succeeded",
                "outputs": [
                    {
                        "name": "benchmark_results.json",
                        "storage_path": str(remote_report),
                        "primary": True,
                    },
                    {
                        "name": "resolved_config.json",
                        "storage_path": str(remote_config),
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
                "preset_sha256": "a" * 64,
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
    return evidence, mount, profile


def test_handoff_persists_mount_relative_paths_and_verified_inputs(
    tmp_path: Path,
) -> None:
    evidence, mount, profile = _build_evidence(tmp_path)

    handoff = contract.build_handoff(
        evidence_dir=evidence,
        profile=profile,
        parity_run_id="parity-1",
        image="registry/image@sha256:abc",
        mount_target=str(mount),
    )

    assert handoff["rest"]["report_relpath"].endswith("benchmark_results.json")
    assert not handoff["rest"]["report_relpath"].startswith("/")
    input_root = contract.resolve_existing_mount_path(
        str(mount), handoff["rest"]["input_root_relpath"]
    )
    verified = contract.verify_staged_inputs(
        profile,
        handoff,
        input_root=input_root,
    )
    assert [item["size_bytes"] for item in verified] == [4, 8]


def test_direct_config_is_profile_driven_not_rest_config(tmp_path: Path) -> None:
    evidence, mount, profile = _build_evidence(tmp_path)
    handoff = contract.build_handoff(
        evidence_dir=evidence,
        profile=profile,
        parity_run_id="parity-1",
        image="registry/image:tag",
        mount_target=str(mount),
    )

    direct = contract.build_direct_config(
        profile,
        handoff,
        input_root=mount / handoff["rest"]["input_root_relpath"],
        output_dir=mount / "direct",
    )

    assert direct["run"]["seed"] == 42
    assert direct["metrics"] == ["l2_pressure"]
    assert direct["benchmark"]["models"] == [
        {
            "name": "domino_surface",
            "inference_domain": "surface",
            "kwargs": {},
        }
    ]
    assert direct["benchmark"]["datasets"][0]["case_ids"] == ["run_1"]
    assert direct["run"]["metrics_cache"]["enabled"] is False


def test_report_comparison_is_symmetric_and_structure_aware() -> None:
    comparison = _profile()["comparison"]

    exact = contract.compare_reports(
        rest_report=_report(),
        direct_report=_report(),
        comparison=comparison,
    )
    assert exact["status"] == "passed"
    assert all(metric["matches"] for metric in exact["metrics"])

    mismatch = contract.compare_reports(
        rest_report=_report(),
        direct_report=_report(0.17),
        comparison=comparison,
    )
    assert mismatch["status"] == "failed"
    assert mismatch["errors"]

    changed_structure = _report()
    changed_structure[0]["per_case"] = []
    with pytest.raises(contract.ParityContractError, match="per-case IDs differ"):
        contract.compare_reports(
            rest_report=_report(),
            direct_report=changed_structure,
            comparison=comparison,
        )


def test_report_comparison_supports_bounded_per_model_nondeterminism() -> None:
    comparison = copy.deepcopy(_profile()["comparison"])
    comparison["models"] = {
        "domino_surface": {
            "default_rtol": 0.001,
            "default_atol": 1e-6,
        }
    }

    result = contract.compare_reports(
        rest_report=_report(0.16),
        direct_report=_report(0.1601),
        comparison=comparison,
    )

    assert result["status"] == "passed"
    assert all(metric["rtol"] == 0.001 for metric in result["metrics"])


def test_zero_baseline_relative_difference_is_json_null(tmp_path: Path) -> None:
    result = contract.compare_reports(
        rest_report=_report(0.0),
        direct_report=_report(5e-7),
        comparison=_profile()["comparison"],
    )

    assert result["status"] == "passed"
    assert all(metric["relative_difference"] is None for metric in result["metrics"])
    output = tmp_path / "comparison.json"
    contract.write_json_atomic(output, result)
    text = output.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert all(
        metric["relative_difference"] is None for metric in json.loads(text)["metrics"]
    )
    both_zero = contract.compare_reports(
        rest_report=_report(0.0),
        direct_report=_report(0.0),
        comparison=_profile()["comparison"],
    )
    assert all(metric["relative_difference"] == 0.0 for metric in both_zero["metrics"])


def test_report_comparison_rejects_non_finite_metrics() -> None:
    with pytest.raises(contract.ParityContractError, match="must be finite"):
        contract.compare_reports(
            rest_report=_report(),
            direct_report=_report(float("nan")),
            comparison=_profile()["comparison"],
        )


def test_mount_resolution_rejects_escape_and_symlink(tmp_path: Path) -> None:
    mount = tmp_path / "outputs"
    mount.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (mount / "escape").symlink_to(outside)

    with pytest.raises(contract.ParityContractError, match="outside mount"):
        contract.mount_relative_path(str(outside), str(mount))
    with pytest.raises(contract.ParityContractError, match="escapes mount"):
        contract.resolve_existing_mount_path(str(mount), "escape")


def test_same_contract_builds_future_volume_direct_config(tmp_path: Path) -> None:
    evidence, mount, surface_profile = _build_evidence(tmp_path)
    handoff = contract.build_handoff(
        evidence_dir=evidence,
        profile=surface_profile,
        parity_run_id="parity-1",
        image="registry/image:tag",
        mount_target=str(mount),
    )
    volume_profile = copy.deepcopy(surface_profile)
    volume_profile["profile_id"] = "physicsnemo-cfd-volume-domino-run1-v1"
    volume_profile["workflow_id"] = "physicsnemo-cfd-volume-benchmark"
    volume_profile["domain"] = "volume"
    volume_profile["request"]["models"] = ["domino_volume"]
    volume_profile["request"]["metrics"] = ["l2_pressure", "l2_velocity"]
    volume_profile["runner"]["module"] = (
        "physicsnemo_cfd_runtime.volume_benchmark_runner"
    )
    volume_profile["config"]["models"] = [
        {
            "name": "domino_volume",
            "inference_domain": "volume",
            "kwargs": {},
        }
    ]
    volume_profile["config"]["dataset"] = {
        "name": "drivaerml_rust",
        "kwargs": {
            "inference_domain": "volume",
            "gt_data_type": "point",
        },
    }
    handoff["profile_id"] = volume_profile["profile_id"]
    handoff["workflow_id"] = volume_profile["workflow_id"]
    handoff["domain"] = "volume"
    handoff["request"]["models"] = ["domino_volume"]
    handoff["request"]["metrics"] = ["l2_pressure", "l2_velocity"]

    contract.validate_handoff(volume_profile, handoff)
    direct = contract.build_direct_config(
        volume_profile,
        handoff,
        input_root=tmp_path / "volume-input",
        output_dir=tmp_path / "volume-output",
    )

    assert direct["benchmark"]["models"][0]["name"] == "domino_volume"
    assert direct["benchmark"]["datasets"][0]["name"] == "drivaerml_rust"
    assert direct["metrics"] == ["l2_pressure", "l2_velocity"]


def test_full_surface_profile_covers_25_model_metric_tuples() -> None:
    profile = json.loads(FULL_PROFILE_PATH.read_text(encoding="utf-8"))
    request = json.loads(FULL_REQUEST_PATH.read_text(encoding="utf-8"))

    contract.validate_profile(profile)
    contract.validate_request(profile, request)

    assert len(profile["request"]["models"]) == 5
    assert len(profile["request"]["metrics"]) == 5
    assert len(request["cases"]) == 1
    assert (
        len(profile["request"]["models"])
        * len(profile["request"]["metrics"])
        * len(request["cases"])
        == 25
    )
    assert [model["name"] for model in profile["config"]["models"]] == request["models"]
    assert profile["rest"]["request_path"].endswith(
        "public_run_1_full_matrix_request.json"
    )
    assert (
        profile["comparison"]["models"]["geotransolver_surface"]["default_rtol"]
        == 0.005
    )
    assert (
        profile["comparison"]["models"]["geotransolver_surface"]["metrics"][
            "drag_error"
        ]["rtol"]
        == 0.01
    )
    assert profile["comparison"]["models"]["xmgn_surface"]["default_rtol"] == 0.001


def test_full_surface_profile_accepts_pinned_run_1_and_run_11() -> None:
    profile = json.loads(FULL_PROFILE_PATH.read_text(encoding="utf-8"))
    request = json.loads(TWO_CASE_REQUEST_PATH.read_text(encoding="utf-8"))

    contract.validate_profile(profile)
    contract.validate_request(profile, request)

    assert [case["case_id"] for case in request["cases"]] == ["run_1", "run_11"]
    assert (
        len(request["models"]) * len(request["metrics"]) * len(request["cases"]) == 50
    )
    assert request["cases"][1]["sha256"] == (
        "aec39be9198f2229fb80b04a1a30d049cb0d195bdb7df566e2520222c8af679e"
    )
    assert request["cases"][1]["geometry_sha256"] == (
        "6ee0dc50946fda73b00a693dc2a74d1b0a8f3c37a6d5c4d8cd5cfb5bb25830a6"
    )
