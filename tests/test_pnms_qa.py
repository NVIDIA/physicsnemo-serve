from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import struct

import yaml

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.native_inference import pnms_local_job as job  # noqa: E402

IDENTITY = {
    "run_id": "unit-run",
    "source_sha": "b" * 40,
    "image_digest": "registry.example/pnms-qa@sha256:" + "a" * 64,
    "profile": "local",
}


def passed_report():
    records = []
    for name in job.expected_cases():
        details = {}
        if name.endswith(".build") and name.startswith("native."):
            details = {
                "build_id": "c" * 32,
                "bundle_digest": "d" * 64,
                "model_identity": "e" * 64,
                "native_checks": {"verified": True, "case_count": 3},
            }
        if name.endswith(".held_out"):
            details = {
                "case_count": 3,
                "parity": "passed",
                "input_kind": "raw-fp32-le" if ".files." in name else "json-array-fp32",
            }
        if (
            name.endswith(".cache_reuse") and name.startswith("native.")
        ) or name == "environment_lock.refresh":
            details = {
                "model_reused": True,
                "workflow_reused": True,
                "environment_lock_digest": "1" * 64,
                "build_id": "f" * 32,
                "bundle_digest": "d" * 64,
            }
            if name == "environment_lock.refresh":
                details.update(
                    drift_rejected=True,
                    prior_success_preserved=True,
                    refresh_passed=True,
                )
        records.append({"name": name, "status": "passed", "details": details})
    candidates = [
        {
            "backend": backend,
            "template": template,
            "ingress": ingress,
            "build_id": "f" * 32,
            "bundle_digest": "d" * 64,
        }
        for backend, template, ingress in job.NATIVE_TARGETS
    ]
    return {
        "schema_version": 1,
        **IDENTITY,
        "status": "passed",
        "qa_scope": "local-workflow",
        "expected_cases": job.expected_cases(),
        "cases": records,
        "http_qa": "not_run",
        "release_candidates": candidates,
    }


def test_required_inventory_covers_both_backends_templates_and_python_paths():
    names = job.expected_cases()
    assert "environment" in names
    assert len(names) == len(set(names))
    for backend in ("aoti", "tensorrt"):
        for template in ("tensor", "custom"):
            for operation in (
                "init",
                "check",
                "build",
                "held_out",
                "cache_reuse",
                "relocation",
            ):
                assert f"native.{backend}.{template}.{operation}" in names
    assert "python.new.run" in names and "python.legacy.run" in names
    assert "environment_lock.refresh" in names


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "skipped",
        "duplicate",
        "wrong-image",
        "unverified-native",
        "missing-parity",
        "workflow-miss",
        "missing-environment-lock",
    ],
)
def test_summary_rejects_incomplete_or_foreign_acceptance(mutation):
    report = copy.deepcopy(passed_report())
    if mutation == "missing":
        report["cases"].pop()
    elif mutation == "skipped":
        report["cases"][-1]["status"] = "skipped"
    elif mutation == "duplicate":
        report["cases"].append(report["cases"][0])
    elif mutation == "wrong-image":
        report["image_digest"] = "other"
    elif mutation == "unverified-native":
        next(
            case
            for case in report["cases"]
            if case["name"] == "native.aoti.tensor.build"
        )["details"]["native_checks"]["verified"] = False
    elif mutation == "missing-parity":
        next(
            case
            for case in report["cases"]
            if case["name"] == "native.aoti.tensor.held_out"
        )["details"]["parity"] = "not_run"
    elif mutation == "workflow-miss":
        next(
            case
            for case in report["cases"]
            if case["name"] == "native.aoti.tensor.cache_reuse"
        )["details"]["workflow_reused"] = False
    else:
        next(
            case
            for case in report["cases"]
            if case["name"] == "native.aoti.tensor.cache_reuse"
        )["details"].pop("environment_lock_digest")
    with pytest.raises(ValueError):
        job.validate_summary(report, **IDENTITY)


def test_complete_local_report_does_not_claim_http_acceptance():
    report = passed_report()
    job.validate_summary(report, **IDENTITY)
    assert report["http_qa"] == "not_run"


def test_candidate_must_follow_latest_verified_cache_reuse_build():
    report = passed_report()
    report["release_candidates"][0]["build_id"] = "c" * 32
    with pytest.raises(ValueError, match="candidate|Candidate"):
        job.validate_summary(report, **IDENTITY)


def test_lock_refresh_runs_before_relocation_changes_model_source_identity(tmp_path):
    from types import SimpleNamespace

    names = []
    context = SimpleNamespace(
        root=tmp_path,
        args=SimpleNamespace(python=Path(sys.executable), sdk_prefix=tmp_path / "sdk"),
        case=lambda name, function: names.append(name),
    )
    job.native_cases(context, "aoti", "tensor")
    assert "environment_lock.refresh" in names
    assert names.index("environment_lock.refresh") < names.index(
        "native.aoti.tensor.relocation"
    )


def test_summary_extracts_only_complete_marked_json():
    text = (
        f"logs\n{job.SUMMARY_BEGIN}\n{json.dumps(passed_report())}\n{job.SUMMARY_END}\n"
    )
    assert job.extract_summary(text)["run_id"] == "unit-run"
    with pytest.raises(ValueError):
        job.extract_summary(text.split(job.SUMMARY_END)[0])


def test_qa_path_keeps_compilers_but_removes_docker_and_lepton(tmp_path):
    programs = tmp_path / "bin"
    programs.mkdir()
    for name in ("cmake", "c++", "python3", "docker", "lep"):
        program = programs / name
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755)
    selected = Path(job.docker_free_path(str(programs), tmp_path / "host-tools"))
    assert (selected / "cmake").is_file()
    assert (selected / "python3").is_file()
    assert not (selected / "docker").exists()
    assert not (selected / "lep").exists()


@pytest.mark.parametrize("profile", ["local", "fresh-venv"])
def test_qa_commands_bind_the_selected_repository(tmp_path, monkeypatch, profile):
    from types import SimpleNamespace

    from qa.native_inference import pnms_venv_job

    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import cli, common

    alternate = tmp_path / "alternate-checkout"
    (alternate / "scripts").mkdir(parents=True)
    (alternate / "scripts/plugin_dev.py").write_text("# alternate source helper\n")
    (alternate / "native-inference").mkdir()
    monkeypatch.setenv("PNMS_REPOSITORY", str(alternate))
    monkeypatch.setattr(job, "docker_free_path", lambda original, _output: original)
    args = job.parser().parse_args(
        [
            "--repository",
            str(ROOT),
            "--output",
            str(tmp_path / "qa"),
            "--run-id",
            IDENTITY["run_id"],
            "--expected-source-sha",
            IDENTITY["source_sha"],
            "--image-digest",
            IDENTITY["image_digest"],
        ]
    )
    context_type = job.Context if profile == "local" else pnms_venv_job.Context
    context = context_type(args)
    captured = {}

    def launch(argv, **options):
        captured.update(argv=argv, environment=options["env"])
        options["stdout"].write("{}")
        return SimpleNamespace(wait=lambda **_options: 0)

    monkeypatch.setattr(job.subprocess, "Popen", launch)
    context.pnms("init", "init", tmp_path / "app", "--runtime", "cpp")
    child_args = cli.parser().parse_args(captured["argv"][2:])
    monkeypatch.setenv("PNMS_REPOSITORY", captured["environment"]["PNMS_REPOSITORY"])
    assert common.repository_root() == alternate
    assert common.repository_root(child_args.repository) == ROOT


def test_release_handoff_requires_all_five_tested_bundles():
    report = passed_report()
    report["release_candidates"].pop()
    with pytest.raises(ValueError, match="candidate|bundle"):
        job.validate_summary(report, **IDENTITY)


def controller_args(tmp_path, monkeypatch):
    from scripts import run_lepton_pnms_qa as controller

    monkeypatch.delenv("LEPTON_WORKSPACE_TOKEN", raising=False)
    args = controller.build_parser().parse_args(
        [
            "--image",
            IDENTITY["image_digest"],
            "--expected-source-sha",
            IDENTITY["source_sha"],
            "--run-id",
            IDENTITY["run_id"],
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "unit-workspace",
            "--node-group",
            "unit-group",
            "--pull-secret",
            "unit-secret",
            "--resource-shape",
            "gpu.h100-sxm",
            "--nfs-mount-base",
            "/mnt/shared",
        ]
    )
    return controller, args


def test_controller_submits_one_finite_job_and_requires_strict_summary(
    tmp_path, monkeypatch
):
    controller, args = controller_args(tmp_path, monkeypatch)
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: pnms-unit-job\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Completed"}'
        if command[:2] == ["log", "get"]:
            lines = [job.SUMMARY_BEGIN, json.dumps(passed_report()), job.SUMMARY_END]
            return 0, "\n".join(
                "2026-10-28 16:00:00.000001|" + json.dumps(line) for line in lines
            )
        return 0, ""

    monkeypatch.setattr(controller.lifecycle, "lep", fake)
    assert controller.run(args) == 0
    create = calls[0]
    assert create[create.index("--num-workers") + 1] == "1"
    assert create[create.index("--max-failure-retry") + 1] == "0"
    remote = create[create.index("--command") + 1]
    assert "pnms_local_job.py" in remote
    assert "docker" not in remote and "pip install" not in remote
    assert calls[-1] == ["job", "remove", "-i", "pnms-unit-job"]


def test_controller_dry_run_records_command_without_cloud_access(tmp_path, monkeypatch):
    controller, args = controller_args(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(
        controller.lifecycle,
        "lep",
        lambda *a, **k: pytest.fail("dry-run contacted Lepton"),
    )
    assert controller.run(args) == 0
    summary = json.loads((tmp_path / args.run_id / "summary.json").read_text())
    assert summary["status"] == "dry-run"


def test_acceptance_requires_a_separate_aoti_file_input_target():
    for operation in (
        "init",
        "check",
        "build",
        "held_out",
        "cache_reuse",
        "relocation",
        "reject_invalid",
    ):
        assert f"native.aoti.files.{operation}" in job.expected_cases()


def test_file_acceptance_project_has_separate_binary_fixture_inputs(tmp_path):
    (tmp_path / "pnms.yaml").write_text(
        yaml.safe_dump({"workflow": {"template": "tensor"}})
    )
    fixture = tmp_path / "fixtures/cases.json"
    fixture.parent.mkdir()
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "request": {"values": [1, -2, 0.25, 4]},
                        "expected": {"values": [3, -3, 1.5, 9]},
                    }
                ]
            }
        )
    )
    job.configure_file_project(tmp_path)
    project = yaml.safe_load((tmp_path / "pnms.yaml").read_text())
    case = json.loads(fixture.read_text())["cases"][0]
    assert project["ingress"]["content_type"] == "multipart/form-data"
    assert (
        project["workflow"]["tensor_adapter"]["inputs"]["input"]["codec"]
        == "raw-fp32-le"
    )
    assert case["request"] == job.raw_parameters()
    assert (tmp_path / case["inputs"]["features"]).read_bytes() == struct.pack(
        "<4f", 1, -2, 0.25, 4
    )
    assert case["expected"] == {"values": [3, -3, 1.5, 9]}


def test_summary_requires_actual_file_inference_evidence():
    report = passed_report()
    case = next(
        case for case in report["cases"] if case["name"] == "native.aoti.files.held_out"
    )
    del case["details"]["input_kind"]
    with pytest.raises(ValueError, match="Raw-file"):
        job.validate_summary(report, **IDENTITY)
