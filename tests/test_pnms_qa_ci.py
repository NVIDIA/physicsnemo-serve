"""Check required Lepton profile dispatch without contacting the service."""

from __future__ import annotations

import json
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/lepton_pnms_qa.yml"


def workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def step_containing(controller, includes=None):
    steps = workflow()["jobs"]["qualify"]["steps"]
    matches = [
        step
        for step in steps
        if controller in step.get("run", "")
        and (includes is None or includes in step["run"])
    ]
    assert len(matches) == 1, f"required profile must invoke {controller} once"
    return matches[0]


def test_dispatch_requires_both_images_without_manual_staging_inputs():
    inputs = workflow()["on"]["workflow_dispatch"]["inputs"]
    for name in (
        "image",
        "source_sha",
        "builder_image",
    ):
        assert name in inputs, f"missing reproducible QA input: {name}"
        assert inputs[name]["required"] == "true"
    for manual_input in (
        "container_qa_image",
        "builder_runner",
        "builder_source_root",
        "builder_source_manifest",
        "venv_requirements",
    ):
        assert manual_input not in inputs


def test_shared_inputs_are_staged_on_lepton_before_both_required_profiles():
    steps = workflow()["jobs"]["qualify"]["steps"]
    stage = step_containing("scripts/run_lepton_pnms_stage_qa.py")
    assert "if" not in stage and "continue-on-error" not in stage
    for controller in (
        "scripts/run_lepton_model_builder_image_qa.py",
        "scripts/run_lepton_pnms_venv_qa.py",
    ):
        assert steps.index(stage) < steps.index(step_containing(controller))
    assert "--image" in stage["run"] and "--expected-source-sha" in stage["run"]
    document = WORKFLOW.read_text()
    assert "tsh ssh" not in document and "scp " not in document


def test_both_compilation_profiles_are_required_and_need_no_docker_privilege():
    for controller in (
        "scripts/run_lepton_model_builder_image_qa.py",
        "scripts/run_lepton_pnms_venv_qa.py",
    ):
        step = step_containing(controller)
        assert "if" not in step, "both QA profiles must run on every dispatch"
        assert "continue-on-error" not in step
    document = WORKFLOW.read_text()
    for forbidden in (
        "run_lepton_pnms_container_qa.py",
        "PNMS_DOCKER_AUTH_SECRET",
        "--privileged",
        "docker.sock",
    ):
        assert forbidden not in document


def invoke_step(tmp_path, controller, *, wheelhouse="", includes=None):
    capture = tmp_path / "argv.json"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    fake_uv.chmod(0o755)
    environment = os.environ | {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
        "PNMS_QA_IMAGE": "registry/qa@sha256:" + "a" * 64,
        "PNMS_QA_SOURCE_SHA": "b" * 40,
        "PNMS_QA_SHAPE": "gpu.h100-sxm",
        "PNMS_QA_RUN": "qa-test",
        "PNMS_BUILDER_IMAGE": "registry/builder@sha256:" + "c" * 64,
        "PNMS_BUILDER_RUNNER": "/outputs/qa inputs/runner $(touch injected).py",
        "PNMS_QA_SOURCE_ROOT": "/outputs/qa inputs/source $(touch injected)",
        "PNMS_BUILDER_SOURCE_ROOT": "/outputs/qa inputs/source $(touch injected)",
        "PNMS_BUILDER_SOURCE_MANIFEST": "/outputs/qa inputs/source.json",
        "PNMS_BUILDER_SOURCE_MANIFEST_SHA256": "e" * 64,
        "PNMS_VENV_REQUIREMENTS": "/outputs/qa inputs/requirements $(touch injected).txt",
        "PNMS_VENV_REQUIREMENTS_SHA256": "d" * 64,
        "PNMS_VENV_WHEELHOUSE": wheelhouse,
    }
    subprocess.run(
        ["bash", "-c", step_containing(controller, includes)["run"]],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    assert not (tmp_path / "injected").exists()
    return json.loads(capture.read_text()), environment


def test_builder_profile_uses_shipped_image_and_mounted_runner_literally(tmp_path):
    controller = "scripts/run_lepton_model_builder_image_qa.py"
    arguments, environment = invoke_step(tmp_path, controller)
    assert arguments[:3] == ["run", "python", controller]
    for flag, variable in (
        ("--image", "PNMS_BUILDER_IMAGE"),
        ("--runner", "PNMS_BUILDER_RUNNER"),
        ("--source-root", "PNMS_BUILDER_SOURCE_ROOT"),
        ("--source-manifest", "PNMS_BUILDER_SOURCE_MANIFEST"),
        ("--source-manifest-sha256", "PNMS_BUILDER_SOURCE_MANIFEST_SHA256"),
        ("--expected-source-sha", "PNMS_QA_SOURCE_SHA"),
        ("--resource-shape", "PNMS_QA_SHAPE"),
    ):
        assert arguments[arguments.index(flag) + 1] == environment[variable]
    assert arguments[arguments.index("--run-id") + 1] == "qa-test-builder-image"


@pytest.mark.parametrize("wheelhouse", ["", "/outputs/wheels with spaces"])
def test_fresh_venv_profile_binds_requirements_and_optional_offline_wheels(
    tmp_path, wheelhouse
):
    controller = "scripts/run_lepton_pnms_venv_qa.py"
    arguments, environment = invoke_step(tmp_path, controller, wheelhouse=wheelhouse)
    assert arguments[:3] == ["run", "python", controller]
    for flag, variable in (
        ("--image", "PNMS_QA_IMAGE"),
        ("--expected-source-sha", "PNMS_QA_SOURCE_SHA"),
        ("--resource-shape", "PNMS_QA_SHAPE"),
        ("--requirements", "PNMS_VENV_REQUIREMENTS"),
        ("--requirements-sha256", "PNMS_VENV_REQUIREMENTS_SHA256"),
        ("--repository", "PNMS_QA_SOURCE_ROOT"),
    ):
        assert arguments[arguments.index(flag) + 1] == environment[variable]
    if wheelhouse:
        assert arguments[arguments.index("--wheelhouse") + 1] == wheelhouse
    else:
        assert "--wheelhouse" not in arguments
    assert arguments[arguments.index("--run-id") + 1] == "qa-test-venv"


def test_http_keeps_exact_local_bundle_handoff_and_always_preserves_evidence():
    steps = workflow()["jobs"]["qualify"]["steps"]
    local = step_containing("scripts/run_lepton_pnms_qa.py")
    http = step_containing(
        "scripts/run_lepton_pnms_http_qa.py", "$PNMS_QA_RUN-local/summary.json"
    )
    assert steps.index(local) < steps.index(http)
    assert (
        '--local-summary "/outputs/native-inference/$PNMS_QA_RUN-local/summary.json"'
        in http["run"]
    )
    evidence = next(
        step for step in steps if "actions/upload-artifact@" in step.get("uses", "")
    )
    assert evidence["if"] == "always()"
    assert evidence["with"]["path"] == "qa/artifacts/pnms/"


def test_fresh_venv_http_uses_the_same_bundles_interpreter_and_staged_source(tmp_path):
    controller = "scripts/run_lepton_pnms_http_qa.py"
    selector = "$PNMS_QA_RUN-venv/summary.json"
    step = step_containing(controller, selector)
    steps = workflow()["jobs"]["qualify"]["steps"]
    assert steps.index(
        step_containing("scripts/run_lepton_pnms_venv_qa.py")
    ) < steps.index(step)
    assert "if" not in step and "continue-on-error" not in step
    arguments, environment = invoke_step(tmp_path, controller, includes=selector)
    assert (
        arguments[arguments.index("--local-summary") + 1]
        == "/outputs/native-inference/qa-test-venv/summary.json"
    )
    assert (
        arguments[arguments.index("--python") + 1]
        == "/outputs/native-inference/qa-test-venv/venv/bin/python"
    )
    assert (
        arguments[arguments.index("--repository") + 1]
        == environment["PNMS_QA_SOURCE_ROOT"]
    )
    assert arguments[arguments.index("--mode") + 1] == "host"
    assert "--include-python" not in arguments
    sys.path.insert(0, str(ROOT))
    module = import_module("scripts.run_lepton_pnms_http_qa")
    module.build_parser().parse_args(arguments[3:])


def test_staging_exports_validated_shared_paths_and_content_hashes(tmp_path):
    summary = {
        "source_root": "/outputs/native-inference/qa-test-stage/source",
        "source_manifest": "/outputs/native-inference/qa-test-stage/source-manifest.json",
        "source_manifest_sha256": "a" * 64,
        "requirements_file": "/outputs/native-inference/qa-test-stage/source/qa/native_inference/venv-requirements.txt",
        "requirements_sha256": "b" * 64,
    }
    destination = tmp_path / "qa/artifacts/pnms/qa-test-stage/job-summary.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(summary))
    capture = tmp_path / "github-env"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 0\n")
    fake_uv.chmod(0o755)
    subprocess.run(
        ["bash", "-c", step_containing("scripts/run_lepton_pnms_stage_qa.py")["run"]],
        cwd=tmp_path,
        env=os.environ
        | {
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PNMS_QA_IMAGE": "registry/qa@sha256:" + "c" * 64,
            "PNMS_QA_SOURCE_SHA": "d" * 40,
            "PNMS_QA_RUN": "qa-test",
            "GITHUB_ENV": str(capture),
        },
        check=True,
    )
    values = dict(line.split("=", 1) for line in capture.read_text().splitlines())
    assert values["PNMS_BUILDER_SOURCE_ROOT"] == summary["source_root"]
    assert values["PNMS_BUILDER_SOURCE_MANIFEST"] == summary["source_manifest"]
    assert (
        values["PNMS_BUILDER_SOURCE_MANIFEST_SHA256"]
        == summary["source_manifest_sha256"]
    )
    assert (
        values["PNMS_BUILDER_RUNNER"]
        == summary["source_root"] + "/qa/native_inference/model_builder_image_job.py"
    )
    assert values["PNMS_VENV_REQUIREMENTS"] == summary["requirements_file"]
    assert values["PNMS_VENV_REQUIREMENTS_SHA256"] == summary["requirements_sha256"]


@pytest.mark.parametrize(
    "controller",
    [
        "scripts/run_lepton_model_builder_image_qa.py",
        "scripts/run_lepton_pnms_venv_qa.py",
    ],
)
def test_workflow_flags_match_the_real_controller_parser(tmp_path, controller):
    arguments, _ = invoke_step(tmp_path, controller)
    sys.path.insert(0, str(ROOT))
    module = import_module(controller.removesuffix(".py").replace("/", "."))
    module.build_parser().parse_args(arguments[3:])
