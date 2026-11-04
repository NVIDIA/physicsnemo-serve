"""Native release CI wiring; these checks do not qualify a release image."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/native-workflow-release.yml"


def workflow(path=WORKFLOW):
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def step_containing(steps, command):
    matches = [step for step in steps if command in step.get("run", "")]
    assert len(matches) == 1, f"one CI step must invoke {command}"
    return matches[0]


def test_dispatch_exposes_all_carried_native_workflows_and_backends():
    document = workflow()
    inputs = document["on"]["workflow_dispatch"]["inputs"]
    assert set(document["on"]) == {"workflow_dispatch"}
    assert inputs["workflow"]["options"] == [
        "transolver",
        "tensor",
        "raw",
        "custom",
        "multipart",
    ]
    assert inputs["backend"]["options"] == ["aoti", "tensorrt"]
    select = step_containing(document["jobs"]["release"]["steps"], "case ")
    for value in inputs["workflow"]["options"]:
        assert value + ")" in select["run"]
    assert "--content-type multipart/form-data" in select["run"]
    assert "--template tensor" in select["run"]
    assert "--template cpp" in select["run"]
    assert "qa/scripts/create_native_raw_variant.py" in select["run"]


def test_component_qa_is_a_source_bound_lepton_gate_before_release():
    steps = workflow()["jobs"]["release"]["steps"]
    components = step_containing(steps, "qa/scripts/run_lepton_native_components.py")
    release = step_containing(steps, "args=(--workspace")
    assert steps.index(components) < steps.index(release)
    assert "continue-on-error" not in components
    for flag in (
        "--image",
        "--expected-source-sha",
        "--resource-shape",
        "--artifact-dir",
        "--run-id",
    ):
        assert flag in components["run"]
    assert "cpu.medium" in components["run"]


def test_component_image_is_built_from_sealed_source_and_locked_dependencies():
    steps = workflow()["jobs"]["release"]["steps"]
    build = step_containing(steps, "qa/native_inference/components.Dockerfile")
    for value in (
        "git bundle create",
        "source.bundle",
        "cargo-source",
        "uv export",
        "--frozen",
        "requirements.txt",
        "SOURCE_SHA=",
        "--push",
        "containerimage.digest",
    ):
        assert value in build["run"]


@pytest.mark.skipif(os.name == "nt", reason="Linux Docker RUN shell contract")
@pytest.mark.parametrize("inherited_qa", [False, True])
def test_component_image_replaces_inherited_qa_from_sealed_source(
    tmp_path, inherited_qa
):
    source = tmp_path / "workspace"
    deploy = source / "native-inference/deploy"
    deploy.mkdir(parents=True)
    (deploy / "native_release.py").write_text("current deploy\n")
    current_qa = {
        "native_inference/assets.py": "current assets\n",
        "scripts/run_native_cfd_job.py": "current controller\n",
    }
    for name in (
        "native_inference/contract.py",
        "scripts/run_lepton_native_inference_qa.py",
    ):
        current_qa[name] = (ROOT / "qa" / name).read_text()
    for name, content in current_qa.items():
        path = source / "qa" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = source / "deploy/config.py"
    config.parent.mkdir()
    config.write_text(
        "def load_deploy_config():\n    return {'marker': 'current deploy config'}\n"
    )

    installed = tmp_path / "opt/native-release"
    installed.mkdir(parents=True)
    preserved = installed / "plugins/base.py"
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_text("preserved base file\n")
    installed_qa = installed / "source/qa"
    if inherited_qa:
        installed_qa.mkdir(parents=True)
        (installed_qa / "obsolete.py").write_text("stale QA\n")
        stale_config = installed / "source/deploy/config.py"
        stale_config.parent.mkdir()
        stale_config.write_text("raise RuntimeError('stale deploy config')\n")
    dockerfile = (ROOT / "qa/native_inference/components.Dockerfile").read_text()
    install = dockerfile.split("WORKDIR /workspace\nRUN ", 1)[1].split("\nENV ", 1)[0]
    install = install.replace("/opt/native-release", shlex.quote(str(installed)))
    install = install.replace("/workspace", shlex.quote(str(source)))
    child = subprocess.run(
        ["/bin/sh", "-c", install],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert {
        path.relative_to(installed_qa).as_posix(): path.read_text()
        for path in installed_qa.rglob("*")
        if path.is_file()
    } == current_qa
    assert preserved.read_text() == "preserved base file\n"
    assert (installed / "native_release.py").read_text() == "current deploy\n"
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json, runpy, sys; "
            "module = runpy.run_path(sys.argv[1]); "
            "print(json.dumps({'root': str(module['REPO_ROOT']), "
            "'config': module['load_deploy_config']()}))",
            str(installed_qa / "scripts/run_lepton_native_inference_qa.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert json.loads(probe.stdout) == {
        "root": str(source),
        "config": {"marker": "current deploy config"},
    }


def test_release_image_is_qualified_before_optional_exact_digest_promotion():
    steps = workflow()["jobs"]["release"]["steps"]
    release = step_containing(steps, "args=(--workspace")
    promotion = step_containing(steps, "imagetools create")
    assert steps.index(release) < steps.index(promotion)
    assert "--promote" not in release["run"]
    assert (
        'scripts/plugin_dev.py release "$NATIVE_WORKFLOW_PATH" "${args[@]}"'
        in release["run"]
    )
    assert promotion.get("if") == "inputs.promote != ''"
    assert "continue-on-error" not in release and "continue-on-error" not in promotion
    assert "promotion requires the passed source-bound release" in promotion["run"]
    assert (
        "promotion requires passed source-bound Lepton component evidence"
        in promotion["run"]
    )
    assert "--prefer-index=false" in promotion["run"]
    assert "promotion changed the exact qualified image digest" in promotion["run"]


def test_release_workflow_scope_and_always_retained_evidence():
    document = workflow()
    body = WORKFLOW.read_text()
    for excluded in (
        "release-export",
        "native-managed",
        "managed endpoint",
        "run_lepton_native_managed",
        "scripts/pnms",
        "onboard",
    ):
        assert excluded not in body
    steps = document["jobs"]["release"]["steps"]
    evidence = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert evidence.get("if") == "always()"
    paths = evidence["with"]["path"]
    for required in (
        "native-components",
        "native-release-plan.json",
        "native-promotion.json",
        "native-release/release.json",
        "native-release/qualification.json",
        "qualification-evidence/extracted/",
    ):
        assert required in paths
    assert ".token" not in paths and "private" not in paths


@pytest.mark.parametrize(
    "change",
    [
        None,
        "release-failed",
        "release-source",
        "components-failed",
        "components-source",
        "components-image",
        "candidate-tag",
        "repository",
        "tag",
    ],
)
def test_promotion_guard_rejects_unqualified_or_different_receipts(tmp_path, change):
    steps = workflow()["jobs"]["release"]["steps"]
    body = step_containing(steps, "imagetools create")["run"]
    guard = re.search(r"python - <<'PY'\n(.*?)\nPY", body, flags=re.S).group(1)
    source = "a" * 40
    repository = "registry.example/native"
    image = repository + "@sha256:" + "b" * 64
    component_image = repository + "@sha256:" + "c" * 64
    release = {"status": "passed", "source_sha": source, "candidate": {"image": image}}
    components = {
        "status": "passed",
        "source_sha": source,
        "image_digest": component_image,
    }
    if change == "release-failed":
        release["status"] = "failed"
    elif change == "release-source":
        release["source_sha"] = "d" * 40
    elif change == "components-failed":
        components["status"] = "failed"
    elif change == "components-source":
        components["source_sha"] = "d" * 40
    elif change == "components-image":
        components["image_digest"] = image
    elif change == "candidate-tag":
        release["candidate"]["image"] = repository + ":latest"
    elif change == "repository":
        release["candidate"]["image"] = "other.example/native@sha256:" + "b" * 64
    workspace = tmp_path / "release"
    workspace.mkdir()
    (workspace / "release.json").write_text(json.dumps(release))
    receipt = tmp_path / "native-components" / "run"
    receipt.mkdir(parents=True)
    (receipt / "component-receipt.json").write_text(json.dumps(components))
    result = subprocess.run(
        [sys.executable, "-c", guard],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "RUNNER_TEMP": str(tmp_path),
            "NATIVE_WORKSPACE": str(workspace),
            "NATIVE_SOURCE_SHA": source,
            "NATIVE_COMPONENT_RUN_ID": "run",
            "NATIVE_COMPONENT_IMAGE": component_image,
            "NATIVE_REPOSITORY": repository,
            "NATIVE_PROMOTE": "invalid/tag" if change == "tag" else "qualified",
        },
    )
    if change is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == repository + ":qualified " + image
    else:
        assert result.returncode != 0
        assert not result.stdout


def test_main_ci_explicitly_runs_native_authoring_and_release_cpu_contracts():
    steps = workflow(ROOT / ".github/workflows/ci.yml")["jobs"]["inference-cpu"][
        "steps"
    ]
    step = step_containing(steps, "native-release-contract-tests.xml")
    assert "continue-on-error" not in step
    for suite in (
        "tests/test_native_authoring.py",
        "tests/test_native_user_cli.py",
        "tests/test_native_user_release.py",
        "tests/test_native_release*.py",
        "tests/test_native_lepton_components.py",
        "tests/test_native_workflow_release_ci.py",
        "qa/scripts/test_create_native_raw_variant.py",
    ):
        assert suite in step["run"]
    assert "xfail_strict=true" in step["run"]


def test_main_ci_preserves_pr3_windows_and_installed_consumer_gates():
    jobs = workflow(ROOT / ".github/workflows/ci.yml")["jobs"]
    windows = "\n".join(
        step.get("run", "") for step in jobs["inference-windows"]["steps"]
    )
    cpu = "\n".join(step.get("run", "") for step in jobs["inference-cpu"]["steps"])
    for fragment in (
        "PNMS_CUSTOM_WORKFLOW_SOURCE_DIR",
        "pnms_test_custom",
        "tests/install_consumer",
        "tests/cli_test.py",
        "workflow-host-windows/bin/pnms-native-run.exe",
        "transolver-windows-contract",
        "--no-tests=error",
    ):
        assert fragment in windows
    for fragment in (
        "test_python_image_planner.py",
        "tests/install_consumer",
        "tests/cli_test.py",
        "cargo test --locked -p native-worker --features cpp",
        "PNMS_CUSTOM_WORKFLOW_SOURCE_DIR",
        "cpp-runtime/tests/install_consumer.py",
        "--component onnxruntime --shared",
    ):
        assert fragment in cpu
    assert "inference-windows-aoti" in jobs
