"""CPU contracts for pnms CI wiring; acceptance executes only on Lepton."""

import hashlib
import json
from pathlib import Path
import struct

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_profile_workflow_requires_fresh_venv_and_exact_bundle_http():
    document = workflow("lepton_pnms_qa.yml")
    steps = document["jobs"]["qualify"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    required = (
        "run_lepton_pnms_stage_qa.py",
        "run_lepton_pnms_venv_qa.py",
        "run_lepton_pnms_qa.py",
    )
    assert all(command in commands for command in required)
    assert commands.count("run_lepton_pnms_http_qa.py") == 2
    assert (
        '--python "/outputs/native-inference/$PNMS_QA_RUN-venv/venv/bin/python"'
        in commands
    )
    assert (
        '--local-summary "/outputs/native-inference/$PNMS_QA_RUN-venv/summary.json"'
        in commands
    )
    assert (
        '--local-summary "/outputs/native-inference/$PNMS_QA_RUN-local/summary.json"'
        in commands
    )
    assert "--include-python" in commands
    assert "leptonai==0.27.2" in commands
    assert "run_lepton_model_builder_image_qa.py" in commands
    assert "run_pnms_release" not in commands
    assert not any(step.get("continue-on-error") for step in steps)
    assert steps[-1]["if"] == "always()"
    assert steps[-1]["uses"] == "actions/upload-artifact@v4"


def test_cpu_ci_adds_pnms_contracts_and_keeps_existing_sdk_gates():
    jobs = workflow("ci.yml")["jobs"]
    steps = jobs["inference-cpu"]["steps"]
    selected = next(
        step for step in steps if step.get("name") == "Test unified pnms CPU contracts"
    )
    assert "tests/test_pnms*.py" in selected["run"]
    assert "redis==" in selected["run"]
    assert "-o xfail_strict=true" in selected["run"]
    names = {step.get("name") for step in steps}
    assert {
        "Run builder contract tests without ML frameworks",
        "Test native authoring and sealed release CPU contracts",
        "Build and test the core C++ SDK",
        "Test reusable native tensor and custom workflow host on CPU",
        "Test Rust native worker with the installed CPU workflow host",
    } <= names
    assert {"inference-windows", "inference-windows-aoti", "rust-test"} <= jobs.keys()


def test_development_recipe_binds_current_source_and_builder():
    text = (ROOT / "qa/native_inference/Dockerfile.pnms").read_text()
    assert "ARG PNMS_QA_BASE_IMAGE\n" in text
    assert "COPY source-inventory.json /tmp/pnms-source-inventory.json" in text
    assert 'metadata["source_dirty"] is False' in text
    assert 'metadata["source_inventory_sha256"]' in text
    assert "set(records) == names" in text
    assert 'record["sha256"]' in text and 'record["mode"]' in text
    assert "/workspace/native-inference/model-builder" in text
    assert "rm -rf /workspace /opt/physicsnemo-qa /opt/native-release" in text


def test_python_base_uses_explicit_current_source_and_fresh_runtime_binaries():
    text = (ROOT / "qa/native_inference/Dockerfile.pnms-python-base").read_text()
    assert "ARG PNMS_PYTHON_QA_SOURCE\n" in text
    assert "ARG SOURCE_SHA\n" in text
    assert 'metadata["source_sha"] == os.environ["SOURCE_SHA"]' in text
    assert "rm -rf /app /tmp/pnms-python-qa-target /opt/physicsnemo-serve-venv" in text
    assert "cargo build --locked -p inference_server -p worker-runtime" in text
    assert "cp -a scripts python /app/" in text


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
def test_release_projects_have_independent_raw_upload_goldens(backend):
    root = ROOT / "qa/native_inference/projects" / f"tensor-files-{backend}"
    project = yaml.safe_load((root / "pnms.yaml").read_text())
    assert project["build"]["backend"] == backend
    assert project["ingress"]["content_type"] == "multipart/form-data"
    cases = json.loads((root / "fixtures/cases.json").read_text())["cases"]
    assert len(cases) == 3
    for index, case in enumerate(cases):
        source = root / case["files"]["features"]["source"]
        values = struct.unpack("<4f", source.read_bytes())
        golden = struct.pack("<4f", *(2 * value + 1 for value in values))
        assert golden == (root / f"fixtures/golden-{index}.f32").read_bytes()
        assert case["expected_artifacts"]["output.f32"] == {
            "sha256": hashlib.sha256(golden).hexdigest(),
            "size_bytes": len(golden),
        }
