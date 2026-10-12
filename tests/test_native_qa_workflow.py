"""Exercise the manual workflow command without contacting Lepton."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/lepton_native_qa.yml"
)


@pytest.mark.parametrize("assets", ["", "/outputs/assets/manifest with spaces.json"])
def test_workflow_passes_literal_inputs_to_the_standalone_native_runner(
    tmp_path, assets
):
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    steps = workflow["jobs"]["qualify"]["steps"]
    step = next(item for item in steps if item.get("name") == "Run native inference QA")
    capture = tmp_path / "argv.json"
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    image = "registry/qa@sha256:" + "a" * 64 + "$(touch injected)"
    environment = os.environ | {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
        "NATIVE_QA_IMAGE": image,
        "NATIVE_QA_SOURCE_SHA": "b" * 40,
        "NATIVE_QA_PROFILE": "full" if assets else "smoke",
        "NATIVE_QA_RESOURCE_SHAPE": "gpu.h100-sxm",
        "NATIVE_QA_ASSETS": assets,
        "NATIVE_QA_TIMEOUT": "3600",
        "NATIVE_QA_RUN_ID": "native-test-1",
    }
    subprocess.run(
        ["bash", "-c", step["run"]], env=environment, cwd=tmp_path, check=True
    )
    arguments = json.loads(capture.read_text())
    assert arguments[:3] == [
        "run",
        "python",
        "qa/scripts/run_lepton_native_inference_qa.py",
    ]
    assert arguments[arguments.index("--image") + 1] == image
    assert arguments[arguments.index("--expected-source-sha") + 1] == "b" * 40
    assert arguments[arguments.index("--profile") + 1] == (
        "full" if assets else "smoke"
    )
    assert not (tmp_path / "injected").exists()
    if assets:
        assert arguments[-2:] == ["--assets", assets]
    else:
        assert "--assets" not in arguments
    assert steps[-1]["if"] == "always()"
