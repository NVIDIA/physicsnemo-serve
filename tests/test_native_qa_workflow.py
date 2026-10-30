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


def test_provider_workflow_can_build_a_source_pinned_image_before_qualification():
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "provider" in inputs["profile"]["options"]
    assert inputs["image"]["required"] == "false"
    qualify = workflow["jobs"]["qualify"]
    assert qualify["needs"] == "prepare-image"
    step = next(
        s for s in qualify["steps"] if s.get("name") == "Run native inference QA"
    )
    assert "needs.prepare-image.outputs.image" in step["env"]["NATIVE_QA_IMAGE"]
    image_step = next(
        s for s in workflow["jobs"]["prepare-image"]["steps"] if s.get("id") == "image"
    )
    assert "--target native-qa" in image_step["run"]
    assert "CMAKE_CUDA_ARCHITECTURES=90" in image_step["run"]
    assert 'SOURCE_SHA="$NATIVE_QA_SOURCE_SHA"' in image_step["run"]


@pytest.mark.parametrize("prebuilt", [False, True])
def test_image_step_publishes_only_the_verified_digest_for_the_requested_source(
    tmp_path, prebuilt
):
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    step = next(
        s for s in workflow["jobs"]["prepare-image"]["steps"] if s.get("id") == "image"
    )
    calls = tmp_path / "docker-calls.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + """import json,os,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as output:output.write(json.dumps(args)+'\\n')
if args[0]=='login':sys.stdin.read()
if args[:2]==['buildx','build']:
    path=args[args.index('--metadata-file')+1]
    with open(path,'w') as output:json.dump({'containerimage.digest':'sha256:'+'d'*64},output)
"""
    )
    git = tmp_path / "git"
    git.write_text(f"#!{sys.executable}\nprint('" + "a" * 40 + "')\n")
    for executable in (docker, git):
        executable.chmod(0o755)
    image = "registry.example/qa@sha256:" + "b" * 64
    env = os.environ | dict(
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        CALLS=str(calls),
        GITHUB_OUTPUT=str(tmp_path / "output"),
        RUNNER_TEMP=str(tmp_path),
        NATIVE_QA_IMAGE=image if prebuilt else "",
        NATIVE_QA_SOURCE_SHA="a" * 40,
        NATIVE_QA_REPOSITORY="registry.example/qa",
        NGC_REGISTRY="registry.example",
        NGC_ORG="org",
        NGC_IMAGE="qa",
        NGC_REGISTRY_PASSWORD="fixture-password",
    )
    subprocess.run(["bash", "-c", step["run"]], env=env, cwd=tmp_path, check=True)
    expected = image if prebuilt else "registry.example/qa@sha256:" + "d" * 64
    assert (tmp_path / "output").read_text() == f"image={expected}\n"
    if prebuilt:
        assert not calls.exists(), "selecting an existing digest must not invoke Docker"
    else:
        invocations = [json.loads(line) for line in calls.read_text().splitlines()]
        build = next(args for args in invocations if args[:2] == ["buildx", "build"])
        assert (
            "SOURCE_SHA=" + "a" * 40 in build and "CMAKE_CUDA_ARCHITECTURES=90" in build
        )
        assert "--push" in build and build[build.index("--target") + 1] == "native-qa"


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
