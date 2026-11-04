"""CPU regressions for GPU selection across build and launch allocations."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import build, cli, container, service  # noqa: E402


def test_launch_reuses_a_build_binding_within_the_current_allocation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    built = {"environment": {"gpu": {"binding": "3", "uuid": "GPU-old"}}}
    assert service._gpu_binding({"build": {}}, built) == "3"


def test_launch_does_not_reuse_a_build_binding_outside_the_allocation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    built = {"environment": {"gpu": {"binding": "7", "uuid": "GPU-old"}}}
    with pytest.raises(ValueError, match="--gpu|allocat"):
        service._gpu_binding({"build": {}}, built)


def test_unallocated_launch_selects_the_current_single_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    built = {"environment": {"gpu": {"binding": "7", "uuid": "GPU-producer"}}}
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "GPU-current-node\n", "")

    monkeypatch.setattr(service.subprocess, "run", run)
    assert service._gpu_binding({"build": {}}, built) == "GPU-current-node"
    assert commands == [["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]]


def test_unallocated_launch_requires_selection_when_current_node_has_many_gpus(
    monkeypatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "GPU-first\nGPU-second\n", ""
        ),
    )
    built = {"environment": {"gpu": {"uuid": "GPU-producer"}}}
    with pytest.raises(ValueError, match="--gpu|one|select"):
        service._gpu_binding({"build": {}}, built)


@pytest.mark.parametrize("operation", ["run", "serve"])
def test_launch_commands_accept_an_explicit_gpu(operation):
    arguments = [operation, "--gpu", "1"]
    if operation == "run":
        arguments += ["--input", "request.json"]
    assert cli.parser().parse_args(arguments).gpu == "1"


def test_container_run_remaps_the_inner_gpu(tmp_path, monkeypatch):
    build_root = tmp_path / ".pnms/builds/new"
    (build_root / "bundle").mkdir(parents=True)
    project = {
        "root": tmp_path,
        "runtime": "cpp",
        "build": {"gpu": "1"},
    }
    built = {
        "root": str(build_root),
        "bundle": "bundle",
        "source_sha": "a" * 40,
        "environment": {
            "kind": "container",
            "builder_image": "compiler@sha256:" + "b" * 64,
        },
    }
    inner_arguments = []

    def command(project, **options):
        inner_arguments.extend(options["arguments"])
        return ["docker", "run"]

    monkeypatch.setattr(cli, "load_project", lambda *args, **kwargs: project)
    monkeypatch.setattr(build, "last_build", lambda *args, **kwargs: built)
    monkeypatch.setattr(container, "command", command)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps({"status": "passed"}), ""
        ),
    )
    args = cli.parser().parse_args(["run", str(tmp_path), "--input", "request.json"])
    assert cli.execute(args)["status"] == "passed"
    assert inner_arguments[inner_arguments.index("--gpu") + 1] == "0"
