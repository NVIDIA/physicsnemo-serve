import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import container  # noqa: E402

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def _project(root, *, gpu=None):
    root.mkdir(parents=True, exist_ok=True)
    settings = {"builder_image": "builder@sha256:" + "a" * 64}
    if gpu is not None:
        settings["gpu"] = gpu
    return {"root": root, "build": settings, "models": {}}


def _completed_build(root):
    from pnms_app.common import write_json

    (root / "bundle").mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "build_id": root.name,
        "runtime": "cpp",
        "bundle": "bundle",
        "source_sha": "b" * 40,
        "environment": {"kind": "local", "gpu": {"binding": "0"}},
    }
    write_json(root / "build.json", receipt)
    return {**receipt, "root": str(root)}


def test_whole_application_container_binds_only_allocated_gpu(tmp_path):
    project = {
        "root": tmp_path,
        "build": {"builder_image": "builder@sha256:" + "a" * 64},
        "models": {},
    }
    args = ["build", str(tmp_path), "--environment", "local"]
    command = container.command(
        project,
        repository=ROOT,
        arguments=args,
        environment={"CUDA_VISIBLE_DEVICES": GPU},
        source_sha="b" * 40,
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--gpus") + 1] == "device=" + GPU
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert "PNMS_SOURCE_SHA=" + "b" * 40 in command
    assert command[-4:] == args
    assert "--gpus=all" not in command
    image = command.index(project["build"]["builder_image"])
    assert command[image + 1 : image + 3] == ["python3", str(ROOT / "pnms")]
    assert "--entrypoint" not in command


def test_ambiguous_allocation_needs_explicit_selection(tmp_path):
    project = {
        "root": tmp_path,
        "build": {"builder_image": "builder@sha256:" + "a" * 64},
        "models": {},
    }
    with pytest.raises(ValueError, match="one|select|GPU"):
        container.command(
            project,
            repository=ROOT,
            arguments=[],
            environment={"CUDA_VISIBLE_DEVICES": "1,3"},
        )


def test_container_image_must_be_pinned_before_docker(tmp_path):
    project = {
        "root": tmp_path,
        "build": {"builder_image": "builder:latest"},
        "models": {},
    }
    with pytest.raises(ValueError, match="immutable|pinned"):
        container.command(
            project,
            repository=ROOT,
            arguments=[],
            environment={"CUDA_VISIBLE_DEVICES": GPU},
        )


def test_build_remaps_configured_gpu_into_the_isolated_container(tmp_path, monkeypatch):
    project = _project(tmp_path / "app", gpu="1")
    built = _completed_build(project["root"] / ".pnms/builds/new")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(built), "")

    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(container.subprocess, "run", run)
    result = container.build(project, repository=ROOT)
    command = commands[0]
    assert command[command.index("--gpus") + 1] == "device=3"
    inner = command[command.index(str(ROOT / "pnms")) + 1 :]
    assert inner[inner.index("--gpu") + 1] == "0"
    assert "--container-process" in inner
    assert "--no-publish" in inner
    assert result["environment"]["kind"] == "container"


def test_container_build_retains_nested_json_after_base_startup_banner(
    tmp_path, monkeypatch, capsys
):
    project = _project(tmp_path / "app")
    built = _completed_build(project["root"] / ".pnms/builds/new")
    built["nested"] = {"items": [{"value": 3}]}
    banner = "======== NVIDIA runtime ========\nCUDA compatibility initialized\n"
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(
        container.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, banner + json.dumps(built, indent=2), ""
        ),
    )

    result = container.build(project, repository=ROOT)

    assert result["nested"] == {"items": [{"value": 3}]}
    assert capsys.readouterr().err == banner


@pytest.mark.parametrize(
    "output", ["multiple", "trailing", "missing", "wrong_document"]
)
def test_container_build_rejects_ambiguous_or_incomplete_results(
    tmp_path, monkeypatch, output
):
    project = _project(tmp_path / "app")
    built = _completed_build(project["root"] / ".pnms/builds/new")
    document = json.dumps(built, indent=2)
    stdout = {
        "multiple": '{"other": 1}\n' + document,
        "trailing": document + "\nmore output",
        "missing": "base started its own server",
        "wrong_document": '{"server": "different application"}',
    }[output]
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(
        container.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout, ""),
    )

    with pytest.raises(ValueError, match="unambiguous|final.*JSON|forward"):
        container.build(project, repository=ROOT)


def test_container_inference_preserves_base_banner_and_returns_result(
    tmp_path, monkeypatch, capsys
):
    from pnms_app import build, cli

    project = {**_project(tmp_path / "app"), "runtime": "cpp"}
    built = _completed_build(project["root"] / ".pnms/builds/new")
    built["environment"].update(
        kind="container", builder_image=project["build"]["builder_image"]
    )
    request = tmp_path / "input.json"
    request.write_text("{}")
    response = {"result": {"payload": {"values": [2, 4]}}}
    banner = "NVIDIA initialization complete\n"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(cli, "load_project", lambda *args, **kwargs: project)
    monkeypatch.setattr(build, "last_build", lambda *args, **kwargs: built)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, banner + json.dumps(response, indent=2), ""
        ),
    )
    args = cli.parser().parse_args(
        ["run", str(project["root"]), "--input", str(request)]
    )

    result = cli.execute(args)

    assert result == response
    assert capsys.readouterr().err == banner


def test_command_mounts_external_request_and_output_paths(tmp_path):
    project = _project(tmp_path / "app")
    request = tmp_path / "requests/request.json"
    request.parent.mkdir()
    request.write_text("{}")
    artifact = tmp_path / "inputs/tensor.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"input")
    output_parent = tmp_path / "results"
    output_parent.mkdir()
    extra_mounts = {request: True, artifact: True, output_parent: False}
    original = dict(extra_mounts)
    arguments = ["run", str(project["root"]), "--input", str(request)]
    command = container.command(
        project,
        repository=ROOT,
        arguments=arguments,
        environment={"CUDA_VISIBLE_DEVICES": GPU},
        extra_mounts=extra_mounts,
    )
    mounts = [
        command[index + 1] for index, part in enumerate(command) if part == "--mount"
    ]
    for path, readonly in extra_mounts.items():
        expected = f"type=bind,src={path.resolve()},dst={path.resolve()}"
        if readonly:
            expected += ",readonly"
        assert expected in mounts
    assert extra_mounts == original
    assert command[-len(arguments) :] == arguments


def test_failed_image_stage_preserves_a_concurrent_success(tmp_path, monkeypatch):
    from pnms_app import images

    project = _project(tmp_path / "app")
    built = _completed_build(project["root"] / ".pnms/builds/new")
    pointer = project["root"] / ".pnms/last-build.json"
    pointer.write_bytes(b"previous success")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(built), "")

    def fail_image(*args, **kwargs):
        pointer.write_bytes(b"concurrent success")
        raise RuntimeError("image assembly failed")

    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(container.subprocess, "run", run)
    monkeypatch.setattr(images, "build_image", fail_image)
    with pytest.raises(RuntimeError, match="image assembly failed"):
        container.build(project, repository=ROOT, image="app:1")
    assert pointer.read_bytes() == b"concurrent success"
    assert "--no-publish" in commands[0]


def test_unpublished_container_build_keeps_the_success_pointer(tmp_path, monkeypatch):
    project = _project(tmp_path / "app")
    built = _completed_build(project["root"] / ".pnms/builds/new")
    pointer = project["root"] / ".pnms/last-build.json"
    pointer.write_bytes(b"previous success")
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(
        container.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(built), ""
        ),
    )
    result = container.build(project, repository=ROOT, publish_result=False)
    assert result["status"] == "complete"
    assert pointer.read_bytes() == b"previous success"


def _mounts(command):
    return [
        command[index + 1] for index, part in enumerate(command) if part == "--mount"
    ]


def test_container_inference_does_not_read_or_mount_original_model_sources(tmp_path):
    project = _project(tmp_path / "app")
    removed = tmp_path / "removed-training-project/model-build.json"
    project["models"] = {"primary": {"project": removed}}
    source = tmp_path / "removed-native-source"
    project["workflow"] = {"template": "custom", "source_dir": str(source)}
    command = container.command(
        project,
        repository=ROOT,
        arguments=["run", str(project["root"])],
        environment={"CUDA_VISIBLE_DEVICES": GPU},
    )
    assert all(str(removed.parent) not in mount for mount in _mounts(command))
    assert all(str(source) not in mount for mount in _mounts(command))


@pytest.mark.parametrize("action", ["build", "check"])
@pytest.mark.parametrize("absolute", [False, True])
def test_container_mounts_external_custom_workflow_source(tmp_path, action, absolute):
    from pnms_app.config import load_project

    repository = tmp_path / "checkout"
    repository.mkdir()
    root = tmp_path / "app"
    root.mkdir()
    source = tmp_path / "shared-native"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(custom_workflow)\n")
    (root / "model-build.json").write_text("{}")
    (root / "pnms.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "custom-app",
                "runtime": "cpp",
                "build": {"builder_image": "builder@sha256:" + "a" * 64},
                "models": {"primary": {"project": "model-build.json"}},
                "workflow": {
                    "template": "custom",
                    "source_dir": str(source) if absolute else "../shared-native",
                    "target": "custom_workflow",
                },
            }
        )
    )
    project = load_project(root, repository)

    command = container.command(
        project,
        repository=repository,
        arguments=[action, str(root)],
        environment={"CUDA_VISIBLE_DEVICES": GPU},
    )

    assert f"type=bind,src={source},dst={source},readonly" in _mounts(command)
    assert f"type=bind,src={root},dst={root}" in _mounts(command)


@pytest.mark.parametrize("location", ["application", "application-child", "output"])
def test_workflow_source_mount_preserves_writable_bindings(tmp_path, location):
    project = _project(tmp_path / "app")
    root = project["root"]
    source = {
        "application": root,
        "application-child": root / "native",
        "output": tmp_path / "output",
    }[location]
    source.mkdir(exist_ok=True)
    project["workflow"] = {"template": "custom", "source_dir": str(source)}
    extra_mounts = {source: False} if location == "output" else {}

    command = container.command(
        project,
        repository=ROOT,
        arguments=["build", str(root)],
        environment={"CUDA_VISIBLE_DEVICES": GPU},
        extra_mounts=extra_mounts,
    )

    assert f"type=bind,src={root},dst={root}" in _mounts(command)
    assert f"type=bind,src={source},dst={source},readonly" not in _mounts(command)
    if location == "output":
        assert f"type=bind,src={source},dst={source}" in _mounts(command)


@pytest.mark.parametrize("field", ["checkpoint", "config", "toolchain_lock", "assets"])
def test_container_build_mounts_external_model_profile_inputs(tmp_path, field):
    project = _project(tmp_path / "app")
    adopted = tmp_path / "adopted-model/model-build.json"
    adopted.parent.mkdir()
    dependency = tmp_path / "profile-inputs" / (field + ".bin")
    dependency.parent.mkdir()
    dependency.write_bytes(b"profile input")
    value = "../profile-inputs/" + dependency.name
    selected = {field: {"asset": value} if field == "assets" else value}
    adopted.write_text(
        json.dumps({"format_version": 2, "profiles": {"alternate": selected}})
    )
    project["models"] = {"primary": {"project": adopted, "profile": "alternate"}}
    command = container.command(
        project,
        repository=ROOT,
        arguments=["build", str(project["root"])],
        environment={"CUDA_VISIBLE_DEVICES": GPU},
    )
    parent = dependency.parent.resolve()
    assert f"type=bind,src={parent},dst={parent},readonly" in _mounts(command)


def test_container_build_mounts_legacy_recipe_and_its_external_inputs(tmp_path):
    project = _project(tmp_path / "app")
    adopted = tmp_path / "adopted-model/model-build.json"
    adopted.parent.mkdir()
    recipe = tmp_path / "legacy-recipe/recipe.json"
    recipe.parent.mkdir()
    paths = {}
    for name in ("config", "checkpoint", "asset"):
        path = tmp_path / ("external-" + name) / "input.bin"
        path.parent.mkdir()
        path.write_bytes(b"recipe input")
        paths[name] = path
    recipe.write_text(
        json.dumps(
            {
                "config": {"path": "../external-config/input.bin"},
                "checkpoint": {"path": str(paths["checkpoint"])},
                "assets": {"weights": {"path": "../external-asset/input.bin"}},
            }
        )
    )
    adopted.write_text(
        json.dumps({"format_version": 1, "recipe": "../legacy-recipe/recipe.json"})
    )
    project["models"] = {"primary": {"project": adopted}}
    command = container.command(
        project,
        repository=ROOT,
        arguments=["build", str(project["root"])],
        environment={"CUDA_VISIBLE_DEVICES": GPU},
    )
    for path in [recipe, *paths.values()]:
        parent = path.parent.resolve()
        assert f"type=bind,src={parent},dst={parent},readonly" in _mounts(command)
