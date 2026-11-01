import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import images  # noqa: E402


def _git(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _remove_git(repository):
    def writable(function, path, error):
        os.chmod(path, 0o777)
        function(path)

    shutil.rmtree(repository / ".git", onerror=writable)


@pytest.fixture
def source_repository(tmp_path):
    repository = tmp_path / "repository"
    for relative in (
        "Cargo.toml",
        "Cargo.lock",
        ".cargo/config.toml",
        "crates/service/main.rs",
        "scripts/plugin_dev.py",
        "python/helper.py",
        "application-cli/docker/Dockerfile.native",
        "application-cli/docker/Dockerfile.python",
        "application-cli/src/pnms_app/python_release.py",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / relative
        path.write_bytes(original.read_bytes() if original.is_file() else b"source\n")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "core.autocrlf", "false")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "source",
    )
    return repository


def _image_inputs(root, runtime, docker, source_sha):
    project = {"name": "app", "serve": {"image": "runtime@sha256:" + "d" * 64}}
    built = {"runtime": runtime, "source_sha": source_sha}
    if runtime == "cpp":
        built.update(
            bundle="bundle",
            bundle_digest="a" * 64,
            environment={"stack": docker.stack},
            workflow={"compatibility": docker.stack},
        )
    else:
        from pnms_app.common import inventory

        plugin = root / "plugin"
        plugin.mkdir()
        (plugin / "plugin.yaml").write_text("metadata:\n  id: python-demo\n")
        built["python"] = {"plugin_root": "plugin", "files": inventory(plugin)}
    return project, built


@pytest.mark.parametrize("fault", ["revision", "rust", "python", "missing_git"])
def test_image_rejects_unverified_packaged_source(
    tmp_path, monkeypatch, source_repository, fault
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    sha = _git(source_repository, "rev-parse", "HEAD")
    runtime = "python" if fault == "python" else "cpp"
    if fault == "revision":
        _git(
            source_repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "different source",
        )
    elif fault == "missing_git":
        _remove_git(source_repository)
        monkeypatch.setenv("PNMS_SOURCE_SHA", sha)
    else:
        relative = (
            "application-cli/src/pnms_app/python_release.py"
            if runtime == "python"
            else "crates/service/main.rs"
        )
        (source_repository / relative).write_text("dirty source\n")
    docker = Docker()
    project, built = _image_inputs(tmp_path, runtime, docker, sha)
    with pytest.raises(
        ValueError, match="source.*(--from-build|metadata|revision|changed)"
    ):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=docker,
        )
    assert not docker.commands


@pytest.mark.parametrize("runtime", ["cpp", "python"])
def test_image_uses_frozen_source_if_checkout_changes_during_docker_checks(
    tmp_path, monkeypatch, source_repository, runtime
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    sha = _git(source_repository, "rev-parse", "HEAD")
    relative = (
        "crates/service/main.rs"
        if runtime == "cpp"
        else "application-cli/src/pnms_app/python_release.py"
    )
    original = (source_repository / relative).read_bytes()
    untracked = source_repository / "application-cli/private-workspace.txt"
    untracked.write_text("must not enter release")
    docker = Docker()

    def mutate_checkout(command, **kwargs):
        if not docker.commands:
            (source_repository / relative).write_text("changed after validation")
        if command[:3] == ["docker", "buildx", "build"]:
            context = Path(command[-1])
            assert context != source_repository
            assert (context / relative).read_bytes() == original
            assert not (context / untracked.relative_to(source_repository)).exists()
            assert Path(command[command.index("--file") + 1]).is_relative_to(context)
        return docker(command, **kwargs)

    project, built = _image_inputs(tmp_path, runtime, docker, sha)
    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="app:0.1",
        runner=mutate_checkout,
    )
    assert result["packaging_source"]["source_sha"] == sha
    assert result["packaging_source"]["kind"] == "git"
    assert result["status"] == "complete"


def _export_source(repository, output):
    from pnms_app.common import file_identity

    sha = _git(repository, "rev-parse", "HEAD")
    records = {
        relative: {
            **file_identity(repository / relative),
            "mode": (repository / relative).stat().st_mode & 0o777,
            "symlink": None,
        }
        for relative in _git(repository, "ls-files").splitlines()
    }
    output.mkdir()
    inventory = output / "source-inventory.json"
    inventory.write_text(json.dumps(records))
    metadata = output / "source.json"
    metadata.write_text(
        json.dumps(
            {
                "source_sha": sha,
                "source_dirty": False,
                "source_archive_sha256": "f" * 64,
                "source_inventory_sha256": file_identity(inventory)["sha256"],
            }
        )
    )
    _remove_git(repository)
    return sha, metadata


@pytest.mark.parametrize("runtime", ["cpp", "python"])
def test_gitless_qa_source_inventory_is_explicitly_verified_and_frozen(
    tmp_path, monkeypatch, source_repository, runtime
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    sha, metadata = _export_source(source_repository, tmp_path / "qa-receipt")
    monkeypatch.setenv("PNMS_SOURCE_METADATA", str(metadata))
    docker = Docker()
    project, built = _image_inputs(tmp_path, runtime, docker, sha)
    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="app:0.1",
        runner=docker,
    )
    assert result["packaging_source"]["kind"] == "source-inventory"
    assert (
        result["packaging_source"]["source_inventory_sha256"]
        == json.loads(metadata.read_text())["source_inventory_sha256"]
    )
    assert result["status"] == "complete"


@pytest.mark.parametrize("fault", ["bytes", "inventory", "revision", "dirty"])
def test_exported_source_rejects_changed_content_or_unbound_inventory(
    tmp_path, source_repository, fault
):
    sha, metadata = _export_source(source_repository, tmp_path / "qa-receipt")
    if fault == "bytes":
        (source_repository / "crates/service/main.rs").write_text("changed source")
    elif fault == "inventory":
        metadata.with_name("source-inventory.json").write_text("{}")
    else:
        identity = json.loads(metadata.read_text())
        identity["source_sha" if fault == "revision" else "source_dirty"] = (
            "0" * 40 if fault == "revision" else True
        )
        metadata.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="source.*(changed|differs|dirty)"):
        images.source_context(
            source_repository, sha, tmp_path / "frozen", native=True, metadata=metadata
        )


def test_python_context_uses_only_release_helper_and_pinned_base_source(
    tmp_path, source_repository
):
    sha = _git(source_repository, "rev-parse", "HEAD")
    (source_repository / "crates/service/main.rs").write_text("unrelated dirty Rust")
    context = tmp_path / "frozen"
    images.source_context(source_repository, sha, context, native=False)
    assert {
        path.relative_to(context).as_posix()
        for path in context.rglob("*")
        if path.is_file()
    } == {
        "application-cli/docker/Dockerfile.python",
        "application-cli/src/pnms_app/python_release.py",
    }


def test_packaging_accepts_actual_clean_qa_capture_receipt(tmp_path, source_repository):
    from scripts.prepare_pnms_qa_context import prepare

    for relative in (
        "pnms",
        "qa/native_inference/Dockerfile.pnms",
        "qa/native_inference/venv-requirements.txt",
        "native-inference/model-builder/src/model_builder/build/cli.py",
        "native-inference/model-builder/images/Dockerfile.builder",
        "native-inference/model-builder/images/Dockerfile.builder.dockerignore",
    ):
        path = source_repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    _git(source_repository, "add", ".")
    _git(
        source_repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "QA source",
    )
    sha = _git(source_repository, "rev-parse", "HEAD")
    captured = tmp_path / "capture"
    identity = prepare(source_repository, captured, expected_source_sha=sha)
    _remove_git(source_repository)
    result = images.source_context(
        source_repository,
        sha,
        tmp_path / "frozen",
        native=True,
        metadata=captured / "dev-context/source.json",
    )
    assert result["source_inventory_sha256"] == identity["source_inventory_sha256"]


def test_exported_inventory_parses_only_the_bytes_whose_digest_was_verified(
    tmp_path, monkeypatch, source_repository
):
    sha, metadata = _export_source(source_repository, tmp_path / "qa-receipt")
    inventory = metadata.with_name("source-inventory.json")
    original = Path.read_bytes

    def replace_after_read(path):
        data = original(path)
        if path == inventory:
            inventory.write_text("{}")
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    result = images.source_context(
        source_repository, sha, tmp_path / "frozen", native=True, metadata=metadata
    )
    assert result["source_sha"] == sha
    assert (tmp_path / "frozen/crates/service/main.rs").is_file()


def _native_image_cargo_recipe():
    recipe = (ROOT / "application-cli/docker/Dockerfile.native").read_text()
    build_stage = recipe.split("\nFROM ")[1].replace("\\\n", " ").splitlines()
    workdir = PurePosixPath(
        next(
            line.removeprefix("WORKDIR ")
            for line in build_stage
            if line.startswith("WORKDIR ")
        )
    )
    cargo_run = next(line for line in build_stage if "&& cargo build " in line)
    prefix, arguments = cargo_run.split("&& cargo build ", 1)
    arguments = shlex.split(arguments)
    target = next(
        (
            line.removeprefix("ENV CARGO_TARGET_DIR=")
            for line in build_stage
            if line.startswith("ENV CARGO_TARGET_DIR=")
        ),
        str(workdir / "target"),
    )
    if "--target-dir" in arguments:
        target = arguments[arguments.index("--target-dir") + 1]
    cleanup = [
        PurePosixPath(parts[2])
        for command in prefix.split("&&")
        if (parts := shlex.split(command))[:2] == ["rm", "-rf"]
    ]
    copied = next(
        shlex.split(line)[2:-1]
        for line in recipe.splitlines()
        if line.startswith("COPY --from=service-build ")
    )
    return workdir, PurePosixPath(target), cleanup, copied


def test_native_image_cargo_target_is_isolated_cleared_and_copied():
    workdir, target, cleanup, copied = _native_image_cargo_recipe()
    assert target.is_absolute() and target != workdir / "target"
    assert target in cleanup, "a pinned runtime may already contain the build target"
    assert copied == [
        str(target / "release" / name)
        for name in ("native-worker", "inference_server", "worker-runtime")
    ]


class Docker:
    def __init__(
        self,
        stack=None,
        *,
        corrupted_contents=False,
        wrong_image_id=False,
        wrong_published_config=False,
        python_server=True,
        containerd=False,
        missing_config_digest=False,
        overwritten_entrypoint=False,
        base_forwards_commands=True,
    ):
        self.commands = []
        self.corrupted_contents = corrupted_contents
        self.wrong_image_id = wrong_image_id
        self.wrong_published_config = wrong_published_config
        self.python_server = python_server
        self.containerd = containerd
        self.missing_config_digest = missing_config_digest
        self.overwritten_entrypoint = overwritten_entrypoint
        self.base_forwards_commands = base_forwards_commands
        self.stack = stack or {
            "system": "Linux",
            "machine": "x86_64",
            "libc": ["glibc", "2.39"],
            "torch": "2.10",
            "cuda": "13.1",
            "cxx11_abi": True,
            "tensorrt": None,
            "torch_libraries": {
                "lib/libtorch.so": {"sha256": "f" * 64, "size_bytes": 1024}
            },
            "system_libraries": {"libc.so.6": {"sha256": "1" * 64, "size_bytes": 2048}},
            "schema_version": 1,
            "driver_libraries": [],
            "workflow_binary": {"sha256": "3" * 64, "size_bytes": 4096},
        }

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[:3] == ["docker", "run", "--rm"]:
            if "PNMS_BASE_COMMAND_FORWARD_V1" in " ".join(command):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "PNMS_BASE_COMMAND_FORWARD_V1\n"
                    if self.base_forwards_commands
                    else "default server started\n",
                    "",
                )
            if "PNMS_PYTHON_BASE_V1" in " ".join(command):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        {
                            "status": "passed" if self.python_server else "failed",
                            "capability": "python-server-v1",
                        }
                    ),
                    "",
                )
            if "PNMS_PYTHON_CONTENT_V1" in " ".join(command):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        {
                            "status": "passed",
                            "plugin_digest": command[-1],
                            "workflow_id": "python-demo",
                            "files": 3,
                        }
                    ),
                    "",
                )
            if "PNMS_IMAGE_CONTENT_CHECK_V1" in " ".join(command):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        {
                            "status": "failed" if self.corrupted_contents else "passed",
                            "bundle_digest": "a" * 64,
                            "source_sha": command[-1],
                            "native_load": "passed",
                            "files": 3,
                        }
                    ),
                    "",
                )
            return subprocess.CompletedProcess(command, 0, json.dumps(self.stack), "")
        if "--iidfile" in command:
            Path(command[command.index("--iidfile") + 1]).write_text(
                "sha256:" + ("8" if self.containerd else "b") * 64
            )
        if command[:3] == ["docker", "image", "inspect"]:
            inspected = {
                "Id": "sha256:"
                + ("0" if self.wrong_image_id else "8" if self.containerd else "b")
                * 64,
                "RepoDigests": ["registry/app@sha256:" + "c" * 64],
                "Config": {
                    "Entrypoint": ["/opt/vendor/initialize.sh"],
                    "Cmd": ["python3", "-m", "pnms_app.image_entrypoint"],
                },
            }
            if "@sha256:" not in command[-1] and self.overwritten_entrypoint:
                inspected["Config"]["Entrypoint"] = [
                    "python3",
                    "-m",
                    "pnms_app.image_entrypoint",
                ]
            if self.containerd:
                inspected["Descriptor"] = {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": inspected["Id"],
                    "annotations": {}
                    if self.missing_config_digest
                    else {
                        "config.digest": "sha256:" + "b" * 64,
                    },
                }
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps([inspected]),
                "",
            )
        if command[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "config": {
                            "digest": "sha256:"
                            + ("0" if self.wrong_published_config else "b") * 64
                        },
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")


def test_native_packaging_loads_local_image_and_keeps_truthful_identity(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {"bundle_digest": "a" * 64})
    (tmp_path / "bundle").mkdir()
    docker = Docker()
    project = {
        "name": "app",
        "runtime": "cpp",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    built = {
        "runtime": "cpp",
        "build_id": "id",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="app:0.1",
        runner=docker,
    )
    assert result["status"] == "complete"
    assert result["image_id"] == "sha256:" + "b" * 64
    assert "registry_digest" not in result
    command = next(cmd for cmd in docker.commands if "buildx" in cmd)
    assert "--load" in command and "--push" not in command
    assert not any("--gpus" in cmd for cmd in docker.commands)
    assert result["checks"]["gpu_startup"] == "not_run"
    assert not any("pnms-model-builder" in str(cmd) for cmd in docker.commands)


def test_containerd_manifest_id_is_kept_separate_from_runtime_config_identity(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker(containerd=True)
    project = {"name": "app", "serve": {"image": "runtime@sha256:" + "d" * 64}}
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }

    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="registry/app:0.1",
        runner=docker,
        push=True,
    )

    assert result.get("image_config_digest") == "sha256:" + "b" * 64
    assert result["image_id"] == "sha256:" + "8" * 64
    assert result.get("runtime_identity") == {
        "kind": "image-config",
        "value": result["image_config_digest"],
    }
    assert result.get("docker_identity") == {
        "kind": "oci-manifest",
        "value": result["image_id"],
    }


def test_containerd_image_without_config_metadata_is_not_mislabeled(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker(containerd=True, missing_config_digest=True)
    project = {"name": "app", "serve": {"image": "runtime@sha256:" + "d" * 64}}
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }

    with pytest.raises(ValueError, match="config.*(digest|identity|metadata)"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=docker,
        )
    assert not any(cmd[:2] == ["docker", "push"] for cmd in docker.commands)


@pytest.mark.parametrize("fault", ["overwritten_entrypoint", "base_forwards_commands"])
def test_native_packaging_rejects_lost_or_incompatible_base_initialization(
    tmp_path, monkeypatch, fault, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker(**{fault: fault != "base_forwards_commands"})
    project = {"name": "app", "serve": {"image": "runtime@sha256:" + "d" * 64}}
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }

    with pytest.raises(ValueError, match="entrypoint|initialization|forward"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="registry/app:0.1",
            runner=docker,
            push=True,
        )

    assert not any(cmd[:2] == ["docker", "push"] for cmd in docker.commands)


@pytest.mark.parametrize("fault", ["corrupted_contents", "wrong_image_id"])
def test_packaging_rejects_bad_image_content_or_config_binding(
    tmp_path, monkeypatch, fault, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {"bundle_digest": "a" * 64})
    (tmp_path / "bundle").mkdir()
    docker = Docker(**{fault: True})
    project = {
        "name": "app",
        "runtime": "cpp",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    with pytest.raises(ValueError, match="content|configuration|identity|binding"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=docker,
        )
    assert not any(cmd[:2] == ["docker", "push"] for cmd in docker.commands)
    receipts = list((tmp_path / "images").glob("*/image.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"


def test_same_versions_with_different_native_provider_bytes_are_rejected(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker()
    compatibility = json.loads(json.dumps(docker.stack))
    compatibility["system_libraries"]["libc.so.6"]["sha256"] = "2" * 64
    project = {
        "name": "app",
        "runtime": "cpp",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": compatibility},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    with pytest.raises(ValueError, match="native|provider|libraries|compatible"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=docker,
        )
    assert not any("buildx" in command for command in docker.commands)


def test_python_image_verifies_plugin_and_keeps_native_build_path_out(
    tmp_path, source_repository
):
    from pnms_app.common import digest, inventory

    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("metadata:\n  id: python-demo\n")
    files = inventory(plugin)
    built = {
        "runtime": "python",
        "python": {"plugin_root": "plugin", "files": files},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    project = {
        "name": "python-demo",
        "runtime": "python",
        "serve": {"image": "server@sha256:" + "d" * 64},
    }
    docker = Docker()
    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="python-demo:0.1",
        runner=docker,
    )
    assert result["checks"]["python_base"] == "passed"
    assert result["checks"]["contents"] == "passed"
    assert result["contents"]["plugin_digest"] == digest(files)
    assert result["checks"]["gpu_startup"] == "not_run"
    assert not any(
        "PNMS_IMAGE_CONTENT_CHECK_V1" in " ".join(command)
        for command in docker.commands
    )
    assert not any(
        "pnms-model-builder" in " ".join(command) for command in docker.commands
    )


@pytest.mark.parametrize(
    "relative", [".env", "nested/.env.production", ".env.backup/fake.txt", ".ENV"]
)
@pytest.mark.parametrize("sealed", [False, True])
def test_python_image_rejects_dotenv_snapshot_before_any_docker_command(
    tmp_path, source_repository, relative, sealed
):
    from pnms_app.common import inventory

    docker = Docker()
    project, built = _image_inputs(
        tmp_path, "python", docker, _git(source_repository, "rev-parse", "HEAD")
    )
    plugin = tmp_path / built["python"]["plugin_root"]
    private = plugin / relative
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("PNMS_FAKE_CREDENTIAL=old-snapshot-regression-sentinel\n")
    if sealed:
        # An old successful receipt may legitimately seal the unsafe file.
        built["python"]["files"] = inventory(plugin)
    original = inventory(plugin)

    with pytest.raises(ValueError, match=r"\.env|inventory differs"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="registry/app:0.1",
            push=True,
            runner=docker,
        )

    assert not docker.commands
    assert inventory(plugin) == original, "immutable snapshots must not be filtered"


def test_python_image_rejects_empty_dotenv_directory(tmp_path, source_repository):
    from pnms_app.common import inventory

    docker = Docker()
    project, built = _image_inputs(
        tmp_path, "python", docker, _git(source_repository, "rev-parse", "HEAD")
    )
    plugin = tmp_path / built["python"]["plugin_root"]
    (plugin / ".env.local").mkdir()
    assert inventory(plugin) == built["python"]["files"]

    with pytest.raises(ValueError, match=r"\.env"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="registry/app:0.1",
            push=True,
            runner=docker,
        )

    assert not docker.commands
    assert (plugin / ".env.local").is_dir()


def test_published_manifest_must_reference_the_exact_local_config(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker()
    project = {
        "name": "app",
        "runtime": "cpp",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    built = {
        "runtime": "cpp",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "environment": {"stack": docker.stack},
        "workflow": {"compatibility": docker.stack},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    result = images.build_image(
        project,
        built,
        root=tmp_path,
        repository=source_repository,
        image="registry/app:0.1",
        push=True,
        runner=docker,
    )
    assert result["registry_digest"] == "registry/app@sha256:" + "c" * 64
    assert result["image_id"] != result["registry_digest"]
    docker = Docker(wrong_published_config=True)
    with pytest.raises(ValueError, match="manifest configuration identity"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="registry/app:0.1",
            push=True,
            runner=docker,
        )


def test_cpu_content_verifier_hashes_actual_bundle_without_model_startup(tmp_path):
    from pnms_app.common import digest, inventory

    bundle = tmp_path / "bundle"
    runner = bundle / "workflow/bin/pnms-native-run"
    runner.parent.mkdir(parents=True)
    if sys.platform == "win32":
        from pip._vendor.distlib.scripts import ScriptMaker

        sources = tmp_path / "launcher-sources"
        sources.mkdir()
        (sources / runner.name).write_text(
            '#!python\nprint("pnms-native-run.exe --help")\n', encoding="utf-8"
        )
        maker = ScriptMaker(str(sources), str(runner.parent))
        maker.executable = sys.executable
        [launcher] = maker.make(runner.name)
        runner = Path(launcher)
    else:
        runner.write_text('#!/bin/sh\nprintf "pnms-native-run --help\\n"\n')
        runner.chmod(0o755)
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "runtime": "cpp",
                "source_sha": "e" * 40,
                "workflow": {"runner": runner.relative_to(bundle).as_posix()},
            }
        )
    )
    model = bundle / "model.bin"
    model.write_bytes(b"compiled-native-bytes")
    files = inventory(bundle)
    seal = {"schema_version": 1, "files": files, "bundle_digest": digest(files)}
    (bundle / "inventory.json").write_text(json.dumps(seal))
    command = [
        sys.executable,
        "-c",
        images.CONTENT_PROBE,
        str(bundle),
        seal["bundle_digest"],
        "e" * 40,
    ]
    passed = subprocess.run(command, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(passed.stdout)["status"] == "passed"
    model.write_bytes(b"corrupted-native-bytes")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "content differs from inventory" in rejected.stderr
    optimized = subprocess.run(
        command,
        env={**os.environ, "PYTHONOPTIMIZE": "1"},
        capture_output=True,
        text=True,
    )
    assert optimized.returncode != 0, (
        "image verification must remain active with Python optimization enabled"
    )


def test_incompatible_stack_fails_before_image_build(
    tmp_path, monkeypatch, source_repository
):
    from pnms_app import bundle

    monkeypatch.setattr(bundle, "verify", lambda *args: {})
    docker = Docker()
    project = {"runtime": "cpp", "serve": {"image": "runtime@sha256:" + "d" * 64}}
    built = {
        "runtime": "cpp",
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
        "bundle": "bundle",
        "environment": {"stack": {**docker.stack, "torch": "different"}},
        "workflow": {"compatibility": docker.stack},
    }
    with pytest.raises(ValueError, match="torch|compatible|stack"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=docker,
        )
    assert not any("buildx" in cmd for cmd in docker.commands)


def test_image_requires_explicit_pinned_matched_runtime(tmp_path, source_repository):
    with pytest.raises(ValueError, match="serve.image|immutable|pinned"):
        images.build_image(
            {"runtime": "cpp"},
            {"runtime": "cpp"},
            root=tmp_path,
            repository=source_repository,
            image="app:0.1",
            runner=Docker(),
        )


def test_python_release_rejects_a_framework_only_base_before_build(
    tmp_path, source_repository
):
    from pnms_app.common import inventory

    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("metadata:\n  id: python-demo\n")
    built = {
        "runtime": "python",
        "python": {"plugin_root": "plugin", "files": inventory(plugin)},
        "source_sha": _git(source_repository, "rev-parse", "HEAD"),
    }
    project = {
        "name": "python-demo",
        "runtime": "python",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    docker = Docker(python_server=False)
    with pytest.raises(ValueError, match="server|base|entrypoint"):
        images.build_image(
            project,
            built,
            root=tmp_path,
            repository=source_repository,
            image="python-demo:0.1",
            runner=docker,
        )
    assert not any("buildx" in command for command in docker.commands)
