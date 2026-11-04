"""Application locks guard the selected stack without changing model locks."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import build, model, native  # noqa: E402
from pnms_app.common import read_json, write_json  # noqa: E402
from test_pnms_build import prepared as _prepared  # noqa: E402

prepared = _prepared


def _selected_stack(monkeypatch, *, change=None, compiled=None):
    original = model.build_models

    def selected(project, output, **kwargs):
        result = original(project, output, **kwargs)
        environment = result["environment"]
        environment["stack"].update(
            python="3.12.0",
            cxx11_abi=True,
            native_providers={
                "schema_version": 1,
                "system_libraries": {
                    "libstdc++.so.6": {"sha256": "a" * 64, "size_bytes": 23}
                },
            },
        )
        environment["gpu"].update(uuid="GPU-first", binding="0", mig_mode="disabled")
        if change:
            change(environment)
        if kwargs.get("environment_check"):
            kwargs["environment_check"](environment)
        if compiled is not None:
            compiled.append("model compilation")
        return result

    monkeypatch.setattr(model, "build_models", selected)
    return selected


def test_first_success_generates_lock_and_retains_snapshot_in_receipt(
    prepared, monkeypatch
):
    _selected_stack(monkeypatch)
    result = build.build_application(prepared, repository=ROOT)
    path = prepared["root"] / "pnms.lock"
    assert path.is_file(), "a successful application build must create pnms.lock"
    lock = read_json(path)
    record = result["environment_lock"]
    assert record["snapshot"] == lock
    assert len(record["digest"]) == 64
    assert read_json(Path(result["root"]) / "build.json")["environment_lock"] == record
    encoded = path.read_text()
    assert str(prepared["root"]) not in encoded
    assert "GPU-first" not in encoded
    assert "sdk_prefix" not in encoded
    assert lock["native"]["gpu"]["policy"] == "build-product-only"


def test_native_stack_drift_rejects_before_model_compilation(prepared, monkeypatch):
    first = _selected_stack(monkeypatch)
    build.build_application(prepared, repository=ROOT)
    compiled = []

    def changed(project, output, **kwargs):
        def gate(environment):
            environment["gpu"]["name"] = "NVIDIA H200"
            kwargs["environment_check"](environment)

        result = first(project, output, **{**kwargs, "environment_check": gate})
        compiled.append("model compilation")
        return result

    monkeypatch.setattr(model, "build_models", changed)
    with pytest.raises(ValueError, match="pnms build.*--refresh-lock"):
        build.build_application(prepared, repository=ROOT)
    assert compiled == []


def test_same_version_provider_replacement_is_lock_drift(prepared, monkeypatch):
    first = _selected_stack(monkeypatch)
    build.build_application(prepared, repository=ROOT)

    def changed(project, output, **kwargs):
        def gate(environment):
            environment["stack"]["native_providers"]["system_libraries"][
                "libstdc++.so.6"
            ]["sha256"] = "b" * 64
            kwargs["environment_check"](environment)

        return first(project, output, **{**kwargs, "environment_check": gate})

    monkeypatch.setattr(model, "build_models", changed)
    with pytest.raises(ValueError, match="provider|stack|environment"):
        build.build_application(prepared, repository=ROOT)


def test_lock_ignores_allocation_paths_and_model_workflow_source(prepared, monkeypatch):
    first = _selected_stack(monkeypatch)
    previous = build.build_application(prepared, repository=ROOT)
    original = (prepared["root"] / "pnms.lock").read_bytes()
    (prepared["root"] / "model/model-build.json").write_text('{"new": "weights"}')
    (prepared["root"] / "native").mkdir()
    (prepared["root"] / "native/workflow.cpp").write_text(
        "// edited user workflow source"
    )

    def relocated(project, output, **kwargs):
        def gate(environment):
            environment["python"] = "/different/venv/bin/python"
            environment["sdk_prefix"] = "/another/sdk"
            environment["gpu"].update(uuid="GPU-other", binding="3")
            environment["stack"]["torch_root"] = "/another/venv/torch"
            kwargs["environment_check"](environment)

        return first(project, output, **{**kwargs, "environment_check": gate})

    monkeypatch.setattr(model, "build_models", relocated)
    current = build.build_application(prepared, repository=ROOT)
    assert current["environment_lock"] == previous["environment_lock"]
    assert (prepared["root"] / "pnms.lock").read_bytes() == original


def test_failed_refresh_preserves_lock_and_both_success_pointers(prepared, monkeypatch):
    first = _selected_stack(monkeypatch)
    previous = build.build_application(prepared, repository=ROOT)
    state = prepared["root"] / ".pnms"
    write_json(state / "last-image.json", {"sentinel": "prior image"})
    saved = {
        path: path.read_bytes()
        for path in (
            prepared["root"] / "pnms.lock",
            state / "last-build.json",
            state / "last-image.json",
        )
    }

    def changed(project, output, **kwargs):
        def gate(environment):
            environment["stack"]["torch"] = "new version"
            kwargs["environment_check"](environment)

        return first(project, output, **{**kwargs, "environment_check": gate})

    monkeypatch.setattr(model, "build_models", changed)
    monkeypatch.setattr(
        native,
        "build_workflow",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("workflow compilation failed")
        ),
    )
    with pytest.raises(RuntimeError, match="workflow compilation failed"):
        build.build_application(prepared, repository=ROOT, refresh_lock=True)
    for path, contents in saved.items():
        assert path.read_bytes() == contents
    assert build.last_build(prepared)["build_id"] == previous["build_id"]


def test_successful_refresh_replaces_lock_without_rewriting_prior_receipt(
    prepared, monkeypatch
):
    first = _selected_stack(monkeypatch)
    previous = build.build_application(prepared, repository=ROOT)
    path = Path(previous["root"]) / "build.json"
    original_receipt = path.read_bytes()

    def changed(project, output, **kwargs):
        def gate(environment):
            environment["stack"]["torch"] = "new version"
            kwargs["environment_check"](environment)

        return first(project, output, **{**kwargs, "environment_check": gate})

    monkeypatch.setattr(model, "build_models", changed)
    current = build.build_application(prepared, repository=ROOT, refresh_lock=True)
    assert (
        current["environment_lock"]["digest"] != previous["environment_lock"]["digest"]
    )
    assert (
        read_json(prepared["root"] / "pnms.lock")
        == current["environment_lock"]["snapshot"]
    )
    assert path.read_bytes() == original_receipt


def test_unpublished_container_stage_never_creates_lock(prepared, monkeypatch):
    _selected_stack(monkeypatch)
    prepared["_container_image"] = "builder@sha256:" + "c" * 64
    built = build.build_application(prepared, repository=ROOT, publish_result=False)
    assert not (prepared["root"] / "pnms.lock").exists()
    assert not (prepared["root"] / ".pnms/last-build.json").exists()
    assert (
        built["environment_lock"]["snapshot"]["environment"]["image"]
        == prepared["_container_image"]
    )


def test_cli_exposes_explicit_lock_refresh_without_model_builder_changes():
    from pnms_app.cli import parser

    parsed = parser().parse_args(["build", "--refresh-lock"])
    assert parsed.refresh_lock is True


@pytest.mark.parametrize("drift", ["distribution", "gpu"])
def test_observed_stack_drift_stops_before_sdk_creation(
    prepared, monkeypatch, tmp_path, drift
):
    import json
    import subprocess
    from test_pnms_model import Runner, fake_runtime
    from pnms_app import environment_lock

    runtime, prefix = fake_runtime(tmp_path)
    selected = Runner()

    def observed_runner(command, **kwargs):
        response = selected(command, **kwargs)
        if "-c" in command and command[list(command).index("-c") + 1] == model._PROBE:
            observed = json.loads(response.stdout)
            observed["dependencies"] = {
                "distributions": [
                    ["example", "2" if changed and drift == "distribution" else "1"]
                ]
            }
            if changed and drift == "gpu":
                observed["gpus"][0]["name"] = "NVIDIA H200"
            return subprocess.CompletedProcess(command, 0, json.dumps(observed), "")
        assert str(command[0]) != "cmake", (
            "lock drift must stop before creating the SDK"
        )
        return response

    changed = False
    baseline = model.prepare_environment(
        prepared,
        tmp_path / "baseline",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=observed_runner,
    )
    write_json(
        prepared["root"] / "pnms.lock", environment_lock.snapshot(prepared, baseline)
    )
    changed = True

    def compile_models(project, output, **kwargs):
        options = {}
        if kwargs.get("environment_preflight"):
            options["before_compile"] = kwargs["environment_preflight"]
        model.prepare_environment(
            project,
            output / "environment",
            repository=ROOT,
            runner=observed_runner,
            **options,
        )
        pytest.fail("model compilation must not run on a drifting stack")

    monkeypatch.setattr(model, "build_models", compile_models)
    with pytest.raises(ValueError, match="pnms build.*--refresh-lock"):
        build.build_application(prepared, repository=ROOT)


def test_python_lock_uses_selected_venv_distribution_identity_without_ml_imports(
    tmp_path, monkeypatch
):
    import venv
    from pnms_app import python

    selected = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(selected)
    selected_python = selected / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "torch.py").write_text(
        "raise RuntimeError('Python plugin lock must not import torch')"
    )
    metadata = packages / "example-1.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Metadata-Version: 2.1\nName: Example\nVersion: 1\n")
    monkeypatch.setenv("PYTHONPATH", str(packages))
    app = tmp_path / "app"
    app.mkdir()
    (app / "plugin.yaml").write_text("plugin source")
    project = {
        "root": app,
        "plugin": app,
        "name": "python-app",
        "runtime": "python",
        "build": {"python": str(selected_python)},
    }
    compiled = []

    def source_build(project, output, **kwargs):
        compiled.append(True)
        write_json(output / "plugin/plugin.json", {"source": "frozen plugin"})
        return {
            "plugin_root": str(output / "plugin"),
            "environment": {"python_executable": project["build"]["python"]},
        }

    monkeypatch.setattr(python, "build_project", source_build)
    first = build.build_application(project, repository=ROOT)
    identity = first["environment_lock"]["snapshot"]["environment"]["python"]
    assert any(
        package["name"] == "example" and package["version"] == "1"
        for package in identity["distributions"]
    )
    assert str(selected) not in (app / "pnms.lock").read_text()
    compiled.clear()
    metadata.write_text("Metadata-Version: 2.1\nName: Example\nVersion: 2\n")
    with pytest.raises(ValueError, match="pnms build.*--refresh-lock"):
        build.build_application(project, repository=ROOT)
    assert not compiled
    assert build.last_build(project)["build_id"] == first["build_id"]


def test_failed_image_publication_restores_lock_and_both_pointers(
    prepared, monkeypatch
):
    first = _selected_stack(monkeypatch)
    old = build.build_application(prepared, repository=ROOT)
    state = prepared["root"] / ".pnms"
    write_json(state / "last-image.json", {"sentinel": "last image"})
    previous = {
        path: path.read_bytes()
        for path in (
            prepared["root"] / "pnms.lock",
            state / "last-build.json",
            state / "last-image.json",
        )
    }
    from pnms_app import images

    monkeypatch.setattr(images, "build_image", lambda *a, **kw: {"status": "complete"})
    monkeypatch.setattr(
        build,
        "publish_image",
        lambda *a: (_ for _ in ()).throw(OSError("image pointer write failed")),
    )

    def changed(project, output, **kwargs):
        def gate(environment):
            environment["stack"]["torch"] = "new version"
            kwargs["environment_check"](environment)

        return first(project, output, **{**kwargs, "environment_check": gate})

    monkeypatch.setattr(model, "build_models", changed)
    with pytest.raises(OSError, match="image pointer write failed"):
        build.build_application(
            prepared, repository=ROOT, image="app:2", refresh_lock=True
        )
    for path, contents in previous.items():
        assert path.read_bytes() == contents
    assert build.last_build(prepared)["build_id"] == old["build_id"]


@pytest.mark.parametrize("fail", [False, True])
def test_container_refresh_is_committed_only_after_outer_packaging(
    tmp_path, monkeypatch, fail
):
    import json
    import subprocess
    from pnms_app import container, environment_lock, images
    from test_pnms_container import _completed_build, _project, GPU

    project = _project(tmp_path / "app")
    project.update(name="app", runtime="cpp", backend="aoti")
    staged = _completed_build(project["root"] / ".pnms/builds/staged")
    staged["environment_lock"] = environment_lock.check(
        {**project, "_container_image": project["build"]["builder_image"]},
        staged["environment"],
    )
    write_json(
        Path(staged["root"]) / "build.json",
        {key: value for key, value in staged.items() if key != "root"},
    )
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(staged), "")

    def package(*args, **kwargs):
        assert not (project["root"] / "pnms.lock").exists()
        assert not (project["root"] / ".pnms/last-build.json").exists()
        if fail:
            raise RuntimeError("image packaging failed")
        return {"status": "complete", "image_id": "sha256:" + "f" * 64}

    monkeypatch.setattr(container.subprocess, "run", run)
    monkeypatch.setattr(images, "build_image", package)
    if fail:
        with pytest.raises(RuntimeError, match="image packaging failed"):
            container.build(project, repository=ROOT, image="app:1", refresh_lock=True)
        assert not (project["root"] / "pnms.lock").exists()
        assert not (project["root"] / ".pnms/last-build.json").exists()
    else:
        result = container.build(
            project, repository=ROOT, image="app:1", refresh_lock=True
        )
        assert (
            read_json(project["root"] / "pnms.lock")
            == result["environment_lock"]["snapshot"]
        )
    assert "--refresh-lock" in commands[0]


def test_toolchain_install_paths_do_not_change_native_lock():
    from pnms_app import environment_lock
    from copy import deepcopy

    project = {"runtime": "cpp", "backend": "aoti"}
    selected = {
        "stack": {
            "compiler": {
                "cxx": "clang version 19.1\nTarget: x86_64-linux-gnu\nInstalledDir: /original/toolchain/bin\n"
            }
        },
        "gpu": {"name": "H100", "architecture": "sm90"},
    }
    moved = deepcopy(selected)
    moved["stack"]["compiler"]["cxx"] = selected["stack"]["compiler"]["cxx"].replace(
        "/original/toolchain", "/relocated/toolchain"
    )
    assert environment_lock.snapshot(project, selected) == environment_lock.snapshot(
        project, moved
    )

    moved["stack"]["compiler"]["cxx"] = moved["stack"]["compiler"]["cxx"].replace(
        "19.1", "20.0"
    )
    assert environment_lock.snapshot(project, selected) != environment_lock.snapshot(
        project, moved
    )
