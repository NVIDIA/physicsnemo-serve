"""The new common CLI is additive and can initialize without ML/cloud tools."""

import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "pnms"), *map(str, args)],
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "backend,template", [("aoti", "tensor"), ("tensorrt", "custom")]
)
def test_native_init_is_local_and_keeps_existing_builder_names(
    tmp_path, backend, template
):
    path = tmp_path / "native-app"
    result = cli(
        "init", path, "--runtime", "cpp", "--backend", backend, "--template", template
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "initialized"
    config = yaml.safe_load((path / "pnms.yaml").read_text())
    assert config["runtime"] == "cpp"
    assert config["build"]["environment"] == "local"
    assert config["build"]["backend"] == backend
    assert not (path / "release.yaml").exists()
    assert not (path / "plugin.yaml").exists(), (
        "native serving projection is generated at build"
    )
    model = json.loads((path / "model/model-build.json").read_text())
    assert model["adapter"] == "build_adapter.py"
    assert model["device"] == "cuda"
    assert "required_gpu_arch" not in model, "probe the allocated GPU during build"
    assert (path / "model/build_adapter.py").is_file()
    assert (path / "native/workflow.cpp").exists() == (template == "custom")
    assert "lepton" not in (path / "pnms.yaml").read_text().lower()
    planned = cli("build", path, "--plan")
    assert planned.returncode == 0, planned.stderr
    plan = json.loads(planned.stdout)
    assert plan["runtime"] == "cpp"
    assert plan["device_verification"] == "pending"
    assert not (path / ".pnms/builds").exists()


def test_native_adopts_external_project_without_changing_its_lock(tmp_path):
    source = tmp_path / "training"
    source.mkdir()
    original = b'{"format_version":2,"name":"model","adapter":"adapter.py"}\n'
    model = source / "model-build.json"
    model.write_bytes(original)
    (source / "model-build.lock.json").write_bytes(b"author lock\n")
    path = tmp_path / "serving"
    result = cli("init", path, "--runtime", "cpp", "--model-project", model)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "initialized"
    assert model.read_bytes() == original
    assert (source / "model-build.lock.json").read_bytes() == b"author lock\n"
    doc = yaml.safe_load((path / "pnms.yaml").read_text())
    assert (path / doc["models"]["primary"]["project"]).resolve() == model
    assert not (path / "model").exists()


def test_conflicting_python_native_settings_fail_before_work(tmp_path):
    path = tmp_path / "python-app"
    path.mkdir()
    (path / "pnms.yaml").write_text(
        "schema_version: 1\nname: example\nruntime: python\nplugin: .\nbuild:\n  backend: aoti\n"
    )
    result = cli("build", path, "--plan")
    assert result.returncode == 2
    assert "backend" in result.stderr and "python" in result.stderr
    assert not (path / ".pnms").exists()


def test_init_refuses_to_overwrite_and_missing_build_is_actionable(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    (path / "keep").write_text("original")
    result = cli("init", path, "--runtime", "cpp")
    assert result.returncode == 2
    assert (path / "keep").read_text() == "original"
    app = tmp_path / "new-app"
    assert cli("init", app, "--runtime", "cpp").returncode == 0
    result = cli("run", app, "--input", app / "examples/request.json")
    assert result.returncode == 2
    assert "pnms build" in result.stderr


def test_duplicate_config_fields_are_not_silently_discarded(tmp_path):
    (tmp_path / "pnms.yaml").write_text(
        "schema_version: 1\nname: a\nruntime: python\nruntime: cpp\n"
    )
    result = cli("build", tmp_path, "--plan")
    assert result.returncode == 2
    assert "duplicate" in result.stderr.lower()


@pytest.mark.parametrize("location", ["plugin", "nested", "symlink"])
def test_python_run_keeps_output_outside_the_selected_frozen_plugin(tmp_path, location):
    app = tmp_path / "app"
    assert cli("init", app).returncode == 0
    built = cli("build", app)
    assert built.returncode == 0, built.stderr
    receipt = json.loads(built.stdout)
    sealed = Path(receipt["root"]) / receipt["python"]["plugin_root"]
    output = sealed if location == "plugin" else sealed / "attempt/nested/run"
    if location == "symlink":
        alias = tmp_path / "selected-plugin"
        alias.symlink_to(sealed, target_is_directory=True)
        output = alias / "attempt/nested/run"
    request = tmp_path / "request.json"
    request.write_text('{"value":7}')

    def snapshot(root):
        return {
            str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
            for path in root.rglob("*")
        }

    original = snapshot(app)
    sealed_original = snapshot(sealed)
    rejected = cli("run", app, "--input", request, "--output", output)
    assert rejected.returncode == 2, rejected.stdout or rejected.stderr
    assert "output" in rejected.stderr and "outside" in rejected.stderr
    assert snapshot(app) == original

    # A sibling of the plugin within the build root remains a valid run directory.
    outside = Path(receipt["root"]) / "outside-plugin"
    accepted = cli("run", app, "--input", request, "--output", outside)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["result"]["doubled"] == 14
    assert (outside / "result.json").is_file()
    assert snapshot(sealed) == sealed_original


def test_python_dev_run_accepts_output_inside_the_source_project(tmp_path):
    app = tmp_path / "app"
    assert cli("init", app).returncode == 0
    request = tmp_path / "request.json"
    request.write_text('{"value":7}')
    output = app / "development-output"
    result = cli("run", app, "--dev", "--input", request, "--output", output)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["doubled"] == 14
    assert (output / "result.json").is_file()


def test_native_container_run_rejects_sealed_output_before_creating_its_parent(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import build, cli as app_cli, container

    bundle = tmp_path / ".pnms/builds/selected/bundle"
    bundle.mkdir(parents=True)
    marker = bundle / "sealed.txt"
    marker.write_bytes(b"sealed\n")
    project = {"root": tmp_path, "runtime": "cpp", "name": "test", "build": {}}
    built = {
        "root": str(bundle.parent),
        "bundle": "bundle",
        "environment": {"kind": "container", "builder_image": "example:sealed"},
        "source_sha": "stub",
    }
    monkeypatch.setattr(app_cli, "load_project", lambda *a, **kw: project)
    monkeypatch.setattr(build, "last_build", lambda *a, **kw: built)
    launches = []
    monkeypatch.setattr(container, "command", lambda *a, **kw: ["stub-container"])

    def run(command, **kwargs):
        launches.append(command)
        return subprocess.CompletedProcess(command, 0, '{"status":"ran"}', "")

    monkeypatch.setattr(subprocess, "run", run)
    output = bundle / "attempt/nested/run"
    args = app_cli.parser().parse_args(
        [
            "run",
            str(tmp_path),
            "--input",
            str(tmp_path / "request.json"),
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="output.*outside"):
        app_cli.execute(args)
    assert not launches
    assert list(bundle.iterdir()) == [marker]
    assert marker.read_bytes() == b"sealed\n"
    args.output = bundle.parent / "outside/attempt"
    assert app_cli.execute(args)["status"] == "ran"
    assert launches == [["stub-container"]]
    assert args.output.parent.is_dir()
    assert list(bundle.iterdir()) == [marker]


@pytest.mark.parametrize(
    "runtime,dev", [("cpp", True), ("cpp", False), ("python", False)]
)
def test_serve_rejects_sealed_output_before_dispatch(
    tmp_path, monkeypatch, runtime, dev
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import build, cli as app_cli, service

    selected = tmp_path / "build" / ("bundle" if runtime == "cpp" else "plugin")
    selected.mkdir(parents=True)
    marker = selected / "sealed.txt"
    marker.write_bytes(b"sealed\n")
    alias = tmp_path / "selected"
    alias.symlink_to(selected, target_is_directory=True)
    project = {"root": tmp_path / "app", "runtime": runtime, "name": "test"}
    built = {
        "root": str(selected.parent),
        "bundle": "bundle",
        "python": {"plugin_root": "plugin"},
    }
    monkeypatch.setattr(app_cli, "load_project", lambda *a, **kw: project)
    monkeypatch.setattr(build, "last_build", lambda *a, **kw: built)
    launches = []

    def serve(*args, **kwargs):
        launches.append(kwargs["output"])
        return {"status": "served"}

    monkeypatch.setattr(service, "serve", serve)
    arguments = [
        "serve",
        str(project["root"]),
        "--output",
        str(alias / "new/../service"),
    ]
    if dev:
        arguments.append("--dev")
    args = app_cli.parser().parse_args(arguments)
    with pytest.raises(ValueError, match="output.*outside"):
        app_cli.execute(args)
    assert launches == []
    assert list(selected.iterdir()) == [marker]
    assert marker.read_bytes() == b"sealed\n"

    args.output = selected.parent / "outside/service"
    assert app_cli.execute(args)["status"] == "served"
    assert launches == [args.output]


def test_python_dev_serve_accepts_workspace_inside_the_source_project(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import build, cli as app_cli, python

    project = {"root": tmp_path, "runtime": "python", "name": "test"}
    monkeypatch.setattr(app_cli, "load_project", lambda *a, **kw: project)

    def unexpected_build(*args, **kwargs):
        pytest.fail("Python development serving must not select a frozen build")

    monkeypatch.setattr(build, "last_build", unexpected_build)
    launches = []

    def serve(*args, **kwargs):
        launches.append(kwargs["workspace"])
        return {"status": "served"}

    monkeypatch.setattr(python, "serve_project", serve)
    output = tmp_path / "development-output"
    args = app_cli.parser().parse_args(
        ["serve", str(tmp_path), "--dev", "--output", str(output)]
    )
    assert app_cli.execute(args)["status"] == "served"
    assert launches == [output]


@pytest.mark.parametrize("runtime", ["cpp", "python"])
@pytest.mark.parametrize("selected", [False, True])
def test_image_serve_dispatches_only_selected_runtime_env_names(
    tmp_path, monkeypatch, runtime, selected
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import build, cli as app_cli, service

    project = {"root": tmp_path / "app", "runtime": runtime, "name": "test"}
    built = {
        "root": str(tmp_path / "build"),
        "bundle": "bundle",
        "python": {"plugin_root": "plugin"},
    }
    monkeypatch.setenv("APP_RUNTIME_TOKEN", "credential-fixture-value")
    monkeypatch.setenv("APP_RUNTIME_ENDPOINT", "endpoint-fixture-value")
    monkeypatch.setenv("APP_UNSELECTED_TOKEN", "unselected-fixture-value")
    monkeypatch.setattr(app_cli, "load_project", lambda *a, **kw: project)
    monkeypatch.setattr(build, "last_build", lambda *a, **kw: built)
    launches = []

    def serve(*args, **kwargs):
        launches.append(kwargs)
        return {"status": "served"}

    monkeypatch.setattr(service, "serve", serve)
    arguments = ["serve", str(project["root"])]
    if selected:
        arguments += [
            "--env",
            "APP_RUNTIME_TOKEN",
            "--env",
            "APP_RUNTIME_ENDPOINT",
            "--env",
            "APP_RUNTIME_TOKEN",
        ]
    args = app_cli.parser().parse_args(arguments)
    assert app_cli.execute(args)["status"] == "served"
    assert len(launches) == 1
    assert launches[0]["env_names"] == (
        ("APP_RUNTIME_TOKEN", "APP_RUNTIME_ENDPOINT") if selected else ()
    )
    assert "fixture-value" not in repr(launches)


@pytest.mark.parametrize("runtime", ["cpp", "python"])
def test_runtime_env_with_dev_fails_before_project_loading(
    tmp_path, monkeypatch, capsys, runtime
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import cli as app_cli

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid runtime environment selection must fail before work")

    monkeypatch.setenv("APP_RUNTIME_TOKEN", "credential-fixture-value")
    monkeypatch.setattr(app_cli, "load_project", unexpected)
    output = tmp_path / runtime / "service"
    assert (
        app_cli.main(
            [
                "serve",
                str(tmp_path / runtime),
                "--dev",
                "--env",
                "APP_RUNTIME_TOKEN",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "--env" in error and "--dev" in error and "inherits" in error
    assert "credential-fixture-value" not in error
    assert not output.exists()


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("APP_RUNTIME_TOKEN=credential-fixture-value", "names only"),
        ("INVALID-NAME", "names only"),
        ("1INVALID", "names only"),
        ("", "names only"),
        ("APP_MISSING_TOKEN", "not exported"),
        ("PORT", "launcher-owned"),
    ],
)
def test_invalid_runtime_env_fails_without_values_or_workspace_side_effects(
    tmp_path, monkeypatch, capsys, selection, expected
):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import cli as app_cli

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid runtime environment selection must fail before work")

    monkeypatch.delenv("APP_MISSING_TOKEN", raising=False)
    monkeypatch.setenv("PORT", "credential-fixture-value")
    monkeypatch.setattr(app_cli, "load_project", unexpected)
    output = tmp_path / "service"
    assert (
        app_cli.main(
            [
                "serve",
                str(tmp_path / "app"),
                "--env",
                selection,
                "--output",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "credential-fixture-value" not in captured.out + captured.err
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["success", "error", "interrupt"])
def test_main_restores_callers_sigterm_handler(monkeypatch, outcome):
    monkeypatch.syspath_prepend(str(ROOT / "application-cli/src"))
    from pnms_app import cli as app_cli

    previous = signal.getsignal(signal.SIGTERM)

    def caller_handler(*_):
        pass

    signal.signal(signal.SIGTERM, caller_handler)

    def execute(_args):
        assert signal.getsignal(signal.SIGTERM) is not caller_handler
        if outcome == "interrupt":
            signal.raise_signal(signal.SIGTERM)
        if outcome == "error":
            raise ValueError("expected failure")
        return {"status": "done"}

    monkeypatch.setattr(app_cli, "execute", execute)
    try:
        assert (
            app_cli.main(["serve"])
            == {
                "success": 0,
                "error": 2,
                "interrupt": 130,
            }[outcome]
        )
        assert signal.getsignal(signal.SIGTERM) is caller_handler
    finally:
        signal.signal(signal.SIGTERM, previous)
