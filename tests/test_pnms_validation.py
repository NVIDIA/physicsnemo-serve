"""Reject unsupported authored native inputs before any model work."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import cli  # noqa: E402


def invoke(*arguments):
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "pnms"),
            *map(str, arguments),
            "--repository",
            str(ROOT),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def custom_app(tmp_path):
    app = tmp_path / "native-app"
    initialized = invoke("init", app, "--runtime", "cpp", "--template", "custom")
    assert initialized.returncode == 0, initialized.stderr
    return app


def change(app, mutation):
    descriptor = app / "pnms.yaml"
    document = yaml.safe_load(descriptor.read_text())
    if mutation == "target":
        document["build"]["target"] = "another-gpu"
        diagnostic = "build.target"
    elif mutation == "cpp-template":
        document["workflow"]["template"] = "cpp"
        diagnostic = "workflow.template"
    elif mutation == "missing-source-dir":
        document["workflow"]["source_dir"] = "absent-native"
        diagnostic = "workflow.source_dir"
    elif mutation == "missing-cmake":
        (app / "native/CMakeLists.txt").unlink()
        diagnostic = "CMakeLists.txt"
    elif mutation == "missing-target":
        document["workflow"].pop("target")
        diagnostic = "workflow.target"
    elif mutation == "missing-source-setting":
        document["workflow"].pop("source_dir")
        diagnostic = "workflow.source_dir"
    else:
        raise AssertionError(mutation)
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False))
    return diagnostic


INVALID = (
    "target",
    "cpp-template",
    "missing-source-dir",
    "missing-cmake",
    "missing-target",
    "missing-source-setting",
)


@pytest.mark.parametrize("operation", ["check", "plan"])
@pytest.mark.parametrize("mutation", INVALID)
def test_public_static_commands_reject_invalid_native_inputs(
    custom_app, operation, mutation
):
    diagnostic = change(custom_app, mutation)
    arguments = (
        ("check", custom_app, "--config-only")
        if operation == "check"
        else ("build", custom_app, "--plan")
    )
    result = invoke(*arguments)
    assert result.returncode != 0, (
        f"pnms {operation} accepted invalid {mutation}: {result.stdout}"
    )
    assert diagnostic in result.stderr, result.stderr
    assert not (custom_app / ".pnms").exists(), (
        "invalid inputs must fail before attempts/probes"
    )


@pytest.mark.parametrize("mutation", INVALID)
def test_public_build_rejects_invalid_inputs_before_model_stage(
    custom_app, mutation, monkeypatch, capsys
):
    from pnms_app import model

    diagnostic = change(custom_app, mutation)
    reached = []

    def model_stage(*args, **kwargs):
        reached.append("model-stage")
        raise ValueError("model-stage trap: no compiler or framework may run")

    monkeypatch.setattr(model, "build_models", model_stage)
    code = cli.main(["build", str(custom_app), "--repository", str(ROOT)])
    output = capsys.readouterr()
    assert reached == [], f"invalid {mutation} reached the model build stage"
    assert code != 0 and diagnostic in output.err, output.err
    assert not (custom_app / ".pnms").exists()


@pytest.mark.parametrize(
    "alias",
    [
        "absolute",
        "../victim.dat",
        "nested/../../victim.dat",
        r"..\victim.dat",
        r"nested\..\victim.dat",
        r"\victim.dat",
        r"C:\victim.dat",
        r"C:victim.dat",
        r"\\server\share\victim.dat",
    ],
)
def test_unsafe_asset_alias_is_rejected_before_native_stages(
    custom_app, tmp_path, monkeypatch, capsys, alias
):
    from pnms_app import model, native

    victim = tmp_path / "victim.dat"
    victim.write_bytes(b"original outside bytes")
    descriptor = custom_app / "pnms.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["assets"] = {str(victim) if alias == "absolute" else alias: "asset.dat"}
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False))
    (custom_app / "asset.dat").write_bytes(b"replacement asset bytes")
    reached = []

    def unexpected_stage(*args, **kwargs):
        reached.append("native-stage")
        raise ValueError("native stage trap")

    monkeypatch.setattr(model, "build_models", unexpected_stage)
    monkeypatch.setattr(native, "build_workflow", unexpected_stage)
    code = cli.main(["build", str(custom_app), "--repository", str(ROOT)])
    output = capsys.readouterr()

    assert reached == [], f"unsafe asset alias reached a native stage: {alias}"
    assert code != 0 and "asset alias" in output.err, output.err
    assert victim.read_bytes() == b"original outside bytes"
    assert not (custom_app / ".pnms").exists()


@pytest.mark.parametrize("selected", [None, "current-gpu"])
def test_default_and_current_gpu_target_allow_static_commands(custom_app, selected):
    path = custom_app / "pnms.yaml"
    document = yaml.safe_load(path.read_text())
    if selected is None:
        document["build"].pop("target", None)
    else:
        document["build"]["target"] = selected
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    for command in (
        ("check", custom_app, "--config-only"),
        ("build", custom_app, "--plan"),
    ):
        result = invoke(*command)
        assert result.returncode == 0, result.stderr
    assert not (custom_app / ".pnms/builds").exists()


def test_source_free_packaging_does_not_require_custom_sources(
    custom_app, tmp_path, monkeypatch, capsys
):
    from pnms_app import build

    shutil.rmtree(custom_app / "native")
    shutil.rmtree(custom_app / "model")
    receipt = tmp_path / "completed-build.json"
    receipt.write_text('{"status":"complete"}\n')
    called = []

    def package(path, **options):
        called.append((path, options))
        return {"status": "complete"}

    monkeypatch.setattr(build, "package_build", package)
    code = cli.main(
        [
            "build",
            str(custom_app),
            "--from-build",
            str(receipt),
            "--image",
            "release:test",
            "--repository",
            str(ROOT),
        ]
    )
    output = capsys.readouterr()
    assert code == 0, output.err
    assert json.loads(output.out)["status"] == "complete"
    assert called[0][0] == receipt
    assert not (custom_app / ".pnms").exists()
