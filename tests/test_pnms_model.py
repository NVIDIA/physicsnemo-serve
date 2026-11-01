from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import compatibility, model, workflow_cache  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_unit_driver_probe(monkeypatch):
    original = shutil.which
    monkeypatch.setattr(
        model.shutil,
        "which",
        lambda command: None if command == "nvidia-smi" else original(command),
    )


def project(tmp_path, backend="aoti"):
    root = tmp_path / "app"
    source = root / "model"
    source.mkdir(parents=True)
    weights = tmp_path / "trained" / "weights.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"model weights")
    (source / "build_adapter.py").write_text("def create_model(config): pass\n")
    (source / "model-build.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "name": "affine",
                "version": "1",
                "adapter": "build_adapter.py",
                "source": [],
                "checkpoint": str(weights),
                "config": {"scale": 2},
                "backends": [backend],
                "executor": "container",
                "device": "cuda",
            }
        )
    )
    (source / "model-build.lock.json").write_text('{"original": true}')
    return {
        "root": root,
        "name": "app",
        "backend": backend,
        "models": {"primary": {"project": source / "model-build.json"}},
        "build": {},
    }


def recipe_project(tmp_path):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "recipe": "recipe/recipe.json",
                "backends": ["aoti"],
            }
        )
    )
    recipe_root = path.parent / "recipe"
    recipe_root.mkdir()
    recipe_path = recipe_root / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "name": "affine",
                "version": "1",
                "adapter": "export.py",
                "factory": "create_one",
                "cases": "create_cases",
                "input_names": ["input"],
                "output_names": ["output"],
                "dtype": "float32",
                "shape": [4],
                "supported_backends": ["aoti"],
                "default_backend": "aoti",
                "checkpoint": {"path": "weights.pt", "format": "torch-state-dict"},
            },
            indent=4,
        )
        + "\n"
    )
    (recipe_root / "export.py").write_text(
        "def create_one(): pass\ndef create_two(): pass\ndef create_cases(): pass\n"
    )
    (recipe_root / "weights.pt").write_bytes(b"recipe weights")
    return app, recipe_path


def fake_runtime(tmp_path):
    prefix = tmp_path / "sdk"
    executable = prefix / "bin/physicsnemo-infer"
    executable.parent.mkdir(parents=True)
    executable = python_executable(executable, "raise SystemExit(0)\n")
    config = prefix / "lib/cmake/PhysicsNeMoInference/PhysicsNeMoInferenceConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# installed SDK\n")
    header = prefix / "include/physicsnemo/inference/runtime.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("// installed SDK header\n")
    (prefix / "lib/libphysicsnemo_inference.a").write_bytes(b"SDK library")
    for backend in ("aoti", "tensorrt"):
        (prefix / f"lib/libphysicsnemo_inference_{backend}.a").write_bytes(
            b"backend SDK library"
        )
    return executable, prefix


class Runner:
    def __init__(self, *, failure=None, gpus=None):
        self.calls = []
        self.failure = failure
        self.gpus = gpus or [
            {
                "name": "NVIDIA H100",
                "uuid": "GPU-abcd",
                "major": 9,
                "minor": 0,
                "total_memory": 85899345920,
            }
        ]

    def __call__(self, command, **kwargs):
        command = [str(part) for part in command]
        self.calls.append((command, kwargs))
        if any(part.startswith("--query-gpu=") for part in command):
            return subprocess.CompletedProcess(command, 0, "570.195.03, Disabled\n", "")
        if "-c" in command and command[command.index("-c") + 1] == compatibility.PROBE:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"schema_version": 1, "system_libraries": {}}),
                "",
            )
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "python": "3.12.0",
                        "torch": "2.8.0+cu128",
                        "cuda": "12.8",
                        "tensorrt": "10.14.1",
                        "torch_root": "/torch",
                        "libc": ["glibc", "2.35"],
                        "cxx11_abi": True,
                        "gpus": self.gpus,
                    }
                ),
                "",
            )
        operation = "check" if "check" in command else "build"
        if self.failure:
            return subprocess.CompletedProcess(
                command,
                1,
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "stage": "export",
                        "diagnostics": [
                            {"code": "TEST_FAILURE", "message": self.failure}
                        ],
                    }
                ),
                "original compiler diagnostic\n",
            )
        if "--output" in command:
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True)
            package = destination / "model"
            package.mkdir()
            backend = (
                command[command.index("--backend") + 1]
                if "--backend" in command
                else "aoti"
            )
            variant = package / "backends" / backend
            variant.mkdir(parents=True)
            (variant / "model.json").write_text("{}")
            (package / "model-release.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "variants": {backend: {"package": f"backends/{backend}"}},
                    }
                )
            )
        else:
            package = None
        status = (
            ("configuration-ok" if "--config-only" in command else "checked")
            if operation == "check"
            else "complete"
        )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "schema_version": 1,
                    "status": status,
                    "artifacts": {"package": str(package)} if package else {},
                }
            ),
            "builder diagnostics\n",
        )


def test_build_captures_model_inputs_preserves_original_and_returns_package(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    app = project(tmp_path)
    source = app["models"]["primary"]["project"]
    original = {p: p.read_bytes() for p in source.parent.iterdir()}
    runtime, prefix = fake_runtime(tmp_path)
    runner = Runner()
    result = model.build_models(
        app,
        tmp_path / "attempt",
        repository=ROOT,
        python=Path(sys.executable),
        runtime=runtime,
        sdk_prefix=prefix,
        runner=runner,
    )
    assert "primary" in result["models"]
    built = result["models"]["primary"]
    assert Path(built["package"]).is_dir()
    assert (
        Path(built["package"])
        .relative_to(Path(built["output_root"]) / "model")
        .as_posix()
        == "backends/aoti"
    )
    assert built["backend"] == "aoti"
    calls = [(cmd, kw) for cmd, kw in runner.calls if "build" in cmd]
    assert len(calls) == 1
    cmd, kw = calls[0]
    captured = Path(cmd[cmd.index("build") + 1])
    assert captured != source
    captured_config = json.loads(captured.read_text())
    copied_weights = Path(captured_config["checkpoint"])
    assert copied_weights.is_relative_to(tmp_path / "attempt")
    assert copied_weights.read_bytes() == b"model weights"
    assert cmd[cmd.index("--executor") + 1] == "local"
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert all("docker" not in cmd for cmd, _ in runner.calls)
    assert original == {p: p.read_bytes() for p in source.parent.iterdir()}


def test_build_failure_retains_original_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    with pytest.raises(model.ModelError, match="unsupported operator"):
        model.build_models(
            project(tmp_path),
            tmp_path / "attempt",
            repository=ROOT,
            runtime=runtime,
            sdk_prefix=prefix,
            runner=Runner(failure="unsupported operator"),
        )
    assert any(
        "original compiler diagnostic" in path.read_text()
        for path in (tmp_path / "attempt").rglob("stderr.log")
    )


def test_config_only_does_not_probe_cuda_or_require_runtime(tmp_path):
    runner = Runner()
    result = model.check_models(
        project(tmp_path),
        tmp_path / "checks",
        repository=ROOT,
        config_only=True,
        runner=runner,
    )
    assert result["models"]["primary"]["result"]["status"] == "configuration-ok"
    assert not any(
        "-c" in command or "docker" in command for command, _ in runner.calls
    )


@pytest.mark.parametrize(
    ("format_version", "invalid_input"),
    [(1, "project"), (2, "project"), (1, "recipe")],
    ids=["format-1-project", "format-2-project", "format-1-recipe"],
)
@pytest.mark.parametrize(
    "invalid_json",
    ["top-duplicate", "nested-duplicate", "NaN", "Infinity", "-Infinity"],
)
def test_public_config_check_rejects_invalid_model_json_before_capture(
    tmp_path, format_version, invalid_input, invalid_json
):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    document = json.loads(path.read_text())
    document.update(
        format_version=format_version,
        executor="local",
        device="cpu",
        config="config.json",
        assets={"auxiliary": "asset.dat"},
    )
    (path.parent / "config.json").write_text('{"scale": 2}')
    (path.parent / "asset.dat").write_bytes(b"project asset")
    (path.parent / "build_adapter.py").write_text(
        "def create_model(config, assets): pass\ndef create_cases(config, assets): pass\n"
    )
    if format_version == 1:
        for field in ("name", "version", "adapter", "source"):
            document.pop(field)
        document["recipe"] = "recipe/recipe.json"
        recipe_root = path.parent / "recipe"
        recipe_root.mkdir()
        recipe = {
            "format_version": 2,
            "name": "configured",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "input_names": ["input"],
            "output_names": ["output"],
            "dtype": "float32",
            "shape": [4],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"path": "weights.pt", "format": "torch-state-dict"},
            "assets": {"auxiliary": {"path": "asset.dat"}},
        }
        (recipe_root / "recipe.json").write_text(json.dumps(recipe))
        (recipe_root / "export.py").write_text(
            "raise AssertionError('configuration checks must not import model code')\n"
        )
        (recipe_root / "config.json").write_text("{}")
        (recipe_root / "weights.pt").write_bytes(b"recipe weights")
        (recipe_root / "asset.dat").write_bytes(b"recipe asset")
    path.write_text(json.dumps(document))
    invalid_path = path if invalid_input == "project" else recipe_root / "recipe.json"
    invalid_document = json.loads(invalid_path.read_text())
    if invalid_json == "top-duplicate":
        fragment = (
            '"config": {"path": "config.json"}'
            if invalid_input == "recipe"
            else '"config": "config.json"'
        )
        diagnostic = "duplicate JSON field: config"
    elif invalid_json == "nested-duplicate":
        if invalid_input == "recipe":
            invalid_document.pop("config")
            fragment = '"config": {"path": "config.json", "path": "config.json"}'
            diagnostic = "duplicate JSON field: path"
        else:
            fragment = '"profiles": {"unused": {"device": "cpu", "device": "cpu"}}'
            diagnostic = "duplicate JSON field: device"
    else:
        if invalid_input == "recipe":
            invalid_document.pop("shape")
            fragment = f'"shape": [{invalid_json}]'
        else:
            fragment = f'"profiles": {{"unused": {{"device": {invalid_json}}}}}'
        diagnostic = f"nonfinite JSON value: {invalid_json}"
    invalid_path.write_text(json.dumps(invalid_document)[:-1] + ", " + fragment + "}\n")
    (app["root"] / "pnms.yaml").write_text(
        "schema_version: 1\nname: app\nruntime: cpp\n"
        "models:\n  primary:\n    project: model/model-build.json\n"
    )
    original = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if invalid_input == "project":
        builder = subprocess.run(
            [
                sys.executable,
                str(ROOT / "native-inference/pnms-model-builder"),
                "check",
                str(path),
                "--executor",
                "local",
                "--config-only",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert builder.returncode == 2, builder.stderr or builder.stdout
        assert (diagnostic if "duplicate" in invalid_json else "non-finite") in (
            builder.stderr + builder.stdout
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pnms"),
            "check",
            str(app["root"]),
            "--config-only",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert original == {p: p.read_bytes() for p in original}
    assert completed.returncode == 2, completed.stderr or completed.stdout
    assert diagnostic in completed.stderr
    assert not any(p.is_file() for p in (app["root"] / ".pnms/checks").rglob("*"))


@pytest.mark.parametrize(
    "asset_name",
    ["../" * 9 + "victim", "..\\" * 9 + "victim", "/victim", "C:\\victim"],
    ids=["posix-traversal", "windows-traversal", "posix-absolute", "windows-absolute"],
)
def test_public_config_check_rejects_recipe_asset_names_before_copy(
    tmp_path, asset_name
):
    app = tmp_path / "app"
    recipe_root = app / "model/recipe"
    recipe_root.mkdir(parents=True)
    (app / "pnms.yaml").write_text(
        "schema_version: 1\nname: app\nruntime: cpp\n"
        "models:\n  primary:\n    project: model/model-build.json\n"
    )
    (recipe_root.parent / "model-build.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "recipe": "recipe/recipe.json",
                "backends": ["aoti"],
                "executor": "local",
                "device": "cpu",
            }
        )
    )
    recipe = {
        "format_version": 2,
        "name": "configured",
        "version": "0.1.0",
        "adapter": "export.py",
        "factory": "create_model",
        "cases": "create_cases",
        "input_names": ["input"],
        "output_names": ["output"],
        "dtype": "float32",
        "shape": [4],
        "supported_backends": ["aoti"],
        "default_backend": "aoti",
        "config": {"path": "config.json"},
        "checkpoint": {"path": "weights.pt", "format": "torch-state-dict"},
        "assets": {asset_name: {"path": "value.dat"}},
    }
    (recipe_root / "recipe.json").write_text(json.dumps(recipe))
    (recipe_root / "export.py").write_text(
        "raise AssertionError('configuration checks must not import model code')\n"
    )
    (recipe_root / "config.json").write_text("{}")
    (recipe_root / "weights.pt").write_bytes(b"opaque weights")
    (recipe_root / "value.dat").write_bytes(b"REPLACED")
    victim = app / "victim/value.dat"
    victim.parent.mkdir()
    victim.write_bytes(b"ORIGINAL")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "pnms"), "check", str(app), "--config-only"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert victim.read_bytes() == b"ORIGINAL"
    assert completed.returncode == 2, completed.stderr or completed.stdout
    assert "Invalid model recipe asset name" in completed.stderr
    assert not list((app / ".pnms/checks").rglob("export.py"))


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
@pytest.mark.parametrize("default_profile", [False, True], ids=["explicit", "default"])
def test_public_config_check_selects_backend_from_multi_backend_profile(
    tmp_path, backend, default_profile
):
    app = project(tmp_path, backend)
    path = app["models"]["primary"]["project"]
    (path.parent / "build_adapter.py").write_text(
        "def create_model(config, assets): pass\ndef create_cases(config, assets): pass\n"
    )
    document = json.loads(path.read_text())
    dual = {
        "backends": ["aoti", "tensorrt"],
        "aoti_profile": "aten-boundary-exact-v3",
        "aoti_options": {"shape_padding": False},
        "tensorrt_profile": (
            "baseline" if backend == "aoti" else "provider-default-exact-v1"
        ),
        "required_gpu_arch": "sm90",
    }
    document["profiles"] = {"dual": dual, "other": {"backends": ["aoti"]}}
    document["default_profile"] = "dual" if default_profile else "other"
    path.write_text(json.dumps(document, indent=2) + "\n")
    original = {p: p.read_bytes() for p in path.parent.iterdir()}
    profile_setting = "" if default_profile else "    profile: dual\n"
    (app["root"] / "pnms.yaml").write_text(
        "schema_version: 1\nname: app\nruntime: cpp\n"
        "models:\n  primary:\n    project: model/model-build.json\n"
        f"{profile_setting}build:\n  backend: {backend}\n"
    )
    builder = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native-inference/pnms-model-builder"),
            "check",
            str(path),
            "--profile",
            "dual",
            "--backend",
            backend,
            "--executor",
            "local",
            "--config-only",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert builder.returncode == 0, builder.stderr or builder.stdout
    expected = json.loads(builder.stdout)["effective_config"]
    assert expected["backends"] == [backend]

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pnms"),
            "check",
            str(app["root"]),
            "--config-only",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert original == {p: p.read_bytes() for p in path.parent.iterdir()}
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads(completed.stdout)["models"]["primary"]
    captured = Path(record["project_capture"])
    staged = json.loads(captured.read_text())
    assert staged["profiles"] == {
        "dual": {**dual, "backends": [backend]},
        "other": document["profiles"]["other"],
    }
    assert (
        json.loads((captured.parent / "original-model-build.json").read_text())
        == document
    )
    assert (
        captured.parent / "original-model-build.lock.json"
    ).read_bytes() == original[path.parent / "model-build.lock.json"]
    assert record["result"]["status"] == "configuration-ok"
    assert record["result"]["profile"] == "dual"
    for field in (
        "backends",
        "aoti_profile",
        "aoti_options",
        "tensorrt_profile",
        "required_gpu_arch",
    ):
        assert record["result"]["effective_config"][field] == expected[field]


@pytest.mark.parametrize(
    "backends",
    ["aoti", None, ["aoti", "aoti"], ["aoti", "unknown"]],
    ids=["string", "null", "duplicate", "unknown"],
)
def test_public_config_check_rejects_malformed_profile_backends(tmp_path, backends):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    (path.parent / "build_adapter.py").write_text(
        "def create_model(config, assets): pass\ndef create_cases(config, assets): pass\n"
    )
    document = json.loads(path.read_text())
    document.update(
        profiles={"invalid": {"backends": backends}}, default_profile="invalid"
    )
    path.write_text(json.dumps(document))
    original = {p: p.read_bytes() for p in path.parent.iterdir()}
    (app["root"] / "pnms.yaml").write_text(
        "schema_version: 1\nname: app\nruntime: cpp\n"
        "models:\n  primary:\n    project: model/model-build.json\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pnms"),
            "check",
            str(app["root"]),
            "--config-only",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert original == {p: p.read_bytes() for p in path.parent.iterdir()}
    assert completed.returncode == 2, completed.stderr or completed.stdout
    assert "backend" in completed.stderr and "profile" in completed.stderr
    assert not list((app["root"] / ".pnms/checks").rglob("build_adapter.py"))


def test_selected_profile_conflicting_backend_fails_before_execution(tmp_path):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    document = json.loads(path.read_text())
    document["profiles"] = {"trt": {"backends": ["tensorrt"]}}
    path.write_text(json.dumps(document))
    app["models"]["primary"]["profile"] = "trt"
    runner = Runner()
    with pytest.raises(model.ModelError, match="backend.*profile"):
        model.check_models(
            app, tmp_path / "checks", repository=ROOT, config_only=True, runner=runner
        )
    assert not runner.calls


def test_gpu_selection_respects_allocated_logical_index(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    runtime, prefix = fake_runtime(tmp_path)
    gpus = [
        {
            "name": "H100",
            "uuid": f"GPU-{i}",
            "major": 9,
            "minor": 0,
            "total_memory": 80_000_000_000,
        }
        for i in range(2)
    ]
    runner = Runner(gpus=gpus)
    result = model.build_models(
        project(tmp_path),
        tmp_path / "attempt",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        gpu="1",
        runner=runner,
    )
    assert result["environment"]["gpu"]["binding"] == "7"
    assert result["environment"]["gpu"]["uuid"] == "GPU-1"
    command, options = next((c, o) for c, o in runner.calls if "build" in c)
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert command[command.index("--device") + 1] == "cuda:0"


def test_multiple_visible_gpus_require_explicit_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    runtime, prefix = fake_runtime(tmp_path)
    gpu = {"name": "H100", "uuid": "GPU-a", "major": 9, "minor": 0, "total_memory": 80}
    runner = Runner(gpus=[gpu, dict(gpu, uuid="GPU-b")])
    with pytest.raises(model.ModelError, match="Several GPUs"):
        model.build_models(
            project(tmp_path),
            tmp_path / "attempt",
            repository=ROOT,
            runtime=runtime,
            sdk_prefix=prefix,
            runner=runner,
        )
    assert not any("build" in command for command, _ in runner.calls)


def test_cache_verifies_outputs_and_model_changes_recompile(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    cache = tmp_path / "cache"
    runner = Runner()
    first = model.build_models(
        app,
        tmp_path / "first",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    second = model.build_models(
        app,
        tmp_path / "second",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    assert second["models"]["primary"]["reused"] is True
    assert len([c for c, _ in runner.calls if "build" in c]) == 1
    compiled = next(cache.glob("*/output/model/model-release.json"))
    compiled.write_text("corrupted")
    third = model.build_models(
        app,
        tmp_path / "third",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    assert third["models"]["primary"]["reused"] is False
    weights = tmp_path / "trained/weights.pt"
    weights.write_bytes(b"updated weights")
    fourth = model.build_models(
        app,
        tmp_path / "fourth",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    assert (
        fourth["models"]["primary"]["identity"]
        != first["models"]["primary"]["identity"]
    )


def test_recipe_only_factory_change_invalidates_model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    app, recipe_path = recipe_project(tmp_path)
    original = {p: p.read_bytes() for p in app["root"].rglob("*") if p.is_file()}
    runtime, prefix = fake_runtime(tmp_path)
    runner = Runner()
    options = dict(
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=tmp_path / "cache",
        runner=runner,
    )
    first = model.build_models(app, tmp_path / "first", **options)["models"]["primary"]
    same = model.build_models(app, tmp_path / "same", **options)["models"]["primary"]
    assert first["reused"] is False
    assert same["reused"] is True
    assert same["identity"] == first["identity"]
    assert len([command for command, _ in runner.calls if "build" in command]) == 1
    assert original == {
        p: p.read_bytes() for p in app["root"].rglob("*") if p.is_file()
    }

    recipe = json.loads(recipe_path.read_text())
    recipe["factory"] = "create_two"
    recipe_path.write_text(json.dumps(recipe, indent=4) + "\n")
    original[recipe_path] = recipe_path.read_bytes()
    changed = model.build_models(app, tmp_path / "changed", **options)["models"][
        "primary"
    ]
    assert changed["identity"] != first["identity"]
    assert changed["reused"] is False
    assert len([command for command, _ in runner.calls if "build" in command]) == 2
    capture = json.loads(
        (Path(changed["project_capture"]).parent / "capture.json").read_text()
    )
    assert capture["inputs"][str(recipe_path)] == {
        "sha256": hashlib.sha256(original[recipe_path]).hexdigest(),
        "size_bytes": len(original[recipe_path]),
    }
    assert original == {
        p: p.read_bytes() for p in app["root"].rglob("*") if p.is_file()
    }


def test_recipe_changed_during_capture_preserves_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    app, recipe_path = recipe_project(tmp_path)
    original = {p: p.read_bytes() for p in app["root"].rglob("*") if p.is_file()}
    runtime, prefix = fake_runtime(tmp_path)
    runner = Runner()
    cache = tmp_path / "cache"
    options = dict(
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    model.build_models(app, tmp_path / "first", **options)
    published = {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    copyfile = model.shutil.copyfile

    def change_recipe(source, target, **kwargs):
        result = copyfile(source, target, **kwargs)
        if Path(source) == recipe_path.parent / "export.py":
            recipe = json.loads(recipe_path.read_text())
            recipe["factory"] = "create_two"
            recipe_path.write_text(json.dumps(recipe, indent=4) + "\n")
        return result

    monkeypatch.setattr(model.shutil, "copyfile", change_recipe)
    with pytest.raises(model.ModelError, match="recipe.*changed during capture"):
        model.build_models(app, tmp_path / "changed", **options)
    assert len([command for command, _ in runner.calls if "build" in command]) == 1
    assert published == {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    original[recipe_path] = recipe_path.read_bytes()
    assert json.loads(recipe_path.read_text())["factory"] == "create_two"
    assert original == {
        p: p.read_bytes() for p in app["root"].rglob("*") if p.is_file()
    }


def test_model_cache_preserves_compilation_source_across_service_revisions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("PNMS_SOURCE_SHA", "a" * 40)
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    cache = tmp_path / "cache"
    runner = Runner()
    options = dict(
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    first = model.build_models(app, tmp_path / "first", **options)["models"]["primary"]
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    second = model.build_models(app, tmp_path / "second", **options)["models"][
        "primary"
    ]
    assert second["reused"] is True
    assert first["identity"] == second["identity"]
    assert len([command for command, _ in runner.calls if "build" in command]) == 1
    assert first.get("compilation_source_sha") == "a" * 40
    assert second.get("compilation_source_sha") == "a" * 40
    for built in (first, second):
        provenance = json.loads(
            (Path(built["output_root"]) / "compilation-source.json").read_text()
        )
        assert provenance == {"schema_version": 1, "source_sha": "a" * 40}


@pytest.mark.parametrize("provenance", ["missing", "invalid", "unbound"])
def test_unproven_model_cache_rebuilds_instead_of_inventing_compilation_source(
    tmp_path, monkeypatch, provenance
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("PNMS_SOURCE_SHA", "a" * 40)
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    runner = Runner()
    cache = tmp_path / "cache"
    options = dict(
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=cache,
        runner=runner,
    )
    model.build_models(app, tmp_path / "first", **options)
    seal_path = next(cache.glob("*/seal.json"))
    seal = json.loads(seal_path.read_text())
    if provenance == "missing":
        seal.pop("compilation_source_sha", None)
    else:
        seal["compilation_source_sha"] = (
            "short" if provenance == "invalid" else "c" * 40
        )
    seal_path.write_text(json.dumps(seal))
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    rebuilt = model.build_models(app, tmp_path / "second", **options)["models"][
        "primary"
    ]
    assert rebuilt["reused"] is False
    assert len([command for command, _ in runner.calls if "build" in command]) == 2
    assert rebuilt.get("compilation_source_sha") == "b" * 40
    reused = model.build_models(app, tmp_path / "third", **options)["models"]["primary"]
    assert reused["reused"] is True
    assert reused.get("compilation_source_sha") == "b" * 40


def test_container_limits_gpu_and_uses_unchanged_process_executor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(model.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(model.os, "getgid", lambda: 5678, raising=False)
    app = project(tmp_path)
    app["build"]["builder_image"] = "registry.example/builder@sha256:" + "a" * 64
    runner = Runner()
    result = model.build_models(
        app,
        tmp_path / "attempt",
        repository=ROOT,
        environment="container",
        gpu="1",
        runner=runner,
    )
    command = next(command for command, _ in runner.calls if "docker" in command)
    assert command[command.index("--gpus") + 1] == "device=5"
    assert command[command.index("--user") + 1] == "1234:5678"
    assert command[command.index("--executor") + 1] == "local"
    assert "all" not in command
    assert result["environment"]["kind"] == "container"


def test_original_relative_external_inputs_are_resolved_from_project_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    document = json.loads(path.read_text())
    document["checkpoint"] = "../../trained/weights.pt"
    path.write_text(json.dumps(document))
    runtime, prefix = fake_runtime(tmp_path)
    result = model.build_models(
        app,
        tmp_path / "attempt",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=Runner(),
    )
    capture = Path(result["models"]["primary"]["project_capture"])
    assert (
        Path(json.loads(capture.read_text())["checkpoint"]).read_bytes()
        == b"model weights"
    )


def test_sdk_prefix_requires_headers_and_libraries_not_just_cmake_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    (prefix / "include/physicsnemo/inference/runtime.hpp").unlink()
    with pytest.raises(model.ModelError, match="headers|libraries"):
        model.build_models(
            project(tmp_path),
            tmp_path / "attempt",
            repository=ROOT,
            runtime=runtime,
            sdk_prefix=prefix,
            runner=Runner(),
        )


def test_source_symlink_parent_cannot_escape_capture(tmp_path):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    external = tmp_path / "outside"
    external.mkdir()
    (external / "adapter.py").write_text("untrusted outside source")
    (path.parent / "linked").symlink_to(external, target_is_directory=True)
    document = json.loads(path.read_text())
    document["adapter"] = "linked/adapter.py"
    path.write_text(json.dumps(document))
    with pytest.raises(model.ModelError, match="symlink"):
        model.check_models(
            app, tmp_path / "checks", repository=ROOT, config_only=True, runner=Runner()
        )


def test_profile_inputs_with_same_filename_do_not_overwrite_inherited_inputs(tmp_path):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    default_lock = tmp_path / "default/toolchain.json"
    profile_lock = tmp_path / "profile/toolchain.json"
    for lock, value in [
        (default_lock, "base lock"),
        (profile_lock, "other profile lock"),
    ]:
        lock.parent.mkdir()
        lock.write_text(value)
    document = json.loads(path.read_text())
    document["toolchain_lock"] = str(default_lock)
    document["profiles"] = {"other": {"toolchain_lock": str(profile_lock)}}
    path.write_text(json.dumps(document))
    result = model.check_models(
        app, tmp_path / "checks", repository=ROOT, config_only=True, runner=Runner()
    )
    captured = json.loads(
        Path(result["models"]["primary"]["project_capture"]).read_text()
    )
    assert Path(captured["toolchain_lock"]).read_text() == "base lock"
    assert (
        Path(captured["profiles"]["other"]["toolchain_lock"]).read_text()
        == "other profile lock"
    )


def test_invalid_asset_name_cannot_write_outside_attempt(tmp_path):
    app = project(tmp_path)
    path = app["models"]["primary"]["project"]
    document = json.loads(path.read_text())
    document["assets"] = {"../../../../escape": document["checkpoint"]}
    path.write_text(json.dumps(document))
    with pytest.raises(model.ModelError, match="asset name"):
        model.check_models(
            app, tmp_path / "checks", repository=ROOT, config_only=True, runner=Runner()
        )


def test_installed_sdk_relative_library_symlinks_are_fingerprinted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    library = prefix / "lib/libphysicsnemo_inference.so.1"
    library.write_bytes(b"shared SDK")
    (prefix / "lib/libphysicsnemo_inference.so").symlink_to(library.name)
    result = model.prepare_environment(
        project(tmp_path),
        tmp_path / "environment",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=Runner(),
    )
    inventory = result["stack"]["sdk_inventory"]
    assert inventory["lib/libphysicsnemo_inference.so"]["symlink"] == library.name


@pytest.mark.parametrize("cache_state", ["hit", "concurrent", "corrupt", "unresolved"])
def test_default_sdk_is_cached_independently_of_workflow_build(
    tmp_path, monkeypatch, cache_state
):
    # The fake builder consumes none of the inherited compiler search paths.
    for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "INCLUDE", "SDKROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    # This fake SDK consumes only the immutable fixture copied below.
    def configured_inputs(*args, **kwargs):
        if cache_state == "unresolved":
            raise ValueError("missing CMake discovery metadata")
        return {"fixture": "immutable"}

    monkeypatch.setattr(workflow_cache, "configured_inputs", configured_inputs)
    app = project(tmp_path)
    _, fixture_sdk = fake_runtime(tmp_path)

    class SDKRunner(Runner):
        prefix = None
        on_compile = None

        def __call__(self, command, **kwargs):
            command = [str(part) for part in command]
            if command[0] == "cmake" and "-S" in command:
                self.prefix = Path(
                    next(
                        part.split("=", 1)[1]
                        for part in command
                        if part.startswith("-DCMAKE_INSTALL_PREFIX=")
                    )
                )
                self.calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, "configured", "")
            if command[0] == "cmake" and "--build" in command:
                self.calls.append((command, kwargs))
                if self.on_compile:
                    self.on_compile()
                return subprocess.CompletedProcess(command, 0, "compiled", "")
            if command[0] == "cmake" and "--install" in command:
                shutil.copytree(fixture_sdk, self.prefix)
                self.calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, "installed", "")
            return super().__call__(command, **kwargs)

    runner = SDKRunner()
    winner_runner = SDKRunner()
    winner = {}

    def publish_winner():
        # The late builder has already missed and selected its staging prefix.
        result = model.prepare_environment(
            app, tmp_path / "winner", repository=ROOT, runner=winner_runner
        )
        cache = Path(result["sdk_prefix"]).parent
        assert runner.prefix.parent != cache
        assert runner.prefix.parent.name.startswith(".pending-")
        winner.update(result=result, cache=cache)

    rename = Path.rename

    def preserve_returned_paths(source, target):
        result = rename(source, target)
        if winner:
            prefix = Path(winner["result"]["sdk_prefix"])
            assert Path(winner["result"]["runtime"]).is_file()
            assert (prefix / "include/physicsnemo/inference/runtime.hpp").is_file()
        return result

    if cache_state == "concurrent":
        runner.on_compile = publish_winner
        monkeypatch.setattr(Path, "rename", preserve_returned_paths)
    first = model.prepare_environment(
        app, tmp_path / "first", repository=ROOT, runner=runner
    )
    prefix = Path(first["sdk_prefix"])
    cache = prefix.parent
    sealed_inventory = json.loads((cache / "seal.json").read_text())["inventory"]
    assert model._inventory(prefix, sdk=True) == sealed_inventory
    if cache_state == "corrupt":
        (prefix / "include/physicsnemo/inference/runtime.hpp").write_text("corrupt")
        corrupted_inventory = model._inventory(prefix, sdk=True)
    second = model.prepare_environment(
        app, tmp_path / "second", repository=ROOT, runner=runner
    )
    assert (first["sdk_prefix"] == second["sdk_prefix"]) == (
        cache_state != "unresolved"
    )
    assert prefix.is_relative_to(app["root"] / ".pnms/cache/environments")
    assert model._inventory(prefix, sdk=True) == sealed_inventory
    quarantined = list(cache.parent.glob(cache.name + ".invalid.*"))
    if cache_state == "corrupt":
        assert len(quarantined) == 1
        assert model._inventory(quarantined[0] / "sdk", sdk=True) == corrupted_inventory
    else:
        assert not quarantined
    if winner:
        assert first["sdk_prefix"] == winner["result"]["sdk_prefix"]
        assert not runner.prefix.parent.exists()
    calls = runner.calls + winner_runner.calls
    assert len([cmd for cmd, _ in calls if cmd[0] == "cmake" and "-S" in cmd]) == (
        3 if cache_state == "concurrent" else 2
    )
    assert len([cmd for cmd, _ in calls if "--build" in cmd]) == (
        1 if cache_state == "hit" else 2
    )
    for operation in ("--build", "--install"):
        command = next(cmd for cmd, _ in runner.calls if operation in cmd)
        assert command[command.index("--config") + 1] == "Release"


@pytest.fixture
def cpu_sdk(tmp_path, monkeypatch):
    if not shutil.which("cmake"):
        pytest.skip("CMake is required for native CPU SDK cache checks")
    compilers = (
        ("cl", "c++", "clang++", "g++")
        if sys.platform == "win32"
        else (
            "c++",
            "clang++",
            "g++",
        )
    )
    compiler = next(
        (name for name in compilers if shutil.which(name)),
        None,
    )
    if not compiler:
        pytest.skip("A C++ compiler is required for native CPU SDK cache checks")
    if sys.platform == "win32" and not shutil.which("ninja"):
        pytest.skip("Ninja is required to select the C++ compiler on Windows")
    for name in workflow_cache._BUILD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CMAKE_GENERATOR_INSTANCE", raising=False)
    monkeypatch.setenv("CXX", compiler)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv(
        "CMAKE_GENERATOR", "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    )
    repository = tmp_path / "repository"
    source = repository / "native-inference/cpp-runtime"
    for directory in ("src", "app", "include/physicsnemo/inference", "cmake"):
        (source / directory).mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CpuSdkCache LANGUAGES CXX)\n"
        "add_library(physicsnemo_inference STATIC src/runtime.cpp)\n"
        'set_target_properties(physicsnemo_inference PROPERTIES PREFIX "lib")\n'
        "target_include_directories(physicsnemo_inference PUBLIC include)\n"
        "add_executable(physicsnemo-infer app/main.cpp)\n"
        "target_link_libraries(physicsnemo-infer PRIVATE physicsnemo_inference)\n"
        "install(TARGETS physicsnemo_inference physicsnemo-infer\n"
        "  ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)\n"
        "install(DIRECTORY include/ DESTINATION include)\n"
        "install(FILES cmake/PhysicsNeMoInferenceConfig.cmake\n"
        "  DESTINATION lib/cmake/PhysicsNeMoInference)\n"
    )
    (source / "include/physicsnemo/inference/runtime.hpp").write_text(
        "int sdk_setting();\n"
    )
    (source / "src/runtime.cpp").write_text(
        "int sdk_setting() { return PNMS_SDK_SETTING; }\n"
    )
    (source / "app/main.cpp").write_text(
        '#include "physicsnemo/inference/runtime.hpp"\n'
        "#include <cstdio>\n"
        'int main() { std::printf("%d\\n", sdk_setting()); }\n'
    )
    (source / "cmake/PhysicsNeMoInferenceConfig.cmake").write_text("# CPU SDK\n")
    probes = Runner()
    metadata_probes = []

    def run(command, **kwargs):
        command = [str(part) for part in command]
        if "-c" in command and command[command.index("-c") + 1] in (
            model._PROBE,
            compatibility.PROBE,
        ):
            return probes(command, **kwargs)
        probes.calls.append((command, kwargs))
        if (
            compiler == "cl"
            and command[-1:] == ["--version"]
            and (
                command[0] == kwargs.get("env", {}).get("CXX")
                or command[:-1]
                == workflow_cache._command_tokens(kwargs.get("env", {}).get("CXX", ""))
            )
        ):
            # MSVC has no --version. Only adapt metadata; builds run unchanged.
            observed = subprocess.run([command[0]], **kwargs)
            banner = (observed.stdout or "") + (observed.stderr or "")
            assert observed.returncode in (0, 2) and "Microsoft" in banner, banner
            metadata_probes.append(
                {
                    "requested": command,
                    "executed": [command[0]],
                    "returncode": observed.returncode,
                    "banner": banner,
                }
            )
            (tmp_path / "msvc-compiler-metadata.json").write_text(
                json.dumps(metadata_probes, indent=2) + "\n"
            )
            return subprocess.CompletedProcess(command, 0, banner, observed.stderr)
        return subprocess.run(command, **kwargs)

    return SimpleNamespace(
        repository=repository,
        project=project(tmp_path),
        runner=run,
        calls=probes.calls,
    )


def test_cpu_sdk_fixture_selects_msvc_and_adapts_only_version_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CMAKE_GENERATOR_INSTANCE", r"C:\BuildTools2022")
    monkeypatch.setenv("CMAKE_GENERATOR_PLATFORM", "x64")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: {"cmake": "cmake.exe", "ninja": "ninja.exe", "cl": "cl.exe"}.get(
            name
        ),
    )
    try:
        configured = cpu_sdk.__wrapped__(tmp_path, monkeypatch)
    except pytest.skip.Exception as exc:
        pytest.fail(f"The CPU SDK regressions must run with MSVC and Ninja: {exc}")
    assert os.environ["CXX"] == "cl"
    assert os.environ["CMAKE_GENERATOR"] == "Ninja"
    assert "CMAKE_GENERATOR_INSTANCE" not in os.environ
    assert "CMAKE_GENERATOR_PLATFORM" not in os.environ
    calls = []
    banner = "Microsoft (R) C/C++ Optimizing Compiler Version 19.44 for x64\n"

    def metadata(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 2, "", banner)

    monkeypatch.setattr(subprocess, "run", metadata)
    completed = configured.runner(["cl", "--version"], env={"CXX": "cl"})
    assert calls == [["cl"]]
    assert completed.returncode == 0
    assert completed.stdout == banner


@pytest.mark.parametrize(
    ("platform_name", "compiler", "expected"),
    [
        (
            "posix",
            "'/Compiler Tools/c++' '-DPNMS_LABEL=two words'",
            ["/Compiler Tools/c++", "-DPNMS_LABEL=two words"],
        ),
        (
            "nt",
            r'"C:\Program Files\Compiler\cl.exe" /DPNMS_SETTING=1',
            [r"C:\Program Files\Compiler\cl.exe", "/DPNMS_SETTING=1"],
        ),
        (
            "nt",
            r"C:\Program Files\Compiler\cl.exe",
            [r"C:\Program Files\Compiler\cl.exe"],
        ),
    ],
)
def test_cxx_version_probe_preserves_compiler_path_and_argument_quoting(
    tmp_path, monkeypatch, platform_name, compiler, expected
):
    monkeypatch.setenv("CXX", compiler)
    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name=platform_name))
    monkeypatch.setattr(
        model.shutil,
        "which",
        lambda name: compiler if name == compiler == expected[0] else None,
    )
    probes = Runner()

    class CompilerProbeReached(Exception):
        pass

    def run(command, **kwargs):
        if command[-1:] == ["--version"] and command[0] not in (
            "cmake",
            kwargs.get("env", {}).get("CUDACXX"),
        ):
            assert command == [*expected, "--version"]
            raise CompilerProbeReached
        return probes(command, **kwargs)

    with pytest.raises(CompilerProbeReached):
        model.prepare_environment(
            project(tmp_path), tmp_path / "attempt", repository=ROOT, runner=run
        )


@pytest.mark.parametrize("discovery", ["prefix", "system-prefix"])
@pytest.mark.parametrize("changed_input", ["header", "config"])
def test_discovered_package_inputs_invalidate_default_sdk_cache(
    tmp_path, monkeypatch, cpu_sdk, discovery, changed_input
):
    source = cpu_sdk.repository / "native-inference/cpp-runtime"
    # PATH-derived prefixes precede CMAKE_SYSTEM_PREFIX_PATH in package discovery.
    installed = tmp_path / "installed-package"
    (installed / "bin").mkdir(parents=True)
    installed_config = installed / "lib/cmake/nlohmann_json"
    installed_config.mkdir(parents=True)
    (installed_config / "nlohmann_jsonConfig.cmake").write_text(
        "add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)\n"
    )
    monkeypatch.setenv("PATH", str(installed / "bin") + os.pathsep + os.environ["PATH"])
    package = tmp_path / "external-package"
    header = package / "include/external.setting"
    header.parent.mkdir(parents=True)
    header.write_text("#define PNMS_SDK_SETTING 1\n")
    config = package / "lib/cmake/PnmsSdkCacheFixture/PnmsSdkCacheFixtureConfig.cmake"
    config.parent.mkdir(parents=True)

    def configure_package(value):
        config.write_text(
            "add_library(PnmsSdkCacheFixture::setting INTERFACE IMPORTED)\n"
            "set_target_properties(PnmsSdkCacheFixture::setting PROPERTIES\n"
            f'  INTERFACE_INCLUDE_DIRECTORIES "{header.parent.as_posix()}")\n'
            + (
                "set_target_properties(PnmsSdkCacheFixture::setting PROPERTIES\n"
                f'  INTERFACE_COMPILE_DEFINITIONS "PNMS_SDK_SETTING={value}")\n'
                if changed_input == "config"
                else ""
            )
        )

    if changed_input == "config":
        header.write_text("// setting supplied by the package configuration\n")
    configure_package(1)
    cmake = source / "CMakeLists.txt"
    cmake.write_text(
        cmake.read_text()
        + (
            f'list(PREPEND CMAKE_SYSTEM_PREFIX_PATH "{package.as_posix()}")\n'
            if discovery == "system-prefix"
            else ""
        )
        + "find_package(nlohmann_json CONFIG REQUIRED)\n"
        "find_package(PnmsSdkCacheFixture CONFIG REQUIRED)\n"
        "target_link_libraries(physicsnemo_inference PRIVATE "
        "PnmsSdkCacheFixture::setting)\n"
    )
    if discovery == "prefix":
        monkeypatch.setenv("CMAKE_PREFIX_PATH", str(package))
    else:
        assert "CMAKE_PREFIX_PATH" not in os.environ
    (source / "src/runtime.cpp").write_text(
        "#include <external.setting>\nint sdk_setting() { return PNMS_SDK_SETTING; }\n"
    )

    def prepare(name):
        prepared = model.prepare_environment(
            cpu_sdk.project,
            tmp_path / name,
            repository=cpu_sdk.repository,
            runner=cpu_sdk.runner,
        )
        cache = (tmp_path / name / "sdk-build/CMakeCache.txt").read_text()
        assert f"nlohmann_json_DIR:PATH={installed_config.as_posix()}\n" in cache
        assert f"PnmsSdkCacheFixture_DIR:PATH={config.parent.as_posix()}\n" in cache
        return prepared

    def observed(prepared):
        return subprocess.check_output([prepared["runtime"]], text=True).strip()

    first = prepare("first")
    assert observed(first) == "1"
    seal = (Path(first["sdk_prefix"]).parent / "seal.json").read_bytes()
    same = prepare("same")
    assert observed(same) == "1"
    assert (same["sdk_prefix"] == first["sdk_prefix"]) == ("INCLUDE" not in os.environ)
    if changed_input == "header":
        header.write_text("#define PNMS_SDK_SETTING 2\n")
    else:
        configure_package(2)
    changed = prepare("changed")
    assert observed(changed) == "2"
    assert changed["sdk_prefix"] != first["sdk_prefix"]
    assert observed(first) == "1"
    assert (Path(first["sdk_prefix"]).parent / "seal.json").read_bytes() == seal


@pytest.mark.parametrize(
    "setting",
    [
        "CXX",
        "CXXFLAGS",
        "absolute-toolchain",
        "relative-toolchain",
        "CXX-executable",
        "cmake-executable",
    ],
)
def test_compiler_settings_invalidate_default_sdk_cache(
    tmp_path, monkeypatch, cpu_sdk, setting
):
    source = cpu_sdk.repository / "native-inference/cpp-runtime"
    toolchain = source / "toolchain.cmake"
    executables = {
        "cmake": shutil.which("cmake"),
        "CXX": shutil.which(os.environ["CXX"]),
    }
    tools = tmp_path / "tools"
    tools.mkdir()

    def select(value):
        flags = f"-DPNMS_SDK_SETTING={value}"
        if setting == "CXX":
            monkeypatch.setenv(
                "CXX", f'"{Path(executables["CXX"]).as_posix()}" {flags}'
            )
        elif setting == "CXXFLAGS":
            monkeypatch.setenv(setting, flags)
        elif setting.endswith("-executable"):
            name = setting.removesuffix("-executable")
            condition = "'-S' in arguments" if name == "cmake" else "True"
            argument = f"-DCMAKE_CXX_FLAGS={flags}" if name == "cmake" else flags
            wrapper = python_executable(
                tmp_path / f"{name}-{value}",
                "import subprocess, sys\n"
                "arguments = sys.argv[1:]\n"
                f"if arguments != ['--version'] and ({condition}):\n"
                f"    arguments.append({argument!r})\n"
                f"raise SystemExit(subprocess.call([{executables[name]!r}, *arguments]))\n",
            )
            selected = tools / (f"{name}.exe" if sys.platform == "win32" else name)
            shutil.copyfile(wrapper, selected)
            selected.chmod(0o755)
            if name == "cmake":
                monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
            else:
                monkeypatch.setenv("CXX", str(selected))
        else:
            toolchain.write_text(f'set(CMAKE_CXX_FLAGS_INIT "{flags}")\n')
            monkeypatch.setenv(
                "CMAKE_TOOLCHAIN_FILE",
                toolchain.name if setting == "relative-toolchain" else str(toolchain),
            )

    def prepare(name):
        return model.prepare_environment(
            cpu_sdk.project,
            tmp_path / name,
            repository=cpu_sdk.repository,
            runner=cpu_sdk.runner,
        )

    def runtime_setting(prepared):
        return subprocess.check_output([prepared["runtime"]], text=True).strip()

    def compilations():
        return [cmd for cmd, _ in cpu_sdk.calls if "--build" in cmd]

    select(1)
    first = prepare("first")
    same = prepare("same")
    # MSVC's required INCLUDE environment has undiscovered header inputs.
    reusable = not setting.endswith("toolchain") and "INCLUDE" not in os.environ
    assert (same["sdk_prefix"] == first["sdk_prefix"]) == reusable
    assert runtime_setting(same) == "1"
    count = 1 if reusable else 2
    assert len(compilations()) == count

    select(2)
    changed = prepare("changed")
    assert changed["stack"]["compiler"] == first["stack"]["compiler"]
    assert runtime_setting(changed) == "2"
    assert changed["sdk_prefix"] != first["sdk_prefix"]
    assert len(compilations()) == count + 1


@pytest.mark.parametrize(
    "reference", ["header", "response", "nested-header", "forwarded", "toolchain"]
)
def test_referenced_compiler_header_invalidates_default_sdk_cache(
    tmp_path, monkeypatch, cpu_sdk, reference
):
    header = tmp_path / "external setting.hpp"
    option = "/FI" if os.environ["CXX"] == "cl" else "-include "
    forced = header
    if reference == "nested-header":
        forced = tmp_path / "indirect.hpp"
        forced.write_text(f'#include "{header.name}"\n')
    flags = f'{option}"{forced}"'
    if reference == "forwarded" and os.environ["CXX"] != "cl":
        flags = f'-Wp,-include,"{forced}"'
    if reference == "response":
        response = tmp_path / "compiler flags.rsp"
        response.write_text(flags + "\n")
        flags = f'@"{response}"'
    if reference == "toolchain":
        toolchain = tmp_path / "toolchain.cmake"
        toolchain.write_text(f"set(CMAKE_CXX_FLAGS_INIT [=[{flags}]=])\n")
        monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", str(toolchain))
    else:
        monkeypatch.setenv("CXXFLAGS", flags)

    def prepare(name):
        return model.prepare_environment(
            cpu_sdk.project,
            tmp_path / name,
            repository=cpu_sdk.repository,
            runner=cpu_sdk.runner,
        )

    def observed(prepared):
        return subprocess.check_output([prepared["runtime"]], text=True).strip()

    header.write_text("#define PNMS_SDK_SETTING 1\n")
    first = prepare("first")
    prefix = Path(first["sdk_prefix"])
    seal = (prefix.parent / "seal.json").read_bytes()
    same = prepare("same")
    # Indirect includes and custom toolchains have undiscovered dependencies.
    reusable = (
        reference not in ("nested-header", "toolchain") and "INCLUDE" not in os.environ
    )
    assert (same["sdk_prefix"] == first["sdk_prefix"]) == reusable
    assert observed(same) == "1"
    count = 1 if reusable else 2
    assert len([cmd for cmd, _ in cpu_sdk.calls if "--build" in cmd]) == count

    header.write_text("#define PNMS_SDK_SETTING 2\n")
    changed = prepare("changed")
    assert observed(changed) == "2"
    assert changed["sdk_prefix"] != first["sdk_prefix"]
    assert observed(first) == "1"
    assert (prefix.parent / "seal.json").read_bytes() == seal
    assert model._valid_sdk_cache(prefix.parent)
    assert len([cmd for cmd, _ in cpu_sdk.calls if "--build" in cmd]) == count + 1


@pytest.mark.parametrize(
    "forwarding", ["-Wp,-include,", "-Xpreprocessor -include -Xpreprocessor "]
)
def test_forwarded_compiler_inputs_fingerprint_header_bytes(tmp_path, forwarding):
    header = tmp_path / "external.hpp"
    environment = {"CXXFLAGS": f'{forwarding}"{header}"'}
    header.write_text("#define PNMS_SDK_SETTING 1\n")
    first = model._sdk_compiler_inputs(environment)
    assert first == {str(header): model._hash(header)}
    assert model._sdk_compiler_inputs(environment) == first
    header.write_text("#define PNMS_SDK_SETTING 2\n")
    assert model._sdk_compiler_inputs(environment) != first


@pytest.mark.parametrize(
    ("variable", "flag"),
    [
        ("CXXFLAGS", "-fplugin="),
        ("CXXFLAGS", "-fmodule-file="),
        ("CXXFLAGS", "-I"),
        ("CXXFLAGS", "-Xclang -load -Xclang "),
        ("LDFLAGS", "-Wl,--version-script,"),
    ],
)
def test_unknown_compiler_file_options_prevent_reuse(tmp_path, variable, flag):
    path = tmp_path / "external-input"
    path.write_text("external compiler input\n")
    environment = {variable: f'{flag}"{path}"'}
    first = model._sdk_compiler_inputs(environment)
    assert "unresolved" in first
    assert model._sdk_compiler_inputs(environment) != first


@pytest.mark.parametrize(
    "variable", ["CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "INCLUDE", "SDKROOT"]
)
@pytest.mark.parametrize("value", ["", ".", "include tree"])
def test_include_search_environment_prevents_sdk_reuse(tmp_path, variable, value):
    # Empty entries can search the working directory. No suffix or tree scan can
    # prove the dependencies of arbitrary includes, symlinks, or generated files.
    environment = {
        variable: str(tmp_path / value) if value == "include tree" else value
    }
    first = model._sdk_compiler_inputs(environment)
    assert "unresolved" in first
    assert model._sdk_compiler_inputs(environment) != first
    assert model._sdk_compiler_inputs(environment, unresolved_key="invocation") == {
        "unresolved": "invocation"
    }


def test_include_search_fallback_still_fingerprints_known_inputs(tmp_path):
    header = tmp_path / "forced.hpp"
    header.write_text("#define PNMS_SDK_SETTING 1\n")
    environment = {"CPATH": str(tmp_path), "CXXFLAGS": f'-include "{header}"'}
    first = model._sdk_compiler_inputs(environment, unresolved_key="invocation")
    assert first == {str(header): model._hash(header), "unresolved": "invocation"}
    header.write_text("#define PNMS_SDK_SETTING 2\n")
    assert model._sdk_compiler_inputs(environment, unresolved_key="invocation") != first


def test_include_search_header_changes_rebuild_sdk(tmp_path, monkeypatch, cpu_sdk):
    # The search directory deliberately contains the output/cache directories, so
    # ordinary build-generated files must not make input revalidation fail.
    header = tmp_path / "external.setting"
    source = cpu_sdk.repository / "native-inference/cpp-runtime/src/runtime.cpp"
    source.write_text(
        "#include <external.setting>\nint sdk_setting() { return PNMS_SDK_SETTING; }\n"
    )
    if os.environ["CXX"] == "cl":
        monkeypatch.setenv(
            "INCLUDE", str(tmp_path) + os.pathsep + os.environ.get("INCLUDE", "")
        )
    else:
        monkeypatch.setenv("CPATH", str(tmp_path))

    def prepare(name):
        return model.prepare_environment(
            cpu_sdk.project,
            tmp_path / name,
            repository=cpu_sdk.repository,
            runner=cpu_sdk.runner,
        )

    def observed(prepared):
        return subprocess.check_output([prepared["runtime"]], text=True).strip()

    header.write_text("#define PNMS_SDK_SETTING 1\n")
    first = prepare("first")
    prefix = Path(first["sdk_prefix"])
    seal = (prefix.parent / "seal.json").read_bytes()
    assert observed(first) == "1"

    header.write_text("#define PNMS_SDK_SETTING 2\n")
    changed = prepare("changed")
    assert observed(changed) == "2"
    assert changed["sdk_prefix"] != first["sdk_prefix"]
    same = prepare("same")
    assert observed(same) == "2"
    assert same["sdk_prefix"] != changed["sdk_prefix"]
    assert observed(first) == "1"
    assert (prefix.parent / "seal.json").read_bytes() == seal
    assert model._valid_sdk_cache(prefix.parent)
    assert len([cmd for cmd, _ in cpu_sdk.calls if "-S" in cmd]) == 3


@pytest.mark.parametrize(
    "input_kind",
    ["header", "source", "toolchain", "CXX-executable", "cmake-executable"],
)
def test_sdk_inputs_changed_during_compile_are_not_published(
    tmp_path, monkeypatch, cpu_sdk, input_kind
):
    source = cpu_sdk.repository / "native-inference/cpp-runtime"
    if input_kind == "header":
        path = tmp_path / "external.hpp"
        option = "/FI" if os.environ["CXX"] == "cl" else "-include "
        monkeypatch.setenv("CXXFLAGS", f'{option}"{path}"')
        template = "#define PNMS_SDK_SETTING {}\n"
    elif input_kind == "source":
        path = source / "src/runtime.cpp"
        template = "int sdk_setting() {{ return {}; }}\n"
    elif input_kind == "toolchain":
        path = source / "toolchain.cmake"
        monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", str(path))
        template = 'set(CMAKE_CXX_FLAGS_INIT "-DPNMS_SDK_SETTING={}")\n'
    else:
        name = input_kind.removesuffix("-executable")
        executable = shutil.which("cmake" if name == "cmake" else os.environ["CXX"])
        tools = tmp_path / "tools"
        tools.mkdir()
        path = tools / name
        condition = "'-S' in arguments" if name == "cmake" else "True"
        argument = (
            "-DCMAKE_CXX_FLAGS=-DPNMS_SDK_SETTING={}"
            if name == "cmake"
            else "-DPNMS_SDK_SETTING={}"
        )
        template = (
            "import subprocess, sys\n"
            "arguments = sys.argv[1:]\n"
            f"if arguments and arguments != ['--version'] and ({condition}):\n"
            f"    arguments.append({argument!r})\n"
            f"raise SystemExit(subprocess.call([{executable!r}, *arguments]))\n"
        )
        if name == "cmake":
            monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])

    wrapper_versions = {}

    def select(value):
        if input_kind.endswith("-executable"):
            if value not in wrapper_versions:
                selected = python_executable(path, template.format(value))
                wrapper_versions[value] = (selected, selected.read_bytes())
            selected, contents = wrapper_versions[value]
            selected.write_bytes(contents)
            if name == "CXX":
                monkeypatch.setenv("CXX", str(selected))
        else:
            path.write_text(template.format(value))

    def prepare(name, runner=cpu_sdk.runner):
        return model.prepare_environment(
            cpu_sdk.project,
            tmp_path / name,
            repository=cpu_sdk.repository,
            runner=runner,
        )

    select(1)
    first = prepare("first")
    prefix = Path(first["sdk_prefix"])
    seal = (prefix.parent / "seal.json").read_bytes()
    select(2)

    def mutate_before_compile(command, **kwargs):
        if "--build" in command:
            select(3)
        return cpu_sdk.runner(command, **kwargs)

    with pytest.raises(model.ModelError, match="SDK inputs changed"):
        prepare("mutated", mutate_before_compile)
    assert (prefix.parent / "seal.json").read_bytes() == seal
    assert model._valid_sdk_cache(prefix.parent)
    assert len(list(prefix.parent.parent.glob("*/seal.json"))) == 1

    select(1)
    stages = [
        cmd
        for cmd, _ in cpu_sdk.calls
        if "-S" in cmd or "--build" in cmd or "--install" in cmd
    ]
    same = prepare("unchanged")
    reusable = input_kind != "toolchain" and "INCLUDE" not in os.environ
    assert (same["sdk_prefix"] == first["sdk_prefix"]) == reusable
    unchanged_stages = [
        cmd
        for cmd, _ in cpu_sdk.calls
        if "-S" in cmd or "--build" in cmd or "--install" in cmd
    ]
    assert unchanged_stages[: len(stages)] == stages
    assert len(unchanged_stages) == len(stages) + (1 if reusable else 3)
    select(2)
    recovered = prepare("recovered")
    assert subprocess.check_output([recovered["runtime"]], text=True).strip() == "2"
    assert recovered["sdk_prefix"] != first["sdk_prefix"]
    assert subprocess.check_output([first["runtime"]], text=True).strip() == "1"
    assert (prefix.parent / "seal.json").read_bytes() == seal


@pytest.mark.parametrize("selection", ["argument", "project"])
def test_explicit_sdk_prefix_survives_compiler_setting_changes(
    tmp_path, monkeypatch, selection
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    options = {"runtime": runtime, "repository": ROOT, "runner": Runner()}
    if selection == "argument":
        options["sdk_prefix"] = prefix
    else:
        app["build"]["sdk_prefix"] = str(prefix)
    toolchain = tmp_path / "toolchain.cmake"
    monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", str(toolchain))
    for name, value in (("first", 1), ("changed", 2)):
        monkeypatch.setenv("CXXFLAGS", f"-DPNMS_SDK_SETTING={value}")
        toolchain.write_text(f"# selected toolchain {value}\n")
        prepared = model.prepare_environment(app, tmp_path / name, **options)
        assert prepared["sdk_prefix"] == str(prefix)
    assert all(command[0] != "cmake" for command, _ in options["runner"].calls)


@pytest.mark.parametrize("existing", [False, True])
def test_managed_windows_venv_uses_selected_scripts_interpreter(
    tmp_path, monkeypatch, existing
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(
        model, "sys", SimpleNamespace(platform="win32", executable=sys.executable)
    )
    app = project(tmp_path)
    destination = tmp_path / "managed"
    interpreter = destination / "Scripts/python.exe"
    app["build"]["venv"] = str(destination)

    def create_interpreter():
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"selected Windows interpreter")
        interpreter.chmod(0o755)

    if existing:
        create_interpreter()

    class EnvironmentRunner(Runner):
        def __call__(self, command, **kwargs):
            if "setup-env" in command:
                create_interpreter()
            return super().__call__(command, **kwargs)

    runtime, prefix = fake_runtime(tmp_path)
    runner = EnvironmentRunner()
    prepared = model.prepare_environment(
        app,
        tmp_path / "environment",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=runner,
    )
    assert prepared["python"] == str(interpreter)
    probes = [command for command, _ in runner.calls if "-c" in command]
    assert len(probes) == 2
    assert all(command[0] == str(interpreter) for command in probes)
    assert any("setup-env" in command for command, _ in runner.calls) is not existing


@pytest.mark.parametrize("explicit_runtime", [False, True])
def test_windows_environment_preserves_selected_dll_paths(
    tmp_path, monkeypatch, explicit_runtime
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("PATH", "selected-tools;system")
    monkeypatch.setattr(
        model, "sys", SimpleNamespace(platform="win32", executable=sys.executable)
    )
    monkeypatch.setattr(model, "os", SimpleNamespace(**{**vars(os), "pathsep": ";"}))
    runtime, prefix = fake_runtime(tmp_path)
    executable = runtime.with_suffix(".exe")
    if runtime != executable:
        runtime.rename(executable)
    runner = Runner()
    result = model.prepare_environment(
        project(tmp_path),
        tmp_path / "environment",
        repository=ROOT,
        runtime=executable if explicit_runtime else None,
        sdk_prefix=prefix,
        runner=runner,
    )
    assert result["runtime"] == str(executable)
    selected = result["native_environment"]
    assert selected["PATH"] == result["process_env"]["PATH"]
    assert selected["PATH"].split(";") == [
        str(prefix / "bin"),
        str(prefix / "lib"),
        str(prefix / "lib64"),
        str(Path("/torch") / "lib"),
        "selected-tools",
        "system",
    ]
    assert runner.calls[-1][1]["env"]["PATH"] == selected["PATH"]


def test_config_only_container_project_remains_a_host_static_check(tmp_path):
    app = project(tmp_path)
    runner = Runner()
    result = model.check_models(
        app,
        tmp_path / "checks",
        repository=ROOT,
        config_only=True,
        environment="container",
        runner=runner,
    )
    assert result["models"]["primary"]["result"]["status"] == "configuration-ok"
    assert not any("docker" in cmd or "-c" in cmd for cmd, _ in runner.calls)


def test_corrupt_model_cache_is_replaced_after_successful_rebuild(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    cache = tmp_path / "cache"
    runner = Runner()
    kwargs = {
        "repository": ROOT,
        "runtime": runtime,
        "sdk_prefix": prefix,
        "cache_dir": cache,
        "runner": runner,
    }
    model.build_models(app, tmp_path / "first", **kwargs)
    next(cache.glob("*/output/model/model-release.json")).write_text("corrupt")
    entry = next(cache.glob("*/output")).parent
    corrupted = model._inventory(entry)
    with pytest.raises(model.ModelError, match="rebuild failed"):
        model.build_models(
            app,
            tmp_path / "failed",
            **{**kwargs, "runner": Runner(failure="rebuild failed")},
        )
    assert model._inventory(entry) == corrupted
    assert not list(cache.glob("*.invalid.*"))
    second = model.build_models(app, tmp_path / "second", **kwargs)
    third = model.build_models(app, tmp_path / "third", **kwargs)
    assert second["models"]["primary"]["reused"] is False
    assert third["models"]["primary"]["reused"] is True
    quarantined = list(cache.glob(entry.name + ".invalid.*"))
    assert len(quarantined) == 1
    assert model._inventory(quarantined[0]) == corrupted


@pytest.fixture
def model_cache_build(tmp_path, monkeypatch):
    app = project(tmp_path)
    context = {
        "kind": "local",
        "python": sys.executable,
        "gpu": {"architecture": "sm90"},
        "stack": {},
        "process_env": dict(os.environ),
    }
    monkeypatch.setattr(model, "prepare_environment", lambda *args, **kwargs: context)
    options = {"repository": ROOT, "cache_dir": tmp_path / "cache"}
    return app, options


def test_model_cache_preserves_valid_winner_published_during_compilation(
    tmp_path, model_cache_build
):
    app, options = model_cache_build
    winner = {}

    class CompetingRunner(Runner):
        def __call__(self, command, **kwargs):
            if "build" in command:
                built = model.build_models(
                    app,
                    tmp_path / "winner",
                    **options,
                    runner=Runner(),
                    source_sha="1" * 40,
                )["models"]["primary"]
                cache = options["cache_dir"] / built["identity"]
                winner.update(
                    cache=cache,
                    inventory=model._inventory(cache),
                    inode=cache.stat().st_ino,
                )
            return super().__call__(command, **kwargs)

    own = model.build_models(
        app,
        tmp_path / "own",
        **options,
        runner=CompetingRunner(),
        source_sha="2" * 40,
    )["models"]["primary"]

    assert own["reused"] is False
    assert own["compilation_source_sha"] == "2" * 40
    assert Path(own["package"]).is_dir()
    assert model._compilation_source(Path(own["output_root"])) == "2" * 40
    assert own["result"]["artifacts"]["package"] == str(
        Path(own["output_root"]) / "model"
    )
    assert winner["cache"].stat().st_ino == winner["inode"]
    assert model._inventory(winner["cache"]) == winner["inventory"]
    assert not list(options["cache_dir"].glob("*.invalid.*"))
    hit = model.build_models(
        app, tmp_path / "hit", **options, runner=Runner(), source_sha="3" * 40
    )["models"]["primary"]
    assert hit["reused"] is True
    assert hit["compilation_source_sha"] == "1" * 40


@pytest.mark.parametrize("rename_error", [None, errno.EACCES])
def test_model_cache_accepts_verified_winner_after_rename_collision(
    tmp_path, monkeypatch, model_cache_build, rename_error
):
    app, options = model_cache_build
    rename = Path.rename
    winner = {}

    def publish_winner(staging, target):
        if (staging / "seal.json").is_file() and (staging / "output").is_dir():
            target = Path(target)
            assert not target.exists()
            shutil.copytree(staging, target)
            winner.update(cache=target, inventory=model._inventory(target))
            if rename_error is not None:
                raise PermissionError(rename_error, "competing directory publisher")
        return rename(staging, target)

    monkeypatch.setattr(Path, "rename", publish_winner)
    own = model.build_models(app, tmp_path / "own", **options, runner=Runner())[
        "models"
    ]["primary"]

    assert Path(own["package"]).is_dir()
    assert own["reused"] is False
    assert model._inventory(winner["cache"]) == winner["inventory"]
    assert list(options["cache_dir"].glob(winner["cache"].name + ".*")) == []


@pytest.mark.parametrize(
    ("winner_state", "error_number"),
    [
        ("absent", errno.EACCES),
        ("corrupt", errno.EACCES),
        ("mismatched", errno.EACCES),
        ("valid", errno.EIO),
    ],
)
def test_model_cache_does_not_hide_unrelated_or_unverified_rename_errors(
    tmp_path, monkeypatch, model_cache_build, winner_state, error_number
):
    app, options = model_cache_build
    rename = Path.rename
    denied = OSError(error_number, "permanent publication failure")
    targets = []

    def deny_publication(staging, target):
        if (staging / "seal.json").is_file() and (staging / "output").is_dir():
            target = Path(target)
            targets.append(target)
            assert not target.exists()
            if winner_state != "absent":
                shutil.copytree(staging, target)
            if winner_state == "corrupt":
                (target / "output/model/model-release.json").write_text("corrupt")
            elif winner_state == "mismatched":
                seal_path = target / "seal.json"
                seal = json.loads(seal_path.read_text())
                seal["identity"]["profile"] = "other"
                seal_path.write_text(json.dumps(seal))
            raise denied
        return rename(staging, target)

    monkeypatch.setattr(Path, "rename", deny_publication)
    with pytest.raises(OSError) as failure:
        model.build_models(app, tmp_path / "own", **options, runner=Runner())
    assert failure.value is denied
    assert len(targets) == 1
    assert targets[0].exists() == (winner_state != "absent")


def test_real_builder_static_check_uses_captured_project_without_ml_imports(tmp_path):
    app = project(tmp_path)
    source = app["models"]["primary"]["project"]
    (source.parent / "build_adapter.py").write_text(
        "def create_model(config, assets): pass\ndef create_cases(config, assets): pass\n"
    )
    result = model.check_models(
        app,
        tmp_path / "checks",
        repository=ROOT,
        config_only=True,
        python=Path(sys.executable),
    )
    record = result["models"]["primary"]
    assert record["result"]["status"] == "configuration-ok"
    assert record["result"]["effective_config"]["executor"] == "local"
    assert (source.parent / "model-build.lock.json").read_text() == '{"original": true}'
    assert (tmp_path / "checks/logs/primary/result.json").is_file()


def test_bare_torch_gpu_uuid_is_normalized_for_cuda_binding_and_nvidia_smi():
    bare = "52091b7d-860a-28ea-7bdc-f929e5f2d532"
    result = model._selected_gpu(
        [{"uuid": bare, "name": "H100", "major": 9, "minor": 0, "total_memory": 80}],
        None,
        {},
    )
    assert result["uuid"] == "GPU-" + bare
    assert result["binding"] == "GPU-" + bare


def test_cuda_home_exposes_nvcc_even_when_not_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    toolkit = tmp_path / "cuda"
    compiler = toolkit / "bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.delenv("CUDACXX", raising=False)
    runtime, prefix = fake_runtime(tmp_path)
    result = model.prepare_environment(
        project(tmp_path),
        tmp_path / "environment",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=Runner(),
    )
    assert result["native_environment"]["CUDACXX"] == str(compiler)
    assert result["native_environment"]["CUDA_HOME"] == str(toolkit)


@pytest.mark.parametrize("environment_kind", ["process_env", "native_environment"])
@pytest.mark.parametrize("override", [None, "CUDA_HOME", "CUDAToolkit_ROOT"])
def test_symlinked_path_nvcc_is_discovered_by_legacy_cmake(
    tmp_path, monkeypatch, environment_kind, override
):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is required for legacy CUDA discovery checks")
    toolkit = tmp_path / "cuda-13.1"
    compiler = toolkit / "bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler = python_executable(
        compiler,
        "print('nvcc: NVIDIA (R) Cuda compiler driver')\n"
        "print('Cuda compilation tools, release 13.1, V13.1.80')\n",
    )
    (toolkit / "include").mkdir()
    (toolkit / "include/device_functions.h").write_text("// CUDA toolkit header\n")
    (toolkit / "lib").mkdir()
    runtime_library = toolkit / "lib/libcudart.a"
    runtime_library.write_bytes(b"CUDA runtime discovery fixture")
    tools = tmp_path / "host-tools"
    tools.mkdir()
    (tools / compiler.name).symlink_to(compiler)
    monkeypatch.setenv("PATH", str(tools))
    for name in (
        "CUDACXX",
        "CUDA_HOME",
        "CUDAToolkit_ROOT",
        "CUDA_PATH",
        "CUDA_TOOLKIT_ROOT",
        "CUDA_BIN_PATH",
        "CUDA_NVCC_EXECUTABLE",
        "CUDA_INC_PATH",
        "CUDA_LIB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    if override:
        monkeypatch.setenv(override, str(toolkit))
    runtime, prefix = fake_runtime(tmp_path)
    result = model.prepare_environment(
        project(tmp_path),
        tmp_path / "environment",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=Runner(),
    )
    selected = result["native_environment"]
    assert selected["CUDA_HOME"] == str(toolkit)
    assert selected["CUDAToolkit_ROOT"] == str(toolkit)
    assert selected["CUDACXX"] == str(compiler)
    script = tmp_path / "discover.cmake"
    discovery = tmp_path / "discovery.txt"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "if(POLICY CMP0146)\n"
        "  cmake_policy(SET CMP0146 OLD)\n"
        "endif()\n"
        "include(CMakeInitializeConfigs)\n"
        # Run only discovery: no CUDA compilation, linking, or GPU is needed.
        "set(CMAKE_FIND_LIBRARY_PREFIXES lib)\n"
        "set(CMAKE_FIND_LIBRARY_SUFFIXES .a)\n"
        "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)\n"
        "find_package(CUDA REQUIRED)\n"
        f'file(WRITE "{discovery.as_posix()}"\n'
        '  "${CUDA_TOOLKIT_ROOT_DIR}\\n${CUDA_NVCC_EXECUTABLE}\\n"\n'
        '  "${CUDA_INCLUDE_DIRS}\\n${CUDA_CUDART_LIBRARY}\\n${CUDA_VERSION}\\n")\n'
    )
    completed = subprocess.run(
        [cmake, "-P", str(script)],
        env={**os.environ, **result[environment_kind]},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    discovered = discovery.read_text().splitlines()
    assert [Path(value) for value in discovered[:4]] == [
        toolkit,
        compiler,
        toolkit / "include",
        runtime_library,
    ]
    assert discovered[4] == "13.1"


@pytest.mark.parametrize("override", ["CUDA_HOME", "CUDAToolkit_ROOT", "CUDA_PATH"])
def test_symlinked_nvcc_preserves_explicit_toolkit_override(
    tmp_path, monkeypatch, override
):
    compiler = tmp_path / "cuda/bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    link = tmp_path / "host-tools/nvcc"
    link.parent.mkdir()
    link.symlink_to(compiler)
    selected_root = str(tmp_path / "selected-toolkit")
    environment = {"CUDACXX": str(link), override: selected_root}

    model._cuda_environment(environment, {})

    assert environment[override] == selected_root
    assert environment["CUDA_HOME"] == selected_root
    assert environment["CUDAToolkit_ROOT"] == selected_root
    assert environment["CUDA_BIN_PATH"] == selected_root


def test_cuda_environment_matches_workflow_toolkit_precedence(tmp_path):
    compiler = tmp_path / "cuda/bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    selected_root = str(tmp_path / "selected-toolkit")
    other_root = str(tmp_path / "other-toolkit")
    environment = {
        "CUDACXX": str(compiler),
        "CUDA_HOME": other_root,
        "CUDAToolkit_ROOT": selected_root,
    }

    model._cuda_environment(environment, {})

    assert environment["CUDA_HOME"] == other_root
    assert environment["CUDAToolkit_ROOT"] == selected_root
    assert environment["CUDA_BIN_PATH"] == selected_root


def fake_torch(tmp_path):
    root = tmp_path / "torch"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = '2.8.0+cu128'\n"
        "version = SimpleNamespace(cuda='12.8')\n"
        "_C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)\n"
        "cuda = SimpleNamespace(device_count=lambda: 1, get_device_properties=lambda i: "
        "SimpleNamespace(name='H100', uuid='GPU-abcd', major=9, minor=0, total_memory=80))\n"
    )
    for relative in ("lib/libtorch.so", "include/torch.h"):
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original provider")
    return root


def selected_probe(tmp_path, *, trt_root=None, version="1.0"):
    # Run the real probe in an isolated stub environment, without importing ML
    # libraries or requiring any GPU on the test host.
    setup = (
        "import importlib.metadata as metadata\n"
        "from types import SimpleNamespace\n"
        f"metadata.distributions = lambda: [SimpleNamespace(metadata={{'Name': 'numpy'}}, version={version!r})]\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", setup + model._PROBE, "aoti", str(trt_root or "")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("relative", ["lib/libtorch.so", "include/torch.h"])
def test_same_version_torch_native_or_header_change_changes_probe(tmp_path, relative):
    root = fake_torch(tmp_path)
    before = selected_probe(tmp_path)
    (root / relative).write_bytes(b"replaced provider with same reported version")
    after = selected_probe(tmp_path)
    assert before != after
    assert before["torch"] == after["torch"]


@pytest.mark.parametrize("relative", ["lib/libnvinfer.so.10", "include/NvInfer.h"])
def test_explicit_tensorrt_native_or_header_change_changes_probe(tmp_path, relative):
    fake_torch(tmp_path)
    root = tmp_path / "tensorrt"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original TensorRT provider")
    before = selected_probe(tmp_path, trt_root=root)
    path.write_bytes(b"replaced TensorRT provider")
    assert selected_probe(tmp_path, trt_root=root) != before


def test_selected_python_distribution_version_changes_probe(tmp_path):
    fake_torch(tmp_path)
    assert selected_probe(tmp_path, version="1.0") != selected_probe(
        tmp_path, version="2.0"
    )


def test_native_provider_replacement_invalidates_verified_model_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)

    class ProviderRunner(Runner):
        provider = "old-libstdc++"
        fail = False

        def __call__(self, command, **kwargs):
            if (
                "-c" in command
                and command[command.index("-c") + 1] == compatibility.PROBE
            ):
                self.calls.append((command, kwargs))
                return subprocess.CompletedProcess(
                    command,
                    1 if self.fail else 0,
                    json.dumps(
                        {
                            "schema_version": 1,
                            "system_libraries": {
                                "libstdc++.so.6": {
                                    "sha256": self.provider,
                                    "size_bytes": 10,
                                }
                            },
                        }
                    ),
                    "native provider is unavailable" if self.fail else "",
                )
            return super().__call__(command, **kwargs)

    runner = ProviderRunner()
    options = dict(
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=tmp_path / "cache",
        runner=runner,
    )
    first = model.build_models(app, tmp_path / "first", **options)
    same = model.build_models(app, tmp_path / "same", **options)
    assert same["models"]["primary"]["reused"] is True
    runner.provider = "new-libstdc++"
    changed = model.build_models(app, tmp_path / "changed", **options)
    assert changed["models"]["primary"]["reused"] is False
    assert (
        changed["models"]["primary"]["identity"]
        != first["models"]["primary"]["identity"]
    )
    runner.fail = True
    calls = len([cmd for cmd, _ in runner.calls if "build" in cmd])
    with pytest.raises(model.ModelError, match="provider"):
        model.build_models(app, tmp_path / "failed", **options)
    assert len([cmd for cmd, _ in runner.calls if "build" in cmd]) == calls


@pytest.mark.parametrize("relative", ["lib/libtorch.so", "include/torch.h"])
def test_same_version_framework_bytes_invalidate_sdk_cache(
    tmp_path, monkeypatch, relative
):
    # The fake builder consumes none of the inherited compiler search paths.
    for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "INCLUDE", "SDKROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    # This fake SDK consumes only the immutable fixture copied below.
    monkeypatch.setattr(
        workflow_cache,
        "configured_inputs",
        lambda *args, **kwargs: {"fixture": "immutable"},
    )
    probe_root = tmp_path / "selected-environment"
    probe_root.mkdir()
    torch_root = fake_torch(probe_root)
    app = project(tmp_path)
    _, fixture_sdk = fake_runtime(tmp_path)

    class SDKRunner(Runner):
        prefix = None

        def __call__(self, command, **kwargs):
            command = [str(part) for part in command]
            if "-c" in command and command[command.index("-c") + 1] == model._PROBE:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(selected_probe(probe_root)), ""
                )
            if command[0] == "cmake" and "-S" in command:
                self.prefix = Path(
                    next(
                        part.split("=", 1)[1]
                        for part in command
                        if part.startswith("-DCMAKE_INSTALL_PREFIX=")
                    )
                )
                self.calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, "configured", "")
            if command[0] == "cmake" and "--install" in command:
                shutil.copytree(fixture_sdk, self.prefix)
                self.calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, "installed", "")
            return super().__call__(command, **kwargs)

    runner = SDKRunner()
    options = dict(repository=ROOT, runner=runner)
    first = model.prepare_environment(app, tmp_path / "first", **options)
    same = model.prepare_environment(app, tmp_path / "same", **options)
    assert same["sdk_prefix"] == first["sdk_prefix"]
    (torch_root / relative).write_bytes(b"replaced framework provider")
    changed = model.prepare_environment(app, tmp_path / "changed", **options)
    assert changed["sdk_prefix"] != first["sdk_prefix"]
    assert len([cmd for cmd, _ in runner.calls if "--build" in cmd]) == 2


def test_builder_implementation_change_invalidates_model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("PNMS_SOURCE_SHA", "a" * 40)
    repository = tmp_path / "repository"
    builder = repository / "native-inference/model-builder"
    source = builder / "src/model_builder/build/export.py"
    source.parent.mkdir(parents=True)
    source.write_text("# original exporter\n")
    (repository / "native-inference/pnms-model-builder").write_text("# launcher\n")
    (builder / "toolchain.lock.json").write_text("{}")
    runtime, prefix = fake_runtime(tmp_path)
    app = project(tmp_path)
    options = dict(
        repository=repository,
        runtime=runtime,
        sdk_prefix=prefix,
        cache_dir=tmp_path / "cache",
        runner=Runner(),
    )
    first = model.build_models(app, tmp_path / "first", **options)
    (builder / "README.md").write_text("Documentation-only update\n")
    same = model.build_models(app, tmp_path / "same", **options)
    assert same["models"]["primary"]["reused"] is True
    source.write_text("# changed exporter\n")
    changed = model.build_models(app, tmp_path / "changed", **options)
    assert changed["models"]["primary"]["reused"] is False
    assert (
        changed["models"]["primary"]["identity"]
        != first["models"]["primary"]["identity"]
    )


def test_native_environment_preserves_exact_selected_builder_loader_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    compatibility_libraries = str(tmp_path / "cuda-compat")
    loader = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    monkeypatch.setenv(loader, compatibility_libraries)
    runtime, prefix = fake_runtime(tmp_path)
    result = model.prepare_environment(
        project(tmp_path),
        tmp_path / "environment",
        repository=ROOT,
        runtime=runtime,
        sdk_prefix=prefix,
        runner=Runner(),
    )
    selected = result["native_environment"]
    assert selected[loader] == result["process_env"][loader]
    assert selected[loader].split(os.pathsep) == (
        [str(prefix / "bin")] if sys.platform == "win32" else []
    ) + [
        str(prefix / "lib"),
        str(prefix / "lib64"),
        str(Path("/torch") / "lib"),
        compatibility_libraries,
    ]
