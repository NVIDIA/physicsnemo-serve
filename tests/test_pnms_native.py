"""CPU component contracts for the installed shared workflow and direct runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_NAME = "pnms-native-run.exe" if sys.platform == "win32" else "pnms-native-run"
sys.path.insert(0, str(REPOSITORY / "application-cli/src"))
from pnms_app import native  # noqa: E402


def command(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(argument) for argument in arguments],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def sdk(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not shutil.which("cmake"):
        pytest.skip("CMake is required for native CPU component checks")
    root = tmp_path_factory.mktemp("pnms-sdk-component")
    command(
        "cmake",
        "-S",
        REPOSITORY / "native-inference/cpp-runtime",
        "-B",
        root / "build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPNMIR_BUILD_TESTS=OFF",
        f"-DCMAKE_INSTALL_PREFIX={root / 'install'}",
    )
    command(
        "cmake", "--build", root / "build", "--config", "Release", "--parallel", "4"
    )
    command("cmake", "--install", root / "build", "--config", "Release")
    return root / "install"


@pytest.fixture(scope="module")
def workflow(sdk: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pnms-workflow-component")
    command(
        "cmake",
        "-S",
        REPOSITORY / "native-inference/workflows/runtime",
        "-B",
        root / "build",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={sdk}",
        f"-DCMAKE_INSTALL_PREFIX={root / 'install'}",
        "-DPNMS_ENABLE_MOCK_BACKEND=ON",
        "-DBUILD_TESTING=ON",
    )
    command(
        "cmake", "--build", root / "build", "--config", "Release", "--parallel", "4"
    )
    command(
        "ctest", "--test-dir", root / "build", "-C", "Release", "--output-on-failure"
    )
    command("cmake", "--install", root / "build", "--config", "Release")
    return root / "install"


def make_config(root: Path) -> dict:
    package = root / "package"
    package.mkdir()
    (package / "identity.mock").write_text("identity")
    (package / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": {"name": "identity", "version": "1"},
                "inputs": [{"name": "input", "dtype": "float32", "shape": [4]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [4]}],
                "artifacts": [
                    {
                        "backend": "mock",
                        "target": "cpu",
                        "precision": "fp32",
                        "path": "identity.mock",
                    }
                ],
            }
        )
    )
    return {
        "schema_version": 2,
        "adapter": "tensor.v1",
        "models": {
            "primary": {
                "package_path": str(package),
                "backend": "mock",
                "device": "cpu",
                "device_index": 0,
                "self_test": {
                    "inputs": {"input": [0, 1, -1, 4]},
                    "expected_outputs": {"output": [0, 1, -1, 4]},
                },
            }
        },
        "tensor_adapter": {
            "model_alias": "primary",
            "inputs": {
                "input": {"request_field": "values", "codec": "json-array-fp32"}
            },
            "outputs": {
                "output": {"response_field": "values", "codec": "json-array-fp32"}
            },
        },
    }


def test_installed_runner_executes_same_host_and_preserves_admission(
    workflow: Path, tmp_path: Path
):
    runner = workflow / "bin" / RUNNER_NAME
    assert runner.is_file(), "workflow installation must include the standalone runner"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(make_config(tmp_path)))
    request = tmp_path / "request.json"
    envelope = {
        "request_id": "cpu-case",
        "parameters": {"values": [4, 3, 2, 1]},
        "inputs": {},
        "output_dir": str(tmp_path / "outputs"),
    }
    request.write_text(json.dumps(envelope))
    result = json.loads(
        command(runner, "--config", config, "--request", request).stdout
    )
    assert result["payload"] == {"values": [4, 3, 2, 1]}
    assert result["metadata"]["session_load_count"] == 1
    envelope["parameters"]["extra"] = 1
    request.write_text(json.dumps(envelope))
    rejected = subprocess.run(
        [str(runner), "--config", str(config), "--request", str(request)],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert "mapping exactly" in rejected.stderr
    assert rejected.stdout == ""


def test_build_workflow_installs_source_linked_custom_runner(
    sdk: Path, workflow: Path, tmp_path: Path
):
    installed = native.build_workflow(
        {"root": tmp_path, "workflow": {"template": "custom"}},
        tmp_path / "workflow",
        repository=REPOSITORY,
        sdk_prefix=sdk,
        python=Path(sys.executable),
        backend="mock",
        custom_source=REPOSITORY / "native-inference/workflows/runtime/tests/custom",
        custom_target="pnms_test_custom",
    )
    runner = Path(installed.get("runner", tmp_path / "missing"))
    assert runner.is_file(), "native workflow build must expose the installed runner"
    from pnms_app.common import file_identity

    assert installed["compatibility"]["schema_version"] == 1
    assert installed["compatibility"]["workflow_binary"] == file_identity(
        installed["library"]
    )
    config = make_config(tmp_path)
    config["adapter"] = "custom.v1"
    del config["tensor_adapter"]
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(config))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "request_id": "custom-case",
                "parameters": {"values": [1, 2, 3, 4]},
                "output_dir": str(tmp_path / "attempt"),
                "inputs": {},
            }
        )
    )
    result = json.loads(
        command(runner, "--config", config_path, "--request", request).stdout
    )
    assert result["payload"] == {"values": [11, 12, 13, 14]}
    assert result["artifacts"][0]["path"] == "custom.f32"

    # Another installed workflow with the same library name must not override
    # the sealed custom workflow, even when its path belongs to the build env.
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    shutil.copytree(installed["install_root"], sealed / "workflow")
    config = make_config(sealed)
    config["adapter"] = "custom.v1"
    del config["tensor_adapter"]
    config["models"]["primary"]["package_path"] = "package"
    (sealed / "configuration").mkdir()
    (sealed / "configuration/workflow.json").write_text(json.dumps(config))
    (sealed / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": {
                    "config": "configuration/workflow.json",
                    "runner": f"workflow/bin/{RUNNER_NAME}",
                },
            }
        )
    )
    parameters = tmp_path / "custom-parameters.json"
    parameters.write_text(json.dumps({"values": [1, 2, 3, 4]}))
    loader = (
        "PATH"
        if sys.platform == "win32"
        else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    executed = native.run_bundle(
        sealed,
        parameters,
        tmp_path / "sealed-run",
        environment={loader: str(workflow / "lib")},
    )
    assert executed["result"]["payload"] == {"values": [11, 12, 13, 14]}


def test_build_workflow_selects_cuda_prefix_for_legacy_torch_cmake(
    sdk: Path, tmp_path: Path
):
    toolkit = tmp_path / "selected-cuda"
    native.build_workflow(
        {"root": tmp_path},
        tmp_path / "workflow",
        repository=REPOSITORY,
        sdk_prefix=sdk,
        python=Path(sys.executable),
        backend="mock",
        environment={
            "CUDA_HOME": str(tmp_path / "other-cuda"),
            "CUDAToolkit_ROOT": str(toolkit),
        },
    )
    # Torch's legacy FindCUDA consumes this cache variable rather than the
    # newer CUDAToolkit_ROOT or CUDA_HOME environment variables.
    cache = (tmp_path / "workflow/build/CMakeCache.txt").read_text().splitlines()
    selected = [
        line.split("=", 1)[1]
        for line in cache
        if line.startswith("CUDA_TOOLKIT_ROOT_DIR:")
    ]
    assert selected == [str(toolkit)]


@pytest.mark.parametrize(
    "source_layout",
    [
        "external-relative",
        "external-absolute",
        "project-relative",
        "project-absolute",
    ],
)
def test_application_staging_preserves_custom_workflow_source(
    sdk: Path, tmp_path: Path, monkeypatch, source_layout: str
):
    from pnms_app import build, model
    from pnms_app.common import read_json, write_json
    from pnms_app.config import load_project

    app = tmp_path / "app"
    app.mkdir()
    model_config = make_config(app)
    source = (
        app / "native"
        if source_layout.startswith("project-")
        else tmp_path / "shared-native"
    )
    source.mkdir()
    shared = source.parent / "shared-inputs"
    shared.mkdir()
    (shared / "response.hpp").write_text(
        'inline constexpr auto response = R"({"values":[11,12,13,14]})";\n'
    )
    (shared / "settings.cmake").write_text(
        'target_include_directories(source_probe PRIVATE "${CMAKE_CURRENT_LIST_DIR}")\n'
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(SourceProbe LANGUAGES CXX)\n"
        "add_library(source_probe STATIC workflow.cpp)\n"
        "set_target_properties(source_probe PROPERTIES POSITION_INDEPENDENT_CODE ON)\n"
        "target_link_libraries(source_probe PRIVATE PhysicsNeMoWorkflow::authoring)\n"
        'include("${CMAKE_CURRENT_LIST_DIR}/../shared-inputs/settings.cmake")\n'
    )
    (source / "workflow.cpp").write_text(
        '#include "physicsnemo/workflow/v1/authoring.hpp"\n'
        '#include "response.hpp"\n'
        "namespace physicsnemo::workflow::v1 {\n"
        "class SourceProbe final : public Workflow {\n"
        " public:\n"
        '  Outcome run(const Request&, Context&) override { return {response, "{}"}; }\n'
        "};\n"
        "std::unique_ptr<Workflow> create_custom_workflow() {\n"
        "  return std::make_unique<SourceProbe>();\n"
        "}\n"
        "}\n"
    )
    source_dir = (
        str(source)
        if source_layout.endswith("-absolute")
        else "native"
        if source_layout == "project-relative"
        else "../shared-native"
    )
    document = {
        "schema_version": 1,
        "name": "source-probe",
        "runtime": "cpp",
        "models": {"primary": {"project": "package/model.json"}},
        "workflow": {
            "template": "custom",
            "source_dir": source_dir,
            "target": "source_probe",
        },
    }
    write_json(app / "pnms.yaml", document)
    write_json(
        app / "fixtures/self-test.json",
        {"primary": model_config["models"]["primary"]["self_test"]},
    )
    authored_bytes = (app / "pnms.yaml").read_bytes()
    project = load_project(app, REPOSITORY)
    project["backend"] = "mock"

    def prepare_models(project, output, **kwargs):
        if source_layout.startswith("project-"):
            (source / "workflow.cpp").write_text("#error authored source changed\n")
        package = output / "primary/model/package"
        shutil.copytree(Path(project["root"]) / "package", package)
        return {
            "models": {
                "primary": {
                    "output_root": str(output / "primary"),
                    "package": str(package),
                    "compilation_source_sha": "a" * 40,
                }
            },
            "environment": {
                "python": sys.executable,
                "sdk_prefix": str(sdk),
                "stack": {},
                "gpu": {},
            },
        }

    class WorkflowInstalled(Exception):
        pass

    installed = []
    build_workflow = native.build_workflow

    def workflow_stage(*args, **kwargs):
        installed.append(build_workflow(*args, **kwargs))
        # The production bundle selects CUDA; stop after the real CPU build.
        raise WorkflowInstalled

    monkeypatch.setattr(model, "build_models", prepare_models)
    monkeypatch.setattr(native, "build_workflow", workflow_stage)
    with pytest.raises(WorkflowInstalled):
        build.build_application(project, repository=REPOSITORY)
    workflow = installed[0]
    assert not workflow["reused"]
    assert project["workflow"] == document["workflow"]
    assert (app / "pnms.yaml").read_bytes() == authored_bytes
    receipt = read_json(Path(workflow["install_root"]).parents[1] / "build.json")
    assert receipt["application"] == document
    cache = (Path(workflow["install_root"]).parent / "build/CMakeCache.txt").read_text()
    expected_source = (
        Path(workflow["install_root"]).parents[1] / "inputs/native"
        if source_layout.startswith("project-")
        else source
    )
    assert f"PNMS_CUSTOM_WORKFLOW_SOURCE_DIR:PATH={expected_source.as_posix()}" in (
        cache.replace("\\", "/")
    )

    sealed = tmp_path / "sealed"
    shutil.copytree(workflow["install_root"], sealed / "workflow")
    config = make_config(sealed)
    config["adapter"] = "custom.v1"
    del config["tensor_adapter"]
    config["models"]["primary"]["package_path"] = "package"
    write_json(sealed / "configuration/workflow.json", config)
    write_json(
        sealed / "bundle.json",
        {
            "schema_version": 1,
            "workflow": {
                "config": "configuration/workflow.json",
                "runner": f"workflow/bin/{RUNNER_NAME}",
            },
        },
    )
    shutil.rmtree(source)
    shutil.rmtree(shared)
    shutil.rmtree(Path(workflow["install_root"]).parents[1] / "inputs")
    parameters = tmp_path / "parameters.json"
    write_json(parameters, {"values": [1, 2, 3, 4]})
    executed = native.run_bundle(sealed, parameters, tmp_path / "source-free-run")
    assert executed["result"]["payload"] == {"values": [11, 12, 13, 14]}


def test_run_bundle_materializes_paths_without_mutating_release(
    workflow: Path, tmp_path: Path
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copytree(workflow, bundle / "workflow")
    config = make_config(bundle)
    config["models"]["primary"]["package_path"] = "package"
    (bundle / "configuration").mkdir()
    config_path = bundle / "configuration/workflow.json"
    config_bytes = json.dumps(config).encode()
    config_path.write_bytes(config_bytes)
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": {
                    "config": "configuration/workflow.json",
                    "runner": f"workflow/bin/{RUNNER_NAME}",
                },
            }
        )
    )
    values = tmp_path / "parameters.json"
    values.write_text(json.dumps({"values": [5, 6, 7, 8]}))
    result = native.run_bundle(bundle, values, tmp_path / "attempt")
    assert result.get("result", {}).get("payload") == {"values": [5, 6, 7, 8]}, (
        "the direct application path must execute the installed host"
    )
    assert config_path.read_bytes() == config_bytes
    assert Path(result["startup_path"]).is_file()
    with pytest.raises((ValueError, FileExistsError), match="exists"):
        native.run_bundle(bundle, values, tmp_path / "attempt")


@pytest.mark.parametrize(
    "layout", ["nested", "relative", "dotdot", "output-symlink", "bundle-symlink"]
)
@pytest.mark.parametrize("adapter", ["custom.v1", "transolver_surface"])
def test_run_bundle_rejects_output_inside_sealed_bundle_before_mutation(
    tmp_path: Path, monkeypatch, layout: str, adapter: str
):
    from pnms_app import bundle, cfd
    from pnms_app.common import digest, inventory, write_json

    selected = tmp_path / "bundle"
    write_json(selected / "workflow.json", {"models": {}})
    (selected / RUNNER_NAME).write_bytes(b"runner")
    write_json(
        selected / "bundle.json",
        {
            "adapter": adapter,
            "workflow": {"config": "workflow.json", "runner": RUNNER_NAME},
        },
    )
    files = inventory(selected)
    write_json(
        selected / "inventory.json",
        {"schema_version": 1, "files": files, "bundle_digest": digest(files)},
    )
    parameters = tmp_path / "parameters.json"
    write_json(parameters, {})
    output = selected / "new-parent/attempt"
    location = selected
    if layout == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("bundle/new-parent/attempt")
    elif layout == "dotdot":
        output = tmp_path / "unused/../bundle/new-parent/attempt"
    elif layout in {"output-symlink", "bundle-symlink"}:
        alias = tmp_path / "bundle-alias"
        alias.symlink_to(selected, target_is_directory=True)
        if layout == "output-symlink":
            output = alias / "new-parent/attempt"
        else:
            location = alias
    executed = []

    def execute(arguments, *, log, environment):
        executed.append(arguments)
        Path(arguments[arguments.index("--output") + 1]).write_text('{"payload": {}}')

    monkeypatch.setattr(native, "_execute", execute)
    monkeypatch.setattr(
        cfd, "run_bundle", lambda *args, **kwargs: executed.append(args) or {}
    )
    seal = bundle.verify(selected)
    before = inventory(selected)
    paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="outside.*bundle"):
        native.run_bundle(location, parameters, output)
    assert not executed
    assert bundle.verify(selected) == seal
    assert inventory(selected) == before
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == paths


def test_runner_startup_failure_never_executes_request(workflow: Path, tmp_path: Path):
    configuration = make_config(tmp_path)
    configuration["models"]["primary"]["self_test"]["expected_outputs"]["output"][0] = (
        999
    )
    config = tmp_path / "bad-goldens.json"
    config.write_text(json.dumps(configuration))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "request_id": "should-not-run",
                "parameters": {"values": [1, 2, 3, 4]},
                "inputs": {},
                "output_dir": str(tmp_path / "outputs"),
            }
        )
    )
    result = subprocess.run(
        [
            str(workflow / "bin" / RUNNER_NAME),
            "--config",
            str(config),
            "--request",
            str(request),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "self_test output mismatch" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "outputs").exists()


def test_installed_runner_is_relocatable(workflow: Path, tmp_path: Path):
    relocated = tmp_path / "relocated"
    shutil.copytree(workflow, relocated)
    configuration = make_config(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration))
    result = json.loads(
        command(
            relocated / "bin" / RUNNER_NAME, "--config", config, "--self-test"
        ).stdout
    )
    assert result["status"] == "passed"


def test_run_bundle_rejects_escape_before_creating_attempt(
    workflow: Path, tmp_path: Path
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copytree(workflow, bundle / "workflow")
    configuration = make_config(bundle)
    configuration["models"]["primary"]["package_path"] = "../private-package"
    (bundle / "workflow.json").write_text(json.dumps(configuration))
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": {
                    "config": "workflow.json",
                    "runner": f"workflow/bin/{RUNNER_NAME}",
                },
            }
        )
    )
    parameters = tmp_path / "request.json"
    parameters.write_text('{"values":[1,2,3,4]}')
    with pytest.raises(ValueError, match="relative and contained"):
        native.run_bundle(bundle, parameters, tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_native_loader_uses_selected_sdk_instead_of_global_libraries(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "sdk/lib").mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unrelated/cuda/lib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/unrelated/torch/lib")
    environment = native.loader_environment(sdk_prefix=tmp_path / "sdk")
    key = (
        "PATH"
        if sys.platform == "win32"
        else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    assert environment[key].split(os.pathsep)[0] == str(tmp_path / "sdk/lib")
    assert "/unrelated/" not in environment[key]


def test_windows_loader_preserves_system_path_after_selected_sdk(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        native,
        "os",
        SimpleNamespace(environ={"PATH": "host-tools;system"}, pathsep=";"),
    )
    selected = native.loader_environment(sdk_prefix=sdk)
    assert selected["PATH"].split(";") == [str(sdk / "bin"), "host-tools", "system"]


@pytest.mark.parametrize("contract", ["configuration", "artifacts", "loader"])
def test_windows_workflow_build_selects_release_and_records_dll_path(
    tmp_path, monkeypatch, contract
):
    from pnms_app import compatibility, workflow_cache

    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        native, "loader_environment", lambda *args, **kwargs: {"PATH": "sdk;system"}
    )
    monkeypatch.setattr(workflow_cache, "prepare_configuration", lambda *args: None)
    monkeypatch.setattr(workflow_cache, "configured_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        workflow_cache, "fingerprint", lambda *args, **kwargs: {"key": "windows"}
    )
    monkeypatch.setattr(workflow_cache, "restore", lambda *args: None)
    monkeypatch.setattr(workflow_cache, "publish", lambda *args: None)
    monkeypatch.setattr(compatibility, "capture", lambda *args, **kwargs: {})
    commands = []
    install = tmp_path / "workflow/install"

    def execute(arguments, *, log, environment):
        commands.append(arguments)
        if "--install" in arguments:
            (install / "bin").mkdir(parents=True)
            (install / "bin/pnms-native-run.exe").write_bytes(b"runner")
            (install / "bin/physicsnemo_workflow.dll").write_bytes(b"workflow")

    monkeypatch.setattr(native, "_execute", execute)
    built = native.build_workflow(
        {"root": tmp_path},
        tmp_path / "workflow",
        repository=REPOSITORY,
        sdk_prefix=tmp_path / "sdk",
        python=Path(sys.executable),
        backend="mock",
    )
    if contract == "configuration":
        for operation in ("--build", "--install"):
            selected = next(command for command in commands if operation in command)
            assert selected[selected.index("--config") + 1] == "Release"
    elif contract == "artifacts":
        assert built["runner"] == str(install / "bin/pnms-native-run.exe")
        assert built["library"] == str(install / "bin/physicsnemo_workflow.dll")
    else:
        assert built["loader_environment"] == {"PATH": "sdk;system"}


def test_windows_bundle_loads_sealed_dll_before_selected_stack(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "workflow/bin").mkdir(parents=True)
    (bundle / "workflow/bin/pnms-native-run.exe").write_bytes(b"runner")
    (bundle / "workflow/bin/physicsnemo_workflow.dll").write_bytes(b"workflow")
    (bundle / "workflow.json").write_text(json.dumps({"models": {}}))
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "workflow": {
                    "config": "workflow.json",
                    "runner": "workflow/bin/pnms-native-run.exe",
                }
            }
        )
    )
    parameters = tmp_path / "parameters.json"
    parameters.write_text("{}")
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        native, "os", SimpleNamespace(environ={"PATH": "system"}, pathsep=";")
    )
    captured = {}

    def execute(arguments, *, log, environment):
        captured.update(environment)
        Path(arguments[arguments.index("--output") + 1]).write_text('{"payload": {}}')

    monkeypatch.setattr(native, "_execute", execute)
    native.run_bundle(
        bundle,
        parameters,
        tmp_path / "attempt",
        environment={"PATH": "selected-sdk;system"},
    )
    assert captured["PATH"].split(";") == [
        str(bundle / "workflow/bin"),
        str(bundle / "workflow/lib"),
        str(bundle / "workflow/lib64"),
        "selected-sdk",
        "system",
    ]


def ingress_bundle(workflow: Path, root: Path, rule: dict) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    shutil.copytree(workflow, bundle / "workflow")
    configuration = make_config(bundle)
    configuration["models"]["primary"]["package_path"] = "package"
    (bundle / "workflow.json").write_text(json.dumps(configuration))
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": {
                    "config": "workflow.json",
                    "runner": f"workflow/bin/{RUNNER_NAME}",
                },
                "input_files": {"payload": rule},
            }
        )
    )
    return bundle


@pytest.mark.parametrize("invalid", ["missing", "unknown", "media", "oversized"])
def test_direct_native_ingress_rejects_before_execution(
    workflow: Path, tmp_path: Path, invalid
):
    bundle = ingress_bundle(
        workflow,
        tmp_path,
        {
            "required": invalid != "unknown",
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 8 if invalid == "oversized" else 16,
        },
    )
    request = tmp_path / "request.json"
    request.write_text('{"values":[1,2,3,4]}')
    upload = tmp_path / ("input.png" if invalid == "media" else "input.bin")
    upload.write_bytes(bytes(16))
    inputs = (
        {}
        if invalid == "missing"
        else {
            "undeclared" if invalid == "unknown" else "payload": upload,
        }
    )
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="required|undeclared|media type|size limit"):
        native.run_bundle(bundle, request, output, inputs=inputs)
    assert not output.exists(), "rejected ingress must not start an inference attempt"


@pytest.mark.parametrize("supplied", [False, True])
def test_direct_native_ingress_accepts_declared_optional_files(
    workflow: Path, tmp_path: Path, supplied
):
    bundle = ingress_bundle(
        workflow,
        tmp_path,
        {
            "required": False,
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 16,
        },
    )
    request = tmp_path / "request.json"
    request.write_text('{"values":[1,2,3,4]}')
    upload = tmp_path / "input.bin"
    upload.write_bytes(bytes(16))
    result = native.run_bundle(
        bundle,
        request,
        tmp_path / "attempt",
        inputs={"payload": upload} if supplied else {},
    )
    assert result["result"]["payload"] == {"values": [1, 2, 3, 4]}
    envelope = json.loads(
        (Path(result["output_dir"]) / ".pnms-run/request.json").read_text()
    )
    assert set(envelope["inputs"]) == ({"payload"} if supplied else set())
    if supplied:
        assert envelope["input_metadata"]["payload"]["size_bytes"] == 16
        assert (
            envelope["input_metadata"]["payload"]["media_type"]
            == "application/octet-stream"
        )
