"""Framework-free contracts for adopting the existing native CFD deployment."""

import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app.common import file_identity, read_json, write_json  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402


def test_existing_native_cfd_plugin_has_a_local_common_application_path():
    from pnms_app.config import load_project

    project = load_project(
        ROOT / "plugins/physicsnemo-cfd-transolver-native", repository=ROOT
    )
    assert project["runtime"] == "cpp"
    assert project["workflow"]["template"] == "transolver_surface"
    assert project["backend"] == "aoti"
    assert project["serve"]["memory_mb"] == 60000


@pytest.fixture
def imported(tmp_path):
    root = tmp_path / "compiled"
    packages, assets = [], []
    for points in (60000, 8095):
        package = root / f"points-{points}"
        write_json(
            package / "model.json",
            {
                "format_version": 1,
                "inputs": [
                    {"name": "fx", "shape": [1, points, 2], "dtype": "float32"},
                    {"name": "embedding", "shape": [1, points, 6], "dtype": "float32"},
                ],
                "outputs": [
                    {"name": "output", "shape": [1, points, 4], "dtype": "float32"}
                ],
                "artifacts": [
                    {
                        "backend": "aoti",
                        "target": "cuda",
                        "precision": "fp32",
                        "path": "model.pt2",
                        "correctness_profile": {"name": "aten-boundary-exact-v2"},
                    }
                ],
            },
        )
        (package / "model.pt2").write_bytes(
            b"exact compiled package " + str(points).encode()
        )
        packages.append(str(package))
    stats = root / "stats/global_stats.json"
    write_json(stats, {"statistics": "pinned"})
    startup = root / "startup/startup.json"
    fixtures = []
    for points in (60000, 8095):
        record = {"point_count": points, "inputs": {}, "rtol": 0.0, "atol": 0.0}
        for name, width in (("fx", 2), ("embedding", 6), ("expected", 4)):
            path = startup.parent / f"{points}-{name}.f32"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\0" * (points * width * 4))
            tensor = {"path": path.name, "shape": [1, points, width]}
            if name == "expected":
                record["expected_output"] = tensor
            else:
                record["inputs"][name] = tensor
        fixtures.append(record)
    write_json(startup, {"schema_version": 1, "models": fixtures})
    for path in sorted(root.rglob("*")):
        if path.is_file():
            assets.append({"path": str(path), **file_identity(path)})
    config = {
        "schema_version": 1,
        "workflow_id": "physicsnemo-cfd-transolver-native",
        "executor_class": "native.transolver.aoti",
        "backend": "aoti",
        "device_index": 0,
        "compute_capability": [9, 0],
        "target": {
            "profile": "h100-sxm-80gb-linux-amd64-v1",
            **{
                key: value
                for key, value in read_json(
                    ROOT
                    / "native-inference/deploy/targets/h100-sxm-80gb-linux-amd64-v1.json"
                ).items()
                if key in ("compilation", "deployment")
            },
        },
        "packages": packages,
        "model_build_sources": {"full": "a" * 40, "tail": "c" * 40},
        "stats_path": str(stats),
        "startup_fixture_path": str(startup),
        "model_assets": assets,
        "block_size": 60000,
        "artifact_root": "/old/artifacts",
        "input_cache": "/old/cache",
        "case": {
            "case_id": "run_1",
            "point_count": 8828095,
            "mesh": {"url": "https://example.test/mesh", **file_identity(stats)},
            "geometry": {
                "url": "https://example.test/geometry",
                **file_identity(stats),
            },
        },
        "request_timeout_seconds": 600,
        "drain_timeout_seconds": 60,
    }
    path = root / "deployment.json"
    write_json(path, config)
    project = {
        "root": tmp_path / "application",
        "name": config["workflow_id"],
        "runtime": "cpp",
        "backend": "aoti",
        "workflow": {"template": "transolver_surface", "deployment": str(path)},
        "build": {},
        "serve": {"memory_mb": 60000},
    }
    project["root"].mkdir()
    shutil.copy2(
        ROOT / "plugins/physicsnemo-cfd-transolver-native/plugin.yaml",
        project["root"] / "plugin.yaml",
    )
    return project, config, path


def test_compiled_cfd_import_is_verified_without_builder_or_lepton(
    imported, monkeypatch
):
    from pnms_app import cfd

    monkeypatch.setattr(
        cfd.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("static import launched a process"),
    )
    project, config, _ = imported
    selected = cfd.import_deployment(project)
    assert selected["configuration"] == config
    assert set(selected["packages"]) == {"full", "tail"}
    assert selected["model_build_sources"] == config["model_build_sources"]
    assert selected["recompiled"] is False


@pytest.mark.parametrize(
    "mutation", ["bytes", "missing-file", "extra-file", "shape", "origin", "startup"]
)
def test_compiled_cfd_import_rejects_unproven_payload_before_execution(
    imported, mutation
):
    from pnms_app import cfd

    project, config, path = imported
    package = Path(config["packages"][0])
    if mutation == "bytes":
        (package / "model.pt2").write_bytes(b"replaced")
    elif mutation == "missing-file":
        (package / "model.pt2").unlink()
    elif mutation == "extra-file":
        (package / "unverified.so").write_bytes(b"extra native library")
    elif mutation == "shape":
        manifest = read_json(package / "model.json")
        manifest["inputs"][0]["shape"][1] = 75
        write_json(package / "model.json", manifest)
        for item in config["model_assets"]:
            if item["path"] == str(package / "model.json"):
                item.update(file_identity(package / "model.json"))
    elif mutation == "origin":
        config.pop("model_build_sources")
    else:
        config.pop("startup_fixture_path")
    write_json(path, config)
    with pytest.raises(ValueError):
        cfd.import_deployment(project)


def test_cfd_bundle_is_relocatable_and_materializes_only_schema_one_runner(
    imported, tmp_path
):
    from pnms_app import bundle, cfd

    project, config, _ = imported
    installed = tmp_path / "installed"
    for relative in (
        "bin/physicsnemo-transolver",
        "lib/libtransolver_workflow.so",
        "lib/libphysicsnemo_workflow.so",
    ):
        path = installed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"compiled native consumer")
    workflow = {"install_root": str(installed)}
    source = cfd.import_deployment(project)
    location = tmp_path / "bundle"
    cfd.assemble(
        project,
        source,
        workflow,
        location,
        repository=ROOT,
        environment={"python": sys.executable, "gpu": {"name": "NVIDIA H100"}},
        source_sha="b" * 40,
    )
    shutil.rmtree(Path(config["stats_path"]).parents[1])
    moved = tmp_path / "moved"
    location.rename(moved)
    bundle.verify(moved)
    actual = cfd.materialize_deployment(
        moved,
        tmp_path / "serve",
        artifact_root=tmp_path / "artifacts",
        input_cache=tmp_path / "inputs",
    )
    assert actual["schema_version"] == 1
    assert actual["packages"] == [
        str(moved / "models/full"),
        str(moved / "models/tail"),
    ]
    assert actual["model_build_sources"] == {"full": "a" * 40, "tail": "c" * 40}
    assert actual["stats_path"] == str(moved / "stats/global_stats.json")
    assert actual["artifact_root"] == str(tmp_path / "artifacts")
    assert "workflow" not in actual
    assert all(
        Path(item["path"]).is_relative_to(moved) for item in actual["model_assets"]
    )
    assert read_json(moved / "bundle.json")["source_sha"] == "b" * 40


def test_cfd_preset_never_claims_fresh_compilation_without_local_import(imported):
    from pnms_app import cfd

    project, _, _ = imported
    project["workflow"].pop("deployment")
    with pytest.raises(ValueError, match="compiled.*import|deployment|not.*compil"):
        cfd.import_deployment(project)


def test_cfd_bundle_capture_rejects_payload_changed_after_import(imported, tmp_path):
    from pnms_app import cfd

    project, config, _ = imported
    source = cfd.import_deployment(project)
    Path(config["packages"][0], "model.pt2").write_bytes(b"changed after validation")
    installed = tmp_path / "installed"
    installed.mkdir()
    with pytest.raises(ValueError, match="changed"):
        cfd.assemble(
            project,
            source,
            {"install_root": str(installed)},
            tmp_path / "bundle",
            repository=ROOT,
            environment={},
            source_sha="b" * 40,
        )


@pytest.fixture(scope="module")
def domain_cli(tmp_path_factory):
    """Compile the real CLI front end against a CPU instrumented Runner ABI."""
    compiler = shutil.which("cl" if sys.platform == "win32" else "c++")
    assert compiler, "the CPU CLI contract check requires a C++20 compiler"
    root = tmp_path_factory.mktemp("cfd-cli")
    includes = root / "include"
    for name in (
        "ATen/BlasBackend.h",
        "ATen/Context.h",
        "ATen/Parallel.h",
        "c10/core/InferenceMode.h",
        "torch/torch.h",
        "vtkVersion.h",
    ):
        path = includes / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    path = includes / "nlohmann/json.hpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("namespace nlohmann { struct json {}; }\n")
    path = includes / "transolver_workflow/workflow.hpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#include <string>
namespace transolver_workflow {
inline std::string parse_domain(const std::string& value) { return value; }
inline std::string to_string(const std::string& value) { return value; }
}
""")
    mock = root / "runner.cpp"
    mock.write_text("""#include "transolver_workflow/runner.hpp"
#include <cstdlib>
#include <fstream>
namespace transolver_workflow {
class Runner::Impl {};
static void event(const std::string& value) {
  std::ofstream(std::getenv("PNMS_CFD_TRACE"), std::ios::app) << value << "\\n";
}
Runner::Runner(RunnerConfig config) : impl_(std::make_unique<Impl>()) {
  event("construct|" + config.math_profile + "|" + config.startup_fixture_path + "|" + std::to_string(config.packages.size()));
}
Runner::~Runner() = default;
std::string Runner::self_test(const CancellationToken&) {
  event("self-test");
  return R"({"status":"passed","policy":"expected-fp32-v1","session_load_count":2,"models":[{"point_count":60000},{"point_count":8095}]})";
}
RunnerResult Runner::run(const RunnerRequest&, const CancellationToken&) {
  event("run"); return {};
}
}
""")
    executable = root / (
        "physicsnemo-transolver.exe"
        if sys.platform == "win32"
        else "physicsnemo-transolver"
    )
    options = (
        ["/nologo", "/std:c++20", "/EHsc", "/Fe:" + str(executable)]
        if sys.platform == "win32"
        else ["-std=c++20", "-o", str(executable)]
    )
    if sys.platform == "win32":
        # Adapt the Linux CLI's environment calls only in this CPU test build.
        shim = root / "posix_environment.hpp"
        shim.write_text("""#include <cstdlib>
inline int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite && std::getenv(name)) return 0;
  return _putenv_s(name, value);
}
""")
        options.append("/FI" + str(shim))
    subprocess.run(
        [
            compiler,
            *options,
            "-I" + str(includes),
            "-I" + str(ROOT / "native-inference/workflows/transolver/include"),
            str(ROOT / "native-inference/workflows/transolver/src/main.cpp"),
            str(mock),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


@pytest.mark.parametrize("startup_only", [False, True])
def test_domain_cli_uses_one_runner_for_authored_startup_and_provider_math(
    domain_cli, tmp_path, startup_only
):
    report, trace = tmp_path / "startup.json", tmp_path / "trace"
    args = [
        str(domain_cli),
        "--package",
        "full",
        "--package",
        "tail",
        "--domain",
        "surface",
        "--stats",
        "stats.json",
        "--math-profile",
        "provider-default-v1",
        "--startup-fixture",
        "goldens.json",
        "--startup-report",
        str(report),
    ]
    args += (
        ["--startup-only", "true"]
        if startup_only
        else [
            "--mesh",
            "mesh.vtp",
            "--stl",
            "geometry.stl",
            "--physical-output",
            str(tmp_path / "physical.bin"),
        ]
    )
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env={**os.environ, "PNMS_CFD_TRACE": str(trace)},
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text().splitlines() == [
        "construct|provider-default-v1|goldens.json|2",
        "self-test",
    ] + ([] if startup_only else ["run"])
    assert read_json(report)["status"] == "passed"


def test_domain_cli_preserves_existing_strict_math_default(domain_cli, tmp_path):
    trace = tmp_path / "trace"
    result = subprocess.run(
        [
            str(domain_cli),
            "--package",
            "full",
            "--domain",
            "surface",
            "--mesh",
            "mesh",
            "--stl",
            "stl",
            "--physical-output",
            "physical",
            "--stats",
            "stats",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PNMS_CFD_TRACE": str(trace)},
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text().splitlines() == ["construct|strict-native-v1||1", "run"]


def test_common_run_dispatches_cfd_to_its_existing_runner(
    imported, tmp_path, monkeypatch
):
    from pnms_app import cfd, native

    project, _, _ = imported
    location = tmp_path / "bundle"
    write_json(location / "bundle.json", {"adapter": "transolver_surface"})
    forwarded = []
    monkeypatch.setattr(
        cfd,
        "run_bundle",
        lambda *a, **kw: forwarded.append((a, kw)) or {"status": "domain-ran"},
    )
    result = native.run_bundle(
        location,
        tmp_path / "request.json",
        tmp_path / "outputs",
        inputs={"mesh": Path("mesh"), "geometry": Path("geometry")},
        environment={"SELECTED": "1"},
        request_id="request",
    )
    assert result == {"status": "domain-ran"}
    assert forwarded == [
        (
            (location.resolve(), tmp_path / "request.json", tmp_path / "outputs"),
            {
                "inputs": {"mesh": Path("mesh"), "geometry": Path("geometry")},
                "environment": {"SELECTED": "1"},
                "request_id": "request",
            },
        )
    ]


def test_cfd_provider_admission_inspects_domain_library(tmp_path, monkeypatch):
    from pnms_app import admission, compatibility

    location = tmp_path / "bundle"
    library = location / "workflow/lib/libtransolver_workflow.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"domain binary")
    built = {
        "root": str(tmp_path),
        "bundle": "bundle",
        "workflow": {"library_name": library.name, "compatibility": {"domain": True}},
        "environment": {"python": sys.executable},
    }
    calls = []
    monkeypatch.setattr(
        compatibility,
        "verify",
        lambda expected, **kw: calls.append((expected, kw)) or {"status": "verified"},
    )
    assert admission.verify_providers(
        built, environment={}, output=tmp_path / "providers.json"
    ) == {"status": "verified"}
    assert calls[0][1]["library"] == library


def test_cfd_image_uses_domain_library_and_cpp_domain_linkage(tmp_path, monkeypatch):
    from pnms_app import bundle, images
    from test_pnms_images import Docker

    docker = Docker()
    monkeypatch.setattr(bundle, "verify", lambda *a: {})
    # Source verification has its own Git/export regression coverage; this
    # fixture exercises only the selected domain library and Docker arguments.
    monkeypatch.setattr(images, "source_context", lambda *a, **kw: {})
    project = {
        "name": "physicsnemo-cfd-transolver-native",
        "runtime": "cpp",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    built = {
        "runtime": "cpp",
        "adapter": "transolver_surface",
        "bundle": "bundle",
        "bundle_digest": "a" * 64,
        "source_sha": "e" * 40,
        "environment": {"stack": docker.stack},
        "workflow": {
            "library_name": "libtransolver_workflow.so",
            "compatibility": docker.stack,
        },
    }
    actual = images.build_image(
        project, built, root=tmp_path, repository=ROOT, image="cfd:local", runner=docker
    )
    assert actual["status"] == "complete"
    probes = [cmd for cmd in docker.commands if images.STACK_PROBE in cmd]
    assert len(probes) == 2
    assert all(
        cmd[-1] == "/opt/physicsnemo-release/workflow/lib/libtransolver_workflow.so"
        for cmd in probes
    )
    command = next(cmd for cmd in docker.commands if "--iidfile" in cmd)
    assert "PNMS_NATIVE_ADAPTER=transolver_surface" in command


def test_cfd_image_content_probe_loads_selected_domain_runner(tmp_path):
    from pnms_app import cfd, images  # noqa: F401 - preserve the recorded RED test AST
    from pnms_app.common import digest, inventory

    location = tmp_path / "bundle"
    runner = location / "workflow/bin/physicsnemo-transolver"
    runner.parent.mkdir(parents=True)
    runner = python_executable(
        runner,
        "from pathlib import Path\nimport sys\n"
        "print(f'usage: {Path(sys.argv[0]).name} --package PATH')\n",
    )
    write_json(
        location / "bundle.json",
        {
            "runtime": "cpp",
            "source_sha": "e" * 40,
            "adapter": "transolver_surface",
            "workflow": {"runner": runner.relative_to(location).as_posix()},
        },
    )
    files = inventory(location)
    write_json(
        location / "inventory.json",
        {"schema_version": 1, "files": files, "bundle_digest": digest(files)},
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            images.CONTENT_PROBE,
            str(location),
            digest(files),
            "e" * 40,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_tensorrt_cfd_bundle_routes_only_to_selected_executor(imported, tmp_path):
    from pnms_app import cfd
    import yaml

    project, config, path = imported
    project["backend"] = config["backend"] = "tensorrt"
    config["executor_class"] = "native.transolver.tensorrt"
    for package in config["packages"]:
        manifest_path = Path(package) / "model.json"
        manifest = read_json(manifest_path)
        manifest["artifacts"][0].update(
            backend="tensorrt",
            correctness_profile={"name": "provider-default-exact-v1"},
        )
        write_json(manifest_path, manifest)
        next(
            item
            for item in config["model_assets"]
            if item["path"] == str(manifest_path)
        ).update(file_identity(manifest_path))
    write_json(path, config)
    installed = tmp_path / "installed"
    installed.mkdir()
    location = tmp_path / "bundle"
    cfd.assemble(
        project,
        cfd.import_deployment(project),
        {"install_root": str(installed)},
        location,
        repository=ROOT,
        environment={},
        source_sha="b" * 40,
    )
    manifest = yaml.safe_load(
        (
            location
            / "configuration/plugin/physicsnemo-cfd-transolver-native/plugin.yaml"
        ).read_text()
    )
    assert manifest["runtime"]["executor_class"] == "native.transolver.tensorrt"
    assert (
        next(
            stage
            for stage in manifest["pipeline"]["stages"]
            if stage["phase"] == "execute"
        )["queue"]
        == "execute.native.transolver.tensorrt"
    )


def test_cfd_compiler_forwards_selected_cuda_and_tensorrt_toolchain(
    imported, tmp_path, monkeypatch
):
    from pnms_app import cfd, compatibility, native

    project, _, _ = imported
    cuda = tmp_path / "selected/cuda"
    tensorrt = tmp_path / "selected/tensorrt"
    for path in (cuda, tensorrt):
        path.mkdir(parents=True)
    project.update(backend="tensorrt", build={"tensorrt_root": str(tensorrt)})
    environment = {
        "python": sys.executable,
        "sdk_prefix": "/selected/sdk",
        "native_environment": {
            "CUDAToolkit_ROOT": str(cuda),
            "LD_LIBRARY_PATH": "/selected/providers",
        },
    }
    installed = tmp_path / "installed"
    calls, commands = [], []

    def generic(*a, **kw):
        calls.append(kw)
        installed.mkdir()
        return {
            "install_root": str(installed),
            "loader_environment": {"LD_LIBRARY_PATH": "/selected/providers"},
        }

    def execute(args, **kw):
        commands.append((args, kw))
        if "--install" in args:
            for relative in (
                "lib/libtransolver_workflow.so",
                "bin/physicsnemo-transolver",
            ):
                path = installed / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"compiled")

    monkeypatch.setattr(native, "build_workflow", generic)
    monkeypatch.setattr(native, "_execute", execute)
    monkeypatch.setattr(
        cfd.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, '["/selected/torch","/selected/warp"]', ""
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "capture",
        lambda python, library, **kw: {"workflow_binary": file_identity(library)},
    )
    actual = cfd.build_workflow(
        project, tmp_path / "workflow", repository=ROOT, environment=environment
    )
    assert calls[0]["tensorrt_root"] == tensorrt
    configure = commands[0][0]
    assert f"-DCUDA_TOOLKIT_ROOT_DIR={cuda}" in configure
    assert f"-DPNMIR_TENSORRT_ROOT={tensorrt}" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=90" in configure
    assert "-DPNMIR_ENABLE_AOTI=OFF" in configure
    assert "-DPNMIR_ENABLE_TENSORRT=ON" in configure
    assert actual["library_name"] == "libtransolver_workflow.so"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Docker RUN contract executes /bin/sh and POSIX shell programs",
)
@pytest.mark.parametrize(
    "adapter,expected", [("tensor", "1"), ("transolver_surface", "unset")]
)
def test_native_image_cargo_shell_selects_one_worker_family(
    tmp_path, adapter, expected
):
    import re

    source = (ROOT / "application-cli/docker/Dockerfile.native").read_text()
    run = next(
        block
        for block in re.split(r"(?m)^(?=[A-Z]+ )", source)
        if block.startswith("RUN --mount") and "cargo build" in block
    )
    script = re.sub(
        r"--mount=\S+\s*", "", run.removeprefix("RUN ").replace("\\\n", " ")
    )
    target = tmp_path / "cargo-target"
    target.mkdir()
    (target / "previous").write_text("must be removed by the container cleanup")
    assert script.count("/tmp/pnms-image-target") == 2
    script = script.replace("/tmp/pnms-image-target", shlex.quote(str(target)))
    assert "/tmp/pnms-image-target" not in script, (
        "container cleanup must use an owned test directory"
    )
    programs = tmp_path / "bin"
    programs.mkdir()
    python = programs / "python3"
    python.write_text("#!/bin/sh\necho /selected/torch/lib\n")
    cargo = programs / "cargo"
    cargo.write_text(
        "#!/bin/sh\n"
        '[ "$4" = "--target-dir" ] && [ "$5" = "$PNMS_CARGO_TARGET" ] || exit 91\n'
        '[ ! -e "$PNMS_CARGO_TARGET/previous" ] || exit 92\n'
        'printf "%s" "${NATIVE_WORKFLOW_ONLY-unset}" > "$PNMS_CARGO_TRACE"\n'
    )
    for path in (python, cargo):
        path.chmod(0o755)
    trace = tmp_path / "cargo-trace"
    result = subprocess.run(
        ["/bin/sh", "-c", script],
        env={
            **os.environ,
            "PATH": str(programs) + os.pathsep + os.environ["PATH"],
            "PNMS_NATIVE_ADAPTER": adapter,
            "NATIVE_WORKFLOW_ONLY": "1",
            "PNMS_CARGO_TRACE": str(trace),
            "PNMS_CARGO_TARGET": str(target),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text() == expected


def _installed(root):
    for relative in (
        "bin/physicsnemo-transolver",
        "lib/libtransolver_workflow.so",
        "lib/libphysicsnemo_workflow.so",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native compiled consumer")
    return {
        "install_root": str(root),
        "loader_environment": {"LD_LIBRARY_PATH": "/selected/providers"},
        "library_name": "libtransolver_workflow.so",
        "compatibility": {"schema_version": 1},
    }


def _startup_report():
    return {
        "status": "passed",
        "policy": "expected-fp32-v1",
        "session_load_count": 2,
        "models": [{"point_count": points} for points in (60000, 8095)],
    }


def test_cfd_direct_run_preserves_provider_seed_and_all_original_artifacts(
    imported, tmp_path, monkeypatch
):
    from pnms_app import cfd, native

    project, config, _ = imported
    mesh, geometry = tmp_path / "mesh", tmp_path / "geometry"
    for path in (mesh, geometry):
        shutil.copy2(config["stats_path"], path)
    location = tmp_path / "bundle"
    cfd.assemble(
        project,
        cfd.import_deployment(project),
        _installed(tmp_path / "installed"),
        location,
        repository=ROOT,
        environment={},
        source_sha="b" * 40,
    )
    shutil.rmtree(Path(config["stats_path"]).parents[1])
    request = tmp_path / "request.json"
    write_json(request, {"case_id": "run_1", "seed": 42, "capture_inputs": True})
    calls = []

    def execute(args, **kw):
        calls.append((args, kw))

        def option(name):
            return args[args.index(name) + 1]

        assert option("--math-profile") == "provider-default-v1"
        assert option("--seed") == "1615895419752914611"
        assert option("--block-size") == "60000"
        assert option("--mesh") == str(mesh)
        assert option("--stl") == str(geometry)
        assert args.count("--package") == 2
        assert option("--startup-fixture") == str(location / "startup/startup.json")
        write_json(option("--startup-report"), _startup_report())
        write_json(
            option("--metadata"),
            {
                "point_count": 8828095,
                "session_load_count": 2,
                "math_profile": "provider-default-v1",
                "permutation_seed": 1615895419752914611,
            },
        )
        for key in ("--physical-output", "--standardized-output"):
            with Path(option(key)).open("wb") as stream:
                stream.truncate(8828095 * 16)
        for name, size in (
            ("fx.f32", 8),
            ("embedding.f32", 24),
            ("permutation.i64", 8),
            ("source_cell_ids.i64", 8),
        ):
            path = Path(option("--dump-input-dir")) / name
            path.parent.mkdir(exist_ok=True)
            with path.open("wb") as stream:
                stream.truncate(8828095 * size)

    monkeypatch.setattr(native, "_execute", execute)
    result = native.run_bundle(
        location,
        request,
        tmp_path / "run",
        inputs={"mesh": mesh, "geometry": geometry},
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        request_id="case-request",
    )
    assert len(calls) == 1
    assert calls[0][1]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert result["result"]["request_id"] == "case-request"
    payload = result["result"]["payload"]
    assert payload["native"]["source_sha"] == "b" * 40
    assert payload["native"]["model_build_sources"] == {
        "full": "a" * 40,
        "tail": "c" * 40,
    }
    assert {item["name"] for item in payload["artifacts"]} == {
        "physical_fields",
        "standardized_fields",
        "native_metadata",
        "fx.f32",
        "embedding.f32",
        "permutation.i64",
        "source_cell_ids.i64",
    }
    assert read_json(result["result_path"]) == result["result"]


def test_common_cfd_build_imports_without_recompiling_models(
    imported, tmp_path, monkeypatch
):
    from pnms_app import build, cfd, environment_lock, model, native

    project, _, _ = imported
    project["document"] = {
        "schema_version": 1,
        "name": project["name"],
        "runtime": "cpp",
        "workflow": project["workflow"],
    }
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    env = {
        "kind": "local",
        "python": sys.executable,
        "sdk_prefix": "/selected/sdk",
        "gpu": {
            "name": "NVIDIA H100 80GB HBM3",
            "architecture": "sm90",
            "driver_version": "570.195.03",
            "mig_mode": "disabled",
        },
    }
    monkeypatch.setattr(model, "prepare_environment", lambda *a, **kw: env)
    monkeypatch.setattr(
        model,
        "build_models",
        lambda *a, **kw: pytest.fail("import adapter recompiled models"),
    )
    monkeypatch.setattr(environment_lock, "capture", lambda *a: None)
    monkeypatch.setattr(
        environment_lock, "check", lambda *a, **kw: {"status": "matched"}
    )
    monkeypatch.setattr(
        cfd, "build_workflow", lambda *a, **kw: _installed(tmp_path / "installed")
    )
    commands = []

    def execute(args, **kw):
        commands.append(args)
        assert args[args.index("--startup-only") + 1] == "true"
        assert "--mesh" not in args
        write_json(args[args.index("--startup-report") + 1], _startup_report())

    monkeypatch.setattr(native, "_execute", execute)
    actual = build.build_application(project, repository=ROOT, publish_result=False)
    assert actual["status"] == "complete"
    assert actual["checks"] == {
        "model": "passed",
        "application": "passed",
        "http": "not_run",
        "scientific": "not_run",
    }
    assert actual["recompiled"] is False
    assert actual["source_sha"] == "b" * 40
    assert actual["application"] == project["document"]
    assert len(commands) == 1
    assert actual["models"]["full"]["compilation_source_sha"] == "a" * 40
    assert actual["models"]["tail"]["compilation_source_sha"] == "c" * 40


@pytest.mark.parametrize(
    "mutation", ["target", "timeout", "case-input", "unknown-field"]
)
def test_cfd_import_rejects_contracts_the_schema_one_worker_cannot_admit(
    imported, mutation
):
    from pnms_app import cfd

    project, config, path = imported
    if mutation == "target":
        config["target"]["compilation"]["gpu_product"] = "NVIDIA A100"
    elif mutation == "timeout":
        config["request_timeout_seconds"] = 0
    elif mutation == "case-input":
        config["case"]["mesh"]["url"] = "http://insecure.test/mesh"
    else:
        config["startup_timeout_seconds"] = 300
    write_json(path, config)
    with pytest.raises(ValueError):
        cfd.import_deployment(project)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "NVIDIA H100 NVL"),
        ("architecture", "sm80"),
        ("driver_version", "580.0"),
        ("mig_mode", "Enabled"),
    ],
)
def test_import_build_preserves_original_gpu_admission_policy(
    imported, tmp_path, monkeypatch, field, value
):
    from pnms_app import cfd, environment_lock, model, native

    project, _, _ = imported
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    gpu = {
        "name": "NVIDIA H100 80GB HBM3",
        "architecture": "sm90",
        "driver_version": "570.195.03",
        "mig_mode": "Disabled",
        field: value,
    }
    monkeypatch.setattr(model, "prepare_environment", lambda *a, **kw: {"gpu": gpu})
    monkeypatch.setattr(environment_lock, "capture", lambda *a: None)
    monkeypatch.setattr(environment_lock, "check", lambda *a, **kw: {})
    monkeypatch.setattr(
        cfd, "build_workflow", lambda *a, **kw: _installed(tmp_path / "installed")
    )
    monkeypatch.setattr(
        native,
        "_execute",
        lambda args, **kw: write_json(
            args[args.index("--startup-report") + 1], _startup_report()
        ),
    )
    with pytest.raises(ValueError, match="target|driver|MIG"):
        cfd.build_application(project, repository=ROOT, publish_result=False)


def test_frozen_cfd_application_needs_no_authored_sources_or_original_packages(
    imported, tmp_path
):
    from pnms_app import build, cfd
    from pnms_app.config import load_project

    project, config, _ = imported
    state = project["root"] / ".pnms"
    completed = state / "builds/frozen"
    seal = cfd.assemble(
        project,
        cfd.import_deployment(project),
        _installed(tmp_path / "installed"),
        completed / "bundle",
        repository=ROOT,
        environment={},
        source_sha="b" * 40,
    )
    receipt = {
        "schema_version": 1,
        "build_id": "frozen",
        "status": "complete",
        "runtime": "cpp",
        "name": project["name"],
        "adapter": "transolver_surface",
        "source_sha": "b" * 40,
        "bundle": "bundle",
        "bundle_digest": seal["bundle_digest"],
    }
    write_json(completed / "build.json", receipt)
    write_json(
        state / "last-build.json",
        {
            "receipt": "builds/frozen/build.json",
            "identity": file_identity(completed / "build.json"),
        },
    )
    (project["root"] / "plugin.yaml").unlink()
    shutil.rmtree(Path(config["stats_path"]).parents[1])
    frozen = load_project(project["root"], ROOT, require_sources=False)
    assert frozen["workflow"] == {"template": "transolver_surface"}
    assert frozen["models"] == {}
    assert frozen["backend"] == "aoti"
    assert build.last_build(frozen)["build_id"] == "frozen"
    assert build.last_build(frozen)["bundle_digest"] == seal["bundle_digest"]
    write_json(completed / "build.json", {**receipt, "source_sha": "d" * 40})
    with pytest.raises(ValueError, match="receipt changed"):
        load_project(project["root"], ROOT, require_sources=False)


@pytest.fixture
def competing_cfd_workflow_libraries(tmp_path):
    """Two real CPU shared libraries expose the loader's selected backend."""
    compiler = shutil.which("cc")
    assert compiler, "native CLI CPU tests require a C compiler"
    workflow = _installed(tmp_path / "compiled-workflow")
    old = tmp_path / "ambient-workflow"
    old.mkdir()
    for directory, backend in ((old, 0), (Path(workflow["install_root"]) / "lib", 1)):
        source = tmp_path / f"backend-{backend}.c"
        source.write_text(f"int pnms_backend(void) {{ return {backend}; }}\n")
        subprocess.run(
            [
                compiler,
                "-dynamiclib" if sys.platform == "darwin" else "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(directory / "libtransolver_workflow.so"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    workflow["loader_environment"] = {loader: str(old)}
    runner = Path(workflow["install_root"]) / "bin/physicsnemo-transolver"
    runner.write_text(
        f"#!{sys.executable}\n"
        + """import ctypes,json,os,sys
from pathlib import Path
library=ctypes.CDLL("libtransolver_workflow.so")
assert library.pnms_backend()==1, "ambient workflow library displaced the selected bundle"
def option(name):return sys.argv[sys.argv.index(name)+1]
report={"status":"passed","policy":"expected-fp32-v1","session_load_count":2,"models":[{"point_count":points} for points in (60000,8095)],"environment":{"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES")}}
Path(option("--startup-report")).write_text(json.dumps(report))
if "--startup-only" not in sys.argv:
 metadata={"point_count":8828095,"session_load_count":2,"math_profile":"provider-default-v1","permutation_seed":int(option("--seed"))}
 Path(option("--metadata")).write_text(json.dumps(metadata))
 for name in ("--physical-output","--standardized-output"):
  with Path(option(name)).open("wb") as stream:stream.truncate(8828095*16)
"""
    )
    runner.chmod(0o755)
    return workflow, old, loader


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="Unix native loader"
)
@pytest.mark.parametrize("entrypoint", ["build-startup", "direct-run"])
@pytest.mark.parametrize("binding", ["GPU-selected", "1"])
def test_cfd_binds_its_real_shared_library_before_an_ambient_workflow(
    imported,
    competing_cfd_workflow_libraries,
    monkeypatch,
    tmp_path,
    entrypoint,
    binding,
):
    from pnms_app import build, cfd, environment_lock, model, native

    project, config, _ = imported
    workflow, ambient, loader = competing_cfd_workflow_libraries
    monkeypatch.setenv(loader, str(ambient))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other,GPU-selected")
    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    if entrypoint == "build-startup":
        monkeypatch.setattr(
            model,
            "prepare_environment",
            lambda *a, **kw: {
                "kind": "local",
                "python": sys.executable,
                "sdk_prefix": "/selected/sdk",
                "gpu": {
                    "name": "NVIDIA H100 80GB HBM3",
                    "architecture": "sm90",
                    "driver_version": "570.195.03",
                    "mig_mode": "disabled",
                    "binding": binding,
                },
                "process_env": {**os.environ, "CUDA_VISIBLE_DEVICES": binding},
            },
        )
        monkeypatch.setattr(environment_lock, "capture", lambda *a: None)
        monkeypatch.setattr(environment_lock, "check", lambda *a, **kw: {})
        monkeypatch.setattr(cfd, "build_workflow", lambda *a, **kw: workflow)
        result = build.build_application(project, repository=ROOT, publish_result=False)
        assert result["status"] == "complete"
        startup = read_json(Path(result["root"]) / "checks/startup.json")
        assert startup["status"] == "passed"
    else:
        location = tmp_path / "frozen-bundle"
        cfd.assemble(
            project,
            cfd.import_deployment(project),
            workflow,
            location,
            repository=ROOT,
            environment={},
            source_sha="b" * 40,
        )
        mesh, geometry = tmp_path / "mesh", tmp_path / "geometry"
        for path in (mesh, geometry):
            shutil.copy2(config["stats_path"], path)
        request = tmp_path / "request.json"
        write_json(request, {"case_id": "run_1", "seed": 42})
        result = native.run_bundle(
            location,
            request,
            tmp_path / "direct-output",
            inputs={"mesh": mesh, "geometry": geometry},
            environment={loader: str(ambient), "CUDA_VISIBLE_DEVICES": binding},
            request_id="bound-library",
        )
        assert result["result"]["request_id"] == "bound-library"
        startup = read_json(tmp_path / "direct-output/startup.json")
        assert startup["status"] == "passed"
    assert startup["environment"].get("CUDA_VISIBLE_DEVICES") == binding
