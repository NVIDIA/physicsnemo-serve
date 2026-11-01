"""Independent native workflow reuse and immutable artifact ownership contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import native  # noqa: E402
from pnms_app.common import file_identity  # noqa: E402


@pytest.fixture
def compiled(tmp_path, monkeypatch, request):
    target_platform = getattr(request, "param", sys.platform)
    if target_platform != sys.platform:
        monkeypatch.setattr(native, "sys", SimpleNamespace(platform=target_platform))
    repository = tmp_path / "repository"
    for relative in (
        "native-inference/workflows/runtime",
        "native-inference/cpp-runtime",
    ):
        directory = repository / relative
        (directory / "src").mkdir(parents=True)
        (directory / "CMakeLists.txt").write_text("cmake project")
        (directory / "src/runtime.cpp").write_text("runtime version 1")
    sdk = tmp_path / "sdk"
    (sdk / "include").mkdir(parents=True)
    (sdk / "include/sdk.hpp").write_text("ABI version 1")
    (sdk / "bin").mkdir()
    (sdk / "bin/physicsnemo-infer").write_text("installed runtime")
    app = tmp_path / "app"
    (app / "native").mkdir(parents=True)
    (app / "native/CMakeLists.txt").write_text("custom target")
    (app / "native/workflow.cpp").write_text("custom source")
    (app / "weights.pt").write_bytes(b"model weights")
    external = tmp_path / "system/include"
    external.mkdir(parents=True)
    (external / "json.hpp").write_text("same-version nlohmann header")
    package_config = tmp_path / "system/jsonConfig.cmake"
    package_config.write_text("nlohmann config")
    commands, captures, installs = [], [], {}
    runner_name = (
        "pnms-native-run.exe" if target_platform == "win32" else "pnms-native-run"
    )
    library_name = (
        "physicsnemo_workflow.dll"
        if target_platform == "win32"
        else "libphysicsnemo_workflow.dylib"
        if target_platform == "darwin"
        else "libphysicsnemo_workflow.so"
    )

    def execute(arguments, *, log, environment):
        commands.append(arguments)
        log.write_text("build executed\n")
        if "-S" in arguments:
            prefix = next(
                argument.split("=", 1)[1]
                for argument in arguments
                if argument.startswith("-DCMAKE_INSTALL_PREFIX=")
            )
            installs[arguments[arguments.index("-B") + 1]] = Path(prefix)
            build = Path(arguments[arguments.index("-B") + 1])
            reply = build / ".cmake/api/v1/reply"
            reply.mkdir(parents=True)
            objects = {
                "codemodel": {
                    "paths": {
                        "source": str(
                            repository / "native-inference/workflows/runtime"
                        ),
                        "build": str(build),
                    },
                    "configurations": [{"targets": [{"jsonFile": "target.json"}]}],
                },
                "cmakeFiles": {"inputs": [{"path": str(package_config)}]},
                "toolchains": {
                    "toolchains": [
                        {
                            "language": "CXX",
                            "compiler": {
                                "id": "Test",
                                "version": "1",
                                "implicit": {"includeDirectories": [str(external)]},
                            },
                        }
                    ]
                },
            }
            for kind, value in objects.items():
                (reply / (kind + ".json")).write_text(json.dumps(value))
            (reply / "target.json").write_text(
                json.dumps(
                    {
                        "compileGroups": [
                            {
                                "includes": [{"path": str(sdk / "include")}],
                                "defines": [{"define": "MOCK=1"}],
                            }
                        ]
                    }
                )
            )
            (reply / "index-test.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {"kind": kind, "jsonFile": kind + ".json"}
                            for kind in objects
                        ]
                    }
                )
            )
        elif "--install" in arguments:
            install = installs[arguments[arguments.index("--install") + 1]]
            (install / "bin").mkdir(parents=True)
            runner = install / "bin" / runner_name
            runner.write_text("compiled runner")
            runner.chmod(0o755)
            library_directory = install / (
                "bin" if target_platform == "win32" else "lib"
            )
            library_directory.mkdir(exist_ok=True)
            (library_directory / library_name).write_text("compiled workflow")

    def capture(python, library, *, environment, output):
        captures.append(str(library))
        report = {
            "schema_version": 1,
            "workflow_binary": file_identity(library),
            "system_libraries": {"libc": {"sha256": "a" * 64}},
            "torch_libraries": {},
        }
        output.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(native, "_execute", execute)
    monkeypatch.setattr(
        native,
        "loader_environment",
        lambda *a, **k: {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": str(sdk / "lib")},
    )
    from pnms_app import compatibility, workflow_cache

    monkeypatch.setattr(
        workflow_cache,
        "_toolchain",
        lambda environment, toolchains: {"cxx": "test compiler"},
    )
    monkeypatch.setattr(
        workflow_cache,
        "environment_identity",
        lambda *args: {"providers": "selected test providers"},
    )
    monkeypatch.setattr(compatibility, "capture", capture)
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )
    project = {
        "root": app,
        "workflow": {"template": "custom", "source_dir": "native", "target": "custom"},
    }
    return SimpleNamespace(
        root=tmp_path,
        project=project,
        repository=repository,
        sdk=sdk,
        external=external,
        package_config=package_config,
        commands=commands,
        captures=captures,
    )


def compile_attempt(fixture, name, **options):
    return native.build_workflow(
        fixture.project,
        fixture.root / name,
        repository=fixture.repository,
        sdk_prefix=fixture.sdk,
        python=Path(sys.executable),
        backend="mock",
        **options,
    )


@pytest.mark.parametrize("compiled", ["win32"], indirect=True)
def test_windows_fixture_installs_the_native_platform_artifacts(compiled):
    first = compile_attempt(compiled, "first")
    second = compile_attempt(compiled, "second")
    assert Path(first["runner"]).name == "pnms-native-run.exe"
    assert Path(first["library"]).name == "physicsnemo_workflow.dll"
    assert second["reused"] is True


def test_repeat_workflow_build_restores_attempt_install_without_recompiling(compiled):
    first = compile_attempt(compiled, "first")
    second = compile_attempt(compiled, "second")
    assert len(compiled.commands) == 4, (
        "repeat configure discovers dependencies, while compile/install runs once"
    )
    assert first["reused"] is False and second["reused"] is True
    assert first["identity"] == second["identity"]
    assert Path(second["runner"]).is_relative_to(compiled.root / "second")
    assert compiled.captures[-1] == second["library"]
    assert len(compiled.captures) == 2, (
        "reuse must recheck the current runtime providers"
    )
    Path(first["library"]).write_text("modified first attempt")
    assert Path(second["library"]).read_text() == "compiled workflow"


def test_workflow_reuse_ignores_model_weights_http_settings_and_attempt_paths(compiled):
    first = compile_attempt(compiled, "first")
    original = compiled.project["root"]
    snapshot = compiled.root / "different-attempt/inputs"
    shutil.copytree(original, snapshot, ignore=shutil.ignore_patterns(".pnms"))
    (snapshot / "weights.pt").write_bytes(b"new model weights")
    compiled.project.update(
        root=snapshot,
        _state_root=original,
        serve={"port": 9001, "image": "different-http"},
    )
    second = compile_attempt(compiled, "second")
    assert len(compiled.commands) == 4, (
        "model or HTTP changes must not rebuild an unchanged C++ workflow"
    )
    assert first["identity"] == second["identity"]


@pytest.mark.parametrize(
    "changed",
    [
        "custom-source",
        "workflow-source",
        "sdk-source",
        "sdk-header",
        "custom-dependency",
        "compiler-flags",
        "provider-bytes",
    ],
)
def test_relevant_workflow_changes_invalidate_reuse(compiled, monkeypatch, changed):
    from pnms_app import workflow_cache

    dependency = compiled.project["root"] / "dependency.hpp"
    dependency.write_text("dependency version 1")
    compiled.project["workflow"]["dependencies"] = ["dependency.hpp"]
    first = compile_attempt(compiled, "first")
    paths = {
        "custom-source": compiled.project["root"] / "native/workflow.cpp",
        "workflow-source": compiled.repository
        / "native-inference/workflows/runtime/src/runtime.cpp",
        "sdk-source": compiled.repository
        / "native-inference/cpp-runtime/src/runtime.cpp",
        "sdk-header": compiled.sdk / "include/sdk.hpp",
        "custom-dependency": dependency,
    }
    if changed in paths:
        paths[changed].write_text("different bytes, same advertised version")
    elif changed == "compiler-flags":
        monkeypatch.setattr(
            native, "loader_environment", lambda *a, **k: {"CXXFLAGS": "-DNEW_ABI=1"}
        )
    else:
        monkeypatch.setattr(
            workflow_cache,
            "environment_identity",
            lambda *a: {"providers": "replacement same-version library"},
        )
    second = compile_attempt(compiled, "second")
    assert second["identity"] != first["identity"] and second["reused"] is False
    assert len(compiled.commands) == 6


@pytest.mark.parametrize(
    "damage", ["bytes", "extra-file", "permissions", "missing-seal", "missing-install"]
)
def test_corrupt_or_incomplete_cache_is_rejected(compiled, damage):
    first = compile_attempt(compiled, "first")
    entry = Path(first["cache"]["entry"])
    if damage == "bytes":
        (
            entry
            / "install"
            / Path(first["library"]).relative_to(first["install_root"])
        ).write_text("changed")
    elif damage == "extra-file":
        (entry / "install/extra").write_text("unexpected")
    elif damage == "permissions":
        (
            entry / "install" / Path(first["runner"]).relative_to(first["install_root"])
        ).chmod(0o444)
    elif damage == "missing-seal":
        (entry / "seal.json").unlink()
    else:
        shutil.rmtree(entry / "install")
    with pytest.raises(ValueError, match="cache is corrupt or incomplete"):
        compile_attempt(compiled, "second")
    assert len(compiled.commands) == 4


def test_changed_current_provider_closure_is_rejected_after_cache_restore(
    compiled, monkeypatch
):
    from pnms_app import compatibility

    first = compile_attempt(compiled, "first")
    capture = compatibility.capture

    def changed(*args, **kwargs):
        result = capture(*args, **kwargs)
        result["system_libraries"]["libc"] = {"sha256": "b" * 64}
        return result

    monkeypatch.setattr(compatibility, "capture", changed)
    with pytest.raises(ValueError, match="providers changed"):
        compile_attempt(compiled, "second")
    assert len(compiled.commands) == 4
    assert Path(first["cache"]["seal"]).is_file()


def test_compile_input_mutation_cannot_publish_a_reusable_entry(compiled, monkeypatch):
    execute = native._execute

    def changing(arguments, **kwargs):
        execute(arguments, **kwargs)
        if "--build" in arguments:
            (compiled.project["root"] / "native/workflow.cpp").write_text(
                "edited during compile"
            )

    monkeypatch.setattr(native, "_execute", changing)
    with pytest.raises(ValueError, match="changed during compilation"):
        compile_attempt(compiled, "first")
    assert not (compiled.project["root"] / ".pnms/cache/workflows").exists()


def test_authored_path_project_resolves_declared_dependencies_from_project_root(
    compiled,
):
    import yaml

    (compiled.project["root"] / "dependency.hpp").write_text("external source input")
    compiled.project["workflow"]["dependencies"] = ["dependency.hpp"]
    (compiled.project["root"] / "pnms.yaml").write_text(
        yaml.safe_dump({"workflow": compiled.project["workflow"]})
    )
    original = compiled.project
    compiled.project = original["root"]
    result = compile_attempt(compiled, "first")
    assert result["reused"] is False


def test_interrupted_publication_does_not_expose_partial_cache(compiled, monkeypatch):
    from pnms_app import workflow_cache

    first = compile_attempt(compiled, "first")
    seal = json.loads(Path(first["cache"]["seal"]).read_text())
    identity = {key: seal[key] for key in ("key", "inputs")}
    destination = compiled.root / "interrupted-cache"

    def interrupted(source, target, **kwargs):
        Path(target).mkdir()
        raise OSError("simulated interruption")

    monkeypatch.setattr(workflow_cache.shutil, "copytree", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        workflow_cache.publish(
            destination, identity, first["install_root"], first["compatibility"]
        )
    assert not (destination / identity["key"]).exists()
    assert list(destination.iterdir()) == []


def test_concurrent_publishers_preserve_one_complete_immutable_entry(compiled):
    from concurrent.futures import ThreadPoolExecutor
    from pnms_app import workflow_cache

    first = compile_attempt(compiled, "first")
    seal = json.loads(Path(first["cache"]["seal"]).read_text())
    identity = {key: seal[key] for key in ("key", "inputs")}
    destination = compiled.root / "concurrent-cache"
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: workflow_cache.publish(
                    destination, identity, first["install_root"], first["compatibility"]
                ),
                range(6),
            )
        )
    assert all(result == results[0] for result in results)
    assert [entry.name for entry in destination.iterdir()] == [identity["key"]]
    restored = compiled.root / "restored"
    workflow_cache.restore(destination, identity, restored)
    assert (
        restored / Path(first["runner"]).relative_to(first["install_root"])
    ).read_text() == "compiled runner"


def test_explicit_custom_compile_environment_invalidates_workflow(compiled):
    first = compile_attempt(
        compiled, "first", environment={"CUSTOM_WORKFLOW_SCALE": "2"}
    )
    second = compile_attempt(
        compiled, "second", environment={"CUSTOM_WORKFLOW_SCALE": "3"}
    )
    assert second["reused"] is False
    assert first["identity"] != second["identity"]


def test_compiler_byte_changes_invalidate_even_with_the_same_version(
    tmp_path, monkeypatch
):
    from pnms_app import workflow_cache

    compiler = tmp_path / "compiler"
    compiler.write_text("compiler version one")
    compiler.chmod(0o755)
    cmake = tmp_path / "cmake"
    cmake.write_text("unchanged CMake")
    monkeypatch.setattr(workflow_cache.shutil, "which", lambda *a, **k: str(cmake))
    monkeypatch.setattr(
        workflow_cache.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="same version", returncode=0),
    )
    toolchains = [
        {
            "language": "CXX",
            "compiler": {"path": str(compiler), "id": "Test", "version": "1"},
        }
    ]
    first = workflow_cache._toolchain({}, toolchains)
    compiler.write_text("different compiler implementation")
    assert first != workflow_cache._toolchain({}, toolchains)


@pytest.mark.parametrize(
    ("location", "cwd_decoy"),
    [("source", False), ("build", False), ("build", True), ("absolute", True)],
)
def test_configured_toolchain_file_controls_workflow_identity(
    tmp_path, monkeypatch, location, cwd_decoy
):
    from pnms_app import workflow_cache

    if not shutil.which("cmake"):
        pytest.skip("CMake is required for workflow toolchain checks")
    compiler = next(
        (name for name in ("cl", "c++", "clang++", "g++") if shutil.which(name)),
        None,
    )
    if compiler is None or (sys.platform == "win32" and not shutil.which("ninja")):
        pytest.skip("A C++ compiler and native CMake generator are required")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in workflow_cache._BUILD_VARIABLES
        and key != "CMAKE_GENERATOR_INSTANCE"
    }
    environment.update(
        CXX=compiler,
        CMAKE_GENERATOR="Ninja" if shutil.which("ninja") else "Unix Makefiles",
    )
    repository = tmp_path / "repository"
    source = repository / "native-inference/workflows/runtime"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\n"
        "project(WorkflowToolchain LANGUAGES CXX)\n"
    )
    name = "toolchain.cmake"
    contents = "set(PNMS_TOOLCHAIN_SETTING selected)\n"
    source_toolchain = source / name
    source_toolchain.write_text(
        contents if location == "source" else "# unselected source toolchain\n"
    )
    absolute = tmp_path / name
    absolute.write_text(contents)
    environment["CMAKE_TOOLCHAIN_FILE"] = (
        str(absolute) if location == "absolute" else name
    )
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    if cwd_decoy:
        (cwd / name).write_text("# unselected working-directory toolchain\n")
    monkeypatch.chdir(cwd)

    def fingerprint(build, snapshot):
        return workflow_cache.fingerprint(
            {"root": tmp_path / "app"},
            repository=repository,
            sdk_prefix=sdk,
            python=Path(sys.executable),
            backend="mock",
            custom_source=None,
            custom_target=None,
            tensorrt_root=None,
            environment=environment,
            configured=build,
            configured_snapshot=snapshot,
            build_environment={"stack": {}},
        )

    records = []
    for attempt in ("first", "same"):
        build = tmp_path / attempt
        workflow_cache.prepare_configuration(build)
        selected = (
            build / name
            if location == "build"
            else source_toolchain
            if location == "source"
            else absolute
        )
        selected.write_text(contents)
        subprocess.run(
            ["cmake", "-S", str(source), "-B", str(build)],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (
            f"CMAKE_TOOLCHAIN_FILE:FILEPATH={selected.as_posix()}"
            in (build / "CMakeCache.txt").read_text()
        )
        snapshot = workflow_cache.configured_inputs(build)
        record = fingerprint(build, snapshot)
        assert record["inputs"]["toolchain_file"] == [file_identity(selected)]
        records.append(record)
    assert records[0]["key"] == records[1]["key"]

    selected.write_text("set(PNMS_TOOLCHAIN_SETTING changed)\n")
    assert fingerprint(build, snapshot)["key"] != records[1]["key"]
    selected.unlink()
    with pytest.raises(ValueError, match="workflow cache dependency is missing"):
        fingerprint(build, snapshot)


def test_configured_msvc_compiler_and_explicit_wrapper_define_toolchain(
    tmp_path, monkeypatch
):
    from pnms_app import workflow_cache

    paths = {name: tmp_path / name for name in ("cmake", "cl.exe", "launcher")}
    for name, path in paths.items():
        path.write_text(name + " implementation")
    commands = []

    def execute(arguments, **kwargs):
        commands.append(arguments)
        assert arguments == ["cmake", "--version"]
        return SimpleNamespace(stdout="cmake version 4", returncode=0)

    monkeypatch.setattr(
        workflow_cache.shutil,
        "which",
        lambda command, **kwargs: str(paths[command]) if command in paths else None,
    )
    monkeypatch.setattr(workflow_cache.subprocess, "run", execute)
    toolchains = [
        {
            "language": "CXX",
            "compiler": {
                "path": str(paths["cl.exe"]),
                "id": "MSVC",
                "version": "19.44.35221.0",
            },
        }
    ]
    environment = {"CXX": "launcher --target=x64"}
    first = workflow_cache._toolchain(environment, toolchains)
    assert first["compilers"]["CXX"] == {
        "path": str(paths["cl.exe"].resolve()),
        "executable": file_identity(paths["cl.exe"]),
        "id": "MSVC",
        "version": "19.44.35221.0",
    }
    assert first["commands"]["CXX"]["arguments"] == ["--target=x64"]
    paths["launcher"].write_text("same wrapper path, different implementation")
    assert first != workflow_cache._toolchain(environment, toolchains)
    assert commands == [["cmake", "--version"], ["cmake", "--version"]]


def test_windows_direct_sdk_probe_uses_installed_executable_suffix(
    tmp_path, monkeypatch
):
    from pnms_app import workflow_cache

    runtime = tmp_path / "sdk/bin/physicsnemo-infer.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"Windows runtime")

    def execute(arguments, **kwargs):
        if len(arguments) > 3:
            assert Path(arguments[3]) == runtime
            return SimpleNamespace(stdout='{"schema_version": 1}', returncode=0)
        return SimpleNamespace(stdout="null", returncode=0)

    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(workflow_cache.subprocess, "run", execute)
    assert workflow_cache.environment_identity(
        sys.executable, runtime.parents[1], {}, None
    ) == {"providers": {"schema_version": 1}, "torch_development": {}}


@pytest.mark.parametrize("reference", ["absolute", "search-path", "response"])
def test_windows_link_fragments_preserve_native_path_backslashes(
    tmp_path, monkeypatch, reference
):
    from pnms_app import workflow_cache

    # A backslash is a literal filename character on POSIX and a separator on
    # Windows, so this exercises Windows tokenization on either test host.
    archive = tmp_path / r"external\library" / "libcustom.a"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"external library")
    fragment = (
        str(archive) if reference == "absolute" else f"-L{archive.parent} -lcustom"
    )
    if reference == "response":
        response = tmp_path / "link.rsp"
        response.write_text(fragment)
        fragment = f'@"{response}"'
    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name="nt"))
    result = workflow_cache._link_inputs(
        {"link": {"commandFragments": [{"fragment": fragment}]}}, [], tmp_path
    )
    assert result[str(archive)] == file_identity(archive)


@pytest.mark.parametrize(
    ("host", "fragment", "expected"),
    [
        ("nt", r"C:\sdk\libcustom.a", [r"C:\sdk\libcustom.a"]),
        ("nt", r"-LC:\sdk -lcustom", [r"-LC:\sdk", "-lcustom"]),
        (
            "nt",
            r'"C:\Program Files\sdk\libcustom.a"',
            [r"C:\Program Files\sdk\libcustom.a"],
        ),
        (
            "nt",
            r'-L"C:\Program Files\sdk" -lcustom',
            [r"-LC:\Program Files\sdk", "-lcustom"],
        ),
        (
            "nt",
            r'@"C:\Program Files\sdk\link.rsp"',
            [r"@C:\Program Files\sdk\link.rsp"],
        ),
        ("nt", r"C:\user's\libcustom.a", [r"C:\user's\libcustom.a"]),
        (
            "posix",
            r"/sdk/with\ space/libcustom.a",
            ["/sdk/with space/libcustom.a"],
        ),
    ],
)
def test_cmake_command_tokens_follow_host_path_quoting(
    monkeypatch, host, fragment, expected
):
    from pnms_app import workflow_cache

    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name=host))
    assert workflow_cache._command_tokens(fragment) == expected


@pytest.mark.parametrize(
    "changed", ["implicit-header", "package-config", "external-header-symlink"]
)
def test_actual_cmake_compile_dependencies_invalidate_same_version_cache(
    compiled, changed
):
    if changed == "external-header-symlink":
        target = compiled.root / "outside-system-header.hpp"
        target.write_text("external target version one")
        (compiled.external / "json.hpp").unlink()
        (compiled.external / "json.hpp").symlink_to(target)
    first = compile_attempt(compiled, "first")
    if changed == "package-config":
        compiled.package_config.write_text(
            "different configuration at same package version"
        )
    elif changed == "external-header-symlink":
        target.write_text("different actual bytes through symlink")
    else:
        (compiled.external / "json.hpp").write_text(
            "changed actual header with unchanged version"
        )
    second = compile_attempt(compiled, "second")
    assert second["identity"] != first["identity"] and second["reused"] is False


@pytest.mark.parametrize(
    "name", ["external.setting", "external.H", "Makefile", "makefile"]
)
def test_cmake_include_files_with_arbitrary_names_invalidate_workflow(compiled, name):
    dependency = compiled.external / name
    dependency.write_text("#define RESULT 1\n")
    first = compile_attempt(compiled, "first")
    unchanged = compile_attempt(compiled, "unchanged")
    assert unchanged["identity"] == first["identity"]
    assert unchanged["reused"] is True

    dependency.write_text("#define RESULT 2\n")
    changed = compile_attempt(compiled, "changed")
    assert changed["identity"] != first["identity"]
    assert changed["reused"] is False


def test_build_generated_headers_do_not_change_configured_input_identity(
    compiled, monkeypatch
):
    execute = native._execute

    def generated(arguments, **kwargs):
        execute(arguments, **kwargs)
        if "-B" in arguments:
            build = Path(arguments[arguments.index("-B") + 1])
            target_path = build / ".cmake/api/v1/reply/target.json"
            target = json.loads(target_path.read_text())
            target["sources"] = [
                {"path": str(build / "generated/declared.hpp"), "isGenerated": True}
            ]
            target["compileGroups"][0]["includes"].append(
                {"path": str(build / "generated")}
            )
            target_path.write_text(json.dumps(target))
        elif "--build" in arguments:
            generated_dir = Path(arguments[2]) / "generated"
            generated_dir.mkdir()
            (generated_dir / "declared.hpp").write_text("generated target source")
            (generated_dir / "included.hpp").write_text(
                "generated header included indirectly"
            )

    monkeypatch.setattr(native, "_execute", generated)
    first = compile_attempt(compiled, "first")
    second = compile_attempt(compiled, "second")
    assert first["reused"] is False and second["reused"] is True


def test_windows_cmake_forward_slash_roots_preserve_cache_reuse(monkeypatch):
    from pnms_app import workflow_cache

    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name="nt"))

    def inputs(build):
        include = build.replace("\\", "/") + "/_deps/json/include"
        return {
            "compile": {"include": include},
            "link": build + r"\Release\custom.lib",
            "headers": {include: [{"sha256": "same header bytes"}]},
        }

    first, second = r"C:\project\first\build", r"C:\project\second\build"
    normalized = workflow_cache._normalize(inputs(first), [(first, "<build>")])
    assert normalized == workflow_cache._normalize(
        inputs(second), [(second, "<build>")]
    )
    changed = inputs(second)
    changed["headers"][second.replace("\\", "/") + "/_deps/json/include"][0][
        "sha256"
    ] = "changed header bytes"
    assert normalized != workflow_cache._normalize(changed, [(second, "<build>")])


def test_posix_dependency_roots_keep_literal_backslashes(monkeypatch):
    from pnms_app import workflow_cache

    monkeypatch.setattr(workflow_cache, "os", SimpleNamespace(name="posix"))
    assert (
        workflow_cache._normalize(
            "/project/component/include", [(r"/project\component", "<project>")]
        )
        == "/project/component/include"
    )


@pytest.mark.parametrize("reference", ["absolute", "search-path"])
def test_external_link_library_bytes_invalidate_workflow(
    compiled, monkeypatch, reference
):
    archive = compiled.root / "external/libcustom.a"
    archive.parent.mkdir()
    archive.write_text("static implementation one")
    execute = native._execute

    def linked(arguments, **kwargs):
        execute(arguments, **kwargs)
        if "-B" in arguments:
            build = Path(arguments[arguments.index("-B") + 1])
            target_path = build / ".cmake/api/v1/reply/target.json"
            target = json.loads(target_path.read_text())
            fragment = (
                str(archive)
                if reference == "absolute"
                else f"-L{archive.parent} -lcustom"
            )
            target["link"] = {
                "commandFragments": [{"role": "libraries", "fragment": fragment}]
            }
            target_path.write_text(json.dumps(target))

    monkeypatch.setattr(native, "_execute", linked)
    first = compile_attempt(compiled, "first")
    archive.write_text("static implementation two at same path")
    second = compile_attempt(compiled, "second")
    assert first["identity"] != second["identity"] and second["reused"] is False


def test_configure_fetched_headers_remain_inputs_inside_build_directory(
    compiled, monkeypatch
):
    execute = native._execute
    header = "fetched dependency version one"

    def fetched(arguments, **kwargs):
        execute(arguments, **kwargs)
        if "-B" in arguments:
            build = Path(arguments[arguments.index("-B") + 1])
            include = build / "_deps/json-src/include"
            include.mkdir(parents=True)
            (include / "json.hpp").write_text(header)
            target_path = build / ".cmake/api/v1/reply/target.json"
            target = json.loads(target_path.read_text())
            target["compileGroups"][0]["includes"].append({"path": str(include)})
            target_path.write_text(json.dumps(target))

    monkeypatch.setattr(native, "_execute", fetched)
    first = compile_attempt(compiled, "first")
    header = "changed configure-time fetched dependency"
    second = compile_attempt(compiled, "second")
    assert first["identity"] != second["identity"] and second["reused"] is False
