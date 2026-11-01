"""CPU subprocess components for the public CLI's existing Python contracts.

These checks execute Python hooks, never HTTP/queue acceptance or GPU tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
FORBIDDEN_TOOLS = ("docker", "tsh", "lep", "nvidia-smi", "nvcc", "pnms-model-builder")


def executable(path, program):
    if sys.platform == "win32":
        # Use pip's native launcher so CreateProcess can resolve the bare tool name.
        from pip._vendor.distlib.scripts import ScriptMaker

        sources = path.parent / ".launcher-sources"
        sources.mkdir(exist_ok=True)
        (sources / path.name).write_text("#!python\n" + program, encoding="utf-8")
        maker = ScriptMaker(str(sources), str(path.parent))
        maker.executable = sys.executable
        [launcher] = maker.make(path.name)
        return Path(launcher)
    path.write_text(f"#!{sys.executable}\n" + program, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def environment(tmp_path):
    binaries = tmp_path / "forbidden-tools"
    binaries.mkdir()
    marker = tmp_path / "unexpected-tool.txt"
    for name in FORBIDDEN_TOOLS:
        executable(
            binaries / name,
            f"from pathlib import Path\nPath({str(marker)!r}).write_text({name!r})\nraise SystemExit(97)\n",
        )
    guard = tmp_path / "subprocess-guard"
    guard.mkdir()
    # Windows may search System32 before PATH. Intercept in Python before launch.
    (guard / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""\
            import os
            from pathlib import Path
            import sys

            def guard(event, arguments):
                if event != "subprocess.Popen":
                    return
                program, argv, *_ = arguments
                if program is None:
                    if isinstance(argv, (str, bytes)):
                        command = os.fsdecode(argv).lstrip()
                        if command.startswith('"'):
                            program = command[1:].split('"', 1)[0]
                        else:
                            program = command.split(maxsplit=1)[0] if command else ""
                    elif isinstance(argv, (list, tuple)) and argv:
                        program = argv[0]
                if not isinstance(program, (str, bytes, os.PathLike)):
                    return
                name = os.fsdecode(program).replace("\\\\", "/").rsplit("/", 1)[-1].casefold()
                if name.endswith(".exe"):
                    name = name[:-4]
                if name in {FORBIDDEN_TOOLS!r}:
                    Path({str(marker)!r}).write_text(name, encoding="utf-8")
                    os._exit(97)

            sys.addaudithook(guard)
            """
        ),
        encoding="utf-8",
    )
    return {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(guard) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PNMS_FORBIDDEN_TOOL_MARKER": str(marker),
    }


@pytest.mark.parametrize("tool", FORBIDDEN_TOOLS)
def test_forbidden_tool_sentinels_execute_real_programs(environment, tool):
    binaries = Path(environment["PATH"].split(os.pathsep, 1)[0])
    program = binaries / (f"{tool}.exe" if sys.platform == "win32" else tool)
    result = subprocess.run(
        [str(program)],
        # Prove the native launcher works independently of the Python audit hook.
        env={**environment, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 97, result.stderr or result.stdout
    assert Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).read_text() == tool


@pytest.mark.parametrize("tool", FORBIDDEN_TOOLS)
def test_forbidden_tool_guard_intercepts_bare_subprocess_lookup(environment, tool):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; raise SystemExit(subprocess.call([sys.argv[1]]))",
            tool,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 97, result.stderr or result.stdout
    assert Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).read_text() == tool


@pytest.mark.parametrize("tool", FORBIDDEN_TOOLS)
@pytest.mark.parametrize("launch", ["argv", "executable"])
def test_forbidden_tool_guard_intercepts_resolved_executables(
    tmp_path, environment, tool, launch
):
    resolved = tmp_path / "system tools"
    resolved.mkdir()
    program = executable(resolved / tool, "raise SystemExit(0)\n")
    command = (
        "subprocess.call([sys.argv[1]])"
        if launch == "argv"
        else "subprocess.call(['harmless-argv-alias'], executable=sys.argv[1])"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import subprocess, sys; raise SystemExit({command})",
            str(program),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 97, result.stderr or result.stdout
    assert Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).read_text() == tool


def cli(*arguments, cwd, environment):
    return subprocess.run(
        [sys.executable, str(REPOSITORY / "pnms"), *map(str, arguments)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def succeeded(result):
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def legacy_init(path, environment, *arguments):
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/plugin_dev.py"),
            "init",
            str(path),
            *arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    succeeded(result)


def test_new_python_public_workflow_is_local_and_uses_requested_input(
    tmp_path, environment
):
    source = tmp_path / "new-python"
    initialized = succeeded(
        cli(
            "init", source, "--runtime", "python", cwd=tmp_path, environment=environment
        )
    )
    assert initialized["runtime_profile"] == "python-test"
    assert (source / "plugin.yaml").is_file()
    assert not (source / "pnms.yaml").exists()
    request = source / "examples/request.json"
    request.write_text('{"value": 19}')
    checked = succeeded(
        cli("check", "--config-only", cwd=source, environment=environment)
    )
    assert checked["status"] == "valid"
    assert checked["coverage"]["executed_phases"] == []
    built = succeeded(
        cli("build", "--environment", "local", cwd=source, environment=environment)
    )
    assert built["runtime"] == "python"
    assert "models" not in built
    assert built["checks"]["http"] == "not_run"
    assert built["checks"]["model"] == "not_run"
    result = succeeded(
        cli("run", "--input", request, cwd=source, environment=environment)
    )
    assert result["result"]["doubled"] == 38
    assert not Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).exists()


def test_existing_plugin_preserves_files_and_frozen_run_while_dev_sees_edits(
    tmp_path, environment
):
    source = tmp_path / "legacy-python"
    legacy_init(source, environment)
    originals = {
        path: path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    request = tmp_path / "request.json"
    request.write_text('{"value": 7}')
    built = succeeded(cli("build", source, cwd=tmp_path, environment=environment))
    receipt = Path(built["root"]) / "build.json"
    receipt_bytes = receipt.read_bytes()
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not (source / "pnms.yaml").exists()
    workflow = source / "workflow.py"
    workflow.write_text(workflow.read_text().replace("value * 2", "value * 3"))
    dev = succeeded(
        cli(
            "run",
            source,
            "--dev",
            "--input",
            request,
            cwd=tmp_path,
            environment=environment,
        )
    )
    normal = succeeded(
        cli("run", source, "--input", request, cwd=tmp_path, environment=environment)
    )
    assert dev["result"]["doubled"] == 21
    assert normal["result"]["doubled"] == 14
    assert receipt.read_bytes() == receipt_bytes


def test_dev_execution_does_not_require_or_publish_a_build(tmp_path, environment):
    source = tmp_path / "developer-python"
    succeeded(cli("init", source, cwd=tmp_path, environment=environment))
    request = tmp_path / "request.json"
    request.write_text('{"value": 8}')
    dev = succeeded(
        cli(
            "run",
            source,
            "--dev",
            "--input",
            request,
            cwd=tmp_path,
            environment=environment,
        )
    )
    assert dev["result"]["doubled"] == 16
    assert not (source / ".pnms/last-build.json").exists()
    normal = cli(
        "run", source, "--input", request, cwd=tmp_path, environment=environment
    )
    assert normal.returncode != 0
    assert "pnms build" in normal.stderr


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("frozen", [False, True])
def test_python_execute_failure_skips_postprocess_and_preserves_terminal_result(
    tmp_path, environment, status, frozen
):
    source = tmp_path / "terminal-python"
    succeeded(
        cli(
            "init",
            source,
            "--pipeline",
            "postprocess",
            cwd=tmp_path,
            environment=environment,
        )
    )
    marker = tmp_path / "postprocess-called.txt"
    error = {"message": f"execute {status}"}
    workflow = source / "workflow.py"
    workflow.write_text(
        workflow.read_text()
        + textwrap.dedent(
            f"""

            def execute(ctx):
                return {{"status": {status!r}, "error": {error!r}}}

            def postprocess(result, ctx):
                from pathlib import Path
                Path({str(marker)!r}).write_text("called", encoding="utf-8")
                return {{"value": 7, "doubled": 14}}
            """
        )
    )
    if frozen:
        succeeded(cli("build", source, cwd=tmp_path, environment=environment))
    request = tmp_path / "request.json"
    request.write_text('{"value": 7}')
    current = cli(
        "run",
        source,
        *([] if frozen else ["--dev"]),
        "--input",
        request,
        cwd=tmp_path,
        environment=environment,
    )
    assert not marker.exists(), current.stderr or current.stdout
    assert current.returncode != 0
    report = json.loads(current.stdout)
    assert report["status"] == status
    assert report["result"]["status"] == status
    assert report["result"]["error"] == error
    assert json.loads(Path(report["result_path"]).read_text()) == report["result"]
    assert report["coverage"]["executed_phases"] == ["prepare", "execute"]
    assert report["coverage"]["complete_workflow"] is False
    assert not Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).exists()


@pytest.mark.parametrize("routing", ["prepare", "authored"])
def test_python_run_follows_stage_transitions_without_executing_bypassed_hooks(
    tmp_path, environment, routing
):
    source = tmp_path / "routed-python"
    succeeded(
        cli(
            "init",
            source,
            "--pipeline",
            "postprocess",
            cwd=tmp_path,
            environment=environment,
        )
    )
    workflow = source / "workflow.py"
    override = {"next_stage_id": "postprocess"} if routing == "prepare" else {}
    workflow.write_text(
        workflow.read_text()
        + textwrap.dedent(
            f"""

            def prepare(ctx):
                return {{"parameters": {{"value": 7}}, **{override!r}}}

            def execute(ctx):
                raise RuntimeError("execute must be bypassed")

            def postprocess(result, ctx):
                return {{"bypassed": True}}
            """
        )
    )
    if routing == "authored":
        manifest_path = source / "plugin.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["pipeline"]["stages"] = [
            {
                "id": phase,
                "phase": phase,
                "handler": "persist_results" if phase == "results" else "plugin_phase",
                "queue": phase,
                "next": next_stage,
            }
            for phase, next_stage in [
                ("prepare", "postprocess"),
                ("execute", "postprocess"),
                ("postprocess", "results"),
                ("results", None),
            ]
        ]
        manifest_path.write_text(yaml.safe_dump(manifest))
    request = tmp_path / "request.json"
    request.write_text('{"value": 2}')
    report = succeeded(
        cli(
            "run",
            source,
            "--dev",
            "--input",
            request,
            cwd=tmp_path,
            environment=environment,
        )
    )
    assert report["result"]["bypassed"] is True
    assert report["coverage"]["executed_phases"] == ["prepare", "postprocess"]
    assert not Path(environment["PNMS_FORBIDDEN_TOOL_MARKER"]).exists()


def test_partial_check_is_not_reported_as_full_success_and_legacy_exit_is_unchanged(
    tmp_path, environment
):
    source = tmp_path / "partial-python"
    legacy_init(source, environment)
    path = source / "plugin.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["developer"] = {
        "readiness": {
            "recommended_check_phase": "prepare",
            "env": [{"name": "PNMS_COMPONENT_MISSING_DEPENDENCY", "required": True}],
        }
    }
    path.write_text(yaml.safe_dump(manifest))
    environment.pop("PNMS_COMPONENT_MISSING_DEPENDENCY", None)
    current = cli("check", source, cwd=tmp_path, environment=environment)
    assert current.returncode != 0
    report = json.loads(current.stdout)
    assert report["status"] == "needs_setup"
    assert report["coverage"]["executed_phases"] == ["prepare"]
    assert report["coverage"]["complete_workflow"] is False
    original = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/plugin_dev.py"),
            "check",
            str(source),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert original.returncode == 0, original.stderr
    assert json.loads(original.stdout)["status"] == "needs_setup"


def test_custom_python_phase_executors_survive_common_build(tmp_path, environment):
    source = tmp_path / "custom-python"
    legacy_init(
        source,
        environment,
        "--runtime",
        "custom",
        "--executor-class",
        "custom.execute",
        "--phase-executor",
        "prepare=custom.prepare",
    )
    original = (source / "plugin.yaml").read_bytes()
    built = succeeded(cli("build", source, cwd=tmp_path, environment=environment))
    snapshot = Path(built["root"]) / built["python"]["plugin_root"]
    assert (snapshot / "plugin.yaml").read_bytes() == original
    manifest = yaml.safe_load(original)
    assert manifest["runtime"]["prepare_executor_class"] == "custom.prepare"
    assert (source / "plugin.yaml").read_bytes() == original


def test_fanout_direct_run_requires_service_and_prepare_check_remains_available(
    tmp_path, environment
):
    source = tmp_path / "ensemble-python"
    succeeded(
        cli(
            "init",
            source,
            "--pipeline",
            "ensemble",
            cwd=tmp_path,
            environment=environment,
        )
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 8}')
    partial = succeeded(
        cli(
            "check",
            source,
            "--through-phase",
            "prepare",
            cwd=tmp_path,
            environment=environment,
        )
    )
    assert partial["coverage"]["executed_phases"] == ["prepare"]
    assert partial["coverage"]["complete_workflow"] is False
    direct = cli(
        "run",
        source,
        "--dev",
        "--input",
        request,
        cwd=tmp_path,
        environment=environment,
    )
    assert direct.returncode != 0
    assert "fanout" in direct.stderr and "pnms serve" in direct.stderr


def test_python_rejects_native_flags_and_bad_requests_before_execution(
    tmp_path, environment
):
    source = tmp_path / "python-flags"
    succeeded(cli("init", source, cwd=tmp_path, environment=environment))
    flags = cli("build", source, "--gpu", "0", cwd=tmp_path, environment=environment)
    assert flags.returncode != 0
    assert "python" in flags.stderr.lower() and "native" in flags.stderr
    assert not (source / ".pnms/builds").exists()
    request = tmp_path / "request.json"
    request.write_text('{"value": "invalid"}')
    bad = cli(
        "run",
        source,
        "--dev",
        "--input",
        request,
        cwd=tmp_path,
        environment=environment,
    )
    assert bad.returncode != 0
    assert "request" in bad.stderr.lower()
    extra = cli(
        "run",
        source,
        "--dev",
        "--input",
        request,
        "--input-file",
        "mesh=mesh.dat",
        cwd=tmp_path,
        environment=environment,
    )
    assert extra.returncode != 0
    assert "native-only" in extra.stderr


def test_frozen_run_keeps_its_interpreter_while_dev_uses_new_configuration(
    tmp_path, environment
):
    source = tmp_path / "selected-environment"
    succeeded(cli("init", source, cwd=tmp_path, environment=environment))
    succeeded(cli("build", source, cwd=tmp_path, environment=environment))
    broken = executable(
        tmp_path / "new-environment-python",
        'import sys\nprint("selected development environment unavailable",file=sys.stderr)\nraise SystemExit(95)\n',
    )
    (source / "pnms.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": source.name,
                "runtime": "python",
                "plugin": ".",
                "build": {"python": str(broken)},
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 13}')
    normal = succeeded(
        cli("run", source, "--input", request, cwd=tmp_path, environment=environment)
    )
    assert normal["result"]["doubled"] == 26
    dev = cli(
        "run",
        source,
        "--dev",
        "--input",
        request,
        cwd=tmp_path,
        environment=environment,
    )
    assert dev.returncode != 0
    assert "selected development environment unavailable" in dev.stderr


@pytest.mark.parametrize("selection", ["flag", "descriptor"])
def test_python_container_environment_is_rejected_instead_of_ignored(
    tmp_path, environment, selection
):
    source = tmp_path / "python-environment"
    succeeded(cli("init", source, cwd=tmp_path, environment=environment))
    arguments = ["build", source]
    if selection == "flag":
        arguments += ["--environment", "container"]
    else:
        (source / "pnms.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": source.name,
                    "runtime": "python",
                    "plugin": ".",
                    "build": {"environment": "container"},
                }
            )
        )
        arguments += ["--plan"]
    result = cli(*arguments, cwd=tmp_path, environment=environment)
    assert result.returncode != 0
    assert "python" in result.stderr.lower() and "container" in result.stderr.lower()
    assert not (source / ".pnms/builds").exists()
