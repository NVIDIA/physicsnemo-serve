"""CPU checks for the finite job; no GPU qualification or remote resources."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qa.native_inference import pnms_cfd_job as job
from pnms_app.common import file_identity, read_json, write_json
from test_pnms_cfd import imported  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mirrored(imported, tmp_path):  # noqa: F811
    _, original, deployment = imported
    source = deployment.parent
    payload = tmp_path / "mirrored"
    shutil.copytree(source, payload)
    prefix = Path("/opt/physicsnemo-native")
    original["model_build_sources"] = {
        "full": job.IMPORT_ORIGIN,
        "tail": job.IMPORT_ORIGIN,
    }

    def relocated(raw):
        return str(prefix / Path(raw).relative_to(source))

    original["packages"] = [relocated(raw) for raw in original["packages"]]
    for key in ("stats_path", "startup_fixture_path"):
        original[key] = relocated(original[key])
    for item in original["model_assets"]:
        item["path"] = relocated(item["path"])
    write_json(payload / "deployment.json", original)
    return payload, original


def test_import_relocation_preserves_every_verified_byte_and_original_alias_origin(
    mirrored, tmp_path, monkeypatch
):
    payload, original = mirrored
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("static relocation started a process"),
    )
    output = tmp_path / "output/deployment.json"
    record = job.relocate_import(
        payload / "deployment.json", payload, output, backend="aoti"
    )
    actual = read_json(output)
    assert record["source_image"] == job.IMPORT_IMAGES["aoti"]
    assert record["model_build_sources"] == original["model_build_sources"]
    assert record["recompiled"] is False
    assert [item["sha256"] for item in actual["model_assets"]] == [
        item["sha256"] for item in original["model_assets"]
    ]
    assert all(
        Path(item["path"]).is_relative_to(payload) for item in actual["model_assets"]
    )
    assert actual["packages"] == [
        str(payload / "points-60000"),
        str(payload / "points-8095"),
    ]
    assert read_json(output.with_name("import.json")) == record


@pytest.mark.parametrize(
    "mutation", ["bytes", "missing", "escape", "symlink", "source", "extra"]
)
def test_mirrored_import_rejects_changed_or_unproven_payload(
    mirrored, tmp_path, mutation
):
    payload, original = mirrored
    if mutation == "bytes":
        (payload / "points-60000/model.pt2").write_bytes(b"replacement")
    elif mutation == "missing":
        (payload / "startup/startup.json").unlink()
    elif mutation == "escape":
        original["model_assets"][0]["path"] = "/opt/physicsnemo-native/../not-an-asset"
    elif mutation == "symlink":
        path = payload / "points-60000/model.pt2"
        target = tmp_path / "external.pt2"
        path.rename(target)
        path.symlink_to(target)
    elif mutation == "source":
        original["model_build_sources"]["tail"] = "d" * 40
    else:
        (payload / "points-60000/extra.so").write_bytes(b"not inventoried")
    write_json(payload / "deployment.json", original)
    with pytest.raises(ValueError):
        job.relocate_import(
            payload / "deployment.json",
            payload,
            tmp_path / "output/deployment.json",
            backend="aoti",
        )
    assert not (tmp_path / "output/import.json").exists()


def test_job_subprocess_propagates_real_nonzero_exit_and_keeps_diagnostics(tmp_path):
    log = tmp_path / "command.log"
    with pytest.raises(ValueError, match="exited 7"):
        job.execute(
            [
                sys.executable,
                "-c",
                "print('actual failure', flush=True); raise SystemExit(7)",
            ],
            output=log,
            environment={},
            cwd=tmp_path,
            timeout=5,
        )
    assert "actual failure" in log.read_text()
    assert read_json(log.with_suffix(".command.json"))[-1].endswith("SystemExit(7)")


def test_job_timeout_retires_real_child_process(tmp_path):
    pid = tmp_path / "pid"
    source = (
        "import os,pathlib,time; pathlib.Path("
        + repr(str(pid))
        + ").write_text(str(os.getpid())); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        job.execute(
            [sys.executable, "-c", source],
            output=tmp_path / "timeout.log",
            environment={},
            cwd=tmp_path,
            timeout=0.5,
        )
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_descendant_cleanup_follows_nested_and_detached_service_children(tmp_path):
    for pid, ppid in ((100, 1), (101, 100), (102, 101), (103, 100), (200, 1)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} (a process with spaces) S {ppid} 0 0")
    assert job.descendants(100, proc=tmp_path) == {101, 102, 103}


def test_session_commands_use_common_cli_and_keep_six_counterbalanced_service_sessions(
    tmp_path,
):
    from pnms_app import cli
    from qa.scripts import benchmark_pnms_cfd

    assert job.ORDER == ("python", "aoti", "tensorrt", "tensorrt", "aoti", "python")
    args = SimpleNamespace(
        output=tmp_path,
        repository=ROOT,
        python=Path(sys.executable),
        image="registry/cfd@sha256:" + "f" * 64,
        source_sha="a" * 40,
        profile=Path("profile.json"),
        request=Path("request.json"),
        reference_fields=Path("fields.json"),
        input_root=Path("inputs"),
        checkpoint=Path("checkpoint"),
        stats=Path("stats"),
        run_id="finite",
    )
    for ordinal, runtime in enumerate(job.ORDER):
        built = {"root": str(tmp_path / runtime), "bundle": "bundle"}
        argv, receipt = job.session_argv(
            args,
            runtime,
            ordinal,
            tmp_path / "application",
            built,
            tmp_path / "inventory.json",
        )
        assert argv[0] == sys.executable
        assert argv[argv.index("--expected-image") + 1] == args.image
        assert argv[argv.index("--ordinal") + 1] == str(ordinal)
        command = read_json(Path(argv[argv.index("--service-command-json") + 1]))
        assert command[:3] == [sys.executable, str(ROOT / "pnms"), "serve"]
        assert "--dev" in command and "--skip-build" in command
        assert cli.parser().parse_args(command[2:]).command == "serve"
        assert benchmark_pnms_cfd.parser().parse_args(argv[2:]).runtime == runtime
        assert (
            "--native-bundle" in argv
            if runtime != "python"
            else "--service-artifact-root" in argv
        )
        assert (
            receipt == tmp_path / "sessions" / f"{ordinal}-{runtime}" / "session.json"
        )
        assert not any("docker" == value or "lepton" == value for value in argv)


def test_preflight_failure_writes_failed_summary_and_never_submits_jobs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        job,
        "execute",
        lambda *a, **kw: pytest.fail("failed preflight executed a command"),
    )
    args = job.parser().parse_args(
        [
            "--image",
            "mutable:tag",
            "--source-sha",
            "a" * 40,
            "--run-id",
            "failed",
            "--output",
            str(tmp_path / "output"),
            "--reference-fields",
            "fields",
            "--input-root",
            "inputs",
            "--expected-source-inventory-sha256",
            "b" * 64,
            "--expected-runner-sha256",
            "c" * 64,
        ]
    )
    with pytest.raises(ValueError, match="immutable image"):
        job.run(args)
    summary = read_json(args.output / "summary.json")
    assert summary["status"] == "failed"
    assert summary["cases"][0]["name"] == "preflight"
    assert summary["cases"][0]["status"] == "failed"
    assert summary["sessions"] == []


def test_prepare_only_can_never_be_reported_as_gpu_qualification(tmp_path, monkeypatch):
    from qa.native_inference import cfd_contract, pnms_stage_job
    from qa.scripts import run_native_cfd_reference
    from qa.native_inference import cfd_benchmark_reference

    checkpoint, stats, fields = [
        tmp_path / name for name in ("checkpoint", "stats", "fields")
    ]
    for path in (checkpoint, stats, fields):
        write_json(path, {})
    source_inventory = tmp_path / "inventory.json"
    write_json(source_inventory, [])
    monkeypatch.setenv("PNMS_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(pnms_stage_job, "stage", lambda *a: {"source_root": str(ROOT)})
    monkeypatch.setattr(cfd_contract, "verify_file", lambda *a: None)
    monkeypatch.setattr(run_native_cfd_reference, "_verify_inputs", lambda *a: None)
    monkeypatch.setattr(cfd_benchmark_reference, "qualified_reference", lambda *a: None)
    monkeypatch.setattr(
        job,
        "relocate_import",
        lambda *a, **kw: {"backend": kw["backend"], "recompiled": False},
    )
    monkeypatch.setattr(
        job,
        "python_readiness",
        lambda *a, **kw: {"status": "ready", "scope": "CPU readiness fixture"},
    )
    monkeypatch.setattr(
        job,
        "execute",
        lambda *a, **kw: pytest.fail("prepare-only started models or services"),
    )
    args = job.parser().parse_args(
        [
            "--prepare-only",
            "--image",
            "registry/cfd@sha256:" + "f" * 64,
            "--source-sha",
            "a" * 40,
            "--run-id",
            "prepare",
            "--output",
            str(tmp_path / "output"),
            "--reference-fields",
            str(fields),
            "--input-root",
            "inputs",
            "--checkpoint",
            str(checkpoint),
            "--stats",
            str(stats),
            "--source-inventory",
            str(source_inventory),
            "--expected-source-inventory-sha256",
            file_identity(source_inventory)["sha256"],
            "--expected-runner-sha256",
            file_identity(Path(job.__file__))["sha256"],
        ]
    )
    result = job.run(args)
    assert result["status"] == "prepared"
    assert result["expected_cases"] == ["preflight", "import.aoti", "import.tensorrt"]
    assert result["sessions"] == []
    assert read_json(args.output / "summary.json")["status"] == "prepared"


@pytest.fixture
def finite_job(tmp_path, monkeypatch):
    from qa.native_inference import cfd_contract, pnms_stage_job
    from qa.scripts import run_native_cfd_reference
    from qa.native_inference import cfd_benchmark_reference

    repository = tmp_path / "source"
    for plugin in (
        "physicsnemo-cfd-transolver-native",
        "physicsnemo-cfd-surface-benchmark",
    ):
        shutil.copytree(ROOT / "plugins" / plugin, repository / "plugins" / plugin)
    profile = repository / "qa/inference/cfd_parity_surface_run1_transolver.json"
    profile.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "qa/inference/cfd_parity_surface_run1_transolver.json", profile)
    (repository / "pnms").write_text("immutable public command fixture")
    (repository / "scripts").mkdir()
    (repository / "scripts/plugin_dev.py").write_text(
        "def check_env_plugin(root):\n"
        " return {'workflow_id':'physicsnemo-cfd-surface-benchmark',"
        "'status':'ready','checks':[{'name':'yaml','required':True,'ok':True}]}\n"
        "def ensure_run_local_prerequisites(*, skip_build):\n"
        " assert skip_build is False\n"
    )
    write_json(tmp_path / "inventory.json", {"source": "fixture"})
    for name in ("checkpoint", "stats", "fields"):
        write_json(tmp_path / name, {})
    monkeypatch.setenv("PNMS_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NATIVE_WORKFLOW_ONLY", "1")
    monkeypatch.setattr(
        pnms_stage_job, "stage", lambda *a: {"source_root": str(repository)}
    )
    monkeypatch.setattr(cfd_contract, "verify_file", lambda *a: None)
    monkeypatch.setattr(run_native_cfd_reference, "_verify_inputs", lambda *a: None)
    monkeypatch.setattr(cfd_benchmark_reference, "qualified_reference", lambda *a: None)
    monkeypatch.setattr(
        job,
        "relocate_import",
        lambda *a, **kw: {
            "backend": kw["backend"],
            "model_build_sources": {
                "full": job.IMPORT_ORIGIN,
                "tail": job.IMPORT_ORIGIN,
            },
            "recompiled": False,
        },
    )
    builds = {}
    for runtime in ("aoti", "tensorrt", "python"):
        root = tmp_path / "completed" / runtime
        record = {
            "root": str(root),
            "runtime": "python" if runtime == "python" else "cpp",
            "source_sha": "a" * 40,
            "bundle": "bundle",
            "bundle_digest": "b" * 64,
            "recompiled": False,
            "model_build_sources": {
                "full": job.IMPORT_ORIGIN,
                "tail": job.IMPORT_ORIGIN,
            },
            "environment": {"stack": {"cxx11_abi": True}},
            "loader_environment": {"LD_LIBRARY_PATH": "/selected/providers"},
        }
        write_json(root / "build.json", record)
        if runtime == "python":
            record["python"] = {"plugin_root": "plugin"}
            shutil.copytree(
                repository / "plugins/physicsnemo-cfd-surface-benchmark",
                root / "plugin",
            )
        else:
            write_json(
                root / "bundle/bundle.json",
                {"adapter": "transolver_surface", "backend": runtime},
            )
        builds[runtime] = record
    monkeypatch.setattr(
        job,
        "completed_build",
        lambda project, repository: builds[
            project.name if project.name == "python" else project.parent.name
        ],
    )
    args = job.parser().parse_args(
        [
            "--image",
            "registry/cfd@sha256:" + "f" * 64,
            "--source-sha",
            "a" * 40,
            "--run-id",
            "finite",
            "--output",
            str(tmp_path / "output"),
            "--reference-fields",
            str(tmp_path / "fields"),
            "--input-root",
            str(tmp_path / "inputs"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--stats",
            str(tmp_path / "stats"),
            "--source-inventory",
            str(tmp_path / "inventory.json"),
            "--expected-source-inventory-sha256",
            file_identity(tmp_path / "inventory.json")["sha256"],
            "--expected-runner-sha256",
            file_identity(Path(job.__file__))["sha256"],
        ]
    )
    return args, repository


def test_unready_python_environment_stops_preflight_before_native_import(
    finite_job, monkeypatch
):
    args, repository = finite_job
    (repository / "scripts/plugin_dev.py").write_text(
        "def check_env_plugin(root):\n"
        " return {'workflow_id':'physicsnemo-cfd-surface-benchmark',"
        "'status':'not_ready','checks':["
        "{'name':'physicsnemo.cfd.evaluation.benchmarks.run','required':True,"
        "'ok':False,'detail':'Numba needs NumPy 2.3 or less. Got NumPy 2.4.'},"
        "{'name':'torch_scatter','required':True,'ok':False,"
        "'detail':\"No module named 'torch_scatter'\"}]}\n"
        "def ensure_run_local_prerequisites(*, skip_build):\n"
        " assert skip_build is False\n"
    )

    def native_import(*args, **options):
        raise RuntimeError("native import started with an unready Python workflow")

    monkeypatch.setattr(job, "relocate_import", native_import)
    with pytest.raises(ValueError, match="Python workflow is not ready"):
        job.run(args)
    report = read_json(args.output / "logs/python-readiness.json")
    assert report["status"] == "not_ready"
    assert [row["name"] for row in report["checks"]] == [
        "physicsnemo.cfd.evaluation.benchmarks.run",
        "torch_scatter",
        "serving_prerequisites",
    ]
    assert report["checks"][-1]["ok"] is True
    summary = read_json(args.output / "summary.json")
    assert summary["status"] == "failed" and summary["imports"] == {}
    assert [row["name"] for row in summary["cases"]] == ["preflight"]
    assert "Numba needs NumPy" in summary["error"]
    assert "torch_scatter" in summary["error"]


@pytest.mark.parametrize(
    "missing", ["compatible python package 'scicomp_rq'", "redis-server", "cargo"]
)
def test_missing_serving_prerequisite_stops_before_native_import(
    finite_job, monkeypatch, missing
):
    args, repository = finite_job
    plugin = repository / "scripts/plugin_dev.py"
    plugin.write_text(
        plugin.read_text()
        + f" raise ValueError({'Local run prerequisites are missing: ' + missing!r})\n"
    )

    def native_import(*args, **options):
        raise RuntimeError("native import started before serving prerequisites passed")

    monkeypatch.setattr(job, "relocate_import", native_import)
    with pytest.raises(ValueError, match="Python workflow is not ready"):
        job.run(args)
    report = read_json(args.output / "logs/python-readiness.json")
    assert report["status"] == "not_ready"
    assert report["checks"][-1]["name"] == "serving_prerequisites"
    assert report["checks"][-1]["required"] is True
    assert report["checks"][-1]["ok"] is False
    assert missing in report["checks"][-1]["detail"]
    summary = read_json(args.output / "summary.json")
    assert summary["status"] == "failed" and summary["imports"] == {}
    assert [row["name"] for row in summary["cases"]] == ["preflight"]
    assert missing in summary["error"]


@pytest.mark.parametrize("failed_session", [None, 2])
def test_finite_job_keeps_real_gate_order_and_never_passes_a_failed_session(
    finite_job, monkeypatch, failed_session
):
    args, repository = finite_job
    commands = []
    real_execute = job.execute

    def execute(argv, **options):
        commands.append((argv, options))
        if options["output"].name == "python-readiness.log":
            return real_execute(argv, **options)
        if argv[0] == "cargo":
            assert Path(options["environment"]["NATIVE_RUNNER_LIB_DIR"]).parts[-4:] == (
                "aoti",
                "bundle",
                "workflow",
                "lib",
            )
            assert "NATIVE_WORKFLOW_ONLY" not in options["environment"]
            for name in ("inference_server", "worker-runtime", "native-worker"):
                path = repository / "target/debug" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"current service binary")
                path.chmod(0o755)
        elif "session" in argv:
            ordinal = int(argv[argv.index("--ordinal") + 1])
            if ordinal == failed_session:
                raise ValueError("actual session process exit 9")
            runtime = argv[argv.index("--runtime") + 1]
            write_json(
                Path(argv[argv.index("--output") + 1]) / "session.json",
                {"status": "passed", "runtime": runtime, "ordinal": ordinal},
            )
        elif "compare" in argv:
            write_json(argv[argv.index("--output") + 1], {"status": "passed"})

    monkeypatch.setattr(job, "execute", execute)
    if failed_session is None:
        result = job.run(args)
        assert result["status"] == "passed"
        assert [row["name"] for row in result["cases"]] == job.CASES
        assert len(result["sessions"]) == 6
        assert all(
            record["recompiled"] is False for record in result["builds"].values()
        )
        assert commands[-1][0].count("compare") == 1
    else:
        with pytest.raises(ValueError, match="exit 9"):
            job.run(args)
        result = read_json(args.output / "summary.json")
        assert result["status"] == "failed"
        assert result["cases"][-1]["name"] == "session.2.tensorrt"
        assert result["cases"][-1]["status"] == "failed"
        assert len(result["sessions"]) == 2
        assert not any("compare" in argv for argv, _ in commands)
    builds = [
        (argv, options)
        for argv, options in commands
        if "build" in argv and argv[0] != "cargo"
    ]
    assert len(builds) == 3
    assert builds[0][0][builds[0][0].index("--python") + 1] == sys.executable
    assert builds[1][0][3].endswith("release-tensorrt.yaml")
    assert Path(builds[0][1]["environment"]["NATIVE_DEPLOYMENT_CONFIG"]).parts[-3:] == (
        "imports",
        "aoti",
        "deployment.json",
    )
    assert Path(builds[1][1]["environment"]["NATIVE_DEPLOYMENT_CONFIG"]).parts[-3:] == (
        "imports",
        "tensorrt",
        "deployment.json",
    )


@pytest.mark.parametrize(
    "field", ["expected_runner_sha256", "expected_source_inventory_sha256"]
)
def test_preflight_rejects_stale_driver_or_dirty_source_with_the_same_git_sha(
    finite_job, monkeypatch, field
):
    args, _ = finite_job
    setattr(args, field, "0" * 64)
    monkeypatch.setattr(
        job, "execute", lambda *a, **kw: pytest.fail("stale source launched GPU work")
    )
    with pytest.raises(ValueError, match="differs from the selected version"):
        job.run(args)
    result = read_json(args.output / "summary.json")
    assert result["status"] == "failed"
    assert result["imports"] == {}
    assert result["cases"][0]["status"] == "failed"


def test_native_session_project_selects_completed_bundle_without_original_project(
    imported,  # noqa: F811
    tmp_path,
):
    from pnms_app import build, cfd
    from pnms_app.config import load_project
    from test_pnms_cfd import _installed

    original, config, _ = imported
    completed = tmp_path / "completed"
    seal = cfd.assemble(
        original,
        cfd.import_deployment(original),
        _installed(tmp_path / "installed"),
        completed / "bundle",
        repository=ROOT,
        environment={},
        source_sha="a" * 40,
    )
    receipt = completed / "build.json"
    write_json(
        receipt,
        {
            "status": "complete",
            "runtime": "cpp",
            "name": original["name"],
            "build_id": "completed",
            "bundle": "bundle",
            "bundle_digest": seal["bundle_digest"],
        },
    )
    shutil.rmtree(original["root"])
    shutil.rmtree(Path(config["stats_path"]).parents[1])
    artifact_root = tmp_path / "session/service-artifacts"
    project = job.session_project(
        tmp_path / "session-project",
        runtime="aoti",
        receipt=receipt,
        artifact_root=artifact_root,
        python=Path(sys.executable),
    )
    selected = load_project(project, ROOT)
    assert selected["serve"]["artifact_root"] == str(artifact_root)
    assert selected["models"] == {}
    assert build.last_build(selected)["bundle_digest"] == seal["bundle_digest"]
