# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "application-cli" / "src"))

from pnms_app import python as adapter  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402


@pytest.fixture
def plugin(tmp_path):
    root = tmp_path / "demo"
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/plugin_dev.py"), "init", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"root": root, "plugin": root, "name": "demo", "runtime": "python"}


def test_build_captures_plugin_without_modifying_source(plugin, tmp_path):
    source = Path(plugin["plugin"])
    before = {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    (source / ".pnms/dev").mkdir(parents=True)
    (source / ".pnms/dev/generated.json").write_text("{}")
    report = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)
    snapshot = Path(report["plugin_root"])

    assert snapshot == tmp_path / "build/plugin"
    assert report["runtime"] == {"kind": "python"}
    assert not (snapshot / ".pnms").exists()
    assert all(
        (snapshot / name).read_bytes() == content for name, content in before.items()
    )
    assert all(
        (source / name).read_bytes() == content for name, content in before.items()
    )
    assert not (source / "pnms.yaml").exists()
    assert report["environment"]["python_executable"] == sys.executable


def test_build_rejects_external_symlink(plugin, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("external-content")
    (Path(plugin["plugin"]) / "outside.txt").symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)


def test_build_excludes_ignored_dotenv_paths_before_validation(
    plugin, tmp_path, monkeypatch
):
    source = Path(plugin["plugin"])
    (source / ".gitignore").write_text(".env\n.env.*\n.ENV\n.Env.*\n")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    private_paths = [
        ".env",
        ".env.local",
        "assets/.env",
        "assets/nested/.env.production",
        "assets/.env.backup/credentials.txt",
        ".ENV",
        "assets/.Env.local",
    ]
    # On case-insensitive filesystems .ENV aliases .env; either spelling must
    # be excluded without reading or copying these fake credentials.
    for relative in private_paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PNMS_FAKE_CREDENTIAL=dotenv-regression-sentinel\n")
        subprocess.run(
            ["git", "-C", str(source), "check-ignore", "--quiet", relative],
            check=True,
        )
    resources = [".settings.json", ".environment", ".envrc", "assets/data.json"]
    for relative in resources:
        (source / relative).write_text("ordinary plugin resource\n")

    def validate(selected, _repository, operation):
        assert operation == "validate"
        captured = Path(selected["plugin"])
        assert not any((captured / relative).exists() for relative in private_paths)
        assert all((captured / relative).is_file() for relative in resources)
        return {"validation": {}, "environment": {}}

    monkeypatch.setattr(adapter, "_invoke", validate)
    report = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)

    assert all(
        str(Path(relative)) not in report["inventory"] for relative in private_paths
    )
    assert all(str(Path(relative)) in report["inventory"] for relative in resources)
    assert all((source / relative).is_file() for relative in private_paths)


def test_build_excludes_dotenv_symlink_without_opening_its_target(
    plugin, tmp_path, monkeypatch
):
    outside = tmp_path / "fake-private.txt"
    outside.write_text("PNMS_FAKE_CREDENTIAL=external-regression-sentinel\n")
    link = Path(plugin["plugin"]) / ".env.link"
    link.symlink_to(outside)
    monkeypatch.setattr(
        adapter,
        "_invoke",
        lambda *args: {"validation": {}, "environment": {}},
    )

    report = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)

    assert ".env.link" not in report["inventory"]
    assert not (Path(report["plugin_root"]) / ".env.link").is_symlink()
    assert link.is_symlink()


def test_build_rejects_dotenv_created_by_plugin_validation(
    plugin, tmp_path, monkeypatch
):
    def validate(selected, *_args):
        (Path(selected["plugin"]) / ".env.generated").write_text(
            "PNMS_FAKE_CREDENTIAL=validation-regression-sentinel\n"
        )
        return {"validation": {}, "environment": {}}

    monkeypatch.setattr(adapter, "_invoke", validate)

    with pytest.raises(ValueError, match=r"\.env"):
        adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)

    assert not (tmp_path / "build/plugin").exists()


def test_init_retains_existing_manifest_and_profile_options(tmp_path):
    root = tmp_path / "custom"
    adapter.init_project(
        root,
        "custom",
        repository=REPOSITORY,
        pipeline_profile="postprocess",
        runtime_profile="custom",
        executor_class="custom.execute",
        phase_executor_overrides=["prepare=custom.prepare"],
    )
    manifest = yaml.safe_load((root / "plugin.yaml").read_text())
    assert manifest["pipeline"]["profile"] == "postprocess"
    assert manifest["runtime"]["executor_class"] == "custom.execute"
    assert manifest["runtime"]["prepare_executor_class"] == "custom.prepare"
    assert not (root / "pnms.yaml").exists()


def test_run_uses_user_input_and_does_not_modify_example_fixture(plugin, tmp_path):
    source = Path(plugin["plugin"])
    fixture = source / "examples/default_request.json"
    fixture.write_text('{"value": 3}')
    request = tmp_path / "request.json"
    request.write_text('{"value": 17}')
    report = adapter.run_project(
        plugin, request, tmp_path / "run", repository=REPOSITORY, dev=True
    )

    assert report["result"]["value"] == 17
    assert report["result"]["doubled"] == 34
    assert fixture.read_text() == '{"value": 3}'
    assert report["coverage"]["executed_phases"] == ["prepare", "execute"]
    assert report["coverage"]["service_stages_not_run"] == ["results"]
    assert json.loads(Path(report["result_path"]).read_text())["value"] == 17


def test_normal_run_uses_build_snapshot_and_dev_run_sees_edit(plugin, tmp_path):
    built = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)
    source = Path(plugin["plugin"])
    workflow = source / "workflow.py"
    workflow.write_text(workflow.read_text().replace("value * 2", "value * 3"))
    request = tmp_path / "request.json"
    request.write_text('{"value": 9}')

    normal = adapter.run_project(
        {**plugin, "plugin": Path(built["plugin_root"])},
        request,
        tmp_path / "normal",
        repository=REPOSITORY,
    )
    dev = adapter.run_project(
        plugin, request, tmp_path / "dev", repository=REPOSITORY, dev=True
    )
    assert normal["result"]["doubled"] == 18
    assert dev["result"]["doubled"] == 27
    assert not (Path(built["plugin_root"]) / ".example-output").exists()


@pytest.mark.parametrize("mode", ["dev", "frozen", "check"])
@pytest.mark.parametrize("outcome", [True, False], ids=["outcome", "dictionary"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_postprocess_registered_reports_persist_with_other_artifacts(
    tmp_path, mode, outcome, explicit_output
):
    source = tmp_path / "postprocess"
    adapter.init_project(
        source, "postprocess", repository=REPOSITORY, pipeline_profile="postprocess"
    )
    workflow = source / "workflow.py"
    workflow.write_text(
        """import json
from plugin_sdk import PluginWorkflow, PostprocessOutcome


class ReportWorkflow(PluginWorkflow):
    def execute(self, ctx):
        explicit = ctx["run_dir"] / "execute.json"
        explicit.parent.mkdir(parents=True, exist_ok=True)
        explicit.write_text('{"execute": true}', encoding="utf-8")
        registered = ctx["outputs"].create(
            "data", filename="data.json", media_type="application/json"
        )
        registered.write_text('{"data": true}', encoding="utf-8")
        return {
            "value": ctx["parameters"]["value"],
            "artifacts": [{"name": "execute", "media_type": "application/json",
                           "storage_path": str(explicit)}],
            "output_path": str(explicit),
        }

    def postprocess(self, result, ctx):
        explicit = ctx.run_dir / "summary.json"
        explicit.write_text('{"summary": true}', encoding="utf-8")
        report = ctx.outputs.create(
            "report", filename="report.json", media_type="application/json", primary=True
        )
        report.write_text(json.dumps({"value": result.payload["value"], "version": "original"}), encoding="utf-8")
        payload = {**result.payload, "postprocessed": True,
                   "artifacts": [{"name": "summary", "media_type": "application/json",
                                  "storage_path": str(explicit)}]}
        EXPLICIT_OUTPUT
        RETURN_OUTCOME


WORKFLOW = ReportWorkflow
""".replace(
            "EXPLICIT_OUTPUT",
            'payload["output_path"] = str(explicit)' if explicit_output else "",
        ).replace(
            "RETURN_OUTCOME",
            "return PostprocessOutcome(payload=payload)"
            if outcome
            else "return payload",
        )
    )
    request = source / "examples/default_request.json"
    request.write_text('{"value": 7}')
    project = {"root": source, "plugin": source}
    if mode == "frozen":
        built = adapter.build_project(
            project, tmp_path / "build", repository=REPOSITORY
        )
        project["plugin"] = Path(built["plugin_root"])
        before = {
            path: path.read_bytes()
            for path in project["plugin"].rglob("*")
            if path.is_file()
        }
        workflow.write_text(workflow.read_text().replace('"original"', '"source-edit"'))
    if mode == "check":
        response = adapter.check_project(project, repository=REPOSITORY)
        result = response["example_run"]
    else:
        response = adapter.run_project(
            project, request, tmp_path / "run", repository=REPOSITORY, dev=mode == "dev"
        )
        result = response["result"]
        assert json.loads(Path(response["result_path"]).read_text()) == result

    assert response["status"] in {"succeeded", "ready"}
    assert result["postprocessed"] is True
    artifacts = {artifact["name"]: artifact for artifact in result["artifacts"]}
    assert set(artifacts) == {"execute", "data", "summary", "report"}
    assert artifacts["report"]["media_type"] == "application/json"
    assert artifacts["report"]["primary"] is True
    expected_output = artifacts["summary" if explicit_output else "report"][
        "storage_path"
    ]
    assert result["output_path"] == expected_output
    assert json.loads(Path(artifacts["report"]["storage_path"]).read_text()) == {
        "value": 7,
        "version": "original",
    }
    assert all(
        Path(artifact["storage_path"]).is_file() for artifact in artifacts.values()
    )
    if mode == "frozen":
        assert {
            path: path.read_bytes()
            for path in project["plugin"].rglob("*")
            if path.is_file()
        } == before


@pytest.mark.parametrize("mode", ["dev", "frozen", "check"])
@pytest.mark.parametrize("supplementary", [False, True])
def test_postprocess_preserves_execute_output_selection(tmp_path, mode, supplementary):
    source = tmp_path / "postprocess"
    adapter.init_project(
        source, "postprocess", repository=REPOSITORY, pipeline_profile="postprocess"
    )
    (source / "workflow.py").write_text(
        """from plugin_sdk import PluginWorkflow, PostprocessOutcome


class SelectedWorkflow(PluginWorkflow):
    def execute(self, ctx):
        auxiliary = ctx["outputs"].create(
            "auxiliary", filename="auxiliary.json", media_type="application/json"
        )
        auxiliary.write_text('{"auxiliary": true}', encoding="utf-8")
        selected = ctx["run_dir"] / "selected.json"
        selected.write_text('{"selected": true}', encoding="utf-8")
        return {
            "value": ctx["parameters"]["value"],
            "output_path": str(selected),
            "artifacts": [{"name": "selected", "storage_path": str(selected),
                           "media_type": "application/json", "primary": True}],
        }
SUPPLEMENTARY


WORKFLOW = SelectedWorkflow
""".replace(
            "SUPPLEMENTARY",
            """
    def postprocess(self, result, ctx):
        supplementary = ctx.outputs.create(
            "supplementary", filename="supplementary.json", media_type="application/json"
        )
        supplementary.write_text('{"supplementary": true}', encoding="utf-8")
        return PostprocessOutcome(payload=result.payload)
"""
            if supplementary
            else "",
        )
    )
    request = source / "examples/default_request.json"
    request.write_text('{"value": 7}')
    project = {"root": source, "plugin": source}
    if mode == "frozen":
        built = adapter.build_project(
            project, tmp_path / "build", repository=REPOSITORY
        )
        project["plugin"] = Path(built["plugin_root"])
    if mode == "check":
        result = adapter.check_project(project, repository=REPOSITORY)["example_run"]
    else:
        result = adapter.run_project(
            project,
            request,
            tmp_path / "run",
            repository=REPOSITORY,
            dev=mode == "dev",
        )["result"]

    artifacts = {artifact["name"]: artifact for artifact in result["artifacts"]}
    assert result["output_path"] == artifacts["selected"]["storage_path"]
    assert set(artifacts) == {"selected", "auxiliary"} | (
        {"supplementary"} if supplementary else set()
    )
    assert all(
        json.loads(Path(artifact["storage_path"]).read_text()) == {name: True}
        for name, artifact in artifacts.items()
    )


@pytest.mark.parametrize("mode", ["dev", "frozen", "check"])
@pytest.mark.parametrize("metadata_source", ["execute", "postprocess", "registry"])
def test_postprocess_preserves_explicit_artifact_metadata(
    tmp_path, mode, metadata_source
):
    source = tmp_path / "postprocess"
    adapter.init_project(
        source, "postprocess", repository=REPOSITORY, pipeline_profile="postprocess"
    )
    (source / "workflow.py").write_text(
        """from plugin_sdk import PluginWorkflow, PostprocessOutcome


class MetadataWorkflow(PluginWorkflow):
    def execute(self, ctx):
        selected = ctx["outputs"].create(
            "selected", filename="selected.json", media_type="application/octet-stream",
            primary=METADATA_SOURCE == "registry",
        )
        selected.write_text('{"selected": true}', encoding="utf-8")
        artifact = {
            "name": "selected", "storage_path": str(selected),
            "media_type": "application/json", "size_bytes": selected.stat().st_size,
        }
        if METADATA_SOURCE != "registry":
            artifact["primary"] = True
        return {"output_path": str(selected), "artifacts": [artifact]}
POSTPROCESS_METADATA


WORKFLOW = MetadataWorkflow
""".replace("METADATA_SOURCE", repr(metadata_source)).replace(
            "POSTPROCESS_METADATA",
            """
    def postprocess(self, result, ctx):
        selected = ctx.run_dir / "selected.json"
        return PostprocessOutcome(payload={"artifacts": [{
            "name": "selected", "storage_path": str(selected),
            "media_type": "application/vnd.report+json", "primary": True,
            "size_bytes": selected.stat().st_size, "stage": "postprocess",
        }]})
"""
            if metadata_source == "postprocess"
            else "",
        )
    )
    request = source / "examples/default_request.json"
    request.write_text('{"value": 7}')
    project = {"root": source, "plugin": source}
    if mode == "frozen":
        built = adapter.build_project(
            project, tmp_path / "build", repository=REPOSITORY
        )
        project["plugin"] = Path(built["plugin_root"])
    if mode == "check":
        result = adapter.check_project(project, repository=REPOSITORY)["example_run"]
    else:
        result = adapter.run_project(
            project,
            request,
            tmp_path / "run",
            repository=REPOSITORY,
            dev=mode == "dev",
        )["result"]

    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["media_type"] == (
        "application/vnd.report+json"
        if metadata_source == "postprocess"
        else "application/json"
    )
    assert artifact["primary"] is True
    assert artifact["size_bytes"] == Path(artifact["storage_path"]).stat().st_size
    if metadata_source == "postprocess":
        assert artifact["stage"] == "postprocess"
    assert json.loads(Path(artifact["storage_path"]).read_text()) == {"selected": True}


def test_run_rejects_invalid_input_before_hook_executes(plugin, tmp_path):
    request = tmp_path / "request.json"
    request.write_text('{"value": "wrong-type"}')
    with pytest.raises(ValueError, match="request|Request"):
        adapter.run_project(
            plugin, request, tmp_path / "run", repository=REPOSITORY, dev=True
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [("missing", "unknown next stage 'missing'"), ("prepare", "cycle.*'prepare'")],
)
def test_prepare_transition_rejects_unknown_targets_and_cycles(
    plugin, tmp_path, target, message
):
    workflow = Path(plugin["plugin"]) / "workflow.py"
    workflow.write_text(
        workflow.read_text()
        + f'\n\ndef prepare(ctx):\n    return {{"next_stage_id": {target!r}}}\n'
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 7}')
    with pytest.raises(ValueError, match=message):
        adapter.run_project(
            plugin, request, tmp_path / "run", repository=REPOSITORY, dev=True
        )


def test_check_reports_partial_readiness_without_claiming_full_pass(plugin):
    source = Path(plugin["plugin"])
    manifest_path = source / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["developer"] = {
        "readiness": {
            "recommended_check_phase": "prepare",
            "env": [{"name": "PNMS_TEST_DELIBERATELY_MISSING_ENV", "required": True}],
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    report = adapter.check_project(plugin, repository=REPOSITORY)
    assert report["status"] == "needs_setup"
    assert report["coverage"]["executed_phases"] == ["prepare"]
    assert report["coverage"]["complete_workflow"] is False
    assert report["_exit_code"] != 0


def test_check_config_only_does_not_execute_hooks(plugin):
    report = adapter.check_project(plugin, repository=REPOSITORY, config_only=True)
    assert report["status"] == "valid"
    assert report["coverage"]["executed_phases"] == []
    assert report["coverage"]["complete_workflow"] is False


def test_direct_run_rejects_fanout_with_service_action(tmp_path):
    source = tmp_path / "ensemble"
    adapter.init_project(
        source, "ensemble", repository=REPOSITORY, pipeline_profile="ensemble"
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="fanout.*pnms serve"):
        adapter.run_project(
            {"root": source, "plugin": source},
            request,
            tmp_path / "run",
            repository=REPOSITORY,
            dev=True,
        )


def test_serve_dev_reuses_local_topology_without_docker_or_native_builder(
    plugin, tmp_path
):
    workspace = tmp_path / "service"
    report = adapter.serve_project(
        plugin,
        repository=REPOSITORY,
        workspace=workspace,
        port=18081,
        redis_port=16381,
        dry_run=True,
    )
    assert report["status"] == "planned"
    assert report["port"] == 18081
    assert report["redis_port"] == 16381
    assert {process["name"] for process in report["processes"]} >= {
        "redis",
        "inference_server",
    }
    assert all("docker" not in process["argv"] for process in report["processes"])
    runtime_config = json.loads((workspace / "runtime_config.json").read_text())
    assert (
        runtime_config["python_runtime_envs"]["python.test"]["python_executable"]
        == sys.executable
    )
    server = next(
        process
        for process in report["processes"]
        if process["name"] == "inference_server"
    )
    assert server["env"]["PLUGIN_DIR"] == str(Path(plugin["plugin"]).parent)
    assert server["env"]["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"] == "demo"


@pytest.mark.parametrize("workspace_kind", ["default", "internal", "external"])
def test_serve_dev_preserves_adjacent_dotenv(plugin, tmp_path, workspace_kind):
    source = Path(plugin["plugin"])
    credential = "PNMS_FAKE_CREDENTIAL=dev-regression-sentinel\n"
    (source / ".env").write_text(credential)
    workflow = source / "workflow.py"
    workflow.write_text(
        workflow.read_text()
        + "\nfrom pathlib import Path\n"
        + f'assert Path(__file__).with_name(".env").read_text() == {credential!r}\n'
    )
    workspace = {
        "default": None,
        "internal": source / "service-output",
        "external": tmp_path / "service",
    }[workspace_kind]

    report = adapter.serve_project(
        plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
    )

    assert report["status"] == "planned"
    assert (source / ".env").read_text() == credential


def test_python_dev_service_process_receives_loopback_binding(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "plugin_dev.py").write_text(
        "import os\n"
        "def run_local_plugin(*args, **kwargs):\n"
        "    return {'bind_host': os.environ.get('PHYSICSNEMO_SERVE_BIND_HOST')}\n"
    )
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    monkeypatch.setenv("PHYSICSNEMO_SERVE_BIND_HOST", "0.0.0.0")
    report = adapter.serve_project(
        {"root": plugin, "plugin": plugin},
        repository=repository,
        workspace=tmp_path / "workspace",
        dry_run=True,
    )
    assert report["bind_host"] == "127.0.0.1"


@pytest.mark.parametrize("location", ["snapshot", "renamed", "snapshot-postprocess"])
def test_local_serve_discovers_plugin_by_manifest_id_without_authored_source(
    plugin, tmp_path, location
):
    source = Path(plugin["plugin"])
    if location == "snapshot-postprocess":
        adapter.init_project(
            tmp_path / "postprocess",
            "postprocess",
            repository=REPOSITORY,
            pipeline_profile="postprocess",
        )
        source = tmp_path / "postprocess"
        plugin = {**plugin, "root": source, "plugin": source, "name": "postprocess"}
    if location == "renamed":
        selected = source.rename(tmp_path / "different-directory")
    else:
        built = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)
        selected = Path(built["plugin_root"])
        shutil.rmtree(source)
    manifest = (selected / "plugin.yaml").read_bytes()
    workflow_id = yaml.safe_load(manifest)["metadata"]["id"]
    report = adapter.serve_project(
        {**plugin, "plugin": selected},
        repository=REPOSITORY,
        workspace=tmp_path / "service",
        port=0,
        redis_port=0,
        dry_run=True,
    )
    server = next(
        row for row in report["processes"] if row["name"] == "inference_server"
    )
    for resolver in (
        "from plugin_runtime import resolve_plugin_manifest as resolve",
        "from inference_worker import _resolve_plugin_manifest as resolve",
    ):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                resolver + "; import json,sys; "
                "root,manifest=resolve(sys.argv[1]); "
                "print(json.dumps({'root':str(root),'id':manifest['metadata']['id']}))",
                workflow_id,
            ],
            env={
                **os.environ,
                **server["env"],
                "PYTHONPATH": str(REPOSITORY / "scripts"),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert probe.returncode == 0, probe.stderr
        discovered = json.loads(probe.stdout)
        assert discovered["id"] == workflow_id
        assert (Path(discovered["root"]) / "workflow.py").read_bytes() == (
            selected / "workflow.py"
        ).read_bytes()
    assert (selected / "plugin.yaml").read_bytes() == manifest


def test_postprocess_result_operations_require_service(plugin, tmp_path):
    root = tmp_path / "post"
    adapter.init_project(
        root, "post", repository=REPOSITORY, pipeline_profile="postprocess"
    )
    workflow = root / "workflow.py"
    workflow.write_text(
        workflow.read_text().replace(
            'return PostprocessOutcome(payload=final_payload, status="succeeded")',
            'return {"result_ops": [{"op": "test"}], **final_payload}',
        )
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="postprocess result operations.*pnms serve"):
        adapter.run_project(
            {"root": root, "plugin": root},
            request,
            tmp_path / "run",
            repository=REPOSITORY,
            dev=True,
        )


def test_full_check_rejects_batching_but_prepare_only_reports_coverage(tmp_path):
    root = tmp_path / "batch"
    adapter.init_project(root, "batch", repository=REPOSITORY, pipeline_profile="batch")
    project = {"root": root, "plugin": root}
    partial = adapter.check_project(
        project, repository=REPOSITORY, through_phase="prepare"
    )
    assert partial["coverage"]["executed_phases"] == ["prepare"]
    assert partial["coverage"]["complete_workflow"] is False
    with pytest.raises(ValueError, match="batching.*pnms serve"):
        adapter.check_project(project, repository=REPOSITORY)


def test_serve_default_workspace_avoids_copying_itself_into_legacy_cpu_plugin(
    plugin, monkeypatch
):
    source = Path(plugin["plugin"])
    generated = source / ".pnms/dev/old-session/logs"
    generated.mkdir(parents=True)
    (generated / "server.log").write_text("prior generated service output")
    captured = {}

    def invoke(project, _repository, _operation, **options):
        captured.update(plugin=Path(project["plugin"]), options=options)
        return {"status": "planned"}

    monkeypatch.setattr(adapter, "_invoke", invoke)
    adapter.serve_project(plugin, repository=REPOSITORY, dry_run=True)
    assert captured["plugin"] != source
    assert (captured["plugin"] / "workflow.py").read_bytes() == (
        source / "workflow.py"
    ).read_bytes()
    assert not (captured["plugin"] / ".pnms").exists()


@pytest.mark.parametrize(
    "relative_workspace", ["service-output", "assets/service-output"]
)
def test_serve_custom_workspace_copies_source_without_recursive_service_output(
    plugin, monkeypatch, relative_workspace
):
    source = Path(plugin["plugin"])
    assets = source / "assets"
    assets.mkdir()
    (assets / "factor.json").write_bytes(b"3")
    original = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    workspace = source / relative_workspace
    workspace.mkdir(parents=True)
    prior_output = workspace / "prior.log"
    prior_output.write_bytes(b"prior service output")
    captured = {}

    def invoke(project, _repository, operation, **options):
        captured.update(plugin=Path(project["plugin"]), operation=operation, **options)
        return {"status": "planned"}

    monkeypatch.setattr(adapter, "_invoke", invoke)
    recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(90)
        report = adapter.serve_project(
            plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
        )
    finally:
        sys.setrecursionlimit(recursion_limit)

    staged = workspace / "source" / source.name
    assert report == {"status": "planned"}
    assert captured["plugin"] == staged
    assert captured["operation"] == "serve"
    assert captured["workspace"] == str(workspace)
    assert not (staged / relative_workspace).exists()
    assert prior_output.read_bytes() == b"prior service output"
    assert all(
        (staged / name).read_bytes() == content for name, content in original.items()
    )
    assert all(
        (source / name).read_bytes() == content for name, content in original.items()
    )


@pytest.mark.parametrize(
    "relative_workspace", ["service-output", "assets/service-output"]
)
def test_serve_reuses_custom_workspace_without_overwriting_previous_snapshot(
    plugin, monkeypatch, relative_workspace
):
    source = Path(plugin["plugin"])
    workspace = source / relative_workspace
    workspace.mkdir(parents=True)
    prior_output = workspace / "prior.log"
    prior_output.write_bytes(b"prior service output")
    workflow = source / "workflow.py"
    original_workflow = workflow.read_bytes()
    snapshots = []

    def invoke(project, *_args, **_options):
        snapshots.append(Path(project["plugin"]))
        return {"status": "planned"}

    monkeypatch.setattr(adapter, "_invoke", invoke)
    adapter.serve_project(
        plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
    )
    updated_workflow = original_workflow + b"\n# edited between service sessions\n"
    workflow.write_bytes(updated_workflow)
    adapter.serve_project(
        plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
    )

    assert len(snapshots) == 2
    assert snapshots[0] != snapshots[1]
    assert (snapshots[0] / "workflow.py").read_bytes() == original_workflow
    assert (snapshots[1] / "workflow.py").read_bytes() == updated_workflow
    assert not (snapshots[1] / relative_workspace).exists()
    assert workflow.read_bytes() == updated_workflow
    assert prior_output.read_bytes() == b"prior service output"


@pytest.mark.parametrize("link_in_workspace", [False, True])
def test_serve_custom_workspace_validates_symlinks_only_in_copied_source(
    plugin, tmp_path, monkeypatch, link_in_workspace
):
    source = Path(plugin["plugin"])
    workspace = source / "assets/service-output"
    workspace.mkdir(parents=True)
    target = tmp_path / "outside.txt"
    target.write_text("external content")
    link = (workspace if link_in_workspace else workspace.parent) / "shortcut"
    link.symlink_to(target)
    invoked = []

    def invoke(*args, **options):
        invoked.append(options)
        return {"status": "planned"}

    monkeypatch.setattr(adapter, "_invoke", invoke)
    if link_in_workspace:
        report = adapter.serve_project(
            plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
        )
        assert report == {"status": "planned"}
        assert link.is_symlink()
        assert not (
            workspace / "source" / source.name / "assets/service-output"
        ).exists()
    else:
        with pytest.raises(ValueError, match="symlink"):
            adapter.serve_project(
                plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
            )
        assert not invoked


@pytest.mark.parametrize("escapes_workspace", [False, True])
def test_serve_staging_alias_stays_inside_workspace(
    plugin, monkeypatch, escapes_workspace
):
    source = Path(plugin["plugin"])
    workflow = (source / "workflow.py").read_bytes()
    workspace = source / "service-output"
    workspace.mkdir()
    target = source / "assets" if escapes_workspace else workspace / "snapshots"
    target.mkdir()
    staging_alias = workspace / "source"
    staging_alias.symlink_to(target, target_is_directory=True)
    snapshots = []

    def invoke(project, *_args, **_options):
        snapshots.append(Path(project["plugin"]))
        return {"status": "planned"}

    monkeypatch.setattr(adapter, "_invoke", invoke)
    recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(90)
        if escapes_workspace:
            with pytest.raises(ValueError, match="staging.*workspace"):
                adapter.serve_project(
                    plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
                )
            assert not snapshots
            assert not list(target.iterdir())
        else:
            report = adapter.serve_project(
                plugin, repository=REPOSITORY, workspace=workspace, dry_run=True
            )
            assert report == {"status": "planned"}
            assert snapshots[0].resolve().is_relative_to(workspace)
            assert (snapshots[0] / "workflow.py").read_bytes() == workflow
    finally:
        sys.setrecursionlimit(recursion_limit)
    assert staging_alias.is_symlink()
    assert (source / "workflow.py").read_bytes() == workflow


def test_serve_rejects_plugin_directory_workspace_before_copying(plugin, monkeypatch):
    source = Path(plugin["plugin"])
    existing_source = source / "source"
    existing_source.mkdir()
    monkeypatch.setattr(
        adapter, "_invoke", lambda *args, **kwargs: pytest.fail("invoked")
    )
    recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(90)
        with pytest.raises(ValueError, match="workspace.*plugin directory"):
            adapter.serve_project(
                plugin, repository=REPOSITORY, workspace=source, dry_run=True
            )
    finally:
        sys.setrecursionlimit(recursion_limit)
    assert not list(existing_source.iterdir())


def test_declared_python_environment_is_used_for_plugin_checks(
    plugin, tmp_path, monkeypatch
):
    selected = tmp_path / "chosen-python"
    selected = python_executable(
        selected,
        f"import os,sys\nos.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
    )
    commands = []
    original = subprocess.run

    def record(argv, **kwargs):
        commands.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(adapter.subprocess, "run", record)
    adapter.check_project(
        {**plugin, "build": {"python": str(selected)}},
        repository=REPOSITORY,
        config_only=True,
    )
    assert commands[0][0] == str(selected)


def test_failed_python_result_is_not_reported_as_a_successful_check(plugin, tmp_path):
    workflow = Path(plugin["plugin"]) / "workflow.py"
    workflow.write_text(
        workflow.read_text()
        + '\n\ndef execute(ctx):\n    return {"status": "failed", "value": 7, "doubled": 14}\n'
    )
    request = tmp_path / "request.json"
    request.write_text('{"value": 7}')
    report = adapter.run_project(
        plugin, request, tmp_path / "run", repository=REPOSITORY, dev=True
    )
    assert report["status"] == "failed"
    assert report.get("_exit_code", 0) != 0
    checked = adapter.check_project(plugin, repository=REPOSITORY)
    assert checked["status"] == "failed"
    assert checked.get("_exit_code", 0) != 0


def test_built_python_run_reads_snapshot_relative_assets(plugin, tmp_path):
    source = Path(plugin["plugin"])
    (source / "factor.json").write_text("2")
    workflow = source / "workflow.py"
    workflow.write_text(
        workflow.read_text()
        + '\n\ndef execute(ctx):\n    import json\n    from pathlib import Path\n    value = ctx["parameters"]["value"]\n    return {"value": value, "doubled": value * json.loads(Path("factor.json").read_text())}\n'
    )
    built = adapter.build_project(plugin, tmp_path / "build", repository=REPOSITORY)
    (source / "factor.json").write_text("3")
    request = tmp_path / "request.json"
    request.write_text('{"value": 7}')
    normal = adapter.run_project(
        {**plugin, "plugin": Path(built["plugin_root"])},
        request,
        tmp_path / "normal",
        repository=REPOSITORY,
    )
    dev = adapter.run_project(
        plugin, request, tmp_path / "dev", repository=REPOSITORY, dev=True
    )
    assert normal["result"]["doubled"] == 14
    assert dev["result"]["doubled"] == 21


@pytest.fixture
def multipart_plugin(plugin, tmp_path):
    source = Path(plugin["plugin"])
    marker = tmp_path / "multipart-hooks-called"
    manifest_path = source / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["ingress"] = {
        "content_type": "multipart/form-data",
        "form_schema_inline": {
            "type": "object",
            "required": ["note"],
            "properties": {"note": {"type": "string"}},
            "additionalProperties": False,
        },
        "files": [
            {
                "name": "sample",
                "required": True,
                "media_types": ["application/octet-stream"],
                "max_size_mb": 1,
            },
            {
                "name": "optional",
                "required": False,
                "media_types": ["text/plain"],
                "max_size_mb": 1,
            },
        ],
    }
    manifest.setdefault("outputs", {}).pop("result_schema", None)
    manifest["outputs"]["result_schema_inline"] = {"type": "object"}
    manifest_path.write_text(yaml.safe_dump(manifest))
    (source / "workflow.py").write_text(
        "from pathlib import Path\n"
        "def prepare(ctx):\n"
        f"    Path({str(marker)!r}).write_text('prepare invoked')\n"
        "    return {}\n"
        "def execute(ctx):\n"
        "    return {'status': 'succeeded', 'note': ctx['parameters'].get('note'),\n"
        "            'uploads': [item['field_name'] for item in ctx['request']['input_artifacts']]}\n"
    )
    (source / "sample.bin").write_bytes(b"sample bytes")
    (source / "optional.txt").write_text("optional upload")
    (source / "examples/default_request.multipart.json").write_text(
        json.dumps(
            {"form_fields": {"note": "example"}, "files": {"sample": "sample.bin"}}
        )
    )
    return plugin, marker


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "violation,diagnostic",
    [
        ("form-schema", "note|schema|required"),
        ("missing-file", "required.*sample|sample.*required"),
        ("unknown-file", "unknown|unexpected"),
        ("media-type", "media.type"),
        ("oversized", "max_size|size|limit"),
        ("zero-limit", "max_size|size|limit"),
    ],
)
def test_multipart_run_rejects_invalid_contract_before_hooks(
    multipart_plugin, tmp_path, frozen, violation, diagnostic
):
    project, marker = multipart_plugin
    if violation == "zero-limit":
        manifest_path = Path(project["plugin"]) / "plugin.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["ingress"]["files"][0]["max_size_mb"] = 0
        manifest_path.write_text(yaml.safe_dump(manifest))
    if frozen:
        built = adapter.build_project(
            project, tmp_path / "build", repository=REPOSITORY
        )
        project = {**project, "plugin": Path(built["plugin_root"])}
    request = tmp_path / "request.json"
    upload = tmp_path / "sample.bin"
    upload.write_bytes(b"sample bytes")
    body = {"form_fields": {"note": "accepted form"}, "files": {"sample": upload.name}}
    if violation == "form-schema":
        body["form_fields"] = {}
    elif violation == "missing-file":
        body["files"] = {}
    elif violation == "unknown-file":
        body["files"]["unknown"] = upload.name
    elif violation == "media-type":
        upload = upload.rename(tmp_path / "sample.png")
        body["files"]["sample"] = upload.name
    elif violation == "oversized":
        upload.write_bytes(b"x" * (1024 * 1024 + 1))
    request.write_text(json.dumps(body))
    assert not marker.exists(), "plugin snapshot validation must not invoke hooks"

    with pytest.raises(ValueError, match=diagnostic):
        adapter.run_project(
            project, request, tmp_path / "run", repository=REPOSITORY, dev=not frozen
        )

    assert not marker.exists(), "invalid multipart requests must not invoke user hooks"


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("optional", [False, True])
def test_multipart_run_accepts_declared_optional_uploads_and_relative_paths(
    multipart_plugin, tmp_path, frozen, optional
):
    project, marker = multipart_plugin
    if frozen:
        built = adapter.build_project(
            project, tmp_path / "build", repository=REPOSITORY
        )
        project = {**project, "plugin": Path(built["plugin_root"])}
    # The required upload resolves beside the external request. The optional
    # upload resolves from the selected plugin source or frozen snapshot.
    (tmp_path / "sample.bin").write_bytes(b"request relative upload")
    files = {"sample": "sample.bin"}
    if optional:
        files["optional"] = "optional.txt"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"form_fields": {"note": "accepted form"}, "files": files})
    )

    report = adapter.run_project(
        project, request, tmp_path / "run", repository=REPOSITORY, dev=not frozen
    )

    assert report["status"] == "succeeded"
    assert report["result"]["note"] == "accepted form"
    assert set(report["result"]["uploads"]) == set(files)
    assert marker.read_text() == "prepare invoked"
