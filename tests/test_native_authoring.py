"""Exercise the native authoring commands through their actual public CLI."""

import json
import shutil
from pathlib import Path, PureWindowsPath
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/plugin_dev.py"), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def scaffold(tmp_path, template="tensor"):
    path = tmp_path / "my-native-model"
    result = cli("init", path, "--runtime", "cpp", "--template", template)
    assert result.returncode == 0, result.stderr
    return path, json.loads(result.stdout)


def test_tensor_scaffold_validates_without_python_or_compiled_packages(tmp_path):
    path, report = scaffold(tmp_path)
    assert report["runtime_profile"] == "cpp"
    assert report["template"] == "tensor"
    assert not (path / "workflow.py").exists()
    assert not (path / "native").exists()
    manifest = yaml.safe_load((path / "plugin.yaml").read_text())
    assert manifest["runtime"]["entrypoint"] == "tensor.v1"
    assert manifest["configuration"]["native"]["tensor_adapter"]["inputs"] == {
        "input": {"request_field": "values", "codec": "json-array-fp32"}
    }
    assert [x["phase"] for x in manifest["pipeline"]["stages"]] == [
        "schedule",
        "execute",
        "results",
    ]
    assert (path / "model/model-build.json").is_file()
    assert (path / "fixtures/self-test.json").is_file()
    checked = cli("validate", path, "--release-profile", "release.yaml")
    assert checked.returncode == 0, checked.stderr
    validation = json.loads(checked.stdout)
    assert validation["status"] == "valid"
    assert validation["native"]["adapter"] == "tensor.v1"
    assert validation["native"]["configuration_required"] == ["output.repository"]


def test_custom_scaffold_links_factory_without_core_source_changes(tmp_path):
    path, _ = scaffold(tmp_path, "cpp")
    manifest = yaml.safe_load((path / "plugin.yaml").read_text())
    assert manifest["runtime"]["entrypoint"] == "custom.v1"
    source = (path / "native/workflow.cpp").read_text()
    assert "create_custom_workflow" in source
    assert 'context.infer("primary"' in source
    assert "redis" not in source.lower()
    assert (
        "PhysicsNeMoWorkflow::authoring" in (path / "native/CMakeLists.txt").read_text()
    )
    checked = cli("validate", path)
    assert checked.returncode == 0, checked.stderr


def test_native_tensor_multipart_scaffold_has_independent_raw_input_cases(tmp_path):
    import native_workflow
    import struct

    path = tmp_path / "multipart-native"
    native_workflow.init_native(path, content_type="multipart/form-data")
    manifest = yaml.safe_load((path / "plugin.yaml").read_text())
    assert manifest["ingress"]["content_type"] == "multipart/form-data"
    assert (
        "form_schema" in manifest["ingress"]
        and "json_schema" not in manifest["ingress"]
    )
    assert (
        manifest["configuration"]["native"]["tensor_adapter"]["inputs"]["input"][
            "codec"
        ]
        == "raw-fp32-le"
    )
    resolved = native_workflow.resolve_release(path)
    cases = resolved["qualification"]["cases_document"]["cases"]
    assert len(cases) == 2 and cases[0]["expected"] != cases[1]["expected"]
    expected_inputs = [[0, 1, -1, 4], [2.5, -8, 0.125, 13]]
    for case, values in zip(cases, expected_inputs, strict=True):
        source = path / case["files"]["features"]["source"]
        assert list(struct.unpack("<4f", source.read_bytes())) == values
        assert case["request"]["values"]["artifact"] == "features"
    assert (
        resolved["startup_cases"][0]["input_artifacts"]["features"]["size_bytes"] == 16
    )
    checked = cli("validate", path)
    assert checked.returncode == 0, checked.stderr


def test_public_native_init_cli_selects_multipart_contract(tmp_path):
    path = tmp_path / "native-multipart-cli"
    created = cli(
        "init",
        path,
        "--runtime",
        "cpp",
        "--template",
        "tensor",
        "--content-type",
        "multipart/form-data",
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["content_type"] == "multipart/form-data"
    assert (
        yaml.safe_load((path / "plugin.yaml").read_text())["ingress"]["content_type"]
        == "multipart/form-data"
    )
    checked = cli("validate", path)
    assert checked.returncode == 0, checked.stderr


def test_native_custom_multipart_is_rejected_before_creating_or_replacing_directory(
    tmp_path,
):
    import native_workflow

    path = tmp_path / "unsupported-custom"
    with pytest.raises(ValueError, match="multipart.*tensor"):
        native_workflow.init_native(
            path, template="cpp", content_type="multipart/form-data"
        )
    assert not path.exists()
    path.mkdir()
    (path / "authored.txt").write_text("keep me")
    with pytest.raises(ValueError, match="multipart.*tensor"):
        native_workflow.init_native(
            path, template="cpp", content_type="multipart/form-data", force=True
        )
    assert (path / "authored.txt").read_text() == "keep me"


def test_startup_rejects_empty_or_nested_empty_expected_without_artifact_assertions(
    tmp_path,
):
    import native_workflow

    path, _ = scaffold(tmp_path, "cpp")
    (path / "schemas/result.json").write_text(json.dumps({"type": "object"}))
    cases_path = path / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    for expected in [{}, {"nested": {}}]:
        cases["cases"][0]["expected"] = expected
        cases_path.write_text(json.dumps(cases))
        with pytest.raises(ValueError, match="startup.*assert"):
            native_workflow.resolve_release(path)
    cases["cases"][0]["expected_artifacts"] = {
        "report": {"sha256": "a" * 64, "size_bytes": 0}
    }
    cases_path.write_text(json.dumps(cases))
    assert native_workflow.resolve_release(path)["startup_cases"][0][
        "expected_artifacts"
    ]


def test_native_validation_reports_unknown_tensor_against_model_contract(tmp_path):
    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    inputs = manifest["configuration"]["native"]["tensor_adapter"]["inputs"]
    inputs["wrong_tensor"] = inputs.pop("input")
    manifest_path.write_text(yaml.safe_dump(manifest))
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "input_names" in checked.stderr and "wrong_tensor" in checked.stderr


def test_release_profile_paths_cannot_escape_workflow(tmp_path):
    path, _ = scaffold(tmp_path)
    profile = path / "release.yaml"
    value = yaml.safe_load(profile.read_text())
    value["model_project"] = "../other/model-build.json"
    profile.write_text(yaml.safe_dump(value))
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "model_project" in checked.stderr and "inside" in checked.stderr


def test_native_scaffold_refuses_to_overwrite_existing_work(tmp_path):
    path, _ = scaffold(tmp_path)
    original = (path / "plugin.yaml").read_bytes()
    checked = cli("init", path, "--runtime", "cpp", "--template", "tensor")
    assert checked.returncode != 0
    assert "already exists" in checked.stderr
    assert (path / "plugin.yaml").read_bytes() == original


def configure_registry(path):
    profile = path / "release.yaml"
    value = yaml.safe_load(profile.read_text())
    value["output"]["repository"] = "registry.example/team/my-native-model"
    profile.write_text(yaml.safe_dump(value))


def select_backend(path, backend, *, profile=None):
    release_path = path / "release.yaml"
    release = yaml.safe_load(release_path.read_text())
    release["backend"] = backend
    release_path.write_text(yaml.safe_dump(release))
    project_path = path / "model/model-build.json"
    project = json.loads(project_path.read_text())
    project["backends"] = ["aoti", "tensorrt"]
    if profile is not None:
        project["tensorrt_profile"] = profile
    project_path.write_text(json.dumps(project))


def test_tensorrt_release_is_selected_and_has_a_distinct_compilation_key(tmp_path):
    path, _ = scaffold(tmp_path)
    configure_registry(path)
    select_backend(path, "aoti")
    aoti = cli("release", path, "--plan")
    assert aoti.returncode == 0, aoti.stderr
    select_backend(path, "tensorrt")
    trt = cli("release", path, "--plan")
    assert trt.returncode == 0, trt.stderr
    assert (
        json.loads(trt.stdout)["models"]["primary"]["compilation_key"]
        != (json.loads(aoti.stdout)["models"]["primary"]["compilation_key"])
    )


def test_tensorrt_exact_profile_requires_plugins_during_cpu_validation(tmp_path):
    path, _ = scaffold(tmp_path)
    select_backend(path, "tensorrt", profile="layout-order-exact-v2")
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "tensorrt_exact_linear_plugin" in checked.stderr
    assert "layout-order-exact-v2" in checked.stderr


def test_provider_math_profile_rejects_aoti_during_cpu_validation(tmp_path):
    path, _ = scaffold(tmp_path)
    select_backend(path, "aoti", profile="provider-default-exact-v1")
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "TensorRT-only" in checked.stderr


def test_selected_tensorrt_backend_must_be_declared_by_project(tmp_path):
    path, _ = scaffold(tmp_path)
    profile_path = path / "release.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["backend"] = "tensorrt"
    profile_path.write_text(yaml.safe_dump(profile))
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "not declared" in checked.stderr


def test_native_init_selects_tensorrt_and_matched_immutable_stack(tmp_path):
    path = tmp_path / "trt-model"
    image = "registry.example/native-stack@sha256:" + "d" * 64
    created = cli(
        "init",
        path,
        "--runtime",
        "cpp",
        "--backend",
        "tensorrt",
        "--native-image",
        image,
    )
    assert created.returncode == 0, created.stderr
    profile = yaml.safe_load((path / "release.yaml").read_text())
    project = json.loads((path / "model/model-build.json").read_text())
    assert profile["backend"] == "tensorrt"
    assert profile["images"] == {"builder": image, "server": image}
    assert project["backends"] == ["tensorrt"]
    assert project["tensorrt_profile"] == "baseline"
    checked = cli("validate", path)
    assert checked.returncode == 0, checked.stderr


def test_cfd_tensorrt_release_derives_matching_worker_class_and_queue(tmp_path):
    import native_workflow

    path = tmp_path / "cfd"
    shutil.copytree(ROOT / "plugins/physicsnemo-cfd-transolver-native", path)
    profile_path = path / "release.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["backend"] = "tensorrt"
    profile["runtime"]["executor_class"] = "native.transolver.tensorrt"
    profile_path.write_text(yaml.safe_dump(profile))
    resolved = native_workflow.resolve_release(path)
    manifest = resolved["plugin"]["document"]
    assert manifest["runtime"]["executor_class"] == "native.transolver.tensorrt"
    execute = next(x for x in manifest["pipeline"]["stages"] if x["phase"] == "execute")
    assert execute["queue"] == "execute.native.transolver.tensorrt"


def test_release_plan_tracks_model_bytes_but_not_acceptance_fixture_edits(tmp_path):
    path, _ = scaffold(tmp_path)
    configure_registry(path)
    before = cli("release", path, "--plan", "--cache-dir", tmp_path / "cache")
    assert before.returncode == 0, before.stderr
    first = json.loads(before.stdout)
    assert first["models"]["primary"]["action"] == "compile"
    assert first["final_image_qualification_required"] is True
    key = first["models"]["primary"]["compilation_key"]
    cases_path = path / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    cases["cases"][0]["expected"]["values"][0] = 1.00001
    cases_path.write_text(json.dumps(cases))
    second = cli("release", path, "--plan", "--cache-dir", tmp_path / "cache")
    assert second.returncode == 0, second.stderr
    second = json.loads(second.stdout)
    assert second["models"]["primary"]["compilation_key"] == key
    assert second["qualification_policy_key"] != first["qualification_policy_key"]
    adapter = path / "model/adapter.py"
    adapter.write_text(adapter.read_text().replace("2.0 + 1.0", "3.0 + 1.0"))
    third = cli("release", path, "--plan", "--cache-dir", tmp_path / "cache")
    assert third.returncode == 0, third.stderr
    assert json.loads(third.stdout)["models"]["primary"]["compilation_key"] != key


def test_release_plan_normalizes_windows_qualification_fixture_paths(
    tmp_path, monkeypatch
):
    import native_workflow

    path, _ = scaffold(tmp_path)
    configure_registry(path)
    host_contained = native_workflow.contained

    class WindowsRelativePath(type(path)):
        def relative_to(self, *other):
            return PureWindowsPath(super().relative_to(*other))

    def contained_with_windows_fixture_paths(root, relative, label, **kwargs):
        result = host_contained(root, relative, label, **kwargs)
        if label in {"qualification.cases", "qualification.self_test"}:
            return WindowsRelativePath(result)
        return result

    monkeypatch.setattr(
        native_workflow, "contained", contained_with_windows_fixture_paths
    )
    resolved = native_workflow.resolve_release(path, require_output=True)
    plan = native_workflow.release_module().plan_release(resolved, tmp_path / "cache")
    assert plan["status"] == "planned"
    assert plan["models"]["primary"]["action"] == "compile"
    assert [item["path"] for item in resolved["qualification"]["fixtures"]] == [
        "fixtures/cases.json",
        "fixtures/self-test.json",
    ]


def test_raw_codec_cannot_use_array_schema(tmp_path):
    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["configuration"]["native"]["tensor_adapter"]["inputs"]["input"][
        "codec"
    ] = "raw-fp32-le"
    manifest_path.write_text(yaml.safe_dump(manifest))
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "raw-fp32-le" in checked.stderr and "object" in checked.stderr


def test_native_multipart_schema_and_file_rules_resolve_without_json_schema(tmp_path):
    import native_workflow

    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    ingress = manifest["ingress"]
    ingress["content_type"] = "multipart/form-data"
    ingress["form_schema"] = ingress.pop("json_schema")
    ingress["files"] = [
        {
            "name": "features",
            "required": True,
            "media_types": ["application/octet-stream"],
            "max_size_mb": 1,
        }
    ]
    manifest_path.write_text(yaml.safe_dump(manifest))
    (path / "fixtures/features.bin").write_bytes(b"fixture input")
    cases_path = path / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    for case in cases["cases"]:
        case["files"] = {
            "features": {
                "source": "fixtures/features.bin",
                "media_type": "application/octet-stream",
            }
        }
    cases_path.write_text(json.dumps(cases))
    resolved = native_workflow.resolve_release(path)
    deployed = resolved["plugin"]["document"]["ingress"]
    assert "form_schema" not in deployed
    assert deployed["form_schema_inline"] == resolved["request_schema"]
    assert "json_schema_inline" not in deployed
    assert resolved["manifest_version"] == "0.1.0"
    assert resolved["input_files"] == {
        "features": {
            "required": True,
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 1048576,
        }
    }


def test_native_json_and_multipart_must_share_canonical_parameters_schema(tmp_path):
    import native_workflow

    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    ingress = manifest["ingress"]
    ingress.pop("content_type")
    ingress.update(
        content_types=["application/json", "multipart/form-data"],
        default_content_type="application/json",
        form_schema_inline={
            "type": "object",
            "properties": {"other": {"type": "number"}},
        },
    )
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="canonical parameters schema"):
        native_workflow.resolve_release(path)


@pytest.mark.parametrize(
    "change", ["default_content_type", "default_operation", "allowed_operations"]
)
def test_public_contract_digest_binds_normalized_ingress_defaults_and_operations(
    tmp_path, change
):
    import native_workflow

    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    ingress = manifest["ingress"]
    ingress.pop("content_type")
    ingress.update(
        content_types=["application/json", "multipart/form-data"],
        default_content_type="application/json",
        form_schema=ingress["json_schema"],
        operations={"default": "run", "allowed": ["run", "inspect"]},
    )
    manifest_path.write_text(yaml.safe_dump(manifest))
    before = native_workflow.resolve_release(path)
    if change == "default_content_type":
        ingress["default_content_type"] = "multipart/form-data"
    elif change == "default_operation":
        ingress["operations"]["default"] = "inspect"
    else:
        ingress["operations"]["allowed"].append("validate")
    manifest_path.write_text(yaml.safe_dump(manifest))
    after = native_workflow.resolve_release(path)
    assert after["public_contract_digest"] != before["public_contract_digest"], change
    assert after["model_aliases"] == before["model_aliases"]


def test_public_contract_digest_treats_legacy_operation_alias_as_normalized_operations(
    tmp_path,
):
    import native_workflow

    path, _ = scaffold(tmp_path)
    before = native_workflow.resolve_release(path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    ingress = manifest["ingress"]
    ingress["operations"] = ingress.pop("operation")
    manifest_path.write_text(yaml.safe_dump(manifest))
    after = native_workflow.resolve_release(path)
    assert after["public_contract_digest"] == before["public_contract_digest"]


def test_model_only_resolution_does_not_require_http_schemas_or_workflow_fixtures(
    tmp_path,
):
    import native_workflow

    path, _ = scaffold(tmp_path, "cpp")
    shutil.rmtree(path / "schemas")
    shutil.rmtree(path / "fixtures")
    shutil.rmtree(path / "native")
    resolved = native_workflow.resolve_release(path, model_only=True)
    assert resolved["backend"] == "aoti"
    assert set(resolved["model_aliases"]) == {"primary"}
    assert (
        resolved["model_aliases"]["primary"]["project_document"]["checkpoint"]
        == "checkpoint.pt"
    )
    assert resolved["model_only"] is True


def test_startup_workflow_cases_are_selected_independent_qualification_cases(tmp_path):
    import native_workflow

    path, _ = scaffold(tmp_path, "cpp")
    release_path = path / "release.yaml"
    release = yaml.safe_load(release_path.read_text())
    release["startup"] = {"workflow_cases": ["alternate"]}
    release_path.write_text(yaml.safe_dump(release))
    resolved = native_workflow.resolve_release(path)
    assert [case["name"] for case in resolved["startup_cases"]] == ["alternate"]
    assert resolved["startup_cases"][0]["expected"] == {"values": [6, -15, 1.25, 27]}
    release["startup"]["workflow_cases"] = ["unknown"]
    release_path.write_text(yaml.safe_dump(release))
    with pytest.raises(ValueError, match="startup.workflow_cases"):
        native_workflow.resolve_release(path)


def test_multipart_fixture_bytes_are_hashed_and_shared_with_startup_subset(tmp_path):
    import hashlib
    import native_workflow

    path, _ = scaffold(tmp_path, "cpp")
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["ingress"]["content_type"] = "multipart/form-data"
    manifest["ingress"]["form_schema"] = manifest["ingress"].pop("json_schema")
    manifest["ingress"]["files"] = [
        {
            "name": "features",
            "required": True,
            "media_types": ["application/octet-stream"],
            "max_size_mb": 1,
        }
    ]
    manifest_path.write_text(yaml.safe_dump(manifest))
    data = b"independent fixture input"
    (path / "fixtures/features.bin").write_bytes(data)
    cases_path = path / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    for case in cases["cases"]:
        case["files"] = {
            "features": {
                "source": "fixtures/features.bin",
                "media_type": "application/octet-stream",
            }
        }
    cases_path.write_text(json.dumps(cases))
    resolved = native_workflow.resolve_release(path)
    fixture = resolved["qualification"]["cases_document"]["cases"][0]["files"][
        "features"
    ]
    assert fixture["sha256"] == hashlib.sha256(data).hexdigest()
    assert fixture["size_bytes"] == len(data)
    startup = resolved["startup_cases"][0]
    assert "files" not in startup
    assert startup["input_artifacts"]["features"] == {
        "path": str((path / "fixtures/features.bin").resolve()),
        "media_type": "application/octet-stream",
        "sha256": fixture["sha256"],
        "size_bytes": len(data),
    }
    assert any(
        item["path"] == "fixtures/features.bin"
        for item in resolved["qualification"]["fixtures"]
    )
    cases["cases"][0]["files"]["features"]["source"] = "../outside.bin"
    cases_path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="inside"):
        native_workflow.resolve_release(path)


def test_release_profile_rejects_ambiguous_duplicate_keys(tmp_path):
    path, _ = scaffold(tmp_path)
    profile = path / "release.yaml"
    profile.write_text(profile.read_text() + "\nbackend: aoti\n")
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "duplicate" in checked.stderr and "backend" in checked.stderr


def test_native_run_local_plan_uses_real_packaged_worker_without_python_hooks(tmp_path):
    path, _ = scaffold(tmp_path)
    image = "registry.example/team/native@sha256:" + "a" * 64
    workspace = tmp_path / "session"
    plan = cli(
        "run-local", path, "--image", image, "--workspace", workspace, "--dry-run"
    )
    assert plan.returncode == 0, plan.stderr
    plan = json.loads(plan.stdout)
    assert plan["command"][:3] == ["docker", "run", "--rm"]
    assert plan["command"][-1] == image
    assert "--gpus" in plan["command"]
    assert "REDIS_STREAM_PREFIX=physicsnemo:" in plan["command"]
    assert "RELEASE_IMAGE_DIGEST=" + image in plan["command"]
    assert plan["execution"] == "not_run"
    assert "inference_worker.py" not in json.dumps(plan)
    assert (workspace / "service-plan.json").is_file()


def test_release_cli_leaves_workspace_creation_to_release_transaction(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import native_workflow

    path, _ = scaffold(tmp_path)
    configure_registry(path)
    workspace = tmp_path / "new-release"

    def release(resolved, *, cache_dir, workspace, promote):
        assert not workspace.exists(), "CLI consumed the fresh release workspace"
        assert resolved["workflow"] == "my-native-model"
        return {"status": "dispatch_contract_checked"}

    monkeypatch.setattr(
        native_workflow, "release_module", lambda: SimpleNamespace(release=release)
    )
    report = native_workflow.run_command(
        SimpleNamespace(
            command="release",
            workspace=str(workspace),
            cache_dir=str(tmp_path / "cache"),
        ),
        path,
    )
    assert report["status"] == "dispatch_contract_checked"


def test_runtime_paths_use_linux_container_semantics_on_windows_hosts(
    tmp_path, monkeypatch
):
    from pathlib import PureWindowsPath
    import native_workflow

    path, _ = scaffold(tmp_path)
    container_paths = {
        "/outputs/native-workflows",
        "/outputs/.cache/native-inputs",
    }

    def host_path(value):
        if isinstance(value, str) and value in container_paths:
            return PureWindowsPath(value)
        return Path(value)

    monkeypatch.setattr(native_workflow, "Path", host_path)
    resolved = native_workflow.resolve_release(path)
    assert resolved["runtime"]["artifact_root"] == "/outputs/native-workflows"
    assert resolved["runtime"]["input_cache"] == "/outputs/.cache/native-inputs"


@pytest.mark.parametrize("field", ["artifact_root", "input_cache"])
@pytest.mark.parametrize(
    "value",
    [
        "outputs/native",
        "C:/outputs/native",
        r"C:\outputs\native",
        r"\\server\share\native",
        "//server/share/native",
        r"/outputs\native",
    ],
)
def test_runtime_paths_reject_host_specific_or_relative_paths(tmp_path, field, value):
    import native_workflow

    path, _ = scaffold(tmp_path)
    profile = path / "release.yaml"
    document = yaml.safe_load(profile.read_text())
    document["runtime"] = {field: value}
    profile.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=rf"runtime\.{field}.*absolute"):
        native_workflow.resolve_release(path)


def test_profile_preserves_explicit_lepton_execution_location(tmp_path):
    import native_workflow

    path, _ = scaffold(tmp_path)
    profile = path / "release.yaml"
    document = yaml.safe_load(profile.read_text())
    location = {
        "workspace_id": "workspace",
        "node_group": "qualified-h100",
        "pull_secret": "registry-secret-name",
        "nfs_mount_base": "/project/qa",
    }
    document["lepton"] = location
    profile.write_text(yaml.safe_dump(document))
    assert native_workflow.resolve_release(path)["lepton"] == location


def test_raw_output_requires_independent_artifact_acceptance_identity(tmp_path):
    path, _ = scaffold(tmp_path)
    manifest_path = path / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["configuration"]["native"]["tensor_adapter"]["outputs"]["output"][
        "codec"
    ] = "raw-fp32-le"
    manifest_path.write_text(yaml.safe_dump(manifest))
    schema_path = path / "schemas/result.json"
    schema = json.loads(schema_path.read_text())
    schema["properties"]["values"] = {"type": "object"}
    schema_path.write_text(json.dumps(schema))
    cases_path = path / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    for case in cases["cases"]:
        case["expected"]["values"] = {
            "artifact": "output.f32",
            "dtype": "float32",
            "shape": [4],
            "size_bytes": 16,
            "byte_order": "little",
            "layout": "contiguous",
        }
    cases_path.write_text(json.dumps(cases))
    checked = cli("validate", path)
    assert checked.returncode != 0
    assert "expected_artifacts" in checked.stderr


@pytest.mark.parametrize("name", ["x" * 125, "nested/name", "tensor.name"])
def test_raw_output_names_fit_the_registered_artifact_contract(name):
    import native_workflow

    mapping = {
        "inputs": {"input": {"codec": "json-array-fp32", "request_field": "values"}},
        "outputs": {name: {"codec": "raw-fp32-le", "response_field": "values"}},
    }
    projects = {
        "primary": {
            "project_document": {"input_names": ["input"], "output_names": [name]}
        }
    }
    with pytest.raises(ValueError, match="raw output tensor name"):
        native_workflow._mapping(
            mapping,
            projects,
            {"properties": {"values": {"type": "array"}}},
            {"properties": {"values": {"type": "object"}}},
        )


@pytest.mark.parametrize("direction", ["inputs", "outputs"])
@pytest.mark.parametrize("field", ["nested.values", "two words", "x" * 129])
def test_mapped_http_fields_match_the_native_host_contract(direction, field):
    import native_workflow

    mapping = {
        "inputs": {"input": {"codec": "json-array-fp32", "request_field": "values"}},
        "outputs": {"output": {"codec": "json-array-fp32", "response_field": "values"}},
    }
    key = "request_field" if direction == "inputs" else "response_field"
    mapping[direction]["input" if direction == "inputs" else "output"][key] = field
    projects = {
        "primary": {
            "project_document": {"input_names": ["input"], "output_names": ["output"]}
        }
    }
    schema = {"properties": {"values": {"type": "array"}, field: {"type": "array"}}}
    with pytest.raises(ValueError, match="HTTP field must be a safe identifier"):
        native_workflow._mapping(mapping, projects, schema, schema)
