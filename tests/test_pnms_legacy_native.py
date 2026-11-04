"""Existing native plugins use the common local application path unchanged."""

import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app.common import legacy_module  # noqa: E402
from pnms_app.config import load_project  # noqa: E402
from test_pnms_build import prepared  # noqa: E402,F401


def files(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "backend,template,content_type",
    [
        ("aoti", "tensor", "application/json"),
        ("tensorrt", "cpp", "application/json"),
        ("tensorrt", "tensor", "multipart/form-data"),
    ],
)
def test_legacy_native_project_is_adopted_without_migration(
    tmp_path, backend, template, content_type
):
    root = tmp_path / "existing-native"
    legacy_module(ROOT, "native_workflow").init_native(
        root, backend=backend, template=template, content_type=content_type
    )
    original = files(root)

    project = load_project(root, ROOT)

    assert project["runtime"] == "cpp"
    assert project["name"] == "existing-native"
    assert project["backend"] == backend
    assert project["build"]["environment"] == "local"
    assert project["models"]["primary"]["project"] == root / "model/model-build.json"
    assert project["workflow"]["template"] == (
        "tensor" if template == "tensor" else "custom"
    )
    assert project["ingress"]["default_content_type"] == content_type
    assert project["fixtures"] == {
        "cases": "fixtures/cases.json",
        "self_test": "fixtures/self-test.json",
    }
    if template == "cpp":
        assert project["workflow"]["source_dir"] == "native"
        assert project["workflow"]["target"] == "my_native_workflow"
    assert files(root) == original
    assert not (root / "pnms.yaml").exists()
    assert "lepton" not in project["document"]


@pytest.mark.parametrize(
    "alias", ["absolute", "../victim.dat", r"..\victim.dat", r"C:\victim.dat"]
)
def test_legacy_unsafe_asset_alias_is_rejected_before_adoption(tmp_path, alias):
    import yaml

    root = tmp_path / "existing-native"
    legacy_module(ROOT, "native_workflow").init_native(root)
    victim = tmp_path / "victim.dat"
    victim.write_bytes(b"original outside bytes")
    (root / "asset.dat").write_bytes(b"replacement asset bytes")
    path = root / "release.yaml"
    release = yaml.safe_load(path.read_text())
    release["assets"] = [
        {
            "name": str(victim) if alias == "absolute" else alias,
            "source": "asset.dat",
            "destination": "assets/asset.dat",
        }
    ]
    path.write_text(yaml.safe_dump(release, sort_keys=False))
    original = files(root)

    with pytest.raises(ValueError, match="asset alias"):
        load_project(root, ROOT)

    assert victim.read_bytes() == b"original outside bytes"
    assert files(root) == original
    assert not (root / ".pnms").exists()


def test_legacy_build_preserves_the_authored_public_contract(prepared, monkeypatch):  # noqa: F811
    import json
    import yaml
    from pnms_app import build, native

    root = prepared["root"]
    helper = legacy_module(ROOT, "native_workflow")
    helper.init_native(root, force=True)
    manifest_path = root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["metadata"]["version"] = "2.3.0"
    manifest["metadata"]["display_name"] = "Original customer workflow"
    manifest["runtime"]["executor_class"] = "native.original"
    manifest["pipeline"]["stages"][1]["queue"] = "execute.native.original"
    manifest["resources"]["defaults"]["memory_mb"] = 4096
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = files(root)
    project = load_project(root, ROOT)

    def run_bundle(_bundle, request, _output, **kwargs):
        values = json.loads(Path(request).read_text())["values"]
        return {"result": {"payload": {"values": [2 * value + 1 for value in values]}}}

    monkeypatch.setattr(native, "run_bundle", run_bundle)
    built = build.build_application(project, repository=ROOT)

    projection = Path(built["root"]) / "bundle/configuration/plugin/app/plugin.yaml"
    materialized = json.loads(projection.read_text())
    assert materialized["metadata"] == manifest["metadata"]
    assert materialized["runtime"]["executor_class"] == "native.original"
    assert materialized["pipeline"]["stages"] == manifest["pipeline"]["stages"]
    assert materialized["resources"] == manifest["resources"]
    assert all(files(root)[name] == digest for name, digest in original.items())
    assert not (root / "pnms.yaml").exists()


@pytest.mark.parametrize("schedule_queue", ["schedule", "schedule.customer"])
def test_legacy_worker_inputs_follow_the_preserved_pipeline(
    prepared,  # noqa: F811
    monkeypatch,
    schedule_queue,
):
    import yaml

    from pnms_app import build, native
    from pnms_app.common import read_json

    root = prepared["root"]
    legacy_module(ROOT, "native_workflow").init_native(root, force=True)
    path = root / "plugin.yaml"
    manifest = yaml.safe_load(path.read_text())
    stages = manifest["pipeline"]["stages"]
    stages[0]["queue"] = schedule_queue
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    def run_bundle(_bundle, request, _output, **kwargs):
        values = read_json(request)["values"]
        return {"result": {"payload": {"values": [2 * value + 1 for value in values]}}}

    monkeypatch.setattr(native, "run_bundle", run_bundle)
    built = build.build_application(load_project(root, ROOT), repository=ROOT)
    location = Path(built["root"]) / built["bundle"]
    worker = read_json(location / "configuration/worker_runtime_config.json")

    assert [item["stream"] for item in worker["roles"]["scheduler"]["inputs"]] == [
        schedule_queue,
        "release",
    ]
    assert [item["stream"] for item in worker["roles"]["results"]["inputs"]] == [
        "results"
    ]
    assert worker["streams"] == [schedule_queue, "release", "results", "dlq"]
    assert worker["roles"]["scheduler"]["config"]["max_batch_size"] == 1
    sealed = read_json(location / "configuration/plugin/app/plugin.yaml")
    assert sealed["pipeline"]["stages"] == stages


@pytest.mark.parametrize(
    "stage_index,field,value,diagnostic",
    [
        (2, "queue", "results.customer", "native worker requires.*results.*queue"),
        (2, "id", "customer_results", "native worker requires.*results.*stage"),
        (2, "next", "schedule", "native worker requires.*terminal.*results"),
        (0, "queue", "release", "schedule queue.*reserved.*release"),
        (0, "queue", "results", "schedule queue.*reserved.*results"),
        (0, "queue", "dlq", "schedule queue.*reserved.*dlq"),
        (0, "queue", "schedule:grp", "schedule queue.*:grp"),
    ],
)
def test_legacy_unsupported_worker_pipeline_is_rejected_before_compilation(
    prepared,  # noqa: F811
    monkeypatch,
    stage_index,
    field,
    value,
    diagnostic,
):
    import yaml

    from pnms_app import build, model, native
    from pnms_app.common import read_json

    root = prepared["root"]
    legacy_module(ROOT, "native_workflow").init_native(root, force=True)

    def run_bundle(_bundle, request, _output, **kwargs):
        values = read_json(request)["values"]
        return {"result": {"payload": {"values": [2 * value + 1 for value in values]}}}

    monkeypatch.setattr(native, "run_bundle", run_bundle)
    project = load_project(root, ROOT)
    built = build.build_application(project, repository=ROOT)
    previous = (root / ".pnms/last-build.json").read_bytes()
    path = root / "plugin.yaml"
    manifest = yaml.safe_load(path.read_text())
    stages = manifest["pipeline"]["stages"]
    stages[stage_index][field] = value
    if field == "id":
        stages[1]["next"] = value
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = files(root)

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("unsupported native pipeline reached compilation")

    monkeypatch.setattr(model, "build_models", unexpected_compilation)
    monkeypatch.setattr(native, "build_workflow", unexpected_compilation)

    with pytest.raises(ValueError, match=diagnostic):
        build.build_application(load_project(root, ROOT), repository=ROOT)

    assert files(root) == original
    assert (root / ".pnms/last-build.json").read_bytes() == previous
    assert build.last_build(project)["build_id"] == built["build_id"]


def test_legacy_frozen_execution_needs_no_authored_files(prepared):  # noqa: F811
    import shutil
    from pnms_app import build

    built = build.build_application(prepared, repository=ROOT)
    root = prepared["root"]
    for path in root.iterdir():
        if path.name != ".pnms":
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    frozen = load_project(root, ROOT, require_sources=False)

    assert frozen["runtime"] == "cpp" and frozen["name"] == "app"
    assert build.last_build(frozen)["build_id"] == built["build_id"]
    assert build.last_build(frozen)["bundle_digest"] == built["bundle_digest"]


@pytest.mark.parametrize("alias", ["omitted", "primary", "secondary"])
@pytest.mark.parametrize("authored_deadlines", [False, True])
def test_legacy_tensor_alias_and_deadlines_survive_build_and_deployment(
    prepared,  # noqa: F811
    monkeypatch,
    tmp_path,
    alias,
    authored_deadlines,
):
    import copy
    import shutil

    import yaml

    from pnms_app import build, bundle, model, native
    from pnms_app.common import read_json, write_json

    root = prepared["root"]
    helper = legacy_module(ROOT, "native_workflow")
    helper.init_native(root, force=True)
    manifest_path, release_path = root / "plugin.yaml", root / "release.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    release = yaml.safe_load(release_path.read_text())
    mapping = manifest["configuration"]["native"]["tensor_adapter"]
    if alias == "omitted":
        mapping.pop("model_alias")
    elif alias == "secondary":
        mapping["model_alias"] = alias
        mapping["inputs"]["secondary_input"] = mapping["inputs"].pop("input")
        mapping["outputs"]["secondary_output"] = mapping["outputs"].pop("output")
        shutil.copytree(root / "model", root / "secondary")
        selected = read_json(root / "secondary/model-build.json")
        selected.update(
            input_names=["secondary_input"], output_names=["secondary_output"]
        )
        write_json(root / "secondary/model-build.json", selected)
        release["models"] = {
            "primary": {"project": release.pop("model_project")},
            "secondary": {"project": "secondary/model-build.json"},
        }
        fixtures = read_json(root / "fixtures/self-test.json")
        fixtures["secondary"] = copy.deepcopy(fixtures["primary"])
        for category, before, after in (
            ("inputs", "input", "secondary_input"),
            ("expected_outputs", "output", "secondary_output"),
        ):
            fixtures["secondary"][category][after] = fixtures["secondary"][
                category
            ].pop(before)
        write_json(root / "fixtures/self-test.json", fixtures)
    if authored_deadlines:
        release["runtime"] = {
            "request_timeout_seconds": 1800,
            "drain_timeout_seconds": 90,
            "startup_timeout_seconds": 900,
        }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    release_path.write_text(yaml.safe_dump(release, sort_keys=False))
    original = files(root)
    managed = helper.resolve_release(root)
    project = load_project(root, ROOT)

    original_models = model.build_models

    def compiled_models(project, output, **kwargs):
        result = original_models(project, output, **kwargs)
        if alias == "secondary":
            primary = result["models"]["primary"]
            target = output / "secondary"
            shutil.copytree(primary["output_root"], target)
            package = target / "model/backends/aoti"
            selected = read_json(package / "model.json")
            selected["inputs"][0]["name"] = "secondary_input"
            selected["outputs"][0]["name"] = "secondary_output"
            write_json(package / "model.json", selected)
            result["models"]["secondary"] = {
                **primary,
                "output_root": str(target),
                "package": str(package),
            }
        return result

    def run_bundle(_bundle, request, _output, **kwargs):
        values = read_json(request)["values"]
        return {"result": {"payload": {"values": [2 * value + 1 for value in values]}}}

    monkeypatch.setattr(model, "build_models", compiled_models)
    monkeypatch.setattr(native, "run_bundle", run_bundle)
    built = build.build_application(project, repository=ROOT)
    location = Path(built["root"]) / built["bundle"]
    deployment = bundle.materialize_deployment(
        location,
        tmp_path / "deployment",
        artifact_root=tmp_path / "artifacts",
        input_cache=tmp_path / "inputs",
    )

    assert deployment["workflow"]["tensor_adapter"] == managed["adapter_config"]
    for key in (
        "request_timeout_seconds",
        "drain_timeout_seconds",
        "startup_timeout_seconds",
    ):
        assert deployment[key] == managed["runtime"][key]
    assert all(files(root)[name] == digest for name, digest in original.items())


@pytest.mark.parametrize(
    "field",
    ["request_timeout_seconds", "drain_timeout_seconds", "startup_timeout_seconds"],
)
@pytest.mark.parametrize("invalid", [0, -1, 7201, True, 1.5, "900"])
def test_legacy_adoption_validates_authored_deadlines(tmp_path, field, invalid):
    import yaml

    root = tmp_path / "existing-native"
    legacy_module(ROOT, "native_workflow").init_native(root)
    path = root / "release.yaml"
    release = yaml.safe_load(path.read_text())
    release["runtime"] = {field: invalid}
    path.write_text(yaml.safe_dump(release, sort_keys=False))

    with pytest.raises(ValueError, match=rf"runtime\.{field}.*1 through 7200"):
        load_project(root, ROOT)
