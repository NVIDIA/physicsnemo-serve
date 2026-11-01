"""Artifact identity, immutable handoff and failure behavior at stage boundaries."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import build  # noqa: E402
from pnms_app.common import read_json, write_json  # noqa: E402
from pnms_app.native import run_bundle as run_native_bundle  # noqa: E402


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from pnms_app import model, native

    app = tmp_path / "app"
    app.mkdir()
    (app / "pnms.yaml").write_text("app")
    (app / "model").mkdir()
    write_json(app / "model/model-build.json", {"format_version": 2})
    write_json(
        app / "fixtures/self-test.json",
        {
            "primary": {
                "inputs": {"input": [1, 2]},
                "expected_outputs": {"output": [3, 5]},
                "rtol": 0,
                "atol": 0,
            }
        },
    )
    write_json(
        app / "fixtures/cases.json",
        {
            "schema_version": 1,
            "cases": [
                {
                    "name": "example",
                    "request": {"values": [1, 2]},
                    "expected": {"values": [3, 5]},
                }
            ],
        },
    )
    project = {
        "root": app,
        "name": "app",
        "runtime": "cpp",
        "backend": "aoti",
        "models": {"primary": {"project": app / "model/model-build.json"}},
        "workflow": {"template": "tensor"},
        "build": {"environment": "local"},
        "document": {"runtime": "cpp"},
    }

    def models(project, output, **kwargs):
        root = output / "primary"
        package = root / "model/backends/aoti"
        write_json(
            package / "model.json",
            {
                "format_version": 1,
                "model": {"name": "example", "version": "1"},
                "inputs": [{"name": "input", "dtype": "float32", "shape": [2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [2]}],
                "artifacts": [{"backend": "aoti", "path": "model.pt2"}],
            },
        )
        (package / "model.pt2").write_bytes(b"compiled model exact bytes")
        write_json(root / "build.json", {"status": "complete"})
        return {
            "models": {
                "primary": {
                    "output_root": str(root),
                    "package": str(package),
                    "backend": "aoti",
                    "compilation_source_sha": "a" * 40,
                }
            },
            "environment": {
                "kind": "local",
                "python": sys.executable,
                "sdk_prefix": str(tmp_path / "sdk"),
                "stack": {"torch": "test"},
                "gpu": {
                    "name": "NVIDIA H100 80GB HBM3",
                    "major": 9,
                    "minor": 0,
                    "architecture": "sm90",
                    "driver_version": "570.1",
                },
            },
        }

    def workflow(project, output, **kwargs):
        installed = output / "install"
        (installed / "bin").mkdir(parents=True)
        (installed / "bin/pnms-native-run").write_text("native runner")
        (installed / "lib").mkdir()
        (installed / "lib/libphysicsnemo_workflow.so").write_bytes(b"shared workflow")
        return {
            "install_root": str(installed),
            "runner": str(installed / "bin/pnms-native-run"),
            "library": str(installed / "lib/libphysicsnemo_workflow.so"),
            "loader_environment": {"LD_LIBRARY_PATH": "/selected/lib"},
        }

    monkeypatch.setattr(model, "build_models", models)
    monkeypatch.setattr(native, "build_workflow", workflow)
    monkeypatch.setattr(
        native,
        "run_bundle",
        lambda *args, **kwargs: {
            "result": {"payload": {"values": [3, 5]}},
            "startup_path": "startup.json",
        },
    )
    return project


def test_complete_build_exposes_models_and_seals_relocatable_bundle(prepared):
    result = build.build_application(prepared, repository=ROOT)
    assert result["status"] == "complete"
    root = Path(result["root"])
    package = root / result["models"]["primary"]["package"]
    bundle = root / result["bundle"]
    assert (package / "model.pt2").read_bytes() == (
        bundle / "models/primary/backends/aoti/model.pt2"
    ).read_bytes()
    config = read_json(bundle / "configuration/workflow.json")
    assert config["models"]["primary"]["package_path"] == "models/primary/backends/aoti"
    assert config["models"]["primary"]["self_test"]["expected_outputs"] == {
        "output": [3, 5]
    }
    receipt = build.last_build(prepared)
    assert receipt["build_id"] == result["build_id"]
    assert result["checks"]["http"] == "not_run"
    assert "image" not in result


@pytest.mark.parametrize("entrypoint", ["build", "assemble"])
@pytest.mark.parametrize(
    "alias",
    [
        "absolute",
        "../../victim.dat",
        r"..\victim.dat",
        r"\victim.dat",
        r"C:\victim.dat",
        r"C:victim.dat",
        r"\\server\share\victim.dat",
    ],
)
def test_unsafe_asset_alias_cannot_overwrite_outside_bundle(
    prepared, tmp_path, entrypoint, alias
):
    from pnms_app import bundle

    model_lock = prepared["root"] / "model/model-build.lock.json"
    model_lock.write_bytes(b"original model lock")
    previous = build.build_application(prepared, repository=ROOT)
    protected = {
        path: path.read_bytes()
        for path in (
            prepared["root"] / ".pnms/last-build.json",
            prepared["root"] / "pnms.lock",
            model_lock,
        )
    }
    victim = tmp_path / "victim.dat"
    victim.write_bytes(b"original outside bytes")
    (prepared["root"] / "asset.dat").write_bytes(b"replacement asset bytes")
    prepared["assets"] = {str(victim) if alias == "absolute" else alias: "asset.dat"}

    with pytest.raises(ValueError):
        if entrypoint == "build":
            build.build_application(prepared, repository=ROOT)
        else:
            root = Path(previous["root"])
            models = {
                alias: {
                    **info,
                    "output_root": root / info["output_root"],
                    "package": root / info["package"],
                }
                for alias, info in previous["models"].items()
            }
            bundle.assemble(
                prepared,
                models,
                previous["workflow"],
                tmp_path / "untrusted-bundle",
                repository=ROOT,
                environment=previous["environment"],
                source_sha=previous["source_sha"],
            )

    assert victim.read_bytes() == b"original outside bytes"
    assert {path: path.read_bytes() for path in protected} == protected
    assert build.last_build(prepared)["build_id"] == previous["build_id"]


def test_build_preserves_nested_asset_alias(prepared):
    (prepared["root"] / "asset.dat").write_bytes(b"nested asset bytes")
    prepared["assets"] = {"lookup/table.dat": "asset.dat"}

    result = build.build_application(prepared, repository=ROOT)

    location = Path(result["root"]) / result["bundle"]
    configuration = read_json(location / "configuration/workflow.json")
    assert configuration["assets"] == {"lookup/table.dat": "assets/lookup/table.dat"}
    assert (location / "assets/lookup/table.dat").read_bytes() == b"nested asset bytes"


@pytest.mark.parametrize("corrupt", [False, True])
def test_build_checks_actual_golden_artifact_bytes(prepared, monkeypatch, corrupt):
    from pnms_app import native
    from pnms_app.common import file_identity

    golden = prepared["root"] / "fixtures/answer.bin"
    golden.write_bytes(b"independent golden output")
    cases_path = prepared["root"] / "fixtures/cases.json"
    cases = read_json(cases_path)
    cases["cases"][0]["expected_artifacts"] = {"answer.bin": file_identity(golden)}
    write_json(cases_path, cases)

    def run(bundle, request, output, **kwargs):
        output.mkdir(parents=True)
        (output / "answer.bin").write_bytes(
            b"wrong output" if corrupt else golden.read_bytes()
        )
        return {
            "output_dir": str(output),
            "result": {
                "payload": {"values": [3, 5]},
                "artifacts": [{"name": "answer.bin", "path": "answer.bin"}],
            },
        }

    monkeypatch.setattr(native, "run_bundle", run)
    if corrupt:
        with pytest.raises(ValueError, match="artifact.*(identity|golden|differs)"):
            build.build_application(prepared, repository=ROOT)
        assert not (prepared["root"] / ".pnms/last-build.json").exists()
    else:
        assert (
            build.build_application(prepared, repository=ROOT)["checks"]["application"]
            == "passed"
        )


def test_modified_bundle_is_rejected_before_run(prepared):
    result = build.build_application(prepared, repository=ROOT)
    assert result["status"] == "complete"
    model = (
        Path(result["root"])
        / result["bundle"]
        / "models/primary/backends/aoti/model.pt2"
    )
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inventory|digest|changed"):
        build.last_build(prepared)


def test_source_snapshot_preserves_source_exclusions_and_filters_dotenv_case_variants(
    tmp_path,
):
    from pnms_app.common import inventory, snapshot

    source = tmp_path / "source"
    excluded = [".env", ".env.local", "nested/.ENV", "nested/.Env.local", ".envrc"]
    for relative in [*excluded, "nested/.settings.json"]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake regression data\n")

    snapshot(source, tmp_path / "captured")

    assert [item["path"] for item in inventory(tmp_path / "captured")] == [
        "nested/.settings.json"
    ]
    assert all((source / relative).is_file() for relative in excluded)
    assert set(excluded) <= {item["path"] for item in inventory(source)}


@pytest.mark.parametrize("operation", ["last_build", "package_build"])
@pytest.mark.parametrize("sealed", [False, True])
def test_python_build_handoff_rejects_dotenv_without_mutating_snapshot(
    tmp_path, monkeypatch, operation, sealed
):
    from pnms_app import images
    from pnms_app.common import inventory

    project = {
        "root": tmp_path,
        "name": "app",
        "runtime": "python",
        "serve": {"image": "runtime@sha256:" + "d" * 64},
    }
    root = tmp_path / ".pnms/builds/old-python-build"
    plugin = root / "python/plugin"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("metadata:\n  id: app\n")
    files = inventory(plugin)
    private = plugin / "assets/.env.production"
    private.parent.mkdir()
    private.write_text("PNMS_FAKE_CREDENTIAL=old-receipt-regression-sentinel\n")
    result = {
        "build_id": root.name,
        "status": "complete",
        "runtime": "python",
        "name": "app",
        "source_sha": "a" * 40,
        "python": {
            "plugin_root": "python/plugin",
            "files": inventory(plugin) if sealed else files,
        },
    }
    receipt = root / "build.json"
    write_json(receipt, result)
    build.publish(project, root)
    original_receipt = receipt.read_bytes()
    original = inventory(plugin)
    monkeypatch.setattr(images, "source_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        images.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Docker ran for an unsafe snapshot"),
    )

    with pytest.raises(ValueError, match=r"\.env|inventory differs"):
        if operation == "last_build":
            build.last_build(project)
        else:
            build.package_build(
                receipt,
                project=project,
                repository=ROOT,
                image="registry/app:0.1",
                push=True,
            )

    assert receipt.read_bytes() == original_receipt
    assert inventory(plugin) == original
    assert not (tmp_path / ".pnms/last-image.json").exists()


def test_bundle_and_deployment_preserve_per_alias_compilation_sources(
    prepared, monkeypatch, tmp_path
):
    from pnms_app import bundle, model

    monkeypatch.setenv("PNMS_SOURCE_SHA", "b" * 40)
    original = model.build_models

    def cached_models(*args, **kwargs):
        result = original(*args, **kwargs)
        result["models"]["primary"]["reused"] = True
        result["models"]["secondary"] = {
            **result["models"]["primary"],
            "compilation_source_sha": "c" * 40,
        }
        return result

    self_test = prepared["root"] / "fixtures/self-test.json"
    fixtures = read_json(self_test)
    fixtures["secondary"] = fixtures["primary"]
    write_json(self_test, fixtures)
    monkeypatch.setattr(model, "build_models", cached_models)
    result = build.build_application(prepared, repository=ROOT)
    location = Path(result["root"]) / result["bundle"]
    expected = {"primary": "a" * 40, "secondary": "c" * 40}
    assert result["source_sha"] == "b" * 40
    assert {
        alias: info.get("compilation_source_sha")
        for alias, info in result["models"].items()
    } == expected
    descriptor = read_json(location / "bundle.json")
    assert descriptor["source_sha"] == "b" * 40
    assert descriptor.get("model_build_sources") == expected
    deployed = bundle.materialize_deployment(
        location,
        tmp_path / "deployment",
        artifact_root=tmp_path / "artifacts",
        input_cache=tmp_path / "inputs",
    )
    assert deployed["model_build_sources"] == expected


def test_unproven_model_origin_cannot_enter_a_new_bundle(prepared, monkeypatch):
    from pnms_app import model

    original = model.build_models

    def unproven(*args, **kwargs):
        result = original(*args, **kwargs)
        result["models"]["primary"].pop("compilation_source_sha")
        return result

    monkeypatch.setattr(model, "build_models", unproven)
    with pytest.raises(ValueError, match="compilation source|rebuild"):
        build.build_application(prepared, repository=ROOT)


def test_legacy_bundle_without_model_origins_requires_rebuild(prepared, tmp_path):
    from pnms_app import bundle
    from pnms_app.common import digest, inventory

    result = build.build_application(prepared, repository=ROOT)
    location = Path(result["root"]) / result["bundle"]
    descriptor = read_json(location / "bundle.json")
    descriptor.pop("model_build_sources", None)
    write_json(location / "bundle.json", descriptor)
    files = inventory(location, exclude=("inventory.json",))
    write_json(
        location / "inventory.json",
        {"schema_version": 1, "files": files, "bundle_digest": digest(files)},
    )
    with pytest.raises(ValueError, match="compilation source|rebuild"):
        bundle.materialize_deployment(
            location,
            tmp_path / "deployment",
            artifact_root=tmp_path / "artifacts",
            input_cache=tmp_path / "inputs",
        )


def test_failed_workflow_retains_models_and_prior_success_pointer(
    prepared, monkeypatch
):
    from pnms_app import native

    previous = build.build_application(prepared, repository=ROOT)
    assert previous["status"] == "complete"

    def fails(*args, **kwargs):
        raise RuntimeError("compile custom failed")

    monkeypatch.setattr(native, "build_workflow", fails)
    with pytest.raises(RuntimeError, match="compile custom failed"):
        build.build_application(prepared, repository=ROOT)
    assert build.last_build(prepared)["build_id"] == previous["build_id"]
    failed = [
        read_json(path)
        for path in (prepared["root"] / ".pnms/builds").glob("*/build.json")
        if read_json(path)["status"] == "failed"
    ]
    assert len(failed) == 1
    assert failed[0]["models"]["primary"]["package"]
    assert failed[0]["checks"]["application"] == "not_run"


def test_independent_fixture_mismatch_prevents_success(prepared, monkeypatch):
    from pnms_app import native

    monkeypatch.setattr(
        native,
        "run_bundle",
        lambda *a, **kw: {"result": {"payload": {"values": [3, 999]}}},
    )
    with pytest.raises(ValueError, match="fixture|expected|parity"):
        build.build_application(prepared, repository=ROOT)
    assert not (prepared["root"] / ".pnms/last-build.json").exists()


def test_build_freezes_application_inputs_before_model_compilation(
    prepared, monkeypatch
):
    from pnms_app import model

    original = model.build_models

    def change_inputs(*args, **kwargs):
        write_json(
            prepared["root"] / "fixtures/cases.json",
            {
                "schema_version": 1,
                "cases": [
                    {
                        "name": "changed",
                        "request": {"values": [1, 2]},
                        "expected": {"values": [999, 999]},
                    }
                ],
            },
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "build_models", change_inputs)
    result = build.build_application(prepared, repository=ROOT)
    assert result["status"] == "complete"
    cases = read_json(Path(result["root"]) / "bundle/fixtures/cases.json")
    assert cases["cases"][0]["name"] == "example"


def test_separate_packaging_selects_image_without_rewriting_build(
    prepared, monkeypatch
):
    from pnms_app import images

    result = build.build_application(prepared, repository=ROOT)
    assert result["status"] == "complete"
    path = Path(result["root"]) / "build.json"
    original = path.read_bytes()
    monkeypatch.setattr(
        images,
        "build_image",
        lambda *a, **kw: {"status": "complete", "image_id": "sha256:" + "c" * 64},
    )
    build.package_build(path, project=prepared, repository=ROOT, image="app:1")
    selected = build.last_build(prepared, image=True)
    assert selected["image"]["image_id"] == "sha256:" + "c" * 64
    assert path.read_bytes() == original


def test_unpublished_container_stage_does_not_replace_prior_success(prepared):
    previous = build.build_application(prepared, repository=ROOT)
    staged = build.build_application(prepared, repository=ROOT, publish_result=False)
    assert staged["status"] == "complete"
    assert staged["build_id"] != previous["build_id"]
    assert build.last_build(prepared)["build_id"] == previous["build_id"]


def _raw_project(project):
    import struct

    project["ingress"] = {
        "content_type": "multipart/form-data",
        "files": [
            {
                "name": "features",
                "required": True,
                "media_types": ["application/octet-stream"],
                "max_size_mb": 1,
            }
        ],
    }
    project["workflow"]["tensor_adapter"] = {
        "model_alias": "primary",
        "inputs": {"input": {"request_field": "values", "codec": "raw-fp32-le"}},
        "outputs": {"output": {"response_field": "values", "codec": "json-array-fp32"}},
    }
    source = project["root"] / "fixtures/features.f32"
    source.write_bytes(struct.pack("<2f", 1, 2))
    cases = read_json(project["root"] / "fixtures/cases.json")
    cases["cases"][0]["request"] = {
        "values": {
            "artifact": "features",
            "dtype": "float32",
            "shape": [2],
            "byte_order": "little",
            "layout": "contiguous",
            "size_bytes": 8,
        }
    }
    cases["cases"][0]["inputs"] = {"features": "fixtures/features.f32"}
    write_json(project["root"] / "fixtures/cases.json", cases)
    return source


@pytest.mark.parametrize("fixture_style", ["inputs", "files"])
def test_raw_fixture_inputs_are_sealed_passed_and_projected_to_http(
    prepared, monkeypatch, tmp_path, fixture_style
):
    from pnms_app import bundle, native
    from pnms_app.common import legacy_module
    import shutil

    source = _raw_project(prepared)
    expected = source.read_bytes()
    if fixture_style == "files":
        cases = read_json(prepared["root"] / "fixtures/cases.json")
        case = cases["cases"][0]
        case.pop("inputs")
        case["files"] = {
            "features": {
                "source": "fixtures/features.f32",
                "media_type": "application/octet-stream",
            }
        }
        write_json(prepared["root"] / "fixtures/cases.json", cases)
    calls = []

    def run(bundle_path, request, output, **kwargs):
        inputs = kwargs.get("inputs", {})
        assert set(inputs) == {"features"}, "fixture files must reach the native runner"
        path = Path(inputs["features"])
        assert path.is_relative_to(bundle_path)
        assert path.read_bytes() == expected
        calls.append(inputs)
        return {"result": {"payload": {"values": [3, 5]}}}

    monkeypatch.setattr(native, "run_bundle", run)
    result = build.build_application(prepared, repository=ROOT)
    original = Path(result["root"]) / "bundle"
    moved = tmp_path / "moved-bundle"
    shutil.copytree(original, moved)
    shutil.rmtree(prepared["root"])
    bundle.verify(moved, result["bundle_digest"])
    cases = read_json(moved / "fixtures/cases.json")
    assert (moved / cases["cases"][0]["inputs"]["features"]).read_bytes() == expected
    plugin = moved / "configuration/plugin/app/plugin.yaml"
    manifest = legacy_module(ROOT, "plugin_runtime").load_plugin_manifest(plugin)
    assert manifest["ingress"]["content_types"] == ["multipart/form-data"]
    assert manifest["ingress"]["form_schema"] == "request.schema.json"
    assert manifest["ingress"]["files"] == prepared["ingress"]["files"]
    deployed = bundle.materialize_deployment(
        moved,
        tmp_path / "launch",
        artifact_root=tmp_path / "uploads",
        input_cache=tmp_path / "cache",
    )
    assert deployed["input_files"] == {
        "features": {
            "required": True,
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 1048576,
        }
    }
    assert deployed["ingress_root"] == str(tmp_path / "uploads")
    assert deployed["manifest_version"] == manifest["metadata"]["version"]
    assert len(calls) == 1


@pytest.mark.parametrize("tamper", [False, True])
def test_build_executes_fixture_with_its_explicit_captured_media_type(
    prepared, monkeypatch, tamper
):
    from pnms_app import native

    source = prepared["root"] / "fixtures" / ("upload.txt" if tamper else "upload.data")
    source.write_bytes(b"customer input")
    prepared["ingress"] = {
        "content_type": "multipart/form-data",
        "files": [
            {
                "name": "features",
                "required": True,
                "media_types": ["text/plain"],
                "max_size_mb": 1,
            }
        ],
    }
    cases_path = prepared["root"] / "fixtures/cases.json"
    cases = read_json(cases_path)
    cases["cases"][0]["files"] = {
        "features": {
            "source": source.relative_to(prepared["root"]).as_posix(),
            "media_type": "text/plain",
        }
    }
    write_json(cases_path, cases)
    envelopes = []

    def execute(arguments, **kwargs):
        envelope = read_json(arguments[arguments.index("--request") + 1])
        envelopes.append(envelope)
        write_json(
            arguments[arguments.index("--output") + 1],
            {
                "payload": {
                    "values": [
                        2 * value + 1 for value in envelope["parameters"]["values"]
                    ]
                }
            },
        )

    def run(bundle_path, request, output, **kwargs):
        if tamper:
            Path(kwargs["inputs"]["features"]).write_bytes(b"tampered input")
        return run_native_bundle(bundle_path, request, output, **kwargs)

    monkeypatch.setattr(native, "run_bundle", run)
    monkeypatch.setattr(native, "_execute", execute)
    if tamper:
        with pytest.raises(ValueError, match="captured identity"):
            build.build_application(prepared, repository=ROOT)
        assert not envelopes
        assert not (prepared["root"] / ".pnms/last-build.json").exists()
        return
    result = build.build_application(prepared, repository=ROOT)

    assert result["checks"]["application"] == "passed"
    assert len(envelopes) == 1
    assert envelopes[0]["input_metadata"]["features"]["media_type"] == "text/plain"
    location = Path(result["root"]) / result["bundle"]
    sealed = read_json(location / "fixtures/cases.json")["cases"][0]["files"][
        "features"
    ]
    assert sealed["media_type"] == "text/plain"
    assert envelopes[0]["input_metadata"]["features"]["sha256"] == sealed["sha256"]
    assert (
        envelopes[0]["input_metadata"]["features"]["size_bytes"] == sealed["size_bytes"]
    )


@pytest.mark.parametrize(
    "violation",
    [
        "missing-ingress",
        "missing-file",
        "undeclared-file",
        "escape",
        "media-type",
        "oversized",
    ],
)
def test_raw_fixture_contract_errors_fail_before_native_execution(
    prepared, monkeypatch, violation
):
    from pnms_app import native

    _raw_project(prepared)
    cases = read_json(prepared["root"] / "fixtures/cases.json")
    case = cases["cases"][0]
    if violation == "missing-ingress":
        prepared.pop("ingress")
    elif violation == "missing-file":
        case.pop("inputs")
    elif violation == "undeclared-file":
        case["inputs"]["untrusted"] = "fixtures/features.f32"
    elif violation == "media-type":
        case.pop("inputs")
        case["files"] = {
            "features": {"source": "fixtures/features.f32", "media_type": "image/png"}
        }
    elif violation == "oversized":
        (prepared["root"] / "fixtures/features.f32").write_bytes(
            b"x" * (1024 * 1024 + 1)
        )
    else:
        case["inputs"]["features"] = "../outside.f32"
        (prepared["root"].parent / "outside.f32").write_bytes(b"outside")
    write_json(prepared["root"] / "fixtures/cases.json", cases)
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        return {"result": {"payload": {"values": [3, 5]}}}

    monkeypatch.setattr(native, "run_bundle", execute)
    with pytest.raises(ValueError, match="ingress|fixture|contained|file|input"):
        build.build_application(prepared, repository=ROOT)
    assert not calls


def test_pnms_native_ingress_uses_existing_form_schema_contract(prepared):
    from pnms_app.config import load_project
    import yaml

    _raw_project(prepared)
    custom_schema = {
        "type": "object",
        "required": ["values"],
        "properties": {"values": {"type": "object"}},
        "additionalProperties": False,
    }
    write_json(prepared["root"] / "schemas/raw-request.json", custom_schema)
    prepared["ingress"]["form_schema"] = "schemas/raw-request.json"
    document = {
        "schema_version": 1,
        "name": "app",
        "runtime": "cpp",
        "models": {"primary": {"project": "model/model-build.json"}},
        "ingress": prepared["ingress"],
        "workflow": prepared["workflow"],
    }
    (prepared["root"] / "pnms.yaml").write_text(yaml.safe_dump(document))
    parsed = load_project(prepared["root"], ROOT)
    assert parsed["ingress"] == prepared["ingress"]
    result = build.build_application(parsed, repository=ROOT)
    root = Path(result["root"]) / "bundle"
    assert read_json(root / "configuration/request.schema.json") == custom_schema
    assert (
        read_json(root / "configuration/plugin/app/request.schema.json")
        == custom_schema
    )


def test_native_ingress_rejects_different_json_and_form_schemas(prepared):
    _raw_project(prepared)
    prepared["ingress"].update(
        content_types=["application/json", "multipart/form-data"],
        json_schema_inline={"type": "object", "required": ["one"]},
        form_schema_inline={"type": "object", "required": ["two"]},
    )
    with pytest.raises(ValueError, match="same request schema"):
        build.build_application(prepared, repository=ROOT)


@pytest.mark.parametrize(
    "runner_path", ["bin/pnms-native-run.exe", "libexec/custom-run"]
)
def test_bundle_records_actual_installed_runner_path(
    prepared, monkeypatch, runner_path
):
    from pnms_app import native

    original = native.build_workflow

    def workflow(*args, **kwargs):
        installed = original(*args, **kwargs)
        runner = Path(installed["install_root"]) / runner_path
        runner.parent.mkdir(parents=True, exist_ok=True)
        Path(installed["runner"]).rename(runner)
        return {**installed, "runner": str(runner)}

    monkeypatch.setattr(native, "build_workflow", workflow)
    result = build.build_application(prepared, repository=ROOT)
    location = Path(result["root"]) / "bundle"
    relative = f"workflow/{runner_path}"
    assert read_json(location / "bundle.json")["workflow"]["runner"] == relative
    assert (location / relative).read_text() == "native runner"


def test_bundle_rejects_runner_metadata_outside_install(prepared, monkeypatch):
    from pnms_app import native

    original = native.build_workflow
    external = prepared["root"] / "external-runner"
    external.write_text("external executable")

    def workflow(*args, **kwargs):
        return {**original(*args, **kwargs), "runner": str(external)}

    monkeypatch.setattr(native, "build_workflow", workflow)
    with pytest.raises(ValueError, match="is not in the subpath"):
        build.build_application(prepared, repository=ROOT)
