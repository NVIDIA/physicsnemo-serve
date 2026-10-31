from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_release_orchestration", ROOT / "native-inference/deploy/native_release.py"
)
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def resolved():
    return {
        "schema_version": 1,
        "workflow": "example",
        "adapter": "tensor.v1",
        "model_aliases": {
            "primary": {
                "project": "/checkout/model/model-build.json",
                "project_document": {
                    "format_version": 2,
                    "name": "affine",
                    "adapter": "adapter.py",
                },
                "project_files": [
                    {"path": "adapter.py", "sha256": "a" * 64, "size_bytes": 4}
                ],
            }
        },
        "backend": "aoti",
        "target": {
            "profile": "h100",
            "compilation": {
                "platform": "linux/amd64",
                "gpu_product": "NVIDIA H100 80GB HBM3",
                "compute_capability": [9, 0],
                "cuda_architecture": "sm90",
                "precision": "fp32",
                "mig": "disabled",
                "hardware_compatibility": "qualified-product-only",
            },
            "deployment": {"minimum_available_memory_mb": 70000},
        },
        "images": {
            "builder": "registry.example/builder@sha256:" + "b" * 64,
            "server": "registry.example/server@sha256:" + "c" * 64,
        },
        "qualification": {
            "policy": "tensor-v1",
            "required_cases": ["http"],
            "fixtures": [],
        },
        "assets": [],
    }


def test_gpu_product_change_invalidates_compilation_even_when_same_sm():
    before = resolved()
    after = copy.deepcopy(before)
    after["target"]["compilation"]["gpu_product"] = "NVIDIA H200"
    assert release.compilation_key(before, "primary") != release.compilation_key(
        after, "primary"
    )


def test_model_key_changes_for_bytes_backend_builder_and_compiler_options():
    before = resolved()
    mutations = [
        lambda d: d["model_aliases"]["primary"]["project_files"][0].update(
            sha256="d" * 64
        ),
        lambda d: d.update(backend="tensorrt"),
        lambda d: d["images"].update(
            builder="registry.example/builder@sha256:" + "e" * 64
        ),
        lambda d: d["model_aliases"]["primary"]["project_document"].update(
            aoti_options={"shape_padding": False}
        ),
    ]
    for mutate in mutations:
        after = copy.deepcopy(before)
        mutate(after)
        assert release.compilation_key(before, "primary") != release.compilation_key(
            after, "primary"
        )


def test_model_key_reuses_for_server_source_capacity_and_acceptance_changes():
    before = resolved()
    after = copy.deepcopy(before)
    after["source_sha"] = "f" * 40
    after["images"]["server"] = "registry.example/server@sha256:" + "d" * 64
    after["target"]["deployment"]["minimum_available_memory_mb"] = 75000
    after["qualification"]["required_cases"].append("new-service-fixture")
    after["model_aliases"]["primary"]["project"] = (
        "/another/checkout/model/model-build.json"
    )
    assert release.compilation_key(before, "primary") == release.compilation_key(
        after, "primary"
    )


def candidate(tmp_path):
    """Synthetic Builder receipts; this fixture never runs a GPU or a model."""
    import json
    import struct

    out = tmp_path / "candidate"
    package = out / "model/backends/aoti"
    package.mkdir(parents=True)
    source = out / "source"
    source.mkdir()
    recipe = {
        "format_version": 1,
        "name": "affine",
        "version": "1",
        "adapter": "adapter.py",
        "factory": "create_model",
        "cases": "create_cases",
        "supported_backends": ["aoti"],
        "default_backend": "aoti",
        "input_names": ["input"],
        "output_names": ["output"],
        "dtype": "float32",
        "shape": [4],
    }
    (source / "recipe.json").write_text(json.dumps(recipe))
    (source / "adapter.py").write_text("# synthetic retained source\n")
    (package / "model.pt2").write_bytes(
        b"synthetic package for byte-verification unit tests"
    )
    (package / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [4]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [4]}],
                "artifacts": [
                    {"backend": "aoti", "target": "cuda", "path": "model.pt2"}
                ],
            }
        )
    )
    checks = out / "checks/aoti/case-0"
    checks.mkdir(parents=True)
    input_bytes = struct.pack("<4f", 0, 1, -1, 4)
    output_bytes = struct.pack("<4f", 1, 3, -1, 9)
    for name, data in [
        ("input-0.bin", input_bytes),
        ("reference-0.bin", output_bytes),
        ("output-0.bin", output_bytes),
    ]:
        (checks / name).write_bytes(data)
    runtime = {"sha256": "1" * 64, "size_bytes": 123}
    output = {
        "name": "output",
        "dtype": "float32",
        "shape": [4],
        "max_abs": 0.0,
        "relative_l2": 0.0,
        "actual_sha256": release.file_identity(checks / "output-0.bin")["sha256"],
        "reference_sha256": release.file_identity(checks / "reference-0.bin")["sha256"],
    }
    check = {
        "format_version": 1,
        "backend": "aoti",
        "passed": True,
        "runtime": runtime,
        "limits": {"max_abs": 1e-4, "relative_l2": 1e-4},
        "cases": [
            {
                "passed": True,
                "metadata": {
                    "schema_version": 1,
                    "completed": True,
                    "backend": "aoti",
                    "execution_device": {"type": "cuda", "index": 0},
                    "outputs": [
                        {
                            "name": "output",
                            "dtype": "float32",
                            "shape": [4],
                            "device": {"type": "cpu", "index": 0},
                            "byte_size": 16,
                        }
                    ],
                },
                "outputs": [output],
                "inputs": [
                    {
                        "name": "input",
                        "dtype": "float32",
                        "shape": [4],
                        "sha256": release.file_identity(checks / "input-0.bin")[
                            "sha256"
                        ],
                    }
                ],
            }
        ],
    }
    (out / "checks/aoti.json").write_text(json.dumps(check))
    graph = out / "exported/aoti"
    graph.mkdir(parents=True)
    (graph / "model.pt2").write_bytes(b"synthetic exported graph")
    files = [
        {**item, "path": "backends/aoti/" + item["path"]}
        for item in release.inventory(package)
    ]
    variant = {
        "status": "complete",
        "package": "backends/aoti",
        "package_base": "model",
        "evidence_base": "build",
        "files": files,
        "graph": {"entrypoint": "model.pt2"},
        "graphs": [
            {
                "path": "exported/aoti/model.pt2",
                **release.file_identity(graph / "model.pt2"),
            }
        ],
        "checks": {
            "path": "checks/aoti.json",
            **release.file_identity(out / "checks/aoti.json"),
        },
    }
    environment = {"gpu_name": "NVIDIA H100 80GB HBM3", "compute_capability": [9, 0]}
    model = {
        "format_version": 1,
        "model": {"name": "affine", "version": "1"},
        "runtime": runtime,
        "environment": environment,
        "variants": {"aoti": {key: variant[key] for key in ("package", "files")}},
    }
    (out / "model/model-release.json").write_text(json.dumps(model))
    build = {
        "format_version": 1,
        "status": "complete",
        "requested_backends": ["aoti"],
        "device": "cuda",
        "case_count": 1,
        "runtime": runtime,
        "environment": environment,
        "source": {
            "files": [
                {
                    "path": "source/" + item["path"],
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for item in release.inventory(source)
            ]
        },
        "variants": {"aoti": variant},
        "release": {
            "path": "model/model-release.json",
            **release.file_identity(out / "model/model-release.json"),
        },
    }
    (out / "build.json").write_text(json.dumps(build))
    (out / "release-provenance.json").write_text(
        json.dumps(
            {
                "source_sha": "f" * 40,
                "compilation_key": release.compilation_key(resolved(), "primary"),
            }
        )
    )
    return out


@pytest.mark.skipif(
    os.name == "nt", reason="Release-controller cache publication uses POSIX file locks"
)
def test_verified_cache_reuse_rechecks_payload_and_first_publisher_wins(tmp_path):
    import shutil

    output = candidate(tmp_path)
    cache = tmp_path / "cache"
    key = release.compilation_key(resolved(), "primary")
    first = release.cache_publish(cache, key, output, backend="aoti")
    assert first["status"] == "hit"
    second = tmp_path / "another-nondeterministic-build"
    shutil.copytree(output, second)
    (second / "optional-build.log").write_text("different compiler diagnostics\n")
    later = release.cache_publish(cache, key, second, backend="aoti")
    assert later["artifact_digest"] == first["artifact_digest"]
    (Path(first["path"]) / "model/backends/aoti/model.pt2").write_bytes(b"tampered")
    hit = release.cache_lookup(cache, key, backend="aoti")
    assert hit["status"] == "invalid"
    recovered = release.cache_publish(cache, key, output, backend="aoti")
    assert recovered["status"] == "hit"
    assert len(list(cache.glob(".invalid-*"))) == 1


def test_handoff_recomputes_native_parity_instead_of_trusting_pass_boolean(tmp_path):
    import struct
    import pytest

    output = candidate(tmp_path)
    (output / "checks/aoti/case-0/output-0.bin").write_bytes(
        struct.pack("<4f", 1, 3, -1, 9999)
    )
    with pytest.raises(ValueError, match="parity|output|metrics"):
        release.verify_completed_build(output, backend="aoti")


def staging_resolved(tmp_path):
    import json

    value = resolved()
    model = tmp_path / "project"
    model.mkdir()
    (model / "adapter.py").write_text("# unit test source\n")
    (model / "model-build.json").write_text(
        json.dumps(value["model_aliases"]["primary"]["project_document"])
    )
    value["model_aliases"]["primary"].update(
        project=str(model / "model-build.json"), project_files=release.inventory(model)
    )
    value.update(
        source_sha="f" * 40,
        self_tests={
            "primary": {
                "inputs": {"input": [0, 1, -1, 4]},
                "expected_outputs": {"output": [1, 3, -1, 9]},
                "rtol": 1e-5,
                "atol": 1e-6,
            }
        },
        request_schema={"type": "object"},
        plugin={
            "document": {
                "metadata": {"id": "example"},
                "runtime": {"kind": "cpp", "entrypoint": "tensor.v1"},
            }
        },
        adapter_config={
            "model_alias": "primary",
            "inputs": {
                "input": {"request_field": "values", "codec": "json-array-fp32"}
            },
            "outputs": {
                "output": {"response_field": "values", "codec": "json-array-fp32"}
            },
        },
        paths={"workflow_root": str(tmp_path), "repository_root": str(ROOT)},
    )
    value["target"]["deployment"].update(
        mig="disabled", driver_requirements={"minimum_driver_version": "570.195.03"}
    )
    return value


def test_staging_uses_manifest_package_path_and_never_copies_build_sources(
    tmp_path, monkeypatch
):
    output = candidate(tmp_path)
    value = staging_resolved(tmp_path)
    monkeypatch.setattr(release, "model_input_identity", lambda *_: None)
    bundle = tmp_path / "bundle"
    manifest = release.stage_release(
        value,
        {"primary": output},
        bundle,
        sources=[{"path": "unit-only", "sha256": "f" * 64, "size_bytes": 1}],
    )
    deployment = release.read_json(bundle / "deployment.json")
    assert (
        deployment["workflow"]["models"]["primary"]["package_path"]
        == "/opt/physicsnemo-release/model/primary/backends/aoti"
    )
    assert deployment["schema_version"] == 2
    assert deployment["target"]["compilation"]["gpu_product"] == "NVIDIA H100 80GB HBM3"
    assert manifest["models"]["primary"]["package"] == "backends/aoti"
    assert not list(bundle.rglob("*.py"))
    assert not (bundle / "source").exists()
    release.verify_bundle(bundle)
    (bundle / "model/primary/backends/aoti/model.pt2").write_bytes(
        b"changed after staging"
    )
    import pytest

    with pytest.raises(ValueError, match="changed"):
        release.verify_bundle(bundle)


def test_partial_alias_or_wrong_self_test_cannot_create_release_context(
    tmp_path, monkeypatch
):
    import pytest

    output = candidate(tmp_path)
    value = staging_resolved(tmp_path)
    monkeypatch.setattr(release, "model_input_identity", lambda *_: None)
    with pytest.raises(ValueError, match="all fixed model aliases"):
        release.stage_release(value, {}, tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
    value["self_tests"]["primary"]["inputs"]["input"] = [0]
    with pytest.raises(ValueError, match="element count"):
        release.stage_release(value, {"primary": output}, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_sensitive_timeout_cannot_leak_exporter_bearer_token(tmp_path, monkeypatch):
    import subprocess
    import pytest

    token = "secret-ephemeral-export-token"

    def timeout(argv, **_):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    runner = release.ProcessRunner(tmp_path / "commands")
    with pytest.raises(Exception) as failure:
        runner.run(
            ["lep", "deployment", "create", "--tokens", token],
            sensitive=True,
            timeout=1,
        )
    assert token not in str(failure.value)
    assert token not in "".join(
        path.read_text() for path in (tmp_path / "commands").iterdir()
    )


def test_compile_driver_preserves_nondefault_project_filename(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "native-inference/deploy"))
    from native_release_job import relocate_projects

    value = resolved()
    value["model_aliases"]["primary"]["project"] = "/checkout/model/custom-project.json"
    relocate_projects(value, tmp_path)
    assert value["model_aliases"]["primary"]["project"] == str(
        tmp_path / "projects/primary/custom-project.json"
    )


def test_default_builder_toolchain_lock_changes_compilation_identity(tmp_path):
    config = resolved()
    root = tmp_path / "repo/native-inference/model-builder"
    (root / "src").mkdir(parents=True)
    (root / "src/builder.py").write_text("compiler source")
    lock = root / "toolchain.lock.json"
    lock.write_text('{"format_version":1,"revision":"first"}')
    config["paths"] = {"repository_root": str(tmp_path / "repo")}
    first = release.compilation_key(config, "primary")
    lock.write_text('{"format_version":1,"revision":"changed-compiler"}')
    assert first != release.compilation_key(config, "primary"), (
        "default toolchain lock must invalidate compiled cache"
    )


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ({"primary": "b" * 40}, "b" * 40),
        ({"first": "b" * 40, "second": "c" * 40}, ""),
    ],
)
def test_generic_image_global_compilation_env_is_actual_common_source(
    tmp_path, sources, expected
):
    from types import SimpleNamespace

    config = resolved()
    config.update(source_sha="a" * 40, output={"repository": "registry.example/native"})
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    release.write_json(bundle / "deployment.json", {"model_build_sources": sources})
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            calls.append(argv)
            release.write_json(
                Path(argv[argv.index("--metadata-file") + 1]),
                {"containerimage.digest": "sha256:" + "d" * 64},
            )
            return SimpleNamespace(returncode=0, stdout="")

    release.build_image(
        config,
        source=tmp_path,
        custom=tmp_path,
        inputs=tmp_path,
        bundle=bundle,
        workspace=tmp_path,
        runner=Runner(),
        target="release",
    )
    args = calls[0]
    actual = [value for value in args if value.startswith("MODEL_BUILD_SOURCE_SHA=")]
    assert actual == ["MODEL_BUILD_SOURCE_SHA=" + expected], (
        "image env must not falsely claim a cached model was built from the new release"
    )
