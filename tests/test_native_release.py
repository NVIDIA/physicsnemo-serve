from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "native-inference" / "deploy"
spec = importlib.util.spec_from_file_location(
    "native_release", DEPLOY / "prepare_release.py"
)
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
IMAGE = "registry.example/native-builder@sha256:" + "a" * 64
SOURCE = "b" * 40


def _fixture(tmp_path, backend="aoti"):
    packages = []
    for count in (60000, 8095):
        package = tmp_path / f"build-{count}"
        package.mkdir()
        manifest = {
            "format_version": 1,
            "inputs": [
                {"name": "fx", "dtype": "float32", "shape": [1, count, 2]},
                {"name": "embedding", "dtype": "float32", "shape": [1, count, 6]},
            ],
            "outputs": [
                {
                    "name": "surface_fields_standardized",
                    "dtype": "float32",
                    "shape": [1, count, 4],
                }
            ],
            "artifacts": [
                {
                    "backend": backend,
                    "target": "cuda",
                    "precision": "fp32",
                    "path": "model.pt2" if backend == "aoti" else "model.plan",
                    "runtime_version": "qualified-torch"
                    if backend == "aoti"
                    else "10.14.1.48",
                    "correctness_profile": {
                        "name": "aten-boundary-exact-v2"
                        if backend == "aoti"
                        else "provider-default-exact-v1"
                    },
                }
            ],
        }
        (package / "model.json").write_text(json.dumps(manifest))
        (package / ("model.pt2" if backend == "aoti" else "model.plan")).write_bytes(
            b"stand-in compiled model; packaging does not execute it"
        )
        packages.append(package)
    stats = tmp_path / "global_stats.json"
    stats.write_text(
        json.dumps(
            {
                "mean": {"pressure": [0], "shear_stress": [0, 0, 0]},
                "std_dev": {"pressure": [1], "shear_stress": [1, 1, 1]},
            }
        )
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status": "passed"}))
    evidence = tmp_path / "build-receipt.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **({"backend": backend} if backend == "tensorrt" else {}),
                "builder_image": IMAGE,
                "source_sha": SOURCE,
                "status": "passed",
                "qualification": "full-run1-fields-and-metrics-v1",
                "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "environment": {
                    "compute_capability": [9, 0],
                    "gpu_name": "NVIDIA H100 80GB HBM3",
                    "torch_version": "qualified-torch",
                    "cuda_version": "13.1",
                    **(
                        {"tensorrt_version": "10.14.1.48"}
                        if backend == "tensorrt"
                        else {}
                    ),
                },
                "point_count": 8828095,
                "block_points": [60000, 8095],
                "packages": [
                    {
                        "manifest_sha256": hashlib.sha256(
                            (p / "model.json").read_bytes()
                        ).hexdigest(),
                        "files": [
                            {
                                "path": f.name,
                                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                "size_bytes": f.stat().st_size,
                            }
                            for f in sorted(p.iterdir())
                        ],
                    }
                    for p in packages
                ],
                "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            }
        )
    )
    return dict(
        builder_image=IMAGE,
        packages=packages,
        stats=stats,
        case_config=ROOT
        / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json",
        compile_evidence=evidence,
        source_sha=SOURCE,
        output=tmp_path / "release",
    )


def test_native_release_contains_every_compiled_file_with_exact_hashes(tmp_path):
    args = _fixture(tmp_path)
    receipt = release.prepare_release(**args)
    config = json.loads((args["output"] / "deployment.json").read_text())
    assert config["workflow_id"] == "physicsnemo-cfd-transolver-native"
    assert config["executor_class"] == "native.transolver.aoti"
    assert config["block_size"] == 60000
    assert config["compute_capability"] == [9, 0]
    assert config["device_index"] == 0
    assert config["case"]["point_count"] == 8828095
    assert len(config["packages"]) == 2
    assert len(config["model_assets"]) == 5
    for asset in config["model_assets"]:
        assert asset["path"].startswith("/opt/physicsnemo-native/")
        path = args["output"] / Path(asset["path"]).relative_to(
            "/opt/physicsnemo-native"
        )
        assert asset["size_bytes"] == path.stat().st_size
        assert asset["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["builder_image"] == IMAGE
    assert receipt["source_sha"] == SOURCE
    dockerfile = (args["output"] / "Dockerfile").read_text()
    assert f"ARG NATIVE_BUILDER_IMAGE={IMAGE}" in dockerfile
    assert "COPY --from=release-assets / /opt/physicsnemo-native/" in dockerfile


def test_native_release_retains_vendor_torch_suffix_but_matches_sdk_public_version(
    tmp_path,
):
    args = _fixture(tmp_path)
    evidence = json.loads(args["compile_evidence"].read_text())
    evidence["environment"]["torch_version"] += "+vendor-build"
    args["compile_evidence"].write_text(json.dumps(evidence))
    inventory = release.prepare_release(**args)
    assert inventory["environment"]["torch_version"] == "qualified-torch+vendor-build"


def test_native_release_reuses_qualified_packages_for_a_later_api_release(tmp_path):
    args = _fixture(tmp_path)
    args["source_sha"] = "d" * 40
    inventory = release.prepare_release(**args)
    assert inventory["source_sha"] == "d" * 40
    assert inventory["compilation_source_sha"] == SOURCE
    retained = json.loads((args["output"] / "build-receipt.json").read_text())
    assert retained["source_sha"] == SOURCE
    dockerfile = (args["output"] / "Dockerfile").read_text()
    assert "ARG SOURCE_SHA=" + "d" * 40 in dockerfile
    assert f"ARG MODEL_BUILD_SOURCE_SHA={SOURCE}" in dockerfile
    assert "MODEL_BUILD_SOURCE_SHA=${MODEL_BUILD_SOURCE_SHA}" in dockerfile


@pytest.mark.parametrize("compilation_source", [None, "", "a84cc", "A" * 40])
def test_native_release_requires_a_valid_compilation_revision(
    tmp_path, compilation_source
):
    args = _fixture(tmp_path)
    evidence = json.loads(args["compile_evidence"].read_text())
    evidence["source_sha"] = compilation_source
    args["compile_evidence"].write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="compilation.*Git revision"):
        release.prepare_release(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "mutable_image",
        "different_image",
        "selected_builder_changed",
        "gpu_type",
        "gpu_type_same_arch",
        "torch_version",
        "package_tampered",
        "payload_tampered",
        "stats_tampered",
        "missing_tail",
        "symlink",
        "unqualified",
        "summary_tampered",
    ],
)
def test_native_release_rejects_unqualified_or_changed_inputs_before_staging(
    tmp_path, mutation
):
    args = _fixture(tmp_path)
    args["source_sha"] = "d" * 40
    evidence = json.loads(args["compile_evidence"].read_text())
    if mutation == "mutable_image":
        args["builder_image"] = "registry.example/native:latest"
    elif mutation == "different_image":
        evidence["builder_image"] = "registry.example/native@sha256:" + "c" * 64
    elif mutation == "selected_builder_changed":
        args["builder_image"] = "registry.example/native@sha256:" + "c" * 64
    elif mutation == "gpu_type":
        evidence["environment"].update(gpu_name="A10G", compute_capability=[8, 6])
    elif mutation == "gpu_type_same_arch":
        evidence["environment"]["gpu_name"] = "NVIDIA H200"
    elif mutation == "torch_version":
        evidence["environment"]["torch_version"] = "other-torch"
    elif mutation == "package_tampered":
        (args["packages"][0] / "model.json").write_text("{}")
    elif mutation == "payload_tampered":
        (args["packages"][0] / "model.pt2").write_bytes(b"changed-compiled-model")
    elif mutation == "stats_tampered":
        args["stats"].write_text("{}")
    elif mutation == "missing_tail":
        args["packages"] = args["packages"][:1]
    elif mutation == "symlink":
        (args["packages"][0] / "external.so").symlink_to(args["stats"])
    elif mutation == "unqualified":
        evidence["status"] = "building"
    elif mutation == "summary_tampered":
        (args["compile_evidence"].parent / "summary.json").write_text("{}")
    args["compile_evidence"].write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        release.prepare_release(**args)
    assert not args["output"].exists(), (
        "validation must finish before creating a partial context"
    )


def test_tensorrt_release_preserves_backend_engine_and_runtime_identity(tmp_path):
    args = _fixture(tmp_path, backend="tensorrt")
    receipt = release.prepare_release(**args, backend="tensorrt")
    config = json.loads((args["output"] / "deployment.json").read_text())
    assert config["backend"] == "tensorrt"
    assert config["executor_class"] == "native.transolver.tensorrt"
    assert receipt["backend"] == "tensorrt"
    assert all(
        (args["output"] / "models" / f"points-{n}" / "model.plan").is_file()
        for n in (60000, 8095)
    )


@pytest.mark.parametrize(
    "mutation",
    ["receipt_backend", "runtime", "profile", "ieee_profile", "artifact_backend"],
)
def test_tensorrt_release_rejects_cross_backend_or_unqualified_engines(
    tmp_path, mutation
):
    args = _fixture(tmp_path, backend="tensorrt")
    evidence = json.loads(args["compile_evidence"].read_text())
    if mutation == "receipt_backend":
        evidence["backend"] = "aoti"
    elif mutation == "runtime":
        evidence["environment"]["tensorrt_version"] = "other-trt"
    else:
        package = args["packages"][0]
        manifest = json.loads((package / "model.json").read_text())
        if mutation in {"profile", "ieee_profile"}:
            manifest["artifacts"][0]["correctness_profile"]["name"] = (
                "layout-order-exact-v2"
                if mutation == "ieee_profile"
                else "other-profile"
            )
        else:
            manifest["artifacts"][0]["backend"] = "aoti"
        (package / "model.json").write_text(json.dumps(manifest))
        evidence["packages"][0] = {
            "manifest_sha256": hashlib.sha256(
                (package / "model.json").read_bytes()
            ).hexdigest(),
            "files": release._files(package),
        }
    args["compile_evidence"].write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        release.prepare_release(**args, backend="tensorrt")
    assert not args["output"].exists()


def _image_start_guard():
    spec = importlib.util.spec_from_file_location(
        "native_release_start_guard", DEPLOY / "native_deploy_start.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not hasattr(os, "O_DIRECTORY"), reason="Linux image directory-fd startup guard"
)
def test_image_start_guard_claims_empty_mounted_scope_once(tmp_path):
    guard = _image_start_guard()
    root = tmp_path / "mounted"
    root.mkdir()
    result = guard.claim(
        root, operation_id="d" * 32, deployment_id="demo-fresh", image=IMAGE
    )
    assert result.get("status") == "claimed"
    assert (
        json.loads((root / ".native-deployment/owner.json").read_text())["operation_id"]
        == "d" * 32
    )
    with pytest.raises(ValueError, match="fresh|used|claim"):
        guard.claim(
            root, operation_id="d" * 32, deployment_id="demo-fresh", image=IMAGE
        )


@pytest.mark.skipif(
    not hasattr(os, "O_DIRECTORY"), reason="Linux image directory-fd startup guard"
)
def test_image_start_guard_never_overwrites_nonempty_scope(tmp_path):
    guard = _image_start_guard()
    (tmp_path / "existing-result.bin").write_bytes(b"keep")
    with pytest.raises(ValueError, match="fresh|nonempty"):
        guard.claim(
            tmp_path, operation_id="d" * 32, deployment_id="demo-fresh", image=IMAGE
        )
    assert (tmp_path / "existing-result.bin").read_bytes() == b"keep"


@pytest.mark.skipif(
    not hasattr(os, "O_DIRECTORY"), reason="Linux image directory-fd startup guard"
)
def test_image_start_guard_serializes_competing_claims(tmp_path):
    guard = _image_start_guard()
    from concurrent.futures import ThreadPoolExecutor

    def run(operation):
        try:
            return guard.claim(
                tmp_path,
                operation_id=operation * 32,
                deployment_id="demo-fresh",
                image=IMAGE,
            ).get("status")
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, ["a", "b"]))
    assert outcomes.count("claimed") == 1
    assert outcomes.count("rejected") == 1
