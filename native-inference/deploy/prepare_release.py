"""Assemble a native Serve release context from verified GPU build outputs.

This standard-library-only tool neither imports Torch nor compiles a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

INSTALL_ROOT = PurePosixPath("/opt/physicsnemo-native")
RUN1_POINT_COUNT = 8_828_095
BLOCK_POINTS = {60000, 8095}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    value = json.loads(Path(path).read_text())
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(package):
    _require(
        package.is_dir() and not package.is_symlink(),
        f"package must be a regular directory: {package}",
    )
    inventory = []
    for path in sorted(package.rglob("*")):
        _require(not path.is_symlink(), f"package symlinks are unsupported: {path}")
        if path.is_dir():
            continue
        _require(
            path.is_file() and path.stat().st_size > 0,
            f"nonempty regular package file required: {path}",
        )
        inventory.append(
            {
                "path": path.relative_to(package).as_posix(),
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def _package_points(package, manifest, environment, backend):
    inputs = {value.get("name"): value for value in manifest.get("inputs", [])}
    fx, embedding = inputs.get("fx", {}), inputs.get("embedding", {})
    shape = fx.get("shape", [])
    _require(
        len(shape) == 3
        and shape[0] == 1
        and type(shape[1]) is int
        and shape[1] > 1
        and shape[2] == 2,
        "native surface fx requires static [1,N,2]",
    )
    count = shape[1]
    outputs = manifest.get("outputs", [])
    _require(
        set(inputs) == {"fx", "embedding"}
        and embedding.get("shape") == [1, count, 6]
        and len(outputs) == 1
        and outputs[0].get("shape") == [1, count, 4],
        "native surface tensor names, widths and point dimensions must agree",
    )
    _require(
        all(tensor.get("dtype") == "float32" for tensor in [fx, embedding, outputs[0]]),
        "initial native release requires FP32 tensors",
    )
    artifacts = manifest.get("artifacts", [])
    _require(
        len(artifacts) == 1,
        "native release requires one selected-backend artifact per package",
    )
    artifact = artifacts[0]
    _require(
        (artifact.get("backend"), artifact.get("target"), artifact.get("precision"))
        == (backend, "cuda", "fp32"),
        f"native release requires CUDA {backend} FP32 packages",
    )
    # SDK manifests use Torch's public version; qualification also retains the
    # vendor build suffix. The immutable builder identity binds the full stack.
    _require(
        artifact.get("runtime_version")
        == (
            environment["torch_version"].split("+", 1)[0]
            if backend == "aoti"
            else environment.get("tensorrt_version")
        )
        and bool(artifact.get("runtime_version")),
        "package runtime differs from qualified environment",
    )
    _require(
        artifact.get("correctness_profile", {}).get("name")
        == (
            "aten-boundary-exact-v2"
            if backend == "aoti"
            else "provider-default-exact-v1"
        ),
        "Transolver release requires the selected backend exact correctness profile",
    )
    path = PurePosixPath(artifact.get("path", ""))
    _require(
        not path.is_absolute() and ".." not in path.parts and path.parts,
        "artifact path must remain inside its package",
    )
    _require((package / path).is_file(), "compiled package artifact is missing")
    return count


def prepare_release(
    *,
    builder_image: str,
    packages: list[Path],
    stats: Path,
    case_config: Path,
    compile_evidence: Path,
    source_sha: str,
    output: Path,
    device_index: int = 0,
    backend: str = "aoti",
) -> dict:
    _require(backend in {"aoti", "tensorrt"}, "unsupported native CFD backend")
    _require(
        re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", builder_image),
        "builder image must use an immutable registry digest",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", source_sha),
        "source_sha must be a full Git revision",
    )
    _require(
        type(device_index) is int and device_index >= 0,
        "device_index must be nonnegative",
    )
    output, stats, compile_evidence = Path(output), Path(stats), Path(compile_evidence)
    _require(
        not output.exists() and not output.is_symlink(), "release output already exists"
    )
    evidence = _read(compile_evidence)
    _require(
        evidence.get("backend", "aoti") == backend,
        "qualified backend differs from release selection",
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("status") == "passed"
        and evidence.get("qualification") == "full-run1-fields-and-metrics-v1",
        "completed full run_1 field and shared-metric qualification is required",
    )
    _require(
        evidence.get("builder_image") == builder_image,
        "compile evidence builder image differs from release selection",
    )
    compilation_source_sha = evidence.get("source_sha")
    _require(
        isinstance(compilation_source_sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", compilation_source_sha),
        "compilation source_sha must be a full Git revision",
    )
    environment = evidence.get("environment", {})
    _require(
        environment.get("compute_capability") == [9, 0]
        and "H100" in environment.get("gpu_name", ""),
        "initial release requires an H100 sm90 qualification",
    )
    _require(
        environment.get("torch_version") and environment.get("cuda_version"),
        "Torch and CUDA versions are required",
    )
    _require(
        evidence.get("point_count") == RUN1_POINT_COUNT
        and set(evidence.get("block_points", [])) == BLOCK_POINTS,
        "qualification must cover full run_1 with 60000-point blocks and its exact 8095-point tail",
    )
    summary_path = compile_evidence.parent / "summary.json"
    _require(
        _sha(summary_path) == evidence.get("summary_sha256")
        and _read(summary_path).get("status") == "passed",
        "qualified summary is missing or changed",
    )
    _require(
        _sha(stats) == evidence.get("stats_sha256"),
        "normalization statistics changed after qualification",
    )

    declared = evidence.get("packages", [])
    _require(
        isinstance(declared, list)
        and len(declared) == len(packages) == len(BLOCK_POINTS),
        "both qualified full and tail packages are required",
    )
    by_manifest = {item.get("manifest_sha256"): item for item in declared}
    _require(len(by_manifest) == len(declared), "duplicate package receipt identity")
    staged = []
    counts = set()
    for package in map(Path, packages):
        files = _files(package)
        digest = _sha(package / "model.json")
        receipt = by_manifest.get(digest)
        _require(
            receipt is not None, "compiled package manifest changed after qualification"
        )
        _require(
            sorted(receipt.get("files", []), key=lambda item: item["path"]) == files,
            "compiled package bytes or file inventory changed after qualification",
        )
        count = _package_points(
            package, _read(package / "model.json"), environment, backend
        )
        _require(count not in counts, "duplicate package point dimension")
        counts.add(count)
        staged.append((count, package, files))
    _require(
        counts == BLOCK_POINTS,
        "compiled full/tail package coverage differs from pinned run_1",
    )
    request = _read(case_config)
    _require(
        request.get("models") == ["transolver_surface"]
        and len(request.get("cases", [])) == 1,
        "case configuration must select the Transolver-only run_1 request",
    )
    case = request["cases"][0]
    _require(
        case.get("case_id") == "run_1", "initial release only supports pinned run_1"
    )
    objects = []
    for url_key, digest_key, size_key in [
        ("mesh_uri", "sha256", "size_bytes"),
        ("geometry_uri", "geometry_sha256", "geometry_size_bytes"),
    ]:
        _require(
            isinstance(case.get(url_key), str)
            and case[url_key].startswith("https://")
            and re.fullmatch(r"[0-9a-f]{64}", case.get(digest_key, ""))
            and type(case.get(size_key)) is int
            and case[size_key] > 0,
            "invalid pinned case input descriptor",
        )
        objects.append(
            {
                "url": case[url_key],
                "sha256": case[digest_key],
                "size_bytes": case[size_key],
            }
        )
    # Hashes pin the known run_1 geometry whose full cell count was qualified.
    _require(
        [item["size_bytes"] for item in objects] == [659606189, 142385186]
        and [item["sha256"] for item in objects]
        == [
            "01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40",
            "411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2",
        ],
        "run_1 geometry digests differ",
    )

    assets, package_paths = [], []
    for count, _, files in sorted(staged, reverse=True):
        relative = PurePosixPath(f"models/points-{count}")
        package_paths.append(str(INSTALL_ROOT / relative))
        assets.extend(
            {**item, "path": str(INSTALL_ROOT / relative / item["path"])}
            for item in files
        )
    stats_relative = PurePosixPath("stats/global_stats.json")
    assets.append(
        {
            "path": str(INSTALL_ROOT / stats_relative),
            "sha256": _sha(stats),
            "size_bytes": stats.stat().st_size,
        }
    )
    config = dict(
        schema_version=1,
        workflow_id="physicsnemo-cfd-transolver-native",
        backend=backend,
        executor_class=f"native.transolver.{backend}",
        device_index=device_index,
        compute_capability=[9, 0],
        packages=package_paths,
        stats_path=str(INSTALL_ROOT / stats_relative),
        model_assets=assets,
        block_size=60000,
        artifact_root="/outputs/native-cfd",
        input_cache="/outputs/.cache/native-cfd",
        case=dict(
            case_id="run_1",
            point_count=RUN1_POINT_COUNT,
            mesh=objects[0],
            geometry=objects[1],
        ),
        request_timeout_seconds=3600,
        drain_timeout_seconds=60,
    )
    inventory = dict(
        schema_version=1,
        backend=backend,
        builder_image=builder_image,
        source_sha=source_sha,
        compilation_source_sha=compilation_source_sha,
        qualification=evidence["qualification"],
        environment=environment,
        model_assets=assets,
        compile_evidence_sha256=_sha(compile_evidence),
        summary_sha256=_sha(summary_path),
    )
    dockerfile = (Path(__file__).parent / "Dockerfile.native").read_text()
    dockerfile = dockerfile.replace(
        "ARG NATIVE_BUILDER_IMAGE\n", f"ARG NATIVE_BUILDER_IMAGE={builder_image}\n"
    )
    dockerfile = dockerfile.replace(
        "ARG SOURCE_SHA\n", f"ARG SOURCE_SHA={source_sha}\n"
    )
    dockerfile = dockerfile.replace(
        "ARG MODEL_BUILD_SOURCE_SHA\n",
        f"ARG MODEL_BUILD_SOURCE_SHA={compilation_source_sha}\n",
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        for count, package, files in staged:
            shutil.copytree(package, output / "models" / f"points-{count}")
            _require(
                _files(output / "models" / f"points-{count}") == files,
                "package changed while copying",
            )
        (output / "stats").mkdir()
        shutil.copy2(stats, output / stats_relative)
        _require(
            _sha(output / stats_relative) == evidence["stats_sha256"],
            "statistics changed while copying",
        )
        shutil.copy2(compile_evidence, output / "build-receipt.json")
        shutil.copy2(summary_path, output / "summary.json")
        shutil.copy2(
            Path(__file__).parent / "worker_runtime_config.json",
            output / "worker_runtime_config.json",
        )
        (output / "deployment.json").write_text(json.dumps(config, indent=2) + "\n")
        (output / "release-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        (output / "Dockerfile").write_text(dockerfile)
    except BaseException:
        shutil.rmtree(output)
        raise
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder-image", required=True)
    parser.add_argument(
        "--package", type=Path, action="append", required=True, dest="packages"
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    args = parser.parse_args()
    try:
        prepare_release(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"native release rejected: {error}\n")
    print(f"Native release context ready: {args.output}")


if __name__ == "__main__":
    main()
