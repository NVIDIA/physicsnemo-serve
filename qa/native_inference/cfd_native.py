"""Package native-produced field bytes as independently checked QA evidence."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from qa.native_inference.cfd_contract import (
    ARRAY_NAMES,
    ARRAY_WIDTHS,
    ASSET_PINS,
    CASE_PINS,
    MODEL_REVISION,
    FieldContractError,
    array_descriptor,
    block_plan,
    inference_seed,
    validate_evidence,
    verify_file,
)
from qa.scripts.cfd_parity_contract import (
    read_json_object,
    sha256_file,
    write_json_atomic,
)


def verify_package_inventory(package: Path, receipt: dict) -> None:
    records = receipt.get("files", [])
    observed = set()
    for record in records:
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise FieldContractError("unsafe compiled package receipt path")
        if relative in observed:
            raise FieldContractError("duplicate compiled package receipt path")
        observed.add(relative)
        path = package / relative
        if path.is_symlink() or not path.resolve().is_relative_to(package.resolve()):
            raise FieldContractError(
                "compiled package contains symlinks or escaped paths"
            )
        verify_file(path, record)
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    }
    if observed != actual or "model.json" not in observed:
        raise FieldContractError("compiled package file inventory mismatch")


def validate_compilation_binding(receipt, *, source_sha, compilation_source_sha=None):
    sources = receipt.get("model_build_sources")
    if sources:
        if not isinstance(sources, dict) or not all(
            isinstance(alias, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", alias)
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{40}", value)
            for alias, value in sources.items()
        ):
            raise FieldContractError("compiled alias source provenance is invalid")
        actual = set(sources.values())
        compiled = next(iter(actual)) if len(actual) == 1 else None
        if compilation_source_sha is not None and compiled != compilation_source_sha:
            raise FieldContractError(
                "selected compilation source differs from actual alias sources"
            )
        return compiled
    compiled = compilation_source_sha or source_sha
    if (
        not re.fullmatch(r"[0-9a-f]{40}", compiled)
        or receipt.get("source_sha") != compiled
    ):
        raise FieldContractError(
            "compiled package source_sha differs from selected compilation source"
        )
    return compiled


def package_native_fields(
    *,
    profile,
    metadata_path: Path,
    receipt_path: Path,
    input_dumps: Path,
    output_path: Path,
    image: str,
    source_sha: str,
    compilation_source_sha: str | None = None,
):
    metadata, receipt = read_json_object(metadata_path), read_json_object(receipt_path)
    backend = receipt.get("backend", "aoti")
    if backend not in {"aoti", "tensorrt"}:
        raise FieldContractError("unsupported native CFD receipt backend")
    expected = {
        "schema_version": 2,
        "domain": "surface",
        "backend": backend,
        "point_limit": 0,
        "block_size": 60000,
        "permutation": "torch.cuda.randperm",
        "permutation_seed": inference_seed(42, "run_1"),
        "air_density": 1.205,
        "stream_velocity": 30.0,
        "math_profile": "provider-default-v1",
        "output_dtype": "float32",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise FieldContractError(f"native runner {key} mismatch")
    compiled = validate_compilation_binding(
        receipt, source_sha=source_sha, compilation_source_sha=compilation_source_sha
    )
    if not re.fullmatch(
        r".+@sha256:[0-9a-f]{64}", str(receipt.get("builder_image", ""))
    ):
        raise FieldContractError(
            "compiled package builder image must use an immutable digest"
        )
    for key in ("sha256", "size_bytes"):
        if receipt.get("checkpoint", {}).get(key) != ASSET_PINS["checkpoint"][key]:
            raise FieldContractError(f"compiled checkpoint {key} mismatch")
    stats = verify_file(metadata["stats_path"], ASSET_PINS["statistics"])
    if receipt.get("stats_sha256") != stats["sha256"]:
        raise FieldContractError(
            "compiled statistics differ from native runner statistics"
        )
    pin = CASE_PINS["run_1"]
    verify_file(
        metadata["mesh"], {"sha256": pin["sha256"], "size_bytes": pin["size_bytes"]}
    )
    verify_file(
        metadata["stl"],
        {"sha256": pin["geometry_sha256"], "size_bytes": pin["geometry_size_bytes"]},
    )
    points = metadata["point_count"]
    plan = block_plan(points)
    if (
        metadata.get("block_count") != len(plan)
        or [block["points"] for block in metadata["block_plan"]] != plan
    ):
        raise FieldContractError(
            "native executed block plan differs from provider plan"
        )
    if {item["point_dimension"] for item in metadata["package_profiles"]} != set(plan):
        raise FieldContractError(
            "native static package shape inventory differs from full/tail plan"
        )
    manifests = []
    for value in metadata["packages"]:
        path = Path(value)
        if path.is_dir():
            path = path / "model.json"
        digest = sha256_file(path)
        matches = [
            item for item in receipt["packages"] if item["manifest_sha256"] == digest
        ]
        if len(matches) != 1:
            raise FieldContractError(
                "native package is absent or duplicated in build receipt"
            )
        manifest = read_json_object(path)
        artifacts = manifest.get("artifacts", [])
        if len(artifacts) != 1 or artifacts[0].get("backend") != backend:
            raise FieldContractError(
                "native package backend differs from qualified build receipt"
            )
        verify_package_inventory(path.parent, matches[0])
        manifests.append(digest)
    if sorted(manifests) != sorted(
        item["manifest_sha256"] for item in receipt["packages"]
    ):
        raise FieldContractError(
            "native package bytes differ from qualified build receipt"
        )
    deployment = metadata["deployment"]
    cuda = deployment["cuda_runtime_version"]
    environment = {
        "gpu_name": deployment["name"],
        "compute_capability": deployment["compute_capability"],
        "torch": deployment["torch_version"],
        "cuda": f"{cuda // 1000}.{(cuda % 1000) // 10}",
        "vtk": metadata["vtk_version"],
        "math": metadata["math_flags"],
    }
    root = output_path.parent.resolve()
    arrays = {}
    paths = {
        "fx": input_dumps / "fx.f32",
        "embedding": input_dumps / "embedding.f32",
        "permutation": input_dumps / "permutation.i64",
        "source_cell_ids": input_dumps / "source_cell_ids.i64",
        "standardized": Path(metadata["standardized_output"]),
        "physical": Path(metadata["physical_output"]),
    }
    for name in ARRAY_NAMES:
        shape = (
            (1, points, ARRAY_WIDTHS[name])
            if name in ("fx", "embedding")
            else (points, ARRAY_WIDTHS[name])
            if name in ARRAY_WIDTHS
            else (points,)
        )
        dtype = "<f4" if name in ARRAY_WIDTHS else "<i8"
        path = paths[name].resolve(strict=True)
        if not path.is_relative_to(root):
            raise FieldContractError(
                "native artifacts must stay inside the immutable attempt evidence directory"
            )
        if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise FieldContractError(f"native {name} byte count mismatch")
        value = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        arrays[name] = array_descriptor(path, root, value, encoding="raw")
    evidence = {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "runtime": f"native-{backend}",
        "model": "transolver_surface",
        "model_revision": MODEL_REVISION,
        "reference_provider": profile["provider"],
        "source_sha": source_sha,
        "compilation_source_sha": compiled,
        **(
            {"model_build_sources": receipt["model_build_sources"]}
            if receipt.get("model_build_sources")
            else {}
        ),
        "image": image,
        "checkpoint": receipt["checkpoint"],
        "statistics": stats,
        "environment": environment,
        "build_receipt_sha256": sha256_file(receipt_path),
        "builder_image": receipt["builder_image"],
        "package_manifest_sha256": manifests,
        "cases": [
            {
                "case_id": "run_1",
                "point_count": points,
                "extent": "all-source-cells",
                "run_seed": 42,
                "inference_seed": metadata["permutation_seed"],
                "batch_resolution": 60000,
                "block_points": plan,
                "output_location": "cell",
                "source_order": True,
                "inputs": pin,
                "arrays": arrays,
            }
        ],
    }
    validate_evidence(profile, evidence, root)
    write_json_atomic(output_path, evidence)
    return evidence
