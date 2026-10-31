"""Export independent Python cases for native Transolver startup checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

import numpy as np

from qa.native_inference.cfd_contract import validate_evidence, verify_file
from qa.scripts.cfd_parity_contract import read_json_object


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("startup fixture must be a regular file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def stage(build_outputs, destination, *, backend="aoti"):
    """Copy Builder's IEEE-policy cases, for runners using that same policy.

    The provider-default runner must instead use ``stage_provider_reference``:
    Builder captures these references with highest matmul precision/TF32 off.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("startup fixture output already exists")
    models = []
    selected = []
    for points, build in sorted(build_outputs.items()):
        build = Path(build)
        check = json.loads((build / f"checks/{backend}.json").read_text())
        case = check["cases"][0]
        if check.get("passed") is not True or case.get("passed") is not True:
            raise ValueError("startup fixture requires passed Builder parity")
        if [item["name"] for item in case["inputs"]] != ["fx", "embedding"]:
            raise ValueError("unexpected Transolver startup input names")
        model = {"point_count": points, "inputs": {}, "rtol": 1e-5, "atol": 1e-6}
        definitions = [
            (
                "fx",
                build / f"checks/{backend}/case-0/input-0.bin",
                [1, points, 2],
                case["inputs"][0]["sha256"],
            ),
            (
                "embedding",
                build / f"checks/{backend}/case-0/input-1.bin",
                [1, points, 6],
                case["inputs"][1]["sha256"],
            ),
            (
                "output",
                build / f"checks/{backend}/case-0/reference-0.bin",
                [1, points, 4],
                case["outputs"][0]["reference_sha256"],
            ),
        ]
        for name, source, shape, digest in definitions:
            record = identity(source)
            if (
                record["sha256"] != digest
                or record["size_bytes"] != math.prod(shape) * 4
            ):
                raise ValueError("trusted Builder fixture bytes changed")
            data = source.read_bytes()
            if not all(
                math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)
            ):
                raise ValueError("startup reference contains nonfinite float32")
            path = f"{points}/{name}.bin"
            tensor = {"path": path, "shape": shape}
            if name == "output":
                model["expected_output"] = tensor
            else:
                model["inputs"][name] = tensor
            selected.append((source, path, record))
        models.append(model)
    if not models:
        raise ValueError("startup fixture needs at least one model")
    try:
        destination.mkdir(parents=True, exist_ok=False)
        files = []
        for source, path, expected in selected:
            output = destination / path
            output.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, output)
            if identity(output) != expected:
                raise ValueError("startup fixture changed while copying")
            files.append({"path": path, **expected})
        config = destination / "startup.json"
        config.write_text(
            json.dumps({"schema_version": 1, "models": models}, indent=2) + "\n"
        )
        files.append({"path": "startup.json", **identity(config)})
        return {
            "path": "startup.json",
            "files": sorted(files, key=lambda item: item["path"]),
        }
    except BaseException:
        shutil.rmtree(destination)
        raise


def stage_provider_reference(
    build_outputs, destination, *, profile, reference_path, backend="aoti"
):
    """Gather complete observed model calls under the deployed provider policy.

    Provider evidence stores prepared inputs and standardized outputs in source
    order. Gather both using the same recorded forward-block permutation slice;
    Builder's synthetic inputs and IEEE-policy references are never substituted.
    """
    destination, reference_path = Path(destination), Path(reference_path)
    if destination.exists() or destination.is_symlink():
        raise ValueError("startup fixture output already exists")
    reference_identity = identity(reference_path)
    reference = read_json_object(reference_path)
    if reference.get("runtime") != "python-provider":
        raise ValueError("startup truth must be an independent Python provider capture")
    loaded = validate_evidence(profile, reference, reference_path.parent)
    case = reference["cases"][0]
    if set(build_outputs) != set(case["block_points"]):
        raise ValueError("startup fixtures must cover every full/tail model shape")
    for build in build_outputs.values():
        check = read_json_object(Path(build) / f"checks/{backend}.json")
        if (
            check.get("passed") is not True
            or not all(item.get("passed") is True for item in check.get("cases", []))
            or not check.get("cases")
        ):
            raise ValueError("startup fixture requires passed Builder parity")

    arrays = loaded[case["case_id"]]
    models, files, seen = [], [], set()
    try:
        destination.mkdir(parents=True, exist_ok=False)
        begin = 0
        for points in case["block_points"]:
            if points not in seen:
                seen.add(points)
                indices = arrays["permutation"][begin : begin + points]
                model = {
                    "point_count": points,
                    "inputs": {},
                    "rtol": 1e-5,
                    "atol": 1e-6,
                    "reference_block_begin": begin,
                }
                for name, values in (
                    ("fx", arrays["fx"][:, indices, :]),
                    ("embedding", arrays["embedding"][:, indices, :]),
                    ("output", arrays["standardized"][indices, :][None]),
                ):
                    values = np.ascontiguousarray(values, dtype="<f4")
                    relative = f"{points}/{name}.bin"
                    output = destination / relative
                    output.parent.mkdir(exist_ok=True)
                    values.tofile(output)
                    record = {"path": relative, **identity(output)}
                    files.append(record)
                    tensor = {"path": relative, "shape": list(values.shape)}
                    if name == "output":
                        model["expected_output"] = tensor
                    else:
                        model["inputs"][name] = tensor
                models.append(model)
            begin += points
        # Bind the copied values to the same independent capture throughout the
        # gather, rather than accepting a source changed after validation.
        for name in ("fx", "embedding", "standardized", "permutation"):
            descriptor = case["arrays"][name]
            verify_file(reference_path.parent / descriptor["path"], descriptor)
        if identity(reference_path) != reference_identity:
            raise ValueError(
                "provider reference changed while staging startup fixtures"
            )
        provenance = {
            **reference_identity,
            **{
                key: reference[key]
                for key in (
                    "runtime",
                    "profile_id",
                    "source_sha",
                    "image",
                    "model_revision",
                    "provider",
                    "checkpoint",
                    "statistics",
                    "environment",
                )
            },
            "case_id": case["case_id"],
            "inference_seed": case["inference_seed"],
            "point_count": case["point_count"],
            "arrays": case["arrays"],
        }
        config = destination / "startup.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": models,
                    "math_flags": reference["environment"]["math"],
                    "reference": provenance,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        files.append({"path": "startup.json", **identity(config)})
        return {
            "path": "startup.json",
            "files": sorted(files, key=lambda item: item["path"]),
        }
    except BaseException:
        shutil.rmtree(destination)
        raise
