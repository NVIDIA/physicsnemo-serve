# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a raw FP32 QA workflow without changing its model project.

Usage: python qa/scripts/create_native_raw_variant.py JSON_WORKFLOW RAW_WORKFLOW
The source must be the public CLI's tensor scaffold with authored shape-[4]
acceptance cases. This performs no model export, compilation or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contained(root, relative):
    path = Path(relative)
    require(
        not path.is_absolute() and ".." not in path.parts,
        "scaffold paths must stay inside its directory",
    )
    result = root / path
    require(result.is_file(), f"missing scaffold file: {relative}")
    return result


def identity(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def model_inventory(root):
    return {
        path.relative_to(root).as_posix(): identity(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def fp32(values):
    require(
        isinstance(values, list)
        and len(values) == 4
        and all(
            type(value) in (int, float)
            and abs(value) <= 3.4028234663852886e38
            and math.isfinite(value)
            for value in values
        ),
        "each authored tensor requires four finite FP32 values",
    )
    try:
        return struct.pack("<4f", *values)
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "each authored tensor requires four finite FP32 values"
        ) from error


def descriptor(artifact):
    return {
        "artifact": artifact,
        "dtype": "float32",
        "shape": [4],
        "byte_order": "little",
        "layout": "contiguous",
        "size_bytes": 16,
    }


def schema(artifacts):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["values"],
        "properties": {
            "values": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "artifact",
                    "dtype",
                    "shape",
                    "byte_order",
                    "layout",
                    "size_bytes",
                ],
                "properties": {
                    "artifact": {"type": "string", "enum": artifacts},
                    "dtype": {"const": "float32"},
                    "shape": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"const": 4},
                    },
                    "byte_order": {"const": "little"},
                    "layout": {"const": "contiguous"},
                    "size_bytes": {"type": "integer", "const": 16},
                },
            }
        },
    }


def create_raw_variant(source_workflow, destination):
    """Copy a fresh tensor scaffold, preserving every byte of its model tree."""
    source = Path(source_workflow)
    require(
        source.is_dir() and not source.is_symlink(),
        "source workflow must be a regular directory",
    )
    source = source.resolve()
    destination = Path(destination).absolute()
    require(
        not destination.exists() and not destination.is_symlink(),
        "destination already exists",
    )
    destination = destination.parent.resolve() / destination.name
    require(
        not destination.is_relative_to(source),
        "destination must not be inside the source workflow",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", destination.name),
        "destination must name a lowercase workflow identifier",
    )
    require(
        all(not path.is_symlink() for path in source.rglob("*")),
        "source scaffold must not contain symlinks",
    )
    manifest = yaml.safe_load(contained(source, "plugin.yaml").read_text())
    profile = yaml.safe_load(contained(source, "release.yaml").read_text())
    require(
        manifest["runtime"]["kind"] == "cpp"
        and manifest["runtime"]["entrypoint"] == "tensor.v1",
        "source must be a native tensor.v1 scaffold",
    )
    mapping = manifest["configuration"]["native"]["tensor_adapter"]
    require(
        set(mapping["inputs"]) == {"input"} and set(mapping["outputs"]) == {"output"},
        "source must use the scaffold input/output tensor names",
    )
    require(
        mapping["inputs"]["input"]["request_field"] == "values"
        and mapping["outputs"]["output"]["response_field"] == "values",
        "source must map the scaffold values fields",
    )
    model = contained(source, profile["model_project"]).parent
    model_files = model_inventory(model)
    case_path = profile["qualification"]["cases"]
    cases = read_json(contained(source, case_path))
    require(
        cases.get("schema_version") == 1 and cases.get("cases"),
        "authored acceptance cases are required",
    )
    tensors = []
    names = set()
    for case in cases["cases"]:
        name = case["name"]
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name)
            and name not in names,
            "unique simple case names are required",
        )
        names.add(name)
        require(
            set(case["request"]) == {"values"} and set(case["expected"]) == {"values"},
            "source cases must contain the scaffold values fields only",
        )
        tensors.append(
            (case, fp32(case["request"]["values"]), fp32(case["expected"]["values"]))
        )
    require(
        not profile.get("input_artifacts"),
        "fresh scaffold must not already declare raw input artifacts",
    )
    profile["input_artifacts"] = {}
    for case, inputs, expected in tensors:
        logical = "raw-" + case["name"]
        relative = "fixtures/raw/" + case["name"] + ".bin"
        profile["input_artifacts"][logical] = {"source": relative, **identity(inputs)}
        case["request"]["values"] = descriptor(logical)
        case["expected"]["values"] = descriptor("output.f32")
        case["expected_artifacts"] = {"output.f32": identity(expected)}
    workflow = destination.name
    executor = "native." + workflow
    manifest["metadata"].update(
        id=workflow, display_name=workflow.replace("-", " ").title()
    )
    manifest["runtime"]["executor_class"] = executor
    for stage in manifest["pipeline"]["stages"]:
        if stage["phase"] == "execute":
            stage["queue"] = "execute." + executor
    mapping["inputs"]["input"]["codec"] = "raw-fp32-le"
    mapping["outputs"]["output"]["codec"] = "raw-fp32-le"
    manifest["outputs"]["primary_artifact"] = {
        "name": "output.f32",
        "media_type": "application/octet-stream",
    }
    profile["workflow"] = workflow
    if "executor_class" in profile.get("runtime", {}):
        profile["runtime"]["executor_class"] = executor
    created = False
    try:
        destination.mkdir(parents=True, exist_ok=False)
        created = True
        shutil.copytree(source, destination, dirs_exist_ok=True)
        for case, inputs, expected in tensors:
            path = destination / "fixtures/raw" / (case["name"] + ".bin")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(inputs)
            path.with_suffix(".expected.bin").write_bytes(expected)
        (destination / "plugin.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        (destination / "release.yaml").write_text(
            yaml.safe_dump(profile, sort_keys=False)
        )
        write_json(destination / case_path, cases)
        write_json(
            destination / manifest["ingress"]["json_schema"],
            schema(list(profile["input_artifacts"])),
        )
        write_json(
            destination / manifest["outputs"]["result_schema"], schema(["output.f32"])
        )
        write_json(
            destination / "examples/default_request.json", cases["cases"][0]["request"]
        )
        require(
            model_inventory(destination / model.relative_to(source)) == model_files,
            "model project bytes changed while deriving raw variant",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/plugin_dev.py"),
                "validate",
                str(destination),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        require(
            result.returncode == 0,
            "derived workflow failed public CLI validation: " + result.stderr.strip(),
        )
        return destination
    except BaseException:
        if created:
            shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_workflow", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        destination = create_raw_variant(args.source_workflow, args.destination)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "created",
                "workflow_id": destination.name,
                "destination": str(destination),
                "model_project_bytes": "preserved",
            }
        )
    )


if __name__ == "__main__":
    main()
