# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "create_native_raw_variant",
    Path(__file__).with_name("create_native_raw_variant.py"),
)
assert SPEC and SPEC.loader
raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw)


def cli(*args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/plugin_dev.py"), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def scaffold(tmp_path):
    path = tmp_path / "affine-json"
    cli("init", path, "--runtime", "cpp", "--template", "tensor")
    profile = yaml.safe_load((path / "release.yaml").read_text())
    profile["output"]["repository"] = "registry.example/native/affine"
    (path / "release.yaml").write_text(yaml.safe_dump(profile))
    return path


def tree(path):
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def test_raw_variant_uses_public_scaffold_and_reuses_exact_model_key(tmp_path):
    source = scaffold(tmp_path)
    before = tree(source)
    destination = tmp_path / "affine-raw"
    assert raw.create_raw_variant(source, destination) == destination.resolve()
    manifest = yaml.safe_load((destination / "plugin.yaml").read_text())
    assert manifest["metadata"]["id"] == "affine-raw", (
        "raw variant must have its own workflow ID"
    )
    assert manifest["runtime"]["executor_class"] == "native.affine-raw"
    assert (
        next(
            stage
            for stage in manifest["pipeline"]["stages"]
            if stage["phase"] == "execute"
        )["queue"]
        == "execute.native.affine-raw"
    )
    profile = yaml.safe_load((destination / "release.yaml").read_text())
    assert profile["workflow"] == "affine-raw"
    mapping = manifest["configuration"]["native"]["tensor_adapter"]
    assert mapping["inputs"]["input"]["codec"] == "raw-fp32-le"
    assert mapping["outputs"]["output"]["codec"] == "raw-fp32-le"
    assert tree(source) == before
    assert tree(destination / "model") == tree(source / "model")
    assert (destination / "fixtures/self-test.json").read_bytes() == (
        source / "fixtures/self-test.json"
    ).read_bytes()
    assert (
        json.loads((destination / "model/model-build.json").read_text())["name"]
        == "affine-json"
    )
    assert cli("validate", destination)["status"] == "valid"
    keys = [
        cli("release", path, "--plan", "--cache-dir", tmp_path / "cache")["models"][
            "primary"
        ]["compilation_key"]
        for path in (source, destination)
    ]
    assert keys[0] == keys[1], "codec-only workflow must reuse the verified model cache"


def test_raw_bytes_and_independent_authored_expected_artifacts(tmp_path):
    source = scaffold(tmp_path)
    authored = json.loads((source / "fixtures/cases.json").read_text())
    authored["cases"][1]["expected"]["values"] = [40, -4, 1.25, 27]
    (source / "fixtures/cases.json").write_text(json.dumps(authored))
    destination = raw.create_raw_variant(source, tmp_path / "affine-raw")
    profile = yaml.safe_load((destination / "release.yaml").read_text())
    cases = json.loads((destination / "fixtures/cases.json").read_text())["cases"]
    for original, case in zip(authored["cases"], cases, strict=True):
        descriptor = case["request"]["values"]
        assert descriptor == {
            "artifact": "raw-" + case["name"],
            "dtype": "float32",
            "shape": [4],
            "byte_order": "little",
            "layout": "contiguous",
            "size_bytes": 16,
        }
        selected = profile["input_artifacts"][descriptor["artifact"]]
        data = (destination / selected["source"]).read_bytes()
        assert selected == {
            "source": "fixtures/raw/" + case["name"] + ".bin",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": 16,
        }
        assert data == struct.pack("<4f", *original["request"]["values"])
        assert case["expected"]["values"] == {**descriptor, "artifact": "output.f32"}
        expected = struct.pack("<4f", *original["expected"]["values"])
        assert (
            destination / "fixtures/raw" / (case["name"] + ".expected.bin")
        ).read_bytes() == expected
        assert case["expected_artifacts"] == {
            "output.f32": {
                "sha256": hashlib.sha256(expected).hexdigest(),
                "size_bytes": 16,
            }
        }
    assert (
        json.loads((destination / "examples/default_request.json").read_text())
        == cases[0]["request"]
    )


def test_raw_request_and_response_schemas_reject_contract_mismatches(tmp_path):
    destination = raw.create_raw_variant(scaffold(tmp_path), tmp_path / "affine-raw")
    case = json.loads((destination / "fixtures/cases.json").read_text())["cases"][0]
    for filename, key in (("request.json", "request"), ("result.json", "expected")):
        validator = Draft202012Validator(
            json.loads((destination / "schemas" / filename).read_text())
        )
        validator.validate(case[key])
        invalid = [
            {"values": [0, 1, -1, 4]},
            {**case[key], "extra": True},
            *[
                {"values": {**case[key]["values"], field: value}}
                for field, value in (
                    ("shape", [2, 2]),
                    ("dtype", "float64"),
                    ("byte_order", "big"),
                    ("layout", "strided"),
                    ("size_bytes", 15),
                    ("artifact", "unknown"),
                    ("path", "/tmp/input.bin"),
                )
            ],
        ]
        assert all(not validator.is_valid(value) for value in invalid)


def test_existing_destination_and_nested_copy_are_rejected_without_mutation(tmp_path):
    source = scaffold(tmp_path)
    before = tree(source)
    destination = tmp_path / "affine-raw"
    destination.mkdir()
    (destination / "preserve.txt").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        raw.create_raw_variant(source, destination)
    assert (destination / "preserve.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="inside"):
        raw.create_raw_variant(source, source / "nested")
    assert tree(source) == before


@pytest.mark.parametrize(
    "values", ([1], [0, 1, True, 4], [0, 1, 1e40, 4], [0, 1, 10**400, 4])
)
def test_invalid_raw_fixture_is_rejected_before_destination_is_created(
    tmp_path, values
):
    source = scaffold(tmp_path)
    path = source / "fixtures/cases.json"
    cases = json.loads(path.read_text())
    cases["cases"][0]["request"]["values"] = values
    path.write_text(json.dumps(cases))
    destination = tmp_path / "affine-raw"
    with pytest.raises(ValueError, match="finite FP32"):
        raw.create_raw_variant(source, destination)
    assert not destination.exists()


def test_command_line_derives_and_validates_raw_workflow(tmp_path):
    source = scaffold(tmp_path)
    destination = tmp_path / "affine-raw"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("create_native_raw_variant.py")),
            str(source),
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        "status": "created",
        "workflow_id": "affine-raw",
        "destination": str(destination.resolve()),
        "model_project_bytes": "preserved",
    }


def test_custom_adapter_is_not_silently_rewritten_to_tensor(tmp_path):
    source = tmp_path / "affine-cpp"
    cli("init", source, "--runtime", "cpp", "--template", "cpp")
    destination = tmp_path / "affine-raw"
    with pytest.raises(ValueError, match="tensor.v1"):
        raw.create_raw_variant(source, destination)
    assert not destination.exists()


def test_explicit_runtime_executor_follows_derived_workflow(tmp_path):
    source = scaffold(tmp_path)
    profile = yaml.safe_load((source / "release.yaml").read_text())
    profile["runtime"] = {
        "executor_class": "native.affine-json",
        "request_timeout_seconds": 90,
    }
    (source / "release.yaml").write_text(yaml.safe_dump(profile))
    destination = raw.create_raw_variant(source, tmp_path / "affine-raw")
    derived = yaml.safe_load((destination / "release.yaml").read_text())
    assert derived["runtime"] == {
        "executor_class": "native.affine-raw",
        "request_timeout_seconds": 90,
    }
    assert cli("validate", destination)["status"] == "valid"
