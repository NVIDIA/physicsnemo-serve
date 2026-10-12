import copy
import hashlib
import json
import struct

import pytest

from qa.native_inference.contract import (
    compare_f32,
    expected_cases,
    validate_summary,
    verify_model,
    verify_build,
)


def summary():
    return {
        "schema_version": 1,
        "run_id": "example",
        "source_sha": "a" * 40,
        "image_digest": "registry/qa@sha256:" + "b" * 64,
        "profile": "smoke",
        "status": "passed",
        "stages": {"build": {"status": "passed"}, "consumer": {"status": "passed"}},
        "expected_cases": expected_cases("smoke"),
        "cases": [{"name": n, "status": "passed"} for n in expected_cases("smoke")],
    }


def validate(value):
    original = summary()
    validate_summary(
        value,
        **{k: original[k] for k in ("run_id", "source_sha", "image_digest", "profile")},
    )


def test_complete_summary_passes():
    validate(summary())


@pytest.mark.parametrize(
    "mutation",
    [
        "omit",
        "missing_case",
        "duplicate",
        "append_duplicate",
        "skip",
        "source_sha",
        "image_digest",
        "run_id",
        "profile",
        "schema",
        "stage_blocked",
        "stage_skipped",
        "empty",
        "unknown",
        "failed",
    ],
)
def test_incomplete_or_wrong_run_cannot_pass(mutation):
    value = copy.deepcopy(summary())
    if mutation == "omit":
        value["cases"].pop()
        value["expected_cases"].pop()
    elif mutation == "missing_case":
        value["cases"].pop()
    elif mutation == "duplicate":
        value["cases"][-1] = value["cases"][0]
    elif mutation == "append_duplicate":
        value["cases"].append(copy.deepcopy(value["cases"][0]))
    elif mutation == "skip":
        value["cases"][0]["status"] = "skipped"
    elif mutation in {"source_sha", "image_digest", "run_id", "profile"}:
        value[mutation] = "different"
    elif mutation == "schema":
        value["schema_version"] = True
    elif mutation.startswith("stage_"):
        value["stages"]["consumer"]["status"] = mutation.removeprefix("stage_")
    elif mutation == "empty":
        value["cases"] = value["expected_cases"] = []
    elif mutation == "unknown":
        value["cases"].append({"name": "extra", "status": "passed"})
    else:
        value["status"] = "failed"
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize(
    "values,reference",
    [([float("nan")], [0.0]), ([float("inf")], [0.0]), ([2.0], [1.0]), ([1e-5], [0.0])],
)
def test_parity_rejects_nonfinite_and_either_error_limit(values, reference):
    with pytest.raises(ValueError):
        compare_f32(struct.pack(f"<{len(values)}f", *values), reference)


def test_parity_rejects_wrong_byte_count():
    with pytest.raises(ValueError):
        compare_f32(b"\0", [0.0])


def test_parity_records_actual_errors():
    metrics = compare_f32(struct.pack("<f", 1.00001), [1.0])
    assert 0 < metrics["max_abs"] < 1e-4
    assert metrics["relative_l2"] == metrics["max_abs"]


def model_fixture(tmp_path):
    model = tmp_path / "model"
    package = model / "backends/aoti"
    package.mkdir(parents=True)
    (package / "model.pt2").write_bytes(b"compiled")
    (package / "model.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "inputs": [{"name": "input", "shape": [4], "dtype": "float32"}],
                "outputs": [{"name": "output", "shape": [4], "dtype": "float32"}],
                "artifacts": [
                    {"backend": "aoti", "target": "cuda", "path": "model.pt2"}
                ],
            }
        )
    )
    files = [
        {
            "path": p.relative_to(model).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in package.iterdir()
    ]
    (model / "model-release.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "variants": {"aoti": {"package": "backends/aoti", "files": files}},
            }
        )
    )
    return model, package


def test_model_inventory_is_verified_and_paths_resolved(tmp_path):
    model, package = model_fixture(tmp_path)
    assert verify_model(model, ["aoti"]) == {"aoti": str(package)}


def test_corrupt_package_cannot_be_handed_off(tmp_path):
    model, package = model_fixture(tmp_path)
    (package / "model.pt2").write_bytes(b"modified")
    with pytest.raises(ValueError):
        verify_model(model, ["aoti"])


def test_missing_requested_backend_cannot_be_handed_off(tmp_path):
    model, _ = model_fixture(tmp_path)
    with pytest.raises(ValueError):
        verify_model(model, ["aoti", "tensorrt"])


def build_fixture(tmp_path, *, actual=0.0, omit_metadata=False):
    model, _ = model_fixture(tmp_path)
    cases = []
    for index in range(3):
        directory = tmp_path / "checks/aoti" / f"case-{index}"
        directory.mkdir(parents=True)
        native, reference = struct.pack("<4f", *([actual] * 4)), bytes(16)
        (directory / "output-0.bin").write_bytes(native)
        (directory / "reference-0.bin").write_bytes(reference)
        metadata = {
            "completed": True,
            "backend": "aoti",
            "execution_device": {"type": "cuda", "index": 0},
        }
        if not omit_metadata:
            metadata["outputs"] = [
                {
                    "name": "output",
                    "dtype": "float32",
                    "shape": [4],
                    "byte_size": 16,
                    "device": {"type": "cpu", "index": 0},
                }
            ]
        cases.append(
            {
                "passed": True,
                "metadata": metadata,
                "outputs": [
                    {
                        "actual_sha256": hashlib.sha256(native).hexdigest(),
                        "reference_sha256": hashlib.sha256(reference).hexdigest(),
                    }
                ],
            }
        )
    checks = tmp_path / "checks/aoti.json"
    checks.write_text(json.dumps({"passed": True, "backend": "aoti", "cases": cases}))

    def identity(path):
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    (tmp_path / "build.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "case_count": 3,
                "release": identity(model / "model-release.json"),
                "variants": {
                    "aoti": {"status": "complete", "checks": identity(checks)}
                },
            }
        )
    )


@pytest.mark.parametrize("actual,omit_metadata", [(999.0, False), (0.0, True)])
def test_build_pass_flags_do_not_replace_numerical_and_metadata_checks(
    tmp_path, actual, omit_metadata
):
    build_fixture(tmp_path, actual=actual, omit_metadata=omit_metadata)
    with pytest.raises(ValueError):
        verify_build(tmp_path, ["aoti"])


def test_build_verifies_complete_matching_evidence(tmp_path):
    build_fixture(tmp_path)
    assert set(verify_build(tmp_path, ["aoti"])) == {"aoti"}


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
@pytest.mark.parametrize("operation", ["surface75", "surface161", "missing_tail"])
def test_full_summary_requires_every_transolver_backend_case(backend, operation):
    required = expected_cases("full")
    name = f"transolver.{backend}.{operation}"
    assert len(required) == 33
    assert name in required
    value = summary()
    value["profile"] = "full"
    value["expected_cases"] = required
    value["cases"] = [
        {"name": case, "status": "passed"} for case in required if case != name
    ]
    with pytest.raises(ValueError, match="executed case inventory"):
        validate_summary(
            value,
            **{
                key: value[key]
                for key in ("run_id", "source_sha", "image_digest", "profile")
            },
        )


def test_build_rejects_aoti_only_receipt_when_both_backends_requested(tmp_path):
    build_fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete backend/case coverage"):
        verify_build(tmp_path, ["aoti", "tensorrt"])
