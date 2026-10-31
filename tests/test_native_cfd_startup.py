from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from qa.native_inference import cfd_contract
from qa.native_inference import cfd_startup


def fixture(tmp_path, points=3, backend="aoti"):
    build = tmp_path / "build"
    case = build / f"checks/{backend}/case-0"
    case.mkdir(parents=True)
    values = {
        "input-0.bin": [0.25] * (points * 2),
        "input-1.bin": [0.5] * (points * 6),
        "reference-0.bin": [1.0] * (points * 4),
        "output-0.bin": [99.0] * (points * 4),
    }
    for name, numbers in values.items():
        (case / name).write_bytes(struct.pack(f"<{len(numbers)}f", *numbers))

    def digest(name):
        return hashlib.sha256((case / name).read_bytes()).hexdigest()

    check = {
        "passed": True,
        "cases": [
            {
                "passed": True,
                "inputs": [
                    {"name": "fx", "sha256": digest("input-0.bin")},
                    {"name": "embedding", "sha256": digest("input-1.bin")},
                ],
                "outputs": [{"reference_sha256": digest("reference-0.bin")}],
            }
        ],
    }
    (build / f"checks/{backend}.json").write_text(json.dumps(check))
    return build, points


def test_startup_expected_values_come_from_python_reference_not_native_output(tmp_path):
    build, points = fixture(tmp_path)
    destination = tmp_path / "startup"
    result = cfd_startup.stage({points: build}, destination)
    assert result["path"] == "startup.json"
    config = json.loads((destination / "startup.json").read_text())
    model = config["models"][0]
    assert model["expected_output"]["shape"] == [1, points, 4]
    output = destination / model["expected_output"]["path"]
    assert (
        output.read_bytes()
        == (build / "checks/aoti/case-0/reference-0.bin").read_bytes()
    )
    assert (
        output.read_bytes() != (build / "checks/aoti/case-0/output-0.bin").read_bytes()
    )
    assert len(result["files"]) == 4


def test_changed_reference_cannot_be_packaged_as_startup_truth(tmp_path):
    build, points = fixture(tmp_path)
    (build / "checks/aoti/case-0/reference-0.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="trusted.*changed"):
        cfd_startup.stage({points: build}, tmp_path / "startup")
    assert not (tmp_path / "startup").exists()


def provider_fixture(tmp_path, monkeypatch):
    """Small full+tail case using the real provenance/array validator."""
    count = 60011
    monkeypatch.setitem(cfd_contract.CASE_POINT_COUNTS, "run_1", count)
    root = tmp_path / "provider"
    root.mkdir()
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "qa/inference/cfd_parity_surface_run1_transolver.json"
        ).read_text()
    )
    source = np.arange(count, dtype="<f4")
    values = {
        "fx": np.stack((source, source + 0.25), axis=-1)[None],
        "embedding": np.stack([source + n * 0.125 for n in range(6)], axis=-1)[None],
        "standardized": np.stack(
            [(source % 17) * 0.03125 + n * 0.125 for n in range(4)], axis=-1
        ),
        "physical": np.zeros((count, 4), dtype="<f4"),
        "permutation": np.roll(np.arange(count, dtype="<i8"), 17),
        "source_cell_ids": np.arange(count, dtype="<i8"),
    }
    arrays = {}
    for name, array in values.items():
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        arrays[name] = cfd_contract.array_descriptor(path, root, array)
    math_flags = dict.fromkeys(cfd_contract.MATH_KEYS, False)
    math_flags.update(
        allow_tf32_matmul=True,
        allow_tf32_cudnn=True,
        float32_matmul_precision="high",
        preferred_blas_library="cublas",
    )
    manifest = {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "runtime": "python-provider",
        "model": "transolver_surface",
        "model_revision": cfd_contract.MODEL_REVISION,
        "provider": profile["provider"],
        "source_sha": "a" * 40,
        "image": "registry.test/provider@sha256:" + "b" * 64,
        **cfd_contract.ASSET_PINS,
        "environment": {
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "torch": "2.10.0",
            "cuda": "13.1",
            "vtk": "9.5.2",
            "math": math_flags,
        },
        "cases": [
            {
                "case_id": "run_1",
                "point_count": count,
                "extent": "all-source-cells",
                "run_seed": 42,
                "inference_seed": cfd_contract.inference_seed(42, "run_1"),
                "block_points": [60000, 11],
                "batch_resolution": 60000,
                "output_location": "cell",
                "source_order": True,
                "inputs": cfd_contract.CASE_PINS["run_1"],
                "arrays": arrays,
            }
        ],
    }
    reference = root / "fields.json"
    reference.write_text(json.dumps(manifest))
    builds = {
        points: fixture(tmp_path / str(points), points)[0] for points in (60000, 11)
    }
    return builds, profile, reference, manifest, values


def test_provider_startup_gathers_matching_inputs_and_outputs_from_full_and_tail(
    tmp_path, monkeypatch
):
    builds, profile, reference, manifest, arrays = provider_fixture(
        tmp_path, monkeypatch
    )
    destination = tmp_path / "startup"
    result = cfd_startup.stage_provider_reference(
        builds, destination, profile=profile, reference_path=reference
    )
    config = json.loads((destination / result["path"]).read_text())
    models = {model["point_count"]: model for model in config["models"]}
    for points, begin in ((60000, 0), (11, 60000)):
        model = models[points]
        indices = arrays["permutation"][begin : begin + points]
        expected = arrays["standardized"][indices][None]
        actual = np.fromfile(
            destination / model["expected_output"]["path"], dtype="<f4"
        ).reshape(1, points, 4)
        # Builder's IEEE-policy Python reference is deliberately 1.0. Native
        # candidate output is deliberately 99.0; neither is provider truth.
        np.testing.assert_array_equal(actual, expected)
        assert not np.all(actual == 1.0)
        assert not np.all(actual == 99.0)
        for name, width in (("fx", 2), ("embedding", 6)):
            tensor = model["inputs"][name]
            np.testing.assert_array_equal(
                np.fromfile(destination / tensor["path"], dtype="<f4").reshape(
                    1, points, width
                ),
                arrays[name][:, indices, :],
            )
        assert model["rtol"] == 1e-5
        assert model["atol"] == 1e-6
        assert model["reference_block_begin"] == begin
    assert config["math_flags"] == manifest["environment"]["math"]
    assert config["reference"]["sha256"] == cfd_startup.identity(reference)["sha256"]
    assert config["reference"]["runtime"] == "python-provider"
    assert config["reference"]["arrays"] == manifest["cases"][0]["arrays"]


@pytest.mark.parametrize("mutation", ["native", "changed", "missing_tail"])
def test_provider_startup_rejects_nonindependent_changed_or_incomplete_evidence(
    tmp_path, monkeypatch, mutation
):
    builds, profile, reference, manifest, _ = provider_fixture(tmp_path, monkeypatch)
    if mutation == "native":
        manifest["runtime"] = "native-aoti"
        reference.write_text(json.dumps(manifest))
    elif mutation == "changed":
        path = reference.parent / manifest["cases"][0]["arrays"]["standardized"]["path"]
        value = np.load(path, allow_pickle=False)
        value[1, 0] += 1
        np.save(path, value, allow_pickle=False)
    else:
        builds.pop(11)
    with pytest.raises(ValueError):
        cfd_startup.stage_provider_reference(
            builds,
            tmp_path / "startup",
            profile=profile,
            reference_path=reference,
        )
    assert not (tmp_path / "startup").exists()


def test_tensorrt_startup_requires_selected_backend_parity(tmp_path, monkeypatch):
    builds, profile, reference, _, _ = provider_fixture(tmp_path, monkeypatch)
    for build in builds.values():
        (build / "checks/aoti.json").rename(build / "checks/tensorrt.json")
    destination = tmp_path / "startup"
    result = cfd_startup.stage_provider_reference(
        builds,
        destination,
        profile=profile,
        reference_path=reference,
        backend="tensorrt",
    )
    assert (
        json.loads((destination / result["path"]).read_text())["reference"]["runtime"]
        == "python-provider"
    )
    for build in builds.values():
        (build / "checks/aoti.json").write_text(
            '{"passed":true,"cases":[{"passed":true}]}'
        )
        (build / "checks/tensorrt.json").write_text(
            '{"passed":false,"cases":[{"passed":false}]}'
        )
    with pytest.raises(ValueError, match="passed Builder parity"):
        cfd_startup.stage_provider_reference(
            builds,
            tmp_path / "rejected",
            profile=profile,
            reference_path=reference,
            backend="tensorrt",
        )
