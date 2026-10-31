from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import numpy as np

from qa.native_inference import cfd_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "qa/inference/cfd_parity_surface_run1_transolver.json"


def profile():
    return json.loads(PROFILE.read_text())


@pytest.fixture(autouse=True)
def small_synthetic_field_fixture(monkeypatch):
    # Keep contract tests tiny; the checked-in profile and GPU capture retain
    # the real 8,828,095-cell case and cannot opt into this test-only substitute.
    monkeypatch.setitem(contract.CASE_POINT_COUNTS, "run_1", 60011)


def test_profile_is_separate_transolver_only_and_retains_domino_default():
    contract.validate_reference_profile(profile())
    default = json.loads(
        (ROOT / "qa/inference/cfd_parity_surface_run1.json").read_text()
    )
    assert default["request"]["models"] == ["domino_surface"]
    request = json.loads((ROOT / profile()["rest"]["request_path"]).read_text())
    assert request["models"] == ["transolver_surface"]
    assert request["metrics"] == [
        "l2_pressure",
        "l2_shear_stress",
        "l2_pressure_area_weighted",
        "drag",
        "lift",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["request"].update(models=["domino_surface"]),
        lambda p: p["config"]["models"][0].update(
            package="hf://nvidia/transolver_drivaerml@wrong"
        ),
        lambda p: p["native_reference"]["statistics"].update(sha256="0" * 64),
        lambda p: p["native_reference"].update(case_ids=[]),
        lambda p: p["config"]["models"][0]["kwargs"].update(batch_resolution=2048),
        lambda p: p["native_reference"].update(resolution=75),
        lambda p: p["config"]["run"].update(metrics_cache={"enabled": True}),
    ],
)
def test_rejects_changed_reference_semantics(mutation):
    value = copy.deepcopy(profile())
    mutation(value)
    with pytest.raises(contract.ParityContractError):
        contract.validate_reference_profile(value)


@pytest.mark.parametrize(
    "seed,case,expected",
    [
        (0, "run_1", 2260079506302488169),
        (42, "run_1", 1615895419752914611),
        (42, "run_11", 3962535327529237276),
    ],
)
def test_provider_seed_vectors(seed, case, expected):
    assert contract.inference_seed(seed, case) == expected


@pytest.mark.parametrize(
    "points,expected",
    [
        (59999, [59999]),
        (60000, [60000]),
        (60001, [60000, 1]),
        (60011, [60000, 11]),
        (120000, [60000, 60000]),
        (120011, [60000, 60000, 11]),
    ],
)
def test_provider_block_plan_preserves_exact_tail(points, expected):
    # Oracle records the provider plan even if a native release rejects singleton tails.
    assert contract.block_plan(points) == expected


def make_evidence(root: Path, *, runtime="python-provider", raw=False):
    root.mkdir()
    count = 60011
    values = {
        "fx": np.full((1, count, 2), 2.0, dtype="<f4"),
        "embedding": np.full((1, count, 6), 3.0, dtype="<f4"),
        "standardized": np.full((count, 4), 4.0, dtype="<f4"),
        "physical": np.full((count, 4), 5.0, dtype="<f4"),
        "permutation": np.arange(count - 1, -1, -1, dtype="<i8"),
        "source_cell_ids": np.arange(count, dtype="<i8"),
    }
    arrays = {}
    for name, value in values.items():
        path = root / f"{name}.{'bin' if raw else 'npy'}"
        value.tofile(path) if raw else np.save(path, value, allow_pickle=False)
        arrays[name] = contract.array_descriptor(
            path, root, value, encoding="raw" if raw else "npy"
        )
    manifest = {
        "schema_version": 1,
        "profile_id": profile()["profile_id"],
        "runtime": runtime,
        "model": "transolver_surface",
        "model_revision": contract.MODEL_REVISION,
        "provider" if runtime == "python-provider" else "reference_provider": profile()[
            "provider"
        ],
        "source_sha": "a" * 40,
        "image": "registry.test/reference@sha256:" + "b" * 64,
        **copy.deepcopy(contract.ASSET_PINS),
        "environment": {
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "torch": "2.10.0",
            "cuda": "13.1",
            "vtk": "9.5.2",
            "math": {key: False for key in contract.MATH_KEYS},
        },
        "cases": [
            {
                "case_id": "run_1",
                "point_count": count,
                "extent": "all-source-cells",
                "run_seed": 42,
                "inference_seed": contract.inference_seed(42, "run_1"),
                "block_points": [60000, 11],
                "batch_resolution": 60000,
                "output_location": "cell",
                "source_order": True,
                "inputs": copy.deepcopy(contract.CASE_PINS["run_1"]),
                "arrays": arrays,
            }
        ],
    }
    path = root / "fields.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_npy_reference_and_raw_native_arrays_compare(tmp_path):
    ref, _ = make_evidence(tmp_path / "reference")
    candidate, _ = make_evidence(tmp_path / "native", runtime="native-aoti", raw=True)
    result = contract.compare_evidence(profile(), ref, candidate)
    assert result["passed"]
    assert result["scientific_reports_compared"] is False
    assert len(result["fields"]) == 4


def test_field_mismatch_is_reported_and_does_not_claim_metric_parity(tmp_path):
    ref, _ = make_evidence(tmp_path / "reference")
    candidate, manifest = make_evidence(
        tmp_path / "native", runtime="native-aoti", raw=True
    )
    descriptor = manifest["cases"][0]["arrays"]["physical"]
    path = candidate.parent / descriptor["path"]
    value = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    value[7, 2] += 1
    value.tofile(path)
    descriptor["sha256"] = contract.sha256_file(path)
    candidate.write_text(json.dumps(manifest))
    result = contract.compare_evidence(profile(), ref, candidate)
    assert result["passed"] is False
    assert (
        next(field for field in result["fields"] if field["array"] == "physical")[
            "mismatches"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.update(model="domino_surface"), "model mismatch"),
        (lambda m: m["statistics"].update(sha256="0" * 64), "statistics.sha256"),
        (lambda m: m.update(cases=[]), "case inventory"),
        (lambda m: m["cases"][0].update(block_points=[60011]), "block_points"),
        (lambda m: m["cases"][0].update(inference_seed=42), "inference_seed"),
        (lambda m: m["cases"][0].update(extent="first-60011"), "inference extent"),
        (lambda m: m["cases"][0]["arrays"].pop("permutation"), "array inventory"),
        (lambda m: m["cases"][0]["arrays"]["physical"].update(dtype="<f8"), "dtype"),
        (
            lambda m: m["cases"][0]["arrays"]["physical"].update(shape=[60011, 3]),
            "shape",
        ),
        (
            lambda m: m["cases"][0]["arrays"]["physical"].update(path="../outside"),
            "unsafe artifact",
        ),
        (lambda m: m.update(image="registry.test/mutable:latest"), "immutable"),
        (
            lambda m: m["environment"]["math"].pop("allow_tf32_matmul"),
            "math.allow_tf32",
        ),
    ],
)
def test_rejects_unqualified_or_incomplete_field_evidence(tmp_path, mutation, match):
    _, manifest = make_evidence(tmp_path / "reference")
    mutation(manifest)
    with pytest.raises(contract.FieldContractError, match=match):
        contract.validate_evidence(profile(), manifest, tmp_path / "reference")


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("physical", np.nan, "nonfinite"),
        ("permutation", 0, "permutation bijection"),
        ("source_cell_ids", 7, "source cell identities"),
    ],
)
def test_rejects_invalid_array_values_even_when_digest_is_updated(
    tmp_path, name, value, error
):
    path, manifest = make_evidence(tmp_path / "reference", raw=True)
    descriptor = manifest["cases"][0]["arrays"][name]
    array_path = path.parent / descriptor["path"]
    array = np.fromfile(array_path, dtype=descriptor["dtype"])
    array[0] = value
    array.tofile(array_path)
    descriptor["sha256"] = contract.sha256_file(array_path)
    with pytest.raises(contract.FieldContractError, match=error):
        contract.validate_evidence(profile(), manifest, path.parent)


def test_tampered_file_and_different_math_flags_fail_closed(tmp_path):
    ref, _ = make_evidence(tmp_path / "reference")
    candidate, manifest = make_evidence(tmp_path / "native", runtime="native-aoti")
    manifest["environment"]["math"]["allow_tf32_matmul"] = True
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(contract.FieldContractError, match="math settings"):
        contract.compare_evidence(profile(), ref, candidate)
    manifest["environment"]["math"]["allow_tf32_matmul"] = False
    candidate.write_text(json.dumps(manifest))
    array_path = candidate.parent / manifest["cases"][0]["arrays"]["physical"]["path"]
    with array_path.open("r+b") as output:
        output.seek(-1, 2)
        output.write(b"x")
    with pytest.raises(contract.FieldContractError, match="SHA-256"):
        contract.compare_evidence(profile(), ref, candidate)


def test_capture_config_uses_full_case_and_cannot_skip_inference(tmp_path):
    from qa.native_inference.cfd_reference import build_capture_config

    request = json.loads((ROOT / profile()["rest"]["request_path"]).read_text())
    config = build_capture_config(
        profile(), request, tmp_path / "inputs", tmp_path / "outputs"
    )
    assert config["run"]["metrics_cache"] == {"enabled": False}
    assert config["benchmark"]["models"][0]["kwargs"]["batch_resolution"] == 60000
    assert config["benchmark"]["datasets"][0]["case_ids"] == ["run_1"]
    assert config["benchmark"]["models"][0]["package"] == contract.PACKAGE


@pytest.mark.parametrize(
    "change",
    [
        lambda request: request.update(cases=[]),
        lambda request: request["cases"][0].update(sha256="f" * 64),
    ],
)
def test_reference_request_cannot_substitute_case_inputs(change):
    request = json.loads((ROOT / profile()["rest"]["request_path"]).read_text())
    change(request)
    with pytest.raises(contract.ParityContractError):
        contract.validate_reference_request(profile(), request)


def report():
    # Actual pinned provider report shape, independently specified rather than
    # generated from the validator's own metric mapping.
    metrics = {
        "l2_pressure": 0.1,
        "l2_shear_stress_wall_shear_stress_true_x_l2_error": 0.2,
        "l2_shear_stress_wall_shear_stress_true_y_l2_error": 0.3,
        "l2_shear_stress_wall_shear_stress_true_z_l2_error": 0.4,
        "l2_pressure_area_weighted": 0.1,
        "drag_error": 0.01,
        "drag_true": 100.0,
        "drag_pred": 101.0,
        "lift_error": 0.02,
        "lift_true": 50.0,
        "lift_pred": 51.0,
    }
    return [
        {
            "model": "transolver_surface",
            "dataset": "drivaerml",
            "cases": ["run_1"],
            "metrics": metrics,
            "per_case": [
                {
                    "case_id": "run_1",
                    "metrics": copy.deepcopy(metrics),
                    "metric_dtype": "cell",
                }
            ],
        }
    ]


def test_scientific_report_requires_all_requested_metrics():
    value = report()
    contract.validate_report(profile(), value)
    value[0]["per_case"][0]["metrics"].pop("lift_pred")
    with pytest.raises(contract.FieldContractError, match="metric inventory"):
        contract.validate_report(profile(), value)


def test_native_compiled_payload_hash_is_bound_to_receipt(tmp_path):
    from qa.native_inference.cfd_native import verify_package_inventory
    from qa.scripts.run_native_cfd_job import package_files

    (tmp_path / "model.json").write_text('{"artifact":"model.pt2"}')
    (tmp_path / "model.pt2").write_bytes(b"qualified compiled model")
    receipt = {"files": package_files(tmp_path)}
    verify_package_inventory(tmp_path, receipt)
    (tmp_path / "model.pt2").write_bytes(b"different compiled model")
    with pytest.raises(contract.FieldContractError):
        verify_package_inventory(tmp_path, receipt)


def test_native_job_stages_only_aoti_and_changes_static_point_count(tmp_path):
    from qa.scripts.run_native_cfd_job import stage_project

    template = ROOT / "native-inference/examples/transolver-surface"
    destination = tmp_path / "tail"
    stage_project(template, destination, 8095)
    recipe = json.loads((destination / "model-build.json").read_text())
    assert recipe["backends"] == ["aoti"]
    assert recipe["assets"] == {}
    assert "tensorrt_profile" not in recipe
    assert "POINT_COUNT = 8095" in (destination / "adapter.py").read_text()


def test_gpu_job_records_container_limits_separately_from_host_memory(tmp_path):
    from qa.scripts.run_native_cfd_job import resource_envelope

    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    proc.mkdir()
    cgroup.mkdir()
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 600 kB\n")
    (cgroup / "memory.max").write_text("4096\n")
    (cgroup / "memory.current").write_text("2048\n")
    observed = resource_envelope(proc_root=proc, cgroup_root=cgroup, shm_path=tmp_path)
    assert observed["host_memory_total_bytes"] == 1000 * 1024
    assert observed["host_memory_available_bytes"] == 600 * 1024
    assert observed["cgroup_memory_limit_bytes"] == 4096
    assert observed["cgroup_memory_current_bytes"] == 2048
    assert observed["cgroup_version"] == "v2"
    assert observed["shm_total_bytes"] >= observed["shm_free_bytes"] >= 0


def test_reusing_compiled_packages_requires_explicit_unchanged_compilation_identity():
    from qa.native_inference.cfd_native import validate_compilation_binding

    receipt = {"source_sha": "a" * 40}
    with pytest.raises(contract.FieldContractError):
        validate_compilation_binding(receipt, source_sha="b" * 40)
    assert (
        validate_compilation_binding(
            receipt, source_sha="b" * 40, compilation_source_sha="a" * 40
        )
        == "a" * 40
    )
    with pytest.raises(contract.FieldContractError):
        validate_compilation_binding(
            receipt, source_sha="b" * 40, compilation_source_sha="c" * 40
        )


def test_builder_runtime_argument_is_resolved_before_changing_to_job_output(
    monkeypatch, tmp_path
):
    from qa.scripts.run_native_cfd_job import resolve_runtime_executable

    sdk = tmp_path / "sdk-bin"
    sdk.mkdir()
    runtime = sdk / "physicsnemo-infer"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    output = tmp_path / "job-output"
    output.mkdir()
    monkeypatch.setenv("PATH", str(sdk))
    monkeypatch.chdir(output)
    assert resolve_runtime_executable("physicsnemo-infer") == str(runtime)
    assert resolve_runtime_executable(str(runtime)) == str(runtime)
    with pytest.raises(contract.FieldContractError):
        resolve_runtime_executable("missing-runtime")


def test_native_field_receipt_preserves_actual_cached_compilation_sources():
    from qa.native_inference.cfd_native import validate_compilation_binding

    receipt = {
        "source_sha": "a" * 40,
        "qualification_source_sha": "a" * 40,
        "model_build_sources": {"full": "b" * 40, "tail": "b" * 40},
    }
    assert validate_compilation_binding(receipt, source_sha="a" * 40) == "b" * 40
    receipt["model_build_sources"]["tail"] = "c" * 40
    assert validate_compilation_binding(receipt, source_sha="a" * 40) is None


def test_tensorrt_native_evidence_keeps_the_same_scientific_contract(tmp_path):
    reference, _ = make_evidence(tmp_path / "python")
    candidate, evidence = make_evidence(tmp_path / "native", runtime="native-tensorrt")
    assert contract.validate_evidence(profile(), evidence, candidate.parent)
    report = contract.compare_evidence(profile(), reference, candidate)
    assert report["passed"] is True


def test_native_job_stages_tensorrt_exact_plugins_and_static_point_count(tmp_path):
    from qa.scripts.run_native_cfd_job import stage_project

    template = ROOT / "native-inference/examples/transolver-surface"
    original = json.loads((template / "model-build.json").read_text())
    assert original["tensorrt_profile"] == "layout-order-exact-v2"
    plugin_root = tmp_path / "sdk/lib"
    plugin_root.mkdir(parents=True)
    for relative in original["assets"].values():
        (plugin_root / Path(relative).name).write_bytes(Path(relative).name.encode())
    destination = tmp_path / "project"
    stage_project(
        template, destination, 8095, backend="tensorrt", plugin_root=plugin_root
    )
    recipe = json.loads((destination / "model-build.json").read_text())
    assert recipe["backends"] == ["tensorrt"]
    assert recipe["tensorrt_profile"] == "provider-default-exact-v1"
    assert recipe["assets"] == original["assets"]
    assert "POINT_COUNT = 8095" in (destination / "adapter.py").read_text()
    for relative in recipe["assets"].values():
        assert (destination / relative).read_bytes() == (
            plugin_root / Path(relative).name
        ).read_bytes()
