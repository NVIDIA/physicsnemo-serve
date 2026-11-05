# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A successful job must contain matching references and executed GPU checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aifs2ens_qualification", ROOT / "qa/plugins/earth2-aifs2ens/run_qualification.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def reference(path, *, seed=0):
    path.mkdir()
    steps = []
    for step in range(1, 5):
        target = path / f"step-{step:04d}.npy"
        target.write_bytes(f"frozen reference step {step}".encode())
        steps.append(
            {
                "step": step,
                "path": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    report = {
        "status": "passed",
        "seed_base": seed,
        "nensemble": 2,
        "batch_size": 2,
        "nsteps": 4,
        "rng_policy": "torch-seed-once-native-v1",
        "model_seed": None,
        "steps": steps,
        "numerical_policy": "earth2studio-native-v1",
        "checkpoint_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "reference_script_sha256": "c" * 64,
        "output_variables": ["t2m"],
        "noise_projection_device": "cuda",
        "model_adapters": [],
        "torch": "native-stack",
        "cuda": "native-cuda",
        "flash_attn": "native-flash",
        "anemoi_models": "0.11.2",
        "deterministic_warn_only": False,
        "gpu_name": "H100",
        "allow_tf32_cudnn": True,
        "deterministic_algorithms": False,
        "allow_bf16_reduced_precision_reduction": True,
        "allow_tf32_matmul": False,
        "cublas_workspace_config": ":4096:8",
    }
    (path / "report.json").write_text(json.dumps(report))
    return report


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "different_values",
        "tampered_file",
        "missing_step",
        "wrong_seed",
        "wrong_policy",
        "wrong_batch",
        "wrong_rng",
        "wrong_model_seed",
    ],
)
def test_reference_gate_checks_bytes_coverage_and_numerical_identity(tmp_path, failure):
    first, second = tmp_path / "first", tmp_path / "second"
    reference(first)
    report = reference(second)
    if failure in {"different_values", "tampered_file"}:
        target = second / "step-0004.npy"
        target.write_bytes(b"different values")
        if failure == "different_values":
            report["steps"][-1]["sha256"] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    elif failure == "missing_step":
        report["steps"].pop()
    elif failure == "wrong_seed":
        report["seed_base"] = 1024
    elif failure == "wrong_batch":
        report["batch_size"] = 1
    elif failure == "wrong_rng":
        report["rng_policy"] = "per-member-seed-offsets"
    elif failure == "wrong_model_seed":
        report["model_seed"] = 0
    elif failure == "wrong_policy":
        report["numerical_policy"] = "cpu-noise-projection-bf16-v1"
    (second / "report.json").write_text(json.dumps(report))
    expected = {key: value for key, value in reference_expected().items()}
    if failure and failure != "different_values":
        with pytest.raises(ValueError):
            MODULE.verify_reference_pair(first, second, expected, nsteps=4)
    else:
        result = MODULE.verify_reference_pair(first, second, expected, nsteps=4)
        assert len(result["steps"]) == 4
        assert result["bitwise_identical"] is (failure != "different_values")


def reference_expected():
    return {
        "seed_base": 0,
        "nensemble": 2,
        "batch_size": 2,
        "nsteps": 4,
        "checkpoint_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "reference_script_sha256": "c" * 64,
        "output_variables": ["t2m"],
    }


@pytest.mark.parametrize("failure", [None, "skipped", "failed", "missing", "empty"])
def test_job_gate_rejects_skipped_failed_or_absent_h100_cases(tmp_path, failure):
    suite = ET.Element("testsuite")
    names = sorted(MODULE.H100_CASES)
    if failure == "missing":
        names.pop()
    if failure == "empty":
        names = []
    for index, name in enumerate(names):
        case = ET.SubElement(suite, "testcase", classname="test_h100_parity", name=name)
        if index == 0 and failure in {"skipped", "failed"}:
            ET.SubElement(case, "skipped" if failure == "skipped" else "failure")
    path = tmp_path / "junit.xml"
    ET.ElementTree(suite).write(path)
    if failure:
        with pytest.raises(ValueError):
            MODULE.verify_junit(path)
    else:
        assert MODULE.verify_junit(path)["h100_passed"] == 7


def test_combined_evidence_includes_portable_and_gpu_environment_results(tmp_path):
    portable = tmp_path / "portable.xml"
    portable.write_text(
        '<testsuites><testsuite><testcase classname="portable" name="contract"/></testsuite></testsuites>'
    )
    gpu = tmp_path / "gpu.xml"
    suite = ET.Element("testsuite")
    for name in MODULE.H100_CASES:
        ET.SubElement(suite, "testcase", classname="test_h100_parity", name=name)
    ET.ElementTree(suite).write(gpu)
    combined = tmp_path / "combined.xml"
    MODULE.merge_junit([portable, gpu], combined)
    assert MODULE.verify_junit(combined) == {
        "passed": 8,
        "h100_passed": 7,
        "skipped": 0,
    }


def test_parent_release_waits_for_worker_after_completion_without_deduplicating():
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    clock = Clock()
    observations = iter([[], [("1-0", {"run_id": "parent"})]])
    result = MODULE.wait_for_parent_release(
        lambda: next(observations),
        "parent",
        timeout=1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result == [("1-0", {"run_id": "parent"})]
    assert clock.now > 0


@pytest.mark.parametrize(
    "run_ids", [["parent:item:0"], ["other"], ["parent", "parent"]]
)
def test_parent_release_rejects_unexpected_child_or_duplicate_entries(run_ids):
    entries = [(str(index), {"run_id": run_id}) for index, run_id in enumerate(run_ids)]
    with pytest.raises(AssertionError, match="release|allocation"):
        MODULE.wait_for_parent_release(lambda: entries, "parent", timeout=1)


def test_parent_release_times_out_when_completion_has_no_allocation_release():
    now = [0.0]

    def sleep(duration):
        now[0] += duration

    with pytest.raises(TimeoutError, match="release"):
        MODULE.wait_for_parent_release(
            lambda: [], "parent", timeout=0.2, monotonic=lambda: now[0], sleep=sleep
        )


def test_structured_result_waits_for_finalization_of_the_same_accepted_parent():
    provisional = {"status": "succeeded", "execution": {"run_id": "parent"}}
    final = {
        "request": {},
        "execution": {"run_id": "parent"},
        "payload": {"complete": True},
    }
    snapshots = json.loads(json.dumps([provisional, final]))
    responses = iter([provisional, final])
    reads = []
    now = [0.0]

    def read():
        reads.append("/v1/infer/earth2-aifs2ens/parent/results")
        return next(responses)

    def sleep(duration):
        now[0] += duration

    result = MODULE.wait_for_structured_result(
        read, timeout=1, monotonic=lambda: now[0], sleep=sleep
    )
    assert result is final and result["execution"]["run_id"] == "parent"
    assert reads == ["/v1/infer/earth2-aifs2ens/parent/results"] * 2
    assert now[0] > 0
    assert [provisional, final] == snapshots


@pytest.mark.parametrize(
    "provisional",
    [
        None,
        {},
        {"status": "succeeded", "execution": {}},
        {"request": {}, "execution": {}, "payload": None},
        {"request": [], "execution": {}, "payload": {}},
    ],
)
def test_structured_result_times_out_without_all_result_sections(provisional):
    now = [0.0]

    def sleep(duration):
        now[0] += duration

    with pytest.raises(TimeoutError, match="structured result"):
        MODULE.wait_for_structured_result(
            lambda: provisional,
            timeout=0.2,
            monotonic=lambda: now[0],
            sleep=sleep,
        )
    assert now[0] == pytest.approx(0.2)


def test_structured_result_does_not_relax_the_callers_payload_assertions():
    final = {"request": {}, "execution": {}, "payload": {"complete": False}}
    result = MODULE.wait_for_structured_result(lambda: final)
    assert result is final
    with pytest.raises(AssertionError):
        assert result["payload"]["complete"] is True


@pytest.mark.parametrize(
    "failure",
    [None, "wrong_math", "wrong_model_seed", "missing_model_seed", "wrong_anemoi"],
)
def test_candidate_runtime_is_checked_from_actual_worker_record(failure):
    reference = {
        "torch": "native-torch",
        "cuda": "native-cuda",
        "flash_attn": "native-flash",
        "anemoi_models": "0.11.2",
        "allow_tf32_matmul": True,
        "allow_tf32_cudnn": True,
        "allow_bf16_reduced_precision_reduction": True,
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "cublas_workspace_config": None,
        "gpu_name": "H100",
        "model_seed": None,
    }
    candidate = dict(reference)
    if failure == "wrong_math":
        candidate["allow_tf32_matmul"] = False
    elif failure == "wrong_model_seed":
        candidate["model_seed"] = 0
    elif failure == "missing_model_seed":
        candidate.pop("model_seed")
    elif failure == "wrong_anemoi":
        candidate["anemoi_models"] = "0.11.1"
    if failure:
        with pytest.raises(AssertionError, match="runtime|worker|model_seed"):
            MODULE.verify_candidate_runtime(candidate, reference)
    else:
        MODULE.verify_candidate_runtime(candidate, reference)
