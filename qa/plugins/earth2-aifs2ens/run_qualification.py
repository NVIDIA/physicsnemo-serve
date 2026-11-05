# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run frozen-reference calibration and the complete AIFS2ENS QA suite in one job."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_parity import POLICY, RNG_POLICY, calibrate_ensemble, validate_holdouts  # noqa: E402

LAYOUTS = {
    "full": {"nensemble": 2, "batch_size": 2, "nsteps": 4, "seed_base": 0},
    "serial": {"nensemble": 2, "batch_size": 1, "nsteps": 4, "seed_base": 0},
    "uneven": {"nensemble": 3, "batch_size": 2, "nsteps": 2, "seed_base": 0},
}
H100_CASES = frozenset(
    {
        "test_reference_repeatability",
        "test_full_ensemble_four_steps[python]",
        "test_full_ensemble_four_steps[rust]",
        "test_serial_batches_shared_rng",
        "test_uneven_final_batch",
        "test_cached_ensemble_A_B_A",
        "test_rest_single_gpu_no_fanout",
    }
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_source(expected):
    manifest = json.loads((REPO / "qa-source-inventory.json").read_text())
    encoded = json.dumps(
        manifest["files"], sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("image source inventory differs from the requested snapshot")
    for item in manifest["files"]:
        if digest(REPO / item["path"]) != item["sha256"]:
            raise ValueError(f"image source bytes differ: {item['path']}")
    return manifest["base_commit"]


def verify_reference_pair(first, second, expected, *, nsteps):
    identity = {
        **expected,
        "status": "passed",
        "numerical_policy": POLICY,
        "rng_policy": RNG_POLICY,
        "model_seed": None,
        "noise_projection_device": "cuda",
        "model_adapters": [],
    }
    reports = []
    for root in (first, second):
        report = json.loads((root / "report.json").read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"reference provenance or numerical policy differs: {root}"
            )
        if len(report.get("steps", [])) != nsteps:
            raise ValueError("reference forecast coverage is incomplete")
        for step, record in enumerate(report["steps"], start=1):
            filename = f"step-{step:04d}.npy"
            if record.get("step") != step or record.get("path") != filename:
                raise ValueError("reference step identity differs")
            if digest(root / filename) != record.get("sha256"):
                raise ValueError("reference forecast bytes differ from the report")
        reports.append(report)
    flags = (
        "torch",
        "cuda",
        "flash_attn",
        "anemoi_models",
        "deterministic_warn_only",
        "gpu_name",
        "deterministic_algorithms",
        "allow_bf16_reduced_precision_reduction",
        "allow_tf32_matmul",
        "allow_tf32_cudnn",
        "cublas_workspace_config",
    )
    if any(
        key not in reports[0] or reports[0][key] != reports[1].get(key) for key in flags
    ):
        raise ValueError("native reference runtime or math settings differ")
    return {
        "original": str(first),
        "repeat": str(second),
        "bitwise_identical": reports[0]["steps"] == reports[1]["steps"],
        "steps": reports[0]["steps"],
        "repeat_steps": reports[1]["steps"],
    }


def wait_for_parent_release(
    read_entries, run_id, *, timeout=30, monotonic=None, sleep=None
):
    """Completion can precede the worker's release; reject child/duplicate releases."""
    import time

    monotonic = monotonic or time.monotonic
    sleep = sleep or time.sleep
    deadline = monotonic() + timeout
    while True:
        entries = list(read_entries())
        assert all(entry.get("run_id") == run_id for _, entry in entries), (
            "unexpected GPU allocation release (child or other run)",
            entries,
        )
        assert len(entries) <= 1, ("duplicate GPU allocation release", entries)
        if entries:
            return entries
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("completed ensemble has no GPU allocation release")
        sleep(min(0.1, remaining))


def wait_for_structured_result(read_result, *, timeout=60, monotonic=None, sleep=None):
    """A succeeded run can precede persistence of its structured final result."""
    import time

    monotonic = monotonic or time.monotonic
    sleep = sleep or time.sleep
    deadline = monotonic() + timeout
    while True:
        result = read_result()
        if isinstance(result, dict) and all(
            isinstance(result.get(section), dict)
            for section in ("request", "execution", "payload")
        ):
            return result
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("completed ensemble has no structured result")
        sleep(min(0.1, remaining))


def verify_candidate_runtime(candidate, reference):
    """Compare the executed worker's runtime, including REST subprocess settings."""
    keys = (
        "torch",
        "cuda",
        "flash_attn",
        "anemoi_models",
        "allow_tf32_matmul",
        "allow_tf32_cudnn",
        "allow_bf16_reduced_precision_reduction",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cublas_workspace_config",
        "gpu_name",
        "model_seed",
    )
    assert isinstance(candidate, dict), "actual worker runtime record is missing"
    assert candidate.get("model_seed") is None, "worker model_seed must be None"
    for name in keys:
        assert name in candidate and name in reference, (
            "worker/reference runtime key missing",
            name,
        )
        assert candidate[name] == reference[name], (
            "actual worker runtime differs",
            name,
            candidate[name],
            reference[name],
        )
    assert candidate["anemoi_models"] == "0.11.2", "worker anemoi runtime is unpinned"


def verify_junit(path):
    cases = list(ET.parse(path).iter("testcase"))
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError("QA contains failed, skipped, or absent test cases")
    gpu = [
        case.get("name")
        for case in cases
        if "test_h100_parity" in case.get("classname", "")
    ]
    if len(gpu) != len(H100_CASES) or set(gpu) != H100_CASES:
        raise ValueError("QA did not execute the complete H100 case inventory")
    return {"passed": len(cases), "h100_passed": len(gpu), "skipped": 0}


def merge_junit(sources, destination):
    combined = ET.Element("testsuites")
    for source in sources:
        root = ET.parse(source).getroot()
        for suite in [root] if root.tag == "testsuite" else root.findall("testsuite"):
            combined.append(suite)
    ET.ElementTree(combined).write(destination, encoding="utf-8", xml_declaration=True)


def checked(argv, env, log):
    write_json(log.with_suffix(".command.json"), {"argv": argv})
    with log.open("w") as stream:
        subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=14400,
            check=True,
        )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix="aifs2ens-single-gpu-qa-"))
    summary = {
        "status": "failed",
        "image": args.image,
        "source_content_sha256": args.source_content_sha256,
        "numerical_policy": POLICY,
        "rng_policy": RNG_POLICY,
        "work": str(work),
        "execution_mode": "single-gpu-complete-ensemble",
        "raw_array_retention": "ephemeral forecast scratch; frozen input and reports persist",
        "excluded_other_runtime_cases": [
            "test_spherical_gaussian_skip_ahead_matches_sequential_gpu_batches",
            "test_spherical_gaussian_skip_ahead_matches_fcn_input_shape_on_gpu",
        ],
    }
    try:
        summary["base_commit"] = verify_source(args.source_content_sha256)
        original = args.assets.resolve()
        fixture = args.output / "fixture"
        env = {
            **os.environ,
            "AIFS2ENS_QUALIFICATION_ROOT": str(fixture),
            "CPATH": "/usr/local/cuda/include",
            "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
            "EARTH2STUDIO_MODEL_CACHE": str(original / "cache/models"),
            "EARTH2STUDIO_DATA_CACHE": str(original / "cache/weather"),
            "TRITON_CACHE_DIR": str(original / "cache/triton"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        summary["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        print(summary["gpu"], flush=True)
        print(
            "Capturing one common unmodified Earth2Studio input independently",
            flush=True,
        )
        checked(
            [
                sys.executable,
                str(Path(__file__).with_name("capture_earth2studio_inputs.py")),
                "--original-assets",
                str(original),
                "--output",
                str(fixture),
            ],
            env,
            args.output / "native-input-capture.log",
        )
        (fixture / "cache").symlink_to(original / "cache")
        qualification = fixture / "qualification"
        qualification.mkdir()
        contract = json.loads((fixture / "qualified-input-contract.json").read_text())
        capture = json.loads((fixture / "weather-capture-report.json").read_text())
        if (
            capture.get("status") != "passed"
            or capture.get("preprocessing_policy")
            != "earth2studio-latlon-frozen-invariants-v1"
            or capture.get("shared_input") is not True
        ):
            raise ValueError(
                "independent common Earth2Studio input capture is incomplete"
            )
        if digest(fixture / "assets/input.npy") != capture["input"]["sha256"]:
            raise ValueError("frozen common input bytes differ")
        if digest(fixture / "assets/invariants.npy") != contract["invariants_sha256"]:
            raise ValueError("frozen invariant bytes differ")
        summary["input_capture"] = capture
        script = Path(__file__).with_name("reference_earth2studio.py")
        paths = {name: [] for name in LAYOUTS}
        integrity = []
        # Prespecified reference-only protocol: 4 calibration + 2 holdouts;
        # expand to 8 + 2 new holdouts before any candidate if required.
        calibrated = None
        for count in (4, 8):
            layouts = {}
            for name, layout in LAYOUTS.items():
                expected = {
                    **layout,
                    "input_sha256": capture["input"]["sha256"],
                    "checkpoint_sha256": contract["checkpoint_sha256"],
                    "reference_script_sha256": digest(script),
                    "output_variables": contract["output_variables"],
                }
                while len(paths[name]) < count + 2:
                    repeat = len(paths[name]) + 1
                    target = work / f"reference-{name}-{repeat}"
                    print(
                        f"Stock single-GPU reference {name}, process {repeat}: {layout}",
                        flush=True,
                    )
                    checked(
                        [
                            sys.executable,
                            str(script),
                            "--checkpoint-dir",
                            str(fixture / "assets/checkpoint-xet"),
                            "--input",
                            str(fixture / "assets/input.npy"),
                            "--input-sha256",
                            capture["input"]["sha256"],
                            "--contract",
                            str(fixture / "qualified-input-contract.json"),
                            "--start-time",
                            capture["initialization"],
                            "--seed",
                            str(layout["seed_base"]),
                            "--nsteps",
                            str(layout["nsteps"]),
                            "--nensemble",
                            str(layout["nensemble"]),
                            "--batch-size",
                            str(layout["batch_size"]),
                            "--output",
                            str(target),
                        ],
                        env,
                        args.output / f"reference-{name}-{repeat}.log",
                    )
                    paths[name].append(target)
                    integrity.append(
                        {
                            "layout": name,
                            **verify_reference_pair(
                                paths[name][0],
                                target,
                                expected,
                                nsteps=layout["nsteps"],
                            ),
                        }
                    )
                print(
                    f"Calibrating complete ensemble {name} from {count} reference processes",
                    flush=True,
                )
                limits = calibrate_ensemble(paths[name][:count])
                holdout = validate_holdouts(
                    paths[name][0], paths[name][count : count + 2], limits
                )
                layouts[name] = {**limits, "holdout": holdout}
            calibrated = {
                "status": "passed"
                if all(row["holdout"]["status"] == "passed" for row in layouts.values())
                else "failed",
                "numerical_policy": POLICY,
                "rng_policy": RNG_POLICY,
                "calibration_runs_per_layout": count,
                "holdout_runs_per_layout": 2,
                "margin": 2.0,
                "frozen_before_candidate": True,
                "layouts": layouts,
                "reference_integrity": integrity,
            }
            write_json(qualification / f"native-calibration-{count}.json", calibrated)
            if calibrated["status"] == "passed":
                break
        if calibrated["status"] != "passed":
            raise ValueError(
                "complete-ensemble reference holdouts exceed limits; no candidate evaluated"
            )
        write_json(qualification / "calibration.json", calibrated)
        write_json(
            qualification / "references.json",
            {name: {"path": str(paths[name][0])} for name in LAYOUTS},
        )
        env["AIFS2ENS_PARITY_CALIBRATION_SHA256"] = digest(
            qualification / "calibration.json"
        )
        summary.update(
            calibration_sha256=env["AIFS2ENS_PARITY_CALIBRATION_SHA256"],
            reference_repeatability="complete_ensemble_variability_measured_and_holdouts_validated",
            calibration_runs_per_layout=calibrated["calibration_runs_per_layout"],
            holdout_runs_per_layout=2,
            reference_layouts=LAYOUTS,
            comparison_policy="max_abs_and_rmse_within_frozen_whole_ensemble_reference_envelope",
        )
        print(
            "All complete-ensemble holdouts passed; candidate limits now frozen",
            flush=True,
        )
        tests = sorted(
            str(path) for path in (REPO / "tests").glob("test_aifs2ens_*.py")
        )
        tests.extend(
            str(REPO / "tests" / name)
            for name in (
                "test_earth2_plugins.py",
                "test_plugin_dev.py",
                "test_runtime_env_launcher.py",
            )
        )
        print(
            "Running portable AIFS2ENS and shared plugin/runtime regressions",
            flush=True,
        )
        checked(
            [
                os.environ["AIFS2ENS_PORTABLE_PYTHON"],
                "-m",
                "pytest",
                "-q",
                "-p",
                "pytest_asyncio.plugin",
                "-k",
                "not spherical_gaussian_skip_ahead_matches",
                *tests,
                "--basetemp",
                str(work / "portable-pytest"),
                "--junitxml",
                str(args.output / "portable-junit.xml"),
            ],
            env,
            args.output / "portable-pytest.log",
        )
        print(f"Running all {len(H100_CASES)} single-GPU H100 cases", flush=True)
        checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(Path(__file__).with_name("test_h100_parity.py")),
                "--basetemp",
                str(work / "pytest"),
                "--junitxml",
                str(args.output / "gpu-junit.xml"),
            ],
            env,
            args.output / "gpu-pytest.log",
        )
        merge_junit(
            [args.output / "portable-junit.xml", args.output / "gpu-junit.xml"],
            args.output / "junit.xml",
        )
        summary.update(verify_junit(args.output / "junit.xml"), status="passed")
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        for pattern in (
            "report.json",
            "parity-metrics.json",
            "forecast_metadata.json",
            "service-result.json",
            "*.log",
        ):
            for source in work.rglob(pattern):
                target = args.output / "reports" / source.relative_to(work)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        write_json(args.output / "summary.json", summary)
        print(
            "AIFS2ENS_QA_SUMMARY_BEGIN\n"
            + json.dumps(summary)
            + "\nAIFS2ENS_QA_SUMMARY_END",
            flush=True,
        )
    return 0 if summary["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-content-sha256", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
