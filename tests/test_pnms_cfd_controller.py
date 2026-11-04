"""Host-only controller checks; real GPU acceptance stays on Lepton."""

import base64
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

import pytest

from scripts import run_lepton_pnms_cfd_qa as controller


def test_controller_binds_exact_shipped_cfd_job():
    selected = controller.source_files()
    assert (
        controller.identity(selected.read_bytes())["sha256"] == controller.FROZEN_RUNNER
    )


@pytest.fixture
def args(tmp_path):
    return controller.build_parser().parse_args(
        [
            "--image",
            "registry/cfd@sha256:" + "f" * 64,
            "--expected-source-sha",
            "a" * 40,
            "--run-id",
            "cfd-test",
            "--expected-source-inventory-sha256",
            "b" * 64,
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--workspace-id",
            "workspace",
            "--node-group",
            "h100",
            "--pull-secret",
            "pull",
            "--nfs-mount-base",
            "/PhysicsNeMo/platform",
            "--lustre-dir",
            "qa",
        ]
    )


def observed(value, count):
    return {
        "count": count,
        "minimum": value,
        "maximum": value,
        "mean": value,
        "median": value,
    }


@pytest.fixture
def evidence(args):
    root = controller.remote_root(args)
    sources = {
        "full": controller.job.IMPORT_ORIGIN,
        "tail": controller.job.IMPORT_ORIGIN,
    }
    report = {
        "schema_version": 1,
        "qualification": "pnms-cfd-common-cli-http-session6-v1",
        "status": "passed",
        "run_id": args.run_id,
        "source_sha": args.expected_source_sha,
        "image_digest": args.image,
        "recompiled": False,
        "expected_cases": controller.job.CASES,
        "cases": [
            {"name": name, "status": "passed", "seconds": 1.0}
            for name in controller.job.CASES
        ],
        "driver": controller.identity(Path(controller.job.__file__).read_bytes()),
        "source_inventory": {
            "path": root + "/verified-source-inventory.json",
            "sha256": "b" * 64,
            "size_bytes": 100,
        },
        "imports": {},
        "builds": {},
        "sessions": [
            f"{root}/sessions/{i}-{runtime}/session.json"
            for i, runtime in enumerate(controller.job.ORDER)
        ],
    }
    for backend, image in controller.job.IMPORT_IMAGES.items():
        report["imports"][backend] = {
            "schema_version": 1,
            "backend": backend,
            "source_image": image,
            "model_build_sources": sources,
            "recompiled": False,
            "original_deployment": {
                "path": f"/opt/pnms-qa/cfd-imports/{backend}/deployment.json",
                "sha256": "c" * 64,
                "size_bytes": 100,
            },
            "relocated_deployment": {
                "path": f"{root}/imports/{backend}/deployment.json",
                "sha256": "d" * 64,
                "size_bytes": 100,
            },
        }
        report["builds"][backend] = {
            "root": f"{root}/applications/{backend}/.pnms/builds/" + "e" * 32,
            "source_sha": args.expected_source_sha,
            "model_build_sources": sources,
            "recompiled": False,
            "bundle_digest": ("1" if backend == "aoti" else "2") * 64,
        }
    comparison = {
        "schema_version": 1,
        "status": "passed",
        "boundary": controller.BOUNDARY,
        "execution_order": list(controller.job.ORDER),
        "kernel_throughput_measured": False,
        "tail_latency_estimated": False,
        "hardware": {
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": "GPU-one",
            "compute_capability": [9, 0],
        },
        "environment": {
            "python": "3.12.3",
            "torch": "2.10",
            "cuda": "13.1",
            "vtk": "9.1",
            "math": {"allow_tf32_matmul": False},
        },
        "workload": {
            "case": "run_1",
            "point_count": 8828095,
            "seed": 42,
            "block_size": 60000,
            "precision": "float32",
            "profile_sha256": controller.identity(
                (
                    controller.ROOT
                    / "qa/inference/cfd_parity_surface_run1_transolver.json"
                ).read_bytes()
            )["sha256"],
            "request_sha256": controller.identity(
                (
                    controller.ROOT
                    / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
                ).read_bytes()
            )["sha256"],
        },
        "runtime_statistics": {},
        "warm_speedup_vs_python": {"aoti": 2.0, "tensorrt": 5.0},
        "session_receipts": [],
    }
    times = {"python": 10.0, "aoti": 5.0, "tensorrt": 2.0}
    for runtime, value in times.items():
        comparison["runtime_statistics"][runtime] = {
            "cold_startup_seconds": observed(1.0, 2),
            "first_seconds": observed(value, 2),
            "warm_seconds": observed(value, 6),
        }
    proofs = []
    for i, (runtime, path) in enumerate(
        zip(controller.job.ORDER, report["sessions"], strict=True)
    ):
        receipt = {"path": path, "sha256": str(i) * 64, "size_bytes": 1000}
        comparison["session_receipts"].append(receipt)
        proof = {
            **receipt,
            "run_id": args.run_id,
            "runtime": runtime,
            "ordinal": i,
            "status": "passed",
            "boundary": controller.BOUNDARY,
            "image_digest": args.image,
            "source_sha": args.expected_source_sha,
            "model_build_sources": None if runtime == "python" else sources,
            "bundle_digest": None
            if runtime == "python"
            else report["builds"][runtime]["bundle_digest"],
            "cold_startup_seconds": 1.0,
            "numerical_libraries": {
                name: {"sha256": "f" * 64, "size_bytes": 100}
                for name in ("libtorch_cpu.so", "libtorch_cuda.so")
            },
            "requests": [
                {
                    "label": label,
                    "seconds": times[runtime],
                    "canonical_report_sha256": "a" * 64,
                    "parity": "passed",
                    "artifacts_verified": True,
                }
                for label in (
                    "prerequisite",
                    "first",
                    "warm-1",
                    "warm-2",
                    "warm-3",
                    "postrun",
                )
            ],
        }
        for key in ("hardware", "workload", "environment"):
            proof[key + "_sha256"] = controller.identity(
                controller.canonical(comparison[key])
            )["sha256"]
        proof["evaluator_environment_sha256"] = proof["environment_sha256"]
        proofs.append(proof)
    raw = bind(report, comparison)
    return report, comparison, proofs, raw


def bind(report, comparison):
    raw = (json.dumps(comparison, indent=2, sort_keys=True) + "\n").encode()
    root = str(PurePosixPath(report["sessions"][0]).parents[2])
    report["comparison"] = {
        "path": root + "/comparison.json",
        **controller.identity(raw),
    }
    return raw


def validate(report, args):
    controller.validate_summary(
        report,
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image,
        profile="cfd",
        source_inventory_sha256=args.expected_source_inventory_sha256,
        mount_target=args.mount_target,
    )


def test_passed_frozen_job_and_actual_comparison_are_accepted(evidence, args):
    report, comparison, proofs, raw = evidence
    validate(report, args)
    assert controller.validate_comparison(raw, proofs, report) == comparison


def test_qualified_runtime_reports_can_differ_within_scientific_tolerance(evidence):
    report, comparison, proofs, raw = evidence
    for proof in proofs:
        value = {"python": "a", "aoti": "b", "tensorrt": "c"}[proof["runtime"]]
        for request in proof["requests"]:
            request["canonical_report_sha256"] = value * 64
    assert controller.validate_comparison(raw, proofs, report) == comparison


def test_foreign_session_run_cannot_borrow_valid_comparison_paths(evidence):
    report, _, proofs, raw = evidence
    proofs[2]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="identity|run"):
        controller.validate_comparison(raw, proofs, report)


def test_actual_wrapper_preserves_selected_python_and_transports_exact_file_bytes(
    evidence, args, tmp_path
):
    report, comparison, proofs, _ = evidence
    for i, proof in enumerate(proofs):
        path = tmp_path / "sessions" / str(i) / "session.json"
        path.parent.mkdir(parents=True)
        row = {
            key: proof[key]
            for key in (
                "run_id",
                "image_digest",
                "source_sha",
                "runtime",
                "ordinal",
                "status",
                "boundary",
                "model_build_sources",
                "bundle_digest",
                "cold_startup_seconds",
                "numerical_libraries",
            )
        }
        for key in ("hardware", "workload", "environment"):
            row[key] = comparison[key]
        row["evaluator_environment"] = comparison["environment"]
        row.update(
            prerequisite=proof["requests"][0],
            samples=proof["requests"][1:5],
            postrun=proof["requests"][-1],
        )
        path.write_text(json.dumps(row))
        comparison["session_receipts"][i] = {
            "path": path.as_posix(),
            **controller.identity(path.read_bytes()),
        }
        report["sessions"][i] = path.as_posix()
    comparison_file = tmp_path / "comparison.json"
    raw = bind(report, comparison)
    comparison_file.write_bytes(raw)
    report["comparison"]["path"] = str(comparison_file)
    stored = tmp_path / "report.json"
    stored.write_text(json.dumps(report))
    driver = tmp_path / "job.py"
    driver.write_text(
        """import argparse,json,sys\nfrom pathlib import Path\ndef parser():\n p=argparse.ArgumentParser();p.add_argument("--python",type=Path);p.add_argument("--proof",type=Path);return p\ndef run(args):\n assert args.python.is_absolute() and args.python.is_file(), "selected Python must be its actual executable"\n assert str(args.python)==sys.executable\n result=json.loads(args.proof.read_text())\n print("PNMS_CFD_QA_SUMMARY_BEGIN");print(json.dumps(result));print("PNMS_CFD_QA_SUMMARY_END")\n return result\n"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            controller.WRAPPER,
            str(driver),
            controller.identity(driver.read_bytes())["sha256"],
            "--python",
            "python3",
            "--proof",
            str(stored),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    observed, transport = controller.extract_evidence(result.stdout)
    assert observed == report
    assert transport["raw"] == raw
    assert [proof["image_digest"] for proof in transport["sessions"]] == [
        args.image
    ] * 6


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "image",
        "run",
        "driver",
        "inventory",
        "prepare-only",
        "missing-gate",
        "failed-gate",
        "order",
        "origin",
        "recompiled",
        "session-path",
        "build-source",
    ],
)
def test_foreign_stale_or_incomplete_job_cannot_qualify(evidence, args, mutation):
    report, _, _, _ = evidence
    if mutation in ("source", "image", "run"):
        report[
            {"source": "source_sha", "image": "image_digest", "run": "run_id"}[mutation]
        ] = "foreign"
    elif mutation in ("driver", "inventory"):
        report[{"driver": "driver", "inventory": "source_inventory"}[mutation]][
            "sha256"
        ] = "0" * 64
    elif mutation == "prepare-only":
        report["status"] = "prepared"
    elif mutation == "missing-gate":
        report["cases"].pop()
    elif mutation == "failed-gate":
        report["cases"][5]["status"] = "failed"
    elif mutation == "order":
        report["cases"][1:3] = reversed(report["cases"][1:3])
    elif mutation == "origin":
        report["imports"]["tensorrt"]["model_build_sources"] = {
            "full": "d" * 40,
            "tail": "d" * 40,
        }
    elif mutation == "recompiled":
        report["recompiled"] = True
    elif mutation == "session-path":
        report["sessions"][2] = report["sessions"][1]
    else:
        report["builds"]["aoti"]["source_sha"] = "d" * 40
    with pytest.raises(ValueError):
        validate(report, args)


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes",
        "image",
        "source",
        "runtime",
        "order",
        "file",
        "boundary",
        "parity",
        "timings",
        "statistics",
        "speedup",
        "stack",
        "library",
    ],
)
def test_comparison_is_bound_to_exact_session_identity_qualification_and_timings(
    evidence, mutation
):
    report, comparison, proofs, raw = evidence
    if mutation == "bytes":
        raw += b" "
    elif mutation in ("image", "source", "runtime"):
        proofs[3][
            {"image": "image_digest", "source": "source_sha", "runtime": "runtime"}[
                mutation
            ]
        ] = "foreign"
    elif mutation == "order":
        proofs[1:3] = reversed(proofs[1:3])
    elif mutation == "file":
        proofs[0]["sha256"] = "f" * 64
    elif mutation == "boundary":
        comparison["boundary"] = "kernel-only"
        raw = bind(report, comparison)
    elif mutation == "parity":
        proofs[2]["requests"][-1]["parity"] = "failed"
    elif mutation == "timings":
        proofs[2]["requests"][3]["seconds"] = -1
    elif mutation == "statistics":
        comparison["runtime_statistics"]["aoti"]["warm_seconds"]["median"] = 0.1
        raw = bind(report, comparison)
    elif mutation == "speedup":
        comparison["warm_speedup_vs_python"]["tensorrt"] = 50
        raw = bind(report, comparison)
    elif mutation == "stack":
        proofs[1]["environment_sha256"] = "f" * 64
    else:
        proofs[1]["numerical_libraries"] = {
            "libtorch_cpu.so": {"sha256": "a" * 64, "size_bytes": 5}
        }
    with pytest.raises(ValueError):
        controller.validate_comparison(raw, proofs, report)


def log_evidence(report, proofs, raw):
    envelope = {
        "schema_version": 1,
        "comparison_base64": base64.b64encode(raw).decode(),
        "sessions": proofs,
    }
    return "\n".join(
        [
            controller.job.BEGIN,
            json.dumps(report),
            controller.job.END,
            controller.COMPARISON_BEGIN,
            json.dumps(envelope),
            controller.COMPARISON_END,
        ]
    )


def test_extractor_retains_failed_summaries_but_retries_incomplete_success(evidence):
    report, _, proofs, raw = evidence
    text = log_evidence(report, proofs, raw)
    observed, transported = controller.extract_evidence(text)
    assert observed == report and transported["raw"] == raw
    with pytest.raises(ValueError, match="complete"):
        controller.extract_evidence(text.split(controller.COMPARISON_BEGIN)[0])
    report["status"] = "failed"
    observed, transported = controller.extract_evidence(
        controller.job.BEGIN + json.dumps(report) + controller.job.END
    )
    assert observed["status"] == "failed" and transported is None


def test_extractor_recovers_one_failed_summary_from_interleaved_traceback(evidence):
    report, _, _, _ = evidence
    report.update(status="failed", error="public TensorRT build exited 2")
    report["sessions"] = []
    output = "\n".join(
        [
            "Traceback (most recent call last):",
            controller.job.BEGIN,
            '  File "<string>", line 12, in <module>',
            '  File "/workspace/qa/native_inference/pnms_cfd_job.py", line 488, in run',
            json.dumps(report),
            "    gate(",
            controller.job.END,
            "AppError: public TensorRT build exited 2",
        ]
    )
    observed, transport = controller.extract_evidence(output)
    assert observed == report
    assert observed["status"] == "failed"
    assert observed["error"] == "public TensorRT build exited 2"
    assert transport is None


@pytest.mark.parametrize(
    "extra",
    [
        '{"status":"failed"}',
        "[]",
        "null",
        '{"status":"failed",',
        '{"status":"failed","status":"passed"}',
    ],
)
def test_extractor_rejects_ambiguous_interleaved_summary_records(evidence, extra):
    report, _, _, _ = evidence
    report["status"] = "failed"
    output = "\n".join(
        [
            controller.job.BEGIN,
            '  File "<string>", line 12, in <module>',
            json.dumps(report),
            extra,
            controller.job.END,
        ]
    )
    with pytest.raises(ValueError):
        controller.extract_evidence(output)


def test_extractor_rejects_duplicate_complete_summary_blocks(evidence):
    report, _, _, _ = evidence
    report["status"] = "failed"
    record = "\n".join([controller.job.BEGIN, json.dumps(report), controller.job.END])
    with pytest.raises(ValueError, match="ambiguous"):
        controller.extract_evidence(record + "\n" + record)


def test_job_command_binds_public_parser_mount_identities_and_nested_deadlines(args):
    command = controller.build_job_command(args, "owned")
    assert (
        command[command.index("--mount") + 1]
        == "/PhysicsNeMo/platform/qa:/outputs:node-nfs:" + args.lustre_storage
    )
    assert command[command.index("--num-workers") + 1] == "1"
    assert command[command.index("--max-failure-retry") + 1] == "0"
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[:5] == [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "7140s",
    ]
    script = runner.index("-c")
    assert runner[script + 1] == controller.WRAPPER
    assert runner[script + 2 : script + 4] == [
        "/workspace/qa/native_inference/pnms_cfd_job.py",
        controller.FROZEN_RUNNER,
    ]
    forwarded = controller.job.parser().parse_args(runner[script + 4 :])
    assert forwarded.timeout == 7080 and forwarded.timeout < 7140 < args.job_timeout
    assert (
        forwarded.image == args.image
        and forwarded.source_sha == args.expected_source_sha
    )
    assert (
        forwarded.expected_source_inventory_sha256
        == args.expected_source_inventory_sha256
    )
    assert forwarded.expected_runner_sha256 == controller.FROZEN_RUNNER
    assert (
        forwarded.reference_fields.as_posix()
        == "/outputs/native-inference/cfd-full-20261021t164815-r3/reference/fields.json"
    )
    assert (
        forwarded.input_root.as_posix()
        == "/outputs/native-inference/cfd-full-20261021t164815-r3/inputs/drivaerml"
    )
    assert "--prepare-only" not in runner


@pytest.mark.parametrize("value", [0, 180, 7201, float("nan"), float("inf")])
def test_unbounded_or_invalid_deadlines_never_reach_lepton(args, monkeypatch, value):
    args.job_timeout = value
    monkeypatch.setattr(
        controller.lifecycle,
        "run",
        lambda *a, **kw: pytest.fail("invalid deadline reached Lepton"),
    )
    with pytest.raises(ValueError, match="timeout"):
        controller.run(args)


def test_keep_job_is_disallowed(args, monkeypatch):
    args.keep_job = True
    monkeypatch.setattr(
        controller.lifecycle,
        "run",
        lambda *a, **kw: pytest.fail("keep-job reached Lepton"),
    )
    with pytest.raises(ValueError, match="cleanup"):
        controller.run(args)


@pytest.mark.parametrize("failure", [None, "source", "cleanup"])
def test_real_lifecycle_validates_historical_logs_preserves_comparison_and_removes_only_owned_job(
    evidence, args, monkeypatch, failure
):
    report, _, proofs, raw = evidence
    if failure == "source":
        report["source_sha"] = "d" * 40
    log = log_evidence(report, proofs, raw)
    history = "\n".join(
        "2026-10-29 12:00:00.123456|" + json.dumps(line) for line in log.splitlines()
    )
    calls = []

    def lep(command, **kw):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: owned-cfd-job\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Completed"}'
        if command[:2] == ["log", "get"]:
            return 0, history
        if command == ["job", "remove", "-i", "owned-cfd-job"]:
            return (7, "cleanup failed") if failure == "cleanup" else (0, "removed")
        pytest.fail("unexpected Lepton operation: " + repr(command))

    monkeypatch.setattr(controller.lifecycle, "lep", lep)
    code = controller.run(args)
    root = args.artifact_dir / args.run_id
    result = json.loads((root / "summary.json").read_text())
    assert code == (0 if failure is None else 1)
    assert result["status"] == ("passed" if failure is None else "failed")
    assert calls[-1] == ["job", "remove", "-i", "owned-cfd-job"]
    assert not any(call[:2] == ["job", "log"] for call in calls)
    if failure == "source":
        assert not (root / "comparison.json").exists()
    else:
        assert (root / "comparison.json").read_bytes() == raw
        proof = json.loads((root / "comparison-proof.json").read_text())
        assert proof["comparison"] == controller.identity(raw)
        assert proof["status"] == ("passed" if failure is None else "failed")
    assert (root / "controller-identity.json").is_file()
