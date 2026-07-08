# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sample_comparison_payload() -> dict[str, object]:
    return {
        "config": {
            "model_type": "fcn",
            "nsteps": 20,
            "backends": ["rust", "py_async", "zarr_sync"],
        },
        "performance": {
            "rust": {
                "timings_sec": {"total_wall_sec": 9.0},
                "io_metrics": {"total_io": 0.8, "io_write": 0.7},
                "compute_metrics": {"total_compute": 7.0},
            },
            "py_async": {
                "timings_sec": {"total_wall_sec": 12.0},
                "io_metrics": {"total_io": 3.6, "io_write": 3.4},
                "compute_metrics": {"total_compute": 7.1},
            },
            "zarr_sync": {
                "timings_sec": {"total_wall_sec": 26.0},
                "io_metrics": {"total_io": 18.1, "io_write": 18.0},
                "compute_metrics": {"total_compute": 7.2},
            },
            "comparisons": {
                "rust_vs_py_async": {
                    "total_wall_ratio": 1.3333333333,
                    "total_io_ratio": 4.5,
                    "io_write_ratio": 4.8571428571,
                },
                "rust_vs_zarr_sync": {
                    "total_wall_ratio": 2.8888888889,
                    "total_io_ratio": 22.625,
                    "io_write_ratio": 25.7142857143,
                },
            },
        },
        "consistency": {
            "rust_vs_py_async": {
                "all_consistent": True,
                "max_abs_diff_global": 0.0,
            },
            "rust_vs_zarr_sync": {
                "all_consistent": True,
                "max_abs_diff_global": 0.0,
            },
        },
        "artifacts": {"run_dir": "/outputs/model-run"},
    }


def test_job_runner_builds_per_model_summary(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    comparison_path = tmp_path / "comparison.json"
    write_json(comparison_path, sample_comparison_payload())

    summary = module.summarize_comparison("fcn", comparison_path, returncode=0)

    assert summary["status"] == "passed"
    assert summary["comparison_json"] == str(comparison_path)
    assert summary["backends"]["rust"]["total_wall_sec"] == 9.0
    assert summary["backends"]["rust"]["total_io_sec"] == 0.8
    assert summary["backends"]["py_async"]["io_write_sec"] == 3.4
    assert summary["ratios"]["rust_vs_py_async"]["total_io_ratio"] == 4.5
    assert summary["consistency"]["all_consistent"] is True
    assert summary["consistency"]["max_abs_diff_global"] == 0.0


def test_job_runner_marks_missing_consistency_as_failure(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    payload = sample_comparison_payload()
    payload.pop("consistency")
    comparison_path = tmp_path / "comparison.json"
    write_json(comparison_path, payload)

    summary = module.summarize_comparison("fcn", comparison_path, returncode=0)

    assert summary["status"] == "failed"
    assert summary["consistency"]["all_consistent"] is False


def test_job_runner_marks_skipped_consistency_as_passed(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    payload = sample_comparison_payload()
    payload["config"]["model_type"] = "dlwp"
    payload["config"]["skip_consistency"] = True
    payload["consistency"] = {}
    comparison_path = tmp_path / "comparison.json"
    write_json(comparison_path, payload)

    summary = module.summarize_comparison("dlwp", comparison_path, returncode=0)

    assert summary["status"] == "passed"
    assert summary["consistency"]["all_consistent"] is True
    assert summary["consistency"]["skipped"] is True


def test_job_runner_generates_perf_compare_report(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    comparison_path = tmp_path / "comparison.json"
    write_json(comparison_path, sample_comparison_payload())
    model_summary = module.summarize_comparison("sfno", comparison_path, returncode=0)
    summary = {
        "final_result": "passed",
        "config": {
            "preset": "benchmark-report",
            "device": "cuda:0",
            "backends": ["rust", "py_async", "zarr_sync"],
            "models": ["sfno"],
        },
        "models": {"sfno": model_summary},
    }

    report = module.generate_perf_compare_report(summary)

    assert "# Rust I/O Performance Comparison Report" in report
    assert "## Total I/O" in report
    assert "Compared With Existing Doc" not in report
    assert "| sfno | rust | 800.0 ms | 700.0 ms |" in report
    assert "| sfno | Rust vs Async total I/O | 4.5x |" in report


def test_job_runner_report_inverts_reversed_rust_ratio_keys() -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    summary = {
        "final_result": "passed",
        "config": {
            "preset": "benchmark-report",
            "device": "cuda:0",
            "backends": ["py_async", "rust", "zarr_sync"],
            "models": ["sfno"],
        },
        "models": {
            "sfno": {
                "status": "passed",
                "backends": {},
                "ratios": {
                    "py_async_vs_rust": {
                        "total_io_ratio": 0.25,
                        "total_wall_ratio": 0.75,
                    },
                    "zarr_sync_vs_rust": {
                        "total_io_ratio": 0.05,
                        "total_wall_ratio": 0.5,
                    },
                },
            }
        },
    }

    report = module.generate_perf_compare_report(summary)

    assert "| sfno | Rust vs Async total I/O | 4.0x |" in report
    assert "| sfno | Rust vs Sync total I/O | 20.0x |" in report


def test_job_runner_report_derives_missing_rust_ratio_from_backends() -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    summary = {
        "final_result": "passed",
        "config": {
            "preset": "benchmark-report",
            "device": "cuda:0",
            "backends": ["zarr_sync", "py_async", "rust"],
            "models": ["sfno"],
        },
        "models": {
            "sfno": {
                "status": "passed",
                "backends": {
                    "rust": {"total_io_sec": 1.0, "total_wall_sec": 10.0},
                    "py_async": {"total_io_sec": 4.0, "total_wall_sec": 20.0},
                    "zarr_sync": {"total_io_sec": 10.0, "total_wall_sec": 30.0},
                },
                "ratios": {
                    "zarr_sync_vs_py_async": {
                        "total_io_ratio": 0.4,
                        "total_wall_ratio": 0.6666666667,
                    },
                    "zarr_sync_vs_rust": {
                        "total_io_ratio": 0.1,
                        "total_wall_ratio": 0.3333333333,
                    },
                },
            }
        },
    }

    report = module.generate_perf_compare_report(summary)

    assert "| sfno | Rust vs Async total I/O | 4.0x |" in report
    assert "| sfno | Rust vs Async wall | 2.0x |" in report


def test_job_runner_marks_model_failure(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )

    result = module.run_models(
        models=["fcn", "dlwp"],
        output_dir=tmp_path,
        benchmark_script=Path(
            "/workspace/scripts/compare_deterministic_rust_vs_py_async.py"
        ),
        start_time="2024-01-01T00:00:00",
        nsteps=1,
        device="cuda:0",
        backends=["rust", "py_async", "zarr_sync"],
        seed=1337,
        rust_profile="default",
        warmup_steps=0,
        profile_top_n=1,
        earth2studio_root=None,
        runner=lambda cmd, **kwargs: SimpleNamespace(returncode=2, stdout="boom"),
    )

    assert result["final_result"] == "failed"
    assert result["models"]["fcn"]["status"] == "failed"
    assert result["models"]["fcn"]["returncode"] == 2
    assert "comparison.json not found" in result["models"]["fcn"]["error"]
    assert result["models"]["dlwp"]["status"] == "failed"


def test_job_runner_benchmark_report_preset_uses_doc_configs() -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )

    fcn = module.model_config_for("fcn", preset="benchmark-report", default_nsteps=1)
    dlwp = module.model_config_for("dlwp", preset="benchmark-report", default_nsteps=1)
    sfno = module.model_config_for("sfno", preset="benchmark-report", default_nsteps=1)
    fcn3 = module.model_config_for("fcn3", preset="benchmark-report", default_nsteps=1)

    assert fcn["nsteps"] == 10
    assert fcn["rust_options"]["max_pool_buffers"] == 64
    assert fcn["rust_options"]["queue_capacity"] == 128
    assert dlwp["nsteps"] == 20
    assert dlwp["rust_options"]["max_pool_buffers"] == 128
    assert dlwp["skip_consistency"] is True
    assert sfno["nsteps"] == 10
    assert sfno["rust_options"]["max_pool_buffers"] == 300
    assert sfno["rust_options"]["max_pool_bytes"] == 2_147_483_648
    assert fcn3["nsteps"] == 10
    assert fcn3["rust_options"]["max_pool_buffers"] == 150
    assert fcn3["rust_options"]["max_pool_bytes"] == 1_073_741_824


def test_job_runner_benchmark_report_command_includes_pool_flags(
    tmp_path: Path,
) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    model_config = module.model_config_for(
        "stormcast", preset="benchmark-report", default_nsteps=1
    )

    cmd = module.build_benchmark_command(
        benchmark_script=Path("/bench.py"),
        model="stormcast",
        model_output_dir=tmp_path / "stormcast",
        start_time="2024-01-01T00:00:00",
        nsteps=model_config["nsteps"],
        device="cuda:0",
        backends=["rust", "py_async", "zarr_sync"],
        seed=1337,
        rust_profile="default",
        warmup_steps=3,
        profile_top_n=40,
        earth2studio_root=Path("/site-packages"),
        rust_options=model_config["rust_options"],
    )

    assert "--nsteps" in cmd
    assert cmd[cmd.index("--nsteps") + 1] == "10"
    assert "--max-pool-buffers" in cmd
    assert cmd[cmd.index("--max-pool-buffers") + 1] == "200"
    assert "--hot-slab-buffers" in cmd
    assert cmd[cmd.index("--hot-slab-buffers") + 1] == "100"
    assert "--max-pool-bytes" in cmd
    assert cmd[cmd.index("--max-pool-bytes") + 1] == "1073741824"


def test_job_runner_dlwp_benchmark_report_command_skips_consistency(
    tmp_path: Path,
) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    model_config = module.model_config_for(
        "dlwp", preset="benchmark-report", default_nsteps=1
    )

    cmd = module.build_benchmark_command(
        benchmark_script=Path("/bench.py"),
        model="dlwp",
        model_output_dir=tmp_path / "dlwp",
        start_time="2024-01-01T00:00:00",
        nsteps=model_config["nsteps"],
        device="cuda:0",
        backends=["rust", "py_async", "zarr_sync"],
        seed=1337,
        rust_profile="default",
        warmup_steps=3,
        profile_top_n=40,
        earth2studio_root=Path("/site-packages"),
        rust_options=model_config["rust_options"],
        skip_consistency=model_config["skip_consistency"],
    )

    assert "--skip-consistency" in cmd
    assert cmd[cmd.index("--nsteps") + 1] == "20"


def test_job_runner_default_backends_build_valid_command(tmp_path: Path) -> None:
    module = load_module(
        "run_rust_io_benchmark_job",
        SCRIPTS_DIR / "run_rust_io_benchmark_job.py",
    )
    args = module.build_parser().parse_args(["--output-dir", str(tmp_path)])

    cmd = module.build_benchmark_command(
        benchmark_script=Path("/bench.py"),
        model="fcn",
        model_output_dir=tmp_path / "fcn",
        start_time=args.start_time,
        nsteps=args.nsteps,
        device=args.device,
        backends=args.backends,
        seed=args.seed,
        rust_profile=args.rust_profile,
        warmup_steps=args.warmup_steps,
        profile_top_n=args.profile_top_n,
        earth2studio_root=args.earth2studio_root,
    )

    assert args.backends == ["rust", "py_async", "zarr_sync"]
    assert cmd[cmd.index("--backends") + 1] == "rust,py_async,zarr_sync"


def test_lepton_wrapper_dry_run_builds_expected_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFS_MOUNT_BASE", "/mnt/test-nfs")
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    args = module.build_parser().parse_args(
        [
            "--image-tag",
            "nvcr.io/example/scicomp-ferroflux:test",
            "--run-id",
            "abc123xy",
            "--artifact-dir",
            str(tmp_path),
            "--lustre-dir",
            "rust_io_tests",
            "--models",
            "fcn,dlwp",
            "--nsteps",
            "1",
            "--device",
            "cuda:0",
            "--dry-run",
        ]
    )

    result = module.run(args)

    summary_path = tmp_path / "rust-io-benchmark" / "abc123xy" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    job_args = summary["lepton"]["job_args"]
    command = job_args[job_args.index("--command") + 1]
    job_name = job_args[job_args.index("--name") + 1]

    assert result == 0
    assert summary["final_result"] == "dry-run"
    assert job_name == "ff-rio-custom-fcn-dlwp-abc123xy"
    assert summary["lepton"]["job_name"] == job_name
    assert summary["artifacts"]["job_log"].endswith(f"{job_name}.log")
    assert "--container-image" in job_args
    assert "nvcr.io/example/scicomp-ferroflux:test" in job_args
    assert "/mnt/test-nfs/rust_io_tests:/outputs:node-nfs:lustre" in job_args
    assert "run_rust_io_benchmark_job.py" in command
    assert "--models fcn,dlwp" in command
    assert "--nsteps 1" in command
    assert "--device cuda:0" in command
    assert command.startswith("mkdir -p /outputs/rust-io-benchmark/abc123xy; { ")
    assert " | tee /outputs/rust-io-benchmark/abc123xy/job-output.log" in command
    assert "PIPESTATUS[0]" in command
    assert "> /outputs/rust-io-benchmark/abc123xy/job-output.log" not in command
    assert "earth2studio[pangu]==0.13.0" not in command


def test_lepton_wrapper_base64_encodes_embedded_script() -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    payload = "PYRUNNER\nPYBENCHMARK\n$(echo unsafe)"

    command = module.base64_write_command("/tmp/runner.py", payload)
    encoded = shlex.split(command)[2]

    assert "<<" not in command
    assert payload not in command
    assert base64.b64decode(encoded).decode("utf-8") == payload

    python_command = module.base64_python_command(payload)
    python_encoded = shlex.split(python_command)[2]
    assert "<<" not in python_command
    assert payload not in python_command
    assert base64.b64decode(python_encoded).decode("utf-8") == payload


def test_lepton_wrapper_benchmark_report_preset_defaults_models(tmp_path: Path) -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    args = module.build_parser().parse_args(
        [
            "--image-tag",
            "v0.1.20260529.0",
            "--run-id",
            "report01",
            "--artifact-dir",
            str(tmp_path),
            "--lustre-dir",
            "rust_io_tests",
            "--preset",
            "benchmark-report",
            "--dry-run",
        ]
    )

    result = module.run(args)

    summary_path = tmp_path / "rust-io-benchmark" / "report01" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    job_args = summary["lepton"]["job_args"]
    command = job_args[job_args.index("--command") + 1]
    job_name = job_args[job_args.index("--name") + 1]

    assert result == 0
    assert summary["config"]["preset"] == "benchmark-report"
    assert summary["config"]["models"] == "fcn,dlwp,sfno,stormcast,fcn3"
    assert job_name == "ff-rio-report-report01"
    assert summary["lepton"]["job_name"] == job_name
    assert "--preset benchmark-report" in command
    assert "--models fcn,dlwp,sfno,stormcast,fcn3" in command
    assert "base64 -d >" in command
    assert "pangu3" not in command
    assert "onnxruntime-gpu" not in command
    assert "earth2studio[pangu]==0.13.0" not in command


def test_lepton_wrapper_writes_perf_report_without_scripts_on_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(
        "qa.scripts.run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    monkeypatch.delitem(sys.modules, "run_rust_io_benchmark_job", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPTS_DIR],
    )
    args = SimpleNamespace(artifact_dir=str(tmp_path))
    (tmp_path / "rust-io-benchmark" / "report01").mkdir(parents=True)
    report = {
        "final_result": "passed",
        "config": {"preset": "benchmark-report", "models": ["sfno"]},
        "models": {"sfno": {"status": "passed"}},
    }

    report_path = module.write_perf_report_if_available(
        args=args, run_id="report01", benchmark_report=report
    )

    assert report_path is not None
    assert Path(report_path).is_file()
    assert "# Rust I/O Performance Comparison Report" in Path(report_path).read_text(
        encoding="utf-8"
    )


def test_lepton_wrapper_reader_summary_preserves_skipped_consistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    args = module.build_parser().parse_args(
        [
            "--image-tag",
            "v0.1.20260529.0",
            "--run-id",
            "report01",
            "--artifact-dir",
            str(tmp_path),
            "--lustre-dir",
            "rust_io_tests",
            "--preset",
            "benchmark-report",
        ]
    )
    args = module.normalize_args(args)
    remote_root = tmp_path / "remote"
    remote_summary = remote_root / "rust-io-benchmark" / "report01" / "summary.json"
    write_json(
        remote_summary,
        {
            "final_result": "passed",
            "config": {
                "models": ["dlwp"],
                "backends": ["rust", "py_async", "zarr_sync"],
                "preset": "benchmark-report",
                "model_configs": {"dlwp": {"skip_consistency": True}},
                "nsteps": 20,
                "device": "cuda:0",
                "rust_profile": "default",
                "warmup_steps": 3,
            },
            "models": {
                "dlwp": {
                    "status": "passed",
                    "returncode": 0,
                    "backends": {},
                    "ratios": {},
                    "consistency": {
                        "all_consistent": True,
                        "skipped": True,
                        "max_abs_diff_global": 0.0,
                    },
                }
            },
        },
    )
    args.mount_target = str(remote_root)
    created_commands: list[str] = []

    def fake_run_streaming(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
        if cmd[:3] == ["lep", "job", "create"]:
            created_commands.append(cmd[cmd.index("--command") + 1])
            return 0, "ID: reader-1\n"
        return 0, ""

    def fake_capture_job_logs(
        *,
        job_id: str,
        env: dict[str, str],
        artifact_path: Path,
        timeout_seconds: int = 120,
    ) -> str:
        assert created_commands
        encoded = shlex.split(created_commands[-1])[2]
        script = base64.b64decode(encoded).decode("utf-8")
        namespace = {"__name__": "__main__"}
        import contextlib
        import io

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(script, namespace)
        return stream.getvalue()

    monkeypatch.setattr(module, "run_streaming", fake_run_streaming)
    monkeypatch.setattr(module, "wait_for_loggable_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "capture_job_logs", fake_capture_job_logs)

    payload = module.fetch_summary_via_reader_job(args, run_id="report01", env={})
    report_path = module.write_perf_report_if_available(
        args=args, run_id="report01", benchmark_report=payload
    )

    assert payload["models"]["dlwp"]["consistency"]["skipped"] is True
    assert report_path is not None
    report = Path(report_path).read_text(encoding="utf-8")
    assert "| dlwp | passed | 0 | skipped | n/a | Consistency skipped" in report


def test_lepton_wrapper_log_capture_timeout_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )

    def timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(
            cmd=["lep", "job", "log", "-i", "job-1"],
            timeout=120,
            output="partial log\n",
        )

    monkeypatch.setattr(module.subprocess, "run", timeout)
    log_path = tmp_path / "job.log"

    output = module.capture_job_logs(
        job_id="job-1", env={}, artifact_path=log_path, timeout_seconds=120
    )

    assert "partial log" in output
    assert "continuing summary and cleanup" in output
    assert "continuing summary and cleanup" in log_path.read_text(encoding="utf-8")


def test_lepton_wrapper_stops_failed_job_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )
    calls: list[list[str]] = []

    def record(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
        calls.append(cmd)
        return 0, ""

    monkeypatch.setattr(module, "run_streaming", record)

    module.stop_job_if_needed(job_id="job-1", env={}, job_exit_code=1)
    module.stop_job_if_needed(job_id="job-2", env={}, job_exit_code=0)

    assert calls == [["lep", "job", "stop", "-i", "job-1"]]


def test_lepton_wrapper_requires_image_tag() -> None:
    module = load_module(
        "run_lepton_rust_io_benchmark",
        SCRIPTS_DIR / "run_lepton_rust_io_benchmark.py",
    )

    with pytest.raises(SystemExit):
        module.build_parser().parse_args([])
