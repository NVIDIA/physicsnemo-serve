import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qa.native_inference import run_job


def arguments(tmp_path):
    return run_job.parser().parse_args(
        [
            "--run-id",
            "test-run",
            "--output",
            str(tmp_path / "run"),
            "--expected-source-sha",
            "a" * 40,
            "--image-digest",
            "registry/qa@sha256:" + "b" * 64,
        ]
    )


def test_failed_environment_blocks_both_stages_and_preserves_summary(
    tmp_path, monkeypatch, capsys
):
    def unavailable(ctx):
        raise ValueError("no CUDA available")

    monkeypatch.setattr(run_job, "preflight", unavailable)
    args = arguments(tmp_path)
    assert run_job.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"]["build"]["status"] == "failed"
    assert report["stages"]["consumer"]["status"] == "blocked"
    assert len(report["cases"]) == len(report["expected_cases"])
    assert report["cases"][0]["status"] == "failed"
    assert (args.output / "junit.xml").is_file()
    assert run_job.SUMMARY_END in capsys.readouterr().out


def test_handoff_removes_original_producer_paths_and_preserves_evidence(tmp_path):
    for name in ("projects", "build", "builds", "check-affine"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sentinel").write_text(name)
    ctx = SimpleNamespace(
        root=tmp_path,
        args=SimpleNamespace(run_id="test", expected_source_sha="a" * 40),
        relocate=lambda record: {**record, "model_dir": "consumer/model"},
    )
    run_job.handoff_models(ctx, [{"name": "affine-a"}])
    for name in ("projects", "build", "builds", "check-affine"):
        assert not (tmp_path / name).exists(), (
            "native consumer must not resolve original producer paths"
        )
        assert (tmp_path / "producer" / name / "sentinel").read_text() == name
    assert (tmp_path / "handoff.json").is_file()


def test_sigterm_finalizes_failed_report_instead_of_silent_exit(tmp_path):
    source = Path(__file__).resolve().parents[1]
    code = """
import os, signal, sys
from qa.native_inference import run_job
args = run_job.parser().parse_args(sys.argv[1:])
def terminate(ctx):
    os.kill(os.getpid(), signal.SIGTERM)
run_job.preflight = terminate
raise SystemExit(run_job.execute(args))
"""
    args = arguments(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--run-id",
            args.run_id,
            "--output",
            str(args.output),
            "--expected-source-sha",
            args.expected_source_sha,
            "--image-digest",
            args.image_digest,
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert (args.output / "summary.json").is_file()
    report = json.loads((args.output / "summary.json").read_text())
    assert report["status"] == "failed", (
        "termination must not leave an in-progress report"
    )
    assert run_job.SUMMARY_END in result.stdout


def test_existing_attempt_is_never_overwritten(tmp_path):
    args = arguments(tmp_path)
    args.output.mkdir()
    sentinel = args.output / "summary.json"
    sentinel.write_text("preserve failed attempt")
    with pytest.raises(FileExistsError):
        run_job.execute(args)
    assert sentinel.read_text() == "preserve failed attempt"


def test_full_profile_without_manifest_reaches_job_execution(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job.py",
            "--profile",
            "full",
            "--run-id",
            args.run_id,
            "--output",
            str(args.output),
            "--expected-source-sha",
            args.expected_source_sha,
            "--image-digest",
            args.image_digest,
        ],
    )
    executed = []
    monkeypatch.setattr(run_job, "execute", lambda parsed: executed.append(parsed) or 0)
    assert run_job.main() == 0
    assert executed[0].profile == "full" and executed[0].assets is None


def test_asset_download_failure_is_recorded_and_blocks_consumers(tmp_path, monkeypatch):
    from qa.native_inference import transolver

    def download_unavailable(cache_root):
        raise OSError("asset download unavailable")

    monkeypatch.setattr(transolver, "prepare_assets", download_unavailable)
    monkeypatch.setattr(run_job, "preflight", lambda ctx: None)
    monkeypatch.setattr(run_job, "build_affine", lambda ctx: [])
    args = arguments(tmp_path)
    args.profile = "full"
    assert run_job.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    cases = {case["name"]: case for case in report["cases"]}
    assert len(cases) == 33
    assert cases["transolver.assets"]["status"] == "failed"
    assert "asset download unavailable" in cases["transolver.assets"]["error"]
    assert cases["transolver.import"]["status"] == "blocked"
    assert report["stages"]["consumer"]["status"] == "blocked"


def test_subprocess_timeout_preserves_logs(tmp_path):
    args = arguments(tmp_path)
    args.output.mkdir()
    args.command_timeout = 0.05
    ctx = run_job.Context(args)
    with pytest.raises(subprocess.TimeoutExpired):
        ctx.run(
            "timeout",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
        )
    assert "started" in (args.output / "logs/timeout.stdout.log").read_text()


def test_failed_command_reports_bounded_output_tails_and_keeps_full_logs(
    tmp_path, monkeypatch, capsys
):
    def failed_command(ctx):
        ctx.run(
            "diagnostic",
            [
                sys.executable,
                "-c",
                "import sys; "
                "print('stdout-start-' + 'x' * 8192 + '-stdout-cause'); "
                "print('stderr-start-' + 'y' * 8192 + '-stderr-cause', file=sys.stderr); "
                "sys.exit(7)",
            ],
        )

    monkeypatch.setattr(run_job, "preflight", failed_command)
    args = arguments(tmp_path)
    assert run_job.execute(args) == 1
    report = json.loads((args.output / "summary.json").read_text())
    error = report["error"]
    assert "exit 7" in error
    assert "stdout-cause" in error
    assert "stderr-cause" in error
    assert "stdout tail" in error and "stderr tail" in error
    assert "stdout-start" not in error and "stderr-start" not in error
    assert len(error.encode()) < 9000, "failed commands must not flood job summaries"
    assert report["cases"][0]["error"] == error
    emitted = capsys.readouterr().out
    assert "stdout-cause" in emitted and "stderr-cause" in emitted
    for stream in ("stdout", "stderr"):
        content = (args.output / f"logs/diagnostic.{stream}.log").read_text()
        assert content.startswith(f"{stream}-start-")
        assert content.endswith(f"-{stream}-cause\n")
        assert len(content) > 8192
