"""Native QA reports must prove complete execution of the requested candidate."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa" / "scripts"))
import run_lepton_native_inference_qa as qa


IMAGE = "registry.example.test/native-qa@sha256:" + "a" * 64
SOURCE = "b" * 40
IDENTITY = dict(
    run_id="unit-run", source_sha=SOURCE, image_digest=IMAGE, profile="smoke"
)


def passed_report():
    from qa.native_inference.contract import expected_cases

    names = expected_cases("smoke")
    return {
        "schema_version": 1,
        **IDENTITY,
        "status": "passed",
        "stages": {"build": {"status": "passed"}, "consumer": {"status": "passed"}},
        "expected_cases": names,
        "cases": [{"name": name, "status": "passed"} for name in names],
    }


def args_for(tmp_path, monkeypatch, *extra):
    monkeypatch.delenv("LEPTON_WORKSPACE_TOKEN", raising=False)
    return qa.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--run-id",
            "unit-run",
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "workspace",
            "--node-group",
            "node-group",
            "--pull-secret",
            "pull-secret",
            "--resource-shape",
            "gpu.test-shape",
            "--nfs-mount-base",
            "/mnt/shared",
            *extra,
        ]
    )


def job_log(report):
    return f"diagnostics\n{qa.SUMMARY_BEGIN}\n{json.dumps(report)}\n{qa.SUMMARY_END}\n"


def fake_job(
    monkeypatch, args, *, state="Completed", report=None, interrupt=False, remove_code=0
):
    calls = []
    report = passed_report() if report is None else report

    def fake(command, *, env, timeout):
        calls.append(command)
        assert 0 < timeout <= args.command_timeout
        action = command[1]
        if action == "create":
            assert (args.artifact_dir / args.run_id / "job-command.json").is_file()
            return 0, "ID: native-job-123\n"
        if action == "get":
            persisted = json.loads(
                (args.artifact_dir / args.run_id / "summary.json").read_text()
            )
            assert persisted["job_id"] == "native-job-123"
            if interrupt:
                raise KeyboardInterrupt()
            return 0, json.dumps({"status": {"state": state}})
        if action == "log":
            return 0, job_log(report)
        if action == "remove":
            return remove_code, ""
        return 0, ""

    monkeypatch.setattr(qa, "lep", fake)
    return calls


def test_one_job_success_requires_persisted_identity_and_validated_report(
    tmp_path, monkeypatch
):
    args = args_for(tmp_path, monkeypatch)
    calls = fake_job(monkeypatch, args)
    assert qa.run(args) == 0
    actions = [call[1] for call in calls]
    assert actions == ["create", "get", "log", "remove"]
    created = calls[0]
    assert created[created.index("--resource-shape") + 1] == "gpu.test-shape"
    assert created[created.index("--num-workers") + 1] == "1"
    assert created[created.index("--max-failure-retry") + 1] == "0"
    assert created[created.index("--max-job-failure-retry") + 1] == "0"
    assert "docker" not in created[-1]
    assert "--image-digest " + IMAGE in created[-1]
    summary = json.loads((tmp_path / "unit-run/summary.json").read_text())
    assert summary["status"] == "passed"
    assert summary["remote_artifacts"] == "/outputs/native-inference/unit-run"


@pytest.mark.parametrize(
    "state,wrong_identity,expected_code",
    [
        ("Completed", False, 0),
        ("Failed", False, 1),
        ("Completed", True, 1),
    ],
)
def test_finished_job_recovers_delayed_history_without_trusting_wrong_identity(
    tmp_path, monkeypatch, state, wrong_identity, expected_code
):
    args = args_for(tmp_path, monkeypatch)
    calls, sleeps = [], []
    report = passed_report()
    if wrong_identity:
        report["run_id"] = "another-run"
    history = "Fetching logs...\n" + "\n".join(
        "2026-10-06 16:00:00.000001|" + json.dumps(line)
        for line in [qa.SUMMARY_BEGIN, report, qa.SUMMARY_END]
    )

    def fake(command, *, env, timeout):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: native-job-123\n"
        if command[:2] == ["job", "get"]:
            return 0, json.dumps({"status": {"state": state}})
        if command[:2] == ["job", "log"]:
            return 0, "Connection stopped\n"
        if command[:2] == ["log", "get"]:
            assert command[command.index("--job") + 1] == "native-job-123"
            assert command[command.index("--limit") + 1] == "5000"
            assert "--start" in command and "--end" in command
            assert 0 < timeout <= min(args.command_timeout, 30)
            attempts = sum(call[:2] == ["log", "get"] for call in calls)
            return 0, "No logs yet\n" if attempts == 1 else history
        return 0, ""

    monkeypatch.setattr(qa, "lep", fake)
    monkeypatch.setattr(qa.time, "sleep", sleeps.append)
    assert qa.run(args) == expected_code
    assert [call[:2] for call in calls] == [
        ["job", "create"],
        ["job", "get"],
        ["job", "log"],
        ["log", "get"],
        ["log", "get"],
        ["job", "remove"],
    ]
    assert sleeps == [5]
    evidence = tmp_path / args.run_id
    assert (evidence / "job.log").read_text() == "Connection stopped\n"
    assert (evidence / "job-history-2.log").read_text() == history
    assert qa.extract_summary((evidence / "job-history.log").read_text()) == report
    assert json.loads((evidence / "job-summary.json").read_text()) == report


def test_unavailable_history_has_bounded_retries_and_still_removes_job(
    tmp_path, monkeypatch
):
    args = args_for(tmp_path, monkeypatch)
    calls, sleeps = [], []

    def fake(command, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: native-job-123\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Failed"}'
        return 0, "Connection stopped\n"

    monkeypatch.setattr(qa, "lep", fake)
    monkeypatch.setattr(qa.time, "sleep", sleeps.append)
    assert qa.run(args) == 1
    assert sum(call[:2] == ["log", "get"] for call in calls) == 3
    assert sleeps == [5, 5]
    assert calls[-1] == ["job", "remove", "-i", "native-job-123"]


@pytest.mark.parametrize(
    "mode", ["failed-job", "wrong-report", "no-summary", "cleanup-error"]
)
def test_job_and_report_must_both_succeed(tmp_path, monkeypatch, mode):
    args = args_for(tmp_path, monkeypatch)
    report = passed_report()
    if mode == "wrong-report":
        report["run_id"] = "other"
    elif mode == "no-summary":
        report = {}
    fake_job(
        monkeypatch,
        args,
        state="Failed" if mode == "failed-job" else "Completed",
        report=report,
        remove_code=1 if mode == "cleanup-error" else 0,
    )
    assert qa.run(args) == 1
    summary = json.loads((tmp_path / "unit-run/summary.json").read_text())
    assert summary["status"] == "failed"


def test_cancelled_job_stops_collects_logs_and_removes(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    calls = fake_job(monkeypatch, args, interrupt=True)
    assert qa.run(args) == 130
    assert [call[1] for call in calls] == ["create", "get", "stop", "log", "remove"]
    assert (
        json.loads((tmp_path / "unit-run/summary.json").read_text())["status"]
        == "cancelled"
    )


def test_keep_job_still_captures_diagnostics_after_cancellation(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch, "--keep-job")
    calls = fake_job(monkeypatch, args, interrupt=True)
    assert qa.run(args) == 130
    assert [call[1] for call in calls] == ["create", "get", "log"]
    assert (tmp_path / "unit-run/job.log").is_file()


def test_timeout_stops_job_and_leaves_diagnostics(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch, "--job-timeout", "1")
    calls = fake_job(monkeypatch, args, state="Running")
    times = iter([0, 0, 0, 2])
    monkeypatch.setattr(qa.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(qa.time, "sleep", lambda _: None)
    assert qa.run(args) == 1
    assert [call[1] for call in calls][-3:] == ["stop", "log", "remove"]
    assert (tmp_path / "unit-run/job.log").is_file()


def test_create_unknown_outcome_removes_only_its_unique_job_name(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        return (124, "") if command[1] == "create" else (0, "")

    monkeypatch.setattr(qa, "lep", fake)
    assert qa.run(args) == 1
    assert len(calls) == 2
    assert calls[1] == ["job", "remove", "-n", calls[0][calls[0].index("--name") + 1]]


def test_keep_job_skips_cleanup_and_dry_run_never_calls_lepton(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch, "--keep-job")
    calls = fake_job(monkeypatch, args)
    assert qa.run(args) == 0
    assert [call[1] for call in calls] == ["create", "get", "log"]
    args.run_id = "dry-run"
    args.dry_run = True
    monkeypatch.setattr(
        qa, "lep", lambda *_args, **_kwargs: pytest.fail("dry run launched a command")
    )
    assert qa.run(args) == 0
    assert (
        json.loads((tmp_path / "dry-run/summary.json").read_text())["status"]
        == "dry-run"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--image", "registry.example.test/native:latest"],
        ["--expected-source-sha", "abc123"],
        ["--resource-shape", ""],
        ["--assets", "/outside/assets.json"],
        ["--lustre-dir", "../other"],
        ["--mount-target", "/"],
        ["--job-timeout", "nan"],
    ],
)
def test_invalid_requests_fail_before_launch(tmp_path, monkeypatch, extra):
    args = args_for(tmp_path, monkeypatch, *extra)
    with pytest.raises(ValueError):
        qa.validate_args(args)


def test_full_profile_forwards_mounted_manifest(tmp_path, monkeypatch):
    args = args_for(
        tmp_path,
        monkeypatch,
        "--profile",
        "full",
        "--assets",
        "/outputs/assets/manifest.json",
    )
    qa.validate_args(args)
    command = qa.build_job_command(args, "unit-name")
    assert "--assets /outputs/assets/manifest.json" in command[-1]


@pytest.mark.parametrize("mount_target", ["/outputs", "/custom/shared"])
def test_full_profile_without_manifest_uses_in_job_download(
    tmp_path, monkeypatch, mount_target
):
    args = args_for(
        tmp_path, monkeypatch, "--profile", "full", "--mount-target", mount_target
    )
    qa.validate_args(args)
    command = qa.build_job_command(args, "unit-name")
    runner = shlex.split(command[-1])
    assert runner[runner.index("--profile") + 1] == "full"
    assert (
        runner[runner.index("--output") + 1]
        == f"{mount_target}/native-inference/unit-run"
    )
    assert "--assets" not in runner


def test_lepton_command_explicitly_runs_nvidia_initialization(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    command = qa.build_job_command(args, "unit-name")
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[:4] == [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        "python3",
        "/opt/physicsnemo-qa/run_job.py",
    ]


def test_retry_limits_are_serialized_in_job_spec_before_submission(
    tmp_path, monkeypatch
):
    args = args_for(tmp_path, monkeypatch)
    calls = []
    observed = {}

    def fake(command, **kwargs):
        calls.append(command)
        if command[1] == "create":
            if "--file" in command:
                path = Path(command[command.index("--file") + 1])
                observed["path"] = path
                observed["specification"] = json.loads(path.read_text())
            return 1, "offline rejected submission"
        return 0, ""

    monkeypatch.setattr(qa, "lep", fake)
    assert qa.run(args) == 1
    assert observed.get("specification") == {
        "max_failure_retry": 0,
        "max_job_failure_retry": 0,
    }
    assert observed["path"].parent == (tmp_path / args.run_id).resolve()
    assert [command[1] for command in calls] == ["create", "remove"]


def test_cli_timeout_redacts_partial_output(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], kwargs["timeout"], output=b"secret-token partial logs"
        )

    monkeypatch.setattr(qa.subprocess, "run", timeout)
    code, output = qa.lep(
        ["login", "-c", "workspace:secret-token"],
        env={"LEPTON_WORKSPACE_TOKEN": "secret-token"},
        timeout=1,
    )
    assert code == 124
    assert output == "<redacted> partial logs"


def test_cli_receives_eof_instead_of_waiting_for_interactive_log_input(monkeypatch):
    def fake(command, **kwargs):
        assert kwargs.get("stdin") == subprocess.DEVNULL
        return subprocess.CompletedProcess(command, 0, "logs")

    monkeypatch.setattr(qa.subprocess, "run", fake)
    assert qa.lep(["log", "get", "--job", "job-id"], env={}, timeout=1) == (0, "logs")


def test_missing_or_malformed_log_summary_is_rejected():
    for value in (
        "",
        f"{qa.SUMMARY_BEGIN}\n{{}}",
        f"{qa.SUMMARY_BEGIN}\n[]\n{qa.SUMMARY_END}",
    ):
        with pytest.raises(ValueError):
            qa.extract_summary(value)
