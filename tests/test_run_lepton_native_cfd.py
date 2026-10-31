from __future__ import annotations

import copy
import json
import math
import shlex

import pytest
import yaml

from qa.scripts import run_lepton_native_cfd as qa


IMAGE = "registry.test/native-cfd@sha256:" + "a" * 64
SOURCE = "b" * 40


def args_for(tmp_path, monkeypatch, mode="compile", *extra):
    monkeypatch.delenv("LEPTON_WORKSPACE_TOKEN", raising=False)
    http = (
        [
            "--reference",
            "/outputs/qualified/reference/fields.json",
            "--input-root",
            "/outputs/qualified/inputs",
        ]
        if mode == "http"
        else []
    )
    return qa.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--run-id",
            "unit-cfd",
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "workspace",
            "--node-group",
            "node-group",
            "--pull-secret",
            "pull-secret",
            "--nfs-mount-base",
            "/mnt/shared",
            "--mode",
            mode,
            *http,
            *extra,
        ]
    )


def passed_report(mode):
    cases = qa.COMPILE_CASES if mode == "compile" else qa.HTTP_CASES
    report = {
        "schema_version": 1,
        "run_id": "unit-cfd",
        "source_sha": SOURCE,
        "image_digest": IMAGE,
        "status": "passed",
        "expected_cases": list(cases),
        "cases": [
            {"name": name, "status": "passed", "seconds": 0.25} for name in cases
        ],
    }
    if mode == "compile":
        report.update(
            qualification="standalone-persistent-runner",
            profile_id="physicsnemo-cfd-surface-transolver-run1-v1",
            serve_http_tested=False,
        )
    else:
        report.update(
            qualification="native-http-fields-and-metrics-v1",
            compiled_models_rebuilt=False,
        )
    return report


def validate(report, mode):
    qa.validate_summary(
        report, run_id="unit-cfd", source_sha=SOURCE, image_digest=IMAGE, profile=mode
    )


def historical(report, mode):
    begin, end = qa.MARKERS[mode]
    return "\n".join(
        "2026-10-21 16:00:00.000001|" + json.dumps(line)
        for line in (begin, report, end)
    )


@pytest.mark.parametrize("mode", ["compile", "http"])
def test_command_runs_exact_driver_with_one_gpu_zero_retries_and_16gib_shm(
    tmp_path, monkeypatch, mode
):
    args = args_for(tmp_path, monkeypatch, mode)
    qa.validate_args(args)
    command = qa.build_job_command(args, "unit-job")
    assert args.job_timeout == 7200
    for key, expected in (
        ("--num-workers", "1"),
        ("--shared-memory-size", "16384"),
        ("--max-failure-retry", "0"),
        ("--max-job-failure-retry", "0"),
    ):
        assert command[command.index(key) + 1] == expected
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[:5] == [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=60s",
        "7140s",
    ]
    assert runner[5:8] == ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m"]
    assert (
        runner[8] == "qa.scripts.run_native_cfd_job"
        if mode == "compile"
        else runner[8] == "qa.scripts.run_native_cfd_http_job"
    )
    assert runner[runner.index("--image") + 1] == IMAGE
    assert runner[runner.index("--source-sha") + 1] == SOURCE
    assert runner[runner.index("--command-timeout") + 1] == "7200"
    assert runner[runner.index("--output") + 1] == "/outputs/native-inference/unit-cfd"
    if mode == "compile":
        assert (
            runner[runner.index("--runtime") + 1]
            == "/opt/physicsnemo-inference/bin/physicsnemo-infer"
        )
        assert "--reference" not in runner
    else:
        assert runner[runner.index("--reference") + 1] == args.reference
        assert runner[runner.index("--input-root") + 1] == args.input_root
        assert "--runtime" not in runner


@pytest.mark.parametrize("mode", ["compile", "http"])
@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "image",
        "status",
        "missing",
        "duplicate",
        "reorder",
        "skipped",
        "nonfinite",
    ],
)
def test_exact_identity_and_every_ordered_gate_are_mandatory(mode, mutation):
    report = passed_report(mode)
    validate(report, mode)
    report = copy.deepcopy(report)
    if mutation == "source":
        report["source_sha"] = "c" * 40
    elif mutation == "image":
        report["image_digest"] = IMAGE.replace("a" * 64, "c" * 64)
    elif mutation == "status":
        report["status"] = "failed"
    elif mutation == "missing":
        report["cases"].pop()
    elif mutation == "duplicate":
        report["cases"].append(report["cases"][-1])
    elif mutation == "reorder":
        report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]
    elif mutation == "skipped":
        report["cases"][0]["status"] = "skipped"
    elif mutation == "nonfinite":
        report["cases"][0]["seconds"] = math.nan
    with pytest.raises(ValueError):
        validate(report, mode)


@pytest.mark.parametrize("mode", ["compile", "http"])
def test_historical_summary_is_validated_before_any_live_log_stream(
    tmp_path, monkeypatch, mode
):
    args = args_for(tmp_path, monkeypatch, mode)
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            spec = command[command.index("--file") + 1]
            assert json.loads(open(spec).read()) == {
                "max_failure_retry": 0,
                "max_job_failure_retry": 0,
            }
            return 0, "ID: unit-job-123\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Completed"}'
        if command[:2] == ["log", "get"]:
            return 0, historical(passed_report(mode), mode)
        if command[:2] == ["job", "log"]:
            pytest.fail("completed CFD job must use historical logs first")
        return 0, ""

    monkeypatch.setattr(qa.lifecycle, "lep", fake)
    assert qa.run(args) == 0
    assert [command[:2] for command in calls] == [
        ["job", "create"],
        ["job", "get"],
        ["log", "get"],
        ["job", "remove"],
    ]
    summary = json.loads((tmp_path / "unit-cfd/summary.json").read_text())
    assert summary["status"] == "passed"


def test_missing_history_has_finite_fallback_and_removes_only_owned_job(
    tmp_path, monkeypatch
):
    args = args_for(tmp_path, monkeypatch)
    calls, sleeps = [], []

    def fake(command, *, timeout, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: unit-job-123\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Failed"}'
        if command[:2] == ["job", "log"]:
            assert timeout <= 10
        return 0, "No complete logs\n"

    monkeypatch.setattr(qa.lifecycle, "lep", fake)
    monkeypatch.setattr(qa.lifecycle.time, "sleep", sleeps.append)
    assert qa.run(args) == 1
    assert sum(command[:2] == ["log", "get"] for command in calls) == 3
    assert calls[-2:] == [
        ["job", "log", "-i", "unit-job-123"],
        ["job", "remove", "-i", "unit-job-123"],
    ]
    assert sleeps == [5, 5]


@pytest.mark.parametrize(
    "extra",
    [
        ["--job-timeout", "7201"],
        ["--job-timeout", "60"],
        ["--runtime", "physicsnemo-infer"],
        ["--asset-cache", "/outside/assets"],
        ["--mode", "http"],
    ],
)
def test_invalid_contract_rejected_before_lepton(tmp_path, monkeypatch, extra):
    args = args_for(tmp_path, monkeypatch, "compile", *extra)
    monkeypatch.setattr(
        qa.lifecycle, "lep", lambda *a, **k: pytest.fail("invalid job launched")
    )
    with pytest.raises(ValueError):
        qa.run(args)


def test_http_paths_cannot_escape_remote_mount(tmp_path, monkeypatch):
    args = args_for(
        tmp_path, monkeypatch, "http", "--reference", "/outputs/../outside/fields.json"
    )
    with pytest.raises(ValueError):
        qa.validate_args(args)


def test_container_term_and_kill_are_both_inside_requested_deadline(
    tmp_path, monkeypatch
):
    args = args_for(tmp_path, monkeypatch, "compile", "--job-timeout", "601.8")
    qa.validate_args(args)
    command = qa.build_job_command(args, "unit-job")
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[3] == "--kill-after=60s"
    assert runner[4] == "541s"
    assert 541 + 60 <= args.job_timeout
    assert runner[runner.index("--command-timeout") + 1] == "601"


def test_http_cannot_lose_native_artifacts_on_unmounted_outputs(tmp_path, monkeypatch):
    args = args_for(
        tmp_path,
        monkeypatch,
        "http",
        "--mount-target",
        "/shared",
        "--reference",
        "/shared/reference/fields.json",
        "--input-root",
        "/shared/inputs",
    )
    with pytest.raises(ValueError, match="persistent /outputs"):
        qa.validate_args(args)


def test_dry_run_records_reviewable_cfd_job_without_network(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch, "compile", "--dry-run")
    monkeypatch.setattr(
        qa.lifecycle, "lep", lambda *a, **k: pytest.fail("dry run contacted Lepton")
    )
    assert qa.run(args) == 0
    evidence = tmp_path / "unit-cfd"
    summary = json.loads((evidence / "summary.json").read_text())
    command = json.loads((evidence / "job-command.json").read_text())["argv"]
    assert summary["status"] == "dry-run"
    assert "--shared-memory-size" in command
    assert "qa.scripts.run_native_cfd_job" in command[command.index("--command") + 1]


def test_manual_workflow_exposes_modes_and_reuses_existing_secret_configuration():
    workflow = yaml.safe_load(
        (qa.lifecycle.REPO_ROOT / ".github/workflows/lepton_native_cfd.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["mode"]["options"] == ["compile", "http"]
    assert inputs["job_timeout"]["default"] == 7200
    assert inputs["image"]["required"] and inputs["source_sha"]["required"]
    assert {"reference", "input_root"} <= inputs.keys()
    steps = workflow["jobs"]["qualify"]["steps"]
    execution = next(
        step for step in steps if step.get("name") == "Qualify full native CFD"
    )
    assert (
        execution["env"]["LEPTON_WORKSPACE_TOKEN"]
        == "${{ secrets.LEPTON_WORKSPACE_TOKEN }}"
    )
    assert execution["env"]["LEPTON_PULL_SECRET"] == "${{ secrets.PULL_SECRET }}"
    assert "run_lepton_native_cfd.py" in execution["run"]
    assert (
        '--reference "$CFD_REFERENCE" --input-root "$CFD_INPUT_ROOT"'
        in execution["run"]
    )
    assert any(
        step.get("if") == "always()" and "upload-artifact" in step.get("uses", "")
        for step in steps
    )
