"""The CPU component contract cannot silently omit required remote execution."""

import copy
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from qa.native_inference import components
from qa.scripts import run_lepton_native_components as launcher


IMAGE = "registry.example/native-components@sha256:" + "a" * 64
SOURCE = "b" * 40
IDENTITY = {
    "run_id": "component-unit",
    "source_sha": SOURCE,
    "image_digest": IMAGE,
    "profile": "full",
}


def passed_report():
    return {
        "schema_version": 1,
        **IDENTITY,
        "qualification": components.QUALIFICATION,
        "status": "passed",
        "execution": {
            "provider": "lepton",
            "resource_shape": "cpu.medium",
            "cpu_only": True,
        },
        "expected_cases": components.expected_cases(),
        "cases": [
            {
                "name": name,
                "status": "passed",
                "seconds": 0.1,
                **(
                    {"counts": {"tests": 2, "failed": 0, "skipped": 0}}
                    if name in components.TEST_CASES
                    else {}
                ),
            }
            for name in components.expected_cases()
        ],
    }


def test_complete_component_inventory_passes():
    components.validate_summary(passed_report(), **IDENTITY)


def test_component_summary_rejects_local_execution_provider():
    report = passed_report()
    report["execution"]["provider"] = "local"
    with pytest.raises(ValueError, match="Lepton|lepton|provider"):
        components.validate_summary(report, **IDENTITY)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "order",
        "duplicate",
        "skip",
        "wrong-source",
        "empty-tests",
        "skipped-tests",
        "nonfinite",
    ],
)
def test_component_summary_rejects_incomplete_or_different_evidence(change):
    report = passed_report()
    if change == "missing":
        report["cases"].pop()
    elif change == "order":
        report["cases"].reverse()
    elif change == "duplicate":
        report["cases"][-1] = copy.deepcopy(report["cases"][0])
    elif change == "skip":
        report["cases"][0]["status"] = "skipped"
    elif change == "wrong-source":
        report["source_sha"] = "c" * 40
    elif change == "empty-tests":
        report["cases"][1]["counts"]["tests"] = 0
    elif change == "skipped-tests":
        report["cases"][1]["counts"]["skipped"] = 1
    else:
        report["cases"][0]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        components.validate_summary(report, **IDENTITY)


@pytest.mark.parametrize("status", ["<skipped/>", "<failure/>", "<error/>"])
def test_junit_cannot_count_skips_or_failures_as_component_pass(tmp_path, status):
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuite><testcase name="actual">' + status + "</testcase></testsuite>"
    )
    with pytest.raises(ValueError, match="without failures or skips"):
        components.junit_counts(report)


def test_rust_ignored_tests_cannot_satisfy_required_component_scope(tmp_path):
    report = tmp_path / "rust.log"
    report.write_text(
        "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
    )
    with pytest.raises(ValueError, match="without failures or skips"):
        components.rust_counts(report)


def test_cpu_launcher_uses_finite_lepton_command_and_no_test_selector(tmp_path):
    args = launcher.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--run-id",
            IDENTITY["run_id"],
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "workspace",
            "--node-group",
            "nodes",
            "--pull-secret",
            "pull",
            "--nfs-mount-base",
            "/PhysicsNeMo/platform",
        ]
    )
    launcher.validate_args(args)
    assert args.resource_shape == "cpu.medium" and args.profile == "full"
    command = launcher.build_job_command(args, "component-job")
    assert command[:2] == ["job", "create"]
    assert command[command.index("--num-workers") + 1] == "1"
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[:2] == ["/bin/sh", "-c"]
    assert "cd /workspace && exec /usr/bin/timeout" in runner[2]
    assert "--kill-after=60s 7140s" in runner[2]
    assert "-m qa.native_inference.components" in runner[2]
    assert "/opt/nvidia/nvidia_entrypoint.sh" not in runner[2]
    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(
            [
                "--image",
                IMAGE,
                "--expected-source-sha",
                SOURCE,
                "--tests",
                "selected.py",
            ]
        )


def test_component_log_markers_recover_original_report():
    report = passed_report()
    text = (
        components.SUMMARY_BEGIN
        + "\n"
        + json.dumps(report)
        + "\n"
        + components.SUMMARY_END
    )
    assert launcher.extract_summary(text) == report


def test_native_python_inventory_covers_existing_launcher_export_and_raw_variant_suites(
    tmp_path,
):
    captured = []

    def command(name, argv):
        assert name == "python-native"
        captured.extend(str(value) for value in argv)
        report = next(value for value in captured if value.startswith("--junitxml="))
        Path(report.removeprefix("--junitxml=")).write_text(
            '<testsuite><testcase name="executed"/></testsuite>'
        )

    components.python_native(SimpleNamespace(reports=tmp_path, command=command))
    required = {
        "tests/test_run_lepton_native_inference_qa.py",
        "tests/test_run_lepton_native_cfd.py",
        "tests/test_export_native_cfd_release.py",
        "qa/scripts/test_create_native_raw_variant.py",
    }
    selected = {
        str(Path(value).relative_to(components.REPO_ROOT))
        for value in captured
        if value.endswith(".py") and Path(value).is_absolute()
    }
    assert required <= selected, (
        "native Python QA omitted existing suites: "
        + ", ".join(sorted(required - selected))
    )


def test_native_rust_inventory_runs_worker_runtime_tests_explicitly(tmp_path):
    captured = []

    def command(name, argv):
        assert name == "rust-native"
        captured.extend(str(value) for value in argv)
        report = tmp_path / "rust.log"
        report.write_text(
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
        )
        return report

    components.rust_native(SimpleNamespace(command=command))
    packages = {
        captured[index + 1]
        for index, value in enumerate(captured[:-1])
        if value == "-p"
    }
    assert "worker-runtime" in packages, (
        "native Rust QA omitted dependency crate worker-runtime test inventory"
    )


@pytest.mark.parametrize("missing_required_suite", [False, True])
def test_native_python_inventory_is_sealed_to_carried_suites(
    tmp_path, monkeypatch, missing_required_suite
):
    source = tmp_path / "source"
    carried = {
        str(path.relative_to(components.REPO_ROOT))
        for path in (components.REPO_ROOT / "tests").glob("test_native*.py")
    } | {
        "tests/test_run_lepton_native_inference_qa.py",
        "tests/test_run_lepton_native_cfd.py",
        "tests/test_export_native_cfd_release.py",
        "qa/scripts/test_create_native_raw_variant.py",
    }
    for name in carried | {"tests/test_native_managed.py"}:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# inventory fixture\n")
    if missing_required_suite:
        (source / "tests/test_native_release.py").unlink()
    monkeypatch.setattr(components, "REPO_ROOT", source)
    reports = tmp_path / "reports"
    reports.mkdir()
    captured = []

    def command(name, argv):
        captured.extend(str(value) for value in argv)
        report = next(value for value in captured if value.startswith("--junitxml="))
        Path(report.removeprefix("--junitxml=")).write_text(
            '<testsuite><testcase name="executed"/></testsuite>'
        )

    ctx = SimpleNamespace(reports=reports, command=command)
    if missing_required_suite:
        with pytest.raises(ValueError, match="inventory is empty or incomplete"):
            components.python_native(ctx)
        assert not captured
    else:
        components.python_native(ctx)
        inventory = json.loads((reports / "python-inventory.json").read_text())
        assert {item["path"] for item in inventory} == carried
