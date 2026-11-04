from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native-inference/deploy"))
import native_release_lepton as lepton  # noqa: E402
from native_release import read_json, write_json  # noqa: E402


def test_controller_captures_node_before_cleanup_and_bounds_actual_command(
    tmp_path, monkeypatch
):
    real = lepton._lifecycle()
    state = {"alive": True}
    image = "registry/image@sha256:" + "b" * 64
    cases = ["preflight", "seal"]
    report = {
        "schema_version": 1,
        "status": "passed",
        "run_id": "unit-release",
        "source_sha": "a" * 40,
        "image_digest": image,
        "expected_cases": cases,
        "cases": [{"name": name, "status": "passed", "seconds": 0.1} for name in cases],
    }

    def lep(argv, **kwargs):
        assert state["alive"], "node placement must be captured while job exists"
        assert argv == ["job", "nodes", "-i", "job-1"]
        return 0, json.dumps(["node-1"])

    def run(args, **hooks):
        root = args.artifact_dir / args.run_id
        root.mkdir(parents=True)
        write_json(
            root / "summary.json",
            {
                "job_id": "job-1",
                "status": "passed",
                "cleanup_errors": [],
                "remote_artifacts": "/outputs/native-inference/unit-release",
            },
        )
        command = hooks["command_builder"](args, "name")
        state["command"] = command
        hooks["summary_validator"](
            report,
            required_cases=cases,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image,
            profile=args.profile,
        )
        state["alive"] = False
        write_json(root / "job-summary.json", report)
        return 0

    fake = SimpleNamespace(
        build_parser=real.build_parser,
        validate_args=real.validate_args,
        build_job_command=lambda *args: [
            "lep",
            "job",
            "create",
            "--command",
            "placeholder",
        ],
        run=run,
        lep=lep,
    )
    monkeypatch.setattr(lepton, "_lifecycle", lambda: fake)

    # validate_summary receives required_cases from the wrapper, not lifecycle.
    def run_no_duplicate(args, **hooks):
        original = hooks["summary_validator"]
        hooks["summary_validator"] = lambda report, **kwargs: original(
            report,
            **{key: value for key, value in kwargs.items() if key != "required_cases"},
        )
        return run(args, **hooks)

    fake.run = run_no_duplicate
    result = lepton.run_job(
        image=image,
        source_sha="a" * 40,
        run_id="unit-release",
        mode="compile",
        required_cases=cases,
        artifact_dir=tmp_path,
        config={
            "workspace_id": "workspace",
            "node_group": "nodes",
            "pull_secret": "pull",
            "resource_shape": "gpu.h100-sxm",
            "nfs_mount_base": "/PhysicsNeMo/platform",
            "lustre_dir": "qa",
            "lustre_storage": "lustre",
            "job_timeout": 7200,
        },
        node_id="node-1",
    )
    assert result["controller"]["node_id"] == "node-1"
    assert not state["alive"]
    command = state["command"]
    actual = command[command.index("--command") + 1]
    assert "/usr/bin/timeout --signal=TERM --kill-after=60s 7140s" in actual
    assert "native_release_job.py" in actual
    assert command[-4:] == ["--shared-memory-size", "16384", "--node-id", "node-1"]
    assert (
        read_json(tmp_path / "unit-release/job-placement.json")["node_id"] == "node-1"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(status="failed"),
        lambda report: report["cases"].reverse(),
        lambda report: report["cases"][0].update(status="skipped"),
        lambda report: report.update(image_digest="other"),
        lambda report: report["cases"][0].update(seconds=float("nan")),
    ],
)
def test_release_summary_rejects_not_executed_or_different_job(mutation):
    report = {
        "schema_version": 1,
        "status": "passed",
        "run_id": "run",
        "source_sha": "a" * 40,
        "image_digest": "registry@sha256:" + "b" * 64,
        "expected_cases": ["first", "last"],
        "cases": [
            {"name": name, "status": "passed", "seconds": 0.1}
            for name in ["first", "last"]
        ],
    }
    mutation(report)
    with pytest.raises(ValueError):
        lepton.validate_summary(
            report,
            required_cases=["first", "last"],
            run_id="run",
            source_sha="a" * 40,
            image_digest="registry@sha256:" + "b" * 64,
            profile="unused",
        )


def test_artifact_proxy_is_only_a_sensitive_curl_argument(monkeypatch, tmp_path):
    proxy = "socks5h://127.0.0.1:58443"
    monkeypatch.setenv("NATIVE_RELEASE_ARTIFACT_PROXY", proxy)
    calls = []

    class Runner:
        def run(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0, stdout="200")

    assert (
        lepton._curl(
            Runner(), "https://private.example/bundle.tar", tmp_path / "bundle.tar"
        )
        == 200
    )
    argv, options = calls[0]
    assert "--proxy" in argv, (
        "artifact transport must honor the explicit authenticated TLS route"
    )
    assert argv[argv.index("--proxy") + 1] == proxy
    assert options["sensitive"] is True
