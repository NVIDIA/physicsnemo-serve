"""CPU contracts for qualifying an exact Python release image on Lepton."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from qa.native_inference import pnms_http_job as job
from scripts import run_lepton_pnms_http_qa as controller


IMAGE = "registry/python@sha256:" + "c" * 64
SOURCE = "a" * 40


@pytest.fixture
def baked(tmp_path, monkeypatch):
    root = tmp_path / "pnms"
    plugin = root / "plugin"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "metadata:\n  id: python-release\nruntime:\n  profile: python-test\n"
    )
    (plugin / "workflow.py").write_text(
        "raise RuntimeError('QA must not import user code')\n"
    )
    (root / "python-launch.json").write_text(
        json.dumps({"workflow_id": "python-release"})
    )
    files = [
        {
            "path": path.relative_to(plugin).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(plugin.rglob("*"))
        if path.is_file()
    ]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setenv("SOURCE_SHA", SOURCE)
    return root, digest


@pytest.mark.parametrize("mutation", [None, "plugin", "source", "workflow"])
def test_python_image_target_verifies_baked_bytes_and_launch_identity(
    baked, monkeypatch, mutation
):
    root, digest = baked
    if mutation == "plugin":
        (root / "plugin/workflow.py").write_text("changed")
    elif mutation == "source":
        monkeypatch.setenv("SOURCE_SHA", "f" * 40)
    elif mutation == "workflow":
        (root / "python-launch.json").write_text('{"workflow_id":"foreign"}')
    options = dict(
        source_sha=SOURCE,
        runtime_identity=job._expected_identity(IMAGE),
        workflow="python-release",
        plugin_digest=digest,
        receipt_sha256="d" * 64,
    )
    if mutation:
        with pytest.raises(ValueError):
            job._python_image_target(root, **options)
    else:
        target = job._python_image_target(root, **options)
        assert target["runtime"] == "python"
        assert target["workflow"] == "python-release"
        assert target["plugin_digest"] == digest
        assert target["runtime_identity"] == {"kind": "registry-image", "value": IMAGE}


def release_receipt(path):
    document = {
        "status": "complete",
        "runtime": "python",
        "name": "python-release",
        "source_sha": SOURCE,
        "image": {
            "status": "complete",
            "registry_digest": IMAGE,
            "checks": {"contents": "passed", "python_base": "passed"},
            "contents": {
                "status": "passed",
                "workflow_id": "python-release",
                "plugin_digest": "b" * 64,
            },
        },
    }
    path.write_text(json.dumps(document))
    return document


def image_args(tmp_path):
    receipt = tmp_path / "build.json"
    release_receipt(receipt)
    return controller.build_parser().parse_args(
        [
            "--mode",
            "image",
            "--image-runtime",
            "python",
            "--python-build-receipt",
            str(receipt),
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--run-id",
            "python-image-unit",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--workspace-id",
            "unit",
            "--node-group",
            "unit",
            "--pull-secret",
            "unit",
            "--resource-shape",
            "gpu.h100-sxm",
            "--nfs-mount-base",
            "/mnt/shared",
        ]
    )


def test_python_image_receipt_keeps_plugin_id_with_independent_app_name(tmp_path):
    args = image_args(tmp_path)
    document = json.loads(args.python_build_receipt.read_text())
    document["name"] = "release-application"
    args.python_build_receipt.write_text(json.dumps(document))
    assert controller._python_release_receipt(args)["workflow"] == "python-release"


@pytest.mark.parametrize("field", ["name", "workflow_id"])
@pytest.mark.parametrize("value", [None, "", 123, True, [], {}])
def test_python_image_receipt_rejects_malformed_identity(tmp_path, field, value):
    args = image_args(tmp_path)
    document = json.loads(args.python_build_receipt.read_text())
    identity = document if field == "name" else document["image"]["contents"]
    identity[field] = value
    args.python_build_receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        controller._python_release_receipt(args)


@pytest.mark.parametrize("mutation", [None, "image", "plugin", "source", "failed"])
def test_python_image_controller_binds_public_build_receipt(tmp_path, mutation):
    args = image_args(tmp_path)
    if mutation:
        document = json.loads(args.python_build_receipt.read_text())
        if mutation == "image":
            document["image"]["registry_digest"] = "registry/foreign@sha256:" + "f" * 64
        elif mutation == "plugin":
            document["image"]["contents"]["plugin_digest"] = "not-a-digest"
        elif mutation == "source":
            document["source_sha"] = "f" * 40
        else:
            document["image"]["status"] = "failed"
        args.python_build_receipt.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            controller.validate_args(args)
        return
    controller.validate_args(args)
    command = controller.build_job_command(args, "unit-job")
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[runner.index("--image-runtime") + 1] == "python"
    assert runner[runner.index("--python-plugin-digest") + 1] == "b" * 64
    assert runner[runner.index("--python-workflow") + 1] == "python-release"
    assert runner[runner.index("--python-receipt-sha256") + 1] == job.sha256(
        args.python_build_receipt
    )
    assert "--include-python" not in runner


def test_python_image_report_cannot_substitute_another_plugin_or_receipt(tmp_path):
    args = image_args(tmp_path)
    target = {
        "name": "python.release",
        "runtime": "python",
        "workflow": "python-release",
        "source_sha": SOURCE,
        "plugin": "/opt/pnms/plugin",
        "plugin_digest": "b" * 64,
        "build_receipt_sha256": job.sha256(args.python_build_receipt),
        "runtime_identity": job._expected_identity(IMAGE),
    }
    names = job.expected_cases([target])
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "source_sha": SOURCE,
        "image_digest": IMAGE,
        "status": "passed",
        "qa_scope": "image-http",
        "targets": [target],
        "expected_cases": names,
        "cases": [
            {
                "name": name,
                "status": "passed",
                "details": {"runner_sha256": job.sha256(job.__file__)},
            }
            for name in names
        ],
    }
    report["cases"][0]["details"]["python_release"] = target
    controller.validate_report(report, args=args)
    for field in ("plugin_digest", "build_receipt_sha256"):
        foreign = deepcopy(report)
        foreign["targets"][0][field] = "f" * 64
        with pytest.raises(ValueError):
            controller.validate_report(foreign, args=args)


def test_python_image_starts_public_release_command_without_source_or_bootstrap(
    tmp_path, monkeypatch
):
    launches = []
    args = SimpleNamespace(
        mode="image",
        deployment_id=None,
        base_url=None,
        request_timeout=1,
        startup_timeout=1,
        shutdown_timeout=1,
        python=Path("python3"),
        repository=Path("/absent-source"),
        service_bin_dir=None,
    )
    ctx = SimpleNamespace(args=args, root=tmp_path, environment={})
    ctx.case = lambda name, function: (
        function() if name.endswith((".start", ".shutdown")) else {}
    )
    child = SimpleNamespace(pid=123)
    monkeypatch.setattr(
        job.subprocess,
        "Popen",
        lambda argv, **kwargs: launches.append((argv, kwargs)) or child,
    )
    monkeypatch.setattr(
        job,
        "OwnedService",
        lambda *a, **k: SimpleNamespace(close=lambda: {"stopped": True}),
    )
    target = {
        "name": "python.release",
        "runtime": "python",
        "workflow": "python-release",
        "runtime_identity": job._expected_identity(IMAGE),
    }
    job.run_target(ctx, target)
    assert launches[0][0] == [
        "/usr/bin/dumb-init",
        "--",
        "python3",
        "/opt/pnms/python_release.py",
        "start",
    ]
    assert launches[0][1]["cwd"] == Path("/app")
    assert launches[0][1]["env"]["HEALTH_STUB_ENABLED"] == "false"


def test_python_image_main_uses_baked_target_without_source_or_bootstrap(
    tmp_path, monkeypatch
):
    target = {
        "name": "python.release",
        "runtime": "python",
        "workflow": "python-release",
        "source_sha": SOURCE,
        "runtime_identity": job._expected_identity(IMAGE),
        "plugin": "/opt/pnms/plugin",
        "plugin_digest": "b" * 64,
        "build_receipt_sha256": "d" * 64,
    }
    monkeypatch.setattr(job, "_python_image_target", lambda *a, **k: target)
    monkeypatch.setattr(
        job, "_bundle_target", lambda *a, **k: pytest.fail("Loaded native bundle")
    )
    monkeypatch.setattr(
        job.subprocess, "run", lambda *a, **k: pytest.fail("Bootstrapped Python image")
    )

    def serve(ctx, selected):
        assert selected == target
        for operation in job.OPERATIONS:
            ctx.case(target["name"] + "." + operation, lambda: {})

    monkeypatch.setattr(job, "run_target", serve)
    output = tmp_path / "qa"
    assert (
        job.main(
            [
                "--mode",
                "image",
                "--image-runtime",
                "python",
                "--python-workflow",
                "python-release",
                "--python-plugin-digest",
                "b" * 64,
                "--python-receipt-sha256",
                "d" * 64,
                "--run-id",
                "unit-image",
                "--output",
                str(output),
                "--expected-source-sha",
                SOURCE,
                "--image-digest",
                IMAGE,
                "--repository",
                "/no-source",
            ]
        )
        == 0
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["cases"][0]["details"]["python_release"] == target
    assert len(report["cases"]) == 9
