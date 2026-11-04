"""Fresh-venv handoff contracts; no local GPU or live HTTP execution."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qa.native_inference import pnms_http_job as job
from scripts import run_lepton_pnms_http_qa as controller
from test_pnms_venv_qa import passed_report
from test_pnms_http_qa import local_report

SOURCE = "a" * 40
IMAGE = "registry.example/qa@sha256:" + "b" * 64


def venv_summary(mount):
    report = passed_report()
    root = mount / "native-inference" / report["run_id"]
    check = next(
        case["details"] for case in report["cases"] if case["name"] == "venv.verify"
    )
    check.update(python=str(root / "venv/bin/python"), prefix=str(root / "venv"))
    for candidate in report["release_candidates"]:
        candidate["build_receipt"] = str(
            root
            / f"relocated/{candidate['backend']}/.pnms/builds/{candidate['build_id']}/build.json"
        )
    path = root / "summary.json"
    job.write_json(path, report)
    return path, report


def test_http_accepts_two_backend_fresh_venv_handoff_and_records_selected_python(
    tmp_path,
):
    path, report = venv_summary(tmp_path)
    candidates, evidence = job.local_candidates(
        path, mount_root=tmp_path, source_sha=SOURCE, image_digest=IMAGE
    )
    assert len(candidates) == 2
    assert evidence["profile"] == "fresh-venv"
    assert evidence["python"] == str(path.parent / "venv/bin/python")
    assert evidence["sha256"] == job.sha256(path)


@pytest.mark.parametrize(
    "mutation", ["global-python", "global-prefix", "subset", "other-template"]
)
def test_fresh_handoff_rejects_global_environment_or_wrong_bundle_inventory(
    tmp_path, mutation
):
    path, report = venv_summary(tmp_path)
    check = next(
        case["details"] for case in report["cases"] if case["name"] == "venv.verify"
    )
    if mutation == "global-python":
        check["python"] = "/usr/bin/python3"
    elif mutation == "global-prefix":
        check["prefix"] = "/usr"
    elif mutation == "subset":
        report["release_candidates"].pop()
    else:
        report["release_candidates"][1]["template"] = "tensor"
    job.write_json(path, report)
    with pytest.raises(ValueError):
        job.local_candidates(
            path, mount_root=tmp_path, source_sha=SOURCE, image_digest=IMAGE
        )


def test_fresh_venv_support_does_not_relax_existing_five_bundle_profile(tmp_path):
    report = local_report(tmp_path)
    report["release_candidates"] = report["release_candidates"][:2]
    path = tmp_path / "native-inference/local-run/summary.json"
    job.write_json(path, report)
    with pytest.raises(ValueError):
        job.local_candidates(
            path,
            mount_root=tmp_path,
            source_sha=SOURCE,
            image_digest=report["image_digest"],
        )


def http_report(path, candidates):
    targets = [
        {
            "name": f"native.{item['backend']}.{item['template']}",
            "runtime": "cpp",
            "source_sha": SOURCE,
            "runtime_identity": {"kind": "bundle", "value": item["bundle_digest"]},
            "ingress": "application/json",
            **item,
        }
        for item in candidates
    ]
    evidence = {
        "path": path.as_posix(),
        "source_sha": SOURCE,
        "image_digest": IMAGE,
        "sha256": "e" * 64,
        "profile": "fresh-venv",
        "python": (path.parent / "venv/bin/python").as_posix(),
        "candidates": candidates,
    }
    names = job.expected_cases(targets)
    return {
        "schema_version": 1,
        "run_id": "venv-http",
        "source_sha": SOURCE,
        "image_digest": IMAGE,
        "status": "passed",
        "qa_scope": "host-http",
        "targets": targets,
        "input_local_summary": evidence,
        "expected_cases": names,
        "cases": [
            {
                "name": name,
                "status": "passed",
                "details": {
                    "runner_sha256": job.sha256(job.__file__),
                    "python": evidence["python"],
                    "input_profile": "fresh-venv",
                },
            }
            for name in names
        ],
    }


def test_http_controller_accepts_exact_two_fresh_venv_candidates(tmp_path):
    path, document = venv_summary(tmp_path)
    report = http_report(path, document["release_candidates"])
    args = SimpleNamespace(
        mode="host",
        image_runtime="cpp",
        include_python=False,
        expected_source_sha=SOURCE,
        image=IMAGE,
        local_summary=path.as_posix(),
        run_id="venv-http",
        python=(path.parent / "venv/bin/python").as_posix(),
    )
    controller.validate_report(report, args=args)
    assert len(report["expected_cases"]) == 17
    for mutation in ("subset", "wrong-profile", "global-python"):
        changed = copy.deepcopy(report)
        if mutation == "subset":
            changed["input_local_summary"]["candidates"].pop()
        elif mutation == "wrong-profile":
            changed["input_local_summary"]["profile"] = "local"
        else:
            changed["cases"][0]["details"]["python"] = "/usr/bin/python3"
        with pytest.raises(ValueError):
            controller.validate_report(changed, args=args)


def test_fresh_http_environment_selects_recorded_python_and_rebuilds_service(tmp_path):
    path = tmp_path / "native-inference/venv-unit/summary.json"
    python = path.parent / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("test executable")
    evidence = {"profile": "fresh-venv", "path": str(path), "python": str(python)}
    args = SimpleNamespace(python=Path("python3"), service_bin_dir=None)
    ctx = SimpleNamespace(
        args=args,
        environment={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/image/python",
            "LD_LIBRARY_PATH": os.pathsep.join(
                ["/image/site-packages/torch/lib", "/usr/local/cuda/compat"]
            ),
            "PNMS_SERVICE_BIN_DIR": "/old/bin",
            "CARGO_TARGET_DIR": "/old/target",
        },
    )
    job.configure_local_environment(ctx, evidence)
    assert args.python == python
    assert "PYTHONPATH" not in ctx.environment
    assert ctx.environment["LD_LIBRARY_PATH"] == "/usr/local/cuda/compat"
    assert (
        "PNMS_SERVICE_BIN_DIR" not in ctx.environment
        and "CARGO_TARGET_DIR" not in ctx.environment
    )
    assert ctx.environment["VIRTUAL_ENV"] == str(path.parent / "venv")
    assert ctx.environment["PYO3_PYTHON"] == str(python)
    ctx.args.service_bin_dir = Path("/prebuilt")
    with pytest.raises(ValueError, match="fresh|prebuilt"):
        job.configure_local_environment(ctx, evidence)


def test_http_controller_disables_initial_driver_bytecode(tmp_path, monkeypatch):
    import shlex
    from test_pnms_http_qa import controller_args

    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--local-summary",
        "/outputs/native-inference/venv-unit/summary.json",
        "--python",
        "/outputs/native-inference/venv-unit/venv/bin/python",
    )
    command = controller.build_job_command(args, "unit-job")
    remote = shlex.split(command[command.index("--command") + 1])
    assert remote[remote.index(args.python) + 1] == "-B"
