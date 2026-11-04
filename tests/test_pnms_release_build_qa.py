"""Public build execution and compact verified release handoff, without GPUs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
sys.path.insert(0, str(ROOT / "native-inference/deploy"))
from pnms_app.common import digest, file_identity, inventory, write_json  # noqa: E402
from native_release_artifacts import extract  # noqa: E402
from qa.native_inference import pnms_release_build_job as job  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402

SOURCE = "a" * 40
IMAGE = "registry.example/build@sha256:" + "b" * 64
RUNTIME = "registry.example/runtime@sha256:" + "c" * 64
SOURCE_INVENTORY_BYTES = (
    '{"source.py": {"sha256": "'
    + "e" * 64
    + '", "size_bytes": 3, "mode": 420, "symlink": null}}\n'
).encode()


def built_project(tmp_path):
    project = tmp_path / "original"
    project.mkdir()
    (project / "pnms.yaml").write_text(
        "schema_version: 1\nname: pilot\nruntime: cpp\n"
        "models:\n  primary:\n    project: model/model-build.json\n"
        "build:\n  backend: aoti\nworkflow:\n  template: tensor\n"
        "serve:\n  image: " + RUNTIME + "\n"
    )
    root = project / ".pnms/builds" / ("d" * 32)
    bundle = root / "bundle"
    runner = bundle / "workflow/bin/pnms-native-run"
    runner.parent.mkdir(parents=True)
    runner = python_executable(runner, "raise SystemExit(0)\n")
    write_json(
        bundle / "bundle.json",
        {
            "schema_version": 1,
            "name": "pilot",
            "workflow": {"runner": runner.relative_to(bundle).as_posix()},
        },
    )
    files = inventory(bundle)
    bundle_digest = digest(files)
    write_json(
        bundle / "inventory.json",
        {
            "schema_version": 1,
            "files": files,
            "bundle_digest": bundle_digest,
        },
    )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "runtime": "cpp",
        "name": "pilot",
        "source_sha": SOURCE,
        "build_id": root.name,
        "bundle": "bundle",
        "bundle_digest": bundle_digest,
        "checks": {"model": "passed", "application": "passed", "http": "not_run"},
    }
    write_json(root / "build.json", receipt)
    for path in (
        project / "model/weights.pt",
        root / "model-builds/primary/weights.pt",
        project / ".pnms/cache/sdk/lib.so",
        project / ".pnms/env/bin/python",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private or duplicate producer input")
    return project, root / "build.json", receipt


def provenance():
    return dict(
        run_id="release-unit",
        source_sha=SOURCE,
        image_digest=IMAGE,
        runtime_image=RUNTIME,
        source_inventory_sha256=hashlib.sha256(SOURCE_INVENTORY_BYTES).hexdigest(),
    )


def frozen_inventory(tmp_path):
    path = tmp_path / "frozen-source-inventory.json"
    path.write_bytes(SOURCE_INVENTORY_BYTES)
    return path


def test_transport_retains_exact_original_source_inventory_and_local_path(tmp_path):
    project, receipt, _ = built_project(tmp_path)
    source_inventory = frozen_inventory(tmp_path)
    original = source_inventory.read_bytes()
    output = tmp_path / "remote"
    output.mkdir()
    manifest = job.seal_build(
        project, receipt, output, source_inventory=str(source_inventory), **provenance()
    )
    assert "source-inventory.json" in {item["path"] for item in manifest["files"]}, (
        "compact transport must include the original verified source inventory"
    )
    extracted = extract(output / "bundle.tar", manifest, tmp_path / "downloaded")
    restored = job.restore_handoff(extracted, manifest)
    local_inventory = Path(restored["source_inventory"])
    assert local_inventory == extracted / "source-inventory.json"
    assert local_inventory.read_bytes() == original
    assert restored["source_inventory_identity"] == file_identity(source_inventory)
    assert source_inventory.read_bytes() == original


def test_release_handoff_uses_the_sealed_domain_runner(tmp_path):
    project, receipt, record = built_project(tmp_path)
    bundle = receipt.parent / "bundle"
    descriptor = json.loads((bundle / "bundle.json").read_text())
    runner = bundle / descriptor["workflow"]["runner"]
    domain_runner = runner.with_name("physicsnemo-transolver" + runner.suffix)
    runner.rename(domain_runner)
    descriptor["workflow"]["runner"] = domain_runner.relative_to(bundle).as_posix()
    write_json(bundle / "bundle.json", descriptor)
    files = inventory(bundle, exclude=("inventory.json",))
    record["bundle_digest"] = digest(files)
    write_json(
        bundle / "inventory.json",
        {
            "schema_version": 1,
            "files": files,
            "bundle_digest": record["bundle_digest"],
        },
    )
    write_json(receipt, record)
    accepted, location = job.check_build(
        project, receipt, source_sha=SOURCE, runtime_image=RUNTIME
    )
    assert location == bundle and accepted["build_id"] == record["build_id"]
    output = tmp_path / "handoff"
    output.mkdir()
    manifest = job.seal_build(
        project,
        receipt,
        output,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )
    extracted = extract(output / "bundle.tar", manifest, tmp_path / "restored")
    restored = job.restore_handoff(extracted, manifest)
    sealed_runner = (
        Path(restored["build_receipt"]).parent
        / "bundle"
        / descriptor["workflow"]["runner"]
    )
    assert sealed_runner.stat().st_mode & 0o111


def test_legacy_transport_projects_a_descriptor_without_changing_authored_inputs(
    tmp_path,
):
    import yaml

    project, receipt, record = built_project(tmp_path)
    document = yaml.safe_load((project / "pnms.yaml").read_text())
    document["_legacy_native"] = {"manifest": "plugin.yaml", "profile": "release.yaml"}
    record["application"] = document
    write_json(receipt, record)
    (project / "pnms.yaml").unlink()
    (project / "plugin.yaml").write_text("original authored plugin\n")
    original = receipt.read_bytes()
    output = tmp_path / "remote"
    output.mkdir()

    manifest = job.seal_build(
        project,
        receipt,
        output,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )

    extracted = extract(output / "bundle.tar", manifest, tmp_path / "downloaded")
    projected = yaml.safe_load((extracted / "project/pnms.yaml").read_text())
    assert projected["name"] == "pilot" and projected["runtime"] == "cpp"
    assert projected["serve"]["image"] == RUNTIME
    assert "_legacy_native" not in projected
    assert not (project / "pnms.yaml").exists()
    assert (project / "plugin.yaml").read_text() == "original authored plugin\n"
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_sealing_rejects_missing_or_changed_original_source_inventory(
    tmp_path, failure
):
    project, receipt, _ = built_project(tmp_path)
    source_inventory = frozen_inventory(tmp_path)
    if failure == "missing":
        source_inventory.unlink()
    else:
        source_inventory.write_bytes(SOURCE_INVENTORY_BYTES + b" ")
    output = tmp_path / "remote"
    output.mkdir()
    with pytest.raises(ValueError, match="source inventory|regular file"):
        job.seal_build(
            project,
            receipt,
            output,
            source_inventory=str(source_inventory),
            **provenance(),
        )
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_restore_rejects_source_inventory_changed_after_download(tmp_path, failure):
    project, receipt, _ = built_project(tmp_path)
    output = tmp_path / "remote"
    output.mkdir()
    manifest = job.seal_build(
        project,
        receipt,
        output,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )
    extracted = extract(output / "bundle.tar", manifest, tmp_path / "downloaded")
    source_inventory = extracted / "source-inventory.json"
    if failure == "missing":
        source_inventory.unlink()
    else:
        source_inventory.write_bytes(source_inventory.read_bytes() + b" ")
    with pytest.raises(ValueError):
        job.restore_handoff(extracted, manifest)


def test_compact_transport_preserves_receipt_bundle_and_executable_runner(tmp_path):
    project, receipt, record = built_project(tmp_path)
    original = receipt.read_bytes()
    output = tmp_path / "remote"
    output.mkdir()
    manifest = job.seal_build(
        project,
        receipt,
        output,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )
    paths = {item["path"] for item in manifest["files"]}
    assert "project/pnms.yaml" in paths
    assert not any(
        "weights" in name
        or "/cache/" in name
        or "/env/" in name
        or "model-builds" in name
        for name in paths
    )
    extracted = extract(output / "bundle.tar", manifest, tmp_path / "downloaded")
    local = job.restore_handoff(extracted, manifest)
    assert Path(local["build_receipt"]).read_bytes() == original
    assert local["build_receipt_identity"] == file_identity(receipt)
    assert local["build_id"] == record["build_id"]
    assert local["bundle_digest"] == record["bundle_digest"]
    local_bundle = Path(local["build_receipt"]).parent / "bundle"
    runner = (
        local_bundle
        / json.loads((local_bundle / "bundle.json").read_text())["workflow"]["runner"]
    )
    assert runner.stat().st_mode & 0o111, "the shared extractor drops executable mode"
    assert local["project"] == str(extracted / "project")
    assert receipt.read_bytes() == original


@pytest.mark.parametrize(
    "failure", ["incomplete", "foreign-source", "bad-bundle", "wrong-image"]
)
def test_compact_handoff_rejects_failed_or_unbound_inputs(tmp_path, failure):
    project, receipt, record = built_project(tmp_path)
    if failure == "incomplete":
        record["checks"]["application"] = "not_run"
    elif failure == "foreign-source":
        record["source_sha"] = "f" * 40
    elif failure == "bad-bundle":
        (receipt.parent / "bundle/bundle.json").write_text("changed")
    else:
        (project / "pnms.yaml").write_text(
            (project / "pnms.yaml").read_text().replace(RUNTIME, IMAGE)
        )
    write_json(receipt, record)
    output = tmp_path / "remote"
    output.mkdir()
    with pytest.raises(ValueError):
        job.seal_build(
            project,
            receipt,
            output,
            source_inventory=frozen_inventory(tmp_path),
            **provenance(),
        )
    assert not (output / "manifest.json").exists()


def test_transport_rejects_sealed_executable_traversal(tmp_path):
    project, receipt, _ = built_project(tmp_path)
    output = tmp_path / "remote"
    output.mkdir()
    manifest = job.seal_build(
        project,
        receipt,
        output,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )
    extracted = extract(output / "bundle.tar", manifest, tmp_path / "downloaded")
    handoff = json.loads((extracted / "handoff.json").read_text())
    handoff["executables"] = ["../outside"]
    write_json(extracted / "handoff.json", handoff)
    with pytest.raises(ValueError):
        job.restore_handoff(extracted, manifest)


def test_generated_pilot_uses_public_cli_local_build_and_own_descriptor(tmp_path):
    calls = []
    args = SimpleNamespace(
        project=None,
        backend="tensorrt",
        template=None,
        runtime_image=RUNTIME,
        python="/image/bin/python",
        sdk_prefix=Path("/installed/sdk"),
    )
    context = SimpleNamespace(args=args, root=tmp_path, repository=ROOT)

    def pnms(label, *arguments):
        calls.append([str(part) for part in arguments])
        if arguments[0] == "init":
            app = Path(arguments[1])
            app.mkdir()
            (app / "pnms.yaml").write_text(
                "schema_version: 1\nname: pilot\nruntime: cpp\nmodels: {}\n"
                "build:\n  backend: tensorrt\nworkflow:\n  template: custom\n"
            )
            return {"status": "initialized"}
        descriptor = Path(arguments[1]) / "pnms.yaml"
        import yaml

        assert yaml.safe_load(descriptor.read_text())["serve"]["image"] == RUNTIME
        return {"status": "complete", "root": str(tmp_path / "build")}

    context.pnms = pnms
    project, receipt = job.build_application(context)
    assert calls[0][0] == "init" and "cpp" in calls[0]
    assert "--repository" in calls[0], (
        "public init must bind the verified source explicitly"
    )
    assert calls[0][calls[0].index("--repository") + 1] == str(ROOT), (
        "An inherited PNMS_REPOSITORY must not override the verified frozen source"
    )
    assert calls[0][calls[0].index("--template") + 1] == "custom"
    command = calls[1]
    assert command[:2] == ["build", str(project)]
    assert command[command.index("--environment") + 1] == "local"
    assert command[command.index("--python") + 1] == args.python
    assert command[command.index("--sdk-prefix") + 1] == str(args.sdk_prefix)
    assert command[command.index("--repository") + 1] == str(ROOT)
    assert "--image" not in command and "docker" not in command
    assert "--gpu" not in command, (
        "the public build must require an unambiguous allocation"
    )
    assert receipt == tmp_path / "build/build.json"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows chmod cannot remove POSIX executable permission bits",
)
def test_sealing_rejects_runner_whose_executable_mode_was_removed(tmp_path):
    project, receipt, _ = built_project(tmp_path)
    (receipt.parent / "bundle/workflow/bin/pnms-native-run").chmod(0o644)
    output = tmp_path / "remote"
    output.mkdir()
    with pytest.raises(ValueError, match="executable"):
        job.seal_build(
            project,
            receipt,
            output,
            source_inventory=frozen_inventory(tmp_path),
            **provenance(),
        )
    assert not (output / "manifest.json").exists()


def test_supplied_project_uses_prepared_image_interpreter_and_sdk(tmp_path):
    import yaml

    source = tmp_path / "owned-source"
    project = source / "apps/pilot"
    project.mkdir(parents=True)
    descriptor = project / "pnms.yaml"
    descriptor.write_text(
        "schema_version: 1\nname: pilot\nruntime: cpp\n"
        "build:\n  backend: aoti\n  venv: /producer/venv\n"
        "  runtime: /producer/physicsnemo-infer\n"
    )
    args = SimpleNamespace(
        project="apps/pilot",
        runtime_image=RUNTIME,
        python="/prepared/python",
        sdk_prefix=Path("/prepared/sdk"),
    )
    calls = []

    def pnms(label, *arguments):
        calls.append([str(value) for value in arguments])
        assert "venv" not in yaml.safe_load(descriptor.read_text()).get("build", {}), (
            "a supplied source descriptor must not bootstrap another producer venv"
        )
        return {"status": "complete", "root": str(tmp_path / "build")}

    ctx = SimpleNamespace(args=args, root=tmp_path, repository=source, pnms=pnms)
    job.build_application(ctx)
    assert calls[0][calls[0].index("--runtime") + 1] == str(
        args.sdk_prefix / "bin/physicsnemo-infer"
    )


@pytest.mark.parametrize("project", ["../outside", "/absolute"])
def test_supplied_project_must_be_inside_frozen_snapshot(tmp_path, project):
    context = SimpleNamespace(
        args=SimpleNamespace(project=project),
        root=tmp_path,
        repository=tmp_path / "source",
    )
    with pytest.raises(ValueError, match="project"):
        job.build_application(context)


def test_controller_uses_finite_unprivileged_public_build_job(tmp_path):
    from scripts import run_lepton_pnms_release_build as controller

    args = controller.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--runtime-image",
            RUNTIME,
            "--run-id",
            "release-unit",
            "--artifact-dir",
            str(tmp_path),
            "--backend",
            "aoti",
        ]
    )
    command = controller.build_job_command(args, "unit-job")
    runner = shlex.split(command[command.index("--command") + 1])
    assert "/usr/bin/timeout" in runner
    assert "--privileged" not in command
    assert "--expected-runner-sha256" in runner
    assert "pnms_release_build_job.py" in " ".join(runner)
    assert "--runtime-image" in runner and RUNTIME in runner
    assert "docker" not in runner and "--image" not in runner


def test_summary_requires_exact_public_build_and_sealed_transfer_inventory():
    report = {
        "schema_version": 1,
        "status": "passed",
        **provenance(),
        "profile": "release-build",
        "expected_cases": job.expected_cases(),
        "cases": [{"name": name, "status": "passed"} for name in job.expected_cases()],
        "build_id": "d" * 32,
        "bundle_digest": "f" * 64,
        "archive": {"sha256": "1" * 64, "size_bytes": 5120},
        "build_receipt_identity": {"sha256": "2" * 64, "size_bytes": 123},
        "source_inventory_identity": {
            "sha256": provenance()["source_inventory_sha256"],
            "size_bytes": len(SOURCE_INVENTORY_BYTES),
        },
        "http_runner": "/outputs/native-inference/release-unit/http-runner/pnms_http_job.py",
        "http_runner_sha256": "3" * 64,
        "contract_sha256": "4" * 64,
        "runner_sha256": "5" * 64,
    }
    identity = dict(
        run_id="release-unit",
        source_sha=SOURCE,
        image_digest=IMAGE,
        profile="release-build",
    )
    job.validate_summary(report, **identity)
    for mutation in (
        "duplicate",
        "missing-case",
        "archive",
        "receipt",
        "source-inventory",
        "source-inventory-identity",
    ):
        invalid = copy.deepcopy(report)
        if mutation == "duplicate":
            invalid["cases"].append(invalid["cases"][0])
        elif mutation == "missing-case":
            invalid["cases"].pop()
        elif mutation == "archive":
            invalid["archive"]["sha256"] = ""
        elif mutation == "receipt":
            invalid.pop("build_receipt_identity")
        elif mutation == "source-inventory-identity":
            invalid["source_inventory_identity"]["sha256"] = "6" * 64
        else:
            invalid.pop("source_inventory_sha256")
        with pytest.raises(ValueError):
            job.validate_summary(invalid, **identity)


@pytest.fixture
def controlled_build(tmp_path, monkeypatch):
    from scripts import run_lepton_pnms_release_build as controller

    project, receipt, record = built_project(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    manifest = job.seal_build(
        project,
        receipt,
        remote,
        source_inventory=frozen_inventory(tmp_path),
        **provenance(),
    )
    args = controller.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--runtime-image",
            RUNTIME,
            "--run-id",
            "release-unit",
            "--artifact-dir",
            str(tmp_path / "controller"),
        ]
    )
    root = args.artifact_dir / args.run_id
    events = []
    summary = {
        "schema_version": 1,
        "status": "passed",
        **provenance(),
        "profile": "release-build",
        "expected_cases": job.expected_cases(),
        "cases": [{"name": name, "status": "passed"} for name in job.expected_cases()],
        "build_id": record["build_id"],
        "bundle_digest": record["bundle_digest"],
        "archive": manifest["archive"],
        "build_receipt_identity": file_identity(receipt),
        "source_inventory_identity": manifest["source_inventory_identity"],
        "runner_sha256": job.sha256(job.__file__),
        "http_runner": "/outputs/native-inference/release-unit/http-runner/pnms_http_job.py",
        "http_runner_sha256": job.sha256(ROOT / "qa/native_inference/pnms_http_job.py"),
        "contract_sha256": job.sha256(ROOT / "qa/native_inference/contract.py"),
    }

    def lifecycle(arguments, **options):
        root.mkdir(parents=True)
        write_json(
            root / "summary.json",
            {
                "status": "passed",
                "cleanup_errors": [],
                "job_id": "owned-job",
                "remote_artifacts": "/outputs/native-inference/release-unit",
            },
        )
        write_json(root / "job-summary.json", summary)
        options["summary_validator"](
            summary,
            run_id=args.run_id,
            source_sha=SOURCE,
            image_digest=IMAGE,
            profile="release-build",
        )
        events.append("gpu-cleanup")
        return 0

    def placement(arguments, **options):
        assert arguments == ["job", "nodes", "-i", "owned-job"]
        events.append("capture-node")
        return 0, '["owned-node"]'

    def fetch(**options):
        assert events == ["capture-node", "gpu-cleanup"]
        assert options["job"]["controller"]["node_id"] == "owned-node"
        assert options["expected"]["archive"] == manifest["archive"]
        output = options["output"]
        output.mkdir()
        shutil.copyfile(remote / "bundle.tar", output / "bundle.tar")
        write_json(output / "manifest.json", manifest)
        extracted = extract(output / "bundle.tar", manifest, output / "extracted")
        result = dict(
            status="passed",
            cleanup_errors=[],
            deployment_deleted=True,
            extracted=str(extracted),
        )
        write_json(output / "transfer-summary.json", result)
        return result

    monkeypatch.setattr(controller.lifecycle, "run", lifecycle)
    monkeypatch.setattr(controller.lifecycle, "lep", placement)
    monkeypatch.setattr(controller, "fetch_artifact", fetch)
    return SimpleNamespace(
        controller=controller,
        args=args,
        root=root,
        events=events,
        manifest=manifest,
        fetch=fetch,
    )


def test_controller_downloads_only_after_gpu_cleanup_and_publishes_local_inputs(
    controlled_build,
):
    state = controlled_build
    assert state.controller.run(state.args) == 0
    inputs = json.loads((state.root / "release-inputs.json").read_text())
    assert inputs["status"] == "passed" and inputs["node_id"] == "owned-node"
    assert Path(inputs["project"]).is_relative_to(state.root)
    assert Path(inputs["build_receipt"]).is_file()
    assert inputs["archive"] == state.manifest["archive"]
    assert inputs["http_runner"].startswith("/outputs/")
    assert state.events == ["capture-node", "gpu-cleanup"]


@pytest.mark.parametrize("failure", ["download", "cpu-cleanup", "gpu-cleanup"])
def test_controller_never_publishes_inputs_after_transfer_or_cleanup_failure(
    controlled_build, monkeypatch, failure
):
    state = controlled_build
    if failure == "gpu-cleanup":
        original = state.controller.lifecycle.run

        def failed_lifecycle(*args, **kwargs):
            original(*args, **kwargs)
            summary = json.loads((state.root / "summary.json").read_text())
            summary["cleanup_errors"] = ["owned GPU remove failed"]
            write_json(state.root / "summary.json", summary)
            return 0

        monkeypatch.setattr(state.controller.lifecycle, "run", failed_lifecycle)
    else:

        def failed_fetch(**options):
            if failure == "download":
                raise RuntimeError("archive download failed")
            result = state.fetch(**options)
            result.update(
                cleanup_errors=["CPU exporter remove failed"], deployment_deleted=False
            )
            return result

        monkeypatch.setattr(state.controller, "fetch_artifact", failed_fetch)
    assert state.controller.run(state.args) == 1
    assert not (state.root / "release-inputs.json").exists()
    assert json.loads((state.root / "summary.json").read_text())["status"] == "failed"


def test_controller_records_cancelled_export_instead_of_leaving_passed_status(
    controlled_build, monkeypatch
):
    state = controlled_build

    def interrupted(**options):
        raise KeyboardInterrupt

    monkeypatch.setattr(state.controller, "fetch_artifact", interrupted)
    try:
        code = state.controller.run(state.args)
    except KeyboardInterrupt:
        pytest.fail("interrupted export left the controller summary marked passed")
    assert code == 130
    assert not (state.root / "release-inputs.json").exists()
    assert (
        json.loads((state.root / "summary.json").read_text())["status"] == "cancelled"
    )


def test_main_rejects_source_drift_before_any_public_build(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source"
    requirements = source / "qa/native_inference/venv-requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("numpy==2.2.6\n")
    metadata, records = tmp_path / "source.json", tmp_path / "inventory.json"
    write_json(
        records,
        {
            str(requirements.relative_to(source)): {
                **file_identity(requirements),
                "mode": 0o644,
                "symlink": None,
            }
        },
    )
    write_json(
        metadata,
        {
            "source_sha": SOURCE,
            "source_archive_sha256": "a" * 64,
            "source_inventory_sha256": job.sha256(records),
        },
    )
    requirements.write_text("mutated after capture\n")
    called = []

    class Context:
        def __init__(self, args):
            self.root = args.output
            self.root.mkdir()
            self.environment = {}

    monkeypatch.setattr(job.local, "Context", Context)
    monkeypatch.setattr(job, "build_application", lambda ctx: called.append("build"))
    output = tmp_path / "outputs"
    code = job.main(
        [
            "--repository",
            str(source),
            "--source-metadata",
            str(metadata),
            "--source-inventory",
            str(records),
            "--output",
            str(output),
            "--run-id",
            "release-unit",
            "--expected-source-sha",
            SOURCE,
            "--image-digest",
            IMAGE,
            "--runtime-image",
            RUNTIME,
            "--expected-runner-sha256",
            job.sha256(job.__file__),
        ]
    )
    assert code == 1 and called == []
    assert not (output / "manifest.json").exists()
    report = job.extract_summary(capsys.readouterr().out)
    assert report["status"] == "failed" and report["cases"] == []
    assert "source file identity differs" in report["error"]


def test_controller_default_python_reaches_public_build_as_current_interpreter(
    tmp_path, monkeypatch, capsys
):
    from scripts import run_lepton_pnms_release_build as controller

    project, receipt, _ = built_project(tmp_path)
    source = tmp_path / "source"
    paths = {
        "app/pnms.yaml": (project / "pnms.yaml").read_bytes(),
        "qa/native_inference/venv-requirements.txt": b"numpy==2.2.6\n",
        "qa/native_inference/pnms_http_job.py": b"# frozen HTTP runner\n",
        "qa/native_inference/contract.py": b"# frozen contract\n",
    }
    records = {}
    for relative, body in paths.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        records[relative] = {**file_identity(path), "mode": 0o644, "symlink": None}
    metadata, source_inventory = tmp_path / "source.json", tmp_path / "inventory.json"
    write_json(source_inventory, records)
    write_json(
        metadata,
        {
            "source_sha": SOURCE,
            "source_archive_sha256": "a" * 64,
            "source_inventory_sha256": job.sha256(source_inventory),
        },
    )
    args = controller.build_parser().parse_args(
        [
            "--image",
            IMAGE,
            "--expected-source-sha",
            SOURCE,
            "--runtime-image",
            RUNTIME,
            "--run-id",
            "release-default-python",
            "--repository",
            str(source),
            "--source-metadata",
            str(metadata),
            "--source-inventory",
            str(source_inventory),
            "--project",
            "app",
            "--mount-target",
            str(tmp_path / "mounted"),
        ]
    )
    assert args.python == "python3", "exercise the actual controller default"
    command = controller.build_job_command(args, "default-python-job")
    runner = shlex.split(command[command.index("--command") + 1])
    script = str(source / "qa/native_inference/pnms_release_build_job.py")
    driver_arguments = runner[runner.index(script) + 1 :]
    calls = []

    class Context:
        def __init__(self, arguments):
            self.args = arguments
            self.root = arguments.output.resolve()
            self.root.mkdir(parents=True)
            self.repository = arguments.repository.resolve()
            self.environment = {}

        def pnms(self, label, *arguments):
            calls.append(arguments)
            selected = str(arguments[arguments.index("--python") + 1])
            assert selected == sys.executable, (
                f"public build must use the actual current interpreter, got {selected!r}"
            )
            return {"status": "complete", "root": str(receipt.parent)}

    monkeypatch.setattr(job.local, "Context", Context)
    code = job.main(driver_arguments)
    report = job.extract_summary(capsys.readouterr().out)
    assert code == 0, report.get("error")
    assert report["status"] == "passed" and len(calls) == 1
