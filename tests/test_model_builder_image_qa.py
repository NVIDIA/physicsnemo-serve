"""CPU contracts for directly executing the shipped builder image on Lepton."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import stat
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.native_inference import model_builder_image_job as job  # noqa: E402
from scripts import run_lepton_model_builder_image_qa as controller  # noqa: E402

IDENTITY = {
    "run_id": "builder-image-unit",
    "source_sha": "a" * 40,
    "image_digest": "registry.example/builder@sha256:" + "b" * 64,
    "profile": "builder-image",
}


def report():
    cases = []
    for name in job.expected_cases():
        if name == "environment":
            details = {
                "runner_sha256": "c" * 64,
                "contract_sha256": "d" * 64,
                "source_manifest_sha256": "e" * 64,
                "installed_builder_sha256": "f" * 64,
                "entrypoint_sha256": "0" * 64,
                "runtime_sha256": "1" * 64,
                "gpu_uuid": "GPU-allocated",
                "device": {"type": "cuda", "index": 0},
                "installed_builder_matches_source": True,
            }
        elif name == "sdk.consumer.build":
            details = {"consumer_sha256": "2" * 64}
        elif name == "handoff":
            details = {"inventory_sha256": "3" * 64, "file_count": 20, "verified": True}
        elif name.endswith((".build", ".repeat")):
            details = {
                "build_sha256": "4" * 64,
                "release_sha256": "5" * 64,
                "case_count": 3,
                "verified": True,
            }
        elif name.endswith(".check"):
            details = {"configuration_only": True, "status": "configuration-ok"}
        else:
            details = {
                "case_count": 3,
                "parity": "passed",
                "device": {"type": "cuda", "index": 0},
                "backend": name.split(".")[1],
                "package_sha256": "6" * 64,
            }
        cases.append({"name": name, "status": "passed", "details": details})
    return {
        "schema_version": 1,
        **IDENTITY,
        "status": "passed",
        "qa_scope": "model-builder-image-payload",
        "docker_launcher_qa": "component-only",
        "expected_cases": job.expected_cases(),
        "cases": cases,
    }


def arguments(tmp_path):
    return controller.build_parser().parse_args(
        [
            "--image",
            IDENTITY["image_digest"],
            "--expected-source-sha",
            IDENTITY["source_sha"],
            "--run-id",
            IDENTITY["run_id"],
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "unit-workspace",
            "--node-group",
            "unit-nodes",
            "--pull-secret",
            "unit-secret",
            "--nfs-mount-base",
            "/unit/storage",
            "--runner",
            "/outputs/staged/model_builder_image_job.py",
            "--source-root",
            "/outputs/staged/source",
            "--source-manifest",
            "/outputs/staged/source.json",
            "--source-manifest-sha256",
            "e" * 64,
        ]
    )


def test_complete_inventory_covers_both_backends_and_explicit_scope():
    job.validate_summary(report(), **IDENTITY)
    names = job.expected_cases()
    assert len(names) == 15
    for backend in ("aoti", "tensorrt"):
        assert f"affine.{backend}.sdk" in names
        assert f"affine.{backend}.relocation" in names
        assert f"affine.{backend}.repeat" in names


@pytest.mark.parametrize(
    "mutation",
    [
        "skip",
        "duplicate",
        "scope",
        "source",
        "receipt",
        "parity",
        "device",
        "backend",
        "handoff",
        "schema",
    ],
)
def test_summary_fails_closed_for_incomplete_or_foreign_evidence(mutation):
    value = copy.deepcopy(report())
    cases = {case["name"]: case for case in value["cases"]}
    if mutation == "skip":
        value["cases"].pop()
    elif mutation == "duplicate":
        value["cases"].append(value["cases"][0])
    elif mutation == "scope":
        value["docker_launcher_qa"] = "passed"
    elif mutation == "source":
        cases["environment"]["details"]["installed_builder_matches_source"] = False
    elif mutation == "receipt":
        cases["affine.aoti.repeat"]["details"]["release_sha256"] = ""
    elif mutation == "parity":
        cases["affine.tensorrt.sdk"]["details"]["case_count"] = 0
    elif mutation == "device":
        cases["affine.aoti.cli"]["details"]["device"]["type"] = "cpu"
    elif mutation == "backend":
        cases["affine.aoti.relocation"]["details"]["backend"] = "tensorrt"
    elif mutation == "handoff":
        cases["handoff"]["details"]["verified"] = False
    else:
        value["schema_version"] = True
    with pytest.raises(ValueError):
        job.validate_summary(value, **IDENTITY)


def test_controller_runs_mounted_driver_without_docker_or_privilege(tmp_path):
    command = controller.build_job_command(arguments(tmp_path), "unit-job")
    assert "--privileged" not in command
    assert command[command.index("--container-image") + 1] == IDENTITY["image_digest"]
    argv = shlex.split(command[command.index("--command") + 1])
    assert argv[:3] == ["exec", "/opt/nvidia/nvidia_entrypoint.sh", "python3"]
    assert "/outputs/staged/model_builder_image_job.py" in argv
    assert "--expected-runner-sha256" in argv
    assert "--expected-contract-sha256" in argv
    assert "docker" not in argv


@pytest.mark.parametrize("field", ["runner", "source_root", "source_manifest"])
def test_controller_rejects_paths_outside_shared_mount_before_submit(
    tmp_path, monkeypatch, field
):
    args = arguments(tmp_path)
    setattr(args, field, "/tmp/unstaged")
    calls = []
    monkeypatch.setattr(controller.lifecycle, "run", lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError, match="mount"):
        controller.run(args)
    assert calls == []


def test_builder_command_preserves_installed_entrypoint_and_never_imports_checkout():
    argv = job.builder_command(
        "build",
        Path("/outputs/source/recipe.json"),
        Path("/outputs/build"),
        "aoti",
        "/opt/sdk/bin/physicsnemo-infer",
    )
    assert argv[:2] == ["/opt/nvidia/nvidia_entrypoint.sh", "pnms-model-builder"]
    assert argv[argv.index("--executor") + 1] == "local"
    assert argv[argv.index("--runtime") + 1] == "/opt/sdk/bin/physicsnemo-infer"
    assert "--builder-image" not in argv


def test_result_document_accepts_entrypoint_banner_but_rejects_multiple_results():
    assert job.result_document('NVIDIA initialization\n{"status":"complete"}\n') == {
        "status": "complete"
    }
    with pytest.raises(ValueError):
        job.result_document('{"status":"complete"}\n{"status":"complete"}\n')


def test_source_manifest_verifies_complete_inventory_and_pinned_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("value = 1\n")
    manifest = tmp_path / "source.json"
    files = {
        record["path"]: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "mode": stat.S_IMODE((source / record["path"]).stat().st_mode),
            "symlink": None,
        }
        for record in job.inventory(source)
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": IDENTITY["source_sha"],
                "source_archive_sha256": "a" * 64,
                "files": files,
            }
        )
    )
    digest = job.sha256(manifest)
    job.verify_source(source, manifest, digest, IDENTITY["source_sha"])
    (source / "extra.py").write_text("pass\n")
    with pytest.raises(ValueError, match="inventory"):
        job.verify_source(source, manifest, digest, IDENTITY["source_sha"])


def test_recipe_check_uses_supported_configuration_only_operation():
    argv = job.builder_command(
        "check",
        Path("/outputs/source/recipe.json"),
        Path("/outputs/check"),
        "aoti",
        "/opt/sdk/bin/physicsnemo-infer",
    )
    assert "--config-only" in argv, (
        "Recipe eager checking is unsupported by Model Builder; native build performs numerical checking"
    )


def test_installed_builder_fingerprint_rejects_a_different_python_module(tmp_path):
    installed = tmp_path / "installed"
    source = tmp_path / "source"
    for root in (installed, source):
        root.mkdir()
        (root / "__init__.py").write_text("# package\n")
    assert job.compare_builder_sources(installed, source)
    (source / "worker.py").write_text("different = True\n")
    with pytest.raises(ValueError, match="installed Model Builder"):
        job.compare_builder_sources(installed, source)


@pytest.fixture(params=job.BACKENDS)
def relocation_context(tmp_path, request):
    backend = request.param
    ctx = job.Context(
        SimpleNamespace(
            output=tmp_path / "qa",
            source_root=tmp_path / "mounted-source",
            sdk_prefix=tmp_path / "installed-sdk",
            run_id=IDENTITY["run_id"],
            expected_source_sha=IDENTITY["source_sha"],
            image_digest=IDENTITY["image_digest"],
            command_timeout=30,
        )
    )
    producer_roots = [ctx.root / name for name in ("build", "repeat")]
    for root in producer_roots:
        source = root / backend / "source/producer.bin"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"producer-only dependency")
        source.chmod(0o444)
        source.parent.chmod(0o751)
    model = producer_roots[0] / backend / "model"
    package = model / "backends" / backend
    package.mkdir(parents=True)
    (package / "model.pt2").write_text(
        str(producer_roots[0] / backend / "source/producer.bin")
    )
    (package / "model.pt2").chmod(0o444)
    job.write_json(
        package / "model.json",
        {"artifacts": [{"backend": backend, "target": "cuda", "path": "model.pt2"}]},
    )
    job.write_json(
        model / "model-release.json",
        {
            "format_version": 1,
            "variants": {
                backend: {
                    "package": f"backends/{backend}",
                    "files": job.inventory(model),
                }
            },
        },
    )
    for path in (ctx.source / "fixture.py", ctx.args.sdk_prefix / "lib/runtime.so"):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"shared input remains available")
    # Run the real relocation case without repeating the preceding GPU builds.
    ctx.case = lambda name, function: (
        function() if name.endswith(".relocation") else None
    )
    return ctx, backend, producer_roots


def producer_snapshot(roots):
    return {
        str(path): (
            stat.S_IMODE(path.stat().st_mode),
            path.read_bytes() if path.is_file() else None,
        )
        for root in roots
        for path in (root, *sorted(root.rglob("*")))
    }


def install_relocation_consumer(ctx, roots, *, requires_original=False, fail=False):
    ctx.consumer.parent.mkdir(parents=True)
    ctx.consumer.write_text(
        "import json, struct, sys\nfrom pathlib import Path\n"
        f"roots = {list(map(str, roots))!r}\n"
        "package, backend, destination, *inputs = sys.argv[1:]\n"
        "package, destination = Path(package), Path(destination)\n"
        + (
            '(Path((package / "model.pt2").read_text())).read_bytes()\n'
            if requires_original
            else "assert all(not Path(root).exists() for root in roots), roots\n"
        )
        + f"assert Path({str(ctx.source / 'fixture.py')!r}).is_file()\n"
        + f"assert Path({str(ctx.args.sdk_prefix / 'lib/runtime.so')!r}).is_file()\n"
        + ('raise RuntimeError("consumer requested failure")\n' if fail else "")
        + "for index, path in enumerate(inputs):\n"
        "    values = struct.unpack('<4f', Path(path).read_bytes())\n"
        "    (destination / f'output-{index}.f32').write_bytes(\n"
        "        struct.pack('<4f', *(2 * value + 1 for value in values)))\n"
        "(destination / 'metadata.json').write_text(json.dumps({\n"
        "    'completed': True, 'backend': backend, 'executor_count': 1,\n"
        "    'execution_device': {'type': 'cuda', 'index': 0}, 'requests': len(inputs),\n"
        "    'original_producer_paths_unavailable': all(not Path(p).exists() for p in roots)\n"
        "}))\n"
    )
    # Keep the real subprocess/logging path while making the fixture portable.
    run = ctx.run
    ctx.run = lambda name, arguments: run(name, [sys.executable, *arguments])


def test_relocation_rejects_absolute_dependency_on_original_producer(
    relocation_context,
):
    ctx, backend, roots = relocation_context
    before = producer_snapshot(roots)
    install_relocation_consumer(ctx, roots, requires_original=True)
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        job.backend_cases(ctx, backend)
    assert producer_snapshot(roots) == before
    assert not list(ctx.root.glob(".relocation-producers-*"))


@pytest.mark.parametrize("fail", [False, True])
def test_relocation_isolates_producers_and_restores_bytes_and_modes(
    relocation_context, fail
):
    ctx, backend, roots = relocation_context
    before = producer_snapshot(roots)
    install_relocation_consumer(ctx, roots, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="consumer requested failure"):
            job.backend_cases(ctx, backend)
    else:
        job.backend_cases(ctx, backend)
        metadata = json.loads(
            (
                ctx.root / "consumers" / backend / "relocated-sdk/metadata.json"
            ).read_text()
        )
        assert metadata["original_producer_paths_unavailable"] is True
        original = ctx.root / "build" / backend / "model"
        relocated = ctx.root / "relocated" / backend / "model"
        assert job.inventory(original) == job.inventory(relocated)
        payload = relocated / "backends" / backend / "model.pt2"
        assert stat.S_IMODE(payload.stat().st_mode) == 0o444
    assert producer_snapshot(roots) == before
    assert not list(ctx.root.glob(".relocation-producers-*"))


def test_relocation_reuses_existing_builds_with_separate_evidence(relocation_context):
    ctx, backend, roots = relocation_context
    install_relocation_consumer(ctx, roots)
    producer_root = ctx.root
    for name in (
        "logs/previous.log",
        "consumers/previous.f32",
        "relocated/previous.json",
    ):
        path = producer_root / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"immutable prior evidence")
        path.chmod(0o444)
    before = producer_snapshot([producer_root])
    ctx.root = producer_root.with_name("new-evidence")
    ctx.root.mkdir()
    details = job.relocated_consume(ctx, backend, producer_root=producer_root)
    assert details["parity"] == "passed"
    assert details["producer_paths"] == list(map(str, roots))
    assert details["producer_paths_unavailable"] is True
    assert producer_snapshot([producer_root]) == before
    assert (ctx.root / "logs" / f"{backend}.relocated-sdk/exit.json").is_file()
