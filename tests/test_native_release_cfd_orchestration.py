from __future__ import annotations

from pathlib import Path, PureWindowsPath
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release as release  # noqa:E402
import native_release_cfd_job as driver  # noqa:E402
from native_release_cfd_http_job import verify_reference  # noqa:E402
from test_native_release_orchestration import candidate, resolved  # noqa:E402


def test_cfd_handoff_retains_actual_compilation_source_and_joint_proof(tmp_path):
    full = tmp_path / "full"
    output = full / "build/transolver60000"
    original = candidate(tmp_path)
    output.parent.mkdir(parents=True)
    shutil.copytree(original, output)
    (output / "cfd-release").mkdir()
    (output / "cfd-release/old.json").write_text("{}")
    for path, content in [
        ("reference/fields.json", "{}"),
        ("native/global_stats.json", "{}"),
        ("summary.json", "{}"),
        ("environment.json", "{}"),
        ("startup/startup.json", "{}"),
    ]:
        target = full / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    release.write_json(
        full / "build-receipt.json", {"status": "passed", "source_sha": "f" * 40}
    )
    config = resolved()
    config.update(
        source_sha="a" * 40,
        model_aliases={"full": {"project_document": {"points": 60000}}},
    )
    config.update(
        provider_profile="unit-test",
        cfd={
            name: {"sha256": "a" * 64, "size_bytes": 1}
            for name in ("profile", "request", "checkpoint", "statistics")
        },
    )
    handoff = tmp_path / "handoff"
    driver.collect_builds(
        full,
        handoff,
        config,
        image="registry@sha256:" + "b" * 64,
        keys={"full": "c" * 64},
        stack_fingerprint="d" * 64,
    )
    got = handoff / "aliases/full"
    assert release.read_json(got / "release-provenance.json")["source_sha"] == "f" * 40
    assert (
        release.read_json(got / "release-provenance.json")["qualification_source_sha"]
        == "a" * 40
    )
    proof = release.read_json(got / "cfd-release/build-receipt.json")
    assert proof["compilation_keys"] == {"full": "c" * 64}
    assert proof["reference"] == {
        "path": str(full / "reference/fields.json"),
        **release.file_identity(full / "reference/fields.json"),
    }
    assert not (got / "cfd-release/old.json").exists()
    assert (got / "cfd-release/startup/startup.json").read_text() == "{}"


def test_cfd_reference_rejects_mutated_or_unavailable_reference(tmp_path):
    path = tmp_path / "fields.json"
    path.write_text("{}")
    record = {"path": str(path), **release.file_identity(path)}
    assert verify_reference(record) == path
    path.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="reference bytes changed"):
        verify_reference(record)


def test_compile_preflight_hashes_actual_native_libraries(monkeypatch):
    import json

    monkeypatch.setattr(
        driver.qualify_stack.__globals__["subprocess"],
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"files": {"lib.so": "different"}})
        ),
    )
    with pytest.raises(ValueError, match="native stack differs"):
        driver.qualify_stack({"stack_fingerprint": "a" * 64})


def test_cfd_runtime_output_and_api_roots_match_resolved_profile():
    config = {
        "artifact_root": "/outputs/native-cfd",
        "input_cache": "/outputs/.cache/native-cfd",
        "request_timeout_seconds": 3600,
        "drain_timeout_seconds": 60,
    }
    profile = {
        "runtime": {
            "artifact_root": "/outputs/native-transolver",
            "input_cache": "/outputs/.cache/native-transolver",
            "request_timeout_seconds": 1200,
            "drain_timeout_seconds": 90,
            "startup_timeout_seconds": 300,
        }
    }
    helper = getattr(release, "configure_cfd_runtime", lambda config, resolved: config)
    got = helper(config, profile)
    assert got["artifact_root"] == "/outputs/native-transolver", (
        "worker output root must match API image root"
    )
    assert got["input_cache"] == "/outputs/.cache/native-transolver"
    assert got["request_timeout_seconds"] == 1200
    assert got["drain_timeout_seconds"] == 90
    assert "startup_timeout_seconds" not in got, (
        "v1 strict schema uses its fixed startup budget"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {
            "status": "passed",
            "policy": "finite-static-inputs-v1",
            "session_load_count": 2,
            "models": [],
        },
        {
            "status": "passed",
            "policy": "expected-fp32-v1",
            "session_load_count": 2,
            "models": [
                {
                    "point_count": 60000,
                    "shape": [1, 60000, 4],
                    "dtype": "float32",
                    "finite": True,
                    "backend": "aoti",
                    "comparison": {"rtol": 1e-5, "atol": 1e-6, "max_absolute_error": 0},
                }
            ],
        },
    ],
)
def test_cfd_startup_observer_rejects_legacy_or_incomplete_golden_proof(bad):
    import native_release_cfd_http_job as http_job

    helper = getattr(http_job, "verify_cfd_startup", lambda proof: proof)
    with pytest.raises(ValueError):
        helper(bad)


def test_cfd_startup_observer_accepts_both_expected_output_sessions():
    from native_release_cfd_http_job import verify_cfd_startup

    proof = {
        "status": "passed",
        "policy": "expected-fp32-v1",
        "session_load_count": 2,
        "models": [
            {
                "point_count": n,
                "shape": [1, n, 4],
                "dtype": "float32",
                "finite": True,
                "backend": "aoti",
                "comparison": {"rtol": 1e-5, "atol": 1e-6, "max_absolute_error": 0},
            }
            for n in (60000, 8095)
        ],
    }
    assert verify_cfd_startup(proof) == proof


def joint_fixture(tmp_path):
    from test_native_release import _fixture
    from scripts.native_workflow import resolve_release

    root = Path(__file__).resolve().parents[1]
    cfg = resolve_release(root / "plugins/physicsnemo-cfd-transolver-native")
    base = _fixture(tmp_path)
    cfg["images"]["builder"] = base["builder_image"]
    cfg["cfd"]["statistics"] = release.file_identity(base["stats"])
    cfg["_native_stack_fingerprint"] = "d" * 64
    original_keys = {
        alias: release.compilation_key(cfg, alias) for alias in cfg["model_aliases"]
    }
    cfg["model_aliases"] = {
        "large": cfg["model_aliases"]["full"],
        "remainder": cfg["model_aliases"]["tail"],
    }
    proof = tmp_path / "proof"
    proof.mkdir()
    receipt = release.read_json(base["compile_evidence"])
    receipt.update(
        builder_stack_image=cfg["images"]["builder"],
        compilation_keys=original_keys,
        native_stack_fingerprint="d" * 64,
        reference={
            "path": "/node/qualified/fields.json",
            "sha256": "e" * 64,
            "size_bytes": 2,
        },
        input_root="/node/qualified/inputs",
    )
    startup = proof / "startup"
    startup.mkdir()
    release.write_json(startup / "startup.json", {"schema_version": 1, "models": []})
    receipt["reference_contract"] = release.cfd_reference_contract(cfg)
    receipt["startup_fixture"] = {
        "path": "startup/startup.json",
        "files": [
            {
                "path": "startup/startup.json",
                **release.file_identity(startup / "startup.json"),
            }
        ],
    }
    release.write_json(proof / "build-receipt.json", receipt)
    shutil.copyfile(
        base["compile_evidence"].parent / "summary.json", proof / "summary.json"
    )
    shutil.copyfile(base["stats"], proof / "global_stats.json")
    candidates = {}
    for alias, package in zip(("large", "remainder"), base["packages"], strict=True):
        target = tmp_path / alias
        shutil.copytree(package, target / "model/backends/aoti")
        shutil.copytree(proof, target / "cfd-release")
        release.write_json(
            target / "release-provenance.json",
            {"source_sha": "b" * 40, "placement": {"node_id": "node-original"}},
        )
        candidates[alias] = target
    return cfg, candidates, proof


def test_cfd_staging_reuses_exact_packages_when_alias_labels_change(tmp_path):
    cfg, candidates, _ = joint_fixture(tmp_path)
    # Actual complete Builder receipts/parity are separately exercised by the
    # verifier tests. This test isolates joint-proof→legacy-packager staging;
    # the stand-in packages are never executed or called GPU-qualified here.
    from unittest.mock import patch

    with patch.object(
        release, "verify_completed_build", return_value={"package": "backends/aoti"}
    ):
        staged = release.stage_cfd_release(
            cfg,
            candidates,
            tmp_path / "renamed-release",
            sources=[{"path": "example.cpp", "sha256": "f" * 64, "size_bytes": 1}],
        )
    assert set(staged["models"]) == {"large", "remainder"}
    assert (
        release.read_json(tmp_path / "renamed-release/orchestration.json")["node_id"]
        == "node-original"
    )
    deployment = release.read_json(tmp_path / "renamed-release/deployment.json")
    assert deployment["artifact_root"] == cfg["runtime"]["artifact_root"]


@pytest.mark.parametrize(
    ("reference", "valid"),
    [
        ("/node/qualified/fields.json", True),
        ("node/qualified/fields.json", False),
        (r"C:\node\qualified\fields.json", False),
    ],
)
def test_cfd_joint_proof_uses_linux_reference_paths_on_windows_hosts(
    tmp_path, monkeypatch, reference, valid
):
    cfg, candidates, proof = joint_fixture(tmp_path)
    receipt = release.read_json(proof / "build-receipt.json")
    receipt["reference"]["path"] = reference
    release.write_json(proof / "build-receipt.json", receipt)

    def controller_path(value):
        return PureWindowsPath(value) if value == reference else Path(value)

    monkeypatch.setattr(
        release,
        "verify_completed_build",
        lambda *args, **kwargs: {"package": "backends/aoti"},
    )
    identity = release._cfd_joint_identity(cfg, candidates)
    monkeypatch.setattr(release, "Path", controller_path)
    if valid:
        assert release._validate_cfd_joint(proof, identity) == receipt
    else:
        with pytest.raises(ValueError, match="CFD reference location or byte identity"):
            release._validate_cfd_joint(proof, identity)


def test_cfd_policy_inventory_covers_active_recovery_separately_from_tiny_workflows():
    cfd = release.required_http_cases({"adapter": "transolver_surface"})
    tensor = release.required_http_cases(
        {"adapter": "tensor.v1", "qualification": {"required_cases": ["default"]}}
    )
    assert "active_recovery" in cfd
    assert "active_recovery" not in tensor
    assert (
        cfd.index("parity")
        < cfd.index("active_recovery")
        < cfd.index("final_completion")
    )
    assert "worker_restart" in cfd and "worker_restart" in tensor


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Release-controller CFD cache publication uses POSIX file locks",
)
def test_cfd_joint_proof_requalifies_changed_reference_without_compiling_models(
    tmp_path,
):
    from unittest.mock import patch

    cfg, candidates, proof = joint_fixture(tmp_path)
    reference_contract = {
        "provider_profile": cfg["provider_profile"],
        **{
            name: cfg["cfd"][name]
            for name in ("profile", "request", "checkpoint", "statistics")
        },
    }
    receipt = release.read_json(proof / "build-receipt.json")
    receipt["reference_contract"] = reference_contract
    release.write_json(proof / "build-receipt.json", receipt)
    publish = getattr(
        release, "cfd_joint_publish", lambda *args, **kwargs: {"status": "hit"}
    )
    lookup = getattr(
        release, "cfd_joint_lookup", lambda *args, **kwargs: {"status": "hit"}
    )
    keys_before = [
        release.compilation_key(cfg, alias) for alias in cfg["model_aliases"]
    ]
    with patch.object(
        release, "verify_completed_build", return_value={"package": "backends/aoti"}
    ):
        publish(tmp_path / "cache", cfg, candidates, proof, node_id="node-original")
        assert lookup(tmp_path / "cache", cfg, candidates)["status"] == "hit"
        cfg["cfd"]["statistics"]["sha256"] = "0" * 64
        assert lookup(tmp_path / "cache", cfg, candidates)["status"] != "hit", (
            "changed statistics must requalify the joint proof"
        )
    assert keys_before == [
        release.compilation_key(cfg, alias) for alias in cfg["model_aliases"]
    ]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Release-controller CFD cache publication uses POSIX file locks",
)
def test_individually_valid_mixed_aliases_need_new_joint_proof(tmp_path):
    from unittest.mock import patch

    cfg, candidates, proof = joint_fixture(tmp_path)
    contract = {
        "provider_profile": cfg["provider_profile"],
        **{
            name: cfg["cfd"][name]
            for name in ("profile", "request", "checkpoint", "statistics")
        },
    }
    receipt = release.read_json(proof / "build-receipt.json")
    receipt["reference_contract"] = contract
    release.write_json(proof / "build-receipt.json", receipt)
    publish = getattr(
        release, "cfd_joint_publish", lambda *args, **kwargs: {"status": "hit"}
    )
    lookup = getattr(
        release, "cfd_joint_lookup", lambda *args, **kwargs: {"status": "hit"}
    )
    with patch.object(
        release, "verify_completed_build", return_value={"package": "backends/aoti"}
    ):
        publish(tmp_path / "cache", cfg, candidates, proof, node_id="node-original")
        package = candidates["large"] / "model/backends/aoti"
        (package / "model.pt2").write_bytes(
            b"another individually verified publisher's bytes"
        )
        assert lookup(tmp_path / "cache", cfg, candidates)["status"] != "hit", (
            "a receipt for different paired bytes cannot cover a mixed cache hit"
        )


def test_joint_requalification_handoff_reuses_both_model_bytes(tmp_path, monkeypatch):
    cfg, candidates, proof = joint_fixture(tmp_path)
    plan = {
        "models": {
            alias: {
                "compilation_key": release.compilation_key(cfg, alias),
                "action": "reuse",
            }
            for alias in cfg["model_aliases"]
        }
    }
    inputs = tmp_path / "input"
    inputs.mkdir()
    extracted = tmp_path / "received"
    for alias in candidates:
        shutil.copytree(proof, extracted / "aliases" / alias / "cfd-release")
    invocations = []

    def build_image(*args, **kwargs):
        request = release.read_json(inputs / "request.json")
        assert {item["action"] for item in request["build_aliases"].values()} == {
            "reuse"
        }
        for alias, candidate_path in candidates.items():
            assert (
                inputs / "reuse" / alias / "model/backends/aoti/model.pt2"
            ).read_bytes() == (
                candidate_path / "model/backends/aoti/model.pt2"
            ).read_bytes()
        return {"image": "registry@sha256:" + "a" * 64}

    def run_job(**kwargs):
        invocations.append(kwargs)
        assert kwargs["mode"] == "cfd_compile"
        return {"controller": {"node_id": "node-new", "job_id": "new-job"}}

    monkeypatch.setattr(release, "build_image", build_image)
    monkeypatch.setattr(
        release,
        "cache_publish",
        lambda *args, **kwargs: pytest.fail(
            "unchanged models must not republish or compile"
        ),
    )
    before = {alias: release.inventory(path) for alias, path in candidates.items()}
    updated, record, received_proof = release._compile_handoff(
        cfg,
        plan,
        candidates,
        {},
        source=tmp_path,
        custom=tmp_path,
        inputs=inputs,
        empty=tmp_path,
        workspace=tmp_path,
        artifact_workspace=tmp_path / "attempt",
        runner=None,
        lepton=SimpleNamespace(
            run_job=run_job,
            fetch_artifact=lambda **kwargs: {"extracted": str(extracted)},
        ),
        cache_dir=tmp_path / "cache",
        stack={"fingerprint": "d" * 64},
        process_runner=None,
    )
    assert len(invocations) == 1
    assert updated == candidates
    assert before == {
        alias: release.inventory(path) for alias, path in candidates.items()
    }
    assert received_proof.is_dir()
    assert record["job"]["node_id"] == "node-new"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Release-controller CFD cache publication uses POSIX file locks",
)
def test_joint_proof_cache_reverifies_bytes_and_retains_invalid_entry(tmp_path):
    from unittest.mock import patch

    cfg, candidates, proof = joint_fixture(tmp_path)
    with patch.object(
        release, "verify_completed_build", return_value={"package": "backends/aoti"}
    ):
        cached = release.cfd_joint_publish(
            tmp_path / "cache", cfg, candidates, proof, node_id="node-original"
        )
        assert cached["status"] == "hit"
        (Path(cached["path"]) / "startup/startup.json").write_text('{"mutated":true}')
        assert (
            release.cfd_joint_lookup(tmp_path / "cache", cfg, candidates)["status"]
            == "invalid"
        )
        repaired = release.cfd_joint_publish(
            tmp_path / "cache", cfg, candidates, proof, node_id="node-new"
        )
        assert repaired["status"] == "hit"
        assert repaired["node_id"] == "node-new"
    assert len(list((tmp_path / "cache/.cfd-qualified").glob(".invalid-*"))) == 1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Release-controller CFD cache publication uses POSIX file locks",
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LEPTON_WORKSPACE_ID", "other-workspace"),
        ("NFS_MOUNT_BASE", "/different/nfs-root"),
        ("LEPTON_NODE_GROUP", "other-nodes"),
        ("LEPTON_LUSTRE_STORAGE", "other-storage"),
        ("QA_LUSTRE_DIR", "other-qa"),
        ("LEPTON_WORKSPACE_URL", "https://other-workspace.example"),
        ("DEFAULT_NFS", "/default/changed"),
        ("PROFILE_MOUNT_TARGET", "/other-outputs"),
    ],
)
def test_joint_proof_follows_effective_environment_location_without_model_recompile(
    tmp_path, monkeypatch, field, value
):
    from unittest.mock import patch
    import native_release_lepton as lepton

    cfg, candidates, proof = joint_fixture(tmp_path)
    cfg["lepton"] = {}
    lifecycle = lepton._lifecycle()
    config = tmp_path / "deploy-config.yaml"
    config.write_text(
        "lepton_workspace_id: original-workspace\nlepton_node_group: original-nodes\npull_secret: pull\nnfs_mount_base: /original/nfs\n"
    )
    monkeypatch.setitem(
        lifecycle.load_deploy_config.__globals__, "_CONFIG_PATH", config
    )
    for key in (
        "LEPTON_WORKSPACE_ID",
        "NFS_MOUNT_BASE",
        "LEPTON_NODE_GROUP",
        "LEPTON_LUSTRE_STORAGE",
        "QA_LUSTRE_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    bind = getattr(release, "reference_storage_identity", lambda controller: controller)
    cfg["_reference_storage"] = bind(lepton.validate_controller_config(cfg))
    receipt = release.read_json(proof / "build-receipt.json")
    receipt["reference_storage"] = cfg["_reference_storage"]
    release.write_json(proof / "build-receipt.json", receipt)
    before = {
        alias: release.compilation_key(cfg, alias) for alias in cfg["model_aliases"]
    }
    with patch.object(
        release, "verify_completed_build", return_value={"package": "backends/aoti"}
    ):
        release.cfd_joint_publish(
            tmp_path / "cache", cfg, candidates, proof, node_id="old-node"
        )
        if field == "DEFAULT_NFS":
            config.write_text(config.read_text().replace("/original/nfs", value))
        elif field == "PROFILE_MOUNT_TARGET":
            cfg["lepton"]["mount_target"] = value
        else:
            monkeypatch.setenv(field, value)
        cfg["_reference_storage"] = bind(lepton.validate_controller_config(cfg))
        assert (
            release.cfd_joint_lookup(tmp_path / "cache", cfg, candidates)["status"]
            != "hit"
        ), (
            "an oracle from another effective workspace/storage location cannot be reused"
        )
    assert before == {
        alias: release.compilation_key(cfg, alias) for alias in cfg["model_aliases"]
    }


def test_cfd_builder_launcher_propagates_real_cli_failure(tmp_path):
    import os
    import subprocess

    package = tmp_path / "stub/model_builder/build"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package.parent / "__init__.py").write_text("")
    (package / "cli.py").write_text("def main():\n    return 7\n")
    launcher = tmp_path / "model-builder"
    helper = getattr(driver, "write_builder_launcher", None)
    if helper:
        helper(launcher)
    else:
        launcher.write_text(
            "#!/usr/bin/env python3\nfrom model_builder.build.cli import main\nmain()\n"
        )
        launcher.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(launcher)],
        env={**os.environ, "PYTHONPATH": str(tmp_path / "stub")},
        capture_output=True,
    )
    assert result.returncode == 7, (
        "a failed Model Builder import/build must not become a passed CFD gate"
    )


def test_cfd_http_environment_identifies_selected_tensorrt_stack(tmp_path, monkeypatch):
    import native_release_cfd_http_job as http_job

    runner = http_job.CFDJob.__new__(http_job.CFDJob)
    runner.args = SimpleNamespace(release_root=tmp_path)
    runner.release = {"backend": "tensorrt", "target": {"profile": "h100"}}
    runner.summary = {}
    monkeypatch.setattr(http_job, "verify_bundle", lambda root: None)
    monkeypatch.setattr(http_job, "read_json", lambda path: {"reference": {}})
    monkeypatch.setattr(http_job, "verify_reference", lambda record: None)
    monkeypatch.setattr(
        http_job.scientific.HTTPJob,
        "preflight",
        lambda self: {"backend": "tensorrt"},
    )
    monkeypatch.setattr(
        http_job,
        "environment",
        lambda target, backend="aoti": {"backend": backend},
    )
    monkeypatch.setattr(
        http_job, "MemoryMonitor", lambda: SimpleNamespace(start=lambda: None)
    )
    assert runner.preflight()["backend"] == "tensorrt"
    assert runner.summary["environment"]["backend"] == "tensorrt"
