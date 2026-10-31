from test_native_release_orchestration import candidate, staging_resolved, release
import hashlib
import pytest


def test_new_release_binds_public_contract_and_workflow_startup(tmp_path, monkeypatch):
    output = candidate(tmp_path)
    value = staging_resolved(tmp_path)
    value.update(
        manifest_version="0.1.0",
        public_contract_digest="a" * 64,
        input_files={},
        startup_cases=[
            {
                "name": "affine",
                "request": {"values": [0, 1, -1, 4]},
                "expected": {"values": [1, 3, -1, 9]},
            }
        ],
    )
    monkeypatch.setattr(release, "model_input_identity", lambda *_: None)
    bundle = tmp_path / "bundle"
    inventory = release.stage_release(
        value,
        {"primary": output},
        bundle,
        sources=[{"path": "unit-only", "sha256": "f" * 64, "size_bytes": 1}],
    )
    config = release.read_json(bundle / "deployment.json")
    assert config["schema_version"] == 3
    assert config["workflow"]["schema_version"] == 2
    assert config["manifest_version"] == "0.1.0"
    assert config["public_contract_digest"] == "a" * 64
    assert config["startup_cases"] == value["startup_cases"]
    assert set(inventory["management_capabilities"]) == {
        "live-native-readiness-v1",
        "fresh-deployment-v1",
    }
    release.verify_bundle(bundle)


def test_startup_uploaded_fixture_is_sealed_into_image_inventory(tmp_path, monkeypatch):
    output = candidate(tmp_path)
    value = staging_resolved(tmp_path)
    fixture = tmp_path / "features.bin"
    fixture.write_bytes(b"independent fixture bytes")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    value.update(
        manifest_version="0.1.0",
        public_contract_digest="a" * 64,
        startup_cases=[
            {
                "name": "affine",
                "request": {},
                "expected": {"ok": True},
                "input_artifacts": {
                    "features": {
                        "path": str(fixture),
                        "sha256": digest,
                        "size_bytes": fixture.stat().st_size,
                        "media_type": "application/octet-stream",
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(release, "model_input_identity", lambda *_: None)
    bundle = tmp_path / "bundle"
    inventory = release.stage_release(
        value,
        {"primary": output},
        bundle,
        sources=[{"path": "unit-only", "sha256": "f" * 64, "size_bytes": 1}],
    )
    config = release.read_json(bundle / "deployment.json")
    path = config["startup_cases"][0]["input_artifacts"]["features"]["path"]
    assert path == "/opt/physicsnemo-release/startup/affine/features"
    assert any(
        item["path"] == path and item["sha256"] == digest
        for item in config["model_assets"]
    )
    assert any(item["path"] == "startup/affine/features" for item in inventory["files"])
    release.verify_bundle(bundle)
    fixture.write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="startup fixture file bytes changed"):
        release.stage_release(
            value,
            {"primary": output},
            tmp_path / "changed",
            sources=[{"path": "unit-only", "sha256": "f" * 64, "size_bytes": 1}],
        )


def test_startup_case_cannot_stage_files_outside_bundle(tmp_path, monkeypatch):
    output = candidate(tmp_path)
    value = staging_resolved(tmp_path)
    fixture = tmp_path / "input.bin"
    fixture.write_bytes(b"fixture")
    value.update(
        manifest_version="0.1.0",
        public_contract_digest="a" * 64,
        startup_cases=[
            {
                "name": "../../escaped",
                "request": {},
                "expected": {"ok": True},
                "input_artifacts": {
                    "features": {
                        "path": str(fixture),
                        **release.file_identity(fixture),
                        "media_type": "application/octet-stream",
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(release, "model_input_identity", lambda *_: None)
    with pytest.raises(ValueError, match="startup case name"):
        release.stage_release(
            value,
            {"primary": output},
            tmp_path / "bundle",
            sources=[{"path": "unit-only", "sha256": "f" * 64, "size_bytes": 1}],
        )
    assert not (tmp_path / "escaped").exists()
