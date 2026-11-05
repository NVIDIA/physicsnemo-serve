# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A clean image context binds the actual source bytes and excludes local assets."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "qa/plugins/earth2-aifs2ens/prepare_source.py"


@pytest.fixture
def source_builder():
    spec = importlib.util.spec_from_file_location("aifs2ens_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.py").write_text("print('initial')\n")
    (repo / ".gitignore").write_text("ignored/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Source Test",
            "-c",
            "user.email=source-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "tracked.py").write_text("print('working tree')\n")
    (repo / "new-plugin.py").write_text("print('untracked feature')\n")
    (repo / "new-plugin.py").chmod(0o755)
    return repo


def test_context_has_modified_and_untracked_source_with_deterministic_inventory(
    source_builder, repository, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    identity = source_builder.prepare_source(repository, first)
    assert source_builder.prepare_source(repository, second) == identity
    manifest = json.loads((first / "qa-source-inventory.json").read_text())
    assert (first / "qa-source-inventory.json").read_bytes() == (
        second / "qa-source-inventory.json"
    ).read_bytes()
    assert manifest["source_content_sha256"] == identity
    assert (
        manifest["base_commit"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    )
    assert [item["path"] for item in manifest["files"]] == [
        ".gitignore",
        "new-plugin.py",
        "tracked.py",
    ]
    assert (
        hashlib.sha256(
            json.dumps(
                manifest["files"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == identity
    )
    for item in manifest["files"]:
        data = (first / item["path"]).read_bytes()
        assert data == (repository / item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
        assert item["bytes"] == len(data)
    assert (first / "new-plugin.py").stat().st_mode & 0o111


def test_context_works_with_image_verifier_and_detects_tampering(
    source_builder, repository, tmp_path, monkeypatch
):
    context = tmp_path / "context"
    identity = source_builder.prepare_source(repository, context)
    path = ROOT / "qa/plugins/earth2-aifs2ens/run_qualification.py"
    spec = importlib.util.spec_from_file_location("source_qualification", path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    monkeypatch.setattr(verifier, "REPO", context)
    assert (
        verifier.verify_source(identity)
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    )
    (context / "tracked.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="source bytes differ"):
        verifier.verify_source(identity)


def test_context_excludes_credentials_caches_assets_and_old_inventories(
    source_builder, repository, tmp_path
):
    blocked = [
        ".env",
        "nested/.env.token",
        "nested/private.pem",
        "nested/private.key",
        ".venv/package.py",
        "nested/__pycache__/code.pyc",
        "target/generated.rs",
        ".run-local/worker.json",
        "qa/artifacts/report.json",
        "deploy/config.yaml",
        "ignored/unrelated.py",
        "assets/model.ckpt",
        "assets/frozen.npy",
        "assets/forecast.zarr/chunk.json",
        "qa-source-inventory.json",
    ]
    for name in blocked:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded fixture\n")
    # Explicit exclusions still apply if a sensitive file was accidentally tracked.
    subprocess.run(["git", "add", "-f", ".env"], cwd=repository, check=True)
    context = tmp_path / "context"
    source_builder.prepare_source(repository, context)
    manifest = json.loads((context / "qa-source-inventory.json").read_text())
    assert [item["path"] for item in manifest["files"]] == [
        ".gitignore",
        "new-plugin.py",
        "tracked.py",
    ]
    assert not any((context / name).exists() for name in blocked[:-1])


@pytest.mark.parametrize("existing", [False, True])
def test_context_refuses_to_replace_existing_destination(
    source_builder, repository, tmp_path, existing
):
    context = tmp_path / "context"
    if existing:
        context.write_text("preserve this file")
    else:
        context.mkdir()
        (context / "keep.txt").write_text("preserve this directory")
    with pytest.raises(FileExistsError):
        source_builder.prepare_source(repository, context)
    assert (
        context.read_text() if existing else (context / "keep.txt").read_text()
    ).startswith("preserve this")


def test_context_must_be_outside_the_repository(source_builder, repository):
    with pytest.raises(ValueError, match="outside"):
        source_builder.prepare_source(repository, repository / "context")
    assert not (repository / "context").exists()


def test_context_refuses_existing_dangling_destination_symlink(
    source_builder, repository, tmp_path
):
    context, target = tmp_path / "context", tmp_path / "uncreated-target"
    context.symlink_to(target)
    with pytest.raises(FileExistsError):
        source_builder.prepare_source(repository, context)
    assert context.is_symlink()
    assert not target.exists()


def test_context_rejects_external_symlinks_without_copying_target(
    source_builder, repository, tmp_path
):
    secret = tmp_path / "private.txt"
    secret.write_text("private fixture")
    (repository / "linked.py").symlink_to(secret)
    context = tmp_path / "context"
    with pytest.raises(ValueError, match="symlink"):
        source_builder.prepare_source(repository, context)
    assert not context.exists()


def test_context_fingerprint_changes_with_working_tree_bytes(
    source_builder, repository, tmp_path
):
    before = source_builder.prepare_source(repository, tmp_path / "before")
    (repository / "new-plugin.py").write_text("updated implementation\n")
    after = source_builder.prepare_source(repository, tmp_path / "after")
    assert before != after


def test_cli_prints_fingerprint_used_by_build_argument(repository, tmp_path):
    context = tmp_path / "context"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repository),
            "--output",
            str(context),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((context / "qa-source-inventory.json").read_text())
    assert result.stdout == manifest["source_content_sha256"] + "\n"
