# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU characterization of the sealed native build artifact transport."""

from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import sys
import tarfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release_artifacts as artifacts


def file_record(name, data):
    return {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def archive(tmp_path, members, expected=None):
    """Create an intentionally untrusted tar and an independently selected inventory."""
    path = tmp_path / "transport.tar"
    files = []
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as bundle:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../outside.txt"
            if kind == tarfile.REGTYPE:
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
            else:
                bundle.addfile(member)
            files.append(file_record(name, data))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "archive": artifacts.file_identity(path),
        "files": files if expected is None else expected,
    }
    return path, manifest


def test_sealed_transport_roundtrip_preserves_bytes_and_provenance(tmp_path):
    source = tmp_path / "handoff"
    payload = {
        "aliases/primary/model/backends/aoti/model.pt2": b"transport-only fixture\x00\xff",
        "aliases/primary/build.json": b'{"status":"fixture"}\n',
        "aliases/primary/empty.bin": b"",
    }
    for name, data in payload.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    bundle = tmp_path / "bundle.tar"
    provenance = {
        "run_id": "cpu-transport-fixture",
        "image_digest": "registry.example/compiler@sha256:" + "a" * 64,
        "source_sha": "b" * 40,
        "compilation_keys": {"primary": "c" * 64},
    }
    manifest = artifacts.seal(source, bundle, **provenance)
    assert all(manifest[key] == value for key, value in provenance.items())
    assert manifest["files"] == sorted(
        [file_record(name, data) for name, data in payload.items()],
        key=lambda item: item["path"],
    )
    output = tmp_path / "verified"
    assert artifacts.extract(bundle, manifest, output) == output
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == payload


@pytest.mark.parametrize(
    "name",
    ["../outside.txt", "/outside.txt", "nested/../../outside.txt", "C:/outside.txt"],
)
def test_tar_paths_cannot_escape_destination(tmp_path, name):
    bundle, manifest = archive(
        tmp_path,
        [(name, b"escape", tarfile.REGTYPE)],
        [file_record("safe.bin", b"escape")],
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="path|relative"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE]
)
def test_tar_links_and_special_files_are_rejected(tmp_path, kind):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"preserve")
    bundle, manifest = archive(tmp_path, [("selected.bin", b"", kind)])
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="link, special file"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()
    assert outside.read_bytes() == b"preserve"


@pytest.mark.parametrize("duplicate", ["selected.bin", "./selected.bin"])
def test_duplicate_and_equivalent_tar_paths_are_rejected(tmp_path, duplicate):
    bundle, manifest = archive(
        tmp_path,
        [
            ("selected.bin", b"first", tarfile.REGTYPE),
            (duplicate, b"second", tarfile.REGTYPE),
        ],
        [file_record("selected.bin", b"first")],
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="duplicate file paths"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()


def test_unexpected_tar_file_is_not_admitted_by_manifest(tmp_path):
    bundle, manifest = archive(
        tmp_path,
        [
            ("selected.bin", b"selected", tarfile.REGTYPE),
            ("unexpected.bin", b"extra", tarfile.REGTYPE),
        ],
        [file_record("selected.bin", b"selected")],
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="members differ"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()


def test_missing_tar_file_is_not_admitted_by_manifest(tmp_path):
    bundle, manifest = archive(
        tmp_path,
        [("selected.bin", b"selected", tarfile.REGTYPE)],
        [
            file_record("selected.bin", b"selected"),
            file_record("missing.bin", b"missing"),
        ],
    )
    with pytest.raises(ValueError, match="members differ"):
        artifacts.extract(bundle, manifest, tmp_path / "verified")


def test_archive_hash_tamper_is_rejected_before_extraction(tmp_path):
    bundle, manifest = archive(
        tmp_path, [("selected.bin", b"selected", tarfile.REGTYPE)]
    )
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="archive SHA256 or size differs"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()


def test_per_file_hash_is_checked_even_with_correct_archive_hash(tmp_path):
    bundle, manifest = archive(
        tmp_path,
        [("selected.bin", b"changed!", tarfile.REGTYPE)],
        [file_record("selected.bin", b"original")],
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="artifact bytes changed"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists(), (
        "failed byte verification must remove partially extracted files"
    )


def test_tar_member_size_must_match_inventory_before_writing(tmp_path):
    bundle, manifest = archive(
        tmp_path,
        [("selected.bin", b"longer", tarfile.REGTYPE)],
        [file_record("selected.bin", b"tiny")],
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="wrong size"):
        artifacts.extract(bundle, manifest, output)
    assert not output.exists()


def test_archive_byte_limit_is_enforced(tmp_path):
    bundle, manifest = archive(
        tmp_path, [("selected.bin", b"selected", tarfile.REGTYPE)]
    )
    output = tmp_path / "verified"
    with pytest.raises(ValueError, match="archive exceeds byte limit"):
        artifacts.extract(
            bundle, manifest, output, maximum_bytes=bundle.stat().st_size - 1
        )
    assert not output.exists()


def test_incomplete_manifest_and_duplicate_inventory_are_rejected(tmp_path):
    bundle, manifest = archive(
        tmp_path, [("selected.bin", b"selected", tarfile.REGTYPE)]
    )
    incomplete = {**manifest, "status": "running"}
    with pytest.raises(ValueError, match="completed artifact transport"):
        artifacts.extract(bundle, incomplete, tmp_path / "incomplete")
    duplicate = copy.deepcopy(manifest)
    duplicate["files"].append(duplicate["files"][0])
    with pytest.raises(ValueError, match="duplicate file inventory"):
        artifacts.extract(bundle, duplicate, tmp_path / "duplicate")


def test_existing_output_is_never_overwritten_or_removed(tmp_path):
    bundle, manifest = archive(
        tmp_path, [("selected.bin", b"selected", tarfile.REGTYPE)]
    )
    output = tmp_path / "existing"
    output.mkdir()
    (output / "preserve.txt").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="output already exists"):
        artifacts.extract(bundle, manifest, output)
    assert (output / "preserve.txt").read_bytes() == b"preserve"


def test_seal_rejects_source_symlink_and_preserves_existing_archive(tmp_path):
    source = tmp_path / "handoff"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    (source / "link").symlink_to(outside)
    bundle = tmp_path / "bundle.tar"
    with pytest.raises(ValueError, match="symlink"):
        artifacts.seal(source, bundle)
    assert not bundle.exists()
    (source / "link").unlink()
    (source / "selected.bin").write_bytes(b"selected")
    bundle.write_bytes(b"preserve existing")
    with pytest.raises(ValueError, match="archive output already exists"):
        artifacts.seal(source, bundle)
    assert bundle.read_bytes() == b"preserve existing"


def test_seal_rejects_archive_inside_its_input(tmp_path):
    source = tmp_path / "handoff"
    source.mkdir()
    (source / "selected.bin").write_bytes(b"selected")
    with pytest.raises(ValueError, match="inside its own input"):
        artifacts.seal(source, source / "bundle.tar")
    assert not (source / "bundle.tar").exists()
