import json
from pathlib import Path
import tarfile

import pytest

from qa.scripts.export_native_cfd_release import build_bundle, digest, inventory


def fixture(root: Path):
    receipt = {
        "status": "passed",
        "source_sha": "a" * 40,
        "qualification": "full-run1-fields-and-metrics-v1",
        "builder_image": "registry/image@sha256:" + "b" * 64,
        "packages": [],
    }
    for count in (60000, 8095):
        package = root / "build" / str(count)
        package.mkdir(parents=True)
        (package / "model.json").write_text(
            json.dumps({"inputs": [{"name": "fx", "shape": [1, count, 2]}]})
        )
        (package / "model.pt2").write_bytes(f"compiled-{count}".encode())
        receipt["packages"].append(
            {
                "manifest_sha256": digest(package / "model.json"),
                "files": inventory(package),
            }
        )
    (root / "native").mkdir()
    (root / "native/global_stats.json").write_text("{}")
    (root / "summary.json").write_text(
        json.dumps({"status": "passed", "source_sha": "a" * 40})
    )
    (root / "environment.json").write_text("{}")
    (root / "unrelated.txt").write_text("never export this")
    receipt.update(
        stats_sha256=digest(root / "native/global_stats.json"),
        summary_sha256=digest(root / "summary.json"),
    )
    (root / "build-receipt.json").write_text(json.dumps(receipt))


def test_export_contains_only_receipted_release_inputs(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "release.tar"
    manifest = build_bundle(tmp_path, path, "a" * 40)
    assert manifest["sha256"] == digest(path)
    with tarfile.open(path) as archive:
        names = archive.getnames()
    assert sorted(names) == sorted(manifest["files"])
    assert "unrelated.txt" not in names
    assert "packages/points-60000/model.pt2" in names
    assert "packages/points-8095/model.pt2" in names


@pytest.mark.parametrize("mutation", ["payload", "summary", "stats", "source", "extra"])
def test_export_rejects_changed_qualification_inputs(tmp_path, mutation):
    fixture(tmp_path)
    source = "a" * 40
    if mutation == "payload":
        (tmp_path / "build/60000/model.pt2").write_bytes(b"changed")
    elif mutation == "summary":
        (tmp_path / "summary.json").write_text("{}")
    elif mutation == "stats":
        (tmp_path / "native/global_stats.json").write_text("changed")
    elif mutation == "source":
        source = "c" * 40
    else:
        (tmp_path / "build/60000/unreceipted.txt").write_text("extra")
    with pytest.raises(ValueError):
        build_bundle(tmp_path, tmp_path / "release.tar", source)
    assert not (tmp_path / "release.tar").exists()
