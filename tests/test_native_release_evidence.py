from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
from native_release import read_json, write_json
import native_release_artifacts as artifacts


def test_evidence_sealing_preserves_reports_and_binds_omitted_large_files(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    write_json(root / "summary.json", {"status": "passed"})
    (root / "stack.log").write_text("service exited cleanly\n")
    (root / "fields.bin").write_bytes(b"123456789")
    helper = getattr(artifacts, "seal_evidence", lambda *args, **kwargs: {})
    report = helper(
        root,
        {
            "run_id": "sample",
            "source_sha": "a" * 40,
            "image_digest": "repo@sha256:" + "b" * 64,
        },
        maximum_file_bytes=8,
    )
    assert report.get("status") == "complete", (
        "qualification evidence must be transportable"
    )
    manifest = read_json(root / "manifest.json")
    extracted = artifacts.extract(root / "bundle.tar", manifest, tmp_path / "recovered")
    assert json.loads((extracted / "summary.json").read_text()) == {"status": "passed"}
    assert (extracted / "stack.log").read_text() == "service exited cleanly\n"
    retained = read_json(extracted / "retained-files.json")
    assert retained["omitted_large_files"] == [
        {"path": "fields.bin", **artifacts.file_identity(root / "fields.bin")}
    ]
    assert not (extracted / "fields.bin").exists()
