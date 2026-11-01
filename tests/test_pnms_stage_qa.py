import hashlib
import json
from pathlib import Path

import pytest

from qa.native_inference import pnms_stage_job as job


def frozen_source(tmp_path):
    source = tmp_path / "source"
    path = source / "qa/native_inference/venv-requirements.txt"
    path.parent.mkdir(parents=True)
    path.write_text("numpy==2.2.6\n")
    record = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
        mode=0o644,
        symlink=None,
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({path.relative_to(source).as_posix(): record}))
    metadata = tmp_path / "source.json"
    metadata.write_text(
        json.dumps(
            dict(
                source_sha="a" * 40,
                source_archive_sha256="b" * 64,
                source_inventory_sha256=hashlib.sha256(
                    inventory.read_bytes()
                ).hexdigest(),
            )
        )
    )
    return source, metadata, inventory


def test_stage_preserves_frozen_bytes_and_provides_hashed_handoff(tmp_path):
    source, metadata, inventory = frozen_source(tmp_path)
    output = tmp_path / "outputs"
    result = job.stage(source, metadata, inventory, output, "a" * 40)
    staged = output / "source/qa/native_inference/venv-requirements.txt"
    assert staged.is_file(), "QA source must be staged without a private host transfer"
    assert (
        staged.read_bytes()
        == (source / staged.relative_to(output / "source")).read_bytes()
    )
    assert (
        result["requirements_sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()
    )
    manifest = Path(result["source_manifest"])
    assert (
        result["source_manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert json.loads(manifest.read_text())["files"] == json.loads(
        inventory.read_text()
    )


def test_stage_rejects_source_drift_before_publishing_handoff(tmp_path):
    source, metadata, inventory = frozen_source(tmp_path)
    (source / "qa/native_inference/venv-requirements.txt").write_text("changed")
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="identity"):
        job.stage(source, metadata, inventory, output, "a" * 40)
    assert not (output / "source-manifest.json").exists()


def test_stage_rejects_traversal_even_when_inventory_hash_matches(tmp_path):
    source, metadata, inventory = frozen_source(tmp_path)
    records = json.loads(inventory.read_text())
    records["../outside"] = next(iter(records.values()))
    inventory.write_text(json.dumps(records))
    meta = json.loads(metadata.read_text())
    meta["source_inventory_sha256"] = hashlib.sha256(inventory.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="path"):
        job.stage(source, metadata, inventory, tmp_path / "outputs", "a" * 40)


def test_stage_does_not_overwrite_existing_output(tmp_path):
    source, metadata, inventory = frozen_source(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("previous run")
    with pytest.raises(FileExistsError):
        job.stage(source, metadata, inventory, output, "a" * 40)
    assert marker.read_text() == "previous run"


def test_main_preserves_previous_run_summary(tmp_path, capsys):
    source, metadata, inventory = frozen_source(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text('{"status":"previous-success"}')
    previous = summary.read_bytes()
    result = job.main(
        [
            "--source",
            str(source),
            "--metadata",
            str(metadata),
            "--inventory",
            str(inventory),
            "--output",
            str(output),
            "--run-id",
            "stage-retry",
            "--expected-source-sha",
            "a" * 40,
            "--image-digest",
            "registry/image@sha256:" + "c" * 64,
            "--expected-runner-sha256",
            job.sha256(job.__file__),
        ]
    )
    assert result == 1
    assert summary.read_bytes() == previous, (
        "A failed retry must preserve previous QA evidence"
    )
    assert job.extract_summary(capsys.readouterr().out)["status"] == "failed"
