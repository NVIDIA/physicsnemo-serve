"""Receipt and cache gates for the matched CFD benchmark."""

from __future__ import annotations

import json

import pytest

from qa.native_inference import cfd_benchmark_reference as reference
from qa.scripts.cfd_parity_contract import sha256_file


@pytest.fixture
def qualified(tmp_path):
    capture = tmp_path / "reference"
    capture.mkdir()
    report = capture / "report.json"
    report.write_text('[{"metric":1.0}]')
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"passed"}')
    fields = capture / "fields.json"
    evidence = {
        "source_sha": "a" * 40,
        "image": "builder@sha256:" + "b" * 64,
        "report": {
            "path": "report.json",
            "sha256": sha256_file(report),
            "size_bytes": report.stat().st_size,
        },
    }
    fields.write_text(json.dumps(evidence))
    receipt = {
        "status": "passed",
        "qualification": "full-run1-fields-and-metrics-v1",
        "source_sha": evidence["source_sha"],
        "builder_image": evidence["image"],
        "summary_sha256": sha256_file(summary),
    }
    path = tmp_path / "build-receipt.json"
    path.write_text(json.dumps(receipt))
    return fields, evidence, path, receipt, summary, report


def test_qualified_reference_binds_report_and_compilation_identity(qualified):
    fields, _, _, receipt, _, report = qualified
    value, identity = reference.qualified_reference(fields)
    assert value == [{"metric": 1.0}]
    assert identity == {
        "fields_sha256": sha256_file(fields),
        "report_sha256": sha256_file(report),
        "compilation_source_sha": receipt["source_sha"],
        "builder_image": receipt["builder_image"],
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("status", "provisional", "qualification"),
        ("qualification", "finite-static-inputs-v1", "qualification"),
        ("summary_sha256", "0" * 64, "summary"),
        ("source_sha", "c" * 40, "source/image"),
        ("builder_image", "other@sha256:" + "d" * 64, "source/image"),
    ],
)
def test_reference_rejects_partial_foreign_or_unbound_receipts(
    qualified, key, value, message
):
    fields, _, path, receipt, _, _ = qualified
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=message):
        reference.qualified_reference(fields)


def test_reference_rejects_failed_summary_even_with_updated_hash(qualified):
    fields, _, path, receipt, summary, _ = qualified
    summary.write_text('{"status":"failed"}')
    receipt["summary_sha256"] = sha256_file(summary)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="summary"):
        reference.qualified_reference(fields)


@pytest.mark.parametrize("content", ['[{"metric":2.0}]', "[]"])
def test_reference_rejects_changed_report_bytes_or_length(qualified, content):
    fields, _, _, _, _, report = qualified
    report.write_text(content)
    with pytest.raises(ValueError, match="report bytes"):
        reference.qualified_reference(fields)


def test_reference_report_cannot_escape_capture_directory(qualified):
    fields, evidence, _, _, _, report = qualified
    outside = fields.parent.parent / "outside.json"
    outside.write_bytes(report.read_bytes())
    evidence["report"]["path"] = "../outside.json"
    fields.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="escaped"):
        reference.qualified_reference(fields)


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "artifacts"
    output = root / "run" / "attempt" / "benchmark-output"
    output.mkdir(parents=True)
    return root, {
        "run": {
            "output_dir": str(output),
            "metrics_cache": {"enabled": True, "path": str(output / "metrics_cache")},
        }
    }


def test_metric_cache_is_observed_without_changing_shipped_policy(cache):
    root, config = cache
    observation = reference.cold_cache_observation(config, root)
    assert observation == {
        "enabled": True,
        "path": config["run"]["metrics_cache"]["path"],
        "initially_absent": True,
    }
    from pathlib import Path

    Path(observation["path"]).mkdir()
    assert reference.cold_cache_observation(config, root)["initially_absent"] is False
    assert config["run"]["metrics_cache"]["enabled"] is True


@pytest.mark.parametrize("policy", [False, None, 1, "true"])
def test_metric_cache_observation_rejects_changed_shipped_policy(cache, policy):
    root, config = cache
    config["run"]["metrics_cache"]["enabled"] = policy
    with pytest.raises(ValueError, match="policy"):
        reference.cold_cache_observation(config, root)


def test_metric_cache_must_belong_to_fresh_attempt_output(cache, tmp_path):
    root, config = cache
    config["run"]["metrics_cache"]["path"] = str(tmp_path / "old-cache")
    with pytest.raises(ValueError, match="cache path"):
        reference.cold_cache_observation(config, root)


def test_metric_cache_output_cannot_escape_artifact_root(cache, tmp_path):
    root, config = cache
    config["run"]["output_dir"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="escaped"):
        reference.cold_cache_observation(config, root)
