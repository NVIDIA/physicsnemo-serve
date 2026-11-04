"""Import-light reference and cache gates shared by matched CFD QA tools."""

from __future__ import annotations

import json
from pathlib import Path

from qa.scripts.cfd_parity_contract import read_json_object, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cold_cache_observation(config, artifact_root):
    """Observe the unchanged plugin's per-attempt metric cache before execution."""
    output = Path(config["run"]["output_dir"]).resolve()
    cache = config["run"]["metrics_cache"]
    cache_path = Path(cache["path"]).resolve()
    require(
        output.is_relative_to(artifact_root.resolve()),
        "benchmark output escaped the fresh artifact root",
    )
    require(
        cache_path == output / "metrics_cache", "unexpected shipped metric cache path"
    )
    require(cache["enabled"] is True, "the shipped plugin metric cache policy changed")
    return {
        "enabled": True,
        "path": str(cache_path),
        "initially_absent": not cache_path.exists(),
    }


def qualified_reference(path):
    """Bind captured reports to a completed full-case qualification receipt."""
    fields = read_json_object(path)
    root = path.parent.parent
    receipt = read_json_object(root / "build-receipt.json")
    summary = read_json_object(root / "summary.json")
    require(
        receipt.get("status") == "passed"
        and receipt.get("qualification") == "full-run1-fields-and-metrics-v1",
        "reference qualification did not pass",
    )
    require(
        receipt.get("summary_sha256") == sha256_file(root / "summary.json")
        and summary.get("status") == "passed",
        "qualified summary hash/status differs",
    )
    require(
        fields.get("source_sha") == receipt.get("source_sha")
        and fields.get("image") == receipt.get("builder_image"),
        "reference source/image differs from qualified receipt",
    )
    report_path = (path.parent / fields["report"]["path"]).resolve()
    require(
        report_path.is_relative_to(path.parent.resolve()),
        "reference report escaped its capture directory",
    )
    require(
        sha256_file(report_path) == fields["report"]["sha256"]
        and report_path.stat().st_size == fields["report"]["size_bytes"],
        "reference report bytes differ",
    )
    return json.loads(report_path.read_text()), {
        "fields_sha256": sha256_file(path),
        "report_sha256": sha256_file(report_path),
        "compilation_source_sha": receipt["source_sha"],
        "builder_image": receipt["builder_image"],
    }
