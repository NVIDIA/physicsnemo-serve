from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "inference"))

import test_output_publication as publication_test  # noqa: E402


class Adapter:
    def _resolve_workflow(self, workflow_name):
        return {
            "earth2-deterministic": "earth2-deterministic",
            "earth2-ensemble": "earth2-ensemble",
        }[workflow_name]


def test_publication_requests_are_filtered_to_enabled_plugin(monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "earth2-deterministic")
    requests = [
        ("earth2-deterministic", {"value": 1}),
        ("earth2-ensemble", {"value": 2}),
    ]

    assert publication_test._publication_requests_for_enabled_plugin(
        Adapter(), requests
    ) == [("earth2-deterministic", {"value": 1})]


def test_publication_request_missing_for_enabled_plugin_fails(monkeypatch):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "earth2-ensemble")
    monkeypatch.setattr(publication_test.pytest, "skip", lambda _reason: None)

    with pytest.raises(AssertionError, match="no publication request payload"):
        publication_test._publication_requests_for_enabled_plugin(
            Adapter(), [("earth2-deterministic", {"value": 1})]
        )
