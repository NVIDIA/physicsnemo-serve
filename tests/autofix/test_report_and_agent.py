# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ci_autofix import COMMENT_MARKER
from ci_autofix import report as rpt
from ci_autofix.agent import (
    PullRequest,
    build_task,
    environment_config,
    load_config,
    model_config,
)
from ci_autofix.failures import Failure, JobFailures
from ci_autofix.jobs import load_jobs
from ci_autofix.triage import FLAKY, REPRODUCIBLE, Classified

PY = Failure("Python Tests", "pytest", "tests/t.py::test_a")
FLAKY_F = Failure("Python Tests", "pytest", "tests/t.py::test_b")


def _classified():
    return [Classified(PY, REPRODUCIBLE), Classified(FLAKY_F, FLAKY, "passed on rerun")]


def test_build_task_lists_only_reproducible_failures():
    pr = PullRequest(7, "Title", "Body text", ["feat: x\n\nmore"], "+changed line")
    job = JobFailures("Python Tests", [PY, FLAKY_F], "FAILED tests/t.py::test_a")
    task = build_task(pr, _classified(), [job], load_jobs(), Path("/tmp"), "be careful")
    assert f"`{PY.key}`" in task
    assert f"`{FLAKY_F.key}`" not in task
    assert "cd /workspace && PYTHONPATH=" in task
    assert "+changed line" in task and "be careful" in task and "- feat: x" in task


def test_config_wiring_keeps_secrets_out_of_the_container(tmp_path):
    config = load_config()
    env = environment_config(
        config, "img", tmp_path / "ws", tmp_path / "out", tmp_path / "cache"
    )
    assert env["image"] == "img"
    assert f"{tmp_path / 'ws'}:/workspace" in env["run_args"]
    assert not any("KEY" in k or "TOKEN" in k for k in env["env"])
    model = model_config(config, "openai/m", "http://endpoint", "secret")
    assert model["model_kwargs"]["api_base"] == "http://endpoint"
    assert config["agent"]["step_limit"] > 0


def test_render_fixed_with_test_update():
    state = {
        "outcome": rpt.FIXED,
        "mode": "push",
        "push_result": "pushed",
        "classified": [c.to_dict() for c in _classified()],
        "flaky": True,
        "verdict": {
            "summary": "Renamed field.",
            "failures": [
                {
                    "failure": PY.key,
                    "verdict": "test-outdated",
                    "reason": "intended",
                    "evidence": [{"file": "src/x.py", "lines": "3", "why": "renamed"}],
                }
            ],
        },
        "commits": [{"title": "ci-fix(agent): update outdated test", "kind": "test"}],
        "diff": "+x",
        "run_url": "https://example/run",
    }
    body = rpt.render(state)
    assert body.startswith(COMMENT_MARKER)
    assert "pushed a fix" in body
    assert "Tests updated: please review" in body
    assert "`src/x.py` lines 3: renamed" in body
    assert "Re-run the failed CI jobs" in body


def test_render_dry_run_and_rejection():
    assert "dry run" in rpt.render({"outcome": rpt.FIXED, "mode": "dry-run"})
    body = rpt.render(
        {"outcome": rpt.REJECTED, "validation": {"violations": ["adds a skip marker"]}}
    )
    assert "Rejected because" in body and "adds a skip marker" in body


def test_upsert_comment_updates_existing():
    calls = []

    class GH:
        def paginate(self, path, key=None, params=None):
            return [
                {"id": 3, "body": "hi"},
                {"id": 9, "body": f"{COMMENT_MARKER}\nold"},
            ]

        def patch(self, path, body):
            calls.append(("patch", path))

        def post(self, path, body):
            calls.append(("post", path))

    rpt.upsert_comment(GH(), 7, "new")
    assert calls == [("patch", "/repos/{repo}/issues/comments/9")]
