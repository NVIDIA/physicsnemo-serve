# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of `ci_autofix fix` on a real git repo with fake GitHub, sandbox, and model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ci_autofix import cli
from ci_autofix import report as rpt
from ci_autofix.agent import AgentOutcome
from ci_autofix.sandbox import Result

TEST_ID = "tests/test_calc.py::test_add"
KEY = f"Python Tests :: {TEST_ID}"
GOOD = "def add(a, b):\n    return a + b\n"
BROKEN = "def add(a, b):\n    return a - b  # refactor\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """main has a working add(); the PR commit breaks it."""
    repo = tmp_path / "head"
    (repo / "tests").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "calc.py").write_text(GOOD)
    (repo / "tests" / "test_calc.py").write_text(TEST)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "calc.py").write_text(BROKEN)
    _git(repo, "commit", "-qam", "refactor add")
    return repo


class FakeGitHub:
    def __init__(self, log: str):
        self.log = log

    def paginate(self, path, key=None, params=None):
        if path.endswith("/jobs"):
            return [
                {"id": 1, "name": "Python Tests", "conclusion": "failure"},
                {"id": 2, "name": "Lepton QA", "conclusion": "failure"},
                {"id": 3, "name": "Rust Test", "conclusion": "success"},
            ]
        if path.endswith("/commits"):
            return [{"commit": {"message": "refactor add"}}]
        raise AssertionError(path)

    def get(self, path, params=None):
        return {"title": "Refactor add", "body": "Cleanup."}

    def get_text(self, path):
        return self.log


class CalcSandbox:
    """Test passes iff calc.py in the mounted checkout implements addition."""

    def __init__(self, flaky=False):
        self.flaky = flaky

    def run(self, command, checkout, timeout=0):
        if self.flaky:
            return Result(0, "")
        ok = "a + b" in (checkout / "calc.py").read_text()
        return (
            Result(0, "1 passed")
            if ok
            else Result(1, f"FAILED {TEST_ID} - assert 0 == 4")
        )


def _setup(monkeypatch, tmp_path, repo, *, sandbox, agent):
    state_dir = tmp_path / "state"
    target = {
        "pr_number": 7,
        "run_id": 55,
        "head_sha": _git(repo, "rev-parse", "HEAD"),
        "head_ref": "feature",
        "base_ref": "main",
        "trigger": "workflow_run",
    }
    cli.save_state(
        state_dir,
        {
            "decision": {"proceed": True, "pr_number": 7, "target": target},
            "mode": "push",
        },
    )
    monkeypatch.setattr(
        cli, "github_from_env", lambda: FakeGitHub(f"FAILED {TEST_ID} - assert 0 == 4")
    )
    monkeypatch.setattr(cli, "DockerSandbox", lambda image, cache: sandbox)
    monkeypatch.setattr("ci_autofix.agent.run_agent", agent)
    monkeypatch.setenv("AUTOFIX_MODEL", "openai/test")
    monkeypatch.setenv("AUTOFIX_API_BASE", "http://model")
    monkeypatch.setenv("AUTOFIX_API_KEY", "k")
    monkeypatch.setenv("GITHUB_RUN_ID", "901")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    args = ["fix", "--state-dir", str(state_dir), "--head", str(repo), "--image", "img"]
    assert cli.main(args) == 0
    return cli.load_state(state_dir)


def _agent(edit: dict[str, str], verdict: dict | None, exit_status="Submitted"):
    def run_agent(task, *, workspace, out_dir, **kwargs):
        assert KEY in task and "return a - b" in task
        for path, content in edit.items():
            (workspace / path).write_text(content)
        return AgentOutcome(
            exit_status, json.dumps(verdict) if verdict else None, out_dir / "t.json"
        )

    return run_agent


def test_code_fix_is_validated_verified_and_committed(monkeypatch, tmp_path, repo):
    verdict = {
        "summary": "s",
        "failures": [{"failure": KEY, "verdict": "code-fix", "reason": "sign flipped"}],
    }
    state = _setup(
        monkeypatch,
        tmp_path,
        repo,
        sandbox=CalcSandbox(),
        agent=_agent({"calc.py": GOOD}, verdict),
    )

    assert state["outcome"] == rpt.FIXED, state
    assert state["out_of_scope_jobs"] == ["Lepton QA"]
    assert [c["kind"] for c in state["commits"]] == ["fix"]
    message = _git(repo, "log", "-1", "--format=%B")
    assert message.startswith("ci-fix(agent): fix CI failures caused by this PR")
    assert "Autofix-Run: 901" in message
    assert _git(repo, "status", "--porcelain") == ""
    assert "+    return a + b" in state["diff"]
    assert "pushed a fix" in rpt.render(state | {"push_result": "pushed"})


def test_weakened_test_is_rejected_and_discarded(monkeypatch, tmp_path, repo):
    verdict = {
        "failures": [
            {
                "failure": KEY,
                "verdict": "test-outdated",
                "reason": "subtraction is intended",
                "evidence": [
                    {"file": "calc.py", "lines": "2", "why": "changed operator"}
                ],
            }
        ]
    }
    weakened = TEST.replace("    assert add(2, 2) == 4\n", "    pass\n")
    agent = _agent({"tests/test_calc.py": weakened}, verdict)
    head_before = _git(repo, "rev-parse", "HEAD")
    state = _setup(monkeypatch, tmp_path, repo, sandbox=CalcSandbox(), agent=agent)

    assert state["outcome"] == rpt.REJECTED
    assert any(
        "reduces the number of assertions" in v
        for v in state["validation"]["violations"]
    )
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "status", "--porcelain") == ""


def test_fix_that_does_not_pass_tests_is_rejected(monkeypatch, tmp_path, repo):
    verdict = {"failures": [{"failure": KEY, "verdict": "code-fix", "reason": "r"}]}
    wrong = "def add(a, b):\n    return a * b\n"
    state = _setup(
        monkeypatch,
        tmp_path,
        repo,
        sandbox=CalcSandbox(),
        agent=_agent({"calc.py": wrong}, verdict),
    )
    assert state["outcome"] == rpt.REJECTED
    assert any("still fails" in v for v in state["validation"]["violations"])


def test_flaky_failures_skip_the_agent(monkeypatch, tmp_path, repo):
    def no_agent(*args, **kwargs):
        raise AssertionError("agent must not run")

    state = _setup(
        monkeypatch, tmp_path, repo, sandbox=CalcSandbox(flaky=True), agent=no_agent
    )
    assert state["outcome"] == rpt.NOTHING_TO_FIX
    assert state["classified"][0]["status"] == "flaky"


def test_agent_error_is_reported_as_unavailable(monkeypatch, tmp_path, repo):
    agent = _agent(
        {"calc.py": GOOD}, None, exit_status="error: APIConnectionError: down"
    )
    state = _setup(monkeypatch, tmp_path, repo, sandbox=CalcSandbox(), agent=agent)
    assert state["outcome"] == rpt.AGENT_UNAVAILABLE
    assert _git(repo, "status", "--porcelain") == ""


def test_missing_verdict_becomes_needs_human(monkeypatch, tmp_path, repo):
    agent = _agent({"calc.py": GOOD}, None, exit_status="LimitsExceeded")
    state = _setup(monkeypatch, tmp_path, repo, sandbox=CalcSandbox(), agent=agent)
    assert state["outcome"] == rpt.NEEDS_HUMAN
    assert state["verdict"]["failures"][0]["verdict"] == "needs-human"
    assert _git(repo, "status", "--porcelain") == ""
