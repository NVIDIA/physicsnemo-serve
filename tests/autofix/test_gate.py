# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ci_autofix.gate import agent_attempts, decide, is_autofix_command

REPO = "NVIDIA/physicsnemo-serve"
SHA = "a" * 40


class FakeGitHub:
    def __init__(
        self,
        *,
        labels=("ci-autofix",),
        fork=False,
        head_sha=SHA,
        commits=(),
        permission="write",
        runs=None,
    ):
        head_repo = {"full_name": "someone/fork" if fork else REPO}
        self.pr = {
            "state": "open",
            "labels": [{"name": n} for n in labels],
            "head": {"sha": head_sha, "ref": "feature", "repo": head_repo},
            "base": {"ref": "main", "repo": {"full_name": REPO}},
        }
        self.commits = [
            {"sha": f"c{i}", "commit": {"message": m}} for i, m in enumerate(commits)
        ]
        self.permission = permission
        self.runs = runs if runs is not None else [{"id": 77, "head_sha": SHA}]

    def get(self, path, params=None):
        if path.endswith("/permission"):
            return {"permission": self.permission}
        if "/actions/workflows/" in path:
            return {"workflow_runs": self.runs}
        if "/actions/runs/" in path:
            return {"id": int(path.rsplit("/", 1)[1]), "head_sha": SHA}
        if "/pulls/" in path:
            return self.pr
        raise AssertionError(path)

    def paginate(self, path, key=None, params=None):
        return self.commits


def _run_event(conclusion="failure", branch="pull-request/12"):
    return {
        "workflow_run": {
            "id": 55,
            "head_branch": branch,
            "conclusion": conclusion,
            "head_sha": SHA,
        }
    }


def _comment_event(body="/autofix", user="dev"):
    return {
        "issue": {"number": 12, "pull_request": {}},
        "comment": {"body": body, "user": {"login": user}},
    }


def test_workflow_run_failure_with_label_proceeds():
    decision = decide("workflow_run", _run_event(), {}, FakeGitHub())
    assert decision.proceed
    assert decision.target.pr_number == 12
    assert decision.target.run_id == 55
    assert decision.target.head_ref == "feature"


def test_workflow_run_skips_silently_without_label_or_on_success():
    assert not decide("workflow_run", _run_event(), {}, FakeGitHub(labels=())).notify
    assert not decide(
        "workflow_run", _run_event(conclusion="success"), {}, FakeGitHub()
    ).proceed
    assert not decide(
        "workflow_run", _run_event(branch="main"), {}, FakeGitHub()
    ).proceed


def test_fork_is_refused_with_comment():
    decision = decide("workflow_run", _run_event(), {}, FakeGitHub(fork=True))
    assert not decision.proceed and decision.notify


def test_stale_run_is_skipped():
    decision = decide("workflow_run", _run_event(), {}, FakeGitHub(head_sha="b" * 40))
    assert not decision.proceed and not decision.notify


def test_attempt_cap_counts_runs_not_commits():
    commits = [
        "ci-fix(agent): fix x\n\nAutofix-Run: 1",
        "ci-fix(agent): update outdated test y\n\nAutofix-Run: 1",
        "feat: unrelated",
    ]
    assert (
        agent_attempts(
            [{"sha": str(i), "commit": {"message": m}} for i, m in enumerate(commits)]
        )
        == 1
    )
    assert decide("workflow_run", _run_event(), {}, FakeGitHub(commits=commits)).proceed

    commits.append("ci-fix(agent): again\n\nAutofix-Run: 2")
    decision = decide("workflow_run", _run_event(), {}, FakeGitHub(commits=commits))
    assert not decision.proceed and decision.notify


def test_comment_requires_write_access_and_needs_no_label():
    assert decide("issue_comment", _comment_event(), {}, FakeGitHub(labels=())).proceed
    decision = decide(
        "issue_comment", _comment_event(), {}, FakeGitHub(permission="read")
    )
    assert not decision.proceed and decision.notify


def test_comment_without_failed_run_is_reported():
    decision = decide("issue_comment", _comment_event(), {}, FakeGitHub(runs=[]))
    assert not decision.proceed and decision.notify


def test_dispatch_uses_inputs():
    decision = decide(
        "workflow_dispatch",
        {},
        {"pr_number": "12", "ci_run_id": "99"},
        FakeGitHub(labels=()),
    )
    assert decision.proceed and decision.target.run_id == 99


def test_is_autofix_command():
    assert is_autofix_command("  /autofix\nplease")
    assert not is_autofix_command("can you /autofix")
    assert not is_autofix_command("")
