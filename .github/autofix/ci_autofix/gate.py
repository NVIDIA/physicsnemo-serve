# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve which PR and CI run a trigger refers to, and decide whether to act."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ci_autofix import (
    AGENT_COMMIT_PREFIX,
    AUTOFIX_LABEL,
    MAX_AGENT_ATTEMPTS,
    RUN_TRAILER,
)

CI_WORKFLOW = "ci.yml"
COMMAND = "/autofix"
WRITE_PERMISSIONS = frozenset({"admin", "maintain", "write"})
_MIRROR_BRANCH = re.compile(r"^pull-request/(\d+)$")


@dataclass
class Target:
    pr_number: int
    run_id: int
    head_sha: str
    head_ref: str
    base_ref: str
    trigger: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """Outcome of the gate.

    ``notify`` says whether the PR should get a comment explaining a skip; silent
    skips cover runs nobody asked the agent to look at.
    """

    proceed: bool
    reason: str
    target: Target | None = None
    pr_number: int | None = None
    notify: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["target"] = self.target.to_dict() if self.target else None
        return data


def _skip(reason: str, pr_number: int | None = None, notify: bool = False) -> Decision:
    return Decision(proceed=False, reason=reason, pr_number=pr_number, notify=notify)


def is_autofix_command(body: str) -> bool:
    lines = body.strip().splitlines()
    return bool(lines) and lines[0].strip() == COMMAND


def agent_attempts(commits: list[dict]) -> int:
    """Number of distinct autofix runs that pushed commits to the PR."""
    runs = set()
    for commit in commits:
        message = commit["commit"]["message"]
        if not message.startswith(AGENT_COMMIT_PREFIX):
            continue
        trailer = next(
            (line for line in message.splitlines() if line.startswith(RUN_TRAILER)),
            commit["sha"],
        )
        runs.add(trailer)
    return len(runs)


def latest_failed_run(gh, pr_number: int, head_sha: str) -> dict | None:
    runs = gh.get(
        f"/repos/{{repo}}/actions/workflows/{CI_WORKFLOW}/runs",
        {
            "branch": f"pull-request/{pr_number}",
            "head_sha": head_sha,
            "status": "failure",
        },
    )["workflow_runs"]
    return runs[0] if runs else None


def decide(event_name: str, event: dict, inputs: dict, gh) -> Decision:
    """Apply the eligibility rules from the design to one trigger."""
    explicit = event_name != "workflow_run"

    if event_name == "workflow_run":
        run = event["workflow_run"]
        match = _MIRROR_BRANCH.match(run.get("head_branch") or "")
        if not match:
            return _skip("not a pull-request run")
        pr_number = int(match.group(1))
        if run.get("conclusion") != "failure":
            return _skip(f"CI concluded {run.get('conclusion')}", pr_number)
    elif event_name == "issue_comment":
        issue = event["issue"]
        if "pull_request" not in issue or not is_autofix_command(
            event["comment"]["body"]
        ):
            return _skip("not an /autofix command on a PR")
        pr_number = int(issue["number"])
        user = event["comment"]["user"]["login"]
        permission = gh.get(f"/repos/{{repo}}/collaborators/{user}/permission")[
            "permission"
        ]
        if permission not in WRITE_PERMISSIONS:
            return _skip(f"@{user} lacks write access", pr_number, notify=True)
        run = None
    elif event_name == "workflow_dispatch":
        pr_number = int(inputs["pr_number"])
        run = gh.get(f"/repos/{{repo}}/actions/runs/{int(inputs['ci_run_id'])}")
    else:
        return _skip(f"unsupported event {event_name}")

    pr = gh.get(f"/repos/{{repo}}/pulls/{pr_number}")
    if pr["state"] != "open":
        return _skip("PR is not open", pr_number)
    labels = {label["name"] for label in pr["labels"]}
    if not explicit and AUTOFIX_LABEL not in labels:
        return _skip(f"PR lacks the {AUTOFIX_LABEL} label", pr_number)
    if (
        pr["head"]["repo"] is None
        or pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]
    ):
        return _skip("PRs from forks are not supported yet", pr_number, notify=True)

    head_sha = pr["head"]["sha"]
    if run is None:
        run = latest_failed_run(gh, pr_number, head_sha)
        if run is None:
            return _skip("no failed CI run for the PR head", pr_number, notify=True)
    if run["head_sha"] != head_sha:
        return _skip(
            "the PR has new commits since this CI run", pr_number, notify=explicit
        )

    attempts = agent_attempts(gh.paginate(f"/repos/{{repo}}/pulls/{pr_number}/commits"))
    if attempts >= MAX_AGENT_ATTEMPTS:
        return _skip(
            f"the agent already pushed fixes from {attempts} runs on this PR",
            pr_number,
            notify=True,
        )

    target = Target(
        pr_number=pr_number,
        run_id=int(run["id"]),
        head_sha=head_sha,
        head_ref=pr["head"]["ref"],
        base_ref=pr["base"]["ref"],
        trigger=event_name,
    )
    return Decision(proceed=True, reason="eligible", target=target, pr_number=pr_number)
