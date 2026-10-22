# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the single PR comment that reports an autofix run."""

from __future__ import annotations

from ci_autofix import COMMENT_MARKER

# Outcomes written by the CLI into state["outcome"].
SKIPPED = "skipped"
NOTHING_TO_FIX = "nothing-to-fix"
NEEDS_HUMAN = "needs-human"
AGENT_UNAVAILABLE = "agent-unavailable"
REJECTED = "rejected"
FIXED = "fixed"
ERROR = "error"

HEADLINES = {
    SKIPPED: "⏭️ CI autofix skipped",
    NOTHING_TO_FIX: "✅ No failures caused by this PR",
    NEEDS_HUMAN: "🙋 CI autofix needs a human",
    AGENT_UNAVAILABLE: "⚠️ CI autofix agent unavailable",
    REJECTED: "🚫 CI autofix discarded its fix",
    FIXED: "🔧 CI autofix pushed a fix",
    ERROR: "⚠️ CI autofix failed",
}
MAX_DIFF_CHARS = 20_000


def _triage_table(classified: list[dict]) -> list[str]:
    if not classified:
        return []
    rows = ["| Failure | Triage | Detail |", "|---|---|---|"]
    for item in classified:
        failure = item["failure"]
        key = (
            f"{failure['job']} :: {failure['test_id']}"
            if failure.get("test_id")
            else failure["job"]
        )
        rows.append(f"| `{key}` | {item['status']} | {item.get('detail', '')} |")
    return ["### Triage", *rows]


def _verdict_section(verdict: dict | None) -> list[str]:
    if not verdict:
        return []
    lines = ["### Agent verdict"]
    if verdict.get("summary"):
        lines.append(verdict["summary"])
    test_updates = []
    for entry in verdict["failures"]:
        lines.append(
            f"- `{entry['failure']}`: **{entry['verdict']}**. {entry['reason']}"
        )
        if entry["verdict"] == "test-outdated":
            test_updates.append(entry)
    if test_updates:
        lines += ["", "#### ⚠️ Tests updated: please review", ""]
        for entry in test_updates:
            lines.append(f"`{entry['failure']}` was updated because the PR changed:")
            for item in entry.get("evidence", []):
                lines.append(f"- `{item['file']}` lines {item['lines']}: {item['why']}")
    return lines


def render(state: dict) -> str:
    outcome = state.get("outcome", ERROR)
    mode = state.get("mode", "dry-run")
    headline = HEADLINES.get(outcome, HEADLINES[ERROR])
    push = state.get("push_result")
    if outcome == FIXED and mode != "push":
        headline = "🔍 CI autofix proposes a fix (dry run, not pushed)"
    elif outcome == FIXED and push != "pushed":
        headline = "⚠️ CI autofix could not push its fix"

    lines = [COMMENT_MARKER, f"## {headline}", ""]
    if state.get("outcome_detail"):
        lines += [state["outcome_detail"], ""]
    if push == "branch-moved":
        lines += [
            "The branch changed while the agent worked. Comment `/autofix` to retry.",
            "",
        ]

    lines += _triage_table(state.get("classified", []))
    unparsed = state.get("unparsed_jobs", [])
    if unparsed:
        lines += ["", f"Could not identify failing tests in: {', '.join(unparsed)}."]
    lines += ["", *_verdict_section(state.get("verdict"))]

    violations = state.get("validation", {}).get("violations", [])
    if violations:
        lines += ["", "### Rejected because", *[f"- {v}" for v in violations]]

    commits = state.get("commits", [])
    if commits and outcome == FIXED:
        lines += ["", "### Commits", *[f"- {c['title']}" for c in commits]]
        diff = state.get("diff", "")
        if diff:
            shown = diff[:MAX_DIFF_CHARS]
            more = "\n... (truncated)" if len(diff) > MAX_DIFF_CHARS else ""
            lines += [
                "",
                "<details><summary>Diff</summary>",
                "",
                "```diff",
                shown + more,
                "```",
                "</details>",
            ]

    if state.get("flaky"):
        lines += [
            "",
            "Flaky failures were not fixed. Re-run the failed CI jobs to clear them.",
        ]
    if state.get("run_url"):
        lines += ["", f"[Autofix run, trajectory and artifacts]({state['run_url']})"]
    return "\n".join(lines).strip() + "\n"


def upsert_comment(gh, pr_number: int, body: str) -> None:
    """Update the autofix comment on the PR, or create it."""
    comments = gh.paginate(f"/repos/{{repo}}/issues/{pr_number}/comments")
    existing = next(
        (c for c in comments if COMMENT_MARKER in (c.get("body") or "")), None
    )
    if existing:
        gh.patch(f"/repos/{{repo}}/issues/comments/{existing['id']}", {"body": body})
    else:
        gh.post(f"/repos/{{repo}}/issues/{pr_number}/comments", {"body": body})
