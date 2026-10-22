# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry points called by .github/workflows/ci-autofix.yml.

    gate    decide whether to act; writes step outputs for the checkout step
    fix     collect, triage, run the agent, validate, and commit locally
    report  update the PR comment (and label) after the push step

Stages share state through ``<state-dir>/state.json``. ``fix`` never sees a
token that can push: the workflow mints one only for the push step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ci_autofix import (
    AGENT_COMMIT_PREFIX,
    RUN_TRAILER,
    TEST_CHANGED_LABEL,
)
from ci_autofix import report as rpt
from ci_autofix.diff_rules import parse_diff, validate
from ci_autofix.failures import (
    Failure,
    JobFailures,
    clean_log,
    extract_ids,
    parse_job_log,
)
from ci_autofix.gate import decide
from ci_autofix.github import GitHub
from ci_autofix.jobs import JobSpec, load_jobs, rerun_commands, suite_command
from ci_autofix.sandbox import DockerSandbox
from ci_autofix.triage import AUTOFIXED, FLAKY, REPRODUCIBLE, triage
from ci_autofix.verdict import (
    CODE_FIX,
    FIX_VERDICTS,
    NOT_CAUSED,
    Entry,
    Verdict,
    VerdictError,
    needs_human_for_all,
    parse_verdict,
)

OUTPUT_TAIL = 4_000


# --------------------------------------------------------------------------- state


def load_state(state_dir: Path) -> dict:
    path = state_dir / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(state_dir: Path, state: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(json.dumps(state, indent=2))


def github_from_env() -> GitHub:
    return GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])


def run_url() -> str | None:
    server, repo, run = (
        os.environ.get(k)
        for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None


def write_outputs(values: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in values.items()]
    if path:
        with open(path, "a") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


# --------------------------------------------------------------------------- git


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def discard_changes(repo: Path) -> None:
    git(repo, "reset", "--hard", "-q", "HEAD")
    git(repo, "clean", "-fdq")


def export_tree(repo: Path, rev: str, dest: Path) -> None:
    """Materialise ``rev`` without git metadata, so the sandbox can mount it."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "-C", str(repo), "archive", rev], stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(dest, filter="data")


# --------------------------------------------------------------------------- gate


def cmd_gate(args: argparse.Namespace) -> int:
    event = json.loads(Path(args.event_path).read_text()) if args.event_path else {}
    inputs = json.loads(args.inputs or "{}")
    decision = decide(args.event_name, event, inputs, github_from_env())
    state = {"decision": decision.to_dict(), "mode": args.mode, "run_url": run_url()}
    if not decision.proceed:
        state["outcome"] = rpt.SKIPPED
        state["outcome_detail"] = f"Skipped: {decision.reason}."
    save_state(Path(args.state_dir), state)

    target = decision.target
    write_outputs(
        {
            "proceed": str(decision.proceed).lower(),
            "notify": str(decision.notify).lower(),
            "pr_number": decision.pr_number or "",
            "head_sha": target.head_sha if target else "",
            "head_ref": target.head_ref if target else "",
            "base_ref": target.base_ref if target else "",
        }
    )
    print(f"gate: proceed={decision.proceed} reason={decision.reason}")
    return 0


# --------------------------------------------------------------------------- fix


def collect(
    gh: GitHub, run_id: int, specs: dict[str, JobSpec]
) -> tuple[list[JobFailures], list[str]]:
    """Failed in-scope jobs with their failures, and failed out-of-scope job names."""
    jobs, out_of_scope = [], []
    for job in gh.paginate(f"/repos/{{repo}}/actions/runs/{run_id}/jobs", key="jobs"):
        if job.get("conclusion") != "failure":
            continue
        spec = specs.get(job["name"])
        if spec is None:
            out_of_scope.append(job["name"])
            continue
        log = gh.get_text(f"/repos/{{repo}}/actions/jobs/{job['id']}/logs")
        jobs.append(parse_job_log(job["name"], log, spec.kinds))
    return jobs, out_of_scope


def verify(
    sandbox,
    specs: dict[str, JobSpec],
    fixed: list[Failure],
    known_failing: list[Failure],
    head: Path,
) -> tuple[bool, str]:
    """Rerun fixed failures, then each affected job's suite.

    The suite may still fail on failures the agent did not claim to fix; any
    other failure means the fix broke something.
    """
    for job_name in dict.fromkeys(f.job for f in fixed):
        spec = specs[job_name]
        job_fixed = [f for f in fixed if f.job == job_name]
        for command in rerun_commands(spec, job_fixed, head):
            result = sandbox.run(command, head)
            if not result.ok:
                return (
                    False,
                    f"`{command.script}` still fails:\n{result.output[-OUTPUT_TAIL:]}",
                )
        result = sandbox.run(suite_command(spec), head)
        if result.ok:
            continue
        allowed = {f.test_id for f in known_failing if f.job == job_name and f.test_id}
        lines = clean_log(result.output)
        failing = {
            i
            for kind in spec.kinds
            if kind not in ("clippy", "fmt", "cargo-build")
            for i in extract_ids(lines, kind)
        }
        if not failing or not failing <= allowed:
            new = sorted(failing - allowed) or [
                "(suite failed without identifiable tests)"
            ]
            return (
                False,
                f"`{spec.suite}` has new failures: {new}\n{result.output[-OUTPUT_TAIL:]}",
            )
    return True, ""


def commit_fix(
    head: Path,
    pr_head: str,
    validation,
    verdict: Verdict,
    run_id: str,
    failures_by_key: dict,
) -> list[dict]:
    """Commit source fixes and test updates separately on top of the PR head."""
    identity = [
        "-c",
        f"user.name={os.environ.get('AUTOFIX_GIT_NAME', 'ci-autofix[bot]')}",
        "-c",
        f"user.email={os.environ.get('AUTOFIX_GIT_EMAIL', 'ci-autofix[bot]@users.noreply.github.com')}",
    ]
    # Unwind the local autofix commit, keeping the working tree.
    git(head, "reset", "-q", pr_head)
    commits = []
    groups = [
        (
            validation.source_files,
            [e for e in verdict.entries if e.verdict == CODE_FIX],
            "fix",
        ),
        (
            validation.test_files,
            [e for e in verdict.entries if e.verdict == "test-outdated"],
            "test",
        ),
    ]
    for files, entries, kind in groups:
        if not files:
            continue
        git(head, "add", "-A", "--", *files)
        names = ", ".join(
            failures_by_key[e.failure].test_id or e.failure for e in entries
        )
        title = (
            f"{AGENT_COMMIT_PREFIX} fix CI failures caused by this PR"
            if kind == "fix"
            else f"{AGENT_COMMIT_PREFIX} update outdated test {names}"
        )[:120]
        body = "\n".join(f"- {e.failure}: {e.reason}" for e in entries)
        git(
            head,
            *identity,
            "commit",
            "-q",
            "-m",
            title,
            "-m",
            body,
            "-m",
            f"{RUN_TRAILER} {run_id}",
        )
        commits.append(
            {
                "sha": git(head, "rev-parse", "HEAD").strip(),
                "title": title,
                "kind": kind,
            }
        )
    return commits


def finish(state_dir: Path, state: dict, outcome: str, detail: str = "") -> int:
    state["outcome"] = outcome
    if detail:
        state["outcome_detail"] = detail
    save_state(state_dir, state)
    print(f"fix: outcome={outcome} {detail}")
    write_outputs({"outcome": outcome, "commits": len(state.get("commits", []))})
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    state = load_state(state_dir)
    target = state["decision"]["target"]
    head = Path(args.head).resolve()
    specs = load_jobs()
    gh = github_from_env()
    sandbox = DockerSandbox(args.image, state_dir / "cache")

    jobs, out_of_scope = collect(gh, target["run_id"], specs)
    state["jobs"] = [j.to_dict() for j in jobs]
    state["out_of_scope_jobs"] = out_of_scope
    state["unparsed_jobs"] = [j.job for j in jobs if not j.parsed]
    if not jobs:
        detail = "No job in the autofix scope failed."
        if out_of_scope:
            detail += f" Out of scope: {', '.join(out_of_scope)}."
        return finish(state_dir, state, rpt.NOTHING_TO_FIX, detail)

    merge_base = git(head, "merge-base", "HEAD", f"origin/{target['base_ref']}").strip()
    base = state_dir / "base"
    export_tree(head, merge_base, base)

    classified = triage(sandbox, specs, [j for j in jobs if j.parsed], head, base)
    state["classified"] = [c.to_dict() for c in classified]
    state["flaky"] = any(c.status == FLAKY for c in classified)
    reproducible = [c.failure for c in classified if c.status == REPRODUCIBLE]
    autofixed = [c.failure for c in classified if c.status == AUTOFIXED]
    failures_by_key = {c.failure.key: c.failure for c in classified}

    if not reproducible and not autofixed:
        if state["unparsed_jobs"] or any(c.status == "unrunnable" for c in classified):
            return finish(
                state_dir,
                state,
                rpt.NEEDS_HUMAN,
                "Some failures could not be rerun automatically.",
            )
        return finish(
            state_dir,
            state,
            rpt.NOTHING_TO_FIX,
            "Every failure is flaky or already fails on the target branch.",
        )

    if autofixed:
        # Park the formatter/linter fixes in a local commit so discarding the
        # agent's work later does not discard them too. commit_fix unwinds it.
        git(head, "add", "-A")
        git(
            head,
            "-c",
            "user.name=ci-autofix",
            "-c",
            "user.email=ci-autofix@localhost",
            "commit",
            "-qm",
            "wip: autofix",
        )

    verdict = Verdict("", [])
    if reproducible:
        pr = gh.get(f"/repos/{{repo}}/pulls/{target['pr_number']}")
        commits = gh.paginate(f"/repos/{{repo}}/pulls/{target['pr_number']}/commits")
        from ci_autofix.agent import PullRequest, build_task, run_agent

        pull = PullRequest(
            number=target["pr_number"],
            title=pr["title"],
            body=pr.get("body") or "",
            commit_messages=[c["commit"]["message"] for c in commits],
            diff=git(head, "diff", merge_base, target["head_sha"]),
        )
        agents_md = head / "AGENTS.md"
        task = build_task(
            pull,
            classified,
            jobs,
            specs,
            head,
            agents_md.read_text() if agents_md.is_file() else "",
        )
        (state_dir / "task.md").write_text(task)
        outcome = run_agent(
            task,
            image=args.image,
            workspace=head,
            out_dir=state_dir / "agent",
            cache_dir=state_dir / "cache",
            model_name=os.environ["AUTOFIX_MODEL"],
            api_base=os.environ["AUTOFIX_API_BASE"],
            api_key=os.environ["AUTOFIX_API_KEY"],
            model_class=os.environ.get("AUTOFIX_MODEL_CLASS") or "litellm",
        )
        state["agent"] = {"exit_status": outcome.exit_status}
        keys = [f.key for f in reproducible]
        if outcome.exit_status.startswith("error:"):
            discard_changes(head)
            return finish(
                state_dir,
                state,
                rpt.AGENT_UNAVAILABLE,
                f"The agent did not complete: {outcome.exit_status}",
            )
        try:
            if outcome.verdict_text is None:
                raise VerdictError(
                    f"agent exited ({outcome.exit_status}) without writing verdict.json"
                )
            verdict = parse_verdict(outcome.verdict_text, keys)
        except VerdictError as error:
            verdict = needs_human_for_all(keys, str(error))
            discard_changes(head)
        state["verdict"] = verdict.to_dict()

    # Deterministic fixes (cargo fmt, clippy --fix) count as code fixes.
    full = Verdict(
        verdict.summary,
        verdict.entries
        + [
            Entry(f.key, CODE_FIX, "fixed by the job's formatter or linter")
            for f in autofixed
        ],
    )

    pr_head = target["head_sha"]
    git(head, "add", "-A")
    changes = parse_diff(
        # --no-renames: a rename is a delete plus an add, so both paths get
        # validated and committed.
        git(head, "diff", "--cached", "--name-status", "--no-renames", pr_head),
        git(head, "diff", "--cached", "--no-renames", "-U0", pr_head),
    )
    pr_files = set(git(head, "diff", "--name-only", merge_base, pr_head).split())
    validation = validate(changes, full, failures_by_key, pr_files, head)
    state["validation"] = {
        "violations": validation.violations,
        "source_files": validation.source_files,
        "test_files": validation.test_files,
    }
    if not validation.ok:
        (state_dir / "rejected.diff").write_text(git(head, "diff", "--cached", pr_head))
        discard_changes(head)
        return finish(state_dir, state, rpt.REJECTED)
    if not changes:
        if verdict.entries and full.verdicts() == {NOT_CAUSED}:
            return finish(
                state_dir,
                state,
                rpt.NOTHING_TO_FIX,
                "The agent found no failure caused by this PR.",
            )
        return finish(state_dir, state, rpt.NEEDS_HUMAN)

    fixed = [
        failures_by_key[e.failure] for e in full.entries if e.verdict in FIX_VERDICTS
    ]
    known = [
        failures_by_key[e.failure]
        for e in full.entries
        if e.verdict not in FIX_VERDICTS
    ]
    known += [
        c.failure for c in classified if c.status not in (REPRODUCIBLE, AUTOFIXED)
    ]
    ok, detail = verify(sandbox, specs, fixed, known, head)
    if not ok:
        state["validation"]["violations"].append(detail)
        (state_dir / "rejected.diff").write_text(git(head, "diff", "--cached", pr_head))
        discard_changes(head)
        return finish(state_dir, state, rpt.REJECTED)

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    state["commits"] = commit_fix(
        head, pr_head, validation, full, run_id, failures_by_key
    )
    state["diff"] = git(head, "diff", pr_head, "HEAD")
    (state_dir / "fix.diff").write_text(state["diff"])
    return finish(state_dir, state, rpt.FIXED)


# --------------------------------------------------------------------------- report


def cmd_report(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    state = load_state(state_dir)
    decision = state.get("decision", {})
    if args.push_result:
        state["push_result"] = args.push_result
    if not state.get("outcome"):
        state["outcome"] = rpt.ERROR
        state["outcome_detail"] = "The run stopped unexpectedly. See the workflow logs."
    save_state(state_dir, state)

    pr_number = decision.get("pr_number")
    if not pr_number or not (decision.get("proceed") or decision.get("notify")):
        print("report: nothing to post")
        return 0
    gh = github_from_env()
    rpt.upsert_comment(gh, pr_number, rpt.render(state))
    tests_changed = any(c["kind"] == "test" for c in state.get("commits", []))
    if state.get("push_result") == "pushed" and tests_changed:
        gh.post(
            f"/repos/{{repo}}/issues/{pr_number}/labels",
            {"labels": [TEST_CHANGED_LABEL]},
        )
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci_autofix")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate")
    gate.add_argument("--event-name", required=True)
    gate.add_argument("--event-path")
    gate.add_argument("--inputs")
    gate.add_argument("--mode", choices=("dry-run", "push"), default="dry-run")
    gate.add_argument("--state-dir", required=True)
    gate.set_defaults(func=cmd_gate)

    fix = sub.add_parser("fix")
    fix.add_argument("--state-dir", required=True)
    fix.add_argument("--head", required=True, help="checkout of the PR head")
    fix.add_argument("--image", required=True, help="ci-autofix-env image")
    fix.set_defaults(func=cmd_fix)

    report = sub.add_parser("report")
    report.add_argument("--state-dir", required=True)
    report.add_argument(
        "--push-result", choices=("pushed", "branch-moved", "failed", "skipped")
    )
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
