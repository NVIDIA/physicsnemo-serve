# CI Autofix Agent Implementation Plan

**Design:** [2026-10-21-ci-autofix-agent-design.md](2026-10-21-ci-autofix-agent-design.md)
**Branch:** `feat/ci-autofix-agent`

This plan delivers rollout step 1 (dry run) with the push path behind a repo
variable, so step 2 needs only configuration.

## Layout

```
.github/
  workflows/ci-autofix.yml        # workflow_run / issue_comment / workflow_dispatch
  autofix/
    agent.yaml                    # mini-swe-agent templates and limits
    jobs.yaml                     # in-scope CI jobs and their rerun commands
    Dockerfile                    # ci-autofix-env sandbox image
    ci_autofix/                   # Python package (stdlib + PyYAML; mini-swe-agent only in agent.py)
      gate.py  failures.py  jobs.py  triage.py  verdict.py
      diff_rules.py  agent.py  report.py  cli.py
tests/autofix/                    # unit tests, run by the existing python-test job
```

`pyproject.toml` adds `.github/autofix` to pytest's `pythonpath`.

## Tasks

Each task lands with unit tests and its own commit.

1. **Skeleton.** Package, pytest path, empty test module.
2. **Job registry** (`jobs.yaml`, `jobs.py`). Maps CI job display names
   (`Python Tests`, `Rust Test`, `Model Builder and Native SDK CPU`,
   `Rust Clippy`, `Rust Format`) to a test-ID kind and command templates for
   rerunning one test and the job's suite.
3. **Failure extraction** (`failures.py`). Split `gh run view --log-failed`
   output by job; extract pytest node IDs, cargo test names, and unittest IDs;
   keep a trimmed log excerpt per job.
4. **Gate** (`gate.py`). Resolve the PR number from the trigger; check label or
   comment authorization, same-repo head, agent-commit cap, and head SHA.
5. **Triage** (`triage.py`). Classify each failure as `flaky`, `pre-existing`,
   or `reproducible` using an injectable command runner.
6. **Verdict schema** (`verdict.py`). Parse and validate `verdict.json`.
7. **Diff rules** (`diff_rules.py`). Path allowlist per verdict, size limit,
   no deleted tests, no new skip/xfail/ignore markers, non-decreasing assert
   count, golden files only beside failing tests.
8. **Agent runner** (`agent.py`, `agent.yaml`). Build the task prompt, run
   mini-swe-agent's `DefaultAgent` with a LiteLLM model and Docker environment,
   save the trajectory.
9. **Report** (`report.py`). Render the PR comment with a hidden marker so the
   workflow updates one comment in place.
10. **CLI** (`cli.py`). Subcommands the workflow calls: `gate`, `collect`,
    `triage`, `agent`, `validate`, `commit`, `report`.
11. **Sandbox image** (`Dockerfile`). Rust 1.94.1, uv, redis, cmake, Python.
12. **Workflow** (`ci-autofix.yml`). `AUTOFIX_MODE` repo variable: `dry-run`
    (default) comments the diff; `push` commits and pushes with a GitHub App
    token.

## Deferred

- Replay eval set and `eval_mode` (design Testing items 2–3): next PR, once the
  dry run produces real trajectories to seed cases.
- Moving `ci.yml` onto a shared `setup-ci-env` action: the sandbox image covers
  the agent's needs; `ci.yml` stays untouched in this PR.

## Prerequisites outside the repo

- GitHub App with `contents: write` and `pull-requests: write`, stored as
  `vars.AUTOFIX_APP_ID` and `secrets.AUTOFIX_APP_PRIVATE_KEY`.
- `vars.AUTOFIX_MODEL`, `vars.AUTOFIX_API_BASE`, `secrets.AUTOFIX_API_KEY`.
- The `ci-autofix` and `ci-autofix:test-changed` labels.
