# CI Autofix Agent Design

**Date:** 2026-10-21 (simplified 2026-10-27)
**Status:** Implemented in dry-run mode; push mode is configuration only

## Goal

When CI fails on a pull request, an agent works out whether the PR caused the
failure and, if so, proposes a small fix. In dry-run mode it comments the diff;
in push mode it commits the fix to the PR branch so CI runs again.

## Decisions

| Topic | Decision |
|---|---|
| Harness | [mini-swe-agent](https://github.com/swe-agent/mini-swe-agent), bash only, LiteLLM |
| Model | Any OpenAI-compatible endpoint, set by repo variables (`AUTOFIX_MODEL`, `AUTOFIX_API_BASE`, `AUTOFIX_API_KEY`) |
| Scope | Any failed job in `ci.yml`; the agent reads the failed log and the CI commands itself |
| Trigger | Opt-in: `ci-autofix` label (runs after each failed CI run) or `/autofix` comment from a user with write access |
| Delivery | `AUTOFIX_MODE=dry-run` (default) comments the diff; `push` adds a commit to the PR branch. Same-repo PRs only |
| Cap | Skip a PR that already has two `ci-fix(agent):` commits |

## How it works

One workflow, `.github/workflows/ci-autofix.yml`, with two jobs.

1. **Gate** (shell and `gh`): resolve the PR, then check that it is open, from
   this repository, labeled or explicitly requested, under the cap, and has a
   failed `CI` run for its current head.
2. **Fix**: check out the PR head without credentials, copy it without git
   metadata to `work/`, download the failed logs, and run mini-swe-agent in the
   `ci-autofix-env` container with only `work/` mounted. The agent edits files and
   writes `summary.md`. Then:
   - `guard.py` checks the staged diff (rules below).
   - In push mode, a separate step mints a GitHub App token, commits, and pushes
     without force. The push retriggers CI, which is the real verification.
   - A final step updates one PR comment with the summary, guard notes, and diff.

The prompt asks the agent to fix source for real regressions, to update a test
only when the PR intentionally changed that behavior (quoting the evidence), and
to make no edits when the cause is unclear. A person reads the comment; there is
no verdict schema.

## Security rules

- The workflow and `.github/autofix/` always run from the default branch for
  `workflow_run` and `issue_comment`, so a PR cannot change the agent's
  permissions or prompt. Iterate with `gh workflow run ci-autofix.yml --ref <branch> -f pr_number=<n>`;
  the workflow must already exist on the default branch.
- The agent never holds the push token or the model key. The key stays in the
  host process running LiteLLM; the container gets no secrets.
- The container sees a copy of the tree without `.git`. After it exits, git
  runs on the trusted clone's `.git` with that copy as the work tree, so nothing
  the agent wrote (hooks, config, attributes) executes on the host.
- Pushes use a GitHub App token, because pushes made with `GITHUB_TOKEN` do not
  trigger workflows. copy-pr-bot must sync the bot commit to `pull-request/N`;
  verify this before enabling push mode.

### Diff guard (`guard.py`)

Rejects a diff that:

- touches any path component starting with `.git` (`.github/`, `.gitattributes`,
  `.gitmodules`), `Cargo.lock`, or `uv.lock`;
- adds symlinks, submodules, or binary files;
- deletes a test file, removes a test function, or adds a skip/xfail/ignore marker;
- changes more than 400 lines.

Changed test files are listed in the PR comment for the reviewer.

## Deliberately left out

These were in an earlier, larger design. Add one only if real failures show the need.

| Piece | Add it when |
|---|---|
| Deterministic triage (rerun on head, rerun on merge-base to drop flaky and pre-existing failures) | Many agent runs are wasted on flaky or already-broken tests |
| Per-job rerun and suite commands (`jobs.yaml`) and host-side verification | The post-push CI run is too slow a check, or fixes often break other tests |
| Assertion-count and trivial-assert heuristics, tolerance checks | Reviewers see weakened tests that the guard missed |
| Structured verdicts, separate test-update commit, `ci-autofix:test-changed` label | Branch protection should block merges on agent-edited tests |
| Phase 2 (Lepton/GPU QA) and phase 3 (e2e suites) | The CPU jobs are working well |
