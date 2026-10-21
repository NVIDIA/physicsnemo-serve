# CI Autofix Agent Design

**Date:** 2026-10-21
**Status:** Approved design, not yet implemented

## Goal

When CI fails on a pull request, an agent determines whether the PR caused the
failure, fixes the code (or an intentionally outdated test), commits the fix to
the PR branch, and triggers a new CI run.

## Decisions

| Topic | Decision |
|---|---|
| Agent harness | [mini-swe-agent](https://github.com/swe-agent/mini-swe-agent) (MIT, bash-only, LiteLLM) |
| Model | Hosted on internal NVIDIA OpenAI-compatible endpoints; swappable through repo variables |
| Phase 1 scope | CPU jobs in `ci.yml`: `python-test`, `rust-test`, `inference-cpu`, `rust-clippy`, `rust-fmt` |
| Later phases | Phase 2: Lepton/GPU QA (`lepton_qa.yml`, `lepton_native_qa.yml`). Phase 3: e2e integration suites |
| Fix delivery | Push a commit directly to the PR source branch; same-repo PRs only (forks skipped in phase 1) |
| Trigger | Opt-in: `ci-autofix` label or `/autofix` comment from a user with write access |
| Wiring | Separate `ci-autofix.yml` workflow triggered by `workflow_run` on `CI` |

`google-research/harness2` was considered and rejected: it optimizes agent
harnesses against benchmarks and is not a CI repair agent.

## Architecture

### Why a separate `workflow_run` workflow

`workflow_run` and `issue_comment` always execute the workflow file from
`main`. A PR therefore cannot change the agent's prompt, permissions, or
guardrails. The agent config under `.github/autofix/` is also checked out from
`main`, never from the PR head.

Alternatives rejected:

- A trailing `autofix` job in `ci.yml`: the PR controls the job definition, and
  `cancel-in-progress` kills in-flight fixes.
- An `if: failure()` step per CI job: concurrent agents race to push, and the PR
  controls the definition.

### Credential rule

The agent never holds the push credential. It works in a checkout made with
`persist-credentials: false` inside a container that receives no secrets. A
separate wrapper step validates the diff and performs the commit and push.

### Triggers

```yaml
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
  issue_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      ci_run_id: { description: "Failed CI run ID to fix", required: true }
      pr_number: { description: "PR number", required: true }
      eval_mode: { description: "Run the replay eval set; never push", type: boolean, default: false }
      model:     { description: "Override AUTOFIX_MODEL (eval only)", required: false }

concurrency:
  group: ci-autofix-pr-${{ <resolved PR number> }}
  cancel-in-progress: true
```

### Changing the workflow

- Changes to `ci-autofix.yml` and `.github/autofix/` take effect only after they
  merge to `main`.
- To iterate before merging, dispatch from a branch:
  `gh workflow run ci-autofix.yml --ref <branch> -f ci_run_id=<id> -f pr_number=<n>`.
  The workflow file must already exist on `main`, so the first PR merges a
  minimal version.
- Model and endpoint come from `vars.AUTOFIX_MODEL` and `vars.AUTOFIX_API_BASE`,
  and the API key from `secrets.AUTOFIX_API_KEY`. Switching models needs no code
  change.

### Retriggering CI

Pushes made with `GITHUB_TOKEN` do not trigger workflows. The wrapper pushes with
a GitHub App (or bot account) token. copy-pr-bot then syncs the commit to
`pull-request/N`, and `CI` runs again.

## Pipeline

A single job on `linux-amd64-cpu8`. Any stage can exit early with a PR comment.

1. **Resolve and gate.** Resolve the PR number from `pull-request/N`, the comment,
   or the dispatch input. Continue only if:
   - the PR has the `ci-autofix` label, or `/autofix` came from a user with write access;
   - the head branch lives in this repository (not a fork);
   - the PR has fewer than two `ci-fix(agent):` commits;
   - the failed run's SHA is still the PR head.
2. **Collect.** Run `gh run view <id> --log-failed`, filter to in-scope jobs,
   extract failing test IDs (pytest node IDs, cargo test names, unittest IDs),
   and trim logs to the relevant sections.
3. **Triage (deterministic).** Set up the environment with the shared
   `setup-ci-env` composite action, which `ci.yml` also uses.
   - Rerun each failing test on the PR head. A pass marks it **flaky**.
   - Run it on the merge-base with `main`. A failure there marks it **pre-existing**.
   - Only failures that reproduce on the head and pass on the merge-base go to the agent.
   - `rust-fmt` failures are fixed by `cargo fmt`. `rust-clippy` failures first
     try `cargo clippy --fix`, and fall through to the agent if lints remain.
4. **Agent.** mini-swe-agent runs in a credential-free checkout of the PR head,
   with a step limit (~60) and a ~20-minute wall-clock limit. It writes
   `verdict.json`.
5. **Validate (deterministic).** Apply the diff rules (see below), then rerun
   the failing tests and the rest of each affected job's suite.
6. **Commit, push, report.** Commit as the bot, push without force, comment on
   the PR with the diagnosis, a diff summary, and a link to the trajectory.

## Verdicts

The agent classifies each failure:

| Verdict | Meaning | May edit |
|---|---|---|
| `code-fix` | The PR broke behavior that should still work | Source files only |
| `test-outdated` | The PR intentionally changed behavior; the test encodes the old behavior | Only the failing tests and their fixtures/golden files |
| `not-caused-by-pr` | Triage missed it, or an environment problem | Nothing |
| `needs-human` | Intent is unclear or the fix is too large | Nothing |

`test-outdated` requires evidence: the agent must cite the specific lines in the
PR diff that intentionally changed the behavior, and explain why the new
expected value is correct. It receives the PR title, body, and commit messages
to judge intent. Without that link, it must return `needs-human`.

### Diff rules (non-negotiable)

- No changes under `.github/`.
- Diff size below a configured limit.
- `code-fix` changes touch no test files.
- `test-outdated` changes touch only test functions or files in the failure list.
- No deleted tests; no added `skip`, `xfail`, `#[ignore]`, or `pytest.mark.skip`.
- Assert count per modified test must not decrease. A heuristic rejects replacing
  an assertion with a weaker one, such as `assert True`.
- Golden or fixture files change only alongside a failing test that uses them.

### Commits and visibility

- Source fixes: `ci-fix(agent): <summary>`.
- Test updates: a separate commit, `ci-fix(agent): update outdated test <name>`,
  made after the source fix.
- Test updates appear in a flagged section of the PR comment that quotes the
  evidence lines, and the wrapper adds the `ci-autofix:test-changed` label.
  Branch protection can optionally require removing that label before merge.

## Agent

### Runtime and config

- mini-swe-agent pinned to an exact version, installed with `uv tool install`,
  isolated from the project environment.
- `.github/autofix/agent.yaml`: system and instance templates, step limit,
  timeouts.
- `.github/autofix/run_autofix.py`: a thin wrapper (~200 lines) that assembles
  inputs, runs `DefaultAgent`, and reads `verdict.json`.

```yaml
model:
  model_name: "openai/${AUTOFIX_MODEL}"
  model_kwargs:
    api_base: "${AUTOFIX_API_BASE}"
    temperature: 0.0
```

LiteLLM does not know internal model prices, so cost tracking is disabled
(`MSWEA_COST_TRACKING=ignore_errors`); step and time limits bound spend. The
model needs a context window of at least 128k tokens and strong coding ability.

### Sandbox

- Commands run in mini-swe-agent's Docker environment using a
  `ci-autofix-env` image built from the same definitions as `setup-ci-env`
  (Rust 1.94.1, uv, redis, cmake).
- The PR checkout is mounted read-write and holds no git credentials.
- No secrets enter the container. The model key stays in the host-side wrapper,
  where LiteLLM runs.
- Egress is limited to package registries where the runner allows it. If not,
  phase 1 relies on having no credentials inside the container.

### Prompt inputs

- Triaged failures with trimmed logs
- `git diff merge-base..HEAD`, PR title, body, and commit messages
- `AGENTS.md`
- The exact command each failing job runs, taken from `ci.yml`
- Rules: the four verdicts, the evidence requirement, no git commit/push, rerun
  failing tests before submitting

### Output

`/workspace/.autofix/verdict.json`, with one entry per failure: verdict,
reasoning, evidence lines. Edits outside the allowed set for the verdict fail
the run.

## Error handling

Each run maintains one PR comment, updated in place. Outcomes are also emitted
as job annotations.

| Situation | Behavior | Counts toward cap |
|---|---|---|
| Not eligible (fork, no label, cap reached) | Comment once, exit | No |
| All failures flaky or pre-existing | Comment with triage table; offer `/autofix rerun` for flaky | No |
| Environment setup or log download fails | Comment "infrastructure failure" with run link | No |
| Model endpoint error or timeout | Three retries with backoff, then "agent unavailable" | No |
| Step/time limit reached, or `verdict.json` missing/invalid | Treat as `needs-human`; upload partial trajectory | Yes |
| Diff fails validation | Discard; comment with the violated rule; upload trajectory | Yes |
| Tests still fail after the fix | Same as above | Yes |
| PR head moved during the run | Drop the fix; comment "branch updated, re-run /autofix" | No |
| Push rejected | Comment and exit; no retry | No |
| `CI` fails again after an agent commit | Runs again until two agent commits exist, then `needs-human` | — |

Artifacts kept for 30 days: triage report, trajectory (viewable in
mini-swe-agent's inspector), `verdict.json`, and the applied or rejected diff.

## Testing

1. **Unit tests** in `tests/autofix/` for deterministic logic: PR resolution and
   gating, test-ID extraction from logs, and diff rules (assert-count heuristic,
   skip/xfail detection, path allowlist). They run in the existing `python-test`
   job.
2. **Replay eval set** in `.github/autofix/evals/`: commits that introduce known
   breakages, each with an expected verdict. Cases cover a real regression, an
   intentionally outdated test, a pre-existing failure, a flaky test, and a trap
   where weakening a test is the easy fix.
3. **Model comparison.** `workflow_dispatch` with `eval_mode: true` runs the
   replay set against `AUTOFIX_MODEL` or a supplied model, never pushes, and
   reports verdict accuracy and fix pass rate. Run it before changing
   `AUTOFIX_MODEL`.

## Rollout

1. Dry run: the agent comments its diff and never pushes.
2. Label-gated pushes on a handful of PRs.
3. Phase 2 (Lepton/GPU QA, fixing from logs only) and phase 3 (e2e suites).
