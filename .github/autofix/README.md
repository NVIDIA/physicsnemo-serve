# CI autofix agent

When CI fails on a PR, this agent works out which failures the PR caused, fixes
them, and (in push mode) commits the fix to the PR branch so CI runs again.
Design: [docs/plans/2026-10-21-ci-autofix-agent-design.md](../../docs/plans/2026-10-21-ci-autofix-agent-design.md).

## Using it on a PR

- Add the `ci-autofix` label. The agent runs after each failed `CI` run.
- Or comment `/autofix` (requires write access) to run it on the latest failed run.

The agent posts one comment on the PR and updates it on each run. It handles
the jobs listed in [`jobs.yaml`](jobs.yaml), skips PRs from forks, and stops
after fixes from two runs have been pushed to a PR.

## Pipeline

| Stage | Model? | What happens |
|---|---|---|
| Gate | no | Label/permission checks, fork check, two-run cap, stale-run check |
| Collect | no | Download failed job logs, extract test IDs |
| Triage | no | Rerun on PR head (pass → flaky), rerun on merge-base (fail → pre-existing); `cargo fmt` / `clippy --fix` for lint jobs |
| Agent | yes | mini-swe-agent in the `ci-autofix-env` container, no credentials; writes `verdict.json` |
| Validate | no | Diff rules in `ci_autofix/diff_rules.py`, then rerun fixed tests and job suites |
| Commit / push | no | Source fix and test updates as separate `ci-fix(agent):` commits; push with a GitHub App token |

## Repository setup

| Setting | Purpose |
|---|---|
| `vars.AUTOFIX_MODE` | `dry-run` (default: comment the proposed diff) or `push` |
| `vars.AUTOFIX_MODEL` | LiteLLM model name, e.g. `openai/<model>` for an OpenAI-compatible NVIDIA endpoint |
| `vars.AUTOFIX_API_BASE` | Endpoint base URL |
| `secrets.AUTOFIX_API_KEY` | Endpoint key |
| `vars.AUTOFIX_MODEL_CLASS` | Optional. `litellm_textbased` for models without tool calling |
| `vars.AUTOFIX_APP_ID`, `secrets.AUTOFIX_APP_PRIVATE_KEY` | GitHub App with `contents: write` (push mode only) |
| Labels | `ci-autofix`, `ci-autofix:test-changed` |

Runners need Docker. Changing the model is a variable change; no code change.

## Changing the agent

`workflow_run` and `issue_comment` always use the default branch's copy of the
workflow and this directory, so a PR cannot change the agent's behaviour. To
try changes before merging:

```bash
gh workflow run ci-autofix.yml --ref <your-branch> -f pr_number=<n> -f ci_run_id=<failed-run-id>
```

Run the unit tests with `uv run pytest tests/autofix`.

## Known limits

- Rust tests inside `#[cfg(test)]` modules of source files count as source,
  so a `test-outdated` fix there is rejected; use `needs-human` for those.
- C++ `ctest` failures in the native SDK job are not parsed and are reported
  as needing a human.
