# Documentation Index

Use the files in this directory as the current, user-facing reference set for
the plugin-based PhysicsNeMo Serve workflow.

## Start Here

- `onboarding.md`
  - shortest entry point for service consumers and plugin authors
- `inference-service-user-guide.md`
  - REST flow, worker stages, and result retrieval
- `plugin-authoring-guide.md`
  - plugin layout, SDK hooks, runtime hints, and local authoring flow
- `earth2-deterministic-batching.md`
  - Earth2-specific batching notes for `plugins/earth2-deterministic-batch`

## Repo Touch Points

- `plugins/`
  - manifest-driven plugins and examples
- `scripts/`
  - local authoring tools, runtime expansion, and worker entrypoints
- `python/e2s_tools/`
  - shared Python helpers
- `crates/`
  - Rust services and runtime stages
- `tests/`
  - repo-level regression and contract tests

This directory should track shipped behavior. Long-lived design review notes
belong elsewhere once the implementation lands.
