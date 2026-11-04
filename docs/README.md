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
- [Native worker and serving integration](native-worker-serving-integration.md)
  - C++ linkage, runtime configuration, readiness and Redis recovery contracts
- [Native workflow authoring and sealed releases](native-workflow-user-guide.md)
  - native project contracts, verified model handoff, release images and QA
- [CFD applications and matched benchmarking](cfd-native-workflow-and-benchmark.md)
  - imported Transolver builds, Python/native parity and comparable HTTP timing

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

- [Unified application CLI](../application-cli/README.md): Python and native local development, verified builds and serving.
