# Changelog

All notable changes to `scicomp-rq` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reproducible python-feature validation gate script and CI workflow.
- Strict all-target clippy verification in default quality gates.
- Additional contract tests for malformed XAUTOCLAIM parsing and handoff logical-name mapping.

### Changed
- Health liveness semantics now treat Redis connectivity as healthy even before lazy Lua script loading.
- Handoff builder now enforces a logical-name contract for `from(...)`/`to(...)`.
- Python bindings now explicitly require Python 3.11+ (matches `abi3-py311` build target).
- `QueueError` is now `#[non_exhaustive]` to preserve semver flexibility for future variants.

## [0.1.0] - 2026-02-20

### Added
- Initial `scicomp-rq` release with Redis Streams queue operations.
- Atomic handoff and fan-out primitives via Lua scripts.
- Rust and Python API surfaces for enqueue/read/ack/recovery workflows.
