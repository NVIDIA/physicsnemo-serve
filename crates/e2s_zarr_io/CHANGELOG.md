# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- No unreleased changes recorded yet.

## [0.1.0] - 2026-02-16

### Added

- Rust-native synchronous Zarr backend lifecycle (`add_array`, `write`, `close`).
- Python bindings for `E2sZarrIoBackend` with context manager support.
- Zarr v2 and v3 write-path support with validated target configuration.
- Deterministic parity harness and CI parity gate coverage.
- Crate-level and Python package metadata and user-facing README docs.

### Changed

- `close(timeout_seconds=None)` now uses configured `close_lease_timeout_seconds`.
- Python binding long-running backend calls run detached from the GIL after input parsing.
- Public docs and rustdoc coverage tightened with crate-wide missing-doc warnings.

### Fixed

- Safe Rust API no longer exposes external construction of unchecked host-pointer input sources.
- Close-path teardown is now best-effort across failure paths with bounded wait behavior.
- Deferred write failures now surface consistently on close with improved diagnostics.
- Warm slab background warmup thread lifecycle is joined during shutdown.
- Panic recovery in async worker paths now releases uncommitted reserved chunk IDs.

### Known Limitations (v0.1.0)

- Native Rust read runtime is deferred; read support is currently parity-oriented via Python path.
- Compression codecs are deferred in v1 (raw chunk writes only).
- Only `InputStabilityPolicy::StrictGilHold` is supported by runtime config validation in v1.

[unreleased]: https://github.com/NVIDIA/earth2studio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NVIDIA/earth2studio/releases/tag/v0.1.0
