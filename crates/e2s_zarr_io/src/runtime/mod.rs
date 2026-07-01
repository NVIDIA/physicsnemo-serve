/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime components for write-path orchestration.
//!
//! This module contains the execution-time machinery:
//!
//! - [`array_registry`] — Registration contract manager for `add_array()` / `write()` names.
//! - [`buffer_pool`] — Shared reusable buffer pool with hot/warm slab lifecycle.
//! - `coordinator` (internal) — Write orchestration (plan → reserve → copy → write).
//! - `copy_engine` (internal) — Host memcpy and CUDA copy path selection.
//! - [`planner`] — Mixed-radix chunk planner with axis resolver and template caches.
//! - [`registry`] — Thread-safe `ChunkId` reservation and commit registry.
//! - [`thread_pool`] — Work scheduler and Rayon work-stealing runtime.

pub(crate) mod array_registry;
pub(crate) mod buffer_pool;
#[cfg(feature = "test-utils")]
pub mod coordinator;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod coordinator;
#[cfg(feature = "test-utils")]
pub mod copy_engine;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod copy_engine;
pub mod cuda_runtime;
pub(crate) mod inflight_guard;
pub(crate) mod planner;
pub(crate) mod registry;
pub(crate) mod thread_pool;
pub(crate) mod write_task;

#[cfg(test)]
mod tests {
    use std::process::Command;

    #[test]
    fn internal_runtime_modules_use_restricted_visibility() {
        let source = include_str!("mod.rs");
        let production = source.split("\n#[cfg(test)]").next().unwrap_or(source);
        for module in [
            "array_registry",
            "buffer_pool",
            "planner",
            "registry",
            "thread_pool",
        ] {
            let non_test_pub = format!("pub mod {module};");
            assert!(
                !production.contains(&non_test_pub),
                "runtime::{module} should be pub(crate), not pub — \
                 external access should go through lib.rs re-exports",
            );
        }
    }

    #[test]
    fn extracted_runtime_modules_are_tracked_in_git_index() {
        let output = Command::new("git")
            .args([
                "ls-files",
                "--error-unmatch",
                "src/runtime/coordinator_submit.rs",
                "src/runtime/inflight_guard.rs",
                "src/runtime/write_task.rs",
            ])
            .output()
            .expect("git ls-files should be executable in repository test runs");
        assert!(
            output.status.success(),
            "all extracted runtime modules must be tracked in git; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
