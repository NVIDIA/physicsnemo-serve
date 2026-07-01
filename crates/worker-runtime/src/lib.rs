/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub mod config;
pub mod engine;
pub mod health;
pub mod metrics;
pub mod retry_dlq;
pub mod roles;
pub mod traits;
pub mod transport;

/// Process-wide test utilities for env-var synchronisation.
///
/// All tests that mutate environment variables **must** acquire this lock
/// to prevent data races across parallel `#[test]` threads.
///
/// Uses `tokio::sync::Mutex` so the guard can be held across `.await`
/// points in async tests. Sync callers use `blocking_lock()`.
#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::OnceLock;
    use tokio::sync::Mutex;

    /// Returns a process-global mutex that guards env-var mutations.
    ///
    /// - **Async tests**: `let _guard = env_lock().lock().await;`
    /// - **Sync helpers**: `let _guard = env_lock().blocking_lock();`
    pub fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Run `f` while holding the process-wide env lock, with `key`
    /// temporarily set to `value` (or removed if `None`).
    /// Restores the previous value on return.
    pub fn with_env_var<R>(key: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = env_lock().blocking_lock();
        with_env_var_unguarded(key, value, f)
    }

    /// Same as [`with_env_var`] but **does not** acquire the lock.
    /// Use only when the caller already holds the lock (nested env-var setup).
    pub fn with_env_var_unguarded<R>(key: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let previous = std::env::var(key).ok();
        match value {
            // SAFETY: all callers synchronize via `env_lock()`.
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        let out = f();
        match previous {
            // SAFETY: restoring previous value under the same lock.
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        out
    }

    /// Set or remove an env var. Caller **must** hold [`env_lock()`].
    pub fn set_env_var(key: &str, value: Option<&str>) {
        match value {
            // SAFETY: caller holds env_lock().
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }
}
