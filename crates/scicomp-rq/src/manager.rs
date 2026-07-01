/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! QueueManager struct definition, constructors, and accessors.

use redis::aio::ConnectionManager;
use std::sync::Arc;
use std::time::Duration;

use crate::builder;
use crate::constants::{defaults, env};
use crate::error::{QueueError, Result};

fn resolve_redis_url(
    var_result: std::result::Result<String, std::env::VarError>,
) -> Result<String> {
    match var_result {
        Ok(url) => Ok(url),
        Err(std::env::VarError::NotPresent) => Ok(defaults::REDIS_URL.to_string()),
        Err(e) => Err(QueueError::Config(format!(
            "invalid {}: {e}",
            env::REDIS_URL
        ))),
    }
}

/// Connection and reconnect tuning for Redis `ConnectionManager`.
///
/// Values are optional; unset fields use `redis` crate defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectionManagerConfig {
    /// Timeout for establishing TCP/TLS connection to Redis.
    pub connection_timeout: Option<Duration>,
    /// Timeout for waiting on Redis command responses.
    pub response_timeout: Option<Duration>,
    /// Exponential reconnect base.
    pub retry_exponent_base: Option<u64>,
    /// Exponential reconnect factor (milliseconds multiplier).
    pub retry_factor: Option<u64>,
    /// Number of reconnect retries before giving up.
    pub retry_count: Option<usize>,
    /// Maximum reconnect delay in milliseconds.
    pub max_retry_delay_ms: Option<u64>,
}

impl ConnectionManagerConfig {
    fn to_redis_config(&self) -> redis::aio::ConnectionManagerConfig {
        let mut config = redis::aio::ConnectionManagerConfig::new();

        if let Some(connection_timeout) = self.connection_timeout {
            config = config.set_connection_timeout(connection_timeout);
        }
        if let Some(response_timeout) = self.response_timeout {
            config = config.set_response_timeout(response_timeout);
        }
        if let Some(retry_exponent_base) = self.retry_exponent_base {
            config = config.set_exponent_base(retry_exponent_base);
        }
        if let Some(retry_factor) = self.retry_factor {
            config = config.set_factor(retry_factor);
        }
        if let Some(retry_count) = self.retry_count {
            config = config.set_number_of_retries(retry_count);
        }
        if let Some(max_retry_delay_ms) = self.max_retry_delay_ms {
            config = config.set_max_delay(max_retry_delay_ms);
        }

        config
    }
}

/// Queue manager for Redis Streams operations.
///
/// Provides methods for enqueueing messages, reading from streams,
/// and performing atomic handoffs between processing stages.
///
/// # Cloning
///
/// `QueueManager` is cheap to clone. The internal `ConnectionManager` uses
/// `Arc` internally, so cloning only increments a reference count.
/// SHA locks are shared across clones, so Lua script cache state is shared too.
/// Multiple tasks can share the same `QueueManager` safely.
///
/// # Encapsulation Contract
///
/// External crates must use accessors (`connection()`) instead of
/// reading internal fields directly.
///
/// ```compile_fail
/// fn external_code(qm: &scicomp_rq::QueueManager) {
///     let _ = &qm.conn;
/// }
/// ```
///
/// ```compile_fail
/// use scicomp_rq::{LogicalStreamName, QueueManager};
///
/// fn external_code(qm: &QueueManager) {
///     let _ = qm.stream_key(LogicalStreamName::new("prefetch"));
/// }
/// ```
///
/// ```compile_fail
/// use scicomp_rq::{LogicalStreamName, QueueManager};
///
/// fn external_code(qm: &QueueManager) {
///     let _ = qm.group_name(LogicalStreamName::new("prefetch"));
/// }
/// ```
///
/// ```compile_fail
/// use scicomp_rq::StreamsConfig;
///
/// fn external_code(_cfg: StreamsConfig) {}
/// ```
///
/// ```compile_fail
/// # async fn external_code(qm: &scicomp_rq::QueueManager) {
/// let _ = qm.hset("run:1", "stage", "prefetch").await;
/// # }
/// ```
///
#[derive(Clone)]
pub struct QueueManager {
    /// Redis connection manager (crate-visible; use accessors from other crates).
    pub(crate) conn: ConnectionManager,
    /// Cached SHA for the Lua handoff script
    pub(crate) lua_handoff_sha: Arc<tokio::sync::RwLock<Option<String>>>,
    /// Cached SHA for the Lua forward_many script
    pub(crate) lua_forward_many_sha: Arc<tokio::sync::RwLock<Option<String>>>,
}

impl std::fmt::Debug for QueueManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueManager").finish_non_exhaustive()
    }
}

impl QueueManager {
    /// Create a new builder for constructing a `QueueManager`.
    ///
    /// This is the recommended way to create a `QueueManager` with the fluent API.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use scicomp_rq::QueueManager;
    ///
    /// # async fn example() -> scicomp_rq::Result<()> {
    /// let qm = QueueManager::builder()
    ///     .redis_url("redis://localhost:6379")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> builder::QueueManagerBuilder {
        builder::QueueManagerBuilder::new()
    }

    /// Create a new QueueManager from a Redis URL.
    pub async fn new(redis_url: &str) -> Result<Self> {
        Self::new_with_connection_config(redis_url, &ConnectionManagerConfig::default()).await
    }

    /// Create a new QueueManager from a Redis URL with connection tuning.
    pub async fn new_with_connection_config(
        redis_url: &str,
        config: &ConnectionManagerConfig,
    ) -> Result<Self> {
        let client = redis::Client::open(redis_url)?;
        let conn = client
            .get_connection_manager_with_config(config.to_redis_config())
            .await?;
        Ok(Self {
            conn,
            lua_handoff_sha: Arc::new(tokio::sync::RwLock::new(None)),
            lua_forward_many_sha: Arc::new(tokio::sync::RwLock::new(None)),
        })
    }

    /// Create a QueueManager with Redis-only configuration.
    ///
    /// This constructor keeps `scicomp-rq` focused on queue primitives; stream
    /// topology and logical-to-physical mapping can be owned by callers.
    pub async fn from_redis_url(redis_url: &str) -> Result<Self> {
        Self::new(redis_url).await
    }

    /// Load from environment variables.
    ///
    /// Environment variables:
    /// - `REDIS_URL` (default: redis://127.0.0.1:6379)
    ///
    /// Stream naming is resolved by callers (for example in `worker-runtime`).
    pub async fn from_env() -> Result<Self> {
        let url = resolve_redis_url(std::env::var(env::REDIS_URL))?;
        Self::from_redis_url(&url).await
    }

    /// Get a clone of the connection manager.
    ///
    /// This is the recommended way to get a connection for custom Redis operations.
    ///
    /// # Why Clone?
    ///
    /// `ConnectionManager` is designed to be cloned for each async operation.
    /// This is **intentional and cheap** - it uses `Arc` internally, so cloning
    /// only increments a reference count (no actual connection duplication).
    ///
    /// The clone pattern is required for async safety: each async task needs
    /// its own handle to avoid borrowing issues across `.await` points.
    #[inline]
    pub fn connection(&self) -> ConnectionManager {
        self.conn.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectionManagerConfig, resolve_redis_url};
    use crate::constants::{defaults, env};
    use crate::error::QueueError;
    use std::ffi::OsString;
    use std::time::Duration;

    #[test]
    fn resolve_redis_url_returns_env_value_when_present() {
        let value = "redis://example:6379".to_string();
        let actual = resolve_redis_url(Ok(value.clone())).expect("env value should be used");
        assert_eq!(actual, value);
    }

    #[test]
    fn resolve_redis_url_uses_default_when_not_present() {
        let actual = resolve_redis_url(Err(std::env::VarError::NotPresent))
            .expect("missing env should use default");
        assert_eq!(actual, defaults::REDIS_URL);
    }

    #[test]
    fn resolve_redis_url_returns_error_when_not_unicode() {
        let result = resolve_redis_url(Err(std::env::VarError::NotUnicode(OsString::from(
            "not-unicode",
        ))));
        assert!(result.is_err(), "non-utf8 REDIS_URL must be rejected");
        let err = result.expect_err("non-utf8 REDIS_URL must fail");
        assert!(
            matches!(err, QueueError::Config(_)),
            "expected QueueError::Config for non-utf8 env value"
        );
        let message = err.to_string();
        assert!(
            message.contains(env::REDIS_URL),
            "error should include REDIS_URL variable name"
        );
    }

    #[test]
    fn connection_manager_config_maps_selected_fields_to_redis_config() {
        let cfg = ConnectionManagerConfig {
            connection_timeout: Some(Duration::from_millis(500)),
            response_timeout: Some(Duration::from_secs(2)),
            retry_exponent_base: Some(3),
            retry_factor: Some(250),
            retry_count: Some(9),
            max_retry_delay_ms: Some(4_000),
        };

        let redis_cfg = cfg.to_redis_config();
        let dbg = format!("{redis_cfg:?}");
        assert!(dbg.contains("connection_timeout: Some("));
        assert!(dbg.contains("response_timeout: Some("));
        assert!(dbg.contains("exponent_base: 3"));
        assert!(dbg.contains("factor: 250"));
        assert!(dbg.contains("number_of_retries: 9"));
        assert!(dbg.contains("max_delay: Some(4000)"));
    }

    #[test]
    fn connection_manager_config_default_leaves_optional_fields_unset() {
        let cfg = ConnectionManagerConfig::default();
        let redis_cfg = cfg.to_redis_config();
        let dbg = format!("{redis_cfg:?}");
        assert!(dbg.contains("connection_timeout: None"));
        assert!(dbg.contains("response_timeout: None"));
        assert!(dbg.contains("max_delay: None"));
    }

    #[test]
    fn queue_manager_clone_docs_explain_shared_sha_locks() {
        let source = include_str!("manager.rs");
        let docs_section = source
            .split("#[cfg(test)]")
            .next()
            .expect("manager.rs should contain non-test source");
        assert!(
            docs_section.contains("SHA locks are shared across clones"),
            "QueueManager clone docs should explicitly describe shared Lua SHA caches"
        );
    }
}
