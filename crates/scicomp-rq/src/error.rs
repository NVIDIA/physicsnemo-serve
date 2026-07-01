/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for scicomp-rq.
//!
//! This module provides strongly-typed errors for better error handling
//! compared to opaque `anyhow::Error`.
//!
//! # Error Variants
//!
//! Some variants like [`QueueError::StreamNotFound`] and [`QueueError::Timeout`]
//! are provided for API completeness and custom validation. They are not currently
//! returned by the core library methods but can be used by applications that
//! implement their own stream validation or timeout handling.

use std::fmt;

/// Errors that can occur when using scicomp-rq.
#[derive(Debug)]
#[non_exhaustive]
pub enum QueueError {
    /// Redis connection or command failed.
    Redis(redis::RedisError),

    /// Configuration is invalid.
    Config(String),

    /// JSON serialization/deserialization failed.
    Json(serde_json::Error),

    /// I/O error (e.g., reading config file).
    Io(std::io::Error),

    /// Stream not found in configuration.
    ///
    /// This variant is provided for custom validation. Use caller-side stream
    /// topology validation before operations if strict validation is needed.
    StreamNotFound { stream: String },

    /// Lua script execution failed.
    Script(String),

    /// Connection timeout.
    ///
    /// This variant is provided for custom timeout handling. The core library
    /// relies on Redis's built-in timeout mechanisms.
    Timeout { operation: &'static str },

    /// One or more consumer groups failed to create during provisioning.
    GroupCreation {
        /// Per-stream errors (stream key, error message) for non-BUSYGROUP failures.
        failures: Vec<(String, String)>,
    },
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Redis(e) => write!(f, "Redis error: {e}"),
            Self::Config(msg) => write!(f, "Configuration error: {msg}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::StreamNotFound { stream } => {
                write!(f, "Stream '{stream}' not found in configuration")
            }
            Self::Script(msg) => write!(f, "Lua script error: {msg}"),
            Self::Timeout { operation } => write!(f, "Timeout during {operation}"),
            Self::GroupCreation { failures } => {
                write!(f, "Failed to create {} consumer group(s): ", failures.len())?;
                for (i, (stream, err)) in failures.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{stream}: {err}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for QueueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Redis(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<redis::RedisError> for QueueError {
    fn from(err: redis::RedisError) -> Self {
        Self::Redis(err)
    }
}

impl From<serde_json::Error> for QueueError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<std::io::Error> for QueueError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Result type alias for scicomp-rq operations.
pub type Result<T> = std::result::Result<T, QueueError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_group_creation_display_lists_failures() {
        let err = QueueError::GroupCreation {
            failures: vec![
                ("stream:a".into(), "connection refused".into()),
                ("stream:b".into(), "timeout".into()),
            ],
        };
        let message = err.to_string();
        assert!(message.contains("2 consumer group(s)"));
        assert!(message.contains("stream:a"));
        assert!(message.contains("stream:b"));
    }

    #[test]
    fn test_stream_not_found_display_mentions_stream_name() {
        let err = QueueError::StreamNotFound {
            stream: "stream:missing".into(),
        };
        let message = err.to_string();
        assert!(
            message.contains("stream:missing"),
            "StreamNotFound display should include missing stream key"
        );
    }

    #[test]
    fn test_script_and_timeout_display_messages() {
        let script_err = QueueError::Script("xack failed".into());
        assert!(
            script_err.to_string().contains("Lua script error"),
            "Script display should include Lua context"
        );

        let timeout_err = QueueError::Timeout {
            operation: "XREADGROUP",
        };
        let timeout_msg = timeout_err.to_string();
        assert!(
            timeout_msg.contains("Timeout during XREADGROUP"),
            "Timeout display should include operation name"
        );
    }

    #[test]
    fn test_from_redis_error_maps_to_variant() {
        let redis_err = redis::RedisError::from((redis::ErrorKind::IoError, "boom"));
        let err: QueueError = redis_err.into();
        assert!(matches!(err, QueueError::Redis(_)));
    }

    #[test]
    fn test_from_json_error_maps_to_variant() {
        let json_err = serde_json::from_str::<serde_json::Value>("{").expect_err("must fail");
        let err: QueueError = json_err.into();
        assert!(matches!(err, QueueError::Json(_)));
    }

    #[test]
    fn test_from_io_error_maps_to_variant() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: QueueError = io_err.into();
        assert!(matches!(err, QueueError::Io(_)));
    }

    #[test]
    fn test_source_available_for_wrapped_errors() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err = QueueError::Io(io_err);
        assert!(
            err.source().is_some(),
            "Io variant should keep source chain"
        );
    }

    #[test]
    fn test_source_available_for_redis_and_json_variants() {
        let redis_err =
            QueueError::Redis(redis::RedisError::from((redis::ErrorKind::IoError, "boom")));
        assert!(
            redis_err.source().is_some(),
            "Redis variant should expose source error"
        );

        let json_parse_err = serde_json::from_str::<serde_json::Value>("{")
            .expect_err("invalid JSON should produce serde_json::Error");
        let json_err = QueueError::Json(json_parse_err);
        assert!(
            json_err.source().is_some(),
            "Json variant should expose source error"
        );
    }

    #[test]
    fn test_source_none_for_string_backed_errors() {
        let err = QueueError::Config("bad config".into());
        assert!(
            err.source().is_none(),
            "Config variant should not expose source"
        );
    }

    #[test]
    fn test_source_none_for_timeout_variant() {
        let err = QueueError::Timeout {
            operation: "XREADGROUP",
        };
        assert!(
            err.source().is_none(),
            "Timeout variant should not expose a nested source"
        );
    }

    #[test]
    fn test_result_alias_supports_ok_and_err() {
        let ok: Result<i32> = Ok(1);
        let err: Result<i32> = Err(QueueError::Config("x".into()));
        assert!(matches!(ok, Ok(1)));
        assert!(matches!(err, Err(QueueError::Config(_))));
    }
}
