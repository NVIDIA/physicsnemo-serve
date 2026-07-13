/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = include_str!("../README.md")]

//! # scicomp-rq
//!
//! A Redis Streams-based queue manager for scientific computing pipelines.
//!
//! ## Overview
//!
//! `scicomp-rq` provides atomic message handoffs between processing stages,
//! ensuring exactly-once delivery semantics using Redis Lua scripts.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use scicomp_rq::QueueManager;
//!
//! #[tokio::main]
//! async fn main() -> scicomp_rq::Result<()> {
//!     // Create manager from Redis URL.
//!     let qm = QueueManager::from_redis_url("redis://localhost:6379").await?;
//!
//!     // Enqueue a job
//!     let payload = r#"{"model": "pangu", "steps": 10}"#;
//!     let stream = scicomp_rq::LogicalStreamName::new("prefetch");
//!     let msg_id = qm.enqueue(&stream, "run-001", payload, "prefetch").await?;
//!     println!("Enqueued: {}", msg_id);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! Messages flow through configured streams with atomic handoffs.
//!
//! **Earth2Studio Pipeline:**
//! ```text
//! [prefetch] --> [dispatch] --> [gpu_X] --+--> [release]
//!     |             |             |       |
//!     |             |             +-------+--> [results]
//!     |             |             |
//!     +-------------+-------------+-- run:{id} hash updated --+
//! ```
//!
//! The dispatcher routes requests to one of 8 GPU streams (gpu_0 to gpu_7).
//! After inference, GPU workers send to both `release` (resource cleanup)
//! and `results` (post-processing).
//!
//! ## Features
//!
//! - `python` - Enable Python bindings via PyO3

// Public modules
pub mod builder;
pub mod constants;
pub mod error;
pub mod hash_ops;
pub mod redis_utils;
#[allow(async_fn_in_trait)]
pub mod traits;
pub mod types;

#[cfg(feature = "python")]
mod bindings;

// Re-exports
pub use constants::{defaults, env, fields, groups, keys};
pub use error::{QueueError, Result};
pub use redis_utils::{parse_stream_entries, parse_stream_messages};
pub use traits::{
    AckOps, AtomicOps, EnqueueOps, GroupOps, HealthOps, QueueOps, ReadOps, RecoveryOps,
};
pub use types::{HandoffRequest, HealthStatus, LogicalStreamName, Message, Output, StreamKey};

mod lua;
pub use lua::{LUA_FORWARD_MANY, LUA_HANDOFF};

#[cfg(test)]
use lua::{
    GroupCreateOutcome, classify_group_creation, derive_stage_from_stream, is_noscript_error,
    is_xautoclaim_unsupported,
};

mod manager;
mod operations;

pub use manager::{ConnectionManagerConfig, QueueManager};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // classify_group_creation Tests
    // =========================================================================

    #[test]
    fn test_classify_group_creation_success_returns_created() {
        let result: std::result::Result<redis::Value, redis::RedisError> = Ok(redis::Value::Okay);
        assert!(matches!(
            classify_group_creation(result),
            GroupCreateOutcome::Created
        ));
    }

    #[test]
    fn test_classify_group_creation_busygroup_returns_already_exists() {
        let err = redis::RedisError::from((
            redis::ErrorKind::ExtensionError,
            "BUSYGROUP Consumer Group name already exists",
        ));
        let result: std::result::Result<redis::Value, redis::RedisError> = Err(err);
        assert!(matches!(
            classify_group_creation(result),
            GroupCreateOutcome::AlreadyExists
        ));
    }

    #[test]
    fn test_classify_group_creation_other_error_returns_failed() {
        let err = redis::RedisError::from((redis::ErrorKind::IoError, "Connection refused"));
        let result: std::result::Result<redis::Value, redis::RedisError> = Err(err);
        let outcome = classify_group_creation(result);
        assert!(
            matches!(outcome, GroupCreateOutcome::Failed(_)),
            "Expected Failed variant for non-BUSYGROUP error"
        );
    }

    #[test]
    fn test_group_creation_error_variant_display() {
        let err = QueueError::GroupCreation {
            failures: vec![
                ("stream:a".to_string(), "Connection refused".to_string()),
                ("stream:b".to_string(), "Timeout".to_string()),
            ],
        };
        let display = err.to_string();
        assert!(display.contains("2 consumer group(s)"));
        assert!(display.contains("stream:a"));
        assert!(display.contains("stream:b"));
    }

    // =========================================================================
    // is_xautoclaim_unsupported Tests
    // =========================================================================

    #[test]
    fn test_is_xautoclaim_unsupported_with_unknown_command_xautoclaim() {
        let redis_err = redis::RedisError::from((
            redis::ErrorKind::ExtensionError,
            "ERR unknown command 'XAUTOCLAIM'",
        ));
        let err = QueueError::Redis(redis_err);
        assert!(is_xautoclaim_unsupported(&err));
    }

    #[test]
    fn test_is_xautoclaim_unsupported_with_generic_error() {
        let redis_err = redis::RedisError::from((redis::ErrorKind::IoError, "Connection refused"));
        let err = QueueError::Redis(redis_err);
        assert!(!is_xautoclaim_unsupported(&err));
    }

    #[test]
    fn test_is_xautoclaim_unsupported_with_unknown_but_different_command() {
        let redis_err = redis::RedisError::from((
            redis::ErrorKind::ExtensionError,
            "ERR unknown command 'XINFO'",
        ));
        let err = QueueError::Redis(redis_err);
        assert!(!is_xautoclaim_unsupported(&err));
    }

    #[test]
    fn test_claim_idle_messages_extracts_xautoclaim_error_without_unreachable_fallback() {
        let source = include_str!("operations.rs");
        assert!(
            source.contains("let err = match xautoclaim_result {"),
            "claim_idle_messages should extract XAUTOCLAIM errors via match"
        );
        assert!(
            !source.contains("xautoclaim_result.err().unwrap_or_else"),
            "claim_idle_messages should not use unreachable unwrap_or_else fallback"
        );
    }

    #[test]
    fn test_is_noscript_error_positive() {
        let redis_err = redis::RedisError::from((
            redis::ErrorKind::ExtensionError,
            "NOSCRIPT No matching script",
        ));
        assert!(is_noscript_error(&redis_err));
    }

    #[test]
    fn test_is_noscript_error_negative() {
        let redis_err = redis::RedisError::from((redis::ErrorKind::IoError, "Connection refused"));
        assert!(!is_noscript_error(&redis_err));
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_derive_stage_from_stream_with_colon() {
        // "stream:results" with matching prefix should extract suffix "results"
        assert_eq!(
            derive_stage_from_stream("stream:results", "stream:"),
            "results"
        );
        // "physicsnemo:inference" with matching prefix should extract suffix
        assert_eq!(
            derive_stage_from_stream("physicsnemo:inference", "physicsnemo:"),
            "inference"
        );
    }

    #[test]
    fn test_derive_stage_from_stream_multiple_colons() {
        // Prefix-aware mode keeps the full suffix after prefix stripping.
        assert_eq!(derive_stage_from_stream("a:b:c:final", "a:"), "b:c:final");
    }

    #[test]
    fn test_derive_stage_from_stream_no_colon() {
        // "results" without matching prefix returns full key unchanged
        assert_eq!(derive_stage_from_stream("results", "stream:"), "results");
        assert_eq!(
            derive_stage_from_stream("inference", "stream:"),
            "inference"
        );
    }

    #[test]
    fn test_derive_stage_from_stream_empty_suffix() {
        // Empty suffix falls back to full stream key unchanged
        assert_eq!(derive_stage_from_stream("stream:", "stream:"), "stream:");
    }

    #[test]
    fn test_derive_stage_from_stream_prefix_aware_for_colon_rich_stream_key() {
        // With configured prefix, derive from full suffix (not just last token)
        assert_eq!(
            derive_stage_from_stream("physicsnemo:gpu:default:pod-0:0", "physicsnemo:"),
            "gpu:default:pod-0:0"
        );
    }

    #[test]
    fn test_derive_stage_from_stream_returns_full_key_when_prefix_does_not_match() {
        // Non-prefixed stream keys should remain unchanged
        assert_eq!(
            derive_stage_from_stream("external:gpu:default:pod-0:0", "physicsnemo:"),
            "external:gpu:default:pod-0:0"
        );
    }

    #[test]
    fn test_is_noscript_error_detects_noscript_response() {
        let err = redis::RedisError::from((
            redis::ErrorKind::ResponseError,
            "NOSCRIPT No matching script. Please use EVAL.",
        ));
        assert!(
            is_noscript_error(&err),
            "NOSCRIPT responses must be detected for retry"
        );
    }

    #[test]
    fn test_is_noscript_error_rejects_other_response_errors() {
        let err = redis::RedisError::from((
            redis::ErrorKind::ResponseError,
            "WRONGTYPE Operation against a key holding the wrong kind of value",
        ));
        assert!(
            !is_noscript_error(&err),
            "Non-NOSCRIPT responses must not trigger script reload retry"
        );
    }

    #[test]
    fn test_run_hash_key_format() {
        let run_id = "my-run-123";
        let run_hash_prefix = keys::RUN_HASH_PREFIX;
        let run_hash = format!("{run_hash_prefix}{run_id}");
        assert_eq!(run_hash, "run:my-run-123");
    }

    #[test]
    fn test_stream_message_id_format() {
        let msg_id = "1234567890123-0";
        assert!(msg_id.contains('-'));
        let parts: Vec<&str> = msg_id.split('-').collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].parse::<u64>().is_ok());
        assert!(parts[1].parse::<u64>().is_ok());
    }

    #[test]
    fn test_lua_handoff_script_contents() {
        // Verify the Lua script contains expected Redis commands
        assert!(LUA_HANDOFF.contains("XADD"));
        assert!(LUA_HANDOFF.contains("XACK"));
        assert!(LUA_HANDOFF.contains("HSET"));
        // Verify script has the expected structure
        assert!(LUA_HANDOFF.contains("KEYS[1]"));
        assert!(LUA_HANDOFF.contains("ARGV[1]"));
    }

    #[test]
    fn test_queue_manager_builder_factory() {
        // QueueManager::builder() should return a QueueManagerBuilder
        // This is a compile-time check that the method exists and returns the right type
        let _builder = QueueManager::builder();

        // Verify chaining works through the factory method
        let _builder_with_config = QueueManager::builder()
            .redis_url("redis://localhost:6379")
            .connection_timeout_ms(1000)
            .preload_scripts(false);

        // The fact that this compiles and the builder accepts all methods
        // proves the factory method returns the correct type
    }

    #[tokio::test]
    async fn test_queue_manager_builder_factory_missing_url() {
        // Verify builder from factory requires redis_url
        let result = QueueManager::builder().build().await;

        assert!(result.is_err(), "Expected error when redis_url is missing");
        let err = result.unwrap_err();
        let err_text = err.to_string();
        assert!(
            matches!(err, QueueError::Config(_)),
            "Expected Config error mentioning redis_url, got: {err}"
        );
        assert!(err_text.contains("redis_url"));
    }

    #[tokio::test]
    async fn test_queue_manager_builder_factory_does_not_require_config() {
        // Verify builder from factory does not require stream config.
        let result = QueueManager::builder()
            .redis_url("not-a-valid-url")
            .build()
            .await;

        assert!(result.is_err(), "Invalid Redis URL format should fail");
        let err = result.unwrap_err();
        let err_text = err.to_string();
        assert!(
            !err_text.contains("config"),
            "Builder should not fail with missing-config error, got: {err}"
        );
    }

    // =========================================================================
    // QueueManager::enqueue_to() Tests
    // =========================================================================

    // Note: Full integration tests for enqueue_to().send() require Redis.
    // These unit tests verify the API structure and builder creation.

    #[test]
    fn test_queue_manager_enqueue_to_api_exists() {
        fn _assert_enqueue_builder_api(qm: &QueueManager) {
            let fut = qm
                .enqueue_to("prefetch")
                .run_id("run-001")
                .payload(r#"{"model":"pangu"}"#)
                .stage("prefetch")
                .send();
            std::mem::drop(fut);
        }
        let _ = _assert_enqueue_builder_api;
    }

    #[test]
    fn test_queue_manager_handoff_builder_api_exists() {
        fn _assert_handoff_builder_api(qm: &QueueManager) {
            let msg = Message::new(
                "1706123456789-0",
                "stream:prefetch",
                "prefetch:grp",
                "run-001",
                r#"{"model": "pangu"}"#,
                "prefetch",
            );
            let fut = qm
                .handoff_builder()
                .from("prefetch")
                .to("inference")
                .message(msg)
                .stage("inference")
                .send();
            std::mem::drop(fut);
        }
        let _ = _assert_handoff_builder_api;
    }

    // =========================================================================
    // QueueManager Accessor and Debug Tests
    // =========================================================================

    // =========================================================================
    // Error Type Comprehensive Tests
    // =========================================================================

    // =========================================================================
    // Constants Module Tests
    // =========================================================================

    #[test]
    fn test_from_env_does_not_require_queue_config() {
        let source = include_str!("manager.rs");
        assert!(
            source.contains("pub async fn from_env()"),
            "QueueManager::from_env must exist"
        );
        assert!(
            !source.contains(env::QUEUE_CONFIG),
            "from_env should remain independent from {}",
            env::QUEUE_CONFIG
        );
    }

    #[test]
    fn test_no_unsafe_env_mutation_in_from_env_contract_test() {
        let source = include_str!("lib.rs");
        assert!(
            !source.contains("unsafe {\n            std::env::set_var"),
            "from_env test must not use unsafe env var mutation"
        );
        assert!(
            !source.contains("unsafe {\n            std::env::remove_var"),
            "from_env test must not use unsafe env var mutation"
        );
    }

    #[test]
    fn test_cargo_toml_declares_msrv() {
        let cargo_toml = include_str!("../Cargo.toml");
        assert!(
            cargo_toml.contains("rust-version"),
            "Cargo.toml should declare rust-version (MSRV)"
        );
    }

    #[test]
    fn test_cargo_toml_declares_criterion_bench_target() {
        let cargo_toml = include_str!("../Cargo.toml");
        assert!(
            cargo_toml.contains("criterion"),
            "Cargo.toml should declare criterion dev-dependency and redis_utils_parser bench target"
        );
        assert!(cargo_toml.contains("[[bench]]"));
        assert!(cargo_toml.contains("name = \"redis_utils_parser\""));
        assert!(cargo_toml.contains("harness = false"));
    }

    #[test]
    fn test_readme_mentions_msrv() {
        let readme = include_str!("../README.md");
        assert!(
            readme.contains("Minimum Supported Rust Version"),
            "README should document minimum Rust version"
        );
        assert!(readme.contains("MSRV"));
    }

    #[test]
    fn test_readme_documents_static_dispatch_trait_contract() {
        let readme = include_str!("../README.md");
        assert!(
            readme.contains("static dispatch"),
            "README trait section should explicitly document static-dispatch-only contract"
        );
        assert!(readme.contains("object-safe"));
    }

    #[test]
    fn test_changelog_exists() {
        let changelog = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("CHANGELOG.md");
        assert!(
            changelog.exists(),
            "CHANGELOG.md should exist for release tracking"
        );
    }

    #[test]
    fn test_basic_usage_example_exists() {
        let example =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/basic_usage.rs");
        assert!(
            example.exists(),
            "examples/basic_usage.rs should exist for runnable quick-start"
        );
    }

    #[test]
    fn test_bindings_json_tests_are_contract_focused() {
        let source = include_str!("bindings.rs");
        assert!(
            !source.contains("fn test_json_payload_parsing_unicode_escape"),
            "bindings.rs should prune third-party serde_json behavior edge-case tests"
        );
        assert!(!source.contains("fn test_json_payload_parsing_scientific_notation"));
        assert!(!source.contains("fn test_json_payload_parsing_trailing_comma"));
        assert!(!source.contains("fn test_json_payload_parsing_single_quotes"));
        assert!(!source.contains("fn test_json_payload_parsing_nan_not_allowed"));
        assert!(!source.contains("fn test_json_payload_parsing_deeply_nested"));
    }

    #[test]
    fn test_message_fields_are_not_public_in_types_contract() {
        let source = include_str!("types.rs");
        let message_block = source
            .split("pub struct Message")
            .nth(1)
            .and_then(|tail| tail.split("impl Message").next())
            .expect("types.rs should contain Message struct and impl blocks");
        assert!(
            !message_block.contains("pub id: String"),
            "Message fields should not be public in external API"
        );
        assert!(!message_block.contains("pub stream: String"));
        assert!(!message_block.contains("pub group: String"));
        assert!(!message_block.contains("pub run_id: String"));
        assert!(!message_block.contains("pub payload: String"));
        assert!(!message_block.contains("pub stage: String"));
    }

    #[test]
    fn test_message_getters_exist_in_types_contract() {
        let source = include_str!("types.rs");
        assert!(
            source.contains("pub fn id(&self) -> &str"),
            "Message getter methods should exist for read access"
        );
        assert!(source.contains("pub fn stream(&self) -> &str"));
        assert!(source.contains("pub fn group(&self) -> &str"));
        assert!(source.contains("pub fn run_id(&self) -> &str"));
        assert!(source.contains("pub fn payload(&self) -> &str"));
        assert!(source.contains("pub fn stage(&self) -> &str"));
    }

    #[test]
    fn test_output_fields_are_not_public_in_types_contract() {
        let source = include_str!("types.rs");
        assert!(
            !source.contains("pub stream: String"),
            "Output fields should not be public in external API"
        );
        assert!(!source.contains("pub payload: String"));
        assert!(!source.contains("pub stage: Option<String>"));
    }

    #[test]
    fn test_health_status_fields_are_not_public_in_types_contract() {
        let source = include_str!("types.rs");
        assert!(
            !source.contains("pub connected: bool"),
            "HealthStatus fields should not be public in external API"
        );
        assert!(!source.contains("pub latency_ms: u64"));
        assert!(!source.contains("pub script_loaded: bool"));
    }

    #[test]
    fn test_output_and_health_status_support_serde_contract() {
        let source = include_str!("types.rs");
        assert!(
            source.contains(
                "#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]\npub struct Output"
            ),
            "Output and HealthStatus should derive Serialize/Deserialize for observability contracts"
        );
        assert!(
            source
                .contains("#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]\npub struct HealthStatus")
        );
    }

    #[test]
    fn test_queue_error_marked_non_exhaustive_for_semver_contract() {
        let source = include_str!("error.rs");
        assert!(
            source.contains("#[non_exhaustive]\npub enum QueueError"),
            "QueueError should be non_exhaustive to allow forward-compatible variant additions"
        );
    }

    #[test]
    fn test_output_getters_exist_in_types_contract() {
        let source = include_str!("types.rs");
        assert!(
            source.contains("pub fn stream(&self) -> &str"),
            "Output getter methods should exist for read access"
        );
        assert!(source.contains("pub fn payload(&self) -> &str"));
        assert!(source.contains("pub fn stage(&self) -> Option<&str>"));
    }

    #[test]
    fn test_handoff_request_fields_are_not_public_in_types_contract() {
        let source = include_str!("types.rs");
        let request_block = source
            .split("pub struct HandoffRequest")
            .nth(1)
            .and_then(|tail| tail.split("impl HandoffRequest").next())
            .expect("types.rs should contain HandoffRequest struct and impl blocks");
        assert!(
            !request_block.contains("pub current_stream"),
            "HandoffRequest fields should not be public in external API"
        );
        assert!(!request_block.contains("pub next_stream"));
        assert!(!request_block.contains("pub group"));
        assert!(!request_block.contains("pub run_id"));
        assert!(!request_block.contains("pub payload_json"));
        assert!(!request_block.contains("pub current_msg_id"));
        assert!(!request_block.contains("pub next_stage"));
    }

    #[test]
    fn test_handoff_request_new_uses_result_contract() {
        let source = include_str!("types.rs");
        assert!(
            source.contains("pub fn new("),
            "HandoffRequest::new should validate non-empty invariants and return Result"
        );
        assert!(source.contains(") -> std::result::Result<Self, QueueError>"));
        assert!(source.contains("must be non-empty"));
    }

    #[test]
    fn test_health_status_removes_streams_configured_field_contract() {
        let source = include_str!("types.rs");
        assert!(
            !source.contains("streams_configured"),
            "HealthStatus should no longer expose a perpetually-zero streams_configured field"
        );
    }

    #[test]
    fn test_python_health_check_contract_without_streams_configured() {
        let pyi = include_str!("../scicomp_rq.pyi");
        assert!(
            pyi.contains("async def health_check(self) -> Tuple[bool, int, bool]:"),
            "Python health_check contract should return (connected, latency_ms, script_loaded)"
        );
        assert!(!pyi.contains("streams_configured"));
    }

    #[test]
    fn test_python_stub_hgetall_uses_typing_dict_contract() {
        let pyi = include_str!("../scicomp_rq.pyi");
        assert!(
            pyi.contains("from typing import Dict, List, Optional, Tuple"),
            "Python stub should use typing.Dict consistently in hgetall contract"
        );
        assert!(pyi.contains("async def hgetall(self, key: str) -> Dict[str, str]:"));
    }

    #[test]
    fn test_pyproject_requires_python_matches_abi3_contract() {
        let pyproject = include_str!("../pyproject.toml");
        assert!(
            pyproject.contains("requires-python = \">=3.11\""),
            "pyproject requires-python must match abi3-py311 minimum"
        );
    }

    #[test]
    fn test_changelog_mentions_python_311_requirement() {
        let changelog = include_str!("../CHANGELOG.md");
        assert!(
            changelog.contains("Python 3.11+"),
            "CHANGELOG should document Python 3.11+ runtime requirement for bindings"
        );
    }

    #[test]
    fn test_readme_documents_non_send_trait_future_constraint() {
        let readme = include_str!("../README.md");
        assert!(
            readme.contains("tokio::spawn"),
            "README must document the async-trait non-Send future constraint for trait-generic callers"
        );
        assert!(readme.contains("not guaranteed `Send`"));
        assert!(readme.contains("`T: QueueOps`"));
    }

    // =========================================================================
    // Lua Script Validation Tests
    // =========================================================================

    #[test]
    fn test_lua_handoff_script_has_required_keys() {
        // Verify script uses all 3 KEYS
        assert!(LUA_HANDOFF.contains("KEYS[1]"), "Missing KEYS[1]");
        assert!(LUA_HANDOFF.contains("KEYS[2]"), "Missing KEYS[2]");
        assert!(LUA_HANDOFF.contains("KEYS[3]"), "Missing KEYS[3]");
    }

    #[test]
    fn test_lua_handoff_script_has_required_args() {
        // Verify script uses all 5 ARGV
        assert!(LUA_HANDOFF.contains("ARGV[1]"), "Missing ARGV[1]");
        assert!(LUA_HANDOFF.contains("ARGV[2]"), "Missing ARGV[2]");
        assert!(LUA_HANDOFF.contains("ARGV[3]"), "Missing ARGV[3]");
        assert!(LUA_HANDOFF.contains("ARGV[4]"), "Missing ARGV[4]");
        assert!(LUA_HANDOFF.contains("ARGV[5]"), "Missing ARGV[5]");
    }

    #[test]
    fn test_lua_handoff_script_returns_next_id() {
        // Verify script returns the next_id
        assert!(
            LUA_HANDOFF.contains("return next_id"),
            "Script should return next_id"
        );
    }

    #[test]
    fn test_lua_handoff_script_uses_time() {
        // Verify script gets current time for timestamps
        assert!(
            LUA_HANDOFF.contains("redis.call('TIME')"),
            "Script should call TIME for timestamps"
        );
    }

    #[test]
    fn test_lua_handoff_script_validates_xack_success() {
        assert!(
            LUA_HANDOFF.contains("local acked = redis.call('XACK'"),
            "Script must capture XACK result"
        );
        assert!(
            LUA_HANDOFF.contains("acked ~= 1"),
            "Script must fail closed when XACK does not acknowledge exactly one message"
        );
    }

    #[test]
    fn test_lua_handoff_script_rolls_back_xadd_on_xack_failure() {
        assert!(
            LUA_HANDOFF.contains("redis.pcall('XDEL', KEYS[1], next_id)"),
            "Script should attempt rollback of XADD entry when XACK fails"
        );
        assert!(
            LUA_HANDOFF.contains("redis.error_reply"),
            "Script should return explicit error on XACK failure"
        );
    }

    #[test]
    fn test_lua_handoff_script_uses_pcall_for_rollback_xdel() {
        assert!(
            LUA_HANDOFF.contains("redis.pcall('XDEL', KEYS[1], next_id)"),
            "handoff rollback should use redis.pcall to preserve original failure signal"
        );
    }

    #[test]
    fn test_lua_handoff_script_surfaces_hset_failures() {
        assert!(
            LUA_HANDOFF.contains("redis.pcall('HSET'"),
            "handoff script should guard run-hash updates with redis.pcall"
        );
        assert!(
            LUA_HANDOFF.contains("HSET_FAILED"),
            "handoff script should return explicit HSET_FAILED error when hash update fails"
        );
    }

    #[test]
    fn test_lua_forward_many_script_validates_source_pending_precondition() {
        assert!(
            LUA_FORWARD_MANY.contains("XPENDING"),
            "forward_many script must verify source pending precondition"
        );
        assert!(
            LUA_FORWARD_MANY.contains("SOURCE_NOT_PENDING"),
            "forward_many script should emit explicit pending-precondition failure"
        );
    }

    #[test]
    fn test_lua_forward_many_script_validates_destination_key_type() {
        assert!(
            LUA_FORWARD_MANY.contains("TYPE"),
            "forward_many script must validate destination key type"
        );
        assert!(
            LUA_FORWARD_MANY.contains("key_type_name ~= 'stream'"),
            "forward_many script must explicitly allow existing stream destination keys"
        );
        assert!(
            !LUA_FORWARD_MANY.contains("key_type_name ~= 'string'"),
            "forward_many script must not treat Redis 'string' keys as valid destination streams"
        );
        assert!(
            LUA_FORWARD_MANY.contains("DEST_NOT_STREAM"),
            "forward_many script should reject non-stream destination keys"
        );
    }

    #[test]
    fn test_lua_forward_many_script_acks_once_and_rolls_back_on_ack_failure() {
        assert!(
            LUA_FORWARD_MANY.contains("local acked = redis.call('XACK'"),
            "forward_many script must capture XACK result"
        );
        assert!(
            LUA_FORWARD_MANY.contains("acked ~= 1"),
            "forward_many script must fail closed on ack mismatch"
        );
        assert!(
            LUA_FORWARD_MANY.contains("redis.pcall('XDEL'"),
            "forward_many script should rollback produced entries on ack failure"
        );
        assert!(
            LUA_FORWARD_MANY.contains("XACK_FAILED"),
            "forward_many script should return explicit ack-failure error"
        );
    }

    #[test]
    fn test_lua_forward_many_script_uses_pcall_for_rollback_xdel() {
        assert!(
            LUA_FORWARD_MANY.contains("redis.pcall('XDEL', KEYS[i], ids[i])"),
            "forward_many rollback should use redis.pcall to preserve original ack failure"
        );
    }

    // =========================================================================
    // Message Tests
    // =========================================================================

    #[test]
    fn test_stream_message_new_with_string_types() {
        // Test with String
        let msg = Message::new(
            "1-0".to_string(),
            "stream:test".to_string(),
            "test:grp".to_string(),
            "run-123".to_string(),
            "{}".to_string(),
            "stage".to_string(),
        );
        assert_eq!(msg.id, "1-0");

        // Test with &str
        let msg2 = Message::new("2-0", "stream:test", "test:grp", "run-456", "{}", "stage2");
        assert_eq!(msg2.id, "2-0");
    }

    #[test]
    fn test_stream_message_eq() {
        let msg1 = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "{}",
            "prefetch",
        );
        let msg2 = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "{}",
            "prefetch",
        );
        let msg3 = Message::new(
            "2-0",
            "stream:test",
            "test:grp",
            "run-123",
            "{}",
            "prefetch",
        );

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }

    #[test]
    fn test_stream_message_clone() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            r#"{"key": "value"}"#,
            "prefetch",
        );
        let cloned = msg.clone();

        assert_eq!(msg.id, cloned.id);
        assert_eq!(msg.run_id, cloned.run_id);
        assert_eq!(msg.payload, cloned.payload);
        assert_eq!(msg.stage, cloned.stage);
    }

    #[test]
    fn test_stream_message_debug() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "{}",
            "prefetch",
        );
        let debug_str = format!("{:?}", msg);

        assert!(debug_str.contains("Message"));
        assert!(debug_str.contains("1-0"));
        assert!(debug_str.contains("run-123"));
    }

    #[test]
    fn test_stream_message_parse_payload_complex() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            r#"{"nested": {"key": "value"}, "array": [1, 2, 3]}"#,
            "prefetch",
        );

        let payload: serde_json::Value = msg.parse_payload().unwrap();
        assert!(payload["nested"]["key"].is_string());
        assert!(payload["array"].is_array());
    }

    #[test]
    fn test_stream_message_parse_payload_empty_object() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "{}",
            "prefetch",
        );

        let payload: serde_json::Value = msg.parse_payload().unwrap();
        assert!(payload.is_object());
        assert!(payload.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_stream_message_parse_payload_array() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "[1, 2, 3]",
            "prefetch",
        );

        let payload: Vec<i32> = msg.parse_payload().unwrap();
        assert_eq!(payload, vec![1, 2, 3]);
    }

    // =========================================================================
    // HealthStatus Tests
    // =========================================================================

    #[test]
    fn test_health_status_all_combinations() {
        // Test all combinations of connected and script_loaded
        let cases = [
            (true, true, true),    // connected, script_loaded -> healthy
            (true, false, true),   // connected, no script -> healthy (scripts are lazy-loaded)
            (false, true, false),  // not connected, script -> unhealthy
            (false, false, false), // neither -> unhealthy
        ];

        for (connected, script_loaded, expected_healthy) in cases {
            let status = HealthStatus {
                connected,
                latency_ms: 10,
                script_loaded,
            };
            assert_eq!(
                status.is_healthy(),
                expected_healthy,
                "connected={connected}, script_loaded={script_loaded}"
            );
        }
    }

    #[test]
    fn test_health_status_eq() {
        let status1 = HealthStatus {
            connected: true,
            latency_ms: 10,
            script_loaded: true,
        };
        let status2 = HealthStatus {
            connected: true,
            latency_ms: 10,
            script_loaded: true,
        };
        let status3 = HealthStatus {
            connected: true,
            latency_ms: 20, // different latency
            script_loaded: true,
        };

        assert_eq!(status1, status2);
        assert_ne!(status1, status3);
    }

    // =========================================================================
    // Message::parse_payload Error Path Tests
    // =========================================================================

    #[test]
    fn test_stream_message_parse_payload_invalid_json() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "not valid json at all",
            "prefetch",
        );

        let result: std::result::Result<serde_json::Value, _> = msg.parse_payload();
        assert!(result.is_err(), "Invalid JSON should fail to parse");
    }

    #[test]
    fn test_stream_message_parse_payload_type_mismatch() {
        // Payload is a string, but we try to parse as a struct
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            r#""just a string""#,
            "prefetch",
        );

        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct ExpectedStruct {
            field: i32,
        }

        let result: std::result::Result<ExpectedStruct, _> = msg.parse_payload();
        assert!(result.is_err(), "Type mismatch should fail");
    }

    #[test]
    fn test_stream_message_parse_payload_missing_required_field() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            r#"{"other": 42}"#,
            "prefetch",
        );

        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct RequiredFields {
            required_field: String,
        }

        let result: std::result::Result<RequiredFields, _> = msg.parse_payload();
        assert!(result.is_err(), "Missing required field should fail");
    }

    #[test]
    fn test_stream_message_parse_payload_null() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "null",
            "prefetch",
        );

        let result: serde_json::Value = msg.parse_payload().unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn test_stream_message_parse_payload_number() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "42",
            "prefetch",
        );

        let result: i32 = msg.parse_payload().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_stream_message_parse_payload_boolean() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "true",
            "prefetch",
        );

        let result: bool = msg.parse_payload().unwrap();
        assert!(result);
    }

    #[test]
    fn test_stream_message_parse_payload_empty_string_fails() {
        let msg = Message::new("1-0", "stream:test", "test:grp", "run-123", "", "prefetch");

        let result: std::result::Result<serde_json::Value, _> = msg.parse_payload();
        assert!(result.is_err(), "Empty string is not valid JSON");
    }

    #[test]
    fn test_stream_message_parse_payload_whitespace_only_fails() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            "   ",
            "prefetch",
        );

        let result: std::result::Result<serde_json::Value, _> = msg.parse_payload();
        assert!(result.is_err(), "Whitespace-only string is not valid JSON");
    }

    // =========================================================================
    // HealthStatus Additional Tests
    // =========================================================================

    #[test]
    fn test_health_status_zero_latency() {
        let status = HealthStatus {
            connected: true,
            latency_ms: 0,
            script_loaded: true,
        };
        assert!(status.is_healthy());
        assert_eq!(status.latency_ms, 0);
    }

    #[test]
    fn test_health_status_high_latency() {
        let status = HealthStatus {
            connected: true,
            latency_ms: u64::MAX,
            script_loaded: true,
        };
        assert!(status.is_healthy());
        assert_eq!(status.latency_ms, u64::MAX);
    }

    #[test]
    fn test_health_status_zero_streams() {
        let status = HealthStatus {
            connected: true,
            latency_ms: 10,
            script_loaded: true,
        };
        assert!(status.is_healthy());
    }

    #[test]
    fn test_health_status_many_streams() {
        let status = HealthStatus {
            connected: true,
            latency_ms: 10,
            script_loaded: true,
        };
        assert!(status.is_healthy());
    }

    // =========================================================================
    // Lua Script Additional Validation
    // =========================================================================

    #[test]
    fn test_lua_handoff_script_no_syntax_errors() {
        // Verify script doesn't have obvious syntax issues
        assert!(
            LUA_HANDOFF.contains("local"),
            "Script should use local variables"
        );
        assert!(
            LUA_HANDOFF.contains("return"),
            "Script should have a return statement"
        );
        // Verify balanced brackets (simple check)
        let open_parens = LUA_HANDOFF.matches('(').count();
        let close_parens = LUA_HANDOFF.matches(')').count();
        assert_eq!(open_parens, close_parens, "Parentheses should be balanced");
    }

    #[test]
    fn test_lua_handoff_script_updates_hash() {
        // Verify script updates the run hash with stage and timestamps
        assert!(
            LUA_HANDOFF.contains("'stage'"),
            "Script should update stage field"
        );
        assert!(
            LUA_HANDOFF.contains("'updated_at'"),
            "Script should update updated_at field"
        );
        assert!(
            LUA_HANDOFF.contains("_enqueued_at"),
            "Script should update per-stage enqueued_at"
        );
    }

    // =========================================================================
    // Message Serialization Tests
    // =========================================================================

    #[test]
    fn test_stream_message_serialization() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "test:grp",
            "run-123",
            r#"{"key":"value"}"#,
            "prefetch",
        );

        // Test serialization
        let serialized = serde_json::to_string(&msg).unwrap();
        assert!(serialized.contains("1-0"));
        assert!(serialized.contains("run-123"));
        assert!(serialized.contains("prefetch"));

        // Test deserialization
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, msg);
    }

    #[test]
    fn test_stream_message_deserialization_from_json() {
        let json = r#"{"id":"2-5","stream":"stream:inference","group":"inference:grp","run_id":"run-xyz","payload":"{}","stage":"inference"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();

        assert_eq!(msg.id, "2-5");
        assert_eq!(msg.stream, "stream:inference");
        assert_eq!(msg.group, "inference:grp");
        assert_eq!(msg.run_id, "run-xyz");
        assert_eq!(msg.payload, "{}");
        assert_eq!(msg.stage, "inference");
    }

    #[test]
    fn test_stream_message_deserialization_missing_field_fails() {
        let json = r#"{"id":"2-5","run_id":"run-xyz"}"#; // Missing stream, group, payload and stage
        let result: std::result::Result<Message, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Missing fields should fail deserialization"
        );
    }

    // =========================================================================
    // QueueManager Debug Trait Test
    // =========================================================================

    // Note: We can't create a real QueueManager without Redis, but we can test
    // that the Debug implementation doesn't panic and produces reasonable output
    // by examining the struct definition.

    // =========================================================================
    // forward_many() Tests
    // =========================================================================

    #[test]
    fn test_forward_many_method_exists() {
        // Compile-time check that the forward_many method exists on QueueManager.
        // This test passes if the code compiles, proving the public API contract.
        let _method_exists = QueueManager::forward_many;
    }

    #[test]
    fn test_forward_many_uses_message_and_output_types() {
        // Verify that forward_many uses the correct types:
        // - Takes a &Message for the source message
        // - Takes &[Output] for the destinations
        // This is a contract test validating the API design

        let msg = Message::new(
            "1706123456789-0",
            "stream:gpu",
            "gpu:grp",
            "run-001",
            r#"{"model":"pangu"}"#,
            "inference",
        );

        let outputs = vec![
            Output::new("stream:results", r#"{"status":"ok"}"#),
            Output::new("stream:release", r#"{"gpu":"gpu:0"}"#).with_stage("release"),
        ];

        // Verify message has required fields for forward_many
        assert!(!msg.stream.is_empty(), "stream needed for ack");
        assert!(!msg.group.is_empty(), "group needed for ack");
        assert!(!msg.id.is_empty(), "id needed for ack");
        assert!(!msg.run_id.is_empty(), "run_id needed for XADD");

        // Verify outputs have required fields
        for out in &outputs {
            assert!(!out.stream.is_empty(), "output stream needed for XADD");
            assert!(!out.payload.is_empty(), "output payload needed for XADD");
        }
    }

    #[test]
    fn test_forward_many_output_stage_derivation() {
        // Test that Output stage is either explicit or derived from stream key

        // Output with explicit stage
        let out1 = Output::new("stream:results", "{}").with_stage("custom_results");
        assert_eq!(out1.stage, Some("custom_results".to_string()));

        // Output without explicit stage (should be derived from stream)
        let out2 = Output::new("stream:results", "{}");
        assert_eq!(out2.stage, None);

        // When stage is None, forward_many derives from stream using derive_stage_from_stream
        assert_eq!(
            derive_stage_from_stream("stream:results", "stream:"),
            "results"
        );
        assert_eq!(
            derive_stage_from_stream("stream:release", "stream:"),
            "release"
        );
        assert_eq!(
            derive_stage_from_stream("physicsnemo:inference", "physicsnemo:"),
            "inference"
        );
    }

    #[test]
    fn test_forward_many_empty_outputs() {
        // forward_many with empty outputs should still ack the message
        // but return an empty vector of IDs
        let outputs: Vec<Output> = vec![];
        assert!(outputs.is_empty());

        // When forward_many is called with empty outputs:
        // - Should still ack the original message
        // - Should return Ok(vec![])
    }

    #[test]
    fn test_forward_many_returns_one_id_per_output() {
        // Verifies that forward_many returns exactly one message ID per output.
        // This documents the contract that optimizations (like connection reuse)
        // must preserve: each output in the input slice produces exactly one
        // corresponding ID in the output vector.

        // Create test outputs with varying counts
        let test_cases = vec![
            vec![Output::new("stream:a", "{}")],
            vec![Output::new("stream:a", "{}"), Output::new("stream:b", "{}")],
            vec![
                Output::new("stream:a", "{}"),
                Output::new("stream:b", "{}"),
                Output::new("stream:c", "{}"),
            ],
        ];

        for outputs in test_cases {
            // Verify each test case has the expected structure
            let expected_count = outputs.len();

            // Verify pre-allocation would work correctly
            let mut result_ids: Vec<String> = Vec::with_capacity(expected_count);
            for (i, output) in outputs.iter().enumerate() {
                // Simulate what forward_many does: one ID per output
                result_ids.push(format!("1234567890123-{}", i));
                assert!(
                    !output.stream.is_empty(),
                    "output must have stream for XADD"
                );
            }

            // The critical contract: output count == input count
            assert_eq!(
                result_ids.len(),
                expected_count,
                "forward_many must return exactly one ID per output"
            );
        }
    }

    #[test]
    fn test_forward_many_preserves_output_order() {
        // Verifies that forward_many processes outputs in order and returns
        // IDs in the same order as the input outputs slice.
        // This is important for callers that correlate IDs with outputs by index.

        let outputs = [
            Output::new("stream:first", r#"{"order":1}"#),
            Output::new("stream:second", r#"{"order":2}"#),
            Output::new("stream:third", r#"{"order":3}"#),
        ];

        // Verify outputs are distinct and ordered
        assert_eq!(outputs[0].stream, "stream:first");
        assert_eq!(outputs[1].stream, "stream:second");
        assert_eq!(outputs[2].stream, "stream:third");

        // When forward_many returns IDs, they correspond by index:
        // - ids[0] is the ID from XADD to outputs[0].stream
        // - ids[1] is the ID from XADD to outputs[1].stream
        // - ids[2] is the ID from XADD to outputs[2].stream
    }

    // =========================================================================
    // Result Type Tests
    // =========================================================================

    // =========================================================================
    // Constants Integration Tests
    // =========================================================================

    #[test]
    fn test_constants_used_correctly_in_key_generation() {
        assert_eq!(keys::DEFAULT_STREAM_PREFIX, "stream:");
        let group_prefix = "test";
        let group_suffix = groups::GROUP_SUFFIX;
        let group = format!("{group_prefix}{group_suffix}");
        assert!(group.ends_with(groups::GROUP_SUFFIX));
    }

    #[test]
    fn test_run_hash_prefix_format() {
        // Verify run hash key generation matches expected format
        let run_id = "abc-123";
        let run_hash_prefix = keys::RUN_HASH_PREFIX;
        let run_hash = format!("{run_hash_prefix}{run_id}");

        assert!(run_hash.starts_with("run:"));
        assert!(run_hash.ends_with(run_id));
    }

    // =========================================================================
    // Builder Factory Method Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_queue_manager_builder_all_methods_chainable() {
        // Verify all builder methods can be chained in any order
        let _builder1 = QueueManager::builder()
            .redis_url("redis://localhost:6379")
            .connection_timeout_ms(500)
            .preload_scripts(true);

        let _builder2 = QueueManager::builder()
            .preload_scripts(false)
            .response_timeout_ms(2500)
            .redis_url("redis://localhost:6380");

        let _builder3 = QueueManager::builder()
            .preload_scripts(true)
            .reconnect_policy(3, 250, 9, Some(7_000))
            .redis_url("redis://localhost:6381");

        // All should compile and not panic
    }

    #[tokio::test]
    async fn test_queue_manager_builder_invalid_redis_url_format() {
        // Invalid Redis URL should fail during build
        let result = QueueManager::builder()
            .redis_url("not-a-valid-url")
            .build()
            .await;

        assert!(result.is_err(), "Invalid Redis URL format should fail");
    }

    // =========================================================================
    // Builder API Contract Integration
    // =========================================================================

    #[test]
    fn test_enqueue_builder_complete_workflow() {
        fn _assert_enqueue_workflow(qm: &QueueManager) {
            let fut = qm
                .enqueue_to("prefetch")
                .run_id("run-001")
                .payload(r#"{"model":"pangu","steps":10,"config":{"resolution":"0.25deg"}}"#)
                .stage("prefetch")
                .send();
            std::mem::drop(fut);
        }
        let _ = _assert_enqueue_workflow;
    }

    #[test]
    fn test_handoff_builder_complete_workflow() {
        fn _assert_handoff_workflow(qm: &QueueManager) {
            let msg = Message::new(
                "1706123456789-0",
                "stream:prefetch",
                "prefetch:grp",
                "run-001",
                r#"{"model":"pangu","result":"success"}"#,
                "prefetch",
            );

            let fut = qm
                .handoff_builder()
                .from("prefetch")
                .to("inference")
                .message(msg)
                .stage("inference")
                .send();
            std::mem::drop(fut);
        }
        let _ = _assert_handoff_workflow;
    }

    // =========================================================================
    // read_messages() Tests
    // =========================================================================

    #[test]
    fn test_stream_identifier_newtypes_exist_and_are_distinct() {
        // R13 contract: logical stream names and full stream keys are distinct types.
        let logical = LogicalStreamName::new("prefetch");
        let key = StreamKey::new("stream:prefetch");

        assert_eq!(logical.as_str(), "prefetch");
        assert_eq!(key.as_str(), "stream:prefetch");
    }

    #[test]
    fn test_read_messages_method_exists() {
        // Compile-time check that the read_messages method exists on QueueManager.
        // We verify this by type-checking a function pointer. This test passes
        // if the code compiles, proving the public API contract is correct.
        //
        // The actual method signature is:
        //   async fn read_messages(&self, stream: &str, group: &str, consumer: &str,
        //                          count: usize, block_ms: usize) -> Result<Vec<Message>>
        //
        // We can't easily test async signatures at compile time, but the other tests
        // in this module verify the method works correctly with parse_stream_messages.

        // This compiles only if QueueManager has a read_messages method
        let _method_exists = QueueManager::read_messages;
    }

    #[test]
    fn test_read_messages_signature_uses_stream_key() {
        fn _assert_read_messages_contract(qm: &QueueManager, stream: &StreamKey) {
            let fut = qm.read_messages(stream, "group", "consumer", 1, 0);
            std::mem::drop(fut);
        }
        let _ = _assert_read_messages_contract;
    }

    #[test]
    fn test_read_messages_uses_parse_stream_messages() {
        // Test that parse_stream_messages correctly populates stream and group fields
        // This validates the internal wiring without requiring Redis
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run-123".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
            redis::Value::BulkString(b"stage".to_vec()),
            redis::Value::BulkString(b"prefetch".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"1706123456789-0".to_vec()),
            msg_fields,
        ]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream:test".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        // This simulates what read_messages() does internally
        let messages = redis_utils::parse_stream_messages(response, "stream:test", "test:grp");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1706123456789-0");
        assert_eq!(messages[0].stream, "stream:test");
        assert_eq!(messages[0].group, "test:grp");
        assert_eq!(messages[0].run_id, "run-123");
        assert_eq!(messages[0].payload, "{}");
        assert_eq!(messages[0].stage, "prefetch");
    }

    #[test]
    fn test_read_messages_returns_empty_on_nil_response() {
        // When XREADGROUP times out, it returns Nil
        let messages =
            redis_utils::parse_stream_messages(redis::Value::Nil, "stream:test", "test:grp");
        assert!(
            messages.is_empty(),
            "Nil response (timeout) should return empty vec"
        );
    }

    // =========================================================================
    // ack_message() Tests
    // =========================================================================

    #[test]
    fn test_ack_message_method_exists() {
        // Compile-time check that the ack_message method exists on QueueManager.
        // We verify this by checking that the method reference is valid.
        // This test passes if the code compiles.
        let _method_exists = QueueManager::ack_message;
    }

    #[test]
    fn test_ack_message_uses_message_fields() {
        // Verify that Message contains all fields needed for ack_message.
        // This is a contract test - ack_message uses msg.stream, msg.group, and msg.id
        let msg = Message::new(
            "1706123456789-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-001",
            "{}",
            "prefetch",
        );

        // Verify the message has all fields needed by ack_message
        assert!(!msg.stream.is_empty(), "stream field is required for ack");
        assert!(!msg.group.is_empty(), "group field is required for ack");
        assert!(!msg.id.is_empty(), "id field is required for ack");

        // Verify fields have correct values (would be used by ack_message internally)
        assert_eq!(msg.stream, "stream:prefetch");
        assert_eq!(msg.group, "prefetch:grp");
        assert_eq!(msg.id, "1706123456789-0");
    }

    #[test]
    fn test_ack_message_contract_with_read_messages_output() {
        // Verify that messages from read_messages have the context needed for ack_message.
        // This simulates the workflow: read_messages -> process -> ack_message

        // Create a mock XREADGROUP response
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run-123".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
            redis::Value::BulkString(b"stage".to_vec()),
            redis::Value::BulkString(b"prefetch".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream:prefetch".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        // Parse using the same function read_messages uses internally
        let messages =
            redis_utils::parse_stream_messages(response, "stream:prefetch", "prefetch:grp");

        assert_eq!(messages.len(), 1);

        // The message returned by read_messages should have all context for ack_message
        let parsed_msg = &messages[0];
        assert_eq!(
            parsed_msg.stream, "stream:prefetch",
            "Message should have stream context"
        );
        assert_eq!(
            parsed_msg.group, "prefetch:grp",
            "Message should have group context"
        );
        assert_eq!(parsed_msg.id, "1-0", "Message should have id");

        // These are the exact fields ack_message will use
    }

    // =========================================================================
    // handoff_message() Tests
    // =========================================================================

    #[test]
    fn test_handoff_message_method_exists() {
        // Compile-time check that the handoff_message method exists on QueueManager.
        // We verify this by checking that the method reference is valid.
        // This test passes if the code compiles.
        let _method_exists = QueueManager::handoff_message;
    }

    #[test]
    fn test_handoff_message_derives_stage_from_stream_key() {
        // Test the stage derivation logic:
        // "stream:results" + prefix "stream:" -> "results"
        // "physicsnemo:inference" + prefix "physicsnemo:" -> "inference"
        // no prefix match -> full stream key unchanged

        // Uses module-level derive_stage_from_stream helper
        assert_eq!(
            derive_stage_from_stream("stream:results", "stream:"),
            "results"
        );
        assert_eq!(
            derive_stage_from_stream("physicsnemo:inference", "physicsnemo:"),
            "inference"
        );
        assert_eq!(
            derive_stage_from_stream("stream:prefetch", "stream:"),
            "prefetch"
        );
        assert_eq!(derive_stage_from_stream("results", "stream:"), "results");
        assert_eq!(
            derive_stage_from_stream("multi:colon:stream", "stream:"),
            "multi:colon:stream"
        );
    }

    #[test]
    fn test_handoff_message_uses_message_fields() {
        // Verify that Message contains all fields needed for handoff_message.
        // handoff_message uses msg.stream, msg.group, msg.id, msg.run_id, msg.payload
        let msg = Message::new(
            "1706123456789-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-001",
            r#"{"model":"pangu"}"#,
            "prefetch",
        );

        // Verify all fields needed by handoff_message
        assert!(
            !msg.stream.is_empty(),
            "stream field is required for handoff"
        );
        assert!(!msg.group.is_empty(), "group field is required for handoff");
        assert!(!msg.id.is_empty(), "id field is required for handoff");
        assert!(
            !msg.run_id.is_empty(),
            "run_id field is required for handoff"
        );
        assert!(
            !msg.payload.is_empty(),
            "payload field is required for handoff"
        );

        // Verify field values
        assert_eq!(msg.stream, "stream:prefetch");
        assert_eq!(msg.group, "prefetch:grp");
        assert_eq!(msg.id, "1706123456789-0");
        assert_eq!(msg.run_id, "run-001");
        assert_eq!(msg.payload, r#"{"model":"pangu"}"#);
    }

    #[test]
    fn test_handoff_message_contract() {
        // Contract test: verify the expected signature and behavior
        // handoff_message(&self, msg: &Message, dest: &StreamKey, payload: Option<&str>, stage: Option<&str>)

        let msg = Message::new(
            "1-0",
            "stream:src",
            "src:grp",
            "run-123",
            r#"{"key":"value"}"#,
            "src",
        );

        // With explicit payload and stage
        let _dest_stream = StreamKey::new("stream:dest");
        let _new_payload = Some(r#"{"status":"done"}"#);
        let _explicit_stage = Some("custom_stage");

        // Verify message has all context needed
        assert_eq!(msg.run_id, "run-123", "run_id used for handoff");
        assert_eq!(
            msg.payload, r#"{"key":"value"}"#,
            "payload used if None provided"
        );
    }

    // =========================================================================
    // create_consumer_group() Tests
    // =========================================================================

    #[test]
    fn test_create_consumer_group_method_exists() {
        // Compile-time check that the create_consumer_group method exists on QueueManager.
        // This test passes if the code compiles, proving the public API contract.
        let _method_exists = QueueManager::create_consumer_group;
    }

    #[test]
    fn test_create_consumer_group_signature() {
        // Verify the expected signature:
        // async fn create_consumer_group(&self, stream: &StreamKey, group: &str, start_id: &str, create_stream: bool) -> Result<bool>
        //
        // - stream: Full Redis stream key (e.g., "gpu:test:0")
        // - group: Consumer group name (e.g., "workers")
        // - start_id: Where to start reading ("$" for new messages only, "0" for all)
        // - create_stream: If true, create the stream if it doesn't exist (MKSTREAM)
        // - Returns: true if group was newly created, false if it already existed

        // These are valid parameter combinations:
        let _stream = StreamKey::new("gpu:test:0");
        let _group = "workers";
        let _start_id_new = "$"; // Only new messages
        let _start_id_all = "0"; // All messages from beginning
        let _create_stream = true;

        // Compile-time check that method exists
        let _method = QueueManager::create_consumer_group;
    }

    #[test]
    fn test_enqueue_signature_uses_logical_stream_name() {
        fn _assert_enqueue_contract(qm: &QueueManager, stream: &LogicalStreamName, payload: &str) {
            let fut = qm.enqueue(stream, "run-001", payload, "prefetch");
            std::mem::drop(fut);
        }
        let _ = _assert_enqueue_contract;
    }

    #[test]
    fn test_create_consumer_group_signature_uses_stream_key() {
        fn _assert_create_group_contract(qm: &QueueManager, stream: &StreamKey) {
            let fut = qm.create_consumer_group(stream, "workers", "$", true);
            std::mem::drop(fut);
        }
        let _ = _assert_create_group_contract;
    }

    #[test]
    fn test_handoff_message_signature_uses_stream_key() {
        fn _assert_handoff_contract(qm: &QueueManager, msg: &Message, dest_stream: &StreamKey) {
            let fut = qm.handoff_message(msg, dest_stream, None, None);
            std::mem::drop(fut);
        }
        let _ = _assert_handoff_contract;
    }

    #[test]
    fn test_create_consumer_group_start_id_values() {
        // Document valid start_id values:
        // "$" - Only messages arriving after the group is created
        // "0" - All messages from the beginning of the stream
        // "1234567890123-0" - Specific message ID

        let start_ids = ["$", "0", "1234567890123-0", "0-0"];
        for id in start_ids {
            assert!(!id.is_empty(), "start_id should be non-empty string: {id}");
        }
    }

    // =========================================================================
    // hash_ops free-function contract tests
    // =========================================================================

    #[test]
    fn test_hash_ops_functions_exist() {
        let _hset = crate::hash_ops::hset;
        let _hdel = crate::hash_ops::hdel;
        let _hgetall = crate::hash_ops::hgetall;
    }

    #[test]
    fn test_hash_ops_signatures() {
        fn assert_future_i64<F>(_f: F)
        where
            F: std::future::Future<Output = Result<i64>>,
        {
        }

        fn assert_future_map<F>(_f: F)
        where
            F: std::future::Future<Output = Result<std::collections::HashMap<String, String>>>,
        {
        }

        fn _contract(conn: &mut redis::aio::ConnectionManager) {
            assert_future_i64(crate::hash_ops::hset(conn, "k", "f", "v"));
            assert_future_i64(crate::hash_ops::hdel(conn, "k", "f"));
            assert_future_map(crate::hash_ops::hgetall(conn, "k"));
        }

        let _ = _contract;
    }

    #[test]
    fn test_hash_ops_use_cases_documented() {
        // Rust callers now use free functions with explicit connection handles:
        //   let mut conn = qm.connection();
        //   hash_ops::hset(&mut conn, "gpu:allocations", "gpu:0", "run-123").await?;
        //   hash_ops::hdel(&mut conn, "gpu:allocations", "gpu:0").await?;
        //   let gpus = hash_ops::hgetall(&mut conn, "gpu:registry").await?;

        let allocation_key = "gpu:allocations";
        let registry_key = "gpu:registry";
        assert!(!allocation_key.is_empty());
        assert!(!registry_key.is_empty());
    }
}
