/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core types for scicomp-rq.
//!
//! This module provides strongly-typed structs for messages and operations,
//! enabling a unified API across Rust and Python workers.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::QueueError;

/// Logical stream identifier (pipeline-level name).
///
/// Examples: `"prefetch"`, `"inference"`, `"results"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogicalStreamName(String);

impl LogicalStreamName {
    /// Create a logical stream name wrapper.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the byte length of the wrapped string.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the wrapped string is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AsRef<str> for LogicalStreamName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for LogicalStreamName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for LogicalStreamName {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for LogicalStreamName {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&String> for LogicalStreamName {
    fn from(value: &String) -> Self {
        Self::new(value.clone())
    }
}

impl From<&LogicalStreamName> for LogicalStreamName {
    fn from(value: &LogicalStreamName) -> Self {
        value.clone()
    }
}

impl PartialEq<&str> for LogicalStreamName {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Full Redis stream key identifier.
///
/// Examples: `"stream:prefetch"`, `"physicsnemo:gpu:0"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamKey(String);

impl StreamKey {
    /// Create a stream key wrapper.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the byte length of the wrapped string.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the wrapped string is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AsRef<str> for StreamKey {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for StreamKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for StreamKey {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for StreamKey {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&String> for StreamKey {
    fn from(value: &String) -> Self {
        Self::new(value.clone())
    }
}

impl From<&StreamKey> for StreamKey {
    fn from(value: &StreamKey) -> Self {
        value.clone()
    }
}

impl PartialEq<&str> for StreamKey {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Typed input for atomic handoff operations.
///
/// This replaces the positional-argument handoff contract to prevent parameter
/// reordering mistakes at call sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffRequest {
    /// Source stream key where the current message was consumed.
    pub(crate) current_stream: StreamKey,
    /// Destination stream key where the next message should be written.
    pub(crate) next_stream: StreamKey,
    /// Source consumer group used for acknowledging the current message.
    pub(crate) group: String,
    /// Workflow run identifier.
    pub(crate) run_id: String,
    /// JSON payload to write into the destination stream entry.
    pub(crate) payload_json: String,
    /// Redis message ID of the current (source) message.
    pub(crate) current_msg_id: String,
    /// Stage value for the destination stream entry.
    pub(crate) next_stage: String,
}

impl HandoffRequest {
    /// Construct a typed handoff request.
    pub fn new(
        current_stream: impl Into<StreamKey>,
        next_stream: impl Into<StreamKey>,
        group: impl Into<String>,
        run_id: impl Into<String>,
        payload_json: impl Into<String>,
        current_msg_id: impl Into<String>,
        next_stage: impl Into<String>,
    ) -> std::result::Result<Self, QueueError> {
        let current_stream = current_stream.into();
        if current_stream.as_str().trim().is_empty() {
            return Err(QueueError::Config(
                "current_stream must be non-empty".into(),
            ));
        }

        let next_stream = next_stream.into();
        if next_stream.as_str().trim().is_empty() {
            return Err(QueueError::Config("next_stream must be non-empty".into()));
        }

        let group = group.into();
        if group.trim().is_empty() {
            return Err(QueueError::Config("group must be non-empty".into()));
        }

        let run_id = run_id.into();
        if run_id.trim().is_empty() {
            return Err(QueueError::Config("run_id must be non-empty".into()));
        }

        let payload_json = payload_json.into();
        if payload_json.trim().is_empty() {
            return Err(QueueError::Config("payload_json must be non-empty".into()));
        }

        let current_msg_id = current_msg_id.into();
        if current_msg_id.trim().is_empty() {
            return Err(QueueError::Config(
                "current_msg_id must be non-empty".into(),
            ));
        }

        let next_stage = next_stage.into();
        if next_stage.trim().is_empty() {
            return Err(QueueError::Config("next_stage must be non-empty".into()));
        }

        Ok(Self {
            current_stream,
            next_stream,
            group,
            run_id,
            payload_json,
            current_msg_id,
            next_stage,
        })
    }

    /// Source stream key where the current message was consumed.
    pub fn current_stream(&self) -> &StreamKey {
        &self.current_stream
    }

    /// Destination stream key where the next message should be written.
    pub fn next_stream(&self) -> &StreamKey {
        &self.next_stream
    }

    /// Source consumer group used for acknowledging the current message.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// Workflow run identifier.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// JSON payload to write into the destination stream entry.
    pub fn payload_json(&self) -> &str {
        &self.payload_json
    }

    /// Redis message ID of the current (source) message.
    pub fn current_msg_id(&self) -> &str {
        &self.current_msg_id
    }

    /// Stage value for the destination stream entry.
    pub fn next_stage(&self) -> &str {
        &self.next_stage
    }
}

/// A message read from a Redis stream.
///
/// Contains all information needed for acknowledgment and handoff operations.
/// The `stream` and `group` fields enable self-contained operations like
/// `ack_message()` and `handoff_message()` without requiring the caller
/// to track this context separately.
///
/// This type is used identically in Rust and Python via PyO3 bindings.
///
/// # Example
///
/// ```
/// use scicomp_rq::Message;
///
/// let msg = Message::new(
///     "1706123456789-0",   // message ID
///     "stream:prefetch",    // stream key
///     "prefetch:grp",       // consumer group
///     "run-001",            // run ID
///     r#"{"model": "pangu"}"#, // payload
///     "prefetch",           // stage
/// );
///
/// assert_eq!(msg.stream(), "stream:prefetch");
/// assert_eq!(msg.group(), "prefetch:grp");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Redis stream message ID (e.g., "1706123456789-0")
    pub(crate) id: String,
    /// Full Redis stream key where message was read from (e.g., "stream:prefetch")
    pub(crate) stream: String,
    /// Consumer group name (e.g., "prefetch:grp")
    pub(crate) group: String,
    /// Unique identifier for this workflow run
    pub(crate) run_id: String,
    /// JSON-encoded payload data
    pub(crate) payload: String,
    /// Current processing stage
    pub(crate) stage: String,
}

impl Message {
    /// Try to create a new Message with validation.
    ///
    /// Critical fields used by Redis operations must be non-empty:
    /// `id`, `stream`, `group`, and `run_id`.
    pub fn try_new(
        id: impl Into<String>,
        stream: impl Into<String>,
        group: impl Into<String>,
        run_id: impl Into<String>,
        payload: impl Into<String>,
        stage: impl Into<String>,
    ) -> std::result::Result<Self, QueueError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(QueueError::Config("id must be non-empty".into()));
        }

        let stream = stream.into();
        if stream.trim().is_empty() {
            return Err(QueueError::Config("stream must be non-empty".into()));
        }

        let group = group.into();
        if group.trim().is_empty() {
            return Err(QueueError::Config("group must be non-empty".into()));
        }

        let run_id = run_id.into();
        if run_id.trim().is_empty() {
            return Err(QueueError::Config("run_id must be non-empty".into()));
        }

        Ok(Self {
            id,
            stream,
            group,
            run_id,
            payload: payload.into(),
            stage: stage.into(),
        })
    }

    /// Create a new Message with all fields.
    ///
    /// # Arguments
    ///
    /// * `id` - Redis stream message ID (e.g., "1706123456789-0")
    /// * `stream` - Full Redis stream key (e.g., "stream:prefetch")
    /// * `group` - Consumer group name (e.g., "prefetch:grp")
    /// * `run_id` - Unique workflow run identifier
    /// * `payload` - JSON-encoded payload data
    /// * `stage` - Current processing stage name
    pub fn new(
        id: impl Into<String>,
        stream: impl Into<String>,
        group: impl Into<String>,
        run_id: impl Into<String>,
        payload: impl Into<String>,
        stage: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            stream: stream.into(),
            group: group.into(),
            run_id: run_id.into(),
            payload: payload.into(),
            stage: stage.into(),
        }
    }

    /// Redis stream message ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Full Redis stream key where message was read from.
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// Consumer group that currently owns the message.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// Workflow run identifier associated with this message.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// JSON payload text.
    pub fn payload(&self) -> &str {
        &self.payload
    }

    /// Current processing stage.
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// Parse the payload as JSON.
    ///
    /// # Example
    ///
    /// ```
    /// use scicomp_rq::Message;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct MyPayload {
    ///     model: String,
    /// }
    ///
    /// let msg = Message::new("1-0", "stream:test", "grp", "run-1", r#"{"model": "pangu"}"#, "stage");
    /// let payload: MyPayload = msg.parse_payload().unwrap();
    /// assert_eq!(payload.model, "pangu");
    /// ```
    pub fn parse_payload<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.payload)
    }
}

/// Output destination for `forward_many()` operation.
///
/// Specifies where to send a message and optionally what stage name to use.
/// Used identically in Rust and Python via PyO3 bindings.
///
/// # Example
///
/// ```
/// use scicomp_rq::Output;
///
/// // Auto-derive stage from stream name
/// let out1 = Output::new("stream:results", r#"{"status": "ok"}"#);
/// assert_eq!(out1.stage(), None);
///
/// // Explicit stage name
/// let out2 = Output::new("stream:results", r#"{"status": "ok"}"#)
///     .with_stage("final_results");
/// assert_eq!(out2.stage(), Some("final_results"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// Destination stream (full Redis key)
    pub(crate) stream: String,
    /// Explicit run_id override for this output (None = preserve source run_id)
    pub(crate) run_id: Option<String>,
    /// JSON payload for this destination
    pub(crate) payload: String,
    /// Stage name (None = auto-derive from stream)
    pub(crate) stage: Option<String>,
}

impl Output {
    /// Create a new Output with auto-derived stage.
    ///
    /// The stage will be derived from the stream name when the message is sent.
    pub fn new(stream: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            stream: stream.into(),
            run_id: None,
            payload: payload.into(),
            stage: None,
        }
    }

    /// Set an explicit run_id for this output.
    ///
    /// Use this when a fanout operation is creating child messages that must
    /// carry a different run identity than the source message.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set an explicit stage name.
    ///
    /// Use this when you want a stage name different from what would be
    /// auto-derived from the stream name.
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Destination stream (full Redis key).
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// Explicit run_id when provided.
    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    /// JSON payload text for this destination.
    pub fn payload(&self) -> &str {
        &self.payload
    }

    /// Explicit stage name when provided.
    pub fn stage(&self) -> Option<&str> {
        self.stage.as_deref()
    }
}

/// Health status of the queue manager.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether Redis is reachable
    pub(crate) connected: bool,
    /// Round-trip latency to Redis in milliseconds
    pub(crate) latency_ms: u64,
    /// Lua script loaded and cached
    pub(crate) script_loaded: bool,
}

impl HealthStatus {
    /// Whether Redis responded successfully to `PING`.
    pub fn connected(&self) -> bool {
        self.connected
    }

    /// Round-trip Redis latency in milliseconds.
    pub fn latency_ms(&self) -> u64 {
        self.latency_ms
    }

    /// Whether required Lua scripts are currently cached.
    pub fn script_loaded(&self) -> bool {
        self.script_loaded
    }

    /// Returns true if Redis connectivity is healthy.
    ///
    /// Script loading is lazy by default, so `script_loaded` is informational and
    /// does not gate liveness.
    pub fn is_healthy(&self) -> bool {
        self.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logical_stream_name_new_and_as_str() {
        let logical = LogicalStreamName::new("prefetch");
        assert_eq!(logical.as_str(), "prefetch");
        assert_eq!(logical.to_string(), "prefetch");
    }

    #[test]
    fn test_logical_stream_name_is_empty() {
        let empty = LogicalStreamName::new("");
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = LogicalStreamName::new("prefetch");
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_stream_key_is_empty() {
        let empty = StreamKey::new("");
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = StreamKey::new("stream:prefetch");
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_stream_key_new_and_as_str() {
        let key = StreamKey::new("stream:prefetch");
        assert_eq!(key.as_str(), "stream:prefetch");
        assert_eq!(key.to_string(), "stream:prefetch");
    }

    #[test]
    fn test_handoff_request_new_sets_all_fields() {
        let req = HandoffRequest::new(
            "stream:prefetch",
            "stream:inference",
            "prefetch:grp",
            "run-001",
            r#"{"x":1}"#,
            "1700000000000-0",
            "inference",
        )
        .expect("valid handoff request should construct");

        assert_eq!(req.current_stream().as_str(), "stream:prefetch");
        assert_eq!(req.next_stream().as_str(), "stream:inference");
        assert_eq!(req.group(), "prefetch:grp");
        assert_eq!(req.run_id(), "run-001");
        assert_eq!(req.payload_json(), r#"{"x":1}"#);
        assert_eq!(req.current_msg_id(), "1700000000000-0");
        assert_eq!(req.next_stage(), "inference");
    }

    #[test]
    fn test_handoff_request_new_rejects_empty_run_id() {
        let req = HandoffRequest::new(
            "stream:prefetch",
            "stream:inference",
            "prefetch:grp",
            "   ",
            r#"{"x":1}"#,
            "1700000000000-0",
            "inference",
        );
        assert!(req.is_err(), "empty run_id must be rejected");
    }

    #[test]
    fn test_handoff_request_new_rejects_other_empty_critical_fields() {
        let cases = [
            (
                "",
                "stream:inference",
                "prefetch:grp",
                "run-1",
                r#"{"x":1}"#,
                "1700000000000-0",
                "inference",
                "current_stream",
            ),
            (
                "stream:prefetch",
                "",
                "prefetch:grp",
                "run-1",
                r#"{"x":1}"#,
                "1700000000000-0",
                "inference",
                "next_stream",
            ),
            (
                "stream:prefetch",
                "stream:inference",
                " ",
                "run-1",
                r#"{"x":1}"#,
                "1700000000000-0",
                "inference",
                "group",
            ),
            (
                "stream:prefetch",
                "stream:inference",
                "prefetch:grp",
                "run-1",
                "   ",
                "1700000000000-0",
                "inference",
                "payload_json",
            ),
            (
                "stream:prefetch",
                "stream:inference",
                "prefetch:grp",
                "run-1",
                r#"{"x":1}"#,
                "",
                "inference",
                "current_msg_id",
            ),
            (
                "stream:prefetch",
                "stream:inference",
                "prefetch:grp",
                "run-1",
                r#"{"x":1}"#,
                "1700000000000-0",
                "   ",
                "next_stage",
            ),
        ];

        for (
            current_stream,
            next_stream,
            group,
            run_id,
            payload_json,
            current_msg_id,
            next_stage,
            field_name,
        ) in cases
        {
            let req = HandoffRequest::new(
                current_stream,
                next_stream,
                group,
                run_id,
                payload_json,
                current_msg_id,
                next_stage,
            );
            assert!(
                matches!(req, Err(QueueError::Config(_))),
                "empty {field_name} must be rejected"
            );
        }
    }

    // =========================================================================
    // Message Tests
    // =========================================================================

    #[test]
    fn test_message_new() {
        let msg = Message::new(
            "1706123456789-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-123",
            r#"{"key": "value"}"#,
            "prefetch",
        );
        assert_eq!(msg.id, "1706123456789-0");
        assert_eq!(msg.stream, "stream:prefetch");
        assert_eq!(msg.group, "prefetch:grp");
        assert_eq!(msg.run_id, "run-123");
        assert_eq!(msg.payload, r#"{"key": "value"}"#);
        assert_eq!(msg.stage, "prefetch");
    }

    #[test]
    fn test_message_try_new_rejects_empty_critical_fields() {
        let cases = [
            ("", "stream:test", "grp", "run-1", "id"),
            ("1-0", "", "grp", "run-1", "stream"),
            ("1-0", "stream:test", "", "run-1", "group"),
            ("1-0", "stream:test", "grp", "", "run_id"),
        ];

        for (id, stream, group, run_id, expected_field) in cases {
            let result = Message::try_new(id, stream, group, run_id, "{}", "stage");
            assert!(
                matches!(result, Err(QueueError::Config(_))),
                "expected QueueError::Config for empty {expected_field}"
            );
        }
    }

    #[test]
    fn test_message_try_new_accepts_valid_message() {
        let result = Message::try_new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        assert!(result.is_ok(), "valid message should construct");
    }

    #[test]
    fn test_message_has_stream_field() {
        let msg = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        assert_eq!(msg.stream, "stream:test");
    }

    #[test]
    fn test_message_has_group_field() {
        let msg = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        assert_eq!(msg.group, "grp");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""id":"1-0""#));
        assert!(json.contains(r#""stream":"stream:test""#));
        assert!(json.contains(r#""group":"grp""#));
        assert!(json.contains(r#""run_id":"run-1""#));
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{"id":"1-0","stream":"stream:test","group":"grp","run_id":"run-123","payload":"{}","stage":"prefetch"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id, "1-0");
        assert_eq!(msg.stream, "stream:test");
        assert_eq!(msg.group, "grp");
        assert_eq!(msg.run_id, "run-123");
        assert_eq!(msg.payload, "{}");
        assert_eq!(msg.stage, "prefetch");
    }

    #[test]
    fn test_message_parse_payload() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "grp",
            "run-123",
            r#"{"count": 42}"#,
            "prefetch",
        );

        #[derive(Deserialize)]
        struct Payload {
            count: i32,
        }

        let payload: Payload = msg.parse_payload().unwrap();
        assert_eq!(payload.count, 42);
    }

    #[test]
    fn test_message_parse_payload_invalid() {
        let msg = Message::new(
            "1-0",
            "stream:test",
            "grp",
            "run-123",
            "not json",
            "prefetch",
        );

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Payload {
            count: i32,
        }

        let result: Result<Payload, _> = msg.parse_payload();
        assert!(result.is_err());
    }

    #[test]
    fn test_message_eq() {
        let msg1 = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        let msg2 = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        let msg3 = Message::new("2-0", "stream:test", "grp", "run-1", "{}", "stage");

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }

    #[test]
    fn test_message_clone() {
        let msg = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        let cloned = msg.clone();
        assert_eq!(msg, cloned);
    }

    // =========================================================================
    // Output Tests
    // =========================================================================

    #[test]
    fn test_output_new() {
        let out = Output::new("stream:results", r#"{"status": "ok"}"#);
        assert_eq!(out.stream, "stream:results");
        assert_eq!(out.payload, r#"{"status": "ok"}"#);
        assert!(out.stage.is_none());
    }

    #[test]
    fn test_output_with_stage() {
        let out = Output::new("stream:results", "{}").with_stage("final_results");
        assert_eq!(out.stream, "stream:results");
        assert_eq!(out.stage, Some("final_results".to_string()));
    }

    #[test]
    fn test_output_eq() {
        let out1 = Output::new("stream:test", "{}");
        let out2 = Output::new("stream:test", "{}");
        let out3 = Output::new("stream:other", "{}");

        assert_eq!(out1, out2);
        assert_ne!(out1, out3);
    }

    #[test]
    fn test_output_clone() {
        let out = Output::new("stream:test", "{}").with_stage("stage");
        let cloned = out.clone();
        assert_eq!(out, cloned);
    }

    // =========================================================================
    // HealthStatus Tests
    // =========================================================================

    #[test]
    fn test_health_status_is_healthy() {
        let healthy = HealthStatus {
            connected: true,
            latency_ms: 5,
            script_loaded: true,
        };
        assert!(healthy.is_healthy());

        let not_connected = HealthStatus {
            connected: false,
            latency_ms: 0,
            script_loaded: true,
        };
        assert!(!not_connected.is_healthy());

        let no_script = HealthStatus {
            connected: true,
            latency_ms: 5,
            script_loaded: false,
        };
        assert!(
            no_script.is_healthy(),
            "health should report connected managers as healthy even before lazy script loading"
        );
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
        assert_eq!(status1, status2);
    }
}
