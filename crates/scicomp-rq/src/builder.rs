/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder pattern for ergonomic API construction.
//!
//! This module provides fluent builder APIs for constructing `QueueManager`
//! and queue operations.
//!
//! ```compile_fail
//! // Standalone builder types are intentionally not public.
//! use scicomp_rq::builder::EnqueueBuilder;
//! ```
//!
//! ```compile_fail
//! // Standalone builder types are intentionally not public.
//! use scicomp_rq::builder::HandoffBuilder;
//! ```

use crate::traits::{AtomicOps, EnqueueOps};
use crate::{
    ConnectionManagerConfig, HandoffRequest, LogicalStreamName, Message, QueueError, QueueManager,
    Result, StreamKey,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptKind {
    Handoff,
    ForwardMany,
}

fn preload_script_kinds(preload: bool) -> &'static [ScriptKind] {
    if preload {
        &[ScriptKind::Handoff, ScriptKind::ForwardMany]
    } else {
        &[]
    }
}

fn validate_logical_stream_name(stream: &LogicalStreamName, field_name: &str) -> Result<()> {
    if stream.as_str().contains(':') {
        return Err(QueueError::Config(format!(
            "{field_name} must be a logical stream name without ':'; got '{}'",
            stream.as_str()
        )));
    }
    Ok(())
}

fn split_stream_key_suffix(stream_key: &str) -> (&str, &str) {
    match stream_key.rsplit_once(':') {
        Some((prefix, logical_name)) => (prefix, logical_name),
        None => ("", stream_key),
    }
}

fn validate_handoff_source_stream(
    from_stream: &LogicalStreamName,
    message_stream: &str,
) -> Result<String> {
    let (prefix, source_logical_name) = split_stream_key_suffix(message_stream);
    if from_stream.as_str() != source_logical_name {
        return Err(QueueError::Config(format!(
            "from_stream '{}' must match logical source '{}' derived from message.stream '{}'",
            from_stream.as_str(),
            source_logical_name,
            message_stream
        )));
    }
    Ok(prefix.to_string())
}

fn map_logical_destination_stream(
    message_stream: &str,
    to_stream: &LogicalStreamName,
) -> StreamKey {
    let (prefix, _) = split_stream_key_suffix(message_stream);
    if prefix.is_empty() {
        StreamKey::new(to_stream.as_str())
    } else {
        StreamKey::new(format!("{prefix}:{}", to_stream.as_str()))
    }
}

/// Builder for constructing a `QueueManager` with fluent API.
#[derive(Debug, Default)]
#[must_use = "builders do nothing until .build()/.send() is called"]
pub struct QueueManagerBuilder {
    redis_url: Option<String>,
    connection_manager: ConnectionManagerConfig,
    preload_scripts: bool,
}

impl QueueManagerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Redis connection URL.
    pub fn redis_url(mut self, url: impl Into<String>) -> Self {
        self.redis_url = Some(url.into());
        self
    }

    /// Set whether to preload Lua scripts when building.
    pub fn preload_scripts(mut self, preload: bool) -> Self {
        self.preload_scripts = preload;
        self
    }

    /// Set Redis connection timeout in milliseconds.
    pub fn connection_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.connection_manager.connection_timeout =
            Some(std::time::Duration::from_millis(timeout_ms));
        self
    }

    /// Set Redis response timeout in milliseconds.
    pub fn response_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.connection_manager.response_timeout =
            Some(std::time::Duration::from_millis(timeout_ms));
        self
    }

    /// Set reconnect policy tuning.
    pub fn reconnect_policy(
        mut self,
        exponent_base: u64,
        factor: u64,
        retries: usize,
        max_delay_ms: Option<u64>,
    ) -> Self {
        self.connection_manager.retry_exponent_base = Some(exponent_base);
        self.connection_manager.retry_factor = Some(factor);
        self.connection_manager.retry_count = Some(retries);
        self.connection_manager.max_retry_delay_ms = max_delay_ms;
        self
    }

    /// Build the `QueueManager`.
    pub async fn build(self) -> Result<QueueManager> {
        let redis_url = self
            .redis_url
            .ok_or_else(|| QueueError::Config("redis_url is required".into()))?;

        let qm =
            QueueManager::new_with_connection_config(&redis_url, &self.connection_manager).await?;

        for script in preload_script_kinds(self.preload_scripts) {
            match script {
                ScriptKind::Handoff => {
                    qm.load_handoff_script().await?;
                }
                ScriptKind::ForwardMany => {
                    qm.load_forward_many_script().await?;
                }
            }
        }

        Ok(qm)
    }
}

#[derive(Debug, Clone)]
struct ValidatedEnqueueParams {
    stream: LogicalStreamName,
    run_id: String,
    payload: String,
    stage: String,
}

/// Builder for constructing enqueue operations with a queue backend.
#[derive(Debug)]
#[must_use = "builders do nothing until .build()/.send() is called"]
pub struct EnqueueBuilderWithQM<'a, Q: EnqueueOps + ?Sized = QueueManager> {
    qm: &'a Q,
    stream: LogicalStreamName,
    run_id: Option<String>,
    payload: Option<String>,
    stage: Option<String>,
}

impl<'a, Q> EnqueueBuilderWithQM<'a, Q>
where
    Q: EnqueueOps + ?Sized,
{
    pub(crate) fn new(qm: &'a Q, stream: impl Into<LogicalStreamName>) -> Self {
        Self {
            qm,
            stream: stream.into(),
            run_id: None,
            payload: None,
            stage: None,
        }
    }

    /// Set the run identifier.
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set the JSON payload text.
    pub fn payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Set the stage name (defaults to stream name when omitted).
    pub fn stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    fn validate(self) -> Result<ValidatedEnqueueParams> {
        let Self {
            stream,
            run_id,
            payload,
            stage,
            ..
        } = self;

        if stream.as_str().trim().is_empty() {
            return Err(QueueError::Config("stream must be non-empty".into()));
        }

        let run_id = run_id.ok_or_else(|| QueueError::Config("run_id is required".into()))?;
        if run_id.trim().is_empty() {
            return Err(QueueError::Config("run_id must be non-empty".into()));
        }

        let payload = payload.ok_or_else(|| QueueError::Config("payload is required".into()))?;

        let stage = match stage {
            Some(stage) => {
                if stage.trim().is_empty() {
                    return Err(QueueError::Config(
                        "stage must be non-empty when explicitly provided".into(),
                    ));
                }
                stage
            }
            None => stream.as_str().to_string(),
        };

        Ok(ValidatedEnqueueParams {
            stream,
            run_id,
            payload,
            stage,
        })
    }

    /// Validate and send the message to the stream.
    pub async fn send(self) -> Result<String> {
        let qm = self.qm;
        let params = self.validate()?;
        qm.enqueue(
            &params.stream,
            &params.run_id,
            &params.payload,
            &params.stage,
        )
        .await
    }
}

#[derive(Debug, Clone)]
struct ValidatedHandoffParams {
    current_stream: StreamKey,
    next_stream: StreamKey,
    current_group: String,
    message_id: String,
    run_id: String,
    payload: String,
    next_stage: String,
}

/// Builder for constructing handoff operations with a queue backend.
#[derive(Debug)]
#[must_use = "builders do nothing until .build()/.send() is called"]
pub struct HandoffBuilderWithQM<'a, Q: AtomicOps + ?Sized = QueueManager> {
    qm: &'a Q,
    from_stream: Option<LogicalStreamName>,
    to_stream: Option<LogicalStreamName>,
    message: Option<Message>,
    next_stage: Option<String>,
}

impl<'a, Q> HandoffBuilderWithQM<'a, Q>
where
    Q: AtomicOps + ?Sized,
{
    pub(crate) fn new(qm: &'a Q) -> Self {
        Self {
            qm,
            from_stream: None,
            to_stream: None,
            message: None,
            next_stage: None,
        }
    }

    /// Set the source logical stream name (no `:`).
    ///
    /// The value must match the logical suffix derived from `message.stream`.
    pub fn from(mut self, stream: impl Into<LogicalStreamName>) -> Self {
        self.from_stream = Some(stream.into());
        self
    }

    /// Set the destination logical stream name (no `:`).
    ///
    /// The destination stream key reuses the source prefix from `message.stream`.
    pub fn to(mut self, stream: impl Into<LogicalStreamName>) -> Self {
        self.to_stream = Some(stream.into());
        self
    }

    /// Set the message to hand off.
    pub fn message(mut self, msg: Message) -> Self {
        self.message = Some(msg);
        self
    }

    /// Set the stage name for the destination.
    pub fn stage(mut self, stage: impl Into<String>) -> Self {
        self.next_stage = Some(stage.into());
        self
    }

    fn validate(self) -> Result<ValidatedHandoffParams> {
        let Self {
            from_stream,
            to_stream,
            message,
            next_stage,
            ..
        } = self;

        let from_stream =
            from_stream.ok_or_else(|| QueueError::Config("from_stream is required".into()))?;
        if from_stream.as_str().trim().is_empty() {
            return Err(QueueError::Config("from_stream must be non-empty".into()));
        }
        validate_logical_stream_name(&from_stream, "from_stream")?;

        let to_stream =
            to_stream.ok_or_else(|| QueueError::Config("to_stream is required".into()))?;
        if to_stream.as_str().trim().is_empty() {
            return Err(QueueError::Config("to_stream must be non-empty".into()));
        }
        validate_logical_stream_name(&to_stream, "to_stream")?;

        let message = message.ok_or_else(|| QueueError::Config("message is required".into()))?;
        if message.run_id.trim().is_empty() {
            return Err(QueueError::Config(
                "message.run_id must be non-empty".into(),
            ));
        }
        if message.payload.trim().is_empty() {
            return Err(QueueError::Config(
                "message.payload must be non-empty".into(),
            ));
        }
        validate_handoff_source_stream(&from_stream, &message.stream)?;

        let next_stage = match next_stage {
            Some(stage) => {
                if stage.trim().is_empty() {
                    return Err(QueueError::Config(
                        "stage must be non-empty when explicitly provided".into(),
                    ));
                }
                stage
            }
            None => to_stream.as_str().to_string(),
        };

        Ok(ValidatedHandoffParams {
            current_stream: StreamKey::new(message.stream.clone()),
            next_stream: map_logical_destination_stream(&message.stream, &to_stream),
            current_group: message.group,
            message_id: message.id,
            run_id: message.run_id,
            payload: message.payload,
            next_stage,
        })
    }

    /// Validate and execute the handoff operation.
    pub async fn send(self) -> Result<String> {
        let qm = self.qm;
        let params = self.validate()?;
        let request = HandoffRequest::new(
            params.current_stream,
            params.next_stream,
            params.current_group,
            params.run_id,
            params.payload,
            params.message_id,
            params.next_stage,
        )?;

        qm.handoff_and_ack(&request).await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EnqueueBuilderWithQM, HandoffBuilderWithQM, ScriptKind, map_logical_destination_stream,
        preload_script_kinds, split_stream_key_suffix, validate_handoff_source_stream,
        validate_logical_stream_name,
    };
    use crate::{AtomicOps, EnqueueOps, LogicalStreamName, Message, QueueError};
    use std::sync::{Arc, Mutex};

    type EnqueueCall = (String, String, String, String);

    #[derive(Default, Clone)]
    struct RecordingEnqueueOps {
        calls: Arc<Mutex<Vec<EnqueueCall>>>,
    }

    impl EnqueueOps for RecordingEnqueueOps {
        async fn enqueue(
            &self,
            stream_name: &LogicalStreamName,
            run_id: &str,
            payload: &str,
            stage: &str,
        ) -> crate::Result<String> {
            self.calls
                .lock()
                .expect("recording lock should be available")
                .push((
                    stream_name.as_str().to_string(),
                    run_id.to_string(),
                    payload.to_string(),
                    stage.to_string(),
                ));
            Ok("42-0".to_string())
        }
    }

    struct NoopAtomicOps;

    impl AtomicOps for NoopAtomicOps {
        async fn handoff_and_ack(&self, _request: &crate::HandoffRequest) -> crate::Result<String> {
            Ok("1-0".to_string())
        }

        async fn handoff_message(
            &self,
            _message: &crate::Message,
            _dest_stream: &crate::StreamKey,
            _payload: Option<&str>,
            _stage: Option<&str>,
        ) -> crate::Result<String> {
            Ok("1-0".to_string())
        }

        async fn handoff_message_to_run(
            &self,
            _message: &crate::Message,
            _dest_stream: &crate::StreamKey,
            _payload: Option<&str>,
            _stage: Option<&str>,
            _run_id: Option<&str>,
        ) -> crate::Result<String> {
            Ok("1-0".to_string())
        }

        async fn forward_many(
            &self,
            _message: &crate::Message,
            _outputs: &[crate::Output],
        ) -> crate::Result<Vec<String>> {
            Ok(vec![])
        }
    }

    #[test]
    fn preload_script_kinds_includes_handoff_and_forward_many() {
        let kinds = preload_script_kinds(true);
        assert!(
            kinds.contains(&ScriptKind::Handoff),
            "preload_scripts(true) should include handoff script preload"
        );
        assert!(
            kinds.contains(&ScriptKind::ForwardMany),
            "preload_scripts(true) should include forward_many script preload"
        );
    }

    #[test]
    fn preload_script_kinds_returns_empty_when_disabled() {
        assert!(
            preload_script_kinds(false).is_empty(),
            "preload_scripts(false) should not preload Lua scripts"
        );
    }

    #[test]
    fn split_stream_key_suffix_handles_prefixed_and_unprefixed_stream_keys() {
        let (prefix, logical) = split_stream_key_suffix("physicsnemo:prefetch");
        assert_eq!(prefix, "physicsnemo");
        assert_eq!(logical, "prefetch");

        let (prefix, logical) = split_stream_key_suffix("prefetch");
        assert_eq!(prefix, "");
        assert_eq!(logical, "prefetch");
    }

    #[test]
    fn validate_logical_stream_name_accepts_plain_logical_name() {
        let stream = LogicalStreamName::new("prefetch");
        assert!(
            validate_logical_stream_name(&stream, "from_stream").is_ok(),
            "logical names without ':' should be accepted"
        );
    }

    #[test]
    fn validate_handoff_source_stream_rejects_mismatch() {
        let from_stream = LogicalStreamName::new("stream:prefetch");
        let result = validate_handoff_source_stream(&from_stream, "stream:other");
        assert!(
            result.is_err(),
            "builder should reject when from() does not match message.stream"
        );
    }

    #[test]
    fn validate_handoff_source_stream_returns_prefix_for_matching_source() {
        let from_stream = LogicalStreamName::new("prefetch");
        let prefix = validate_handoff_source_stream(&from_stream, "physicsnemo:prefetch")
            .expect("matching logical stream should validate");
        assert_eq!(prefix, "physicsnemo");

        let prefix_no_namespace = validate_handoff_source_stream(&from_stream, "prefetch")
            .expect("matching non-prefixed stream should validate");
        assert_eq!(prefix_no_namespace, "");
    }

    #[test]
    fn map_logical_destination_stream_reuses_source_prefix_when_present() {
        let destination = map_logical_destination_stream(
            "physicsnemo:prefetch",
            &LogicalStreamName::new("inference"),
        );
        assert_eq!(
            destination.as_str(),
            "physicsnemo:inference",
            "destination stream should preserve source prefix"
        );
    }

    #[test]
    fn map_logical_destination_stream_uses_logical_name_without_prefix() {
        let destination =
            map_logical_destination_stream("prefetch", &LogicalStreamName::new("inference"));
        assert_eq!(
            destination.as_str(),
            "inference",
            "destination stream should be bare logical name when no source prefix exists"
        );
    }

    #[test]
    fn enqueue_builder_validate_rejects_missing_run_id() {
        let backend = RecordingEnqueueOps::default();
        let builder = EnqueueBuilderWithQM::new(&backend, "prefetch").payload("{}");
        let result = builder.validate();
        let err = result.expect_err("missing run_id should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "enqueue builder must require run_id"
        );
        assert!(err.to_string().contains("run_id is required"));
    }

    #[test]
    fn enqueue_builder_validate_rejects_missing_payload() {
        let backend = RecordingEnqueueOps::default();
        let builder = EnqueueBuilderWithQM::new(&backend, "prefetch").run_id("run-1");
        let result = builder.validate();
        let err = result.expect_err("missing payload should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "enqueue builder must require payload"
        );
        assert!(err.to_string().contains("payload is required"));
    }

    #[test]
    fn enqueue_builder_validate_rejects_empty_run_id() {
        let backend = RecordingEnqueueOps::default();
        let builder = EnqueueBuilderWithQM::new(&backend, "prefetch")
            .run_id("   ")
            .payload("{}");
        let result = builder.validate();
        let err = result.expect_err("empty run_id should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "enqueue builder must reject empty run_id values"
        );
        assert!(err.to_string().contains("run_id must be non-empty"));
    }

    #[test]
    fn enqueue_builder_validate_rejects_empty_stream_name() {
        let backend = RecordingEnqueueOps::default();
        let builder = EnqueueBuilderWithQM::new(&backend, "  ")
            .run_id("run-1")
            .payload("{}");
        let result = builder.validate();
        let err = result.expect_err("empty stream should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "enqueue builder should reject empty/whitespace stream names"
        );
        assert!(err.to_string().contains("stream must be non-empty"));
    }

    #[test]
    fn enqueue_builder_validate_rejects_empty_explicit_stage() {
        let backend = RecordingEnqueueOps::default();
        let builder = EnqueueBuilderWithQM::new(&backend, "prefetch")
            .run_id("run-1")
            .payload("{}")
            .stage("   ");
        let result = builder.validate();
        let err = result.expect_err("empty explicit stage should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "enqueue builder should reject empty explicit stage values"
        );
        assert!(err.to_string().contains("stage must be non-empty"));
    }

    #[test]
    fn enqueue_builder_validate_defaults_stage_to_stream_name() {
        let backend = RecordingEnqueueOps::default();
        let params = EnqueueBuilderWithQM::new(&backend, "prefetch")
            .run_id("run-1")
            .payload("{}")
            .validate()
            .expect("builder should validate with default stage");
        assert_eq!(params.stage, "prefetch");
    }

    #[tokio::test]
    async fn enqueue_builder_send_passes_validated_values_to_enqueue_backend() {
        let backend = RecordingEnqueueOps::default();

        let first_id = EnqueueBuilderWithQM::new(&backend, "prefetch")
            .run_id("run-1")
            .payload(r#"{"ok":true}"#)
            .send()
            .await
            .expect("send should forward validated values");
        assert_eq!(first_id, "42-0");

        let second_id = EnqueueBuilderWithQM::new(&backend, "prefetch")
            .run_id("run-2")
            .payload("{}")
            .stage("dispatch")
            .send()
            .await
            .expect("send with explicit stage should succeed");
        assert_eq!(second_id, "42-0");

        let calls = backend
            .calls
            .lock()
            .expect("recording lock should be available");
        assert_eq!(calls.len(), 2, "expected two enqueue calls to be recorded");
        assert_eq!(
            calls[0],
            (
                "prefetch".to_string(),
                "run-1".to_string(),
                r#"{"ok":true}"#.to_string(),
                "prefetch".to_string(),
            )
        );
        assert_eq!(
            calls[1],
            (
                "prefetch".to_string(),
                "run-2".to_string(),
                "{}".to_string(),
                "dispatch".to_string(),
            )
        );
    }

    #[test]
    fn handoff_builder_validate_rejects_from_stream_mismatch() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let builder = HandoffBuilderWithQM::new(&ops)
            .from("stream:other")
            .to("stream:next")
            .message(msg);
        let result = builder.validate();
        assert!(
            matches!(result, Err(QueueError::Config(_))),
            "mismatch between from() and message.stream should fail validation"
        );
    }

    #[test]
    fn handoff_builder_validate_maps_logical_destination_using_source_prefix() {
        let msg = Message::new(
            "1-0",
            "physicsnemo:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let builder = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference")
            .message(msg);

        let params = builder
            .validate()
            .expect("logical-name contract should map destination using source prefix");
        assert_eq!(params.current_stream.as_str(), "physicsnemo:prefetch");
        assert_eq!(params.next_stream.as_str(), "physicsnemo:inference");
        assert_eq!(params.next_stage, "inference");
    }

    #[test]
    fn handoff_builder_validate_rejects_stream_key_values_for_logical_contract() {
        let msg = Message::new(
            "1-0",
            "physicsnemo:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let builder = HandoffBuilderWithQM::new(&ops)
            .from("physicsnemo:prefetch")
            .to("physicsnemo:inference")
            .message(msg);

        let result = builder.validate();
        assert!(
            matches!(result, Err(QueueError::Config(_))),
            "handoff builder must reject stream-key values when API contract is logical names"
        );
    }

    #[test]
    fn handoff_builder_validate_rejects_missing_required_inputs() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;

        let missing_from = HandoffBuilderWithQM::new(&ops)
            .to("inference")
            .message(msg.clone());
        let missing_from_err = missing_from
            .validate()
            .expect_err("missing from_stream should fail validation");
        assert!(matches!(missing_from_err, QueueError::Config(_)));
        assert!(
            missing_from_err
                .to_string()
                .contains("from_stream is required")
        );

        let missing_to = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .message(msg.clone());
        let missing_to_err = missing_to
            .validate()
            .expect_err("missing to_stream should fail validation");
        assert!(matches!(missing_to_err, QueueError::Config(_)));
        assert!(missing_to_err.to_string().contains("to_stream is required"));

        let missing_message = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference");
        let missing_message_err = missing_message
            .validate()
            .expect_err("missing message should fail validation");
        assert!(matches!(missing_message_err, QueueError::Config(_)));
        assert!(
            missing_message_err
                .to_string()
                .contains("message is required")
        );
    }

    #[test]
    fn handoff_builder_validate_rejects_empty_from_stream() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let result = HandoffBuilderWithQM::new(&ops)
            .from("   ")
            .to("inference")
            .message(msg)
            .validate();
        let err = result.expect_err("empty from_stream should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "handoff builder should reject empty from_stream"
        );
        assert!(err.to_string().contains("from_stream must be non-empty"));
    }

    #[test]
    fn handoff_builder_validate_rejects_empty_to_stream() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let result = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("   ")
            .message(msg)
            .validate();
        let err = result.expect_err("empty to_stream should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "handoff builder should reject empty to_stream"
        );
        assert!(err.to_string().contains("to_stream must be non-empty"));
    }

    #[test]
    fn handoff_builder_validate_rejects_empty_message_context_fields() {
        let ops = NoopAtomicOps;

        let empty_run_id = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "  ",
            "{}",
            "prefetch",
        );
        let run_id_result = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference")
            .message(empty_run_id)
            .validate();
        let run_id_err = run_id_result.expect_err("empty message.run_id should fail");
        assert!(
            matches!(run_id_err, QueueError::Config(_)),
            "handoff builder should reject empty message.run_id"
        );
        assert!(
            run_id_err
                .to_string()
                .contains("message.run_id must be non-empty")
        );

        let empty_payload = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "   ",
            "prefetch",
        );
        let payload_result = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference")
            .message(empty_payload)
            .validate();
        let payload_err = payload_result.expect_err("empty message.payload should fail");
        assert!(
            matches!(payload_err, QueueError::Config(_)),
            "handoff builder should reject empty message.payload"
        );
        assert!(
            payload_err
                .to_string()
                .contains("message.payload must be non-empty")
        );
    }

    #[test]
    fn handoff_builder_validate_rejects_empty_explicit_stage() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let result = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference")
            .message(msg)
            .stage("   ")
            .validate();
        let err = result.expect_err("empty explicit stage should fail validation");
        assert!(
            matches!(err, QueueError::Config(_)),
            "handoff builder should reject empty explicit stage"
        );
        assert!(err.to_string().contains("stage must be non-empty"));
    }

    #[test]
    fn handoff_builder_validate_accepts_explicit_stage_override() {
        let msg = Message::new(
            "1-0",
            "stream:prefetch",
            "prefetch:grp",
            "run-1",
            "{}",
            "prefetch",
        );
        let ops = NoopAtomicOps;
        let params = HandoffBuilderWithQM::new(&ops)
            .from("prefetch")
            .to("inference")
            .message(msg)
            .stage("dispatch")
            .validate()
            .expect("explicit stage should be accepted");
        assert_eq!(params.next_stage, "dispatch");
    }

    #[test]
    fn builder_types_are_marked_must_use() {
        let source = include_str!("builder.rs");
        let must_use_text =
            "#[must_use = \"builders do nothing until .build()/.send() is called\"]";
        assert!(
            source.contains("pub struct QueueManagerBuilder"),
            "QueueManagerBuilder should be marked #[must_use]"
        );
        assert!(
            source.contains("pub struct EnqueueBuilderWithQM"),
            "EnqueueBuilderWithQM should be marked #[must_use]"
        );
        assert!(
            source.contains("pub struct HandoffBuilderWithQM"),
            "HandoffBuilderWithQM should be marked #[must_use]"
        );
        assert!(
            source.contains(must_use_text),
            "builder types should carry the must_use annotation message"
        );
    }
}
