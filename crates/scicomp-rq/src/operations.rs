/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! QueueManager operation implementations.
//!
//! All queue-domain Redis stream operations (read, enqueue, handoff, ack,
//! claim, group management) live here.

use tracing::warn;

use crate::builder;
use crate::constants::{fields, keys};
use crate::error::{QueueError, Result};
use crate::lua::{
    LUA_FORWARD_MANY, LUA_HANDOFF, derive_stage_from_stream, is_noscript_error,
    is_xautoclaim_unsupported,
};
use crate::manager::QueueManager;
use crate::redis_utils;
use crate::types::{HandoffRequest, HealthStatus, LogicalStreamName, Message, Output, StreamKey};

fn validate_enqueue_stream_name(stream_name: &LogicalStreamName) -> Result<()> {
    if stream_name.as_str().trim().is_empty() {
        return Err(QueueError::Config("stream name must be non-empty".into()));
    }
    if stream_name.as_str().contains(':') {
        return Err(QueueError::Config(
            "enqueue expects a logical stream name without ':'; use enqueue_to_stream for explicit Redis stream keys".into(),
        ));
    }
    Ok(())
}

fn validate_payload_json(payload: &str) -> Result<()> {
    serde_json::from_str::<serde_json::Value>(payload).map_err(|e| {
        QueueError::Config(format!(
            "payload must be valid JSON for enqueue operations: {e}"
        ))
    })?;
    Ok(())
}

fn validate_message_context(message: &Message) -> Result<()> {
    if message.id().trim().is_empty() {
        return Err(QueueError::Config("message.id must be non-empty".into()));
    }
    if message.stream().trim().is_empty() {
        return Err(QueueError::Config(
            "message.stream must be non-empty".into(),
        ));
    }
    if message.group().trim().is_empty() {
        return Err(QueueError::Config("message.group must be non-empty".into()));
    }
    if message.run_id().trim().is_empty() {
        return Err(QueueError::Config(
            "message.run_id must be non-empty".into(),
        ));
    }
    Ok(())
}

fn validate_forward_many_outputs(outputs: &[Output]) -> Result<()> {
    if outputs.is_empty() {
        return Err(QueueError::Config(
            "forward_many requires at least one output".into(),
        ));
    }
    for output in outputs {
        if let Some(run_id) = output.run_id()
            && run_id.trim().is_empty()
        {
            return Err(QueueError::Config(
                "forward_many output run_id must be non-empty when provided".into(),
            ));
        }
    }
    Ok(())
}

impl QueueManager {
    /// Check Redis connection health and return status.
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let mut conn = self.conn.clone();
        let start = std::time::Instant::now();

        let pong: String = redis::cmd("PING").query_async(&mut conn).await?;
        let latency = start.elapsed();

        let script_loaded = self.lua_handoff_sha.read().await.is_some()
            || self.lua_forward_many_sha.read().await.is_some();

        Ok(HealthStatus {
            connected: pong == "PONG",
            latency_ms: latency.as_millis() as u64,
            script_loaded,
        })
    }

    /// Read messages from a Redis stream using XREADGROUP.
    ///
    /// This is the primary method for consuming messages from a stream. Each returned
    /// [`Message`] contains the stream and group context needed for subsequent operations
    /// like [`ack_message()`](Self::ack_message) and [`handoff_message()`](Self::handoff_message).
    ///
    /// # Arguments
    ///
    /// * `stream` - Full Redis stream key (e.g., "stream:prefetch")
    /// * `group` - Consumer group name (e.g., "prefetch:grp")
    /// * `consumer` - Consumer name within the group
    /// * `count` - Maximum number of messages to read
    /// * `block_ms` - Block timeout in milliseconds (0 = non-blocking)
    ///
    /// # Returns
    ///
    /// A vector of [`Message`] objects. Empty if no messages are available within
    /// the block timeout.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example() -> scicomp_rq::Result<()> {
    /// # let qm = scicomp_rq::QueueManager::from_redis_url("redis://localhost").await?;
    /// # let stream = scicomp_rq::StreamKey::new("stream:prefetch");
    /// let messages = qm.read_messages(
    ///     &stream,
    ///     "prefetch:grp",
    ///     "worker-1",
    ///     10,  // count
    ///     5000, // block_ms
    /// ).await?;
    ///
    /// for msg in messages {
    ///     println!("Processing: {} from {}", msg.id(), msg.stream());
    ///     qm.ack_message(&msg).await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        count: usize,
        block_ms: usize,
    ) -> Result<Vec<Message>> {
        let mut conn = self.conn.clone();

        let response: redis::Value = redis::cmd("XREADGROUP")
            .arg("GROUP")
            .arg(group)
            .arg(consumer)
            .arg("BLOCK")
            .arg(block_ms)
            .arg("COUNT")
            .arg(count)
            .arg("STREAMS")
            .arg(stream.as_str())
            .arg(">")
            .query_async(&mut conn)
            .await?;

        Ok(redis_utils::parse_stream_messages(
            response,
            stream.as_str(),
            group,
        ))
    }

    /// Create a builder for enqueueing a message to a stream.
    ///
    /// This is the recommended fluent API for enqueueing messages.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example() -> scicomp_rq::Result<()> {
    /// # let qm = scicomp_rq::QueueManager::from_redis_url("redis://localhost").await?;
    /// let msg_id = qm.enqueue_to("prefetch")
    ///     .run_id("run-001")
    ///     .payload(r#"{"model":"pangu"}"#)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn enqueue_to(
        &self,
        stream_name: impl Into<LogicalStreamName>,
    ) -> builder::EnqueueBuilderWithQM<'_> {
        builder::EnqueueBuilderWithQM::new(self, stream_name)
    }

    /// Create a builder for handing off a message to another stream.
    ///
    /// This is the recommended fluent API for handoff operations.
    /// `from(...)` and `to(...)` accept logical stream names. The destination stream
    /// key reuses the prefix already present in `message.stream`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example() -> scicomp_rq::Result<()> {
    /// # let qm = scicomp_rq::QueueManager::from_redis_url("redis://localhost").await?;
    /// # let msg = scicomp_rq::Message::new("id", "stream", "grp", "run", "{}", "stage");
    /// let next_id = qm.handoff_builder()
    ///     .from("prefetch")
    ///     .to("inference")
    ///     .message(msg)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn handoff_builder(&self) -> builder::HandoffBuilderWithQM<'_> {
        builder::HandoffBuilderWithQM::new(self)
    }

    /// Enqueue a message to a stream.
    ///
    /// Payloads are contractually JSON text in both Rust and Python APIs.
    pub async fn enqueue(
        &self,
        stream_name: &LogicalStreamName,
        run_id: &str,
        payload: &str,
        stage: &str,
    ) -> Result<String> {
        validate_enqueue_stream_name(stream_name)?;
        let stream_key = StreamKey::new(stream_name.as_str());
        self.enqueue_to_stream(&stream_key, run_id, payload, stage)
            .await
    }

    /// Enqueue a message to an explicit Redis stream key.
    ///
    /// This is an advanced API/escape hatch that bypasses logical-name validation.
    /// Prefer `enqueue(&LogicalStreamName, ...)` for caller-facing contracts.
    ///
    /// Payloads are validated as JSON text to keep Rust/Python behavior aligned.
    pub async fn enqueue_to_stream(
        &self,
        stream_key: &StreamKey,
        run_id: &str,
        payload: &str,
        stage: &str,
    ) -> Result<String> {
        if run_id.trim().is_empty() {
            return Err(QueueError::Config("run_id must be non-empty".into()));
        }
        if stage.trim().is_empty() {
            return Err(QueueError::Config("stage must be non-empty".into()));
        }
        validate_payload_json(payload)?;

        let mut conn = self.conn.clone();

        let id: String = redis::cmd("XADD")
            .arg(stream_key.as_str())
            .arg("*")
            .arg(fields::RUN_ID)
            .arg(run_id)
            .arg(fields::PAYLOAD)
            .arg(payload)
            .arg(fields::STAGE)
            .arg(stage)
            .query_async(&mut conn)
            .await?;

        Ok(id)
    }

    /// Load (and cache) the SHA of the Lua handoff script on the Redis server.
    pub async fn load_handoff_script(&self) -> Result<String> {
        if let Some(existing) = self.lua_handoff_sha.read().await.clone() {
            return Ok(existing);
        }

        let mut conn = self.conn.clone();
        let sha: String = redis::cmd("SCRIPT")
            .arg("LOAD")
            .arg(LUA_HANDOFF)
            .query_async(&mut conn)
            .await?;

        {
            let mut w = self.lua_handoff_sha.write().await;
            *w = Some(sha.clone());
        }
        Ok(sha)
    }

    /// Force-reload handoff Lua script in Redis and refresh local SHA cache.
    async fn force_reload_handoff_script(&self) -> Result<String> {
        let mut conn = self.conn.clone();
        let sha: String = redis::cmd("SCRIPT")
            .arg("LOAD")
            .arg(LUA_HANDOFF)
            .query_async(&mut conn)
            .await
            .map_err(|e| QueueError::Script(format!("failed to reload handoff script: {e}")))?;

        let mut w = self.lua_handoff_sha.write().await;
        *w = Some(sha.clone());
        Ok(sha)
    }

    /// Load (and cache) the SHA of the Lua forward_many script on Redis.
    pub(crate) async fn load_forward_many_script(&self) -> Result<String> {
        if let Some(existing) = self.lua_forward_many_sha.read().await.clone() {
            return Ok(existing);
        }

        let mut conn = self.conn.clone();
        let sha: String = redis::cmd("SCRIPT")
            .arg("LOAD")
            .arg(LUA_FORWARD_MANY)
            .query_async(&mut conn)
            .await?;

        let mut w = self.lua_forward_many_sha.write().await;
        *w = Some(sha.clone());
        Ok(sha)
    }

    /// Force-reload forward_many Lua script in Redis and refresh local SHA cache.
    async fn force_reload_forward_many_script(&self) -> Result<String> {
        let mut conn = self.conn.clone();
        let sha: String = redis::cmd("SCRIPT")
            .arg("LOAD")
            .arg(LUA_FORWARD_MANY)
            .query_async(&mut conn)
            .await
            .map_err(|e| {
                QueueError::Script(format!("failed to reload forward_many script: {e}"))
            })?;

        let mut w = self.lua_forward_many_sha.write().await;
        *w = Some(sha.clone());
        Ok(sha)
    }

    async fn eval_forward_many_with_sha(
        &self,
        sha: &str,
        message: &Message,
        outputs: &[(String, String, String, String)],
    ) -> Result<Vec<String>> {
        let run_hash_prefix = keys::RUN_HASH_PREFIX;
        let output_count = outputs.len();
        let mut conn = self.conn.clone();

        let mut cmd = redis::cmd("EVALSHA");
        cmd.arg(sha).arg(output_count + 1);

        for (stream, _, _, _) in outputs {
            cmd.arg(stream);
        }
        cmd.arg(&message.stream);

        cmd.arg(&message.group)
            .arg(&message.id)
            .arg(output_count)
            .arg(run_hash_prefix);
        for (_, run_id, payload, stage) in outputs {
            cmd.arg(run_id).arg(payload).arg(stage);
        }

        let ids: Vec<String> = cmd.query_async(&mut conn).await?;
        Ok(ids)
    }

    async fn eval_handoff_with_sha(&self, sha: &str, request: &HandoffRequest) -> Result<String> {
        let run_hash_prefix = keys::RUN_HASH_PREFIX;
        let run_id = &request.run_id;
        let run_hash = format!("{run_hash_prefix}{run_id}");
        let mut conn = self.conn.clone();

        let next_id: String = redis::cmd("EVALSHA")
            .arg(sha)
            .arg(3)
            .arg(request.next_stream.as_str())
            .arg(request.current_stream.as_str())
            .arg(&run_hash)
            .arg(&request.run_id)
            .arg(&request.payload_json)
            .arg(&request.group)
            .arg(&request.current_msg_id)
            .arg(&request.next_stage)
            .query_async(&mut conn)
            .await?;

        Ok(next_id)
    }

    /// Atomically hand off a message to the next stream and acknowledge the current delivery.
    ///
    /// ```compile_fail
    /// # async fn old_signature(
    /// #     qm: &scicomp_rq::QueueManager,
    /// #     current_stream: &scicomp_rq::StreamKey,
    /// #     next_stream: &scicomp_rq::StreamKey,
    /// # ) {
    /// // Old positional-argument contract must not be available.
    /// let _ = qm
    ///     .handoff_and_ack(
    ///         current_stream,
    ///         next_stream,
    ///         "prefetch:grp",
    ///         "run-001",
    ///         r#"{"x":1}"#,
    ///         "1700000000000-0",
    ///         "inference",
    ///     )
    ///     .await;
    /// # }
    /// ```
    pub async fn handoff_and_ack(&self, request: &HandoffRequest) -> Result<String> {
        let sha = self.load_handoff_script().await?;
        match self.eval_handoff_with_sha(&sha, request).await {
            Ok(next_id) => Ok(next_id),
            Err(QueueError::Redis(err)) if is_noscript_error(&err) => {
                warn!(
                    error = %err,
                    "handoff EVALSHA returned NOSCRIPT, reloading script and retrying once"
                );
                let reloaded_sha = self.force_reload_handoff_script().await?;
                self.eval_handoff_with_sha(&reloaded_sha, request)
                    .await
                    .map_err(|retry_err| {
                        QueueError::Script(format!(
                            "handoff EVALSHA failed after NOSCRIPT retry: {retry_err}"
                        ))
                    })
            }
            Err(err) => Err(err),
        }
    }

    /// Claim idle pending messages and return full [`Message`] objects.
    pub async fn claim_idle_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        min_idle_ms: u64,
        start_id: &str,
        count: usize,
    ) -> Result<(String, Vec<Message>)> {
        let mut conn = self.conn.clone();
        let xautoclaim_result: Result<redis::Value> = redis::cmd("XAUTOCLAIM")
            .arg(stream.as_str())
            .arg(group)
            .arg(consumer)
            .arg(min_idle_ms)
            .arg(start_id)
            .arg("COUNT")
            .arg(count)
            .query_async(&mut conn)
            .await
            .map_err(QueueError::Redis);

        let err = match xautoclaim_result {
            Ok(val) => return redis_utils::parse_xautoclaim_messages(val, stream.as_str(), group),
            Err(e) => e,
        };

        if !is_xautoclaim_unsupported(&err) {
            return Err(err);
        }

        let pending_val: redis::Value = redis::cmd("XPENDING")
            .arg(stream.as_str())
            .arg(group)
            .arg("-")
            .arg("+")
            .arg(count)
            .query_async(&mut conn)
            .await
            .map_err(QueueError::Redis)?;

        let ids = redis_utils::parse_xpending_ids(pending_val);
        if ids.is_empty() {
            return Ok(("0-0".to_string(), Vec::new()));
        }

        let mut cmd = redis::cmd("XCLAIM");
        cmd.arg(stream.as_str())
            .arg(group)
            .arg(consumer)
            .arg(min_idle_ms);
        for id in &ids {
            cmd.arg(id);
        }

        let val: redis::Value = cmd
            .query_async(&mut conn)
            .await
            .map_err(QueueError::Redis)?;
        Ok((
            "0-0".to_string(),
            redis_utils::parse_xclaim_messages(val, stream.as_str(), group),
        ))
    }

    /// Acknowledge a single message using its stream and group context.
    pub async fn ack_message(&self, message: &Message) -> Result<i64> {
        validate_message_context(message)?;
        let mut conn = self.conn.clone();

        let n: i64 = redis::cmd("XACK")
            .arg(&message.stream)
            .arg(&message.group)
            .arg(&message.id)
            .query_async(&mut conn)
            .await?;

        Ok(n)
    }

    /// Atomically hand off a message to a destination stream.
    pub async fn handoff_message(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
    ) -> Result<String> {
        self.handoff_message_to_run(message, dest_stream, payload, stage, None)
            .await
    }

    /// Atomically hand off a message to a destination stream, with an optional
    /// destination run_id override.
    pub async fn handoff_message_to_run(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<String> {
        validate_message_context(message)?;
        let payload_json = payload.unwrap_or(message.payload());

        let next_stage = match stage {
            Some(s) => s.to_string(),
            None => derive_stage_from_stream(dest_stream.as_str(), ""),
        };
        let next_run_id = run_id.unwrap_or(message.run_id());

        let request = HandoffRequest::new(
            StreamKey::new(message.stream().to_string()),
            dest_stream.clone(),
            message.group().to_string(),
            next_run_id.to_string(),
            payload_json.to_string(),
            message.id().to_string(),
            next_stage,
        )?;

        self.handoff_and_ack(&request).await
    }

    /// Forward a message to multiple destination streams atomically.
    pub async fn forward_many(&self, message: &Message, outputs: &[Output]) -> Result<Vec<String>> {
        validate_message_context(message)?;
        validate_forward_many_outputs(outputs)?;
        let resolved_outputs: Vec<(String, String, String, String)> = outputs
            .iter()
            .map(|output| {
                let run_id = output
                    .run_id
                    .clone()
                    .unwrap_or_else(|| message.run_id().to_string());
                let stage = output
                    .stage
                    .clone()
                    .unwrap_or_else(|| derive_stage_from_stream(&output.stream, ""));
                (output.stream.clone(), run_id, output.payload.clone(), stage)
            })
            .collect();

        let sha = self.load_forward_many_script().await?;
        match self
            .eval_forward_many_with_sha(&sha, message, &resolved_outputs)
            .await
        {
            Ok(ids) => Ok(ids),
            Err(QueueError::Redis(err)) if is_noscript_error(&err) => {
                warn!(
                    error = %err,
                    "forward_many EVALSHA returned NOSCRIPT, reloading script and retrying once"
                );
                let reloaded_sha = self.force_reload_forward_many_script().await?;
                self.eval_forward_many_with_sha(&reloaded_sha, message, &resolved_outputs)
                    .await
                    .map_err(|retry_err| {
                        QueueError::Script(format!(
                            "forward_many EVALSHA failed after NOSCRIPT retry: {retry_err}"
                        ))
                    })
            }
            Err(err) => Err(err),
        }
    }

    /// Create a consumer group for a Redis stream.
    pub async fn create_consumer_group(
        &self,
        stream: &StreamKey,
        group: &str,
        start_id: &str,
        create_stream: bool,
    ) -> Result<bool> {
        let mut conn = self.conn.clone();

        let mut cmd = redis::cmd("XGROUP");
        cmd.arg("CREATE")
            .arg(stream.as_str())
            .arg(group)
            .arg(start_id);

        if create_stream {
            cmd.arg("MKSTREAM");
        }

        let result: std::result::Result<redis::Value, redis::RedisError> =
            cmd.query_async(&mut conn).await;

        match result {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("BUSYGROUP") {
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_enqueue_stream_name, validate_forward_many_outputs, validate_message_context,
        validate_payload_json,
    };
    use crate::{LogicalStreamName, Message, Output, QueueError};

    #[test]
    fn validate_enqueue_stream_name_rejects_empty() {
        let stream = LogicalStreamName::new("");
        let result = validate_enqueue_stream_name(&stream);
        assert!(result.is_err(), "empty stream names must be rejected");
    }

    #[test]
    fn validate_enqueue_stream_name_rejects_colon_delimited_stream_keys() {
        let stream = LogicalStreamName::new("stream:prefetch");
        let result = validate_enqueue_stream_name(&stream);
        assert!(
            result.is_err(),
            "logical stream names should not accept stream-key formatted values"
        );
    }

    #[test]
    fn validate_enqueue_stream_name_accepts_logical_name() {
        let stream = LogicalStreamName::new("prefetch");
        let result = validate_enqueue_stream_name(&stream);
        assert!(
            result.is_ok(),
            "plain logical stream names should pass enqueue validation"
        );
    }

    #[test]
    fn validate_enqueue_stream_name_rejects_whitespace_only_values() {
        let stream = LogicalStreamName::new("   ");
        let result = validate_enqueue_stream_name(&stream);
        assert!(
            matches!(result, Err(QueueError::Config(_))),
            "whitespace-only stream names must be rejected"
        );
    }

    #[test]
    fn validate_forward_many_outputs_rejects_empty_list() {
        let outputs: Vec<Output> = Vec::new();
        let result = validate_forward_many_outputs(&outputs);
        assert!(
            result.is_err(),
            "forward_many must reject empty outputs to avoid accidental source ack without forwarding"
        );
    }

    #[test]
    fn validate_forward_many_outputs_accepts_non_empty_list() {
        let outputs = vec![Output::new("stream:results", r#"{"status":"ok"}"#)];
        let result = validate_forward_many_outputs(&outputs);
        assert!(
            result.is_ok(),
            "forward_many should accept at least one output destination"
        );
    }

    #[test]
    fn validate_payload_json_rejects_invalid_json() {
        let result = validate_payload_json("not-json");
        assert!(
            matches!(result, Err(QueueError::Config(_))),
            "enqueue payload must be valid JSON"
        );
    }

    #[test]
    fn validate_payload_json_accepts_valid_json() {
        let result = validate_payload_json(r#"{"model":"pangu","steps":4}"#);
        assert!(result.is_ok(), "valid JSON payloads should pass validation");
    }

    #[test]
    fn validate_message_context_rejects_empty_critical_fields() {
        let cases = [
            Message::new("", "stream:test", "grp", "run-1", "{}", "stage"),
            Message::new("1-0", "", "grp", "run-1", "{}", "stage"),
            Message::new("1-0", "stream:test", "", "run-1", "{}", "stage"),
            Message::new("1-0", "stream:test", "grp", "", "{}", "stage"),
        ];
        for message in cases {
            let result = validate_message_context(&message);
            assert!(
                matches!(result, Err(QueueError::Config(_))),
                "message context must reject empty id/stream/group/run_id"
            );
        }
    }

    #[test]
    fn validate_message_context_accepts_valid_message() {
        let message = Message::new("1-0", "stream:test", "grp", "run-1", "{}", "stage");
        let result = validate_message_context(&message);
        assert!(
            result.is_ok(),
            "message context with non-empty id/stream/group/run_id should pass"
        );
    }

    #[test]
    fn validate_message_context_rejects_whitespace_run_id() {
        let message = Message::new("1-0", "stream:test", "grp", "   ", "{}", "stage");
        let result = validate_message_context(&message);
        assert!(
            matches!(result, Err(QueueError::Config(_))),
            "whitespace-only run_id should be treated as empty and rejected"
        );
    }

    #[test]
    fn ensure_groups_is_removed_from_operations_api_contract() {
        let source = include_str!("operations.rs");
        let removed_signature = ["pub async fn", "ensure_groups(&self) -> Result<()>"].join(" ");
        assert!(
            !source.contains(&removed_signature),
            "ensure_groups should be removed from QueueManager operations API"
        );
    }

    #[test]
    fn enqueue_to_stream_contract_is_explicitly_advanced_api() {
        let source = include_str!("operations.rs");
        assert!(
            source.contains("advanced API"),
            "enqueue_to_stream docs should mark the API as advanced"
        );
        assert!(
            source.contains("escape hatch"),
            "enqueue_to_stream docs should describe escape-hatch semantics"
        );
        assert!(
            source.contains("explicit Redis stream key"),
            "enqueue_to_stream docs should mention explicit stream-key behavior"
        );
    }
}
