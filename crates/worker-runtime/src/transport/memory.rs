/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use scicomp_rq::{Message, Output};

use crate::traits::{BoxFuture, MessageSink, QueueTransport};
use crate::transport::consumer_group_name;

/// In-memory transport for testing.
///
/// All engines sharing the same `InMemoryTransport` instance see each other's
/// enqueued messages, enabling multi-worker pipeline tests without Redis.
///
/// Messages are stored in per-stream `Vec`s protected by a `Mutex`. Polling
/// drains the stream's buffer and returns all available messages.
pub struct InMemoryTransport {
    streams: Mutex<HashMap<String, Vec<Message>>>,
    /// Messages delivered via `poll_stream` but not yet acked.
    pending: Mutex<HashMap<String, Vec<Message>>>,
    acked: Mutex<Vec<String>>,
    msg_counter: Mutex<u64>,
    failure_attempts: Mutex<HashMap<String, usize>>,
    prefix: String,
}

impl InMemoryTransport {
    pub fn new(stream_names: &[&str], prefix: &str) -> Self {
        let mut streams = HashMap::new();
        for name in stream_names {
            streams.insert(name.to_string(), Vec::new());
        }
        Self {
            streams: Mutex::new(streams),
            pending: Mutex::new(HashMap::new()),
            acked: Mutex::new(Vec::new()),
            msg_counter: Mutex::new(0),
            failure_attempts: Mutex::new(HashMap::new()),
            prefix: prefix.to_string(),
        }
    }

    /// Inject a message into a logical stream (for test setup).
    pub fn inject(&self, stream: &str, run_id: &str, payload: &str, stage: &str) -> Result<()> {
        let msg = self.build_message(stream, run_id, payload, stage)?;
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;
        let queue = streams
            .get_mut(stream)
            .ok_or_else(|| anyhow!("stream '{}' not registered", stream))?;
        queue.push(msg);
        Ok(())
    }

    /// Read all acked message IDs (for test assertions).
    pub fn acked_ids(&self) -> Vec<String> {
        self.acked.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Read all messages currently pending in a stream (non-destructive, for assertions).
    pub fn pending_in(&self, stream: &str) -> Vec<Message> {
        self.streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(stream)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if all streams are empty (for `run_until_idle` style tests).
    pub fn all_drained(&self) -> bool {
        self.streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .all(|q| q.is_empty())
    }

    fn build_message(
        &self,
        stream: &str,
        run_id: &str,
        payload: &str,
        stage: &str,
    ) -> Result<Message> {
        let mut counter = self
            .msg_counter
            .lock()
            .map_err(|e| anyhow!("counter lock poisoned: {e}"))?;
        *counter += 1;
        let id = format!("{}-0", *counter);
        let physical_stream = format!("{}{}", self.prefix, stream);
        let group = consumer_group_name(stream);
        Ok(Message::new(
            &id,
            &physical_stream,
            &group,
            run_id,
            payload,
            stage,
        ))
    }

    fn physical_to_logical(&self, physical: &str) -> String {
        physical
            .strip_prefix(&self.prefix)
            .unwrap_or(physical)
            .to_string()
    }

    fn failure_attempt_key(msg: &Message) -> String {
        format!("{}::{}", msg.stream(), msg.id())
    }
}

impl MessageSink for InMemoryTransport {
    fn enqueue<'a>(
        &'a self,
        stream: &'a str,
        run_id: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let msg = self.build_message(stream, run_id, payload, stage)?;
            let msg_id = msg.id().to_string();
            let mut streams = self
                .streams
                .lock()
                .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;
            let queue = streams.entry(stream.to_string()).or_default();
            queue.push(msg);
            Ok(msg_id)
        })
    }

    fn enqueue_to_stream<'a>(
        &'a self,
        stream_key: &'a str,
        run_id: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let logical_stream = self.physical_to_logical(stream_key);
            let msg = self.build_message(&logical_stream, run_id, payload, stage)?;
            let msg_id = msg.id().to_string();
            let mut streams = self
                .streams
                .lock()
                .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;
            let queue = streams.entry(logical_stream).or_default();
            queue.push(msg);
            Ok(msg_id)
        })
    }

    fn ack_message<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        <Self as QueueTransport>::ack(self, msg)
    }

    fn handoff<'a>(
        &'a self,
        msg: &'a Message,
        dest_stream: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        self.handoff_to_run(msg, dest_stream, payload, stage, msg.run_id())
    }

    fn handoff_to_run<'a>(
        &'a self,
        msg: &'a Message,
        dest_stream: &'a str,
        payload: &'a str,
        stage: &'a str,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            self.acked
                .lock()
                .map_err(|e| anyhow!("acked lock poisoned: {e}"))?
                .push(msg.id().to_string());

            let logical = self.physical_to_logical(msg.stream());

            {
                let mut pending = self
                    .pending
                    .lock()
                    .map_err(|e| anyhow!("pending lock poisoned: {e}"))?;
                if let Some(queue) = pending.get_mut(&logical) {
                    queue.retain(|m| m.id() != msg.id());
                }
            }
            let new_msg = self.build_message(dest_stream, run_id, payload, stage)?;
            let msg_id = new_msg.id().to_string();

            let mut streams = self
                .streams
                .lock()
                .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;
            if let Some(src_queue) = streams.get_mut(&logical) {
                src_queue.retain(|m| m.id() != msg.id());
            }
            streams
                .entry(dest_stream.to_string())
                .or_default()
                .push(new_msg);

            Ok(msg_id)
        })
    }

    fn forward_many<'a>(
        &'a self,
        msg: &'a Message,
        outputs: &'a [Output],
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        Box::pin(async move {
            self.acked
                .lock()
                .map_err(|e| anyhow!("acked lock poisoned: {e}"))?
                .push(msg.id().to_string());

            let logical = self.physical_to_logical(msg.stream());

            {
                let mut pending = self
                    .pending
                    .lock()
                    .map_err(|e| anyhow!("pending lock poisoned: {e}"))?;
                if let Some(queue) = pending.get_mut(&logical) {
                    queue.retain(|m| m.id() != msg.id());
                }
            }

            let new_messages: Vec<(String, Message)> = {
                let mut counter = self
                    .msg_counter
                    .lock()
                    .map_err(|e| anyhow!("counter lock poisoned: {e}"))?;
                outputs
                    .iter()
                    .map(|output| {
                        *counter += 1;
                        let id = format!("{}-0", *counter);
                        let physical = format!("{}{}", self.prefix, output.stream());
                        let group = consumer_group_name(output.stream());
                        let stage = output.stage().unwrap_or(msg.stage());
                        let run_id = output.run_id().unwrap_or(msg.run_id());
                        let new_msg =
                            Message::new(&id, &physical, &group, run_id, output.payload(), stage);
                        (output.stream().to_string(), new_msg)
                    })
                    .collect()
            };

            let mut ids = Vec::with_capacity(new_messages.len());
            let mut streams = self
                .streams
                .lock()
                .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;

            if let Some(src_queue) = streams.get_mut(&logical) {
                src_queue.retain(|m| m.id() != msg.id());
            }
            for (dest, new_msg) in new_messages {
                ids.push(new_msg.id().to_string());
                streams.entry(dest).or_default().push(new_msg);
            }

            Ok(ids)
        })
    }
}

impl QueueTransport for InMemoryTransport {
    fn poll_stream<'a>(
        &'a self,
        stream: &'a str,
        _consumer: &'a str,
        count: usize,
        _block_ms: u64,
    ) -> BoxFuture<'a, Result<Vec<Message>>> {
        Box::pin(async move {
            let messages = {
                let mut streams = self
                    .streams
                    .lock()
                    .map_err(|e| anyhow!("streams lock poisoned: {e}"))?;
                let queue = streams.get_mut(stream).unwrap_or(&mut Vec::new()).to_vec();
                if queue.is_empty() {
                    return Ok(Vec::new());
                }
                let take = count.min(queue.len());
                let messages: Vec<Message> = queue[..take].to_vec();
                if let Some(q) = streams.get_mut(stream) {
                    q.drain(..take);
                }
                messages
            };

            let mut pending = self
                .pending
                .lock()
                .map_err(|e| anyhow!("pending lock poisoned: {e}"))?;
            pending
                .entry(stream.to_string())
                .or_default()
                .extend(messages.clone());

            Ok(messages)
        })
    }

    fn ack<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.acked
                .lock()
                .map_err(|e| anyhow!("acked lock poisoned: {e}"))?
                .push(msg.id().to_string());

            let logical = self.physical_to_logical(msg.stream());
            let mut pending = self
                .pending
                .lock()
                .map_err(|e| anyhow!("pending lock poisoned: {e}"))?;
            if let Some(queue) = pending.get_mut(&logical) {
                queue.retain(|m| m.id() != msg.id());
            }

            Ok(())
        })
    }

    fn reclaim_idle<'a>(
        &'a self,
        stream: &'a str,
        _consumer: &'a str,
        _min_idle_ms: u64,
        count: usize,
    ) -> BoxFuture<'a, Result<Vec<Message>>> {
        Box::pin(async move {
            let pending = self
                .pending
                .lock()
                .map_err(|e| anyhow!("pending lock poisoned: {e}"))?;
            let messages = pending
                .get(stream)
                .map(|q| q.iter().take(count).cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            Ok(messages)
        })
    }

    fn create_consumer_group<'a>(
        &'a self,
        _stream: &'a str,
        _group: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(()) })
    }

    fn increment_failure_attempt<'a>(
        &'a self,
        msg: &'a Message,
    ) -> BoxFuture<'a, Result<Option<usize>>> {
        Box::pin(async move {
            let key = Self::failure_attempt_key(msg);
            let mut attempts = self
                .failure_attempts
                .lock()
                .map_err(|e| anyhow!("failure attempts lock poisoned: {e}"))?;
            let entry = attempts.entry(key).or_insert(0);
            *entry = entry.saturating_add(1);
            Ok(Some(*entry))
        })
    }

    fn clear_failure_attempt<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let key = Self::failure_attempt_key(msg);
            let mut attempts = self
                .failure_attempts
                .lock()
                .map_err(|e| anyhow!("failure attempts lock poisoned: {e}"))?;
            attempts.remove(&key);
            Ok(())
        })
    }

    fn as_sink(&self) -> &dyn MessageSink {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_and_poll_roundtrip() {
        let transport = InMemoryTransport::new(&["prefetch"], "test:");
        transport
            .inject("prefetch", "run-1", r#"{"ok":true}"#, "prefetch")
            .expect("inject should succeed");

        let pending = transport.pending_in("prefetch");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].run_id(), "run-1");
    }

    #[tokio::test]
    async fn enqueue_makes_message_available_for_poll() {
        let transport = InMemoryTransport::new(&["schedule"], "test:");
        let sink: &dyn MessageSink = &transport;

        sink.enqueue("schedule", "run-1", r#"{"job":"x"}"#, "schedule")
            .await
            .expect("enqueue should succeed");

        let msgs = transport
            .poll_stream("schedule", "consumer-1", 10, 0)
            .await
            .expect("poll should succeed");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].run_id(), "run-1");
    }

    #[tokio::test]
    async fn poll_drains_messages() {
        let transport = InMemoryTransport::new(&["results"], "test:");
        transport
            .inject("results", "run-1", "{}", "results")
            .unwrap();
        transport
            .inject("results", "run-2", "{}", "results")
            .unwrap();

        let msgs = transport.poll_stream("results", "c", 10, 0).await.unwrap();
        assert_eq!(msgs.len(), 2);

        let msgs2 = transport.poll_stream("results", "c", 10, 0).await.unwrap();
        assert!(
            msgs2.is_empty(),
            "second poll should return empty after drain"
        );
    }

    #[tokio::test]
    async fn ack_records_message_id() {
        let transport = InMemoryTransport::new(&["prefetch"], "test:");
        transport
            .inject("prefetch", "run-1", "{}", "prefetch")
            .unwrap();
        let msgs = transport.poll_stream("prefetch", "c", 1, 0).await.unwrap();

        transport.ack(&msgs[0]).await.unwrap();
        assert_eq!(transport.acked_ids(), vec![msgs[0].id().to_string()]);
    }

    #[tokio::test]
    async fn handoff_acks_source_and_enqueues_dest() {
        let transport = InMemoryTransport::new(&["gpu_0", "results"], "test:");
        transport
            .inject("gpu_0", "run-1", r#"{"data":"x"}"#, "inference")
            .unwrap();
        let msgs = transport.poll_stream("gpu_0", "c", 1, 0).await.unwrap();
        let sink: &dyn MessageSink = &transport;

        let new_id = sink
            .handoff(&msgs[0], "results", r#"{"result":"y"}"#, "results")
            .await
            .unwrap();

        assert!(!new_id.is_empty());
        assert!(
            transport.acked_ids().contains(&msgs[0].id().to_string()),
            "handoff should ack original message"
        );
        let results_pending = transport.pending_in("results");
        assert_eq!(results_pending.len(), 1);
        assert_eq!(results_pending[0].run_id(), "run-1");
    }

    #[tokio::test]
    async fn handoff_to_run_overrides_destination_run_id_when_present() {
        let transport = InMemoryTransport::new(&["collect", "postprocess"], "test:");
        transport
            .inject("collect", "child-run-1", r#"{"data":"x"}"#, "collect")
            .unwrap();
        let msgs = transport.poll_stream("collect", "c", 1, 0).await.unwrap();
        let sink: &dyn MessageSink = &transport;

        sink.handoff_to_run(
            &msgs[0],
            "postprocess",
            r#"{"result":"y"}"#,
            "postprocess",
            "parent-run",
        )
        .await
        .unwrap();

        let pending = transport.pending_in("postprocess");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].run_id(), "parent-run");
    }

    #[test]
    fn all_drained_reports_correctly() {
        let transport = InMemoryTransport::new(&["a", "b"], "");
        assert!(transport.all_drained());

        transport.inject("a", "run-1", "{}", "a").unwrap();
        assert!(!transport.all_drained());
    }

    // --- PR-005: reclaim idle messages ---

    #[tokio::test]
    async fn reclaim_idle_returns_unacked_messages() {
        let transport = InMemoryTransport::new(&["work"], "test:");
        transport.inject("work", "run-1", "{}", "work").unwrap();
        transport.inject("work", "run-2", "{}", "work").unwrap();

        // Poll both messages
        let msgs = transport.poll_stream("work", "c1", 10, 0).await.unwrap();
        assert_eq!(msgs.len(), 2);

        // Ack only the first
        transport.ack(&msgs[0]).await.unwrap();

        // Reclaim with min_idle_ms=0 should return the un-acked message
        let reclaimed = transport.reclaim_idle("work", "c1", 0, 10).await.unwrap();
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0].run_id(), "run-2");
    }

    #[tokio::test]
    async fn reclaim_idle_returns_empty_after_all_acked() {
        let transport = InMemoryTransport::new(&["work"], "test:");
        transport.inject("work", "run-1", "{}", "work").unwrap();
        let msgs = transport.poll_stream("work", "c1", 10, 0).await.unwrap();
        transport.ack(&msgs[0]).await.unwrap();

        let reclaimed = transport.reclaim_idle("work", "c1", 0, 10).await.unwrap();
        assert!(reclaimed.is_empty());
    }

    #[tokio::test]
    async fn forward_many_fans_out_to_multiple_streams() {
        let transport = InMemoryTransport::new(&["src", "dst1", "dst2"], "");
        transport.inject("src", "run-1", "{}", "src").unwrap();
        let msgs = transport.poll_stream("src", "c", 1, 0).await.unwrap();
        let sink: &dyn MessageSink = &transport;

        let outputs = vec![
            Output::new("dst1", r#"{"a":1}"#),
            Output::new("dst2", r#"{"b":2}"#),
        ];
        let ids = sink.forward_many(&msgs[0], &outputs).await.unwrap();
        assert_eq!(ids.len(), 2);

        assert_eq!(transport.pending_in("dst1").len(), 1);
        assert_eq!(transport.pending_in("dst2").len(), 1);
    }

    #[tokio::test]
    async fn forward_many_uses_output_run_id_override_when_present() {
        let transport = InMemoryTransport::new(&["src", "dst"], "");
        transport.inject("src", "parent-run", "{}", "src").unwrap();
        let msgs = transport.poll_stream("src", "c", 1, 0).await.unwrap();
        let sink: &dyn MessageSink = &transport;

        let outputs = vec![Output::new("dst", r#"{"ok":true}"#).with_run_id("child-run-1")];
        sink.forward_many(&msgs[0], &outputs).await.unwrap();

        let forwarded = transport.pending_in("dst");
        assert_eq!(forwarded.len(), 1);
        assert_eq!(forwarded[0].run_id(), "child-run-1");
    }
}
