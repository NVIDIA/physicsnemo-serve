/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result};
use scicomp_rq::{Message, Output, QueueManager, StreamKey, keys};

use crate::traits::{BoxFuture, MessageSink, QueueTransport};
use crate::transport::consumer_group_name;

/// Production transport backed by Redis Streams via `scicomp_rq::QueueManager`.
///
/// Translates between logical stream names (used by roles/engine) and physical
/// `StreamKey`s (prefixed Redis keys) used by the queue manager.
pub struct RedisTransport {
    qm: QueueManager,
    prefix: String,
    failure_attempts_key: String,
}

impl RedisTransport {
    pub fn new(qm: QueueManager, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        Self {
            qm,
            failure_attempts_key: format!("{}worker-runtime:failure-attempts", prefix),
            prefix,
        }
    }

    fn stream_key(&self, logical: &str) -> StreamKey {
        StreamKey::new(format!("{}{}", self.prefix, logical))
    }

    fn failure_attempt_field(msg: &Message) -> String {
        format!("{}::{}", msg.stream(), msg.id())
    }
}

impl MessageSink for RedisTransport {
    fn enqueue<'a>(
        &'a self,
        stream: &'a str,
        run_id: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let key = self.stream_key(stream);
            self.qm
                .enqueue_to_stream(&key, run_id, payload, stage)
                .await
                .map_err(Into::into)
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
            let key = StreamKey::new(stream_key);
            self.qm
                .enqueue_to_stream(&key, run_id, payload, stage)
                .await
                .map_err(Into::into)
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

    fn mark_request_failed<'a>(&'a self, run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let run_key = format!("{}{}", keys::RUN_HASH_PREFIX, run_id);
            let mut conn = self.qm.connection();
            let run_exists: bool = redis::cmd("EXISTS")
                .arg(&run_key)
                .query_async(&mut conn)
                .await
                .context("redis transport: failed to check DLQ run hash")?;
            if !run_exists {
                return Ok(());
            }

            let _: usize = redis::cmd("HSET")
                .arg(&run_key)
                .arg("status")
                .arg("failed")
                .query_async(&mut conn)
                .await
                .context("redis transport: failed to mark DLQ run failed")?;
            Ok(())
        })
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
            let dest_key = self.stream_key(dest_stream);
            self.qm
                .handoff_message_to_run(msg, &dest_key, Some(payload), Some(stage), Some(run_id))
                .await
                .map_err(Into::into)
        })
    }

    fn forward_many<'a>(
        &'a self,
        msg: &'a Message,
        outputs: &'a [Output],
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        Box::pin(async move {
            let prefixed: Vec<Output> = outputs
                .iter()
                .map(|o| {
                    let prefixed_stream = format!("{}{}", self.prefix, o.stream());
                    let mut out = Output::new(prefixed_stream, o.payload().to_string());
                    if let Some(run_id) = o.run_id() {
                        out = out.with_run_id(run_id.to_string());
                    }
                    if let Some(stage) = o.stage() {
                        out = out.with_stage(stage.to_string());
                    }
                    out
                })
                .collect();
            self.qm
                .forward_many(msg, &prefixed)
                .await
                .map_err(Into::into)
        })
    }
}

impl QueueTransport for RedisTransport {
    fn poll_stream<'a>(
        &'a self,
        stream: &'a str,
        consumer: &'a str,
        count: usize,
        block_ms: u64,
    ) -> BoxFuture<'a, Result<Vec<Message>>> {
        Box::pin(async move {
            let key = self.stream_key(stream);
            let group = consumer_group_name(stream);
            let block_ms_usize: usize = block_ms.try_into().unwrap_or(usize::MAX);
            self.qm
                .read_messages(&key, &group, consumer, count, block_ms_usize)
                .await
                .map_err(Into::into)
        })
    }

    fn ack<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.qm
                .ack_message(msg)
                .await
                .context("failed to ack message")?;
            Ok(())
        })
    }

    fn reclaim_idle<'a>(
        &'a self,
        stream: &'a str,
        consumer: &'a str,
        min_idle_ms: u64,
        count: usize,
    ) -> BoxFuture<'a, Result<Vec<Message>>> {
        Box::pin(async move {
            let key = self.stream_key(stream);
            let group = consumer_group_name(stream);
            let (_cursor, messages) = self
                .qm
                .claim_idle_messages(&key, &group, consumer, min_idle_ms, "0-0", count)
                .await?;
            Ok(messages)
        })
    }

    fn create_consumer_group<'a>(
        &'a self,
        stream: &'a str,
        group: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let key = self.stream_key(stream);
            self.qm
                .create_consumer_group(&key, group, "0", true)
                .await?;
            Ok(())
        })
    }

    fn increment_failure_attempt<'a>(
        &'a self,
        msg: &'a Message,
    ) -> BoxFuture<'a, Result<Option<usize>>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let field = Self::failure_attempt_field(msg);
            let attempts: i64 = redis::cmd("HINCRBY")
                .arg(&self.failure_attempts_key)
                .arg(&field)
                .arg(1)
                .query_async(&mut conn)
                .await
                .context("redis transport: failed to increment failure attempts")?;
            let _: bool = redis::cmd("EXPIRE")
                .arg(&self.failure_attempts_key)
                .arg(24 * 60 * 60)
                .query_async(&mut conn)
                .await
                .context("redis transport: failed to refresh failure attempts ttl")?;
            Ok(Some(attempts.max(0) as usize))
        })
    }

    fn clear_failure_attempt<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let field = Self::failure_attempt_field(msg);
            let _: i64 = redis::cmd("HDEL")
                .arg(&self.failure_attempts_key)
                .arg(field)
                .query_async(&mut conn)
                .await
                .context("redis transport: failed to clear failure attempt")?;
            Ok(())
        })
    }

    fn as_sink(&self) -> &dyn MessageSink {
        self
    }
}
