/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, anyhow};
use redis::Script;
use scicomp_rq::{Message, Output, QueueManager, StreamKey, keys};

use crate::traits::{BoxFuture, MessageSink, QueueTransport, message_ownership_lost};
use crate::transport::consumer_group_name;

const FINALIZATION_CLAIM_ROLLBACK_TTL_SECS: u64 = 600;
const FINALIZATION_COMMITTED_TTL_SECS: u64 = 24 * 60 * 60;
const SOURCE_MESSAGE_NOT_PENDING: &str = "SOURCE_MESSAGE_NOT_PENDING";
const SOURCE_MESSAGE_NOT_OWNED_BY_CONSUMER: &str = "SOURCE_MESSAGE_NOT_OWNED_BY_CONSUMER";

const MARK_REQUEST_FAILED_LUA: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end

local output_location = redis.call('HGET', KEYS[1], 'output_location')
local publication_status = redis.call('HGET', KEYS[1], 'output_publication_status')
if output_location == 'local_and_cloud' and not publication_status then
  local now = redis.call('TIME')[1]
  redis.call(
    'HSET', KEYS[1],
    'status', 'failed',
    'output_publication_status', 'skipped',
    'publish_completed_at', tostring(now),
    'published_artifact_count', '0'
  )
  redis.call('HDEL', KEYS[1], 'publish_error')
elseif output_location == 'local_and_cloud' and publication_status == 'uploading' then
  local now = redis.call('TIME')[1]
  redis.call(
    'HSET', KEYS[1],
    'status', 'failed',
    'output_publication_status', 'failed',
    'publish_completed_at', tostring(now),
    'published_artifact_count', '0'
  )
else
  redis.call('HSET', KEYS[1], 'status', 'failed')
end
return 1
"#;

const HANDOFF_AND_COMMIT_FINALIZATION_LUA: &str = r#"
local function key_type(key)
  local result = redis.call('TYPE', key)
  if type(result) == 'table' then
    return result['ok']
  end
  return result
end

if redis.call('GET', KEYS[4]) ~= ARGV[6] then
  return redis.error_reply('FINALIZATION_OWNER_MISMATCH')
end
local pending = redis.call('XPENDING', KEYS[2], ARGV[3], ARGV[4], ARGV[4], 1)
if #pending ~= 1 or pending[1][1] ~= ARGV[4] then
  return redis.error_reply('SOURCE_MESSAGE_NOT_PENDING')
end
if pending[1][2] ~= ARGV[8] then
  return redis.error_reply('SOURCE_MESSAGE_NOT_OWNED_BY_CONSUMER')
end
local destination_type = key_type(KEYS[1])
if destination_type ~= 'none' and destination_type ~= 'stream' then
  return redis.error_reply('DESTINATION_WRONG_TYPE:' .. destination_type)
end
local run_hash_type = key_type(KEYS[3])
if run_hash_type ~= 'none' and run_hash_type ~= 'hash' then
  return redis.error_reply('RUN_HASH_WRONG_TYPE:' .. run_hash_type)
end

local now = redis.call('TIME')[1]
local next_id_result = redis.pcall(
  'XADD', KEYS[1], '*',
  'run_id', ARGV[1],
  'payload', ARGV[2],
  'stage', ARGV[5]
)
if type(next_id_result) == 'table' and next_id_result['err'] then
  return redis.error_reply('XADD_FAILED:' .. tostring(next_id_result['err']))
end
local next_id = next_id_result
local hset_result = redis.pcall(
  'HSET', KEYS[3],
  'stage', ARGV[5],
  'updated_at', tostring(now),
  ARGV[5] .. '_enqueued_at', tostring(now)
)
if type(hset_result) == 'table' and hset_result['err'] then
  redis.pcall('XDEL', KEYS[1], next_id)
  return redis.error_reply('HSET_FAILED:' .. tostring(hset_result['err']))
end
local commit_result = redis.pcall(
  'SET', KEYS[4], 'committed:' .. ARGV[6], 'EX', ARGV[9]
)
if type(commit_result) == 'table' and commit_result['err'] then
  redis.pcall('XDEL', KEYS[1], next_id)
  return redis.error_reply('FINALIZATION_COMMIT_FAILED:' .. tostring(commit_result['err']))
end
local acked = redis.call('XACK', KEYS[2], ARGV[3], ARGV[4])
if acked ~= 1 then
  redis.pcall('XDEL', KEYS[1], next_id)
  redis.pcall('SET', KEYS[4], ARGV[6], 'EX', ARGV[7])
  return redis.error_reply('XACK_FAILED:' .. tostring(acked))
end
for key_index = 5, #KEYS do
  redis.call('EXPIRE', KEYS[key_index], ARGV[9])
end
return next_id
"#;

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

    fn is_source_ownership_error(error: &redis::RedisError) -> bool {
        [
            SOURCE_MESSAGE_NOT_PENDING,
            SOURCE_MESSAGE_NOT_OWNED_BY_CONSUMER,
        ]
        .iter()
        .any(|code| {
            error.code() == Some(*code)
                || error.detail().is_some_and(|detail| detail.contains(code))
                || error.to_string().contains(code)
        })
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
            let _: i64 = Script::new(MARK_REQUEST_FAILED_LUA)
                .key(run_key)
                .invoke_async(&mut conn)
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

    #[allow(clippy::too_many_arguments)]
    fn handoff_to_run_and_commit<'a>(
        &'a self,
        _msg: &'a Message,
        _dest_stream: &'a str,
        _payload: &'a str,
        _stage: &'a str,
        _run_id: &'a str,
        _finalization_key: &'a str,
        _owner_token: &'a str,
        _recovery_keys: &'a [String],
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async {
            Err(anyhow!(
                "redis transport: guarded handoff requires an expected consumer"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn handoff_to_run_and_commit_for_consumer<'a>(
        &'a self,
        msg: &'a Message,
        dest_stream: &'a str,
        payload: &'a str,
        stage: &'a str,
        run_id: &'a str,
        finalization_key: &'a str,
        owner_token: &'a str,
        recovery_keys: &'a [String],
        consumer: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let destination_key = self.stream_key(dest_stream);
            let run_key = format!("{}{}", keys::RUN_HASH_PREFIX, run_id);
            let mut conn = self.qm.connection();
            let script = Script::new(HANDOFF_AND_COMMIT_FINALIZATION_LUA);
            let mut invocation = script.prepare_invoke();
            invocation
                .key(destination_key.as_str())
                .key(msg.stream())
                .key(run_key)
                .key(finalization_key);
            for key in recovery_keys {
                invocation.key(key);
            }
            let result: redis::RedisResult<String> = invocation
                .arg(run_id)
                .arg(payload)
                .arg(msg.group())
                .arg(msg.id())
                .arg(stage)
                .arg(owner_token)
                .arg(FINALIZATION_CLAIM_ROLLBACK_TTL_SECS)
                .arg(consumer)
                .arg(FINALIZATION_COMMITTED_TTL_SECS)
                .invoke_async(&mut conn)
                .await;
            match result {
                Ok(next_id) => Ok(next_id),
                Err(error) if Self::is_source_ownership_error(&error) => {
                    Err(message_ownership_lost(error.to_string()))
                }
                Err(error) => {
                    Err(error).context("redis transport: failed guarded finalization handoff")
                }
            }
        })
    }

    fn forward_many<'a>(
        &'a self,
        msg: &'a Message,
        outputs: &'a [Output],
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        self.forward_many_from(std::slice::from_ref(msg), outputs)
    }

    fn forward_many_from<'a>(
        &'a self,
        msgs: &'a [Message],
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
                .forward_many_from(msgs, &prefixed)
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

    fn renew_message_lease<'a>(
        &'a self,
        msg: &'a Message,
        consumer: &'a str,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            self.qm
                .renew_message_lease(msg, consumer)
                .await
                .map_err(Into::into)
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
