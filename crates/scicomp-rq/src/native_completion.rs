/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owner-fenced, retryable completion for a single native execution delivery.

use crate::{Message, Output, QueueError, QueueManager, Result, keys};

// One state hash is retained without a TTL. Expiring a committed marker while
// its source identity can still be retried would permit duplicate release. Any
// operational retention policy must therefore outlive the delivery/retry horizon.
const ATTEMPT_LUA: &str = r#"
local function key_type(key)
  local value = redis.call('TYPE', key)
  if type(value) == 'table' then return value['ok'] end
  return value
end
local mode, group, id, consumer, token = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local count = tonumber(ARGV[6])
if (mode ~= 'begin' and mode ~= 'complete') or not count or count < 0 or count > 64 or
   count ~= math.floor(count) or #KEYS ~= 2 + 2 * count or #ARGV ~= 6 + 3 * count or
   group == '' or id == '' or consumer == '' or token == '' or
   (mode == 'begin' and count ~= 0) or (mode == 'complete' and count == 0) then
  return redis.error_reply('INVALID_ATTEMPT_ARGUMENTS')
end
local state_type = key_type(KEYS[2])
if state_type ~= 'none' and state_type ~= 'hash' then
  return redis.error_reply('ATTEMPT_STATE_WRONG_TYPE:' .. state_type)
end
-- This branch is deliberately before ownership checks: a successful commit has
-- already removed the source from the PEL, and its reply may have been lost.
if redis.call('HGET', KEYS[2], 'state') == 'committed' then
  return { 'committed', redis.call('HGET', KEYS[2], 'ids') }
end
if key_type(KEYS[1]) ~= 'stream' then
  return redis.error_reply('SOURCE_MESSAGE_NOT_PENDING')
end
local pending = redis.call('XPENDING', KEYS[1], group, id, id, 1)
if #pending ~= 1 or pending[1][1] ~= id then
  return redis.error_reply('SOURCE_MESSAGE_NOT_PENDING')
end
if pending[1][2] ~= consumer then
  return redis.error_reply('SOURCE_MESSAGE_NOT_OWNED_BY_CONSUMER')
end
local previous_consumer = redis.call('HGET', KEYS[2], 'consumer')
local previous_token = redis.call('HGET', KEYS[2], 'token')
if mode == 'begin' then
  if previous_consumer == consumer and previous_token ~= token then
    return redis.error_reply('ATTEMPT_OWNER_MISMATCH')
  end
  redis.call('HSET', KEYS[2], 'state', 'active', 'consumer', consumer, 'token', token)
  return { 'active', '' }
end
if previous_consumer ~= consumer or previous_token ~= token then
  return redis.error_reply('ATTEMPT_OWNER_MISMATCH')
end
-- Lua does not roll back runtime errors. Validate every key and every argument,
-- including run hashes for later outputs, before the first XADD.
for i = 1, count do
  local destination = KEYS[2 + i]
  local run_hash = KEYS[2 + count + i]
  local destination_type, run_type = key_type(destination), key_type(run_hash)
  if destination_type ~= 'none' and destination_type ~= 'stream' then
    return redis.error_reply('DESTINATION_WRONG_TYPE:' .. destination_type)
  end
  if run_type ~= 'none' and run_type ~= 'hash' then
    return redis.error_reply('RUN_HASH_WRONG_TYPE:' .. run_type)
  end
  if destination == KEYS[1] or destination == KEYS[2] or run_hash == KEYS[1] or run_hash == KEYS[2] then
    return redis.error_reply('ATTEMPT_KEY_ALIAS')
  end
  for j = 1, count do
    if destination == KEYS[2 + count + j] then return redis.error_reply('ATTEMPT_KEY_ALIAS') end
  end
  local offset = 7 + (i - 1) * 3
  if ARGV[offset] == '' or ARGV[offset + 2] == '' then
    return redis.error_reply('INVALID_OUTPUT_METADATA')
  end
  local valid_json = pcall(cjson.decode, ARGV[offset + 1])
  if not valid_json then return redis.error_reply('INVALID_OUTPUT_JSON') end
end
local ids = {}
local now = redis.call('TIME')[1]
for i = 1, count do
  local offset = 7 + (i - 1) * 3
  ids[i] = redis.call('XADD', KEYS[2 + i], '*',
    'run_id', ARGV[offset], 'payload', ARGV[offset + 1], 'stage', ARGV[offset + 2])
  redis.call('HSET', KEYS[2 + count + i], 'stage', ARGV[offset + 2],
    'updated_at', now, ARGV[offset + 2] .. '_enqueued_at', now)
end
redis.call('XACK', KEYS[1], group, id)
local encoded_ids = cjson.encode(ids)
redis.call('HSET', KEYS[2], 'state', 'committed', 'ids', encoded_ids)
return { 'committed', encoded_ids }
"#;

fn state_key(message: &Message) -> String {
    // Length-prefixed components avoid collisions even if a stream or group
    // contains separators. Source identity, rather than a process token, dedupes.
    format!(
        "native-attempt:{}:{}:{}:{}:{}:{}",
        message.stream().len(),
        message.stream(),
        message.group().len(),
        message.group(),
        message.id().len(),
        message.id()
    )
}

fn validate_identifier(value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 4096 {
        return Err(QueueError::Config(
            "attempt identifiers must contain 1..4096 bytes".into(),
        ));
    }
    Ok(())
}

impl QueueManager {
    /// Validate deterministic source metadata before a native role claims work.
    /// Invalid messages must be quarantined rather than classified as uncertain
    /// infrastructure operations after an attempt has started.
    pub fn validate_attempt_message(message: &Message) -> Result<()> {
        for value in [
            message.stream(),
            message.group(),
            message.id(),
            message.run_id(),
        ] {
            validate_identifier(value)?;
        }
        Ok(())
    }

    /// Claim an execution attempt, or recover the output IDs of an already
    /// committed delivery. The consumer identity must be unique to this worker.
    pub async fn begin_attempt(
        &self,
        message: &Message,
        consumer: &str,
        attempt_token: &str,
    ) -> Result<Option<Vec<String>>> {
        self.attempt_operation(message, &[], consumer, attempt_token, "begin")
            .await
    }

    /// Atomically emit every output, acknowledge the source, and persist a
    /// completion marker. Include the original reservation release in outputs.
    pub async fn complete_attempt(
        &self,
        message: &Message,
        outputs: &[Output],
        consumer: &str,
        attempt_token: &str,
    ) -> Result<Vec<String>> {
        if outputs.is_empty() {
            return Err(QueueError::Config(
                "native completion requires outputs".into(),
            ));
        }
        self.attempt_operation(message, outputs, consumer, attempt_token, "complete")
            .await?
            .ok_or_else(|| QueueError::Script("completion returned an active attempt".into()))
    }

    async fn attempt_operation(
        &self,
        message: &Message,
        outputs: &[Output],
        consumer: &str,
        attempt_token: &str,
        mode: &str,
    ) -> Result<Option<Vec<String>>> {
        Self::validate_attempt_message(message)?;
        for value in [consumer, attempt_token] {
            validate_identifier(value)?;
        }
        if outputs.len() > 64 {
            return Err(QueueError::Config(
                "native completion accepts at most 64 outputs".into(),
            ));
        }
        let mut resolved = Vec::with_capacity(outputs.len());
        for output in outputs {
            let run_id = output.run_id().unwrap_or(message.run_id());
            let stage = output.stage().unwrap_or(output.stream());
            if [output.stream(), run_id, stage]
                .iter()
                .any(|value| value.trim().is_empty() || value.len() > 4096)
            {
                return Err(QueueError::Config(
                    "output metadata must contain 1..4096 bytes".into(),
                ));
            }
            serde_json::from_str::<serde_json::Value>(output.payload())
                .map_err(|error| QueueError::Config(format!("invalid output JSON: {error}")))?;
            resolved.push((output, run_id, stage));
        }
        let script = redis::Script::new(ATTEMPT_LUA);
        let mut invoke = script.prepare_invoke();
        invoke.key(message.stream()).key(state_key(message));
        for (output, _, _) in &resolved {
            invoke.key(output.stream());
        }
        for (_, run_id, _) in &resolved {
            invoke.key(format!("{}{run_id}", keys::RUN_HASH_PREFIX));
        }
        invoke
            .arg(mode)
            .arg(message.group())
            .arg(message.id())
            .arg(consumer)
            .arg(attempt_token)
            .arg(outputs.len());
        for (output, run_id, stage) in resolved {
            invoke.arg(run_id).arg(output.payload()).arg(stage);
        }
        let mut conn = self.connection();
        let (state, ids): (String, String) = invoke.invoke_async(&mut conn).await?;
        match state.as_str() {
            "active" => Ok(None),
            "committed" => serde_json::from_str(&ids).map(Some).map_err(|error| {
                QueueError::Script(format!("invalid committed output IDs: {error}"))
            }),
            _ => Err(QueueError::Script(format!(
                "unknown attempt state: {state}"
            ))),
        }
    }
}
