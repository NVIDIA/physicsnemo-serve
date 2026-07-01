/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lua scripts and Redis error classification helpers.

use crate::error::QueueError;

/// Outcome of a single consumer group creation attempt.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum GroupCreateOutcome {
    /// Group was created successfully.
    Created,
    /// Group already exists (BUSYGROUP) -- safe to ignore.
    AlreadyExists,
    /// A real error occurred.
    Failed(redis::RedisError),
}

/// Return true when Redis error indicates missing cached Lua script.
#[inline]
pub(crate) fn is_noscript_error(err: &redis::RedisError) -> bool {
    err.to_string().contains("NOSCRIPT")
}

/// Returns `true` when the error indicates XAUTOCLAIM is not supported by the
/// Redis server (older than 6.2). Used to fall back to XPENDING+XCLAIM.
#[inline]
pub(crate) fn is_xautoclaim_unsupported(err: &QueueError) -> bool {
    let msg = err.to_string();
    msg.contains("unknown command") && msg.contains("XAUTOCLAIM")
}

/// Classify the result of an XGROUP CREATE command.
#[allow(dead_code)]
pub(crate) fn classify_group_creation(
    result: std::result::Result<redis::Value, redis::RedisError>,
) -> GroupCreateOutcome {
    match result {
        Ok(_) => GroupCreateOutcome::Created,
        Err(e) if e.to_string().contains("BUSYGROUP") => GroupCreateOutcome::AlreadyExists,
        Err(e) => GroupCreateOutcome::Failed(e),
    }
}

/// Derive stage name from a stream key with prefix awareness.
///
/// Rules:
/// 1. If `stream_key` starts with `prefix` and the remaining suffix is non-empty,
///    return that suffix.
/// 2. Otherwise, return the full `stream_key` unchanged.
///
/// # Examples
///
/// - `("stream:results", "stream:")` -> `"results"`
/// - `("physicsnemo:gpu:default:pod-0:0", "physicsnemo:")` -> `"gpu:default:pod-0:0"`
/// - `("external:inference", "stream:")` -> `"external:inference"`
#[inline]
pub(crate) fn derive_stage_from_stream(stream_key: &str, prefix: &str) -> String {
    if let Some(suffix) = stream_key.strip_prefix(prefix)
        && !suffix.is_empty()
    {
        return suffix.to_string();
    }
    stream_key.to_string()
}

/// Lua script for atomic handoff: XADD to next stream, XACK current, update run hash stage.
///
/// # Keys
/// - `KEYS[1]`: next_stream
/// - `KEYS[2]`: current_stream
/// - `KEYS[3]`: run_hash (run:{run_id})
///
/// # Arguments
/// - `ARGV[1]`: run_id
/// - `ARGV[2]`: payload_json
/// - `ARGV[3]`: group
/// - `ARGV[4]`: current_msg_id
/// - `ARGV[5]`: next_stage
pub const LUA_HANDOFF: &str = r#"
-- KEYS[1]=next_stream, KEYS[2]=current_stream, KEYS[3]=run_hash
-- ARGV[1]=run_id, ARGV[2]=payload_json, ARGV[3]=group, ARGV[4]=current_msg_id, ARGV[5]=next_stage
local now = redis.call('TIME')[1]
local next_id = redis.call('XADD', KEYS[1], '*', 'run_id', ARGV[1], 'payload', ARGV[2], 'stage', ARGV[5])
local acked = redis.call('XACK', KEYS[2], ARGV[3], ARGV[4])
if acked ~= 1 then
  -- Best-effort rollback to avoid leaking handoff entry when source ack fails.
  redis.pcall('XDEL', KEYS[1], next_id)
  return redis.error_reply('XACK_FAILED:' .. tostring(acked))
end
-- Persist current stage and per-stage enqueue timestamp; surface write failures explicitly.
local hset_res = redis.pcall('HSET', KEYS[3], 'stage', ARGV[5], 'updated_at', tostring(now), ARGV[5] .. '_enqueued_at', tostring(now))
if type(hset_res) == 'table' and hset_res['err'] then
  return redis.error_reply('HSET_FAILED:' .. tostring(hset_res['err']))
end
return next_id
"#;

/// Lua script for atomic forward_many fan-out.
///
/// # Keys
/// - `KEYS[1..N]`: destination streams
/// - `KEYS[N+1]`: source stream
/// - `KEYS[N+2]`: run hash (run:{run_id})
///
/// # Arguments
/// - `ARGV[1]`: run_id
/// - `ARGV[2]`: source_group
/// - `ARGV[3]`: source_msg_id
/// - `ARGV[4]`: output_count
/// - `ARGV[5..]`: pairs of `(payload_i, stage_i)` for each output
pub const LUA_FORWARD_MANY: &str = r#"
-- KEYS[1..N]=dest_streams, KEYS[N+1]=source_stream
-- ARGV[1]=source_group, ARGV[2]=source_msg_id, ARGV[3]=output_count, ARGV[4]=run_hash_prefix,
-- ARGV[5..]=run_id/payload/stage triples
local source_group = ARGV[1]
local source_msg_id = ARGV[2]
local output_count = tonumber(ARGV[3])
local run_hash_prefix = ARGV[4]
local source_stream = KEYS[output_count + 1]

-- Preconditions (fail before writes):
-- 1) Source message must still be pending for source stream/group.
local pending = redis.call('XPENDING', source_stream, source_group, source_msg_id, source_msg_id, 1)
if #pending == 0 or pending[1][1] ~= source_msg_id then
  return redis.error_reply('SOURCE_NOT_PENDING:' .. source_msg_id)
end

-- 2) Destination keys must be none/stream.
for i = 1, output_count do
  local key_type = redis.call('TYPE', KEYS[i])
  local key_type_name = key_type
  if type(key_type) == 'table' then
    key_type_name = key_type['ok']
  end
  if key_type_name ~= 'none' and key_type_name ~= 'stream' then
    return redis.error_reply('DEST_NOT_STREAM:' .. KEYS[i] .. ':' .. tostring(key_type_name))
  end
end

local now = redis.call('TIME')[1]
local ids = {}

for i = 1, output_count do
  local run_id_idx = 5 + (i - 1) * 3
  local payload_idx = run_id_idx + 1
  local stage_idx = run_id_idx + 2
  local run_id = ARGV[run_id_idx]
  local payload = ARGV[payload_idx]
  local stage = ARGV[stage_idx]
  local run_hash = run_hash_prefix .. run_id

  local next_id = redis.call('XADD', KEYS[i], '*', 'run_id', run_id, 'payload', payload, 'stage', stage)
  table.insert(ids, next_id)

  redis.call('HSET', run_hash, 'stage', stage, 'updated_at', tostring(now), stage .. '_enqueued_at', tostring(now))
end

local acked = redis.call('XACK', source_stream, source_group, source_msg_id)
if acked ~= 1 then
  -- Best-effort rollback for all produced fan-out entries.
  for i = 1, #ids do
    redis.pcall('XDEL', KEYS[i], ids[i])
  end
  return redis.error_reply('XACK_FAILED:' .. tostring(acked))
end

return ids
"#;
