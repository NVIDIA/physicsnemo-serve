/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Immutable round admission. Mutable outcome/retry transitions belong to gather.
use anyhow::{Result, ensure};
use scicomp_rq::QueueManager;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod gather;
use gather::RoundState;

/// JSON encoding makes compound identities unambiguous, including arbitrary run IDs.
pub(crate) fn identity(parts: &[&str]) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(parts).unwrap()))
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RoundPlan {
    pub(crate) round_id: String,
    pub(crate) parent_run_id: String,
    pub(crate) stage_invocation_id: String,
    pub(crate) continuation_stage_id: String,
    pub(crate) parent_payload: Value,
    pub(crate) children: Vec<Value>,
    pub(crate) max_attempts: u32,
    pub(crate) schedule_stream: String,
}

// No active-state expiry: a slow round must not disappear or admit a second round.
// Terminal retention/cleanup is owned by gather; retained admission digests fence replay.
// Arguments contain serialized payloads, not Lua-decoded numbers (scientific parameters
// can include integers larger than Lua's exact numeric range).
const REGISTER: &str = r#"
local function type_is(key, expected)
  local t = redis.call('TYPE', key)['ok']
  return t == 'none' or t == expected
end
if not type_is(KEYS[1], 'hash') or not type_is(KEYS[2], 'string')
   or not type_is(KEYS[3], 'stream') then
  return redis.error_reply('ROUND_KEY_WRONG_TYPE')
end
local previous = redis.call('HGET', KEYS[1], 'digest')
if previous then
  if previous ~= ARGV[1] then return redis.error_reply('ROUND_PLAN_CONFLICT') end
  return 0
end
if redis.call('EXISTS', KEYS[4]) == 1 then
  return redis.error_reply('PARENT_TERMINAL')
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return redis.error_reply('PARENT_ROUND_ACTIVE')
end
local added = {}
local function rollback()
  for _, id in ipairs(added) do redis.call('XDEL', KEYS[3], id) end
  redis.call('DEL', KEYS[1])
end
for i = 5, #ARGV, 2 do
  local result = redis.pcall('XADD', KEYS[3], '*',
      'run_id', ARGV[i], 'payload', ARGV[i+1], 'stage', 'schedule')
  if type(result) == 'table' and result.err then
    rollback()
    return redis.error_reply(result.err)
  end
  table.insert(added, result)
end
local result = redis.pcall('HSET', KEYS[1], 'digest', ARGV[1],
    'plan', ARGV[2], 'state', ARGV[4], 'status', 'waiting')
if type(result) == 'table' and result.err then
  rollback()
  return redis.error_reply(result.err)
end
result = redis.pcall('SET', KEYS[2], ARGV[3])
if type(result) == 'table' and result.err then
  rollback()
  return redis.error_reply(result.err)
end
return 1
"#;

#[derive(Clone)]
pub(crate) struct RedisRoundStore {
    qm: QueueManager,
    prefix: String,
}

impl RedisRoundStore {
    pub(crate) fn new(qm: QueueManager, prefix: String) -> Self {
        Self { qm, prefix }
    }

    pub(crate) async fn register(&self, plan: &RoundPlan, schedule_stream: &str) -> Result<bool> {
        ensure!(!plan.children.is_empty(), "round must contain children");
        ensure!(plan.max_attempts > 0, "round max_attempts must be positive");
        ensure!(
            plan.schedule_stream == schedule_stream,
            "round schedule stream mismatch"
        );
        let serialized = serde_json::to_string(plan)?;
        let digest = format!("{:x}", Sha256::digest(serialized.as_bytes()));
        let script = redis::Script::new(REGISTER);
        let mut invocation = script.prepare_invoke();
        invocation
            .key(format!("{}round:{}", self.prefix, plan.round_id))
            .key(format!(
                "{}active_round:{}",
                self.prefix, plan.parent_run_id
            ))
            .key(format!("{}{schedule_stream}", self.prefix))
            // Match the existing parent cancellation/terminal store.
            .key(format!("parent_terminal:{}", plan.parent_run_id))
            .arg(digest)
            .arg(serialized)
            .arg(&plan.round_id)
            .arg(serde_json::to_string(&RoundState::new(plan))?);
        for child in &plan.children {
            let run_id = child["run_id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("child run_id missing"))?;
            invocation.arg(run_id).arg(serde_json::to_string(child)?);
        }
        let created: bool = invocation.invoke_async(&mut self.qm.connection()).await?;
        Ok(created)
    }
}
