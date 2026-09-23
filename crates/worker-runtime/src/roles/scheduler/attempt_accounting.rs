/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, anyhow, ensure};
use redis::Script;
use scicomp_rq::QueueManager;

use super::reserved_memory::{ACTIVE_RESERVED_MEMORY_HASH_KEY, RESERVED_MEMORY_HASH_KEY};
use crate::roles::parent_run_state::STATE_TOMBSTONE_TTL_SECS;

const ACQUIRE: &str = r#"
local function type_is(key, expected)
  local kind = redis.call('TYPE', key)['ok']
  return kind == 'none' or kind == expected
end
if not type_is(KEYS[1], 'hash') or not type_is(KEYS[2], 'string')
   or not type_is(KEYS[3], 'zset') or not type_is(KEYS[4], 'hash')
   or not type_is(KEYS[5], 'hash') then
  return redis.error_reply('ATTEMPT_ACCOUNTING_KEY_WRONG_TYPE')
end

local existing_status = redis.call('HGET', KEYS[1], 'status')
if existing_status then
  if existing_status ~= 'reserved' and existing_status ~= 'dispatched'
     and existing_status ~= 'released' then
    return redis.error_reply('ATTEMPT_ALLOCATION_INVALID_STATUS')
  end
  local expected = {ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6]}
  local fields = {'run_id', 'parent_run_id', 'attempt_token', 'resource_id', 'memory_mb', 'stream'}
  for i = 1, #fields do
    if redis.call('HGET', KEYS[1], fields[i]) ~= expected[i] then
      return redis.error_reply('ATTEMPT_ALLOCATION_CONFLICT')
    end
  end
  if existing_status == 'released' then return {2, 0, 0} end
  return {1, 0, 0}
end

local max_in_flight = tonumber(ARGV[7]) or 0
local active_slots = tonumber(redis.call('GET', KEYS[2])) or 0
if max_in_flight <= 0 then
  return redis.error_reply('MAX_IN_FLIGHT_MUST_BE_POSITIVE')
end
if active_slots >= max_in_flight then return {0, active_slots, 0} end

local resource_id = ARGV[4]
local memory_mb = tonumber(ARGV[5]) or 0
local observed_mb = tonumber(ARGV[8]) or 0
local usable_mb = tonumber(ARGV[9]) or 0
if memory_mb <= 0 or observed_mb < 0 or usable_mb <= 0 then
  return redis.error_reply('INVALID_GPU_ACCOUNTING_VALUE')
end
local accounted = tonumber(redis.call('HGET', KEYS[4], resource_id)) or 0
local active_memory = tonumber(redis.call('HGET', KEYS[5], resource_id)) or 0
local effective = math.max(accounted, observed_mb)
if effective + memory_mb > usable_mb then return {3, active_slots, effective} end

local accounted_after = effective + memory_mb
local active_after = active_memory + memory_mb
redis.call('HSET', KEYS[4], resource_id, accounted_after)
redis.call('HSET', KEYS[5], resource_id, active_after)
local slots_after = redis.call('INCR', KEYS[2])
redis.call('ZADD', KEYS[3], slots_after, KEYS[2])
redis.call('HSET', KEYS[1],
  'status', 'reserved',
  'run_id', ARGV[1],
  'parent_run_id', ARGV[2],
  'attempt_token', ARGV[3],
  'resource_id', resource_id,
  'memory_mb', ARGV[5],
  'stream', ARGV[6])
return {1, slots_after, accounted_after}
"#;

const RELEASE: &str = r#"
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return 0 end
if redis.call('HGET', KEYS[1], 'run_id') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'resource_id') ~= ARGV[2]
   or redis.call('HGET', KEYS[1], 'memory_mb') ~= ARGV[3] then
  return redis.error_reply('ATTEMPT_RELEASE_MISMATCH')
end
if status == 'released' then
  redis.call('EXPIRE', KEYS[1], ARGV[4])
  return 2
end

local resource_id = ARGV[2]
local amount = tonumber(ARGV[3]) or 0
local active = tonumber(redis.call('HGET', KEYS[5], resource_id)) or 0
local accounted = tonumber(redis.call('HGET', KEYS[4], resource_id)) or 0
if amount <= 0 or active < amount then
  return redis.error_reply('ATTEMPT_RELEASE_UNDERFLOW')
end
local active_after = active - amount
if active_after == 0 then
  redis.call('HDEL', KEYS[4], resource_id)
  redis.call('HDEL', KEYS[5], resource_id)
else
  local accounted_after = accounted - amount
  if accounted_after < active_after then accounted_after = active_after end
  redis.call('HSET', KEYS[4], resource_id, accounted_after)
  redis.call('HSET', KEYS[5], resource_id, active_after)
end

local slots = tonumber(redis.call('GET', KEYS[2])) or 0
if slots <= 1 then
  redis.call('DEL', KEYS[2])
  redis.call('ZREM', KEYS[3], KEYS[2])
else
  local slots_after = redis.call('DECR', KEYS[2])
  redis.call('ZADD', KEYS[3], slots_after, KEYS[2])
end
redis.call('HSET', KEYS[1], 'status', 'released')
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
"#;

const MARK_DISPATCHED: &str = r#"
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return 0 end
if status == 'reserved' then
  redis.call('HSET', KEYS[1], 'status', 'dispatched')
end
return 1
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AttemptAllocation {
    pub(super) allocation_id: String,
    pub(super) run_id: String,
    pub(super) parent_run_id: String,
    pub(super) attempt_token: String,
    pub(super) resource_id: u32,
    pub(super) memory_mb: u64,
    pub(super) stream: String,
    pub(super) released: bool,
}

pub(super) struct AttemptReservation<'a> {
    pub(super) allocation_id: &'a str,
    pub(super) run_id: &'a str,
    pub(super) parent_run_id: &'a str,
    pub(super) attempt_token: &'a str,
    pub(super) resource_id: u32,
    pub(super) memory_mb: u64,
    pub(super) stream: &'a str,
    pub(super) max_in_flight: usize,
    pub(super) observed_used_mb: u64,
    pub(super) usable_memory_mb: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttemptAcquire {
    Acquired,
    ParentSaturated,
    MemoryBlocked,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttemptRelease {
    Released,
    AlreadyReleased,
    Unknown,
}

#[derive(Clone)]
pub(super) struct RedisAttemptAccounting {
    qm: QueueManager,
    prefix: String,
}

impl RedisAttemptAccounting {
    pub(super) fn new(qm: QueueManager, prefix: String) -> Self {
        Self { qm, prefix }
    }

    fn allocation_key(&self, allocation_id: &str) -> String {
        format!(
            "{}scheduler:attempt_allocation:{allocation_id}",
            self.prefix
        )
    }

    fn parent_key(parent_run_id: &str) -> String {
        format!("parent_slots:{parent_run_id}")
    }

    fn parent_index_key() -> &'static str {
        "parent_slots:index"
    }

    pub(super) async fn get(&self, allocation_id: &str) -> Result<Option<AttemptAllocation>> {
        let key = self.allocation_key(allocation_id);
        let mut conn = self.qm.connection();
        let values: std::collections::HashMap<String, String> = redis::cmd("HGETALL")
            .arg(&key)
            .query_async(&mut conn)
            .await?;
        if values.is_empty() {
            return Ok(None);
        }
        let field = |name: &str| {
            values
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("attempt allocation '{allocation_id}' missing {name}"))
        };
        let status = field("status")?;
        ensure!(
            matches!(status.as_str(), "reserved" | "dispatched" | "released"),
            "attempt allocation '{allocation_id}' has invalid status"
        );
        let resource_id = field("resource_id")?.parse().with_context(|| {
            format!("attempt allocation '{allocation_id}' has invalid resource_id")
        })?;
        let memory_mb = field("memory_mb")?.parse().with_context(|| {
            format!("attempt allocation '{allocation_id}' has invalid memory_mb")
        })?;
        Ok(Some(AttemptAllocation {
            allocation_id: allocation_id.to_string(),
            run_id: field("run_id")?,
            parent_run_id: field("parent_run_id")?,
            attempt_token: field("attempt_token")?,
            resource_id,
            memory_mb,
            stream: field("stream")?,
            released: status == "released",
        }))
    }

    pub(super) async fn acquire(&self, request: &AttemptReservation<'_>) -> Result<AttemptAcquire> {
        ensure!(
            !request.allocation_id.is_empty(),
            "allocation_id is required"
        );
        let values: Vec<i64> = Script::new(ACQUIRE)
            .key(self.allocation_key(request.allocation_id))
            .key(Self::parent_key(request.parent_run_id))
            .key(Self::parent_index_key())
            .key(RESERVED_MEMORY_HASH_KEY)
            .key(ACTIVE_RESERVED_MEMORY_HASH_KEY)
            .arg(request.run_id)
            .arg(request.parent_run_id)
            .arg(request.attempt_token)
            .arg(request.resource_id)
            .arg(request.memory_mb)
            .arg(request.stream)
            .arg(request.max_in_flight)
            .arg(request.observed_used_mb)
            .arg(request.usable_memory_mb)
            .invoke_async(&mut self.qm.connection())
            .await?;
        match values.first().copied() {
            Some(0) => Ok(AttemptAcquire::ParentSaturated),
            Some(1) => Ok(AttemptAcquire::Acquired),
            Some(2) => Ok(AttemptAcquire::Released),
            Some(3) => Ok(AttemptAcquire::MemoryBlocked),
            other => Err(anyhow!("invalid attempt acquire response: {other:?}")),
        }
    }

    pub(super) async fn mark_dispatched(&self, allocation_id: &str) -> Result<()> {
        let found: i64 = Script::new(MARK_DISPATCHED)
            .key(self.allocation_key(allocation_id))
            .invoke_async(&mut self.qm.connection())
            .await?;
        ensure!(found == 1, "attempt allocation '{allocation_id}' not found");
        Ok(())
    }

    pub(super) async fn release(
        &self,
        allocation_id: &str,
        run_id: &str,
        resource_id: u32,
        memory_mb: u64,
    ) -> Result<AttemptRelease> {
        let allocation = self.get(allocation_id).await?;
        let Some(allocation) = allocation else {
            return Ok(AttemptRelease::Unknown);
        };
        let result: i64 = Script::new(RELEASE)
            .key(self.allocation_key(allocation_id))
            .key(Self::parent_key(&allocation.parent_run_id))
            .key(Self::parent_index_key())
            .key(RESERVED_MEMORY_HASH_KEY)
            .key(ACTIVE_RESERVED_MEMORY_HASH_KEY)
            .arg(run_id)
            .arg(resource_id)
            .arg(memory_mb)
            .arg(STATE_TOMBSTONE_TTL_SECS)
            .invoke_async(&mut self.qm.connection())
            .await?;
        match result {
            0 => Ok(AttemptRelease::Unknown),
            1 => Ok(AttemptRelease::Released),
            2 => Ok(AttemptRelease::AlreadyReleased),
            other => Err(anyhow!("invalid attempt release response: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::scheduler::reserved_memory::{RedisReservedMemoryStore, ReservedMemoryStore};
    use redis::AsyncCommands;
    mod test_support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    fn reservation<'a>(
        allocation_id: &'a str,
        run_id: &'a str,
        resource_id: u32,
    ) -> AttemptReservation<'a> {
        AttemptReservation {
            allocation_id,
            run_id,
            parent_run_id: "parent",
            attempt_token: allocation_id,
            resource_id,
            memory_mb: 4_096,
            stream: "gpu:0",
            max_in_flight: 1,
            observed_used_mb: 1_000,
            usable_memory_mb: 16_000,
        }
    }

    #[tokio::test]
    async fn redis_attempt_acquire_and_release_are_idempotent() {
        let (_server, qm) =
            test_support::spawn_test_queue_manager("attempt-accounting-idempotent").await;
        let store = RedisAttemptAccounting::new(qm.clone(), "test:".to_string());
        let first = reservation("allocation-a", "child-a", 0);

        assert_eq!(
            store.acquire(&first).await.unwrap(),
            AttemptAcquire::Acquired
        );
        let active_ttl: i64 = redis::cmd("TTL")
            .arg("test:scheduler:attempt_allocation:allocation-a")
            .query_async(&mut qm.connection())
            .await
            .unwrap();
        assert_eq!(active_ttl, -1, "active allocations must not expire");
        assert_eq!(
            store.acquire(&first).await.unwrap(),
            AttemptAcquire::Acquired
        );
        assert_eq!(
            store
                .acquire(&reservation("allocation-b", "child-b", 0))
                .await
                .unwrap(),
            AttemptAcquire::ParentSaturated
        );

        let active: Option<u64> = redis::cmd("HGET")
            .arg(ACTIVE_RESERVED_MEMORY_HASH_KEY)
            .arg("0")
            .query_async(&mut qm.connection())
            .await
            .unwrap();
        assert_eq!(
            active,
            Some(4_096),
            "duplicate acquire must not double count"
        );

        let store = RedisAttemptAccounting::new(qm.clone(), "test:".to_string());
        assert_eq!(
            store
                .release("allocation-a", "child-a", 0, 4_096)
                .await
                .unwrap(),
            AttemptRelease::Released
        );
        let released_ttl: i64 = redis::cmd("TTL")
            .arg("test:scheduler:attempt_allocation:allocation-a")
            .query_async(&mut qm.connection())
            .await
            .unwrap();
        assert!(
            released_ttl > 0 && released_ttl <= STATE_TOMBSTONE_TTL_SECS as i64,
            "released allocation must have a bounded tombstone TTL, got {released_ttl}"
        );
        assert_eq!(
            store
                .release("allocation-a", "child-a", 0, 4_096)
                .await
                .unwrap(),
            AttemptRelease::AlreadyReleased
        );
        assert_eq!(
            store
                .acquire(&reservation("allocation-b", "child-b", 0))
                .await
                .unwrap(),
            AttemptAcquire::Acquired,
            "one release must reopen exactly one parent slot"
        );
    }

    #[tokio::test]
    async fn redis_attempt_rejects_mismatched_release_without_freeing_capacity() {
        let (_server, qm) =
            test_support::spawn_test_queue_manager("attempt-accounting-mismatch").await;
        let store = RedisAttemptAccounting::new(qm, "test:".to_string());
        store
            .acquire(&reservation("allocation-a", "child-a", 0))
            .await
            .unwrap();

        let error = store
            .release("allocation-a", "wrong-child", 0, 4_096)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ATTEMPT_RELEASE_MISMATCH"));
        assert_eq!(
            store
                .acquire(&reservation("allocation-b", "child-b", 0))
                .await
                .unwrap(),
            AttemptAcquire::ParentSaturated,
            "rejected release must retain the parent slot"
        );
        assert!(!store.get("allocation-a").await.unwrap().unwrap().released);
    }

    #[tokio::test]
    async fn redis_attempt_blocks_memory_without_consuming_parent_slot() {
        let (_server, qm) =
            test_support::spawn_test_queue_manager("attempt-accounting-memory").await;
        let store = RedisAttemptAccounting::new(qm, "test:".to_string());
        let mut too_large = reservation("allocation-a", "child-a", 0);
        too_large.memory_mb = 15_001;

        assert_eq!(
            store.acquire(&too_large).await.unwrap(),
            AttemptAcquire::MemoryBlocked
        );
        assert_eq!(
            store
                .acquire(&reservation("allocation-b", "child-b", 0))
                .await
                .unwrap(),
            AttemptAcquire::Acquired
        );
    }

    #[tokio::test]
    async fn normal_and_attempt_reservations_share_effective_memory_rule() {
        let (_server, qm) =
            test_support::spawn_test_queue_manager("attempt-accounting-conformance").await;
        let normal = RedisReservedMemoryStore::new(qm.clone());
        let attempts = RedisAttemptAccounting::new(qm.clone(), "test:".to_string());

        for (index, (accounted, active, observed, requested)) in [
            (0_u64, 0_u64, 1_000_u64, 4_096_u64),
            (8_000, 3_000, 2_000, 1_024),
            (4_000, 3_000, 7_000, 2_048),
        ]
        .into_iter()
        .enumerate()
        {
            let mut conn = qm.connection();
            let _: usize = redis::cmd("DEL")
                .arg(RESERVED_MEMORY_HASH_KEY)
                .arg(ACTIVE_RESERVED_MEMORY_HASH_KEY)
                .query_async(&mut conn)
                .await
                .unwrap();
            if accounted > 0 {
                let _: usize = conn
                    .hset(RESERVED_MEMORY_HASH_KEY, "0", accounted)
                    .await
                    .unwrap();
            }
            if active > 0 {
                let _: usize = conn
                    .hset(ACTIVE_RESERVED_MEMORY_HASH_KEY, "0", active)
                    .await
                    .unwrap();
            }

            let normal_accounted = normal.reserve(0, observed, requested).await.unwrap();
            let normal_active: u64 = conn
                .hget(ACTIVE_RESERVED_MEMORY_HASH_KEY, "0")
                .await
                .unwrap();

            let _: usize = redis::cmd("DEL")
                .arg(RESERVED_MEMORY_HASH_KEY)
                .arg(ACTIVE_RESERVED_MEMORY_HASH_KEY)
                .query_async(&mut conn)
                .await
                .unwrap();
            if accounted > 0 {
                let _: usize = conn
                    .hset(RESERVED_MEMORY_HASH_KEY, "0", accounted)
                    .await
                    .unwrap();
            }
            if active > 0 {
                let _: usize = conn
                    .hset(ACTIVE_RESERVED_MEMORY_HASH_KEY, "0", active)
                    .await
                    .unwrap();
            }

            let allocation_id = format!("allocation-{index}");
            let run_id = format!("child-{index}");
            let parent_run_id = format!("parent-{index}");
            let attempt = AttemptReservation {
                allocation_id: &allocation_id,
                run_id: &run_id,
                parent_run_id: &parent_run_id,
                attempt_token: &allocation_id,
                resource_id: 0,
                memory_mb: requested,
                stream: "gpu:0",
                max_in_flight: 1,
                observed_used_mb: observed,
                usable_memory_mb: 100_000,
            };
            assert_eq!(
                attempts.acquire(&attempt).await.unwrap(),
                AttemptAcquire::Acquired
            );
            let attempt_accounted: u64 = conn.hget(RESERVED_MEMORY_HASH_KEY, "0").await.unwrap();
            let attempt_active: u64 = conn
                .hget(ACTIVE_RESERVED_MEMORY_HASH_KEY, "0")
                .await
                .unwrap();
            assert_eq!(attempt_accounted, normal_accounted);
            assert_eq!(attempt_active, normal_active);
            attempts
                .release(&allocation_id, &run_id, 0, requested)
                .await
                .unwrap();
        }
    }
}
