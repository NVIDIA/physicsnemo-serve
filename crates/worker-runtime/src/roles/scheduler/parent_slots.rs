/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Arc;

use anyhow::{Result, anyhow};
use redis::Script;
use scicomp_rq::QueueManager;
#[cfg(test)]
use tokio::sync::Mutex;

use crate::traits::BoxFuture;

const ACQUIRE_PARENT_SLOT_LUA: &str = r#"
local current = tonumber(redis.call('GET', KEYS[1])) or 0
local max = tonumber(ARGV[1]) or 0
local slot_key = ARGV[3]
if max <= 0 then
  return {0, current}
end
if current >= max then
  redis.call('ZADD', KEYS[2], current, slot_key)
  return {0, current}
end
current = redis.call('INCR', KEYS[1])
local ttl = tonumber(ARGV[2]) or 0
if ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
end
redis.call('ZADD', KEYS[2], current, slot_key)
return {1, current}
"#;

const RELEASE_PARENT_SLOT_LUA: &str = r#"
local current = tonumber(redis.call('GET', KEYS[1])) or 0
local slot_key = ARGV[1]
if current <= 1 then
  redis.call('DEL', KEYS[1])
  redis.call('ZREM', KEYS[2], slot_key)
  return 0
end
current = redis.call('DECR', KEYS[1])
redis.call('ZADD', KEYS[2], current, slot_key)
return current
"#;

const DEFAULT_PARENT_SLOT_TTL_SECS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParentSlotAcquire {
    Acquired { active_count: usize },
    Saturated { active_count: usize },
}

pub(crate) trait ParentSlotStore: Send + Sync {
    fn try_acquire<'a>(
        &'a self,
        parent_run_id: &'a str,
        max_in_flight: usize,
    ) -> BoxFuture<'a, Result<ParentSlotAcquire>>;

    fn release<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<usize>>;
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct InMemoryParentSlotStore {
    state: Arc<Mutex<HashMap<String, usize>>>,
}

#[cfg(test)]
impl InMemoryParentSlotStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
impl ParentSlotStore for InMemoryParentSlotStore {
    fn try_acquire<'a>(
        &'a self,
        parent_run_id: &'a str,
        max_in_flight: usize,
    ) -> BoxFuture<'a, Result<ParentSlotAcquire>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let current = *state.get(parent_run_id).unwrap_or(&0);
            if current >= max_in_flight {
                return Ok(ParentSlotAcquire::Saturated {
                    active_count: current,
                });
            }
            let updated = current + 1;
            state.insert(parent_run_id.to_string(), updated);
            Ok(ParentSlotAcquire::Acquired {
                active_count: updated,
            })
        })
    }

    fn release<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let current = *state.get(parent_run_id).unwrap_or(&0);
            if current <= 1 {
                state.remove(parent_run_id);
                return Ok(0);
            }
            let updated = current - 1;
            state.insert(parent_run_id.to_string(), updated);
            Ok(updated)
        })
    }
}

pub(crate) struct RedisParentSlotStore {
    qm: QueueManager,
    key_prefix: String,
    ttl_secs: u64,
}

impl RedisParentSlotStore {
    pub(crate) fn new(qm: QueueManager, key_prefix: impl Into<String>) -> Self {
        Self {
            qm,
            key_prefix: key_prefix.into(),
            ttl_secs: DEFAULT_PARENT_SLOT_TTL_SECS,
        }
    }

    fn key(&self, parent_run_id: &str) -> String {
        format!("{}:{}", self.key_prefix, parent_run_id)
    }

    fn index_key(&self) -> String {
        format!("{}:index", self.key_prefix)
    }
}

fn parse_parent_slot_acquire_result(values: &[i64]) -> Result<ParentSlotAcquire> {
    let [acquired_raw, active_count_raw] = values else {
        return Err(anyhow!(
            "redis parent slot acquire returned {}, expected exactly 2 values",
            values.len()
        ));
    };
    if *active_count_raw < 0 {
        return Err(anyhow!(
            "redis parent slot acquire returned negative active_count={active_count_raw}"
        ));
    }
    let active_count = *active_count_raw as usize;
    match *acquired_raw {
        1 => Ok(ParentSlotAcquire::Acquired { active_count }),
        0 => Ok(ParentSlotAcquire::Saturated { active_count }),
        other => Err(anyhow!(
            "redis parent slot acquire returned invalid acquired flag={other}"
        )),
    }
}

fn parse_release_remaining(remaining: i64) -> Result<usize> {
    if remaining < 0 {
        return Err(anyhow!(
            "redis parent slot release returned negative remaining count={remaining}"
        ));
    }
    Ok(remaining as usize)
}

impl ParentSlotStore for RedisParentSlotStore {
    fn try_acquire<'a>(
        &'a self,
        parent_run_id: &'a str,
        max_in_flight: usize,
    ) -> BoxFuture<'a, Result<ParentSlotAcquire>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let slot_key = self.key(parent_run_id);
            let values: Vec<i64> = Script::new(ACQUIRE_PARENT_SLOT_LUA)
                .key(&slot_key)
                .key(self.index_key())
                .arg(max_in_flight as i64)
                .arg(self.ttl_secs as i64)
                .arg(&slot_key)
                .invoke_async(&mut conn)
                .await?;
            parse_parent_slot_acquire_result(&values)
        })
    }

    fn release<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let slot_key = self.key(parent_run_id);
            let remaining: i64 = Script::new(RELEASE_PARENT_SLOT_LUA)
                .key(&slot_key)
                .key(self.index_key())
                .arg(&slot_key)
                .invoke_async(&mut conn)
                .await?;
            parse_release_remaining(remaining)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_store_acquires_until_parent_saturates() {
        let store = InMemoryParentSlotStore::new();
        assert_eq!(
            store.try_acquire("parent", 2).await.unwrap(),
            ParentSlotAcquire::Acquired { active_count: 1 }
        );
        assert_eq!(
            store.try_acquire("parent", 2).await.unwrap(),
            ParentSlotAcquire::Acquired { active_count: 2 }
        );
        assert_eq!(
            store.try_acquire("parent", 2).await.unwrap(),
            ParentSlotAcquire::Saturated { active_count: 2 }
        );
    }

    #[tokio::test]
    async fn in_memory_store_release_reopens_parent_slot() {
        let store = InMemoryParentSlotStore::new();
        let _ = store.try_acquire("parent", 1).await.unwrap();
        assert_eq!(store.release("parent").await.unwrap(), 0);
        assert_eq!(
            store.try_acquire("parent", 1).await.unwrap(),
            ParentSlotAcquire::Acquired { active_count: 1 }
        );
    }

    #[test]
    fn parse_parent_slot_acquire_result_accepts_acquired_and_saturated_flags() {
        assert_eq!(
            parse_parent_slot_acquire_result(&[1, 3]).unwrap(),
            ParentSlotAcquire::Acquired { active_count: 3 }
        );
        assert_eq!(
            parse_parent_slot_acquire_result(&[0, 2]).unwrap(),
            ParentSlotAcquire::Saturated { active_count: 2 }
        );
    }

    #[test]
    fn parse_parent_slot_acquire_result_rejects_invalid_shapes_and_flags() {
        let shape_error = parse_parent_slot_acquire_result(&[1])
            .unwrap_err()
            .to_string();
        assert!(shape_error.contains("expected exactly 2 values"));

        let flag_error = parse_parent_slot_acquire_result(&[9, 2])
            .unwrap_err()
            .to_string();
        assert!(flag_error.contains("invalid acquired flag"));
    }

    #[test]
    fn parse_parent_slot_acquire_result_rejects_negative_active_count() {
        let error = parse_parent_slot_acquire_result(&[1, -1])
            .unwrap_err()
            .to_string();
        assert!(error.contains("negative active_count"));
    }

    #[test]
    fn parse_release_remaining_rejects_negative_values() {
        let error = parse_release_remaining(-1).unwrap_err().to_string();
        assert!(error.contains("negative remaining count"));
        assert_eq!(parse_release_remaining(0).unwrap(), 0);
        assert_eq!(parse_release_remaining(3).unwrap(), 3);
    }

    #[test]
    fn redis_lua_scripts_coerce_non_numeric_counters_to_zero() {
        assert!(
            ACQUIRE_PARENT_SLOT_LUA.contains("tonumber(redis.call('GET', KEYS[1])) or 0"),
            "acquire script should treat non-numeric counter values as zero"
        );
        assert!(
            RELEASE_PARENT_SLOT_LUA.contains("tonumber(redis.call('GET', KEYS[1])) or 0"),
            "release script should treat non-numeric counter values as zero"
        );
    }
}
