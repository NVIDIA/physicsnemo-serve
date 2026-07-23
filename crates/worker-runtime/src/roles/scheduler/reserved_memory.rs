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
use scicomp_rq::hash_ops;
#[cfg(test)]
use tokio::sync::Mutex;

use crate::traits::BoxFuture;

const RESERVED_MEMORY_HASH_KEY: &str = "scheduler:reserved_memory_mb";
const ACTIVE_RESERVED_MEMORY_HASH_KEY: &str = "scheduler:active_reserved_memory_mb";

pub(super) trait ReservedMemoryStore: Send + Sync {
    #[cfg(test)]
    /// Returns the scheduler-accounted memory totals for the requested resources.
    ///
    /// Resources with no tracked reservation are omitted from the returned map.
    fn get_many<'a>(&'a self, resource_ids: &'a [u32]) -> BoxFuture<'a, Result<HashMap<u32, u64>>>;

    /// Increases the scheduler-accounted memory and returns the updated total.
    fn reserve<'a>(
        &'a self,
        resource_id: u32,
        observed_used_mb: u64,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<u64>>;

    /// Decreases active reserved memory for a resource and returns the updated
    /// scheduler-accounted total.
    ///
    /// Returns `Ok(None)` when the resource does not have a tracked reservation.
    fn decrement<'a>(
        &'a self,
        resource_id: u32,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<Option<u64>>>;
}

#[derive(Clone)]
pub(super) struct RedisReservedMemoryStore {
    qm: QueueManager,
    hash_key: String,
    active_hash_key: String,
}

impl RedisReservedMemoryStore {
    pub(super) fn new(qm: QueueManager) -> Self {
        Self {
            qm,
            hash_key: RESERVED_MEMORY_HASH_KEY.to_string(),
            active_hash_key: ACTIVE_RESERVED_MEMORY_HASH_KEY.to_string(),
        }
    }

    fn field_name(resource_id: u32) -> String {
        resource_id.to_string()
    }
}

impl ReservedMemoryStore for RedisReservedMemoryStore {
    #[cfg(test)]
    fn get_many<'a>(&'a self, resource_ids: &'a [u32]) -> BoxFuture<'a, Result<HashMap<u32, u64>>> {
        Box::pin(async move {
            let fields: Vec<String> = resource_ids
                .iter()
                .map(|id| Self::field_name(*id))
                .collect();
            let mut conn = self.qm.connection();
            let raw_values = hash_ops::hmget(&mut conn, self.hash_key.as_str(), &fields).await?;
            raw_values
                .into_iter()
                .map(|(resource_id, value)| {
                    let resource_id = resource_id.parse::<u32>().map_err(|error| {
                        anyhow!(
                            "scheduler: invalid reserved-memory resource id '{resource_id}': {error}"
                        )
                    })?;
                    let parsed = value.parse::<u64>().map_err(|error| {
                        anyhow!(
                            "scheduler: invalid reserved memory value '{value}' for resource '{resource_id}': {error}"
                        )
                    })?;
                    Ok((resource_id, parsed))
                })
                .collect()
        })
    }

    fn reserve<'a>(
        &'a self,
        resource_id: u32,
        observed_used_mb: u64,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<u64>> {
        Box::pin(async move {
            let field = Self::field_name(resource_id);
            let floor = i64::try_from(observed_used_mb).map_err(|_| {
                anyhow!("scheduler: observed used memory overflow for resource '{resource_id}'")
            })?;
            let delta = i64::try_from(memory_mb).map_err(|_| {
                anyhow!(
                    "scheduler: reserved memory increment overflow for resource '{resource_id}'"
                )
            })?;
            let mut conn = self.qm.connection();
            let script = Script::new(
                r#"
local accounted_current = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
local active_current = tonumber(redis.call('HGET', KEYS[2], ARGV[1])) or 0
local floor = tonumber(ARGV[2]) or 0
local delta = tonumber(ARGV[3]) or 0
if floor < 0 then
  return redis.error_reply('floor must be non-negative')
end
if delta < 0 then
  return redis.error_reply('delta must be non-negative')
end
local updated_accounted = math.max(accounted_current, floor) + delta
local updated_active = active_current + delta
redis.call('HSET', KEYS[1], ARGV[1], updated_accounted)
redis.call('HSET', KEYS[2], ARGV[1], updated_active)
return updated_accounted
"#,
            );
            let updated: i64 = script
                .key(self.hash_key.as_str())
                .key(self.active_hash_key.as_str())
                .arg(field.as_str())
                .arg(floor)
                .arg(delta)
                .invoke_async(&mut conn)
                .await?;
            u64::try_from(updated).map_err(|_| {
                anyhow!("scheduler: reserved memory became negative for resource '{resource_id}'")
            })
        })
    }

    fn decrement<'a>(
        &'a self,
        resource_id: u32,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<Option<u64>>> {
        Box::pin(async move {
            let field = Self::field_name(resource_id);
            let amount = i64::try_from(memory_mb).map_err(|_| {
                anyhow!(
                    "scheduler: reserved memory decrement overflow for resource '{resource_id}'"
                )
            })?;
            let mut conn = self.qm.connection();
            let script = Script::new(
                r#"
local accounted_raw = redis.call('HGET', KEYS[1], ARGV[1])
local active_raw = redis.call('HGET', KEYS[2], ARGV[1])
local amount = tonumber(ARGV[2]) or 0
if amount < 0 then
  return redis.error_reply('amount must be non-negative')
end
if not accounted_raw and not active_raw then
  return {-1, 0}
end

local accounted_current = tonumber(accounted_raw) or 0
local active_current = tonumber(active_raw) or 0

if active_raw and active_current > amount then
  local active_after = active_current - amount
  local accounted_after = accounted_current - amount
  if accounted_after < active_after then
    accounted_after = active_after
  end
  redis.call('HSET', KEYS[1], ARGV[1], accounted_after)
  redis.call('HSET', KEYS[2], ARGV[1], active_after)
  return {accounted_after, active_after}
end

redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
return {0, 0}
"#,
            );
            let updated: Vec<i64> = script
                .key(self.hash_key.as_str())
                .key(self.active_hash_key.as_str())
                .arg(field.as_str())
                .arg(amount)
                .invoke_async(&mut conn)
                .await?;
            let Some(accounted_after) = updated.first().copied() else {
                return Err(anyhow!(
                    "scheduler: invalid reserved memory decrement response for resource '{resource_id}'"
                ));
            };
            if accounted_after < 0 {
                return Ok(None);
            }
            u64::try_from(accounted_after).map(Some).map_err(|_| {
                anyhow!("scheduler: reserved memory became negative for resource '{resource_id}'")
            })
        })
    }
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(super) struct InMemoryReservedMemoryStore {
    accounted_values: Arc<Mutex<HashMap<u32, u64>>>,
    active_values: Arc<Mutex<HashMap<u32, u64>>>,
}

#[cfg(test)]
impl ReservedMemoryStore for InMemoryReservedMemoryStore {
    fn get_many<'a>(&'a self, resource_ids: &'a [u32]) -> BoxFuture<'a, Result<HashMap<u32, u64>>> {
        Box::pin(async move {
            let values = self.accounted_values.lock().await;
            Ok(resource_ids
                .iter()
                .filter_map(|resource_id| {
                    values
                        .get(resource_id)
                        .copied()
                        .map(|value| (*resource_id, value))
                })
                .collect())
        })
    }

    fn reserve<'a>(
        &'a self,
        resource_id: u32,
        observed_used_mb: u64,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<u64>> {
        Box::pin(async move {
            let mut accounted_values = self.accounted_values.lock().await;
            let mut active_values = self.active_values.lock().await;
            let current_accounted = accounted_values.get(&resource_id).copied().unwrap_or(0);
            let updated_accounted = current_accounted
                .max(observed_used_mb)
                .saturating_add(memory_mb);
            let updated_active = active_values
                .get(&resource_id)
                .copied()
                .unwrap_or(0)
                .saturating_add(memory_mb);
            accounted_values.insert(resource_id, updated_accounted);
            active_values.insert(resource_id, updated_active);
            Ok(updated_accounted)
        })
    }

    fn decrement<'a>(
        &'a self,
        resource_id: u32,
        memory_mb: u64,
    ) -> BoxFuture<'a, Result<Option<u64>>> {
        Box::pin(async move {
            let mut accounted_values = self.accounted_values.lock().await;
            let mut active_values = self.active_values.lock().await;
            let Some(current_accounted) = accounted_values.get(&resource_id).copied() else {
                return Ok(None);
            };
            let current_active = active_values.get(&resource_id).copied().unwrap_or(0);
            let updated_active = current_active.saturating_sub(memory_mb);
            if updated_active == 0 {
                accounted_values.remove(&resource_id);
                active_values.remove(&resource_id);
                Ok(Some(0))
            } else {
                let updated_accounted = current_accounted
                    .saturating_sub(memory_mb)
                    .max(updated_active);
                accounted_values.insert(resource_id, updated_accounted);
                active_values.insert(resource_id, updated_active);
                Ok(Some(updated_accounted))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryReservedMemoryStore, ReservedMemoryStore};
    use std::collections::HashMap;

    #[tokio::test]
    async fn in_memory_store_tracks_incremented_resources() {
        let store = InMemoryReservedMemoryStore::default();

        assert_eq!(store.reserve(0, 0, 4_096).await.unwrap(), 4_096);
        assert_eq!(store.reserve(0, 0, 1_024).await.unwrap(), 5_120);
        assert_eq!(store.reserve(1, 0, 2_048).await.unwrap(), 2_048);

        let reserved = store.get_many(&[0, 1, 2]).await.unwrap();

        assert_eq!(reserved, HashMap::from([(0, 5_120_u64), (1, 2_048_u64)]));
    }

    #[tokio::test]
    async fn in_memory_store_decrement_clamps_to_zero_and_deletes_resource_entry() {
        let store = InMemoryReservedMemoryStore::default();
        store.reserve(7, 0, 8_192).await.unwrap();

        assert_eq!(store.decrement(7, 2_048).await.unwrap(), Some(6_144));
        assert_eq!(store.decrement(7, 10_000).await.unwrap(), Some(0));

        let reserved = store.get_many(&[7]).await.unwrap();
        assert!(reserved.is_empty(), "zero-valued entries should be removed");
    }

    #[tokio::test]
    async fn in_memory_store_decrement_removes_observed_floor_after_last_reservation() {
        let store = InMemoryReservedMemoryStore::default();
        store.reserve(3, 20_000, 10_000).await.unwrap();

        assert_eq!(store.decrement(3, 10_000).await.unwrap(), Some(0));

        let reserved = store.get_many(&[3]).await.unwrap();
        assert!(
            reserved.is_empty(),
            "observed memory floor should not remain as reserved accounting"
        );
    }

    #[tokio::test]
    async fn in_memory_store_decrement_keeps_floor_while_reservations_remain() {
        let store = InMemoryReservedMemoryStore::default();
        store.reserve(3, 20_000, 10_000).await.unwrap();
        store.reserve(3, 20_000, 5_000).await.unwrap();

        assert_eq!(store.decrement(3, 10_000).await.unwrap(), Some(25_000));
        assert_eq!(store.get_many(&[3]).await.unwrap().get(&3), Some(&25_000));

        assert_eq!(store.decrement(3, 5_000).await.unwrap(), Some(0));
        assert!(store.get_many(&[3]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn in_memory_store_returns_none_for_unknown_resource_decrement() {
        let store = InMemoryReservedMemoryStore::default();

        assert_eq!(store.decrement(99, 1_000).await.unwrap(), None);
    }

    #[tokio::test]
    async fn in_memory_store_reserve_uses_observed_memory_floor() {
        let store = InMemoryReservedMemoryStore::default();

        assert_eq!(store.reserve(3, 20_000, 10_000).await.unwrap(), 30_000);
        assert_eq!(store.reserve(3, 22_000, 5_000).await.unwrap(), 35_000);
    }
}
