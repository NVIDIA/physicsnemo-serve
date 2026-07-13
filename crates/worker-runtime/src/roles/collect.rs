/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use redis::Script;
use scicomp_rq::QueueManager;
use scicomp_rq::hash_ops;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use tokio::sync::Mutex;

use crate::roles::parent_run_state::{
    InMemoryParentRunStateStore, ParentRunStateStore, RedisParentRunStateStore,
};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole, message_deferred};

pub(crate) const DEFAULT_COLLECT_STORE_PREFIX: &str = "collect";
const RUN_KEY_PREFIX: &str = "run:";
const COLLECT_FINALIZATION_CLAIM_TTL_SECS: u64 = 600;
const COLLECT_FINALIZATION_COMMITTED_TTL_SECS: u64 = 24 * 60 * 60;

const RELEASE_COLLECT_FINALIZATION_IF_OWNER_LUA: &str = r#"
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"#;

const COMMIT_COLLECT_FINALIZATION_IF_OWNER_LUA: &str = r#"
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2])
  return 1
end
if current == ARGV[2] then
  return 1
end
return 0
"#;

const DISCARD_COLLECT_GROUP_LUA: &str = r#"
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
local finalization = redis.call('GET', KEYS[5])
if finalization and string.sub(finalization, 1, 10) == 'committed:' then
  redis.call('EXPIRE', KEYS[5], ARGV[1])
else
  redis.call('DEL', KEYS[5])
end
return 1
"#;

const APPEND_MEMBER_RESULT_LUA: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 then
  return {-1, 0, 0, 0, 0}
end
if redis.call('EXISTS', KEYS[4]) == 1 then
  return {
    -2,
    redis.call('LLEN', KEYS[2]),
    tonumber(redis.call('HGET', KEYS[5], 'succeeded') or '0'),
    tonumber(redis.call('HGET', KEYS[5], 'failed') or '0'),
    tonumber(redis.call('HGET', KEYS[5], 'cancelled') or '0')
  }
end
local added = redis.call('SADD', KEYS[3], ARGV[1])
if added == 1 then
  redis.call('RPUSH', KEYS[2], ARGV[2])
  redis.call('HINCRBY', KEYS[5], 'collected', 1)
  redis.call('HINCRBY', KEYS[5], ARGV[3], 1)
end
local size = redis.call('LLEN', KEYS[2])
return {
  added,
  size,
  tonumber(redis.call('HGET', KEYS[5], 'succeeded') or '0'),
  tonumber(redis.call('HGET', KEYS[5], 'failed') or '0'),
  tonumber(redis.call('HGET', KEYS[5], 'cancelled') or '0')
}
"#;

const PERSIST_COLLECT_PROGRESS_IF_OPEN_LUA: &str = r#"
if redis.call('EXISTS', KEYS[2]) == 1 then
  return 0
end
redis.call(
  'HSET', KEYS[1],
  'status', 'running',
  'stage', 'executing',
  'updated_at', ARGV[1],
  'workflow', ARGV[2],
  'fanout_expected_count', ARGV[3],
  'fanout_collected_count', ARGV[4],
  'fanout_succeeded_count', ARGV[5],
  'fanout_failed_count', ARGV[6],
  'fanout_cancelled_count', ARGV[7]
)
if ARGV[8] == '' then
  redis.call('HDEL', KEYS[1], 'fanout_child_run_ids')
else
  redis.call('HSET', KEYS[1], 'fanout_child_run_ids', ARGV[8])
end
redis.call('HDEL', KEYS[1], 'error')
return 1
"#;

fn collect_finalization_key(key_prefix: &str, parent_run_id: &str) -> String {
    format!("{key_prefix}:group:{parent_run_id}:finalizing")
}

fn committed_finalization_value(owner_token: &str) -> String {
    format!("committed:{owner_token}")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectedMemberResult {
    pub child_run_id: String,
    pub item_index: u64,
    #[serde(default)]
    pub fanout_item: JsonValue,
    pub result: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectGroup {
    pub parent_run_id: String,
    pub workflow_id: String,
    pub parent_payload: JsonValue,
    pub expected_count: usize,
    pub results: Vec<CollectedMemberResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectAppendResult {
    pub appended: bool,
    pub item_count: usize,
    pub succeeded_count: usize,
    pub failed_count: usize,
    pub cancelled_count: usize,
    pub child_run_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectFinalizationClaim {
    parent_run_id: String,
    state_key: String,
    owner_token: String,
    recovery_keys: Vec<String>,
}

pub(crate) trait CollectStore: Send + Sync {
    fn init_group<'a>(
        &'a self,
        parent_run_id: &'a str,
        group: CollectGroup,
    ) -> BoxFuture<'a, Result<()>>;

    fn discard_group<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>>;

    fn add_result<'a>(
        &'a self,
        parent_run_id: &'a str,
        result: CollectedMemberResult,
    ) -> BoxFuture<'a, Result<CollectAppendResult>>;

    fn get_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>>;

    fn claim_finalization<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectFinalizationClaim>>>;

    fn is_finalization_committed<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<bool>>;

    fn commit_finalization<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>>;

    fn release_finalization_claim<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectProgressUpdate {
    pub parent_run_id: String,
    pub workflow_id: String,
    pub expected_count: usize,
    pub collected_count: usize,
    pub succeeded_count: usize,
    pub failed_count: usize,
    pub cancelled_count: usize,
    pub child_run_ids: Vec<String>,
}

pub(crate) trait CollectProgressPersistence: Send + Sync {
    fn persist_progress<'a>(&'a self, update: CollectProgressUpdate) -> BoxFuture<'a, Result<()>>;

    fn persist_publication_skipped<'a>(
        &'a self,
        _parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

pub(crate) struct NoopCollectProgressPersistence;

impl CollectProgressPersistence for NoopCollectProgressPersistence {
    fn persist_progress<'a>(&'a self, _update: CollectProgressUpdate) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

pub(crate) struct RedisCollectProgressPersistence {
    qm: QueueManager,
    collect_store_prefix: String,
}

impl RedisCollectProgressPersistence {
    pub(crate) fn new(qm: QueueManager, collect_store_prefix: impl Into<String>) -> Self {
        Self {
            qm,
            collect_store_prefix: collect_store_prefix.into(),
        }
    }

    fn run_key(parent_run_id: &str) -> String {
        format!("{RUN_KEY_PREFIX}{parent_run_id}")
    }
}

impl CollectProgressPersistence for RedisCollectProgressPersistence {
    fn persist_progress<'a>(&'a self, update: CollectProgressUpdate) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let now_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("collect: system clock before unix epoch")?
                .as_secs()
                .to_string();
            let run_key = Self::run_key(update.parent_run_id.as_str());
            let finalization_key = collect_finalization_key(
                self.collect_store_prefix.as_str(),
                update.parent_run_id.as_str(),
            );
            let child_run_ids = if update.child_run_ids.is_empty() {
                String::new()
            } else {
                serde_json::to_string(&update.child_run_ids)
                    .context("collect: failed to encode child run ids")?
            };
            let mut conn = self.qm.connection();
            let _: i64 = Script::new(PERSIST_COLLECT_PROGRESS_IF_OPEN_LUA)
                .key(run_key)
                .key(finalization_key)
                .arg(now_secs)
                .arg(update.workflow_id)
                .arg(update.expected_count)
                .arg(update.collected_count)
                .arg(update.succeeded_count)
                .arg(update.failed_count)
                .arg(update.cancelled_count)
                .arg(child_run_ids)
                .invoke_async(&mut conn)
                .await
                .context("collect: failed to persist fanout progress hash fields")?;
            Ok(())
        })
    }

    fn persist_publication_skipped<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let now_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("collect: system clock before unix epoch")?
                .as_secs()
                .to_string();
            let run_key = Self::run_key(parent_run_id);
            let mut conn = self.qm.connection();
            let _: usize = redis::cmd("HSET")
                .arg(&run_key)
                .arg("updated_at")
                .arg(&now_secs)
                .arg("output_location")
                .arg("local_and_cloud")
                .arg("output_publication_status")
                .arg("skipped")
                .arg("publish_completed_at")
                .arg(&now_secs)
                .arg("published_artifact_count")
                .arg("0")
                .query_async(&mut conn)
                .await
                .context("collect: failed to persist skipped publication status")?;
            let _: i64 = hash_ops::hdel(&mut conn, &run_key, "publish_error")
                .await
                .context("collect: failed to clear stale publish error field")?;
            Ok(())
        })
    }
}

#[derive(Debug, Default)]
struct InMemoryCollectState {
    groups: HashMap<String, CollectGroup>,
    finalizations: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub(crate) struct InMemoryCollectStore {
    state: Arc<Mutex<InMemoryCollectState>>,
}

impl InMemoryCollectStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl CollectStore for InMemoryCollectStore {
    fn init_group<'a>(
        &'a self,
        parent_run_id: &'a str,
        group: CollectGroup,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            if !state.groups.contains_key(parent_run_id) {
                state.groups.insert(parent_run_id.to_string(), group);
                state.finalizations.remove(parent_run_id);
            }
            Ok(())
        })
    }

    fn discard_group<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            state.groups.remove(parent_run_id);
            if state
                .finalizations
                .get(parent_run_id)
                .is_some_and(|value| !value.starts_with("committed:"))
            {
                state.finalizations.remove(parent_run_id);
            }
            Ok(())
        })
    }

    fn add_result<'a>(
        &'a self,
        parent_run_id: &'a str,
        result: CollectedMemberResult,
    ) -> BoxFuture<'a, Result<CollectAppendResult>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let finalization_started = state.finalizations.contains_key(parent_run_id);
            let Some(group) = state.groups.get_mut(parent_run_id) else {
                return Err(anyhow!(
                    "collect: parent group '{}' was not initialized",
                    parent_run_id
                ));
            };
            if finalization_started {
                return Ok(CollectAppendResult {
                    appended: false,
                    item_count: group.results.len(),
                    succeeded_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Succeeded,
                    ),
                    failed_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Failed,
                    ),
                    cancelled_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Cancelled,
                    ),
                    child_run_ids: group
                        .results
                        .iter()
                        .map(|entry| entry.child_run_id.clone())
                        .collect(),
                });
            }
            if group
                .results
                .iter()
                .any(|existing| existing.child_run_id == result.child_run_id)
            {
                return Ok(CollectAppendResult {
                    appended: false,
                    item_count: group.results.len(),
                    succeeded_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Succeeded,
                    ),
                    failed_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Failed,
                    ),
                    cancelled_count: count_results_with_status(
                        &group.results,
                        MemberResultStatus::Cancelled,
                    ),
                    child_run_ids: group
                        .results
                        .iter()
                        .map(|entry| entry.child_run_id.clone())
                        .collect(),
                });
            }
            group.results.push(result);
            Ok(CollectAppendResult {
                appended: true,
                item_count: group.results.len(),
                succeeded_count: count_results_with_status(
                    &group.results,
                    MemberResultStatus::Succeeded,
                ),
                failed_count: count_results_with_status(&group.results, MemberResultStatus::Failed),
                cancelled_count: count_results_with_status(
                    &group.results,
                    MemberResultStatus::Cancelled,
                ),
                child_run_ids: group
                    .results
                    .iter()
                    .map(|entry| entry.child_run_id.clone())
                    .collect(),
            })
        })
    }

    fn get_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>> {
        Box::pin(async move {
            let state = self.state.lock().await;
            Ok(state.groups.get(parent_run_id).cloned())
        })
    }

    fn claim_finalization<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectFinalizationClaim>>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            if state.finalizations.contains_key(parent_run_id) {
                return Ok(None);
            }
            let owner_token = uuid::Uuid::new_v4().to_string();
            state
                .finalizations
                .insert(parent_run_id.to_string(), owner_token.clone());
            Ok(Some(CollectFinalizationClaim {
                parent_run_id: parent_run_id.to_string(),
                state_key: parent_run_id.to_string(),
                owner_token,
                recovery_keys: Vec::new(),
            }))
        })
    }

    fn is_finalization_committed<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let state = self.state.lock().await;
            Ok(state
                .finalizations
                .get(parent_run_id)
                .is_some_and(|value| value.starts_with("committed:")))
        })
    }

    fn commit_finalization<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let committed = committed_finalization_value(claim.owner_token.as_str());
            match state.finalizations.get(claim.parent_run_id.as_str()) {
                Some(current) if current == &claim.owner_token => {
                    state
                        .finalizations
                        .insert(claim.parent_run_id.clone(), committed);
                    Ok(())
                }
                Some(current) if current == &committed => Ok(()),
                _ => Err(anyhow!(
                    "collect: finalization claim is no longer owned by this worker"
                )),
            }
        })
    }

    fn release_finalization_claim<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            if state.finalizations.get(claim.parent_run_id.as_str()) == Some(&claim.owner_token) {
                state.finalizations.remove(claim.parent_run_id.as_str());
            }
            Ok(())
        })
    }
}

pub(crate) struct RedisCollectStore {
    qm: QueueManager,
    key_prefix: String,
}

impl RedisCollectStore {
    pub(crate) fn new(qm: QueueManager, key_prefix: impl Into<String>) -> Self {
        Self {
            qm,
            key_prefix: key_prefix.into(),
        }
    }

    fn meta_key(&self, parent_run_id: &str) -> String {
        format!("{}:group:{}:meta", self.key_prefix, parent_run_id)
    }

    fn items_key(&self, parent_run_id: &str) -> String {
        format!("{}:group:{}:items", self.key_prefix, parent_run_id)
    }

    fn runids_key(&self, parent_run_id: &str) -> String {
        format!("{}:group:{}:runids", self.key_prefix, parent_run_id)
    }

    fn stats_key(&self, parent_run_id: &str) -> String {
        format!("{}:group:{}:stats", self.key_prefix, parent_run_id)
    }

    fn finalization_key(&self, parent_run_id: &str) -> String {
        collect_finalization_key(self.key_prefix.as_str(), parent_run_id)
    }

    async fn load_group(&self, parent_run_id: &str) -> Result<Option<CollectGroup>> {
        let mut conn = self.qm.connection();
        let meta_json: Option<String> = redis::cmd("GET")
            .arg(self.meta_key(parent_run_id))
            .query_async(&mut conn)
            .await
            .context("collect: failed to load redis collect group metadata")?;
        let Some(meta_json) = meta_json else {
            return Ok(None);
        };

        let mut group: CollectGroup = serde_json::from_str(&meta_json)?;
        let item_json: Vec<String> = redis::cmd("LRANGE")
            .arg(self.items_key(parent_run_id))
            .arg(0)
            .arg(-1)
            .query_async(&mut conn)
            .await
            .context("collect: failed to load redis collect group results")?;
        group.results = item_json
            .into_iter()
            .map(|item_json| serde_json::from_str::<CollectedMemberResult>(&item_json))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(group))
    }
}

impl CollectStore for RedisCollectStore {
    fn init_group<'a>(
        &'a self,
        parent_run_id: &'a str,
        group: CollectGroup,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let meta_json = serde_json::to_string(&CollectGroup {
                results: Vec::new(),
                ..group
            })?;
            let mut conn = self.qm.connection();
            let created: bool = redis::cmd("SETNX")
                .arg(self.meta_key(parent_run_id))
                .arg(meta_json)
                .query_async(&mut conn)
                .await
                .context("collect: failed to initialize redis collect group")?;
            if created {
                let _: i64 = redis::cmd("DEL")
                    .arg(self.items_key(parent_run_id))
                    .arg(self.runids_key(parent_run_id))
                    .arg(self.stats_key(parent_run_id))
                    .arg(self.finalization_key(parent_run_id))
                    .query_async(&mut conn)
                    .await
                    .context("collect: failed to reset redis collect group state")?;
            }
            let now_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_string();
            let run_key = format!("{RUN_KEY_PREFIX}{parent_run_id}");
            let _: () = redis::cmd("HSET")
                .arg(&run_key)
                .arg("stage")
                .arg("executing")
                .arg("updated_at")
                .arg(&now_secs)
                .query_async(&mut conn)
                .await
                .context("collect: failed to set parent stage to executing")?;
            Ok(())
        })
    }

    fn discard_group<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let _: i64 = Script::new(DISCARD_COLLECT_GROUP_LUA)
                .key(self.meta_key(parent_run_id))
                .key(self.items_key(parent_run_id))
                .key(self.runids_key(parent_run_id))
                .key(self.stats_key(parent_run_id))
                .key(self.finalization_key(parent_run_id))
                .arg(COLLECT_FINALIZATION_COMMITTED_TTL_SECS)
                .invoke_async(&mut conn)
                .await
                .context("collect: failed to discard redis collect group")?;
            Ok(())
        })
    }

    fn add_result<'a>(
        &'a self,
        parent_run_id: &'a str,
        result: CollectedMemberResult,
    ) -> BoxFuture<'a, Result<CollectAppendResult>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let status = member_result_status(&result);
            let status_field = match status {
                MemberResultStatus::Succeeded => "succeeded",
                MemberResultStatus::Failed => "failed",
                MemberResultStatus::Cancelled => "cancelled",
            };
            let item_json = serde_json::to_string(&result)?;
            let values: Vec<i64> = Script::new(APPEND_MEMBER_RESULT_LUA)
                .key(self.meta_key(parent_run_id))
                .key(self.items_key(parent_run_id))
                .key(self.runids_key(parent_run_id))
                .key(self.finalization_key(parent_run_id))
                .key(self.stats_key(parent_run_id))
                .arg(&result.child_run_id)
                .arg(item_json)
                .arg(status_field)
                .invoke_async(&mut conn)
                .await?;
            if values.len() != 5 {
                return Err(anyhow!(
                    "collect: append redis script returned {} values, expected 5",
                    values.len()
                ));
            }
            if values[0] == -1 {
                return Err(anyhow!(
                    "collect: parent group '{}' was not initialized",
                    parent_run_id
                ));
            }
            let mut child_run_ids: Vec<String> = redis::cmd("SMEMBERS")
                .arg(self.runids_key(parent_run_id))
                .query_async(&mut conn)
                .await
                .context("collect: failed to read child run ids for progress snapshot")?;
            child_run_ids.sort_unstable();
            Ok(CollectAppendResult {
                appended: values[0] == 1,
                item_count: values[1].max(0) as usize,
                succeeded_count: values[2].max(0) as usize,
                failed_count: values[3].max(0) as usize,
                cancelled_count: values[4].max(0) as usize,
                child_run_ids,
            })
        })
    }

    fn get_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>> {
        Box::pin(async move { self.load_group(parent_run_id).await })
    }

    fn claim_finalization<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectFinalizationClaim>>> {
        Box::pin(async move {
            let state_key = self.finalization_key(parent_run_id);
            let owner_token = uuid::Uuid::new_v4().to_string();
            let mut conn = self.qm.connection();
            let claimed: Option<String> = redis::cmd("SET")
                .arg(&state_key)
                .arg(&owner_token)
                .arg("NX")
                .arg("EX")
                .arg(COLLECT_FINALIZATION_CLAIM_TTL_SECS)
                .query_async(&mut conn)
                .await
                .context("collect: failed to claim parent finalization")?;
            Ok(claimed.map(|_| CollectFinalizationClaim {
                parent_run_id: parent_run_id.to_string(),
                state_key,
                owner_token,
                recovery_keys: vec![
                    self.meta_key(parent_run_id),
                    self.items_key(parent_run_id),
                    self.runids_key(parent_run_id),
                    self.stats_key(parent_run_id),
                ],
            }))
        })
    }

    fn is_finalization_committed<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let state: Option<String> = redis::cmd("GET")
                .arg(self.finalization_key(parent_run_id))
                .query_async(&mut conn)
                .await
                .context("collect: failed to inspect parent finalization state")?;
            Ok(state.is_some_and(|value| value.starts_with("committed:")))
        })
    }

    fn commit_finalization<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let committed = committed_finalization_value(claim.owner_token.as_str());
            let committed_by_owner: bool = Script::new(COMMIT_COLLECT_FINALIZATION_IF_OWNER_LUA)
                .key(&claim.state_key)
                .arg(&claim.owner_token)
                .arg(committed)
                .invoke_async(&mut conn)
                .await
                .context("collect: failed to commit parent finalization claim")?;
            if !committed_by_owner {
                return Err(anyhow!(
                    "collect: finalization claim is no longer owned by this worker"
                ));
            }
            Ok(())
        })
    }

    fn release_finalization_claim<'a>(
        &'a self,
        claim: &'a CollectFinalizationClaim,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let _: i64 = Script::new(RELEASE_COLLECT_FINALIZATION_IF_OWNER_LUA)
                .key(&claim.state_key)
                .arg(&claim.owner_token)
                .invoke_async(&mut conn)
                .await
                .context("collect: failed to release parent finalization claim")?;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CollectEnvelope {
    workflow_id: String,
    parent_run_id: String,
    fanout_profile: FanoutProfile,
    fanout_item: FanoutItemRef,
    result: JsonValue,
    stage_context: StageContext,
}

#[derive(Debug, Clone, Deserialize)]
struct FanoutProfile {
    item_count: usize,
    #[serde(default)]
    failure_policy: Option<FailurePolicy>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FailurePolicy {
    CollectAll,
    FailFast,
}

impl FanoutProfile {
    fn failure_policy(&self) -> FailurePolicy {
        self.failure_policy.unwrap_or(FailurePolicy::CollectAll)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct FanoutItemRef {
    item_index: u64,
}

use crate::roles::stage::StageContext;

pub struct CollectRole {
    input_streams: Vec<String>,
    store: Arc<dyn CollectStore>,
    progress_persistence: Arc<dyn CollectProgressPersistence>,
    terminal_state: Arc<dyn ParentRunStateStore>,
}

impl CollectRole {
    pub fn from_env(env: &RoleEnv) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Self::from_env_with_store(
            env,
            Arc::new(InMemoryCollectStore::new()),
            Arc::new(NoopCollectProgressPersistence),
            Arc::new(InMemoryParentRunStateStore::new()),
        )
    }

    pub fn from_env_with_queue_manager(
        env: &RoleEnv,
        qm: QueueManager,
    ) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Self::from_env_with_store(
            env,
            Arc::new(RedisCollectStore::new(
                qm.clone(),
                DEFAULT_COLLECT_STORE_PREFIX,
            )),
            Arc::new(RedisCollectProgressPersistence::new(
                qm.clone(),
                DEFAULT_COLLECT_STORE_PREFIX,
            )),
            Arc::new(RedisParentRunStateStore::new(qm)),
        )
    }

    pub(crate) fn from_env_with_store(
        env: &RoleEnv,
        store: Arc<dyn CollectStore>,
        progress_persistence: Arc<dyn CollectProgressPersistence>,
        terminal_state: Arc<dyn ParentRunStateStore>,
    ) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Ok((
            Self {
                input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
                store,
                progress_persistence,
                terminal_state,
            },
            vec![],
        ))
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "collect: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn complete_finalization_cleanup(&self, parent_run_id: &str) -> Result<()> {
        self.terminal_state.mark_terminal(parent_run_id).await?;
        self.store.discard_group(parent_run_id).await
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let (typed, payload) = decode_collect_payload(msg.payload())?;
        if self
            .terminal_state
            .is_terminal(typed.parent_run_id.as_str())
            .await?
        {
            self.store
                .discard_group(typed.parent_run_id.as_str())
                .await?;
            return Ok(());
        }
        let append_result = match self
            .store
            .add_result(
                typed.parent_run_id.as_str(),
                CollectedMemberResult {
                    child_run_id: typed_child_run_id(&payload, msg.run_id()),
                    item_index: typed.fanout_item.item_index,
                    fanout_item: payload
                        .get("fanout_item")
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                    result: typed.result.clone(),
                },
            )
            .await
        {
            Ok(append_result) => append_result,
            Err(error) => {
                if error.to_string().contains("was not initialized")
                    && self
                        .terminal_state
                        .is_terminal(typed.parent_run_id.as_str())
                        .await?
                {
                    return Ok(());
                }
                return Err(error);
            }
        };
        if append_result.appended {
            self.progress_persistence
                .persist_progress(CollectProgressUpdate {
                    parent_run_id: typed.parent_run_id.clone(),
                    workflow_id: typed.workflow_id.clone(),
                    expected_count: typed.fanout_profile.item_count,
                    collected_count: append_result.item_count,
                    succeeded_count: append_result.succeeded_count,
                    failed_count: append_result.failed_count,
                    cancelled_count: append_result.cancelled_count,
                    child_run_ids: append_result.child_run_ids.clone(),
                })
                .await?;
        }
        if self
            .store
            .is_finalization_committed(typed.parent_run_id.as_str())
            .await?
        {
            return self
                .complete_finalization_cleanup(typed.parent_run_id.as_str())
                .await;
        }
        let fail_fast = typed.fanout_profile.failure_policy() == FailurePolicy::FailFast;
        let current_failed = typed
            .result
            .get("status")
            .and_then(JsonValue::as_str)
            .map(|status| {
                matches!(
                    status.trim().to_ascii_lowercase().as_str(),
                    "failed" | "fail" | "error"
                )
            })
            .unwrap_or(false);
        let should_finalize = (fail_fast && current_failed)
            || append_result.item_count >= typed.fanout_profile.item_count;
        if !should_finalize {
            return Ok(());
        }

        let Some(finalization_claim) = self
            .store
            .claim_finalization(typed.parent_run_id.as_str())
            .await?
        else {
            if self
                .store
                .is_finalization_committed(typed.parent_run_id.as_str())
                .await?
            {
                return self
                    .complete_finalization_cleanup(typed.parent_run_id.as_str())
                    .await;
            }
            return Err(message_deferred(
                "collect: parent finalization is already in progress",
            ));
        };

        let finalization_result = self
            .finalize_parent_group(msg, sink, &typed, &finalization_claim)
            .await;
        if finalization_result.is_err() {
            self.store
                .release_finalization_claim(&finalization_claim)
                .await?;
        }
        finalization_result
    }

    async fn finalize_parent_group(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
        typed: &CollectEnvelope,
        finalization_claim: &CollectFinalizationClaim,
    ) -> Result<()> {
        let Some(group) = self.store.get_group(typed.parent_run_id.as_str()).await? else {
            if self
                .terminal_state
                .is_terminal(typed.parent_run_id.as_str())
                .await?
            {
                return Ok(());
            }
            return Err(anyhow!(
                "collect: completed parent group '{}' disappeared before flush",
                typed.parent_run_id
            ));
        };

        let stage_context = parent_stage_context(&group.parent_payload)?;
        let mut next_stage = stage_context.next_stage("collect")?;
        let result_payload = build_parent_result(&group);
        let mut bypassed_publish = false;
        if next_stage.phase == "publish"
            && has_output_publication_target(&group.parent_payload)
            && !result_has_publishable_output(&result_payload)
        {
            bypassed_publish = true;
            let publish_stage_id = next_stage.id.clone();
            let results_stage_id = next_stage.next.clone().ok_or_else(|| {
                anyhow!("collect: publish stage '{publish_stage_id}' has no next stage")
            })?;
            next_stage = stage_context
                .pipeline
                .iter()
                .find(|stage| stage.id == results_stage_id)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "collect: publish bypass target stage '{results_stage_id}' not found in pipeline"
                    )
                })?;
            if next_stage.phase != "results" {
                return Err(anyhow!(
                    "collect: publish bypass target must be 'results', got '{}'",
                    next_stage.phase
                ));
            }
        }
        if matches!(next_stage.phase.as_str(), "postprocess" | "publish") {
            let mut handoff_payload = group.parent_payload.clone();
            let handoff_map = handoff_payload
                .as_object_mut()
                .ok_or_else(|| anyhow!("collect: parent payload must be a JSON object"))?;
            handoff_map.insert("result".to_string(), result_payload);
            crate::roles::stage::update_stage_context(handoff_map, &next_stage, "collect")?;
            let encoded = serde_json::to_string(&handoff_payload)
                .context("collect: encode downstream payload")?;
            sink.handoff_to_run_and_commit(
                msg,
                &next_stage.queue,
                &encoded,
                &next_stage.phase,
                group.parent_run_id.as_str(),
                &finalization_claim.state_key,
                &finalization_claim.owner_token,
                &finalization_claim.recovery_keys,
            )
            .await
            .with_context(|| {
                format!(
                    "collect: failed to hand off aggregated parent to {}",
                    next_stage.phase
                )
            })?;
            self.store.commit_finalization(finalization_claim).await?;
            self.complete_finalization_cleanup(typed.parent_run_id.as_str())
                .await?;
            return Ok(());
        }
        if next_stage.phase != "results" {
            return Err(anyhow!(
                "collect: next stage must be 'results', 'publish', or 'postprocess', got '{}'",
                next_stage.phase
            ));
        }
        if bypassed_publish {
            self.progress_persistence
                .persist_publication_skipped(group.parent_run_id.as_str())
                .await?;
        }

        let status = result_payload
            .get("status")
            .and_then(JsonValue::as_str)
            .unwrap_or("succeeded");
        let (execution, payload) = build_execution_and_payload(
            group.parent_run_id.as_str(),
            group.workflow_id.as_str(),
            status,
            &result_payload,
        )?;
        let results_envelope = json!({
            "run_id": group.parent_run_id,
            "status": status,
            "workflow": group.workflow_id,
            "request": build_request_envelope(&group.parent_payload),
            "execution": execution,
            "payload": payload,
        });
        let encoded =
            serde_json::to_string(&results_envelope).context("collect: encode results payload")?;
        sink.handoff_to_run_and_commit(
            msg,
            &next_stage.queue,
            &encoded,
            &next_stage.phase,
            group.parent_run_id.as_str(),
            &finalization_claim.state_key,
            &finalization_claim.owner_token,
            &finalization_claim.recovery_keys,
        )
        .await
        .context("collect: failed to hand off aggregated parent to results")?;
        self.store.commit_finalization(finalization_claim).await?;
        self.complete_finalization_cleanup(typed.parent_run_id.as_str())
            .await?;
        Ok(())
    }
}

impl WorkerRole for CollectRole {
    fn name(&self) -> &'static str {
        "collect"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            self.process_message(msg, sink).await
        })
    }
}

fn typed_child_run_id(payload: &JsonValue, fallback: &str) -> String {
    payload
        .get("run_id")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

fn decode_collect_payload(raw: &str) -> Result<(CollectEnvelope, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("collect: empty payload"));
    }
    let value: JsonValue =
        serde_json::from_str(raw).context("collect: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("collect: payload must be a JSON object"));
    }
    let typed: CollectEnvelope =
        serde_json::from_value(value.clone()).context("collect: invalid payload schema")?;
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "collect: workflow_id is required and must be non-empty"
        ));
    }
    if typed.parent_run_id.trim().is_empty() {
        return Err(anyhow!(
            "collect: parent_run_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "collect" {
        return Err(anyhow!(
            "collect: payload current_phase must be 'collect', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    if typed.fanout_profile.item_count == 0 {
        return Err(anyhow!(
            "collect: fanout_profile.item_count must be greater than zero"
        ));
    }
    Ok((typed, value))
}

fn parent_stage_context(payload: &JsonValue) -> Result<StageContext> {
    payload
        .get("stage_context")
        .cloned()
        .ok_or_else(|| anyhow!("collect: parent payload missing stage_context"))
        .and_then(|value| {
            serde_json::from_value(value)
                .context("collect: parent payload stage_context must be a valid object")
        })
}

fn build_parent_result(group: &CollectGroup) -> JsonValue {
    let mut results = group.results.clone();
    results.sort_by_key(|entry| entry.item_index);
    let succeeded_count = results
        .iter()
        .filter(|entry| {
            entry
                .result
                .get("status")
                .and_then(JsonValue::as_str)
                .unwrap_or("succeeded")
                == "succeeded"
        })
        .count();
    let failed_count = results.len().saturating_sub(succeeded_count);
    json!({
        "status": if failed_count == 0 { "succeeded" } else { "failed" },
        "artifacts": [],
        "child_results": results.iter().map(|entry| json!({
            "item_index": entry.item_index,
            "child_run_id": entry.child_run_id,
            "fanout_item": entry.fanout_item,
            "result": entry.result,
        })).collect::<Vec<_>>(),
        "aggregation_summary": {
            "item_count": group.expected_count,
            "collected_count": results.len(),
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
        }
    })
}

fn result_has_publishable_output(result: &JsonValue) -> bool {
    let Some(result) = result.as_object() else {
        return false;
    };
    if result
        .get("output_path")
        .and_then(JsonValue::as_str)
        .is_some_and(|path| !path.trim().is_empty())
    {
        return true;
    }
    ["artifacts", "outputs"].iter().any(|key| {
        result
            .get(*key)
            .and_then(JsonValue::as_array)
            .is_some_and(|entries| entries.iter().any(artifact_entry_has_storage_path))
    })
}

fn has_output_publication_target(parent_payload: &JsonValue) -> bool {
    parent_payload
        .get("output_publication")
        .and_then(JsonValue::as_object)
        .and_then(|publication| publication.get("target"))
        .is_some()
}

fn artifact_entry_has_storage_path(entry: &JsonValue) -> bool {
    ["storage_path", "path", "output_path"].iter().any(|key| {
        entry
            .get(*key)
            .and_then(JsonValue::as_str)
            .is_some_and(|path| !path.trim().is_empty())
    })
}

fn build_request_envelope(parent_payload: &JsonValue) -> JsonValue {
    let mut request = parent_payload
        .get("request")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    if !request.contains_key("operation")
        && let Some(operation) = parent_payload.get("operation").and_then(JsonValue::as_str)
    {
        request.insert(
            "operation".to_string(),
            JsonValue::String(operation.to_string()),
        );
    }
    if !request.contains_key("parameters")
        && let Some(parameters) = parent_payload.get("parameters")
    {
        request.insert("parameters".to_string(), parameters.clone());
    }
    JsonValue::Object(request)
}

fn move_execution_field(
    payload: &mut JsonMap<String, JsonValue>,
    execution: &mut JsonMap<String, JsonValue>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = payload.remove(source_key) {
        execution.entry(target_key.to_string()).or_insert(value);
    }
}

fn derive_primary_output_path(outputs: Option<&JsonValue>) -> Option<String> {
    let outputs = outputs?.as_array()?;
    let primary = outputs
        .iter()
        .find(|entry| {
            entry
                .get("primary")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| outputs.first())?;
    primary
        .get("storage_path")
        .or_else(|| primary.get("path"))
        .or_else(|| primary.get("output_path"))
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

fn build_execution_and_payload(
    run_id: &str,
    workflow_id: &str,
    status: &str,
    result_payload: &JsonValue,
) -> Result<(JsonValue, JsonValue)> {
    let mut payload = result_payload
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("collect: result payload must be a JSON object"))?;
    let mut execution = JsonMap::new();
    execution.insert("run_id".to_string(), JsonValue::String(run_id.to_string()));
    execution.insert("status".to_string(), JsonValue::String(status.to_string()));
    execution.insert(
        "workflow".to_string(),
        JsonValue::String(workflow_id.to_string()),
    );
    move_execution_field(&mut payload, &mut execution, "outputs", "outputs");
    move_execution_field(&mut payload, &mut execution, "artifacts", "outputs");
    move_execution_field(&mut payload, &mut execution, "output_path", "output_path");
    move_execution_field(
        &mut payload,
        &mut execution,
        "output_archive",
        "output_archive",
    );
    move_execution_field(&mut payload, &mut execution, "error", "error");
    move_execution_field(
        &mut payload,
        &mut execution,
        "execution_time_seconds",
        "execution_time_seconds",
    );
    move_execution_field(&mut payload, &mut execution, "batch_info", "batch_info");
    if !execution.contains_key("output_path")
        && let Some(path) = derive_primary_output_path(execution.get("outputs"))
    {
        execution.insert("output_path".to_string(), JsonValue::String(path));
    }
    Ok((JsonValue::Object(execution), JsonValue::Object(payload)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemberResultStatus {
    Succeeded,
    Failed,
    Cancelled,
}

fn member_result_status(result: &CollectedMemberResult) -> MemberResultStatus {
    let status = result
        .result
        .get("status")
        .and_then(JsonValue::as_str)
        .unwrap_or("succeeded")
        .trim()
        .to_ascii_lowercase();
    match status.as_str() {
        "failed" | "fail" | "error" => MemberResultStatus::Failed,
        "cancelled" | "canceled" => MemberResultStatus::Cancelled,
        _ => MemberResultStatus::Succeeded,
    }
}

fn count_results_with_status(
    results: &[CollectedMemberResult],
    desired: MemberResultStatus,
) -> usize {
    results
        .iter()
        .filter(|entry| member_result_status(entry) == desired)
        .count()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use anyhow::{Result, anyhow};
    mod test_support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    use scicomp_rq::StreamKey;
    use serde_json::{Value as JsonValue, json};
    use tokio::sync::Barrier;
    use tokio::sync::Mutex as TokioMutex;
    use uuid::Uuid;

    use super::*;
    use crate::config::InputStreamSpec;
    use crate::traits::{
        BoxFuture, MessageSink, RoleEnv, WorkerRole, is_message_deferred_error,
        is_message_ownership_lost_error,
    };
    use crate::transport::redis::RedisTransport;
    use test_support::spawn_test_queue_manager;

    static REDIS_TEST_LOCK: TokioMutex<()> = TokioMutex::const_new(());

    #[derive(Debug, Clone)]
    struct HandoffRecord {
        stream: String,
        payload: String,
        stage: String,
        run_id: String,
    }

    struct RecordingSink {
        handoffs: StdMutex<Vec<HandoffRecord>>,
    }

    struct BlockingGetGroupStore {
        inner: InMemoryCollectStore,
        get_group_barrier: Arc<Barrier>,
    }

    struct FailingOnceTerminalStore {
        inner: InMemoryParentRunStateStore,
        mark_attempts: StdMutex<usize>,
    }

    impl FailingOnceTerminalStore {
        fn new() -> Self {
            Self {
                inner: InMemoryParentRunStateStore::new(),
                mark_attempts: StdMutex::new(0),
            }
        }
    }

    impl BlockingGetGroupStore {
        fn new(get_group_barrier: Arc<Barrier>) -> Self {
            Self {
                inner: InMemoryCollectStore::new(),
                get_group_barrier,
            }
        }
    }

    impl ParentRunStateStore for FailingOnceTerminalStore {
        fn is_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<bool>> {
            self.inner.is_terminal(parent_run_id)
        }

        fn mark_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let should_fail = {
                    let mut attempts = self.mark_attempts.lock().unwrap();
                    *attempts += 1;
                    *attempts == 1
                };
                if should_fail {
                    return Err(anyhow!("transient collect terminal persistence failure"));
                }
                self.inner.mark_terminal(parent_run_id).await
            })
        }
    }

    impl CollectStore for BlockingGetGroupStore {
        fn init_group<'a>(
            &'a self,
            parent_run_id: &'a str,
            group: CollectGroup,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.init_group(parent_run_id, group)
        }

        fn discard_group<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
            self.inner.discard_group(parent_run_id)
        }

        fn add_result<'a>(
            &'a self,
            parent_run_id: &'a str,
            result: CollectedMemberResult,
        ) -> BoxFuture<'a, Result<CollectAppendResult>> {
            self.inner.add_result(parent_run_id, result)
        }

        fn get_group<'a>(
            &'a self,
            parent_run_id: &'a str,
        ) -> BoxFuture<'a, Result<Option<CollectGroup>>> {
            Box::pin(async move {
                let group = self.inner.get_group(parent_run_id).await?;
                let _ = tokio::time::timeout(
                    std::time::Duration::from_millis(100),
                    self.get_group_barrier.wait(),
                )
                .await;
                Ok(group)
            })
        }

        fn claim_finalization<'a>(
            &'a self,
            parent_run_id: &'a str,
        ) -> BoxFuture<'a, Result<Option<CollectFinalizationClaim>>> {
            self.inner.claim_finalization(parent_run_id)
        }

        fn is_finalization_committed<'a>(
            &'a self,
            parent_run_id: &'a str,
        ) -> BoxFuture<'a, Result<bool>> {
            self.inner.is_finalization_committed(parent_run_id)
        }

        fn commit_finalization<'a>(
            &'a self,
            claim: &'a CollectFinalizationClaim,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.commit_finalization(claim)
        }

        fn release_finalization_claim<'a>(
            &'a self,
            claim: &'a CollectFinalizationClaim,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.release_finalization_claim(claim)
        }
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                handoffs: StdMutex::new(Vec::new()),
            }
        }
    }

    #[derive(Default)]
    struct RecordingProgressPersistence {
        updates: StdMutex<Vec<CollectProgressUpdate>>,
        publication_skipped: StdMutex<Vec<String>>,
    }

    impl RecordingProgressPersistence {
        fn updates(&self) -> Vec<CollectProgressUpdate> {
            self.updates.lock().unwrap().clone()
        }

        fn publication_skipped(&self) -> Vec<String> {
            self.publication_skipped.lock().unwrap().clone()
        }
    }

    #[derive(Default)]
    struct FailingOncePublicationSkippedPersistence {
        attempts: StdMutex<usize>,
    }

    impl FailingOncePublicationSkippedPersistence {
        fn attempts(&self) -> usize {
            *self.attempts.lock().unwrap()
        }
    }

    impl CollectProgressPersistence for FailingOncePublicationSkippedPersistence {
        fn persist_progress<'a>(
            &'a self,
            _update: CollectProgressUpdate,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn persist_publication_skipped<'a>(
            &'a self,
            _parent_run_id: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let mut attempts = self.attempts.lock().unwrap();
                *attempts += 1;
                if *attempts == 1 {
                    return Err(anyhow!("transient skipped-publication persistence failure"));
                }
                Ok(())
            })
        }
    }

    impl CollectProgressPersistence for RecordingProgressPersistence {
        fn persist_progress<'a>(
            &'a self,
            update: CollectProgressUpdate,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.updates.lock().unwrap().push(update);
                Ok(())
            })
        }

        fn persist_publication_skipped<'a>(
            &'a self,
            parent_run_id: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.publication_skipped
                    .lock()
                    .unwrap()
                    .push(parent_run_id.to_string());
                Ok(())
            })
        }
    }

    impl MessageSink for RecordingSink {
        fn enqueue<'a>(
            &'a self,
            _stream: &'a str,
            _run_id: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("collect tests do not use enqueue")) })
        }

        fn ack_message<'a>(&'a self, _msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            self.handoff_to_run(msg, dest_stream, payload, stage, msg.run_id())
        }

        fn handoff_to_run<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
            run_id: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.handoffs.lock().unwrap().push(HandoffRecord {
                    stream: dest_stream.to_string(),
                    payload: payload.to_string(),
                    stage: stage.to_string(),
                    run_id: run_id.to_string(),
                });
                Ok("1-0".to_string())
            })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Err(anyhow!("collect tests do not use forward_many")) })
        }
    }

    fn collect_env() -> RoleEnv {
        RoleEnv {
            role_name: "collect".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "collect".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec!["results".to_string()],
            role_config: None,
            python_runtime_envs: Default::default(),
        }
    }

    fn collect_msg(run_id: &str, item_index: u64, status: &str) -> scicomp_rq::Message {
        collect_msg_with_policy(run_id, item_index, status, None)
    }

    fn collect_msg_with_policy(
        run_id: &str,
        item_index: u64,
        status: &str,
        failure_policy: Option<&str>,
    ) -> scicomp_rq::Message {
        let payload = json!({
            "run_id": run_id,
            "parent_run_id": "parent-run",
            "workflow_id": "demo-fanout",
            "fanout_profile": {
                "item_count": 2,
                "aggregation_mode": "all_members",
                "failure_policy": failure_policy
            },
            "fanout_item": {
                "item_index": item_index,
                "member_seed": 1000 + item_index
            },
            "result": {
                "run_id": run_id,
                "status": status,
                "artifacts": [],
                "member_value": item_index
            },
            "stage_context": {
                "current_stage_id": "collect",
                "current_phase": "collect",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "fanout"},
                    {"id": "fanout", "phase": "fanout", "queue": "fanout", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "collect"},
                    {"id": "collect", "phase": "collect", "queue": "collect", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
        .to_string();
        scicomp_rq::Message::new(
            "test:collect",
            "collect:grp",
            "1-0",
            run_id,
            &payload,
            "collect",
        )
    }

    fn parent_payload() -> JsonValue {
        json!({
            "run_id": "parent-run",
            "workflow_id": "demo-fanout",
            "operation": "run",
            "fanout_profile": {
                "item_count": 2,
                "aggregation_mode": "all_members"
            },
            "stage_context": {
                "current_stage_id": "collect",
                "current_phase": "collect",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "fanout"},
                    {"id": "fanout", "phase": "fanout", "queue": "fanout", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "collect"},
                    {"id": "collect", "phase": "collect", "queue": "collect", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
    }

    fn parent_payload_with_publish() -> JsonValue {
        json!({
            "run_id": "parent-run",
            "workflow_id": "demo-fanout",
            "operation": "run",
            "fanout_profile": {
                "item_count": 2,
                "aggregation_mode": "all_members"
            },
            "output_publication": {
                "target": {
                    "artifact": "primary",
                    "provider": "s3",
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "outputs/demo-fanout/parent-run"
                    }
                }
            },
            "stage_context": {
                "current_stage_id": "collect",
                "current_phase": "collect",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "fanout"},
                    {"id": "fanout", "phase": "fanout", "queue": "fanout", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "collect"},
                    {"id": "collect", "phase": "collect", "queue": "collect", "next": "publish"},
                    {"id": "publish", "phase": "publish", "queue": "publish", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
    }

    fn parent_payload_with_publish_stage_only() -> JsonValue {
        let mut payload = parent_payload_with_publish();
        payload
            .as_object_mut()
            .expect("parent payload should be an object")
            .remove("output_publication");
        payload
    }

    #[tokio::test]
    async fn collect_role_emits_parent_result_when_last_member_arrives() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store,
            progress.clone(),
            terminal,
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        assert!(sink.handoffs.lock().unwrap().is_empty());
        let updates = progress.updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].parent_run_id, "parent-run");
        assert_eq!(updates[0].expected_count, 2);
        assert_eq!(updates[0].collected_count, 1);
        assert_eq!(updates[0].succeeded_count, 1);
        assert_eq!(updates[0].failed_count, 0);
        assert_eq!(updates[0].cancelled_count, 0);
        assert_eq!(updates[0].child_run_ids, vec!["parent-run:item:0"]);

        role.handle(
            &collect_msg("parent-run:item:1", 1, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("second member should flush");

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "results");
        assert_eq!(handoffs[0].stage, "results");
        assert_eq!(handoffs[0].run_id, "parent-run");

        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        assert_eq!(forwarded["run_id"], "parent-run");
        assert_eq!(forwarded["status"], "succeeded");
        assert_eq!(forwarded["payload"]["aggregation_summary"]["item_count"], 2);
        assert_eq!(
            forwarded["payload"]["aggregation_summary"]["succeeded_count"],
            2
        );
        assert_eq!(
            forwarded["payload"]["child_results"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let updates = progress.updates();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[1].collected_count, 2);
        assert_eq!(updates[1].succeeded_count, 2);
        assert_eq!(updates[1].failed_count, 0);
        assert_eq!(
            updates[1].child_run_ids,
            vec!["parent-run:item:0", "parent-run:item:1"]
        );
    }

    #[tokio::test]
    async fn collect_role_handoffs_structured_parent_results_envelope() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        let mut parent_payload = parent_payload();
        let parent_map = parent_payload
            .as_object_mut()
            .expect("parent payload should be an object");
        parent_map.insert(
            "operation".to_string(),
            JsonValue::String("fanout_run".to_string()),
        );
        parent_map.insert(
            "parameters".to_string(),
            json!({
                "member_count": 2
            }),
        );
        parent_map.insert(
            "request".to_string(),
            json!({
                "content_type": "application/json",
                "raw_fields": {
                    "member_count": 2
                },
                "input_artifacts": []
            }),
        );
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload,
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) =
            CollectRole::from_env_with_store(&collect_env(), shared_store, progress, terminal)
                .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        role.handle(
            &collect_msg("parent-run:item:1", 1, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("second member should flush");

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        assert_eq!(forwarded["request"]["operation"], "fanout_run");
        assert_eq!(forwarded["request"]["content_type"], "application/json");
        assert_eq!(forwarded["request"]["parameters"]["member_count"], 2);
        assert_eq!(forwarded["execution"]["run_id"], "parent-run");
        assert_eq!(forwarded["execution"]["status"], "succeeded");
        assert_eq!(forwarded["execution"]["workflow"], "demo-fanout");
        assert_eq!(forwarded["payload"]["aggregation_summary"]["item_count"], 2);
        assert_eq!(
            forwarded["payload"]["child_results"]
                .as_array()
                .expect("child results should be present")
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn collect_role_skips_publish_for_aggregate_only_parent_result() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload_with_publish(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store,
            progress.clone(),
            terminal,
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        role.handle(
            &collect_msg("parent-run:item:1", 1, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("second member should flush to results");

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "results");
        assert_eq!(handoffs[0].stage, "results");
        assert_eq!(handoffs[0].run_id, "parent-run");

        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        assert_eq!(forwarded["run_id"], "parent-run");
        assert_eq!(forwarded["status"], "succeeded");
        assert_eq!(
            forwarded["payload"]["aggregation_summary"]["succeeded_count"],
            2
        );
        assert_eq!(
            forwarded["payload"]["child_results"]
                .as_array()
                .expect("child results should be present")
                .len(),
            2
        );
        assert_eq!(
            progress.publication_skipped(),
            vec!["parent-run".to_string()]
        );
    }

    #[tokio::test]
    async fn collect_role_keeps_declared_publish_stage_when_publication_is_disabled() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload_with_publish_stage_only(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store,
            progress.clone(),
            terminal,
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        role.handle(
            &collect_msg("parent-run:item:1", 1, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("second member should flush to publish");

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "publish");
        assert_eq!(handoffs[0].stage, "publish");
        assert_eq!(handoffs[0].run_id, "parent-run");
        assert!(progress.publication_skipped().is_empty());
    }

    #[tokio::test]
    async fn collect_role_retains_group_when_skipped_publication_persistence_fails() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(FailingOncePublicationSkippedPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload_with_publish(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store.clone(),
            progress.clone(),
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        let final_msg = collect_msg("parent-run:item:1", 1, "succeeded");
        let error = role
            .handle(&final_msg, "collect", &sink)
            .await
            .expect_err("first skipped-publication persistence attempt should fail");

        assert!(
            error
                .to_string()
                .contains("transient skipped-publication persistence failure")
        );
        assert!(sink.handoffs.lock().unwrap().is_empty());
        let retained = shared_store
            .get_group("parent-run")
            .await
            .expect("group lookup should succeed")
            .expect("group should be retained for retry");
        assert_eq!(retained.results.len(), 2);

        role.handle(&final_msg, "collect", &sink)
            .await
            .expect("retry should flush retained group");

        assert_eq!(progress.attempts(), 2);
        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "results");
        assert!(terminal.is_terminal("parent-run").await.unwrap());
        assert!(
            shared_store
                .get_group("parent-run")
                .await
                .expect("group lookup should succeed")
                .is_none(),
            "group should be discarded after successful handoff"
        );
    }

    #[tokio::test]
    async fn collect_role_fail_fast_emits_parent_result_on_first_failed_member() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store,
            progress.clone(),
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg_with_policy("parent-run:item:0", 0, "failed", Some("fail_fast")),
            "collect",
            &sink,
        )
        .await
        .expect("first failed member should fail-fast flush");

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "results");
        assert_eq!(handoffs[0].run_id, "parent-run");
        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        assert_eq!(forwarded["payload"]["aggregation_summary"]["item_count"], 2);
        assert_eq!(
            forwarded["payload"]["aggregation_summary"]["collected_count"],
            1
        );
        assert_eq!(
            forwarded["payload"]["aggregation_summary"]["failed_count"],
            1
        );
        assert_eq!(forwarded["status"], "failed");
        assert!(terminal.is_terminal("parent-run").await.unwrap());
        let updates = progress.updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].succeeded_count, 0);
        assert_eq!(updates[0].failed_count, 1);
        assert_eq!(updates[0].cancelled_count, 0);
        assert_eq!(updates[0].child_run_ids, vec!["parent-run:item:0"]);

        role.handle(
            &collect_msg_with_policy("parent-run:item:1", 1, "succeeded", Some("fail_fast")),
            "collect",
            &sink,
        )
        .await
        .expect("late member after fail-fast should be ignored");
        assert_eq!(sink.handoffs.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn collect_role_claims_finalization_before_handoff() {
        let shared_store: Arc<dyn CollectStore> =
            Arc::new(BlockingGetGroupStore::new(Arc::new(Barrier::new(2))));
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: vec![
                        CollectedMemberResult {
                            child_run_id: "parent-run:item:0".to_string(),
                            item_index: 0,
                            fanout_item: json!({
                                "item_index": 0,
                                "member_seed": 1000
                            }),
                            result: json!({
                                "run_id": "parent-run:item:0",
                                "status": "succeeded",
                                "artifacts": [],
                                "member_value": 0
                            }),
                        },
                        CollectedMemberResult {
                            child_run_id: "parent-run:item:1".to_string(),
                            item_index: 1,
                            fanout_item: json!({
                                "item_index": 1,
                                "member_seed": 1001
                            }),
                            result: json!({
                                "run_id": "parent-run:item:1",
                                "status": "succeeded",
                                "artifacts": [],
                                "member_value": 1
                            }),
                        },
                    ],
                },
            )
            .await
            .unwrap();
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store,
            progress,
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();
        let final_msg_a = collect_msg("parent-run:item:1", 1, "succeeded");
        let final_msg_b = collect_msg("parent-run:item:1", 1, "succeeded");

        let (first_result, second_result) = tokio::join!(
            role.handle(&final_msg_a, "collect", &sink),
            role.handle(&final_msg_b, "collect", &sink)
        );
        let results = [first_result, second_result];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let deferred = results
            .iter()
            .find_map(|result| result.as_ref().err())
            .expect("one duplicate final child should defer");
        assert!(is_message_deferred_error(deferred));

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(
            handoffs.len(),
            1,
            "concurrent duplicate final child deliveries must not hand off the parent twice"
        );
        assert!(terminal.is_terminal("parent-run").await.unwrap());
    }

    #[tokio::test]
    async fn collect_role_defers_while_another_finalizer_owns_the_claim() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(InMemoryParentRunStateStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: vec![
                        CollectedMemberResult {
                            child_run_id: "parent-run:item:0".to_string(),
                            item_index: 0,
                            fanout_item: json!({"item_index": 0}),
                            result: json!({"status": "succeeded"}),
                        },
                        CollectedMemberResult {
                            child_run_id: "parent-run:item:1".to_string(),
                            item_index: 1,
                            fanout_item: json!({"item_index": 1}),
                            result: json!({"status": "succeeded"}),
                        },
                    ],
                },
            )
            .await
            .expect("collect group should initialize");
        let active_claim = shared_store
            .claim_finalization("parent-run")
            .await
            .expect("claim lookup should succeed")
            .expect("another worker should acquire finalization");
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store.clone(),
            progress,
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();
        let final_msg = collect_msg("parent-run:item:1", 1, "succeeded");

        let error = role
            .handle(&final_msg, "collect", &sink)
            .await
            .expect_err("active finalization ownership must defer the retry trigger");
        assert!(is_message_deferred_error(&error));
        assert!(
            error
                .to_string()
                .contains("finalization is already in progress")
        );
        assert!(sink.handoffs.lock().unwrap().is_empty());

        shared_store
            .release_finalization_claim(&active_claim)
            .await
            .expect("expired owner should release the claim");
        role.handle(&final_msg, "collect", &sink)
            .await
            .expect("retry should take over and finalize");

        assert_eq!(sink.handoffs.lock().unwrap().len(), 1);
        assert!(terminal.is_terminal("parent-run").await.unwrap());
    }

    #[tokio::test]
    async fn collect_role_does_not_reopen_finalization_after_committed_handoff() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(FailingOnceTerminalStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("collect group should initialize");
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store.clone(),
            progress,
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &collect_msg("parent-run:item:0", 0, "succeeded"),
            "collect",
            &sink,
        )
        .await
        .expect("first member should buffer");
        let final_msg = collect_msg("parent-run:item:1", 1, "succeeded");
        let error = role
            .handle(&final_msg, "collect", &sink)
            .await
            .expect_err("cleanup failure after handoff should surface");
        assert!(
            error
                .to_string()
                .contains("transient collect terminal persistence failure")
        );
        assert_eq!(
            sink.handoffs.lock().unwrap().len(),
            1,
            "first attempt should hand off the parent before cleanup fails"
        );

        role.handle(&final_msg, "collect", &sink)
            .await
            .expect("retry after committed handoff should be ignored");
        assert_eq!(
            sink.handoffs.lock().unwrap().len(),
            1,
            "retry must not duplicate the already committed parent handoff"
        );
        assert!(
            terminal.is_terminal("parent-run").await.unwrap(),
            "retry must finish the terminal marker after the handoff committed"
        );
        assert!(
            shared_store
                .get_group("parent-run")
                .await
                .expect("group lookup should succeed")
                .is_none(),
            "retry must delete retained group data after the handoff committed"
        );
    }

    #[tokio::test]
    async fn collect_role_recovers_fail_fast_cleanup_from_later_successful_child() {
        let shared_store: Arc<dyn CollectStore> = Arc::new(InMemoryCollectStore::new());
        let progress = Arc::new(RecordingProgressPersistence::default());
        let terminal = Arc::new(FailingOnceTerminalStore::new());
        shared_store
            .init_group(
                "parent-run",
                CollectGroup {
                    parent_run_id: "parent-run".to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("collect group should initialize");
        let (role, tasks) = CollectRole::from_env_with_store(
            &collect_env(),
            shared_store.clone(),
            progress,
            terminal.clone(),
        )
        .expect("collect role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        let error = role
            .handle(
                &collect_msg_with_policy("parent-run:item:0", 0, "failed", Some("fail_fast")),
                "collect",
                &sink,
            )
            .await
            .expect_err("cleanup failure after fail-fast handoff should surface");
        assert!(
            error
                .to_string()
                .contains("transient collect terminal persistence failure")
        );
        assert_eq!(sink.handoffs.lock().unwrap().len(), 1);

        role.handle(
            &collect_msg_with_policy("parent-run:item:1", 1, "succeeded", Some("fail_fast")),
            "collect",
            &sink,
        )
        .await
        .expect("later successful child should finish committed cleanup");

        assert_eq!(
            sink.handoffs.lock().unwrap().len(),
            1,
            "cleanup recovery must not duplicate the committed parent handoff"
        );
        assert!(
            terminal.is_terminal("parent-run").await.unwrap(),
            "later child must finish the terminal marker"
        );
        assert!(
            shared_store
                .get_group("parent-run")
                .await
                .expect("group lookup should succeed")
                .is_none(),
            "later child must delete retained group data"
        );
    }

    #[tokio::test]
    async fn redis_collect_path_keeps_child_run_ids_for_status_tracking() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) = spawn_test_queue_manager("redis-collect-child-ids").await;
        let parent_run_id = "parent-run";
        let store = RedisCollectStore::new(qm.clone(), format!("collect:test:{}:", Uuid::new_v4()));
        store
            .init_group(
                parent_run_id,
                CollectGroup {
                    parent_run_id: parent_run_id.to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("redis collect group should initialize");

        let append_one = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status":"succeeded"}),
                },
            )
            .await
            .expect("first child result should append");
        assert_eq!(
            append_one.child_run_ids,
            vec!["parent-run:item:0".to_string()]
        );

        let append_two = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:1".to_string(),
                    item_index: 1,
                    fanout_item: json!({}),
                    result: json!({"status":"failed"}),
                },
            )
            .await
            .expect("second child result should append");
        assert_eq!(
            append_two.child_run_ids,
            vec![
                "parent-run:item:0".to_string(),
                "parent-run:item:1".to_string()
            ]
        );

        let persistence =
            RedisCollectProgressPersistence::new(qm.clone(), DEFAULT_COLLECT_STORE_PREFIX);
        persistence
            .persist_progress(CollectProgressUpdate {
                parent_run_id: parent_run_id.to_string(),
                workflow_id: "demo-fanout".to_string(),
                expected_count: 2,
                collected_count: append_two.item_count,
                succeeded_count: append_two.succeeded_count,
                failed_count: append_two.failed_count,
                cancelled_count: append_two.cancelled_count,
                child_run_ids: append_two.child_run_ids.clone(),
            })
            .await
            .expect("progress snapshot with child IDs should persist");

        let run_key = format!("{RUN_KEY_PREFIX}{parent_run_id}");
        let mut conn = qm.connection();
        let child_run_ids_raw: Option<String> = redis::cmd("HGET")
            .arg(&run_key)
            .arg("fanout_child_run_ids")
            .query_async(&mut conn)
            .await
            .expect("fanout_child_run_ids should be queryable");
        let child_run_ids_raw = child_run_ids_raw.expect("fanout_child_run_ids should be present");
        let persisted_ids: Vec<String> = serde_json::from_str(&child_run_ids_raw)
            .expect("fanout_child_run_ids should be valid JSON");
        assert_eq!(persisted_ids, append_two.child_run_ids);

        persistence
            .persist_progress(CollectProgressUpdate {
                parent_run_id: parent_run_id.to_string(),
                workflow_id: "demo-fanout".to_string(),
                expected_count: 2,
                collected_count: append_two.item_count,
                succeeded_count: append_two.succeeded_count,
                failed_count: append_two.failed_count,
                cancelled_count: append_two.cancelled_count,
                child_run_ids: Vec::new(),
            })
            .await
            .expect("empty child IDs progress should persist");
        let child_run_ids_after_clear: Option<String> = redis::cmd("HGET")
            .arg(&run_key)
            .arg("fanout_child_run_ids")
            .query_async(&mut conn)
            .await
            .expect("fanout_child_run_ids should be queryable after clear");
        assert!(
            child_run_ids_after_clear.is_none(),
            "empty child IDs update should clear stale fanout_child_run_ids"
        );
    }

    #[tokio::test]
    async fn in_memory_collect_store_rejects_results_after_finalization_is_claimed() {
        let store = InMemoryCollectStore::new();
        let parent_run_id = "parent-run";
        store
            .init_group(
                parent_run_id,
                CollectGroup {
                    parent_run_id: parent_run_id.to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("collect group should initialize");
        let _claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("finalization claim should succeed")
            .expect("finalization should not already be claimed");

        let append = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status":"succeeded"}),
                },
            )
            .await
            .expect("a frozen collect group should reject without failing");

        assert!(!append.appended);
        assert_eq!(append.item_count, 0);
        let group = store
            .get_group(parent_run_id)
            .await
            .expect("collect group should load")
            .expect("collect group should still exist");
        assert!(group.results.is_empty());
    }

    #[tokio::test]
    async fn redis_collect_store_rejects_results_and_progress_after_finalization_is_claimed() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) =
            spawn_test_queue_manager("redis-collect-finalization-freeze").await;
        let parent_run_id = "parent-run";
        let store = RedisCollectStore::new(qm.clone(), DEFAULT_COLLECT_STORE_PREFIX);
        store
            .init_group(
                parent_run_id,
                CollectGroup {
                    parent_run_id: parent_run_id.to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("redis collect group should initialize");
        let append_before_claim = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status":"succeeded"}),
                },
            )
            .await
            .expect("first result should append before finalization");
        let claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("finalization claim should succeed")
            .expect("finalization should not already be claimed");

        let append_after_claim = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:1".to_string(),
                    item_index: 1,
                    fanout_item: json!({}),
                    result: json!({"status":"failed"}),
                },
            )
            .await
            .expect("a frozen redis collect group should reject without failing");
        assert!(!append_after_claim.appended);
        assert_eq!(append_after_claim.item_count, 1);

        let run_key = format!("{RUN_KEY_PREFIX}{parent_run_id}");
        let mut conn = qm.connection();
        let _: usize = redis::cmd("HSET")
            .arg(&run_key)
            .arg("status")
            .arg("succeeded")
            .arg("stage")
            .arg("results")
            .arg("fanout_collected_count")
            .arg("99")
            .query_async(&mut conn)
            .await
            .expect("downstream run status should be writable");

        RedisCollectProgressPersistence::new(qm.clone(), DEFAULT_COLLECT_STORE_PREFIX)
            .persist_progress(CollectProgressUpdate {
                parent_run_id: parent_run_id.to_string(),
                workflow_id: "demo-fanout".to_string(),
                expected_count: 2,
                collected_count: append_before_claim.item_count,
                succeeded_count: append_before_claim.succeeded_count,
                failed_count: append_before_claim.failed_count,
                cancelled_count: append_before_claim.cancelled_count,
                child_run_ids: append_before_claim.child_run_ids,
            })
            .await
            .expect("stale progress persistence should be ignored");

        let (status, stage, collected_count): (String, String, String) = redis::cmd("HMGET")
            .arg(&run_key)
            .arg("status")
            .arg("stage")
            .arg("fanout_collected_count")
            .query_async(&mut conn)
            .await
            .expect("run status should be readable");
        assert_eq!(status, "succeeded");
        assert_eq!(stage, "results");
        assert_eq!(collected_count, "99");

        store
            .commit_finalization(&claim)
            .await
            .expect("current owner should commit finalization");
        let append_after_commit = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:2".to_string(),
                    item_index: 2,
                    fanout_item: json!({}),
                    result: json!({"status":"failed"}),
                },
            )
            .await
            .expect("a committed redis collect group should reject without failing");
        assert!(!append_after_commit.appended);
        assert_eq!(append_after_commit.item_count, 1);
    }

    #[tokio::test]
    async fn redis_collect_progress_stays_frozen_after_committed_group_cleanup() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) =
            spawn_test_queue_manager("redis-collect-progress-after-cleanup").await;
        let parent_run_id = "parent-run";
        let store = RedisCollectStore::new(qm.clone(), DEFAULT_COLLECT_STORE_PREFIX);
        store
            .init_group(
                parent_run_id,
                CollectGroup {
                    parent_run_id: parent_run_id.to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("redis collect group should initialize");
        let stale_snapshot = store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status":"succeeded"}),
                },
            )
            .await
            .expect("result should append before finalization");
        let claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("finalization claim should succeed")
            .expect("finalization should not already be claimed");
        store
            .commit_finalization(&claim)
            .await
            .expect("current owner should commit finalization");
        store
            .discard_group(parent_run_id)
            .await
            .expect("completed group data should be discarded");

        let run_key = format!("{RUN_KEY_PREFIX}{parent_run_id}");
        let mut conn = qm.connection();
        let _: usize = redis::cmd("HSET")
            .arg(&run_key)
            .arg("status")
            .arg("succeeded")
            .arg("stage")
            .arg("results")
            .arg("fanout_collected_count")
            .arg("99")
            .query_async(&mut conn)
            .await
            .expect("terminal run status should be writable");

        RedisCollectProgressPersistence::new(qm.clone(), DEFAULT_COLLECT_STORE_PREFIX)
            .persist_progress(CollectProgressUpdate {
                parent_run_id: parent_run_id.to_string(),
                workflow_id: "demo-fanout".to_string(),
                expected_count: 2,
                collected_count: stale_snapshot.item_count,
                succeeded_count: stale_snapshot.succeeded_count,
                failed_count: stale_snapshot.failed_count,
                cancelled_count: stale_snapshot.cancelled_count,
                child_run_ids: stale_snapshot.child_run_ids,
            })
            .await
            .expect("stale progress persistence should be ignored after cleanup");

        let (status, stage, collected_count): (String, String, String) = redis::cmd("HMGET")
            .arg(&run_key)
            .arg("status")
            .arg("stage")
            .arg("fanout_collected_count")
            .query_async(&mut conn)
            .await
            .expect("terminal run status should be readable");
        assert_eq!(status, "succeeded");
        assert_eq!(stage, "results");
        assert_eq!(collected_count, "99");
        let tombstone_ttl: i64 = redis::cmd("TTL")
            .arg(&claim.state_key)
            .query_async(&mut conn)
            .await
            .expect("committed finalization tombstone ttl should be readable");
        assert!(tombstone_ttl > 0);
    }

    #[tokio::test]
    async fn redis_collect_finalization_claim_is_single_owner_and_releasable() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) =
            spawn_test_queue_manager("redis-collect-finalization-claim").await;
        let parent_run_id = "parent-run";
        let store = RedisCollectStore::new(qm.clone(), format!("collect:test:{}:", Uuid::new_v4()));

        let first_claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("first finalization claim should succeed")
            .expect("first finalization claim should be acquired");
        assert!(
            store
                .claim_finalization(parent_run_id)
                .await
                .expect("second finalization claim should be rejected")
                .is_none()
        );

        let mut conn = qm.connection();
        let ttl_secs: i64 = redis::cmd("TTL")
            .arg(store.finalization_key(parent_run_id))
            .query_async(&mut conn)
            .await
            .expect("finalization claim ttl should be readable");
        assert!(
            ttl_secs > 0,
            "collect finalization claims should expire if a worker exits before release"
        );

        store
            .release_finalization_claim(&first_claim)
            .await
            .expect("finalization claim should release");
        assert!(
            store
                .claim_finalization(parent_run_id)
                .await
                .expect("released finalization claim should be claimable again")
                .is_some()
        );
    }

    #[tokio::test]
    async fn redis_collect_stale_owner_cannot_commit_or_release_new_claim() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) =
            spawn_test_queue_manager("redis-collect-finalization-owner-fence").await;
        let parent_run_id = "parent-run";
        let store = RedisCollectStore::new(qm.clone(), format!("collect:test:{}", Uuid::new_v4()));

        let first_claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("first finalization claim should succeed")
            .expect("first owner should acquire finalization");
        let mut conn = qm.connection();
        let _: bool = redis::cmd("PEXPIRE")
            .arg(&first_claim.state_key)
            .arg(20_i64)
            .query_async(&mut conn)
            .await
            .expect("test should shorten first claim ttl");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let second_claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("expired finalization should be claimable")
            .expect("second owner should acquire finalization");
        assert_ne!(first_claim.owner_token, second_claim.owner_token);

        assert!(
            store.commit_finalization(&first_claim).await.is_err(),
            "a stale owner must not commit a replacement owner's claim"
        );
        store
            .release_finalization_claim(&first_claim)
            .await
            .expect("stale release should be a harmless no-op");
        let marker: String = redis::cmd("GET")
            .arg(&second_claim.state_key)
            .query_async(&mut conn)
            .await
            .expect("replacement claim should remain readable");
        assert_eq!(marker, second_claim.owner_token);

        store
            .commit_finalization(&second_claim)
            .await
            .expect("current owner should commit finalization");
        store
            .release_finalization_claim(&second_claim)
            .await
            .expect("release after commit should be a harmless no-op");
        let committed_marker: String = redis::cmd("GET")
            .arg(&second_claim.state_key)
            .query_async(&mut conn)
            .await
            .expect("committed marker should remain readable");
        assert_eq!(
            committed_marker,
            format!("committed:{}", second_claim.owner_token)
        );
    }

    #[tokio::test]
    async fn redis_collect_handoff_and_finalization_commit_are_atomic_and_owner_fenced() {
        let _lock = REDIS_TEST_LOCK.lock().await;
        let (_redis_server, qm) =
            spawn_test_queue_manager("redis-collect-atomic-finalization-handoff").await;
        let parent_run_id = "parent-run";
        let source_stream = StreamKey::new("test:collect");
        let destination_stream = "test:results";
        let group = "collect:grp";
        qm.create_consumer_group(&source_stream, group, "0", true)
            .await
            .expect("collect consumer group should exist");
        qm.enqueue_to_stream(
            &source_stream,
            "parent-run:item:1",
            r#"{"child":true}"#,
            "collect",
        )
        .await
        .expect("collect child should enqueue");
        let msg = qm
            .read_messages(&source_stream, group, "consumer-a", 1, 1)
            .await
            .expect("collect child should be readable")
            .into_iter()
            .next()
            .expect("one collect child should be pending");

        let store = RedisCollectStore::new(qm.clone(), format!("collect:test:{}", Uuid::new_v4()));
        store
            .init_group(
                parent_run_id,
                CollectGroup {
                    parent_run_id: parent_run_id.to_string(),
                    workflow_id: "demo-fanout".to_string(),
                    parent_payload: parent_payload(),
                    expected_count: 2,
                    results: Vec::new(),
                },
            )
            .await
            .expect("collect group should initialize");
        store
            .add_result(
                parent_run_id,
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status":"succeeded"}),
                },
            )
            .await
            .expect("collect state should exist before guarded handoff");
        let first_claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("first finalization claim should succeed")
            .expect("first owner should acquire finalization");
        let mut conn = qm.connection();
        let _: bool = redis::cmd("PEXPIRE")
            .arg(&first_claim.state_key)
            .arg(20_i64)
            .query_async(&mut conn)
            .await
            .expect("test should shorten first claim ttl");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let second_claim = store
            .claim_finalization(parent_run_id)
            .await
            .expect("expired finalization should be claimable")
            .expect("second owner should acquire finalization");

        let sink = RedisTransport::new(qm.clone(), "test:");
        let stale_handoff = sink
            .handoff_to_run_and_commit_for_consumer(
                &msg,
                "results",
                r#"{"parent":true}"#,
                "results",
                parent_run_id,
                &first_claim.state_key,
                &first_claim.owner_token,
                &first_claim.recovery_keys,
                "consumer-a",
            )
            .await;
        assert!(stale_handoff.is_err());
        let destination_count: usize = redis::cmd("XLEN")
            .arg(destination_stream)
            .query_async(&mut conn)
            .await
            .expect("destination stream length should be readable");
        assert_eq!(destination_count, 0);
        let marker: String = redis::cmd("GET")
            .arg(&second_claim.state_key)
            .query_async(&mut conn)
            .await
            .expect("replacement claim should remain readable");
        assert_eq!(marker, second_claim.owner_token);

        let (_cursor, claimed) = qm
            .claim_idle_messages(&source_stream, group, "consumer-b", 0, "0-0", 1)
            .await
            .expect("second consumer should reclaim the source message");
        assert_eq!(claimed.len(), 1);
        let stale_consumer_handoff = sink
            .handoff_to_run_and_commit_for_consumer(
                &msg,
                "results",
                r#"{"parent":true}"#,
                "results",
                parent_run_id,
                &second_claim.state_key,
                &second_claim.owner_token,
                &second_claim.recovery_keys,
                "consumer-a",
            )
            .await;
        let stale_consumer_error = stale_consumer_handoff
            .expect_err("the prior consumer must not hand off a reclaimed message");
        assert!(is_message_ownership_lost_error(&stale_consumer_error));
        assert!(
            qm.renew_message_lease(&msg, "consumer-b").await.unwrap(),
            "failed stale handoff must leave the source owned by the reclaiming consumer"
        );

        sink.handoff_to_run_and_commit_for_consumer(
            &msg,
            "results",
            r#"{"parent":true}"#,
            "results",
            parent_run_id,
            &second_claim.state_key,
            &second_claim.owner_token,
            &second_claim.recovery_keys,
            "consumer-b",
        )
        .await
        .expect("current owner should atomically hand off and commit");
        let destination_count: usize = redis::cmd("XLEN")
            .arg(destination_stream)
            .query_async(&mut conn)
            .await
            .expect("destination stream length should be readable");
        assert_eq!(destination_count, 1);
        assert_eq!(
            qm.ack_message(&msg)
                .await
                .expect("source acknowledgement should be queryable"),
            0,
            "atomic handoff should already acknowledge the source child"
        );
        let (marker, stage): (String, String) = redis::pipe()
            .cmd("GET")
            .arg(&second_claim.state_key)
            .cmd("HGET")
            .arg(format!("{RUN_KEY_PREFIX}{parent_run_id}"))
            .arg("stage")
            .query_async(&mut conn)
            .await
            .expect("committed marker and downstream stage should be readable");
        assert_eq!(marker, format!("committed:{}", second_claim.owner_token));
        assert_eq!(stage, "results");
        let marker_ttl_ms: i64 = redis::cmd("PTTL")
            .arg(&second_claim.state_key)
            .query_async(&mut conn)
            .await
            .expect("committed marker ttl should be readable");
        assert!(
            marker_ttl_ms > 0,
            "guarded handoff must leave a bounded recovery marker, got PTTL={marker_ttl_ms}"
        );
        for key in [
            store.meta_key(parent_run_id),
            store.items_key(parent_run_id),
            store.runids_key(parent_run_id),
            store.stats_key(parent_run_id),
        ] {
            let ttl_ms: i64 = redis::cmd("PTTL")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .expect("collect recovery key ttl should be readable");
            assert!(
                ttl_ms > 0,
                "guarded handoff must bound retained collect key '{key}', got PTTL={ttl_ms}"
            );
        }
    }
}
