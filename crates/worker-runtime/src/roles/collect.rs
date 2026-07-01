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
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

pub(crate) const DEFAULT_COLLECT_STORE_PREFIX: &str = "collect";
const RUN_KEY_PREFIX: &str = "run:";

const APPEND_MEMBER_RESULT_LUA: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 then
  return {0, 0}
end
local added = redis.call('SADD', KEYS[3], ARGV[1])
if added == 1 then
  redis.call('RPUSH', KEYS[2], ARGV[2])
end
local size = redis.call('LLEN', KEYS[2])
return {added, size}
"#;

const POP_COLLECT_GROUP_LUA: &str = r#"
local meta = redis.call('GET', KEYS[1])
if not meta then
  return {}
end
local items = redis.call('LRANGE', KEYS[2], 0, -1)
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
table.insert(items, 1, meta)
return items
"#;

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

    fn take_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>>;
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
}

pub(crate) struct NoopCollectProgressPersistence;

impl CollectProgressPersistence for NoopCollectProgressPersistence {
    fn persist_progress<'a>(&'a self, _update: CollectProgressUpdate) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

pub(crate) struct RedisCollectProgressPersistence {
    qm: QueueManager,
}

impl RedisCollectProgressPersistence {
    pub(crate) fn new(qm: QueueManager) -> Self {
        Self { qm }
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
            let mut conn = self.qm.connection();
            let mut hset = redis::cmd("HSET");
            hset.arg(&run_key)
                .arg("status")
                .arg("running")
                .arg("stage")
                .arg("executing")
                .arg("updated_at")
                .arg(&now_secs)
                .arg("workflow")
                .arg(update.workflow_id)
                .arg("fanout_expected_count")
                .arg(update.expected_count.to_string())
                .arg("fanout_collected_count")
                .arg(update.collected_count.to_string())
                .arg("fanout_succeeded_count")
                .arg(update.succeeded_count.to_string())
                .arg("fanout_failed_count")
                .arg(update.failed_count.to_string())
                .arg("fanout_cancelled_count")
                .arg(update.cancelled_count.to_string());
            if !update.child_run_ids.is_empty() {
                hset.arg("fanout_child_run_ids").arg(
                    serde_json::to_string(&update.child_run_ids)
                        .context("collect: failed to encode child run ids")?,
                );
            }
            let _: usize = hset
                .query_async(&mut conn)
                .await
                .context("collect: failed to persist fanout progress hash fields")?;
            if update.child_run_ids.is_empty() {
                let _: i64 = hash_ops::hdel(&mut conn, &run_key, "fanout_child_run_ids")
                    .await
                    .context("collect: failed to clear stale fanout_child_run_ids field")?;
            }
            let _: i64 = hash_ops::hdel(&mut conn, &run_key, "error")
                .await
                .context("collect: failed to clear stale parent error field")?;
            Ok(())
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct InMemoryCollectStore {
    state: Arc<Mutex<HashMap<String, CollectGroup>>>,
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
            state.entry(parent_run_id.to_string()).or_insert(group);
            Ok(())
        })
    }

    fn discard_group<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            state.remove(parent_run_id);
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
            let Some(group) = state.get_mut(parent_run_id) else {
                return Err(anyhow!(
                    "collect: parent group '{}' was not initialized",
                    parent_run_id
                ));
            };
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

    fn take_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            Ok(state.remove(parent_run_id))
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

    fn parse_nonnegative_count(value: Option<i64>) -> usize {
        value.unwrap_or(0).max(0) as usize
    }

    async fn pop_group(&self, parent_run_id: &str) -> Result<Option<CollectGroup>> {
        let mut conn = self.qm.connection();
        let values: Vec<String> = Script::new(POP_COLLECT_GROUP_LUA)
            .key(self.meta_key(parent_run_id))
            .key(self.items_key(parent_run_id))
            .key(self.runids_key(parent_run_id))
            .key(self.stats_key(parent_run_id))
            .invoke_async(&mut conn)
            .await?;
        if values.is_empty() {
            return Ok(None);
        }

        let mut iter = values.into_iter();
        let Some(meta_json) = iter.next() else {
            return Ok(None);
        };
        let mut group: CollectGroup = serde_json::from_str(&meta_json)?;
        group.results = iter
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
            let _: i64 = redis::cmd("DEL")
                .arg(self.meta_key(parent_run_id))
                .arg(self.items_key(parent_run_id))
                .arg(self.runids_key(parent_run_id))
                .arg(self.stats_key(parent_run_id))
                .query_async(&mut conn)
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
            let item_json = serde_json::to_string(&result)?;
            let values: Vec<i64> = Script::new(APPEND_MEMBER_RESULT_LUA)
                .key(self.meta_key(parent_run_id))
                .key(self.items_key(parent_run_id))
                .key(self.runids_key(parent_run_id))
                .arg(&result.child_run_id)
                .arg(item_json)
                .invoke_async(&mut conn)
                .await?;
            if values.len() != 2 {
                return Err(anyhow!(
                    "collect: append redis script returned {} values, expected 2",
                    values.len()
                ));
            }
            if values[0] == 0 && values[1] == 0 {
                return Err(anyhow!(
                    "collect: parent group '{}' was not initialized",
                    parent_run_id
                ));
            }
            let stats_key = self.stats_key(parent_run_id);
            if values[0] == 1 {
                let status_field = match status {
                    MemberResultStatus::Succeeded => "succeeded",
                    MemberResultStatus::Failed => "failed",
                    MemberResultStatus::Cancelled => "cancelled",
                };
                let _: i64 = redis::cmd("HINCRBY")
                    .arg(&stats_key)
                    .arg("collected")
                    .arg(1)
                    .query_async(&mut conn)
                    .await
                    .context("collect: failed to increment collected count")?;
                let _: i64 = redis::cmd("HINCRBY")
                    .arg(&stats_key)
                    .arg(status_field)
                    .arg(1)
                    .query_async(&mut conn)
                    .await
                    .with_context(|| {
                        format!("collect: failed to increment {status_field} count")
                    })?;
            }
            let counts: Vec<Option<i64>> = redis::cmd("HMGET")
                .arg(&stats_key)
                .arg("succeeded")
                .arg("failed")
                .arg("cancelled")
                .arg("collected")
                .query_async(&mut conn)
                .await
                .context("collect: failed to read redis collect progress counters")?;
            if counts.len() != 4 {
                return Err(anyhow!(
                    "collect: HMGET returned {} counters, expected 4",
                    counts.len()
                ));
            }
            let collected_count =
                Self::parse_nonnegative_count(counts[3]).max(values[1].max(0) as usize);
            let mut child_run_ids: Vec<String> = redis::cmd("SMEMBERS")
                .arg(self.runids_key(parent_run_id))
                .query_async(&mut conn)
                .await
                .context("collect: failed to read child run ids for progress snapshot")?;
            child_run_ids.sort_unstable();
            Ok(CollectAppendResult {
                appended: values[0] == 1,
                item_count: collected_count,
                succeeded_count: Self::parse_nonnegative_count(counts[0]),
                failed_count: Self::parse_nonnegative_count(counts[1]),
                cancelled_count: Self::parse_nonnegative_count(counts[2]),
                child_run_ids,
            })
        })
    }

    fn take_group<'a>(
        &'a self,
        parent_run_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<CollectGroup>>> {
        Box::pin(async move { self.pop_group(parent_run_id).await })
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
            Arc::new(RedisCollectProgressPersistence::new(qm.clone())),
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
        if !append_result.appended {
            return Ok(());
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

        let Some(group) = self.store.take_group(typed.parent_run_id.as_str()).await? else {
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

        let next_stage = parent_stage_context(&group.parent_payload)?.next_stage("collect")?;
        let result_payload = build_parent_result(&group);
        if next_stage.phase == "postprocess" {
            let mut handoff_payload = group.parent_payload.clone();
            let handoff_map = handoff_payload
                .as_object_mut()
                .ok_or_else(|| anyhow!("collect: parent payload must be a JSON object"))?;
            handoff_map.insert("result".to_string(), result_payload);
            crate::roles::stage::update_stage_context(handoff_map, &next_stage, "collect")?;
            let encoded = serde_json::to_string(&handoff_payload)
                .context("collect: encode postprocess payload")?;
            sink.handoff_to_run(
                msg,
                &next_stage.queue,
                &encoded,
                &next_stage.phase,
                group.parent_run_id.as_str(),
            )
            .await
            .context("collect: failed to hand off aggregated parent to postprocess")?;
            self.terminal_state
                .mark_terminal(typed.parent_run_id.as_str())
                .await?;
            return Ok(());
        }
        if next_stage.phase != "results" {
            return Err(anyhow!(
                "collect: next stage must be 'results' or 'postprocess', got '{}'",
                next_stage.phase
            ));
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
        sink.handoff_to_run(
            msg,
            &next_stage.queue,
            &encoded,
            &next_stage.phase,
            group.parent_run_id.as_str(),
        )
        .await
        .context("collect: failed to hand off aggregated parent to results")?;
        self.terminal_state
            .mark_terminal(typed.parent_run_id.as_str())
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

    use serde_json::{Value as JsonValue, json};
    use tokio::sync::Mutex as TokioMutex;
    use uuid::Uuid;

    use super::*;
    use crate::config::InputStreamSpec;
    use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};
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
    }

    impl RecordingProgressPersistence {
        fn updates(&self) -> Vec<CollectProgressUpdate> {
            self.updates.lock().unwrap().clone()
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

        let persistence = RedisCollectProgressPersistence::new(qm.clone());
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
}
