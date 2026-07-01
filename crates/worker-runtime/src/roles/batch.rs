/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use redis::Script;
use scicomp_rq::QueueManager;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use crate::traits::{BackgroundTask, BoxFuture, MessageSink, RoleEnv, TaskCriticality, WorkerRole};

const DEFAULT_BATCH_FLUSH_INTERVAL_MS: u64 = 25;
const DEFAULT_MAX_DUE_GROUPS_PER_TICK: usize = 64;
const DEFAULT_BATCH_STORE_PREFIX: &str = "batch";

const APPEND_GROUP_ITEM_LUA: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 then
  redis.call('SET', KEYS[1], ARGV[2])
  redis.call('ZADD', KEYS[4], ARGV[3], ARGV[1])
end
local added = redis.call('SADD', KEYS[3], ARGV[4])
if added == 1 then
  redis.call('RPUSH', KEYS[2], ARGV[5])
end
local size = redis.call('LLEN', KEYS[2])
return {added, size}
"#;

const POP_GROUP_LUA: &str = r#"
local meta = redis.call('GET', KEYS[1])
if not meta then
  return {}
end
local items = redis.call('LRANGE', KEYS[2], 0, -1)
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3])
redis.call('ZREM', KEYS[4], ARGV[1])
table.insert(items, 1, meta)
return items
"#;

const CLAIM_DUE_GROUPS_LUA: &str = r#"
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
if #ids > 0 then
  redis.call('ZREM', KEYS[1], unpack(ids))
end
return ids
"#;

#[derive(Debug, Clone, Deserialize)]
struct BatchEnvelope {
    run_id: String,
    workflow_id: String,
    #[serde(default)]
    operation: Option<String>,
    #[serde(default)]
    manifest_version: Option<String>,
    #[serde(default)]
    resource_profile: Option<JsonValue>,
    #[serde(default)]
    runtime: Option<JsonValue>,
    #[serde(default)]
    batch_profile: Option<BatchProfile>,
    stage_context: StageContext,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BatchProfile {
    #[serde(default)]
    enabled: bool,
    batch_key: String,
    max_batch_size: usize,
    max_wait_ms: u64,
    #[serde(default)]
    shared_memory_mb: Option<u64>,
    #[serde(default)]
    incremental_memory_mb: Option<u64>,
}

use crate::roles::stage::StageContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BufferedItem {
    run_id: String,
    payload: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BufferedGroup {
    workflow_id: String,
    operation: Option<String>,
    manifest_version: Option<String>,
    resource_profile: Option<JsonValue>,
    runtime: Option<JsonValue>,
    batch_profile: BatchProfile,
    stage_context: StageContext,
    items: Vec<BufferedItem>,
    first_seen_ms: u64,
}

#[derive(Debug, Default)]
struct BatchState {
    groups: HashMap<String, BufferedGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchAppendResult {
    appended: bool,
    item_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlushReason {
    MaxBatchSize,
    MaxWaitMs,
}

impl FlushReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::MaxBatchSize => "max_batch_size",
            Self::MaxWaitMs => "max_wait_ms",
        }
    }
}

trait BatchStore: Send + Sync {
    fn add_item<'a>(
        &'a self,
        group_id: &'a str,
        group: BufferedGroup,
        item: BufferedItem,
    ) -> BoxFuture<'a, Result<BatchAppendResult>>;

    fn take_group<'a>(&'a self, group_id: &'a str) -> BoxFuture<'a, Result<Option<BufferedGroup>>>;

    fn take_due_groups<'a>(&'a self, now_ms: u64) -> BoxFuture<'a, Result<Vec<BufferedGroup>>>;
}

#[derive(Debug, Default)]
struct InMemoryBatchStore {
    state: Arc<Mutex<BatchState>>,
}

impl InMemoryBatchStore {
    fn new() -> Self {
        Self::default()
    }
}

impl BatchStore for InMemoryBatchStore {
    fn add_item<'a>(
        &'a self,
        group_id: &'a str,
        group: BufferedGroup,
        item: BufferedItem,
    ) -> BoxFuture<'a, Result<BatchAppendResult>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let entry = state
                .groups
                .entry(group_id.to_string())
                .or_insert_with(|| group);
            if entry
                .items
                .iter()
                .any(|existing| existing.run_id == item.run_id)
            {
                return Ok(BatchAppendResult {
                    appended: false,
                    item_count: entry.items.len(),
                });
            }
            entry.items.push(item);
            Ok(BatchAppendResult {
                appended: true,
                item_count: entry.items.len(),
            })
        })
    }

    fn take_group<'a>(&'a self, group_id: &'a str) -> BoxFuture<'a, Result<Option<BufferedGroup>>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            Ok(state.groups.remove(group_id))
        })
    }

    fn take_due_groups<'a>(&'a self, now_ms: u64) -> BoxFuture<'a, Result<Vec<BufferedGroup>>> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let ready_keys: Vec<String> = state
                .groups
                .iter()
                .filter_map(|(key, group)| {
                    let due_at = group
                        .first_seen_ms
                        .saturating_add(group.batch_profile.max_wait_ms);
                    if now_ms >= due_at {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();
            let ready = ready_keys
                .into_iter()
                .filter_map(|key| state.groups.remove(&key))
                .collect();
            Ok(ready)
        })
    }
}

struct RedisBatchStore {
    qm: QueueManager,
    key_prefix: String,
}

impl RedisBatchStore {
    fn new(qm: QueueManager, key_prefix: impl Into<String>) -> Self {
        Self {
            qm,
            key_prefix: key_prefix.into(),
        }
    }

    fn due_key(&self) -> String {
        format!("{}:due", self.key_prefix)
    }

    fn meta_key(&self, group_id: &str) -> String {
        format!("{}:group:{}:meta", self.key_prefix, group_id)
    }

    fn items_key(&self, group_id: &str) -> String {
        format!("{}:group:{}:items", self.key_prefix, group_id)
    }

    fn runids_key(&self, group_id: &str) -> String {
        format!("{}:group:{}:runids", self.key_prefix, group_id)
    }

    async fn pop_group(&self, group_id: &str) -> Result<Option<BufferedGroup>> {
        let mut conn = self.qm.connection();
        let values: Vec<String> = Script::new(POP_GROUP_LUA)
            .key(self.meta_key(group_id))
            .key(self.items_key(group_id))
            .key(self.runids_key(group_id))
            .key(self.due_key())
            .arg(group_id)
            .invoke_async(&mut conn)
            .await?;
        if values.is_empty() {
            return Ok(None);
        }

        let mut iter = values.into_iter();
        let Some(meta_json) = iter.next() else {
            return Ok(None);
        };
        let mut group: BufferedGroup = serde_json::from_str(&meta_json)?;
        group.items = iter
            .map(|item_json| serde_json::from_str::<BufferedItem>(&item_json))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(group))
    }
}

impl BatchStore for RedisBatchStore {
    fn add_item<'a>(
        &'a self,
        group_id: &'a str,
        group: BufferedGroup,
        item: BufferedItem,
    ) -> BoxFuture<'a, Result<BatchAppendResult>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let due_at_ms = group
                .first_seen_ms
                .saturating_add(group.batch_profile.max_wait_ms);
            let meta_json = serde_json::to_string(&BufferedGroup {
                items: Vec::new(),
                ..group
            })?;
            let item_json = serde_json::to_string(&item)?;
            let result: Vec<i64> = Script::new(APPEND_GROUP_ITEM_LUA)
                .key(self.meta_key(group_id))
                .key(self.items_key(group_id))
                .key(self.runids_key(group_id))
                .key(self.due_key())
                .arg(group_id)
                .arg(meta_json)
                .arg(due_at_ms)
                .arg(&item.run_id)
                .arg(item_json)
                .invoke_async(&mut conn)
                .await?;
            if result.len() != 2 {
                return Err(anyhow!(
                    "batch: append redis script returned {} values, expected 2",
                    result.len()
                ));
            }
            Ok(BatchAppendResult {
                appended: result[0] == 1,
                item_count: result[1].max(0) as usize,
            })
        })
    }

    fn take_group<'a>(&'a self, group_id: &'a str) -> BoxFuture<'a, Result<Option<BufferedGroup>>> {
        Box::pin(async move { self.pop_group(group_id).await })
    }

    fn take_due_groups<'a>(&'a self, now_ms: u64) -> BoxFuture<'a, Result<Vec<BufferedGroup>>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let group_ids: Vec<String> = Script::new(CLAIM_DUE_GROUPS_LUA)
                .key(self.due_key())
                .arg(now_ms)
                .arg(DEFAULT_MAX_DUE_GROUPS_PER_TICK)
                .invoke_async(&mut conn)
                .await?;
            let mut ready = Vec::with_capacity(group_ids.len());
            for group_id in group_ids {
                if let Some(group) = self.pop_group(&group_id).await? {
                    ready.push(group);
                }
            }
            Ok(ready)
        })
    }
}

pub struct BatchRole {
    input_streams: Vec<String>,
    store: Arc<dyn BatchStore>,
}

impl BatchRole {
    pub fn from_env(env: &RoleEnv) -> Result<(Self, Vec<Box<dyn BackgroundTask>>)> {
        Self::from_env_with_store(env, Arc::new(InMemoryBatchStore::new()))
    }

    pub fn from_env_with_queue_manager(
        env: &RoleEnv,
        qm: QueueManager,
    ) -> Result<(Self, Vec<Box<dyn BackgroundTask>>)> {
        Self::from_env_with_store(
            env,
            Arc::new(RedisBatchStore::new(qm, DEFAULT_BATCH_STORE_PREFIX)),
        )
    }

    fn from_env_with_store(
        env: &RoleEnv,
        store: Arc<dyn BatchStore>,
    ) -> Result<(Self, Vec<Box<dyn BackgroundTask>>)> {
        let role = Self {
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
            store: store.clone(),
        };
        let task: Box<dyn BackgroundTask> = Box::new(BatchFlushTask { store });
        Ok((role, vec![task]))
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "batch: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let (typed, payload) = decode_batch_payload(msg.payload())?;
        let next_stage = typed.stage_context.next_stage("batch")?;
        if next_stage.phase != "schedule" {
            return Err(anyhow!(
                "batch: next stage must be 'schedule', got '{}'",
                next_stage.phase
            ));
        }

        let Some(batch_profile) = typed
            .batch_profile
            .clone()
            .filter(|profile| profile.enabled)
        else {
            return sink
                .handoff(msg, &next_stage.queue, msg.payload(), &next_stage.phase)
                .await
                .map(|_| ());
        };

        let group_id = format!("{}::{}", typed.workflow_id, batch_profile.batch_key);
        let outcome = self
            .store
            .add_item(
                &group_id,
                BufferedGroup {
                    workflow_id: typed.workflow_id.clone(),
                    operation: typed.operation.clone(),
                    manifest_version: typed.manifest_version.clone(),
                    resource_profile: typed.resource_profile.clone(),
                    runtime: typed.runtime.clone(),
                    batch_profile: batch_profile.clone(),
                    stage_context: typed.stage_context.clone(),
                    items: Vec::new(),
                    first_seen_ms: now_ms()?,
                },
                BufferedItem {
                    run_id: typed.run_id.clone(),
                    payload,
                },
            )
            .await?;

        if !outcome.appended || outcome.item_count < batch_profile.max_batch_size {
            return Ok(());
        }

        let Some(group) = self.store.take_group(&group_id).await? else {
            return Ok(());
        };
        let batch_payload = build_batch_payload(&group, FlushReason::MaxBatchSize)?;
        let encoded = serde_json::to_string(&batch_payload)?;
        log_flushed_batch(&group, &batch_payload, FlushReason::MaxBatchSize);
        sink.handoff(msg, &next_stage.queue, &encoded, &next_stage.phase)
            .await?;
        Ok(())
    }
}

impl WorkerRole for BatchRole {
    fn name(&self) -> &'static str {
        "batch"
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

struct BatchFlushTask {
    store: Arc<dyn BatchStore>,
}

impl BackgroundTask for BatchFlushTask {
    fn name(&self) -> &'static str {
        "batch_flush"
    }

    fn interval(&self) -> Duration {
        Duration::from_millis(DEFAULT_BATCH_FLUSH_INTERVAL_MS)
    }

    fn criticality(&self) -> TaskCriticality {
        TaskCriticality::BestEffort
    }

    fn run<'a>(&'a self, sink: &'a dyn MessageSink) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let ready = self.store.take_due_groups(now_ms()?).await?;

            for group in ready {
                let batch_payload = build_batch_payload(&group, FlushReason::MaxWaitMs)?;
                let encoded = serde_json::to_string(&batch_payload)?;
                let next_stage = group.stage_context.next_stage("batch")?;
                log_flushed_batch(&group, &batch_payload, FlushReason::MaxWaitMs);
                sink.enqueue(
                    &next_stage.queue,
                    batch_payload["batch_id"].as_str().unwrap_or("batch"),
                    &encoded,
                    &next_stage.phase,
                )
                .await?;
            }
            Ok(())
        })
    }
}

fn decode_batch_payload(raw: &str) -> Result<(BatchEnvelope, JsonValue)> {
    let value: JsonValue = serde_json::from_str(raw)?;
    let typed: BatchEnvelope = serde_json::from_value(value.clone())?;
    if typed.stage_context.current_phase != "batch" {
        return Err(anyhow!(
            "batch: payload current_phase must be 'batch', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    Ok((typed, value))
}

fn build_batch_payload(group: &BufferedGroup, flush_reason: FlushReason) -> Result<JsonValue> {
    let batch_size = group.items.len() as u64;
    let formed_at_ms = now_ms()?;
    let waited_ms = formed_at_ms.saturating_sub(group.first_seen_ms);
    let mut resource_profile = group.resource_profile.clone().unwrap_or_else(|| json!({}));
    if let Some(resource_map) = resource_profile.as_object_mut()
        && let (Some(shared), Some(incremental)) = (
            group.batch_profile.shared_memory_mb,
            group.batch_profile.incremental_memory_mb,
        )
    {
        let incremental_total = incremental.checked_mul(batch_size).ok_or_else(|| {
            anyhow!(
                "batch: memory_mb overflow while scaling incremental_memory_mb={} by batch_size={}",
                incremental,
                batch_size
            )
        })?;
        let total_memory = shared.checked_add(incremental_total).ok_or_else(|| {
            anyhow!(
                "batch: memory_mb overflow while adding shared_memory_mb={} and incremental_total={}",
                shared,
                incremental_total
            )
        })?;
        resource_map.insert(
            "memory_mb".to_string(),
            JsonValue::Number(total_memory.into()),
        );
    }
    let batch_id = Uuid::new_v4().to_string();

    Ok(json!({
        "batch_id": batch_id,
        "batch_info": {
            "batch_id": batch_id,
            "batch_size": batch_size,
            "flush_reason": flush_reason.as_str(),
            "first_seen_ms": group.first_seen_ms,
            "formed_at_ms": formed_at_ms,
            "waited_ms": waited_ms,
        },
        "workflow_id": group.workflow_id,
        "operation": group.operation,
        "manifest_version": group.manifest_version,
        "batch_profile": {
            "batch_key": group.batch_profile.batch_key,
            "max_batch_size": group.batch_profile.max_batch_size,
            "max_wait_ms": group.batch_profile.max_wait_ms,
            "shared_memory_mb": group.batch_profile.shared_memory_mb,
            "incremental_memory_mb": group.batch_profile.incremental_memory_mb,
        },
        "items": group.items.iter().map(|item| json!({
            "run_id": item.run_id,
            "payload": item.payload,
        })).collect::<Vec<_>>(),
        "resource_profile": resource_profile,
        "runtime": group.runtime,
        "stage_context": {
            "current_stage_id": group.stage_context.current_stage_id,
            "current_phase": group.stage_context.current_phase,
            "pipeline": group.stage_context.pipeline.iter().map(|stage| json!({
                "id": stage.id,
                "phase": stage.phase,
                "queue": stage.queue,
                "next": stage.next,
            })).collect::<Vec<_>>(),
        }
    }))
}

fn log_flushed_batch(group: &BufferedGroup, payload: &JsonValue, flush_reason: FlushReason) {
    info!(
        workflow_id = %group.workflow_id,
        batch_key = %group.batch_profile.batch_key,
        batch_id = %payload["batch_id"].as_str().unwrap_or("unknown"),
        batch_size = payload["batch_info"]["batch_size"]
            .as_u64()
            .unwrap_or(group.items.len() as u64),
        waited_ms = payload["batch_info"]["waited_ms"].as_u64().unwrap_or_default(),
        flush_reason = %flush_reason.as_str(),
        "batch flushed"
    );
}

fn now_ms() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("batch: system clock before unix epoch: {err}"))?;
    Ok(duration.as_millis().try_into().unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use super::*;
    use crate::config::InputStreamSpec;
    use crate::traits::{MessageSink, RoleEnv};

    #[derive(Debug, Clone)]
    struct HandoffRecord {
        dest_stream: String,
        payload: String,
        stage: String,
    }

    struct RecordingSink {
        enqueues: StdMutex<Vec<HandoffRecord>>,
        handoffs: StdMutex<Vec<HandoffRecord>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                enqueues: StdMutex::new(Vec::new()),
                handoffs: StdMutex::new(Vec::new()),
            }
        }
    }

    impl MessageSink for RecordingSink {
        fn enqueue<'a>(
            &'a self,
            stream: &'a str,
            _run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.enqueues.lock().unwrap().push(HandoffRecord {
                    dest_stream: stream.to_string(),
                    payload: payload.to_string(),
                    stage: stage.to_string(),
                });
                Ok("1-0".to_string())
            })
        }

        fn ack_message<'a>(&'a self, _msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.handoffs.lock().unwrap().push(HandoffRecord {
                    dest_stream: dest_stream.to_string(),
                    payload: payload.to_string(),
                    stage: stage.to_string(),
                });
                Ok("1-0".to_string())
            })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    fn batch_env() -> RoleEnv {
        RoleEnv {
            role_name: "batch".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "batch".to_string(),
                max_dequeue_items: 8,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec!["schedule".to_string()],
            role_config: None,
            python_runtime_envs: Default::default(),
        }
    }

    fn batch_msg(run_id: &str, payload: &str) -> scicomp_rq::Message {
        scicomp_rq::Message::new("test:batch", "batch:grp", "1-0", run_id, payload, "batch")
    }

    fn payload(run_id: &str, max_batch_size: usize, max_wait_ms: u64) -> String {
        json!({
            "run_id": run_id,
            "workflow_id": "demo-batch",
            "operation": "run",
            "manifest_version": "1.0.0",
            "parameters": { "value": run_id },
            "resource_profile": {
                "executor_class": "python.gpu.test",
                "gpus_required": 1,
                "memory_mb": 100
            },
            "runtime": {
                "entrypoint": "workflow.py",
                "executor_class": "python.gpu.test"
            },
            "batch_profile": {
                "enabled": true,
                "batch_key": "same-key",
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "shared_memory_mb": 100,
                "incremental_memory_mb": 10
            },
            "stage_context": {
                "current_stage_id": "batch",
                "current_phase": "batch",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "batch"},
                    {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.test", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
        .to_string()
    }

    #[tokio::test]
    async fn batch_role_flushes_when_group_reaches_max_batch_size() {
        let (role, _tasks) = BatchRole::from_env(&batch_env()).unwrap();
        let sink = RecordingSink::new();

        role.handle(
            &batch_msg("run-a", &payload("run-a", 2, 1_000)),
            "batch",
            &sink,
        )
        .await
        .unwrap();
        assert!(sink.handoffs.lock().unwrap().is_empty());

        role.handle(
            &batch_msg("run-b", &payload("run-b", 2, 1_000)),
            "batch",
            &sink,
        )
        .await
        .unwrap();

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        assert_eq!(handoffs[0].dest_stream, "schedule");
        assert_eq!(handoffs[0].stage, "schedule");
        assert_eq!(forwarded["items"].as_array().unwrap().len(), 2);
        assert_eq!(forwarded["resource_profile"]["memory_mb"], 120);
        assert_eq!(forwarded["batch_info"]["flush_reason"], "max_batch_size");
        assert_eq!(forwarded["batch_info"]["batch_size"], 2);
    }

    #[tokio::test]
    async fn batch_flush_task_flushes_when_group_wait_expires() {
        let (role, tasks) = BatchRole::from_env(&batch_env()).unwrap();
        let sink = RecordingSink::new();

        role.handle(&batch_msg("run-a", &payload("run-a", 8, 1)), "batch", &sink)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        tasks[0].run(&sink).await.unwrap();

        let enqueues = sink.enqueues.lock().unwrap().clone();
        assert_eq!(enqueues.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&enqueues[0].payload).unwrap();
        assert_eq!(enqueues[0].dest_stream, "schedule");
        assert_eq!(forwarded["items"].as_array().unwrap().len(), 1);
        assert_eq!(forwarded["batch_info"]["flush_reason"], "max_wait_ms");
        assert_eq!(forwarded["batch_info"]["batch_size"], 1);
    }

    #[tokio::test]
    async fn batch_role_persists_group_across_role_restart_with_shared_store() {
        let shared_store: Arc<dyn BatchStore> = Arc::new(InMemoryBatchStore::new());
        let sink = RecordingSink::new();

        let (first_role, _tasks) =
            BatchRole::from_env_with_store(&batch_env(), shared_store.clone()).unwrap();
        first_role
            .handle(
                &batch_msg("run-a", &payload("run-a", 2, 1_000)),
                "batch",
                &sink,
            )
            .await
            .unwrap();
        assert!(sink.handoffs.lock().unwrap().is_empty());

        let (second_role, _tasks) =
            BatchRole::from_env_with_store(&batch_env(), shared_store).unwrap();
        second_role
            .handle(
                &batch_msg("run-b", &payload("run-b", 2, 1_000)),
                "batch",
                &sink,
            )
            .await
            .unwrap();

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&handoffs[0].payload).unwrap();
        let items = forwarded["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["run_id"], "run-a");
        assert_eq!(items[1]["run_id"], "run-b");
    }

    #[test]
    fn build_batch_payload_rejects_memory_overflow() {
        let stage_context = crate::roles::stage::StageContext {
            current_stage_id: "batch".to_string(),
            current_phase: "batch".to_string(),
            pipeline: vec![
                crate::roles::stage::StageDescriptor {
                    id: "batch".to_string(),
                    phase: "batch".to_string(),
                    queue: "batch".to_string(),
                    next: Some("schedule".to_string()),
                },
                crate::roles::stage::StageDescriptor {
                    id: "schedule".to_string(),
                    phase: "schedule".to_string(),
                    queue: "schedule".to_string(),
                    next: None,
                },
            ],
        };
        let group = BufferedGroup {
            workflow_id: "wf".to_string(),
            operation: Some("run".to_string()),
            manifest_version: Some("1.0.0".to_string()),
            resource_profile: Some(json!({ "memory_mb": 1 })),
            runtime: None,
            batch_profile: BatchProfile {
                enabled: true,
                batch_key: "same-key".to_string(),
                max_batch_size: 2,
                max_wait_ms: 1000,
                shared_memory_mb: Some(u64::MAX),
                incremental_memory_mb: Some(1),
            },
            stage_context,
            items: vec![
                BufferedItem {
                    run_id: "run-a".to_string(),
                    payload: json!({}),
                },
                BufferedItem {
                    run_id: "run-b".to_string(),
                    payload: json!({}),
                },
            ],
            first_seen_ms: 0,
        };

        let result = build_batch_payload(&group, FlushReason::MaxBatchSize);
        assert!(
            result.is_err(),
            "memory_mb overflow must return an explicit error"
        );
    }
}
