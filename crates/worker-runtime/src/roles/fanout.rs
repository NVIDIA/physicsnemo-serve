/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use scicomp_rq::{Output, QueueManager};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tracing::warn;

use crate::roles::collect::{
    CollectGroup, CollectStore, DEFAULT_COLLECT_STORE_PREFIX, InMemoryCollectStore,
    RedisCollectStore,
};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

#[derive(Debug, Clone, Deserialize)]
struct FanoutEnvelope {
    run_id: String,
    workflow_id: String,
    #[serde(default)]
    operation: Option<String>,
    #[serde(default)]
    fanout_profile: Option<FanoutProfile>,
    stage_context: StageContext,
}

#[derive(Debug, Clone, Deserialize)]
struct FanoutProfile {
    #[serde(default)]
    item_count: Option<usize>,
}

use crate::roles::stage::{StageContext, StageDescriptor};

pub struct FanoutRole {
    input_streams: Vec<String>,
    collect_store: Arc<dyn CollectStore>,
}

impl FanoutRole {
    pub fn from_env(env: &RoleEnv) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Self::from_env_with_store(env, Arc::new(InMemoryCollectStore::new()))
    }

    pub fn from_env_with_queue_manager(
        env: &RoleEnv,
        qm: QueueManager,
    ) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Self::from_env_with_store(
            env,
            Arc::new(RedisCollectStore::new(qm, DEFAULT_COLLECT_STORE_PREFIX)),
        )
    }

    pub(crate) fn from_env_with_store(
        env: &RoleEnv,
        collect_store: Arc<dyn CollectStore>,
    ) -> Result<(Self, Vec<Box<dyn crate::traits::BackgroundTask>>)> {
        Ok((
            Self {
                input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
                collect_store,
            },
            vec![],
        ))
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "fanout: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let (typed, payload) = decode_fanout_payload(msg.payload())?;
        let schedule_stage = typed.stage_context.next_stage("fanout")?;
        if schedule_stage.phase != "schedule" {
            return Err(anyhow!(
                "fanout: next stage must be 'schedule', got '{}'",
                schedule_stage.phase
            ));
        }
        let collect_stage = typed.stage_context.find_phase("collect", "fanout")?;
        let items = payload
            .get("fanout_items")
            .and_then(JsonValue::as_array)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| anyhow!("fanout: payload must include non-empty fanout_items"))?;
        let expected_count = typed
            .fanout_profile
            .as_ref()
            .and_then(|profile| profile.item_count)
            .unwrap_or(items.len());
        if expected_count != items.len() {
            return Err(anyhow!(
                "fanout: fanout_profile.item_count={} does not match fanout_items len={}",
                expected_count,
                items.len()
            ));
        }

        let parent_payload =
            build_parent_payload_for_collect(&payload, &typed.run_id, &collect_stage)?;
        self.collect_store
            .init_group(
                typed.run_id.as_str(),
                CollectGroup {
                    parent_run_id: typed.run_id.clone(),
                    workflow_id: typed.workflow_id.clone(),
                    parent_payload,
                    expected_count,
                    results: Vec::new(),
                },
            )
            .await?;

        let outputs = items
            .iter()
            .enumerate()
            .map(|(item_position, item)| {
                build_child_output(&typed, &payload, item, item_position, &schedule_stage)
            })
            .collect::<Result<Vec<_>>>()?;

        if let Err(error) = sink.forward_many(msg, &outputs).await {
            if let Err(discard_err) = self
                .collect_store
                .discard_group(typed.run_id.as_str())
                .await
            {
                warn!(
                    run_id = %typed.run_id,
                    error = %discard_err,
                    "fanout: failed to discard collect group after forward failure"
                );
            }
            return Err(error).context("fanout: failed to fan out child items");
        }

        Ok(())
    }
}

impl WorkerRole for FanoutRole {
    fn name(&self) -> &'static str {
        "fanout"
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

fn decode_fanout_payload(raw: &str) -> Result<(FanoutEnvelope, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("fanout: empty payload"));
    }
    let value: JsonValue =
        serde_json::from_str(raw).context("fanout: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("fanout: payload must be a JSON object"));
    }
    let typed: FanoutEnvelope =
        serde_json::from_value(value.clone()).context("fanout: invalid payload schema")?;
    if typed.run_id.trim().is_empty() {
        return Err(anyhow!("fanout: run_id is required and must be non-empty"));
    }
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "fanout: workflow_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "fanout" {
        return Err(anyhow!(
            "fanout: payload current_phase must be 'fanout', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    Ok((typed, value))
}

fn build_parent_payload_for_collect(
    payload: &JsonValue,
    parent_run_id: &str,
    collect_stage: &StageDescriptor,
) -> Result<JsonValue> {
    let mut parent_payload = payload.clone();
    let map = parent_payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("fanout: payload must remain an object"))?;
    map.insert(
        "run_id".to_string(),
        JsonValue::String(parent_run_id.to_string()),
    );
    map.remove("fanout_items");
    map.remove("result");
    crate::roles::stage::update_stage_context(map, collect_stage, "fanout")?;
    Ok(parent_payload)
}

fn build_child_output(
    typed: &FanoutEnvelope,
    payload: &JsonValue,
    item: &JsonValue,
    item_position: usize,
    schedule_stage: &StageDescriptor,
) -> Result<Output> {
    let mut child_payload = payload.clone();
    let child_map = child_payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("fanout: payload must remain an object"))?;
    child_map.remove("fanout_items");
    child_map.remove("result");

    let child_run_id = format!("{}:item:{}", typed.run_id, item_position);
    child_map.insert(
        "run_id".to_string(),
        JsonValue::String(child_run_id.clone()),
    );
    child_map.insert(
        "parent_run_id".to_string(),
        JsonValue::String(typed.run_id.clone()),
    );
    child_map.insert("fanout_item".to_string(), item.clone());

    if let Some(operation) = item.get("operation").and_then(JsonValue::as_str) {
        child_map.insert(
            "operation".to_string(),
            JsonValue::String(operation.to_string()),
        );
    } else if let Some(operation) = &typed.operation {
        child_map
            .entry("operation".to_string())
            .or_insert_with(|| JsonValue::String(operation.clone()));
    }
    if let Some(parameters) = item.get("parameters") {
        child_map.insert("parameters".to_string(), parameters.clone());
    }
    if let Some(resource_profile) = item.get("resource_profile") {
        child_map.insert("resource_profile".to_string(), resource_profile.clone());
    }
    crate::roles::stage::update_stage_context(child_map, schedule_stage, "fanout")?;

    Ok(
        Output::new(schedule_stage.queue.clone(), child_payload.to_string())
            .with_run_id(child_run_id)
            .with_stage(schedule_stage.phase.clone()),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use anyhow::{Result, anyhow};
    use serde_json::{Value as JsonValue, json};

    use super::*;
    use crate::config::InputStreamSpec;
    use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

    #[derive(Debug, Clone)]
    struct ForwardRecord {
        stream: String,
        payload: String,
        stage: Option<String>,
        run_id: String,
    }

    struct RecordingSink {
        forwards: StdMutex<Vec<ForwardRecord>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                forwards: StdMutex::new(Vec::new()),
            }
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
            Box::pin(async { Err(anyhow!("fanout tests do not use enqueue")) })
        }

        fn ack_message<'a>(&'a self, _msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _dest_stream: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("fanout tests do not use handoff")) })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async move {
                let mut records = self.forwards.lock().unwrap();
                for output in outputs {
                    records.push(ForwardRecord {
                        stream: output.stream().to_string(),
                        payload: output.payload().to_string(),
                        stage: output.stage().map(ToOwned::to_owned),
                        run_id: output.run_id().unwrap_or_default().to_string(),
                    });
                }
                Ok((0..outputs.len()).map(|idx| format!("1-{idx}")).collect())
            })
        }
    }

    fn fanout_env() -> RoleEnv {
        RoleEnv {
            role_name: "fanout".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "fanout".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec!["schedule".to_string()],
            role_config: None,
            python_runtime_envs: Default::default(),
        }
    }

    fn fanout_msg(run_id: &str, payload: &str) -> scicomp_rq::Message {
        scicomp_rq::Message::new(
            "test:fanout",
            "fanout:grp",
            "1-0",
            run_id,
            payload,
            "fanout",
        )
    }

    fn parent_payload() -> String {
        json!({
            "run_id": "parent-run",
            "workflow_id": "demo-fanout",
            "operation": "run",
            "parameters": {
                "start_time": ["2026-03-21T00:00:00Z"],
                "num_steps": 4
            },
            "resource_profile": {
                "executor_class": "python.gpu.demo",
                "gpus_required": 1,
                "memory_mb": 24000
            },
            "fanout_profile": {
                "item_count": 2,
                "aggregation_mode": "all_members"
            },
            "fanout_items": [
                {
                    "item_index": 0,
                    "member_seed": 1000,
                    "parameters": {
                        "start_time": ["2026-03-21T00:00:00Z"],
                        "num_steps": 4,
                        "member_seed": 1000
                    }
                },
                {
                    "item_index": 1,
                    "member_seed": 1001,
                    "parameters": {
                        "start_time": ["2026-03-21T00:00:00Z"],
                        "num_steps": 4,
                        "member_seed": 1001
                    }
                }
            ],
            "stage_context": {
                "current_stage_id": "fanout",
                "current_phase": "fanout",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "fanout"},
                    {"id": "fanout", "phase": "fanout", "queue": "fanout", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "collect"},
                    {"id": "collect", "phase": "collect", "queue": "collect", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        }).to_string()
    }

    #[tokio::test]
    async fn fanout_role_expands_parent_request_into_child_schedule_messages() {
        let (role, tasks) = FanoutRole::from_env(&fanout_env()).expect("fanout role should build");
        assert!(tasks.is_empty());
        let sink = RecordingSink::new();

        role.handle(
            &fanout_msg("parent-run", &parent_payload()),
            "fanout",
            &sink,
        )
        .await
        .expect("fanout should succeed");

        let forwards = sink.forwards.lock().unwrap().clone();
        assert_eq!(forwards.len(), 2);
        assert_eq!(forwards[0].stream, "schedule");
        assert_eq!(forwards[0].stage.as_deref(), Some("schedule"));

        let first_child: JsonValue = serde_json::from_str(&forwards[0].payload).unwrap();
        assert_eq!(forwards[0].run_id, "parent-run:item:0");
        assert_eq!(first_child["parent_run_id"], "parent-run");
        assert_eq!(first_child["fanout_item"]["item_index"], 0);
        assert_eq!(first_child["parameters"]["member_seed"], 1000);
        assert_eq!(first_child["stage_context"]["current_phase"], "schedule");

        let second_child: JsonValue = serde_json::from_str(&forwards[1].payload).unwrap();
        assert_eq!(forwards[1].run_id, "parent-run:item:1");
        assert_eq!(second_child["parent_run_id"], "parent-run");
        assert_eq!(second_child["fanout_item"]["item_index"], 1);
        assert_eq!(second_child["parameters"]["member_seed"], 1001);
        assert_eq!(second_child["stage_context"]["current_phase"], "schedule");
        assert_ne!(forwards[0].run_id, forwards[1].run_id);
    }
}
