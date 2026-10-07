/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scheduler-owned fanout.
//!
//! A schedule request carrying non-empty `fanout_items` is a fan-out parent. The
//! scheduler registers its collect group and atomically replaces the parent
//! message with one child schedule request per item. Children route from execute
//! to `collect` through a hidden `_gather` stage spliced into their pipeline, so
//! plugin manifests never declare fanout or collect stages.

use anyhow::{Context, Result, anyhow};
use scicomp_rq::{Message, Output};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use crate::roles::collect::{CollectGroup, CollectStore};
use crate::roles::stage::{StageContext, StageDescriptor, update_stage_context};
use crate::traits::MessageSink;

const GATHER_STAGE_ID: &str = "_gather";
const GATHER_PHASE: &str = "collect";
pub(super) const GATHER_QUEUE: &str = "collect";

/// A request is a fanout parent when it carries `fanout_items`. An empty array
/// means no fanout; any other non-array value is treated as a parent so
/// expansion rejects it instead of silently scheduling the request once.
pub(super) fn is_fanout_parent(payload: &JsonValue) -> bool {
    payload
        .get("fanout_items")
        .is_some_and(|items| match items {
            JsonValue::Null => false,
            JsonValue::Array(items) => !items.is_empty(),
            _ => true,
        })
}

pub(super) async fn expand_fanout_parent(
    msg: &Message,
    payload: JsonValue,
    schedule_stream: &str,
    collect_store: &dyn CollectStore,
    sink: &dyn MessageSink,
) -> Result<()> {
    let parent_run_id = msg.run_id();
    let JsonValue::Object(mut parent) = payload else {
        return Err(anyhow!("scheduler: fanout payload must be a JSON object"));
    };
    let workflow_id = parent
        .get("workflow_id")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("scheduler: fanout parent requires a non-empty workflow_id"))?
        .to_string();
    let Some(JsonValue::Array(items)) = parent.remove("fanout_items") else {
        return Err(anyhow!("scheduler: fanout_items must be a non-empty array"));
    };
    parent.remove("result");
    set_item_count(&mut parent, items.len())?;
    let gather_stage = splice_gather_stage(&mut parent)?;
    let expected_count = items.len();
    let children = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| build_child(&parent, parent_run_id, index, item, schedule_stream))
        .collect::<Result<Vec<_>>>()?;

    let mut collect_payload = parent;
    collect_payload.insert("run_id".to_string(), json!(parent_run_id));
    update_stage_context(&mut collect_payload, &gather_stage, "scheduler")?;
    collect_store
        .init_group(
            parent_run_id,
            CollectGroup {
                parent_run_id: parent_run_id.to_string(),
                workflow_id,
                parent_payload: JsonValue::Object(collect_payload),
                expected_count,
                results: Vec::new(),
            },
        )
        .await?;

    // The group is kept when forwarding fails: the outcome may be ambiguous (the
    // script can commit before the client sees an error), and a retry reuses the
    // group because `init_group` never overwrites an existing one.
    sink.forward_many(msg, &children)
        .await
        .context("scheduler: failed to fan out child items")?;
    Ok(())
}

/// Builds a collect envelope reporting a fanout child as failed, so its parent
/// still finalizes when the child never reaches execute. Returns `None` for
/// requests that are not fanout children.
pub(super) fn failed_child_envelope(payload: &JsonValue, error: &str) -> Option<String> {
    let is_child = payload
        .pointer("/stage_context/pipeline")
        .and_then(JsonValue::as_array)?
        .iter()
        .any(|stage| stage.get("id").and_then(JsonValue::as_str) == Some(GATHER_STAGE_ID));
    if !is_child {
        return None;
    }
    let mut envelope = payload.clone();
    envelope["result"] = json!({ "status": "failed", "error": error });
    envelope["stage_context"]["current_stage_id"] = json!(GATHER_STAGE_ID);
    envelope["stage_context"]["current_phase"] = json!(GATHER_PHASE);
    Some(envelope.to_string())
}

/// Defaults `fanout_profile.item_count` to the number of items and rejects a
/// mismatching explicit count.
fn set_item_count(parent: &mut JsonMap<String, JsonValue>, item_count: usize) -> Result<()> {
    if parent.get("fanout_profile").is_none_or(JsonValue::is_null) {
        parent.insert("fanout_profile".to_string(), json!({}));
    }
    let profile = parent
        .get_mut("fanout_profile")
        .and_then(JsonValue::as_object_mut)
        .ok_or_else(|| anyhow!("scheduler: fanout_profile must be a JSON object"))?;
    if let Some(explicit) = profile.get("item_count").filter(|value| !value.is_null())
        && explicit.as_u64() != Some(item_count as u64)
    {
        return Err(anyhow!(
            "scheduler: fanout_profile.item_count={explicit} does not match fanout_items len={item_count}"
        ));
    }
    profile.insert("item_count".to_string(), json!(item_count));
    Ok(())
}

/// Inserts `_gather` between the execute stage that follows this schedule stage
/// and whatever came after it, and returns the gather stage.
fn splice_gather_stage(parent: &mut JsonMap<String, JsonValue>) -> Result<StageDescriptor> {
    let stage_context: StageContext = parent
        .get("stage_context")
        .cloned()
        .ok_or_else(|| anyhow!("scheduler: fanout parent is missing stage_context"))
        .and_then(|value| {
            serde_json::from_value(value)
                .context("scheduler: fanout parent has invalid stage_context")
        })?;
    if stage_context
        .pipeline
        .iter()
        .any(|stage| stage.id == GATHER_STAGE_ID)
    {
        return Err(anyhow!(
            "scheduler: pipeline stage id '{GATHER_STAGE_ID}' is reserved for fanout"
        ));
    }
    let execute = stage_context.next_stage("scheduler")?;
    if execute.phase != "execute" {
        return Err(anyhow!(
            "scheduler: fanout requires schedule stage '{}' to be followed by an execute stage, got '{}' with phase '{}'",
            stage_context.current_stage_id,
            execute.id,
            execute.phase
        ));
    }
    let after_execute = execute
        .next
        .as_deref()
        .and_then(|next_id| stage_context.pipeline.iter().find(|stage| stage.id == next_id))
        .filter(|stage| matches!(stage.phase.as_str(), "postprocess" | "publish" | "results"))
        .ok_or_else(|| {
            anyhow!(
                "scheduler: fanout requires stage '{}' to be followed by postprocess, publish, or results",
                execute.id
            )
        })?;
    let gather_stage = StageDescriptor {
        id: GATHER_STAGE_ID.to_string(),
        phase: GATHER_PHASE.to_string(),
        queue: GATHER_QUEUE.to_string(),
        next: Some(after_execute.id.clone()),
    };

    let pipeline = parent
        .get_mut("stage_context")
        .and_then(|value| value.get_mut("pipeline"))
        .and_then(JsonValue::as_array_mut)
        .ok_or_else(|| {
            anyhow!("scheduler: fanout parent stage_context.pipeline must be an array")
        })?;
    for stage in pipeline.iter_mut() {
        if stage.get("id").and_then(JsonValue::as_str) == Some(execute.id.as_str()) {
            stage["next"] = json!(GATHER_STAGE_ID);
        }
    }
    pipeline.push(serde_json::to_value(&gather_stage)?);
    Ok(gather_stage)
}

fn build_child(
    parent: &JsonMap<String, JsonValue>,
    parent_run_id: &str,
    index: usize,
    mut item: JsonValue,
    schedule_stream: &str,
) -> Result<Output> {
    let item_map = item
        .as_object_mut()
        .ok_or_else(|| anyhow!("scheduler: fanout_items[{index}] must be a JSON object"))?;
    item_map.entry("item_index").or_insert_with(|| json!(index));

    let child_run_id = format!("{parent_run_id}:item:{index}");
    let mut child = parent.clone();
    for key in ["operation", "parameters", "resource_profile"] {
        if let Some(value) = item_map.get(key) {
            child.insert(key.to_string(), value.clone());
        }
    }
    child.insert("run_id".to_string(), json!(child_run_id));
    child.insert("parent_run_id".to_string(), json!(parent_run_id));
    child.insert("fanout_item".to_string(), item);

    // Reject children the scheduler could not admit, e.g. an invalid per-item
    // resource_profile, before any of them exist; otherwise they would fail
    // outside the scheduler queue and never report to collect.
    let encoded = JsonValue::Object(child).to_string();
    super::decode_schedule_payload(&encoded, &child_run_id).with_context(|| {
        format!("scheduler: fanout_items[{index}] is not a valid schedule request")
    })?;
    Ok(Output::new(schedule_stream, encoded)
        .with_run_id(child_run_id)
        .with_stage("schedule"))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex as StdMutex};

    use serde_json::{Value as JsonValue, json};

    use super::*;
    use crate::config::InputStreamSpec;
    use crate::roles::collect::{
        CollectRole, CollectedMemberResult, InMemoryCollectStore, NoopCollectProgressPersistence,
    };
    use crate::roles::parent_run_state::InMemoryParentRunStateStore;
    use crate::traits::{BoxFuture, RoleEnv, WorkerRole};

    /// (stream, run_id, payload) records.
    type Records = StdMutex<Vec<(String, String, String)>>;

    struct RecordingSink {
        forwards: Records,
        handoffs: Records,
        fail: bool,
    }

    impl RecordingSink {
        fn new(fail: bool) -> Self {
            Self {
                forwards: Records::default(),
                handoffs: Records::default(),
                fail,
            }
        }

        fn children(&self) -> Vec<(String, JsonValue)> {
            self.forwards
                .lock()
                .unwrap()
                .iter()
                .map(|(stream, run_id, payload)| {
                    assert_eq!(stream, "schedule");
                    (run_id.clone(), serde_json::from_str(payload).unwrap())
                })
                .collect()
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

        fn ack_message<'a>(&'a self, _msg: &'a Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a Message,
            _dest_stream: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("fanout tests do not use handoff")) })
        }

        fn handoff_to_run_and_commit<'a>(
            &'a self,
            _msg: &'a Message,
            dest_stream: &'a str,
            payload: &'a str,
            _stage: &'a str,
            run_id: &'a str,
            _finalization_key: &'a str,
            _owner_token: &'a str,
            _recovery_keys: &'a [String],
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.handoffs.lock().unwrap().push((
                    dest_stream.to_string(),
                    run_id.to_string(),
                    payload.to_string(),
                ));
                Ok(String::new())
            })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a Message,
            outputs: &'a [Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async move {
                if self.fail {
                    return Err(anyhow!("forward_many failed"));
                }
                let mut forwards = self.forwards.lock().unwrap();
                for output in outputs {
                    assert_eq!(output.stage(), Some("schedule"));
                    forwards.push((
                        output.stream().to_string(),
                        output.run_id().unwrap_or_default().to_string(),
                        output.payload().to_string(),
                    ));
                }
                Ok(vec![String::new(); outputs.len()])
            })
        }
    }

    fn parent_payload() -> JsonValue {
        json!({
            "run_id": "parent-run",
            "workflow_id": "demo",
            "operation": "run",
            "parameters": {"num_steps": 4},
            "resource_profile": {"executor_class": "python.gpu.demo", "gpus_required": 1, "memory_mb": 1000},
            "fanout_profile": {"max_in_flight": 2},
            "fanout_items": [
                {"item_index": 0, "parameters": {"num_steps": 4, "seed": 1000}},
                {"parameters": {"num_steps": 4, "seed": 1001}, "operation": "member"}
            ],
            "stage_context": {
                "current_stage_id": "schedule",
                "current_phase": "schedule",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "postprocess"},
                    {"id": "postprocess", "phase": "postprocess", "queue": "postprocess", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
    }

    fn parent_msg(payload: &JsonValue) -> Message {
        Message::new(
            "1-0",
            "test:schedule",
            "schedule:grp",
            "parent-run",
            payload.to_string(),
            "schedule",
        )
    }

    /// Follows `next` from stage `from` through the payload's pipeline.
    fn stage_after(payload: &JsonValue, from: &str) -> StageDescriptor {
        let mut context: StageContext =
            serde_json::from_value(payload["stage_context"].clone()).unwrap();
        let current = context
            .pipeline
            .iter()
            .find(|stage| stage.id == from)
            .unwrap();
        context.current_phase = current.phase.clone();
        context.current_stage_id = from.to_string();
        context.next_stage("test").unwrap()
    }

    async fn expand(
        payload: JsonValue,
        sink: &RecordingSink,
        store: &InMemoryCollectStore,
    ) -> Result<()> {
        expand_fanout_parent(&parent_msg(&payload), payload, "schedule", store, sink).await
    }

    #[test]
    fn is_fanout_parent_requires_non_empty_items() {
        assert!(is_fanout_parent(&parent_payload()));
        assert!(!is_fanout_parent(&json!({"fanout_items": []})));
        assert!(!is_fanout_parent(&json!({"fanout_items": null})));
        assert!(!is_fanout_parent(&json!({"workflow_id": "demo"})));
        assert!(is_fanout_parent(
            &json!({"fanout_items": {"item_index": 0}})
        ));
    }

    #[tokio::test]
    async fn rejects_items_the_scheduler_cannot_admit() {
        let mut payload = parent_payload();
        payload["fanout_items"][1]["resource_profile"] =
            json!({"executor_class": "python.gpu.demo", "gpus_required": 0, "memory_mb": 1000});
        let store = InMemoryCollectStore::new();
        let error = expand(payload, &RecordingSink::new(false), &store)
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("fanout_items[1] is not a valid schedule request"),
            "{error:#}"
        );
        assert!(store.get_group("parent-run").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rejects_malformed_fanout_items() {
        let mut payload = parent_payload();
        payload["fanout_items"] = json!({"item_index": 0});
        let error = expand(
            payload,
            &RecordingSink::new(false),
            &InMemoryCollectStore::new(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("non-empty array"), "{error:#}");
    }

    #[tokio::test]
    async fn rejects_pipeline_that_already_uses_gather_stage_id() {
        let mut payload = parent_payload();
        payload["stage_context"]["pipeline"][3]["id"] = json!("_gather");
        payload["stage_context"]["pipeline"][2]["next"] = json!("_gather");
        let error = expand(
            payload,
            &RecordingSink::new(false),
            &InMemoryCollectStore::new(),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("is reserved for fanout"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn expands_parent_into_child_schedule_requests() {
        let sink = RecordingSink::new(false);
        let store = InMemoryCollectStore::new();
        expand(parent_payload(), &sink, &store).await.unwrap();

        let children = sink.children();
        assert_eq!(children.len(), 2);
        let (first_run_id, first) = &children[0];
        assert_eq!(first_run_id, "parent-run:item:0");
        assert_eq!(first["run_id"], "parent-run:item:0");
        assert_eq!(first["parent_run_id"], "parent-run");
        assert_eq!(first["parameters"]["seed"], 1000);
        assert_eq!(first["operation"], "run");
        assert_eq!(
            first["fanout_profile"],
            json!({"max_in_flight": 2, "item_count": 2})
        );
        assert_eq!(first["stage_context"]["current_stage_id"], "schedule");
        assert!(first.get("fanout_items").is_none());

        let (second_run_id, second) = &children[1];
        assert_eq!(second_run_id, "parent-run:item:1");
        assert_eq!(second["operation"], "member");
        assert_eq!(second["fanout_item"]["item_index"], 1);
    }

    #[tokio::test]
    async fn splices_gather_stage_after_execute_for_children_and_collect_group() {
        let sink = RecordingSink::new(false);
        let store = InMemoryCollectStore::new();
        expand(parent_payload(), &sink, &store).await.unwrap();

        let (_, child) = &sink.children()[0];
        assert_eq!(stage_after(child, "schedule").id, "execute");
        let gather = stage_after(child, "execute");
        assert_eq!(
            (
                gather.id.as_str(),
                gather.phase.as_str(),
                gather.queue.as_str()
            ),
            ("_gather", "collect", "collect")
        );
        assert_eq!(gather.next.as_deref(), Some("postprocess"));

        let group = store.get_group("parent-run").await.unwrap().unwrap();
        assert_eq!(group.expected_count, 2);
        assert_eq!(group.workflow_id, "demo");
        assert_eq!(group.parent_payload["run_id"], "parent-run");
        assert_eq!(
            group.parent_payload["stage_context"]["current_stage_id"],
            "_gather"
        );
        assert_eq!(
            group.parent_payload["stage_context"]["current_phase"],
            "collect"
        );
        assert!(group.parent_payload.get("fanout_items").is_none());
    }

    #[tokio::test]
    async fn rejects_item_count_mismatch() {
        let mut payload = parent_payload();
        payload["fanout_profile"]["item_count"] = json!(3);
        let store = InMemoryCollectStore::new();
        let error = expand(payload, &RecordingSink::new(false), &store)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error:#}");
        assert!(store.get_group("parent-run").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rejects_schedule_not_followed_by_execute() {
        let mut payload = parent_payload();
        payload["stage_context"]["pipeline"][1]["next"] = json!("postprocess");
        let store = InMemoryCollectStore::new();
        let error = expand(payload, &RecordingSink::new(false), &store)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("to be followed by an execute stage, got 'postprocess'"),
            "{error:#}"
        );
        assert!(store.get_group("parent-run").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rejects_execute_without_valid_gather_target() {
        let mut payload = parent_payload();
        payload["stage_context"]["pipeline"][2]["next"] = json!("schedule");
        let error = expand(
            payload,
            &RecordingSink::new(false),
            &InMemoryCollectStore::new(),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("followed by postprocess, publish, or results"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn keeps_collect_group_when_forward_outcome_is_ambiguous() {
        // The forward may have committed even though the client saw an error, so
        // children can already be reporting into the group.
        let store = InMemoryCollectStore::new();
        let error = expand(parent_payload(), &RecordingSink::new(true), &store)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("failed to fan out"), "{error:#}");
        store
            .add_result(
                "parent-run",
                CollectedMemberResult {
                    child_run_id: "parent-run:item:0".to_string(),
                    item_index: 0,
                    fanout_item: json!({}),
                    result: json!({"status": "succeeded"}),
                },
            )
            .await
            .unwrap();

        // A redelivered parent expands again without resetting the group.
        expand(parent_payload(), &RecordingSink::new(false), &store)
            .await
            .unwrap();
        let group = store.get_group("parent-run").await.unwrap().unwrap();
        assert_eq!(group.results.len(), 1);
    }

    #[tokio::test]
    async fn failed_child_envelope_targets_gather_stage_only_for_children() {
        let sink = RecordingSink::new(false);
        expand(parent_payload(), &sink, &InMemoryCollectStore::new())
            .await
            .unwrap();
        let (_, child) = &sink.children()[1];

        let envelope: JsonValue =
            serde_json::from_str(&failed_child_envelope(child, "no worker").unwrap()).unwrap();
        assert_eq!(
            envelope["result"],
            json!({"status": "failed", "error": "no worker"})
        );
        assert_eq!(envelope["stage_context"]["current_stage_id"], "_gather");
        assert_eq!(envelope["stage_context"]["current_phase"], "collect");
        assert_eq!(envelope["fanout_item"]["item_index"], 1);

        assert!(failed_child_envelope(&parent_payload(), "no worker").is_none());
    }

    #[tokio::test]
    async fn collect_finalizes_parent_after_all_children_are_gathered() {
        let store = Arc::new(InMemoryCollectStore::new());
        let sink = RecordingSink::new(false);
        expand(parent_payload(), &sink, store.as_ref())
            .await
            .unwrap();
        let collect_env = RoleEnv {
            role_name: "collect".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "collect".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec![],
            role_config: None,
            python_runtime_envs: Default::default(),
        };
        let (collect, _) = CollectRole::from_env_with_store(
            &collect_env,
            store,
            Arc::new(NoopCollectProgressPersistence),
            Arc::new(InMemoryParentRunStateStore::new()),
        )
        .unwrap();

        // Stand in for dispatch + execute: attach a result and hand off to the
        // stage after execute, exactly as the execute worker does.
        for (run_id, mut child) in sink.children() {
            let gather = stage_after(&child, "execute");
            child["result"] = json!({"status": "succeeded", "seed": child["parameters"]["seed"]});
            child["stage_context"]["current_stage_id"] = json!(gather.id);
            child["stage_context"]["current_phase"] = json!(gather.phase);
            let msg = Message::new(
                "2-0",
                "test:collect",
                "collect:grp",
                run_id,
                child.to_string(),
                "collect",
            );
            collect.handle(&msg, &gather.queue, &sink).await.unwrap();
        }

        let handoffs = sink.handoffs.lock().unwrap().clone();
        assert_eq!(handoffs.len(), 1);
        let (stream, run_id, payload) = &handoffs[0];
        assert_eq!(
            (stream.as_str(), run_id.as_str()),
            ("postprocess", "parent-run")
        );
        let parent: JsonValue = serde_json::from_str(payload).unwrap();
        assert_eq!(parent["stage_context"]["current_stage_id"], "postprocess");
        assert_eq!(parent["result"]["status"], "succeeded");
        let seeds: Vec<_> = parent["result"]["child_results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|child| child["result"]["seed"].clone())
            .collect();
        assert_eq!(seeds, vec![json!(1000), json!(1001)]);
    }
}
