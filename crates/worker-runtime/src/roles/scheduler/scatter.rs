/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Result, ensure};
use serde_json::{Value, json};

use super::{SchedulerRole, decode_schedule_payload};
use crate::roles::round_state::{RoundPlan, identity};
use crate::roles::scatter::{ScatterPayloadTarget, ScatterResult, project_scatter_payload};
use crate::roles::stage::StageContext;

/// Build and validate *every* child before touching Redis. Plugins own scientific
/// parameters; scheduling must never rewrite their numerical batch size or seed.
fn build_plan(payload: &Value, run_id: &str) -> Result<RoundPlan> {
    ensure!(
        payload["run_id"].as_str() == Some(run_id),
        "scatter parent run_id mismatch"
    );
    let invocation = payload["stage_invocation_id"]
        .as_str()
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("scatter requires a durable stage_invocation_id"))?;
    let context: StageContext = serde_json::from_value(payload["stage_context"].clone())?;
    let instruction: ScatterResult = serde_json::from_value(payload["scatter"].clone())?;
    instruction.validate(&context, payload["parent_run_id"].as_str())?;
    ensure!(
        context
            .pipeline
            .iter()
            .filter(|stage| stage.id == context.current_stage_id
                && stage.phase == context.current_phase)
            .count()
            == 1,
        "scatter source stage does not match pipeline"
    );
    let round_id = identity(&[run_id, invocation]);
    let mut children = Vec::with_capacity(instruction.children.len());
    for (index, child) in instruction.children.iter().enumerate() {
        let child_run_id = format!("{run_id}:round:{round_id}:item:{index}");
        let mut value = project_scatter_payload(payload, ScatterPayloadTarget::Child)?;
        let map = value.as_object_mut().unwrap();
        map.insert("run_id".into(), json!(child_run_id));
        map.insert("parent_run_id".into(), json!(run_id));
        map.insert("operation".into(), json!(child.operation));
        map.insert("parameters".into(), json!(child.parameters));
        if let Some(profile) = &child.resource_profile {
            // Explicit child requirements replace, rather than partly override, the parent's.
            map.insert("resource_profile".into(), json!(profile));
        }
        if let Some(profile) = &child.batch_profile {
            map.insert("batch_profile".into(), json!(profile));
        }
        map.insert(
            "fanout_profile".into(),
            json!({
                "item_count": instruction.children.len(), "max_in_flight": instruction.max_in_flight
            }),
        );
        map.insert(
            "round_context".into(),
            json!({
                "round_id": round_id, "child_index": index, "attempt": 1,
                "attempt_token": identity(&[&round_id, &index.to_string(), "1"])
            }),
        );
        map.insert(
            "dispatch_stage_id".into(),
            json!(instruction.child_stage_id),
        );
        // Child admission deliberately uses the ordinary single-request scheduler.
        decode_schedule_payload(&serde_json::to_string(&value)?, &child_run_id)?;
        children.push(value);
    }
    Ok(RoundPlan {
        round_id,
        parent_run_id: run_id.into(),
        stage_invocation_id: invocation.into(),
        continuation_stage_id: instruction.continuation_stage_id,
        parent_payload: payload.clone(),
        children,
        max_attempts: 3,
        schedule_stream: "schedule".into(),
    })
}

impl SchedulerRole {
    pub(super) async fn register_scatter(&self, payload: &Value, run_id: &str) -> Result<()> {
        let mut plan = build_plan(payload, run_id)?;
        plan.max_attempts = self.config.scatter_max_attempts;
        plan.schedule_stream = self.schedule_stream.clone();
        self.rounds.register(&plan, &self.schedule_stream).await?;
        // Engine ACK follows this commit. If it crashes before ACK, replay compares
        // the accepted digest and does not enqueue any children a second time.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::round_state::RedisRoundStore;
    use redis::AsyncCommands;

    mod support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    fn payload() -> Value {
        json!({
            "run_id": "parent", "workflow_id": "synthetic", "stage_invocation_id": "invocation-1",
            "allocation_id": "parent-attempt-must-not-leak",
            "future_scatter_field": "must-not-leak",
            "resource_profile": {"gpus_required": 1, "memory_mb": 128, "executor_class": "python"},
            "stage_context": {"current_stage_id": "prepare", "current_phase": "prepare", "pipeline": [
                {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "other"},
                {"id": "other", "phase": "execute", "queue": "execute", "next": null},
                {"id": "child", "phase": "execute", "queue": "execute", "next": null},
                {"id": "combine", "phase": "execute", "queue": "execute", "next": null}
            ]},
            "scatter": {"kind": "scatter", "child_stage_id": "child", "continuation_stage_id": "combine",
                "max_in_flight": 2, "children": [
                    {"operation": "run", "parameters": {"batch_size": 2, "seed": 9007199254740993u64}},
                    {"operation": "run", "parameters": {"batch_size": 1, "seed": 0}}
                ]}
        })
    }

    #[test]
    fn stable_identity_order_parameters_and_explicit_target() {
        let input = payload();
        let plan = build_plan(&input, "parent").unwrap();
        let replay = build_plan(&input, "parent").unwrap();
        assert_eq!(plan.round_id, replay.round_id);
        assert_eq!(plan.children, replay.children);
        assert_eq!(plan.parent_payload, input);
        for (index, child) in plan.children.iter().enumerate() {
            assert_eq!(
                child["parameters"],
                input["scatter"]["children"][index]["parameters"]
            );
            assert_eq!(child["round_context"]["child_index"], index);
            assert_eq!(child["round_context"]["attempt"], 1);
            assert!(child.get("scatter").is_none());
            assert!(child.get("allocation_id").is_none());
            assert!(child.get("future_scatter_field").is_none());
            let mut dispatched = child.clone();
            super::super::advance_stage_context_for_dispatch(&mut dispatched, "execute");
            assert_eq!(dispatched["stage_context"]["current_stage_id"], "child");
        }
        let mut next = input;
        next["stage_invocation_id"] = json!("invocation-2");
        let next = build_plan(&next, "parent").unwrap();
        assert_ne!(plan.round_id, next.round_id);
        assert_ne!(plan.children[0]["run_id"], next.children[0]["run_id"]);
        assert_ne!(
            plan.children[0]["round_context"]["attempt_token"],
            next.children[0]["round_context"]["attempt_token"]
        );
    }

    #[test]
    fn reject_invalid_identity_source_and_late_child_before_registration() {
        for change in 0..5 {
            let mut input = payload();
            match change {
                0 => input["stage_invocation_id"] = Value::Null,
                1 => input["run_id"] = json!("different"),
                2 => input["stage_context"]["current_stage_id"] = json!("missing"),
                3 => {
                    input["scatter"]["children"][1]["resource_profile"] =
                        json!({"gpus_required": 1})
                }
                _ => {
                    input["scatter"]["children"][1]["batch_profile"] =
                        json!({"max_batch_size": "invalid"})
                }
            }
            assert!(build_plan(&input, "parent").is_err(), "case {change}");
        }
    }

    #[tokio::test]
    async fn redis_registration_survives_lost_ack_and_rejects_conflicting_replay() {
        let (_server, qm) = support::spawn_test_queue_manager("round-replay").await;
        let store = RedisRoundStore::new(qm.clone(), "test:".into());
        let plan = build_plan(&payload(), "parent").unwrap();
        let (first, duplicate) = tokio::join!(
            store.register(&plan, "schedule"),
            store.register(&plan, "schedule")
        );
        assert_ne!(first.unwrap(), duplicate.unwrap());
        // Simulate process loss after Redis committed, before the source ACK.
        drop(store);
        let restarted = RedisRoundStore::new(qm.clone(), "test:".into());
        assert!(!restarted.register(&plan, "schedule").await.unwrap());
        let mut changed = payload();
        changed["scatter"]["children"][1]["parameters"]["seed"] = json!(10);
        let error = restarted
            .register(&build_plan(&changed, "parent").unwrap(), "schedule")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ROUND_PLAN_CONFLICT"));
        let mut conn = qm.connection();
        let count: usize = conn.xlen("test:schedule").await.unwrap();
        assert_eq!(count, 2);
        let stored: String = conn
            .hget(format!("test:round:{}", plan.round_id), "plan")
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&stored).unwrap(),
            serde_json::to_value(&plan).unwrap()
        );
        let stream: redis::streams::StreamRangeReply =
            conn.xrange_all("test:schedule").await.unwrap();
        for (index, entry) in stream.ids.iter().enumerate() {
            let encoded: String = redis::from_redis_value(&entry.map["payload"]).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&encoded).unwrap(),
                plan.children[index]
            );
        }
    }

    #[tokio::test]
    async fn redis_active_round_is_distinct_from_terminal_parent() {
        let (_server, qm) = support::spawn_test_queue_manager("round-successive").await;
        let store = RedisRoundStore::new(qm.clone(), "test:".into());
        let plan = build_plan(&payload(), "parent").unwrap();
        store.register(&plan, "schedule").await.unwrap();
        let mut next_input = payload();
        next_input["stage_invocation_id"] = json!("invocation-2");
        let next = build_plan(&next_input, "parent").unwrap();
        assert!(
            store
                .register(&next, "schedule")
                .await
                .unwrap_err()
                .to_string()
                .contains("PARENT_ROUND_ACTIVE")
        );
        // Emulate phase 4's committed round completion, not parent termination.
        let mut conn = qm.connection();
        let _: () = conn
            .hset(
                format!("test:round:{}", plan.round_id),
                "status",
                "succeeded",
            )
            .await
            .unwrap();
        let _: usize = conn.del("test:active_round:parent").await.unwrap();
        assert!(store.register(&next, "schedule").await.unwrap());
        assert!(!store.register(&plan, "schedule").await.unwrap());
        let active: String = conn.get("test:active_round:parent").await.unwrap();
        assert_eq!(active, next.round_id);
        let terminal: bool = conn.exists("parent_terminal:parent").await.unwrap();
        assert!(!terminal);
        let count: usize = conn.xlen("test:schedule").await.unwrap();
        assert_eq!(count, 4);
    }

    #[tokio::test]
    async fn redis_rejects_terminal_parent_and_wrong_type_without_partial_children() {
        let (_server, qm) = support::spawn_test_queue_manager("round-reject").await;
        let store = RedisRoundStore::new(qm.clone(), "test:".into());
        let plan = build_plan(&payload(), "parent").unwrap();
        let mut conn = qm.connection();
        let _: () = conn
            .set("parent_terminal:parent", "terminal")
            .await
            .unwrap();
        assert!(
            store
                .register(&plan, "schedule")
                .await
                .unwrap_err()
                .to_string()
                .contains("PARENT_TERMINAL")
        );
        let _: usize = conn.del("parent_terminal:parent").await.unwrap();
        let _: () = conn.set("test:schedule", "not a stream").await.unwrap();
        assert!(
            store
                .register(&plan, "schedule")
                .await
                .unwrap_err()
                .to_string()
                .contains("ROUND_KEY_WRONG_TYPE")
        );
        let state_exists: bool = conn
            .exists(format!("test:round:{}", plan.round_id))
            .await
            .unwrap();
        assert!(!state_exists);
        let active_exists: bool = conn.exists("test:active_round:parent").await.unwrap();
        assert!(!active_exists);
        let _: usize = conn.del("test:schedule").await.unwrap();
        assert!(store.register(&plan, "schedule").await.unwrap());
    }
}
