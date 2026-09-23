/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

mod support {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
}

fn plan(invocation: &str, phase: &str) -> RoundPlan {
    let round_id = identity(&["parent", invocation]);
    let mut pipeline = vec![
        json!({"id": "scatter", "phase": "execute", "queue": "execute", "next": null}),
        json!({"id": "child", "phase": "execute", "queue": "execute", "next": null}),
        json!({"id": "resume", "phase": phase, "queue": "continuation", "next": null}),
    ];
    if phase != "results" {
        pipeline.push(json!({
            "id": "results",
            "phase": "results",
            "queue": "custom-results",
            "next": null
        }));
    }
    let context = json!({
        "current_stage_id": "scatter",
        "current_phase": "execute",
        "pipeline": pipeline
    });
    RoundPlan {
        parent_run_id: "parent".into(), stage_invocation_id: invocation.into(),
        continuation_stage_id: "resume".into(), max_attempts: 3, schedule_stream: "schedule".into(),
        parent_payload: json!({"run_id": "parent", "workflow_id": "generic", "stage_context": context,
            "resource_profile": {"gpus_required": 1, "memory_mb": 128, "executor_class": "test"},
            "parameters": {"batch_size": 2}, "batch_profile": {"enabled": true},
            "stage_invocation_id": invocation, "allocation_id": "must-not-leak",
            "future_scatter_field": "must-not-leak"}),
        children: (0..2).map(|index| json!({
            "run_id": format!("parent:{round_id}:{index}"), "parent_run_id": "parent", "workflow_id": "generic",
            "round_context": {"round_id": round_id, "child_index": index, "attempt": 1,
                "attempt_token": identity(&[&round_id, &index.to_string(), "1"])},
            "parameters": {"batch_size": 2, "seed": 9007199254740993u64}, "stage_context": context
        })).collect(),
        round_id,
    }
}

fn outcome(child: &Value, result: Value) -> Value {
    let mut value = child.clone();
    value["result"] = result;
    value
}

async fn deliver(store: &RedisRoundStore, child: &Value, result: Value) {
    store
        .gather(child["run_id"].as_str().unwrap(), &outcome(child, result))
        .await
        .unwrap();
}

async fn state(qm: &scicomp_rq::QueueManager, plan: &RoundPlan) -> RoundState {
    let raw: String = qm
        .connection()
        .hget(format!("test:round:{}", plan.round_id), "state")
        .await
        .unwrap();
    serde_json::from_str(&raw).unwrap()
}

async fn messages(qm: &scicomp_rq::QueueManager, stream: &str) -> Vec<Value> {
    let entries: redis::streams::StreamRangeReply = qm
        .connection()
        .xrange_all(format!("test:{stream}"))
        .await
        .unwrap();
    entries
        .ids
        .iter()
        .map(|entry| {
            let raw: String = redis::from_redis_value(&entry.map["payload"]).unwrap();
            serde_json::from_str(&raw).unwrap()
        })
        .collect()
}

#[tokio::test]
async fn concurrent_success_is_ordered_and_continuation_is_committed_once() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-concurrent").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let plan = plan("first", "execute");
    store.register(&plan, "schedule").await.unwrap();
    let active_ttl: i64 = redis::cmd("TTL")
        .arg(format!("test:round:{}", plan.round_id))
        .query_async(&mut qm.connection())
        .await
        .unwrap();
    assert_eq!(active_ttl, -1, "active rounds must not expire");
    tokio::join!(
        deliver(
            &store,
            &plan.children[1],
            json!({"status": "succeeded", "value": 9007199254740993u64})
        ),
        deliver(
            &store,
            &plan.children[0],
            json!({"status": "succeeded", "value": 0})
        )
    );
    let restarted = RedisRoundStore::new(qm.clone(), "test:".into());
    deliver(
        &restarted,
        &plan.children[0],
        json!({"status": "succeeded", "value": 99}),
    )
    .await;
    let entries = messages(&qm, "schedule").await;
    assert_eq!(entries.len(), 3);
    let resumed = &entries[2];
    assert_eq!(resumed["run_id"], "parent");
    assert_eq!(resumed["dispatch_stage_id"], "resume");
    assert_eq!(resumed["child_results"][0]["result"]["value"], 0);
    assert_eq!(
        resumed["child_results"][1]["result"]["value"],
        9007199254740993u64
    );
    assert!(resumed.get("stage_invocation_id").is_none());
    assert!(resumed.get("allocation_id").is_none());
    assert!(resumed.get("future_scatter_field").is_none());
    assert_eq!(resumed["batch_profile"]["enabled"], true);
    assert_eq!(state(&qm, &plan).await.status, Status::Succeeded);
    let terminal_ttl: i64 = redis::cmd("TTL")
        .arg(format!("test:round:{}", plan.round_id))
        .query_async(&mut qm.connection())
        .await
        .unwrap();
    assert!(
        terminal_ttl > 0 && terminal_ttl <= STATE_TOMBSTONE_TTL_SECS as i64,
        "terminal round must have a bounded tombstone TTL, got {terminal_ttl}"
    );
    let terminal: bool = qm
        .connection()
        .exists("parent_terminal:parent")
        .await
        .unwrap();
    assert!(!terminal);
}

#[tokio::test]
async fn successive_rounds_preserve_previous_tombstones() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-successive").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let first = plan("first", "prepare");
    store.register(&first, "schedule").await.unwrap();
    for child in &first.children {
        deliver(&store, child, json!({"status": "succeeded"})).await;
    }
    let second = plan("second", "postprocess");
    store.register(&second, "schedule").await.unwrap();
    deliver(&store, &first.children[0], json!({"status": "failed"})).await;
    assert!(!store.register(&first, "schedule").await.unwrap());
    for child in &second.children {
        deliver(&store, child, json!({"status": "succeeded"})).await;
    }
    assert_eq!(messages(&qm, "continuation").await.len(), 2);
    assert_eq!(state(&qm, &first).await.status, Status::Succeeded);
    assert_eq!(state(&qm, &second).await.status, Status::Succeeded);
}

#[tokio::test]
async fn retries_preserve_successes_and_fence_old_attempts_until_exhaustion() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-retries").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let plan = plan("first", "execute");
    store.register(&plan, "schedule").await.unwrap();
    let _: usize = qm
        .connection()
        .hset("run:parent", "status", "running")
        .await
        .unwrap();
    deliver(
        &store,
        &plan.children[0],
        json!({"status": "succeeded", "value": 42}),
    )
    .await;
    let failure = json!({"status": "failed", "retryable": true, "error": "transient"});
    let mut child = plan.children[1].clone();
    for attempt in 1..=3 {
        deliver(&store, &child, failure.clone()).await;
        // Simulate lost ACK and an old worker returning late after retry dispatch.
        deliver(&store, &child, failure.clone()).await;
        deliver(&store, &child, json!({"status": "succeeded"})).await;
        assert!(
            !store
                .is_current_child(child["run_id"].as_str().unwrap(), &child)
                .await
                .unwrap()
        );
        if attempt < 3 {
            let entries = messages(&qm, "schedule").await;
            let retry = entries.last().unwrap().clone();
            assert_eq!(retry["round_context"]["attempt"], attempt + 1);
            assert_ne!(
                retry["round_context"]["attempt_token"],
                child["round_context"]["attempt_token"]
            );
            assert_eq!(retry["parameters"], child["parameters"]);
            assert!(retry["retry_not_before_ms"].as_u64().unwrap() > 0);
            assert_eq!(state(&qm, &plan).await.status, Status::Waiting);
            child = retry;
        }
    }
    let finished = state(&qm, &plan).await;
    assert_eq!(finished.status, Status::Failed);
    assert_eq!(finished.children[0].result.as_ref().unwrap()["value"], 42);
    assert_eq!(messages(&qm, "schedule").await.len(), 4);
    let status: String = qm.connection().hget("run:parent", "status").await.unwrap();
    assert_eq!(
        status, "running",
        "gather must leave final run status to the results/status path"
    );
    let failed = messages(&qm, "custom-results").await;
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["status"], "failed");
    assert!(
        failed[0]["execution"]["outputs"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn permanent_failure_and_cancellation_never_emit_partial_success() {
    for cancelled in [false, true] {
        let (_server, qm) = support::spawn_test_queue_manager("gather-terminal").await;
        let store = RedisRoundStore::new(qm.clone(), "test:".into());
        let plan = plan("first", "results");
        store.register(&plan, "schedule").await.unwrap();
        if cancelled {
            let _: () = qm
                .connection()
                .set("parent_terminal:parent", "terminal")
                .await
                .unwrap();
        }
        deliver(
            &store,
            &plan.children[0],
            if cancelled {
                json!({"status": "succeeded"})
            } else {
                json!({"status": "failed", "error": "invalid input"})
            },
        )
        .await;
        deliver(&store, &plan.children[1], json!({"status": "succeeded"})).await;
        assert_eq!(
            state(&qm, &plan).await.status,
            if cancelled {
                Status::Cancelled
            } else {
                Status::Failed
            }
        );
        assert_eq!(
            messages(&qm, "continuation").await.len(),
            usize::from(!cancelled)
        );
        assert_eq!(messages(&qm, "schedule").await.len(), 2);
        let round_ttl: i64 = redis::cmd("TTL")
            .arg(format!("test:round:{}", plan.round_id))
            .query_async(&mut qm.connection())
            .await
            .unwrap();
        assert!(round_ttl > 0 && round_ttl <= STATE_TOMBSTONE_TTL_SECS as i64);
        let terminal_ttl: i64 = redis::cmd("TTL")
            .arg("parent_terminal:parent")
            .query_async(&mut qm.connection())
            .await
            .unwrap();
        assert!(terminal_ttl > 0 && terminal_ttl <= STATE_TOMBSTONE_TTL_SECS as i64);
        assert!(
            !store
                .is_current_child(
                    plan.children[1]["run_id"].as_str().unwrap(),
                    &plan.children[1]
                )
                .await
                .unwrap()
        );
    }
}

#[tokio::test]
async fn destination_error_retains_result_for_safe_redelivery() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-destination-error").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let plan = plan("first", "postprocess");
    store.register(&plan, "schedule").await.unwrap();
    deliver(&store, &plan.children[0], json!({"status": "succeeded"})).await;
    let _: () = qm
        .connection()
        .set("test:continuation", "wrong type")
        .await
        .unwrap();
    let last = outcome(&plan.children[1], json!({"status": "succeeded"}));
    assert!(
        store
            .gather(last["run_id"].as_str().unwrap(), &last)
            .await
            .is_err()
    );
    assert_eq!(state(&qm, &plan).await.status, Status::Waiting);
    let _: usize = qm.connection().del("test:continuation").await.unwrap();
    store
        .gather(last["run_id"].as_str().unwrap(), &last)
        .await
        .unwrap();
    store
        .gather(last["run_id"].as_str().unwrap(), &last)
        .await
        .unwrap();
    assert_eq!(messages(&qm, "continuation").await.len(), 1);
}

#[tokio::test]
async fn malformed_identity_is_rejected_without_state_mutation() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-invalid").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let plan = plan("first", "execute");
    store.register(&plan, "schedule").await.unwrap();
    let value = outcome(&plan.children[0], json!({"status": "succeeded"}));
    assert!(store.gather("different-child", &value).await.is_err());
    let mut value = value;
    value["round_context"]["child_index"] = json!(99);
    assert!(
        store
            .gather(plan.children[0]["run_id"].as_str().unwrap(), &value)
            .await
            .is_err()
    );
    assert!(
        state(&qm, &plan)
            .await
            .children
            .iter()
            .all(|child| child.result.is_none())
    );
}

#[tokio::test]
async fn configured_attempt_limit_and_results_continuation_are_respected() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-configured-budget").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let mut first = plan("first", "results");
    first.schedule_stream = "custom-schedule".into();
    store.register(&first, "custom-schedule").await.unwrap();
    for child in &first.children {
        deliver(&store, child, json!({"status": "succeeded", "value": 3})).await;
    }
    let entries = messages(&qm, "continuation").await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["execution"]["status"], "succeeded");
    assert_eq!(
        entries[0]["payload"]["child_results"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let mut second = plan("second", "prepare");
    second.max_attempts = 1;
    store.register(&second, "schedule").await.unwrap();
    deliver(
        &store,
        &second.children[0],
        json!({"status": "failed", "retryable": true}),
    )
    .await;
    assert_eq!(state(&qm, &second).await.status, Status::Failed);
    assert_eq!(messages(&qm, "schedule").await.len(), 2);
}

#[tokio::test]
async fn externally_terminal_parent_status_is_not_overwritten() {
    let (_server, qm) = support::spawn_test_queue_manager("gather-external-terminal").await;
    let store = RedisRoundStore::new(qm.clone(), "test:".into());
    let plan = plan("first", "execute");
    store.register(&plan, "schedule").await.unwrap();
    let _: () = qm
        .connection()
        .hset("run:parent", "status", "failed")
        .await
        .unwrap();
    let _: () = qm
        .connection()
        .set("parent_terminal:parent", "terminal")
        .await
        .unwrap();
    deliver(&store, &plan.children[0], json!({"status": "succeeded"})).await;
    let status: String = qm.connection().hget("run:parent", "status").await.unwrap();
    assert_eq!(status, "failed");
    assert_eq!(state(&qm, &plan).await.status, Status::Cancelled);
}
