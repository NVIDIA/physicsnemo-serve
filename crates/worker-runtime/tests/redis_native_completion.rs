/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod support;

use scicomp_rq::{Message, Output, QueueManager, StreamKey};
use support::spawn_test_queue_manager;
use worker_runtime::traits::MessageSink;
use worker_runtime::transport::redis::RedisTransport;

type PendingSummary = (usize, Option<String>, Option<String>, Vec<(String, usize)>);

async fn pending(qm: &QueueManager) -> Message {
    let stream = StreamKey::new("test:gpu:0");
    qm.create_consumer_group(&stream, "native:grp", "0", true)
        .await
        .unwrap();
    qm.enqueue_to_stream(&stream, "run-1", "{}", "execute")
        .await
        .unwrap();
    qm.read_messages(&stream, "native:grp", "worker-boot-1", 1, 1)
        .await
        .unwrap()
        .remove(0)
}

fn outputs() -> Vec<Output> {
    vec![
        Output::new("test:release", r#"{"resource_id":0,"memory_mb":12000}"#).with_stage("release"),
        Output::new("test:results", r#"{"status":"success"}"#).with_stage("results"),
    ]
}

async fn lengths(qm: &QueueManager) -> (usize, usize, usize) {
    let mut conn = qm.connection();
    let first = redis::cmd("XLEN")
        .arg("test:release")
        .query_async(&mut conn)
        .await
        .unwrap();
    let second = redis::cmd("XLEN")
        .arg("test:results")
        .query_async(&mut conn)
        .await
        .unwrap();
    let pending: PendingSummary = redis::cmd("XPENDING")
        .arg("test:gpu:0")
        .arg("native:grp")
        .query_async(&mut conn)
        .await
        .unwrap();
    (first, second, pending.0)
}

#[tokio::test]
async fn native_redis_sink_prefixes_outputs_and_requires_engine_consumer() {
    let (_server, qm) = spawn_test_queue_manager("native-completion-transport").await;
    let msg = pending(&qm).await;
    let transport = RedisTransport::new(qm.clone(), "test:");
    assert!(
        transport.begin_attempt(&msg, "attempt-1").await.is_err(),
        "direct transport callers cannot bypass expected-consumer injection"
    );
    assert_eq!(
        transport
            .begin_attempt_for_consumer(&msg, "attempt-1", "worker-boot-1")
            .await
            .unwrap(),
        None
    );
    let logical = [
        Output::new("release", outputs()[0].payload()).with_stage("release"),
        Output::new("results", outputs()[1].payload()).with_stage("results"),
    ];
    assert!(
        transport
            .complete_attempt(&msg, &logical, "attempt-1")
            .await
            .is_err()
    );
    let ids = transport
        .complete_attempt_for_consumer(&msg, &logical, "attempt-1", "worker-boot-1")
        .await
        .unwrap();
    assert_eq!(
        transport
            .complete_attempt_for_consumer(&msg, &logical, "attempt-1", "worker-boot-1")
            .await
            .unwrap(),
        ids
    );
    assert_eq!(lengths(&qm).await, (1, 1, 0));
}

#[tokio::test]
async fn native_completion_recovers_lost_response_without_duplicate_release() {
    let (_server, qm) = spawn_test_queue_manager("native-completion-retry").await;
    let msg = pending(&qm).await;
    assert_eq!(
        qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
            .await
            .unwrap(),
        None
    );
    let ids = qm
        .complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(lengths(&qm).await, (1, 1, 0));
    // Model the successful response being lost: retry the same immutable source.
    assert_eq!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
            .await
            .unwrap(),
        ids
    );
    assert_eq!(
        qm.begin_attempt(&msg, "worker-boot-2", "attempt-2")
            .await
            .unwrap(),
        Some(ids)
    );
    assert_eq!(lengths(&qm).await, (1, 1, 0));
}

#[tokio::test]
async fn native_completion_rejects_stale_consumer_and_attempt_without_writes() {
    let (_server, qm) = spawn_test_queue_manager("native-completion-fences").await;
    let msg = pending(&qm).await;
    qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
        .await
        .unwrap();
    assert!(
        qm.begin_attempt(&msg, "worker-boot-1", "different-attempt")
            .await
            .is_err()
    );
    assert!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "wrong-attempt")
            .await
            .is_err()
    );
    assert!(
        qm.begin_attempt(&msg, "other-worker", "attempt-2")
            .await
            .is_err()
    );
    let mut conn = qm.connection();
    let _: Vec<String> = redis::cmd("XCLAIM")
        .arg(msg.stream())
        .arg(msg.group())
        .arg("worker-boot-2")
        .arg(0)
        .arg(msg.id())
        .arg("JUSTID")
        .query_async(&mut conn)
        .await
        .unwrap();
    assert!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
            .await
            .is_err()
    );
    qm.begin_attempt(&msg, "worker-boot-2", "attempt-2")
        .await
        .unwrap();
    assert!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-2", "attempt-1")
            .await
            .is_err()
    );
    assert_eq!(lengths(&qm).await, (0, 0, 1));
    qm.complete_attempt(&msg, &outputs(), "worker-boot-2", "attempt-2")
        .await
        .unwrap();
    assert_eq!(lengths(&qm).await, (1, 1, 0));
}

#[tokio::test]
async fn native_completion_rejects_bad_marker_and_metadata_without_writes() {
    let (_server, qm) = spawn_test_queue_manager("native-completion-marker").await;
    let msg = pending(&qm).await;
    assert!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
            .await
            .is_err(),
        "completion must require an active attempt claim"
    );
    qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
        .await
        .unwrap();
    for invalid in [
        Output::new("test:results", "not-json"),
        Output::new("test:results", "{}").with_stage(""),
        Output::new("test:results", "{}").with_run_id(""),
        Output::new("", "{}"),
    ] {
        assert!(
            qm.complete_attempt(
                &msg,
                &[outputs()[0].clone(), invalid],
                "worker-boot-1",
                "attempt-1"
            )
            .await
            .is_err()
        );
        assert_eq!(lengths(&qm).await, (0, 0, 1));
    }
    let mut conn = qm.connection();
    let state_keys: Vec<String> = redis::cmd("KEYS")
        .arg("native-attempt:*")
        .query_async(&mut conn)
        .await
        .unwrap();
    assert_eq!(state_keys.len(), 1);
    let _: () = redis::cmd("SET")
        .arg(&state_keys[0])
        .arg("wrong-type")
        .query_async(&mut conn)
        .await
        .unwrap();
    assert!(
        qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
            .await
            .is_err()
    );
    assert!(
        qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
            .await
            .is_err()
    );
    assert_eq!(lengths(&qm).await, (0, 0, 1));
}

#[tokio::test]
async fn native_completion_preflights_aliasing_new_destination_and_run_hash() {
    let (_server, qm) = spawn_test_queue_manager("native-completion-alias").await;
    let msg = pending(&qm).await;
    qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
        .await
        .unwrap();
    // Both keys start absent and have valid types in isolation, but the first
    // XADD would make the second output's HSET fail without alias preflight.
    let aliases = [
        outputs()[0].clone(),
        Output::new("run:run-1", "{}").with_stage("results"),
    ];
    assert!(
        qm.complete_attempt(&msg, &aliases, "worker-boot-1", "attempt-1")
            .await
            .is_err()
    );
    assert_eq!(lengths(&qm).await, (0, 0, 1));
    let mut conn = qm.connection();
    let exists: usize = redis::cmd("EXISTS")
        .arg("run:run-1")
        .query_async(&mut conn)
        .await
        .unwrap();
    assert_eq!(exists, 0);
}

#[tokio::test]
async fn native_completion_preflights_all_output_and_run_hash_types() {
    for bad_key in ["test:results", "run:run-1"] {
        let (_server, qm) = spawn_test_queue_manager("native-completion-types").await;
        let msg = pending(&qm).await;
        qm.begin_attempt(&msg, "worker-boot-1", "attempt-1")
            .await
            .unwrap();
        let mut conn = qm.connection();
        let _: () = redis::cmd("SET")
            .arg(bad_key)
            .arg("wrong-type")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(
            qm.complete_attempt(&msg, &outputs(), "worker-boot-1", "attempt-1")
                .await
                .is_err()
        );
        let released: usize = redis::cmd("XLEN")
            .arg("test:release")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(
            released, 0,
            "a later invalid key must not leak the reservation release"
        );
        let acked = qm.ack_message(&msg).await.unwrap();
        assert_eq!(acked, 1, "failed completion must leave the source pending");
    }
}
