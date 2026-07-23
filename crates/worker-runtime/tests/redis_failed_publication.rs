/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod support;

use std::collections::HashMap;

use worker_runtime::traits::MessageSink;
use worker_runtime::transport::redis::RedisTransport;

use support::spawn_test_queue_manager;

async fn run_fields(qm: &scicomp_rq::QueueManager, run_id: &str) -> HashMap<String, String> {
    let mut conn = qm.connection();
    redis::cmd("HGETALL")
        .arg(format!("run:{run_id}"))
        .query_async(&mut conn)
        .await
        .expect("run fields should be readable")
}

#[tokio::test]
async fn mark_request_failed_terminalizes_pending_and_uploading_publication() {
    let (_redis, qm) = spawn_test_queue_manager("failed-publication-state").await;
    let transport = RedisTransport::new(qm.clone(), "test:");
    let mut conn = qm.connection();

    let _: usize = redis::cmd("HSET")
        .arg("run:pending-publication")
        .arg("status")
        .arg("queued")
        .arg("output_location")
        .arg("local_and_cloud")
        .query_async(&mut conn)
        .await
        .expect("pending publication run should be seeded");
    let _: usize = redis::cmd("HSET")
        .arg("run:local-only")
        .arg("status")
        .arg("queued")
        .arg("output_location")
        .arg("local")
        .query_async(&mut conn)
        .await
        .expect("local-only run should be seeded");
    let _: usize = redis::cmd("HSET")
        .arg("run:publication-complete")
        .arg("status")
        .arg("running")
        .arg("output_location")
        .arg("local_and_cloud")
        .arg("output_publication_status")
        .arg("uploaded")
        .arg("publish_completed_at")
        .arg("123")
        .arg("published_artifact_count")
        .arg("2")
        .query_async(&mut conn)
        .await
        .expect("completed publication run should be seeded");
    let _: usize = redis::cmd("HSET")
        .arg("run:publication-uploading")
        .arg("status")
        .arg("running")
        .arg("output_location")
        .arg("local_and_cloud")
        .arg("output_publication_status")
        .arg("uploading")
        .arg("publish_started_at")
        .arg("100")
        .query_async(&mut conn)
        .await
        .expect("uploading publication run should be seeded");

    for run_id in [
        "pending-publication",
        "local-only",
        "publication-complete",
        "publication-uploading",
    ] {
        transport
            .mark_request_failed(run_id)
            .await
            .expect("DLQ failure state should persist");
    }

    let pending = run_fields(&qm, "pending-publication").await;
    assert_eq!(pending.get("status").map(String::as_str), Some("failed"));
    assert_eq!(
        pending.get("output_publication_status").map(String::as_str),
        Some("skipped")
    );
    assert_eq!(
        pending.get("published_artifact_count").map(String::as_str),
        Some("0")
    );
    let publish_completed_at = pending
        .get("publish_completed_at")
        .expect("skipped publication should have a completion timestamp");
    assert!(
        publish_completed_at.parse::<u64>().is_ok(),
        "completion timestamp should be Unix seconds: {publish_completed_at:?}"
    );

    let local_only = run_fields(&qm, "local-only").await;
    assert_eq!(local_only.get("status").map(String::as_str), Some("failed"));
    assert!(!local_only.contains_key("output_publication_status"));
    assert!(!local_only.contains_key("publish_completed_at"));
    assert!(!local_only.contains_key("published_artifact_count"));

    let complete = run_fields(&qm, "publication-complete").await;
    assert_eq!(complete.get("status").map(String::as_str), Some("failed"));
    assert_eq!(
        complete
            .get("output_publication_status")
            .map(String::as_str),
        Some("uploaded")
    );
    assert_eq!(
        complete.get("publish_completed_at").map(String::as_str),
        Some("123")
    );
    assert_eq!(
        complete.get("published_artifact_count").map(String::as_str),
        Some("2")
    );

    let uploading = run_fields(&qm, "publication-uploading").await;
    assert_eq!(uploading.get("status").map(String::as_str), Some("failed"));
    assert_eq!(
        uploading
            .get("output_publication_status")
            .map(String::as_str),
        Some("failed")
    );
    assert_eq!(
        uploading
            .get("published_artifact_count")
            .map(String::as_str),
        Some("0")
    );
    let publish_completed_at = uploading
        .get("publish_completed_at")
        .expect("failed publication should have a completion timestamp");
    assert!(
        publish_completed_at.parse::<u64>().is_ok(),
        "completion timestamp should be Unix seconds: {publish_completed_at:?}"
    );
}
