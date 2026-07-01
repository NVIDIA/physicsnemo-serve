/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use scicomp_rq::{QueueManager, hash_ops};
use serde_json::{Value as JsonValue, json};

pub struct TestRedisServer {
    child: Child,
    _data_dir: tempfile::TempDir,
}

impl Drop for TestRedisServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn reserve_port(test_name: &str) -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap_or_else(|error| {
            panic!("failed to reserve ephemeral redis port for {test_name}: {error}")
        })
        .local_addr()
        .unwrap_or_else(|error| {
            panic!("listener should expose local addr for {test_name}: {error}")
        })
        .port()
}

async fn wait_for_tcp_listener(port: u16, test_name: &str) {
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("redis-server for {test_name} on port {port} did not become ready in time");
}

pub async fn spawn_test_queue_manager(test_name: &str) -> (TestRedisServer, QueueManager) {
    let port = reserve_port(test_name);
    let data_dir = tempfile::tempdir().unwrap_or_else(|error| {
        panic!("redis data dir should be created for {test_name}: {error}")
    });
    let child = Command::new("redis-server")
        .arg("--save")
        .arg("")
        .arg("--appendonly")
        .arg("no")
        .arg("--bind")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--dir")
        .arg(data_dir.path())
        .arg("--loglevel")
        .arg("warning")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn redis-server for {test_name}: {error}"));
    wait_for_tcp_listener(port, test_name).await;
    let server = TestRedisServer {
        child,
        _data_dir: data_dir,
    };
    let redis_url = format!("redis://127.0.0.1:{port}");
    let qm = QueueManager::new(redis_url.as_str())
        .await
        .unwrap_or_else(|error| {
            panic!("queue manager should connect to test redis for {test_name}: {error}")
        });
    (server, qm)
}

#[allow(dead_code)]
pub async fn seed_registry_from_discovery_json(
    qm: &QueueManager,
    registry_key: &str,
    field_prefix: &str,
) {
    let Ok(raw_json) = std::env::var("SCHEDULER_DISCOVERY_JSON") else {
        eprintln!(
            "seed_registry_from_discovery_json: skipping registry seed for '{registry_key}' \
             because SCHEDULER_DISCOVERY_JSON is unset"
        );
        return;
    };
    if raw_json.trim().is_empty() {
        eprintln!(
            "seed_registry_from_discovery_json: skipping registry seed for '{registry_key}' \
             because SCHEDULER_DISCOVERY_JSON is empty"
        );
        return;
    }

    let discovered: Vec<JsonValue> =
        serde_json::from_str(&raw_json).expect("discovery override should parse for tests");
    let mut conn = qm.connection();
    for (index, gpu) in discovered.into_iter().enumerate() {
        let metadata = json!({
            "stream": gpu["stream_name"].as_str().expect("stream_name should be present"),
            "device_index": gpu["resource_id"]
                .as_u64()
                .expect("resource_id should be present"),
            "memory_mb": gpu["total_memory_mb"].as_u64().expect("total_memory_mb should be present"),
            "device_kind": gpu["device_kind"].as_str().unwrap_or("gpu"),
            "executor_class": gpu.get("executor_class").cloned().unwrap_or(JsonValue::Null),
            "tags": gpu.get("tags").cloned().unwrap_or_else(|| json!([])),
            "status": gpu.get("status").cloned().unwrap_or_else(|| json!("available")),
            "model_cache": gpu.get("model_cache").cloned().unwrap_or_else(|| {
                json!({
                    "schema_version": 1,
                    "scope": "process",
                    "entries": [],
                    "total_entries": 0,
                    "warmup": {
                        "workflow_id": null,
                        "status": "skipped"
                    }
                })
            })
        });
        let field = format!("{field_prefix}:{index}");
        hash_ops::hset(
            &mut conn,
            registry_key,
            field.as_str(),
            metadata.to_string().as_str(),
        )
        .await
        .expect("test registry entry should be written");
    }
}
