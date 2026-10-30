/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU component checks for the native registry protocol with the existing
//! scheduler's public role and periodic-discovery interfaces. No C++ host or
//! GPU is needed: only local GPU inventory is replaced; registry and stream
//! operations use a real Redis server and the production Redis transport.

#[path = "../../worker-runtime/tests/support/mod.rs"]
mod support;

use std::ffi::OsString;
use std::sync::LazyLock;

use native_worker::registry::WorkerRegistration;
use redis::streams::StreamRangeReply;
use scicomp_rq::{Message, QueueManager};
use serde_json::{Value, json};
use worker_runtime::config::RuntimeConfig;
use worker_runtime::retry_dlq::RetryDlqPolicy;
use worker_runtime::roles::scheduler::SchedulerRole;
use worker_runtime::traits::{BackgroundTask, MessageSink, QueueTransport, WorkerRole};
use worker_runtime::transport::redis::RedisTransport;

const SLOT: &str = "native:demo:replica-0:0";
const EXECUTE: &str = "execute.native.demo.replica-0.0";
const MAX_RETRIES: usize = 2;
static ENV_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

struct DiscoveryOverride {
    previous: Option<OsString>,
    _lock: tokio::sync::MutexGuard<'static, ()>,
}

impl DiscoveryOverride {
    async fn install() -> Self {
        let lock = ENV_LOCK.lock().await;
        let previous = std::env::var_os("SCHEDULER_DISCOVERY_JSON");
        // SAFETY: this integration-test binary serializes its environment
        // access with ENV_LOCK and uses current-thread Tokio runtimes.
        unsafe {
            std::env::set_var(
                "SCHEDULER_DISCOVERY_JSON",
                json!([{
                    "resource_id": 0,
                    "stream_name": "unused.local.inventory.stream",
                    "total_memory_mb": 16000,
                    "used_memory_mb": 0
                }])
                .to_string(),
            );
        }
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for DiscoveryOverride {
    fn drop(&mut self) {
        // SAFETY: the guard still owns ENV_LOCK during restoration.
        unsafe {
            match self.previous.as_ref() {
                Some(previous) => std::env::set_var("SCHEDULER_DISCOVERY_JSON", previous),
                None => std::env::remove_var("SCHEDULER_DISCOVERY_JSON"),
            }
        }
    }
}

fn metadata() -> Value {
    json!({
        "stream": EXECUTE,
        "device_index": 0,
        "memory_mb": 16000,
        "device_kind": "gpu",
        "executor_class": "native.demo",
        "workflow_id": "demo",
        "tags": ["demo"]
    })
}

struct Harness {
    _redis: support::TestRedisServer,
    qm: QueueManager,
    role: SchedulerRole,
    tasks: Vec<Box<dyn BackgroundTask>>,
    sink: RedisTransport,
}

impl Harness {
    async fn new(name: &str) -> Self {
        let (server, qm) = support::spawn_test_queue_manager(name).await;
        let config: RuntimeConfig = serde_json::from_value(json!({
            "stream_prefix": "test:",
            "streams": ["schedule", "release", EXECUTE, "dlq"],
            "roles": {"scheduler": {
                "inputs": [
                    {"stream":"schedule", "max_dequeue_items":1, "poll_interval_ms":10, "block_ms":1},
                    {"stream":"release", "max_dequeue_items":1, "poll_interval_ms":10, "block_ms":1}
                ],
                "outputs": [],
                "config": {"batching_enabled": false, "gpu_discovery_interval_secs": 1}
            }}
        }))
        .unwrap();
        config.validate().unwrap();
        let (role, tasks) = SchedulerRole::from_env(
            &config.resolve_env("scheduler").unwrap(),
            qm.clone(),
            RetryDlqPolicy::new(MAX_RETRIES, "dlq"),
            None,
        )
        .unwrap();
        let sink = RedisTransport::new(qm.clone(), "test:");
        sink.create_consumer_group("schedule", "schedule:grp")
            .await
            .unwrap();
        Self {
            _redis: server,
            qm,
            role,
            tasks,
            sink,
        }
    }

    async fn register(&self) -> WorkerRegistration {
        WorkerRegistration::warming(self.qm.clone(), SLOT.into(), metadata())
            .await
            .unwrap()
    }

    async fn registry(&self) -> Value {
        let value: String = redis::cmd("HGET")
            .arg("gpu:registry")
            .arg(SLOT)
            .query_async(&mut self.qm.connection())
            .await
            .expect("logical slot must remain discoverable across process retirement");
        serde_json::from_str(&value).unwrap()
    }

    async fn task(&self, name: &str) {
        self.tasks
            .iter()
            .find(|task| task.name() == name)
            .expect("scheduler exposes its existing background task")
            .run(&self.sink)
            .await
            .unwrap();
    }

    async fn queue(&self, run_id: &str) -> Message {
        self.sink
            .enqueue(
                "schedule",
                run_id,
                &json!({
                    "workflow_id": "demo",
                    "runtime": {"kind":"cpp", "executor_class":"native.demo"},
                    "resource_profile": {"gpus_required":1, "memory_mb":2048, "executor_class":"native.demo"}
                })
                .to_string(),
                "schedule",
            )
            .await
            .unwrap();
        let mut messages = self
            .sink
            .poll_stream("schedule", "scheduler-test", 1, 1)
            .await
            .unwrap();
        assert_eq!(messages.len(), 1);
        let msg = messages.remove(0);
        let error = self
            .role
            .handle(&msg, "schedule", &self.sink)
            .await
            .expect_err("scheduler holds accepted work for background placement");
        assert_eq!(
            error.to_string(),
            "scheduler: request queued for background scheduling"
        );
        msg
    }

    async fn pending(&self) -> Vec<(String, String, u64, u64)> {
        redis::cmd("XPENDING")
            .arg("test:schedule")
            .arg("schedule:grp")
            .arg("-")
            .arg("+")
            .arg(10)
            .query_async(&mut self.qm.connection())
            .await
            .unwrap()
    }

    async fn rows(&self, logical_stream: &str) -> StreamRangeReply {
        redis::cmd("XRANGE")
            .arg(format!("test:{logical_stream}"))
            .arg("-")
            .arg("+")
            .query_async(&mut self.qm.connection())
            .await
            .unwrap()
    }

    async fn assert_waiting(&self, msg: &Message, already_dispatched: usize) {
        for _ in 0..MAX_RETRIES + 3 {
            self.task("scheduler_task").await;
            assert_eq!(self.rows(EXECUTE).await.ids.len(), already_dispatched);
            assert!(self.rows("dlq").await.ids.is_empty());
            let pending = self.pending().await;
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].0, msg.id());
            let status: Option<String> = redis::cmd("HGET")
                .arg(format!("run:{}", msg.run_id()))
                .arg("status")
                .query_async(&mut self.qm.connection())
                .await
                .unwrap();
            assert_ne!(status.as_deref(), Some("failed"));
        }
    }

    async fn assert_dispatched_once(&self, run_ids: &[&str]) {
        for _ in 0..3 {
            self.task("scheduler_task").await;
        }
        let rows = self.rows(EXECUTE).await.ids;
        let actual: Vec<String> = rows.iter().map(|row| row.get("run_id").unwrap()).collect();
        assert_eq!(actual, run_ids);
        assert!(
            rows.iter()
                .all(|row| row.get::<String>("stage").as_deref() == Some("execute"))
        );
        assert!(self.rows("dlq").await.ids.is_empty());
        assert!(self.pending().await.is_empty());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_bootstrap_waits_for_self_test_and_existing_periodic_discovery() {
    let _discovery = DiscoveryOverride::install().await;
    let harness = Harness::new("native-scheduler-bootstrap").await;
    let worker = harness.register().await;
    harness.task("resource_discovery").await;
    let msg = harness.queue("native-bootstrap").await;
    harness.assert_waiting(&msg, 0).await;
    assert!(worker.available(&json!({"status":"failed"})).await.is_err());
    harness.task("resource_discovery").await;
    harness.assert_waiting(&msg, 0).await;

    worker.available(&json!({"status":"passed"})).await.unwrap();
    // There is no native-only refresh path: the existing periodic task
    // observes the available transition before placement can proceed.
    harness.assert_waiting(&msg, 0).await;
    harness.task("resource_discovery").await;
    harness.assert_dispatched_once(&["native-bootstrap"]).await;
    worker.release().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn native_restart_retains_queued_work_and_routes_stale_snapshots_to_the_same_slot() {
    let _discovery = DiscoveryOverride::install().await;
    let harness = Harness::new("native-scheduler-restart").await;
    harness
        .sink
        .create_consumer_group(EXECUTE, &format!("{EXECUTE}:grp"))
        .await
        .unwrap();
    let original = harness.register().await;
    original
        .available(&json!({"status":"passed"}))
        .await
        .unwrap();
    harness.task("resource_discovery").await;
    let original_instance = harness.registry().await["worker_instance_id"].clone();

    original.restarting().await.unwrap();
    original.release().await.unwrap();
    assert_eq!(harness.registry().await["status"], "warming");
    assert_eq!(harness.registry().await["stream"], EXECUTE);

    // A scheduler still holding its last available snapshot may enqueue work
    // during process retirement. Its destination is the durable logical slot,
    // so that delivery is retained for the replacement rather than orphaned.
    harness.queue("native-stale-snapshot").await;
    harness.task("scheduler_task").await;
    harness
        .assert_dispatched_once(&["native-stale-snapshot"])
        .await;

    harness.task("resource_discovery").await;
    let queued = harness.queue("native-restart-waiting").await;
    harness.assert_waiting(&queued, 1).await;
    let retired = harness.registry().await;
    assert_eq!(
        retired["model_cache"]["warmup"]["reason"],
        "restart_pending"
    );

    let replacement = harness.register().await;
    let next = harness.registry().await;
    assert_ne!(next["worker_instance_id"], original_instance);
    assert_eq!(next["stream"], retired["stream"]);
    assert_eq!(next["status"], "warming");
    harness.task("resource_discovery").await;
    harness.assert_waiting(&queued, 1).await;
    replacement
        .available(&json!({"status":"passed"}))
        .await
        .unwrap();
    harness.assert_waiting(&queued, 1).await;
    harness.task("resource_discovery").await;
    harness
        .assert_dispatched_once(&["native-stale-snapshot", "native-restart-waiting"])
        .await;
    let deliveries = harness
        .sink
        .poll_stream(EXECUTE, "replacement-process", 2, 1)
        .await
        .unwrap();
    let recovered: Vec<&str> = deliveries.iter().map(Message::run_id).collect();
    assert_eq!(
        recovered,
        ["native-stale-snapshot", "native-restart-waiting"]
    );
    replacement.release().await.unwrap();
}
