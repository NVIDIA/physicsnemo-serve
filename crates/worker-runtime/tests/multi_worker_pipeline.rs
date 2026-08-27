/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test: full pipeline using InMemoryTransport.
//!
//! Wires prefetch -> scheduler -> gpu_stream -> results with a shared transport.
//! The scheduler discovers local GPUs via scheduler discovery logic, not from
//! static config outputs.

mod support;

use std::path::Path;
use std::sync::{Arc, Mutex};

use std::sync::LazyLock;

use anyhow::Result;
use serde_json::json;
use support::{seed_registry_from_discovery_json, spawn_test_queue_manager};
use worker_runtime::config::RuntimeConfig;
use worker_runtime::engine::EngineBuilder;
use worker_runtime::roles;
use worker_runtime::roles::RoleProviders;
use worker_runtime::roles::prefetch::{
    DownloadStats, MaterializationResult, MaterializedArtifact, PlanMaterializer, PrefetchPlanItem,
};
use worker_runtime::roles::results::NoopResultsPersistence;
use worker_runtime::traits::{BoxFuture, QueueTransport};
use worker_runtime::transport::memory::InMemoryTransport;

const EXAMPLE_CONFIG: &str = include_str!("../examples/runtime_config.json");
static ENV_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

fn set_env_var(key: &str, value: Option<&str>) {
    match value {
        Some(v) => {
            // SAFETY: integration tests serialize env access via ENV_LOCK.
            unsafe { std::env::set_var(key, v) };
        }
        None => {
            // SAFETY: integration tests serialize env access via ENV_LOCK.
            unsafe { std::env::remove_var(key) };
        }
    }
}

struct RecordingMaterializer {
    calls: Mutex<Vec<(usize, String)>>,
    result: MaterializationResult,
}
impl RecordingMaterializer {
    fn with_result(result: MaterializationResult) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            result,
        }
    }
    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}
impl PlanMaterializer for RecordingMaterializer {
    fn materialize<'a>(
        &'a self,
        plan: &'a [PrefetchPlanItem],
        _cache_root: &'a Path,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<MaterializationResult>> {
        self.calls
            .lock()
            .unwrap()
            .push((plan.len(), run_id.to_string()));
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

fn plugin_prefetch_payload(run_id: &str, plan: serde_json::Value) -> String {
    json!({
        "run_id": run_id,
        "workflow_id": "demo-plugin",
        "operation": "both",
        "parameters": {
            "batch_size": 128000
        },
        "request": {
            "content_type": "application/json",
            "raw_fields": {
                "batch_size": 128000
            },
            "input_artifacts": []
        },
        "resource_profile": {
            "gpus_required": 1,
            "memory_mb": 4096,
            "executor_class": "python.gpu.physicsnemo",
            "device_kind": "gpu",
            "tags": ["physicsnemo", "gpu"]
        },
        "prefetch_plan": plan,
        "stage_context": {
            "current_stage_id": "prefetch",
            "current_phase": "prefetch",
            "pipeline": [
                {
                    "id": "prefetch",
                    "phase": "prefetch",
                    "queue": "prefetch",
                    "next": "schedule"
                },
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "queue": "schedule",
                    "next": "execute"
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute",
                    "next": "results"
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": null
                }
            ]
        },
        "runtime": {
            "kind": "python",
            "entrypoint": "plugin.py",
            "executor_class": "python.gpu.physicsnemo"
        }
    })
    .to_string()
}

fn pipeline_config() -> RuntimeConfig {
    serde_json::from_value(serde_json::json!({
        "stream_prefix": "",
        "streams": ["prefetch", "schedule", "release", "results"],
        "roles": {
            "prefetch": {
                "inputs": [{"stream": "prefetch", "max_dequeue_items": 4,
                            "poll_interval_ms": 10, "block_ms": 50}],
                "outputs": ["schedule"]
            },
            "scheduler": {
                "inputs": [
                    {"stream": "schedule", "max_dequeue_items": 4,
                     "poll_interval_ms": 10, "block_ms": 50},
                    {"stream": "release", "max_dequeue_items": 2,
                     "poll_interval_ms": 10, "block_ms": 50}
                ],
                "outputs": [],
                "config": {
                    "batching_enabled": false
                }
            },
            "results": {
                "inputs": [{"stream": "results", "max_dequeue_items": 8,
                            "poll_interval_ms": 10, "block_ms": 50}],
                "outputs": []
            }
        }
    }))
    .expect("pipeline config should parse")
}

fn build_engine_for(
    config: &RuntimeConfig,
    role_name: &str,
    transport: Arc<InMemoryTransport>,
    providers: RoleProviders,
) -> worker_runtime::engine::WorkerEngine {
    let env = config.resolve_env(role_name).unwrap();
    let (role, tasks) = roles::build_role(&env, providers).unwrap();

    let mut builder = EngineBuilder::new(config, role_name)
        .transport(transport)
        .role(role)
        .consumer(format!("test-{role_name}"));
    for task in tasks {
        builder = builder.background_task(task);
    }
    builder.build().unwrap()
}

#[tokio::test]
async fn full_pipeline_with_dynamic_gpu_discovery() {
    let _guard = ENV_LOCK.lock().await;
    let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
    set_env_var(
        "SCHEDULER_DISCOVERY_JSON",
        Some(
            r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo","gpu"]}]"#,
        ),
    );

    let config = pipeline_config();
    let (_redis_server, qm) = spawn_test_queue_manager("full-pipeline-scheduler").await;
    seed_registry_from_discovery_json(&qm, "gpu:registry", "multi-worker-test").await;

    // Transport includes the GPU stream even though it's NOT in the config.
    // In production, inference_worker.py creates the stream.
    let transport = Arc::new(InMemoryTransport::new(
        &["prefetch", "schedule", "release", "results", "gpu:ns:pod:0"],
        "",
    ));

    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats::default(),
        artifacts: vec![],
    }));

    let prefetch = build_engine_for(
        &config,
        "prefetch",
        transport.clone(),
        RoleProviders {
            materializer: Some(mat),
            ..RoleProviders::empty()
        },
    );
    let scheduler = build_engine_for(
        &config,
        "scheduler",
        transport.clone(),
        RoleProviders::empty().with_queue_manager(qm),
    );

    // Simulate: inference worker already running on gpu:ns:pod:0, writing results
    // For this test we just manually move the message after scheduler enqueues.

    transport
        .inject(
            "prefetch",
            "run-1",
            &plugin_prefetch_payload(
                "run-1",
                json!([
                    {
                        "source_uri": "https://example.com/reference.txt",
                        "target_artifact_name": "reference",
                        "required": false
                    }
                ]),
            ),
            "prefetch",
        )
        .unwrap();

    // prefetch -> schedule
    let stats = prefetch.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 1);
    assert!(
        !transport.pending_in("schedule").is_empty(),
        "prefetch should enqueue to schedule"
    );

    // first scheduler loop ingests the schedule message; second loop dispatches it
    let stats = scheduler.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 0);
    let stats = scheduler.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 0);
    assert!(
        !transport.pending_in("gpu:ns:pod:0").is_empty(),
        "scheduler should route to dynamically discovered GPU stream"
    );

    // Simulate inference_worker.py completing and writing to results
    let gpu_msgs = transport.pending_in("gpu:ns:pod:0");
    assert_eq!(gpu_msgs.len(), 1);
    assert_eq!(gpu_msgs[0].run_id(), "run-1");

    // Simulate inference_worker.py: drain GPU stream and write to results.
    let _: Vec<scicomp_rq::Message> = transport
        .poll_stream("gpu:ns:pod:0", "sim", 1, 0)
        .await
        .unwrap();
    transport
        .inject(
            "results",
            "run-1",
            r#"{
                "status":"succeeded",
                "completed_at":"2026-02-20T00:00:00Z",
                "payload":"{\"output_path\":\"/out/run-1\",\"execution_time_seconds\":1.0}"
            }"#,
            "results",
        )
        .unwrap();

    // results worker processes the terminal message
    let results_engine = build_engine_for(
        &config,
        "results",
        transport.clone(),
        RoleProviders {
            results_persistence: Some(Arc::new(NoopResultsPersistence::new())),
            ..RoleProviders::empty()
        },
    );
    let stats = results_engine.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 1);

    assert!(
        transport.all_drained(),
        "all streams should be empty after full pipeline"
    );

    set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
}

#[tokio::test]
async fn scheduler_round_robins_requests_across_gpus() {
    let _guard = ENV_LOCK.lock().await;
    let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
    set_env_var(
        "SCHEDULER_DISCOVERY_JSON",
        Some(
            r#"[{"resource_id":0,"stream_name":"gpu_a","total_memory_mb":30000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]},{"resource_id":1,"stream_name":"gpu_b","total_memory_mb":30000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
        ),
    );

    let config = pipeline_config();
    let (_redis_server, qm) = spawn_test_queue_manager("scheduler-memory-saturation").await;
    seed_registry_from_discovery_json(&qm, "gpu:registry", "multi-worker-test").await;
    let transport = Arc::new(InMemoryTransport::new(
        &["schedule", "release", "gpu_a", "gpu_b"],
        "",
    ));

    let scheduler = build_engine_for(
        &config,
        "scheduler",
        transport.clone(),
        RoleProviders::empty().with_queue_manager(qm),
    );

    transport
        .inject(
            "schedule",
            "run-1",
            r#"{"workflow":"wf","workflow_id":"wf","resource_profile":{"gpus_required":1,"memory_mb":20000,"executor_class":"python.gpu.demo","device_kind":"gpu","tags":["demo"]}}"#,
            "schedule",
        )
        .unwrap();
    transport
        .inject(
            "schedule",
            "run-2",
            r#"{"workflow":"wf","workflow_id":"wf","resource_profile":{"gpus_required":1,"memory_mb":20000,"executor_class":"python.gpu.demo","device_kind":"gpu","tags":["demo"]}}"#,
            "schedule",
        )
        .unwrap();

    let stats = scheduler.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 0);
    let stats = scheduler.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 0);
    let stats = scheduler.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 0);

    let on_a = transport.pending_in("gpu_a").len();
    let on_b = transport.pending_in("gpu_b").len();
    assert_eq!(on_a, 1, "first job should be routed to gpu_a");
    assert_eq!(on_b, 1, "second job should be routed to gpu_b");

    set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
}

#[tokio::test]
async fn terminal_consumer_works_with_empty_outputs() {
    let config: RuntimeConfig = serde_json::from_value(serde_json::json!({
        "stream_prefix": "",
        "streams": ["results"],
        "roles": {
            "results": {
                "inputs": [{"stream": "results", "max_dequeue_items": 8,
                            "poll_interval_ms": 10, "block_ms": 50}],
                "outputs": []
            }
        }
    }))
    .unwrap();

    let transport = Arc::new(InMemoryTransport::new(&["results"], ""));
    let results = build_engine_for(
        &config,
        "results",
        transport.clone(),
        RoleProviders {
            results_persistence: Some(Arc::new(NoopResultsPersistence::new())),
            ..RoleProviders::empty()
        },
    );

    transport
        .inject(
            "results",
            "run-1",
            r#"{"status":"succeeded","payload":"{\"output_path\":\"/tmp/out\"}"}"#,
            "results",
        )
        .unwrap();

    let stats = results.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 1);
    assert!(transport.all_drained());
}

/// Load the real `runtime_config.json`, build the prefetch role through
/// `build_role`, inject a message, and verify it processes end-to-end
/// through the engine with recording materializer — no real HTTP calls.
#[tokio::test]
async fn prefetch_from_real_config_processes_message_through_engine() {
    let config: RuntimeConfig =
        serde_json::from_str(EXAMPLE_CONFIG).expect("example config should parse");
    config.validate().expect("example config should validate");

    let stream_refs: Vec<&str> = config.streams.iter().map(|s| s.as_str()).collect();
    let transport = Arc::new(InMemoryTransport::new(&stream_refs, &config.stream_prefix));

    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 5,
            cached: 2,
            ..Default::default()
        },
        artifacts: vec![MaterializedArtifact {
            name: "prepared-input".to_string(),
            source_uri: "s3://bucket/path/input.bin".to_string(),
            storage_path: "/tmp/cache/prepared-input.bin".to_string(),
            size_bytes: 512,
            sha256: None,
            media_type: Some("application/octet-stream".to_string()),
            downloaded: true,
        }],
    }));

    let prefetch = build_engine_for(
        &config,
        "prefetch",
        transport.clone(),
        RoleProviders {
            materializer: Some(mat.clone()),
            ..RoleProviders::empty()
        },
    );

    transport
        .inject(
            "prefetch",
            "run-42",
            &plugin_prefetch_payload(
                "run-42",
                json!([
                    {
                        "source_uri": "s3://bucket/path/input.bin",
                        "target_artifact_name": "prepared-input",
                        "media_type": "application/octet-stream"
                    },
                    {
                        "source_uri": "https://example.com/reference.txt",
                        "target_artifact_name": "reference",
                        "required": false
                    }
                ]),
            ),
            "prefetch",
        )
        .unwrap();

    let stats = prefetch.run_once().await.unwrap();
    assert_eq!(stats.succeeded, 1, "prefetch should process one message");

    assert_eq!(mat.call_count(), 1, "materializer should be invoked once");

    let output_msgs = transport.pending_in("schedule");
    assert_eq!(output_msgs.len(), 1, "should hand off to schedule stream");

    let out_payload: serde_json::Value = serde_json::from_str(output_msgs[0].payload()).unwrap();
    assert_eq!(out_payload["workflow_id"], "demo-plugin");
    assert_eq!(out_payload["prefetch_downloaded"], 5);
    assert_eq!(out_payload["prefetch_cached"], 2);
    assert_eq!(out_payload["prefetch_plan_count"], 2);
    assert!(out_payload["prefetch_plan"].is_array());
    assert_eq!(
        out_payload["prefetch_artifacts"][0]["storage_path"],
        "/tmp/cache/prepared-input.bin"
    );
    assert_eq!(out_payload["stage_context"]["current_phase"], "schedule");
}
