/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use serde_json::{Value as JsonValue, json};

use super::PrefetchRole;
use super::download::{DownloadStats, MaterializationResult, MaterializedArtifact};
use super::materializer::PlanMaterializer;
use super::plan::PrefetchPlanItem;
use crate::config::InputStreamSpec;
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

struct RecordingSink {
    handoffs: Mutex<Vec<(String, String, String)>>,
}

impl RecordingSink {
    fn new() -> Self {
        Self {
            handoffs: Mutex::new(Vec::new()),
        }
    }

    fn handoffs(&self) -> Vec<(String, String, String)> {
        self.handoffs.lock().expect("handoff lock poisoned").clone()
    }
}

impl MessageSink for RecordingSink {
    fn enqueue<'a>(
        &'a self,
        _: &'a str,
        _: &'a str,
        _: &'a str,
        _: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async { Ok("1-0".into()) })
    }

    fn ack_message<'a>(&'a self, _: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn handoff<'a>(
        &'a self,
        _msg: &'a scicomp_rq::Message,
        dest: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            self.handoffs.lock().expect("handoff lock poisoned").push((
                dest.to_string(),
                payload.to_string(),
                stage.to_string(),
            ));
            Ok("1-0".into())
        })
    }

    fn forward_many<'a>(
        &'a self,
        _: &'a scicomp_rq::Message,
        _: &'a [scicomp_rq::Output],
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        Box::pin(async { Ok(vec![]) })
    }
}

struct RecordingMaterializer {
    calls: Mutex<Vec<(Vec<PrefetchPlanItem>, String)>>,
    result: MaterializationResult,
}

impl RecordingMaterializer {
    fn with_result(result: MaterializationResult) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            result,
        }
    }

    fn noop() -> Self {
        Self::with_result(MaterializationResult::default())
    }

    fn call_count(&self) -> usize {
        self.calls.lock().expect("calls lock poisoned").len()
    }

    fn last_plan(&self) -> Vec<PrefetchPlanItem> {
        self.calls
            .lock()
            .expect("calls lock poisoned")
            .last()
            .map(|entry| entry.0.clone())
            .unwrap_or_default()
    }
}

impl PlanMaterializer for RecordingMaterializer {
    fn materialize<'a>(
        &'a self,
        plan: &'a [PrefetchPlanItem],
        _cache_root: &'a Path,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<MaterializationResult>> {
        let captured_plan = plan.to_vec();
        let run = run_id.to_string();
        self.calls
            .lock()
            .expect("calls lock poisoned")
            .push((captured_plan, run));
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

struct FailingMaterializer;

impl PlanMaterializer for FailingMaterializer {
    fn materialize<'a>(
        &'a self,
        _: &'a [PrefetchPlanItem],
        _: &'a Path,
        _: &'a str,
    ) -> BoxFuture<'a, Result<MaterializationResult>> {
        Box::pin(async { Err(anyhow!("download failed: connection refused")) })
    }
}

fn msg(run_id: &str, payload: &str) -> scicomp_rq::Message {
    scicomp_rq::Message::new(
        "1-0",
        "test:prefetch",
        "prefetch:grp",
        run_id,
        payload,
        "prefetch",
    )
}

fn plugin_payload(prefetch_plan: JsonValue) -> String {
    json!({
        "run_id": "run-1",
        "workflow_id": "demo-plugin",
        "operation": "both",
        "parameters": {
            "batch_size": 128000,
        },
        "request": {
            "content_type": "multipart/form-data",
            "raw_fields": {
                "batch_size": "128000",
            },
            "input_artifacts": [
                {
                    "field_name": "design_stl",
                    "name": "design.stl",
                    "artifact_id": "artifact-1",
                    "media_type": "model/stl",
                    "size_bytes": 64,
                    "storage_path": "/tmp/uploads/design.stl",
                    "original_filename": "design.stl",
                }
            ],
        },
        "resource_profile": {
            "executor_class": "python.gpu.physicsnemo",
            "gpus_required": 1,
            "memory_mb": 4096,
        },
        "prefetch_plan": prefetch_plan,
        "stage_context": {
            "current_stage_id": "prefetch",
            "current_phase": "prefetch",
            "pipeline": [
                {
                    "id": "prefetch",
                    "phase": "prefetch",
                    "queue": "prefetch",
                    "next": "schedule",
                },
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "queue": "schedule",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "queue": "results",
                    "next": null,
                }
            ],
        },
        "runtime": {
            "kind": "python",
            "entrypoint": "plugin.py",
            "executor_class": "python.gpu.physicsnemo",
        },
    })
    .to_string()
}

fn role_with(materializer: Arc<dyn PlanMaterializer>) -> PrefetchRole {
    PrefetchRole::new_for_test("schedule", materializer)
}

fn env_with_fail_closed(fail_on_prefetch_error: bool) -> RoleEnv {
    RoleEnv {
        role_name: "prefetch".to_string(),
        stream_prefix: "test:".to_string(),
        inputs: vec![InputStreamSpec {
            stream: "prefetch".to_string(),
            max_dequeue_items: 1,
            poll_interval_ms: 10,
            block_ms: 50,
            reclaim_idle_ms: 60_000,
        }],
        resolved_outputs: vec!["schedule".to_string()],
        role_config: Some(json!({
            "handoff_stream": "schedule",
            "fail_on_plan_generation_error": fail_on_prefetch_error,
        })),
        python_runtime_envs: Default::default(),
    }
}

fn artifact(name: &str, storage_path: &str, size_bytes: u64) -> MaterializedArtifact {
    MaterializedArtifact {
        name: name.to_string(),
        source_uri: "s3://bucket/path/input.bin".to_string(),
        storage_path: storage_path.to_string(),
        size_bytes,
        media_type: Some("application/octet-stream".to_string()),
        sha256: None,
    }
}

#[tokio::test]
async fn rejects_payload_with_non_array_prefetch_plan() {
    let mat = Arc::new(RecordingMaterializer::noop());
    let role = role_with(mat.clone());
    let sink = RecordingSink::new();

    let payload = plugin_payload(json!({"source_uri": "s3://bucket/file"}));
    let err = role
        .handle(&msg("run-1", &payload), "prefetch", &sink)
        .await
        .unwrap_err();

    assert!(err.to_string().contains("prefetch_plan"));
    assert_eq!(mat.call_count(), 0);
}

#[tokio::test]
async fn missing_prefetch_plan_defaults_to_empty_and_advances_stage_context() {
    let mat = Arc::new(RecordingMaterializer::noop());
    let role = role_with(mat.clone());
    let sink = RecordingSink::new();

    let mut payload: JsonValue = serde_json::from_str(&plugin_payload(json!([]))).unwrap();
    payload
        .as_object_mut()
        .expect("payload should be object")
        .remove("prefetch_plan");

    role.handle(
        &msg("run-1", &serde_json::to_string(&payload).unwrap()),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    assert_eq!(mat.call_count(), 1);
    assert!(mat.last_plan().is_empty());

    let calls = sink.handoffs();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "schedule");
    assert_eq!(calls[0].2, "schedule");

    let out: JsonValue = serde_json::from_str(&calls[0].1).unwrap();
    assert_eq!(out["prefetch_plan_count"], 0);
    assert_eq!(out["prefetch_downloaded"], 0);
    assert_eq!(out["prefetch_artifacts"], json!([]));
    assert_eq!(out["stage_context"]["current_stage_id"], "schedule");
    assert_eq!(out["stage_context"]["current_phase"], "schedule");
    assert_eq!(
        out["request"]["input_artifacts"][0]["field_name"],
        "design_stl"
    );
}

#[tokio::test]
async fn happy_path_materializes_explicit_plan_and_preserves_payload_fields() {
    let result = MaterializationResult {
        stats: DownloadStats {
            downloaded: 1,
            cached: 1,
            errors: 0,
            required_errors: 0,
            required_verified_errors: 0,
            optional_errors: 0,
            total_time_secs: 0.25,
            throughput_mbps: 42.0,
            total_mb: 10.0,
        },
        artifacts: vec![artifact(
            "prepared-input",
            "/tmp/cache/prepared-input.bin",
            512,
        )],
    };
    let mat = Arc::new(RecordingMaterializer::with_result(result));
    let role = role_with(mat.clone());
    let sink = RecordingSink::new();

    let mut payload: JsonValue = serde_json::from_str(&plugin_payload(json!([
        {
            "source_uri": "s3://bucket/path/input.bin",
            "target_artifact_name": "prepared-input",
            "media_type": "application/octet-stream",
            "byte_range": {
                "offset": 0,
                "length": 512,
            },
            "headers": {
                "x-test-header": "123"
            }
        },
        {
            "source_uri": "https://example.com/reference.txt",
            "target_artifact_name": "reference",
            "required": false
        }
    ])))
    .unwrap();
    payload["custom_field"] = json!("preserved");

    role.handle(
        &msg("run-1", &serde_json::to_string(&payload).unwrap()),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    assert_eq!(mat.call_count(), 1);
    let last_plan = mat.last_plan();
    assert_eq!(last_plan.len(), 2);
    assert_eq!(last_plan[0].target_artifact_name, "prepared-input");
    assert_eq!(
        last_plan[0].headers.get("x-test-header"),
        Some(&"123".to_string())
    );
    assert_eq!(
        last_plan[0].byte_range.as_ref().expect("byte_range").length,
        512
    );
    assert!(!last_plan[1].required);

    let calls = sink.handoffs();
    let out: JsonValue = serde_json::from_str(&calls[0].1).unwrap();
    assert_eq!(out["prefetch_downloaded"], 1);
    assert_eq!(out["prefetch_cached"], 1);
    assert_eq!(out["prefetch_errors"], 0);
    assert_eq!(out["prefetch_plan_count"], 2);
    assert_eq!(out["prefetch_artifacts"][0]["name"], "prepared-input");
    assert_eq!(
        out["prefetch_artifacts"][0]["storage_path"],
        "/tmp/cache/prepared-input.bin"
    );
    assert_eq!(out["custom_field"], "preserved");
    assert_eq!(out["stage_context"]["current_stage_id"], "schedule");
}

#[tokio::test]
async fn materializer_failure_propagates_as_error() {
    let role = role_with(Arc::new(FailingMaterializer));
    let sink = RecordingSink::new();

    let err = role
        .handle(
            &msg(
                "run-1",
                &plugin_payload(json!([{
                    "source_uri": "s3://bucket/path/input.bin",
                    "target_artifact_name": "prepared-input"
                }])),
            ),
            "prefetch",
            &sink,
        )
        .await
        .unwrap_err();

    let error_text = format!("{err:#}");
    assert!(error_text.contains("materializer failed"));
    assert!(error_text.contains("connection refused"));
    assert!(sink.handoffs().is_empty());
}

#[tokio::test]
async fn fail_open_mode_handoffs_with_degraded_marker_for_optional_failures() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 0,
            required_verified_errors: 0,
            optional_errors: 1,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat.clone()))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    role.handle(
        &msg(
            "run-1",
            &plugin_payload(json!([{
                "source_uri": "https://example.com/reference.txt",
                "target_artifact_name": "reference",
                "required": false
            }])),
        ),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    let out: JsonValue = serde_json::from_str(&sink.handoffs()[0].1).unwrap();
    assert_eq!(out["prefetch_degraded"], true);
    assert_eq!(out["prefetch_optional_errors"], 1);
    assert_eq!(out["prefetch_required_errors"], 0);
}

#[tokio::test]
async fn fail_open_mode_returns_error_on_required_checksum_verified_failure() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 1,
            required_verified_errors: 1,
            optional_errors: 0,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    let err = role
        .handle(
            &msg(
                "run-1",
                &plugin_payload(json!([{
                    "source_uri": "https://assets.example.com/mesh.vtp",
                    "target_artifact_name": "mesh",
                    "expected_sha256": "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
                }])),
            ),
            "prefetch",
            &sink,
        )
        .await
        .unwrap_err();

    assert!(err.to_string().contains("required download failures"));
    assert!(sink.handoffs().is_empty());
}

#[tokio::test]
async fn fail_open_mode_returns_error_on_required_size_verified_failure() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 1,
            required_verified_errors: 1,
            optional_errors: 0,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    let err = role
        .handle(
            &msg(
                "run-1",
                &plugin_payload(json!([{
                    "source_uri": "https://assets.example.com/mesh.vtp",
                    "target_artifact_name": "mesh",
                    "expected_size_bytes": 512
                }])),
            ),
            "prefetch",
            &sink,
        )
        .await
        .unwrap_err();

    assert!(err.to_string().contains("required download failures"));
    assert!(sink.handoffs().is_empty());
}

#[tokio::test]
async fn fail_open_mode_preserves_legacy_required_failure_handoff() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 1,
            required_verified_errors: 0,
            optional_errors: 0,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    role.handle(
        &msg(
            "run-1",
            &plugin_payload(json!([{
                "source_uri": "s3://bucket/path/input.bin",
                "target_artifact_name": "prepared-input"
            }])),
        ),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    let handoffs = sink.handoffs();
    assert_eq!(handoffs.len(), 1);
    let out: JsonValue = serde_json::from_str(&handoffs[0].1).unwrap();
    assert_eq!(out["prefetch_degraded"], true);
    assert_eq!(out["prefetch_required_errors"], 1);
}

#[tokio::test]
async fn fail_open_mode_preserves_legacy_failure_in_mixed_verified_plan() {
    let expected_sha256 = "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd";
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 1,
            cached: 0,
            errors: 1,
            required_errors: 1,
            required_verified_errors: 0,
            optional_errors: 0,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![MaterializedArtifact {
            name: "mesh".to_string(),
            source_uri: "https://assets.example.com/mesh.vtp".to_string(),
            storage_path: "/tmp/cache/mesh.vtp".to_string(),
            size_bytes: 4,
            media_type: None,
            sha256: Some(expected_sha256.to_string()),
        }],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    role.handle(
        &msg(
            "run-1",
            &plugin_payload(json!([
                {
                    "source_uri": "https://assets.example.com/mesh.vtp",
                    "target_artifact_name": "mesh",
                    "expected_sha256": expected_sha256,
                    "expected_size_bytes": 4
                },
                {
                    "source_uri": "s3://bucket/path/input.bin",
                    "target_artifact_name": "legacy-input"
                }
            ])),
        ),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    let handoffs = sink.handoffs();
    assert_eq!(handoffs.len(), 1);
    let out: JsonValue = serde_json::from_str(&handoffs[0].1).unwrap();
    assert_eq!(out["prefetch_degraded"], true);
    assert_eq!(out["prefetch_required_errors"], 1);
    assert_eq!(out["prefetch_artifacts"][0]["name"], "mesh");
}

#[tokio::test]
async fn fail_open_mode_preserves_optional_verified_failure_handoff() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 0,
            required_verified_errors: 0,
            optional_errors: 1,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(false), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    role.handle(
        &msg(
            "run-1",
            &plugin_payload(json!([{
                "source_uri": "https://assets.example.com/optional.bin",
                "target_artifact_name": "optional",
                "required": false,
                "expected_sha256": "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                "expected_size_bytes": 512
            }])),
        ),
        "prefetch",
        &sink,
    )
    .await
    .unwrap();

    let handoffs = sink.handoffs();
    assert_eq!(handoffs.len(), 1);
    let out: JsonValue = serde_json::from_str(&handoffs[0].1).unwrap();
    assert_eq!(out["prefetch_degraded"], true);
    assert_eq!(out["prefetch_optional_errors"], 1);
    assert_eq!(out["prefetch_required_errors"], 0);
}

#[tokio::test]
async fn fail_closed_mode_returns_error_on_required_download_failures() {
    let mat = Arc::new(RecordingMaterializer::with_result(MaterializationResult {
        stats: DownloadStats {
            downloaded: 0,
            cached: 0,
            errors: 1,
            required_errors: 1,
            required_verified_errors: 0,
            optional_errors: 0,
            total_time_secs: 0.1,
            throughput_mbps: 0.0,
            total_mb: 0.0,
        },
        artifacts: vec![],
    }));
    let role = PrefetchRole::from_env(&env_with_fail_closed(true), Some(mat))
        .expect("prefetch role should build");
    let sink = RecordingSink::new();

    let err = role
        .handle(
            &msg(
                "run-1",
                &plugin_payload(json!([{
                    "source_uri": "s3://bucket/path/input.bin",
                    "target_artifact_name": "prepared-input"
                }])),
            ),
            "prefetch",
            &sink,
        )
        .await
        .unwrap_err();

    assert!(err.to_string().contains("required download failures"));
    assert!(sink.handoffs().is_empty());
}

#[tokio::test]
async fn rejects_unexpected_stream() {
    let role = role_with(Arc::new(RecordingMaterializer::noop()));
    let sink = RecordingSink::new();

    let result = role
        .handle(
            &msg("run-1", &plugin_payload(json!([]))),
            "not-configured",
            &sink,
        )
        .await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unexpected stream")
    );
}
