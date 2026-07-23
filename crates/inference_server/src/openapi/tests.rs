/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::config::ServerConfig;
use crate::plugin_registry::{
    PluginManifest, PythonModuleProbe, PythonRuntimeEnvConfig, RegisteredPlugin,
    plugin_manifest_path,
};
use crate::redis_ops::RedisService;
use crate::state::{AppState, CachedWorkflowContract};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn create_mock_state() -> Arc<AppState> {
    create_mock_state_with_max_body_size(2 * 1024 * 1024)
}

fn create_mock_state_with_max_body_size(max_body_size: usize) -> Arc<AppState> {
    create_mock_state_with_options(max_body_size, None)
}

fn create_mock_state_with_enabled_plugin_id(enabled_plugin_id: &str) -> Arc<AppState> {
    create_mock_state_with_options(2 * 1024 * 1024, Some(enabled_plugin_id))
}

fn create_mock_state_with_python_runtime(
    executor_class: &str,
    python_executable: String,
) -> Arc<AppState> {
    let state = create_mock_state();
    let mut runtime_envs = std::collections::HashMap::new();
    runtime_envs.insert(
        executor_class.to_string(),
        PythonRuntimeEnvConfig {
            python_executable,
            env: std::collections::HashMap::new(),
        },
    );

    let config = ServerConfig {
        python_runtime_envs: runtime_envs,
        ..(*state.config).clone()
    };
    Arc::new(AppState::new_for_testing(config))
}

fn create_mock_state_with_options(
    max_body_size: usize,
    enabled_plugin_id: Option<&str>,
) -> Arc<AppState> {
    let config = ServerConfig {
        addr: "127.0.0.1:8080".parse().unwrap(),
        redis_url: "redis://127.0.0.1:6379".to_string(),
        redis_stream: "inference".to_string(),
        prefetch_stream: "prefetch".to_string(),
        use_prefetch: true,
        plugin_dirs: vec![],
        enabled_plugin_id: enabled_plugin_id.map(str::to_string),
        artifact_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-openapi-artifacts-{}",
            uuid::Uuid::new_v4()
        )),
        default_output_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-openapi-outputs-{}",
            uuid::Uuid::new_v4()
        )),
        artifact_retention_hours: 24,
        artifact_cleanup_interval_secs: 30,
        cors_allowed_origins: vec![],
        max_body_size,
        stream_prefix: String::new(),
        swagger_cdn_url: None,
        python_runtime_envs: std::collections::HashMap::new(),
        output_publication: Default::default(),
    };

    Arc::new(AppState::new_for_testing(config))
}

struct TestRedisServer {
    child: Child,
    data_dir: PathBuf,
}

impl TestRedisServer {
    async fn spawn(test_name: &str, port: u16) -> Self {
        let data_dir = std::env::temp_dir().join(format!(
            "physicsnemo-serve-openapi-redis-{test_name}-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&data_dir).expect("redis data dir should be created");
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
            .arg(&data_dir)
            .arg("--loglevel")
            .arg("warning")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("redis-server should start");

        wait_for_tcp_listener(port).await;

        Self { child, data_dir }
    }
}

impl Drop for TestRedisServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.data_dir);
    }
}

fn reserve_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("ephemeral port should be allocated")
        .local_addr()
        .expect("listener should have a local addr")
        .port()
}

async fn wait_for_tcp_listener(port: u16) {
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("redis-server on port {port} did not become ready in time");
}

async fn wait_for_tcp_listener_to_close(port: u16) {
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_err()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("redis-server on port {port} did not shut down in time");
}

async fn create_mock_state_with_redis(
    test_name: &str,
    max_body_size: usize,
) -> (Arc<AppState>, TestRedisServer) {
    let port = reserve_port();
    let redis_server = TestRedisServer::spawn(test_name, port).await;
    let state =
        create_mock_state_for_redis_url(format!("redis://127.0.0.1:{port}/0"), max_body_size).await;
    (state, redis_server)
}

async fn create_mock_state_for_redis_url(redis_url: String, max_body_size: usize) -> Arc<AppState> {
    let config = ServerConfig {
        addr: "127.0.0.1:8080".parse().unwrap(),
        redis_url,
        redis_stream: "inference".to_string(),
        prefetch_stream: "prefetch".to_string(),
        use_prefetch: true,
        plugin_dirs: vec![],
        enabled_plugin_id: None,
        artifact_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-openapi-artifacts-{}",
            uuid::Uuid::new_v4()
        )),
        default_output_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-openapi-outputs-{}",
            uuid::Uuid::new_v4()
        )),
        artifact_retention_hours: 24,
        artifact_cleanup_interval_secs: 30,
        cors_allowed_origins: vec![],
        max_body_size,
        stream_prefix: String::new(),
        swagger_cdn_url: None,
        python_runtime_envs: std::collections::HashMap::new(),
        output_publication: Default::default(),
    };
    let redis_service = RedisService::connect(&config)
        .await
        .expect("test Redis service should connect");
    Arc::new(AppState::new(config, redis_service))
}

fn plugin_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: demo-prefetch
  display_name: Demo Prefetch
  version: 1.0.0
  description: Generic prefetch plugin example
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.test
resources:
  defaults:
    gpus_required: 1
    memory_mb: 24000
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: forecast_dataset
    media_type: application/x-zarr
  retention_hours: 24
"#
}

fn multipart_plugin_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: multipart-demo
  display_name: Multipart Demo
  version: 1.0.0
  description: Multipart inference with uploaded geometry
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: both
    allowed: [both, volume]
  form_schema: schemas/request.multipart.json
  files:
    - name: design_stl
      required: true
      media_types: [model/stl, application/sla]
      max_size_mb: 4
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.gpu.demo
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.gpu.demo
resources:
  defaults:
    gpus_required: 1
    memory_mb: 24000
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: pressure_field
    media_type: application/x-npz
  retention_hours: 24
"#
}

fn readiness_plugin_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: readiness-demo
  display_name: Readiness Demo
  version: 1.0.0
  description: Plugin manifest used for readiness reporting tests
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.test
resources:
  defaults:
    gpus_required: 0
    memory_mb: 1024
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    recommended_check_phase: prepare
    python_modules: [json]
    env:
      - name: READINESS_DEMO_ROOT
        kind: dir
"#
}

fn inline_schema_plugin_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: inline-schema-demo
  display_name: Inline Schema Demo
  version: 1.0.0
  description: Plugin manifest using inline request and result schemas
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
    required: [value]
    properties:
      value:
        type: integer
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    gpus_required: 0
    memory_mb: 1024
outputs:
  result_schema_inline:
    type: object
    properties:
      status:
        type: string
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#
}

fn model_driven_plugin_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: model-driven-demo
  display_name: Model Driven Demo
  version: 1.0.0
  description: Plugin manifest deriving schemas from workflow models
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    gpus_required: 0
    memory_mb: 1024
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#
}

fn create_plugin_root(test_name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-openapi-plugin-{test_name}-{}",
        uuid::Uuid::new_v4()
    ));
    let schemas_dir = root.join("schemas");
    fs::create_dir_all(&schemas_dir).expect("schemas directory should be created");
    fs::write(root.join("plugin.yaml"), plugin_manifest_yaml())
        .expect("plugin manifest should be written");
    fs::write(
        schemas_dir.join("request.json"),
        r#"{
                "type":"object",
                "required":["start_time"],
                "properties":{
                    "start_time":{"type":"array","items":{"type":"string","format":"date-time"}},
                    "num_steps":{"type":"integer","minimum":1}
                },
                "additionalProperties":false
            }"#,
    )
    .expect("request schema should be written");
    fs::write(
        schemas_dir.join("result.json"),
        r#"{
                "type":"object",
                "properties":{"status":{"type":"string"}}
            }"#,
    )
    .expect("result schema should be written");
    root
}

fn create_multipart_plugin_root(test_name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-openapi-multipart-plugin-{test_name}-{}",
        uuid::Uuid::new_v4()
    ));
    let schemas_dir = root.join("schemas");
    fs::create_dir_all(&schemas_dir).expect("schemas directory should be created");
    fs::write(root.join("plugin.yaml"), multipart_plugin_manifest_yaml())
        .expect("plugin manifest should be written");
    fs::write(
        schemas_dir.join("request.multipart.json"),
        r#"{
                "type":"object",
                "required":["batch_size"],
                "properties":{
                    "batch_size":{"type":"integer","minimum":1}
                },
                "additionalProperties":false
            }"#,
    )
    .expect("multipart request schema should be written");
    fs::write(
        schemas_dir.join("result.json"),
        r#"{
                "type":"object",
                "properties":{"status":{"type":"string"}}
            }"#,
    )
    .expect("result schema should be written");
    root
}

fn create_plugin_root_with_manifest(test_name: &str, manifest_yaml: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-openapi-plugin-custom-{test_name}-{}",
        uuid::Uuid::new_v4()
    ));
    let schemas_dir = root.join("schemas");
    fs::create_dir_all(&schemas_dir).expect("schemas directory should be created");
    fs::write(root.join("plugin.yaml"), manifest_yaml).expect("plugin manifest should be written");
    fs::write(
        schemas_dir.join("request.json"),
        r#"{
                "type":"object",
                "required":["start_time"],
                "properties":{
                    "start_time":{"type":"array","items":{"type":"string","format":"date-time"}}
                },
                "additionalProperties":false
            }"#,
    )
    .expect("request schema should be written");
    fs::write(
        schemas_dir.join("result.json"),
        r#"{
                "type":"object",
                "properties":{"status":{"type":"string"}}
            }"#,
    )
    .expect("result schema should be written");
    root
}

fn create_model_driven_plugin_root(test_name: &str) -> PathBuf {
    let root = create_plugin_root_with_manifest(test_name, model_driven_plugin_manifest_yaml());
    fs::write(
        root.join("workflow.py"),
        r#"
from __future__ import annotations

from dataclasses import dataclass

from plugin_sdk import PluginWorkflow


@dataclass
class DemoInput:
    value: int


@dataclass
class DemoOutput:
    value: int
    doubled: int


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput
    output_model = DemoOutput

    def run(self, inputs: DemoInput, ctx):
        return DemoOutput(value=inputs.value, doubled=inputs.value * 2)


WORKFLOW = DemoWorkflow()
"#,
    )
    .expect("workflow should be written");
    root
}

fn create_counting_python_probe(test_name: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-openapi-python-probe-{test_name}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&root).expect("probe directory should be created");
    let executable = root.join("python-probe");
    let counter = root.join("probe-count.txt");
    fs::write(&counter, "0\n").expect("probe counter should be initialized");
    fs::write(
        &executable,
        format!(
            r#"#!/bin/sh
count_file="{}"
count=$(cat "$count_file")
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
printf '%s\n' 'PHYSICSNEMO_SERVE_JSON:{{"found":true}}'
"#,
            counter.display()
        ),
    )
    .expect("probe executable should be written");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(&executable)
            .expect("probe executable metadata should be readable")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions)
            .expect("probe executable should be executable");
    }

    (executable, counter)
}

async fn register_plugin(state: &Arc<AppState>, test_name: &str) -> RegisteredPlugin {
    let root = create_plugin_root(test_name);
    let manifest =
        PluginManifest::from_yaml_str(plugin_manifest_yaml()).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };
    register_plugin_contract(state, &plugin).await;
    plugin
}

async fn register_multipart_plugin(state: &Arc<AppState>, test_name: &str) -> RegisteredPlugin {
    let root = create_multipart_plugin_root(test_name);
    let manifest = PluginManifest::from_yaml_str(multipart_plugin_manifest_yaml())
        .expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };
    register_plugin_contract(state, &plugin).await;
    plugin
}

async fn register_custom_plugin(
    state: &Arc<AppState>,
    test_name: &str,
    manifest_yaml: &str,
) -> RegisteredPlugin {
    let root = create_plugin_root_with_manifest(test_name, manifest_yaml);
    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };
    register_plugin_contract(state, &plugin).await;
    plugin
}

async fn register_model_driven_plugin(state: &Arc<AppState>, test_name: &str) -> RegisteredPlugin {
    let root = create_model_driven_plugin_root(test_name);
    let manifest = PluginManifest::from_yaml_str(model_driven_plugin_manifest_yaml())
        .expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };
    register_plugin_contract(state, &plugin).await;
    plugin
}

async fn register_plugin_contract(state: &Arc<AppState>, plugin: &RegisteredPlugin) {
    let mut readiness_probe =
        PythonModuleProbe::with_runtime_envs(state.config.python_runtime_envs.clone());
    let contract = CachedWorkflowContract::build(plugin.clone(), &mut readiness_probe);
    let plugin_id = plugin.manifest.metadata.id.clone();

    let mut registry = state.workflow_registry.write().await;
    registry.contracts.insert(plugin_id, contract);
}

#[allow(clippy::type_complexity)]
fn multipart_body(boundary: &str, parts: &[(&str, Option<(&str, &str)>, &str)]) -> String {
    let mut body = String::new();
    for (name, file_meta, value) in parts {
        body.push_str(&format!("--{boundary}\r\n"));
        match file_meta {
            Some((filename, media_type)) => {
                body.push_str(&format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                ));
                body.push_str(&format!("Content-Type: {media_type}\r\n\r\n"));
                body.push_str(value);
                body.push_str("\r\n");
            }
            None => {
                body.push_str(&format!(
                    "Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                ));
                body.push_str(value);
                body.push_str("\r\n");
            }
        }
    }
    body.push_str(&format!("--{boundary}--\r\n"));
    body
}

#[test]
fn test_build_openapi_json() {
    let json = build_openapi_json();

    assert_eq!(json["openapi"], "3.0.3");
    assert_eq!(json["info"]["title"], "PhysicsNeMo Serve Inference API");
    assert_eq!(json["info"]["version"], "1.0.0");

    let paths = json["paths"]
        .as_object()
        .expect("paths should be an object");
    assert!(paths.contains_key("/healthz"));
    assert!(paths.contains_key("/doc"));
    assert!(paths.contains_key("/v1/infer/workflows"));
    assert!(paths.contains_key("/v1/infer/{name}/schema"));
    assert!(paths.contains_key("/v1/infer/{name}/readiness"));
    assert!(paths.contains_key("/v1/infer/{workflow}/{run_id}/status"));
    assert!(paths.contains_key("/v1/infer/{workflow}/{run_id}/results"));
    assert!(paths.contains_key("/openapi.json"));
    assert!(paths.contains_key("/openapi"));

    let components = json["components"]
        .as_object()
        .expect("components should be an object");
    assert!(components.contains_key("schemas"));
}

#[test]
fn test_openapi_includes_readyz_endpoint() {
    let json = build_openapi_json();
    let paths = json["paths"].as_object().expect("paths should be object");
    assert!(
        paths.contains_key("/readyz"),
        "OpenAPI must include /readyz (PR-092)"
    );

    let readyz = &json["paths"]["/readyz"]["get"];
    assert!(readyz.is_object(), "/readyz should have a GET operation");

    let responses = readyz["responses"]
        .as_object()
        .expect("readyz should have responses");
    assert!(responses.contains_key("200"), "readyz needs 200 response");
    assert!(responses.contains_key("503"), "readyz needs 503 response");
}

#[test]
fn test_openapi_includes_run_workflow_endpoint() {
    let json = build_openapi_json();
    let paths = json["paths"].as_object().expect("paths should be object");
    assert!(
        paths.contains_key("/v1/infer/{name}/run"),
        "OpenAPI must include /v1/infer/{{name}}/run (PR-092)"
    );

    let run = &json["paths"]["/v1/infer/{name}/run"]["post"];
    assert!(
        run.is_object(),
        "/v1/infer/{{name}}/run should have a POST operation"
    );

    let responses = run["responses"]
        .as_object()
        .expect("run endpoint should have responses");
    assert!(responses.contains_key("202"), "run needs 202 response");
    assert!(responses.contains_key("404"), "run needs 404 response");
    assert!(responses.contains_key("422"), "run needs 422 response");
    assert!(responses.contains_key("503"), "run needs 503 response");
}

#[test]
fn test_openapi_uses_workflow_namespaced_run_routes() {
    let json = build_openapi_json();
    let paths = json["paths"].as_object().expect("paths should be object");

    assert!(paths.contains_key("/v1/infer/{workflow}/{run_id}/status"));
    assert!(paths.contains_key("/v1/infer/{workflow}/{run_id}/results"));
    assert!(
        !paths.contains_key("/v1/infer/{run_id}/status"),
        "flat run status route should no longer be the documented API"
    );
    assert!(
        !paths.contains_key("/v1/infer/{run_id}/results"),
        "flat run results route should no longer be the documented API"
    );
}

#[tokio::test]
async fn test_openapi_json_endpoint() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["openapi"], "3.0.3");
    assert_eq!(json["info"]["title"], "PhysicsNeMo Serve Inference API");
    assert!(json["servers"].is_array());
}

#[tokio::test]
async fn test_openapi_endpoint_alternate() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/openapi")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["openapi"], "3.0.3");
}

#[tokio::test]
async fn test_healthz_endpoint() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], b"ok");
}

#[tokio::test]
async fn test_list_workflows_empty() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/workflows")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Should be empty initially
    assert_eq!(json["count"], 0);
    assert!(json["workflows"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_list_workflows_returns_registered_plugins() {
    let state = create_mock_state();
    let plugin = register_plugin(&state, "list").await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/workflows")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let workflows = json["workflows"]
        .as_array()
        .expect("workflows should be an array");

    assert_eq!(json["count"], 1);
    assert_eq!(workflows[0]["name"], plugin.manifest.metadata.id);
    assert_eq!(
        workflows[0]["executor_class"],
        plugin.manifest.runtime.executor_class
    );
    assert_eq!(
        workflows[0]["default_operation"],
        plugin.manifest.ingress.operations.default
    );
    assert_eq!(workflows[0]["plugin"], true);
    assert_eq!(workflows[0]["readiness"]["ready"], true);
}

#[tokio::test]
async fn test_list_workflows_includes_plugin_readiness_status() {
    let state = create_mock_state();
    let plugin =
        register_custom_plugin(&state, "list-readiness", readiness_plugin_manifest_yaml()).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/workflows")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let workflows = json["workflows"]
        .as_array()
        .expect("workflows should be an array");
    let workflow = workflows
        .iter()
        .find(|item| item["name"] == plugin.manifest.metadata.id)
        .expect("workflow should be present");

    assert_eq!(workflow["readiness"]["ready"], false);
    assert_eq!(workflow["readiness"]["recommended_check_phase"], "prepare");
    assert!(
        workflow["readiness"]["checks"]
            .as_array()
            .expect("checks should be an array")
            .iter()
            .any(|check| check["type"] == "env" && check["ok"] == false),
        "expected at least one failed env readiness check, got: {workflow}"
    );
    assert!(
        workflow["readiness"]["checks"]
            .as_array()
            .expect("checks should be an array")
            .iter()
            .any(|check| check["type"] == "python_module"),
        "expected python module readiness check, got: {workflow}"
    );
}

#[tokio::test]
async fn test_list_workflows_uses_cached_readiness_contract() {
    let (probe_executable, counter_path) = create_counting_python_probe("list-cache");
    let state = create_mock_state_with_python_runtime(
        "python.test",
        probe_executable.display().to_string(),
    );
    let _plugin =
        register_custom_plugin(&state, "list-cache", readiness_plugin_manifest_yaml()).await;
    let app = build_router(state);

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/infer/workflows")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let probe_count = fs::read_to_string(&counter_path)
        .expect("probe counter should be readable")
        .trim()
        .parse::<u32>()
        .expect("probe counter should be numeric");
    assert_eq!(
        probe_count, 1,
        "workflow list should reuse the cached readiness contract instead of probing per request"
    );
}

#[tokio::test]
async fn test_readyz_returns_503_when_redis_unavailable() {
    let state = create_mock_state(); // redis_service = None
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "readyz must return 503 when Redis is not initialized (PR-096)"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "not ready");
}

#[tokio::test]
async fn test_get_run_status_uses_workflow_namespaced_route() {
    let state = create_mock_state();
    state.runs.write().await.insert(
        "existing-run-id".to_string(),
        json!({
            "workflow": "demo-prefetch",
            "status": "queued"
        }),
    );
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/demo-prefetch/existing-run-id/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["workflow"], "demo-prefetch");
    assert_eq!(json["status"], "queued");
}

#[tokio::test]
async fn test_get_run_status_returns_404_for_unknown_run() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/demo-prefetch/nonexistent-run-id/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "get_run must return 404 for unknown run_id (PR-096)"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"].as_str().unwrap().contains("not found"),
        "Error message should mention 'not found'"
    );
}

#[tokio::test]
async fn test_get_run_status_returns_404_for_workflow_mismatch() {
    let state = create_mock_state();
    state.runs.write().await.insert(
        "existing-run-id".to_string(),
        json!({
            "workflow": "demo-prefetch",
            "status": "queued"
        }),
    );
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/other-workflow/existing-run-id/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_run_status_returns_404_for_disabled_workflow() {
    let state = create_mock_state_with_enabled_plugin_id("enabled-workflow");
    state.runs.write().await.insert(
        "disabled-run-id".to_string(),
        json!({
            "workflow": "disabled-workflow",
            "status": "queued"
        }),
    );
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/disabled-workflow/disabled-run-id/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_result_uses_workflow_namespaced_route() {
    let state = create_mock_state(); // redis_service = None
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/demo-prefetch/some-run-id/results")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "get_result must return 503 when Redis is unavailable (PR-096)"
    );
}

#[tokio::test]
async fn test_get_result_returns_404_for_disabled_workflow_before_redis_lookup() {
    let state = create_mock_state_with_enabled_plugin_id("enabled-workflow");
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/disabled-workflow/disabled-run-id/results")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[test]
fn test_openapi_result_endpoint_documents_artifact_query_parameter() {
    let spec = build_openapi_json();
    let get = &spec["paths"]["/v1/infer/{workflow}/{run_id}/results"]["get"];
    let params = get["parameters"]
        .as_array()
        .expect("result endpoint parameters should be an array");

    assert!(params.iter().any(|param| {
        param["name"] == "artifact" && param["in"] == "query" && param["schema"]["type"] == "string"
    }));

    assert!(params.iter().any(|param| {
        param["name"] == "format"
            && param["in"] == "query"
            && param["schema"]["enum"] == json!(["netcdf", "zarr_zip"])
    }));

    assert!(params.iter().any(|param| {
        param["name"] == "vars" && param["in"] == "query" && param["schema"]["type"] == "string"
    }));
}

#[test]
fn test_openapi_result_endpoint_documents_structured_result_envelope() {
    let spec = build_openapi_json();
    let schema = &spec["paths"]["/v1/infer/{workflow}/{run_id}/results"]["get"]["responses"]["200"]
        ["content"]["application/json"]["schema"];

    assert_eq!(schema["type"], "object");
    assert_eq!(
        schema["required"],
        json!(["request", "execution", "payload"])
    );
    assert_eq!(schema["properties"]["request"]["type"], "object");
    assert_eq!(schema["properties"]["execution"]["type"], "object");
    assert_eq!(
        schema["properties"]["execution"]["properties"]["output_path"]["type"],
        "string"
    );
    assert_eq!(
        schema["properties"]["execution"]["properties"]["outputs"]["type"],
        "array"
    );
    assert_eq!(
        schema["properties"]["execution"]["properties"]["outputs"]["items"]["properties"]["name"]["type"],
        "string"
    );
    assert_eq!(schema["properties"]["payload"]["type"], "object");
}

#[test]
fn test_openapi_workflow_listing_documents_readiness() {
    let spec = build_openapi_json();
    let workflows = &spec["paths"]["/v1/infer/workflows"]["get"]["responses"]["200"]["content"]["application/json"]
        ["schema"]["properties"]["workflows"]["items"]["properties"];

    assert_eq!(workflows["readiness"]["type"], "object");
    assert_eq!(
        workflows["readiness"]["properties"]["ready"]["type"],
        "boolean"
    );
}

#[test]
fn test_openapi_workflow_schema_documents_readiness() {
    let spec = build_openapi_json();
    let schema_properties = &spec["paths"]["/v1/infer/{name}/schema"]["get"]["responses"]["200"]["content"]
        ["application/json"]["schema"]["properties"];

    assert_eq!(schema_properties["readiness"]["type"], "object");
    assert_eq!(
        schema_properties["readiness"]["properties"]["recommended_check_phase"]["type"],
        "string"
    );
}

#[test]
fn test_openapi_workflow_readiness_endpoint_documents_response() {
    let spec = build_openapi_json();
    let readiness_properties = &spec["paths"]["/v1/infer/{name}/readiness"]["get"]["responses"]["200"]
        ["content"]["application/json"]["schema"]["properties"];

    assert_eq!(readiness_properties["workflow_id"]["type"], "string");
    assert_eq!(readiness_properties["plugin"]["type"], "boolean");
    assert_eq!(readiness_properties["readiness"]["type"], "object");
}

#[test]
fn test_openapi_run_workflow_mentions_readiness_gate() {
    let spec = build_openapi_json();
    let run_post = &spec["paths"]["/v1/infer/{name}/run"]["post"];
    let responses = run_post["responses"]
        .as_object()
        .expect("run responses should be an object");

    assert!(responses.contains_key("503"));
    assert!(
        run_post["description"]
            .as_str()
            .expect("run description should be a string")
            .contains("readiness"),
        "run endpoint description should mention readiness gating"
    );
}

#[tokio::test]
async fn test_run_workflow_returns_404_for_unknown_workflow() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/nonexistent/run")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"parameters":{}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "run_workflow must return 404 for unknown workflow (PR-096)"
    );
}

#[tokio::test]
async fn test_get_nonexistent_workflow_schema() {
    let state = create_mock_state();
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/nonexistent_workflow/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 404 Not Found
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_plugin_workflow_schema_returns_404_for_disabled_workflow() {
    let state = create_mock_state_with_enabled_plugin_id("enabled-workflow");
    let plugin = register_plugin(&state, "schema-disabled").await;
    let app = build_router(state);

    let uri = format!("/v1/infer/{}/schema", plugin.manifest.metadata.id);
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_plugin_workflow_schema_returns_manifest_backed_contract() {
    let state = create_mock_state();
    let _plugin = register_plugin(&state, "schema").await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/infer/demo-prefetch/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["workflow_id"], "demo-prefetch");
    assert_eq!(json["default_content_type"], "application/json");
    assert_eq!(json["operations"]["default"], "run");
    assert_eq!(
        json["request_schemas"]["application/json"]["required"],
        serde_json::json!(["start_time"])
    );
    assert_eq!(json["primary_artifact"]["name"], "forecast_dataset");
    assert_eq!(json["readiness"]["ready"], true);
}

#[tokio::test]
async fn test_run_workflow_uses_cached_request_schema_after_schema_probe() {
    let state = create_mock_state();
    let plugin = register_plugin(&state, "schema-cache").await;
    let app = build_router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/infer/demo-prefetch/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    fs::write(
        plugin.root_dir.join("schemas").join("request.json"),
        r#"{
            "type":"object",
            "required":["different_field"],
            "properties":{
                "different_field":{"type":"string"}
            },
            "additionalProperties":false
        }"#,
    )
    .expect("request schema should be rewritten");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "valid parameters under the cached schema should pass validation and reach the Redis boundary"
    );
}

#[tokio::test]
async fn test_run_workflow_uses_request_schema_when_result_schema_cache_fails() {
    let state = create_mock_state();
    let plugin = register_plugin(&state, "result-schema-broken").await;
    fs::remove_file(plugin.root_dir.join("schemas").join("result.json"))
        .expect("result schema should be removed");
    register_plugin_contract(&state, &plugin).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "result schema cache failures must not block request-schema validation for run submission"
    );
}

#[tokio::test]
async fn test_get_plugin_workflow_schema_includes_readiness_status() {
    let state = create_mock_state();
    let plugin =
        register_custom_plugin(&state, "schema-readiness", readiness_plugin_manifest_yaml()).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/{}/schema", plugin.manifest.metadata.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["readiness"]["ready"], false);
    assert_eq!(json["readiness"]["recommended_check_phase"], "prepare");
    assert!(
        json["readiness"]["checks"]
            .as_array()
            .expect("checks should be an array")
            .iter()
            .any(|check| check["type"] == "env" && check["ok"] == false),
        "expected failed readiness env check, got: {json}"
    );
}

#[tokio::test]
async fn test_get_plugin_workflow_schema_supports_inline_manifest_schemas() {
    let state = create_mock_state();
    let plugin = register_custom_plugin(
        &state,
        "inline-schema",
        inline_schema_plugin_manifest_yaml(),
    )
    .await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/{}/schema", plugin.manifest.metadata.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        json["request_schemas"]["application/json"]["required"],
        serde_json::json!(["value"])
    );
    assert_eq!(
        json["result_schema"]["properties"]["status"]["type"],
        "string"
    );
}

#[tokio::test]
#[ignore = "requires Python with PyYAML"]
async fn test_get_plugin_workflow_schema_supports_model_driven_schema_derivation() {
    let state = create_mock_state();
    let plugin = register_model_driven_plugin(&state, "model-driven-schema").await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/{}/schema", plugin.manifest.metadata.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        json["request_schemas"]["application/json"]["required"],
        serde_json::json!(["value"])
    );
    assert_eq!(
        json["result_schema"]["required"],
        serde_json::json!(["value", "doubled"])
    );
}

#[tokio::test]
async fn test_get_plugin_workflow_readiness_returns_manifest_backed_status() {
    let state = create_mock_state();
    let plugin = register_custom_plugin(
        &state,
        "endpoint-readiness",
        readiness_plugin_manifest_yaml(),
    )
    .await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/infer/{}/readiness",
                    plugin.manifest.metadata.id
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["workflow_id"], plugin.manifest.metadata.id);
    assert_eq!(json["plugin"], true);
    assert_eq!(json["readiness"]["ready"], false);
    assert!(
        json["readiness"]["checks"]
            .as_array()
            .expect("checks should be an array")
            .iter()
            .any(|check| check["type"] == "env" && check["ok"] == false),
        "expected failed readiness env check, got: {json}"
    );
}

#[tokio::test]
async fn test_run_plugin_workflow_rejects_invalid_parameters_before_redis() {
    let state = create_mock_state();
    let _plugin = register_plugin(&state, "run-invalid").await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"parameters":{"num_steps":20}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["error"], "Parameter validation failed");
    assert!(
        json["validation_errors"]
            .as_array()
            .expect("validation_errors should be an array")
            .iter()
            .any(|err| err.as_str().unwrap_or_default().contains("start_time")),
        "expected validation error mentioning start_time, got: {json}"
    );
}

#[tokio::test]
async fn test_run_plugin_workflow_rejects_not_ready_plugin_before_validation() {
    let state = create_mock_state();
    let plugin =
        register_custom_plugin(&state, "run-not-ready", readiness_plugin_manifest_yaml()).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/infer/{}/run", plugin.manifest.metadata.id))
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["error"], "Workflow is not ready");
    assert_eq!(json["workflow"], plugin.manifest.metadata.id);
    assert_eq!(json["readiness"]["ready"], false);
    assert!(
        json["readiness"]["checks"]
            .as_array()
            .expect("checks should be an array")
            .iter()
            .any(|check| check["type"] == "env" && check["ok"] == false),
        "expected failed readiness env check, got: {json}"
    );
}

#[tokio::test]
async fn test_run_ready_plugin_still_reaches_parameter_validation() {
    let state = create_mock_state();
    let _plugin = register_plugin(&state, "run-ready-invalid").await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"parameters":{"num_steps":20}}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_run_multipart_plugin_workflow_accepts_valid_request_shape() {
    let state = create_mock_state();
    let _plugin = register_multipart_plugin(&state, "run-multipart-valid").await;
    let app = build_router(state);
    let boundary = "X-BOUNDARY";
    let body = multipart_body(
        boundary,
        &[
            ("operation", None, "both"),
            ("batch_size", None, "128000"),
            (
                "design_stl",
                Some(("mesh.stl", "model/stl")),
                "solid cube\nendsolid cube",
            ),
        ],
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/multipart-demo/run")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "valid multipart plugin requests should reach the redis boundary"
    );
}

#[tokio::test]
async fn test_run_multipart_plugin_workflow_rolls_back_staged_artifacts_after_late_failure() {
    let state = create_mock_state();
    let artifact_root = state.config.artifact_dir.clone();
    let _plugin = register_multipart_plugin(&state, "run-multipart-artifact-rollback").await;
    let app = build_router(state);
    let boundary = "X-BOUNDARY";
    let body = multipart_body(
        boundary,
        &[
            ("operation", None, "both"),
            ("batch_size", None, "128000"),
            (
                "design_stl",
                Some(("mesh.stl", "model/stl")),
                "solid cube\nendsolid cube",
            ),
        ],
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/multipart-demo/run")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "multipart staging should still reach the later Redis boundary in this regression"
    );

    let leaked_run_dirs = if artifact_root.exists() {
        fs::read_dir(&artifact_root)
            .expect("artifact root should be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name != ".incoming")
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    assert!(
        leaked_run_dirs.is_empty(),
        "late failures must roll back staged per-run artifact directories, found: {leaked_run_dirs:?}"
    );
}

#[tokio::test]
async fn test_run_multipart_plugin_workflow_rejects_missing_required_file() {
    let state = create_mock_state();
    let _plugin = register_multipart_plugin(&state, "run-multipart-missing-file").await;
    let app = build_router(state);
    let boundary = "X-BOUNDARY";
    let body = multipart_body(boundary, &[("batch_size", None, "128000")]);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/multipart-demo/run")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_run_multipart_plugin_workflow_accepts_upload_larger_than_old_default_limit() {
    let state = create_mock_state_with_max_body_size(8 * 1024 * 1024);
    let _plugin = register_multipart_plugin(&state, "run-multipart-large-valid").await;
    let app = build_router(state);
    let boundary = "X-BOUNDARY";
    let large_stl = "s".repeat(3 * 1024 * 1024);
    let body = multipart_body(
        boundary,
        &[
            ("operation", None, "both"),
            ("batch_size", None, "128000"),
            (
                "design_stl",
                Some(("mesh.stl", "model/stl")),
                large_stl.as_str(),
            ),
        ],
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/multipart-demo/run")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "multipart upload larger than the old 2 MiB default should reach plugin validation/enqueue boundary"
    );
}

#[tokio::test]
async fn test_run_multipart_plugin_workflow_rejects_file_exceeding_plugin_max_size() {
    let state = create_mock_state_with_max_body_size(8 * 1024 * 1024);
    let _plugin = register_multipart_plugin(&state, "run-multipart-too-large").await;
    let app = build_router(state);
    let boundary = "X-BOUNDARY";
    let oversized_stl = "s".repeat(5 * 1024 * 1024);
    let body = multipart_body(
        boundary,
        &[
            ("operation", None, "both"),
            ("batch_size", None, "128000"),
            (
                "design_stl",
                Some(("mesh.stl", "model/stl")),
                oversized_stl.as_str(),
            ),
        ],
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/multipart-demo/run")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[test]
fn test_run_endpoint_openapi_documents_multipart_and_pipeline_array() {
    let spec = build_openapi_json();
    let run_post = &spec["paths"]["/v1/infer/{name}/run"]["post"];
    let content = &run_post["requestBody"]["content"];

    assert!(
        content.get("application/json").is_some(),
        "run endpoint should continue to document JSON requests"
    );
    assert!(
        content.get("multipart/form-data").is_some(),
        "run endpoint should document multipart requests supported by the handler"
    );

    let pipeline_schema = &run_post["responses"]["202"]["content"]["application/json"]["schema"]["properties"]
        ["pipeline"];
    assert_eq!(
        pipeline_schema["type"], "array",
        "run response pipeline should match the handler's array payload"
    );
    assert_eq!(
        pipeline_schema["items"]["type"], "string",
        "pipeline entries should be stage phase strings"
    );
}

#[tokio::test]
async fn test_run_plugin_workflow_does_not_leave_stale_run_after_failure() {
    let state = create_mock_state();
    let _plugin = register_plugin(&state, "run-stale-run-rollback").await;
    let app = build_router(state.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        state.runs.read().await.is_empty(),
        "failed submissions must not leave stale queued runs in memory"
    );
}

#[tokio::test]
async fn test_run_submission_fails_when_initial_queued_run_record_cannot_be_persisted() {
    let port = reserve_port();
    let redis_server = TestRedisServer::spawn("queued-run-persistence-failure", port).await;
    let state =
        create_mock_state_for_redis_url(format!("redis://127.0.0.1:{port}/0"), 2 * 1024 * 1024)
            .await;
    let _plugin = register_plugin(&state, "queued-run-persistence-failure").await;
    let app = build_router(state.clone());

    drop(redis_server);
    wait_for_tcp_listener_to_close(port).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "the API must not accept a run unless the initial queued run record is durable"
    );
    assert!(
        state.runs.read().await.is_empty(),
        "failed submissions must not leave local queued runs behind"
    );
}

#[tokio::test]
async fn test_immediate_status_poll_after_run_submission_returns_queued() {
    let (state, _redis_server) =
        create_mock_state_with_redis("immediate-status", 2 * 1024 * 1024).await;
    let _plugin = register_plugin(&state, "immediate-status").await;
    let app = build_router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let queued: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let run_id = queued["run_id"]
        .as_str()
        .expect("run response should contain a run_id");

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/demo-prefetch/{run_id}/status"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "immediate status polling should succeed after a run is accepted"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["workflow"], "demo-prefetch");
    assert_eq!(status["status"], "queued");
}

#[tokio::test]
async fn test_immediate_status_poll_falls_back_to_in_memory_when_redis_read_fails() {
    let port = reserve_port();
    let redis_server = TestRedisServer::spawn("immediate-status-redis-read-failure", port).await;
    let state =
        create_mock_state_for_redis_url(format!("redis://127.0.0.1:{port}/0"), 2 * 1024 * 1024)
            .await;
    let run_id = "local-run-redis-read-failure";
    state.runs.write().await.insert(
        run_id.to_string(),
        json!({
            "workflow": "demo-prefetch",
            "status": "queued",
            "stage": "prepare",
            "updated_at": "1",
            "api_received_at": "1",
            "api_enqueued_at": "1"
        }),
    );
    let app = build_router(state);

    drop(redis_server);
    wait_for_tcp_listener_to_close(port).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/demo-prefetch/{run_id}/status"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "same-instance status polling should use in-memory state when Redis reads fail"
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["workflow"], "demo-prefetch");
    assert_eq!(status["status"], "queued");
}

#[tokio::test]
async fn test_immediate_status_poll_succeeds_from_fresh_instance() {
    let (state, _redis_server) =
        create_mock_state_with_redis("immediate-status-fresh-instance", 2 * 1024 * 1024).await;
    let redis_url = state.config.redis_url.clone();
    let _plugin = register_plugin(&state, "immediate-status-fresh-instance").await;
    let app = build_router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/infer/demo-prefetch/run")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"parameters":{"start_time":["2026-03-18T00:00:00Z"]}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let queued: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let run_id = queued["run_id"]
        .as_str()
        .expect("run response should contain a run_id");

    let fresh_state = create_mock_state_for_redis_url(redis_url, 2 * 1024 * 1024).await;
    let fresh_app = build_router(fresh_state);
    let response = fresh_app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/infer/demo-prefetch/{run_id}/status"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "immediate status polling should work from a fresh server instance"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["workflow"], "demo-prefetch");
    assert_eq!(status["status"], "queued");
}

// ---------------------------------------------------------------------------
// Request schema validation tests for plugin ingress.json_schema_inline
// ---------------------------------------------------------------------------

fn inline_schema_manifest(workflow_id: &str, schema_json: &str) -> String {
    format!(
        r#"
metadata:
  id: {workflow_id}
  display_name: Test {workflow_id}
  version: 1.0.0
  description: Schema validation test
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline: {schema_json}
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
outputs:
  result_schema_inline:
    type: object
    properties:
      status:
        type: string
"#
    )
}

async fn assert_schema_rejects(workflow_id: &str, schema_json: &str, body: &str) {
    let state = create_mock_state();
    let manifest = inline_schema_manifest(workflow_id, schema_json);
    let _plugin = register_custom_plugin(&state, &format!("neg-{workflow_id}"), &manifest).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/infer/{workflow_id}/run"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "expected 422 for workflow={workflow_id} body={body}"
    );
}

async fn assert_schema_accepts(workflow_id: &str, schema_json: &str, body: &str) {
    let state = create_mock_state();
    let manifest = inline_schema_manifest(workflow_id, schema_json);
    let _plugin = register_custom_plugin(&state, &format!("pos-{workflow_id}"), &manifest).await;
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/infer/{workflow_id}/run"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_ne!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "schema should accept valid params for workflow={workflow_id} body={body}"
    );
}

const E2S_ENSEMBLE_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["forecast_times","nsteps","nensemble"],"properties":{"forecast_times":{"type":"array","items":{"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"},"minItems":1},"nsteps":{"type":"integer","minimum":1},"nensemble":{"type":"integer","minimum":1},"batch_size":{"type":"integer","minimum":1},"model_type":{"type":"string","enum":["fcn"]},"noise_amplitude":{"type":"number","exclusiveMinimum":0},"data_source":{"type":"string","enum":["gfs"]},"output_format":{"type":"string","enum":["zarr"]},"output_variables":{"type":"array","items":{"type":"string"}},"create_plots":{"type":"boolean"},"plot_variable":{"type":"string","enum":["t2m","msl","u10m","v10m","tcwv","z500"]},"plot_step":{"type":"integer","minimum":0}}}"#;

const EARTH2_DETERMINISTIC_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["model","start_time","nsteps"],"properties":{"model":{"type":"string","enum":["dlwp"]},"start_time":{"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}","minLength":1},"nsteps":{"type":"integer","minimum":1}}}"#;

const EARTH2_ENSEMBLE_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["model","start_time","nsteps","nensemble"],"properties":{"model":{"type":"string","enum":["dlwp"]},"start_time":{"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}","minLength":1},"nsteps":{"type":"integer","minimum":1},"nensemble":{"type":"integer","minimum":1},"batch_size":{"type":"integer","minimum":1},"perturbation":{"type":"string","enum":["gaussian","brown"]},"noise_amplitude":{"type":"number","exclusiveMinimum":0},"seed_base":{"type":"integer"}}}"#;

const E2S_STORMCAST_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["start_time"],"properties":{"start_time":{"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}","minLength":1},"num_hours":{"type":"integer","minimum":1},"run_stormcast":{"type":"boolean"}}}"#;

const E2S_EXAMPLE_USER_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"task_name":{"type":"string"},"num_iterations":{"type":"integer","minimum":1},"delay_seconds":{"type":"number","minimum":0},"generate_output":{"type":"boolean"}}}"#;

#[tokio::test]
async fn test_e2s_ensemble_rejects_nensemble_zero() {
    assert_schema_rejects(
        "e2s-ensemble",
        E2S_ENSEMBLE_SCHEMA,
        r#"{"forecast_times":["2024-01-01T00:00:00"],"nsteps":4,"nensemble":0}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_ensemble_rejects_empty_forecast_times() {
    assert_schema_rejects(
        "e2s-ensemble",
        E2S_ENSEMBLE_SCHEMA,
        r#"{"forecast_times":[],"nsteps":4,"nensemble":2}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_ensemble_rejects_invalid_datetime_pattern() {
    assert_schema_rejects(
        "e2s-ensemble",
        E2S_ENSEMBLE_SCHEMA,
        r#"{"forecast_times":["not-a-datetime"],"nsteps":4,"nensemble":2}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_deterministic_rejects_invalid_datetime_pattern() {
    assert_schema_rejects(
        "earth2-deterministic",
        EARTH2_DETERMINISTIC_SCHEMA,
        r#"{"model":"dlwp","start_time":"not-a-datetime","nsteps":3}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_ensemble_accepts_valid_params() {
    assert_schema_accepts(
        "e2s-ensemble",
        E2S_ENSEMBLE_SCHEMA,
        r#"{"forecast_times":["2024-01-01T00:00:00"],"nsteps":4,"nensemble":2}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_deterministic_rejects_invalid_model() {
    assert_schema_rejects(
        "earth2-deterministic",
        EARTH2_DETERMINISTIC_SCHEMA,
        r#"{"model":"no_such_model","start_time":"2024-01-01T00:00:00","nsteps":3}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_deterministic_rejects_nsteps_zero() {
    assert_schema_rejects(
        "earth2-deterministic",
        EARTH2_DETERMINISTIC_SCHEMA,
        r#"{"model":"dlwp","start_time":"2024-01-01T00:00:00","nsteps":0}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_deterministic_accepts_valid_params() {
    assert_schema_accepts(
        "earth2-deterministic",
        EARTH2_DETERMINISTIC_SCHEMA,
        r#"{"model":"dlwp","start_time":"2024-01-01T00:00:00","nsteps":6}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_ensemble_rejects_invalid_perturbation() {
    assert_schema_rejects(
        "earth2-ensemble",
        EARTH2_ENSEMBLE_SCHEMA,
        r#"{"model":"dlwp","start_time":"2024-01-01T00:00:00","nsteps":3,"nensemble":2,"perturbation":"invalid_method"}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_ensemble_rejects_negative_noise() {
    assert_schema_rejects(
        "earth2-ensemble",
        EARTH2_ENSEMBLE_SCHEMA,
        r#"{"model":"dlwp","start_time":"2024-01-01T00:00:00","nsteps":3,"nensemble":2,"noise_amplitude":-0.5}"#,
    )
    .await;
}

#[tokio::test]
async fn test_earth2_ensemble_accepts_valid_params() {
    assert_schema_accepts(
        "earth2-ensemble",
        EARTH2_ENSEMBLE_SCHEMA,
        r#"{"model":"dlwp","start_time":"2024-01-01T00:00:00","nsteps":3,"nensemble":2,"perturbation":"gaussian","noise_amplitude":0.05}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_stormcast_rejects_empty_start_time() {
    assert_schema_rejects(
        "e2s-stormcast-fcn3",
        E2S_STORMCAST_SCHEMA,
        r#"{"start_time":"","num_hours":6}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_stormcast_accepts_valid_params() {
    assert_schema_accepts(
        "e2s-stormcast-fcn3",
        E2S_STORMCAST_SCHEMA,
        r#"{"start_time":"2024-01-01T00:00:00","num_hours":6,"run_stormcast":true}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_example_user_rejects_zero_iterations() {
    assert_schema_rejects(
        "e2s-example-user",
        E2S_EXAMPLE_USER_SCHEMA,
        r#"{"num_iterations":0}"#,
    )
    .await;
}

#[tokio::test]
async fn test_e2s_example_user_accepts_valid_params() {
    assert_schema_accepts(
        "e2s-example-user",
        E2S_EXAMPLE_USER_SCHEMA,
        r#"{"task_name":"test","num_iterations":3,"delay_seconds":0.1}"#,
    )
    .await;
}
