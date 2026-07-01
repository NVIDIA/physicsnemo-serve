/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Application state with bounded in-memory run cache.

use crate::config::ServerConfig;
use crate::metrics::SharedMetrics;
use crate::plugin_registry::{
    PluginReadinessReport, PythonModuleProbe, RegisteredPlugin, discover_plugins_with_enabled_id,
};
use crate::redis_ops::RedisService;
use serde_json::Value;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Maximum number of in-memory run entries before eviction.
const DEFAULT_MAX_RUNS: usize = 10_000;

/// Bounded map that evicts the oldest entries when full.
pub struct BoundedRunMap {
    map: HashMap<String, Value>,
    insertion_order: VecDeque<String>,
    capacity: usize,
}

impl BoundedRunMap {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            insertion_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn insert(&mut self, key: String, value: Value) {
        let is_update = self.map.contains_key(&key);
        if is_update {
            self.map.insert(key, value);
            return;
        }
        while self.map.len() >= self.capacity {
            if let Some(oldest) = self.insertion_order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
        self.insertion_order.push_back(key.clone());
        self.map.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Clone)]
pub struct CachedWorkflowContract {
    pub plugin: RegisteredPlugin,
    pub summary: Value,
    pub schema_contract: Result<Value, String>,
    pub request_schemas: Result<HashMap<String, Value>, String>,
    pub readiness: PluginReadinessReport,
}

impl CachedWorkflowContract {
    pub fn build(plugin: RegisteredPlugin, readiness_probe: &mut PythonModuleProbe) -> Self {
        let request_schemas = plugin.load_request_schemas().map_err(|err| err.to_string());
        let result_schema = plugin.load_result_schema().map_err(|err| err.to_string());
        let readiness = plugin.evaluate_readiness(readiness_probe);
        let summary = workflow_summary(&plugin, &readiness);
        let schema_contract = match (&request_schemas, &result_schema) {
            (Ok(request_schemas), Ok(result_schema)) => Ok(workflow_schema_contract(
                &plugin,
                &readiness,
                request_schemas,
                result_schema,
            )),
            (Err(err), _) | (_, Err(err)) => Err(err.clone()),
        };

        Self {
            plugin,
            summary,
            schema_contract,
            request_schemas,
            readiness,
        }
    }
}

#[derive(Clone, Default)]
pub struct WorkflowRegistrySnapshot {
    pub contracts: HashMap<String, CachedWorkflowContract>,
}

fn workflow_summary(plugin: &RegisteredPlugin, readiness: &PluginReadinessReport) -> Value {
    json!({
        "name": plugin.manifest.metadata.id,
        "display_name": plugin.manifest.metadata.display_name,
        "description": plugin.manifest.metadata.description,
        "version": plugin.manifest.metadata.version,
        "content_types": plugin.manifest.ingress.content_types,
        "default_content_type": plugin.manifest.ingress.default_content_type,
        "operations": plugin.manifest.ingress.operations.allowed,
        "default_operation": plugin.manifest.ingress.operations.default,
        "executor_class": plugin.manifest.runtime.executor_class,
        "pipeline": plugin
            .manifest
            .pipeline
            .stages
            .iter()
            .map(|stage| stage.phase.clone())
            .collect::<Vec<_>>(),
        "readiness": readiness,
        "plugin": true
    })
}

fn workflow_schema_contract(
    plugin: &RegisteredPlugin,
    readiness: &PluginReadinessReport,
    request_schemas: &HashMap<String, Value>,
    result_schema: &Value,
) -> Value {
    json!({
        "workflow_id": plugin.manifest.metadata.id,
        "display_name": plugin.manifest.metadata.display_name,
        "description": plugin.manifest.metadata.description,
        "version": plugin.manifest.metadata.version,
        "content_types": plugin.manifest.ingress.content_types,
        "default_content_type": plugin.manifest.ingress.default_content_type,
        "operations": plugin.manifest.ingress.operations,
        "readiness": readiness,
        "request_schemas": request_schemas,
        "files": plugin.manifest.ingress.files,
        "result_schema": result_schema,
        "primary_artifact": plugin.manifest.outputs.primary_artifact,
        "retention_hours": plugin.manifest.outputs.retention_hours
    })
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub redis_service: Arc<RwLock<Option<RedisService>>>,
    pub workflow_registry: Arc<RwLock<WorkflowRegistrySnapshot>>,
    pub runs: Arc<RwLock<BoundedRunMap>>,
    pub openapi: Arc<RwLock<Value>>,
    pub metrics: SharedMetrics,
}

impl AppState {
    pub fn new(config: ServerConfig, redis_service: RedisService) -> Self {
        Self {
            config: Arc::new(config),
            redis_service: Arc::new(RwLock::new(Some(redis_service))),
            workflow_registry: Arc::new(RwLock::new(WorkflowRegistrySnapshot::default())),
            runs: Arc::new(RwLock::new(BoundedRunMap::new(DEFAULT_MAX_RUNS))),
            openapi: Arc::new(RwLock::new(crate::openapi::build_openapi_json())),
            metrics: crate::metrics::create_shared_metrics(),
        }
    }

    #[doc(hidden)]
    pub fn new_for_testing(config: ServerConfig) -> Self {
        Self {
            config: Arc::new(config),
            redis_service: Arc::new(RwLock::new(None)),
            workflow_registry: Arc::new(RwLock::new(WorkflowRegistrySnapshot::default())),
            runs: Arc::new(RwLock::new(BoundedRunMap::new(DEFAULT_MAX_RUNS))),
            openapi: Arc::new(RwLock::new(crate::openapi::build_openapi_json())),
            metrics: crate::metrics::create_shared_metrics(),
        }
    }

    pub async fn refresh_plugins(&self) -> anyhow::Result<usize> {
        let discovered = discover_plugins_with_enabled_id(
            &self.config.plugin_dirs,
            self.config.enabled_plugin_id.as_deref(),
        )?;
        let mut contracts_by_id = HashMap::with_capacity(discovered.len());
        let mut readiness_probe =
            PythonModuleProbe::with_runtime_envs(self.config.python_runtime_envs.clone());
        for plugin in discovered {
            let plugin_id = plugin.manifest.metadata.id.clone();
            let contract = CachedWorkflowContract::build(plugin.clone(), &mut readiness_probe);
            if self.config.enabled_plugin_id.as_deref() == Some(plugin_id.as_str())
                && let Err(err) = &contract.schema_contract
            {
                anyhow::bail!(
                    "enabled plugin '{}' failed workflow contract probe: {}",
                    plugin_id,
                    err
                );
            }
            contracts_by_id.insert(plugin_id, contract);
        }

        let count = contracts_by_id.len();
        *self.workflow_registry.write().await = WorkflowRegistrySnapshot {
            contracts: contracts_by_id,
        };
        Ok(count)
    }

    pub async fn get_workflow_contract(&self, workflow_id: &str) -> Option<CachedWorkflowContract> {
        self.workflow_registry
            .read()
            .await
            .contracts
            .get(workflow_id)
            .cloned()
    }

    pub async fn list_workflow_summaries(&self) -> Vec<Value> {
        self.workflow_registry
            .read()
            .await
            .contracts
            .values()
            .filter(|contract| self.is_workflow_enabled(&contract.plugin.manifest.metadata.id))
            .map(|contract| contract.summary.clone())
            .collect()
    }

    pub fn is_workflow_enabled(&self, workflow_id: &str) -> bool {
        match self.config.enabled_plugin_id.as_deref() {
            Some(enabled_plugin_id) => enabled_plugin_id == workflow_id,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;

    fn plugin_manifest_yaml(id: &str) -> String {
        format!(
            r#"
metadata:
  id: {id}
  display_name: {id}
  version: 1.0.0
  description: Plugin {id}
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
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.gpu
resources:
  defaults:
    gpus_required: 1
    memory_mb: 24000
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: output
    media_type: application/json
  retention_hours: 24
"#
        )
    }

    fn create_plugin_dir(parent: &Path, id: &str) {
        let plugin_dir = parent.join(id);
        let schemas_dir = plugin_dir.join("schemas");
        fs::create_dir_all(&schemas_dir).expect("plugin schema directory should be created");
        fs::write(plugin_dir.join("plugin.yaml"), plugin_manifest_yaml(id))
            .expect("plugin manifest should be written");
        fs::write(
            schemas_dir.join("request.json"),
            r#"{"type":"object","properties":{"value":{"type":"integer"}}}"#,
        )
        .expect("request schema should be written");
        fs::write(
            schemas_dir.join("result.json"),
            r#"{"type":"object","properties":{"status":{"type":"string"}}}"#,
        )
        .expect("result schema should be written");
    }

    #[test]
    fn test_app_state_initialization() {
        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);

        let registry = state.workflow_registry.blocking_read();
        assert!(registry.contracts.is_empty());
        drop(registry);
        assert!(state.runs.blocking_read().is_empty());
        assert!(state.redis_service.blocking_read().is_none());

        let openapi = state.openapi.blocking_read();
        assert_eq!(openapi["openapi"], "3.0.3");
        assert_eq!(openapi["info"]["title"], "PhysicsNeMo Serve Inference API");
    }

    #[test]
    fn test_bounded_run_map_evicts_oldest_when_full() {
        let mut map = BoundedRunMap::new(3);
        map.insert("a".into(), json!({"status": "a"}));
        map.insert("b".into(), json!({"status": "b"}));
        map.insert("c".into(), json!({"status": "c"}));
        assert_eq!(map.len(), 3);

        // Insert 4th - "a" should be evicted
        map.insert("d".into(), json!({"status": "d"}));
        assert_eq!(map.len(), 3);
        assert!(map.get("a").is_none(), "oldest entry should be evicted");
        assert!(map.get("d").is_some());
    }

    #[test]
    fn test_bounded_run_map_update_existing_does_not_evict() {
        let mut map = BoundedRunMap::new(2);
        map.insert("a".into(), json!({"v": 1}));
        map.insert("b".into(), json!({"v": 2}));

        // Update "a" - should NOT evict anything
        map.insert("a".into(), json!({"v": 99}));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a").unwrap(), &json!({"v": 99}));
        assert!(map.get("b").is_some());
    }

    #[tokio::test]
    async fn test_refresh_plugins_loads_plugins_from_configured_dirs() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "demo-a");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        let count = state
            .refresh_plugins()
            .await
            .expect("plugin refresh should succeed");

        let registry = state.workflow_registry.read().await;
        assert_eq!(count, 1);
        assert!(registry.contracts.contains_key("demo-a"));
    }

    #[tokio::test]
    async fn test_refresh_plugins_isolates_workflow_contract_build_failures() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-contract-failure-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "healthy-plugin");
        create_plugin_dir(&root, "broken-plugin");
        fs::remove_file(
            root.join("broken-plugin")
                .join("schemas")
                .join("request.json"),
        )
        .expect("request schema should be removed to force contract failure");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-contract-failure-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-contract-failure-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        let count = state
            .refresh_plugins()
            .await
            .expect("one broken workflow contract must not abort plugin refresh");

        let registry = state.workflow_registry.read().await;
        assert_eq!(count, 2);
        assert!(registry.contracts.contains_key("healthy-plugin"));
        assert!(registry.contracts.contains_key("broken-plugin"));
    }

    #[tokio::test]
    async fn test_refresh_plugins_fails_when_enabled_plugin_contract_probe_fails() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-enabled-contract-failure-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "healthy-plugin");
        create_plugin_dir(&root, "broken-plugin");
        fs::remove_file(
            root.join("broken-plugin")
                .join("schemas")
                .join("request.json"),
        )
        .expect("request schema should be removed to force contract failure");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root],
            enabled_plugin_id: Some("broken-plugin".to_string()),
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-contract-failure-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-contract-failure-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        let err = state
            .refresh_plugins()
            .await
            .expect_err("enabled plugin contract probe failure should fail refresh");

        assert!(
            format!("{err:#}").contains("broken-plugin"),
            "error should identify the enabled plugin whose contract probe failed: {err:#}"
        );
    }

    #[tokio::test]
    async fn test_refresh_plugins_registers_only_enabled_plugin_id() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-enabled-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "demo-a");
        create_plugin_dir(&root, "demo-b");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root],
            enabled_plugin_id: Some("demo-b".to_string()),
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        let count = state
            .refresh_plugins()
            .await
            .expect("plugin refresh should succeed");

        let registry = state.workflow_registry.read().await;
        assert_eq!(count, 1);
        assert!(registry.contracts.contains_key("demo-b"));
        assert!(!registry.contracts.contains_key("demo-a"));
    }

    #[tokio::test]
    async fn test_refresh_plugins_fails_when_enabled_plugin_id_is_unknown() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-enabled-missing-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "demo-a");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root],
            enabled_plugin_id: Some("missing-plugin".to_string()),
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-missing-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-enabled-missing-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        let err = state
            .refresh_plugins()
            .await
            .expect_err("unknown enabled plugin id should fail refresh");

        assert!(
            format!("{err:#}").contains("missing-plugin"),
            "error should mention the configured plugin id: {err:#}"
        );
    }

    #[tokio::test]
    async fn test_refresh_plugins_replaces_removed_plugins() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-state-refresh-replace-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("root plugin dir should be created");
        create_plugin_dir(&root, "demo-a");

        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![root.clone()],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-replace-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-state-refresh-replace-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
        };

        let state = AppState::new_for_testing(config);
        state
            .refresh_plugins()
            .await
            .expect("initial plugin refresh should succeed");
        fs::remove_dir_all(root.join("demo-a")).expect("plugin directory should be removed");

        let count = state
            .refresh_plugins()
            .await
            .expect("second plugin refresh should succeed");

        assert_eq!(count, 0);
        assert!(state.workflow_registry.read().await.contracts.is_empty());
    }
}
