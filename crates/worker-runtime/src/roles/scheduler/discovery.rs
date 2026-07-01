/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, anyhow};
use nvml_wrapper::Nvml;
use scicomp_rq::{QueueManager, hash_ops};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::ffi::OsStr;
use tracing::{debug, warn};

/// Default memory utilization limit as an integer percentage.
pub const DEFAULT_MEMORY_UTILIZATION_PERCENT: u64 = 80;

/// Default GPU discovery interval in seconds.
pub const DEFAULT_GPU_DISCOVERY_INTERVAL_SECS: u64 = 60;

const WORKER_STATUS_AVAILABLE: &str = "available";
const WORKER_STATUS_WARMING: &str = "warming";
pub(crate) const WARMUP_STATUS_SUCCEEDED: &str = "succeeded";
const WARMUP_STATUS_SKIPPED: &str = "skipped";
const WARMUP_STATUS_NOT_STARTED: &str = "not_started";

/// Runtime resource inventory used by scheduler reservation logic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceInfo {
    pub resource_id: u32,
    pub stream_name: String,
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    #[serde(default)]
    pub executor_class: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub model_cache_workflow_ids: Vec<String>,
    #[serde(default)]
    pub warmup_workflow_id: Option<String>,
    #[serde(default)]
    pub warmup_status: Option<String>,
}

impl ResourceInfo {
    /// Usable memory in MiB at the given utilization percentage (0–100).
    pub fn usable_memory_mb(&self, utilization_percent: u64) -> u64 {
        self.total_memory_mb * utilization_percent / 100
    }

    /// Available memory in MiB after subtracting currently used memory.
    pub fn available_memory_mb(&self, utilization_percent: u64) -> u64 {
        self.usable_memory_mb(utilization_percent)
            .saturating_sub(self.used_memory_mb)
    }
}

fn default_registry_device_kind() -> String {
    "gpu".to_string()
}

#[derive(Debug, Deserialize)]
struct RegistryWorkerMetadata {
    stream: String,
    device_index: u32,
    #[serde(default = "default_registry_device_kind")]
    device_kind: String,
    #[serde(default)]
    executor_class: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_worker_status")]
    status: String,
    #[serde(default)]
    model_cache: Option<RegistryModelCache>,
}

fn default_worker_status() -> String {
    WORKER_STATUS_AVAILABLE.to_string()
}

#[derive(Debug, Deserialize, Default)]
struct RegistryModelCache {
    #[serde(default)]
    entries: Vec<RegistryModelCacheEntry>,
    #[serde(default)]
    warmup: Option<RegistryModelCacheWarmup>,
}

#[derive(Debug, Deserialize)]
struct RegistryModelCacheEntry {
    #[serde(default)]
    workflow_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RegistryModelCacheWarmup {
    #[serde(default)]
    workflow_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

impl RegistryWorkerMetadata {
    fn is_discoverable_gpu(&self) -> bool {
        (self.status.eq_ignore_ascii_case(WORKER_STATUS_AVAILABLE)
            || self.status.eq_ignore_ascii_case(WORKER_STATUS_WARMING))
            && self.device_kind.eq_ignore_ascii_case("gpu")
    }

    fn warmup_status(&self) -> Option<String> {
        let cache_status = self
            .model_cache
            .as_ref()
            .and_then(|cache| cache.warmup.as_ref())
            .and_then(|warmup| normalize_non_empty(warmup.status.as_deref()));

        // A worker whose top-level status is "warming" is still loading regardless
        // of its cache warmup outcome. Return "warming" so callers treat it as a
        // temporary block rather than inspecting a potentially misleading cache status.
        if self.status.eq_ignore_ascii_case(WORKER_STATUS_WARMING)
            && cache_status
                .as_deref()
                .is_none_or(|status| warmup_status_allows_scheduling(Some(status)))
        {
            return Some(WORKER_STATUS_WARMING.to_string());
        }

        cache_status
    }

    fn cached_workflow_ids(&self) -> Vec<String> {
        let mut workflow_ids = Vec::new();
        let Some(cache) = self.model_cache.as_ref() else {
            return workflow_ids;
        };

        for entry in &cache.entries {
            if let Some(workflow_id) = normalize_non_empty(entry.workflow_id.as_deref())
                && !workflow_ids.contains(&workflow_id)
            {
                workflow_ids.push(workflow_id);
            }
        }

        if let Some(warmup) = cache.warmup.as_ref()
            && warmup
                .status
                .as_deref()
                .map(str::trim)
                .filter(|status| !status.is_empty())
                .is_some_and(|status| status.eq_ignore_ascii_case(WARMUP_STATUS_SUCCEEDED))
            && let Some(workflow_id) = normalize_non_empty(warmup.workflow_id.as_deref())
            && !workflow_ids.contains(&workflow_id)
        {
            workflow_ids.push(workflow_id);
        }

        workflow_ids
    }

    fn warmup_workflow_id(&self) -> Option<String> {
        self.model_cache
            .as_ref()
            .and_then(|cache| cache.warmup.as_ref())
            .and_then(|warmup| normalize_non_empty(warmup.workflow_id.as_deref()))
    }
}

fn normalize_non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub(crate) fn warmup_status_allows_scheduling(status: Option<&str>) -> bool {
    match status.map(str::trim).filter(|value| !value.is_empty()) {
        None => true,
        Some(status) if status.eq_ignore_ascii_case(WARMUP_STATUS_SUCCEEDED) => true,
        Some(status) if status.eq_ignore_ascii_case(WARMUP_STATUS_SKIPPED) => true,
        Some(_) => false,
    }
}

pub(crate) fn warmup_status_is_loading(status: Option<&str>) -> bool {
    matches!(
        status.map(str::trim).filter(|value| !value.is_empty()),
        Some(status)
            if status.eq_ignore_ascii_case(WARMUP_STATUS_NOT_STARTED)
                || status.eq_ignore_ascii_case(WORKER_STATUS_WARMING)
    )
}

/// Discovery result for the reservation table cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryUpdate {
    /// Replace the tracked GPU inventory with this authoritative snapshot.
    Authoritative(Vec<ResourceInfo>),
    /// Preserve the previous snapshot because local discovery was stale/unavailable.
    Stale,
}

/// Build a logical GPU stream name from pod identity.
pub fn build_gpu_stream_name(gpu_id: u32) -> String {
    let namespace = std::env::var("POD_NAMESPACE")
        .or_else(|_| std::env::var("NAMESPACE"))
        .unwrap_or_else(|_| "default".to_string());
    let pod_name = std::env::var("POD_NAME").unwrap_or_else(|_| "unknown".to_string());
    format!("gpu:{namespace}:{pod_name}:{gpu_id}")
}

fn nvml_library_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    if let Ok(raw_paths) = std::env::var("SCHEDULER_NVML_LIB_PATHS")
        && !raw_paths.trim().is_empty()
    {
        for candidate in raw_paths
            .split(':')
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if seen.insert(candidate.to_string()) {
                candidates.push(candidate.to_string());
            }
        }
    }

    for default in ["libnvidia-ml.so", "libnvidia-ml.so.1"] {
        if seen.insert(default.to_string()) {
            candidates.push(default.to_string());
        }
    }

    candidates
}

fn init_with_candidates<T, F>(candidates: &[String], mut init: F) -> Result<T>
where
    F: FnMut(&str) -> Result<T>,
{
    if candidates.is_empty() {
        return Err(anyhow!("no NVML library candidates configured"));
    }

    let mut errors = Vec::new();
    for candidate in candidates {
        match init(candidate.as_str()) {
            Ok(value) => return Ok(value),
            Err(error) => errors.push(format!("{candidate}: {error}")),
        }
    }

    Err(anyhow!(
        "failed to initialize NVML from any candidate ({})",
        errors.join("; ")
    ))
}

/// Discover local GPUs from NVML or the JSON test override.
async fn discover_local_gpus() -> Result<Vec<ResourceInfo>> {
    if let Ok(raw_json) = std::env::var("SCHEDULER_DISCOVERY_JSON")
        && !raw_json.trim().is_empty()
    {
        let mut gpus: Vec<ResourceInfo> =
            serde_json::from_str(&raw_json).context("failed to parse SCHEDULER_DISCOVERY_JSON")?;
        gpus.sort_by_key(|g| g.resource_id);
        return Ok(gpus);
    }

    let candidates = nvml_library_candidates();
    let nvml = init_with_candidates(&candidates, |lib_path| {
        let mut builder = Nvml::builder();
        builder.lib_path(OsStr::new(lib_path));
        builder
            .init()
            .map_err(|e| anyhow!("failed to initialize NVML from '{lib_path}': {e}"))
    })?;
    let device_count = nvml
        .device_count()
        .map_err(|e| anyhow!("failed to query device count: {e}"))?;

    let mut gpus = Vec::new();
    for index in 0..device_count {
        let device = nvml
            .device_by_index(index)
            .map_err(|e| anyhow!("failed to query device {index}: {e}"))?;
        let mem_info = device
            .memory_info()
            .map_err(|e| anyhow!("failed to query memory for device {index}: {e}"))?;

        let total_mb = mem_info.total / (1024 * 1024);
        let used_mb = mem_info.used / (1024 * 1024);
        gpus.push(ResourceInfo {
            resource_id: index,
            stream_name: build_gpu_stream_name(index),
            total_memory_mb: total_mb,
            used_memory_mb: used_mb,
            executor_class: None,
            tags: Vec::new(),
            model_cache_workflow_ids: Vec::new(),
            warmup_workflow_id: None,
            warmup_status: None,
        });
    }
    Ok(gpus)
}

/// Discover schedulable resources by joining registry workers with local NVML GPUs.
pub async fn discover_resources(qm: &QueueManager, registry_key: &str) -> Result<DiscoveryUpdate> {
    debug!(registry_key = %registry_key, "starting scheduler resource discovery");
    let registry_workers = discover_registry_workers(qm, registry_key).await?;
    if registry_workers.is_empty() {
        debug!(
            registry_key = %registry_key,
            "discovery found no GPU workers in registry; reporting empty GPU inventory"
        );
        return Ok(DiscoveryUpdate::Authoritative(Vec::new()));
    }

    match discover_local_gpus().await {
        Ok(local_gpus) => {
            debug!(
                registry_key = %registry_key,
                local_gpu_count = local_gpus.len(),
                registry_gpu_worker_count = registry_workers.len(),
                "collected scheduler discovery inputs"
            );
            let mut gpu_workers = find_schedulable_gpus(local_gpus, registry_workers);
            gpu_workers.sort_by(|left, right| {
                left.resource_id
                    .cmp(&right.resource_id)
                    .then_with(|| left.stream_name.cmp(&right.stream_name))
            });
            debug!(
                registry_key = %registry_key,
                gpu_worker_count = gpu_workers.len(),
                "scheduler resource discovery completed"
            );
            Ok(DiscoveryUpdate::Authoritative(gpu_workers))
        }
        Err(error) => {
            debug!(
                registry_key = %registry_key,
                registry_gpu_worker_count = registry_workers.len(),
                local_gpu_error = %error,
                "local GPU discovery failed after registry lookup"
            );
            warn!(
                error = %error,
                "failed to discover local GPUs via NVML; keeping previous GPU inventory"
            );
            Ok(DiscoveryUpdate::Stale)
        }
    }
}

async fn discover_registry_workers(
    qm: &QueueManager,
    registry_key: &str,
) -> Result<Vec<RegistryWorkerMetadata>> {
    let mut conn = qm.connection();
    let entries = hash_ops::hgetall(&mut conn, registry_key)
        .await
        .with_context(|| format!("failed to read worker registry hash '{registry_key}'"))?;

    let mut workers = Vec::new();
    for (field, raw_json) in entries {
        let metadata: RegistryWorkerMetadata = serde_json::from_str(&raw_json)
            .with_context(|| format!("failed to parse worker registry entry '{field}' as JSON"))?;
        if metadata.is_discoverable_gpu() {
            workers.push(metadata);
        }
    }
    workers.sort_by(|left, right| left.stream.cmp(&right.stream));
    debug!(
        registry_key = %registry_key,
        discoverable_gpu_worker_count = workers.len(),
        "loaded discoverable GPU workers from scheduler registry"
    );
    Ok(workers)
}

fn find_schedulable_gpus(
    local_gpus: Vec<ResourceInfo>,
    registry_workers: Vec<RegistryWorkerMetadata>,
) -> Vec<ResourceInfo> {
    let mut registry_workers_by_device: HashMap<u32, RegistryWorkerMetadata> = HashMap::new();
    for worker in registry_workers {
        match registry_workers_by_device.entry(worker.device_index) {
            Entry::Vacant(entry) => {
                entry.insert(worker);
            }
            Entry::Occupied(existing) => {
                warn!(
                    resource_id = worker.device_index,
                    existing_stream = %existing.get().stream,
                    duplicate_stream = %worker.stream,
                    existing_executor_class = ?existing.get().executor_class,
                    duplicate_executor_class = ?worker.executor_class,
                    "duplicate GPU registry worker discovered for device_index; keeping first entry"
                );
            }
        }
    }

    let mut discovered = Vec::new();
    let mut seen_local_gpu_ids = HashSet::new();
    for local_gpu in local_gpus {
        if !seen_local_gpu_ids.insert(local_gpu.resource_id) {
            warn!(
                resource_id = local_gpu.resource_id,
                stream = %local_gpu.stream_name,
                "duplicate local GPU id discovered; skipping duplicate entry"
            );
            continue;
        }

        let Some(worker) = registry_workers_by_device.get(&local_gpu.resource_id) else {
            debug!(
                resource_id = local_gpu.resource_id,
                local_stream = %local_gpu.stream_name,
                "skipping local GPU because no matching registry worker was found"
            );
            continue;
        };

        discovered.push(ResourceInfo {
            resource_id: worker.device_index,
            stream_name: worker.stream.clone(),
            total_memory_mb: local_gpu.total_memory_mb,
            used_memory_mb: local_gpu.used_memory_mb,
            executor_class: worker.executor_class.clone(),
            tags: worker.tags.clone(),
            model_cache_workflow_ids: worker.cached_workflow_ids(),
            warmup_workflow_id: worker.warmup_workflow_id(),
            warmup_status: worker.warmup_status(),
        });
    }

    discovered
}

#[cfg(test)]
fn parse_gpu_registry_entries(
    entries: &std::collections::HashMap<String, String>,
) -> Result<Vec<RegistryWorkerMetadata>> {
    let mut workers = Vec::new();
    for (field, raw_json) in entries {
        let metadata: RegistryWorkerMetadata = serde_json::from_str(raw_json)
            .with_context(|| format!("failed to parse worker registry entry '{field}' as JSON"))?;
        if metadata.is_discoverable_gpu() {
            workers.push(metadata);
        }
    }
    Ok(workers)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::test_env::{with_env_var, with_env_var_unguarded};

    #[test]
    fn stream_name_uses_namespace_and_pod_defaults() {
        let stream = with_env_var("NAMESPACE", None, || {
            with_env_var_unguarded("POD_NAME", None, || build_gpu_stream_name(2))
        });
        assert_eq!(stream, "gpu:default:unknown:2");
    }

    #[test]
    fn stream_name_uses_namespace_and_pod_from_env() {
        let stream = with_env_var("NAMESPACE", Some("ns-a"), || {
            with_env_var_unguarded("POD_NAME", Some("pod-b"), || build_gpu_stream_name(3))
        });
        assert_eq!(stream, "gpu:ns-a:pod-b:3");
    }

    #[test]
    fn stream_name_prefers_pod_namespace_over_namespace() {
        let stream = with_env_var("POD_NAMESPACE", Some("ns-pod"), || {
            with_env_var_unguarded("NAMESPACE", Some("ns-legacy"), || {
                with_env_var_unguarded("POD_NAME", Some("pod-b"), || build_gpu_stream_name(4))
            })
        });
        assert_eq!(stream, "gpu:ns-pod:pod-b:4");
    }

    #[test]
    fn discover_gpus_parses_json_override() {
        let json = r#"[{"resource_id":0,"stream_name":"gpu:test:0","total_memory_mb":10000,"used_memory_mb":512}]"#;
        let gpus = with_env_var("SCHEDULER_DISCOVERY_JSON", Some(json), || {
            tokio::runtime::Runtime::new()
                .expect("runtime should construct")
                .block_on(async { discover_local_gpus().await })
        })
        .expect("discovery override should parse");

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].resource_id, 0);
        assert_eq!(gpus[0].stream_name, "gpu:test:0");
        assert_eq!(gpus[0].total_memory_mb, 10_000);
        assert_eq!(gpus[0].used_memory_mb, 512);
        assert_eq!(gpus[0].executor_class, None);
        assert!(gpus[0].tags.is_empty());
    }

    #[test]
    fn discover_gpus_parses_capability_fields_from_json_override() {
        let json = r#"[{"resource_id":1,"stream_name":"gpu:test:1","total_memory_mb":12000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo","hopper"]}]"#;
        let gpus = with_env_var("SCHEDULER_DISCOVERY_JSON", Some(json), || {
            tokio::runtime::Runtime::new()
                .expect("runtime should construct")
                .block_on(async { discover_local_gpus().await })
        })
        .expect("discovery override should parse");

        assert_eq!(gpus.len(), 1);
        assert_eq!(
            gpus[0].executor_class.as_deref(),
            Some("python.gpu.physicsnemo")
        );
        assert_eq!(
            gpus[0].tags,
            vec!["physicsnemo".to_string(), "hopper".to_string()]
        );
    }

    // --- PR-027: configurable memory utilization limit ---

    #[test]
    fn usable_memory_mb_respects_custom_limit() {
        let gpu = ResourceInfo {
            resource_id: 0,
            stream_name: "gpu:0".to_string(),
            total_memory_mb: 10_000,
            used_memory_mb: 0,
            executor_class: None,
            tags: Vec::new(),
            model_cache_workflow_ids: Vec::new(),
            warmup_workflow_id: None,
            warmup_status: None,
        };
        assert_eq!(gpu.usable_memory_mb(50), 5_000);
        assert_eq!(gpu.usable_memory_mb(90), 9_000);
        assert_eq!(
            gpu.usable_memory_mb(DEFAULT_MEMORY_UTILIZATION_PERCENT),
            8_000
        );
    }

    #[test]
    fn available_memory_mb_subtracts_used() {
        let gpu = ResourceInfo {
            resource_id: 0,
            stream_name: "gpu:0".to_string(),
            total_memory_mb: 10_000,
            used_memory_mb: 3_000,
            executor_class: None,
            tags: Vec::new(),
            model_cache_workflow_ids: Vec::new(),
            warmup_workflow_id: None,
            warmup_status: None,
        };
        assert_eq!(gpu.available_memory_mb(50), 2_000);
        assert_eq!(gpu.available_memory_mb(90), 6_000);
        assert_eq!(
            gpu.available_memory_mb(DEFAULT_MEMORY_UTILIZATION_PERCENT),
            5_000
        );
    }

    #[test]
    fn nvml_library_candidates_include_versioned_fallback() {
        let candidates = nvml_library_candidates();
        assert_eq!(
            candidates,
            vec![
                "libnvidia-ml.so".to_string(),
                "libnvidia-ml.so.1".to_string()
            ]
        );
    }

    #[test]
    fn init_with_candidates_retries_after_failure() {
        let candidates = vec![
            "libnvidia-ml.so".to_string(),
            "libnvidia-ml.so.1".to_string(),
        ];
        let mut attempts = Vec::new();
        let selected = init_with_candidates(&candidates, |candidate| {
            attempts.push(candidate.to_string());
            if candidate == "libnvidia-ml.so" {
                return Err(anyhow!("primary SONAME missing"));
            }
            Ok(candidate.to_string())
        })
        .expect("fallback candidate should be used");

        assert_eq!(selected, "libnvidia-ml.so.1".to_string());
        assert_eq!(
            attempts,
            vec![
                "libnvidia-ml.so".to_string(),
                "libnvidia-ml.so.1".to_string()
            ]
        );
    }

    #[test]
    fn parse_gpu_registry_entries_extracts_worker_capabilities() {
        let entries = HashMap::from([(
            "gpu:pod-a:0:worker:0".to_string(),
            serde_json::json!({
                "stream": "gpu:pod-a:0",
                "device_index": 0,
                "executor_class": "python.gpu.physicsnemo",
                "tags": ["physicsnemo", "hopper"],
                "status": "available",
                "model_cache": {
                    "entries": [{
                        "workflow_id": "demo-plugin"
                    }],
                    "warmup": {
                        "workflow_id": "demo-plugin",
                        "status": "succeeded"
                    }
                }
            })
            .to_string(),
        )]);

        let parsed =
            parse_gpu_registry_entries(&entries).expect("registry payload should parse cleanly");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].stream, "gpu:pod-a:0");
        assert_eq!(parsed[0].device_index, 0);
        assert_eq!(
            parsed[0].executor_class.as_deref(),
            Some("python.gpu.physicsnemo")
        );
        assert_eq!(
            parsed[0].tags,
            vec!["physicsnemo".to_string(), "hopper".to_string()]
        );
        assert_eq!(
            parsed[0].cached_workflow_ids(),
            vec!["demo-plugin".to_string()]
        );
        assert_eq!(parsed[0].warmup_status().as_deref(), Some("succeeded"));
    }

    #[test]
    fn parse_gpu_registry_entries_skips_unavailable_workers() {
        let entries = HashMap::from([(
            "gpu:pod-a:0:worker:0".to_string(),
            serde_json::json!({
                "stream": "gpu:pod-a:0",
                "device_index": 0,
                "status": "busy"
            })
            .to_string(),
        )]);

        let parsed =
            parse_gpu_registry_entries(&entries).expect("busy workers should be filtered out");

        assert!(parsed.is_empty());
    }

    #[test]
    fn parse_gpu_registry_entries_keeps_workers_while_warmup_is_incomplete() {
        let entries = HashMap::from([(
            "gpu:pod-a:0:worker:0".to_string(),
            serde_json::json!({
                "stream": "gpu:pod-a:0",
                "device_index": 0,
                "status": "available",
                "model_cache": {
                    "warmup": {
                        "workflow_id": "demo-plugin",
                        "status": "warming"
                    }
                }
            })
            .to_string(),
        )]);

        let parsed =
            parse_gpu_registry_entries(&entries).expect("warming workers should parse cleanly");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].stream, "gpu:pod-a:0");
        assert_eq!(
            parsed[0].warmup_workflow_id().as_deref(),
            Some("demo-plugin")
        );
        assert_eq!(parsed[0].warmup_status().as_deref(), Some("warming"));
    }

    #[test]
    fn find_schedulable_gpus_uses_nvml_memory_for_schedulable_workers() {
        let merged = find_schedulable_gpus(
            vec![
                ResourceInfo {
                    resource_id: 0,
                    stream_name: "local-gpu-0".to_string(),
                    total_memory_mb: 80_000,
                    used_memory_mb: 12_000,
                    executor_class: None,
                    tags: Vec::new(),
                    model_cache_workflow_ids: Vec::new(),
                    warmup_workflow_id: None,
                    warmup_status: None,
                },
                ResourceInfo {
                    resource_id: 1,
                    stream_name: "local-gpu-1".to_string(),
                    total_memory_mb: 64_000,
                    used_memory_mb: 4_000,
                    executor_class: None,
                    tags: Vec::new(),
                    model_cache_workflow_ids: Vec::new(),
                    warmup_workflow_id: None,
                    warmup_status: None,
                },
            ],
            vec![
                RegistryWorkerMetadata {
                    stream: "execute.python.gpu.demo:gpu:local:0".to_string(),
                    device_index: 0,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: Some(RegistryModelCache {
                        entries: vec![RegistryModelCacheEntry {
                            workflow_id: Some("demo-plugin".to_string()),
                        }],
                        warmup: Some(RegistryModelCacheWarmup {
                            workflow_id: Some("demo-plugin".to_string()),
                            status: Some("succeeded".to_string()),
                        }),
                    }),
                },
                RegistryWorkerMetadata {
                    stream: "execute.python.gpu.demo:gpu:local:2".to_string(),
                    device_index: 2,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: None,
                },
            ],
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].resource_id, 0);
        assert_eq!(merged[0].stream_name, "execute.python.gpu.demo:gpu:local:0");
        assert_eq!(merged[0].total_memory_mb, 80_000);
        assert_eq!(merged[0].used_memory_mb, 12_000);
        assert_eq!(merged[0].executor_class.as_deref(), Some("python.gpu.demo"));
        assert_eq!(merged[0].tags, vec!["demo".to_string()]);
        assert_eq!(
            merged[0].model_cache_workflow_ids,
            vec!["demo-plugin".to_string()]
        );
        assert_eq!(merged[0].warmup_workflow_id.as_deref(), Some("demo-plugin"));
        assert_eq!(merged[0].warmup_status.as_deref(), Some("succeeded"));
    }

    #[test]
    fn find_schedulable_gpus_skips_registry_gpus_without_local_match() {
        let merged = find_schedulable_gpus(
            Vec::new(),
            vec![RegistryWorkerMetadata {
                stream: "execute.python.gpu.demo:gpu:local:0".to_string(),
                device_index: 0,
                device_kind: "gpu".to_string(),
                executor_class: Some("python.gpu.demo".to_string()),
                tags: vec!["demo".to_string()],
                status: "available".to_string(),
                model_cache: None,
            }],
        );
        assert!(merged.is_empty());
    }

    #[test]
    fn find_schedulable_gpus_collapses_duplicate_registry_workers_per_device() {
        let merged = find_schedulable_gpus(
            vec![ResourceInfo {
                resource_id: 0,
                stream_name: "gpu:pod-a:0".to_string(),
                total_memory_mb: 80_000,
                used_memory_mb: 12_000,
                executor_class: None,
                tags: Vec::new(),
                model_cache_workflow_ids: Vec::new(),
                warmup_workflow_id: None,
                warmup_status: None,
            }],
            vec![
                RegistryWorkerMetadata {
                    stream: "gpu:pod-a:0".to_string(),
                    device_index: 0,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: None,
                },
                RegistryWorkerMetadata {
                    stream: "gpu:pod-b:0".to_string(),
                    device_index: 0,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: None,
                },
            ],
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].stream_name, "gpu:pod-a:0");
        assert_eq!(merged[0].total_memory_mb, 80_000);
        assert_eq!(merged[0].used_memory_mb, 12_000);
    }

    #[test]
    fn find_schedulable_gpus_skips_duplicate_local_gpu_ids() {
        let merged = find_schedulable_gpus(
            vec![
                ResourceInfo {
                    resource_id: 0,
                    stream_name: "gpu:pod-a:0".to_string(),
                    total_memory_mb: 80_000,
                    used_memory_mb: 12_000,
                    executor_class: None,
                    tags: Vec::new(),
                    model_cache_workflow_ids: Vec::new(),
                    warmup_workflow_id: None,
                    warmup_status: None,
                },
                ResourceInfo {
                    resource_id: 0,
                    stream_name: "gpu:pod-b:0".to_string(),
                    total_memory_mb: 80_000,
                    used_memory_mb: 12_000,
                    executor_class: None,
                    tags: Vec::new(),
                    model_cache_workflow_ids: Vec::new(),
                    warmup_workflow_id: None,
                    warmup_status: None,
                },
            ],
            vec![
                RegistryWorkerMetadata {
                    stream: "gpu:pod-a:0".to_string(),
                    device_index: 0,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: None,
                },
                RegistryWorkerMetadata {
                    stream: "gpu:pod-b:0".to_string(),
                    device_index: 0,
                    device_kind: "gpu".to_string(),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: vec!["demo".to_string()],
                    status: "available".to_string(),
                    model_cache: None,
                },
            ],
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].stream_name, "gpu:pod-a:0");
    }

    #[test]
    fn parse_gpu_registry_entries_skips_non_gpu_workers() {
        let cpu_workers = parse_gpu_registry_entries(&HashMap::from([(
            "execute.python.cpu.demo".to_string(),
            serde_json::json!({
                "stream": "execute.python.cpu.demo",
                "device_index": 0,
                "device_kind": "cpu",
                "executor_class": "python.cpu.demo",
                "tags": ["demo", "cpu"],
                "status": "available"
            })
            .to_string(),
        )]))
        .expect("cpu registry payload should parse cleanly");

        assert!(cpu_workers.is_empty());
    }
}
