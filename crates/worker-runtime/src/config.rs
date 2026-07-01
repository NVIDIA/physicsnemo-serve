/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::roles::scheduler::{
    DEFAULT_GPU_DISCOVERY_INTERVAL_SECS, DEFAULT_MEMORY_UTILIZATION_PERCENT, SchedulingStrategy,
};
use crate::traits::RoleEnv;

/// Environment variable for the shared pipeline config path.
/// When set, `--config-path` CLI arg is optional and this takes precedence.
pub const ENV_PIPELINE_CONFIG: &str = "WORKER_PIPELINE_CONFIG";
/// Environment variable for selecting active role from pipeline config.
pub const ENV_WORKER_ROLE: &str = "WORKER_ROLE";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub stream_prefix: String,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_shared_dlq_stream")]
    pub shared_dlq_stream: String,
    #[serde(default)]
    pub python_runtime_envs: BTreeMap<String, PythonRuntimeEnvConfig>,
    pub streams: Vec<String>,
    pub roles: BTreeMap<String, RoleConfig>,
}

fn default_max_retries() -> usize {
    5
}

fn default_shared_dlq_stream() -> String {
    "dlq".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleConfig {
    pub inputs: Vec<InputStreamSpec>,
    pub outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonRuntimeEnvConfig {
    pub python_executable: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputStreamSpec {
    pub stream: String,
    pub max_dequeue_items: usize,
    pub poll_interval_ms: u64,
    pub block_ms: u64,
    #[serde(default = "default_reclaim_idle_ms")]
    pub reclaim_idle_ms: u64,
}

fn default_reclaim_idle_ms() -> u64 {
    60_000
}

// ---------------------------------------------------------------------------
// Typed role-specific configs
// ---------------------------------------------------------------------------

/// Scheduler-specific configuration parsed from the `config` block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SchedulerRoleConfig {
    /// Redis key for the GPU registry hash (default: `gpu:registry`).
    #[serde(default = "default_gpu_registry_key")]
    pub gpu_registry_key: String,
    /// Scheduling strategy (default: `round_robin`).
    #[serde(default = "default_scheduling_strategy")]
    pub scheduling_strategy: SchedulingStrategy,
    /// Memory utilization cap as a percentage of total GPU memory (default: 80).
    #[serde(default = "default_memory_utilization_percent")]
    pub memory_utilization_percent: u64,
    /// GPU discovery refresh interval in seconds (default: 60).
    #[serde(default = "default_resource_discovery_interval_secs")]
    pub gpu_discovery_interval_secs: u64,
}

fn default_gpu_registry_key() -> String {
    "gpu:registry".to_string()
}

fn default_scheduling_strategy() -> SchedulingStrategy {
    SchedulingStrategy::RoundRobin
}

fn default_memory_utilization_percent() -> u64 {
    DEFAULT_MEMORY_UTILIZATION_PERCENT
}

fn default_resource_discovery_interval_secs() -> u64 {
    DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
}

impl Default for SchedulerRoleConfig {
    fn default() -> Self {
        Self {
            gpu_registry_key: default_gpu_registry_key(),
            scheduling_strategy: default_scheduling_strategy(),
            memory_utilization_percent: default_memory_utilization_percent(),
            gpu_discovery_interval_secs: default_resource_discovery_interval_secs(),
        }
    }
}

/// Prefetch-specific configuration parsed from the `config` block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PrefetchRoleConfig {
    /// Logical stream name to hand off prefetched messages (default: `schedule`).
    #[serde(default = "default_handoff_stream")]
    pub handoff_stream: String,
    /// When true, prefetch fails closed on metadata/plan generation errors and
    /// partial download failures (`prefetch_errors > 0`).
    /// When false (default), prefetch forwards degraded payloads for these cases.
    #[serde(default = "default_fail_on_plan_generation_error")]
    pub fail_on_plan_generation_error: bool,
}

fn default_handoff_stream() -> String {
    "schedule".to_string()
}

fn default_fail_on_plan_generation_error() -> bool {
    false
}

impl Default for PrefetchRoleConfig {
    fn default() -> Self {
        Self {
            handoff_stream: default_handoff_stream(),
            fail_on_plan_generation_error: default_fail_on_plan_generation_error(),
        }
    }
}

/// Prepare-specific configuration parsed from the `config` block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PrepareRoleConfig {
    /// Python executable used to invoke the plugin hook runner.
    #[serde(default = "default_prepare_python_executable")]
    pub python_executable: String,
    /// Absolute or working-directory-relative path to the hook runner script.
    #[serde(default = "default_prepare_runner_path")]
    pub runner_path: String,
    /// Timeout in seconds for a single prepare hook invocation.
    #[serde(default = "default_prepare_hook_timeout_secs")]
    pub hook_timeout_secs: u64,
}

fn default_prepare_python_executable() -> String {
    "python3".to_string()
}

fn default_prepare_runner_path() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/plugin_hook_runner.py")
        .display()
        .to_string()
}

fn default_prepare_hook_timeout_secs() -> u64 {
    60
}

impl Default for PrepareRoleConfig {
    fn default() -> Self {
        Self {
            python_executable: default_prepare_python_executable(),
            runner_path: default_prepare_runner_path(),
            hook_timeout_secs: default_prepare_hook_timeout_secs(),
        }
    }
}

/// Results-specific configuration parsed from the `config` block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ResultsRoleConfig {
    /// TTL (seconds) for `result:{run_id}` JSON payload in Redis.
    #[serde(default = "default_result_ttl_seconds")]
    pub result_ttl_seconds: u64,
}

fn default_result_ttl_seconds() -> u64 {
    86_400
}

impl Default for ResultsRoleConfig {
    fn default() -> Self {
        Self {
            result_ttl_seconds: default_result_ttl_seconds(),
        }
    }
}

/// Parse a typed role config from the raw `serde_json::Value` config block.
///
/// Returns `Ok(T::default())` when `raw` is `None` (missing block uses defaults).
pub fn parse_role_config<T>(raw: Option<&serde_json::Value>) -> Result<T>
where
    T: serde::de::DeserializeOwned + Default,
{
    match raw {
        Some(value) => {
            serde_json::from_value(value.clone()).context("failed to parse role config block")
        }
        None => Ok(T::default()),
    }
}

impl RuntimeConfig {
    /// Validate the config structure.
    ///
    /// # Rules
    /// - `streams` must be non-empty.
    /// - `roles` must be non-empty.
    /// - Each role must have at least one input.
    /// - Each input stream must exist in stream inventory.
    /// - `outputs` may be empty (terminal consumers are valid).
    /// - Exact output selectors must exist in stream inventory.
    /// - Prefix selectors (`name.*`) expand against stream inventory.
    /// - `max_dequeue_items`, `poll_interval_ms`, `block_ms` must be > 0.
    pub fn validate(&self) -> Result<()> {
        if self.max_retries == 0 {
            return Err(anyhow!("config max_retries must be > 0"));
        }
        if self.shared_dlq_stream.trim().is_empty() {
            return Err(anyhow!("config shared_dlq_stream must be non-empty"));
        }
        if self.streams.is_empty() {
            return Err(anyhow!(
                "config must define at least one stream in inventory"
            ));
        }
        let mut seen_streams = std::collections::HashSet::new();
        for stream in &self.streams {
            if !seen_streams.insert(stream.as_str()) {
                return Err(anyhow!("duplicate stream '{}' in inventory", stream));
            }
            if stream.ends_with(":grp") {
                return Err(anyhow!(
                    "stream '{}' ends with ':grp' — this suffix is reserved for \
                     consumer group names and will collide with RedisTransport",
                    stream
                ));
            }
        }
        if self.roles.is_empty() {
            return Err(anyhow!("config must define at least one role"));
        }

        for (executor_class, runtime_env) in &self.python_runtime_envs {
            if executor_class.trim().is_empty() {
                return Err(anyhow!(
                    "python_runtime_envs keys must be non-empty executor_class values"
                ));
            }
            if runtime_env.python_executable.trim().is_empty() {
                return Err(anyhow!(
                    "python_runtime_envs['{executor_class}'].python_executable must be non-empty"
                ));
            }
        }

        let stream_inventory: std::collections::HashSet<&str> =
            self.streams.iter().map(|s| s.as_str()).collect();

        for (role_name, role_cfg) in &self.roles {
            if role_cfg.inputs.is_empty() {
                return Err(anyhow!(
                    "role '{role_name}' must define at least one input stream"
                ));
            }

            for input in &role_cfg.inputs {
                if input.stream.trim().is_empty() {
                    return Err(anyhow!(
                        "role '{role_name}' has an input with empty stream name"
                    ));
                }
                if !stream_inventory.contains(input.stream.as_str()) {
                    return Err(anyhow!(
                        "role '{role_name}' input '{}' not found in stream inventory",
                        input.stream
                    ));
                }
                if input.max_dequeue_items == 0 {
                    return Err(anyhow!(
                        "role '{role_name}' input '{}' max_dequeue_items must be > 0",
                        input.stream
                    ));
                }
                if input.poll_interval_ms == 0 {
                    return Err(anyhow!(
                        "role '{role_name}' input '{}' poll_interval_ms must be > 0",
                        input.stream
                    ));
                }
                if input.block_ms == 0 {
                    return Err(anyhow!(
                        "role '{role_name}' input '{}' block_ms must be > 0",
                        input.stream
                    ));
                }
                if input.reclaim_idle_ms == 0 {
                    return Err(anyhow!(
                        "role '{role_name}' input '{}' reclaim_idle_ms must be > 0",
                        input.stream
                    ));
                }
            }

            for output in &role_cfg.outputs {
                if output.trim().is_empty() {
                    return Err(anyhow!(
                        "role '{role_name}' has an output with empty selector"
                    ));
                }
                if output.ends_with(".*") {
                    let prefix = &output[..output.len() - 2];
                    let has_match = stream_inventory
                        .iter()
                        .any(|s| s.starts_with(prefix) && s.len() > prefix.len());
                    if !has_match {
                        return Err(anyhow!(
                            "role '{role_name}' prefix selector '{}' matches no streams in inventory",
                            output
                        ));
                    }
                } else if !stream_inventory.contains(output.as_str()) {
                    return Err(anyhow!(
                        "role '{role_name}' output '{}' not found in stream inventory",
                        output
                    ));
                }
            }
        }

        Ok(())
    }

    /// Look up a role config by name.
    pub fn resolve_role(&self, role_name: &str) -> Result<&RoleConfig> {
        self.roles
            .get(role_name)
            .ok_or_else(|| anyhow!("role '{}' not found in config", role_name))
    }

    /// Expand output selectors against the stream inventory for a given role.
    /// Exact selectors are kept as-is. Prefix selectors (`gpu.*`) expand to
    /// all matching stream names.
    pub fn expand_outputs(&self, role_name: &str) -> Result<Vec<String>> {
        let role_cfg = self.resolve_role(role_name)?;
        let mut expanded = Vec::new();

        for output in &role_cfg.outputs {
            if output.ends_with(".*") {
                let prefix = &output[..output.len() - 2];
                for stream in &self.streams {
                    if stream.starts_with(prefix) && stream.len() > prefix.len() {
                        expanded.push(stream.clone());
                    }
                }
            } else {
                expanded.push(output.clone());
            }
        }

        Ok(expanded)
    }

    /// Build a fully-resolved `RoleEnv` for the given role name.
    pub fn resolve_env(&self, role_name: &str) -> Result<RoleEnv> {
        let role_cfg = self.resolve_role(role_name)?;
        let resolved_outputs = self.expand_outputs(role_name)?;

        Ok(RoleEnv {
            role_name: role_name.to_string(),
            stream_prefix: self.stream_prefix.clone(),
            inputs: role_cfg.inputs.clone(),
            resolved_outputs,
            role_config: role_cfg.config.clone(),
            python_runtime_envs: self.python_runtime_envs.clone(),
        })
    }
}

/// Load and parse a RuntimeConfig from a JSON file.
pub fn load_config_from_file(path: &Path) -> Result<RuntimeConfig> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("failed to read config file '{}': {}", path.display(), e))?;
    let config: RuntimeConfig = serde_json::from_str(&contents)
        .map_err(|e| anyhow!("failed to parse config file '{}': {}", path.display(), e))?;
    Ok(config)
}

/// Resolve a RuntimeConfig from multiple sources in priority order:
///
/// 1. Explicit file path (from `--config-path` CLI)
/// 2. `WORKER_PIPELINE_CONFIG` env var
///
/// Returns `(config, role_name)` where `role_name` comes from:
/// 1. Explicit `--role` CLI arg (if provided)
/// 2. `WORKER_ROLE` env var
pub fn resolve_config_and_role(
    cli_config_path: Option<&Path>,
    cli_role: Option<&str>,
) -> Result<(RuntimeConfig, String)> {
    let config = if let Some(path) = cli_config_path {
        load_config_from_file(path)?
    } else if let Ok(env_path) = std::env::var(ENV_PIPELINE_CONFIG) {
        load_config_from_file(Path::new(&env_path))?
    } else {
        return Err(anyhow!(
            "no config source: provide --config-path or set {ENV_PIPELINE_CONFIG}"
        ));
    };

    let role = if let Some(r) = cli_role {
        r.to_string()
    } else if let Ok(r) = std::env::var(ENV_WORKER_ROLE) {
        r
    } else {
        return Err(anyhow!(
            "no role specified: provide --role or set {ENV_WORKER_ROLE}"
        ));
    };

    Ok((config, role))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> RuntimeConfig {
        serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["prefetch", "schedule", "results"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 4,
                                "poll_interval_ms": 50, "block_ms": 500}],
                    "outputs": ["schedule"]
                }
            }
        }))
        .expect("minimal config should parse")
    }

    fn config_with_terminal_consumer() -> RuntimeConfig {
        serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["results"],
            "roles": {
                "results": {
                    "inputs": [{"stream": "results", "max_dequeue_items": 8,
                                "poll_interval_ms": 100, "block_ms": 1000}],
                    "outputs": []
                }
            }
        }))
        .expect("terminal consumer config should parse")
    }

    fn config_with_prefix_selector() -> RuntimeConfig {
        serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["schedule", "gpu_0", "gpu_1", "results"],
            "roles": {
                "scheduler": {
                    "inputs": [{"stream": "schedule", "max_dequeue_items": 4,
                                "poll_interval_ms": 50, "block_ms": 500}],
                    "outputs": ["gpu.*"]
                }
            }
        }))
        .expect("prefix selector config should parse")
    }

    // --- Validation Tests ---

    #[test]
    fn validate_accepts_minimal_valid_config() {
        let cfg = minimal_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn runtime_config_defaults_dlq_policy_when_omitted() {
        let cfg = minimal_config();
        let serialized = serde_json::to_value(&cfg).expect("config should serialize");
        assert_eq!(
            serialized.get("max_retries").and_then(|v| v.as_u64()),
            Some(5),
            "missing max_retries should default to 5"
        );
        assert_eq!(
            serialized.get("shared_dlq_stream").and_then(|v| v.as_str()),
            Some("dlq"),
            "missing shared_dlq_stream should default to 'dlq'"
        );
    }

    #[test]
    fn validate_accepts_terminal_consumer_with_empty_outputs() {
        let cfg = config_with_terminal_consumer();
        assert!(
            cfg.validate().is_ok(),
            "outputs: [] must be valid for terminal consumers"
        );
    }

    #[test]
    fn validate_rejects_empty_stream_inventory() {
        let mut cfg = minimal_config();
        cfg.streams.clear();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("at least one stream"),
            "expected stream inventory error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_roles() {
        let mut cfg = minimal_config();
        cfg.roles.clear();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("at least one role"),
            "expected roles error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_role_with_no_inputs() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["results"],
            "roles": {
                "results": {
                    "inputs": [],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("at least one input"),
            "expected input error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_input_not_in_inventory() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["results"],
            "roles": {
                "results": {
                    "inputs": [{"stream": "nonexistent", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("not found in stream inventory"),
            "expected inventory error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_zero_max_dequeue_items() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["prefetch"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 0,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("max_dequeue_items must be > 0"),
            "expected max_dequeue error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_zero_reclaim_idle_ms() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["prefetch"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100, "reclaim_idle_ms": 0}],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("reclaim_idle_ms must be > 0"),
            "expected reclaim_idle_ms error, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_output_not_in_inventory() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["prefetch"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": ["nonexistent"]
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("not found in stream inventory"),
            "expected output inventory error, got: {err}"
        );
    }

    #[test]
    fn validate_accepts_prefix_selector_with_matches() {
        let cfg = config_with_prefix_selector();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_prefix_selector_with_no_matches() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["schedule", "results"],
            "roles": {
                "scheduler": {
                    "inputs": [{"stream": "schedule", "max_dequeue_items": 4,
                                "poll_interval_ms": 50, "block_ms": 500}],
                    "outputs": ["gpu.*"]
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("matches no streams"),
            "expected prefix selector error, got: {err}"
        );
    }

    // --- Output Expansion Tests ---

    #[test]
    fn expand_outputs_exact_selector() {
        let cfg = minimal_config();
        let expanded = cfg.expand_outputs("prefetch").unwrap();
        assert_eq!(expanded, vec!["schedule"]);
    }

    #[test]
    fn expand_outputs_prefix_selector() {
        let cfg = config_with_prefix_selector();
        let expanded = cfg.expand_outputs("scheduler").unwrap();
        assert_eq!(expanded, vec!["gpu_0", "gpu_1"]);
    }

    #[test]
    fn expand_outputs_empty_for_terminal_consumer() {
        let cfg = config_with_terminal_consumer();
        let expanded = cfg.expand_outputs("results").unwrap();
        assert!(expanded.is_empty());
    }

    // --- RoleEnv Resolution Tests ---

    #[test]
    fn resolve_env_populates_all_fields() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "physicsnemo:",
            "streams": ["prefetch", "schedule"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 4,
                                "poll_interval_ms": 50, "block_ms": 500}],
                    "outputs": ["schedule"],
                    "config": {"handoff_stage": "schedule"}
                }
            }
        }))
        .unwrap();

        let env = cfg.resolve_env("prefetch").unwrap();
        assert_eq!(env.role_name, "prefetch");
        assert_eq!(env.stream_prefix, "physicsnemo:");
        assert_eq!(env.inputs.len(), 1);
        assert_eq!(env.inputs[0].stream, "prefetch");
        assert_eq!(env.resolved_outputs, vec!["schedule"]);
        assert!(env.role_config.is_some());
    }

    #[test]
    fn resolve_env_returns_error_for_unknown_role() {
        let cfg = minimal_config();
        let err = cfg.resolve_env("nonexistent").unwrap_err();
        assert!(
            err.to_string().contains("not found in config"),
            "expected unknown role error, got: {err}"
        );
    }

    #[test]
    fn resolve_env_role_config_is_none_when_omitted() {
        let cfg = minimal_config();
        let env = cfg.resolve_env("prefetch").unwrap();
        assert!(
            env.role_config.is_none(),
            "role_config should be None when not specified"
        );
    }

    // --- Typed Role Config Parsing Tests ---

    #[test]
    fn scheduler_role_config_parses_from_json_value() {
        let value = serde_json::json!({
            "gpu_registry_key": "test:gpu:registry",
            "scheduling_strategy": "round_robin"
        });
        let cfg: SchedulerRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.gpu_registry_key, "test:gpu:registry");
        assert_eq!(cfg.scheduling_strategy, SchedulingStrategy::RoundRobin);
    }

    #[test]
    fn scheduler_role_config_defaults_when_none() {
        let cfg: SchedulerRoleConfig = parse_role_config(None).unwrap();
        assert_eq!(cfg.gpu_registry_key, "gpu:registry");
        assert_eq!(cfg.scheduling_strategy, SchedulingStrategy::RoundRobin);
        assert_eq!(cfg.memory_utilization_percent, 80);
        assert_eq!(
            cfg.gpu_discovery_interval_secs,
            DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
        );
    }

    #[test]
    fn scheduler_role_config_defaults_missing_fields() {
        let value = serde_json::json!({});
        let cfg: SchedulerRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.gpu_registry_key, "gpu:registry");
        assert_eq!(cfg.scheduling_strategy, SchedulingStrategy::RoundRobin);
        assert_eq!(
            cfg.gpu_discovery_interval_secs,
            DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
        );
    }

    #[test]
    fn scheduler_role_config_ignores_unknown_fields() {
        let value = serde_json::json!({
            "gpu_registry_key": "gpu:registry",
            "scheduling_strategy": "round_robin",
            "future_field": 42
        });
        let cfg: SchedulerRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.gpu_registry_key, "gpu:registry");
        assert_eq!(cfg.scheduling_strategy, SchedulingStrategy::RoundRobin);
    }

    #[test]
    fn scheduler_role_config_rejects_invalid_scheduling_strategy_string() {
        let value = serde_json::json!({
            "gpu_registry_key": "gpu:registry",
            "scheduling_strategy": "best_fti"
        });
        let err = parse_role_config::<SchedulerRoleConfig>(Some(&value)).unwrap_err();
        assert!(
            format!("{err:#}").contains("unknown scheduling strategy"),
            "expected scheduling strategy parse error, got: {err:#}"
        );
    }

    #[test]
    fn prefetch_role_config_parses_handoff_stream() {
        let value = serde_json::json!({"handoff_stream": "custom_schedule"});
        let cfg: PrefetchRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.handoff_stream, "custom_schedule");
        assert!(!cfg.fail_on_plan_generation_error);
    }

    #[test]
    fn prefetch_role_config_parses_fail_open_override() {
        let value = serde_json::json!({
            "handoff_stream": "custom_schedule",
            "fail_on_plan_generation_error": false
        });
        let cfg: PrefetchRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.handoff_stream, "custom_schedule");
        assert!(!cfg.fail_on_plan_generation_error);
    }

    #[test]
    fn prefetch_role_config_parses_fail_closed_override() {
        let value = serde_json::json!({
            "handoff_stream": "custom_schedule",
            "fail_on_plan_generation_error": true
        });
        let cfg: PrefetchRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.handoff_stream, "custom_schedule");
        assert!(cfg.fail_on_plan_generation_error);
    }

    #[test]
    fn prefetch_role_config_defaults_when_none() {
        let cfg: PrefetchRoleConfig = parse_role_config(None).unwrap();
        assert_eq!(cfg.handoff_stream, "schedule");
        assert!(!cfg.fail_on_plan_generation_error);
    }

    #[test]
    fn prepare_role_config_parses_values_from_json() {
        let value = serde_json::json!({
            "python_executable": "python3",
            "runner_path": "/tmp/plugin_hook_runner.py",
            "hook_timeout_secs": 15
        });
        let cfg: PrepareRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.python_executable, "python3");
        assert_eq!(cfg.runner_path, "/tmp/plugin_hook_runner.py");
        assert_eq!(cfg.hook_timeout_secs, 15);
    }

    #[test]
    fn prepare_role_config_defaults_when_none() {
        let cfg: PrepareRoleConfig = parse_role_config(None).unwrap();
        assert_eq!(cfg.python_executable, "python3");
        assert_eq!(cfg.hook_timeout_secs, 60);
        assert!(cfg.runner_path.ends_with("scripts/plugin_hook_runner.py"));
    }

    #[test]
    fn validate_rejects_runtime_env_with_empty_python_executable() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "streams": ["prepare"],
            "python_runtime_envs": {
                "python.cpu.test": {
                    "python_executable": ""
                }
            },
            "roles": {
                "prepare": {
                    "inputs": [{"stream": "prepare", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": []
                }
            }
        }))
        .unwrap();

        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("python_executable"),
            "expected runtime env validation error, got: {err}"
        );
    }

    #[test]
    fn resolve_env_includes_python_runtime_env_registry() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "streams": ["prepare"],
            "python_runtime_envs": {
                "python.cpu.test": {
                    "python_executable": "/opt/envs/python.cpu.test/bin/python",
                    "env": {
                        "PYTHONPATH": "/opt/test"
                    }
                }
            },
            "roles": {
                "prepare": {
                    "inputs": [{"stream": "prepare", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": []
                }
            }
        }))
        .unwrap();

        let env = cfg.resolve_env("prepare").unwrap();
        assert_eq!(
            env.python_runtime_envs["python.cpu.test"].python_executable,
            "/opt/envs/python.cpu.test/bin/python"
        );
        assert_eq!(
            env.python_runtime_envs["python.cpu.test"].env["PYTHONPATH"],
            "/opt/test"
        );
    }

    #[test]
    fn parse_role_config_returns_error_on_type_mismatch() {
        let value = serde_json::json!({"gpu_registry_key": 42});
        let result: Result<SchedulerRoleConfig> = parse_role_config(Some(&value));
        assert!(result.is_err());
    }

    #[test]
    fn results_role_config_parses_ttl_from_json_value() {
        let value = serde_json::json!({"result_ttl_seconds": 7200});
        let cfg: ResultsRoleConfig = parse_role_config(Some(&value)).unwrap();
        assert_eq!(cfg.result_ttl_seconds, 7200);
    }

    #[test]
    fn results_role_config_defaults_when_none() {
        let cfg: ResultsRoleConfig = parse_role_config(None).unwrap();
        assert_eq!(cfg.result_ttl_seconds, 86_400);
    }

    // --- PR-042: stream names ending in :grp are rejected ---

    #[test]
    fn validate_rejects_stream_name_ending_with_grp() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["results:grp"],
            "roles": {
                "results": {
                    "inputs": [{"stream": "results:grp", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 100}],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains(":grp"),
            "expected :grp suffix error, got: {err}"
        );
    }

    // --- PR-053: duplicate stream names rejected ---

    #[test]
    fn validate_rejects_duplicate_stream_names() {
        let cfg: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["prefetch", "schedule", "prefetch"],
            "roles": {
                "prefetch": {
                    "inputs": [{"stream": "prefetch", "max_dequeue_items": 4,
                                "poll_interval_ms": 50, "block_ms": 500}],
                    "outputs": ["schedule"]
                }
            }
        }))
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("duplicate stream"),
            "expected duplicate stream error, got: {err}"
        );
    }

    // --- Config Resolution Tests ---

    #[test]
    fn resolve_config_and_role_fails_without_any_source() {
        let err = resolve_config_and_role(None, None).unwrap_err();
        assert!(
            err.to_string().contains("no config source"),
            "expected config source error, got: {err}"
        );
    }

    #[test]
    fn resolve_config_and_role_loads_from_explicit_path() {
        let tmp =
            std::env::temp_dir().join(format!("worker-runtime-test-{}.json", std::process::id()));
        let json = serde_json::json!({
            "stream_prefix": "test:",
            "streams": ["a"],
            "roles": {
                "worker": {
                    "inputs": [{"stream": "a", "max_dequeue_items": 1,
                                "poll_interval_ms": 10, "block_ms": 50}],
                    "outputs": []
                }
            }
        });
        std::fs::write(&tmp, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let (config, role) = resolve_config_and_role(Some(&tmp), Some("worker")).unwrap();
        assert_eq!(config.stream_prefix, "test:");
        assert_eq!(role, "worker");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn resolve_config_and_role_fails_when_role_missing() {
        let tmp = std::env::temp_dir().join(format!(
            "worker-runtime-test-norole-{}.json",
            std::process::id()
        ));
        let json = serde_json::json!({
            "stream_prefix": "",
            "streams": ["a"],
            "roles": {"w": {"inputs": [{"stream": "a", "max_dequeue_items": 1,
                             "poll_interval_ms": 10, "block_ms": 50}], "outputs": []}}
        });
        std::fs::write(&tmp, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let err = resolve_config_and_role(Some(&tmp), None).unwrap_err();
        assert!(
            err.to_string().contains("no role"),
            "expected role error, got: {err}"
        );

        std::fs::remove_file(&tmp).ok();
    }
}
