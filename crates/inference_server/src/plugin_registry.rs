/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub const DEFAULT_PLUGIN_MANIFEST_NAME: &str = "plugin.yaml";
const DEFAULT_EXECUTOR_CLASS: &str = "default";
const SUPPORTED_PIPELINE_PROFILES: &str = "batch, default, ensemble, postprocess, prefetch, simple";

mod discovery;
mod readiness;
mod registered;

pub use discovery::{
    discover_plugins, discover_plugins_with_enabled_id, plugin_manifest_path, resolve_plugin_dirs,
    resolve_script_path,
};

use self::readiness::validate_readiness_kind;

#[cfg(test)]
use self::readiness::{
    probe_python_module_with_candidates_and_env, python_probe_candidates_from_env,
};

fn canonical_pipeline_profile(profile_name: &str) -> &str {
    match profile_name {
        "default" => "prefetch",
        _ => profile_name,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginManifest {
    pub metadata: PluginMetadata,
    pub ingress: PluginIngress,
    pub pipeline: PluginPipeline,
    pub runtime: PluginRuntime,
    #[serde(default)]
    pub resources: PluginResources,
    pub outputs: PluginOutputs,
    #[serde(default)]
    pub developer: PluginDeveloper,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginMetadata {
    pub id: String,
    pub display_name: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginIngress {
    pub content_types: Vec<String>,
    pub default_content_type: String,
    pub operations: PluginOperations,
    #[serde(default)]
    pub json_schema: Option<String>,
    #[serde(default)]
    pub json_schema_inline: Option<serde_json::Value>,
    #[serde(default)]
    pub form_schema: Option<String>,
    #[serde(default)]
    pub form_schema_inline: Option<serde_json::Value>,
    #[serde(default)]
    pub files: Vec<PluginFileField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginOperations {
    pub default: String,
    pub allowed: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginFileField {
    pub name: String,
    pub required: bool,
    pub media_types: Vec<String>,
    pub max_size_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginPipeline {
    pub stages: Vec<PluginStage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginStage {
    pub id: String,
    pub phase: String,
    pub handler: String,
    pub queue: String,
    #[serde(default)]
    pub next: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginRuntime {
    pub kind: String,
    pub entrypoint: String,
    pub executor_class: String,
    #[serde(default)]
    pub prepare_executor_class: Option<String>,
    #[serde(default)]
    pub postprocess_executor_class: Option<String>,
    #[serde(default)]
    pub readiness_executor_class: Option<String>,
    #[serde(default)]
    pub hook_timeout_seconds: PluginHookTimeouts,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginHookTimeouts {
    #[serde(default)]
    pub prepare: u64,
    #[serde(default)]
    pub execute: u64,
    #[serde(default)]
    pub postprocess: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginResources {
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub defaults: PluginResourceDefaults,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginResourceDefaults {
    #[serde(default)]
    pub gpus_required: u32,
    #[serde(default)]
    pub memory_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginOutputs {
    #[serde(default)]
    pub result_schema: Option<String>,
    #[serde(default)]
    pub result_schema_inline: Option<serde_json::Value>,
    pub primary_artifact: PluginPrimaryArtifact,
    pub retention_hours: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct PluginPrimaryArtifact {
    pub name: String,
    pub media_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginDeveloper {
    #[serde(default)]
    pub readiness: PluginReadinessConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginReadinessConfig {
    #[serde(default)]
    pub recommended_check_phase: Option<String>,
    #[serde(default)]
    pub python_modules: Vec<String>,
    #[serde(default)]
    pub env: Vec<PluginReadinessEnvCheck>,
    #[serde(default)]
    pub paths: Vec<PluginReadinessPathCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginReadinessEnvCheck {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub any_of: Vec<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default = "default_true")]
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct PluginReadinessPathCheck {
    pub path: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default = "default_true")]
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PluginReadinessReport {
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommended_check_phase: Option<String>,
    pub checks: Vec<PluginReadinessCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PluginReadinessCheck {
    #[serde(rename = "type")]
    pub check_type: String,
    pub name: String,
    pub required: bool,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonRuntimeEnvConfig {
    pub python_executable: String,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct DerivedWorkflowSchemas {
    #[serde(default)]
    request_schema: Option<serde_json::Value>,
    #[serde(default)]
    form_schema: Option<serde_json::Value>,
    #[serde(default)]
    result_schema: Option<serde_json::Value>,
}

#[derive(Debug, Default)]
pub struct PythonModuleProbe {
    cache: HashMap<String, std::result::Result<bool, String>>,
    runtime_envs: HashMap<String, PythonRuntimeEnvConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredPlugin {
    pub root_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: PluginManifest,
}

impl PluginManifest {
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        let mut value: serde_json::Value =
            serde_yaml::from_str(input).context("failed to parse plugin manifest YAML")?;
        expand_manifest_defaults(&mut value)?;
        serde_json::from_value(value).context("failed to parse plugin manifest YAML")
    }

    pub fn validate(&self) -> Result<()> {
        validate_non_empty("metadata.id", &self.metadata.id)?;
        validate_non_empty("metadata.display_name", &self.metadata.display_name)?;
        validate_non_empty("metadata.version", &self.metadata.version)?;
        validate_non_empty("metadata.description", &self.metadata.description)?;

        if self.ingress.content_types.is_empty() {
            bail!("ingress.content_types must contain at least one content type");
        }
        for content_type in &self.ingress.content_types {
            validate_non_empty("ingress.content_types[]", content_type)?;
        }
        if !self
            .ingress
            .content_types
            .iter()
            .any(|content_type| content_type == &self.ingress.default_content_type)
        {
            bail!(
                "default content type '{}' must be present in ingress.content_types",
                self.ingress.default_content_type
            );
        }

        if self.ingress.operations.allowed.is_empty() {
            bail!("ingress.operations.allowed must contain at least one operation");
        }
        for operation in &self.ingress.operations.allowed {
            validate_non_empty("ingress.operations.allowed[]", operation)?;
        }
        if !self
            .ingress
            .operations
            .allowed
            .iter()
            .any(|operation| operation == &self.ingress.operations.default)
        {
            bail!(
                "default operation '{}' must be present in ingress.operations.allowed",
                self.ingress.operations.default
            );
        }

        if !self.ingress.files.is_empty()
            && !self
                .ingress
                .content_types
                .iter()
                .any(|content_type| content_type == "multipart/form-data")
        {
            bail!("file upload fields require 'multipart/form-data' in ingress.content_types");
        }
        for file in &self.ingress.files {
            validate_non_empty("ingress.files[].name", &file.name)?;
            if file.media_types.is_empty() {
                bail!(
                    "ingress.files entry '{}' must declare at least one media type",
                    file.name
                );
            }
        }

        validate_schema_source(
            self.ingress.json_schema.as_deref(),
            self.ingress.json_schema_inline.as_ref(),
            "ingress.json_schema",
            "ingress.json_schema_inline",
            self.ingress
                .content_types
                .iter()
                .any(|content_type| content_type == "application/json")
                && self.runtime.kind != "python",
        )?;
        validate_schema_source(
            self.ingress.form_schema.as_deref(),
            self.ingress.form_schema_inline.as_ref(),
            "ingress.form_schema",
            "ingress.form_schema_inline",
            self.ingress
                .content_types
                .iter()
                .any(|content_type| content_type == "multipart/form-data")
                && self.runtime.kind != "python",
        )?;

        if self.pipeline.stages.is_empty() {
            bail!("pipeline.stages must contain at least one stage");
        }

        let mut seen_stage_ids = HashSet::new();
        let mut stage_ids = HashSet::new();
        for stage in &self.pipeline.stages {
            validate_non_empty("pipeline.stages[].id", &stage.id)?;
            validate_non_empty("pipeline.stages[].phase", &stage.phase)?;
            validate_non_empty("pipeline.stages[].handler", &stage.handler)?;
            validate_non_empty("pipeline.stages[].queue", &stage.queue)?;

            if !seen_stage_ids.insert(stage.id.clone()) {
                bail!("duplicate pipeline stage id '{}'", stage.id);
            }
            stage_ids.insert(stage.id.clone());
        }

        for stage in &self.pipeline.stages {
            if let Some(next) = &stage.next {
                validate_non_empty("pipeline.stages[].next", next)?;
                if !stage_ids.contains(next) {
                    bail!(
                        "pipeline stage '{}' references unknown next stage '{}'",
                        stage.id,
                        next
                    );
                }
            }
        }

        validate_non_empty("runtime.kind", &self.runtime.kind)?;
        validate_non_empty("runtime.entrypoint", &self.runtime.entrypoint)?;
        validate_non_empty("runtime.executor_class", &self.runtime.executor_class)?;
        if let Some(executor_class) = &self.runtime.prepare_executor_class {
            validate_non_empty("runtime.prepare_executor_class", executor_class)?;
        }
        if let Some(executor_class) = &self.runtime.postprocess_executor_class {
            validate_non_empty("runtime.postprocess_executor_class", executor_class)?;
        }
        if let Some(executor_class) = &self.runtime.readiness_executor_class {
            validate_non_empty("runtime.readiness_executor_class", executor_class)?;
        }

        validate_schema_source(
            self.outputs.result_schema.as_deref(),
            self.outputs.result_schema_inline.as_ref(),
            "outputs.result_schema",
            "outputs.result_schema_inline",
            self.runtime.kind != "python",
        )?;
        validate_non_empty(
            "outputs.primary_artifact.name",
            &self.outputs.primary_artifact.name,
        )?;
        validate_non_empty(
            "outputs.primary_artifact.media_type",
            &self.outputs.primary_artifact.media_type,
        )?;

        if let Some(phase) = &self.developer.readiness.recommended_check_phase {
            validate_non_empty("developer.readiness.recommended_check_phase", phase)?;
        }

        for module_name in &self.developer.readiness.python_modules {
            validate_non_empty("developer.readiness.python_modules[]", module_name)?;
        }

        for env_check in &self.developer.readiness.env {
            if env_check
                .name
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
                && env_check.any_of.iter().all(|value| value.trim().is_empty())
            {
                bail!(
                    "developer.readiness.env entries must define 'name' or at least one non-empty 'any_of' entry"
                );
            }
            validate_readiness_kind("developer.readiness.env[].kind", env_check.kind.as_deref())?;
        }

        for path_check in &self.developer.readiness.paths {
            validate_non_empty("developer.readiness.paths[].path", &path_check.path)?;
            validate_readiness_kind(
                "developer.readiness.paths[].kind",
                path_check.kind.as_deref(),
            )?;
        }

        Ok(())
    }
}

fn expand_manifest_defaults(value: &mut serde_json::Value) -> Result<()> {
    let manifest = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("plugin manifest must be a mapping"))?;

    {
        let ingress = ensure_object(manifest, "ingress");
        normalize_ingress_aliases(ingress)?;
    }

    let runtime_profile = manifest
        .get("runtime")
        .and_then(serde_json::Value::as_object)
        .and_then(|runtime| runtime.get("profile"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if let Some(profile_name) = runtime_profile.as_deref() {
        let defaults = runtime_profile_defaults(profile_name)?;
        merge_missing(value, &defaults);
    }

    let manifest = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("plugin manifest must be a mapping"))?;

    {
        let runtime = ensure_object(manifest, "runtime");
        if runtime
            .get("executor_class")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            runtime.insert(
                "executor_class".to_string(),
                serde_json::Value::String(DEFAULT_EXECUTOR_CLASS.to_string()),
            );
        }
        if let Some(phases) = runtime.get("phases").cloned() {
            let phases = phases.as_object().cloned().ok_or_else(|| {
                anyhow::anyhow!("plugin manifest runtime.phases must be an object")
            })?;
            for (phase_name, field_name) in [
                ("prepare", "prepare_executor_class"),
                ("postprocess", "postprocess_executor_class"),
                ("readiness", "readiness_executor_class"),
            ] {
                let Some(profile_name) = phases
                    .get(phase_name)
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    continue;
                };
                let should_fill = runtime
                    .get(field_name)
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .is_none();
                if !should_fill {
                    continue;
                }
                runtime.insert(
                    field_name.to_string(),
                    serde_json::Value::String(resolve_runtime_profile_executor(profile_name)?),
                );
            }
        }
    }

    let pipeline_profile = manifest
        .get("pipeline")
        .and_then(serde_json::Value::as_object)
        .and_then(|pipeline| pipeline.get("profile"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if let Some(profile_name) = pipeline_profile.as_deref() {
        let execute_queue = {
            let runtime = ensure_object(manifest, "runtime");
            let executor_class = runtime
                .get("executor_class")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(DEFAULT_EXECUTOR_CLASS)
                .trim()
                .to_string();
            format!("execute.{executor_class}")
        };

        let pipeline = ensure_object(manifest, "pipeline");
        if !pipeline.contains_key("stages") {
            let options = pipeline
                .get("options")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let options = options.as_object().cloned().ok_or_else(|| {
                anyhow::anyhow!("plugin manifest pipeline.options must be an object")
            })?;
            pipeline.insert(
                "stages".to_string(),
                serde_json::Value::Array(build_pipeline_stages(
                    profile_name,
                    &options,
                    &execute_queue,
                )?),
            );
        }
    }

    let manifest = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("plugin manifest must be a mapping"))?;

    let runtime = ensure_object(manifest, "runtime");
    runtime
        .entry("kind".to_string())
        .or_insert_with(|| serde_json::Value::String("python".to_string()));
    runtime
        .entry("entrypoint".to_string())
        .or_insert_with(|| serde_json::Value::String("workflow.py".to_string()));

    let outputs = ensure_object(manifest, "outputs");
    outputs
        .entry("primary_artifact".to_string())
        .or_insert_with(|| {
            serde_json::json!({
                "name": "primary",
                "media_type": "application/json"
            })
        });
    outputs
        .entry("retention_hours".to_string())
        .or_insert_with(|| serde_json::Value::Number(serde_json::Number::from(24u64)));

    let developer = ensure_object(manifest, "developer");
    developer
        .entry("readiness".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(profile_name) = pipeline_profile.as_deref() {
        let readiness = ensure_object(developer, "readiness");
        readiness
            .entry("recommended_check_phase".to_string())
            .or_insert_with(|| {
                serde_json::Value::String(
                    pipeline_recommended_check_phase(profile_name).to_string(),
                )
            });
    }

    Ok(())
}

fn runtime_profile_defaults(profile_name: &str) -> Result<serde_json::Value> {
    let defaults = match profile_name {
        "python-test" => serde_json::json!({
            "runtime": {
                "kind": "python",
                "entrypoint": "workflow.py",
                "executor_class": "python.test",
                "hook_timeout_seconds": {
                    "prepare": 15,
                    "execute": 300,
                    "postprocess": 300
                }
            },
            "resources": {
                "defaults": {
                    "gpus_required": 0,
                    "memory_mb": 1024,
                }
            },
            "outputs": {
                "primary_artifact": {
                    "name": "primary",
                    "media_type": "application/json"
                },
                "retention_hours": 24
            },
            "developer": {
                "readiness": {
                    "python_modules": [],
                    "env": [],
                    "paths": []
                }
            }
        }),
        _ => bail!(
            "plugin manifest runtime.profile '{}' is not supported; supported profiles: python-test",
            profile_name
        ),
    };
    Ok(defaults)
}

fn resolve_runtime_profile_executor(profile_name: &str) -> Result<String> {
    let defaults = runtime_profile_defaults(profile_name)?;
    Ok(defaults
        .get("runtime")
        .and_then(serde_json::Value::as_object)
        .and_then(|runtime| runtime.get("executor_class"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string())
}

fn pipeline_recommended_check_phase(profile_name: &str) -> &'static str {
    match canonical_pipeline_profile(profile_name) {
        "simple" => "execute",
        "prefetch" | "postprocess" | "batch" | "ensemble" => "prepare",
        _ => "prepare",
    }
}

fn build_pipeline_stages(
    profile_name: &str,
    options: &serde_json::Map<String, serde_json::Value>,
    execute_queue: &str,
) -> Result<Vec<serde_json::Value>> {
    let profile_name = canonical_pipeline_profile(profile_name);
    let postprocess_enabled = options
        .get("postprocess")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let mut phases: Vec<&str> = match profile_name {
        "simple" => vec!["prepare", "execute"],
        "prefetch" => vec!["prepare", "prefetch", "schedule", "execute"],
        "postprocess" => vec!["prepare", "schedule", "execute", "postprocess"],
        "batch" => vec!["prepare", "batch", "schedule", "execute"],
        "ensemble" => {
            let mut phases = vec!["prepare"];
            match options.get("prefetch") {
                Some(serde_json::Value::Bool(true)) => phases.push("prefetch"),
                Some(serde_json::Value::String(scope)) if scope == "parent" => {
                    phases.push("prefetch")
                }
                Some(serde_json::Value::Bool(false)) | None => {}
                Some(_) => {
                    bail!(
                        "plugin manifest pipeline.options.prefetch must be true, false, or 'parent'"
                    )
                }
            }
            phases.extend(["fanout", "schedule", "execute", "collect"]);
            if postprocess_enabled {
                phases.push("postprocess");
            }
            phases
        }
        _ => {
            bail!(
                "plugin manifest pipeline.profile '{}' is not supported; supported profiles: {}",
                profile_name,
                SUPPORTED_PIPELINE_PROFILES
            )
        }
    };
    if postprocess_enabled && !phases.contains(&"postprocess") {
        phases.push("postprocess");
    }
    phases.push("results");

    let mut stages = Vec::with_capacity(phases.len());
    for (index, phase) in phases.iter().enumerate() {
        let next = phases.get(index + 1).map(|value| (*value).to_string());
        stages.push(stage_definition(phase, execute_queue, next.as_deref())?);
    }
    Ok(stages)
}

fn stage_definition(
    phase: &str,
    execute_queue: &str,
    next: Option<&str>,
) -> Result<serde_json::Value> {
    let stage = match phase {
        "prepare" => serde_json::json!({
            "id": "prepare",
            "phase": "prepare",
            "handler": "plugin_phase",
            "queue": "prepare",
            "next": next
        }),
        "prefetch" => serde_json::json!({
            "id": "prefetch",
            "phase": "prefetch",
            "handler": "prefetch",
            "queue": "prefetch",
            "next": next
        }),
        "batch" => serde_json::json!({
            "id": "batch",
            "phase": "batch",
            "handler": "batch",
            "queue": "batch",
            "next": next
        }),
        "fanout" => serde_json::json!({
            "id": "fanout",
            "phase": "fanout",
            "handler": "fanout",
            "queue": "fanout",
            "next": next
        }),
        "schedule" => serde_json::json!({
            "id": "schedule",
            "phase": "schedule",
            "handler": "schedule",
            "queue": "schedule",
            "next": next
        }),
        "execute" => serde_json::json!({
            "id": "execute",
            "phase": "execute",
            "handler": "plugin_phase",
            "queue": execute_queue,
            "next": next
        }),
        "collect" => serde_json::json!({
            "id": "collect",
            "phase": "collect",
            "handler": "collect",
            "queue": "collect",
            "next": next
        }),
        "postprocess" => serde_json::json!({
            "id": "postprocess",
            "phase": "postprocess",
            "handler": "plugin_phase",
            "queue": "postprocess",
            "next": next
        }),
        "results" => serde_json::json!({
            "id": "results",
            "phase": "results",
            "handler": "persist_results",
            "queue": "results",
            "next": null
        }),
        _ => bail!("unsupported pipeline phase '{}'", phase),
    };
    Ok(stage)
}

fn normalize_ingress_aliases(
    ingress: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    let content_type = ingress
        .get("content_type")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let has_content_types = ingress.contains_key("content_types");
    if !has_content_types {
        let inferred_content_type = if let Some(content_type) = content_type {
            content_type
        } else if ingress.contains_key("form_schema")
            || ingress.contains_key("form_schema_inline")
            || ingress.contains_key("files")
        {
            "multipart/form-data".to_string()
        } else {
            "application/json".to_string()
        };
        ingress.insert(
            "content_types".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::String(inferred_content_type)]),
        );
    }

    let normalized_content_type = ingress
        .get("content_types")
        .and_then(serde_json::Value::as_array)
        .and_then(|content_types| content_types.first())
        .and_then(serde_json::Value::as_str)
        .unwrap_or("application/json")
        .to_string();

    if let Some(request_schema) = ingress.get("request_schema").cloned() {
        let target = if normalized_content_type == "multipart/form-data" {
            "form_schema"
        } else {
            "json_schema"
        };
        ingress.entry(target.to_string()).or_insert(request_schema);
    }
    if let Some(request_schema_inline) = ingress.get("request_schema_inline").cloned() {
        let target = if normalized_content_type == "multipart/form-data" {
            "form_schema_inline"
        } else {
            "json_schema_inline"
        };
        ingress
            .entry(target.to_string())
            .or_insert(request_schema_inline);
    }
    if !ingress.contains_key("operations")
        && let Some(operation) = ingress.get("operation").cloned()
    {
        let operations = if let Some(operation_name) = operation.as_str() {
            serde_json::json!({"default": operation_name, "allowed": [operation_name]})
        } else {
            operation
        };
        ingress.insert("operations".to_string(), operations);
    }
    if let Some(content_types) = ingress
        .get("content_types")
        .and_then(serde_json::Value::as_array)
        .filter(|content_types| !content_types.is_empty())
        && !ingress.contains_key("default_content_type")
    {
        ingress.insert("default_content_type".to_string(), content_types[0].clone());
    }
    if !ingress.contains_key("operations") {
        ingress.insert(
            "operations".to_string(),
            serde_json::json!({"default": "run", "allowed": ["run"]}),
        );
    } else if let Some(operations) = ingress
        .get_mut("operations")
        .and_then(serde_json::Value::as_object_mut)
    {
        if !operations.contains_key("default") {
            operations.insert(
                "default".to_string(),
                serde_json::Value::String("run".to_string()),
            );
        }
        if !operations.contains_key("allowed") {
            let default = operations
                .get("default")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("run")
                .to_string();
            operations.insert(
                "allowed".to_string(),
                serde_json::Value::Array(vec![serde_json::Value::String(default)]),
            );
        }
    }
    Ok(())
}

fn ensure_object<'a>(
    parent: &'a mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, serde_json::Value> {
    let needs_insert = !matches!(parent.get(key), Some(serde_json::Value::Object(_)));
    if needs_insert {
        parent.insert(key.to_string(), serde_json::json!({}));
    }

    parent
        .get_mut(key)
        .and_then(serde_json::Value::as_object_mut)
        .expect("object just inserted")
}

fn merge_missing(target: &mut serde_json::Value, defaults: &serde_json::Value) {
    if let (Some(target_map), Some(default_map)) = (target.as_object_mut(), defaults.as_object()) {
        for (key, default_value) in default_map {
            match target_map.get_mut(key) {
                Some(existing) => merge_missing(existing, default_value),
                None => {
                    target_map.insert(key.clone(), default_value.clone());
                }
            }
        }
        return;
    }

    if target.is_null() {
        *target = defaults.clone();
    }
}

fn default_true() -> bool {
    true
}

fn validate_non_empty(field_name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
}

fn validate_schema_source(
    path_value: Option<&str>,
    inline_value: Option<&serde_json::Value>,
    path_field: &str,
    inline_field: &str,
    required: bool,
) -> Result<()> {
    if let Some(path) = path_value {
        validate_non_empty(path_field, path)?;
    }

    if let Some(document) = inline_value
        && !document.is_object()
    {
        bail!("{inline_field} must be a JSON object");
    }

    if path_value.is_some() && inline_value.is_some() {
        bail!("manifest must not define both '{path_field}' and '{inline_field}'");
    }

    if required && path_value.is_none() && inline_value.is_none() {
        bail!("manifest must define '{path_field}' or '{inline_field}'");
    }

    Ok(())
}

#[cfg(test)]
mod tests;
