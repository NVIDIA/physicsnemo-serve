/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::{PrepareRoleConfig, PythonRuntimeEnvConfig, parse_role_config};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

#[derive(Debug, Clone, Deserialize)]
struct PostprocessEnvelope {
    workflow_id: String,
    operation: Option<String>,
    result: JsonValue,
    stage_context: StageContext,
}

use crate::roles::stage::StageContext;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ResultOp {
    ObjectStorePublish {
        #[serde(default)]
        artifact: Option<String>,
        destination_uri: String,
    },
    DatasetExportNetcdf {
        #[serde(default)]
        artifact: Option<String>,
        #[serde(default)]
        target_artifact_name: Option<String>,
        #[serde(default)]
        filename: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct SelectedArtifact {
    name: String,
    storage_path: PathBuf,
    filename: String,
}

#[derive(Debug, Clone, Default)]
struct ObjectStoreRoots {
    s3: Option<PathBuf>,
    gcs: Option<PathBuf>,
    azure: Option<PathBuf>,
}

impl ObjectStoreRoots {
    fn from_env() -> Self {
        Self {
            s3: std::env::var("PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_S3")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(PathBuf::from),
            gcs: std::env::var("PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_GCS")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(PathBuf::from),
            azure: std::env::var("PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_AZURE")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(PathBuf::from),
        }
    }
}

pub struct PostprocessRole {
    python_executable: String,
    runner_path: PathBuf,
    dataset_runner_path: PathBuf,
    hook_timeout: Duration,
    input_streams: Vec<String>,
    python_runtime_envs: std::collections::BTreeMap<String, PythonRuntimeEnvConfig>,
    object_store_roots: ObjectStoreRoots,
}

impl PostprocessRole {
    pub fn from_env(env: &RoleEnv) -> Result<Self> {
        let cfg: PrepareRoleConfig = parse_role_config(env.role_config.as_ref())?;
        if cfg.python_executable.trim().is_empty() {
            return Err(anyhow!(
                "postprocess role config python_executable must be non-empty"
            ));
        }
        if cfg.runner_path.trim().is_empty() {
            return Err(anyhow!(
                "postprocess role config runner_path must be non-empty"
            ));
        }
        if cfg.hook_timeout_secs == 0 {
            return Err(anyhow!(
                "postprocess role config hook_timeout_secs must be greater than zero"
            ));
        }

        let runner_path = PathBuf::from(cfg.runner_path);

        Ok(Self {
            python_executable: cfg.python_executable,
            dataset_runner_path: runner_path.with_file_name("dataset_ops_runner.py"),
            runner_path,
            hook_timeout: Duration::from_secs(cfg.hook_timeout_secs),
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
            python_runtime_envs: env.python_runtime_envs.clone(),
            object_store_roots: ObjectStoreRoots::from_env(),
        })
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "postprocess: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn run_postprocess_hook(&self, payload: &JsonValue) -> Result<JsonValue> {
        let input =
            serde_json::to_vec(payload).context("postprocess: failed to serialize payload")?;
        let runtime_env =
            self.runtime_env_for_phase(payload, "postprocess", "postprocess_executor_class");
        let python_executable = runtime_env
            .map(|env| env.python_executable.as_str())
            .unwrap_or(self.python_executable.as_str());
        let mut command = Command::new(python_executable);
        command
            .arg(&self.runner_path)
            .arg("--phase")
            .arg("postprocess")
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(runtime_env) = runtime_env {
            command.envs(&runtime_env.env);
        }
        let mut child = command.spawn().with_context(|| {
            format!(
                "postprocess: failed to spawn hook runner '{}' with executable '{}'",
                self.runner_path.display(),
                python_executable
            )
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&input)
                .await
                .context("postprocess: failed to write payload to hook runner stdin")?;
        }

        let output = tokio::time::timeout(self.hook_timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                anyhow!(
                    "postprocess: plugin hook timed out after {}s",
                    self.hook_timeout.as_secs()
                )
            })?
            .context("postprocess: failed waiting for hook runner process")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let details = if stderr.is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr
            };
            return Err(anyhow!("postprocess: plugin hook failed: {details}"));
        }

        let stdout = String::from_utf8(output.stdout)
            .context("postprocess: hook runner stdout is not valid UTF-8")?;
        if stdout.trim().is_empty() {
            return Ok(json!({}));
        }

        let value: JsonValue =
            serde_json::from_str(&stdout).context("postprocess: hook output must be valid JSON")?;
        if !value.is_object() {
            return Err(anyhow!("postprocess: hook output must be a JSON object"));
        }
        Ok(value)
    }

    fn runtime_env_for_phase<'a>(
        &'a self,
        payload: &JsonValue,
        phase: &str,
        selector_field: &str,
    ) -> Option<&'a PythonRuntimeEnvConfig> {
        let executor_class = payload
            .get("runtime")
            .and_then(JsonValue::as_object)
            .and_then(|runtime| {
                runtime
                    .get(selector_field)
                    .and_then(JsonValue::as_str)
                    .or_else(|| runtime.get("executor_class").and_then(JsonValue::as_str))
            })
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let executor_class = executor_class?;
        self.python_runtime_envs.get(executor_class).or_else(|| {
            tracing::debug!(
                phase,
                executor_class,
                "postprocess: runtime env not found for requested executor_class, falling back to default python_executable"
            );
            None
        })
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let (typed, raw_payload) = decode_postprocess_payload(msg.payload())?;
        let next_stage = typed.stage_context.next_stage("postprocess")?;
        if !matches!(next_stage.phase.as_str(), "publish" | "results") {
            return Err(anyhow!(
                "postprocess: next stage must be 'publish' or 'results', got '{}'",
                next_stage.phase
            ));
        }

        let mut result_payload = self.run_postprocess_hook(&raw_payload).await?;
        self.apply_result_ops(&raw_payload, msg.run_id(), &mut result_payload)
            .await?;
        let status = result_payload
            .get("status")
            .and_then(JsonValue::as_str)
            .or_else(|| typed.result.get("status").and_then(JsonValue::as_str))
            .unwrap_or("succeeded")
            .to_string();
        if next_stage.phase == "publish" {
            let mut handoff_payload = raw_payload.clone();
            let handoff_map = handoff_payload
                .as_object_mut()
                .ok_or_else(|| anyhow!("postprocess: payload must remain a JSON object"))?;
            merge_prior_output_fields_into_publish_result(&typed.result, &mut result_payload)?;
            ensure_result_status(&mut result_payload, status.as_str())?;
            handoff_map.insert("result".to_string(), result_payload);
            crate::roles::stage::update_stage_context(handoff_map, &next_stage, "postprocess")?;
            let encoded = serde_json::to_string(&handoff_payload)
                .context("postprocess: encode publish payload")?;
            sink.handoff(msg, &next_stage.queue, &encoded, &next_stage.phase)
                .await
                .context("postprocess: failed to hand off to publish")?;
            return Ok(());
        }

        let completed_at = Utc::now().to_rfc3339();
        let (execution, payload) = build_execution_and_payload(
            msg.run_id(),
            typed.workflow_id.as_str(),
            status.as_str(),
            completed_at.as_str(),
            &typed.result,
            result_payload,
        )?;

        let results_envelope = json!({
            "run_id": msg.run_id(),
            "status": status,
            "workflow": typed.workflow_id,
            "completed_at": completed_at,
            "request": build_request_envelope(&raw_payload, typed.operation.as_deref()),
            "execution": execution,
            "payload": payload,
        });
        let encoded = serde_json::to_string(&results_envelope)
            .context("postprocess: encode results payload")?;

        sink.handoff(msg, &next_stage.queue, &encoded, &next_stage.phase)
            .await
            .context("postprocess: failed to hand off to results")?;
        Ok(())
    }

    async fn apply_result_ops(
        &self,
        raw_payload: &JsonValue,
        run_id: &str,
        result_payload: &mut JsonValue,
    ) -> Result<()> {
        let map = result_payload
            .as_object_mut()
            .ok_or_else(|| anyhow!("postprocess: result payload must be a JSON object"))?;
        let ops = take_result_ops(map)?;
        if ops.is_empty() {
            return Ok(());
        }

        for op in ops {
            match op {
                ResultOp::ObjectStorePublish {
                    artifact,
                    destination_uri,
                } => {
                    let selected =
                        select_result_artifact(map, artifact.as_deref().unwrap_or("primary"))?;
                    let published = self
                        .publish_artifact_to_object_store(&selected, &destination_uri)
                        .await?;
                    append_json_array_entry(
                        map,
                        "published_artifacts",
                        json!({
                            "kind": "object_store_publish",
                            "name": selected.name,
                            "source_artifact": selected.name,
                            "destination_uri": destination_uri,
                            "mirror_path": published.display().to_string(),
                            "filename": selected.filename,
                        }),
                    )?;
                }
                ResultOp::DatasetExportNetcdf {
                    artifact,
                    target_artifact_name,
                    filename,
                } => {
                    let selected =
                        select_result_artifact(map, artifact.as_deref().unwrap_or("primary"))?;
                    let generated = self
                        .export_dataset_to_netcdf(
                            raw_payload,
                            &selected,
                            run_id,
                            target_artifact_name.as_deref(),
                            filename.as_deref(),
                        )
                        .await?;
                    merge_artifact(
                        map,
                        json!({
                            "name": generated.name,
                            "media_type": "application/x-netcdf",
                            "storage_path": generated.storage_path.display().to_string(),
                            "filename": generated.filename,
                        }),
                    )?;
                }
            }
        }

        Ok(())
    }

    async fn publish_artifact_to_object_store(
        &self,
        selected: &SelectedArtifact,
        destination_uri: &str,
    ) -> Result<PathBuf> {
        let destination_path =
            resolve_object_store_destination(destination_uri, &self.object_store_roots)?;
        copy_path_recursive(&selected.storage_path, &destination_path).await?;
        Ok(destination_path)
    }

    async fn export_dataset_to_netcdf(
        &self,
        raw_payload: &JsonValue,
        selected: &SelectedArtifact,
        run_id: &str,
        target_artifact_name: Option<&str>,
        filename: Option<&str>,
    ) -> Result<SelectedArtifact> {
        let artifact_name = target_artifact_name
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}_netcdf", selected.name));
        let output_filename = filename
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}.nc", sanitize_name(&artifact_name)));
        let destination_path = selected
            .storage_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&output_filename);

        self.run_dataset_export_runner(raw_payload, &selected.storage_path, &destination_path)
            .await
            .with_context(|| {
                format!(
                    "postprocess: failed dataset_export_netcdf for run '{}' and artifact '{}'",
                    run_id, selected.name
                )
            })?;

        Ok(SelectedArtifact {
            name: artifact_name,
            storage_path: destination_path,
            filename: output_filename,
        })
    }

    async fn run_dataset_export_runner(
        &self,
        raw_payload: &JsonValue,
        source_path: &Path,
        destination_path: &Path,
    ) -> Result<()> {
        let input = serde_json::to_vec(&json!({
            "source_path": source_path.display().to_string(),
            "destination_path": destination_path.display().to_string(),
        }))
        .context("postprocess: failed to encode dataset export payload")?;
        let runtime_env =
            self.runtime_env_for_phase(raw_payload, "postprocess", "postprocess_executor_class");
        let python_executable = runtime_env
            .map(|env| env.python_executable.as_str())
            .unwrap_or(self.python_executable.as_str());
        let mut command = Command::new(python_executable);
        command
            .arg(&self.dataset_runner_path)
            .arg("--op")
            .arg("export_netcdf")
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(runtime_env) = runtime_env {
            command.envs(&runtime_env.env);
        }
        let mut child = command.spawn().with_context(|| {
            format!(
                "postprocess: failed to spawn dataset runner '{}' with executable '{}'",
                self.dataset_runner_path.display(),
                python_executable
            )
        })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&input)
                .await
                .context("postprocess: failed to write dataset op payload to stdin")?;
        }
        let output = tokio::time::timeout(self.hook_timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                anyhow!(
                    "postprocess: dataset export runner timed out after {}s",
                    self.hook_timeout.as_secs()
                )
            })?
            .context("postprocess: failed waiting for dataset export runner")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let details = if stderr.is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr
            };
            return Err(anyhow!(
                "postprocess: dataset export runner failed: {details}"
            ));
        }
        Ok(())
    }
}

fn decode_postprocess_payload(raw: &str) -> Result<(PostprocessEnvelope, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("postprocess: empty payload"));
    }

    let value: JsonValue =
        serde_json::from_str(raw).context("postprocess: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("postprocess: payload must be a JSON object"));
    }

    let typed: PostprocessEnvelope =
        serde_json::from_value(value.clone()).context("postprocess: invalid payload schema")?;
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "postprocess: workflow_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "postprocess" {
        return Err(anyhow!(
            "postprocess: payload current_phase must be 'postprocess', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    if !typed.result.is_object() {
        return Err(anyhow!("postprocess: payload result must be a JSON object"));
    }

    Ok((typed, value))
}

fn build_request_envelope(raw_payload: &JsonValue, operation: Option<&str>) -> JsonValue {
    let mut request = raw_payload
        .get("request")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    if !request.contains_key("operation")
        && let Some(operation) = operation
    {
        request.insert(
            "operation".to_string(),
            JsonValue::String(operation.to_string()),
        );
    }
    if !request.contains_key("parameters")
        && let Some(parameters) = raw_payload.get("parameters")
    {
        request.insert("parameters".to_string(), parameters.clone());
    }
    JsonValue::Object(request)
}

fn move_execution_field(
    payload: &mut JsonMap<String, JsonValue>,
    execution: &mut JsonMap<String, JsonValue>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = payload.remove(source_key) {
        execution.entry(target_key.to_string()).or_insert(value);
    }
}

fn remove_empty_execution_array(execution: &mut JsonMap<String, JsonValue>, key: &str) {
    let should_remove = execution
        .get(key)
        .and_then(JsonValue::as_array)
        .is_some_and(|entries| entries.is_empty());
    if should_remove {
        execution.remove(key);
    }
}

fn copy_execution_field(
    source: &JsonMap<String, JsonValue>,
    execution: &mut JsonMap<String, JsonValue>,
    source_key: &str,
    target_key: &str,
) {
    if !execution.contains_key(target_key)
        && let Some(value) = source.get(source_key)
    {
        execution.insert(target_key.to_string(), value.clone());
    }
}

fn copy_result_field_if_missing(
    source: &JsonMap<String, JsonValue>,
    result: &mut JsonMap<String, JsonValue>,
    key: &str,
) {
    if !result.contains_key(key)
        && let Some(value) = source.get(key)
    {
        result.insert(key.to_string(), value.clone());
    }
}

fn merge_prior_output_fields_into_publish_result(
    prior_result: &JsonValue,
    result_payload: &mut JsonValue,
) -> Result<()> {
    let result = result_payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("postprocess: result payload must be a JSON object"))?;
    let Some(prior) = prior_result.as_object() else {
        return Ok(());
    };
    for key in [
        "outputs",
        "artifacts",
        "published_outputs",
        "published_artifacts",
        "batch_info",
        "output_path",
        "output_archive",
        "error",
        "execution_time_seconds",
    ] {
        copy_result_field_if_missing(prior, result, key);
    }
    if !result.contains_key("output_path") {
        let outputs = result.get("outputs").or_else(|| result.get("artifacts"));
        if let Some(path) = derive_primary_output_path(outputs) {
            result.insert("output_path".to_string(), JsonValue::String(path));
        }
    }
    Ok(())
}

fn ensure_result_status(result_payload: &mut JsonValue, status: &str) -> Result<()> {
    let result = result_payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("postprocess: result payload must be a JSON object"))?;
    if !result.get("status").is_some_and(JsonValue::is_string) {
        result.insert("status".to_string(), JsonValue::String(status.to_string()));
    }
    Ok(())
}

fn derive_primary_output_path(outputs: Option<&JsonValue>) -> Option<String> {
    let outputs = outputs?.as_array()?;
    let primary = outputs
        .iter()
        .find(|entry| {
            entry
                .get("primary")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| outputs.first())?;
    primary
        .get("storage_path")
        .or_else(|| primary.get("path"))
        .or_else(|| primary.get("output_path"))
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

fn build_execution_and_payload(
    run_id: &str,
    workflow_id: &str,
    status: &str,
    completed_at: &str,
    prior_result: &JsonValue,
    result_payload: JsonValue,
) -> Result<(JsonValue, JsonValue)> {
    let mut payload = result_payload
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("postprocess: result payload must be a JSON object"))?;
    let mut execution = JsonMap::new();
    execution.insert("run_id".to_string(), JsonValue::String(run_id.to_string()));
    execution.insert("status".to_string(), JsonValue::String(status.to_string()));
    execution.insert(
        "workflow".to_string(),
        JsonValue::String(workflow_id.to_string()),
    );
    execution.insert(
        "completed_at".to_string(),
        JsonValue::String(completed_at.to_string()),
    );
    move_execution_field(&mut payload, &mut execution, "run_id", "run_id");
    move_execution_field(&mut payload, &mut execution, "status", "status");
    move_execution_field(&mut payload, &mut execution, "outputs", "outputs");
    move_execution_field(&mut payload, &mut execution, "artifacts", "outputs");
    move_execution_field(
        &mut payload,
        &mut execution,
        "published_outputs",
        "published_outputs",
    );
    move_execution_field(
        &mut payload,
        &mut execution,
        "published_artifacts",
        "published_artifacts",
    );
    move_execution_field(&mut payload, &mut execution, "batch_info", "batch_info");
    move_execution_field(&mut payload, &mut execution, "output_path", "output_path");
    move_execution_field(
        &mut payload,
        &mut execution,
        "output_archive",
        "output_archive",
    );
    move_execution_field(&mut payload, &mut execution, "error", "error");
    move_execution_field(
        &mut payload,
        &mut execution,
        "execution_time_seconds",
        "execution_time_seconds",
    );
    remove_empty_execution_array(&mut execution, "outputs");
    remove_empty_execution_array(&mut execution, "published_outputs");
    remove_empty_execution_array(&mut execution, "published_artifacts");
    if let Some(prior) = prior_result.as_object() {
        copy_execution_field(prior, &mut execution, "outputs", "outputs");
        copy_execution_field(prior, &mut execution, "artifacts", "outputs");
        copy_execution_field(
            prior,
            &mut execution,
            "published_outputs",
            "published_outputs",
        );
        copy_execution_field(
            prior,
            &mut execution,
            "published_artifacts",
            "published_artifacts",
        );
        copy_execution_field(prior, &mut execution, "batch_info", "batch_info");
        copy_execution_field(prior, &mut execution, "output_path", "output_path");
        copy_execution_field(prior, &mut execution, "output_archive", "output_archive");
        copy_execution_field(prior, &mut execution, "error", "error");
        copy_execution_field(
            prior,
            &mut execution,
            "execution_time_seconds",
            "execution_time_seconds",
        );
    }
    if !execution.contains_key("output_path")
        && let Some(path) = derive_primary_output_path(execution.get("outputs"))
    {
        execution.insert("output_path".to_string(), JsonValue::String(path));
    }
    Ok((JsonValue::Object(execution), JsonValue::Object(payload)))
}

fn take_result_ops(map: &mut JsonMap<String, JsonValue>) -> Result<Vec<ResultOp>> {
    let Some(raw_ops) = map.remove("result_ops") else {
        return Ok(Vec::new());
    };
    serde_json::from_value(raw_ops)
        .context("postprocess: result_ops must be a valid operation array")
}

fn select_result_artifact(
    result_payload: &JsonMap<String, JsonValue>,
    requested: &str,
) -> Result<SelectedArtifact> {
    let artifacts = result_payload
        .get("artifacts")
        .and_then(JsonValue::as_array)
        .filter(|entries| !entries.is_empty());
    let output_path = result_payload
        .get("output_path")
        .and_then(JsonValue::as_str);

    if let Some(entries) = artifacts {
        let entry = if requested == "primary" {
            entries.first()
        } else {
            entries
                .iter()
                .find(|entry| entry.get("name").and_then(JsonValue::as_str) == Some(requested))
        };
        if let Some(entry) = entry {
            let name = entry
                .get("name")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| anyhow!("postprocess: artifact entry missing name"))?;
            let storage_path = entry
                .get("storage_path")
                .or_else(|| entry.get("path"))
                .or_else(|| entry.get("output_path"))
                .and_then(JsonValue::as_str)
                .or(output_path)
                .ok_or_else(|| anyhow!("postprocess: artifact '{}' missing storage path", name))?;
            let filename = entry
                .get("filename")
                .or_else(|| entry.get("original_filename"))
                .and_then(JsonValue::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| derive_filename(storage_path, name));
            return Ok(SelectedArtifact {
                name: name.to_string(),
                storage_path: PathBuf::from(storage_path),
                filename,
            });
        }
    }

    if requested == "primary"
        && let Some(path) = output_path
    {
        return Ok(SelectedArtifact {
            name: "primary".to_string(),
            filename: derive_filename(path, "primary"),
            storage_path: PathBuf::from(path),
        });
    }

    Err(anyhow!(
        "postprocess: artifact '{}' is not present in result payload",
        requested
    ))
}

fn resolve_object_store_destination(
    destination_uri: &str,
    roots: &ObjectStoreRoots,
) -> Result<PathBuf> {
    if let Some(path) = destination_uri.strip_prefix("s3://") {
        let root = roots.s3.as_ref().ok_or_else(|| {
            anyhow!("postprocess: PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_S3 is not configured")
        })?;
        return join_relative_destination(root, path);
    }
    if let Some(path) = destination_uri.strip_prefix("gs://") {
        let root = roots.gcs.as_ref().ok_or_else(|| {
            anyhow!("postprocess: PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_GCS is not configured")
        })?;
        return join_relative_destination(root, path);
    }
    if let Some(path) = destination_uri
        .strip_prefix("az://")
        .or_else(|| destination_uri.strip_prefix("azure://"))
    {
        let root = roots.azure.as_ref().ok_or_else(|| {
            anyhow!("postprocess: PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_AZURE is not configured")
        })?;
        return join_relative_destination(root, path);
    }
    Err(anyhow!(
        "postprocess: unsupported object store destination_uri '{}'",
        destination_uri
    ))
}

fn join_relative_destination(root: &Path, relative: &str) -> Result<PathBuf> {
    let mut destination = root.to_path_buf();
    for segment in relative.split('/').filter(|segment| !segment.is_empty()) {
        let segment_path = Path::new(segment);
        if segment.contains('\\') {
            return Err(anyhow!(
                "postprocess: destination_uri contains invalid path separator in segment '{}'",
                segment
            ));
        }
        if segment_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(anyhow!(
                "postprocess: destination_uri segment '{}' is not a safe relative path segment",
                segment
            ));
        }
        destination.push(segment);
    }
    Ok(destination)
}

async fn copy_path_recursive(source: &Path, destination: &Path) -> Result<()> {
    let source = source.to_path_buf();
    let destination = destination.to_path_buf();
    tokio::task::spawn_blocking(move || copy_path_recursive_blocking(&source, &destination))
        .await
        .context("postprocess: copy task join failed")?
}

fn copy_path_recursive_blocking(source: &Path, destination: &Path) -> Result<()> {
    let metadata = std::fs::metadata(source).with_context(|| {
        format!(
            "postprocess: failed to read source metadata '{}'",
            source.display()
        )
    })?;
    if metadata.is_dir() {
        std::fs::create_dir_all(destination).with_context(|| {
            format!(
                "postprocess: failed to create destination directory '{}'",
                destination.display()
            )
        })?;
        for entry in std::fs::read_dir(source).with_context(|| {
            format!(
                "postprocess: failed to read source directory '{}'",
                source.display()
            )
        })? {
            let entry = entry.context("postprocess: failed to read directory entry")?;
            let child_source = entry.path();
            let child_destination = destination.join(entry.file_name());
            copy_path_recursive_blocking(&child_source, &child_destination)?;
        }
        return Ok(());
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "postprocess: failed to create destination parent '{}'",
                parent.display()
            )
        })?;
    }
    std::fs::copy(source, destination).with_context(|| {
        format!(
            "postprocess: failed to copy '{}' to '{}'",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn merge_artifact(map: &mut JsonMap<String, JsonValue>, artifact: JsonValue) -> Result<()> {
    let artifact_name = artifact
        .get("name")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| anyhow!("postprocess: generated artifact missing name"))?
        .to_string();
    let artifacts = map
        .entry("artifacts".to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()));
    let array = artifacts
        .as_array_mut()
        .ok_or_else(|| anyhow!("postprocess: artifacts must be an array"))?;
    if let Some(existing) = array
        .iter_mut()
        .find(|entry| entry.get("name").and_then(JsonValue::as_str) == Some(artifact_name.as_str()))
    {
        *existing = artifact;
    } else {
        array.push(artifact);
    }
    Ok(())
}

fn append_json_array_entry(
    map: &mut JsonMap<String, JsonValue>,
    key: &str,
    entry: JsonValue,
) -> Result<()> {
    let value = map
        .entry(key.to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()));
    let array = value
        .as_array_mut()
        .ok_or_else(|| anyhow!("postprocess: field '{}' must be an array", key))?;
    array.push(entry);
    Ok(())
}

fn derive_filename(storage_path: &str, fallback: &str) -> String {
    Path::new(storage_path)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

fn sanitize_name(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        "artifact".to_string()
    } else {
        sanitized
    }
}

impl WorkerRole for PostprocessRole {
    fn name(&self) -> &'static str {
        "postprocess"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            self.process_message(msg, sink).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::Duration;

    use anyhow::Result;
    use serde_json::{Value, json};

    use crate::config::InputStreamSpec;
    use crate::test_env;
    use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

    use super::{
        ObjectStoreRoots, PostprocessRole, ensure_result_status,
        merge_prior_output_fields_into_publish_result, resolve_object_store_destination,
    };
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[derive(Debug, Clone)]
    struct HandoffRecord {
        dest_stream: String,
        payload: String,
        stage: String,
    }

    struct RecordingSink {
        writes: Mutex<Vec<HandoffRecord>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                writes: Mutex::new(Vec::new()),
            }
        }

        fn writes(&self) -> Vec<HandoffRecord> {
            self.writes.lock().expect("recording lock poisoned").clone()
        }
    }

    impl MessageSink for RecordingSink {
        fn enqueue<'a>(
            &'a self,
            _stream: &'a str,
            _run_id: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("1-0".to_string()) })
        }

        fn ack_message<'a>(&'a self, _msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.writes
                    .lock()
                    .expect("recording lock poisoned")
                    .push(HandoffRecord {
                        dest_stream: dest_stream.to_string(),
                        payload: payload.to_string(),
                        stage: stage.to_string(),
                    });
                Ok("1-0".to_string())
            })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    struct EnvRestore {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvRestore {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(key).ok();
            test_env::set_env_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            test_env::set_env_var(self.key, self.previous.as_deref());
        }
    }

    fn test_python_executable() -> String {
        if let Ok(explicit) = std::env::var("PHYSICSNEMO_SERVE_TEST_PYTHON")
            && !explicit.trim().is_empty()
        {
            return explicit;
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        #[cfg(windows)]
        let candidates = [manifest_dir.join("../../.venv/Scripts/python.exe")];

        #[cfg(not(windows))]
        let candidates = [manifest_dir.join("../../.venv/bin/python")];

        for candidate in candidates {
            if candidate.is_file() {
                return candidate.display().to_string();
            }
        }

        "python3".to_string()
    }

    fn postprocess_env(runner_path: &str) -> RoleEnv {
        RoleEnv {
            role_name: "postprocess".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "postprocess".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec![],
            role_config: Some(json!({
                "python_executable": test_python_executable(),
                "runner_path": runner_path,
                "hook_timeout_secs": 5
            })),
            python_runtime_envs: Default::default(),
        }
    }

    fn runner_path() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/plugin_hook_runner.py")
            .display()
            .to_string()
    }

    fn write_plugin_with_postprocess(tmp: &tempfile::TempDir, postprocess_body: &str) -> PathBuf {
        let plugin_root = tmp.path().join("plugins").join("demo-plugin");
        std::fs::create_dir_all(&plugin_root).expect("plugin dir should be created");
        std::fs::write(
            plugin_root.join("plugin.yaml"),
            r#"
metadata:
  id: demo-plugin
  display_name: Demo Plugin
  version: 1.0.0
  description: Example plugin
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: both
    allowed: [both]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute
      next: postprocess
    - id: postprocess
      phase: postprocess
      handler: plugin_phase
      queue: postprocess
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.gpu.physicsnemo
resources:
  defaults:
    gpus_required: 1
    memory_mb: 4096
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: pressure_field
    media_type: application/x-npz
  retention_hours: 24
"#,
        )
        .expect("manifest should be written");
        std::fs::write(plugin_root.join("plugin.py"), postprocess_body)
            .expect("plugin entrypoint should be written");
        plugin_root
    }

    fn write_plugin(tmp: &tempfile::TempDir) -> PathBuf {
        write_plugin_with_postprocess(
            tmp,
            r#"
def postprocess(ctx):
    return {
        "status": "succeeded",
        "output_path": ctx["result"]["output_path"],
        "artifacts": [
            {
                "name": "pressure_field",
                "media_type": "application/x-npz",
            }
        ],
        "echo_operation": ctx["operation"],
    }
"#,
        )
    }

    #[cfg(unix)]
    fn write_fake_python_runtime(tmp: &tempfile::TempDir, name: &str, stdout_json: &str) -> String {
        let script_path = tmp.path().join(name);
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{}'\n",
                stdout_json.replace('\'', "'\"'\"'")
            ),
        )
        .expect("fake runtime script should be written");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata should load")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("script should be executable");
        script_path.display().to_string()
    }

    #[cfg(unix)]
    fn write_sleeping_runtime_script(
        tmp: &tempfile::TempDir,
        name: &str,
        pid_file: &Path,
    ) -> String {
        let script_path = tmp.path().join(name);
        let pid_file_escaped = pid_file.display().to_string().replace('\'', "'\"'\"'");
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\necho \"$$\" > '{}'\nsleep 30\nprintf '{{}}\\n'\n",
                pid_file_escaped
            ),
        )
        .expect("sleeping runtime script should be written");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata should load")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("script should be executable");
        script_path.display().to_string()
    }

    #[cfg(unix)]
    async fn wait_for_pid_file(pid_file: &Path) -> u32 {
        let mut last_error = String::new();
        for _ in 0..40 {
            match std::fs::read_to_string(pid_file) {
                Ok(raw) => {
                    let trimmed = raw.trim();
                    match trimmed.parse::<u32>() {
                        Ok(pid) => return pid,
                        Err(err) => {
                            last_error = format!(
                                "pid file '{}' contained {:?}: {err}",
                                pid_file.display(),
                                trimmed
                            );
                        }
                    }
                }
                Err(err) => {
                    last_error = format!("failed to read pid file '{}': {err}", pid_file.display());
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("pid file never contained a parseable pid: {last_error}");
    }

    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(unix)]
    fn kill_process(pid: u32) {
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.to_string())
            .status();
    }

    fn create_tiny_zarr_dataset(tmp: &tempfile::TempDir) -> PathBuf {
        let dataset_path = tmp.path().join("forecast.zarr");
        let status = std::process::Command::new("python3")
            .arg("-c")
            .arg(
                r#"
from pathlib import Path
import numpy as np
import xarray as xr

path = Path(__import__("sys").argv[1])
ds = xr.Dataset({"temperature": (("time",), np.array([280.0, 281.5], dtype=np.float32))}, coords={"time": np.array([0, 1], dtype=np.int64)})
ds.to_zarr(path, mode="w")
"#,
            )
            .arg(dataset_path.display().to_string())
            .status()
            .expect("python should create zarr dataset");
        assert!(
            status.success(),
            "expected python zarr generation to succeed"
        );
        dataset_path
    }

    #[tokio::test]
    async fn postprocess_role_invokes_plugin_hook_and_handoffs_results_envelope() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin(&tmp);
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "parameters": { "batch_size": 128000 },
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "result": {
                    "status": "succeeded",
                    "output_path": "/tmp/run-plugin.npz"
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "postprocess"
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
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
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "results");
        assert_eq!(writes[0].stage, "results");

        let forwarded: Value =
            serde_json::from_str(&writes[0].payload).expect("payload should remain valid JSON");
        assert_eq!(forwarded["run_id"], "run-plugin");
        assert_eq!(forwarded["workflow"], "demo-plugin");
        assert_eq!(forwarded["status"], "succeeded");
        assert_eq!(forwarded["request"]["operation"], "both");
        assert_eq!(forwarded["request"]["content_type"], "application/json");
        assert_eq!(forwarded["request"]["raw_fields"]["batch_size"], 128000);
        assert_eq!(forwarded["request"]["parameters"]["batch_size"], 128000);
        assert_eq!(forwarded["execution"]["status"], "succeeded");
        assert_eq!(forwarded["execution"]["workflow"], "demo-plugin");
        assert_eq!(forwarded["execution"]["output_path"], "/tmp/run-plugin.npz");
        assert_eq!(forwarded["payload"]["echo_operation"], "both");
        assert_eq!(
            forwarded["execution"]["outputs"][0]["name"],
            "pressure_field"
        );
    }

    #[tokio::test]
    async fn postprocess_role_handoffs_result_payload_to_publish_when_publish_is_next() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin(&tmp);
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "parameters": { "batch_size": 128000 },
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "result": {
                    "status": "succeeded",
                    "output_path": "/tmp/run-plugin.npz"
                },
                "output_publication": {
                    "target": {
                        "artifact": "primary",
                        "provider": "s3",
                        "storage": {
                            "type": "s3",
                            "bucket": "bucket",
                            "prefix": "outputs/demo-plugin/run-plugin"
                        }
                    }
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "postprocess"
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
                            "next": "publish"
                        },
                        {
                            "id": "publish",
                            "phase": "publish",
                            "queue": "publish",
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
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "publish");
        assert_eq!(writes[0].stage, "publish");

        let forwarded: Value =
            serde_json::from_str(&writes[0].payload).expect("payload should remain valid JSON");
        assert_eq!(forwarded["stage_context"]["current_stage_id"], "publish");
        assert_eq!(forwarded["stage_context"]["current_phase"], "publish");
        assert_eq!(forwarded["result"]["status"], "succeeded");
        assert_eq!(forwarded["result"]["echo_operation"], "both");
        assert!(forwarded.get("execution").is_none());
    }

    #[test]
    fn postprocess_publish_result_preserves_failed_prior_status_when_hook_omits_status() {
        let prior = json!({
            "status": "failed",
            "error": "execute failed before output"
        });
        let mut result = json!({
            "phase_source": "payload_only"
        });
        let status = result
            .get("status")
            .and_then(Value::as_str)
            .or_else(|| prior.get("status").and_then(Value::as_str))
            .unwrap_or("succeeded")
            .to_string();

        merge_prior_output_fields_into_publish_result(&prior, &mut result)
            .expect("prior fields should merge");
        ensure_result_status(&mut result, status.as_str()).expect("status should be inserted");

        assert_eq!(result["status"], "failed");
        assert_eq!(result["error"], "execute failed before output");
        assert_eq!(result["phase_source"], "payload_only");
    }

    #[test]
    fn postprocess_publish_result_preserves_failed_prior_status_when_hook_status_is_null() {
        let prior = json!({
            "status": "failed",
            "error": "execute failed before output"
        });
        let mut result = json!({
            "status": null,
            "phase_source": "payload_only"
        });
        let status = result
            .get("status")
            .and_then(Value::as_str)
            .or_else(|| prior.get("status").and_then(Value::as_str))
            .unwrap_or("succeeded")
            .to_string();

        merge_prior_output_fields_into_publish_result(&prior, &mut result)
            .expect("prior fields should merge");
        ensure_result_status(&mut result, status.as_str()).expect("status should be inserted");

        assert_eq!(result["status"], "failed");
        assert_eq!(result["error"], "execute failed before output");
        assert_eq!(result["phase_source"], "payload_only");
    }

    #[tokio::test]
    async fn postprocess_role_forwards_failed_status_to_publish_when_hook_status_is_null() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin_with_postprocess(
            &tmp,
            r#"
def postprocess(ctx):
    return {
        "status": None,
        "phase_source": "payload_only"
    }
"#,
        );
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "run",
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "value": 1 },
                    "input_artifacts": []
                },
                "result": {
                    "status": "failed",
                    "error": "execute failed after writing partial output",
                    "output_path": "/tmp/run-plugin.npz",
                    "artifacts": [
                        {
                            "name": "pressure_field",
                            "media_type": "application/x-npz",
                            "storage_path": "/tmp/run-plugin.npz",
                            "primary": true
                        }
                    ],
                    "execution_time_seconds": 1.25
                },
                "output_publication": {
                    "target": {
                        "artifact": "primary",
                        "provider": "s3",
                        "storage": {
                            "type": "s3",
                            "bucket": "bucket",
                            "prefix": "outputs/demo-plugin/run-plugin"
                        }
                    }
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "postprocess"
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
                            "next": "publish"
                        },
                        {
                            "id": "publish",
                            "phase": "publish",
                            "queue": "publish",
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
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "publish");
        assert_eq!(writes[0].stage, "publish");

        let forwarded: Value = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["result"]["status"], "failed");
        assert_eq!(
            forwarded["result"]["error"],
            "execute failed after writing partial output"
        );
        assert_eq!(forwarded["result"]["phase_source"], "payload_only");
        assert_eq!(forwarded["result"]["output_path"], "/tmp/run-plugin.npz");
        assert_eq!(
            forwarded["result"]["artifacts"][0]["name"],
            "pressure_field"
        );
        assert_eq!(forwarded["result"]["execution_time_seconds"], json!(1.25));
        assert!(forwarded.get("execution").is_none());
    }

    #[tokio::test]
    async fn postprocess_role_preserves_prior_execution_outputs_for_payload_only_hook_results() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin_with_postprocess(
            &tmp,
            r#"
def postprocess(ctx):
    return {
        "status": "succeeded",
        "phase_source": "payload_only"
    }
"#,
        );
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "run",
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "value": 1 },
                    "input_artifacts": []
                },
                "result": {
                    "status": "succeeded",
                    "output_path": "/tmp/run-plugin.npz",
                    "artifacts": [
                        {
                            "name": "pressure_field",
                            "media_type": "application/x-npz",
                            "storage_path": "/tmp/run-plugin.npz",
                            "primary": true
                        }
                    ],
                    "execution_time_seconds": 1.25
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "postprocess"
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
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
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let forwarded: Value = serde_json::from_str(&sink.writes()[0].payload).unwrap();
        assert_eq!(forwarded["payload"]["phase_source"], "payload_only");
        assert_eq!(forwarded["execution"]["output_path"], "/tmp/run-plugin.npz");
        assert_eq!(
            forwarded["execution"]["outputs"][0]["name"],
            "pressure_field"
        );
        assert_eq!(
            forwarded["execution"]["execution_time_seconds"],
            json!(1.25)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn postprocess_role_uses_runtime_env_python_for_postprocess_phase() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin(&tmp);
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let manifest = std::fs::read_to_string(plugin_root.join("plugin.yaml"))
            .expect("manifest should be readable")
            .replace(
                "executor_class: python.gpu.physicsnemo\nresources:",
                "executor_class: python.gpu.physicsnemo\n  postprocess_executor_class: python.postprocess.custom\nresources:",
            );
        std::fs::write(plugin_root.join("plugin.yaml"), manifest).expect("manifest should update");
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let fake_runtime = write_fake_python_runtime(
            &tmp,
            "fake-postprocess-python.sh",
            r#"{"status":"succeeded","output_path":"/tmp/runtime-env.npz","artifacts":[{"name":"preview","media_type":"text/html"}],"phase_source":"runtime_env"}"#,
        );

        let mut env = postprocess_env(&runner_path());
        env.python_runtime_envs.insert(
            "python.postprocess.custom".to_string(),
            crate::config::PythonRuntimeEnvConfig {
                python_executable: fake_runtime,
                env: Default::default(),
            },
        );

        let role = PostprocessRole::from_env(&env).expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "parameters": { "batch_size": 128000 },
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "result": {
                    "status": "succeeded",
                    "output_path": "/tmp/run-plugin.npz"
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "postprocess"
                        },
                        {
                            "id": "postprocess",
                            "phase": "postprocess",
                            "queue": "postprocess",
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
                    "executor_class": "python.gpu.physicsnemo",
                    "postprocess_executor_class": "python.postprocess.custom"
                }
            })
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: Value =
            serde_json::from_str(&writes[0].payload).expect("payload should remain valid JSON");
        assert_eq!(forwarded["payload"]["phase_source"], "runtime_env");
        assert_eq!(
            forwarded["execution"]["output_path"],
            "/tmp/runtime-env.npz"
        );
    }

    #[tokio::test]
    async fn postprocess_role_applies_object_store_publish_op() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let publish_root = tmp.path().join("s3-root");
        std::fs::create_dir_all(&publish_root).expect("publish root should exist");
        let plugin_root = write_plugin_with_postprocess(
            &tmp,
            r#"
def postprocess(ctx):
    return {
        "status": "succeeded",
        "output_path": ctx["result"]["output_path"],
        "artifacts": [
            {
                "name": "pressure_field",
                "media_type": "application/x-npz",
                "storage_path": ctx["result"]["output_path"],
            }
        ],
        "result_ops": [
            {
                "kind": "object_store_publish",
                "artifact": "pressure_field",
                "destination_uri": "s3://forecast-bucket/runs/run-plugin/pressure_field.npz"
            }
        ]
    }
"#,
        );
        let plugin_parent = plugin_root.parent().unwrap().to_string_lossy().to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));
        let _s3_root = EnvRestore::set(
            "PHYSICSNEMO_SERVE_OBJECT_STORE_ROOT_S3",
            Some(publish_root.to_string_lossy().as_ref()),
        );

        let output_path = tmp.path().join("pressure_field.npz");
        std::fs::write(&output_path, b"npz-bytes").expect("source artifact should be written");

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "result": {
                    "status": "succeeded",
                    "output_path": output_path.display().to_string()
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {"id": "execute", "phase": "execute", "queue": "execute", "next": "postprocess"},
                        {"id": "postprocess", "phase": "postprocess", "queue": "postprocess", "next": "results"},
                        {"id": "results", "phase": "results", "queue": "results", "next": null}
                    ]
                },
                "runtime": {
                    "kind": "python",
                    "entrypoint": "plugin.py",
                    "executor_class": "python.gpu.physicsnemo"
                }
            })
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let forwarded: Value = serde_json::from_str(&sink.writes()[0].payload).unwrap();
        assert_eq!(
            forwarded["execution"]["published_artifacts"][0]["destination_uri"],
            "s3://forecast-bucket/runs/run-plugin/pressure_field.npz"
        );

        let mirrored = publish_root
            .join("forecast-bucket")
            .join("runs")
            .join("run-plugin")
            .join("pressure_field.npz");
        assert!(
            mirrored.exists(),
            "expected published mirror at {}",
            mirrored.display()
        );
    }

    #[ignore = "requires Python scientific stack (numpy, xarray, zarr, netCDF4)"]
    #[tokio::test]
    async fn postprocess_role_exports_zarr_dataset_to_netcdf_artifact() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let dataset_path = create_tiny_zarr_dataset(&tmp);
        let plugin_root = write_plugin_with_postprocess(
            &tmp,
            r#"
def postprocess(ctx):
    return {
        "status": "succeeded",
        "output_path": ctx["result"]["output_path"],
        "artifacts": [
            {
                "name": "forecast_dataset",
                "media_type": "application/x-zarr",
                "storage_path": ctx["result"]["output_path"],
            }
        ],
        "result_ops": [
            {
                "kind": "dataset_export_netcdf",
                "artifact": "forecast_dataset",
                "target_artifact_name": "forecast_netcdf",
                "filename": "forecast.nc"
            }
        ]
    }
"#,
        );
        let plugin_parent = plugin_root.parent().unwrap().to_string_lossy().to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role = PostprocessRole::from_env(&postprocess_env(&runner_path()))
            .expect("postprocess role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:postprocess",
            "postprocess:grp",
            "run-zarr",
            json!({
                "run_id": "run-zarr",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "result": {
                    "status": "succeeded",
                    "output_path": dataset_path.display().to_string()
                },
                "stage_context": {
                    "current_stage_id": "postprocess",
                    "current_phase": "postprocess",
                    "pipeline": [
                        {"id": "execute", "phase": "execute", "queue": "execute", "next": "postprocess"},
                        {"id": "postprocess", "phase": "postprocess", "queue": "postprocess", "next": "results"},
                        {"id": "results", "phase": "results", "queue": "results", "next": null}
                    ]
                },
                "runtime": {
                    "kind": "python",
                    "entrypoint": "plugin.py",
                    "executor_class": "python.gpu.physicsnemo"
                }
            })
            .to_string(),
            "postprocess",
        );

        role.handle(&msg, "postprocess", &sink).await.unwrap();

        let forwarded: Value = serde_json::from_str(&sink.writes()[0].payload).unwrap();
        assert_eq!(
            forwarded["execution"]["outputs"][1]["name"],
            "forecast_netcdf"
        );
        let generated_path = PathBuf::from(
            forwarded["execution"]["outputs"][1]["storage_path"]
                .as_str()
                .expect("generated artifact should have storage_path"),
        );
        assert!(
            generated_path.exists(),
            "expected generated netcdf artifact"
        );
        let bytes = std::fs::read(&generated_path).expect("generated netcdf should be readable");
        assert!(
            !bytes.is_empty(),
            "expected generated netcdf artifact to contain data"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn postprocess_timeout_terminates_hook_and_dataset_processes() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let hook_pid_file = tmp.path().join("postprocess-hook-timeout.pid");
        let dataset_pid_file = tmp.path().join("postprocess-dataset-timeout.pid");
        let sleeping_hook = write_sleeping_runtime_script(
            &tmp,
            "sleeping-postprocess-hook.sh",
            hook_pid_file.as_path(),
        );
        let sleeping_dataset = write_sleeping_runtime_script(
            &tmp,
            "sleeping-postprocess-dataset.sh",
            dataset_pid_file.as_path(),
        );
        let role = PostprocessRole {
            python_executable: "/bin/sh".to_string(),
            runner_path: PathBuf::from(sleeping_hook),
            dataset_runner_path: PathBuf::from(sleeping_dataset),
            hook_timeout: Duration::from_millis(500),
            input_streams: vec!["postprocess".to_string()],
            python_runtime_envs: Default::default(),
            object_store_roots: ObjectStoreRoots::from_env(),
        };

        let hook_result = role
            .run_postprocess_hook(&json!({"runtime": {"executor_class": "python.gpu.physicsnemo"}}))
            .await;
        assert!(hook_result.is_err(), "hook timeout should return an error");

        let hook_pid = wait_for_pid_file(&hook_pid_file).await;
        let mut hook_alive = process_is_alive(hook_pid);
        for _ in 0..20 {
            if !hook_alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            hook_alive = process_is_alive(hook_pid);
        }
        if hook_alive {
            kill_process(hook_pid);
        }
        assert!(
            !hook_alive,
            "postprocess hook subprocess should be terminated after timeout"
        );

        let dataset_result = role
            .run_dataset_export_runner(
                &json!({"runtime": {"executor_class": "python.gpu.physicsnemo"}}),
                Path::new("/tmp/source.zarr"),
                Path::new("/tmp/output.nc"),
            )
            .await;
        assert!(
            dataset_result.is_err(),
            "dataset timeout should return an error"
        );
        let dataset_pid = wait_for_pid_file(&dataset_pid_file).await;
        let mut dataset_alive = process_is_alive(dataset_pid);
        for _ in 0..20 {
            if !dataset_alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            dataset_alive = process_is_alive(dataset_pid);
        }
        if dataset_alive {
            kill_process(dataset_pid);
        }
        assert!(
            !dataset_alive,
            "dataset export subprocess should be terminated after timeout"
        );
    }

    #[test]
    fn object_store_destination_rejects_parent_dir_segments() {
        let roots = ObjectStoreRoots {
            s3: Some(PathBuf::from("/tmp/root-s3")),
            gcs: None,
            azure: None,
        };
        let result = resolve_object_store_destination("s3://bucket/../escape.txt", &roots);
        assert!(
            result.is_err(),
            "parent-dir segments in destination_uri should be rejected"
        );
    }
}
