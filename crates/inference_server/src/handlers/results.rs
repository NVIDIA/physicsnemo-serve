/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

fn extract_run_workflow_id(run_data: &Value) -> Option<&str> {
    run_data.as_object().and_then(|obj| {
        obj.get("workflow")
            .and_then(Value::as_str)
            .or_else(|| obj.get("workflow_id").and_then(Value::as_str))
    })
}

fn default_dataset_python_executable() -> String {
    for key in ["PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", "PYTHON"] {
        if let Ok(value) = std::env::var(key)
            && !value.trim().is_empty()
        {
            return value;
        }
    }
    "python3".to_string()
}

/// Get status of a specific run.
pub async fn get_run(
    State(state): State<Arc<AppState>>,
    Path((workflow, run_id)): Path<(String, String)>,
) -> (StatusCode, Json<Value>) {
    if let Err(response) = ensure_workflow_enabled(&state, &workflow) {
        return response;
    }

    let mut redis_read_failed = false;
    if let Ok(service) = get_redis_service(&state).await {
        match service.get_run_data(&run_id).await {
            Ok(Some(run_data)) => {
                if extract_run_workflow_id(&run_data) == Some(workflow.as_str()) {
                    return (StatusCode::OK, Json(run_data));
                }
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "Run not found"})),
                );
            }
            Ok(None) => {}
            Err(e) => {
                error!(error=%e, %run_id, "Redis error fetching run status");
                redis_read_failed = true;
            }
        }
    }

    match state.runs.read().await.get(&run_id) {
        Some(v) if extract_run_workflow_id(v) == Some(workflow.as_str()) => {
            (StatusCode::OK, Json(v.clone()))
        }
        None if redis_read_failed => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Service temporarily unavailable",
                "message": "Could not retrieve run status due to a backend error. Please retry."
            })),
        ),
        Some(_) | None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Run not found"})),
        ),
    }
}

/// Get full result payload for a completed run.
pub async fn get_result(
    State(state): State<Arc<AppState>>,
    Path((workflow, run_id)): Path<(String, String)>,
    Query(query): Query<ResultQuery>,
) -> Response {
    if let Err(response) = ensure_workflow_enabled(&state, &workflow) {
        return response.into_response();
    }

    let service = match get_redis_service(&state).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };

    match service.get_result_payload(&run_id).await {
        Ok(Some(mut result)) => {
            if let Ok(Some(run_data)) = service.get_run_data(&run_id).await {
                if extract_run_workflow_id(&run_data) != Some(workflow.as_str()) {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(json!({"error": "Result not found"})),
                    )
                        .into_response();
                }
                merge_run_output_archive(&mut result, &run_data);
            }

            if extract_result_workflow_id(&result).as_deref() != Some(workflow.as_str()) {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "Result not found"})),
                )
                    .into_response();
            }

            if let Some(requested_artifact) = query.artifact.as_deref() {
                let response = if query.format.is_some() {
                    build_artifact_download_response_with_query(
                        &state.config.artifact_dir,
                        &state.config.default_output_dir,
                        &run_id,
                        &result,
                        requested_artifact,
                        &query,
                        state.as_ref(),
                    )
                    .await
                } else {
                    build_artifact_download_response(
                        &state.config.artifact_dir,
                        &state.config.default_output_dir,
                        &run_id,
                        &result,
                        requested_artifact,
                    )
                    .await
                };
                match response {
                    Ok(response) => return response,
                    Err(e) => {
                        warn!(
                            error = %e,
                            %run_id,
                            artifact = requested_artifact,
                            "Failed to resolve requested artifact from result payload"
                        );
                        return (
                            StatusCode::NOT_FOUND,
                            Json(json!({
                                "error": "Artifact not found",
                                "message": e.to_string()
                            })),
                        )
                            .into_response();
                    }
                }
            }

            if query.format.is_some() {
                let requested_artifact = query.artifact.as_deref().unwrap_or("primary");
                match build_artifact_download_response_with_query(
                    &state.config.artifact_dir,
                    &state.config.default_output_dir,
                    &run_id,
                    &result,
                    requested_artifact,
                    &query,
                    state.as_ref(),
                )
                .await
                {
                    Ok(response) => return response,
                    Err(e) => {
                        warn!(
                            error = %e,
                            %run_id,
                            format = query.format.as_deref().unwrap_or(""),
                            artifact = requested_artifact,
                            "Failed to generate requested dataset download from result payload"
                        );
                        return (
                            StatusCode::NOT_FOUND,
                            Json(json!({
                                "error": "Dataset download not available",
                                "message": e.to_string()
                            })),
                        )
                            .into_response();
                    }
                }
            }

            (StatusCode::OK, Json(result)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": "Result not found",
                "message": "Result may not be ready yet, or has expired (24h TTL)"
            })),
        )
            .into_response(),
        Err(e) => {
            error!(error=%e, %run_id, "Redis error fetching result");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": "Service temporarily unavailable",
                    "message": "Could not retrieve result due to a backend error. Please retry."
                })),
            )
                .into_response()
        }
    }
}

pub(crate) async fn build_artifact_download_response(
    artifact_root: &StdPath,
    output_root: &StdPath,
    run_id: &str,
    result: &Value,
    requested_artifact: &str,
) -> anyhow::Result<Response> {
    build_artifact_download_response_with_query(
        artifact_root,
        output_root,
        run_id,
        result,
        requested_artifact,
        &ResultQuery::default(),
        dummy_state_ref(),
    )
    .await
}

fn dummy_state_ref() -> &'static AppState {
    static DUMMY: std::sync::LazyLock<AppState> = std::sync::LazyLock::new(|| {
        AppState::new_for_testing(crate::config::ServerConfig {
            addr: "127.0.0.1:0".parse().expect("dummy addr"),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: PathBuf::from("artifacts"),
            default_output_dir: PathBuf::from("outputs"),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: std::collections::HashMap::new(),
        })
    });
    &DUMMY
}

pub(crate) async fn build_artifact_download_response_with_query(
    artifact_root: &StdPath,
    output_root: &StdPath,
    run_id: &str,
    result: &Value,
    requested_artifact: &str,
    query: &ResultQuery,
    state: &AppState,
) -> anyhow::Result<Response> {
    if let Some(format) = query.format.as_deref() {
        let selected = resolve_result_artifact_path(
            artifact_root,
            output_root,
            run_id,
            result,
            requested_artifact,
            true,
        )
        .await?;
        let generated = materialize_dataset_download(
            output_root,
            run_id,
            result,
            &selected,
            query,
            format,
            state,
        )
        .await?;
        return build_file_download_response(
            &generated.storage_path,
            &generated.media_type,
            &generated.filename,
        )
        .await;
    }
    let selected = resolve_result_artifact(
        artifact_root,
        output_root,
        run_id,
        result,
        requested_artifact,
    )
    .await?;
    build_file_download_response(
        &selected.storage_path,
        &selected.media_type,
        &selected.filename,
    )
    .await
}

async fn build_file_download_response(
    storage_path: &StdPath,
    media_type: &str,
    filename: &str,
) -> anyhow::Result<Response> {
    let body = tokio::fs::read(storage_path).await.with_context(|| {
        format!(
            "failed to read artifact '{}' for download '{}'",
            storage_path.display(),
            filename
        )
    })?;

    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_str(media_type)
            .unwrap_or_else(|_| header::HeaderValue::from_static("application/octet-stream")),
    );
    let disposition = format!("attachment; filename=\"{}\"", filename);
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        header::HeaderValue::from_str(&disposition)
            .unwrap_or_else(|_| header::HeaderValue::from_static("attachment")),
    );
    Ok(response)
}

async fn materialize_dataset_download(
    output_root: &StdPath,
    run_id: &str,
    result: &Value,
    selected: &ResultArtifactSelection,
    query: &ResultQuery,
    format_raw: &str,
    state: &AppState,
) -> anyhow::Result<ResultArtifactSelection> {
    let format = DatasetFormat::parse(format_raw)?;
    let vars = parse_dataset_vars(query.vars.as_deref());
    let derived = dataset_download_path(output_root, run_id, &selected.name, format, &vars);
    if !tokio::fs::try_exists(&derived).await.unwrap_or(false) {
        run_dataset_download_op(result, selected, &derived, format, &vars, state).await?;
    }
    Ok(ResultArtifactSelection {
        name: selected.name.clone(),
        media_type: format.media_type().to_string(),
        filename: derived
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset")
            .to_string(),
        storage_path: derived,
    })
}

fn parse_dataset_vars(vars: Option<&str>) -> Vec<String> {
    vars.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn dataset_download_path(
    output_root: &StdPath,
    run_id: &str,
    artifact_name: &str,
    format: DatasetFormat,
    vars: &[String],
) -> PathBuf {
    let vars_suffix = if vars.is_empty() {
        "all".to_string()
    } else {
        sanitize_filename(&vars.join("_"))
    };
    output_root
        .join(run_id)
        .join("dataset-downloads")
        .join(format!(
            "{}-{}.{}",
            sanitize_filename(artifact_name),
            vars_suffix,
            format.extension()
        ))
}

fn sanitize_filename(input: &str) -> String {
    crate::artifact_store::sanitize_filename(input)
}

async fn run_dataset_download_op(
    result: &Value,
    selected: &ResultArtifactSelection,
    destination_path: &StdPath,
    format: DatasetFormat,
    vars: &[String],
    state: &AppState,
) -> anyhow::Result<()> {
    let (python_executable, envs) = resolve_dataset_runtime(result, state).await;
    let runner_path = crate::plugin_registry::resolve_script_path("dataset_ops_runner.py");
    let payload = serde_json::to_vec(&json!({
        "source_path": selected.storage_path.display().to_string(),
        "destination_path": destination_path.display().to_string(),
        "variables": vars,
    }))
    .context("failed to encode dataset download payload")?;
    let mut command = Command::new(&python_executable);
    command
        .arg(&runner_path)
        .arg("--op")
        .arg(format.runner_op())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !envs.is_empty() {
        command.envs(&envs);
    }
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn dataset runner '{}' with executable '{}'",
            runner_path.display(),
            python_executable
        )
    })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&payload)
            .await
            .context("failed to write dataset payload to runner stdin")?;
    }
    let output = child
        .wait_with_output()
        .await
        .context("failed waiting for dataset runner")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let details = if stderr.is_empty() {
            format!("exit status {}", output.status)
        } else {
            stderr
        };
        return Err(anyhow::anyhow!("dataset runner failed: {details}"));
    }
    Ok(())
}

async fn resolve_dataset_runtime(
    result: &Value,
    state: &AppState,
) -> (String, std::collections::HashMap<String, String>) {
    let workflow = extract_result_workflow_id(result);
    if let Some(workflow_id) = workflow
        && let Some(plugin) = state
            .workflow_registry
            .read()
            .await
            .contracts
            .get(workflow_id.as_str())
            .map(|contract| contract.plugin.clone())
    {
        let runtime_selector = plugin
            .manifest
            .runtime
            .postprocess_executor_class
            .as_deref()
            .or(plugin.manifest.runtime.readiness_executor_class.as_deref())
            .unwrap_or(plugin.manifest.runtime.executor_class.as_str());
        if let Some(runtime_env) = state.config.python_runtime_envs.get(runtime_selector) {
            return (
                runtime_env.python_executable.clone(),
                runtime_env.env.clone(),
            );
        }
    }
    (
        default_dataset_python_executable(),
        std::collections::HashMap::new(),
    )
}

fn extract_result_workflow_id(result: &Value) -> Option<String> {
    let obj = result.as_object()?;
    obj.get("workflow")
        .and_then(Value::as_str)
        .or_else(|| obj.get("workflow_id").and_then(Value::as_str))
        .or_else(|| {
            obj.get("execution")
                .and_then(Value::as_object)
                .and_then(|execution| {
                    execution
                        .get("workflow")
                        .and_then(Value::as_str)
                        .or_else(|| execution.get("workflow_id").and_then(Value::as_str))
                })
        })
        .or_else(|| {
            obj.get("result")
                .and_then(Value::as_object)
                .and_then(|nested| {
                    nested
                        .get("workflow")
                        .and_then(Value::as_str)
                        .or_else(|| nested.get("workflow_id").and_then(Value::as_str))
                })
        })
        .map(ToString::to_string)
}

fn merge_run_output_archive(result: &mut Value, run_data: &Value) {
    let Some(archive) = run_data.get("output_archive").cloned() else {
        return;
    };
    let Some(obj) = result.as_object_mut() else {
        return;
    };
    if let Some(execution) = obj.get_mut("execution").and_then(Value::as_object_mut) {
        if execution
            .get("output_archive")
            .map(Value::is_null)
            .unwrap_or(true)
        {
            execution.insert("output_archive".to_string(), archive);
        }
        return;
    }
    if obj
        .get("output_archive")
        .map(Value::is_null)
        .unwrap_or(true)
    {
        obj.insert("output_archive".to_string(), archive);
    }
}

async fn resolve_result_artifact(
    artifact_root: &StdPath,
    output_root: &StdPath,
    run_id: &str,
    result: &Value,
    requested_artifact: &str,
) -> anyhow::Result<ResultArtifactSelection> {
    resolve_result_artifact_path(
        artifact_root,
        output_root,
        run_id,
        result,
        requested_artifact,
        false,
    )
    .await
}

async fn resolve_result_artifact_path(
    artifact_root: &StdPath,
    output_root: &StdPath,
    run_id: &str,
    result: &Value,
    requested_artifact: &str,
    allow_directories: bool,
) -> anyhow::Result<ResultArtifactSelection> {
    let selection = select_result_artifact(result, requested_artifact)?;
    let storage_path = if allow_directories {
        resolve_download_path_allowing_directories(
            artifact_root,
            &selection.storage_path,
            &[output_root.to_path_buf()],
        )
        .await
    } else {
        let store = ArtifactStore::new(artifact_root.to_path_buf());
        store
            .resolve_download_path_with_additional_roots(
                &selection.storage_path,
                &[output_root.to_path_buf()],
            )
            .await
    }
    .with_context(|| {
        format!(
            "artifact '{}' for run '{}' is unavailable",
            selection.name, run_id
        )
    })?;

    Ok(ResultArtifactSelection {
        storage_path,
        ..selection
    })
}

async fn resolve_download_path_allowing_directories(
    artifact_root: &StdPath,
    candidate: &StdPath,
    additional_roots: &[PathBuf],
) -> anyhow::Result<PathBuf> {
    let candidate_path = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        artifact_root.join(candidate)
    };
    let canonical_candidate = tokio::fs::canonicalize(&candidate_path)
        .await
        .with_context(|| {
            format!(
                "failed to canonicalize artifact path '{}'",
                candidate_path.display()
            )
        })?;

    let mut canonical_roots = Vec::with_capacity(1 + additional_roots.len());
    collect_existing_canonical_root(artifact_root, &mut canonical_roots).await?;
    for root in additional_roots {
        collect_existing_canonical_root(root, &mut canonical_roots).await?;
    }
    if canonical_roots.is_empty() {
        anyhow::bail!("no configured download roots exist on disk");
    }
    if !canonical_roots
        .iter()
        .any(|root| canonical_candidate.starts_with(root))
    {
        let allowed_roots = canonical_roots
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "artifact path '{}' is outside the configured download roots [{}]",
            canonical_candidate.display(),
            allowed_roots
        );
    }
    Ok(canonical_candidate)
}

async fn collect_existing_canonical_root(
    root: &StdPath,
    output: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    match tokio::fs::canonicalize(root).await {
        Ok(path) => output.push(path),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err).with_context(|| {
                format!("failed to canonicalize artifact root '{}'", root.display())
            });
        }
    }
    Ok(())
}

fn select_result_artifact(
    result: &Value,
    requested_artifact: &str,
) -> anyhow::Result<ResultArtifactSelection> {
    let result_object = result
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("result payload must be a JSON object"))?;
    let execution = result_object.get("execution").and_then(Value::as_object);
    let nested_result = result_object.get("result").and_then(Value::as_object);
    let output_path = result_object
        .get("output_path")
        .and_then(Value::as_str)
        .or_else(|| execution.and_then(|value| value.get("output_path").and_then(Value::as_str)))
        .or_else(|| {
            nested_result.and_then(|value| value.get("output_path").and_then(Value::as_str))
        });
    let output_archive = result_object
        .get("output_archive")
        .and_then(Value::as_str)
        .or_else(|| execution.and_then(|value| value.get("output_archive").and_then(Value::as_str)))
        .or_else(|| {
            nested_result.and_then(|value| value.get("output_archive").and_then(Value::as_str))
        });
    let artifacts = result_object
        .get("artifacts")
        .and_then(Value::as_array)
        .filter(|entries| !entries.is_empty())
        .or_else(|| execution.and_then(|value| value.get("outputs").and_then(Value::as_array)))
        .filter(|entries| !entries.is_empty())
        .or_else(|| execution.and_then(|value| value.get("artifacts").and_then(Value::as_array)))
        .filter(|entries| !entries.is_empty())
        .or_else(|| {
            nested_result.and_then(|value| value.get("artifacts").and_then(Value::as_array))
        });

    if matches!(requested_artifact, "archive" | "output_archive") {
        let archive_path = output_archive.ok_or_else(|| {
            anyhow::anyhow!("archive artifact is not present in the result payload")
        })?;
        return Ok(selection_from_path(
            "output_archive",
            "application/octet-stream",
            archive_path,
        ));
    }

    if let Some(artifact_entries) = artifacts {
        let requested_entry = if requested_artifact == "primary" {
            artifact_entries.first()
        } else {
            artifact_entries
                .iter()
                .find(|entry| entry.get("name").and_then(Value::as_str) == Some(requested_artifact))
        };

        if let Some(entry) = requested_entry {
            return selection_from_artifact_entry(
                entry,
                output_path,
                requested_artifact == "primary" || artifact_entries.len() == 1,
            );
        }
    }

    if requested_artifact == "primary"
        && let Some(path) = output_path
    {
        return Ok(selection_from_path(
            "primary",
            "application/octet-stream",
            path,
        ));
    }

    Err(anyhow::anyhow!(
        "artifact '{}' is not present in the result payload",
        requested_artifact
    ))
}

fn selection_from_artifact_entry(
    entry: &Value,
    output_path_fallback: Option<&str>,
    allow_output_path_fallback: bool,
) -> anyhow::Result<ResultArtifactSelection> {
    let name = entry
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("artifact entry is missing name"))?;
    let media_type = entry
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    let storage_path = entry
        .get("storage_path")
        .or_else(|| entry.get("path"))
        .or_else(|| entry.get("output_path"))
        .and_then(Value::as_str)
        .or({
            if allow_output_path_fallback {
                output_path_fallback
            } else {
                None
            }
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "artifact '{}' does not include a downloadable storage path",
                name
            )
        })?;

    let filename = entry
        .get("filename")
        .or_else(|| entry.get("original_filename"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| derive_filename(storage_path, name));

    Ok(ResultArtifactSelection {
        name: name.to_string(),
        media_type: media_type.to_string(),
        filename,
        storage_path: PathBuf::from(storage_path),
    })
}

fn selection_from_path(
    name: &str,
    media_type: &str,
    storage_path: &str,
) -> ResultArtifactSelection {
    ResultArtifactSelection {
        name: name.to_string(),
        media_type: media_type.to_string(),
        filename: derive_filename(storage_path, name),
        storage_path: PathBuf::from(storage_path),
    }
}

fn derive_filename(storage_path: &str, fallback: &str) -> String {
    StdPath::new(storage_path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::redis_ops::RedisService;
    use crate::state::AppState;
    use axum::body::to_bytes;
    use serde_json::json;
    use std::fs;
    use std::net::TcpListener;
    use std::process::{Child, Command, Stdio};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    static PYTHON_ENV_LOCK: Mutex<()> = Mutex::const_new(());

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.previous {
                    Some(prev) => std::env::set_var(self.key, prev),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    fn unique_temp_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "physicsnemo-serve-results-tests-{test_name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("temp directory should be created");
        dir
    }

    fn create_mock_state() -> Arc<AppState> {
        let config = ServerConfig {
            addr: "127.0.0.1:0".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: unique_temp_dir("artifacts"),
            default_output_dir: unique_temp_dir("outputs"),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: std::collections::HashMap::new(),
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
                "physicsnemo-serve-results-redis-{test_name}-{}",
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

    async fn create_connected_state(test_name: &str) -> (TestRedisServer, Arc<AppState>) {
        let port = reserve_port();
        let server = TestRedisServer::spawn(test_name, port).await;
        let config = ServerConfig {
            addr: "127.0.0.1:0".parse().unwrap(),
            redis_url: format!("redis://127.0.0.1:{port}"),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: unique_temp_dir(&format!("{test_name}-artifacts")),
            default_output_dir: unique_temp_dir(&format!("{test_name}-outputs")),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: std::collections::HashMap::new(),
        };
        let redis_service = RedisService::connect(&config)
            .await
            .expect("redis service should connect");
        (server, Arc::new(AppState::new(config, redis_service)))
    }

    async fn seed_run_and_result(
        state: &Arc<AppState>,
        run_id: &str,
        workflow: &str,
        result: &Value,
    ) {
        let redis_service = state
            .redis_service
            .read()
            .await
            .as_ref()
            .expect("redis service should be initialized")
            .clone();
        let mut conn = redis_service.get_connection();
        let run_key = format!("run:{run_id}");
        let result_key = format!("result:{run_id}");
        let result_json = serde_json::to_string(result).expect("result json should serialize");
        let _: usize = redis::cmd("HSET")
            .arg(&run_key)
            .arg("workflow")
            .arg(workflow)
            .query_async(&mut conn)
            .await
            .expect("run hash should be seeded");
        let _: () = redis::cmd("SETEX")
            .arg(&result_key)
            .arg(86_400)
            .arg(result_json)
            .query_async(&mut conn)
            .await
            .expect("result payload should be seeded");
    }

    #[test]
    fn extract_result_workflow_id_supports_nested_result_workflow_id() {
        let payload = json!({
            "result": {
                "workflow_id": "demo-prefetch"
            }
        });

        assert_eq!(
            extract_result_workflow_id(&payload).as_deref(),
            Some("demo-prefetch")
        );
    }

    #[test]
    fn extract_result_workflow_id_supports_execution_workflow_field() {
        let payload = json!({
            "execution": {
                "workflow": "demo-prefetch"
            }
        });

        assert_eq!(
            extract_result_workflow_id(&payload).as_deref(),
            Some("demo-prefetch")
        );
    }

    #[test]
    fn select_result_artifact_uses_output_path_fallback_for_single_named_artifact() {
        let selection = select_result_artifact(
            &json!({
                "output_path": "/tmp/forecast.nc",
                "artifacts": [
                    {
                        "name": "forecast_dataset",
                        "media_type": "application/x-netcdf"
                    }
                ]
            }),
            "forecast_dataset",
        )
        .expect("single named artifact should use output_path fallback");

        assert_eq!(selection.name, "forecast_dataset");
        assert_eq!(selection.media_type, "application/x-netcdf");
        assert_eq!(selection.filename, "forecast.nc");
        assert_eq!(selection.storage_path, PathBuf::from("/tmp/forecast.nc"));
    }

    #[test]
    fn select_result_artifact_rejects_missing_storage_path_for_nonprimary_artifact() {
        let err = select_result_artifact(
            &json!({
                "output_path": "/tmp/forecast.nc",
                "artifacts": [
                    {
                        "name": "forecast_dataset",
                        "media_type": "application/x-netcdf"
                    },
                    {
                        "name": "secondary",
                        "media_type": "application/octet-stream",
                        "storage_path": "/tmp/secondary.bin"
                    }
                ]
            }),
            "forecast_dataset",
        )
        .expect_err("nonprimary artifact without storage path must fail");

        assert!(
            err.to_string()
                .contains("does not include a downloadable storage path")
        );
    }

    #[test]
    fn select_result_artifact_reads_archive_from_nested_result_payload() {
        let selection = select_result_artifact(
            &json!({
                "result": {
                    "output_archive": "/tmp/archive.zip"
                }
            }),
            "output_archive",
        )
        .expect("nested output_archive should be selected");

        assert_eq!(selection.name, "output_archive");
        assert_eq!(selection.media_type, "application/octet-stream");
        assert_eq!(selection.filename, "archive.zip");
        assert_eq!(selection.storage_path, PathBuf::from("/tmp/archive.zip"));
    }

    #[test]
    fn select_result_artifact_rejects_non_object_payload() {
        let err = select_result_artifact(&json!(["not", "an", "object"]), "primary")
            .expect_err("non-object payload must fail");

        assert!(
            err.to_string()
                .contains("result payload must be a JSON object")
        );
    }

    #[test]
    fn select_result_artifact_prefers_original_filename_when_filename_missing() {
        let selection = select_result_artifact(
            &json!({
                "artifacts": [
                    {
                        "name": "forecast_dataset",
                        "media_type": "application/x-zarr",
                        "original_filename": "forecast.zarr",
                        "storage_path": "/tmp/generated-forecast"
                    }
                ]
            }),
            "forecast_dataset",
        )
        .expect("artifact should use original filename");

        assert_eq!(selection.filename, "forecast.zarr");
    }

    #[test]
    fn selection_from_path_falls_back_to_artifact_name_when_path_has_no_filename() {
        let selection = selection_from_path("archive", "application/octet-stream", "/");

        assert_eq!(selection.filename, "archive");
    }

    #[test]
    fn parse_dataset_vars_trims_and_filters_empty_values() {
        let vars = parse_dataset_vars(Some(" temperature , ,humidity, pressure ,, "));

        assert_eq!(vars, vec!["temperature", "humidity", "pressure"]);
    }

    #[tokio::test]
    async fn build_file_download_response_read_error_mentions_download_name_not_run() {
        let missing_path = unique_temp_dir("download-read-error").join("missing-dataset.zarr");

        let err = build_file_download_response(&missing_path, "application/x-zarr", "dataset.zarr")
            .await
            .expect_err("missing file should fail the download response");

        let message = format!("{err:#}");
        assert!(
            message.contains("failed to read artifact '"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("for download 'dataset.zarr'"),
            "unexpected message: {message}"
        );
        assert!(
            !message.contains("for run 'dataset.zarr'"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn dataset_download_path_uses_all_suffix_when_vars_empty() {
        let path = dataset_download_path(
            StdPath::new("/tmp/output-root"),
            "run-42",
            "forecast dataset",
            DatasetFormat::Netcdf,
            &[],
        );

        assert_eq!(
            path,
            PathBuf::from("/tmp/output-root/run-42/dataset-downloads/forecast_dataset-all.nc")
        );
    }

    #[tokio::test]
    async fn get_run_uses_in_memory_state_when_redis_is_uninitialized() {
        let state = create_mock_state();
        state.runs.write().await.insert(
            "run-1".to_string(),
            json!({
                "workflow_id": "demo-prefetch",
                "status": "completed"
            }),
        );

        let (status, body) = get_run(
            State(state),
            Path(("demo-prefetch".to_string(), "run-1".to_string())),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.0["status"], "completed");
    }

    #[tokio::test]
    async fn get_run_returns_not_found_for_workflow_mismatch_in_memory_state() {
        let state = create_mock_state();
        state.runs.write().await.insert(
            "run-2".to_string(),
            json!({
                "workflow_id": "other-workflow",
                "status": "completed"
            }),
        );

        let (status, body) = get_run(
            State(state),
            Path(("demo-prefetch".to_string(), "run-2".to_string())),
        )
        .await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body.0, json!({"error": "Run not found"}));
    }

    #[tokio::test]
    async fn get_result_returns_service_unavailable_when_redis_is_uninitialized() {
        let state = create_mock_state();

        let response = get_result(
            State(state),
            Path(("demo-prefetch".to_string(), "run-3".to_string())),
            Query(ResultQuery::default()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let payload: Value = serde_json::from_slice(&body).expect("body should be json");
        assert_eq!(
            payload,
            json!({"error": "Redis unavailable - service not initialized"})
        );
    }

    #[tokio::test]
    async fn get_result_returns_structured_envelope_from_redis() {
        let (_redis_server, state) = create_connected_state("structured-envelope").await;
        let result = json!({
            "request": {
                "operation": "run",
                "content_type": "application/json",
                "raw_fields": {
                    "batch_size": 128000
                },
                "input_artifacts": []
            },
            "execution": {
                "run_id": "run-structured",
                "workflow": "demo-prefetch",
                "status": "succeeded",
                "output_path": "/outputs/run-structured/result.nc",
                "outputs": [
                    {
                        "name": "primary",
                        "media_type": "application/x-netcdf",
                        "storage_path": "/outputs/run-structured/result.nc",
                        "primary": true
                    }
                ]
            },
            "payload": {
                "echo_operation": "both"
            }
        });
        seed_run_and_result(&state, "run-structured", "demo-prefetch", &result).await;

        let response = get_result(
            State(state),
            Path(("demo-prefetch".to_string(), "run-structured".to_string())),
            Query(ResultQuery::default()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        let payload: Value = serde_json::from_slice(&body).expect("body should be json");
        assert_eq!(payload, result);
    }

    #[tokio::test]
    async fn get_result_downloads_primary_artifact_from_execution_outputs() {
        let (_redis_server, state) = create_connected_state("execution-outputs").await;
        let output_path = state
            .config
            .default_output_dir
            .join("run-artifact")
            .join("forecast.nc");
        tokio::fs::create_dir_all(
            output_path
                .parent()
                .expect("output file should have a parent directory"),
        )
        .await
        .expect("output directory should be created");
        tokio::fs::write(&output_path, b"netcdf-bytes")
            .await
            .expect("artifact bytes should be written");

        let result = json!({
            "request": {
                "operation": "run"
            },
            "execution": {
                "run_id": "run-artifact",
                "workflow": "demo-prefetch",
                "status": "succeeded",
                "output_path": output_path.display().to_string(),
                "outputs": [
                    {
                        "name": "forecast_dataset",
                        "media_type": "application/x-netcdf",
                        "storage_path": output_path.display().to_string(),
                        "primary": true
                    }
                ]
            },
            "payload": {
                "echo_operation": "download"
            }
        });
        seed_run_and_result(&state, "run-artifact", "demo-prefetch", &result).await;

        let response = get_result(
            State(state),
            Path(("demo-prefetch".to_string(), "run-artifact".to_string())),
            Query(ResultQuery {
                artifact: Some("primary".to_string()),
                format: None,
                vars: None,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/x-netcdf")
        );
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        assert_eq!(&body[..], b"netcdf-bytes");
    }

    #[tokio::test]
    async fn resolve_download_path_allowing_directories_accepts_relative_directory_under_root() {
        let artifact_root = unique_temp_dir("relative-dir");
        let relative_dir = PathBuf::from("run-1").join("dataset.zarr");
        let absolute_dir = artifact_root.join(&relative_dir);
        tokio::fs::create_dir_all(&absolute_dir)
            .await
            .expect("directory should be created");

        let resolved =
            resolve_download_path_allowing_directories(&artifact_root, &relative_dir, &[])
                .await
                .expect("directory under artifact root should be allowed");

        assert_eq!(
            resolved,
            tokio::fs::canonicalize(&absolute_dir)
                .await
                .expect("directory should canonicalize")
        );
    }

    #[tokio::test]
    async fn resolve_download_path_allowing_directories_errors_when_no_roots_exist() {
        let candidate_dir = unique_temp_dir("existing-candidate").join("dataset.zarr");
        tokio::fs::create_dir_all(&candidate_dir)
            .await
            .expect("candidate directory should exist");
        let missing_artifact_root = candidate_dir.join("..").join("missing-artifacts");
        let missing_output_root = candidate_dir.join("..").join("missing-outputs");

        let err = resolve_download_path_allowing_directories(
            &missing_artifact_root,
            &candidate_dir,
            &[missing_output_root],
        )
        .await
        .expect_err("missing roots should fail even when candidate exists");

        assert!(
            err.to_string()
                .contains("no configured download roots exist on disk")
        );
    }

    #[tokio::test]
    async fn resolve_dataset_runtime_prefers_physicsnemo_serve_python_env_fallback() {
        let _env_lock = PYTHON_ENV_LOCK.lock().await;
        let _python_guard = EnvGuard::set("PYTHON", "/usr/bin/python3");
        let _physicsnemo_serve_guard = EnvGuard::set(
            "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE",
            "/opt/physicsnemo-serve-venv/bin/python",
        );

        let state = create_mock_state();
        let (python_executable, envs) =
            resolve_dataset_runtime(&json!({"workflow": "missing"}), &state).await;

        assert_eq!(python_executable, "/opt/physicsnemo-serve-venv/bin/python");
        assert!(envs.is_empty());
    }
}
