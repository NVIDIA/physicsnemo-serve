/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! HTTP request handlers for manifest-driven plugins.

use crate::artifact_store::ArtifactStore;
use crate::plugin_registry::{PluginReadinessReport, RegisteredPlugin};
use crate::redis_ops::RedisService;
use crate::run_envelope::{ArtifactRef, RunEnvelope, RunRequest};
use crate::state::{AppState, CachedWorkflowContract};
use anyhow::Context;
use axum::{
    Json,
    extract::{FromRequest, Multipart, Path, Query, Request, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::{Path as StdPath, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::{fs, io::AsyncWriteExt, process::Command};
use tracing::{error, info, warn};
use uuid::Uuid;

mod docs;
mod health;
mod metrics_handler;
mod results;

pub use docs::{get_docs, get_openapi};
pub use health::{healthz, readyz};
pub use metrics_handler::{get_metrics, prometheus_proxy};
pub use results::{get_result, get_run};

#[cfg(test)]
use self::results::{
    build_artifact_download_response, build_artifact_download_response_with_query,
};

struct PendingUploadedFile {
    field_name: String,
    original_filename: Option<String>,
    media_type: String,
    temp_path: PathBuf,
    size_bytes: u64,
}

struct ParsedPluginRequest {
    content_type: String,
    operation: String,
    raw_fields: Value,
    parameters: Value,
    pending_files: Vec<PendingUploadedFile>,
}

async fn cleanup_pending_uploads(pending_files: &[PendingUploadedFile]) {
    for pending_file in pending_files {
        match fs::remove_file(&pending_file.temp_path).await {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                warn!(
                    path = %pending_file.temp_path.display(),
                    error = %err,
                    "Failed to remove pending upload temp file"
                );
            }
        }
    }
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct ResultQuery {
    artifact: Option<String>,
    format: Option<String>,
    vars: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetFormat {
    Netcdf,
    ZarrZip,
}

impl DatasetFormat {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "netcdf" | "nc" => Ok(Self::Netcdf),
            "zarr_zip" | "zip" | "zarr-zip" => Ok(Self::ZarrZip),
            other => Err(anyhow::anyhow!(
                "unsupported dataset format '{}'; expected 'netcdf' or 'zarr_zip'",
                other
            )),
        }
    }

    fn media_type(self) -> &'static str {
        match self {
            Self::Netcdf => "application/x-netcdf",
            Self::ZarrZip => "application/zip",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Netcdf => "nc",
            Self::ZarrZip => "zip",
        }
    }

    fn runner_op(self) -> &'static str {
        match self {
            Self::Netcdf => "export_netcdf",
            Self::ZarrZip => "export_zarr_zip",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResultArtifactSelection {
    name: String,
    media_type: String,
    filename: String,
    storage_path: PathBuf,
}

/// List all registered plugins
pub async fn list_workflows(State(state): State<Arc<AppState>>) -> Json<Value> {
    let list = state.list_workflow_summaries().await;
    if list.is_empty() {
        return Json(json!({
            "workflows": [],
            "count": 0,
            "message": "No plugins registered yet."
        }));
    }

    Json(json!({
        "workflows": list,
        "count": list.len()
    }))
}

pub(super) fn workflow_not_found_response(name: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": format!("Workflow '{}' not found", name),
            "hint": "Use GET /v1/infer/workflows to see available plugins"
        })),
    )
}

pub(super) fn ensure_workflow_enabled(
    state: &Arc<AppState>,
    name: &str,
) -> Result<(), (StatusCode, Json<Value>)> {
    if state.is_workflow_enabled(name) {
        Ok(())
    } else {
        Err(workflow_not_found_response(name))
    }
}

async fn get_contract_or_404(
    state: &Arc<AppState>,
    name: &str,
) -> Result<CachedWorkflowContract, (StatusCode, Json<Value>)> {
    ensure_workflow_enabled(state, name)?;

    state
        .get_workflow_contract(name)
        .await
        .ok_or_else(|| workflow_not_found_response(name))
}

/// Get workflow schema
pub async fn get_workflow_schema(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> (StatusCode, Json<Value>) {
    match get_contract_or_404(&state, &name).await {
        Ok(contract) => match contract.schema_contract {
            Ok(schema_contract) => (StatusCode::OK, Json(schema_contract)),
            Err(err) => workflow_schema_error_response(err),
        },
        Err(e) => e,
    }
}

/// Get workflow readiness
pub async fn get_workflow_readiness(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> (StatusCode, Json<Value>) {
    match get_contract_or_404(&state, &name).await {
        Ok(contract) => (
            StatusCode::OK,
            Json(build_plugin_readiness_contract(
                &contract.plugin,
                contract.readiness,
            )),
        ),
        Err(e) => e,
    }
}

/// Get the shared `RedisService` from `AppState`.
///
/// Returns `503 Service Unavailable` if the service is not initialized.
/// The `ConnectionManager` inside `RedisService` handles reconnection
/// internally - there is no need for a per-request fallback connection.
async fn get_redis_service(
    state: &Arc<AppState>,
) -> Result<RedisService, (StatusCode, Json<Value>)> {
    let service_lock = state.redis_service.read().await;
    match service_lock.as_ref() {
        Some(service) => Ok(service.clone()),
        None => {
            error!("Redis service not initialized");
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "Redis unavailable - service not initialized"})),
            ))
        }
    }
}

/// Execute a workflow
pub async fn run_workflow(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    request: Request,
) -> (StatusCode, Json<Value>) {
    match get_contract_or_404(&state, &name).await {
        Ok(contract) => run_plugin_workflow(state, contract, request).await,
        Err(e) => e,
    }
}

fn build_plugin_readiness_contract(
    plugin: &RegisteredPlugin,
    readiness: PluginReadinessReport,
) -> Value {
    json!({
        "workflow_id": plugin.manifest.metadata.id,
        "display_name": plugin.manifest.metadata.display_name,
        "version": plugin.manifest.metadata.version,
        "plugin": true,
        "readiness": readiness
    })
}

fn workflow_schema_error_response(details: String) -> (StatusCode, Json<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": "Failed to load workflow schema",
            "details": details
        })),
    )
}

/// Helper to validate parameters against schema
fn validate_parameters(
    schema: &Value,
    parameters: &Value,
    workflow_name: &str,
) -> Result<(), (StatusCode, Json<Value>)> {
    let validator = match jsonschema::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(v) => v,
        Err(e) => {
            error!("Schema compilation error for {}: {}", workflow_name, e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Invalid workflow schema",
                    "details": e.to_string()
                })),
            ));
        }
    };

    if let Err(e) = validator.validate(parameters) {
        // Use iter_errors to get all validation errors
        let errs: Vec<String> = validator
            .iter_errors(parameters)
            .map(|err| err.to_string())
            .collect();

        // If iter_errors returns empty but validate failed, use the main error
        let errs = if errs.is_empty() {
            vec![e.to_string()]
        } else {
            errs
        };

        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({
                "error": "Parameter validation failed",
                "validation_errors": errs,
                "schema": schema,
                "hint": "Check that your parameters match the required schema"
            })),
        ));
    }

    Ok(())
}

fn extract_plugin_operation_and_parameters(
    plugin: &RegisteredPlugin,
    payload: &Value,
) -> Result<(String, Value), (StatusCode, Json<Value>)> {
    let operation = payload
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or(&plugin.manifest.ingress.operations.default)
        .to_string();

    if !plugin
        .manifest
        .ingress
        .operations
        .allowed
        .iter()
        .any(|allowed| allowed == &operation)
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({
                "error": "Unsupported operation",
                "operation": operation,
                "allowed_operations": plugin.manifest.ingress.operations.allowed
            })),
        ));
    }

    let parameters = if let Some(parameters) = payload.get("parameters") {
        parameters.clone()
    } else if let Some(obj) = payload.as_object() {
        let mut params = obj.clone();
        params.remove("operation");
        Value::Object(params)
    } else {
        payload.clone()
    };

    Ok((operation, parameters))
}

fn normalized_content_type(request: &Request) -> Option<String> {
    request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
}

async fn parse_json_body(
    request: Request,
    max_body_size: usize,
) -> Result<Value, (StatusCode, Json<Value>)> {
    let bytes = axum::body::to_bytes(request.into_body(), max_body_size)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "Failed to read request body",
                    "details": e.to_string()
                })),
            )
        })?;

    serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Invalid JSON request body",
                "details": e.to_string()
            })),
        )
    })
}

fn parse_multipart_scalar(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

async fn collect_multipart_plugin_request(
    plugin: &RegisteredPlugin,
    artifact_root: &StdPath,
    request: Request,
) -> Result<ParsedPluginRequest, (StatusCode, Json<Value>)> {
    let mut multipart = Multipart::from_request(request, &()).await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Invalid multipart request",
                "details": e.to_string()
            })),
        )
    })?;

    let file_rules: HashMap<&str, _> = plugin
        .manifest
        .ingress
        .files
        .iter()
        .map(|rule| (rule.name.as_str(), rule))
        .collect();
    let mut seen_files = HashSet::new();
    let mut raw_fields = serde_json::Map::new();
    let mut pending_files = Vec::new();

    macro_rules! multipart_error {
        ($status:expr, $body:expr) => {{
            cleanup_pending_uploads(&pending_files).await;
            return Err(($status, Json($body)));
        }};
    }

    loop {
        let next_field = match multipart.next_field().await {
            Ok(value) => value,
            Err(e) => {
                cleanup_pending_uploads(&pending_files).await;
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": "Failed to read multipart field",
                        "details": e.to_string()
                    })),
                ));
            }
        };
        let Some(field) = next_field else {
            break;
        };
        let field_name = match field.name() {
            Some(name) => name.to_string(),
            None => {
                cleanup_pending_uploads(&pending_files).await;
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": "Multipart field missing name"
                    })),
                ));
            }
        };
        let file_name = field.file_name().map(ToString::to_string);
        let media_type = field
            .content_type()
            .unwrap_or("application/octet-stream")
            .to_string();

        let is_file_field = file_name.is_some() || file_rules.contains_key(field_name.as_str());
        if is_file_field {
            let Some(rule) = file_rules.get(field_name.as_str()) else {
                multipart_error!(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({
                        "error": "Unexpected file field",
                        "field": field_name
                    })
                );
            };

            if !seen_files.insert(field_name.clone()) {
                multipart_error!(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({
                        "error": "Duplicate file field",
                        "field": field_name
                    })
                );
            }

            if !rule
                .media_types
                .iter()
                .any(|allowed| allowed == &media_type)
            {
                multipart_error!(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({
                        "error": "Unsupported file media type",
                        "field": field_name,
                        "media_type": media_type,
                        "allowed_media_types": rule.media_types
                    })
                );
            }

            let max_size_bytes = rule.max_size_mb * 1024 * 1024;
            let incoming_dir = artifact_root.join(".incoming");
            if let Err(e) = fs::create_dir_all(&incoming_dir).await {
                multipart_error!(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({
                        "error": "Failed to prepare upload staging directory",
                        "details": e.to_string()
                    })
                );
            }
            let temp_path = incoming_dir.join(format!("{}.upload", Uuid::new_v4()));
            let mut temp_file = match fs::File::create(&temp_path).await {
                Ok(file) => file,
                Err(e) => {
                    multipart_error!(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        json!({
                            "error": "Failed to create upload staging file",
                            "field": field_name,
                            "details": e.to_string()
                        })
                    );
                }
            };
            let mut field = field;
            let mut size_bytes = 0_u64;
            while let Some(chunk) = match field.chunk().await {
                Ok(value) => value,
                Err(e) => {
                    let _ = fs::remove_file(&temp_path).await;
                    multipart_error!(
                        StatusCode::BAD_REQUEST,
                        json!({
                            "error": "Failed to read uploaded file",
                            "field": field_name,
                            "details": e.to_string()
                        })
                    );
                }
            } {
                size_bytes += chunk.len() as u64;
                if size_bytes > max_size_bytes {
                    let _ = fs::remove_file(&temp_path).await;
                    multipart_error!(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        json!({
                            "error": "Uploaded file exceeds max size",
                            "field": field_name,
                            "max_size_mb": rule.max_size_mb
                        })
                    );
                }
                if let Err(e) = temp_file.write_all(&chunk).await {
                    let _ = fs::remove_file(&temp_path).await;
                    multipart_error!(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        json!({
                            "error": "Failed to stage uploaded file chunk",
                            "field": field_name,
                            "details": e.to_string()
                        })
                    );
                }
            }
            if let Err(e) = temp_file.flush().await {
                let _ = fs::remove_file(&temp_path).await;
                multipart_error!(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({
                        "error": "Failed to finalize uploaded file staging",
                        "field": field_name,
                        "details": e.to_string()
                    })
                );
            }

            pending_files.push(PendingUploadedFile {
                field_name,
                original_filename: file_name,
                media_type,
                temp_path,
                size_bytes,
            });
        } else {
            let text = match field.text().await {
                Ok(value) => value,
                Err(e) => {
                    cleanup_pending_uploads(&pending_files).await;
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": "Failed to read multipart form field",
                            "field": field_name,
                            "details": e.to_string()
                        })),
                    ));
                }
            };
            raw_fields.insert(field_name, parse_multipart_scalar(&text));
        }
    }

    for rule in &plugin.manifest.ingress.files {
        if rule.required && !seen_files.contains(&rule.name) {
            multipart_error!(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({
                    "error": "Missing required file field",
                    "field": rule.name
                })
            );
        }
    }

    let raw_fields = Value::Object(raw_fields);
    let (operation, parameters) = match extract_plugin_operation_and_parameters(plugin, &raw_fields)
    {
        Ok(values) => values,
        Err(response) => {
            cleanup_pending_uploads(&pending_files).await;
            return Err(response);
        }
    };

    Ok(ParsedPluginRequest {
        content_type: "multipart/form-data".to_string(),
        operation,
        raw_fields,
        parameters,
        pending_files,
    })
}

async fn stage_pending_artifacts(
    artifact_dir: std::path::PathBuf,
    run_id: &str,
    pending_files: Vec<PendingUploadedFile>,
) -> Result<Vec<ArtifactRef>, (StatusCode, Json<Value>)> {
    let store = ArtifactStore::new(artifact_dir);
    let mut artifacts = Vec::with_capacity(pending_files.len());

    for pending_file in &pending_files {
        let staged = match store
            .stage_file_from_path(
                run_id,
                &pending_file.field_name,
                pending_file.original_filename.as_deref(),
                Some(&pending_file.media_type),
                &pending_file.temp_path,
                pending_file.size_bytes,
            )
            .await
        {
            Ok(staged) => staged,
            Err(stage_err) => {
                cleanup_pending_uploads(&pending_files).await;
                let rollback_err = store.remove_run_dir(run_id).await.err();
                if let Some(err) = &rollback_err {
                    warn!(
                        run_id,
                        error = %err,
                        "Failed to roll back partially staged artifacts"
                    );
                }
                let details = match rollback_err {
                    Some(err) => format!("{}; rollback also failed: {}", stage_err, err),
                    None => stage_err.to_string(),
                };
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": "Failed to stage uploaded artifact",
                        "details": details
                    })),
                ));
            }
        };

        let default_name = pending_file.field_name.clone();
        artifacts.push(ArtifactRef {
            field_name: staged.field_name.clone(),
            name: staged.original_filename.clone().unwrap_or(default_name),
            artifact_id: staged.artifact_id,
            media_type: staged.media_type,
            size_bytes: staged.size_bytes,
            storage_path: staged.storage_path.display().to_string(),
            original_filename: staged.original_filename,
        });
    }

    Ok(artifacts)
}

async fn rollback_staged_run_artifacts(artifact_dir: &StdPath, run_id: &str) {
    let store = ArtifactStore::new(artifact_dir.to_path_buf());
    if let Err(err) = store.remove_run_dir(run_id).await {
        warn!(
            error = %err,
            %run_id,
            "Failed to roll back staged artifacts after late workflow submission failure"
        );
    }
}

async fn run_plugin_workflow(
    state: Arc<AppState>,
    contract: CachedWorkflowContract,
    request: Request,
) -> (StatusCode, Json<Value>) {
    let plugin = contract.plugin;
    let readiness = contract.readiness;
    if !readiness.ready {
        return plugin_not_ready_response(&plugin, readiness);
    }

    let request_schemas = match contract.request_schemas {
        Ok(request_schemas) => request_schemas,
        Err(err) => {
            error!(
                error = %err,
                workflow = %plugin.manifest.metadata.id,
                "Failed to load cached plugin request schemas"
            );
            return workflow_schema_error_response(err);
        }
    };

    let Some(content_type) = normalized_content_type(&request) else {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(json!({
                "error": "Missing Content-Type header"
            })),
        );
    };

    if !plugin
        .manifest
        .ingress
        .content_types
        .iter()
        .any(|supported| supported == &content_type)
    {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(json!({
                "error": "Workflow does not accept this content type",
                "content_type": content_type,
                "supported_content_types": plugin.manifest.ingress.content_types
            })),
        );
    }

    let parsed_request = match content_type.as_str() {
        "application/json" => {
            let payload = match parse_json_body(request, state.config.max_body_size).await {
                Ok(payload) => payload,
                Err(response) => return response,
            };
            let (operation, parameters) =
                match extract_plugin_operation_and_parameters(&plugin, &payload) {
                    Ok(extracted) => extracted,
                    Err(response) => return response,
                };

            ParsedPluginRequest {
                content_type,
                operation,
                raw_fields: payload,
                parameters,
                pending_files: Vec::new(),
            }
        }
        "multipart/form-data" => {
            match collect_multipart_plugin_request(&plugin, &state.config.artifact_dir, request)
                .await
            {
                Ok(parsed) => parsed,
                Err(response) => return response,
            }
        }
        _ => {
            return (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                Json(json!({
                    "error": "Unsupported content type",
                    "content_type": content_type
                })),
            );
        }
    };

    let request_schema = match request_schemas.get(parsed_request.content_type.as_str()) {
        Some(schema) => schema,
        None => {
            return (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                Json(json!({
                    "error": "Workflow does not provide a request schema for this content type",
                    "content_type": parsed_request.content_type
                })),
            );
        }
    };

    if let Err(response) = validate_parameters(
        request_schema,
        &parsed_request.parameters,
        &plugin.manifest.metadata.id,
    ) {
        cleanup_pending_uploads(&parsed_request.pending_files).await;
        return response;
    }

    let run_id = Uuid::new_v4().to_string();
    let timestamp = get_timestamp();

    let input_artifacts = match stage_pending_artifacts(
        state.config.artifact_dir.clone(),
        &run_id,
        parsed_request.pending_files,
    )
    .await
    {
        Ok(artifacts) => artifacts,
        Err(response) => return response,
    };
    let has_staged_artifacts = !input_artifacts.is_empty();

    let redis_service = match get_redis_service(&state).await {
        Ok(service) => service,
        Err(e) => {
            if has_staged_artifacts {
                rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
            }
            return e;
        }
    };

    let envelope = match RunEnvelope::for_plugin(
        &plugin,
        run_id.clone(),
        parsed_request.operation.clone(),
        RunRequest {
            content_type: parsed_request.content_type.clone(),
            raw_fields: parsed_request.raw_fields,
            input_artifacts,
        },
        parsed_request.parameters,
        state.config.use_prefetch,
    ) {
        Ok(envelope) => envelope,
        Err(e) => {
            if has_staged_artifacts {
                rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
            }
            error!(
                error=%e,
                workflow=%plugin.manifest.metadata.id,
                "Failed to build plugin run envelope"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Failed to build workflow request",
                    "details": e.to_string()
                })),
            );
        }
    };
    let envelope_json = match serde_json::to_value(&envelope) {
        Ok(value) => value,
        Err(e) => {
            if has_staged_artifacts {
                rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
            }
            error!(error=%e, run_id=%run_id, "Failed to serialize plugin run envelope");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Failed to serialize workflow request",
                    "details": e.to_string()
                })),
            );
        }
    };
    let Some(first_stage) = envelope.stage_context.pipeline.first() else {
        if has_staged_artifacts {
            rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
        }
        error!(
            run_id=%run_id,
            workflow=%plugin.manifest.metadata.id,
            "Run envelope missing initial stage context"
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Failed to build workflow request",
                "details": "Run envelope is missing the initial pipeline stage"
            })),
        );
    };

    let workflow_id = plugin.manifest.metadata.id.clone();
    let workflow_version = plugin.manifest.metadata.version.clone();
    let operation = parsed_request.operation.clone();

    if let Err(e) = redis_service
        .store_queued_run(
            &run_id,
            &workflow_id,
            &workflow_version,
            &operation,
            &first_stage.phase,
            &timestamp,
            &timestamp,
        )
        .await
    {
        if has_staged_artifacts {
            rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
        }
        error!(
            error=%e,
            %run_id,
            workflow=%workflow_id,
            "Failed to persist initial queued run record in Redis"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Failed to persist workflow status",
                "message": "The workflow was not accepted because its initial status could not be persisted. Please retry."
            })),
        );
    }

    let prefixed_queue = format!("{}{}", state.config.stream_prefix, first_stage.queue);
    match redis_service
        .enqueue_value_to_stream(&prefixed_queue, &first_stage.phase, &run_id, &envelope_json)
        .await
    {
        Ok(_) => {
            info!(
                %run_id,
                workflow=%workflow_id,
                version=%workflow_version,
                stream=%first_stage.queue,
                stage=%first_stage.phase,
                "Enqueued plugin workflow"
            );
            state.runs.write().await.insert(
                run_id.clone(),
                json!({
                    "workflow": workflow_id.clone(),
                    "version": workflow_version.clone(),
                    "operation": operation.clone(),
                    "status": "queued",
                    "stage": first_stage.phase.clone(),
                    "updated_at": timestamp.clone(),
                    "api_received_at": timestamp.clone(),
                    "api_enqueued_at": timestamp.clone()
                }),
            );

            (
                StatusCode::ACCEPTED,
                Json(json!({
                    "run_id": run_id,
                    "workflow": workflow_id,
                    "operation": operation,
                    "status": "queued",
                    "pipeline": envelope
                        .stage_context
                        .pipeline
                        .iter()
                        .map(|stage| stage.phase.clone())
                        .collect::<Vec<_>>()
                })),
            )
        }
        Err(e) => {
            if let Err(cleanup_err) = redis_service.delete_run_data(&run_id).await {
                warn!(
                    error=%cleanup_err,
                    %run_id,
                    "Failed to remove queued run record after enqueue failure"
                );
            }
            if has_staged_artifacts {
                rollback_staged_run_artifacts(&state.config.artifact_dir, &run_id).await;
            }
            error!(error=%e, run_id=%run_id, "Failed to enqueue plugin workflow");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": "Failed to enqueue workflow",
                    "message": "The workflow was not accepted because the queue backend is unavailable. Please retry."
                })),
            )
        }
    }
}

fn plugin_not_ready_response(
    plugin: &RegisteredPlugin,
    readiness: PluginReadinessReport,
) -> (StatusCode, Json<Value>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": "Workflow is not ready",
            "workflow": plugin.manifest.metadata.id,
            "message": "This plugin is missing required dependencies, environment variables, or local assets. Check the readiness details before retrying.",
            "hint": format!(
                "Inspect GET /v1/infer/{}/schema or run `python scripts/plugin_dev.py check-env {}`",
                plugin.manifest.metadata.id,
                plugin.root_dir.display()
            ),
            "readiness": readiness
        })),
    )
}

/// Helper to get current timestamp as milliseconds since UNIX epoch.
///
/// Returns `"0"` in the astronomically unlikely case the system clock is
/// before the UNIX epoch (avoids `.unwrap()`).
fn get_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
        .to_string()
}

#[cfg(test)]
mod tests;
