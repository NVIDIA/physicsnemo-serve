/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::handlers;
use crate::metrics::{ApiDurationLabels, ApiLabels, StatusClass};
use crate::state::AppState;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::{
    Router,
    extract::{DefaultBodyLimit, MatchedPath, Request, State},
    routing::{any, get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

/// Build the router for the HTTP server.
///
/// CORS origins and request body size limit are driven by `AppState.config`.
pub fn build_router(state: Arc<AppState>) -> Router<()> {
    let cors = build_cors_layer(&state.config.cors_allowed_origins);

    Router::new()
        .route("/health", get(handlers::healthz))
        .route("/healthz", get(handlers::healthz))
        .route("/readyz", get(handlers::readyz))
        .route("/doc", get(handlers::get_docs))
        .route("/docs", get(handlers::get_docs))
        .route("/v1/infer/workflows", get(handlers::list_workflows))
        .route("/v1/infer/:name/run", post(handlers::run_workflow))
        .route("/v1/infer/:name/schema", get(handlers::get_workflow_schema))
        .route(
            "/v1/infer/:name/readiness",
            get(handlers::get_workflow_readiness),
        )
        .route("/v1/infer/:workflow/:run_id/status", get(handlers::get_run))
        .route(
            "/v1/infer/:workflow/:run_id/results",
            get(handlers::get_result),
        )
        .route("/openapi.json", get(handlers::get_openapi))
        .route("/openapi", get(handlers::get_openapi))
        .route("/api/openapi.json", get(handlers::get_openapi))
        .route("/v1/openapi.json", get(handlers::get_openapi))
        .route("/swagger.json", get(handlers::get_openapi))
        .route("/metrics", get(handlers::get_metrics))
        .route("/v1/metrics", get(handlers::get_metrics))
        .route("/prometheus/*rest", any(handlers::prometheus_proxy))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics_middleware,
        ))
        .layer(DefaultBodyLimit::max(state.config.max_body_size))
        .layer(cors)
        .with_state(state)
}

/// Axum middleware that records request count and duration for every HTTP request.
async fn metrics_middleware(
    State(state): State<Arc<AppState>>,
    matched: Option<MatchedPath>,
    req: Request,
    next: Next,
) -> axum::response::Response {
    let method = req.method().to_string();
    let path_template = matched
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let start = std::time::Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();

    let status_class = StatusClass::from_status(response.status().as_u16());

    state
        .metrics
        .api_requests_total
        .get_or_create(&ApiLabels {
            method: method.clone(),
            path_template: path_template.clone(),
            status_class,
        })
        .inc();

    state
        .metrics
        .api_request_duration_seconds
        .get_or_create(&ApiDurationLabels {
            method,
            path_template,
        })
        .observe(elapsed);

    response
}

fn build_cors_layer(allowed_origins: &[String]) -> CorsLayer {
    if allowed_origins.is_empty() {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let origins: Vec<HeaderValue> = allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    }
}

/// Build OpenAPI JSON specification
pub fn build_openapi_json() -> Value {
    json!({
        "openapi": "3.0.3",
        "info": build_info(),
        "servers": [{"url": "/", "description": "Current server"}],
        "paths": build_paths(),
        "components": build_components()
    })
}

fn build_info() -> Value {
    json!({
        "title": "PhysicsNeMo Serve Inference API",
        "version": "1.0.0",
        "description": "Generic inference API for manifest-driven model workflows"
    })
}

fn build_paths() -> Value {
    json!({
        "/healthz": {
            "get": {
                "summary": "Health check",
                "responses": {
                    "200": { "description": "OK" }
                }
            }
        },
        "/readyz": {
            "get": {
                "summary": "Readiness probe",
                "description": "Checks Redis connectivity. Returns 200 when the server can accept traffic.",
                "responses": {
                    "200": {
                        "description": "Service is ready",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": { "type": "string", "example": "ready" }
                                    }
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Service not ready (Redis unreachable or not initialized)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": { "type": "string" },
                                        "reason": { "type": "string" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/doc": {
            "get": {
                "summary": "API Documentation (Swagger UI)",
                "description": "Interactive API documentation using Swagger UI",
                "responses": {
                    "200": { "description": "Swagger UI HTML page" }
                }
            }
        },
        "/v1/infer/workflows": {
            "get": {
                "summary": "List all registered plugins",
                "description": "Returns the manifest-driven plugins discovered from the configured plugin directories",
                "responses": {
                    "200": {
                        "description": "List of workflows with schemas",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "workflows": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "display_name": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "version": {"type": "string"},
                                                    "plugin": {"type": "boolean"},
                                                    "readiness": {
                                                        "type": "object",
                                                        "properties": {
                                                            "ready": {"type": "boolean"},
                                                            "recommended_check_phase": {"type": "string"},
                                                            "checks": {
                                                                "type": "array",
                                                                "items": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "type": {"type": "string"},
                                                                        "name": {"type": "string"},
                                                                        "required": {"type": "boolean"},
                                                                        "ok": {"type": "boolean"},
                                                                        "detail": {"type": "string"}
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        "count": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/v1/infer/{name}/run": {
            "post": {
                "summary": "Run a workflow",
                "description": "Enqueue a workflow execution. Returns 202 Accepted when the job is queued. Plugin workflows are gated by readiness checks and return 503 when required dependencies, env vars, or local assets are missing.",
                "parameters": [{
                    "name": "name",
                    "in": "path",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "Name of the workflow to execute"
                }],
                "requestBody": {
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "parameters": { "type": "object" }
                                }
                            }
                        },
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "description": "Plugin-defined multipart form fields and uploaded files"
                            }
                        }
                    }
                },
                "responses": {
                    "202": {
                        "description": "Workflow queued for execution",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "run_id": { "type": "string" },
                                        "workflow": { "type": "string" },
                                        "operation": { "type": "string" },
                                        "status": { "type": "string" },
                                        "pipeline": {
                                            "type": "array",
                                            "items": { "type": "string" }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": { "description": "Workflow not found" },
                    "422": { "description": "Parameter validation failed" },
                    "503": { "description": "Workflow not ready or backend unavailable" }
                }
            }
        },
        "/v1/infer/{name}/schema": {
            "get": {
                "summary": "Get workflow schema",
                "description": "Returns the JSON schema for the specified workflow's parameters",
                "parameters": [{
                    "name": "name",
                    "in": "path",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "Name of the workflow"
                }],
                "responses": {
                    "200": {
                        "description": "Workflow schema",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "workflow_id": {"type": "string"},
                                        "display_name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "version": {"type": "string"},
                                        "content_types": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "default_content_type": {"type": "string"},
                                        "operations": {"type": "object"},
                                        "readiness": {
                                            "type": "object",
                                            "properties": {
                                                "ready": {"type": "boolean"},
                                                "recommended_check_phase": {"type": "string"},
                                                "checks": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "type": {"type": "string"},
                                                            "name": {"type": "string"},
                                                            "required": {"type": "boolean"},
                                                            "ok": {"type": "boolean"},
                                                            "detail": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        "request_schemas": {"type": "object"},
                                        "files": {"type": "array"},
                                        "result_schema": {"type": "object"},
                                        "primary_artifact": {"type": "object"},
                                        "retention_hours": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    },
                    "404": { "description": "Workflow not found" }
                }
            }
        },
        "/v1/infer/{name}/readiness": {
            "get": {
                "summary": "Get workflow readiness",
                "description": "Returns readiness information for the specified manifest-driven plugin.",
                "parameters": [{
                    "name": "name",
                    "in": "path",
                    "required": true,
                    "schema": { "type": "string" },
                    "description": "Name of the workflow"
                }],
                "responses": {
                    "200": {
                        "description": "Workflow readiness",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "workflow_id": {"type": "string"},
                                        "display_name": {"type": "string"},
                                        "version": {"type": "string"},
                                        "plugin": {"type": "boolean"},
                                        "readiness": {
                                            "type": "object",
                                            "properties": {
                                                "ready": {"type": "boolean"},
                                                "recommended_check_phase": {"type": "string"},
                                                "checks": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "type": {"type": "string"},
                                                            "name": {"type": "string"},
                                                            "required": {"type": "boolean"},
                                                            "ok": {"type": "boolean"},
                                                            "detail": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": { "description": "Workflow not found" }
                }
            }
        },
        "/v1/infer/{workflow}/{run_id}/status": {
            "get": {
                "summary": "Get run status",
                "parameters": [
                    {
                        "name": "workflow",
                        "in": "path",
                        "required": true,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "run_id",
                        "in": "path",
                        "required": true,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {"description": "Run details"},
                    "404": {"description": "Not found"},
                    "503": {"description": "Backend unavailable"}
                }
            }
        },
        "/v1/infer/{workflow}/{run_id}/results": {
            "get": {
                "summary": "Get run result",
                "description": "Returns the structured local result envelope `{ request, execution, payload }` by default. Local output paths are exposed under `execution`. When the optional `artifact` query parameter is set, streams the named artifact file instead. Dataset artifacts can also be exported on demand with `format=netcdf` or `format=zarr_zip`, and optionally filtered with `vars=var1,var2`.",
                "parameters": [
                    {
                        "name": "workflow",
                        "in": "path",
                        "required": true,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "run_id",
                        "in": "path",
                        "required": true,
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "artifact",
                        "in": "query",
                        "required": false,
                        "schema": {"type": "string"},
                        "description": "Name of the artifact to stream, for example `primary` or `pressure_field`"
                    },
                    {
                        "name": "format",
                        "in": "query",
                        "required": false,
                        "schema": {"type": "string", "enum": ["netcdf", "zarr_zip"]},
                        "description": "Optional on-demand dataset export format for dataset artifacts"
                    },
                    {
                        "name": "vars",
                        "in": "query",
                        "required": false,
                        "schema": {"type": "string"},
                        "description": "Optional comma-separated variable list used when generating an on-demand dataset export"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Structured result payload or streamed artifact",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["request", "execution", "payload"],
                                    "properties": {
                                        "request": {
                                            "type": "object",
                                            "description": "Normalized request metadata captured for the run"
                                        },
                                        "execution": {
                                            "type": "object",
                                            "description": "Platform-owned execution metadata and local output locations",
                                            "properties": {
                                                "run_id": {"type": "string"},
                                                "status": {"type": "string"},
                                                "workflow": {"type": "string"},
                                                "completed_at": {"type": "string"},
                                                "execution_time_seconds": {"type": "number"},
                                                "output_path": {"type": "string"},
                                                "output_archive": {"type": "string"},
                                                "outputs": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "media_type": {"type": "string"},
                                                            "storage_path": {"type": "string"},
                                                            "filename": {"type": "string"},
                                                            "primary": {"type": "boolean"}
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        "payload": {
                                            "type": "object",
                                            "description": "Plugin-defined result payload"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": {"description": "Result not found or expired"},
                    "503": {"description": "Backend unavailable"}
                }
            }
        },
        "/openapi.json": {
            "get": {
                "summary": "OpenAPI specification",
                "responses": {
                    "200": {"description": "OpenAPI JSON"}
                }
            }
        },
        "/openapi": {
            "get": {
                "summary": "OpenAPI specification (alternate path)",
                "responses": {
                    "200": {"description": "OpenAPI JSON"}
                }
            }
        }
    })
}

fn build_components() -> Value {
    json!({
        "schemas": {}
    })
}

#[cfg(test)]
mod tests;
