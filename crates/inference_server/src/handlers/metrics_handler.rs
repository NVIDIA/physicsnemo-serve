/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `/metrics` endpoint and `/prometheus/*` reverse proxy.

use crate::state::AppState;
use axum::{
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tracing::{debug, warn};

/// Serve Prometheus text exposition format from the in-process registry.
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    debug!("GET /v1/metrics requested");
    let body = crate::metrics::encode_metrics(&state.metrics);
    debug!(body_bytes = body.len(), "GET /v1/metrics encoded");
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// Reverse proxy to the in-container Prometheus instance.
///
/// Forwards `GET/POST /prometheus/{rest}?query=...` to
/// `http://localhost:9090/{rest}?query=...` so that Grafana running outside
/// the container can reach Prometheus through the single exposed API port.
pub async fn prometheus_proxy(
    Path(rest): Path<String>,
    headers: HeaderMap,
    req: Request,
) -> Response {
    let base_url =
        std::env::var("PROMETHEUS_URL").unwrap_or_else(|_| "http://localhost:9090".to_string());

    let query_string = req.uri().query().unwrap_or("");
    let target = if query_string.is_empty() {
        format!("{}/{}", base_url.trim_end_matches('/'), rest)
    } else {
        format!(
            "{}/{}?{}",
            base_url.trim_end_matches('/'),
            rest,
            query_string
        )
    };

    let method = req.method().clone();
    debug!(%method, %target, "Prometheus proxy request received");

    let body = match axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            warn!(%target, error = %e, "Prometheus proxy failed to read request body");
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to read request body: {e}"),
            )
                .into_response();
        }
    };

    let client = reqwest::Client::new();
    let mut builder = client.request(method, &target);
    for (key, value) in headers.iter() {
        if key == axum::http::header::HOST {
            continue;
        }
        builder = builder.header(key, value);
    }
    builder = builder.body(body);

    match builder.send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            debug!(%target, %status, "Prometheus proxy response");
            let mut response_headers = HeaderMap::new();
            for (key, value) in resp.headers().iter() {
                if let Ok(v) = value.to_str()
                    && let Ok(hv) = v.parse()
                {
                    response_headers.insert(key.clone(), hv);
                }
            }
            let response_body = resp.bytes().await.unwrap_or_default();
            (status, response_headers, response_body).into_response()
        }
        Err(e) => {
            warn!(%target, error = %e, "Prometheus proxy upstream unreachable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Prometheus proxy error: {e}"),
            )
                .into_response()
        }
    }
}
