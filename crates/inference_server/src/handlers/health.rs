/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

/// Lightweight liveness probe - always returns 200 if the process is alive.
pub async fn healthz() -> &'static str {
    "ok"
}

/// Readiness probe - checks Redis connectivity.
/// Returns 200 when the server can accept traffic, 503 otherwise.
pub async fn readyz(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    let service_lock = state.redis_service.read().await;
    match service_lock.as_ref() {
        Some(service) => {
            let mut conn = service.get_connection();
            match redis::cmd("PING").query_async::<String>(&mut conn).await {
                Ok(_) => (StatusCode::OK, Json(json!({"status": "ready"}))),
                Err(e) => {
                    warn!(error=%e, "readyz: Redis PING failed");
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(json!({"status": "not ready", "reason": "Redis unreachable"})),
                    )
                }
            }
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "not ready", "reason": "Redis not initialized"})),
        ),
    }
}
