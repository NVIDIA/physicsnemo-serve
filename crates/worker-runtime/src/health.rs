/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;
use tracing::{debug, error, info};

use crate::metrics::WorkerMetrics;

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Shared health state with heartbeat-based liveness detection.
///
/// A worker is considered alive only if:
///   1. `mark_alive()` has been called at least once, AND
///   2. The most recent heartbeat is within `liveness_threshold` (if set).
#[derive(Clone)]
pub struct HealthState {
    alive: Arc<AtomicBool>,
    last_heartbeat_ms: Arc<AtomicU64>,
    liveness_threshold_ms: u64,
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthState {
    /// Create a `HealthState` with no heartbeat threshold (legacy behaviour).
    pub fn new() -> Self {
        Self {
            alive: Arc::new(AtomicBool::new(false)),
            last_heartbeat_ms: Arc::new(AtomicU64::new(0)),
            liveness_threshold_ms: 0,
        }
    }

    /// Create a `HealthState` that degrades to unhealthy when no heartbeat
    /// has been recorded within `threshold`.
    pub fn with_liveness_threshold(threshold: Duration) -> Self {
        Self {
            alive: Arc::new(AtomicBool::new(false)),
            last_heartbeat_ms: Arc::new(AtomicU64::new(0)),
            liveness_threshold_ms: threshold.as_millis() as u64,
        }
    }

    /// Signal that the engine has started successfully.
    /// Also records the initial heartbeat.
    pub fn mark_alive(&self) {
        self.alive.store(true, Ordering::Release);
        self.record_heartbeat();
    }

    /// Record a heartbeat (call after each successful engine poll).
    pub fn record_heartbeat(&self) {
        self.last_heartbeat_ms
            .store(now_millis(), Ordering::Release);
    }

    /// Check if the worker is alive and responsive.
    pub fn is_alive(&self) -> bool {
        if !self.alive.load(Ordering::Acquire) {
            return false;
        }
        if self.liveness_threshold_ms == 0 {
            return true;
        }
        let last = self.last_heartbeat_ms.load(Ordering::Acquire);
        now_millis().saturating_sub(last) < self.liveness_threshold_ms
    }
}

const HTTP_200: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
const HTTP_503: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 11\r\n\r\nnot healthy";
const HTTP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found";
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

fn request_path(request: &str) -> Option<&str> {
    let request_line = request.lines().next()?;
    let mut parts = request_line.split_whitespace();
    let _method = parts.next()?;
    parts.next()
}

fn metrics_response(body: String) -> Vec<u8> {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
    .into_bytes()
}

async fn handle_health_connection(
    mut stream: TcpStream,
    state: HealthState,
    metrics: Option<WorkerMetrics>,
) {
    let mut request_buf = [0u8; 1024];
    let read_len = match timeout(HEALTH_REQUEST_TIMEOUT, stream.read(&mut request_buf)).await {
        Ok(Ok(len)) => len,
        Ok(Err(e)) => {
            debug!(error = %e, "health: failed to read request");
            0
        }
        Err(_) => {
            debug!("health: closing idle connection after read timeout");
            if let Err(e) = stream.shutdown().await {
                debug!(error = %e, "health: failed to shutdown idle stream");
            }
            return;
        }
    };
    let request = String::from_utf8_lossy(&request_buf[..read_len]);
    let response: std::borrow::Cow<'_, [u8]> = match request_path(&request) {
        Some("/metrics") => {
            if let Some(metrics) = metrics.as_ref() {
                std::borrow::Cow::Owned(metrics_response(metrics.encode()))
            } else {
                std::borrow::Cow::Borrowed(HTTP_404)
            }
        }
        _ => {
            if state.is_alive() {
                std::borrow::Cow::Borrowed(HTTP_200)
            } else {
                std::borrow::Cow::Borrowed(HTTP_503)
            }
        }
    };
    if let Err(e) = stream.write_all(&response).await {
        debug!(error = %e, "health: failed to write response");
    }
    if let Err(e) = stream.shutdown().await {
        debug!(error = %e, "health: failed to shutdown stream");
    }
}

/// Start a minimal TCP health server on the given listener.
///
/// Returns health responses on regular requests and Prometheus text exposition
/// at `/metrics` when a metrics registry is provided.
/// Runs until the `shutdown` receiver fires or the listener fails.
pub async fn serve_health(
    listener: TcpListener,
    state: HealthState,
    metrics: Option<WorkerMetrics>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let addr = listener.local_addr()?;
    info!(port = addr.port(), "health endpoint listening");

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _addr) = match accept {
                    Ok(conn) => conn,
                    Err(e) => {
                        error!(error = %e, "health listener accept error");
                        continue;
                    }
                };
                let state = state.clone();
                let metrics = metrics.clone();
                tokio::spawn(async move {
                    handle_health_connection(stream, state, metrics).await;
                });
            }
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    info!("health server shutting down");
                    break;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpStream;

    async fn start_health_server(
        state: HealthState,
        metrics: Option<WorkerMetrics>,
    ) -> Option<(
        u16,
        tokio::sync::watch::Sender<bool>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                tracing::warn!("skipping health bind test in sandbox: {err}");
                return None;
            }
            Err(err) => panic!("failed to bind health test listener: {err}"),
        };
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = tokio::sync::watch::channel(false);
        let handle = tokio::spawn(async move { serve_health(listener, state, metrics, rx).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        Some((port, tx, handle))
    }

    async fn get_status(port: u16) -> u16 {
        let resp = get_response(port, "/healthz").await;
        if resp.starts_with("HTTP/1.1 200") {
            200
        } else if resp.starts_with("HTTP/1.1 503") {
            503
        } else {
            panic!("unexpected response: {resp}");
        }
    }

    async fn get_response(port: u16, path: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let request = format!("GET {path} HTTP/1.1\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        String::from_utf8_lossy(&buf[..n]).to_string()
    }

    // --- PR-037: tests use ephemeral ports ---

    #[tokio::test]
    async fn health_returns_503_before_alive() {
        let state = HealthState::new();
        let Some((port, tx, handle)) = start_health_server(state, None).await else {
            return;
        };

        assert_eq!(get_status(port).await, 503);

        tx.send(true).unwrap();
        let _ = handle.await;
    }

    #[tokio::test]
    async fn health_returns_200_after_alive() {
        let state = HealthState::new();
        state.mark_alive();
        let Some((port, tx, handle)) = start_health_server(state, None).await else {
            return;
        };

        assert_eq!(get_status(port).await, 200);

        tx.send(true).unwrap();
        let _ = handle.await;
    }

    #[tokio::test]
    async fn idle_connection_does_not_block_other_health_requests() {
        let state = HealthState::new();
        state.mark_alive();
        let Some((port, tx, handle)) = start_health_server(state, None).await else {
            return;
        };

        let idle_stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;

        let status = tokio::time::timeout(Duration::from_secs(1), get_status(port))
            .await
            .expect("health request should not wait behind an idle connection");
        assert_eq!(status, 200);

        drop(idle_stream);
        tx.send(true).unwrap();
        let _ = handle.await;
    }

    // --- PR-043: heartbeat-based liveness degradation ---

    #[tokio::test]
    async fn health_degrades_when_heartbeat_stale() {
        let state = HealthState::with_liveness_threshold(Duration::from_millis(100));
        state.mark_alive();
        assert!(state.is_alive(), "should be alive right after mark_alive");

        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!state.is_alive(), "should degrade after heartbeat timeout");
    }

    #[tokio::test]
    async fn health_stays_alive_with_fresh_heartbeat() {
        let state = HealthState::with_liveness_threshold(Duration::from_millis(200));
        state.mark_alive();

        tokio::time::sleep(Duration::from_millis(100)).await;
        state.record_heartbeat();

        tokio::time::sleep(Duration::from_millis(100)).await;
        state.record_heartbeat();

        assert!(
            state.is_alive(),
            "should remain alive with regular heartbeats"
        );
    }

    #[tokio::test]
    async fn health_no_threshold_stays_alive_forever() {
        let state = HealthState::new();
        state.mark_alive();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            state.is_alive(),
            "no threshold means always alive once marked"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_exposition() {
        let state = HealthState::new();
        state.mark_alive();
        let metrics = crate::metrics::create_shared_metrics();
        metrics.record_scheduler_attempt("blocked");
        let Some((port, tx, handle)) = start_health_server(state, Some(metrics)).await else {
            return;
        };

        let response = get_response(port, "/metrics").await;

        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.lines().any(
            |line| line == "physicsnemo_serve_scheduler_attempts_total{outcome=\"blocked\"} 1"
        ));

        tx.send(true).unwrap();
        let _ = handle.await;
    }
}
