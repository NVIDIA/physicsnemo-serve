/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::Context;
use inference_server::artifact_store::ArtifactStore;
use inference_server::config::ServerConfig;
use inference_server::state::AppState;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config = ServerConfig::from_env().context("Failed to load configuration")?;
    info!("Starting PhysicsNeMo Serve Inference Server");
    info!("Config: {}", config.redacted_display());

    let redis_service = inference_server::redis_ops::RedisService::connect(&config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create Redis service: {}", e))?;

    let state = Arc::new(AppState::new(config.clone(), redis_service.clone()));

    let plugin_count = state
        .refresh_plugins()
        .await
        .with_context(|| "Failed to discover configured plugin manifests")?;
    info!("Loaded {} plugin manifest(s)", plugin_count);

    let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);

    start_background_tasks(state.clone(), shutdown_tx.subscribe());
    start_metrics_poller(state.clone(), shutdown_tx.subscribe());

    let app = inference_server::openapi::build_router(state);

    info!("Server listening on {}", config.addr);
    info!("Plugins loaded from configured plugin directories");
    info!("API documentation: http://{}/openapi.json", config.addr);

    let listener = tokio::net::TcpListener::bind(config.addr)
        .await
        .with_context(|| format!("Failed to bind to {}", config.addr))?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            info!("Shutting down background tasks...");
            let _ = shutdown_tx.send(());
        })
        .await
        .context("Server error")?;

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

/// Start background tasks for artifact cleanup.
fn start_background_tasks(
    state: Arc<AppState>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    // Plugin manifests and workflow contracts are loaded once during startup.
    // Workflow changes require a container restart; this background task only
    // handles periodic artifact cleanup.
    let interval_secs = state.config.artifact_cleanup_interval_secs;

    tokio::spawn(async move {
        use tokio::time::interval;
        // Ensure interval is at least 1 second
        let secs = if interval_secs < 1 { 30 } else { interval_secs };
        let mut ticker = interval(Duration::from_secs(secs));

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    if state.config.artifact_retention_hours > 0 {
                        let max_age = Duration::from_secs(state.config.artifact_retention_hours * 3600);
                        let cleanup_roots = unique_cleanup_roots(&state.config);
                        for root in cleanup_roots {
                            let store = ArtifactStore::new(root.clone());
                            match store.cleanup_expired_run_dirs(max_age).await {
                                Ok(removed) if removed > 0 => {
                                    info!(
                                        removed,
                                        artifact_dir = %root.display(),
                                        "Cleaned expired artifact directories"
                                    );
                                }
                                Ok(_) => {}
                                Err(e) => warn!(artifact_dir = %root.display(), "Artifact cleanup failed: {}", e),
                            }
                        }
                    }

                }
                _ = shutdown_rx.recv() => {
                    info!("Stopping artifact cleanup task");
                    break;
                }
            }
        }
    });
}

/// Background task that polls GPU, CPU/system, and Redis metrics every 10s.
fn start_metrics_poller(
    state: Arc<AppState>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    tokio::spawn(async move {
        use tokio::time::interval;

        let mut sys = sysinfo::System::new_all();
        // sysinfo needs two refresh cycles to compute CPU usage deltas;
        // do a warm-up refresh so the first real tick returns valid data.
        tokio::time::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL).await;
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let mut ticker = interval(Duration::from_secs(10));

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    state.metrics.poll_gpu_metrics();

                    sys.refresh_cpu_all();
                    sys.refresh_memory();
                    state.metrics.poll_cpu_metrics(&sys);

                    let redis_guard = state.redis_service.read().await;
                    if let Some(service) = redis_guard.as_ref() {
                        let mut conn = service.get_connection();
                        state
                            .metrics
                            .poll_redis_streams(&mut conn, &state.config.stream_prefix)
                            .await;
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Stopping metrics poller task");
                    break;
                }
            }
        }
    });
}

fn unique_cleanup_roots(config: &ServerConfig) -> Vec<std::path::PathBuf> {
    let mut seen = HashSet::new();
    let mut roots = Vec::new();
    for root in [&config.artifact_dir, &config.default_output_dir] {
        let key = root.to_string_lossy().to_string();
        if seen.insert(key) {
            roots.push(root.clone());
        }
    }
    roots
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Signal received, starting graceful shutdown");
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_server::config::ServerConfig;
    use inference_server::state::AppState;

    fn create_mock_state() -> Arc<AppState> {
        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-main-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-main-outputs-{}",
                uuid::Uuid::new_v4()
            )),
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

    #[tokio::test]
    async fn test_start_background_tasks_shutdown() {
        let state = create_mock_state();
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);

        start_background_tasks(state, shutdown_rx);

        // Send shutdown signal
        assert!(shutdown_tx.send(()).is_ok());
    }

    #[test]
    fn unique_cleanup_roots_deduplicates_artifact_and_output_dirs() {
        let shared_root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-main-shared-root-{}",
            uuid::Uuid::new_v4()
        ));
        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: shared_root.clone(),
            default_output_dir: shared_root.clone(),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: std::collections::HashMap::new(),
        };

        let roots = unique_cleanup_roots(&config);
        assert_eq!(roots, vec![shared_root]);
    }

    #[test]
    fn unique_cleanup_roots_keeps_distinct_artifact_and_output_dirs_in_order() {
        let artifact_root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-main-artifact-root-{}",
            uuid::Uuid::new_v4()
        ));
        let output_root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-main-output-root-{}",
            uuid::Uuid::new_v4()
        ));
        let config = ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: artifact_root.clone(),
            default_output_dir: output_root.clone(),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: std::collections::HashMap::new(),
        };

        let roots = unique_cleanup_roots(&config);
        assert_eq!(roots, vec![artifact_root, output_root]);
    }
}
