/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use scicomp_rq::QueueManager;
use tokio::sync::watch;
use tracing::info;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use worker_runtime::config::resolve_config_and_role;
use worker_runtime::engine::EngineBuilder;
use worker_runtime::health::{HealthState, serve_health};
use worker_runtime::metrics::{WorkerMetrics, create_shared_metrics};
use worker_runtime::retry_dlq::RetryDlqPolicy;
use worker_runtime::roles::results::RedisResultsPersistence;
use worker_runtime::roles::{RoleProviders, build_role};
use worker_runtime::transport::redis::RedisTransport;

/// CLI arguments for the worker-runtime binary.
///
/// Both `--role` and `--config-path` are optional when the corresponding
/// environment variables (`WORKER_ROLE`, `WORKER_PIPELINE_CONFIG`) are set.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    role: Option<String>,
    config_path: Option<PathBuf>,
}

fn parse_args_from<I, S>(raw_args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut role: Option<String> = None;
    let mut config_path: Option<PathBuf> = None;

    let mut args = raw_args.into_iter().map(Into::into);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--role" => {
                role = Some(
                    args.next()
                        .ok_or_else(|| anyhow!("missing value for --role"))?,
                );
            }
            "--config-path" => {
                config_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("missing value for --config-path"))?,
                ));
            }
            unsupported => {
                return Err(anyhow!(
                    "unsupported argument: {unsupported}. Supports: --role, --config-path"
                ));
            }
        }
    }

    Ok(CliArgs { role, config_path })
}

fn parse_args() -> Result<CliArgs> {
    parse_args_from(std::env::args().skip(1))
}

fn consumer_name(role: &str) -> String {
    format!("{role}-{}-{}", std::process::id(), Uuid::new_v4())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn role_requires_queue_manager(role_name: &str) -> bool {
    matches!(role_name, "batch" | "collect" | "fanout" | "scheduler")
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let args = parse_args()?;

    let (config, role_name) =
        resolve_config_and_role(args.config_path.as_deref(), args.role.as_deref())?;
    config.validate()?;

    let env = config.resolve_env(&role_name)?;
    info!(
        role = %env.role_name,
        inputs = ?env.inputs.iter().map(|i| i.stream.as_str()).collect::<Vec<_>>(),
        outputs = ?env.resolved_outputs,
        "startup: config resolved"
    );

    let qm = QueueManager::from_env().await?;
    let transport = Arc::new(RedisTransport::new(qm.clone(), &config.stream_prefix));

    let scheduler_metrics: Option<WorkerMetrics> = if role_name == "scheduler" {
        Some(create_shared_metrics())
    } else {
        None
    };

    let mut providers = RoleProviders::empty();
    if role_name == "results" {
        providers =
            providers.with_results_persistence(Arc::new(RedisResultsPersistence::new(qm.clone())));
    }
    if role_requires_queue_manager(role_name.as_str()) {
        providers = providers.with_queue_manager(qm.clone());
    }
    if role_name == "scheduler" {
        providers = providers.with_retry_dlq_policy(RetryDlqPolicy::new(
            config.max_retries,
            config.shared_dlq_stream.clone(),
        ));
        if let Some(metrics) = scheduler_metrics.as_ref() {
            providers = providers.with_metrics(metrics.clone());
        }
    }

    let (role, tasks) = build_role(&env, providers)?;
    let consumer = consumer_name(&role_name);

    let mut builder = EngineBuilder::new(&config, &role_name)
        .transport(transport)
        .role(role)
        .consumer(&consumer);
    for task in tasks {
        builder = builder.background_task(task);
    }
    let engine = builder.build()?;

    info!(
        role = %env.role_name,
        consumer = %consumer,
        "engine built, starting poll loop"
    );

    let health_state = HealthState::with_liveness_threshold(std::time::Duration::from_secs(120));

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    if let Ok(port_str) = std::env::var("HEALTH_PORT")
        && let Ok(port) = port_str.parse::<u16>()
    {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
            .await
            .context(format!("failed to bind health endpoint on port {port}"))?;
        let h = health_state.clone();
        let metrics = scheduler_metrics.clone();
        let rx = shutdown_rx.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_health(listener, h, metrics, rx).await {
                tracing::error!(error = %e, "health server failed");
            }
        });
    }

    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            info!("shutdown signal received");
            let _ = shutdown_tx.send(true);
        }
    });

    let stats = engine
        .run_until_shutdown(shutdown_rx, Some(health_state))
        .await?;
    info!(
        iterations = stats.iterations,
        polled = stats.polled,
        succeeded = stats.succeeded,
        failed = stats.failed,
        "engine stopped"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_both_args() {
        let args = parse_args_from(["--role", "scheduler", "--config-path", "/tmp/cfg.json"])
            .expect("valid args should parse");
        assert_eq!(args.role.as_deref(), Some("scheduler"));
        assert_eq!(
            args.config_path.as_deref(),
            Some(std::path::Path::new("/tmp/cfg.json"))
        );
    }

    #[test]
    fn cli_allows_role_only() {
        let args = parse_args_from(["--role", "prefetch"]).expect("role-only should parse");
        assert_eq!(args.role.as_deref(), Some("prefetch"));
        assert!(args.config_path.is_none());
    }

    #[test]
    fn cli_allows_config_path_only() {
        let args =
            parse_args_from(["--config-path", "config.json"]).expect("config-only should parse");
        assert!(args.role.is_none());
        assert_eq!(
            args.config_path.as_deref(),
            Some(std::path::Path::new("config.json"))
        );
    }

    #[test]
    fn cli_allows_no_args() {
        let args = parse_args_from(Vec::<String>::new()).expect("no args should parse");
        assert!(args.role.is_none());
        assert!(args.config_path.is_none());
    }

    #[test]
    fn cli_rejects_unsupported_args() {
        let result = parse_args_from(["--role", "prefetch", "--redis-url", "redis://localhost"]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn consumer_name_is_unique() {
        let a = consumer_name("prefetch");
        let b = consumer_name("prefetch");
        assert_ne!(a, b);
    }

    #[test]
    fn role_requires_queue_manager_for_shared_state_roles() {
        assert!(role_requires_queue_manager("batch"));
        assert!(role_requires_queue_manager("collect"));
        assert!(role_requires_queue_manager("fanout"));
        assert!(role_requires_queue_manager("scheduler"));
        assert!(!role_requires_queue_manager("prepare"));
        assert!(!role_requires_queue_manager("results"));
    }
}
