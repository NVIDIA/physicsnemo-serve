#[cfg(not(feature = "cpp"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("native-worker requires --features cpp and a packaged native deployment")
}

#[cfg(feature = "cpp")]
fn main() -> anyhow::Result<()> {
    // Leave enough stack for Redis response decoding, including debug builds.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024)
        .build()?
        .block_on(native_main())
}

#[cfg(feature = "cpp")]
async fn native_main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use native_worker::{deployment::RuntimeDeploymentConfig, registry::WorkerRegistration};
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use tokio::sync::watch;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("wait-registration") {
        let path = args
            .next()
            .context("usage: native-worker wait-registration <deployment.json>")?;
        ensure!(
            args.next().is_none(),
            "unexpected native registration wait argument"
        );
        return native_worker::startup_gate::wait_for_registration(std::path::Path::new(&path))
            .await;
    }
    let path = first
        .or_else(|| std::env::var("NATIVE_DEPLOYMENT_CONFIG").ok())
        .context("usage: native-worker <deployment.json>")?;
    let config_bytes = tokio::fs::read(&path).await?;
    let config: RuntimeDeploymentConfig = serde_json::from_slice(&config_bytes)?;
    let mut source_names = vec!["SOURCE_SHA"];
    if matches!(&config, RuntimeDeploymentConfig::Transolver(config) if config.model_build_sources.is_empty())
    {
        source_names.push("MODEL_BUILD_SOURCE_SHA");
    }
    for name in source_names {
        let sha = std::env::var(name)
            .with_context(|| format!("{name} must identify the release or model source"))?;
        ensure!(
            sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "{name} must be a full Git revision"
        );
    }
    let identity = native_worker::identity::RuntimeIdentity::from_environment()?;
    native_worker::inputs::verify_deployment(&config).await?;
    let admitted = native_worker::admission::admit(&config)?;
    tracing::info!(device = %admitted.device_name, "native release and device admission passed");
    let prefix = std::env::var("REDIS_STREAM_PREFIX").unwrap_or("physicsnemo:".into());
    ensure!(
        (prefix.is_empty() || prefix.ends_with(':')) && !prefix.chars().any(char::is_whitespace),
        "invalid Redis stream prefix"
    );
    let deployment =
        std::env::var("NATIVE_DEPLOYMENT_ID").unwrap_or_else(|_| config.workflow_id().into());
    let slot = std::env::var("NATIVE_WORKER_SLOT").unwrap_or_else(|_| "0".into());
    let (field, stream) = config.stream_identity(&deployment, &slot)?;
    let qm = scicomp_rq::QueueManager::from_env().await?;
    let mut metadata = serde_json::to_value(admitted)?;
    metadata["stream"] = json!(format!("{prefix}{stream}"));
    metadata["device_index"] = json!(config.device_index());
    metadata["executor_class"] = json!(config.executor_class());
    metadata["workflow_id"] = json!(config.workflow_id());
    metadata["deployment_id"] = json!(deployment);
    metadata["worker_slot"] = json!(slot);
    use sha2::{Digest, Sha256};
    metadata["deployment_sha256"] = json!(format!("{:x}", Sha256::digest(&config_bytes)));
    let inventory = std::path::Path::new(&path)
        .parent()
        .context("native config parent")?
        .join("release-inventory.json");
    if inventory.exists() {
        let inventory: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(inventory).await?)?;
        identity.verify_bundle_digest(
            inventory["bundle_digest"]
                .as_str()
                .context("native inventory bundle digest")?,
        )?;
        metadata["bundle_digest"] = inventory["bundle_digest"].clone();
    }
    metadata["pid"] = json!(std::process::id());
    metadata["source_sha"] = json!(std::env::var("SOURCE_SHA")?);
    metadata
        .as_object_mut()
        .context("native metadata object")?
        .extend(identity.fields());
    if let Ok(operation) = std::env::var("NATIVE_DEPLOYMENT_OPERATION_ID") {
        let image = identity
            .registry_image()
            .context("managed native fresh storage claims require a registry image identity")?;
        let claim_path = std::path::Path::new("/outputs/.native-deployment/owner.json");
        let info = tokio::fs::symlink_metadata(claim_path).await?;
        ensure!(
            info.is_file() && info.len() <= 4096,
            "invalid native fresh storage claim file"
        );
        let claim: serde_json::Value = serde_json::from_slice(&tokio::fs::read(claim_path).await?)?;
        ensure!(
            claim["schema_version"] == 1
                && claim["status"] == "claimed"
                && claim["operation_id"] == operation
                && claim["deployment_id"] == deployment
                && claim["image"] == image,
            "native fresh storage claim differs from this deployment"
        );
        metadata["fresh_start"] = claim;
    }
    match &config {
        RuntimeDeploymentConfig::Workflow(config) => {
            metadata["manifest_version"] = json!(config.manifest_version);
            metadata["public_contract_digest"] = json!(config.public_contract_digest);
            metadata["backend"] = config.workflow["models"]
                .as_object()
                .and_then(|models| models.values().next())
                .map(|model| model["backend"].clone())
                .unwrap_or_default();
            metadata["model_build_sources"] = json!(config.model_build_sources);
        }
        RuntimeDeploymentConfig::Transolver(config) => {
            metadata["backend"] = json!(config.backend);
            metadata
                .as_object_mut()
                .expect("device metadata object")
                .extend(config.model_build_provenance(
                    &std::env::var("MODEL_BUILD_SOURCE_SHA").unwrap_or_default(),
                ));
        }
    }
    let startup_deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(config.startup_timeout_seconds());
    let registration =
        Arc::new(WorkerRegistration::acquire(qm.clone(), field, metadata, startup_deadline).await?);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (heartbeat_registry, heartbeat_shutdown) = (registration.clone(), shutdown_tx.clone());
    let heartbeat = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if let Err(error) = heartbeat_registry.heartbeat().await {
                tracing::error!(%error,"native worker slot heartbeat failed; closing admission");
                let _ = heartbeat_shutdown.send(true);
                break;
            }
        }
    });
    let requested = Arc::new(AtomicBool::new(false));
    let (signal_requested, signal_tx) = (requested.clone(), shutdown_tx.clone());
    let signal = tokio::spawn(async move {
        #[cfg(unix)]
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = async {
                #[cfg(unix)] { term.recv().await; }
                #[cfg(not(unix))] { std::future::pending::<()>().await; }
            } => {},
        }
        signal_requested.store(true, Ordering::Release);
        let _ = signal_tx.send(true);
    });
    let result = serve(
        config,
        qm,
        prefix,
        stream,
        WorkerControl {
            registration: registration.clone(),
            shutdown_tx,
            shutdown_rx,
            requested,
            startup_deadline,
        },
    )
    .await;
    signal.abort();
    heartbeat.abort();
    let cleanup = registration.release().await;
    if let Err(error) = &cleanup {
        tracing::error!(%error,"native registry cleanup failed");
    }
    result?;
    cleanup
}

#[cfg(feature = "cpp")]
struct WorkerControl {
    registration: std::sync::Arc<native_worker::registry::WorkerRegistration>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
    startup_deadline: tokio::time::Instant,
}
#[cfg(feature = "cpp")]
async fn serve(
    config: native_worker::deployment::RuntimeDeploymentConfig,
    qm: scicomp_rq::QueueManager,
    prefix: String,
    stream: String,
    control: WorkerControl,
) -> anyhow::Result<()> {
    use anyhow::ensure;
    use native_worker::{host::HostClient, role::NativeRole};
    use std::{
        sync::{Arc, atomic::Ordering},
        time::Duration,
    };
    use worker_runtime::{
        config::RuntimeConfig, engine::EngineBuilder, transport::redis::RedisTransport,
    };
    let WorkerControl {
        registration,
        shutdown_tx,
        shutdown_rx,
        requested,
        startup_deadline,
    } = control;
    let host_config = config.clone();
    let (host, report) = HostClient::spawn_with_self_test(
        move || native_worker::bridge::CppHost::from_deployment(&host_config),
        startup_deadline.saturating_duration_since(tokio::time::Instant::now()),
        shutdown_rx.clone(),
    )
    .await?;
    let host_monitor = host.clone();
    let registry_monitor = registration.clone();
    let mut monitor_shutdown = shutdown_rx.clone();
    let monitor = tokio::spawn(async move {
        loop {
            tokio::select! {
                _=tokio::time::sleep(Duration::from_millis(100)) => {},
                _=monitor_shutdown.changed() => {},
            }
            if host_monitor.is_retired() || *monitor_shutdown.borrow() {
                host_monitor.retire();
                let _ = shutdown_tx.send(true);
                let _ = registry_monitor.restarting().await;
                break;
            }
        }
    });
    let result = async {
        let startup = native_worker::startup::replay(&config,&host,serde_json::from_str(&report)?);
        let mut startup_shutdown=shutdown_rx.clone();
        let stop = async {
            loop {
                if *startup_shutdown.borrow() { return; }
                if startup_shutdown.changed().await.is_err() { return; }
            }
        };
        let report = tokio::select! {
            biased;
            _=stop => anyhow::bail!("native workflow startup cancelled"),
            report=tokio::time::timeout_at(startup_deadline,startup) => report.map_err(|_|anyhow::anyhow!("native workflow startup deadline exceeded"))??,
        };
        let runtime: RuntimeConfig=serde_json::from_value(serde_json::json!({"stream_prefix":prefix,"max_retries":3,"shared_dlq_stream":"dlq","streams":[stream,"results","release","dlq"],"roles":{"native":{"inputs":[{"stream":stream,"max_dequeue_items":1,"poll_interval_ms":20,"block_ms":500,"reclaim_idle_ms":60000}],"outputs":["results","release"]}}}))?;
        runtime.validate()?;
        let engine=EngineBuilder::new(&runtime,"native").role(Box::new(NativeRole { config:config.clone(),host:host.clone(),shutdown:shutdown_rx.clone() })).transport(Arc::new(RedisTransport::new(qm,&prefix))).consumer(format!("native-{}-{}",std::process::id(),uuid::Uuid::new_v4())).build()?;
        engine.ensure_consumer_groups().await?;
        ensure!(!*shutdown_rx.borrow() && !host.is_retired(), "native startup was cancelled before admission");
        registration.available(&report).await?;
        tracing::info!("native startup checks passed; worker available");
        let _ = engine.run_until_shutdown(shutdown_rx,None).await?;
        ensure!(requested.load(Ordering::Acquire),"native host or registry ownership retired; restart required");
        Ok(())
    }.await;
    let _ = registration.restarting().await;
    monitor.abort();
    host.retire();
    let destroyed = host
        .shutdown(Duration::from_secs(config.drain_timeout_seconds()))
        .await;
    tracing::info!(destroyed, "native thread shutdown finished");
    result?;
    ensure!(
        destroyed,
        "native thread did not drain before shutdown deadline"
    );
    Ok(())
}
