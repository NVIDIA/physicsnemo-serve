//! Native scheduler startup waits for this deployment's live logical slot.
use anyhow::{Context, Result, ensure};
use scicomp_rq::QueueManager;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Duration};
use tokio::time::Instant;

#[cfg(test)]
use crate::test_support as support;

struct ExpectedSlot {
    field: String,
    metadata: Value,
}

pub async fn wait_for_registration(path: &Path) -> Result<()> {
    let bytes = tokio::fs::read(path).await?;
    let config: crate::deployment::RuntimeDeploymentConfig = serde_json::from_slice(&bytes)?;
    config.validate()?;
    let source =
        std::env::var("SOURCE_SHA").context("SOURCE_SHA must identify the native release")?;
    ensure!(
        source.len() == 40 && source.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "SOURCE_SHA must be a full Git revision"
    );
    let identity = crate::identity::RuntimeIdentity::from_environment()?;
    let deployment =
        std::env::var("NATIVE_DEPLOYMENT_ID").unwrap_or_else(|_| config.workflow_id().to_owned());
    let slot = std::env::var("NATIVE_WORKER_SLOT").unwrap_or_else(|_| "0".into());
    let prefix = std::env::var("REDIS_STREAM_PREFIX").unwrap_or_else(|_| "physicsnemo:".into());
    ensure!(
        (prefix.is_empty() || prefix.ends_with(':')) && !prefix.chars().any(char::is_whitespace),
        "invalid Redis stream prefix"
    );
    let (field, stream) = config.stream_identity(&deployment, &slot)?;
    let expected = ExpectedSlot {
        field,
        metadata: json!({
            "stream":format!("{prefix}{stream}"),
            "workflow_id":config.workflow_id(),"executor_class":config.executor_class(),
            "device_index":config.device_index(),"deployment_id":deployment,"worker_slot":slot,
            "source_sha":source,"deployment_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "runtime_identity":identity,
        }),
    };
    wait_connected(
        QueueManager::from_env(),
        &expected,
        Instant::now() + Duration::from_secs(config.startup_timeout_seconds()),
    )
    .await
}

async fn wait_connected(
    connection: impl std::future::Future<
        Output = std::result::Result<QueueManager, scicomp_rq::QueueError>,
    >,
    expected: &ExpectedSlot,
    deadline: Instant,
) -> Result<()> {
    let qm = tokio::time::timeout_at(deadline, connection)
        .await
        .context("native registration connection deadline exceeded")??;
    wait_for_slot(&qm, expected, deadline).await
}

async fn wait_for_slot(
    qm: &QueueManager,
    expected: &ExpectedSlot,
    deadline: Instant,
) -> Result<()> {
    let lease_key = crate::registry::lease_key(&expected.field);
    loop {
        ensure!(
            Instant::now() < deadline,
            "native registration wait deadline exceeded"
        );
        let mut connection = qm.connection();
        let (raw, owner): (Option<String>, Option<String>) = tokio::time::timeout_at(
            deadline.min(Instant::now() + Duration::from_secs(3)),
            redis::pipe()
                .atomic()
                .cmd("HGET")
                .arg("gpu:registry")
                .arg(&expected.field)
                .cmd("GET")
                .arg(&lease_key)
                .query_async(&mut connection),
        )
        .await
        .context("native registration lookup deadline exceeded")??;
        if let (Some(raw), Some(owner)) = (raw, owner) {
            let metadata: Value =
                serde_json::from_str(&raw).context("invalid native registration")?;
            if !owner.is_empty()
                && metadata["worker_instance_id"].as_str() == Some(owner.as_str())
                && matches!(metadata["status"].as_str(), Some("warming" | "available"))
                && metadata["model_cache"]["warmup"]["reason"] != "restart_pending"
                && expected
                    .metadata
                    .as_object()
                    .context("expected slot metadata")?
                    .iter()
                    .all(|(key, value)| metadata.get(key) == Some(value))
            {
                return Ok(());
            }
        }
        tokio::time::sleep_until((Instant::now() + Duration::from_millis(100)).min(deadline)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::WorkerRegistration;

    fn expected() -> ExpectedSlot {
        ExpectedSlot {
            field: "native:execute.native.demo.demo.0.0".into(),
            metadata: json!({"stream":"test:execute.native.demo.demo.0.0",
                "workflow_id":"demo","executor_class":"native.demo","device_index":0,
                "deployment_id":"demo","worker_slot":"0","source_sha":"a".repeat(40),
                "deployment_sha256":"b".repeat(64),
                "runtime_identity":{"kind":"bundle","value":"c".repeat(64)}}),
        }
    }

    #[tokio::test]
    async fn startup_deadline_bounds_initial_redis_handshake() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        // TCP accepts the connection, but no Redis server completes its handshake.
        let expected = expected();
        let url = format!("redis://{address}");
        let waiting = wait_connected(
            QueueManager::new(&url),
            &expected,
            Instant::now() + Duration::from_millis(40),
        );
        let result = tokio::time::timeout(Duration::from_millis(250), waiting).await;
        assert!(
            result.is_ok(),
            "registration startup deadline did not bound Redis initialization"
        );
        assert!(result.unwrap().is_err());
    }

    #[tokio::test]
    async fn startup_gate_rejects_empty_or_unrelated_registry() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-empty-startup").await;
        let expected = expected();
        let unrelated =
            WorkerRegistration::warming(qm.clone(), "unrelated".into(), expected.metadata.clone())
                .await
                .unwrap();
        assert!(
            wait_for_slot(&qm, &expected, Instant::now() + Duration::from_millis(40))
                .await
                .is_err()
        );
        unrelated.release().await.unwrap();
    }

    #[tokio::test]
    async fn startup_gate_requires_exact_config_source_and_live_owner() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-bound-startup").await;
        let expected = expected();
        for field in [
            "stream",
            "workflow_id",
            "executor_class",
            "device_index",
            "deployment_id",
            "worker_slot",
            "source_sha",
            "deployment_sha256",
            "runtime_identity",
        ] {
            let mut wrong = expected.metadata.clone();
            wrong[field] = json!("wrong");
            let registration =
                WorkerRegistration::warming(qm.clone(), expected.field.clone(), wrong)
                    .await
                    .unwrap();
            assert!(
                wait_for_slot(&qm, &expected, Instant::now() + Duration::from_millis(20))
                    .await
                    .is_err(),
                "mismatched {field} passed startup gate"
            );
            registration.release().await.unwrap();
        }
        let registration = WorkerRegistration::warming(
            qm.clone(),
            expected.field.clone(),
            expected.metadata.clone(),
        )
        .await
        .unwrap();
        wait_for_slot(&qm, &expected, Instant::now() + Duration::from_secs(1))
            .await
            .unwrap();
        registration.release().await.unwrap();
        assert!(
            wait_for_slot(&qm, &expected, Instant::now() + Duration::from_millis(30))
                .await
                .is_err(),
            "retired logical slot has no live owner"
        );
    }

    #[tokio::test]
    async fn startup_gate_waits_until_exact_warming_registration_exists() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-delayed-startup").await;
        let expected = expected();
        let waiter = wait_for_slot(&qm, &expected, Instant::now() + Duration::from_secs(1));
        tokio::pin!(waiter);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut waiter)
                .await
                .is_err(),
            "scheduler started before worker registered"
        );
        let registration = WorkerRegistration::warming(
            qm.clone(),
            expected.field.clone(),
            expected.metadata.clone(),
        )
        .await
        .unwrap();
        waiter.await.unwrap();
        registration.release().await.unwrap();
    }
}
