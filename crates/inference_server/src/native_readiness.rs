//! Native readiness is a live owner/lease check, separate from capacity.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
use worker_runtime::identity::RuntimeIdentity;

use crate::state::AppState;

#[derive(Clone, Debug)]
pub struct NativeExpectation {
    pub workflow_id: String,
    pub deployment_id: String,
    pub worker_slot: String,
    pub backend: String,
    pub release_image: String,
    pub runtime_identity: RuntimeIdentity,
    pub bundle_digest: Option<String>,
    pub deployment_sha256: String,
    pub manifest_version: Option<String>,
    pub public_contract_digest: Option<String>,
    pub registry_field: String,
    pub lease_key: String,
    pub workflow_cases: Vec<String>,
    pub requires_initialization: bool,
    self_test_policy: &'static str,
    pub operation_id: Option<String>,
}

impl NativeExpectation {
    pub fn from_environment() -> Option<std::result::Result<Self, String>> {
        let path = std::env::var_os("NATIVE_DEPLOYMENT_CONFIG")?;
        Some(Self::load(Path::new(&path)).map_err(|error| error.to_string()))
    }

    fn load(path: &Path) -> Result<Self> {
        ensure!(
            std::fs::metadata(path)?.len() <= 16 * 1024 * 1024,
            "native config exceeds limit"
        );
        let bytes = std::fs::read(path)?;
        let config: Value = serde_json::from_slice(&bytes)?;
        let text = |key: &str| -> Result<String> {
            Ok(config[key]
                .as_str()
                .with_context(|| format!("missing native {key}"))?
                .to_owned())
        };
        let workflow_id = text("workflow_id")?;
        let deployment_id =
            std::env::var("NATIVE_DEPLOYMENT_ID").unwrap_or_else(|_| workflow_id.clone());
        let worker_slot = std::env::var("NATIVE_WORKER_SLOT").unwrap_or_else(|_| "0".into());
        let executor = text("executor_class")?;
        let device = config["device_index"]
            .as_u64()
            .context("native device_index")?;
        let registry_field =
            format!("native:execute.{executor}.{deployment_id}.{worker_slot}.{device}");
        let lease_key = format!(
            "native:worker-lease:{:x}",
            Sha256::digest(registry_field.as_bytes())
        );
        let backend = config["backend"]
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                config["workflow"]["models"].as_object()?.values().next()?["backend"]
                    .as_str()
                    .map(str::to_owned)
            })
            .context("native backend")?;
        let inventory = path
            .parent()
            .context("native config parent")?
            .join("release-inventory.json");
        let mut bundle_digest = if inventory.exists() {
            let value: Value = serde_json::from_slice(&std::fs::read(inventory)?)?;
            Some(
                value["bundle_digest"]
                    .as_str()
                    .context("native bundle digest")?
                    .to_owned(),
            )
        } else {
            None
        };
        let runtime_identity = RuntimeIdentity::from_environment()?;
        if let Some(digest) = &bundle_digest {
            runtime_identity.verify_bundle_digest(digest)?;
        } else if let RuntimeIdentity::Bundle(digest) = &runtime_identity {
            bundle_digest = Some(digest.clone());
        }
        let workflow_cases = config["startup_cases"]
            .as_array()
            .map(|cases| {
                cases
                    .iter()
                    .map(|case| {
                        case["name"]
                            .as_str()
                            .context("startup case name")
                            .map(str::to_owned)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            workflow_id,
            deployment_id,
            worker_slot,
            backend,
            release_image: runtime_identity.registry_image().unwrap_or_default().into(),
            runtime_identity,
            bundle_digest,
            deployment_sha256: format!("{:x}", Sha256::digest(&bytes)),
            manifest_version: config["manifest_version"].as_str().map(str::to_owned),
            public_contract_digest: config["public_contract_digest"].as_str().map(str::to_owned),
            registry_field,
            lease_key,
            workflow_cases,
            requires_initialization: config["workflow"]["schema_version"] == 2
                && config["workflow"]["adapter"] == "custom.v1",
            self_test_policy: required_self_test_policy(&config),
            operation_id: std::env::var("NATIVE_DEPLOYMENT_OPERATION_ID").ok(),
        })
    }
}

fn required_self_test_policy(config: &Value) -> &'static str {
    if config["schema_version"] == 1 && config["startup_fixture_path"].is_null() {
        "finite-static-inputs-v1"
    } else {
        "expected-fp32-v1"
    }
}

pub fn unavailable(reason: impl Into<String>) -> Value {
    json!({"schema_version":1,"capabilities":["live-native-readiness-v1"],"ready":false,"reason":reason.into()})
}

pub fn assess(expected: &NativeExpectation, metadata: &Value, owner: &str, ttl: i64) -> Value {
    let check = || -> Result<()> {
        ensure!(
            !owner.is_empty() && ttl > 0 && metadata["worker_instance_id"].as_str() == Some(owner),
            "native worker owner lease is missing, expired or changed"
        );
        for (key, wanted) in [
            ("workflow_id", &expected.workflow_id),
            ("deployment_id", &expected.deployment_id),
            ("worker_slot", &expected.worker_slot),
            ("backend", &expected.backend),
            ("deployment_sha256", &expected.deployment_sha256),
        ] {
            ensure!(
                metadata[key].as_str() == Some(wanted),
                "native {key} differs from the deployed release"
            );
        }
        if let Some(identity) = metadata.get("runtime_identity") {
            ensure!(
                serde_json::from_value::<RuntimeIdentity>(identity.clone())?
                    == expected.runtime_identity,
                "native runtime_identity differs from the deployed execution source"
            );
        } else {
            ensure!(
                expected.runtime_identity.registry_image().is_some(),
                "native runtime_identity is missing"
            );
        }
        if let Some(image) = expected.runtime_identity.registry_image() {
            ensure!(
                metadata["release_image"].as_str() == Some(image),
                "native release_image differs from the deployed release"
            );
        } else {
            ensure!(
                metadata.get("release_image").is_none(),
                "a host bundle or local image must not claim a registry release_image"
            );
        }
        for (key, wanted) in [
            ("bundle_digest", &expected.bundle_digest),
            ("manifest_version", &expected.manifest_version),
            ("public_contract_digest", &expected.public_contract_digest),
        ] {
            if let Some(wanted) = wanted {
                ensure!(
                    metadata[key].as_str() == Some(wanted),
                    "native {key} differs from the deployed release"
                );
            }
        }
        ensure!(
            matches!(
                metadata["status"].as_str(),
                Some("available" | "busy" | "executing")
            ),
            "native worker is warming, stopping or failed"
        );
        if let Some(operation) = &expected.operation_id {
            ensure!(
                expected.runtime_identity.registry_image().is_some(),
                "managed fresh storage claims require a registry image"
            );
            let fresh = &metadata["fresh_start"];
            ensure!(
                fresh["schema_version"] == 1
                    && fresh["status"] == "claimed"
                    && fresh["operation_id"].as_str() == Some(operation)
                    && fresh["deployment_id"] == expected.deployment_id
                    && fresh["image"] == expected.release_image,
                "managed native startup has no matching fresh storage claim"
            );
        }
        let warmup = &metadata["model_cache"]["warmup"];
        let report = &warmup["self_test"];
        ensure!(
            warmup["status"] == "succeeded"
                && report["status"] == "passed"
                && report["policy"] == expected.self_test_policy
                && report["models"]
                    .as_array()
                    .is_some_and(|models| !models.is_empty()),
            "native model startup checks have not passed"
        );
        if expected.requires_initialization {
            ensure!(
                report["initialization"]["status"] == "passed"
                    && report["initialization"]["calls"] == 1,
                "native workflow initialization has not passed"
            );
        }
        if !expected.workflow_cases.is_empty() {
            let workflow = &report["workflow"];
            let cases = workflow["cases"]
                .as_array()
                .context("native workflow startup evidence is missing")?;
            let names: std::collections::BTreeSet<_> = cases
                .iter()
                .filter_map(|case| case["name"].as_str())
                .collect();
            ensure!(
                workflow["status"] == "passed"
                    && cases.len() == expected.workflow_cases.len()
                    && names == expected.workflow_cases.iter().map(String::as_str).collect()
                    && cases.iter().all(|case| case["status"] == "passed"),
                "native workflow startup cases have not all passed"
            );
        }
        Ok(())
    };
    let reason = check().err().map(|error| error.to_string());
    let mut response = json!({"schema_version":1,"capabilities":["live-native-readiness-v1"],"ready":reason.is_none(),"reason":reason,
        "workflow_id":expected.workflow_id,"deployment_id":expected.deployment_id,"worker_slot":expected.worker_slot,
        "backend":metadata["backend"],"release_image":metadata["release_image"],"bundle_digest":metadata["bundle_digest"],
        "runtime_identity":metadata.get("runtime_identity").cloned().unwrap_or_else(|| json!({"kind":"registry-image","value":metadata["release_image"]})),
        "deployment_sha256":metadata["deployment_sha256"],"public_contract_digest":metadata["public_contract_digest"],
        "manifest_version":metadata["manifest_version"],"worker_instance_id":metadata["worker_instance_id"],"status":metadata["status"],
        "startup":metadata["model_cache"]["warmup"]["self_test"],"fresh_start":metadata["fresh_start"],
        "target":{"device_name":metadata["device_name"],"driver_version":metadata["driver_version"],"compute_capability":metadata["compute_capability"],"mig_enabled":metadata["mig_enabled"],"available_memory_mb":metadata["available_memory_mb"],"qualification":metadata["qualification"]}});
    if expected.runtime_identity.registry_image().is_none() {
        response
            .as_object_mut()
            .expect("readiness object")
            .remove("release_image");
    }
    response
}

pub async fn probe(state: &AppState, expected: &NativeExpectation) -> Value {
    let service = state.redis_service.read().await;
    let Some(service) = service.as_ref() else {
        return unavailable("Redis not initialized");
    };
    let mut connection = service.get_connection();
    // One Redis execution observes both the registry row and current owner.
    let snapshot = redis::Script::new(
        "return {redis.call('HGET',KEYS[1],ARGV[1]) or '', redis.call('GET',KEYS[2]) or '', redis.call('PTTL',KEYS[2])}",
    );
    let mut invocation = snapshot.prepare_invoke();
    invocation
        .key("gpu:registry")
        .key(&expected.lease_key)
        .arg(&expected.registry_field);
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        invocation.invoke_async::<(String, String, i64)>(&mut connection),
    )
    .await;
    match result {
        Ok(Ok((raw, owner, ttl))) => match serde_json::from_str::<Value>(&raw) {
            Ok(metadata) => assess(expected, &metadata, &owner, ttl),
            Err(_) => unavailable("native worker has no valid registry record"),
        },
        _ => unavailable("native ownership lookup failed"),
    }
}

#[cfg(test)]
#[path = "../../worker-runtime/tests/support/mod.rs"]
mod support;

#[cfg(test)]
mod tests {
    use super::*;
    fn expected() -> NativeExpectation {
        NativeExpectation {
            workflow_id: "affine".into(),
            deployment_id: "demo".into(),
            worker_slot: "0".into(),
            backend: "aoti".into(),
            release_image: "registry/image@sha256:123".into(),
            runtime_identity: RuntimeIdentity::RegistryImage("registry/image@sha256:123".into()),
            bundle_digest: Some("bundle".into()),
            deployment_sha256: "config".into(),
            manifest_version: Some("1.0".into()),
            public_contract_digest: Some("contract".into()),
            registry_field: "field".into(),
            lease_key: "lease".into(),
            workflow_cases: vec!["affine".into()],
            requires_initialization: true,
            self_test_policy: "expected-fp32-v1",
            operation_id: None,
        }
    }
    fn metadata() -> Value {
        json!({"workflow_id":"affine","deployment_id":"demo","worker_slot":"0","backend":"aoti","manifest_version":"1.0",
            "public_contract_digest":"contract","release_image":"registry/image@sha256:123","bundle_digest":"bundle","deployment_sha256":"config",
            "status":"available","worker_instance_id":"owner","device_name":"NVIDIA H100 80GB HBM3",
            "model_cache":{"warmup":{"status":"succeeded","self_test":{"status":"passed","policy":"expected-fp32-v1","models":[{"status":"passed"}],
                "initialization":{"status":"passed","calls":1},"workflow":{"status":"passed","cases":[{"name":"affine","status":"passed"}]}}}}})
    }
    #[test]
    fn readiness_uses_the_deployments_startup_policy() {
        for (config, required) in [
            (json!({"schema_version":1}), "finite-static-inputs-v1"),
            (
                json!({"schema_version":1,"startup_fixture_path":null}),
                "finite-static-inputs-v1",
            ),
            (
                json!({"schema_version":1,"startup_fixture_path":"/release/startup.json"}),
                "expected-fp32-v1",
            ),
            (json!({"schema_version":2}), "expected-fp32-v1"),
            (json!({"schema_version":3}), "expected-fp32-v1"),
        ] {
            let mut expected = expected();
            expected.self_test_policy = required_self_test_policy(&config);
            assert_eq!(expected.self_test_policy, required);
            for policy in ["finite-static-inputs-v1", "expected-fp32-v1", "unknown"] {
                let mut value = metadata();
                value["model_cache"]["warmup"]["self_test"]["policy"] = json!(policy);
                assert_eq!(
                    assess(&expected, &value, "owner", 1000)["ready"],
                    policy == required,
                    "config={config}, policy={policy}"
                );
            }
        }
    }

    #[test]
    fn healthy_busy_native_worker_remains_ready() {
        let mut value = metadata();
        value["status"] = json!("busy");
        assert_eq!(assess(&expected(), &value, "owner", 1000)["ready"], true);
    }
    #[test]
    fn host_bundle_readiness_requires_matching_typed_identity_without_image_claim() {
        let digest = "a".repeat(64);
        let mut wanted = expected();
        wanted.runtime_identity = RuntimeIdentity::Bundle(digest.clone());
        wanted.release_image = String::new();
        wanted.bundle_digest = Some(digest.clone());
        let mut value = metadata();
        value.as_object_mut().unwrap().remove("release_image");
        value["runtime_identity"] = json!({"kind":"bundle","value":digest});
        value["bundle_digest"] = json!(digest);
        assert_eq!(
            assess(&wanted, &value, "owner", 1000)["ready"],
            true,
            "a host native worker does not have a registry image"
        );
        assert!(
            assess(&wanted, &value, "owner", 1000)
                .get("release_image")
                .is_none()
        );
        value["runtime_identity"]["kind"] = json!("image-config");
        assert_eq!(assess(&wanted, &value, "owner", 1000)["ready"], false);
    }
    #[test]
    fn local_image_readiness_checks_config_identity_and_rejects_registry_claims() {
        let digest = format!("sha256:{}", "b".repeat(64));
        let mut wanted = expected();
        wanted.runtime_identity = RuntimeIdentity::ImageConfig(digest.clone());
        wanted.release_image = String::new();
        let mut value = metadata();
        value.as_object_mut().unwrap().remove("release_image");
        value["runtime_identity"] = json!({"kind":"image-config","value":digest});
        assert_eq!(assess(&wanted, &value, "owner", 1000)["ready"], true);
        value["runtime_identity"]["value"] = json!(format!("sha256:{}", "c".repeat(64)));
        assert_eq!(assess(&wanted, &value, "owner", 1000)["ready"], false);
        value["runtime_identity"]["value"] = json!(digest);
        value["release_image"] = json!("registry/image@sha256:123");
        assert_eq!(assess(&wanted, &value, "owner", 1000)["ready"], false);
    }
    #[test]
    fn stale_registry_or_changed_owner_cannot_pass_readiness() {
        for (owner, ttl) in [("", -2), ("other", 1000), ("owner", 0)] {
            assert_eq!(
                assess(&expected(), &metadata(), owner, ttl)["ready"],
                false,
                "owner={owner}, ttl={ttl}"
            );
        }
    }
    #[test]
    fn warming_wrong_release_and_bad_workflow_stay_unready() {
        for key in [
            "status",
            "bundle_digest",
            "deployment_sha256",
            "backend",
            "manifest_version",
            "public_contract_digest",
        ] {
            let mut value = metadata();
            value[key] = json!("wrong");
            assert_eq!(
                assess(&expected(), &value, "owner", 1000)["ready"],
                false,
                "{key}"
            );
        }
        let mut value = metadata();
        value["model_cache"]["warmup"]["self_test"]["workflow"]["status"] = json!("failed");
        assert_eq!(assess(&expected(), &value, "owner", 1000)["ready"], false);
    }
    #[test]
    fn managed_readiness_requires_the_actual_fresh_start_claim() {
        let mut expected = expected();
        expected.operation_id = Some("operation".into());
        let mut value = metadata();
        value["fresh_start"] = json!({"schema_version":1,"status":"claimed","operation_id":"operation","deployment_id":"demo","image":expected.release_image});
        assert_eq!(assess(&expected, &value, "owner", 1000)["ready"], true);
        value["fresh_start"]["operation_id"] = json!("another");
        assert_eq!(assess(&expected, &value, "owner", 1000)["ready"], false);
    }
    #[tokio::test]
    async fn api_readyz_observes_live_redis_owner_and_ignores_busy_capacity() {
        use axum::{extract::State, http::StatusCode};
        let (_server, qm) = super::support::spawn_test_queue_manager("native-api-readiness").await;
        let mut state = AppState::new_for_testing(crate::config::ServerConfig::from_env().unwrap());
        let expectation = expected();
        state.native_runtime = Some(Ok(expectation.clone()));
        *state.redis_service.write().await = Some(crate::redis_ops::RedisService::new(qm.clone()));
        let state = std::sync::Arc::new(state);
        let mut connection = qm.connection();
        let mut metadata = metadata();
        metadata["status"] = json!("warming");
        let _: () = redis::cmd("HSET")
            .arg("gpu:registry")
            .arg(&expectation.registry_field)
            .arg(metadata.to_string())
            .query_async(&mut connection)
            .await
            .unwrap();
        let _: () = redis::cmd("SET")
            .arg(&expectation.lease_key)
            .arg("owner")
            .arg("PX")
            .arg(15000)
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(
            crate::handlers::readyz(State(state.clone())).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        metadata["status"] = json!("busy");
        let _: () = redis::cmd("HSET")
            .arg("gpu:registry")
            .arg(&expectation.registry_field)
            .arg(metadata.to_string())
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(
            crate::handlers::readyz(State(state.clone())).await.0,
            StatusCode::OK
        );
        let _: () = redis::cmd("DEL")
            .arg(&expectation.lease_key)
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(
            crate::handlers::readyz(State(state)).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
