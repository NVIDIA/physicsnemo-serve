//! Independently authored workflow replay before external worker admission.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGolden {
    pub sha256: String,
    pub size_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StartupInput {
    pub path: std::path::PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    pub media_type: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StartupCase {
    pub name: String,
    pub request: Value,
    pub expected: Value,
    #[serde(default)]
    pub expected_artifacts: BTreeMap<String, ArtifactGolden>,
    #[serde(default)]
    pub input_artifacts: BTreeMap<String, StartupInput>,
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde(default = "default_atol")]
    pub atol: f64,
}
fn default_rtol() -> f64 {
    1e-5
}
fn default_atol() -> f64 {
    1e-6
}
fn has_assertion(expected: &Value) -> bool {
    match expected {
        Value::Object(fields) => fields.values().any(has_assertion),
        _ => true,
    }
}

pub fn validate_cases(cases: &[StartupCase]) -> Result<()> {
    ensure!(cases.len() <= 16, "too many startup workflow cases");
    let mut names = std::collections::BTreeSet::new();
    for case in cases {
        ensure!(
            !case.name.is_empty()
                && case.name.len() <= 128
                && case
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                && names.insert(&case.name),
            "invalid or duplicate startup case name"
        );
        ensure!(
            case.request.is_object() && case.expected.is_object(),
            "startup request/expected must be objects"
        );
        ensure!(
            has_assertion(&case.expected) || !case.expected_artifacts.is_empty(),
            "startup workflow case needs an authored payload assertion or artifact golden"
        );
        ensure!(
            case.rtol.is_finite() && case.rtol >= 0.0 && case.atol.is_finite() && case.atol >= 0.0,
            "invalid startup tolerances"
        );
        for golden in case.expected_artifacts.values() {
            crate::request::validate_digest(&golden.sha256)?;
        }
        ensure!(
            case.input_artifacts.len() <= 64,
            "too many startup input files"
        );
        for (name, input) in &case.input_artifacts {
            ensure!(
                !name.is_empty()
                    && name.len() <= 128
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
                    && input.path.is_absolute()
                    && !input.media_type.is_empty()
                    && input.media_type.len() <= 128
                    && !input.media_type.contains(['\r', '\n']),
                "invalid startup input descriptor"
            );
            crate::request::validate_digest(&input.sha256)?;
        }
    }
    Ok(())
}

pub fn compare_expected(actual: &Value, expected: &Value, rtol: f64, atol: f64) -> Result<()> {
    match expected {
        Value::Object(fields) => {
            let actual = actual
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("startup result must be an object"))?;
            for (name, value) in fields {
                compare_expected(
                    actual
                        .get(name)
                        .ok_or_else(|| anyhow::anyhow!("startup result omitted {name}"))?,
                    value,
                    rtol,
                    atol,
                )?;
            }
        }
        Value::Array(values) => {
            let actual = actual
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("startup result must be an array"))?;
            ensure!(
                actual.len() == values.len(),
                "startup result array shape differs"
            );
            for (left, right) in actual.iter().zip(values) {
                compare_expected(left, right, rtol, atol)?;
            }
        }
        Value::Number(expected) => {
            let right = expected
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("startup golden is not finite"))?;
            let left = actual
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("startup result number has wrong type"))?;
            ensure!(
                left.is_finite()
                    && right.is_finite()
                    && (left - right).abs() <= atol.max(rtol * left.abs().max(right.abs())),
                "startup numerical result differs from authored golden"
            );
        }
        _ => ensure!(
            actual == expected,
            "startup result differs from authored golden"
        ),
    }
    Ok(())
}

pub async fn replay(
    config: &crate::deployment::RuntimeDeploymentConfig,
    host: &crate::host::HostClient,
    mut report: Value,
) -> Result<Value> {
    use anyhow::Context;
    use serde_json::json;
    let cases = match config {
        crate::deployment::RuntimeDeploymentConfig::Workflow(config) => &config.startup_cases,
        _ => {
            report["workflow"] = json!({"status":"model_only","cases":[]});
            return Ok(report);
        }
    };
    validate_cases(cases)?;
    if cases.is_empty() {
        report["workflow"] = json!({"status":"model_only","cases":[]});
        return Ok(report);
    }
    let crate::deployment::RuntimeDeploymentConfig::Workflow(config) = config else {
        unreachable!()
    };
    let mut isolated = config.clone();
    // Retention cannot remove active inputs/outputs while a startup call runs.
    // Failed/cancelled scratch is retained; only known-completed replay is removed.
    isolated.artifact_root = config
        .artifact_root
        .join(".cache")
        .join("startup")
        .join(uuid::Uuid::new_v4().to_string());
    let mut evidence = Vec::new();
    for case in cases {
        let started = std::time::Instant::now();
        let run_id = format!("startup-{}", uuid::Uuid::new_v4());
        let mut request_artifacts = Vec::new();
        let mut ingress_dir = None;
        if !case.input_artifacts.is_empty() {
            let dir = isolated
                .ingress_root
                .as_ref()
                .context("startup inputs require ingress root")?
                .join(&run_id);
            tokio::fs::create_dir_all(dir.parent().context("startup ingress parent")?).await?;
            tokio::fs::create_dir(&dir).await?;
            for (name, input) in &case.input_artifacts {
                crate::inputs::verify_file(&input.path, input.size_bytes, &input.sha256).await?;
                let artifact_id = uuid::Uuid::new_v4().to_string();
                let path = dir.join(format!("{artifact_id}-startup-{name}"));
                tokio::fs::copy(&input.path, &path).await?;
                crate::inputs::verify_file(&path, input.size_bytes, &input.sha256).await?;
                request_artifacts.push(json!({"field_name":name,"name":name,"artifact_id":artifact_id,"storage_path":path,"sha256":input.sha256,"size_bytes":input.size_bytes,"media_type":input.media_type}));
            }
            ingress_dir = Some(dir);
        }
        let mut completion = None;
        let payload = json!({"parameters":case.request,"manifest_version":config.manifest_version,"request":{"content_type":if request_artifacts.is_empty(){"application/json"}else{"multipart/form-data"},"input_artifacts":request_artifacts}});
        let actual = crate::role::execute_workflow(
            &isolated,
            host,
            &run_id,
            &payload,
            &uuid::Uuid::new_v4().to_string(),
            &mut completion,
        )
        .await
        .with_context(|| format!("startup workflow case {} failed", case.name))?;
        compare_expected(&actual, &case.expected, case.rtol, case.atol)
            .with_context(|| format!("startup workflow case {} differs", case.name))?;
        let artifacts = actual["artifacts"]
            .as_array()
            .context("startup artifacts are missing")?;
        ensure!(
            artifacts.len() == case.expected_artifacts.len(),
            "startup artifact golden inventory differs"
        );
        for artifact in artifacts {
            let name = artifact["name"].as_str().context("startup artifact name")?;
            let expected = case
                .expected_artifacts
                .get(name)
                .context("startup artifact has no authored golden")?;
            ensure!(
                artifact["sha256"] == expected.sha256
                    && artifact["size_bytes"] == expected.size_bytes,
                "startup artifact bytes differ from authored golden"
            );
        }
        completion
            .take()
            .context("startup native completion is missing")?
            .committed();
        if let Some(dir) = ingress_dir {
            tokio::fs::remove_dir_all(dir).await?;
        }
        evidence.push(
            json!({"name":case.name,"status":"passed","seconds":started.elapsed().as_secs_f64()}),
        );
    }
    tokio::fs::remove_dir_all(&isolated.artifact_root).await?;
    report["workflow"] = json!({"status":"passed","cases":evidence});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn startup_comparison_rejects_wrong_values_types_and_shapes() {
        let expected = json!({"output":[3.0,5.0],"ok":true});
        for actual in [
            json!({"output":[3.0,6.0],"ok":true}),
            json!({"output":[3.0],"ok":true}),
            json!({"output":[3.0,5.0],"ok":1}),
            json!({"ok":true}),
        ] {
            assert!(
                compare_expected(&actual, &expected, 1e-5, 1e-6).is_err(),
                "{actual}"
            );
        }
        compare_expected(
            &json!({"output":[3.000001,5],"ok":true,"extra":2}),
            &expected,
            1e-5,
            1e-6,
        )
        .unwrap();
    }
    #[test]
    fn startup_fixtures_are_bounded_and_unique() {
        let case: StartupCase =
            serde_json::from_value(json!({"name":"affine","request":{},"expected":{"value":1}}))
                .unwrap();
        validate_cases(std::slice::from_ref(&case)).unwrap();
        assert!(validate_cases(&[case.clone(), case]).is_err());
    }
    #[test]
    fn startup_cannot_claim_workflow_coverage_without_an_authored_assertion() {
        for expected in [json!({}), json!({"nested":{}})] {
            let case: StartupCase =
                serde_json::from_value(json!({"name":"vacuous","request":{},"expected":expected}))
                    .unwrap();
            assert!(validate_cases(&[case]).is_err());
        }
        let artifact_only:StartupCase=serde_json::from_value(json!({"name":"artifact","request":{},"expected":{},"expected_artifacts":{"field.bin":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size_bytes":4}}})).unwrap();
        validate_cases(&[artifact_only]).unwrap();
    }
    struct FakeWorkflow;
    impl crate::host::NativeHost for FakeWorkflow {
        fn run(&mut self, _: &str, _: &crate::host::Cancellation) -> Result<String> {
            Ok(json!({"payload":{"output":[3,5]},"artifacts":[],"metadata":{}}).to_string())
        }
    }
    fn deployment(
        root: &std::path::Path,
        expected: Value,
    ) -> crate::deployment::RuntimeDeploymentConfig {
        serde_json::from_value(json!({"schema_version":2,"workflow_id":"startup-test","executor_class":"cpp.test","device_index":0,
            "target":{"profile":"cpu-test-only","compilation":{"platform":"linux/amd64","gpu_product":"cpu-test-only","compute_capability":[0,0],"cuda_architecture":"none","precision":"fp32","mig":"disabled","hardware_compatibility":"qualified-product-only"},"deployment":{"minimum_available_memory_mb":1,"mig":"disabled","driver_requirements":{"minimum_driver_version":"0"}}},
            "workflow":{"schema_version":1,"adapter":"custom.v1","models":{"primary":{"package_path":root,"backend":"ort","device":"cpu","device_index":0}}},
            "model_build_sources":{"primary":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"request_schema":{"type":"object"},"response_schema":{"type":"object","required":["output"]},
            "model_assets":[],"artifact_root":root,"input_cache":root.join("cache"),"request_timeout_seconds":2,"drain_timeout_seconds":1,"startup_timeout_seconds":2,
            "startup_cases":[{"name":"affine","request":{"values":[1,2]},"expected":expected}]})).unwrap()
    }
    #[tokio::test]
    async fn model_checks_alone_do_not_accept_wrong_workflow_output() {
        let root = tempfile::tempdir().unwrap();
        let config = deployment(root.path(), json!({"output":[3,6]}));
        let host = crate::host::HostClient::spawn(|| Ok(FakeWorkflow))
            .await
            .unwrap();
        let report = replay(
            &config,
            &host,
            json!({"status":"passed","models":[{"status":"passed"}]}),
        )
        .await;
        assert!(
            report.is_err(),
            "model success must not admit a bad custom workflow"
        );
        host.retire();
        host.shutdown(std::time::Duration::from_secs(1)).await;
    }
    #[tokio::test]
    async fn startup_replays_production_validation_without_queue_publication() {
        let root = tempfile::tempdir().unwrap();
        let config = deployment(root.path(), json!({"output":[3,5]}));
        let host = crate::host::HostClient::spawn(|| Ok(FakeWorkflow))
            .await
            .unwrap();
        let report = replay(
            &config,
            &host,
            json!({"status":"passed","models":[{"status":"passed"}]}),
        )
        .await
        .unwrap();
        assert_eq!(report["workflow"]["status"], "passed");
        assert_eq!(report["workflow"]["cases"][0]["name"], "affine");
        assert!(!host.is_retired());
        host.retire();
        host.shutdown(std::time::Duration::from_secs(1)).await;
    }
}
