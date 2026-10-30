use crate::request::{DeploymentConfig, InputObject, ModelAsset};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RuntimeDeploymentConfig {
    Workflow(WorkflowDeploymentConfig),
    Transolver(DeploymentConfig),
}
impl From<DeploymentConfig> for RuntimeDeploymentConfig {
    fn from(value: DeploymentConfig) -> Self {
        Self::Transolver(value)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowDeploymentConfig {
    pub schema_version: u32,
    pub workflow_id: String,
    #[serde(default)]
    pub manifest_version: Option<String>,
    #[serde(default)]
    pub public_contract_digest: Option<String>,
    pub executor_class: String,
    pub device_index: u32,
    pub target: TargetPolicy,
    pub workflow: Value,
    pub model_build_sources: BTreeMap<String, String>,
    pub request_schema: Value,
    #[serde(default = "any_schema")]
    pub response_schema: Value,
    pub model_assets: Vec<ModelAsset>,
    #[serde(default)]
    pub input_artifacts: BTreeMap<String, InputArtifact>,
    #[serde(default)]
    pub ingress_root: Option<PathBuf>,
    #[serde(default)]
    pub input_files: BTreeMap<String, RequestInputRule>,
    #[serde(default)]
    pub startup_cases: Vec<crate::startup::StartupCase>,
    pub artifact_root: PathBuf,
    pub input_cache: PathBuf,
    pub request_timeout_seconds: u64,
    pub drain_timeout_seconds: u64,
    #[serde(default = "startup_timeout")]
    pub startup_timeout_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestInputRule {
    pub required: bool,
    pub media_types: Vec<String>,
    pub max_size_bytes: u64,
}
fn startup_timeout() -> u64 {
    300
}
fn any_schema() -> Value {
    Value::Bool(true)
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputArtifact {
    Local(ModelAsset),
    Remote(InputObject),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPolicy {
    pub profile: String,
    pub compilation: CompilationTarget,
    pub deployment: DeploymentTarget,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompilationTarget {
    pub platform: String,
    pub gpu_product: String,
    pub compute_capability: [i32; 2],
    pub cuda_architecture: String,
    pub precision: String,
    pub mig: String,
    pub hardware_compatibility: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentTarget {
    pub minimum_available_memory_mb: u64,
    pub mig: String,
    pub driver_requirements: DriverRequirements,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DriverRequirements {
    pub minimum_driver_version: String,
    #[serde(default)]
    pub maximum_driver_version: Option<String>,
}
pub struct DeviceSnapshot {
    pub platform: String,
    pub product: String,
    pub compute_capability: [i32; 2],
    pub mig_enabled: bool,
    pub available_memory_mb: u64,
    pub driver_version: String,
}
fn version(value: &str) -> Result<[u32; 4]> {
    let parts: Vec<_> = value.split('.').collect();
    ensure!(
        !parts.is_empty() && parts.len() <= 4,
        "invalid driver version"
    );
    let mut parsed = [0; 4];
    for (index, part) in parts.iter().enumerate() {
        ensure!(
            !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()),
            "invalid driver version"
        );
        parsed[index] = part.parse()?;
    }
    Ok(parsed)
}
impl TargetPolicy {
    pub fn is_cpu_fixture(&self) -> bool {
        self.profile == "cpu-test-only"
    }
    pub fn validate(&self) -> Result<()> {
        let c = &self.compilation;
        let user_build = self.profile == "build-product-v1";
        ensure!(
            !self.profile.is_empty()
                && (c.platform == "linux/amd64" || (user_build && c.platform == "linux/aarch64")),
            "unsupported target platform/profile"
        );
        ensure!(
            c.precision == "fp32",
            "initial workflow runtime requires FP32"
        );
        ensure!(
            c.mig == "disabled" && self.deployment.mig == "disabled",
            "MIG targets are not supported"
        );
        ensure!(
            c.hardware_compatibility
                == if user_build {
                    "build-product-only"
                } else {
                    "qualified-product-only"
                },
            "target must require its declared exact GPU product policy"
        );
        if self.is_cpu_fixture() {
            ensure!(
                c.gpu_product == "cpu-test-only" && c.compute_capability == [0, 0],
                "CPU fixtures must be explicitly labelled"
            );
        } else if user_build {
            ensure!(
                !c.gpu_product.trim().is_empty()
                    && c.gpu_product != "cpu-test-only"
                    && c.gpu_product.len() <= 256
                    && c.compute_capability[0] > 0
                    && (0..=9).contains(&c.compute_capability[1])
                    && c.cuda_architecture
                        == format!("sm{}{}", c.compute_capability[0], c.compute_capability[1]),
                "local build target requires an exact GPU product and matching CUDA architecture"
            );
            ensure!(
                self.deployment.minimum_available_memory_mb > 0,
                "available-memory admission threshold is required"
            );
        } else {
            ensure!(
                c.gpu_product == "NVIDIA H100 80GB HBM3"
                    && c.compute_capability == [9, 0]
                    && c.cuda_architecture == "sm90",
                "initial production target is the qualified H100 product/sm90"
            );
            ensure!(
                self.deployment.minimum_available_memory_mb > 0,
                "available-memory admission threshold is required"
            );
        }
        let low = version(&self.deployment.driver_requirements.minimum_driver_version)?;
        if let Some(high) = &self.deployment.driver_requirements.maximum_driver_version {
            ensure!(version(high)? >= low, "driver range is empty");
        }
        Ok(())
    }
    pub fn admit(&self, observed: &DeviceSnapshot) -> Result<()> {
        self.validate()?;
        let c = &self.compilation;
        let scope = if self.profile == "build-product-v1" {
            "compiled"
        } else {
            "qualified"
        };
        ensure!(
            observed.platform == c.platform,
            "deployment platform differs"
        );
        ensure!(
            observed.product == c.gpu_product,
            "GPU product differs from {scope} target"
        );
        ensure!(
            observed.compute_capability == c.compute_capability,
            "GPU compute capability differs"
        );
        ensure!(!observed.mig_enabled, "MIG must be disabled");
        ensure!(
            observed.available_memory_mb >= self.deployment.minimum_available_memory_mb,
            "insufficient available device memory"
        );
        let current = version(&observed.driver_version)?;
        let required = &self.deployment.driver_requirements;
        ensure!(
            current >= version(&required.minimum_driver_version)?,
            "GPU driver is below {scope} minimum"
        );
        if let Some(maximum) = &required.maximum_driver_version {
            ensure!(
                current <= version(maximum)?,
                "GPU driver exceeds {scope} maximum"
            );
        }
        Ok(())
    }
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}
fn local_schema(value: &Value) -> Result<()> {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref") {
                ensure!(
                    reference.as_str().is_some_and(|r| r.starts_with('#')),
                    "external request-schema references are not allowed"
                );
            }
            for child in object.values() {
                local_schema(child)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                local_schema(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}
impl WorkflowDeploymentConfig {
    pub fn validate(&self) -> Result<()> {
        crate::startup::validate_cases(&self.startup_cases)?;
        ensure!(
            matches!(self.schema_version, 2 | 3),
            "unsupported native workflow deployment schema"
        );
        if self.schema_version == 3 {
            ensure!(
                self.workflow["schema_version"] == 2,
                "managed deployment requires workflow host schema2"
            );
            ensure!(
                self.manifest_version
                    .as_ref()
                    .is_some_and(|v| !v.is_empty() && v.len() <= 128),
                "native deployment requires manifest_version"
            );
            crate::request::validate_digest(self.public_contract_digest.as_deref().unwrap_or(""))?;
        }
        ensure!(
            self.input_files.len() <= 64,
            "too many declared native request files"
        );
        if !self.input_files.is_empty() {
            ensure!(
                self.ingress_root.as_ref().is_some_and(|p| p.is_absolute()),
                "native request files require an absolute trusted ingress_root"
            );
            ensure!(
                self.manifest_version
                    .as_ref()
                    .is_some_and(|v| !v.is_empty()),
                "native request files require manifest_version"
            );
        }
        for (name, rule) in &self.input_files {
            ensure!(
                identifier(name)
                    && !matches!(name.as_str(), "." | "..")
                    && !self.input_artifacts.contains_key(name),
                "invalid or conflicting request input name"
            );
            ensure!(
                rule.max_size_bytes > 0
                    && !rule.media_types.is_empty()
                    && rule
                        .media_types
                        .iter()
                        .all(|v| !v.is_empty() && !v.contains(['\r', '\n'])),
                "native request file policy requires bounded size and media types"
            );
        }
        ensure!(
            identifier(&self.workflow_id) && identifier(&self.executor_class),
            "invalid workflow/executor identity"
        );
        ensure!(
            self.artifact_root.is_absolute() && self.input_cache.is_absolute(),
            "native storage roots must be absolute"
        );
        ensure!(
            self.request_timeout_seconds > 0
                && self.drain_timeout_seconds > 0
                && self.startup_timeout_seconds > 0,
            "timeouts must be positive"
        );
        self.target.validate()?;
        local_schema(&self.request_schema)?;
        local_schema(&self.response_schema)?;
        jsonschema::validator_for(&self.request_schema)
            .map_err(|e| anyhow::anyhow!("invalid request schema: {e}"))?;
        jsonschema::validator_for(&self.response_schema)
            .map_err(|e| anyhow::anyhow!("invalid response schema: {e}"))?;
        ensure!(
            matches!(self.workflow["schema_version"].as_u64(), Some(1 | 2))
                && matches!(
                    self.workflow["adapter"].as_str(),
                    Some("tensor.v1" | "custom.v1")
                ),
            "unsupported native workflow adapter/schema"
        );
        let models = self.workflow["models"]
            .as_object()
            .context("workflow models are required")?;
        ensure!(!models.is_empty(), "workflow needs compiled models");
        ensure!(
            self.model_build_sources.len() == models.len(),
            "compilation sources must identify every model alias exactly"
        );
        for (alias, model) in models {
            ensure!(identifier(alias), "invalid model alias");
            let source = self
                .model_build_sources
                .get(alias)
                .context("missing model compilation source")?;
            ensure!(
                source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()),
                "model compilation source must be a full Git revision"
            );
            let path = Path::new(
                model["package_path"]
                    .as_str()
                    .context("model package path")?,
            );
            ensure!(path.is_absolute(), "model package path must be absolute");
            let expected = if self.target.is_cpu_fixture() {
                "cpu"
            } else {
                "cuda"
            };
            ensure!(
                model["device"] == expected
                    && model["device_index"].as_u64() == Some(u64::from(self.device_index)),
                "model device differs from deployment"
            );
        }
        ensure!(
            !self.model_assets.is_empty(),
            "compiled asset inventory is required"
        );
        for asset in &self.model_assets {
            validate_asset(asset)?;
        }
        for (name, artifact) in &self.input_artifacts {
            ensure!(identifier(name), "invalid input artifact name");
            match artifact {
                InputArtifact::Local(asset) => validate_asset(asset)?,
                InputArtifact::Remote(object) => {
                    ensure!(object.url.starts_with("https://"), "inputs require HTTPS");
                    crate::request::validate_digest(&object.sha256)?;
                    ensure!(object.size_bytes > 0, "input size must be positive");
                }
            }
        }
        Ok(())
    }
    pub fn validate_parameters(&self, parameters: &Value) -> Result<()> {
        let validator = jsonschema::validator_for(&self.request_schema)
            .map_err(|e| anyhow::anyhow!("invalid request schema: {e}"))?;
        ensure!(
            validator.is_valid(parameters),
            "request parameters do not match the deployed schema"
        );
        Ok(())
    }
}
fn validate_asset(asset: &ModelAsset) -> Result<()> {
    ensure!(
        asset.path.is_absolute() && asset.size_bytes > 0,
        "asset path/size is invalid"
    );
    crate::request::validate_digest(&asset.sha256)
}
impl RuntimeDeploymentConfig {
    pub fn stream_identity(&self, deployment: &str, slot: &str) -> Result<(String, String)> {
        ensure!(
            identifier(deployment) && identifier(slot),
            "invalid stable deployment/worker slot"
        );
        let stream = format!(
            "execute.{}.{}.{}.{}",
            self.executor_class(),
            deployment,
            slot,
            self.device_index()
        );
        Ok((format!("native:{stream}"), stream))
    }
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Workflow(c) => c.validate(),
            Self::Transolver(c) => c.validate(),
        }
    }
    pub fn workflow_id(&self) -> &str {
        match self {
            Self::Workflow(c) => &c.workflow_id,
            Self::Transolver(c) => &c.workflow_id,
        }
    }
    pub fn executor_class(&self) -> &str {
        match self {
            Self::Workflow(c) => &c.executor_class,
            Self::Transolver(c) => &c.executor_class,
        }
    }
    pub fn device_index(&self) -> u32 {
        match self {
            Self::Workflow(c) => c.device_index,
            Self::Transolver(c) => c.device_index,
        }
    }
    pub fn artifact_root(&self) -> &Path {
        match self {
            Self::Workflow(c) => &c.artifact_root,
            Self::Transolver(c) => &c.artifact_root,
        }
    }
    pub fn input_cache(&self) -> &Path {
        match self {
            Self::Workflow(c) => &c.input_cache,
            Self::Transolver(c) => &c.input_cache,
        }
    }
    pub fn request_timeout_seconds(&self) -> u64 {
        match self {
            Self::Workflow(c) => c.request_timeout_seconds,
            Self::Transolver(c) => c.request_timeout_seconds,
        }
    }
    pub fn drain_timeout_seconds(&self) -> u64 {
        match self {
            Self::Workflow(c) => c.drain_timeout_seconds,
            Self::Transolver(c) => c.drain_timeout_seconds,
        }
    }
    pub fn startup_timeout_seconds(&self) -> u64 {
        match self {
            Self::Workflow(c) => c.startup_timeout_seconds,
            Self::Transolver(_) => startup_timeout(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn policy() -> TargetPolicy {
        serde_json::from_value(json!({"profile":"h100-sxm-v1","compilation":{
            "platform":"linux/amd64","gpu_product":"NVIDIA H100 80GB HBM3","compute_capability":[9,0],"cuda_architecture":"sm90","precision":"fp32","mig":"disabled","hardware_compatibility":"qualified-product-only"},
            "deployment":{"minimum_available_memory_mb":70000,"mig":"disabled","driver_requirements":{"minimum_driver_version":"570.158.01","maximum_driver_version":"580.99"}}})).unwrap()
    }
    fn observed() -> DeviceSnapshot {
        DeviceSnapshot {
            platform: "linux/amd64".into(),
            product: "NVIDIA H100 80GB HBM3".into(),
            compute_capability: [9, 0],
            mig_enabled: false,
            available_memory_mb: 75000,
            driver_version: "580.65.06".into(),
        }
    }
    #[test]
    fn same_sm_different_gpu_product_is_not_admitted() {
        let target = policy();
        target.admit(&observed()).unwrap();
        let mut actual = observed();
        actual.product = "NVIDIA H200".into();
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("GPU product")
        );
    }
    #[test]
    fn local_build_target_admits_only_its_compiled_gpu_product() {
        let mut target = policy();
        target.profile = "build-product-v1".into();
        target.compilation.hardware_compatibility = "build-product-only".into();
        target.compilation.gpu_product = "NVIDIA A100-SXM4-80GB".into();
        target.compilation.compute_capability = [8, 0];
        target.compilation.cuda_architecture = "sm80".into();
        assert!(
            target.validate().is_ok(),
            "a user build must admit its own exact GPU product"
        );
        let mut actual = observed();
        actual.product = target.compilation.gpu_product.clone();
        actual.compute_capability = [8, 0];
        target.admit(&actual).unwrap();
        actual.product = "NVIDIA A100-PCIE-80GB".into();
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("GPU product")
        );
        actual.product = target.compilation.gpu_product.clone();
        actual.compute_capability = [8, 6];
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("compute capability")
        );
        actual.compute_capability = [8, 0];
        actual.mig_enabled = true;
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("MIG")
        );
    }
    #[test]
    fn mig_memory_and_driver_bounds_are_independent_admission_gates() {
        let target = policy();
        let mut actual = observed();
        actual.mig_enabled = true;
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("MIG")
        );
        actual = observed();
        actual.available_memory_mb = 69999;
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("memory")
        );
        actual = observed();
        actual.driver_version = "550.1".into();
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("minimum")
        );
        actual.driver_version = "590.1".into();
        assert!(
            target
                .admit(&actual)
                .unwrap_err()
                .to_string()
                .contains("maximum")
        );
    }
    #[test]
    fn driver_comparison_is_numeric_and_rejects_ambiguous_versions() {
        assert!(version("580.10").unwrap() > version("580.9.99").unwrap());
        assert_eq!(version("570.1").unwrap(), version("570.1.0").unwrap());
        for invalid in ["", "570.x", "570..1", ">=570"] {
            assert!(version(invalid).is_err());
        }
    }
    #[test]
    fn worker_schemas_cannot_resolve_remote_references() {
        assert!(
            local_schema(&json!({"$ref":"https://must-not-be-contacted.invalid/schema"})).is_err()
        );
        assert!(local_schema(&json!({"$ref":"#/$defs/input"})).is_ok());
    }
}
