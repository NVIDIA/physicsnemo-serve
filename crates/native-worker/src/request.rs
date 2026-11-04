use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputObject {
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub case_id: String,
    pub point_count: u64,
    pub mesh: InputObject,
    pub geometry: InputObject,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAsset {
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentConfig {
    pub schema_version: u32,
    pub workflow_id: String,
    pub executor_class: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    pub device_index: u32,
    pub compute_capability: [i32; 2],
    #[serde(default)]
    pub target: Option<crate::deployment::TargetPolicy>,
    pub packages: Vec<PathBuf>,
    #[serde(default)]
    pub model_build_sources: BTreeMap<String, String>,
    pub stats_path: PathBuf,
    #[serde(default)]
    pub startup_fixture_path: Option<PathBuf>,
    pub model_assets: Vec<ModelAsset>,
    pub block_size: i64,
    pub artifact_root: PathBuf,
    pub input_cache: PathBuf,
    pub case: Case,
    pub request_timeout_seconds: u64,
    pub drain_timeout_seconds: u64,
}
impl DeploymentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported native deployment schema"
        );
        ensure!(
            self.workflow_id == "physicsnemo-cfd-transolver-native",
            "unsupported compiled workflow"
        );
        ensure!(
            matches!(self.backend.as_str(), "aoti" | "tensorrt"),
            "unsupported native backend"
        );
        ensure!(
            self.executor_class == format!("native.transolver.{}", self.backend),
            "native executor class differs from selected backend"
        );
        ensure!(
            self.block_size == 60000,
            "CFD provider requires 60000 point blocks"
        );
        ensure!(
            self.compute_capability == [9, 0],
            "initial qualification requires H100 sm90"
        );
        if let Some(target) = &self.target {
            target.validate()?;
            ensure!(
                !target.is_cpu_fixture()
                    && target.compilation.compute_capability == self.compute_capability,
                "CFD target differs from the qualified GPU compute capability"
            );
        }
        ensure!(
            self.packages.len() == 2 && !self.model_assets.is_empty(),
            "full run_1 requires both compiled full and tail packages"
        );
        if !self.model_build_sources.is_empty() {
            ensure!(
                self.model_build_sources.len() == self.packages.len(),
                "compilation sources must identify every compiled CFD model"
            );
            for (alias, source) in &self.model_build_sources {
                ensure!(
                    !alias.is_empty()
                        && alias.len() <= 128
                        && alias
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')),
                    "invalid compiled CFD model alias"
                );
                ensure!(
                    source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()),
                    "model compilation source must be a full Git revision"
                );
            }
        }
        ensure!(
            self.artifact_root.is_absolute() && self.input_cache.is_absolute(),
            "native storage roots must be absolute"
        );
        ensure!(
            self.request_timeout_seconds > 0 && self.drain_timeout_seconds > 0,
            "timeouts must be positive"
        );
        ensure!(
            self.case.case_id == "run_1",
            "initial release only qualifies run_1"
        );
        ensure!(
            self.case.point_count == 8_828_095,
            "full run_1 must contain 8828095 source cells"
        );
        for object in [&self.case.mesh, &self.case.geometry] {
            ensure!(object.url.starts_with("https://"), "inputs require HTTPS");
            validate_digest(&object.sha256)?;
            ensure!(object.size_bytes > 0, "input size must be positive");
        }
        for asset in &self.model_assets {
            ensure!(
                asset.path.is_absolute(),
                "model asset paths must be absolute"
            );
            validate_digest(&asset.sha256)?;
            ensure!(asset.size_bytes > 0, "model asset size must be positive");
        }
        // Every file in each compiled package, including shared libraries, must
        // be in the release inventory. Startup verifies that inventory below.
        for package in &self.packages {
            ensure!(package.is_absolute(), "package paths must be absolute");
        }
        ensure!(
            self.model_assets
                .iter()
                .any(|asset| asset.path == self.stats_path),
            "normalization statistics must be hashed"
        );
        if let Some(path) = &self.startup_fixture_path {
            ensure!(
                path.is_absolute() && self.model_assets.iter().any(|asset| asset.path == *path),
                "startup fixture must be an absolute hashed release asset"
            );
        }
        Ok(())
    }

    /// A qualification revision never substitutes for the packages' sources.
    /// The legacy global value applies only when no per-model map was supplied.
    pub fn model_build_provenance(
        &self,
        legacy_source: &str,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut fields = serde_json::Map::new();
        if self.model_build_sources.is_empty() {
            fields.insert(
                "model_build_source_sha".into(),
                serde_json::json!(legacy_source),
            );
            return fields;
        }
        fields.insert(
            "model_build_sources".into(),
            serde_json::json!(self.model_build_sources),
        );
        let mut sources = self.model_build_sources.values();
        let first = sources.next().expect("nonempty compilation sources");
        if sources.all(|source| source == first) {
            fields.insert("model_build_source_sha".into(), serde_json::json!(first));
        }
        fields
    }
}
pub fn validate_digest(digest: &str) -> Result<()> {
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "expected lowercase SHA256"
    );
    Ok(())
}
fn default_backend() -> String {
    "aoti".into()
}
fn default_seed() -> u32 {
    42
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRequest {
    pub case_id: String,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default)]
    pub capture_inputs: bool,
}
impl NativeRequest {
    pub fn validate(&self, config: &DeploymentConfig) -> Result<()> {
        ensure!(
            self.case_id == config.case.case_id,
            "case is not in the qualified deployment inventory"
        );
        Ok(())
    }
}
pub fn provider_case_seed(seed: u32, case_id: &str) -> i64 {
    let mut digest = Sha256::new();
    digest.update(seed.to_string().as_bytes());
    digest.update([0]);
    digest.update(case_id.as_bytes());
    let bytes = digest.finalize();
    (u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) & 0x7fff_ffff_ffff_ffff) as i64
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionRequest {
    pub mesh_path: PathBuf,
    pub stl_path: PathBuf,
    pub output_dir: PathBuf,
    pub permutation_seed: i64,
    pub capture_inputs: bool,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_provider_seed_vectors() {
        assert_eq!(provider_case_seed(0, "run_1"), 2260079506302488169);
        assert_eq!(provider_case_seed(42, "run_1"), 1615895419752914611);
        assert_eq!(provider_case_seed(42, "run_11"), 3962535327529237276);
    }
    #[test]
    fn request_cannot_select_models_paths_or_unknown_fields() {
        assert!(
            serde_json::from_str::<NativeRequest>(r#"{"case_id":"run_1","model":"other"}"#)
                .is_err()
        );
        assert!(
            serde_json::from_str::<NativeRequest>(
                r#"{"case_id":"run_1","mesh_path":"/tmp/other"}"#
            )
            .is_err()
        );
        assert_eq!(
            serde_json::from_str::<NativeRequest>(r#"{"case_id":"run_1"}"#)
                .unwrap()
                .seed,
            42
        );
    }
}
