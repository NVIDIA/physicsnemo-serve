use crate::host::{Cancellation, NativeHost};
use crate::request::ExecutionRequest;
use anyhow::Result;

#[cxx::bridge(namespace = "physicsnemo::serve")]
mod ffi {
    struct RunnerConfig {
        packages: Vec<String>,
        stats_path: String,
        startup_fixture_path: String,
        backend: String,
        domain: String,
        math_profile: String,
        block_size: i64,
        device_index: i32,
    }
    struct RunnerRequest {
        mesh_path: String,
        stl_path: String,
        standardized_output: String,
        physical_output: String,
        metadata_path: String,
        dump_input_dir: String,
        permutation_seed: i64,
        air_density: f64,
        stream_velocity: f64,
    }
    extern "Rust" {
        type Cancellation;
        fn is_cancelled(self: &Cancellation) -> bool;
    }
    unsafe extern "C++" {
        include!("native_worker/bridge.hpp");
        type CxxRunner;
        fn create_runner(config: &RunnerConfig) -> Result<UniquePtr<CxxRunner>>;
        fn create_workflow(config_json: &str) -> Result<UniquePtr<CxxRunner>>;
        fn self_test(self: Pin<&mut CxxRunner>, cancellation: &Cancellation) -> Result<String>;
        fn run_workflow(
            self: Pin<&mut CxxRunner>,
            request_json: &str,
            cancellation: &Cancellation,
        ) -> Result<String>;
        fn run(
            self: Pin<&mut CxxRunner>,
            request: &RunnerRequest,
            cancellation: &Cancellation,
        ) -> Result<String>;
    }
}

pub struct CppHost {
    runner: cxx::UniquePtr<ffi::CxxRunner>,
    workflow: bool,
    requires_goldens: bool,
    model_count: usize,
    expected_point_counts: Option<[u64; 2]>,
}
impl CppHost {
    pub fn new(config: &crate::request::DeploymentConfig) -> Result<Self> {
        Ok(Self {
            workflow: false,
            requires_goldens: config.startup_fixture_path.is_some(),
            model_count: config.packages.len(),
            expected_point_counts: if config.block_size > 0 {
                Some([
                    config.block_size as u64,
                    config.case.point_count % config.block_size as u64,
                ])
            } else {
                None
            },
            runner: ffi::create_runner(&ffi::RunnerConfig {
                packages: config
                    .packages
                    .iter()
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect(),
                stats_path: config.stats_path.to_string_lossy().into_owned(),
                startup_fixture_path: config
                    .startup_fixture_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                backend: config.backend.clone(),
                domain: "surface".into(),
                math_profile: "provider-default-v1".into(),
                block_size: config.block_size,
                device_index: config.device_index as i32,
            })?,
        })
    }
    pub fn from_deployment(config: &crate::deployment::RuntimeDeploymentConfig) -> Result<Self> {
        match config {
            crate::deployment::RuntimeDeploymentConfig::Transolver(config) => Self::new(config),
            crate::deployment::RuntimeDeploymentConfig::Workflow(config) => Ok(Self {
                runner: ffi::create_workflow(&serde_json::to_string(&config.workflow)?)?,
                workflow: true,
                requires_goldens: true,
                model_count: config.workflow["models"]
                    .as_object()
                    .map_or(0, |models| models.len()),
                expected_point_counts: None,
            }),
        }
    }
}
impl NativeHost for CppHost {
    fn self_test(&mut self, cancellation: &Cancellation) -> Result<String> {
        let report = self.runner.pin_mut().self_test(cancellation)?;
        anyhow::ensure!(
            report.len() <= 1024 * 1024,
            "native self-test report is too large"
        );
        let value: serde_json::Value = serde_json::from_str(&report)?;
        let models = value["models"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("native self-test omitted model evidence"))?;
        anyhow::ensure!(
            value["status"] == "passed"
                && !models.is_empty()
                && models.len() == self.model_count
                && value["session_load_count"].as_u64() == Some(models.len() as u64),
            "native model self-test did not pass for every session"
        );
        anyhow::ensure!(
            !self.requires_goldens || value["policy"] == "expected-fp32-v1",
            "native startup did not verify authored numerical goldens"
        );
        if let Some(expected) = self.expected_point_counts {
            let actual: std::collections::BTreeSet<_> = models
                .iter()
                .filter_map(|model| model["point_count"].as_u64())
                .collect();
            anyhow::ensure!(
                actual == expected.into_iter().collect(),
                "startup models do not cover the qualified full/tail point counts"
            );
        }
        Ok(report)
    }
    fn run(&mut self, request_json: &str, cancellation: &Cancellation) -> Result<String> {
        if self.workflow {
            return Ok(self
                .runner
                .pin_mut()
                .run_workflow(request_json, cancellation)?);
        }
        let request: ExecutionRequest = serde_json::from_str(request_json)?;
        Ok(self.runner.pin_mut().run(
            &ffi::RunnerRequest {
                mesh_path: request.mesh_path.to_string_lossy().into_owned(),
                stl_path: request.stl_path.to_string_lossy().into_owned(),
                standardized_output: request
                    .output_dir
                    .join("standardized.bin")
                    .to_string_lossy()
                    .into_owned(),
                physical_output: request
                    .output_dir
                    .join("physical.bin")
                    .to_string_lossy()
                    .into_owned(),
                metadata_path: request
                    .output_dir
                    .join("metadata.json")
                    .to_string_lossy()
                    .into_owned(),
                dump_input_dir: if request.capture_inputs {
                    request
                        .output_dir
                        .join("inputs")
                        .to_string_lossy()
                        .into_owned()
                } else {
                    String::new()
                },
                permutation_seed: request.permutation_seed,
                air_density: 1.205,
                stream_velocity: 30.0,
            },
            cancellation,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(native_workflow_only))]
    use crate::host::HostClient;
    #[cfg(not(native_workflow_only))]
    use crate::request::{Case, DeploymentConfig, InputObject, ModelAsset};

    #[test]
    fn native_bridge_reports_workflow_configuration_errors() {
        let error = match ffi::create_workflow("{}") {
            Ok(_) => panic!("the workflow host accepted missing model configuration"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("schema_version"));
    }

    #[tokio::test]
    #[cfg(not(native_workflow_only))]
    async fn native_thread_preserves_cpp_configuration_exception_without_cuda() {
        let stats_path = "/native ABI fixture/normalization-μ.json".into();
        let input = InputObject {
            url: "https://fixture.invalid/run_1.vtp".into(),
            sha256: "a".repeat(64),
            size_bytes: 1,
        };
        let mut config = DeploymentConfig {
            schema_version: 1,
            workflow_id: "physicsnemo-cfd-transolver-native".into(),
            executor_class: "native.transolver.aoti".into(),
            backend: "aoti".into(),
            device_index: 0,
            compute_capability: [9, 0],
            target: None,
            packages: vec![
                "/native ABI fixture/full-60000".into(),
                "/native ABI fixture/tail-8095".into(),
            ],
            stats_path,
            model_build_sources: Default::default(),
            startup_fixture_path: None,
            model_assets: vec![ModelAsset {
                path: "/native ABI fixture/normalization-μ.json".into(),
                sha256: "b".repeat(64),
                size_bytes: 1,
            }],
            block_size: 60000,
            artifact_root: "/native ABI fixture/artifacts".into(),
            input_cache: "/native ABI fixture/cache".into(),
            case: Case {
                case_id: "run_1".into(),
                point_count: 8_828_095,
                mesh: input.clone(),
                geometry: input,
            },
            request_timeout_seconds: 60,
            drain_timeout_seconds: 10,
        };
        config.validate().unwrap();
        // The C++ constructor rejects this before CUDA, file I/O or process
        // initialization. This is an unconditional CPU test of the real bridge,
        // shared runner ABI and exception propagation on the dedicated thread.
        config.block_size = 1;
        let error = match HostClient::spawn(move || CppHost::new(&config)).await {
            Ok(_) => panic!("the native constructor accepted a one-point block"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), "block_size must be at least two");
    }
}
