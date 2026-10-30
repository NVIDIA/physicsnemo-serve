use crate::{
    deployment::RuntimeDeploymentConfig,
    host::{Completion, HostClient},
    inputs::materialize,
    request::{ExecutionRequest, NativeRequest, provider_case_seed},
};
use anyhow::{Context, Result, anyhow, ensure};
use scicomp_rq::{Message, Output};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use tokio::sync::watch;
use worker_runtime::traits::{BoxFuture, MessageSink, RoleCancellation, WorkerRole};

pub struct NativeRole {
    pub config: RuntimeDeploymentConfig,
    pub host: HostClient,
    pub shutdown: watch::Receiver<bool>,
}

struct AttemptGuard {
    host: HostClient,
    committed: bool,
}
impl Drop for AttemptGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.host.retire();
        }
    }
}

#[derive(Debug)]
struct Reservation {
    resource_id: u32,
    memory_mb: u64,
}
fn reservation(payload: &Value, device: u32) -> Result<Reservation> {
    let resource_id = payload["resource_id"]
        .as_u64()
        .context("dispatch missing resource_id")?;
    let memory_mb = payload["memory_mb"]
        .as_u64()
        .filter(|n| *n > 0)
        .context("dispatch missing memory_mb")?;
    ensure!(
        resource_id == u64::from(device),
        "dispatch assigned to another resource"
    );
    Ok(Reservation {
        resource_id: device,
        memory_mb,
    })
}

async fn shutdown_requested(mut shutdown: watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

/// Execute and validate one workflow without queue publication. Startup replay uses
/// this same boundary while external admission remains warming. The caller retains
/// Completion until it has accepted/compared the result.
pub async fn execute_workflow(
    config: &crate::deployment::WorkflowDeploymentConfig,
    host: &HostClient,
    run_id: &str,
    payload: &Value,
    attempt: &str,
    completion: &mut Option<Completion>,
) -> Result<Value> {
    let parameters = payload
        .get("parameters")
        .context("missing canonical parameters")?;
    config.validate_parameters(parameters)?;
    let run_key = format!("{:x}", Sha256::digest(run_id.as_bytes()));
    let output_dir = config
        .artifact_root
        .join(format!("native-{run_key}"))
        .join(attempt);
    tokio::fs::create_dir_all(output_dir.parent().context("attempt parent")?).await?;
    tokio::fs::create_dir(&output_dir).await?;
    let mut admitted =
        crate::request_inputs::admit_inputs(config, payload, run_id, &output_dir).await?;
    let request = json!({"request_id":run_id,"parameters":parameters,"output_dir":output_dir,"inputs":admitted.paths,"input_metadata":admitted.metadata});
    admitted.begin_native_call();
    let mut active = match host.submit(request.to_string()).await {
        Ok(active) => active,
        Err(error) => {
            // A returned submit error means no command was enqueued. A dropped
            // submit future is uncertain and deliberately keeps its input bytes.
            admitted.native_call_finished();
            return Err(error);
        }
    };
    let done = active.wait().await?;
    admitted.native_call_finished();
    let raw = done
        .result
        .as_ref()
        .map_err(|e| anyhow!("native workflow failed: {e:#}"))?;
    ensure!(
        raw.len() <= 2 * 1024 * 1024,
        "native result JSON exceeds its limit"
    );
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ResultObject {
        payload: Value,
        #[serde(default)]
        artifacts: Vec<NativeArtifact>,
        #[serde(default)]
        metadata: Value,
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct NativeArtifact {
        name: String,
        path: std::path::PathBuf,
        media_type: String,
        #[serde(default)]
        primary: bool,
    }
    let result: ResultObject = serde_json::from_str(raw)?;
    *completion = Some(done);
    let mut response = result
        .payload
        .as_object()
        .context("native response payload must be an object")?
        .clone();
    ensure!(
        !response.contains_key("artifacts")
            && !response.contains_key("native")
            && !response.contains_key("output_path"),
        "native adapter overwrote reserved response fields"
    );
    let schema = jsonschema::validator_for(&config.response_schema)
        .map_err(|e| anyhow!("invalid response schema: {e}"))?;
    ensure!(
        schema.is_valid(&result.payload),
        "native response does not match deployed schema"
    );
    ensure!(
        result.artifacts.len() <= 64,
        "too many native output artifacts"
    );
    let root = tokio::fs::canonicalize(&output_dir).await?;
    let mut names = std::collections::BTreeSet::new();
    let mut outputs = Vec::new();
    let mut primary = None;
    let limit = config.workflow["limits"]["max_tensor_bytes"]
        .as_u64()
        .unwrap_or(64 * 1024 * 1024);
    for output in result.artifacts {
        ensure!(
            !output.name.is_empty()
                && output.name.len() <= 128
                && output
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                && names.insert(output.name.clone()),
            "invalid/duplicate native artifact name"
        );
        ensure!(
            !output.path.as_os_str().is_empty()
                && output
                    .path
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "native artifact path must be relative and contained"
        );
        ensure!(
            !output.media_type.is_empty()
                && output.media_type.len() <= 128
                && !output.media_type.contains(['\n', '\r']),
            "invalid native artifact media type"
        );
        let path = output_dir.join(&output.path);
        ensure!(
            tokio::fs::canonicalize(&path).await?.starts_with(&root),
            "native artifact escaped its attempt directory"
        );
        ensure!(
            tokio::fs::symlink_metadata(&path).await?.is_file(),
            "native artifact must be a regular file"
        );
        ensure!(
            tokio::fs::metadata(&path).await?.len() <= limit,
            "native artifact exceeds configured size limit"
        );
        let mut descriptor = artifact(&path, &output.name, &output.media_type).await?;
        descriptor["primary"] = json!(output.primary);
        outputs.push(descriptor);
        if output.primary {
            ensure!(primary.is_none(), "more than one primary artifact");
            primary = Some(path);
        }
    }
    response.insert("artifacts".into(), json!(outputs));
    if let Some(path) = primary {
        response.insert("output_path".into(), json!(path));
    }
    response.insert("native".into(), json!({"schema_version":2,"attempt":attempt,"runtime":"cpp",
        "adapter":config.workflow["adapter"],"source_sha":std::env::var("SOURCE_SHA").unwrap_or_default(),
        "model_build_sources":config.model_build_sources,"metadata":result.metadata}));
    response
        .get_mut("native")
        .and_then(Value::as_object_mut)
        .context("native provenance object")?
        .extend(crate::identity::optional_environment_fields()?);
    Ok(Value::Object(response))
}

impl NativeRole {
    async fn run_native(
        &self,
        msg: &Message,
        payload: &Value,
        attempt: &str,
        completion: &mut Option<Completion>,
    ) -> Result<Value> {
        ensure!(
            payload["workflow_id"] == self.config.workflow_id(),
            "workflow does not match compiled deployment"
        );
        ensure!(
            payload["runtime"]["kind"] == "cpp"
                && payload["runtime"]["executor_class"] == self.config.executor_class(),
            "runtime does not match native deployment"
        );
        ensure!(
            payload.get("parent_run_id").is_none_or(Value::is_null)
                && payload.get("batch_info").is_none_or(Value::is_null),
            "native worker does not support fanout or scheduler batch envelopes"
        );
        if let RuntimeDeploymentConfig::Workflow(config) = &self.config {
            return self
                .run_workflow(msg, payload, attempt, completion, config)
                .await;
        }
        let RuntimeDeploymentConfig::Transolver(config) = &self.config else {
            unreachable!()
        };
        // The API keeps the original HTTP body (including optional operation
        // and parameters wrappers) in raw_fields. Execute only its canonical,
        // schema-validated parameters, as other workflow roles do.
        let request: NativeRequest = serde_json::from_value(payload["parameters"].clone())?;
        request.validate(config)?;
        // Inputs are fixed by the release inventory, never by arbitrary paths
        // or URLs in a queued request. Keep attempts in immutable unique dirs.
        let mesh = materialize(&config.case.mesh, &config.input_cache).await?;
        let geometry = materialize(&config.case.geometry, &config.input_cache).await?;
        let run_key = format!("{:x}", Sha256::digest(msg.run_id().as_bytes()));
        let output_dir = self
            .config
            .artifact_root()
            .join(format!("native-{run_key}"))
            .join(attempt);
        tokio::fs::create_dir_all(output_dir.parent().context("attempt parent")?).await?;
        tokio::fs::create_dir(&output_dir).await?;
        let execution = ExecutionRequest {
            mesh_path: mesh,
            stl_path: geometry,
            output_dir: output_dir.clone(),
            permutation_seed: provider_case_seed(request.seed, &request.case_id),
            capture_inputs: request.capture_inputs,
        };
        let mut active = self.host.submit(serde_json::to_string(&execution)?).await?;
        // The outer timeout also covers download/queue time. The execution
        // owns cancellation even if that outer future or the engine is dropped.
        let done = active.wait().await?;
        let metadata: Value = serde_json::from_str(
            done.result
                .as_ref()
                .map_err(|e| anyhow!("native workflow failed: {e:#}"))?,
        )?;
        *completion = Some(done);
        ensure!(
            metadata["point_count"].as_u64() == Some(config.case.point_count),
            "native output does not cover the qualified full case"
        );
        let mut artifacts = vec![];
        for (name, file, media) in [
            (
                "physical_fields",
                "physical.bin",
                "application/octet-stream",
            ),
            (
                "standardized_fields",
                "standardized.bin",
                "application/octet-stream",
            ),
            ("native_metadata", "metadata.json", "application/json"),
        ] {
            let descriptor = artifact(&output_dir.join(file), name, media).await?;
            if media == "application/octet-stream" {
                ensure!(
                    descriptor["size_bytes"].as_u64() == Some(config.case.point_count * 16),
                    "native field artifact size does not match [N,4] FP32"
                );
            }
            artifacts.push(descriptor);
        }
        if request.capture_inputs {
            for file in [
                "fx.f32",
                "embedding.f32",
                "permutation.i64",
                "source_cell_ids.i64",
            ] {
                artifacts.push(
                    artifact(
                        &output_dir.join("inputs").join(file),
                        file,
                        "application/octet-stream",
                    )
                    .await?,
                );
            }
        }
        let mut response = json!({"output_path":output_dir.join("physical.bin"),"artifacts":artifacts,
            "native":{"schema_version":1,"attempt":attempt,"case_id":request.case_id,
                "run_seed":request.seed,"permutation_seed":execution.permutation_seed,
                "runtime":"cpp","backend":config.backend,"block_size":config.block_size,
                "source_sha":std::env::var("SOURCE_SHA").unwrap_or_default(),
                "mesh_sha256":config.case.mesh.sha256,"geometry_sha256":config.case.geometry.sha256,
                "metadata":metadata}});
        response["native"]
            .as_object_mut()
            .expect("native provenance object")
            .extend(crate::identity::optional_environment_fields()?);
        response["native"]
            .as_object_mut()
            .expect("native provenance object")
            .extend(config.model_build_provenance(
                &std::env::var("MODEL_BUILD_SOURCE_SHA").unwrap_or_default(),
            ));
        Ok(response)
    }

    async fn run_workflow(
        &self,
        msg: &Message,
        payload: &Value,
        attempt: &str,
        completion: &mut Option<Completion>,
        config: &crate::deployment::WorkflowDeploymentConfig,
    ) -> Result<Value> {
        execute_workflow(
            config,
            &self.host,
            msg.run_id(),
            payload,
            attempt,
            completion,
        )
        .await
    }

    async fn process(
        &self,
        msg: &Message,
        sink: &dyn MessageSink,
        cancellation: Option<RoleCancellation>,
    ) -> Result<()> {
        scicomp_rq::QueueManager::validate_attempt_message(msg)?;
        let payload: Value = msg.parse_payload()?;
        // Malformed dispatch cannot safely invent resource release information.
        // Leave it pending for the engine's quarantine policy.
        let reservation = reservation(&payload, self.config.device_index())?;
        let attempt = uuid::Uuid::new_v4().to_string();
        let mut guard = AttemptGuard {
            host: self.host.clone(),
            committed: false,
        };
        // BEGIN may have stored its token even if its response is lost.
        // Keep the token on retries and retire admission if the claim remains
        // uncertain, so a new unique consumer can safely reclaim the source.
        for retry in 0..3 {
            let claim =
                tokio::time::timeout(Duration::from_secs(10), sink.begin_attempt(msg, &attempt))
                    .await;
            match claim {
                Ok(Ok(Some(_))) => {
                    guard.committed = true;
                    return Ok(());
                }
                Ok(Ok(None)) => break,
                error if retry == 2 => {
                    tracing::warn!(
                        run_id = msg.run_id(),
                        ?error,
                        "native attempt claim remains uncertain; retiring host and retaining pending delivery"
                    );
                    return Err(worker_runtime::traits::message_deferred(format!(
                        "native attempt claim remains uncertain: {error:?}"
                    )));
                }
                _ => tokio::time::sleep(Duration::from_millis(100 * (retry + 1))).await,
            }
        }
        let mut completion = None;
        let started = Instant::now();
        let ownership = async {
            match cancellation {
                Some(ref c) => c.cancelled().await,
                None => std::future::pending::<()>().await,
            }
        };
        let result = tokio::select! {
            biased;
            _ = ownership => return Err(anyhow!("native request lost its lease ownership")),
            result = tokio::time::timeout(Duration::from_secs(self.config.request_timeout_seconds()),
                self.run_native(msg,&payload,&attempt,&mut completion)) => match result {
                    Ok(value) => value,
                    Err(_) => { self.host.retire(); Err(anyhow!("native request deadline exceeded")) }
                },
            _ = shutdown_requested(self.shutdown.clone()) => {
                self.host.retire(); Err(anyhow!("native worker is shutting down"))
            }
        };
        let status = if result.is_ok() {
            "completed"
        } else {
            "failed"
        };
        if self.host.is_retired() && completion.is_none() {
            let drained = self
                .host
                .drain(Duration::from_secs(self.config.drain_timeout_seconds()))
                .await;
            tracing::warn!(drained, "native host retired; admission remains closed");
        }
        let mut outcome = json!({"run_id":msg.run_id(),"status":status,"workflow":self.config.workflow_id(),
            "completed_at":chrono::Utc::now().to_rfc3339(),"gpu_stream":msg.stream(),
            "execution_time_seconds":started.elapsed().as_secs_f64()});
        match result {
            Ok(value) => outcome["payload"] = value,
            Err(error) => outcome["error"] = Value::String(format!("{error:#}")),
        }
        if matches!(self.config, RuntimeDeploymentConfig::Workflow(_)) {
            let mut execution = outcome.as_object().expect("execution object").clone();
            // Canonical execution uses API lifecycle names; scheduler outcomes
            // and reservation releases retain their completion status.
            execution.insert(
                "status".into(),
                json!(if status == "completed" {
                    "succeeded"
                } else {
                    status
                }),
            );
            let mut response = execution.remove("payload").unwrap_or_else(|| json!({}));
            if let Some(object) = response.as_object_mut() {
                if let Some(artifacts) = object.remove("artifacts") {
                    execution.insert("outputs".into(), artifacts);
                }
                if let Some(path) = object.remove("output_path") {
                    execution.insert("output_path".into(), path);
                }
                if let Some(native) = object.remove("native") {
                    execution.insert("native".into(), native);
                }
            }
            let request = payload
                .get("request")
                .cloned()
                .unwrap_or_else(|| json!({"parameters":payload["parameters"]}));
            outcome = json!({"run_id":msg.run_id(),"status":status,"workflow":self.config.workflow_id(),"request":request,"execution":execution,"payload":response});
        }
        let release = json!({"run_id":msg.run_id(),"resource_id":reservation.resource_id,
            "memory_mb":reservation.memory_mb,"status":status,"released_at":chrono::Utc::now().to_rfc3339()});
        let outputs = [
            Output::new("release", release.to_string()).with_stage("release"),
            Output::new("results", outcome.to_string()).with_stage("results"),
        ];
        // Keep the token, payload bytes and execution gate unchanged across
        // response loss. The transport returns the durable committed marker.
        for retry in 0..3 {
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(anyhow!("native completion lost its lease ownership"));
            }
            let committed = tokio::time::timeout(
                Duration::from_secs(10),
                sink.complete_attempt(msg, &outputs, &attempt),
            )
            .await;
            match committed {
                Ok(Ok(_)) => break,
                error if retry == 2 => {
                    tracing::warn!(
                        run_id = msg.run_id(),
                        ?error,
                        "native completion remains uncertain; retiring host and retaining pending delivery"
                    );
                    return Err(worker_runtime::traits::message_deferred(format!(
                        "native completion remains uncertain: {error:?}"
                    )));
                }
                _ => tokio::time::sleep(Duration::from_millis(100 * (retry + 1))).await,
            }
        }
        if let Some(done) = completion {
            done.committed();
        }
        guard.committed = true;
        Ok(())
    }
}

async fn artifact(path: &Path, name: &str, media_type: &str) -> Result<Value> {
    use tokio::io::AsyncReadExt;
    ensure!(
        tokio::fs::symlink_metadata(path).await?.is_file(),
        "native artifact must be a regular file"
    );
    let mut file = tokio::fs::File::open(path).await?;
    let size = file.metadata().await?.len();
    let mut hash = Sha256::new();
    let mut bytes = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut bytes).await?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(
        json!({"name":name,"storage_path":path,"media_type":media_type,"size_bytes":size,
        "sha256":format!("{:x}",hash.finalize())}),
    )
}

impl WorkerRole for NativeRole {
    fn name(&self) -> &'static str {
        "native"
    }
    fn handle<'a>(
        &'a self,
        msg: &'a Message,
        _stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.process(msg, sink, None))
    }
    fn handle_with_cancellation<'a>(
        &'a self,
        msg: &'a Message,
        _stream: &'a str,
        sink: &'a dyn MessageSink,
        cancellation: RoleCancellation,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.process(msg, sink, Some(cancellation)))
    }
}
