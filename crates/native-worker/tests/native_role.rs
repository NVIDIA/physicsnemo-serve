use anyhow::{Result, bail};
use native_worker::{
    host::{Cancellation, HostClient, NativeHost},
    request::*,
    role::NativeRole,
};
use scicomp_rq::{Message, Output};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::watch;
use worker_runtime::traits::{BoxFuture, MessageSink, WorkerRole};
#[path = "../../worker-runtime/tests/support/mod.rs"]
mod support;

#[derive(Default)]
struct Sink {
    outputs: Mutex<Vec<Output>>,
    fail_commit: bool,
    failures_left: AtomicUsize,
    commits: Mutex<Vec<(String, Vec<Output>)>>,
    begins: Mutex<Vec<String>>,
    begin_failures_left: AtomicUsize,
    fail_begin: bool,
    retire_on_begin: Option<HostClient>,
    shutdown_on_begin: Option<watch::Sender<bool>>,
}
impl MessageSink for Sink {
    fn enqueue<'a>(
        &'a self,
        _: &'a str,
        _: &'a str,
        _: &'a str,
        _: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async { bail!("unexpected enqueue") })
    }
    fn ack_message<'a>(&'a self, _: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { bail!("unexpected ack") })
    }
    fn handoff<'a>(
        &'a self,
        _: &'a Message,
        _: &'a str,
        _: &'a str,
        _: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async { bail!("unexpected handoff") })
    }
    fn forward_many<'a>(
        &'a self,
        _: &'a Message,
        _: &'a [Output],
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        Box::pin(async { bail!("unguarded completion") })
    }
    fn begin_attempt<'a>(
        &'a self,
        _: &'a Message,
        attempt: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>>> {
        Box::pin(async move {
            self.begins.lock().unwrap().push(attempt.to_owned());
            if let Some(host) = &self.retire_on_begin {
                host.retire();
            }
            if let Some(shutdown) = &self.shutdown_on_begin {
                shutdown.send_replace(true);
            }
            if self.fail_begin
                || self
                    .begin_failures_left
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_ok()
            {
                bail!("BEGIN reply lost");
            }
            Ok(None)
        })
    }
    fn complete_attempt<'a>(
        &'a self,
        _: &'a Message,
        outputs: &'a [Output],
        attempt: &'a str,
    ) -> BoxFuture<'a, Result<Vec<String>>> {
        Box::pin(async move {
            self.commits
                .lock()
                .unwrap()
                .push((attempt.to_owned(), outputs.to_vec()));
            if self
                .failures_left
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                bail!("temporary Redis outage");
            }
            if self.fail_commit {
                bail!("Redis unavailable")
            }
            self.outputs.lock().unwrap().extend_from_slice(outputs);
            Ok(vec!["1-0".into(), "2-0".into()])
        })
    }
}
struct Fake {
    calls: Arc<AtomicUsize>,
}
impl NativeHost for Fake {
    fn run(&mut self, json: &str, cancel: &Cancellation) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let request: ExecutionRequest = serde_json::from_str(json)?;
        assert_eq!(request.permutation_seed, 1615895419752914611);
        assert!(!cancel.is_cancelled());
        for file in ["standardized.bin", "physical.bin"] {
            std::fs::write(request.output_dir.join(file), [0; 32])?;
        }
        std::fs::write(
            request.output_dir.join("metadata.json"),
            b"{\"point_count\":2}",
        )?;
        Ok("{\"point_count\":2}".into())
    }
}
async fn fixture() -> (
    tempfile::TempDir,
    NativeRole,
    watch::Sender<bool>,
    Arc<AtomicUsize>,
) {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("cache");
    tokio::fs::create_dir(&cache).await.unwrap();
    let sha = format!("{:x}", Sha256::digest(b"verified"));
    tokio::fs::write(cache.join(&sha), b"verified")
        .await
        .unwrap();
    let object = InputObject {
        url: "https://must-not-be-called.invalid/asset".into(),
        sha256: sha,
        size_bytes: 8,
    };
    let config = DeploymentConfig {
        schema_version: 1,
        workflow_id: "physicsnemo-cfd-transolver-native".into(),
        executor_class: "native.transolver.aoti".into(),
        backend: "aoti".into(),
        device_index: 0,
        compute_capability: [9, 0],
        target: None,
        packages: vec![],
        model_build_sources: Default::default(),
        stats_path: root.path().join("stats"),
        startup_fixture_path: None,
        model_assets: vec![],
        block_size: 60000,
        artifact_root: root.path().join("artifacts"),
        input_cache: cache,
        case: Case {
            case_id: "run_1".into(),
            point_count: 2,
            mesh: object.clone(),
            geometry: object,
        },
        request_timeout_seconds: 2,
        drain_timeout_seconds: 1,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let host = HostClient::spawn(move || Ok(Fake { calls: c }))
        .await
        .unwrap();
    let (shutdown, rx) = watch::channel(false);
    (
        root,
        NativeRole {
            config: config.into(),
            host,
            shutdown: rx,
        },
        shutdown,
        calls,
    )
}

struct GenericFake {
    calls: Arc<AtomicUsize>,
    escaping: bool,
}
impl NativeHost for GenericFake {
    fn run(&mut self, request: &str, _: &Cancellation) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let request: Value = serde_json::from_str(request)?;
        assert_eq!(request["parameters"]["x"], json!([1.0, 2.0]));
        assert!(std::path::Path::new(request["inputs"]["fixed"].as_str().unwrap()).is_absolute());
        let root = std::path::Path::new(request["output_dir"].as_str().unwrap());
        std::fs::write(
            root.join("answer.f32"),
            [2f32.to_le_bytes(), 4f32.to_le_bytes()].concat(),
        )?;
        Ok(json!({"payload":{"y":[2.0,4.0]},"metadata":{"adapter":"tensor.v1","session_load_count":1},
            "artifacts":[{"name":"vector","path":if self.escaping {"../outside"} else {"answer.f32"},"media_type":"application/octet-stream","primary":true}]}).to_string())
    }
}
async fn generic_fixture(
    escaping: bool,
) -> (
    tempfile::TempDir,
    NativeRole,
    watch::Sender<bool>,
    Arc<AtomicUsize>,
) {
    let (root, legacy, shutdown, calls) = fixture().await;
    let native_worker::deployment::RuntimeDeploymentConfig::Transolver(old) = legacy.config else {
        panic!()
    };
    let asset = old.input_cache.join(&old.case.mesh.sha256);
    let config=native_worker::deployment::WorkflowDeploymentConfig {
        schema_version:2,workflow_id:"native-tensor-test".into(),manifest_version:None,public_contract_digest:None,executor_class:"native.tensor.test".into(),device_index:0,
        target:serde_json::from_value(json!({"profile":"cpu-test-only","compilation":{"platform":"linux/amd64","gpu_product":"cpu-test-only","compute_capability":[0,0],"cuda_architecture":"none","precision":"fp32","mig":"disabled","hardware_compatibility":"qualified-product-only"},"deployment":{"minimum_available_memory_mb":1,"mig":"disabled","driver_requirements":{"minimum_driver_version":"0"}}})).unwrap(),
        workflow:json!({"schema_version":1,"adapter":"tensor.v1","models":{"primary":{"package_path":old.input_cache,"backend":"ort","device":"cpu","device_index":0}},"limits":{"max_tensor_bytes":1024}}),
        model_build_sources:std::collections::BTreeMap::from([("primary".into(),"a".repeat(40))]),
        request_schema:json!({"type":"object","additionalProperties":false,"required":["x"],"properties":{"x":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2}}}),
        response_schema:json!({"type":"object","additionalProperties":false,"required":["y"],"properties":{"y":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2}}}),
        model_assets:vec![ModelAsset { path:asset.clone(),sha256:old.case.mesh.sha256.clone(),size_bytes:8 }],
        input_artifacts:std::collections::BTreeMap::from([("fixed".into(),native_worker::deployment::InputArtifact::Local(ModelAsset { path:asset,sha256:old.case.mesh.sha256,size_bytes:8 }))]),
        startup_cases:vec![],ingress_root:None,input_files:Default::default(),artifact_root:old.artifact_root,input_cache:old.input_cache,request_timeout_seconds:2,drain_timeout_seconds:1,startup_timeout_seconds:1,
    };
    config.validate().unwrap();
    let c = calls.clone();
    let host = HostClient::spawn(move || Ok(GenericFake { calls: c, escaping }))
        .await
        .unwrap();
    let role = NativeRole {
        config: native_worker::deployment::RuntimeDeploymentConfig::Workflow(config),
        host,
        shutdown: legacy.shutdown,
    };
    (root, role, shutdown, calls)
}
fn generic_message(id: &str, parameters: Value) -> Message {
    let mut payload: Value = message(id, parameters).parse_payload().unwrap();
    payload["workflow_id"] = json!("native-tensor-test");
    payload["runtime"]["executor_class"] = json!("native.tensor.test");
    Message::new(
        id,
        "test:gpu0",
        "native:grp",
        "generic-run",
        payload.to_string(),
        "execute",
    )
}

struct UploadedFake {
    calls: Arc<AtomicUsize>,
    source: std::path::PathBuf,
}
impl NativeHost for UploadedFake {
    fn run(&mut self, text: &str, _: &Cancellation) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let request: Value = serde_json::from_str(text)?;
        let path = request["inputs"]["features"]
            .as_str()
            .expect("uploaded input was not admitted");
        assert_ne!(std::path::Path::new(path), self.source);
        assert_eq!(request["input_metadata"]["features"]["source"], "request");
        // An API staging change after admission cannot alter this native call.
        std::fs::write(&self.source, b"changed source")?;
        let data = std::fs::read(path)?;
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) * 2.0 + 1.0)
            .collect();
        Ok(json!({"payload":{"y":values},"metadata":{},"artifacts":[]}).to_string())
    }
}

async fn uploaded_fixture() -> (
    tempfile::TempDir,
    NativeRole,
    watch::Sender<bool>,
    std::path::PathBuf,
    Arc<AtomicUsize>,
) {
    let (root, old, shutdown, calls) = generic_fixture(false).await;
    let canonical_root = std::fs::canonicalize(root.path()).unwrap();
    let ingress = canonical_root.join("ingress");
    let source = ingress
        .join("generic-run")
        .join("12345678-1234-1234-1234-123456789abc-data.bin");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let mut value = serde_json::to_value(&old.config).unwrap();
    value["manifest_version"] = json!("1.2.3");
    value["public_contract_digest"] = json!("f".repeat(64));
    value["ingress_root"] = json!(ingress);
    value["artifact_root"] = json!(canonical_root.join("artifacts"));
    value["input_files"] = json!({"features":{"required":true,"media_types":["application/octet-stream"],"max_size_bytes":8}});
    value["request_schema"] = json!({"type":"object","required":["x"],"properties":{"x":{"type":"object"}},"additionalProperties":false});
    value["workflow"]["tensor_adapter"] = json!({"model_alias":"primary","inputs":{"input":{"request_field":"x","codec":"raw-fp32-le"}}});
    let package = std::path::Path::new(
        value["workflow"]["models"]["primary"]["package_path"]
            .as_str()
            .unwrap(),
    );
    std::fs::write(
        package.join("model.json"),
        json!({"inputs":[{"name":"input","dtype":"float32","shape":[2]}]}).to_string(),
    )
    .unwrap();
    let config: native_worker::deployment::WorkflowDeploymentConfig = serde_json::from_value(value)
        .expect("generated request-artifact deployment contract must be accepted");
    config.validate().unwrap();
    let captured = calls.clone();
    let original = source.clone();
    let host = HostClient::spawn(move || {
        Ok(UploadedFake {
            calls: captured,
            source: original,
        })
    })
    .await
    .unwrap();
    let role = NativeRole {
        config: native_worker::deployment::RuntimeDeploymentConfig::Workflow(config),
        host,
        shutdown: old.shutdown,
    };
    (root, role, shutdown, source, calls)
}

fn uploaded_message(source: &std::path::Path, id: &str, values: [f32; 2]) -> Message {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    std::fs::write(source, &bytes).unwrap();
    let mut payload: Value = generic_message(id, json!({"x":{"artifact":"features","dtype":"float32","shape":[2],"byte_order":"little","layout":"contiguous","size_bytes":8}})).parse_payload().unwrap();
    payload["manifest_version"] = json!("1.2.3");
    payload["request"] = json!({"content_type":"multipart/form-data","input_artifacts":[{
        "field_name":"features","name":"data.bin","artifact_id":"12345678-1234-1234-1234-123456789abc",
        "media_type":"application/octet-stream","size_bytes":8,"storage_path":source,
        "original_filename":"data.bin","sha256":format!("{:x}",Sha256::digest(&bytes))}]});
    Message::new(
        id,
        "test:gpu0",
        "native:grp",
        "generic-run",
        payload.to_string(),
        "execute",
    )
}

#[tokio::test]
async fn uploaded_inputs_are_owned_per_attempt_and_two_requests_use_the_same_host() {
    let (_root, role, _shutdown, source, calls) = uploaded_fixture().await;
    let sink = Sink::default();
    for (id, values) in [("1-0", [1.0, 2.0]), ("2-0", [3.0, 5.0])] {
        role.handle(&uploaded_message(&source, id, values), "gpu0", &sink)
            .await
            .unwrap();
    }
    let outputs = sink.outputs.lock().unwrap();
    let first: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    let second: Value = serde_json::from_str(outputs[3].payload()).unwrap();
    assert_eq!(first["status"], "completed", "{first}");
    assert_eq!(first["payload"]["y"], json!([3.0, 5.0]));
    assert_eq!(second["payload"]["y"], json!([7.0, 11.0]));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn uploaded_artifact_policy_rejects_invalid_bytes_ownership_and_contracts_before_cpp() {
    let mut violations = vec![
        "hash",
        "missing_digest",
        "cross_run",
        "size",
        "media",
        "unknown",
        "duplicate",
        "missing",
        "version",
        "shape",
        "dtype",
        "nonfinite",
    ];
    #[cfg(unix)]
    violations.push("symlink");
    for violation in violations {
        let (_root, role, _shutdown, source, calls) = uploaded_fixture().await;
        let mut payload: Value = uploaded_message(&source, "1-0", [1.0, 2.0])
            .parse_payload()
            .unwrap();
        match violation {
            "hash" => payload["request"]["input_artifacts"][0]["sha256"] = json!("0".repeat(64)),
            "missing_digest" => {
                payload["request"]["input_artifacts"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("sha256");
            }
            "cross_run" => {
                let other = source
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("another-run");
                std::fs::create_dir(&other).unwrap();
                let path = other.join(source.file_name().unwrap());
                std::fs::copy(&source, &path).unwrap();
                payload["request"]["input_artifacts"][0]["storage_path"] = json!(path);
            }
            "size" => payload["request"]["input_artifacts"][0]["size_bytes"] = json!(9),
            "media" => payload["request"]["input_artifacts"][0]["media_type"] = json!("text/plain"),
            "unknown" => payload["request"]["input_artifacts"][0]["field_name"] = json!("other"),
            "duplicate" => {
                let item = payload["request"]["input_artifacts"][0].clone();
                payload["request"]["input_artifacts"]
                    .as_array_mut()
                    .unwrap()
                    .push(item);
            }
            "missing" => payload["request"]["input_artifacts"] = json!([]),
            "version" => payload["manifest_version"] = json!("wrong-version"),
            "shape" => payload["parameters"]["x"]["shape"] = json!([1, 2]),
            "dtype" => payload["parameters"]["x"]["dtype"] = json!("float16"),
            "nonfinite" => {
                let data: Vec<u8> = [f32::NAN, 2.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect();
                std::fs::write(&source, &data).unwrap();
                payload["request"]["input_artifacts"][0]["sha256"] =
                    json!(format!("{:x}", Sha256::digest(&data)));
            }
            #[cfg(unix)]
            "symlink" => {
                let target = source.with_extension("original");
                std::fs::rename(&source, &target).unwrap();
                std::os::unix::fs::symlink(&target, &source).unwrap();
            }
            _ => unreachable!(),
        }
        let message = Message::new(
            "1-0",
            "test:gpu0",
            "native:grp",
            "generic-run",
            payload.to_string(),
            "execute",
        );
        let sink = Sink::default();
        role.handle(&message, "gpu0", &sink).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{violation} reached C++");
        let result: Value =
            serde_json::from_str(sink.outputs.lock().unwrap()[1].payload()).unwrap();
        assert_eq!(result["status"], "failed", "{violation}: {result}");
    }
}

#[tokio::test]
async fn request_input_names_cannot_override_release_assets() {
    let (_root, role, _shutdown, _source, _calls) = uploaded_fixture().await;
    let mut config = serde_json::to_value(&role.config).unwrap();
    config["input_files"]["fixed"] = config["input_files"]["features"].clone();
    let config: native_worker::deployment::WorkflowDeploymentConfig =
        serde_json::from_value(config).unwrap();
    assert!(
        config.validate().is_err(),
        "request file can override release input"
    );
}

#[tokio::test]
async fn interrupted_native_call_keeps_admitted_bytes_until_drain_then_reclaims_them() {
    use std::sync::atomic::AtomicBool;
    struct HoldingUpload {
        entered: Option<tokio::sync::oneshot::Sender<std::path::PathBuf>>,
        finish: Arc<AtomicBool>,
        survived: Arc<AtomicBool>,
    }
    impl NativeHost for HoldingUpload {
        fn run(&mut self, text: &str, _: &Cancellation) -> Result<String> {
            let request: Value = serde_json::from_str(text)?;
            let path = std::path::PathBuf::from(request["inputs"]["features"].as_str().unwrap());
            self.entered.take().unwrap().send(path.clone()).unwrap();
            // Model a native call that cannot immediately respond to cancellation.
            while !self.finish.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            self.survived.store(
                std::fs::read(path)? == [1f32.to_le_bytes(), 2f32.to_le_bytes()].concat(),
                Ordering::SeqCst,
            );
            Ok(json!({"payload":{"y":[3,5]},"metadata":{},"artifacts":[]}).to_string())
        }
    }
    let (_root, mut role, _shutdown, source, _calls) = uploaded_fixture().await;
    let (entered, ready) = tokio::sync::oneshot::channel();
    let finish = Arc::new(AtomicBool::new(false));
    let survived = Arc::new(AtomicBool::new(false));
    let host_finish = finish.clone();
    let host_survived = survived.clone();
    role.host = HostClient::spawn(move || {
        Ok(HoldingUpload {
            entered: Some(entered),
            finish: host_finish,
            survived: host_survived,
        })
    })
    .await
    .unwrap();
    let host = role.host.clone();
    let artifact_root = role.config.artifact_root().to_owned();
    let message = uploaded_message(&source, "1-0", [1.0, 2.0]);
    let task = tokio::spawn(async move { role.handle(&message, "gpu0", &Sink::default()).await });
    let pinned = ready.await.unwrap();
    task.abort();
    let _ = task.await;
    std::fs::remove_dir_all(source.parent().unwrap()).unwrap();
    // The API's retention cleaner excludes .cache but may remove normal attempts.
    for entry in std::fs::read_dir(&artifact_root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != ".cache" {
            std::fs::remove_dir_all(entry.path()).unwrap();
        }
    }
    assert!(
        pinned.is_file(),
        "cancellation or artifact retention unlinked active input"
    );
    finish.store(true, Ordering::SeqCst);
    assert!(host.shutdown(std::time::Duration::from_secs(2)).await);
    assert!(
        survived.load(Ordering::SeqCst),
        "native call lost its owned input bytes"
    );
    assert!(
        !pinned.parent().unwrap().exists(),
        "completed native drain leaked the owned attempt inputs"
    );
}

#[tokio::test]
async fn native_thread_failure_reclaims_its_owned_input_copy() {
    struct CrashingUpload {
        entered: Option<tokio::sync::oneshot::Sender<std::path::PathBuf>>,
    }
    impl NativeHost for CrashingUpload {
        fn run(&mut self, text: &str, _: &Cancellation) -> Result<String> {
            let request: Value = serde_json::from_str(text)?;
            self.entered
                .take()
                .unwrap()
                .send(std::path::PathBuf::from(
                    request["inputs"]["features"].as_str().unwrap(),
                ))
                .unwrap();
            panic!("simulated native upload failure")
        }
    }
    let (_root, mut role, _shutdown, source, _calls) = uploaded_fixture().await;
    let (entered, ready) = tokio::sync::oneshot::channel();
    role.host = HostClient::spawn(move || {
        Ok(CrashingUpload {
            entered: Some(entered),
        })
    })
    .await
    .unwrap();
    let host = role.host.clone();
    let msg = uploaded_message(&source, "1-0", [1.0, 2.0]);
    let task = tokio::spawn(async move { role.handle(&msg, "gpu0", &Sink::default()).await });
    let pinned = ready.await.unwrap();
    assert!(
        task.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("deferred")
    );
    assert!(host.shutdown(std::time::Duration::from_secs(2)).await);
    assert!(
        !pinned.parent().unwrap().exists(),
        "exited native thread leaked its input copy"
    );
    assert!(source.is_file(), "native cleanup removed API staging input");
}

#[tokio::test]
async fn cancelled_submission_reclaims_inputs_that_never_entered_cpp() {
    struct Ready;
    impl NativeHost for Ready {
        fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
            Ok("{}".into())
        }
    }
    let (_root, mut role, _shutdown, source, _calls) = uploaded_fixture().await;
    role.host = HostClient::spawn(|| Ok(Ready)).await.unwrap();
    let host = role.host.clone();
    let first = host
        .submit("{}".into())
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    let cache = role
        .config
        .artifact_root()
        .join(".cache/native-inputs")
        .join(format!("{:x}", Sha256::digest(b"generic-run")));
    let msg = uploaded_message(&source, "1-0", [1.0, 2.0]);
    let task = tokio::spawn(async move { role.handle(&msg, "gpu0", &Sink::default()).await });
    let pinned = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Ok(entries) = std::fs::read_dir(&cache) {
                for entry in entries {
                    let path = entry.unwrap().path().join("features");
                    if std::fs::metadata(&path).is_ok_and(|m| m.len() == 8) {
                        return path;
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    let _ = task.await;
    assert!(
        !pinned.parent().unwrap().exists(),
        "cancelled submission leaked its unqueued input copy"
    );
    assert!(source.is_file(), "cleanup removed the API's staging input");
    first.committed();
    assert!(host.shutdown(std::time::Duration::from_secs(2)).await);
}

#[tokio::test]
async fn retired_worker_defers_deliveries_before_claiming_or_copying_inputs() {
    let (_root, role, _shutdown, source, calls) = uploaded_fixture().await;
    role.host.retire();
    let cache = role
        .config
        .artifact_root()
        .join(".cache/native-inputs")
        .join(format!("{:x}", Sha256::digest(b"generic-run")));
    let sink = Sink::default();
    let error = role
        .handle(&uploaded_message(&source, "1-0", [1.0, 2.0]), "gpu0", &sink)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("deferred"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(sink.begins.lock().unwrap().is_empty());
    assert!(sink.commits.lock().unwrap().is_empty());
    assert!(sink.outputs.lock().unwrap().is_empty());
    assert!(!cache.exists(), "retired admission copied input bytes");
}

#[tokio::test]
async fn rejected_native_submission_removes_inputs_that_never_entered_cpp() {
    let (_root, role, _shutdown, source, calls) = uploaded_fixture().await;
    let cache = role
        .config
        .artifact_root()
        .join(".cache/native-inputs")
        .join(format!("{:x}", Sha256::digest(b"generic-run")));
    let sink = Sink {
        retire_on_begin: Some(role.host.clone()),
        ..Default::default()
    };
    let error = role
        .handle(&uploaded_message(&source, "1-0", [1.0, 2.0]), "gpu0", &sink)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("deferred"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(sink.outputs.lock().unwrap().is_empty());
    assert!(
        std::fs::read_dir(cache).unwrap().next().is_none(),
        "rejected submission leaked its owned input copy"
    );
}

#[tokio::test]
async fn native_thread_exit_defers_the_delivery_without_publishing_failure() {
    struct Crashing;
    impl NativeHost for Crashing {
        fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
            panic!("simulated native thread failure")
        }
    }
    let (_root, mut role, _shutdown, _calls) = generic_fixture(false).await;
    role.host = HostClient::spawn(|| Ok(Crashing)).await.unwrap();
    let sink = Sink::default();
    let error = role
        .handle(
            &generic_message("1-0", json!({"x":[1.0,2.0]})),
            "gpu0",
            &sink,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("deferred"));
    assert!(role.host.is_retired());
    assert!(sink.commits.lock().unwrap().is_empty());
    assert!(sink.outputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn retirement_after_claim_defers_unexecuted_requests_without_release() {
    for generic in [false, true] {
        for shutdown_after_claim in [false, true] {
            let (_root, role, shutdown, calls) = if generic {
                generic_fixture(false).await
            } else {
                fixture().await
            };
            let sink = Sink {
                retire_on_begin: Some(role.host.clone()),
                shutdown_on_begin: shutdown_after_claim.then_some(shutdown),
                ..Default::default()
            };
            let msg = if generic {
                generic_message("1-0", json!({"x":[1.0,2.0]}))
            } else {
                message("1-0", json!({"case_id":"run_1"}))
            };
            let error = role.handle(&msg, "gpu0", &sink).await.unwrap_err();
            assert!(error.to_string().contains("deferred"), "{error:#}");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(sink.begins.lock().unwrap().len(), 1);
            assert!(sink.commits.lock().unwrap().is_empty());
            assert!(sink.outputs.lock().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn generic_models_require_exact_per_alias_compilation_sources() {
    let (_root, role, _shutdown, _calls) = generic_fixture(false).await;
    let mut value = serde_json::to_value(&role.config).unwrap();
    value["model_build_sources"] = json!({"primary":"a".repeat(40)});
    let valid: native_worker::deployment::WorkflowDeploymentConfig =
        serde_json::from_value(value.clone()).expect("per-model compilation identity is supported");
    valid.validate().unwrap();
    for invalid in [
        json!({}),
        json!({"other":"a".repeat(40)}),
        json!({"primary":"short"}),
        json!({"primary":"z".repeat(40)}),
        json!({"primary":"a".repeat(40),"extra":"b".repeat(40)}),
    ] {
        value["model_build_sources"] = invalid;
        let invalid: native_worker::deployment::WorkflowDeploymentConfig =
            serde_json::from_value(value.clone()).unwrap();
        assert!(
            invalid.validate().is_err(),
            "unbound model source was admitted"
        );
    }
}

#[tokio::test]
async fn managed_deployment_cannot_use_a_host_that_ignores_the_new_contract() {
    let (_root, role, _shutdown, _calls) = generic_fixture(false).await;
    let mut value = serde_json::to_value(&role.config).unwrap();
    value["schema_version"] = json!(3);
    value["manifest_version"] = json!("1.2.3");
    value["public_contract_digest"] = json!("a".repeat(64));
    let config: native_worker::deployment::WorkflowDeploymentConfig =
        serde_json::from_value(value.clone()).unwrap();
    assert!(
        config.validate().is_err(),
        "managed deployment admitted legacy host schema1"
    );
    value["workflow"]["schema_version"] = json!(2);
    let config: native_worker::deployment::WorkflowDeploymentConfig =
        serde_json::from_value(value).unwrap();
    config.validate().unwrap();
}

#[tokio::test]
async fn transolver_backend_is_explicit_and_matches_executor_class() {
    let (_root, role, _shutdown, _calls) = fixture().await;
    let mut value = serde_json::to_value(&role.config).unwrap();
    value["case"]["point_count"] = json!(8_828_095);
    value["packages"] = json!(["/release/full", "/release/tail"]);
    value["model_assets"] =
        json!([{"path":value["stats_path"],"sha256":"a".repeat(64),"size_bytes":1}]);
    value.as_object_mut().unwrap().remove("backend");
    let legacy: DeploymentConfig = serde_json::from_value(value.clone()).unwrap();
    legacy.validate().unwrap();
    for backend in ["aoti", "tensorrt"] {
        value["backend"] = json!(backend);
        value["executor_class"] = json!(format!("native.transolver.{backend}"));
        let config: DeploymentConfig = serde_json::from_value(value.clone())
            .expect("selected native backend must deserialize");
        config.validate().unwrap();
        value["executor_class"] = json!("native.transolver.other");
        let config: DeploymentConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(config.validate().is_err());
    }
    value["backend"] = json!("unknown");
    value["executor_class"] = json!("native.transolver.unknown");
    let config: DeploymentConfig = serde_json::from_value(value).unwrap();
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn transolver_result_reports_selected_tensorrt_backend() {
    let (_root, mut role, _shutdown, calls) = fixture().await;
    let mut config = serde_json::to_value(&role.config).unwrap();
    config["backend"] = json!("tensorrt");
    config["executor_class"] = json!("native.transolver.tensorrt");
    role.config = serde_json::from_value(config).expect("TensorRT CFD configuration must parse");
    let original = message("1-0", json!({"case_id":"run_1","seed":42}));
    let mut payload: Value = serde_json::from_str(original.payload()).unwrap();
    payload["runtime"]["executor_class"] = json!("native.transolver.tensorrt");
    let msg = Message::new(
        "1-0",
        "test:gpu0",
        "native:grp",
        "test-run",
        payload.to_string(),
        "execute",
    );
    let sink = Sink::default();
    role.handle(&msg, "gpu0", &sink).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "completed");
    assert_eq!(result["payload"]["native"]["backend"], "tensorrt");
}

#[tokio::test]
async fn transolver_deployments_accept_and_validate_full_target_policy() {
    let (_root, role, _shutdown, _calls) = fixture().await;
    let mut value = serde_json::to_value(&role.config).unwrap();
    value["case"]["point_count"] = json!(8_828_095);
    value["packages"] = json!(["/release/full", "/release/tail"]);
    value["model_assets"] =
        json!([{"path":value["stats_path"],"sha256":"a".repeat(64),"size_bytes":1}]);
    value["target"] = json!({"profile":"h100-sxm-v1","compilation":{"platform":"linux/amd64","gpu_product":"NVIDIA H100 80GB HBM3","compute_capability":[9,0],"cuda_architecture":"sm90","precision":"fp32","mig":"disabled","hardware_compatibility":"qualified-product-only"},"deployment":{"minimum_available_memory_mb":70000,"mig":"disabled","driver_requirements":{"minimum_driver_version":"570.195.03","maximum_driver_version":"570.195.03"}}});
    let config: DeploymentConfig =
        serde_json::from_value(value.clone()).expect("v1 full target policy is supported");
    config.validate().unwrap();
    value["target"]["compilation"]["gpu_product"] = json!("NVIDIA H200");
    let config: DeploymentConfig = serde_json::from_value(value).unwrap();
    assert!(
        config.validate().is_err(),
        "same-SM target widened v1 qualification"
    );
}

#[tokio::test]
async fn transolver_models_require_valid_per_alias_compilation_identity_when_supplied() {
    let (_root, role, _shutdown, _calls) = fixture().await;
    let mut value = serde_json::to_value(&role.config).unwrap();
    value["case"]["point_count"] = json!(8_828_095);
    value["packages"] = json!(["/release/full", "/release/tail"]);
    value["model_assets"] =
        json!([{"path":value["stats_path"],"sha256":"a".repeat(64),"size_bytes":1}]);
    value["model_build_sources"] = json!({"coarse":"a".repeat(40),"tail":"b".repeat(40)});
    let config: DeploymentConfig = serde_json::from_value(value.clone())
        .expect("v1 per-model compilation sources are supported");
    config.validate().unwrap();
    for invalid in [
        json!({"coarse":"a".repeat(40)}),
        json!({"coarse":"a".repeat(40),"tail":"b".repeat(40),"extra":"c".repeat(40)}),
        json!({"../coarse":"a".repeat(40),"tail":"b".repeat(40)}),
        json!({"coarse":"short","tail":"b".repeat(40)}),
        json!({"coarse":"z".repeat(40),"tail":"b".repeat(40)}),
    ] {
        value["model_build_sources"] = invalid;
        let config: DeploymentConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(
            config.validate().is_err(),
            "unbound CFD model source was admitted"
        );
    }
}

#[tokio::test]
async fn transolver_results_preserve_legacy_common_and_mixed_model_sources() {
    use std::collections::BTreeMap;
    let first = "a".repeat(40);
    let second = "b".repeat(40);
    let legacy = "c".repeat(40);
    for sources in [
        BTreeMap::new(),
        BTreeMap::from([
            ("renamed-full".into(), first.clone()),
            ("tail".into(), first.clone()),
        ]),
        BTreeMap::from([
            ("renamed-full".into(), first.clone()),
            ("tail".into(), second.clone()),
        ]),
    ] {
        let (_root, mut role, _shutdown, _calls) = fixture().await;
        let native_worker::deployment::RuntimeDeploymentConfig::Transolver(config) =
            &mut role.config
        else {
            panic!()
        };
        config.model_build_sources = sources.clone();
        let proof = config.model_build_provenance(&legacy);
        if sources.is_empty() {
            assert_eq!(proof["model_build_source_sha"], legacy);
            assert!(!proof.contains_key("model_build_sources"));
        } else {
            assert_eq!(proof["model_build_sources"], json!(sources));
            assert_ne!(
                proof.get("model_build_source_sha"),
                Some(&json!(legacy)),
                "qualification/global SHA replaced package sources"
            );
        }
        let sink = Sink::default();
        role.handle(
            &message("1-0", json!({"case_id":"run_1","seed":42})),
            "gpu0",
            &sink,
        )
        .await
        .unwrap();
        let outputs = sink.outputs.lock().unwrap();
        let outcome: Value = serde_json::from_str(outputs[1].payload()).unwrap();
        assert_eq!(outcome["status"], "completed");
        let native = &outcome["payload"]["native"];
        if sources.is_empty() {
            assert!(native.get("model_build_sources").is_none());
            assert_eq!(
                native["model_build_source_sha"],
                std::env::var("MODEL_BUILD_SOURCE_SHA").unwrap_or_default()
            );
        } else {
            assert_eq!(native["model_build_sources"], json!(sources));
            if sources.values().all(|source| source == &first) {
                assert_eq!(native["model_build_source_sha"], first);
            } else {
                assert!(
                    native.get("model_build_source_sha").is_none(),
                    "mixed-source model result advertised a misleading global compilation source"
                );
            }
        }
    }
}

#[cfg(all(feature = "cpp", target_os = "linux"))]
#[tokio::test]
async fn cpu_fixture_requires_opt_in_before_nvml_or_redis() {
    let (root, role, _shutdown, _calls) = generic_fixture(false).await;
    let native_worker::deployment::RuntimeDeploymentConfig::Workflow(mut config) = role.config
    else {
        panic!()
    };
    let package = root.path().join("cpu-package");
    std::fs::create_dir(&package).unwrap();
    let path = package.join("model.fixture");
    std::fs::write(&path, b"meshdata").unwrap();
    config.workflow["models"]["primary"]["package_path"] = json!(package);
    config.model_assets = vec![ModelAsset {
        path,
        sha256: format!("{:x}", Sha256::digest(b"meshdata")),
        size_bytes: 8,
    }];
    let path = root.path().join("cpu-deployment.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_native-worker"))
        .arg(path)
        .env_remove("NATIVE_ALLOW_CPU_TEST_ONLY")
        .env("SOURCE_SHA", "a".repeat(40))
        .env(
            "RELEASE_IMAGE_DIGEST",
            format!("local/cpu-fixture@sha256:{}", "b".repeat(64)),
        )
        .env("REDIS_URL", "redis://127.0.0.1:1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("CPU fixture execution requires NATIVE_ALLOW_CPU_TEST_ONLY=1"),
        "CPU fixture reached NVML/Redis instead of rejecting missing opt-in: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(all(feature = "cpp", target_os = "linux"))]
#[test]
fn cpu_fixture_reaches_cpp_startup_with_small_ambient_worker_stacks() {
    // Keep the test's Redis client off the intentionally small child stacks.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let (_redis, qm) =
                        support::spawn_test_queue_manager("native-runtime-stack").await;
                    let mut connection = qm.connection();
                    let settings: std::collections::BTreeMap<String, String> = redis::cmd("CONFIG")
                        .arg("GET")
                        .arg("port")
                        .query_async(&mut connection)
                        .await
                        .unwrap();
                    let (root, role, _shutdown, _calls) = generic_fixture(false).await;
                    let mut config = serde_json::to_value(&role.config).unwrap();
                    let package = root.path().join("empty-cpu-package");
                    std::fs::create_dir(&package).unwrap();
                    config["workflow"]["adapter"] = json!("custom.v1");
                    config["workflow"]["models"]["primary"]["package_path"] = json!(package);
                    config["startup_cases"] = json!([{
                        "name":"cpu-fixture", "request":{"x":[1,2]},
                        "expected":{"y":[3,5]}
                    }]);
                    // No model is loaded: the C++ host rejects missing self_test first.
                    let path = root.path().join("cpu-deployment.json");
                    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
                    let stdout_path = root.path().join("worker.stdout");
                    let stderr_path = root.path().join("worker.stderr");
                    let mut child =
                        tokio::process::Command::new(env!("CARGO_BIN_EXE_native-worker"))
                            .arg(path)
                            .env("NATIVE_ALLOW_CPU_TEST_ONLY", "1")
                            .env("RUST_MIN_STACK", (2 * 1024 * 1024).to_string())
                            .env("SOURCE_SHA", "a".repeat(40))
                            .env(
                                "RELEASE_IMAGE_DIGEST",
                                format!("local/cpu-fixture@sha256:{}", "b".repeat(64)),
                            )
                            .env(
                                "REDIS_URL",
                                format!("redis://127.0.0.1:{}", settings["port"]),
                            )
                            .env_remove("PNMS_BUNDLE_DIGEST")
                            .env_remove("PNMS_IMAGE_CONFIG_DIGEST")
                            .env_remove("NATIVE_DEPLOYMENT_OPERATION_ID")
                            .env_remove("PNMS_DEVICE_UUID")
                            .stdout(std::fs::File::create(stdout_path).unwrap())
                            .stderr(std::fs::File::create(&stderr_path).unwrap())
                            .kill_on_drop(true)
                            .spawn()
                            .unwrap();
                    let status = match tokio::time::timeout(
                        std::time::Duration::from_secs(15),
                        child.wait(),
                    )
                    .await
                    {
                        Ok(status) => status.unwrap(),
                        Err(_) => {
                            child.kill().await.unwrap();
                            child.wait().await.unwrap();
                            panic!("CPU fixture worker exceeded its startup deadline");
                        }
                    };
                    let stderr = std::fs::read_to_string(stderr_path).unwrap();
                    assert_eq!(
                        status.code(),
                        Some(1),
                        "worker did not report a normal C++ configuration failure: {stderr}"
                    );
                    assert!(stderr.contains("key 'self_test' not found"), "{stderr}");
                    let stats: String = redis::cmd("INFO")
                        .arg("commandstats")
                        .query_async(&mut connection)
                        .await
                        .unwrap();
                    assert!(
                        stats
                            .lines()
                            .any(|line| line.starts_with("cmdstat_hset:calls=")),
                        "worker never registered its warming state: {stats}"
                    );
                    let registrations: std::collections::HashMap<String, String> =
                        redis::cmd("HGETALL")
                            .arg("gpu:registry")
                            .query_async(&mut connection)
                            .await
                            .unwrap();
                    assert_eq!(
                        registrations.len(),
                        1,
                        "failed startup lost its logical slot"
                    );
                    let (field, raw) = registrations.into_iter().next().unwrap();
                    let slot: Value = serde_json::from_str(&raw).unwrap();
                    assert_eq!(slot["workflow_id"], config["workflow_id"]);
                    assert_eq!(slot["status"], "warming");
                    assert_eq!(slot["model_cache"]["entries"], json!([]));
                    assert_eq!(slot["model_cache"]["warmup"]["status"], "warming");
                    assert_eq!(slot["model_cache"]["warmup"]["reason"], "restart_pending");
                    let lease: Option<String> = redis::cmd("GET")
                        .arg(format!(
                            "native:worker-lease:{:x}",
                            Sha256::digest(field.as_bytes())
                        ))
                        .query_async(&mut connection)
                        .await
                        .unwrap();
                    assert!(lease.is_none(), "failed startup left a live process lease");
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn generic_tensor_outputs_are_not_forced_into_cfd_field_shapes() {
    let (_root, role, _shutdown, calls) = generic_fixture(false).await;
    let sink = Sink::default();
    for id in ["1-0", "2-0"] {
        role.handle(&generic_message(id, json!({"x":[1.0,2.0]})), "gpu0", &sink)
            .await
            .unwrap();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "completed");
    assert_eq!(result["payload"]["y"], json!([2.0, 4.0]));
    assert_eq!(
        result["execution"]["status"], "succeeded",
        "canonical execution status must use the persisted API contract"
    );
    let native_worker::deployment::RuntimeDeploymentConfig::Workflow(config) = &role.config else {
        panic!()
    };
    assert!(
        jsonschema::validator_for(&config.response_schema)
            .unwrap()
            .is_valid(&result["payload"]),
        "canonical response no longer satisfies its authored strict schema"
    );
    assert_eq!(result["execution"]["outputs"][0]["size_bytes"], 8);
    assert_eq!(result["execution"]["native"]["adapter"], "tensor.v1");
    assert_eq!(
        result["execution"]["native"]["model_build_sources"],
        json!({"primary":"a".repeat(40)})
    );
    assert!(
        result["execution"]["native"]
            .get("model_build_source_sha")
            .is_none()
    );
    let release: Value = serde_json::from_str(outputs[0].payload()).unwrap();
    assert_eq!(release["memory_mb"], 12345);
    assert_eq!(release["status"], "completed");
}
#[tokio::test]
async fn generic_schema_rejects_invalid_input_before_native_execution() {
    let (_root, role, _shutdown, calls) = generic_fixture(false).await;
    let sink = Sink::default();
    role.handle(
        &generic_message("1-0", json!({"x":[1.0],"path":"/caller/path"})),
        "gpu0",
        &sink,
    )
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "failed");
    assert_eq!(result["execution"]["status"], "failed");
    let release: Value = serde_json::from_str(outputs[0].payload()).unwrap();
    assert_eq!(release["status"], "failed");
    assert!(
        result["execution"]["error"]
            .as_str()
            .unwrap()
            .contains("schema")
    );
}
#[tokio::test]
async fn generic_output_path_escape_is_failed_without_registering_artifacts() {
    let (_root, role, _shutdown, calls) = generic_fixture(true).await;
    let sink = Sink::default();
    role.handle(
        &generic_message("1-0", json!({"x":[1.0,2.0]})),
        "gpu0",
        &sink,
    )
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "failed");
    assert!(
        result["execution"]["error"]
            .as_str()
            .unwrap()
            .contains("relative and contained")
    );
    assert_eq!(result["payload"], json!({}));
}
fn message(id: &str, fields: Value) -> Message {
    message_with_raw_fields(id, fields.clone(), fields)
}

fn message_with_raw_fields(id: &str, parameters: Value, raw_fields: Value) -> Message {
    Message::new(
        id,
        "test:gpu0",
        "native:grp",
        "../../untrusted-run",
        json!({
            "workflow_id":"physicsnemo-cfd-transolver-native","resource_id":0,"memory_mb":12345,
            "runtime":{"kind":"cpp","executor_class":"native.transolver.aoti"},
            "parameters":parameters,
            "request":{"raw_fields":raw_fields}
        })
        .to_string(),
        "execute",
    )
}

#[tokio::test]
async fn api_wrapped_parameters_reach_native_execution() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink::default();
    let parameters = json!({"case_id":"run_1","seed":42,"capture_inputs":false});
    // The HTTP API retains the original body in request.raw_fields and puts
    // schema-validated parameters in the canonical envelope field.
    let msg = message_with_raw_fields(
        "1-0",
        parameters.clone(),
        json!({"operation":"run","parameters":parameters}),
    );
    role.handle(&msg, "gpu0", &sink).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "completed");
    assert_eq!(result["payload"]["native"]["case_id"], "run_1");
}

#[tokio::test]
async fn raw_fields_cannot_override_invalid_canonical_parameters() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink::default();
    let msg = message_with_raw_fields(
        "1-0",
        json!({"case_id":"run_2"}),
        json!({"case_id":"run_1"}),
    );
    role.handle(&msg, "gpu0", &sink).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "failed");
}

#[tokio::test]
async fn two_requests_reuse_host_and_commit_exact_reservations_and_distinct_artifacts() {
    let (root, role, _shutdown, calls) = fixture().await;
    let sink = Sink::default();
    for id in ["1-0", "2-0"] {
        role.handle(&message(id, json!({"case_id":"run_1"})), "gpu0", &sink)
            .await
            .unwrap();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let outputs = sink.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 4);
    let mut paths = vec![];
    for pair in outputs.chunks(2) {
        assert_eq!(pair[0].stream(), "release");
        assert_eq!(pair[1].stream(), "results");
        let result: Value = serde_json::from_str(pair[1].payload()).unwrap();
        let release: Value = serde_json::from_str(pair[0].payload()).unwrap();
        assert_eq!(result["status"], "completed");
        assert_eq!(result["payload"]["artifacts"].as_array().unwrap().len(), 3);
        assert_eq!(release["memory_mb"], 12345);
        assert_eq!(release["resource_id"], 0);
        let path = std::path::PathBuf::from(result["payload"]["output_path"].as_str().unwrap());
        assert!(path.starts_with(root.path().join("artifacts")));
        assert!(
            path.strip_prefix(root.path().join("artifacts"))
                .unwrap()
                .components()
                .next()
                .unwrap()
                .as_os_str()
                .to_str()
                .unwrap()
                .starts_with("native-")
        );
        assert!(path.is_file());
        paths.push(path);
    }
    assert_ne!(paths[0], paths[1]);
    assert!(!role.host.is_retired());
}
#[tokio::test]
async fn invalid_case_is_terminal_failure_and_releases_original_reservation_without_execution() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink::default();
    role.handle(&message("1-0", json!({"case_id":"run_2"})), "gpu0", &sink)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let outputs = sink.outputs.lock().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(outputs[1].payload()).unwrap()["status"],
        "failed"
    );
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0].payload()).unwrap()["memory_mb"],
        12345
    );
}
#[tokio::test]
async fn retired_worker_preserves_queued_backlog_for_replacement() {
    use worker_runtime::{
        config::RuntimeConfig, engine::EngineBuilder, transport::redis::RedisTransport,
    };
    let (_redis, qm) = support::spawn_test_queue_manager("native-retired-backlog").await;
    let (_root, role, _shutdown, calls) = generic_fixture(false).await;
    let host = role.host.clone();
    let config: RuntimeConfig = serde_json::from_value(json!({"stream_prefix":"test:","max_retries":1,"shared_dlq_stream":"dlq","streams":["gpu0","results","release","dlq"],"roles":{"native":{"inputs":[{"stream":"gpu0","max_dequeue_items":1,"poll_interval_ms":1,"block_ms":1,"reclaim_idle_ms":60000}],"outputs":["results","release"]}}})).unwrap();
    let engine = EngineBuilder::new(&config, "native")
        .role(Box::new(role))
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .consumer("original-native")
        .build()
        .unwrap();
    engine.ensure_consumer_groups().await.unwrap();
    let payload: Value = generic_message("unused", json!({"x":[1.0,2.0]}))
        .parse_payload()
        .unwrap();
    let mut ids = Vec::new();
    for run in ["first-run", "queued-run"] {
        ids.push(
            qm.enqueue_to_stream(
                &scicomp_rq::StreamKey::new("test:gpu0"),
                run,
                &payload.to_string(),
                "execute",
            )
            .await
            .unwrap(),
        );
    }
    let group = "gpu0:grp";
    let id = &ids[0];
    let poison = format!(
        "native-attempt:9:test:gpu0:{}:{group}:{}:{id}",
        group.len(),
        id.len()
    );
    let mut conn = qm.connection();
    let _: () = redis::cmd("SET")
        .arg(&poison)
        .arg("wrong-type-sentinel")
        .query_async(&mut conn)
        .await
        .unwrap();
    let _ = engine.run_once().await.unwrap();
    assert!(host.is_retired());
    let second = engine.run_once().await.unwrap();
    assert_eq!(
        second.acked, 0,
        "retired worker acknowledged a request it never ran"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let pending: Vec<(String, String, u64, u64)> = redis::cmd("XPENDING")
        .arg("test:gpu0")
        .arg(group)
        .arg("-")
        .arg("+")
        .arg(10)
        .query_async(&mut conn)
        .await
        .unwrap();
    assert_eq!(
        pending.iter().map(|item| &item.0).collect::<Vec<_>>(),
        ids.iter().collect::<Vec<_>>()
    );
    for stream in ["test:results", "test:release", "test:dlq"] {
        let count: u64 = redis::cmd("XLEN")
            .arg(stream)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(count, 0, "retired worker published to {stream}");
    }
    let _: usize = redis::cmd("DEL")
        .arg(&poison)
        .query_async(&mut conn)
        .await
        .unwrap();
    assert!(host.shutdown(std::time::Duration::from_secs(1)).await);
    let (_replacement_root, replacement, _replacement_shutdown, replacement_calls) =
        generic_fixture(false).await;
    // Age the pending deliveries without shrinking the production lease.
    for id in &ids {
        let _: Vec<String> = redis::cmd("XCLAIM")
            .arg("test:gpu0")
            .arg(group)
            .arg("original-native")
            .arg(0)
            .arg(id)
            .arg("IDLE")
            .arg(61000)
            .arg("JUSTID")
            .query_async(&mut conn)
            .await
            .unwrap();
    }
    let replacement = EngineBuilder::new(&config, "native")
        .role(Box::new(replacement))
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .consumer("replacement-native")
        .build()
        .unwrap();
    replacement.ensure_consumer_groups().await.unwrap();
    let _ = replacement.run_once().await.unwrap();
    let _ = replacement.run_once().await.unwrap();
    assert_eq!(replacement_calls.load(Ordering::SeqCst), 2);
    let pending: Vec<(String, String, u64, u64)> = redis::cmd("XPENDING")
        .arg("test:gpu0")
        .arg(group)
        .arg("-")
        .arg("+")
        .arg(10)
        .query_async(&mut conn)
        .await
        .unwrap();
    assert!(pending.is_empty());
    for stream in ["test:results", "test:release"] {
        let entries: Vec<(String, Vec<String>)> = redis::cmd("XRANGE")
            .arg(stream)
            .arg("-")
            .arg("+")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(entries.len(), 2);
        let mut runs = std::collections::BTreeSet::new();
        for (_, fields) in entries {
            let fields: std::collections::BTreeMap<_, _> = fields
                .chunks_exact(2)
                .map(|pair| (pair[0].as_str(), pair[1].as_str()))
                .collect();
            let output: Value = serde_json::from_str(fields["payload"]).unwrap();
            assert_eq!(output["status"], "completed");
            assert!(
                runs.insert(output["run_id"].as_str().unwrap().to_owned()),
                "duplicate completion"
            );
        }
        assert_eq!(
            runs,
            std::collections::BTreeSet::from(["first-run".to_owned(), "queued-run".to_owned()])
        );
    }
}

#[tokio::test]
async fn uncertain_commit_retires_host_and_does_not_use_unguarded_fallback() {
    let (_root, role, _shutdown, _) = fixture().await;
    let sink = Sink {
        fail_commit: true,
        ..Default::default()
    };
    assert!(
        role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
            .await
            .is_err()
    );
    assert!(role.host.is_retired());
    assert!(sink.outputs.lock().unwrap().is_empty());
}
#[tokio::test]
async fn missing_resource_amount_is_quarantined_without_invented_release() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink::default();
    let message = Message::new("1-0", "test:gpu0", "native:grp", "run1", "{}", "execute");
    assert!(role.handle(&message, "gpu0", &sink).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(sink.outputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_source_identifiers_are_rejected_before_claim_or_host_retirement() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let payload = message("unused", json!({"case_id":"run_1"}))
        .payload()
        .to_owned();
    let sink = Sink::default();
    for index in 0..4 {
        let mut fields = [
            "1-0".to_owned(),
            "test:gpu0".into(),
            "native:grp".into(),
            "run-one".into(),
        ];
        fields[index] = "x".repeat(4097);
        let invalid = Message::new(
            fields[0].clone(),
            fields[1].clone(),
            fields[2].clone(),
            fields[3].clone(),
            payload.clone(),
            "execute",
        );
        let error = role.handle(&invalid, "gpu0", &sink).await.unwrap_err();
        assert!(error.to_string().contains("1..4096"));
        assert!(
            !role.host.is_retired(),
            "deterministic malformed metadata retired an unclaimed host"
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        sink.begins.lock().unwrap().is_empty(),
        "malformed identifiers reached BEGIN"
    );
}

#[tokio::test]
async fn redis_engine_delivers_to_existing_results_and_release_streams() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let (_server, qm) = support::spawn_test_queue_manager("native-role-routing").await;
    let config:worker_runtime::config::RuntimeConfig=serde_json::from_value(json!({
        "stream_prefix":"test:","streams":["gpu0","release","results"],
        "roles":{"native":{"inputs":[{"stream":"gpu0","max_dequeue_items":1,"poll_interval_ms":1,"block_ms":1}],"outputs":["results","release"]}}
    })).unwrap();
    let transport = Arc::new(worker_runtime::transport::redis::RedisTransport::new(
        qm.clone(),
        "test:",
    ));
    let engine = worker_runtime::engine::EngineBuilder::new(&config, "native")
        .transport(transport)
        .role(Box::new(role))
        .consumer("native-test-boot-1")
        .build()
        .unwrap();
    engine.ensure_consumer_groups().await.unwrap();
    let original = message("unused", json!({"case_id":"run_1"}));
    qm.enqueue_to_stream(
        &scicomp_rq::StreamKey::new("test:gpu0"),
        original.run_id(),
        original.payload(),
        "execute",
    )
    .await
    .unwrap();
    let _ = engine.run_once().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut conn = qm.connection();
    for stream in ["test:results", "test:release"] {
        let count: usize = redis::cmd("XLEN")
            .arg(stream)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(
            count, 1,
            "outcome must reach the existing physical stream {stream}"
        );
    }
}

#[tokio::test]
async fn deadline_cancels_and_drains_before_owned_failure_release() {
    struct Slow {
        stopped: Arc<std::sync::atomic::AtomicBool>,
    }
    impl NativeHost for Slow {
        fn run(&mut self, _: &str, cancel: &Cancellation) -> Result<String> {
            while !cancel.is_cancelled() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.stopped.store(true, Ordering::Release);
            bail!("cancelled")
        }
    }
    let (_root, mut role, _shutdown, _) = fixture().await;
    let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = stopped.clone();
    role.host = HostClient::spawn(move || Ok(Slow { stopped: s }))
        .await
        .unwrap();
    let native_worker::deployment::RuntimeDeploymentConfig::Transolver(config) = &mut role.config
    else {
        panic!()
    };
    config.request_timeout_seconds = 1;
    let sink = Sink::default();
    role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
        .await
        .unwrap();
    assert!(stopped.load(Ordering::Acquire));
    assert!(role.host.is_retired());
    let outputs = sink.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 2);
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "failed");
    assert!(result["error"].as_str().unwrap().contains("deadline"));
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0].payload()).unwrap()["memory_mb"],
        12345
    );
}

#[tokio::test]
async fn uncertain_attempts_never_enter_generic_dlq_and_new_owner_recovers_after_fault_clears() {
    use worker_runtime::{
        config::RuntimeConfig, engine::EngineBuilder, transport::redis::RedisTransport,
    };
    for fault in ["begin", "complete"] {
        let (_redis, qm) = support::spawn_test_queue_manager("native-uncertain-engine").await;
        let (_root, role, _shutdown, calls) = fixture().await;
        let host = role.host.clone();
        let config: RuntimeConfig = serde_json::from_value(json!({"stream_prefix":"test:","max_retries":1,"shared_dlq_stream":"dlq","streams":["gpu0","results","release","dlq"],"roles":{"native":{"inputs":[{"stream":"gpu0","max_dequeue_items":1,"poll_interval_ms":1,"block_ms":1,"reclaim_idle_ms":30}],"outputs":["results","release"]}}})).unwrap();
        let engine = EngineBuilder::new(&config, "native")
            .role(Box::new(role))
            .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
            .consumer("original-native")
            .build()
            .unwrap();
        engine.ensure_consumer_groups().await.unwrap();
        let payload: Value = message("unused", json!({"case_id":"run_1"}))
            .parse_payload()
            .unwrap();
        let id = qm
            .enqueue_to_stream(
                &scicomp_rq::StreamKey::new("test:gpu0"),
                "same-run",
                &payload.to_string(),
                "execute",
            )
            .await
            .unwrap();
        let group = "gpu0:grp";
        let marker = format!(
            "native-attempt:9:test:gpu0:{}:{group}:{}:{id}",
            group.len(),
            id.len()
        );
        let poison_key = if fault == "begin" {
            marker.as_str()
        } else {
            "test:release"
        };
        let mut conn = qm.connection();
        let _: () = redis::cmd("SET")
            .arg(poison_key)
            .arg("wrong-type-sentinel")
            .query_async(&mut conn)
            .await
            .unwrap();
        let _ = engine.run_once().await.unwrap();
        assert!(
            host.is_retired(),
            "uncertain infrastructure outcome kept native admission open"
        );
        assert!(
            host.submit("{}".into()).await.is_err(),
            "retired host accepted more execution"
        );
        let dlq: u64 = redis::cmd("XLEN")
            .arg("test:dlq")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(
            dlq, 0,
            "uncertain native {fault} bypassed guarded completion through generic DLQ"
        );
        let pending: Vec<(String, String, u64, u64)> = redis::cmd("XPENDING")
            .arg("test:gpu0")
            .arg(group)
            .arg("-")
            .arg("+")
            .arg(10)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(
            pending.len(),
            1,
            "uncertain native {fault} ACKed its original source"
        );
        assert_eq!(pending[0].0, id);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            usize::from(fault == "complete")
        );
        let status: Option<String> = redis::cmd("HGET")
            .arg(format!("{}same-run", scicomp_rq::keys::RUN_HASH_PREFIX))
            .arg("status")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_ne!(
            status.as_deref(),
            Some("failed"),
            "uncertain native operation was marked terminally failed"
        );
        let _: usize = redis::cmd("DEL")
            .arg(poison_key)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(host.shutdown(std::time::Duration::from_secs(1)).await);
        let (_replacement_root, replacement, _replacement_shutdown, replacement_calls) =
            fixture().await;
        let replacement = EngineBuilder::new(&config, "native")
            .role(Box::new(replacement))
            .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
            .consumer("replacement-native")
            .build()
            .unwrap();
        replacement.ensure_consumer_groups().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        let _ = replacement.run_once().await.unwrap();
        assert_eq!(
            replacement_calls.load(Ordering::SeqCst),
            1,
            "replacement did not reclaim the original request"
        );
        let pending: Vec<(String, String, u64, u64)> = redis::cmd("XPENDING")
            .arg("test:gpu0")
            .arg(group)
            .arg("-")
            .arg("+")
            .arg(10)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(pending.is_empty());
        for stream in ["test:release", "test:results"] {
            let count: u64 = redis::cmd("XLEN")
                .arg(stream)
                .query_async(&mut conn)
                .await
                .unwrap();
            assert_eq!(
                count, 1,
                "recovered native request produced duplicate/missing output on{stream}"
            );
        }
        let events: Vec<(String, Vec<String>)> = redis::cmd("XRANGE")
            .arg("test:release")
            .arg("-")
            .arg("+")
            .query_async(&mut conn)
            .await
            .unwrap();
        let fields: std::collections::BTreeMap<_, _> = events[0]
            .1
            .chunks_exact(2)
            .map(|pair| (pair[0].as_str(), pair[1].as_str()))
            .collect();
        let release: Value = serde_json::from_str(fields["payload"]).unwrap();
        assert_eq!(release["resource_id"], 0);
        assert_eq!(release["memory_mb"], 12345);
        let committed: String = redis::cmd("HGET")
            .arg(&marker)
            .arg("state")
            .query_async(&mut conn)
            .await
            .unwrap();
        let owner: String = redis::cmd("HGET")
            .arg(&marker)
            .arg("consumer")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(committed, "committed");
        assert_eq!(owner, "replacement-native");
    }
}

#[tokio::test]
async fn partial_case_outputs_cannot_be_published_as_full_case_success() {
    let (_root, mut role, _shutdown, _) = fixture().await;
    let native_worker::deployment::RuntimeDeploymentConfig::Transolver(config) = &mut role.config
    else {
        panic!()
    };
    config.case.point_count = 3;
    let sink = Sink::default();
    role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
        .await
        .unwrap();
    let outputs = sink.outputs.lock().unwrap();
    let result: Value = serde_json::from_str(outputs[1].payload()).unwrap();
    assert_eq!(result["status"], "failed");
    assert!(result["error"].as_str().unwrap().contains("full case"));
    assert!(result.get("payload").is_none());
}

#[tokio::test]
async fn malformed_dispatch_still_uses_generic_quarantine_without_native_execution() {
    use worker_runtime::{
        config::RuntimeConfig, engine::EngineBuilder, transport::redis::RedisTransport,
    };
    let oversized = "x".repeat(4097);
    let valid_reservation = message("unused", json!({"case_id":"run_1"}))
        .payload()
        .to_owned();
    for (run_id, payload) in [
        ("malformed-run", "{}"),
        (oversized.as_str(), valid_reservation.as_str()),
    ] {
        let (_redis, qm) = support::spawn_test_queue_manager("native-malformed-engine").await;
        let (_root, role, _shutdown, calls) = fixture().await;
        let host = role.host.clone();
        let config: RuntimeConfig = serde_json::from_value(json!({"stream_prefix":"test:","max_retries":1,"shared_dlq_stream":"dlq","streams":["gpu0","results","release","dlq"],"roles":{"native":{"inputs":[{"stream":"gpu0","max_dequeue_items":1,"poll_interval_ms":1,"block_ms":1,"reclaim_idle_ms":30}],"outputs":["results","release"]}}})).unwrap();
        let engine = EngineBuilder::new(&config, "native")
            .role(Box::new(role))
            .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
            .consumer("malformed-native")
            .build()
            .unwrap();
        engine.ensure_consumer_groups().await.unwrap();
        qm.enqueue_to_stream(
            &scicomp_rq::StreamKey::new("test:gpu0"),
            run_id,
            payload,
            "execute",
        )
        .await
        .unwrap();
        let _ = engine.run_once().await.unwrap();
        let mut conn = qm.connection();
        let dlq: u64 = redis::cmd("XLEN")
            .arg("test:dlq")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(
            dlq, 1,
            "malformed dispatch lost its existing quarantine policy"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!host.is_retired());
        for stream in ["test:release", "test:results"] {
            let count: u64 = redis::cmd("XLEN")
                .arg(stream)
                .query_async(&mut conn)
                .await
                .unwrap();
            assert_eq!(
                count, 0,
                "malformed dispatch invented a native completion/release"
            );
        }
    }
}

#[tokio::test]
async fn transient_commit_error_retries_identical_outcome_without_reexecuting() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink {
        failures_left: AtomicUsize::new(1),
        ..Default::default()
    };
    role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
        .await
        .unwrap();
    let commits = sink.commits.lock().unwrap();
    assert_eq!(commits.len(), 2);
    assert_eq!(commits[0], commits[1]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(sink.outputs.lock().unwrap().len(), 2);
    assert!(!role.host.is_retired());
}

#[tokio::test]
async fn lost_begin_reply_reuses_claim_token_before_native_execution() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink {
        begin_failures_left: AtomicUsize::new(1),
        ..Default::default()
    };
    role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
        .await
        .unwrap();
    let begins = sink.begins.lock().unwrap();
    assert_eq!(begins.len(), 2);
    assert_eq!(begins[0], begins[1]);
    assert_eq!(sink.commits.lock().unwrap()[0].0, begins[0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!role.host.is_retired());
}

#[tokio::test]
async fn uncertain_begin_retires_before_any_native_work_or_release() {
    let (_root, role, _shutdown, calls) = fixture().await;
    let sink = Sink {
        fail_begin: true,
        ..Default::default()
    };
    assert!(
        role.handle(&message("1-0", json!({"case_id":"run_1"})), "gpu0", &sink)
            .await
            .is_err()
    );
    assert!(role.host.is_retired());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(sink.outputs.lock().unwrap().is_empty());
}
