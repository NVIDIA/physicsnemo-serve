/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::{PrepareRoleConfig, PythonRuntimeEnvConfig, parse_role_config};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

#[derive(Debug, Clone, Deserialize)]
struct PrepareEnvelope {
    workflow_id: String,
    stage_context: StageContext,
}

use crate::roles::stage::{StageContext, StageDescriptor};

#[derive(Debug, Default, Deserialize)]
struct PrepareHookOutput {
    #[serde(default)]
    operation: Option<String>,
    #[serde(default)]
    parameters: Option<JsonValue>,
    #[serde(default)]
    request: Option<JsonValue>,
    #[serde(default)]
    resource_profile: Option<JsonValue>,
    #[serde(default)]
    batch_profile: Option<JsonValue>,
    #[serde(default)]
    prefetch_plan: Option<JsonValue>,
    #[serde(default)]
    fanout_profile: Option<JsonValue>,
    #[serde(default)]
    fanout_items: Option<JsonValue>,
    #[serde(default)]
    next_stage_id: Option<String>,
}

pub struct PrepareRole {
    python_executable: String,
    runner_path: PathBuf,
    hook_timeout: Duration,
    input_streams: Vec<String>,
    python_runtime_envs: std::collections::BTreeMap<String, PythonRuntimeEnvConfig>,
}

impl PrepareRole {
    pub fn from_env(env: &RoleEnv) -> Result<Self> {
        let cfg: PrepareRoleConfig = parse_role_config(env.role_config.as_ref())?;
        if cfg.python_executable.trim().is_empty() {
            return Err(anyhow!(
                "prepare role config python_executable must be non-empty"
            ));
        }
        if cfg.runner_path.trim().is_empty() {
            return Err(anyhow!("prepare role config runner_path must be non-empty"));
        }
        if cfg.hook_timeout_secs == 0 {
            return Err(anyhow!(
                "prepare role config hook_timeout_secs must be greater than zero"
            ));
        }

        Ok(Self {
            python_executable: cfg.python_executable,
            runner_path: PathBuf::from(cfg.runner_path),
            hook_timeout: Duration::from_secs(cfg.hook_timeout_secs),
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
            python_runtime_envs: env.python_runtime_envs.clone(),
        })
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "prepare: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn run_prepare_hook(&self, payload: &JsonValue) -> Result<PrepareHookOutput> {
        let input = serde_json::to_vec(payload).context("prepare: failed to serialize payload")?;
        let runtime_env = self.runtime_env_for_phase(payload, "prepare", "prepare_executor_class");
        let python_executable = runtime_env
            .map(|env| env.python_executable.as_str())
            .unwrap_or(self.python_executable.as_str());
        let mut command = Command::new(python_executable);
        command
            .arg(&self.runner_path)
            .arg("--phase")
            .arg("prepare")
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(runtime_env) = runtime_env {
            command.envs(&runtime_env.env);
        }
        let mut child = command.spawn().with_context(|| {
            format!(
                "prepare: failed to spawn hook runner '{}' with executable '{}'",
                self.runner_path.display(),
                python_executable
            )
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&input)
                .await
                .context("prepare: failed to write payload to hook runner stdin")?;
        }

        let output = tokio::time::timeout(self.hook_timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                anyhow!(
                    "prepare: plugin hook timed out after {}s",
                    self.hook_timeout.as_secs()
                )
            })?
            .context("prepare: failed waiting for hook runner process")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let details = if stderr.is_empty() {
                format!("exit status {}", output.status)
            } else {
                stderr
            };
            return Err(anyhow!("prepare: plugin hook failed: {details}"));
        }

        let stdout = String::from_utf8(output.stdout)
            .context("prepare: hook runner stdout is not valid UTF-8")?;
        if stdout.trim().is_empty() {
            return Ok(PrepareHookOutput::default());
        }

        serde_json::from_str(&stdout).context("prepare: hook output must be valid JSON")
    }

    fn runtime_env_for_phase<'a>(
        &'a self,
        payload: &JsonValue,
        phase: &str,
        selector_field: &str,
    ) -> Option<&'a PythonRuntimeEnvConfig> {
        let executor_class = payload
            .get("runtime")
            .and_then(JsonValue::as_object)
            .and_then(|runtime| {
                runtime
                    .get(selector_field)
                    .and_then(JsonValue::as_str)
                    .or_else(|| runtime.get("executor_class").and_then(JsonValue::as_str))
            })
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let executor_class = executor_class?;
        self.python_runtime_envs.get(executor_class).or_else(|| {
            tracing::debug!(
                phase,
                executor_class,
                "prepare: runtime env not found for requested executor_class, falling back to default python_executable"
            );
            None
        })
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let (typed, mut payload) = decode_prepare_payload(msg.payload())?;
        let default_next_stage = typed.stage_context.next_stage("prepare")?;
        let hook_output = self.run_prepare_hook(&payload).await?;

        let next_stage = apply_prepare_output(
            &mut payload,
            hook_output,
            &typed.stage_context,
            &default_next_stage,
        )?;
        let updated_payload =
            serde_json::to_string(&payload).context("prepare: failed to encode payload")?;

        sink.handoff(msg, &next_stage.queue, &updated_payload, &next_stage.phase)
            .await
            .with_context(|| {
                format!(
                    "prepare: failed to hand off workflow '{}' to next stage '{}'",
                    typed.workflow_id, next_stage.phase
                )
            })?;
        Ok(())
    }
}

fn decode_prepare_payload(raw: &str) -> Result<(PrepareEnvelope, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("prepare: empty payload"));
    }

    let value: JsonValue =
        serde_json::from_str(raw).context("prepare: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("prepare: payload must be a JSON object"));
    }

    let typed: PrepareEnvelope =
        serde_json::from_value(value.clone()).context("prepare: invalid payload schema")?;
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "prepare: workflow_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "prepare" {
        return Err(anyhow!(
            "prepare: payload current_phase must be 'prepare', got '{}'",
            typed.stage_context.current_phase
        ));
    }

    Ok((typed, value))
}

fn apply_prepare_output(
    payload: &mut JsonValue,
    hook_output: PrepareHookOutput,
    stage_context: &StageContext,
    default_next_stage: &StageDescriptor,
) -> Result<StageDescriptor> {
    let target_stage = if let Some(next_stage_id) = hook_output.next_stage_id.as_deref() {
        stage_context
            .pipeline
            .iter()
            .find(|stage| stage.id == next_stage_id)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "prepare: requested next_stage_id '{}' not found in pipeline",
                    next_stage_id
                )
            })?
    } else {
        default_next_stage.clone()
    };

    let map = payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("prepare: payload must remain a JSON object"))?;

    if let Some(operation) = hook_output.operation {
        map.insert("operation".to_string(), JsonValue::String(operation));
    }
    if let Some(parameters) = hook_output.parameters {
        map.insert("parameters".to_string(), parameters);
    }
    if let Some(request) = hook_output.request {
        map.insert("request".to_string(), request);
    }
    if let Some(resource_profile) = hook_output.resource_profile {
        map.insert("resource_profile".to_string(), resource_profile);
    }
    if let Some(batch_profile) = hook_output.batch_profile {
        map.insert("batch_profile".to_string(), batch_profile);
    }
    if let Some(prefetch_plan) = hook_output.prefetch_plan {
        map.insert("prefetch_plan".to_string(), prefetch_plan);
    }
    if let Some(fanout_profile) = hook_output.fanout_profile {
        map.insert("fanout_profile".to_string(), fanout_profile);
    }
    if let Some(fanout_items) = hook_output.fanout_items {
        map.insert("fanout_items".to_string(), fanout_items);
    }

    let stage_context = map
        .get_mut("stage_context")
        .and_then(JsonValue::as_object_mut)
        .ok_or_else(|| anyhow!("prepare: payload missing object stage_context"))?;
    stage_context.insert(
        "current_stage_id".to_string(),
        JsonValue::String(target_stage.id.clone()),
    );
    stage_context.insert(
        "current_phase".to_string(),
        JsonValue::String(target_stage.phase.clone()),
    );

    Ok(target_stage)
}

impl WorkerRole for PrepareRole {
    fn name(&self) -> &'static str {
        "prepare"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            self.process_message(msg, sink).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::Duration;

    use anyhow::Result;
    use serde_json::{Value, json};

    use crate::config::InputStreamSpec;
    use crate::test_env;
    use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

    use super::{PrepareHookOutput, PrepareRole, apply_prepare_output};
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[derive(Debug, Clone)]
    struct HandoffRecord {
        dest_stream: String,
        payload: String,
        stage: String,
    }

    struct RecordingSink {
        writes: Mutex<Vec<HandoffRecord>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                writes: Mutex::new(Vec::new()),
            }
        }

        fn writes(&self) -> Vec<HandoffRecord> {
            self.writes.lock().expect("recording lock poisoned").clone()
        }
    }

    struct EnvRestore {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvRestore {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(key).ok();
            test_env::set_env_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            test_env::set_env_var(self.key, self.previous.as_deref());
        }
    }

    impl MessageSink for RecordingSink {
        fn enqueue<'a>(
            &'a self,
            _stream: &'a str,
            _run_id: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("1-0".to_string()) })
        }

        fn ack_message<'a>(&'a self, _msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.writes
                    .lock()
                    .expect("recording lock poisoned")
                    .push(HandoffRecord {
                        dest_stream: dest_stream.to_string(),
                        payload: payload.to_string(),
                        stage: stage.to_string(),
                    });
                Ok("1-0".to_string())
            })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    fn test_python_executable() -> String {
        if let Ok(explicit) = std::env::var("PHYSICSNEMO_SERVE_TEST_PYTHON")
            && !explicit.trim().is_empty()
        {
            return explicit;
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        #[cfg(windows)]
        let candidates = [manifest_dir.join("../../.venv/Scripts/python.exe")];

        #[cfg(not(windows))]
        let candidates = [manifest_dir.join("../../.venv/bin/python")];

        for candidate in candidates {
            if candidate.is_file() {
                return candidate.display().to_string();
            }
        }

        "python3".to_string()
    }

    fn prepare_env(runner_path: &str) -> RoleEnv {
        RoleEnv {
            role_name: "prepare".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "prepare".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec![],
            role_config: Some(json!({
                "python_executable": test_python_executable(),
                "runner_path": runner_path,
                "hook_timeout_secs": 5
            })),
            python_runtime_envs: Default::default(),
        }
    }

    fn runner_path() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/plugin_hook_runner.py")
            .display()
            .to_string()
    }

    fn write_plugin(tmp: &tempfile::TempDir) -> PathBuf {
        let plugin_root = tmp.path().join("plugins").join("demo-plugin");
        std::fs::create_dir_all(&plugin_root).expect("plugin dir should be created");
        std::fs::write(
            plugin_root.join("plugin.yaml"),
            r#"
metadata:
  id: demo-plugin
  display_name: Demo Plugin
  version: 1.0.0
  description: Example plugin
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: both
    allowed: [both]
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: schedule
    - id: schedule
      phase: schedule
      handler: schedule
      queue: schedule
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.gpu.physicsnemo
resources:
  defaults:
    gpus_required: 1
    memory_mb: 4096
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: pressure_field
    media_type: application/x-npz
  retention_hours: 24
"#,
        )
        .expect("manifest should be written");
        std::fs::write(
            plugin_root.join("plugin.py"),
            r#"
def prepare(ctx):
    return {
        "parameters": {
            "batch_size": ctx["parameters"]["batch_size"],
            "normalized": True,
        },
        "resource_profile": {
            "executor_class": "python.gpu.physicsnemo",
            "gpus_required": 1,
            "memory_mb": 8192,
        },
        "batch_profile": {
            "enabled": True,
            "batch_key": "physicsnemo-demo",
            "max_batch_size": 4,
            "max_wait_ms": 25,
            "shared_memory_mb": 4096,
            "incremental_memory_mb": 512,
        },
        "prefetch_plan": [
            {
                "source_uri": "s3://bucket/input.bin",
                "target_artifact_name": "prepared-input",
            }
        ],
        "fanout_profile": {
            "item_count": 2,
            "aggregation_mode": "all_members",
        },
        "fanout_items": [
            {
                "item_index": 0,
                "parameters": {
                    "batch_size": ctx["parameters"]["batch_size"],
                    "member_seed": 1000,
                },
            },
            {
                "item_index": 1,
                "parameters": {
                    "batch_size": ctx["parameters"]["batch_size"],
                    "member_seed": 1001,
                },
            },
        ],
    }
"#,
        )
        .expect("plugin entrypoint should be written");
        plugin_root
    }

    #[cfg(unix)]
    fn write_fake_python_runtime(tmp: &tempfile::TempDir, name: &str, stdout_json: &str) -> String {
        let script_path = tmp.path().join(name);
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{}'\n",
                stdout_json.replace('\'', "'\"'\"'")
            ),
        )
        .expect("fake runtime script should be written");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata should load")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("script should be executable");
        script_path.display().to_string()
    }

    #[cfg(unix)]
    fn write_sleeping_runtime_script(
        tmp: &tempfile::TempDir,
        name: &str,
        pid_file: &std::path::Path,
    ) -> String {
        let script_path = tmp.path().join(name);
        let pid_file_escaped = pid_file.display().to_string().replace('\'', "'\"'\"'");
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\necho \"$$\" > '{}'\nsleep 30\nprintf '{{}}\\n'\n",
                pid_file_escaped
            ),
        )
        .expect("sleeping runtime script should be written");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata should load")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("script should be executable");
        script_path.display().to_string()
    }

    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(unix)]
    fn kill_process(pid: u32) {
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.to_string())
            .status();
    }

    #[tokio::test]
    async fn prepare_role_invokes_plugin_hook_and_handoffs_to_next_stage() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin(&tmp);
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let role =
            PrepareRole::from_env(&prepare_env(&runner_path())).expect("prepare role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:prepare",
            "prepare:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "parameters": { "batch_size": 128000 },
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "resource_profile": {
                    "executor_class": "python.gpu.physicsnemo",
                    "gpus_required": 1,
                    "memory_mb": 4096
                },
                "prefetch_plan": [],
                "stage_context": {
                    "current_stage_id": "prepare",
                    "current_phase": "prepare",
                    "pipeline": [
                        {
                            "id": "prepare",
                            "phase": "prepare",
                            "queue": "prepare",
                            "next": "schedule"
                        },
                        {
                            "id": "schedule",
                            "phase": "schedule",
                            "queue": "schedule",
                            "next": "execute"
                        },
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "results"
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": null
                        }
                    ]
                },
                "result": null,
                "runtime": {
                    "kind": "python",
                    "entrypoint": "plugin.py",
                    "executor_class": "python.gpu.physicsnemo"
                }
            })
            .to_string(),
            "prepare",
        );

        role.handle(&msg, "prepare", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "schedule");
        assert_eq!(writes[0].stage, "schedule");

        let forwarded: Value =
            serde_json::from_str(&writes[0].payload).expect("payload should remain valid JSON");
        assert_eq!(forwarded["parameters"]["normalized"], Value::Bool(true));
        assert_eq!(
            forwarded["resource_profile"]["memory_mb"],
            Value::from(8192)
        );
        assert_eq!(forwarded["batch_profile"]["batch_key"], "physicsnemo-demo");
        assert_eq!(forwarded["batch_profile"]["max_batch_size"], 4);
        assert_eq!(
            forwarded["prefetch_plan"][0]["target_artifact_name"],
            "prepared-input"
        );
        assert_eq!(forwarded["fanout_profile"]["item_count"], 2);
        assert_eq!(forwarded["fanout_items"][0]["item_index"], 0);
        assert_eq!(
            forwarded["fanout_items"][1]["parameters"]["member_seed"],
            Value::from(1001)
        );
        assert_eq!(forwarded["stage_context"]["current_stage_id"], "schedule");
        assert_eq!(forwarded["stage_context"]["current_phase"], "schedule");
    }

    #[test]
    fn prepare_output_can_override_next_stage_id() {
        let mut payload = json!({
            "stage_context": {
                "current_stage_id": "prepare",
                "current_phase": "prepare",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "schedule_materialize"},
                    {"id": "schedule_materialize", "phase": "schedule", "queue": "schedule", "next": "materialize_perturbations"},
                    {"id": "materialize_perturbations", "phase": "execute", "queue": "execute.earth2-gpu", "next": "fanout"},
                    {"id": "fanout", "phase": "fanout", "queue": "fanout", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        });
        let stage_context: crate::roles::stage::StageContext =
            serde_json::from_value(payload["stage_context"].clone()).unwrap();
        let default_next = stage_context.next_stage("prepare").unwrap();
        let output = PrepareHookOutput {
            operation: Some("run".to_string()),
            next_stage_id: Some("fanout".to_string()),
            fanout_profile: Some(json!({"item_count": 1, "max_in_flight": 1})),
            fanout_items: Some(json!([{"item_index": 0, "parameters": {"value": 1}}])),
            ..Default::default()
        };

        let next_stage =
            apply_prepare_output(&mut payload, output, &stage_context, &default_next).unwrap();

        assert_eq!(next_stage.id, "fanout");
        assert_eq!(next_stage.phase, "fanout");
        assert_eq!(next_stage.queue, "fanout");
        assert_eq!(payload["operation"], "run");
        assert_eq!(payload["fanout_profile"]["item_count"], 1);
        assert_eq!(payload["stage_context"]["current_stage_id"], "fanout");
        assert_eq!(payload["stage_context"]["current_phase"], "fanout");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn prepare_role_uses_runtime_env_python_for_prepare_phase() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let plugin_root = write_plugin(&tmp);
        let plugin_parent = plugin_root
            .parent()
            .expect("plugin root should have a parent")
            .to_string_lossy()
            .to_string();
        let manifest = std::fs::read_to_string(plugin_root.join("plugin.yaml"))
            .expect("manifest should be readable")
            .replace(
                "executor_class: python.gpu.physicsnemo\nresources:",
                "executor_class: python.gpu.physicsnemo\n  prepare_executor_class: python.prepare.custom\nresources:",
            );
        std::fs::write(plugin_root.join("plugin.yaml"), manifest).expect("manifest should update");
        let _plugin_dir = EnvRestore::set("PLUGIN_DIR", Some(plugin_parent.as_str()));

        let fake_runtime = write_fake_python_runtime(
            &tmp,
            "fake-prepare-python.sh",
            r#"{"parameters":{"from_runtime_env":true},"resource_profile":{"executor_class":"python.custom.runtime","gpus_required":1,"memory_mb":2048},"prefetch_plan":[]}"#,
        );

        let mut env = prepare_env(&runner_path());
        env.python_runtime_envs.insert(
            "python.prepare.custom".to_string(),
            crate::config::PythonRuntimeEnvConfig {
                python_executable: fake_runtime,
                env: Default::default(),
            },
        );

        let role = PrepareRole::from_env(&env).expect("prepare role should build");
        let sink = RecordingSink::new();
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:prepare",
            "prepare:grp",
            "run-plugin",
            json!({
                "run_id": "run-plugin",
                "workflow_id": "demo-plugin",
                "operation": "both",
                "parameters": { "batch_size": 128000 },
                "request": {
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "resource_profile": {
                    "executor_class": "python.gpu.physicsnemo",
                    "gpus_required": 1,
                    "memory_mb": 4096
                },
                "prefetch_plan": [],
                "stage_context": {
                    "current_stage_id": "prepare",
                    "current_phase": "prepare",
                    "pipeline": [
                        {
                            "id": "prepare",
                            "phase": "prepare",
                            "queue": "prepare",
                            "next": "schedule"
                        },
                        {
                            "id": "schedule",
                            "phase": "schedule",
                            "queue": "schedule",
                            "next": "execute"
                        },
                        {
                            "id": "execute",
                            "phase": "execute",
                            "queue": "execute",
                            "next": "results"
                        },
                        {
                            "id": "results",
                            "phase": "results",
                            "queue": "results",
                            "next": null
                        }
                    ]
                },
                "result": null,
                "runtime": {
                    "kind": "python",
                    "entrypoint": "plugin.py",
                    "executor_class": "python.gpu.physicsnemo",
                    "prepare_executor_class": "python.prepare.custom"
                }
            })
            .to_string(),
            "prepare",
        );

        role.handle(&msg, "prepare", &sink).await.unwrap();

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: Value =
            serde_json::from_str(&writes[0].payload).expect("payload should remain valid JSON");
        assert_eq!(
            forwarded["parameters"]["from_runtime_env"],
            Value::Bool(true)
        );
        assert_eq!(
            forwarded["resource_profile"]["executor_class"],
            Value::from("python.custom.runtime")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn prepare_timeout_terminates_hook_process() {
        let _guard = test_env::env_lock().lock().await;
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let pid_file = tmp.path().join("prepare-timeout.pid");
        let sleeping_runtime =
            write_sleeping_runtime_script(&tmp, "sleeping-prepare.sh", pid_file.as_path());
        let role = PrepareRole {
            python_executable: sleeping_runtime,
            runner_path: PathBuf::from("unused-runner.py"),
            hook_timeout: Duration::from_secs(2),
            input_streams: vec!["prepare".to_string()],
            python_runtime_envs: Default::default(),
        };
        let payload = json!({
            "runtime": {
                "executor_class": "python.gpu.physicsnemo"
            }
        });

        let result = role.run_prepare_hook(&payload).await;
        assert!(result.is_err(), "timeout should return an error");
        assert!(
            result
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("timed out")),
            "expected timeout error, got: {result:?}"
        );

        let mut pid_raw = None;
        for _ in 0..40 {
            if let Ok(contents) = std::fs::read_to_string(&pid_file) {
                pid_raw = Some(contents);
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let pid_raw = pid_raw.expect("pid file should be written");
        let pid: u32 = pid_raw.trim().parse().expect("pid should parse as u32");
        let mut alive = process_is_alive(pid);
        for _ in 0..20 {
            if !alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            alive = process_is_alive(pid);
        }
        if alive {
            kill_process(pid);
        }
        assert!(
            !alive,
            "prepare hook subprocess should be terminated after timeout"
        );
    }
}
