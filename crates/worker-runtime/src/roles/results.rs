/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use scicomp_rq::{QueueManager, hash_ops};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::config::{ResultsRoleConfig, parse_role_config};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

const RUN_KEY_PREFIX: &str = "run:";
const RESULT_KEY_PREFIX: &str = "result:";
const REDIS_OPERATION_TIMEOUT: Duration = Duration::from_secs(10);

async fn with_redis_timeout<T, F>(operation: &'static str, timeout: Duration, fut: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_) => Err(anyhow!(
            "results: redis operation '{operation}' timed out after {}s",
            timeout.as_secs()
        )),
    }
}

/// Result record persisted by the results role.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedResult {
    pub run_id: String,
    pub status: String,
    pub stage: String,
    pub completed_at: Option<String>,
    pub workflow: Option<String>,
    pub gpu_stream: Option<String>,
    pub output_path: Option<String>,
    pub output_archive: Option<String>,
    pub error: Option<String>,
    pub execution_time_seconds: Option<f64>,
    pub result_payload: Value,
    pub result_ttl_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchHashFields {
    batch_id: Option<String>,
    batch_size: Option<String>,
    batch_flush_reason: Option<String>,
    batch_first_seen_ms: Option<String>,
    batch_formed_at_ms: Option<String>,
    batch_waited_ms: Option<String>,
}

impl BatchHashFields {
    fn from_result_payload(payload: &Value) -> Self {
        let batch_info = payload
            .get("batch_info")
            .or_else(|| {
                payload
                    .get("execution")
                    .and_then(Value::as_object)
                    .and_then(|execution| execution.get("batch_info"))
            })
            .or_else(|| {
                payload
                    .get("payload")
                    .and_then(Value::as_object)
                    .and_then(|inner| inner.get("batch_info"))
            })
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        Self {
            batch_id: batch_info
                .get("batch_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            batch_size: value_to_hash_string(batch_info.get("batch_size")),
            batch_flush_reason: batch_info
                .get("flush_reason")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            batch_first_seen_ms: value_to_hash_string(batch_info.get("first_seen_ms")),
            batch_formed_at_ms: value_to_hash_string(batch_info.get("formed_at_ms")),
            batch_waited_ms: value_to_hash_string(batch_info.get("waited_ms")),
        }
    }

    fn to_hash_fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = Vec::new();
        if let Some(value) = &self.batch_id {
            fields.push(("batch_id", value.clone()));
        }
        if let Some(value) = &self.batch_size {
            fields.push(("batch_size", value.clone()));
        }
        if let Some(value) = &self.batch_flush_reason {
            fields.push(("batch_flush_reason", value.clone()));
        }
        if let Some(value) = &self.batch_first_seen_ms {
            fields.push(("batch_first_seen_ms", value.clone()));
        }
        if let Some(value) = &self.batch_formed_at_ms {
            fields.push(("batch_formed_at_ms", value.clone()));
        }
        if let Some(value) = &self.batch_waited_ms {
            fields.push(("batch_waited_ms", value.clone()));
        }
        fields
    }
}

fn value_to_hash_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(v)) => Some(v.clone()),
        Some(Value::Number(v)) => Some(v.to_string()),
        _ => None,
    }
}

/// Persistence boundary for the results role.
pub trait ResultsPersistence: Send + Sync + 'static {
    fn persist_result<'a>(&'a self, result: PersistedResult) -> BoxFuture<'a, Result<()>>;
}

/// No-op persistence implementation for tests that don't assert persistence side effects.
pub struct NoopResultsPersistence;

impl NoopResultsPersistence {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoopResultsPersistence {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultsPersistence for NoopResultsPersistence {
    fn persist_result<'a>(&'a self, _result: PersistedResult) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// Redis-backed results persistence (`run:{id}` hash + `result:{id}` JSON payload with TTL).
pub struct RedisResultsPersistence {
    qm: QueueManager,
}

impl RedisResultsPersistence {
    pub fn new(qm: QueueManager) -> Self {
        Self { qm }
    }

    async fn persist_hash_fields(&self, run_key: &str, result: &PersistedResult) -> Result<()> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("results: system clock before unix epoch")?
            .as_secs()
            .to_string();

        let mut fields: Vec<(&str, String)> = vec![
            ("status", result.status.clone()),
            ("stage", result.stage.clone()),
            ("updated_at", now_secs.clone()),
            ("results_completed_at", now_secs),
        ];

        if let Some(value) = &result.completed_at {
            fields.push(("inference_completed_at", value.clone()));
        }
        if let Some(value) = &result.workflow {
            fields.push(("workflow", value.clone()));
        }
        if let Some(value) = &result.gpu_stream {
            fields.push(("gpu_stream", value.clone()));
        }
        if let Some(value) = &result.output_path {
            fields.push(("output_path", value.clone()));
        }
        if let Some(value) = &result.output_archive {
            fields.push(("output_archive", value.clone()));
        }
        if let Some(value) = result.execution_time_seconds {
            fields.push(("execution_time_seconds", value.to_string()));
        }
        if let Some(value) = &result.error {
            fields.push(("error", value.clone()));
        }
        fields
            .extend(BatchHashFields::from_result_payload(&result.result_payload).to_hash_fields());

        let mut conn = self.qm.connection();
        let mut hset = redis::cmd("HSET");
        hset.arg(run_key);
        for (field, value) in &fields {
            hset.arg(field).arg(value);
        }
        let _: usize = with_redis_timeout("HSET run hash fields", REDIS_OPERATION_TIMEOUT, async {
            hset.query_async(&mut conn)
                .await
                .context("results: failed to persist run hash fields")
        })
        .await?;

        if result.error.is_none() {
            let _: i64 =
                with_redis_timeout("HDEL error hash field", REDIS_OPERATION_TIMEOUT, async {
                    hash_ops::hdel(&mut conn, run_key, "error")
                        .await
                        .context("results: failed to clear hash field 'error'")
                })
                .await?;
        }

        Ok(())
    }
}

impl ResultsPersistence for RedisResultsPersistence {
    fn persist_result<'a>(&'a self, result: PersistedResult) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let run_key = format!("{RUN_KEY_PREFIX}{}", result.run_id);
            let result_key = format!("{RESULT_KEY_PREFIX}{}", result.run_id);

            self.persist_hash_fields(&run_key, &result).await?;

            let payload_json = serde_json::to_string(&result.result_payload)
                .context("results: failed to serialize result payload")?;
            let mut conn = self.qm.connection();
            let _: () =
                with_redis_timeout("SETEX result payload", REDIS_OPERATION_TIMEOUT, async {
                    redis::cmd("SETEX")
                        .arg(&result_key)
                        .arg(result.result_ttl_seconds)
                        .arg(payload_json)
                        .query_async(&mut conn)
                        .await
                        .with_context(|| format!("results: failed writing key '{result_key}'"))
                })
                .await?;

            Ok(())
        })
    }
}

/// Results role: terminal consumer that validates and persists completed run metadata.
pub struct ResultsRole {
    persistence: Arc<dyn ResultsPersistence>,
    result_ttl_seconds: u64,
    input_streams: Vec<String>,
}

impl ResultsRole {
    pub fn from_env(env: &RoleEnv, persistence: Arc<dyn ResultsPersistence>) -> Result<Self> {
        let cfg: ResultsRoleConfig = parse_role_config(env.role_config.as_ref())?;
        Ok(Self {
            persistence,
            result_ttl_seconds: cfg.result_ttl_seconds,
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
        })
    }

    #[cfg(test)]
    fn new_for_test(persistence: Arc<dyn ResultsPersistence>, result_ttl_seconds: u64) -> Self {
        Self {
            persistence,
            result_ttl_seconds,
            input_streams: vec!["results".to_string()],
        }
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "results: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }
}

#[derive(Debug, Deserialize)]
struct ResultsEnvelope {
    run_id: Option<String>,
    status: Option<String>,
    completed_at: Option<String>,
    workflow: Option<String>,
    gpu_stream: Option<String>,
    output_archive: Option<String>,
    operation: Option<String>,
    request: Option<Value>,
    execution: Option<Value>,
    payload: Option<Value>,
    output_path: Option<String>,
    error: Option<String>,
    execution_time_seconds: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct NestedResultPayload {
    status: Option<String>,
    completed_at: Option<String>,
    workflow: Option<String>,
    gpu_stream: Option<String>,
    output_archive: Option<String>,
    output_path: Option<String>,
    error: Option<String>,
    execution_time_seconds: Option<f64>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

/// Merged view of top-level envelope fields and nested payload fields.
/// Top-level takes precedence over nested.
struct MergedFields {
    completed_at: Option<String>,
    workflow: Option<String>,
    gpu_stream: Option<String>,
    output_archive: Option<String>,
    output_path: Option<String>,
    error: Option<String>,
    execution_time_seconds: Option<f64>,
    extra_payload_fields: Map<String, Value>,
}

impl MergedFields {
    fn from_envelope_and_nested(envelope: &ResultsEnvelope, nested: &NestedResultPayload) -> Self {
        Self {
            completed_at: envelope
                .completed_at
                .clone()
                .or(nested.completed_at.clone()),
            workflow: envelope.workflow.clone().or(nested.workflow.clone()),
            gpu_stream: envelope.gpu_stream.clone().or(nested.gpu_stream.clone()),
            output_archive: envelope
                .output_archive
                .clone()
                .or(nested.output_archive.clone()),
            output_path: envelope.output_path.clone().or(nested.output_path.clone()),
            error: envelope.error.clone().or(nested.error.clone()),
            execution_time_seconds: envelope
                .execution_time_seconds
                .or(nested.execution_time_seconds),
            extra_payload_fields: nested.extra.clone(),
        }
    }

    fn build_execution_and_payload(
        &self,
        run_id: &str,
        status: &str,
    ) -> (Map<String, Value>, Value) {
        let mut execution = Map::new();
        execution.insert("run_id".into(), Value::String(run_id.to_owned()));
        execution.insert("status".into(), Value::String(status.to_owned()));

        let optional_string_fields: &[(&str, &Option<String>)] = &[
            ("workflow", &self.workflow),
            ("gpu_stream", &self.gpu_stream),
            ("completed_at", &self.completed_at),
            ("output_path", &self.output_path),
            ("output_archive", &self.output_archive),
            ("error", &self.error),
        ];
        for &(key, value) in optional_string_fields {
            if let Some(v) = value {
                execution.insert(key.into(), Value::String(v.clone()));
            }
        }

        if let Some(secs) = self.execution_time_seconds
            && let Some(num) = serde_json::Number::from_f64(secs)
        {
            execution.insert("execution_time_seconds".into(), Value::Number(num));
        }

        let mut payload = self.extra_payload_fields.clone();
        move_execution_field(&mut payload, &mut execution, "run_id", "run_id");
        move_execution_field(&mut payload, &mut execution, "outputs", "outputs");
        move_execution_field(&mut payload, &mut execution, "artifacts", "outputs");
        move_execution_field(
            &mut payload,
            &mut execution,
            "published_outputs",
            "published_outputs",
        );
        move_execution_field(
            &mut payload,
            &mut execution,
            "published_artifacts",
            "published_artifacts",
        );
        move_execution_field(&mut payload, &mut execution, "batch_info", "batch_info");

        (execution, Value::Object(payload))
    }
}

fn normalize_status(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "success" | "succeeded" | "completed" => "succeeded".to_string(),
        "fail" | "failed" | "error" => "failed".to_string(),
        other => other.to_string(),
    }
}

fn stage_for_status(status: &str) -> String {
    if status == "failed" {
        "failed".to_string()
    } else {
        "completed".to_string()
    }
}

fn move_execution_field(
    payload: &mut Map<String, Value>,
    execution: &mut Map<String, Value>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = payload.remove(source_key) {
        execution.entry(target_key.to_string()).or_insert(value);
    }
}

fn object_field(value: Option<Value>, field_name: &str) -> Result<Map<String, Value>> {
    match value {
        Some(Value::Object(map)) => Ok(map),
        Some(other) => Err(anyhow!(
            "results: {field_name} field must be a JSON object, got {other}"
        )),
        None => Ok(Map::new()),
    }
}

fn string_field(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(ToOwned::to_owned)
}

fn f64_field(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn insert_optional_string(map: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if !map.contains_key(key)
        && let Some(value) = value
    {
        map.insert(key.to_string(), Value::String(value));
    }
}

fn insert_optional_f64(map: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    if map.contains_key(key) {
        return;
    }
    if let Some(value) = value
        && let Some(number) = serde_json::Number::from_f64(value)
    {
        map.insert(key.to_string(), Value::Number(number));
    }
}

fn derive_primary_output_path(outputs: Option<&Value>) -> Option<String> {
    let outputs = outputs?.as_array()?;
    let primary = outputs
        .iter()
        .find(|entry| {
            entry
                .get("primary")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| outputs.first())?;
    primary
        .get("storage_path")
        .or_else(|| primary.get("path"))
        .or_else(|| primary.get("output_path"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn normalize_execution_map(execution: &mut Map<String, Value>) {
    if let Some(artifacts) = execution.remove("artifacts") {
        execution.entry("outputs".to_string()).or_insert(artifacts);
    }
    if !execution.contains_key("output_path")
        && let Some(path) = derive_primary_output_path(execution.get("outputs"))
    {
        execution.insert("output_path".to_string(), Value::String(path));
    }
}

fn resolve_run_id(
    explicit_run_id: Option<String>,
    execution_run_id: Option<&Value>,
    fallback_run_id: &str,
) -> Result<String> {
    let run_id = explicit_run_id
        .or_else(|| string_field(execution_run_id))
        .unwrap_or_else(|| fallback_run_id.to_string())
        .trim()
        .to_string();
    if run_id.is_empty() {
        return Err(anyhow!("results: missing run_id"));
    }
    Ok(run_id)
}

fn build_persisted_result(
    run_id: String,
    status: String,
    ttl_seconds: u64,
    request: Map<String, Value>,
    mut execution: Map<String, Value>,
    payload: Value,
) -> PersistedResult {
    normalize_execution_map(&mut execution);
    let stage = stage_for_status(&status);
    let completed_at = string_field(execution.get("completed_at"));
    let workflow = string_field(execution.get("workflow"));
    let gpu_stream = string_field(execution.get("gpu_stream"));
    let output_path = string_field(execution.get("output_path"));
    let output_archive = string_field(execution.get("output_archive"));
    let error = string_field(execution.get("error"));
    let execution_time_seconds = f64_field(execution.get("execution_time_seconds"));
    let mut result_payload = Map::new();
    result_payload.insert("request".to_string(), Value::Object(request));
    result_payload.insert("execution".to_string(), Value::Object(execution));
    result_payload.insert("payload".to_string(), payload);
    PersistedResult {
        run_id,
        status,
        stage,
        completed_at,
        workflow,
        gpu_stream,
        output_path,
        output_archive,
        error,
        execution_time_seconds,
        result_payload: Value::Object(result_payload),
        result_ttl_seconds: ttl_seconds,
    }
}

fn parse_nested_payload(payload: Option<Value>) -> Result<NestedResultPayload> {
    match payload {
        Some(Value::String(raw)) => serde_json::from_str::<NestedResultPayload>(&raw)
            .context("results: payload field must contain valid nested JSON"),
        Some(Value::Object(map)) => {
            serde_json::from_value(Value::Object(map)).context("results: invalid payload object")
        }
        Some(other) => Err(anyhow!(
            "results: payload field must be a JSON string/object, got {}",
            other
        )),
        None => Ok(NestedResultPayload::default()),
    }
}

fn parse_result_message(msg: &scicomp_rq::Message, ttl_seconds: u64) -> Result<PersistedResult> {
    if msg.payload().trim().is_empty() {
        return Err(anyhow!("results: empty payload"));
    }

    let envelope: ResultsEnvelope =
        serde_json::from_str(msg.payload()).context("results: payload must be valid JSON")?;
    if envelope.request.is_some() || envelope.execution.is_some() {
        let mut request = object_field(envelope.request, "request")?;
        if !request.contains_key("operation")
            && let Some(operation) = envelope.operation.clone()
        {
            request.insert("operation".to_string(), Value::String(operation));
        }

        let mut execution = object_field(envelope.execution, "execution")?;
        let run_id = resolve_run_id(
            envelope.run_id.clone(),
            execution.get("run_id"),
            msg.run_id(),
        )?;
        let raw_status = envelope
            .status
            .clone()
            .or_else(|| string_field(execution.get("status")))
            .ok_or_else(|| anyhow!("results: missing status"))?;
        let status = normalize_status(&raw_status);
        insert_optional_string(&mut execution, "run_id", Some(run_id.clone()));
        insert_optional_string(&mut execution, "status", Some(status.clone()));
        insert_optional_string(&mut execution, "workflow", envelope.workflow.clone());
        insert_optional_string(&mut execution, "gpu_stream", envelope.gpu_stream.clone());
        insert_optional_string(
            &mut execution,
            "completed_at",
            envelope.completed_at.clone(),
        );
        insert_optional_string(&mut execution, "output_path", envelope.output_path.clone());
        insert_optional_string(
            &mut execution,
            "output_archive",
            envelope.output_archive.clone(),
        );
        insert_optional_string(&mut execution, "error", envelope.error.clone());
        insert_optional_f64(
            &mut execution,
            "execution_time_seconds",
            envelope.execution_time_seconds,
        );
        let payload = envelope
            .payload
            .unwrap_or_else(|| Value::Object(Map::new()));
        return Ok(build_persisted_result(
            run_id,
            status,
            ttl_seconds,
            request,
            execution,
            payload,
        ));
    }

    let nested = parse_nested_payload(envelope.payload.clone())?;
    let run_id = resolve_run_id(envelope.run_id.clone(), None, msg.run_id())?;
    let raw_status = envelope
        .status
        .clone()
        .or(nested.status.clone())
        .ok_or_else(|| anyhow!("results: missing status"))?;
    let status = normalize_status(&raw_status);
    let merged = MergedFields::from_envelope_and_nested(&envelope, &nested);
    let (execution, payload) = merged.build_execution_and_payload(&run_id, &status);
    let mut request = Map::new();
    if let Some(operation) = envelope.operation {
        request.insert("operation".to_string(), Value::String(operation));
    }
    Ok(build_persisted_result(
        run_id,
        status,
        ttl_seconds,
        request,
        execution,
        payload,
    ))
}

impl WorkerRole for ResultsRole {
    fn name(&self) -> &'static str {
        "results"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        _sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            let parsed = parse_result_message(msg, self.result_ttl_seconds)?;
            let run_id = parsed.run_id.clone();
            let status = parsed.status.clone();
            self.persistence.persist_result(parsed).await?;
            tracing::info!(run_id = %run_id, status = %status, "result persisted");
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    struct NoopSink;
    impl MessageSink for NoopSink {
        fn enqueue<'a>(
            &'a self,
            _: &'a str,
            _: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("x".into()) })
        }
        fn ack_message<'a>(&'a self, _: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn handoff<'a>(
            &'a self,
            _: &'a scicomp_rq::Message,
            _: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("x".into()) })
        }
        fn forward_many<'a>(
            &'a self,
            _: &'a scicomp_rq::Message,
            _: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    #[derive(Default)]
    struct RecordingStore {
        persisted: Mutex<Vec<PersistedResult>>,
    }

    impl RecordingStore {
        fn persisted(&self) -> Vec<PersistedResult> {
            self.persisted.lock().expect("recording lock").clone()
        }
    }

    impl ResultsPersistence for RecordingStore {
        fn persist_result<'a>(&'a self, result: PersistedResult) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.persisted.lock().expect("recording lock").push(result);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn results_accepts_valid_payload() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{
                "status":"success",
                "workflow":"deterministic_workflow",
                "gpu_stream":"physicsnemo:gpu:default:pod-0:0",
                "completed_at":"2026-02-20T10:00:00Z",
                "payload":"{\"output_path\":\"/outputs/run-1/results.nc\",\"execution_time_seconds\":3.5}"
            }"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].run_id, "run-1");
        assert_eq!(persisted[0].status, "succeeded");
        assert_eq!(persisted[0].stage, "completed");
        assert_eq!(
            persisted[0].result_payload["execution"]["output_path"],
            Value::String("/outputs/run-1/results.nc".to_string())
        );
    }

    #[tokio::test]
    async fn results_rejects_empty_payload() {
        let role = ResultsRole::new_for_test(Arc::new(NoopResultsPersistence::new()), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            "   ",
            "results",
        );

        let result = role.handle(&msg, "results", &NoopSink).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn results_rejects_non_json_payload() {
        let role = ResultsRole::new_for_test(Arc::new(NoopResultsPersistence::new()), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            "not-json",
            "results",
        );

        let result = role.handle(&msg, "results", &NoopSink).await;
        assert!(
            result.is_err(),
            "results role must reject non-JSON payloads"
        );
    }

    #[tokio::test]
    async fn results_rejects_unexpected_stream() {
        let role = ResultsRole::new_for_test(Arc::new(NoopResultsPersistence::new()), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"status":"succeeded","payload":"{\"output_path\":\"/tmp/out\"}"}"#,
            "results",
        );

        let result = role.handle(&msg, "not-results", &NoopSink).await;
        assert!(
            result.is_err(),
            "results role must reject messages from unexpected streams"
        );
    }

    #[tokio::test]
    async fn results_rejects_invalid_nested_payload_json() {
        let role = ResultsRole::new_for_test(Arc::new(NoopResultsPersistence::new()), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"status":"succeeded","payload":"{invalid-json"}"#,
            "results",
        );

        let result = role.handle(&msg, "results", &NoopSink).await;
        assert!(
            result.is_err(),
            "results role must reject invalid nested payload JSON"
        );
    }

    #[tokio::test]
    async fn normalize_status_passes_through_unknown_values() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"status":"cancelled"}"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted[0].status, "cancelled");
        assert_eq!(persisted[0].stage, "completed");
    }

    #[tokio::test]
    async fn results_rejects_missing_status() {
        let role = ResultsRole::new_for_test(Arc::new(NoopResultsPersistence::new()), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"workflow":"wf"}"#,
            "results",
        );

        let err = role.handle(&msg, "results", &NoopSink).await.unwrap_err();
        assert!(
            format!("{err:?}").contains("missing status"),
            "expected 'missing status' in error, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn results_accepts_nested_payload_as_object() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"status":"success","payload":{"output_path":"/out/run-1","execution_time_seconds":1.5}}"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].output_path.as_deref(), Some("/out/run-1"));
        assert_eq!(persisted[0].execution_time_seconds, Some(1.5));
    }

    #[tokio::test]
    async fn results_preserves_plugin_specific_payload_fields() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-plugin",
            r#"{
                "status":"success",
                "workflow":"demo-plugin",
                "payload":{
                    "output_path":"/artifacts/run-plugin/pressure_field.npz",
                    "echo_operation":"both",
                    "artifacts":[{"name":"pressure_field","media_type":"application/x-npz"}]
                }
            }"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(
            persisted[0].result_payload["payload"]["echo_operation"],
            Value::String("both".to_string())
        );
        assert_eq!(
            persisted[0].result_payload["execution"]["outputs"][0]["name"],
            Value::String("pressure_field".to_string())
        );
    }

    #[tokio::test]
    async fn results_moves_legacy_run_id_into_execution_block() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-legacy",
            r#"{
                "status":"success",
                "payload":{
                    "run_id":"run-legacy",
                    "output_path":"/out/run-legacy",
                    "echo_operation":"run"
                }
            }"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(
            persisted[0].result_payload,
            json!({
                "request": {},
                "execution": {
                    "run_id": "run-legacy",
                    "status": "succeeded",
                    "output_path": "/out/run-legacy"
                },
                "payload": {
                    "echo_operation": "run"
                }
            })
        );
    }

    #[tokio::test]
    async fn results_persists_structured_result_envelope() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-structured",
            r#"{
                "run_id":"run-structured",
                "status":"success",
                "workflow":"demo-plugin",
                "gpu_stream":"physicsnemo:gpu:default:pod-0:0",
                "completed_at":"2026-02-20T10:00:00Z",
                "request":{
                    "operation":"run",
                    "content_type":"application/json",
                    "raw_fields":{"batch_size":128000},
                    "input_artifacts":[]
                },
                "execution":{
                    "output_path":"/outputs/run-structured/results.nc",
                    "execution_time_seconds":3.5,
                    "outputs":[
                        {
                            "name":"pressure_field",
                            "media_type":"application/x-netcdf",
                            "storage_path":"/outputs/run-structured/results.nc",
                            "primary":true
                        }
                    ]
                },
                "payload":{
                    "echo_operation":"both"
                }
            }"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].run_id, "run-structured");
        assert_eq!(persisted[0].status, "succeeded");
        assert_eq!(persisted[0].stage, "completed");
        assert_eq!(
            persisted[0].result_payload,
            json!({
                "request": {
                    "operation": "run",
                    "content_type": "application/json",
                    "raw_fields": { "batch_size": 128000 },
                    "input_artifacts": []
                },
                "execution": {
                    "run_id": "run-structured",
                    "status": "succeeded",
                    "workflow": "demo-plugin",
                    "gpu_stream": "physicsnemo:gpu:default:pod-0:0",
                    "completed_at": "2026-02-20T10:00:00Z",
                    "output_path": "/outputs/run-structured/results.nc",
                    "execution_time_seconds": 3.5,
                    "outputs": [
                        {
                            "name": "pressure_field",
                            "media_type": "application/x-netcdf",
                            "storage_path": "/outputs/run-structured/results.nc",
                            "primary": true
                        }
                    ]
                },
                "payload": {
                    "echo_operation": "both"
                }
            })
        );
    }

    #[test]
    fn batch_hash_fields_extracts_batch_info_summary() {
        let payload = serde_json::json!({
            "run_id": "run-a",
            "status": "succeeded",
            "batch_info": {
                "batch_id": "batch-1",
                "batch_size": 2,
                "flush_reason": "max_batch_size",
                "first_seen_ms": 1000,
                "formed_at_ms": 1025,
                "waited_ms": 25
            }
        });

        let fields = BatchHashFields::from_result_payload(&payload);
        assert_eq!(fields.batch_id.as_deref(), Some("batch-1"));
        assert_eq!(fields.batch_size.as_deref(), Some("2"));
        assert_eq!(fields.batch_flush_reason.as_deref(), Some("max_batch_size"));
        assert_eq!(fields.batch_first_seen_ms.as_deref(), Some("1000"));
        assert_eq!(fields.batch_formed_at_ms.as_deref(), Some("1025"));
        assert_eq!(fields.batch_waited_ms.as_deref(), Some("25"));
    }

    #[tokio::test]
    async fn results_propagates_persistence_failure() {
        struct FailingStore;
        impl ResultsPersistence for FailingStore {
            fn persist_result<'a>(&'a self, _result: PersistedResult) -> BoxFuture<'a, Result<()>> {
                Box::pin(async { Err(anyhow!("simulated persistence failure")) })
            }
        }

        let role = ResultsRole::new_for_test(Arc::new(FailingStore), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "run-1",
            r#"{"status":"succeeded"}"#,
            "results",
        );

        let err = role.handle(&msg, "results", &NoopSink).await.unwrap_err();
        assert!(
            format!("{err:?}").contains("simulated persistence failure"),
            "expected persistence error to propagate, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn results_falls_back_to_message_run_id_when_envelope_omits_it() {
        let store = Arc::new(RecordingStore::default());
        let role = ResultsRole::new_for_test(store.clone(), 86_400);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "test:results",
            "results:grp",
            "header-run-id",
            r#"{"status":"success"}"#,
            "results",
        );

        role.handle(&msg, "results", &NoopSink).await.unwrap();

        let persisted = store.persisted();
        assert_eq!(persisted[0].run_id, "header-run-id");
    }

    #[tokio::test]
    async fn redis_timeout_wrapper_returns_value_before_deadline() {
        let result = with_redis_timeout("test-op", Duration::from_millis(20), async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok::<usize, anyhow::Error>(42)
        })
        .await
        .expect("operation should complete before timeout");

        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn redis_timeout_wrapper_returns_clear_timeout_error() {
        let err = with_redis_timeout("test-op", Duration::from_millis(1), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok::<(), anyhow::Error>(())
        })
        .await
        .expect_err("operation should time out");

        let message = err.to_string();
        assert!(
            message.contains("timed out"),
            "expected timeout error message, got: {message}"
        );
        assert!(
            message.contains("test-op"),
            "expected operation name in timeout message, got: {message}"
        );
    }
}
