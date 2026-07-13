/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Redis operations helpers

use crate::config::ServerConfig;
use scicomp_rq::{LogicalStreamName, QueueManager, StreamKey};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use tracing::{error, info, warn};

/// Errors from Redis operations.
#[derive(Debug)]
pub enum RedisOpsError {
    /// Failed to serialize the enqueue payload.
    Serialization(serde_json::Error),
    /// Failed to enqueue to the Redis stream.
    Enqueue(String),
}

impl fmt::Display for RedisOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RedisOpsError::Serialization(e) => write!(f, "payload serialization failed: {}", e),
            RedisOpsError::Enqueue(msg) => write!(f, "enqueue failed: {}", msg),
        }
    }
}

impl std::error::Error for RedisOpsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RedisOpsError::Serialization(e) => Some(e),
            RedisOpsError::Enqueue(_) => None,
        }
    }
}

/// Redis key prefix for run data
const REDIS_RUN_PREFIX: &str = "run:";
/// Redis key prefix for results
const REDIS_RESULT_PREFIX: &str = "result:";

/// Heuristic: stream names containing `:` are treated as explicit Redis
/// stream keys (e.g. `physicsnemo:prefetch`). Names without `:` are treated
/// as logical stream names resolved by `QueueManager`.
///
/// This convention is intentional - explicit keys use the colon-separated
/// namespace pattern from our Redis key schema.
fn uses_explicit_stream_api(stream_name: &str) -> bool {
    stream_name.contains(':')
}

/// Service for Redis operations
#[derive(Clone)]
pub struct RedisService {
    qm: QueueManager,
}

impl RedisService {
    /// Create a new RedisService
    pub fn new(qm: QueueManager) -> Self {
        Self { qm }
    }

    /// Create a new RedisService with a shared client from connection string
    pub async fn from_config(
        config: &ServerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Queue primitives do not own stream-topology config; server config does.
        let qm = QueueManager::from_redis_url(&config.redis_url).await?;
        Ok(Self::new(qm))
    }

    /// Get the underlying QueueManager
    pub fn queue_manager(&self) -> &QueueManager {
        &self.qm
    }

    /// Get a clone of the Redis connection manager
    pub fn get_connection(&self) -> redis::aio::ConnectionManager {
        self.qm.connection()
    }

    /// Connect to Redis and create a service instance (Legacy/Convenience)
    pub async fn connect(
        config: &ServerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let qm = QueueManager::from_redis_url(&config.redis_url).await?;

        let mut conn = qm.connection();
        let _: String = redis::cmd("PING").query_async(&mut conn).await?;
        info!("Connected to Redis: {}", config.redis_url);

        Ok(Self::new(qm))
    }

    /// Store the initial queued run record in Redis.
    #[allow(clippy::too_many_arguments)]
    pub async fn store_queued_run(
        &self,
        run_id: &str,
        workflow: &str,
        version: &str,
        operation: &str,
        initial_stage: &str,
        api_received_at: &str,
        api_enqueued_at: &str,
        output_publication_configured: bool,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.qm.connection();
        let mut hset = redis::cmd("HSET");
        hset.arg(Self::run_key(run_id))
            .arg("workflow")
            .arg(workflow)
            .arg("version")
            .arg(version)
            .arg("operation")
            .arg(operation)
            .arg("status")
            .arg("queued")
            .arg("stage")
            .arg(initial_stage)
            .arg("updated_at")
            .arg(api_enqueued_at)
            .arg("api_received_at")
            .arg(api_received_at)
            .arg("api_enqueued_at")
            .arg(api_enqueued_at);
        if output_publication_configured {
            hset.arg("output_location").arg("local_and_cloud");
        } else {
            hset.arg("output_location").arg("local");
        }
        hset.query_async(&mut conn).await
    }

    /// Delete queued run data from Redis.
    pub(crate) async fn delete_run_data(&self, run_id: &str) -> Result<(), redis::RedisError> {
        let mut conn = self.qm.connection();
        redis::cmd("DEL")
            .arg(Self::run_key(run_id))
            .query_async::<i64>(&mut conn)
            .await
            .map(|_| ())
    }

    /// Get run data from Redis
    pub async fn get_run_data(&self, run_id: &str) -> Result<Option<Value>, redis::RedisError> {
        let mut conn = self.qm.connection();
        let fields: HashMap<String, String> = redis::cmd("HGETALL")
            .arg(Self::run_key(run_id))
            .query_async(&mut conn)
            .await?;

        if !fields.is_empty() {
            return Ok(Some(run_hash_fields_to_value(fields)));
        }
        Ok(None)
    }

    /// Get full result payload from Redis (stored by results worker)
    pub async fn get_result_payload(
        &self,
        run_id: &str,
    ) -> Result<Option<Value>, redis::RedisError> {
        let mut conn = self.qm.connection();
        let result_key = Self::result_key(run_id);

        let data: Option<String> = redis::cmd("GET")
            .arg(&result_key)
            .query_async(&mut conn)
            .await?;

        match data {
            Some(d) => match serde_json::from_str::<Value>(&d) {
                Ok(json) => Ok(Some(json)),
                Err(e) => {
                    warn!(
                        run_id = %run_id,
                        key = %result_key,
                        error = %e,
                        "Failed to parse result payload JSON - data may be corrupted"
                    );
                    Ok(None)
                }
            },
            None => Ok(None),
        }
    }

    pub async fn enqueue_value_to_stream(
        &self,
        stream_name: &str,
        stage: &str,
        run_id: &str,
        payload: &Value,
    ) -> Result<String, RedisOpsError> {
        self.enqueue_json_value(stream_name, stage, run_id, payload)
            .await
    }

    async fn enqueue_json_value(
        &self,
        stream_name: &str,
        stage: &str,
        run_id: &str,
        payload: &Value,
    ) -> Result<String, RedisOpsError> {
        let payload_json = serde_json::to_string(payload).map_err(RedisOpsError::Serialization)?;
        let enqueue_result = if uses_explicit_stream_api(stream_name) {
            let stream_key = StreamKey::new(stream_name);
            self.qm
                .enqueue_to_stream(&stream_key, run_id, &payload_json, stage)
                .await
        } else {
            let logical_stream = LogicalStreamName::new(stream_name);
            self.qm
                .enqueue_to(logical_stream)
                .run_id(run_id)
                .payload(payload_json.as_str())
                .stage(stage)
                .send()
                .await
        };

        match enqueue_result {
            Ok(stream_id) => Ok(stream_id),
            Err(e) => {
                error!(error=%e, "enqueue to stream {} failed", stream_name);
                Err(RedisOpsError::Enqueue(e.to_string()))
            }
        }
    }

    fn run_key(run_id: &str) -> String {
        format!("{}{}", REDIS_RUN_PREFIX, run_id)
    }

    fn result_key(run_id: &str) -> String {
        format!("{}{}", REDIS_RESULT_PREFIX, run_id)
    }
}

fn parse_hash_field_value(raw: &str) -> Value {
    if let Ok(parsed) = raw.parse::<u64>() {
        return Value::Number(parsed.into());
    }
    Value::String(raw.to_string())
}

fn run_hash_fields_to_value(fields: HashMap<String, String>) -> Value {
    let mut obj = serde_json::Map::new();
    for (k, v) in &fields {
        obj.insert(k.clone(), Value::String(v.clone()));
    }

    let mut fanout_progress = serde_json::Map::new();
    for (hash_key, json_key) in [
        ("fanout_expected_count", "expected_count"),
        ("fanout_collected_count", "collected_count"),
        ("fanout_succeeded_count", "succeeded_count"),
        ("fanout_failed_count", "failed_count"),
        ("fanout_cancelled_count", "cancelled_count"),
    ] {
        if let Some(value) = fields.get(hash_key) {
            fanout_progress.insert(json_key.to_string(), parse_hash_field_value(value));
        }
    }
    if let (Some(expected), Some(collected)) = (
        fields
            .get("fanout_expected_count")
            .and_then(|value| value.parse::<u64>().ok()),
        fields
            .get("fanout_collected_count")
            .and_then(|value| value.parse::<u64>().ok()),
    ) {
        fanout_progress.insert(
            "remaining_count".to_string(),
            Value::Number(expected.saturating_sub(collected).into()),
        );
    }
    if let Some(value) = fields.get("fanout_child_run_ids")
        && let Ok(parsed) = serde_json::from_str::<Value>(value)
    {
        fanout_progress.insert("child_run_ids".to_string(), parsed);
    }
    if !fanout_progress.is_empty() {
        obj.insert(
            "fanout_progress".to_string(),
            Value::Object(fanout_progress),
        );
    }

    let mut batch_info = serde_json::Map::new();
    for (hash_key, json_key) in [
        ("batch_id", "batch_id"),
        ("batch_size", "batch_size"),
        ("batch_flush_reason", "flush_reason"),
        ("batch_first_seen_ms", "first_seen_ms"),
        ("batch_formed_at_ms", "formed_at_ms"),
        ("batch_waited_ms", "waited_ms"),
    ] {
        if let Some(value) = fields.get(hash_key) {
            batch_info.insert(json_key.to_string(), parse_hash_field_value(value));
        }
    }
    if !batch_info.is_empty() {
        obj.insert("batch_info".to_string(), Value::Object(batch_info));
    }

    for json_field in ["published_artifacts", "outputs", "artifacts"] {
        if let Some(value) = fields.get(json_field)
            && let Ok(parsed) = serde_json::from_str::<Value>(value)
        {
            obj.insert(json_field.to_string(), parsed);
        }
    }

    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use serde_json::json;
    use std::collections::HashMap;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::Duration;
    use tokio::sync::Mutex;

    static REDIS_ENV_LOCK: Mutex<()> = Mutex::const_new(());

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: serialized by REDIS_ENV_LOCK; no other test mutates
            // this env var while the guard is alive.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: same serialization guarantee as `set`.
            unsafe {
                match &self.previous {
                    Some(prev) => std::env::set_var(self.key, prev),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    struct TestRedisServer {
        child: Child,
        data_dir: PathBuf,
    }

    impl TestRedisServer {
        async fn spawn(test_name: &str, port: u16) -> Self {
            let data_dir = std::env::temp_dir().join(format!(
                "physicsnemo-serve-redis-{}-{}",
                test_name,
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&data_dir).expect("redis data dir should be created");
            let child = Command::new("redis-server")
                .arg("--save")
                .arg("")
                .arg("--appendonly")
                .arg("no")
                .arg("--bind")
                .arg("127.0.0.1")
                .arg("--port")
                .arg(port.to_string())
                .arg("--dir")
                .arg(&data_dir)
                .arg("--loglevel")
                .arg("warning")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("redis-server should start");

            wait_for_tcp_listener(port).await;

            Self { child, data_dir }
        }
    }

    impl Drop for TestRedisServer {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = std::fs::remove_dir_all(&self.data_dir);
        }
    }

    fn reserve_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral port should be allocated")
            .local_addr()
            .expect("listener should have a local addr")
            .port()
    }

    async fn wait_for_tcp_listener(port: u16) {
        for _ in 0..50 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("redis-server on port {port} did not become ready in time");
    }

    fn test_config(redis_url: String) -> ServerConfig {
        ServerConfig {
            addr: "127.0.0.1:8080".parse().unwrap(),
            redis_url,
            redis_stream: "inference".to_string(),
            prefetch_stream: "prefetch".to_string(),
            use_prefetch: true,
            plugin_dirs: vec![],
            enabled_plugin_id: None,
            artifact_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-redis-artifacts-{}",
                uuid::Uuid::new_v4()
            )),
            default_output_dir: std::env::temp_dir().join(format!(
                "physicsnemo-serve-redis-outputs-{}",
                uuid::Uuid::new_v4()
            )),
            artifact_retention_hours: 24,
            artifact_cleanup_interval_secs: 30,
            cors_allowed_origins: vec![],
            max_body_size: 2 * 1024 * 1024,
            stream_prefix: String::new(),
            swagger_cdn_url: None,
            python_runtime_envs: HashMap::new(),
            output_publication: Default::default(),
        }
    }

    #[test]
    fn test_run_key_generation() {
        let run_id = "test-run-123";
        let key = RedisService::run_key(run_id);
        assert_eq!(key, "run:test-run-123");
    }

    #[test]
    fn test_result_key_generation() {
        let run_id = "test-run-123";
        let key = RedisService::result_key(run_id);
        assert_eq!(key, "result:test-run-123");
    }

    #[test]
    fn test_run_hash_fields_to_value_rehydrates_fanout_progress_object() {
        let mut fields = HashMap::new();
        fields.insert("status".to_string(), "running".to_string());
        fields.insert("fanout_expected_count".to_string(), "20".to_string());
        fields.insert("fanout_collected_count".to_string(), "3".to_string());
        fields.insert("fanout_succeeded_count".to_string(), "2".to_string());
        fields.insert("fanout_failed_count".to_string(), "1".to_string());
        fields.insert("fanout_cancelled_count".to_string(), "0".to_string());
        fields.insert(
            "fanout_child_run_ids".to_string(),
            r#"["parent:item:0","parent:item:1","parent:item:2"]"#.to_string(),
        );

        let value = run_hash_fields_to_value(fields);
        assert_eq!(value["fanout_progress"]["expected_count"], 20);
        assert_eq!(value["fanout_progress"]["collected_count"], 3);
        assert_eq!(value["fanout_progress"]["remaining_count"], 17);
        assert_eq!(value["fanout_progress"]["succeeded_count"], 2);
        assert_eq!(value["fanout_progress"]["failed_count"], 1);
        assert_eq!(value["fanout_progress"]["cancelled_count"], 0);
        assert_eq!(
            value["fanout_progress"]["child_run_ids"],
            json!(["parent:item:0", "parent:item:1", "parent:item:2"])
        );
    }

    #[test]
    fn test_uses_explicit_stream_api_for_prefixed_stream_keys() {
        assert!(uses_explicit_stream_api("physicsnemo:prefetch"));
    }

    #[test]
    fn test_uses_explicit_stream_api_for_logical_stream_names() {
        assert!(!uses_explicit_stream_api("prefetch"));
        assert!(!uses_explicit_stream_api("inference"));
    }

    #[test]
    fn test_redis_ops_error_display_serialization() {
        let json_err = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let err = RedisOpsError::Serialization(json_err);
        let msg = format!("{}", err);
        assert!(msg.contains("payload serialization failed"), "got: {}", msg);
    }

    #[test]
    fn test_redis_ops_error_display_enqueue() {
        let err = RedisOpsError::Enqueue("connection refused".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("enqueue failed"), "got: {}", msg);
        assert!(msg.contains("connection refused"), "got: {}", msg);
    }

    #[test]
    fn test_run_hash_fields_to_value_rehydrates_batch_info_object() {
        let value = run_hash_fields_to_value(HashMap::from([
            ("status".to_string(), "succeeded".to_string()),
            ("batch_id".to_string(), "batch-1".to_string()),
            ("batch_size".to_string(), "2".to_string()),
            (
                "batch_flush_reason".to_string(),
                "max_batch_size".to_string(),
            ),
            ("batch_waited_ms".to_string(), "25".to_string()),
        ]));

        assert_eq!(value["status"], "succeeded");
        assert_eq!(value["batch_info"]["batch_id"], "batch-1");
        assert_eq!(value["batch_info"]["batch_size"], 2);
        assert_eq!(value["batch_info"]["flush_reason"], "max_batch_size");
        assert_eq!(value["batch_info"]["waited_ms"], 25);
    }

    #[test]
    fn test_run_hash_fields_to_value_rehydrates_publication_metadata() {
        let value = run_hash_fields_to_value(HashMap::from([
            ("status".to_string(), "running".to_string()),
            (
                "published_artifacts".to_string(),
                r#"[{"provider":"s3","source_artifact":"primary","destination_uri":"s3://bucket/result.json","status":"uploaded"}]"#
                    .to_string(),
            ),
            (
                "outputs".to_string(),
                r#"[{"name":"primary","storage_path":"/outputs/run/result.json","primary":true}]"#
                    .to_string(),
            ),
        ]));

        assert_eq!(value["published_artifacts"][0]["provider"], "s3");
        assert_eq!(
            value["outputs"][0]["storage_path"],
            "/outputs/run/result.json"
        );
    }

    #[tokio::test]
    async fn connect_uses_server_config_redis_url() {
        let _lock = REDIS_ENV_LOCK.lock().await;
        let env_port = reserve_port();
        let config_port = reserve_port();
        let _env_server = TestRedisServer::spawn("env", env_port).await;
        let _config_server = TestRedisServer::spawn("config", config_port).await;
        let _env_guard = EnvGuard::set("REDIS_URL", format!("redis://127.0.0.1:{env_port}/0"));

        let service =
            RedisService::connect(&test_config(format!("redis://127.0.0.1:{config_port}/0")))
                .await
                .expect("redis connect should succeed");

        let mut conn = service.get_connection();
        let info: String = redis::cmd("INFO")
            .arg("server")
            .query_async(&mut conn)
            .await
            .expect("INFO server should succeed");

        assert!(
            info.contains(&format!("tcp_port:{config_port}")),
            "connect() should honor ServerConfig.redis_url, got INFO: {info}"
        );
    }

    #[tokio::test]
    async fn delete_run_data_removes_persisted_queued_run_hash() {
        let port = reserve_port();
        let _redis_server = TestRedisServer::spawn("delete-run-data", port).await;
        let service = RedisService::connect(&test_config(format!("redis://127.0.0.1:{port}/0")))
            .await
            .expect("redis connect should succeed");

        service
            .store_queued_run(
                "run-delete",
                "demo-prefetch",
                "1.0.0",
                "run",
                "prepare",
                "100",
                "100",
                false,
            )
            .await
            .expect("queued run hash should be stored");
        assert!(
            service
                .get_run_data("run-delete")
                .await
                .expect("stored run should be readable")
                .is_some(),
            "expected queued run hash to exist before deletion"
        );

        service
            .delete_run_data("run-delete")
            .await
            .expect("queued run hash deletion should succeed");

        assert!(
            service
                .get_run_data("run-delete")
                .await
                .expect("deleted run lookup should succeed")
                .is_none(),
            "queued run hash should be removed after deletion"
        );
    }

    #[tokio::test]
    async fn store_queued_run_defers_publication_status_when_configured() {
        let port = reserve_port();
        let _redis_server = TestRedisServer::spawn("queued-run-publication", port).await;
        let service = RedisService::connect(&test_config(format!("redis://127.0.0.1:{port}/0")))
            .await
            .expect("redis connect should succeed");

        service
            .store_queued_run(
                "run-publish",
                "demo-prefetch",
                "1.0.0",
                "run",
                "prepare",
                "100",
                "100",
                true,
            )
            .await
            .expect("queued run hash should be stored");

        let run_data = service
            .get_run_data("run-publish")
            .await
            .expect("stored run should be readable")
            .expect("queued run should exist");
        assert_eq!(run_data["status"], "queued");
        assert_eq!(run_data["output_location"], "local_and_cloud");
        assert!(run_data.get("output_publication_status").is_none());
    }
}
