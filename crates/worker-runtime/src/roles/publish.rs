use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::path::Path as ObjectPath;
use object_store::{ClientConfigKey, ObjectStore, PutMode, PutOptions, PutPayload, RetryConfig};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::sync::watch;
use tracing::info;
use uuid::Uuid;

use crate::config::{PublishRoleConfig, parse_role_config};
use crate::roles::stage::StageContext;
use crate::traits::{
    BoxFuture, MessageSink, RoleCancellation, RoleEnv, WorkerRole, is_message_ownership_lost_error,
    message_deferred, message_ownership_lost,
};

const PRIMARY_TARGET_NAME: &str = "primary";
const MIN_MULTIPART_PART_SIZE_BYTES: usize = 5 * 1024 * 1024;
const ENV_PUBLISH_ROLE_CONFIG_JSON: &str = "PHYSICSNEMO_SERVE_PUBLISH_ROLE_CONFIG_JSON";
const PUBLISH_MANIFEST_FILENAME: &str = "_physicsnemo_serve_publish_manifest.json";
const PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS: u64 = 600;
const PUBLISH_CLAIM_RENEW_INTERVAL_SECS: u64 = 60;
const PUBLISH_CLAIM_TERMINAL_TTL_SECS: u64 = 24 * 60 * 60;
const REPLACE_PUBLISH_CLAIM_IF_UNCHANGED_LUA: &str = r#"
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0
"#;
const REPLACE_PUBLISH_CLAIM_VALUE_IF_UNCHANGED_LUA: &str = r#"
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0
"#;

#[derive(Debug, Clone, Deserialize)]
struct PublishEnvelope {
    workflow_id: String,
    operation: Option<String>,
    result: JsonValue,
    #[serde(default)]
    output_publication: Option<ResolvedOutputPublication>,
    stage_context: StageContext,
}

#[derive(Debug, Clone, Deserialize)]
struct ResolvedOutputPublication {
    target: ResolvedOutputPublicationTarget,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OutputPublicationProvider {
    S3,
    Azure,
}

#[derive(Debug, Clone, Deserialize)]
struct ResolvedOutputPublicationTarget {
    artifact: String,
    provider: OutputPublicationProvider,
    storage: ResolvedOutputPublicationStorage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResolvedOutputPublicationStorage {
    S3 {
        bucket: String,
        prefix: String,
        #[serde(default)]
        region: Option<String>,
        #[serde(default)]
        endpoint: Option<String>,
    },
    Azure {
        container: String,
        prefix: String,
        endpoint: String,
    },
}

#[derive(Debug, Clone)]
struct SelectedArtifact {
    name: String,
    storage_path: PathBuf,
    filename: String,
}

struct PublicationTarget {
    store: Arc<dyn ObjectStore>,
    prefix: ObjectPath,
    destination_uri: String,
}

#[derive(Debug, Clone)]
struct UploadStats {
    destination_uri: String,
    manifest_uri: Option<String>,
    object_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone)]
struct PublishStatusUpdate {
    run_id: String,
    status: &'static str,
    published_artifact_count: Option<usize>,
    published_artifacts: Option<JsonValue>,
    outputs: Option<JsonValue>,
    artifacts: Option<JsonValue>,
    output_path: Option<String>,
    output_archive: Option<String>,
    error: Option<String>,
}

impl PublishStatusUpdate {
    fn new(run_id: impl Into<String>, status: &'static str) -> Self {
        Self {
            run_id: run_id.into(),
            status,
            published_artifact_count: None,
            published_artifacts: None,
            outputs: None,
            artifacts: None,
            output_path: None,
            output_archive: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PublishedArtifactRecord {
    provider: String,
    source_artifact: String,
    destination_uri: String,
    manifest_uri: Option<String>,
    object_count: usize,
    total_bytes: u64,
    filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PublishClaim {
    Acquired { owner_token: String },
    AlreadyCompleted(PublishedArtifactRecord),
    InProgress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PublishClaimStatus {
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishClaimState {
    status: PublishClaimStatus,
    fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    owner_token: Option<String>,
    artifact: Option<PublishedArtifactRecord>,
    error: Option<String>,
    updated_at: u64,
}

struct ResultsHandoff<'a> {
    msg: &'a scicomp_rq::Message,
    sink: &'a dyn MessageSink,
    cancellation: &'a RoleCancellation,
    next_queue: &'a str,
    raw_payload: &'a JsonValue,
    workflow_id: &'a str,
    operation: Option<&'a str>,
    status: &'a str,
    result_payload: JsonMap<String, JsonValue>,
}

trait PublishStatusPersistence: Send + Sync + 'static {
    fn persist_status<'a>(&'a self, update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>>;

    fn try_claim_publish<'a>(
        &'a self,
        _run_id: &'a str,
        _target_fingerprint: &'a str,
    ) -> BoxFuture<'a, Result<PublishClaim>> {
        Box::pin(async {
            Ok(PublishClaim::Acquired {
                owner_token: new_publish_owner_token(),
            })
        })
    }

    fn renew_publish_claim<'a>(
        &'a self,
        _run_id: &'a str,
        _target_fingerprint: &'a str,
        _owner_token: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn complete_publish<'a>(
        &'a self,
        _run_id: &'a str,
        _target_fingerprint: &'a str,
        _owner_token: &'a str,
        _artifact: PublishedArtifactRecord,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn fail_publish_claim<'a>(
        &'a self,
        _run_id: &'a str,
        _target_fingerprint: &'a str,
        _owner_token: &'a str,
        _error: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

struct NoopPublishStatusPersistence;

impl PublishStatusPersistence for NoopPublishStatusPersistence {
    fn persist_status<'a>(&'a self, _update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

fn unix_timestamp_secs() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("publish: system clock before unix epoch")?
        .as_secs())
}

fn publish_claim_key(run_id: &str, target_fingerprint: &str) -> String {
    let digest = Sha256::digest(target_fingerprint.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    format!("run:{run_id}:publish:{hex}")
}

fn new_publish_owner_token() -> String {
    Uuid::new_v4().to_string()
}

fn in_progress_claim_state(target_fingerprint: &str, owner_token: &str) -> Result<String> {
    serde_json::to_string(&PublishClaimState {
        status: PublishClaimStatus::InProgress,
        fingerprint: target_fingerprint.to_string(),
        owner_token: Some(owner_token.to_string()),
        artifact: None,
        error: None,
        updated_at: unix_timestamp_secs()?,
    })
    .context("publish: failed to serialize publish claim state")
}

fn in_progress_claim_is_stale(state: &PublishClaimState, now_secs: u64) -> bool {
    state
        .updated_at
        .saturating_add(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
        <= now_secs
}

fn claim_is_owned_in_progress(
    state: &PublishClaimState,
    target_fingerprint: &str,
    owner_token: &str,
) -> bool {
    state.status == PublishClaimStatus::InProgress
        && state.fingerprint == target_fingerprint
        && state.owner_token.as_deref() == Some(owner_token)
}

pub struct RedisPublishStatusPersistence {
    qm: scicomp_rq::QueueManager,
}

impl RedisPublishStatusPersistence {
    pub fn new(qm: scicomp_rq::QueueManager) -> Self {
        Self { qm }
    }
}

async fn renew_publish_claim_until_upload_completes(
    status_persistence: Arc<dyn PublishStatusPersistence>,
    run_id: String,
    target_fingerprint: String,
    owner_token: String,
) -> Result<()> {
    let mut interval =
        tokio::time::interval(Duration::from_secs(PUBLISH_CLAIM_RENEW_INTERVAL_SECS));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        status_persistence
            .renew_publish_claim(
                run_id.as_str(),
                target_fingerprint.as_str(),
                owner_token.as_str(),
            )
            .await?;
    }
}

#[derive(Clone)]
struct UploadCancellation {
    cancelled: watch::Sender<bool>,
}

impl UploadCancellation {
    fn new() -> Self {
        let (cancelled, _) = watch::channel(false);
        Self { cancelled }
    }

    fn cancel(&self) {
        self.cancelled.send_replace(true);
    }

    fn check(&self) -> Result<()> {
        if *self.cancelled.borrow() {
            return Err(anyhow!("publish: upload cancelled"));
        }
        Ok(())
    }

    async fn cancelled(&self) {
        let mut cancelled = self.cancelled.subscribe();
        if *cancelled.borrow() {
            return;
        }
        while cancelled.changed().await.is_ok() {
            if *cancelled.borrow() {
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn upload_selected_artifact_with_claim_renewal(
    status_persistence: Arc<dyn PublishStatusPersistence>,
    run_id: &str,
    target_fingerprint: &str,
    owner_token: &str,
    selected: &SelectedArtifact,
    target: &PublicationTarget,
    config: &PublishRoleConfig,
    ownership_cancellation: &RoleCancellation,
) -> Result<UploadStats> {
    ownership_cancellation.check_ownership()?;
    let cancellation = UploadCancellation::new();
    let renew_claim = renew_publish_claim_until_upload_completes(
        Arc::clone(&status_persistence),
        run_id.to_string(),
        target_fingerprint.to_string(),
        owner_token.to_string(),
    );
    tokio::pin!(renew_claim);
    let upload_future =
        upload_selected_artifact(selected, target, true, true, config, &cancellation);
    tokio::pin!(upload_future);

    let result = tokio::select! {
        biased;
        _ = ownership_cancellation.cancelled() => {
            cancellation.cancel();
            let _ = upload_future.await;
            Err(message_ownership_lost(
                "publish: message ownership was lost during artifact upload",
            ))
        },
        upload = &mut upload_future => upload,
        renewal = &mut renew_claim => {
            let error = match renewal {
                Ok(()) => anyhow!("publish: publish claim renewal stopped unexpectedly"),
                Err(error) => error.context("publish: publish claim renewal failed"),
            };
            cancellation.cancel();
            let _ = upload_future.await;
            Err(error)
        },
    };
    if let Err(error) = result {
        status_persistence
            .fail_publish_claim(run_id, target_fingerprint, owner_token, &error.to_string())
            .await?;
        if ownership_cancellation.is_cancelled() && !is_message_ownership_lost_error(&error) {
            return Err(message_ownership_lost(
                "publish: message ownership was lost while stopping artifact upload",
            ));
        }
        return Err(error);
    }
    result
}

impl PublishStatusPersistence for RedisPublishStatusPersistence {
    fn persist_status<'a>(&'a self, update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let now_secs = unix_timestamp_secs()?.to_string();
            let run_key = format!("run:{}", update.run_id);
            let mut conn = self.qm.connection();
            let mut hset = redis::cmd("HSET");
            hset.arg(&run_key)
                .arg("stage")
                .arg("publish")
                .arg("updated_at")
                .arg(&now_secs)
                .arg("output_publication_status")
                .arg(update.status);
            if update.status == "uploaded" {
                hset.arg("output_location").arg("local_and_cloud");
            }
            match update.status {
                "uploading" | "uploaded" => {
                    hset.arg("status").arg("running");
                }
                "failed" => {
                    hset.arg("status").arg("failed");
                }
                "skipped" => {}
                _ => {}
            }
            match update.status {
                "uploading" => {
                    hset.arg("publish_started_at").arg(&now_secs);
                }
                "uploaded" | "skipped" => {
                    hset.arg("publish_completed_at").arg(&now_secs);
                    if let Some(count) = update.published_artifact_count {
                        hset.arg("published_artifact_count").arg(count.to_string());
                    }
                    if let Some(published_artifacts) = &update.published_artifacts {
                        hset.arg("published_artifacts").arg(
                            serde_json::to_string(published_artifacts)
                                .context("publish: failed to encode published artifacts")?,
                        );
                    }
                    if let Some(outputs) = &update.outputs {
                        hset.arg("outputs").arg(
                            serde_json::to_string(outputs)
                                .context("publish: failed to encode output metadata")?,
                        );
                    }
                    if let Some(artifacts) = &update.artifacts {
                        hset.arg("artifacts").arg(
                            serde_json::to_string(artifacts)
                                .context("publish: failed to encode artifact metadata")?,
                        );
                    }
                    if let Some(output_path) = &update.output_path {
                        hset.arg("output_path").arg(output_path);
                    }
                    if let Some(output_archive) = &update.output_archive {
                        hset.arg("output_archive").arg(output_archive);
                    }
                }
                "failed" => {
                    if let Some(error) = &update.error {
                        hset.arg("publish_error").arg(error);
                    }
                }
                _ => {}
            }
            let _: usize = hset
                .query_async(&mut conn)
                .await
                .context("publish: failed to persist publication status")?;
            if matches!(update.status, "uploaded" | "skipped") {
                let _: i64 = redis::cmd("HDEL")
                    .arg(&run_key)
                    .arg("publish_error")
                    .query_async(&mut conn)
                    .await
                    .context("publish: failed to clear stale publish error")?;
            }
            Ok(())
        })
    }

    fn try_claim_publish<'a>(
        &'a self,
        run_id: &'a str,
        target_fingerprint: &'a str,
    ) -> BoxFuture<'a, Result<PublishClaim>> {
        Box::pin(async move {
            let key = publish_claim_key(run_id, target_fingerprint);
            let owner_token = new_publish_owner_token();
            let in_progress = in_progress_claim_state(target_fingerprint, &owner_token)?;
            let mut conn = self.qm.connection();
            let created: Option<String> = redis::cmd("SET")
                .arg(&key)
                .arg(&in_progress)
                .arg("NX")
                .arg("EX")
                .arg(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
                .query_async(&mut conn)
                .await
                .context("publish: failed to claim publish target")?;
            if created.is_some() {
                return Ok(PublishClaim::Acquired { owner_token });
            }

            let raw: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .context("publish: failed to load publish claim")?;
            let Some(raw) = raw else {
                let created: Option<String> = redis::cmd("SET")
                    .arg(&key)
                    .arg(&in_progress)
                    .arg("NX")
                    .arg("EX")
                    .arg(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
                    .query_async(&mut conn)
                    .await
                    .context("publish: failed to claim expired publish target")?;
                return Ok(if created.is_some() {
                    PublishClaim::Acquired { owner_token }
                } else {
                    PublishClaim::InProgress
                });
            };
            let state: PublishClaimState =
                serde_json::from_str(&raw).context("publish: invalid publish claim state")?;
            match state.status {
                PublishClaimStatus::Completed => state
                    .artifact
                    .map(PublishClaim::AlreadyCompleted)
                    .ok_or_else(|| anyhow!("publish: completed claim missing artifact record")),
                PublishClaimStatus::InProgress => {
                    if !in_progress_claim_is_stale(&state, unix_timestamp_secs()?) {
                        return Ok(PublishClaim::InProgress);
                    }
                    let replaced: bool = redis::Script::new(REPLACE_PUBLISH_CLAIM_IF_UNCHANGED_LUA)
                        .key(&key)
                        .arg(&raw)
                        .arg(&in_progress)
                        .arg(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
                        .invoke_async(&mut conn)
                        .await
                        .context("publish: failed to reacquire stale publish claim")?;
                    Ok(if replaced {
                        PublishClaim::Acquired { owner_token }
                    } else {
                        PublishClaim::InProgress
                    })
                }
                PublishClaimStatus::Failed => {
                    let replaced: bool = redis::Script::new(REPLACE_PUBLISH_CLAIM_IF_UNCHANGED_LUA)
                        .key(&key)
                        .arg(&raw)
                        .arg(&in_progress)
                        .arg(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
                        .invoke_async(&mut conn)
                        .await
                        .context("publish: failed to reacquire failed publish claim")?;
                    Ok(if replaced {
                        PublishClaim::Acquired { owner_token }
                    } else {
                        PublishClaim::InProgress
                    })
                }
            }
        })
    }

    fn renew_publish_claim<'a>(
        &'a self,
        run_id: &'a str,
        target_fingerprint: &'a str,
        owner_token: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let key = publish_claim_key(run_id, target_fingerprint);
            let mut conn = self.qm.connection();
            let raw: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .context("publish: failed to load publish claim for renewal")?;
            let Some(raw) = raw else {
                return Err(anyhow!("publish: publish claim disappeared before renewal"));
            };
            let mut state: PublishClaimState =
                serde_json::from_str(&raw).context("publish: invalid publish claim state")?;
            if !claim_is_owned_in_progress(&state, target_fingerprint, owner_token) {
                return Err(anyhow!(
                    "publish: publish claim is no longer owned by this worker"
                ));
            }
            state.updated_at = unix_timestamp_secs()?;
            let encoded = serde_json::to_string(&state)
                .context("publish: failed to serialize renewed publish claim")?;
            let renewed: bool = redis::Script::new(REPLACE_PUBLISH_CLAIM_IF_UNCHANGED_LUA)
                .key(&key)
                .arg(&raw)
                .arg(&encoded)
                .arg(PUBLISH_CLAIM_IN_PROGRESS_TTL_SECS)
                .invoke_async(&mut conn)
                .await
                .context("publish: failed to renew publish claim")?;
            if !renewed {
                return Err(anyhow!("publish: publish claim changed before renewal"));
            }
            Ok(())
        })
    }

    fn complete_publish<'a>(
        &'a self,
        run_id: &'a str,
        target_fingerprint: &'a str,
        owner_token: &'a str,
        artifact: PublishedArtifactRecord,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let key = publish_claim_key(run_id, target_fingerprint);
            let mut conn = self.qm.connection();
            let raw: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .context("publish: failed to load publish claim for completion")?;
            let Some(raw) = raw else {
                return Err(anyhow!(
                    "publish: publish claim disappeared before completion"
                ));
            };
            let current: PublishClaimState =
                serde_json::from_str(&raw).context("publish: invalid publish claim state")?;
            if !claim_is_owned_in_progress(&current, target_fingerprint, owner_token) {
                return Err(anyhow!(
                    "publish: publish claim is no longer owned by this worker"
                ));
            }
            let state = PublishClaimState {
                status: PublishClaimStatus::Completed,
                fingerprint: target_fingerprint.to_string(),
                owner_token: None,
                artifact: Some(artifact),
                error: None,
                updated_at: unix_timestamp_secs()?,
            };
            let encoded = serde_json::to_string(&state)
                .context("publish: failed to serialize completed publish claim")?;
            let completed: bool = redis::Script::new(REPLACE_PUBLISH_CLAIM_VALUE_IF_UNCHANGED_LUA)
                .key(&key)
                .arg(&raw)
                .arg(&encoded)
                .arg(PUBLISH_CLAIM_TERMINAL_TTL_SECS)
                .invoke_async(&mut conn)
                .await
                .context("publish: failed to complete publish claim")?;
            if !completed {
                return Err(anyhow!("publish: publish claim changed before completion"));
            }
            Ok(())
        })
    }

    fn fail_publish_claim<'a>(
        &'a self,
        run_id: &'a str,
        target_fingerprint: &'a str,
        owner_token: &'a str,
        error: &'a str,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let key = publish_claim_key(run_id, target_fingerprint);
            let mut conn = self.qm.connection();
            let raw: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .context("publish: failed to load publish claim for failure")?;
            let Some(raw) = raw else {
                return Err(anyhow!("publish: publish claim disappeared before failure"));
            };
            let current: PublishClaimState =
                serde_json::from_str(&raw).context("publish: invalid publish claim state")?;
            if !claim_is_owned_in_progress(&current, target_fingerprint, owner_token) {
                return Err(anyhow!(
                    "publish: publish claim is no longer owned by this worker"
                ));
            }
            let state = PublishClaimState {
                status: PublishClaimStatus::Failed,
                fingerprint: target_fingerprint.to_string(),
                owner_token: None,
                artifact: None,
                error: Some(error.to_string()),
                updated_at: unix_timestamp_secs()?,
            };
            let encoded = serde_json::to_string(&state)
                .context("publish: failed to serialize failed claim")?;
            let failed: bool = redis::Script::new(REPLACE_PUBLISH_CLAIM_VALUE_IF_UNCHANGED_LUA)
                .key(&key)
                .arg(&raw)
                .arg(&encoded)
                .arg(PUBLISH_CLAIM_TERMINAL_TTL_SECS)
                .invoke_async(&mut conn)
                .await
                .context("publish: failed to persist failed publish claim")?;
            if !failed {
                return Err(anyhow!("publish: publish claim changed before failure"));
            }
            Ok(())
        })
    }
}

pub struct PublishRole {
    input_streams: Vec<String>,
    status_persistence: Arc<dyn PublishStatusPersistence>,
    config: PublishRoleConfig,
}

impl PublishRole {
    pub fn from_env(env: &RoleEnv) -> Result<Self> {
        Self::from_env_with_status_persistence(env, Arc::new(NoopPublishStatusPersistence))
    }

    pub fn from_env_with_queue_manager(
        env: &RoleEnv,
        qm: scicomp_rq::QueueManager,
    ) -> Result<Self> {
        Self::from_env_with_status_persistence(
            env,
            Arc::new(RedisPublishStatusPersistence::new(qm)),
        )
    }

    fn from_env_with_status_persistence(
        env: &RoleEnv,
        status_persistence: Arc<dyn PublishStatusPersistence>,
    ) -> Result<Self> {
        let config = resolve_publish_role_config(env.role_config.as_ref())?;
        validate_publish_role_config(&config)?;
        Ok(Self {
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
            status_persistence,
            config,
        })
    }

    fn validate_input_stream(&self, stream: &str) -> Result<()> {
        if self.input_streams.iter().any(|allowed| allowed == stream) {
            return Ok(());
        }
        Err(anyhow!(
            "publish: unexpected stream '{stream}' (expected one of: {})",
            self.input_streams.join(", ")
        ))
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
        cancellation: &RoleCancellation,
    ) -> Result<()> {
        cancellation.check_ownership()?;
        let (typed, raw_payload) = decode_publish_payload(msg.payload())?;
        let next_stage = typed.stage_context.next_stage("publish")?;
        if next_stage.phase != "results" {
            return Err(anyhow!(
                "publish: next stage must be 'results', got '{}'",
                next_stage.phase
            ));
        }

        let mut result_payload = typed
            .result
            .as_object()
            .cloned()
            .ok_or_else(|| anyhow!("publish: result payload must be a JSON object"))?;
        let mut status = normalize_status(
            result_payload
                .get("status")
                .and_then(JsonValue::as_str)
                .unwrap_or("succeeded"),
        );

        let mut published_count = 0usize;
        if status == "succeeded"
            && let Some(publication) = typed.output_publication.as_ref()
        {
            self.status_persistence
                .persist_status(PublishStatusUpdate {
                    ..PublishStatusUpdate::new(msg.run_id(), "uploading")
                })
                .await?;
            cancellation.check_ownership()?;

            let target = &publication.target;
            let Some(selected) = select_result_artifact(&result_payload, &target.artifact)? else {
                let error = format!(
                    "required artifact '{}' for publication target is not present in result payload",
                    target.artifact
                );
                self.persist_publish_failure(msg.run_id(), &error).await?;
                record_publication_failure(&mut result_payload, target, None, &error)?;
                status = "failed".to_string();
                return self
                    .handoff_to_results(ResultsHandoff {
                        msg,
                        sink,
                        cancellation,
                        next_queue: next_stage.queue.as_str(),
                        raw_payload: &raw_payload,
                        workflow_id: typed.workflow_id.as_str(),
                        operation: typed.operation.as_deref(),
                        status: status.as_str(),
                        result_payload,
                    })
                    .await;
            };
            let publication_target = match build_publication_target(target, &self.config) {
                Ok(target) => target,
                Err(error) => {
                    let error = error.to_string();
                    self.persist_publish_failure(msg.run_id(), &error).await?;
                    record_publication_failure(
                        &mut result_payload,
                        target,
                        Some(&selected),
                        &error,
                    )?;
                    status = "failed".to_string();
                    return self
                        .handoff_to_results(ResultsHandoff {
                            msg,
                            sink,
                            cancellation,
                            next_queue: next_stage.queue.as_str(),
                            raw_payload: &raw_payload,
                            workflow_id: typed.workflow_id.as_str(),
                            operation: typed.operation.as_deref(),
                            status: status.as_str(),
                            result_payload,
                        })
                        .await;
                }
            };
            let target_fingerprint = publish_target_fingerprint(target, &selected);
            match self
                .status_persistence
                .try_claim_publish(msg.run_id(), &target_fingerprint)
                .await?
            {
                PublishClaim::AlreadyCompleted(record) => {
                    published_count += 1;
                    append_json_array_entry(
                        &mut result_payload,
                        "published_artifacts",
                        published_artifact_json(&record),
                    )?;
                }
                PublishClaim::InProgress => {
                    return Err(message_deferred(format!(
                        "publish: target '{}' is already being published for run '{}'",
                        target.artifact,
                        msg.run_id()
                    )));
                }
                PublishClaim::Acquired { owner_token } => {
                    let upload_started = Instant::now();
                    let upload_result = upload_selected_artifact_with_claim_renewal(
                        Arc::clone(&self.status_persistence),
                        msg.run_id(),
                        &target_fingerprint,
                        &owner_token,
                        &selected,
                        &publication_target,
                        &self.config,
                        cancellation,
                    )
                    .await;
                    let upload = match upload_result {
                        Ok(upload) => upload,
                        Err(error) => {
                            let ownership_lost = is_message_ownership_lost_error(&error);
                            let error_text = error.to_string();
                            if ownership_lost {
                                return Err(error);
                            }
                            if cancellation.is_cancelled() {
                                return Err(message_ownership_lost(
                                    "publish: message ownership was lost after artifact upload failed",
                                ));
                            }
                            self.persist_publish_failure(msg.run_id(), &error_text)
                                .await?;
                            record_publication_failure(
                                &mut result_payload,
                                target,
                                Some(&selected),
                                &error_text,
                            )?;
                            status = "failed".to_string();
                            return self
                                .handoff_to_results(ResultsHandoff {
                                    msg,
                                    sink,
                                    cancellation,
                                    next_queue: next_stage.queue.as_str(),
                                    raw_payload: &raw_payload,
                                    workflow_id: typed.workflow_id.as_str(),
                                    operation: typed.operation.as_deref(),
                                    status: status.as_str(),
                                    result_payload,
                                })
                                .await;
                        }
                    };
                    let upload_elapsed = upload_started.elapsed();
                    info!(
                        run_id = msg.run_id(),
                        source_artifact = selected.name.as_str(),
                        destination_uri = upload.destination_uri.as_str(),
                        object_count = upload.object_count,
                        total_bytes = upload.total_bytes,
                        elapsed_ms = upload_elapsed.as_millis() as u64,
                        throughput_mib_s =
                            throughput_mib_per_sec(upload.total_bytes, upload_elapsed),
                        max_concurrent_files = self.config.max_concurrent_files,
                        "publish: artifact upload completed"
                    );
                    let record = PublishedArtifactRecord {
                        provider: provider_name(target.provider).to_string(),
                        source_artifact: selected.name.clone(),
                        destination_uri: upload.destination_uri,
                        manifest_uri: upload.manifest_uri,
                        object_count: upload.object_count,
                        total_bytes: upload.total_bytes,
                        filename: selected.filename.clone(),
                    };
                    if let Err(error) = cancellation.check_ownership() {
                        self.status_persistence
                            .fail_publish_claim(
                                msg.run_id(),
                                &target_fingerprint,
                                &owner_token,
                                &error.to_string(),
                            )
                            .await?;
                        return Err(error);
                    }
                    let complete_publish = self.status_persistence.complete_publish(
                        msg.run_id(),
                        &target_fingerprint,
                        &owner_token,
                        record.clone(),
                    );
                    tokio::pin!(complete_publish);
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => {
                            let error = message_ownership_lost(
                                "publish: message ownership was lost before claim completion",
                            );
                            self.status_persistence
                                .fail_publish_claim(
                                    msg.run_id(),
                                    &target_fingerprint,
                                    &owner_token,
                                    &error.to_string(),
                                )
                                .await?;
                            return Err(error);
                        },
                        result = &mut complete_publish => result?,
                    }
                    published_count += 1;
                    append_json_array_entry(
                        &mut result_payload,
                        "published_artifacts",
                        published_artifact_json(&record),
                    )?;
                }
            }

            self.status_persistence
                .persist_status(PublishStatusUpdate {
                    published_artifact_count: Some(published_count),
                    published_artifacts: result_payload_execution_field(
                        &result_payload,
                        "published_artifacts",
                    )
                    .cloned(),
                    outputs: result_payload_execution_field(&result_payload, "outputs").cloned(),
                    artifacts: result_payload_execution_field(&result_payload, "artifacts")
                        .cloned(),
                    output_path: result_payload_string_field(&result_payload, "output_path"),
                    output_archive: result_payload_string_field(&result_payload, "output_archive"),
                    ..PublishStatusUpdate::new(msg.run_id(), "uploaded")
                })
                .await?;
        } else if typed.output_publication.is_some() {
            self.status_persistence
                .persist_status(PublishStatusUpdate {
                    published_artifact_count: Some(0),
                    ..PublishStatusUpdate::new(msg.run_id(), "skipped")
                })
                .await?;
        }

        self.handoff_to_results(ResultsHandoff {
            msg,
            sink,
            cancellation,
            next_queue: next_stage.queue.as_str(),
            raw_payload: &raw_payload,
            workflow_id: typed.workflow_id.as_str(),
            operation: typed.operation.as_deref(),
            status: status.as_str(),
            result_payload,
        })
        .await
    }

    async fn handoff_to_results(&self, handoff: ResultsHandoff<'_>) -> Result<()> {
        handoff.cancellation.check_ownership()?;
        let completed_at = Utc::now().to_rfc3339();
        let (execution, payload) = build_execution_and_payload(
            handoff.msg.run_id(),
            handoff.workflow_id,
            handoff.status,
            completed_at.as_str(),
            JsonValue::Object(handoff.result_payload),
        )?;
        let results_envelope = json!({
            "run_id": handoff.msg.run_id(),
            "status": handoff.status,
            "workflow": handoff.workflow_id,
            "completed_at": completed_at,
            "request": build_request_envelope(handoff.raw_payload, handoff.operation),
            "execution": execution,
            "payload": payload,
        });
        let encoded =
            serde_json::to_string(&results_envelope).context("publish: encode results payload")?;

        tokio::select! {
            biased;
            _ = handoff.cancellation.cancelled() => {
                return Err(message_ownership_lost(
                    "publish: message ownership was lost before results handoff",
                ));
            },
            result = handoff
                .sink
                .handoff(handoff.msg, handoff.next_queue, &encoded, "results") => {
                    result.context("publish: failed to hand off to results")?;
                },
        }
        Ok(())
    }

    async fn persist_publish_failure(&self, run_id: &str, error: &str) -> Result<()> {
        self.status_persistence
            .persist_status(PublishStatusUpdate {
                error: Some(error.to_string()),
                ..PublishStatusUpdate::new(run_id, "failed")
            })
            .await
    }
}

fn resolve_publish_role_config(raw: Option<&JsonValue>) -> Result<PublishRoleConfig> {
    if let Ok(override_json) = std::env::var(ENV_PUBLISH_ROLE_CONFIG_JSON)
        && !override_json.trim().is_empty()
    {
        return serde_json::from_str(&override_json)
            .with_context(|| format!("publish: failed to parse {ENV_PUBLISH_ROLE_CONFIG_JSON}"));
    }
    parse_role_config(raw)
}

fn validate_publish_role_config(config: &PublishRoleConfig) -> Result<()> {
    if config.max_concurrent_files == 0 {
        return Err(anyhow!(
            "publish role config max_concurrent_files must be greater than zero"
        ));
    }
    if config.multipart_threshold_bytes == 0 {
        return Err(anyhow!(
            "publish role config multipart_threshold_bytes must be greater than zero"
        ));
    }
    if config.multipart_part_size_bytes < MIN_MULTIPART_PART_SIZE_BYTES {
        return Err(anyhow!(
            "publish role config multipart_part_size_bytes must be at least {MIN_MULTIPART_PART_SIZE_BYTES}"
        ));
    }
    if config.multipart_max_concurrency == 0 {
        return Err(anyhow!(
            "publish role config multipart_max_concurrency must be greater than zero"
        ));
    }
    if matches!(config.client_options.timeout_secs, Some(0)) {
        return Err(anyhow!(
            "publish role config client_options.timeout_secs must be greater than zero"
        ));
    }
    if matches!(config.client_options.connect_timeout_secs, Some(0)) {
        return Err(anyhow!(
            "publish role config client_options.connect_timeout_secs must be greater than zero"
        ));
    }
    if matches!(config.client_options.pool_max_idle_per_host, Some(0)) {
        return Err(anyhow!(
            "publish role config client_options.pool_max_idle_per_host must be greater than zero"
        ));
    }
    if matches!(config.retry.max_retries, Some(0)) {
        return Err(anyhow!(
            "publish role config retry.max_retries must be greater than zero"
        ));
    }
    if matches!(config.retry.timeout_secs, Some(0)) {
        return Err(anyhow!(
            "publish role config retry.timeout_secs must be greater than zero"
        ));
    }
    Ok(())
}

fn decode_publish_payload(raw: &str) -> Result<(PublishEnvelope, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("publish: empty payload"));
    }

    let value: JsonValue =
        serde_json::from_str(raw).context("publish: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("publish: payload must be a JSON object"));
    }

    let typed: PublishEnvelope =
        serde_json::from_value(value.clone()).context("publish: invalid payload schema")?;
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "publish: workflow_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "publish" {
        return Err(anyhow!(
            "publish: payload current_phase must be 'publish', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    if !typed.result.is_object() {
        return Err(anyhow!("publish: payload result must be a JSON object"));
    }

    Ok((typed, value))
}

fn provider_name(provider: OutputPublicationProvider) -> &'static str {
    match provider {
        OutputPublicationProvider::S3 => "s3",
        OutputPublicationProvider::Azure => "azure",
    }
}

fn publish_target_fingerprint(
    target: &ResolvedOutputPublicationTarget,
    selected: &SelectedArtifact,
) -> String {
    match &target.storage {
        ResolvedOutputPublicationStorage::S3 { bucket, prefix, .. } => format!(
            "provider={};bucket={};prefix={};artifact={};filename={}",
            provider_name(target.provider),
            bucket,
            prefix,
            selected.name,
            selected.filename
        ),
        ResolvedOutputPublicationStorage::Azure {
            container, prefix, ..
        } => format!(
            "provider={};container={};prefix={};artifact={};filename={}",
            provider_name(target.provider),
            container,
            prefix,
            selected.name,
            selected.filename
        ),
    }
}

fn published_artifact_json(record: &PublishedArtifactRecord) -> JsonValue {
    json!({
        "kind": "object_store_publish",
        "name": PRIMARY_TARGET_NAME,
        "source_artifact": record.source_artifact.as_str(),
        "provider": record.provider.as_str(),
        "destination_uri": record.destination_uri.as_str(),
        "manifest_uri": record.manifest_uri.as_deref(),
        "object_count": record.object_count,
        "total_bytes": record.total_bytes,
        "overwrite": true,
        "filename": record.filename.as_str(),
        "status": "uploaded",
    })
}

fn result_payload_execution_field<'a>(
    result_payload: &'a JsonMap<String, JsonValue>,
    key: &str,
) -> Option<&'a JsonValue> {
    result_payload.get(key).or_else(|| {
        result_payload
            .get("execution")
            .and_then(JsonValue::as_object)
            .and_then(|execution| execution.get(key))
    })
}

fn result_payload_string_field(
    result_payload: &JsonMap<String, JsonValue>,
    key: &str,
) -> Option<String> {
    result_payload_execution_field(result_payload, key)
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

fn normalize_status(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "success" | "succeeded" | "completed" => "succeeded".to_string(),
        "fail" | "failed" | "error" => "failed".to_string(),
        other => other.to_string(),
    }
}

fn build_request_envelope(raw_payload: &JsonValue, operation: Option<&str>) -> JsonValue {
    let mut request = raw_payload
        .get("request")
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    if !request.contains_key("operation")
        && let Some(operation) = operation
    {
        request.insert(
            "operation".to_string(),
            JsonValue::String(operation.to_string()),
        );
    }
    if !request.contains_key("parameters")
        && let Some(parameters) = raw_payload.get("parameters")
    {
        request.insert("parameters".to_string(), parameters.clone());
    }
    JsonValue::Object(request)
}

fn move_execution_field(
    payload: &mut JsonMap<String, JsonValue>,
    execution: &mut JsonMap<String, JsonValue>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = payload.remove(source_key) {
        execution.entry(target_key.to_string()).or_insert(value);
    }
}

fn remove_empty_execution_array(execution: &mut JsonMap<String, JsonValue>, key: &str) {
    let should_remove = execution
        .get(key)
        .and_then(JsonValue::as_array)
        .is_some_and(|entries| entries.is_empty());
    if should_remove {
        execution.remove(key);
    }
}

fn move_output_fields(
    payload: &mut JsonMap<String, JsonValue>,
    execution: &mut JsonMap<String, JsonValue>,
) {
    move_execution_field(payload, execution, "outputs", "outputs");
    if let Some(artifacts) = payload.remove("artifacts") {
        execution
            .entry("outputs".to_string())
            .or_insert_with(|| artifacts.clone());
        execution
            .entry("artifacts".to_string())
            .or_insert(artifacts);
    }
    if !execution.contains_key("outputs")
        && let Some(nested_execution) = payload.get("execution").and_then(JsonValue::as_object)
        && let Some(outputs) = nested_execution
            .get("outputs")
            .or_else(|| nested_execution.get("artifacts"))
    {
        execution.insert("outputs".to_string(), outputs.clone());
    }
    if !execution.contains_key("artifacts")
        && let Some(nested_execution) = payload.get("execution").and_then(JsonValue::as_object)
        && let Some(artifacts) = nested_execution.get("artifacts")
    {
        execution.insert("artifacts".to_string(), artifacts.clone());
    }
    if !execution.contains_key("output_path")
        && let Some(nested_execution) = payload.get("execution").and_then(JsonValue::as_object)
        && let Some(output_path) = nested_execution.get("output_path")
    {
        execution.insert("output_path".to_string(), output_path.clone());
    }
}

fn derive_primary_output_path(outputs: Option<&JsonValue>) -> Option<String> {
    let outputs = outputs?.as_array()?;
    let primary = outputs
        .iter()
        .find(|entry| {
            entry
                .get("primary")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| outputs.first())?;
    primary
        .get("storage_path")
        .or_else(|| primary.get("path"))
        .or_else(|| primary.get("output_path"))
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

fn build_execution_and_payload(
    run_id: &str,
    workflow_id: &str,
    status: &str,
    completed_at: &str,
    result_payload: JsonValue,
) -> Result<(JsonValue, JsonValue)> {
    let mut payload = result_payload
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("publish: result payload must be a JSON object"))?;
    let mut execution = JsonMap::new();
    execution.insert("run_id".to_string(), JsonValue::String(run_id.to_string()));
    execution.insert("status".to_string(), JsonValue::String(status.to_string()));
    execution.insert(
        "workflow".to_string(),
        JsonValue::String(workflow_id.to_string()),
    );
    execution.insert(
        "completed_at".to_string(),
        JsonValue::String(completed_at.to_string()),
    );
    move_output_fields(&mut payload, &mut execution);
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
    move_execution_field(&mut payload, &mut execution, "output_path", "output_path");
    move_execution_field(
        &mut payload,
        &mut execution,
        "output_archive",
        "output_archive",
    );
    move_execution_field(&mut payload, &mut execution, "error", "error");
    move_execution_field(
        &mut payload,
        &mut execution,
        "execution_time_seconds",
        "execution_time_seconds",
    );
    remove_empty_execution_array(&mut execution, "outputs");
    remove_empty_execution_array(&mut execution, "artifacts");
    remove_empty_execution_array(&mut execution, "published_outputs");
    remove_empty_execution_array(&mut execution, "published_artifacts");
    if !execution.contains_key("output_path")
        && let Some(path) = derive_primary_output_path(execution.get("outputs"))
    {
        execution.insert("output_path".to_string(), JsonValue::String(path));
    }
    Ok((JsonValue::Object(execution), JsonValue::Object(payload)))
}

fn result_artifact_groups(
    result_payload: &JsonMap<String, JsonValue>,
) -> [Option<&Vec<JsonValue>>; 2] {
    [
        result_payload
            .get("outputs")
            .and_then(JsonValue::as_array)
            .filter(|entries| !entries.is_empty()),
        result_payload
            .get("artifacts")
            .and_then(JsonValue::as_array)
            .filter(|entries| !entries.is_empty()),
    ]
}

fn entry_storage_path(entry: &JsonValue) -> Option<&str> {
    entry
        .get("storage_path")
        .or_else(|| entry.get("path"))
        .or_else(|| entry.get("output_path"))
        .and_then(JsonValue::as_str)
}

fn select_primary_result_entry<'a>(
    entry_groups: &[Option<&'a Vec<JsonValue>>],
    output_path: Option<&str>,
) -> Option<&'a JsonValue> {
    for entries in entry_groups.iter().flatten() {
        if let Some(entry) = entries.iter().find(|entry| {
            entry
                .get("primary")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        }) {
            return Some(entry);
        }
    }

    for entries in entry_groups.iter().flatten() {
        if let Some(entry) = entries.iter().find(|entry| {
            entry.get("name").and_then(JsonValue::as_str) == Some(PRIMARY_TARGET_NAME)
        }) {
            return Some(entry);
        }
    }

    if let Some(output_path) = output_path {
        for entries in entry_groups.iter().flatten() {
            if let Some(entry) = entries
                .iter()
                .find(|entry| entry_storage_path(entry) == Some(output_path))
            {
                return Some(entry);
            }
        }
    }

    entry_groups
        .iter()
        .flatten()
        .find_map(|entries| entries.first())
}

fn selected_artifact_from_entry(
    entry: &JsonValue,
    output_path: Option<&str>,
) -> Result<SelectedArtifact> {
    let name = entry
        .get("name")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| anyhow!("publish: artifact entry missing name"))?;
    let storage_path = entry_storage_path(entry)
        .or(output_path)
        .ok_or_else(|| anyhow!("publish: artifact '{}' missing storage path", name))?;
    let filename = entry
        .get("filename")
        .or_else(|| entry.get("original_filename"))
        .and_then(JsonValue::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| derive_filename(storage_path, name));
    Ok(SelectedArtifact {
        name: name.to_string(),
        storage_path: PathBuf::from(storage_path),
        filename,
    })
}

fn select_result_artifact(
    result_payload: &JsonMap<String, JsonValue>,
    requested: &str,
) -> Result<Option<SelectedArtifact>> {
    let output_path = result_payload
        .get("output_path")
        .and_then(JsonValue::as_str);

    let entry_groups = result_artifact_groups(result_payload);

    if requested == PRIMARY_TARGET_NAME
        && let Some(entry) = select_primary_result_entry(&entry_groups, output_path)
    {
        return selected_artifact_from_entry(entry, output_path).map(Some);
    }

    for entries in entry_groups.into_iter().flatten() {
        let entry = entries
            .iter()
            .find(|entry| entry.get("name").and_then(JsonValue::as_str) == Some(requested));
        if let Some(entry) = entry {
            return selected_artifact_from_entry(entry, output_path).map(Some);
        }
    }

    if requested == PRIMARY_TARGET_NAME
        && let Some(path) = output_path
    {
        return Ok(Some(SelectedArtifact {
            name: "primary".to_string(),
            filename: derive_filename(path, "primary"),
            storage_path: PathBuf::from(path),
        }));
    }

    Ok(None)
}

fn derive_filename(storage_path: &str, fallback: &str) -> String {
    Path::new(storage_path)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

fn build_publication_target(
    target: &ResolvedOutputPublicationTarget,
    config: &PublishRoleConfig,
) -> Result<PublicationTarget> {
    match &target.storage {
        ResolvedOutputPublicationStorage::S3 {
            bucket,
            prefix,
            region,
            endpoint,
        } => {
            let bucket = bucket.trim();
            if bucket.is_empty() {
                return Err(anyhow!("publish: s3 bucket is required"));
            }
            let prefix = prefix.trim_matches('/').to_string();
            let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
            builder = apply_s3_client_config(builder, config);
            if let Some(region) = region
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                builder = builder.with_region(region);
            }
            if let Some(endpoint) = endpoint
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                if endpoint
                    .split_once("://")
                    .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("http"))
                {
                    builder = builder.with_allow_http(true);
                }
                builder = builder.with_config(AmazonS3ConfigKey::S3Endpoint, endpoint);
            } else if let Ok(endpoint) = std::env::var("S3_ENDPOINT_URL")
                && !endpoint.trim().is_empty()
            {
                let endpoint = endpoint.trim();
                if endpoint
                    .split_once("://")
                    .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("http"))
                {
                    builder = builder.with_allow_http(true);
                }
                builder = builder.with_config(AmazonS3ConfigKey::Endpoint, endpoint);
            }
            let store =
                Arc::new(builder.build().with_context(|| {
                    format!("publish: failed to build S3 store for '{bucket}'")
                })?);
            Ok(PublicationTarget {
                store,
                prefix: ObjectPath::from(prefix.clone()),
                destination_uri: format!("s3://{bucket}"),
            })
        }
        ResolvedOutputPublicationStorage::Azure {
            container,
            prefix,
            endpoint,
        } => {
            let container = container.trim_matches('/');
            if container.is_empty() {
                return Err(anyhow!("publish: azure container is required"));
            }
            let endpoint = endpoint.trim().trim_end_matches('/');
            if endpoint.is_empty() {
                return Err(anyhow!("publish: azure endpoint is required"));
            }
            let container_url = join_uri_path(endpoint, container);
            let prefix = prefix.trim_matches('/').to_string();
            let mut builder = MicrosoftAzureBuilder::from_env();
            if builder
                .get_config_value(&AzureConfigKey::AccountName)
                .is_none()
                && let Ok(account) = std::env::var("AZURE_STORAGE_ACCOUNT")
            {
                builder = builder.with_account(account);
            }
            let builder = apply_azure_client_config(
                builder
                    .with_endpoint(endpoint.to_string())
                    .with_container_name(container),
                config,
            );
            let store = Arc::new(
                builder
                    .build()
                    .context("publish: failed to build Azure store")?,
            );
            Ok(PublicationTarget {
                store,
                prefix: ObjectPath::from(prefix.clone()),
                destination_uri: container_url,
            })
        }
    }
}

fn apply_s3_client_config(builder: AmazonS3Builder, config: &PublishRoleConfig) -> AmazonS3Builder {
    let mut builder = builder.with_retry(build_object_store_retry_config(config));
    for (key, value) in configured_object_store_client_options(config) {
        builder = builder.with_config(AmazonS3ConfigKey::Client(key), value);
    }
    builder
}

fn apply_azure_client_config(
    builder: MicrosoftAzureBuilder,
    config: &PublishRoleConfig,
) -> MicrosoftAzureBuilder {
    let mut builder = builder.with_retry(build_object_store_retry_config(config));
    for (key, value) in configured_object_store_client_options(config) {
        builder = builder.with_config(AzureConfigKey::Client(key), value);
    }
    builder
}

fn configured_object_store_client_options(
    config: &PublishRoleConfig,
) -> Vec<(ClientConfigKey, String)> {
    let mut options = Vec::with_capacity(3);
    if let Some(timeout_secs) = config.client_options.timeout_secs {
        options.push((ClientConfigKey::Timeout, format!("{timeout_secs}s")));
    }
    if let Some(connect_timeout_secs) = config.client_options.connect_timeout_secs {
        options.push((
            ClientConfigKey::ConnectTimeout,
            format!("{connect_timeout_secs}s"),
        ));
    }
    if let Some(pool_max_idle_per_host) = config.client_options.pool_max_idle_per_host {
        options.push((
            ClientConfigKey::PoolMaxIdlePerHost,
            pool_max_idle_per_host.to_string(),
        ));
    }
    options
}

fn build_object_store_retry_config(config: &PublishRoleConfig) -> RetryConfig {
    let mut retry = RetryConfig::default();
    if let Some(max_retries) = config.retry.max_retries {
        retry.max_retries = max_retries;
    }
    if let Some(timeout_secs) = config.retry.timeout_secs {
        retry.retry_timeout = Duration::from_secs(timeout_secs);
    }
    retry
}

fn join_uri_path(base: impl AsRef<str>, suffix: impl AsRef<str>) -> String {
    let base = base.as_ref().trim_end_matches('/');
    let suffix = suffix.as_ref().trim_matches('/');
    if suffix.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{suffix}")
    }
}

fn object_path_uri(base: impl AsRef<str>, object_path: &ObjectPath) -> String {
    join_uri_path(base, object_path.as_ref())
}

async fn upload_selected_artifact(
    selected: &SelectedArtifact,
    target: &PublicationTarget,
    overwrite: bool,
    publish_manifest: bool,
    config: &PublishRoleConfig,
    cancellation: &UploadCancellation,
) -> Result<UploadStats> {
    cancellation.check()?;
    let metadata = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
        metadata = tokio::fs::symlink_metadata(&selected.storage_path) => metadata,
    }
    .with_context(|| {
        format!(
            "publish: failed to stat local artifact '{}'",
            selected.storage_path.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!(
            "publish: local artifact '{}' must not be a symlink",
            selected.storage_path.display()
        ));
    }
    let artifact_prefix =
        join_object_path(&target.prefix, &ObjectPath::from(selected.filename.clone()));
    let destination_uri = object_path_uri(&target.destination_uri, &artifact_prefix);
    if metadata.is_dir() {
        let scan_started = Instant::now();
        let entries = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
            entries = collect_directory_entries(&selected.storage_path) => entries?,
        };
        let scan_elapsed = scan_started.elapsed();

        let upload_started = Instant::now();
        let mut total_bytes = 0u64;
        let object_count = entries.len();
        let mut uploads =
            futures::stream::iter(entries.into_iter().map(|(relative, full_path)| {
                let store = Arc::clone(&target.store);
                let artifact_prefix = artifact_prefix.clone();
                let config = config.clone();
                let cancellation = cancellation.clone();
                async move {
                    cancellation.check()?;
                    let relative_object = path_to_object_path(&relative)?;
                    let object_path = join_object_path(&artifact_prefix, &relative_object);
                    upload_local_file(
                        store.as_ref(),
                        &object_path,
                        &full_path,
                        overwrite,
                        &config,
                        &cancellation,
                    )
                    .await
                }
            }))
            .buffer_unordered(config.max_concurrent_files);
        let mut first_error = None;
        while let Some(result) = uploads.next().await {
            match result {
                Ok(byte_len) => total_bytes += byte_len,
                Err(error) if first_error.is_none() => {
                    first_error = Some(error);
                    cancellation.cancel();
                }
                Err(_) => {}
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        let upload_elapsed = upload_started.elapsed();
        info!(
            source_artifact = selected.name.as_str(),
            destination_uri = destination_uri.as_str(),
            object_count,
            total_bytes,
            scan_ms = scan_elapsed.as_millis() as u64,
            upload_ms = upload_elapsed.as_millis() as u64,
            throughput_mib_s = throughput_mib_per_sec(total_bytes, upload_elapsed),
            max_concurrent_files = config.max_concurrent_files,
            "publish: directory artifact upload completed"
        );

        let manifest_uri = if publish_manifest {
            Some(
                upload_publish_manifest(
                    target.store.as_ref(),
                    &artifact_prefix,
                    overwrite,
                    &selected.name,
                    (object_count, total_bytes),
                    &destination_uri,
                    cancellation,
                )
                .await?,
            )
        } else {
            None
        };

        return Ok(UploadStats {
            destination_uri,
            manifest_uri,
            object_count,
            total_bytes,
        });
    }

    if !metadata.is_file() {
        return Err(anyhow!(
            "publish: local artifact '{}' is neither a file nor directory",
            selected.storage_path.display()
        ));
    }

    let total_bytes = upload_local_file(
        target.store.as_ref(),
        &artifact_prefix,
        &selected.storage_path,
        overwrite,
        config,
        cancellation,
    )
    .await?;
    Ok(UploadStats {
        destination_uri,
        manifest_uri: None,
        object_count: 1,
        total_bytes,
    })
}

async fn upload_publish_manifest(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
    overwrite: bool,
    source_artifact: &str,
    object_stats: (usize, u64),
    destination_uri: &str,
    cancellation: &UploadCancellation,
) -> Result<String> {
    cancellation.check()?;
    let (object_count, total_bytes) = object_stats;
    let manifest_filename = PUBLISH_MANIFEST_FILENAME.to_string();
    let manifest_path = join_object_path(prefix, &ObjectPath::from(manifest_filename.clone()));
    let manifest_uri = join_uri_path(destination_uri, &manifest_filename);
    let payload = serde_json::to_vec(&json!({
        "version": 1,
        "source_artifact": source_artifact,
        "object_count": object_count,
        "total_bytes": total_bytes,
        "published_at": Utc::now().to_rfc3339(),
    }))
    .context("publish: failed to serialize publish manifest")?;
    put_bytes_object(store, &manifest_path, payload, overwrite, cancellation).await?;
    Ok(manifest_uri)
}

async fn upload_local_file(
    store: &dyn ObjectStore,
    path: &ObjectPath,
    local_path: &Path,
    overwrite: bool,
    config: &PublishRoleConfig,
    cancellation: &UploadCancellation,
) -> Result<u64> {
    cancellation.check()?;
    let metadata = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
        metadata = tokio::fs::symlink_metadata(local_path) => metadata,
    }
    .with_context(|| {
        format!(
            "publish: failed to stat local artifact file '{}'",
            local_path.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!(
            "publish: local artifact entry '{}' must not be a symlink",
            local_path.display()
        ));
    }
    if !metadata.is_file() {
        return Err(anyhow!(
            "publish: local artifact entry '{}' is not a file",
            local_path.display()
        ));
    }
    let byte_len = metadata.len();
    if overwrite && byte_len >= config.multipart_threshold_bytes {
        put_file_multipart(store, path, local_path, config, cancellation).await?;
    } else {
        let bytes = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
            bytes = tokio::fs::read(local_path) => bytes,
        }
        .with_context(|| {
            format!(
                "publish: failed to read local artifact file '{}'",
                local_path.display()
            )
        })?;
        put_bytes_object(store, path, bytes, overwrite, cancellation).await?;
    }
    Ok(byte_len)
}

async fn put_bytes_object(
    store: &dyn ObjectStore,
    path: &ObjectPath,
    bytes: Vec<u8>,
    overwrite: bool,
    cancellation: &UploadCancellation,
) -> Result<()> {
    cancellation.check()?;
    let mode = if overwrite {
        PutMode::Overwrite
    } else {
        PutMode::Create
    };
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
        result = store.put_opts(
            path,
            PutPayload::from(bytes),
            PutOptions {
                mode,
                ..Default::default()
            },
        ) => result,
    }
    .with_context(|| format!("publish: failed to upload object '{}'", path))?;
    Ok(())
}

async fn put_file_multipart(
    store: &dyn ObjectStore,
    path: &ObjectPath,
    local_path: &Path,
    config: &PublishRoleConfig,
    cancellation: &UploadCancellation,
) -> Result<()> {
    cancellation.check()?;
    let mut upload = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(anyhow!("publish: upload cancelled")),
        upload = store.put_multipart_opts(path, Default::default()) => upload,
    }
    .with_context(|| format!("publish: failed to start multipart upload for '{}'", path))?;
    let file = tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            let _ = upload.abort().await;
            return Err(anyhow!("publish: upload cancelled"));
        }
        file = tokio::fs::File::open(local_path) => file,
    };
    let mut file = match file {
        Ok(file) => file,
        Err(error) => {
            let _ = upload.abort().await;
            return Err(error).with_context(|| {
                format!(
                    "publish: failed to open local artifact file '{}'",
                    local_path.display()
                )
            });
        }
    };
    let mut buffer = vec![0u8; config.multipart_part_size_bytes];
    let mut in_flight = FuturesUnordered::new();
    loop {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = tokio::select! {
                biased;
                _ = cancellation.cancelled() => None,
                read = file.read(&mut buffer[bytes_read..]) => Some(read),
            };
            let Some(read) = read else {
                drop(in_flight);
                let _ = upload.abort().await;
                return Err(anyhow!("publish: upload cancelled"));
            };
            match read {
                Ok(0) => break,
                Ok(read) => bytes_read += read,
                Err(error) => {
                    drop(in_flight);
                    let _ = upload.abort().await;
                    return Err(error).with_context(|| {
                        format!(
                            "publish: failed to read local artifact file '{}'",
                            local_path.display()
                        )
                    });
                }
            }
        }
        if bytes_read == 0 {
            break;
        }
        while in_flight.len() >= config.multipart_max_concurrency {
            let next = tokio::select! {
                biased;
                _ = cancellation.cancelled() => None,
                next = in_flight.next() => Some(next),
            };
            let Some(next) = next else {
                drop(in_flight);
                let _ = upload.abort().await;
                return Err(anyhow!("publish: upload cancelled"));
            };
            if let Some(Err(error)) = next {
                drop(in_flight);
                let _ = upload.abort().await;
                return Err(error)
                    .with_context(|| format!("publish: multipart upload failed for '{}'", path));
            }
        }
        in_flight.push(upload.put_part(buffer[..bytes_read].to_vec().into()));
        if bytes_read < buffer.len() {
            break;
        }
    }
    loop {
        let next = tokio::select! {
            biased;
            _ = cancellation.cancelled() => None,
            next = in_flight.next() => Some(next),
        };
        let Some(next) = next else {
            drop(in_flight);
            let _ = upload.abort().await;
            return Err(anyhow!("publish: upload cancelled"));
        };
        let Some(result) = next else {
            break;
        };
        if let Err(error) = result {
            drop(in_flight);
            let _ = upload.abort().await;
            return Err(error)
                .with_context(|| format!("publish: multipart upload failed for '{}'", path));
        }
    }
    let completion = tokio::select! {
        biased;
        _ = cancellation.cancelled() => None,
        completion = upload.complete() => Some(completion),
    };
    let Some(completion) = completion else {
        let _ = upload.abort().await;
        return Err(anyhow!("publish: upload cancelled"));
    };
    if let Err(error) = completion {
        let _ = upload.abort().await;
        return Err(error).with_context(|| {
            format!(
                "publish: failed to complete multipart upload for '{}'",
                path
            )
        });
    }
    Ok(())
}

fn throughput_mib_per_sec(bytes: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0)) / seconds
}

async fn collect_directory_entries(root: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let root_canonical = tokio::fs::canonicalize(root).await.with_context(|| {
        format!(
            "publish: failed to resolve local artifact directory '{}'",
            root.display()
        )
    })?;
    let mut pending = vec![(root.to_path_buf(), vec![root_canonical])];
    let mut files = Vec::new();
    while let Some((dir, ancestors)) = pending.pop() {
        let mut entries = tokio::fs::read_dir(&dir).await.with_context(|| {
            format!(
                "publish: failed to read local artifact directory '{}'",
                dir.display()
            )
        })?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .context("publish: failed to read directory entry")?
        {
            let path = entry.path();
            match classify_directory_entry(&path).await? {
                DirectoryEntryKind::Directory { canonical_path } => {
                    if ancestors.iter().any(|ancestor| ancestor == &canonical_path) {
                        return Err(anyhow!(
                            "publish: recursive directory symlink detected at '{}'",
                            path.display()
                        ));
                    }
                    let mut child_ancestors = ancestors.clone();
                    child_ancestors.push(canonical_path);
                    pending.push((path, child_ancestors));
                }
                DirectoryEntryKind::File => {
                    let relative = path
                        .strip_prefix(root)
                        .with_context(|| {
                            format!(
                                "publish: failed to compute relative artifact path for '{}'",
                                path.display()
                            )
                        })?
                        .to_path_buf();
                    if relative == Path::new(PUBLISH_MANIFEST_FILENAME) {
                        return Err(anyhow!(
                            "publish: directory artifact '{}' contains reserved file '{}'",
                            root.display(),
                            PUBLISH_MANIFEST_FILENAME
                        ));
                    }
                    files.push((relative, path));
                }
            }
        }
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(files)
}

enum DirectoryEntryKind {
    Directory { canonical_path: PathBuf },
    File,
}

async fn classify_directory_entry(path: &Path) -> Result<DirectoryEntryKind> {
    let metadata = tokio::fs::symlink_metadata(path)
        .await
        .with_context(|| format!("publish: failed to inspect local path '{}'", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!(
            "publish: local artifact entry '{}' must not be a symlink",
            path.display()
        ));
    }
    if metadata.is_dir() {
        let canonical_path = tokio::fs::canonicalize(path).await.with_context(|| {
            format!(
                "publish: failed to resolve local artifact directory '{}'",
                path.display()
            )
        })?;
        return Ok(DirectoryEntryKind::Directory { canonical_path });
    }
    if metadata.is_file() {
        return Ok(DirectoryEntryKind::File);
    }
    Err(anyhow!(
        "publish: unsupported artifact entry type '{}'",
        path.display()
    ))
}

fn path_to_object_path(path: &Path) -> Result<ObjectPath> {
    let mut object_path = ObjectPath::default();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().ok_or_else(|| {
                    anyhow!(
                        "publish: path '{}' contains a non-UTF-8 component",
                        path.display()
                    )
                })?;
                object_path = object_path.join(part);
            }
            _ => {
                return Err(anyhow!(
                    "publish: path '{}' contains unsupported relative components",
                    path.display()
                ));
            }
        }
    }
    Ok(object_path)
}

fn join_object_path(base: &ObjectPath, child: &ObjectPath) -> ObjectPath {
    if base.as_ref().is_empty() {
        child.clone()
    } else if child.as_ref().is_empty() {
        base.clone()
    } else {
        child
            .as_ref()
            .split('/')
            .filter(|part| !part.is_empty())
            .fold(base.clone(), |path, part| path.join(part))
    }
}

fn append_json_array_entry(
    map: &mut JsonMap<String, JsonValue>,
    key: &str,
    entry: JsonValue,
) -> Result<()> {
    let value = map
        .entry(key.to_string())
        .or_insert_with(|| JsonValue::Array(Vec::new()));
    let array = value
        .as_array_mut()
        .ok_or_else(|| anyhow!("publish: field '{}' must be an array", key))?;
    array.push(entry);
    Ok(())
}

fn record_publication_failure(
    result_payload: &mut JsonMap<String, JsonValue>,
    target: &ResolvedOutputPublicationTarget,
    selected: Option<&SelectedArtifact>,
    error: &str,
) -> Result<()> {
    result_payload.insert(
        "status".to_string(),
        JsonValue::String("failed".to_string()),
    );
    result_payload.insert("error".to_string(), JsonValue::String(error.to_string()));

    let mut entry = json!({
        "kind": "object_store_publish",
        "name": PRIMARY_TARGET_NAME,
        "source_artifact": selected.map(|artifact| artifact.name.as_str()).unwrap_or(target.artifact.as_str()),
        "provider": provider_name(target.provider),
        "status": "failed",
        "error": error,
    });
    if let Some(selected) = selected
        && let Some(entry) = entry.as_object_mut()
    {
        entry.insert(
            "filename".to_string(),
            JsonValue::String(selected.filename.clone()),
        );
    }
    append_json_array_entry(result_payload, "published_artifacts", entry)
}

impl WorkerRole for PublishRole {
    fn name(&self) -> &'static str {
        "publish"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        let cancellation = RoleCancellation::new();
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            self.process_message(msg, sink, &cancellation).await
        })
    }

    fn handle_with_cancellation<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
        cancellation: RoleCancellation,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.validate_input_stream(stream)?;
            self.process_message(msg, sink, &cancellation).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio::sync::Barrier;

    mod test_support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    use crate::config::{InputStreamSpec, RuntimeConfig};
    use crate::engine::EngineBuilder;
    use crate::test_env;
    use crate::traits::{BoxFuture, MessageSink, QueueTransport, WorkerRole};
    use crate::transport::memory::InMemoryTransport;
    use object_store::ObjectStore;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use object_store::path::Path as ObjectPath;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutResult, UploadPart,
    };
    use serde_json::json;
    use test_support::spawn_test_queue_manager;

    use super::*;

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

    struct RecordingStatusPersistence {
        updates: Mutex<Vec<PublishStatusUpdate>>,
    }

    impl RecordingStatusPersistence {
        fn new() -> Self {
            Self {
                updates: Mutex::new(Vec::new()),
            }
        }

        fn updates(&self) -> Vec<PublishStatusUpdate> {
            self.updates.lock().expect("status lock poisoned").clone()
        }
    }

    impl PublishStatusPersistence for RecordingStatusPersistence {
        fn persist_status<'a>(&'a self, update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.updates
                    .lock()
                    .expect("status lock poisoned")
                    .push(update);
                Ok(())
            })
        }
    }

    struct DeferredThenCompletedPersistence {
        claim_attempts: AtomicUsize,
    }

    impl PublishStatusPersistence for DeferredThenCompletedPersistence {
        fn persist_status<'a>(&'a self, _update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn try_claim_publish<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
        ) -> BoxFuture<'a, Result<PublishClaim>> {
            Box::pin(async move {
                if self.claim_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(PublishClaim::InProgress);
                }
                Ok(PublishClaim::AlreadyCompleted(PublishedArtifactRecord {
                    provider: "s3".to_string(),
                    source_artifact: "primary".to_string(),
                    destination_uri: "s3://bucket/runs/run-claim/dataset.zarr".to_string(),
                    manifest_uri: None,
                    object_count: 1,
                    total_bytes: 42,
                    filename: "dataset.zarr".to_string(),
                }))
            })
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

    #[derive(Debug)]
    struct FinalPartFailingStore {
        aborted: Arc<AtomicBool>,
    }

    impl fmt::Display for FinalPartFailingStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("FinalPartFailingStore")
        }
    }

    #[derive(Debug)]
    struct FinalPartFailingUpload {
        aborted: Arc<AtomicBool>,
    }

    #[async_trait]
    impl MultipartUpload for FinalPartFailingUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            Box::pin(async {
                Err(object_store::Error::Generic {
                    store: "multipart-test",
                    source: Box::new(std::io::Error::other("final part failed")),
                })
            })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            unreachable!("a failed final part must prevent completion")
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            self.aborted.store(true, Ordering::SeqCst);
            Err(object_store::Error::Generic {
                store: "multipart-test",
                source: Box::new(std::io::Error::other("abort failed")),
            })
        }
    }

    #[async_trait]
    impl ObjectStore for FinalPartFailingStore {
        async fn put_opts(
            &self,
            _location: &ObjectPath,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unreachable!("single-part upload is not used by this test")
        }

        async fn put_multipart_opts(
            &self,
            _location: &ObjectPath,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            Ok(Box::new(FinalPartFailingUpload {
                aborted: Arc::clone(&self.aborted),
            }))
        }

        async fn get_opts(
            &self,
            _location: &ObjectPath,
            _options: GetOptions,
        ) -> object_store::Result<GetResult> {
            unreachable!("get is not used by this test")
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
        ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
            unreachable!("delete is not used by this test")
        }

        fn list(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            unreachable!("list is not used by this test")
        }

        async fn list_with_delimiter(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> object_store::Result<ListResult> {
            unreachable!("list is not used by this test")
        }

        async fn copy_opts(
            &self,
            _from: &ObjectPath,
            _to: &ObjectPath,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unreachable!("copy is not used by this test")
        }
    }

    #[derive(Debug)]
    struct BlockingMultipartState {
        upload_and_renewal_started: Barrier,
        aborts: AtomicUsize,
        completes: AtomicUsize,
    }

    #[derive(Debug)]
    struct BlockingMultipartStore {
        state: Arc<BlockingMultipartState>,
        fail_multipart_path: Option<ObjectPath>,
    }

    impl fmt::Display for BlockingMultipartStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("BlockingMultipartStore")
        }
    }

    #[derive(Debug)]
    struct BlockingMultipartUpload {
        state: Arc<BlockingMultipartState>,
    }

    #[async_trait]
    impl MultipartUpload for BlockingMultipartUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            let state = Arc::clone(&self.state);
            Box::pin(async move {
                state.upload_and_renewal_started.wait().await;
                std::future::pending().await
            })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            self.state.completes.fetch_add(1, Ordering::SeqCst);
            Ok(PutResult {
                e_tag: None,
                version: None,
                extensions: Default::default(),
            })
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            self.state.aborts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait]
    impl ObjectStore for BlockingMultipartStore {
        async fn put_opts(
            &self,
            _location: &ObjectPath,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            unreachable!("single-part upload is not used by this test")
        }

        async fn put_multipart_opts(
            &self,
            location: &ObjectPath,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            if self.fail_multipart_path.as_ref() == Some(location) {
                self.state.upload_and_renewal_started.wait().await;
                return Err(object_store::Error::Generic {
                    store: "multipart-test",
                    source: Box::new(std::io::Error::other("sibling upload failed")),
                });
            }
            Ok(Box::new(BlockingMultipartUpload {
                state: Arc::clone(&self.state),
            }))
        }

        async fn get_opts(
            &self,
            _location: &ObjectPath,
            _options: GetOptions,
        ) -> object_store::Result<GetResult> {
            unreachable!("get is not used by this test")
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
        ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
            unreachable!("delete is not used by this test")
        }

        fn list(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            unreachable!("list is not used by this test")
        }

        async fn list_with_delimiter(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> object_store::Result<ListResult> {
            unreachable!("list is not used by this test")
        }

        async fn copy_opts(
            &self,
            _from: &ObjectPath,
            _to: &ObjectPath,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unreachable!("copy is not used by this test")
        }
    }

    struct RenewalFailsDuringMultipartUpload {
        state: Arc<BlockingMultipartState>,
    }

    impl PublishStatusPersistence for RenewalFailsDuringMultipartUpload {
        fn persist_status<'a>(&'a self, _update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn renew_publish_claim<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
            _owner_token: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.state.upload_and_renewal_started.wait().await;
                Err(anyhow!("renewal backend unavailable"))
            })
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EnginePublishClaimState {
        Unclaimed,
        InProgress,
        Failed,
        Completed,
    }

    struct EnginePublishClaimPersistence {
        state: Arc<BlockingMultipartState>,
        claim_state: Mutex<EnginePublishClaimState>,
        failed_claims: AtomicUsize,
        completed_claims: AtomicUsize,
    }

    impl EnginePublishClaimPersistence {
        fn claim_state(&self) -> EnginePublishClaimState {
            *self
                .claim_state
                .lock()
                .expect("publish claim state lock should not be poisoned")
        }
    }

    impl PublishStatusPersistence for EnginePublishClaimPersistence {
        fn persist_status<'a>(&'a self, _update: PublishStatusUpdate) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn try_claim_publish<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
        ) -> BoxFuture<'a, Result<PublishClaim>> {
            Box::pin(async move {
                let mut claim_state = self
                    .claim_state
                    .lock()
                    .expect("publish claim state lock should not be poisoned");
                match *claim_state {
                    EnginePublishClaimState::Unclaimed | EnginePublishClaimState::Failed => {
                        *claim_state = EnginePublishClaimState::InProgress;
                        Ok(PublishClaim::Acquired {
                            owner_token: "engine-owner-token".to_string(),
                        })
                    }
                    EnginePublishClaimState::InProgress => Ok(PublishClaim::InProgress),
                    EnginePublishClaimState::Completed => {
                        Ok(PublishClaim::AlreadyCompleted(PublishedArtifactRecord {
                            provider: "s3".to_string(),
                            source_artifact: "primary".to_string(),
                            destination_uri: "s3://bucket/artifact.bin".to_string(),
                            manifest_uri: None,
                            object_count: 1,
                            total_bytes: 9,
                            filename: "artifact.bin".to_string(),
                        }))
                    }
                }
            })
        }

        fn renew_publish_claim<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
            _owner_token: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.state.upload_and_renewal_started.wait().await;
                Ok(())
            })
        }

        fn complete_publish<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
            _owner_token: &'a str,
            _artifact: PublishedArtifactRecord,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                *self
                    .claim_state
                    .lock()
                    .expect("publish claim state lock should not be poisoned") =
                    EnginePublishClaimState::Completed;
                self.completed_claims.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }

        fn fail_publish_claim<'a>(
            &'a self,
            _run_id: &'a str,
            _target_fingerprint: &'a str,
            _owner_token: &'a str,
            _error: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let mut claim_state = self
                    .claim_state
                    .lock()
                    .expect("publish claim state lock should not be poisoned");
                if *claim_state != EnginePublishClaimState::InProgress {
                    return Err(anyhow!("publish claim is not owned by this worker"));
                }
                *claim_state = EnginePublishClaimState::Failed;
                self.failed_claims.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    struct EngineMultipartPublishRole {
        persistence: Arc<EnginePublishClaimPersistence>,
        selected: SelectedArtifact,
        target: PublicationTarget,
        config: PublishRoleConfig,
    }

    impl EngineMultipartPublishRole {
        fn run<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            sink: &'a dyn MessageSink,
            cancellation: RoleCancellation,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let target_fingerprint = "s3:bucket:runs/run-engine:primary:artifact.bin";
                let claim = self
                    .persistence
                    .try_claim_publish(msg.run_id(), target_fingerprint)
                    .await?;
                let PublishClaim::Acquired { owner_token } = claim else {
                    return Err(message_deferred("publish target is already claimed"));
                };
                let persistence: Arc<dyn PublishStatusPersistence> = self.persistence.clone();
                match upload_selected_artifact_with_claim_renewal(
                    persistence,
                    msg.run_id(),
                    target_fingerprint,
                    &owner_token,
                    &self.selected,
                    &self.target,
                    &self.config,
                    &cancellation,
                )
                .await
                {
                    Ok(upload) => {
                        self.persistence
                            .complete_publish(
                                msg.run_id(),
                                target_fingerprint,
                                &owner_token,
                                PublishedArtifactRecord {
                                    provider: "s3".to_string(),
                                    source_artifact: "primary".to_string(),
                                    destination_uri: upload.destination_uri,
                                    manifest_uri: upload.manifest_uri,
                                    object_count: upload.object_count,
                                    total_bytes: upload.total_bytes,
                                    filename: "artifact.bin".to_string(),
                                },
                            )
                            .await?;
                        sink.handoff(msg, "results", msg.payload(), "results")
                            .await?;
                        Ok(())
                    }
                    Err(error) => Err(error),
                }
            })
        }
    }

    impl WorkerRole for EngineMultipartPublishRole {
        fn name(&self) -> &'static str {
            "engine-multipart-publish"
        }

        fn handle<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            sink: &'a dyn MessageSink,
        ) -> BoxFuture<'a, Result<()>> {
            self.run(msg, sink, RoleCancellation::new())
        }

        fn handle_with_cancellation<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            sink: &'a dyn MessageSink,
            cancellation: RoleCancellation,
        ) -> BoxFuture<'a, Result<()>> {
            self.run(msg, sink, cancellation)
        }
    }

    struct EngineOwnershipLossTransport {
        inner: InMemoryTransport,
        state: Arc<BlockingMultipartState>,
        acked: AtomicUsize,
        handoffs: AtomicUsize,
        failure_attempts: AtomicUsize,
    }

    impl MessageSink for EngineOwnershipLossTransport {
        fn enqueue<'a>(
            &'a self,
            stream: &'a str,
            run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            self.inner.enqueue(stream, run_id, payload, stage)
        }

        fn ack_message<'a>(&'a self, msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            self.inner.ack_message(msg)
        }

        fn handoff<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.handoffs.fetch_add(1, Ordering::SeqCst);
                self.inner.handoff(msg, dest_stream, payload, stage).await
            })
        }

        fn forward_many<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            self.inner.forward_many(msg, outputs)
        }
    }

    impl QueueTransport for EngineOwnershipLossTransport {
        fn poll_stream<'a>(
            &'a self,
            stream: &'a str,
            consumer: &'a str,
            count: usize,
            block_ms: u64,
        ) -> BoxFuture<'a, Result<Vec<scicomp_rq::Message>>> {
            self.inner.poll_stream(stream, consumer, count, block_ms)
        }

        fn ack<'a>(&'a self, msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.acked.fetch_add(1, Ordering::SeqCst);
                self.inner.ack(msg).await
            })
        }

        fn reclaim_idle<'a>(
            &'a self,
            stream: &'a str,
            consumer: &'a str,
            min_idle_ms: u64,
            count: usize,
        ) -> BoxFuture<'a, Result<Vec<scicomp_rq::Message>>> {
            self.inner
                .reclaim_idle(stream, consumer, min_idle_ms, count)
        }

        fn renew_message_lease<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _consumer: &'a str,
        ) -> BoxFuture<'a, Result<bool>> {
            Box::pin(async move {
                self.state.upload_and_renewal_started.wait().await;
                Ok(false)
            })
        }

        fn create_consumer_group<'a>(
            &'a self,
            stream: &'a str,
            group: &'a str,
        ) -> BoxFuture<'a, Result<()>> {
            self.inner.create_consumer_group(stream, group)
        }

        fn increment_failure_attempt<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
        ) -> BoxFuture<'a, Result<Option<usize>>> {
            Box::pin(async move {
                self.failure_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(Some(1))
            })
        }

        fn as_sink(&self) -> &dyn MessageSink {
            self
        }
    }

    fn expect_acquired_owner_token(claim: PublishClaim) -> String {
        match claim {
            PublishClaim::Acquired { owner_token } => owner_token,
            other => panic!("expected acquired publish claim, got {other:?}"),
        }
    }

    fn publish_env_with_config(config: serde_json::Value) -> RoleEnv {
        RoleEnv {
            role_name: "publish".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![InputStreamSpec {
                stream: "publish".to_string(),
                max_dequeue_items: 4,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }],
            resolved_outputs: vec!["results".to_string()],
            role_config: Some(config),
            python_runtime_envs: Default::default(),
        }
    }

    #[test]
    fn from_env_parses_publish_role_config() {
        let role = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({
                "max_concurrent_files": 12,
                "multipart_threshold_bytes": 67_108_864u64,
                "multipart_part_size_bytes": 16_777_216,
                "multipart_max_concurrency": 3,
                "client_options": {
                    "timeout_secs": 300,
                    "connect_timeout_secs": 10,
                    "pool_max_idle_per_host": 64
                },
                "retry": {
                    "max_retries": 9,
                    "timeout_secs": 240
                }
            })),
            Arc::new(NoopPublishStatusPersistence),
        )
        .expect("publish role config should parse");

        assert_eq!(role.config.max_concurrent_files, 12);
        assert_eq!(role.config.multipart_threshold_bytes, 67_108_864);
        assert_eq!(role.config.multipart_part_size_bytes, 16_777_216);
        assert_eq!(role.config.multipart_max_concurrency, 3);
        assert_eq!(role.config.client_options.timeout_secs, Some(300));
        assert_eq!(role.config.client_options.connect_timeout_secs, Some(10));
        assert_eq!(role.config.client_options.pool_max_idle_per_host, Some(64));
        assert_eq!(role.config.retry.max_retries, Some(9));
        assert_eq!(role.config.retry.timeout_secs, Some(240));
    }

    #[tokio::test]
    async fn s3_http_endpoint_sources_enable_http_without_ambient_config() {
        let _guard = test_env::env_lock().lock().await;
        let _access_key = EnvRestore::set("AWS_ACCESS_KEY_ID", Some("test"));
        let _secret_key = EnvRestore::set("AWS_SECRET_ACCESS_KEY", Some("test"));
        let _region = EnvRestore::set("AWS_DEFAULT_REGION", Some("us-east-1"));
        let _allow_http = EnvRestore::set("AWS_ALLOW_HTTP", None);
        for explicit_endpoint in [true, false] {
            let endpoint_source = if explicit_endpoint {
                "explicit"
            } else {
                "S3_ENDPOINT_URL fallback"
            };
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("mock S3 listener should bind");
            let endpoint = format!(
                "http://{}",
                listener
                    .local_addr()
                    .expect("mock S3 listener should have an address")
            );
            let server = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.expect("S3 request should arrive");
                let mut request = vec![0; 4096];
                let _ = stream
                    .read(&mut request)
                    .await
                    .expect("S3 request should be readable");
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nETag: \"test\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .expect("mock S3 response should be written");
            });
            let _endpoint_env = EnvRestore::set(
                "S3_ENDPOINT_URL",
                (!explicit_endpoint).then_some(endpoint.as_str()),
            );
            let mut config = PublishRoleConfig::default();
            config.client_options.timeout_secs = Some(15);
            let resolved = ResolvedOutputPublicationTarget {
                artifact: "primary".to_string(),
                provider: OutputPublicationProvider::S3,
                storage: ResolvedOutputPublicationStorage::S3 {
                    bucket: "bucket".to_string(),
                    prefix: "runs/run-1".to_string(),
                    region: Some("us-east-1".to_string()),
                    endpoint: explicit_endpoint.then_some(endpoint),
                },
            };

            let target = build_publication_target(&resolved, &config)
                .unwrap_or_else(|error| panic!("{endpoint_source} HTTP endpoint: {error:#}"));
            let upload = target
                .store
                .put(&ObjectPath::from("artifact.bin"), b"data".as_slice().into())
                .await;
            server.abort();

            assert_eq!(target.destination_uri, "s3://bucket");
            upload.unwrap_or_else(|error| {
                panic!("{endpoint_source} HTTP endpoint should allow an upload: {error:#}")
            });
        }
    }

    #[tokio::test]
    async fn azure_sovereign_endpoint_builds_from_explicit_endpoint() {
        let _guard = test_env::env_lock().lock().await;
        let _account = EnvRestore::set("AZURE_STORAGE_ACCOUNT_NAME", Some("acct"));
        let _access_key = EnvRestore::set(
            "AZURE_STORAGE_ACCOUNT_KEY",
            Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        );
        let _use_emulator = EnvRestore::set("AZURE_STORAGE_USE_EMULATOR", None);
        let resolved = ResolvedOutputPublicationTarget {
            artifact: "primary".to_string(),
            provider: OutputPublicationProvider::Azure,
            storage: ResolvedOutputPublicationStorage::Azure {
                container: "forecast-results".to_string(),
                prefix: "runs/run-1".to_string(),
                endpoint: "https://acct.blob.core.usgovcloudapi.net".to_string(),
            },
        };

        let target = build_publication_target(&resolved, &PublishRoleConfig::default())
            .expect("sovereign Azure endpoint should build");

        assert_eq!(
            target.destination_uri,
            "https://acct.blob.core.usgovcloudapi.net/forecast-results"
        );
    }

    #[tokio::test]
    async fn azure_account_alias_builds_from_explicit_endpoint() {
        let _guard = test_env::env_lock().lock().await;
        let _account_name = EnvRestore::set("AZURE_STORAGE_ACCOUNT_NAME", None);
        let _account = EnvRestore::set("AZURE_STORAGE_ACCOUNT", Some("acct"));
        let _access_key = EnvRestore::set(
            "AZURE_STORAGE_ACCOUNT_KEY",
            Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        );
        let _use_emulator = EnvRestore::set("AZURE_STORAGE_USE_EMULATOR", None);
        let resolved = ResolvedOutputPublicationTarget {
            artifact: "primary".to_string(),
            provider: OutputPublicationProvider::Azure,
            storage: ResolvedOutputPublicationStorage::Azure {
                container: "forecast-results".to_string(),
                prefix: "runs/run-1".to_string(),
                endpoint: "https://acct.blob.core.windows.net".to_string(),
            },
        };

        let target = build_publication_target(&resolved, &PublishRoleConfig::default())
            .expect("AZURE_STORAGE_ACCOUNT should be accepted as an account name alias");

        assert_eq!(
            target.destination_uri,
            "https://acct.blob.core.windows.net/forecast-results"
        );
    }

    #[tokio::test]
    async fn multipart_final_part_failure_aborts_without_masking_error() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let local_path = tmp.path().join("artifact.bin");
        std::fs::write(&local_path, b"final").expect("test artifact should be written");
        let aborted = Arc::new(AtomicBool::new(false));
        let store = FinalPartFailingStore {
            aborted: Arc::clone(&aborted),
        };
        let config = PublishRoleConfig {
            multipart_part_size_bytes: 8,
            multipart_max_concurrency: 1,
            ..Default::default()
        };
        let cancellation = UploadCancellation::new();

        let error = put_file_multipart(
            &store,
            &ObjectPath::from("artifact.bin"),
            &local_path,
            &config,
            &cancellation,
        )
        .await
        .expect_err("final multipart part should fail");

        assert!(
            aborted.load(Ordering::SeqCst),
            "multipart upload must be aborted before returning"
        );
        assert!(
            format!("{error:#}").contains("final part failed"),
            "abort failure must not mask the original part error: {error:#}"
        );
    }

    #[tokio::test]
    async fn claim_renewal_failure_aborts_active_multipart_upload() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let local_path = tmp.path().join("artifact.bin");
        std::fs::write(&local_path, b"multipart").expect("test artifact should be written");
        let state = Arc::new(BlockingMultipartState {
            upload_and_renewal_started: Barrier::new(2),
            aborts: AtomicUsize::new(0),
            completes: AtomicUsize::new(0),
        });
        let persistence = Arc::new(RenewalFailsDuringMultipartUpload {
            state: Arc::clone(&state),
        }) as Arc<dyn PublishStatusPersistence>;
        let target = PublicationTarget {
            store: Arc::new(BlockingMultipartStore {
                state: Arc::clone(&state),
                fail_multipart_path: None,
            }),
            prefix: ObjectPath::from("runs/run-renewal"),
            destination_uri: "s3://bucket".to_string(),
        };
        let selected = SelectedArtifact {
            name: "primary".to_string(),
            storage_path: local_path,
            filename: "artifact.bin".to_string(),
        };
        let config = PublishRoleConfig {
            multipart_threshold_bytes: 1,
            multipart_part_size_bytes: 8,
            multipart_max_concurrency: 1,
            ..Default::default()
        };
        let ownership_cancellation = RoleCancellation::new();

        let upload_result = upload_selected_artifact_with_claim_renewal(
            persistence,
            "run-renewal",
            "s3:bucket:runs/run-renewal:primary:artifact.bin",
            "owner-token",
            &selected,
            &target,
            &config,
            &ownership_cancellation,
        )
        .await;

        let error = upload_result.expect_err("claim renewal failure must stop publication");
        assert!(
            format!("{error:#}").contains("renewal backend unavailable"),
            "the original renewal error must be preserved: {error:#}"
        );
        assert_eq!(
            state.aborts.load(Ordering::SeqCst),
            1,
            "an active multipart upload must be aborted exactly once"
        );
        assert_eq!(
            state.completes.load(Ordering::SeqCst),
            0,
            "a cancelled multipart upload must not be completed"
        );
    }

    #[tokio::test]
    async fn engine_ownership_loss_aborts_multipart_and_releases_publish_claim() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let local_path = tmp.path().join("artifact.bin");
        std::fs::write(&local_path, b"multipart").expect("test artifact should be written");
        let state = Arc::new(BlockingMultipartState {
            upload_and_renewal_started: Barrier::new(3),
            aborts: AtomicUsize::new(0),
            completes: AtomicUsize::new(0),
        });
        let persistence = Arc::new(EnginePublishClaimPersistence {
            state: Arc::clone(&state),
            claim_state: Mutex::new(EnginePublishClaimState::Unclaimed),
            failed_claims: AtomicUsize::new(0),
            completed_claims: AtomicUsize::new(0),
        });
        let role = EngineMultipartPublishRole {
            persistence: Arc::clone(&persistence),
            selected: SelectedArtifact {
                name: "primary".to_string(),
                storage_path: local_path,
                filename: "artifact.bin".to_string(),
            },
            target: PublicationTarget {
                store: Arc::new(BlockingMultipartStore {
                    state: Arc::clone(&state),
                    fail_multipart_path: None,
                }),
                prefix: ObjectPath::from("runs/run-engine"),
                destination_uri: "s3://bucket".to_string(),
            },
            config: PublishRoleConfig {
                multipart_threshold_bytes: 1,
                multipart_part_size_bytes: 8,
                multipart_max_concurrency: 1,
                ..Default::default()
            },
        };
        let config: RuntimeConfig = serde_json::from_value(json!({
            "stream_prefix": "",
            "max_retries": 1,
            "shared_dlq_stream": "dlq",
            "streams": ["publish", "results", "dlq"],
            "roles": {
                "publish": {
                    "inputs": [{"stream": "publish", "max_dequeue_items": 1,
                                "poll_interval_ms": 1, "block_ms": 1, "reclaim_idle_ms": 3}],
                    "outputs": ["results"]
                }
            }
        }))
        .expect("runtime config should parse");
        let inner = InMemoryTransport::new(&["publish", "results", "dlq"], "");
        inner
            .inject(
                "publish",
                "run-engine",
                r#"{"result":{"status":"succeeded"}}"#,
                "publish",
            )
            .expect("publish message should inject");
        let transport = Arc::new(EngineOwnershipLossTransport {
            inner,
            state: Arc::clone(&state),
            acked: AtomicUsize::new(0),
            handoffs: AtomicUsize::new(0),
            failure_attempts: AtomicUsize::new(0),
        });
        let engine = EngineBuilder::new(&config, "publish")
            .transport(transport.clone())
            .role(Box::new(role))
            .consumer("stale-publish-consumer")
            .build()
            .expect("engine should build");

        let stats = tokio::time::timeout(Duration::from_secs(1), engine.run_once())
            .await
            .expect("ownership-loss cancellation should be bounded")
            .expect("ownership loss should not fail the engine run");

        let aborts = state.aborts.load(Ordering::SeqCst);
        let claim_state = persistence.claim_state();
        assert_eq!(
            (aborts, claim_state),
            (1, EnginePublishClaimState::Failed),
            "ownership loss must abort the multipart upload and release its claim; \
             observed aborts={aborts}, claim_state={claim_state:?}"
        );
        assert_eq!(state.completes.load(Ordering::SeqCst), 0);
        assert_eq!(persistence.failed_claims.load(Ordering::SeqCst), 1);
        assert_eq!(persistence.completed_claims.load(Ordering::SeqCst), 0);
        assert_eq!(stats.acked, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(transport.acked.load(Ordering::SeqCst), 0);
        assert_eq!(transport.handoffs.load(Ordering::SeqCst), 0);
        assert_eq!(transport.failure_attempts.load(Ordering::SeqCst), 0);

        let retry_claim = persistence
            .try_claim_publish(
                "run-engine",
                "s3:bucket:runs/run-engine:primary:artifact.bin",
            )
            .await
            .expect("a retry should inspect the released claim");
        assert!(
            matches!(retry_claim, PublishClaim::Acquired { .. }),
            "ownership cleanup must avoid a stale 600-second InProgress deferral"
        );
    }

    #[test]
    fn from_env_rejects_invalid_publish_role_config() {
        let result = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({
                "max_concurrent_files": 0
            })),
            Arc::new(NoopPublishStatusPersistence),
        );
        let err = match result {
            Ok(_) => panic!("zero upload concurrency should be rejected"),
            Err(error) => error,
        };

        assert!(
            err.to_string().contains("max_concurrent_files"),
            "expected max_concurrent_files error, got: {err}"
        );
    }

    #[tokio::test]
    async fn from_env_uses_publish_role_config_json_env_override() {
        let _guard = test_env::env_lock().lock().await;
        let previous = std::env::var(ENV_PUBLISH_ROLE_CONFIG_JSON).ok();
        test_env::set_env_var(
            ENV_PUBLISH_ROLE_CONFIG_JSON,
            Some(r#"{"max_concurrent_files":5}"#),
        );

        let role = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({
                "max_concurrent_files": 12
            })),
            Arc::new(NoopPublishStatusPersistence),
        )
        .expect("env override should parse");
        test_env::set_env_var(ENV_PUBLISH_ROLE_CONFIG_JSON, previous.as_deref());

        assert_eq!(role.config.max_concurrent_files, 5);
    }

    #[tokio::test]
    async fn process_message_marks_publication_skipped_when_run_failed() {
        let status_persistence = Arc::new(RecordingStatusPersistence::new());
        let role = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({"max_concurrent_files": 1})),
            status_persistence.clone(),
        )
        .expect("publish role should build");
        let payload = json!({
            "workflow_id": "demo",
            "operation": "run",
            "request": {
                "content_type": "application/json"
            },
            "result": {
                "status": "failed",
                "error": "boom"
            },
            "output_publication": {
                "target": {
                    "artifact": "primary",
                    "provider": "s3",
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "runs/run-1"
                    }
                }
            },
            "stage_context": {
                "current_stage_id": "publish",
                "current_phase": "publish",
                "pipeline": [
                    {
                        "id": "publish",
                        "phase": "publish",
                        "queue": "publish",
                        "next": "results"
                    },
                    {
                        "id": "results",
                        "phase": "results",
                        "queue": "results",
                        "next": null
                    }
                ]
            }
        });
        let encoded = serde_json::to_string(&payload).expect("payload should encode");
        let msg =
            scicomp_rq::Message::new("1-0", "publish", "publish:grp", "run-1", encoded, "publish");
        let sink = RecordingSink::new();
        let cancellation = RoleCancellation::new();

        role.process_message(&msg, &sink, &cancellation)
            .await
            .expect("failed run should hand off to results");

        let updates = status_persistence.updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].run_id, "run-1");
        assert_eq!(updates[0].status, "skipped");
        assert_eq!(updates[0].published_artifact_count, Some(0));

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "results");
        assert_eq!(writes[0].stage, "results");
        let results_payload: serde_json::Value =
            serde_json::from_str(&writes[0].payload).expect("results payload should parse");
        assert_eq!(results_payload["status"], "failed");
    }

    #[tokio::test]
    async fn process_message_hands_off_local_results_when_target_construction_fails() {
        let status_persistence = Arc::new(RecordingStatusPersistence::new());
        let role = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({"max_concurrent_files": 1})),
            status_persistence.clone(),
        )
        .expect("publish role should build");
        let payload = json!({
            "workflow_id": "demo",
            "operation": "run",
            "request": {
                "content_type": "application/json"
            },
            "result": {
                "status": "succeeded",
                "output_path": "/tmp/run-1/output.zarr"
            },
            "output_publication": {
                "target": {
                    "artifact": "primary",
                    "provider": "s3",
                    "storage": {
                        "type": "s3",
                        "bucket": "",
                        "prefix": "runs/run-1"
                    }
                }
            },
            "stage_context": {
                "current_stage_id": "publish",
                "current_phase": "publish",
                "pipeline": [
                    {
                        "id": "publish",
                        "phase": "publish",
                        "queue": "publish",
                        "next": "results"
                    },
                    {
                        "id": "results",
                        "phase": "results",
                        "queue": "results",
                        "next": null
                    }
                ]
            }
        });
        let encoded = serde_json::to_string(&payload).expect("payload should encode");
        let msg =
            scicomp_rq::Message::new("1-0", "publish", "publish:grp", "run-1", encoded, "publish");
        let sink = RecordingSink::new();
        let cancellation = RoleCancellation::new();

        role.process_message(&msg, &sink, &cancellation)
            .await
            .expect("invalid publication target should still hand off to results");

        let updates = status_persistence.updates();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].status, "uploading");
        assert_eq!(updates[1].status, "failed");
        assert_eq!(
            updates[1].error.as_deref(),
            Some("publish: s3 bucket is required")
        );

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].dest_stream, "results");
        assert_eq!(writes[0].stage, "results");
        let results_payload: serde_json::Value =
            serde_json::from_str(&writes[0].payload).expect("results payload should parse");
        assert_eq!(results_payload["status"], "failed");
        assert_eq!(
            results_payload["execution"]["output_path"],
            "/tmp/run-1/output.zarr"
        );
        assert_eq!(
            results_payload["execution"]["error"],
            "publish: s3 bucket is required"
        );
        assert_eq!(
            results_payload["execution"]["published_artifacts"][0]["status"],
            "failed"
        );
    }

    #[tokio::test]
    async fn process_message_hands_off_local_results_when_local_artifact_is_unreadable() {
        let _guard = test_env::env_lock().lock().await;
        let _access_key = EnvRestore::set("AWS_ACCESS_KEY_ID", Some("test"));
        let _secret_key = EnvRestore::set("AWS_SECRET_ACCESS_KEY", Some("test"));
        let _region = EnvRestore::set("AWS_DEFAULT_REGION", Some("us-east-1"));
        let _allow_http = EnvRestore::set("AWS_ALLOW_HTTP", Some("true"));

        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let missing_output = tmp.path().join("missing.zarr");
        let missing_output = missing_output.to_string_lossy().to_string();

        let status_persistence = Arc::new(RecordingStatusPersistence::new());
        let role = PublishRole::from_env_with_status_persistence(
            &publish_env_with_config(json!({"max_concurrent_files": 1})),
            status_persistence.clone(),
        )
        .expect("publish role should build");
        let payload = json!({
            "workflow_id": "demo",
            "operation": "run",
            "request": {
                "content_type": "application/json"
            },
            "result": {
                "status": "succeeded",
                "output_path": missing_output.clone(),
                "outputs": [
                    {
                        "name": "primary",
                        "storage_path": missing_output.clone(),
                        "filename": "missing.zarr",
                        "primary": true
                    }
                ]
            },
            "output_publication": {
                "target": {
                    "artifact": "primary",
                    "provider": "s3",
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "runs/run-1",
                        "region": "us-east-1",
                        "endpoint": "http://127.0.0.1:9"
                    }
                }
            },
            "stage_context": {
                "current_stage_id": "publish",
                "current_phase": "publish",
                "pipeline": [
                    {
                        "id": "publish",
                        "phase": "publish",
                        "queue": "publish",
                        "next": "results"
                    },
                    {
                        "id": "results",
                        "phase": "results",
                        "queue": "results",
                        "next": null
                    }
                ]
            }
        });
        let encoded = serde_json::to_string(&payload).expect("payload should encode");
        let msg =
            scicomp_rq::Message::new("1-0", "publish", "publish:grp", "run-1", encoded, "publish");
        let sink = RecordingSink::new();
        let cancellation = RoleCancellation::new();

        role.process_message(&msg, &sink, &cancellation)
            .await
            .expect("local artifact read failure should still hand off to results");

        let updates = status_persistence.updates();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].status, "uploading");
        assert_eq!(updates[1].status, "failed");
        assert!(
            updates[1]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("failed to stat local artifact")),
            "expected local stat failure, got {:?}",
            updates[1].error
        );

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let results_payload: serde_json::Value =
            serde_json::from_str(&writes[0].payload).expect("results payload should parse");
        assert_eq!(results_payload["status"], "failed");
        assert_eq!(results_payload["execution"]["output_path"], missing_output);
        assert_eq!(
            results_payload["execution"]["published_artifacts"][0]["status"],
            "failed"
        );
        assert!(
            results_payload["execution"]["error"]
                .as_str()
                .is_some_and(|error| error.contains("failed to stat local artifact")),
            "expected result payload to preserve publish failure"
        );
    }

    #[tokio::test]
    async fn in_progress_publish_claim_defers_without_dlq_and_can_complete_later() {
        let _guard = test_env::env_lock().lock().await;
        let _access_key = EnvRestore::set("AWS_ACCESS_KEY_ID", Some("test"));
        let _secret_key = EnvRestore::set("AWS_SECRET_ACCESS_KEY", Some("test"));
        let _region = EnvRestore::set("AWS_DEFAULT_REGION", Some("us-east-1"));
        let _allow_http = EnvRestore::set("AWS_ALLOW_HTTP", Some("true"));
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let output = tmp.path().join("dataset.zarr");
        std::fs::write(&output, b"data").expect("local output should exist");
        let output = output.to_string_lossy().to_string();
        let payload = json!({
            "workflow_id": "demo",
            "operation": "run",
            "request": {"content_type": "application/json"},
            "result": {
                "status": "succeeded",
                "output_path": output,
                "outputs": [{
                    "name": "primary",
                    "storage_path": output,
                    "filename": "dataset.zarr",
                    "primary": true
                }]
            },
            "output_publication": {
                "target": {
                    "artifact": "primary",
                    "provider": "s3",
                    "storage": {
                        "type": "s3",
                        "bucket": "bucket",
                        "prefix": "runs/run-claim",
                        "region": "us-east-1",
                        "endpoint": "http://127.0.0.1:9"
                    }
                }
            },
            "stage_context": {
                "current_stage_id": "publish",
                "current_phase": "publish",
                "pipeline": [
                    {"id": "publish", "phase": "publish", "queue": "publish", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        });
        let config: RuntimeConfig = serde_json::from_value(json!({
            "stream_prefix": "",
            "max_retries": 1,
            "shared_dlq_stream": "dlq",
            "streams": ["publish", "results", "dlq"],
            "roles": {
                "publish": {
                    "inputs": [{"stream": "publish", "max_dequeue_items": 1,
                                "poll_interval_ms": 1, "block_ms": 1, "reclaim_idle_ms": 10}],
                    "outputs": ["results"],
                    "config": {"max_concurrent_files": 1}
                }
            }
        }))
        .expect("runtime config should parse");
        let persistence = Arc::new(DeferredThenCompletedPersistence {
            claim_attempts: AtomicUsize::new(0),
        });
        let role = PublishRole::from_env_with_status_persistence(
            &config
                .resolve_env("publish")
                .expect("publish env should resolve"),
            persistence.clone(),
        )
        .expect("publish role should build");
        let transport = Arc::new(InMemoryTransport::new(&["publish", "results", "dlq"], ""));
        transport
            .inject(
                "publish",
                "run-claim",
                &serde_json::to_string(&payload).expect("payload should encode"),
                "publish",
            )
            .expect("publish message should inject");
        let engine = EngineBuilder::new(&config, "publish")
            .transport(transport.clone())
            .role(Box::new(role))
            .consumer("publish-consumer")
            .build()
            .expect("engine should build");

        let deferred = engine
            .run_once()
            .await
            .expect("deferred run should succeed");
        assert_eq!(deferred.acked, 0);
        assert!(
            transport.pending_in("dlq").is_empty(),
            "active ownership must not consume a retry or move the message to DLQ"
        );

        let completed = engine.run_once().await.expect("retry should succeed");
        assert_eq!(completed.succeeded, 1);
        assert_eq!(persistence.claim_attempts.load(Ordering::SeqCst), 2);
        assert!(transport.pending_in("dlq").is_empty());
        assert_eq!(transport.pending_in("results").len(), 1);
    }

    #[tokio::test]
    async fn redis_publish_status_clears_stale_error_after_uploaded() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-clear-stale-error").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_key = "run:run-clear-publish-error";
        let mut conn = qm.connection();
        let _: usize = redis::cmd("HSET")
            .arg(run_key)
            .arg("publish_error")
            .arg("previous upload failure")
            .query_async(&mut conn)
            .await
            .expect("stale publish error should seed");

        persistence
            .persist_status(PublishStatusUpdate {
                published_artifact_count: Some(1),
                ..PublishStatusUpdate::new("run-clear-publish-error", "uploaded")
            })
            .await
            .expect("uploaded status should persist");

        let status: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("output_publication_status")
            .query_async(&mut conn)
            .await
            .expect("publication status should be readable");
        let publish_error: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("publish_error")
            .query_async(&mut conn)
            .await
            .expect("publish error should be queryable");
        let output_location: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("output_location")
            .query_async(&mut conn)
            .await
            .expect("output location should be queryable");

        assert_eq!(status.as_deref(), Some("uploaded"));
        assert!(publish_error.is_none());
        assert_eq!(output_location.as_deref(), Some("local_and_cloud"));
    }

    #[tokio::test]
    async fn redis_publish_status_skipped_preserves_failed_run_status() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-skipped-preserve-failed").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_key = "run:run-skipped-after-failure";
        let mut conn = qm.connection();
        let _: usize = redis::cmd("HSET")
            .arg(run_key)
            .arg("status")
            .arg("failed")
            .query_async(&mut conn)
            .await
            .expect("failed run status should seed");

        persistence
            .persist_status(PublishStatusUpdate {
                published_artifact_count: Some(0),
                ..PublishStatusUpdate::new("run-skipped-after-failure", "skipped")
            })
            .await
            .expect("skipped status should persist");

        let run_status: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("status")
            .query_async(&mut conn)
            .await
            .expect("run status should be readable");
        let publication_status: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("output_publication_status")
            .query_async(&mut conn)
            .await
            .expect("publication status should be readable");
        let output_location: Option<String> = redis::cmd("HGET")
            .arg(run_key)
            .arg("output_location")
            .query_async(&mut conn)
            .await
            .expect("output location should be readable");

        assert_eq!(run_status.as_deref(), Some("failed"));
        assert_eq!(publication_status.as_deref(), Some("skipped"));
        assert!(
            output_location.is_none(),
            "a skipped publication must not advertise cloud output"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_returns_completed_record_after_successful_publish() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-idempotency-claim").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let fingerprint = "s3:bucket:runs/run-1:primary:dataset.zarr";
        let artifact = PublishedArtifactRecord {
            provider: "s3".to_string(),
            source_artifact: "primary".to_string(),
            destination_uri: "s3://bucket/runs/run-1/dataset.zarr".to_string(),
            manifest_uri: Some(
                "s3://bucket/runs/run-1/dataset.zarr/_physicsnemo_serve_publish_manifest.json"
                    .to_string(),
            ),
            object_count: 3,
            total_bytes: 42,
            filename: "dataset.zarr".to_string(),
        };

        let first_claim = persistence
            .try_claim_publish("run-1", fingerprint)
            .await
            .expect("first publish claim should succeed");
        let owner_token = expect_acquired_owner_token(first_claim);

        persistence
            .complete_publish("run-1", fingerprint, &owner_token, artifact.clone())
            .await
            .expect("publish completion marker should persist");

        let second_claim = persistence
            .try_claim_publish("run-1", fingerprint)
            .await
            .expect("second publish claim should read completed marker");
        assert_eq!(
            second_claim,
            PublishClaim::AlreadyCompleted(artifact),
            "completed publish marker should suppress duplicate upload work"
        );

        let mut conn = qm.connection();
        let ttl_secs: i64 = redis::cmd("TTL")
            .arg(publish_claim_key("run-1", fingerprint))
            .query_async(&mut conn)
            .await
            .expect("completed publish claim ttl should be readable");
        assert!(
            ttl_secs > 0 && ttl_secs <= (24 * 60 * 60),
            "completed publish claims must have a bounded positive ttl, got {ttl_secs}"
        );
    }

    #[tokio::test]
    async fn redis_failed_publish_claim_has_bounded_terminal_ttl() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-failed-claim-ttl").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_id = "run-failed-claim-ttl";
        let fingerprint = "s3:bucket:runs/run-failed-claim-ttl:primary:dataset.zarr";
        let owner_token = expect_acquired_owner_token(
            persistence
                .try_claim_publish(run_id, fingerprint)
                .await
                .expect("publish claim should succeed"),
        );

        persistence
            .fail_publish_claim(run_id, fingerprint, &owner_token, "upload failed")
            .await
            .expect("failed publish claim should persist");

        let mut conn = qm.connection();
        let ttl_secs: i64 = redis::cmd("TTL")
            .arg(publish_claim_key(run_id, fingerprint))
            .query_async(&mut conn)
            .await
            .expect("failed publish claim ttl should be readable");
        assert!(
            ttl_secs > 0 && ttl_secs <= (24 * 60 * 60),
            "failed publish claims must have a bounded positive ttl, got {ttl_secs}"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_reacquires_expired_in_progress_marker() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-claim-expiry").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_id = "run-stale-publish-claim";
        let fingerprint = "s3:bucket:runs/run-stale-publish-claim:primary:dataset.zarr";
        let key = publish_claim_key(run_id, fingerprint);

        let first_claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("first publish claim should succeed");
        let owner_token = expect_acquired_owner_token(first_claim);
        assert!(!owner_token.is_empty());

        let mut conn = qm.connection();
        let initial_ttl_ms: i64 = redis::cmd("PTTL")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .expect("publish claim ttl should be readable");
        assert!(
            initial_ttl_ms > 0,
            "in-progress publish claims must expire so crashed workers do not block retries"
        );

        let _: bool = redis::cmd("PEXPIRE")
            .arg(&key)
            .arg(20_i64)
            .query_async(&mut conn)
            .await
            .expect("test should shorten publish claim ttl");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let second_claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("expired publish claim should be claimable again");
        let owner_token = expect_acquired_owner_token(second_claim);
        assert!(
            !owner_token.is_empty(),
            "expired in-progress publish claims should be recoverable"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_renewal_keeps_active_owner_claimed() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-claim-renewal").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_id = "run-renew-publish-claim";
        let fingerprint = "s3:bucket:runs/run-renew-publish-claim:primary:dataset.zarr";
        let key = publish_claim_key(run_id, fingerprint);

        let owner_token = expect_acquired_owner_token(
            persistence
                .try_claim_publish(run_id, fingerprint)
                .await
                .expect("first publish claim should succeed"),
        );

        let mut conn = qm.connection();
        let _: bool = redis::cmd("PEXPIRE")
            .arg(&key)
            .arg(20_i64)
            .query_async(&mut conn)
            .await
            .expect("test should shorten publish claim ttl");
        persistence
            .renew_publish_claim(run_id, fingerprint, &owner_token)
            .await
            .expect("current owner should renew the publish claim");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let competing_claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("competing claim check should succeed");
        assert_eq!(
            competing_claim,
            PublishClaim::InProgress,
            "renewed active publish claims must not be reacquired"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_reacquires_legacy_stale_in_progress_marker() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-claim-legacy-stale").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_id = "run-legacy-stale-publish-claim";
        let fingerprint = "s3:bucket:runs/run-legacy-stale-publish-claim:primary:dataset.zarr";
        let key = publish_claim_key(run_id, fingerprint);
        let legacy_state = serde_json::to_string(&PublishClaimState {
            status: PublishClaimStatus::InProgress,
            fingerprint: fingerprint.to_string(),
            owner_token: None,
            artifact: None,
            error: None,
            updated_at: 0,
        })
        .expect("legacy claim state should serialize");

        let mut conn = qm.connection();
        let _: () = redis::cmd("SET")
            .arg(&key)
            .arg(&legacy_state)
            .query_async(&mut conn)
            .await
            .expect("legacy in-progress claim should seed");

        let claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("legacy stale publish claim should be claimable");
        let owner_token = expect_acquired_owner_token(claim);
        assert!(
            !owner_token.is_empty(),
            "stale in-progress publish claims from crashed workers should be recoverable"
        );

        let ttl_ms: i64 = redis::cmd("PTTL")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .expect("reacquired publish claim ttl should be readable");
        assert!(
            ttl_ms > 0,
            "reacquired in-progress publish claims should receive an expiry"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_stale_owner_cannot_overwrite_new_owner_state() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-claim-stale-owner").await;
        let persistence = RedisPublishStatusPersistence::new(qm.clone());
        let run_id = "run-stale-owner-publish-claim";
        let fingerprint = "s3:bucket:runs/run-stale-owner-publish-claim:primary:dataset.zarr";
        let key = publish_claim_key(run_id, fingerprint);

        let first_claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("first publish claim should succeed");
        let first_owner_token = expect_acquired_owner_token(first_claim);

        let mut conn = qm.connection();
        let _: bool = redis::cmd("PEXPIRE")
            .arg(&key)
            .arg(20_i64)
            .query_async(&mut conn)
            .await
            .expect("test should shorten publish claim ttl");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let second_claim = persistence
            .try_claim_publish(run_id, fingerprint)
            .await
            .expect("expired publish claim should be claimable by a new owner");
        let second_owner_token = expect_acquired_owner_token(second_claim);

        persistence
            .fail_publish_claim(
                run_id,
                fingerprint,
                &second_owner_token,
                "new owner failure",
            )
            .await
            .expect("new owner failure should persist");
        let stale_completion = persistence
            .complete_publish(
                run_id,
                fingerprint,
                &first_owner_token,
                PublishedArtifactRecord {
                    provider: "s3".to_string(),
                    source_artifact: "primary".to_string(),
                    destination_uri: "s3://bucket/old-owner/dataset.zarr".to_string(),
                    manifest_uri: None,
                    object_count: 1,
                    total_bytes: 42,
                    filename: "dataset.zarr".to_string(),
                },
            )
            .await;
        assert!(
            stale_completion.is_err(),
            "stale owner completion should be rejected"
        );

        let raw: String = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .expect("claim state should be readable");
        let state: PublishClaimState =
            serde_json::from_str(&raw).expect("claim state should decode");
        assert_eq!(
            state.status,
            PublishClaimStatus::Failed,
            "a stale owner must not overwrite the newer owner's claim state"
        );
    }

    #[tokio::test]
    async fn redis_publish_claim_allows_only_one_failed_claim_reacquirer() {
        let (_redis_server, qm) = spawn_test_queue_manager("publish-claim-failed-concurrent").await;
        let persistence = Arc::new(RedisPublishStatusPersistence::new(qm));
        let run_id = "run-failed-publish-claim";
        let fingerprint = "s3:bucket:runs/run-failed-publish-claim:primary:dataset.zarr";

        let failed_owner_token = expect_acquired_owner_token(
            persistence
                .try_claim_publish(run_id, fingerprint)
                .await
                .expect("initial claim should seed failed state"),
        );
        persistence
            .fail_publish_claim(run_id, fingerprint, &failed_owner_token, "previous failure")
            .await
            .expect("failed claim should seed");

        let contender_count = 16;
        let barrier = Arc::new(Barrier::new(contender_count + 1));
        let mut tasks = Vec::with_capacity(contender_count);
        for _ in 0..contender_count {
            let persistence = Arc::clone(&persistence);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                persistence.try_claim_publish(run_id, fingerprint).await
            }));
        }
        barrier.wait().await;

        let mut acquired_count = 0;
        for task in tasks {
            let claim = task
                .await
                .expect("claim task should join")
                .expect("claim attempt should not error");
            if matches!(claim, PublishClaim::Acquired { .. }) {
                acquired_count += 1;
            }
        }

        assert_eq!(
            acquired_count, 1,
            "only one retry may atomically reacquire a failed publish claim"
        );
    }

    #[derive(Debug)]
    struct AlwaysFailingPutStore {
        put_attempts: AtomicUsize,
    }

    impl fmt::Display for AlwaysFailingPutStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("AlwaysFailingPutStore")
        }
    }

    #[async_trait]
    impl ObjectStore for AlwaysFailingPutStore {
        async fn put_opts(
            &self,
            _location: &ObjectPath,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.put_attempts.fetch_add(1, Ordering::SeqCst);
            Err(object_store::Error::Generic {
                store: "failing-put-test",
                source: Box::new(std::io::Error::other("permanent upload failure")),
            })
        }

        async fn put_multipart_opts(
            &self,
            _location: &ObjectPath,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            unreachable!("multipart upload is not used by this test")
        }

        async fn get_opts(
            &self,
            _location: &ObjectPath,
            _options: GetOptions,
        ) -> object_store::Result<GetResult> {
            unreachable!("get is not used by this test")
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
        ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
            unreachable!("delete is not used by this test")
        }

        fn list(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            unreachable!("list is not used by this test")
        }

        async fn list_with_delimiter(
            &self,
            _prefix: Option<&ObjectPath>,
        ) -> object_store::Result<ListResult> {
            unreachable!("list is not used by this test")
        }

        async fn copy_opts(
            &self,
            _from: &ObjectPath,
            _to: &ObjectPath,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unreachable!("copy is not used by this test")
        }
    }

    #[tokio::test]
    async fn directory_upload_stops_after_first_file_failure() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_dir = tmp.path().join("dataset.zarr");
        std::fs::create_dir_all(&source_dir).expect("source directory should exist");
        for index in 0..50 {
            std::fs::write(source_dir.join(format!("chunk-{index}")), b"x")
                .expect("chunk should be written");
        }

        let store = Arc::new(AlwaysFailingPutStore {
            put_attempts: AtomicUsize::new(0),
        });
        let cancellation = UploadCancellation::new();
        let config = PublishRoleConfig {
            max_concurrent_files: 4,
            retry: crate::config::PublishRetryConfig {
                max_retries: Some(1),
                timeout_secs: Some(1),
            },
            ..Default::default()
        };

        let error = upload_selected_artifact(
            &SelectedArtifact {
                name: "primary".to_string(),
                storage_path: source_dir,
                filename: "dataset.zarr".to_string(),
            },
            &PublicationTarget {
                store: store.clone(),
                prefix: ObjectPath::from("runs/run-fail-fast"),
                destination_uri: "s3://bucket".to_string(),
            },
            true,
            false,
            &config,
            &cancellation,
        )
        .await
        .expect_err("directory upload should fail on the first permanent error");

        assert!(
            format!("{error:#}").contains("permanent upload failure"),
            "the original upload error must be preserved: {error:#}"
        );
        assert!(
            store.put_attempts.load(Ordering::SeqCst) < 50,
            "directory upload must stop launching remaining files after the first failure"
        );
    }

    #[tokio::test]
    async fn directory_upload_drains_cancelled_multipart_uploads() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_dir = tmp.path().join("dataset.zarr");
        std::fs::create_dir_all(&source_dir).expect("source directory should exist");
        std::fs::write(source_dir.join("a-blocked"), b"multipart")
            .expect("blocking file should be written");
        std::fs::write(source_dir.join("b-failing"), b"multipart")
            .expect("failing file should be written");

        let state = Arc::new(BlockingMultipartState {
            upload_and_renewal_started: Barrier::new(2),
            aborts: AtomicUsize::new(0),
            completes: AtomicUsize::new(0),
        });
        let store = Arc::new(BlockingMultipartStore {
            state: Arc::clone(&state),
            fail_multipart_path: Some(ObjectPath::from("runs/run-drain/dataset.zarr/b-failing")),
        });
        let cancellation = UploadCancellation::new();
        let config = PublishRoleConfig {
            max_concurrent_files: 2,
            multipart_threshold_bytes: 1,
            multipart_part_size_bytes: 8,
            multipart_max_concurrency: 1,
            ..Default::default()
        };

        let error = tokio::time::timeout(
            Duration::from_secs(5),
            upload_selected_artifact(
                &SelectedArtifact {
                    name: "primary".to_string(),
                    storage_path: source_dir,
                    filename: "dataset.zarr".to_string(),
                },
                &PublicationTarget {
                    store,
                    prefix: ObjectPath::from("runs/run-drain"),
                    destination_uri: "s3://bucket".to_string(),
                },
                true,
                false,
                &config,
                &cancellation,
            ),
        )
        .await
        .expect("directory upload should not hang")
        .expect_err("sibling multipart failure should fail the directory upload");

        assert!(
            format!("{error:#}").contains("sibling upload failed"),
            "the original sibling error must be preserved: {error:#}"
        );
        assert_eq!(
            state.aborts.load(Ordering::SeqCst),
            1,
            "the active multipart upload must finish its cancellation abort"
        );
        assert_eq!(state.completes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn upload_selected_artifact_writes_directory_and_manifest() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_dir = tmp.path().join("dataset.zarr");
        std::fs::create_dir_all(source_dir.join("group")).expect("source directory should exist");
        std::fs::write(source_dir.join("group").join("0"), b"abc")
            .expect("chunk should be written");
        std::fs::write(source_dir.join(".zgroup"), b"{}").expect("metadata should be written");

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let cancellation = UploadCancellation::new();
        let stats = upload_selected_artifact(
            &SelectedArtifact {
                name: "primary".to_string(),
                storage_path: source_dir.clone(),
                filename: "dataset.zarr".to_string(),
            },
            &PublicationTarget {
                store: Arc::clone(&store),
                prefix: ObjectPath::from("runs/run-1"),
                destination_uri: "s3://bucket".to_string(),
            },
            true,
            true,
            &PublishRoleConfig::default(),
            &cancellation,
        )
        .await
        .expect("directory should upload");

        assert_eq!(stats.object_count, 2);
        assert_eq!(stats.total_bytes, 5);
        assert_eq!(stats.destination_uri, "s3://bucket/runs/run-1/dataset.zarr");
        assert_eq!(
            stats.manifest_uri.as_deref(),
            Some("s3://bucket/runs/run-1/dataset.zarr/_physicsnemo_serve_publish_manifest.json")
        );

        let chunk = store
            .get(&ObjectPath::from("runs/run-1/dataset.zarr/group/0"))
            .await
            .expect("chunk should exist")
            .bytes()
            .await
            .expect("chunk should read");
        assert_eq!(chunk.as_ref(), b"abc");
    }

    #[test]
    fn explicit_s3_endpoint_uses_s3_specific_config_key() {
        let source = include_str!("publish.rs");

        assert!(
            source.matches("AmazonS3ConfigKey::S3Endpoint").count() > 1,
            "explicit publication S3 endpoints must override ambient AWS_ENDPOINT_URL_S3"
        );
    }

    #[tokio::test]
    async fn collect_directory_entries_rejects_reserved_manifest_filename() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_dir = tmp.path().join("dataset.zarr");
        std::fs::create_dir_all(&source_dir).expect("source directory should exist");
        std::fs::write(
            source_dir.join("_physicsnemo_serve_publish_manifest.json"),
            b"source manifest",
        )
        .expect("reserved source file should be written");

        let error = collect_directory_entries(&source_dir)
            .await
            .expect_err("reserved manifest source file should be rejected");

        assert!(
            error
                .to_string()
                .contains("_physicsnemo_serve_publish_manifest.json")
        );
    }

    #[cfg(unix)]
    #[test]
    fn path_to_object_path_rejects_non_utf8_components() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = PathBuf::from(OsString::from_vec(vec![b'n', b'a', b'm', b'e', 0x80]));

        let error = path_to_object_path(&path)
            .expect_err("non-UTF-8 paths must not be mapped to lossy object keys");

        assert!(error.to_string().contains("non-UTF-8"));
    }

    #[tokio::test]
    async fn upload_selected_artifact_reports_encoded_destination_uri() {
        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_file = tmp.path().join("result.json");
        std::fs::write(&source_file, b"{}").expect("source file should be written");

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let prefix = ObjectPath::from("runs/run #1");
        let filename = "forecast #?.json".to_string();
        let artifact_path = join_object_path(&prefix, &ObjectPath::from(filename.clone()));
        let cancellation = UploadCancellation::new();
        let stats = upload_selected_artifact(
            &SelectedArtifact {
                name: "primary".to_string(),
                storage_path: source_file,
                filename,
            },
            &PublicationTarget {
                store: Arc::clone(&store),
                prefix,
                destination_uri: "s3://bucket".to_string(),
            },
            true,
            false,
            &PublishRoleConfig::default(),
            &cancellation,
        )
        .await
        .expect("file should upload");

        assert_eq!(
            stats.destination_uri,
            object_path_uri("s3://bucket", &artifact_path)
        );
        assert!(!stats.destination_uri.contains('#'));
        assert!(!stats.destination_uri.contains('?'));
        let uploaded = store
            .get(&artifact_path)
            .await
            .expect("uploaded object should exist")
            .bytes()
            .await
            .expect("uploaded object should read");
        assert_eq!(uploaded.as_ref(), b"{}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn collect_directory_entries_rejects_symlink_entries() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_dir = tmp.path().join("dataset.zarr");
        let external = tmp.path().join("external.txt");
        std::fs::create_dir_all(&source_dir).expect("source directory should exist");
        std::fs::write(&external, b"secret").expect("external file should be written");
        symlink(&external, source_dir.join("linked.txt")).expect("symlink should be created");

        let error = collect_directory_entries(&source_dir)
            .await
            .expect_err("directory scan should reject symlink entries");

        assert!(error.to_string().contains("must not be a symlink"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upload_selected_artifact_rejects_symlink_root() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("temp dir should be created");
        let source_file = tmp.path().join("source.txt");
        let linked_file = tmp.path().join("linked.txt");
        std::fs::write(&source_file, b"secret").expect("source file should be written");
        symlink(&source_file, &linked_file).expect("symlink should be created");

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let cancellation = UploadCancellation::new();
        let error = upload_selected_artifact(
            &SelectedArtifact {
                name: "primary".to_string(),
                storage_path: linked_file,
                filename: "linked.txt".to_string(),
            },
            &PublicationTarget {
                store,
                prefix: ObjectPath::from("runs/run-1"),
                destination_uri: "s3://bucket".to_string(),
            },
            true,
            true,
            &PublishRoleConfig::default(),
            &cancellation,
        )
        .await
        .expect_err("symlink artifact roots should be rejected");

        assert!(error.to_string().contains("must not be a symlink"));
    }

    #[test]
    fn select_result_artifact_prefers_flagged_primary_output() {
        let result = json!({
            "outputs": [
                {
                    "name": "thumbnail",
                    "storage_path": "/outputs/run-1/thumbnail.png",
                    "filename": "thumbnail.png"
                },
                {
                    "name": "forecast_dataset",
                    "storage_path": "/outputs/run-1/forecast.zarr",
                    "filename": "forecast.zarr",
                    "primary": true
                }
            ],
            "output_path": "/outputs/run-1/forecast.zarr"
        });

        let selected = select_result_artifact(result.as_object().unwrap(), PRIMARY_TARGET_NAME)
            .expect("primary selection should succeed")
            .expect("primary artifact should be present");

        assert_eq!(selected.name, "forecast_dataset");
        assert_eq!(
            selected.storage_path.to_string_lossy(),
            "/outputs/run-1/forecast.zarr"
        );
        assert_eq!(selected.filename, "forecast.zarr");
    }

    #[test]
    fn select_result_artifact_prefers_output_path_before_first_output() {
        let result = json!({
            "outputs": [
                {
                    "name": "thumbnail",
                    "storage_path": "/outputs/run-1/thumbnail.png",
                    "filename": "thumbnail.png"
                },
                {
                    "name": "forecast_dataset",
                    "storage_path": "/outputs/run-1/forecast.zarr",
                    "filename": "forecast.zarr"
                }
            ],
            "output_path": "/outputs/run-1/forecast.zarr"
        });

        let selected = select_result_artifact(result.as_object().unwrap(), PRIMARY_TARGET_NAME)
            .expect("primary selection should succeed")
            .expect("primary artifact should be present");

        assert_eq!(selected.name, "forecast_dataset");
        assert_eq!(
            selected.storage_path.to_string_lossy(),
            "/outputs/run-1/forecast.zarr"
        );
    }

    #[test]
    fn build_execution_and_payload_preserves_outputs_and_published_artifacts() {
        let (execution, payload) = build_execution_and_payload(
            "run-1",
            "demo",
            "succeeded",
            "2026-01-01T00:00:00Z",
            json!({
                "status": "succeeded",
                "artifacts": [
                    {
                        "name": "primary",
                        "media_type": "application/json",
                        "storage_path": "/outputs/run-1/result.json"
                    }
                ],
                "published_artifacts": [
                    {
                        "kind": "object_store_publish",
                        "provider": "s3",
                        "source_artifact": "primary",
                        "destination_uri": "s3://bucket/run-1/result.json",
                        "status": "uploaded"
                    }
                ],
                "value": 7
            }),
        )
        .expect("execution envelope should build");

        assert_eq!(
            execution["outputs"][0]["storage_path"],
            "/outputs/run-1/result.json"
        );
        assert_eq!(
            execution["published_artifacts"][0]["destination_uri"],
            "s3://bucket/run-1/result.json"
        );
        assert!(payload.get("artifacts").is_none());
        assert!(payload.get("published_artifacts").is_none());
        assert_eq!(payload["value"], 7);
    }

    #[test]
    fn build_execution_and_payload_preserves_nested_execution_outputs() {
        let (execution, _payload) = build_execution_and_payload(
            "run-1",
            "demo",
            "succeeded",
            "2026-01-01T00:00:00Z",
            json!({
                "status": "succeeded",
                "execution": {
                    "outputs": [
                        {
                            "name": "forecast_dataset",
                            "media_type": "application/x-zarr",
                            "storage_path": "/outputs/run-1/forecast.zarr",
                            "filename": "forecast.zarr",
                            "primary": true
                        }
                    ],
                    "output_path": "/outputs/run-1/forecast.zarr"
                },
                "published_artifacts": [
                    {
                        "kind": "object_store_publish",
                        "provider": "azure",
                        "source_artifact": "forecast_dataset",
                        "destination_uri": "https://account.blob.core.windows.net/container/run-1/forecast.zarr",
                        "status": "uploaded"
                    }
                ]
            }),
        )
        .expect("execution envelope should build");

        assert_eq!(
            execution["outputs"][0]["storage_path"],
            "/outputs/run-1/forecast.zarr"
        );
        assert_eq!(execution["output_path"], "/outputs/run-1/forecast.zarr");
        assert_eq!(
            execution["published_artifacts"][0]["source_artifact"],
            "forecast_dataset"
        );
    }
}
