/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use fs2::FileExt;
use futures::{StreamExt, future::join_all};
use reqwest::{Client, Url, redirect};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tracing::{debug, error, info};

use super::plan::{ByteRange, PrefetchOpKind, PrefetchPlanItem};
use super::prefetch_config::PrefetchConfig;

fn build_http_client(config: &PrefetchConfig) -> Result<Client, String> {
    Client::builder()
        .http2_adaptive_window(true)
        .pool_max_idle_per_host(config.http_pool_max_idle_per_host)
        .pool_idle_timeout(config.http_pool_timeout())
        .timeout(config.http_timeout())
        .build()
        .map_err(|e| e.to_string())
}

fn build_verified_http_client(config: &PrefetchConfig) -> Result<Client, String> {
    let allowed_hosts = config.allowed_https_hosts.clone();
    let allowed_signed_redirect_hosts = config.allowed_signed_redirect_hosts.clone();
    Client::builder()
        .http2_adaptive_window(true)
        .pool_max_idle_per_host(config.http_pool_max_idle_per_host)
        .pool_idle_timeout(config.http_pool_timeout())
        .timeout(config.http_timeout())
        .redirect(redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("prefetch: too many verified HTTPS redirects");
            }
            match validate_verified_redirect_url(
                attempt.url(),
                &allowed_hosts,
                &allowed_signed_redirect_hosts,
            ) {
                Ok(()) => attempt.follow(),
                Err(error) => attempt.error(error.to_string()),
            }
        }))
        .build()
        .map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Default)]
#[must_use]
pub struct DownloadStats {
    pub downloaded: usize,
    pub cached: usize,
    pub errors: usize,
    pub required_errors: usize,
    pub required_verified_errors: usize,
    pub optional_errors: usize,
    pub total_time_secs: f64,
    pub throughput_mbps: f64,
    pub total_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaterializedArtifact {
    pub name: String,
    pub source_uri: String,
    pub storage_path: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MaterializationResult {
    pub stats: DownloadStats,
    pub artifacts: Vec<MaterializedArtifact>,
}

#[derive(Debug, Clone)]
struct DownloadRequest {
    cache_key: String,
    kind: PrefetchOpKind,
    source_uri: String,
    byte_range: Option<ByteRange>,
    headers: BTreeMap<String, String>,
    local_path: PathBuf,
    required: bool,
    expected_sha256: Option<String>,
    expected_size_bytes: Option<u64>,
}

impl DownloadRequest {
    fn is_verified(&self) -> bool {
        self.expected_sha256.is_some() || self.expected_size_bytes.is_some()
    }
}

#[derive(Debug, Clone)]
struct RequestSuccess {
    local_path: PathBuf,
    size_bytes: u64,
    downloaded: bool,
    sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CacheMetadata {
    version: u8,
    sha256: String,
    size_bytes: u64,
}

struct TemporaryFileGuard {
    path: PathBuf,
}

impl TemporaryFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    path = ?self.path,
                    error = %error,
                    "failed to clean up temporary prefetch file"
                );
            }
        }
    }
}

#[derive(Debug)]
struct RequestBudget {
    used_bytes: AtomicU64,
    max_bytes: u64,
}

impl RequestBudget {
    fn new(max_bytes: u64) -> Self {
        Self {
            used_bytes: AtomicU64::new(0),
            max_bytes,
        }
    }

    fn charge(&self, bytes: u64) -> Result<()> {
        self.used_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= self.max_bytes)
            })
            .map(|_| ())
            .map_err(|_| {
                anyhow!(
                    "prefetch: materialization exceeds request limit of {} bytes",
                    self.max_bytes
                )
            })
    }
}

#[derive(Debug, Clone)]
enum RequestOutcome {
    Success(RequestSuccess),
    Failure,
}

pub struct HttpDownloader {
    config: PrefetchConfig,
    http_client: Result<Client, String>,
    verified_http_client: Result<Client, String>,
}

impl Default for HttpDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpDownloader {
    pub fn new() -> Self {
        Self::with_config(PrefetchConfig::from_env())
    }

    pub fn with_config(config: PrefetchConfig) -> Self {
        let http_client = build_http_client(&config);
        let verified_http_client = build_verified_http_client(&config);
        Self {
            config,
            http_client,
            verified_http_client,
        }
    }

    pub fn config(&self) -> &PrefetchConfig {
        &self.config
    }

    pub async fn materialize_plan(
        &self,
        plan: &[PrefetchPlanItem],
        cache_root: &Path,
        run_id: &str,
    ) -> Result<MaterializationResult> {
        let requests = build_requests(plan, cache_root, self.config())?;
        let overall_start = std::time::Instant::now();
        let semaphore = Arc::new(Semaphore::new(
            self.config().effective_max_concurrent_files(),
        ));
        let request_budget = Arc::new(RequestBudget::new(self.config.max_request_bytes));

        info!(
            run_id = %run_id,
            plan_items = plan.len(),
            unique_requests = requests.len(),
            "starting generic prefetch materialization"
        );

        let tasks: Vec<_> = requests
            .values()
            .map(|request| {
                let request = request.clone();
                let semaphore = semaphore.clone();
                let request_budget = request_budget.clone();
                let config = self.config.clone();
                let client = if request.is_verified() {
                    self.verified_http_client.clone()
                } else {
                    self.http_client.clone()
                };
                tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire_owned()
                        .await
                        .map_err(|e| anyhow!("failed to acquire download permit: {e}"))?;
                    let outcome = materialize_request(
                        &request,
                        &config,
                        client
                            .as_ref()
                            .map_err(|error| anyhow!("HTTP client initialization failed: {error}")),
                        &request_budget,
                    )
                    .await;
                    Ok::<_, anyhow::Error>((request.cache_key.clone(), outcome))
                })
            })
            .collect();

        let joined = join_all(tasks).await;

        let mut outcomes: HashMap<String, RequestOutcome> = HashMap::new();
        let mut downloaded = 0usize;
        let mut cached = 0usize;
        let mut errors = 0usize;
        let mut required_errors = 0usize;
        let mut required_verified_errors = 0usize;
        let mut optional_errors = 0usize;
        let mut total_bytes = 0u64;

        for joined_task in joined {
            let (cache_key, outcome) = joined_task
                .context("prefetch: request worker join failed")?
                .context("prefetch: request worker failed")?;
            let request = requests.get(&cache_key).ok_or_else(|| {
                anyhow!(
                    "prefetch: download request missing for cache key '{}'",
                    cache_key
                )
            })?;

            match &outcome {
                RequestOutcome::Success(success) => {
                    if success.downloaded {
                        downloaded += 1;
                        total_bytes += success.size_bytes;
                    } else {
                        cached += 1;
                    }
                }
                RequestOutcome::Failure => {
                    errors += 1;
                    if request.required {
                        required_errors += 1;
                        if request.is_verified() {
                            required_verified_errors += 1;
                        }
                    } else {
                        optional_errors += 1;
                    }
                }
            }

            outcomes.insert(cache_key, outcome);
        }

        let mut artifacts = Vec::new();
        for item in plan {
            let cache_key = item.effective_cache_key();
            let Some(outcome) = outcomes.get(&cache_key) else {
                return Err(anyhow!(
                    "prefetch: materialization outcome missing for cache key '{}'",
                    cache_key
                ));
            };

            if let RequestOutcome::Success(success) = outcome {
                artifacts.push(MaterializedArtifact {
                    name: item.target_artifact_name.clone(),
                    source_uri: item.source_uri.clone(),
                    storage_path: success.local_path.display().to_string(),
                    size_bytes: success.size_bytes,
                    media_type: item.media_type.clone(),
                    sha256: success.sha256.clone(),
                });
            }
        }

        let total_time = overall_start.elapsed().as_secs_f64();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let throughput_mbps = if total_time > 0.0 {
            total_mb / total_time
        } else {
            0.0
        };

        info!(
            run_id = %run_id,
            downloaded,
            cached,
            errors,
            required_errors,
            required_verified_errors,
            optional_errors,
            total_mb = format!("{total_mb:.2}"),
            throughput_mbps = format!("{throughput_mbps:.2}"),
            "prefetch materialization complete"
        );

        Ok(MaterializationResult {
            stats: DownloadStats {
                downloaded,
                cached,
                errors,
                required_errors,
                required_verified_errors,
                optional_errors,
                total_time_secs: total_time,
                throughput_mbps,
                total_mb,
            },
            artifacts,
        })
    }
}

fn build_requests(
    plan: &[PrefetchPlanItem],
    cache_root: &Path,
    config: &PrefetchConfig,
) -> Result<HashMap<String, DownloadRequest>> {
    let mut requests: HashMap<String, DownloadRequest> = HashMap::new();
    for item in plan {
        validate_plan_item(item, config)?;
        let cache_key = item.effective_cache_key();
        let local_path = cache_path_for_item(item, cache_root);
        let expected_sha256 = item.expected_sha256.as_deref().map(str::to_ascii_lowercase);

        match requests.get_mut(&cache_key) {
            Some(existing) => {
                let conflicts = if expected_sha256.is_some() {
                    existing.byte_range != item.byte_range
                        || existing.expected_sha256 != expected_sha256
                        || existing.expected_size_bytes != item.expected_size_bytes
                } else {
                    existing.source_uri != item.source_uri
                        || existing.byte_range != item.byte_range
                        || existing.headers != item.headers
                };
                if conflicts {
                    return Err(anyhow!(
                        "prefetch: cache key '{}' maps to conflicting download requests",
                        cache_key
                    ));
                }
                existing.required |= item.required;
            }
            None => {
                requests.insert(
                    cache_key.clone(),
                    DownloadRequest {
                        cache_key,
                        kind: item.effective_kind(),
                        source_uri: item.source_uri.clone(),
                        byte_range: item.byte_range.clone(),
                        headers: item.headers.clone(),
                        local_path,
                        required: item.required,
                        expected_sha256,
                        expected_size_bytes: item.expected_size_bytes,
                    },
                );
            }
        }
    }

    let expected_total = requests.values().try_fold(0u64, |total, request| {
        request
            .expected_size_bytes
            .unwrap_or(0)
            .checked_add(total)
            .ok_or_else(|| anyhow!("prefetch: expected request size overflow"))
    })?;
    if expected_total > config.max_request_bytes {
        return Err(anyhow!(
            "prefetch: expected request size {} exceeds limit of {} bytes",
            expected_total,
            config.max_request_bytes
        ));
    }
    Ok(requests)
}

fn validate_plan_item(item: &PrefetchPlanItem, config: &PrefetchConfig) -> Result<()> {
    if item.source_uri.trim().is_empty() {
        return Err(anyhow!("prefetch: plan item source_uri must be non-empty"));
    }
    if item.target_artifact_name.trim().is_empty() {
        return Err(anyhow!(
            "prefetch: plan item target_artifact_name must be non-empty"
        ));
    }
    if let Some(byte_range) = &item.byte_range
        && byte_range.length == 0
    {
        return Err(anyhow!(
            "prefetch: byte_range.length must be greater than zero for '{}'",
            item.target_artifact_name
        ));
    }

    let verification_requested =
        item.expected_sha256.is_some() || item.expected_size_bytes.is_some();
    if verification_requested {
        let expected_sha256 = item.expected_sha256.as_deref().ok_or_else(|| {
            anyhow!(
                "prefetch: expected_sha256 is required with expected_size_bytes for '{}'",
                item.target_artifact_name
            )
        })?;
        validate_sha256(expected_sha256)?;

        let expected_size_bytes = item.expected_size_bytes.ok_or_else(|| {
            anyhow!(
                "prefetch: expected_size_bytes is required with expected_sha256 for '{}'",
                item.target_artifact_name
            )
        })?;
        if expected_size_bytes == 0 {
            return Err(anyhow!(
                "prefetch: expected_size_bytes must be greater than zero for '{}'",
                item.target_artifact_name
            ));
        }
        if expected_size_bytes > config.max_object_bytes {
            return Err(anyhow!(
                "prefetch: expected object size {} exceeds limit of {} bytes for '{}'",
                expected_size_bytes,
                config.max_object_bytes,
                item.target_artifact_name
            ));
        }
        if item.effective_kind() != PrefetchOpKind::HttpFetch {
            return Err(anyhow!(
                "prefetch: integrity verification requires an HTTPS http_fetch source for '{}'",
                item.target_artifact_name
            ));
        }
        if !item.headers.is_empty() {
            return Err(anyhow!(
                "prefetch: custom headers are not allowed for verified HTTPS downloads"
            ));
        }
        let url = Url::parse(&item.source_uri).with_context(|| {
            format!(
                "prefetch: invalid verified source_uri '{}'",
                item.source_uri
            )
        })?;
        validate_verified_source_url(&url, &config.allowed_https_hosts)?;
    }

    match item.effective_kind() {
        PrefetchOpKind::HttpFetch => {
            let _ = source_uri_to_url(&item.source_uri)?;
        }
        PrefetchOpKind::ObjectStoreFetch => {
            let _ = object_store_source_uri_to_url(&item.source_uri)?;
        }
        PrefetchOpKind::FileCopy => {
            let _ = source_uri_to_local_path(&item.source_uri)?;
        }
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "prefetch: expected_sha256 must contain exactly 64 hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_verified_redirect_url(
    url: &Url,
    allowed_hosts: &BTreeSet<String>,
    allowed_signed_redirect_hosts: &BTreeSet<String>,
) -> Result<()> {
    let allow_secret_query = url
        .host_str()
        .map(str::to_ascii_lowercase)
        .is_some_and(|host| allowed_signed_redirect_hosts.contains(&host));
    validate_verified_url(url, allowed_hosts, allow_secret_query)
}

fn validate_verified_source_url(url: &Url, allowed_hosts: &BTreeSet<String>) -> Result<()> {
    validate_verified_url(url, allowed_hosts, false)
}

fn validate_verified_url(
    url: &Url,
    allowed_hosts: &BTreeSet<String>,
    allow_secret_query: bool,
) -> Result<()> {
    if url.scheme() != "https" {
        return Err(anyhow!("prefetch: verified downloads require HTTPS"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(anyhow!(
            "prefetch: verified source URLs must not contain userinfo"
        ));
    }
    if url.fragment().is_some() {
        return Err(anyhow!(
            "prefetch: verified source URLs must not contain fragments"
        ));
    }

    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("prefetch: verified source URL must contain a DNS host"))?
        .to_ascii_lowercase();
    if host.parse::<IpAddr>().is_ok() {
        return Err(anyhow!(
            "prefetch: IP literal hosts are not allowed for verified downloads"
        ));
    }
    if !allowed_hosts.contains(&host) {
        return Err(anyhow!(
            "prefetch: HTTPS host '{}' is not in the configured allowlist",
            host
        ));
    }

    for (key, _) in url.query_pairs() {
        let normalized: String = key
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if !allow_secret_query
            && (matches!(normalized.as_str(), "auth" | "key" | "sig")
                || [
                    "token",
                    "apikey",
                    "accesskey",
                    "accessid",
                    "signature",
                    "credential",
                    "authorization",
                    "password",
                    "secret",
                ]
                .iter()
                .any(|sensitive| normalized.contains(sensitive)))
        {
            return Err(anyhow!(
                "prefetch: verified source URL contains a secret-bearing query parameter"
            ));
        }
    }

    Ok(())
}

fn sanitize_artifact_name(name: &str) -> String {
    let mut sanitized = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        "artifact".to_string()
    } else {
        sanitized
    }
}

fn artifact_extension(item: &PrefetchPlanItem) -> String {
    let source = item
        .source_uri
        .split('?')
        .next()
        .unwrap_or(item.source_uri.as_str());
    let source_path = source.rsplit('/').next().unwrap_or("");
    Path::new(source_path)
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy()))
        .unwrap_or_default()
}

fn cache_path_for_item(item: &PrefetchPlanItem, cache_root: &Path) -> PathBuf {
    if let Some(expected_sha256) = item.expected_sha256.as_deref() {
        return cache_root
            .join("prefetch")
            .join("sha256")
            .join(expected_sha256.to_ascii_lowercase());
    }

    let hash = Sha256::digest(item.effective_cache_key().as_bytes());
    let hash_hex = format!("{hash:x}");
    let name = sanitize_artifact_name(&item.target_artifact_name);
    let extension = artifact_extension(item);
    cache_root
        .join("prefetch")
        .join(format!("{hash_hex}-{name}{extension}"))
}

fn cache_metadata_path(path: &Path) -> PathBuf {
    let mut metadata_path = path.as_os_str().to_os_string();
    metadata_path.push(".metadata.json");
    PathBuf::from(metadata_path)
}

fn cache_lock_path(path: &Path) -> PathBuf {
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

fn unique_temporary_path(path: &Path, suffix: &str) -> PathBuf {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(".{suffix}-{}", uuid::Uuid::new_v4()));
    PathBuf::from(temporary)
}

struct CacheEntryLock {
    _file: std::fs::File,
}

async fn acquire_verified_cache_lock(request: &DownloadRequest) -> Result<CacheEntryLock> {
    let lock_path = cache_lock_path(&request.local_path);
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open cache lock '{}'", lock_path.display()))?;
        file.lock_exclusive()
            .with_context(|| format!("failed to acquire cache lock '{}'", lock_path.display()))?;
        Ok(CacheEntryLock { _file: file })
    })
    .await
    .context("prefetch: cache lock worker failed")?
}

async fn remove_file_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove '{}'", path.display())),
    }
}

async fn hash_file(path: &Path, max_bytes: u64) -> Result<(u64, String)> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("failed to open cached file '{}'", path.display()))?;
    let mut buffer = vec![0u8; 64 * 1024];
    let mut size_bytes = 0u64;
    let mut hasher = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("failed to read cached file '{}'", path.display()))?;
        if read == 0 {
            break;
        }
        size_bytes = size_bytes
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("prefetch: cached object size overflow"))?;
        if size_bytes > max_bytes {
            return Err(anyhow!(
                "prefetch: cached object exceeds limit of {} bytes",
                max_bytes
            ));
        }
        hasher.update(&buffer[..read]);
    }
    Ok((size_bytes, format!("{:x}", hasher.finalize())))
}

async fn invalidate_verified_cache(request: &DownloadRequest) -> Result<()> {
    remove_file_if_present(&request.local_path).await?;
    remove_file_if_present(&cache_metadata_path(&request.local_path)).await
}

async fn write_cache_metadata(path: &Path, metadata: &CacheMetadata) -> Result<()> {
    let temporary_path = unique_temporary_path(path, "part");
    let _temporary_file_guard = TemporaryFileGuard::new(temporary_path.clone());
    let encoded = serde_json::to_vec(metadata)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)
        .await
        .with_context(|| {
            format!(
                "failed to create temporary cache metadata '{}'",
                temporary_path.display()
            )
        })?;
    file.write_all(&encoded).await.with_context(|| {
        format!(
            "failed to write temporary cache metadata '{}'",
            temporary_path.display()
        )
    })?;
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(&temporary_path, path).await.with_context(|| {
        format!(
            "failed to atomically publish cache metadata '{}'",
            path.display()
        )
    })
}

async fn cached_request_success(
    request: &DownloadRequest,
    config: &PrefetchConfig,
    request_budget: &RequestBudget,
) -> Result<Option<RequestSuccess>> {
    if !fs::try_exists(&request.local_path).await.with_context(|| {
        format!(
            "failed to check cache path '{}'",
            request.local_path.display()
        )
    })? {
        if request.is_verified() {
            remove_file_if_present(&cache_metadata_path(&request.local_path)).await?;
        }
        return Ok(None);
    }

    if !request.is_verified() {
        let size_bytes = fs::metadata(&request.local_path)
            .await
            .with_context(|| {
                format!(
                    "failed to stat cached file '{}'",
                    request.local_path.display()
                )
            })?
            .len();
        debug!(path = ?request.local_path, "prefetch cache hit");
        return Ok(Some(RequestSuccess {
            local_path: request.local_path.clone(),
            size_bytes,
            downloaded: false,
            sha256: None,
        }));
    }

    let file_metadata = fs::symlink_metadata(&request.local_path)
        .await
        .with_context(|| {
            format!(
                "failed to inspect verified cache file '{}'",
                request.local_path.display()
            )
        })?;
    if !file_metadata.file_type().is_file()
        || file_metadata.len() > config.max_object_bytes
        || Some(file_metadata.len()) != request.expected_size_bytes
    {
        debug!(path = ?request.local_path, "discarding invalid verified prefetch cache entry");
        invalidate_verified_cache(request).await?;
        return Ok(None);
    }

    let (size_bytes, sha256) = hash_file(&request.local_path, config.max_object_bytes).await?;
    if Some(size_bytes) != request.expected_size_bytes
        || Some(sha256.as_str()) != request.expected_sha256.as_deref()
    {
        debug!(path = ?request.local_path, "discarding verified prefetch cache entry with invalid digest");
        invalidate_verified_cache(request).await?;
        return Ok(None);
    }

    let expected_metadata = CacheMetadata {
        version: 1,
        sha256: sha256.clone(),
        size_bytes,
    };
    let metadata_path = cache_metadata_path(&request.local_path);
    let metadata_matches = match fs::read(&metadata_path).await {
        Ok(metadata_bytes) => serde_json::from_slice::<CacheMetadata>(&metadata_bytes)
            .map(|metadata| metadata == expected_metadata)
            .unwrap_or(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read verified cache metadata '{}'",
                    metadata_path.display()
                )
            });
        }
    };
    if !metadata_matches {
        write_cache_metadata(&metadata_path, &expected_metadata).await?;
    }

    request_budget.charge(size_bytes)?;
    debug!(path = ?request.local_path, "verified prefetch cache hit");
    Ok(Some(RequestSuccess {
        local_path: request.local_path.clone(),
        size_bytes,
        downloaded: false,
        sha256: Some(sha256),
    }))
}

async fn materialize_request(
    request: &DownloadRequest,
    config: &PrefetchConfig,
    client: Result<&Client>,
    request_budget: &RequestBudget,
) -> RequestOutcome {
    let _cache_lock = if request.is_verified() {
        if let Some(parent) = request.local_path.parent()
            && let Err(error) = fs::create_dir_all(parent).await
        {
            error!(path = ?request.local_path, error = %error, "prefetch failed to create cache dir");
            return RequestOutcome::Failure;
        }
        match acquire_verified_cache_lock(request).await {
            Ok(lock) => Some(lock),
            Err(error) => {
                error!(
                    path = ?request.local_path,
                    error = %error,
                    "prefetch failed to lock verified cache path"
                );
                return RequestOutcome::Failure;
            }
        }
    } else {
        None
    };

    match cached_request_success(request, config, request_budget).await {
        Ok(Some(success)) => return RequestOutcome::Success(success),
        Ok(None) => {}
        Err(error) => {
            error!(
                path = ?request.local_path,
                error = %error,
                "prefetch failed to validate cache path"
            );
            return RequestOutcome::Failure;
        }
    }

    if let Some(parent) = request.local_path.parent()
        && let Err(error) = fs::create_dir_all(parent).await
    {
        error!(path = ?request.local_path, error = %error, "prefetch failed to create cache dir");
        return RequestOutcome::Failure;
    }

    match materialize_request_by_kind(request, config, client, request_budget).await {
        Ok((size_bytes, sha256)) => RequestOutcome::Success(RequestSuccess {
            local_path: request.local_path.clone(),
            size_bytes,
            downloaded: true,
            sha256,
        }),
        Err(error) => {
            error!(
                source_uri = %request.source_uri,
                error = %error,
                "prefetch download failed"
            );
            RequestOutcome::Failure
        }
    }
}

async fn materialize_request_by_kind(
    request: &DownloadRequest,
    config: &PrefetchConfig,
    client: Result<&Client>,
    request_budget: &RequestBudget,
) -> Result<(u64, Option<String>)> {
    match request.kind {
        PrefetchOpKind::HttpFetch | PrefetchOpKind::ObjectStoreFetch => {
            download_request(request, config, client?, request_budget).await
        }
        PrefetchOpKind::FileCopy => copy_local_request(request).await,
    }
}

async fn download_request(
    request: &DownloadRequest,
    config: &PrefetchConfig,
    client: &Client,
    request_budget: &RequestBudget,
) -> Result<(u64, Option<String>)> {
    let source_url = match request.kind {
        PrefetchOpKind::HttpFetch => source_uri_to_url(&request.source_uri)?,
        PrefetchOpKind::ObjectStoreFetch => object_store_source_uri_to_url(&request.source_uri)?,
        PrefetchOpKind::FileCopy => {
            return Err(anyhow!(
                "prefetch: file_copy requests must not be routed through HTTP download"
            ));
        }
    };
    let mut http_request = client.get(source_url);
    for (key, value) in &request.headers {
        http_request = http_request.header(key, value);
    }
    if let Some(byte_range) = &request.byte_range {
        let end = byte_range
            .offset
            .checked_add(byte_range.length - 1)
            .ok_or_else(|| anyhow!("prefetch: byte range overflow for '{}'", request.source_uri))?;
        http_request = http_request.header("Range", format!("bytes={}-{}", byte_range.offset, end));
    }

    let response = http_request.send().await;
    let response = if request.is_verified() {
        let response = response.map_err(|error| {
            let category = if error.is_timeout() {
                "timeout"
            } else if error.is_connect() {
                "connection failure"
            } else {
                "transport failure"
            };
            anyhow!(
                "prefetch: verified HTTPS {category} for '{}'",
                request.source_uri
            )
        })?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "prefetch: verified HTTPS status {} downloading '{}'",
                response.status(),
                request.source_uri
            ));
        }
        response
    } else {
        response
            .with_context(|| format!("failed to send request for '{}'", request.source_uri))?
            .error_for_status()
            .with_context(|| format!("HTTP error downloading '{}'", request.source_uri))?
    };
    stream_response_to_cache(request, config, response, request_budget).await
}

async fn stream_response_to_cache(
    request: &DownloadRequest,
    config: &PrefetchConfig,
    response: reqwest::Response,
    request_budget: &RequestBudget,
) -> Result<(u64, Option<String>)> {
    if request.is_verified()
        && let Some(content_length) = response.content_length()
    {
        if content_length > config.max_object_bytes {
            return Err(anyhow!(
                "prefetch: content length {} exceeds object limit of {} bytes",
                content_length,
                config.max_object_bytes
            ));
        }
        if let Some(expected_size_bytes) = request.expected_size_bytes
            && content_length != expected_size_bytes
        {
            return Err(anyhow!(
                "prefetch: content length {} does not match expected size {}",
                content_length,
                expected_size_bytes
            ));
        }
    }

    let temporary_path = unique_temporary_path(&request.local_path, "part");
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)
        .await
        .with_context(|| {
            format!(
                "failed to create temporary file '{}'",
                temporary_path.display()
            )
        })?;
    let _temporary_file_guard = TemporaryFileGuard::new(temporary_path.clone());
    let result = async {
        let mut stream = response.bytes_stream();
        let mut size_bytes = 0u64;
        let mut hasher = Sha256::new();
        while let Some(chunk) = stream.next().await {
            let chunk = if request.is_verified() {
                chunk.map_err(|_| {
                    anyhow!(
                        "prefetch: failed to read verified HTTPS body for '{}'",
                        request.source_uri
                    )
                })?
            } else {
                chunk
                    .with_context(|| format!("failed to read body for '{}'", request.source_uri))?
            };
            size_bytes = size_bytes
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| anyhow!("prefetch: object size overflow"))?;
            if request.is_verified() {
                if let Some(expected_size_bytes) = request.expected_size_bytes
                    && size_bytes > expected_size_bytes
                {
                    return Err(anyhow!(
                        "prefetch: streamed object exceeds expected size of {} bytes",
                        expected_size_bytes
                    ));
                }
                if size_bytes > config.max_object_bytes {
                    return Err(anyhow!(
                        "prefetch: streamed object exceeds limit of {} bytes",
                        config.max_object_bytes
                    ));
                }
                request_budget.charge(chunk.len() as u64)?;
                hasher.update(&chunk);
            }
            file.write_all(&chunk).await.with_context(|| {
                format!(
                    "failed to write temporary file '{}'",
                    temporary_path.display()
                )
            })?;
        }

        let sha256 = request
            .is_verified()
            .then(|| format!("{:x}", hasher.finalize()));
        verify_download(request, size_bytes, sha256.as_deref())?;
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        publish_cache_file(request, &temporary_path, size_bytes, sha256.as_deref()).await?;
        Ok((size_bytes, sha256))
    }
    .await;

    if result.is_err() {
        remove_file_if_present(&temporary_path).await?;
    }
    result
}

fn verify_download(request: &DownloadRequest, size_bytes: u64, sha256: Option<&str>) -> Result<()> {
    if let Some(expected_size_bytes) = request.expected_size_bytes
        && size_bytes != expected_size_bytes
    {
        return Err(anyhow!(
            "prefetch: downloaded size {} does not match expected size {}",
            size_bytes,
            expected_size_bytes
        ));
    }
    if let Some(expected_sha256) = request.expected_sha256.as_deref()
        && sha256 != Some(expected_sha256)
    {
        return Err(anyhow!(
            "prefetch: downloaded SHA-256 does not match expected digest"
        ));
    }
    Ok(())
}

async fn publish_cache_file(
    request: &DownloadRequest,
    temporary_path: &Path,
    size_bytes: u64,
    sha256: Option<&str>,
) -> Result<()> {
    let metadata_path = cache_metadata_path(&request.local_path);
    if let Err(error) = fs::rename(temporary_path, &request.local_path).await {
        return Err(error).with_context(|| {
            format!(
                "failed to atomically publish cache file '{}'",
                request.local_path.display()
            )
        });
    }

    if request.is_verified() {
        let metadata = CacheMetadata {
            version: 1,
            sha256: sha256
                .ok_or_else(|| anyhow!("prefetch: missing verified SHA-256"))?
                .to_string(),
            size_bytes,
        };
        write_cache_metadata(&metadata_path, &metadata).await?;
    }
    Ok(())
}

async fn copy_local_request(request: &DownloadRequest) -> Result<(u64, Option<String>)> {
    let source_path = source_uri_to_local_path(&request.source_uri)?;
    let temporary_path = unique_temporary_path(&request.local_path, "part");
    let mut source = fs::File::open(&source_path).await?;
    let mut destination = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)
        .await?;
    let _temporary_file_guard = TemporaryFileGuard::new(temporary_path.clone());
    let result = async {
        let mut buffer = vec![0u8; 64 * 1024];
        let mut size_bytes = 0u64;
        loop {
            let read = source.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            size_bytes = size_bytes
                .checked_add(read as u64)
                .ok_or_else(|| anyhow!("prefetch: local object size overflow"))?;
            destination.write_all(&buffer[..read]).await?;
        }
        destination.flush().await?;
        destination.sync_all().await?;
        drop(destination);
        publish_cache_file(request, &temporary_path, size_bytes, None).await?;
        Ok((size_bytes, None))
    }
    .await;
    if result.is_err() {
        remove_file_if_present(&temporary_path).await?;
    }
    result
}

fn source_uri_to_url(source_uri: &str) -> Result<String> {
    if source_uri.starts_with("https://") || source_uri.starts_with("http://") {
        return Ok(source_uri.to_string());
    }

    object_store_source_uri_to_url(source_uri)
}

fn object_store_source_uri_to_url(source_uri: &str) -> Result<String> {
    if let Some(path) = source_uri.strip_prefix("s3://") {
        let mut parts = path.splitn(2, '/');
        let bucket = parts
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("prefetch: invalid s3 source_uri '{}'", source_uri))?;
        let key = parts.next().unwrap_or("");
        if key.is_empty() {
            return Ok(format!("https://{bucket}.s3.amazonaws.com"));
        }
        return Ok(format!("https://{bucket}.s3.amazonaws.com/{key}"));
    }

    if let Some(path) = source_uri.strip_prefix("gs://") {
        let mut parts = path.splitn(2, '/');
        let bucket = parts
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("prefetch: invalid gs source_uri '{}'", source_uri))?;
        let key = parts.next().unwrap_or("");
        if key.is_empty() {
            return Ok(format!("https://storage.googleapis.com/{bucket}"));
        }
        return Ok(format!("https://storage.googleapis.com/{bucket}/{key}"));
    }

    if let Some(path) = source_uri
        .strip_prefix("az://")
        .or_else(|| source_uri.strip_prefix("azure://"))
    {
        let mut parts = path.splitn(3, '/');
        let account = parts
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("prefetch: invalid azure source_uri '{}'", source_uri))?;
        let container = parts
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("prefetch: invalid azure source_uri '{}'", source_uri))?;
        let blob = parts.next().unwrap_or("");
        if blob.is_empty() {
            return Ok(format!(
                "https://{account}.blob.core.windows.net/{container}"
            ));
        }
        return Ok(format!(
            "https://{account}.blob.core.windows.net/{container}/{blob}"
        ));
    }

    Err(anyhow!(
        "prefetch: unsupported source_uri '{}'; expected s3://, gs://, az://, or azure://",
        source_uri
    ))
}

fn source_uri_to_local_path(source_uri: &str) -> Result<PathBuf> {
    let raw = source_uri
        .strip_prefix("file://")
        .unwrap_or(source_uri)
        .trim();
    if raw.is_empty() {
        return Err(anyhow!("prefetch: local file source_uri must be non-empty"));
    }
    Ok(PathBuf::from(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    async fn spawn_http_server(
        chunks: Vec<Vec<u8>>,
        content_length: Option<u64>,
        delay_between_chunks: Duration,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request_buffer = vec![0u8; 4096];
            let _ = socket.read(&mut request_buffer).await;

            if let Some(content_length) = content_length {
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                for chunk in chunks {
                    if socket.write_all(&chunk).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(delay_between_chunks).await;
                }
            } else {
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
                for chunk in chunks {
                    if socket
                        .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                        .await
                        .is_err()
                    {
                        return;
                    }
                    if socket.write_all(&chunk).await.is_err()
                        || socket.write_all(b"\r\n").await.is_err()
                    {
                        return;
                    }
                    tokio::time::sleep(delay_between_chunks).await;
                }
                let _ = socket.write_all(b"0\r\n\r\n").await;
            }
        });
        (format!("http://{address}/artifact.bin"), handle)
    }

    fn legacy_http_item(source_uri: String) -> PrefetchPlanItem {
        PrefetchPlanItem {
            kind: Some(PrefetchOpKind::HttpFetch),
            source_uri,
            target_artifact_name: "artifact".to_string(),
            required: true,
            byte_range: None,
            cache_key: None,
            media_type: Some("application/octet-stream".to_string()),
            expected_sha256: None,
            expected_size_bytes: None,
            headers: BTreeMap::new(),
        }
    }

    fn verified_item(source_uri: &str, contents: &[u8]) -> PrefetchPlanItem {
        PrefetchPlanItem {
            kind: Some(PrefetchOpKind::HttpFetch),
            source_uri: source_uri.to_string(),
            target_artifact_name: "mesh".to_string(),
            required: true,
            byte_range: None,
            cache_key: Some("ignored-for-verified-content".to_string()),
            media_type: Some("application/vnd.vtk".to_string()),
            expected_sha256: Some(format!("{:x}", Sha256::digest(contents))),
            expected_size_bytes: Some(contents.len() as u64),
            headers: BTreeMap::new(),
        }
    }

    fn verified_config(host: &str) -> PrefetchConfig {
        PrefetchConfig {
            allowed_https_hosts: BTreeSet::from([host.to_string()]),
            ..PrefetchConfig::default()
        }
    }

    fn only_request(
        item: &PrefetchPlanItem,
        cache_root: &Path,
        config: &PrefetchConfig,
    ) -> DownloadRequest {
        build_requests(std::slice::from_ref(item), cache_root, config)
            .unwrap()
            .into_values()
            .next()
            .unwrap()
    }

    async fn assert_no_partial_files(directory: &Path) {
        let mut entries = fs::read_dir(directory).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            assert!(
                !entry.file_name().to_string_lossy().contains(".part-"),
                "partial cache file was not cleaned up: {:?}",
                entry.path()
            );
        }
    }

    #[test]
    fn source_uri_to_url_supports_https() {
        let url = source_uri_to_url("https://example.com/data.bin").unwrap();
        assert_eq!(url, "https://example.com/data.bin");
    }

    #[test]
    fn source_uri_to_url_converts_s3_scheme() {
        let url = source_uri_to_url("s3://bucket/path/data.bin").unwrap();
        assert_eq!(url, "https://bucket.s3.amazonaws.com/path/data.bin");
    }

    #[test]
    fn source_uri_to_url_converts_gcs_scheme() {
        let url = source_uri_to_url("gs://bucket/path/data.bin").unwrap();
        assert_eq!(url, "https://storage.googleapis.com/bucket/path/data.bin");
    }

    #[test]
    fn source_uri_to_url_converts_azure_blob_scheme() {
        let url = source_uri_to_url("az://account/container/path/data.bin").unwrap();
        assert_eq!(
            url,
            "https://account.blob.core.windows.net/container/path/data.bin"
        );
    }

    #[test]
    fn source_uri_to_url_rejects_unknown_scheme() {
        let err = source_uri_to_url("file:///tmp/data.bin").unwrap_err();
        assert!(err.to_string().contains("unsupported source_uri"));
    }

    #[test]
    fn cache_path_for_item_uses_hashed_prefix_and_safe_name() {
        let dir = tempdir().unwrap();
        let item = PrefetchPlanItem {
            kind: Some(super::super::plan::PrefetchOpKind::ObjectStoreFetch),
            source_uri: "s3://bucket/path/data.bin".to_string(),
            target_artifact_name: "../unsafe name".to_string(),
            required: true,
            byte_range: None,
            cache_key: None,
            media_type: None,
            expected_sha256: None,
            expected_size_bytes: None,
            headers: BTreeMap::new(),
        };

        let path = cache_path_for_item(&item, dir.path());
        let file_name = path.file_name().unwrap().to_string_lossy();

        assert!(path.starts_with(dir.path().join("prefetch")));
        assert!(file_name.contains(".._unsafe_name"));
        assert!(file_name.ends_with(".bin"));
    }

    #[tokio::test]
    async fn legacy_cached_artifact_ignores_verified_download_limits() {
        let dir = tempdir().unwrap();
        let item = PrefetchPlanItem {
            kind: Some(super::super::plan::PrefetchOpKind::HttpFetch),
            source_uri: "https://example.com/reference.txt".to_string(),
            target_artifact_name: "reference".to_string(),
            required: false,
            byte_range: None,
            cache_key: None,
            media_type: Some("text/plain".to_string()),
            expected_sha256: None,
            expected_size_bytes: None,
            headers: BTreeMap::new(),
        };
        let local_path = cache_path_for_item(&item, dir.path());
        fs::create_dir_all(local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&local_path, b"prefetched").await.unwrap();

        let downloader = HttpDownloader::with_config(PrefetchConfig {
            max_object_bytes: 1,
            max_request_bytes: 1,
            ..PrefetchConfig::default()
        });
        let result = downloader
            .materialize_plan(&[item], dir.path(), "run-1")
            .await
            .unwrap();

        assert_eq!(result.stats.cached, 1);
        assert_eq!(result.stats.downloaded, 0);
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].size_bytes, 10);
        assert_eq!(result.artifacts[0].sha256, None);
    }

    #[tokio::test]
    async fn materialize_plan_copies_local_file_for_file_copy_kind() {
        let dir = tempdir().unwrap();
        let source_path = dir.path().join("input.txt");
        fs::write(&source_path, b"local-copy").await.unwrap();

        let item = PrefetchPlanItem {
            kind: Some(super::super::plan::PrefetchOpKind::FileCopy),
            source_uri: source_path.display().to_string(),
            target_artifact_name: "copied-input".to_string(),
            required: true,
            byte_range: None,
            cache_key: None,
            media_type: Some("text/plain".to_string()),
            expected_sha256: None,
            expected_size_bytes: None,
            headers: BTreeMap::new(),
        };

        let downloader = HttpDownloader::with_config(PrefetchConfig::default());
        let result = downloader
            .materialize_plan(&[item], dir.path(), "run-1")
            .await
            .unwrap();

        assert_eq!(result.stats.downloaded, 1);
        assert_eq!(result.stats.cached, 0);
        assert_eq!(result.artifacts.len(), 1);
        let artifact_path = PathBuf::from(&result.artifacts[0].storage_path);
        assert!(artifact_path.exists());
        let contents = fs::read_to_string(&artifact_path).await.unwrap();
        assert_eq!(contents, "local-copy");
    }

    #[tokio::test]
    async fn legacy_chunked_stream_ignores_verified_limits_and_publishes_atomically() {
        let dir = tempdir().unwrap();
        let (source_uri, server) = spawn_http_server(
            vec![b"chunk-one".to_vec(), b"-chunk-two".to_vec()],
            None,
            Duration::from_millis(5),
        )
        .await;
        let item = legacy_http_item(source_uri);
        let local_path = cache_path_for_item(&item, dir.path());
        let downloader = HttpDownloader::with_config(PrefetchConfig {
            max_object_bytes: 1,
            max_request_bytes: 1,
            ..PrefetchConfig::default()
        });

        let result = downloader
            .materialize_plan(&[item], dir.path(), "run-chunked")
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(result.stats.downloaded, 1);
        assert_eq!(fs::read(&local_path).await.unwrap(), b"chunk-one-chunk-two");
        assert_no_partial_files(local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn keeps_streamed_bytes_in_temporary_file_until_atomic_publication() {
        let dir = tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (first_chunk_sent, first_chunk_received) = tokio::sync::oneshot::channel();
        let (continue_send, continue_receive) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request_buffer = vec![0u8; 4096];
            let _ = socket.read(&mut request_buffer).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nfirst\r\n",
                )
                .await
                .unwrap();
            socket.flush().await.unwrap();
            first_chunk_sent.send(()).unwrap();
            continue_receive.await.unwrap();
            socket.write_all(b"6\r\nsecond\r\n0\r\n\r\n").await.unwrap();
        });

        let item = legacy_http_item(format!("http://{address}/artifact.bin"));
        let local_path = cache_path_for_item(&item, dir.path());
        let cache_root = dir.path().to_path_buf();
        let materialization = tokio::spawn(async move {
            HttpDownloader::with_config(PrefetchConfig {
                max_object_bytes: 1024,
                max_request_bytes: 1024,
                ..PrefetchConfig::default()
            })
            .materialize_plan(&[item], &cache_root, "run-atomic")
            .await
        });

        first_chunk_received.await.unwrap();
        let cache_directory = local_path.parent().unwrap();
        let mut saw_partial_file = false;
        for _ in 0..50 {
            if fs::try_exists(cache_directory).await.unwrap_or(false) {
                let mut entries = fs::read_dir(cache_directory).await.unwrap();
                while let Some(entry) = entries.next_entry().await.unwrap() {
                    saw_partial_file |= entry.file_name().to_string_lossy().contains(".part-");
                }
            }
            if saw_partial_file {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(
            saw_partial_file,
            "streaming should write to a temporary file"
        );
        assert!(
            !local_path.exists(),
            "final cache path must stay hidden until the stream is verified and complete"
        );

        continue_send.send(()).unwrap();
        let result = materialization.await.unwrap().unwrap();
        server.await.unwrap();
        assert_eq!(result.stats.downloaded, 1);
        assert_eq!(fs::read(&local_path).await.unwrap(), b"firstsecond");
        assert_no_partial_files(cache_directory).await;
    }

    #[tokio::test]
    async fn rejects_content_length_over_object_limit_without_partial_file() {
        let dir = tempdir().unwrap();
        let (source_uri, server) =
            spawn_http_server(vec![b"short".to_vec()], Some(100), Duration::ZERO).await;
        let item = verified_item("https://assets.example.com/mesh.vtp", b"short");
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();
        let limited_config = PrefetchConfig {
            max_object_bytes: 10,
            max_request_bytes: 100,
            ..config
        };

        let error = stream_response_to_cache(
            &request,
            &limited_config,
            response,
            &RequestBudget::new(100),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("object limit"));
        assert!(!request.local_path.exists());
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn rejects_chunked_body_that_crosses_midstream_limit_and_cleans_partial_file() {
        let dir = tempdir().unwrap();
        let (source_uri, server) = spawn_http_server(
            vec![b"12345".to_vec(), b"67890".to_vec()],
            None,
            Duration::ZERO,
        )
        .await;
        let contents = b"1234567890";
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();
        let limited_config = PrefetchConfig {
            max_object_bytes: 8,
            max_request_bytes: 100,
            ..config
        };

        let error = stream_response_to_cache(
            &request,
            &limited_config,
            response,
            &RequestBudget::new(100),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("streamed object exceeds"));
        assert!(!request.local_path.exists());
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn verified_stream_writes_sidecar_and_returns_digest() {
        let dir = tempdir().unwrap();
        let contents = b"verified-mesh";
        let (source_uri, server) =
            spawn_http_server(vec![contents.to_vec()], None, Duration::ZERO).await;
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();

        let (size_bytes, sha256) =
            stream_response_to_cache(&request, &config, response, &RequestBudget::new(1024))
                .await
                .unwrap();
        server.await.unwrap();

        assert_eq!(size_bytes, contents.len() as u64);
        assert_eq!(sha256, item.expected_sha256);
        assert_eq!(fs::read(&request.local_path).await.unwrap(), contents);
        let metadata: CacheMetadata = serde_json::from_slice(
            &fs::read(cache_metadata_path(&request.local_path))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata.sha256, sha256.unwrap());
        assert_eq!(metadata.size_bytes, contents.len() as u64);
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn hash_mismatch_removes_data_and_sidecar_temporary_files() {
        let dir = tempdir().unwrap();
        let contents = b"actual-mesh";
        let (source_uri, server) =
            spawn_http_server(vec![contents.to_vec()], None, Duration::ZERO).await;
        let mut item = verified_item("https://assets.example.com/mesh.vtp", contents);
        item.expected_sha256 = Some("0".repeat(64));
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();

        let error =
            stream_response_to_cache(&request, &config, response, &RequestBudget::new(1024))
                .await
                .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("SHA-256"));
        assert!(!request.local_path.exists());
        assert!(!cache_metadata_path(&request.local_path).exists());
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn chunked_size_mismatch_removes_partial_file() {
        let dir = tempdir().unwrap();
        let contents = b"mesh";
        let (source_uri, server) =
            spawn_http_server(vec![contents.to_vec()], None, Duration::ZERO).await;
        let mut item = verified_item("https://assets.example.com/mesh.vtp", contents);
        item.expected_size_bytes = Some(contents.len() as u64 + 1);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();

        let error =
            stream_response_to_cache(&request, &config, response, &RequestBudget::new(1024))
                .await
                .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("expected size"));
        assert!(!request.local_path.exists());
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[tokio::test]
    async fn chunked_body_stops_when_it_exceeds_expected_size() {
        let dir = tempdir().unwrap();
        let contents = b"1234567890";
        let (source_uri, server) = spawn_http_server(
            vec![b"12345".to_vec(), b"67890".to_vec()],
            None,
            Duration::ZERO,
        )
        .await;
        let mut item = verified_item("https://assets.example.com/mesh.vtp", contents);
        item.expected_size_bytes = Some(5);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        let response = Client::new().get(source_uri).send().await.unwrap();

        let error =
            stream_response_to_cache(&request, &config, response, &RequestBudget::new(1024))
                .await
                .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("exceeds expected size"));
        assert!(!request.local_path.exists());
        assert_no_partial_files(request.local_path.parent().unwrap()).await;
    }

    #[test]
    fn verified_cache_path_is_digest_only_and_reused_across_artifact_names() {
        let dir = tempdir().unwrap();
        let contents = b"shared-mesh";
        let first = verified_item("https://assets.example.com/one.vtp", contents);
        let mut second = verified_item("https://assets.example.com/two.vtp", contents);
        second.target_artifact_name = "different-name".to_string();

        let first_path = cache_path_for_item(&first, dir.path());
        let second_path = cache_path_for_item(&second, dir.path());
        assert_eq!(first_path, second_path);
        assert_eq!(
            first_path,
            dir.path()
                .join("prefetch")
                .join("sha256")
                .join(first.expected_sha256.as_deref().unwrap())
        );

        let config = verified_config("assets.example.com");
        let requests = build_requests(&[first, second], dir.path(), &config).unwrap();
        assert_eq!(
            requests.len(),
            1,
            "identical digests should share one fetch"
        );
    }

    #[tokio::test]
    async fn concurrent_cache_hits_repair_missing_sidecar_without_deleting_data() {
        let dir = tempdir().unwrap();
        let contents = b"published-before-sidecar";
        let first = verified_item("https://assets.example.com/one.vtp", contents);
        let mut second = verified_item("https://assets.example.com/two.vtp", contents);
        second.target_artifact_name = "same-content-different-name".to_string();
        let config = verified_config("assets.example.com");
        let first_request = only_request(&first, dir.path(), &config);
        let second_request = only_request(&second, dir.path(), &config);
        assert_eq!(first_request.local_path, second_request.local_path);
        fs::create_dir_all(first_request.local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&first_request.local_path, contents)
            .await
            .unwrap();
        assert!(!cache_metadata_path(&first_request.local_path).exists());

        let first_budget = RequestBudget::new(config.max_request_bytes);
        let second_budget = RequestBudget::new(config.max_request_bytes);
        let (first_hit, second_hit) = tokio::join!(
            cached_request_success(&first_request, &config, &first_budget),
            cached_request_success(&second_request, &config, &second_budget),
        );

        assert!(first_hit.unwrap().is_some());
        assert!(second_hit.unwrap().is_some());
        assert_eq!(fs::read(&first_request.local_path).await.unwrap(), contents);
        let metadata: CacheMetadata = serde_json::from_slice(
            &fs::read(cache_metadata_path(&first_request.local_path))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata.sha256, first.expected_sha256.unwrap());
        assert_eq!(metadata.size_bytes, contents.len() as u64);
    }

    #[tokio::test]
    async fn verified_cache_hit_repairs_stale_sidecar_after_rehashing_data() {
        let dir = tempdir().unwrap();
        let contents = b"valid-digest-addressed-data";
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&request.local_path, contents).await.unwrap();
        fs::write(
            cache_metadata_path(&request.local_path),
            br#"{"version":0,"sha256":"stale","size_bytes":1}"#,
        )
        .await
        .unwrap();

        let hit = cached_request_success(
            &request,
            &config,
            &RequestBudget::new(config.max_request_bytes),
        )
        .await
        .unwrap();

        assert!(hit.is_some());
        assert_eq!(fs::read(&request.local_path).await.unwrap(), contents);
        let repaired: CacheMetadata = serde_json::from_slice(
            &fs::read(cache_metadata_path(&request.local_path))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(repaired.sha256, item.expected_sha256.unwrap());
        assert_eq!(repaired.size_bytes, contents.len() as u64);
    }

    #[tokio::test]
    async fn verified_cache_hit_rehashes_sidecar_and_returns_digest() {
        let dir = tempdir().unwrap();
        let contents = b"cached-verified-mesh";
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&request.local_path, contents).await.unwrap();
        fs::write(
            cache_metadata_path(&request.local_path),
            serde_json::to_vec(&CacheMetadata {
                version: 1,
                sha256: item.expected_sha256.clone().unwrap(),
                size_bytes: contents.len() as u64,
            })
            .unwrap(),
        )
        .await
        .unwrap();

        let downloader = HttpDownloader::with_config(config);
        let result = downloader
            .materialize_plan(std::slice::from_ref(&item), dir.path(), "run-cache-hit")
            .await
            .unwrap();

        assert_eq!(result.stats.cached, 1);
        assert_eq!(result.stats.downloaded, 0);
        assert_eq!(result.artifacts[0].sha256, item.expected_sha256);
    }

    #[tokio::test]
    async fn corrupted_verified_cache_is_rejected_and_removed() {
        let dir = tempdir().unwrap();
        let contents = b"good";
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&request.local_path, b"evil").await.unwrap();
        fs::write(
            cache_metadata_path(&request.local_path),
            serde_json::to_vec(&CacheMetadata {
                version: 1,
                sha256: item.expected_sha256.clone().unwrap(),
                size_bytes: contents.len() as u64,
            })
            .unwrap(),
        )
        .await
        .unwrap();

        let cached = cached_request_success(
            &request,
            &config,
            &RequestBudget::new(config.max_request_bytes),
        )
        .await
        .unwrap();

        assert!(cached.is_none());
        assert!(!request.local_path.exists());
        assert!(!cache_metadata_path(&request.local_path).exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn verified_materialization_does_not_invalidate_concurrent_publication() {
        let dir = tempdir().unwrap();
        let contents = b"good";
        let item = verified_item("https://assets.example.com/mesh.vtp", contents);
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(request.local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&request.local_path, b"evil").await.unwrap();

        let publisher_lock = acquire_verified_cache_lock(&request).await.unwrap();
        let cache_root = dir.path().to_path_buf();
        let materialized_item = item.clone();
        let materialized_config = config.clone();
        let mut materialization = tokio::spawn(async move {
            HttpDownloader::with_config(materialized_config)
                .materialize_plan(&[materialized_item], &cache_root, "run-concurrent-cache")
                .await
        });

        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut materialization)
                .await
                .is_err(),
            "cache validation must wait for the digest lock"
        );

        let temporary_path = unique_temporary_path(&request.local_path, "test-publish");
        fs::write(&temporary_path, contents).await.unwrap();
        publish_cache_file(
            &request,
            &temporary_path,
            contents.len() as u64,
            item.expected_sha256.as_deref(),
        )
        .await
        .unwrap();
        drop(publisher_lock);

        let result = tokio::time::timeout(Duration::from_secs(2), materialization)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(result.stats.cached, 1);
        assert_eq!(result.stats.downloaded, 0);
        assert_eq!(fs::read(&request.local_path).await.unwrap(), contents);
        assert!(cache_metadata_path(&request.local_path).exists());
    }

    #[tokio::test]
    async fn required_verified_failures_are_tracked_separately() {
        let dir = tempdir().unwrap();
        let item = verified_item("https://assets.example.com/mesh.vtp", b"good");
        let config = verified_config("assets.example.com");
        let request = only_request(&item, dir.path(), &config);
        fs::create_dir_all(&request.local_path).await.unwrap();

        let result = HttpDownloader::with_config(config)
            .materialize_plan(&[item], dir.path(), "run-verified-failure")
            .await
            .unwrap();

        assert_eq!(result.stats.errors, 1);
        assert_eq!(result.stats.required_errors, 1);
        assert_eq!(result.stats.required_verified_errors, 1);
    }

    #[test]
    fn verified_url_policy_requires_exact_allowlisted_dns_https_host() {
        let allowed = BTreeSet::from(["assets.example.com".to_string()]);
        validate_verified_source_url(
            &Url::parse("https://assets.example.com/mesh.vtp?version=2").unwrap(),
            &allowed,
        )
        .unwrap();

        for rejected in [
            "http://assets.example.com/mesh.vtp",
            "https://sub.assets.example.com/mesh.vtp",
            "https://127.0.0.1/mesh.vtp",
            "https://user@assets.example.com/mesh.vtp",
            "https://assets.example.com/mesh.vtp#fragment",
            "https://assets.example.com/mesh.vtp?X-Amz-Signature=secret",
            "https://assets.example.com/mesh.vtp?sig=azure-secret",
        ] {
            assert!(
                validate_verified_source_url(&Url::parse(rejected).unwrap(), &allowed).is_err(),
                "URL should be rejected, including as a redirect target: {rejected}"
            );
        }
    }

    #[test]
    fn signed_queries_are_allowed_only_on_explicit_redirect_hosts() {
        let allowed = BTreeSet::from([
            "assets.example.com".to_string(),
            "huggingface.co".to_string(),
            "us.aws.cdn.hf.co".to_string(),
        ]);
        let signed_redirect_hosts = BTreeSet::from([
            "signed-only.example.com".to_string(),
            "us.aws.cdn.hf.co".to_string(),
        ]);
        let canonical = Url::parse(
            "https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/boundary_1.vtp",
        )
        .unwrap();
        let signed_redirect = Url::parse(
            "https://us.aws.cdn.hf.co/object?Policy=opaque&Signature=opaque&Key-Pair-Id=key",
        )
        .unwrap();

        validate_verified_source_url(&canonical, &allowed).unwrap();
        assert!(validate_verified_source_url(&signed_redirect, &allowed).is_err());
        validate_verified_redirect_url(&signed_redirect, &allowed, &signed_redirect_hosts).unwrap();

        let signed_source =
            Url::parse("https://us.aws.cdn.hf.co/object?Signature=client-supplied").unwrap();
        assert!(validate_verified_source_url(&signed_source, &allowed).is_err());

        let untrusted_redirect =
            Url::parse("https://assets.example.com/object?Signature=opaque").unwrap();
        assert!(
            validate_verified_redirect_url(&untrusted_redirect, &allowed, &signed_redirect_hosts,)
                .is_err()
        );

        let signed_but_not_generally_allowed =
            Url::parse("https://signed-only.example.com/object?Signature=opaque").unwrap();
        assert!(
            validate_verified_redirect_url(
                &signed_but_not_generally_allowed,
                &allowed,
                &signed_redirect_hosts,
            )
            .is_err()
        );

        let signed_subdomain =
            Url::parse("https://sub.us.aws.cdn.hf.co/object?Signature=opaque").unwrap();
        assert!(
            validate_verified_redirect_url(&signed_subdomain, &allowed, &signed_redirect_hosts)
                .is_err()
        );

        let config = PrefetchConfig {
            allowed_https_hosts: allowed,
            allowed_signed_redirect_hosts: signed_redirect_hosts,
            ..PrefetchConfig::default()
        };
        let direct_signed_item = verified_item(signed_source.as_str(), b"mesh");
        assert!(validate_plan_item(&direct_signed_item, &config).is_err());
    }

    #[test]
    fn verified_plan_rejects_partial_integrity_fields_and_custom_headers() {
        let config = verified_config("assets.example.com");
        let mut item = verified_item("https://assets.example.com/mesh.vtp", b"mesh");
        item.expected_size_bytes = None;
        assert!(validate_plan_item(&item, &config).is_err());

        item.expected_size_bytes = Some(4);
        item.headers.insert("authorization".into(), "secret".into());
        assert!(validate_plan_item(&item, &config).is_err());
    }

    #[test]
    fn expected_sizes_enforce_aggregate_request_limit_before_download() {
        let dir = tempdir().unwrap();
        let mut config = verified_config("assets.example.com");
        config.max_object_bytes = 10;
        config.max_request_bytes = 7;
        let first = verified_item("https://assets.example.com/one.vtp", b"1234");
        let mut second = verified_item("https://assets.example.com/two.vtp", b"5678");
        second.target_artifact_name = "mesh-two".to_string();

        let error = build_requests(&[first, second], dir.path(), &config).unwrap_err();
        assert!(error.to_string().contains("expected request size"));
    }
}
