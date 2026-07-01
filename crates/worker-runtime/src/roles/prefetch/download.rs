/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::sync::Semaphore;
use tracing::{debug, error, info};

use super::plan::{ByteRange, PrefetchOpKind, PrefetchPlanItem};
use super::prefetch_config::PrefetchConfig;

static HTTP_CLIENT: LazyLock<Result<reqwest::Client, String>> = LazyLock::new(|| {
    let config = PrefetchConfig::from_env();
    reqwest::Client::builder()
        .http2_adaptive_window(true)
        .pool_max_idle_per_host(config.http_pool_max_idle_per_host)
        .pool_idle_timeout(config.http_pool_timeout())
        .timeout(config.http_timeout())
        .build()
        .map_err(|e| e.to_string())
});

fn http_client() -> Result<&'static reqwest::Client> {
    HTTP_CLIENT
        .as_ref()
        .map_err(|e| anyhow!("HTTP client initialization failed: {e}"))
}

#[derive(Debug, Clone, Default)]
#[must_use]
pub struct DownloadStats {
    pub downloaded: usize,
    pub cached: usize,
    pub errors: usize,
    pub required_errors: usize,
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
}

#[derive(Debug, Clone)]
struct RequestSuccess {
    local_path: PathBuf,
    size_bytes: u64,
    downloaded: bool,
}

#[derive(Debug, Clone)]
enum RequestOutcome {
    Success(RequestSuccess),
    Failure { required: bool },
}

pub struct HttpDownloader {
    config: PrefetchConfig,
}

impl Default for HttpDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpDownloader {
    pub fn new() -> Self {
        Self {
            config: PrefetchConfig::from_env(),
        }
    }

    pub fn with_config(config: PrefetchConfig) -> Self {
        Self { config }
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
        let requests = build_requests(plan, cache_root)?;
        let overall_start = std::time::Instant::now();
        let semaphore = Arc::new(Semaphore::new(
            self.config().effective_max_concurrent_files(),
        ));

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
                tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire_owned()
                        .await
                        .map_err(|e| anyhow!("failed to acquire download permit: {e}"))?;
                    let outcome = materialize_request(&request).await;
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
        let mut optional_errors = 0usize;
        let mut total_bytes = 0u64;

        for joined_task in joined {
            let (cache_key, outcome) = joined_task
                .context("prefetch: request worker join failed")?
                .context("prefetch: request worker failed")?;

            match &outcome {
                RequestOutcome::Success(success) => {
                    if success.downloaded {
                        downloaded += 1;
                        total_bytes += success.size_bytes;
                    } else {
                        cached += 1;
                    }
                }
                RequestOutcome::Failure { required } => {
                    errors += 1;
                    if *required {
                        required_errors += 1;
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
) -> Result<HashMap<String, DownloadRequest>> {
    let mut requests: HashMap<String, DownloadRequest> = HashMap::new();
    for item in plan {
        validate_plan_item(item)?;
        let cache_key = item.effective_cache_key();
        let local_path = cache_path_for_item(item, cache_root);

        match requests.get_mut(&cache_key) {
            Some(existing) => {
                if existing.source_uri != item.source_uri
                    || existing.byte_range != item.byte_range
                    || existing.headers != item.headers
                {
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
                    },
                );
            }
        }
    }
    Ok(requests)
}

fn validate_plan_item(item: &PrefetchPlanItem) -> Result<()> {
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
    let cache_key = item.effective_cache_key();
    let hash = Sha256::digest(cache_key.as_bytes());
    let hash_hex = format!("{hash:x}");
    let name = sanitize_artifact_name(&item.target_artifact_name);
    let extension = artifact_extension(item);
    cache_root
        .join("prefetch")
        .join(format!("{hash_hex}-{name}{extension}"))
}

async fn materialize_request(request: &DownloadRequest) -> RequestOutcome {
    match fs::try_exists(&request.local_path).await {
        Ok(true) => match fs::metadata(&request.local_path).await {
            Ok(metadata) => {
                debug!(path = ?request.local_path, "prefetch cache hit");
                return RequestOutcome::Success(RequestSuccess {
                    local_path: request.local_path.clone(),
                    size_bytes: metadata.len(),
                    downloaded: false,
                });
            }
            Err(error) => {
                error!(
                    path = ?request.local_path,
                    error = %error,
                    "prefetch failed to stat cached file"
                );
                return RequestOutcome::Failure {
                    required: request.required,
                };
            }
        },
        Ok(false) => {}
        Err(error) => {
            error!(
                path = ?request.local_path,
                error = %error,
                "prefetch failed to check cache path"
            );
            return RequestOutcome::Failure {
                required: request.required,
            };
        }
    }

    if let Some(parent) = request.local_path.parent()
        && let Err(error) = fs::create_dir_all(parent).await
    {
        error!(path = ?request.local_path, error = %error, "prefetch failed to create cache dir");
        return RequestOutcome::Failure {
            required: request.required,
        };
    }

    match materialize_request_by_kind(request).await {
        Ok(size_bytes) => RequestOutcome::Success(RequestSuccess {
            local_path: request.local_path.clone(),
            size_bytes,
            downloaded: true,
        }),
        Err(error) => {
            error!(
                source_uri = %request.source_uri,
                error = %error,
                "prefetch download failed"
            );
            RequestOutcome::Failure {
                required: request.required,
            }
        }
    }
}

async fn materialize_request_by_kind(request: &DownloadRequest) -> Result<u64> {
    match request.kind {
        PrefetchOpKind::HttpFetch | PrefetchOpKind::ObjectStoreFetch => {
            download_request(request).await
        }
        PrefetchOpKind::FileCopy => copy_local_request(request).await,
    }
}

async fn download_request(request: &DownloadRequest) -> Result<u64> {
    let client = http_client()?;
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

    let response = http_request
        .send()
        .await
        .with_context(|| format!("failed to send request for '{}'", request.source_uri))?;
    let response = response
        .error_for_status()
        .with_context(|| format!("HTTP error downloading '{}'", request.source_uri))?;
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("failed to read body for '{}'", request.source_uri))?;

    fs::write(&request.local_path, &bytes)
        .await
        .with_context(|| {
            format!(
                "failed to write cache file '{}'",
                request.local_path.display()
            )
        })?;
    Ok(bytes.len() as u64)
}

async fn copy_local_request(request: &DownloadRequest) -> Result<u64> {
    let source_path = source_uri_to_local_path(&request.source_uri)?;
    let copied = fs::copy(&source_path, &request.local_path)
        .await
        .with_context(|| {
            format!(
                "failed to copy local source '{}' into '{}'",
                source_path.display(),
                request.local_path.display()
            )
        })?;
    Ok(copied)
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
    use tempfile::tempdir;

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
            headers: BTreeMap::new(),
        };

        let path = cache_path_for_item(&item, dir.path());
        let file_name = path.file_name().unwrap().to_string_lossy();

        assert!(path.starts_with(dir.path().join("prefetch")));
        assert!(file_name.contains(".._unsafe_name"));
        assert!(file_name.ends_with(".bin"));
    }

    #[tokio::test]
    async fn materialize_plan_returns_cached_artifact_for_existing_file() {
        let dir = tempdir().unwrap();
        let item = PrefetchPlanItem {
            kind: Some(super::super::plan::PrefetchOpKind::HttpFetch),
            source_uri: "https://example.com/reference.txt".to_string(),
            target_artifact_name: "reference".to_string(),
            required: false,
            byte_range: None,
            cache_key: None,
            media_type: Some("text/plain".to_string()),
            headers: BTreeMap::new(),
        };
        let local_path = cache_path_for_item(&item, dir.path());
        fs::create_dir_all(local_path.parent().unwrap())
            .await
            .unwrap();
        fs::write(&local_path, b"prefetched").await.unwrap();

        let downloader = HttpDownloader::with_config(PrefetchConfig::default());
        let result = downloader
            .materialize_plan(&[item], dir.path(), "run-1")
            .await
            .unwrap();

        assert_eq!(result.stats.cached, 1);
        assert_eq!(result.stats.downloaded, 0);
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].size_bytes, 10);
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
}
