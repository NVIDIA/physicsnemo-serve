/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Download infrastructure configuration for prefetch operations.
//!
//! Controls HTTP concurrency, timeouts, rate limiting, and cache settings.
//! Transport/engine concerns (read_block_ms, read_count, handoff_stream) live
//! in [`crate::config::PrefetchRoleConfig`] / [`crate::config::RuntimeConfig`].

use std::path::PathBuf;
use std::time::Duration;
use tracing::warn;

/// Download infrastructure configuration for prefetch operations.
///
/// All settings can be overridden via environment variables.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Download concurrency for parallel HTTP requests.
    pub download_concurrency: usize,

    /// Download timeout in seconds.
    pub download_timeout_secs: u64,

    /// Maximum concurrent prefetch requests to process simultaneously.
    pub max_concurrent_prefetch_requests: usize,

    /// S3 rate limit (requests per second).
    pub s3_rate_limit_per_sec: u32,

    /// Maximum concurrent files to download (auto-calculated from rate limit if `None`).
    pub max_concurrent_files: Option<usize>,

    /// HTTP connection pool max idle connections per host.
    pub http_pool_max_idle_per_host: usize,

    /// HTTP connection pool idle timeout in seconds.
    pub http_pool_idle_timeout_secs: u64,

    /// Cache directory for downloaded data.
    pub ext_cache_dir: PathBuf,

    /// Index cache TTL in days.
    pub idx_cache_ttl_days: i64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            download_concurrency: 32,
            download_timeout_secs: 120,
            max_concurrent_prefetch_requests: 10,
            s3_rate_limit_per_sec: 500,
            max_concurrent_files: None,
            http_pool_max_idle_per_host: 50,
            http_pool_idle_timeout_secs: 90,
            ext_cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("e2s_ext"),
            idx_cache_ttl_days: 7,
        }
    }
}

impl PrefetchConfig {
    fn parse_env_usize(var_name: &str) -> Option<usize> {
        std::env::var(var_name).ok().and_then(|s| s.parse().ok())
    }

    fn parse_env_u64(var_name: &str) -> Option<u64> {
        std::env::var(var_name).ok().and_then(|s| s.parse().ok())
    }

    fn parse_env_u32(var_name: &str) -> Option<u32> {
        std::env::var(var_name).ok().and_then(|s| s.parse().ok())
    }

    fn validate_non_zero_usize(
        value: usize,
        default_value: usize,
        field_name: &'static str,
        env_name: &'static str,
    ) -> usize {
        if value == 0 {
            warn!(
                field = field_name,
                env_var = env_name,
                default_value,
                "prefetch config value is 0, falling back to default"
            );
            default_value
        } else {
            value
        }
    }

    fn validate_non_zero_u64(
        value: u64,
        default_value: u64,
        field_name: &'static str,
        env_name: &'static str,
    ) -> u64 {
        if value == 0 {
            warn!(
                field = field_name,
                env_var = env_name,
                default_value,
                "prefetch config value is 0, falling back to default"
            );
            default_value
        } else {
            value
        }
    }

    fn validate_non_zero_u32(
        value: u32,
        default_value: u32,
        field_name: &'static str,
        env_name: &'static str,
    ) -> u32 {
        if value == 0 {
            warn!(
                field = field_name,
                env_var = env_name,
                default_value,
                "prefetch config value is 0, falling back to default"
            );
            default_value
        } else {
            value
        }
    }

    fn validate(mut self) -> Self {
        let default = Self::default();
        self.download_concurrency = Self::validate_non_zero_usize(
            self.download_concurrency,
            default.download_concurrency,
            "download_concurrency",
            "E2S_DOWNLOAD_CONCURRENCY",
        );
        self.download_timeout_secs = Self::validate_non_zero_u64(
            self.download_timeout_secs,
            default.download_timeout_secs,
            "download_timeout_secs",
            "E2S_DOWNLOAD_TIMEOUT_SECS",
        );
        self.max_concurrent_prefetch_requests = Self::validate_non_zero_usize(
            self.max_concurrent_prefetch_requests,
            default.max_concurrent_prefetch_requests,
            "max_concurrent_prefetch_requests",
            "E2S_MAX_CONCURRENT_PREFETCH",
        );
        self.s3_rate_limit_per_sec = Self::validate_non_zero_u32(
            self.s3_rate_limit_per_sec,
            default.s3_rate_limit_per_sec,
            "s3_rate_limit_per_sec",
            "E2S_S3_RATE_LIMIT_PER_SEC",
        );
        self
    }

    /// Load configuration from environment variables, falling back to defaults.
    ///
    /// # Environment Variables
    ///
    /// | Variable | Default | Description |
    /// |----------|---------|-------------|
    /// | `E2S_DOWNLOAD_CONCURRENCY` | 32 | Parallel download threads |
    /// | `E2S_DOWNLOAD_TIMEOUT_SECS` | 120 | HTTP request timeout |
    /// | `E2S_MAX_CONCURRENT_PREFETCH` | 10 | Max concurrent prefetch jobs |
    /// | `E2S_S3_RATE_LIMIT_PER_SEC` | 500 | S3 request rate limit |
    /// | `MAX_CONCURRENT_FILES` | (auto) | Override file concurrency |
    /// | `E2S_HTTP_POOL_MAX_IDLE` | 50 | HTTP pool idle connections |
    /// | `E2S_HTTP_POOL_TIMEOUT_SECS` | 90 | HTTP pool timeout |
    /// | `E2S_EXT_CACHE` | ~/.cache/e2s_ext | Cache directory |
    /// | `E2S_IDX_CACHE_TTL_DAYS` | 7 | Index cache TTL |
    pub fn from_env() -> Self {
        let default = Self::default();

        Self {
            download_concurrency: Self::parse_env_usize("E2S_DOWNLOAD_CONCURRENCY")
                .unwrap_or(default.download_concurrency),

            download_timeout_secs: Self::parse_env_u64("E2S_DOWNLOAD_TIMEOUT_SECS")
                .unwrap_or(default.download_timeout_secs),

            max_concurrent_prefetch_requests: Self::parse_env_usize("E2S_MAX_CONCURRENT_PREFETCH")
                .unwrap_or(default.max_concurrent_prefetch_requests),

            s3_rate_limit_per_sec: Self::parse_env_u32("E2S_S3_RATE_LIMIT_PER_SEC")
                .unwrap_or(default.s3_rate_limit_per_sec),

            max_concurrent_files: std::env::var("MAX_CONCURRENT_FILES")
                .ok()
                .and_then(|s| s.parse().ok()),

            http_pool_max_idle_per_host: std::env::var("E2S_HTTP_POOL_MAX_IDLE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default.http_pool_max_idle_per_host),

            http_pool_idle_timeout_secs: std::env::var("E2S_HTTP_POOL_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default.http_pool_idle_timeout_secs),

            ext_cache_dir: std::env::var("E2S_EXT_CACHE")
                .ok()
                .map(PathBuf::from)
                .unwrap_or(default.ext_cache_dir),

            idx_cache_ttl_days: std::env::var("E2S_IDX_CACHE_TTL_DAYS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default.idx_cache_ttl_days),
        }
        .validate()
    }

    /// Get the HTTP timeout as a Duration.
    pub fn http_timeout(&self) -> Duration {
        Duration::from_secs(self.download_timeout_secs)
    }

    /// Get the HTTP pool idle timeout as a Duration.
    pub fn http_pool_timeout(&self) -> Duration {
        Duration::from_secs(self.http_pool_idle_timeout_secs)
    }

    /// Get the index cache TTL in seconds.
    #[cfg(test)]
    pub(crate) fn idx_cache_ttl_secs(&self) -> i64 {
        self.idx_cache_ttl_days * 24 * 60 * 60
    }

    /// Calculate effective max concurrent files.
    ///
    /// If not explicitly set, calculates from S3 rate limit:
    /// `(rate_limit * 0.9) / avg_chunks_per_file`
    pub fn effective_max_concurrent_files(&self) -> usize {
        if let Some(explicit) = self.max_concurrent_files {
            return explicit;
        }

        let effective_limit = (self.s3_rate_limit_per_sec as usize * 9) / 10;
        let avg_chunks_per_file = 26;
        (effective_limit / avg_chunks_per_file).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::with_env_var;

    #[test]
    fn defaults_set_download_concurrency_and_timeout() {
        let config = PrefetchConfig::default();
        assert_eq!(config.download_concurrency, 32);
        assert_eq!(config.download_timeout_secs, 120);
    }

    #[test]
    fn defaults_set_s3_rate_limit_and_pool_settings() {
        let config = PrefetchConfig::default();
        assert_eq!(config.s3_rate_limit_per_sec, 500);
        assert_eq!(config.http_pool_max_idle_per_host, 50);
        assert_eq!(config.http_pool_idle_timeout_secs, 90);
    }

    #[test]
    fn http_timeout_converts_seconds_to_duration() {
        let config = PrefetchConfig::default();
        assert_eq!(config.http_timeout(), Duration::from_secs(120));
        assert_eq!(config.http_pool_timeout(), Duration::from_secs(90));
    }

    #[test]
    fn idx_cache_ttl_converts_days_to_seconds() {
        let config = PrefetchConfig::default();
        assert_eq!(config.idx_cache_ttl_secs(), 7 * 24 * 60 * 60);
    }

    #[test]
    fn effective_max_concurrent_files_uses_explicit_when_set() {
        let config = PrefetchConfig {
            max_concurrent_files: Some(100),
            ..Default::default()
        };
        assert_eq!(config.effective_max_concurrent_files(), 100);
    }

    #[test]
    fn effective_max_concurrent_files_auto_calculates_from_rate_limit() {
        let config = PrefetchConfig::default();
        // (500 * 0.9) / 26 = 450 / 26 = 17
        assert_eq!(config.effective_max_concurrent_files(), 17);
    }

    #[test]
    fn cache_dir_defaults_to_e2s_ext_suffix() {
        let config = PrefetchConfig::default();
        assert!(config.ext_cache_dir.to_string_lossy().contains("e2s_ext"));
    }

    #[test]
    fn from_env_zero_download_concurrency_falls_back_to_default() {
        let config = with_env_var(
            "E2S_DOWNLOAD_CONCURRENCY",
            Some("0"),
            PrefetchConfig::from_env,
        );
        assert_eq!(
            config.download_concurrency,
            PrefetchConfig::default().download_concurrency
        );
    }

    #[test]
    fn from_env_zero_download_timeout_falls_back_to_default() {
        let config = with_env_var(
            "E2S_DOWNLOAD_TIMEOUT_SECS",
            Some("0"),
            PrefetchConfig::from_env,
        );
        assert_eq!(
            config.download_timeout_secs,
            PrefetchConfig::default().download_timeout_secs
        );
    }

    #[test]
    fn from_env_zero_prefetch_concurrency_falls_back_to_default() {
        let config = with_env_var(
            "E2S_MAX_CONCURRENT_PREFETCH",
            Some("0"),
            PrefetchConfig::from_env,
        );
        assert_eq!(
            config.max_concurrent_prefetch_requests,
            PrefetchConfig::default().max_concurrent_prefetch_requests
        );
    }

    #[test]
    fn from_env_zero_s3_rate_limit_falls_back_to_default() {
        let config = with_env_var(
            "E2S_S3_RATE_LIMIT_PER_SEC",
            Some("0"),
            PrefetchConfig::from_env,
        );
        assert_eq!(
            config.s3_rate_limit_per_sec,
            PrefetchConfig::default().s3_rate_limit_per_sec
        );
    }
}
