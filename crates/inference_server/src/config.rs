/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Server configuration

use crate::plugin_registry::PythonRuntimeEnvConfig;
use anyhow::{Context, anyhow};
use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Default server port
pub const DEFAULT_PORT: &str = "8080";
const ENV_RUNTIME_CONFIG: &str = "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG";
const ENV_OUTPUT_PUBLICATION_CONFIG_JSON: &str = "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputPublicationProvider {
    S3,
    Azure,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputPublicationStorageConfig {
    S3 {
        bucket: String,
        #[serde(default = "default_publication_prefix")]
        prefix: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint: Option<String>,
    },
    Azure {
        container: String,
        #[serde(default = "default_publication_prefix")]
        prefix: String,
        endpoint: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OutputPublicationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<OutputPublicationStorageConfig>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResolvedOutputPublicationStorage {
    S3 {
        bucket: String,
        prefix: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint: Option<String>,
    },
    Azure {
        container: String,
        prefix: String,
        endpoint: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedOutputPublicationTarget {
    pub artifact: String,
    pub provider: OutputPublicationProvider,
    pub storage: ResolvedOutputPublicationStorage,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedOutputPublication {
    pub target: ResolvedOutputPublicationTarget,
}

fn default_publication_prefix() -> String {
    "outputs".to_string()
}

fn publication_prefix(prefix: &str, workflow_id: &str, run_id: &str) -> String {
    [prefix.trim_matches('/'), workflow_id.trim(), run_id.trim()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

impl OutputPublicationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| anyhow!("output_publication.enabled=true requires a storage config"))?;
        match storage {
            OutputPublicationStorageConfig::S3 {
                bucket,
                prefix,
                region,
                endpoint,
            } => {
                if bucket.trim().is_empty() {
                    return Err(anyhow!(
                        "output_publication.storage.bucket must be non-empty"
                    ));
                }
                if prefix.trim_matches('/').trim().is_empty() {
                    return Err(anyhow!(
                        "output_publication.storage.prefix must be non-empty"
                    ));
                }
                if region
                    .as_deref()
                    .is_some_and(|value| value.trim().is_empty())
                {
                    return Err(anyhow!(
                        "output_publication.storage.region must be non-empty"
                    ));
                }
                if let Some(endpoint) = endpoint.as_deref() {
                    let endpoint = endpoint.trim();
                    if endpoint.is_empty() {
                        return Err(anyhow!(
                            "output_publication.storage.endpoint must be non-empty"
                        ));
                    }
                    let endpoint_url = reqwest::Url::parse(endpoint).map_err(|_| {
                        anyhow!("output_publication.storage.endpoint for s3 must be a valid URL")
                    })?;
                    if !matches!(endpoint_url.scheme(), "http" | "https") {
                        return Err(anyhow!(
                            "output_publication.storage.endpoint for s3 must start with http:// or https://"
                        ));
                    }
                }
            }
            OutputPublicationStorageConfig::Azure {
                container,
                prefix,
                endpoint,
            } => {
                if container.trim().is_empty() {
                    return Err(anyhow!(
                        "output_publication.storage.container must be non-empty"
                    ));
                }
                if container.trim() != container {
                    return Err(anyhow!(
                        "output_publication.storage.container must not contain surrounding whitespace"
                    ));
                }
                if container.contains('/') || container.contains('\\') {
                    return Err(anyhow!(
                        "output_publication.storage.container must not contain path separators"
                    ));
                }
                if prefix.trim_matches('/').trim().is_empty() {
                    return Err(anyhow!(
                        "output_publication.storage.prefix must be non-empty"
                    ));
                }
                let endpoint = endpoint.trim();
                if endpoint.is_empty() {
                    return Err(anyhow!(
                        "output_publication.storage.endpoint must be non-empty"
                    ));
                }
                let endpoint_url = reqwest::Url::parse(endpoint).map_err(|_| {
                    anyhow!("output_publication.storage.endpoint for azure must be a valid URL")
                })?;
                if endpoint_url.scheme() != "https" {
                    return Err(anyhow!(
                        "output_publication.storage.endpoint for azure must start with https://"
                    ));
                }
                if endpoint_url.path() != "/"
                    || endpoint_url.query().is_some()
                    || endpoint_url.fragment().is_some()
                {
                    return Err(anyhow!(
                        "output_publication.storage.endpoint for azure must be an account-root URL without a path, query, or fragment"
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn resolve_for_workflow(
        &self,
        workflow_id: &str,
        run_id: &str,
        _request_fields: &serde_json::Value,
    ) -> anyhow::Result<Option<ResolvedOutputPublication>> {
        if !self.enabled {
            return Ok(None);
        };
        self.validate()?;
        let storage = self
            .storage
            .as_ref()
            .expect("validated enabled config has storage");
        let target = match storage {
            OutputPublicationStorageConfig::S3 {
                bucket,
                prefix,
                region,
                endpoint,
            } => ResolvedOutputPublicationTarget {
                artifact: "primary".to_string(),
                provider: OutputPublicationProvider::S3,
                storage: ResolvedOutputPublicationStorage::S3 {
                    bucket: bucket.trim().to_string(),
                    prefix: publication_prefix(prefix, workflow_id, run_id),
                    region: region
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                    endpoint: endpoint
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                },
            },
            OutputPublicationStorageConfig::Azure {
                container,
                prefix,
                endpoint,
            } => ResolvedOutputPublicationTarget {
                artifact: "primary".to_string(),
                provider: OutputPublicationProvider::Azure,
                storage: ResolvedOutputPublicationStorage::Azure {
                    container: container.trim_matches('/').to_string(),
                    prefix: publication_prefix(prefix, workflow_id, run_id),
                    endpoint: endpoint.trim_end_matches('/').to_string(),
                },
            },
        };
        Ok(Some(ResolvedOutputPublication { target }))
    }
}

/// Server configuration
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Server listening address
    pub addr: SocketAddr,
    /// Redis connection URL
    pub redis_url: String,
    /// Redis stream for queuing inference requests (direct inference, no prefetch)
    pub redis_stream: String,
    /// Redis stream for prefetch requests (prefetch -> inference pipeline)
    pub prefetch_stream: String,
    /// Whether to use prefetch pipeline (default: true)
    pub use_prefetch: bool,
    /// Directories containing manifest-driven plugin definitions
    pub plugin_dirs: Vec<PathBuf>,
    /// Optional manifest metadata.id for the only enabled plugin in this deployment.
    pub enabled_plugin_id: Option<String>,
    /// Root directory used for staging uploaded artifacts
    pub artifact_dir: PathBuf,
    /// Root directory used for generated workflow outputs served by the results endpoint
    pub default_output_dir: PathBuf,
    /// Retention window for staged and generated artifacts under `artifact_dir`
    pub artifact_retention_hours: u64,
    /// Interval in seconds for periodic artifact cleanup (default: 30).
    ///
    /// Workflow manifests and schema contracts are loaded once at startup.
    pub artifact_cleanup_interval_secs: u64,
    /// Allowed CORS origins. Empty means allow all (dev mode).
    pub cors_allowed_origins: Vec<String>,
    /// Maximum request body size in bytes (default: 256 MiB)
    pub max_body_size: usize,
    /// Prefix prepended to stream names when enqueuing plugin runs (default: empty).
    /// Must match the `stream_prefix` in the worker-runtime config.
    pub stream_prefix: String,
    /// Swagger UI CDN base URL. `None` uses the default unpkg CDN.
    pub swagger_cdn_url: Option<String>,
    /// Runtime-env registry keyed by executor_class for readiness probing.
    pub python_runtime_envs: HashMap<String, PythonRuntimeEnvConfig>,
    pub output_publication: OutputPublicationConfig,
}

impl ServerConfig {
    /// Create configuration from environment variables.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_env_vars(&|key| env::var(key).ok())
    }

    /// Return a display string with sensitive fields redacted.
    ///
    /// Replaces the password portion of Redis URLs (`redis://:PASSWORD@...`)
    /// with `***` so credentials are never emitted into logs.
    pub fn redacted_display(&self) -> String {
        let redacted_url = redact_redis_url(&self.redis_url);
        format!(
            "ServerConfig {{ addr: {}, redis_url: {}, redis_stream: {}, \
             prefetch_stream: {}, use_prefetch: {}, plugin_dirs: {:?}, enabled_plugin_id: {:?}, artifact_dir: {:?}, default_output_dir: {:?}, artifact_retention_hours: {}, \
             artifact_cleanup_interval_secs: {}, cors_allowed_origins: {:?}, \
             max_body_size: {} }}",
            self.addr,
            redacted_url,
            self.redis_stream,
            self.prefetch_stream,
            self.use_prefetch,
            self.plugin_dirs,
            self.enabled_plugin_id,
            self.artifact_dir,
            self.default_output_dir,
            self.artifact_retention_hours,
            self.artifact_cleanup_interval_secs,
            self.cors_allowed_origins,
            self.max_body_size,
        )
    }

    /// Create configuration from a custom environment variable provider (for testing).
    fn from_env_vars<F>(env_provider: &F) -> anyhow::Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let port = env_provider("PORT").unwrap_or_else(|| DEFAULT_PORT.to_string());
        let addr: SocketAddr = format!("0.0.0.0:{}", port)
            .parse()
            .with_context(|| format!("invalid PORT value: {}", port))?;

        let redis_url = if let Some(url) = env_provider("REDIS_URL") {
            url
        } else {
            let host = env_provider("REDIS_HOST").unwrap_or_else(|| "127.0.0.1".to_string());
            let port = env_provider("REDIS_PORT").unwrap_or_else(|| "6379".to_string());
            let db = env_provider("REDIS_DB").unwrap_or_else(|| "0".to_string());
            let password = env_provider("REDIS_PASSWORD").unwrap_or_default();

            if !password.is_empty() {
                format!("redis://:{}@{}:{}/{}", password, host, port, db)
            } else {
                format!("redis://{}:{}/{}", host, port, db)
            }
        };

        let redis_stream =
            env_provider("INFERENCE_STREAM").unwrap_or_else(|| "inference".to_string());

        let prefetch_stream =
            env_provider("PREFETCH_STREAM").unwrap_or_else(|| "prefetch".to_string());

        let use_prefetch = env_provider("USE_PREFETCH")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        let plugin_dirs = crate::plugin_registry::resolve_plugin_dirs(env_provider);
        let enabled_plugin_id = env_provider("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let artifact_dir = env_provider("ARTIFACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("artifacts"));
        let default_output_dir = env_provider("DEFAULT_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("outputs"));
        let artifact_retention_hours = env_provider("ARTIFACT_RETENTION_HOURS")
            .unwrap_or_else(|| "24".to_string())
            .parse()
            .context("Invalid ARTIFACT_RETENTION_HOURS, must be a number")?;

        let artifact_cleanup_interval_secs = env_provider("ARTIFACT_CLEANUP_INTERVAL_SECS")
            .or_else(|| env_provider("WORKFLOW_DISCOVERY_INTERVAL"))
            .unwrap_or_else(|| "30".to_string())
            .parse()
            .context("Invalid ARTIFACT_CLEANUP_INTERVAL_SECS, must be a number")?;

        let cors_allowed_origins = env_provider("CORS_ALLOWED_ORIGINS")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        const DEFAULT_MAX_BODY: usize = 256 * 1024 * 1024; // 256 MiB
        let max_body_size = env_provider("MAX_BODY_SIZE")
            .map(|v| v.parse().context("Invalid MAX_BODY_SIZE, must be a number"))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_BODY);

        let swagger_cdn_url = env_provider("SWAGGER_CDN_URL");
        let (python_runtime_envs, output_publication) =
            resolve_server_runtime_config(env_provider)?;
        let stream_prefix = env_provider("REDIS_STREAM_PREFIX").unwrap_or_default();

        Ok(Self {
            addr,
            redis_url,
            redis_stream,
            prefetch_stream,
            use_prefetch,
            plugin_dirs,
            enabled_plugin_id,
            artifact_dir,
            default_output_dir,
            artifact_retention_hours,
            artifact_cleanup_interval_secs,
            cors_allowed_origins,
            max_body_size,
            stream_prefix,
            swagger_cdn_url,
            python_runtime_envs,
            output_publication,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct ServerRuntimeConfigFile {
    #[serde(default)]
    python_runtime_envs: HashMap<String, PythonRuntimeEnvConfig>,
    #[serde(default)]
    output_publication: OutputPublicationConfig,
}

/// Redact the password from a Redis URL.
///
/// `redis://:SECRET@host:6379/0` becomes `redis://:***@host:6379/0`.
/// URLs without a password are returned unchanged.
fn redact_redis_url(url: &str) -> String {
    if let Some(at_pos) = url.find('@')
        && let Some(colon_colon) = url.find("//:")
    {
        let pw_start = colon_colon + 3;
        return format!("{}***{}", &url[..pw_start], &url[at_pos..]);
    }
    url.to_string()
}

fn resolve_server_runtime_config<F>(
    env_provider: &F,
) -> anyhow::Result<(
    HashMap<String, PythonRuntimeEnvConfig>,
    OutputPublicationConfig,
)>
where
    F: Fn(&str) -> Option<String>,
{
    let output_publication_override = resolve_output_publication_config_override(env_provider)?;
    let Some(path) = env_provider(ENV_RUNTIME_CONFIG).map(PathBuf::from) else {
        return Ok((
            HashMap::new(),
            output_publication_override.unwrap_or_default(),
        ));
    };

    let contents = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "failed to read {ENV_RUNTIME_CONFIG} file '{}'",
            path.display()
        )
    })?;
    let parsed: ServerRuntimeConfigFile = serde_json::from_str(&contents).with_context(|| {
        format!(
            "failed to parse {ENV_RUNTIME_CONFIG} file '{}'",
            path.display()
        )
    })?;
    let output_publication = output_publication_override.unwrap_or(parsed.output_publication);
    output_publication.validate().with_context(|| {
        format!(
            "invalid output_publication config in {ENV_RUNTIME_CONFIG} file '{}'",
            path.display()
        )
    })?;
    Ok((parsed.python_runtime_envs, output_publication))
}

fn resolve_output_publication_config_override<F>(
    env_provider: &F,
) -> anyhow::Result<Option<OutputPublicationConfig>>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(raw) = env_provider(ENV_OUTPUT_PUBLICATION_CONFIG_JSON) else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let config: OutputPublicationConfig = serde_json::from_str(&raw).with_context(|| {
        format!("failed to parse {ENV_OUTPUT_PUBLICATION_CONFIG_JSON} as output_publication JSON")
    })?;
    config
        .validate()
        .with_context(|| format!("invalid {ENV_OUTPUT_PUBLICATION_CONFIG_JSON}"))?;
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Helper to create a test environment provider from a list of key-value pairs
    fn create_test_env(vars: Vec<(&str, &str)>) -> impl Fn(&str) -> Option<String> {
        let mut env_map = HashMap::new();
        for (key, value) in vars {
            env_map.insert(key.to_string(), value.to_string());
        }
        move |key: &str| env_map.get(key).cloned()
    }

    /// Helper to parse a .env file and return a HashMap of key-value pairs
    fn parse_env_file(content: &str) -> HashMap<String, String> {
        let mut env_map = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Parse KEY=VALUE
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();
                if !value.is_empty() {
                    env_map.insert(key, value);
                }
            }
        }
        env_map
    }

    #[test]
    fn test_redacted_display_hides_redis_password() {
        let env_provider = create_test_env(vec![
            ("REDIS_HOST", "myhost"),
            ("REDIS_PORT", "6380"),
            ("REDIS_DB", "2"),
            ("REDIS_PASSWORD", "supersecret"),
        ]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        let display = config.redacted_display();

        assert!(
            !display.contains("supersecret"),
            "Redacted display must not contain password. Got: {}",
            display
        );
        assert!(
            display.contains("myhost"),
            "Redacted display should contain host. Got: {}",
            display
        );
        assert!(
            display.contains("***"),
            "Redacted display should contain redaction marker. Got: {}",
            display
        );
    }

    #[test]
    fn test_redacted_display_no_password_shows_url() {
        let env_provider = create_test_env(vec![("REDIS_URL", "redis://localhost:6379/0")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        let display = config.redacted_display();

        assert!(
            display.contains("redis://localhost:6379/0"),
            "URL without password should be shown. Got: {}",
            display
        );
    }

    #[test]
    fn test_from_env_defaults() {
        let env_provider = create_test_env(vec![]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(config.addr.port(), 8080);
        assert_eq!(config.addr.ip(), std::net::Ipv4Addr::new(0, 0, 0, 0));
        assert_eq!(config.redis_url, "redis://127.0.0.1:6379/0");
        assert_eq!(config.redis_stream, "inference");
        assert_eq!(config.prefetch_stream, "prefetch");
        assert!(config.use_prefetch);
        assert_eq!(config.artifact_cleanup_interval_secs, 30);
        assert!(config.plugin_dirs.is_empty());
        assert_eq!(config.enabled_plugin_id, None);
        assert_eq!(config.artifact_dir, PathBuf::from("artifacts"));
        assert_eq!(config.default_output_dir, PathBuf::from("outputs"));
        assert_eq!(config.artifact_retention_hours, 24);
        assert!(config.cors_allowed_origins.is_empty());
        assert_eq!(config.max_body_size, 256 * 1024 * 1024);
        assert!(config.python_runtime_envs.is_empty());
        assert_eq!(
            config.output_publication,
            OutputPublicationConfig::default()
        );
    }

    #[test]
    fn test_output_publication_config_loaded_from_runtime_config() {
        let temp_path = std::env::temp_dir().join(format!(
            "physicsnemo-serve-runtime-publication-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &temp_path,
            r#"{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "s3",
      "bucket": "forecast-bucket",
      "prefix": "outputs",
      "region": "us-east-1",
      "endpoint": "https://s3.us-east-1.amazonaws.com"
    }
  }
}"#,
        )
        .unwrap();

        let temp_path_string = temp_path.to_string_lossy().to_string();
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG",
            temp_path_string.as_str(),
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        let resolved = config
            .output_publication
            .resolve_for_workflow("earth2-deterministic", "run-123", &serde_json::json!({}))
            .unwrap()
            .expect("profile should resolve");

        assert_eq!(resolved.target.artifact, "primary");
        assert_eq!(resolved.target.provider, OutputPublicationProvider::S3);
        assert_eq!(
            resolved.target.storage,
            ResolvedOutputPublicationStorage::S3 {
                bucket: "forecast-bucket".to_string(),
                prefix: "outputs/earth2-deterministic/run-123".to_string(),
                region: Some("us-east-1".to_string()),
                endpoint: Some("https://s3.us-east-1.amazonaws.com".to_string()),
            }
        );
    }

    #[test]
    fn test_output_publication_azure_config_loaded_from_runtime_config() {
        let temp_path = std::env::temp_dir().join(format!(
            "physicsnemo-serve-runtime-azure-publication-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &temp_path,
            r#"{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "azure",
      "container": "forecast-results",
      "prefix": "outputs",
      "endpoint": "https://example.blob.core.windows.net"
    }
  }
}"#,
        )
        .unwrap();

        let temp_path_string = temp_path.to_string_lossy().to_string();
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG",
            temp_path_string.as_str(),
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        let resolved = config
            .output_publication
            .resolve_for_workflow("earth2-deterministic", "run-abc", &serde_json::json!({}))
            .unwrap()
            .expect("azure target should resolve");

        assert_eq!(resolved.target.artifact, "primary");
        assert_eq!(resolved.target.provider, OutputPublicationProvider::Azure);
        assert_eq!(
            resolved.target.storage,
            ResolvedOutputPublicationStorage::Azure {
                container: "forecast-results".to_string(),
                prefix: "outputs/earth2-deterministic/run-abc".to_string(),
                endpoint: "https://example.blob.core.windows.net".to_string(),
            }
        );
    }

    #[test]
    fn test_output_publication_azure_rejects_container_path_separators() {
        for container in [
            "forecast-results/nested",
            "forecast-results\\nested",
            "/forecast-results",
            "forecast-results/",
        ] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::Azure {
                    container: container.to_string(),
                    prefix: "outputs".to_string(),
                    endpoint: "https://example.blob.core.windows.net".to_string(),
                }),
            };

            assert!(config.validate().is_err(), "container {container:?}");
        }
    }

    #[test]
    fn test_output_publication_azure_rejects_container_surrounding_whitespace() {
        for container in [
            " forecast-results",
            "forecast-results ",
            "\tforecast-results\n",
        ] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::Azure {
                    container: container.to_string(),
                    prefix: "outputs".to_string(),
                    endpoint: "https://example.blob.core.windows.net".to_string(),
                }),
            };

            assert!(config.validate().is_err(), "container {container:?}");
        }
    }

    #[test]
    fn test_output_publication_azure_requires_account_root_endpoint() {
        for endpoint in [
            "https://example.blob.core.windows.net/container",
            "https://example.blob.core.windows.net?sig=secret",
            "https://example.blob.core.windows.net/#fragment",
        ] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::Azure {
                    container: "forecast-results".to_string(),
                    prefix: "outputs".to_string(),
                    endpoint: endpoint.to_string(),
                }),
            };

            assert!(config.validate().is_err(), "endpoint {endpoint:?}");
        }
    }

    #[test]
    fn test_output_publication_azure_accepts_account_root_endpoint_with_optional_slash() {
        for endpoint in [
            "https://example.blob.core.windows.net",
            "https://example.blob.core.windows.net/",
        ] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::Azure {
                    container: "forecast-results".to_string(),
                    prefix: "outputs".to_string(),
                    endpoint: endpoint.to_string(),
                }),
            };

            config
                .validate()
                .expect("account-root endpoint should pass");
        }
    }

    #[test]
    fn test_output_publication_s3_rejects_malformed_endpoint() {
        for endpoint in ["minio:9000", "://missing-scheme", "ftp://host:9000", "   "] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::S3 {
                    bucket: "forecast-results".to_string(),
                    prefix: "outputs".to_string(),
                    region: None,
                    endpoint: Some(endpoint.to_string()),
                }),
            };

            assert!(config.validate().is_err(), "endpoint {endpoint:?}");
        }
    }

    #[test]
    fn test_output_publication_s3_accepts_http_and_https_endpoint() {
        for endpoint in ["http://minio:9000", "https://s3.example.com"] {
            let config = OutputPublicationConfig {
                enabled: true,
                storage: Some(OutputPublicationStorageConfig::S3 {
                    bucket: "forecast-results".to_string(),
                    prefix: "outputs".to_string(),
                    region: None,
                    endpoint: Some(endpoint.to_string()),
                }),
            };

            config.validate().expect("http(s) endpoint should pass");
        }
    }

    #[test]
    fn test_output_publication_invalid_runtime_config_reports_path() {
        let temp_path = std::env::temp_dir().join(format!(
            "physicsnemo-serve-runtime-invalid-publication-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &temp_path,
            r#"{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "s3",
      "bucket": "",
      "prefix": "outputs"
    }
  }
}"#,
        )
        .unwrap();

        let temp_path_string = temp_path.to_string_lossy().to_string();
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG",
            temp_path_string.as_str(),
        )]);
        let err = ServerConfig::from_env_vars(&env_provider).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("invalid output_publication config"));
        assert!(message.contains(temp_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn test_output_publication_config_loaded_from_json_env_override() {
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON",
            r#"{
                "enabled": true,
                "storage": {
                    "type": "s3",
                    "bucket": "env-bucket",
                    "prefix": "env-prefix",
                    "region": "region-1",
                    "endpoint": "https://object.example"
                }
            }"#,
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        let resolved = config
            .output_publication
            .resolve_for_workflow("workflow-a", "run-b", &serde_json::json!({}))
            .unwrap()
            .expect("env config should resolve");

        assert_eq!(
            resolved.target.storage,
            ResolvedOutputPublicationStorage::S3 {
                bucket: "env-bucket".to_string(),
                prefix: "env-prefix/workflow-a/run-b".to_string(),
                region: Some("region-1".to_string()),
                endpoint: Some("https://object.example".to_string()),
            }
        );
    }

    #[test]
    fn test_enabled_plugin_id_from_env() {
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID",
            " demo-plugin ",
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(config.enabled_plugin_id.as_deref(), Some("demo-plugin"));
    }

    #[test]
    fn test_enabled_plugin_id_empty_env_is_ignored() {
        let env_provider = create_test_env(vec![("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "   ")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(config.enabled_plugin_id, None);
    }

    #[test]
    fn test_python_runtime_envs_loaded_from_config_file() {
        let temp_path = std::env::temp_dir().join(format!(
            "physicsnemo-serve-runtime-envs-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &temp_path,
            r#"{
  "python_runtime_envs": {
    "python.cpu.demo": {
      "python_executable": "/opt/envs/python.cpu.demo/bin/python",
      "env": {
        "PYTHONPATH": "/opt/demo"
      }
    }
  }
}"#,
        )
        .unwrap();

        let temp_path_string = temp_path.to_string_lossy().to_string();
        let env_provider = create_test_env(vec![(
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG",
            temp_path_string.as_str(),
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(
            config.python_runtime_envs["python.cpu.demo"].python_executable,
            "/opt/envs/python.cpu.demo/bin/python"
        );
        assert_eq!(
            config.python_runtime_envs["python.cpu.demo"].env["PYTHONPATH"],
            "/opt/demo"
        );
    }

    #[test]
    fn test_cors_allowed_origins_from_env() {
        let env_provider = create_test_env(vec![(
            "CORS_ALLOWED_ORIGINS",
            "https://a.com, https://b.com",
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(
            config.cors_allowed_origins,
            vec!["https://a.com", "https://b.com"]
        );
    }

    #[test]
    fn test_max_body_size_from_env() {
        let env_provider = create_test_env(vec![("MAX_BODY_SIZE", "1048576")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.max_body_size, 1_048_576);
    }

    #[test]
    fn test_artifact_dir_from_env() {
        let env_provider =
            create_test_env(vec![("ARTIFACT_DIR", "/tmp/physicsnemo-serve-artifacts")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(
            config.artifact_dir,
            PathBuf::from("/tmp/physicsnemo-serve-artifacts")
        );
    }

    #[test]
    fn test_default_output_dir_from_env() {
        let env_provider = create_test_env(vec![(
            "DEFAULT_OUTPUT_DIR",
            "/tmp/physicsnemo-serve-outputs",
        )]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(
            config.default_output_dir,
            PathBuf::from("/tmp/physicsnemo-serve-outputs")
        );
    }

    #[test]
    fn test_artifact_retention_hours_from_env() {
        let env_provider = create_test_env(vec![("ARTIFACT_RETENTION_HOURS", "48")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.artifact_retention_hours, 48);
    }

    #[test]
    fn test_from_env_custom_port() {
        let env_provider = create_test_env(vec![("PORT", "9000")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.addr.port(), 9000);
        assert_eq!(config.addr.ip(), std::net::Ipv4Addr::new(0, 0, 0, 0));
    }

    #[test]
    fn test_from_env_redis_url() {
        let env_provider = create_test_env(vec![("REDIS_URL", "redis://localhost:6380/1")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_url, "redis://localhost:6380/1");
    }

    #[test]
    fn test_from_env_redis_components() {
        let env_provider = create_test_env(vec![
            ("REDIS_HOST", "myhost"),
            ("REDIS_PORT", "6380"),
            ("REDIS_DB", "2"),
            ("REDIS_PASSWORD", "secret"),
        ]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_url, "redis://:secret@myhost:6380/2");
    }

    #[test]
    fn test_from_env_redis_components_no_password() {
        let env_provider = create_test_env(vec![
            ("REDIS_HOST", "myhost"),
            ("REDIS_PORT", "6380"),
            ("REDIS_DB", "2"),
            ("REDIS_PASSWORD", ""),
        ]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_url, "redis://myhost:6380/2");
    }

    #[test]
    fn test_from_env_redis_components_partial() {
        // Test with only some components set
        let env_provider = create_test_env(vec![("REDIS_HOST", "customhost")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_url, "redis://customhost:6379/0");
    }

    #[test]
    fn test_from_env_inference_stream() {
        let env_provider = create_test_env(vec![("INFERENCE_STREAM", "custom_stream")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_stream, "custom_stream");
    }

    #[test]
    fn test_artifact_cleanup_interval_from_env() {
        let env_provider = create_test_env(vec![("ARTIFACT_CLEANUP_INTERVAL_SECS", "60")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.artifact_cleanup_interval_secs, 60);
    }

    #[test]
    fn test_artifact_cleanup_interval_falls_back_to_legacy_workflow_discovery_env() {
        let env_provider = create_test_env(vec![("WORKFLOW_DISCOVERY_INTERVAL", "45")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.artifact_cleanup_interval_secs, 45);
    }

    #[test]
    fn test_plugin_dirs_from_plugin_dir_env_provider() {
        let env_provider = create_test_env(vec![("PLUGIN_DIR", "/tmp/plugins")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(config.plugin_dirs, vec![PathBuf::from("/tmp/plugins")]);
    }

    #[test]
    fn test_plugin_dirs_use_plugin_dir_only() {
        let env_provider = create_test_env(vec![("PLUGIN_DIR", "/tmp/plugins_a:/tmp/plugins_b")]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(
            config.plugin_dirs,
            vec![
                PathBuf::from("/tmp/plugins_a"),
                PathBuf::from("/tmp/plugins_b")
            ]
        );
    }

    #[test]
    fn test_plugin_dirs_empty_when_unset() {
        let env_provider = create_test_env(vec![]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert!(config.plugin_dirs.is_empty());
    }

    #[test]
    fn test_from_env_invalid_port() {
        let env_provider = create_test_env(vec![("PORT", "invalid")]);
        let result = ServerConfig::from_env_vars(&env_provider);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_env_port_out_of_range() {
        let env_provider = create_test_env(vec![("PORT", "99999")]);
        let result = ServerConfig::from_env_vars(&env_provider);
        assert!(result.is_err());
    }

    #[test]
    fn test_redis_url_priority_over_components() {
        // When REDIS_URL is set, it should take precedence over individual components
        let env_provider = create_test_env(vec![
            ("REDIS_URL", "redis://url-host:7000/5"),
            ("REDIS_HOST", "component-host"),
            ("REDIS_PORT", "6380"),
            ("REDIS_DB", "2"),
        ]);
        let config = ServerConfig::from_env_vars(&env_provider).unwrap();
        assert_eq!(config.redis_url, "redis://url-host:7000/5");
    }

    #[test]
    fn test_parse_env_file_helper() {
        let env_content = r#"
# This is a comment
PORT=8080

# Redis config
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Empty line above
INFERENCE_STREAM=inference
"#;
        let env_map = parse_env_file(env_content);

        assert_eq!(env_map.get("PORT"), Some(&"8080".to_string()));
        assert_eq!(env_map.get("REDIS_HOST"), Some(&"localhost".to_string()));
        assert_eq!(env_map.get("REDIS_PORT"), Some(&"6379".to_string()));
        assert_eq!(env_map.get("REDIS_DB"), Some(&"0".to_string()));
        assert_eq!(
            env_map.get("INFERENCE_STREAM"),
            Some(&"inference".to_string())
        );
        // Empty values should not be included
        assert_eq!(env_map.get("REDIS_PASSWORD"), None);
    }

    #[test]
    fn test_from_config_example_env() {
        // Read and parse the example config file
        let example_config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.example.env");

        // Only run this test if the file exists
        if !example_config_path.exists() {
            eprintln!("Skipping test_from_config_example_env: config.example.env not found");
            return;
        }

        let env_content = std::fs::read_to_string(&example_config_path)
            .expect("Failed to read config.example.env");
        let env_map = parse_env_file(&env_content);

        // Create env_provider from parsed values
        let env_provider = move |key: &str| env_map.get(key).cloned();

        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        // Verify values from config.example.env
        assert_eq!(
            config.addr.port(),
            8080,
            "Port should match config.example.env"
        );

        // Redis should be constructed from components (no REDIS_URL in example)
        assert_eq!(
            config.redis_url, "redis://localhost:6379/0",
            "Redis URL should be constructed from config.example.env components"
        );

        assert_eq!(
            config.redis_stream, "inference",
            "Inference stream should match config.example.env"
        );

        assert!(!config.plugin_dirs.is_empty());
    }

    #[test]
    fn test_all_config_fields_set() {
        // Comprehensive test with all fields explicitly set
        let env_provider = create_test_env(vec![
            ("PORT", "3000"),
            ("REDIS_URL", "redis://:pass@example.com:7000/3"),
            ("INFERENCE_STREAM", "my_inference_stream"),
        ]);

        let config = ServerConfig::from_env_vars(&env_provider).unwrap();

        assert_eq!(config.addr.port(), 3000);
        assert_eq!(config.redis_url, "redis://:pass@example.com:7000/3");
        assert_eq!(config.redis_stream, "my_inference_stream");
        assert!(config.plugin_dirs.is_empty());
    }
}
