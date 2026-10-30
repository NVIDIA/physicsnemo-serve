//! Immutable execution identities distinguish registry images, local Docker
//! configuration objects and host bundles. These digests are not interchangeable.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum RuntimeIdentity {
    RegistryImage(String),
    ImageConfig(String),
    Bundle(String),
}

impl RuntimeIdentity {
    pub fn from_sources(
        release_image: Option<&str>,
        image_config: Option<&str>,
        bundle: Option<&str>,
    ) -> Result<Self> {
        ensure!(
            [release_image, image_config, bundle]
                .iter()
                .filter(|value| value.is_some())
                .count()
                == 1,
            "set exactly one of RELEASE_IMAGE_DIGEST, PNMS_IMAGE_CONFIG_DIGEST or PNMS_BUNDLE_DIGEST"
        );
        if let Some(image) = release_image {
            let (repository, digest) = image
                .rsplit_once("@sha256:")
                .context("RELEASE_IMAGE_DIGEST must be registry/repository@sha256:<digest>")?;
            ensure!(
                !repository.is_empty()
                    && !repository.contains('@')
                    && !repository.chars().any(char::is_whitespace),
                "invalid release image repository"
            );
            validate_digest(digest)?;
            return Ok(Self::RegistryImage(image.into()));
        }
        if let Some(image) = image_config {
            validate_digest(
                image
                    .strip_prefix("sha256:")
                    .context("PNMS_IMAGE_CONFIG_DIGEST must be sha256:<digest>")?,
            )?;
            return Ok(Self::ImageConfig(image.into()));
        }
        let digest = bundle.context("missing bundle digest")?;
        validate_digest(digest)?;
        Ok(Self::Bundle(digest.into()))
    }

    pub fn from_environment() -> Result<Self> {
        let image = optional("RELEASE_IMAGE_DIGEST")?;
        let config = optional("PNMS_IMAGE_CONFIG_DIGEST")?;
        let bundle = optional("PNMS_BUNDLE_DIGEST")?;
        Self::from_sources(image.as_deref(), config.as_deref(), bundle.as_deref())
    }

    pub fn registry_image(&self) -> Option<&str> {
        match self {
            Self::RegistryImage(value) => Some(value),
            _ => None,
        }
    }

    pub fn fields(&self) -> Map<String, Value> {
        let mut fields = Map::new();
        fields.insert("runtime_identity".into(), json!(self));
        match self {
            Self::RegistryImage(value) => {
                fields.insert("release_image".into(), json!(value));
            }
            Self::ImageConfig(value) => {
                fields.insert("image_config_digest".into(), json!(value));
            }
            Self::Bundle(value) => {
                fields.insert("bundle_digest".into(), json!(value));
            }
        }
        fields
    }

    pub fn verify_bundle_digest(&self, digest: &str) -> Result<()> {
        validate_digest(digest)?;
        if let Self::Bundle(expected) = self {
            ensure!(
                expected == digest,
                "host bundle identity differs from the release inventory"
            );
        }
        Ok(())
    }
}

pub fn validate_digest(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()),
        "immutable identity must contain a 64-digit hexadecimal SHA-256 digest"
    );
    Ok(())
}

fn optional(key: &str) -> Result<Option<String>> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid {key}")),
    }
}

/// Direct role component tests can execute without a process release. Worker
/// startup always requires from_environment(); supplied identities still fail
/// closed here if invalid or ambiguous.
pub fn optional_environment_fields() -> Result<Map<String, Value>> {
    if [
        "RELEASE_IMAGE_DIGEST",
        "PNMS_IMAGE_CONFIG_DIGEST",
        "PNMS_BUNDLE_DIGEST",
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some())
    {
        Ok(RuntimeIdentity::from_environment()?.fields())
    } else {
        Ok(Map::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_bundle_and_local_image_have_distinct_typed_identities() {
        let digest = "a".repeat(64);
        assert!(
            RuntimeIdentity::from_sources(None, None, Some(&digest)).is_ok(),
            "a host bundle must not require a registry image"
        );
        assert!(
            RuntimeIdentity::from_sources(None, Some(&format!("sha256:{digest}")), None).is_ok(),
            "a local Docker image must not require a registry image"
        );
    }

    #[test]
    fn ambiguous_or_malformed_execution_sources_are_rejected() {
        let digest = "a".repeat(64);
        let image = format!("registry.example/app@sha256:{digest}");
        assert!(
            RuntimeIdentity::from_sources(Some(&image), None, Some(&digest)).is_err(),
            "two different identity kinds cannot describe one execution source"
        );
        for value in ["", "latest", "sha256:bad", &"g".repeat(64)] {
            assert!(RuntimeIdentity::from_sources(None, None, Some(value)).is_err());
        }
        assert!(RuntimeIdentity::from_sources(None, None, None).is_err());
        assert!(RuntimeIdentity::from_sources(None, Some("my-app:latest"), None).is_err());
        assert!(RuntimeIdentity::from_sources(Some("my-app:latest"), None, None).is_err());
        assert!(
            RuntimeIdentity::from_sources(None, Some(&format!("sha256:{digest}")), Some(&digest))
                .is_err()
        );
    }

    #[test]
    fn provenance_never_labels_a_host_bundle_as_a_registry_image() {
        let digest = "b".repeat(64);
        let identity = RuntimeIdentity::from_sources(None, None, Some(&digest)).unwrap();
        let fields = identity.fields();
        assert_eq!(
            fields["runtime_identity"],
            json!({"kind":"bundle","value":digest})
        );
        assert_eq!(fields["bundle_digest"], digest);
        assert!(!fields.contains_key("release_image"));
        assert!(identity.verify_bundle_digest(&"c".repeat(64)).is_err());
        assert!(identity.verify_bundle_digest(&digest).is_ok());
        let config =
            RuntimeIdentity::from_sources(None, Some(&format!("sha256:{digest}")), None).unwrap();
        assert_eq!(config.fields()["runtime_identity"]["kind"], "image-config");
        assert!(!config.fields().contains_key("release_image"));
    }
}
