use crate::deployment::RuntimeDeploymentConfig;
use crate::request::{DeploymentConfig, InputObject};
use anyhow::{Context, Result, ensure};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
};

struct PartialDownload(PathBuf);
impl Drop for PartialDownload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub async fn verify_file(path: &Path, size: u64, digest: &str) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path).await?.file_type().is_file(),
        "asset must be a regular file: {}",
        path.display()
    );
    let mut file = fs::File::open(path).await?;
    ensure!(
        file.metadata().await?.len() == size,
        "asset size mismatch: {}",
        path.display()
    );
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == digest,
        "asset SHA256 mismatch: {}",
        path.display()
    );
    Ok(())
}

pub async fn verify_release(config: &DeploymentConfig) -> Result<()> {
    config.validate()?;
    for asset in &config.model_assets {
        verify_file(&asset.path, asset.size_bytes, &asset.sha256).await?;
    }
    if let Some(path) = &config.startup_fixture_path {
        let bytes = fs::read(path).await?;
        ensure!(
            bytes.len() <= 1024 * 1024,
            "startup fixture metadata is too large"
        );
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(
            fixture["schema_version"] == 1,
            "unsupported startup fixture schema"
        );
        let models = fixture["models"]
            .as_array()
            .context("startup fixture models")?;
        ensure!(
            models.len() == config.packages.len(),
            "startup fixture must cover every resident package"
        );
        let parent = path.parent().context("startup fixture parent")?;
        for model in models {
            let mut descriptors: Vec<_> = model["inputs"]
                .as_object()
                .context("startup inputs")?
                .values()
                .collect();
            descriptors.push(&model["expected_output"]);
            for descriptor in descriptors {
                let relative =
                    Path::new(descriptor["path"].as_str().context("startup tensor path")?);
                ensure!(
                    !relative.as_os_str().is_empty()
                        && relative
                            .components()
                            .all(|part| matches!(part, std::path::Component::Normal(_))),
                    "startup tensor path must be relative and contained"
                );
                let tensor = parent.join(relative);
                ensure!(
                    config.model_assets.iter().any(|asset| asset.path == tensor),
                    "startup tensor is missing from hashed inventory: {}",
                    tensor.display()
                );
            }
        }
    }
    // Fail closed on added/unhashed package files, not only changed known files.
    let mut pending = config.packages.clone();
    while let Some(path) = pending.pop() {
        let metadata = fs::symlink_metadata(&path).await?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "package symlinks are not supported"
        );
        if metadata.is_dir() {
            let mut entries = fs::read_dir(path).await?;
            while let Some(entry) = entries.next_entry().await? {
                pending.push(entry.path());
            }
        } else {
            ensure!(
                config.model_assets.iter().any(|asset| asset.path == path),
                "unhashed package file: {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub async fn verify_deployment(config: &RuntimeDeploymentConfig) -> Result<()> {
    match config {
        RuntimeDeploymentConfig::Transolver(config) => verify_release(config).await,
        RuntimeDeploymentConfig::Workflow(config) => {
            config.validate()?;
            for asset in &config.model_assets {
                verify_file(&asset.path, asset.size_bytes, &asset.sha256).await?;
            }
            let models = config.workflow["models"]
                .as_object()
                .context("workflow models")?;
            let mut pending: Vec<PathBuf> = models
                .values()
                .map(|model| {
                    PathBuf::from(
                        model["package_path"]
                            .as_str()
                            .expect("validated package path"),
                    )
                })
                .collect();
            while let Some(path) = pending.pop() {
                let metadata = fs::symlink_metadata(&path).await?;
                ensure!(
                    !metadata.file_type().is_symlink(),
                    "package symlinks are not supported"
                );
                if metadata.is_dir() {
                    let mut entries = fs::read_dir(path).await?;
                    while let Some(entry) = entries.next_entry().await? {
                        pending.push(entry.path());
                    }
                } else {
                    ensure!(
                        metadata.is_file()
                            && config.model_assets.iter().any(|asset| asset.path == path),
                        "unhashed/special package file: {}",
                        path.display()
                    );
                }
            }
            if let Some(assets) = config.workflow.get("assets") {
                for path in assets
                    .as_object()
                    .context("workflow assets must be a map")?
                    .values()
                {
                    let path = Path::new(path.as_str().context("asset path must be a string")?);
                    ensure!(
                        config.model_assets.iter().any(|asset| asset.path == path),
                        "workflow asset is not in hashed inventory"
                    );
                }
            }
            Ok(())
        }
    }
}

pub async fn materialize(object: &InputObject, root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(root).await?;
    let path = root.join(&object.sha256);
    if path.exists() {
        verify_file(&path, object.size_bytes, &object.sha256).await?;
        return Ok(path);
    }
    let temporary = root.join(format!(".{}.{}", object.sha256, uuid::Uuid::new_v4()));
    // A cancelled async future must unlink partial bytes too. The nested
    // download/file future drops before this guard, including on lease loss.
    let _cleanup = PartialDownload(temporary.clone());
    async {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(1800))
            .https_only(true)
            .build()?;
        let response = client.get(&object.url).send().await?.error_for_status()?;
        if let Some(length) = response.content_length() {
            ensure!(length == object.size_bytes, "input Content-Length mismatch");
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .await?;
        let mut stream = response.bytes_stream();
        let mut hash = Sha256::new();
        let mut size = 0_u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            size = size
                .checked_add(chunk.len() as u64)
                .context("input size overflow")?;
            ensure!(size <= object.size_bytes, "input exceeds declared size");
            hash.update(&chunk);
            file.write_all(&chunk).await?;
        }
        ensure!(size == object.size_bytes, "truncated input");
        ensure!(
            format!("{:x}", hash.finalize()) == object.sha256,
            "input SHA256 mismatch"
        );
        file.sync_all().await?;
        drop(file);
        fs::rename(&temporary, &path).await?;
        Ok(path)
    }
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancelled_download_unlinks_partial_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(".partial");
        let pending = async {
            let _cleanup = PartialDownload(path.clone());
            fs::write(&path, b"partial bytes").await.unwrap();
            std::future::pending::<()>().await;
        };
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), pending)
                .await
                .is_err()
        );
        assert!(!path.exists());
    }
    #[tokio::test]
    async fn poisoned_cache_is_rejected_before_any_network_or_execution() {
        let root = tempfile::tempdir().unwrap();
        let sha = format!("{:x}", Sha256::digest(b"original"));
        let object = InputObject {
            url: "https://must-not-be-called.invalid/input".into(),
            sha256: sha.clone(),
            size_bytes: 8,
        };
        fs::write(root.path().join(sha), b"tampered").await.unwrap();
        let error = materialize(&object, root.path()).await.unwrap_err();
        assert!(error.to_string().contains("SHA256 mismatch"));
    }
    #[tokio::test]
    async fn verified_cached_input_is_reused_and_incomplete_bytes_rejected() {
        let root = tempfile::tempdir().unwrap();
        let sha = format!("{:x}", Sha256::digest(b"original"));
        let path = root.path().join(&sha);
        fs::write(&path, b"original").await.unwrap();
        let object = InputObject {
            url: "https://must-not-be-called.invalid/input".into(),
            sha256: sha,
            size_bytes: 8,
        };
        assert_eq!(materialize(&object, root.path()).await.unwrap(), path);
        fs::write(&path, b"short").await.unwrap();
        assert!(
            materialize(&object, root.path())
                .await
                .unwrap_err()
                .to_string()
                .contains("size mismatch")
        );
    }
}
