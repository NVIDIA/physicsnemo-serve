/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::fs;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedArtifact {
    pub artifact_id: String,
    pub field_name: String,
    pub original_filename: Option<String>,
    pub media_type: String,
    pub size_bytes: u64,
    pub storage_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root_dir: PathBuf,
}

impl ArtifactStore {
    pub fn new(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub async fn stage_file(
        &self,
        run_id: &str,
        field_name: &str,
        original_filename: Option<&str>,
        media_type: Option<&str>,
        bytes: &[u8],
    ) -> Result<StagedArtifact> {
        let artifact_id = Uuid::new_v4().to_string();
        let staged_dir = self.root_dir.join(run_id);
        fs::create_dir_all(&staged_dir).await.with_context(|| {
            format!(
                "failed to create artifact staging directory '{}'",
                staged_dir.display()
            )
        })?;

        let safe_name = sanitize_filename(original_filename.unwrap_or(field_name));
        let storage_path = staged_dir.join(format!("{artifact_id}-{safe_name}"));
        fs::write(&storage_path, bytes).await.with_context(|| {
            format!(
                "failed to write staged artifact '{}'",
                storage_path.display()
            )
        })?;

        Ok(StagedArtifact {
            artifact_id,
            field_name: field_name.to_string(),
            original_filename: original_filename.map(ToString::to_string),
            media_type: media_type.unwrap_or("application/octet-stream").to_string(),
            size_bytes: bytes.len() as u64,
            storage_path,
        })
    }

    pub async fn stage_file_from_path(
        &self,
        run_id: &str,
        field_name: &str,
        original_filename: Option<&str>,
        media_type: Option<&str>,
        source_path: &Path,
        size_bytes: u64,
    ) -> Result<StagedArtifact> {
        let artifact_id = Uuid::new_v4().to_string();
        let staged_dir = self.root_dir.join(run_id);
        fs::create_dir_all(&staged_dir).await.with_context(|| {
            format!(
                "failed to create artifact staging directory '{}'",
                staged_dir.display()
            )
        })?;

        let safe_name = sanitize_filename(original_filename.unwrap_or(field_name));
        let storage_path = staged_dir.join(format!("{artifact_id}-{safe_name}"));
        match fs::rename(source_path, &storage_path).await {
            Ok(()) => {}
            Err(_rename_err) => {
                fs::copy(source_path, &storage_path)
                    .await
                    .with_context(|| {
                        format!(
                            "failed to copy staged artifact from '{}' to '{}'",
                            source_path.display(),
                            storage_path.display()
                        )
                    })?;
                let _ = fs::remove_file(source_path).await;
            }
        }

        Ok(StagedArtifact {
            artifact_id,
            field_name: field_name.to_string(),
            original_filename: original_filename.map(ToString::to_string),
            media_type: media_type.unwrap_or("application/octet-stream").to_string(),
            size_bytes,
            storage_path,
        })
    }

    pub async fn resolve_download_path(&self, candidate: &Path) -> Result<PathBuf> {
        self.resolve_download_path_with_additional_roots(candidate, &[])
            .await
    }

    pub async fn remove_run_dir(&self, run_id: &str) -> Result<()> {
        let run_dir = self.root_dir.join(run_id);
        match fs::remove_dir_all(&run_dir).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err).with_context(|| {
                format!(
                    "failed to remove staged artifact directory '{}'",
                    run_dir.display()
                )
            }),
        }
    }

    pub async fn resolve_download_path_with_additional_roots(
        &self,
        candidate: &Path,
        additional_roots: &[PathBuf],
    ) -> Result<PathBuf> {
        let candidate_path = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.root_dir.join(candidate)
        };

        let canonical_candidate = fs::canonicalize(&candidate_path).await.with_context(|| {
            format!(
                "failed to canonicalize artifact path '{}'",
                candidate_path.display()
            )
        })?;

        let mut canonical_roots = Vec::with_capacity(1 + additional_roots.len());
        if let Some(root) = canonicalize_root_dir(&self.root_dir).await? {
            canonical_roots.push(root);
        }
        for root in additional_roots {
            if let Some(root) = canonicalize_root_dir(root).await? {
                canonical_roots.push(root);
            }
        }

        if canonical_roots.is_empty() {
            anyhow::bail!("no configured download roots exist on disk");
        }

        if !canonical_roots
            .iter()
            .any(|root| canonical_candidate.starts_with(root))
        {
            let allowed_roots = canonical_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "artifact path '{}' is outside the configured download roots [{}]",
                canonical_candidate.display(),
                allowed_roots
            );
        }

        let metadata = fs::metadata(&canonical_candidate).await.with_context(|| {
            format!(
                "failed to read artifact metadata '{}'",
                canonical_candidate.display()
            )
        })?;
        if !metadata.is_file() {
            anyhow::bail!(
                "artifact path '{}' is not a regular file",
                canonical_candidate.display()
            );
        }

        Ok(canonical_candidate)
    }

    pub async fn cleanup_expired_run_dirs(&self, max_age: Duration) -> Result<u64> {
        if !self.root_dir.exists() {
            return Ok(0);
        }

        let mut removed = 0_u64;
        let mut entries = fs::read_dir(&self.root_dir).await.with_context(|| {
            format!(
                "failed to read artifact root directory '{}'",
                self.root_dir.display()
            )
        })?;

        while let Some(entry) = entries.next_entry().await.with_context(|| {
            format!(
                "failed to iterate artifact root directory '{}'",
                self.root_dir.display()
            )
        })? {
            let path = entry.path();
            let metadata = entry.metadata().await.with_context(|| {
                format!(
                    "failed to read artifact directory metadata '{}'",
                    path.display()
                )
            })?;
            if !metadata.is_dir() {
                continue;
            }

            let modified_at = metadata.modified().with_context(|| {
                format!(
                    "failed to read last-modified time for artifact directory '{}'",
                    path.display()
                )
            })?;
            let age = SystemTime::now()
                .duration_since(modified_at)
                .unwrap_or(Duration::ZERO);
            if age < max_age {
                continue;
            }

            fs::remove_dir_all(&path).await.with_context(|| {
                format!(
                    "failed to remove expired artifact directory '{}'",
                    path.display()
                )
            })?;
            removed += 1;
        }

        Ok(removed)
    }
}

async fn canonicalize_root_dir(root: &Path) -> Result<Option<PathBuf>> {
    match fs::canonicalize(root).await {
        Ok(path) => Ok(Some(path)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err)
            .with_context(|| format!("failed to canonicalize artifact root '{}'", root.display())),
    }
}

pub(crate) fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' => c,
            _ => '_',
        })
        .collect();

    if sanitized.is_empty() {
        "artifact".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stage_file_writes_artifact_under_run_directory() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-{}",
            uuid::Uuid::new_v4()
        ));
        let store = ArtifactStore::new(root.clone());

        let artifact = store
            .stage_file(
                "run-123",
                "design_stl",
                Some("mesh.stl"),
                Some("model/stl"),
                b"solid cube\nendsolid cube",
            )
            .await
            .expect("artifact staging should succeed");

        assert_eq!(artifact.field_name, "design_stl");
        assert_eq!(artifact.media_type, "model/stl");
        assert_eq!(artifact.size_bytes, 24);
        assert!(artifact.storage_path.starts_with(root.join("run-123")));
        let contents = fs::read_to_string(&artifact.storage_path)
            .await
            .expect("staged artifact should be readable");
        assert_eq!(contents, "solid cube\nendsolid cube");
    }

    #[tokio::test]
    async fn stage_file_from_path_moves_artifact_under_run_directory() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-from-path-{}",
            uuid::Uuid::new_v4()
        ));
        let incoming_dir = root.join(".incoming");
        fs::create_dir_all(&incoming_dir).await.unwrap();
        let temp_upload_path = incoming_dir.join("upload.tmp");
        fs::write(&temp_upload_path, b"solid cube\nendsolid cube")
            .await
            .unwrap();

        let store = ArtifactStore::new(root.clone());
        let artifact = store
            .stage_file_from_path(
                "run-456",
                "design_stl",
                Some("mesh.stl"),
                Some("model/stl"),
                &temp_upload_path,
                24,
            )
            .await
            .expect("artifact staging from path should succeed");

        assert_eq!(artifact.field_name, "design_stl");
        assert_eq!(artifact.media_type, "model/stl");
        assert_eq!(artifact.size_bytes, 24);
        assert!(artifact.storage_path.starts_with(root.join("run-456")));
        assert!(!temp_upload_path.exists());
        let contents = fs::read_to_string(&artifact.storage_path)
            .await
            .expect("staged artifact should be readable");
        assert_eq!(contents, "solid cube\nendsolid cube");
    }

    #[test]
    fn sanitize_filename_replaces_path_separators_and_spaces() {
        assert_eq!(sanitize_filename("../bad name.stl"), ".._bad_name.stl");
    }

    #[tokio::test]
    async fn resolve_download_path_accepts_file_inside_root() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-resolve-{}",
            uuid::Uuid::new_v4()
        ));
        let store = ArtifactStore::new(root.clone());
        let staged = root.join("run-1").join("result.bin");
        fs::create_dir_all(staged.parent().unwrap()).await.unwrap();
        fs::write(&staged, b"payload").await.unwrap();

        let resolved = store.resolve_download_path(&staged).await.unwrap();
        assert_eq!(resolved, fs::canonicalize(staged).await.unwrap());
    }

    #[tokio::test]
    async fn resolve_download_path_rejects_file_outside_root() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-outside-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).await.unwrap();
        let store = ArtifactStore::new(root);

        let outside_dir = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-external-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&outside_dir).await.unwrap();
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, b"nope").await.unwrap();

        let err = store
            .resolve_download_path(&outside_file)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("outside the configured download roots")
        );
    }

    #[tokio::test]
    async fn resolve_download_path_accepts_file_inside_additional_root() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-extra-root-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).await.unwrap();
        let store = ArtifactStore::new(root);

        let extra_root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-extra-output-{}",
            uuid::Uuid::new_v4()
        ));
        let extra_file = extra_root.join("run-1").join("result.bin");
        fs::create_dir_all(extra_file.parent().unwrap())
            .await
            .unwrap();
        fs::write(&extra_file, b"payload").await.unwrap();

        let resolved = store
            .resolve_download_path_with_additional_roots(
                &extra_file,
                std::slice::from_ref(&extra_root),
            )
            .await
            .unwrap();
        assert_eq!(resolved, fs::canonicalize(extra_file).await.unwrap());
    }

    #[tokio::test]
    async fn cleanup_expired_run_dirs_removes_only_old_directories() {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-artifact-store-cleanup-{}",
            uuid::Uuid::new_v4()
        ));
        let store = ArtifactStore::new(root.clone());
        let max_age = std::time::Duration::from_secs(1);

        let old_dir = root.join("run-old");
        fs::create_dir_all(&old_dir).await.unwrap();
        fs::write(old_dir.join("artifact.bin"), b"old")
            .await
            .unwrap();

        tokio::time::sleep(max_age + std::time::Duration::from_millis(250)).await;

        let fresh_dir = root.join("run-fresh");
        fs::create_dir_all(&fresh_dir).await.unwrap();
        fs::write(fresh_dir.join("artifact.bin"), b"fresh")
            .await
            .unwrap();

        let removed = store
            .cleanup_expired_run_dirs(max_age)
            .await
            .expect("cleanup should succeed");

        assert_eq!(removed, 1);
        assert!(!old_dir.exists());
        assert!(fresh_dir.exists());
    }
}
