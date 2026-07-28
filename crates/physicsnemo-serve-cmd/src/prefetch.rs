/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::io::Read;
use std::path::Path;

use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use worker_runtime::roles::prefetch::{PrefetchPlanItem, materialize_prefetch_plan};

use crate::digest::sha256_file_hex;

pub fn read_prefetch_plan(reader: impl Read) -> Result<Value> {
    serde_json::from_reader(reader)
        .map_err(|error| anyhow!("prefetch plan must be valid JSON: {error}"))
}

pub async fn materialize_direct_plan(plan: Value, cache_dir: &Path, run_id: &str) -> Result<Value> {
    let direct_items: Vec<DirectPlanItem> =
        serde_json::from_value(plan).map_err(|error| anyhow!("invalid prefetch plan: {error}"))?;
    let worker_items: Vec<PrefetchPlanItem> =
        direct_items.iter().map(|item| item.plan.clone()).collect();
    let materialized = materialize_prefetch_plan(&worker_items, cache_dir, run_id).await?;
    if materialized.stats.required_errors > 0 {
        return Err(anyhow!(
            "required prefetch operations failed: {}",
            materialized.stats.required_errors
        ));
    }

    let mut artifacts = Vec::with_capacity(materialized.artifacts.len());
    for artifact in &materialized.artifacts {
        let contract = direct_items
            .iter()
            .find(|item| {
                item.plan.target_artifact_name == artifact.name
                    && item.plan.source_uri == artifact.source_uri
            })
            .ok_or_else(|| {
                anyhow!(
                    "prefetch result '{}' did not match its input plan",
                    artifact.name
                )
            })?;
        let actual_size = fs::metadata(&artifact.storage_path)?.len();
        if let Some(expected_size) = contract.expected_size_bytes
            && actual_size != expected_size
        {
            let _ = fs::remove_file(&artifact.storage_path);
            return Err(anyhow!(
                "prefetched artifact '{}' size mismatch: expected {}, got {}",
                artifact.name,
                expected_size,
                actual_size
            ));
        }

        let mut encoded = serde_json::to_value(artifact)?;
        if let Some(expected_sha256) = contract.expected_sha256.as_deref() {
            let actual_sha256 = sha256_file_hex(Path::new(&artifact.storage_path))?;
            if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
                let _ = fs::remove_file(&artifact.storage_path);
                return Err(anyhow!(
                    "prefetched artifact '{}' SHA-256 mismatch",
                    artifact.name
                ));
            }
            let object = encoded
                .as_object_mut()
                .ok_or_else(|| anyhow!("prefetch artifact did not serialize as an object"))?;
            object.insert("sha256".to_string(), json!(actual_sha256));
            object.insert("verified_sha256".to_string(), json!(actual_sha256));
        }
        artifacts.push(encoded);
    }

    Ok(json!({
        "artifacts": artifacts,
        "stats": {
            "downloaded": materialized.stats.downloaded,
            "cached": materialized.stats.cached,
            "errors": materialized.stats.errors,
            "required_errors": materialized.stats.required_errors,
            "optional_errors": materialized.stats.optional_errors,
            "total_time_secs": materialized.stats.total_time_secs,
            "throughput_mbps": materialized.stats.throughput_mbps,
            "total_mb": materialized.stats.total_mb,
        }
    }))
}

#[derive(Debug, Deserialize)]
struct DirectPlanItem {
    #[serde(flatten)]
    plan: PrefetchPlanItem,
    #[serde(default)]
    expected_sha256: Option<String>,
    #[serde(default)]
    expected_size_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;
    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn materializes_and_verifies_local_file() {
        let temp = tempdir().expect("temp directory should be created");
        let source = temp.path().join("source.bin");
        fs::write(&source, b"verified payload").expect("source should be written");
        let digest = Sha256::digest(b"verified payload")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();

        let result = materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source,
                "target_artifact_name": "input",
                "required": true,
                "expected_sha256": digest,
                "expected_size_bytes": 16
            }]),
            &temp.path().join("cache"),
            "prefetch-test",
        )
        .await
        .expect("prefetch should succeed");

        assert_eq!(result["stats"]["downloaded"], 1);
        assert_eq!(result["artifacts"][0]["name"], "input");
        assert_eq!(result["artifacts"][0]["verified_sha256"], digest);
        assert_eq!(result["artifacts"][0]["sha256"], digest);
    }

    #[tokio::test]
    async fn rejects_checksum_mismatch() {
        let temp = tempdir().expect("temp directory should be created");
        let source = temp.path().join("source.bin");
        fs::write(&source, b"unexpected payload").expect("source should be written");

        let error = materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source,
                "target_artifact_name": "input",
                "required": true,
                "expected_sha256": "0".repeat(64),
                "expected_size_bytes": 18
            }]),
            &temp.path().join("cache"),
            "prefetch-test",
        )
        .await
        .expect_err("checksum mismatch should fail");

        assert!(error.to_string().contains("SHA-256 mismatch"));
    }

    #[test]
    fn rejects_malformed_prefetch_json() {
        let error = read_prefetch_plan(b"{not-json".as_slice()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prefetch plan must be valid JSON")
        );
    }
}
