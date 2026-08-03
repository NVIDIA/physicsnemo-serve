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
    let worker_items: Vec<PrefetchPlanItem> = direct_items.iter().map(worker_plan_item).collect();
    let mut materialized = materialize_prefetch_plan(&worker_items, cache_dir, run_id).await?;
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
        let mut integrity_error = None;
        if let Some(expected_size) = contract.expected_size_bytes
            && actual_size != expected_size
        {
            integrity_error = Some(format!(
                "prefetched artifact '{}' size mismatch: expected {}, got {}",
                artifact.name, expected_size, actual_size
            ));
        }

        let mut encoded = serde_json::to_value(artifact)?;
        if integrity_error.is_none()
            && let Some(expected_sha256) = contract.expected_sha256.as_deref()
        {
            let actual_sha256 = sha256_file_hex(Path::new(&artifact.storage_path))?;
            if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
                integrity_error = Some(format!(
                    "prefetched artifact '{}' SHA-256 mismatch",
                    artifact.name
                ));
            } else {
                let object = encoded
                    .as_object_mut()
                    .ok_or_else(|| anyhow!("prefetch artifact did not serialize as an object"))?;
                object.insert("sha256".to_string(), json!(actual_sha256));
                object.insert("verified_sha256".to_string(), json!(actual_sha256));
            }
        }
        if let Some(error) = integrity_error {
            if contract.plan.required {
                return Err(anyhow!(error));
            }
            if artifact.downloaded {
                materialized.stats.downloaded = materialized.stats.downloaded.saturating_sub(1);
            } else {
                materialized.stats.cached = materialized.stats.cached.saturating_sub(1);
            }
            materialized.stats.errors += 1;
            materialized.stats.optional_errors += 1;
            continue;
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

fn worker_plan_item(item: &DirectPlanItem) -> PrefetchPlanItem {
    let mut plan = item.plan.clone();
    plan.expected_sha256.clone_from(&item.expected_sha256);
    plan.expected_size_bytes = item.expected_size_bytes;
    plan
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

        assert!(
            error
                .to_string()
                .contains("required prefetch operations failed: 1")
        );
    }

    #[tokio::test]
    async fn rejects_size_mismatch_before_populating_cache() {
        let temp = tempdir().expect("temp directory should be created");
        let source = temp.path().join("source.bin");
        fs::write(&source, b"payload").expect("source should be written");
        let cache = temp.path().join("cache");

        let error = materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source,
                "target_artifact_name": "input",
                "required": true,
                "expected_size_bytes": 8
            }]),
            &cache,
            "prefetch-test",
        )
        .await
        .expect_err("size mismatch should fail");

        assert!(
            error
                .to_string()
                .contains("required prefetch operations failed: 1")
        );
        let cache_files = fs::read_dir(cache.join("prefetch"))
            .expect("prefetch cache should exist")
            .map(|entry| entry.expect("cache entry should be readable").path())
            .collect::<Vec<_>>();
        assert!(
            cache_files.iter().all(|path| path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(".lock"))),
            "a size mismatch must not publish cached data"
        );
    }

    #[tokio::test]
    async fn omits_optional_local_file_with_integrity_mismatch() {
        let temp = tempdir().expect("temp directory should be created");
        let source = temp.path().join("source.bin");
        fs::write(&source, b"payload").expect("source should be written");

        let result = materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source,
                "target_artifact_name": "optional-input",
                "required": false,
                "expected_sha256": "0".repeat(64)
            }]),
            &temp.path().join("cache"),
            "prefetch-test",
        )
        .await
        .expect("optional integrity mismatch should not fail prefetch");

        assert_eq!(result["artifacts"], json!([]));
        assert_eq!(result["stats"]["downloaded"], 0);
        assert_eq!(result["stats"]["errors"], 1);
        assert_eq!(result["stats"]["required_errors"], 0);
        assert_eq!(result["stats"]["optional_errors"], 1);
        assert!(
            fs::read_dir(temp.path().join("cache/prefetch"))
                .expect("prefetch cache should exist")
                .next()
                .is_some(),
            "optional validation must not delete a shared cache entry"
        );
    }

    #[tokio::test]
    async fn removes_cache_hit_from_stats_after_optional_integrity_mismatch() {
        let temp = tempdir().expect("temp directory should be created");
        let source = temp.path().join("source.bin");
        fs::write(&source, b"payload").expect("source should be written");
        let cache = temp.path().join("cache");
        let source_uri = source.display().to_string();

        materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source_uri,
                "target_artifact_name": "optional-input",
                "required": false
            }]),
            &cache,
            "prefetch-prime",
        )
        .await
        .expect("local file should prime the prefetch cache");

        let result = materialize_direct_plan(
            json!([{
                "kind": "file_copy",
                "source_uri": source.display().to_string(),
                "target_artifact_name": "optional-input",
                "required": false,
                "expected_sha256": "0".repeat(64)
            }]),
            &cache,
            "prefetch-cache-hit",
        )
        .await
        .expect("optional cached integrity mismatch should not fail prefetch");

        assert_eq!(result["artifacts"], json!([]));
        assert_eq!(result["stats"]["downloaded"], 0);
        assert_eq!(result["stats"]["cached"], 0);
        assert_eq!(result["stats"]["optional_errors"], 1);
    }

    #[tokio::test]
    async fn rejects_prefetch_json_with_the_wrong_shape() {
        let plan = read_prefetch_plan(br#"{"source_uri":"file:///tmp/input"}"#.as_slice())
            .expect("JSON object itself is valid");
        let temp = tempdir().expect("temp directory should be created");

        let error = materialize_direct_plan(plan, temp.path(), "prefetch-test")
            .await
            .expect_err("prefetch plan must be an array");

        assert!(error.to_string().contains("invalid prefetch plan"));
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

    #[test]
    fn preserves_integrity_fields_for_http_materialization() {
        let item: DirectPlanItem = serde_json::from_value(json!({
            "kind": "http_fetch",
            "source_uri": "https://assets.example.com/input.bin",
            "target_artifact_name": "input",
            "expected_sha256": "a".repeat(64),
            "expected_size_bytes": 1024
        }))
        .expect("direct plan should parse");

        assert!(item.plan.expected_sha256.is_none());
        let worker_item = worker_plan_item(&item);
        assert_eq!(worker_item.expected_sha256, Some("a".repeat(64)));
        assert_eq!(worker_item.expected_size_bytes, Some(1024));
    }

    #[test]
    fn preserves_integrity_fields_for_local_materialization() {
        let item: DirectPlanItem = serde_json::from_value(json!({
            "kind": "file_copy",
            "source_uri": "/inputs/input.bin",
            "target_artifact_name": "input",
            "expected_sha256": "b".repeat(64),
            "expected_size_bytes": 2048
        }))
        .expect("direct plan should parse");

        let worker_item = worker_plan_item(&item);
        assert_eq!(worker_item.expected_sha256, Some("b".repeat(64)));
        assert_eq!(worker_item.expected_size_bytes, Some(2048));
        assert_eq!(item.expected_sha256, Some("b".repeat(64)));
        assert_eq!(item.expected_size_bytes, Some(2048));
    }

    #[test]
    fn preserves_integrity_fields_for_object_store_materialization() {
        let item: DirectPlanItem = serde_json::from_value(json!({
            "kind": "object_store_fetch",
            "source_uri": "s3://bucket/input.bin",
            "target_artifact_name": "input",
            "expected_sha256": "c".repeat(64),
            "expected_size_bytes": 4096
        }))
        .expect("direct plan should parse");

        let worker_item = worker_plan_item(&item);
        assert_eq!(worker_item.expected_sha256, Some("c".repeat(64)));
        assert_eq!(worker_item.expected_size_bytes, Some(4096));
        assert!(worker_item.effective_cache_key().starts_with("sha256:"));
    }
}
