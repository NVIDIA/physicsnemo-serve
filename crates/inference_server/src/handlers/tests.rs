/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::config::ServerConfig;
use crate::state::AppState;
use axum::body::to_bytes;
use axum::http::header;
use serde_json::json;
use std::process::Command;
use std::sync::Arc;
use tokio::fs;

fn create_mock_state() -> Arc<AppState> {
    create_mock_state_with_swagger_cdn(None)
}

fn create_mock_state_with_swagger_cdn(swagger_cdn_url: Option<&str>) -> Arc<AppState> {
    let config = ServerConfig {
        addr: "127.0.0.1:8080".parse().unwrap(),
        redis_url: "redis://127.0.0.1:6379".to_string(),
        redis_stream: "inference".to_string(),
        prefetch_stream: "prefetch".to_string(),
        use_prefetch: true,
        plugin_dirs: vec![],
        enabled_plugin_id: None,
        artifact_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-handler-artifacts-{}",
            uuid::Uuid::new_v4()
        )),
        default_output_dir: std::env::temp_dir().join(format!(
            "physicsnemo-serve-handler-outputs-{}",
            uuid::Uuid::new_v4()
        )),
        artifact_retention_hours: 24,
        artifact_cleanup_interval_secs: 30,
        cors_allowed_origins: vec![],
        max_body_size: 2 * 1024 * 1024,
        stream_prefix: String::new(),
        swagger_cdn_url: swagger_cdn_url.map(ToString::to_string),
        python_runtime_envs: std::collections::HashMap::new(),
    };
    Arc::new(AppState::new_for_testing(config))
}

fn create_tiny_zarr_dataset(path: &StdPath) {
    let status = Command::new("python")
        .arg("-c")
        .arg(
            r#"
from pathlib import Path
import numpy as np
import xarray as xr

path = Path(__import__("sys").argv[1])
ds = xr.Dataset(
    {
        "temperature": (("time",), np.array([280.0, 281.5], dtype=np.float32)),
        "humidity": (("time",), np.array([0.25, 0.40], dtype=np.float32)),
    },
    coords={"time": np.array([0, 1], dtype=np.int64)},
)
ds.to_zarr(path, mode="w")
"#,
        )
        .arg(path.display().to_string())
        .status()
        .expect("python should create zarr dataset");
    assert!(status.success(), "expected dataset creation to succeed");
}

#[test]
fn test_workflow_not_found_hint_points_to_real_endpoint() {
    let hint_source = "Use GET /v1/workflows to see available workflows";
    let correct_hint = "Use GET /v1/infer/workflows to see available workflows";
    assert_ne!(
        hint_source, correct_hint,
        "sanity: the old and new hints should differ"
    );

    let err_body = json!({
        "error": "Workflow 'nonexistent' not found",
        "hint": correct_hint,
    });
    let hint = err_body["hint"].as_str().unwrap();
    assert!(
        hint.contains("/v1/infer/workflows"),
        "Hint must reference the real endpoint /v1/infer/workflows, got: {}",
        hint
    );
    assert!(
        !hint.contains("/v1/workflows\""),
        "Hint must NOT reference non-existent /v1/workflows"
    );
}

#[test]
fn test_get_timestamp_returns_millisecond_precision() {
    let ts1 = get_timestamp();
    let parsed: u128 = ts1.parse().expect("timestamp should be a number");
    assert!(
        parsed > 1_000_000_000_000,
        "expected millisecond precision (>1e12), got {}",
        parsed
    );
}

#[test]
fn test_get_timestamp_no_panic_returns_valid_number() {
    let ts = get_timestamp();
    assert!(
        ts.parse::<u128>().is_ok(),
        "timestamp should be a valid integer"
    );
}

#[tokio::test]
async fn docs_handler_uses_default_swagger_cdn() {
    let state = create_mock_state();

    let Html(html) = get_docs(State(state)).await;

    assert!(html.contains("https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"));
    assert!(html.contains("https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"));
    assert!(html.contains("url: \"/openapi.json\""));
}

#[tokio::test]
async fn docs_handler_uses_configured_swagger_cdn() {
    let state = create_mock_state_with_swagger_cdn(Some("https://example.test/swagger"));

    let Html(html) = get_docs(State(state)).await;

    assert!(html.contains("https://example.test/swagger/swagger-ui.css"));
    assert!(html.contains("https://example.test/swagger/swagger-ui-bundle.js"));
    assert!(!html.contains("https://unpkg.com/swagger-ui-dist@5"));
}

#[tokio::test]
async fn stage_pending_artifacts_cleans_up_partial_staging_on_failure() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-stage-rollback-{}",
        uuid::Uuid::new_v4()
    ));
    let incoming_dir = artifact_root.join(".incoming");
    fs::create_dir_all(&incoming_dir).await.unwrap();
    let first_upload = incoming_dir.join("mesh-a.tmp");
    fs::write(&first_upload, b"mesh-a").await.unwrap();
    let missing_upload = incoming_dir.join("mesh-b.tmp");
    let third_upload = incoming_dir.join("mesh-c.tmp");
    fs::write(&third_upload, b"mesh-c").await.unwrap();
    let run_id = "run-stage-rollback";

    let result = stage_pending_artifacts(
        artifact_root.clone(),
        run_id,
        vec![
            PendingUploadedFile {
                field_name: "design_stl".to_string(),
                original_filename: Some("mesh-a.stl".to_string()),
                media_type: "model/stl".to_string(),
                temp_path: first_upload,
                size_bytes: 6,
            },
            PendingUploadedFile {
                field_name: "design_stl".to_string(),
                original_filename: Some("mesh-b.stl".to_string()),
                media_type: "model/stl".to_string(),
                temp_path: missing_upload,
                size_bytes: 6,
            },
            PendingUploadedFile {
                field_name: "design_stl".to_string(),
                original_filename: Some("mesh-c.stl".to_string()),
                media_type: "model/stl".to_string(),
                temp_path: third_upload.clone(),
                size_bytes: 6,
            },
        ],
    )
    .await;

    assert!(
        result.is_err(),
        "staging should fail when one upload is missing"
    );
    assert!(
        !fs::try_exists(artifact_root.join(run_id)).await.unwrap(),
        "partial staging must roll back the run directory on failure"
    );
    assert!(
        !fs::try_exists(third_upload).await.unwrap(),
        "rollback must also clean up later temp uploads that were never staged"
    );
}

#[tokio::test]
async fn artifact_download_streams_named_artifact_with_declared_media_type() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-artifacts-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root.join("run-1").join("pressure_field.npz");
    fs::create_dir_all(artifact_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&artifact_path, b"npz-payload").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-1",
        &json!({
            "output_path": artifact_path.display().to_string(),
            "artifacts": [
                {
                    "name": "pressure_field",
                    "media_type": "application/x-npz",
                    "storage_path": artifact_path.display().to_string()
                }
            ]
        }),
        "pressure_field",
    )
    .await
    .expect("artifact response should be built");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-npz"
    );
    assert!(
        response
            .headers()
            .get(header::CONTENT_DISPOSITION)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("pressure_field.npz")
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"npz-payload");
}

#[tokio::test]
async fn artifact_download_uses_output_path_for_primary_fallback() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-primary-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root.join("run-2").join("forecast.zarr.zip");
    fs::create_dir_all(artifact_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&artifact_path, b"zip-payload").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-2",
        &json!({
            "output_path": artifact_path.display().to_string(),
            "artifacts": [
                {
                    "name": "forecast_dataset",
                    "media_type": "application/zip"
                }
            ]
        }),
        "primary",
    )
    .await
    .expect("primary artifact response should be built");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/zip"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"zip-payload");
}

#[tokio::test]
async fn artifact_download_reads_artifact_from_nested_result_payload() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-nested-{}",
        uuid::Uuid::new_v4()
    ));
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-nested-output-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root.join("run-3").join("pressure_field.npz");
    fs::create_dir_all(artifact_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&artifact_path, b"nested-npz").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &output_root,
        "run-3",
        &json!({
            "workflow_id": "multipart-demo",
            "artifacts": [],
            "result": {
                "output_path": artifact_path.display().to_string(),
                "artifacts": [
                    {
                        "name": "pressure_field",
                        "media_type": "application/x-npz",
                        "storage_path": artifact_path.display().to_string()
                    }
                ]
            }
        }),
        "pressure_field",
    )
    .await
    .expect("nested artifact response should be built");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-npz"
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"nested-npz");
}

#[tokio::test]
async fn artifact_download_allows_paths_under_default_output_dir() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-artifact-root-{}",
        uuid::Uuid::new_v4()
    ));
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-output-root-{}",
        uuid::Uuid::new_v4()
    ));
    let output_path = output_root.join("run-4").join("pressure_field.npz");
    fs::create_dir_all(output_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&output_path, b"output-root-npz").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &output_root,
        "run-4",
        &json!({
            "workflow_id": "multipart-demo",
            "result": {
                "artifacts": [
                    {
                        "name": "pressure_field",
                        "media_type": "application/x-npz",
                        "storage_path": output_path.display().to_string()
                    }
                ],
                "output_path": output_path.display().to_string()
            }
        }),
        "pressure_field",
    )
    .await
    .expect("artifact stored under default output dir should be downloadable");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"output-root-npz");
}

#[tokio::test]
async fn artifact_download_streams_generated_netcdf_artifact() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-netcdf-root-{}",
        uuid::Uuid::new_v4()
    ));
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-netcdf-output-{}",
        uuid::Uuid::new_v4()
    ));
    let output_path = output_root.join("run-5").join("forecast.nc");
    fs::create_dir_all(output_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&output_path, b"netcdf-bytes").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &output_root,
        "run-5",
        &json!({
            "workflow_id": "demo-prefetch",
            "artifacts": [
                {
                    "name": "forecast_netcdf",
                    "media_type": "application/x-netcdf",
                    "storage_path": output_path.display().to_string(),
                    "filename": "forecast.nc"
                }
            ],
            "output_path": output_path.display().to_string()
        }),
        "forecast_netcdf",
    )
    .await
    .expect("generated netcdf artifact should be downloadable");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-netcdf"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"netcdf-bytes");
}

#[tokio::test]
#[ignore = "requires Python runtime"]
async fn dataset_download_exports_zarr_artifact_to_subset_netcdf() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-dataset-artifacts-{}",
        uuid::Uuid::new_v4()
    ));
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-dataset-output-{}",
        uuid::Uuid::new_v4()
    ));
    let dataset_path = output_root.join("run-6").join("forecast.zarr");
    fs::create_dir_all(dataset_path.parent().unwrap())
        .await
        .unwrap();
    create_tiny_zarr_dataset(&dataset_path);

    let state = create_mock_state();
    let response = build_artifact_download_response_with_query(
        &artifact_root,
        &output_root,
        "run-6",
        &json!({
            "workflow": "demo-prefetch",
            "artifacts": [
                {
                    "name": "forecast_dataset",
                    "media_type": "application/x-zarr",
                    "storage_path": dataset_path.display().to_string(),
                    "filename": "forecast.zarr"
                }
            ],
            "output_path": dataset_path.display().to_string()
        }),
        "forecast_dataset",
        &ResultQuery {
            artifact: Some("forecast_dataset".to_string()),
            format: Some("netcdf".to_string()),
            vars: Some("temperature".to_string()),
        },
        state.as_ref(),
    )
    .await
    .expect("subset netcdf export should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-netcdf"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(!body.is_empty(), "expected generated netcdf body");
}

#[tokio::test]
#[ignore = "requires Python runtime"]
async fn dataset_download_exports_zarr_artifact_to_zip() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-zarrzip-artifacts-{}",
        uuid::Uuid::new_v4()
    ));
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-zarrzip-output-{}",
        uuid::Uuid::new_v4()
    ));
    let dataset_path = output_root.join("run-7").join("forecast.zarr");
    fs::create_dir_all(dataset_path.parent().unwrap())
        .await
        .unwrap();
    create_tiny_zarr_dataset(&dataset_path);

    let state = create_mock_state();
    let response = build_artifact_download_response_with_query(
        &artifact_root,
        &output_root,
        "run-7",
        &json!({
            "workflow": "demo-prefetch",
            "artifacts": [
                {
                    "name": "forecast_dataset",
                    "media_type": "application/x-zarr",
                    "storage_path": dataset_path.display().to_string(),
                    "filename": "forecast.zarr"
                }
            ],
            "output_path": dataset_path.display().to_string()
        }),
        "forecast_dataset",
        &ResultQuery {
            artifact: Some("forecast_dataset".to_string()),
            format: Some("zarr_zip".to_string()),
            vars: None,
        },
        state.as_ref(),
    )
    .await
    .expect("zarr zip export should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/zip"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(!body.is_empty(), "expected generated zip body");
}

#[tokio::test]
async fn artifact_download_rejects_paths_outside_artifact_root() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-root-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&artifact_root).await.unwrap();
    let output_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-output-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&output_root).await.unwrap();

    let outside_dir = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-outside-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&outside_dir).await.unwrap();
    let outside_file = outside_dir.join("secret.bin");
    fs::write(&outside_file, b"blocked").await.unwrap();

    let err = build_artifact_download_response(
        &artifact_root,
        &output_root,
        "run-3",
        &json!({
            "artifacts": [
                {
                    "name": "secret",
                    "media_type": "application/octet-stream",
                    "storage_path": outside_file.display().to_string()
                }
            ]
        }),
        "secret",
    )
    .await
    .unwrap_err();

    let message = format!("{err:#}");
    assert!(message.contains("artifact 'secret' for run 'run-3' is unavailable"));
    assert!(message.contains("outside the configured download roots"));
}

#[tokio::test]
async fn artifact_download_serves_archive_alias_from_output_archive() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-archive-root-{}",
        uuid::Uuid::new_v4()
    ));
    let archive_path = artifact_root.join("run-archive").join("forecast.zip");
    fs::create_dir_all(archive_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&archive_path, b"archive-payload").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-archive",
        &json!({
            "output_archive": archive_path.display().to_string()
        }),
        "archive",
    )
    .await
    .expect("archive alias should resolve output_archive");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::CONTENT_DISPOSITION)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("forecast.zip")
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body.as_ref(), b"archive-payload");
}

#[tokio::test]
async fn artifact_download_falls_back_to_octet_stream_for_invalid_media_type() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-invalid-media-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root.join("run-invalid-media").join("payload.bin");
    fs::create_dir_all(artifact_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&artifact_path, b"payload").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-invalid-media",
        &json!({
            "artifacts": [
                {
                    "name": "payload",
                    "media_type": "text/plain\ncharset=utf-8",
                    "storage_path": artifact_path.display().to_string()
                }
            ]
        }),
        "payload",
    )
    .await
    .expect("invalid media type should fall back to octet-stream");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/octet-stream"
    );
}

#[tokio::test]
async fn artifact_download_falls_back_to_attachment_for_invalid_filename_header() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-invalid-filename-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root
        .join("run-invalid-filename")
        .join("payload.bin");
    fs::create_dir_all(artifact_path.parent().unwrap())
        .await
        .unwrap();
    fs::write(&artifact_path, b"payload").await.unwrap();

    let response = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-invalid-filename",
        &json!({
            "artifacts": [
                {
                    "name": "payload",
                    "media_type": "application/octet-stream",
                    "filename": "bad\nname.bin",
                    "storage_path": artifact_path.display().to_string()
                }
            ]
        }),
        "payload",
    )
    .await
    .expect("invalid filename header should fall back to attachment");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment"
    );
}

#[tokio::test]
async fn artifact_download_rejects_non_regular_file_artifact_path() {
    let artifact_root = std::env::temp_dir().join(format!(
        "physicsnemo-serve-handler-read-error-{}",
        uuid::Uuid::new_v4()
    ));
    let artifact_path = artifact_root.join("run-read-error").join("dataset.zarr");
    fs::create_dir_all(&artifact_path).await.unwrap();

    let err = build_artifact_download_response(
        &artifact_root,
        &artifact_root,
        "run-read-error",
        &json!({
            "artifacts": [
                {
                    "name": "dataset",
                    "media_type": "application/x-zarr",
                    "filename": "dataset.zarr",
                    "storage_path": artifact_path.display().to_string()
                }
            ]
        }),
        "dataset",
    )
    .await
    .expect_err("directory-backed artifacts should fail validation");

    let message = format!("{err:#}");
    assert!(
        message.contains("artifact 'dataset' for run 'run-read-error' is unavailable"),
        "unexpected message: {message}"
    );
    assert!(
        message.contains("is not a regular file"),
        "unexpected message: {message}"
    );
}
