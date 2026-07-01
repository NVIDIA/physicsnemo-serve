/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public factory functions for constructing backends.
//!
//! Use [`build_sync_zarr_backend`] for the default (panicking) factory, or
//! [`try_build_sync_zarr_backend`] for the fallible variant.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::SyncZarrBackend;
use crate::core::contracts::{
    ArrayRegistry, BufferPool, ChunkKeyRegistry, ChunkPlanner, ChunkWriter, CopyEngine,
    MetadataConsolidator, WorkScheduler, ZarrLayoutAdapter,
};
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    BufferPoolConfig, CoordMap, InputStabilityPolicy, SizeOverride, WriteExecutionConfig,
};
use crate::runtime::array_registry::InMemoryArrayRegistry;
use crate::runtime::buffer_pool::MemoryBufferPool;
use crate::runtime::coordinator::{WriteCoordinator, WriteCoordinatorComponents};
use crate::runtime::copy_engine::DefaultCopyEngine;
use crate::runtime::planner::MixedRadixChunkPlanner;
use crate::runtime::registry::InMemoryChunkKeyRegistry;
use crate::runtime::thread_pool::RayonWorkScheduler;
use crate::zarr::metadata::LocalFsMetadataConsolidator;
use crate::zarr::writer::LocalFsChunkWriter;
use crate::zarr::zarr_layout::DefaultZarrLayoutAdapter;

fn default_dataset_root() -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let pid = std::process::id();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("e2s_zarr_io_dataset_{pid}_{id}.zarr"))
}

/// Returns the crate version string embedded at compile time.
///
/// The returned value matches the `version` field in `Cargo.toml` and is
/// suitable for exposing through Python bindings or CLI output.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Configuration for constructing a [`SyncZarrBackend`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct SyncZarrBackendConfig {
    /// Dataset root path where chunk/metadata files are written.
    ///
    /// When `None`, backend uses a unique temporary root.
    pub dataset_root: Option<PathBuf>,
    /// Write execution configuration (format, scheduler, planner, etc.).
    pub write_execution: WriteExecutionConfig,
    /// Buffer pool configuration.
    pub buffer_pool: BufferPoolConfig,
}

impl SyncZarrBackendConfig {
    /// Validate API-level backend configuration constraints.
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        self.write_execution.zarr_target.validate()?;
        if self.write_execution.num_threads == 0 {
            return Err(SyncWriteError::Validation {
                message: "num_threads must be greater than 0".to_string(),
            });
        }
        if self.write_execution.input_stability_policy != InputStabilityPolicy::StrictGilHold {
            return Err(SyncWriteError::UnsupportedInputStabilityPolicy {
                policy: format!("{:?}", self.write_execution.input_stability_policy),
            });
        }
        if self.write_execution.queue_capacity == 0 {
            return Err(SyncWriteError::Validation {
                message: "queue_capacity must be greater than 0".to_string(),
            });
        }
        if self
            .write_execution
            .parallel_coords_config
            .default_parallel_coord_names
            .is_empty()
        {
            return Err(SyncWriteError::Validation {
                message: "default_parallel_coord_names must not be empty".to_string(),
            });
        }
        if let Some(explicit) = &self.write_execution.parallel_coords_config.parallel_coords {
            if explicit.is_empty() {
                return Err(SyncWriteError::Validation {
                    message: "parallel_coords override must not be empty when provided".to_string(),
                });
            }
        }
        if self.buffer_pool.max_pool_buffers == 0 {
            return Err(SyncWriteError::Validation {
                message: "max_pool_buffers must be greater than 0".to_string(),
            });
        }
        if self.buffer_pool.max_pool_bytes == 0 {
            return Err(SyncWriteError::Validation {
                message: "max_pool_bytes must be greater than 0".to_string(),
            });
        }
        if let Some(path) = &self.dataset_root {
            if path.as_os_str().is_empty() {
                return Err(SyncWriteError::Validation {
                    message: "dataset_root must not be empty when provided".to_string(),
                });
            }
        }
        if let SizeOverride::Fixed(pool_bytes) = self.buffer_pool.pool_buffer_bytes {
            if pool_bytes == 0 {
                return Err(SyncWriteError::Validation {
                    message: "pool_buffer_bytes must be greater than 0 when provided".to_string(),
                });
            }
        }
        if self.buffer_pool.first_write_sizing.min_hot_slab_buffers == 0 {
            return Err(SyncWriteError::Validation {
                message: "min_hot_slab_buffers must be greater than 0".to_string(),
            });
        }
        if self.buffer_pool.first_write_sizing.max_warm_to_hot_ratio == 0 {
            return Err(SyncWriteError::Validation {
                message: "max_warm_to_hot_ratio must be greater than 0".to_string(),
            });
        }
        if let SizeOverride::Fixed(hot) = self.buffer_pool.hot_slab_buffers {
            if hot == 0 {
                return Err(SyncWriteError::Validation {
                    message: "hot_slab_buffers must be greater than 0 when provided".to_string(),
                });
            }
        }
        if let SizeOverride::Fixed(warm) = self.buffer_pool.warm_slab_buffers {
            if warm == 0 {
                return Err(SyncWriteError::Validation {
                    message: "warm_slab_buffers must be greater than 0 when provided".to_string(),
                });
            }
        }
        if let (SizeOverride::Fixed(hot), SizeOverride::Fixed(warm)) = (
            self.buffer_pool.hot_slab_buffers,
            self.buffer_pool.warm_slab_buffers,
        ) {
            let total = hot
                .checked_add(warm)
                .ok_or_else(|| SyncWriteError::Validation {
                    message: "hot_slab_buffers + warm_slab_buffers overflowed usize".to_string(),
                })?;
            if total > self.buffer_pool.max_pool_buffers {
                return Err(SyncWriteError::Validation {
                    message: "hot_slab_buffers + warm_slab_buffers must be <= max_pool_buffers"
                        .to_string(),
                });
            }
        }
        if !self.buffer_pool.hot_slab_ready_timeout_seconds.is_finite()
            || self.buffer_pool.hot_slab_ready_timeout_seconds <= 0.0
        {
            return Err(SyncWriteError::Validation {
                message: "hot_slab_ready_timeout_seconds must be finite and > 0".to_string(),
            });
        }
        if !self.buffer_pool.close_lease_timeout_seconds.is_finite()
            || self.buffer_pool.close_lease_timeout_seconds <= 0.0
        {
            return Err(SyncWriteError::Validation {
                message: "close_lease_timeout_seconds must be finite and > 0".to_string(),
            });
        }
        if self.buffer_pool.max_transient_buffer_bytes == Some(0) {
            return Err(SyncWriteError::Validation {
                message: "max_transient_buffer_bytes must not be Some(0); \
                          use None for unlimited or a positive value"
                    .to_string(),
            });
        }
        if self.buffer_pool.max_inflight_transient_bytes == Some(0) {
            return Err(SyncWriteError::Validation {
                message: "max_inflight_transient_bytes must not be Some(0); \
                          use None for unlimited or a positive value"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Build a [`SyncZarrBackend`] from the given configuration (fallible).
pub fn new_sync_zarr_backend(
    config: SyncZarrBackendConfig,
) -> Result<SyncZarrBackend, SyncWriteError> {
    try_build_sync_zarr_backend(config)
}

/// Build a [`SyncZarrBackend`] from the given configuration.
///
/// # Panics
///
/// Panics if backend configuration is invalid. Prefer
/// [`try_build_sync_zarr_backend`] for fallible construction.
#[deprecated(
    since = "0.1.0",
    note = "prefer try_build_sync_zarr_backend (or new_sync_zarr_backend) to avoid panic-based construction"
)]
#[must_use]
pub fn build_sync_zarr_backend(config: SyncZarrBackendConfig) -> SyncZarrBackend {
    try_build_sync_zarr_backend(config)
        .expect("invalid zarr target configuration for sync zarr backend")
}

/// Produce a deterministic ordering for explicit parallel coord names.
///
/// Keys present in `default_parallel_coord_names` come first (preserving the
/// default order), followed by any remaining keys sorted alphabetically.
/// This ensures that dimension ordering in metadata and the planner is
/// reproducible regardless of the iteration order of the `CoordMap` container.
fn canonicalize_parallel_coord_names(explicit: &CoordMap, default_order: &[String]) -> Vec<String> {
    let mut ordered = Vec::with_capacity(explicit.len());
    let mut seen = HashSet::with_capacity(explicit.len());
    // First pass: add keys that appear in `default_order` (preserving default order)
    for name in default_order {
        if explicit.contains_key(name.as_str()) && seen.insert(name.clone()) {
            ordered.push(name.clone());
        }
    }
    // Second pass: add any remaining keys not in `default_order` (alphabetical)
    let mut remaining: Vec<String> = explicit
        .keys()
        .filter(|k| !seen.contains(k.as_str()))
        .cloned()
        .collect();
    remaining.sort();
    ordered.extend(remaining);
    ordered
}

/// Build a [`SyncZarrBackend`] from the given configuration (fallible).
pub fn try_build_sync_zarr_backend(
    config: SyncZarrBackendConfig,
) -> Result<SyncZarrBackend, SyncWriteError> {
    let dataset_root = config
        .dataset_root
        .clone()
        .unwrap_or_else(default_dataset_root);
    config.validate()?;
    let write_execution = config.write_execution;
    let queue_capacity = write_execution.queue_capacity;
    let num_threads = write_execution.num_threads;
    let fsync_policy = write_execution.fsync_policy;
    let buffer_pool_config = config.buffer_pool;
    let default_close_timeout_seconds = buffer_pool_config.close_lease_timeout_seconds;
    let zarr_target = write_execution.zarr_target;
    let metadata_parallel_coord_names: Vec<String> =
        if let Some(explicit) = &write_execution.parallel_coords_config.parallel_coords {
            canonicalize_parallel_coord_names(
                explicit,
                &write_execution
                    .parallel_coords_config
                    .default_parallel_coord_names,
            )
        } else {
            write_execution
                .parallel_coords_config
                .default_parallel_coord_names
                .clone()
        };

    let array_registry: Arc<dyn ArrayRegistry> = Arc::new(InMemoryArrayRegistry::new());
    let planner: Arc<dyn ChunkPlanner> = Arc::new(MixedRadixChunkPlanner::new(write_execution));
    let registry: Arc<dyn ChunkKeyRegistry> = Arc::new(InMemoryChunkKeyRegistry::new());
    let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
    let buffer_pool: Arc<dyn BufferPool> = Arc::new(MemoryBufferPool::new(buffer_pool_config));
    let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
    let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
        Arc::new(DefaultZarrLayoutAdapter::new(zarr_target)?);
    let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(LocalFsChunkWriter::with_fsync_policy(
        dataset_root.clone(),
        Arc::clone(&layout_adapter),
        fsync_policy,
    ));
    let metadata_consolidator: Arc<dyn MetadataConsolidator> = Arc::new(
        LocalFsMetadataConsolidator::with_fsync_policy(dataset_root, fsync_policy),
    );

    let coordinator = Arc::new(WriteCoordinator::try_new_with_num_threads(
        WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer,
            metadata_consolidator,
            layout_adapter,
            parallel_coord_names: metadata_parallel_coord_names,
            queue_capacity,
        },
        num_threads,
    )?);
    Ok(SyncZarrBackend::new_with_close_timeout(
        coordinator,
        array_registry,
        default_close_timeout_seconds,
    ))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::contracts::ZarrIoBackend;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        ArrayRegistration, CoordMap, CoordValues, InferenceWriteRequest, InputArray,
        InputArraySource, SizeOverride,
    };

    use super::{
        SyncZarrBackendConfig, canonicalize_parallel_coord_names, try_build_sync_zarr_backend,
    };

    fn unique_dataset_root(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock moved backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}.zarr"))
    }

    fn assert_validation_error_contains(config: SyncZarrBackendConfig, expected_message: &str) {
        let err = config
            .validate()
            .expect_err("configuration should fail validation");
        match err {
            SyncWriteError::Validation { message } => {
                assert!(
                    message.contains(expected_message),
                    "expected validation error containing '{expected_message}', got '{message}'"
                );
            }
            other => panic!(
                "expected SyncWriteError::Validation containing '{expected_message}', got {other:?}"
            ),
        }
    }

    #[test]
    fn backend_factory_writes_chunks_and_metadata_to_configured_dataset_root() {
        let dataset_root = unique_dataset_root("e2s_api_dataset_root");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        backend
            .write(InferenceWriteRequest {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: 4,
                    source: InputArraySource::HostBytes(vec![9, 8, 7, 6].into()),
                }],
            })
            .expect("write should succeed");
        backend.close(300.0).expect("close should succeed");

        assert!(dataset_root.join("temperature").join("0").exists());
        assert!(dataset_root.join(".zmetadata").exists());
        assert!(dataset_root.join("temperature").join(".zarray").exists());
    }

    #[test]
    fn backend_factory_applies_configured_default_close_timeout() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.close_lease_timeout_seconds = 12.5;
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");
        assert_eq!(backend.configured_close_timeout_seconds(), 12.5);
    }

    #[test]
    fn backend_factory_rejects_zero_num_threads() {
        let mut config = SyncZarrBackendConfig::default();
        config.write_execution.num_threads = 0;
        let err = match try_build_sync_zarr_backend(config) {
            Ok(_) => panic!("num_threads=0 must be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                SyncWriteError::Validation { ref message } if message.contains("num_threads")
            ),
            "expected num_threads validation error, got: {err:?}"
        );
    }

    #[test]
    fn backend_config_validate_rejects_zero_queue_capacity() {
        let mut config = SyncZarrBackendConfig::default();
        config.write_execution.queue_capacity = 0;
        assert_validation_error_contains(config, "queue_capacity must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_empty_default_parallel_coord_names() {
        let mut config = SyncZarrBackendConfig::default();
        config
            .write_execution
            .parallel_coords_config
            .default_parallel_coord_names = Vec::new();
        assert_validation_error_contains(config, "default_parallel_coord_names must not be empty");
    }

    #[test]
    fn backend_config_validate_rejects_empty_parallel_coords_override() {
        let mut config = SyncZarrBackendConfig::default();
        config
            .write_execution
            .parallel_coords_config
            .parallel_coords = Some(CoordMap::new());
        assert_validation_error_contains(
            config,
            "parallel_coords override must not be empty when provided",
        );
    }

    #[test]
    fn backend_factory_accepts_non_empty_parallel_coords_override() {
        let mut config = SyncZarrBackendConfig::default();
        let mut explicit = CoordMap::new();
        let _ = explicit.insert("ensemble".to_string(), CoordValues::I64(vec![0]));
        let _ = explicit.insert("time".to_string(), CoordValues::I64(vec![0]));
        config
            .write_execution
            .parallel_coords_config
            .parallel_coords = Some(explicit);
        let backend =
            try_build_sync_zarr_backend(config).expect("non-empty parallel_coords should validate");
        backend
            .close(1.0)
            .expect("close should succeed for backend created from valid override config");
    }

    #[test]
    fn backend_config_validate_rejects_empty_dataset_root_when_provided() {
        let config = SyncZarrBackendConfig {
            dataset_root: Some(std::path::PathBuf::from(OsString::new())),
            ..Default::default()
        };
        assert_validation_error_contains(config, "dataset_root must not be empty when provided");
    }

    #[test]
    fn backend_config_validate_rejects_zero_max_pool_buffers() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.max_pool_buffers = 0;
        assert_validation_error_contains(config, "max_pool_buffers must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_zero_max_pool_bytes() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.max_pool_bytes = 0;
        assert_validation_error_contains(config, "max_pool_bytes must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_zero_pool_buffer_bytes_override() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.pool_buffer_bytes = SizeOverride::Fixed(0);
        assert_validation_error_contains(
            config,
            "pool_buffer_bytes must be greater than 0 when provided",
        );
    }

    #[test]
    fn backend_config_validate_accepts_positive_pool_buffer_bytes_override() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.pool_buffer_bytes = SizeOverride::Fixed(4096);
        config
            .validate()
            .expect("positive pool_buffer_bytes override should validate");
    }

    #[test]
    fn backend_config_validate_rejects_zero_min_hot_slab_buffers() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.first_write_sizing.min_hot_slab_buffers = 0;
        assert_validation_error_contains(config, "min_hot_slab_buffers must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_zero_max_warm_to_hot_ratio() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.first_write_sizing.max_warm_to_hot_ratio = 0;
        assert_validation_error_contains(config, "max_warm_to_hot_ratio must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_zero_hot_slab_buffers_override() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.hot_slab_buffers = SizeOverride::Fixed(0);
        assert_validation_error_contains(config, "hot_slab_buffers must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_zero_warm_slab_buffers_override() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.warm_slab_buffers = SizeOverride::Fixed(0);
        assert_validation_error_contains(config, "warm_slab_buffers must be greater than 0");
    }

    #[test]
    fn backend_config_validate_rejects_hot_plus_warm_slab_overflow() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.hot_slab_buffers = SizeOverride::Fixed(usize::MAX);
        config.buffer_pool.warm_slab_buffers = SizeOverride::Fixed(1);
        config.buffer_pool.max_pool_buffers = usize::MAX;
        assert_validation_error_contains(
            config,
            "hot_slab_buffers + warm_slab_buffers overflowed usize",
        );
    }

    #[test]
    fn backend_config_validate_rejects_hot_plus_warm_slab_exceeding_max_pool_buffers() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.hot_slab_buffers = SizeOverride::Fixed(3);
        config.buffer_pool.warm_slab_buffers = SizeOverride::Fixed(2);
        config.buffer_pool.max_pool_buffers = 4;
        assert_validation_error_contains(
            config,
            "hot_slab_buffers + warm_slab_buffers must be <= max_pool_buffers",
        );
    }

    #[test]
    fn backend_config_validate_rejects_non_finite_hot_slab_ready_timeout() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.hot_slab_ready_timeout_seconds = f64::NAN;
        assert_validation_error_contains(
            config,
            "hot_slab_ready_timeout_seconds must be finite and > 0",
        );
    }

    #[test]
    fn backend_config_validate_rejects_non_positive_hot_slab_ready_timeout() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.hot_slab_ready_timeout_seconds = 0.0;
        assert_validation_error_contains(
            config,
            "hot_slab_ready_timeout_seconds must be finite and > 0",
        );
    }

    #[test]
    fn backend_config_validate_rejects_non_finite_close_lease_timeout() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.close_lease_timeout_seconds = f64::INFINITY;
        assert_validation_error_contains(
            config,
            "close_lease_timeout_seconds must be finite and > 0",
        );
    }

    #[test]
    fn backend_config_validate_rejects_non_positive_close_lease_timeout() {
        let mut config = SyncZarrBackendConfig::default();
        config.buffer_pool.close_lease_timeout_seconds = -1.0;
        assert_validation_error_contains(
            config,
            "close_lease_timeout_seconds must be finite and > 0",
        );
    }

    #[test]
    fn canonicalize_parallel_coord_names_preserves_default_order_then_sorts_remaining() {
        let mut explicit = CoordMap::new();
        let _ = explicit.insert("z_extra".to_string(), CoordValues::I64(vec![0]));
        let _ = explicit.insert("ensemble".to_string(), CoordValues::I64(vec![0]));
        let _ = explicit.insert("alpha_extra".to_string(), CoordValues::I64(vec![0]));
        let _ = explicit.insert("time".to_string(), CoordValues::I64(vec![0]));
        let default_order = vec![
            "time".to_string(),
            "lead_time".to_string(),
            "ensemble".to_string(),
        ];

        let ordered = canonicalize_parallel_coord_names(&explicit, &default_order);

        assert_eq!(
            ordered,
            vec![
                "time".to_string(),
                "ensemble".to_string(),
                "alpha_extra".to_string(),
                "z_extra".to_string(),
            ]
        );
    }

    #[test]
    fn canonicalize_parallel_coord_names_uses_hashset_membership_not_vec_contains() {
        let source = include_str!("api.rs");
        let canonicalize_section = source
            .split("fn canonicalize_parallel_coord_names")
            .nth(1)
            .expect("canonicalize_parallel_coord_names should be present");
        let canonicalize_block = canonicalize_section
            .split("/// Build a [`SyncZarrBackend`] from the given configuration (fallible).")
            .next()
            .expect("canonicalize_parallel_coord_names block should end before builder docs");
        assert!(
            canonicalize_block.contains("HashSet"),
            "dedup should use HashSet membership checks for O(1) lookups",
        );
        assert!(
            !canonicalize_block.contains("ordered.contains"),
            "dedup should not rely on Vec::contains O(n) lookups",
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_build_sync_zarr_backend_panics_for_invalid_config() {
        let mut config = SyncZarrBackendConfig::default();
        config.write_execution.num_threads = 0;

        let panic_result = std::panic::catch_unwind(|| {
            let _ = super::build_sync_zarr_backend(config);
        });

        assert!(
            panic_result.is_err(),
            "deprecated panic constructor should panic for invalid config"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_build_sync_zarr_backend_succeeds_for_valid_config() {
        let backend = super::build_sync_zarr_backend(SyncZarrBackendConfig::default());
        backend
            .close(1.0)
            .expect("close should succeed for backend created by deprecated constructor");
    }

    #[test]
    fn backend_factory_emits_multidimensional_v2_array_metadata_from_registered_coords() {
        let dataset_root = unique_dataset_root("e2s_api_multidim_metadata");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = total_coords.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
        let _ = total_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = total_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords.clone(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        let step0_coords = {
            let mut c = CoordMap::new();
            let _ = c.insert("time".to_string(), CoordValues::I64(vec![0]));
            let _ = c.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
            let _ = c.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
            let _ = c.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
            c
        };
        let step1_coords = {
            let mut c = CoordMap::new();
            let _ = c.insert("time".to_string(), CoordValues::I64(vec![1]));
            let _ = c.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
            let _ = c.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
            let _ = c.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
            c
        };
        let step0_payload: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let step1_payload: Vec<u8> = [5.0_f32, 6.0, 7.0, 8.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();

        backend
            .write(InferenceWriteRequest {
                coords: step0_coords,
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: step0_payload.len(),
                    source: InputArraySource::HostBytes(step0_payload.into()),
                }],
            })
            .expect("step0 write should succeed");
        backend
            .write(InferenceWriteRequest {
                coords: step1_coords,
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: step1_payload.len(),
                    source: InputArraySource::HostBytes(step1_payload.into()),
                }],
            })
            .expect("step1 write should succeed");
        backend.close(300.0).expect("close should succeed");

        let consolidated = std::fs::read_to_string(dataset_root.join(".zmetadata"))
            .expect("v2 consolidated metadata should be present");
        assert!(
            consolidated.contains("\"temperature/.zarray\":"),
            "expected consolidated metadata entry for temperature/.zarray, got: {consolidated}"
        );
        assert!(
            consolidated.contains("\"shape\":[2,1,2,2]"),
            "expected multidimensional shape from add_array coords, got: {consolidated}"
        );
        assert!(
            consolidated.contains("\"chunks\":[1,1,2,2]"),
            "expected chunking semantics (parallel coords chunked by 1), got: {consolidated}"
        );
        assert!(
            dataset_root.join("temperature").join("0.0.0.0").exists(),
            "expected v2 multidimensional tuple chunk key temperature/0.0.0.0"
        );
        assert!(
            dataset_root.join("temperature").join("1.0.0.0").exists(),
            "expected v2 multidimensional tuple chunk key temperature/1.0.0.0"
        );
    }

    #[test]
    fn add_array_persists_schema_and_coordinate_arrays_before_first_write() {
        let dataset_root = unique_dataset_root("e2s_api_add_array_persists_schema");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = total_coords.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = total_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = total_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords,
                array_names: vec!["t2m".to_string(), "tcwv".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        // Registration metadata is written on a background thread; close() joins
        // the thread before returning, making file assertions safe.
        backend.close(300.0).expect("close should succeed");

        assert!(
            dataset_root.join(".zgroup").exists(),
            "expected root .zgroup to be persisted during add_array"
        );
        assert!(
            dataset_root.join("t2m").join(".zarray").exists(),
            "expected data-array metadata for t2m during add_array"
        );
        assert!(
            dataset_root.join("tcwv").join(".zarray").exists(),
            "expected data-array metadata for tcwv during add_array"
        );

        for coord_name in ["time", "lead_time", "lat", "lon"] {
            assert!(
                dataset_root.join(coord_name).join(".zarray").exists(),
                "expected coordinate metadata for {coord_name} during add_array"
            );
            assert!(
                dataset_root.join(coord_name).join("0").exists(),
                "expected coordinate chunk payload for {coord_name} during add_array"
            );
        }

        let time_chunk = std::fs::read(dataset_root.join("time").join("0"))
            .expect("time coordinate chunk should be readable");
        let lead_time_chunk = std::fs::read(dataset_root.join("lead_time").join("0"))
            .expect("lead_time coordinate chunk should be readable");
        let lat_chunk = std::fs::read(dataset_root.join("lat").join("0"))
            .expect("lat chunk should be readable");
        let lon_chunk = std::fs::read(dataset_root.join("lon").join("0"))
            .expect("lon chunk should be readable");

        let decode_i64 = |bytes: &[u8]| -> Vec<i64> {
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let mut raw = [0_u8; 8];
                    raw.copy_from_slice(chunk);
                    i64::from_le_bytes(raw)
                })
                .collect()
        };
        let decode_f64 = |bytes: &[u8]| -> Vec<f64> {
            bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let mut raw = [0_u8; 8];
                    raw.copy_from_slice(chunk);
                    f64::from_le_bytes(raw)
                })
                .collect()
        };

        assert_eq!(decode_i64(&time_chunk), vec![0, 1]);
        assert_eq!(decode_i64(&lead_time_chunk), vec![0, 6, 12]);
        assert_eq!(decode_f64(&lat_chunk), vec![10.0, 20.0]);
        assert_eq!(decode_f64(&lon_chunk), vec![30.0, 40.0]);
    }

    #[test]
    fn add_array_persists_utf8_coordinate_array_for_v2() {
        let dataset_root = unique_dataset_root("e2s_api_add_array_utf8_coord_v2");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = total_coords.insert(
            "member".to_string(),
            CoordValues::Utf8(vec![
                "control".to_string(),
                "pert01".to_string(),
                "pert02".to_string(),
            ]),
        );
        let _ = total_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = total_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords,
                array_names: vec!["t2m".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed for v2 Utf8 coord materialization");
        // Registration metadata is written on a background thread; close() joins it.
        backend
            .close(300.0)
            .expect("close should succeed before file assertions");

        let member_zarray = std::fs::read_to_string(dataset_root.join("member").join(".zarray"))
            .expect("member/.zarray should be present");
        assert!(
            member_zarray.contains("\"dtype\":\"<U7\""),
            "expected fixed-width Utf8 dtype <U7 in member/.zarray, got: {member_zarray}"
        );
        assert!(
            member_zarray.contains("\"shape\":[3]"),
            "expected shape [3] for member coord, got: {member_zarray}"
        );
        assert!(
            member_zarray.contains("\"chunks\":[3]"),
            "expected chunk shape [3] for member coord, got: {member_zarray}"
        );

        let member_chunk = std::fs::read(dataset_root.join("member").join("0"))
            .expect("member coordinate chunk should be readable");
        assert_eq!(
            member_chunk.len(),
            3 * 7 * 4,
            "expected UTF-32LE fixed-width payload bytes for 3 values × 7 codepoints"
        );

        let decode_utf32le_fixed = |bytes: &[u8], width: usize| -> Vec<String> {
            bytes
                .chunks_exact(width * 4)
                .map(|item| {
                    let mut out = String::new();
                    for cp_bytes in item.chunks_exact(4) {
                        let mut raw = [0_u8; 4];
                        raw.copy_from_slice(cp_bytes);
                        let cp = u32::from_le_bytes(raw);
                        if cp == 0 {
                            break;
                        }
                        let ch = char::from_u32(cp)
                            .expect("UTF-32LE coord payload should contain valid codepoints");
                        out.push(ch);
                    }
                    out
                })
                .collect()
        };

        assert_eq!(
            decode_utf32le_fixed(&member_chunk, 7),
            vec![
                "control".to_string(),
                "pert01".to_string(),
                "pert02".to_string(),
            ]
        );
    }

    #[test]
    fn add_array_persists_utf8_coordinate_array_for_v3() {
        let dataset_root = unique_dataset_root("e2s_api_add_array_utf8_coord_v3");
        let mut config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        config.write_execution.zarr_target.zarr_format = crate::core::types::ZarrFormat::V3;
        config.write_execution.zarr_target.chunk_key_encoding =
            crate::core::types::ChunkKeyEncoding::Default;
        config.write_execution.zarr_target.chunk_key_separator =
            crate::core::types::ChunkKeySeparator::Slash;
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = total_coords.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["control".to_string(), "pert01".to_string()]),
        );
        let _ = total_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = total_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords,
                array_names: vec!["t2m".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed for v3 Utf8 coord materialization");
        // Registration metadata is written on a background thread; close() joins it.
        backend
            .close(300.0)
            .expect("close should succeed before file assertions");

        let member_zarr_json =
            std::fs::read_to_string(dataset_root.join("member").join("zarr.json"))
                .expect("member/zarr.json should be present");
        assert!(
            member_zarr_json.contains(
                "\"data_type\":{\"name\":\"fixed_length_utf32\",\"configuration\":{\"length_bytes\":28}}"
            ),
            "expected fixed_length_utf32 data_type in member/zarr.json, got: {member_zarr_json}"
        );
        assert!(
            member_zarr_json.contains("\"shape\":[2]"),
            "expected shape [2] for member coord, got: {member_zarr_json}"
        );
        assert!(
            member_zarr_json.contains("\"chunk_shape\":[2]"),
            "expected chunk_shape [2] for member coord, got: {member_zarr_json}"
        );
        assert!(
            member_zarr_json.contains("\"fill_value\":\"\""),
            "expected empty-string fill_value for member coord, got: {member_zarr_json}"
        );

        let member_chunk = std::fs::read(dataset_root.join("member").join("c").join("0"))
            .expect("member coordinate chunk should be readable");
        assert_eq!(
            member_chunk.len(),
            2 * 7 * 4,
            "expected UTF-32LE fixed-width payload bytes for 2 values × 7 codepoints"
        );

        let decode_utf32le_fixed = |bytes: &[u8], width: usize| -> Vec<String> {
            bytes
                .chunks_exact(width * 4)
                .map(|item| {
                    let mut out = String::new();
                    for cp_bytes in item.chunks_exact(4) {
                        let mut raw = [0_u8; 4];
                        raw.copy_from_slice(cp_bytes);
                        let cp = u32::from_le_bytes(raw);
                        if cp == 0 {
                            break;
                        }
                        let ch = char::from_u32(cp)
                            .expect("UTF-32LE coord payload should contain valid codepoints");
                        out.push(ch);
                    }
                    out
                })
                .collect()
        };
        assert_eq!(
            decode_utf32le_fixed(&member_chunk, 7),
            vec!["control".to_string(), "pert01".to_string()]
        );
    }

    /// Verifies that the Rust backend produces chunk files with correct byte-level
    /// data content, matching what Python zarr would produce for the same workflow.
    ///
    /// This test simulates the Earth2Studio inference pattern:
    ///   add_array(total_coords) → write(step0) → write(step1) → close()
    ///
    /// After close(), we verify:
    /// 1. Chunk files exist at the correct v2 tuple-key paths
    /// 2. Chunk bytes are bitwise-identical to the input payloads
    /// 3. Metadata shape/chunks/dtype match what Python zarr would emit
    /// 4. The dataset_root path is written to a marker file for Python parity checks
    #[test]
    fn chunk_data_bytes_match_expected_payloads_for_single_parallel_combo_writes() {
        let dataset_root = unique_dataset_root("e2s_chunk_data_parity");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        // Register arrays: 2 start times, 3 lead_time steps, 2x2 spatial grid
        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = total_coords.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = total_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = total_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords.clone(),
                array_names: vec!["t2m".to_string(), "tcwv".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        // Simulate Earth2Studio workflow: iterate over time × lead_time steps.
        // Each write() call has 1 value per parallel dim and full spatial axes.
        let write_steps: Vec<(i64, i64, Vec<f32>, Vec<f32>)> = vec![
            (0, 0, vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]),
            (0, 6, vec![5.0, 6.0, 7.0, 8.0], vec![50.0, 60.0, 70.0, 80.0]),
            (
                0,
                12,
                vec![9.0, 10.0, 11.0, 12.0],
                vec![90.0, 100.0, 110.0, 120.0],
            ),
            (
                1,
                0,
                vec![13.0, 14.0, 15.0, 16.0],
                vec![130.0, 140.0, 150.0, 160.0],
            ),
            (
                1,
                6,
                vec![17.0, 18.0, 19.0, 20.0],
                vec![170.0, 180.0, 190.0, 200.0],
            ),
            (
                1,
                12,
                vec![21.0, 22.0, 23.0, 24.0],
                vec![210.0, 220.0, 230.0, 240.0],
            ),
        ];

        let mut expected_chunks: Vec<(String, Vec<u8>)> = Vec::new();

        for (time_val, lt_val, t2m_data, tcwv_data) in &write_steps {
            let mut step_coords = CoordMap::new();
            let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![*time_val]));
            let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![*lt_val]));
            let _ = step_coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
            let _ = step_coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));

            let t2m_bytes: Vec<u8> = t2m_data.iter().flat_map(|v| f32::to_le_bytes(*v)).collect();
            let tcwv_bytes: Vec<u8> = tcwv_data
                .iter()
                .flat_map(|v| f32::to_le_bytes(*v))
                .collect();

            // Compute expected tuple key:
            // Dim order: time(2), lead_time(3), lat(2), lon(2)
            // Parallel dims: time, lead_time; chunk_size=1 for each
            // Non-parallel dims: lat, lon; chunk_size=full
            // Chunk grid: [2, 3, 1, 1]
            // time_idx: position of time_val in [0, 1]
            // lt_idx: position of lt_val in [0, 6, 12]
            let time_idx = if *time_val == 0 { 0 } else { 1 };
            let lt_idx = match *lt_val {
                0 => 0,
                6 => 1,
                12 => 2,
                _ => panic!("unexpected lead_time value"),
            };
            let tuple_key = format!("{time_idx}.{lt_idx}.0.0");

            expected_chunks.push((format!("t2m/{tuple_key}"), t2m_bytes.clone()));
            expected_chunks.push((format!("tcwv/{tuple_key}"), tcwv_bytes.clone()));

            backend
                .write(InferenceWriteRequest {
                    coords: step_coords,
                    array_names: vec!["t2m".to_string(), "tcwv".to_string()],
                    arrays: vec![
                        InputArray {
                            nbytes: t2m_bytes.len(),
                            source: InputArraySource::HostBytes(t2m_bytes.into()),
                        },
                        InputArray {
                            nbytes: tcwv_bytes.len(),
                            source: InputArraySource::HostBytes(tcwv_bytes.into()),
                        },
                    ],
                })
                .expect("write should succeed");
        }

        backend.close(300.0).expect("close should succeed");

        // ── Verify chunk file bytes ──────────────────────────────────────────
        for (rel_path, expected_bytes) in &expected_chunks {
            let chunk_path = dataset_root.join(rel_path);
            assert!(
                chunk_path.exists(),
                "chunk file missing: {}",
                chunk_path.display()
            );
            let actual_bytes = std::fs::read(&chunk_path).unwrap_or_else(|e| {
                panic!("failed reading chunk file '{}': {e}", chunk_path.display())
            });
            assert_eq!(
                actual_bytes,
                *expected_bytes,
                "chunk data mismatch at {rel_path}: expected {} bytes, got {} bytes",
                expected_bytes.len(),
                actual_bytes.len()
            );
        }

        // ── Verify metadata correctness ─────────────────────────────────────
        let consolidated = std::fs::read_to_string(dataset_root.join(".zmetadata"))
            .expect("v2 consolidated metadata should be present");

        // Shape: [time=2, lead_time=3, lat=2, lon=2]
        assert!(
            consolidated.contains("\"shape\":[2,3,2,2]"),
            "expected shape [2,3,2,2], got: {consolidated}"
        );
        // Chunks: [1, 1, 2, 2] — parallel dims chunked by 1
        assert!(
            consolidated.contains("\"chunks\":[1,1,2,2]"),
            "expected chunks [1,1,2,2], got: {consolidated}"
        );
        // Dtype: inferred from chunk byte size and registered chunk shape.
        // Each chunk is 1×1×2×2 = 4 elements × 4 bytes = 16 bytes → float32.
        assert!(
            consolidated.contains("\"dtype\":\"<f4\""),
            "expected dtype <f4 (float32), got: {consolidated}"
        );
        // Both arrays should have metadata
        assert!(
            consolidated.contains("\"t2m/.zarray\""),
            "expected t2m/.zarray in consolidated metadata"
        );
        assert!(
            consolidated.contains("\"tcwv/.zarray\""),
            "expected tcwv/.zarray in consolidated metadata"
        );
        // Dimension names in attrs
        assert!(
            consolidated.contains("\"_ARRAY_DIMENSIONS\""),
            "expected _ARRAY_DIMENSIONS attr in consolidated metadata"
        );

        // ── Verify total chunk count ────────────────────────────────────────
        // 2 time × 3 lead_time = 6 chunks per array, 2 arrays = 12 chunks
        assert_eq!(
            expected_chunks.len(),
            12,
            "expected 12 total chunks (6 per array × 2 arrays)"
        );

        // ── Write marker file for Python parity check ───────────────────────
        let marker_path = dataset_root.with_extension("zarr.marker");
        std::fs::write(&marker_path, dataset_root.to_string_lossy().as_bytes())
            .expect("marker file write should succeed");

        eprintln!(
            "✅ Rust chunk data parity test passed. Dataset at: {}",
            dataset_root.display()
        );
    }

    /// Verifies that two arrays registered in add_array() produce independent
    /// chunk namespaces with correct data isolation.
    #[test]
    fn multi_array_writes_produce_isolated_chunk_namespaces() {
        let dataset_root = unique_dataset_root("e2s_multi_array_isolation");
        let config = SyncZarrBackendConfig {
            dataset_root: Some(dataset_root.clone()),
            ..SyncZarrBackendConfig::default()
        };
        let backend = try_build_sync_zarr_backend(config).expect("backend build should succeed");

        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![100]));
        let _ = coords.insert("lat".to_string(), CoordValues::F64(vec![1.0, 2.0]));

        backend
            .add_array(ArrayRegistration {
                coords: coords.clone(),
                array_names: vec!["arr_a".to_string(), "arr_b".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        let payload_a: Vec<u8> = [42.0_f32, 43.0]
            .iter()
            .flat_map(|v| f32::to_le_bytes(*v))
            .collect();
        let payload_b: Vec<u8> = [99.0_f32, 100.0]
            .iter()
            .flat_map(|v| f32::to_le_bytes(*v))
            .collect();

        backend
            .write(InferenceWriteRequest {
                coords: {
                    let mut c = CoordMap::new();
                    let _ = c.insert("time".to_string(), CoordValues::I64(vec![100]));
                    let _ = c.insert("lat".to_string(), CoordValues::F64(vec![1.0, 2.0]));
                    c
                },
                array_names: vec!["arr_a".to_string(), "arr_b".to_string()],
                arrays: vec![
                    InputArray {
                        nbytes: payload_a.len(),
                        source: InputArraySource::HostBytes(payload_a.clone().into()),
                    },
                    InputArray {
                        nbytes: payload_b.len(),
                        source: InputArraySource::HostBytes(payload_b.clone().into()),
                    },
                ],
            })
            .expect("multi-array write should succeed");
        backend.close(300.0).expect("close should succeed");

        // Chunk key: time_idx=0, lat is non-parallel → "0.0"
        let chunk_a = std::fs::read(dataset_root.join("arr_a").join("0.0"))
            .expect("arr_a chunk should exist");
        let chunk_b = std::fs::read(dataset_root.join("arr_b").join("0.0"))
            .expect("arr_b chunk should exist");

        assert_eq!(chunk_a, payload_a, "arr_a chunk data mismatch");
        assert_eq!(chunk_b, payload_b, "arr_b chunk data mismatch");
        assert_ne!(
            chunk_a, chunk_b,
            "arr_a and arr_b should have different data"
        );
    }
}
