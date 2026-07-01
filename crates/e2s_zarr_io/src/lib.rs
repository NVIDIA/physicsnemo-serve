/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # e2s_zarr_io
//!
//! High-performance Zarr write backend for Earth2Studio inference pipelines.
//!
//! This crate provides a Rust-native implementation of the `add_array → write → close`
//! lifecycle for writing Zarr v2 and v3 datasets from Python (via pyo3) or pure Rust.
//!
//! ## Key design properties
//!
//! - **Copy-barrier semantics**: `write()` returns only after all input arrays are fully
//!   copied into Rust-owned buffers; filesystem writes continue asynchronously.
//! - **Bounded memory**: pooled buffers from contiguous hot/warm slabs with configurable
//!   alignment, pinning, and optional CUDA registration.
//! - **No-overwrite safety**: each `ChunkId` can be written at most once per backend lifetime.
//! - **Dual-format support**: Zarr v2 and v3 through a shared write pipeline with
//!   format-specific metadata and chunk-path adapters.
//!
//! ## Module layout
//!
//! - [`core`] — Domain types, trait contracts, and error definitions.
//! - [`runtime`] — Execution-time components (coordinator, planner, buffer pool, workers).
//! - [`zarr`] — Format-specific layout adapters, chunk writers, and metadata consolidation.
//! - [`api`] — Public factory functions for constructing backends.
//! - [`backend`] — Backend lifecycle implementation.
#![warn(missing_docs)]

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

pub mod api;
pub mod backend;
pub mod core;
#[cfg(feature = "python-bindings")]
mod python_bindings;
pub mod runtime;
pub mod zarr;

// ── Explicit public re-exports ──────────────────────────────────────────────
// These form the crate's public API surface. Items not listed here are
// accessible through their full module path but are not part of the
// "quick import" ergonomic surface.

// Core domain types
pub use core::chunk_id::ChunkId;
pub use core::errors::{DeferredWriteError, SyncWriteError};
pub use core::types::{
    ArrayRegistration, BufferPoolConfig, CloseReport, CoordMap, CoordValues, InferenceWriteRequest,
    InputArray, InputArraySource, SizeOverride, WriteCopyAck, WriteExecutionConfig,
    WriteInternalTiming, ZarrFormat, ZarrTargetConfig,
};

// Trait contracts
pub use core::contracts::{
    ArrayRegistry, BufferPool, ChunkKeyRegistry, ChunkPlanner, ChunkWriter, CopyEngine,
    MetadataConsolidator, WorkScheduler, ZarrIoBackend, ZarrLayoutAdapter,
};

// Public backend API
#[allow(deprecated)]
pub use api::build_sync_zarr_backend;
pub use api::{SyncZarrBackendConfig, new_sync_zarr_backend, try_build_sync_zarr_backend};
pub use backend::SyncZarrBackend;
// Default component implementations
pub use runtime::array_registry::InMemoryArrayRegistry;
pub use runtime::buffer_pool::MemoryBufferPool;
pub use runtime::planner::MixedRadixChunkPlanner;
pub use runtime::registry::InMemoryChunkKeyRegistry;
pub use runtime::thread_pool::{RayonWorkScheduler, SynchronousWorkScheduler};
pub use zarr::metadata::LocalFsMetadataConsolidator;
pub use zarr::writer::LocalFsChunkWriter;
pub use zarr::zarr_layout::DefaultZarrLayoutAdapter;

/// Returns the crate version string (matches `Cargo.toml` version).
#[must_use]
pub fn version() -> &'static str {
    api::version()
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn py_version() -> &'static str {
    version()
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn e2s_zarr_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_version, m)?)?;
    python_bindings::register_python_bindings_module(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::core::types::{
        BufferLease, BufferPoolConfig, BufferSizingPolicy, ChunkKeyEncoding, ChunkKeySeparator,
        DEFAULT_NUM_THREADS, DEFAULT_TRANSIENT_LIMIT_BYTES, FirstWriteSizingConfig,
        InputStabilityPolicy, ModelProfileHint, SchedulerBackend, WarmSlabState, ZarrTargetConfig,
    };
    use crate::runtime::coordinator::{WriteCoordinator, WriteCoordinatorComponents};
    use crate::runtime::copy_engine::DefaultCopyEngine;
    use crate::runtime::registry::InMemoryChunkKeyRegistry;
    use crate::zarr::metadata::NoopMetadataConsolidator;
    use crate::zarr::writer::NoopChunkWriter;

    #[derive(Default)]
    struct CountingMetadataConsolidator {
        calls: AtomicUsize,
    }

    impl CountingMetadataConsolidator {
        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    impl MetadataConsolidator for CountingMetadataConsolidator {
        fn consolidate(
            &self,
            _layout: &dyn ZarrLayoutAdapter,
            _registration: Option<&ArrayRegistration>,
            _parallel_coord_names: &[String],
        ) -> Result<(), SyncWriteError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
    }

    fn backend_with_counting_metadata_consolidator()
    -> (SyncZarrBackend, Arc<CountingMetadataConsolidator>) {
        let write_execution = WriteExecutionConfig::default();
        let queue_capacity = write_execution.queue_capacity;
        let parallel_coord_names = write_execution
            .parallel_coords_config
            .default_parallel_coord_names
            .clone();

        let array_registry: Arc<dyn ArrayRegistry> = Arc::new(InMemoryArrayRegistry::new());
        let planner: Arc<dyn ChunkPlanner> = Arc::new(MixedRadixChunkPlanner::new(write_execution));
        let chunk_registry: Arc<dyn ChunkKeyRegistry> = Arc::new(InMemoryChunkKeyRegistry::new());
        let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
        let buffer_pool: Arc<dyn BufferPool> =
            Arc::new(MemoryBufferPool::new(BufferPoolConfig::default()));
        let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
        let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
            Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(NoopChunkWriter::new(
            std::env::temp_dir().join("e2s_drop_cleanup_noop"),
            Arc::clone(&layout_adapter),
        ));
        let metadata_probe = Arc::new(CountingMetadataConsolidator::default());
        let metadata_consolidator: Arc<dyn MetadataConsolidator> = metadata_probe.clone();

        let coordinator = Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer,
            metadata_consolidator,
            layout_adapter,
            parallel_coord_names,
            queue_capacity,
        }));
        let backend = SyncZarrBackend::new_with_close_timeout(coordinator, array_registry, 300.0);
        (backend, metadata_probe)
    }

    #[test]
    fn version_matches_package() {
        assert_eq!(api::version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn defaults_match_spec_decisions() {
        let pool = BufferPoolConfig::default();
        assert_eq!(pool.max_pool_buffers, DEFAULT_NUM_THREADS);
        assert_eq!(
            pool.max_transient_buffer_bytes,
            Some(DEFAULT_TRANSIENT_LIMIT_BYTES)
        );
        assert_eq!(
            pool.max_inflight_transient_bytes,
            Some(DEFAULT_TRANSIENT_LIMIT_BYTES)
        );
        assert!(pool.pin_pooled_slabs);
        assert!(pool.cuda_register_each_slab_once);
        assert_eq!(
            pool.first_write_sizing.buffer_sizing_policy,
            BufferSizingPolicy::FirstWriteModelAwareAuto
        );

        let exec = WriteExecutionConfig::default();
        assert_eq!(exec.num_threads, DEFAULT_NUM_THREADS);
        assert_eq!(
            exec.input_stability_policy,
            InputStabilityPolicy::StrictGilHold
        );
        assert_eq!(exec.scheduler_backend, SchedulerBackend::RayonWorkStealing);
        assert_eq!(exec.zarr_target.zarr_format, ZarrFormat::V2);
        assert_eq!(exec.zarr_target.chunk_key_encoding, ChunkKeyEncoding::V2);
        assert_eq!(exec.zarr_target.chunk_key_separator, ChunkKeySeparator::Dot);
    }

    #[test]
    fn chunk_id_orders_deterministically() {
        let a = ChunkId::new(1, 3);
        let b = ChunkId::new(1, 5);
        let c = ChunkId::new(2, 0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn scaffold_backend_wires_traits_end_to_end() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");

        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed in scaffold");

        let ack = backend
            .write(InferenceWriteRequest {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: 4,
                    source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
                }],
            })
            .expect("write should succeed in scaffold");
        assert_eq!(ack.copied_tasks, 1);
        let last_timing = backend
            .last_write_timing()
            .expect("write timing should be available after successful write");
        assert_eq!(last_timing.batch_id, ack.batch_id);
        assert_eq!(last_timing.task_count, ack.copied_tasks);

        let close = backend
            .close(300.0)
            .expect("close should succeed in scaffold");
        assert!(close.closed);
        assert!(close.metadata_consolidated);
        assert!(close.resources_released);
        assert!(backend.is_closed());
    }

    #[test]
    fn write_before_add_array_is_rejected() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        let result = backend.write(InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            arrays: vec![InputArray {
                nbytes: 4,
                source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
            }],
        });
        assert!(matches!(
            result,
            Err(SyncWriteError::ContractViolation { message }) if message.contains("before add_array")
        ));
    }

    #[test]
    fn io_is_rejected_after_close() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        backend.close(300.0).expect("close should succeed");

        let write_result = backend.write(InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            arrays: vec![InputArray {
                nbytes: 4,
                source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
            }],
        });
        assert!(matches!(write_result, Err(SyncWriteError::ObjectClosed)));

        let add_result = backend.add_array(ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        });
        assert!(matches!(add_result, Err(SyncWriteError::ObjectClosed)));
    }

    #[test]
    fn add_array_is_single_registration_only() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("first add_array should succeed");
        let second = backend.add_array(ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["pressure".to_string()],
            array_dtypes: Vec::new(),
        });
        assert!(matches!(
            second,
            Err(SyncWriteError::ContractViolation { message }) if message.contains("only be called once")
        ));
    }

    #[test]
    fn write_rejects_unknown_array_names() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        let result = backend.write(InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: vec!["humidity".to_string()],
            arrays: vec![InputArray {
                nbytes: 4,
                source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
            }],
        });
        assert!(matches!(
            result,
            Err(SyncWriteError::ContractViolation { message }) if message.contains("unknown array name")
        ));
    }

    #[test]
    fn registry_rejects_duplicate_chunk_id() {
        let registry = InMemoryChunkKeyRegistry::new();
        let id = ChunkId::new(7, 42);
        registry
            .reserve_many_ids(&[id])
            .expect("initial reserve should succeed");
        let second = registry.reserve_many_ids(&[id]);
        assert!(matches!(
            second,
            Err(SyncWriteError::ChunkKeyConflict { chunk_id }) if chunk_id == id
        ));
    }

    #[test]
    fn factory_rejects_invalid_zarr_target_config() {
        let mut config = api::SyncZarrBackendConfig::default();
        config.write_execution.zarr_target = ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::V2,
            chunk_key_separator: ChunkKeySeparator::Slash,
        };
        let result = api::try_build_sync_zarr_backend(config);
        assert!(matches!(
            result,
            Err(SyncWriteError::UnsupportedZarrTargetConfig { .. })
        ));
    }

    #[test]
    #[allow(deprecated)]
    fn factory_rejects_non_strict_input_stability_policy() {
        let mut config = api::SyncZarrBackendConfig::default();
        config.write_execution.input_stability_policy = InputStabilityPolicy::ContractOnly;
        let result = api::try_build_sync_zarr_backend(config);
        assert!(matches!(
            result,
            Err(SyncWriteError::UnsupportedInputStabilityPolicy { .. })
        ));
    }

    #[test]
    fn add_array_rejects_duplicate_names() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        let result = backend.add_array(ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string(), "temperature".to_string()],
            array_dtypes: Vec::new(),
        });
        assert!(matches!(
            result,
            Err(SyncWriteError::Validation { message })
            if message.contains("must be unique")
        ));
    }

    #[test]
    fn close_rejects_non_positive_timeout() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        let result = backend.close(0.0);
        assert!(matches!(
            result,
            Err(SyncWriteError::Validation { message })
            if message.contains("timeout_seconds")
        ));
    }

    #[test]
    fn close_rejects_non_finite_timeout_without_transitioning_to_closed() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        for timeout in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = backend.close(timeout);
            assert!(matches!(
                result,
                Err(SyncWriteError::Validation { message })
                if message.contains("timeout_seconds")
            ));
            assert!(
                !backend.is_closed(),
                "invalid close timeout must not transition backend to closed"
            );
        }

        backend
            .close(300.0)
            .expect("valid close should still succeed after invalid timeout attempts");
        assert!(backend.is_closed());
    }

    #[test]
    fn second_close_returns_object_closed() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        backend.close(300.0).expect("first close should succeed");
        let second = backend.close(300.0);
        assert!(
            matches!(second, Err(SyncWriteError::ObjectClosed)),
            "second close should report object closed"
        );
    }

    #[test]
    fn drop_without_explicit_close_triggers_best_effort_cleanup() {
        let (backend, metadata_probe) = backend_with_counting_metadata_consolidator();
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        // Background thread may not have finished yet — do not assert calls() here.

        drop(backend);

        assert_eq!(
            metadata_probe.calls(),
            2,
            "drop should join background registration thread (1) and run close consolidation (1)"
        );
    }

    #[test]
    fn explicit_close_then_drop_does_not_double_close_cleanup() {
        let (backend, metadata_probe) = backend_with_counting_metadata_consolidator();
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");
        backend.close(300.0).expect("explicit close should succeed");
        let calls_after_explicit_close = metadata_probe.calls();

        drop(backend);

        assert_eq!(
            metadata_probe.calls(),
            calls_after_explicit_close,
            "drop after explicit close must not run cleanup a second time"
        );
    }

    #[test]
    fn factory_rejects_invalid_buffer_pool_config() {
        let mut config = api::SyncZarrBackendConfig::default();
        config.buffer_pool.max_pool_buffers = 0;
        let result = api::try_build_sync_zarr_backend(config);
        assert!(matches!(
            result,
            Err(SyncWriteError::Validation { message })
            if message.contains("max_pool_buffers")
        ));
    }

    #[test]
    fn add_array_write_derives_pool_sizing_from_planner_chunk_pressure() {
        let pool_config = BufferPoolConfig {
            max_pool_buffers: 6,
            max_pool_bytes: 64 * 1024 * 1024,
            first_write_sizing: FirstWriteSizingConfig {
                min_hot_slab_buffers: 1,
                max_warm_to_hot_ratio: 3,
                model_profile_hint: Some(ModelProfileHint::GraphCastSmall),
                ..Default::default()
            },
            ..Default::default()
        };

        let write_execution = WriteExecutionConfig::default();
        let queue_capacity = write_execution.queue_capacity;
        let parallel_coord_names = write_execution
            .parallel_coords_config
            .default_parallel_coord_names
            .clone();

        let array_registry: Arc<dyn ArrayRegistry> = Arc::new(InMemoryArrayRegistry::new());
        let planner: Arc<dyn ChunkPlanner> = Arc::new(MixedRadixChunkPlanner::new(write_execution));
        let chunk_registry: Arc<dyn ChunkKeyRegistry> = Arc::new(InMemoryChunkKeyRegistry::new());
        let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
        let memory_pool = Arc::new(MemoryBufferPool::new(pool_config));
        let buffer_pool: Arc<dyn BufferPool> = memory_pool.clone();
        let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
        let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
            Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(NoopChunkWriter::new(
            std::env::temp_dir().join("e2s_noop"),
            Arc::clone(&layout_adapter),
        ));
        let metadata_consolidator: Arc<dyn MetadataConsolidator> =
            Arc::new(NoopMetadataConsolidator::new());

        let coordinator = Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer,
            metadata_consolidator,
            layout_adapter,
            parallel_coord_names,
            queue_capacity,
        }));
        let backend = SyncZarrBackend::new_with_close_timeout(
            Arc::clone(&coordinator),
            Arc::clone(&array_registry),
            300.0,
        );

        let mut total_coords = CoordMap::new();
        let _ = total_coords.insert("time".to_string(), CoordValues::I64(vec![0, 1, 2]));
        let _ = total_coords.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = total_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1]));
        backend
            .add_array(ArrayRegistration {
                coords: total_coords,
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        // Chunk planner expansion for this step: time(2) x lead_time(1) x ensemble(2) = 4 tasks.
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));

        let payload_len = 1_000_000;
        let ack = backend
            .write(InferenceWriteRequest {
                coords: step_coords,
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: payload_len,
                    source: InputArraySource::HostBytes(vec![7_u8; payload_len].into()),
                }],
            })
            .expect("write should succeed");
        assert_eq!(
            ack.copied_tasks, 4,
            "expected planner-derived cartesian chunk expansion task count"
        );

        // pooled_buffer_bytes should align_up(max(required_bytes_per_task, model_baseline), 4KiB)
        // where required_bytes_per_task = payload_len / task_count = 250_000.
        // -> align_up(max(250_000, 260_640), 4096) = 262_144.
        let required_bytes_per_task = payload_len / ack.copied_tasks;
        let pooled_deadline = Instant::now() + Duration::from_millis(300);
        let sized = loop {
            let lease = memory_pool
                .acquire(required_bytes_per_task)
                .expect("acquire should succeed after first write sizing");
            if matches!(lease, BufferLease::Pooled(_)) {
                break lease;
            }
            memory_pool.release(lease);
            if Instant::now() >= pooled_deadline {
                panic!("expected pooled lease after first-write sizing");
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        match &sized {
            BufferLease::Pooled(handle) => {
                assert_eq!(handle.capacity_bytes(), 262_144);
            }
            BufferLease::Transient(_) => unreachable!("loop exits only on pooled lease"),
        }
        memory_pool.release(sized);

        // Oversized requests above frozen pooled_buffer_bytes must use transient leases.
        let oversized = memory_pool
            .acquire(262_145)
            .expect("oversized acquire should succeed with transient fallback");
        assert!(matches!(oversized, BufferLease::Transient(_)));
        memory_pool.release(oversized);

        // Expected pool split for first_write_task_count=4:
        // hot=4, warm=4, then capped by max_pool_buffers=6 => warm reduced to 2.
        // Total pooled handles available = 6.
        let warm_ready_deadline = Instant::now() + Duration::from_millis(300);
        loop {
            let status = memory_pool.warmup_status();
            if status.warm_state == WarmSlabState::Ready {
                break;
            }
            if Instant::now() >= warm_ready_deadline {
                panic!(
                    "warm slab did not become ready before timeout; state={:?}",
                    status.warm_state
                );
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        memory_pool
            .wait_pooled_leases_returned(1.0)
            .expect("write-stage pooled leases should be returned before capacity probe");

        let mut outstanding = Vec::new();
        let mut pooled_count = 0usize;
        for _ in 0..8 {
            let lease = memory_pool
                .acquire(1)
                .expect("small acquire should succeed while probing pool capacity");
            if matches!(lease, BufferLease::Pooled(_)) {
                pooled_count += 1;
            }
            let stop = matches!(lease, BufferLease::Transient(_));
            outstanding.push(lease);
            if stop {
                break;
            }
        }
        for lease in outstanding {
            memory_pool.release(lease);
        }
        assert_eq!(
            pooled_count, 6,
            "pooled handle count should match derived hot+warm cap"
        );

        backend.close(300.0).expect("close should succeed");
    }

    #[test]
    fn write_rejects_host_bytes_with_mismatched_declared_nbytes() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        let err = backend
            .write(InferenceWriteRequest {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: 4096,
                    source: InputArraySource::HostBytes(vec![1_u8, 2, 3, 4].into()),
                }],
            })
            .expect_err("HostBytes payload length mismatch should be rejected");
        assert!(
            matches!(
                err,
                SyncWriteError::Validation { ref message }
                if message.contains("nbytes") && message.contains("HostBytes")
            ),
            "mismatched HostBytes size should be rejected as a validation error, got: {err}"
        );
    }

    #[test]
    fn write_rejects_host_bytes_when_declared_nbytes_is_smaller_than_payload() {
        let backend = api::try_build_sync_zarr_backend(api::SyncZarrBackendConfig::default())
            .expect("backend construction should succeed");
        backend
            .add_array(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("add_array should succeed");

        let err = backend
            .write(InferenceWriteRequest {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                arrays: vec![InputArray {
                    nbytes: 2,
                    source: InputArraySource::HostBytes(vec![1_u8, 2, 3, 4].into()),
                }],
            })
            .expect_err("HostBytes payload length mismatch should be rejected");
        assert!(
            matches!(
                err,
                SyncWriteError::Validation { ref message }
                if message.contains("HostBytes") && message.contains("nbytes")
            ),
            "declared-smaller HostBytes mismatch should be rejected as validation error, got: {err}"
        );
    }

    #[test]
    fn core_types_are_split_into_focused_submodules() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let type_module_files = [
            "src/core/types.rs",
            "src/core/types/config.rs",
            "src/core/types/pool_config.rs",
            "src/core/types/requests.rs",
            "src/core/types/responses.rs",
            "src/core/types/buffer.rs",
            "src/core/types/planner.rs",
        ];

        for rel in type_module_files {
            let file = root.join(rel);
            assert!(
                file.exists(),
                "expected split core types module file to exist: {}",
                file.display()
            );
            let line_count = std::fs::read_to_string(&file)
                .expect("split types module file should be readable")
                .lines()
                .count();
            assert!(
                line_count <= 500,
                "split types module file should stay focused (<= 500 lines): {} has {} lines",
                file.display(),
                line_count
            );
        }
    }

    #[test]
    fn write_copy_barrier_bench_avoids_private_coordinator_type_imports() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_source =
            std::fs::read_to_string(root.join("benches/write_copy_barrier_latency.rs"))
                .expect("write_copy_barrier_latency benchmark should be readable");
        assert!(
            !bench_source
                .contains("runtime::coordinator::{WriteCoordinator, WriteCoordinatorComponents}"),
            "benchmark should not import private coordinator types directly"
        );
        assert!(
            bench_source.contains("TestWriteCoordinator")
                && bench_source.contains("TestWriteCoordinatorComponents"),
            "benchmark should use test-utils coordinator wrapper exports"
        );
    }

    #[test]
    fn write_copy_barrier_bench_uses_short_best_effort_close_timeout() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_source =
            std::fs::read_to_string(root.join("benches/write_copy_barrier_latency.rs"))
                .expect("write_copy_barrier_latency benchmark should be readable");
        assert!(
            bench_source.contains("const BENCH_CLOSE_TIMEOUT_SECONDS: f64 = 0.05;"),
            "benchmark teardown close timeout should stay short to avoid long wall-clock stalls"
        );
        assert!(
            bench_source.contains("let _ = coordinator.close(BENCH_CLOSE_TIMEOUT_SECONDS, None);"),
            "benchmark teardown should use best-effort close without panicking on timeout"
        );
    }

    #[test]
    fn write_copy_barrier_localfs_bench_keeps_transient_limit_guardrails_enabled() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_source =
            std::fs::read_to_string(root.join("benches/write_copy_barrier_latency.rs"))
                .expect("write_copy_barrier_latency benchmark should be readable");
        assert!(
            !bench_source.contains("max_inflight_transient_bytes = None"),
            "localfs realistic bench must not disable transient in-flight guardrails (prevents OOM)"
        );
    }

    #[test]
    fn write_copy_barrier_realistic_steady_state_bench_uses_isolated_iteration_helper() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_source =
            std::fs::read_to_string(root.join("benches/write_copy_barrier_latency.rs"))
                .expect("write_copy_barrier_latency benchmark should be readable");
        assert!(
            bench_source.contains("measure_realistic_steady_state_submit_write_once"),
            "realistic steady-state bench should use isolated per-iteration helper to bound async memory growth"
        );
    }

    #[test]
    fn write_copy_barrier_bench_uses_shared_rayon_pools_for_first_write() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bench_source =
            std::fs::read_to_string(root.join("benches/write_copy_barrier_latency.rs"))
                .expect("write_copy_barrier_latency benchmark should be readable");
        assert!(
            bench_source.contains("shared_bench_rayon_pools()"),
            "first-write bench must use shared Rayon pools to avoid per-iteration thread pool \
             creation overhead that inflates small-payload latency measurements"
        );
    }

    #[test]
    fn test_write_coordinator_exposes_shared_pool_constructor() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let coordinator_source = std::fs::read_to_string(root.join("src/runtime/coordinator.rs"))
            .expect("coordinator.rs should be readable");
        assert!(
            coordinator_source.contains("fn new_with_shared_pools("),
            "TestWriteCoordinator must expose new_with_shared_pools() so benchmarks can \
             inject pre-warmed thread pools instead of creating fresh pools per iteration"
        );
    }
}
