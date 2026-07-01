/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Trait boundaries between internal components.
//!
//! Every trait in this module represents a replaceable component boundary in the
//! write pipeline. Production and test code provide different implementations.
//!
//! # Safety contract
//!
//! All traits require `Send + Sync` because components are shared across worker
//! threads behind `Arc`.

use std::collections::BTreeMap;

use crate::core::chunk_id::ChunkId;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ArrayRegistration, BatchId, BufferLease, ChunkPathSpec, ChunkTask, CloseReport, CoordMap,
    CopyCompletion, FirstWriteSizingHint, InferenceWriteRequest, InputArray, MetadataPathSpec,
    PlannedWriteBatch, PoolWarmupStatus, TupleChunkKey, WriteCopyAck, ZarrFormat,
};

/// Controls which consolidation phases to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationScope {
    /// Root metadata + coordinate arrays only (registration-time).
    RegistrationOnly,
    /// Per-array metadata + consolidated .zmetadata only (close-time).
    CloseOnly,
    /// Everything (backward-compatible fallback).
    Full,
}

/// Top-level backend contract for the `add_array → write → close` lifecycle.
///
/// Implementations must enforce:
/// - `add_array()` is called exactly once before any `write()`.
/// - `write()` returns only after all input arrays are copied into Rust-owned buffers.
/// - `close()` blocks until all accepted writes complete (or timeout), then consolidates
///   metadata and releases resources.
/// - After `close()`, all subsequent operations return [`SyncWriteError::ObjectClosed`].
pub trait ZarrIoBackend: Send + Sync {
    /// Register array names and the full coordinate contract.
    ///
    /// Must be called exactly once before the first `write()`.
    fn add_array(&self, req: ArrayRegistration) -> Result<(), SyncWriteError>;

    /// Write one or more input arrays for the current inference step.
    ///
    /// Returns only after all arrays are fully copied into Rust-owned buffers
    /// (copy-barrier semantics). Filesystem writes may continue asynchronously.
    fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError>;

    /// Finalize all pending writes, consolidate metadata, and release resources.
    ///
    /// `timeout_seconds` bounds the wait for pooled lease return (spec default: 300.0).
    fn close(&self, timeout_seconds: f64) -> Result<CloseReport, SyncWriteError>;

    /// Returns `true` after `close()` has been called (success or error).
    fn is_closed(&self) -> bool;
}

/// Registration boundary for `add_array()` metadata and write-time name validation.
///
/// This component owns the immutable registration contract for backend lifetime:
/// - `add_array()` can succeed only once.
/// - `write()` name checks must be validated against registered arrays.
pub trait ArrayRegistry: Send + Sync {
    /// Register array names and coordinate contract.
    ///
    /// Must fail if already registered.
    fn register(&self, req: ArrayRegistration) -> Result<(), SyncWriteError>;

    /// Validate that all `array_names` are known in the registered contract.
    fn validate_write_array_names(&self, array_names: &[String]) -> Result<(), SyncWriteError>;

    /// Resolve registered deterministic array IDs for `array_names`.
    ///
    /// IDs are stable for the backend lifetime and derived from registration order.
    fn resolve_array_ids(&self, array_names: &[String]) -> Result<Vec<u32>, SyncWriteError>;

    /// Return a clone of the registered coordinate contract.
    fn registered_coords(&self) -> Result<CoordMap, SyncWriteError>;

    /// Return a clone of the current registration snapshot, if present.
    fn registration_snapshot(&self) -> Result<Option<ArrayRegistration>, SyncWriteError>;
}

/// Shared reusable buffer pool with hot/warm slab lifecycle.
///
/// The pool lazily initializes on first write using planner-derived sizing hints
/// and provides two lease classes:
/// - **Pooled**: fixed-size buffers from contiguous hot/warm slabs.
/// - **Transient**: one-shot oversized buffers that are dropped after write.
pub trait BufferPool: Send + Sync {
    /// Bootstrap pool state from first-write planner statistics (idempotent).
    fn initialize_if_needed(&self, hint: &FirstWriteSizingHint) -> Result<(), SyncWriteError>;

    /// Borrow a buffer lease for the given byte requirement.
    ///
    /// Returns `Pooled` if the request fits a ready pooled buffer, otherwise `Transient`.
    fn acquire(&self, required_bytes: usize) -> Result<BufferLease, SyncWriteError>;

    /// Return a previously acquired lease to the pool (pooled) or drop it (transient).
    fn release(&self, lease: BufferLease);

    /// Query current hot/warm slab readiness for diagnostics.
    fn warmup_status(&self) -> PoolWarmupStatus;

    /// Block until all outstanding leases are returned (with timeout).
    ///
    /// Implementations should account for both pooled and transient leases that
    /// are still in-flight.
    fn wait_pooled_leases_returned(&self, timeout_seconds: f64) -> Result<(), SyncWriteError>;

    /// Release all pooled resources. Called during `close()` teardown.
    ///
    /// After shutdown, `acquire()` and `release()` calls are rejected.
    fn shutdown(&self) -> Result<(), SyncWriteError>;

    /// Whether the pool supports early initialization without a first-write hint.
    ///
    /// Returns `true` when the pool's buffer size is explicitly configured
    /// (not derived from the first write), making it safe to pre-allocate
    /// during `add_array()` without waiting for the first `write()`.
    fn supports_early_init(&self) -> bool {
        false
    }
}

/// Writes buffered bytes to a local filesystem Zarr chunk file.
///
/// The writer receives a `ChunkId` and the raw byte payload. Path rendering
/// and temp-file + atomic-rename safety are handled internally.
pub trait ChunkWriter: Send + Sync {
    /// Write `bytes` to the chunk path derived from a pre-computed `TupleChunkKey`.
    ///
    /// This is the preferred write path: the planner pre-computes the tuple key
    /// during `plan_batch()`, so the writer can emit the final Zarr chunk path
    /// directly without a deferred rename during `close()`.
    fn write_chunk_by_tuple_key(
        &self,
        array_name: &str,
        tuple_key: &TupleChunkKey,
        bytes: &[u8],
    ) -> Result<(), SyncWriteError>;

    /// Write `bytes` as the chunk identified by `chunk_id` (linear index path).
    ///
    /// Legacy compatibility path used by older callers that do not pass tuple
    /// chunk keys. New implementations can rely on the tuple-key method only.
    fn write_chunk_by_id(
        &self,
        _array_name: &str,
        _chunk_id: &ChunkId,
        _bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        Err(SyncWriteError::ContractViolation {
            message: "write_chunk_by_id not supported; use write_chunk_by_tuple_key".to_string(),
        })
    }
}

/// Format-specific layout adapter for Zarr v2/v3 metadata and chunk paths.
///
/// Converts internal identities (`ChunkId`, array metadata) into format-aware
/// filesystem layout (chunk paths, metadata file paths).
pub trait ZarrLayoutAdapter: Send + Sync {
    /// The Zarr format this adapter targets (immutable for backend lifetime).
    fn zarr_format(&self) -> ZarrFormat;

    /// Render the relative chunk path for the given `ChunkId`.
    fn chunk_path_for(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
    ) -> Result<ChunkPathSpec, SyncWriteError>;

    /// Render the relative chunk path from a pre-computed `TupleChunkKey`.
    ///
    /// The adapter applies the format-specific prefix and separator:
    /// - **V2 dot**: `"{array_name}/0.4.0.0"`
    /// - **V2 slash**: `"{array_name}/0/4/0/0"`
    /// - **V3**: `"{array_name}/c/0/4/0/0"`
    fn chunk_path_for_tuple_key(
        &self,
        array_name: &str,
        tuple_key: &TupleChunkKey,
    ) -> Result<ChunkPathSpec, SyncWriteError>;

    /// Return the set of metadata file paths for root and per-array nodes.
    fn metadata_paths(&self) -> Result<MetadataPathSpec, SyncWriteError>;
}

/// Copy engine that transfers input array bytes into a leased buffer.
///
/// Selects copy strategy per task based on `(source kind, lease kind)`:
/// - Host source → CPU memcpy (synchronous).
/// - CUDA source + registered lease → CUDA D2H DMA.
/// - CUDA source + transient lease → fallback copy path.
pub trait CopyEngine: Send + Sync {
    /// Copy `required_bytes` from `src` into `lease`.
    fn copy_into_lease(
        &self,
        src: &InputArray,
        lease: &mut BufferLease,
        required_bytes: usize,
    ) -> Result<CopyCompletion, SyncWriteError>;

    /// Block until an asynchronous copy operation completes.
    fn wait_copy_completion(&self, completion: CopyCompletion) -> Result<(), SyncWriteError>;
}

/// Work scheduler for dispatching chunk tasks to worker threads.
///
/// Uses bounded admission for backpressure. The v1 backend is a Rayon
/// fixed-size pool with work stealing.
pub trait WorkScheduler: Send + Sync {
    /// Enqueue a chunk task for worker execution.
    fn submit(&self, task: ChunkTask) -> Result<(), SyncWriteError>;

    /// Mark one task in `batch_id` as copy-complete.
    ///
    /// This signal must be emitted only after copy completion is observed
    /// (including asynchronous CUDA completion where applicable).
    fn mark_copied(&self, batch_id: BatchId) -> Result<(), SyncWriteError>;

    /// Block until all tasks in `batch_id` reach copy-complete state.
    fn wait_copied(&self, batch_id: BatchId) -> Result<(), SyncWriteError>;

    /// Abort tracking for a batch that cannot reach copy-barrier completion.
    ///
    /// Used by early-error submit paths to prevent stale per-batch accounting
    /// entries from accumulating until `drain()`/`shutdown()`.
    fn abort_batch(&self, batch_id: BatchId) -> Result<(), SyncWriteError>;

    /// Drain all queued and in-flight tasks.
    fn drain(&self) -> Result<(), SyncWriteError>;

    /// Shut down worker threads and release scheduler resources.
    fn shutdown(&self) -> Result<(), SyncWriteError>;
}

/// Thread-safe `ChunkId` reservation and commit registry.
///
/// Enforces the no-overwrite policy: once a `ChunkId` is reserved or committed,
/// subsequent reservation attempts for the same ID are rejected.
pub trait ChunkKeyRegistry: Send + Sync {
    /// Atomically reserve all `chunk_ids` for an in-flight write batch.
    ///
    /// Fails with [`SyncWriteError::ChunkKeyConflict`] if any ID is already
    /// reserved or committed.
    fn reserve_many_ids(&self, chunk_ids: &[ChunkId]) -> Result<(), SyncWriteError>;

    /// Transition a reserved `ChunkId` to committed state after successful write.
    fn mark_committed_id(&self, chunk_id: &ChunkId);

    /// Release a reserved `ChunkId` after a failed write (allows future retry if policy permits).
    fn release_failed_id(&self, chunk_id: &ChunkId);
}

/// Format-aware metadata consolidator invoked once during `close()`.
///
/// Runs on a single thread after all chunk writes are drained.
pub trait MetadataConsolidator: Send + Sync {
    /// Write consolidated metadata for the selected Zarr format.
    fn consolidate(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: Option<&ArrayRegistration>,
        parallel_coord_names: &[String],
    ) -> Result<(), SyncWriteError>;

    /// Scoped consolidation: write only the phases indicated by `scope`.
    ///
    /// - `RegistrationOnly`: root metadata + coordinate arrays (called from `add_array` background thread).
    /// - `CloseOnly`: per-array metadata + `.zmetadata` (called from `close()`).
    /// - `Full`: everything (backward-compatible default).
    ///
    /// `observed_chunk_bytes` provides per-array chunk sizes recorded at write time,
    /// allowing `CloseOnly` to skip filesystem scanning via `infer_chunk_descriptor`.
    ///
    /// # Default implementation
    ///
    /// The default delegates to [`consolidate`](MetadataConsolidator::consolidate),
    /// ignoring `scope` and `observed_chunk_bytes`. New implementors that need
    /// scoped behavior **must** override this method; otherwise calling with
    /// `RegistrationOnly` will silently perform a full consolidation.
    fn consolidate_scoped(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: Option<&ArrayRegistration>,
        parallel_coord_names: &[String],
        scope: ConsolidationScope,
        observed_chunk_bytes: Option<&BTreeMap<String, usize>>,
    ) -> Result<(), SyncWriteError> {
        let _ = (scope, observed_chunk_bytes);
        self.consolidate(layout, registration, parallel_coord_names)
    }
}

/// Chunk planner that translates inference write requests into planned task batches.
///
/// The v1 algorithm is `MixedRadixStreaming` which enumerates chunk tasks without
/// full Cartesian meshgrid materialization.
pub trait ChunkPlanner: Send + Sync {
    /// Plan a write batch: decompose the request into individual chunk tasks.
    fn plan_batch(
        &self,
        req: &InferenceWriteRequest,
        array_ids: &[u32],
        registered_coords: &CoordMap,
    ) -> Result<PlannedWriteBatch, SyncWriteError>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn work_scheduler_trait_excludes_buffer_pool_waiting_concern() {
        let source = include_str!("contracts.rs");
        let work_scheduler_section = source
            .split("pub trait WorkScheduler: Send + Sync")
            .nth(1)
            .expect("WorkScheduler trait declaration should exist");
        let work_scheduler_block = work_scheduler_section
            .split("pub trait ChunkKeyRegistry: Send + Sync")
            .next()
            .expect("WorkScheduler block should end before ChunkKeyRegistry");

        assert!(
            !work_scheduler_block.contains("wait_pooled_leases_returned"),
            "WorkScheduler should remain focused on scheduling and must not expose BufferPool lease waiting"
        );
    }

    #[test]
    fn chunk_writer_trait_provides_default_legacy_chunk_id_method() {
        let source = include_str!("contracts.rs");
        let chunk_writer_section = source
            .split("pub trait ChunkWriter: Send + Sync")
            .nth(1)
            .expect("ChunkWriter trait declaration should exist");
        let chunk_writer_block = chunk_writer_section
            .split("pub trait ZarrLayoutAdapter: Send + Sync")
            .next()
            .expect("ChunkWriter block should end before ZarrLayoutAdapter");
        assert!(
            chunk_writer_block.contains("fn write_chunk_by_tuple_key("),
            "ChunkWriter should require tuple-key writes for the production path"
        );
        assert!(
            chunk_writer_block.contains("fn write_chunk_by_id(")
                && chunk_writer_block.contains("write_chunk_by_id not supported"),
            "ChunkWriter should provide a default legacy write_chunk_by_id() implementation"
        );
    }
}
