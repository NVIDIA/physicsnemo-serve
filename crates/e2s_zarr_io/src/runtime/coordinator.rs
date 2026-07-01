/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Write orchestration: plan → reserve → submit → copy-barrier → ack.

use std::collections::{BTreeMap, HashSet};
use std::panic::{self, AssertUnwindSafe};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::inflight_guard::{InflightAsyncWriteGuard, InflightWriteCounter};
use super::write_task::PendingChunkWrite;
use crate::core::chunk_id::ChunkId;
use crate::core::contracts::{
    BufferPool, ChunkKeyRegistry, ChunkPlanner, ChunkWriter, ConsolidationScope, CopyEngine,
    MetadataConsolidator, WorkScheduler, ZarrLayoutAdapter,
};
use crate::core::errors::{DeferredWriteError, SyncWriteError};
#[cfg(any(test, feature = "test-utils"))]
use crate::core::types::DEFAULT_NUM_THREADS;
use crate::core::types::{
    ArrayRegistration, BatchId, BufferLease, CloseInternalTiming, CloseReport, Nanoseconds,
    WriteInternalTiming,
};

/// Bundled dependencies for constructing a [`WriteCoordinator`].
///
/// Using a struct avoids a long positional-argument list and silences
/// `clippy::too_many_arguments`.
pub(crate) struct WriteCoordinatorComponents {
    /// Planner that expands write requests into concrete chunk tasks.
    pub planner: Arc<dyn ChunkPlanner>,
    /// Registry enforcing no-overwrite chunk identity reservation/commit semantics.
    pub chunk_registry: Arc<dyn ChunkKeyRegistry>,
    /// Scheduler that executes copy+flush worker tasks.
    pub scheduler: Arc<dyn WorkScheduler>,
    /// Buffer pool used to acquire pooled/transient leases for chunk payloads.
    pub buffer_pool: Arc<dyn BufferPool>,
    /// Copy engine that transfers caller payloads into leased buffers.
    pub copy_engine: Arc<dyn CopyEngine>,
    /// Chunk writer that persists encoded chunk bytes to storage.
    pub chunk_writer: Arc<dyn ChunkWriter>,
    /// Metadata consolidator used for registration-time and close-time consolidation.
    pub metadata_consolidator: Arc<dyn MetadataConsolidator>,
    /// Layout adapter that maps logical chunk keys to on-disk paths.
    pub layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    /// Names of coordinates that define write-time parallelism.
    pub parallel_coord_names: Vec<String>,
    /// Base scheduler queue capacity before first-write auto-scaling.
    pub queue_capacity: usize,
}

/// Test-utils-visible constructor bundle for coordinator benchmarks.
///
/// This mirrors [`WriteCoordinatorComponents`] but is exposed only when the
/// `test-utils` feature is enabled so benchmark targets can construct a
/// coordinator without widening the default public API surface.
#[cfg(feature = "test-utils")]
pub struct TestWriteCoordinatorComponents {
    /// Planner that expands write requests into concrete chunk tasks.
    pub planner: Arc<dyn ChunkPlanner>,
    /// Registry enforcing no-overwrite chunk identity reservation/commit semantics.
    pub chunk_registry: Arc<dyn ChunkKeyRegistry>,
    /// Scheduler that executes copy+flush worker tasks.
    pub scheduler: Arc<dyn WorkScheduler>,
    /// Buffer pool used to acquire pooled/transient leases for chunk payloads.
    pub buffer_pool: Arc<dyn BufferPool>,
    /// Copy engine that transfers caller payloads into leased buffers.
    pub copy_engine: Arc<dyn CopyEngine>,
    /// Chunk writer that persists encoded chunk bytes to storage.
    pub chunk_writer: Arc<dyn ChunkWriter>,
    /// Metadata consolidator used for registration-time and close-time consolidation.
    pub metadata_consolidator: Arc<dyn MetadataConsolidator>,
    /// Layout adapter that maps logical chunk keys to on-disk paths.
    pub layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    /// Names of coordinates that define write-time parallelism.
    pub parallel_coord_names: Vec<String>,
    /// Base scheduler queue capacity before first-write auto-scaling.
    pub queue_capacity: usize,
}

#[cfg(feature = "test-utils")]
impl From<TestWriteCoordinatorComponents> for WriteCoordinatorComponents {
    fn from(value: TestWriteCoordinatorComponents) -> Self {
        Self {
            planner: value.planner,
            chunk_registry: value.chunk_registry,
            scheduler: value.scheduler,
            buffer_pool: value.buffer_pool,
            copy_engine: value.copy_engine,
            chunk_writer: value.chunk_writer,
            metadata_consolidator: value.metadata_consolidator,
            layout_adapter: value.layout_adapter,
            parallel_coord_names: value.parallel_coord_names,
            queue_capacity: value.queue_capacity,
        }
    }
}

/// Test-utils wrapper over the internal [`WriteCoordinator`].
///
/// This type exists only for benchmark/test harnesses that compile the crate
/// with `--features test-utils`.
#[cfg(feature = "test-utils")]
pub struct TestWriteCoordinator {
    inner: WriteCoordinator,
}

#[cfg(feature = "test-utils")]
impl TestWriteCoordinator {
    /// Create a new benchmark coordinator from bundled components.
    ///
    /// # Panics
    ///
    /// Panics if thread pool construction fails. For fallible construction,
    /// use [`try_new_with_num_threads`](Self::try_new_with_num_threads).
    #[must_use]
    pub fn new(components: TestWriteCoordinatorComponents) -> Self {
        Self {
            inner: WriteCoordinator::try_new_with_num_threads(
                components.into(),
                DEFAULT_NUM_THREADS,
            )
            .expect("failed to construct TestWriteCoordinator with default thread pools"),
        }
    }

    /// Create a benchmark coordinator with explicit worker thread count.
    pub fn try_new_with_num_threads(
        components: TestWriteCoordinatorComponents,
        num_threads: usize,
    ) -> Result<Self, SyncWriteError> {
        WriteCoordinator::try_new_with_num_threads(components.into(), num_threads)
            .map(|inner| Self { inner })
    }

    /// Create a benchmark coordinator that re-uses pre-built Rayon pools.
    ///
    /// Use this in benchmarks where a new coordinator is created per iteration
    /// (e.g. first-write benches) to avoid measuring thread pool creation
    /// overhead instead of `submit_write` latency.
    pub fn new_with_shared_pools(
        components: TestWriteCoordinatorComponents,
        copy_pool: Arc<rayon::ThreadPool>,
        flush_pool: Arc<rayon::ThreadPool>,
    ) -> Self {
        Self {
            inner: WriteCoordinator::new_with_pools(components.into(), copy_pool, flush_pool),
        }
    }

    /// Submit a write request through the internal pipeline.
    pub fn submit_write(
        &self,
        req: &crate::core::types::InferenceWriteRequest,
        registered_array_ids: &[u32],
        registered_coords: &crate::core::types::CoordMap,
    ) -> Result<crate::core::types::WriteCopyAck, SyncWriteError> {
        self.inner
            .submit_write(req, registered_array_ids, registered_coords)
    }

    /// Run coordinator close logic with timeout and optional registration context.
    pub fn close(
        &self,
        timeout_seconds: f64,
        registration: Option<&ArrayRegistration>,
    ) -> Result<CloseReport, SyncWriteError> {
        self.inner.close(timeout_seconds, registration)
    }

    /// Return timing for the last write submitted through this coordinator.
    #[must_use]
    pub fn last_write_timing(&self) -> Option<WriteInternalTiming> {
        self.inner.last_write_timing()
    }
}

const DEFAULT_QUEUE_BURST_STEPS: usize = 2;
#[path = "coordinator_submit.rs"]
mod submit_write_impl;

fn build_named_rayon_pool(
    pool_kind: &'static str,
    thread_name_prefix: &'static str,
    num_threads: usize,
) -> Result<Arc<rayon::ThreadPool>, SyncWriteError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(move |idx| format!("{thread_name_prefix}-{idx}"))
        .build()
        .map(Arc::new)
        .map_err(|err| SyncWriteError::ContractViolation {
            message: format!(
                "failed to build {pool_kind} thread pool: {err}. \
This usually means the process cannot create additional threads (check ulimit -u, cgroup pids.max, or seccomp thread restrictions)."
            ),
        })
}

fn build_copy_rayon_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError> {
    build_named_rayon_pool("copy", "e2s-zarr-copy", num_threads)
}

fn build_flush_rayon_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError> {
    build_named_rayon_pool("flush", "e2s-zarr-flush", num_threads)
}

/// Ensures async flush resources unwind in the required order.
///
/// The drop contract is:
/// 1) return the buffer lease to `BufferPool`,
/// 2) then decrement/notify inflight async-write accounting via guard drop.
struct AsyncFlushCleanup {
    buffer_pool: Arc<dyn BufferPool>,
    lease: Option<BufferLease>,
    inflight_guard: Option<InflightAsyncWriteGuard>,
}

impl AsyncFlushCleanup {
    fn new(
        buffer_pool: Arc<dyn BufferPool>,
        lease: BufferLease,
        inflight_guard: InflightAsyncWriteGuard,
    ) -> Self {
        Self {
            buffer_pool,
            lease: Some(lease),
            inflight_guard: Some(inflight_guard),
        }
    }

    fn with_lease_bytes<F>(&mut self, required_bytes: usize, f: F) -> Result<(), SyncWriteError>
    where
        F: FnOnce(&[u8]) -> Result<(), SyncWriteError>,
    {
        let lease = self
            .lease
            .as_mut()
            .ok_or_else(|| SyncWriteError::ContractViolation {
                message: "async flush cleanup lost lease before write".to_string(),
            })?;
        lease.with_bytes(required_bytes, f)
    }
}

impl Drop for AsyncFlushCleanup {
    fn drop(&mut self) {
        // ORDERING CONTRACT (async flush): release buffer lease before inflight guard drops.
        // close() waits for lease return before waiting for inflight async writes.
        if let Some(lease) = self.lease.take() {
            self.buffer_pool.release(lease);
        }
        if let Some(inflight_guard) = self.inflight_guard.take() {
            drop(inflight_guard);
        }
    }
}

/// Orchestrates the write pipeline: planning, reservation, submission, and
/// copy-barrier acknowledgment.
pub(crate) struct WriteCoordinator {
    planner: Arc<dyn ChunkPlanner>,
    chunk_registry: Arc<dyn ChunkKeyRegistry>,
    scheduler: Arc<dyn WorkScheduler>,
    buffer_pool: Arc<dyn BufferPool>,
    copy_engine: Arc<dyn CopyEngine>,
    chunk_writer: Arc<dyn ChunkWriter>,
    metadata_consolidator: Arc<dyn MetadataConsolidator>,
    layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    parallel_coord_names: Vec<String>,
    num_threads: usize,
    copy_rayon_pool: Arc<rayon::ThreadPool>,
    flush_rayon_pool: Arc<rayon::ThreadPool>,
    configured_queue_capacity: usize,
    effective_queue_capacity: OnceLock<usize>,
    write_batches_seen: AtomicUsize,
    inflight_async_writes: InflightWriteCounter,
    deferred_write_errors: Arc<Mutex<Vec<DeferredWriteError>>>,
    last_write_timing: Arc<Mutex<Option<WriteInternalTiming>>>,
    /// Set to `true` by `persist_registration_metadata` and back to `false`
    /// after the handle is joined.  Acts as a lock-free fast path for
    /// `wait_for_registration_metadata` — the mutex below is only acquired
    /// when this flag is `true`.
    has_pending_registration: AtomicBool,
    pending_registration_join: Mutex<Option<std::thread::JoinHandle<Result<(), SyncWriteError>>>>,
    /// Per-array chunk byte sizes observed from the first committed write.
    /// Populated by flush workers; read at close time to avoid expensive
    /// `infer_chunk_descriptor` directory scanning.
    observed_chunk_bytes: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl WriteCoordinator {
    /// Create a coordinator from bundled components using `DEFAULT_NUM_THREADS`.
    ///
    /// # Panics
    ///
    /// Panics if thread pool construction fails. Restricted to test code; production
    /// callers must use [`try_new_with_num_threads`](Self::try_new_with_num_threads).
    #[cfg(test)]
    #[must_use]
    pub fn new(components: WriteCoordinatorComponents) -> Self {
        Self::try_new_with_num_threads(components, DEFAULT_NUM_THREADS)
            .expect("failed to construct WriteCoordinator with default thread pools")
    }

    /// Create a new coordinator from bundled components using the provided thread count.
    pub fn try_new_with_num_threads(
        components: WriteCoordinatorComponents,
        num_threads: usize,
    ) -> Result<Self, SyncWriteError> {
        Self::try_new_with_pool_builders(
            components,
            num_threads,
            build_copy_rayon_pool,
            build_flush_rayon_pool,
        )
    }

    /// Create a coordinator with externally-owned Rayon pools.
    ///
    /// Thread count is inferred from the copy pool. This avoids the overhead of
    /// creating/destroying OS threads per-coordinator, which matters when
    /// benchmarks construct a fresh coordinator every iteration.
    #[cfg(feature = "test-utils")]
    fn new_with_pools(
        components: WriteCoordinatorComponents,
        copy_pool: Arc<rayon::ThreadPool>,
        flush_pool: Arc<rayon::ThreadPool>,
    ) -> Self {
        let num_threads = copy_pool.current_num_threads();
        Self {
            planner: components.planner,
            chunk_registry: components.chunk_registry,
            scheduler: components.scheduler,
            buffer_pool: components.buffer_pool,
            copy_engine: components.copy_engine,
            chunk_writer: components.chunk_writer,
            metadata_consolidator: components.metadata_consolidator,
            layout_adapter: components.layout_adapter,
            parallel_coord_names: components.parallel_coord_names,
            num_threads,
            copy_rayon_pool: copy_pool,
            flush_rayon_pool: flush_pool,
            configured_queue_capacity: components.queue_capacity,
            effective_queue_capacity: OnceLock::new(),
            write_batches_seen: AtomicUsize::new(0),
            inflight_async_writes: Arc::new((Mutex::new(0), Condvar::new())),
            deferred_write_errors: Arc::new(Mutex::new(Vec::new())),
            last_write_timing: Arc::new(Mutex::new(None)),
            has_pending_registration: AtomicBool::new(false),
            pending_registration_join: Mutex::new(None),
            observed_chunk_bytes: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    #[cfg(test)]
    fn try_new_with_pool_builders_for_test<CopyBuilder, FlushBuilder>(
        components: WriteCoordinatorComponents,
        num_threads: usize,
        copy_pool_builder: CopyBuilder,
        flush_pool_builder: FlushBuilder,
    ) -> Result<Self, SyncWriteError>
    where
        CopyBuilder: FnOnce(usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError>,
        FlushBuilder: FnOnce(usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError>,
    {
        Self::try_new_with_pool_builders(
            components,
            num_threads,
            copy_pool_builder,
            flush_pool_builder,
        )
    }

    fn try_new_with_pool_builders<CopyBuilder, FlushBuilder>(
        components: WriteCoordinatorComponents,
        num_threads: usize,
        copy_pool_builder: CopyBuilder,
        flush_pool_builder: FlushBuilder,
    ) -> Result<Self, SyncWriteError>
    where
        CopyBuilder: FnOnce(usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError>,
        FlushBuilder: FnOnce(usize) -> Result<Arc<rayon::ThreadPool>, SyncWriteError>,
    {
        if num_threads == 0 {
            return Err(SyncWriteError::Validation {
                message: "num_threads must be greater than 0".to_string(),
            });
        }
        let copy_rayon_pool =
            copy_pool_builder(num_threads).map_err(|err| SyncWriteError::ContractViolation {
                message: format!("failed to build copy thread pool: {err}"),
            })?;
        let flush_rayon_pool =
            flush_pool_builder(num_threads).map_err(|err| SyncWriteError::ContractViolation {
                message: format!("failed to build flush thread pool: {err}"),
            })?;

        Ok(Self {
            planner: components.planner,
            chunk_registry: components.chunk_registry,
            scheduler: components.scheduler,
            buffer_pool: components.buffer_pool,
            copy_engine: components.copy_engine,
            chunk_writer: components.chunk_writer,
            metadata_consolidator: components.metadata_consolidator,
            layout_adapter: components.layout_adapter,
            parallel_coord_names: components.parallel_coord_names,
            num_threads,
            copy_rayon_pool,
            flush_rayon_pool,
            configured_queue_capacity: components.queue_capacity,
            effective_queue_capacity: OnceLock::new(),
            write_batches_seen: AtomicUsize::new(0),
            inflight_async_writes: Arc::new((Mutex::new(0), Condvar::new())),
            deferred_write_errors: Arc::new(Mutex::new(Vec::new())),
            last_write_timing: Arc::new(Mutex::new(None)),
            has_pending_registration: AtomicBool::new(false),
            pending_registration_join: Mutex::new(None),
            observed_chunk_bytes: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    /// Dispatch registration-time metadata consolidation to a background thread
    /// and, when possible, eagerly initialize the buffer pool on a second thread.
    ///
    /// Returns immediately; both operations are guaranteed to be complete before
    /// the first `submit_write()` or `close()` call, both of which call
    /// `wait_for_registration_metadata()` before proceeding.
    pub fn persist_registration_metadata(
        &self,
        registration: &ArrayRegistration,
    ) -> Result<(), SyncWriteError> {
        let consolidator = Arc::clone(&self.metadata_consolidator);
        let layout_adapter = Arc::clone(&self.layout_adapter);
        let parallel_coord_names = self.parallel_coord_names.clone();
        let registration = registration.clone();
        let buffer_pool = Arc::clone(&self.buffer_pool);

        let handle = std::thread::Builder::new()
            .name("e2s-zarr-reg-meta".to_string())
            .spawn(move || {
                // Spawn early pool init on a separate thread so it runs
                // concurrently with metadata consolidation (~120ms each).
                let pool_init_handle = if buffer_pool.supports_early_init() {
                    let pool = Arc::clone(&buffer_pool);
                    let array_count = registration.array_names.len();
                    Some(
                        std::thread::Builder::new()
                            .name("e2s-zarr-pool-init".to_string())
                            .spawn(move || {
                                // SAFETY CONTRACT: supports_early_init() guarantees
                                // pool_buffer_bytes is Fixed, so the pool ignores
                                // first_write_max_required_bytes entirely. Passing 0
                                // is safe — the Fixed value is used for buffer sizing.
                                let hint = crate::core::types::FirstWriteSizingHint {
                                    first_write_task_count: array_count,
                                    first_write_max_required_bytes: 0,
                                };
                                pool.initialize_if_needed(&hint)
                            })
                            .ok(),
                    )
                } else {
                    None
                };

                // Consolidate registration metadata (root + coords) to disk.
                consolidator.consolidate_scoped(
                    layout_adapter.as_ref(),
                    Some(&registration),
                    &parallel_coord_names,
                    ConsolidationScope::RegistrationOnly,
                    None,
                )?;

                // Join pool init thread (if spawned).
                if let Some(Some(handle)) = pool_init_handle {
                    handle
                        .join()
                        .map_err(|_| SyncWriteError::ContractViolation {
                            message: "early pool init thread panicked".to_string(),
                        })??;
                }
                Ok(())
            })
            .map_err(|e| SyncWriteError::ContractViolation {
                message: format!("failed to spawn registration metadata thread: {e}"),
            })?;

        *self.pending_registration_join.lock().map_err(|_| {
            SyncWriteError::ContractViolation {
                message: "lock poisoned".to_string(),
            }
        })? = Some(handle);
        // Set the flag AFTER storing the handle so the fast path in
        // wait_for_registration_metadata never observes true with a None handle.
        self.has_pending_registration.store(true, Ordering::Release);
        Ok(())
    }

    /// Join the background registration metadata thread if one is pending.
    ///
    /// Idempotent: a second call returns `Ok(())` immediately.
    /// Fast path: a single atomic load when no join is pending.
    pub fn wait_for_registration_metadata(&self) -> Result<(), SyncWriteError> {
        // Lock-free fast path — avoids mutex acquisition on every submit_write
        // after the background thread has already been joined.
        if !self.has_pending_registration.load(Ordering::Acquire) {
            return Ok(());
        }
        let handle = self
            .pending_registration_join
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "lock poisoned".to_string(),
            })?
            .take();
        let Some(handle) = handle else {
            // Another caller joined first (shouldn't happen in practice).
            return Ok(());
        };
        let result = handle
            .join()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "registration metadata thread panicked".to_string(),
            })?;
        self.has_pending_registration
            .store(false, Ordering::Release);
        result
    }

    fn resolve_queue_capacity(&self, first_write_task_count: usize) -> usize {
        *self.effective_queue_capacity.get_or_init(|| {
            let burst_capacity = first_write_task_count
                .saturating_mul(DEFAULT_QUEUE_BURST_STEPS)
                .max(1);
            self.configured_queue_capacity.max(burst_capacity)
        })
    }

    #[cfg(test)]
    fn debug_effective_queue_capacity(&self) -> Option<usize> {
        self.effective_queue_capacity.get().copied()
    }

    #[must_use]
    /// Return the internal timing snapshot for the most recent `submit_write()` call.
    pub fn last_write_timing(&self) -> Option<WriteInternalTiming> {
        match self.last_write_timing.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn store_last_write_timing(&self, timing: WriteInternalTiming) {
        match self.last_write_timing.lock() {
            Ok(mut guard) => {
                *guard = Some(timing);
            }
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = Some(timing);
            }
        }
    }

    fn elapsed_ns(start: Instant) -> Nanoseconds {
        Nanoseconds(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }

    /// Execute close-time coordinator actions through trait boundaries.
    ///
    /// Callers must validate `timeout_seconds` before calling — `SyncZarrBackend::close()`
    /// is the single validation point for this parameter.
    pub fn close(
        &self,
        timeout_seconds: f64,
        registration: Option<&ArrayRegistration>,
    ) -> Result<CloseReport, SyncWriteError> {
        debug_assert!(
            timeout_seconds.is_finite() && timeout_seconds > 0.0,
            "caller must validate timeout_seconds before reaching coordinator"
        );
        let close_start = Instant::now();
        let pool_warmup_status = self.buffer_pool.warmup_status();
        let close_deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);

        // OPTIMIZATION: Overlap metadata consolidation with async write drain.
        // Per-array metadata (.zarray, zarr.json) and .zmetadata files are written to
        // different paths than chunk data, so consolidation can safely run while the
        // last async chunk flushes complete. We use std::thread::scope to spawn a
        // scoped thread for metadata consolidation while the main thread drains.

        // Wait for the registration metadata thread first (fast path via AtomicBool).
        if let Err(e) = self.wait_for_registration_metadata() {
            let _teardown = self.teardown_runtime_resources();
            return Err(e);
        }

        // Drain scheduler so no new tasks are enqueued during close.
        if let Err(e) = self.scheduler.drain() {
            let _teardown = self.teardown_runtime_resources();
            return Err(e);
        }

        // Snapshot observed_chunk_bytes AFTER drain so it captures entries from all flush
        // workers that completed during drain. Taking it before drain could miss entries
        // inserted between the snapshot and drain completion, causing `can_overlap` to be
        // false even though the chunk files exist on disk.
        let chunk_bytes_snapshot = self
            .observed_chunk_bytes
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "observed_chunk_bytes lock poisoned".to_string(),
            })?
            .clone();

        // Determine if we can safely overlap consolidation with async drain.
        // Overlap is safe only when observed_chunk_bytes covers all registered arrays,
        // so per-array metadata uses the cached sizes (O(1)) instead of scanning the
        // filesystem for chunk files that may still be in-flight.
        let can_overlap = registration
            .map(|r| {
                !r.coords.is_empty()
                    && r.array_names
                        .iter()
                        .all(|name| chunk_bytes_snapshot.contains_key(name.as_str()))
            })
            .unwrap_or(false);

        let drain_start = Instant::now();
        let (drain_result, consolidate_result, async_drain_ns, consolidate_ns) = if can_overlap {
            // FAST PATH: overlap metadata consolidation with async write drain.
            // Per-array metadata (.zarray, zarr.json) writes to different paths than chunk data,
            // so consolidation can safely run while the last async chunk flushes complete.
            std::thread::scope(|s| {
                // Spawn metadata consolidation on a background thread.
                let meta_handle = s.spawn(|| {
                    let start = Instant::now();
                    let result = self.metadata_consolidator.consolidate_scoped(
                        self.layout_adapter.as_ref(),
                        registration,
                        &self.parallel_coord_names,
                        ConsolidationScope::CloseOnly,
                        Some(&chunk_bytes_snapshot),
                    );
                    let ns: Nanoseconds = (start.elapsed().as_nanos() as u64).into();
                    (result, ns)
                });

                // Main thread: drain async writes (lease return + inflight wait + error check).
                // ORDERING CONTRACT (close): wait_pooled_leases_returned must run before wait_for_async_writes.
                // Async flush workers release the buffer lease before the inflight guard drops. If
                // these waits are swapped, lease waiting can time out while flush workers are still
                // making forward progress.
                let drain_r: Result<(), SyncWriteError> = (|| {
                    self.buffer_pool.wait_pooled_leases_returned(
                        Self::remaining_timeout_seconds(close_deadline),
                    )?;
                    self.wait_for_async_writes(close_deadline)?;

                    let deferred_write_errors =
                        std::mem::take(&mut *self.deferred_write_errors.lock().map_err(|_| {
                            SyncWriteError::ContractViolation {
                                message: "deferred write errors lock poisoned".to_string(),
                            }
                        })?);
                    if !deferred_write_errors.is_empty() {
                        return Err(SyncWriteError::DeferredWriteFailures {
                            failures: deferred_write_errors,
                        });
                    }
                    Ok(())
                })();
                let drain_ns: Nanoseconds = (drain_start.elapsed().as_nanos() as u64).into();

                let meta_join = meta_handle
                    .join()
                    .map_err(|_| SyncWriteError::ContractViolation {
                        message: "metadata consolidation thread panicked during close".to_string(),
                    });
                match meta_join {
                    Ok((meta_result, meta_ns)) => (drain_r, meta_result, drain_ns, meta_ns),
                    Err(e) => (drain_r, Err(e), drain_ns, Nanoseconds::from(0u64)),
                }
            })
        } else {
            // SEQUENTIAL PATH: drain first, then consolidate (needs FS-visible chunks).
            let drain_r: Result<(), SyncWriteError> = (|| {
                // ORDERING CONTRACT (close): wait_pooled_leases_returned must run before wait_for_async_writes.
                self.buffer_pool
                    .wait_pooled_leases_returned(Self::remaining_timeout_seconds(close_deadline))?;
                self.wait_for_async_writes(close_deadline)?;

                let deferred_write_errors =
                    std::mem::take(&mut *self.deferred_write_errors.lock().map_err(|_| {
                        SyncWriteError::ContractViolation {
                            message: "deferred write errors lock poisoned".to_string(),
                        }
                    })?);
                if !deferred_write_errors.is_empty() {
                    return Err(SyncWriteError::DeferredWriteFailures {
                        failures: deferred_write_errors,
                    });
                }
                Ok(())
            })();
            let drain_ns: Nanoseconds = (drain_start.elapsed().as_nanos() as u64).into();

            let cons_start = Instant::now();
            let cons_result = if drain_r.is_ok() {
                self.metadata_consolidator.consolidate_scoped(
                    self.layout_adapter.as_ref(),
                    registration,
                    &self.parallel_coord_names,
                    ConsolidationScope::CloseOnly,
                    Some(&chunk_bytes_snapshot),
                )
            } else {
                Ok(())
            };
            let cons_ns: Nanoseconds = (cons_start.elapsed().as_nanos() as u64).into();

            (drain_r, cons_result, drain_ns, cons_ns)
        };

        // Merge drain + consolidation results, preserving both error messages
        // when both fail (possible in the overlap path where they run concurrently).
        let combined_result = match (drain_result, consolidate_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(drain_err), Ok(())) => Err(drain_err),
            (Ok(()), Err(meta_err)) => Err(meta_err),
            (Err(drain_err), Err(meta_err)) => Err(SyncWriteError::ContractViolation {
                message: format!(
                    "async drain failed: {drain_err}; metadata consolidation also failed: {meta_err}"
                ),
            }),
        };

        // Phase 3: teardown
        let teardown_start = Instant::now();
        let teardown_result = self.teardown_runtime_resources();
        let teardown_ns: Nanoseconds = (teardown_start.elapsed().as_nanos() as u64).into();

        let total_close_ns: Nanoseconds = (close_start.elapsed().as_nanos() as u64).into();
        let close_timing = Some(CloseInternalTiming {
            async_drain_ns,
            consolidate_ns,
            teardown_ns,
            total_close_ns,
        });

        match (combined_result, teardown_result) {
            (Ok(()), Ok(())) => Ok(CloseReport {
                write_batches_seen: self.write_batches_seen.load(Ordering::Acquire),
                deferred_write_errors: Vec::new(),
                metadata_consolidated: true,
                resources_released: true,
                pool_warmup_status,
                closed: true,
                close_timing,
            }),
            (Ok(()), Err(teardown_err)) => Err(teardown_err),
            (Err(primary_err), Ok(())) => Err(primary_err),
            (Err(primary_err), Err(teardown_err)) => Err(SyncWriteError::ContractViolation {
                message: format!(
                    "close stage failed: {primary_err}; teardown also failed: {teardown_err}"
                ),
            }),
        }
    }

    fn rollback_reserved_ids_not_dispatched(
        &self,
        chunk_ids: &[ChunkId],
        flushed_ids: &HashSet<ChunkId>,
    ) {
        for chunk_id in chunk_ids {
            if !flushed_ids.contains(chunk_id) {
                self.chunk_registry.release_failed_id(chunk_id);
            }
        }
    }

    fn snapshot_dispatched_chunk_ids(
        flush_dispatched_chunk_ids: &Arc<Mutex<Vec<ChunkId>>>,
    ) -> HashSet<ChunkId> {
        let vec = match flush_dispatched_chunk_ids.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        vec.into_iter().collect()
    }

    fn remaining_timeout_seconds(deadline: Instant) -> f64 {
        deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
            .max(f64::EPSILON)
    }

    fn teardown_runtime_resources(&self) -> Result<(), SyncWriteError> {
        let mut failures = Vec::new();
        if let Err(err) = self.buffer_pool.shutdown() {
            failures.push(format!("buffer_pool.shutdown: {err}"));
        }
        if let Err(err) = self.scheduler.shutdown() {
            failures.push(format!("scheduler.shutdown: {err}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(SyncWriteError::ContractViolation {
                message: format!("close teardown failed: {}", failures.join("; ")),
            })
        }
    }

    fn wait_for_async_writes(&self, deadline: Instant) -> Result<(), SyncWriteError> {
        let (lock, cv) = &*self.inflight_async_writes;
        let mut inflight = lock.lock().map_err(|_| SyncWriteError::ContractViolation {
            message: "inflight async write counter lock poisoned".to_string(),
        })?;
        while *inflight > 0 {
            let now = Instant::now();
            if now >= deadline {
                return Err(SyncWriteError::LeaseReturnTimeout {
                    outstanding_leases: *inflight,
                });
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next_inflight, wait_result) =
                cv.wait_timeout(inflight, remaining).map_err(|_| {
                    SyncWriteError::ContractViolation {
                        message: "inflight async write counter lock poisoned while waiting"
                            .to_string(),
                    }
                })?;
            inflight = next_inflight;
            if wait_result.timed_out() && *inflight > 0 {
                return Err(SyncWriteError::LeaseReturnTimeout {
                    outstanding_leases: *inflight,
                });
            }
        }
        Ok(())
    }

    fn record_first_error(
        first_error: &Mutex<Option<SyncWriteError>>,
        cancelled: &AtomicBool,
        err: SyncWriteError,
    ) {
        let mut slot = match first_error.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.is_none() {
            *slot = Some(err);
        }
        cancelled.store(true, Ordering::Release);
    }

    fn fail_with_abort_batch_cleanup(
        &self,
        batch_id: BatchId,
        err: SyncWriteError,
    ) -> SyncWriteError {
        match self.scheduler.abort_batch(batch_id) {
            Ok(()) => err,
            Err(abort_err) => SyncWriteError::ContractViolation {
                message: format!(
                    "submit_write failed for batch {batch_id}: {err}; scheduler abort_batch failed: {abort_err}"
                ),
            },
        }
    }

    fn spawn_async_write_handle(&self, task: PendingChunkWrite) -> Result<(), SyncWriteError> {
        let PendingChunkWrite {
            batch_id,
            array_name,
            chunk_id,
            tuple_key,
            required_bytes,
            lease,
        } = task;

        let inflight_guard = match InflightAsyncWriteGuard::register(&self.inflight_async_writes) {
            Ok(guard) => guard,
            Err(err) => {
                self.buffer_pool.release(lease);
                return Err(err);
            }
        };

        let chunk_writer = Arc::clone(&self.chunk_writer);
        let chunk_registry = Arc::clone(&self.chunk_registry);
        let buffer_pool = Arc::clone(&self.buffer_pool);
        let deferred_write_errors = Arc::clone(&self.deferred_write_errors);
        let observed_chunk_bytes = Arc::clone(&self.observed_chunk_bytes);
        let flush_rayon_pool = Arc::clone(&self.flush_rayon_pool);
        flush_rayon_pool.spawn(move || {
            let mut cleanup = AsyncFlushCleanup::new(buffer_pool, lease, inflight_guard);
            let deferred_error = match panic::catch_unwind(AssertUnwindSafe(|| {
                let write_result = cleanup.with_lease_bytes(required_bytes, |payload| {
                    chunk_writer.write_chunk_by_tuple_key(&array_name, &tuple_key, payload)
                });
                match write_result {
                    Ok(()) => {
                        chunk_registry.mark_committed_id(&chunk_id);
                        // Record chunk byte size for this array (first write wins).
                        {
                            let mut map = observed_chunk_bytes.lock().unwrap_or_else(|p| {
                                eprintln!(
                                    "[e2s_zarr_io] observed_chunk_bytes mutex poisoned \
                                     (another flush worker panicked); recovering"
                                );
                                p.into_inner()
                            });
                            map.entry(array_name.clone()).or_insert(required_bytes);
                        }
                        None
                    }
                    Err(err) => {
                        chunk_registry.release_failed_id(&chunk_id);
                        Some(DeferredWriteError {
                            batch_id,
                            chunk_id: Some(chunk_id),
                            message: err.to_string(),
                        })
                    }
                }
            })) {
                Ok(error) => error,
                Err(_) => {
                    chunk_registry.release_failed_id(&chunk_id);
                    Some(DeferredWriteError {
                        batch_id,
                        chunk_id: Some(chunk_id),
                        message: "asynchronous chunk write worker panicked".to_string(),
                    })
                }
            };

            if let Some(err) = deferred_error {
                let mut errors = match deferred_write_errors.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                errors.push(err);
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::{
        BufferPool, ChunkKeyRegistry, ChunkPlanner, ChunkWriter, CopyEngine, MetadataConsolidator,
        WorkScheduler, ZarrLayoutAdapter,
    };
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        BatchId, BufferLease, ChunkPathSpec, ChunkTask, CoordMap, CopyCompletion,
        FirstWriteSizingHint, InferenceWriteRequest, InputArray, InputArraySource,
        MetadataPathSpec, PlannedWriteBatch, PoolWarmupStatus, TransientBuffer, TupleChunkKey,
        ZarrFormat,
    };

    use super::{WriteCoordinator, WriteCoordinatorComponents};

    struct FixedPlanner {
        planned: PlannedWriteBatch,
    }

    impl ChunkPlanner for FixedPlanner {
        fn plan_batch(
            &self,
            _req: &InferenceWriteRequest,
            _array_ids: &[u32],
            _registered_coords: &CoordMap,
        ) -> Result<PlannedWriteBatch, SyncWriteError> {
            Ok(self.planned.clone())
        }
    }

    struct SequencedPlanner {
        planned_batches: Mutex<VecDeque<PlannedWriteBatch>>,
    }

    impl SequencedPlanner {
        fn new(planned_batches: Vec<PlannedWriteBatch>) -> Self {
            Self {
                planned_batches: Mutex::new(VecDeque::from(planned_batches)),
            }
        }
    }

    impl ChunkPlanner for SequencedPlanner {
        fn plan_batch(
            &self,
            _req: &InferenceWriteRequest,
            _array_ids: &[u32],
            _registered_coords: &CoordMap,
        ) -> Result<PlannedWriteBatch, SyncWriteError> {
            let mut guard =
                self.planned_batches
                    .lock()
                    .map_err(|_| SyncWriteError::ContractViolation {
                        message: "sequenced planner lock poisoned".to_string(),
                    })?;
            guard
                .pop_front()
                .ok_or_else(|| SyncWriteError::ContractViolation {
                    message: "sequenced planner has no remaining planned batches".to_string(),
                })
        }
    }

    #[derive(Default)]
    struct RecordingRegistry {
        committed: Mutex<Vec<ChunkId>>,
        released: Mutex<Vec<ChunkId>>,
    }

    impl RecordingRegistry {
        fn committed_ids(&self) -> Vec<ChunkId> {
            self.committed
                .lock()
                .expect("committed lock poisoned")
                .clone()
        }

        fn released_ids(&self) -> Vec<ChunkId> {
            self.released
                .lock()
                .expect("released lock poisoned")
                .clone()
        }
    }

    impl ChunkKeyRegistry for RecordingRegistry {
        fn reserve_many_ids(&self, _chunk_ids: &[ChunkId]) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn mark_committed_id(&self, chunk_id: &ChunkId) {
            self.committed
                .lock()
                .expect("committed lock poisoned")
                .push(*chunk_id);
        }

        fn release_failed_id(&self, chunk_id: &ChunkId) {
            self.released
                .lock()
                .expect("released lock poisoned")
                .push(*chunk_id);
        }
    }

    #[derive(Clone, Copy)]
    enum SchedulerMode {
        FailOnSubmit,
        FailOnWait,
        AlwaysOk,
    }

    struct ControlledScheduler {
        mode: SchedulerMode,
    }

    impl WorkScheduler for ControlledScheduler {
        fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
            match self.mode {
                SchedulerMode::FailOnSubmit => Err(SyncWriteError::io_failed("submit failed")),
                SchedulerMode::FailOnWait | SchedulerMode::AlwaysOk => Ok(()),
            }
        }

        fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            match self.mode {
                SchedulerMode::FailOnSubmit | SchedulerMode::AlwaysOk => Ok(()),
                SchedulerMode::FailOnWait => Err(SyncWriteError::io_failed("wait failed")),
            }
        }

        fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn drain(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct AccountingLeakTrackingScheduler {
        submitted_by_batch: Mutex<HashMap<BatchId, usize>>,
        copied_by_batch: Mutex<HashMap<BatchId, usize>>,
        submit_calls: AtomicUsize,
        fail_on_submit_call: usize,
    }

    impl AccountingLeakTrackingScheduler {
        fn new(fail_on_submit_call: usize) -> Self {
            Self {
                submitted_by_batch: Mutex::new(HashMap::new()),
                copied_by_batch: Mutex::new(HashMap::new()),
                submit_calls: AtomicUsize::new(0),
                fail_on_submit_call,
            }
        }

        fn submitted_for_batch(&self, batch_id: BatchId) -> usize {
            self.submitted_by_batch
                .lock()
                .expect("submitted-by-batch lock poisoned")
                .get(&batch_id)
                .copied()
                .unwrap_or(0)
        }
    }

    impl WorkScheduler for AccountingLeakTrackingScheduler {
        fn submit(&self, task: ChunkTask) -> Result<(), SyncWriteError> {
            let submit_call_idx = self.submit_calls.fetch_add(1, Ordering::AcqRel) + 1;
            let mut submitted =
                self.submitted_by_batch
                    .lock()
                    .map_err(|_| SyncWriteError::ContractViolation {
                        message: "accounting scheduler submit lock poisoned".to_string(),
                    })?;
            submitted
                .entry(task.batch_id)
                .and_modify(|count| *count += 1)
                .or_insert(1);

            if submit_call_idx == self.fail_on_submit_call {
                return Err(SyncWriteError::io_failed("forced submit failure"));
            }
            Ok(())
        }

        fn mark_copied(&self, batch_id: BatchId) -> Result<(), SyncWriteError> {
            let mut copied =
                self.copied_by_batch
                    .lock()
                    .map_err(|_| SyncWriteError::ContractViolation {
                        message: "accounting scheduler copied lock poisoned".to_string(),
                    })?;
            copied
                .entry(batch_id)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            Ok(())
        }

        fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn abort_batch(&self, batch_id: BatchId) -> Result<(), SyncWriteError> {
            self.submitted_by_batch
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "accounting scheduler abort submitted lock poisoned".to_string(),
                })?
                .remove(&batch_id);
            self.copied_by_batch
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "accounting scheduler abort copied lock poisoned".to_string(),
                })?
                .remove(&batch_id);
            Ok(())
        }

        fn drain(&self) -> Result<(), SyncWriteError> {
            self.submitted_by_batch
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "accounting scheduler drain submitted lock poisoned".to_string(),
                })?
                .clear();
            self.copied_by_batch
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "accounting scheduler drain copied lock poisoned".to_string(),
                })?
                .clear();
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            self.drain()
        }
    }

    #[derive(Default)]
    struct TrackingBarrierScheduler {
        submitted: AtomicUsize,
        copied: AtomicUsize,
        wait_lock: Mutex<()>,
        wait_cv: Condvar,
    }

    impl TrackingBarrierScheduler {
        fn submitted_tasks(&self) -> usize {
            self.submitted.load(Ordering::Acquire)
        }

        fn copied_tasks(&self) -> usize {
            self.copied.load(Ordering::Acquire)
        }
    }

    impl WorkScheduler for TrackingBarrierScheduler {
        fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
            self.submitted.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            self.copied.fetch_add(1, Ordering::AcqRel);
            self.wait_cv.notify_all();
            Ok(())
        }

        fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            let mut guard =
                self.wait_lock
                    .lock()
                    .map_err(|_| SyncWriteError::ContractViolation {
                        message: "tracking scheduler wait lock poisoned".to_string(),
                    })?;
            loop {
                if self.copied_tasks() >= self.submitted_tasks() {
                    return Ok(());
                }
                guard =
                    self.wait_cv
                        .wait(guard)
                        .map_err(|_| SyncWriteError::ContractViolation {
                            message: "tracking scheduler wait lock poisoned while waiting"
                                .to_string(),
                        })?;
            }
        }

        fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn drain(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct StubLayoutAdapter;

    impl ZarrLayoutAdapter for StubLayoutAdapter {
        fn zarr_format(&self) -> ZarrFormat {
            ZarrFormat::V2
        }

        fn chunk_path_for(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
        ) -> Result<ChunkPathSpec, SyncWriteError> {
            Ok(ChunkPathSpec {
                relative_path: "a0/c.0".to_string(),
            })
        }

        fn chunk_path_for_tuple_key(
            &self,
            array_name: &str,
            tuple_key: &TupleChunkKey,
        ) -> Result<ChunkPathSpec, SyncWriteError> {
            Ok(ChunkPathSpec {
                relative_path: format!("{}/{}", array_name, tuple_key.render('.')),
            })
        }

        fn metadata_paths(&self) -> Result<MetadataPathSpec, SyncWriteError> {
            Ok(MetadataPathSpec {
                root_paths: vec![".zgroup".to_string()],
                per_array_paths: vec![".zarray".to_string()],
            })
        }
    }

    #[derive(Default)]
    struct StubBufferPool;

    impl BufferPool for StubBufferPool {
        fn initialize_if_needed(&self, _hint: &FirstWriteSizingHint) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn acquire(&self, required_bytes: usize) -> Result<BufferLease, SyncWriteError> {
            Ok(BufferLease::Transient(TransientBuffer::new(required_bytes)))
        }

        fn release(&self, _lease: BufferLease) {}

        fn warmup_status(&self) -> PoolWarmupStatus {
            PoolWarmupStatus::default()
        }

        fn wait_pooled_leases_returned(&self, _timeout_seconds: f64) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct TrackingBufferPool {
        release_count: AtomicUsize,
    }

    impl TrackingBufferPool {
        fn releases(&self) -> usize {
            self.release_count.load(Ordering::Relaxed)
        }
    }

    impl BufferPool for TrackingBufferPool {
        fn initialize_if_needed(&self, _hint: &FirstWriteSizingHint) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn acquire(&self, required_bytes: usize) -> Result<BufferLease, SyncWriteError> {
            Ok(BufferLease::Transient(TransientBuffer::new(required_bytes)))
        }

        fn release(&self, _lease: BufferLease) {
            self.release_count.fetch_add(1, Ordering::Relaxed);
        }

        fn warmup_status(&self) -> PoolWarmupStatus {
            PoolWarmupStatus::default()
        }

        fn wait_pooled_leases_returned(&self, _timeout_seconds: f64) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum LeaseWaitMode {
        AlwaysOk,
        Timeout,
    }

    struct CloseTrackingBufferPool {
        wait_mode: LeaseWaitMode,
        fail_shutdown: bool,
        shutdown_calls: AtomicUsize,
    }

    impl CloseTrackingBufferPool {
        fn new(wait_mode: LeaseWaitMode, fail_shutdown: bool) -> Self {
            Self {
                wait_mode,
                fail_shutdown,
                shutdown_calls: AtomicUsize::new(0),
            }
        }

        fn shutdown_calls(&self) -> usize {
            self.shutdown_calls.load(Ordering::Acquire)
        }
    }

    impl BufferPool for CloseTrackingBufferPool {
        fn initialize_if_needed(&self, _hint: &FirstWriteSizingHint) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn acquire(&self, required_bytes: usize) -> Result<BufferLease, SyncWriteError> {
            Ok(BufferLease::Transient(TransientBuffer::new(required_bytes)))
        }

        fn release(&self, _lease: BufferLease) {}

        fn warmup_status(&self) -> PoolWarmupStatus {
            PoolWarmupStatus::default()
        }

        fn wait_pooled_leases_returned(&self, _timeout_seconds: f64) -> Result<(), SyncWriteError> {
            match self.wait_mode {
                LeaseWaitMode::AlwaysOk => Ok(()),
                LeaseWaitMode::Timeout => Err(SyncWriteError::LeaseReturnTimeout {
                    outstanding_leases: 1,
                }),
            }
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            self.shutdown_calls.fetch_add(1, Ordering::AcqRel);
            if self.fail_shutdown {
                return Err(SyncWriteError::io_failed("buffer pool shutdown failed"));
            }
            Ok(())
        }
    }

    struct CloseTrackingScheduler {
        fail_drain: bool,
        fail_shutdown: bool,
        shutdown_calls: AtomicUsize,
    }

    impl CloseTrackingScheduler {
        fn new(fail_drain: bool, fail_shutdown: bool) -> Self {
            Self {
                fail_drain,
                fail_shutdown,
                shutdown_calls: AtomicUsize::new(0),
            }
        }

        fn shutdown_calls(&self) -> usize {
            self.shutdown_calls.load(Ordering::Acquire)
        }
    }

    impl WorkScheduler for CloseTrackingScheduler {
        fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn drain(&self) -> Result<(), SyncWriteError> {
            if self.fail_drain {
                return Err(SyncWriteError::io_failed("drain failed"));
            }
            Ok(())
        }

        fn shutdown(&self) -> Result<(), SyncWriteError> {
            self.shutdown_calls.fetch_add(1, Ordering::AcqRel);
            if self.fail_shutdown {
                return Err(SyncWriteError::io_failed("scheduler shutdown failed"));
            }
            Ok(())
        }
    }

    #[derive(Default)]
    struct StubCopyEngine;

    impl CopyEngine for StubCopyEngine {
        fn copy_into_lease(
            &self,
            _src: &InputArray,
            _lease: &mut BufferLease,
            _required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            Ok(CopyCompletion::ImmediateHostCopy)
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct StubChunkWriter;

    impl ChunkWriter for StubChunkWriter {
        fn write_chunk_by_id(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn write_chunk_by_tuple_key(
            &self,
            _array_name: &str,
            _tuple_key: &TupleChunkKey,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct ThreadNameRecordingCopyEngine {
        thread_names: Mutex<Vec<String>>,
    }

    impl ThreadNameRecordingCopyEngine {
        fn thread_names(&self) -> Vec<String> {
            self.thread_names
                .lock()
                .expect("copy thread name lock poisoned")
                .clone()
        }
    }

    impl CopyEngine for ThreadNameRecordingCopyEngine {
        fn copy_into_lease(
            &self,
            _src: &InputArray,
            _lease: &mut BufferLease,
            _required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            let name = std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_string();
            self.thread_names
                .lock()
                .expect("copy thread name lock poisoned")
                .push(name);
            Ok(CopyCompletion::ImmediateHostCopy)
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct ThreadNameRecordingChunkWriter {
        thread_names: Mutex<Vec<String>>,
    }

    impl ThreadNameRecordingChunkWriter {
        fn thread_names(&self) -> Vec<String> {
            self.thread_names
                .lock()
                .expect("flush thread name lock poisoned")
                .clone()
        }
    }

    impl ChunkWriter for ThreadNameRecordingChunkWriter {
        fn write_chunk_by_id(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn write_chunk_by_tuple_key(
            &self,
            _array_name: &str,
            _tuple_key: &TupleChunkKey,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            let name = std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_string();
            self.thread_names
                .lock()
                .expect("flush thread name lock poisoned")
                .push(name);
            Ok(())
        }
    }

    struct BlockingChunkWriter {
        gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
    }

    impl ChunkWriter for BlockingChunkWriter {
        fn write_chunk_by_id(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            self.wait_gate()
        }

        fn write_chunk_by_tuple_key(
            &self,
            _array_name: &str,
            _tuple_key: &TupleChunkKey,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            self.wait_gate()
        }
    }

    impl BlockingChunkWriter {
        fn wait_gate(&self) -> Result<(), SyncWriteError> {
            let (lock, condvar) = &*self.gate;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            while !*ready {
                ready = condvar
                    .wait(ready)
                    .expect("blocking gate lock poisoned while waiting");
            }
            Ok(())
        }
    }

    struct CountingBlockingChunkWriter {
        gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
        started: AtomicUsize,
    }

    impl CountingBlockingChunkWriter {
        fn started_count(&self) -> usize {
            self.started.load(Ordering::Acquire)
        }

        fn wait_gate(&self) -> Result<(), SyncWriteError> {
            let (lock, condvar) = &*self.gate;
            let mut ready = lock.lock().expect("counting gate lock poisoned");
            while !*ready {
                ready = condvar
                    .wait(ready)
                    .expect("counting gate lock poisoned while waiting");
            }
            Ok(())
        }
    }

    impl ChunkWriter for CountingBlockingChunkWriter {
        fn write_chunk_by_id(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            self.started.fetch_add(1, Ordering::AcqRel);
            self.wait_gate()
        }

        fn write_chunk_by_tuple_key(
            &self,
            _array_name: &str,
            _tuple_key: &TupleChunkKey,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            self.started.fetch_add(1, Ordering::AcqRel);
            self.wait_gate()
        }
    }

    #[derive(Default)]
    struct RecordingPayloadWriter {
        payloads: Mutex<Vec<Vec<u8>>>,
    }

    impl RecordingPayloadWriter {
        fn payloads(&self) -> Vec<Vec<u8>> {
            self.payloads
                .lock()
                .expect("recording payload lock poisoned")
                .clone()
        }
    }

    impl ChunkWriter for RecordingPayloadWriter {
        fn write_chunk_by_id(
            &self,
            _array_name: &str,
            _chunk_id: &ChunkId,
            _bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn write_chunk_by_tuple_key(
            &self,
            _array_name: &str,
            _tuple_key: &TupleChunkKey,
            bytes: &[u8],
        ) -> Result<(), SyncWriteError> {
            self.payloads
                .lock()
                .expect("recording payload lock poisoned")
                .push(bytes.to_vec());
            Ok(())
        }
    }

    #[derive(Default)]
    struct StubMetadataConsolidator;

    impl MetadataConsolidator for StubMetadataConsolidator {
        fn consolidate(
            &self,
            _layout: &dyn ZarrLayoutAdapter,
            _registration: Option<&crate::core::types::ArrayRegistration>,
            _parallel_coord_names: &[String],
        ) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct FailingMetadataConsolidator;

    impl MetadataConsolidator for FailingMetadataConsolidator {
        fn consolidate(
            &self,
            _layout: &dyn ZarrLayoutAdapter,
            _registration: Option<&crate::core::types::ArrayRegistration>,
            _parallel_coord_names: &[String],
        ) -> Result<(), SyncWriteError> {
            Err(SyncWriteError::metadata_consolidation_failed(
                "metadata consolidation failed",
            ))
        }
    }

    struct SlowCountingMetadataConsolidator {
        delay: Duration,
        calls: AtomicUsize,
    }

    impl SlowCountingMetadataConsolidator {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    impl MetadataConsolidator for SlowCountingMetadataConsolidator {
        fn consolidate(
            &self,
            _layout: &dyn ZarrLayoutAdapter,
            _registration: Option<&crate::core::types::ArrayRegistration>,
            _parallel_coord_names: &[String],
        ) -> Result<(), SyncWriteError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            std::thread::sleep(self.delay);
            Ok(())
        }
    }

    struct LeaseOverwritingCopyEngine {
        staged_payload: Vec<u8>,
    }

    impl CopyEngine for LeaseOverwritingCopyEngine {
        fn copy_into_lease(
            &self,
            _src: &InputArray,
            lease: &mut BufferLease,
            required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            lease.write_from_host_bytes(&self.staged_payload, required_bytes)?;
            Ok(CopyCompletion::ImmediateHostCopy)
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct SlowCopyEngine {
        delay: Duration,
    }

    impl CopyEngine for SlowCopyEngine {
        fn copy_into_lease(
            &self,
            src: &InputArray,
            lease: &mut BufferLease,
            required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            std::thread::sleep(self.delay);
            if let InputArraySource::HostBytes(payload) = &src.source {
                lease.write_from_host_bytes(payload, required_bytes)?;
            }
            Ok(CopyCompletion::ImmediateHostCopy)
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct FailOnPayloadPrefixCopyEngine {
        fail_first_byte: u8,
        fail_delay: Duration,
    }

    impl CopyEngine for FailOnPayloadPrefixCopyEngine {
        fn copy_into_lease(
            &self,
            src: &InputArray,
            lease: &mut BufferLease,
            required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            if let InputArraySource::HostBytes(payload) = &src.source {
                if payload.first().copied() == Some(self.fail_first_byte) {
                    std::thread::sleep(self.fail_delay);
                    return Err(SyncWriteError::copy_failed(
                        "simulated copy failure after partial batch progress",
                    ));
                }
                lease.write_from_host_bytes(payload, required_bytes)?;
                return Ok(CopyCompletion::ImmediateHostCopy);
            }
            Err(SyncWriteError::copy_failed(
                "fail-on-prefix copy engine supports only HostBytes inputs",
            ))
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    struct GateControlledCopyEngine {
        gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
        started: AtomicUsize,
    }

    impl GateControlledCopyEngine {
        fn new(open: bool) -> Self {
            Self {
                gate: std::sync::Arc::new((Mutex::new(open), Condvar::new())),
                started: AtomicUsize::new(0),
            }
        }

        fn set_open(&self, open: bool) {
            let (lock, condvar) = &*self.gate;
            let mut state = lock.lock().expect("copy gate lock poisoned");
            *state = open;
            if open {
                condvar.notify_all();
            }
        }

        fn started_count(&self) -> usize {
            self.started.load(Ordering::Acquire)
        }
    }

    impl CopyEngine for GateControlledCopyEngine {
        fn copy_into_lease(
            &self,
            src: &InputArray,
            lease: &mut BufferLease,
            required_bytes: usize,
        ) -> Result<CopyCompletion, SyncWriteError> {
            self.started.fetch_add(1, Ordering::AcqRel);
            let (lock, condvar) = &*self.gate;
            let mut open = lock.lock().expect("copy gate lock poisoned");
            while !*open {
                open = condvar
                    .wait(open)
                    .expect("copy gate lock poisoned while waiting");
            }
            if let InputArraySource::HostBytes(payload) = &src.source {
                lease.write_from_host_bytes(payload, required_bytes)?;
            }
            Ok(CopyCompletion::ImmediateHostCopy)
        }

        fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
            Ok(())
        }
    }

    fn planned_two_task_batch() -> PlannedWriteBatch {
        let chunk_id_a = ChunkId::new(1, 10);
        let chunk_id_b = ChunkId::new(1, 11);
        PlannedWriteBatch {
            batch_id: BatchId(99),
            chunk_ids: vec![chunk_id_a, chunk_id_b],
            tasks: vec![
                ChunkTask {
                    batch_id: BatchId(99),
                    array_name: "a".to_string(),
                    chunk_id: chunk_id_a,
                    tuple_key: TupleChunkKey::new(vec![]),
                    required_bytes: 4,
                    input: InputArray {
                        nbytes: 4,
                        source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
                    },
                },
                ChunkTask {
                    batch_id: BatchId(99),
                    array_name: "b".to_string(),
                    chunk_id: chunk_id_b,
                    tuple_key: TupleChunkKey::new(vec![]),
                    required_bytes: 4,
                    input: InputArray {
                        nbytes: 4,
                        source: InputArraySource::HostBytes(vec![5, 6, 7, 8].into()),
                    },
                },
            ],
        }
    }

    fn planned_single_task_batch() -> PlannedWriteBatch {
        let chunk_id = ChunkId::new(1, 10);
        PlannedWriteBatch {
            batch_id: BatchId(99),
            chunk_ids: vec![chunk_id],
            tasks: vec![ChunkTask {
                batch_id: BatchId(99),
                array_name: "a".to_string(),
                chunk_id,
                tuple_key: TupleChunkKey::new(vec![]),
                required_bytes: 4,
                input: InputArray {
                    nbytes: 4,
                    source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
                },
            }],
        }
    }

    fn planned_single_cuda_task_batch() -> PlannedWriteBatch {
        let chunk_id = ChunkId::new(1, 12);
        PlannedWriteBatch {
            batch_id: BatchId(101),
            chunk_ids: vec![chunk_id],
            tasks: vec![ChunkTask {
                batch_id: BatchId(101),
                array_name: "cuda_arr".to_string(),
                chunk_id,
                tuple_key: TupleChunkKey::new(vec![0]),
                required_bytes: 4,
                input: InputArray {
                    nbytes: 4,
                    source: InputArraySource::CudaDevicePtr {
                        ptr: 0xDEAD_BEEF,
                        device_ordinal: 0,
                        producer_stream: None,
                    },
                },
            }],
        }
    }

    fn planned_multi_task_batch(task_count: usize) -> PlannedWriteBatch {
        let mut chunk_ids = Vec::with_capacity(task_count);
        let mut tasks = Vec::with_capacity(task_count);
        for idx in 0..task_count {
            let linear = u64::try_from(idx).expect("task index fits u64");
            let chunk_id = ChunkId::new(1, linear);
            chunk_ids.push(chunk_id);
            tasks.push(ChunkTask {
                batch_id: BatchId(88),
                array_name: "a".to_string(),
                chunk_id,
                tuple_key: TupleChunkKey::new(vec![idx]),
                required_bytes: 4,
                input: InputArray {
                    nbytes: 4,
                    source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
                },
            });
        }
        PlannedWriteBatch {
            batch_id: BatchId(88),
            chunk_ids,
            tasks,
        }
    }

    fn empty_request() -> InferenceWriteRequest {
        InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: Vec::new(),
            arrays: Vec::new(),
        }
    }

    #[test]
    fn thread_pool_build_failure_returns_error_instead_of_panicking() {
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned_single_task_batch(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let err = match WriteCoordinator::try_new_with_pool_builders_for_test(
            WriteCoordinatorComponents {
                planner,
                chunk_registry: registry,
                scheduler,
                buffer_pool,
                copy_engine,
                chunk_writer: writer,
                metadata_consolidator: metadata,
                layout_adapter: layout,
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            2,
            |_num_threads| {
                Err(SyncWriteError::ContractViolation {
                    message: "simulated thread creation failure".to_string(),
                })
            },
            |_num_threads| {
                panic!("flush pool builder should not run when copy pool build already failed")
            },
        ) {
            Ok(_) => panic!("copy pool build failure must be surfaced"),
            Err(err) => err,
        };

        assert!(
            matches!(
                err,
                SyncWriteError::ContractViolation { ref message }
                if message.contains("failed to build copy thread pool")
                    && message.contains("simulated thread creation failure")
            ),
            "expected actionable copy thread-pool build error, got: {err:?}"
        );
    }

    #[test]
    fn thread_pool_size_is_configurable_per_coordinator_instance() {
        let one_thread = WriteCoordinator::try_new_with_num_threads(
            WriteCoordinatorComponents {
                planner: std::sync::Arc::new(FixedPlanner {
                    planned: planned_single_task_batch(),
                }),
                chunk_registry: std::sync::Arc::new(RecordingRegistry::default()),
                scheduler: std::sync::Arc::new(ControlledScheduler {
                    mode: SchedulerMode::AlwaysOk,
                }),
                buffer_pool: std::sync::Arc::new(StubBufferPool),
                copy_engine: std::sync::Arc::new(StubCopyEngine),
                chunk_writer: std::sync::Arc::new(StubChunkWriter),
                metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
                layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            1,
        )
        .expect("1-thread coordinator should construct");

        let four_threads = WriteCoordinator::try_new_with_num_threads(
            WriteCoordinatorComponents {
                planner: std::sync::Arc::new(FixedPlanner {
                    planned: planned_single_task_batch(),
                }),
                chunk_registry: std::sync::Arc::new(RecordingRegistry::default()),
                scheduler: std::sync::Arc::new(ControlledScheduler {
                    mode: SchedulerMode::AlwaysOk,
                }),
                buffer_pool: std::sync::Arc::new(StubBufferPool),
                copy_engine: std::sync::Arc::new(StubCopyEngine),
                chunk_writer: std::sync::Arc::new(StubChunkWriter),
                metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
                layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            4,
        )
        .expect("4-thread coordinator should construct");

        assert_eq!(one_thread.copy_rayon_pool.current_num_threads(), 1);
        assert_eq!(one_thread.flush_rayon_pool.current_num_threads(), 1);
        assert_eq!(four_threads.copy_rayon_pool.current_num_threads(), 4);
        assert_eq!(four_threads.flush_rayon_pool.current_num_threads(), 4);
    }

    #[test]
    fn try_new_with_num_threads_rejects_zero() {
        let err = match WriteCoordinator::try_new_with_num_threads(
            WriteCoordinatorComponents {
                planner: std::sync::Arc::new(FixedPlanner {
                    planned: planned_single_task_batch(),
                }),
                chunk_registry: std::sync::Arc::new(RecordingRegistry::default()),
                scheduler: std::sync::Arc::new(ControlledScheduler {
                    mode: SchedulerMode::AlwaysOk,
                }),
                buffer_pool: std::sync::Arc::new(StubBufferPool),
                copy_engine: std::sync::Arc::new(StubCopyEngine),
                chunk_writer: std::sync::Arc::new(StubChunkWriter),
                metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
                layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            0,
        ) {
            Ok(_) => panic!("num_threads=0 must be rejected"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("num_threads must be greater than 0")
        ));
    }

    #[test]
    fn async_flush_cleanup_drop_skips_release_when_parts_taken() {
        let inflight_counter: crate::runtime::inflight_guard::InflightWriteCounter =
            std::sync::Arc::new((Mutex::new(0), Condvar::new()));
        let inflight_guard =
            crate::runtime::inflight_guard::InflightAsyncWriteGuard::register(&inflight_counter)
                .expect("inflight guard registration should succeed");
        let tracking_pool = std::sync::Arc::new(TrackingBufferPool::default());
        let lease = BufferLease::Transient(TransientBuffer::new(4));

        let mut cleanup =
            super::AsyncFlushCleanup::new(tracking_pool.clone(), lease, inflight_guard);
        let detached_lease = cleanup
            .lease
            .take()
            .expect("lease should be present before cleanup drop");
        let detached_guard = cleanup
            .inflight_guard
            .take()
            .expect("inflight guard should be present before cleanup drop");
        drop(cleanup);
        assert_eq!(
            tracking_pool.releases(),
            0,
            "drop should skip buffer pool release when lease was already taken"
        );

        drop(detached_lease);
        drop(detached_guard);
        let (lock, _) = &*inflight_counter;
        assert_eq!(
            *lock
                .lock()
                .expect("inflight counter lock should not be poisoned"),
            0,
            "detached guard drop should decrement inflight counter"
        );
    }

    #[test]
    fn record_first_error_preserves_existing_error_slot() {
        let first_error = Mutex::new(Some(SyncWriteError::io_failed("first")));
        let cancelled = AtomicBool::new(false);

        WriteCoordinator::record_first_error(
            &first_error,
            &cancelled,
            SyncWriteError::io_failed("second"),
        );

        let stored = first_error
            .lock()
            .expect("record_first_error test lock should not be poisoned");
        assert!(matches!(
            stored.as_ref(),
            Some(SyncWriteError::IoFailed { message, .. }) if message.contains("first")
        ));
        assert!(
            cancelled.load(Ordering::Acquire),
            "record_first_error should always raise cancellation flag"
        );
    }

    #[test]
    fn record_first_error_sets_cancelled_when_error_slot_lock_poisoned() {
        let first_error = std::sync::Arc::new(Mutex::new(None::<SyncWriteError>));
        let first_error_for_poison = std::sync::Arc::clone(&first_error);
        let poison_handle = std::thread::spawn(move || {
            let _held = first_error_for_poison
                .lock()
                .expect("poison helper should acquire first-error lock");
            panic!("intentional poison for record_first_error lock path");
        });
        assert!(
            poison_handle.join().is_err(),
            "poison helper must panic to poison lock"
        );

        let cancelled = AtomicBool::new(false);
        WriteCoordinator::record_first_error(
            first_error.as_ref(),
            &cancelled,
            SyncWriteError::io_failed("stored through poison recovery"),
        );
        assert!(
            cancelled.load(Ordering::Acquire),
            "record_first_error must still set cancelled when lock is poisoned"
        );
        let stored = match first_error.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert!(
            matches!(
                stored.as_ref(),
                Some(SyncWriteError::IoFailed { message, .. })
                if message.contains("stored through poison recovery")
            ),
            "record_first_error should recover poisoned lock and preserve first error detail",
        );
    }

    #[test]
    fn wait_for_async_writes_returns_timeout_when_deadline_already_elapsed() {
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner {
                planned: planned_single_task_batch(),
            }),
            chunk_registry: std::sync::Arc::new(RecordingRegistry::default()),
            scheduler: std::sync::Arc::new(ControlledScheduler {
                mode: SchedulerMode::AlwaysOk,
            }),
            buffer_pool: std::sync::Arc::new(StubBufferPool),
            copy_engine: std::sync::Arc::new(StubCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let (lock, _) = &*coordinator.inflight_async_writes;
        {
            let mut inflight = lock
                .lock()
                .expect("inflight counter lock should not be poisoned");
            *inflight = 1;
        }
        let err = coordinator
            .wait_for_async_writes(Instant::now())
            .expect_err("elapsed deadline with inflight writes should timeout");
        assert!(matches!(
            err,
            SyncWriteError::LeaseReturnTimeout { outstanding_leases }
            if outstanding_leases == 1
        ));

        {
            let mut inflight = lock
                .lock()
                .expect("inflight counter lock should not be poisoned after timeout");
            *inflight = 0;
        }
        coordinator
            .close(1.0, None)
            .expect("close should succeed after inflight counter reset");
    }

    #[test]
    fn submit_write_empty_plan_returns_zero_copy_ack() {
        let planned = PlannedWriteBatch {
            batch_id: BatchId(404),
            chunk_ids: Vec::new(),
            tasks: Vec::new(),
        };
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner { planned }),
            chunk_registry: std::sync::Arc::new(RecordingRegistry::default()),
            scheduler: std::sync::Arc::new(ControlledScheduler {
                mode: SchedulerMode::AlwaysOk,
            }),
            buffer_pool: std::sync::Arc::new(StubBufferPool),
            copy_engine: std::sync::Arc::new(StubCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("empty planned batch should short-circuit successfully");
        assert_eq!(ack.batch_id, BatchId(404));
        assert_eq!(ack.copied_tasks, 0);
        let timing = coordinator
            .last_write_timing()
            .expect("timing snapshot should be present for empty batch");
        assert_eq!(timing.task_count, 0);
        assert_eq!(timing.worker_count, 0);
        assert_eq!(timing.enqueued_task_count, 0);
        assert_eq!(timing.copied_task_count, 0);

        coordinator
            .close(1.0, None)
            .expect("close should succeed for empty planned batch");
    }

    #[test]
    fn submit_write_propagates_wait_copy_completion_error() {
        struct WaitFailCopyEngine;

        impl CopyEngine for WaitFailCopyEngine {
            fn copy_into_lease(
                &self,
                src: &InputArray,
                lease: &mut BufferLease,
                required_bytes: usize,
            ) -> Result<CopyCompletion, SyncWriteError> {
                if let InputArraySource::HostBytes(payload) = &src.source {
                    lease.write_from_host_bytes(payload, required_bytes)?;
                }
                Ok(CopyCompletion::ImmediateHostCopy)
            }

            fn wait_copy_completion(
                &self,
                _completion: CopyCompletion,
            ) -> Result<(), SyncWriteError> {
                Err(SyncWriteError::copy_failed("wait copy completion failed"))
            }
        }

        let planned = planned_single_task_batch();
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner {
                planned: planned.clone(),
            }),
            chunk_registry: registry.clone(),
            scheduler: std::sync::Arc::new(ControlledScheduler {
                mode: SchedulerMode::AlwaysOk,
            }),
            buffer_pool: std::sync::Arc::new(TrackingBufferPool::default()),
            copy_engine: std::sync::Arc::new(WaitFailCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("wait_copy_completion failure should surface");
        assert!(matches!(
            err,
            SyncWriteError::CopyFailed { ref message, .. }
            if message.contains("wait copy completion failed")
        ));
        assert_eq!(
            registry.released_ids(),
            planned.chunk_ids,
            "all reserved chunk ids should roll back when wait_copy_completion fails"
        );
    }

    #[test]
    fn submit_write_propagates_mark_copied_error() {
        struct MarkCopiedFailScheduler;

        impl WorkScheduler for MarkCopiedFailScheduler {
            fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Err(SyncWriteError::io_failed("mark_copied failed"))
            }

            fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn drain(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn shutdown(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }
        }

        let planned = planned_single_task_batch();
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner {
                planned: planned.clone(),
            }),
            chunk_registry: registry.clone(),
            scheduler: std::sync::Arc::new(MarkCopiedFailScheduler),
            buffer_pool: std::sync::Arc::new(TrackingBufferPool::default()),
            copy_engine: std::sync::Arc::new(StubCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("mark_copied failure should surface");
        assert!(matches!(
            err,
            SyncWriteError::IoFailed { ref message, .. } if message.contains("mark_copied failed")
        ));
        assert_eq!(
            registry.released_ids(),
            planned.chunk_ids,
            "all reserved chunk ids should roll back when mark_copied fails"
        );
    }

    #[test]
    fn submit_write_recovers_copy_worker_panic_into_error_and_aborts_batch() {
        #[derive(Default)]
        struct AbortTrackingScheduler {
            abort_calls: AtomicUsize,
        }

        impl WorkScheduler for AbortTrackingScheduler {
            fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                self.abort_calls.fetch_add(1, Ordering::AcqRel);
                Ok(())
            }

            fn drain(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn shutdown(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }
        }

        struct PanicCopyEngine;

        impl CopyEngine for PanicCopyEngine {
            fn copy_into_lease(
                &self,
                _src: &InputArray,
                _lease: &mut BufferLease,
                _required_bytes: usize,
            ) -> Result<CopyCompletion, SyncWriteError> {
                panic!("simulated copy worker panic");
            }

            fn wait_copy_completion(
                &self,
                _completion: CopyCompletion,
            ) -> Result<(), SyncWriteError> {
                Ok(())
            }
        }

        let planned = planned_single_task_batch();
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(AbortTrackingScheduler::default());
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner {
                planned: planned.clone(),
            }),
            chunk_registry: registry.clone(),
            scheduler: scheduler.clone(),
            buffer_pool: std::sync::Arc::new(TrackingBufferPool::default()),
            copy_engine: std::sync::Arc::new(PanicCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let submit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            coordinator.submit_write(&empty_request(), &[], &CoordMap::new())
        }));
        assert!(
            submit_result.is_ok(),
            "submit_write should convert copy-worker panic into SyncWriteError, not unwind"
        );
        let err = submit_result
            .expect("panic should be recovered by submit_write")
            .expect_err("copy-worker panic should return SyncWriteError");
        assert!(
            matches!(
                err,
                SyncWriteError::ContractViolation { ref message }
                if message.contains("panic")
            ),
            "panic should surface as contract violation with panic context"
        );
        assert_eq!(
            scheduler.abort_calls.load(Ordering::Acquire),
            1,
            "panic path must abort scheduler batch accounting"
        );
        assert_eq!(
            registry.released_ids(),
            planned.chunk_ids,
            "panic path should roll back reserved chunk IDs"
        );
    }

    #[test]
    fn close_timeout_validation_is_centralized_in_backend() {
        let source = include_str!("../backend.rs");
        let close_section = source
            .split("fn close(&self, timeout_seconds: f64)")
            .nth(1)
            .expect("SyncZarrBackend::close should exist");
        let close_block = close_section
            .split("fn is_closed")
            .next()
            .expect("close block should end before is_closed");
        assert!(
            close_block.contains("timeout_seconds.is_finite()")
                && close_block.contains("timeout_seconds <= 0.0"),
            "SyncZarrBackend::close() must validate timeout_seconds (single validation point)"
        );
    }

    #[test]
    fn submit_write_propagates_inflight_guard_registration_error() {
        let planned = planned_single_task_batch();
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let buffer_pool = std::sync::Arc::new(TrackingBufferPool::default());
        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner: std::sync::Arc::new(FixedPlanner {
                planned: planned.clone(),
            }),
            chunk_registry: registry.clone(),
            scheduler: std::sync::Arc::new(ControlledScheduler {
                mode: SchedulerMode::AlwaysOk,
            }),
            buffer_pool: buffer_pool.clone(),
            copy_engine: std::sync::Arc::new(StubCopyEngine),
            chunk_writer: std::sync::Arc::new(StubChunkWriter),
            metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
            layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let inflight_counter = std::sync::Arc::clone(&coordinator.inflight_async_writes);
        let poison_handle = std::thread::spawn(move || {
            let (lock, _) = &*inflight_counter;
            let _held = lock
                .lock()
                .expect("poison helper should acquire inflight lock");
            panic!("intentional poison for inflight guard register path");
        });
        assert!(
            poison_handle.join().is_err(),
            "poison helper must panic to poison inflight lock"
        );

        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("poisoned inflight lock should fail async guard registration");
        assert!(matches!(
            err,
            SyncWriteError::ContractViolation { ref message }
            if message.contains("inflight async write counter lock poisoned")
        ));
        assert_eq!(
            buffer_pool.releases(),
            1,
            "spawn_async_write_handle failure should release acquired lease immediately"
        );
        assert_eq!(
            registry.released_ids(),
            planned.chunk_ids,
            "failed async registration should roll back reserved chunk ids"
        );
    }

    #[test]
    fn submit_write_reports_queue_disconnect_when_worker_receivers_drop() {
        struct CoordinatedFailCopyEngine {
            gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
        }

        impl CopyEngine for CoordinatedFailCopyEngine {
            fn copy_into_lease(
                &self,
                src: &InputArray,
                lease: &mut BufferLease,
                required_bytes: usize,
            ) -> Result<CopyCompletion, SyncWriteError> {
                if let InputArraySource::HostBytes(payload) = &src.source {
                    lease.write_from_host_bytes(payload, required_bytes)?;
                }
                let (lock, cv) = &*self.gate;
                let mut should_fail = lock.lock().expect("copy gate lock should not be poisoned");
                while !*should_fail {
                    should_fail = cv
                        .wait(should_fail)
                        .expect("copy gate lock should not be poisoned while waiting");
                }
                Err(SyncWriteError::copy_failed(
                    "coordinated copy failure for queue disconnect test",
                ))
            }

            fn wait_copy_completion(
                &self,
                _completion: CopyCompletion,
            ) -> Result<(), SyncWriteError> {
                Ok(())
            }
        }

        struct TriggerFailOnSecondSubmitScheduler {
            gate: std::sync::Arc<(Mutex<bool>, Condvar)>,
            submit_calls: AtomicUsize,
        }

        impl TriggerFailOnSecondSubmitScheduler {
            fn submit_calls(&self) -> usize {
                self.submit_calls.load(Ordering::Acquire)
            }
        }

        impl WorkScheduler for TriggerFailOnSecondSubmitScheduler {
            fn submit(&self, _task: ChunkTask) -> Result<(), SyncWriteError> {
                let call = self.submit_calls.fetch_add(1, Ordering::AcqRel) + 1;
                if call == 2 {
                    let (lock, cv) = &*self.gate;
                    let mut should_fail = lock
                        .lock()
                        .expect("scheduler gate lock should not be poisoned");
                    *should_fail = true;
                    cv.notify_all();
                    drop(should_fail);
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(())
            }

            fn mark_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn wait_copied(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn drain(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn shutdown(&self) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn abort_batch(&self, _batch_id: BatchId) -> Result<(), SyncWriteError> {
                Ok(())
            }
        }

        let planned = planned_two_task_batch();
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let scheduler = std::sync::Arc::new(TriggerFailOnSecondSubmitScheduler {
            gate: gate.clone(),
            submit_calls: AtomicUsize::new(0),
        });
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let coordinator = WriteCoordinator::try_new_with_num_threads(
            WriteCoordinatorComponents {
                planner: std::sync::Arc::new(FixedPlanner {
                    planned: planned.clone(),
                }),
                chunk_registry: registry.clone(),
                scheduler,
                buffer_pool: std::sync::Arc::new(TrackingBufferPool::default()),
                copy_engine: std::sync::Arc::new(CoordinatedFailCopyEngine { gate }),
                chunk_writer: std::sync::Arc::new(StubChunkWriter),
                metadata_consolidator: std::sync::Arc::new(StubMetadataConsolidator),
                layout_adapter: std::sync::Arc::new(StubLayoutAdapter),
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            1,
        )
        .expect("single-thread coordinator should construct");

        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("worker receiver disconnect should surface as submit failure");
        assert!(matches!(
            err,
            SyncWriteError::CopyFailed { ref message, .. }
            if message.contains("coordinated copy failure")
        ));
        assert_eq!(
            scheduler_probe.submit_calls(),
            2,
            "producer should submit second task before queue disconnect is observed"
        );
        assert_eq!(
            registry.released_ids(),
            planned.chunk_ids,
            "queue disconnect path should roll back all reserved chunk ids"
        );
    }

    #[test]
    fn submit_failure_rolls_back_reserved_chunk_ids() {
        let planned = planned_two_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::FailOnSubmit,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("submit failure should be surfaced");
        assert!(matches!(err, SyncWriteError::IoFailed { .. }));
        assert_eq!(registry.released_ids(), planned.chunk_ids);
    }

    #[test]
    fn submit_failure_cleans_scheduler_batch_accounting() {
        let planned = planned_two_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(AccountingLeakTrackingScheduler::new(2));
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("forced submit failure should surface");
        assert!(matches!(err, SyncWriteError::IoFailed { .. }));
        assert_eq!(
            scheduler_probe.submitted_for_batch(planned.batch_id),
            0,
            "failed submit path should clear scheduler batch accounting",
        );
    }

    #[test]
    fn wait_failure_keeps_dispatched_chunk_ids_reserved() {
        let planned = planned_two_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::FailOnWait,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("wait failure should be surfaced");
        assert!(matches!(err, SyncWriteError::IoFailed { .. }));
        assert_eq!(
            registry.released_ids(),
            Vec::<ChunkId>::new(),
            "wait barrier failure should not release ids that already reached async flush"
        );
    }

    #[test]
    fn submit_partial_failure_does_not_release_ids_already_dispatched_to_async_flush() {
        let planned = planned_two_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry =
            std::sync::Arc::new(crate::runtime::registry::InMemoryChunkKeyRegistry::new());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(FailOnPayloadPrefixCopyEngine {
            fail_first_byte: 5,
            fail_delay: Duration::from_millis(30),
        });
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(CountingBlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
            started: AtomicUsize::new(0),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer.clone(),
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        let err = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect_err("partial batch failure should be surfaced");
        assert!(matches!(err, SyncWriteError::CopyFailed { .. }));

        let flush_start_deadline = Instant::now() + Duration::from_millis(500);
        while writer.started_count() == 0 {
            if Instant::now() >= flush_start_deadline {
                panic!("expected at least one async flush to start before reserve check");
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        let reserve_flushing_id = registry.reserve_many_ids(&[planned.chunk_ids[0]]);
        assert!(
            matches!(
                reserve_flushing_id,
                Err(SyncWriteError::ChunkKeyConflict { chunk_id })
                if chunk_id == planned.chunk_ids[0]
            ),
            "chunk id already dispatched to async flush must remain reserved until async write settles"
        );

        registry
            .reserve_many_ids(&[planned.chunk_ids[1]])
            .expect("chunk id that never reached async flush should be released for retry");
        registry.release_failed_id(&planned.chunk_ids[1]);

        {
            let (lock, condvar) = &*gate;
            let mut open = lock.lock().expect("counting gate lock poisoned");
            *open = true;
            condvar.notify_all();
        }
        coordinator
            .close(1.0, None)
            .expect("close should join async flush workers after gate release");
    }

    #[test]
    fn coordinator_close_documents_precondition_for_timeout_validation() {
        let source = include_str!("coordinator.rs");
        assert!(
            source.contains("Callers must validate `timeout_seconds` before calling"),
            "coordinator::close() must document that callers validate timeout"
        );
        assert!(
            source.contains("debug_assert!(")
                && source.contains("caller must validate timeout_seconds"),
            "coordinator::close() must include a debug_assert for timeout precondition"
        );
    }

    #[test]
    fn close_surfaces_deferred_write_failures() {
        struct FailingChunkWriter;

        impl ChunkWriter for FailingChunkWriter {
            fn write_chunk_by_id(
                &self,
                _array_name: &str,
                _chunk_id: &ChunkId,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                Err(SyncWriteError::io_failed("write failed"))
            }

            fn write_chunk_by_tuple_key(
                &self,
                _array_name: &str,
                _tuple_key: &TupleChunkKey,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                Err(SyncWriteError::io_failed("write failed"))
            }
        }

        let planned = planned_two_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(FailingChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("write should pass copy barrier");

        let err = coordinator
            .close(300.0, None)
            .expect_err("close should surface deferred write errors");
        assert!(matches!(
            err,
            SyncWriteError::DeferredWriteFailures { ref failures } if failures.len() == planned.chunk_ids.len()
        ));
        assert_eq!(registry.committed_ids(), Vec::<ChunkId>::new());
    }

    #[test]
    fn async_write_panic_releases_reserved_chunk_id_and_surfaces_deferred_failure() {
        struct PanicChunkWriter;

        impl ChunkWriter for PanicChunkWriter {
            fn write_chunk_by_id(
                &self,
                _array_name: &str,
                _chunk_id: &ChunkId,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                panic!("panic writer should only be called through tuple-key path");
            }

            fn write_chunk_by_tuple_key(
                &self,
                _array_name: &str,
                _tuple_key: &TupleChunkKey,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                panic!("simulated async write worker panic");
            }
        }

        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry =
            std::sync::Arc::new(crate::runtime::registry::InMemoryChunkKeyRegistry::new());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(PanicChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("copy barrier should still succeed before async panic is reported");

        let err = coordinator
            .close(300.0, None)
            .expect_err("close should surface panic as deferred write failure");
        let failures = match err {
            SyncWriteError::DeferredWriteFailures { failures } => failures,
            other => panic!("expected DeferredWriteFailures, got {other:?}"),
        };
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].chunk_id, Some(planned.chunk_ids[0]));
        assert!(
            failures[0].message.contains("panicked"),
            "panic message should be preserved in deferred error payload"
        );

        registry
            .reserve_many_ids(&planned.chunk_ids)
            .expect("panic recovery should release reserved chunk ids");
        for chunk_id in &planned.chunk_ids {
            registry.release_failed_id(chunk_id);
        }
    }

    #[test]
    fn async_write_panics_after_catch_still_releases_lease() {
        struct PanicChunkWriter;

        impl ChunkWriter for PanicChunkWriter {
            fn write_chunk_by_id(
                &self,
                _array_name: &str,
                _chunk_id: &ChunkId,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                panic!("panic writer should only be called through tuple-key path");
            }

            fn write_chunk_by_tuple_key(
                &self,
                _array_name: &str,
                _tuple_key: &TupleChunkKey,
                _bytes: &[u8],
            ) -> Result<(), SyncWriteError> {
                panic!("simulated async write worker panic");
            }
        }

        struct PanicOnReleaseRegistry;

        impl ChunkKeyRegistry for PanicOnReleaseRegistry {
            fn reserve_many_ids(&self, _chunk_ids: &[ChunkId]) -> Result<(), SyncWriteError> {
                Ok(())
            }

            fn mark_committed_id(&self, _chunk_id: &ChunkId) {}

            fn release_failed_id(&self, _chunk_id: &ChunkId) {
                panic!("simulated registry panic after catch_unwind");
            }
        }

        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(PanicOnReleaseRegistry);
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(TrackingBufferPool::default());
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(PanicChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::try_new_with_pool_builders_for_test(
            WriteCoordinatorComponents {
                planner,
                chunk_registry: registry,
                scheduler,
                buffer_pool: buffer_pool.clone(),
                copy_engine,
                chunk_writer: writer,
                metadata_consolidator: metadata,
                layout_adapter: layout,
                parallel_coord_names: Vec::new(),
                queue_capacity: 4,
            },
            1,
            |num_threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|idx| format!("e2s-zarr-copy-test-{idx}"))
                    .build()
                    .map(std::sync::Arc::new)
                    .map_err(|err| SyncWriteError::ContractViolation {
                        message: format!("failed to build test copy thread pool: {err}"),
                    })
            },
            |num_threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|idx| format!("e2s-zarr-flush-test-{idx}"))
                    // Keep this test deterministic: panic in detached flush workers must not
                    // abort the whole test process.
                    .panic_handler(|_| {})
                    .build()
                    .map(std::sync::Arc::new)
                    .map_err(|err| SyncWriteError::ContractViolation {
                        message: format!("failed to build test flush thread pool: {err}"),
                    })
            },
        )
        .expect("coordinator should construct with test panic handler");

        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("copy barrier should complete before async panic path settles");
        let _ = coordinator.close(0.5, None);
        assert_eq!(
            buffer_pool.releases(),
            1,
            "lease should still be released even when post-catch registry logic panics"
        );
    }

    #[test]
    fn close_times_out_when_async_writes_remain_inflight() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(BlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit should pass copy barrier before async write completion");
        assert_eq!(ack.copied_tasks, planned.tasks.len());

        let err = coordinator
            .close(0.05, None)
            .expect_err("close should timeout while async write remains inflight");
        assert!(matches!(
            err,
            SyncWriteError::LeaseReturnTimeout { outstanding_leases } if outstanding_leases >= 1
        ));

        {
            let (lock, condvar) = &*gate;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    #[test]
    fn backend_configured_close_timeout_can_trigger_timeout_with_inflight_async_write() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(BlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));
        let array_registry: std::sync::Arc<dyn crate::core::contracts::ArrayRegistry> =
            std::sync::Arc::new(crate::runtime::array_registry::InMemoryArrayRegistry::new());
        let backend = crate::backend::SyncZarrBackend::new_with_close_timeout(
            std::sync::Arc::clone(&coordinator),
            array_registry,
            0.05,
        );

        let ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit should pass copy barrier before async write completion");
        assert_eq!(ack.copied_tasks, planned.tasks.len());

        let err = backend
            .close_with_configured_timeout()
            .expect_err("configured short close timeout should fail while async write is inflight");
        assert!(matches!(
            err,
            SyncWriteError::LeaseReturnTimeout { outstanding_leases } if outstanding_leases >= 1
        ));

        {
            let (lock, condvar) = &*gate;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    #[test]
    fn backend_configured_close_timeout_allows_close_after_async_write_finishes() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(BlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));
        let array_registry: std::sync::Arc<dyn crate::core::contracts::ArrayRegistry> =
            std::sync::Arc::new(crate::runtime::array_registry::InMemoryArrayRegistry::new());
        let backend = crate::backend::SyncZarrBackend::new_with_close_timeout(
            std::sync::Arc::clone(&coordinator),
            array_registry,
            1.0,
        );

        let ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit should pass copy barrier before async write completion");
        assert_eq!(ack.copied_tasks, planned.tasks.len());

        let gate_for_release = std::sync::Arc::clone(&gate);
        let release_handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let (lock, condvar) = &*gate_for_release;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        });

        backend
            .close_with_configured_timeout()
            .expect("configured close timeout should allow close after async writer unblocks");
        release_handle
            .join()
            .expect("gate release helper thread should join");
    }

    #[test]
    fn close_runs_teardown_when_metadata_consolidation_fails() {
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned_single_task_batch(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(CloseTrackingScheduler::new(false, false));
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let buffer_pool =
            std::sync::Arc::new(CloseTrackingBufferPool::new(LeaseWaitMode::AlwaysOk, false));
        let buffer_pool_probe = std::sync::Arc::clone(&buffer_pool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(FailingMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .close(1.0, None)
            .expect_err("close should surface metadata consolidation failure");
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { .. }
        ));
        assert_eq!(
            buffer_pool_probe.shutdown_calls(),
            1,
            "buffer pool teardown should still run on metadata failure"
        );
        assert_eq!(
            scheduler_probe.shutdown_calls(),
            1,
            "scheduler teardown should still run on metadata failure"
        );
    }

    #[test]
    fn close_runs_teardown_when_drain_fails() {
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned_single_task_batch(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(CloseTrackingScheduler::new(true, false));
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let buffer_pool =
            std::sync::Arc::new(CloseTrackingBufferPool::new(LeaseWaitMode::AlwaysOk, false));
        let buffer_pool_probe = std::sync::Arc::clone(&buffer_pool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .close(1.0, None)
            .expect_err("close should surface drain failure");
        assert!(
            matches!(err, SyncWriteError::IoFailed { ref message, .. } if message.contains("drain failed"))
        );
        assert_eq!(
            buffer_pool_probe.shutdown_calls(),
            1,
            "buffer pool teardown should run even when drain fails"
        );
        assert_eq!(
            scheduler_probe.shutdown_calls(),
            1,
            "scheduler teardown should run even when drain fails"
        );
    }

    #[test]
    fn close_reports_primary_and_teardown_failures_together() {
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned_single_task_batch(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(CloseTrackingScheduler::new(false, true));
        let buffer_pool =
            std::sync::Arc::new(CloseTrackingBufferPool::new(LeaseWaitMode::Timeout, true));
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let err = coordinator
            .close(1.0, None)
            .expect_err("close should report both primary and teardown failures");
        assert!(matches!(
            err,
            SyncWriteError::ContractViolation { ref message }
            if message.contains("close stage failed")
                && message.contains("close timed out waiting for lease return")
                && message.contains("teardown also failed")
        ));
    }

    #[test]
    fn close_does_not_defer_cuda_source_if_copy_barrier_succeeded() {
        let planned = planned_single_cuda_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner { planned });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(LeaseOverwritingCopyEngine {
            staged_payload: vec![7, 8, 9, 10],
        });
        let writer = std::sync::Arc::new(RecordingPayloadWriter::default());
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer.clone(),
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });
        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("copy barrier should succeed for CUDA-source task");
        coordinator
            .close(300.0, None)
            .expect("close should succeed when CUDA-source task already staged into lease bytes");
        assert_eq!(
            writer.payloads(),
            vec![vec![7, 8, 9, 10]],
            "async writer should persist staged payload for CUDA-source task"
        );
        assert_eq!(
            registry.committed_ids(),
            vec![ChunkId::new(1, 12)],
            "chunk id should transition to committed after successful async write"
        );
    }

    #[test]
    fn submit_write_returns_before_async_chunk_writes_finish() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(BlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry.clone(),
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));
        let coordinator_for_thread = std::sync::Arc::clone(&coordinator);
        let (tx, rx) = mpsc::channel();
        let submit_handle = std::thread::spawn(move || {
            let result =
                coordinator_for_thread.submit_write(&empty_request(), &[], &CoordMap::new());
            tx.send(result)
                .expect("submit result channel send should succeed");
        });

        let submit_result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("submit_write should not block on async chunk write completion");
        let ack = submit_result.expect("submit_write should succeed");
        assert_eq!(ack.copied_tasks, planned.tasks.len());
        assert_eq!(registry.committed_ids(), Vec::<ChunkId>::new());

        {
            let (lock, condvar) = &*gate;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        }
        submit_handle
            .join()
            .expect("submit worker thread should join");
        coordinator
            .close(300.0, None)
            .expect("close should join async chunk writers");
    }

    #[test]
    fn lease_release_happens_after_async_write_completion() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(TrackingBufferPool::default());
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(BlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
        });
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool: buffer_pool.clone(),
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));
        let coordinator_for_thread = std::sync::Arc::clone(&coordinator);
        let (tx, rx) = mpsc::channel();
        let submit_handle = std::thread::spawn(move || {
            let result =
                coordinator_for_thread.submit_write(&empty_request(), &[], &CoordMap::new());
            tx.send(result)
                .expect("submit result channel send should succeed");
        });

        let submit_result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("submit_write should not block on async chunk write completion");
        submit_result.expect("submit_write should succeed");
        assert_eq!(
            buffer_pool.releases(),
            0,
            "lease should not be released before async writer finishes"
        );

        {
            let (lock, condvar) = &*gate;
            let mut ready = lock.lock().expect("blocking gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        }
        submit_handle
            .join()
            .expect("submit worker thread should join");
        coordinator
            .close(300.0, None)
            .expect("close should join async chunk writers");
        assert_eq!(
            buffer_pool.releases(),
            1,
            "lease should release after write completion"
        );
    }

    #[test]
    fn ordering_contract_comments_are_present_for_close_and_async_flush() {
        let source = include_str!("coordinator.rs");
        let production_only = source
            .split("\n#[cfg(test)]")
            .next()
            .expect("coordinator.rs should contain production module before tests");
        assert!(
            production_only.contains(
                "// ORDERING CONTRACT (close): wait_pooled_leases_returned must run before wait_for_async_writes."
            ),
            "close() must document why lease wait precedes inflight async-write wait"
        );
        assert!(
            production_only.contains(
                "// ORDERING CONTRACT (async flush): release buffer lease before inflight guard drops."
            ),
            "spawn_async_write_handle() must document lease-release-before-guard-drop contract"
        );
    }

    #[test]
    fn submit_write_impl_uses_explicit_imports_not_super_glob() {
        let submit_source = include_str!("coordinator_submit.rs");
        assert!(
            !submit_source.contains("use super::*;"),
            "coordinator_submit.rs should use explicit imports instead of use super::*"
        );
    }

    #[test]
    fn write_coordinator_types_use_crate_visibility_not_hidden_public_semver_surface() {
        let source = include_str!("coordinator.rs");
        let production_only = source
            .split("\n#[cfg(test)]")
            .next()
            .expect("coordinator.rs should contain production module before tests");
        assert!(
            production_only.contains("pub(crate) struct WriteCoordinatorComponents"),
            "WriteCoordinatorComponents should be crate-visible, not public API"
        );
        assert!(
            production_only.contains("pub(crate) struct WriteCoordinator"),
            "WriteCoordinator should be crate-visible, not public API"
        );
    }

    #[test]
    fn async_write_uses_staged_lease_bytes_not_original_input_payload() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner { planned });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(LeaseOverwritingCopyEngine {
            staged_payload: vec![9, 8, 7, 6],
        });
        let writer = std::sync::Arc::new(RecordingPayloadWriter::default());
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer.clone(),
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit_write should succeed");
        coordinator
            .close(300.0, None)
            .expect("close should flush staged payload writes");

        assert_eq!(
            writer.payloads(),
            vec![vec![9, 8, 7, 6]],
            "writer should receive staged bytes copied into lease memory"
        );
    }

    #[test]
    fn submit_write_hot_path_does_not_schedule_periodic_consolidation() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner { planned });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(SlowCountingMetadataConsolidator::new(
            Duration::from_millis(200),
        ));
        let metadata_probe = std::sync::Arc::clone(&metadata);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit_write should succeed");
        assert!(
            metadata_probe.calls() == 0,
            "submit_write hot path should not trigger background metadata consolidation"
        );

        coordinator
            .close(300.0, None)
            .expect("close should succeed");
        assert_eq!(
            metadata_probe.calls(),
            1,
            "metadata consolidation should run once at close"
        );
    }

    #[test]
    fn copy_and_flush_execute_on_dedicated_rayon_pools() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner { planned });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(ThreadNameRecordingCopyEngine::default());
        let chunk_writer = std::sync::Arc::new(ThreadNameRecordingChunkWriter::default());
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine: copy_engine.clone(),
            chunk_writer: chunk_writer.clone(),
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit_write should succeed");
        coordinator
            .close(300.0, None)
            .expect("close should drain async writes");

        let copy_thread_names = copy_engine.thread_names();
        assert!(
            !copy_thread_names.is_empty(),
            "copy stage should record at least one worker thread"
        );
        assert!(
            copy_thread_names
                .iter()
                .all(|name| name.starts_with("e2s-zarr-copy-")),
            "copy stage should execute on dedicated copy pool threads: {:?}",
            copy_thread_names
        );

        let flush_thread_names = chunk_writer.thread_names();
        assert!(
            !flush_thread_names.is_empty(),
            "flush stage should record at least one worker thread"
        );
        assert!(
            flush_thread_names
                .iter()
                .all(|name| name.starts_with("e2s-zarr-flush-")),
            "flush stage should execute on dedicated flush pool threads: {:?}",
            flush_thread_names
        );
    }

    #[test]
    fn submit_write_records_internal_timing_breakdown() {
        let planned = planned_single_task_batch();
        let batch_id = planned.batch_id;
        let planner = std::sync::Arc::new(FixedPlanner { planned });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let chunk_writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        let ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit_write should succeed");
        coordinator
            .close(300.0, None)
            .expect("close should drain async writes");

        let timing = coordinator
            .last_write_timing()
            .expect("timing snapshot should be available after write");
        assert_eq!(timing.batch_id, batch_id);
        assert_eq!(timing.task_count, ack.copied_tasks);
        assert_eq!(timing.enqueued_task_count, ack.copied_tasks);
        assert_eq!(timing.copied_task_count, ack.copied_tasks);
        assert!(timing.worker_count >= 1);
    }

    #[test]
    fn queue_capacity_scales_from_first_write_to_cover_two_step_burst() {
        let planned = planned_multi_task_batch(26);
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(TrackingBarrierScheduler::default());
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 16,
        });
        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("first write should succeed");

        assert_eq!(
            coordinator.debug_effective_queue_capacity(),
            Some(planned.tasks.len() * 2),
            "first write should freeze queue capacity to at least a 2-step burst"
        );
        coordinator
            .close(300.0, None)
            .expect("close should succeed");
    }

    #[test]
    fn bounded_admission_prevents_full_second_batch_enqueue_when_workers_are_blocked() {
        let first_batch = planned_multi_task_batch(2);
        let second_batch = planned_multi_task_batch(26);
        let planner = std::sync::Arc::new(SequencedPlanner::new(vec![
            first_batch.clone(),
            second_batch.clone(),
        ]));
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(TrackingBarrierScheduler::default());
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(GateControlledCopyEngine::new(true));
        let copy_probe = std::sync::Arc::clone(&copy_engine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));

        let first_ack = coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("first write should succeed");
        assert_eq!(first_ack.copied_tasks, first_batch.tasks.len());
        assert_eq!(
            coordinator.debug_effective_queue_capacity(),
            Some(4),
            "first write should freeze queue capacity at configured minimum for small batches"
        );

        let baseline_submitted = scheduler_probe.submitted_tasks();
        let baseline_started = copy_probe.started_count();
        copy_probe.set_open(false);

        let coordinator_for_thread = std::sync::Arc::clone(&coordinator);
        let (tx, rx) = mpsc::channel();
        let submit_handle = std::thread::spawn(move || {
            let result =
                coordinator_for_thread.submit_write(&empty_request(), &[], &CoordMap::new());
            tx.send(result)
                .expect("second submit result channel send should succeed");
        });

        let start_deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < start_deadline {
            let submitted_delta = scheduler_probe
                .submitted_tasks()
                .saturating_sub(baseline_submitted);
            let started_delta = copy_probe.started_count().saturating_sub(baseline_started);
            if submitted_delta > 0 && started_delta > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        let mut reached_full_enqueue = false;
        let enqueue_deadline = Instant::now() + Duration::from_millis(250);
        while Instant::now() < enqueue_deadline {
            let submitted_delta = scheduler_probe
                .submitted_tasks()
                .saturating_sub(baseline_submitted);
            if submitted_delta >= second_batch.tasks.len() {
                reached_full_enqueue = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "second write should still be blocked while copy workers are gated"
        );
        assert!(
            !reached_full_enqueue,
            "bounded admission should prevent enqueuing every task while workers are blocked"
        );

        copy_probe.set_open(true);
        let second_result = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second write should complete once gate opens");
        let second_ack = second_result.expect("second write should succeed");
        assert_eq!(second_ack.copied_tasks, second_batch.tasks.len());
        submit_handle.join().expect("submit thread should join");

        coordinator
            .close(300.0, None)
            .expect("close should succeed");
    }

    #[test]
    fn async_write_starts_while_copy_barrier_is_still_in_progress() {
        let planned = planned_multi_task_batch(26);
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(TrackingBarrierScheduler::default());
        let scheduler_probe = std::sync::Arc::clone(&scheduler);
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(SlowCopyEngine {
            delay: Duration::from_millis(20),
        });
        let gate = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        let writer = std::sync::Arc::new(CountingBlockingChunkWriter {
            gate: std::sync::Arc::clone(&gate),
            started: AtomicUsize::new(0),
        });
        let writer_probe = std::sync::Arc::clone(&writer);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = std::sync::Arc::new(WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler: scheduler.clone(),
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        }));
        let coordinator_for_thread = std::sync::Arc::clone(&coordinator);
        let (tx, rx) = mpsc::channel();
        let submit_handle = std::thread::spawn(move || {
            let result =
                coordinator_for_thread.submit_write(&empty_request(), &[], &CoordMap::new());
            tx.send(result)
                .expect("submit result channel send should succeed");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut observed_overlap = false;
        while Instant::now() < deadline {
            let started = writer_probe.started_count();
            let copied = scheduler_probe.copied_tasks();
            let submitted = scheduler_probe.submitted_tasks();
            if submitted > 0 && started > 0 && copied < submitted {
                observed_overlap = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        {
            let (lock, condvar) = &*gate;
            let mut ready = lock.lock().expect("counting gate lock poisoned");
            *ready = true;
            condvar.notify_all();
        }

        let submit_result = rx
            .recv_timeout(Duration::from_secs(4))
            .expect("submit_write should complete");
        let ack = submit_result.expect("submit_write should succeed");
        assert_eq!(ack.copied_tasks, planned.tasks.len());
        assert!(
            observed_overlap,
            "expected async chunk writes to start while copy barrier was still in progress"
        );

        submit_handle.join().expect("submit thread should join");
        coordinator
            .close(300.0, None)
            .expect("close should join async chunk writers");
    }

    #[test]
    fn close_report_includes_timing_breakdown() {
        let planned = planned_single_task_batch();
        let planner = std::sync::Arc::new(FixedPlanner {
            planned: planned.clone(),
        });
        let registry = std::sync::Arc::new(RecordingRegistry::default());
        let scheduler = std::sync::Arc::new(ControlledScheduler {
            mode: SchedulerMode::AlwaysOk,
        });
        let buffer_pool = std::sync::Arc::new(StubBufferPool);
        let copy_engine = std::sync::Arc::new(StubCopyEngine);
        let writer = std::sync::Arc::new(StubChunkWriter);
        let metadata = std::sync::Arc::new(StubMetadataConsolidator);
        let layout = std::sync::Arc::new(StubLayoutAdapter);

        let coordinator = WriteCoordinator::new(WriteCoordinatorComponents {
            planner,
            chunk_registry: registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer: writer,
            metadata_consolidator: metadata,
            layout_adapter: layout,
            parallel_coord_names: Vec::new(),
            queue_capacity: 4,
        });

        coordinator
            .submit_write(&empty_request(), &[], &CoordMap::new())
            .expect("submit should succeed");
        let report = coordinator
            .close(300.0, None)
            .expect("close should succeed");

        let timing = report
            .close_timing
            .expect("close_timing should be populated on success");
        assert!(
            timing.total_close_ns.as_nanos() > 0,
            "total_close_ns must be non-zero"
        );
        // Phases must sum to at most total (allow scheduling slack)
        let phase_sum = timing.async_drain_ns.as_nanos()
            + timing.consolidate_ns.as_nanos()
            + timing.teardown_ns.as_nanos();
        assert!(
            phase_sum <= timing.total_close_ns.as_nanos() + 1_000_000, // 1ms slack
            "phases ({phase_sum}ns) should not exceed total ({}ns) by >1ms",
            timing.total_close_ns.as_nanos()
        );
    }
}
