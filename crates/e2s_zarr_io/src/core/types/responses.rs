/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Write acknowledgement and close-report types.

use std::time::Duration;

use crate::core::errors::DeferredWriteError;

use super::pool_config::PoolWarmupStatus;

/// Type-safe write-batch identifier.
///
/// Prevents accidental mixing with other `u64` values (e.g., linear chunk indices,
/// nanosecond durations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchId(pub(crate) u64);

impl BatchId {
    /// Raw numeric identifier.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for BatchId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<BatchId> for u64 {
    fn from(id: BatchId) -> Self {
        id.0
    }
}

impl std::fmt::Display for BatchId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchId({})", self.0)
    }
}

/// Type-safe wrapper for nanosecond durations in internal timing fields.
///
/// Prevents accidental mixing with raw `u64` batch IDs or other counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Nanoseconds(pub(crate) u64);

impl Nanoseconds {
    /// Raw nanosecond value.
    #[must_use]
    pub fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Convert to [`Duration`].
    #[must_use]
    pub fn to_duration(self) -> Duration {
        Duration::from_nanos(self.0)
    }
}

impl From<u64> for Nanoseconds {
    fn from(nanos: u64) -> Self {
        Self(nanos)
    }
}

impl From<Nanoseconds> for u64 {
    fn from(ns: Nanoseconds) -> Self {
        ns.0
    }
}

/// Acknowledgment returned by `write()` after the copy barrier is satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriteCopyAck {
    /// Identifier for this write batch.
    pub batch_id: BatchId,
    /// Number of tasks whose data was copied into Rust-owned buffers.
    pub copied_tasks: usize,
}

/// Internal per-write timing breakdown captured by the coordinator.
///
/// All duration fields use [`Nanoseconds`] wrappers representing:
/// - top-level wall-clock phases in `submit_write()`,
/// - aggregated worker phase durations across all tasks in the batch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriteInternalTiming {
    /// Identifier for this write batch.
    pub batch_id: BatchId,
    /// Planned task count in this batch.
    pub task_count: usize,
    /// Number of copy workers used for this batch.
    pub worker_count: usize,
    /// Number of tasks accepted by producer enqueue loop.
    pub enqueued_task_count: usize,
    /// Number of tasks that reached copied-marked state.
    pub copied_task_count: usize,
    /// Planner phase time.
    pub plan_ns: Nanoseconds,
    /// Buffer-pool lazy initialization check time.
    pub buffer_init_ns: Nanoseconds,
    /// Chunk-id reservation time.
    pub reserve_ns: Nanoseconds,
    /// Total time spent in scheduler `submit()` calls.
    pub scheduler_submit_ns: Nanoseconds,
    /// Total time spent in bounded-queue `send()` calls.
    pub queue_send_ns: Nanoseconds,
    /// Time waiting for copy barrier completion (`wait_copied`).
    pub barrier_wait_ns: Nanoseconds,
    /// Aggregated worker lease-acquire time.
    pub worker_acquire_ns: Nanoseconds,
    /// Aggregated worker copy-into-lease time.
    pub worker_copy_ns: Nanoseconds,
    /// Aggregated worker copy-completion-wait time.
    pub worker_wait_copy_ns: Nanoseconds,
    /// Aggregated worker scheduler `mark_copied()` time.
    pub worker_mark_copied_ns: Nanoseconds,
    /// Aggregated worker async-flush enqueue time.
    pub worker_enqueue_flush_ns: Nanoseconds,
    /// End-to-end `submit_write()` wall time for this batch.
    pub total_submit_write_ns: Nanoseconds,
}

/// Report returned by `close()` summarizing the backend's final state.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CloseReport {
    /// Total write batches processed.
    pub write_batches_seen: usize,
    /// Write failures collected from prior `write()` calls (post-copy-barrier).
    pub deferred_write_errors: Vec<DeferredWriteError>,
    /// Whether metadata consolidation completed successfully.
    pub metadata_consolidated: bool,
    /// Whether pool and runtime resources were fully released.
    pub resources_released: bool,
    /// Final observed hot/warm pool warmup status during close orchestration.
    pub pool_warmup_status: PoolWarmupStatus,
    /// Whether the backend is now in terminal `Closed` state.
    pub closed: bool,
    /// Internal timing breakdown for close phases (nanoseconds).
    pub close_timing: Option<CloseInternalTiming>,
}

/// Nanosecond-resolution timing for each phase of `close()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloseInternalTiming {
    /// Time waiting for async flush workers to complete (lease return + inflight drain).
    pub async_drain_ns: Nanoseconds,
    /// Time spent in metadata consolidation (per-array metadata + .zmetadata).
    pub consolidate_ns: Nanoseconds,
    /// Time spent tearing down runtime resources (pool shutdown, scheduler shutdown).
    pub teardown_ns: Nanoseconds,
    /// Total close wall-clock time.
    pub total_close_ns: Nanoseconds,
}
