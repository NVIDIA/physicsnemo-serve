/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helper types used by coordinator write workers.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::core::chunk_id::ChunkId;
use crate::core::types::{BatchId, BufferLease, TupleChunkKey};

/// Async flush task descriptor passed from copy workers to flush workers.
#[derive(Debug)]
pub(crate) struct PendingChunkWrite {
    pub(crate) batch_id: BatchId,
    pub(crate) array_name: String,
    pub(crate) chunk_id: ChunkId,
    pub(crate) tuple_key: TupleChunkKey,
    pub(crate) required_bytes: usize,
    pub(crate) lease: BufferLease,
}

/// Aggregated worker timing counters for the last submitted batch.
#[derive(Default)]
pub(crate) struct WorkerTimingAccumulator {
    acquire_ns: AtomicU64,
    copy_ns: AtomicU64,
    wait_copy_ns: AtomicU64,
    mark_copied_ns: AtomicU64,
    enqueue_flush_ns: AtomicU64,
    copied_task_count: AtomicUsize,
}

/// Snapshot of accumulated worker timings.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct WorkerTimingSnapshot {
    pub(crate) acquire_ns: u64,
    pub(crate) copy_ns: u64,
    pub(crate) wait_copy_ns: u64,
    pub(crate) mark_copied_ns: u64,
    pub(crate) enqueue_flush_ns: u64,
    pub(crate) copied_task_count: usize,
}

impl WorkerTimingAccumulator {
    pub(crate) fn add_snapshot(&self, snapshot: &WorkerTimingSnapshot) {
        self.acquire_ns
            .fetch_add(snapshot.acquire_ns, Ordering::Relaxed);
        self.copy_ns.fetch_add(snapshot.copy_ns, Ordering::Relaxed);
        self.wait_copy_ns
            .fetch_add(snapshot.wait_copy_ns, Ordering::Relaxed);
        self.mark_copied_ns
            .fetch_add(snapshot.mark_copied_ns, Ordering::Relaxed);
        self.enqueue_flush_ns
            .fetch_add(snapshot.enqueue_flush_ns, Ordering::Relaxed);
        self.copied_task_count
            .fetch_add(snapshot.copied_task_count, Ordering::Relaxed);
    }

    /// Returns a snapshot of accumulated worker timings.
    ///
    /// # Ordering
    /// All loads use `Ordering::Relaxed`. This is safe when called only after a
    /// synchronization point (for example, `rayon::scope()` return) that
    /// establishes a happens-before relationship with all worker `fetch_add`
    /// updates in `add_snapshot()`.
    pub(crate) fn snapshot(&self) -> WorkerTimingSnapshot {
        WorkerTimingSnapshot {
            acquire_ns: self.acquire_ns.load(Ordering::Relaxed),
            copy_ns: self.copy_ns.load(Ordering::Relaxed),
            wait_copy_ns: self.wait_copy_ns.load(Ordering::Relaxed),
            mark_copied_ns: self.mark_copied_ns.load(Ordering::Relaxed),
            enqueue_flush_ns: self.enqueue_flush_ns.load(Ordering::Relaxed),
            copied_task_count: self.copied_task_count.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkerTimingAccumulator, WorkerTimingSnapshot};

    #[test]
    fn add_snapshot_accumulates_all_worker_timing_fields() {
        let accumulator = WorkerTimingAccumulator::default();
        let snapshot = WorkerTimingSnapshot {
            acquire_ns: 11,
            copy_ns: 22,
            wait_copy_ns: 33,
            mark_copied_ns: 44,
            enqueue_flush_ns: 55,
            copied_task_count: 3,
        };
        accumulator.add_snapshot(&snapshot);

        let merged = accumulator.snapshot();
        assert_eq!(
            merged, snapshot,
            "accumulator snapshot should equal single added snapshot"
        );
    }

    #[test]
    fn relaxed_ordering_contract_is_documented_for_snapshot_and_call_site() {
        let write_task_source = include_str!("write_task.rs");
        assert!(
            write_task_source.contains("# Ordering")
                && write_task_source.contains("Ordering::Relaxed")
                && write_task_source.contains("rayon::scope()"),
            "WorkerTimingAccumulator::snapshot() should document the Relaxed ordering precondition"
        );

        let submit_source = include_str!("coordinator_submit.rs");
        assert!(
            submit_source
                .contains("// ORDERING: scope() return provides happens-before with all worker"),
            "coordinator_submit.rs should document why Relaxed snapshot loads are safe after scope()"
        );
    }
}
