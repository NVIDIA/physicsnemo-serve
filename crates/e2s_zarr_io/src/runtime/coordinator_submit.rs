/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `WriteCoordinator::submit_write` implementation.
//!
//! This module contains the implementation of the `WriteCoordinator::submit_write`
//! method. It is responsible for submitting a write request through the full
//! pipeline. It is split from `coordinator.rs` to keep the orchestration hub
//! focused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{panic, panic::AssertUnwindSafe};

use crossbeam_channel::{RecvError as ChannelRecvError, SendError as ChannelSendError, bounded};

use super::WriteCoordinator;
use crate::core::chunk_id::ChunkId;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ChunkTask, CoordMap, FirstWriteSizingHint, InferenceWriteRequest, Nanoseconds, WriteCopyAck,
    WriteInternalTiming,
};
use crate::runtime::write_task::{
    PendingChunkWrite, WorkerTimingAccumulator, WorkerTimingSnapshot,
};

impl WriteCoordinator {
    /// Submit a write request through the full pipeline.
    ///
    /// Steps: plan -> reserve chunk IDs -> submit tasks -> wait copy barrier -> ack.
    /// On failure, reserved chunk IDs are rolled back.
    pub fn submit_write(
        &self,
        req: &InferenceWriteRequest,
        array_ids: &[u32],
        registered_coords: &CoordMap,
    ) -> Result<WriteCopyAck, SyncWriteError> {
        self.wait_for_registration_metadata()?;
        let submit_write_started = Instant::now();

        let plan_started = Instant::now();
        let planned = self.planner.plan_batch(req, array_ids, registered_coords)?;
        let plan_ns = Self::elapsed_ns(plan_started);

        let first_write_hint = FirstWriteSizingHint {
            first_write_task_count: planned.tasks.len(),
            first_write_max_required_bytes: planned
                .tasks
                .iter()
                .map(|task| task.required_bytes)
                .max()
                .unwrap_or(0),
        };
        let buffer_init_started = Instant::now();
        self.buffer_pool.initialize_if_needed(&first_write_hint)?;
        let buffer_init_ns = Self::elapsed_ns(buffer_init_started);

        let task_count = planned.tasks.len();
        let reserve_started = Instant::now();
        self.chunk_registry.reserve_many_ids(&planned.chunk_ids)?;
        let reserve_ns = Self::elapsed_ns(reserve_started);

        if task_count == 0 {
            self.write_batches_seen.fetch_add(1, Ordering::Release);
            self.store_last_write_timing(WriteInternalTiming {
                batch_id: planned.batch_id,
                task_count: 0,
                worker_count: 0,
                enqueued_task_count: 0,
                copied_task_count: 0,
                plan_ns,
                buffer_init_ns,
                reserve_ns,
                scheduler_submit_ns: Nanoseconds(0),
                queue_send_ns: Nanoseconds(0),
                barrier_wait_ns: Nanoseconds(0),
                worker_acquire_ns: Nanoseconds(0),
                worker_copy_ns: Nanoseconds(0),
                worker_wait_copy_ns: Nanoseconds(0),
                worker_mark_copied_ns: Nanoseconds(0),
                worker_enqueue_flush_ns: Nanoseconds(0),
                total_submit_write_ns: Self::elapsed_ns(submit_write_started),
            });
            return Ok(WriteCopyAck {
                batch_id: planned.batch_id,
                copied_tasks: 0,
            });
        }

        let queue_capacity = self.resolve_queue_capacity(task_count);
        let (task_tx, task_rx) = bounded::<ChunkTask>(queue_capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let first_error: Arc<Mutex<Option<SyncWriteError>>> = Arc::new(Mutex::new(None));
        let worker_count = task_count.clamp(1, self.num_threads);
        let worker_timing = Arc::new(WorkerTimingAccumulator::default());
        let flush_dispatched_chunk_ids: Arc<Mutex<Vec<ChunkId>>> =
            Arc::new(Mutex::new(Vec::with_capacity(task_count)));
        let mut scheduler_submit_ns = Nanoseconds(0);
        let mut queue_send_ns = Nanoseconds(0);
        let mut enqueued_task_count = 0_usize;

        self.copy_rayon_pool.scope(|scope| {
            for _ in 0..worker_count {
                let task_rx = task_rx.clone();
                let cancelled = Arc::clone(&cancelled);
                let first_error = Arc::clone(&first_error);
                let worker_timing = Arc::clone(&worker_timing);
                let flush_dispatched_chunk_ids = Arc::clone(&flush_dispatched_chunk_ids);
                scope.spawn(move |_| {
                    let mut local_timing = WorkerTimingSnapshot::default();
                    loop {
                        if cancelled.load(Ordering::Acquire) {
                            break;
                        }

                        let task = match task_rx.recv() {
                            Ok(task) => task,
                            Err(ChannelRecvError) => break,
                        };

                        let mut maybe_lease = None;
                        let task_result = panic::catch_unwind(AssertUnwindSafe(
                            || -> Result<(), SyncWriteError> {
                                let acquire_started = Instant::now();
                                let lease = self.buffer_pool.acquire(task.required_bytes)?;
                                local_timing.acquire_ns = local_timing
                                    .acquire_ns
                                    .saturating_add(Self::elapsed_ns(acquire_started).0);
                                maybe_lease = Some(lease);

                                let copy_started = Instant::now();
                                let lease = maybe_lease.as_mut().ok_or_else(|| {
                                    SyncWriteError::ContractViolation {
                                        message: "copy worker lost lease before copy step"
                                            .to_string(),
                                    }
                                })?;
                                let copy_completion = self.copy_engine.copy_into_lease(
                                    &task.input,
                                    lease,
                                    task.required_bytes,
                                )?;
                                local_timing.copy_ns = local_timing
                                    .copy_ns
                                    .saturating_add(Self::elapsed_ns(copy_started).0);

                                let wait_copy_started = Instant::now();
                                self.copy_engine.wait_copy_completion(copy_completion)?;
                                local_timing.wait_copy_ns = local_timing
                                    .wait_copy_ns
                                    .saturating_add(Self::elapsed_ns(wait_copy_started).0);

                                let mark_copied_started = Instant::now();
                                self.scheduler.mark_copied(task.batch_id)?;
                                local_timing.mark_copied_ns = local_timing
                                    .mark_copied_ns
                                    .saturating_add(Self::elapsed_ns(mark_copied_started).0);
                                local_timing.copied_task_count =
                                    local_timing.copied_task_count.saturating_add(1);

                                let pending = PendingChunkWrite {
                                    batch_id: task.batch_id,
                                    array_name: task.array_name,
                                    chunk_id: task.chunk_id,
                                    tuple_key: task.tuple_key,
                                    required_bytes: task.required_bytes,
                                    lease: maybe_lease.take().ok_or_else(|| {
                                        SyncWriteError::ContractViolation {
                                            message: "copy worker lost lease before flush enqueue"
                                                .to_string(),
                                        }
                                    })?,
                                };

                                let enqueue_flush_started = Instant::now();
                                let chunk_id = pending.chunk_id;
                                self.spawn_async_write_handle(pending)?;
                                let mut tracked_ids = match flush_dispatched_chunk_ids.lock() {
                                    Ok(guard) => guard,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                tracked_ids.push(chunk_id);
                                local_timing.enqueue_flush_ns = local_timing
                                    .enqueue_flush_ns
                                    .saturating_add(Self::elapsed_ns(enqueue_flush_started).0);
                                Ok(())
                            },
                        ));

                        match task_result {
                            Ok(Ok(())) => {}
                            Ok(Err(err)) => {
                                if let Some(lease) = maybe_lease.take() {
                                    self.buffer_pool.release(lease);
                                }
                                Self::record_first_error(&first_error, &cancelled, err);
                                break;
                            }
                            Err(_) => {
                                if let Some(lease) = maybe_lease.take() {
                                    self.buffer_pool.release(lease);
                                }
                                Self::record_first_error(
                                    &first_error,
                                    &cancelled,
                                    SyncWriteError::ContractViolation {
                                        message: "copy worker panicked while processing write task"
                                            .to_string(),
                                    },
                                );
                                break;
                            }
                        }
                    }
                    worker_timing.add_snapshot(&local_timing);
                });
            }

            'producer: for task in planned.tasks.iter().cloned() {
                if cancelled.load(Ordering::Acquire) {
                    break;
                }
                let scheduler_submit_started = Instant::now();
                if let Err(err) = self.scheduler.submit(task.clone()) {
                    scheduler_submit_ns.0 = scheduler_submit_ns
                        .0
                        .saturating_add(Self::elapsed_ns(scheduler_submit_started).0);
                    Self::record_first_error(&first_error, &cancelled, err);
                    break;
                }
                scheduler_submit_ns.0 = scheduler_submit_ns
                    .0
                    .saturating_add(Self::elapsed_ns(scheduler_submit_started).0);

                let queue_send_started = Instant::now();
                if let Err(ChannelSendError(_task_back)) = task_tx.send(task) {
                    queue_send_ns.0 = queue_send_ns
                        .0
                        .saturating_add(Self::elapsed_ns(queue_send_started).0);
                    Self::record_first_error(
                        &first_error,
                        &cancelled,
                        SyncWriteError::ContractViolation {
                            message: "task queue disconnected".to_string(),
                        },
                    );
                    break 'producer;
                }
                queue_send_ns.0 = queue_send_ns
                    .0
                    .saturating_add(Self::elapsed_ns(queue_send_started).0);
                enqueued_task_count = enqueued_task_count.saturating_add(1);
            }
            drop(task_tx);
        });

        let maybe_error = match first_error.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => {
                let flushed_ids = Self::snapshot_dispatched_chunk_ids(&flush_dispatched_chunk_ids);
                self.rollback_reserved_ids_not_dispatched(&planned.chunk_ids, &flushed_ids);
                return Err(self.fail_with_abort_batch_cleanup(
                    planned.batch_id,
                    SyncWriteError::ContractViolation {
                        message: "copy worker error state lock poisoned".to_string(),
                    },
                ));
            }
        };
        if let Some(err) = maybe_error {
            let flushed_ids = Self::snapshot_dispatched_chunk_ids(&flush_dispatched_chunk_ids);
            self.rollback_reserved_ids_not_dispatched(&planned.chunk_ids, &flushed_ids);
            return Err(self.fail_with_abort_batch_cleanup(planned.batch_id, err));
        }

        let barrier_wait_started = Instant::now();
        if let Err(err) = self.scheduler.wait_copied(planned.batch_id) {
            let flushed_ids = Self::snapshot_dispatched_chunk_ids(&flush_dispatched_chunk_ids);
            self.rollback_reserved_ids_not_dispatched(&planned.chunk_ids, &flushed_ids);
            return Err(self.fail_with_abort_batch_cleanup(planned.batch_id, err));
        }
        let barrier_wait_ns = Self::elapsed_ns(barrier_wait_started);

        self.write_batches_seen.fetch_add(1, Ordering::Release);
        // ORDERING: scope() return provides happens-before with all worker
        // fetch_add calls, making the Relaxed snapshot below consistent.
        let worker_timing_snapshot = worker_timing.snapshot();
        self.store_last_write_timing(WriteInternalTiming {
            batch_id: planned.batch_id,
            task_count,
            worker_count,
            enqueued_task_count,
            copied_task_count: worker_timing_snapshot.copied_task_count,
            plan_ns,
            buffer_init_ns,
            reserve_ns,
            scheduler_submit_ns,
            queue_send_ns,
            barrier_wait_ns,
            worker_acquire_ns: Nanoseconds(worker_timing_snapshot.acquire_ns),
            worker_copy_ns: Nanoseconds(worker_timing_snapshot.copy_ns),
            worker_wait_copy_ns: Nanoseconds(worker_timing_snapshot.wait_copy_ns),
            worker_mark_copied_ns: Nanoseconds(worker_timing_snapshot.mark_copied_ns),
            worker_enqueue_flush_ns: Nanoseconds(worker_timing_snapshot.enqueue_flush_ns),
            total_submit_write_ns: Self::elapsed_ns(submit_write_started),
        });

        Ok(WriteCopyAck {
            batch_id: planned.batch_id,
            copied_tasks: task_count,
        })
    }
}
