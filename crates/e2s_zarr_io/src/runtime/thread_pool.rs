/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Work scheduler backed by Rayon-ready synchronization semantics.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex};

use crate::core::contracts::WorkScheduler;
use crate::core::errors::SyncWriteError;
use crate::core::types::{BatchId, ChunkTask};

#[derive(Default)]
struct SchedulerState {
    submitted_by_batch: HashMap<BatchId, usize>,
    copied_by_batch: HashMap<BatchId, usize>,
    shutdown: bool,
}

/// Scheduler accounting and copy-barrier coordination for the Rayon worker runtime.
///
/// This component tracks per-batch submission/copy counts and provides a blocking
/// `wait_copied()` barrier used by `write()`.
pub struct RayonWorkScheduler {
    state: Mutex<SchedulerState>,
    copied_cv: Condvar,
}

impl Default for RayonWorkScheduler {
    fn default() -> Self {
        Self {
            state: Mutex::new(SchedulerState::default()),
            copied_cv: Condvar::new(),
        }
    }
}

impl RayonWorkScheduler {
    /// Create a new scheduler instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl WorkScheduler for RayonWorkScheduler {
    fn submit(&self, task: ChunkTask) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        if state.shutdown {
            return Err(SyncWriteError::ObjectClosed);
        }
        state
            .submitted_by_batch
            .entry(task.batch_id)
            .and_modify(|v| *v += 1)
            .or_insert(1);
        Ok(())
    }

    fn mark_copied(&self, batch_id: BatchId) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        let Some(expected) = state.submitted_by_batch.get(&batch_id).copied() else {
            return Err(SyncWriteError::Validation {
                message: format!("unknown batch id: {batch_id}"),
            });
        };
        let copied = state.copied_by_batch.entry(batch_id).or_insert(0);
        *copied += 1;
        if *copied > expected {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "mark_copied exceeded submitted task count for batch {batch_id}: copied={} submitted={expected}",
                    *copied
                ),
            });
        }
        self.copied_cv.notify_all();
        Ok(())
    }

    fn wait_copied(&self, batch_id: BatchId) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        let Some(submitted) = state.submitted_by_batch.get(&batch_id).copied() else {
            return Err(SyncWriteError::Validation {
                message: format!("unknown batch id: {batch_id}"),
            });
        };

        loop {
            let copied = state.copied_by_batch.get(&batch_id).copied().unwrap_or(0);
            if copied == submitted {
                state.submitted_by_batch.remove(&batch_id);
                state.copied_by_batch.remove(&batch_id);
                return Ok(());
            }
            state = self
                .copied_cv
                .wait(state)
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "scheduler wait lock poisoned".to_string(),
                })?;
        }
    }

    fn abort_batch(&self, batch_id: BatchId) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        state.submitted_by_batch.remove(&batch_id);
        state.copied_by_batch.remove(&batch_id);
        self.copied_cv.notify_all();
        Ok(())
    }

    fn drain(&self) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        state.submitted_by_batch.clear();
        state.copied_by_batch.clear();
        self.copied_cv.notify_all();
        Ok(())
    }

    fn shutdown(&self) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "scheduler lock poisoned".to_string(),
            })?;
        state.shutdown = true;
        state.submitted_by_batch.clear();
        state.copied_by_batch.clear();
        self.copied_cv.notify_all();
        Ok(())
    }
}

/// Legacy name retained for compatibility while callsites migrate.
pub type SynchronousWorkScheduler = RayonWorkScheduler;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::WorkScheduler;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{BatchId, ChunkTask, InputArray, InputArraySource, TupleChunkKey};

    use super::RayonWorkScheduler;

    fn task(batch_id: BatchId) -> ChunkTask {
        ChunkTask {
            batch_id,
            array_name: "temperature".to_string(),
            chunk_id: ChunkId::new(1, 0),
            tuple_key: TupleChunkKey::new(vec![0]),
            required_bytes: 4,
            input: InputArray {
                nbytes: 4,
                source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
            },
        }
    }

    #[test]
    fn wait_copied_blocks_until_mark_copied_signals_batch_completion() {
        let scheduler = Arc::new(RayonWorkScheduler::new());
        scheduler
            .submit(task(BatchId(7)))
            .expect("submit should register batch");

        let scheduler_for_wait = Arc::clone(&scheduler);
        let (tx, rx) = mpsc::channel();
        let wait_handle = std::thread::spawn(move || {
            let result = scheduler_for_wait.wait_copied(BatchId(7));
            tx.send(result)
                .expect("wait result channel send should succeed");
        });

        assert!(
            rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "wait_copied should block until mark_copied reaches batch count"
        );

        scheduler
            .mark_copied(BatchId(7))
            .expect("mark_copied should satisfy copy barrier");
        let wait_result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("wait_copied should finish after mark_copied");
        assert!(wait_result.is_ok());
        wait_handle.join().expect("wait thread should join");
    }

    #[test]
    fn completed_batch_is_evicted_after_wait_copied_returns() {
        let scheduler = RayonWorkScheduler::new();
        scheduler
            .submit(task(BatchId(9)))
            .expect("submit should register batch");
        scheduler
            .mark_copied(BatchId(9))
            .expect("mark_copied should satisfy copy barrier");
        scheduler
            .wait_copied(BatchId(9))
            .expect("first wait_copied should succeed");

        let second_wait = scheduler.wait_copied(BatchId(9));
        assert!(
            matches!(
                second_wait,
                Err(SyncWriteError::Validation { message })
                if message.contains("unknown batch id")
            ),
            "completed batch accounting should be evicted after barrier completion"
        );
    }

    #[test]
    fn abort_batch_evicts_partial_accounting() {
        let scheduler = RayonWorkScheduler::new();
        scheduler
            .submit(task(BatchId(13)))
            .expect("submit should register batch");
        scheduler
            .abort_batch(BatchId(13))
            .expect("abort_batch should clear scheduler accounting");

        let wait_result = scheduler.wait_copied(BatchId(13));
        assert!(
            matches!(
                wait_result,
                Err(SyncWriteError::Validation { message })
                if message.contains("unknown batch id")
            ),
            "aborted batch should be removed from wait_copied tracking"
        );
    }

    #[test]
    fn drain_notifies_wait_copied_waiters() {
        let source = include_str!("thread_pool.rs");
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .expect("thread_pool.rs should contain production section");
        let drain_section = production_source
            .split("fn drain(&self) -> Result<(), SyncWriteError> {")
            .nth(1)
            .expect("drain() should exist");
        let drain_block = drain_section
            .split("fn shutdown(&self) -> Result<(), SyncWriteError> {")
            .next()
            .expect("drain() block should end before shutdown()");
        assert!(
            drain_block.contains("self.copied_cv.notify_all()"),
            "drain() must notify wait_copied waiters after clearing accounting",
        );
    }

    #[test]
    fn shutdown_prevents_new_submissions() {
        let scheduler = RayonWorkScheduler::new();
        scheduler.shutdown().expect("shutdown should succeed");
        let submit_result = scheduler.submit(task(BatchId(5)));
        assert!(
            matches!(submit_result, Err(SyncWriteError::ObjectClosed)),
            "submit after shutdown should be rejected"
        );
    }

    #[test]
    fn shutdown_is_idempotent() {
        let scheduler = RayonWorkScheduler::new();
        scheduler.shutdown().expect("first shutdown should succeed");
        scheduler
            .shutdown()
            .expect("second shutdown should be a no-op success");
        let submit_result = scheduler.submit(task(BatchId(11)));
        assert!(
            matches!(submit_result, Err(SyncWriteError::ObjectClosed)),
            "submit should remain rejected after repeated shutdown calls"
        );
    }
}
