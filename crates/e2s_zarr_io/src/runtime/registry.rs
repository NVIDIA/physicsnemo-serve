/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-safe `ChunkId` reservation and commit registry.

use std::collections::HashSet;
use std::sync::Mutex;

use crate::core::chunk_id::ChunkId;
use crate::core::contracts::ChunkKeyRegistry;
use crate::core::errors::SyncWriteError;

#[derive(Default)]
struct RegistryState {
    reserved: HashSet<ChunkId>,
    committed: HashSet<ChunkId>,
}

/// In-memory `ChunkId` registry with strict no-overwrite behavior.
///
/// Thread-safe: uses an internal `Mutex` to serialize reservation checks
/// across concurrent `write()` calls from multiple Python threads.
pub struct InMemoryChunkKeyRegistry {
    state: Mutex<RegistryState>,
}

impl Default for InMemoryChunkKeyRegistry {
    fn default() -> Self {
        Self {
            state: Mutex::new(RegistryState::default()),
        }
    }
}

impl InMemoryChunkKeyRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ChunkKeyRegistry for InMemoryChunkKeyRegistry {
    fn reserve_many_ids(&self, chunk_ids: &[ChunkId]) -> Result<(), SyncWriteError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "chunk registry lock poisoned".to_string(),
            })?;

        // Check all IDs before mutating (atomic batch reservation).
        for chunk_id in chunk_ids {
            if state.reserved.contains(chunk_id) || state.committed.contains(chunk_id) {
                return Err(SyncWriteError::ChunkKeyConflict {
                    chunk_id: *chunk_id,
                });
            }
        }

        for chunk_id in chunk_ids {
            state.reserved.insert(*chunk_id);
        }
        Ok(())
    }

    fn mark_committed_id(&self, chunk_id: &ChunkId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.reserved.remove(chunk_id);
        state.committed.insert(*chunk_id);
    }

    fn release_failed_id(&self, chunk_id: &ChunkId) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.reserved.remove(chunk_id);
    }
}

#[cfg(test)]
mod tests {
    use super::InMemoryChunkKeyRegistry;
    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::ChunkKeyRegistry;

    #[test]
    fn mark_committed_id_recovers_from_poisoned_lock() {
        let registry = InMemoryChunkKeyRegistry::new();
        let chunk_id = ChunkId::new(3, 11);
        registry
            .reserve_many_ids(&[chunk_id])
            .expect("initial reserve should succeed");

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry
                .state
                .lock()
                .expect("state lock should be acquired");
            panic!("poison registry lock");
        }));

        registry.mark_committed_id(&chunk_id);
        let state = registry
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !state.reserved.contains(&chunk_id),
            "mark_committed_id should remove chunk id from reserved set even after poisoning"
        );
        assert!(
            state.committed.contains(&chunk_id),
            "mark_committed_id should insert chunk id into committed set even after poisoning"
        );
    }

    #[test]
    fn release_failed_id_recovers_from_poisoned_lock() {
        let registry = InMemoryChunkKeyRegistry::new();
        let chunk_id = ChunkId::new(4, 29);
        registry
            .reserve_many_ids(&[chunk_id])
            .expect("initial reserve should succeed");

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry
                .state
                .lock()
                .expect("state lock should be acquired");
            panic!("poison registry lock");
        }));

        registry.release_failed_id(&chunk_id);
        let state = registry
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !state.reserved.contains(&chunk_id),
            "release_failed_id should remove chunk id from reserved set even after poisoning"
        );
    }
}
