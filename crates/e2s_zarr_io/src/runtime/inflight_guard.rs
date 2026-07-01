/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inflight async-write guard utilities.

use std::sync::{Arc, Condvar, Mutex};

use crate::core::errors::SyncWriteError;

/// Shared inflight async-write counter and wait condition.
pub(crate) type InflightWriteCounter = Arc<(Mutex<usize>, Condvar)>;

/// RAII guard for inflight async-write accounting.
pub(crate) struct InflightAsyncWriteGuard {
    inflight_counter: InflightWriteCounter,
}

impl InflightAsyncWriteGuard {
    pub(crate) fn register(
        inflight_counter: &InflightWriteCounter,
    ) -> Result<Self, SyncWriteError> {
        let (lock, _) = &**inflight_counter;
        let mut inflight = lock.lock().map_err(|_| SyncWriteError::ContractViolation {
            message: "inflight async write counter lock poisoned".to_string(),
        })?;
        *inflight += 1;
        drop(inflight);
        Ok(Self {
            inflight_counter: Arc::clone(inflight_counter),
        })
    }
}

impl Drop for InflightAsyncWriteGuard {
    fn drop(&mut self) {
        let (lock, cv) = &*self.inflight_counter;
        let mut inflight = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *inflight = inflight.saturating_sub(1);
        cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::{InflightAsyncWriteGuard, InflightWriteCounter};
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::{Arc, Condvar, Mutex};

    #[test]
    fn register_increments_and_drop_decrements_inflight_counter() {
        let counter: InflightWriteCounter = Arc::new((Mutex::new(0), Condvar::new()));

        let guard = InflightAsyncWriteGuard::register(&counter)
            .expect("register should return a guard and increment the counter");
        let (lock, _) = &*counter;
        let after_register = lock.lock().expect("counter lock should not be poisoned");
        assert_eq!(
            *after_register, 1,
            "register() must increment inflight counter to 1"
        );
        drop(after_register);

        drop(guard);
        let after_drop = lock.lock().expect("counter lock should not be poisoned");
        assert_eq!(
            *after_drop, 0,
            "dropping guard must decrement inflight counter back to 0"
        );
    }

    #[test]
    fn drop_recovers_poisoned_lock_and_decrements_counter() {
        let counter: InflightWriteCounter = Arc::new((Mutex::new(0), Condvar::new()));
        let guard = InflightAsyncWriteGuard::register(&counter)
            .expect("register should increment inflight counter before poison scenario");

        let counter_for_poison = Arc::clone(&counter);
        let poison_handle = std::thread::spawn(move || {
            let (lock, _) = &*counter_for_poison;
            let _held = lock
                .lock()
                .expect("poison helper should acquire counter lock");
            panic!("intentional poison to exercise drop fallback");
        });
        assert!(
            poison_handle.join().is_err(),
            "poison helper must panic to poison lock"
        );

        let drop_result = panic::catch_unwind(AssertUnwindSafe(|| drop(guard)));
        assert!(
            drop_result.is_ok(),
            "dropping guard must not panic when lock is poisoned"
        );

        let (lock, _) = &*counter;
        let inflight_after_drop = lock
            .lock()
            .expect_err("lock should remain poisoned for fallback assertion")
            .into_inner();
        assert_eq!(
            *inflight_after_drop, 0,
            "poisoned-lock drop path should still decrement inflight count"
        );
    }
}
