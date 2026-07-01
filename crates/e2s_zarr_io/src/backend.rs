/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend lifecycle implementation.
//!
//! [`SyncZarrBackend`] enforces the core `add_array → write → close` contract
//! and delegates write orchestration to an internal coordinator.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::contracts::{ArrayRegistry, ZarrIoBackend};
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ArrayRegistration, CloseReport, InferenceWriteRequest, WriteCopyAck, WriteInternalTiming,
};
use crate::runtime::coordinator::WriteCoordinator;

/// Synchronous Zarr write backend that enforces core lifecycle contracts.
///
/// Thread-safe: multiple Python threads may call `write()` concurrently
/// when targeting disjoint `ChunkId` spaces.
pub struct SyncZarrBackend {
    coordinator: Arc<WriteCoordinator>,
    array_registry: Arc<dyn ArrayRegistry>,
    default_close_timeout_seconds: f64,
    closed: AtomicBool,
}

impl SyncZarrBackend {
    /// Create a backend with an explicit default `close()` timeout.
    #[must_use]
    pub(crate) fn new_with_close_timeout(
        coordinator: Arc<WriteCoordinator>,
        array_registry: Arc<dyn ArrayRegistry>,
        default_close_timeout_seconds: f64,
    ) -> Self {
        Self {
            coordinator,
            array_registry,
            default_close_timeout_seconds,
            closed: AtomicBool::new(false),
        }
    }

    /// Return the last successful `write()` internal timing snapshot.
    #[must_use]
    pub fn last_write_timing(&self) -> Option<WriteInternalTiming> {
        self.coordinator.last_write_timing()
    }

    /// Return the backend-configured default close timeout in seconds.
    #[must_use]
    pub fn configured_close_timeout_seconds(&self) -> f64 {
        self.default_close_timeout_seconds
    }

    /// Close using the backend-configured default timeout.
    pub fn close_with_configured_timeout(&self) -> Result<CloseReport, SyncWriteError> {
        self.close(self.default_close_timeout_seconds)
    }

    /// Validate that `array_names` are registered via `add_array()`.
    ///
    /// Used by `read()` at the Python boundary to reject unknown array names
    /// before attempting zarr store access.
    ///
    /// # Errors
    ///
    /// Returns [`SyncWriteError::ContractViolation`] if any name is unknown.
    pub fn validate_write_array_names(&self, array_names: &[String]) -> Result<(), SyncWriteError> {
        self.array_registry.validate_write_array_names(array_names)
    }
}

impl ZarrIoBackend for SyncZarrBackend {
    fn add_array(&self, req: ArrayRegistration) -> Result<(), SyncWriteError> {
        // Acquire semantics: see any prior close() store.
        if self.closed.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        req.validate()?;
        self.array_registry.register(req.clone())?;
        self.coordinator.persist_registration_metadata(&req)
    }

    fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        req.validate()?;
        self.array_registry
            .validate_write_array_names(&req.array_names)?;
        let array_ids = self.array_registry.resolve_array_ids(&req.array_names)?;
        let registered_coords = self.array_registry.registered_coords()?;
        if let Some(unknown_coord) = req
            .coords
            .keys()
            .find(|k| !registered_coords.contains_key(*k))
        {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "write() coord key '{unknown_coord}' not found in registered coords"
                ),
            });
        }
        // Best-effort re-check: narrow the race window between validation and submit.
        // This guard is not atomic with submit_write(); close() can still win a TOCTOU
        // race immediately after this check. The goal is to reduce avoidable post-close
        // submissions, not to provide a hard linearizability guarantee.
        if self.closed.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        self.coordinator
            .submit_write(&req, &array_ids, &registered_coords)
    }

    fn close(&self, timeout_seconds: f64) -> Result<CloseReport, SyncWriteError> {
        if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "close() timeout_seconds must be finite and > 0, got {timeout_seconds}"
                ),
            });
        }
        if self.closed.swap(true, Ordering::AcqRel) {
            return Err(SyncWriteError::ObjectClosed);
        }
        let registration = self.array_registry.registration_snapshot()?;
        self.coordinator
            .close(timeout_seconds, registration.as_ref())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

impl Drop for SyncZarrBackend {
    fn drop(&mut self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let registration = self.array_registry.registration_snapshot().ok().flatten();
        let _ = self
            .coordinator
            .close(self.default_close_timeout_seconds, registration.as_ref());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn add_array_and_write_boundary_call_request_validate() {
        let source = include_str!("backend.rs");
        let add_array_section = source
            .split("fn add_array(&self, req: ArrayRegistration) -> Result<(), SyncWriteError> {")
            .nth(1)
            .expect("SyncZarrBackend::add_array should exist");
        let add_array_block = add_array_section
            .split("fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError> {")
            .next()
            .expect("add_array block should end before write()");
        assert!(
            add_array_block.contains("req.validate()?"),
            "add_array() must validate ArrayRegistration at backend boundary",
        );

        let write_section = source
            .split("fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError> {")
            .nth(1)
            .expect("SyncZarrBackend::write should exist");
        let write_block = write_section
            .split("fn close(&self, timeout_seconds: f64) -> Result<CloseReport, SyncWriteError> {")
            .next()
            .expect("write block should end before close()");
        assert!(
            write_block.contains("req.validate()?"),
            "write() must validate InferenceWriteRequest at backend boundary",
        );
    }

    #[test]
    fn write_validates_coord_keys_are_subset_of_registered_coords() {
        let source = include_str!("backend.rs");
        let write_section = source
            .split("fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError> {")
            .nth(1)
            .expect("SyncZarrBackend::write should exist");
        let write_block = write_section
            .split("fn close(&self, timeout_seconds: f64) -> Result<CloseReport, SyncWriteError> {")
            .next()
            .expect("write block should end before close()");
        assert!(
            write_block.contains("validate_write_coord_keys")
                || write_block.contains("coord_keys")
                || write_block.contains("unknown_coord"),
            "write() must validate that request coord keys are a subset of registered coords \
             to produce clear Validation errors instead of late planner failures"
        );
    }

    #[test]
    fn write_toctou_limitation_is_documented_on_trait() {
        let source = include_str!("backend.rs");
        assert!(
            source.contains("TOCTOU") || source.contains("time-of-check"),
            "backend.rs must document the TOCTOU limitation of the write()/close() race"
        );
        let write_doc = source
            .split("fn write(&self, req: InferenceWriteRequest)")
            .next()
            .expect("write method should exist");
        assert!(
            write_doc.contains("TOCTOU")
                || write_doc.contains("linearizability")
                || source.contains("not atomic with submit_write"),
            "write() implementation must document the TOCTOU race semantics"
        );
    }

    #[test]
    fn write_second_close_check_documents_toc_tou_best_effort_behavior() {
        let source = include_str!("backend.rs");
        let write_section = source
            .split("fn write(&self, req: InferenceWriteRequest) -> Result<WriteCopyAck, SyncWriteError> {")
            .nth(1)
            .expect("SyncZarrBackend::write should exist");
        let write_block = write_section
            .split("fn close(&self, timeout_seconds: f64) -> Result<CloseReport, SyncWriteError> {")
            .next()
            .expect("write block should end before close()");
        assert!(
            write_block.contains("Best-effort re-check"),
            "write() should document why it performs a second close-state check",
        );
        assert!(
            write_block.contains("not atomic"),
            "write() documentation should explain TOCTOU semantics of the second close-state check",
        );
    }
}
