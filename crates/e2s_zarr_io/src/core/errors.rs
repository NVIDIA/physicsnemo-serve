/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for the `e2s_zarr_io` write pipeline.
//!
//! All fallible operations in this crate return [`SyncWriteError`].
//! [`DeferredWriteError`] captures post-copy-barrier failures that are surfaced
//! during `close()`.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use crate::core::chunk_id::ChunkId;
use crate::core::types::BatchId;

/// A write failure that occurred after the copy barrier (surfaced on `close()`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeferredWriteError {
    /// The write batch that produced this error.
    pub batch_id: BatchId,
    /// The specific chunk that failed, if identifiable.
    pub chunk_id: Option<ChunkId>,
    /// Human-readable description of the failure.
    pub message: String,
}

impl Display for DeferredWriteError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.chunk_id {
            Some(chunk_id) => write!(
                f,
                "deferred write error (batch={}, chunk={}): {}",
                self.batch_id.as_u64(),
                chunk_id,
                self.message
            ),
            None => write!(
                f,
                "deferred write error (batch={}): {}",
                self.batch_id.as_u64(),
                self.message
            ),
        }
    }
}

impl Error for DeferredWriteError {}

/// Cloneable wrapper for source errors attached to [`SyncWriteError`] variants.
#[derive(Clone)]
pub struct SyncErrorCause(Arc<dyn Error + Send + Sync>);

impl SyncErrorCause {
    /// Create a new wrapped source error.
    #[must_use]
    pub fn from_error<E>(err: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self(Arc::new(err))
    }

    /// Borrow as a standard error trait object.
    #[must_use]
    pub fn as_error(&self) -> &(dyn Error + 'static) {
        self.0.as_ref()
    }
}

impl Display for SyncErrorCause {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.0.as_ref(), f)
    }
}

impl fmt::Debug for SyncErrorCause {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SyncErrorCause")
            .field(&self.to_string())
            .finish()
    }
}

impl PartialEq for SyncErrorCause {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SyncErrorCause {}

impl Error for SyncErrorCause {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0.source()
    }
}

/// Primary error type for all synchronous write-path operations.
///
/// This enum is `#[non_exhaustive]` — new variants may be added in minor
/// releases without breaking downstream `match` exhaustiveness.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SyncWriteError {
    /// The backend has already been closed.
    #[error("backend is closed")]
    ObjectClosed,

    /// A lifecycle contract was violated (e.g., `write()` before `add_array()`).
    #[error("contract violation: {message}")]
    ContractViolation {
        /// Description of the violated contract.
        message: String,
    },

    /// Input validation failed (e.g., mismatched array/name counts).
    #[error("validation error: {message}")]
    Validation {
        /// Description of the validation failure.
        message: String,
    },

    /// An explicit `parallel_coords` key was not found in registered coordinates.
    #[error("unknown parallel_coords key: {coord}")]
    UnknownParallelCoord {
        /// The unrecognized coordinate name.
        coord: String,
    },

    /// A `ChunkId` is already reserved or committed (no-overwrite policy).
    #[error("chunk conflict for {chunk_id}")]
    ChunkKeyConflict {
        /// The conflicting chunk identity.
        chunk_id: ChunkId,
    },

    /// Chunk planner error (e.g., template/resolver failure).
    #[error("planner error: {message}")]
    Planner {
        /// Description of the planner failure.
        message: String,
    },

    /// Mixed-radix linearization overflowed the configured integer range.
    #[error("chunk id linearization overflow")]
    ChunkIdOverflow,

    /// Hot or Warm slab allocation failed.
    #[error("slab allocation failed")]
    SlabAllocationFailed,

    /// Slab memory pinning failed.
    #[error("slab pinning failed")]
    SlabPinningFailed,

    /// CUDA slab registration failed.
    #[error("cuda slab registration failed")]
    CudaSlabRegistrationFailed,

    /// Pool initialization failed (sizing, alignment, or resource error).
    #[error("pool initialization failed: {message}")]
    PoolInitialization {
        /// Description of the initialization failure.
        message: String,
    },

    /// Transient buffer request exceeds `max_transient_buffer_bytes`.
    #[error("transient allocation exceeds limit: requested={requested_bytes} limit={limit_bytes}")]
    TransientAllocationLimitExceeded {
        /// Bytes requested by the caller.
        requested_bytes: usize,
        /// Configured limit.
        limit_bytes: usize,
    },

    /// Total in-flight transient bytes exceed configured cap.
    #[error(
        "transient in-flight allocation exceeds limit: requested={requested_bytes} in_flight={in_flight_bytes} limit={limit_bytes}"
    )]
    TransientInFlightLimitExceeded {
        /// Bytes requested by the caller for the new transient lease.
        requested_bytes: usize,
        /// Current in-flight transient bytes before this allocation attempt.
        in_flight_bytes: usize,
        /// Configured in-flight byte cap.
        limit_bytes: usize,
    },

    /// Transient buffer allocation failed at the OS level.
    #[error("transient allocation failed: {message}")]
    TransientAllocationFailed {
        /// Description of the allocation failure.
        message: String,
    },

    /// Copy operation failed (size mismatch, CUDA error, etc.).
    #[error("copy failed: {message}")]
    CopyFailed {
        /// Description of the copy failure.
        message: String,
        /// Underlying source error when available.
        #[source]
        cause: Option<SyncErrorCause>,
    },

    /// Filesystem I/O error during chunk write.
    #[error("io failed: {message}")]
    IoFailed {
        /// Description of the I/O failure.
        message: String,
        /// Underlying source error when available.
        #[source]
        cause: Option<SyncErrorCause>,
    },

    /// `close()` timed out waiting for pooled leases to return.
    #[error("close timed out waiting for lease return: outstanding={outstanding_leases}")]
    LeaseReturnTimeout {
        /// Number of leases still outstanding at timeout.
        outstanding_leases: usize,
    },

    /// Metadata consolidation failed during `close()`.
    #[error("metadata consolidation failed: {message}")]
    MetadataConsolidationFailed {
        /// Description of the consolidation failure.
        message: String,
        /// Underlying source error when available.
        #[source]
        cause: Option<SyncErrorCause>,
    },

    /// Deferred write failures collected from prior `write()` calls.
    #[error(
        "deferred write failures surfaced on close: {detail}",
        detail = format_deferred_write_failures(failures)
    )]
    DeferredWriteFailures {
        /// The individual deferred errors.
        failures: Vec<DeferredWriteError>,
    },

    /// The configured input stability policy is not supported in this version.
    #[error("unsupported input stability policy '{policy}'; only StrictGilHold is supported in v1")]
    UnsupportedInputStabilityPolicy {
        /// The unsupported policy name.
        policy: String,
    },

    /// The zarr format/encoding/separator combination is unsupported.
    #[error("unsupported zarr target configuration: {message}")]
    UnsupportedZarrTargetConfig {
        /// Description of the invalid combination.
        message: String,
    },
}

impl SyncWriteError {
    /// Create a copy failure without an attached source error.
    #[must_use]
    pub fn copy_failed(message: impl Into<String>) -> Self {
        Self::CopyFailed {
            message: message.into(),
            cause: None,
        }
    }

    /// Create a copy failure with an attached source error.
    #[must_use]
    pub fn copy_failed_with_cause<E>(message: impl Into<String>, cause: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::CopyFailed {
            message: message.into(),
            cause: Some(SyncErrorCause::from_error(cause)),
        }
    }

    /// Create an I/O failure without an attached source error.
    #[must_use]
    pub fn io_failed(message: impl Into<String>) -> Self {
        Self::IoFailed {
            message: message.into(),
            cause: None,
        }
    }

    /// Create an I/O failure with an attached source error.
    #[must_use]
    pub fn io_failed_with_cause<E>(message: impl Into<String>, cause: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::IoFailed {
            message: message.into(),
            cause: Some(SyncErrorCause::from_error(cause)),
        }
    }

    /// Create a metadata-consolidation failure without an attached source error.
    #[must_use]
    pub fn metadata_consolidation_failed(message: impl Into<String>) -> Self {
        Self::MetadataConsolidationFailed {
            message: message.into(),
            cause: None,
        }
    }

    /// Create a metadata-consolidation failure with an attached source error.
    #[must_use]
    pub fn metadata_consolidation_failed_with_cause<E>(message: impl Into<String>, cause: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::MetadataConsolidationFailed {
            message: message.into(),
            cause: Some(SyncErrorCause::from_error(cause)),
        }
    }
}

fn format_deferred_write_failures(failures: &[DeferredWriteError]) -> String {
    if let Some(first) = failures.first() {
        if let Some(chunk_id) = first.chunk_id {
            format!(
                "count={} first_batch_id={} first_chunk_id={} first_error={}",
                failures.len(),
                first.batch_id,
                chunk_id,
                first.message
            )
        } else {
            format!(
                "count={} first_batch_id={} first_error={}",
                failures.len(),
                first.batch_id,
                first.message
            )
        }
    } else {
        "count=0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::{self, Display, Formatter};

    use super::*;
    use crate::core::chunk_id::ChunkId;
    use crate::core::types::BatchId;

    #[derive(Debug)]
    struct CollidingDisplayError {
        _detail: &'static str,
    }

    impl Display for CollidingDisplayError {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "same-display")
        }
    }

    impl Error for CollidingDisplayError {}

    #[test]
    fn deferred_write_error_display_includes_batch_chunk_and_message() {
        let err = DeferredWriteError {
            batch_id: BatchId(42),
            chunk_id: Some(ChunkId::new(1, 99)),
            message: "flush failed".to_string(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("batch=42"));
        assert!(rendered.contains("chunk=ChunkId(array=1, idx=99)"));
        assert!(rendered.contains("flush failed"));
    }

    #[test]
    fn deferred_write_error_display_without_chunk_id() {
        let err = DeferredWriteError {
            batch_id: BatchId(7),
            chunk_id: None,
            message: "metadata write failed".to_string(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("batch=7"));
        assert!(!rendered.contains("chunk="));
        assert!(rendered.contains("metadata write failed"));
    }

    #[test]
    fn deferred_write_error_implements_std_error_traits() {
        fn assert_std_error<E: Error + Send + Sync + 'static>() {}
        assert_std_error::<DeferredWriteError>();
    }

    #[test]
    fn sync_write_error_source_returns_wrapped_io_cause() {
        let io_err = std::io::Error::other("disk full");
        let err = SyncWriteError::io_failed_with_cause("failed writing chunk", io_err);

        let source = err
            .source()
            .expect("io_failed_with_cause should expose underlying source");
        assert!(source.to_string().contains("disk full"));
    }

    #[test]
    fn sync_write_error_without_cause_reports_no_source() {
        let err = SyncWriteError::io_failed("plain message");
        assert!(
            err.source().is_none(),
            "io_failed without cause should not expose source"
        );
    }

    #[test]
    fn sync_error_cause_partial_eq_is_not_based_only_on_display_text() {
        let left = SyncErrorCause::from_error(CollidingDisplayError { _detail: "left" });
        let right = SyncErrorCause::from_error(CollidingDisplayError { _detail: "right" });
        let _ = (left.as_error().to_string(), right.as_error().to_string());
        assert_ne!(
            left, right,
            "distinct source errors with colliding display text must not compare equal"
        );
    }

    #[test]
    fn sync_write_error_uses_thiserror_derive_instead_of_manual_impls() {
        let source = include_str!("errors.rs");
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("errors.rs should contain a production section before tests");
        assert!(
            production_source.contains("#[derive(Debug, Clone, PartialEq, Eq, Error)]")
                || production_source
                    .contains("#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]"),
            "SyncWriteError should derive thiserror::Error",
        );
        assert!(
            !production_source.contains("impl Display for SyncWriteError"),
            "SyncWriteError display formatting should be provided by thiserror derive attributes",
        );
        assert!(
            !production_source.contains("impl Error for SyncWriteError"),
            "SyncWriteError source wiring should be provided by thiserror derive attributes",
        );
    }
}
