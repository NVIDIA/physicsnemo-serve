/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Buffer lease and copy-completion types.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::core::errors::SyncWriteError;

/// Handle to a pooled buffer backed by a slice of slab memory.
#[derive(Clone)]
pub struct BufferHandle {
    id: u32,
    capacity_bytes: usize,
    cuda_registered: bool,
    slab_range: Range<usize>,
    slab_base_addr: usize,
    slab_len: usize,
    slot_lock: Arc<Mutex<()>>,
    _slab_bytes: Arc<Mutex<Vec<u8>>>,
}

impl BufferHandle {
    /// Create a new buffer handle (pool-internal).
    #[must_use]
    pub(crate) fn new(
        id: u32,
        capacity_bytes: usize,
        slab_range: Range<usize>,
        slab_base_addr: usize,
        slab_len: usize,
        slot_lock: Arc<Mutex<()>>,
        slab_bytes: Arc<Mutex<Vec<u8>>>,
    ) -> Self {
        debug_assert!(
            slab_base_addr.checked_add(slab_len).is_some(),
            "slab_base_addr + slab_len must not overflow"
        );

        Self {
            id,
            capacity_bytes,
            cuda_registered: false,
            slab_range,
            slab_base_addr,
            slab_len,
            slot_lock,
            _slab_bytes: slab_bytes,
        }
    }

    /// Pool-assigned buffer identity.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Capacity in bytes of the pooled buffer.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Marks whether this handle originates from a CUDA-registered slab.
    #[must_use]
    pub fn with_cuda_registered(mut self, cuda_registered: bool) -> Self {
        self.cuda_registered = cuda_registered;
        self
    }

    /// Returns `true` when this pooled lease originates from a CUDA-registered slab.
    #[must_use]
    pub fn cuda_registered(&self) -> bool {
        self.cuda_registered
    }

    /// Slab range used by this pooled buffer handle.
    #[must_use]
    pub fn slab_range(&self) -> Range<usize> {
        self.slab_range.clone()
    }

    /// Execute a mutable operation over this handle's slab slice.
    pub fn with_bytes_mut<R>(
        &mut self,
        required_bytes: usize,
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R, SyncWriteError> {
        if required_bytes > self.capacity_bytes {
            return Err(SyncWriteError::copy_failed(format!(
                "required bytes exceed pooled capacity: required={} capacity={}",
                required_bytes, self.capacity_bytes
            )));
        }
        let _slot_guard = self
            .slot_lock
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "pooled slot lock poisoned".to_string(),
            })?;
        if self.slab_range.end > self.slab_len || self.slab_range.start > self.slab_range.end {
            return Err(SyncWriteError::ContractViolation {
                message: "pooled slab range is out of bounds".to_string(),
            });
        }
        let start = self
            .slab_base_addr
            .checked_add(self.slab_range.start)
            .ok_or_else(|| SyncWriteError::ContractViolation {
                message: "pooled slab start pointer overflow".to_string(),
            })?;
        // SAFETY:
        // - `slab_base_addr` points to backing storage held alive by `_slab_bytes`.
        // - `slot_lock` ensures mutable access for this slot is serialized, including
        //   any cloned handles for the same pooled slot.
        // - Range bounds are validated against `slab_len` above.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(start as *mut u8, self.slab_range.len()) };
        Ok(f(slice))
    }

    /// Execute a read-only operation over this handle's slab slice.
    pub fn with_bytes<R>(
        &self,
        required_bytes: usize,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R, SyncWriteError> {
        if required_bytes > self.capacity_bytes {
            return Err(SyncWriteError::copy_failed(format!(
                "required bytes exceed pooled capacity: required={} capacity={}",
                required_bytes, self.capacity_bytes
            )));
        }
        let _slot_guard = self
            .slot_lock
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "pooled slot lock poisoned".to_string(),
            })?;
        if self.slab_range.end > self.slab_len || self.slab_range.start > self.slab_range.end {
            return Err(SyncWriteError::ContractViolation {
                message: "pooled slab range is out of bounds".to_string(),
            });
        }
        let start = self
            .slab_base_addr
            .checked_add(self.slab_range.start)
            .ok_or_else(|| SyncWriteError::ContractViolation {
                message: "pooled slab start pointer overflow".to_string(),
            })?;
        // SAFETY:
        // - `slab_base_addr` points to backing storage held alive by `_slab_bytes`.
        // - `slot_lock` serializes access for this slot across cloned handles.
        // - Range bounds are validated against `slab_len` above.
        let slice =
            unsafe { std::slice::from_raw_parts(start as *const u8, self.slab_range.len()) };
        Ok(f(slice))
    }
}

impl std::fmt::Debug for BufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferHandle")
            .field("id", &self.id)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("cuda_registered", &self.cuda_registered)
            .field("slab_range", &self.slab_range)
            .finish()
    }
}

/// Compares pool identity plus slab location metadata to avoid cross-pool aliasing.
impl PartialEq for BufferHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.capacity_bytes == other.capacity_bytes
            && self.cuda_registered == other.cuda_registered
            && self.slab_range == other.slab_range
            && self.slab_base_addr == other.slab_base_addr
            && self.slab_len == other.slab_len
    }
}

impl Eq for BufferHandle {}

/// Transient (one-shot) buffer for oversized writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientBuffer {
    bytes: Vec<u8>,
}

impl TransientBuffer {
    /// Create a zero-filled transient buffer of the given size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            bytes: vec![0_u8; size],
        }
    }

    /// Read-only access to the buffer contents.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Mutable access to the buffer contents.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Total capacity in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` if the buffer has zero capacity.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A buffer lease acquired from the pool: either pooled or transient.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferLease {
    /// Lease from a pooled slab (may be CUDA-registered).
    Pooled(BufferHandle),
    /// One-shot transient buffer (never CUDA-registered in v1).
    Transient(TransientBuffer),
}

impl BufferLease {
    /// Capacity in bytes for this lease.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        match self {
            Self::Pooled(handle) => handle.capacity_bytes(),
            Self::Transient(buffer) => buffer.len(),
        }
    }

    /// Copy host bytes into this lease.
    pub fn write_from_host_bytes(
        &mut self,
        src: &[u8],
        required_bytes: usize,
    ) -> Result<(), SyncWriteError> {
        if src.len() < required_bytes {
            return Err(SyncWriteError::copy_failed(format!(
                "source host bytes shorter than required bytes: src_len={} required={}",
                src.len(),
                required_bytes
            )));
        }
        match self {
            Self::Pooled(handle) => handle.with_bytes_mut(required_bytes, |dst| {
                dst[..required_bytes].copy_from_slice(&src[..required_bytes]);
            }),
            Self::Transient(buffer) => {
                if required_bytes > buffer.len() {
                    return Err(SyncWriteError::copy_failed(format!(
                        "transient buffer shorter than required bytes: len={} required={}",
                        buffer.len(),
                        required_bytes
                    )));
                }
                buffer.as_bytes_mut()[..required_bytes].copy_from_slice(&src[..required_bytes]);
                Ok(())
            }
        }
    }

    /// Snapshot copied bytes from this lease for async write staging.
    pub fn staged_bytes(&self, required_bytes: usize) -> Result<Vec<u8>, SyncWriteError> {
        match self {
            Self::Pooled(handle) => {
                handle.with_bytes(required_bytes, |src| src[..required_bytes].to_vec())
            }
            Self::Transient(buffer) => {
                if required_bytes > buffer.len() {
                    return Err(SyncWriteError::copy_failed(format!(
                        "transient buffer shorter than required bytes: len={} required={}",
                        buffer.len(),
                        required_bytes
                    )));
                }
                Ok(buffer.as_bytes()[..required_bytes].to_vec())
            }
        }
    }

    /// Run a read-only operation over this lease's bytes without extra staging copy.
    pub fn with_bytes<R>(
        &self,
        required_bytes: usize,
        f: impl FnOnce(&[u8]) -> Result<R, SyncWriteError>,
    ) -> Result<R, SyncWriteError> {
        match self {
            Self::Pooled(handle) => {
                handle.with_bytes(required_bytes, |src| f(&src[..required_bytes]))?
            }
            Self::Transient(buffer) => {
                if required_bytes > buffer.len() {
                    return Err(SyncWriteError::copy_failed(format!(
                        "transient buffer shorter than required bytes: len={} required={}",
                        buffer.len(),
                        required_bytes
                    )));
                }
                f(&buffer.as_bytes()[..required_bytes])
            }
        }
    }
}

/// Planner-derived statistics for first-write pool initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FirstWriteSizingHint {
    /// Number of tasks in the first planned batch.
    pub first_write_task_count: usize,
    /// Maximum `required_bytes` across all tasks in the first batch.
    pub first_write_max_required_bytes: usize,
}

/// Token indicating copy completion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CopyCompletion {
    /// Synchronous host memcpy completed immediately.
    ImmediateHostCopy,
    /// Asynchronous CUDA copy completed (opaque event token).
    CudaCopyDone {
        /// CUDA event handle for completion tracking.
        event: u64,
    },
}
