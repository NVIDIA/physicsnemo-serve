/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared reusable buffer pool with hot/warm slab lifecycle.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;

use crate::core::contracts::BufferPool;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    BufferHandle, BufferLease, BufferPoolConfig, BufferSizingPolicy, FirstWriteSizingHint,
    ModelProfileHint, PoolAlignment, PoolWarmupStatus, SizeOverride, TransientBuffer,
    WarmSlabState,
};
use crate::runtime::cuda_runtime::{CudaRuntimeApi, shared_cuda_runtime_api};

#[derive(Debug)]
struct SizedPoolState {
    fast_path: Arc<FastPoolState>,
    hot_slab: SlabBacking,
    warm_slab: Option<SlabBacking>,
}

#[derive(Debug)]
struct PoolRuntimeState {
    initialized: bool,
    shutdown: bool,
    next_buffer_id: u32,
    sized: Option<SizedPoolState>,
    warmup_thread: Option<JoinHandle<()>>,
}

impl Default for PoolRuntimeState {
    fn default() -> Self {
        Self {
            initialized: false,
            shutdown: false,
            next_buffer_id: 1,
            sized: None,
            warmup_thread: None,
        }
    }
}

#[derive(Debug)]
struct ComputedPoolLayout {
    pooled_buffer_bytes: usize,
    pooled_buffer_stride: usize,
    hot_slab_buffers: usize,
    warm_slab_buffers: usize,
}

#[derive(Debug)]
struct FastPoolState {
    pooled_buffer_bytes: usize,
    hot_slab_buffers: usize,
    warm_slab_buffers: usize,
    hot_start_id: u32,
    hot_id_max: u32,
    warm_start_id: Option<u32>,
    hot_buffer_ranges: Vec<Range<usize>>,
    warm_buffer_ranges: Vec<Range<usize>>,
    hot_slot_locks: Vec<Arc<Mutex<()>>>,
    warm_slot_locks: Vec<Arc<Mutex<()>>>,
    hot_slab_bytes: Arc<Mutex<Vec<u8>>>,
    warm_slab_bytes: Option<Arc<Mutex<Vec<u8>>>>,
    hot_slab_base_addr: usize,
    hot_slab_len: usize,
    warm_slab_base_addr: Option<usize>,
    warm_slab_len: Option<usize>,
    free_hot_ids: ArrayQueue<u32>,
    free_warm_ids: Option<ArrayQueue<u32>>,
    leased_slots: Vec<AtomicBool>,
    warm_state: AtomicU8,
    hot_cuda_registered: AtomicBool,
    warm_cuda_registered: AtomicBool,
}

impl FastPoolState {
    fn encode_warm_state(state: WarmSlabState) -> u8 {
        match state {
            WarmSlabState::NotStarted => 0,
            WarmSlabState::InProgress => 1,
            WarmSlabState::Ready => 2,
            WarmSlabState::FailedDegraded => 3,
        }
    }

    fn decode_warm_state(raw: u8) -> WarmSlabState {
        match raw {
            0 => WarmSlabState::NotStarted,
            1 => WarmSlabState::InProgress,
            2 => WarmSlabState::Ready,
            3 => WarmSlabState::FailedDegraded,
            _ => WarmSlabState::NotStarted,
        }
    }

    fn warm_state(&self) -> WarmSlabState {
        Self::decode_warm_state(self.warm_state.load(Ordering::Acquire))
    }

    fn set_warm_state(&self, state: WarmSlabState) {
        self.warm_state
            .store(Self::encode_warm_state(state), Ordering::Release);
    }
}

#[derive(Debug)]
struct SlabBacking {
    bytes: Arc<Mutex<Vec<u8>>>,
    base_addr: usize,
    total_len: usize,
    aligned_start: usize,
    aligned_len: usize,
    buffer_ranges: Vec<Range<usize>>,
    pinned: bool,
    cuda_registered: bool,
}

impl SlabBacking {
    fn with_aligned_slice_mut<R>(
        &self,
        f: impl FnOnce(&mut [u8]) -> Result<R, SyncWriteError>,
    ) -> Result<R, SyncWriteError> {
        let mut slab = self
            .bytes
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "slab backing lock poisoned".to_string(),
            })?;
        let end = self.aligned_start + self.aligned_len;
        if end > slab.len() || self.aligned_start > end {
            return Err(SyncWriteError::ContractViolation {
                message: "slab aligned range is out of bounds".to_string(),
            });
        }
        f(&mut slab[self.aligned_start..end])
    }

    #[cfg(test)]
    fn total_bytes(&self) -> Result<usize, SyncWriteError> {
        Ok(self.total_len)
    }

    fn bytes_arc(&self) -> Arc<Mutex<Vec<u8>>> {
        Arc::clone(&self.bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlabKind {
    Hot,
    Warm,
}

trait RuntimeMemoryOps: Send + Sync {
    fn cuda_available(&self) -> bool;
    /// Returns `true` if the OS supports `mlock()`-based memory pinning.
    /// Implementations should cache the result so the probe runs at most once.
    fn can_pin_memory(&self) -> bool;
    fn pin_slab(&self, slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError>;
    fn unpin_slab(&self, slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError>;
    fn cuda_register_slab(&self, slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError>;
    fn cuda_unregister_slab(&self, slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError>;
}

struct DefaultRuntimeMemoryOps {
    cuda_runtime: Arc<dyn CudaRuntimeApi>,
    mlock_probe_result: std::sync::OnceLock<bool>,
}

impl std::fmt::Debug for DefaultRuntimeMemoryOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultRuntimeMemoryOps")
            .field("cuda_available", &self.cuda_runtime.available())
            .field("mlock_probe_result", &self.mlock_probe_result.get())
            .finish()
    }
}

impl Default for DefaultRuntimeMemoryOps {
    fn default() -> Self {
        Self {
            cuda_runtime: shared_cuda_runtime_api(),
            mlock_probe_result: std::sync::OnceLock::new(),
        }
    }
}

impl RuntimeMemoryOps for DefaultRuntimeMemoryOps {
    fn cuda_available(&self) -> bool {
        self.cuda_runtime.available()
    }

    fn can_pin_memory(&self) -> bool {
        *self.mlock_probe_result.get_or_init(|| {
            #[cfg(unix)]
            {
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                if page_size <= 0 {
                    return false;
                }
                let page_size = page_size as usize;
                let layout = match std::alloc::Layout::from_size_align(page_size, page_size) {
                    Ok(l) => l,
                    Err(_) => return false,
                };
                // SAFETY: layout has non-zero size (page_size > 0) and power-of-two alignment.
                let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
                if ptr.is_null() {
                    return false;
                }
                // SAFETY: ptr is a valid, page-aligned, page-sized allocation.
                let lock_rc = unsafe { libc::mlock(ptr.cast(), page_size) };
                let result = if lock_rc == 0 {
                    // SAFETY: ptr was successfully mlock'd; munlock releases the pin.
                    unsafe { libc::munlock(ptr.cast(), page_size) };
                    true
                } else {
                    false
                };
                // SAFETY: ptr was allocated with `alloc_zeroed` using `layout`.
                unsafe { std::alloc::dealloc(ptr, layout) };
                result
            }
            #[cfg(not(unix))]
            {
                false
            }
        })
    }

    fn pin_slab(&self, _slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError> {
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(unix)]
        {
            // SAFETY: `bytes.as_ptr()` is valid for `bytes.len()` bytes by slice invariants.
            // `mlock` does not outlive this call and does not take ownership of the pointer.
            let rc = unsafe { ::libc::mlock(bytes.as_ptr().cast(), bytes.len()) };
            if rc == 0 {
                Ok(())
            } else {
                Err(SyncWriteError::SlabPinningFailed)
            }
        }
        #[cfg(not(unix))]
        {
            Err(SyncWriteError::SlabPinningFailed)
        }
    }

    fn unpin_slab(&self, _slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError> {
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(unix)]
        {
            // SAFETY: `bytes.as_ptr()` is valid for `bytes.len()` bytes by slice invariants.
            // `munlock` only consumes the pointer during the call and does not retain it.
            let rc = unsafe { ::libc::munlock(bytes.as_ptr().cast(), bytes.len()) };
            if rc == 0 {
                Ok(())
            } else {
                Err(SyncWriteError::SlabPinningFailed)
            }
        }
        #[cfg(not(unix))]
        {
            Err(SyncWriteError::SlabPinningFailed)
        }
    }

    fn cuda_register_slab(&self, _slab: SlabKind, bytes: &mut [u8]) -> Result<(), SyncWriteError> {
        self.cuda_runtime
            .host_register(bytes.as_mut_ptr(), bytes.len())
            .map_err(|_| SyncWriteError::CudaSlabRegistrationFailed)
    }

    fn cuda_unregister_slab(
        &self,
        _slab: SlabKind,
        bytes: &mut [u8],
    ) -> Result<(), SyncWriteError> {
        self.cuda_runtime
            .host_unregister(bytes.as_mut_ptr())
            .map_err(|_| SyncWriteError::CudaSlabRegistrationFailed)
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct DebugPoolSnapshot {
    pooled_buffer_bytes: usize,
    hot_slab_buffers: usize,
    warm_slab_buffers: usize,
    warm_state: WarmSlabState,
    hot_buffer_ranges: Vec<Range<usize>>,
    warm_buffer_ranges: Vec<Range<usize>>,
    hot_slab_backing_bytes: usize,
    warm_slab_backing_bytes: usize,
    hot_pinned: bool,
    warm_pinned: bool,
    hot_cuda_registered: bool,
    warm_cuda_registered: bool,
    transient_inflight_bytes: usize,
    transient_inflight_count: usize,
    transient_total_acquired: usize,
    transient_total_released: usize,
}

/// In-memory buffer pool implementation with hot/warm lease classes.
///
/// The pooled path returns lightweight handles; the transient path allocates
/// owned `Vec<u8>` buffers. Handle IDs model slab membership and are recycled
/// through hot/warm free lists.
pub struct MemoryBufferPool {
    config: BufferPoolConfig,
    runtime_ops: Arc<dyn RuntimeMemoryOps>,
    fast_path: OnceLock<Arc<FastPoolState>>,
    shutdown: AtomicBool,
    transient_inflight_bytes: AtomicUsize,
    transient_inflight_count: AtomicUsize,
    transient_total_acquired: AtomicUsize,
    transient_total_released: AtomicUsize,
    lease_wait_lock: Mutex<()>,
    lease_wait_cv: Condvar,
    state: Arc<Mutex<PoolRuntimeState>>,
}

impl Default for MemoryBufferPool {
    fn default() -> Self {
        Self::new(BufferPoolConfig::default())
    }
}

impl MemoryBufferPool {
    /// Create a pool with the given configuration.
    #[must_use]
    pub fn new(config: BufferPoolConfig) -> Self {
        Self {
            config,
            runtime_ops: Arc::new(DefaultRuntimeMemoryOps::default()),
            fast_path: OnceLock::new(),
            shutdown: AtomicBool::new(false),
            transient_inflight_bytes: AtomicUsize::new(0),
            transient_inflight_count: AtomicUsize::new(0),
            transient_total_acquired: AtomicUsize::new(0),
            transient_total_released: AtomicUsize::new(0),
            lease_wait_lock: Mutex::new(()),
            lease_wait_cv: Condvar::new(),
            state: Arc::new(Mutex::new(PoolRuntimeState::default())),
        }
    }

    #[cfg(test)]
    fn new_with_runtime_ops(
        config: BufferPoolConfig,
        runtime_ops: Arc<dyn RuntimeMemoryOps>,
    ) -> Self {
        Self {
            config,
            runtime_ops,
            fast_path: OnceLock::new(),
            shutdown: AtomicBool::new(false),
            transient_inflight_bytes: AtomicUsize::new(0),
            transient_inflight_count: AtomicUsize::new(0),
            transient_total_acquired: AtomicUsize::new(0),
            transient_total_released: AtomicUsize::new(0),
            lease_wait_lock: Mutex::new(()),
            lease_wait_cv: Condvar::new(),
            state: Arc::new(Mutex::new(PoolRuntimeState::default())),
        }
    }

    /// Returns the active pool configuration.
    #[must_use]
    pub fn config(&self) -> &BufferPoolConfig {
        &self.config
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, PoolRuntimeState>, SyncWriteError> {
        self.state
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "buffer pool state lock poisoned".to_string(),
            })
    }

    fn alignment_bytes(&self) -> usize {
        match self.config.pool_alignment {
            PoolAlignment::Align4KiB => 4 * 1024,
            PoolAlignment::Align64KiB => 64 * 1024,
        }
    }

    fn align_up(&self, value: usize) -> Result<usize, SyncWriteError> {
        let alignment = self.alignment_bytes();
        if alignment == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "pool alignment must be greater than 0".to_string(),
            });
        }
        let rem = value % alignment;
        if rem == 0 {
            Ok(value)
        } else {
            value
                .checked_add(alignment - rem)
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: format!("alignment overflow while aligning pooled buffer bytes: value={value} alignment={alignment}"),
                })
        }
    }

    fn model_profile_baseline_bytes(&self) -> usize {
        match self.config.first_write_sizing.model_profile_hint {
            Some(ModelProfileHint::Fcn) => 4_147_200,
            Some(ModelProfileHint::Dlwp) => 4_152_960,
            Some(ModelProfileHint::Sfno) => 4_152_960,
            Some(ModelProfileHint::Pangu) => 4_152_960,
            Some(ModelProfileHint::GraphCastSmall) => 260_640,
            Some(ModelProfileHint::StormCast) => 1_310_720,
            Some(ModelProfileHint::PrecipitationAfno) => 4_147_200,
            Some(ModelProfileHint::CorrDiffTaiwan) => 795_664,
            None => self.config.first_write_sizing.global_fallback_chunk_bytes,
        }
    }

    fn clamp_usize(value: usize, min_value: usize, max_value: usize) -> usize {
        value.max(min_value).min(max_value)
    }

    /// Write one byte per OS page to force physical page allocation.
    ///
    /// On Linux, `vec![0u8; n]` is backed by copy-on-write zero pages. Physical
    /// memory is only allocated on first write (page fault). When CUDA DMA
    /// targets these pages, each fault adds significant latency. Pre-faulting
    /// costs ~1-2 ms for a 200 MiB slab but eliminates the per-step penalty.
    fn prefault_pages(bytes: &mut [u8]) {
        const PAGE_SIZE: usize = 4096;
        for offset in (0..bytes.len()).step_by(PAGE_SIZE) {
            // A volatile write ensures the compiler does not elide the store,
            // and the OS must map a physical page for this address.
            unsafe {
                std::ptr::write_volatile(bytes.as_mut_ptr().add(offset), 0);
            }
        }
    }

    fn build_slab_backing(
        &self,
        buffer_count: usize,
        pooled_buffer_bytes: usize,
        pooled_buffer_stride: usize,
        pinned: bool,
        cuda_registered: bool,
    ) -> Result<SlabBacking, SyncWriteError> {
        if buffer_count == 0 {
            return Ok(SlabBacking {
                bytes: Arc::new(Mutex::new(Vec::new())),
                base_addr: 0,
                total_len: 0,
                aligned_start: 0,
                aligned_len: 0,
                buffer_ranges: Vec::new(),
                pinned,
                cuda_registered,
            });
        }

        let alignment = self.alignment_bytes();
        let aligned_len = buffer_count
            .checked_mul(pooled_buffer_stride)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "slab byte count overflowed usize".to_string(),
            })?;
        let alloc_len = aligned_len
            .checked_add(alignment.saturating_sub(1))
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "slab aligned allocation length overflowed usize".to_string(),
            })?;
        let mut bytes = vec![0_u8; alloc_len];
        // Pre-fault every page so the OS maps physical memory now, not lazily
        // during the first CUDA D2H DMA. Without this, the first write to each
        // 4 KiB page triggers a page fault that adds ~10 ms per-worker penalty
        // for the first N steps until the entire slab is faulted in.
        Self::prefault_pages(&mut bytes);
        let base_ptr = bytes.as_ptr() as usize;
        let misalign = base_ptr % alignment;
        let aligned_start = if misalign == 0 {
            0
        } else {
            alignment - misalign
        };
        let aligned_end = aligned_start.checked_add(aligned_len).ok_or_else(|| {
            SyncWriteError::PoolInitialization {
                message: "aligned slab end overflowed usize".to_string(),
            }
        })?;
        if aligned_end > bytes.len() {
            return Err(SyncWriteError::PoolInitialization {
                message: "aligned slab range exceeds allocated backing bytes".to_string(),
            });
        }

        let mut buffer_ranges = Vec::with_capacity(buffer_count);
        for idx in 0..buffer_count {
            let start = aligned_start
                .checked_add(idx.saturating_mul(pooled_buffer_stride))
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: "buffer range start overflowed usize".to_string(),
                })?;
            let end = start.checked_add(pooled_buffer_bytes).ok_or_else(|| {
                SyncWriteError::PoolInitialization {
                    message: "buffer range end overflowed usize".to_string(),
                }
            })?;
            if end > aligned_end {
                return Err(SyncWriteError::PoolInitialization {
                    message: "buffer slice exceeds slab backing range".to_string(),
                });
            }
            buffer_ranges.push(start..end);
        }

        Ok(SlabBacking {
            bytes: Arc::new(Mutex::new(bytes)),
            base_addr: base_ptr,
            total_len: alloc_len,
            aligned_start,
            aligned_len,
            buffer_ranges,
            pinned,
            cuda_registered,
        })
    }

    fn build_id_queue(
        &self,
        ids: impl Iterator<Item = u32>,
        capacity: usize,
        queue_name: &str,
    ) -> Result<ArrayQueue<u32>, SyncWriteError> {
        if capacity == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: format!("{queue_name} queue capacity must be greater than 0"),
            });
        }
        let queue = ArrayQueue::new(capacity);
        for id in ids {
            queue
                .push(id)
                .map_err(|_| SyncWriteError::PoolInitialization {
                    message: format!(
                        "failed populating {queue_name} queue: queue capacity was exceeded"
                    ),
                })?;
        }
        Ok(queue)
    }

    fn slot_index_for_id(
        &self,
        fast_path: &FastPoolState,
        id: u32,
    ) -> Result<usize, SyncWriteError> {
        let relative = id.checked_sub(fast_path.hot_start_id).ok_or_else(|| {
            SyncWriteError::PoolInitialization {
                message: "pooled handle id underflowed hot slab start id".to_string(),
            }
        })?;
        let slot_index =
            usize::try_from(relative).map_err(|_| SyncWriteError::PoolInitialization {
                message: "failed converting pooled handle slot index to usize".to_string(),
            })?;
        if slot_index >= fast_path.leased_slots.len() {
            return Err(SyncWriteError::PoolInitialization {
                message: "pooled handle id exceeds allocated slot range".to_string(),
            });
        }
        Ok(slot_index)
    }

    fn try_mark_leased(&self, fast_path: &FastPoolState, id: u32) -> Result<bool, SyncWriteError> {
        let slot_index = self.slot_index_for_id(fast_path, id)?;
        Ok(fast_path.leased_slots[slot_index]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok())
    }

    fn try_mark_released(
        &self,
        fast_path: &FastPoolState,
        id: u32,
    ) -> Result<(usize, bool), SyncWriteError> {
        let slot_index = self.slot_index_for_id(fast_path, id)?;
        let released = fast_path.leased_slots[slot_index]
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        Ok((slot_index, released))
    }

    fn build_pooled_handle_for_id(
        &self,
        fast_path: &FastPoolState,
        id: u32,
    ) -> Result<BufferHandle, SyncWriteError> {
        if id <= fast_path.hot_id_max {
            let slot_index = usize::try_from(id - fast_path.hot_start_id).map_err(|_| {
                SyncWriteError::PoolInitialization {
                    message: "failed converting hot slab slot index to usize".to_string(),
                }
            })?;
            let slot_range = fast_path.hot_buffer_ranges.get(slot_index).ok_or_else(|| {
                SyncWriteError::PoolInitialization {
                    message: "missing hot slab buffer range for pooled handle".to_string(),
                }
            })?;
            if slot_range.end > fast_path.hot_slab_len || slot_range.start >= slot_range.end {
                return Err(SyncWriteError::PoolInitialization {
                    message: "hot slab buffer range points outside slab backing memory".to_string(),
                });
            }
            return Ok(BufferHandle::new(
                id,
                fast_path.pooled_buffer_bytes,
                slot_range.clone(),
                fast_path.hot_slab_base_addr,
                fast_path.hot_slab_len,
                Arc::clone(&fast_path.hot_slot_locks[slot_index]),
                Arc::clone(&fast_path.hot_slab_bytes),
            )
            .with_cuda_registered(fast_path.hot_cuda_registered.load(Ordering::Acquire)));
        }

        let warm_start_id =
            fast_path
                .warm_start_id
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: "warm slab id range missing while warm slab is ready".to_string(),
                })?;
        let slot_index = usize::try_from(id - warm_start_id).map_err(|_| {
            SyncWriteError::PoolInitialization {
                message: "failed converting warm slab slot index to usize".to_string(),
            }
        })?;
        let slot_range = fast_path
            .warm_buffer_ranges
            .get(slot_index)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "missing warm slab buffer range for pooled handle".to_string(),
            })?;
        let warm_slab_bytes = fast_path.warm_slab_bytes.as_ref().ok_or_else(|| {
            SyncWriteError::PoolInitialization {
                message: "warm slab backing missing while warm slab is ready".to_string(),
            }
        })?;
        let warm_slab_len =
            fast_path
                .warm_slab_len
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: "warm slab backing length missing while warm slab is ready"
                        .to_string(),
                })?;
        if slot_range.end > warm_slab_len || slot_range.start >= slot_range.end {
            return Err(SyncWriteError::PoolInitialization {
                message: "warm slab buffer range points outside slab backing memory".to_string(),
            });
        }
        let warm_base_addr =
            fast_path
                .warm_slab_base_addr
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: "warm slab backing base address missing while warm slab is ready"
                        .to_string(),
                })?;
        Ok(BufferHandle::new(
            id,
            fast_path.pooled_buffer_bytes,
            slot_range.clone(),
            warm_base_addr,
            warm_slab_len,
            Arc::clone(&fast_path.warm_slot_locks[slot_index]),
            Arc::clone(warm_slab_bytes),
        )
        .with_cuda_registered(fast_path.warm_cuda_registered.load(Ordering::Acquire)))
    }

    fn try_acquire_hot_lease(
        &self,
        fast_path: &FastPoolState,
    ) -> Result<Option<BufferLease>, SyncWriteError> {
        while let Some(id) = fast_path.free_hot_ids.pop() {
            if self.try_mark_leased(fast_path, id)? {
                let handle = self.build_pooled_handle_for_id(fast_path, id)?;
                return Ok(Some(BufferLease::Pooled(handle)));
            }
        }
        Ok(None)
    }

    fn try_acquire_warm_lease(
        &self,
        fast_path: &FastPoolState,
    ) -> Result<Option<BufferLease>, SyncWriteError> {
        if fast_path.warm_state() != WarmSlabState::Ready {
            return Ok(None);
        }
        let Some(queue) = fast_path.free_warm_ids.as_ref() else {
            return Ok(None);
        };
        while let Some(id) = queue.pop() {
            if self.try_mark_leased(fast_path, id)? {
                let handle = self.build_pooled_handle_for_id(fast_path, id)?;
                return Ok(Some(BufferLease::Pooled(handle)));
            }
        }
        Ok(None)
    }

    fn reserve_transient_allocation(&self, requested_bytes: usize) -> Result<(), SyncWriteError> {
        if let Some(limit) = self.config.max_inflight_transient_bytes {
            loop {
                let current = self.transient_inflight_bytes.load(Ordering::Acquire);
                let new_total = current.checked_add(requested_bytes).ok_or_else(|| {
                    SyncWriteError::TransientAllocationFailed {
                        message: "transient in-flight byte accounting overflow".to_string(),
                    }
                })?;
                if new_total > limit {
                    return Err(SyncWriteError::TransientInFlightLimitExceeded {
                        requested_bytes,
                        in_flight_bytes: current,
                        limit_bytes: limit,
                    });
                }
                if self
                    .transient_inflight_bytes
                    .compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        } else {
            loop {
                let current = self.transient_inflight_bytes.load(Ordering::Acquire);
                let new_total = current.checked_add(requested_bytes).ok_or_else(|| {
                    SyncWriteError::TransientAllocationFailed {
                        message: "transient in-flight byte accounting overflow".to_string(),
                    }
                })?;
                if self
                    .transient_inflight_bytes
                    .compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }
        self.transient_inflight_count.fetch_add(1, Ordering::AcqRel);
        self.transient_total_acquired.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    fn release_transient_allocation(&self, released_bytes: usize) {
        self.transient_total_released.fetch_add(1, Ordering::AcqRel);
        let _ = self.transient_inflight_count.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_sub(1)),
        );
        loop {
            let current = self.transient_inflight_bytes.load(Ordering::Acquire);
            let new_total = current.saturating_sub(released_bytes);
            if self
                .transient_inflight_bytes
                .compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    fn outstanding_pooled_leases(&self) -> usize {
        let Some(fast_path) = self.fast_path.get() else {
            return 0;
        };
        fast_path
            .leased_slots
            .iter()
            .filter(|leased| leased.load(Ordering::Acquire))
            .count()
    }

    fn outstanding_total_leases(&self) -> usize {
        self.outstanding_pooled_leases()
            .saturating_add(self.transient_inflight_count.load(Ordering::Acquire))
    }

    fn spawn_warmup_task_if_needed(&self) -> Result<(), SyncWriteError> {
        let state = Arc::clone(&self.state);
        let runtime_ops = Arc::clone(&self.runtime_ops);
        let pin_pooled_slabs = self.config.pin_pooled_slabs && self.runtime_ops.can_pin_memory();
        let cuda_register =
            self.config.cuda_register_pool_if_available && self.config.cuda_register_each_slab_once;
        let will_cuda_register_warm = cuda_register && self.runtime_ops.cuda_available();
        let pin_pooled_slabs = pin_pooled_slabs && !will_cuda_register_warm;
        let warmup_thread = match std::thread::Builder::new()
            .name("e2s-zarr-warmup".to_string())
            .spawn(move || {
                // Simulate asynchronous warmup progression in a deterministic, short window.
                std::thread::sleep(Duration::from_millis(50));
                let Ok(mut guard) = state.lock() else {
                    return;
                };
                if guard.shutdown {
                    return;
                }
                let Some(sized) = guard.sized.as_mut() else {
                    return;
                };
                if sized.fast_path.warm_state() != WarmSlabState::InProgress {
                    return;
                }
                if let Some(warm_slab) = sized.warm_slab.as_mut() {
                    let mut warmup_failed = false;
                    if pin_pooled_slabs
                        && warm_slab
                            .with_aligned_slice_mut(|aligned| {
                                runtime_ops.pin_slab(SlabKind::Warm, aligned)
                            })
                            .is_err()
                    {
                        warmup_failed = true;
                    }
                    if !warmup_failed {
                        warm_slab.pinned = pin_pooled_slabs;
                        if cuda_register && runtime_ops.cuda_available() {
                            if warm_slab
                                .with_aligned_slice_mut(|aligned| {
                                    runtime_ops.cuda_register_slab(SlabKind::Warm, aligned)
                                })
                                .is_err()
                            {
                                warmup_failed = true;
                            } else {
                                warm_slab.cuda_registered = true;
                                sized
                                    .fast_path
                                    .warm_cuda_registered
                                    .store(true, Ordering::Release);
                            }
                        }
                    }
                    if warmup_failed {
                        sized
                            .fast_path
                            .warm_cuda_registered
                            .store(false, Ordering::Release);
                        sized
                            .fast_path
                            .set_warm_state(WarmSlabState::FailedDegraded);
                    } else {
                        sized.fast_path.set_warm_state(WarmSlabState::Ready);
                    }
                } else {
                    sized.fast_path.set_warm_state(WarmSlabState::NotStarted);
                }
            }) {
            Ok(handle) => handle,
            Err(_) => {
                let mut guard = self.lock_state()?;
                if let Some(sized) = guard.sized.as_mut() {
                    sized
                        .fast_path
                        .warm_cuda_registered
                        .store(false, Ordering::Release);
                    sized
                        .fast_path
                        .set_warm_state(WarmSlabState::FailedDegraded);
                }
                return Ok(());
            }
        };
        let mut guard = self.lock_state()?;
        if guard.shutdown {
            drop(guard);
            let _ = warmup_thread.join();
            return Ok(());
        }
        guard.warmup_thread = Some(warmup_thread);
        Ok(())
    }

    #[cfg(test)]
    fn debug_snapshot(&self) -> Result<DebugPoolSnapshot, SyncWriteError> {
        let state = self.lock_state()?;
        let Some(sized) = state.sized.as_ref() else {
            return Err(SyncWriteError::PoolInitialization {
                message: "pool not initialized for debug snapshot".to_string(),
            });
        };
        let warm_buffer_ranges = sized
            .warm_slab
            .as_ref()
            .map(|warm| warm.buffer_ranges.clone())
            .unwrap_or_default();
        let warm_slab_backing_bytes = sized
            .warm_slab
            .as_ref()
            .map(SlabBacking::total_bytes)
            .transpose()?
            .unwrap_or(0);
        let warm_pinned = sized
            .warm_slab
            .as_ref()
            .map(|warm| warm.pinned)
            .unwrap_or(false);
        let warm_cuda_registered = sized
            .warm_slab
            .as_ref()
            .map(|warm| warm.cuda_registered)
            .unwrap_or(false);

        Ok(DebugPoolSnapshot {
            pooled_buffer_bytes: sized.fast_path.pooled_buffer_bytes,
            hot_slab_buffers: sized.fast_path.hot_slab_buffers,
            warm_slab_buffers: sized.fast_path.warm_slab_buffers,
            warm_state: sized.fast_path.warm_state(),
            hot_buffer_ranges: sized.hot_slab.buffer_ranges.clone(),
            warm_buffer_ranges,
            hot_slab_backing_bytes: sized.hot_slab.total_bytes()?,
            warm_slab_backing_bytes,
            hot_pinned: sized.hot_slab.pinned,
            warm_pinned,
            hot_cuda_registered: sized.hot_slab.cuda_registered,
            warm_cuda_registered,
            transient_inflight_bytes: self.transient_inflight_bytes.load(Ordering::Acquire),
            transient_inflight_count: self.transient_inflight_count.load(Ordering::Acquire),
            transient_total_acquired: self.transient_total_acquired.load(Ordering::Acquire),
            transient_total_released: self.transient_total_released.load(Ordering::Acquire),
        })
    }

    #[cfg(test)]
    fn debug_wait_for_warm_ready(&self, timeout: Duration) -> Result<(), SyncWriteError> {
        let start = Instant::now();
        loop {
            {
                let state = self.lock_state()?;
                let Some(sized) = state.sized.as_ref() else {
                    return Err(SyncWriteError::PoolInitialization {
                        message: "pool not initialized while waiting warm ready".to_string(),
                    });
                };
                if matches!(
                    sized.fast_path.warm_state(),
                    WarmSlabState::Ready
                        | WarmSlabState::NotStarted
                        | WarmSlabState::FailedDegraded
                ) {
                    return Ok(());
                }
            }
            if start.elapsed() >= timeout {
                return Err(SyncWriteError::PoolInitialization {
                    message: "timed out waiting for warm slab to become ready".to_string(),
                });
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn pooled_bytes_within_budget(
        &self,
        pooled_buffer_stride: usize,
        hot_slab_buffers: usize,
        warm_slab_buffers: usize,
    ) -> Result<bool, SyncWriteError> {
        let total_buffers = hot_slab_buffers
            .checked_add(warm_slab_buffers)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "hot_slab_buffers + warm_slab_buffers overflowed usize".to_string(),
            })?;
        let pooled_bytes = total_buffers
            .checked_mul(pooled_buffer_stride)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "pooled byte budget computation overflowed usize".to_string(),
            })?;
        Ok(pooled_bytes <= self.config.max_pool_bytes)
    }

    fn compute_pool_layout(
        &self,
        hint: &FirstWriteSizingHint,
    ) -> Result<ComputedPoolLayout, SyncWriteError> {
        if self.config.max_pool_buffers == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "max_pool_buffers must be greater than 0".to_string(),
            });
        }
        if self.config.max_pool_bytes == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "max_pool_bytes must be greater than 0".to_string(),
            });
        }
        if self.config.first_write_sizing.min_hot_slab_buffers == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "min_hot_slab_buffers must be greater than 0".to_string(),
            });
        }
        if self.config.first_write_sizing.max_warm_to_hot_ratio == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "max_warm_to_hot_ratio must be greater than 0".to_string(),
            });
        }

        let pooled_buffer_bytes = match self.config.pool_buffer_bytes {
            SizeOverride::Fixed(size) => {
                if size == 0 {
                    return Err(SyncWriteError::PoolInitialization {
                        message: "pool_buffer_bytes must be greater than 0 when provided"
                            .to_string(),
                    });
                }
                size
            }
            SizeOverride::Auto => {
                match self.config.first_write_sizing.buffer_sizing_policy {
                    BufferSizingPolicy::FixedOnly => {
                        return Err(SyncWriteError::PoolInitialization {
                        message: "pool_buffer_bytes must be provided when buffer_sizing_policy=FixedOnly".to_string(),
                    });
                    }
                    BufferSizingPolicy::FirstWriteModelAwareAuto => {
                        let baseline_bytes = self.model_profile_baseline_bytes();
                        let required = hint.first_write_max_required_bytes.max(baseline_bytes);
                        self.align_up(required)?
                    }
                }
            }
        };
        let pooled_buffer_stride = self.align_up(pooled_buffer_bytes)?;

        let task_count = hint.first_write_task_count.max(1);
        let min_hot = self
            .config
            .first_write_sizing
            .min_hot_slab_buffers
            .min(self.config.max_pool_buffers)
            .max(1);

        let derived_hot = Self::clamp_usize(
            task_count.min(self.config.max_pool_buffers),
            min_hot,
            self.config.max_pool_buffers,
        );
        let derived_warm_upper =
            derived_hot.saturating_mul(self.config.first_write_sizing.max_warm_to_hot_ratio);
        let derived_warm = if derived_warm_upper == 0 {
            0
        } else {
            Self::clamp_usize(
                task_count.saturating_sub(derived_hot),
                derived_hot.min(derived_warm_upper),
                derived_warm_upper,
            )
        };

        let hot_explicit = !self.config.hot_slab_buffers.is_auto();
        let warm_explicit = !self.config.warm_slab_buffers.is_auto();
        let mut hot_slab_buffers = self
            .config
            .hot_slab_buffers
            .fixed_value()
            .unwrap_or(derived_hot);
        let mut warm_slab_buffers = self
            .config
            .warm_slab_buffers
            .fixed_value()
            .unwrap_or(derived_warm);

        if hot_slab_buffers == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "hot_slab_buffers must be greater than 0".to_string(),
            });
        }
        if warm_explicit && warm_slab_buffers == 0 {
            return Err(SyncWriteError::PoolInitialization {
                message: "warm_slab_buffers must be greater than 0 when provided".to_string(),
            });
        }

        let total_buffers = hot_slab_buffers
            .checked_add(warm_slab_buffers)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "hot_slab_buffers + warm_slab_buffers overflowed usize".to_string(),
            })?;
        if total_buffers > self.config.max_pool_buffers {
            if hot_explicit && warm_explicit {
                return Err(SyncWriteError::PoolInitialization {
                    message: "hot_slab_buffers + warm_slab_buffers exceeds max_pool_buffers"
                        .to_string(),
                });
            }
            if !warm_explicit {
                warm_slab_buffers = self
                    .config
                    .max_pool_buffers
                    .saturating_sub(hot_slab_buffers);
            } else {
                let max_hot = self
                    .config
                    .max_pool_buffers
                    .saturating_sub(warm_slab_buffers);
                if max_hot < min_hot {
                    return Err(SyncWriteError::PoolInitialization {
                        message: "cannot satisfy min_hot_slab_buffers under max_pool_buffers cap"
                            .to_string(),
                    });
                }
                hot_slab_buffers = max_hot;
            }
        }

        while !self.pooled_bytes_within_budget(
            pooled_buffer_stride,
            hot_slab_buffers,
            warm_slab_buffers,
        )? {
            if !warm_explicit && warm_slab_buffers > 0 {
                warm_slab_buffers -= 1;
                continue;
            }
            if !hot_explicit && hot_slab_buffers > min_hot {
                hot_slab_buffers -= 1;
                continue;
            }
            return Err(SyncWriteError::PoolInitialization {
                message: format!(
                    "pooled memory budget exceeded: pooled_buffer_stride={} pooled_buffer_bytes={} hot_slab_buffers={} warm_slab_buffers={} max_pool_bytes={}",
                    pooled_buffer_stride,
                    pooled_buffer_bytes,
                    hot_slab_buffers,
                    warm_slab_buffers,
                    self.config.max_pool_bytes
                ),
            });
        }

        Ok(ComputedPoolLayout {
            pooled_buffer_bytes,
            pooled_buffer_stride,
            hot_slab_buffers,
            warm_slab_buffers,
        })
    }
}

impl BufferPool for MemoryBufferPool {
    fn initialize_if_needed(&self, hint: &FirstWriteSizingHint) -> Result<(), SyncWriteError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        let mut state = self.lock_state()?;
        if state.shutdown {
            return Err(SyncWriteError::ObjectClosed);
        }
        if state.initialized {
            return Ok(());
        }

        let layout = self.compute_pool_layout(hint)?;
        let total_buffers = layout
            .hot_slab_buffers
            .checked_add(layout.warm_slab_buffers)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "hot_slab_buffers + warm_slab_buffers overflowed usize".to_string(),
            })?;
        let total_buffers_u32 =
            u32::try_from(total_buffers).map_err(|_| SyncWriteError::PoolInitialization {
                message: "total pooled buffers exceeded u32 handle range".to_string(),
            })?;
        let hot_slab_buffers_u32 = u32::try_from(layout.hot_slab_buffers).map_err(|_| {
            SyncWriteError::PoolInitialization {
                message: "hot_slab_buffers exceeded u32 handle range".to_string(),
            }
        })?;

        let first_buffer_id = state.next_buffer_id;
        let next_buffer_id = first_buffer_id
            .checked_add(total_buffers_u32)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "pooled buffer handle id overflow".to_string(),
            })?;

        let warm_start_id = first_buffer_id
            .checked_add(hot_slab_buffers_u32)
            .ok_or_else(|| SyncWriteError::PoolInitialization {
                message: "warm slab start id overflow".to_string(),
            })?;
        let hot_id_max =
            warm_start_id
                .checked_sub(1)
                .ok_or_else(|| SyncWriteError::PoolInitialization {
                    message: "failed to compute hot slab id range".to_string(),
                })?;

        let hot_cuda_registration_enabled =
            self.config.cuda_register_pool_if_available && self.config.cuda_register_each_slab_once;
        let will_cuda_register_hot =
            hot_cuda_registration_enabled && self.runtime_ops.cuda_available();
        let hot_slab = self.build_slab_backing(
            layout.hot_slab_buffers,
            layout.pooled_buffer_bytes,
            layout.pooled_buffer_stride,
            false,
            false,
        )?;
        let mut hot_slab = hot_slab;
        if self.config.pin_pooled_slabs
            && self.runtime_ops.can_pin_memory()
            && !will_cuda_register_hot
        {
            match hot_slab
                .with_aligned_slice_mut(|aligned| self.runtime_ops.pin_slab(SlabKind::Hot, aligned))
            {
                Ok(()) => {
                    hot_slab.pinned = true;
                }
                Err(_) => {
                    eprintln!(
                        "[e2s_zarr_io] pin_pooled_slabs=true and mlock() probe passed, \
                         but hot slab pinning failed (slab may exceed RLIMIT_MEMLOCK); \
                         continuing without pinning"
                    );
                }
            }
        } else if self.config.pin_pooled_slabs && !will_cuda_register_hot {
            eprintln!(
                "[e2s_zarr_io] pin_pooled_slabs=true but mlock() is not available; \
                 skipping slab pinning (CAP_IPC_LOCK or RLIMIT_MEMLOCK may be restricted)"
            );
        }
        // If will_cuda_register_hot: silent skip — cudaHostRegister pins instead
        if hot_cuda_registration_enabled && self.runtime_ops.cuda_available() {
            match hot_slab.with_aligned_slice_mut(|aligned| {
                self.runtime_ops.cuda_register_slab(SlabKind::Hot, aligned)
            }) {
                Ok(()) => {
                    hot_slab.cuda_registered = true;
                }
                Err(SyncWriteError::CudaSlabRegistrationFailed) => {
                    // Registration is best-effort for hot slab: degrade to unregistered
                    // pooled buffers so host-only paths remain available.
                    hot_slab.cuda_registered = false;
                }
                Err(other) => return Err(other),
            }
        }
        let warm_slab = if layout.warm_slab_buffers > 0 {
            Some(self.build_slab_backing(
                layout.warm_slab_buffers,
                layout.pooled_buffer_bytes,
                layout.pooled_buffer_stride,
                false,
                false,
            )?)
        } else {
            None
        };

        let free_hot_ids = self.build_id_queue(
            (first_buffer_id..warm_start_id).rev(),
            layout.hot_slab_buffers,
            "hot_slab_free_ids",
        )?;
        let free_warm_ids = if layout.warm_slab_buffers > 0 {
            Some(self.build_id_queue(
                (warm_start_id..next_buffer_id).rev(),
                layout.warm_slab_buffers,
                "warm_slab_free_ids",
            )?)
        } else {
            None
        };
        let warm_state = if layout.warm_slab_buffers > 0 {
            WarmSlabState::InProgress
        } else {
            WarmSlabState::NotStarted
        };
        let hot_slot_locks = (0..layout.hot_slab_buffers)
            .map(|_| Arc::new(Mutex::new(())))
            .collect::<Vec<_>>();
        let warm_slot_locks = (0..layout.warm_slab_buffers)
            .map(|_| Arc::new(Mutex::new(())))
            .collect::<Vec<_>>();
        let leased_slots = (0..total_buffers)
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>();
        let fast_path = Arc::new(FastPoolState {
            pooled_buffer_bytes: layout.pooled_buffer_bytes,
            hot_slab_buffers: layout.hot_slab_buffers,
            warm_slab_buffers: layout.warm_slab_buffers,
            hot_start_id: first_buffer_id,
            hot_id_max,
            warm_start_id: if layout.warm_slab_buffers > 0 {
                Some(warm_start_id)
            } else {
                None
            },
            hot_buffer_ranges: hot_slab.buffer_ranges.clone(),
            warm_buffer_ranges: warm_slab
                .as_ref()
                .map(|slab| slab.buffer_ranges.clone())
                .unwrap_or_default(),
            hot_slot_locks,
            warm_slot_locks,
            hot_slab_bytes: hot_slab.bytes_arc(),
            warm_slab_bytes: warm_slab.as_ref().map(SlabBacking::bytes_arc),
            hot_slab_base_addr: hot_slab.base_addr,
            hot_slab_len: hot_slab.total_len,
            warm_slab_base_addr: warm_slab.as_ref().map(|slab| slab.base_addr),
            warm_slab_len: warm_slab.as_ref().map(|slab| slab.total_len),
            free_hot_ids,
            free_warm_ids,
            leased_slots,
            warm_state: AtomicU8::new(FastPoolState::encode_warm_state(warm_state)),
            hot_cuda_registered: AtomicBool::new(hot_slab.cuda_registered),
            warm_cuda_registered: AtomicBool::new(false),
        });
        self.fast_path.set(Arc::clone(&fast_path)).map_err(|_| {
            SyncWriteError::PoolInitialization {
                message: "buffer pool fast path has already been initialized".to_string(),
            }
        })?;

        state.sized = Some(SizedPoolState {
            fast_path,
            hot_slab,
            warm_slab,
        });
        state.next_buffer_id = next_buffer_id;
        state.initialized = true;
        let should_spawn_warmup = layout.warm_slab_buffers > 0;
        drop(state);
        if should_spawn_warmup {
            self.spawn_warmup_task_if_needed()?;
        }
        Ok(())
    }

    fn acquire(&self, required_bytes: usize) -> Result<BufferLease, SyncWriteError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        if self.fast_path.get().is_none() {
            self.initialize_if_needed(&FirstWriteSizingHint {
                first_write_task_count: 1,
                first_write_max_required_bytes: required_bytes,
            })?;
        }
        if self.shutdown.load(Ordering::Acquire) {
            return Err(SyncWriteError::ObjectClosed);
        }
        let Some(fast_path) = self.fast_path.get() else {
            return Err(SyncWriteError::PoolInitialization {
                message: "buffer pool acquired without initialized fast path".to_string(),
            });
        };

        if required_bytes <= fast_path.pooled_buffer_bytes {
            if let Some(lease) = self.try_acquire_hot_lease(fast_path)? {
                return Ok(lease);
            }
            if let Some(lease) = self.try_acquire_warm_lease(fast_path)? {
                return Ok(lease);
            }
        }

        if let Some(limit) = self.config.max_transient_buffer_bytes {
            if required_bytes > limit {
                return Err(SyncWriteError::TransientAllocationLimitExceeded {
                    requested_bytes: required_bytes,
                    limit_bytes: limit,
                });
            }
        }
        self.reserve_transient_allocation(required_bytes)?;

        Ok(BufferLease::Transient(TransientBuffer::new(required_bytes)))
    }

    fn release(&self, lease: BufferLease) {
        let handle = match lease {
            BufferLease::Transient(buffer) => {
                self.release_transient_allocation(buffer.len());
                self.lease_wait_cv.notify_all();
                return;
            }
            BufferLease::Pooled(handle) => handle,
        };
        if self.shutdown.load(Ordering::Acquire) {
            return;
        }
        let Some(fast_path) = self.fast_path.get() else {
            return;
        };
        let id = handle.id();
        let Ok((slot_index, released)) = self.try_mark_released(fast_path, id) else {
            return;
        };
        if !released {
            return;
        }
        let push_result = if id <= fast_path.hot_id_max {
            fast_path.free_hot_ids.push(id)
        } else {
            match fast_path.free_warm_ids.as_ref() {
                Some(queue) => queue.push(id),
                None => Err(id),
            }
        };
        if push_result.is_err() {
            // Keep slot conservative (leased=true) if push-back fails unexpectedly.
            let _ = fast_path.leased_slots[slot_index].compare_exchange(
                false,
                true,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        self.lease_wait_cv.notify_all();
    }

    fn warmup_status(&self) -> PoolWarmupStatus {
        if self.shutdown.load(Ordering::Acquire) {
            return PoolWarmupStatus::default();
        }
        let Some(fast_path) = self.fast_path.get() else {
            return PoolWarmupStatus::default();
        };
        let warm_state = if fast_path.warm_slab_buffers == 0 {
            WarmSlabState::NotStarted
        } else {
            fast_path.warm_state()
        };
        PoolWarmupStatus {
            hot_ready: fast_path.hot_slab_buffers > 0,
            warm_state,
        }
    }

    fn wait_pooled_leases_returned(&self, timeout_seconds: f64) -> Result<(), SyncWriteError> {
        if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "wait_pooled_leases_returned() timeout_seconds must be finite and > 0, got {timeout_seconds}"
                ),
            });
        }
        if self.shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
        let mut wait_guard =
            self.lease_wait_lock
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "buffer pool lease wait lock poisoned".to_string(),
                })?;
        loop {
            let outstanding = self.outstanding_total_leases();
            if outstanding == 0 {
                return Ok(());
            }

            let now = Instant::now();
            if now >= deadline {
                return Err(SyncWriteError::LeaseReturnTimeout {
                    outstanding_leases: outstanding,
                });
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next_guard, wait_result) = self
                .lease_wait_cv
                .wait_timeout(wait_guard, remaining)
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "buffer pool lease wait lock poisoned while waiting".to_string(),
                })?;
            wait_guard = next_guard;
            if wait_result.timed_out() {
                let outstanding = self.outstanding_total_leases();
                if outstanding > 0 {
                    return Err(SyncWriteError::LeaseReturnTimeout {
                        outstanding_leases: outstanding,
                    });
                }
                return Ok(());
            }
        }
    }

    fn shutdown(&self) -> Result<(), SyncWriteError> {
        if self.shutdown.swap(true, Ordering::AcqRel) {
            self.lease_wait_cv.notify_all();
            return Ok(());
        }
        let warmup_thread = {
            let mut state = self.lock_state()?;
            if state.shutdown {
                self.lease_wait_cv.notify_all();
                return Ok(());
            }
            state.shutdown = true;
            state.warmup_thread.take()
        };
        let warmup_join_err = warmup_thread.and_then(|handle| {
            handle
                .join()
                .err()
                .map(|_| SyncWriteError::ContractViolation {
                    message: "warm slab warmup thread panicked during shutdown".to_string(),
                })
        });
        let mut state = self.lock_state()?;
        if let Some(sized) = state.sized.as_mut() {
            if let Some(warm_slab) = sized.warm_slab.as_mut() {
                if warm_slab.cuda_registered {
                    warm_slab
                        .with_aligned_slice_mut(|aligned| {
                            self.runtime_ops
                                .cuda_unregister_slab(SlabKind::Warm, aligned)
                        })
                        .map_err(|_| SyncWriteError::CudaSlabRegistrationFailed)?;
                    warm_slab.cuda_registered = false;
                    sized
                        .fast_path
                        .warm_cuda_registered
                        .store(false, Ordering::Release);
                }
                if warm_slab.pinned {
                    warm_slab
                        .with_aligned_slice_mut(|aligned| {
                            self.runtime_ops.unpin_slab(SlabKind::Warm, aligned)
                        })
                        .map_err(|_| SyncWriteError::SlabPinningFailed)?;
                    warm_slab.pinned = false;
                }
            }
            if sized.hot_slab.cuda_registered {
                sized
                    .hot_slab
                    .with_aligned_slice_mut(|aligned| {
                        self.runtime_ops
                            .cuda_unregister_slab(SlabKind::Hot, aligned)
                    })
                    .map_err(|_| SyncWriteError::CudaSlabRegistrationFailed)?;
                sized.hot_slab.cuda_registered = false;
                sized
                    .fast_path
                    .hot_cuda_registered
                    .store(false, Ordering::Release);
            }
            if sized.hot_slab.pinned {
                sized
                    .hot_slab
                    .with_aligned_slice_mut(|aligned| {
                        self.runtime_ops.unpin_slab(SlabKind::Hot, aligned)
                    })
                    .map_err(|_| SyncWriteError::SlabPinningFailed)?;
                sized.hot_slab.pinned = false;
            }
            sized.fast_path.set_warm_state(WarmSlabState::NotStarted);
        }
        state.sized = None;
        state.initialized = false;
        state.shutdown = true;
        self.lease_wait_cv.notify_all();
        if let Some(err) = warmup_join_err {
            return Err(err);
        }
        Ok(())
    }

    fn supports_early_init(&self) -> bool {
        // Early init is safe only when ALL pool dimensions are explicitly fixed.
        // Buffer sizing must NOT be Auto because initialize_if_needed is
        // idempotent — once initialized with a fallback size, it cannot be
        // re-initialized with the correct first-write size.
        !self.config.pool_buffer_bytes.is_auto()
            && !self.config.hot_slab_buffers.is_auto()
            && !self.config.warm_slab_buffers.is_auto()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use crate::core::contracts::BufferPool;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        BufferLease, BufferPoolConfig, BufferSizingPolicy, FirstWriteSizingConfig,
        FirstWriteSizingHint, ModelProfileHint, SizeOverride, WarmSlabState,
    };

    use super::{
        DefaultRuntimeMemoryOps, MemoryBufferPool, RuntimeMemoryOps, SlabBacking, SlabKind,
    };

    #[derive(Default)]
    struct FakeRuntimeOps {
        cuda_available: AtomicBool,
        can_pin_memory: AtomicBool,
        fail_hot_pin: AtomicBool,
        fail_warm_pin: AtomicBool,
        fail_hot_cuda_register: AtomicBool,
        fail_warm_cuda_register: AtomicBool,
        panic_warm_pin: AtomicBool,
        pin_hot_calls: std::sync::atomic::AtomicUsize,
        pin_warm_calls: std::sync::atomic::AtomicUsize,
        unpin_hot_calls: std::sync::atomic::AtomicUsize,
        unpin_warm_calls: std::sync::atomic::AtomicUsize,
        register_hot_calls: std::sync::atomic::AtomicUsize,
        register_warm_calls: std::sync::atomic::AtomicUsize,
        unregister_hot_calls: std::sync::atomic::AtomicUsize,
        unregister_warm_calls: std::sync::atomic::AtomicUsize,
    }

    impl RuntimeMemoryOps for FakeRuntimeOps {
        fn cuda_available(&self) -> bool {
            self.cuda_available.load(Ordering::Relaxed)
        }

        fn can_pin_memory(&self) -> bool {
            self.can_pin_memory.load(Ordering::Relaxed)
        }

        fn pin_slab(&self, slab: SlabKind, _bytes: &mut [u8]) -> Result<(), SyncWriteError> {
            match slab {
                SlabKind::Hot => {
                    self.pin_hot_calls.fetch_add(1, Ordering::Relaxed);
                    if self.fail_hot_pin.load(Ordering::Relaxed) {
                        return Err(SyncWriteError::SlabPinningFailed);
                    }
                }
                SlabKind::Warm => {
                    self.pin_warm_calls.fetch_add(1, Ordering::Relaxed);
                    if self.panic_warm_pin.load(Ordering::Relaxed) {
                        panic!("simulated warm slab pin panic");
                    }
                    if self.fail_warm_pin.load(Ordering::Relaxed) {
                        return Err(SyncWriteError::SlabPinningFailed);
                    }
                }
            }
            Ok(())
        }

        fn unpin_slab(&self, slab: SlabKind, _bytes: &mut [u8]) -> Result<(), SyncWriteError> {
            match slab {
                SlabKind::Hot => {
                    self.unpin_hot_calls.fetch_add(1, Ordering::Relaxed);
                }
                SlabKind::Warm => {
                    self.unpin_warm_calls.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        }

        fn cuda_register_slab(
            &self,
            slab: SlabKind,
            _bytes: &mut [u8],
        ) -> Result<(), SyncWriteError> {
            match slab {
                SlabKind::Hot => {
                    self.register_hot_calls.fetch_add(1, Ordering::Relaxed);
                    if self.fail_hot_cuda_register.load(Ordering::Relaxed) {
                        return Err(SyncWriteError::CudaSlabRegistrationFailed);
                    }
                }
                SlabKind::Warm => {
                    self.register_warm_calls.fetch_add(1, Ordering::Relaxed);
                    if self.fail_warm_cuda_register.load(Ordering::Relaxed) {
                        return Err(SyncWriteError::CudaSlabRegistrationFailed);
                    }
                }
            }
            Ok(())
        }

        fn cuda_unregister_slab(
            &self,
            slab: SlabKind,
            _bytes: &mut [u8],
        ) -> Result<(), SyncWriteError> {
            match slab {
                SlabKind::Hot => {
                    self.unregister_hot_calls.fetch_add(1, Ordering::Relaxed);
                }
                SlabKind::Warm => {
                    self.unregister_warm_calls.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        }
    }

    fn default_hint() -> FirstWriteSizingHint {
        FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 1024,
        }
    }

    #[test]
    fn default_runtime_memory_ops_empty_pin_and_unpin_are_noops() {
        let runtime = DefaultRuntimeMemoryOps::default();
        let mut empty = [];
        runtime
            .pin_slab(SlabKind::Hot, &mut empty)
            .expect("empty-slice pin should be a no-op");
        runtime
            .unpin_slab(SlabKind::Warm, &mut empty)
            .expect("empty-slice unpin should be a no-op");
    }

    #[test]
    fn slab_backing_rejects_out_of_bounds_aligned_range() {
        let backing = SlabBacking {
            bytes: Arc::new(Mutex::new(vec![0_u8; 8])),
            base_addr: 0,
            total_len: 8,
            aligned_start: 6,
            aligned_len: 4,
            buffer_ranges: Vec::new(),
            pinned: false,
            cuda_registered: false,
        };
        let err = backing
            .with_aligned_slice_mut(|_| Ok(()))
            .expect_err("out-of-bounds aligned range must fail");
        assert!(matches!(
            err,
            SyncWriteError::ContractViolation { ref message }
            if message.contains("aligned range is out of bounds")
        ));
    }

    #[test]
    fn build_slab_backing_with_zero_buffers_returns_empty_layout() {
        let pool = MemoryBufferPool::default();
        let slab = pool
            .build_slab_backing(0, 1024, 1024, false, false)
            .expect("zero-buffer slab should be constructible");
        assert_eq!(slab.total_len, 0);
        assert_eq!(slab.aligned_start, 0);
        assert_eq!(slab.aligned_len, 0);
        assert!(slab.buffer_ranges.is_empty());
    }

    #[test]
    fn build_slab_backing_rejects_buffer_ranges_that_exceed_backing_window() {
        let pool = MemoryBufferPool::default();
        let err = pool
            .build_slab_backing(2, 8, 4, false, false)
            .expect_err("buffer size exceeding stride should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("buffer slice exceeds slab backing range")
        ));
    }

    #[test]
    fn build_id_queue_rejects_zero_capacity() {
        let pool = MemoryBufferPool::default();
        let err = pool
            .build_id_queue(0..1, 0, "unit_test")
            .expect_err("queue capacity 0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("capacity must be greater than 0")
        ));
    }

    #[test]
    fn slot_index_for_id_rejects_out_of_slot_range_id() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&default_hint())
            .expect("initialize should succeed");
        let fast_path = pool
            .fast_path
            .get()
            .expect("fast path should be initialized");
        let invalid_id = fast_path.hot_start_id + fast_path.leased_slots.len() as u32;
        let err = pool
            .slot_index_for_id(fast_path, invalid_id)
            .expect_err("slot lookup should fail for out-of-range id");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("exceeds allocated slot range")
        ));
    }

    #[test]
    fn try_acquire_hot_lease_skips_id_if_slot_is_already_marked_leased() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&default_hint())
            .expect("initialize should succeed");
        let fast_path = pool
            .fast_path
            .get()
            .expect("fast path should be initialized");

        let id = fast_path
            .free_hot_ids
            .pop()
            .expect("hot id queue should contain one id");
        fast_path.leased_slots[0].store(true, Ordering::Release);
        fast_path
            .free_hot_ids
            .push(id)
            .expect("re-queue hot id should succeed");

        let lease = pool
            .try_acquire_hot_lease(fast_path)
            .expect("hot acquire helper should not error");
        assert!(
            lease.is_none(),
            "helper should skip ids that are already marked leased"
        );
    }

    #[test]
    fn try_acquire_warm_lease_returns_none_when_queue_missing_even_if_ready() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&default_hint())
            .expect("initialize should succeed");
        let fast_path = pool
            .fast_path
            .get()
            .expect("fast path should be initialized");
        fast_path.set_warm_state(WarmSlabState::Ready);

        let lease = pool
            .try_acquire_warm_lease(fast_path)
            .expect("warm acquire helper should not error");
        assert!(
            lease.is_none(),
            "warm helper should return None when warm queue is absent"
        );
    }

    #[test]
    fn try_acquire_warm_lease_skips_id_if_warm_slot_is_already_marked_leased() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(4096),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warm slab should become ready");
        let fast_path = pool
            .fast_path
            .get()
            .expect("fast path should be initialized");
        let warm_queue = fast_path
            .free_warm_ids
            .as_ref()
            .expect("warm queue should exist");
        let warm_id = warm_queue
            .pop()
            .expect("warm id queue should contain one id");
        let slot_index =
            usize::try_from(warm_id - fast_path.hot_start_id).expect("slot index should fit usize");
        fast_path.leased_slots[slot_index].store(true, Ordering::Release);
        warm_queue
            .push(warm_id)
            .expect("re-queue warm id should succeed");

        let lease = pool
            .try_acquire_warm_lease(fast_path)
            .expect("warm acquire helper should not error");
        assert!(
            lease.is_none(),
            "helper should skip warm ids that are already marked leased"
        );
    }

    #[test]
    fn release_second_call_on_same_pooled_handle_is_noop() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        let lease = pool
            .acquire(64)
            .expect("first pooled acquire should succeed");
        let lease_clone = lease.clone();
        pool.release(lease);
        pool.release(lease_clone);
        assert_eq!(pool.outstanding_pooled_leases(), 0);
    }

    #[test]
    fn release_keeps_slot_marked_leased_when_queue_push_back_fails() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        let lease = pool.acquire(64).expect("pooled acquire should succeed");
        let id = match &lease {
            BufferLease::Pooled(handle) => handle.id(),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        };
        let fast_path = pool
            .fast_path
            .get()
            .expect("fast path should be initialized");
        fast_path
            .free_hot_ids
            .push(id)
            .expect("manual queue fill should succeed");

        pool.release(lease);
        assert_eq!(
            pool.outstanding_pooled_leases(),
            1,
            "slot should remain conservatively marked leased after push-back failure"
        );
    }

    #[test]
    fn compute_pool_layout_rejects_zero_max_pool_buffers() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            max_pool_buffers: 0,
            ..BufferPoolConfig::default()
        });
        let err = pool
            .compute_pool_layout(&default_hint())
            .expect_err("max_pool_buffers=0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("max_pool_buffers")
        ));
    }

    #[test]
    fn compute_pool_layout_rejects_zero_max_pool_bytes() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            max_pool_bytes: 0,
            ..BufferPoolConfig::default()
        });
        let err = pool
            .compute_pool_layout(&default_hint())
            .expect_err("max_pool_bytes=0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("max_pool_bytes")
        ));
    }

    #[test]
    fn compute_pool_layout_rejects_zero_min_hot_slab_buffers() {
        let mut config = BufferPoolConfig::default();
        config.first_write_sizing.min_hot_slab_buffers = 0;
        let pool = MemoryBufferPool::new(config);
        let err = pool
            .compute_pool_layout(&default_hint())
            .expect_err("min_hot_slab_buffers=0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("min_hot_slab_buffers")
        ));
    }

    #[test]
    fn compute_pool_layout_rejects_zero_max_warm_to_hot_ratio() {
        let mut config = BufferPoolConfig::default();
        config.first_write_sizing.max_warm_to_hot_ratio = 0;
        let pool = MemoryBufferPool::new(config);
        let err = pool
            .compute_pool_layout(&default_hint())
            .expect_err("max_warm_to_hot_ratio=0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("max_warm_to_hot_ratio")
        ));
    }

    #[test]
    fn compute_pool_layout_rejects_zero_pool_buffer_bytes_override() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(0),
            ..BufferPoolConfig::default()
        });
        let err = pool
            .compute_pool_layout(&default_hint())
            .expect_err("pool_buffer_bytes=0 should fail");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("pool_buffer_bytes")
        ));
    }

    #[test]
    fn initialize_if_needed_rejects_when_state_shutdown_flag_is_set() {
        let pool = MemoryBufferPool::default();
        {
            let mut state = pool.state.lock().expect("state lock should be valid");
            state.shutdown = true;
        }
        let err = pool
            .initialize_if_needed(&default_hint())
            .expect_err("initialize should reject state-level shutdown");
        assert!(matches!(err, SyncWriteError::ObjectClosed));
    }

    #[test]
    fn initialize_if_needed_rejects_after_shutdown() {
        let pool = MemoryBufferPool::default();
        pool.shutdown().expect("shutdown should succeed");
        let err = pool
            .initialize_if_needed(&default_hint())
            .expect_err("initialize should reject atomic shutdown");
        assert!(matches!(err, SyncWriteError::ObjectClosed));
    }

    #[test]
    fn warmup_status_returns_default_after_shutdown() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(1024),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&default_hint())
            .expect("initialize should succeed");
        pool.shutdown().expect("shutdown should succeed");
        let status = pool.warmup_status();
        assert!(!status.hot_ready);
        assert_eq!(status.warm_state, WarmSlabState::NotStarted);
    }

    #[test]
    fn warmup_status_reports_not_started_when_no_warm_slab_is_configured() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(1024),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&default_hint())
            .expect("initialize should succeed");
        let status = pool.warmup_status();
        assert!(status.hot_ready);
        assert_eq!(status.warm_state, WarmSlabState::NotStarted);
    }

    #[test]
    fn wait_pooled_leases_returned_rejects_non_positive_timeout() {
        let pool = MemoryBufferPool::default();
        for timeout in [0.0, -1.0] {
            let err = pool
                .wait_pooled_leases_returned(timeout)
                .expect_err("non-positive timeout should be rejected");
            assert!(matches!(
                err,
                SyncWriteError::Validation { ref message }
                if message.contains("timeout_seconds")
            ));
        }
    }

    #[test]
    fn wait_pooled_leases_returned_returns_ok_after_shutdown() {
        let pool = MemoryBufferPool::default();
        pool.shutdown().expect("shutdown should succeed");
        pool.wait_pooled_leases_returned(0.1)
            .expect("wait should no-op once shutdown");
    }

    #[test]
    fn wait_pooled_leases_returned_can_return_ok_after_timeout_when_outstanding_becomes_zero() {
        let pool = Arc::new(MemoryBufferPool::default());
        pool.reserve_transient_allocation(64)
            .expect("manual transient reservation should succeed");
        let pool_for_wait = Arc::clone(&pool);
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            tx.send(pool_for_wait.wait_pooled_leases_returned(0.05))
                .expect("waiter result should be sendable");
        });

        std::thread::sleep(Duration::from_millis(20));
        // Intentionally bypass `release(BufferLease::Transient)` to avoid CV notify and
        // force the timed-out path to re-check outstanding leases.
        pool.release_transient_allocation(64);

        let result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("waiter should finish");
        assert!(
            result.is_ok(),
            "wait should return Ok when timeout fires after outstanding reaches zero"
        );
        waiter.join().expect("waiter thread should join");
    }

    #[test]
    fn shutdown_returns_ok_when_runtime_state_is_already_marked_shutdown() {
        let pool = MemoryBufferPool::default();
        {
            let mut state = pool.state.lock().expect("state lock should be valid");
            state.shutdown = true;
        }
        pool.shutdown()
            .expect("shutdown should tolerate pre-marked runtime shutdown state");
    }

    #[test]
    fn shutdown_skips_unpin_calls_when_pinning_is_disabled() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(1024),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                pin_pooled_slabs: false,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 1024,
        })
        .expect("initialize should succeed with pinning disabled");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should settle");
        pool.shutdown().expect("shutdown should succeed");

        assert_eq!(runtime.pin_hot_calls.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.pin_warm_calls.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.unpin_hot_calls.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.unpin_warm_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn debug_snapshot_rejects_before_initialization() {
        let pool = MemoryBufferPool::default();
        let err = pool
            .debug_snapshot()
            .expect_err("snapshot should fail before initialization");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("not initialized")
        ));
    }

    #[test]
    fn debug_wait_for_warm_ready_rejects_before_initialization() {
        let pool = MemoryBufferPool::default();
        let err = pool
            .debug_wait_for_warm_ready(Duration::from_millis(10))
            .expect_err("wait helper should fail before initialization");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("not initialized")
        ));
    }

    #[test]
    fn debug_wait_for_warm_ready_times_out_while_warmup_is_in_progress() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(1024),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 1024,
        })
        .expect("initialize should succeed");
        let err = pool
            .debug_wait_for_warm_ready(Duration::from_millis(1))
            .expect_err("warmup should still be in progress for short timeout");
        assert!(matches!(
            err,
            SyncWriteError::PoolInitialization { ref message }
            if message.contains("timed out waiting for warm slab")
        ));
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should eventually settle");
    }

    #[test]
    fn spawn_warmup_task_joins_immediately_when_shutdown_is_already_set() {
        let pool = MemoryBufferPool::default();
        {
            let mut state = pool.state.lock().expect("state lock should be valid");
            state.shutdown = true;
        }
        pool.spawn_warmup_task_if_needed()
            .expect("spawn helper should gracefully join when shutdown is set");
    }

    #[test]
    fn returns_pooled_lease_when_request_fits_pool_capacity() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(1024),
            ..BufferPoolConfig::default()
        });
        let lease = pool.acquire(256).expect("acquire should succeed");
        match lease {
            BufferLease::Pooled(handle) => assert_eq!(handle.capacity_bytes(), 1024),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        }
    }

    #[test]
    fn returns_transient_lease_when_request_exceeds_pool_capacity() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(64),
            ..BufferPoolConfig::default()
        });
        let lease = pool.acquire(128).expect("acquire should succeed");
        match lease {
            BufferLease::Transient(buffer) => assert_eq!(buffer.len(), 128),
            BufferLease::Pooled(_) => panic!("expected transient lease"),
        }
    }

    #[test]
    fn enforces_transient_allocation_limit() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(64),
            max_transient_buffer_bytes: Some(96),
            ..BufferPoolConfig::default()
        });
        let err = pool
            .acquire(128)
            .expect_err("oversized transient should fail");
        assert!(matches!(
            err,
            SyncWriteError::TransientAllocationLimitExceeded {
                requested_bytes: 128,
                limit_bytes: 96
            }
        ));
    }

    #[test]
    fn enforces_transient_inflight_limit_and_allows_reacquire_after_release() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            max_transient_buffer_bytes: Some(512),
            max_inflight_transient_bytes: Some(384),
            ..BufferPoolConfig::default()
        });

        let pooled = pool.acquire(128).expect("pooled acquire should succeed");
        assert!(matches!(pooled, BufferLease::Pooled(_)));

        let transient_a = pool
            .acquire(256)
            .expect("first fallback transient should succeed");
        assert!(matches!(transient_a, BufferLease::Transient(_)));

        let err = pool
            .acquire(256)
            .expect_err("second fallback transient should fail by in-flight limit");
        assert!(matches!(
            err,
            SyncWriteError::TransientInFlightLimitExceeded {
                requested_bytes: 256,
                in_flight_bytes: 256,
                limit_bytes: 384
            }
        ));

        let snapshot_before_release = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(snapshot_before_release.transient_inflight_bytes, 256);
        assert_eq!(snapshot_before_release.transient_inflight_count, 1);
        assert_eq!(snapshot_before_release.transient_total_acquired, 1);
        assert_eq!(snapshot_before_release.transient_total_released, 0);

        pool.release(transient_a);
        let snapshot_after_release = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(snapshot_after_release.transient_inflight_bytes, 0);
        assert_eq!(snapshot_after_release.transient_inflight_count, 0);
        assert_eq!(snapshot_after_release.transient_total_acquired, 1);
        assert_eq!(snapshot_after_release.transient_total_released, 1);

        let transient_b = pool
            .acquire(256)
            .expect("fallback transient should succeed again after release");
        assert!(matches!(transient_b, BufferLease::Transient(_)));
        pool.release(transient_b);
        pool.release(pooled);

        let final_snapshot = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(final_snapshot.transient_inflight_bytes, 0);
        assert_eq!(final_snapshot.transient_inflight_count, 0);
        assert_eq!(final_snapshot.transient_total_acquired, 2);
        assert_eq!(final_snapshot.transient_total_released, 2);
    }

    #[test]
    fn tracks_multiple_transient_fallback_acquires_and_releases() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            max_transient_buffer_bytes: Some(512),
            max_inflight_transient_bytes: Some(512),
            ..BufferPoolConfig::default()
        });

        let pooled = pool.acquire(128).expect("pooled acquire should succeed");
        assert!(matches!(pooled, BufferLease::Pooled(_)));

        let mut transients = Vec::new();
        for _ in 0..4 {
            let lease = pool
                .acquire(128)
                .expect("fallback transient acquire should succeed");
            assert!(matches!(lease, BufferLease::Transient(_)));
            transients.push(lease);
        }

        let snapshot_inflight = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(snapshot_inflight.transient_inflight_count, 4);
        assert_eq!(snapshot_inflight.transient_inflight_bytes, 512);
        assert_eq!(snapshot_inflight.transient_total_acquired, 4);
        assert_eq!(snapshot_inflight.transient_total_released, 0);

        for lease in transients {
            pool.release(lease);
        }
        pool.release(pooled);

        let snapshot_released = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(snapshot_released.transient_inflight_count, 0);
        assert_eq!(snapshot_released.transient_inflight_bytes, 0);
        assert_eq!(snapshot_released.transient_total_acquired, 4);
        assert_eq!(snapshot_released.transient_total_released, 4);
    }

    #[test]
    fn exposes_warmup_status_before_and_after_initialization() {
        let pool = MemoryBufferPool::default();

        let initial = pool.warmup_status();
        assert!(!initial.hot_ready);
        assert_eq!(initial.warm_state, WarmSlabState::NotStarted);

        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 4,
            first_write_max_required_bytes: 512,
        })
        .expect("initialize should succeed");

        let initialized = pool.warmup_status();
        assert!(initialized.hot_ready);
        assert_ne!(initialized.warm_state, WarmSlabState::NotStarted);
    }

    #[test]
    fn auto_sizing_uses_model_baseline_and_alignment() {
        let mut config = BufferPoolConfig::default();
        config.first_write_sizing.model_profile_hint = Some(ModelProfileHint::GraphCastSmall);
        let pool = MemoryBufferPool::new(config);

        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 1024,
        })
        .expect("initialize should succeed");

        let lease = pool.acquire(1024).expect("acquire should succeed");
        match lease {
            BufferLease::Pooled(handle) => {
                assert_eq!(handle.capacity_bytes(), 262_144);
            }
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        }
    }

    #[test]
    fn reuses_same_pooled_handle_after_release() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });

        let first = pool.acquire(128).expect("first acquire should succeed");
        let first_id = match &first {
            BufferLease::Pooled(handle) => handle.id(),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        };
        pool.release(first);

        let second_id = match pool.acquire(128).expect("second acquire should succeed") {
            BufferLease::Pooled(handle) => handle.id(),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        };
        assert_eq!(first_id, second_id);
    }

    #[test]
    fn falls_back_to_transient_when_no_pooled_handles_are_available() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });

        let first = pool.acquire(128).expect("first acquire should succeed");
        assert!(matches!(first, BufferLease::Pooled(_)));

        let second = pool.acquire(128).expect("second acquire should succeed");
        assert!(
            matches!(second, BufferLease::Transient(_)),
            "expected transient lease once pooled handles are exhausted"
        );
    }

    #[test]
    fn acquire_fast_path_does_not_block_on_state_lock_once_initialized() {
        let pool = Arc::new(MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        }));
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 256,
        })
        .expect("initialize should succeed");

        let (tx, rx) = mpsc::channel();
        let pool_for_thread = Arc::clone(&pool);
        let state_guard = pool
            .state
            .lock()
            .expect("state lock should not be poisoned");
        let worker = std::thread::spawn(move || {
            let lease = pool_for_thread
                .acquire(128)
                .expect("acquire should succeed");
            tx.send(matches!(lease, BufferLease::Pooled(_)))
                .expect("worker should send acquire result");
            pool_for_thread.release(lease);
        });

        let acquired = rx
            .recv_timeout(Duration::from_millis(100))
            .expect("acquire fast path should not block on lifecycle state lock");
        assert!(acquired, "acquire should still return a pooled lease");
        drop(state_guard);
        worker.join().expect("worker thread should join");
    }

    #[test]
    fn release_fast_path_does_not_block_on_state_lock_once_initialized() {
        let pool = Arc::new(MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        }));
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 256,
        })
        .expect("initialize should succeed");
        let lease = pool.acquire(128).expect("acquire should succeed");

        let (tx, rx) = mpsc::channel();
        let pool_for_thread = Arc::clone(&pool);
        let state_guard = pool
            .state
            .lock()
            .expect("state lock should not be poisoned");
        let worker = std::thread::spawn(move || {
            pool_for_thread.release(lease);
            tx.send(())
                .expect("worker should notify when release returns");
        });

        rx.recv_timeout(Duration::from_millis(100))
            .expect("release fast path should not block on lifecycle state lock");
        drop(state_guard);
        worker.join().expect("worker thread should join");
    }

    #[test]
    fn pooled_leases_reference_stable_slab_slice_after_release_reacquire() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(128),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });

        let mut first = pool.acquire(64).expect("first acquire should succeed");
        let first_range = match &first {
            BufferLease::Pooled(handle) => handle.slab_range(),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        };
        first
            .write_from_host_bytes(&[0xAB; 64], 64)
            .expect("writing into pooled lease should succeed");
        pool.release(first);

        let second = pool.acquire(64).expect("second acquire should succeed");
        let second_range = match &second {
            BufferLease::Pooled(handle) => handle.slab_range(),
            BufferLease::Transient(_) => panic!("expected pooled lease"),
        };
        assert_eq!(
            first_range, second_range,
            "single-slot pool should return the same slab slice on reacquire"
        );
        assert_eq!(
            second
                .staged_bytes(64)
                .expect("staged bytes should read from pooled slab slice"),
            vec![0xAB; 64],
            "reacquired pooled lease should reference the same underlying slab slice bytes"
        );
        pool.release(second);
    }

    #[test]
    fn fixed_only_policy_requires_explicit_pool_buffer_bytes() {
        let config = BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Auto,
            first_write_sizing: FirstWriteSizingConfig {
                buffer_sizing_policy: BufferSizingPolicy::FixedOnly,
                ..Default::default()
            },
            ..Default::default()
        };
        let pool = MemoryBufferPool::new(config);

        let err = pool
            .initialize_if_needed(&FirstWriteSizingHint {
                first_write_task_count: 2,
                first_write_max_required_bytes: 1024,
            })
            .expect_err("fixed-only policy without explicit pool size should fail");
        assert!(matches!(err, SyncWriteError::PoolInitialization { .. }));
    }

    #[test]
    fn acquire_fails_after_shutdown() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 256,
        })
        .expect("initialize should succeed");
        pool.shutdown().expect("shutdown should succeed");

        let err = pool
            .acquire(128)
            .expect_err("acquire should fail after shutdown");
        assert!(matches!(err, SyncWriteError::ObjectClosed));
    }

    #[test]
    fn shutdown_is_idempotent() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(1024),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 1024,
        })
        .expect("initialize should succeed");

        pool.shutdown().expect("first shutdown should succeed");
        pool.shutdown()
            .expect("second shutdown should be treated as no-op success");

        let err = pool
            .acquire(128)
            .expect_err("acquire should remain closed after repeated shutdown");
        assert!(matches!(err, SyncWriteError::ObjectClosed));
    }

    #[test]
    fn wait_pooled_leases_returned_rejects_non_finite_timeout() {
        let pool = MemoryBufferPool::default();
        for timeout in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = pool
                .wait_pooled_leases_returned(timeout)
                .expect_err("non-finite timeout should be rejected");
            assert!(matches!(
                err,
                SyncWriteError::Validation { ref message }
                if message.contains("timeout_seconds")
            ));
        }
    }

    #[test]
    fn shutdown_remains_safe_when_release_races_with_shutdown() {
        let pool = Arc::new(MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        }));
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 256,
        })
        .expect("initialize should succeed");
        let lease = pool.acquire(128).expect("pooled lease should be acquired");

        let pool_for_shutdown = Arc::clone(&pool);
        let shutdown_handle = std::thread::spawn(move || pool_for_shutdown.shutdown());
        std::thread::sleep(Duration::from_millis(1));
        pool.release(lease);

        let shutdown_result = shutdown_handle
            .join()
            .expect("shutdown thread should not panic");
        assert!(
            shutdown_result.is_ok(),
            "shutdown should complete successfully during release race: {shutdown_result:?}"
        );
        let err = pool
            .acquire(64)
            .expect_err("pool should be closed after shutdown");
        assert!(matches!(err, SyncWriteError::ObjectClosed));
    }

    #[test]
    fn shutdown_reports_contract_violation_after_warmup_thread_panic() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.panic_warm_pin.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                ..BufferPoolConfig::default()
            },
            runtime,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed before warmup panic");

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match pool.state.lock() {
                Ok(guard) => drop(guard),
                Err(_) => break,
            }
            assert!(
                Instant::now() < deadline,
                "warmup thread did not panic and poison state lock in time"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        let err = pool
            .shutdown()
            .expect_err("shutdown should report contract violation after warmup panic");
        assert!(matches!(
            err,
            SyncWriteError::ContractViolation { ref message }
            if message.contains("state lock poisoned")
        ));
    }

    #[test]
    fn wait_pooled_leases_returned_times_out_with_outstanding_pooled_lease() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(256),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 256,
        })
        .expect("initialize should succeed");

        let lease = pool
            .acquire(128)
            .expect("pooled acquire should succeed before wait");
        let err = pool
            .wait_pooled_leases_returned(0.05)
            .expect_err("wait should timeout while pooled lease is outstanding");
        assert!(matches!(
            err,
            SyncWriteError::LeaseReturnTimeout { outstanding_leases } if outstanding_leases >= 1
        ));

        pool.release(lease);
        pool.wait_pooled_leases_returned(0.5)
            .expect("wait should succeed once pooled lease is released");
    }

    #[test]
    fn wait_pooled_leases_returned_unblocks_after_transient_release() {
        let pool = Arc::new(MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(64),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        }));
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 64,
        })
        .expect("initialize should succeed");

        let transient = pool
            .acquire(128)
            .expect("oversized acquire should use transient fallback");
        assert!(matches!(transient, BufferLease::Transient(_)));

        let pool_for_wait = Arc::clone(&pool);
        let (tx, rx) = mpsc::channel();
        let wait_handle = std::thread::spawn(move || {
            let result = pool_for_wait.wait_pooled_leases_returned(0.5);
            tx.send(result)
                .expect("wait result channel send should succeed");
        });

        assert!(
            rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "wait should still be blocked while transient lease is outstanding"
        );

        pool.release(transient);
        let wait_result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("wait should complete after transient release");
        assert!(
            wait_result.is_ok(),
            "wait should succeed after transient release, got {wait_result:?}"
        );
        wait_handle.join().expect("wait thread should join");
    }

    #[test]
    fn initialization_creates_real_hot_and_warm_slab_backing_layout() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.cuda_available.store(false, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 4,
                hot_slab_buffers: SizeOverride::Fixed(2),
                warm_slab_buffers: SizeOverride::Fixed(2),
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 4,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");

        let snapshot = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(snapshot.pooled_buffer_bytes, 4096);
        assert_eq!(snapshot.hot_slab_buffers, 2);
        assert_eq!(snapshot.warm_slab_buffers, 2);
        assert!(
            matches!(
                snapshot.warm_state,
                WarmSlabState::InProgress | WarmSlabState::Ready
            ),
            "warm slab should be in-progress or ready after initialization trigger"
        );
        assert!(
            snapshot.hot_pinned,
            "hot slab should be pinned when pinning is enabled"
        );
        assert!(
            !snapshot.hot_cuda_registered,
            "cuda registration is expected to be disabled in non-cuda test environment"
        );
        assert!(
            !snapshot.warm_pinned,
            "warm slab should not report pinned before warmup reaches ready"
        );
        assert!(
            !snapshot.warm_cuda_registered,
            "warm slab should not report cuda-registered before warmup reaches ready"
        );

        assert_eq!(snapshot.hot_buffer_ranges.len(), 2);
        assert_eq!(snapshot.warm_buffer_ranges.len(), 2);
        for range in snapshot
            .hot_buffer_ranges
            .iter()
            .chain(snapshot.warm_buffer_ranges.iter())
        {
            assert_eq!(range.end - range.start, 4096);
        }
        assert!(
            snapshot.hot_slab_backing_bytes >= 2 * 4096,
            "hot slab should reserve backing bytes for all hot buffers"
        );
        assert!(
            snapshot.warm_slab_backing_bytes >= 2 * 4096,
            "warm slab should reserve backing bytes for all warm buffers"
        );
    }

    #[test]
    fn warm_slab_transitions_to_ready_and_becomes_eligible_for_acquire() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(4096),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");

        let first = pool.acquire(1024).expect("first acquire should succeed");
        assert!(matches!(first, BufferLease::Pooled(_)));

        let early = pool
            .acquire(1024)
            .expect("early second acquire should succeed");
        assert!(
            matches!(early, BufferLease::Transient(_)),
            "warm slab should not be eligible until warmup becomes ready"
        );
        pool.release(early);

        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warm slab should become ready");
        let status = pool.warmup_status();
        assert_eq!(status.warm_state, WarmSlabState::Ready);

        let second = pool
            .acquire(1024)
            .expect("second acquire should succeed after warmup ready");
        assert!(
            matches!(second, BufferLease::Pooled(_)),
            "warm slab should be eligible after warmup completion"
        );

        pool.release(first);
        pool.release(second);
    }

    #[test]
    fn shutdown_waits_for_warmup_thread_join_before_returning() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(4096),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");

        let started = Instant::now();
        pool.shutdown()
            .expect("shutdown should join warmup thread successfully");
        assert!(
            started.elapsed() >= Duration::from_millis(20),
            "shutdown should wait for warmup thread to exit before returning"
        );
    }

    #[test]
    fn hot_slab_pinning_failure_degrades_gracefully_when_probe_said_yes() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.fail_hot_pin.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect("hot slab pinning failure should degrade gracefully, not fail initialization");

        assert_eq!(
            runtime.pin_hot_calls.load(Ordering::Relaxed),
            1,
            "pin_slab should have been attempted once"
        );
        let snapshot = pool
            .debug_snapshot()
            .expect("debug snapshot should be available after degraded init");
        assert!(
            !snapshot.hot_pinned,
            "hot slab should not be marked pinned after pinning failure"
        );
    }

    #[test]
    fn warm_pinning_failure_transitions_to_failed_degraded() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.fail_warm_pin.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                ..BufferPoolConfig::default()
            },
            runtime,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed and warm failure should degrade");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should settle into ready or failed-degraded");

        let status = pool.warmup_status();
        assert_eq!(status.warm_state, WarmSlabState::FailedDegraded);

        let first = pool.acquire(1024).expect("hot slab acquire should succeed");
        assert!(matches!(first, BufferLease::Pooled(_)));
        let second = pool
            .acquire(1024)
            .expect("second acquire should succeed via transient fallback");
        assert!(
            matches!(second, BufferLease::Transient(_)),
            "warm slab must not become eligible after failed warmup"
        );
        pool.release(first);
        pool.release(second);
    }

    #[test]
    fn shutdown_unpins_hot_and_warm_slabs() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should complete before shutdown assertions");
        pool.shutdown().expect("shutdown should succeed");

        assert_eq!(runtime.pin_hot_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.pin_warm_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.unpin_hot_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.unpin_warm_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn cuda_registration_is_skipped_when_cuda_not_available() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.cuda_available.store(false, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should complete");

        assert_eq!(
            runtime.register_hot_calls.load(Ordering::Relaxed),
            0,
            "hot registration must be skipped when CUDA is unavailable"
        );
        assert_eq!(
            runtime.register_warm_calls.load(Ordering::Relaxed),
            0,
            "warm registration must be skipped when CUDA is unavailable"
        );
    }

    #[test]
    fn cuda_registered_slabs_produce_cuda_registered_pooled_handles() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.cuda_available.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("initialize should succeed");
        pool.debug_wait_for_warm_ready(Duration::from_millis(250))
            .expect("warmup should complete");

        let hot = pool
            .acquire(512)
            .expect("hot pooled acquire should succeed");
        let warm = pool
            .acquire(512)
            .expect("warm pooled acquire should succeed");
        let hot_registered = match &hot {
            BufferLease::Pooled(handle) => handle.cuda_registered(),
            BufferLease::Transient(_) => false,
        };
        let warm_registered = match &warm {
            BufferLease::Pooled(handle) => handle.cuda_registered(),
            BufferLease::Transient(_) => false,
        };
        assert!(
            hot_registered,
            "hot pooled handle should be CUDA-registered"
        );
        assert!(
            warm_registered,
            "warm pooled handle should be CUDA-registered"
        );
        pool.release(hot);
        pool.release(warm);
        pool.shutdown().expect("shutdown should succeed");

        assert_eq!(runtime.register_hot_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.register_warm_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.unregister_hot_calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.unregister_warm_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn hot_cuda_registration_failure_degrades_to_unregistered_pooled_hot_lease() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.cuda_available.store(true, Ordering::Relaxed);
        runtime
            .fail_hot_cuda_register
            .store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect(
            "hot CUDA registration failure should degrade to unregistered pooled mode, not fail initialization",
        );

        let snapshot = pool
            .debug_snapshot()
            .expect("debug snapshot should be available after degraded initialization");
        assert!(
            !snapshot.hot_cuda_registered,
            "hot slab should remain unregistered after failed CUDA registration"
        );

        let lease = pool
            .acquire(512)
            .expect("pooled acquire should still succeed in degraded mode");
        let hot_registered = match &lease {
            BufferLease::Pooled(handle) => handle.cuda_registered(),
            BufferLease::Transient(_) => false,
        };
        assert!(
            !hot_registered,
            "pooled hot lease should report cuda_registered=false in degraded mode"
        );
        pool.release(lease);
    }

    #[test]
    fn probe_returns_false_skips_hot_slab_pinning() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        // can_pin_memory defaults to false via AtomicBool::default()
        assert!(!runtime.can_pin_memory.load(Ordering::Relaxed));
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                pin_pooled_slabs: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed when probe returns false (skip pinning)");
        assert_eq!(
            runtime.pin_hot_calls.load(Ordering::Relaxed),
            0,
            "pin_slab should never be called when can_pin_memory() is false"
        );
    }

    #[test]
    fn probe_returns_true_still_pins_hot_slab() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                pin_pooled_slabs: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed when probe+pin both succeed");
        assert_eq!(
            runtime.pin_hot_calls.load(Ordering::Relaxed),
            1,
            "pin_slab should be called exactly once for the hot slab"
        );
    }

    #[test]
    fn probe_returns_false_skips_warm_slab_pinning() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        // can_pin_memory defaults to false
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                pin_pooled_slabs: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed when probe returns false");
        pool.debug_wait_for_warm_ready(Duration::from_millis(500))
            .expect("warmup should complete");
        assert_eq!(
            runtime.pin_warm_calls.load(Ordering::Relaxed),
            0,
            "pin_slab should never be called for warm slab when can_pin_memory() is false"
        );
    }

    #[test]
    fn default_runtime_ops_can_pin_memory_is_consistent() {
        let ops = DefaultRuntimeMemoryOps::default();
        let first = ops.can_pin_memory();
        let second = ops.can_pin_memory();
        assert_eq!(
            first, second,
            "can_pin_memory() must return the same value on repeated calls (OnceLock caching)"
        );
    }

    #[test]
    fn cuda_registration_available_skips_hot_slab_mlock() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.cuda_available.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                pin_pooled_slabs: true,
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed");
        assert_eq!(
            runtime.pin_hot_calls.load(Ordering::Relaxed),
            0,
            "mlock must be skipped for hot slab when cudaHostRegister will be used"
        );
    }

    #[test]
    fn cuda_registration_available_skips_warm_slab_mlock() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        runtime.cuda_available.store(true, Ordering::Relaxed);
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 2,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Fixed(1),
                pin_pooled_slabs: true,
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 2,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed");
        pool.debug_wait_for_warm_ready(Duration::from_millis(500))
            .expect("warmup should complete");
        assert_eq!(
            runtime.pin_warm_calls.load(Ordering::Relaxed),
            0,
            "mlock must be skipped for warm slab when cudaHostRegister will be used"
        );
    }

    #[test]
    fn supports_early_init_rejects_auto_pool_buffer_bytes() {
        // When pool_buffer_bytes is Auto, early init would use a fallback size
        // (e.g. 4 MiB) that may not match the actual first-write chunk size.
        // supports_early_init() must return false to prevent initializing with
        // a wrong buffer size that can never be corrected (initialize_if_needed
        // is idempotent).
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Auto,
            hot_slab_buffers: SizeOverride::Fixed(4),
            warm_slab_buffers: SizeOverride::Fixed(4),
            ..BufferPoolConfig::default()
        });
        assert!(
            !pool.supports_early_init(),
            "supports_early_init must return false when pool_buffer_bytes is Auto, \
             because the fallback buffer size may not match the actual write size"
        );
    }

    #[test]
    fn supports_early_init_accepts_all_fixed() {
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(4096),
            hot_slab_buffers: SizeOverride::Fixed(4),
            warm_slab_buffers: SizeOverride::Fixed(4),
            ..BufferPoolConfig::default()
        });
        assert!(
            pool.supports_early_init(),
            "supports_early_init must return true when all sizing is explicitly Fixed"
        );
    }

    #[test]
    fn early_init_with_zero_required_bytes_uses_fixed_buffer_size() {
        // When supports_early_init() is true, the coordinator passes
        // first_write_max_required_bytes: 0 because the Fixed config makes this
        // field irrelevant. This test proves the pool uses the Fixed value (8192),
        // not the hint's 0, so early init allocates correctly-sized buffers.
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(8192),
            max_pool_buffers: 2,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Fixed(1),
            ..BufferPoolConfig::default()
        });
        assert!(
            pool.supports_early_init(),
            "precondition: pool must support early init"
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 0, // as coordinator sends during early init
        })
        .expect("early init with zero required_bytes should succeed");

        let snapshot = pool
            .debug_snapshot()
            .expect("debug snapshot should be available");
        assert_eq!(
            snapshot.pooled_buffer_bytes, 8192,
            "buffer size must come from the Fixed config (8192), not the hint (0)"
        );
    }

    #[test]
    fn pin_with_cuda_disabled_still_mlocks() {
        let runtime = Arc::new(FakeRuntimeOps::default());
        runtime.can_pin_memory.store(true, Ordering::Relaxed);
        // cuda_available defaults to false
        assert!(!runtime.cuda_available.load(Ordering::Relaxed));
        let pool = MemoryBufferPool::new_with_runtime_ops(
            BufferPoolConfig {
                pool_buffer_bytes: SizeOverride::Fixed(4096),
                max_pool_buffers: 1,
                hot_slab_buffers: SizeOverride::Fixed(1),
                warm_slab_buffers: SizeOverride::Auto,
                pin_pooled_slabs: true,
                cuda_register_pool_if_available: true,
                cuda_register_each_slab_once: true,
                ..BufferPoolConfig::default()
            },
            Arc::clone(&runtime) as Arc<dyn RuntimeMemoryOps>,
        );
        pool.initialize_if_needed(&FirstWriteSizingHint {
            first_write_task_count: 1,
            first_write_max_required_bytes: 4096,
        })
        .expect("init should succeed");
        assert_eq!(
            runtime.pin_hot_calls.load(Ordering::Relaxed),
            1,
            "mlock must still be called when CUDA is unavailable"
        );
    }
}
