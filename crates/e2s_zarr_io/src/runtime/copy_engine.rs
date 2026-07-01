/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host memcpy and CUDA copy path selection.
//!
//! The copy engine selects the appropriate copy strategy per task based on
//! `(input source kind, lease kind)`:
//! - Host source → CPU memcpy (synchronous).
//! - CUDA source + registered lease → CUDA D2H DMA.
//! - CUDA source + transient lease → fallback copy path.

use std::sync::Arc;

use crate::core::contracts::CopyEngine;
use crate::core::errors::SyncWriteError;
use crate::core::types::{BufferLease, CopyCompletion, InputArray, InputArraySource};
use crate::runtime::cuda_runtime::{CudaRuntimeApi, shared_cuda_runtime_api};

/// Default copy engine with host memcpy and CUDA blocking D2H copy support.
pub struct DefaultCopyEngine {
    cuda_runtime: Arc<dyn CudaRuntimeApi>,
}

impl DefaultCopyEngine {
    /// Create a new default copy engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cuda_runtime: shared_cuda_runtime_api(),
        }
    }

    #[cfg(test)]
    fn new_with_cuda_runtime(cuda_runtime: Arc<dyn CudaRuntimeApi>) -> Self {
        Self { cuda_runtime }
    }

    fn copy_cuda_into_lease(
        &self,
        src_device_ptr: u64,
        lease: &mut BufferLease,
        required_bytes: usize,
    ) -> Result<CopyCompletion, SyncWriteError> {
        if !self.cuda_runtime.available() {
            return Err(SyncWriteError::copy_failed(
                "cuda runtime path is unavailable",
            ));
        }
        match lease {
            BufferLease::Pooled(handle) => {
                let use_registered_path = handle.cuda_registered();
                handle.with_bytes_mut(required_bytes, |dst| {
                    if use_registered_path {
                        self.cuda_runtime.memcpy_device_to_host_registered_blocking(
                            dst.as_mut_ptr(),
                            src_device_ptr,
                            required_bytes,
                        )
                    } else {
                        self.cuda_runtime.memcpy_device_to_host_fallback_blocking(
                            dst.as_mut_ptr(),
                            src_device_ptr,
                            required_bytes,
                        )
                    }
                })??;
            }
            BufferLease::Transient(buffer) => {
                if required_bytes > buffer.len() {
                    return Err(SyncWriteError::copy_failed(format!(
                        "transient buffer shorter than required bytes: len={} required={}",
                        buffer.len(),
                        required_bytes
                    )));
                }
                self.cuda_runtime.memcpy_device_to_host_fallback_blocking(
                    buffer.as_bytes_mut().as_mut_ptr(),
                    src_device_ptr,
                    required_bytes,
                )?;
            }
        }
        Ok(CopyCompletion::CudaCopyDone { event: 0 })
    }
}

impl Default for DefaultCopyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CopyEngine for DefaultCopyEngine {
    fn copy_into_lease(
        &self,
        src: &InputArray,
        lease: &mut BufferLease,
        required_bytes: usize,
    ) -> Result<CopyCompletion, SyncWriteError> {
        if required_bytes > src.nbytes {
            return Err(SyncWriteError::copy_failed(format!(
                "required_bytes exceeds source nbytes: {} > {}",
                required_bytes, src.nbytes
            )));
        }

        if let Some(ptr) = src.source.as_host_buffer_ptr() {
            let ptr_usize = usize::try_from(ptr).map_err(|cause| {
                SyncWriteError::copy_failed_with_cause(
                    format!("host pointer does not fit usize: {ptr}"),
                    cause,
                )
            })?;
            // SAFETY:
            // - `ptr_usize` originates from Python host array interface parsing.
            // - The Python boundary validates `nbytes` and keeps source objects alive
            //   for the full blocking `write()` call (copy barrier), so this pointer is
            //   expected to remain readable for `required_bytes` here.
            // - `required_bytes <= src.nbytes` was validated at function entry.
            let payload =
                unsafe { std::slice::from_raw_parts(ptr_usize as *const u8, required_bytes) };
            lease.write_from_host_bytes(payload, required_bytes)?;
            return Ok(CopyCompletion::ImmediateHostCopy);
        }

        match &src.source {
            InputArraySource::HostBytes(payload) => {
                lease.write_from_host_bytes(payload, required_bytes)?;
                Ok(CopyCompletion::ImmediateHostCopy)
            }
            InputArraySource::CudaDevicePtr { ptr, .. } => {
                self.copy_cuda_into_lease(*ptr, lease, required_bytes)
            }
            InputArraySource::__InternalHostBufferPtr { .. } => {
                Err(SyncWriteError::ContractViolation {
                    message: "internal host pointer source bypassed pre-branch pointer extraction"
                        .to_string(),
                })
            }
        }
    }

    fn wait_copy_completion(&self, _completion: CopyCompletion) -> Result<(), SyncWriteError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::core::contracts::{BufferPool, CopyEngine};
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        BufferHandle, BufferLease, BufferPoolConfig, InputArray, InputArraySource, SizeOverride,
        TransientBuffer,
    };
    use crate::runtime::buffer_pool::MemoryBufferPool;
    use crate::runtime::cuda_runtime::{CudaRuntimeApi, NoCudaRuntimeApi};

    use super::DefaultCopyEngine;

    #[derive(Default)]
    struct FakeCudaRuntimeApi {
        registered_calls: AtomicUsize,
        fallback_calls: AtomicUsize,
        device_payloads: Mutex<HashMap<u64, Vec<u8>>>,
    }

    impl FakeCudaRuntimeApi {
        fn set_payload(&self, src_ptr: u64, payload: Vec<u8>) {
            self.device_payloads
                .lock()
                .expect("device payload lock poisoned")
                .insert(src_ptr, payload);
        }

        fn registered_calls(&self) -> usize {
            self.registered_calls.load(Ordering::Acquire)
        }

        fn fallback_calls(&self) -> usize {
            self.fallback_calls.load(Ordering::Acquire)
        }

        fn copy_from_fake_device(
            &self,
            dst_host_ptr: *mut u8,
            src_device_ptr: u64,
            bytes: usize,
        ) -> Result<(), SyncWriteError> {
            let payloads = self
                .device_payloads
                .lock()
                .map_err(|_| SyncWriteError::copy_failed("fake cuda payload lock poisoned"))?;
            let payload = payloads.get(&src_device_ptr).ok_or_else(|| {
                SyncWriteError::copy_failed(format!(
                    "missing fake cuda payload for source pointer {src_device_ptr}"
                ))
            })?;
            if payload.len() < bytes {
                return Err(SyncWriteError::copy_failed(format!(
                    "fake cuda payload shorter than requested bytes: payload={} requested={bytes}",
                    payload.len()
                )));
            }
            // SAFETY:
            // - Test setup ensures `dst_host_ptr` points to a writable destination buffer for `bytes`.
            // - The destination memory is owned by lease objects that outlive this copy call.
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_host_ptr, bytes) };
            dst.copy_from_slice(&payload[..bytes]);
            Ok(())
        }
    }

    impl CudaRuntimeApi for FakeCudaRuntimeApi {
        fn available(&self) -> bool {
            true
        }

        fn host_register(&self, _host_ptr: *mut u8, _bytes: usize) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn host_unregister(&self, _host_ptr: *mut u8) -> Result<(), SyncWriteError> {
            Ok(())
        }

        fn memcpy_device_to_host_registered_blocking(
            &self,
            dst_host_ptr: *mut u8,
            src_device_ptr: u64,
            bytes: usize,
        ) -> Result<(), SyncWriteError> {
            self.registered_calls.fetch_add(1, Ordering::AcqRel);
            self.copy_from_fake_device(dst_host_ptr, src_device_ptr, bytes)
        }

        fn memcpy_device_to_host_fallback_blocking(
            &self,
            dst_host_ptr: *mut u8,
            src_device_ptr: u64,
            bytes: usize,
        ) -> Result<(), SyncWriteError> {
            self.fallback_calls.fetch_add(1, Ordering::AcqRel);
            self.copy_from_fake_device(dst_host_ptr, src_device_ptr, bytes)
        }
    }

    fn pooled_lease(cuda_registered: bool, capacity: usize) -> BufferLease {
        let slab = Arc::new(Mutex::new(vec![0_u8; capacity]));
        let base_addr = slab.lock().expect("slab lock poisoned").as_ptr() as usize;
        let handle = BufferHandle::new(
            42,
            capacity,
            0..capacity,
            base_addr,
            capacity,
            Arc::new(Mutex::new(())),
            slab,
        )
        .with_cuda_registered(cuda_registered);
        BufferLease::Pooled(handle)
    }

    #[test]
    fn host_copy_writes_into_transient_lease_bytes() {
        let engine = DefaultCopyEngine::new();
        let src = InputArray {
            nbytes: 4,
            source: InputArraySource::HostBytes(vec![1, 2, 3, 4].into()),
        };
        let mut lease = BufferLease::Transient(TransientBuffer::new(4));

        engine
            .copy_into_lease(&src, &mut lease, 4)
            .expect("host copy into transient lease should succeed");

        assert_eq!(
            lease
                .staged_bytes(4)
                .expect("transient staged bytes should be readable"),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn host_copy_writes_into_pooled_slab_slice_bytes() {
        let engine = DefaultCopyEngine::new();
        let pool = MemoryBufferPool::new(BufferPoolConfig {
            pool_buffer_bytes: SizeOverride::Fixed(128),
            max_pool_buffers: 1,
            hot_slab_buffers: SizeOverride::Fixed(1),
            warm_slab_buffers: SizeOverride::Auto,
            ..BufferPoolConfig::default()
        });
        let src = InputArray {
            nbytes: 4,
            source: InputArraySource::HostBytes(vec![9, 8, 7, 6].into()),
        };
        let mut lease = pool.acquire(4).expect("pooled acquire should succeed");
        assert!(matches!(lease, BufferLease::Pooled(_)));

        engine
            .copy_into_lease(&src, &mut lease, 4)
            .expect("host copy into pooled lease should succeed");

        assert_eq!(
            lease
                .staged_bytes(4)
                .expect("pooled staged bytes should be readable"),
            vec![9, 8, 7, 6]
        );
        pool.release(lease);
    }

    #[test]
    fn host_buffer_ptr_copy_writes_into_transient_lease_bytes() {
        let engine = DefaultCopyEngine::new();
        let src_bytes = vec![3_u8, 1, 4, 1, 5, 9];
        let src = InputArray {
            nbytes: src_bytes.len(),
            source: unsafe { InputArraySource::from_host_buffer_ptr(src_bytes.as_ptr() as u64) },
        };
        let mut lease = BufferLease::Transient(TransientBuffer::new(src_bytes.len()));

        engine
            .copy_into_lease(&src, &mut lease, src_bytes.len())
            .expect("host pointer copy into transient lease should succeed");

        assert_eq!(
            lease
                .staged_bytes(src_bytes.len())
                .expect("transient staged bytes should be readable"),
            src_bytes
        );
    }

    #[test]
    fn cuda_source_without_runtime_support_returns_copy_failed() {
        let engine = DefaultCopyEngine::new_with_cuda_runtime(Arc::new(NoCudaRuntimeApi));
        let src = InputArray {
            nbytes: 8,
            source: InputArraySource::CudaDevicePtr {
                ptr: 0x1000,
                device_ordinal: 0,
                producer_stream: None,
            },
        };
        let mut lease = BufferLease::Transient(TransientBuffer::new(8));

        let err = engine
            .copy_into_lease(&src, &mut lease, 8)
            .expect_err("CUDA source copy must fail when CUDA runtime path is unavailable");
        assert!(
            matches!(err, SyncWriteError::CopyFailed { .. }),
            "expected CopyFailed for unavailable CUDA runtime, got: {err:?}"
        );
    }

    #[test]
    fn cuda_registered_pooled_lease_uses_registered_copy_path() {
        let fake_cuda = Arc::new(FakeCudaRuntimeApi::default());
        fake_cuda.set_payload(0x2000, vec![4, 3, 2, 1]);
        let engine = DefaultCopyEngine::new_with_cuda_runtime(fake_cuda.clone());
        let src = InputArray {
            nbytes: 4,
            source: InputArraySource::CudaDevicePtr {
                ptr: 0x2000,
                device_ordinal: 0,
                producer_stream: None,
            },
        };
        let mut lease = pooled_lease(true, 8);

        engine
            .copy_into_lease(&src, &mut lease, 4)
            .expect("registered cuda copy should succeed");
        assert_eq!(
            lease
                .staged_bytes(4)
                .expect("staged bytes should be readable"),
            vec![4, 3, 2, 1]
        );
        assert_eq!(fake_cuda.registered_calls(), 1);
        assert_eq!(fake_cuda.fallback_calls(), 0);
    }

    #[test]
    fn cuda_transient_lease_uses_fallback_copy_path() {
        let fake_cuda = Arc::new(FakeCudaRuntimeApi::default());
        fake_cuda.set_payload(0x3000, vec![9, 8, 7, 6]);
        let engine = DefaultCopyEngine::new_with_cuda_runtime(fake_cuda.clone());
        let src = InputArray {
            nbytes: 4,
            source: InputArraySource::CudaDevicePtr {
                ptr: 0x3000,
                device_ordinal: 0,
                producer_stream: None,
            },
        };
        let mut lease = BufferLease::Transient(TransientBuffer::new(4));

        engine
            .copy_into_lease(&src, &mut lease, 4)
            .expect("transient cuda fallback copy should succeed");
        assert_eq!(
            lease
                .staged_bytes(4)
                .expect("staged bytes should be readable"),
            vec![9, 8, 7, 6]
        );
        assert_eq!(fake_cuda.registered_calls(), 0);
        assert_eq!(fake_cuda.fallback_calls(), 1);
    }

    #[test]
    fn cuda_unregistered_pooled_lease_uses_fallback_copy_path() {
        let fake_cuda = Arc::new(FakeCudaRuntimeApi::default());
        fake_cuda.set_payload(0x4000, vec![1, 3, 3, 7]);
        let engine = DefaultCopyEngine::new_with_cuda_runtime(fake_cuda.clone());
        let src = InputArray {
            nbytes: 4,
            source: InputArraySource::CudaDevicePtr {
                ptr: 0x4000,
                device_ordinal: 0,
                producer_stream: None,
            },
        };
        let mut lease = pooled_lease(false, 8);

        engine
            .copy_into_lease(&src, &mut lease, 4)
            .expect("unregistered pooled cuda fallback copy should succeed");
        assert_eq!(
            lease
                .staged_bytes(4)
                .expect("staged bytes should be readable"),
            vec![1, 3, 3, 7]
        );
        assert_eq!(fake_cuda.registered_calls(), 0);
        assert_eq!(fake_cuda.fallback_calls(), 1);
    }
}
