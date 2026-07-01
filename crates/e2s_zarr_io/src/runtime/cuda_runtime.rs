/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA runtime integration for slab registration and blocking D2H copy.

use std::ffi::CString;
use std::sync::{Arc, OnceLock};

use libc::{c_int, c_uint, c_void};

use crate::core::errors::SyncWriteError;

const CUDA_SUCCESS: c_int = 0;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_HOST_REGISTER_DEFAULT: c_uint = 0;
const CUDA_RUNTIME_CANDIDATES: &[&str] = &[
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
];
const SUPPORTED_DLSYM_FN_CAST_TARGET: bool = cfg!(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
));

type CudaGetDeviceCountFn = unsafe extern "C" fn(*mut c_int) -> c_int;
type CudaHostRegisterFn = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
type CudaHostUnregisterFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;

/// CUDA runtime API contract used by buffer-pool and copy-engine components.
pub trait CudaRuntimeApi: Send + Sync {
    /// Returns `true` when CUDA runtime is loaded and at least one device is present.
    fn available(&self) -> bool;

    /// Register host memory as CUDA-pinned memory.
    fn host_register(&self, host_ptr: *mut u8, bytes: usize) -> Result<(), SyncWriteError>;

    /// De-register host memory from CUDA.
    fn host_unregister(&self, host_ptr: *mut u8) -> Result<(), SyncWriteError>;

    /// Blocking D2H copy path for registered host memory (DMA-eligible path).
    fn memcpy_device_to_host_registered_blocking(
        &self,
        dst_host_ptr: *mut u8,
        src_device_ptr: u64,
        bytes: usize,
    ) -> Result<(), SyncWriteError>;

    /// Blocking D2H copy path for unregistered/transient host memory.
    fn memcpy_device_to_host_fallback_blocking(
        &self,
        dst_host_ptr: *mut u8,
        src_device_ptr: u64,
        bytes: usize,
    ) -> Result<(), SyncWriteError>;
}

/// CUDA runtime stub that reports unavailable runtime and rejects CUDA calls.
#[derive(Debug, Default)]
pub struct NoCudaRuntimeApi;

impl CudaRuntimeApi for NoCudaRuntimeApi {
    fn available(&self) -> bool {
        false
    }

    fn host_register(&self, _host_ptr: *mut u8, _bytes: usize) -> Result<(), SyncWriteError> {
        Err(SyncWriteError::copy_failed(
            "cuda runtime is unavailable for host registration",
        ))
    }

    fn host_unregister(&self, _host_ptr: *mut u8) -> Result<(), SyncWriteError> {
        Ok(())
    }

    fn memcpy_device_to_host_registered_blocking(
        &self,
        _dst_host_ptr: *mut u8,
        _src_device_ptr: u64,
        _bytes: usize,
    ) -> Result<(), SyncWriteError> {
        Err(SyncWriteError::copy_failed(
            "cuda runtime is unavailable for registered D2H copy",
        ))
    }

    fn memcpy_device_to_host_fallback_blocking(
        &self,
        _dst_host_ptr: *mut u8,
        _src_device_ptr: u64,
        _bytes: usize,
    ) -> Result<(), SyncWriteError> {
        Err(SyncWriteError::copy_failed(
            "cuda runtime is unavailable for fallback D2H copy",
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct DlHandle(*mut c_void);

impl DlHandle {
    #[must_use]
    fn from_raw(handle: *mut c_void) -> Self {
        Self(handle)
    }

    #[must_use]
    fn as_ptr(&self) -> *mut c_void {
        self.0
    }

    #[must_use]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    fn clear(&mut self) {
        self.0 = std::ptr::null_mut();
    }
}

// SAFETY:
// - `DlHandle` is an opaque dynamic-loader handle.
// - It is consumed only through thread-safe loader APIs (`dlsym`/`dlclose`).
unsafe impl Send for DlHandle {}
// SAFETY: Same rationale as `Send`; sharing the opaque handle is safe.
unsafe impl Sync for DlHandle {}

#[derive(Debug)]
struct LoadedCudaRuntime {
    handle: DlHandle,
    get_device_count: CudaGetDeviceCountFn,
    host_register: CudaHostRegisterFn,
    host_unregister: CudaHostUnregisterFn,
    memcpy: CudaMemcpyFn,
}

impl Drop for LoadedCudaRuntime {
    fn drop(&mut self) {
        let handle = self.handle.as_ptr();
        if !self.handle.is_null() {
            // SAFETY:
            // - `handle` was returned by `dlopen` and is owned by this struct.
            // - `Drop` is invoked at most once for this instance.
            let _ = unsafe { libc::dlclose(handle) };
            self.handle.clear();
        }
    }
}

impl LoadedCudaRuntime {
    #[cfg(unix)]
    fn load() -> Option<Self> {
        for candidate in CUDA_RUNTIME_CANDIDATES {
            // SAFETY:
            // - `candidate` is converted to a C string that remains alive during the call.
            // - `try_load_from_library` validates every required symbol before returning.
            if let Some(runtime) = unsafe { Self::try_load_from_library(candidate) } {
                // Verify the loaded runtime actually works with the installed driver.
                // A newer toolkit (e.g. CUDA 13) may dlopen + resolve symbols fine but
                // return cudaErrorInsufficientDriver (35) from cudaGetDeviceCount when
                // the driver only supports an older CUDA version. In that case, skip
                // this candidate and try the next one (e.g. libcudart.so.12).
                if runtime.detect_available() {
                    return Some(runtime);
                }
            }
        }
        None
    }

    #[cfg(not(unix))]
    fn load() -> Option<Self> {
        None
    }

    #[cfg(unix)]
    unsafe fn try_load_from_library(library_name: &str) -> Option<Self> {
        let c_library = CString::new(library_name).ok()?;
        // SAFETY:
        // - `c_library` is NUL-terminated and valid for this call.
        let handle =
            unsafe { libc::dlopen(c_library.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return None;
        }
        if !SUPPORTED_DLSYM_FN_CAST_TARGET {
            // SAFETY:
            // - `handle` is a valid `dlopen` handle and is not used after this call.
            let _ = unsafe { libc::dlclose(handle) };
            return None;
        }
        let get_device_count =
            unsafe { load_symbol(handle, "cudaGetDeviceCount") }.map(symbol_to_get_device_count);
        let host_register =
            unsafe { load_symbol(handle, "cudaHostRegister") }.map(symbol_to_host_register);
        let host_unregister =
            unsafe { load_symbol(handle, "cudaHostUnregister") }.map(symbol_to_host_unregister);
        let memcpy = unsafe { load_symbol(handle, "cudaMemcpy") }.map(symbol_to_memcpy);

        match (get_device_count, host_register, host_unregister, memcpy) {
            (Some(get_device_count), Some(host_register), Some(host_unregister), Some(memcpy)) => {
                Some(Self {
                    handle: DlHandle::from_raw(handle),
                    get_device_count,
                    host_register,
                    host_unregister,
                    memcpy,
                })
            }
            _ => {
                // SAFETY:
                // - `handle` is a valid `dlopen` handle and is not used after this call.
                let _ = unsafe { libc::dlclose(handle) };
                None
            }
        }
    }

    fn detect_available(&self) -> bool {
        let mut device_count: c_int = 0;
        // SAFETY:
        // - `device_count` is a valid out pointer for CUDA runtime.
        let status = unsafe { (self.get_device_count)(&mut device_count as *mut c_int) };
        status == CUDA_SUCCESS && device_count > 0
    }
}

/// Runtime CUDA loader backed by `libcudart` dynamic symbol resolution.
#[derive(Debug)]
pub struct DynamicCudaRuntimeApi {
    loaded: Option<LoadedCudaRuntime>,
    available: bool,
}

impl Default for DynamicCudaRuntimeApi {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicCudaRuntimeApi {
    /// Create a dynamic CUDA runtime wrapper.
    #[must_use]
    pub fn new() -> Self {
        let loaded = LoadedCudaRuntime::load();
        let available = loaded
            .as_ref()
            .is_some_and(LoadedCudaRuntime::detect_available);
        Self { loaded, available }
    }

    fn ensure_loaded(&self, action: &str) -> Result<&LoadedCudaRuntime, SyncWriteError> {
        if !self.available {
            return Err(SyncWriteError::copy_failed(format!(
                "cuda runtime unavailable for {action}"
            )));
        }
        self.loaded.as_ref().ok_or_else(|| {
            SyncWriteError::copy_failed(format!("cuda runtime symbols missing for {action}"))
        })
    }

    fn cuda_status_to_error(status: c_int, action: &str) -> SyncWriteError {
        SyncWriteError::copy_failed(format!("{action} failed with cuda status {status}"))
    }

    fn memcpy_blocking(
        &self,
        dst_host_ptr: *mut u8,
        src_device_ptr: u64,
        bytes: usize,
        action: &str,
    ) -> Result<(), SyncWriteError> {
        if bytes == 0 {
            return Ok(());
        }
        if dst_host_ptr.is_null() {
            return Err(SyncWriteError::copy_failed(format!(
                "{action} destination pointer is null"
            )));
        }
        let loaded = self.ensure_loaded(action)?;
        let src_addr = usize::try_from(src_device_ptr).map_err(|cause| {
            SyncWriteError::copy_failed_with_cause(
                format!("{action} source pointer does not fit usize: {src_device_ptr}"),
                cause,
            )
        })?;
        let src_ptr = src_addr as *const c_void;
        // SAFETY:
        // - `dst_host_ptr` points to a host buffer writable for `bytes` bytes.
        // - `src_ptr` is expected to be a valid CUDA device pointer for `bytes` bytes.
        // - This call is blocking (`cudaMemcpy`) and does not retain pointers after return.
        let status = unsafe {
            (loaded.memcpy)(
                dst_host_ptr.cast::<c_void>(),
                src_ptr,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Self::cuda_status_to_error(status, action))
        }
    }
}

impl CudaRuntimeApi for DynamicCudaRuntimeApi {
    fn available(&self) -> bool {
        self.available
    }

    fn host_register(&self, host_ptr: *mut u8, bytes: usize) -> Result<(), SyncWriteError> {
        if bytes == 0 {
            return Ok(());
        }
        if host_ptr.is_null() {
            return Err(SyncWriteError::copy_failed(
                "cuda host register received null pointer",
            ));
        }
        let loaded = self.ensure_loaded("host_register")?;
        // SAFETY:
        // - `host_ptr` points to writable host memory for `bytes` bytes.
        // - CUDA runtime does not retain ownership of the pointer.
        let status = unsafe {
            (loaded.host_register)(host_ptr.cast::<c_void>(), bytes, CUDA_HOST_REGISTER_DEFAULT)
        };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Self::cuda_status_to_error(status, "host_register"))
        }
    }

    fn host_unregister(&self, host_ptr: *mut u8) -> Result<(), SyncWriteError> {
        if host_ptr.is_null() {
            return Ok(());
        }
        let loaded = self.ensure_loaded("host_unregister")?;
        // SAFETY:
        // - `host_ptr` was previously registered with CUDA host registration APIs.
        // - CUDA runtime does not retain ownership after this call.
        let status = unsafe { (loaded.host_unregister)(host_ptr.cast::<c_void>()) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Self::cuda_status_to_error(status, "host_unregister"))
        }
    }

    fn memcpy_device_to_host_registered_blocking(
        &self,
        dst_host_ptr: *mut u8,
        src_device_ptr: u64,
        bytes: usize,
    ) -> Result<(), SyncWriteError> {
        self.memcpy_blocking(
            dst_host_ptr,
            src_device_ptr,
            bytes,
            "memcpy_device_to_host_registered_blocking",
        )
    }

    fn memcpy_device_to_host_fallback_blocking(
        &self,
        dst_host_ptr: *mut u8,
        src_device_ptr: u64,
        bytes: usize,
    ) -> Result<(), SyncWriteError> {
        self.memcpy_blocking(
            dst_host_ptr,
            src_device_ptr,
            bytes,
            "memcpy_device_to_host_fallback_blocking",
        )
    }
}

/// Returns a process-wide shared CUDA runtime provider.
#[must_use]
pub fn shared_cuda_runtime_api() -> Arc<dyn CudaRuntimeApi> {
    static SHARED: OnceLock<Arc<dyn CudaRuntimeApi>> = OnceLock::new();
    Arc::clone(
        SHARED.get_or_init(|| Arc::new(DynamicCudaRuntimeApi::new()) as Arc<dyn CudaRuntimeApi>),
    )
}

#[cfg(unix)]
unsafe fn load_symbol(handle: *mut c_void, symbol_name: &str) -> Option<*mut c_void> {
    let c_symbol = CString::new(symbol_name).ok()?;
    // SAFETY:
    // - `handle` comes from `dlopen`.
    // - `c_symbol` is a valid NUL-terminated symbol name for the call duration.
    let symbol_ptr = unsafe { libc::dlsym(handle, c_symbol.as_ptr()) };
    if symbol_ptr.is_null() {
        return None;
    }
    Some(symbol_ptr)
}

#[cfg(not(unix))]
unsafe fn load_symbol(_handle: *mut c_void, _symbol_name: &str) -> Option<*mut c_void> {
    None
}

#[cfg(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
))]
fn symbol_to_get_device_count(symbol_ptr: *mut c_void) -> CudaGetDeviceCountFn {
    // SAFETY:
    // - We only reach this conversion on supported POSIX targets where function and data
    //   pointers share the ABI representation used by `dlsym`.
    // - Callers request a known CUDA symbol with this exact C signature.
    unsafe { std::mem::transmute::<*mut c_void, CudaGetDeviceCountFn>(symbol_ptr) }
}

#[cfg(not(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
)))]
fn symbol_to_get_device_count(_symbol_ptr: *mut c_void) -> CudaGetDeviceCountFn {
    unreachable!("cuda dynamic symbol conversion is disabled on this target")
}

#[cfg(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
))]
fn symbol_to_host_register(symbol_ptr: *mut c_void) -> CudaHostRegisterFn {
    // SAFETY: same argument as `symbol_to_get_device_count`.
    unsafe { std::mem::transmute::<*mut c_void, CudaHostRegisterFn>(symbol_ptr) }
}

#[cfg(not(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
)))]
fn symbol_to_host_register(_symbol_ptr: *mut c_void) -> CudaHostRegisterFn {
    unreachable!("cuda dynamic symbol conversion is disabled on this target")
}

#[cfg(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
))]
fn symbol_to_host_unregister(symbol_ptr: *mut c_void) -> CudaHostUnregisterFn {
    // SAFETY: same argument as `symbol_to_get_device_count`.
    unsafe { std::mem::transmute::<*mut c_void, CudaHostUnregisterFn>(symbol_ptr) }
}

#[cfg(not(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
)))]
fn symbol_to_host_unregister(_symbol_ptr: *mut c_void) -> CudaHostUnregisterFn {
    unreachable!("cuda dynamic symbol conversion is disabled on this target")
}

#[cfg(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
))]
fn symbol_to_memcpy(symbol_ptr: *mut c_void) -> CudaMemcpyFn {
    // SAFETY: same argument as `symbol_to_get_device_count`.
    unsafe { std::mem::transmute::<*mut c_void, CudaMemcpyFn>(symbol_ptr) }
}

#[cfg(not(all(
    unix,
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )
)))]
fn symbol_to_memcpy(_symbol_ptr: *mut c_void) -> CudaMemcpyFn {
    unreachable!("cuda dynamic symbol conversion is disabled on this target")
}

#[cfg(test)]
mod tests {
    use super::{CudaRuntimeApi, DlHandle, NoCudaRuntimeApi};
    use crate::core::errors::SyncWriteError;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    #[test]
    fn no_cuda_runtime_reports_unavailable() {
        let runtime = NoCudaRuntimeApi;
        assert!(!runtime.available());
    }

    #[test]
    fn no_cuda_runtime_rejects_memcpy_calls() {
        let runtime = NoCudaRuntimeApi;
        let mut host = vec![0_u8; 4];
        let err = runtime
            .memcpy_device_to_host_registered_blocking(host.as_mut_ptr(), 0x1234, host.len())
            .expect_err("no-cuda runtime should reject memcpy");
        assert!(matches!(err, SyncWriteError::CopyFailed { .. }));
    }

    #[test]
    fn dl_handle_roundtrips_raw_pointer() {
        let raw = NonNull::<c_void>::dangling().as_ptr();
        let handle = DlHandle::from_raw(raw);

        assert_eq!(handle.as_ptr(), raw);
        assert!(!handle.is_null());
    }

    #[test]
    fn dl_handle_clear_sets_handle_to_null() {
        let raw = NonNull::<c_void>::dangling().as_ptr();
        let mut handle = DlHandle::from_raw(raw);
        handle.clear();

        assert!(handle.is_null());
    }
}
