/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Python boundary for `e2s_zarr_io` (`pyo3` feature-gated).
//!
//! This module exposes a wheel-friendly class surface that mirrors the
//! lifecycle contract:
//! `add_array(coords, array_name, data=None) -> write(x, coords, array_name) -> close()`.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use pyo3::exceptions::{
    PyImportError, PyOSError, PyRuntimeError, PyTimeoutError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::api::{SyncZarrBackendConfig, try_build_sync_zarr_backend};
use crate::backend::SyncZarrBackend;
use crate::core::contracts::ZarrIoBackend;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ArrayRegistration, BufferSizingPolicy, ChunkKeyEncoding, ChunkKeySeparator, CoordMap,
    CoordValues, DataType, FsyncPolicy, InferenceWriteRequest, InputArray, InputArraySource,
    ModelProfileHint, PoolAlignment, SizeOverride, WarmSlabFailurePolicy, WarmSlabWarmupPolicy,
    ZarrFormat,
};

const SUPPORTED_KWARGS: &[&str] = &[
    "file_name",
    "dataset_path",
    "parallel_coords",
    "default_parallel_coord_names",
    "zarr_format",
    "chunk_key_encoding",
    "chunk_key_separator",
    "buffer_sizing_policy",
    "model_profile_hint",
    "min_hot_slab_buffers",
    "max_warm_to_hot_ratio",
    "hot_slab_buffers",
    "warm_slab_buffers",
    "hot_slab_ready_timeout_seconds",
    "warm_slab_warmup_policy",
    "warm_slab_failure_policy",
    "max_pool_buffers",
    "max_pool_bytes",
    "pool_buffer_bytes",
    "pool_alignment",
    "pin_pooled_slabs",
    "cuda_register_pool_if_available",
    "cuda_register_each_slab_once",
    "max_transient_buffer_bytes",
    "max_inflight_transient_bytes",
    "close_lease_timeout_seconds",
    "queue_capacity",
    "num_threads",
    "require_host_array_interface",
    "fsync_policy",
];

/// Run `body` inside `py.detach()` with a `catch_unwind` panic boundary.
///
/// Converts Rust panics inside the GIL-released closure into
/// `SyncWriteError::ContractViolation` instead of aborting the Python process.
/// The `Ok(T)` value is discarded so that all call sites unify on `Result<(), _>`.
fn detach_with_panic_boundary<F, T>(py: Python<'_>, body: F) -> Result<(), SyncWriteError>
where
    F: FnOnce() -> Result<T, SyncWriteError> + Send,
{
    py.detach(
        move || match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(panic_payload) => {
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("unknown panic");
                Err(SyncWriteError::ContractViolation {
                    message: format!("internal panic: {msg}"),
                })
            }
        },
    )
}

/// Like [`detach_with_panic_boundary`] but preserves the `Ok(T)` return value.
fn detach_returning<F, T>(py: Python<'_>, body: F) -> Result<T, SyncWriteError>
where
    F: FnOnce() -> Result<T, SyncWriteError> + Send,
    T: Send,
{
    py.detach(
        move || match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
            Ok(result) => result,
            Err(panic_payload) => {
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("unknown panic");
                Err(SyncWriteError::ContractViolation {
                    message: format!("internal panic: {msg}"),
                })
            }
        },
    )
}

/// Rust-backed synchronous Zarr IO backend exposed to Python.
///
/// Lifecycle:
/// `add_array(coords, array_name, data=None)` -> repeated `write(x, coords, array_name)` -> `close()`.
#[pyclass(name = "E2sZarrIoBackend")]
pub struct PyE2sZarrIoBackend {
    backend: SyncZarrBackend,
    require_host_array_interface: bool,
    dataset_root: PathBuf,
    zarr_format: ZarrFormat,
    registered_array_dtypes: Mutex<HashMap<String, DataType>>,
}

impl PyE2sZarrIoBackend {
    fn store_registered_array_dtypes(
        &self,
        array_names: &[String],
        array_dtypes: &[DataType],
    ) -> PyResult<()> {
        let mut registered = self
            .registered_array_dtypes
            .lock()
            .map_err(|_| PyRuntimeError::new_err("registered array dtype lock poisoned"))?;
        for (name, dtype) in array_names.iter().zip(array_dtypes.iter().copied()) {
            registered.insert(name.clone(), dtype);
        }
        Ok(())
    }

    fn registered_dtypes_for(&self, array_names: &[String]) -> PyResult<Vec<Option<DataType>>> {
        let registered = self
            .registered_array_dtypes
            .lock()
            .map_err(|_| PyRuntimeError::new_err("registered array dtype lock poisoned"))?;
        Ok(array_names
            .iter()
            .map(|name| registered.get(name).copied())
            .collect())
    }
}

#[pymethods]
impl PyE2sZarrIoBackend {
    /// Create a backend from keyword-only configuration options.
    #[new]
    #[pyo3(signature = (**kwargs), text_signature = "(**kwargs)")]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let config = parse_backend_config(kwargs)?;
        let require_host_array_interface = config.write_execution.require_host_array_interface;
        let zarr_format = config.write_execution.zarr_target.zarr_format;
        let dataset_root = config.dataset_root.clone().ok_or_else(|| {
            PyValueError::new_err("missing required keyword argument: file_name (or dataset_path)")
        })?;
        let backend = try_build_sync_zarr_backend(config).map_err(sync_error_to_pyerr)?;
        Ok(Self {
            backend,
            require_host_array_interface,
            dataset_root,
            zarr_format,
            registered_array_dtypes: Mutex::new(HashMap::new()),
        })
    }

    /// Enter context-manager scope and return `self`.
    #[pyo3(text_signature = "($self)")]
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Exit context-manager scope with best-effort close, without suppressing exceptions.
    #[pyo3(text_signature = "($self, exc_type, exc_val, exc_tb)")]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if !self.backend.is_closed() {
            let timeout_seconds = self.backend.configured_close_timeout_seconds();
            if exc_type.is_none() {
                detach_with_panic_boundary(py, move || self.backend.close(timeout_seconds))
                    .map_err(sync_error_to_pyerr)?;
            } else {
                let _ = detach_with_panic_boundary(py, move || self.backend.close(timeout_seconds));
            }
        }
        Ok(false)
    }

    /// Register one or more arrays and their full coordinate contract.
    ///
    /// If `data` is provided, this method immediately submits one full-coordinate write
    /// using the registered coordinates and array names for initialization compatibility.
    #[pyo3(
        signature = (coords, array_name, data = None),
        text_signature = "($self, coords, array_name, data=None)"
    )]
    fn add_array(
        &self,
        py: Python<'_>,
        coords: &Bound<'_, PyAny>,
        array_name: &Bound<'_, PyAny>,
        data: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let registration_coords = parse_coord_map(coords)?;
        let array_names = parse_array_names(array_name)?;
        let init_arrays_and_dtypes = data
            .filter(|value| !value.is_none())
            .map(|value| parse_input_arrays_with_dtypes(value, self.require_host_array_interface))
            .transpose()?;
        let array_dtypes = init_arrays_and_dtypes.as_ref().map_or_else(
            || vec![DataType::default(); array_names.len()],
            |(_, dtypes)| dtypes.clone(),
        );
        let req = ArrayRegistration {
            coords: registration_coords.clone(),
            array_names: array_names.clone(),
            array_dtypes: array_dtypes.clone(),
        };
        detach_with_panic_boundary(py, move || self.backend.add_array(req))
            .map_err(sync_error_to_pyerr)?;
        self.store_registered_array_dtypes(&array_names, &array_dtypes)?;

        if let Some((arrays, _dtypes)) = init_arrays_and_dtypes {
            let init_write = InferenceWriteRequest {
                coords: registration_coords,
                array_names,
                arrays,
            };
            detach_with_panic_boundary(py, move || self.backend.write(init_write))
                .map_err(sync_error_to_pyerr)?;
        }

        Ok(())
    }

    /// Write one inference step (or subset) for one or more arrays.
    #[pyo3(text_signature = "($self, x, coords, array_name)")]
    fn write(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        coords: &Bound<'_, PyAny>,
        array_name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let array_names = parse_array_names(array_name)?;
        let registered_dtypes = self.registered_dtypes_for(&array_names)?;
        let req = InferenceWriteRequest {
            coords: parse_coord_map(coords)?,
            array_names,
            arrays: parse_input_arrays_with_target_dtypes(
                py,
                x,
                self.require_host_array_interface,
                &registered_dtypes,
            )?,
        };
        detach_with_panic_boundary(py, move || self.backend.write(req))
            .map_err(sync_error_to_pyerr)?;
        Ok(())
    }

    /// Read array values at the requested coordinates as a tensor.
    ///
    /// This method currently performs Python zarr/numpy interop and filesystem
    /// access under the GIL for the full call. It is intentionally not detached
    /// like `write()`/`close()` because the implementation relies on Python APIs
    /// throughout the read path.
    #[pyo3(
        signature = (coords, array_name, device = None),
        text_signature = "($self, coords, array_name, device=None)"
    )]
    fn read(
        &self,
        py: Python<'_>,
        coords: &Bound<'_, PyAny>,
        array_name: &Bound<'_, PyAny>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if self.backend.is_closed() {
            return Err(sync_error_to_pyerr(SyncWriteError::ObjectClosed));
        }

        let array_name = array_name
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("array_name must be a string"))?;
        validate_read_array_name(&array_name)?;
        let device = parse_read_device(device)?;
        let np = import_required_read_dependency(py, "numpy")?;
        let normalized_coords = normalize_coord_mapping(py, coords, &np)?;
        let (adjusted_coords, mapping) =
            convert_multidim_to_singledim(py, &normalized_coords, &np)?;

        let zarr = import_required_read_dependency(py, "zarr")?;
        let storage = zarr.getattr("storage")?;
        let local_store_cls = storage.getattr("LocalStore")?;
        let store = local_store_cls.call1((self.dataset_root.to_string_lossy().as_ref(),))?;
        let open_kwargs = PyDict::new(py);
        open_kwargs.set_item("store", &store)?;
        open_kwargs.set_item("mode", "r")?;
        let root = zarr.call_method("open_group", (), Some(&open_kwargs))?;

        for (dim_any, _) in adjusted_coords.iter() {
            let dim = dim_any
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("coords keys must be strings"))?;
            if !root_contains(&root, &dim)? {
                return Err(PyRuntimeError::new_err(format!(
                    "Coordinate dimension {dim} not in zarr store."
                )));
            }
        }

        for (coord_key_any, _) in mapping.iter() {
            let coord_key = coord_key_any
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("multidim mapping keys must be strings"))?;
            if !root_contains(&root, &coord_key)? {
                return Err(PyRuntimeError::new_err(format!(
                    "Multidimension coordinate {coord_key} not in zarr store."
                )));
            }
            let original = normalized_coords
                .get_item(coord_key.as_str())?
                .ok_or_else(|| {
                    PyTypeError::new_err(format!(
                        "coords must provide multidimension coordinate key '{coord_key}'"
                    ))
                })?;
            let stored = root.get_item(coord_key.as_str())?;
            let original_shape = original
                .getattr("shape")
                .and_then(|shape| shape.extract::<Vec<usize>>())
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "coords['{coord_key}'] must provide a shape attribute"
                    ))
                })?;
            let stored_shape = stored
                .getattr("shape")
                .and_then(|shape| shape.extract::<Vec<usize>>())
                .map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "zarr coordinate '{coord_key}' missing shape metadata"
                    ))
                })?;
            if original_shape != stored_shape {
                return Err(PyRuntimeError::new_err(
                    "Currently reading data with multidimension arrays is only supported when\
the multidimension coordinates are passed in full.",
                ));
            }
        }

        let mut index_arrays: Vec<Py<PyAny>> = Vec::with_capacity(adjusted_coords.len());
        for (dim_any, values_any) in adjusted_coords.iter() {
            let dim = dim_any
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("coords keys must be strings"))?;
            let stored_coord = root.get_item(dim.as_str())?;
            let stored_values = np.call_method1("asarray", (stored_coord,))?;
            let isin = np.call_method1("isin", (stored_values, values_any))?;
            let where_out = np.call_method1("where", (isin,))?;
            let indices = where_out.get_item(0)?;
            index_arrays.push(indices.unbind());
        }

        let ix_args = PyTuple::new(py, index_arrays.iter().map(|array| array.bind(py)))?;
        let ix = np.getattr("ix_")?.call1(ix_args)?;
        let data_array = root.get_item(array_name.as_str()).map_err(|_| {
            PyRuntimeError::new_err(format!("array '{array_name}' not found in zarr store"))
        })?;
        let selected = data_array.get_item(ix)?;

        let torch = import_required_read_dependency(py, "torch")?;
        let tensor_kwargs = PyDict::new(py);
        tensor_kwargs.set_item("device", &device)?;
        let tensor = torch.call_method("as_tensor", (selected,), Some(&tensor_kwargs))?;
        Ok((tensor.unbind(), coords.clone().unbind()))
    }

    /// Close the backend and flush/consolidate metadata.
    #[pyo3(
        signature = (timeout_seconds = None),
        text_signature = "($self, timeout_seconds=None)"
    )]
    fn close(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<Py<PyDict>>> {
        let timeout_seconds =
            timeout_seconds.unwrap_or_else(|| self.backend.configured_close_timeout_seconds());
        // Use detach_returning to preserve the CloseReport with timing data.
        let report = detach_returning(py, move || self.backend.close(timeout_seconds))
            .map_err(sync_error_to_pyerr)?;
        let Some(timing) = report.close_timing else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("async_drain_ns", timing.async_drain_ns.as_nanos())?;
        out.set_item("consolidate_ns", timing.consolidate_ns.as_nanos())?;
        out.set_item("teardown_ns", timing.teardown_ns.as_nanos())?;
        out.set_item("total_close_ns", timing.total_close_ns.as_nanos())?;
        Ok(Some(out.into()))
    }

    /// Return whether `close()` has already been called.
    #[pyo3(text_signature = "($self)")]
    fn is_closed(&self) -> bool {
        self.backend.is_closed()
    }

    /// Return timing information for the most recent `write()` call, if available.
    #[pyo3(text_signature = "($self)")]
    fn last_write_timing(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(timing) = self.backend.last_write_timing() else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("batch_id", timing.batch_id.as_u64())?;
        out.set_item("task_count", timing.task_count)?;
        out.set_item("worker_count", timing.worker_count)?;
        out.set_item("enqueued_task_count", timing.enqueued_task_count)?;
        out.set_item("copied_task_count", timing.copied_task_count)?;
        out.set_item("plan_ns", timing.plan_ns.as_nanos())?;
        out.set_item("buffer_init_ns", timing.buffer_init_ns.as_nanos())?;
        out.set_item("reserve_ns", timing.reserve_ns.as_nanos())?;
        out.set_item("scheduler_submit_ns", timing.scheduler_submit_ns.as_nanos())?;
        out.set_item("queue_send_ns", timing.queue_send_ns.as_nanos())?;
        out.set_item("barrier_wait_ns", timing.barrier_wait_ns.as_nanos())?;
        out.set_item("worker_acquire_ns", timing.worker_acquire_ns.as_nanos())?;
        out.set_item("worker_copy_ns", timing.worker_copy_ns.as_nanos())?;
        out.set_item("worker_wait_copy_ns", timing.worker_wait_copy_ns.as_nanos())?;
        out.set_item(
            "worker_mark_copied_ns",
            timing.worker_mark_copied_ns.as_nanos(),
        )?;
        out.set_item(
            "worker_enqueue_flush_ns",
            timing.worker_enqueue_flush_ns.as_nanos(),
        )?;
        out.set_item(
            "total_submit_write_ns",
            timing.total_submit_write_ns.as_nanos(),
        )?;
        Ok(Some(out.unbind()))
    }

    fn __repr__(&self) -> String {
        let closed = if self.backend.is_closed() {
            "True"
        } else {
            "False"
        };
        format!(
            "E2sZarrIoBackend(dataset_root='{}', zarr_format='{}', closed={})",
            self.dataset_root.display(),
            zarr_format_to_str(self.zarr_format),
            closed
        )
    }
}

fn zarr_format_to_str(value: ZarrFormat) -> &'static str {
    match value {
        ZarrFormat::V2 => "v2",
        ZarrFormat::V3 => "v3",
    }
}

fn validate_read_array_name(array_name: &str) -> PyResult<()> {
    if array_name.trim().is_empty() {
        return Err(PyValueError::new_err(
            "array_name must be a non-empty string",
        ));
    }

    let path = Path::new(array_name);
    let mut components = path.components();
    let Some(first) = components.next() else {
        return Err(PyValueError::new_err(
            "array_name must be a single safe path component",
        ));
    };
    let has_extra_components = components.next().is_some();
    if path.is_absolute()
        || has_extra_components
        || !matches!(first, Component::Normal(_))
        || array_name.contains('\\')
    {
        return Err(PyValueError::new_err(
            "array_name must be a single safe path component",
        ));
    }

    Ok(())
}

fn parse_backend_config(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SyncZarrBackendConfig> {
    let mut config = SyncZarrBackendConfig::default();
    let mut encoding_set = false;
    let mut separator_set = false;
    let mut resolved_dataset_root: Option<PathBuf> = None;

    let Some(kwargs) = kwargs else {
        return Err(PyValueError::new_err(
            "missing required keyword argument: file_name (or dataset_path)",
        ));
    };

    validate_kwargs(kwargs)?;

    if let Some(value) = kwargs.get_item("file_name")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("file_name", &value)?;
        resolved_dataset_root = Some(PathBuf::from(v));
    }
    if let Some(value) = kwargs.get_item("dataset_path")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("dataset_path", &value)?;
        let alias = PathBuf::from(v);
        if let Some(existing) = &resolved_dataset_root {
            if existing != &alias {
                return Err(PyValueError::new_err(
                    "file_name and dataset_path must match when both are provided",
                ));
            }
        } else {
            resolved_dataset_root = Some(alias);
        }
    }
    let Some(dataset_root) = resolved_dataset_root else {
        return Err(PyValueError::new_err(
            "missing required keyword argument: file_name (or dataset_path)",
        ));
    };
    config.dataset_root = Some(dataset_root);

    if let Some(value) = kwargs.get_item("parallel_coords")?
        && !value.is_none()
    {
        config
            .write_execution
            .parallel_coords_config
            .parallel_coords = Some(parse_coord_map(&value)?);
    }

    if let Some(value) = kwargs.get_item("default_parallel_coord_names")?
        && !value.is_none()
    {
        config
            .write_execution
            .parallel_coords_config
            .default_parallel_coord_names =
            parse_string_list_kwarg("default_parallel_coord_names", &value)?;
    }

    if let Some(value) = kwargs.get_item("zarr_format")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("zarr_format", &value)?;
        config.write_execution.zarr_target.zarr_format = parse_zarr_format(&v)?;
    }

    if let Some(value) = kwargs.get_item("chunk_key_encoding")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("chunk_key_encoding", &value)?;
        config.write_execution.zarr_target.chunk_key_encoding = parse_chunk_key_encoding(&v)?;
        encoding_set = true;
    }

    if let Some(value) = kwargs.get_item("chunk_key_separator")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("chunk_key_separator", &value)?;
        config.write_execution.zarr_target.chunk_key_separator = parse_chunk_key_separator(&v)?;
        separator_set = true;
    }

    if let Some(value) = kwargs.get_item("buffer_sizing_policy")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("buffer_sizing_policy", &value)?;
        config.buffer_pool.first_write_sizing.buffer_sizing_policy =
            parse_buffer_sizing_policy(&v)?;
    }

    if let Some(value) = kwargs.get_item("model_profile_hint")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("model_profile_hint", &value)?;
        config.buffer_pool.first_write_sizing.model_profile_hint =
            Some(parse_model_profile_hint(&v)?);
    }

    if let Some(value) = kwargs.get_item("min_hot_slab_buffers")?
        && !value.is_none()
    {
        config.buffer_pool.first_write_sizing.min_hot_slab_buffers =
            parse_usize_kwarg("min_hot_slab_buffers", &value)?;
    }

    if let Some(value) = kwargs.get_item("max_warm_to_hot_ratio")?
        && !value.is_none()
    {
        config.buffer_pool.first_write_sizing.max_warm_to_hot_ratio =
            parse_usize_kwarg("max_warm_to_hot_ratio", &value)?;
    }

    if let Some(value) = kwargs.get_item("hot_slab_buffers")?
        && !value.is_none()
    {
        config.buffer_pool.hot_slab_buffers =
            SizeOverride::Fixed(parse_usize_kwarg("hot_slab_buffers", &value)?);
    }

    if let Some(value) = kwargs.get_item("warm_slab_buffers")?
        && !value.is_none()
    {
        config.buffer_pool.warm_slab_buffers =
            SizeOverride::Fixed(parse_usize_kwarg("warm_slab_buffers", &value)?);
    }

    if let Some(value) = kwargs.get_item("hot_slab_ready_timeout_seconds")?
        && !value.is_none()
    {
        config.buffer_pool.hot_slab_ready_timeout_seconds =
            parse_f64_kwarg("hot_slab_ready_timeout_seconds", &value)?;
    }

    if let Some(value) = kwargs.get_item("warm_slab_warmup_policy")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("warm_slab_warmup_policy", &value)?;
        config.buffer_pool.warm_slab_warmup_policy = parse_warm_slab_warmup_policy(&v)?;
    }

    if let Some(value) = kwargs.get_item("warm_slab_failure_policy")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("warm_slab_failure_policy", &value)?;
        config.buffer_pool.warm_slab_failure_policy = parse_warm_slab_failure_policy(&v)?;
    }

    if let Some(value) = kwargs.get_item("max_pool_buffers")?
        && !value.is_none()
    {
        config.buffer_pool.max_pool_buffers = parse_usize_kwarg("max_pool_buffers", &value)?;
    }

    if let Some(value) = kwargs.get_item("max_pool_bytes")?
        && !value.is_none()
    {
        config.buffer_pool.max_pool_bytes = parse_usize_kwarg("max_pool_bytes", &value)?;
    }

    if let Some(value) = kwargs.get_item("pool_buffer_bytes")?
        && !value.is_none()
    {
        config.buffer_pool.pool_buffer_bytes =
            SizeOverride::Fixed(parse_usize_kwarg("pool_buffer_bytes", &value)?);
    }

    if let Some(value) = kwargs.get_item("pool_alignment")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("pool_alignment", &value)?;
        config.buffer_pool.pool_alignment = parse_pool_alignment(&v)?;
    }

    if let Some(value) = kwargs.get_item("pin_pooled_slabs")?
        && !value.is_none()
    {
        config.buffer_pool.pin_pooled_slabs = parse_bool_kwarg("pin_pooled_slabs", &value)?;
    }

    if let Some(value) = kwargs.get_item("cuda_register_pool_if_available")?
        && !value.is_none()
    {
        config.buffer_pool.cuda_register_pool_if_available =
            parse_bool_kwarg("cuda_register_pool_if_available", &value)?;
    }

    if let Some(value) = kwargs.get_item("cuda_register_each_slab_once")?
        && !value.is_none()
    {
        config.buffer_pool.cuda_register_each_slab_once =
            parse_bool_kwarg("cuda_register_each_slab_once", &value)?;
    }

    if let Some(value) = kwargs.get_item("max_transient_buffer_bytes")? {
        if value.is_none() {
            // Explicit None means unlimited transient buffers.
            config.buffer_pool.max_transient_buffer_bytes = None;
        } else {
            config.buffer_pool.max_transient_buffer_bytes =
                Some(parse_usize_kwarg("max_transient_buffer_bytes", &value)?);
        }
    }

    if let Some(value) = kwargs.get_item("max_inflight_transient_bytes")? {
        if value.is_none() {
            config.buffer_pool.max_inflight_transient_bytes = None;
        } else {
            config.buffer_pool.max_inflight_transient_bytes =
                Some(parse_usize_kwarg("max_inflight_transient_bytes", &value)?);
        }
    }

    if let Some(value) = kwargs.get_item("close_lease_timeout_seconds")?
        && !value.is_none()
    {
        config.buffer_pool.close_lease_timeout_seconds =
            parse_f64_kwarg("close_lease_timeout_seconds", &value)?;
    }

    if let Some(value) = kwargs.get_item("queue_capacity")?
        && !value.is_none()
    {
        config.write_execution.queue_capacity = parse_usize_kwarg("queue_capacity", &value)?;
    }

    if let Some(value) = kwargs.get_item("num_threads")?
        && !value.is_none()
    {
        config.write_execution.num_threads = parse_usize_kwarg("num_threads", &value)?;
    }

    if let Some(value) = kwargs.get_item("require_host_array_interface")?
        && !value.is_none()
    {
        config.write_execution.require_host_array_interface =
            parse_bool_kwarg("require_host_array_interface", &value)?;
    }

    if let Some(value) = kwargs.get_item("fsync_policy")?
        && !value.is_none()
    {
        let v = parse_str_kwarg("fsync_policy", &value)?;
        config.write_execution.fsync_policy = match v.as_str() {
            "always" => FsyncPolicy::Always,
            "never" => FsyncPolicy::Never,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported fsync_policy: {v} (expected 'always' or 'never')"
                )));
            }
        };
    }

    // Apply format-specific defaults when caller only specifies zarr_format.
    if config.write_execution.zarr_target.zarr_format == ZarrFormat::V3 {
        if !encoding_set {
            config.write_execution.zarr_target.chunk_key_encoding = ChunkKeyEncoding::Default;
        }
        if !separator_set {
            config.write_execution.zarr_target.chunk_key_separator = ChunkKeySeparator::Slash;
        }
    } else {
        if !encoding_set {
            config.write_execution.zarr_target.chunk_key_encoding = ChunkKeyEncoding::V2;
        }
        if !separator_set {
            config.write_execution.zarr_target.chunk_key_separator = ChunkKeySeparator::Dot;
        }
    }

    Ok(config)
}

fn validate_kwargs(kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    for (key, _) in kwargs.iter() {
        let key_str = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("constructor kwargs keys must be strings"))?;
        if !SUPPORTED_KWARGS.contains(&key_str.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unsupported keyword argument: {key_str}"
            )));
        }
    }
    Ok(())
}

fn parse_zarr_format(value: &str) -> PyResult<ZarrFormat> {
    match value {
        "v2" => Ok(ZarrFormat::V2),
        "v3" => Ok(ZarrFormat::V3),
        _ => Err(PyValueError::new_err(format!(
            "invalid zarr_format: {value} (expected 'v2' or 'v3')"
        ))),
    }
}

fn parse_chunk_key_encoding(value: &str) -> PyResult<ChunkKeyEncoding> {
    match value {
        "v2" => Ok(ChunkKeyEncoding::V2),
        "default" => Ok(ChunkKeyEncoding::Default),
        _ => Err(PyValueError::new_err(format!(
            "invalid chunk_key_encoding: {value} (expected 'v2' or 'default')"
        ))),
    }
}

fn parse_chunk_key_separator(value: &str) -> PyResult<ChunkKeySeparator> {
    match value {
        "." => Ok(ChunkKeySeparator::Dot),
        "/" => Ok(ChunkKeySeparator::Slash),
        _ => Err(PyValueError::new_err(format!(
            "invalid chunk_key_separator: {value} (expected '.' or '/')"
        ))),
    }
}

fn parse_buffer_sizing_policy(value: &str) -> PyResult<BufferSizingPolicy> {
    match value {
        "fixed_only" => Ok(BufferSizingPolicy::FixedOnly),
        "first_write_model_aware_auto" => Ok(BufferSizingPolicy::FirstWriteModelAwareAuto),
        _ => Err(PyValueError::new_err(format!(
            "invalid buffer_sizing_policy: {value} (expected 'fixed_only' or 'first_write_model_aware_auto')"
        ))),
    }
}

fn parse_model_profile_hint(value: &str) -> PyResult<ModelProfileHint> {
    match value {
        "fcn" => Ok(ModelProfileHint::Fcn),
        "dlwp" => Ok(ModelProfileHint::Dlwp),
        "sfno" => Ok(ModelProfileHint::Sfno),
        "pangu" => Ok(ModelProfileHint::Pangu),
        "graphcast_small" => Ok(ModelProfileHint::GraphCastSmall),
        "stormcast" => Ok(ModelProfileHint::StormCast),
        "precipitation_afno" => Ok(ModelProfileHint::PrecipitationAfno),
        "corrdiff_taiwan" => Ok(ModelProfileHint::CorrDiffTaiwan),
        _ => Err(PyValueError::new_err(format!(
            "invalid model_profile_hint: {value}"
        ))),
    }
}

fn parse_warm_slab_warmup_policy(value: &str) -> PyResult<WarmSlabWarmupPolicy> {
    match value {
        "on_first_write_background" => Ok(WarmSlabWarmupPolicy::OnFirstWriteBackground),
        _ => Err(PyValueError::new_err(format!(
            "invalid warm_slab_warmup_policy: {value} (expected 'on_first_write_background')"
        ))),
    }
}

fn parse_warm_slab_failure_policy(value: &str) -> PyResult<WarmSlabFailurePolicy> {
    match value {
        "degrade_continue" => Ok(WarmSlabFailurePolicy::DegradeContinue),
        _ => Err(PyValueError::new_err(format!(
            "invalid warm_slab_failure_policy: {value} (expected 'degrade_continue')"
        ))),
    }
}

fn parse_pool_alignment(value: &str) -> PyResult<PoolAlignment> {
    match value {
        "4k" | "4KiB" | "align4kib" => Ok(PoolAlignment::Align4KiB),
        "64k" | "64KiB" | "align64kib" => Ok(PoolAlignment::Align64KiB),
        _ => Err(PyValueError::new_err(format!(
            "invalid pool_alignment: {value} (expected '4k' or '64k')"
        ))),
    }
}

fn parse_str_kwarg(name: &str, value: &Bound<'_, PyAny>) -> PyResult<String> {
    value
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err(format!("'{name}' must be a string")))
}

fn parse_usize_kwarg(name: &str, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    value
        .extract::<usize>()
        .map_err(|_| PyTypeError::new_err(format!("'{name}' must be an integer >= 0")))
}

fn parse_f64_kwarg(name: &str, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err(format!("'{name}' must be a float")))
}

fn parse_bool_kwarg(name: &str, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value
        .extract::<bool>()
        .map_err(|_| PyTypeError::new_err(format!("'{name}' must be a bool")))
}

fn parse_read_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(device) = device else {
        return Ok("cpu".to_string());
    };
    if device.is_none() {
        return Ok("cpu".to_string());
    }
    if let Ok(value) = device.extract::<String>() {
        if value.trim().is_empty() {
            return Err(PyValueError::new_err("device must not be empty"));
        }
        return Ok(value);
    }
    let value = device
        .str()
        .and_then(|s| s.extract::<String>())
        .map_err(|_| {
            PyTypeError::new_err("device must be a string or an object with string representation")
        })?;
    if value.trim().is_empty() {
        return Err(PyValueError::new_err("device must not be empty"));
    }
    Ok(value)
}

fn import_required_read_dependency<'py>(
    py: Python<'py>,
    module_name: &str,
) -> PyResult<Bound<'py, PyModule>> {
    py.import(module_name).map_err(|_| {
        PyImportError::new_err(format!(
            "read() requires optional dependency '{module_name}'. Install with: pip install e2s-zarr-io[read]"
        ))
    })
}

fn root_contains(root: &Bound<'_, PyAny>, key: &str) -> PyResult<bool> {
    root.call_method1("__contains__", (key,))?.extract::<bool>()
}

fn normalize_coord_mapping<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    np: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyDict>> {
    let coords_dict = coords
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("coords must be a dict[str, sequence]"))?;
    let normalized = PyDict::new(py);
    for (k, v) in coords_dict.iter() {
        let key = k
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("coords keys must be strings"))?;
        let as_array = np.call_method1("asarray", (v,))?;
        normalized.set_item(key, as_array)?;
    }
    Ok(normalized)
}

fn convert_multidim_to_singledim<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyDict>,
    np: &Bound<'py, PyModule>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let adjusted_coords = PyDict::new(py);
    let mapping = PyDict::new(py);

    let mut items: Vec<(String, Py<PyAny>)> = Vec::with_capacity(coords.len());
    for (key_any, value_any) in coords.iter() {
        let key = key_any
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("coords keys must be strings"))?;
        items.push((key, value_any.unbind()));
    }

    let mut i = 0usize;
    while i < items.len() {
        let (key, value) = (&items[i].0, items[i].1.bind(py));
        if key.starts_with('_') {
            i += 1;
            continue;
        }

        let ndim = value
            .getattr("ndim")
            .and_then(|n| n.extract::<usize>())
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "coords['{key}'] must provide an integer ndim attribute"
                ))
            })?;

        if ndim < 2 {
            adjusted_coords.set_item(key, value)?;
            i += 1;
            continue;
        }

        let shape = value
            .getattr("shape")
            .and_then(|s| s.extract::<Vec<usize>>())
            .map_err(|_| {
                PyTypeError::new_err(format!("coords['{key}'] must provide a shape attribute"))
            })?;
        let mut generated_index_names: Vec<String> = Vec::with_capacity(ndim);

        for j in 0..ndim {
            if i + j >= items.len() {
                return Err(PyValueError::new_err(
                    "multidimensional coordinates must be followed by exactly ndim coordinates with identical shape",
                ));
            }

            let grouped_key = &items[i + j].0;
            let grouped_value = items[i + j].1.bind(py);
            let grouped_shape = grouped_value
                .getattr("shape")
                .and_then(|s| s.extract::<Vec<usize>>())
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "coords['{grouped_key}'] must provide a shape attribute"
                    ))
                })?;
            if grouped_shape != shape {
                return Err(PyValueError::new_err(
                    "multidimensional coordinates must be followed by exactly ndim coordinates with identical shape",
                ));
            }

            let axis_len = shape.get(j).copied().ok_or_else(|| {
                PyRuntimeError::new_err("ndim/shape mismatch while normalizing coordinates")
            })?;
            let index_name = format!("i{grouped_key}");
            let axis_index = np.call_method1("arange", (axis_len,))?;
            adjusted_coords.set_item(index_name.as_str(), axis_index)?;
            generated_index_names.push(index_name);
        }

        let mapping_value = PyList::new(py, generated_index_names.iter().map(String::as_str))?;
        for j in 0..ndim {
            let grouped_key = &items[i + j].0;
            mapping.set_item(grouped_key.as_str(), &mapping_value)?;
        }

        i += ndim;
    }

    Ok((adjusted_coords, mapping))
}

fn parse_array_names(array_name: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = array_name.extract::<String>() {
        return Ok(vec![name]);
    }

    if let Ok(names) = array_name.extract::<Vec<String>>() {
        return Ok(names);
    }

    if let Ok(list) = array_name.cast::<PyList>() {
        let mut names = Vec::with_capacity(list.len());
        for item in list.iter() {
            names
                .push(item.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("array_name list entries must be strings")
                })?);
        }
        return Ok(names);
    }

    if let Ok(tuple) = array_name.cast::<PyTuple>() {
        let mut names = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            names.push(
                item.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("array_name tuple entries must be strings")
                })?,
            );
        }
        return Ok(names);
    }

    Err(PyTypeError::new_err(
        "array_name must be a string or sequence of strings",
    ))
}

fn parse_coord_map(coords: &Bound<'_, PyAny>) -> PyResult<CoordMap> {
    let dict = coords
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("coords must be a dict[str, sequence]"))?;

    let mut parsed = CoordMap::new();
    for (k, v) in dict.iter() {
        let key = k
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("coords keys must be strings"))?;
        parsed
            .insert(key, parse_coord_values(&v)?)
            .map_err(sync_error_to_pyerr)?;
    }
    Ok(parsed)
}

fn parse_string_list_kwarg(key: &str, value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(values) = value.extract::<Vec<String>>() {
        if values.iter().all(|item| !item.trim().is_empty()) {
            return Ok(values);
        }
    }

    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err(format!("{key} must be a list[str]")))?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        let value = item
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err(format!("{key} must be a list[str]")))?;
        if value.trim().is_empty() {
            return Err(PyValueError::new_err(format!(
                "{key} entries must not be empty"
            )));
        }
        values.push(value);
    }
    Ok(values)
}

fn parse_coord_values(value: &Bound<'_, PyAny>) -> PyResult<CoordValues> {
    if let Some(values) = parse_temporal_numpy_coord_values(value)? {
        return Ok(values);
    }
    if let Some(values) = parse_typed_numpy_coord_values(value)? {
        return Ok(values);
    }

    let converted = if value.hasattr("tolist")? {
        Some(value.call_method0("tolist")?)
    } else {
        None
    };
    let candidate = converted.as_ref().map_or(value, Bound::as_any);

    if let Ok(values) = candidate.extract::<Vec<i64>>() {
        return Ok(CoordValues::I64(values));
    }
    if let Ok(values) = candidate.extract::<Vec<u64>>() {
        return Ok(CoordValues::U64(values));
    }
    if let Ok(values) = candidate.extract::<Vec<f64>>() {
        return Ok(CoordValues::F64(values));
    }
    if let Ok(values) = candidate.extract::<Vec<String>>() {
        return Ok(CoordValues::Utf8(values));
    }
    if let Ok(v) = candidate.extract::<i64>() {
        return Ok(CoordValues::I64(vec![v]));
    }
    if let Ok(v) = candidate.extract::<u64>() {
        return Ok(CoordValues::U64(vec![v]));
    }
    if let Ok(v) = candidate.extract::<f64>() {
        return Ok(CoordValues::F64(vec![v]));
    }
    if let Ok(v) = candidate.extract::<String>() {
        return Ok(CoordValues::Utf8(vec![v]));
    }

    Err(PyTypeError::new_err(
        "coord values must be a sequence (or scalar) of int/float/str",
    ))
}

fn parse_typed_numpy_coord_values(value: &Bound<'_, PyAny>) -> PyResult<Option<CoordValues>> {
    let Some(dtype) = numpy_dtype_string(value)? else {
        return Ok(None);
    };
    let Some(data_type) = parse_typestr_dtype(&dtype) else {
        return Ok(None);
    };
    let converted = value.call_method0("tolist")?;
    Ok(Some(match data_type {
        DataType::Int32 => CoordValues::I32(
            converted
                .extract::<Vec<i32>>()
                .or_else(|_| converted.extract::<i32>().map(|value| vec![value]))?,
        ),
        DataType::Int64 => CoordValues::I64(
            converted
                .extract::<Vec<i64>>()
                .or_else(|_| converted.extract::<i64>().map(|value| vec![value]))?,
        ),
        DataType::UInt32 => CoordValues::U32(
            converted
                .extract::<Vec<u32>>()
                .or_else(|_| converted.extract::<u32>().map(|value| vec![value]))?,
        ),
        DataType::UInt64 => CoordValues::U64(
            converted
                .extract::<Vec<u64>>()
                .or_else(|_| converted.extract::<u64>().map(|value| vec![value]))?,
        ),
        DataType::Float32 => CoordValues::F32(
            converted
                .extract::<Vec<f32>>()
                .or_else(|_| converted.extract::<f32>().map(|value| vec![value]))?,
        ),
        DataType::Float64 => CoordValues::F64(
            converted
                .extract::<Vec<f64>>()
                .or_else(|_| converted.extract::<f64>().map(|value| vec![value]))?,
        ),
        DataType::DatetimeNs | DataType::TimedeltaNs => return Ok(None),
        _ => return Ok(None),
    }))
}

fn parse_temporal_numpy_coord_values(value: &Bound<'_, PyAny>) -> PyResult<Option<CoordValues>> {
    if !value.hasattr("dtype")? {
        return Ok(None);
    }

    let dtype = value.getattr("dtype")?;
    let kind = match dtype
        .getattr("kind")
        .and_then(|kind| kind.extract::<String>())
    {
        Ok(kind) => kind,
        Err(_) => return Ok(None),
    };
    let (temporal_dtype, is_datetime) = match kind.as_str() {
        "M" => ("datetime64[ns]", true),
        "m" => ("timedelta64[ns]", false),
        _ => return Ok(None),
    };

    let normalized = value.call_method1("astype", (temporal_dtype,))?;
    let int_values = normalized.call_method1("astype", ("int64",))?;
    let converted = int_values.call_method0("tolist")?;
    let values = if let Ok(values) = converted.extract::<Vec<i64>>() {
        values
    } else if let Ok(value) = converted.extract::<i64>() {
        vec![value]
    } else {
        return Err(PyTypeError::new_err(
            "temporal coord values must be datetime64/timedelta64 scalars or sequences",
        ));
    };

    Ok(Some(if is_datetime {
        CoordValues::DatetimeNs(values)
    } else {
        CoordValues::TimedeltaNs(values)
    }))
}

fn parse_input_arrays_with_target_dtypes(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    require_host_array_interface: bool,
    target_dtypes: &[Option<DataType>],
) -> PyResult<Vec<InputArray>> {
    if is_array_like(x)? {
        let target = target_dtypes.first().copied().flatten();
        return parse_one_input_array_with_target(py, x, require_host_array_interface, target)
            .map(|(array, _dtype)| vec![array]);
    }

    if let Ok(list) = x.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for (index, item) in list.iter().enumerate() {
            let target = target_dtypes.get(index).copied().flatten();
            let (array, _dtype) =
                parse_one_input_array_with_target(py, &item, require_host_array_interface, target)?;
            out.push(array);
        }
        return Ok(out);
    }

    if let Ok(tuple) = x.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for (index, item) in tuple.iter().enumerate() {
            let target = target_dtypes.get(index).copied().flatten();
            let (array, _dtype) =
                parse_one_input_array_with_target(py, &item, require_host_array_interface, target)?;
            out.push(array);
        }
        return Ok(out);
    }

    Err(PyTypeError::new_err(
        "x must be an array-like object or a list/tuple of array-like objects",
    ))
}

fn parse_input_arrays_with_dtypes(
    x: &Bound<'_, PyAny>,
    require_host_array_interface: bool,
) -> PyResult<(Vec<InputArray>, Vec<DataType>)> {
    if is_array_like(x)? {
        let (array, dtype) = parse_one_input_array(x, require_host_array_interface)?;
        return Ok((vec![array], vec![dtype]));
    }

    if let Ok(list) = x.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        let mut dtypes = Vec::with_capacity(list.len());
        for item in list.iter() {
            let (array, dtype) = parse_one_input_array(&item, require_host_array_interface)?;
            out.push(array);
            dtypes.push(dtype);
        }
        return Ok((out, dtypes));
    }

    if let Ok(tuple) = x.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        let mut dtypes = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            let (array, dtype) = parse_one_input_array(&item, require_host_array_interface)?;
            out.push(array);
            dtypes.push(dtype);
        }
        return Ok((out, dtypes));
    }

    Err(PyTypeError::new_err(
        "x must be an array-like object or a list/tuple of array-like objects",
    ))
}

fn is_array_like(any: &Bound<'_, PyAny>) -> PyResult<bool> {
    let has_nbytes = any.hasattr("nbytes")?;
    let has_host = any.hasattr("__array_interface__")? || any.hasattr("tobytes")?;
    let has_cuda = any.hasattr("__cuda_array_interface__")?;
    Ok(has_nbytes && (has_host || has_cuda))
}

fn parse_one_input_array_with_target(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    require_host_array_interface: bool,
    target_dtype: Option<DataType>,
) -> PyResult<(InputArray, DataType)> {
    let dtype = parse_array_dtype(array)?;
    if let Some(target_dtype) = target_dtype
        && dtype != target_dtype
    {
        let casted = cast_array_to_dtype(py, array, target_dtype)?;
        return parse_owned_host_input_array(casted.bind(py), target_dtype);
    }
    parse_one_input_array(array, require_host_array_interface)
}

fn parse_one_input_array(
    array: &Bound<'_, PyAny>,
    require_host_array_interface: bool,
) -> PyResult<(InputArray, DataType)> {
    let nbytes = array
        .getattr("nbytes")
        .and_then(|v| v.extract::<usize>())
        .map_err(|_| PyTypeError::new_err("array-like object must expose integer 'nbytes'"))?;
    let dtype = parse_array_dtype(array)?;

    if array.hasattr("__cuda_array_interface__")? {
        let source = parse_cuda_source(array, nbytes)?;
        let parsed = InputArray { nbytes, source };
        parsed.validate().map_err(sync_error_to_pyerr)?;
        return Ok((parsed, dtype));
    }
    let source = parse_host_source(array, nbytes, require_host_array_interface)?;
    let parsed = InputArray { nbytes, source };
    parsed.validate().map_err(sync_error_to_pyerr)?;
    Ok((parsed, dtype))
}

fn parse_owned_host_input_array(
    array: &Bound<'_, PyAny>,
    dtype: DataType,
) -> PyResult<(InputArray, DataType)> {
    let nbytes = array
        .getattr("nbytes")
        .and_then(|v| v.extract::<usize>())
        .map_err(|_| PyTypeError::new_err("array-like object must expose integer 'nbytes'"))?;
    let source = parse_host_bytes_source(array, nbytes)?;
    let parsed = InputArray { nbytes, source };
    parsed.validate().map_err(sync_error_to_pyerr)?;
    Ok((parsed, dtype))
}

fn numpy_dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Bool => "bool",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        DataType::Float16 => "float16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::DatetimeNs => "datetime64[ns]",
        DataType::TimedeltaNs => "timedelta64[ns]",
    }
}

fn torch_dtype_name(dtype: DataType) -> Option<&'static str> {
    match dtype {
        DataType::Bool => Some("bool"),
        DataType::Int8 => Some("int8"),
        DataType::Int16 => Some("int16"),
        DataType::Int32 => Some("int32"),
        DataType::Int64 => Some("int64"),
        DataType::UInt8 => Some("uint8"),
        DataType::UInt16 => Some("uint16"),
        DataType::Float16 => Some("float16"),
        DataType::Float32 => Some("float32"),
        DataType::Float64 => Some("float64"),
        DataType::UInt32 | DataType::UInt64 | DataType::DatetimeNs | DataType::TimedeltaNs => None,
    }
}

fn cast_array_to_dtype(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    target_dtype: DataType,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    if array.hasattr("to")?
        && array.hasattr("cpu")?
        && let Some(torch_dtype_name) = torch_dtype_name(target_dtype)
    {
        let torch = py.import("torch")?;
        let torch_dtype = torch.getattr(torch_dtype_name)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch_dtype)?;
        let casted = array.call_method("to", (), Some(&kwargs))?;
        let cpu = casted.call_method0("cpu")?;
        let numpy_array = cpu.call_method0("numpy")?;
        return np
            .call_method1("ascontiguousarray", (numpy_array,))
            .map(Bound::unbind);
    }

    let as_array = np.call_method1("asarray", (array,))?;
    let casted = as_array.call_method1("astype", (numpy_dtype_name(target_dtype),))?;
    np.call_method1("ascontiguousarray", (casted,))
        .map(Bound::unbind)
}

fn numpy_dtype_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if !value.hasattr("dtype")? {
        return Ok(None);
    }
    let dtype = value.getattr("dtype")?;
    if let Ok(dtype_str) = dtype.getattr("str").and_then(|v| v.extract::<String>()) {
        return Ok(Some(dtype_str));
    }
    Ok(None)
}

fn parse_array_dtype(array: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let mut saw_dtype = false;
    if let Some(dtype) = numpy_dtype_string(array)?
        && let Some(parsed) = parse_typestr_dtype(&dtype)
    {
        return Ok(parsed);
    } else if array.hasattr("dtype")? {
        saw_dtype = true;
    }
    if let Ok(iface_any) = array.getattr("__array_interface__")
        && let Ok(iface) = iface_any.cast::<PyDict>()
        && let Some(typestr) = iface.get_item("typestr")?
        && let Ok(typestr) = typestr.extract::<String>()
        && let Some(parsed) = parse_typestr_dtype(&typestr)
    {
        return Ok(parsed);
    } else if array.hasattr("__array_interface__")? {
        saw_dtype = true;
    }
    if let Ok(iface_any) = array.getattr("__cuda_array_interface__")
        && let Ok(iface) = iface_any.cast::<PyDict>()
        && let Some(typestr) = iface.get_item("typestr")?
        && let Ok(typestr) = typestr.extract::<String>()
        && let Some(parsed) = parse_typestr_dtype(&typestr)
    {
        return Ok(parsed);
    } else if array.hasattr("__cuda_array_interface__")? {
        saw_dtype = true;
    }
    if !saw_dtype {
        return Ok(DataType::default());
    }
    Err(PyTypeError::new_err(
        "array-like object must expose a supported dtype/typestr",
    ))
}

fn parse_typestr_dtype(typestr: &str) -> Option<DataType> {
    let normalized = typestr.trim();
    let endian = normalized.chars().next()?;
    if endian == '>' {
        return None;
    }
    let body = normalized.trim_start_matches(['<', '|', '=']);
    match body {
        "b1" | "?" => Some(DataType::Bool),
        "i1" => Some(DataType::Int8),
        "i2" => Some(DataType::Int16),
        "i4" => Some(DataType::Int32),
        "i8" => Some(DataType::Int64),
        "u1" => Some(DataType::UInt8),
        "u2" => Some(DataType::UInt16),
        "u4" => Some(DataType::UInt32),
        "u8" => Some(DataType::UInt64),
        "f2" => Some(DataType::Float16),
        "f4" => Some(DataType::Float32),
        "f8" => Some(DataType::Float64),
        "M8[ns]" => Some(DataType::DatetimeNs),
        "m8[ns]" => Some(DataType::TimedeltaNs),
        _ => None,
    }
}

fn parse_host_source(
    array: &Bound<'_, PyAny>,
    nbytes: usize,
    require_host_array_interface: bool,
) -> PyResult<InputArraySource> {
    if let Ok(iface_any) = array.getattr("__array_interface__") {
        let iface = iface_any
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("__array_interface__ must be a dict"))?;
        let data = iface.get_item("data")?.ok_or_else(|| {
            PyTypeError::new_err("__array_interface__ missing required 'data' field")
        })?;
        let (ptr, _readonly): (u64, bool) = data.extract().map_err(|_| {
            PyTypeError::new_err("__array_interface__['data'] must be (ptr: int, readonly: bool)")
        })?;
        if ptr == 0 {
            return Err(PyValueError::new_err(
                "__array_interface__ data pointer cannot be 0",
            ));
        }
        if let Some(strides) = iface.get_item("strides")?
            && !strides.is_none()
        {
            if require_host_array_interface {
                return Err(PyValueError::new_err(
                    "host array-like object must be C-contiguous (array_interface['strides'] must be None)",
                ));
            }
        } else {
            // SAFETY:
            // - `ptr` comes from Python `__array_interface__['data']` and is non-zero.
            // - The Python source object is held by the caller for the full blocking
            //   write copy-barrier, so the pointed host allocation remains readable.
            return Ok(unsafe { InputArraySource::from_host_buffer_ptr(ptr) });
        }
    }

    if require_host_array_interface {
        return Err(PyTypeError::new_err(
            "host array-like object must expose __array_interface__ on the hot path; pass require_host_array_interface=False to allow slower tobytes() fallback",
        ));
    }

    if !array.hasattr("tobytes")? {
        return Err(PyTypeError::new_err(
            "host array-like object must expose __array_interface__ or provide tobytes()",
        ));
    }

    let source = parse_host_bytes_source(array, nbytes)?;
    Ok(source)
}

fn parse_host_bytes_source(array: &Bound<'_, PyAny>, nbytes: usize) -> PyResult<InputArraySource> {
    let bytes_any = array
        .call_method1("tobytes", ("C",))
        .or_else(|_| array.call_method0("tobytes"))
        .map_err(|_| PyTypeError::new_err("failed to call array.tobytes()"))?;
    let payload = bytes_any
        .extract::<Vec<u8>>()
        .map_err(|_| PyTypeError::new_err("array.tobytes() must return bytes"))?;
    if payload.len() != nbytes {
        return Err(PyValueError::new_err(format!(
            "array nbytes mismatch: nbytes={} but tobytes returned {} bytes",
            nbytes,
            payload.len()
        )));
    }
    Ok(InputArraySource::HostBytes(payload.into()))
}

fn parse_cuda_source(array: &Bound<'_, PyAny>, nbytes: usize) -> PyResult<InputArraySource> {
    let iface_any = array.getattr("__cuda_array_interface__").map_err(|_| {
        PyTypeError::new_err("CUDA array-like object must expose __cuda_array_interface__")
    })?;
    let iface = iface_any
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("__cuda_array_interface__ must be a dict"))?;

    let data = iface.get_item("data")?.ok_or_else(|| {
        PyTypeError::new_err("__cuda_array_interface__ missing required 'data' field")
    })?;
    let (ptr, _readonly): (u64, bool) = data.extract().map_err(|_| {
        PyTypeError::new_err("__cuda_array_interface__['data'] must be (ptr: int, readonly: bool)")
    })?;
    if ptr == 0 {
        return Err(PyValueError::new_err(
            "__cuda_array_interface__ data pointer cannot be 0",
        ));
    }

    let shape = iface
        .get_item("shape")?
        .ok_or_else(|| {
            PyTypeError::new_err("__cuda_array_interface__ missing required 'shape' field")
        })?
        .extract::<Vec<usize>>()
        .map_err(|_| {
            PyTypeError::new_err("__cuda_array_interface__['shape'] must be tuple/list[int]")
        })?;
    let typestr = iface
        .get_item("typestr")?
        .ok_or_else(|| {
            PyTypeError::new_err("__cuda_array_interface__ missing required 'typestr' field")
        })?
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err("__cuda_array_interface__['typestr'] must be str"))?;
    let itemsize = parse_typestr_itemsize(&typestr).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "__cuda_array_interface__['typestr'] has unsupported itemsize encoding: {typestr}"
        ))
    })?;
    let element_count = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            PyValueError::new_err("__cuda_array_interface__ shape product overflows usize")
        })
    })?;
    let expected_nbytes = element_count.checked_mul(itemsize).ok_or_else(|| {
        PyValueError::new_err("__cuda_array_interface__ nbytes calculation overflows usize")
    })?;
    if expected_nbytes != nbytes {
        return Err(PyValueError::new_err(format!(
            "CUDA array nbytes mismatch: nbytes={} but shape/typestr imply {} bytes",
            nbytes, expected_nbytes
        )));
    }

    let producer_stream = match iface.get_item("stream")? {
        Some(stream_any) if !stream_any.is_none() => {
            Some(stream_any.extract::<u64>().map_err(|_| {
                PyTypeError::new_err("__cuda_array_interface__['stream'] must be int or None")
            })?)
        }
        _ => None,
    };

    let device_ordinal = if let Some(v) = iface.get_item("device_ordinal")? {
        v.extract::<i32>().map_err(|_| {
            PyTypeError::new_err("__cuda_array_interface__['device_ordinal'] must be int")
        })?
    } else if let Some(v) = iface.get_item("device")? {
        if let Ok((ordinal, _index)) = v.extract::<(i32, i32)>() {
            ordinal
        } else {
            v.extract::<i32>().map_err(|_| {
                PyTypeError::new_err(
                    "__cuda_array_interface__['device'] must be int or tuple[int, int]",
                )
            })?
        }
    } else {
        0
    };

    Ok(InputArraySource::CudaDevicePtr {
        ptr,
        device_ordinal,
        producer_stream,
    })
}

fn parse_typestr_itemsize(typestr: &str) -> Option<usize> {
    let digits_reversed: String = typestr
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits_reversed.is_empty() {
        return None;
    }
    let digits: String = digits_reversed.chars().rev().collect();
    digits.parse::<usize>().ok().filter(|size| *size > 0)
}

fn sync_error_to_pyerr(err: SyncWriteError) -> PyErr {
    match err {
        SyncWriteError::Validation { .. }
        | SyncWriteError::ContractViolation { .. }
        | SyncWriteError::UnknownParallelCoord { .. }
        | SyncWriteError::UnsupportedInputStabilityPolicy { .. }
        | SyncWriteError::UnsupportedZarrTargetConfig { .. }
        | SyncWriteError::ChunkKeyConflict { .. }
        | SyncWriteError::TransientAllocationLimitExceeded { .. }
        | SyncWriteError::TransientInFlightLimitExceeded { .. } => {
            PyValueError::new_err(err.to_string())
        }
        SyncWriteError::ObjectClosed => PyRuntimeError::new_err(err.to_string()),
        SyncWriteError::IoFailed { .. } | SyncWriteError::MetadataConsolidationFailed { .. } => {
            PyOSError::new_err(err.to_string())
        }
        SyncWriteError::LeaseReturnTimeout { .. } => PyTimeoutError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

pub(crate) fn register_python_bindings_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyE2sZarrIoBackend>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::types::{
        BufferPoolConfig, DataType, FirstWriteSizingConfig, ParallelCoordsConfig,
        WriteExecutionConfig, ZarrTargetConfig,
    };
    use pyo3::types::{PyBytes, PyDict, PyString};

    fn unique_dataset_root(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}_{}", std::process::id()))
    }

    fn assert_pyerr_type(py: Python<'_>, err: PyErr, expected_type_name: &str) {
        let type_repr = err.get_type(py).to_string();
        assert!(
            type_repr.contains(expected_type_name),
            "expected error type containing '{expected_type_name}', got '{type_repr}'"
        );
    }

    #[test]
    fn registered_dtype_store_preserves_previous_add_array_batches() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_dtype_registry_batches");
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            let backend = PyE2sZarrIoBackend::new(Some(&kwargs)).expect("backend constructor");

            backend
                .store_registered_array_dtypes(&["t2m".to_string()], &[DataType::Float32])
                .expect("store first dtype batch");
            backend
                .store_registered_array_dtypes(&["u10m".to_string()], &[DataType::Int16])
                .expect("store second dtype batch");

            assert_eq!(
                backend
                    .registered_dtypes_for(&["t2m".to_string(), "u10m".to_string()])
                    .expect("read registered dtypes"),
                vec![Some(DataType::Float32), Some(DataType::Int16)]
            );

            backend.close(py, Some(5.0)).expect("close should succeed");
            let _ = fs::remove_dir_all(&dataset_root);
        });
    }

    #[test]
    fn parses_model_profile_hint_enum_values() {
        assert_eq!(
            parse_model_profile_hint("graphcast_small").expect("valid profile"),
            ModelProfileHint::GraphCastSmall
        );
        assert!(parse_model_profile_hint("unknown-profile").is_err());
    }

    #[test]
    fn parses_zarr_format_values() {
        assert_eq!(parse_zarr_format("v2").expect("v2"), ZarrFormat::V2);
        assert_eq!(parse_zarr_format("v3").expect("v3"), ZarrFormat::V3);
        assert!(parse_zarr_format("v4").is_err());
    }

    #[test]
    fn parse_backend_config_applies_num_threads_override() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_num_threads_parse");
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            kwargs.set_item("num_threads", 3).expect("set num_threads");

            let config =
                parse_backend_config(Some(&kwargs)).expect("parse_backend_config should succeed");
            assert_eq!(config.write_execution.num_threads, 3);
        });
    }

    #[test]
    fn parse_backend_config_applies_default_parallel_coord_names_override() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_default_parallel_names_parse");
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            kwargs
                .set_item(
                    "default_parallel_coord_names",
                    ["ensemble", "time", "lead_time"],
                )
                .expect("set default_parallel_coord_names");

            let config =
                parse_backend_config(Some(&kwargs)).expect("parse_backend_config should succeed");
            assert_eq!(
                config
                    .write_execution
                    .parallel_coords_config
                    .default_parallel_coord_names,
                ["ensemble", "time", "lead_time"]
            );
        });
    }

    #[test]
    fn supported_kwargs_allowlist_matches_parse_backend_config_keys() {
        const NEEDLE: &str = "kwargs.get_item(\"";
        const TERMINATOR: &str = "\")?";
        let source = include_str!("python_bindings.rs");
        let mut parser_keys: Vec<&str> = Vec::new();
        let mut cursor = source;

        while let Some(start_idx) = cursor.find(NEEDLE) {
            let after_needle = &cursor[start_idx + NEEDLE.len()..];
            let end_idx = after_needle
                .find(TERMINATOR)
                .expect("kwargs.get_item(...) key should terminate with \")?\"");
            parser_keys.push(&after_needle[..end_idx]);
            cursor = &after_needle[end_idx + TERMINATOR.len()..];
        }

        let mut parser_unique_in_order = Vec::new();
        for key in parser_keys {
            if parser_unique_in_order.last().copied() != Some(key)
                && !parser_unique_in_order.contains(&key)
            {
                parser_unique_in_order.push(key);
            }
        }

        assert_eq!(
            parser_unique_in_order.len(),
            parser_unique_in_order
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "parse_backend_config kwargs.get_item keys should not contain duplicates",
        );
        assert_eq!(
            SUPPORTED_KWARGS.len(),
            SUPPORTED_KWARGS
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "SUPPORTED_KWARGS should not contain duplicates",
        );
        assert_eq!(
            parser_unique_in_order, SUPPORTED_KWARGS,
            "SUPPORTED_KWARGS must stay in sync with parse_backend_config kwargs handling and ordering",
        );
    }

    #[test]
    fn supported_kwargs_covers_all_configurable_fields() {
        let SyncZarrBackendConfig {
            dataset_root: _dataset_root,
            write_execution,
            buffer_pool,
        } = SyncZarrBackendConfig::default();

        // Keep these patterns exhaustive so newly-added config fields force an
        // explicit decision on Python kwargs exposure.
        let WriteExecutionConfig {
            num_threads: _num_threads,
            input_stability_policy: _input_stability_policy,
            scheduler_backend: _scheduler_backend,
            queue_capacity: _queue_capacity,
            parallel_coords_config,
            planner_algorithm: _planner_algorithm,
            planner_caches: _planner_caches,
            zarr_target,
            require_host_array_interface: _require_host_array_interface,
            fsync_policy: _fsync_policy,
        } = write_execution;
        let ParallelCoordsConfig {
            parallel_coords: _parallel_coords,
            default_parallel_coord_names: _default_parallel_coord_names,
        } = parallel_coords_config;
        let ZarrTargetConfig {
            zarr_format: _zarr_format,
            chunk_key_encoding: _chunk_key_encoding,
            chunk_key_separator: _chunk_key_separator,
        } = zarr_target;

        let BufferPoolConfig {
            max_pool_buffers: _max_pool_buffers,
            max_pool_bytes: _max_pool_bytes,
            pool_buffer_bytes: _pool_buffer_bytes,
            hot_slab_buffers: _hot_slab_buffers,
            warm_slab_buffers: _warm_slab_buffers,
            first_write_sizing,
            pool_alignment: _pool_alignment,
            pin_pooled_slabs: _pin_pooled_slabs,
            cuda_register_pool_if_available: _cuda_register_pool_if_available,
            cuda_register_each_slab_once: _cuda_register_each_slab_once,
            hot_slab_ready_timeout_seconds: _hot_slab_ready_timeout_seconds,
            warm_slab_warmup_policy: _warm_slab_warmup_policy,
            warm_slab_failure_policy: _warm_slab_failure_policy,
            max_transient_buffer_bytes: _max_transient_buffer_bytes,
            max_inflight_transient_bytes: _max_inflight_transient_bytes,
            close_lease_timeout_seconds: _close_lease_timeout_seconds,
            slab_allocation_policy: _slab_allocation_policy,
        } = buffer_pool;
        let FirstWriteSizingConfig {
            buffer_sizing_policy: _buffer_sizing_policy,
            model_profile_hint: _model_profile_hint,
            min_hot_slab_buffers: _min_hot_slab_buffers,
            max_warm_to_hot_ratio: _max_warm_to_hot_ratio,
            global_fallback_chunk_bytes: _global_fallback_chunk_bytes,
        } = first_write_sizing;

        let expected_kwargs = [
            "file_name",
            "dataset_path",
            "parallel_coords",
            "default_parallel_coord_names",
            "zarr_format",
            "chunk_key_encoding",
            "chunk_key_separator",
            "buffer_sizing_policy",
            "model_profile_hint",
            "min_hot_slab_buffers",
            "max_warm_to_hot_ratio",
            "hot_slab_buffers",
            "warm_slab_buffers",
            "hot_slab_ready_timeout_seconds",
            "warm_slab_warmup_policy",
            "warm_slab_failure_policy",
            "max_pool_buffers",
            "max_pool_bytes",
            "pool_buffer_bytes",
            "pool_alignment",
            "pin_pooled_slabs",
            "cuda_register_pool_if_available",
            "cuda_register_each_slab_once",
            "max_transient_buffer_bytes",
            "max_inflight_transient_bytes",
            "close_lease_timeout_seconds",
            "queue_capacity",
            "num_threads",
            "require_host_array_interface",
            "fsync_policy",
        ];

        let expected = expected_kwargs
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let supported = SUPPORTED_KWARGS
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            expected, supported,
            "SUPPORTED_KWARGS must cover every Python-exposed config field and nothing extra",
        );
    }

    #[test]
    fn parse_coord_map_rejects_empty_coordinate_key() {
        Python::initialize();
        Python::attach(|py| {
            let coords = PyDict::new(py);
            coords
                .set_item("", vec![1_i64])
                .expect("set empty coord key for validation test");

            let err = parse_coord_map(coords.as_any())
                .expect_err("empty coordinate key should be rejected");
            assert_pyerr_type(py, err, "ValueError");
        });
    }

    #[test]
    fn parse_coord_map_rejects_empty_coordinate_values() {
        Python::initialize();
        Python::attach(|py| {
            let coords = PyDict::new(py);
            coords
                .set_item("time", Vec::<i64>::new())
                .expect("set empty coordinate values for validation test");

            let err = parse_coord_map(coords.as_any())
                .expect_err("empty coordinate values should be rejected");
            assert_pyerr_type(py, err, "ValueError");
        });
    }

    #[test]
    fn parse_host_source_documents_pointer_safety_contract() {
        let source = include_str!("python_bindings.rs");
        let parse_host_section = source
            .split("fn parse_host_source(")
            .nth(1)
            .expect("parse_host_source should exist");
        let parse_host_block = parse_host_section
            .split("fn parse_cuda_source(")
            .next()
            .expect("parse_host_source block should end before parse_cuda_source");
        assert!(
            parse_host_block.contains("SAFETY"),
            "parse_host_source must document SAFETY invariants for raw host-pointer ingestion",
        );
    }

    #[test]
    fn parse_one_input_array_rejects_cuda_nbytes_shape_mismatch() {
        Python::initialize();
        Python::attach(|py| {
            let types = py.import("types").expect("import types");
            let obj = types
                .getattr("SimpleNamespace")
                .expect("SimpleNamespace should exist")
                .call0()
                .expect("instantiate SimpleNamespace");
            obj.setattr("nbytes", 4_usize)
                .expect("set nbytes attribute on test object");

            let iface = PyDict::new(py);
            iface
                .set_item("data", (0x1234_u64, false))
                .expect("set cuda interface data pointer");
            iface
                .set_item("shape", vec![8_usize])
                .expect("set cuda interface shape");
            iface
                .set_item("typestr", "|u1")
                .expect("set cuda interface dtype");
            obj.setattr("__cuda_array_interface__", iface)
                .expect("attach __cuda_array_interface__");

            let err = parse_one_input_array(&obj, true)
                .expect_err("cuda nbytes/shape mismatch should be rejected");
            assert_pyerr_type(py, err, "ValueError");
        });
    }

    #[test]
    fn read_docstring_mentions_gil_behavior() {
        Python::initialize();
        Python::attach(|py| {
            let cls = py.get_type::<PyE2sZarrIoBackend>();
            let read_method = cls.getattr("read").expect("read method should exist");
            let doc = read_method
                .getattr("__doc__")
                .expect("read method should have __doc__");
            let doc_text = doc
                .extract::<String>()
                .expect("read method __doc__ should be string");
            assert!(
                doc_text.to_ascii_lowercase().contains("gil"),
                "read docstring should explicitly document GIL behavior",
            );
        });
    }

    #[test]
    fn context_manager_exit_documents_and_implements_close_error_propagation() {
        let source = include_str!("python_bindings.rs");
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .expect("python_bindings.rs should have a production section");
        let exit_section = production_source
            .split("fn __exit__(")
            .nth(1)
            .expect("__exit__ should exist");
        let exit_block = exit_section
            .split("/// Register one or more arrays and their full coordinate contract.")
            .next()
            .expect("__exit__ block should end before add_array()");
        assert!(
            exit_block.contains("exc_type.is_none()"),
            "__exit__ should branch on active exception state so close() errors can propagate when no exception is active",
        );
        assert!(
            exit_block.contains("map_err(sync_error_to_pyerr)?"),
            "__exit__ should propagate close() failures into Python exceptions when no prior exception exists",
        );
    }

    #[test]
    fn all_py_detach_call_sites_use_catch_unwind_panic_boundary() {
        let source = include_str!("python_bindings.rs");
        let impl_section = source
            .split("#[pymethods]")
            .nth(1)
            .expect("#[pymethods] block should exist");
        let impl_block = impl_section
            .split("#[cfg(test)]")
            .next()
            .expect("pymethods block should end before test module");

        assert!(
            !impl_block.contains("py.detach("),
            "all py.detach() calls in #[pymethods] must be replaced with \
             detach_with_panic_boundary() to prevent panics from crossing the FFI boundary"
        );
    }

    #[test]
    fn detach_with_panic_boundary_does_not_require_static_closure_bound() {
        let source = include_str!("python_bindings.rs");
        let helper_section = source
            .split("fn detach_with_panic_boundary<F, T>")
            .nth(1)
            .expect("detach_with_panic_boundary helper should exist");
        let helper_block = helper_section
            .split("#[pyclass(name = \"E2sZarrIoBackend\")]")
            .next()
            .expect("helper section should end before PyE2sZarrIoBackend definition");
        assert!(
            !helper_block.contains("+ 'static"),
            "detach_with_panic_boundary should not require a 'static closure; \
             Python methods pass closures borrowing &self while the GIL is released",
        );
    }

    #[test]
    fn sync_error_to_pyerr_maps_core_variants_to_expected_python_types() {
        Python::initialize();
        Python::attach(|py| {
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::Validation {
                    message: "bad input".to_string(),
                }),
                "ValueError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::ContractViolation {
                    message: "bad lifecycle".to_string(),
                }),
                "ValueError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::ObjectClosed),
                "RuntimeError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::io_failed("disk write failed")),
                "OSError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::metadata_consolidation_failed(
                    "metadata write failed",
                )),
                "OSError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::LeaseReturnTimeout {
                    outstanding_leases: 1,
                }),
                "TimeoutError",
            );
            assert_pyerr_type(
                py,
                sync_error_to_pyerr(SyncWriteError::copy_failed("cuda copy failed")),
                "RuntimeError",
            );
        });
    }

    #[test]
    fn python_api_methods_expose_docstrings_and_text_signatures() {
        Python::initialize();
        Python::attach(|py| {
            let cls = py.get_type::<PyE2sZarrIoBackend>();
            let class_doc = cls
                .getattr("__doc__")
                .expect("class must expose __doc__")
                .extract::<String>()
                .expect("class __doc__ must be string");
            assert!(
                !class_doc.trim().is_empty(),
                "class docstring should not be empty"
            );

            for method_name in [
                "__new__",
                "add_array",
                "write",
                "read",
                "close",
                "is_closed",
                "last_write_timing",
            ] {
                let method = cls
                    .getattr(method_name)
                    .unwrap_or_else(|_| panic!("missing method on class: {method_name}"));
                let doc = method
                    .getattr("__doc__")
                    .unwrap_or_else(|_| panic!("missing __doc__ on method: {method_name}"));
                assert!(
                    !doc.is_none(),
                    "method docstring should exist for {method_name}"
                );
                let doc_text = doc
                    .extract::<String>()
                    .unwrap_or_else(|_| panic!("method __doc__ must be string: {method_name}"));
                assert!(
                    !doc_text.trim().is_empty(),
                    "method docstring should not be empty for {method_name}"
                );

                let signature = method.getattr("__text_signature__").unwrap_or_else(|_| {
                    panic!("missing __text_signature__ on method: {method_name}")
                });
                assert!(
                    !signature.is_none(),
                    "__text_signature__ should be present for {method_name}"
                );
                let signature_text = signature.extract::<String>().unwrap_or_else(|_| {
                    panic!("__text_signature__ must be string for method: {method_name}")
                });
                assert!(
                    !signature_text.trim().is_empty(),
                    "__text_signature__ should not be empty for {method_name}"
                );
            }
        });
    }

    #[test]
    fn add_array_accepts_optional_data_and_writes_initial_chunk() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_add_array_data");
            let _ = fs::remove_dir_all(&dataset_root);

            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            kwargs
                .set_item("require_host_array_interface", false)
                .expect("set require_host_array_interface");

            let backend = PyE2sZarrIoBackend::new(Some(&kwargs)).expect("backend constructor");

            let coords = PyDict::new(py);
            coords
                .set_item("lat", vec![0.0_f64])
                .expect("set coord values");
            let array_name = PyString::new(py, "t2m");

            let expected_bytes = vec![0x00, 0x00, 0x80, 0x3f];
            let payload = PyBytes::new(py, &expected_bytes);
            let memoryview = py
                .import("builtins")
                .and_then(|b| b.getattr("memoryview"))
                .and_then(|ctor| ctor.call1((payload,)))
                .expect("memoryview constructor");

            backend
                .add_array(
                    py,
                    coords.as_any(),
                    array_name.as_any(),
                    Some(memoryview.as_any()),
                )
                .expect("add_array(data=...) should succeed");
            backend.close(py, Some(5.0)).expect("close should succeed");

            let chunk_path = dataset_root.join("t2m").join("0");
            let chunk = fs::read(&chunk_path).expect("read initialized chunk bytes");
            assert_eq!(chunk, expected_bytes);

            let _ = fs::remove_dir_all(&dataset_root);
        });
    }

    #[test]
    fn write_casts_mismatched_input_to_registered_dtype() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_write_dtype_cast");
            let _ = fs::remove_dir_all(&dataset_root);

            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            let backend = PyE2sZarrIoBackend::new(Some(&kwargs)).expect("backend constructor");

            let coords = PyDict::new(py);
            coords
                .set_item("lat", vec![0.0_f64, 1.0_f64])
                .expect("set coord values");
            let array_name = PyString::new(py, "t2m");
            backend
                .add_array(py, coords.as_any(), array_name.as_any(), None)
                .expect("add_array without data should register float32");

            let Ok(np) = py.import("numpy") else {
                return;
            };
            let data_kwargs = PyDict::new(py);
            data_kwargs.set_item("dtype", "int16").expect("set dtype");
            let data = np
                .call_method("array", (vec![1_i16, 2_i16],), Some(&data_kwargs))
                .expect("construct int16 array");
            backend
                .write(py, data.as_any(), coords.as_any(), array_name.as_any())
                .expect("write should cast int16 payload to registered float32");
            backend.close(py, Some(5.0)).expect("close should succeed");

            let chunk_path = dataset_root.join("t2m").join("0");
            let chunk = fs::read(&chunk_path).expect("read casted chunk bytes");
            let expected: Vec<u8> = [1.0_f32, 2.0_f32]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(chunk, expected);

            let _ = fs::remove_dir_all(&dataset_root);
        });
    }

    #[test]
    fn read_validates_array_name_against_registration() {
        let source = include_str!("python_bindings.rs");
        let read_section = source
            .split("fn read(")
            .nth(1)
            .expect("PyE2sZarrIoBackend::read should exist");
        let read_block = read_section
            .split("\n    fn ")
            .next()
            .expect("read block should end before next method");
        assert!(
            read_block.contains("validate_write_array_names")
                || read_block.contains("registered_array_names")
                || read_block.contains("validate_read_array_name"),
            "read() must validate array_name against registered arrays before \
             using it for zarr store access"
        );
    }

    #[test]
    fn repr_includes_dataset_root_and_zarr_format() {
        Python::initialize();
        Python::attach(|py| {
            let dataset_root = unique_dataset_root("e2s_py_binding_repr");
            let _ = fs::remove_dir_all(&dataset_root);

            let kwargs = PyDict::new(py);
            kwargs
                .set_item("file_name", dataset_root.to_string_lossy().as_ref())
                .expect("set file_name");
            kwargs
                .set_item("zarr_format", "v3")
                .expect("set zarr_format");

            let backend = PyE2sZarrIoBackend::new(Some(&kwargs)).expect("backend constructor");
            let repr = backend.__repr__();

            assert!(
                repr.contains(&format!("dataset_root='{}'", dataset_root.display())),
                "repr must include dataset_root: {repr}"
            );
            assert!(
                repr.contains("zarr_format='v3'"),
                "repr must include zarr_format: {repr}"
            );
            assert!(
                repr.contains("closed=False"),
                "repr must include closed flag: {repr}"
            );

            backend.close(py, Some(5.0)).expect("close should succeed");
            let closed_repr = backend.__repr__();
            assert!(
                closed_repr.contains("closed=True"),
                "repr must reflect closed state: {closed_repr}"
            );
        });
    }
}
