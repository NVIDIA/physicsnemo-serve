/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Coordinate and request/input payload types.

use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::core::errors::SyncWriteError;

use super::{CoordValues, DataType};

/// Ordered mapping from coordinate name to coordinate values.
/// Canonical coordinate contract crossing API boundaries.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoordMap(IndexMap<String, CoordValues>);

impl CoordMap {
    fn validate_entry(key: &str, values: &CoordValues) -> Result<(), SyncWriteError> {
        if key.trim().is_empty() {
            return Err(SyncWriteError::Validation {
                message: "coordinate key must not be empty".to_string(),
            });
        }
        if values.is_empty() {
            return Err(SyncWriteError::Validation {
                message: format!("coordinate '{key}' must include at least one value"),
            });
        }
        if let CoordValues::F64(vals) = values {
            if vals.iter().any(|v| v.is_nan()) {
                return Err(SyncWriteError::Validation {
                    message: format!("coordinate '{key}' contains NaN values"),
                });
            }
        }
        if let CoordValues::F32(vals) = values {
            if vals.iter().any(|v| v.is_nan()) {
                return Err(SyncWriteError::Validation {
                    message: format!("coordinate '{key}' contains NaN values"),
                });
            }
        }
        Ok(())
    }

    fn validate_entries(map: &IndexMap<String, CoordValues>) -> Result<(), SyncWriteError> {
        for (key, values) in map {
            Self::validate_entry(key, values)?;
        }
        Ok(())
    }

    /// Construct an empty coordinate map for incremental building via `insert()`,
    /// which performs per-entry validation. Use `try_new()` for holistic validation.
    #[must_use]
    pub fn new() -> Self {
        Self(IndexMap::new())
    }

    /// Construct a coordinate map from a raw map with holistic validation.
    ///
    /// # Errors
    ///
    /// Returns [`SyncWriteError::Validation`] for empty keys, empty axes, or NaN values.
    pub fn try_new(map: BTreeMap<String, CoordValues>) -> Result<Self, SyncWriteError> {
        Self::try_from_iter(map)
    }

    /// Validate all entries against coordinate contract invariants.
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        Self::validate_entries(&self.0)
    }

    /// Consume and return the underlying map.
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<String, CoordValues> {
        self.0.into_iter().collect()
    }

    /// Insert one validated coordinate axis entry (rejects empty/blank keys, empty axes, NaN).
    pub fn insert(
        &mut self,
        key: String,
        values: CoordValues,
    ) -> Result<Option<CoordValues>, SyncWriteError> {
        Self::validate_entry(key.as_str(), &values)?;
        Ok(self.0.insert(key, values))
    }

    /// Construct from an iterator with full validation.
    pub fn try_from_iter<T>(iter: T) -> Result<Self, SyncWriteError>
    where
        T: IntoIterator<Item = (String, CoordValues)>,
    {
        let map: IndexMap<String, CoordValues> = iter.into_iter().collect();
        Self::validate_entries(&map)?;
        Ok(Self(map))
    }
}

impl Deref for CoordMap {
    type Target = IndexMap<String, CoordValues>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TryFrom<BTreeMap<String, CoordValues>> for CoordMap {
    type Error = SyncWriteError;

    fn try_from(map: BTreeMap<String, CoordValues>) -> Result<Self, Self::Error> {
        Self::try_new(map)
    }
}

impl From<CoordMap> for BTreeMap<String, CoordValues> {
    fn from(value: CoordMap) -> Self {
        value.0.into_iter().collect()
    }
}

impl IntoIterator for CoordMap {
    type Item = (String, CoordValues);
    type IntoIter = indexmap::map::IntoIter<String, CoordValues>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a CoordMap {
    type Item = (&'a String, &'a CoordValues);
    type IntoIter = indexmap::map::Iter<'a, String, CoordValues>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Input for `add_array()`: registers array names and the full coordinate contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayRegistration {
    /// Full coordinate system for all registered arrays.
    pub coords: CoordMap,
    /// Array names to register (must be non-empty and unique).
    pub array_names: Vec<String>,
    /// Dtypes for `array_names` in the same order.
    pub array_dtypes: Vec<DataType>,
}

impl ArrayRegistration {
    /// Validate registration request invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SyncWriteError::Validation`] when registration inputs violate
    /// the add-array boundary contract.
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        self.coords.validate()?;
        if self.array_names.is_empty() {
            return Err(SyncWriteError::Validation {
                message: "add_array() requires at least one array name".to_string(),
            });
        }
        if self.array_names.iter().any(|name| name.trim().is_empty()) {
            return Err(SyncWriteError::Validation {
                message: "add_array() requires non-empty array names".to_string(),
            });
        }
        let unique_names: std::collections::BTreeSet<&str> =
            self.array_names.iter().map(String::as_str).collect();
        if unique_names.len() != self.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: "add_array() array_names must be unique".to_string(),
            });
        }
        if !self.array_dtypes.is_empty() && self.array_dtypes.len() != self.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "len(array_dtypes) != len(array_names): {} != {}",
                    self.array_dtypes.len(),
                    self.array_names.len()
                ),
            });
        }
        Ok(())
    }
}

/// Input for `write()`: arrays for the current inference step.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceWriteRequest {
    /// Step coordinates (may be a subset of registered coords on parallel dims).
    pub coords: CoordMap,
    /// Array names for this write (must be a subset of registered names).
    pub array_names: Vec<String>,
    /// Input array data (one per `array_name`, same order).
    pub arrays: Vec<InputArray>,
}

impl InferenceWriteRequest {
    /// Validate write request invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SyncWriteError::Validation`] when the request violates
    /// write-path contract expectations.
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        self.coords.validate()?;
        if self.arrays.is_empty() {
            return Err(SyncWriteError::Validation {
                message: "write() requires at least one input array".to_string(),
            });
        }
        if self.arrays.len() != self.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "len(arrays) != len(array_names): {} != {}",
                    self.arrays.len(),
                    self.array_names.len()
                ),
            });
        }
        if self.array_names.iter().any(|name| name.trim().is_empty()) {
            return Err(SyncWriteError::Validation {
                message: "write() requires non-empty array names".to_string(),
            });
        }
        let unique_names: std::collections::BTreeSet<&str> =
            self.array_names.iter().map(String::as_str).collect();
        if unique_names.len() != self.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: "write() array_names must be unique".to_string(),
            });
        }
        for (index, input) in self.arrays.iter().enumerate() {
            input.validate().map_err(|err| match err {
                SyncWriteError::Validation { message } => SyncWriteError::Validation {
                    message: format!("write() arrays[{index}] invalid: {message}"),
                },
                other => other,
            })?;
        }
        Ok(())
    }
}

/// Source memory location for an input array. Host pointer ingestion is
/// internal-only. External callers construct only `HostBytes` or `CudaDevicePtr`.
///
/// ```compile_fail
/// use e2s_zarr_io::InputArraySource;
///
/// let _ = InputArraySource::from_host_buffer_ptr(4096);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputArraySource {
    /// Host-side bytes (owned copy from Python boundary).
    HostBytes(Arc<[u8]>),
    /// CUDA device pointer from Python boundary.
    CudaDevicePtr {
        /// Raw device pointer value.
        ptr: u64,
        /// CUDA device ordinal.
        device_ordinal: i32,
        /// Optional producer stream handle for synchronization.
        producer_stream: Option<u64>,
    },
    /// Crate-internal host-memory pointer from Python array interface.
    #[doc(hidden)]
    #[non_exhaustive]
    __InternalHostBufferPtr {
        /// Raw host pointer value.
        ptr: u64,
    },
}

impl InputArraySource {
    /// Construct internal host pointer descriptor from Python array interface.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` references readable host memory for
    /// at least the `required_bytes` consumed by copy workers, and that the
    /// pointed allocation outlives the copy-barrier window.
    #[must_use]
    pub(crate) unsafe fn from_host_buffer_ptr(ptr: u64) -> Self {
        Self::__InternalHostBufferPtr { ptr }
    }

    #[must_use]
    pub(crate) fn as_host_buffer_ptr(&self) -> Option<u64> {
        match self {
            Self::__InternalHostBufferPtr { ptr } => Some(*ptr),
            _ => None,
        }
    }
}

/// An input array to be written as one or more Zarr chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputArray {
    /// Total byte size of the array payload.
    pub nbytes: usize,
    /// Source memory location.
    pub source: InputArraySource,
}

impl InputArray {
    /// Construct a host-byte input array with validated byte count.
    #[must_use]
    pub fn from_host_bytes(bytes: Vec<u8>) -> Self {
        Self {
            nbytes: bytes.len(),
            source: InputArraySource::HostBytes(bytes.into()),
        }
    }

    /// Construct a CUDA pointer input array.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid CUDA device pointer readable for `nbytes` bytes
    /// on `device_ordinal` for the duration of the write copy-barrier.
    #[must_use]
    pub unsafe fn from_cuda_ptr(
        ptr: u64,
        nbytes: usize,
        device_ordinal: i32,
        producer_stream: Option<u64>,
    ) -> Self {
        Self {
            nbytes,
            source: InputArraySource::CudaDevicePtr {
                ptr,
                device_ordinal,
                producer_stream,
            },
        }
    }

    /// Validate cross-field invariants for this input descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`SyncWriteError::Validation`] when `nbytes` is zero,
    /// `HostBytes` payload length disagrees with declared `nbytes`, or
    /// `CudaDevicePtr` has a null pointer or negative `device_ordinal`.
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        if self.nbytes == 0 {
            return Err(SyncWriteError::Validation {
                message: "input array nbytes must be non-zero".to_string(),
            });
        }
        match &self.source {
            InputArraySource::HostBytes(payload) if self.nbytes != payload.len() => {
                Err(SyncWriteError::Validation {
                    message: format!(
                        "HostBytes nbytes mismatch: declared {} but payload len {}",
                        self.nbytes,
                        payload.len()
                    ),
                })
            }
            InputArraySource::CudaDevicePtr { ptr, .. } if *ptr == 0 => {
                Err(SyncWriteError::Validation {
                    message: "CUDA device pointer must be non-zero".to_string(),
                })
            }
            InputArraySource::CudaDevicePtr { device_ordinal, .. } if *device_ordinal < 0 => {
                Err(SyncWriteError::Validation {
                    message: format!(
                        "CUDA device_ordinal must be non-negative, got {device_ordinal}"
                    ),
                })
            }
            _ => Ok(()),
        }
    }
}
