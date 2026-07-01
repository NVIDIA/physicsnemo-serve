/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Metadata consolidation for Zarr v2 and v3 datasets.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::core::contracts::{ConsolidationScope, MetadataConsolidator, ZarrLayoutAdapter};
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ArrayRegistration, CoordMap, CoordValues, DataType, FsyncPolicy, TupleChunkKey, ZarrFormat,
};

/// Ordered dimension names, shape per dim, and chunk sizes per dim.
type DimShapeChunks = (Vec<String>, Vec<usize>, Vec<usize>);

static METADATA_TEMP_NAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Compile-safe metadata consolidator scaffold.
///
/// This implementation is intentionally minimal and only validates that the
/// adapter boundary is reachable from close-time orchestration.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct NoopMetadataConsolidator;

#[cfg(any(test, feature = "test-utils"))]
impl NoopMetadataConsolidator {
    /// Create a new no-op metadata consolidator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl MetadataConsolidator for NoopMetadataConsolidator {
    fn consolidate(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        _registration: Option<&ArrayRegistration>,
        _parallel_coord_names: &[String],
    ) -> Result<(), SyncWriteError> {
        let _ = layout.zarr_format();
        Ok(())
    }
}

/// Local-filesystem metadata consolidator used by the Earth2Studio-facing backend path.
pub struct LocalFsMetadataConsolidator {
    dataset_root: PathBuf,
    fsync_policy: FsyncPolicy,
}

impl LocalFsMetadataConsolidator {
    /// Create a consolidator rooted at `dataset_root`.
    #[must_use]
    pub fn new(dataset_root: PathBuf) -> Self {
        Self {
            dataset_root,
            fsync_policy: FsyncPolicy::Always,
        }
    }

    /// Create a consolidator with an explicit fsync policy.
    #[must_use]
    pub fn with_fsync_policy(dataset_root: PathBuf, fsync_policy: FsyncPolicy) -> Self {
        Self {
            dataset_root,
            fsync_policy,
        }
    }

    fn temp_path_for(path: &Path, file_context: &str) -> Result<PathBuf, SyncWriteError> {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed(format!(
                    "{file_context} path has no terminal file name: {}",
                    path.display()
                ))
            })?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!(
                        "failed generating temp-file nonce for '{}': {e}",
                        path.display()
                    ),
                    e,
                )
            })?
            .as_nanos();
        let uniq = METADATA_TEMP_NAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(path.with_file_name(format!("{file_name}.tmp-{nanos}-{uniq}")))
    }

    fn remove_temp_file_best_effort(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    fn sync_parent_dir_best_effort(path: &Path) {
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }

    fn atomic_write_to_path(
        path: &Path,
        payload: &[u8],
        dir_context: &str,
        file_context: &str,
        fsync_policy: FsyncPolicy,
    ) -> Result<(), SyncWriteError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!("failed creating {dir_context} '{}': {e}", parent.display()),
                    e,
                )
            })?;
        }
        let temp_path = Self::temp_path_for(path, file_context)?;
        let mut file = std::fs::File::create(&temp_path).map_err(|e| {
            SyncWriteError::metadata_consolidation_failed_with_cause(
                format!(
                    "failed creating {file_context} temp file '{}': {e}",
                    temp_path.display()
                ),
                e,
            )
        })?;
        if let Err(e) = file.write_all(payload) {
            drop(file);
            Self::remove_temp_file_best_effort(&temp_path);
            return Err(SyncWriteError::metadata_consolidation_failed_with_cause(
                format!(
                    "failed writing {file_context} temp file '{}': {e}",
                    temp_path.display()
                ),
                e,
            ));
        }
        if fsync_policy == FsyncPolicy::Always {
            if let Err(e) = file.sync_all() {
                drop(file);
                Self::remove_temp_file_best_effort(&temp_path);
                return Err(SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!(
                        "failed syncing {file_context} temp file '{}': {e}",
                        temp_path.display()
                    ),
                    e,
                ));
            }
        }
        drop(file);
        if let Err(e) = std::fs::rename(&temp_path, path) {
            Self::remove_temp_file_best_effort(&temp_path);
            return Err(SyncWriteError::metadata_consolidation_failed_with_cause(
                format!(
                    "failed renaming {file_context} temp file '{}' -> '{}': {e}",
                    temp_path.display(),
                    path.display()
                ),
                e,
            ));
        }
        if fsync_policy == FsyncPolicy::Always {
            Self::sync_parent_dir_best_effort(path);
        }
        Ok(())
    }

    fn write_metadata_file(
        path: &Path,
        payload: &str,
        fsync_policy: FsyncPolicy,
    ) -> Result<(), SyncWriteError> {
        Self::atomic_write_to_path(
            path,
            payload.as_bytes(),
            "metadata directory",
            "metadata",
            fsync_policy,
        )
    }

    fn write_binary_file(
        path: &Path,
        payload: &[u8],
        fsync_policy: FsyncPolicy,
    ) -> Result<(), SyncWriteError> {
        Self::atomic_write_to_path(
            path,
            payload,
            "coordinate chunk directory",
            "coordinate chunk",
            fsync_policy,
        )
    }

    fn default_v2_dimension_separator(
        layout: &dyn ZarrLayoutAdapter,
    ) -> Result<char, SyncWriteError> {
        if layout.zarr_format() != ZarrFormat::V2 {
            return Ok('.');
        }
        let probe_name = "__e2s_separator_probe__";
        let probe_path = layout
            .chunk_path_for_tuple_key(probe_name, &TupleChunkKey::new(vec![0, 1]))?
            .relative_path;
        let suffix = probe_path
            .strip_prefix(&format!("{probe_name}/"))
            .unwrap_or(probe_path.as_str());
        if suffix.contains('/') {
            Ok('/')
        } else {
            Ok('.')
        }
    }

    fn list_array_dirs(&self) -> Result<Vec<PathBuf>, SyncWriteError> {
        let mut out = Vec::new();
        let root_iter = std::fs::read_dir(&self.dataset_root).map_err(|e| {
            SyncWriteError::metadata_consolidation_failed_with_cause(
                format!(
                    "failed listing dataset root '{}': {e}",
                    self.dataset_root.display()
                ),
                e,
            )
        })?;
        for entry in root_iter {
            let entry = entry.map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!("failed reading dataset root entry: {e}"),
                    e,
                )
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            out.push(path);
        }
        Ok(out)
    }

    /// Parse a chunk-key relative path and return `(optional_linear_index, separator)`.
    ///
    /// Accepts legacy linear keys (`7`, `c.7`, `c/7`) and tuple keys
    /// (`0.4.0.0`, `0/4/0/0`, `c/0/4/0/0`). For tuple keys the linear index
    /// is unknown without a grid descriptor and returns `None`.
    fn parse_chunk_key(rel: &Path) -> Option<(Option<u64>, char)> {
        let components: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        if components.is_empty() {
            return None;
        }

        // Single-component chunk key (v2 dot / single-dim / legacy c.N)
        if components.len() == 1 {
            let name = &components[0];
            if name.starts_with('.') || name == "zarr.json" {
                return None;
            }

            if let Some(idx) = name.strip_prefix("c.") {
                if !idx.is_empty()
                    && idx
                        .split('.')
                        .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
                {
                    return Some((idx.parse::<u64>().ok(), '.'));
                }
                return None;
            }

            if name.contains('.') {
                if name
                    .split('.')
                    .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
                {
                    // Tuple-key dot paths are not directly linearizable without axis lengths.
                    return Some((None, '.'));
                }
                return None;
            }

            if name.chars().all(|ch| ch.is_ascii_digit()) {
                return Some((name.parse::<u64>().ok(), '.'));
            }
            return None;
        }

        // Multi-component chunk key (v2 slash / v3 c/...).
        let start = usize::from(components.first().is_some_and(|c| c == "c"));
        let tail = &components[start..];
        if tail.is_empty() {
            return None;
        }
        if tail.iter().all(|part| {
            !part.is_empty() && !part.starts_with('.') && part.chars().all(|ch| ch.is_ascii_digit())
        }) {
            let linear = if tail.len() == 1 {
                tail[0].parse::<u64>().ok()
            } else {
                None
            };
            return Some((linear, '/'));
        }
        None
    }

    fn walk_files(root: &Path) -> Result<Vec<PathBuf>, SyncWriteError> {
        let mut files = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!("failed reading directory '{}': {e}", dir.display()),
                    e,
                )
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    SyncWriteError::metadata_consolidation_failed_with_cause(
                        format!("failed reading directory entry '{}': {e}", dir.display()),
                        e,
                    )
                })?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    files.push(path);
                }
            }
        }
        Ok(files)
    }

    fn infer_chunk_descriptor(array_dir: &Path) -> Result<Option<ChunkDescriptor>, SyncWriteError> {
        let mut max_linear_index: Option<u64> = None;
        let mut chunk_file_count: u64 = 0;
        let mut chunk_bytes: Option<usize> = None;
        let mut separator: Option<char> = None;
        let files = Self::walk_files(array_dir)?;
        for file_path in files {
            let rel = file_path.strip_prefix(array_dir).map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!(
                        "failed computing chunk path relative prefix '{}': {e}",
                        array_dir.display()
                    ),
                    e,
                )
            })?;
            let Some((linear_index, inferred_separator)) = Self::parse_chunk_key(rel) else {
                continue;
            };
            if let Some(existing) = separator {
                if existing != inferred_separator {
                    return Err(SyncWriteError::metadata_consolidation_failed(format!(
                        "inconsistent chunk key separators in '{}': '{}' vs '{}'",
                        array_dir.display(),
                        existing,
                        inferred_separator
                    )));
                }
            }
            separator = Some(inferred_separator);
            let size_u64 = std::fs::metadata(&file_path)
                .map_err(|e| {
                    SyncWriteError::metadata_consolidation_failed_with_cause(
                        format!(
                            "failed reading chunk metadata '{}': {e}",
                            file_path.display()
                        ),
                        e,
                    )
                })?
                .len();
            let size = usize::try_from(size_u64).map_err(|cause| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!("chunk file too large for usize: '{}'", file_path.display()),
                    cause,
                )
            })?;
            match chunk_bytes {
                Some(prev) if prev != size => {
                    return Err(SyncWriteError::metadata_consolidation_failed(format!(
                        "inconsistent chunk byte sizes in '{}': {prev} vs {size}",
                        array_dir.display()
                    )));
                }
                None => chunk_bytes = Some(size),
                Some(_) => {}
            }
            if let Some(index) = linear_index {
                max_linear_index =
                    Some(max_linear_index.map_or(index, |current| current.max(index)));
            }
            chunk_file_count = chunk_file_count.checked_add(1).ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed(format!(
                    "chunk file count overflowed while scanning '{}'",
                    array_dir.display()
                ))
            })?;
        }

        if chunk_file_count == 0 {
            return Ok(None);
        }
        let Some(chunk_bytes) = chunk_bytes else {
            return Ok(None);
        };
        let inferred_max_index =
            max_linear_index.unwrap_or_else(|| chunk_file_count.saturating_sub(1));
        Ok(Some(ChunkDescriptor {
            max_index: inferred_max_index,
            chunk_bytes,
            separator: separator.unwrap_or('.'),
        }))
    }

    fn dtype_for_numeric_coord_values(values: &CoordValues) -> Option<DTypeDescriptor> {
        match values {
            CoordValues::I64(_) => Some(DTypeDescriptor {
                v2_dtype: "<i8",
                v3_data_type: "int64",
                elem_bytes: 8,
            }),
            CoordValues::DatetimeNs(_) => Some(DTypeDescriptor {
                v2_dtype: "<M8[ns]",
                v3_data_type: "{\"name\":\"numpy.datetime64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}",
                elem_bytes: 8,
            }),
            CoordValues::TimedeltaNs(_) => Some(DTypeDescriptor {
                v2_dtype: "<m8[ns]",
                v3_data_type: "{\"name\":\"numpy.timedelta64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}",
                elem_bytes: 8,
            }),
            CoordValues::U64(_) => Some(DTypeDescriptor {
                v2_dtype: "<u8",
                v3_data_type: "uint64",
                elem_bytes: 8,
            }),
            CoordValues::I32(_) => Some(DTypeDescriptor::from(DataType::Int32)),
            CoordValues::U32(_) => Some(DTypeDescriptor::from(DataType::UInt32)),
            CoordValues::F32(_) => Some(DTypeDescriptor::from(DataType::Float32)),
            CoordValues::F64(_) => Some(DTypeDescriptor {
                v2_dtype: "<f8",
                v3_data_type: "float64",
                elem_bytes: 8,
            }),
            CoordValues::Utf8(_) => None,
        }
    }

    fn numeric_coord_values_to_le_bytes(values: &CoordValues) -> Option<Vec<u8>> {
        match values {
            CoordValues::I64(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::DatetimeNs(vals) | CoordValues::TimedeltaNs(vals) => {
                Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            CoordValues::U64(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::I32(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::U32(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::F32(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::F64(vals) => Some(vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
            CoordValues::Utf8(_) => None,
        }
    }

    fn utf8_coord_values_to_fixed_width_utf32(values: &[String]) -> (usize, Vec<u8>) {
        let width = values
            .iter()
            .map(|value| value.chars().count())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut payload = Vec::with_capacity(values.len() * width * 4);
        for value in values {
            let mut chars = value.chars();
            for _ in 0..width {
                let cp = chars.next().map_or(0_u32, |ch| ch as u32);
                payload.extend_from_slice(&cp.to_le_bytes());
            }
        }
        (width, payload)
    }

    fn build_v2_registered_array_metadata_with_dtype(
        shape: &[usize],
        chunks: &[usize],
        v2_dtype: &str,
        dimension_separator: char,
    ) -> String {
        format!(
            "{{\"zarr_format\":2,\"shape\":[{}],\"chunks\":[{}],\"dtype\":\"{}\",\"compressor\":null,\"fill_value\":null,\"order\":\"C\",\"filters\":null,\"dimension_separator\":\"{}\"}}",
            Self::format_usize_list(shape),
            Self::format_usize_list(chunks),
            v2_dtype,
            dimension_separator
        )
    }

    #[cfg(test)]
    fn infer_dtype(chunk_bytes: usize) -> DTypeDescriptor {
        if chunk_bytes >= 4 && chunk_bytes % 4 == 0 {
            DTypeDescriptor {
                v2_dtype: "<f4",
                v3_data_type: "float32",
                elem_bytes: 4,
            }
        } else {
            DTypeDescriptor {
                v2_dtype: "|u1",
                v3_data_type: "uint8",
                elem_bytes: 1,
            }
        }
    }

    fn conservative_unknown_dtype() -> DTypeDescriptor {
        DTypeDescriptor {
            v2_dtype: "|u1",
            v3_data_type: "uint8",
            elem_bytes: 1,
        }
    }

    fn dtype_from_chunk_bytes_and_elements(
        chunk_bytes: usize,
        chunk_elements: usize,
    ) -> DTypeDescriptor {
        if chunk_elements == 0 {
            return Self::conservative_unknown_dtype();
        }
        let elem_bytes = chunk_bytes / chunk_elements;
        if elem_bytes * chunk_elements != chunk_bytes {
            return Self::conservative_unknown_dtype();
        }
        match elem_bytes {
            1 => DTypeDescriptor {
                v2_dtype: "|u1",
                v3_data_type: "uint8",
                elem_bytes: 1,
            },
            2 => DTypeDescriptor {
                v2_dtype: "<f2",
                v3_data_type: "float16",
                elem_bytes: 2,
            },
            4 => DTypeDescriptor {
                v2_dtype: "<f4",
                v3_data_type: "float32",
                elem_bytes: 4,
            },
            8 => DTypeDescriptor {
                v2_dtype: "<f8",
                v3_data_type: "float64",
                elem_bytes: 8,
            },
            _ => Self::conservative_unknown_dtype(),
        }
    }

    fn json_escape(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for ch in value.chars() {
            match ch {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\u{08}' => escaped.push_str("\\b"),
                '\u{0C}' => escaped.push_str("\\f"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
                c => escaped.push(c),
            }
        }
        escaped
    }

    fn format_usize_list(values: &[usize]) -> String {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn format_string_list(values: &[String]) -> String {
        values
            .iter()
            .map(|value| {
                let escaped = Self::json_escape(value);
                format!("\"{escaped}\"")
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn ordered_dim_names(
        registered_coords: &CoordMap,
        parallel_coord_names: &[String],
    ) -> Vec<String> {
        let mut dim_order = Vec::new();
        for coord_name in parallel_coord_names {
            if registered_coords.contains_key(coord_name.as_str())
                && !dim_order.iter().any(|dim| dim == coord_name)
            {
                dim_order.push(coord_name.clone());
            }
        }
        for coord_name in registered_coords.keys() {
            if !dim_order.iter().any(|dim| dim == coord_name) {
                dim_order.push(coord_name.clone());
            }
        }
        dim_order
    }

    fn shape_and_chunks_from_registration(
        registration: &ArrayRegistration,
        parallel_coord_names: &[String],
    ) -> Result<DimShapeChunks, SyncWriteError> {
        let dim_order = Self::ordered_dim_names(&registration.coords, parallel_coord_names);
        if dim_order.is_empty() {
            return Err(SyncWriteError::metadata_consolidation_failed(
                "registration coords are empty; cannot derive multidimensional metadata",
            ));
        }

        let mut shape = Vec::with_capacity(dim_order.len());
        let mut chunks = Vec::with_capacity(dim_order.len());
        for dim_name in &dim_order {
            let axis = registration.coords.get(dim_name.as_str()).ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed(format!(
                    "missing registered coord axis for dim '{dim_name}'"
                ))
            })?;
            let axis_len = axis.len();
            if axis_len == 0 {
                return Err(SyncWriteError::metadata_consolidation_failed(format!(
                    "registered coord axis '{dim_name}' has zero length"
                )));
            }
            shape.push(axis_len);
            if parallel_coord_names.iter().any(|coord| coord == dim_name) {
                chunks.push(1);
            } else {
                chunks.push(axis_len);
            }
        }

        Ok((dim_order, shape, chunks))
    }

    fn build_v2_registered_array_metadata(
        shape: &[usize],
        chunks: &[usize],
        dtype: &DTypeDescriptor,
        dimension_separator: char,
    ) -> String {
        format!(
            "{{\"zarr_format\":2,\"shape\":[{}],\"chunks\":[{}],\"dtype\":\"{}\",\"compressor\":null,\"fill_value\":null,\"order\":\"C\",\"filters\":null,\"dimension_separator\":\"{}\"}}",
            Self::format_usize_list(shape),
            Self::format_usize_list(chunks),
            dtype.v2_dtype,
            dimension_separator
        )
    }

    fn build_v2_registered_array_attrs(dim_order: &[String]) -> String {
        format!(
            "{{\"_ARRAY_DIMENSIONS\":[{}]}}",
            Self::format_string_list(dim_order)
        )
    }

    fn build_v3_registered_array_metadata(
        shape: &[usize],
        chunks: &[usize],
        dtype: &DTypeDescriptor,
        dim_order: &[String],
    ) -> String {
        Self::build_v3_registered_array_metadata_with_data_type(
            shape,
            chunks,
            &dtype.v3_data_type_json(),
            dtype.v3_fill_value_json(),
            dim_order,
        )
    }

    fn build_v3_registered_utf8_array_metadata(
        shape: &[usize],
        chunks: &[usize],
        utf32_length_bytes: usize,
        dim_order: &[String],
    ) -> String {
        let data_type_json = format!(
            "{{\"name\":\"fixed_length_utf32\",\"configuration\":{{\"length_bytes\":{utf32_length_bytes}}}}}"
        );
        Self::build_v3_registered_array_metadata_with_data_type(
            shape,
            chunks,
            &data_type_json,
            "\"\"",
            dim_order,
        )
    }

    fn build_v3_registered_array_metadata_with_data_type(
        shape: &[usize],
        chunks: &[usize],
        data_type_json: &str,
        fill_value_json: &str,
        dim_order: &[String],
    ) -> String {
        format!(
            "{{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[{}],\"data_type\":{},\"chunk_grid\":{{\"name\":\"regular\",\"configuration\":{{\"chunk_shape\":[{}]}}}},\"chunk_key_encoding\":{{\"name\":\"default\",\"configuration\":{{\"separator\":\"/\"}}}},\"fill_value\":{},\"codecs\":[{{\"name\":\"bytes\",\"configuration\":{{\"endian\":\"little\"}}}}],\"attributes\":{{\"_ARRAY_DIMENSIONS\":[{}]}},\"dimension_names\":[{}]}}",
            Self::format_usize_list(shape),
            data_type_json,
            Self::format_usize_list(chunks),
            fill_value_json,
            Self::format_string_list(dim_order),
            Self::format_string_list(dim_order)
        )
    }

    fn build_v2_array_metadata(desc: &ChunkDescriptor) -> Result<String, SyncWriteError> {
        // Non-registration consolidation has no trusted dtype contract.
        // Use a conservative raw-byte descriptor to avoid silently claiming float32.
        let dtype = Self::conservative_unknown_dtype();
        let chunk_elems = desc
            .chunk_bytes
            .checked_div(dtype.elem_bytes)
            .ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed("failed computing v2 chunk elements")
            })?;
        let chunks_count = usize::try_from(desc.max_index)
            .ok()
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed(
                    "chunk index overflow while building v2 metadata",
                )
            })?;
        let shape_elems = chunks_count.checked_mul(chunk_elems).ok_or_else(|| {
            SyncWriteError::metadata_consolidation_failed(
                "shape overflow while building v2 metadata",
            )
        })?;
        Ok(format!(
            "{{\"zarr_format\":2,\"shape\":[{shape_elems}],\"chunks\":[{chunk_elems}],\"dtype\":\"{}\",\"compressor\":null,\"fill_value\":null,\"order\":\"C\",\"filters\":null,\"dimension_separator\":\"{}\"}}",
            dtype.v2_dtype, desc.separator
        ))
    }

    fn build_v3_array_metadata(desc: &ChunkDescriptor) -> Result<String, SyncWriteError> {
        // Non-registration consolidation has no trusted dtype contract.
        // Use a conservative raw-byte descriptor to avoid silently claiming float32.
        let dtype = Self::conservative_unknown_dtype();
        let chunk_elems = desc
            .chunk_bytes
            .checked_div(dtype.elem_bytes)
            .ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed("failed computing v3 chunk elements")
            })?;
        let chunks_count = usize::try_from(desc.max_index)
            .ok()
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                SyncWriteError::metadata_consolidation_failed(
                    "chunk index overflow while building v3 metadata",
                )
            })?;
        let shape_elems = chunks_count.checked_mul(chunk_elems).ok_or_else(|| {
            SyncWriteError::metadata_consolidation_failed(
                "shape overflow while building v3 metadata",
            )
        })?;
        Ok(format!(
            "{{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[{shape_elems}],\"data_type\":\"{}\",\"chunk_grid\":{{\"name\":\"regular\",\"configuration\":{{\"chunk_shape\":[{chunk_elems}]}}}},\"chunk_key_encoding\":{{\"name\":\"default\",\"configuration\":{{\"separator\":\"/\"}}}},\"fill_value\":0,\"codecs\":[{{\"name\":\"bytes\",\"configuration\":{{\"endian\":\"little\"}}}}],\"attributes\":{{}},\"dimension_names\":[null]}}",
            dtype.v3_data_type
        ))
    }

    fn write_per_array_metadata(
        &self,
        layout: &dyn ZarrLayoutAdapter,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let metadata_paths = layout.metadata_paths()?;
        let array_dirs = self.list_array_dirs()?;
        let mut consolidated_entries = Vec::new();
        for array_dir in array_dirs {
            let Some(chunk_desc) = Self::infer_chunk_descriptor(&array_dir)? else {
                continue;
            };
            let array_name = array_dir
                .file_name()
                .ok_or_else(|| {
                    SyncWriteError::metadata_consolidation_failed(format!(
                        "failed resolving array directory name for '{}'",
                        array_dir.display()
                    ))
                })?
                .to_string_lossy()
                .to_string();
            for rel in &metadata_paths.per_array_paths {
                let path = array_dir.join(rel);
                let payload = match layout.zarr_format() {
                    ZarrFormat::V2 => {
                        if rel.ends_with(".zarray") {
                            Self::build_v2_array_metadata(&chunk_desc)?
                        } else {
                            "{}".to_string()
                        }
                    }
                    ZarrFormat::V3 => {
                        if rel.ends_with("zarr.json") {
                            Self::build_v3_array_metadata(&chunk_desc)?
                        } else {
                            "{}".to_string()
                        }
                    }
                };
                Self::write_metadata_file(&path, &payload, self.fsync_policy)?;
                consolidated_entries.push((format!("{array_name}/{rel}"), payload));
            }
        }
        Ok(consolidated_entries)
    }

    /// Build coordinate metadata entries (key → payload) for a single coordinate axis.
    ///
    /// This is the **single source of truth** for coordinate metadata payloads.
    /// Both `write_coordinate_arrays_from_registration` (disk I/O) and
    /// `collect_coordinate_metadata_entries` (in-memory only) delegate here.
    fn build_coordinate_metadata_entries_for_axis(
        coord_name: &str,
        coord_values: &CoordValues,
        per_array_paths: &[String],
        zarr_format: ZarrFormat,
        default_dimension_separator: char,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let axis_len = coord_values.len();
        if axis_len == 0 {
            return Err(SyncWriteError::metadata_consolidation_failed(format!(
                "registered coord axis '{coord_name}' has zero length"
            )));
        }
        let shape = [axis_len];
        let chunks = [axis_len];
        let dim_order = vec![coord_name.to_string()];
        let (numeric_dtype, v2_utf8_dtype, v3_utf8_length_bytes) = match coord_values {
            CoordValues::Utf8(values) => {
                let (width, _) = Self::utf8_coord_values_to_fixed_width_utf32(values);
                (None, Some(format!("<U{width}")), Some(width * 4))
            }
            _ => {
                let dtype =
                    Self::dtype_for_numeric_coord_values(coord_values).ok_or_else(|| {
                        SyncWriteError::metadata_consolidation_failed(format!(
                            "unsupported coordinate dtype for '{coord_name}' during materialization"
                        ))
                    })?;
                (Some(dtype), None, None)
            }
        };

        let mut entries = Vec::new();
        for rel in per_array_paths {
            let payload = match zarr_format {
                ZarrFormat::V2 => {
                    if rel.ends_with(".zarray") {
                        if let Some(v2_dtype) = &v2_utf8_dtype {
                            Self::build_v2_registered_array_metadata_with_dtype(
                                &shape,
                                &chunks,
                                v2_dtype,
                                default_dimension_separator,
                            )
                        } else {
                            let dtype = numeric_dtype.as_ref().ok_or_else(|| {
                                SyncWriteError::metadata_consolidation_failed(format!(
                                    "missing numeric dtype for coordinate '{coord_name}'"
                                ))
                            })?;
                            Self::build_v2_registered_array_metadata(
                                &shape,
                                &chunks,
                                dtype,
                                default_dimension_separator,
                            )
                        }
                    } else if rel.ends_with(".zattrs") {
                        Self::build_v2_registered_array_attrs(&dim_order)
                    } else {
                        "{}".to_string()
                    }
                }
                ZarrFormat::V3 => {
                    if rel.ends_with("zarr.json") {
                        if let Some(utf32_length_bytes) = v3_utf8_length_bytes {
                            Self::build_v3_registered_utf8_array_metadata(
                                &shape,
                                &chunks,
                                utf32_length_bytes,
                                &dim_order,
                            )
                        } else {
                            let dtype = numeric_dtype.as_ref().ok_or_else(|| {
                                SyncWriteError::metadata_consolidation_failed(format!(
                                    "missing numeric dtype for coordinate '{coord_name}'"
                                ))
                            })?;
                            Self::build_v3_registered_array_metadata(
                                &shape, &chunks, dtype, &dim_order,
                            )
                        }
                    } else {
                        "{}".to_string()
                    }
                }
            };
            entries.push((format!("{coord_name}/{rel}"), payload));
        }
        Ok(entries)
    }

    fn write_coordinate_arrays_from_registration(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: &ArrayRegistration,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let metadata_paths = layout.metadata_paths()?;
        let default_dimension_separator = Self::default_v2_dimension_separator(layout)?;
        let mut consolidated_entries = Vec::new();

        for (coord_name, coord_values) in &registration.coords {
            // Build chunk binary payload (needed only for disk writes).
            let chunk_payload = match coord_values {
                CoordValues::Utf8(values) => {
                    let (_, payload) = Self::utf8_coord_values_to_fixed_width_utf32(values);
                    payload
                }
                _ => Self::numeric_coord_values_to_le_bytes(coord_values).ok_or_else(|| {
                    SyncWriteError::metadata_consolidation_failed(format!(
                        "failed encoding numeric coordinate payload for '{coord_name}'"
                    ))
                })?,
            };

            let coord_dir = self.dataset_root.join(coord_name);
            std::fs::create_dir_all(&coord_dir).map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!(
                        "failed creating coordinate metadata directory '{}': {e}",
                        coord_dir.display()
                    ),
                    e,
                )
            })?;

            // Build metadata entries via shared logic.
            let entries = Self::build_coordinate_metadata_entries_for_axis(
                coord_name,
                coord_values,
                &metadata_paths.per_array_paths,
                layout.zarr_format(),
                default_dimension_separator,
            )?;

            // Write metadata files to disk.
            for (key, payload) in &entries {
                // key is "coord_name/rel" — extract the rel portion for the file path.
                let expected_prefix = format!("{coord_name}/");
                debug_assert!(
                    key.starts_with(&expected_prefix),
                    "coordinate metadata key should start with '{coord_name}/': got '{key}'"
                );
                let rel = key.strip_prefix(&expected_prefix).unwrap_or(key);
                let path = coord_dir.join(rel);
                Self::write_metadata_file(&path, payload, self.fsync_policy)?;
            }
            consolidated_entries.extend(entries);

            let chunk_rel = layout
                .chunk_path_for_tuple_key(coord_name, &TupleChunkKey::new(vec![0]))?
                .relative_path;
            let chunk_path = self.dataset_root.join(chunk_rel);
            Self::write_binary_file(&chunk_path, &chunk_payload, self.fsync_policy)?;
        }

        Ok(consolidated_entries)
    }

    fn write_v2_consolidated_metadata(
        &self,
        entries: &[(String, String)],
    ) -> Result<(), SyncWriteError> {
        let mut json_entries = Vec::with_capacity(entries.len());
        for (path, payload) in entries {
            let escaped_path = Self::json_escape(path);
            json_entries.push(format!("\"{escaped_path}\":{payload}"));
        }
        let payload = format!(
            "{{\"zarr_consolidated_format\":1,\"metadata\":{{{}}}}}",
            json_entries.join(",")
        );
        Self::write_metadata_file(
            &self.dataset_root.join(".zmetadata"),
            &payload,
            self.fsync_policy,
        )
    }

    /// Write root-level metadata files (.zgroup, .zattrs, zarr.json) and return their entries.
    fn write_root_metadata(
        &self,
        layout: &dyn ZarrLayoutAdapter,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let paths = layout.metadata_paths()?;
        let mut entries = Vec::new();
        for rel in &paths.root_paths {
            if rel.ends_with(".zmetadata") {
                continue;
            }
            let path = self.dataset_root.join(rel);
            let payload = Self::root_metadata_payload(rel);
            Self::write_metadata_file(&path, payload, self.fsync_policy)?;
            entries.push((rel.to_string(), payload.to_string()));
        }
        Ok(entries)
    }

    /// Collect root-level metadata entries without writing files (for .zmetadata builder).
    fn collect_root_metadata_entries(
        layout: &dyn ZarrLayoutAdapter,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let paths = layout.metadata_paths()?;
        let mut entries = Vec::new();
        for rel in &paths.root_paths {
            if rel.ends_with(".zmetadata") {
                continue;
            }
            let payload = Self::root_metadata_payload(rel);
            entries.push((rel.to_string(), payload.to_string()));
        }
        Ok(entries)
    }

    fn root_metadata_payload(rel: &str) -> &'static str {
        if rel.ends_with(".zgroup") {
            r#"{"zarr_format":2}"#
        } else if rel.ends_with(".zattrs") {
            "{}"
        } else if rel.ends_with("zarr.json") {
            r#"{"zarr_format":3,"node_type":"group","attributes":{}}"#
        } else {
            "{}"
        }
    }

    /// Collect coordinate metadata entries without writing files (for .zmetadata builder).
    ///
    /// Delegates to `build_coordinate_metadata_entries_for_axis` — the same
    /// shared logic used by `write_coordinate_arrays_from_registration` — so
    /// the two can never drift apart.
    fn collect_coordinate_metadata_entries(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: &ArrayRegistration,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let metadata_paths = layout.metadata_paths()?;
        let default_dimension_separator = Self::default_v2_dimension_separator(layout)?;
        let mut entries = Vec::new();

        for (coord_name, coord_values) in &registration.coords {
            entries.extend(Self::build_coordinate_metadata_entries_for_axis(
                coord_name,
                coord_values,
                &metadata_paths.per_array_paths,
                layout.zarr_format(),
                default_dimension_separator,
            )?);
        }
        Ok(entries)
    }

    /// Write per-array metadata in parallel using rayon, with optional observed chunk bytes
    /// to avoid expensive `infer_chunk_descriptor` directory walking.
    fn write_per_array_metadata_from_registration_parallel(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: &ArrayRegistration,
        parallel_coord_names: &[String],
        observed_chunk_bytes: Option<&BTreeMap<String, usize>>,
    ) -> Result<Vec<(String, String)>, SyncWriteError> {
        let metadata_paths = layout.metadata_paths()?;
        let default_dimension_separator = Self::default_v2_dimension_separator(layout)?;
        let (dim_order, shape, chunks) =
            Self::shape_and_chunks_from_registration(registration, parallel_coord_names)?;
        let chunk_elements: usize = chunks.iter().product();
        let zarr_format = layout.zarr_format();

        // Parallel per-array metadata writes — each array directory is independent.
        // For small array counts (< 4), rayon scheduling overhead exceeds the benefit.
        const PAR_THRESHOLD: usize = 4;
        let map_fn = |array_name: &String| -> Result<Vec<(String, String)>, SyncWriteError> {
            let array_dir = self.dataset_root.join(array_name);
            std::fs::create_dir_all(&array_dir).map_err(|e| {
                SyncWriteError::metadata_consolidation_failed_with_cause(
                    format!(
                        "failed creating array metadata directory '{}': {e}",
                        array_dir.display()
                    ),
                    e,
                )
            })?;

            // Resolve dtype: prefer registration dtypes, then observed bytes, then FS walk.
            let (dtype, dimension_separator) = if let Some(dtype) = registration
                .array_names
                .iter()
                .position(|name| name == array_name)
                .and_then(|index| registration.array_dtypes.get(index))
            {
                (DTypeDescriptor::from(*dtype), default_dimension_separator)
            } else {
                match observed_chunk_bytes.and_then(|m| m.get(array_name.as_str())) {
                    Some(&bytes) => (
                        Self::dtype_from_chunk_bytes_and_elements(bytes, chunk_elements),
                        default_dimension_separator,
                    ),
                    None => {
                        let chunk_desc = Self::infer_chunk_descriptor(&array_dir)?;
                        let sep = chunk_desc
                            .as_ref()
                            .map_or(default_dimension_separator, |desc| desc.separator);
                        let dt = match chunk_desc.as_ref() {
                            Some(desc) => Self::dtype_from_chunk_bytes_and_elements(
                                desc.chunk_bytes,
                                chunk_elements,
                            ),
                            None => Self::conservative_unknown_dtype(),
                        };
                        (dt, sep)
                    }
                }
            };

            let mut entries = Vec::new();
            for rel in &metadata_paths.per_array_paths {
                let path = array_dir.join(rel);
                let payload = match zarr_format {
                    ZarrFormat::V2 => {
                        if rel.ends_with(".zarray") {
                            Self::build_v2_registered_array_metadata(
                                &shape,
                                &chunks,
                                &dtype,
                                dimension_separator,
                            )
                        } else if rel.ends_with(".zattrs") {
                            Self::build_v2_registered_array_attrs(&dim_order)
                        } else {
                            "{}".to_string()
                        }
                    }
                    ZarrFormat::V3 => {
                        if rel.ends_with("zarr.json") {
                            Self::build_v3_registered_array_metadata(
                                &shape, &chunks, &dtype, &dim_order,
                            )
                        } else {
                            "{}".to_string()
                        }
                    }
                };
                Self::write_metadata_file(&path, &payload, self.fsync_policy)?;
                entries.push((format!("{array_name}/{rel}"), payload));
            }
            Ok(entries)
        };

        let per_array_results: Vec<Result<Vec<(String, String)>, SyncWriteError>> =
            if registration.array_names.len() >= PAR_THRESHOLD {
                registration.array_names.par_iter().map(map_fn).collect()
            } else {
                registration.array_names.iter().map(map_fn).collect()
            };

        // Flatten results preserving registration order (par_iter preserves order).
        let mut consolidated_entries = Vec::new();
        for result in per_array_results {
            consolidated_entries.extend(result?);
        }
        Ok(consolidated_entries)
    }

    fn ensure_dataset_root(&self) -> Result<(), SyncWriteError> {
        if self.dataset_root.as_os_str().is_empty() {
            return Err(SyncWriteError::metadata_consolidation_failed(
                "dataset_root cannot be empty for local metadata consolidation",
            ));
        }
        std::fs::create_dir_all(&self.dataset_root).map_err(|e| {
            SyncWriteError::metadata_consolidation_failed_with_cause(
                format!(
                    "failed creating dataset root '{}': {e}",
                    self.dataset_root.display()
                ),
                e,
            )
        })
    }
}

impl MetadataConsolidator for LocalFsMetadataConsolidator {
    fn consolidate(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: Option<&ArrayRegistration>,
        parallel_coord_names: &[String],
    ) -> Result<(), SyncWriteError> {
        self.consolidate_scoped(
            layout,
            registration,
            parallel_coord_names,
            ConsolidationScope::Full,
            None,
        )
    }

    fn consolidate_scoped(
        &self,
        layout: &dyn ZarrLayoutAdapter,
        registration: Option<&ArrayRegistration>,
        parallel_coord_names: &[String],
        scope: ConsolidationScope,
        observed_chunk_bytes: Option<&BTreeMap<String, usize>>,
    ) -> Result<(), SyncWriteError> {
        self.ensure_dataset_root()?;
        let paths = layout.metadata_paths()?;

        let mut consolidated_entries: Vec<(String, String)> = Vec::new();

        // Phase A: Root metadata + coordinate arrays.
        let write_registration =
            scope == ConsolidationScope::Full || scope == ConsolidationScope::RegistrationOnly;
        let collect_only = scope == ConsolidationScope::CloseOnly;

        if write_registration {
            // Write root metadata files to disk.
            consolidated_entries.extend(self.write_root_metadata(layout)?);
            // Write coordinate arrays (metadata + binary chunk data) to disk.
            if let Some(registration) = registration {
                if !registration.coords.is_empty() {
                    consolidated_entries.extend(
                        self.write_coordinate_arrays_from_registration(layout, registration)?,
                    );
                }
            }
        } else if collect_only {
            // Collect entries in memory only (for .zmetadata builder) — no disk I/O.
            consolidated_entries.extend(Self::collect_root_metadata_entries(layout)?);
            if let Some(registration) = registration {
                if !registration.coords.is_empty() {
                    consolidated_entries
                        .extend(self.collect_coordinate_metadata_entries(layout, registration)?);
                }
            }
        }

        if scope == ConsolidationScope::RegistrationOnly {
            return Ok(());
        }

        // Phase B: Per-array metadata.
        let per_array_entries = if let Some(registration) = registration {
            if registration.coords.is_empty() {
                self.write_per_array_metadata(layout)?
            } else {
                self.write_per_array_metadata_from_registration_parallel(
                    layout,
                    registration,
                    parallel_coord_names,
                    observed_chunk_bytes,
                )?
            }
        } else {
            self.write_per_array_metadata(layout)?
        };
        consolidated_entries.extend(per_array_entries);

        // Phase C: Consolidated .zmetadata (v2 only).
        if layout.zarr_format() == ZarrFormat::V2
            && paths.root_paths.iter().any(|p| p.ends_with(".zmetadata"))
        {
            self.write_v2_consolidated_metadata(&consolidated_entries)?;
        }
        Ok(())
    }
}

struct ChunkDescriptor {
    max_index: u64,
    chunk_bytes: usize,
    separator: char,
}

struct DTypeDescriptor {
    v2_dtype: &'static str,
    v3_data_type: &'static str,
    elem_bytes: usize,
}

impl From<DataType> for DTypeDescriptor {
    fn from(value: DataType) -> Self {
        Self {
            v2_dtype: value.v2_dtype(),
            v3_data_type: value.v3_data_type(),
            elem_bytes: value.elem_bytes(),
        }
    }
}

impl DTypeDescriptor {
    fn v3_data_type_json(&self) -> String {
        if self.v3_data_type.starts_with('{') {
            self.v3_data_type.to_string()
        } else {
            format!("\"{}\"", self.v3_data_type)
        }
    }

    fn v3_fill_value_json(&self) -> &'static str {
        if self.v2_dtype.starts_with("<M8") || self.v2_dtype.starts_with("<m8") {
            "-9223372036854775808"
        } else if self.v2_dtype.starts_with("<f") || self.v2_dtype.starts_with("|f") {
            "0.0"
        } else if self.v2_dtype == "|b1" {
            "false"
        } else {
            "0"
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::{MetadataConsolidator, ZarrLayoutAdapter};
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        ArrayRegistration, ChunkKeyEncoding, ChunkKeySeparator, ChunkPathSpec, CoordMap,
        CoordValues, DataType, FsyncPolicy, MetadataPathSpec, TupleChunkKey, ZarrFormat,
        ZarrTargetConfig,
    };
    use crate::zarr::zarr_layout::DefaultZarrLayoutAdapter;

    use super::{DTypeDescriptor, LocalFsMetadataConsolidator};

    #[test]
    fn dtype_descriptor_uses_float_fill_value_for_v3_float_dtypes() {
        for dtype in [DataType::Float16, DataType::Float32, DataType::Float64] {
            assert_eq!(DTypeDescriptor::from(dtype).v3_fill_value_json(), "0.0");
        }
    }

    #[test]
    fn dtype_descriptor_uses_boolean_fill_value_for_v3_bool_dtype() {
        assert_eq!(
            DTypeDescriptor::from(DataType::Bool).v3_fill_value_json(),
            "false"
        );
    }

    fn unique_dataset_root(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock moved backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}.zarr"))
    }

    fn registered_4d_coords() -> CoordMap {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = coords.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = coords.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = coords.insert("lon".to_string(), CoordValues::F64(vec![30.0, 40.0]));
        coords
    }

    #[derive(Clone)]
    struct StaticMetadataLayout {
        format: ZarrFormat,
        root_paths: Vec<String>,
        per_array_paths: Vec<String>,
        tuple_separator: char,
    }

    impl ZarrLayoutAdapter for StaticMetadataLayout {
        fn zarr_format(&self) -> ZarrFormat {
            self.format
        }

        fn chunk_path_for(
            &self,
            array_name: &str,
            chunk_id: &ChunkId,
        ) -> Result<ChunkPathSpec, SyncWriteError> {
            Ok(ChunkPathSpec {
                relative_path: format!("{array_name}/{}", chunk_id.linear_index()),
            })
        }

        fn chunk_path_for_tuple_key(
            &self,
            array_name: &str,
            tuple_key: &TupleChunkKey,
        ) -> Result<ChunkPathSpec, SyncWriteError> {
            let rendered = tuple_key.render(self.tuple_separator);
            let relative_path = match self.format {
                ZarrFormat::V2 => format!("{array_name}/{rendered}"),
                ZarrFormat::V3 => format!("{array_name}/c/{rendered}"),
            };
            Ok(ChunkPathSpec { relative_path })
        }

        fn metadata_paths(&self) -> Result<MetadataPathSpec, SyncWriteError> {
            Ok(MetadataPathSpec {
                root_paths: self.root_paths.clone(),
                per_array_paths: self.per_array_paths.clone(),
            })
        }
    }

    #[test]
    fn parse_chunk_key_rejects_malformed_inputs() {
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("c.")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("c.a")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("1..2")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("not_a_chunk_key")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("c/1/.2")),
            None
        );
        assert_eq!(
            LocalFsMetadataConsolidator::parse_chunk_key(Path::new("c/1/a")),
            None
        );
    }

    #[test]
    fn infer_chunk_descriptor_rejects_mixed_separators() {
        let root = unique_dataset_root("e2s_meta_mixed_separators");
        let array_dir = root.join("temperature");
        std::fs::create_dir_all(array_dir.join("0")).expect("create slash chunk directory");
        std::fs::write(array_dir.join("0.0"), vec![1_u8, 2_u8, 3_u8, 4_u8])
            .expect("seed dot chunk");
        std::fs::write(array_dir.join("0").join("1"), vec![5_u8, 6_u8, 7_u8, 8_u8])
            .expect("seed slash chunk");

        let err = match LocalFsMetadataConsolidator::infer_chunk_descriptor(&array_dir) {
            Ok(_) => panic!("mixed separators should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { ref message, .. }
            if message.contains("inconsistent chunk key separators")
        ));
    }

    #[test]
    fn infer_chunk_descriptor_rejects_mixed_chunk_sizes() {
        let root = unique_dataset_root("e2s_meta_mixed_chunk_sizes");
        let array_dir = root.join("temperature");
        std::fs::create_dir_all(&array_dir).expect("create array directory");
        std::fs::write(array_dir.join("0.0"), vec![1_u8, 2_u8, 3_u8, 4_u8]).expect("seed chunk A");
        std::fs::write(array_dir.join("0.1"), vec![9_u8, 8_u8, 7_u8]).expect("seed chunk B");

        let err = match LocalFsMetadataConsolidator::infer_chunk_descriptor(&array_dir) {
            Ok(_) => panic!("inconsistent chunk sizes should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { ref message, .. }
            if message.contains("inconsistent chunk byte sizes")
        ));
    }

    #[test]
    fn infer_chunk_descriptor_returns_none_when_no_chunk_keys_are_present() {
        let root = unique_dataset_root("e2s_meta_no_chunk_keys");
        let array_dir = root.join("temperature");
        std::fs::create_dir_all(&array_dir).expect("create array directory");
        std::fs::write(array_dir.join(".zarray"), b"{}").expect("seed metadata file");
        std::fs::write(array_dir.join("not_a_chunk"), b"payload").expect("seed non-chunk file");

        let desc = LocalFsMetadataConsolidator::infer_chunk_descriptor(&array_dir)
            .expect("descriptor inference should not fail");
        assert!(desc.is_none(), "non-chunk files must not infer descriptor");
    }

    #[test]
    fn infer_dtype_uses_uint8_fallback_for_non_aligned_chunk_sizes() {
        let inferred = LocalFsMetadataConsolidator::infer_dtype(3);
        assert_eq!(inferred.v2_dtype, "|u1");
        assert_eq!(inferred.v3_data_type, "uint8");
        assert_eq!(inferred.elem_bytes, 1);
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_infers_float32_for_4_byte_elements() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(16, 4);
        assert_eq!(d.v2_dtype, "<f4");
        assert_eq!(d.v3_data_type, "float32");
        assert_eq!(d.elem_bytes, 4);
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_infers_float64_for_8_byte_elements() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(24, 3);
        assert_eq!(d.v2_dtype, "<f8");
        assert_eq!(d.v3_data_type, "float64");
        assert_eq!(d.elem_bytes, 8);
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_infers_float16_for_2_byte_elements() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(10, 5);
        assert_eq!(d.v2_dtype, "<f2");
        assert_eq!(d.v3_data_type, "float16");
        assert_eq!(d.elem_bytes, 2);
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_falls_back_to_uint8_for_zero_elements() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(16, 0);
        assert_eq!(d.v2_dtype, "|u1");
        assert_eq!(d.v3_data_type, "uint8");
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_falls_back_for_non_divisible_sizes() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(17, 4);
        assert_eq!(d.v2_dtype, "|u1");
        assert_eq!(d.v3_data_type, "uint8");
    }

    #[test]
    fn dtype_from_chunk_bytes_and_elements_falls_back_for_unusual_elem_size() {
        let d = LocalFsMetadataConsolidator::dtype_from_chunk_bytes_and_elements(15, 5);
        assert_eq!(d.v2_dtype, "|u1");
        assert_eq!(d.v3_data_type, "uint8");
    }

    #[test]
    fn coordinate_metadata_preserves_temporal_dtypes() {
        let v2_time_entries =
            LocalFsMetadataConsolidator::build_coordinate_metadata_entries_for_axis(
                "time",
                &CoordValues::DatetimeNs(vec![1_704_067_200_000_000_000]),
                &[".zarray".to_string()],
                ZarrFormat::V2,
                '.',
            )
            .expect("v2 datetime metadata");
        assert!(v2_time_entries[0].1.contains("\"dtype\":\"<M8[ns]\""));

        let v2_lead_time_entries =
            LocalFsMetadataConsolidator::build_coordinate_metadata_entries_for_axis(
                "lead_time",
                &CoordValues::TimedeltaNs(vec![21_600_000_000_000]),
                &[".zarray".to_string()],
                ZarrFormat::V2,
                '.',
            )
            .expect("v2 timedelta metadata");
        assert!(v2_lead_time_entries[0].1.contains("\"dtype\":\"<m8[ns]\""));

        let v3_time_entries =
            LocalFsMetadataConsolidator::build_coordinate_metadata_entries_for_axis(
                "time",
                &CoordValues::DatetimeNs(vec![1_704_067_200_000_000_000]),
                &["zarr.json".to_string()],
                ZarrFormat::V3,
                '/',
            )
            .expect("v3 datetime metadata");
        assert!(
            v3_time_entries[0]
                .1
                .contains("\"name\":\"numpy.datetime64\"")
        );
        assert!(
            v3_time_entries[0]
                .1
                .contains("\"fill_value\":-9223372036854775808")
        );
    }

    #[test]
    fn production_metadata_path_has_no_expect_panics_for_numeric_dtype() {
        let source = include_str!("metadata.rs");
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .expect("metadata.rs should contain production section");
        assert!(
            !production_source.contains("numeric dtype should exist"),
            "production metadata path must not rely on .expect() for numeric dtype resolution",
        );
    }

    #[test]
    fn format_string_list_escapes_json_control_characters() {
        let escaped = LocalFsMetadataConsolidator::format_string_list(&[
            "line\nbreak".to_string(),
            "tab\tsep".to_string(),
            "carriage\rreturn".to_string(),
            "ctrl\u{0001}x".to_string(),
        ]);

        assert!(
            escaped.contains("\\n"),
            "newline must be escaped for valid JSON string values"
        );
        assert!(
            escaped.contains("\\t"),
            "tab must be escaped for valid JSON string values"
        );
        assert!(
            escaped.contains("\\r"),
            "carriage return must be escaped for valid JSON string values"
        );
        assert!(
            escaped.contains("\\u0001"),
            "control code points must be emitted as \\uXXXX escapes"
        );
        assert!(
            !escaped.contains("line\nbreak"),
            "escaped JSON string list must not contain raw newline characters"
        );
    }

    #[test]
    fn v2_consolidated_metadata_escapes_control_characters_in_entry_paths() {
        let root = unique_dataset_root("e2s_meta_escape_consolidated_paths");
        std::fs::create_dir_all(&root).expect("create dataset root");
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        consolidator
            .write_v2_consolidated_metadata(&[("line\nname/.zarray".to_string(), "{}".to_string())])
            .expect("consolidated metadata write should succeed");

        let zmetadata =
            std::fs::read_to_string(root.join(".zmetadata")).expect(".zmetadata should be emitted");
        assert!(
            zmetadata.contains("line\\nname/.zarray"),
            "control characters in metadata entry paths must be escaped"
        );
        assert!(
            !zmetadata.contains("line\nname/.zarray"),
            "metadata JSON must not contain raw newline characters in keys"
        );
    }

    #[test]
    fn ordered_dim_names_deduplicates_parallel_coord_hints() {
        let coords = registered_4d_coords();
        let order = LocalFsMetadataConsolidator::ordered_dim_names(
            &coords,
            &[
                "time".to_string(),
                "time".to_string(),
                "lead_time".to_string(),
            ],
        );
        assert_eq!(
            order.iter().filter(|dim| dim.as_str() == "time").count(),
            1,
            "ordered dims should not duplicate parallel coord hints"
        );
        assert_eq!(order.first().map(String::as_str), Some("time"));
    }

    #[test]
    fn ordered_dim_names_preserves_registration_order_for_non_parallel_coords() {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = coords.insert("hrrr_y".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = coords.insert("hrrr_x".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let order = LocalFsMetadataConsolidator::ordered_dim_names(&coords, &["time".to_string()]);

        assert_eq!(order, vec!["time", "hrrr_y", "hrrr_x"]);
    }

    #[test]
    fn shape_and_chunks_from_registration_rejects_empty_coords() {
        let registration = ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let err = LocalFsMetadataConsolidator::shape_and_chunks_from_registration(
            &registration,
            &["time".to_string()],
        )
        .expect_err("empty coords should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { ref message, .. }
            if message.contains("coords are empty")
        ));
    }

    #[test]
    fn coord_map_rejects_zero_length_axis_for_shape_chunk_derivation() {
        let mut coords = CoordMap::new();
        let err = coords
            .insert("time".to_string(), CoordValues::I64(Vec::new()))
            .expect_err("zero-length axis should be rejected when building CoordMap");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("coordinate 'time'") && message.contains("at least one value")
        ));
    }

    #[test]
    fn write_per_array_metadata_skips_array_dirs_without_chunk_files() {
        let root = unique_dataset_root("e2s_meta_skip_empty_dirs");
        std::fs::create_dir_all(root.join("temperature")).expect("create array directory");
        let consolidator = LocalFsMetadataConsolidator::new(root);
        let entries = consolidator
            .write_per_array_metadata(&DefaultZarrLayoutAdapter::v2_default())
            .expect("metadata write should succeed for empty arrays");
        assert!(entries.is_empty(), "empty array dirs should be skipped");
    }

    #[test]
    fn consolidate_rejects_empty_dataset_root() {
        let consolidator = LocalFsMetadataConsolidator::new(PathBuf::new());
        let err = consolidator
            .consolidate(&DefaultZarrLayoutAdapter::v2_default(), None, &[])
            .expect_err("empty dataset_root should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { ref message, .. }
            if message.contains("dataset_root cannot be empty")
        ));
    }

    #[test]
    fn consolidate_v2_without_zmetadata_root_path_skips_consolidated_file() {
        let root = unique_dataset_root("e2s_meta_v2_no_zmetadata");
        std::fs::create_dir_all(root.join("temperature")).expect("create array directory");
        std::fs::write(
            root.join("temperature").join("0"),
            vec![1_u8, 2_u8, 3_u8, 4_u8],
        )
        .expect("seed chunk");

        let layout = StaticMetadataLayout {
            format: ZarrFormat::V2,
            root_paths: vec![
                ".zgroup".to_string(),
                ".zattrs".to_string(),
                "unknown_root.json".to_string(),
            ],
            per_array_paths: vec![
                ".zarray".to_string(),
                ".zattrs".to_string(),
                "extra.meta".to_string(),
            ],
            tuple_separator: '.',
        };
        let registration = ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        consolidator
            .consolidate(&layout, Some(&registration), &[])
            .expect("consolidation should succeed");

        assert!(
            !root.join(".zmetadata").exists(),
            ".zmetadata should not be emitted when not requested by layout root paths"
        );
        assert!(root.join("unknown_root.json").exists());
        assert!(root.join("temperature").join("extra.meta").exists());
    }

    #[test]
    fn consolidate_v2_coordinate_metadata_writes_placeholder_for_unknown_paths() {
        let root = unique_dataset_root("e2s_meta_v2_coord_extra_path");
        let layout = StaticMetadataLayout {
            format: ZarrFormat::V2,
            root_paths: vec![".zgroup".to_string(), ".zattrs".to_string()],
            per_array_paths: vec![
                ".zarray".to_string(),
                ".zattrs".to_string(),
                "extra.meta".to_string(),
            ],
            tuple_separator: '.',
        };
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        consolidator
            .consolidate(&layout, Some(&registration), &["time".to_string()])
            .expect("v2 consolidation should succeed");
        let extra = std::fs::read_to_string(root.join("time").join("extra.meta"))
            .expect("coordinate extra metadata path should exist");
        assert_eq!(extra, "{}");
    }

    #[test]
    fn consolidate_v3_coordinate_metadata_writes_placeholder_for_unknown_paths() {
        let root = unique_dataset_root("e2s_meta_v3_coord_extra_path");
        let layout = StaticMetadataLayout {
            format: ZarrFormat::V3,
            root_paths: vec!["zarr.json".to_string()],
            per_array_paths: vec!["zarr.json".to_string(), "extra.meta".to_string()],
            tuple_separator: '/',
        };
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        consolidator
            .consolidate(&layout, Some(&registration), &["time".to_string()])
            .expect("v3 consolidation should succeed");
        let extra = std::fs::read_to_string(root.join("time").join("extra.meta"))
            .expect("coordinate extra metadata path should exist");
        assert_eq!(extra, "{}");
    }

    #[test]
    fn coord_map_rejects_zero_length_coordinate_axis_before_consolidation() {
        let root = unique_dataset_root("e2s_meta_zero_axis");
        let _consolidator = LocalFsMetadataConsolidator::new(root);
        let mut coords = CoordMap::new();
        let err = coords
            .insert("time".to_string(), CoordValues::I64(Vec::new()))
            .expect_err("zero-length coordinate axis should fail CoordMap construction");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("coordinate 'time'") && message.contains("at least one value")
        ));
    }

    #[test]
    fn temp_path_for_rejects_path_without_terminal_file_name() {
        let err = LocalFsMetadataConsolidator::temp_path_for(Path::new("/"), "metadata")
            .expect_err("path with no terminal file name should fail");
        assert!(matches!(
            err,
            SyncWriteError::MetadataConsolidationFailed { ref message, .. }
            if message.contains("no terminal file name")
        ));
    }

    #[test]
    fn sync_parent_dir_best_effort_tolerates_missing_parent_or_open_failures() {
        LocalFsMetadataConsolidator::sync_parent_dir_best_effort(Path::new(""));
        LocalFsMetadataConsolidator::sync_parent_dir_best_effort(Path::new(
            "/definitely/nonexistent/e2s_meta/file.json",
        ));
    }

    #[test]
    fn local_fs_consolidator_writes_v2_metadata_paths() {
        let root = unique_dataset_root("e2s_meta_v2");
        std::fs::create_dir_all(root.join("temperature")).expect("create array directory");
        std::fs::write(root.join("temperature").join("0"), vec![1_u8, 2, 3, 4])
            .expect("seed chunk");
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        let layout = DefaultZarrLayoutAdapter::v2_default();
        consolidator
            .consolidate(&layout, None, &[])
            .expect("v2 metadata consolidation should succeed");
        assert!(root.join(".zgroup").exists());
        assert!(root.join(".zattrs").exists());
        assert!(root.join(".zmetadata").exists());
        assert!(root.join("temperature").join(".zarray").exists());
        assert!(root.join("temperature").join(".zattrs").exists());

        let zarray = std::fs::read_to_string(root.join("temperature").join(".zarray"))
            .expect("temperature/.zarray should exist");
        assert!(
            zarray.contains("\"dtype\":\"|u1\""),
            "non-registration metadata fallback should use conservative |u1 dtype, got: {zarray}"
        );
    }

    #[test]
    fn local_fs_consolidator_writes_v3_root_metadata_path() {
        let root = unique_dataset_root("e2s_meta_v3");
        std::fs::create_dir_all(root.join("tcwv").join("c")).expect("create array directory");
        std::fs::write(
            root.join("tcwv").join("c").join("0"),
            vec![9_u8, 8_u8, 7_u8, 6_u8],
        )
        .expect("seed chunk");
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::Default,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v3 target");
        consolidator
            .consolidate(&layout, None, &[])
            .expect("v3 metadata consolidation should succeed");
        assert!(root.join("zarr.json").exists());
        assert!(root.join("tcwv").join("zarr.json").exists());

        let array_zarr_json = std::fs::read_to_string(root.join("tcwv").join("zarr.json"))
            .expect("tcwv/zarr.json should exist");
        assert!(
            array_zarr_json.contains("\"data_type\":\"uint8\""),
            "non-registration metadata fallback should use conservative uint8 dtype, got: {array_zarr_json}"
        );
    }

    #[test]
    fn v3_registration_materializes_utf8_coordinate_arrays() {
        let root = unique_dataset_root("e2s_meta_v3_utf8_coords");
        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::Default,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v3 target");

        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = coords.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["control".to_string(), "pert01".to_string()]),
        );
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["t2m".to_string()],
            array_dtypes: Vec::new(),
        };
        let parallel_coord_names = vec!["time".to_string()];
        consolidator
            .consolidate(&layout, Some(&registration), &parallel_coord_names)
            .expect("v3 registration consolidation should support Utf8 coord arrays");

        let member_zarr_json = std::fs::read_to_string(root.join("member").join("zarr.json"))
            .expect("member/zarr.json should exist");
        assert!(
            member_zarr_json.contains(
                "\"data_type\":{\"name\":\"fixed_length_utf32\",\"configuration\":{\"length_bytes\":28}}"
            ),
            "expected fixed_length_utf32 data_type in member/zarr.json, got: {member_zarr_json}"
        );
        assert!(
            member_zarr_json.contains("\"fill_value\":\"\""),
            "expected empty-string fill_value in member/zarr.json, got: {member_zarr_json}"
        );

        let member_chunk = std::fs::read(root.join("member").join("c").join("0"))
            .expect("member coordinate chunk should exist");
        assert_eq!(
            member_chunk.len(),
            2 * 7 * 4,
            "expected UTF-32LE fixed-width payload bytes for 2 values × 7 codepoints"
        );

        let decode_utf32le_fixed = |bytes: &[u8], width: usize| -> Vec<String> {
            bytes
                .chunks_exact(width * 4)
                .map(|item| {
                    let mut out = String::new();
                    for cp_bytes in item.chunks_exact(4) {
                        let mut raw = [0_u8; 4];
                        raw.copy_from_slice(cp_bytes);
                        let cp = u32::from_le_bytes(raw);
                        if cp == 0 {
                            break;
                        }
                        let ch = char::from_u32(cp)
                            .expect("UTF-32LE coord payload should contain valid codepoints");
                        out.push(ch);
                    }
                    out
                })
                .collect()
        };
        assert_eq!(
            decode_utf32le_fixed(&member_chunk, 7),
            vec!["control".to_string(), "pert01".to_string()]
        );
    }

    #[test]
    fn v2_slash_registration_metadata_uses_slash_separator_for_tuple_keys() {
        let root = unique_dataset_root("e2s_meta_v2_slash_tuple");
        let tuple_chunk_path = root
            .join("temperature")
            .join("0")
            .join("1")
            .join("0")
            .join("0");
        std::fs::create_dir_all(
            tuple_chunk_path
                .parent()
                .expect("tuple chunk path should have parent"),
        )
        .expect("create tuple-key chunk directory");
        // 3 bytes -> inferred dtype should be |u1.
        std::fs::write(&tuple_chunk_path, vec![1_u8, 2_u8, 3_u8]).expect("seed tuple-key chunk");

        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V2,
            chunk_key_encoding: ChunkKeyEncoding::V2,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v2 slash target");

        let registration = ArrayRegistration {
            coords: registered_4d_coords(),
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let parallel_coord_names = vec!["time".to_string(), "lead_time".to_string()];
        consolidator
            .consolidate(&layout, Some(&registration), &parallel_coord_names)
            .expect("v2 slash registration consolidation should succeed");

        let zarray_path = root.join("temperature").join(".zarray");
        let zarray = std::fs::read_to_string(&zarray_path)
            .unwrap_or_else(|e| panic!("failed reading '{}': {e}", zarray_path.display()));
        assert!(
            zarray.contains("\"dimension_separator\":\"/\""),
            "expected slash dimension_separator in .zarray, got: {zarray}"
        );
        assert!(
            zarray.contains("\"dtype\":\"|u1\""),
            "expected |u1 dtype inferred from tuple-key chunk bytes, got: {zarray}"
        );
    }

    #[test]
    fn v2_dot_registration_metadata_uses_dot_separator_for_tuple_keys() {
        let root = unique_dataset_root("e2s_meta_v2_dot_tuple");
        std::fs::create_dir_all(root.join("temperature")).expect("create array directory");
        // 4-byte chunk payloads must still use conservative unknown-dtype fallback.
        std::fs::write(
            root.join("temperature").join("0.1.0.0"),
            vec![1_u8, 2_u8, 3_u8, 4_u8],
        )
        .expect("seed dot tuple-key chunk");

        let consolidator = LocalFsMetadataConsolidator::new(root.clone());
        let layout = DefaultZarrLayoutAdapter::v2_default();
        let registration = ArrayRegistration {
            coords: registered_4d_coords(),
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        };
        let parallel_coord_names = vec!["time".to_string(), "lead_time".to_string()];
        consolidator
            .consolidate(&layout, Some(&registration), &parallel_coord_names)
            .expect("v2 dot registration consolidation should succeed");

        let zarray_path = root.join("temperature").join(".zarray");
        let zarray = std::fs::read_to_string(&zarray_path)
            .unwrap_or_else(|e| panic!("failed reading '{}': {e}", zarray_path.display()));
        assert!(
            zarray.contains("\"dimension_separator\":\".\""),
            "expected dot dimension_separator in .zarray, got: {zarray}"
        );
        assert!(
            zarray.contains("\"dtype\":\"|u1\""),
            "registration metadata should use conservative |u1 dtype fallback for unknown chunk dtype, got: {zarray}"
        );
    }

    #[test]
    fn metadata_file_atomic_write_cleans_temp_on_rename_failure() {
        let root = unique_dataset_root("e2s_meta_atomic_metadata_rename_failure");
        let target_path = root.join(".zgroup");

        std::fs::create_dir_all(&target_path)
            .expect("create conflicting destination directory for rename failure");

        let err = LocalFsMetadataConsolidator::write_metadata_file(
            &target_path,
            r#"{"zarr_format":2}"#,
            FsyncPolicy::Always,
        )
        .expect_err("rename into existing directory should fail");

        let message = err.to_string();
        assert!(
            message.contains("failed renaming metadata temp file"),
            "expected rename-path atomic write failure, got: {message}"
        );

        let entries: Vec<String> = std::fs::read_dir(&root)
            .expect("metadata parent directory should exist")
            .map(|entry| {
                entry
                    .expect("directory entry should be readable")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            entries.iter().all(|name| !name.contains(".tmp-")),
            "failed metadata write should clean up temp files, found: {entries:?}"
        );
        let destination_meta =
            std::fs::metadata(&target_path).expect("destination should still exist");
        assert!(
            destination_meta.is_dir(),
            "failed rename must not clobber existing destination directory"
        );
    }

    #[test]
    fn binary_file_atomic_write_cleans_temp_on_rename_failure() {
        let root = unique_dataset_root("e2s_meta_atomic_binary_rename_failure");
        let target_path = root.join("time").join("0");

        std::fs::create_dir_all(&target_path)
            .expect("create conflicting coordinate destination directory for rename failure");

        let err = LocalFsMetadataConsolidator::write_binary_file(
            &target_path,
            &[1_u8, 2_u8, 3_u8],
            FsyncPolicy::Always,
        )
        .expect_err("rename into existing directory should fail");

        let message = err.to_string();
        assert!(
            message.contains("failed renaming coordinate chunk temp file"),
            "expected rename-path atomic write failure, got: {message}"
        );

        let parent = target_path
            .parent()
            .expect("target path should have parent directory");
        let entries: Vec<String> = std::fs::read_dir(parent)
            .expect("coordinate parent directory should exist")
            .map(|entry| {
                entry
                    .expect("directory entry should be readable")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            entries.iter().all(|name| !name.contains(".tmp-")),
            "failed binary write should clean up temp files, found: {entries:?}"
        );
        let destination_meta =
            std::fs::metadata(&target_path).expect("destination should still exist");
        assert!(
            destination_meta.is_dir(),
            "failed rename must not clobber existing destination directory"
        );
    }

    /// Verify that `collect_coordinate_metadata_entries` produces the same
    /// (key, payload) pairs as `write_coordinate_arrays_from_registration`.
    /// If the two implementations drift, `.zmetadata` will be inconsistent
    /// with the files on disk.
    #[test]
    fn collect_coordinate_entries_matches_write_coordinate_entries_v2() {
        let dataset_root = unique_dataset_root("coord_entry_consistency_v2");
        std::fs::create_dir_all(&dataset_root).expect("create dataset root");
        let consolidator = LocalFsMetadataConsolidator::new(dataset_root.clone());
        let layout = StaticMetadataLayout {
            format: ZarrFormat::V2,
            root_paths: vec![".zgroup".to_string(), ".zattrs".to_string()],
            per_array_paths: vec![".zarray".to_string(), ".zattrs".to_string()],
            tuple_separator: '.',
        };
        let coords = registered_4d_coords();
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["t2m".to_string()],
            array_dtypes: Vec::new(),
        };

        let written = consolidator
            .write_coordinate_arrays_from_registration(&layout, &registration)
            .expect("write_coordinate_arrays should succeed");
        let collected = consolidator
            .collect_coordinate_metadata_entries(&layout, &registration)
            .expect("collect_coordinate_metadata_entries should succeed");

        // Sort by key to make comparison deterministic (BTreeMap iteration
        // order may differ between the two methods).
        let mut written_meta: Vec<(String, String)> = written
            .into_iter()
            .filter(|(k, _)| !k.contains("/0"))
            .collect();
        written_meta.sort_by(|a, b| a.0.cmp(&b.0));
        let mut collected_sorted = collected.clone();
        collected_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(
            written_meta, collected_sorted,
            "collect_coordinate_metadata_entries must produce the same (key, payload) pairs \
             as write_coordinate_arrays_from_registration for v2"
        );

        let _ = std::fs::remove_dir_all(&dataset_root);
    }

    /// Same consistency check for v3.
    #[test]
    fn collect_coordinate_entries_matches_write_coordinate_entries_v3() {
        let dataset_root = unique_dataset_root("coord_entry_consistency_v3");
        std::fs::create_dir_all(&dataset_root).expect("create dataset root");
        let consolidator = LocalFsMetadataConsolidator::new(dataset_root.clone());
        let layout = StaticMetadataLayout {
            format: ZarrFormat::V3,
            root_paths: vec!["zarr.json".to_string()],
            per_array_paths: vec!["zarr.json".to_string()],
            tuple_separator: '/',
        };
        let coords = registered_4d_coords();
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["t2m".to_string()],
            array_dtypes: Vec::new(),
        };

        let written = consolidator
            .write_coordinate_arrays_from_registration(&layout, &registration)
            .expect("write_coordinate_arrays should succeed");
        let collected = consolidator
            .collect_coordinate_metadata_entries(&layout, &registration)
            .expect("collect_coordinate_metadata_entries should succeed");

        let mut written_meta: Vec<(String, String)> = written
            .into_iter()
            .filter(|(k, _)| !k.contains("/c/"))
            .collect();
        written_meta.sort_by(|a, b| a.0.cmp(&b.0));
        let mut collected_sorted = collected.clone();
        collected_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(
            written_meta, collected_sorted,
            "collect_coordinate_metadata_entries must produce the same (key, payload) pairs \
             as write_coordinate_arrays_from_registration for v3"
        );

        let _ = std::fs::remove_dir_all(&dataset_root);
    }

    /// Consistency check with Utf8 coordinate values.
    #[test]
    fn collect_coordinate_entries_matches_write_for_utf8_coords() {
        let dataset_root = unique_dataset_root("coord_entry_consistency_utf8");
        std::fs::create_dir_all(&dataset_root).expect("create dataset root");
        let consolidator = LocalFsMetadataConsolidator::new(dataset_root.clone());
        let layout = StaticMetadataLayout {
            format: ZarrFormat::V2,
            root_paths: vec![".zgroup".to_string(), ".zattrs".to_string()],
            per_array_paths: vec![".zarray".to_string(), ".zattrs".to_string()],
            tuple_separator: '.',
        };
        let mut coords = CoordMap::new();
        let _ = coords.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["ens01".to_string(), "ens02".to_string()]),
        );
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0]));
        let registration = ArrayRegistration {
            coords,
            array_names: vec!["t2m".to_string()],
            array_dtypes: Vec::new(),
        };

        let written = consolidator
            .write_coordinate_arrays_from_registration(&layout, &registration)
            .expect("write_coordinate_arrays should succeed");
        let collected = consolidator
            .collect_coordinate_metadata_entries(&layout, &registration)
            .expect("collect_coordinate_metadata_entries should succeed");

        let mut written_meta: Vec<(String, String)> = written
            .into_iter()
            .filter(|(k, _)| !k.contains("/0"))
            .collect();
        written_meta.sort_by(|a, b| a.0.cmp(&b.0));
        let mut collected_sorted = collected.clone();
        collected_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(
            written_meta, collected_sorted,
            "collect_coordinate_metadata_entries must produce the same (key, payload) pairs \
             as write_coordinate_arrays_from_registration for Utf8 coords"
        );

        let _ = std::fs::remove_dir_all(&dataset_root);
    }
}
