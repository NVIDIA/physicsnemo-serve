/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mixed-radix chunk planner with axis resolver and template caches.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::core::chunk_id::ChunkId;
use crate::core::contracts::ChunkPlanner;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    AxisResolver, BatchId, ChunkKeyTemplate, ChunkTask, CoordMap, CoordValues,
    InferenceWriteRequest, InputArray, InputArraySource, PlanTemplate, PlanTemplateKey,
    PlannedWriteBatch, TupleChunkKey, WriteExecutionConfig,
};

/// Per-axis index sets, registered-axis lengths, and subset lengths for active parallel dims.
type ParallelIndexResolution = (Vec<Vec<usize>>, Vec<usize>, Vec<usize>);

#[derive(Default)]
struct PlannerCaches {
    axis_resolvers: RwLock<HashMap<(u32, String), AxisResolver>>,
    plan_templates: RwLock<HashMap<PlanTemplateKey, PlanTemplate>>,
    /// Cached ChunkKeyTemplate, built on first `plan_batch()` call.
    /// Immutable after creation since `registered_coords` doesn't change
    /// after `add_array()`.
    chunk_key_template: RwLock<Option<ChunkKeyTemplate>>,
    /// Number of times the cached `ChunkKeyTemplate` was reused (read-lock fast path).
    template_cache_hits: AtomicUsize,
    /// Number of times the `ChunkKeyTemplate` was built (write-lock slow path).
    template_cache_misses: AtomicUsize,
}

/// Chunk planner using the `MixedRadixStreaming` algorithm.
///
/// Translates inference write requests into planned task batches without
/// full Cartesian meshgrid materialization.
///
/// Each `ChunkTask` produced by `plan_batch()` includes a pre-computed
/// `TupleChunkKey` (e.g., `[0, 4, 0, 0]`) resolved via a cached
/// `ChunkKeyTemplate`, so downstream components can directly render
/// Zarr chunk paths without deferred linear→tuple conversion at `close()` time.
pub struct MixedRadixChunkPlanner {
    config: WriteExecutionConfig,
    next_batch_id: AtomicU64,
    caches: PlannerCaches,
}

impl Default for MixedRadixChunkPlanner {
    fn default() -> Self {
        Self {
            config: WriteExecutionConfig::default(),
            next_batch_id: AtomicU64::new(1),
            caches: PlannerCaches::default(),
        }
    }
}

impl MixedRadixChunkPlanner {
    /// Create a planner with the given execution configuration.
    #[must_use]
    pub fn new(config: WriteExecutionConfig) -> Self {
        Self {
            config,
            next_batch_id: AtomicU64::new(1),
            caches: PlannerCaches::default(),
        }
    }

    /// Returns the active write execution configuration.
    #[must_use]
    pub fn config(&self) -> &WriteExecutionConfig {
        &self.config
    }

    fn get_or_build_chunk_key_template(
        &self,
        registered_coords: &CoordMap,
        active_parallel_dims: &[String],
    ) -> Result<ChunkKeyTemplate, SyncWriteError> {
        // Fast path: read lock to check if already cached.
        if let Ok(guard) = self.caches.chunk_key_template.read() {
            if let Some(ref template) = *guard {
                self.caches
                    .template_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(template.clone());
            }
        }

        // Slow path: build and cache under write lock.
        self.caches
            .template_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        let template = ChunkKeyTemplate::build(registered_coords, active_parallel_dims)?;
        if let Ok(mut guard) = self.caches.chunk_key_template.write() {
            *guard = Some(template.clone());
        }
        Ok(template)
    }

    /// Returns `(cache_hits, cache_misses)` for the `ChunkKeyTemplate` cache.
    ///
    /// Useful for diagnostics and testing: after a full inference loop,
    /// `cache_misses` should be exactly `1` (the first `plan_batch()` build)
    /// and `cache_hits` should equal the number of subsequent `plan_batch()` calls.
    #[must_use]
    pub fn template_cache_stats(&self) -> (usize, usize) {
        let hits = self.caches.template_cache_hits.load(Ordering::Relaxed);
        let misses = self.caches.template_cache_misses.load(Ordering::Relaxed);
        (hits, misses)
    }

    fn resolve_active_parallel_dims(
        &self,
        registered_coords: &CoordMap,
    ) -> Result<Vec<String>, SyncWriteError> {
        if let Some(explicit) = &self.config.parallel_coords_config.parallel_coords {
            let dims = canonicalize_explicit_parallel_names(
                explicit,
                &self
                    .config
                    .parallel_coords_config
                    .default_parallel_coord_names,
            );
            for dim in &dims {
                if !registered_coords.contains_key(dim.as_str()) {
                    return Err(SyncWriteError::UnknownParallelCoord { coord: dim.clone() });
                }
            }
            return Ok(dims);
        }

        Ok(self
            .config
            .parallel_coords_config
            .default_parallel_coord_names
            .iter()
            .filter(|dim| registered_coords.contains_key(dim.as_str()))
            .cloned()
            .collect())
    }

    fn maybe_cache_template(
        &self,
        array_id: u32,
        active_parallel_dims: &[String],
        subset_lengths: &[usize],
        axis_lengths: &[usize],
        task_count: usize,
    ) -> Result<(), SyncWriteError> {
        let key = PlanTemplateKey::new(
            array_id,
            active_parallel_dims.to_vec(),
            subset_lengths.to_vec(),
        )?;
        let radix_strides = Self::axis_radix_strides(axis_lengths)?;
        if let Ok(mut templates) = self.caches.plan_templates.write() {
            templates.entry(key).or_insert(PlanTemplate {
                dim_order: active_parallel_dims.to_vec(),
                radix_strides,
                task_count,
            });
        }
        Ok(())
    }

    fn resolve_parallel_index_sets_and_lengths(
        &self,
        req: &InferenceWriteRequest,
        registered_coords: &CoordMap,
        active_parallel_dims: &[String],
    ) -> Result<ParallelIndexResolution, SyncWriteError> {
        let mut axis_index_sets = Vec::with_capacity(active_parallel_dims.len());
        let mut axis_lengths = Vec::with_capacity(active_parallel_dims.len());
        let mut subset_lengths = Vec::with_capacity(active_parallel_dims.len());

        for dim in active_parallel_dims {
            let registered = registered_coords
                .get(dim.as_str())
                .ok_or_else(|| SyncWriteError::UnknownParallelCoord { coord: dim.clone() })?;
            if registered.is_empty() {
                return Err(SyncWriteError::Validation {
                    message: format!("registered coord '{dim}' has no values"),
                });
            }

            let step = req
                .coords
                .get(dim.as_str())
                .ok_or_else(|| SyncWriteError::Validation {
                    message: format!(
                        "missing active parallel coord '{dim}' in write() step coords"
                    ),
                })?;
            if step.is_empty() {
                return Err(SyncWriteError::Validation {
                    message: format!("active parallel coord '{dim}' has empty step values"),
                });
            }

            let resolver = self.resolver_for_axis(registered);
            let step_index_set =
                index_of_step_values_with_resolver(step, registered, &resolver).ok_or_else(|| {
                SyncWriteError::Validation {
                    message: format!(
                        "active parallel coord '{dim}' has step value not present in registered coords"
                    ),
                }
            })?;
            axis_index_sets.push(step_index_set);
            axis_lengths.push(registered.len());
            subset_lengths.push(step.len());
        }

        Ok((axis_index_sets, axis_lengths, subset_lengths))
    }

    fn mixed_radix_linear_index(
        &self,
        axis_indices: &[usize],
        axis_lengths: &[usize],
    ) -> Result<u64, SyncWriteError> {
        if axis_indices.is_empty() {
            return Ok(0);
        }
        debug_assert_eq!(axis_indices.len(), axis_lengths.len());

        let mut linear_index = 0_u64;
        let mut stride = 1_u64;
        for (index, length) in axis_indices.iter().zip(axis_lengths).rev() {
            let index_u64 = u64::try_from(*index).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
            let length_u64 = u64::try_from(*length).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
            let term = index_u64
                .checked_mul(stride)
                .ok_or(SyncWriteError::ChunkIdOverflow)?;
            linear_index = linear_index
                .checked_add(term)
                .ok_or(SyncWriteError::ChunkIdOverflow)?;
            stride = stride
                .checked_mul(length_u64)
                .ok_or(SyncWriteError::ChunkIdOverflow)?;
        }

        Ok(linear_index)
    }

    fn subset_mixed_radix_strides(subset_lengths: &[usize]) -> Result<Vec<usize>, SyncWriteError> {
        if subset_lengths.is_empty() {
            return Ok(Vec::new());
        }
        let mut strides = vec![1_usize; subset_lengths.len()];
        let mut running = 1_usize;
        for idx in (0..subset_lengths.len()).rev() {
            strides[idx] = running;
            running = running
                .checked_mul(subset_lengths[idx])
                .ok_or(SyncWriteError::ChunkIdOverflow)?;
        }
        Ok(strides)
    }

    fn axis_radix_strides(axis_lengths: &[usize]) -> Result<Vec<u64>, SyncWriteError> {
        if axis_lengths.is_empty() {
            return Ok(vec![1]);
        }

        let mut strides = vec![1_u64; axis_lengths.len()];
        let mut running = 1_u64;
        for idx in (0..axis_lengths.len()).rev() {
            strides[idx] = running;
            let axis_len =
                u64::try_from(axis_lengths[idx]).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
            running = running
                .checked_mul(axis_len)
                .ok_or(SyncWriteError::ChunkIdOverflow)?;
        }

        Ok(strides)
    }

    fn total_subset_combinations(subset_lengths: &[usize]) -> Result<usize, SyncWriteError> {
        if subset_lengths.is_empty() {
            return Ok(1);
        }
        subset_lengths.iter().try_fold(1_usize, |acc, len| {
            acc.checked_mul(*len).ok_or(SyncWriteError::ChunkIdOverflow)
        })
    }

    fn required_bytes_per_task(
        input_nbytes: usize,
        subset_combo_count: usize,
    ) -> Result<usize, SyncWriteError> {
        if subset_combo_count == 0 {
            return Err(SyncWriteError::Validation {
                message: "subset combo count must be greater than zero".to_string(),
            });
        }
        if input_nbytes % subset_combo_count != 0 {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "input.nbytes ({input_nbytes}) must be divisible by derived task count ({subset_combo_count})"
                ),
            });
        }
        Ok(input_nbytes / subset_combo_count)
    }

    fn task_byte_range(
        combo_ordinal: usize,
        required_bytes_per_task: usize,
    ) -> Result<(usize, usize), SyncWriteError> {
        let start = combo_ordinal
            .checked_mul(required_bytes_per_task)
            .ok_or(SyncWriteError::ChunkIdOverflow)?;
        let end = start
            .checked_add(required_bytes_per_task)
            .ok_or(SyncWriteError::ChunkIdOverflow)?;
        Ok((start, end))
    }

    fn slice_input_for_task(
        input: &InputArray,
        combo_ordinal: usize,
        required_bytes_per_task: usize,
    ) -> Result<InputArray, SyncWriteError> {
        let (start, end) = Self::task_byte_range(combo_ordinal, required_bytes_per_task)?;
        if let Some(ptr) = input.source.as_host_buffer_ptr() {
            let start_u64 = u64::try_from(start).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
            let task_ptr = ptr
                .checked_add(start_u64)
                .ok_or(SyncWriteError::Validation {
                    message: format!(
                        "host buffer pointer overflow for base ptr {ptr} with byte offset {start}"
                    ),
                })?;
            return Ok(InputArray {
                nbytes: required_bytes_per_task,
                // SAFETY: `task_ptr` is derived from an existing validated host
                // pointer plus checked byte offset for this derived task slice.
                source: unsafe { InputArraySource::from_host_buffer_ptr(task_ptr) },
            });
        }

        match &input.source {
            InputArraySource::HostBytes(payload) => {
                if payload.len() < input.nbytes {
                    return Err(SyncWriteError::Validation {
                        message: format!(
                            "host payload len ({}) is smaller than declared input.nbytes ({})",
                            payload.len(),
                            input.nbytes
                        ),
                    });
                }
                if end > payload.len() {
                    return Err(SyncWriteError::Validation {
                        message: format!(
                            "derived chunk byte range [{start}..{end}) exceeds host payload len ({})",
                            payload.len()
                        ),
                    });
                }
                let payload_ptr = payload.as_ptr() as usize;
                let payload_ptr_u64 =
                    u64::try_from(payload_ptr).map_err(|_| SyncWriteError::Validation {
                        message: format!(
                            "host payload base pointer does not fit u64: {payload_ptr}"
                        ),
                    })?;
                let start_u64 =
                    u64::try_from(start).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
                let task_ptr = payload_ptr_u64
                    .checked_add(start_u64)
                    .ok_or(SyncWriteError::Validation {
                        message: format!(
                            "host payload pointer overflow for base ptr {payload_ptr_u64} with byte offset {start}"
                        ),
                    })?;
                Ok(InputArray {
                    nbytes: required_bytes_per_task,
                    // SAFETY: `task_ptr` is derived from an in-bounds payload
                    // base pointer plus checked byte offset for this task slice.
                    source: unsafe { InputArraySource::from_host_buffer_ptr(task_ptr) },
                })
            }
            InputArraySource::CudaDevicePtr {
                ptr,
                device_ordinal,
                producer_stream,
            } => {
                let start_u64 =
                    u64::try_from(start).map_err(|_| SyncWriteError::ChunkIdOverflow)?;
                let task_ptr = ptr.checked_add(start_u64).ok_or(SyncWriteError::Validation {
                    message: format!(
                        "cuda device pointer overflow for base ptr {ptr} with byte offset {start}"
                    ),
                })?;
                Ok(InputArray {
                    nbytes: required_bytes_per_task,
                    source: InputArraySource::CudaDevicePtr {
                        ptr: task_ptr,
                        device_ordinal: *device_ordinal,
                        producer_stream: *producer_stream,
                    },
                })
            }
            InputArraySource::__InternalHostBufferPtr { .. } => unreachable!(
                "internal host pointer source must be handled by as_host_buffer_ptr() pre-branch"
            ),
        }
    }

    fn axis_indices_for_subset_combo(
        axis_index_sets: &[Vec<usize>],
        subset_strides: &[usize],
        combo_ordinal: usize,
    ) -> Vec<usize> {
        if axis_index_sets.is_empty() {
            return Vec::new();
        }

        let mut axis_indices = Vec::with_capacity(axis_index_sets.len());
        for (dim_idx, indices) in axis_index_sets.iter().enumerate() {
            let stride = subset_strides[dim_idx];
            let pos = if indices.len() == 1 {
                0
            } else {
                (combo_ordinal / stride) % indices.len()
            };
            axis_indices.push(indices[pos]);
        }
        axis_indices
    }

    fn validate_step_coords_contract(
        &self,
        req: &InferenceWriteRequest,
        registered_coords: &CoordMap,
        active_parallel_dims: &[String],
    ) -> Result<(), SyncWriteError> {
        for step_key in req.coords.keys() {
            if !registered_coords.contains_key(step_key.as_str()) {
                return Err(SyncWriteError::Validation {
                    message: format!(
                        "step coord '{step_key}' is not registered in add_array() coord contract"
                    ),
                });
            }
        }

        for (coord_name, registered_values) in registered_coords {
            if active_parallel_dims.iter().any(|dim| dim == coord_name) {
                continue;
            }
            let Some(step_values) = req.coords.get(coord_name.as_str()) else {
                return Err(SyncWriteError::Validation {
                    message: format!(
                        "missing non-parallel coord '{coord_name}' in write() step coords"
                    ),
                });
            };
            if !coord_values_match_contract(step_values, registered_values) {
                return Err(SyncWriteError::Validation {
                    message: format!(
                        "non-parallel coord '{coord_name}' must match full registered coordinate values"
                    ),
                });
            }
        }

        Ok(())
    }

    fn resolver_for_axis(&self, registered_axis: &CoordValues) -> AxisResolver {
        match registered_axis {
            CoordValues::Utf8(_) => AxisResolver::HashMapResolver,
            CoordValues::I64(values)
            | CoordValues::DatetimeNs(values)
            | CoordValues::TimedeltaNs(values) => {
                if self.config.planner_caches.prefer_affine_resolver {
                    if let Some((start_ns, step_ns, len)) = affine_i64_axis(values) {
                        return AxisResolver::Affine {
                            start_ns,
                            step_ns,
                            len,
                        };
                    }
                }
                AxisResolver::SortedBinarySearch
            }
            CoordValues::U64(_)
            | CoordValues::I32(_)
            | CoordValues::U32(_)
            | CoordValues::F32(_)
            | CoordValues::F64(_) => AxisResolver::SortedBinarySearch,
        }
    }
}

/// Produce a deterministic ordering for explicit parallel coord names.
///
/// Keys present in `default_order` come first (preserving the default order),
/// followed by any remaining keys sorted alphabetically.
fn canonicalize_explicit_parallel_names(
    explicit: &CoordMap,
    default_order: &[String],
) -> Vec<String> {
    let mut ordered = Vec::with_capacity(explicit.len());
    for name in default_order {
        if explicit.contains_key(name.as_str()) && !ordered.contains(name) {
            ordered.push(name.clone());
        }
    }
    let mut remaining: Vec<String> = explicit
        .keys()
        .filter(|k| !ordered.contains(k))
        .cloned()
        .collect();
    remaining.sort();
    ordered.extend(remaining);
    ordered
}

fn affine_i64_axis(values: &[i64]) -> Option<(i64, i64, usize)> {
    if values.len() < 2 {
        return None;
    }
    let start = values[0];
    let step = values[1].checked_sub(start)?;
    if step == 0 {
        return None;
    }
    for pair in values.windows(2) {
        let delta = pair[1].checked_sub(pair[0])?;
        if delta != step {
            return None;
        }
    }
    Some((start, step, values.len()))
}

fn coord_values_match_contract(
    step: &crate::core::types::CoordValues,
    registered: &crate::core::types::CoordValues,
) -> bool {
    use crate::core::types::CoordValues;

    if step == registered {
        return true;
    }

    match (step, registered) {
        (CoordValues::F64(step_vals), CoordValues::F32(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| *step as f32 == *registered)
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::F32(step_vals), CoordValues::F64(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| f64::from(*step) == *registered)
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::I64(step_vals), CoordValues::I32(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| *step == i64::from(*registered))
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::I32(step_vals), CoordValues::I64(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| i64::from(*step) == *registered)
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::I64(step_vals), CoordValues::U32(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| u32::try_from(*step).is_ok_and(|v| v == *registered))
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::U32(step_vals), CoordValues::I64(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| i64::from(*step) == *registered)
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::U64(step_vals), CoordValues::U32(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| u32::try_from(*step).is_ok_and(|v| v == *registered))
                && step_vals.len() == registered_vals.len()
        }
        (CoordValues::U32(step_vals), CoordValues::U64(registered_vals)) => {
            step_vals
                .iter()
                .zip(registered_vals)
                .all(|(step, registered)| u64::from(*step) == *registered)
                && step_vals.len() == registered_vals.len()
        }
        _ => false,
    }
}

fn sorted_i64(values: &[i64]) -> bool {
    values.windows(2).all(|pair| pair[0] <= pair[1])
}

fn sorted_u64(values: &[u64]) -> bool {
    values.windows(2).all(|pair| pair[0] <= pair[1])
}

fn sorted_utf8(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] <= pair[1])
}

impl ChunkPlanner for MixedRadixChunkPlanner {
    fn plan_batch(
        &self,
        req: &InferenceWriteRequest,
        array_ids: &[u32],
        registered_coords: &CoordMap,
    ) -> Result<PlannedWriteBatch, SyncWriteError> {
        if req.arrays.len() != req.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "len(arrays) != len(array_names): {} != {}",
                    req.arrays.len(),
                    req.array_names.len()
                ),
            });
        }
        if array_ids.len() != req.arrays.len() {
            return Err(SyncWriteError::Validation {
                message: format!(
                    "len(array_ids) != len(arrays): {} != {}",
                    array_ids.len(),
                    req.arrays.len()
                ),
            });
        }
        let active_parallel_dims = self.resolve_active_parallel_dims(registered_coords)?;
        self.validate_step_coords_contract(req, registered_coords, &active_parallel_dims)?;
        let (axis_index_sets, axis_lengths, subset_lengths) = self
            .resolve_parallel_index_sets_and_lengths(
                req,
                registered_coords,
                &active_parallel_dims,
            )?;
        let subset_strides = Self::subset_mixed_radix_strides(&subset_lengths)?;
        let subset_combo_count = Self::total_subset_combinations(&subset_lengths)?;

        // Get or build cached ChunkKeyTemplate for write-time tuple key resolution.
        let chunk_key_template =
            self.get_or_build_chunk_key_template(registered_coords, &active_parallel_dims)?;

        let batch_id = BatchId::from(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        let task_capacity = req
            .arrays
            .len()
            .checked_mul(subset_combo_count)
            .ok_or(SyncWriteError::ChunkIdOverflow)?;
        let mut chunk_ids = Vec::with_capacity(task_capacity);
        let mut tasks = Vec::with_capacity(task_capacity);

        for (task_index, (array_name, input)) in req.array_names.iter().zip(&req.arrays).enumerate()
        {
            let array_id_u32 = array_ids[task_index];
            let required_bytes_per_task =
                Self::required_bytes_per_task(input.nbytes, subset_combo_count)?;
            self.maybe_cache_template(
                array_id_u32,
                &active_parallel_dims,
                &subset_lengths,
                &axis_lengths,
                subset_combo_count,
            )?;
            if let Ok(mut resolvers) = self.caches.axis_resolvers.write() {
                if active_parallel_dims.is_empty() {
                    resolvers
                        .entry((array_id_u32, "placeholder".to_string()))
                        .or_insert(AxisResolver::SortedBinarySearch);
                } else {
                    for dim in &active_parallel_dims {
                        let resolver = registered_coords
                            .get(dim.as_str())
                            .map(|axis| self.resolver_for_axis(axis))
                            .unwrap_or(AxisResolver::SortedBinarySearch);
                        resolvers
                            .entry((array_id_u32, dim.clone()))
                            .or_insert(resolver);
                    }
                }
            }

            for combo_ordinal in 0..subset_combo_count {
                let axis_indices = Self::axis_indices_for_subset_combo(
                    &axis_index_sets,
                    &subset_strides,
                    combo_ordinal,
                );
                let linear_index = self.mixed_radix_linear_index(&axis_indices, &axis_lengths)?;
                let chunk_id = ChunkId::new(array_id_u32, linear_index);

                // Resolve tuple key from parallel axis indices via ChunkKeyTemplate.
                // For flat 1D arrays (no coords), fall back to a single-element tuple
                // using the linear index so that the writer emits `array_name/0` etc.
                let tuple_key = if chunk_key_template.dim_order().is_empty() {
                    let idx = usize::try_from(linear_index)
                        .map_err(|_| SyncWriteError::ChunkIdOverflow)?;
                    TupleChunkKey::new(vec![idx])
                } else {
                    chunk_key_template.resolve_tuple_key(&axis_indices)?
                };

                chunk_ids.push(chunk_id);
                let task_input =
                    Self::slice_input_for_task(input, combo_ordinal, required_bytes_per_task)?;
                tasks.push(ChunkTask {
                    batch_id,
                    array_name: array_name.clone(),
                    chunk_id,
                    tuple_key,
                    required_bytes: required_bytes_per_task,
                    input: task_input,
                });
            }
        }

        Ok(PlannedWriteBatch {
            batch_id,
            chunk_ids,
            tasks,
        })
    }
}

fn index_of_step_values_with_resolver(
    step: &crate::core::types::CoordValues,
    registered: &crate::core::types::CoordValues,
    resolver: &AxisResolver,
) -> Option<Vec<usize>> {
    match (step, registered) {
        (
            crate::core::types::CoordValues::I64(step_vals),
            crate::core::types::CoordValues::I64(registered_vals),
        )
        | (
            crate::core::types::CoordValues::DatetimeNs(step_vals),
            crate::core::types::CoordValues::DatetimeNs(registered_vals),
        )
        | (
            crate::core::types::CoordValues::TimedeltaNs(step_vals),
            crate::core::types::CoordValues::TimedeltaNs(registered_vals),
        ) => match resolver {
            AxisResolver::Affine {
                start_ns,
                step_ns,
                len,
            } => {
                if *step_ns == 0 || *len != registered_vals.len() {
                    return None;
                }
                step_vals
                    .iter()
                    .map(|value| {
                        let delta = value.checked_sub(*start_ns)?;
                        if delta % *step_ns != 0 {
                            return None;
                        }
                        let pos_i64 = delta / *step_ns;
                        let pos_usize = usize::try_from(pos_i64).ok()?;
                        if pos_usize >= *len {
                            return None;
                        }
                        if registered_vals.get(pos_usize)? != value {
                            return None;
                        }
                        Some(pos_usize)
                    })
                    .collect()
            }
            AxisResolver::HashMapResolver => {
                let index_map: HashMap<i64, usize> = registered_vals
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, value)| (value, idx))
                    .collect();
                step_vals
                    .iter()
                    .map(|v| index_map.get(v).copied())
                    .collect()
            }
            AxisResolver::SortedBinarySearch => {
                if sorted_i64(registered_vals) {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.binary_search(v).ok())
                        .collect()
                } else {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.iter().position(|candidate| candidate == v))
                        .collect()
                }
            }
        },
        (
            crate::core::types::CoordValues::U64(step_vals),
            crate::core::types::CoordValues::U64(registered_vals),
        ) => match resolver {
            AxisResolver::HashMapResolver => {
                let index_map: HashMap<u64, usize> = registered_vals
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, value)| (value, idx))
                    .collect();
                step_vals
                    .iter()
                    .map(|v| index_map.get(v).copied())
                    .collect()
            }
            AxisResolver::Affine { .. } | AxisResolver::SortedBinarySearch => {
                if sorted_u64(registered_vals) {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.binary_search(v).ok())
                        .collect()
                } else {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.iter().position(|candidate| candidate == v))
                        .collect()
                }
            }
        },
        (
            crate::core::types::CoordValues::F64(step_vals),
            crate::core::types::CoordValues::F64(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| registered_vals.iter().position(|candidate| candidate == v))
            .collect(),
        (
            crate::core::types::CoordValues::F64(step_vals),
            crate::core::types::CoordValues::F32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                let coerced = *v as f32;
                registered_vals
                    .iter()
                    .position(|candidate| *candidate == coerced)
            })
            .collect(),
        (
            crate::core::types::CoordValues::F32(step_vals),
            crate::core::types::CoordValues::F64(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                let coerced = f64::from(*v);
                registered_vals
                    .iter()
                    .position(|candidate| *candidate == coerced)
            })
            .collect(),
        (
            crate::core::types::CoordValues::I32(step_vals),
            crate::core::types::CoordValues::I32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| registered_vals.iter().position(|candidate| candidate == v))
            .collect(),
        (
            crate::core::types::CoordValues::I64(step_vals),
            crate::core::types::CoordValues::I32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                i32::try_from(*v).ok().and_then(|coerced| {
                    registered_vals
                        .iter()
                        .position(|candidate| *candidate == coerced)
                })
            })
            .collect(),
        (
            crate::core::types::CoordValues::I32(step_vals),
            crate::core::types::CoordValues::I64(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                let coerced = i64::from(*v);
                registered_vals
                    .iter()
                    .position(|candidate| *candidate == coerced)
            })
            .collect(),
        (
            crate::core::types::CoordValues::U32(step_vals),
            crate::core::types::CoordValues::U32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| registered_vals.iter().position(|candidate| candidate == v))
            .collect(),
        (
            crate::core::types::CoordValues::I64(step_vals),
            crate::core::types::CoordValues::U32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                u32::try_from(*v).ok().and_then(|coerced| {
                    registered_vals
                        .iter()
                        .position(|candidate| *candidate == coerced)
                })
            })
            .collect(),
        (
            crate::core::types::CoordValues::U32(step_vals),
            crate::core::types::CoordValues::I64(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                let coerced = i64::from(*v);
                registered_vals
                    .iter()
                    .position(|candidate| *candidate == coerced)
            })
            .collect(),
        (
            crate::core::types::CoordValues::U64(step_vals),
            crate::core::types::CoordValues::U32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                u32::try_from(*v).ok().and_then(|coerced| {
                    registered_vals
                        .iter()
                        .position(|candidate| *candidate == coerced)
                })
            })
            .collect(),
        (
            crate::core::types::CoordValues::U32(step_vals),
            crate::core::types::CoordValues::U64(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| {
                let coerced = u64::from(*v);
                registered_vals
                    .iter()
                    .position(|candidate| *candidate == coerced)
            })
            .collect(),
        (
            crate::core::types::CoordValues::F32(step_vals),
            crate::core::types::CoordValues::F32(registered_vals),
        ) => step_vals
            .iter()
            .map(|v| registered_vals.iter().position(|candidate| candidate == v))
            .collect(),
        (
            crate::core::types::CoordValues::Utf8(step_vals),
            crate::core::types::CoordValues::Utf8(registered_vals),
        ) => match resolver {
            AxisResolver::HashMapResolver => {
                let index_map: HashMap<&str, usize> = registered_vals
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| (value.as_str(), idx))
                    .collect();
                step_vals
                    .iter()
                    .map(|v| index_map.get(v.as_str()).copied())
                    .collect()
            }
            AxisResolver::Affine { .. } | AxisResolver::SortedBinarySearch => {
                if sorted_utf8(registered_vals) {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.binary_search(v).ok())
                        .collect()
                } else {
                    step_vals
                        .iter()
                        .map(|v| registered_vals.iter().position(|candidate| candidate == v))
                        .collect()
                }
            }
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::core::contracts::ChunkPlanner;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        AxisResolver, CoordMap, CoordValues, InferenceWriteRequest, InputArray, InputArraySource,
        PlanTemplateKey, WriteExecutionConfig,
    };

    use super::MixedRadixChunkPlanner;

    fn host_input(nbytes: usize) -> InputArray {
        InputArray {
            nbytes,
            source: InputArraySource::HostBytes(vec![0_u8; nbytes].into()),
        }
    }

    fn host_input_with_payload(payload: Vec<u8>) -> InputArray {
        InputArray {
            nbytes: payload.len(),
            source: InputArraySource::HostBytes(payload.into()),
        }
    }

    fn registered_coords() -> CoordMap {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![0, 1, 2]));
        let _ = coords.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = coords.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1]));
        coords
    }

    fn registered_coords_with_latitude() -> CoordMap {
        let mut coords = registered_coords();
        let _ = coords.insert(
            "latitude".to_string(),
            CoordValues::F64(vec![10.0, 20.0, 30.0]),
        );
        coords
    }

    #[test]
    fn computes_mixed_radix_linear_index_from_default_parallel_coords_order() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.chunk_ids.len(), 1);
        assert_eq!(planned.chunk_ids[0].array_id(), 7);
        assert_eq!(
            planned.chunk_ids[0].linear_index(),
            10,
            "expected mixed-radix index in default parallel_coords order (time, lead_time, ensemble)"
        );
    }

    #[test]
    fn uses_provided_array_ids_for_chunk_namespace() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["b".to_string(), "a".to_string()],
            arrays: vec![host_input(4), host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[9, 4], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.chunk_ids[0].array_id(), 9);
        assert_eq!(planned.chunk_ids[1].array_id(), 4);
    }

    #[test]
    fn rejects_unknown_explicit_parallel_coord() {
        let mut config = WriteExecutionConfig::default();
        let mut explicit = CoordMap::new();
        let _ = explicit.insert("invalid_dim".to_string(), CoordValues::I64(vec![1]));
        config.parallel_coords_config.parallel_coords = Some(explicit);
        let planner = MixedRadixChunkPlanner::new(config);
        let req = InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords())
            .expect_err("unknown explicit parallel coord should fail");
        assert!(matches!(
            err,
            SyncWriteError::UnknownParallelCoord { coord } if coord == "invalid_dim"
        ));
    }

    #[test]
    fn rejects_missing_active_parallel_coord_in_step_coords() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![1]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        // ensemble intentionally omitted
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords())
            .expect_err("missing active parallel coord should fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("missing active parallel coord")
        ));
    }

    #[test]
    fn expands_subset_values_to_multiple_chunk_tasks_via_cartesian_product() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.chunk_ids.len(), 4);
        let observed: Vec<u64> = planned
            .chunk_ids
            .iter()
            .map(|chunk_id| chunk_id.linear_index())
            .collect();
        assert_eq!(
            observed,
            vec![3, 2, 11, 10],
            "expected deterministic cartesian expansion order in default active dim order"
        );
    }

    #[test]
    fn partitions_required_bytes_and_host_payload_for_multi_chunk_subset_write() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input_with_payload((0_u8..16).collect())],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.tasks.len(), 4);
        for (task_idx, task) in planned.tasks.iter().enumerate() {
            assert_eq!(task.required_bytes, 4, "task {task_idx} required_bytes");
            assert_eq!(task.input.nbytes, 4, "task {task_idx} input.nbytes");
            let expected_start = task_idx * 4;
            let expected: Vec<u8> = (expected_start..expected_start + 4)
                .map(|value| value as u8)
                .collect();
            let observed = if let Some(ptr) = task.input.source.as_host_buffer_ptr() {
                let ptr_usize = usize::try_from(ptr).expect("host pointer should fit usize");
                // SAFETY:
                // - Planner task source pointer is derived from request-owned payload bytes.
                // - Request payload stays alive for this assertion scope.
                unsafe { std::slice::from_raw_parts(ptr_usize as *const u8, 4) }.to_vec()
            } else {
                match &task.input.source {
                    InputArraySource::HostBytes(bytes) => bytes.to_vec(),
                    source => panic!("expected host-backed task input, got {source:?}"),
                }
            };
            assert_eq!(
                observed, expected,
                "task {task_idx} should receive its own contiguous payload slice"
            );
        }
    }

    #[test]
    fn host_bytes_inputs_plan_as_internal_host_ptr_slices_without_owned_payload_clones() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let payload: Vec<u8> = (0_u8..16).collect();
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input_with_payload(payload)],
        };

        let base_ptr = match &req.arrays[0].source {
            InputArraySource::HostBytes(bytes) => bytes.as_ptr() as usize,
            source => panic!("expected HostBytes request source, got {source:?}"),
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.tasks.len(), 4);
        for (task_idx, task) in planned.tasks.iter().enumerate() {
            let ptr = task
                .input
                .source
                .as_host_buffer_ptr()
                .expect("task source should be lowered to internal host pointer");
            let ptr_usize = usize::try_from(ptr).expect("host pointer should fit usize");
            let expected_ptr = base_ptr
                .checked_add(task_idx * 4)
                .expect("expected pointer offset should not overflow");
            assert_eq!(
                ptr_usize, expected_ptr,
                "task {task_idx} host pointer should advance by task byte span"
            );

            // SAFETY:
            // - `ptr_usize` points into `req` payload storage, which remains alive for
            //   this whole test scope.
            // - Each task requires exactly four bytes in this setup.
            let observed = unsafe { std::slice::from_raw_parts(ptr_usize as *const u8, 4) };
            let expected: Vec<u8> = (task_idx * 4..task_idx * 4 + 4)
                .map(|value| value as u8)
                .collect();
            assert_eq!(
                observed,
                expected.as_slice(),
                "task {task_idx} should reference expected payload subrange"
            );
        }
    }

    #[test]
    fn offsets_cuda_device_ptr_per_derived_chunk_task() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![InputArray {
                nbytes: 16,
                source: InputArraySource::CudaDevicePtr {
                    ptr: 1024,
                    device_ordinal: 2,
                    producer_stream: Some(77),
                },
            }],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.tasks.len(), 4);
        for (task_idx, task) in planned.tasks.iter().enumerate() {
            assert_eq!(task.required_bytes, 4, "task {task_idx} required_bytes");
            match &task.input.source {
                InputArraySource::CudaDevicePtr {
                    ptr,
                    device_ordinal,
                    producer_stream,
                } => {
                    assert_eq!(*ptr, 1024 + (task_idx as u64 * 4));
                    assert_eq!(*device_ordinal, 2);
                    assert_eq!(*producer_stream, Some(77));
                }
                source => panic!("expected CudaDevicePtr slice for task input, got {source:?}"),
            }
        }
    }

    #[test]
    fn offsets_host_buffer_ptr_per_derived_chunk_task() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![InputArray {
                nbytes: 16,
                source: unsafe { InputArraySource::from_host_buffer_ptr(2048) },
            }],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.tasks.len(), 4);
        for (task_idx, task) in planned.tasks.iter().enumerate() {
            assert_eq!(task.required_bytes, 4, "task {task_idx} required_bytes");
            assert_eq!(
                task.input.source.as_host_buffer_ptr(),
                Some(2048 + (task_idx as u64 * 4)),
                "expected host pointer source for task {task_idx}"
            );
        }
    }

    #[test]
    fn rejects_non_divisible_input_nbytes_for_multi_chunk_subset_write() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(10)],
        };

        let err = planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect_err("non-divisible nbytes should fail planning");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("must be divisible by derived task count")
        ));
    }

    #[test]
    fn expands_subset_values_for_each_input_array_namespace() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![2, 0]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["b".to_string(), "a".to_string()],
            arrays: vec![host_input(4), host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[9, 4], &registered_coords())
            .expect("planning should succeed");
        assert_eq!(planned.tasks.len(), 4);
        assert_eq!(planned.chunk_ids[0].array_id(), 9);
        assert_eq!(planned.chunk_ids[1].array_id(), 9);
        assert_eq!(planned.chunk_ids[2].array_id(), 4);
        assert_eq!(planned.chunk_ids[3].array_id(), 4);
    }

    #[test]
    fn rejects_any_unregistered_step_value_in_active_parallel_coord() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 99]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords())
            .expect_err("unregistered subset value should fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("step value")
        ));
    }

    #[test]
    fn rejects_unknown_step_coord_key_not_present_in_registered_contract() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![1]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let _ = step_coords.insert("unknown_dim".to_string(), CoordValues::I64(vec![123]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords())
            .expect_err("unknown step coord key should fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message } if message.contains("not registered")
        ));
    }

    #[test]
    fn rejects_non_parallel_coord_when_not_full_registered_match() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![1]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let _ = step_coords.insert("latitude".to_string(), CoordValues::F64(vec![10.0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords_with_latitude())
            .expect_err("non-parallel coords must match full registered axis");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("non-parallel coord")
        ));
    }

    #[test]
    fn accepts_non_parallel_coord_when_full_registered_match() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![1]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        let _ = step_coords.insert(
            "latitude".to_string(),
            CoordValues::F64(vec![10.0, 20.0, 30.0]),
        );
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered_coords_with_latitude())
            .expect("full non-parallel coord match should succeed");
        assert_eq!(planned.chunk_ids.len(), 1);
    }

    #[test]
    fn rejects_missing_non_parallel_coord_from_step_coords() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![1]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
        // latitude (non-parallel) intentionally omitted
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let err = planner
            .plan_batch(&req, &[0], &registered_coords_with_latitude())
            .expect_err("missing non-parallel coord should fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("missing non-parallel coord")
        ));
    }

    #[test]
    fn caches_plan_template_with_task_count_and_mixed_radix_strides() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut step_coords = CoordMap::new();
        let _ = step_coords.insert("time".to_string(), CoordValues::I64(vec![0, 2]));
        let _ = step_coords.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step_coords.insert("ensemble".to_string(), CoordValues::U64(vec![1, 0]));
        let req = InferenceWriteRequest {
            coords: step_coords,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        planner
            .plan_batch(&req, &[7], &registered_coords())
            .expect("planning should succeed");

        let templates = planner
            .caches
            .plan_templates
            .read()
            .expect("template cache lock should not be poisoned");
        let (key, template) = templates
            .iter()
            .next()
            .expect("one template entry expected");
        assert_eq!(key.array_id(), 7);
        assert_eq!(
            key.active_parallel_dims(),
            &[
                "time".to_string(),
                "lead_time".to_string(),
                "ensemble".to_string()
            ]
        );
        assert_eq!(key.subset_lengths(), &[2_usize, 1, 2]);
        assert_eq!(template.task_count, 4);
        assert_eq!(template.dim_order, key.active_parallel_dims());
        assert_eq!(
            template.radix_strides,
            vec![4, 2, 1],
            "radix strides should match registered active-axis lengths [3,2,2]"
        );
    }

    #[test]
    fn plan_template_key_new_succeeds_with_matching_lengths() {
        let key = PlanTemplateKey::new(1, vec!["time".into(), "ensemble".into()], vec![10, 5])
            .expect("matching lengths should succeed");
        assert_eq!(key.array_id(), 1);
        assert_eq!(key.active_parallel_dims(), &["time", "ensemble"]);
        assert_eq!(key.subset_lengths(), &[10, 5]);
    }

    #[test]
    fn plan_template_key_new_rejects_mismatched_lengths() {
        let err = PlanTemplateKey::new(1, vec!["time".into()], vec![10, 5])
            .expect_err("mismatched lengths should fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
                if message.contains("length 1 != subset_lengths length 2")
        ));
    }

    #[test]
    fn plan_template_key_new_accepts_empty_dims_and_lengths() {
        let key = PlanTemplateKey::new(0, vec![], vec![]).expect("both-empty should succeed");
        assert!(key.active_parallel_dims().is_empty());
    }

    #[test]
    fn plan_template_key_is_hashable_after_validated_construction() {
        use std::collections::HashSet;
        let key = PlanTemplateKey::new(1, vec!["time".into()], vec![2]).expect("valid key");
        let mut set = HashSet::new();
        set.insert(key.clone());
        assert!(set.contains(&key));
    }

    #[test]
    fn plan_template_cache_keys_are_partitioned_by_subset_lengths() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let req_a = InferenceWriteRequest {
            coords: {
                let mut c = CoordMap::new();
                let _ = c.insert("time".to_string(), CoordValues::I64(vec![1]));
                let _ = c.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
                let _ = c.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
                c
            },
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };
        planner
            .plan_batch(&req_a, &[7], &registered_coords())
            .expect("first plan should succeed");

        let req_b = InferenceWriteRequest {
            coords: {
                let mut c = CoordMap::new();
                let _ = c.insert("time".to_string(), CoordValues::I64(vec![1, 2]));
                let _ = c.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
                let _ = c.insert("ensemble".to_string(), CoordValues::U64(vec![0]));
                c
            },
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };
        planner
            .plan_batch(&req_b, &[7], &registered_coords())
            .expect("second plan should succeed");

        let templates = planner
            .caches
            .plan_templates
            .read()
            .expect("template cache lock should not be poisoned");
        assert_eq!(
            templates.len(),
            2,
            "same array_id with different subset lengths should produce distinct cache keys"
        );
    }

    #[test]
    fn caches_affine_resolver_for_regular_i64_axis_when_prefer_affine_enabled() {
        let mut config = WriteExecutionConfig::default();
        config.planner_caches.prefer_affine_resolver = true;
        let planner = MixedRadixChunkPlanner::new(config);
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 2, 4, 6]));
        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![4]));
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        planner
            .plan_batch(&req, &[11], &registered)
            .expect("planning should succeed");
        let resolvers = planner
            .caches
            .axis_resolvers
            .read()
            .expect("resolver cache lock should not be poisoned");
        let resolver = resolvers
            .get(&(11, "time".to_string()))
            .expect("time resolver should be cached");
        assert!(matches!(
            resolver,
            AxisResolver::Affine {
                start_ns: 0,
                step_ns: 2,
                len: 4
            }
        ));
    }

    #[test]
    fn caches_hashmap_resolver_for_utf8_axis() {
        let mut config = WriteExecutionConfig::default();
        let mut explicit = CoordMap::new();
        let _ = explicit.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["member".to_string()]),
        );
        config.parallel_coords_config.parallel_coords = Some(explicit);
        let planner = MixedRadixChunkPlanner::new(config);

        let mut registered = CoordMap::new();
        let _ = registered.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        );
        let mut step = CoordMap::new();
        let _ = step.insert(
            "member".to_string(),
            CoordValues::Utf8(vec!["b".to_string()]),
        );
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        planner
            .plan_batch(&req, &[12], &registered)
            .expect("planning should succeed");
        let resolvers = planner
            .caches
            .axis_resolvers
            .read()
            .expect("resolver cache lock should not be poisoned");
        let resolver = resolvers
            .get(&(12, "member".to_string()))
            .expect("member resolver should be cached");
        assert!(matches!(resolver, AxisResolver::HashMapResolver));
    }

    #[test]
    fn caches_sorted_binary_search_for_irregular_i64_axis() {
        let mut config = WriteExecutionConfig::default();
        config.planner_caches.prefer_affine_resolver = true;
        let planner = MixedRadixChunkPlanner::new(config);
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 3, 4]));
        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![4]));
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        planner
            .plan_batch(&req, &[13], &registered)
            .expect("planning should succeed");
        let resolvers = planner
            .caches
            .axis_resolvers
            .read()
            .expect("resolver cache lock should not be poisoned");
        let resolver = resolvers
            .get(&(13, "time".to_string()))
            .expect("time resolver should be cached");
        assert!(matches!(resolver, AxisResolver::SortedBinarySearch));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Earth2Studio Workflow-Realistic Chunk Planner Tests
    //
    // Each test simulates a real Earth2Studio model workflow by:
    //   1. Setting up realistic `registered_coords` and `parallel_coords`
    //   2. Issuing the exact sequence of `plan_batch()` calls a real
    //      inference loop would produce
    //   3. Asserting every `ChunkId.linear_index` matches the Python
    //      `async_zarr` meshgrid-based chunk indexing
    //
    // The mixed-radix linear index for parallel dims (d0, d1, ..., dN)
    // with axis lengths (L0, L1, ..., LN) is:
    //   linear = d0_idx * (L1*L2*...*LN) + d1_idx * (L2*...*LN) + ... + dN_idx
    // which matches Python's C-order np.meshgrid(..., indexing="ij") flattening.
    // ══════════════════════════════════════════════════════════════════════

    use crate::core::types::ParallelCoordsConfig;

    /// Nanosecond constants for realistic time coordinates.
    const NS_PER_HOUR: i64 = 3_600_000_000_000;

    /// Helper: build a `WriteExecutionConfig` with explicit parallel_coords.
    fn config_with_explicit_parallel_coords(parallel_coords: CoordMap) -> WriteExecutionConfig {
        WriteExecutionConfig {
            parallel_coords_config: ParallelCoordsConfig {
                parallel_coords: Some(parallel_coords),
                default_parallel_coord_names: vec![
                    "time".to_string(),
                    "lead_time".to_string(),
                    "ensemble".to_string(),
                ],
            },
            ..Default::default()
        }
    }

    #[test]
    fn active_f32_parallel_coord_accepts_f64_step_values() {
        let levels = vec![0.0_f32, 0.25, 0.5];
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("level".to_string(), CoordValues::F32(levels.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert("level".to_string(), CoordValues::F32(levels));
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = step.insert("level".to_string(), CoordValues::F64(vec![0.25]));
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("f64 step values should resolve against f32 registered coords");

        assert_eq!(planned.tasks.len(), 1);
        assert_eq!(planned.tasks[0].tuple_key.indices(), &[0, 1]);
        assert_eq!(planned.tasks[0].chunk_id.linear_index(), 1);
    }

    #[test]
    fn active_i32_parallel_coord_accepts_i64_step_values() {
        let levels = vec![0_i32, 1, 2];
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("level".to_string(), CoordValues::I32(levels.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert("level".to_string(), CoordValues::I32(levels));
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = step.insert("level".to_string(), CoordValues::I64(vec![2]));
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("i64 step values should resolve against i32 registered coords");

        assert_eq!(planned.tasks[0].tuple_key.indices(), &[0, 2]);
        assert_eq!(planned.tasks[0].chunk_id.linear_index(), 2);
    }

    #[test]
    fn non_parallel_narrow_numeric_coords_accept_default_width_step_values() {
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lat".to_string(), CoordValues::F32(vec![10.0, 20.0]));
        let _ = registered.insert("level".to_string(), CoordValues::I32(vec![1, 2]));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = step.insert("lat".to_string(), CoordValues::F64(vec![10.0, 20.0]));
        let _ = step.insert("level".to_string(), CoordValues::I64(vec![1, 2]));
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("default-width step coords should match narrow registered coords");

        assert_eq!(planned.tasks.len(), 1);
        assert_eq!(planned.tasks[0].tuple_key.indices(), &[0, 0, 0]);
    }

    /// Helper: build step coords for a deterministic workflow write.
    /// Includes all non-parallel coords at their full registered values.
    fn deterministic_step_coords(
        time_val: i64,
        lead_time_val: i64,
        lat: &[f64],
        lon: &[f64],
    ) -> CoordMap {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![time_val]));
        let _ = coords.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![lead_time_val]),
        );
        let _ = coords.insert("lat".to_string(), CoordValues::F64(lat.to_vec()));
        let _ = coords.insert("lon".to_string(), CoordValues::F64(lon.to_vec()));
        coords
    }

    /// Helper: build step coords for an ensemble workflow write.
    fn ensemble_step_coords(
        time_val: i64,
        lead_time_val: i64,
        ensemble_val: u64,
        lat: &[f64],
        lon: &[f64],
    ) -> CoordMap {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![time_val]));
        let _ = coords.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![lead_time_val]),
        );
        let _ = coords.insert("ensemble".to_string(), CoordValues::U64(vec![ensemble_val]));
        let _ = coords.insert("lat".to_string(), CoordValues::F64(lat.to_vec()));
        let _ = coords.insert("lon".to_string(), CoordValues::F64(lon.to_vec()));
        coords
    }

    /// Helper: build step coords for a batched ensemble write (multiple ensemble values).
    fn batched_ensemble_step_coords(
        time_val: i64,
        lead_time_val: i64,
        ensemble_vals: &[u64],
        lat: &[f64],
        lon: &[f64],
    ) -> CoordMap {
        let mut coords = CoordMap::new();
        let _ = coords.insert("time".to_string(), CoordValues::I64(vec![time_val]));
        let _ = coords.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![lead_time_val]),
        );
        let _ = coords.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_vals.to_vec()),
        );
        let _ = coords.insert("lat".to_string(), CoordValues::F64(lat.to_vec()));
        let _ = coords.insert("lon".to_string(), CoordValues::F64(lon.to_vec()));
        coords
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 1: Deterministic Workflow (FourCastNet / Pangu)
    //
    // parallel_coords = {time, lead_time}  (NO ensemble)
    // time:      [0]                        → 1 init
    // lead_time: [0h, 6h, 12h, 18h, 24h]   → 5 steps
    // lat:       [0, 1, 2, 3]  (non-parallel)
    // lon:       [0, 1, 2, 3]  (non-parallel)
    // variables: [t2m, u10m, z500]
    //
    // active_parallel_dims (canonicalized): ["time", "lead_time"]
    // axis_lengths: [1, 5]
    //
    // Write pattern: 5 sequential writes, each with 1 (time, lead_time) pair.
    // Expected linear indices:
    //   (time=0, lt=0h)  → 0*5 + 0 = 0
    //   (time=0, lt=6h)  → 0*5 + 1 = 1
    //   (time=0, lt=12h) → 0*5 + 2 = 2
    //   (time=0, lt=18h) → 0*5 + 3 = 3
    //   (time=0, lt=24h) → 0*5 + 4 = 4
    //
    // Total chunks: 5 per variable × 3 variables = 15.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_deterministic_single_init_5_steps_3_variables() {
        let lead_times_ns: Vec<i64> = (0..5).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];
        let variables = ["t2m", "u10m", "z500"];
        let array_ids: Vec<u32> = (0..variables.len() as u32).collect();

        // Build registered coords (full coordinate contract from add_array)
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        // Explicit parallel_coords: only time and lead_time
        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Simulate 5 writes (one per lead_time step)
        let mut all_linear_indices: Vec<Vec<u64>> = Vec::new();
        for lt in &lead_times_ns {
            let step = deterministic_step_coords(0, *lt, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: variables.iter().map(|s| s.to_string()).collect(),
                arrays: variables.iter().map(|_| host_input(64)).collect(),
            };
            let planned = planner
                .plan_batch(&req, &array_ids, &registered)
                .expect("plan_batch should succeed");

            // 3 variables × 1 combo = 3 tasks per write
            assert_eq!(
                planned.tasks.len(),
                3,
                "each write step should produce 1 task per variable"
            );

            // All 3 tasks in same write share the same linear_index
            let linears: Vec<u64> = planned.chunk_ids.iter().map(|c| c.linear_index()).collect();
            assert_eq!(linears[0], linears[1]);
            assert_eq!(linears[1], linears[2]);
            all_linear_indices.push(linears);
        }

        // Verify linear indices match Python's meshgrid order
        let expected_linears: Vec<u64> = vec![0, 1, 2, 3, 4];
        for (step_idx, (observed, expected)) in all_linear_indices
            .iter()
            .zip(expected_linears.iter())
            .enumerate()
        {
            assert_eq!(
                observed[0], *expected,
                "step {step_idx}: linear_index mismatch"
            );
        }

        // Verify array_id namespacing is correct per variable
        let step = deterministic_step_coords(0, 0, &lat, &lon);
        let req = InferenceWriteRequest {
            coords: step,
            array_names: variables.iter().map(|s| s.to_string()).collect(),
            arrays: variables.iter().map(|_| host_input(64)).collect(),
        };
        let planned = planner
            .plan_batch(&req, &array_ids, &registered)
            .expect("plan_batch should succeed");
        for (i, task) in planned.tasks.iter().enumerate() {
            assert_eq!(
                task.chunk_id.array_id(),
                array_ids[i],
                "task {i} should have array_id={}",
                array_ids[i],
            );
            assert_eq!(task.array_name, variables[i]);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 2: Ensemble Workflow (FuXi-Ensemble / SEEDS)
    //
    // parallel_coords = {time, lead_time, ensemble}
    // time:      [0]           → 1 init
    // lead_time: [0h, 6h, 12h] → 3 steps
    // ensemble:  [0, 1, 2, 3]  → 4 members
    // lat:       [0, 1, 2, 3]  (non-parallel)
    // lon:       [0, 1, 2, 3]  (non-parallel)
    // variables: [t2m, z500]
    //
    // active_parallel_dims: ["time", "lead_time", "ensemble"]
    // axis_lengths: [1, 3, 4]
    //
    // Write pattern: 12 sequential writes (1 per combo).
    // Linear index = time_idx * 12 + lt_idx * 4 + ens_idx
    //   (0, 0h, 0) → 0    (0, 0h, 1) → 1    (0, 0h, 2) → 2    (0, 0h, 3) → 3
    //   (0, 6h, 0) → 4    (0, 6h, 1) → 5    (0, 6h, 2) → 6    (0, 6h, 3) → 7
    //   (0, 12h,0) → 8    (0, 12h,1) → 9    (0, 12h,2) → 10   (0, 12h,3) → 11
    //
    // Total chunks: 12 per variable × 2 variables = 24.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_ensemble_1_init_3_steps_4_members_2_variables() {
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let ensemble_ids: Vec<u64> = (0..4).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];
        let variables = ["t2m", "z500"];
        let array_ids: Vec<u32> = (0..variables.len() as u32).collect();

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Expected: linear = time_idx * (3*4) + lt_idx * 4 + ens_idx
        let mut observed_linears: Vec<u64> = Vec::new();
        let mut expected_linears: Vec<u64> = Vec::new();

        for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
            for (ens_idx, ens) in ensemble_ids.iter().enumerate() {
                let step = ensemble_step_coords(0, *lt, *ens, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: variables.iter().map(|s| s.to_string()).collect(),
                    arrays: variables.iter().map(|_| host_input(64)).collect(),
                };
                let planned = planner
                    .plan_batch(&req, &array_ids, &registered)
                    .expect("plan_batch should succeed");

                assert_eq!(
                    planned.tasks.len(),
                    2,
                    "each write should produce 1 task per variable"
                );

                let linear = planned.chunk_ids[0].linear_index();
                observed_linears.push(linear);
                // Both variables should share the same linear_index
                assert_eq!(
                    planned.chunk_ids[1].linear_index(),
                    linear,
                    "both variables should share linear_index"
                );

                let time_idx = 0;
                let expected = (time_idx * 3 * 4 + lt_idx * 4 + ens_idx) as u64;
                expected_linears.push(expected);
            }
        }

        assert_eq!(
            observed_linears, expected_linears,
            "all 12 linear indices should match Python async_zarr meshgrid order"
        );
        assert_eq!(observed_linears.len(), 12);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 3: CorrDiff Workflow (Regional Diffusion)
    //
    // parallel_coords = {time, lead_time, ensemble}
    // time:      [0]                          → 1 init
    // lead_time: [0]                          → 1 step
    // ensemble:  [0, 1, 2, 3, 4, 5, 6, 7]    → 8 members
    // lat:       [0, 1, 2, 3]  (non-parallel, regional)
    // lon:       [0, 1, 2, 3]  (non-parallel, regional)
    // variables: [t2m, u10m, v10m]
    //
    // active_parallel_dims: ["time", "lead_time", "ensemble"]
    // axis_lengths: [1, 1, 8]
    //
    // Write pattern: 2 BATCHED writes, each with 4 ensemble members.
    //   Batch 1: ensemble=[0,1,2,3] → Cartesian product 1×1×4=4 combos
    //   Batch 2: ensemble=[4,5,6,7] → 4 combos
    //
    // Linear index = time_idx * (1*8) + lt_idx * 8 + ens_idx
    //   Batch 1: [0, 1, 2, 3]
    //   Batch 2: [4, 5, 6, 7]
    //
    // Total chunks: 8 per variable × 3 variables = 24.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_corrdiff_batched_ensemble_writes() {
        let ensemble_ids: Vec<u64> = (0..8).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];
        let variables = ["t2m", "u10m", "v10m"];
        let array_ids: Vec<u32> = (0..variables.len() as u32).collect();

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Batch 1: ensemble [0,1,2,3]
        let step1 = batched_ensemble_step_coords(0, 0, &[0, 1, 2, 3], &lat, &lon);
        let req1 = InferenceWriteRequest {
            coords: step1,
            array_names: variables.iter().map(|s| s.to_string()).collect(),
            arrays: variables.iter().map(|_| host_input(64)).collect(),
        };
        let planned1 = planner
            .plan_batch(&req1, &array_ids, &registered)
            .expect("batch 1 should succeed");

        // 3 variables × 4 ensemble combos = 12 tasks
        assert_eq!(
            planned1.tasks.len(),
            12,
            "batch 1: 3 variables × 4 ensemble combos = 12 tasks"
        );

        // Extract linear indices grouped by variable
        // Tasks are ordered: all combos for var[0], then all combos for var[1], etc.
        let batch1_linears: Vec<u64> = planned1
            .chunk_ids
            .iter()
            .map(|c| c.linear_index())
            .collect();
        // var[0]: combos [0,1,2,3], var[1]: [0,1,2,3], var[2]: [0,1,2,3]
        assert_eq!(
            batch1_linears,
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "batch 1 linear indices (per-variable then per-combo)"
        );

        // Verify array_id namespacing
        for (i, task) in planned1.tasks.iter().enumerate() {
            let var_idx = i / 4;
            assert_eq!(task.chunk_id.array_id(), array_ids[var_idx]);
            assert_eq!(task.array_name, variables[var_idx]);
        }

        // Batch 2: ensemble [4,5,6,7]
        let step2 = batched_ensemble_step_coords(0, 0, &[4, 5, 6, 7], &lat, &lon);
        let req2 = InferenceWriteRequest {
            coords: step2,
            array_names: variables.iter().map(|s| s.to_string()).collect(),
            arrays: variables.iter().map(|_| host_input(64)).collect(),
        };
        let planned2 = planner
            .plan_batch(&req2, &array_ids, &registered)
            .expect("batch 2 should succeed");

        assert_eq!(planned2.tasks.len(), 12);
        let batch2_linears: Vec<u64> = planned2
            .chunk_ids
            .iter()
            .map(|c| c.linear_index())
            .collect();
        assert_eq!(
            batch2_linears,
            vec![4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
            "batch 2 linear indices"
        );

        // Combined: all 8 unique linear indices per variable across both batches
        let all_unique_per_var: Vec<u64> = (0..8).collect();
        let mut var0_linears: Vec<u64> = batch1_linears[0..4].to_vec();
        var0_linears.extend_from_slice(&batch2_linears[0..4]);
        assert_eq!(
            var0_linears, all_unique_per_var,
            "var 0 should cover all 8 ensemble chunk positions"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 4: Multi-Init Deterministic (Lagged Ensemble)
    //
    // parallel_coords = {time, lead_time}
    // time:      [0, 12h]         → 2 init times
    // lead_time: [0h, 6h, 12h]    → 3 steps
    // lat:       [0, 1, 2, 3]     (non-parallel)
    // lon:       [0, 1, 2, 3]     (non-parallel)
    // variables: [t2m]
    //
    // active_parallel_dims: ["time", "lead_time"]
    // axis_lengths: [2, 3]
    //
    // Write pattern: 6 writes, one per (time, lead_time).
    // Linear index = time_idx * 3 + lt_idx
    //   (0h,  0h)  → 0*3+0 = 0
    //   (0h,  6h)  → 0*3+1 = 1
    //   (0h,  12h) → 0*3+2 = 2
    //   (12h, 0h)  → 1*3+0 = 3
    //   (12h, 6h)  → 1*3+1 = 4
    //   (12h, 12h) → 1*3+2 = 5
    //
    // Total chunks: 6.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_multi_init_deterministic_2_inits_3_steps() {
        let time_ns = vec![0_i64, 12 * NS_PER_HOUR];
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut observed: Vec<u64> = Vec::new();
        let mut expected: Vec<u64> = Vec::new();

        for (t_idx, t) in time_ns.iter().enumerate() {
            for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
                let step = deterministic_step_coords(*t, *lt, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: vec!["t2m".to_string()],
                    arrays: vec![host_input(64)],
                };
                let planned = planner
                    .plan_batch(&req, &[0], &registered)
                    .expect("plan_batch should succeed");

                assert_eq!(planned.tasks.len(), 1);
                observed.push(planned.chunk_ids[0].linear_index());
                expected.push((t_idx * 3 + lt_idx) as u64);
            }
        }

        assert_eq!(
            observed, expected,
            "6 linear indices for 2 init-times × 3 lead-time steps"
        );
        // Verify unique chunk positions
        let unique_count = {
            let mut s = observed.clone();
            s.sort();
            s.dedup();
            s.len()
        };
        assert_eq!(unique_count, 6, "all 6 chunk positions must be unique");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 5: Ensemble Workflow with Batched Lead-Time Steps
    //
    // Tests that writing multiple lead_time values per write (as some models
    // do for multi-step rollouts) produces the correct Cartesian expansion.
    //
    // parallel_coords = {time, lead_time, ensemble}
    // time:      [0]               → 1 init
    // lead_time: [0h, 6h, 12h, 18h] → 4 steps
    // ensemble:  [0, 1]            → 2 members
    // variables: [t2m]
    //
    // active_parallel_dims: ["time", "lead_time", "ensemble"]
    // axis_lengths: [1, 4, 2]
    //
    // Write pattern: 1 batched write with lead_time=[0h,6h], ensemble=[0,1]
    //   → 1×2×2 = 4 Cartesian combos
    //   (0, 0h, 0) → 0*8+0*2+0 = 0
    //   (0, 0h, 1) → 0*8+0*2+1 = 1
    //   (0, 6h, 0) → 0*8+1*2+0 = 2
    //   (0, 6h, 1) → 0*8+1*2+1 = 3
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_ensemble_batched_lead_time_cartesian_expansion() {
        let lead_times_ns: Vec<i64> = (0..4).map(|i| i * 6 * NS_PER_HOUR).collect();
        let ensemble_ids: Vec<u64> = vec![0, 1];
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Write with 2 lead_times and 2 ensemble members → 4 combos
        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = step.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![0, 6 * NS_PER_HOUR]),
        );
        let _ = step.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1]));
        let _ = step.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = step.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["t2m".to_string()],
            arrays: vec![host_input(64)],
        };
        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("batched plan should succeed");

        // 1 variable × (1×2×2) combos = 4 tasks
        assert_eq!(planned.tasks.len(), 4);

        let linears: Vec<u64> = planned.chunk_ids.iter().map(|c| c.linear_index()).collect();
        // Python meshgrid with indexing="ij" and dims (time, lead_time, ensemble):
        // time[0]=0, lt subset [0,1], ens subset [0,1]
        // Enumeration order (C-order): time varies slowest, ensemble varies fastest
        //   (0, 0, 0)→0  (0, 0, 1)→1  (0, 1, 0)→2  (0, 1, 1)→3
        assert_eq!(
            linears,
            vec![0, 1, 2, 3],
            "Cartesian expansion should match C-order meshgrid enumeration"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 6: Deterministic with Default Parallel Coords (No Explicit Override)
    //
    // When `parallel_coords` is None, the planner auto-detects from
    // `default_parallel_coord_names` intersected with registered coords.
    //
    // registered: {time, lead_time, lat, lon}  (no ensemble)
    // → active: ["time", "lead_time"] (ensemble not in registered, skipped)
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_deterministic_with_default_parallel_coords_autodetection() {
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        // Use default config — no explicit parallel_coords
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());

        let mut linears = Vec::new();
        for lt in &lead_times_ns {
            let step = deterministic_step_coords(0, *lt, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: vec!["t2m".to_string()],
                arrays: vec![host_input(16)],
            };
            let planned = planner
                .plan_batch(&req, &[0], &registered)
                .expect("auto-detected parallel_coords plan should succeed");
            assert_eq!(planned.tasks.len(), 1);
            linears.push(planned.chunk_ids[0].linear_index());
        }

        assert_eq!(
            linears,
            vec![0, 1, 2],
            "auto-detected active dims [time, lead_time] → linear = time_idx*3 + lt_idx"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 7: Ensemble Workflow — Non-Sequential Write Order
    //
    // Verifies that writing ensemble members or time steps out of order
    // still produces correct linear indices (the planner must resolve
    // indices by value lookup, not by call order).
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_ensemble_non_sequential_write_order() {
        let ensemble_ids: Vec<u64> = (0..4).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert("lead_time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Write in REVERSE ensemble order: 3, 2, 1, 0
        let write_order: Vec<u64> = vec![3, 2, 1, 0];
        let mut linears: Vec<(u64, u64)> = Vec::new(); // (ensemble_id, linear_index)

        for ens in &write_order {
            let step = ensemble_step_coords(0, 0, *ens, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: vec!["t2m".to_string()],
                arrays: vec![host_input(16)],
            };
            let planned = planner
                .plan_batch(&req, &[0], &registered)
                .expect("out-of-order write should succeed");
            linears.push((*ens, planned.chunk_ids[0].linear_index()));
        }

        // Linear index should always match the position in registered coords,
        // regardless of write order.
        for (ens_id, linear) in &linears {
            assert_eq!(
                *linear, *ens_id,
                "ensemble {ens_id} should map to linear_index={ens_id} regardless of write order"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Case 8: Full Write Sequence Collecting All ChunkIds
    //
    // End-to-end simulation of a complete deterministic workflow that
    // validates every ChunkId across the entire run, ensuring no
    // duplicates and full coverage of the expected chunk space.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn workflow_full_deterministic_no_duplicate_chunk_ids() {
        let time_ns = vec![0_i64, 12 * NS_PER_HOUR];
        let lead_times_ns: Vec<i64> = (0..4).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];
        let variables = ["t2m", "u10m"];
        let array_ids: Vec<u32> = vec![0, 1];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut all_chunk_ids = Vec::new();

        for t in &time_ns {
            for lt in &lead_times_ns {
                let step = deterministic_step_coords(*t, *lt, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: variables.iter().map(|s| s.to_string()).collect(),
                    arrays: variables.iter().map(|_| host_input(16)).collect(),
                };
                let planned = planner
                    .plan_batch(&req, &array_ids, &registered)
                    .expect("plan_batch should succeed");
                all_chunk_ids.extend(planned.chunk_ids);
            }
        }

        // 2 time × 4 lead_time × 2 variables = 16 total chunk IDs
        assert_eq!(all_chunk_ids.len(), 16);

        // Verify no duplicates (no-overwrite invariant)
        let unique_set: std::collections::HashSet<_> = all_chunk_ids.iter().collect();
        assert_eq!(
            unique_set.len(),
            all_chunk_ids.len(),
            "all ChunkIds must be globally unique across the full workflow"
        );

        // Verify per-variable coverage: each variable should have 8 unique linear indices [0..8)
        for &aid in &array_ids {
            let mut var_linears: Vec<u64> = all_chunk_ids
                .iter()
                .filter(|c| c.array_id() == aid)
                .map(|c| c.linear_index())
                .collect();
            var_linears.sort();
            assert_eq!(
                var_linears,
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                "array_id={aid}: should have contiguous linear indices 0..8"
            );
        }
    }

    #[test]
    fn resolver_lookup_affine_maps_i64_values_to_expected_indices() {
        let step = CoordValues::I64(vec![4, 0]);
        let registered = CoordValues::I64(vec![0, 2, 4, 6]);
        let resolver = AxisResolver::Affine {
            start_ns: 0,
            step_ns: 2,
            len: 4,
        };

        let indices = super::index_of_step_values_with_resolver(&step, &registered, &resolver)
            .expect("affine lookup should succeed");
        assert_eq!(indices, vec![2, 0]);
    }

    #[test]
    fn resolver_lookup_hashmap_maps_utf8_values_to_expected_indices() {
        let step = CoordValues::Utf8(vec!["c".to_string(), "a".to_string()]);
        let registered = CoordValues::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let resolver = AxisResolver::HashMapResolver;

        let indices = super::index_of_step_values_with_resolver(&step, &registered, &resolver)
            .expect("hashmap lookup should succeed");
        assert_eq!(indices, vec![2, 0]);
    }

    #[test]
    fn resolver_lookup_sorted_binary_search_maps_i64_values_to_expected_indices() {
        let step = CoordValues::I64(vec![4, 0]);
        let registered = CoordValues::I64(vec![0, 3, 4]);
        let resolver = AxisResolver::SortedBinarySearch;

        let indices = super::index_of_step_values_with_resolver(&step, &registered, &resolver)
            .expect("sorted lookup should succeed");
        assert_eq!(indices, vec![2, 0]);
    }

    #[test]
    fn resolver_lookup_affine_rejects_invalid_affine_params_and_values() {
        let registered = CoordValues::I64(vec![0, 2, 4]);
        let step = CoordValues::I64(vec![2]);
        let zero_step_resolver = AxisResolver::Affine {
            start_ns: 0,
            step_ns: 0,
            len: 3,
        };
        assert!(
            super::index_of_step_values_with_resolver(&step, &registered, &zero_step_resolver)
                .is_none(),
            "zero-step affine resolver must be rejected"
        );

        let len_mismatch_resolver = AxisResolver::Affine {
            start_ns: 0,
            step_ns: 2,
            len: 2,
        };
        assert!(
            super::index_of_step_values_with_resolver(&step, &registered, &len_mismatch_resolver)
                .is_none(),
            "affine resolver length mismatch must be rejected"
        );

        let affine_resolver = AxisResolver::Affine {
            start_ns: 0,
            step_ns: 2,
            len: 3,
        };
        let non_divisible_step = CoordValues::I64(vec![3]);
        assert!(
            super::index_of_step_values_with_resolver(
                &non_divisible_step,
                &registered,
                &affine_resolver
            )
            .is_none(),
            "values that do not align to affine step must be rejected"
        );

        let out_of_bounds_step = CoordValues::I64(vec![8]);
        assert!(
            super::index_of_step_values_with_resolver(
                &out_of_bounds_step,
                &registered,
                &affine_resolver
            )
            .is_none(),
            "affine indices outside axis length must be rejected"
        );
    }

    #[test]
    fn resolver_lookup_u64_and_utf8_cover_sorted_unsorted_and_hashmap_paths() {
        let u64_step = CoordValues::U64(vec![5, 1]);
        let u64_sorted = CoordValues::U64(vec![1, 3, 5]);
        let u64_indices_sorted = super::index_of_step_values_with_resolver(
            &u64_step,
            &u64_sorted,
            &AxisResolver::SortedBinarySearch,
        )
        .expect("u64 sorted binary-search lookup should succeed");
        assert_eq!(u64_indices_sorted, vec![2, 0]);

        let u64_unsorted = CoordValues::U64(vec![5, 1, 3]);
        let u64_indices_unsorted = super::index_of_step_values_with_resolver(
            &u64_step,
            &u64_unsorted,
            &AxisResolver::SortedBinarySearch,
        )
        .expect("u64 unsorted fallback lookup should succeed");
        assert_eq!(u64_indices_unsorted, vec![0, 1]);

        let u64_indices_hash = super::index_of_step_values_with_resolver(
            &u64_step,
            &u64_unsorted,
            &AxisResolver::HashMapResolver,
        )
        .expect("u64 hashmap lookup should succeed");
        assert_eq!(u64_indices_hash, vec![0, 1]);

        let utf8_step = CoordValues::Utf8(vec!["c".to_string(), "a".to_string()]);
        let utf8_sorted =
            CoordValues::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let utf8_indices_sorted = super::index_of_step_values_with_resolver(
            &utf8_step,
            &utf8_sorted,
            &AxisResolver::Affine {
                start_ns: 0,
                step_ns: 1,
                len: 3,
            },
        )
        .expect("utf8 sorted lookup through non-hash branch should succeed");
        assert_eq!(utf8_indices_sorted, vec![2, 0]);

        let utf8_unsorted =
            CoordValues::Utf8(vec!["c".to_string(), "a".to_string(), "b".to_string()]);
        let utf8_indices_unsorted = super::index_of_step_values_with_resolver(
            &utf8_step,
            &utf8_unsorted,
            &AxisResolver::SortedBinarySearch,
        )
        .expect("utf8 unsorted fallback lookup should succeed");
        assert_eq!(utf8_indices_unsorted, vec![0, 1]);
    }

    #[test]
    fn resolver_lookup_returns_none_for_type_mismatch() {
        let step = CoordValues::I64(vec![1]);
        let registered = CoordValues::U64(vec![1]);
        assert!(
            super::index_of_step_values_with_resolver(
                &step,
                &registered,
                &AxisResolver::SortedBinarySearch
            )
            .is_none(),
            "mixed coordinate value types must not resolve indices"
        );
    }

    #[test]
    fn helper_required_bytes_and_task_range_reject_invalid_inputs() {
        let subset_zero_err = MixedRadixChunkPlanner::required_bytes_per_task(16, 0)
            .expect_err("zero subset-combo count must be rejected");
        assert!(matches!(
            subset_zero_err,
            SyncWriteError::Validation { ref message }
            if message.contains("subset combo count")
        ));

        let overflow_err = MixedRadixChunkPlanner::task_byte_range(usize::MAX, 2)
            .expect_err("task range overflow must be rejected");
        assert!(matches!(overflow_err, SyncWriteError::ChunkIdOverflow));
    }

    #[test]
    fn slice_input_for_task_rejects_payload_contract_violations() {
        let declared_larger_than_payload = InputArray {
            nbytes: 8,
            source: InputArraySource::HostBytes(vec![1_u8, 2, 3, 4].into()),
        };
        let err = MixedRadixChunkPlanner::slice_input_for_task(&declared_larger_than_payload, 0, 4)
            .expect_err("payload shorter than declared nbytes must fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("host payload len")
        ));

        let out_of_range_task = InputArray {
            nbytes: 4,
            source: InputArraySource::HostBytes(vec![1_u8, 2, 3, 4].into()),
        };
        let err = MixedRadixChunkPlanner::slice_input_for_task(&out_of_range_task, 1, 3)
            .expect_err("derived chunk byte range outside payload must fail");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("derived chunk byte range")
        ));
    }

    #[test]
    fn coord_map_rejects_empty_parallel_coord_values_before_planning() {
        let mut registered_empty_axis = CoordMap::new();
        let err = registered_empty_axis
            .insert("time".to_string(), CoordValues::I64(Vec::new()))
            .expect_err("empty registered coord axis must be rejected at construction");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("coordinate 'time'") && message.contains("at least one value")
        ));

        let mut step_empty = CoordMap::new();
        let err = step_empty
            .insert("time".to_string(), CoordValues::I64(Vec::new()))
            .expect_err("empty step coord axis must be rejected at construction");
        assert!(matches!(
            err,
            SyncWriteError::Validation { ref message }
            if message.contains("coordinate 'time'") && message.contains("at least one value")
        ));
    }

    #[test]
    fn plan_batch_with_empty_registered_coords_uses_single_linear_tuple_key() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let req = InferenceWriteRequest {
            coords: CoordMap::new(),
            array_names: vec!["flat".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[11], &CoordMap::new())
            .expect("flat write without coordinate contract should still produce one task");
        assert_eq!(planned.chunk_ids.len(), 1);
        assert_eq!(planned.tasks.len(), 1);
        assert_eq!(planned.chunk_ids[0].array_id(), 11);
        assert_eq!(planned.chunk_ids[0].linear_index(), 0);
        assert_eq!(
            planned.tasks[0].tuple_key.indices(),
            &[0],
            "empty dim-order path should emit a single-index tuple key"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // ChunkKeyTemplate Unit Tests
    // ══════════════════════════════════════════════════════════════════════

    use crate::core::types::{ChunkKeyTemplate, TupleChunkKey};

    #[test]
    fn chunk_key_template_build_deterministic_2_parallel_dims() {
        // registered: time(2), lead_time(5), lat(4), lon(4)
        // active parallel: [time, lead_time]
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![0, 6, 12, 18, 24]),
        );
        let _ = registered.insert(
            "lat".to_string(),
            CoordValues::F64(vec![0.0, 1.0, 2.0, 3.0]),
        );
        let _ = registered.insert(
            "lon".to_string(),
            CoordValues::F64(vec![0.0, 1.0, 2.0, 3.0]),
        );

        let active = vec!["time".to_string(), "lead_time".to_string()];
        let template = ChunkKeyTemplate::build(&registered, &active).expect("build should succeed");

        // Dim order: parallel first, then non-parallel alphabetically.
        assert_eq!(
            template.dim_order(),
            &["time", "lead_time", "lat", "lon"],
            "parallel dims first, then non-parallel alphabetically"
        );

        // axis_lengths: parallel use registered length, non-parallel always 1.
        assert_eq!(
            template.axis_lengths(),
            &[2, 5, 1, 1],
            "parallel = registered len, non-parallel = 1"
        );

        // divisors: C-order → [5*1*1, 1*1, 1, 1] = [5, 1, 1, 1].
        assert_eq!(template.divisors(), &[5, 1, 1, 1]);

        assert_eq!(template.parallel_dim_positions(), &[0, 1]);
        assert_eq!(template.fixed_nonparallel_indices(), &[(2, 0), (3, 0)]);
    }

    #[test]
    fn chunk_key_template_build_ensemble_3_parallel_dims() {
        // registered: time(1), lead_time(3), ensemble(4), lat(4), lon(4)
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = registered.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1, 2, 3]));
        let _ = registered.insert(
            "lat".to_string(),
            CoordValues::F64(vec![0.0, 1.0, 2.0, 3.0]),
        );
        let _ = registered.insert(
            "lon".to_string(),
            CoordValues::F64(vec![0.0, 1.0, 2.0, 3.0]),
        );

        let active = vec![
            "time".to_string(),
            "lead_time".to_string(),
            "ensemble".to_string(),
        ];
        let template = ChunkKeyTemplate::build(&registered, &active).expect("build should succeed");

        assert_eq!(
            template.dim_order(),
            &["time", "lead_time", "ensemble", "lat", "lon"]
        );
        assert_eq!(template.axis_lengths(), &[1, 3, 4, 1, 1]);
        // divisors: [3*4*1*1, 4*1*1, 1*1, 1, 1] = [12, 4, 1, 1, 1].
        assert_eq!(template.divisors(), &[12, 4, 1, 1, 1]);
        assert_eq!(template.parallel_dim_positions(), &[0, 1, 2]);
        assert_eq!(template.fixed_nonparallel_indices(), &[(3, 0), (4, 0)]);
    }

    #[test]
    fn chunk_key_template_resolve_tuple_key_deterministic() {
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![0, 6, 12, 18, 24]),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let active = vec!["time".to_string(), "lead_time".to_string()];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();

        // parallel indices: time=0, lead_time=2
        let tuple = template.resolve_tuple_key(&[0, 2]).unwrap();
        assert_eq!(tuple.indices(), &[0, 2, 0, 0]);
    }

    #[test]
    fn chunk_key_template_resolve_tuple_key_ensemble() {
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = registered.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1, 2, 3]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let active = vec![
            "time".to_string(),
            "lead_time".to_string(),
            "ensemble".to_string(),
        ];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();

        // parallel indices: time=0, lead_time=1, ensemble=3
        let tuple = template.resolve_tuple_key(&[0, 1, 3]).unwrap();
        assert_eq!(tuple.indices(), &[0, 1, 3, 0, 0]);
    }

    #[test]
    fn chunk_key_template_linear_roundtrip() {
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(vec![0, 6, 12, 18, 24]),
        );
        let _ = registered.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1, 2]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let active = vec![
            "time".to_string(),
            "lead_time".to_string(),
            "ensemble".to_string(),
        ];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();

        // Exhaustive test: every valid (time, lead_time, ensemble) combo
        for t in 0..2_usize {
            for lt in 0..5_usize {
                for ens in 0..3_usize {
                    let tuple = template.resolve_tuple_key(&[t, lt, ens]).unwrap();
                    let linear = template.linear_from_tuple(&tuple).unwrap();
                    let roundtrip = template.tuple_from_linear(linear).unwrap();
                    assert_eq!(
                        tuple, roundtrip,
                        "roundtrip failed for ({t},{lt},{ens}): linear={linear}"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_key_template_linear_matches_planner_mixed_radix() {
        // Verify ChunkKeyTemplate produces the same linear index as the
        // planner's mixed_radix_linear_index for all combinations.
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6, 12]));
        let _ = registered.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1, 2, 3]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0]));

        let active = vec![
            "time".to_string(),
            "lead_time".to_string(),
            "ensemble".to_string(),
        ];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());

        let axis_lengths = vec![2, 3, 4]; // same as registered parallel dim lengths

        for t in 0..2_usize {
            for lt in 0..3_usize {
                for ens in 0..4_usize {
                    let tuple = template.resolve_tuple_key(&[t, lt, ens]).unwrap();
                    let template_linear = template.linear_from_tuple(&tuple).unwrap();
                    let planner_linear = planner
                        .mixed_radix_linear_index(&[t, lt, ens], &axis_lengths)
                        .unwrap();
                    assert_eq!(
                        template_linear, planner_linear,
                        "mismatch at ({t},{lt},{ens}): template={template_linear}, planner={planner_linear}"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_key_template_cached_across_plan_batch_calls() {
        // Verify the cached template produces the same tuple keys across
        // multiple plan_batch invocations (testing the caching path).
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
        let _ = parallel.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // First call: builds and caches template.
        let step1 = deterministic_step_coords(0, 0, &[0.0, 1.0], &[0.0, 1.0]);
        let req1 = InferenceWriteRequest {
            coords: step1,
            array_names: vec!["t2m".to_string()],
            arrays: vec![host_input(16)],
        };
        let planned1 = planner
            .plan_batch(&req1, &[0], &registered)
            .expect("first call");

        // Second call: reuses cached template.
        let step2 = deterministic_step_coords(1, 6, &[0.0, 1.0], &[0.0, 1.0]);
        let req2 = InferenceWriteRequest {
            coords: step2,
            array_names: vec!["t2m".to_string()],
            arrays: vec![host_input(16)],
        };
        let planned2 = planner
            .plan_batch(&req2, &[0], &registered)
            .expect("second call");

        // Verify different tuple keys for different inputs.
        assert_eq!(planned1.tasks[0].tuple_key.indices(), &[0, 0, 0, 0]);
        assert_eq!(planned2.tasks[0].tuple_key.indices(), &[1, 1, 0, 0]);

        // Verify the template is indeed cached.
        let cached = planner.caches.chunk_key_template.read().expect("lock");
        assert!(
            cached.is_some(),
            "ChunkKeyTemplate should be cached after plan_batch"
        );
    }

    #[test]
    fn tuple_chunk_key_render_dot_separator() {
        let tuple = TupleChunkKey::new(vec![0, 4, 0, 0]);
        assert_eq!(tuple.render('.'), "0.4.0.0");
    }

    #[test]
    fn tuple_chunk_key_render_slash_separator() {
        let tuple = TupleChunkKey::new(vec![0, 4, 0, 0]);
        assert_eq!(tuple.render('/'), "0/4/0/0");
    }

    #[test]
    fn tuple_chunk_key_render_single_dim() {
        let tuple = TupleChunkKey::new(vec![7]);
        assert_eq!(tuple.render('.'), "7");
    }

    #[test]
    fn chunk_key_template_rejects_mismatched_parallel_indices_length() {
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0]));

        let active = vec!["time".to_string(), "lead_time".to_string()];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();

        // Wrong: 3 indices for 2 parallel dims
        let err = template
            .resolve_tuple_key(&[0, 1, 2])
            .expect_err("should reject mismatched length");
        assert!(matches!(err, SyncWriteError::Validation { .. }));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Integration: plan_batch produces correct TupleChunkKey on ChunkTask
    //
    // These tests verify that the planner resolves per-task TupleChunkKey
    // via the cached ChunkKeyTemplate, matching expected Zarr v2 chunk paths.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn plan_batch_produces_correct_tuple_key_deterministic_single_step() {
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = registered.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = step.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step.insert("lat".to_string(), CoordValues::F64(vec![0.0, 1.0]));
        let _ = step.insert("lon".to_string(), CoordValues::F64(vec![0.0, 1.0]));

        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["t2m".to_string()],
            arrays: vec![host_input(16)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("plan_batch should succeed");

        assert_eq!(planned.tasks.len(), 1);
        // Parallel dims: time(len=1), lead_time(len=2)
        // time_idx=0, lead_time_idx=1
        // Non-parallel: lat, lon → index 0 each
        // Expected tuple: [0, 1, 0, 0]
        assert_eq!(
            planned.tasks[0].tuple_key.indices(),
            &[0, 1, 0, 0],
            "tuple_key should be [time_idx=0, lead_time_idx=1, lat=0, lon=0]"
        );
        assert_eq!(planned.tasks[0].tuple_key.render('.'), "0.1.0.0");
    }

    #[test]
    fn plan_batch_produces_correct_tuple_key_ensemble_cartesian() {
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let ensemble_ids: Vec<u64> = vec![0, 1, 2, 3];
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Write step: time=0, lead_time=6h (idx=1), ensemble=3 (idx=3)
        let step = ensemble_step_coords(0, 6 * NS_PER_HOUR, 3, &lat, &lon);
        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["t2m".to_string()],
            arrays: vec![host_input(16)],
        };

        let planned = planner
            .plan_batch(&req, &[0], &registered)
            .expect("plan_batch should succeed");

        assert_eq!(planned.tasks.len(), 1);
        // dims: time(1), lead_time(3), ensemble(4), lat(1chunk), lon(1chunk)
        // tuple: [0, 1, 3, 0, 0]
        assert_eq!(
            planned.tasks[0].tuple_key.indices(),
            &[0, 1, 3, 0, 0],
            "tuple_key = [time=0, lead_time=1, ensemble=3, lat=0, lon=0]"
        );
        assert_eq!(planned.tasks[0].tuple_key.render('.'), "0.1.3.0.0");
        // Verify linear matches: 0*12 + 1*4 + 3*1 + 0 + 0 = 7
        assert_eq!(planned.tasks[0].chunk_id.linear_index(), 7);
    }

    #[test]
    fn plan_batch_tuple_key_linear_consistent_with_chunk_id_linear() {
        // For every task produced by plan_batch, the linear index computed
        // from tuple_key must equal chunk_id.linear_index().
        let lead_times_ns: Vec<i64> = (0..4).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert(
            "time".to_string(),
            CoordValues::I64(vec![0, 12 * NS_PER_HOUR]),
        );
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert(
            "time".to_string(),
            CoordValues::I64(vec![0, 12 * NS_PER_HOUR]),
        );
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        // Build template for verification
        let active = vec!["time".to_string(), "lead_time".to_string()];
        let template = ChunkKeyTemplate::build(&registered, &active).unwrap();

        for t in &[0_i64, 12 * NS_PER_HOUR] {
            for lt in &lead_times_ns {
                let step = deterministic_step_coords(*t, *lt, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: vec!["t2m".to_string()],
                    arrays: vec![host_input(16)],
                };
                let planned = planner
                    .plan_batch(&req, &[0], &registered)
                    .expect("plan_batch should succeed");

                for task in &planned.tasks {
                    let tuple_linear = template
                        .linear_from_tuple(&task.tuple_key)
                        .expect("linear_from_tuple should succeed");
                    assert_eq!(
                        tuple_linear,
                        task.chunk_id.linear_index(),
                        "tuple_key linear must match chunk_id.linear_index for {}",
                        task.chunk_id
                    );
                }
            }
        }
    }

    #[test]
    fn plan_batch_tuple_key_renders_zarr_v2_chunk_path() {
        // Verify that rendering tuple_key with '.' produces a valid Zarr v2 chunk path.
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());
        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1, 2]));
        let _ = registered.insert("lead_time".to_string(), CoordValues::I64(vec![0, 6]));
        let _ = registered.insert("ensemble".to_string(), CoordValues::U64(vec![0, 1]));

        // time=2(idx=2), lead_time=6(idx=1), ensemble=0(idx=0)
        let mut step = CoordMap::new();
        let _ = step.insert("time".to_string(), CoordValues::I64(vec![2]));
        let _ = step.insert("lead_time".to_string(), CoordValues::I64(vec![6]));
        let _ = step.insert("ensemble".to_string(), CoordValues::U64(vec![0]));

        let req = InferenceWriteRequest {
            coords: step,
            array_names: vec!["temperature".to_string()],
            arrays: vec![host_input(4)],
        };

        let planned = planner
            .plan_batch(&req, &[7], &registered)
            .expect("planning should succeed");

        // active_parallel_dims (default): [time, lead_time, ensemble]
        // All registered, so all active. No non-parallel dims.
        // tuple: [2, 1, 0]
        // linear = 2*4 + 1*2 + 0 = 10
        assert_eq!(planned.tasks[0].chunk_id.linear_index(), 10);
        assert_eq!(
            planned.tasks[0].tuple_key.render('.'),
            "2.1.0",
            "rendered tuple key should be Zarr v2 chunk path"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Inference-loop cache-hit tests
    //
    // Each test simulates the real `add_array() → write() × N` lifecycle
    // for a specific Earth2Studio model pattern:
    //   1. Build planner + registered coords  (mimics add_array)
    //   2. Loop over all inference steps calling plan_batch  (mimics write)
    //   3. Assert: cache_misses == 1, cache_hits == total_calls - 1
    //   4. Assert: every TupleChunkKey is correct
    //
    // This ensures the ChunkKeyTemplate is built exactly once on the
    // first plan_batch and reused for every subsequent call.
    // ══════════════════════════════════════════════════════════════════════

    /// Helper: assert cache stats after a full inference loop.
    fn assert_cache_stats(planner: &MixedRadixChunkPlanner, total_plan_calls: usize) {
        let (hits, misses) = planner.template_cache_stats();
        assert_eq!(
            misses, 1,
            "ChunkKeyTemplate should be built exactly once (got {misses} misses)"
        );
        assert_eq!(
            hits,
            total_plan_calls - 1,
            "ChunkKeyTemplate should be reused for {expected} calls (got {hits} hits)",
            expected = total_plan_calls - 1
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 1: Deterministic Single-Init (e.g., DLWP, FuXi, Pangu)
    //
    //   time:       [0]          → 1 init
    //   lead_time:  [0h, 6h, 12h, 18h, 24h]  → 5 steps
    //   lat, lon:   small grid
    //
    // Inference loop: 5 write() calls, one per lead_time step.
    // Expected: 1 build + 4 cache hits.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_deterministic_single_init_cache_hit() {
        let lead_times_ns: Vec<i64> = (0..5).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut all_tuple_keys = Vec::new();
        let total_steps = lead_times_ns.len();

        for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
            let step = deterministic_step_coords(0, *lt, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: vec!["t2m".to_string()],
                arrays: vec![host_input(64)],
            };
            let planned = planner
                .plan_batch(&req, &[0], &registered)
                .expect("plan should succeed");

            assert_eq!(planned.tasks.len(), 1);
            // time=0 → idx 0; lead_time → idx lt_idx; non-parallel → 0
            let expected_indices = vec![0, lt_idx, 0, 0];
            assert_eq!(
                planned.tasks[0].tuple_key.indices(),
                &expected_indices,
                "step {lt_idx}: tuple key mismatch"
            );
            all_tuple_keys.push(planned.tasks[0].tuple_key.render('.'));
        }

        assert_cache_stats(&planner, total_steps);

        // Verify rendered chunk paths.
        let expected_paths: Vec<String> = (0..5).map(|i| format!("0.{i}.0.0")).collect();
        assert_eq!(all_tuple_keys, expected_paths);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 2: Deterministic Multi-Init (e.g., batched GraphCast runs)
    //
    //   time:       [0h, 12h]     → 2 init times
    //   lead_time:  [0h, 6h, 12h, 18h]  → 4 steps per init
    //   lat, lon:   small grid
    //
    // Inference loop: 2×4 = 8 write() calls.
    // Expected: 1 build + 7 cache hits.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_deterministic_multi_init_cache_hit() {
        let time_ns = vec![0_i64, 12 * NS_PER_HOUR];
        let lead_times_ns: Vec<i64> = (0..4).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut all_tuple_keys = Vec::new();
        let total_steps = time_ns.len() * lead_times_ns.len(); // 8

        for (t_idx, t) in time_ns.iter().enumerate() {
            for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
                let step = deterministic_step_coords(*t, *lt, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: vec!["t2m".to_string()],
                    arrays: vec![host_input(16)],
                };
                let planned = planner
                    .plan_batch(&req, &[0], &registered)
                    .expect("plan should succeed");

                assert_eq!(planned.tasks.len(), 1);
                let expected_indices = vec![t_idx, lt_idx, 0, 0];
                assert_eq!(
                    planned.tasks[0].tuple_key.indices(),
                    &expected_indices,
                    "time={t_idx}, lt={lt_idx}: tuple key mismatch"
                );
                all_tuple_keys.push(planned.tasks[0].tuple_key.render('.'));
            }
        }

        assert_cache_stats(&planner, total_steps);

        // Spot-check: first and last paths.
        assert_eq!(all_tuple_keys[0], "0.0.0.0");
        assert_eq!(all_tuple_keys[7], "1.3.0.0");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 3: Ensemble (e.g., FourCastNet ensemble, SFNO)
    //
    //   time:       [0]
    //   lead_time:  [0h, 6h, 12h]  → 3 steps
    //   ensemble:   [0, 1, 2, 3]   → 4 members
    //   lat, lon:   small grid
    //
    // Inference loop: for each member, iterate over lead_time steps.
    // 4 members × 3 steps = 12 write() calls (one member at a time).
    // Expected: 1 build + 11 cache hits.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_ensemble_cache_hit() {
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let ensemble_ids: Vec<u64> = vec![0, 1, 2, 3];
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let mut all_tuple_keys = Vec::new();
        let total_steps = ensemble_ids.len() * lead_times_ns.len(); // 12

        for (ens_idx, ens) in ensemble_ids.iter().enumerate() {
            for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
                let step = ensemble_step_coords(0, *lt, *ens, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: vec!["t2m".to_string()],
                    arrays: vec![host_input(16)],
                };
                let planned = planner
                    .plan_batch(&req, &[0], &registered)
                    .expect("plan should succeed");

                assert_eq!(planned.tasks.len(), 1);
                // dim order: [time, lead_time, ensemble, lat, lon]
                let expected_indices = vec![0, lt_idx, ens_idx, 0, 0];
                assert_eq!(
                    planned.tasks[0].tuple_key.indices(),
                    &expected_indices,
                    "ens={ens_idx}, lt={lt_idx}: tuple key mismatch"
                );
                all_tuple_keys.push(planned.tasks[0].tuple_key.render('.'));
            }
        }

        assert_cache_stats(&planner, total_steps);

        // Spot-check.
        assert_eq!(all_tuple_keys[0], "0.0.0.0.0"); // ens=0, lt=0
        assert_eq!(all_tuple_keys[2], "0.2.0.0.0"); // ens=0, lt=2
        assert_eq!(all_tuple_keys[3], "0.0.1.0.0"); // ens=1, lt=0
        assert_eq!(all_tuple_keys[11], "0.2.3.0.0"); // ens=3, lt=2
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 4: CorrDiff — Batched Ensemble Writes
    //
    //   time:       [0]
    //   lead_time:  [0h, 6h]        → 2 steps
    //   ensemble:   [0, 1, 2, ..7]  → 8 members
    //   lat, lon:   small grid
    //
    // CorrDiff writes all ensemble members in one write() call per lead_time.
    // 2 write() calls, each with 8 ensemble members → 8 tasks each.
    // Expected: 1 build + 1 cache hit.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_corrdiff_batched_ensemble_cache_hit() {
        let lead_times_ns: Vec<i64> = vec![0, 6 * NS_PER_HOUR];
        let ensemble_ids: Vec<u64> = (0..8).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = parallel.insert(
            "ensemble".to_string(),
            CoordValues::U64(ensemble_ids.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let total_steps = lead_times_ns.len(); // 2 batched writes
        let mut all_rendered = Vec::new();

        for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
            // Each write sends ALL ensemble members at once.
            let step = batched_ensemble_step_coords(0, *lt, &ensemble_ids, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: vec!["t2m".to_string()],
                arrays: vec![host_input(16)],
            };
            let planned = planner
                .plan_batch(&req, &[0], &registered)
                .expect("plan should succeed");

            // 1 time × 1 lead_time × 8 ensemble = 8 tasks per write.
            assert_eq!(
                planned.tasks.len(),
                8,
                "batched write at lt_idx={lt_idx} should produce 8 tasks"
            );

            for (task_ens_idx, task) in planned.tasks.iter().enumerate() {
                let expected_indices = vec![0, lt_idx, task_ens_idx, 0, 0];
                assert_eq!(
                    task.tuple_key.indices(),
                    &expected_indices,
                    "lt={lt_idx}, ens={task_ens_idx}: tuple key mismatch"
                );
                all_rendered.push(task.tuple_key.render('.'));
            }
        }

        assert_cache_stats(&planner, total_steps);

        // 16 total chunks across 2 writes.
        assert_eq!(all_rendered.len(), 16);
        // Spot-check first and last.
        assert_eq!(all_rendered[0], "0.0.0.0.0");
        assert_eq!(all_rendered[7], "0.0.7.0.0");
        assert_eq!(all_rendered[8], "0.1.0.0.0");
        assert_eq!(all_rendered[15], "0.1.7.0.0");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 5: Auto-Detection of Parallel Coords (no explicit override)
    //
    //   registered: {time, lead_time, lat, lon}  (no ensemble)
    //   parallel_coords: None → auto-detect ["time", "lead_time"]
    //
    //   time:       [0]
    //   lead_time:  [0h, 6h, 12h, 18h, 24h]  → 5 steps
    //
    // Inference loop: 5 write() calls.
    // Expected: 1 build + 4 cache hits.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_autodetect_parallel_coords_cache_hit() {
        let lead_times_ns: Vec<i64> = (0..5).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0, 2.0, 3.0];
        let lon = vec![0.0, 1.0, 2.0, 3.0];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0]));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        // Default config — no explicit parallel_coords.
        let planner = MixedRadixChunkPlanner::new(WriteExecutionConfig::default());

        let mut all_tuple_keys = Vec::new();
        let total_steps = lead_times_ns.len();

        for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
            let step = deterministic_step_coords(0, *lt, &lat, &lon);
            let req = InferenceWriteRequest {
                coords: step,
                array_names: vec!["t2m".to_string()],
                arrays: vec![host_input(64)],
            };
            let planned = planner
                .plan_batch(&req, &[0], &registered)
                .expect("plan should succeed");

            assert_eq!(planned.tasks.len(), 1);
            // auto-detected: [time, lead_time], non-parallel: [lat, lon]
            let expected_indices = vec![0, lt_idx, 0, 0];
            assert_eq!(
                planned.tasks[0].tuple_key.indices(),
                &expected_indices,
                "step {lt_idx}: tuple key mismatch"
            );
            all_tuple_keys.push(planned.tasks[0].tuple_key.render('.'));
        }

        assert_cache_stats(&planner, total_steps);

        let expected_paths: Vec<String> = (0..5).map(|i| format!("0.{i}.0.0")).collect();
        assert_eq!(all_tuple_keys, expected_paths);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Model 6: Multi-Variable + Multi-Step (e.g., Pangu with t2m, u10m, v10m)
    //
    //   time:       [0h, 12h]        → 2 inits
    //   lead_time:  [0h, 6h, 12h]    → 3 steps
    //   lat, lon:   small grid
    //   variables:  [t2m, u10m, v10m] → 3 vars
    //
    // Inference loop: 2×3 = 6 write() calls, each producing 3 tasks (one
    // per variable). Total plan_batch calls: 6.
    // Expected: 1 build + 5 cache hits.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn inference_loop_multi_variable_multi_step_cache_hit() {
        let time_ns = vec![0_i64, 12 * NS_PER_HOUR];
        let lead_times_ns: Vec<i64> = (0..3).map(|i| i * 6 * NS_PER_HOUR).collect();
        let lat = vec![0.0, 1.0];
        let lon = vec![0.0, 1.0];
        let variables = ["t2m", "u10m", "v10m"];
        let array_ids: Vec<u32> = vec![0, 1, 2];

        let mut registered = CoordMap::new();
        let _ = registered.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = registered.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let _ = registered.insert("lat".to_string(), CoordValues::F64(lat.clone()));
        let _ = registered.insert("lon".to_string(), CoordValues::F64(lon.clone()));

        let mut parallel = CoordMap::new();
        let _ = parallel.insert("time".to_string(), CoordValues::I64(time_ns.clone()));
        let _ = parallel.insert(
            "lead_time".to_string(),
            CoordValues::I64(lead_times_ns.clone()),
        );
        let planner = MixedRadixChunkPlanner::new(config_with_explicit_parallel_coords(parallel));

        let total_plan_calls = time_ns.len() * lead_times_ns.len(); // 6
        let mut all_tuple_keys: Vec<Vec<String>> = Vec::new(); // per-call

        for (t_idx, t) in time_ns.iter().enumerate() {
            for (lt_idx, lt) in lead_times_ns.iter().enumerate() {
                let step = deterministic_step_coords(*t, *lt, &lat, &lon);
                let req = InferenceWriteRequest {
                    coords: step,
                    array_names: variables.iter().map(|s| s.to_string()).collect(),
                    arrays: variables.iter().map(|_| host_input(16)).collect(),
                };
                let planned = planner
                    .plan_batch(&req, &array_ids, &registered)
                    .expect("plan should succeed");

                // 3 variables × 1 combo = 3 tasks per write.
                assert_eq!(planned.tasks.len(), 3);

                // All 3 tasks should share the same tuple key (same chunk position,
                // different array_ids).
                let mut call_keys = Vec::new();
                for task in &planned.tasks {
                    let expected_indices = vec![t_idx, lt_idx, 0, 0];
                    assert_eq!(
                        task.tuple_key.indices(),
                        &expected_indices,
                        "time={t_idx}, lt={lt_idx}, array={}: tuple key mismatch",
                        task.chunk_id.array_id()
                    );
                    call_keys.push(task.tuple_key.render('.'));
                }
                all_tuple_keys.push(call_keys);
            }
        }

        assert_cache_stats(&planner, total_plan_calls);

        // Total: 6 calls × 3 variables = 18 tasks.
        let total_tasks: usize = all_tuple_keys.iter().map(|v| v.len()).sum();
        assert_eq!(total_tasks, 18);

        // Spot-check: first call → all three vars get "0.0.0.0".
        assert!(
            all_tuple_keys[0].iter().all(|k| k == "0.0.0.0"),
            "all variables in first write should share tuple key 0.0.0.0"
        );
        // Last call: time=1, lt=2 → "1.2.0.0".
        assert!(
            all_tuple_keys[5].iter().all(|k| k == "1.2.0.0"),
            "all variables in last write should share tuple key 1.2.0.0"
        );
    }
}
