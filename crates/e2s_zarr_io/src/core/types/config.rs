/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Execution and format configuration types.

use crate::core::errors::SyncWriteError;

use super::requests::CoordMap;

/// Default parallel coordinate names (matched to `async_zarr.py`).
pub const DEFAULT_PARALLEL_COORD_NAMES: [&str; 3] = ["time", "lead_time", "ensemble"];

/// Default transient oversize limit (2 GiB).
pub const DEFAULT_TRANSIENT_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Default max pooled memory budget (512 MiB).
pub const DEFAULT_MAX_POOL_BYTES: usize = 512 * 1024 * 1024;

/// Default thread pool size.
pub const DEFAULT_NUM_THREADS: usize = 8;

/// Default close timeout (5 minutes).
pub const DEFAULT_CLOSE_TIMEOUT_SECONDS: f64 = 300.0;

/// Default plan template LRU capacity.
pub const DEFAULT_PLAN_TEMPLATE_LRU_CAPACITY: usize = 1024;

/// Default chunk key LRU capacity.
pub const DEFAULT_CHUNK_KEY_LRU_CAPACITY: usize = 4096;

/// Default fallback chunk size (4 MiB) when no model profile hint is provided.
pub const DEFAULT_GLOBAL_FALLBACK_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Default minimum hot slab buffer count.
pub const DEFAULT_MIN_HOT_SLAB_BUFFERS: usize = 4;

/// Default warm-to-hot slab ratio cap.
pub const DEFAULT_MAX_WARM_TO_HOT_RATIO: usize = 3;

/// Default hot slab readiness timeout (seconds).
pub const DEFAULT_HOT_SLAB_READY_TIMEOUT_SECONDS: f64 = 2.0;

/// Input stability policy during the copy-barrier window.
///
/// v1 selects `StrictGilHold` as the default and only supported runtime policy.
/// Other variants are kept for forward compatibility and deliberately marked as
/// reserved for future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputStabilityPolicy {
    /// Reserved for future use. Not supported by v1 runtime validation.
    #[deprecated(
        since = "0.1.0",
        note = "reserved for future use; only StrictGilHold is supported in v1"
    )]
    ContractOnly,
    /// Reserved for future use. Not supported by v1 runtime validation.
    #[deprecated(
        since = "0.1.0",
        note = "reserved for future use; only StrictGilHold is supported in v1"
    )]
    FreezeFenceFast,
    /// Hold GIL from descriptor extraction until copy barrier completes (v1 default).
    StrictGilHold,
}

/// Controls when `fsync` / `sync_all` is called after writing chunk and metadata files.
///
/// For inference workloads where data can be regenerated on crash, disabling
/// fsync reduces I/O latency significantly (each fsync forces a disk round-trip).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FsyncPolicy {
    /// Fsync every file and parent directory after write (crash-safe, default).
    #[default]
    Always,
    /// Skip fsync entirely — rely on OS write-back cache. Fastest, but data may
    /// be lost if the system crashes before the OS flushes dirty pages.
    Never,
}

/// Thread scheduler backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SchedulerBackend {
    /// Rayon fixed-size pool with work stealing (v1 default).
    RayonWorkStealing,
}

/// Chunk planner algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlannerAlgorithm {
    /// Mixed-radix streaming enumeration without full meshgrid allocation (v1 default).
    MixedRadixStreaming,
}

/// Zarr format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZarrFormat {
    /// Zarr v2 (default for compatibility).
    V2,
    /// Zarr v3.
    V3,
}

/// Chunk key encoding family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkKeyEncoding {
    /// v2-compatible encoding (only valid with `ZarrFormat::V2`).
    V2,
    /// v3 default encoding family (only valid with `ZarrFormat::V3`).
    Default,
}

/// Chunk key path separator character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkKeySeparator {
    /// `.` separator (v2 default).
    Dot,
    /// `/` separator.
    Slash,
}

impl ChunkKeySeparator {
    /// Return the `char` representation of this separator.
    #[must_use]
    pub const fn as_char(self) -> char {
        match self {
            Self::Dot => '.',
            Self::Slash => '/',
        }
    }
}

/// Immutable format configuration for a backend instance.
///
/// The `(format, encoding, separator)` combination is validated at initialization
/// and must not change for the backend's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZarrTargetConfig {
    /// Selected Zarr format version.
    pub zarr_format: ZarrFormat,
    /// Chunk key encoding family.
    pub chunk_key_encoding: ChunkKeyEncoding,
    /// Chunk key path separator.
    pub chunk_key_separator: ChunkKeySeparator,
}

impl Default for ZarrTargetConfig {
    fn default() -> Self {
        Self {
            zarr_format: ZarrFormat::V2,
            chunk_key_encoding: ChunkKeyEncoding::V2,
            chunk_key_separator: ChunkKeySeparator::Dot,
        }
    }
}

impl ZarrTargetConfig {
    /// Validate that the format/encoding/separator combination is supported.
    ///
    /// Returns `Err` for unsupported combinations (per spec §53).
    pub fn validate(&self) -> Result<(), SyncWriteError> {
        match (
            self.zarr_format,
            self.chunk_key_encoding,
            self.chunk_key_separator,
        ) {
            (ZarrFormat::V2, ChunkKeyEncoding::V2, _) => Ok(()),
            (ZarrFormat::V2, ChunkKeyEncoding::Default, _) => {
                Err(SyncWriteError::UnsupportedZarrTargetConfig {
                    message: "zarr_format=V2 does not support chunk_key_encoding=Default"
                        .to_string(),
                })
            }
            (ZarrFormat::V3, ChunkKeyEncoding::Default, ChunkKeySeparator::Slash) => Ok(()),
            (ZarrFormat::V3, ChunkKeyEncoding::Default, ChunkKeySeparator::Dot) => {
                Err(SyncWriteError::UnsupportedZarrTargetConfig {
                    message: "zarr_format=V3 requires chunk_key_separator=Slash".to_string(),
                })
            }
            (ZarrFormat::V3, ChunkKeyEncoding::V2, _) => {
                Err(SyncWriteError::UnsupportedZarrTargetConfig {
                    message: "zarr_format=V3 does not support chunk_key_encoding=V2".to_string(),
                })
            }
        }
    }
}

/// Configuration for parallel coordinate selection (naming matches `async_zarr.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelCoordsConfig {
    /// Explicit user-provided parallel coords (overrides defaults when `Some`).
    pub parallel_coords: Option<CoordMap>,
    /// Default coordinate names to activate when explicit override is absent.
    pub default_parallel_coord_names: Vec<String>,
}

impl Default for ParallelCoordsConfig {
    fn default() -> Self {
        Self {
            parallel_coords: None,
            default_parallel_coord_names: DEFAULT_PARALLEL_COORD_NAMES
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }
}

/// Configuration for planner-internal caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerCachesConfig {
    /// Maximum plan template LRU entries.
    pub plan_template_lru_capacity: usize,
    /// Maximum chunk key LRU entries.
    pub chunk_key_lru_capacity: usize,
    /// Prefer affine (arithmetic) resolver for regular coordinate arrays.
    pub prefer_affine_resolver: bool,
}

impl Default for PlannerCachesConfig {
    fn default() -> Self {
        Self {
            plan_template_lru_capacity: DEFAULT_PLAN_TEMPLATE_LRU_CAPACITY,
            chunk_key_lru_capacity: DEFAULT_CHUNK_KEY_LRU_CAPACITY,
            prefer_affine_resolver: true,
        }
    }
}

/// Top-level write execution configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteExecutionConfig {
    /// Number of Rayon worker threads for this backend instance's copy/flush pools.
    ///
    /// Each `SyncZarrBackend` owns its own pools; this value is therefore
    /// per-instance and must be greater than zero.
    pub num_threads: usize,
    /// Input stability policy for the copy-barrier window.
    pub input_stability_policy: InputStabilityPolicy,
    /// Scheduler backend selection.
    pub scheduler_backend: SchedulerBackend,
    /// Minimum bounded queue capacity baseline for backpressure.
    ///
    /// Runtime may raise the effective capacity on first write to absorb a
    /// two-step burst (`max(queue_capacity, 2 * first_write_task_count)`).
    pub queue_capacity: usize,
    /// Parallel coordinate selection configuration.
    pub parallel_coords_config: ParallelCoordsConfig,
    /// Chunk planner algorithm.
    pub planner_algorithm: PlannerAlgorithm,
    /// Planner cache settings.
    pub planner_caches: PlannerCachesConfig,
    /// Zarr format/encoding/separator target.
    pub zarr_target: ZarrTargetConfig,
    /// If true, host inputs must provide `__array_interface__` for zero-staging
    /// pointer-based ingestion on the write hot path.
    pub require_host_array_interface: bool,
    /// Controls whether `fsync` is called after writing chunk and metadata files.
    pub fsync_policy: FsyncPolicy,
}

impl Default for WriteExecutionConfig {
    fn default() -> Self {
        Self {
            num_threads: DEFAULT_NUM_THREADS,
            input_stability_policy: InputStabilityPolicy::StrictGilHold,
            scheduler_backend: SchedulerBackend::RayonWorkStealing,
            queue_capacity: DEFAULT_NUM_THREADS * 2,
            parallel_coords_config: ParallelCoordsConfig::default(),
            planner_algorithm: PlannerAlgorithm::MixedRadixStreaming,
            planner_caches: PlannerCachesConfig::default(),
            zarr_target: ZarrTargetConfig::default(),
            require_host_array_interface: true,
            fsync_policy: FsyncPolicy::default(),
        }
    }
}
