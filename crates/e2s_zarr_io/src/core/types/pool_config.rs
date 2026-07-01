/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Buffer-pool sizing and warmup configuration types.

use super::config::{
    DEFAULT_CLOSE_TIMEOUT_SECONDS, DEFAULT_GLOBAL_FALLBACK_CHUNK_BYTES,
    DEFAULT_HOT_SLAB_READY_TIMEOUT_SECONDS, DEFAULT_MAX_POOL_BYTES, DEFAULT_MAX_WARM_TO_HOT_RATIO,
    DEFAULT_MIN_HOT_SLAB_BUFFERS, DEFAULT_NUM_THREADS, DEFAULT_TRANSIENT_LIMIT_BYTES,
};

/// Pooled buffer alignment requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoolAlignment {
    /// 4 KiB alignment (default).
    Align4KiB,
    /// 64 KiB alignment (for large-page or GPU-friendly allocations).
    Align64KiB,
}

/// Buffer sizing policy for pool initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferSizingPolicy {
    /// Use explicit fixed values only (no auto-derivation).
    FixedOnly,
    /// Derive sizes from first-write planner statistics and model profile hints (v1 default).
    FirstWriteModelAwareAuto,
}

/// Optional model profile hint for first-write sizing (Appendix A baselines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelProfileHint {
    /// FourCastNet (~3.96 MiB/chunk).
    Fcn,
    /// DLWP (~3.96 MiB/chunk).
    Dlwp,
    /// SFNO (~3.96 MiB/chunk).
    Sfno,
    /// Pangu-Weather (~3.96 MiB/chunk).
    Pangu,
    /// GraphCast-Small (~0.25 MiB/chunk).
    GraphCastSmall,
    /// StormCast (~1.25 MiB/chunk).
    StormCast,
    /// Precipitation AFNO (~3.96 MiB/chunk).
    PrecipitationAfno,
    /// CorrDiff Taiwan (~0.76 MiB/chunk).
    CorrDiffTaiwan,
}

/// Configuration for first-write auto-sizing policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstWriteSizingConfig {
    /// Sizing policy selection.
    pub buffer_sizing_policy: BufferSizingPolicy,
    /// Optional model profile hint for baseline chunk-size prior.
    pub model_profile_hint: Option<ModelProfileHint>,
    /// Minimum hot slab buffer count (default: 4).
    pub min_hot_slab_buffers: usize,
    /// Maximum warm-to-hot slab ratio (default: 3).
    pub max_warm_to_hot_ratio: usize,
    /// Fallback chunk size when no model hint is available (default: 4 MiB).
    pub global_fallback_chunk_bytes: usize,
}

impl Default for FirstWriteSizingConfig {
    fn default() -> Self {
        Self {
            buffer_sizing_policy: BufferSizingPolicy::FirstWriteModelAwareAuto,
            model_profile_hint: None,
            min_hot_slab_buffers: DEFAULT_MIN_HOT_SLAB_BUFFERS,
            max_warm_to_hot_ratio: DEFAULT_MAX_WARM_TO_HOT_RATIO,
            global_fallback_chunk_bytes: DEFAULT_GLOBAL_FALLBACK_CHUNK_BYTES,
        }
    }
}

/// Slab allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlabAllocationPolicy {
    /// Require contiguous allocation; fail immediately if unavailable.
    ContiguousOnlyFailFast,
}

/// Warm slab warmup trigger policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarmSlabWarmupPolicy {
    /// Trigger warm slab warmup in background on first write (v1 default).
    OnFirstWriteBackground,
}

/// Warm slab warmup failure policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarmSlabFailurePolicy {
    /// Continue in degraded mode (hot slab + transient buffers only).
    DegradeContinue,
}

/// Warm slab lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarmSlabState {
    /// Warmup has not started.
    NotStarted,
    /// Warmup is in progress (background).
    InProgress,
    /// Warmup completed; warm buffers are available.
    Ready,
    /// Warmup failed; backend is in degraded mode.
    FailedDegraded,
}

/// Diagnostic status for the buffer pool's hot/warm slab readiness.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolWarmupStatus {
    /// Whether all hot slab buffers are ready (pinned + optionally CUDA-registered).
    pub hot_ready: bool,
    /// Current warm slab lifecycle state.
    pub warm_state: WarmSlabState,
}

impl Default for PoolWarmupStatus {
    fn default() -> Self {
        Self {
            hot_ready: false,
            warm_state: WarmSlabState::NotStarted,
        }
    }
}

/// Explicit auto-sizing vs fixed-value policy for pool parameters.
///
/// Replaces `Option<usize>` to make the "None = auto" semantics type-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizeOverride {
    /// Derive the value automatically (e.g., from first-write statistics).
    #[default]
    Auto,
    /// Use the provided fixed value.
    Fixed(usize),
}

impl SizeOverride {
    /// Returns the fixed value if present, otherwise `None`.
    #[must_use]
    pub fn fixed_value(&self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Fixed(v) => Some(*v),
        }
    }

    /// Returns `true` if this is `Auto`.
    #[must_use]
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
}

impl From<Option<usize>> for SizeOverride {
    fn from(opt: Option<usize>) -> Self {
        match opt {
            Some(v) => Self::Fixed(v),
            None => Self::Auto,
        }
    }
}

impl From<SizeOverride> for Option<usize> {
    fn from(s: SizeOverride) -> Self {
        s.fixed_value()
    }
}

/// Full buffer pool configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferPoolConfig {
    /// Maximum number of pooled buffers across both slabs.
    pub max_pool_buffers: usize,
    /// Maximum total pooled memory budget in bytes.
    pub max_pool_bytes: usize,
    /// Pooled buffer size policy (`Auto` = derive from first write, `Fixed(n)` = use `n` bytes).
    pub pool_buffer_bytes: SizeOverride,
    /// Hot slab buffer count policy (`Auto` = derive, `Fixed(n)` = use `n` buffers).
    pub hot_slab_buffers: SizeOverride,
    /// Warm slab buffer count policy (`Auto` = derive, `Fixed(n)` = use `n` buffers).
    pub warm_slab_buffers: SizeOverride,
    /// First-write auto-sizing configuration.
    pub first_write_sizing: FirstWriteSizingConfig,
    /// Required pooled buffer alignment.
    pub pool_alignment: PoolAlignment,
    /// Whether to pin pooled slab memory (default: true).
    pub pin_pooled_slabs: bool,
    /// Whether to CUDA-register pooled slabs when CUDA is available (default: true).
    pub cuda_register_pool_if_available: bool,
    /// Whether each slab is CUDA-registered once on readiness (default: true).
    pub cuda_register_each_slab_once: bool,
    /// Timeout for hot slab readiness on first write (seconds).
    pub hot_slab_ready_timeout_seconds: f64,
    /// Warm slab warmup trigger policy.
    pub warm_slab_warmup_policy: WarmSlabWarmupPolicy,
    /// Warm slab warmup failure policy.
    pub warm_slab_failure_policy: WarmSlabFailurePolicy,
    /// Max transient buffer size (`None` = unlimited). Default: 2 GiB.
    pub max_transient_buffer_bytes: Option<usize>,
    /// Max total in-flight transient bytes across outstanding transient leases (`None` = unlimited).
    pub max_inflight_transient_bytes: Option<usize>,
    /// Default close timeout (seconds) for lease-return and close waits. Default: 300.
    pub close_lease_timeout_seconds: f64,
    /// Slab allocation strategy.
    pub slab_allocation_policy: SlabAllocationPolicy,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_buffers: DEFAULT_NUM_THREADS,
            max_pool_bytes: DEFAULT_MAX_POOL_BYTES,
            pool_buffer_bytes: SizeOverride::Auto,
            hot_slab_buffers: SizeOverride::Auto,
            warm_slab_buffers: SizeOverride::Auto,
            first_write_sizing: FirstWriteSizingConfig::default(),
            pool_alignment: PoolAlignment::Align4KiB,
            pin_pooled_slabs: true,
            cuda_register_pool_if_available: true,
            cuda_register_each_slab_once: true,
            hot_slab_ready_timeout_seconds: DEFAULT_HOT_SLAB_READY_TIMEOUT_SECONDS,
            warm_slab_warmup_policy: WarmSlabWarmupPolicy::OnFirstWriteBackground,
            warm_slab_failure_policy: WarmSlabFailurePolicy::DegradeContinue,
            max_transient_buffer_bytes: Some(DEFAULT_TRANSIENT_LIMIT_BYTES),
            max_inflight_transient_bytes: Some(DEFAULT_TRANSIENT_LIMIT_BYTES),
            close_lease_timeout_seconds: DEFAULT_CLOSE_TIMEOUT_SECONDS,
            slab_allocation_policy: SlabAllocationPolicy::ContiguousOnlyFailFast,
        }
    }
}
