/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zarr format-specific layout adapters, chunk writers, and metadata consolidation.
//!
//! This module implements format-aware operations for Zarr v2 and v3:
//!
//! - `metadata` (internal) — Format-aware metadata consolidation helpers.
//! - `writer` (internal) — `ChunkWriter` for local filesystem with `ChunkId`→path rendering.
//! - [`zarr_layout`] — `ZarrLayoutAdapter` for v2/v3 metadata and chunk path policies.

#[cfg(feature = "test-utils")]
pub mod metadata;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod metadata;
#[cfg(feature = "test-utils")]
pub mod writer;
#[cfg(not(feature = "test-utils"))]
pub(crate) mod writer;
pub mod zarr_layout;
