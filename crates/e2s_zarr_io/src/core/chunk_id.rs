/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic chunk identity type.
//!
//! `ChunkId` is the internal identity for a logical Zarr chunk, computed from
//! mixed-radix strides over the array's coordinate space. It is format-agnostic;
//! conversion to an on-disk path is delegated to the [`ZarrLayoutAdapter`](super::contracts::ZarrLayoutAdapter).

/// Unique identity for a logical Zarr chunk within a backend lifetime.
///
/// Composed of an `array_id` (namespace per registered array) and a
/// `linear_index` (mixed-radix linearized chunk position within that array).
///
/// `ChunkId` implements `Ord` for deterministic iteration and `Hash` for
/// efficient conflict detection in the no-overwrite registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId {
    array_id: u32,
    linear_index: u64,
}

impl ChunkId {
    /// Construct a new `ChunkId`.
    #[must_use]
    pub const fn new(array_id: u32, linear_index: u64) -> Self {
        Self {
            array_id,
            linear_index,
        }
    }

    /// The array namespace this chunk belongs to.
    #[must_use]
    pub const fn array_id(&self) -> u32 {
        self.array_id
    }

    /// The mixed-radix linearized chunk index within the array.
    #[must_use]
    pub const fn linear_index(&self) -> u64 {
        self.linear_index
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChunkId(array={}, idx={})",
            self.array_id, self.linear_index
        )
    }
}
