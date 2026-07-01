/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core domain types, error definitions, and trait contracts for `e2s_zarr_io`.
//!
//! This module contains the foundational types that define the crate's public contract:
//!
//! - [`chunk_id`] — `ChunkId` identity type for deterministic chunk addressing.
//! - [`contracts`] — Trait boundaries between internal components.
//! - [`errors`] — Error types used across all write-path operations.
//! - [`types`] — Configuration, request, and response value types.

pub mod chunk_id;
pub mod contracts;
pub mod errors;
pub mod types;
