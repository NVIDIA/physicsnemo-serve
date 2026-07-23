/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![recursion_limit = "256"]

//! PhysicsNeMo Serve Inference Server Library
//!
//! REST API for discovering manifest-driven plugins, validating schemas,
//! and enqueueing inference runs via Redis streams.

pub mod config;
pub mod plugin_registry;
pub mod run_envelope;

#[cfg(feature = "rest")]
pub mod artifact_store;
#[cfg(feature = "rest")]
pub mod handlers;
#[cfg(feature = "rest")]
pub mod metrics;
#[cfg(feature = "rest")]
pub mod openapi;
#[cfg(feature = "rest")]
pub mod redis_ops;
#[cfg(feature = "rest")]
pub mod state;

#[cfg(all(test, feature = "rest"))]
mod docs_examples_tests;
