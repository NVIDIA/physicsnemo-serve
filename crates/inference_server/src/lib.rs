/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![recursion_limit = "256"]

//! PhysicsNeMo Serve Inference Server Library
//!
//! REST API for discovering manifest-driven plugins, validating schemas,
//! and enqueueing inference runs via Redis streams.

pub mod artifact_store;
pub mod config;
pub mod handlers;
pub mod metrics;
pub mod openapi;
pub mod plugin_registry;
pub mod redis_ops;
pub mod run_envelope;
pub mod state;

#[cfg(test)]
mod docs_examples_tests;
