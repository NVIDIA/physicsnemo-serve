/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::Result;
use scicomp_rq::{Message, Output};
use serde::{Deserialize, Serialize};

use crate::config::{InputStreamSpec, PythonRuntimeEnvConfig};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Criticality policy for background tasks.
/// Critical tasks terminate the engine on failure; BestEffort tasks are retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskCriticality {
    Critical,
    BestEffort,
}

/// Pre-resolved environment for a role, built from config at engine construction.
///
/// Role factories receive this instead of raw config, so they never need to
/// parse stream prefixes or expand output selectors themselves.
#[derive(Debug, Clone)]
pub struct RoleEnv {
    pub role_name: String,
    pub stream_prefix: String,
    pub inputs: Vec<InputStreamSpec>,
    pub resolved_outputs: Vec<String>,
    pub role_config: Option<serde_json::Value>,
    pub python_runtime_envs: std::collections::BTreeMap<String, PythonRuntimeEnvConfig>,
}

/// Role-facing sink for enqueuing messages downstream.
///
/// Roles call methods on this trait to send messages to other streams.
/// They never see the underlying transport (Redis, in-memory, etc.).
///
/// # Cancel safety
///
/// All methods are cancel-safe: dropping the future mid-await will not leave
/// shared state in a half-modified condition.
pub trait MessageSink: Send + Sync {
    /// Enqueue a new message to a logical stream.
    fn enqueue<'a>(
        &'a self,
        stream: &'a str,
        run_id: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>>;

    /// Enqueue a message to an explicit physical Redis stream key.
    ///
    /// This is an advanced escape hatch used when a component already holds
    /// a discovered stream key (for example from `gpu:registry`) rather than a
    /// logical stream name.
    ///
    /// Default behavior delegates to `enqueue`, which is suitable for in-memory
    /// tests and simple sinks. Redis-backed sinks should override this to avoid
    /// logical-name validation.
    fn enqueue_to_stream<'a>(
        &'a self,
        stream_key: &'a str,
        run_id: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        self.enqueue(stream_key, run_id, payload, stage)
    }

    /// Acknowledge the current message without forwarding it anywhere.
    fn ack_message<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>>;

    /// Atomic ack-current + enqueue-next via Lua script.
    /// Used when the role wants to guarantee exactly-once handoff.
    fn handoff<'a>(
        &'a self,
        msg: &'a Message,
        dest_stream: &'a str,
        payload: &'a str,
        stage: &'a str,
    ) -> BoxFuture<'a, Result<String>>;

    /// Mark a request/run as failed.
    ///
    /// This is deliberately separate from generic handoff so queue primitives
    /// stay stage-agnostic while failure semantics remain explicit.
    fn mark_request_failed<'a>(&'a self, _run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Atomic ack-current + enqueue-next while overriding the destination
    /// run_id. Used for parent aggregation flows that must preserve atomicity.
    fn handoff_to_run<'a>(
        &'a self,
        msg: &'a Message,
        dest_stream: &'a str,
        payload: &'a str,
        stage: &'a str,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<String>> {
        if run_id == msg.run_id() {
            self.handoff(msg, dest_stream, payload, stage)
        } else {
            Box::pin(async move {
                Err(anyhow::anyhow!(
                    "handoff_to_run override is not supported by this sink implementation"
                ))
            })
        }
    }

    /// Atomic ack-current + fan-out to multiple destinations.
    fn forward_many<'a>(
        &'a self,
        msg: &'a Message,
        outputs: &'a [Output],
    ) -> BoxFuture<'a, Result<Vec<String>>>;
}

/// Business logic for a single worker role.
///
/// The engine calls `handle` for each message polled from the role's input
/// streams. The `stream` parameter is the logical stream name (prefix already
/// stripped by the engine). The `sink` provides downstream enqueue capabilities.
///
/// Returning `Ok(())` causes the engine to ack the message.
/// Returning `Err` causes the engine to skip the ack (message remains pending).
pub trait WorkerRole: Send + Sync + 'static {
    /// Human-readable role name for telemetry.
    fn name(&self) -> &'static str;

    /// Process a single message from the given logical stream.
    fn handle<'a>(
        &'a self,
        msg: &'a Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>>;
}

/// Periodic background work registered alongside a role.
///
/// Background tasks run on the engine's schedule, gated by `interval()`.
/// They receive `&dyn MessageSink` for any enqueue work they need to do
/// (e.g., heartbeats), but most tasks only update internal role state.
pub trait BackgroundTask: Send + Sync + 'static {
    /// Human-readable task name for telemetry.
    fn name(&self) -> &'static str;

    /// Minimum interval between executions.
    fn interval(&self) -> Duration;

    /// Failure policy: Critical stops the engine, BestEffort is retried.
    fn criticality(&self) -> TaskCriticality;

    /// Execute one tick of the background task.
    fn run<'a>(&'a self, sink: &'a dyn MessageSink) -> BoxFuture<'a, Result<()>>;
}

/// Transport abstraction used by the engine.
///
/// Roles never interact with this trait directly — they see `&dyn MessageSink`.
/// The engine uses `QueueTransport` to poll streams, ack messages, and reclaim
/// idle pending entries. The `as_sink()` method bridges to the role-facing API.
pub trait QueueTransport: Send + Sync + 'static {
    fn poll_stream<'a>(
        &'a self,
        stream: &'a str,
        consumer: &'a str,
        count: usize,
        block_ms: u64,
    ) -> BoxFuture<'a, Result<Vec<Message>>>;

    fn ack<'a>(&'a self, msg: &'a Message) -> BoxFuture<'a, Result<()>>;

    fn reclaim_idle<'a>(
        &'a self,
        stream: &'a str,
        consumer: &'a str,
        min_idle_ms: u64,
        count: usize,
    ) -> BoxFuture<'a, Result<Vec<Message>>>;

    fn create_consumer_group<'a>(
        &'a self,
        stream: &'a str,
        group: &'a str,
    ) -> BoxFuture<'a, Result<()>>;

    /// Increment the failure attempt counter for a message.
    ///
    /// Implementations that cannot persist attempts can return `Ok(None)` and
    /// let the engine fall back to process-local tracking.
    fn increment_failure_attempt<'a>(
        &'a self,
        _msg: &'a Message,
    ) -> BoxFuture<'a, Result<Option<usize>>> {
        Box::pin(async { Ok(None) })
    }

    /// Clear the failure attempt counter for a message after success or DLQ handoff.
    ///
    /// Implementations may keep the default no-op if they do not persist
    /// failure attempt counters.
    fn clear_failure_attempt<'a>(&'a self, _msg: &'a Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Expose the MessageSink view for roles and background tasks.
    fn as_sink(&self) -> &dyn MessageSink;
}
