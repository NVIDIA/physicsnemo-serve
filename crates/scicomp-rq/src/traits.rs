/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Trait contracts for queue operations.
//!
//! These traits define the interface-first contract layer for `scicomp-rq`.
//! Concrete Redis execution remains in `QueueManager`, but callers and
//! orchestrators can depend on trait contracts instead of concrete types.

use crate::{
    HandoffRequest, HealthStatus, LogicalStreamName, Message, Output, QueueManager, Result,
    StreamKey,
};

/// Read operations on Redis streams.
pub trait ReadOps {
    /// Read messages from a stream consumer group.
    async fn read_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        count: usize,
        block_ms: usize,
    ) -> Result<Vec<Message>>;
}

/// Enqueue operations.
pub trait EnqueueOps {
    /// Enqueue a message into a logical stream.
    async fn enqueue(
        &self,
        stream_name: &LogicalStreamName,
        run_id: &str,
        payload: &str,
        stage: &str,
    ) -> Result<String>;
}

/// Acknowledgement operations.
pub trait AckOps {
    /// Acknowledge a consumed message.
    async fn ack_message(&self, message: &Message) -> Result<i64>;
}

/// Atomic operations (handoff + fan-out).
pub trait AtomicOps {
    /// Atomically add to next stream and acknowledge current message.
    async fn handoff_and_ack(&self, request: &HandoffRequest) -> Result<String>;

    /// Atomically hand off a message using message context.
    async fn handoff_message(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
    ) -> Result<String>;

    /// Atomically hand off a message using message context, with an optional
    /// destination run_id override.
    async fn handoff_message_to_run(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<String>;

    /// Fan out to multiple outputs.
    async fn forward_many(&self, message: &Message, outputs: &[Output]) -> Result<Vec<String>>;
}

/// Recovery operations for pending/reclaim workflows.
pub trait RecoveryOps {
    /// Claim idle pending messages.
    async fn claim_idle_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        min_idle_ms: u64,
        start_id: &str,
        count: usize,
    ) -> Result<(String, Vec<Message>)>;
}

/// Consumer group management operations.
pub trait GroupOps {
    /// Create a consumer group for a stream.
    async fn create_consumer_group(
        &self,
        stream: &StreamKey,
        group: &str,
        start_id: &str,
        create_stream: bool,
    ) -> Result<bool>;
}

/// Health diagnostics operations.
pub trait HealthOps {
    /// Perform a health check.
    async fn health_check(&self) -> Result<HealthStatus>;
}

/// Composite queue operation contract used by orchestrators/builders.
pub trait QueueOps:
    ReadOps + EnqueueOps + AckOps + AtomicOps + RecoveryOps + GroupOps + HealthOps
{
}

impl<T> QueueOps for T where
    T: ReadOps + EnqueueOps + AckOps + AtomicOps + RecoveryOps + GroupOps + HealthOps
{
}

impl ReadOps for QueueManager {
    async fn read_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        count: usize,
        block_ms: usize,
    ) -> Result<Vec<Message>> {
        QueueManager::read_messages(self, stream, group, consumer, count, block_ms).await
    }
}

impl EnqueueOps for QueueManager {
    async fn enqueue(
        &self,
        stream_name: &LogicalStreamName,
        run_id: &str,
        payload: &str,
        stage: &str,
    ) -> Result<String> {
        QueueManager::enqueue(self, stream_name, run_id, payload, stage).await
    }
}

impl AckOps for QueueManager {
    async fn ack_message(&self, message: &Message) -> Result<i64> {
        QueueManager::ack_message(self, message).await
    }
}

impl AtomicOps for QueueManager {
    async fn handoff_and_ack(&self, request: &HandoffRequest) -> Result<String> {
        QueueManager::handoff_and_ack(self, request).await
    }

    async fn handoff_message(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
    ) -> Result<String> {
        QueueManager::handoff_message(self, message, dest_stream, payload, stage).await
    }

    async fn handoff_message_to_run(
        &self,
        message: &Message,
        dest_stream: &StreamKey,
        payload: Option<&str>,
        stage: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<String> {
        QueueManager::handoff_message_to_run(self, message, dest_stream, payload, stage, run_id)
            .await
    }

    async fn forward_many(&self, message: &Message, outputs: &[Output]) -> Result<Vec<String>> {
        QueueManager::forward_many(self, message, outputs).await
    }
}

impl RecoveryOps for QueueManager {
    async fn claim_idle_messages(
        &self,
        stream: &StreamKey,
        group: &str,
        consumer: &str,
        min_idle_ms: u64,
        start_id: &str,
        count: usize,
    ) -> Result<(String, Vec<Message>)> {
        QueueManager::claim_idle_messages(
            self,
            stream,
            group,
            consumer,
            min_idle_ms,
            start_id,
            count,
        )
        .await
    }
}

impl GroupOps for QueueManager {
    async fn create_consumer_group(
        &self,
        stream: &StreamKey,
        group: &str,
        start_id: &str,
        create_stream: bool,
    ) -> Result<bool> {
        QueueManager::create_consumer_group(self, stream, group, start_id, create_stream).await
    }
}

impl HealthOps for QueueManager {
    async fn health_check(&self) -> Result<HealthStatus> {
        QueueManager::health_check(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_queue_ops_impl<T: QueueOps>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_queue_manager_implements_queue_ops_contract() {
        assert_queue_ops_impl::<QueueManager>();
    }

    #[test]
    fn test_queue_manager_is_send_and_sync() {
        assert_send_sync::<QueueManager>();
    }

    #[test]
    fn backend_marker_traits_are_removed() {
        let source = include_str!("traits.rs");
        let non_test_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("traits.rs should contain non-test source");
        assert!(
            !non_test_source.contains("trait StreamBackend"),
            "dead StreamBackend marker trait should be removed once unused"
        );
        assert!(
            !non_test_source.contains("trait ScriptBackend"),
            "dead ScriptBackend marker trait should be removed once unused"
        );
        assert!(
            !non_test_source.contains("trait HashBackend"),
            "dead HashBackend marker trait should be removed once unused"
        );
    }
}
