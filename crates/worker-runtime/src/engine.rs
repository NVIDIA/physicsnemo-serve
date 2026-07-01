/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use tokio::sync::watch;
use tokio::time::Duration;
use tracing::{debug, info, warn};

use crate::config::{InputStreamSpec, RuntimeConfig};
use crate::health::HealthState;
use crate::retry_dlq::{LocalFailureTracker, RetryDlqPolicy};
use crate::roles::scheduler::is_scheduler_deferred_error;
use crate::traits::{
    BackgroundTask, MessageSink, QueueTransport, RoleEnv, TaskCriticality, WorkerRole,
};
use crate::transport::consumer_group_name;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub struct RunOnceStats {
    pub polled: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub acked: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub struct RunLoopStats {
    pub iterations: usize,
    pub polled: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub acked: usize,
}

/// Config-resolved worker engine.
///
/// Owns the transport, role, background tasks, and resolved environment.
/// The engine handles polling, ack, reclaim, prefix stripping, and background
/// task supervision. Roles only see `&dyn MessageSink` and logical stream names.
pub struct WorkerEngine {
    transport: Arc<dyn QueueTransport>,
    role: Box<dyn WorkerRole>,
    background_tasks: Vec<Box<dyn BackgroundTask>>,
    env: RoleEnv,
    consumer: String,
    retry_dlq_policy: RetryDlqPolicy,
    background_last_run: Mutex<HashMap<String, Instant>>,
    failure_attempts: LocalFailureTracker,
}

impl WorkerEngine {
    pub(crate) fn new(
        transport: Arc<dyn QueueTransport>,
        role: Box<dyn WorkerRole>,
        background_tasks: Vec<Box<dyn BackgroundTask>>,
        env: RoleEnv,
        consumer: String,
        retry_dlq_policy: RetryDlqPolicy,
    ) -> Self {
        Self {
            transport,
            role,
            background_tasks,
            env,
            consumer,
            retry_dlq_policy,
            background_last_run: Mutex::new(HashMap::new()),
            failure_attempts: LocalFailureTracker::default(),
        }
    }

    /// Execute one pass: reclaim + poll all inputs, dispatch, ack, run background tasks.
    pub async fn run_once(&self) -> Result<RunOnceStats> {
        let inputs: Vec<&InputStreamSpec> = self.env.inputs.iter().collect();
        self.run_with_inputs(&inputs).await
    }

    async fn run_with_inputs(&self, inputs: &[&InputStreamSpec]) -> Result<RunOnceStats> {
        let mut stats = RunOnceStats::default();
        let sink = self.transport.as_sink();

        // Background tasks run first so that discovery (e.g. GPU routes)
        // is up-to-date before we dispatch any messages this iteration.
        self.run_due_background_tasks(sink).await?;

        for input in inputs {
            let reclaimed = self
                .transport
                .reclaim_idle(
                    input.stream.as_str(),
                    self.consumer.as_str(),
                    input.reclaim_idle_ms,
                    input.max_dequeue_items,
                )
                .await?;
            stats.polled += reclaimed.len();
            self.dispatch_messages(&reclaimed, &input.stream, sink, &mut stats)
                .await?;

            let messages = self
                .transport
                .poll_stream(
                    input.stream.as_str(),
                    self.consumer.as_str(),
                    input.max_dequeue_items,
                    input.block_ms,
                )
                .await?;
            stats.polled += messages.len();
            self.dispatch_messages(&messages, &input.stream, sink, &mut stats)
                .await?;
        }

        Ok(stats)
    }

    async fn record_failure_attempt(&self, msg: &scicomp_rq::Message) -> usize {
        match self.transport.increment_failure_attempt(msg).await {
            Ok(Some(attempts)) => return attempts,
            Ok(None) => {}
            Err(error) => {
                warn!(
                    role = self.role.name(),
                    stream = msg.stream(),
                    msg_id = msg.id(),
                    error = %error,
                    "failed to persist failure attempt count, falling back to process-local tracking"
                );
            }
        }

        self.failure_attempts.increment(msg)
    }

    async fn clear_failure_attempt(&self, msg: &scicomp_rq::Message) {
        if let Err(error) = self.transport.clear_failure_attempt(msg).await {
            warn!(
                role = self.role.name(),
                stream = msg.stream(),
                msg_id = msg.id(),
                error = %error,
                "failed to clear persisted failure attempt count"
            );
        }
        self.failure_attempts.clear(msg);
    }

    async fn dispatch_messages(
        &self,
        messages: &[scicomp_rq::Message],
        logical_stream: &str,
        sink: &dyn MessageSink,
        stats: &mut RunOnceStats,
    ) -> Result<()> {
        for msg in messages {
            match self.role.handle(msg, logical_stream, sink).await {
                Ok(()) => {
                    self.transport.ack(msg).await?;
                    self.clear_failure_attempt(msg).await;
                    stats.succeeded += 1;
                    stats.acked += 1;
                }
                Err(err) => {
                    if is_scheduler_deferred_error(&err) {
                        self.clear_failure_attempt(msg).await;
                        debug!(
                            role = self.role.name(),
                            stream = logical_stream,
                            msg_id = msg.id(),
                            "message deferred by scheduler role, leaving pending without ack"
                        );
                        continue;
                    }

                    let error_text = err.to_string();
                    let attempts = self.record_failure_attempt(msg).await;
                    if attempts < self.retry_dlq_policy.max_retries() {
                        warn!(
                            role = self.role.name(),
                            stream = logical_stream,
                            msg_id = msg.id(),
                            attempt = attempts,
                            max_retries = self.retry_dlq_policy.max_retries(),
                            error = %error_text,
                            "message handler failed, skipping ack"
                        );
                        stats.failed += 1;
                        continue;
                    }

                    let dlq_payload = self.retry_dlq_policy.build_dlq_payload(
                        msg,
                        logical_stream,
                        &error_text,
                        attempts,
                    )?;
                    match sink
                        .handoff(msg, self.retry_dlq_policy.dlq_stream(), &dlq_payload, "dlq")
                        .await
                    {
                        Ok(_) => {
                            if let Err(status_err) = sink.mark_request_failed(msg.run_id()).await {
                                warn!(
                                    role = self.role.name(),
                                    stream = logical_stream,
                                    msg_id = msg.id(),
                                    run_id = msg.run_id(),
                                    error = %status_err,
                                    "DLQ handoff succeeded but failed to mark run status failed"
                                );
                            }
                            self.clear_failure_attempt(msg).await;
                            stats.failed += 1;
                            stats.acked += 1;
                            warn!(
                                role = self.role.name(),
                                stream = logical_stream,
                                msg_id = msg.id(),
                                attempt = attempts,
                                max_retries = self.retry_dlq_policy.max_retries(),
                                dlq_stream = %self.retry_dlq_policy.dlq_stream(),
                                "message moved to shared DLQ after retry limit"
                            );
                        }
                        Err(dlq_err) => {
                            warn!(
                                role = self.role.name(),
                                stream = logical_stream,
                                msg_id = msg.id(),
                                attempt = attempts,
                                max_retries = self.retry_dlq_policy.max_retries(),
                                dlq_stream = %self.retry_dlq_policy.dlq_stream(),
                                handler_error = %error_text,
                                dlq_error = %dlq_err,
                                "retry limit reached but DLQ handoff failed, keeping message pending"
                            );
                            stats.failed += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn run_due_background_tasks(&self, sink: &dyn MessageSink) -> Result<()> {
        for task in &self.background_tasks {
            let should_run = {
                let mut last_run = self
                    .background_last_run
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let key = task.name().to_string();
                let interval = task.interval();
                match last_run.get(&key) {
                    Some(last) if !interval.is_zero() => {
                        if Instant::now().duration_since(*last) < interval {
                            false
                        } else {
                            last_run.insert(key, Instant::now());
                            true
                        }
                    }
                    _ => {
                        last_run.insert(key, Instant::now());
                        true
                    }
                }
            };

            if !should_run {
                continue;
            }

            if let Err(err) = task.run(sink).await {
                match task.criticality() {
                    TaskCriticality::Critical => {
                        warn!(
                            role = self.role.name(),
                            task = task.name(),
                            criticality = "critical",
                            error = %err,
                            "critical background task failed, stopping engine"
                        );
                        return Err(err).with_context(|| {
                            format!(
                                "critical background task '{}' failed in role '{}'",
                                task.name(),
                                self.role.name()
                            )
                        });
                    }
                    TaskCriticality::BestEffort => {
                        warn!(
                            role = self.role.name(),
                            task = task.name(),
                            criticality = "best_effort",
                            error = %err,
                            "background task failed, will retry"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Ensure consumer groups exist for all input streams before polling.
    ///
    /// This is idempotent — if groups already exist, the call is a no-op.
    pub async fn ensure_consumer_groups(&self) -> Result<()> {
        for input in &self.env.inputs {
            let group = consumer_group_name(&input.stream);
            self.transport
                .create_consumer_group(&input.stream, &group)
                .await
                .with_context(|| {
                    format!(
                        "failed to create consumer group '{}' for stream '{}'",
                        group, input.stream
                    )
                })?;
        }
        Ok(())
    }

    /// Run the poll loop until a shutdown signal is received.
    ///
    /// Automatically creates consumer groups on startup.
    ///
    /// # Cancellation safety
    ///
    /// The two `tokio::select!` blocks in this function race
    /// `tokio::time::sleep` against `shutdown.changed()`. Both futures
    /// are stateless sleep/channel-watch operations that modify no shared
    /// state, so dropping either branch mid-`.await` is safe. If additional
    /// branches are added in the future, verify that they do not leave
    /// `stats`, `last_polled`, or transport state in an inconsistent
    /// condition when dropped.
    /// Run the engine loop until shutdown is signalled.
    ///
    /// When `health` is provided, it is marked alive after the first
    /// successful poll and receives a heartbeat after every iteration.
    pub async fn run_until_shutdown(
        &self,
        mut shutdown: watch::Receiver<bool>,
        health: Option<HealthState>,
    ) -> Result<RunLoopStats> {
        self.ensure_consumer_groups().await?;
        let mut stats = RunLoopStats::default();
        let mut last_polled: HashMap<String, Instant> = HashMap::new();

        loop {
            if *shutdown.borrow() {
                info!(
                    role = self.role.name(),
                    iterations = stats.iterations,
                    polled = stats.polled,
                    succeeded = stats.succeeded,
                    failed = stats.failed,
                    acked = stats.acked,
                    "shutdown signal received, stopping poll loop"
                );
                break;
            }

            let (due_inputs, next_wait) = self.select_due_inputs(&last_polled);
            if due_inputs.is_empty() {
                let wait = next_wait.unwrap_or_else(|| self.idle_interval());
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    changed = shutdown.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
                continue;
            }

            debug!(
                role = self.role.name(),
                consumer = %self.consumer,
                iteration = stats.iterations + 1,
                streams = ?due_inputs.iter().map(|i| i.stream.as_str()).collect::<Vec<_>>(),
                "poll iteration start"
            );

            let once = self.run_with_inputs(&due_inputs).await?;
            stats.iterations += 1;
            stats.polled += once.polled;
            stats.succeeded += once.succeeded;
            stats.failed += once.failed;
            stats.acked += once.acked;

            if let Some(ref h) = health {
                if stats.iterations == 1 {
                    h.mark_alive();
                } else {
                    h.record_heartbeat();
                }
            }

            let polled_at = Instant::now();
            for input in &due_inputs {
                last_polled.insert(input.stream.clone(), polled_at);
            }

            if *shutdown.borrow() {
                info!(
                    role = self.role.name(),
                    iterations = stats.iterations,
                    polled = stats.polled,
                    acked = stats.acked,
                    "shutdown signal received after draining current batch"
                );
                break;
            }

            if let Some(wait) = next_wait {
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    changed = shutdown.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
            }
        }

        Ok(stats)
    }

    fn idle_interval(&self) -> Duration {
        self.env
            .inputs
            .iter()
            .map(|input| Duration::from_millis(input.poll_interval_ms))
            .min()
            .unwrap_or_else(|| Duration::from_millis(10))
    }

    fn select_due_inputs<'a>(
        &'a self,
        last_polled: &HashMap<String, Instant>,
    ) -> (Vec<&'a InputStreamSpec>, Option<Duration>) {
        let now = Instant::now();
        let mut due = Vec::new();
        let mut next_wait: Option<Duration> = None;

        for input in &self.env.inputs {
            let interval = Duration::from_millis(input.poll_interval_ms);
            match last_polled.get(input.stream.as_str()) {
                None => due.push(input),
                Some(last) => {
                    let elapsed = now.saturating_duration_since(*last);
                    if elapsed >= interval {
                        due.push(input);
                    } else {
                        let remaining = interval - elapsed;
                        next_wait = Some(match next_wait {
                            Some(cur) if cur <= remaining => cur,
                            _ => remaining,
                        });
                    }
                }
            }
        }

        (due, next_wait)
    }
}

/// Builder for `WorkerEngine`.
///
/// Single construction path for both production and test usage.
pub struct EngineBuilder {
    config: RuntimeConfig,
    role_name: String,
    transport: Option<Arc<dyn QueueTransport>>,
    role: Option<Box<dyn WorkerRole>>,
    background_tasks: Vec<Box<dyn BackgroundTask>>,
    consumer: Option<String>,
}

impl EngineBuilder {
    pub fn new(config: &RuntimeConfig, role_name: &str) -> Self {
        Self {
            config: config.clone(),
            role_name: role_name.to_string(),
            transport: None,
            role: None,
            background_tasks: Vec::new(),
            consumer: None,
        }
    }

    pub fn transport(mut self, t: Arc<dyn QueueTransport>) -> Self {
        self.transport = Some(t);
        self
    }

    pub fn role(mut self, r: Box<dyn WorkerRole>) -> Self {
        self.role = Some(r);
        self
    }

    pub fn background_task(mut self, t: Box<dyn BackgroundTask>) -> Self {
        self.background_tasks.push(t);
        self
    }

    pub fn consumer(mut self, name: impl Into<String>) -> Self {
        self.consumer = Some(name.into());
        self
    }

    pub fn build(self) -> Result<WorkerEngine> {
        let env = self
            .config
            .resolve_env(&self.role_name)
            .with_context(|| format!("failed to resolve env for role '{}'", self.role_name))?;
        let transport = self
            .transport
            .ok_or_else(|| anyhow!("transport is required"))?;
        let role = self.role.ok_or_else(|| anyhow!("role is required"))?;
        let consumer = self.consumer.unwrap_or_else(|| {
            format!(
                "{}-{}-{}",
                self.role_name,
                std::process::id(),
                uuid::Uuid::new_v4()
            )
        });
        let retry_dlq_policy = RetryDlqPolicy::new(
            self.config.max_retries,
            self.config.shared_dlq_stream.clone(),
        );

        Ok(WorkerEngine::new(
            transport,
            role,
            self.background_tasks,
            env,
            consumer,
            retry_dlq_policy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::scheduler::scheduler_deferred_error;
    use crate::transport::memory::InMemoryTransport;
    use anyhow::anyhow;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Stub role: always succeeds ---
    struct OkRole;
    impl WorkerRole for OkRole {
        fn name(&self) -> &'static str {
            "ok-role"
        }
        fn handle<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            _sink: &'a dyn MessageSink,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    // --- Stub role: fails for specific run_ids ---
    struct FailRole {
        fail_run_ids: Vec<String>,
    }
    impl WorkerRole for FailRole {
        fn name(&self) -> &'static str {
            "fail-role"
        }
        fn handle<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            _sink: &'a dyn MessageSink,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            let should_fail = self.fail_run_ids.contains(&msg.run_id().to_string());
            Box::pin(async move {
                if should_fail {
                    Err(anyhow!("forced failure"))
                } else {
                    Ok(())
                }
            })
        }
    }

    struct DeferredRole;
    impl WorkerRole for DeferredRole {
        fn name(&self) -> &'static str {
            "deferred-role"
        }
        fn handle<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            _sink: &'a dyn MessageSink,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async { Err(scheduler_deferred_error()) })
        }
    }

    struct RecordingStatusSink {
        inner: Arc<InMemoryTransport>,
        failed_run_ids: Mutex<Vec<String>>,
    }

    impl RecordingStatusSink {
        fn new(inner: Arc<InMemoryTransport>) -> Self {
            Self {
                inner,
                failed_run_ids: Mutex::new(Vec::new()),
            }
        }

        fn failed_run_ids(&self) -> Vec<String> {
            self.failed_run_ids
                .lock()
                .expect("failed run ids lock should not be poisoned")
                .clone()
        }
    }

    impl MessageSink for RecordingStatusSink {
        fn enqueue<'a>(
            &'a self,
            stream: &'a str,
            run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<String>> {
            self.inner.enqueue(stream, run_id, payload, stage)
        }

        fn enqueue_to_stream<'a>(
            &'a self,
            stream_key: &'a str,
            run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<String>> {
            self.inner
                .enqueue_to_stream(stream_key, run_id, payload, stage)
        }

        fn ack_message<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            self.inner.ack_message(msg)
        }

        fn handoff<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<String>> {
            self.inner.handoff(msg, dest_stream, payload, stage)
        }

        fn mark_request_failed<'a>(
            &'a self,
            run_id: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.failed_run_ids
                    .lock()
                    .expect("failed run ids lock should not be poisoned")
                    .push(run_id.to_string());
                Ok(())
            })
        }

        fn forward_many<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            outputs: &'a [scicomp_rq::Output],
        ) -> crate::traits::BoxFuture<'a, Result<Vec<String>>> {
            self.inner.forward_many(msg, outputs)
        }
    }

    // --- Stub background task ---
    struct CountingTask {
        name: &'static str,
        interval: Duration,
        criticality: TaskCriticality,
        runs: Arc<AtomicUsize>,
        should_fail: bool,
    }

    struct ReclaimCaptureTransport {
        min_idle_ms: Mutex<Vec<u64>>,
    }

    impl ReclaimCaptureTransport {
        fn new() -> Self {
            Self {
                min_idle_ms: Mutex::new(Vec::new()),
            }
        }

        fn captured_min_idle_ms(&self) -> Vec<u64> {
            self.min_idle_ms
                .lock()
                .expect("capture lock should not be poisoned")
                .clone()
        }
    }

    impl MessageSink for ReclaimCaptureTransport {
        fn enqueue<'a>(
            &'a self,
            _stream: &'a str,
            _run_id: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("x".to_string()) })
        }

        fn ack_message<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _dest_stream: &'a str,
            _payload: &'a str,
            _stage: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<String>> {
            Box::pin(async { Ok("x".to_string()) })
        }

        fn forward_many<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _outputs: &'a [scicomp_rq::Output],
        ) -> crate::traits::BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    impl QueueTransport for ReclaimCaptureTransport {
        fn poll_stream<'a>(
            &'a self,
            _stream: &'a str,
            _consumer: &'a str,
            _count: usize,
            _block_ms: u64,
        ) -> crate::traits::BoxFuture<'a, Result<Vec<scicomp_rq::Message>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn ack<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn reclaim_idle<'a>(
            &'a self,
            _stream: &'a str,
            _consumer: &'a str,
            min_idle_ms: u64,
            _count: usize,
        ) -> crate::traits::BoxFuture<'a, Result<Vec<scicomp_rq::Message>>> {
            Box::pin(async move {
                self.min_idle_ms
                    .lock()
                    .expect("capture lock should not be poisoned")
                    .push(min_idle_ms);
                Ok(Vec::new())
            })
        }

        fn create_consumer_group<'a>(
            &'a self,
            _stream: &'a str,
            _group: &'a str,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn as_sink(&self) -> &dyn MessageSink {
            self
        }
    }
    impl BackgroundTask for CountingTask {
        fn name(&self) -> &'static str {
            self.name
        }
        fn interval(&self) -> Duration {
            self.interval
        }
        fn criticality(&self) -> TaskCriticality {
            self.criticality
        }
        fn run<'a>(
            &'a self,
            _sink: &'a dyn MessageSink,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            let fail = self.should_fail;
            Box::pin(async move {
                if fail {
                    Err(anyhow!("forced background failure"))
                } else {
                    Ok(())
                }
            })
        }
    }

    fn test_config(role_name: &str, inputs: &[&str], outputs: &[&str]) -> RuntimeConfig {
        let input_specs: Vec<serde_json::Value> = inputs
            .iter()
            .map(|s| {
                serde_json::json!({
                    "stream": s, "max_dequeue_items": 10,
                    "poll_interval_ms": 10, "block_ms": 50
                })
            })
            .collect();

        let mut all_streams: Vec<&str> = inputs.to_vec();
        all_streams.extend_from_slice(outputs);

        let mut roles = serde_json::Map::new();
        roles.insert(
            role_name.to_string(),
            serde_json::json!({
                "inputs": input_specs,
                "outputs": outputs
            }),
        );

        serde_json::from_value(serde_json::json!({
            "stream_prefix": "",
            "streams": all_streams,
            "roles": roles
        }))
        .expect("test config should parse")
    }

    fn build_engine(
        config: &RuntimeConfig,
        role_name: &str,
        transport: Arc<InMemoryTransport>,
        role: Box<dyn WorkerRole>,
        tasks: Vec<Box<dyn BackgroundTask>>,
    ) -> WorkerEngine {
        let mut builder = EngineBuilder::new(config, role_name)
            .transport(transport)
            .role(role)
            .consumer("test-consumer");
        for t in tasks {
            builder = builder.background_task(t);
        }
        builder.build().expect("engine should build")
    }

    // === Engine Tests ===

    #[tokio::test]
    async fn run_once_acks_successful_messages() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        transport
            .inject("input", "run-1", r#"{"ok":true}"#, "test")
            .unwrap();

        let engine = build_engine(
            &config,
            "worker",
            transport.clone(),
            Box::new(OkRole),
            vec![],
        );
        let stats = engine.run_once().await.unwrap();

        assert_eq!(stats.polled, 1);
        assert_eq!(stats.succeeded, 1);
        assert_eq!(stats.acked, 1);
        assert_eq!(transport.acked_ids().len(), 1);
    }

    #[tokio::test]
    async fn run_once_uses_configured_reclaim_idle_threshold() {
        let config: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "",
            "streams": ["input"],
            "roles": {
                "worker": {
                    "inputs": [{"stream": "input", "max_dequeue_items": 10,
                                "poll_interval_ms": 10, "block_ms": 50, "reclaim_idle_ms": 1234}],
                    "outputs": []
                }
            }
        }))
        .expect("test config should parse");
        let transport = Arc::new(ReclaimCaptureTransport::new());
        let engine = EngineBuilder::new(&config, "worker")
            .transport(transport.clone())
            .role(Box::new(OkRole))
            .consumer("capture-consumer")
            .build()
            .expect("engine should build");

        let stats = engine.run_once().await.expect("run_once should succeed");
        assert_eq!(stats.polled, 0);
        assert_eq!(transport.captured_min_idle_ms(), vec![1234]);
    }

    #[tokio::test]
    async fn run_once_does_not_ack_failed_messages() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        transport
            .inject("input", "run-ok", r#"{"ok":true}"#, "test")
            .unwrap();
        transport
            .inject("input", "run-fail", r#"{"ok":true}"#, "test")
            .unwrap();

        let role = FailRole {
            fail_run_ids: vec!["run-fail".to_string()],
        };
        let engine = build_engine(&config, "worker", transport.clone(), Box::new(role), vec![]);
        let stats = engine.run_once().await.unwrap();

        assert_eq!(stats.polled, 2);
        assert_eq!(stats.succeeded, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.acked, 1);
    }

    #[tokio::test]
    async fn dispatch_moves_poison_message_to_shared_dlq_after_five_failures() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input", "dlq"], ""));
        let role = FailRole {
            fail_run_ids: vec!["run-poison".to_string()],
        };
        let engine = build_engine(&config, "worker", transport.clone(), Box::new(role), vec![]);
        let sink = RecordingStatusSink::new(transport.clone());
        let sink_ref: &dyn MessageSink = &sink;
        let msg = scicomp_rq::Message::new(
            "fixed-1",
            "input",
            "input:grp",
            "run-poison",
            r#"{"bad":true}"#,
            "input",
        );
        let mut stats = RunOnceStats::default();

        for _ in 0..4 {
            engine
                .dispatch_messages(std::slice::from_ref(&msg), "input", sink_ref, &mut stats)
                .await
                .expect("dispatch should complete");
        }
        assert!(
            transport.pending_in("dlq").is_empty(),
            "message should not move to DLQ before retry limit"
        );

        engine
            .dispatch_messages(std::slice::from_ref(&msg), "input", sink_ref, &mut stats)
            .await
            .expect("dispatch should complete");

        let dlq_pending = transport.pending_in("dlq");
        assert_eq!(
            dlq_pending.len(),
            1,
            "5th failure should move message to DLQ"
        );
        assert!(
            transport.acked_ids().contains(&msg.id().to_string()),
            "DLQ handoff should ack the original message"
        );

        let dlq_payload: serde_json::Value =
            serde_json::from_str(dlq_pending[0].payload()).expect("DLQ payload should be JSON");
        assert_eq!(dlq_payload["status"].as_str(), Some("failed"));
        assert_eq!(dlq_payload["attempts"].as_u64(), Some(5));
        assert_eq!(sink.failed_run_ids(), vec!["run-poison".to_string()]);
    }

    #[tokio::test]
    async fn retry_attempts_persist_across_engine_restarts() {
        let config: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "",
            "max_retries": 2,
            "shared_dlq_stream": "dlq",
            "streams": ["input", "dlq"],
            "roles": {
                "worker": {
                    "inputs": [{"stream": "input", "max_dequeue_items": 10, "poll_interval_ms": 10, "block_ms": 50}],
                    "outputs": []
                }
            }
        }))
        .expect("test config should parse");
        let transport = Arc::new(InMemoryTransport::new(&["input", "dlq"], ""));
        let sink: &dyn MessageSink = transport.as_ref();
        let msg = scicomp_rq::Message::new(
            "fixed-poison-id",
            "input",
            "input:grp",
            "run-poison",
            r#"{"bad":true}"#,
            "input",
        );
        let mut stats = RunOnceStats::default();

        let engine_a = build_engine(
            &config,
            "worker",
            transport.clone(),
            Box::new(FailRole {
                fail_run_ids: vec!["run-poison".to_string()],
            }),
            vec![],
        );
        engine_a
            .dispatch_messages(std::slice::from_ref(&msg), "input", sink, &mut stats)
            .await
            .expect("first dispatch should complete");
        assert!(
            transport.pending_in("dlq").is_empty(),
            "first failure should not move message to DLQ"
        );

        let engine_b = build_engine(
            &config,
            "worker",
            transport.clone(),
            Box::new(FailRole {
                fail_run_ids: vec!["run-poison".to_string()],
            }),
            vec![],
        );
        engine_b
            .dispatch_messages(std::slice::from_ref(&msg), "input", sink, &mut stats)
            .await
            .expect("second dispatch should complete");

        let dlq_pending = transport.pending_in("dlq");
        assert_eq!(
            dlq_pending.len(),
            1,
            "second failure after restart should hit max_retries and move to DLQ"
        );
    }

    #[tokio::test]
    async fn deferred_errors_leave_messages_pending_without_retry_or_dlq() {
        let config: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "",
            "max_retries": 1,
            "shared_dlq_stream": "dlq",
            "streams": ["input", "dlq"],
            "roles": {
                "worker": {
                    "inputs": [{"stream": "input", "max_dequeue_items": 10, "poll_interval_ms": 10, "block_ms": 50}],
                    "outputs": []
                }
            }
        }))
        .expect("test config should parse");
        let transport = Arc::new(InMemoryTransport::new(&["input", "dlq"], ""));
        let sink: &dyn MessageSink = transport.as_ref();
        let msg = scicomp_rq::Message::new(
            "fixed-deferred-id",
            "input",
            "input:grp",
            "run-deferred",
            r#"{"queued":true}"#,
            "input",
        );
        let mut stats = RunOnceStats::default();
        let engine = build_engine(
            &config,
            "worker",
            transport.clone(),
            Box::new(DeferredRole),
            vec![],
        );

        for _ in 0..2 {
            engine
                .dispatch_messages(std::slice::from_ref(&msg), "input", sink, &mut stats)
                .await
                .expect("deferred dispatch should complete");
        }

        assert!(transport.acked_ids().is_empty());
        assert!(
            transport.pending_in("dlq").is_empty(),
            "deferred messages must not be sent to DLQ"
        );
    }

    #[tokio::test]
    async fn build_dlq_payload_truncates_large_original_payload() {
        let oversized = "x".repeat(5_000);
        let msg = scicomp_rq::Message::new(
            "1-0",
            "input",
            "input:grp",
            "run-large",
            &oversized,
            "input",
        );

        let payload = RetryDlqPolicy::new(5, "dlq")
            .build_dlq_payload(&msg, "input", "forced-error", 5)
            .expect("DLQ payload should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("DLQ payload should be valid JSON");
        let dlq_payload = parsed["payload"]
            .as_str()
            .expect("payload field should be string");

        assert!(
            dlq_payload.ends_with("...(truncated)"),
            "oversized payload should be truncated with marker"
        );
        assert!(
            dlq_payload.len() <= 4096 + "...(truncated)".len(),
            "truncated payload should be bounded"
        );
    }

    #[tokio::test]
    async fn run_once_processes_multiple_input_streams() {
        let config: RuntimeConfig = serde_json::from_value(serde_json::json!({
            "stream_prefix": "",
            "streams": ["schedule", "release"],
            "roles": {
                "scheduler": {
                    "inputs": [
                        {"stream": "schedule", "max_dequeue_items": 10,
                         "poll_interval_ms": 10, "block_ms": 50},
                        {"stream": "release", "max_dequeue_items": 10,
                         "poll_interval_ms": 10, "block_ms": 50}
                    ],
                    "outputs": []
                }
            }
        }))
        .unwrap();
        let transport = Arc::new(InMemoryTransport::new(&["schedule", "release"], ""));
        transport.inject("schedule", "run-1", "{}", "test").unwrap();
        transport.inject("release", "run-2", "{}", "test").unwrap();

        let engine = build_engine(
            &config,
            "scheduler",
            transport.clone(),
            Box::new(OkRole),
            vec![],
        );
        let stats = engine.run_once().await.unwrap();

        assert_eq!(stats.polled, 2);
        assert_eq!(stats.succeeded, 2);
        assert_eq!(stats.acked, 2);
    }

    #[tokio::test]
    async fn run_once_invokes_background_tasks() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let runs = Arc::new(AtomicUsize::new(0));
        let task: Box<dyn BackgroundTask> = Box::new(CountingTask {
            name: "test-task",
            interval: Duration::ZERO,
            criticality: TaskCriticality::BestEffort,
            runs: runs.clone(),
            should_fail: false,
        });

        let engine = build_engine(&config, "worker", transport, Box::new(OkRole), vec![task]);
        let _ = engine.run_once().await.unwrap();

        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn run_once_fails_on_critical_background_task_failure() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let task: Box<dyn BackgroundTask> = Box::new(CountingTask {
            name: "critical-task",
            interval: Duration::ZERO,
            criticality: TaskCriticality::Critical,
            runs: Arc::new(AtomicUsize::new(0)),
            should_fail: true,
        });

        let engine = build_engine(&config, "worker", transport, Box::new(OkRole), vec![task]);
        let result = engine.run_once().await;
        assert!(
            result.is_err(),
            "critical background failure must terminate engine"
        );
    }

    #[tokio::test]
    async fn run_once_continues_on_best_effort_background_failure() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let task: Box<dyn BackgroundTask> = Box::new(CountingTask {
            name: "best-effort-task",
            interval: Duration::ZERO,
            criticality: TaskCriticality::BestEffort,
            runs: Arc::new(AtomicUsize::new(0)),
            should_fail: true,
        });

        let engine = build_engine(&config, "worker", transport, Box::new(OkRole), vec![task]);
        let result = engine.run_once().await;
        assert!(
            result.is_ok(),
            "best-effort failure must not terminate engine"
        );
    }

    #[tokio::test]
    async fn run_once_skips_background_task_before_interval() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let runs = Arc::new(AtomicUsize::new(0));
        let task: Box<dyn BackgroundTask> = Box::new(CountingTask {
            name: "slow-task",
            interval: Duration::from_secs(3600),
            criticality: TaskCriticality::BestEffort,
            runs: runs.clone(),
            should_fail: false,
        });

        let engine = build_engine(&config, "worker", transport, Box::new(OkRole), vec![task]);
        let _ = engine.run_once().await.unwrap();
        let _ = engine.run_once().await.unwrap();

        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "task should run once then be gated by interval"
        );
    }

    #[tokio::test]
    async fn run_until_shutdown_honors_immediate_signal() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let engine = build_engine(&config, "worker", transport, Box::new(OkRole), vec![]);
        let (_tx, rx) = watch::channel(true);

        let stats = engine.run_until_shutdown(rx, None).await.unwrap();
        assert_eq!(stats.iterations, 0);
    }

    #[tokio::test]
    async fn run_until_shutdown_processes_then_stops() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        transport.inject("input", "run-1", "{}", "test").unwrap();

        let engine = build_engine(
            &config,
            "worker",
            transport.clone(),
            Box::new(OkRole),
            vec![],
        );
        let (tx, rx) = watch::channel(false);

        let handle = tokio::spawn(async move { engine.run_until_shutdown(rx, None).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send(true).unwrap();

        let stats = tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("should finish promptly")
            .expect("task should join")
            .expect("should succeed");

        assert!(stats.iterations >= 1);
        assert!(stats.polled >= 1);
    }

    // --- PR-056: graceful drain on shutdown ---

    struct SlowRole {
        delay: Duration,
        processed: Arc<AtomicUsize>,
    }
    impl WorkerRole for SlowRole {
        fn name(&self) -> &'static str {
            "slow-role"
        }
        fn handle<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            _stream: &'a str,
            _sink: &'a dyn MessageSink,
        ) -> crate::traits::BoxFuture<'a, Result<()>> {
            let delay = self.delay;
            let processed = self.processed.clone();
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                processed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn run_until_shutdown_drains_current_batch_before_exit() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        transport.inject("input", "run-1", "{}", "test").unwrap();

        let processed = Arc::new(AtomicUsize::new(0));
        let role = SlowRole {
            delay: Duration::from_millis(100),
            processed: processed.clone(),
        };

        let engine = build_engine(&config, "worker", transport.clone(), Box::new(role), vec![]);
        let (tx, rx) = watch::channel(false);

        let handle = tokio::spawn(async move { engine.run_until_shutdown(rx, None).await });

        // Wait for message to be polled, then signal shutdown mid-processing.
        tokio::time::sleep(Duration::from_millis(30)).await;
        tx.send(true).unwrap();

        let stats = tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("should finish promptly")
            .expect("task should join")
            .expect("should succeed");

        assert_eq!(
            processed.load(Ordering::Relaxed),
            1,
            "in-flight message must be processed before shutdown"
        );
        assert_eq!(
            stats.acked, 1,
            "in-flight message must be acked before shutdown"
        );
    }

    #[tokio::test]
    async fn run_until_shutdown_does_not_poll_new_messages_after_signal() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));

        let processed = Arc::new(AtomicUsize::new(0));
        let role = SlowRole {
            delay: Duration::from_millis(10),
            processed: processed.clone(),
        };

        let engine = build_engine(&config, "worker", transport.clone(), Box::new(role), vec![]);
        let (tx, rx) = watch::channel(false);

        let handle = tokio::spawn(async move { engine.run_until_shutdown(rx, None).await });

        // Signal shutdown immediately, then inject a message.
        tx.send(true).unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        transport.inject("input", "run-late", "{}", "test").unwrap();

        let stats = tokio::time::timeout(Duration::from_millis(300), handle)
            .await
            .expect("should finish promptly")
            .expect("task should join")
            .expect("should succeed");

        assert_eq!(
            processed.load(Ordering::Relaxed),
            0,
            "message injected after shutdown should not be processed"
        );
        assert_eq!(stats.polled, 0);
    }

    // === EngineBuilder Tests ===

    #[test]
    fn builder_rejects_missing_transport() {
        let config = test_config("worker", &["input"], &[]);
        let err = EngineBuilder::new(&config, "worker")
            .role(Box::new(OkRole))
            .build()
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string().contains("transport"),
            "expected transport error, got: {err}"
        );
    }

    #[test]
    fn builder_rejects_missing_role() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let err = EngineBuilder::new(&config, "worker")
            .transport(transport)
            .build()
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string().contains("role"),
            "expected role error, got: {err}"
        );
    }

    #[test]
    fn builder_rejects_unknown_role_name() {
        let config = test_config("worker", &["input"], &[]);
        let transport = Arc::new(InMemoryTransport::new(&["input"], ""));
        let err = EngineBuilder::new(&config, "nonexistent")
            .transport(transport)
            .role(Box::new(OkRole))
            .build()
            .map(|_| ())
            .unwrap_err();
        let full_err = format!("{err:#}");
        assert!(
            full_err.contains("not found"),
            "expected not-found error, got: {full_err}"
        );
    }

    #[test]
    fn builder_generates_unique_consumer_ids() {
        let config = test_config("worker", &["input"], &[]);
        let t1 = Arc::new(InMemoryTransport::new(&["input"], ""));
        let t2 = Arc::new(InMemoryTransport::new(&["input"], ""));

        let e1 = EngineBuilder::new(&config, "worker")
            .transport(t1)
            .role(Box::new(OkRole))
            .build()
            .unwrap();
        let e2 = EngineBuilder::new(&config, "worker")
            .transport(t2)
            .role(Box::new(OkRole))
            .build()
            .unwrap();

        assert_ne!(
            e1.consumer, e2.consumer,
            "auto-generated consumer IDs must be unique"
        );
    }
}
