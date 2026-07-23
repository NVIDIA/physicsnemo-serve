/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod batch;
mod discovery;
mod parent_slots;
mod profile;
mod reservation;
mod reserved_memory;

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use scicomp_rq::{Message, Output, QueueManager};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tokio::sync::Mutex;

use tracing::{debug, info, warn};

use crate::config::{SchedulerRoleConfig, parse_role_config};
use crate::metrics::WorkerMetrics;
use crate::retry_dlq::{LocalFailureTracker, RetryDlqPolicy};
use crate::roles::parent_run_state::{ParentRunStateStore, RedisParentRunStateStore};
use crate::traits::{
    BackgroundTask, BoxFuture, MessageSink, RoleEnv, TaskCriticality, WorkerRole, message_deferred,
};

pub use self::batch::BatchProfile;
use self::batch::SchedulerRequest;
use self::discovery::discover_resources;
pub(crate) use self::discovery::{
    DEFAULT_GPU_DISCOVERY_INTERVAL_SECS, DEFAULT_MEMORY_UTILIZATION_PERCENT,
};
use self::parent_slots::{ParentSlotAcquire, ParentSlotStore, RedisParentSlotStore};
use self::profile::ResourceManager;
use self::reservation::{ReservedResource, ResourceReservationTable, is_reservation_blocked_error};

#[derive(Debug, Clone)]
struct QueuedRequest {
    msg: Message,
    payload: SchedulePayload,
    enqueued_at: Instant,
}

fn failed_run_ids(queued: &QueuedRequest) -> Vec<&str> {
    let mut run_ids = vec![queued.msg.run_id()];
    let mut seen = HashSet::from([queued.msg.run_id()]);

    if let Some(items) = queued
        .payload
        .raw_payload
        .get("items")
        .and_then(JsonValue::as_array)
    {
        for item in items {
            let Some(run_id) = item.get("run_id").and_then(JsonValue::as_str) else {
                continue;
            };
            if !run_id.trim().is_empty() && seen.insert(run_id) {
                run_ids.push(run_id);
            }
        }
    }

    run_ids
}

#[derive(Debug, Default)]
struct SchedulerQueueState {
    queue: VecDeque<QueuedRequest>,
    pending_ids: HashSet<String>,
}

pub(crate) fn scheduler_deferred_error() -> anyhow::Error {
    message_deferred("scheduler: request queued for background scheduling")
}

#[cfg(test)]
pub(crate) fn is_scheduler_deferred_error(error: &anyhow::Error) -> bool {
    crate::traits::is_message_deferred_error(error)
}

#[derive(Debug)]
enum ScheduleDecision {
    Blocked,
    Reserved(Box<ReservedSchedule>),
    Dropped,
}

#[derive(Debug)]
enum ScheduleAttemptOutcome {
    Blocked,
    Dispatched(Box<ReservedSchedule>),
    Dropped,
}

impl ScheduleAttemptOutcome {
    fn as_label(&self) -> &'static str {
        match self {
            Self::Blocked => "blocked",
            Self::Dispatched(_) => "dispatched",
            Self::Dropped => "dropped",
        }
    }
}

/// Payload for schedule messages.
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulePayload {
    /// Filled from stream message metadata.
    #[serde(default)]
    pub run_id: String,
    /// Workflow name used for logging and diagnostics.
    #[serde(default)]
    pub workflow: String,
    /// Generic workflow identifier used by manifest-driven envelopes.
    #[serde(default)]
    pub workflow_id: Option<String>,
    /// Parent run identifier for fanout child items.
    #[serde(default)]
    pub parent_run_id: Option<String>,
    /// Optional fanout profile used for parent-level concurrency limits.
    #[serde(default)]
    pub fanout_profile: Option<ScheduleFanoutProfile>,
    /// Optional per-request batching hints from prepare hooks.
    #[serde(default)]
    pub batch_profile: Option<BatchProfile>,
    /// Original request payload preserved for dispatch.
    #[serde(skip, default = "default_raw_payload")]
    pub raw_payload: JsonValue,
    /// Optional resource profile embedded by upstream stages.
    #[serde(default)]
    pub resource_profile: Option<ScheduleResourceProfile>,
    /// Number of GPUs required, derived from `resource_profile`.
    #[serde(default)]
    pub gpus_required: usize,
    /// Memory required per selected worker (MB), derived from `resource_profile`.
    #[serde(default)]
    pub memory_mb: u64,
    /// Stage name to emit on the GPU stream.
    #[serde(skip, default = "default_dispatch_stage")]
    pub dispatch_stage: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScheduleResourceProfile {
    #[serde(default)]
    pub gpus_required: Option<usize>,
    #[serde(default)]
    pub memory_mb: Option<u64>,
    #[serde(default)]
    pub executor_class: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScheduleFanoutProfile {
    #[serde(default)]
    pub item_count: Option<usize>,
    #[serde(default)]
    pub max_in_flight: Option<usize>,
}

#[derive(Debug, Clone)]
struct PendingSchedule {
    source_msg: Message,
    payload: SchedulePayload,
    ack_after_dispatch: Vec<Message>,
}

#[derive(Debug, Clone)]
struct ReservedSchedule {
    source_msg: Message,
    payload: SchedulePayload,
    gpu_targets: Vec<ReservedResource>,
    ack_after_dispatch: Vec<Message>,
    held_parent_slot: Option<String>,
}

fn default_raw_payload() -> JsonValue {
    serde_json::json!({})
}

fn default_dispatch_stage() -> String {
    "execute".to_string()
}

/// Payload for release messages.
#[derive(Debug, Clone, Deserialize)]
struct ReleasePayload {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    parent_run_id: Option<String>,
    #[serde(default)]
    memory_mb: u64,
    resource_id: u32,
}

/// Scheduler role with remote-style discovery and memory-aware best-fit routing.
#[derive(Clone)]
pub struct SchedulerRole {
    /// Tracks discovered worker capacity and active GPU reservations.
    reservations: ResourceReservationTable,
    /// Enforces per-parent in-flight limits for fanout workloads.
    parent_slots: Arc<dyn ParentSlotStore>,
    /// Checks whether a parent run has already reached a terminal state.
    parent_state: Arc<dyn ParentRunStateStore>,
    /// In-memory FIFO queue plus dedupe index for pending schedule requests.
    scheduler_queue_state: Arc<Mutex<SchedulerQueueState>>,
    /// Local retry counter for queued requests handled by the background task.
    request_failures: LocalFailureTracker,
    /// Optional Prometheus metrics registry for scheduler observability.
    metrics: Option<WorkerMetrics>,
    /// Physical stream prefix used when converting between logical and routed streams.
    stream_prefix: String,
    /// Logical input stream name for schedule requests.
    schedule_stream: String,
    /// Logical input stream name for release events.
    release_stream: String,
    /// Shared retry budget and DLQ target for scheduler background failures.
    retry_dlq_policy: RetryDlqPolicy,
    /// Parsed role configuration retained for runtime behavior and test access.
    config: SchedulerRoleConfig,
}

impl SchedulerRole {
    fn normalized_memory_utilization_percent(percent: u64) -> u64 {
        if (1..=100).contains(&percent) {
            percent
        } else {
            DEFAULT_MEMORY_UTILIZATION_PERCENT
        }
    }

    fn normalized_resource_discovery_interval_secs(interval_secs: u64) -> u64 {
        if interval_secs > 0 {
            interval_secs
        } else {
            DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
        }
    }

    fn resolve_input_streams(env: &RoleEnv) -> Result<(String, String)> {
        let schedule_stream = env
            .inputs
            .first()
            .map(|spec| spec.stream.trim().to_string())
            .ok_or_else(|| {
                anyhow!("scheduler role requires two input streams: schedule and release")
            })?;
        let release_stream = env
            .inputs
            .get(1)
            .map(|spec| spec.stream.trim().to_string())
            .ok_or_else(|| {
                anyhow!("scheduler role requires two input streams: schedule and release")
            })?;

        if schedule_stream.is_empty() || release_stream.is_empty() {
            return Err(anyhow!(
                "scheduler role inputs must include non-empty schedule and release stream names"
            ));
        }

        Ok((schedule_stream, release_stream))
    }

    pub fn from_env(
        env: &RoleEnv,
        qm: QueueManager,
        retry_dlq_policy: RetryDlqPolicy,
        metrics: Option<WorkerMetrics>,
    ) -> Result<(Self, Vec<Box<dyn BackgroundTask>>)> {
        let parent_slots = Arc::new(RedisParentSlotStore::new(qm.clone(), "parent_slots"));
        let parent_state = Arc::new(RedisParentRunStateStore::new(qm.clone()));
        Self::build_with_dependencies(
            env,
            qm,
            parent_slots,
            parent_state,
            retry_dlq_policy,
            metrics,
        )
    }

    fn build_with_dependencies(
        env: &RoleEnv,
        qm: QueueManager,
        parent_slots: Arc<dyn ParentSlotStore>,
        parent_state: Arc<dyn ParentRunStateStore>,
        retry_dlq_policy: RetryDlqPolicy,
        metrics: Option<WorkerMetrics>,
    ) -> Result<(Self, Vec<Box<dyn BackgroundTask>>)> {
        let mut config: SchedulerRoleConfig = parse_role_config(env.role_config.as_ref())?;

        config.memory_utilization_percent =
            Self::normalized_memory_utilization_percent(config.memory_utilization_percent);
        config.gpu_discovery_interval_secs =
            Self::normalized_resource_discovery_interval_secs(config.gpu_discovery_interval_secs);

        ResourceManager::warm_known_profile_resources_cache();

        let (schedule_stream, release_stream) = Self::resolve_input_streams(env)?;

        let reservations =
            ResourceReservationTable::new(qm.clone(), config.memory_utilization_percent);

        let interval = Duration::from_secs(config.gpu_discovery_interval_secs);

        let discovery_task: Box<dyn BackgroundTask> = Box::new(ResourceDiscoveryTask {
            reservations: reservations.clone(),
            interval,
            qm: Box::new(qm),
            registry_key: config.gpu_registry_key.clone(),
            metrics: metrics.clone(),
        });

        let role = Self {
            reservations,
            parent_slots,
            parent_state,
            scheduler_queue_state: Arc::new(Mutex::new(SchedulerQueueState::default())),
            request_failures: LocalFailureTracker::default(),
            metrics,
            stream_prefix: env.stream_prefix.clone(),
            schedule_stream,
            release_stream,
            retry_dlq_policy,
            config,
        };

        let scheduler_task: Box<dyn BackgroundTask> =
            Box::new(SchedulerTask { role: role.clone() });

        info!(
            gpu_registry_key = %role.config.gpu_registry_key,
            memory_utilization_percent = role.config.memory_utilization_percent,
            gpu_discovery_interval_secs = role.config.gpu_discovery_interval_secs,
            schedule_stream = %role.schedule_stream,
            release_stream = %role.release_stream,
            "Scheduler config loaded"
        );

        Ok((role, vec![discovery_task, scheduler_task]))
    }

    async fn apply_release(&self, payload: &ReleasePayload) -> Result<()> {
        self.reservations
            .release(payload.resource_id, payload.memory_mb)
            .await
            .with_context(|| {
                format!(
                    "failed to release allocation for run_id '{}' on resource {}",
                    payload.run_id, payload.resource_id
                )
            })?;

        if let Some(parent_run_id) = payload.parent_run_id.as_deref()
            && !parent_run_id.trim().is_empty()
        {
            let _ = self.parent_slots.release(parent_run_id).await?;
        }

        Ok(())
    }

    fn logical_stream_for_route(&self, route: &str) -> String {
        route
            .strip_prefix(self.stream_prefix.as_str())
            .unwrap_or(route)
            .to_string()
    }

    fn queue_key(msg: &Message) -> String {
        format!("{}::{}", msg.stream(), msg.id())
    }

    /// Record one scheduler decision attempt after its outcome is known.
    fn record_scheduler_attempt_outcome(&self, outcome: &str) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.record_scheduler_attempt(outcome);
        }
    }

    /// Increment queue depth when a schedule message is accepted into local FIFO.
    fn increment_scheduler_queue_depth(&self) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.increment_scheduler_queue_depth();
        }
    }

    /// Decrement queue depth when a queued schedule message is removed.
    fn decrement_scheduler_queue_depth(&self) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.decrement_scheduler_queue_depth();
        }
    }

    /// Observe latency for a single scheduler decision attempt.
    fn record_scheduler_attempt_duration(&self, outcome: &str, attempt_duration_seconds: f64) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.observe_scheduler_attempt_duration(outcome, attempt_duration_seconds);
        }
    }

    /// Observe total queue wait once a queued request reaches a terminal outcome.
    fn record_scheduler_queue_wait(&self, outcome: &str, queue_wait_seconds: f64) {
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.observe_scheduler_queue_wait(outcome, queue_wait_seconds);
        }
    }

    async fn enqueue_request(&self, msg: &Message, payload: SchedulePayload) -> bool {
        let queue_key = Self::queue_key(msg);
        let mut state = self.scheduler_queue_state.lock().await;
        if !state.pending_ids.insert(queue_key) {
            debug!(
                msg_id = msg.id(),
                run_id = msg.run_id(),
                stream = msg.stream(),
                queue_depth = state.queue.len(),
                "scheduler request already queued; ignoring duplicate enqueue"
            );
            return false;
        }

        state.queue.push_back(QueuedRequest {
            msg: msg.clone(),
            payload,
            enqueued_at: Instant::now(),
        });
        info!(
            msg_id = msg.id(),
            run_id = msg.run_id(),
            stream = msg.stream(),
            queue_depth = state.queue.len(),
            "queued scheduler request for background scheduling"
        );
        self.increment_scheduler_queue_depth();
        true
    }

    async fn finish_request(&self, msgs: &[Message]) {
        if msgs.is_empty() {
            return;
        }
        let queue_keys: HashSet<String> = msgs.iter().map(Self::queue_key).collect();
        let mut state = self.scheduler_queue_state.lock().await;
        let previous_len = state.queue.len();
        state
            .queue
            .retain(|queued| !queue_keys.contains(Self::queue_key(&queued.msg).as_str()));
        for queue_key in &queue_keys {
            state.pending_ids.remove(queue_key.as_str());
        }
        let removed_count = previous_len.saturating_sub(state.queue.len());
        for _ in 0..removed_count {
            self.decrement_scheduler_queue_depth();
        }
    }

    fn build_dispatch_outputs(
        &self,
        gpu_targets: &[ReservedResource],
        run_id: &str,
        payload: &SchedulePayload,
    ) -> Result<Vec<Output>> {
        let mut outputs = Vec::with_capacity(gpu_targets.len());
        for gpu_target in gpu_targets {
            let mut merged_payload = payload.raw_payload.clone();
            if !merged_payload.is_object() {
                merged_payload = serde_json::json!({ "raw": merged_payload });
            }
            if let Some(map) = merged_payload.as_object_mut() {
                map.insert(
                    "resource_id".to_string(),
                    serde_json::Value::Number(gpu_target.resource_id.into()),
                );
                map.insert(
                    "memory_mb".to_string(),
                    serde_json::Value::Number(payload.memory_mb.into()),
                );
                let should_inject_resource_profile = map
                    .get("resource_profile")
                    .is_none_or(serde_json::Value::is_null);
                if should_inject_resource_profile
                    && let Some(profile) = payload.resource_profile.as_ref()
                {
                    map.insert(
                        "resource_profile".to_string(),
                        schedule_resource_profile_json(profile),
                    );
                }
            }
            advance_stage_context_for_dispatch(
                &mut merged_payload,
                payload.dispatch_stage.as_str(),
            );

            let stream = self.logical_stream_for_route(gpu_target.stream_name.as_str());
            let encoded_payload = merged_payload.to_string();
            outputs.push(
                Output::new(stream, encoded_payload)
                    .with_run_id(run_id.to_string())
                    .with_stage(payload.dispatch_stage.clone()),
            );
        }
        Ok(outputs)
    }

    async fn dispatch_reserved_schedule(
        &self,
        reserved: &ReservedSchedule,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let outputs = match self.build_dispatch_outputs(
            &reserved.gpu_targets,
            reserved.payload.run_id.as_str(),
            &reserved.payload,
        ) {
            Ok(outputs) => outputs,
            Err(error) => {
                self.rollback_reserved_schedule(reserved).await;
                return Err(error);
            }
        };
        let mut source_messages = Vec::with_capacity(1 + reserved.ack_after_dispatch.len());
        source_messages.push(reserved.source_msg.clone());
        source_messages.extend(reserved.ack_after_dispatch.iter().cloned());
        if let Err(dispatch_error) = sink.forward_many_from(&source_messages, &outputs).await {
            self.rollback_reserved_schedule(reserved).await;
            return Err(dispatch_error).context("scheduler: failed to forward scheduled outputs");
        }

        info!(
            run_id = %reserved.payload.run_id,
            workflow = %reserved.payload.workflow,
            dispatch_stage = %reserved.payload.dispatch_stage,
            memory_mb = reserved.payload.memory_mb,
            target_count = reserved.gpu_targets.len(),
            gpu_targets = ?reserved.gpu_targets,
            batch_size = reserved
                .payload
                .raw_payload
                .get("batch_info")
                .and_then(|info| info.get("batch_size"))
                .and_then(serde_json::Value::as_u64),
            "scheduler request dispatched successfully"
        );
        Ok(())
    }

    async fn rollback_reserved_schedule(&self, reserved: &ReservedSchedule) {
        for gpu_target in &reserved.gpu_targets {
            if let Err(release_error) = self
                .reservations
                .release(gpu_target.resource_id, reserved.payload.memory_mb)
                .await
            {
                warn!(
                    run_id = %reserved.payload.run_id,
                    resource_id = gpu_target.resource_id,
                    stream = %gpu_target.stream_name,
                    memory_mb = reserved.payload.memory_mb,
                    error = %release_error,
                    "scheduler rollback release failed after dispatch error"
                );
            } else {
                debug!(
                    run_id = %reserved.payload.run_id,
                    stream = %gpu_target.stream_name,
                    "scheduler rollback released reserved memory after dispatch error"
                );
            }
        }

        if let Some(parent_run_id) = reserved.held_parent_slot.as_deref()
            && let Err(release_error) = self.parent_slots.release(parent_run_id).await
        {
            warn!(
                run_id = %reserved.payload.run_id,
                parent_run_id,
                error = %release_error,
                "scheduler failed to release parent slot after dispatch error"
            );
        }
    }

    async fn schedule(&self, mut pending: PendingSchedule) -> Result<ScheduleDecision> {
        info!(
            msg_id = pending.source_msg.id(),
            run_id = pending.source_msg.run_id(),
            workflow = %pending.payload.workflow,
            parent_run_id = ?pending.payload.parent_run_id,
            "attempting queued scheduler request"
        );
        if let Some(parent_run_id) = pending.payload.parent_run_id.as_deref()
            && !parent_run_id.trim().is_empty()
            && self.parent_state.is_terminal(parent_run_id).await?
        {
            debug!(
                msg_id = pending.source_msg.id(),
                run_id = pending.source_msg.run_id(),
                workflow = %pending.payload.workflow,
                parent_run_id = %parent_run_id,
                "dropping queued scheduler request because parent run is already terminal"
            );
            return Ok(ScheduleDecision::Dropped);
        }

        let held_parent_slot =
            if let Some((parent_run_id, max_in_flight)) = fanout_gate(&pending.payload) {
                match self
                    .parent_slots
                    .try_acquire(parent_run_id, max_in_flight)
                    .await?
                {
                    ParentSlotAcquire::Acquired { .. } => {
                        debug!(
                            msg_id = pending.source_msg.id(),
                            run_id = pending.source_msg.run_id(),
                            workflow = %pending.payload.workflow,
                            parent_run_id = %parent_run_id,
                            max_in_flight,
                            "acquired parent slot for queued scheduler request"
                        );
                        Some(parent_run_id.to_string())
                    }
                    ParentSlotAcquire::Saturated { .. } => {
                        debug!(
                            msg_id = pending.source_msg.id(),
                            run_id = pending.source_msg.run_id(),
                            workflow = %pending.payload.workflow,
                            parent_run_id = %parent_run_id,
                            max_in_flight,
                            "scheduler request blocked because parent slot is saturated"
                        );
                        return Ok(ScheduleDecision::Blocked);
                    }
                }
            } else {
                None
            };

        let gpu_targets = self.reservations.reserve(&mut pending.payload).await;
        let gpu_targets = match gpu_targets {
            Ok(targets) => targets,
            Err(error) => {
                if let Some(parent_run_id) = held_parent_slot.as_deref()
                    && let Err(release_err) = self.parent_slots.release(parent_run_id).await
                {
                    warn!(
                        parent_run_id = %parent_run_id,
                        error = %release_err,
                        "scheduler: failed to release parent slot after reservation error"
                    );
                }

                if is_reservation_blocked_error(&error) {
                    info!(
                        msg_id = pending.source_msg.id(),
                        run_id = pending.source_msg.run_id(),
                        workflow = %pending.payload.workflow,
                        error = %error,
                        "scheduler request remains queued because reservation is currently blocked"
                    );
                    return Ok(ScheduleDecision::Blocked);
                }

                return Err(anyhow!(
                    "failed to reserve GPUs for schedule payload: {error}"
                ));
            }
        };

        Ok(ScheduleDecision::Reserved(Box::new(ReservedSchedule {
            source_msg: pending.source_msg,
            payload: pending.payload,
            gpu_targets,
            ack_after_dispatch: pending.ack_after_dispatch,
            held_parent_slot,
        })))
    }

    async fn execute_schedule_decision(
        &self,
        decision: ScheduleDecision,
        queued: &QueuedRequest,
        sink: &dyn MessageSink,
    ) -> Result<ScheduleAttemptOutcome> {
        match decision {
            ScheduleDecision::Blocked => Ok(ScheduleAttemptOutcome::Blocked),
            ScheduleDecision::Dropped => {
                sink.ack_message(&queued.msg)
                    .await
                    .context("scheduler: failed to ack terminal parent child request")?;
                Ok(ScheduleAttemptOutcome::Dropped)
            }
            ScheduleDecision::Reserved(reserved) => {
                self.dispatch_reserved_schedule(&reserved, sink).await?;
                Ok(ScheduleAttemptOutcome::Dispatched(reserved))
            }
        }
    }

    async fn next_request(&self) -> Result<SchedulerRequest> {
        let mut state = self.scheduler_queue_state.lock().await;
        let Some(head) = state.queue.front().cloned() else {
            return Ok(SchedulerRequest::Empty);
        };
        if let Some(batch_request) = self.create_batch_request(&mut state, head.clone()).await? {
            return Ok(batch_request);
        }
        Ok(SchedulerRequest::Single(head))
    }

    async fn process_next_request(&self, sink: &dyn MessageSink) -> Result<()> {
        let request = self.next_request().await?;
        let attempt_started = Instant::now();

        // Normalize a single request or formed batch into the same scheduling input.
        let (queued, queue_wait_seconds, pending_schedule) = match request {
            SchedulerRequest::Empty => return Ok(()),

            SchedulerRequest::Wait(queued) => {
                debug!(
                    msg_id = queued.msg.id(),
                    run_id = queued.msg.run_id(),
                    "scheduler queue head is waiting for batch flush threshold"
                );
                return Ok(());
            }

            SchedulerRequest::Single(queued) => {
                let queue_wait_seconds = queued.enqueued_at.elapsed().as_secs_f64();
                let pending_schedule = PendingSchedule {
                    source_msg: queued.msg.clone(),
                    payload: queued.payload.clone(),
                    ack_after_dispatch: Vec::new(),
                };
                (queued, queue_wait_seconds, Ok(pending_schedule))
            }

            SchedulerRequest::Batch(request_batch) => {
                let queued = request_batch.head.clone();
                let queue_wait_seconds = request_batch.head.enqueued_at.elapsed().as_secs_f64();
                (queued, queue_wait_seconds, request_batch.prepare_schedule())
            }
        };

        // Reserve resources, then perform the resulting dispatch or acknowledgement.
        let schedule_decision = match pending_schedule {
            Ok(pending) => self.schedule(pending).await,
            Err(error) => Err(error),
        };
        let schedule_result = match schedule_decision {
            Ok(decision) => {
                self.execute_schedule_decision(decision, &queued, sink)
                    .await
            }
            Err(error) => Err(error),
        };

        // Record the complete attempt, including preparation and dispatch.
        self.record_scheduler_attempt_duration(
            schedule_result
                .as_ref()
                .map(|outcome| outcome.as_label())
                .unwrap_or("failed"),
            attempt_started.elapsed().as_secs_f64(),
        );

        let outcome_label = match schedule_result.as_ref() {
            Ok(outcome) => outcome.as_label(),
            Err(_) => "failed",
        };
        self.record_scheduler_attempt_outcome(outcome_label);

        // Finalize successful work, leave blocked work queued, or apply retry policy.
        match schedule_result {
            Ok(ScheduleAttemptOutcome::Blocked) => {
                info!(
                    msg_id = queued.msg.id(),
                    run_id = queued.msg.run_id(),
                    "scheduler queue head remains blocked; leaving request pending"
                );
            }
            Ok(ScheduleAttemptOutcome::Dispatched(reserved)) => {
                self.record_scheduler_queue_wait(outcome_label, queue_wait_seconds);
                let mut completed_messages = reserved.ack_after_dispatch;
                completed_messages.push(reserved.source_msg);
                for msg in &completed_messages {
                    self.request_failures.clear(msg);
                }
                self.finish_request(&completed_messages).await;
                info!(
                    msg_id = queued.msg.id(),
                    run_id = queued.msg.run_id(),
                    "scheduler request finished and was removed from the queue"
                );
                return Ok(());
            }
            Ok(ScheduleAttemptOutcome::Dropped) => {
                self.record_scheduler_queue_wait(outcome_label, queue_wait_seconds);
                self.request_failures.clear(&queued.msg);
                self.finish_request(std::slice::from_ref(&queued.msg)).await;
                info!(
                    msg_id = queued.msg.id(),
                    run_id = queued.msg.run_id(),
                    "scheduler request finished and was removed from the queue"
                );
                return Ok(());
            }
            Err(err) => {
                let error_text = err.to_string();
                let attempts = self.request_failures.increment(&queued.msg);

                // Keep transient failures at the queue head until retries are exhausted.
                if attempts < self.retry_dlq_policy.max_retries() {
                    warn!(
                        msg_id = queued.msg.id(),
                        run_id = queued.msg.run_id(),
                        attempt = attempts,
                        max_retries = self.retry_dlq_policy.max_retries(),
                        error = %error_text,
                        "scheduler request failed, keeping request at queue head"
                    );
                    return Ok(());
                }

                // Move persistent failures to the DLQ before removing them locally.
                let dlq_payload = match self.retry_dlq_policy.build_dlq_payload(
                    &queued.msg,
                    self.schedule_stream.as_str(),
                    &error_text,
                    attempts,
                ) {
                    Ok(payload) => payload,
                    Err(dlq_payload_err) => {
                        warn!(
                            msg_id = queued.msg.id(),
                            run_id = queued.msg.run_id(),
                            attempt = attempts,
                            max_retries = self.retry_dlq_policy.max_retries(),
                            error = %dlq_payload_err,
                            original_error = %error_text,
                            "scheduler request hit retry limit but DLQ payload construction failed"
                        );
                        return Ok(());
                    }
                };

                match sink
                    .handoff(
                        &queued.msg,
                        self.retry_dlq_policy.dlq_stream(),
                        &dlq_payload,
                        "dlq",
                    )
                    .await
                {
                    Ok(_) => {
                        self.record_scheduler_queue_wait(outcome_label, queue_wait_seconds);
                        for failed_run_id in failed_run_ids(&queued) {
                            if let Err(status_err) = sink.mark_request_failed(failed_run_id).await {
                                warn!(
                                    msg_id = queued.msg.id(),
                                    run_id = failed_run_id,
                                    batch_run_id = queued.msg.run_id(),
                                    error = %status_err,
                                    "scheduler DLQ handoff succeeded but failed to mark run status failed"
                                );
                            }
                        }
                        self.request_failures.clear(&queued.msg);
                        self.finish_request(std::slice::from_ref(&queued.msg)).await;
                        warn!(
                            msg_id = queued.msg.id(),
                            run_id = queued.msg.run_id(),
                            attempt = attempts,
                            max_retries = self.retry_dlq_policy.max_retries(),
                            dlq_stream = %self.retry_dlq_policy.dlq_stream(),
                            error = %error_text,
                            "scheduler request moved to shared DLQ after retry limit"
                        );
                        return Ok(());
                    }
                    Err(dlq_err) => {
                        warn!(
                            msg_id = queued.msg.id(),
                            run_id = queued.msg.run_id(),
                            attempt = attempts,
                            max_retries = self.retry_dlq_policy.max_retries(),
                            dlq_stream = %self.retry_dlq_policy.dlq_stream(),
                            error = %dlq_err,
                            original_error = %error_text,
                            "scheduler request hit retry limit but DLQ handoff failed; keeping request queued"
                        );
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    async fn queued_request_count(&self) -> usize {
        self.scheduler_queue_state.lock().await.queue.len()
    }
}

impl WorkerRole for SchedulerRole {
    fn name(&self) -> &'static str {
        "scheduler"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        _sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if stream == self.schedule_stream.as_str() {
                let schedule_payload = decode_schedule_payload(msg.payload(), msg.run_id())?;
                info!(
                    msg_id = msg.id(),
                    run_id = msg.run_id(),
                    workflow = %schedule_payload.workflow,
                    stream = %stream,
                    "received scheduler request on schedule stream"
                );
                self.enqueue_request(msg, schedule_payload).await;
                return Err(scheduler_deferred_error());
            }

            if stream == self.release_stream.as_str() {
                let release_payload = decode_release_payload(msg.payload())?;
                info!(
                    msg_id = msg.id(),
                    run_id = msg.run_id(),
                    stream = %stream,
                    release_payload = ?release_payload,
                    "received scheduler release payload"
                );
                return self.apply_release(&release_payload).await;
            }

            Err(anyhow!("scheduler: unexpected stream '{stream}'"))
        })
    }
}

/// Attempts one queued schedule request per engine loop.
struct SchedulerTask {
    role: SchedulerRole,
}

impl BackgroundTask for SchedulerTask {
    fn name(&self) -> &'static str {
        "scheduler_task"
    }

    fn interval(&self) -> Duration {
        Duration::ZERO
    }

    fn criticality(&self) -> TaskCriticality {
        TaskCriticality::BestEffort
    }

    fn run<'a>(&'a self, sink: &'a dyn MessageSink) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.role.process_next_request(sink).await })
    }
}

/// Periodic GPU discovery task.
struct ResourceDiscoveryTask {
    reservations: ResourceReservationTable,
    interval: Duration,
    qm: Box<QueueManager>,
    registry_key: String,
    metrics: Option<WorkerMetrics>,
}

impl BackgroundTask for ResourceDiscoveryTask {
    fn name(&self) -> &'static str {
        "resource_discovery"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn criticality(&self) -> TaskCriticality {
        TaskCriticality::BestEffort
    }

    fn run<'a>(&'a self, _sink: &'a dyn MessageSink) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let qm = self.qm.clone();
            let registry_key = self.registry_key.clone();
            debug!(
                registry_key = %registry_key,
                interval_secs = self.interval.as_secs(),
                "running scheduler resource discovery task"
            );
            tokio::time::timeout(Duration::from_secs(30), async move {
                let update = discover_resources(&qm, registry_key.as_str()).await?;
                self.reservations.sync_from_discovery_update(update).await
            })
            .await
            .map_err(|_| anyhow!("GPU discovery timed out after 30s"))??;
            if let Some(metrics) = self.metrics.as_ref() {
                metrics.set_scheduler_discovered_workers(
                    self.reservations.tracked_worker_count().await,
                );
            }
            debug!(
                registry_key = %self.registry_key,
                "scheduler resource discovery task completed"
            );
            Ok(())
        })
    }
}

fn decode_schedule_payload(raw_payload: &str, run_id: &str) -> Result<SchedulePayload> {
    if raw_payload.trim().is_empty() {
        return Err(anyhow!("scheduler: empty payload on schedule stream"));
    }

    let parsed_payload: JsonValue = serde_json::from_str(raw_payload)
        .context("scheduler: schedule payload is not valid JSON")?;

    let mut payload: SchedulePayload = serde_json::from_value(parsed_payload.clone())
        .context("scheduler: invalid schedule payload JSON")?;
    payload.run_id = run_id.to_string();
    payload.raw_payload = parsed_payload;
    payload.dispatch_stage = infer_dispatch_stage(&payload);

    if payload.workflow.trim().is_empty()
        && let Some(workflow_id) = payload.workflow_id.as_deref()
    {
        payload.workflow = workflow_id.trim().to_string();
    }

    if payload.workflow.trim().is_empty() {
        return Err(anyhow!(
            "scheduler: workflow is required and must be non-empty"
        ));
    }
    validate_schedule_resource_profile(&payload)?;

    Ok(payload)
}

fn infer_dispatch_stage(payload: &SchedulePayload) -> String {
    let _ = payload;
    "execute".to_string()
}

fn validate_schedule_resource_profile(payload: &SchedulePayload) -> Result<()> {
    let Some(profile) = payload.resource_profile.as_ref() else {
        return Ok(());
    };

    match profile.gpus_required {
        Some(value) if value > 0 => {}
        Some(_) => {
            return Err(anyhow!(
                "scheduler: resource_profile.gpus_required must be at least 1"
            ));
        }
        None => {
            return Err(anyhow!(
                "scheduler: resource_profile.gpus_required is required"
            ));
        }
    }

    match profile.memory_mb {
        Some(value) if value > 0 => {}
        Some(_) => {
            return Err(anyhow!(
                "scheduler: resource_profile.memory_mb must be greater than 0"
            ));
        }
        None => {
            return Err(anyhow!("scheduler: resource_profile.memory_mb is required"));
        }
    }

    match profile.executor_class.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => {}
        _ => {
            return Err(anyhow!(
                "scheduler: resource_profile.executor_class is required"
            ));
        }
    }

    Ok(())
}

fn schedule_resource_profile_json(profile: &ScheduleResourceProfile) -> JsonValue {
    let mut encoded = serde_json::Map::new();

    if let Some(gpus_required) = profile.gpus_required {
        encoded.insert(
            "gpus_required".to_string(),
            serde_json::json!(gpus_required),
        );
    }
    if let Some(memory_mb) = profile.memory_mb {
        encoded.insert("memory_mb".to_string(), serde_json::json!(memory_mb));
    }
    if let Some(executor_class) = profile
        .executor_class
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        encoded.insert(
            "executor_class".to_string(),
            JsonValue::String(executor_class.to_string()),
        );
    }
    if let Some(tags) = profile.tags.as_ref()
        && !tags.is_empty()
    {
        encoded.insert(
            "tags".to_string(),
            JsonValue::Array(tags.iter().cloned().map(JsonValue::String).collect()),
        );
    }

    JsonValue::Object(encoded)
}

fn fanout_gate(payload: &SchedulePayload) -> Option<(&str, usize)> {
    let parent_run_id = payload.parent_run_id.as_deref()?.trim();
    if parent_run_id.is_empty() {
        return None;
    }
    let max_in_flight = payload.fanout_profile.as_ref()?.max_in_flight?;
    if max_in_flight == 0 {
        return None;
    }
    Some((parent_run_id, max_in_flight))
}

fn advance_stage_context_for_dispatch(payload: &mut JsonValue, dispatch_stage: &str) {
    let Some(root) = payload.as_object_mut() else {
        return;
    };
    let Some(stage_context) = root
        .get_mut("stage_context")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };

    let Some(pipeline) = stage_context
        .get("pipeline")
        .and_then(serde_json::Value::as_array)
    else {
        stage_context.insert(
            "current_stage_id".to_string(),
            serde_json::Value::String(dispatch_stage.to_string()),
        );
        stage_context.insert(
            "current_phase".to_string(),
            serde_json::Value::String(dispatch_stage.to_string()),
        );
        return;
    };

    let next_stage = stage_context
        .get("current_stage_id")
        .and_then(serde_json::Value::as_str)
        .and_then(|current_stage_id| {
            pipeline
                .iter()
                .find(|stage| {
                    stage.get("id").and_then(serde_json::Value::as_str) == Some(current_stage_id)
                })
                .and_then(|stage| stage.get("next").and_then(serde_json::Value::as_str))
                .and_then(|next_id| {
                    pipeline.iter().find(|stage| {
                        stage.get("id").and_then(serde_json::Value::as_str) == Some(next_id)
                    })
                })
        })
        .or_else(|| {
            pipeline.iter().find(|stage| {
                stage.get("phase").and_then(serde_json::Value::as_str) == Some(dispatch_stage)
            })
        });

    let Some(next_stage) = next_stage else {
        return;
    };
    let next_id = next_stage
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);
    let next_phase = next_stage
        .get("phase")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);

    if let Some(next_id) = next_id {
        stage_context.insert(
            "current_stage_id".to_string(),
            serde_json::Value::String(next_id),
        );
    }
    if let Some(next_phase) = next_phase {
        stage_context.insert(
            "current_phase".to_string(),
            serde_json::Value::String(next_phase),
        );
    }

    if let Some(items) = root
        .get_mut("items")
        .and_then(serde_json::Value::as_array_mut)
    {
        for item in items {
            let Some(item_payload) = item.get_mut("payload") else {
                continue;
            };
            advance_stage_context_for_dispatch(item_payload, dispatch_stage);
        }
    }
}

fn decode_release_payload(raw_payload: &str) -> Result<ReleasePayload> {
    if raw_payload.trim().is_empty() {
        return Err(anyhow!("scheduler: empty payload on release stream"));
    }
    let payload_json: JsonValue =
        serde_json::from_str(raw_payload).context("scheduler: invalid release payload JSON")?;
    if payload_json
        .as_object()
        .and_then(|map| map.get("resource_id"))
        .is_none_or(JsonValue::is_null)
    {
        return Err(anyhow!(
            "scheduler: release payload must include resource_id and memory_mb"
        ));
    }
    let payload: ReleasePayload =
        serde_json::from_value(payload_json).context("scheduler: invalid release payload JSON")?;

    if payload.memory_mb == 0 {
        return Err(anyhow!(
            "scheduler: release payload must include resource_id and memory_mb"
        ));
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    use crate::config::InputStreamSpec;
    use crate::roles::parent_run_state::InMemoryParentRunStateStore;
    use crate::roles::scheduler::parent_slots::InMemoryParentSlotStore;
    use crate::test_env;
    use serde_json::json;
    use std::collections::HashSet;
    use std::panic::Location;
    use std::sync::Mutex;
    use test_support::{
        TestRedisServer, seed_registry_from_discovery_json, spawn_test_queue_manager,
    };

    #[track_caller]
    fn test_name_with_caller(test_name: &str) -> String {
        let caller = Location::caller();
        format!("{test_name} @ {}:{}", caller.file(), caller.line())
    }

    #[track_caller]
    fn scheduler_with_test_queue_manager<'a>(
        test_name: &'a str,
        env: &'a RoleEnv,
    ) -> impl std::future::Future<
        Output = (TestRedisServer, SchedulerRole, Vec<Box<dyn BackgroundTask>>),
    > + 'a {
        let test_name = test_name_with_caller(test_name);
        async move {
            scheduler_with_test_queue_manager_and_policy(
                test_name.as_str(),
                env,
                retry_dlq_policy(5),
            )
            .await
        }
    }

    async fn scheduler_with_test_queue_manager_and_policy(
        test_name: &str,
        env: &RoleEnv,
        retry_dlq_policy: RetryDlqPolicy,
    ) -> (TestRedisServer, SchedulerRole, Vec<Box<dyn BackgroundTask>>) {
        let (server, qm) = spawn_test_queue_manager(test_name).await;
        let (role, tasks) =
            SchedulerRole::from_env(env, qm.clone(), retry_dlq_policy, None).unwrap();
        seed_registry_from_discovery_json(
            &qm,
            role.config.gpu_registry_key.as_str(),
            "scheduler-test-worker",
        )
        .await;
        (server, role, tasks)
    }

    #[track_caller]
    fn scheduler_with_test_queue_manager_and_dependencies<'a>(
        test_name: &'a str,
        env: &'a RoleEnv,
        parent_slots: Arc<dyn ParentSlotStore>,
        parent_state: Arc<dyn ParentRunStateStore>,
        retry_dlq_policy: RetryDlqPolicy,
    ) -> impl std::future::Future<
        Output = (TestRedisServer, SchedulerRole, Vec<Box<dyn BackgroundTask>>),
    > + 'a {
        let test_name = test_name_with_caller(test_name);
        async move {
            let (server, qm) = spawn_test_queue_manager(test_name.as_str()).await;
            let (role, tasks) = SchedulerRole::build_with_dependencies(
                env,
                qm.clone(),
                parent_slots,
                parent_state,
                retry_dlq_policy,
                None,
            )
            .unwrap();
            seed_registry_from_discovery_json(
                &qm,
                role.config.gpu_registry_key.as_str(),
                "scheduler-test-worker",
            )
            .await;
            (server, role, tasks)
        }
    }

    #[derive(Debug, Clone)]
    struct SinkRecord {
        stream_key: String,
        run_id: String,
        payload: String,
        stage: String,
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct HandoffRecord {
        stream: String,
        payload: String,
        stage: String,
    }

    #[derive(Default)]
    struct TrackingParentSlotStore {
        released_parent_run_ids: Mutex<Vec<String>>,
    }

    impl TrackingParentSlotStore {
        fn new() -> Self {
            Self {
                released_parent_run_ids: Mutex::new(Vec::new()),
            }
        }

        fn released_parent_run_ids(&self) -> Vec<String> {
            self.released_parent_run_ids
                .lock()
                .expect("tracking lock poisoned")
                .clone()
        }
    }

    impl ParentSlotStore for TrackingParentSlotStore {
        fn try_acquire<'a>(
            &'a self,
            _parent_run_id: &'a str,
            _max_in_flight: usize,
        ) -> BoxFuture<'a, Result<ParentSlotAcquire>> {
            Box::pin(async { Ok(ParentSlotAcquire::Acquired { active_count: 1 }) })
        }

        fn release<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<usize>> {
            Box::pin(async move {
                self.released_parent_run_ids
                    .lock()
                    .expect("tracking lock poisoned")
                    .push(parent_run_id.to_string());
                Ok(0)
            })
        }
    }

    #[derive(Default)]
    struct AlwaysFailingParentRunStateStore;

    impl ParentRunStateStore for AlwaysFailingParentRunStateStore {
        fn is_terminal<'a>(&'a self, _parent_run_id: &'a str) -> BoxFuture<'a, Result<bool>> {
            Box::pin(async { Err(anyhow::anyhow!("parent state lookup failed")) })
        }

        fn mark_terminal<'a>(&'a self, _parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    struct RecordingSink {
        writes: Mutex<Vec<SinkRecord>>,
        handoffs: Mutex<Vec<HandoffRecord>>,
        acked_message_ids: Mutex<Vec<String>>,
        failed_run_ids: Mutex<Vec<String>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                writes: Mutex::new(Vec::new()),
                handoffs: Mutex::new(Vec::new()),
                acked_message_ids: Mutex::new(Vec::new()),
                failed_run_ids: Mutex::new(Vec::new()),
            }
        }

        fn writes(&self) -> Vec<SinkRecord> {
            self.writes.lock().expect("recording lock poisoned").clone()
        }

        fn handoffs(&self) -> Vec<HandoffRecord> {
            self.handoffs
                .lock()
                .expect("recording lock poisoned")
                .clone()
        }

        fn acked_ids(&self) -> Vec<String> {
            self.acked_message_ids
                .lock()
                .expect("recording lock poisoned")
                .clone()
        }

        fn failed_run_ids(&self) -> Vec<String> {
            self.failed_run_ids
                .lock()
                .expect("recording lock poisoned")
                .clone()
        }
    }

    impl MessageSink for RecordingSink {
        fn enqueue<'a>(
            &'a self,
            stream: &'a str,
            run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.writes
                    .lock()
                    .expect("recording lock poisoned")
                    .push(SinkRecord {
                        stream_key: stream.to_string(),
                        run_id: run_id.to_string(),
                        payload: payload.to_string(),
                        stage: stage.to_string(),
                    });
                Ok("1-0".to_string())
            })
        }

        fn enqueue_to_stream<'a>(
            &'a self,
            stream_key: &'a str,
            run_id: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.writes
                    .lock()
                    .expect("recording lock poisoned")
                    .push(SinkRecord {
                        stream_key: stream_key.to_string(),
                        run_id: run_id.to_string(),
                        payload: payload.to_string(),
                        stage: stage.to_string(),
                    });
                Ok("1-0".to_string())
            })
        }

        fn ack_message<'a>(&'a self, msg: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.acked_message_ids
                    .lock()
                    .expect("recording lock poisoned")
                    .push(msg.id().to_string());
                Ok(())
            })
        }

        fn handoff<'a>(
            &'a self,
            _msg: &'a scicomp_rq::Message,
            dest_stream: &'a str,
            payload: &'a str,
            stage: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                self.handoffs
                    .lock()
                    .expect("recording lock poisoned")
                    .push(HandoffRecord {
                        stream: dest_stream.to_string(),
                        payload: payload.to_string(),
                        stage: stage.to_string(),
                    });
                Ok("1-0".to_string())
            })
        }

        fn mark_request_failed<'a>(&'a self, run_id: &'a str) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.failed_run_ids
                    .lock()
                    .expect("recording lock poisoned")
                    .push(run_id.to_string());
                Ok(())
            })
        }

        fn forward_many<'a>(
            &'a self,
            msg: &'a scicomp_rq::Message,
            outputs: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async move {
                self.acked_message_ids
                    .lock()
                    .expect("recording lock poisoned")
                    .push(msg.id().to_string());

                let mut writes = self.writes.lock().expect("recording lock poisoned");
                for output in outputs {
                    writes.push(SinkRecord {
                        stream_key: output.stream().to_string(),
                        run_id: output.run_id().unwrap_or(msg.run_id()).to_string(),
                        payload: output.payload().to_string(),
                        stage: output.stage().unwrap_or(msg.stage()).to_string(),
                    });
                }
                Ok(vec![])
            })
        }
    }

    fn scheduler_env(prefix: &str) -> RoleEnv {
        scheduler_env_with_config(prefix, json!({ "batching_enabled": false }))
    }

    fn scheduler_env_with_config(prefix: &str, mut config: serde_json::Value) -> RoleEnv {
        if let Some(map) = config.as_object_mut() {
            map.entry("batching_enabled".to_string())
                .or_insert_with(|| json!(false));
        }
        RoleEnv {
            role_name: "scheduler".to_string(),
            stream_prefix: prefix.to_string(),
            inputs: vec![
                InputStreamSpec {
                    stream: "schedule".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
                InputStreamSpec {
                    stream: "release".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
            ],
            resolved_outputs: vec![],
            role_config: Some(config),
            python_runtime_envs: Default::default(),
        }
    }

    fn scheduler_env_with_batching(prefix: &str, config: serde_json::Value) -> RoleEnv {
        RoleEnv {
            role_name: "scheduler".to_string(),
            stream_prefix: prefix.to_string(),
            inputs: vec![
                InputStreamSpec {
                    stream: "schedule".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
                InputStreamSpec {
                    stream: "release".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
            ],
            resolved_outputs: vec![],
            role_config: Some(config),
            python_runtime_envs: Default::default(),
        }
    }

    fn retry_dlq_policy(max_retries: usize) -> RetryDlqPolicy {
        RetryDlqPolicy::new(max_retries, "shared-dlq")
    }

    fn schedule_msg(run_id: &str, payload: &str) -> scicomp_rq::Message {
        scicomp_rq::Message::new(
            "1-0",
            "test:schedule",
            "schedule:grp",
            run_id,
            payload,
            "schedule",
        )
    }

    fn schedule_msg_with_id(id: &str, run_id: &str, payload: &str) -> scicomp_rq::Message {
        scicomp_rq::Message::new(
            id,
            "test:schedule",
            "schedule:grp",
            run_id,
            payload,
            "schedule",
        )
    }

    fn schedule_msg_for_stream(
        run_id: &str,
        payload: &str,
        logical_stream: &str,
    ) -> scicomp_rq::Message {
        let physical = format!("test:{logical_stream}");
        let group = format!("{logical_stream}:grp");
        scicomp_rq::Message::new("1-0", &physical, &group, run_id, payload, logical_stream)
    }

    fn release_msg(run_id: &str, payload: &str) -> scicomp_rq::Message {
        scicomp_rq::Message::new(
            "2-0",
            "test:release",
            "release:grp",
            run_id,
            payload,
            "release",
        )
    }

    fn release_msg_for_stream(
        run_id: &str,
        payload: &str,
        logical_stream: &str,
    ) -> scicomp_rq::Message {
        let physical = format!("test:{logical_stream}");
        let group = format!("{logical_stream}:grp");
        scicomp_rq::Message::new("2-0", &physical, &group, run_id, payload, logical_stream)
    }

    fn set_env_var(key: &str, value: Option<&str>) {
        test_env::set_env_var(key, value);
    }

    fn basic_schedule_payload(
        workflow: &str,
        executor_class: &str,
        gpus_required: usize,
        memory_mb: u64,
    ) -> String {
        json!({
            "workflow": workflow,
            "workflow_id": workflow,
            "resource_profile": {
                "gpus_required": gpus_required,
                "memory_mb": memory_mb,
                "executor_class": executor_class,
            }
        })
        .to_string()
    }

    fn batchable_schedule_payload(run_id: &str, memory_mb: u64) -> String {
        json!({
            "run_id": run_id,
            "workflow": "demo-batchable",
            "workflow_id": "demo-batchable",
            "operation": "run",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": memory_mb,
                "executor_class": "python.gpu.demo",
                "tags": ["demo"]
            },
            "stage_context": {
                "current_stage_id": "schedule",
                "current_phase": "schedule",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.demo", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            }
        })
        .to_string()
    }

    async fn run_task_by_name_result(
        tasks: &[Box<dyn BackgroundTask>],
        task_name: &str,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let task = tasks
            .iter()
            .find(|task| task.name() == task_name)
            .unwrap_or_else(|| panic!("missing background task '{task_name}'"));
        task.run(sink).await
    }

    async fn run_gpu_discovery(tasks: &[Box<dyn BackgroundTask>], sink: &dyn MessageSink) {
        let raw_json = std::env::var("SCHEDULER_DISCOVERY_JSON").unwrap_or_else(|_| {
            panic!(
                "run_gpu_discovery requires SCHEDULER_DISCOVERY_JSON to be set; \
                 otherwise test registry seeding is skipped and discovery runs against an empty registry"
            )
        });
        assert!(
            !raw_json.trim().is_empty(),
            "run_gpu_discovery requires a non-empty SCHEDULER_DISCOVERY_JSON; \
             otherwise test registry seeding is skipped and discovery runs against an empty registry"
        );
        run_task_by_name_result(tasks, "resource_discovery", sink)
            .await
            .unwrap();
    }

    async fn run_scheduler_task(tasks: &[Box<dyn BackgroundTask>], sink: &dyn MessageSink) {
        run_task_by_name_result(tasks, "scheduler_task", sink)
            .await
            .unwrap();
    }

    async fn queue_schedule(
        role: &SchedulerRole,
        msg: &scicomp_rq::Message,
        stream: &str,
        sink: &dyn MessageSink,
    ) {
        let result = role.handle(msg, stream, sink).await;
        let err = result.expect_err("schedule messages should defer after queueing");
        assert!(
            is_scheduler_deferred_error(&err),
            "expected scheduler deferred error, got: {err:#}"
        );
    }

    #[tokio::test]
    async fn scheduler_batches_default_profile_requests_at_max_size() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "run-a", &batchable_schedule_payload("run-a", 20_000)),
            "schedule",
            &sink,
        )
        .await;
        queue_schedule(
            &role,
            &schedule_msg_with_id("1-1", "run-b", &batchable_schedule_payload("run-b", 20_000)),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["batch_info"]["batch_size"], 2);
        assert_eq!(forwarded["batch_info"]["flush_reason"], "max_batch_size");
        assert_eq!(forwarded["resource_profile"]["memory_mb"], 40_000);
        assert_eq!(forwarded["items"][0]["run_id"], "run-a");
        assert_eq!(forwarded["items"][1]["run_id"], "run-b");
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string(), "1-1".to_string()]);
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_batch_wait_starts_when_request_is_enqueued() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "run-a", &batchable_schedule_payload("run-a", 10_000)),
            "schedule",
            &sink,
        )
        .await;
        {
            let mut state = role.scheduler_queue_state.lock().await;
            let queued = state
                .queue
                .front_mut()
                .expect("queued request should be pending");
            queued.enqueued_at = queued
                .enqueued_at
                .checked_sub(Duration::from_secs(60))
                .expect("test timestamp should support subtracting queue age");
        }
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(
            writes.len(),
            1,
            "an aged queue head should flush immediately"
        );
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["run_id"], "run-a");
        assert!(forwarded.get("batch_info").is_none());
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_processes_max_size_one_batchable_request_as_single() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 1,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "run-a", &batchable_schedule_payload("run-a", 10_000)),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert!(forwarded.get("batch_info").is_none());
        assert_eq!(forwarded["run_id"], "run-a");
        assert_eq!(forwarded["memory_mb"], 10_000);
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string()]);
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_batches_requests_with_different_memory_requirements() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "run-a", &batchable_schedule_payload("run-a", 10_000)),
            "schedule",
            &sink,
        )
        .await;
        queue_schedule(
            &role,
            &schedule_msg_with_id("1-1", "run-b", &batchable_schedule_payload("run-b", 20_000)),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["batch_info"]["batch_size"], 2);
        assert_eq!(forwarded["resource_profile"]["memory_mb"], 30_000);
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_processes_memory_limited_one_item_batch_as_single() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "run-a", &batchable_schedule_payload("run-a", 30_000)),
            "schedule",
            &sink,
        )
        .await;
        queue_schedule(
            &role,
            &schedule_msg_with_id("1-1", "run-b", &batchable_schedule_payload("run-b", 30_000)),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert!(forwarded.get("batch_info").is_none());
        assert_eq!(forwarded["run_id"], "run-a");
        assert_eq!(forwarded["memory_mb"], 30_000);
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string()]);
        assert_eq!(role.queued_request_count().await, 1);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_limits_batch_during_formation_when_full_batch_does_not_fit_memory() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 3,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        for (id, run_id) in [("1-0", "run-a"), ("1-1", "run-b"), ("1-2", "run-c")] {
            queue_schedule(
                &role,
                &schedule_msg_with_id(id, run_id, &batchable_schedule_payload(run_id, 20_000)),
                "schedule",
                &sink,
            )
            .await;
        }
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["batch_info"]["batch_size"], 2);
        assert_eq!(forwarded["batch_info"]["flush_reason"], "memory_fit");
        assert_eq!(forwarded["resource_profile"]["memory_mb"], 40_000);
        assert_eq!(role.queued_request_count().await, 1);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_queues_multiple_batches_on_one_gpu_stream() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        for (id, run_id) in [
            ("1-0", "run-a"),
            ("1-1", "run-b"),
            ("1-2", "run-c"),
            ("1-3", "run-d"),
        ] {
            queue_schedule(
                &role,
                &schedule_msg_with_id(id, run_id, &batchable_schedule_payload(run_id, 1_000)),
                "schedule",
                &sink,
            )
            .await;
        }
        run_scheduler_task(&tasks, &sink).await;
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(sink.writes().len(), 2);
        assert_eq!(role.queued_request_count().await, 0);
        for write in sink.writes() {
            let payload: JsonValue = serde_json::from_str(&write.payload).unwrap();
            assert_eq!(payload["batch_info"]["batch_size"], 2);
            assert_eq!(payload["resource_id"], 0);
        }

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_bypasses_batching_for_fanout_requests() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_batching(
                "test:",
                json!({
                    "memory_utilization_percent": 100,
                    "batching_enabled": true,
                    "max_batch_size": 2,
                    "max_batch_wait_ms": 10_000
                }),
            ),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let mut payload: JsonValue =
            serde_json::from_str(&batchable_schedule_payload("child-a", 1_000)).unwrap();
        payload["parent_run_id"] = json!("parent-a");
        payload["fanout_profile"] = json!({ "max_in_flight": 2 });
        queue_schedule(
            &role,
            &schedule_msg_with_id("1-0", "child-a", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert!(forwarded.get("items").is_none());
        assert_eq!(forwarded["run_id"], "child-a");
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_routes_schedule_payload_and_injects_gpu_fields() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_config("test:", json!({})),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg(
                "run-1",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].stream_key, "gpu:ns:pod:0");
        assert_eq!(writes[0].run_id, "run-1");
        assert_eq!(writes[0].stage, "execute");

        let payload_json: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(payload_json["resource_id"], 0);
        assert_eq!(payload_json["memory_mb"], 20_000);
        assert_eq!(payload_json["workflow"], "demo-deterministic");
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string()]);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_deduplicates_reclaimed_schedule_messages_while_request_is_queued() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let msg = schedule_msg(
            "run-dedupe",
            &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
        );
        queue_schedule(&role, &msg, "schedule", &sink).await;
        queue_schedule(&role, &msg, "schedule", &sink).await;
        assert_eq!(role.queued_request_count().await, 1);

        run_scheduler_task(&tasks, &sink).await;

        assert_eq!(sink.writes().len(), 1);
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string()]);
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[test]
    fn decode_schedule_payload_accepts_plugin_run_envelope() {
        let payload = serde_json::json!({
            "workflow_id": "demo-plugin",
            "resource_profile": {
                "gpus_required": 2,
                "memory_mb": 4096,
                "executor_class": "python.gpu.physicsnemo",
                "tags": ["physicsnemo", "hopper"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        let decoded = decode_schedule_payload(&payload.to_string(), "run-plugin")
            .expect("plugin run envelope should decode");

        assert_eq!(decoded.run_id, "run-plugin");
        assert_eq!(decoded.workflow, "demo-plugin");
        assert_eq!(decoded.workflow_id.as_deref(), Some("demo-plugin"));
        assert_eq!(decoded.dispatch_stage, "execute");
        assert_eq!(
            decoded
                .resource_profile
                .as_ref()
                .and_then(|profile| profile.gpus_required),
            Some(2)
        );
        assert_eq!(
            decoded
                .resource_profile
                .as_ref()
                .and_then(|profile| profile.memory_mb),
            Some(4096)
        );
        assert_eq!(
            decoded
                .resource_profile
                .as_ref()
                .and_then(|profile| profile.executor_class.as_deref()),
            Some("python.gpu.physicsnemo")
        );
        assert_eq!(
            decoded
                .resource_profile
                .as_ref()
                .and_then(|profile| profile.tags.clone()),
            Some(vec!["physicsnemo".to_string(), "hopper".to_string()])
        );
    }

    #[tokio::test]
    async fn scheduler_accepts_non_default_configured_input_stream_names() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":30000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let custom_env = RoleEnv {
            role_name: "scheduler".to_string(),
            stream_prefix: "test:".to_string(),
            inputs: vec![
                InputStreamSpec {
                    stream: "job_schedule".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
                InputStreamSpec {
                    stream: "gpu_release".to_string(),
                    max_dequeue_items: 10,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
            ],
            resolved_outputs: vec![],
            role_config: Some(json!({
                "batching_enabled": false
            })),
            python_runtime_envs: Default::default(),
        };

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &custom_env).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_for_stream(
                "run-a",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
                "job_schedule",
            ),
            "job_schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg_for_stream(
                "run-b",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
                "job_schedule",
            ),
            "job_schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(sink.writes().len(), 2);

        role.handle(
            &release_msg_for_stream(
                "run-a",
                r#"{"run_id":"run-a","resource_id":0,"memory_mb":20000,"status":"completed"}"#,
                "gpu_release",
            ),
            "gpu_release",
            &sink,
        )
        .await
        .unwrap();

        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(sink.writes().len(), 2);

        queue_schedule(
            &role,
            &schedule_msg_for_stream(
                "run-c",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
                "job_schedule",
            ),
            "job_schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(sink.writes().len(), 3);
        assert_eq!(role.queued_request_count().await, 0);

        role.handle(
            &release_msg_for_stream(
                "run-b",
                r#"{"run_id":"run-b","resource_id":0,"memory_mb":20000,"status":"completed"}"#,
                "gpu_release",
            ),
            "gpu_release",
            &sink,
        )
        .await
        .unwrap();
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 3);
        assert_eq!(writes[1].run_id, "run-b");
        assert_eq!(writes[2].run_id, "run-c");
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_routes_plugin_run_envelope_with_execute_stage() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:plugin:0","total_memory_mb":12000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo","hopper"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_config("test:", json!({})),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-plugin",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 1024,
                "executor_class": "python.gpu.physicsnemo",
                "tags": ["physicsnemo"]
            },
            "stage_context": {
                "current_phase": "schedule"
            },
            "parameters": {
                "batch_size": 128000
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("run-plugin", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].stream_key, "gpu:plugin:0");
        assert_eq!(writes[0].stage, "execute");

        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["workflow_id"], "demo-plugin");
        assert_eq!(forwarded["resource_id"], 0);
        assert_eq!(forwarded["memory_mb"], 1024);
        assert_eq!(forwarded["stage_context"]["current_phase"], "execute");
        assert_eq!(forwarded["stage_context"]["current_stage_id"], "execute");

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_advances_nested_batch_item_stage_context_on_dispatch() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.test","total_memory_mb":12000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.test","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-batch",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 512,
                "executor_class": "python.gpu.test",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_stage_id": "schedule",
                "current_phase": "schedule",
                "pipeline": [
                    {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "batch"},
                    {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                    {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                    {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.test", "next": "results"},
                    {"id": "results", "phase": "results", "queue": "results", "next": null}
                ]
            },
            "items": [
                {
                    "run_id": "run-a",
                    "payload": {
                        "run_id": "run-a",
                        "workflow_id": "demo-batch",
                        "stage_context": {
                            "current_stage_id": "schedule",
                            "current_phase": "schedule",
                            "pipeline": [
                                {"id": "prepare", "phase": "prepare", "queue": "prepare", "next": "batch"},
                                {"id": "batch", "phase": "batch", "queue": "batch", "next": "schedule"},
                                {"id": "schedule", "phase": "schedule", "queue": "schedule", "next": "execute"},
                                {"id": "execute", "phase": "execute", "queue": "execute.python.gpu.test", "next": "results"},
                                {"id": "results", "phase": "results", "queue": "results", "next": null}
                            ]
                        }
                    }
                }
            ]
        });

        queue_schedule(
            &role,
            &schedule_msg("batch-1", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        let forwarded: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(forwarded["stage_context"]["current_phase"], "execute");
        assert_eq!(
            forwarded["items"][0]["payload"]["stage_context"]["current_phase"],
            "execute"
        );

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_routes_to_worker_matching_executor_class() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[
                    {"resource_id":0,"stream_name":"gpu:demo:0","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]},
                    {"resource_id":1,"stream_name":"gpu:physicsnemo:1","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo","hopper"]}
                ]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-plugin",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 2048,
                "executor_class": "python.gpu.physicsnemo",
                "tags": ["physicsnemo", "hopper"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("run-plugin", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].stream_key, "gpu:physicsnemo:1");

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[test]
    fn decode_schedule_payload_rejects_zero_gpu_resource_profile() {
        let payload = serde_json::json!({
            "workflow_id": "demo-deterministic",
            "resource_profile": {
                "gpus_required": 0,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            }
        });

        let error = decode_schedule_payload(&payload.to_string(), "run-invalid").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("resource_profile.gpus_required must be at least 1"),
            "zero-GPU payloads should be rejected during decode"
        );
    }

    #[tokio::test]
    async fn scheduler_requeues_fanout_item_when_parent_max_in_flight_is_saturated() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-ensemble",
            "fanout_profile": {
                "item_count": 4,
                "max_in_flight": 1
            },
            "fanout_item": {
                "item_index": 0
            },
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("parent-ensemble:item:0", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let second_payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-ensemble",
            "fanout_profile": {
                "item_count": 4,
                "max_in_flight": 1
            },
            "fanout_item": {
                "item_index": 1
            },
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("parent-ensemble:item:1", &second_payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        let enqueued_at_before_blocked_attempt = role
            .scheduler_queue_state
            .lock()
            .await
            .queue
            .front()
            .expect("queued fanout item should be pending")
            .enqueued_at;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].run_id, "parent-ensemble:item:0");
        assert!(sink.handoffs().is_empty());
        assert_eq!(role.queued_request_count().await, 1);
        let enqueued_at_after_blocked_attempt = role
            .scheduler_queue_state
            .lock()
            .await
            .queue
            .front()
            .expect("blocked fanout item should remain pending")
            .enqueued_at;
        assert_eq!(
            enqueued_at_after_blocked_attempt, enqueued_at_before_blocked_attempt,
            "blocked attempts should preserve the total queue-wait measurement window"
        );

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_release_reopens_parent_slot() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-ensemble",
            "fanout_profile": {
                "item_count": 4,
                "max_in_flight": 1
            },
            "fanout_item": {
                "item_index": 0
            },
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("parent-ensemble:item:0", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        role.handle(
            &release_msg(
                "parent-ensemble:item:0",
                r#"{"run_id":"parent-ensemble:item:0","parent_run_id":"parent-ensemble","resource_id":0,"memory_mb":4096,"status":"completed"}"#,
            ),
            "release",
            &sink,
        )
        .await
        .unwrap();

        let second_payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-ensemble",
            "fanout_profile": {
                "item_count": 4,
                "max_in_flight": 1
            },
            "fanout_item": {
                "item_index": 1
            },
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("parent-ensemble:item:1", &second_payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[1].run_id, "parent-ensemble:item:1");
        assert!(sink.handoffs().is_empty());

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_release_does_not_free_parent_slot_when_no_gpu_allocation_is_released() {
        let parent_slots = Arc::new(TrackingParentSlotStore::new());
        let (_redis_server, role, _) = scheduler_with_test_queue_manager_and_dependencies(
            "scheduler-tests",
            &scheduler_env("test:"),
            parent_slots.clone(),
            Arc::new(InMemoryParentRunStateStore::new()),
            RetryDlqPolicy::new(5, "dlq"),
        )
        .await;

        let result = role
            .apply_release(&ReleasePayload {
                run_id: "run-missing".to_string(),
                parent_run_id: Some("parent-ensemble".to_string()),
                memory_mb: 4096,
                resource_id: 0,
            })
            .await;

        assert!(result.is_err());
        assert!(
            parent_slots.released_parent_run_ids().is_empty(),
            "parent slot must not be released when no GPU allocation was actually released"
        );
    }

    #[tokio::test]
    async fn scheduler_allows_single_parent_to_dispatch_multiple_items_without_requeue() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo.0","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]},{"resource_id":1,"stream_name":"execute.python.gpu.demo.1","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        for item_index in 0..2 {
            let payload = serde_json::json!({
                "workflow_id": "demo-fanout",
                "parent_run_id": "parent-a",
                "fanout_profile": {
                    "item_count": 4,
                    "max_in_flight": 4
                },
                "fanout_item": {
                    "item_index": item_index
                },
                "resource_profile": {
                    "gpus_required": 1,
                    "memory_mb": 4096,
                    "executor_class": "python.gpu.demo",
                    "tags": ["demo", "gpu"]
                },
                "stage_context": {
                    "current_phase": "schedule"
                }
            });

            queue_schedule(
                &role,
                &schedule_msg(
                    format!("parent-a:item:{item_index}").as_str(),
                    &payload.to_string(),
                ),
                "schedule",
                &sink,
            )
            .await;
            run_scheduler_task(&tasks, &sink).await;
        }

        assert_eq!(sink.writes().len(), 2);
        assert!(sink.handoffs().is_empty());

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_preserves_fifo_order_without_parent_fairness_yield() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo.0","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]},{"resource_id":1,"stream_name":"execute.python.gpu.demo.1","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]},{"resource_id":2,"stream_name":"execute.python.gpu.demo.2","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]},{"resource_id":3,"stream_name":"execute.python.gpu.demo.3","total_memory_mb":64000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let make_payload = |parent_run_id: &str, item_index: u64| {
            serde_json::json!({
                "workflow_id": "demo-fanout",
                "parent_run_id": parent_run_id,
                "fanout_profile": {
                    "item_count": 8,
                    "max_in_flight": 4
                },
                "fanout_item": {
                    "item_index": item_index
                },
                "resource_profile": {
                    "gpus_required": 1,
                    "memory_mb": 4096,
                    "executor_class": "python.gpu.demo",
                    "tags": ["demo", "gpu"]
                },
                "stage_context": {
                    "current_phase": "schedule"
                }
            })
        };

        for item_index in 0..2 {
            let payload = make_payload("parent-a", item_index);
            queue_schedule(
                &role,
                &schedule_msg(
                    format!("parent-a:item:{item_index}").as_str(),
                    &payload.to_string(),
                ),
                "schedule",
                &sink,
            )
            .await;
            run_scheduler_task(&tasks, &sink).await;
        }

        let payload_b = make_payload("parent-b", 0);
        queue_schedule(
            &role,
            &schedule_msg("parent-b:item:0", &payload_b.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let payload_a2 = make_payload("parent-a", 2);
        queue_schedule(
            &role,
            &schedule_msg("parent-a:item:2", &payload_a2.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 4);
        assert_eq!(writes[0].run_id, "parent-a:item:0");
        assert_eq!(writes[1].run_id, "parent-a:item:1");
        assert_eq!(writes[2].run_id, "parent-b:item:0");
        assert_eq!(writes[3].run_id, "parent-a:item:2");
        assert!(sink.handoffs().is_empty());
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_drops_child_dispatch_for_terminal_parent_run() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let terminal_state = Arc::new(InMemoryParentRunStateStore::new());
        terminal_state
            .mark_terminal("parent-ensemble")
            .await
            .unwrap();
        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager_and_dependencies(
            "scheduler-tests",
            &scheduler_env("test:"),
            Arc::new(InMemoryParentSlotStore::new()),
            terminal_state,
            RetryDlqPolicy::new(5, "dlq"),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-ensemble",
            "fanout_profile": {
                "item_count": 4,
                "max_in_flight": 1
            },
            "fanout_item": {
                "item_index": 3
            },
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 4096,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("parent-ensemble:item:3", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        assert!(sink.writes().is_empty());
        assert!(sink.handoffs().is_empty());
        assert_eq!(sink.acked_ids(), vec!["1-0".to_string()]);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_keeps_warming_workers_blocked_without_dlq() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"],"status":"warming","model_cache":{"schema_version":1,"scope":"process","entries":[],"total_entries":0,"warmup":{"workflow_id":"demo-plugin","status":"warming"}}}]"#,
            ),
        );
        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager_and_policy(
            "scheduler-tests",
            &scheduler_env("test:"),
            retry_dlq_policy(2),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg(
                "run-warming",
                &basic_schedule_payload("demo-plugin", "python.gpu.demo", 1, 2_048),
            ),
            "schedule",
            &sink,
        )
        .await;

        run_scheduler_task(&tasks, &sink).await;
        run_scheduler_task(&tasks, &sink).await;

        assert!(sink.writes().is_empty());
        assert!(sink.handoffs().is_empty());
        assert!(sink.acked_ids().is_empty());
        assert_eq!(role.queued_request_count().await, 1);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_dlqs_persistent_batch_and_marks_item_runs_failed() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"execute.python.gpu.demo","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo","gpu"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager_and_dependencies(
            "scheduler-tests",
            &scheduler_env("test:"),
            Arc::new(InMemoryParentSlotStore::new()),
            Arc::new(AlwaysFailingParentRunStateStore),
            retry_dlq_policy(2),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let failing_payload = serde_json::json!({
            "workflow_id": "demo-fanout",
            "parent_run_id": "parent-broken",
            "items": [
                {"run_id": "run-broken-item-1", "payload": {"value": 1}},
                {"run_id": "run-broken-item-2", "payload": {"value": 2}}
            ],
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 2048,
                "executor_class": "python.gpu.demo",
                "tags": ["demo", "gpu"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });
        let failing_msg = scicomp_rq::Message::new(
            "1-0",
            "test:schedule",
            "schedule:grp",
            "run-broken",
            failing_payload.to_string(),
            "schedule",
        );
        queue_schedule(&role, &failing_msg, "schedule", &sink).await;

        let followup_msg = scicomp_rq::Message::new(
            "2-0",
            "test:schedule",
            "schedule:grp",
            "run-followup",
            basic_schedule_payload("demo-followup", "python.gpu.demo", 1, 1024),
            "schedule",
        );
        queue_schedule(&role, &followup_msg, "schedule", &sink).await;

        run_scheduler_task(&tasks, &sink).await;
        assert!(sink.handoffs().is_empty());
        assert!(sink.writes().is_empty());
        assert_eq!(role.queued_request_count().await, 2);

        run_scheduler_task(&tasks, &sink).await;
        let handoffs = sink.handoffs();
        assert_eq!(handoffs.len(), 1);
        assert_eq!(handoffs[0].stream, "shared-dlq");
        assert_eq!(handoffs[0].stage, "dlq");
        assert!(handoffs[0].payload.contains("\"attempts\":2"));
        assert!(
            handoffs[0]
                .payload
                .contains("\"source_stream\":\"schedule\"")
        );
        assert_eq!(
            sink.failed_run_ids(),
            vec![
                "run-broken".to_string(),
                "run-broken-item-1".to_string(),
                "run-broken-item-2".to_string(),
            ]
        );
        assert_eq!(role.queued_request_count().await, 1);

        run_scheduler_task(&tasks, &sink).await;
        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].run_id, "run-followup");
        assert_eq!(role.queued_request_count().await, 0);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_rejects_when_no_worker_matches_executor_class() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:demo:0","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-plugin",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 2048,
                "executor_class": "python.gpu.physicsnemo"
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("run-plugin", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        assert_eq!(role.queued_request_count().await, 1);
        assert!(sink.writes().is_empty());

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_skips_duplicate_local_gpu_ids() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[
                    {"resource_id":0,"stream_name":"gpu:pod-a:0","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo"]},
                    {"resource_id":0,"stream_name":"gpu:pod-b:0","total_memory_mb":16000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.physicsnemo","tags":["physicsnemo"]}
                ]"#,
            ),
        );
        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow_id": "demo-plugin",
            "resource_profile": {
                "gpus_required": 2,
                "memory_mb": 2048,
                "executor_class": "python.gpu.physicsnemo",
                "tags": ["physicsnemo"]
            },
            "stage_context": {
                "current_phase": "schedule"
            }
        });

        queue_schedule(
            &role,
            &schedule_msg("run-plugin", &payload.to_string()),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        assert_eq!(role.queued_request_count().await, 1);
        assert!(sink.writes().is_empty());

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_release_updates_reserved_memory_accounting() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":30000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_config("test:", json!({})),
        )
        .await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg(
                "run-a",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg(
                "run-b",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(role.reservations.used_memory_mb(0).await, Some(40_000));

        role.handle(
            &release_msg(
                "run-a",
                r#"{"run_id":"run-a","resource_id":0,"memory_mb":20000,"status":"completed"}"#,
            ),
            "release",
            &sink,
        )
        .await
        .unwrap();
        assert_eq!(role.reservations.used_memory_mb(0).await, Some(20_000));

        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(role.reservations.used_memory_mb(0).await, Some(20_000));
        queue_schedule(
            &role,
            &schedule_msg(
                "run-c",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(sink.writes().len(), 3);
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(role.reservations.used_memory_mb(0).await, Some(40_000));

        role.handle(
            &release_msg(
                "run-b",
                r#"{"run_id":"run-b","resource_id":0,"memory_mb":20000,"status":"completed"}"#,
            ),
            "release",
            &sink,
        )
        .await
        .unwrap();
        run_scheduler_task(&tasks, &sink).await;
        assert_eq!(sink.writes().len(), 3);
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(role.reservations.used_memory_mb(0).await, Some(20_000));

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_preserves_reserved_memory_accounting_after_restart() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":30000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (redis_server, qm) = spawn_test_queue_manager("scheduler-tests").await;
        let env = scheduler_env_with_config("test:", json!({}));
        let (role_before_restart, tasks_before_restart) =
            SchedulerRole::from_env(&env, qm.clone(), retry_dlq_policy(5), None).unwrap();
        seed_registry_from_discovery_json(
            &qm,
            role_before_restart.config.gpu_registry_key.as_str(),
            "scheduler-test-worker",
        )
        .await;

        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks_before_restart, &sink).await;
        queue_schedule(
            &role_before_restart,
            &schedule_msg(
                "run-a",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks_before_restart, &sink).await;
        assert_eq!(sink.writes().len(), 1);

        let (role_after_restart, tasks_after_restart) =
            SchedulerRole::from_env(&env, qm.clone(), retry_dlq_policy(5), None).unwrap();
        run_gpu_discovery(&tasks_after_restart, &sink).await;

        queue_schedule(
            &role_after_restart,
            &schedule_msg(
                "run-b",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks_after_restart, &sink).await;
        assert_eq!(role_after_restart.queued_request_count().await, 0);
        assert_eq!(
            role_after_restart.reservations.used_memory_mb(0).await,
            Some(40_000)
        );

        role_after_restart
            .handle(
                &release_msg(
                    "run-a",
                    r#"{"run_id":"run-a","resource_id":0,"memory_mb":20000,"status":"completed"}"#,
                ),
                "release",
                &sink,
            )
            .await
            .unwrap();
        assert_eq!(
            role_after_restart.reservations.used_memory_mb(0).await,
            Some(20_000)
        );

        run_scheduler_task(&tasks_after_restart, &sink).await;
        assert_eq!(sink.writes().len(), 2);
        assert_eq!(sink.writes()[1].run_id, "run-b");
        assert_eq!(role_after_restart.queued_request_count().await, 0);
        assert_eq!(
            role_after_restart.reservations.used_memory_mb(0).await,
            Some(20_000)
        );

        drop(redis_server);
        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_fans_out_when_profile_requires_multiple_gpus() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();

        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":20000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]},{"resource_id":1,"stream_name":"gpu:ns:pod:1","total_memory_mb":20000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg(
                "run-multi",
                &basic_schedule_payload("multiwf", "python.gpu.demo", 2, 1_024),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 2, "expected two fan-out dispatches");

        let resource_ids: HashSet<u64> = writes
            .iter()
            .map(|record| {
                serde_json::from_str::<JsonValue>(&record.payload).unwrap()["resource_id"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(resource_ids, HashSet::from([0, 1]));

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
    }

    #[tokio::test]
    async fn scheduler_rejects_schedule_without_resource_profile() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );
        set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"known-profile-only","gpus.used":1,"peak":{"memory.used":"1024 MiB"}}]}"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg("run-1", r#"{"workflow_id":"demo"}"#),
            "schedule",
            &sink,
        )
        .await;
        run_task_by_name_result(&tasks, "scheduler_task", &sink)
            .await
            .unwrap();
        assert_eq!(role.queued_request_count().await, 1);
        assert!(sink.writes().is_empty());
        assert!(sink.handoffs().is_empty());

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn scheduler_accepts_schedule_without_resource_profile_when_known_profile_exists() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );
        set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"demo","gpus.used":1,"peak":{"memory.used":"2048 MiB"}}]}"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg("run-1", r#"{"workflow_id":"demo"}"#),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].stream_key, "gpu:ns:pod:0");

        let payload_json: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(payload_json["workflow_id"], "demo");
        assert_eq!(payload_json["resource_id"], 0);
        assert!(payload_json.get("gpu_stream").is_none());
        assert_eq!(payload_json["memory_mb"], 2_048);
        assert_eq!(payload_json["resource_profile"]["gpus_required"], 1);
        assert_eq!(payload_json["resource_profile"]["memory_mb"], 2_048);
        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn scheduler_overwrites_null_resource_profile_with_known_profile_fallback() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );
        set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"demo","gpus.used":1,"peak":{"memory.used":"2048 MiB"}}]}"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        queue_schedule(
            &role,
            &schedule_msg("run-1", r#"{"workflow_id":"demo","resource_profile":null}"#),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);

        let payload_json: JsonValue = serde_json::from_str(&writes[0].payload).unwrap();
        assert_eq!(payload_json["resource_profile"]["gpus_required"], 1);
        assert_eq!(payload_json["resource_profile"]["memory_mb"], 2_048);
        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn scheduler_rejects_invalid_release_payload() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let (_redis_server, role, _) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        let result = role
            .handle(&release_msg("run-1", "not-json"), "release", &sink)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("release payload"));

        let result = role
            .handle(
                &release_msg("run-1", r#"{"run_id":"run-1","status":"completed"}"#),
                "release",
                &sink,
            )
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("resource_id and memory_mb")
        );

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn scheduler_rejects_unexpected_stream() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let (_redis_server, role, _) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        let result = role
            .handle(
                &schedule_msg(
                    "run-1",
                    &basic_schedule_payload("wf", "python.gpu.demo", 1, 1_024),
                ),
                "unknown",
                &sink,
            )
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unexpected stream")
        );

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    // --- PR-019: SchedulerRoleConfig is parsed and stored ---

    #[tokio::test]
    async fn from_env_parses_scheduler_role_config() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let config = serde_json::json!({
            "gpu_registry_key": "custom:registry",
            "gpu_discovery_interval_secs": 15
        });
        let (_redis_server, role, _) = scheduler_with_test_queue_manager(
            "scheduler-tests",
            &scheduler_env_with_config("test:", config),
        )
        .await;
        assert_eq!(role.config.gpu_registry_key, "custom:registry");
        assert_eq!(role.config.gpu_discovery_interval_secs, 15);

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn from_env_uses_default_config_when_none_provided() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let (_redis_server, role, _) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        assert_eq!(role.config.gpu_registry_key, "gpu:registry");
        assert_eq!(
            role.config.gpu_discovery_interval_secs,
            DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
        );

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn from_env_defaults_invalid_memory_utilization_percent() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let (server, qm) = spawn_test_queue_manager("scheduler-invalid-memory-limit").await;
        let config = serde_json::json!({
            "memory_utilization_percent": 0
        });

        let (role, _tasks) = SchedulerRole::from_env(
            &scheduler_env_with_config("test:", config),
            qm,
            retry_dlq_policy(5),
            None,
        )
        .expect("invalid memory_utilization_percent should fall back to default");
        assert_eq!(role.config.memory_utilization_percent, 80);

        drop(server);
        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn from_env_defaults_zero_gpu_discovery_interval_secs() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        set_env_var("SCHEDULER_PROFILES_JSON", None);

        let (server, qm) = spawn_test_queue_manager("scheduler-invalid-discovery-interval").await;
        let config = serde_json::json!({
            "gpu_discovery_interval_secs": 0
        });

        let (role, _tasks) = SchedulerRole::from_env(
            &scheduler_env_with_config("test:", config),
            qm,
            retry_dlq_policy(5),
            None,
        )
        .expect("zero gpu_discovery_interval_secs should fall back to default");
        assert_eq!(
            role.config.gpu_discovery_interval_secs,
            DEFAULT_GPU_DISCOVERY_INTERVAL_SECS
        );

        drop(server);
        set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    // --- PR-046: non-JSON schedule payload produces clear error ---

    #[test]
    fn decode_schedule_rejects_non_json_with_clear_error() {
        let result = decode_schedule_payload("not-json-at-all", "run-1");
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.to_lowercase().contains("json"),
            "non-JSON payload should produce clear JSON error, got: {err_msg}"
        );
    }

    // --- PR-045: rollback restores GPU capacity on dispatch failure ---

    struct FailingSink;
    impl MessageSink for FailingSink {
        fn enqueue<'a>(
            &'a self,
            _: &'a str,
            _: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("simulated dispatch failure")) })
        }
        fn enqueue_to_stream<'a>(
            &'a self,
            _: &'a str,
            _: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("simulated dispatch failure")) })
        }
        fn ack_message<'a>(&'a self, _: &'a scicomp_rq::Message) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn handoff<'a>(
            &'a self,
            _: &'a scicomp_rq::Message,
            _: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> BoxFuture<'a, Result<String>> {
            Box::pin(async { Err(anyhow!("simulated dispatch failure")) })
        }
        fn forward_many<'a>(
            &'a self,
            _: &'a scicomp_rq::Message,
            _: &'a [scicomp_rq::Output],
        ) -> BoxFuture<'a, Result<Vec<String>>> {
            Box::pin(async { Err(anyhow!("simulated dispatch failure")) })
        }
    }

    #[tokio::test]
    async fn dispatch_failure_rolls_back_all_gpu_reservations() {
        let _guard = test_env::env_lock().lock().await;
        let prev = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = FailingSink;
        run_gpu_discovery(&tasks, &sink).await;

        let payload = serde_json::json!({
            "workflow": "demo-deterministic",
            "workflow_id": "demo-deterministic",
            "resource_profile": {
                "gpus_required": 1,
                "memory_mb": 10000,
                "executor_class": "python.gpu.demo"
            }
        });

        let msg = schedule_msg("run-fail", &payload.to_string());
        queue_schedule(&role, &msg, "schedule", &sink).await;
        let result = run_task_by_name_result(&tasks, "scheduler_task", &sink).await;
        assert!(
            result.is_ok(),
            "dispatch failures should stay local to scheduler task"
        );
        assert_eq!(role.queued_request_count().await, 1);

        let recording_sink = RecordingSink::new();
        run_task_by_name_result(&tasks, "scheduler_task", &recording_sink)
            .await
            .expect("retry after rollback should succeed");
        assert_eq!(role.queued_request_count().await, 0);
        assert_eq!(recording_sink.writes().len(), 1);
        assert_eq!(recording_sink.writes()[0].run_id, "run-fail");

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev.as_deref());
    }

    #[tokio::test]
    async fn gpu_discovery_keeps_previous_inventory_when_local_discovery_fails() {
        let _guard = test_env::env_lock().lock().await;
        let prev = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        set_env_var(
            "SCHEDULER_DISCOVERY_JSON",
            Some(
                r#"[{"resource_id":0,"stream_name":"gpu:ns:pod:0","total_memory_mb":50000,"used_memory_mb":0,"device_kind":"gpu","executor_class":"python.gpu.demo","tags":["demo"]}]"#,
            ),
        );

        let (_redis_server, role, tasks) =
            scheduler_with_test_queue_manager("scheduler-tests", &scheduler_env("test:")).await;
        let sink = RecordingSink::new();
        run_gpu_discovery(&tasks, &sink).await;

        set_env_var("SCHEDULER_DISCOVERY_JSON", Some("not-json"));
        run_task_by_name_result(&tasks, "resource_discovery", &sink)
            .await
            .expect("failed local GPU discovery should keep prior inventory");

        queue_schedule(
            &role,
            &schedule_msg(
                "run-stale-inventory",
                &basic_schedule_payload("demo-deterministic", "python.gpu.demo", 1, 20_000),
            ),
            "schedule",
            &sink,
        )
        .await;
        run_scheduler_task(&tasks, &sink).await;

        let writes = sink.writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].stream_key, "gpu:ns:pod:0");
        assert_eq!(writes[0].run_id, "run-stale-inventory");

        set_env_var("SCHEDULER_DISCOVERY_JSON", prev.as_deref());
    }
}
