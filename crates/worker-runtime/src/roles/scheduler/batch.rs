/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashSet;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tracing::debug;
use uuid::Uuid;

use super::{
    PendingSchedule, QueuedRequest, SchedulePayload, ScheduleResourceProfile, SchedulerQueueState,
    SchedulerRole, decode_schedule_payload, fanout_gate, schedule_resource_profile_json,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchProfile {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub batch_key: String,
    #[serde(default)]
    pub max_batch_size: usize,
    #[serde(default)]
    pub max_wait_ms: u64,
    #[serde(default)]
    pub shared_memory_mb: Option<u64>,
    #[serde(default)]
    pub incremental_memory_mb: Option<u64>,
}

#[derive(Debug, Clone)]
struct BatchPolicy {
    batch_key: String,
    max_batch_size: usize,
    max_wait_ms: u64,
    shared_memory_mb: Option<u64>,
    incremental_memory_mb: Option<u64>,
    base_resource_profile: ScheduleResourceProfile,
    single_request_memory_mb: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchFlushReason {
    MaxBatchSize,
    MaxWaitMs,
    MemoryFit,
}

impl BatchFlushReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::MaxBatchSize => "max_batch_size",
            Self::MaxWaitMs => "max_wait_ms",
            Self::MemoryFit => "memory_fit",
        }
    }
}

#[derive(Debug, Clone)]
struct BatchCandidate {
    queued: QueuedRequest,
    policy: BatchPolicy,
}

#[derive(Debug, Clone)]
pub(super) struct BatchRequest {
    pub(super) head: QueuedRequest,
    candidates: Vec<BatchCandidate>,
    policy: BatchPolicy,
    flush_reason: BatchFlushReason,
}

#[derive(Debug, Clone)]
pub(super) enum SchedulerRequest {
    Empty,
    Wait(QueuedRequest),
    Single(QueuedRequest),
    Batch(BatchRequest),
}

fn default_true() -> bool {
    true
}

fn scheduler_now_ms() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("scheduler: system clock before unix epoch: {err}"))?;
    Ok(duration.as_millis().try_into().unwrap_or(u64::MAX))
}

fn default_scheduler_batch_key(payload: &SchedulePayload) -> String {
    payload
        .raw_payload
        .get("operation")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("default")
        .to_string()
}

fn pipeline_contains_phase(payload: &SchedulePayload, phase: &str) -> bool {
    payload
        .raw_payload
        .get("stage_context")
        .and_then(|stage_context| stage_context.get("pipeline"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|pipeline| {
            pipeline.iter().any(|stage| {
                stage
                    .get("phase")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|candidate| candidate == phase)
            })
        })
}

fn scheduler_batch_excluded(payload: &SchedulePayload) -> bool {
    fanout_gate(payload).is_some()
        || payload
            .parent_run_id
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        || payload.fanout_profile.is_some()
        || pipeline_contains_phase(payload, "fanout")
        || pipeline_contains_phase(payload, "collect")
}

fn batch_memory_mb(candidates: &[BatchCandidate]) -> Result<u64> {
    let has_batch_memory_scaling = candidates.iter().all(|candidate| {
        candidate.policy.shared_memory_mb.is_some()
            && candidate.policy.incremental_memory_mb.is_some()
    });

    if !has_batch_memory_scaling {
        return candidates.iter().try_fold(0_u64, |total, candidate| {
            total
                .checked_add(candidate.policy.single_request_memory_mb)
                .ok_or_else(|| {
                    anyhow!(
                        "scheduler: batch memory_mb overflow while summing resource profile memory_mb={}",
                        candidate.policy.single_request_memory_mb
                    )
                })
        });
    }

    let shared_memory_mb = candidates
        .iter()
        .filter_map(|candidate| candidate.policy.shared_memory_mb)
        .max()
        .unwrap_or(0);
    let incremental_total_mb = candidates.iter().try_fold(0_u64, |total, candidate| {
        let incremental_memory_mb = candidate
            .policy
            .incremental_memory_mb
            .expect("batch memory scaling was validated above");
        total.checked_add(incremental_memory_mb).ok_or_else(|| {
            anyhow!(
                "scheduler: batch memory_mb overflow while summing incremental_memory_mb={}",
                incremental_memory_mb
            )
        })
    })?;

    shared_memory_mb
        .checked_add(incremental_total_mb)
        .ok_or_else(|| {
            anyhow!(
                "scheduler: batch memory_mb overflow while adding shared_memory_mb={} and incremental_total_mb={}",
                shared_memory_mb,
                incremental_total_mb
            )
        })
}

fn batch_resource_profile(
    policy: &BatchPolicy,
    candidates: &[BatchCandidate],
) -> Result<ScheduleResourceProfile> {
    let mut profile = policy.base_resource_profile.clone();
    profile.gpus_required = Some(1);
    profile.memory_mb = Some(batch_memory_mb(candidates)?);
    Ok(profile)
}

fn batch_resource_profile_json(
    policy: &BatchPolicy,
    candidates: &[BatchCandidate],
) -> Result<JsonValue> {
    let profile = batch_resource_profile(policy, candidates)?;
    Ok(schedule_resource_profile_json(&profile))
}

fn batch_memory_fit_payload(
    policy: &BatchPolicy,
    candidates: &[BatchCandidate],
) -> Result<SchedulePayload> {
    let Some(head) = candidates.first() else {
        return Err(anyhow!("scheduler: cannot evaluate empty batch"));
    };
    let mut payload = head.queued.payload.clone();
    payload.resource_profile = Some(batch_resource_profile(policy, candidates)?);
    Ok(payload)
}

fn raw_string_field(payload: &JsonValue, field: &str) -> Option<String> {
    payload
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn build_request_batch_payload(
    batch_id: &str,
    candidates: &[BatchCandidate],
    policy: &BatchPolicy,
    flush_reason: BatchFlushReason,
) -> Result<SchedulePayload> {
    let Some(head) = candidates.first() else {
        return Err(anyhow!("scheduler: cannot build empty batch"));
    };
    let batch_size = candidates.len();
    let first_seen_at = head.queued.enqueued_at;
    let first_seen_ms = scheduler_now_ms()?.saturating_sub(
        first_seen_at
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
    );
    let formed_at_ms = scheduler_now_ms()?;
    let waited_ms = formed_at_ms.saturating_sub(first_seen_ms);
    let head_payload = &head.queued.payload.raw_payload;
    let workflow_id = head
        .queued
        .payload
        .workflow_id
        .clone()
        .unwrap_or_else(|| head.queued.payload.workflow.clone());
    let operation = raw_string_field(head_payload, "operation");
    let manifest_version = raw_string_field(head_payload, "manifest_version");
    let runtime = head_payload
        .get("runtime")
        .cloned()
        .unwrap_or(JsonValue::Null);
    let stage_context = head_payload
        .get("stage_context")
        .cloned()
        .ok_or_else(|| anyhow!("scheduler: batch head payload missing stage_context"))?;
    let resource_profile = batch_resource_profile_json(policy, candidates)?;
    let items = candidates
        .iter()
        .map(|candidate| {
            serde_json::json!({
                "run_id": candidate.queued.msg.run_id(),
                "payload": candidate.queued.payload.raw_payload.clone(),
            })
        })
        .collect::<Vec<_>>();

    let batch_payload_json = serde_json::json!({
        "run_id": batch_id,
        "batch_id": batch_id,
        "batch_info": {
            "batch_id": batch_id,
            "batch_size": batch_size,
            "flush_reason": flush_reason.as_str(),
            "first_seen_ms": first_seen_ms,
            "formed_at_ms": formed_at_ms,
            "waited_ms": waited_ms,
        },
        "workflow": head.queued.payload.workflow.clone(),
        "workflow_id": workflow_id,
        "operation": operation,
        "manifest_version": manifest_version,
        "batch_profile": {
            "enabled": true,
            "batch_key": policy.batch_key.clone(),
            "max_batch_size": policy.max_batch_size,
            "max_wait_ms": policy.max_wait_ms,
            "shared_memory_mb": policy.shared_memory_mb,
            "incremental_memory_mb": policy.incremental_memory_mb,
        },
        "items": items,
        "resource_profile": resource_profile,
        "runtime": runtime,
        "stage_context": stage_context,
    });
    let encoded = serde_json::to_string(&batch_payload_json)
        .context("scheduler: failed to encode batch payload")?;
    decode_schedule_payload(&encoded, batch_id)
}

impl SchedulerRole {
    async fn batch_policy_for_payload(&self, payload: &SchedulePayload) -> Option<BatchPolicy> {
        if !self.config.batching_enabled || scheduler_batch_excluded(payload) {
            return None;
        }
        if payload
            .batch_profile
            .as_ref()
            .is_some_and(|profile| !profile.enabled)
        {
            return None;
        }

        let mut resolved = payload.clone();
        let Ok((gpus_required, memory_mb, _source, _resources)) = self
            .reservations
            .prepare_resource_requirements(&mut resolved)
            .await
        else {
            return None;
        };
        if gpus_required != 1 || memory_mb == 0 {
            return None;
        }
        let base_resource_profile = resolved.resource_profile.clone()?;
        if base_resource_profile
            .executor_class
            .as_deref()
            .is_none_or(|executor_class| executor_class.trim().is_empty())
        {
            return None;
        }
        let batch_profile = payload.batch_profile.as_ref();
        let batch_key = batch_profile
            .map(|batch_profile| batch_profile.batch_key.trim())
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| default_scheduler_batch_key(payload));
        let max_batch_size = batch_profile
            .map(|batch_profile| batch_profile.max_batch_size)
            .filter(|value| *value > 0)
            .unwrap_or(self.config.max_batch_size)
            .max(1);
        let max_wait_ms = batch_profile
            .map(|batch_profile| batch_profile.max_wait_ms)
            .filter(|value| *value > 0)
            .unwrap_or(self.config.max_batch_wait_ms);

        Some(BatchPolicy {
            batch_key,
            max_batch_size,
            max_wait_ms,
            shared_memory_mb: batch_profile
                .and_then(|batch_profile| batch_profile.shared_memory_mb),
            incremental_memory_mb: batch_profile
                .and_then(|batch_profile| batch_profile.incremental_memory_mb),
            base_resource_profile,
            single_request_memory_mb: memory_mb,
        })
    }

    fn batch_key(&self, payload: &SchedulePayload, policy: &BatchPolicy) -> String {
        let workflow = payload
            .workflow_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(payload.workflow.as_str());
        let executor_class = policy
            .base_resource_profile
            .executor_class
            .as_deref()
            .unwrap_or("");
        format!("{}::{}::{}", workflow, policy.batch_key, executor_class)
    }

    pub(super) async fn create_batch_request(
        &self,
        state: &mut SchedulerQueueState,
        head: QueuedRequest,
    ) -> Result<Option<SchedulerRequest>> {
        // Leave non-batchable requests on the normal scheduling path.
        let Some(batch_policy) = self.batch_policy_for_payload(&head.payload).await else {
            return Ok(None);
        };
        if batch_policy.max_batch_size <= 1 {
            return Ok(None);
        }

        // Avoid duplicate deliveries of the same run into one batch.
        let mut seen_run_ids = HashSet::new();
        seen_run_ids.insert(head.msg.run_id().to_string());

        // Seed the batch with the queue head and its compatibility key.
        let head_batch_key = self.batch_key(&head.payload, &batch_policy);
        let mut batch_candidates = vec![BatchCandidate {
            queued: head.clone(),
            policy: batch_policy.clone(),
        }];
        let mut effective_max_batch_size = batch_policy.max_batch_size;

        let mut flush_reason = None;
        let mut memory_limited = false;

        // Scan the remaining queue for unique, compatible requests.
        for queued in state.queue.iter().skip(1) {
            if !seen_run_ids.insert(queued.msg.run_id().to_string()) {
                continue;
            }
            let Some(candidate_policy) = self.batch_policy_for_payload(&queued.payload).await
            else {
                continue;
            };
            if self.batch_key(&queued.payload, &candidate_policy) != head_batch_key {
                continue;
            }
            let candidate_max_batch_size =
                effective_max_batch_size.min(candidate_policy.max_batch_size);
            if batch_candidates.len() + 1 > candidate_max_batch_size {
                continue;
            }
            batch_candidates.push(BatchCandidate {
                queued: queued.clone(),
                policy: candidate_policy,
            });

            // Check the combined memory footprint before keeping the candidate.
            let mut memory_fit_payload =
                batch_memory_fit_payload(&batch_policy, &batch_candidates)?;
            if !self
                .reservations
                .batch_can_fit_memory(&mut memory_fit_payload)
                .await?
            {
                let attempted_batch_size = batch_candidates.len();
                batch_candidates.pop();
                memory_limited = true;
                debug!(
                    run_id = queued.msg.run_id(),
                    current_batch_size = batch_candidates.len(),
                    attempted_batch_size,
                    "scheduler skipping batch candidate because combined batch does not fit on an eligible GPU"
                );
                continue;
            }
            effective_max_batch_size = candidate_max_batch_size;

            // Reaching the limit by adding this compatible request forms a full batch
            // and should flush immediately instead of waiting for max_wait_ms.
            if batch_candidates.len() == effective_max_batch_size {
                flush_reason = Some(BatchFlushReason::MaxBatchSize);
                break;
            }
        }

        // With no partner, either keep waiting or resume normal head scheduling.
        if batch_candidates.len() == 1 {
            if memory_limited
                || head.enqueued_at.elapsed() >= Duration::from_millis(batch_policy.max_wait_ms)
            {
                return Ok(None);
            }
            return Ok(Some(SchedulerRequest::Wait(head)));
        }

        // Flush partial batches only when memory or the wait deadline stops growth.
        let flush_reason = if let Some(reason) = flush_reason {
            reason
        } else if memory_limited {
            BatchFlushReason::MemoryFit
        } else if head.enqueued_at.elapsed() >= Duration::from_millis(batch_policy.max_wait_ms) {
            BatchFlushReason::MaxWaitMs
        } else {
            return Ok(Some(SchedulerRequest::Wait(head)));
        };

        // Hand the completed batch to the reservation and dispatch path.
        Ok(Some(SchedulerRequest::Batch(BatchRequest {
            head,
            candidates: batch_candidates,
            policy: batch_policy,
            flush_reason,
        })))
    }
}

impl BatchRequest {
    pub(super) fn prepare_schedule(&self) -> Result<PendingSchedule> {
        let batch_id = Uuid::new_v4().to_string();
        let candidates = self.candidates.as_slice();
        let payload =
            build_request_batch_payload(&batch_id, candidates, &self.policy, self.flush_reason)?;

        Ok(PendingSchedule {
            source_msg: self.head.msg.clone(),
            payload,
            ack_after_dispatch: candidates
                .iter()
                .skip(1)
                .map(|candidate| candidate.queued.msg.clone())
                .collect(),
        })
    }
}
