/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::QueuedRequest;
use super::reservation::{QueueCandidateStatus, ResourceAvailabilityView};

/// A composable score contribution used to rank eligible queue entries.
pub(super) trait QueueScoreMetric<T, C>: Send + Sync {
    fn name(&self) -> &'static str;
    fn score(
        &self,
        item: &T,
        context: &C,
        candidate_status: QueueCandidateStatus,
        now: Instant,
    ) -> f64;
}

/// Associates a metric with its contribution weight.
pub(super) struct WeightedMetric<T, C> {
    weight: f64,
    metric: Arc<dyn QueueScoreMetric<T, C>>,
}

impl<T, C> WeightedMetric<T, C> {
    pub(super) fn new(weight: f64, metric: Arc<dyn QueueScoreMetric<T, C>>) -> Self {
        debug_assert!(!metric.name().is_empty());
        Self { weight, metric }
    }
}

struct QueueEntry<T> {
    item: T,
    deferred_until: Option<Instant>,
}

/// Insertion-ordered storage with metric-based selection and temporary deferral.
///
/// Selection is deliberately O(n): scores such as age and resource affinity are
/// dynamic, and scheduler queues are expected to remain small. Recomputing from
/// one context avoids stale heap ordering and keeps adding a metric trivial.
pub(super) struct PriorityQueue<T, C> {
    entries: VecDeque<QueueEntry<T>>,
    metrics: Vec<WeightedMetric<T, C>>,
    unschedulable_retry_delay: Duration,
}

impl<T, C> PriorityQueue<T, C> {
    pub(super) fn new(
        metrics: Vec<WeightedMetric<T, C>>,
        unschedulable_retry_delay: Duration,
    ) -> Self {
        Self {
            entries: VecDeque::new(),
            metrics,
            unschedulable_retry_delay,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn has_ready_candidate(&self, now: Instant) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.deferred_until.is_none_or(|deadline| deadline <= now))
    }

    pub(super) fn push_back(&mut self, item: T) {
        self.entries.push_back(QueueEntry {
            item,
            deferred_until: None,
        });
    }

    #[cfg(test)]
    pub(super) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.entries.get_mut(index).map(|entry| &mut entry.item)
    }

    #[cfg(test)]
    pub(super) fn front(&self) -> Option<&T> {
        self.entries.front().map(|entry| &entry.item)
    }

    #[cfg(test)]
    pub(super) fn front_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    #[cfg(test)]
    pub(super) fn iter(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|entry| &entry.item)
    }

    pub(super) fn iter_ready(&self, now: Instant) -> impl Iterator<Item = &T> {
        self.entries
            .iter()
            .filter(move |entry| entry.deferred_until.is_none_or(|deadline| deadline <= now))
            .map(|entry| &entry.item)
    }

    pub(super) fn retain(&mut self, mut predicate: impl FnMut(&T) -> bool) {
        self.entries.retain(|entry| predicate(&entry.item));
    }

    pub(super) fn defer(&mut self, now: Instant, mut predicate: impl FnMut(&T) -> bool) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| predicate(&entry.item)) {
            Self::defer_entry(entry, now, self.unschedulable_retry_delay);
        }
    }

    pub(super) fn clear_deferrals(&mut self) {
        for entry in &mut self.entries {
            entry.deferred_until = None;
        }
    }

    fn defer_entry(entry: &mut QueueEntry<T>, now: Instant, retry_delay: Duration) {
        entry.deferred_until = Some(now + retry_delay);
    }
}

pub(super) struct QueueSelection {
    pub(super) request: QueuedRequest,
    pub(super) status: QueueCandidateStatus,
}

struct ScoredCandidate {
    queue_index: usize,
    score: f64,
    status: QueueCandidateStatus,
}

impl PriorityQueue<QueuedRequest, ResourceAvailabilityView> {
    pub(super) fn rank_ready_candidates(
        &mut self,
        resource_snapshot: &ResourceAvailabilityView,
        now: Instant,
    ) -> Vec<QueueSelection> {
        let metrics = &self.metrics;
        let mut ranked = Vec::new();

        for (index, entry) in self.entries.iter_mut().enumerate() {
            if entry.deferred_until.is_some_and(|deadline| deadline > now) {
                continue;
            }
            entry.deferred_until = None;

            let candidate_status = resource_snapshot.evaluate_candidate_status(&entry.item.payload);

            let mut score = 0.0;
            for weighted_metric in metrics {
                let metric_score = weighted_metric.metric.score(
                    &entry.item,
                    resource_snapshot,
                    candidate_status,
                    now,
                );
                if metric_score.is_finite() {
                    score += weighted_metric.weight * metric_score;
                }
            }
            ranked.push(ScoredCandidate {
                queue_index: index,
                score,
                status: candidate_status,
            });
        }

        // Preserve insertion order when scores tie.
        ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
        ranked
            .into_iter()
            .filter_map(|candidate| {
                self.entries
                    .get(candidate.queue_index)
                    .map(|entry| QueueSelection {
                        request: entry.item.clone(),
                        status: candidate.status,
                    })
            })
            .collect()
    }
}

/// Favors workflows with higher configured business priority.
/// Scores are normalized by the largest absolute configured priority.
pub(super) struct WorkflowPriorityMetric {
    priorities: HashMap<String, i64>,
    normalization_factor: f64,
}

impl WorkflowPriorityMetric {
    pub(super) fn new(priorities: HashMap<String, i64>) -> Self {
        let normalization_factor = priorities
            .values()
            .map(|priority| priority.unsigned_abs())
            .max()
            .unwrap_or(1)
            .max(1) as f64;
        Self {
            priorities,
            normalization_factor,
        }
    }
}

impl QueueScoreMetric<QueuedRequest, ResourceAvailabilityView> for WorkflowPriorityMetric {
    fn name(&self) -> &'static str {
        "workflow_priority"
    }

    fn score(
        &self,
        item: &QueuedRequest,
        _context: &ResourceAvailabilityView,
        _candidate_status: QueueCandidateStatus,
        _now: Instant,
    ) -> f64 {
        self.priorities
            .get(item.payload.workflow.as_str())
            .copied()
            .unwrap_or_default() as f64
            / self.normalization_factor
    }
}

/// Gradually raises the priority of waiting requests to prevent starvation.
/// The score reaches one at the configured horizon and continues growing so
/// that age eventually outweighs bounded priority and affinity scores.
pub(super) struct QueueAgeMetric {
    saturation: Duration,
}

impl QueueAgeMetric {
    pub(super) fn new(saturation: Duration) -> Self {
        Self {
            saturation: saturation.max(Duration::from_secs(1)),
        }
    }
}

impl QueueScoreMetric<QueuedRequest, ResourceAvailabilityView> for QueueAgeMetric {
    fn name(&self) -> &'static str {
        "queue_age"
    }

    fn score(
        &self,
        item: &QueuedRequest,
        _context: &ResourceAvailabilityView,
        _candidate_status: QueueCandidateStatus,
        now: Instant,
    ) -> f64 {
        now.saturating_duration_since(item.enqueued_at)
            .as_secs_f64()
            / self.saturation.as_secs_f64()
    }
}

/// Favors requests that can reuse a workflow already cached or warm on an eligible worker.
/// The score is one when warm affinity exists and zero otherwise.
pub(super) struct WarmCacheAffinityMetric;

impl QueueScoreMetric<QueuedRequest, ResourceAvailabilityView> for WarmCacheAffinityMetric {
    fn name(&self) -> &'static str {
        "warm_cache_affinity"
    }

    fn score(
        &self,
        _item: &QueuedRequest,
        _context: &ResourceAvailabilityView,
        candidate_status: QueueCandidateStatus,
        _now: Instant,
    ) -> f64 {
        match candidate_status {
            QueueCandidateStatus::Eligible {
                has_warm_worker: true,
            } => 1.0,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::scheduler::ScheduleResourceProfile;
    use crate::roles::scheduler::discovery::ResourceInfo;
    use scicomp_rq::Message;
    use serde_json::json;

    fn queue() -> PriorityQueue<QueuedRequest, ResourceAvailabilityView> {
        PriorityQueue::new(
            vec![WeightedMetric::new(
                1.0,
                Arc::new(WorkflowPriorityMetric::new(HashMap::from([
                    ("high".to_string(), 3),
                    ("medium".to_string(), 2),
                    ("low".to_string(), 1),
                ]))),
            )],
            Duration::from_millis(10),
        )
    }

    fn queued_request(
        workflow: &str,
        workflow_id: Option<&str>,
        enqueued_at: Instant,
    ) -> QueuedRequest {
        QueuedRequest {
            msg: Message::new("1-0", "schedule", "group", "run-1", "{}", "schedule"),
            payload: super::super::SchedulePayload {
                run_id: "run-1".to_string(),
                workflow: workflow.to_string(),
                workflow_id: workflow_id.map(str::to_string),
                parent_run_id: None,
                fanout_profile: None,
                batch_profile: None,
                raw_payload: json!({}),
                resource_profile: Some(ScheduleResourceProfile {
                    gpus_required: Some(1),
                    memory_mb: Some(1_024),
                    executor_class: Some("python.gpu.demo".to_string()),
                    tags: None,
                }),
                gpus_required: 0,
                memory_mb: 0,
                dispatch_stage: "execute".to_string(),
            },
            enqueued_at,
        }
    }

    fn resource_snapshot_with_cached_workflow(workflow_id: &str) -> ResourceAvailabilityView {
        resource_snapshot_with_cached_workflow_and_reservations(workflow_id, HashMap::new())
    }

    fn resource_snapshot_with_cached_workflow_and_reservations(
        workflow_id: &str,
        active_reserved_memory_mb: HashMap<u32, u64>,
    ) -> ResourceAvailabilityView {
        ResourceAvailabilityView::for_tests(
            vec![ResourceInfo {
                resource_id: 0,
                stream_name: "execute.python.gpu.demo".to_string(),
                total_memory_mb: 16_000,
                used_memory_mb: 15_000,
                executor_class: Some("python.gpu.demo".to_string()),
                tags: Vec::new(),
                model_cache_workflow_ids: vec![workflow_id.to_string()],
                warmup_workflow_id: Some(workflow_id.to_string()),
                warmup_status: Some("succeeded".to_string()),
            }],
            active_reserved_memory_mb,
        )
    }

    #[test]
    fn selects_highest_weighted_score() {
        let mut queue = queue();
        let now = Instant::now();
        queue.push_back(queued_request("low", None, now));
        queue.push_back(queued_request("high", None, now));
        queue.push_back(queued_request("medium", None, now));
        let resource_snapshot = resource_snapshot_with_cached_workflow("demo");

        assert_eq!(
            queue
                .rank_ready_candidates(&resource_snapshot, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("high".to_string())
        );
    }

    #[test]
    fn preserves_fifo_order_for_tied_scores() {
        let mut queue = queue();
        let now = Instant::now();
        queue.push_back(queued_request("tie-a", None, now));
        queue.push_back(queued_request("tie-b", None, now));
        let resource_snapshot = resource_snapshot_with_cached_workflow("demo");

        assert_eq!(
            queue
                .rank_ready_candidates(&resource_snapshot, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("tie-a".to_string())
        );
    }

    #[test]
    fn reports_candidate_status_and_supports_deferral() {
        let now = Instant::now();
        let mut queue = queue();
        queue.push_back(queued_request("high", None, now));
        let mut low_priority = queued_request("low", None, now);
        low_priority
            .payload
            .resource_profile
            .as_mut()
            .unwrap()
            .executor_class = Some("python.gpu.other".to_string());
        queue.push_back(low_priority);
        let blocked_resource_snapshot = resource_snapshot_with_cached_workflow_and_reservations(
            "demo",
            HashMap::from([(0, 1)]),
        );

        let selection = queue
            .rank_ready_candidates(&blocked_resource_snapshot, now)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(selection.request.payload.workflow, "high");
        assert_eq!(selection.status, QueueCandidateStatus::TemporarilyBlocked);
        queue.defer(now, |candidate| candidate.payload.workflow == "high");

        assert_eq!(
            queue
                .rank_ready_candidates(&blocked_resource_snapshot, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("low".to_string())
        );
        assert_eq!(
            queue
                .rank_ready_candidates(&blocked_resource_snapshot, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("low".to_string())
        );

        queue.clear_deferrals();
        let ready_resource_snapshot = resource_snapshot_with_cached_workflow("demo");
        assert_eq!(
            queue
                .rank_ready_candidates(&ready_resource_snapshot, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("high".to_string())
        );
    }

    #[test]
    fn retain_and_iteration_hide_queue_metadata() {
        let mut queue = queue();
        let now = Instant::now();
        queue.push_back(queued_request("low", None, now));
        queue.push_back(queued_request("medium", None, now));
        queue.push_back(queued_request("high", None, now));
        queue.retain(|item| item.payload.workflow != "medium");

        assert_eq!(
            queue
                .iter()
                .map(|item| item.payload.workflow.as_str())
                .collect::<Vec<_>>(),
            vec!["low", "high"]
        );
    }

    #[test]
    fn workflow_priority_metric_normalizes_configured_values() {
        let metric = WorkflowPriorityMetric::new(HashMap::from([
            ("premium".to_string(), 10),
            ("batch".to_string(), -1),
        ]));
        let context = resource_snapshot_with_cached_workflow("premium");
        let now = Instant::now();

        assert_eq!(
            metric.score(
                &queued_request("premium", Some("premium"), now),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: false,
                },
                now
            ),
            1.0
        );
        assert_eq!(
            metric.score(
                &queued_request("unknown", None, now),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: false,
                },
                now
            ),
            0.0
        );
    }

    #[test]
    fn queue_age_metric_continues_growing_after_saturation_horizon() {
        let metric = QueueAgeMetric::new(Duration::from_secs(50));
        let now = Instant::now();
        let context = resource_snapshot_with_cached_workflow("demo");

        assert_eq!(
            metric.score(
                &queued_request("demo", None, now - Duration::from_secs(25)),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: false,
                },
                now
            ),
            0.5
        );
        assert_eq!(
            metric.score(
                &queued_request("demo", None, now - Duration::from_secs(75)),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: false,
                },
                now
            ),
            1.5
        );
    }

    #[test]
    fn overdue_request_eventually_outranks_priority_and_warm_affinity() {
        let now = Instant::now();
        let mut queue = PriorityQueue::new(
            vec![
                WeightedMetric::new(
                    1.0,
                    Arc::new(WorkflowPriorityMetric::new(HashMap::from([(
                        "high".to_string(),
                        10,
                    )]))),
                ),
                WeightedMetric::new(1.0, Arc::new(QueueAgeMetric::new(Duration::from_secs(300)))),
                WeightedMetric::new(1.0, Arc::new(WarmCacheAffinityMetric)),
            ],
            Duration::from_millis(10),
        );
        queue.push_back(queued_request(
            "default",
            Some("cold"),
            now - Duration::from_secs(601),
        ));
        queue.push_back(queued_request("high", Some("warm"), now));
        let context = resource_snapshot_with_cached_workflow("warm");

        assert_eq!(
            queue
                .rank_ready_candidates(&context, now)
                .into_iter()
                .next()
                .map(|selection| selection.request.payload.workflow),
            Some("default".to_string())
        );
    }

    #[test]
    fn warm_affinity_metric_rewards_cached_eligible_worker() {
        let metric = WarmCacheAffinityMetric;
        let now = Instant::now();
        let context = resource_snapshot_with_cached_workflow("demo");

        assert_eq!(
            metric.score(
                &queued_request("demo", Some("demo"), now),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: true,
                },
                now
            ),
            1.0
        );
        assert_eq!(
            metric.score(
                &queued_request("other", Some("other"), now),
                &context,
                QueueCandidateStatus::Eligible {
                    has_warm_worker: false,
                },
                now
            ),
            0.0
        );
    }
}
