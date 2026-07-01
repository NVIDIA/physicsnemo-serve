/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prometheus metrics for worker-runtime roles.
//!
//! The scheduler owns queueing, reservation, and dispatch decisions, so its
//! operational metrics are recorded in this process and scraped by Prometheus
//! alongside inference-server metrics.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use tracing::warn;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SchedulerOutcomeLabels {
    pub outcome: String,
}

pub struct WorkerRuntimeMetrics {
    registry: Registry,

    /// Count of scheduler decision attempts by outcome.
    pub scheduler_attempts_total: Family<SchedulerOutcomeLabels, Counter>,
    /// Current in-memory scheduler backlog.
    pub scheduler_queue_depth: Gauge<f64, AtomicU64>,
    /// Time a scheduler request has spent queued before reaching a terminal outcome.
    pub scheduler_queue_wait_seconds: Family<SchedulerOutcomeLabels, Histogram>,
    /// Duration of one scheduler decision attempt by outcome.
    pub scheduler_attempt_duration_seconds: Family<SchedulerOutcomeLabels, Histogram>,
    /// Number of GPU worker resources currently known to the scheduler.
    pub scheduler_discovered_workers: Gauge<f64, AtomicU64>,
}

impl Default for WorkerRuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRuntimeMetrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let scheduler_attempts_total = Family::default();
        registry.register(
            "physicsnemo_serve_scheduler_attempts",
            "Total scheduler decision attempts by outcome",
            scheduler_attempts_total.clone(),
        );

        let scheduler_queue_depth = Gauge::default();
        registry.register(
            "physicsnemo_serve_scheduler_queue_depth",
            "Current scheduler in-memory queue depth",
            scheduler_queue_depth.clone(),
        );

        let scheduler_queue_wait_seconds =
            Family::<SchedulerOutcomeLabels, Histogram>::new_with_constructor(|| {
                Histogram::new(exponential_buckets(0.001, 2.0, 20))
            });
        registry.register(
            "physicsnemo_serve_scheduler_queue_wait_seconds",
            "Scheduler queue wait time in seconds before a terminal outcome, labeled by outcome",
            scheduler_queue_wait_seconds.clone(),
        );

        let scheduler_attempt_duration_seconds =
            Family::<SchedulerOutcomeLabels, Histogram>::new_with_constructor(|| {
                Histogram::new(exponential_buckets(0.0005, 2.0, 18))
            });
        registry.register(
            "physicsnemo_serve_scheduler_attempt_duration_seconds",
            "Scheduler decision attempt duration in seconds by outcome",
            scheduler_attempt_duration_seconds.clone(),
        );

        let scheduler_discovered_workers = Gauge::default();
        registry.register(
            "physicsnemo_serve_scheduler_discovered_workers",
            "Current number of GPU worker resources discovered by the scheduler",
            scheduler_discovered_workers.clone(),
        );

        Self {
            registry,
            scheduler_attempts_total,
            scheduler_queue_depth,
            scheduler_queue_wait_seconds,
            scheduler_attempt_duration_seconds,
            scheduler_discovered_workers,
        }
    }

    pub fn record_scheduler_attempt(&self, outcome: &str) {
        self.scheduler_attempts_total
            .get_or_create(&SchedulerOutcomeLabels {
                outcome: outcome.to_string(),
            })
            .inc();
    }

    pub fn increment_scheduler_queue_depth(&self) {
        self.scheduler_queue_depth.inc();
    }

    pub fn decrement_scheduler_queue_depth(&self) {
        self.scheduler_queue_depth.dec();
    }

    pub fn observe_scheduler_queue_wait(&self, outcome: &str, seconds: f64) {
        self.scheduler_queue_wait_seconds
            .get_or_create(&SchedulerOutcomeLabels {
                outcome: outcome.to_string(),
            })
            .observe(seconds);
    }

    pub fn observe_scheduler_attempt_duration(&self, outcome: &str, seconds: f64) {
        self.scheduler_attempt_duration_seconds
            .get_or_create(&SchedulerOutcomeLabels {
                outcome: outcome.to_string(),
            })
            .observe(seconds);
    }

    pub fn set_scheduler_discovered_workers(&self, count: usize) {
        self.scheduler_discovered_workers.set(count as f64);
    }

    pub fn encode(&self) -> String {
        let mut buf = String::new();
        if let Err(error) = prometheus_client::encoding::text::encode(&mut buf, &self.registry) {
            warn!(error = %error, "failed to encode worker-runtime Prometheus metrics");
            return String::new();
        }
        buf
    }
}

pub type WorkerMetrics = Arc<WorkerRuntimeMetrics>;

pub fn create_shared_metrics() -> WorkerMetrics {
    Arc::new(WorkerRuntimeMetrics::new())
}
