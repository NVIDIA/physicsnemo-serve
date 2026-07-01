/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prometheus metrics for PhysicsNeMo Serve observability.
//!
//! Exposes GPU telemetry (via NVML), CPU/system telemetry (via sysinfo),
//! Redis stream occupancy, and HTTP API request counters through a
//! `prometheus-client` registry.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Label types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct GpuLabels {
    pub gpu_id: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct StreamLabels {
    pub stream: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ApiLabels {
    pub method: String,
    pub path_template: String,
    pub status_class: StatusClass,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum StatusClass {
    Success2xx,
    ClientError4xx,
    ServerError5xx,
    Other,
}

impl StatusClass {
    pub fn from_status(code: u16) -> Self {
        match code {
            200..=299 => Self::Success2xx,
            400..=499 => Self::ClientError4xx,
            500..=599 => Self::ServerError5xx,
            _ => Self::Other,
        }
    }
}

impl prometheus_client::encoding::EncodeLabelValue for StatusClass {
    fn encode(
        &self,
        encoder: &mut prometheus_client::encoding::LabelValueEncoder,
    ) -> Result<(), std::fmt::Error> {
        use std::fmt::Write;
        let s = match self {
            Self::Success2xx => "2xx",
            Self::ClientError4xx => "4xx",
            Self::ServerError5xx => "5xx",
            Self::Other => "other",
        };
        encoder.write_str(s)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ApiDurationLabels {
    pub method: String,
    pub path_template: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CpuCoreLabels {
    pub core: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LoadAvgLabels {
    pub window: String,
}

// ---------------------------------------------------------------------------
// Metrics struct
// ---------------------------------------------------------------------------

pub struct PhysicsnemoServeMetrics {
    pub registry: Registry,

    // GPU gauges
    pub gpu_compute_utilization: Family<GpuLabels, Gauge<f64, AtomicU64>>,
    pub gpu_memory_bus_utilization: Family<GpuLabels, Gauge<f64, AtomicU64>>,
    pub gpu_memory_used_bytes: Family<GpuLabels, Gauge<f64, AtomicU64>>,
    pub gpu_memory_total_bytes: Family<GpuLabels, Gauge<f64, AtomicU64>>,

    // Redis stream gauges
    pub redis_stream_length: Family<StreamLabels, Gauge<f64, AtomicU64>>,

    // API counters / histograms
    pub api_requests_total: Family<ApiLabels, Counter>,
    pub api_request_duration_seconds: Family<ApiDurationLabels, Histogram>,

    // CPU / system gauges
    pub cpu_usage_percent: Gauge<f64, AtomicU64>,
    pub cpu_core_usage_percent: Family<CpuCoreLabels, Gauge<f64, AtomicU64>>,
    pub system_memory_used_bytes: Gauge<f64, AtomicU64>,
    pub system_memory_total_bytes: Gauge<f64, AtomicU64>,
    pub load_average: Family<LoadAvgLabels, Gauge<f64, AtomicU64>>,

    // NVML handle (None when NVML is unavailable)
    nvml: Option<nvml_wrapper::Nvml>,
}

impl Default for PhysicsnemoServeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsnemoServeMetrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let gpu_compute_utilization = Family::default();
        registry.register(
            "physicsnemo_serve_gpu_compute_utilization_percent",
            "GPU compute utilization percentage (0-100)",
            gpu_compute_utilization.clone(),
        );

        let gpu_memory_bus_utilization = Family::default();
        registry.register(
            "physicsnemo_serve_gpu_memory_bus_utilization_percent",
            "GPU memory bus utilization percentage (0-100)",
            gpu_memory_bus_utilization.clone(),
        );

        let gpu_memory_used_bytes = Family::default();
        registry.register(
            "physicsnemo_serve_gpu_memory_used_bytes",
            "GPU memory currently used in bytes",
            gpu_memory_used_bytes.clone(),
        );

        let gpu_memory_total_bytes = Family::default();
        registry.register(
            "physicsnemo_serve_gpu_memory_total_bytes",
            "GPU total memory in bytes",
            gpu_memory_total_bytes.clone(),
        );

        let redis_stream_length = Family::default();
        registry.register(
            "physicsnemo_serve_redis_stream_length",
            "Number of pending entries in a Redis stream",
            redis_stream_length.clone(),
        );

        let api_requests_total = Family::default();
        registry.register(
            "physicsnemo_serve_api_requests_total",
            "Total HTTP API requests",
            api_requests_total.clone(),
        );

        let api_request_duration_seconds =
            Family::<ApiDurationLabels, Histogram>::new_with_constructor(|| {
                Histogram::new(exponential_buckets(0.001, 2.0, 16))
            });
        registry.register(
            "physicsnemo_serve_api_request_duration_seconds",
            "HTTP API request duration in seconds",
            api_request_duration_seconds.clone(),
        );

        let cpu_usage_percent = Gauge::default();
        registry.register(
            "physicsnemo_serve_cpu_usage_percent",
            "Aggregate CPU utilization percentage across all cores (0-100)",
            cpu_usage_percent.clone(),
        );

        let cpu_core_usage_percent = Family::default();
        registry.register(
            "physicsnemo_serve_cpu_core_usage_percent",
            "Per-logical-core CPU utilization percentage (0-100)",
            cpu_core_usage_percent.clone(),
        );

        let system_memory_used_bytes = Gauge::default();
        registry.register(
            "physicsnemo_serve_system_memory_used_bytes",
            "Host system memory currently used in bytes",
            system_memory_used_bytes.clone(),
        );

        let system_memory_total_bytes = Gauge::default();
        registry.register(
            "physicsnemo_serve_system_memory_total_bytes",
            "Host system total memory in bytes",
            system_memory_total_bytes.clone(),
        );

        let load_average = Family::default();
        registry.register(
            "physicsnemo_serve_load_average",
            "System load average",
            load_average.clone(),
        );

        let nvml = init_nvml();

        Self {
            registry,
            gpu_compute_utilization,
            gpu_memory_bus_utilization,
            gpu_memory_used_bytes,
            gpu_memory_total_bytes,
            redis_stream_length,
            api_requests_total,
            api_request_duration_seconds,
            cpu_usage_percent,
            cpu_core_usage_percent,
            system_memory_used_bytes,
            system_memory_total_bytes,
            load_average,
            nvml,
        }
    }

    /// Poll NVML for GPU telemetry and update gauges.
    pub fn poll_gpu_metrics(&self) {
        let Some(nvml) = &self.nvml else {
            return;
        };

        let device_count = match nvml.device_count() {
            Ok(n) => n,
            Err(e) => {
                warn!(error = %e, "NVML device_count failed");
                return;
            }
        };

        for idx in 0..device_count {
            let device = match nvml.device_by_index(idx) {
                Ok(d) => d,
                Err(e) => {
                    debug!(gpu_id = idx, error = %e, "NVML device_by_index failed");
                    continue;
                }
            };

            let labels = GpuLabels {
                gpu_id: idx.to_string(),
            };

            if let Ok(util) = device.utilization_rates() {
                self.gpu_compute_utilization
                    .get_or_create(&labels)
                    .set(util.gpu as f64);
                self.gpu_memory_bus_utilization
                    .get_or_create(&labels)
                    .set(util.memory as f64);
            }

            if let Ok(mem) = device.memory_info() {
                self.gpu_memory_used_bytes
                    .get_or_create(&labels)
                    .set(mem.used as f64);
                self.gpu_memory_total_bytes
                    .get_or_create(&labels)
                    .set(mem.total as f64);
            }
        }
    }

    /// Poll CPU and system memory metrics via sysinfo and update gauges.
    ///
    /// `sys` must have been refreshed (cpu + memory) by the caller before
    /// calling this method so that the readings are current.
    pub fn poll_cpu_metrics(&self, sys: &sysinfo::System) {
        self.cpu_usage_percent.set(sys.global_cpu_usage() as f64);

        for (i, cpu) in sys.cpus().iter().enumerate() {
            self.cpu_core_usage_percent
                .get_or_create(&CpuCoreLabels {
                    core: i.to_string(),
                })
                .set(cpu.cpu_usage() as f64);
        }

        self.system_memory_used_bytes.set(sys.used_memory() as f64);
        self.system_memory_total_bytes
            .set(sys.total_memory() as f64);

        let load = sysinfo::System::load_average();
        for (window, value) in [("1m", load.one), ("5m", load.five), ("15m", load.fifteen)] {
            self.load_average
                .get_or_create(&LoadAvgLabels {
                    window: window.to_string(),
                })
                .set(value);
        }
    }

    /// Poll Redis XLEN for each known stream and update gauges.
    pub async fn poll_redis_streams(&self, conn: &mut redis::aio::ConnectionManager, prefix: &str) {
        const STREAMS: &[&str] = &[
            "prepare",
            "prefetch",
            "batch",
            "schedule",
            "release",
            "collect",
            "postprocess",
            "results",
        ];

        for logical in STREAMS {
            let key = if prefix.is_empty() {
                (*logical).to_string()
            } else {
                format!("{}{}", prefix, logical)
            };

            match redis::cmd("XLEN").arg(&key).query_async::<i64>(conn).await {
                Ok(len) => {
                    self.redis_stream_length
                        .get_or_create(&StreamLabels {
                            stream: logical.to_string(),
                        })
                        .set(len as f64);
                }
                Err(e) => {
                    debug!(stream = key, error = %e, "XLEN failed");
                }
            }
        }
    }
}

/// Attempt to initialize NVML using common library paths.
fn init_nvml() -> Option<nvml_wrapper::Nvml> {
    let candidates = nvml_library_candidates();

    for path in &candidates {
        let os_path: &std::ffi::OsStr = std::ffi::OsStr::new(path.as_ref());
        match nvml_wrapper::Nvml::builder().lib_path(os_path).init() {
            Ok(nvml) => {
                tracing::info!(lib_path = path.as_ref(), "NVML initialized for metrics");
                return Some(nvml);
            }
            Err(_) => continue,
        }
    }

    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            tracing::info!("NVML initialized (default path)");
            Some(nvml)
        }
        Err(e) => {
            warn!(error = %e, "NVML unavailable — GPU metrics will be skipped");
            None
        }
    }
}

fn nvml_library_candidates() -> Vec<std::borrow::Cow<'static, str>> {
    let mut candidates: Vec<std::borrow::Cow<'static, str>> = Vec::new();

    if let Ok(paths) = std::env::var("SCHEDULER_NVML_LIB_PATHS") {
        for path in paths.split(':') {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                candidates.push(trimmed.to_string().into());
            }
        }
    }
    candidates.push("libnvidia-ml.so".into());
    candidates.push("libnvidia-ml.so.1".into());
    candidates
}

/// Encode the registry to Prometheus text exposition format.
pub fn encode_metrics(metrics: &PhysicsnemoServeMetrics) -> String {
    let mut buf = String::new();
    if let Err(e) = prometheus_client::encoding::text::encode(&mut buf, &metrics.registry) {
        warn!(error = %e, "Failed to encode Prometheus metrics");
        return String::new();
    }
    buf
}

/// Shared handle used by the Axum middleware and background tasks.
pub type SharedMetrics = Arc<PhysicsnemoServeMetrics>;

pub fn create_shared_metrics() -> SharedMetrics {
    Arc::new(PhysicsnemoServeMetrics::new())
}
