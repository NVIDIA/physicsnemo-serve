# PhysicsNeMo Serve Observability — v0 Architecture

This document describes the v0 observability stack for PhysicsNeMo Serve: how metrics are
collected, stored, and visualized. It is designed for a team deploying PhysicsNeMo Serve
as a Docker container on Lepton.ai (or any single-container environment) where
only one port is publicly exposed.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Docker Container (Lepton Cloud)                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  inference-server  :8080                                        │ │
│  │                                                                 │ │
│  │   ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌────────┐│ │
│  │   │ Axum Metrics  │  │ NVML Poller   │  │ Redis Stream │  │sysinfo ││ │
│  │   │ Middleware     │  │ (10s interval)│  │ Poller (10s) │  │CPU Poll││ │
│  │   └──────┬───────┘  └──────┬────────┘  └──────┬───────┘  └───┬────┘│ │
│  │          │                 │                   │              │     │ │
│  │          ▼                 ▼                   ▼              ▼     │ │
│  │   ┌──────────────────────────────────────────────────┐         │ │
│  │   │  prometheus-client Registry (in-process)          │         │ │
│  │   │  exposed at GET /v1/metrics                       │         │ │
│  │   └──────────────────────────────────────────────────┘         │ │
│  │                                                                 │ │
│  │   GET /prometheus/* ──► reverse proxy ──► localhost:9090        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐   │
│  │  Redis :6379      │    │  Prometheus :9090                    │   │
│  │  (streams, runs)  │    │  scrapes /v1/metrics every 15s       │   │
│  │                   │    │  TSDB retention: 6 hours              │   │
│  └──────────────────┘    └──────────────────────────────────────┘   │
│                                                                      │
│  supervisord manages: redis, prometheus, inference-server, workers   │
└──────────────────────────────────────────────────────────────────────┘

         │
         │  Lepton exposes :8080 only
         ▼

┌──────────────────────────────────────────────────────┐
│  Developer Workstation                                │
│                                                       │
│  Grafana :3000                                        │
│    Data source: https://<lepton-url>/prometheus        │
│    (routed through the /prometheus/* reverse proxy)    │
└──────────────────────────────────────────────────────┘
```

---

## Observability Stack — Layered Architecture

The observability system is organized into five distinct layers. Each layer has
a single responsibility, and data flows strictly upward — from raw hardware
counters at the bottom to interactive dashboards at the top.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — VISUALIZATION                                                │
│                                                                         │
│    Grafana (developer workstation :3000)                                │
│    Queries Prometheus via HTTPS through the /prometheus/* proxy          │
│    PromQL → time-series panels, bar charts, gauges, heatmaps            │
│                                                                         │
│    [v1+] Dashboard-as-code (versioned JSON provisioning)                │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — API / ACCESS                                                 │
│                                                                         │
│    inference-server (Axum) :8080                                        │
│    ├─ GET /v1/metrics         serves prometheus-client text encoding     │
│    └─ GET|POST /prometheus/*  reverse proxy → localhost:9090             │
│                                                                         │
│    Axum Metrics Middleware                                               │
│    └─ Wraps every HTTP request: records method, path_template,          │
│       status_class (2xx/4xx/5xx/other) into Counter + Histogram         │
│                                                                         │
│    Lepton exposes only :8080; /prometheus/* tunnels all Prometheus       │
│    API traffic through this single port                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — STORAGE / RETENTION                                          │
│                                                                         │
│    Prometheus :9090 (in-container sidecar, managed by supervisord)       │
│    ├─ Scrapes GET /v1/metrics every 15 seconds                          │
│    ├─ TSDB at /tmp/prometheus-data (ephemeral, 6-hour retention)        │
│    └─ Exposes full PromQL query API on localhost:9090                    │
│                                                                         │
│    [v1+] Remote-write to Thanos / Mimir / Grafana Cloud for             │
│          persistent cross-restart retention                              │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — INSTRUMENTATION                                              │
│                                                                         │
│    prometheus-client Registry (in-process, lock-free)                    │
│    ├─ GPU Poller Task (Tokio, 10s interval)                             │
│    │   Calls NVML → updates Gauge families for utilization + memory     │
│    ├─ CPU/System Poller (sysinfo, same 10s task)                        │
│    │   Reads global + per-core CPU %, system RAM, load averages         │
│    ├─ Redis Stream Poller Task (Tokio, 10s interval)                    │
│    │   Calls XLEN per stream → updates Gauge family for stream length   │
│    └─ Axum Metrics Middleware (on every HTTP request)                    │
│        Increments Counter family + observes Histogram for duration       │
│                                                                         │
│    [v1+] OpenTelemetry SDK: distributed tracing spans across            │
│          pipeline stages (prepare → execute → collect → results)         │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 0 — HARDWARE / DATA SOURCES                                      │
│                                                                         │
│    NVIDIA GPU(s)         Host CPU / RAM       Redis :6379               │
│    ├─ Compute engine     ├─ Global CPU %      ├─ Stream queues          │
│    ├─ Memory bus         ├─ Per-core CPU %    │  (prepare, prefetch,    │
│    ├─ VRAM (used/total)  ├─ RAM used/total    │   batch, schedule, …)   │
│    └─ via libnvidia-ml   ├─ Load avg 1/5/15m  └─ via redis XLEN        │
│       (NVML / nvml-      └─ via sysinfo crate                           │
│        wrapper crate)                          Axum HTTP server          │
│                                                ├─ Request method         │
│                                                ├─ Route pattern          │
│                                                ├─ Status code            │
│                                                └─ Wall-clock duration    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 0 — Hardware and Data Sources

This is the raw substrate from which all metrics originate.

**NVIDIA GPUs** expose hardware performance counters through the NVIDIA Management
Library (NVML). The `nvml-wrapper` Rust crate provides safe FFI bindings to
`libnvidia-ml.so`, which is loaded dynamically at runtime via `libloading` (no
compile-time NVML dependency). The library candidates are resolved in priority
order: first any paths in the `SCHEDULER_NVML_LIB_PATHS` environment variable,
then `libnvidia-ml.so`, then `libnvidia-ml.so.1`. If none are found, NVML
initialization fails gracefully and GPU metrics are simply absent.

From NVML, we extract two classes of data per device:
- **Utilization rates** (`nvmlDeviceGetUtilizationRates`): percentage of time
  over the last sample period that the GPU compute engines and the memory bus
  were active. These are device-wide figures — NVML does not attribute
  utilization to individual processes at this API level.
- **Memory info** (`nvmlDeviceGetMemoryInfo`): bytes of GPU memory currently
  allocated vs. total installed. This is useful for detecting memory pressure
  and estimating headroom for additional workflows.

**Host CPU and system memory** are read via the `sysinfo` crate, which parses
`/proc/stat`, `/proc/meminfo`, and `/proc/loadavg` under the hood. Inside a
Docker container these files reflect the *host* machine (not cgroup-limited
values), which is the right view for v0 — it tells you how loaded the overall
node is. The crate requires two refresh cycles to compute CPU usage deltas, so
the poller performs a warm-up refresh at startup before the first real tick.

**Redis** is the message broker connecting all PhysicsNeMo Serve pipeline stages. Each
stage reads from and writes to Redis streams (e.g. `physicsnemo:prepare`,
`physicsnemo:schedule`). The `XLEN` command returns the number of entries
currently pending in a stream — a direct proxy for queue depth and
backpressure in the pipeline.

**The Axum HTTP server** is itself a data source. Every inbound request carries
implicit telemetry: what method was used, which route was hit, what status code
was returned, and how long the handler took. This data doesn't come from an
external system — it is captured inline by the metrics middleware as requests
flow through the server.

### Layer 1 — Instrumentation

The instrumentation layer bridges raw data sources to a structured metrics
model. It runs entirely inside the `inference-server` process as Rust code.

The central component is the **`prometheus-client` Registry** — a thread-safe,
lock-free in-memory data structure that holds all metric families (counters,
gauges, histograms). It lives as an `Arc<PhysicsnemoServeMetrics>` on `AppState`,
shared between background tasks and HTTP handlers.

Four producers feed the registry:

1. **GPU Poller** — A Tokio background task spawned at server startup. Every 10
   seconds it iterates over all NVML devices, calls `utilization_rates()` and
   `memory_info()`, and writes the values into `Gauge<f64, AtomicU64>` families
   keyed by `gpu_id`. The 10-second interval is a balance between freshness and
   NVML call overhead. If NVML is unavailable, this task is a no-op.

2. **CPU/System Poller** — The same background task refreshes a `sysinfo::System`
   instance on each tick. It writes overall CPU usage (`global_cpu_usage()`),
   per-core CPU usage (one gauge per logical core), host memory used/total, and
   1/5/15-minute load averages. The `sysinfo` crate performs the `/proc` reads
   synchronously, which is fast (microseconds) on Linux.

3. **Redis Stream Poller** — The same background task also polls Redis. It issues
   `XLEN` against each of the 8 known pipeline streams (using the configured
   `REDIS_STREAM_PREFIX`), and writes the counts into a `Gauge` family keyed by
   the logical stream name (e.g. `stream="schedule"`). The prefix is stripped
   from the label to keep it environment-agnostic. Redis errors are logged at
   `debug` level to avoid log noise when Redis is temporarily unreachable.

4. **Axum Metrics Middleware** — A Tower middleware layer installed on the Axum
   router. It wraps every HTTP request:
   - Records the start time before calling the inner handler.
   - After the response is produced, extracts the status code and the matched
     route template (via `axum::extract::MatchedPath`).
   - Increments a `Counter` family with labels `{method, path_template,
     status_class}` and observes the elapsed duration in a `Histogram` family
     with labels `{method, path_template}`.
   - Route templates (e.g. `/v1/infer/:name/run`) are used instead of concrete
     paths (e.g. `/v1/infer/earth2-deterministic/run`) to keep cardinality
     bounded. `run_id` is intentionally excluded from all labels.

The registry is never serialized to disk. It exists only in memory and is
re-populated from scratch on every container restart. This is acceptable for
v0 — the Prometheus TSDB provides persistence across the retention window.

**Future (v1+):** The instrumentation layer is where OpenTelemetry would be
introduced. An OTel SDK could emit tracing spans for each pipeline stage
(prepare → prefetch → batch → schedule → execute → collect → postprocess →
results), enabling distributed-trace-style flame graphs of individual request
lifecycles. The existing `prometheus-client` metrics would remain unchanged;
OTel would add a parallel tracing signal, not replace the metrics signal.

### Layer 2 — Storage and Retention

Prometheus runs as a **sidecar process** inside the same Docker container,
managed by supervisord with priority 40 (after Redis, before the inference
server). It is a stock Prometheus v3.3.0 static binary at `/app/bin/prometheus`.

On container startup, `entrypoint.sh` generates a minimal scrape configuration
at `/tmp/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: physicsnemo_serve_server
    metrics_path: /v1/metrics
    static_configs:
      - targets: ['localhost:8080']

  - job_name: physicsnemo_serve_scheduler
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:9101']
```

Every 15 seconds, Prometheus issues local scrapes for the inference server and,
when the scheduler is running, the scheduler worker-runtime process. It parses
the text exposition responses into its **TSDB** (time-series database),
stored at `/tmp/prometheus-data`. The TSDB is ephemeral — it lives in the
container's tmp filesystem and is lost on restart. Retention is capped at 6
hours (`--storage.tsdb.retention.time=6h`) to limit disk usage in the ephemeral
volume.

The critical detail is that Prometheus scrapes `localhost` — this is an
intra-container HTTP call that never touches Lepton's ingress layer or the
public internet. Even though Lepton intercepts external `GET /metrics` requests,
the internal scrape uses `/v1/metrics` on `localhost` and is unaffected.

Prometheus also exposes its full **PromQL query API** on `localhost:9090`. This
is not directly reachable from outside the container (Lepton only exposes port
8080), which is why Layer 3 exists.

**Future (v1+):** Prometheus supports `remote_write` to ship samples to an
external long-term store (Thanos, Mimir, Grafana Cloud). Adding a
`remote_write` stanza to the generated `prometheus.yml` would provide persistent
retention across container restarts without changing any other layer.

### Layer 3 — API and Access

This layer stitches the internal observability components to the outside world
through the inference server's existing Axum router on port 8080.

**`GET /v1/metrics`** — The metrics endpoint calls
`prometheus_client::encoding::text::encode()` on the in-process registry and
returns the result with content type `text/plain; version=0.0.4; charset=utf-8`.
This is a synchronous, CPU-only operation (no I/O, no locks beyond atomics) and
typically completes in microseconds. It is the target of Prometheus's scrape
cycle, and can also be curled directly for debugging. A secondary route at
`GET /metrics` serves the same handler for local development convenience, but is
not usable through Lepton's ingress (Lepton intercepts that path for its own
monitoring).

**`GET|POST /prometheus/*rest`** — The reverse proxy handler. When Grafana (or
any HTTP client) sends a request to `https://<lepton-url>/prometheus/api/v1/query?query=...`,
the Axum router matches the `/prometheus/*rest` wildcard and invokes the proxy
handler. The handler:

1. Extracts the `rest` path segment and the original query string.
2. Constructs a target URL: `http://localhost:9090/{rest}?{query_string}`.
3. Forwards the full request (method, headers minus `Host`, body) to the
   in-container Prometheus using `reqwest::Client`.
4. Returns Prometheus's response (status, headers, body) verbatim to the caller.

If Prometheus is not running or unreachable, the handler returns
`503 Service Unavailable` with a descriptive error message. The target URL is
configurable via the `PROMETHEUS_URL` environment variable (default:
`http://localhost:9090`).

**Axum Metrics Middleware** also lives at this layer architecturally — it is a
router-level Tower layer that intercepts every request before it reaches any
handler. Notably, requests to `/v1/metrics` and `/prometheus/*` are themselves
measured by the middleware, so you can observe the overhead and frequency of
metrics scraping and proxy calls in the metrics themselves.

### Layer 4 — Visualization

Grafana runs on the developer's workstation (or a shared team server). It
connects to Prometheus through the reverse proxy using a standard Prometheus
data source pointed at `https://<lepton-url>/prometheus`.

From Grafana's perspective, the proxy is invisible — it behaves exactly like a
normal Prometheus endpoint. All PromQL queries, label lookups, metadata
requests, and range queries pass through transparently.

For teams with multiple Lepton deployments (e.g. staging, production, canary),
each deployment is added as a separate Prometheus data source in Grafana. A
single Grafana dashboard can use template variables to switch between them.

**Future (v1+):** Export Grafana dashboards as JSON and version them in the
PhysicsNeMo Serve repository for one-command provisioning.

### Why In-Container Prometheus?

- **Single scrape configuration**: Prometheus runs inside the container and scrapes
  `localhost:8080/v1/metrics`. No external configuration needed.
- **Team access**: Every team member points Grafana at the same
  `https://<lepton-url>/prometheus` data source. No one needs to run their own
  Prometheus instance or individually scrape the `/v1/metrics` endpoint.
- **Multi-endpoint**: If you have multiple Lepton deployments, each has its own
  Prometheus. Grafana simply adds one data source per deployment.
- **Data consistency**: Everyone sees the same time-series history (up to the 6-hour
  retention window). Late-joining teammates don't miss data.
- **Acceptable trade-off**: Container restarts clear the TSDB. This is fine for a
  v0 proof-of-concept.

---

## Metrics Catalog

All metric names are prefixed with `physicsnemo_serve_`.

### GPU Metrics (NVML)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `physicsnemo_serve_gpu_compute_utilization_percent` | Gauge | `gpu_id` | GPU compute engine utilization (0–100%) |
| `physicsnemo_serve_gpu_memory_bus_utilization_percent` | Gauge | `gpu_id` | GPU memory bus utilization (0–100%) |
| `physicsnemo_serve_gpu_memory_used_bytes` | Gauge | `gpu_id` | GPU memory currently in use (bytes) |
| `physicsnemo_serve_gpu_memory_total_bytes` | Gauge | `gpu_id` | GPU total installed memory (bytes) |

**Notes:**
- Polled every 10 seconds via a background Tokio task.
- `gpu_id` is the NVML device index (`"0"`, `"1"`, …).
- If NVML is unavailable (e.g. no GPU drivers), a warning is logged at startup
  and GPU metrics are silently skipped. All other metrics continue working.

### CPU / System Metrics (sysinfo)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `physicsnemo_serve_cpu_usage_percent` | Gauge | — | Aggregate CPU utilization across all cores (0–100%) |
| `physicsnemo_serve_cpu_core_usage_percent` | Gauge | `core` | Per-logical-core CPU utilization (0–100%) |
| `physicsnemo_serve_system_memory_used_bytes` | Gauge | — | Host system memory currently in use (bytes) |
| `physicsnemo_serve_system_memory_total_bytes` | Gauge | — | Host system total memory (bytes) |
| `physicsnemo_serve_load_average` | Gauge | `window` | System load average (`1m`, `5m`, `15m`) |

**Notes:**
- Polled every 10 seconds via the same background Tokio task as GPU/Redis metrics.
- `core` is the logical core index (`"0"`, `"1"`, …). On a 96-core machine this
  produces 96 time series — acceptable cardinality for Prometheus.
- Inside a Docker container, `/proc/stat` and `/proc/meminfo` reflect the
  **host** machine, not cgroup limits. This is the intended view for v0.
- The `sysinfo` crate requires two refresh cycles to compute CPU deltas; the
  poller performs a warm-up refresh at startup so the first real tick returns
  valid data.

### Redis Stream Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `physicsnemo_serve_redis_stream_length` | Gauge | `stream` | Number of pending entries in a Redis stream |

**Monitored streams:** `prepare`, `prefetch`, `batch`, `schedule`, `release`,
`collect`, `postprocess`, `results`.

**Notes:**
- Polled every 10 seconds using `XLEN` on each stream.
- Stream keys are constructed from `REDIS_STREAM_PREFIX` + logical name
  (e.g. `physicsnemo:schedule`), but the `stream` label uses the logical name only.
- If Redis is unavailable, XLEN errors are logged at debug level; metrics are not updated.

### API Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `physicsnemo_serve_api_requests_total` | Counter | `method`, `path_template`, `status_class` | Total HTTP requests |
| `physicsnemo_serve_api_request_duration_seconds` | Histogram | `method`, `path_template` | Request latency distribution |

**Label details:**

- `method`: HTTP method (`GET`, `POST`, etc.)
- `path_template`: The Axum route pattern, **not** the resolved URL.
  Examples: `/v1/infer/:name/run`, `/v1/infer/:workflow/:run_id/status`,
  `/healthz`, `/v1/metrics`.
- `status_class`: One of `2xx`, `4xx`, `5xx`, `other`.

**Cardinality note:** Using route templates instead of concrete paths ensures a
bounded label set. `run_id` is intentionally excluded from labels to avoid
cardinality explosion.

### Scheduler Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `physicsnemo_serve_scheduler_attempts_total` | Counter | `outcome` | Scheduler decision attempts (`dispatched`, `blocked`, `dropped`, `failed`) |
| `physicsnemo_serve_scheduler_queue_depth` | Gauge | — | Current in-memory scheduler queue depth |
| `physicsnemo_serve_scheduler_queue_wait_seconds` | Histogram | `outcome` | Time from scheduler enqueue until a terminal outcome (`dispatched`, `dropped`, `failed`) |
| `physicsnemo_serve_scheduler_attempt_duration_seconds` | Histogram | `outcome` | Duration of one scheduler decision attempt |
| `physicsnemo_serve_scheduler_discovered_workers` | Gauge | — | Number of GPU worker resources currently known to the scheduler |

**Notes:**
- Scheduler metrics are emitted by the `worker-runtime` scheduler process at
  `GET /metrics` on the internal scheduler metrics port (default `9101`).
- Prometheus scrapes this internal endpoint and exposes the time series through
  the existing `/prometheus/*` query proxy.

---

## Endpoints

### `GET /v1/metrics`

Returns the current state of all registered metrics in Prometheus text exposition
format (`text/plain; version=0.0.4; charset=utf-8`).

Also available at `GET /metrics` (for local/direct access). The `/v1/metrics`
path is the primary endpoint because some PaaS platforms (including Lepton)
intercept the bare `/metrics` path for their own monitoring infrastructure.

This is the endpoint that the in-container Prometheus scrapes every 15 seconds.
You can also `curl` it directly for debugging:

```bash
curl https://<lepton-url>/v1/metrics
```

### `GET|POST /prometheus/*`

Reverse proxy to `localhost:9090` (the in-container Prometheus). Allows Grafana
to query the Prometheus API through the single exposed port.

The target URL is configurable via the `PROMETHEUS_URL` environment variable
(default: `http://localhost:9090`). If Prometheus is not running, the proxy
returns `503 Service Unavailable` with a descriptive error message.

**Example — test from your workstation:**

```bash
# Query all metric names
curl https://<lepton-url>/prometheus/api/v1/label/__name__/values

# Run an instant query
curl 'https://<lepton-url>/prometheus/api/v1/query?query=physicsnemo_serve_gpu_compute_utilization_percent'
```

---

## Infrastructure

### Prometheus Sidecar

Prometheus runs as a supervisord-managed process inside the Docker container:

- **Binary**: `/app/bin/prometheus` (static linux-amd64 build, ~70 MB)
- **Config**: `/tmp/prometheus.yml` (generated by `entrypoint.sh`)
- **TSDB**: `/tmp/prometheus-data` (ephemeral, lost on container restart)
- **Retention**: 6 hours (`--storage.tsdb.retention.time=6h`)
- **Scrape interval**: 15 seconds
- **Supervisord priority**: 40 (starts after Redis at 5, before inference-server at 50)

### Supervisord Process Order

| Priority | Process | Port |
|----------|---------|------|
| 5 | Redis | 6379 |
| 10 | runtime-env-launcher (GPU execute workers) | — |
| 40 | **Prometheus** | 9090 |
| 40 | worker-runtime prepare | — |
| 45 | worker-runtime fanout, batch | — |
| 50 | **inference-server** | 8080 |
| 50 | worker-runtime prefetch, results | — |
| 55 | worker-runtime collect, postprocess | — |
| 100 | worker-runtime scheduler | — |

---

## Setting Up Grafana

The repository includes a self-provisioning Grafana setup in `observability/`.
One command starts Grafana with the data source and dashboard pre-configured.

### Prerequisites

- Docker and Docker Compose installed on your workstation.
- A running PhysicsNeMo Serve deployment on Lepton (or any host with the observability
  stack enabled).
- Your deployment URL and bearer token.

### Quick Start

```bash
cd /path/to/physicsnemo-serve

SERVICE_URL=https://your-deployment.lepton.run \
EP_TOKEN=your-bearer-token \
docker compose -f observability/docker-compose.yml up
```

Open **http://localhost:3000** in your browser (no login required — anonymous
access is enabled with Admin role).

The **PhysicsNeMo Serve Overview** dashboard loads automatically as the home page.
Data appears as soon as Prometheus has completed at least one scrape cycle
(~15 seconds after container start).

To stop Grafana:

```bash
docker compose -f observability/docker-compose.yml down
```

### What's Provisioned

The Docker Compose setup auto-configures everything — no manual steps:

| Component | File | What it does |
|-----------|------|-------------|
| Data source | `observability/grafana/provisioning/datasources/prometheus.yml` | Connects Grafana to `$SERVICE_URL/prometheus` with bearer auth |
| Dashboard loader | `observability/grafana/provisioning/dashboards/dashboards.yml` | Tells Grafana to load dashboards from disk on startup |
| Dashboard | `observability/grafana/dashboards/physicsnemo-serve-overview.json` | The PhysicsNeMo Serve Overview dashboard with all panels |

### Multiple Deployments

To monitor multiple Lepton endpoints simultaneously, you can either:

1. **Switch the data source** — stop Grafana, change `SERVICE_URL`, restart.
2. **Add data sources manually** — log into Grafana, go to Connections → Data
   Sources, add another Prometheus source pointing to a different deployment URL.
   Use Grafana's data source selector in each panel to pick which deployment to
   query.

### Dashboard Panels

The provisioned dashboard includes panels across 5 sections:

| Section | Panel | Type | PromQL |
|---------|-------|------|--------|
| **GPU Metrics** | GPU Overview | Table | All GPU metrics joined by `gpu_id` — compute %, memory bus %, memory used/total |
| | GPU Compute Utilization | Time series | `physicsnemo_serve_gpu_compute_utilization_percent` per GPU over time |
| | GPU Memory Bus Utilization | Time series | `physicsnemo_serve_gpu_memory_bus_utilization_percent` per GPU over time |
| | GPU 0 — Memory | Pie chart | Used vs free VRAM for GPU 0 |
| | GPU 1 — Memory | Pie chart | Used vs free VRAM for GPU 1 |
| **CPU / System** | Overall CPU Usage | Time series | `physicsnemo_serve_cpu_usage_percent` — aggregate CPU % over time |
| | Per-Core CPU Usage | Time series | `physicsnemo_serve_cpu_core_usage_percent` — one line per logical core |
| | System Memory | Pie chart | Used vs free host RAM (same style as GPU memory pies) |
| | System Load Average | Time series | `physicsnemo_serve_load_average` — 1m / 5m / 15m load averages |
| **Redis Stream** | Redis Stream Occupancy | Time series | `physicsnemo_serve_redis_stream_length` per stream — one line per pipeline stage |
| **HTTP API** | API Request Rate | Bar chart | `rate(physicsnemo_serve_api_requests_total[5m])` grouped by endpoint and status class |
| | API Latency (p50/p99) | Time series | `histogram_quantile` on `physicsnemo_serve_api_request_duration_seconds_bucket` |
| **Scheduler** | Scheduler Scrape Health | Stat | `up{job="physicsnemo_serve_scheduler"}` |
| | Scheduler GPU Worker Count | Stat | `physicsnemo_serve_scheduler_discovered_workers` |
| | Scheduler Queue Depth | Time series | `physicsnemo_serve_scheduler_queue_depth` |
| | Scheduler Queue Wait (p50/p99) | Time series | `histogram_quantile` on `physicsnemo_serve_scheduler_queue_wait_seconds_bucket` |
| | Scheduler Decision Time (p50/p99) | Time series | `histogram_quantile` on `physicsnemo_serve_scheduler_attempt_duration_seconds_bucket` |
| | Scheduler Launch Latency (p50/p99) | Time series | Approximate queue-wait percentile plus matching attempt-duration percentile |
| | Scheduler Attempt Outcomes | Time series | Cumulative `physicsnemo_serve_scheduler_attempts_total` by outcome |

### Editing the Dashboard

The dashboard JSON is mounted read-only from disk. To make changes:

1. Edit panels in the Grafana UI (the provisioned dashboard allows UI edits).
2. When you're satisfied, click the dashboard settings gear → **JSON Model** →
   copy the JSON.
3. Paste it into `observability/grafana/dashboards/physicsnemo-serve-overview.json`.
4. Restart Grafana to pick up the change (or wait ~10 seconds for the file
   watcher).

This keeps the dashboard versioned in git while allowing interactive editing.

---

## Graceful Degradation

The observability stack is designed to degrade gracefully:

| Failure Mode | Effect |
|---|---|
| **NVML unavailable** (no GPU, driver mismatch) | Warning logged at startup. GPU metrics are absent; all other metrics work. |
| **sysinfo `/proc` unreadable** | CPU/system metrics return zero. Extremely unlikely on Linux; possible in exotic sandboxes. |
| **Redis unavailable** | Stream length metrics are not updated. Errors logged at debug level. API, GPU, and CPU metrics work. |
| **Prometheus not running** | `/v1/metrics` still works (served by the Rust app). `/prometheus/*` proxy returns 503. |
| **Grafana disconnected** | No effect on data collection. Prometheus continues scraping. Reconnect Grafana and data is available from the retention window. |

---

## What Is NOT in Scope for v0

| Feature | Reason | Target |
|---------|--------|--------|
| Per-workflow GPU utilization | Requires per-process NVML attribution via compute instances or MIG. Complex data collection. | v1 |
| Grafana dashboard JSON provisioning | Manual dashboard creation for the initial PoC. | v1 |
| Persistent TSDB storage | Ephemeral storage is acceptable for a v0 PoC. | v1 |
| Authentication on `/prometheus/*` | Relies on Lepton's existing auth layer. | v1 |
| Alerting rules | Manual monitoring is acceptable initially. | v1 |
| Distributed tracing (OpenTelemetry) | Current pull-based Prometheus model is simpler for v0. | v1+ |

---

## v1 Roadmap Notes

- **Per-workflow GPU attribution**: Investigate NVML per-process queries
  (`nvmlDeviceGetComputeRunningProcesses`) to break down GPU utilization by
  workflow name. Requires mapping PID → workflow name through the scheduler's
  GPU registry.
- **Dashboard-as-code**: Export Grafana dashboards as JSON and version them in this
  repository for repeatable provisioning.
- **Persistent storage**: Consider a remote Prometheus / Thanos / Mimir backend for
  long-term retention across container restarts.
- **OpenTelemetry integration**: Add distributed tracing spans across pipeline
  stages (prepare → schedule → execute → collect → results) for end-to-end
  request latency attribution.
- **Alerting**: Define Prometheus alerting rules for GPU memory exhaustion,
  stream backpressure, and elevated error rates. Route through Alertmanager to
  Slack / PagerDuty.
