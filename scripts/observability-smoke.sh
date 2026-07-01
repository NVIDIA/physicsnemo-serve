#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Observability v0 — smoke-test script
#
# Validates the /v1/metrics endpoint, GPU/Redis/API metric presence,
# the /prometheus/* reverse proxy, and end-to-end Prometheus query path.
#
# Usage:
#   ./scripts/observability-smoke.sh <SERVICE_URL> <BEARER_TOKEN>
#
# Example:
#   ./scripts/observability-smoke.sh https://my-deployment.lepton.run "tok_abc123"
# ---------------------------------------------------------------------------
set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 --url <SERVICE_URL> --token <BEARER_TOKEN>

  --url    Base URL of your Lepton deployment (no trailing slash)
  --token  Bearer authentication token
  --help   Show this help message

Example:
  $0 --url https://my-deployment.lepton.run --token "tok_abc123"

Tests performed:
  1. GET /v1/metrics          Endpoint reachable, returns valid Prometheus text format
  2. GPU metrics              NVML gauges present (warns if no GPUs attached)
  3. CPU / system metrics     CPU usage, per-core, memory, load average gauges
  4. Redis stream metrics     physicsnemo_serve_redis_stream_length gauges populated
  5. API request metrics      Counters and histograms recorded after generating traffic
  6. Prometheus health        /prometheus/-/healthy confirms the sidecar is alive
  7. Prometheus label query   Verifies scraped metric names via the reverse proxy
  8. Prometheus instant query End-to-end PromQL query through scrape → TSDB → proxy

Note: tests 5-7 require ~15-30s after container start for the first Prometheus
scrape cycle. If they warn on first run, wait and retry.
EOF
    exit 0
}

die_usage() {
    echo "Usage: $0 --url <SERVICE_URL> --token <BEARER_TOKEN>"
    echo "Run '$0 --help' for details."
    exit 1
}

SERVICE_URL=""
EP_TOKEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)   SERVICE_URL="${2%/}"; shift 2 ;;
        --token) EP_TOKEN="$2";        shift 2 ;;
        --help|-h) usage ;;
        *)       echo "Unknown argument: $1"; die_usage ;;
    esac
done

if [[ -z "$SERVICE_URL" || -z "$EP_TOKEN" ]]; then
    die_usage
fi

PASS=0
FAIL=0
WARN=0

green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

pass() { PASS=$((PASS + 1)); green "  ✓ $1"; }
fail() { FAIL=$((FAIL + 1)); red   "  ✗ $1"; }
warn() { WARN=$((WARN + 1)); yellow "  ⚠ $1"; }

AUTH=(-H "Authorization: Bearer ${EP_TOKEN}")

FETCH_BODY_FILE=$(mktemp)
HTTP_CODE=""

# Helper: curl with auth. Writes body to $FETCH_BODY_FILE, sets $HTTP_CODE.
# Read the body with: cat "$FETCH_BODY_FILE"
# Never exits on curl failure — sets HTTP_CODE="000" if the request fails entirely.
fetch() {
    local url="$1"
    HTTP_CODE=$(curl -s --connect-timeout 10 --max-time 30 -o "$FETCH_BODY_FILE" -w '%{http_code}' "${AUTH[@]}" "$url") || HTTP_CODE="000"
}

cleanup() { rm -f "$FETCH_BODY_FILE"; }
trap cleanup EXIT

bold "============================================"
bold " PhysicsNeMo Serve Observability v0 — Smoke Tests"
bold "============================================"
echo ""
echo "Target: $SERVICE_URL"
echo ""

# -------------------------------------------------------------------------
bold "[1/7] GET /v1/metrics — endpoint reachable"
# -------------------------------------------------------------------------
fetch "$SERVICE_URL/v1/metrics"
METRICS=$(cat "$FETCH_BODY_FILE")

if [[ "$HTTP_CODE" != "200" ]]; then
    warn "/v1/metrics returned HTTP $HTTP_CODE — trying /metrics fallback"
    fetch "$SERVICE_URL/metrics"
    METRICS=$(cat "$FETCH_BODY_FILE")
fi

if [[ "$HTTP_CODE" == "200" ]]; then
    pass "/v1/metrics returned HTTP 200"
else
    fail "/v1/metrics returned HTTP $HTTP_CODE (expected 200)"
fi

if echo "$METRICS" | grep -q "^# HELP"; then
    pass "/v1/metrics body contains Prometheus exposition headers"
else
    fail "/v1/metrics body missing '# HELP' lines — not valid Prometheus format"
fi

# -------------------------------------------------------------------------
bold "[2/7] GPU metrics present"
# -------------------------------------------------------------------------
if echo "$METRICS" | grep -q "physicsnemo_serve_gpu_compute_utilization_percent"; then
    pass "physicsnemo_serve_gpu_compute_utilization_percent found"
    if echo "$METRICS" | grep -q "physicsnemo_serve_gpu_memory_used_bytes"; then
        pass "physicsnemo_serve_gpu_memory_used_bytes found"
    else
        fail "physicsnemo_serve_gpu_memory_used_bytes missing"
    fi
else
    warn "No GPU metrics found — NVML may be unavailable (OK if no GPUs attached)"
fi

# -------------------------------------------------------------------------
bold "[3/8] CPU / system metrics present"
# -------------------------------------------------------------------------
if echo "$METRICS" | grep -q "physicsnemo_serve_cpu_usage_percent"; then
    pass "physicsnemo_serve_cpu_usage_percent found"
else
    warn "physicsnemo_serve_cpu_usage_percent missing — sysinfo poller may not have run yet"
fi

if echo "$METRICS" | grep -q "physicsnemo_serve_cpu_core_usage_percent"; then
    CORE_COUNT=$(echo "$METRICS" | grep -c 'physicsnemo_serve_cpu_core_usage_percent{' || true)
    pass "physicsnemo_serve_cpu_core_usage_percent found ($CORE_COUNT core(s))"
else
    warn "physicsnemo_serve_cpu_core_usage_percent missing — sysinfo poller may not have run yet"
fi

if echo "$METRICS" | grep -q "physicsnemo_serve_system_memory_used_bytes"; then
    pass "physicsnemo_serve_system_memory_used_bytes found"
else
    warn "physicsnemo_serve_system_memory_used_bytes missing"
fi

if echo "$METRICS" | grep -q "physicsnemo_serve_load_average"; then
    pass "physicsnemo_serve_load_average found"
else
    warn "physicsnemo_serve_load_average missing"
fi

# -------------------------------------------------------------------------
bold "[4/8] Redis stream metrics present"
# -------------------------------------------------------------------------
if echo "$METRICS" | grep -q "physicsnemo_serve_redis_stream_length"; then
    STREAM_COUNT=$(echo "$METRICS" | grep -c 'physicsnemo_serve_redis_stream_length{' || true)
    pass "physicsnemo_serve_redis_stream_length found ($STREAM_COUNT stream(s))"
else
    warn "No Redis stream metrics — poller may not have run yet (wait ~10s and retry)"
fi

# -------------------------------------------------------------------------
bold "[5/8] API request metrics recorded"
# -------------------------------------------------------------------------
# Generate a bit of traffic first so counters are non-empty
fetch "$SERVICE_URL/healthz"
fetch "$SERVICE_URL/v1/infer/workflows"

fetch "$SERVICE_URL/v1/metrics"
METRICS2=$(cat "$FETCH_BODY_FILE")

if echo "$METRICS2" | grep -q "physicsnemo_serve_api_requests_total"; then
    pass "physicsnemo_serve_api_requests_total counter present"
    if echo "$METRICS2" | grep 'physicsnemo_serve_api_requests_total' | grep -q 'status_class="2xx"'; then
        pass 'status_class="2xx" label seen on counters'
    else
        warn 'status_class="2xx" not yet visible — may need a successful request first'
    fi
else
    fail "physicsnemo_serve_api_requests_total missing — middleware may not be wired"
fi

if echo "$METRICS2" | grep -q "physicsnemo_serve_api_request_duration_seconds"; then
    pass "physicsnemo_serve_api_request_duration_seconds histogram present"
else
    fail "physicsnemo_serve_api_request_duration_seconds missing"
fi

# -------------------------------------------------------------------------
bold "[6/8] Prometheus health check via proxy"
# -------------------------------------------------------------------------
fetch "$SERVICE_URL/prometheus/-/healthy"

if [[ "$HTTP_CODE" == "200" ]]; then
    pass "/prometheus/-/healthy returned HTTP 200"
else
    fail "/prometheus/-/healthy returned HTTP $HTTP_CODE — Prometheus may not be running"
fi

# -------------------------------------------------------------------------
bold "[7/8] Prometheus API — list scraped metric names"
# -------------------------------------------------------------------------
fetch "$SERVICE_URL/prometheus/api/v1/label/__name__/values"
LABEL_RESP=$(cat "$FETCH_BODY_FILE")

if [[ "$HTTP_CODE" == "200" ]]; then
    pass "/prometheus/api/v1/label/__name__/values returned HTTP 200"
    if echo "$LABEL_RESP" | grep -q "physicsnemo_serve_api_requests_total"; then
        pass "Prometheus has scraped physicsnemo_serve_api_requests_total"
    else
        warn "Prometheus responded but hasn't scraped physicsnemo_serve metrics yet (wait ~15-30s for first scrape)"
    fi
else
    fail "Prometheus label query returned HTTP $HTTP_CODE"
fi

# -------------------------------------------------------------------------
bold "[8/8] Prometheus instant query (end-to-end)"
# -------------------------------------------------------------------------
fetch "$SERVICE_URL/prometheus/api/v1/query?query=physicsnemo_serve_api_requests_total"
QUERY_RESP=$(cat "$FETCH_BODY_FILE")

if [[ "$HTTP_CODE" == "200" ]]; then
    pass "/prometheus/api/v1/query returned HTTP 200"
    if echo "$QUERY_RESP" | grep -q '"resultType"'; then
        pass "Response contains valid Prometheus query result structure"
    else
        warn "Response is HTTP 200 but doesn't look like a Prometheus query result"
    fi
    if echo "$QUERY_RESP" | grep -q '"result":\[\]'; then
        warn "Query returned empty result set — Prometheus may not have scraped yet"
    else
        pass "Query returned non-empty result set — full pipeline confirmed"
    fi
else
    fail "Prometheus instant query returned HTTP $HTTP_CODE"
fi

# -------------------------------------------------------------------------
echo ""
bold "============================================"
bold " Results"
bold "============================================"
green "  Passed:   $PASS"
if [[ $WARN -gt 0 ]]; then
    yellow "  Warnings: $WARN"
fi
if [[ $FAIL -gt 0 ]]; then
    red "  Failed:   $FAIL"
else
    echo "  Failed:   0"
fi
echo ""

if [[ $FAIL -gt 0 ]]; then
    red "Some checks failed. Review output above."
    exit 1
elif [[ $WARN -gt 0 ]]; then
    yellow "All hard checks passed. Warnings may resolve after ~30s (first Prometheus scrape)."
    exit 0
else
    green "All checks passed — observability stack is fully operational."
    exit 0
fi
