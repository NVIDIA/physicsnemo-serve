#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint script for generic inference workers.
#
# This script:
# 1. Starts Redis server and waits for readiness
# 2. Generates supervisord.conf dynamically based on WORKERS env var
# 3. Runs supervisord to manage worker processes with automatic restart
#
# Environment Variables:
#   WORKERS          - Which workers to run (default: all)
#                      Values: all, server, prepare, fanout, scheduler,
#                      prefetch, collect, postprocess, publish, results, execute, gpu,
#                      or comma-separated list
#   REDIS_URL        - Redis connection URL (default: redis://127.0.0.1:6379)
#   QUEUE_CONFIG     - Stream configuration path (default: /app/scripts/worker_runtime_config.json)
#   REDIS_STREAM_PREFIX - Stream key prefix for worker-runtime roles (default: physicsnemo:)
#   INFERENCE_STREAM - Inference server target stream (default: ${REDIS_STREAM_PREFIX}inference)
#   PREFETCH_STREAM  - Inference server prefetch stream (default: ${REDIS_STREAM_PREFIX}prefetch)
#   WORKER_RUNTIME_CONFIG - worker-runtime role config path for role mode
#                           (scheduler/prefetch/publish/results, default: /app/scripts/worker_runtime_config.json)
#   LOG_LEVEL        - Logging level for workers (default: info)
#   REDIS_MAXMEMORY  - Redis max memory (default: 256mb)
#   SERVER_PORT      - HTTP server listen port (default: 8080)
#   WORKERS_PER_GPU  - Number of execute workers per GPU when launch config omits workers_per_device (default: 1)
#   RUNTIME_ENV_LAUNCHER_SCRIPT - Runtime-env-based execute launcher (default: /app/scripts/runtime_env_launcher.py)
#   HEALTH_STUB_PORT - Port for health stub proxy (default: 8001, set as Lepton endpoint port)
#   HEALTH_STUB_ENABLED - Enable/disable health stub (default: true)
#
# Usage:
#   # Run everything including the execute launcher (default)
#   docker run -p 8080:8080 --gpus all e2s-rust
#
#   # Run only scheduler
#   docker run -e WORKERS=scheduler e2s-rust
#
#   # Run HTTP server only
#   docker run -p 8080:8080 -e WORKERS=server e2s-rust
#
#   # Run all orchestration workers without execute pools
#   docker run -p 8080:8080 -e WORKERS=server,prepare,fanout,scheduler,prefetch,collect,postprocess,publish,results e2s-rust
#
#   # Run just prefetch worker
#   docker run -e WORKERS=prefetch e2s-rust

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

WORKERS="${WORKERS:-all}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
QUEUE_CONFIG="${QUEUE_CONFIG:-/app/scripts/worker_runtime_config.json}"
REDIS_STREAM_PREFIX="${REDIS_STREAM_PREFIX:-physicsnemo:}"
INFERENCE_STREAM="${INFERENCE_STREAM:-${REDIS_STREAM_PREFIX}inference}"
PREFETCH_STREAM="${PREFETCH_STREAM:-${REDIS_STREAM_PREFIX}prefetch}"
DEFAULT_WORKER_RUNTIME_CONFIG="/app/scripts/worker_runtime_config.json"
WORKER_RUNTIME_CONFIG="${WORKER_RUNTIME_CONFIG:-$DEFAULT_WORKER_RUNTIME_CONFIG}"
if [[ -z "${PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG:-}" ]] || \
   [[ "$PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG" == "$DEFAULT_WORKER_RUNTIME_CONFIG" && \
      "$WORKER_RUNTIME_CONFIG" != "$DEFAULT_WORKER_RUNTIME_CONFIG" ]]; then
    PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG="$WORKER_RUNTIME_CONFIG"
fi
PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE="${PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE:-python3}"
LOG_LEVEL="${LOG_LEVEL:-info}"
REDIS_MAXMEMORY="${REDIS_MAXMEMORY:-256mb}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
HEALTH_STUB_PORT="${HEALTH_STUB_PORT:-8001}"
HEALTH_STUB_ENABLED="${HEALTH_STUB_ENABLED:-true}"
SUPERVISORD_CONF="/tmp/supervisord.conf"

# Worker binary paths
INFERENCE_SERVER_BIN="${INFERENCE_SERVER_BIN:-/app/bin/inference-server}"
WORKER_RUNTIME_BIN="${WORKER_RUNTIME_BIN:-/app/bin/worker-runtime}"
RUNTIME_ENV_LAUNCHER_SCRIPT="${RUNTIME_ENV_LAUNCHER_SCRIPT:-${GPU_LAUNCHER_SCRIPT:-/app/scripts/runtime_env_launcher.py}}"
SERVER_PORT="${SERVER_PORT:-8080}"
SCHEDULER_METRICS_PORT="${SCHEDULER_METRICS_PORT:-9101}"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[entrypoint] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

error() {
    echo "[entrypoint] ERROR: $*" >&2
}

die() {
    error "$@"
    exit 1
}

# Check if a worker type is requested
is_worker_requested() {
    local worker_type="$1"
    
    # "all" means all workers
    if [[ "$WORKERS" == "all" ]]; then
        return 0
    fi
    
    # Check if worker_type is in the comma-separated list
    if [[ ",$WORKERS," == *",$worker_type,"* ]]; then
        return 0
    fi
    
    return 1
}

# Check if a binary exists and is executable
check_binary() {
    local binary_path="$1"
    local binary_name="$2"
    
    if [[ ! -x "$binary_path" ]]; then
        log "WARNING: $binary_name binary not found at $binary_path (skipping)"
        return 1
    fi
    return 0
}

# =============================================================================
# Redis Startup
# =============================================================================

start_redis() {
    log "Starting Redis server..."
    
    # Start Redis with configuration
    redis-server \
        --daemonize yes \
        --bind 127.0.0.1 \
        --port 6379 \
        --maxmemory "$REDIS_MAXMEMORY" \
        --maxmemory-policy allkeys-lru \
        --appendonly no \
        --save "" \
        --loglevel notice
    
    # Wait for Redis to be ready (max 30 seconds)
    local max_attempts=30
    local attempt=0
    
    while ! redis-cli ping 2>/dev/null | grep -q PONG; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            die "Redis failed to start after ${max_attempts} seconds"
        fi
        sleep 1
    done
    
    log "Redis is ready (took ${attempt}s)"
}

# =============================================================================
# Prometheus Configuration Generation
# =============================================================================

generate_prometheus_config() {
    log "Generating Prometheus scrape config at /tmp/prometheus.yml"
    cat > /tmp/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: physicsnemo_serve_server
    metrics_path: /v1/metrics
    static_configs:
      - targets: ['localhost:${SERVER_PORT}']
EOF
    if is_worker_requested "scheduler"; then
        cat >> /tmp/prometheus.yml << EOF

  - job_name: physicsnemo_serve_scheduler
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:${SCHEDULER_METRICS_PORT}']
EOF
    fi
}

# =============================================================================
# Supervisord Configuration Generation
# =============================================================================

generate_supervisord_config() {
    log "Generating supervisord configuration for: $WORKERS"
    
    # Start with base configuration
    cat > "$SUPERVISORD_CONF" << 'EOF'
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
logfile_maxbytes=50MB
logfile_backups=3
loglevel=info
pidfile=/tmp/supervisord.pid
childlogdir=/tmp

[unix_http_server]
file=/tmp/supervisor.sock
chmod=0700

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock
EOF

    local programs_added=0
    
    # Add inference server (HTTP API) if requested (priority 50 = mid priority)
    if is_worker_requested "server"; then
        if check_binary "$INFERENCE_SERVER_BIN" "inference-server"; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:inference-server]
command=$INFERENCE_SERVER_BIN
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=50
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",SERVER_PORT="$SERVER_PORT",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",INFERENCE_STREAM="$INFERENCE_STREAM",PREFETCH_STREAM="$PREFETCH_STREAM",PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG="$PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + inference-server (HTTP API on port $SERVER_PORT, priority=50)"
        fi
    fi
    
    # Add scheduler worker if requested (priority 100 = starts after execute launcher)
    if is_worker_requested "scheduler"; then
        local scheduler_cmd=""
        local scheduler_display_name=""
        local scheduler_priority=100

        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/scheduler-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="scheduler"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role scheduler --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/scheduler-runtime-wrapper.sh
                scheduler_cmd="/tmp/scheduler-runtime-wrapper.sh"
                scheduler_display_name="worker-runtime (role=scheduler)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$scheduler_cmd" ]]; then
            # If execute workers are also requested, use wrapper that waits for registration
            if is_worker_requested "execute" || is_worker_requested "gpu"; then
                cat > /tmp/scheduler-wrapper.sh << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper script that waits for execute worker registration before starting scheduler
WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-15}"
echo "[scheduler-wrapper] Waiting for execute worker registration (timeout: ${WAIT_TIMEOUT}s)..."
for i in $(seq 1 $WAIT_TIMEOUT); do
    count=$(redis-cli HLEN gpu:registry 2>/dev/null || echo "0")
    if [[ "$count" -gt 0 ]]; then
        echo "[scheduler-wrapper] Worker registration detected: $count worker(s)"
        break
    fi
    sleep 1
done
echo "[scheduler-wrapper] Starting scheduler..."
exec "$@"
WRAPPER_EOF
                chmod +x /tmp/scheduler-wrapper.sh
                scheduler_cmd="/tmp/scheduler-wrapper.sh $scheduler_cmd"
                log "  + $scheduler_display_name (with execute wait wrapper, priority=$scheduler_priority)"
            else
                log "  + $scheduler_display_name (priority=$scheduler_priority)"
            fi
            
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-scheduler]
command=$scheduler_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=$scheduler_priority
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG",GPU_WAIT_TIMEOUT="15",HEALTH_PORT="$SCHEDULER_METRICS_PORT"
EOF
            programs_added=$((programs_added + 1))
        else
            log "WARNING: no scheduler binary available (worker-runtime)"
        fi
    fi
    
    # Add prepare worker if requested (priority 40 = starts before downstream stages)
    if is_worker_requested "prepare"; then
        local prepare_cmd=""
        local prepare_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/prepare-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="prepare"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role prepare --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/prepare-runtime-wrapper.sh
                prepare_cmd="/tmp/prepare-runtime-wrapper.sh"
                prepare_display_name="worker-runtime (role=prepare)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$prepare_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-prepare]
command=$prepare_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=40
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $prepare_display_name (priority=40)"
        else
            log "WARNING: no prepare worker binary available (worker-runtime)"
        fi
    fi

    # Add fanout worker if requested (priority 45 = starts before scheduler)
    if is_worker_requested "fanout"; then
        local fanout_cmd=""
        local fanout_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/fanout-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="fanout"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role fanout --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/fanout-runtime-wrapper.sh
                fanout_cmd="/tmp/fanout-runtime-wrapper.sh"
                fanout_display_name="worker-runtime (role=fanout)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$fanout_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-fanout]
command=$fanout_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=45
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $fanout_display_name (priority=45)"
        else
            log "WARNING: no fanout worker binary available (worker-runtime)"
        fi
    fi

    # Add prefetch worker if requested (priority 50 = mid priority)
    if is_worker_requested "prefetch"; then
        local prefetch_cmd=""
        local prefetch_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/prefetch-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="prefetch"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role prefetch --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/prefetch-runtime-wrapper.sh
                prefetch_cmd="/tmp/prefetch-runtime-wrapper.sh"
                prefetch_display_name="worker-runtime (role=prefetch)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$prefetch_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-prefetch]
command=$prefetch_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=50
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $prefetch_display_name (priority=50)"
        else
            log "WARNING: no prefetch worker binary available (worker-runtime)"
        fi
    fi

    # Add collect worker if requested (priority 55 = after execute results arrive)
    if is_worker_requested "collect"; then
        local collect_cmd=""
        local collect_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/collect-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="collect"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role collect --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/collect-runtime-wrapper.sh
                collect_cmd="/tmp/collect-runtime-wrapper.sh"
                collect_display_name="worker-runtime (role=collect)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$collect_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-collect]
command=$collect_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=55
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $collect_display_name (priority=55)"
        else
            log "WARNING: no collect worker binary available (worker-runtime)"
        fi
    fi

    # Add postprocess worker if requested (priority 55 = after collect/execute)
    if is_worker_requested "postprocess"; then
        local postprocess_cmd=""
        local postprocess_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/postprocess-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="postprocess"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role postprocess --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/postprocess-runtime-wrapper.sh
                postprocess_cmd="/tmp/postprocess-runtime-wrapper.sh"
                postprocess_display_name="worker-runtime (role=postprocess)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$postprocess_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-postprocess]
command=$postprocess_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=55
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $postprocess_display_name (priority=55)"
        else
            log "WARNING: no postprocess worker binary available (worker-runtime)"
        fi
    fi
    
    # Add publish worker if requested (priority 55 = after postprocess/execute and results)
    if is_worker_requested "publish"; then
        local publish_cmd=""
        local publish_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/publish-runtime-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="publish"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role publish --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/publish-runtime-wrapper.sh
                publish_cmd="/tmp/publish-runtime-wrapper.sh"
                publish_display_name="worker-runtime (role=publish)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$publish_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-publish]
command=$publish_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
priority=55
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG"
EOF
            programs_added=$((programs_added + 1))
            log "  + $publish_display_name (priority=55)"
        else
            log "WARNING: no publish worker binary available (worker-runtime)"
        fi
    fi

    # Add results worker if requested (priority 50 = starts before publish)
    if is_worker_requested "results"; then
        local results_cmd=""
        local results_display_name=""
        if check_binary "$WORKER_RUNTIME_BIN" "worker-runtime"; then
            if [[ -f "$WORKER_RUNTIME_CONFIG" ]]; then
                cat > /tmp/results-wrapper.sh << EOF
#!/bin/bash
set -euo pipefail
export WORKER_ROLE="results"
export WORKER_PIPELINE_CONFIG="$WORKER_RUNTIME_CONFIG"
exec "$WORKER_RUNTIME_BIN" --role results --config-path "$WORKER_RUNTIME_CONFIG"
EOF
                chmod +x /tmp/results-wrapper.sh
                results_cmd="/tmp/results-wrapper.sh"
                results_display_name="worker-runtime (role=results)"
            else
                log "WARNING: worker-runtime config not found at $WORKER_RUNTIME_CONFIG"
            fi
        fi

        if [[ -n "$results_cmd" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:worker-runtime-results]
command=$results_cmd
directory=/app
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=30
stopsignal=TERM
priority=50
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=RUST_LOG="$LOG_LEVEL",REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",REDIS_STREAM_PREFIX="$REDIS_STREAM_PREFIX"
EOF
            programs_added=$((programs_added + 1))
            log "  + $results_display_name (priority=50)"
        else
            log "WARNING: no results worker binary available (worker-runtime)"
        fi
    fi
    
    # Add execute launcher if requested (priority 10 = starts first)
    if is_worker_requested "execute" || is_worker_requested "gpu"; then
        if [[ -f "$RUNTIME_ENV_LAUNCHER_SCRIPT" ]]; then
            cat >> "$SUPERVISORD_CONF" << EOF

[program:runtime-env-launcher]
command=$PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE $RUNTIME_ENV_LAUNCHER_SCRIPT
directory=/app
autostart=true
autorestart=true
startsecs=10
startretries=3
stopwaitsecs=60
stopsignal=TERM
priority=10
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
environment=REDIS_URL="$REDIS_URL",QUEUE_CONFIG="$QUEUE_CONFIG",WORKER_SCRIPT="/app/scripts/inference_worker.py",WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}",WORKER_RUNTIME_CONFIG="$WORKER_RUNTIME_CONFIG",PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG="$PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG",PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE="$PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE",STREAM_PREFIX="$REDIS_STREAM_PREFIX"
EOF
            programs_added=$((programs_added + 1))
            log "  + runtime-env-launcher (priority=10, starts first)"
        else
            log "WARNING: runtime env launcher script not found at $RUNTIME_ENV_LAUNCHER_SCRIPT (skipping)"
        fi
    fi
    
    # Add Prometheus metrics sidecar when the inference server is running
    if is_worker_requested "server"; then
        if [[ -x "/app/bin/prometheus" ]]; then
            generate_prometheus_config
            cat >> "$SUPERVISORD_CONF" << EOF

[program:prometheus]
command=/app/bin/prometheus --config.file=/tmp/prometheus.yml --storage.tsdb.path=/tmp/prometheus-data --storage.tsdb.retention.time=6h --web.listen-address=:9090
directory=/tmp
autostart=true
autorestart=true
startsecs=5
startretries=3
stopwaitsecs=10
stopsignal=TERM
priority=40
redirect_stderr=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
EOF
            programs_added=$((programs_added + 1))
            log "  + prometheus (metrics storage on port 9090, priority=40)"
        else
            log "WARNING: prometheus binary not found at /app/bin/prometheus (skipping metrics sidecar)"
        fi
    fi

    if [[ $programs_added -eq 0 ]]; then
        die "No workers configured! Check WORKERS environment variable: $WORKERS"
    fi
    
    log "Generated supervisord config with $programs_added program(s)"
}

# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

shutdown_handler() {
    log "!!!! EXTERNAL SIGTERM RECEIVED !!!!"
    log "!!!! This signal came from OUTSIDE the container (Lepton/K8s orchestrator) !!!!"
    log "!!!! Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ') !!!!"
    log "Stopping workers..."
    
    # Stop supervisord gracefully
    if [[ -f /tmp/supervisord.pid ]]; then
        local pid
        pid=$(cat /tmp/supervisord.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
            # Wait for supervisord to stop (max 60 seconds)
            local timeout=60
            while kill -0 "$pid" 2>/dev/null && [[ $timeout -gt 0 ]]; do
                sleep 1
                timeout=$((timeout - 1))
            done
        fi
    fi
    
    # Stop Redis
    log "Stopping Redis..."
    redis-cli shutdown nosave 2>/dev/null || true
    
    log "Shutdown complete"
    exit 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "Starting inference container"

    # Start health stub immediately so Lepton's port probe gets a 200 on /health
    # while Redis and supervisord are still booting.
    if [[ "${HEALTH_STUB_ENABLED:-true}" == "true" ]] && [[ -f /app/scripts/health_stub.py ]]; then
        local stub_port="${HEALTH_STUB_PORT:-8001}"
        log "Starting health stub on port $stub_port (proxies to $SERVER_PORT)"
        python3 /app/scripts/health_stub.py &
    fi

    log "Workers: $WORKERS"
    log "Redis URL: $REDIS_URL"
    log "Queue Config: $QUEUE_CONFIG"
    log "Redis Stream Prefix: $REDIS_STREAM_PREFIX"
    log "Inference Stream: $INFERENCE_STREAM"
    log "Prefetch Stream: $PREFETCH_STREAM"
    
    # Set up signal handlers for graceful shutdown
    trap shutdown_handler SIGTERM SIGINT SIGHUP
    
    # Start Redis
    start_redis
    
    # Generate supervisord configuration
    generate_supervisord_config
    
    # Run supervisord in background so the shell trap can catch SIGTERM.
    # Without this, `exec` replaces the shell and the trap is dead code.
    log "Starting supervisord..."
    supervisord -c "$SUPERVISORD_CONF" &
    SUPERVISORD_PID=$!
    wait $SUPERVISORD_PID
}

# Run main function
main "$@"
