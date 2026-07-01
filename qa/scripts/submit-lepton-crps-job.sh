#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit a Lepton batch job that runs Earth2Studio's compare_crps.py.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/deploy/config.sh"

JOB_NAME="${JOB_NAME:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_NAME="${IMAGE_NAME:-$(_cfg docker_registry)/$(_cfg python_service_image)}"
FORECAST_A="${FORECAST_A:-}"
FORECAST_B="${FORECAST_B:-}"
THRESHOLD="${THRESHOLD:-0.01}"
VARIABLES="${VARIABLES:-}"
LEAD_TIME_CHUNK_SIZE="${LEAD_TIME_CHUNK_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
COMPARISON_SCRIPT="${COMPARISON_SCRIPT:-/workspace/earth2studio-project/serve/server/scripts/compare_crps.py}"

LEPTON_WORKSPACE_ID="${LEPTON_WORKSPACE_ID:-$(_cfg lepton_workspace_id)}"
LEPTON_WORKSPACE_URL="${LEPTON_WORKSPACE_URL:-}"
LEPTON_WORKSPACE_TOKEN="${LEPTON_WORKSPACE_TOKEN:-}"
LEPTON_NODE_GROUP="${LEPTON_NODE_GROUP:-$(_cfg lepton_node_group)}"
LEPTON_RESOURCE_SHAPE="${LEPTON_RESOURCE_SHAPE:-gpu.h100-sxm}"
LEPTON_LUSTRE_STORAGE="${LEPTON_LUSTRE_STORAGE:-lustre}"
LEPTON_NFS_PATH="${LEPTON_NFS_PATH:-$(_cfg nfs_mount_base)/crps_tests_$(date -u +%Y%m%d)}"
LEPTON_MOUNT_TARGET="${LEPTON_MOUNT_TARGET:-/outputs}"
LEPTON_PULL_SECRET="${LEPTON_PULL_SECRET:-$(_cfg pull_secret)}"
ARTIFACT_LOG="${ARTIFACT_LOG:-}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-3600}"
JOB_POLL_INTERVAL_SECONDS="${JOB_POLL_INTERVAL_SECONDS:-30}"
JOB_LOG_FLUSH_DELAY_SECONDS="${JOB_LOG_FLUSH_DELAY_SECONDS:-15}"
REPORT_TAIL_LINES="${REPORT_TAIL_LINES:-220}"
REPORT_FETCH_TIMEOUT_SECONDS="${REPORT_FETCH_TIMEOUT_SECONDS:-240}"
REPORT_READER_START_DELAY_SECONDS="${REPORT_READER_START_DELAY_SECONDS:-30}"
REPORT_READER_HOLD_SECONDS="${REPORT_READER_HOLD_SECONDS:-120}"
REPORT_READER_ATTACH_DELAY_SECONDS="${REPORT_READER_ATTACH_DELAY_SECONDS:-10}"
KEEP_JOB=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: submit-lepton-crps-job.sh [options]

Required:
  --job-name NAME
  --image-tag TAG_OR_REF       Image tag or full image reference
  --forecast-a PATH            Baseline forecast Zarr path
  --forecast-b PATH            Candidate forecast Zarr path
  --workspace-token TOKEN      LEPTON_WORKSPACE_TOKEN

Options:
  --image-name NAME            Image name when --image-tag is only a tag
  --threshold VALUE            Relative CRPS threshold (default: 0.01)
  --variables LIST             Optional comma-separated variables
  --lead-time-chunk-size N     compare_crps.py lead-time chunk size (default: 1)
  --device DEVICE              compare_crps.py device (default: cuda)
  --comparison-script PATH     compare_crps.py path inside the container
  --workspace-id ID
  --workspace-url URL
  --node-group NAME
  --resource-shape SHAPE
  --nfs-path PATH
  --mount-target PATH
  --lustre-storage NAME
  --pull-secret SECRET
  --artifact-log PATH
  --job-timeout SECONDS
  --job-poll-interval SECONDS
  --keep-job                  Do not remove the Lepton job after completion
  --dry-run                   Print commands without submitting
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --image-tag) IMAGE_TAG="$2"; shift 2 ;;
        --image-name) IMAGE_NAME="$2"; shift 2 ;;
        --forecast-a) FORECAST_A="$2"; shift 2 ;;
        --forecast-b) FORECAST_B="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --variables) VARIABLES="$2"; shift 2 ;;
        --lead-time-chunk-size) LEAD_TIME_CHUNK_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --comparison-script) COMPARISON_SCRIPT="$2"; shift 2 ;;
        --workspace-id) LEPTON_WORKSPACE_ID="$2"; shift 2 ;;
        --workspace-url) LEPTON_WORKSPACE_URL="$2"; shift 2 ;;
        --workspace-token) LEPTON_WORKSPACE_TOKEN="$2"; shift 2 ;;
        --node-group) LEPTON_NODE_GROUP="$2"; shift 2 ;;
        --resource-shape) LEPTON_RESOURCE_SHAPE="$2"; shift 2 ;;
        --nfs-path) LEPTON_NFS_PATH="$2"; shift 2 ;;
        --mount-target) LEPTON_MOUNT_TARGET="$2"; shift 2 ;;
        --lustre-storage) LEPTON_LUSTRE_STORAGE="$2"; shift 2 ;;
        --pull-secret) LEPTON_PULL_SECRET="$2"; shift 2 ;;
        --artifact-log) ARTIFACT_LOG="$2"; shift 2 ;;
        --job-timeout) JOB_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --job-poll-interval) JOB_POLL_INTERVAL_SECONDS="$2"; shift 2 ;;
        --keep-job) KEEP_JOB=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

require() {
    local label="$1" value="$2"
    if [[ -z "$value" ]]; then
        echo "Error: $label is required" >&2
        exit 2
    fi
}

require "job name" "$JOB_NAME"
require "image tag" "$IMAGE_TAG"
require "forecast-a" "$FORECAST_A"
require "forecast-b" "$FORECAST_B"
require "workspace id" "$LEPTON_WORKSPACE_ID"

if [[ -n "$ARTIFACT_LOG" ]]; then
    mkdir -p "$(dirname "$ARTIFACT_LOG")"
    exec > >(tee -a "$ARTIFACT_LOG") 2>&1
fi

if ! command -v lep >/dev/null 2>&1; then
    echo "Error: 'lep' CLI not found. Install with: pip install -U leptonai" >&2
    exit 1
fi

last_slash="${IMAGE_TAG##*/}"
if [[ "$last_slash" == *:* ]]; then
    IMAGE_FULL="$IMAGE_TAG"
else
    IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
fi

shell_quote() {
    printf "%q" "$1"
}

compare_args=(
    python "$COMPARISON_SCRIPT"
    --forecast-a "$FORECAST_A"
    --forecast-b "$FORECAST_B"
    --threshold "$THRESHOLD"
    --lead-time-chunk-size "$LEAD_TIME_CHUNK_SIZE"
    --device "$DEVICE"
)
if [[ -n "$VARIABLES" ]]; then
    compare_args+=(--variables "$VARIABLES")
fi

REMOTE_REPORT_DIR="${LEPTON_MOUNT_TARGET%/}/crps-reports"
REMOTE_REPORT_PATH="${REMOTE_REPORT_DIR}/${JOB_NAME}.txt"
compare_command="mkdir -p $(shell_quote "$REMOTE_REPORT_DIR"); "
for arg in "${compare_args[@]}"; do
    compare_command+="$(shell_quote "$arg") "
done
compare_command+="2>&1 | tee $(shell_quote "$REMOTE_REPORT_PATH"); rc=\${PIPESTATUS[0]}; exit \$rc"

job_args=(
    --name "$JOB_NAME"
    --container-image "$IMAGE_FULL"
    --node-group "$LEPTON_NODE_GROUP"
    --resource-shape "$LEPTON_RESOURCE_SHAPE"
    --image-pull-secrets "$LEPTON_PULL_SECRET"
    --mount "${LEPTON_NFS_PATH}:${LEPTON_MOUNT_TARGET}:node-nfs:${LEPTON_LUSTRE_STORAGE}"
    --command "$compare_command"
)

echo "==> CRPS comparison job configuration"
echo "    job-name       : $JOB_NAME"
echo "    image          : $IMAGE_FULL"
echo "    mount          : $LEPTON_NFS_PATH -> $LEPTON_MOUNT_TARGET (storage: $LEPTON_LUSTRE_STORAGE)"
echo "    forecast-a     : $FORECAST_A"
echo "    forecast-b     : $FORECAST_B"
echo "    threshold      : $THRESHOLD"
echo "    report         : $REMOTE_REPORT_PATH"
echo "    command        : $compare_command"

echo "==> Logging into Lepton workspace"
if [[ -n "$LEPTON_WORKSPACE_TOKEN" ]]; then
    LEPTON_CREDENTIALS="${LEPTON_WORKSPACE_ID}:${LEPTON_WORKSPACE_TOKEN}"
    echo "+ lep login -c ${LEPTON_WORKSPACE_ID}:<redacted>${LEPTON_WORKSPACE_URL:+ -u $LEPTON_WORKSPACE_URL}"
elif [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
    echo "Error: --workspace-url requires --workspace-token so the script can log into that workspace" >&2
    exit 2
else
    echo "+ using existing lep login session"
fi
if [[ "$DRY_RUN" -eq 0 && -n "$LEPTON_WORKSPACE_TOKEN" ]]; then
    if [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
        lep login -c "$LEPTON_CREDENTIALS" -u "$LEPTON_WORKSPACE_URL"
    else
        lep login -c "$LEPTON_CREDENTIALS"
    fi
fi

echo "+ lep job create ${job_args[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "==> Dry run complete; job not submitted."
    exit 0
fi

create_output="$(lep job create "${job_args[@]}" 2>&1)"
printf '%s\n' "$create_output"
JOB_ID="$(printf '%s\n' "$create_output" | sed -n 's/^[[:space:]]*ID:[[:space:]]*//p' | tail -1)"
if [[ -z "$JOB_ID" ]]; then
    echo "Error: could not parse Lepton job ID from create output" >&2
    exit 1
fi

log_stream_pid=""
echo "==> Streaming Lepton job logs"
(
    for attempt in 1 2 3 4 5; do
        lep job log -i "$JOB_ID" 2>&1 && break
        sleep 5
    done
) &
log_stream_pid="$!"

echo "==> Polling Lepton job: $JOB_NAME ($JOB_ID)"
deadline=$(( $(date +%s) + JOB_TIMEOUT_SECONDS ))
last_status=""
while [[ $(date +%s) -lt $deadline ]]; do
    status_out="$(lep job get -i "$JOB_ID" 2>&1 || true)"
    if [[ "$status_out" != "$last_status" ]]; then
        printf '%s\n' "$status_out" | sed 's/^/    /'
        last_status="$status_out"
    fi
    if printf '%s\n' "$status_out" | grep -qiE '"state": "(Completed|Succeeded|Success)"'; then
        echo "==> CRPS comparison job succeeded"
        exit_code=0
        break
    fi
    if printf '%s\n' "$status_out" | grep -qiE '"state": "(Failed|Cancelled|Stopped|Error)"'; then
        echo "Error: CRPS comparison job failed" >&2
        exit_code=1
        break
    fi
    sleep "$JOB_POLL_INTERVAL_SECONDS"
done

if [[ "${exit_code+x}" != x ]]; then
    echo "Error: CRPS comparison job timed out after ${JOB_TIMEOUT_SECONDS}s" >&2
    exit_code=1
fi

if [[ -n "$log_stream_pid" ]]; then
    # compare_crps.py writes through tee, so the primary job log is the report.
    # Give Lepton log collection a short grace period to flush the final table.
    sleep "$JOB_LOG_FLUSH_DELAY_SECONDS"
    if kill -0 "$log_stream_pid" 2>/dev/null; then
        kill "$log_stream_pid" 2>/dev/null || true
    fi
    wait "$log_stream_pid" 2>/dev/null || true
fi

echo "==> Fetching final CRPS report tail"
report_job_args=(
    --name "${JOB_NAME}-report"
    --container-image "$IMAGE_FULL"
    --node-group "$LEPTON_NODE_GROUP"
    --resource-shape "$LEPTON_RESOURCE_SHAPE"
    --image-pull-secrets "$LEPTON_PULL_SECRET"
    --mount "${LEPTON_NFS_PATH}:${LEPTON_MOUNT_TARGET}:node-nfs:${LEPTON_LUSTRE_STORAGE}"
    --command "sleep $(shell_quote "$REPORT_READER_START_DELAY_SECONDS"); tail -n $(shell_quote "$REPORT_TAIL_LINES") $(shell_quote "$REMOTE_REPORT_PATH"); sleep $(shell_quote "$REPORT_READER_HOLD_SECONDS")"
)
report_create_output="$(lep job create "${report_job_args[@]}" 2>&1 || true)"
printf '%s\n' "$report_create_output"
REPORT_JOB_ID="$(printf '%s\n' "$report_create_output" | sed -n 's/^[[:space:]]*ID:[[:space:]]*//p' | tail -1)"
if [[ -n "$REPORT_JOB_ID" ]]; then
    sleep "$REPORT_READER_ATTACH_DELAY_SECONDS"
    if command -v timeout >/dev/null 2>&1; then
        timeout "$REPORT_FETCH_TIMEOUT_SECONDS" lep job log -i "$REPORT_JOB_ID" 2>&1 || true
    else
        lep job log -i "$REPORT_JOB_ID" 2>&1 || true
    fi
    lep job remove -i "$REPORT_JOB_ID" 2>&1 || true
else
    echo "Warning: could not create CRPS report reader job" >&2
fi

if [[ "$KEEP_JOB" -eq 0 ]]; then
    echo "==> Removing Lepton job: $JOB_NAME ($JOB_ID)"
    lep job remove -i "$JOB_ID" 2>&1 || true
fi

exit "$exit_code"
