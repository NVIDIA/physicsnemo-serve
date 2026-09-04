#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from Earth2Studio's serve/server/scripts/profile_workflow.sh.
# Run this script manually inside a dedicated single-GPU service container.

set -euo pipefail

WORKFLOW_NAME=""
INFER_JSON=""
SERVER_URL="http://127.0.0.1:8080"
TOKEN_ARGUMENT=""
OUTPUT_DIR="."
POLL_INTERVAL_SECONDS=5
TIMEOUT_SECONDS=23400
INSECURE_TLS=false

NVIDIA_SMI_PID=""
OUTPUT_FILE=""

usage() {
    cat <<'EOF'
Usage:
  profile_workflow.sh --wf_name <plugin_id> --wf_json <request.json> [options]

Required:
  --wf_name <plugin_id>       PhysicsNeMo Serve plugin ID
  --wf_json <request.json>    Inference request JSON

Options:
  --server_url <url>          Server URL (default: http://127.0.0.1:8080)
  --ep_token <token>          Bearer token (default: LEPTON_ENDPOINT_TOKEN or EP_TOKEN)
  --output_dir <directory>    Artifact directory (default: current directory)
  --poll_interval_seconds <n> Status polling interval (default: 5)
  --timeout_seconds <n>       Overall workflow timeout (default: 23400)
  --insecure                  Disable TLS certificate verification
  -h, --help                  Show this help

Run this script on the GPU host/container serving the workflow. The generated
profiles_<run_id>.json follows the scheduler's {"profiles": [...]} schema.
Review measurements and retain an operational safety margin before promotion.
EOF
}

require_value() {
    local option=$1
    local value=${2-}
    if [[ -z "$value" ]]; then
        echo "Error: $option requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wf_name)
            require_value "$1" "${2-}"
            WORKFLOW_NAME=$2
            shift 2
            ;;
        --wf_json)
            require_value "$1" "${2-}"
            INFER_JSON=$2
            shift 2
            ;;
        --server_url)
            require_value "$1" "${2-}"
            SERVER_URL=$2
            shift 2
            ;;
        --ep_token)
            require_value "$1" "${2-}"
            TOKEN_ARGUMENT=$2
            shift 2
            ;;
        --output_dir)
            require_value "$1" "${2-}"
            OUTPUT_DIR=$2
            shift 2
            ;;
        --poll_interval_seconds)
            require_value "$1" "${2-}"
            POLL_INTERVAL_SECONDS=$2
            shift 2
            ;;
        --timeout_seconds)
            require_value "$1" "${2-}"
            TIMEOUT_SECONDS=$2
            shift 2
            ;;
        --insecure)
            INSECURE_TLS=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$WORKFLOW_NAME" ]]; then
    echo "Error: --wf_name is required" >&2
    exit 2
fi
if [[ ! "$WORKFLOW_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Error: --wf_name must be a plugin ID without path separators" >&2
    exit 2
fi
if [[ -z "$INFER_JSON" ]]; then
    echo "Error: --wf_json is required" >&2
    exit 2
fi
if [[ ! -f "$INFER_JSON" ]]; then
    echo "Error: JSON file not found: $INFER_JSON" >&2
    exit 2
fi
if [[ ! "$POLL_INTERVAL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Error: --poll_interval_seconds must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --timeout_seconds must be a positive integer" >&2
    exit 2
fi

for command in awk curl jq nvidia-smi python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Error: required command not found: $command" >&2
        exit 1
    fi
done

if ! jq -e . "$INFER_JSON" >/dev/null; then
    echo "Error: invalid JSON file: $INFER_JSON" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
SERVER_URL=${SERVER_URL%/}
EP_TOKEN_VALUE=${TOKEN_ARGUMENT:-${LEPTON_ENDPOINT_TOKEN:-${EP_TOKEN:-}}}

CURL_COMMON=(
    --silent
    --show-error
    --write-out $'\n%{http_code}'
    --max-time 30
)
if [[ "$INSECURE_TLS" == true ]]; then
    CURL_COMMON+=(--insecure)
fi
if [[ -n "$EP_TOKEN_VALUE" ]]; then
    CURL_COMMON+=(-H "Authorization: Bearer ${EP_TOKEN_VALUE}")
fi

stop_nvidia_smi() {
    if [[ -z "$NVIDIA_SMI_PID" ]]; then
        return
    fi
    if kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        kill "$NVIDIA_SMI_PID" 2>/dev/null || true
    fi
    wait "$NVIDIA_SMI_PID" 2>/dev/null || true
    NVIDIA_SMI_PID=""
}

trap stop_nvidia_smi EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    echo "Error: $*" >&2
    exit 1
}

if ! GPU_INDEX_OUTPUT=$(nvidia-smi --query-gpu=index --format=csv,noheader); then
    fail "could not query visible GPUs"
fi
GPU_COUNT=$(awk 'NF { count++ } END { print count + 0 }' <<<"$GPU_INDEX_OUTPUT")
if [[ "$GPU_COUNT" != "1" ]]; then
    fail "profiling requires exactly one visible GPU, found ${GPU_COUNT}"
fi

TEMP_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
OUTPUT_FILE="${OUTPUT_DIR}/profile_${TEMP_ID}.csv"
rm -f "$OUTPUT_FILE"

echo "Starting GPU profiling for workflow: $WORKFLOW_NAME"
nvidia-smi \
    --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv \
    --loop=1 \
    -f "$OUTPUT_FILE" &
NVIDIA_SMI_PID=$!

profile_ready=false
for ((attempt = 0; attempt < 240; attempt++)); do
    if [[ -f "$OUTPUT_FILE" ]] && [[ $(wc -l < "$OUTPUT_FILE") -gt 1 ]]; then
        profile_ready=true
        break
    fi
    if ! kill -0 "$NVIDIA_SMI_PID" 2>/dev/null; then
        fail "nvidia-smi exited before collecting a sample"
    fi
    sleep 0.25
done
if [[ "$profile_ready" != true ]]; then
    fail "nvidia-smi did not collect a sample within 60 seconds"
fi

echo "Submitting inference request to ${SERVER_URL}"
if ! RESPONSE=$(
    curl "${CURL_COMMON[@]}" \
        -X POST \
        "${SERVER_URL}/v1/infer/${WORKFLOW_NAME}/run" \
        -H "Content-Type: application/json" \
        --data-binary "@${INFER_JSON}"
); then
    fail "inference submission request failed"
fi

HTTP_CODE=${RESPONSE##*$'\n'}
RESPONSE_BODY=${RESPONSE%$'\n'*}
if [[ "$HTTP_CODE" != "202" ]]; then
    fail "inference submission returned HTTP ${HTTP_CODE}: ${RESPONSE_BODY}"
fi

RUN_ID=$(jq -r '.run_id // empty' <<<"$RESPONSE_BODY")
INITIAL_STATUS=$(jq -r '.status // empty' <<<"$RESPONSE_BODY")
if [[ -z "$RUN_ID" ]]; then
    fail "inference response did not contain run_id"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    fail "inference response contained an unsafe run_id"
fi
if [[ "$INITIAL_STATUS" != "queued" ]]; then
    fail "expected initial status queued, got: ${INITIAL_STATUS:-<missing>}"
fi

echo "Inference submitted successfully"
echo "Run ID: $RUN_ID"

START_SECONDS=$SECONDS
while true; do
    if ((SECONDS - START_SECONDS >= TIMEOUT_SECONDS)); then
        fail "workflow did not finish within ${TIMEOUT_SECONDS} seconds"
    fi

    sleep "$POLL_INTERVAL_SECONDS"
    if ! STATUS_RESPONSE=$(
        curl "${CURL_COMMON[@]}" \
            -X GET \
            "${SERVER_URL}/v1/infer/${WORKFLOW_NAME}/${RUN_ID}/status" \
            -H "Accept: application/json"
    ); then
        fail "workflow status request failed"
    fi

    STATUS_HTTP_CODE=${STATUS_RESPONSE##*$'\n'}
    STATUS_BODY=${STATUS_RESPONSE%$'\n'*}
    if [[ "$STATUS_HTTP_CODE" != "200" ]]; then
        fail "workflow status returned HTTP ${STATUS_HTTP_CODE}: ${STATUS_BODY}"
    fi

    EXEC_STATUS=$(jq -r '.status // empty' <<<"$STATUS_BODY")
    if [[ -z "$EXEC_STATUS" ]]; then
        fail "workflow status response did not contain status"
    fi
    echo "Status: $EXEC_STATUS"

    case "$EXEC_STATUS" in
        succeeded | completed)
            break
            ;;
        failed | cancelled)
            fail "workflow finished with status ${EXEC_STATUS}: ${STATUS_BODY}"
            ;;
    esac
done

stop_nvidia_smi

FINAL_OUTPUT_FILE="${OUTPUT_DIR}/profile_${RUN_ID}.csv"
OUTPUT_SUMMARY="${OUTPUT_DIR}/outputs_${RUN_ID}.txt"
OUTPUT_PROFILE="${OUTPUT_DIR}/profiles_${RUN_ID}.json"

for path in "$FINAL_OUTPUT_FILE" "$OUTPUT_SUMMARY" "$OUTPUT_PROFILE"; do
    if [[ -e "$path" ]]; then
        fail "refusing to overwrite existing output: $path"
    fi
done

mv "$OUTPUT_FILE" "$FINAL_OUTPUT_FILE"
OUTPUT_FILE=$FINAL_OUTPUT_FILE

python3 - "$OUTPUT_FILE" "$OUTPUT_SUMMARY" "$OUTPUT_PROFILE" "$WORKFLOW_NAME" "$RUN_ID" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
profile_path = Path(sys.argv[3])
workflow = sys.argv[4]
run_id = sys.argv[5]

with csv_path.open(encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
    columns = reader.fieldnames or []

if not rows:
    raise SystemExit("profile CSV contains no data rows")


def find_column(fragment: str) -> str:
    matches = [column for column in columns if fragment in column]
    if len(matches) != 1:
        raise SystemExit(f"profile CSV has no unique {fragment!r} column")
    return matches[0]


def numeric_values(fragment: str) -> list[float]:
    column = find_column(fragment)
    values: list[float] = []
    for row in rows:
        match = re.search(r"-?\d+(?:\.\d+)?", row.get(column) or "")
        if match:
            values.append(float(match.group(0)))
    return values


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def average(values: list[float], unit: str) -> str:
    if not values:
        return "N/A"
    return f"{sum(values) / len(values):.2f} {unit}"


def peak(values: list[float], unit: str) -> str:
    if not values:
        return "N/A"
    return f"{format_number(max(values))} {unit}"


gpu_utilization = numeric_values("utilization.gpu")
memory_utilization = numeric_values("utilization.memory")
memory_used = numeric_values("memory.used")
memory_total = numeric_values("memory.total")
if not memory_used:
    raise SystemExit("profile CSV contains no numeric memory.used samples")

profile = {
    "profiles": [
        {
            "workflow": workflow,
            "gpus.used": 1,
            "average": {
                "utilization.gpu": average(gpu_utilization, "%"),
                "utilization.memory": average(memory_utilization, "%"),
            },
            "peak": {
                "utilization.gpu": peak(gpu_utilization, "%"),
                "utilization.memory": peak(memory_utilization, "%"),
                "memory.used": peak(memory_used, "MiB"),
                "memory.total": peak(memory_total, "MiB"),
            },
        }
    ]
}

profile_json = json.dumps(profile, indent=2)
summary = "\n".join(
    [
        "GPU PROFILING SUMMARY",
        f"Workflow: {workflow}",
        f"Run ID: {run_id}",
        f"Samples: {len(rows)}",
        "",
        "Average utilization:",
        f"utilization.gpu: {profile['profiles'][0]['average']['utilization.gpu']}",
        f"utilization.memory: {profile['profiles'][0]['average']['utilization.memory']}",
        "",
        "Peak values:",
        f"utilization.gpu: {profile['profiles'][0]['peak']['utilization.gpu']}",
        f"utilization.memory: {profile['profiles'][0]['peak']['utilization.memory']}",
        f"memory.used: {profile['profiles'][0]['peak']['memory.used']}",
        f"memory.total: {profile['profiles'][0]['peak']['memory.total']}",
        "",
        "Scheduler profile:",
        profile_json,
        "",
    ]
)

summary_path.write_text(summary, encoding="utf-8")
profile_path.write_text(profile_json + "\n", encoding="utf-8")
print(summary, end="")
PY

echo "Raw samples: $OUTPUT_FILE"
echo "Summary: $OUTPUT_SUMMARY"
echo "Scheduler profile: $OUTPUT_PROFILE"
