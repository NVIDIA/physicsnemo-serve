#!/bin/bash
set -euo pipefail
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# Script to test the Earth2Studio Inference Server REST APIs
# Usage: ./test_api.sh [HOST] [PORT]

HOST=${1:-localhost}
PORT=${2:-8080}
BASE_URL="http://$HOST:$PORT"

echo "============================================"
echo "Testing Inference Server APIs at $BASE_URL"
echo "============================================"

# 1. Health Check
echo -e "\n[1/7] Health Check..."
HEALTH=$(curl -s "$BASE_URL/healthz")
if [ "$HEALTH" = "ok" ]; then
    echo "[OK] Server is healthy"
else
    echo "[FAIL] Health check failed: $HEALTH"
    exit 1
fi

# 2. OpenAPI Spec
echo -e "\n[2/7] OpenAPI Specification..."
OPENAPI=$(curl -s "$BASE_URL/openapi.json")
if echo "$OPENAPI" | jq -e '.openapi' > /dev/null 2>&1; then
    VERSION=$(echo "$OPENAPI" | jq -r '.info.version')
    echo "[OK] OpenAPI spec available (v$VERSION)"
else
    echo "[FAIL] Failed to get OpenAPI spec"
fi

# 3. List Workflows
echo -e "\n[3/7] Listing Workflows..."
WORKFLOWS=$(curl -s "$BASE_URL/v1/infer/workflows")
COUNT=$(echo "$WORKFLOWS" | jq -r '.count // 0')
echo "Found $COUNT workflow(s)"
echo "$WORKFLOWS" | jq -r '.workflows[]?.name // empty' 2>/dev/null | head -5

if [ "$COUNT" -eq 0 ]; then
    echo "[WARN] No workflows discovered. Ensure Python workers have registered workflows."
    exit 1
else
    # Prefer deterministic_fcn_workflow or deterministic_earth2_workflow for testing
    if echo "$WORKFLOWS" | jq -r '.workflows[].name' | grep -q "deterministic_fcn_workflow"; then
        WORKFLOW="deterministic_fcn_workflow"
    elif echo "$WORKFLOWS" | jq -r '.workflows[].name' | grep -q "deterministic_earth2_workflow"; then
        WORKFLOW="deterministic_earth2_workflow"
    elif echo "$WORKFLOWS" | jq -r '.workflows[].name' | grep -q "example_user_workflow"; then
        WORKFLOW="example_user_workflow"
    else
        # Fallback to first available
        WORKFLOW=$(echo "$WORKFLOWS" | jq -r '.workflows[0].name')
    fi
    echo "Using workflow: $WORKFLOW"
fi

# 4. Get Workflow Schema
echo -e "\n[4/7] Getting Schema for '$WORKFLOW'..."
SCHEMA=$(curl -s "$BASE_URL/v1/infer/$WORKFLOW/schema")
if echo "$SCHEMA" | jq -e '.error' > /dev/null 2>&1; then
    echo "[FAIL] Schema error: $(echo "$SCHEMA" | jq -r '.error')"
    echo "  Hint: $(echo "$SCHEMA" | jq -r '.hint // empty')"
else
    echo "[OK] Schema retrieved"
    echo "$SCHEMA" | jq -r '.properties | keys | .[]' 2>/dev/null | sed 's/^/  - /'
fi

# 5. Run Workflow
echo -e "\n[5/7] Triggering Workflow Run..."

# Build payload based on workflow type (use ISO8601 datetime with Z suffix)
if [ "$WORKFLOW" = "deterministic_fcn_workflow" ]; then
    PAYLOAD='{"parameters": {"forecast_times": ["2024-01-01T00:00:00Z"], "nsteps": 2, "data_source": "gfs", "create_plots": false}}'
elif [ "$WORKFLOW" = "deterministic_earth2_workflow" ]; then
    PAYLOAD='{"parameters": {"start_time": ["2024-01-01T00:00:00Z"], "num_steps": 4}}'
elif [ "$WORKFLOW" = "stormcast_fcn3_workflow" ]; then
    PAYLOAD='{"parameters": {"start_time": "2024-01-01T00:00:00Z", "num_hours": 6}}'
elif [ "$WORKFLOW" = "example_user_workflow" ]; then
    PAYLOAD='{"parameters": {"task_name": "test_run", "num_iterations": 3, "delay_seconds": 0.1}}'
elif [ "$WORKFLOW" = "deterministic_workflow" ]; then
    PAYLOAD='{"parameters": {"forecast_times": ["2024-01-01T00:00:00Z"], "nsteps": 2, "model_type": "fcn", "create_plots": false}}'
elif [ "$WORKFLOW" = "diagnostic_workflow" ]; then
    PAYLOAD='{"parameters": {"forecast_times": ["2024-01-01T00:00:00Z"], "nsteps": 2, "diagnostic_model_type": "precipitation_afno", "create_plots": false}}'
else
    # Generic payload - may not work for all workflows
    PAYLOAD='{"parameters": {}}'
fi

echo "Payload: $PAYLOAD"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$PAYLOAD" "$BASE_URL/v1/infer/$WORKFLOW/run")
echo "$RESPONSE" | jq .

RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id // empty')

if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
    echo "[OK] Run ID: $RUN_ID"

    # 6. Check Status
    echo -e "\n[6/7] Checking Status..."
    for i in {1..3}; do
        STATUS=$(curl -s "$BASE_URL/v1/infer/$WORKFLOW/$RUN_ID/status")
        CURRENT=$(echo "$STATUS" | jq -r '.status // .stage // empty')
        if [ -n "$CURRENT" ]; then
            echo "  [$i] Status: $CURRENT"
            [ "$CURRENT" = "completed" ] && break
        else
            # Show timestamps if no status field
            ENQUEUED=$(echo "$STATUS" | jq -r '.api_enqueued_at // empty')
            if [ -n "$ENQUEUED" ]; then
                echo "  [$i] Queued (enqueued_at: $ENQUEUED) - awaiting worker"
            else
                echo "  [$i] No status data"
            fi
        fi
        sleep 1
    done

    # 7. Check Results
    echo -e "\n[7/7] Checking Results..."
    RESULT=$(curl -s "$BASE_URL/v1/infer/$WORKFLOW/$RUN_ID/results")
    if echo "$RESULT" | jq -e '.error' > /dev/null 2>&1; then
        echo "  Result: $(echo "$RESULT" | jq -r '.message // .error')"
    else
        echo "$RESULT" | jq .
        if echo "$RESULT" | jq -e 'has("request") and has("execution") and has("payload")' > /dev/null 2>&1; then
            echo "[OK] Structured result envelope returned"

            OUTPUT_PATH=$(echo "$RESULT" | jq -r '.execution.output_path // empty')
            if [ -n "$OUTPUT_PATH" ]; then
                echo "  Primary output path: $OUTPUT_PATH"
            fi

            OUTPUT_NAMES=$(echo "$RESULT" | jq -r '.execution.outputs[]?.name // empty' 2>/dev/null)
            if [ -n "$OUTPUT_NAMES" ]; then
                echo "$OUTPUT_NAMES" | sed 's/^/  - output: /'
            fi

            if echo "$RESULT" | jq -e '(.execution.output_path // "") != "" or ((.execution.outputs // []) | length > 0)' > /dev/null 2>&1; then
                HEADERS_FILE=$(mktemp)
                BODY_FILE=$(mktemp)
                HTTP_CODE=$(curl -s -D "$HEADERS_FILE" -o "$BODY_FILE" -w "%{http_code}" \
                    "$BASE_URL/v1/infer/$WORKFLOW/$RUN_ID/results?artifact=primary")
                if [ "$HTTP_CODE" = "200" ]; then
                    CONTENT_TYPE=$(awk 'BEGIN{IGNORECASE=1} /^content-type:/ {print $2}' "$HEADERS_FILE" | tr -d '\r')
                    BYTE_COUNT=$(wc -c < "$BODY_FILE" | tr -d ' ')
                    echo "[OK] Primary artifact download works (${BYTE_COUNT} bytes, ${CONTENT_TYPE:-unknown})"
                else
                    echo "[WARN] Primary artifact download returned HTTP $HTTP_CODE"
                fi
                rm -f "$HEADERS_FILE" "$BODY_FILE"
            fi
        else
            echo "[WARN] Result payload did not match the structured {request, execution, payload} contract"
        fi
    fi
else
    echo "[FAIL] Failed to get valid Run ID"
    echo "$RESPONSE" | jq -r '.error // empty'
    echo -e "\n[6/7] Skipped - no run ID"
    echo "[7/7] Skipped - no run ID"
fi

echo -e "\n============================================"
echo "Test Complete"
echo "============================================"
