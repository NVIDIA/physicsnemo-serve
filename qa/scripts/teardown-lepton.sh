#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tear down a Lepton deployment by name.
#
# Required env vars (or flags):
#   LEPTON_WORKSPACE_ID    — workspace id (from deploy/config.yaml or env)
#   LEPTON_WORKSPACE_TOKEN — workspace auth token
#   LEPTON_ENDPOINT_NAME   — name of the deployment to remove
#
# Optional:
#   LEPTON_WORKSPACE_URL   — passed to `lep login -u` when set

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/deploy/config.sh"

LEPTON_WORKSPACE_ID="${LEPTON_WORKSPACE_ID:-$(_cfg lepton_workspace_id)}"
LEPTON_WORKSPACE_TOKEN="${LEPTON_WORKSPACE_TOKEN:-}"
LEPTON_WORKSPACE_URL="${LEPTON_WORKSPACE_URL:-}"
LEPTON_ENDPOINT_NAME="${LEPTON_ENDPOINT_NAME:-}"

usage() {
    cat <<'EOF'
Usage: teardown-lepton.sh [options]

Options:
  --workspace-id ID       LEPTON_WORKSPACE_ID
  --workspace-token TOKEN LEPTON_WORKSPACE_TOKEN
  --workspace-url URL     LEPTON_WORKSPACE_URL (optional)
  --endpoint-name NAME    LEPTON_ENDPOINT_NAME
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace-id) LEPTON_WORKSPACE_ID="$2"; shift 2 ;;
        --workspace-token) LEPTON_WORKSPACE_TOKEN="$2"; shift 2 ;;
        --workspace-url) LEPTON_WORKSPACE_URL="$2"; shift 2 ;;
        --endpoint-name) LEPTON_ENDPOINT_NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$LEPTON_WORKSPACE_ID" ]]; then
    echo "Error: LEPTON_WORKSPACE_ID is required" >&2; exit 2
fi
if [[ -z "$LEPTON_ENDPOINT_NAME" ]]; then
    echo "Error: LEPTON_ENDPOINT_NAME is required" >&2; exit 2
fi

if ! command -v lep >/dev/null 2>&1; then
    echo "Error: 'lep' CLI not found. Install with: pip install -U leptonai" >&2
    exit 1
fi

echo "==> Logging into Lepton workspace"
if [[ -n "$LEPTON_WORKSPACE_TOKEN" ]]; then
    LEPTON_CREDENTIALS="${LEPTON_WORKSPACE_ID}:${LEPTON_WORKSPACE_TOKEN}"
    if [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
        lep login -c "$LEPTON_CREDENTIALS" -u "$LEPTON_WORKSPACE_URL"
    else
        lep login -c "$LEPTON_CREDENTIALS"
    fi
elif [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
    echo "Error: --workspace-url requires --workspace-token so the script can log into that workspace" >&2
    exit 2
else
    echo "+ using existing lep login session"
fi

echo "==> Removing deployment: $LEPTON_ENDPOINT_NAME"
lep deployment remove -n "$LEPTON_ENDPOINT_NAME" || {
    echo "Warning: deployment remove failed (may already be gone)" >&2
}

echo "==> Teardown complete"
