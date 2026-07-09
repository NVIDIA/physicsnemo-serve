#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared deploy configuration reader.
#
# Source this file from any shell script in the repo:
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../deploy/config.sh"
#
# Then call:
#   _cfg <key>          # prints the value for <key> from deploy/config.yaml
#
# Example:
#   DOCKER_REPO="${DOCKER_REPO:-$(_cfg docker_registry)}"

_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_cfg() {
    local key="$1"
    local value
    value=$(grep "^${key}:" "$_DEPLOY_DIR/config.yaml" 2>/dev/null \
        | head -1 \
        | sed 's/^[^:]*:[[:space:]]*//' \
        | sed 's/"//g' \
        | sed "s/'//g" \
        | sed 's/[[:space:]]*#.*//' || true)
    printf '%s' "$value"
}
