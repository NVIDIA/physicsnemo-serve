#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUNTIME_DIR="${PHYSICSNEMO_SERVE_RUNTIME_DIR:-${HOME}/.local/share/physicsnemo-serve/runtimes/earth2studio}"
SERVE_BINARY="${PHYSICSNEMO_SERVE_BINARY:-${HOME}/physicsnemo-serve}"

if [[ ! -x "${SERVE_BINARY}" ]]; then
    echo "physicsnemo-serve binary is not executable: ${SERVE_BINARY}" >&2
    exit 1
fi
if [[ ! -x "${RUNTIME_DIR}/bin/python" ]]; then
    echo "external runtime is not initialized: ${RUNTIME_DIR}" >&2
    exit 1
fi

export PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND="${PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND:-python}"
exec "${SERVE_BINARY}" infer --runtime-dir "${RUNTIME_DIR}" "$@"
