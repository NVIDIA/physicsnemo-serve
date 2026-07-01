#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible python-feature gate for scicomp-rq.
# Chosen policy (PR-146): Option B
#   1) cargo feature compile gate
#   2) maturin develop
#   3) pytest integration runtime gate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_RUST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRATE_ROOT="${INFERENCE_RUST_ROOT}/crates/scicomp-rq"
DEFAULT_PYTHON_BIN="${CRATE_ROOT}/.venv/bin/python"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
elif [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
    PYTHON="${DEFAULT_PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "[ERROR] Python interpreter not found. Set PYTHON_BIN or create ${DEFAULT_PYTHON_BIN}." >&2
    exit 1
fi

REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"

require_python_module() {
    local module_name="$1"
    if ! "${PYTHON}" -m "${module_name}" --version >/dev/null 2>&1; then
        echo "[ERROR] Missing Python module '${module_name}' for ${PYTHON}." >&2
        echo "[ERROR] Install with: ${PYTHON} -m pip install ${module_name}" >&2
        exit 1
    fi
}

echo "[INFO] Using python interpreter: ${PYTHON}"
echo "[INFO] Using REDIS_URL: ${REDIS_URL}"

require_python_module maturin
require_python_module pytest

echo "[INFO] Gate 1/3: cargo check -p scicomp-rq --features python"
(
    cd "${INFERENCE_RUST_ROOT}"
    cargo check -p scicomp-rq --features python
)

echo "[INFO] Gate 2/3: maturin develop --features python-extension"
(
    cd "${CRATE_ROOT}"
    "${PYTHON}" -m maturin develop --manifest-path Cargo.toml --features python-extension
)

echo "[INFO] Gate 3/3: pytest integration gate"
(
    cd "${CRATE_ROOT}"
    REDIS_URL="${REDIS_URL}" \
        "${PYTHON}" -m pytest tests/test_scicomp_rq.py -q --run-integration
)

echo "[INFO] scicomp-rq python-feature gate PASSED"
