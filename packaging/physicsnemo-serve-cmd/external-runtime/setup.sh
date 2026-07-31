#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_DIR="${1:-${HOME}/.local/share/physicsnemo-serve/runtimes/earth2studio}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
REQUIREMENTS_LOCK="${SCRIPT_DIR}/requirements-earth2studio.lock"

if [[ -e "${RUNTIME_DIR}" ]]; then
    echo "runtime directory already exists: ${RUNTIME_DIR}" >&2
    echo "choose a new path or remove the existing runtime explicitly" >&2
    exit 1
fi

command -v uv >/dev/null 2>&1 || {
    echo "uv is required to create the external runtime" >&2
    exit 1
}
if [[ "${TORCH_BACKEND}" != "cu130" ]]; then
    echo "requirements-earth2studio.lock is generated for the cu130 backend" >&2
    exit 1
fi

uv venv --python "${PYTHON_VERSION}" "${RUNTIME_DIR}"
uv pip install \
    --python "${RUNTIME_DIR}/bin/python" \
    --torch-backend "${TORCH_BACKEND}" \
    --require-hashes \
    --requirements "${REQUIREMENTS_LOCK}"

mkdir -p "${RUNTIME_DIR}/scripts" "${RUNTIME_DIR}/python"
install -m 0644 \
    "${REPO_ROOT}/scripts/plugin_direct_runner.py" \
    "${REPO_ROOT}/scripts/plugin_runtime.py" \
    "${REPO_ROOT}/scripts/plugin_sdk.py" \
    "${RUNTIME_DIR}/scripts/"
cp -R "${REPO_ROOT}/python/." "${RUNTIME_DIR}/python/"

"${RUNTIME_DIR}/bin/python" -c "import earth2studio, jsonschema, torch, yaml"

echo "Earth2Studio runtime created at ${RUNTIME_DIR}"
echo "PyTorch backend: ${TORCH_BACKEND}"
