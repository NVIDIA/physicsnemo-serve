#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_DIR="${1:-${HOME}/.local/share/physicsnemo-serve/runtimes/earth2studio}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
REQUIREMENTS_LOCK="${SCRIPT_DIR}/requirements-e2s.lock"

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
    echo "requirements-e2s.lock is generated for the cu130 backend" >&2
    exit 1
fi

RUNTIME_PARENT="$(dirname -- "${RUNTIME_DIR}")"
RUNTIME_NAME="$(basename -- "${RUNTIME_DIR}")"
mkdir -p "${RUNTIME_PARENT}"
STAGING_ROOT="$(mktemp -d "${RUNTIME_PARENT}/.${RUNTIME_NAME}.tmp.XXXXXX")"
STAGING_RUNTIME="${STAGING_ROOT}/runtime"
PUBLISHED_RUNTIME=""
cleanup() {
    rm -rf -- "${STAGING_ROOT}"
    if [[ -n "${PUBLISHED_RUNTIME}" ]]; then
        rm -rf -- "${PUBLISHED_RUNTIME}"
    fi
}
trap cleanup EXIT

uv venv --python "${PYTHON_VERSION}" --relocatable "${STAGING_RUNTIME}"
uv pip install \
    --python "${STAGING_RUNTIME}/bin/python" \
    --torch-backend "${TORCH_BACKEND}" \
    --require-hashes \
    --requirements "${REQUIREMENTS_LOCK}"

mkdir -p "${STAGING_RUNTIME}/scripts" "${STAGING_RUNTIME}/python"
install -m 0644 \
    "${REPO_ROOT}/scripts/plugin_direct_runner.py" \
    "${REPO_ROOT}/scripts/plugin_runtime.py" \
    "${REPO_ROOT}/scripts/plugin_sdk.py" \
    "${STAGING_RUNTIME}/scripts/"
cp -R "${REPO_ROOT}/python/." "${STAGING_RUNTIME}/python/"

if [[ -e "${RUNTIME_DIR}" ]]; then
    echo "runtime directory was created concurrently: ${RUNTIME_DIR}" >&2
    exit 1
fi
mv --no-target-directory -- "${STAGING_RUNTIME}" "${RUNTIME_DIR}"
PUBLISHED_RUNTIME="${RUNTIME_DIR}"
"${RUNTIME_DIR}/bin/python" -c "import earth2studio, jsonschema, torch, yaml"
"${RUNTIME_DIR}/bin/dask" --version >/dev/null
PUBLISHED_RUNTIME=""
trap - EXIT
rm -rf -- "${STAGING_ROOT}"

echo "Earth2Studio runtime created at ${RUNTIME_DIR}"
echo "PyTorch backend: ${TORCH_BACKEND}"
