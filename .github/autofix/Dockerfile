# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ci-autofix-env: sandbox for CI autofix triage reruns and the agent's commands.
# Mirrors the toolchain of the CPU jobs in .github/workflows/ci.yml. Keep the
# Rust toolchain and the graph-test pins in sync with that file.
#
# The container runs as the runner's uid with the checkout bind-mounted at
# /workspace; it never receives credentials.

FROM ubuntu:24.04

ARG RUST_TOOLCHAIN=1.94.1
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git jq libssl-dev nlohmann-json3-dev \
    pkg-config python3 python3-venv python3-yaml redis-server ripgrep \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal \
      --default-toolchain "${RUST_TOOLCHAIN}" --component rustfmt,clippy \
 && chmod -R a+rwX "${CARGO_HOME}" "${RUSTUP_HOME}"

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# Matches "Prepare isolated exact graph test environment" in ci.yml.
RUN python3 -m venv /opt/graph-venv \
 && /opt/graph-venv/bin/python -m pip install --no-cache-dir numpy==2.1.0 onnx==1.18.0 pytest==8.3.5

WORKDIR /workspace
