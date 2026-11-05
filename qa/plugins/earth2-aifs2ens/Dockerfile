# syntax=docker/dockerfile:1.4
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e
FROM rust:1.94.1-slim AS rust-toolchain
FROM ${PYTORCH_BASE_IMAGE} AS qualified-runtime

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/opt/aifs2ens-venv/bin:/usr/local/cargo/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    CPATH=/usr/local/cuda/include \
    PYTHONPATH=/opt/physicsnemo-serve/scripts:/opt/physicsnemo-serve/python:/opt/physicsnemo-serve/plugins/earth2-aifs2ens

RUN find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -print0 | xargs -0 sed -i \
      -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
      -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
 && apt-get update -o Acquire::Retries=3 \
 && apt-get install -y --no-install-recommends redis-server redis-tools git pkg-config libssl-dev python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/aifs2ens-venv
COPY qa/plugins/earth2-aifs2ens/qa-requirements.txt /tmp/qa-requirements.txt
COPY qa/plugins/earth2-aifs2ens/h100-runtime-additions.lock /tmp/h100-runtime-additions.lock
RUN python -m pip install --no-cache-dir -r /tmp/qa-requirements.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/h100-runtime-additions.lock \
 && python -c 'import torch; from importlib.metadata import version; assert torch.__version__ == "2.10.0a0+a36e1d39eb.nv26.01.42222806"; assert version("flash-attn") == "2.7.4.post1+nv26.1.42222806"'

WORKDIR /opt/physicsnemo-serve
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY crates ./crates
RUN cargo build --locked -j 8 -p inference_server -p worker-runtime \
 && CARGO_BUILD_JOBS=8 python -m pip install --no-cache-dir --no-build-isolation ./crates/scicomp-rq ./crates/e2s_zarr_io \
 && python -m pip freeze > /opt/aifs2ens-runtime-freeze.txt
RUN python -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/aifs2ens-wheels ./crates/scicomp-rq ./crates/e2s_zarr_io \
 && python3 -m venv /opt/aifs2ens-qa-venv \
 && /opt/aifs2ens-qa-venv/bin/python -m pip install --no-cache-dir -r /tmp/qa-requirements.txt \
      numpy==2.1.0 xarray==2026.9.0 zarr==3.4.2 'dask[array]==2026.8.0' \
 && /opt/aifs2ens-qa-venv/bin/python -m pip install --no-deps /opt/aifs2ens-wheels/*.whl \
 && /opt/aifs2ens-qa-venv/bin/python -c 'import importlib.util as u; assert u.find_spec("torch") is None; assert u.find_spec("earth2studio") is None'
ENV AIFS2ENS_PORTABLE_PYTHON=/opt/aifs2ens-qa-venv/bin/python
COPY . .
ARG SOURCE_CONTENT_SHA256
ENV AIFS2ENS_SOURCE_CONTENT_SHA256=${SOURCE_CONTENT_SHA256}
LABEL io.physicsnemo.aifs2ens.source-content-sha256=${SOURCE_CONTENT_SHA256}
RUN python -c 'import os; from importlib.util import spec_from_file_location,module_from_spec; s=spec_from_file_location("qa", "qa/plugins/earth2-aifs2ens/run_qualification.py"); m=module_from_spec(s); s.loader.exec_module(m); m.verify_source(os.environ["AIFS2ENS_SOURCE_CONTENT_SHA256"])'
RUN /opt/aifs2ens-qa-venv/bin/python -m pytest -q -p pytest_asyncio.plugin \
      -k 'not spherical_gaussian_skip_ahead_matches' \
      tests/test_aifs2ens_*.py tests/test_earth2_plugins.py tests/test_plugin_dev.py tests/test_runtime_env_launcher.py \
 && python scripts/plugin_dev.py validate plugins/earth2-aifs2ens
CMD ["python", "qa/plugins/earth2-aifs2ens/run_qualification.py", "--help"]

FROM qualified-runtime AS release
EXPOSE 8001
CMD ["python", "qa/plugins/earth2-aifs2ens/serve.py"]
