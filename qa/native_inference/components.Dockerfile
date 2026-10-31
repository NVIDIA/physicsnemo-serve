# syntax=docker/dockerfile:1
# Build on the Linux bridge. Tests execute only after this image is submitted
# as a Lepton job. The immutable base must provide Python/Torch.
ARG COMPONENT_QA_BASE_IMAGE
FROM ${COMPONENT_QA_BASE_IMAGE} AS tooling
ARG RUST_TOOLCHAIN=1.94.1
ARG RUSTUP_VERSION=1.28.2
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git libssl-dev pkg-config redis-server \
      nlohmann-json3-dev python3-vtk9 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN if [ ! -x /usr/local/cargo/bin/cargo ]; then \
      curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
        -o /tmp/rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
      && echo '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c  /tmp/rustup-init' | sha256sum -c - \
      && chmod +x /tmp/rustup-init \
      && /tmp/rustup-init -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
      && rm /tmp/rustup-init; \
    fi
# Exported from this checkout's uv.lock with the qa extra; no dependency
# installation or source fetch is needed by the Lepton job.
COPY requirements.txt /tmp/native-component-requirements.txt
RUN python3 -m pip install --require-hashes -r /tmp/native-component-requirements.txt \
    && python3 -m pip install netCDF4==1.7.2 onnx==1.18.0 \
    && rustup component add clippy rustfmt
COPY cargo-source/ /tmp/native-components-cargo/
RUN cargo fetch --locked --manifest-path /tmp/native-components-cargo/Cargo.toml \
    && rm -rf /tmp/native-components-cargo

FROM tooling AS components
ARG SOURCE_SHA
COPY source.bundle /tmp/native-component-source.bundle
RUN rm -rf /workspace \
    && git clone /tmp/native-component-source.bundle /workspace \
    && test "$(git -C /workspace rev-parse HEAD)" = "${SOURCE_SHA}" \
    && git -C /workspace remote remove origin \
    && rm /tmp/native-component-source.bundle
WORKDIR /workspace
RUN cp -a native-inference/deploy/. /opt/native-release/
ENV SOURCE_SHA=${SOURCE_SHA} \
    NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH="/workspace:/workspace/scripts:/workspace/python:/workspace/python/e2s_tools:/usr/lib/python3/dist-packages" \
    CARGO_TARGET_DIR=/tmp/native-components-target
LABEL org.opencontainers.image.revision="${SOURCE_SHA}"
ENTRYPOINT ["python3", "/workspace/qa/native_inference/components.py"]
