# syntax=docker/dockerfile:1
# Controller-side builds use the ordinary runc runtime and require no GPU.
ARG NATIVE_BUILDER_IMAGE
ARG NATIVE_SERVER_IMAGE
FROM ${NATIVE_BUILDER_IMAGE} AS native-build
ARG RUST_TOOLCHAIN=1.94.1
ARG RUSTUP_VERSION=1.28.2
ARG CUSTOM_WORKFLOW_SOURCE_DIR=
ARG CUSTOM_WORKFLOW_TARGET=
ARG ENABLE_AOTI=ON
ARG ENABLE_TENSORRT=OFF
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config cmake libssl-dev curl ca-certificates time \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
       -o /tmp/rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
    && echo '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c  /tmp/rustup-init' | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rm /tmp/rustup-init
WORKDIR /src
COPY --from=release-source /native-inference/workflows/transolver/ /src/native-inference/workflows/transolver/
# This new comparator is header-only. Keep the immutable installed SDK binaries.
COPY --from=release-source /native-inference/cpp-runtime/include/physicsnemo/inference/v1/self_test.hpp /opt/physicsnemo-inference/include/physicsnemo/inference/v1/self_test.hpp
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"
RUN cmake -S native-inference/workflows/transolver -B /tmp/transolver-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/physicsnemo-native-runner -DBUILD_TESTING=ON \
      -DPNMIR_ENABLE_AOTI=${ENABLE_AOTI} -DPNMIR_ENABLE_TENSORRT=${ENABLE_TENSORRT} \
      -DCMAKE_CUDA_ARCHITECTURES=90 -DTORCH_CUDA_ARCH_LIST=9.0 \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/hpcx/ucx/lib \
      -DCMAKE_PREFIX_PATH="/opt/physicsnemo-inference;$(python3 -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent)')" \
      -DPNMIR_WARP_ROOT="$(python3 -c 'import pathlib,warp;print(pathlib.Path(warp.__file__).parent)')" \
    && cmake --build /tmp/transolver-build --parallel 4 \
    && cmake --install /tmp/transolver-build
COPY --from=release-source /native-inference/workflows/runtime/ /src/native-inference/workflows/runtime/
COPY --from=workflow-project / /workflow-custom/
RUN cmake -S native-inference/workflows/runtime -B /tmp/workflow-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/physicsnemo-workflow -DBUILD_TESTING=OFF \
      -DPNMS_ENABLE_MOCK_BACKEND=OFF -DPNMS_ENABLE_AOTI=${ENABLE_AOTI} \
      -DPNMS_ENABLE_TENSORRT=${ENABLE_TENSORRT} \
      -DPNMS_CUSTOM_WORKFLOW_SOURCE_DIR="${CUSTOM_WORKFLOW_SOURCE_DIR}" \
      -DPNMS_CUSTOM_WORKFLOW_TARGET="${CUSTOM_WORKFLOW_TARGET}" \
      -DCMAKE_PREFIX_PATH="/opt/physicsnemo-inference;$(python3 -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent)')" \
    && cmake --build /tmp/workflow-build --parallel 4 \
    && cmake --install /tmp/workflow-build
ENV NATIVE_RUNNER_INCLUDE=/opt/physicsnemo-native-runner/include \
    NATIVE_RUNNER_LIB_DIR=/opt/physicsnemo-native-runner/lib \
    NATIVE_WORKFLOW_INCLUDE=/opt/physicsnemo-workflow/include \
    NATIVE_WORKFLOW_LIB_DIR=/opt/physicsnemo-workflow/lib
ENV LD_LIBRARY_PATH="/opt/physicsnemo-workflow/lib:/opt/physicsnemo-native-runner/lib:/opt/physicsnemo-inference/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/warp/bin:${LD_LIBRARY_PATH}"
COPY --from=release-source /Cargo.toml /Cargo.lock /src/
COPY --from=release-source /.cargo/ /src/.cargo/
COPY --from=release-source /crates/ /src/crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    export CXXFLAGS="${CXXFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')" \
    && cargo build --release --locked -p native-worker --features cpp -p inference_server -p worker-runtime

FROM native-build AS compiler
ARG SOURCE_SHA
COPY --from=compile-input / /opt/native-release/input/
COPY --from=release-source /native-inference/model-builder/ /opt/native-release/source/native-inference/model-builder/
COPY --from=release-source /native-inference/deploy/ /opt/native-release/
COPY --from=release-source /qa/ /opt/native-release/source/qa/
COPY --from=release-source /plugins/ /opt/native-release/source/plugins/
COPY --from=release-source /native-inference/examples/ /opt/native-release/source/native-inference/examples/
ENV PYTHONPATH="/opt/native-release:/opt/native-release/source:/opt/native-release/source/native-inference/model-builder/src"
ENV PATH="/opt/physicsnemo-native-runner/bin:${PATH}"
ENV SOURCE_SHA=${SOURCE_SHA}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

FROM ${NATIVE_SERVER_IMAGE} AS release
ARG SOURCE_SHA
ARG WORKFLOW_ID
ARG RELEASE_ROOT=/opt/physicsnemo-release
ARG ARTIFACT_ROOT=/outputs/native-workflow
ARG MODEL_BUILD_SOURCE_SHA
LABEL org.opencontainers.image.revision="${SOURCE_SHA}"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends redis-server supervisor \
    && rm -rf /var/lib/apt/lists/*
COPY --from=native-build /src/target/release/inference_server /app/bin/inference-server
COPY --from=native-build /src/target/release/native-worker /src/target/release/worker-runtime /app/bin/
COPY --from=native-build /opt/physicsnemo-native-runner /opt/physicsnemo-native-runner
COPY --from=native-build /opt/physicsnemo-workflow /opt/physicsnemo-workflow
COPY --from=compiled-release / ${RELEASE_ROOT}/
COPY --from=compiled-release /plugin/ /app/plugins/
COPY --from=release-source /scripts/entrypoint.sh /scripts/health_stub.py /app/scripts/
COPY --from=release-source /crates/worker-runtime/config/profiles.json /app/config/profiles.json
COPY --from=release-source /native-inference/deploy/ /opt/native-release/
COPY --from=release-source /qa/ /opt/native-release/source/qa/
COPY --from=release-source /plugins/ /opt/native-release/source/plugins/
ENV PYTHONPATH="/opt/native-release:/opt/native-release/source"
ENV NATIVE_DEPLOYMENT_CONFIG=${RELEASE_ROOT}/deployment.json \
    WORKER_RUNTIME_CONFIG=${RELEASE_ROOT}/worker_runtime_config.json \
    QUEUE_CONFIG=${RELEASE_ROOT}/worker_runtime_config.json \
    PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG=${RELEASE_ROOT}/worker_runtime_config.json \
    PLUGIN_DIR=/app/plugins \
    PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=${WORKFLOW_ID} \
    USE_PREFETCH=false WORKERS=server,scheduler,results,execute WORKERS_PER_GPU=1 \
    REDIS_STREAM_PREFIX=physicsnemo: TOKIO_WORKER_THREADS=4 \
    ARTIFACT_DIR=${ARTIFACT_ROOT} DEFAULT_OUTPUT_DIR=${ARTIFACT_ROOT} \
    SOURCE_SHA=${SOURCE_SHA} MODEL_BUILD_SOURCE_SHA=${MODEL_BUILD_SOURCE_SHA}
ENV LD_LIBRARY_PATH="/opt/physicsnemo-workflow/lib:/opt/physicsnemo-native-runner/lib:/opt/physicsnemo-inference/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/warp/bin:${LD_LIBRARY_PATH}"
RUN chmod +x /app/scripts/entrypoint.sh
EXPOSE 8080 8001
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/app/scripts/entrypoint.sh"]
