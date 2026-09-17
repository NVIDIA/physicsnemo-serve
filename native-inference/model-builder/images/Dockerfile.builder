# Build from the repository root. This is a development builder recipe;
# publishing and selecting an immutable builder image is a separate release step.
# Linux x86_64 platform manifest verified from the cached 26.01-py3 image.
FROM nvcr.io/nvidia/pytorch@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e AS builder-environment

COPY native-inference/model-builder/images/geotransolver-requirements.lock \
     native-inference/model-builder/images/geotransolver-environment.json \
     native-inference/model-builder/images/verify_builder_environment.py /opt/physicsnemo/image/

# The explicit overlay avoids resolving PhysicsNeMo's stable Torch requirement
# over NGC's qualified prerelease. Check every active dependency and require
# protected native stack files to remain identical to the pinned base.
RUN python3 /opt/physicsnemo/image/verify_builder_environment.py \
        --contract /opt/physicsnemo/image/geotransolver-environment.json \
        --capture /opt/physicsnemo/image/base-environment.json \
    && python3 -m pip install --no-deps --require-hashes --no-build-isolation \
        -r /opt/physicsnemo/image/geotransolver-requirements.lock \
    && python3 /opt/physicsnemo/image/verify_builder_environment.py \
        --contract /opt/physicsnemo/image/geotransolver-environment.json \
        --verify /opt/physicsnemo/image/base-environment.json \
        --output /opt/physicsnemo/image/builder-environment.json \
    && python3 -c 'from physicsnemo.experimental.models.geotransolver import GeoTransolver; import warp'

FROM builder-environment AS builder

ARG PNMIR_ENABLE_TENSORRT=OFF
ARG CMAKE_BUILD_PARALLEL_LEVEL=4

COPY native-inference /opt/physicsnemo/native-inference

RUN cmake -S /opt/physicsnemo/native-inference/cpp-runtime -B /tmp/physicsnemo-inference-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/physicsnemo-inference \
        -DPNMIR_BUILD_TESTS=OFF \
        -DPNMIR_ENABLE_AOTI=ON \
        -DPNMIR_ENABLE_TENSORRT=${PNMIR_ENABLE_TENSORRT} \
    && cmake --build /tmp/physicsnemo-inference-build --parallel ${CMAKE_BUILD_PARALLEL_LEVEL} \
    && cmake --install /tmp/physicsnemo-inference-build \
    && python3 -m pip install --no-deps --no-build-isolation /opt/physicsnemo/native-inference/model-builder \
    && python3 -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")' \
        > /etc/ld.so.conf.d/physicsnemo-pytorch.conf \
    && ldconfig \
    && /opt/physicsnemo-inference/bin/physicsnemo-infer run /opt/physicsnemo/native-inference/cpp-runtime/tests/fixtures/identity --values 1,2,3 \
    && physicsnemo-model-builder list --json

ENV PATH="/opt/physicsnemo-inference/bin:${PATH}"
# PhysicsNeMo uses LOCAL_CACHE rather than XDG_CACHE_HOME. Set it after build
# steps so the runtime UID creates and owns the directory in each container.
ENV LOCAL_CACHE="/tmp/physicsnemo-cache"
WORKDIR /work

# Retain NGC driver/CUDA compatibility initialization before the installed CLI.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "physicsnemo-model-builder"]
CMD ["--help"]
