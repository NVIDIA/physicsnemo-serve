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
    && pnms-model-builder list --json

ENV PATH="/opt/physicsnemo-inference/bin:${PATH}"
# PhysicsNeMo uses LOCAL_CACHE rather than XDG_CACHE_HOME. Set it after build
# steps so the runtime UID creates and owns the directory in each container.
ENV LOCAL_CACHE="/tmp/physicsnemo-cache"
WORKDIR /work

# Retain NGC driver/CUDA compatibility initialization before the installed CLI.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "pnms-model-builder"]
CMD ["--help"]

# Select explicitly with --target native-qa. Keep the normal builder default below.
FROM builder AS native-qa

ARG CMAKE_CUDA_ARCHITECTURES

# Both QA profiles require both installed SDK backends. This also fails image
# construction when the pinned base does not supply the TensorRT development SDK.
RUN test "${PNMIR_ENABLE_TENSORRT}" = ON \
    && test -n "${CMAKE_CUDA_ARCHITECTURES}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends libvtk9-dev python3-vtk9 \
    && rm -rf /var/lib/apt/lists/*

ARG SOURCE_SHA
RUN python3 -c 'import re, sys; assert re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]), "SOURCE_SHA must be a full Git SHA"' "${SOURCE_SHA}"

COPY qa/native_inference /opt/physicsnemo-qa

# VTK installs distro UCX as a dependency. Keep NGC's UCC paired with its own
# UCX, including when the GPU runtime prepends distro library search paths.
RUN test -r /opt/hpcx/ucx/lib/libucs.so.0 \
    && test -r /opt/hpcx/ucc/lib/libucc.so.1 \
    && printf '%s\n' \
        'export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' \
        > /opt/nvidia/entrypoint.d/99z-physicsnemo-qa-hpcx.sh

# Import only the distro's matched VTK bindings ahead of any pre-existing wheel.
# Do not put the entire distro site-packages ahead of the pinned ML environment.
RUN mkdir -p /opt/physicsnemo-qa/python \
    && ln -s /usr/lib/python3/dist-packages/vtkmodules /opt/physicsnemo-qa/python/vtkmodules \
    && python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "physicsnemo_qa_vtk.pth").write_text("import sys; sys.path.insert(0, \"/opt/physicsnemo-qa/python\")\n")' \
    && python3 /opt/physicsnemo-qa/image_metadata.py \
        --source-sha "${SOURCE_SHA}" --cuda-architectures "${CMAKE_CUDA_ARCHITECTURES}" \
        --output /opt/physicsnemo-qa/source.json \
    && python3 /opt/physicsnemo/image/verify_builder_environment.py \
        --contract /opt/physicsnemo/image/geotransolver-environment.json \
        --verify /opt/physicsnemo/image/base-environment.json \
        --output /opt/physicsnemo/image/qa-environment.json \
    && LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
        /opt/nvidia/nvidia_entrypoint.sh python3 -c 'import torch; import vtkmodules.all as vtk; from pathlib import Path; mappings = Path("/proc/self/maps").read_text(); assert "/opt/hpcx/ucx/lib/libucs.so." in mappings; assert "/opt/hpcx/ucc/lib/libucc.so." in mappings; print("QA entrypoint imports passed:", torch.__version__, vtk.vtkVersion.GetVTKVersion())'

# The real Transolver TensorRT profile needs the SDK's exact operators. Build
# against the pinned NGC source/CUTLASS headers and installed generated headers.
# This upgrade belongs only to native-qa; ordinary builder images stay unchanged.
# The linker searches dependency RPATHs before LD_LIBRARY_PATH; rpath-link keeps
# NGC UCC paired with NGC UCX at link time as well as through the entrypoint.
RUN . /opt/nvidia/entrypoint.d/99z-physicsnemo-qa-hpcx.sh \
    && PNMIR_QA_TORCH_ARCHS="$(python3 -c 'import re, sys; value = sys.argv[1]; assert re.fullmatch(r"[0-9]{2,3}(;[0-9]{2,3})*", value), "CUDA architectures must be numeric, for example 90 or 80;90"; print(";".join(f"{int(arch) // 10}.{int(arch) % 10}" for arch in value.split(";")))' "${CMAKE_CUDA_ARCHITECTURES}")" \
    && cmake -S /opt/physicsnemo/native-inference/cpp-runtime -B /tmp/physicsnemo-inference-build \
        -DPNMIR_ENABLE_TENSORRT_EXACT=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/opt/hpcx/ucx/lib" \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DTORCH_CUDA_ARCH_LIST="${PNMIR_QA_TORCH_ARCHS}" \
        -DPNMIR_PYTORCH_SOURCE_ROOT=/opt/pytorch/pytorch \
        -DPNMIR_CUTLASS_INCLUDE_DIR=/opt/pytorch/pytorch/third_party/cutlass/include \
        -DPNMIR_TORCH_INCLUDE_DIR="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include")')" \
    && cmake --build /tmp/physicsnemo-inference-build --parallel ${CMAKE_BUILD_PARALLEL_LEVEL} \
    && cmake --install /tmp/physicsnemo-inference-build \
    && ldconfig \
    && /opt/physicsnemo-inference/bin/physicsnemo-infer run /opt/physicsnemo/native-inference/cpp-runtime/tests/fixtures/identity --values 1,2,3 \
    && python3 -c 'from pathlib import Path; import tensorrt as trt; from model_builder.export.tensorrt_profiles import plugin_names, load_exact_plugins; profile = "layout-order-exact-v2"; names = plugin_names(profile); libraries = {name: Path("/opt/physicsnemo-inference/lib") / f"libpnmir_tensorrt_{name}_plugin.so" for name in names}; handles, records = load_exact_plugins(trt, libraries, profile); assert len(records) == 9; print("Transolver exact TensorRT plugins verified:", ", ".join(records))'

RUN . /opt/nvidia/entrypoint.d/99z-physicsnemo-qa-hpcx.sh \
    && cmake -S /opt/physicsnemo-qa/cpp -B /tmp/physicsnemo-qa-consumer-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/opt/hpcx/ucx/lib" \
        -DCMAKE_PREFIX_PATH="/opt/physicsnemo-inference;$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" \
        -DCMAKE_INSTALL_PREFIX=/opt/physicsnemo-inference \
    && cmake --build /tmp/physicsnemo-qa-consumer-build --parallel ${CMAKE_BUILD_PARALLEL_LEVEL} \
    && cmake --install /tmp/physicsnemo-qa-consumer-build \
    && python3 -c 'import subprocess; result = subprocess.run(["/opt/physicsnemo-inference/bin/physicsnemo-qa-consumer"], capture_output=True, text=True, timeout=30); print(result.stderr, end=""); assert result.returncode == 1 and result.stderr.startswith("usage: physicsnemo-qa-consumer "), f"installed QA consumer did not start correctly (exit {result.returncode})"' \
    && PNMIR_QA_TORCH_ARCHS="$(python3 -c 'import re, sys; value = sys.argv[1]; assert re.fullmatch(r"[0-9]{2,3}(;[0-9]{2,3})*", value), "CUDA architectures must be numeric, for example 90 or 80;90"; print(";".join(f"{int(arch) // 10}.{int(arch) % 10}" for arch in value.split(";")))' "${CMAKE_CUDA_ARCHITECTURES}")" \
    && cmake -S /opt/physicsnemo/native-inference/workflows/transolver -B /opt/physicsnemo-qa/workflow-build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/opt/hpcx/ucx/lib" \
        -DPNMIR_ENABLE_AOTI=ON -DPNMIR_ENABLE_TENSORRT=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DTORCH_CUDA_ARCH_LIST="${PNMIR_QA_TORCH_ARCHS}" \
        -DCMAKE_PREFIX_PATH="/opt/physicsnemo-inference;$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" \
        -DPNMIR_WARP_ROOT="$(python3 -c 'import pathlib, warp; print(pathlib.Path(warp.__file__).parent)')" \
    && cmake --build /opt/physicsnemo-qa/workflow-build --parallel ${CMAKE_BUILD_PARALLEL_LEVEL} \
    && ln -s /opt/physicsnemo-qa/workflow-build/physicsnemo-transolver /opt/physicsnemo-inference/bin/physicsnemo-transolver \
    && /opt/physicsnemo-inference/bin/physicsnemo-transolver --help

LABEL org.opencontainers.image.revision="${SOURCE_SHA}"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["python3", "/opt/physicsnemo-qa/run_job.py"]

# Untargeted builds still produce the original Model Builder image.
FROM builder AS default-builder
