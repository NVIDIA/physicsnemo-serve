# PhysicsNeMo C++ Inference Runtime

The runtime executes model package directories through the
`physicsnemo::inference::v1` C++ interface and optional AOTInductor, ONNX Runtime
and TensorRT backends. The core build includes a mock backend for API tests.
Model preparation and export live in the sibling [Model Builder](../model-builder);
customer workflows are described in the [inference guide](../README.md).

## Build and install

From the repository root:

```bash
cmake -S native-inference/cpp-runtime -B out/inference/runtime-build \
  -DPNMIR_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/out/inference/runtime"
cmake --build out/inference/runtime-build --parallel
cmake --install out/inference/runtime-build
```

The `native-inference` CMake entry point also builds this runtime. The core source
build needs no Python, Torch or CUDA. The installed executable is
`physicsnemo-infer`; the core library is `physicsnemo_inference`.

An external CMake application can use the installed prefix after it is moved:

```cmake
find_package(PhysicsNeMoInference 0.1 CONFIG REQUIRED COMPONENTS core)
add_executable(my_inference main.cpp)
target_link_libraries(my_inference PRIVATE PhysicsNeMoInference::runtime)
```

Public headers and C++ types use the same product name:

```cpp
#include <physicsnemo/inference/v1/api.hpp>

physicsnemo::inference::v1::Engine engine;
```

Configure an application with
`-DCMAKE_PREFIX_PATH=/path/to/installed/runtime`. The imported target supplies
the C++20 requirement. Source and build directories are not needed after
installation. The interface uses STL types and exceptions; applications must
use a compatible compiler and standard-library ABI.

Existing build flags remain supported: `PNMIR_BUILD_TESTS`,
`PNMIR_ENABLE_AOTI`, `PNMIR_ENABLE_ONNXRUNTIME`,
`PNMIR_ENABLE_ONNXRUNTIME_CUDA` and `PNMIR_ENABLE_TENSORRT`.
Installed consumers request `COMPONENTS aoti`, `onnxruntime` or `tensorrt`,
then link `PhysicsNeMoInference::aoti`, `PhysicsNeMoInference::onnxruntime`
or `PhysicsNeMoInference::tensorrt`. Each backend imports the core target.
AOTI development consumers supply a compatible native Torch CMake prefix;
ONNX Runtime and TensorRT consumers supply `PNMIR_ONNXRUNTIME_ROOT` or
`PNMIR_TENSORRT_ROOT` with their native dependencies.

For a complete external application, see the
[Transolver E2E CLI example](../workflows/transolver/README.md). It links the
installed SDK to C++ mesh preprocessing and physical-unit postprocessing.

Static and shared core installs are tested for relocation. Optional GPU
backend libraries are not bundled into a complete runtime distribution.
AOTI source builds discover matching Torch through Python. Installed GPU
executables require compatible native backend libraries on the loader path;
installed CMake exports do not preserve the builder's absolute library paths.
Native inference does not import the Python model.

### Exact TensorRT operators for Transolver and GeoTransolver

The opt-in `PNMIR_ENABLE_TENSORRT_EXACT=ON` build adds eleven native TensorRT
plugins: Linear, GEMM, TokenSum, SliceBmm, LayerNorm, Softmax, Attention, GELU
and WeightedBlend, plus ScalarDiv and InverseDistanceBlend for DoMINO surface
aggregation.
They preserve the arithmetic and layout choices used by the pinned PyTorch
reference. Model Builder's `layout-order-exact` profile selects the original
eight; `geotransolver-exact` also requires WeightedBlend. Each build verifies
its native results; these profiles do not guarantee bitwise parity for arbitrary
models or different CUDA/cuBLAS/PyTorch versions.
The option defaults to `OFF` and requires `PNMIR_ENABLE_TENSORRT=ON`.

The attention plugin compiles a PyTorch/CUTLASS CUDA template. Supply the
matching source and generated header trees explicitly; they are build-time
dependencies and do not link the plugins to LibTorch or Python. For the H100
environment used by the source implementation:

```bash
cmake -S native-inference/cpp-runtime -B out/inference/runtime-exact \
  -DPNMIR_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DPNMIR_ENABLE_TENSORRT=ON -DPNMIR_ENABLE_TENSORRT_EXACT=ON \
  -DPNMIR_TENSORRT_ROOT=/usr \
  -DPNMIR_PYTORCH_SOURCE_ROOT=/opt/pytorch/pytorch \
  -DPNMIR_CUTLASS_INCLUDE_DIR=/opt/pytorch/pytorch/third_party/cutlass/include \
  -DPNMIR_TORCH_INCLUDE_DIR=/path/to/matching/torch/include \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_INSTALL_PREFIX="$PWD/out/inference/runtime"
cmake --build out/inference/runtime-exact --parallel
cmake --install out/inference/runtime-exact
```

Add `-DPNMIR_ENABLE_AOTI=ON` when configuring the SDK for the GeoTransolver
example's default two-backend build. This also requires the matching installed
Torch development libraries, as described above.

Installation includes all eleven `libpnmir_tensorrt_exact_*_plugin.so` files in
the SDK library directory. The TensorRT CMake target links its matching plugin
targets, and the CLI and device runner register their operator IDs automatically.
Applications using `Runtime` must call the public helper before creating a
session whose manifest declares these required operators:

```cpp
#include <physicsnemo/inference/backends/tensorrt.hpp>
#include <physicsnemo/inference/runtime.hpp>

physicsnemo::inference::Runtime runtime;
physicsnemo::inference::register_tensorrt_exact_operators(runtime);
runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
```

The helper is a no-op in a TensorRT build without the exact option, so such a
build continues rejecting packages that require these operators. TensorRT
creator registration occurs before engine deserialization. Plugin names, C
registration symbols and serialized operator IDs/ABI versions remain compatible
with the source implementation from `gpu_programming` commit
`87b78bf6cec3030cfbaea10f14e4c64cbc836407`; only public C++ include paths and
namespaces were adapted in the eight plugin implementations.

GeoTransolver's WeightedBlend plugin is ported from the same source commit and
preserves its `pnmir.tensorrt-exact-weighted-blend` operator ID, ABI `1`, plugin
name and registration symbol. It evaluates two FP32 scalar-weighted tensors
with separately rounded multiplies followed by a rounded add, preserving the
eager reference's arithmetic instead of fusing it into an FMA. The original
eight operators and Transolver profile remain compatible.

For a Model Builder project, copy the installed libraries into its declared
asset paths before building. The [GeoTransolver example commands](../examples/README.md#geotransolver)
copy all nine from the SDK's `lib/` directory to `assets/tensorrt/`. Model Builder
captures and hashes those copies; run the resulting engine with the same
compatible SDK and plugin libraries. The runtime registers native operators;
the automatic scalar-sigmoid preparation and byte-identical acceptance gate are
provided by the builder's `geotransolver-exact` profile.

### DoMINO AOTI boundary operators

`PNMIR_BUILD_DOMINO_EXACT_OPS=ON` adds `libpnmir_domino_exact_ops.so` to an
AOTI-enabled SDK. The option defaults to `OFF` and requires
`PNMIR_ENABLE_AOTI=ON`. It uses the same matching LibTorch installation as the
AOTI backend; it does not require a separate CUDA compilation step.

The library preserves the ten `pnmir_domino` Tensor-only schemas for scalar
arithmetic, reciprocal, indexed selection, subtraction, last-dimension norm,
nearest 3D upsampling and SDF features. Their C++ implementations call ATen
directly. The exported model's learned layers remain in AOTI. The declared
operator is `physicsnemo-cfd.domino-exact-boundary`, with ABI
`b1c60ddada2438469a1d24b4e53ae196425b73648f6d8ae45ecf64043755d7e6`.
The source is ported from `gpu_programming` with public C++ namespaces and
include paths adapted; the dispatcher schemas and serialized ABI are unchanged.

The CLI and device runner register this ABI automatically when built with the
option. Applications using the SDK call the public helper before creating a
session that requires the sidecar:

```cpp
#include <physicsnemo/inference/backends/aoti.hpp>
#include <physicsnemo/inference/runtime.hpp>

physicsnemo::inference::Runtime runtime;
physicsnemo::inference::register_aoti_exact_operators(runtime);
runtime.register_backend(physicsnemo::inference::create_aoti_backend());
```

The helper leaves the operator registry unchanged when the option is disabled,
so packages requiring that ABI are rejected. Installation exports
`PhysicsNeMoInference::domino_exact_ops` with the `aoti` component. The installed
AOTI target links the matching sidecar. CUDA AOTI loading disables both cuBLAS
and cuDNN TF32 to match the exporter's IEEE FP32 policy.

## Load a model package

New Model Builder outputs place each backend directly under
`model/backends/<backend>/`. Pass that directory to the runtime; it
contains `model.json` beside `model.pt2` for AOTI or `model.plan` for TensorRT.
The manifest records the AOTI CPU/CUDA target; filenames do not distinguish it.
For an affine package built for CUDA AOTI and a compatible installed runtime:

```bash
physicsnemo-infer run /data/build/model/backends/aoti \
  --backend aoti --device cuda --values 0,1,-1,4
```

Package loading uses `model.json`, not a directory suffix. Existing `.pnmir`
and intermediate `.pnm-model` directories load through the same manifest and
artifact contract, including older manifests with nested payload paths.
The runtime follows the manifest's relative artifact path; it does not require
an `artifacts/` directory. The runtime does not load `model-release.json` as a
workflow. Automated consumers should use its selected package path instead of
hardcoding a layout;
see [package compatibility and migration](../docs/packages.md).

## Runtime tests and fixtures

The `tests/fixtures` directory contains the mock identity package, the tiny
AOTI affine factory, and a frozen ONNX channel-flow package with reproduction
source. These assets remain inside this independently copyable C++ project.
They are regression fixtures, separate from customer model recipes and generated
build output. See the [fixture inventory](tests/fixtures/README.md).

```bash
cmake -S native-inference/cpp-runtime -B out/inference/runtime-tests \
  -DPNMIR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/inference/runtime-tests --parallel
ctest --test-dir out/inference/runtime-tests --output-on-failure
```

Core tests use Python as a driver and include static/shared installation tests
that copy this source tree, relocate the installed prefix, remove source/build
directories, and build an external consumer using the public names above.
Optional backend integration tests require the corresponding producer
frameworks and native dependencies. CTest supplies
`native-inference/model-builder/src` through `PYTHONPATH` so those tests use the single
exporter implementation. A standalone copy can instead use an installed
compatible Model Builder package. A skipped CUDA test is not GPU validation.

## Native output metadata

The CLI accepts `--output-metadata <json-file>` with its `run` options. It emits
versioned metadata after successful synchronous inference and output handling:

```json
{
  "schema_version": 1,
  "backend": "aoti",
  "execution_device": {"type": "cuda", "index": 0},
  "completed": true,
  "outputs": [
    {
      "name": "output",
      "dtype": "float32",
      "shape": [3],
      "device": {"type": "cpu", "index": 0},
      "byte_size": 12
    }
  ]
}
```

Output name, dtype, shape, device and byte count come from returned tensor
views; dynamic dimensions report actual dimensions. `execution_device`
identifies the requested device accepted by the selected backend. The CLI
returns CPU output storage even for CUDA inference, after transfer completes.

A harness must use a fresh directory, require zero exit status, and compare
metadata and raw sizes with its independent reference. A previous run's report
is not evidence after a failed invocation. Metadata is an execution record,
not a package digest or scientific qualification report.

See [MIGRATION.md](MIGRATION.md) for public naming changes, source provenance,
and validation scope. Existing `.pnmir` packages continue loading; manifest
and payload formats remain unchanged. New builder outputs use the
[flat backend layout](../docs/packages.md#generic-build-output).
