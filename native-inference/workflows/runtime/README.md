# Reusable native workflow host

This library embeds persistent PhysicsNeMo C++ Inference SDK executors behind a
small JSON facade. `tensor.v1` serves static FP32 models without user C++.
`custom.v1` source-links a user workflow implementing one request hook. Both
use the same model aliases, startup tests, cancellation and artifact registration.
An embedding application creates, uses and destroys the host on one owning OS thread.

The first version supports contiguous CPU FP32 inputs and owned CPU FP32
outputs through the selected SDK backend. Device transfer remains inside that
backend. Shapes and tensor names come from the package manifest. Dynamic shapes,
other dtypes, runtime plugin loading and automatic backend fallback are rejected
or outside this contract. Packages requiring extra custom operator registration
need their own qualified integration; the existing Transolver adapter remains
separate.

## Build and embed

Build against an installed C++ SDK, selecting the production backend explicitly:

```bash
cmake -S native-inference/workflows/runtime -B build/workflow \
  -DCMAKE_PREFIX_PATH=/opt/physicsnemo-inference \
  -DPNMS_ENABLE_AOTI=ON -DPNMS_ENABLE_MOCK_BACKEND=OFF \
  -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/opt/physicsnemo-native-runner
cmake --build build/workflow --parallel
cmake --install build/workflow
```

This installs `pnms-native-run`, `libphysicsnemo_workflow`, headers under
`physicsnemo/workflow/v1/`, and CMake targets
`PhysicsNeMoWorkflow::host` and `PhysicsNeMoWorkflow::authoring`.
The facade in `host.hpp` exposes only strings and a cancellation callback.
SDK objects and tensors remain in C++.

The standalone runner calls the same host on one OS thread. It performs the
authored startup self-test before executing the request and never compiles a
model at startup:

```bash
/opt/physicsnemo-native-runner/bin/pnms-native-run \
  --config /absolute/path/workflow.json \
  --request /absolute/path/request-envelope.json \
  --output /absolute/path/result.json \
  --startup-report /absolute/path/startup.json
```

Use `--self-test` instead of `--request` to verify startup only. Without
`--output`, the single result JSON is written to stdout. Diagnostic errors go
to stderr with exit code 1; invalid command arguments return 2. Existing result
files are not overwritten. SIGINT/SIGTERM request cooperative cancellation.
The config and request formats are described below. Callers supply the compiled
model packages and configuration; this executable does not build or package them.

Direct inference needs the installed native libraries and compatible backend
dependencies. It requires neither Rust, Redis, Docker nor Lepton. This generic
host does not link Transolver, VTK or Warp.

```cpp
physicsnemo::workflow::v1::WorkflowHost host(config_json);
const auto startup = host.self_test();  // Must succeed before readiness.
const auto result = host.run(request_json, [&] { return cancellation_requested; });
```

`run` refuses requests before `self_test` passes. Every configured alias must
have authored finite inputs and expected outputs. Startup executes each actual
model and compares all outputs using the configured symmetric absolute/relative
tolerances. It does not generate expected numerical outputs for unknown models.
The retained startup report records the alias, package, backend, output
signature, maximum error and resident session count. A repeated startup query
returns that report without reloading or rerunning the models.

## Configuration and tensor mapping

This example assumes an affine model with `output = 2 * input + 1` and shape
`[4]`. A different model needs its own expected fixture.

```json
{
  "schema_version": 1,
  "adapter": "tensor.v1",
  "models": {
    "primary": {
      "package_path": "/opt/physicsnemo-release/model/primary/backends/aoti",
      "backend": "aoti",
      "device": "cuda",
      "device_index": 0,
      "self_test": {
        "inputs": {"input": [0, 1, -1, 4]},
        "expected_outputs": {"output": [1, 3, -1, 9]},
        "rtol": 0.00001,
        "atol": 0.000001
      }
    }
  },
  "tensor_adapter": {
    "model_alias": "primary",
    "inputs": {"input": {"request_field": "values", "codec": "json-array-fp32"}},
    "outputs": {"output": {"response_field": "values", "codec": "json-array-fp32"}}
  },
  "assets": {},
  "limits": {"max_json_bytes": 1048576, "max_tensor_bytes": 67108864}
}
```

JSON tensors are flat arrays in contiguous row-major order. Counts must match
the declared shape exactly. The first tensor adapter requires exactly the
mapped request fields. Missing/extra tensors, nonnumeric values, NaN/Inf,
nonstatic shapes, unsupported codecs and oversized payloads fail explicitly.
The byte limit applies to each tensor; deployment capacity qualification must
also account for all inputs, outputs and resident models together.

The trusted embedding caller supplies this request envelope after input admission:

```json
{
  "request_id": "request-123",
  "parameters": {"values": [0, 1, -1, 4]},
  "output_dir": "/outputs/request-123/attempt-456",
  "inputs": {}
}
```

The result is `{"payload":{"values":[1,3,-1,9]},"artifacts":[],"metadata":...}`.
The embedding caller decides how to persist the payload and publish artifacts.

For large tensors choose `raw-fp32-le`. The parameter becomes:

```json
{"artifact":"features","dtype":"float32","shape":[4],"byte_order":"little","layout":"contiguous","size_bytes":16}
```

`features` must resolve through the envelope's trusted `inputs` map. Parameters
cannot introduce filesystem paths or download URLs. The file must be a regular
file with exactly the declared byte count, and its float values must be finite.
Raw outputs are registered as relative `<tensor-name>.f32` artifacts and returned
with the same descriptor fields. An embedding controller must validate registered
files and compute any publication hashes. Attempt output paths and admitted input/asset
paths must be canonical and free of symlinks. Cancellation is cooperative around
model calls and output publication; it cannot preempt an in-flight CUDA kernel.
Raw output tensor names use at most 124 ASCII letters, digits, underscores or
hyphens, leaving room for the `.f32` suffix within the 128-byte artifact name.

## Source-linked custom workflows

Include `authoring.hpp` and supply the factory symbol:

```cpp
std::unique_ptr<physicsnemo::workflow::v1::Workflow>
physicsnemo::workflow::v1::create_custom_workflow();
```

Its workflow implements `Outcome run(const Request&, Context&)`. `Request`
contains the request ID and canonical parameter JSON. `Context` exposes fixed
model aliases through `infer`/`model`, admitted `input` files, configured immutable
`asset` files, cancellation and attempt-relative output registration. Blocking
`infer` returns an SDK `Result` retaining owned CPU output storage; retain that
result while reading its views. The hook returns JSON objects for payload and
metadata. Redis, scheduler state and bridge details are not part of this API.
Registered artifact names use at most 128 ASCII letters, digits, underscores,
hyphens or dots, excluding `.` and `..`; media types must exclude CR/LF.

The tested [custom fixture](tests/custom/workflow.cpp) demonstrates inference,
postprocessing and output registration. A minimal child CMake project is:

```cmake
add_library(my_native_workflow STATIC workflow.cpp)
set_target_properties(my_native_workflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(my_native_workflow PRIVATE
  PhysicsNeMoWorkflow::authoring nlohmann_json::nlohmann_json)
```

Configure the host with
`-DPNMS_CUSTOM_WORKFLOW_SOURCE_DIR=/absolute/path/to/native`
and `-DPNMS_CUSTOM_WORKFLOW_TARGET=my_native_workflow`, then select `custom.v1`
in its deployment configuration. The parent build supplies the SDK and JSON
targets before including that child directory. No Cargo, scheduler or host
source edit is required for a new user workflow. Dependencies beyond the SDK
must be declared by the custom target and included in its release image.

## CPU verification

Use a CPU-only SDK installation and enable its test backend:

```bash
cmake -S native-inference/workflows/runtime -B build/workflow-cpu \
  -DCMAKE_PREFIX_PATH=/path/to/cpu-sdk -DPNMS_ENABLE_MOCK_BACKEND=ON
cmake --build build/workflow-cpu --parallel
ctest --test-dir build/workflow-cpu --output-on-failure
```

The mock backend is test-only and does not establish GPU compatibility. The
tests execute actual SDK model calls, expected-output startup failure, residency,
JSON/raw round trips, cancellation, byte/count limits, unsafe path rejection and
thread ownership. Configure the `tests/custom` child target `pnms_test_custom`
to exercise a source-linked custom hook. After installing the host, configure
`tests/install_consumer` with both the SDK and workflow prefixes in
`CMAKE_PREFIX_PATH`, then build and run its CTest. Its generated `config.json`
also supplies the mock model fixture for standalone-runner checks:

```bash
python3 -S native-inference/workflows/runtime/tests/cli_test.py \
  --runner /path/to/workflow/bin/pnms-native-run \
  --config /path/to/consumer-build/config.json -v
```

CI runs the host/custom, installed-consumer and standalone-runner checks. Actual
CUDA builds and numerical qualification run through Lepton on the configured H100
resources; CPU mock results do not establish GPU compatibility.
