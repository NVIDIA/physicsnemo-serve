# PhysicsNeMo Model Builder

This component owns model preparation, graph export, backend compilation and
native package verification. To bring an existing Python model and weights,
follow [Add your model](../docs/add-model.md): initialize a project, connect two
adapter functions, then check and build. A [model build project](../docs/model-build-projects.md)
stores the checkpoint, builder environment and execution profile, and also
supports bundled recipes and external explicit recipes.

From the repository root, use `./native-inference/physicsnemo-model-builder`. The
installed command is `physicsnemo-model-builder`. Both invoke the same
implementation and validation gates.

```bash
./native-inference/physicsnemo-model-builder init /data/my-model
# Edit the generated model-build.json and build_adapter.py.
./native-inference/physicsnemo-model-builder check /data/my-model --json
./native-inference/physicsnemo-model-builder build /data/my-model --json
```

Initialization does not require weights, Torch or Docker. Before checking or
building, fill in the checkpoint and model/environment settings. Format-2
projects capture explicitly selected local Python sources and generate their
recipes internally; the customer does not need a per-model Dockerfile unless
the environment needs additional dependencies.

## Install the frontend

The Python package source root is `native-inference/model-builder`, with metadata in
`native-inference/model-builder/pyproject.toml`. Build a wheel using the
[scratch packaging instructions](../docs/releasing.md#build-and-check-the-wheel),
then install that artifact:

```bash
python3 -m pip install /path/to/physicsnemo_model_builder-0.1.0-py3-none-any.whl
physicsnemo-model-builder list --json
```

The frontend requires Python 3.10+ and has no mandatory ML dependencies. Its
wheel includes the exporter modules and affine recipe. GeoTransolver lives in
the external [customer example](../examples/README.md#geotransolver);
its project and adapter are not installed as builder resources.
Discovery and container orchestration do not import Torch on the host. Building a model locally requires the separate compatible framework and
native runtime environment described below.

For development in a checkout, use the launcher directly. See
[wheel construction and installed checks](../docs/releasing.md#build-and-check-the-wheel)
for packaging validation.

## Builder environment

Docker and a local Python environment are both supported. Docker remains the
default executor; `setup-env` creates a reusable local environment and can build
an already configured format-2 model project in the same command.

Native Windows uses the local executor. Follow the [Windows guide](../docs/windows.md)
for a PowerShell setup with both AOTInductor and TensorRT, matching CUDA/Triton
dependencies, and L4 validation. Run the checkout launcher as
`python native-inference/physicsnemo-model-builder` on Windows. Setup returns a
PowerShell activation wrapper that also configures native DLL search paths.

For a trusted PhysicsNeMo `.mdlus` checkpoint, the checkout or installed CLI can
generate reusable weights and constructor settings with
`physicsnemo-model-builder import-checkpoint model.mdlus --output imported --json`.
It reads the current project's environment, preserves the project file, and
requires a fresh output directory. See [checkpoint import](../docs/add-model.md#import-a-physicsnemo-checkpoint)
for container/local usage and the generated files.

### Create a local environment and build

From the repository root, after connecting your model's adapter, checkpoint and
validation inputs:

```bash
./native-inference/physicsnemo-model-builder setup-env \
  "$HOME/.venvs/physicsnemo-builder" \
  --requirements /data/my-model/requirements.txt \
  --build /data/my-model --json
```

The environment directory must be new. The command creates a Python venv,
installs Model Builder and its selected export dependencies, installs the
optional model requirements, and compiles the C++ runtime against that
environment. It then runs the normal local model build, including eager checks
and native parity. Omit `--requirements` when the model needs no additional
packages; repeat it to supply multiple files. Use `--python /path/to/python`
to choose the venv's interpreter.

Without `--build`, setup only prepares the environment. The default backend is
AOTI. With `--build`, setup uses the project's backends unless you override them
with repeated `--backend` arguments. For both AOTI and TensorRT:

```bash
./native-inference/physicsnemo-model-builder setup-env \
  "$HOME/.venvs/physicsnemo-builder-dual" \
  --requirements /data/my-model/requirements.txt \
  --backend aoti --backend tensorrt \
  --tensorrt-root /opt/TensorRT \
  --tensorrt-cuda-major 13 \
  --build /data/my-model --json
```

Local setup needs Python 3.10+, a C++ compiler and CMake 3.20+. CUDA builds also
need a compatible NVIDIA driver and CUDA development toolkit already on the
host. TensorRT builds require its full C++ SDK, including headers and libraries,
and a matching Python package; the pip TensorRT package alone does not supply
the C++ headers. Setup pins the TensorRT Python version to the SDK headers and
selects CUDA 13 packages by default. Use `--tensorrt-cuda-major 12` for a CUDA 12
SDK/toolkit; avoid generic `tensorrt` or conflicting CUDA variants in model
requirements. Pin compatible framework and model dependencies in your requirements
file. Setup does not install system toolchains or drivers,
and dependency installation alone does not qualify a model's exportability.

The command prints activation and build commands plus `project_settings` for
`executor: "local"` and the runtime path. It does not edit `model-build.json`.
Copy those settings into the project for subsequent builds, then activate the
environment:

```bash
source "$HOME/.venvs/physicsnemo-builder/bin/activate"
physicsnemo-model-builder check /data/my-model --executor local --json
physicsnemo-model-builder build /data/my-model --json
```

Setup keeps the compiled runtime in the environment's `.physicsnemo/runtime/`
directory; it is reused across model builds. To reuse a compatible runtime
instead of compiling one, pass `--runtime /path/to/physicsnemo-infer`. It must
match the framework and backend libraries in the new environment.

The checkout launcher locates the package and SDK sources automatically. When
using an installed frontend, provide
`--sdk-source /path/to/checkout/native-inference/cpp-runtime` to select the SDK
and its sibling Model Builder source. If reusing a runtime without an SDK
checkout, supply both `--runtime` and
`--builder-package /path/to/physicsnemo_model_builder-0.1.0-py3-none-any.whl`.
`--builder-package` can also select a package source directory.

### Use an existing Python environment

Activate your environment and install the builder with its export dependencies
from the checkout root:

```bash
python -m pip install './native-inference/model-builder[export]'
# Install your model's requirements in this same environment.
physicsnemo-model-builder build /data/my-model \
  --executor local --runtime /path/to/sdk/bin/physicsnemo-infer --json
```

Use the `[tensorrt]` extra for TensorRT export. Local execution needs compatible
frameworks, compiler tools and a [native SDK](../cpp-runtime/README.md) built for
those libraries. Optional wheel dependency ranges alone do not qualify a
toolchain. Qualify changed environments with actual model and native checks.
GeoTransolver uses the generic `import-checkpoint` command, which needs the
model Python environment but no native runtime.

For format-2 authoring projects, local `check` needs only the compatible model
and framework environment; it does not require the SDK or compile a backend.
`build` repeats eager validation and then requires the native runtime for
parity. Set `executor: "local"` and `runtime` in the project to reuse these
settings, or pass the explicit flags as above.

### Use Docker

The default executor runs Docker with a digest-pinned image containing Python,
export frameworks, compilers and an installed C++ SDK. The builder's
`toolchain.lock.json` is explicitly unreleased; it does not select a public
image yet. On a Linux x86_64 Docker host, build a development image from the
repository root:

```bash
docker build \
  -f native-inference/model-builder/images/Dockerfile.builder \
  -t physicsnemo-model-builder:dev .
docker image inspect physicsnemo-model-builder:dev --format '{{.Id}}'
```

This default image includes AOTI and the GeoTransolver model dependencies.
Add `--build-arg PNMIR_ENABLE_TENSORRT=ON` to the Docker build for generic
TensorRT recipes, including GeoTransolver. Request `--backend aoti --backend tensorrt`
to build and validate both backend packages; TensorRT requires CUDA.

Copy the image's `sha256:...` identity into your project's `builder_image`, or
supply `--builder-image`. Registry references must use `name@sha256:...`;
mutable tags are rejected. A local image ID is usable on the Docker host where
that image exists. Model builds use its already-installed SDK; they do not
compile the SDK again.

For example, with NVIDIA GPU access configured for Docker:

```bash
./native-inference/physicsnemo-model-builder build affine \
  --builder-image sha256:<image-id> \
  --backend aoti --device cuda --output /data/builds/affine-first
```

The output must be a fresh directory. Each selected backend produces a directly
loadable package at `model/backends/<backend>/`, with `model.json` beside
`model.pt2` for AOTI or `model.plan` for TensorRT. See the
[package contract](../docs/packages.md). The frontend stages selected source and
data read-only and runs the container as the customer's UID/GID. See the
[image environment contract](images/README.md) for pinned dependencies and
their update checks.

Container `check` and `build` use the configured image, which must
contain the current authoring worker.

## Source ownership

| Directory/file | Responsibility |
| --- | --- |
| `src/pnmir_build/` | Projects, recipes, input capture, executors, native harness, qualification and results |
| `src/pnmir_export/` | AOTI, ONNX and TensorRT export/compilation primitives |
| `models/affine/` | Canonical bundled recipe and adapter |
| `images/` | Builder Dockerfile, dependency locks and environment guards |
| `tests/` | Frontend, export and harness tests |
| `tests/installation/` | Installed-wheel contract tests |
| `tests/images/` | Image environment guard tests |
| `pyproject.toml` | Wheel metadata and canonical model resources |
| `toolchain.lock.json` | Default toolchain selection, currently unreleased |

The C++ runtime stays in `../cpp-runtime/`. Customer examples stay in
`../examples/`, with one project file and adapter per example. Shared preparation
tools live in `../tools/examples/`; tests live in `../tests/examples/`. Generated checkpoints, wheels, graphs, packages and test evidence
belong in separate output directories, not these source directories.

## Development checks

From the repository root:

```bash
python3 -m unittest discover -s native-inference/model-builder/tests -v
python3 -m unittest discover -s native-inference/model-builder/tests/images -v
```

ML-dependent tests need their declared frameworks; report skips separately.
CPU substitutes verify orchestration and failure handling, while GPU builds
and native execution establish backend parity. Use recorded red–green tests
for behavioral changes and before/after characterization for source moves.
See [packaging and verification](../docs/releasing.md) for wheel and builder-image checks.
