# PhysicsNeMo Model Builder and C++ Inference SDK

Build a native model package from Python with **PhysicsNeMo Model Builder**,
then execute its backend packages with the **PhysicsNeMo C++ Inference SDK**.
Both live here in PhysicsNeMo Serve; integration with the Serve execute worker
is still planned.

## Build a model

From the repository root, discover the bundled recipes:

```bash
./native-inference/physicsnemo-model-builder --help
./native-inference/physicsnemo-model-builder list
```

The launcher needs Python 3.10 or newer. Discovery, project validation and
container orchestration need no host Torch or PhysicsNeMo installation.

For repeated builds, put the model, checkpoint and execution settings in a
`model-build.json` file. The [project guide](docs/model-build-projects.md)
describes both authoring projects and explicit recipes, including the external
GeoTransolver example:

```bash
./native-inference/physicsnemo-model-builder doctor /path/to/model-project --json
./native-inference/physicsnemo-model-builder build /path/to/model-project --json
```

For your own Python model and trained weights, start with
[Add your model](docs/add-model.md). `init` creates a project configuration and
an adapter with two functions to fill in; no checkpoint argument is required.
For a PhysicsNeMo `.mdlus` archive, `import-checkpoint` generates a plain tensor
checkpoint and constructor JSON, then prints the settings for you to copy into
the project configuration. It reuses the project's configured environment.

```bash
./native-inference/physicsnemo-model-builder init /path/to/model-project
# Set checkpoint/environment in model-build.json and connect build_adapter.py.
./native-inference/physicsnemo-model-builder check /path/to/model-project --json
./native-inference/physicsnemo-model-builder build /path/to/model-project --json
```

Format-2 authoring projects capture selected Python sources and infer the
tensor contract from your cases. `check` executes the Python model without
compilation; `build` repeats those checks and requires native parity.

`doctor` checks configuration and input identities. `build` prepares the model,
exports and compiles the selected backends, and runs the required native parity
checks. Project profiles select execution settings; the generated lock detects
changed inputs. The commands enforce these contracts without an agent.

Docker is the default executor. There is no published default builder image
yet: follow the [builder setup](model-builder/README.md#builder-environment) to build
a development image once, create a local Python environment, or use an existing
compatible environment and native runtime. Changing model weights does not
rebuild the SDK.

To create a fresh venv and build an already configured format-2 model project
without Docker, run from the repository root:

```bash
./native-inference/physicsnemo-model-builder setup-env \
  "$HOME/.venvs/physicsnemo-builder" \
  --requirements /path/to/model-project/requirements.txt \
  --build /path/to/model-project --json
```

This installs the builder and model dependencies, builds a matching C++ runtime
once, then exports the model and runs native parity checks. Omit `--build` to
prepare the environment first. Setup prints local executor/runtime settings for
you to copy into `model-build.json`. It requires host compiler tools; CUDA and
TensorRT builds also require their development toolkits. See
[local setup and existing environments](model-builder/README.md#create-a-local-environment-and-build)
for prerequisites and both-backend commands.

## Choose a starting point

| Task | Guide |
| --- | --- |
| Configure a repeatable model build | [Projects, profiles and locks](docs/model-build-projects.md) |
| Export a GeoTransolver checkpoint | [External GeoTransolver example](examples/README.md#geotransolver) |
| Export a DoMINO surface core | [External DoMINO example](docs/domino-workflow.md) |
| Supply a custom model and weights | [Initialize, check and build your model](docs/add-model.md) |
| Start from a model example | [Two-file templates and preparation](examples/README.md) |
| Run raw mesh → native model → physical fields from the CLI | [C++ Transolver E2E workflow](workflows/transolver/README.md) |
| Maintain an explicit model recipe | [Recipe/input contract](docs/model-inputs.md) |
| Exercise export and native verification | [Bundled affine recipe](model-builder/models/affine/README.md) |
| Find or move a graph/package | [Package and evidence contract](docs/packages.md) |
| Install the frontend or build its image | [Model Builder](model-builder/README.md) |
| Build, install or embed the native runtime | [C++ Inference SDK](cpp-runtime/README.md) |
| Package and verify the builder | [Packaging instructions](docs/releasing.md) |

## Run a package

A successful build produces a deployable `model/` directory. The
`physicsnemo-infer` CLI loads a backend package directly from
`model/backends/<backend>/`; the SDK can also be embedded through its
`physicsnemo::inference::v1` C++ API. Existing `.pnmir` packages remain loadable.
Native execution does not load the Python model or original checkpoint. A server can run directly
with the compatible native dependencies; Docker is optional for deployment.

The current SDK supports optional AOTInductor, ONNX Runtime and TensorRT
backends. Its core static/shared installations are tested for relocation;
complete GPU dependency bundles and published compatibility matrices remain
work for a release. See [package compatibility](docs/packages.md#deployment-and-compatibility).

The builder supports generic static FP32 AOTI/TensorRT recipes. The external
GeoTransolver and DoMINO examples build surface cores using imported weights and
synthetic feature inputs.
The [Transolver CLI example](workflows/transolver/README.md) runs raw surface
and volume geometry through C++ preprocessing, SDK inference and physical-unit
decoding. It includes a Model Builder project for a bounded surface case.
Complete raw-mesh GeoTransolver/DoMINO flows, scientific model acceptance,
remote build jobs and artifact publication remain planned.
