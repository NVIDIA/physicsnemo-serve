# PhysicsNeMo Model Builder and C++ Inference SDK

Build a native package from your Python model and trained weights with
**PhysicsNeMo Model Builder**, then run it through the **PhysicsNeMo C++
Inference SDK**.

**Start with the [user guide](docs/user-guide.md).** It covers environment setup,
project configuration, the two adapter functions, checking, building and using
an exported model. You provide representative inputs; the builder compares
compiled native outputs with your Python model before accepting the build.

## Choose a guide

| Task | Guide |
| --- | --- |
| Bring your model and weights | [Model Builder user guide](docs/user-guide.md) |
| Look up settings, profiles, recipes or package formats | [Technical reference](docs/reference.md) |
| Start from a prepared model template | [Examples: affine, Transolver, GeoTransolver and DoMINO](examples/README.md) |
| Build, install or embed the native runtime | [C++ Inference SDK](cpp-runtime/README.md) |
| Set up native Windows | [Windows setup](docs/user-guide.md#windows-setup) |
| Run raw mesh → native model → physical fields | [Transolver C++ workflow](workflows/transolver/README.md) |

From the checkout root, the launcher is
`./native-inference/pnms-model-builder`; the installed command is
`pnms-model-builder`, where `pnms` stands for PhysicsNeMo Serve. The frontend
requires Python 3.10+. Model execution and compilation require the selected
framework and native environment; follow
[Linux setup](docs/user-guide.md#linux-setup) or
[Windows setup](docs/user-guide.md#windows-setup) before building.

## What a build produces

A successful build creates a deployable `model/` bundle. Each requested backend
has a loadable package at `model/backends/<backend>/`, containing `model.json`
and its compiled payload. The runtime loads that backend directory directly.
See [using the compiled model](docs/user-guide.md#use-the-compiled-model).

The native executable is `physicsnemo-infer`; applications can also use the
`physicsnemo::inference::v1` C++ API. Runtime inference needs compatible native
libraries and input tensors, but does not load your Python model or original
checkpoint. Docker is optional for deployment.

## Current scope

Model Builder supports static FP32 authoring projects for AOTInductor and
TensorRT. The SDK additionally supports ONNX Runtime packages. Existing
`.pnmir` and `.pnm-model` packages remain loadable through their manifests.
Model-specific exact profiles and their prerequisites are described in the
[reference](docs/reference.md#tensorrt-profiles-and-assets).

Published default builder images, complete prebuilt SDK bundles, artifact
publication and integration with the Serve execute worker remain planned.
GeoTransolver and DoMINO examples verify surface cores with synthetic feature
inputs; complete geometry workflows and scientific CFD acceptance need separate
qualification. The Transolver workflow includes bounded raw-geometry processing.
See [deployment compatibility](docs/reference.md#package-compatibility).

For contributors: [builder development](model-builder/README.md),
[packaging and release checks](docs/releasing.md), and
[native inference QA](../qa/native_inference/README.md).

## Source layout

| Area | Responsibility |
| --- | --- |
| `model-builder/src/model_builder/build/` | The public CLI, project configuration, captured inputs and build orchestration. |
| `model-builder/src/model_builder/export/` | Python export APIs, compiler profiles and graph passes used by the builder. |
| `cpp-runtime/` | C++ SDK, native CLI, backend plugins and their integration tests. |
| `examples/`, `tools/examples/`, `tests/examples/` | Two-file customer projects, input preparation and one discoverable example test suite. |
| `workflows/transolver/` | Optional native geometry workflow built as a consumer of the installed SDK. |
| `docs/` | User manual, configuration reference, Windows SDK and release instructions. |
| [`../qa/native_inference/`](../qa/native_inference/README.md) | Independent end-to-end qualification and artifact checks. |

Builder tests stay beside the builder, native tests beside the SDK, and QA
controller tests under the repository's `tests/`. Exact TensorRT operators keep
separate shared libraries because packages record their individual identities.
