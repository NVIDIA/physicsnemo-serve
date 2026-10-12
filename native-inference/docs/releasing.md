# Builder packaging and verification

Current tooling builds a frontend wheel, a development builder image and model
candidates with native verification. It does not publish images, SDK archives
or models. `model-builder/toolchain.lock.json` remains explicitly unreleased until a
qualified default image is published.

## Build and check the wheel

From the repository root, use an isolated packaging environment with
`setuptools>=80` and `wheel` installed:

```bash
wheel_workdir="$(mktemp -d)"
cp -R native-inference/model-builder "$wheel_workdir/source"
python3 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir out/inference/wheels "$wheel_workdir/source"
PNMIR_TEST_WHEEL="$PWD/out/inference/wheels/pnms_model_builder-0.1.0-py3-none-any.whl" \
  python3 -m unittest discover \
    -s native-inference/model-builder/tests/installation -p 'test_installed*.py' -v
```

The scratch copy keeps setuptools build directories and egg metadata outside
versioned source. Preserve that directory if you need packaging diagnostics.

The wheel has no required ML dependencies. It includes the frontend/exporter
modules and the bundled affine resources from `model-builder/models/`.
GeoTransolver's project and adapter stay in `examples/geotransolver-surface-core/`
and are not part of the frontend wheel.
Installed checks exercise discovery and project/recipe behavior from unrelated
working directories without relying on checkout source. Local export requires
the selected model/backend's framework environment; container execution supplies
that through the builder image.

The canonical Python package version is in `model-builder/pyproject.toml`, and the
native SDK project version is in `cpp-runtime/CMakeLists.txt`, both relative to
`native-inference/`. Update the wheel filename above when selecting another version.
There is no separate `native-inference/VERSION` authority.

## Build and qualify the development image

Follow the [builder image commands](user-guide.md#use-docker).
The [Dockerfile](../model-builder/images/Dockerfile.builder) installs the native SDK
once and the wheel from the same source. Its default native prefix is
`/opt/physicsnemo-inference`, with `bin/physicsnemo-infer` used by the harness.
It retains the NVIDIA base entry point for driver compatibility initialization;
its Dockerfile-specific ignore file limits the context to inference source.

The [image environment contract](../model-builder/images/README.md) documents pinned
producer dependencies and protects the base Torch/CUDA stack. Run its guard
checks after dependency changes:

```bash
python3 -m unittest discover -s native-inference/model-builder/tests/images -v
```

Qualify the resulting immutable image with actual model preparation, backend
compilation and C++ output checks on each intended GPU/runtime combination.
The image's dependency lock alone does not establish model parity. Neither CPU
unit substitutes nor an SDK compilation alone qualify a GPU model build.

The public native interface uses `physicsnemo::inference`, headers under
`physicsnemo/inference/`, CMake package `PhysicsNeMoInference`, target
`PhysicsNeMoInference::runtime` and executable `physicsnemo-infer`. Customers
migrating an older native integration must update their source/build scripts
and rebuild. The manifest/artifact format and existing `.pnmir` loading
contract are unchanged; see [SDK build and usage](../cpp-runtime/README.md). New model
builds use the [flat backend package layout](reference.md#build-output).

## Model candidates today

Each build retains graphs, source/input identities, native checks, environment
and diagnostics. `model-release.json` inventories deployable files relative to
`model/`; its name does not imply registry publication or a production release.
The native CLI directly loads a `backends/<backend>/` package. See the
[build output reference](reference.md#build-output) for artifact locations,
[verification and failures](reference.md#verification-and-failures), and
[package compatibility](reference.md#package-compatibility).

The external [GeoTransolver example](../examples/README.md#geotransolver) uses generic
checkpoint import and core/native checks on synthetic feature inputs. These
checks do not establish full-model/core agreement, full geometry processing,
scientific CFD acceptance or compatibility with another deployment stack.

Generic local builds do not capture the complete installed Python implementation
and native dependency closure. Retain the source revision, dependency identities
and full logs alongside a candidate. Existing receipts are development evidence;
no signing or immutable release catalog is implemented.
