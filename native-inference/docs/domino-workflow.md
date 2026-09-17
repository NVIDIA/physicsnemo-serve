# DoMINO surface-core example

The [DoMINO example](../examples/README.md#domino) contains only
`model-build.json` and `adapter.py`. It uses the generic checkpoint importer and
the normal `check` and `build` commands. The adapter keeps DoMINO's learned
local encoders, positional encoder and upstream surface solution head; its
inputs are prepared FP32 neighborhoods.

## Checkpoint and environment

The public reference checkpoint is
[`nvidia/domino_drivaerml` at revision `35b1bf1edafdaa2600d16182825890cd51c07427`](https://huggingface.co/nvidia/domino_drivaerml/tree/35b1bf1edafdaa2600d16182825890cd51c07427/domino_drivaerml_surface_checkpoint),
file `domino_drivaerml_surface_checkpoint/DoMINO.0.501.mdlus`.
Download this trusted archive separately. You can also pass an expected digest
with `import-checkpoint --checkpoint-sha256`.

Use a compatible PhysicsNeMo 2.1.1, Torch, Warp and CUDA environment with
Model Builder, TensorRT, ONNX/ONNXScript and a matching C++ Inference SDK.
Build the SDK with these options in addition to its dependency paths:

```text
-DPNMIR_ENABLE_AOTI=ON
-DPNMIR_ENABLE_TENSORRT=ON
-DPNMIR_ENABLE_TENSORRT_EXACT=ON
-DPNMIR_BUILD_DOMINO_EXACT_OPS=ON
```

See the [SDK build instructions](../cpp-runtime/README.md) and
[builder environment setup](../model-builder/README.md#builder-environment).
The project needs the installed AOTI sidecar `libpnmir_domino_exact_ops.so`
and four exact TensorRT libraries: Linear, GELU, ScalarDiv and
InverseDistanceBlend. Build and execute with matching native libraries.

From the repository root, using a fresh project directory:

```bash
DEMO_PROJECT=/tmp/domino-demo
SDK_ROOT=/path/to/matching/sdk
cp -R native-inference/examples/domino-surface-core "$DEMO_PROJECT"
mkdir -p "$DEMO_PROJECT/assets/aoti" "$DEMO_PROJECT/assets/tensorrt"
cp "$SDK_ROOT/lib/libpnmir_domino_exact_ops.so" "$DEMO_PROJECT/assets/aoti/"
for plugin in linear gelu scalar_div inverse_distance_blend; do
  cp "$SDK_ROOT/lib/libpnmir_tensorrt_exact_${plugin}_plugin.so" "$DEMO_PROJECT/assets/tensorrt/"
done
physicsnemo-model-builder import-checkpoint \
  /path/to/DoMINO.0.501.mdlus \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
physicsnemo-model-builder check "$DEMO_PROJECT" --json
physicsnemo-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_ROOT/bin/physicsnemo-infer" --json
```

The importer creates `weights/config.json`, `weights/checkpoint.pt` and
`weights/import.json`. The adapter retains upstream state names so the builder
can strictly load the imported weights. It accepts the supported surface
configuration and rejects unsupported branches. Model Builder captures the
adapter, weights, configuration and declared library assets, and binds their
identities into the project lock. Intentional changes after a locked build
require `build --update-lock`.

## Inputs and scope

All nine inputs are FP32. The default cases use a batch of one and 32 surface
points, with six non-self neighbors per point:

| Input | Default shape | Meaning |
| --- | --- | --- |
| `local_neighbors_0` | `[1, 32, K0 * C]` | Gathered, masked local-grid features at the first query scale, flattened in channel order. |
| `local_neighbors_1` | `[1, 32, K1 * C]` | The same prepared features at the second query scale. |
| `centers` | `[1, 32, 3]` | Surface cell centers in model coordinates. |
| `relative_positions` | `[1, 32, 3]` | Relative positions used by the positional encoder. |
| `neighbor_centers` | `[1, 32, 6, 3]` | The six selected neighboring cell centers. |
| `normals` | `[1, 32, 3]` | Surface cell normals. |
| `neighbor_normals` | `[1, 32, 6, 3]` | Normals for those same neighboring cells. |
| `areas` | `[1, 32, 1]` | Positive surface cell areas in the model's input convention. |
| `neighbor_areas` | `[1, 32, 6, 1]` | Positive areas for the same neighboring cells. |

`K0` and `K1` come from
`model_parameters.geometry_local.surface_neighbors_in_radius`.
`C` is one plus the number of
`model_parameters.geometry_rep.geo_conv.surface_radii` entries.
For application inputs, local neighborhoods must follow upstream preparation,
including the global encoding's factor of `0.5`, gathering, zero-index masking
and feature order. Neighbor centers, normals and areas must describe the same
ordered neighbors.

The adapter generates three deterministic synthetic cases directly. They
exercise export and native execution but do not represent a real car mesh.
The output `surface_fields_standardized` has shape `[1, 32, 4]`, containing
pressure and three wall-shear-stress components in standardized model space.

Global geometry/grid projection, spatial queries, neighbor gathering and
physical-unit decoding remain outside this exported core. Applications must
provide matching preprocessing. These build checks establish core Python/C++
parity on the supplied cases; they do not establish complete raw-mesh inference,
full-model/core agreement or CFD prediction accuracy.

## Exact profiles and outputs

The default project builds both backends on CUDA:

| Backend | Profile | Selected mechanism |
| --- | --- | --- |
| AOTI | `aten-boundary-exact-v3` | Existing ATen-preserving settings with `shape_padding: false`, plus the declared DoMINO tensor-only sidecar. |
| TensorRT | `domino-surface-exact` | Exact supported Linear layers throughout the core, following GELUs, scalar division and inverse-distance blending. |

For AOTI, the builder rewrites supported arithmetic boundaries in the captured
graph to the sidecar's ATen C++ operators. It retains the original eager
reference model and checkpoint weights. The package declares the required
sidecar operator ID and ABI; the runtime must register that matching dependency.

TensorRT substitutes supported Linear operations throughout the local encoders,
positional encoder and solution head, together with their following GELUs.
ScalarDiv preserves the selected FP32 division-by-ten behavior, and
InverseDistanceBlend preserves the order and rounding of neighbor weighting,
accumulation and normalization. The package records the selectors, replacement
counts and four plugin identities. TensorRT executes its own engine; the
AOTI sidecar belongs to the AOTI backend.

`check` runs eager validation. `build` exports, compiles and invokes the actual
C++ runtime for every case and selected backend. Both exact profiles require
identical output bytes, zero maximum absolute error and zero relative-L2 error,
alongside shape, dtype and finite-value checks. A mismatch fails the build.
This is a per-build acceptance rule, not a guarantee across untested inputs or
different GPU/software stacks.

Successful builds create:

```text
builds/<build-id>/
  source/                     # captured inputs and adapter
  exported/                   # exported graphs
  checks/{aoti,tensorrt}.json  # byte-parity results and output hashes
  build.json
  project/                    # effective settings and lock snapshot
  model/
    model-release.json
    backends/aoti/{model.json,model.pt2}
    backends/tensorrt/{model.json,model.plan}
```

Each check report records `require_byte_identical: true` and zero error limits.
The C++ CLI loads the selected backend directory with the nine prepared input
tensors. It needs the matching SDK and native dependencies, rather than the
Python adapter or original training archive. See
[package compatibility](packages.md#deployment-and-compatibility).
