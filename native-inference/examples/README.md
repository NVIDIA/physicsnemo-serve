# Model examples

Each example folder contains exactly two files: `model-build.json` and
`adapter.py`. Copy or stage an example in a fresh directory, then import or
generate its inputs using the commands below. Build every project through the
normal Model Builder interface.

Use a compatible Python environment with Model Builder, the model dependencies
and a matching C++ Inference SDK. Run these Bash examples from the repository
root; project and checkpoint-import output directories must be new. See
[environment setup](../docs/user-guide.md#linux-setup) and
[Windows setup](../docs/user-guide.md#windows-setup) for platform prerequisites.

Choose [configured affine](#configured-affine), [AOTI profiles](#aoti-profiles),
[GeoTransolver](#geotransolver), [DoMINO](#domino), or
[Transolver](#transolver-surface-and-native-e2e-cli), then [check and build](#check-and-build).

## Configured affine

Demonstrates a model configuration, checkpoint and normalization asset. Requires
Torch; the project defaults to AOTI on CPU.

```bash
DEMO_PROJECT=/tmp/configured-affine-demo
python native-inference/tools/examples/prepare_configured_affine.py \
  --output "$DEMO_PROJECT"
```

The helper writes `checkpoint-a.pt`, `checkpoint-b.pt` and `normalization.json`.
The project selects checkpoint A; pass `--checkpoint "$DEMO_PROJECT/checkpoint-b.pt"`
and `--update-lock` to a subsequent build to compare B. For input `[0, 1, -1, 4]`,
A produces `[3, 4, 2, 7]` and B produces `[4, 3.5, 4.5, 2]`.

## AOTI profiles

Compares four compiler selections using the same small FP32 MLP, saved weights
and three synthetic cases. Requires a compatible Linux CUDA/Torch environment
and an AOTI-enabled SDK.

```bash
DEMO_PROJECT=/tmp/aoti-profiles-demo
python native-inference/tools/examples/prepare_aoti_profiles.py \
  --output "$DEMO_PROJECT"
```

The helper writes `weights.pt`, `cases.pt` and `features.bin`.

| Profile | Compiler profile | Explicit options |
| --- | --- | --- |
| `accuracy` (default) | `aten-boundary-exact-v2` | All four supported options `false`. |
| `standard` | `baseline` | Keep installed compiler defaults. |
| `autotune` | `baseline` | `max_autotune: true`, `epilogue_fusion: true`. |
| `autotune-no-fusion` | `baseline` | `max_autotune: true`, `epilogue_fusion: false`. |

Choose `--profile NAME` when building. Every selection must pass the same parity
checks; the names do not guarantee accuracy or speed. See
[compiler options](../docs/reference.md#aoti-profiles-and-options) and
[project locks](../docs/reference.md#project-locks).

## GeoTransolver

[`geotransolver-surface-core`](geotransolver-surface-core) exports the cached
surface core with pretrained weights and three deterministic synthetic cases.
The adapter creates its cases directly; no preparation script is required.

### GeoTransolver checkpoint and SDK


Use a compatible PhysicsNeMo 2.1.1, Torch, Warp and CUDA environment with
Model Builder and a matching C++ Inference SDK. TensorRT also requires its
Python/native libraries and ONNX/ONNXScript. See
[builder environments](../docs/reference.md#environment-setup).
The default project requires an AOTI-enabled SDK built with
`PNMIR_ENABLE_TENSORRT=ON` and `PNMIR_ENABLE_TENSORRT_EXACT=ON`. This installs
the matching exact plugin libraries, including all ten used by GeoTransolver v2. Follow the
[SDK build instructions](../cpp-runtime/README.md#exact-tensorrt-operators-for-transolver-and-geotransolver).

The public reference is
[`nvidia/geotransolver_drivaerml` at revision `626c1158e14f6994382924055aa871f863ff8a8c`](https://huggingface.co/nvidia/geotransolver_drivaerml/tree/626c1158e14f6994382924055aa871f863ff8a8c/geotransolver_drivaerml_surface_checkpoint),
file `geotransolver_drivaerml_surface_checkpoint/GeoTransolver.0.501.mdlus`.
Its recorded SHA-256 is
`56b2590af5857775f820b2bfecd74ca58476c5223266f253f12593be9ac9db34`.
Download the archive separately. `import-checkpoint` accepts
`--checkpoint-sha256` when an expected digest is available.

From the repository root, using a fresh project directory:

```bash
DEMO_PROJECT=/tmp/geotransolver-demo
SDK_ROOT=/path/to/matching/sdk
cp -R native-inference/examples/geotransolver-surface-core "$DEMO_PROJECT"
mkdir -p "$DEMO_PROJECT/assets/tensorrt"
cp "$SDK_ROOT"/lib/libpnmir_tensorrt_exact_*_plugin.so "$DEMO_PROJECT/assets/tensorrt/"
pnms-model-builder import-checkpoint \
  /path/to/GeoTransolver.0.501.mdlus \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
```

### Cached-core inputs and scope


The adapter subclasses upstream GeoTransolver, retaining its constructor and
state names. Its forward runs the local embedding MLP, global tokenizer, GALE
blocks and output head. It takes geometry-derived features as tensor inputs,
so the exported core does not execute neighborhood/radius search.

`create_cases` generates three deterministic synthetic cases with 32 points.
Feature dimensions come from the imported configuration. For the public
checkpoint, the input contract is:

| Input | Shape |
| --- | --- |
| `local_embedding` | `[1, 32, 6]` |
| `local_features` | `[1, 32, 192]` |
| `static_context` | `[1, 8, 128, 224]` |
| `global_embedding` | `[1, 1, 2]` |

`surface_fields_standardized` has shape `[1, 32, 4]`: pressure and three
wall-shear-stress components in standardized model space.

These synthetic inputs exercise core export and native execution. They are
not computed from a real geometry and do not establish full-model/core
agreement or physical CFD accuracy. Applications must supply meaningful
geometry-derived features separately. The complete upstream forward uses
`physicsnemo::radius_search_warp`; capturing that graph does not make this
operator available to the current native exporter/runtime.

Both backends target CUDA:

| Backend | Profile | Acceptance |
| --- | --- | --- |
| AOTI | `aten-boundary-exact-v2` | Exactness-preserving compiler settings with the standard AOTI parity checks. |
| TensorRT | `geotransolver-exact-v2` | Ten exact plugins and byte-identical eager/C++ outputs. |

The adapter supplies `NormalizeClampBounds`; the TensorRT profile also applies
scalar sigmoid freezing, WeightedBlend and layout-preserving DesliceBmm rewrites.
See [exact graph scope](../docs/reference.md#exact-graph-scope) for their bounds
and the older nine-plugin profile's compatibility contract.

## DoMINO

[`domino-surface-core`](domino-surface-core) exports the learned local encoders,
positional encoder and upstream surface solution head from prepared FP32
neighborhoods. Its two-file project uses the same importer and build commands.

### DoMINO checkpoint and SDK


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
[builder environment setup](../docs/reference.md#environment-setup).
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
pnms-model-builder import-checkpoint \
  /path/to/DoMINO.0.501.mdlus \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
```

### Surface-core inputs and scope


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

Both backends target CUDA and require byte-identical eager/C++ outputs:

| Backend | Profile | Required assets |
| --- | --- | --- |
| AOTI | `aten-boundary-exact-v3` | DoMINO tensor-only ATen sidecar; `shape_padding: false`. |
| TensorRT | `domino-surface-exact` | Linear, GELU, ScalarDiv and InverseDistanceBlend plugins. |

The AOTI sidecar is separate from the TensorRT engine. The package declares
its operator ID and ABI, and the runtime must register that matching dependency.
See [AOTI profiles](../docs/reference.md#aoti-profiles-and-options) and
[exact graph scope](../docs/reference.md#exact-graph-scope) for the supported
rewrites. A passing build covers its supplied cases and target software/GPU stack.

## Transolver surface and native E2E CLI

[`transolver-surface`](transolver-surface) exports the complete learned surface
model from imported pretrained weights, using three deterministic 75-point
cases, the `aten-boundary-exact-v2` AOTI profile and the
`layout-order-exact-v2` TensorRT profile. Copy the nine declared TensorRT plugin
libraries from an SDK built with `PNMIR_ENABLE_TENSORRT_EXACT=ON` into
`assets/tensorrt/` in your copied project. Model Builder captures and hashes
them and requires byte-identical TensorRT/Python parity before publication.
Its inputs are `fx [1,75,2]`
and `embedding [1,75,6]`; its output is standardized pressure and WSS
`[1,75,4]`.

The [C++ Transolver workflow](../workflows/transolver/README.md) provides the
complete commands to build the SDK, import the checkpoint, export this package,
and run raw VTP/STL geometry through native preprocessing, inference, and
physical-unit decoding with `physicsnemo-transolver`. It also accepts existing
compatible volume packages and fixed-shape tail packages.

## Check and build

For checkpoint-based examples, `import-checkpoint` writes `weights/config.json`,
`weights/checkpoint.pt` and `weights/import.json`, selected by the example
configuration. It verifies state preservation; it does not compare full-model
inference. The builder loads those weights strictly with `weights_only=True`.
See [checkpoint import](../docs/user-guide.md#import-a-physicsnemo-checkpoint).

The copied SDK libraries are declared build assets. Their captured bytes and
hashes participate in the project lock. Keep the SDK and library versions
matched; intentional changes after a locked build require `build --update-lock`.
Windows uses DLL filenames and asset paths from the
[Windows SDK reference](../docs/windows.md#exact-tensorrt-profiles).

After preparing or importing inputs, set the compatible SDK path. For
GeoTransolver, Transolver and DoMINO, select the SDK that supplied the libraries:

```bash
SDK_RUNTIME=/path/to/sdk/bin/physicsnemo-infer
pnms-model-builder check "$DEMO_PROJECT" --json
pnms-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_RUNTIME" --json
```

The checkout launcher `./native-inference/pnms-model-builder` can replace the
installed command. Use `--backend aoti` or `--backend tensorrt` to select one
backend, or repeat the flag for both. TensorRT requires CUDA and a compatible
TensorRT-enabled SDK.

`check` runs the eager model on every case. `build` additionally exports each
selected backend and compares actual C++ outputs with the Python references.
Every case and backend must pass before the build is accepted. Byte-exact
profiles record `require_byte_identical: true`, zero error limits and output
hashes in `checks/<backend>.json`; any byte difference fails the build.

Results go to a fresh `builds/<build-id>/`, including `model/backends/<backend>/`,
`build.json`, `checks/`, and retained source/project identities. The runtime loads
a backend directory and its named tensors. See [build outputs](../docs/reference.md#build-output),
[deployment compatibility](../docs/reference.md#package-compatibility), and
[project locks](../docs/reference.md#project-locks).

Tests live under [`tests/examples/`](../tests/examples/README.md).
