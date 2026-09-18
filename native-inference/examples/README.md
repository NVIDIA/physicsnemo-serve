# Model examples

Each example folder contains exactly two files: `model-build.json` and
`adapter.py`. Copy or stage an example in a fresh directory, then import or
generate its inputs as shown below. Build the project through the normal
Model Builder interface.

Use a compatible Python environment with Model Builder, the model dependencies
and a matching C++ Inference SDK. Run the commands below from the repository
root. Project and checkpoint-import output directories must be new.

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
[compiler options and locks](../docs/model-build-projects.md#aoti-options-in-format-2-profiles).

## GeoTransolver

Builds the surface cached core using pretrained weights and three synthetic
core-input cases. Use a compatible environment with PhysicsNeMo 2.1.1, Torch,
Warp, CUDA and matching AOTI/TensorRT libraries. The SDK must include AOTI and
the nine exact TensorRT plugins built with `PNMIR_ENABLE_TENSORRT_EXACT=ON`;
see the [SDK build instructions](../cpp-runtime/README.md#exact-tensorrt-operators-for-transolver-and-geotransolver).
Download the trusted
`.mdlus` checkpoint separately; see
[checkpoint and environment details](../docs/geotransolver-workflow.md#checkpoint-and-environment).

```bash
DEMO_PROJECT=/tmp/geotransolver-demo
SDK_ROOT=/path/to/matching/sdk
cp -R native-inference/examples/geotransolver-surface-core "$DEMO_PROJECT"
mkdir -p "$DEMO_PROJECT/assets/tensorrt"
cp "$SDK_ROOT"/lib/libpnmir_tensorrt_exact_*_plugin.so "$DEMO_PROJECT/assets/tensorrt/"
physicsnemo-model-builder import-checkpoint \
  /path/to/GeoTransolver.0.501.mdlus \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
```

The importer writes `weights/config.json`, `weights/checkpoint.pt` and
`weights/import.json`. The project already selects those paths and defaults to
AOTI and TensorRT on CUDA. Its adapter creates deterministic synthetic features
for 32 points; there are no fixture files or a model-specific preparation tool.
The copied plugin libraries are declared assets: Model Builder captures and
hashes them for the build and project lock.

The project selects `aten-boundary-exact-v2` for AOTI and `geotransolver-exact`
for TensorRT. The latter automatically freezes constant scalar sigmoid gates
using PyTorch's result and requires byte-identical C++ outputs for every case.
See [exact profile behavior](../docs/geotransolver-workflow.md#build-checks-and-outputs).

The example checks cached-core Python/C++ parity. It does not validate full
geometry preprocessing or physical CFD accuracy. Full GeoTransolver uses a
Warp radius-search operator that the native exporter does not support. See
[tensor inputs and scope](../docs/geotransolver-workflow.md#cached-core-inputs-and-scope).

## DoMINO

Builds the surface local encoders, positional encoder and solution head from
prepared FP32 neighborhoods. The adapter supplies three deterministic synthetic
cases with 32 points. Geometry preprocessing, spatial queries, neighbor gathering
and the global-grid projection are outside this example.

Use a compatible PhysicsNeMo 2.1.1/CUDA environment and an SDK built with
`PNMIR_ENABLE_AOTI=ON`, `PNMIR_ENABLE_TENSORRT=ON`,
`PNMIR_ENABLE_TENSORRT_EXACT=ON` and `PNMIR_BUILD_DOMINO_EXACT_OPS=ON`.
Download the trusted surface `.mdlus` checkpoint separately; see the
[checkpoint and SDK details](../docs/domino-workflow.md#checkpoint-and-environment).

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
```

The project selects `aten-boundary-exact-v3` for AOTI and
`domino-surface-exact` for TensorRT. Both require byte-identical C++ outputs
against the original eager core for every case. The five copied libraries are
declared, hashed build inputs. The repository template remains two files;
weights, libraries and outputs live in your copied project. See
[input meanings, exact profiles and outputs](../docs/domino-workflow.md).

## Transolver surface and native E2E CLI

[`transolver-surface`](transolver-surface) exports the complete learned surface
model from imported pretrained weights, using three deterministic 75-point
cases and the `aten-boundary-exact-v2` AOTI profile. Its inputs are `fx [1,75,2]`
and `embedding [1,75,6]`; its output is standardized pressure and WSS
`[1,75,4]`.

The [C++ Transolver workflow](../workflows/transolver/README.md) provides the
complete commands to build the SDK, import the checkpoint, export this package,
and run raw VTP/STL geometry through native preprocessing, inference, and
physical-unit decoding with `physicsnemo-transolver`. It also accepts existing
compatible volume packages and fixed-shape tail packages.

## Check and build

After preparing or importing inputs above, set the compatible SDK path.
For GeoTransolver and DoMINO, use the same SDK that supplied the copied libraries.

```bash
SDK_RUNTIME=/path/to/sdk/bin/physicsnemo-infer
physicsnemo-model-builder check "$DEMO_PROJECT" --json
physicsnemo-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_RUNTIME" --json
```

The checkout launcher `./native-inference/physicsnemo-model-builder` can replace
the installed command. Use `--backend aoti` or `--backend tensorrt` to select one
backend, or repeat the flag for both. TensorRT requires `--device cuda` and a
compatible TensorRT-enabled SDK.
Builds create fresh directories under the project's `builds/`; each
contains `model/backends/<backend>/`, `build.json` and parity reports in `checks/`.

For repeatable builds and intentional changes, see
[projects, profiles and locks](../docs/model-build-projects.md). Tests live under [`tests/examples/`](../tests/examples/README.md).
