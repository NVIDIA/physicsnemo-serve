# GeoTransolver cached-core example

The [GeoTransolver example](../examples/README.md#geotransolver) is a normal
format-2 authoring project with exactly two files: `model-build.json` and
`adapter.py`. Copy the project, import a trusted checkpoint with the generic
`import-checkpoint` command, and supply the SDK's exact TensorRT plugin libraries
as declared assets before `check` and `build`. The adapter generates its small
synthetic cases directly; no model-specific preparation command is needed.

## Checkpoint and environment

Use a compatible PhysicsNeMo 2.1.1, Torch, Warp and CUDA environment with
Model Builder and a matching C++ Inference SDK. TensorRT also requires its
Python/native libraries and ONNX/ONNXScript. See
[builder environments](../model-builder/README.md#builder-environment).
The default project requires an AOTI-enabled SDK built with
`PNMIR_ENABLE_TENSORRT=ON` and `PNMIR_ENABLE_TENSORRT_EXACT=ON`. This installs
all nine matching exact plugin libraries. Follow the
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
physicsnemo-model-builder import-checkpoint \
  /path/to/GeoTransolver.0.501.mdlus \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
physicsnemo-model-builder check "$DEMO_PROJECT" --json
physicsnemo-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_ROOT/bin/physicsnemo-infer" --json
```

The importer writes constructor settings to `weights/config.json`, the plain
tensor state dictionary to `weights/checkpoint.pt`, and import verification to
`weights/import.json`. The project already selects these paths. Import checks
state preservation; it does not run a full-model inference comparison. The
builder loads the generated weights with `weights_only=True` and strict state
loading. See [checkpoint import](add-model.md#import-a-physicsnemo-checkpoint).

The project declares the copied libraries under `assets/tensorrt/`, with keys
such as `tensorrt_exact_linear_plugin` and `tensorrt_exact_weighted_blend_plugin`.
They are build inputs supplied from the SDK installation. Model Builder captures
their bytes, records their hashes, and includes their identities in the project
lock. Replacing a library after a locked build requires `build --update-lock`.
The repository example still contains only its two source files.

## Cached-core inputs and scope

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

## Build checks and outputs

`check` executes the core eagerly on its cases. `build` repeats eager checks,
exports each requested backend and compares its actual C++ outputs against
core Python outputs. Every case and requested backend must pass the existing
checks before a complete candidate is written. TensorRT's `geotransolver-exact`
profile additionally requires identical output bytes for every case. This is core/native
parity only; there is no full-geometry or full-model/native gate.

The project defaults to both AOTI and TensorRT on CUDA:

| Backend | Profile | Behavior |
| --- | --- | --- |
| AOTI | `aten-boundary-exact-v2` | Uses the existing exactness-preserving compiler settings and AOTI parity checks. |
| TensorRT | `geotransolver-exact` | Uses nine exact plugins and requires byte-identical eager/C++ outputs. |

For TensorRT, the adapter's `NormalizeClampBounds` hook remains in use.
The selected profile automatically adds `FreezeScalarSigmoidGates`: it evaluates
captured FP32 scalar parameter sigmoids with PyTorch on the reference device
before ONNX conversion. This retains the eager result's rounding for GALE mixing
gates. It leaves the original checkpoint, eager model, input-dependent sigmoids
and vector gates unchanged. The WeightedBlend plugin then preserves separately
rounded multiplies and addition for the gated mixture.

`checks/tensorrt.json` records `require_byte_identical: true`, zero error limits,
and native/reference output hashes. A byte difference fails the build even when
it would fall within the normal numerical tolerance. The separate Transolver
`layout-order-exact` profile continues to use its original eight plugins and
existing parity limits. See [TensorRT profiles](add-model.md#tensorrt-profiles).

```text
geotransolver-demo/
  model-build.json
  adapter.py
  weights/{config.json,checkpoint.pt,import.json}
  assets/tensorrt/libpnmir_tensorrt_exact_*_plugin.so
  builds/<build-id>/
    source/                    # captured adapter and selected inputs
    exported/                  # graphs before backend compilation
    checks/{aoti,tensorrt}.json # core Python/native parity reports
    build.json
    project/                   # effective settings and lock snapshot
    model/
      model-release.json
      backends/aoti/{model.json,model.pt2}
      backends/tensorrt/{model.json,model.plan}
```

Only selected backend directories appear. The native runtime loads the backend
package and declared input tensors; it does not need the original checkpoint or
Python adapter. Deployment still requires compatible native dependencies.
See [package compatibility](packages.md#deployment-and-compatibility).
