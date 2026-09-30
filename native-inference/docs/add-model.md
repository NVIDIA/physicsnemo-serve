# Add your existing Python model

Start with your model implementation and trained weights. `init` creates two
files beside them; you connect your existing model and validation inputs, then
run `check` and `build`. You do not need to write a recipe or a Dockerfile for
each model.

`init`, format-2 authoring projects and eager `check` are implemented in this
branch. Published builder images, prebuilt SDK releases and artifact publication
remain release work; see [current packaging support](releasing.md).

## 1. Initialize your existing model directory

From the PhysicsNeMo Serve checkout root, make the checkout launcher available
in this shell. An installed `physicsnemo-model-builder` works the same way.

```bash
export PATH="$PWD/native-inference:$PATH"
physicsnemo-model-builder init /data/my-model
cd /data/my-model
```

For an existing `model.py` and `weights.pt`, the result is:

```text
my-model/
├── model.py                # Existing Python implementation
├── weights.pt              # Existing trained weights
├── model-build.json        # Created: selected inputs and build settings
└── build_adapter.py        # Created: two functions to connect your model
```

`init` needs only Python 3.10+. It does not load weights, import your model or
start Docker. It refuses to overwrite either generated filename and preserves
the rest of your directory. Missing weights and an unconfigured environment
are allowed during initialization.

There is no required checkpoint argument. Set `checkpoint` in the generated
configuration when ready. Optional `--checkpoint FILE` and repeated
`--source PATH` arguments can fill those settings during initialization.

## 2. Select your inputs and environment

Edit `model-build.json`. For a model constructed as `MyModel(input_dim=6,
output_dim=4)`, using an existing builder image, a complete example is:

```json
{
  "format_version": 2,
  "name": "my-model",
  "version": "0.1.0",
  "adapter": "build_adapter.py",
  "source": ["model.py"],
  "checkpoint": "weights.pt",
  "config": {"input_dim": 6, "output_dim": 4},
  "assets": {"validation_inputs": "validation-inputs.pt"},
  "backends": ["aoti"],
  "executor": "container",
  "builder_image": "sha256:<actual-builder-image-id>",
  "device": "cuda",
  "output_root": "builds"
}
```

Replace the model settings and image placeholder with your actual values.
Paths are relative to `model-build.json`.

| Setting | What to supply |
| --- | --- |
| `checkpoint` | Your trained plain tensor PyTorch `state_dict` file. |
| `config` | Constructor settings as an inline object, or a path such as `"config.json"`. |
| `source` | Python files or package directories imported by your adapter. For a package, use e.g. `["my_network"]`. Explicit `["."]` selects Python sources across the project. |
| `assets` | Optional named data files, such as saved validation inputs or normalization values. |
| `backends` | Start with `["aoti"]`; request `["aoti", "tensorrt"]` only with a compatible environment and model. |

`init` selects `model.py` when it exists, otherwise `source` starts empty. The
adapter itself is always captured. Source directories contribute `.py` files;
hidden, cache and build directories are excluded. Declare data files as assets.
Python packages from outside your project must already be installed in the
selected environment. Source capture does not install dependencies or compile
custom Python/native extensions.

Choose the environment once:

| Environment | Configuration and requirements |
| --- | --- |
| Container | Keep `executor: "container"` and set an immutable `builder_image`. For GPU execution, use a compatible Linux x86_64 GPU host with Docker and NVIDIA GPU access. Build the [development image](../model-builder/README.md#builder-environment) once or select your team's image containing the model dependencies. No public default image is configured yet. |
| New Python environment | Run `setup-env` to create a reusable venv, install the builder and model requirements, and build its matching native runtime. Copy the printed `executor: "local"` and `runtime` settings into the project. See the command below. |
| Existing Python environment | Set `executor: "local"`. Use the Python environment containing Torch and your model dependencies. `check` needs no native SDK. `build` also needs the export/compiler dependencies and `runtime: "/path/to/sdk/bin/physicsnemo-infer"`. Set `device: "cpu"` only if your model and selected backend support CPU execution. |

The selected environment must contain this branch's Model Builder code. An
older image will not contain the new authoring worker. Refer to the
[environment guide](../model-builder/README.md#builder-environment) for setup
and [SDK instructions](../cpp-runtime/README.md) for native builds.

### Set up Python without Docker

From the checkout root, create a fresh environment for your model:

```bash
./native-inference/physicsnemo-model-builder setup-env \
  "$HOME/.venvs/my-model-builder" \
  --requirements /data/my-model/requirements.txt --json
```

Put compatible model and framework dependency versions in `requirements.txt`;
omit that flag when no additional packages are needed. Setup installs Model
Builder with AOTI export dependencies and builds its matching C++ runtime once.
It prints the activation command and the `executor` and `runtime` values to
copy into `model-build.json`; it does not edit the project. Activate the venv
before importing a checkpoint, checking the model or building it.

Setup needs Python 3.10+, a C++ compiler and CMake 3.20+. CUDA builds also need
the host's compatible driver and CUDA development toolkit. For TensorRT, add
`--backend aoti --backend tensorrt --tensorrt-root /opt/TensorRT`. Setup pins the
Python package to the SDK version; use `--tensorrt-cuda-major 12` for a CUDA 12
SDK/toolkit (the default is 13). Avoid conflicting TensorRT packages in model
requirements. The full
TensorRT C++ SDK supplies headers and libraries that pip alone does not provide.
See [local environment setup](../model-builder/README.md#create-a-local-environment-and-build)
for interpreter selection, installed-wheel usage and reusing an existing SDK.

When your checkpoint, configuration and adapter are ready, append
`--build /data/my-model` to perform setup and the model build in one command.
This option uses a format-2 authoring project and selects its requested backends
unless overridden with `--backend`.

If you already have a compatible Python environment, activate it and install
the builder from the checkout root:

```bash
python -m pip install './native-inference/model-builder[export]'
physicsnemo-model-builder check /data/my-model --executor local --json
physicsnemo-model-builder build /data/my-model \
  --executor local --runtime /path/to/sdk/bin/physicsnemo-infer --json
```

Use `[tensorrt]` instead of `[export]` for TensorRT. Your model dependencies and
the native runtime must match this environment. Docker remains an independent
option selected with `executor: "container"` and a builder image.

### Import a PhysicsNeMo checkpoint

If your trained model is a trusted PhysicsNeMo `.mdlus` checkpoint, convert it
once instead of writing a conversion script. From your initialized model project:

```bash
physicsnemo-model-builder import-checkpoint \
  weights/Transolver.0.501.mdlus \
  --output imports/transolver --json
```

The command reads `executor` and `builder_image` from the current
`model-build.json`. An unfinished adapter and missing project checkpoint are
allowed. Use `--project /path/to/project` to select another project, or
`--builder-image sha256:<image-id>` to override the image. Checkpoint and output
paths are relative to the calling directory. `--checkpoint-sha256 <digest>` can
verify a downloaded checkpoint before loading it.

The output directory must be new. It contains:

```text
imports/transolver/
├── checkpoint.pt     # Plain tensor PyTorch state_dict
├── config.json       # Constructor keyword arguments
├── import.json       # Source/artifact hashes and reload verification
├── execution.json    # Selected environment and importer identity
└── execution.log     # Checkpoint loader diagnostics
```

The command prints `project_settings` but does **not** edit `model-build.json`,
the adapter or the project lock. Copy the returned values into the existing
fields yourself:

```json
"checkpoint": "imports/transolver/checkpoint.pt",
"config": "imports/transolver/config.json"
```

Import reconstructs the model on CPU, loads its original weights, then verifies
that the saved JSON and plain state dictionary reconstruct the same tensor
names, shapes, dtypes and values. It preserves weights without precision casts.
This verifies checkpoint conversion; validation inputs, backend compilation and
native parity remain separate steps. The report identifies the model's Python
module and class for connecting `create_model` below. Your matching normalization
statistics and real validation inputs remain separately declared assets.

The selected environment must contain compatible Torch, PhysicsNeMo and the
checkpoint model's dependencies. The importer helper ships with the frontend and
runs inside the existing image, so this command does not require rebuilding the
image to add the helper. It needs neither GPU execution nor the C++ SDK.
For an existing Python environment instead of Docker:

```bash
physicsnemo-model-builder import-checkpoint model.mdlus \
  --executor local --output imports/my-model --json
```

Constructor arguments must be JSON-compatible keyword values. Checkpoints whose
constructors contain nested modules or custom Python objects need an explicit
adapter/configuration; import reports that limitation instead of generating an
unusable configuration. Plain PyTorch `.pt` state dictionaries can be selected
directly as `checkpoint` and do not need this import step.

## 3. Connect your model and validation inputs

Replace the two placeholders in `build_adapter.py`. For the example above:

```python
def create_model(config, assets):
    from model import MyModel

    return MyModel(**config).eval()


def create_cases(config, assets):
    import torch

    return torch.load(
        assets["validation_inputs"],
        map_location="cpu",
        weights_only=True,
    )
```

Change the import and constructor to match your model. Construct the
architecture on CPU; the builder strictly loads the selected checkpoint and
places the model and cases on the selected device.

`create_cases` returns a nonempty list of positional input tuples. For
`model(features, context)`, return `[(features, context), ...]`. You can call
your existing validation input loader here instead of reading a saved file.

To use the saved-file example, save real preprocessed inputs from your
validation code:

```python
import torch

# first_input and second_input are existing preprocessed validation tensors.
# This example is for a model taking one positional tensor input.
torch.save(
    [(first_input.detach().cpu(),), (second_input.detach().cpu(),)],
    "/data/my-model/validation-inputs.pt",
)
```

The builder computes expected outputs using your model and weights. You do
not supply expected answers or manually transcribe tensor shapes into a recipe.
Use representative cases; parity on these cases does not establish scientific
accuracy on your full CFD workload.

### Optional export compatibility

Format-2 projects can select `"aoti_profile": "aten-boundary-exact-v2"` in
`model-build.json` when AOTI must preserve PyTorch arithmetic and linear-operation
boundaries. This uses the existing compiler profile and keeps every parity limit
unchanged. The selection is retained in the generated recipe and project lock;
changing it requires `build --update-lock`. Omit the field or select `"baseline"`
for default AOTI compilation. The exact profile requires compatible Torch compiler
controls and does not change TensorRT compilation.

The [DoMINO surface example](domino-workflow.md) selects
`"aoti_profile": "aten-boundary-exact-v3"`. It additionally disables shape
padding and declares `domino_exact_ops` as the matching installed tensor-only
ATen sidecar library. The builder applies its arithmetic rewrite to the captured
graph while retaining the original eager references and weights. AOTI v3
requires byte-identical native outputs for every case.

Use a separate `aoti_options` object to request supported AOTInductor compiler
controls. For example, to try autotuning with epilogue fusion:

```json
{
  "aoti_profile": "baseline",
  "aoti_options": {
    "max_autotune": true,
    "epilogue_fusion": true
  }
}
```

| Option | Compiler control |
| --- | --- |
| `max_autotune` | Search candidate implementations for supported operations during compilation. |
| `epilogue_fusion` | Permit supported operations after a matrix multiplication to fuse into its selected template. |
| `shape_padding` | Permit compiler padding choices for supported matrix operations. |
| `coordinate_descent_tuning` | Enable the compiler's coordinate-descent tuning control. |

Values must be JSON booleans; only these four keys are accepted. There are no
`mode` strings or arbitrary compiler settings. Explicit
`"epilogue_fusion": true` requires explicit `"max_autotune": true` in the same
resolved options object. `aten-boundary-exact-v2` and `aten-boundary-exact-v3` accept only `false` option
values; requesting any of these optimizations with `true` conflicts with that
profile's qualification policy. Use `baseline` to evaluate them with the
existing parity checks.

Omitting `aoti_options`, or selecting `{}`, preserves the existing compiler
behavior. An omitted key retains the installed Torch compiler's default;
`max_autotune: false` alone does not disable fusion or padding. The selected
Torch version must expose each requested control, otherwise compilation fails
with an unsupported-setting error. Options are scoped to the build and do not
change TensorRT compilation, model weights, inputs, or acceptance limits.

For nonempty selections, the AOTI artifact's `compiler_options` records the
`requested` and `applied` option maps, the `effective` values of the four
supported controls, and Torch, Torch Git, and CUDA versions. The named
`correctness_profile` remains a separate record. These are compiler settings;
they do not establish that a particular optimization was selected for every
operation or that it improved runtime performance.

Format-2 named project profiles can select both `aoti_profile` and
`aoti_options`. See [profile examples and lock behavior](model-build-projects.md#aoti-options-in-format-2-profiles)
and the [runnable same-model comparison](../examples/README.md#aoti-profiles).
Use the same representative cases across selections, retain each build's
native parity report, and benchmark only successful packages.

#### TensorRT profiles

For the supported Transolver surface graph, select
`"tensorrt_profile": "layout-order-exact"` to preserve the PyTorch operation and
reduction order with native TensorRT plugins. This profile requires all eight
plugin libraries as explicit project assets:

```json
{
  "tensorrt_profile": "layout-order-exact",
  "assets": {
    "tensorrt_exact_linear_plugin": "plugins/libpnmir_tensorrt_exact_linear_plugin.so",
    "tensorrt_exact_gemm_plugin": "plugins/libpnmir_tensorrt_exact_gemm_plugin.so",
    "tensorrt_exact_token_sum_plugin": "plugins/libpnmir_tensorrt_exact_token_sum_plugin.so",
    "tensorrt_exact_slice_bmm_plugin": "plugins/libpnmir_tensorrt_exact_slice_bmm_plugin.so",
    "tensorrt_exact_layer_norm_plugin": "plugins/libpnmir_tensorrt_exact_layer_norm_plugin.so",
    "tensorrt_exact_softmax_plugin": "plugins/libpnmir_tensorrt_exact_softmax_plugin.so",
    "tensorrt_exact_attention_plugin": "plugins/libpnmir_tensorrt_exact_attention_plugin.so",
    "tensorrt_exact_gelu_plugin": "plugins/libpnmir_tensorrt_exact_gelu_plugin.so"
  }
}
```

Merge these entries with existing validation assets. Build the matching C++
Inference SDK with `PNMIR_ENABLE_TENSORRT=ON` and
`PNMIR_ENABLE_TENSORRT_EXACT=ON`, and select its `physicsnemo-infer` runtime.
The builder captures and hashes each plugin through the existing asset pipeline;
the profile and library identities participate in the project lock. Changing
them requires `build --update-lock`. Missing plugin assets fail before ONNX
export. Plugin binaries and the runtime must match the build's TensorRT/CUDA
environment; an ordinary TensorRT-only SDK cannot load this exact package.

For native Windows, follow the [manual exact SDK build](windows.md#exact-tensorrt-profiles).
Use the same asset keys with the installed
`pnmir_tensorrt_exact_<operator>_plugin.dll` filenames instead of the Linux `.so`
paths. Windows `setup-env` prepares the generic TensorRT runtime, so select the
separately built exact-enabled runtime explicitly.

The `layout-order-exact` profile leaves the eager references, validation inputs
and acceptance limits unchanged. It targets the supported Transolver graph, rather
than promising byte equality for arbitrary models or untested environments.
Native parity remains required for every supplied case. AOTI compilation remains
controlled by `aoti_profile`; selecting this TensorRT profile does not alter it.
Omit `tensorrt_profile` or select `"baseline"` for standard TensorRT compilation.

For byte-identical Transolver qualification, select
`"tensorrt_profile": "layout-order-exact-v2"` and add a ninth asset:
`"tensorrt_exact_deslice_bmm_plugin": "plugins/libpnmir_tensorrt_exact_deslice_bmm_plugin.so"`
(use `pnmir_tensorrt_exact_deslice_bmm_plugin.dll` on Windows). This version also
preserves the original deslicing BMM's physical layout and batched cuBLAS call;
a generic TensorRT MatMul can meet numeric tolerances while producing different
bytes. The rewrite requires static FP32, batch one, more than one head, and a
supported exact-attention producer. Unmatched deslicing fails export.

V2 keeps the original Python model and references. It requires identical native
output bytes for every TensorRT case, in addition to shape, dtype and finite
value checks; `checks/tensorrt.json` records `require_byte_identical: true` and
output hashes. A mismatch fails the build. Update the project lock when selecting
v2 and its plugin. Existing `layout-order-exact` projects keep their eight assets
and numeric acceptance policy; AOTI remains independently configured.

For the supported GeoTransolver cached core, select
`"tensorrt_profile": "geotransolver-exact-v2"`. It requires the same eight assets
plus `tensorrt_exact_weighted_blend_plugin` and
`tensorrt_exact_deslice_bmm_plugin`, pointing to the matching WeightedBlend and
DesliceBmm libraries. The
[two-file GeoTransolver project](../examples/geotransolver-surface-core/model-build.json)
already declares all ten under `assets/tensorrt/`. Copy those libraries from
the matching exact-enabled SDK as shown in the
[example commands](../examples/README.md#geotransolver). Model Builder captures
and hashes these assets; their identities and the profile participate in the
normal project lock.

This profile automatically applies `FreezeScalarSigmoidGates` before ONNX
conversion. It evaluates captured FP32 scalar parameter sigmoids with PyTorch
on the reference device, preserving the rounding used by GeoTransolver's mixing
gates. Original model/checkpoint parameters and input-dependent or vector
sigmoids remain unchanged. WeightedBlend preserves separate rounding of the
two multiplications and the addition when those scalar gates mix tensors.
V2 also preserves the deslicing BMM's physical layout after the weighted
attention mixture. It requires static FP32, batch one, more than one head,
and two supported exact-attention producers; unsupported deslicing fails export.

TensorRT builds with either GeoTransolver exact profile require identical native/reference
output bytes for every case, in addition to the usual shape, dtype and finite
value checks. `checks/tensorrt.json` records `require_byte_identical: true`,
zero error limits, and the output hashes. A mismatch fails the build. This
requirement applies to the selected GeoTransolver TensorRT profile; AOTI and
`layout-order-exact` retain their existing checks. Existing `geotransolver-exact`
projects retain their nine-plugin contract and profile metadata version 2.
The new `geotransolver-exact-v2` records metadata version 3. Changing a project
to v2 requires adding DesliceBmm and updating its lock with `build --update-lock`.

For the supported DoMINO surface core, select
`"tensorrt_profile": "domino-surface-exact"`. It requires four declared
assets: `tensorrt_exact_linear_plugin`, `tensorrt_exact_gelu_plugin`,
`tensorrt_exact_scalar_div_plugin` and
`tensorrt_exact_inverse_distance_blend_plugin`. The
[two-file example](../examples/domino-surface-core/model-build.json) supplies
the paths; copy the corresponding libraries from the matching SDK using the
[example commands](../examples/README.md#domino).

This profile applies exact substitutions to supported Linear operations
throughout the surface core and their following GELUs. It preserves the selected scalar division and
inverse-distance blending order. Its receipt records these selectors,
replacement counts and plugin identities. Every native output must be
byte-identical to the original eager core. See
[DoMINO input scope and exact checks](domino-workflow.md).

Keep model construction and validation cases in your adapter. When a model
needs an ONNX compatibility fix, select it with one optional hook:

```python
from pnmir_export import ExportOptions
from pnmir_export.compat import NormalizeClampBounds


def export_options(context):
    if context.backend == "tensorrt":
        return ExportOptions(onnx_passes=(NormalizeClampBounds(),))
    return ExportOptions()
```

You can name the adapter `exporter.py` by setting `"adapter": "exporter.py"`
in `model-build.json`. The existing `create_model` and `create_cases` functions
stay the same. Adapters without an export hook keep their existing behavior.
The hook receives immutable `backend` and `device` metadata once per backend,
after original eager references have been recorded. `check` does not invoke it.

`NormalizeClampBounds` is reusable across models using FP32 clamps. It converts
scalar bounds into tensors before ONNX decomposition, preserving tensor bounds
and broadcasting. It also handles the mixed `0.5`/`5` scalar bounds that the
pinned GeoTransolver exporter otherwise turns into an invalid scalar/tensor
pair. Non-FP32 inputs or tensor bounds on another device are rejected explicitly.
Select this pass only when needed; it is not applied globally.

For a model-specific graph fix, define a callable in `exporter.py` or in a
helper such as `export_fixes.py`, and select it in `onnx_passes`. Add helper files
to `source` so they are retained and checked with your model. Each pass receives
a private captured FX `GraphModule`, edits its graph in place, and returns a
non-negative rewrite count. Passes run in tuple order before ONNX decomposition
and must preserve the input/output contract and computation. The builder checks
the graph, records pass code hashes and rewrite counts beside the ONNX graph in
`model.export-options.json`, compiles the backend and verifies all native outputs
against the original eager references. A failing pass stops the build.

This hook selects graph preparation; CUDA operator implementations and TensorRT
plugins remain separate native dependencies installed in the builder/runtime
environment. An ONNX graph pass alone does not implement or package them.
AOTI retains its existing compilation path and rejects ONNX passes selected for
that backend.

## 4. Check, then build

From your model directory:

```bash
physicsnemo-model-builder check . --json
physicsnemo-model-builder build . --output builds/first --json
```

`check` runs your model in the selected environment, validates strict checkpoint
loading, executes all supplied cases and infers their tensor contracts. It
creates a fresh evidence directory under `output_root`; its JSON result names
the output and `check.json` report. It does not compile a backend, run C++ or
publish a project lock.

`build` repeats those checks automatically, generates an internal recipe,
exports and compiles every requested backend, then runs the compiled package
through the C++ runtime and compares every case with its Python reference.
Skipping the separate `check` command does not skip validation. A build fails
if any requested export, compilation or required native parity check fails.

The output directory must be new. After a successful first build, deploy:

```text
builds/first/model/
├── model-release.json
├── source-check.json
└── backends/
    └── aoti/
        ├── model.json
        └── model.pt2
```

The surrounding `builds/first/` directory contains retained source, graphs,
checks, logs and project provenance. Those are build evidence; the `model/`
directory is the deployable bundle. `source-check.json` records verification
that the captured Python sources remained unchanged through compilation.
Its native dependencies and GPU/toolchain
compatibility still matter; see [package compatibility](packages.md#deployment-and-compatibility).

For subsequent builds, let the builder select a fresh output directory:

```bash
physicsnemo-model-builder build . --json
```

The first build records input and environment identities in
`model-build.lock.json`. After intentionally changing model code, weights,
configuration, assets or the selected toolchain, update that selection:

```bash
physicsnemo-model-builder build . --update-lock --json
```

`doctor . --json` provides framework-free configuration and input-identity
checks. Use `check` to execute the model; a successful `doctor` result does not
prove that model imports, GPU execution or native dependencies work.

## Current model contract

- Checkpoints must be plain tensor state dictionaries. Convert training
  wrappers such as `{"state_dict": ...}` in your trusted training environment.
- Inputs and outputs use static, positive shapes and FP32 tensors. Every case
  must have the same signature; independently shaped multiple inputs and
  outputs are supported. Optional `input_names` and `output_names` lists give
  the inferred tensors meaningful names.
- The model must support the selected exporter/backend. Eager `check` success
  does not guarantee exportability or native parity.
- The adapter supplies your model's preprocessing results. Full raw-mesh CFD
  preprocessing, automatic weight downloads, dynamic-shape authoring and
  package publication are not provided by `init`.

Format-1 projects selecting an existing named model or recipe remain supported
without changes. Use format 2 for this authoring workflow: it generates the
recipe internally from your two callbacks. See [project settings and profiles](model-build-projects.md)
for repeatable execution settings and [the lower-level model-input contract](model-inputs.md)
when maintaining an explicit recipe.
