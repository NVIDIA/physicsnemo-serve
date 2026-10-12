# Model Builder reference

Use the [user guide](user-guide.md) for the first model build. This reference
covers configuration, advanced export settings, retained evidence and deployment
contracts. Commands use `pnms-model-builder`; the checkout launcher
`./native-inference/pnms-model-builder` has the same behavior.

## Project formats

A **project** selects what to build and where to run it. A **recipe** describes
how to construct, export and verify a model. Their format versions are separate:
a project with version 1 can select a recipe with version 2.

| File | Version | Purpose | Available checks |
| --- | --- | --- | --- |
| Authoring project: `model-build.json` | 2 | Select local Python sources, weights and validation cases; infer tensor contracts and generate a recipe. | `check --config-only`, eager `check`, `build`. |
| Recipe project: `model-build.json` | 1 | Select a bundled model name or explicit recipe and save execution/input settings. | `check --config-only`, `build`. |
| Explicit recipe: selected with `--recipe FILE` | 1 or 2 | Declare adapter callbacks, tensor contracts and supported backends. | `check --config-only`, `build`. |

`init` creates an authoring project and `build_adapter.py`. It needs Python 3.10+
and does not import the model, load weights or launch Docker. Existing generated
filenames are never overwritten. Missing checkpoint and environment settings are
allowed at initialization; finish them before checking or building.

The authoring callbacks are `create_model(config, assets)` and
`create_cases(config, assets)`. Construct the architecture on CPU; the builder
loads weights and moves the model and cases to the selected device. Cases are a
nonempty list of positional input tuples. Inputs and outputs must be FP32 with
positive static dimensions and the same signature across cases; multiple tensors
may have different shapes. Expected outputs come from the original eager model.

An explicit recipe declares its callback names. Recipe version 1 calls
zero-argument factories; recipe version 2 passes configuration and asset paths.
The [explicit recipe contract](#explicit-recipes) applies to recipe files, not
directly to authoring project JSON.

## Project settings

Project paths normally resolve relative to the directory containing
`model-build.json`. Explicit CLI paths resolve relative to the caller's working
directory. Unknown fields, duplicate JSON keys, nonfinite numbers and unknown
profiles are rejected before model execution.

### Authoring project fields

| Field | Meaning and default |
| --- | --- |
| `format_version` | Integer `2`. |
| `name`, `version` | Nonempty simple identifiers for the model. `init` derives a name from the directory and starts at `0.1.0`. |
| `adapter` | Contained relative Python path, usually `build_adapter.py`; always captured. |
| `source` | Relative Python files or package directories; default `[]`. `init` selects `model.py` when present. |
| `checkpoint` | Plain tensor state dictionary path; initially `null` unless supplied to `init`. |
| `checkpoint_sha256` | Optional lowercase SHA-256 pin for the selected checkpoint. |
| `config` | Inline JSON object or JSON file path; default `{}`. |
| `assets` | Named data file paths, such as validation inputs or normalization statistics; default `{}`. |
| `input_names`, `output_names` | Optional ordered, unique tensor names matching the inferred contract. |
| `aoti_profile`, `aoti_options` | AOTI compiler selection; see [profiles and options](#aoti-profiles-and-options). |
| `tensorrt_profile` | TensorRT compiler selection; see [profiles and assets](#tensorrt-profiles-and-assets). |
| `profiles`, `default_profile` | Named execution/compiler settings and optional default selection. |

The adapter and source paths must stay inside the project, without `..` or
symlink traversal. Source directories contribute `.py` files; hidden, cache,
environment and build directories are excluded. Explicit `["."]` selects Python
sources across the project. Declare data files as assets. Capture does not
install third-party packages or compile custom extensions; those dependencies
must already exist in the selected environment.

### Execution fields shared by both project formats

| Field | Meaning and default |
| --- | --- |
| `backends` | Requested backend list: `aoti`, `tensorrt`, or both. Authoring default: `["aoti"]`; a recipe supplies its own default. |
| `executor` | `local` or `container`. Omission resolves to `container`; `init` explicitly writes `local` on Windows and `container` elsewhere. |
| `device` | `cpu`, `cuda`, or `cuda:<index>`; default `cuda`. TensorRT requires CUDA. |
| `runtime` | Local `physicsnemo-infer` executable for native verification. |
| `builder_image` | Immutable container image ID or registry digest. There is no configured public default image. |
| `toolchain_lock` | Toolchain selection JSON; CLI spelling is `--lock`. This is separate from `model-build.lock.json`. |
| `required_gpu_arch` | Optional exact GPU capability such as `sm90` or `sm100`; requires CUDA. |
| `output_root` | Parent for automatically named fresh output directories; authoring default `builds`. Recipe projects may set it explicitly. |

`--output DIR` selects one specific fresh output directory. Existing output
directories cannot be reused. Selecting a different output path does not change
the model's locked input identity.

Recipe builds without `--output` or a project `output_root` use
`out/inference/<model-name>/<generated-name>/` relative to the caller's directory.

### Recipe projects

Recipe projects use integer `format_version: 1` and exactly one of `model` and
`recipe`. `model` names a bundled recipe discoverable with `list --json`;
`recipe` points to an explicit recipe file. They also accept `checkpoint`,
`checkpoint_sha256`, `config`, `assets`, the execution fields above and profiles.
Here `config` is a file path, not an inline object.

```json
{
  "format_version": 1,
  "recipe": "recipe/recipe.json",
  "config": "model-config.json",
  "checkpoint": "weights/trained.pt",
  "assets": {"normalization": "data/normalization.json"},
  "executor": "local",
  "runtime": "/opt/physicsnemo/bin/physicsnemo-infer",
  "backends": ["aoti"],
  "device": "cuda",
  "output_root": "builds"
}
```

This project may select a version-2 recipe. The recipe owns its adapter, tensor
contracts and compiler profiles. Use an authoring project when local model
modules should be captured automatically through `source`.

## Profiles and overrides

Settings resolve in this order: project → selected profile → explicit CLI
arguments. `--profile NAME` overrides `default_profile`.

| Selection | Replacement rule |
| --- | --- |
| Repeated `--backend` | Replace the entire backend list. |
| Repeated `--asset NAME=FILE` | Override individual declared assets; unknown or repeated names fail. |
| `--config FILE` | Replace the whole configuration; dictionaries are not merged. |
| `--checkpoint FILE`, `--checkpoint-sha256 DIGEST` | Select weights and an optional explicit content pin. |
| `--executor`, `--runtime`, `--builder-image`, `--lock`, `--device`, `--required-gpu-arch` | Override the corresponding execution field. |
| Profile `aoti_options` | Replace the whole inherited options object; `{}` removes inherited explicit overrides. |

Both project formats allow profiles to change execution fields. Authoring
profiles additionally allow `aoti_profile`, `aoti_options` and `tensorrt_profile`.
Profiles cannot change model identity, adapter, sources, checkpoint, configuration,
assets or tensor names/contracts. Use a separate project or explicit input
override for a different workload. Compiler profile names are configured in the
authoring JSON or recipe; there are no general CLI flags for those fields.

For example, merge this fragment into an authoring project:

```json
{
  "default_profile": "accuracy",
  "profiles": {
    "accuracy": {
      "aoti_profile": "aten-boundary-exact-v2",
      "aoti_options": {
        "max_autotune": false,
        "epilogue_fusion": false,
        "shape_padding": false,
        "coordinate_descent_tuning": false
      }
    },
    "standard": {"aoti_profile": "baseline", "aoti_options": {}},
    "autotune": {
      "aoti_profile": "baseline",
      "aoti_options": {"max_autotune": true, "epilogue_fusion": true}
    }
  }
}
```

These project profile names are user-defined; they do not guarantee accuracy or
speed. Build each into a fresh output directory, compare the same cases and
acceptance limits, then benchmark successful packages. No automatic latency or
GPU-memory comparison report is produced. See the
[AOTI comparison example](../examples/README.md#aoti-profiles).

## Project locks

The first build records selected inputs and execution identities in
`model-build.lock.json` beside the project. Entries are keyed by `build` when no
profile is selected, or `build:<profile>` otherwise. A profile literally named
`default` has its own `build:default` entry.

Later builds reject changed identities before model execution. These include
selected source/checkpoint/configuration/asset contents, backend/device selection,
compiler profile/options, and builder/runtime/toolchain identity. To acknowledge
an intentional change:

```bash
pnms-model-builder build /data/my-model --profile accuracy --update-lock --json
```

Locks record selected inputs, not successful qualification. An execution that
fails after publishing its selection can leave a lock for a subsequent attempt.
The build output retains the project, effective settings and selected lock in
`project/`.

| Command | Recipe project | Authoring project |
| --- | --- | --- |
| `check --config-only` | Validate selection against an existing lock; do not publish one. | Do not inspect or publish build locks. |
| Eager `check` | Unsupported. | Do not inspect or publish build locks. |
| `build` | Enforce or publish selected entry. | Enforce or publish selected entry, including captured sources. |

Both check modes reject `--update-lock`. Standalone `--recipe` builds do not
have a project lock; use a recipe project for repeated locked selections.

## Checkpoint import

`import-checkpoint` converts a trusted PhysicsNeMo `.mdlus` checkpoint into a
plain state dictionary and constructor JSON. Select ordinary `.pt` state
dictionaries directly as `checkpoint`; they do not need this conversion.

```bash
pnms-model-builder import-checkpoint weights/model.mdlus \
  --project /data/my-model --output imports/model --json
```

The command reads execution settings from the selected authoring project, or
the current directory's `model-build.json` when present. An unfinished adapter
or missing project checkpoint is allowed. `--executor` and `--builder-image`
override the environment. Source checkpoint and output paths are relative to
the caller; `--checkpoint-sha256 DIGEST` optionally verifies the source bytes.
The output directory must be new and its path must not traverse symlinks.

```text
imports/model/
  checkpoint.pt       # Plain tensor PyTorch state_dict
  config.json         # Constructor keyword arguments
  import.json         # Source/artifact hashes and reload verification
  execution.json      # Selected environment and importer identity
  execution.log       # Loader diagnostics
```

The result supplies `project_settings` for `checkpoint` and `config`. Copy them
into the project; import does not edit the project, adapter or lock. The report
identifies the model module/class for connecting `create_model`. Normalization
statistics and representative validation inputs remain separately declared assets.

Import reconstructs the model on CPU, loads its original weights and verifies
that the saved JSON and state dictionary reconstruct the same tensor names,
shapes, dtypes and values. It does not cast weights. Constructors must accept
JSON-compatible keyword values; nested modules or custom Python constructor
objects require an explicit adapter/configuration instead.

The environment needs compatible Torch, PhysicsNeMo and the model dependencies,
but no GPU execution or C++ SDK. The importer helper ships with the frontend and
is mounted into an existing image; rebuilding the image merely to add that
helper is unnecessary. Use `--executor local` in an existing model environment.
Conversion verification does not replace eager checks, compilation or native parity.

## Environment setup

For the basic local setup, existing-environment and Docker commands, follow the
[user guide](user-guide.md). The frontend needs Python 3.10+ and has no mandatory
ML dependencies. Local model execution needs the framework/model dependencies;
native builds additionally need compatible export/compiler and SDK libraries.

### What each command requires

| Operation | Local executor | Container executor |
| --- | --- | --- |
| `init` / `list` | No framework or native SDK. | No container is launched. |
| Authoring `check --config-only` | No framework execution or native SDK; selected files and adapter configuration must be ready. | A valid immutable image selection and Docker executable; no model execution. |
| Authoring `check` | Compatible model/framework environment; no native SDK. | Configured image containing the authoring worker and model dependencies. |
| Recipe/project-version-1 `check --config-only` | Configured executable native runtime, without running inference. | Valid image selection and Docker executable. |
| `build` | Export dependencies, compilers and native runtime for requested backends. | Image containing those dependencies and the native runtime. |
| `.mdlus` import | Compatible Torch/PhysicsNeMo/model environment; no native SDK. | Dependency image; importer helper supplied by frontend. |

Configuration checks validate selections, files and identities. They do not
establish GPU access, dependency importability or exportability. `check` executes
the eager model; only `build` compiles and verifies its native package.

### Advanced local setup

`setup-env` creates a fresh venv, installs the builder and export dependencies,
installs optional model requirements, and builds the native runtime unless one
is supplied. It supports Linux, macOS and Windows; usable backends still depend
on the host and framework. It does not install system drivers or toolchains.

| Option | Behavior |
| --- | --- |
| `--python FILE` | Interpreter used for the venv; defaults to the invoking Python. |
| Repeated `--requirements FILE` | Install additional model/framework dependency selections. |
| Repeated `--backend` | Prepare `aoti`, `tensorrt`, or both. Defaults to the `--build` project's selection, otherwise AOTI. |
| `--build PROJECT` | After setup, run the normal local build of this authoring project. Omission prepares the environment only. |
| `--sdk-source DIR` | C++ SDK source containing `CMakeLists.txt`. |
| `--builder-package PATH` | Builder source directory containing `pyproject.toml`, or a wheel to install. |
| `--runtime FILE` | Reuse a compatible `physicsnemo-infer` instead of compiling one. |
| `--tensorrt-root DIR` | Full TensorRT C++ SDK when compiling its backend. |
| `--tensorrt-cuda-major 12\|13` | TensorRT Python package CUDA major; default `13`. |

Example for both backends, after the project's adapter and inputs are ready:

```bash
pnms-model-builder setup-env "$HOME/.venvs/physicsnemo-builder-dual" \
  --requirements /data/my-model/requirements.txt \
  --backend aoti --backend tensorrt \
  --tensorrt-root /opt/TensorRT --tensorrt-cuda-major 13 \
  --build /data/my-model --json
```

SDK compilation needs CMake 3.20+ and a C++20 compiler. CUDA builds need a
compatible driver and development toolkit already installed. TensorRT needs
headers and libraries from its full C++ SDK; the pip package alone lacks the
headers. When compiling the SDK, setup reads the TensorRT headers and pins the
Python package to that version. Choose CUDA major 12 for a CUDA 12 SDK/toolkit;
avoid generic `tensorrt` or conflicting CUDA variants in requirements.

When reusing a runtime, setup does not infer its TensorRT version from headers;
pin compatible dependencies explicitly. The reused runtime must match the new
environment's framework and backend libraries. Exact TensorRT plugins and the
DoMINO sidecar require the separately configured SDK builds described in the
[SDK guide](../cpp-runtime/README.md); generic setup does not enable them.

The checkout launcher finds sibling builder/SDK sources automatically. An
installed frontend can use `--sdk-source /path/to/native-inference/cpp-runtime`
and its sibling `model-builder` source. Without that checkout, reuse a runtime
and supply the builder artifact explicitly:

```bash
pnms-model-builder setup-env "$HOME/.venvs/physicsnemo-builder" \
  --runtime /opt/physicsnemo/bin/physicsnemo-infer \
  --builder-package /downloads/pnms_model_builder-0.1.0-py3-none-any.whl
```

The environment directory must be new and outside the builder source directory.
Setup retains `.physicsnemo/setup.log`, `.physicsnemo/environment.json` and, when
built, the runtime under `.physicsnemo/runtime/`. Partial environments remain
available for diagnostics; retry setup in a fresh directory. A subsequent model
build failure does not undo a successfully prepared environment.

The result prints activation/build commands and `project_settings` containing
`executor: "local"` and the runtime path. It does not edit the project. Copy those
settings for later builds and activate the environment. Windows receives a
PowerShell wrapper that also sets native DLL search paths; use the
[Windows setup chapter](user-guide.md#windows-setup) for CUDA/Triton requirements
and the [Windows SDK reference](windows.md#exact-tensorrt-profiles) for exact builds.
Native Windows uses local execution; Linux container execution belongs in WSL
or on a Linux host.

### Container identity and dependencies

Container execution requires a digest-pinned image containing compatible model
dependencies, export tools and an installed native SDK. Registry references use
`name@sha256:...`; local image IDs use `sha256:...` and must exist on that Docker
host. Mutable tags are rejected. The bundled toolchain lock is unreleased and
selects neither a public image nor a default runtime.

The development image includes AOTI and GeoTransolver model dependencies. Its
Docker build accepts `--build-arg PNMIR_ENABLE_TENSORRT=ON` for TensorRT support.
The image must contain the authoring worker for project `check`/`build`. Model
builds reuse its installed SDK, stage selected inputs read-only and run under
the caller's UID/GID. See the [image contract](../model-builder/images/README.md)
for dependency pins and update checks. Dependency ranges alone do not qualify
exportability or a complete deployment environment.

## AOTI profiles and options

Set `aoti_profile` in an authoring project, its named profile, or an explicit
recipe. These are all accepted names:

| Profile | Behavior | Native acceptance |
| --- | --- | --- |
| `baseline` (default) | Existing AOTInductor compilation behavior. | Maximum absolute and relative L2 error each ≤ `1e-4`. |
| `aten-boundary-exact-v2` | Preserve ATen arithmetic and parameter-linear boundaries using required Torch compiler controls. | Same numerical limits. |
| `aten-boundary-exact-v3` | Also disable shape padding; supports DoMINO arithmetic rewrites through a declared native sidecar. | Byte-identical output for every case, plus tensor/finite-value checks. |

Exact profiles require compatible private Torch controls; missing controls fail
instead of silently weakening the selection. Settings are scoped to export and
restored even on failure. Builds remain serial within a process because these
compiler settings are global. `correctness_profile` artifact metadata records
the profile, applied controls, graph pass and producer compiler; completion
validation verifies the requested profile was used.

The [DoMINO example](../examples/README.md#domino) uses v3 and declares `domino_exact_ops`
as an asset pointing to the matching installed tensor-only ATen sidecar. The
builder preserves original weights and eager references while rewriting the
captured graph. Its runtime needs `PNMIR_BUILD_DOMINO_EXACT_OPS=ON` with AOTI.
Selecting an AOTI profile does not configure TensorRT.

`aoti_options` accepts only these JSON boolean controls:

| Key | Compiler control |
| --- | --- |
| `max_autotune` | Search candidate implementations for supported operations. |
| `epilogue_fusion` | Permit supported post-matmul operations to fuse into the chosen template. |
| `shape_padding` | Permit compiler padding choices for supported matrix operations. |
| `coordinate_descent_tuning` | Enable the compiler's coordinate-descent tuning control. |

Unknown keys, non-booleans and compiler `mode` strings are rejected. Explicit
`epilogue_fusion: true` requires `max_autotune: true` in the same resolved object.
Both exact profiles accept only `false` option values; use baseline to evaluate
enabled performance options with the normal parity checks.

An omitted options object, `{}`, or an omitted key retains the applicable
compiler/profile behavior; omission does not mean every optimization is disabled.
`max_autotune: false` alone does not disable fusion or padding. Missing requested
compiler controls fail the build. A profile's options object replaces the whole
inherited map: it cannot inherit `max_autotune: true` while separately supplying
only `epilogue_fusion: true`.

Nonempty options produce `artifacts[0].compiler_options` in the AOTI `model.json`,
including requested/applied maps, effective values and Torch/Torch Git/CUDA
versions. This record is separate from `correctness_profile`. It records compiler
selection, not proof that every operation used an optimization or became faster.
Options do not change weights, inputs, TensorRT compilation or acceptance limits.

## TensorRT profiles and assets

Set `tensorrt_profile` in an authoring project, its named profile, or an explicit
recipe. Baseline needs no exact plugins. Other profiles target specific supported
graphs and require all listed plugin assets from a matching exact-enabled SDK.

| Profile | Supported model/graph | Plugin count | Native acceptance | Profile metadata version |
| --- | --- | --- | --- | --- |
| `baseline` (default) | Standard TensorRT export. | 0 | Absolute and relative L2 error each ≤ `1e-4`. | — |
| `layout-order-exact` | Transolver operation/reduction order. | 8 | Same numerical limits. | 1 |
| `layout-order-exact-v2` | Transolver, also preserving deslicing layout. | 9 | Byte-identical outputs. | 2 |
| `geotransolver-exact` | GeoTransolver cached core and scalar-weighted mixing. | 9 | Byte-identical outputs. | 2 |
| `geotransolver-exact-v2` | GeoTransolver, also preserving deslicing after mixing. | 10 | Byte-identical outputs. | 3 |
| `domino-surface-exact` | Supported DoMINO surface core. | 4 | Byte-identical outputs. | 1 |

All profiles retain tensor shape/dtype/finite-value checks and original eager
references. Byte-identical profiles set `require_byte_identical: true`, zero
error limits and output hashes in `checks/tensorrt.json`; a mismatch fails the
build. AOTI remains independently configured.

The full asset matrix follows. “Transolver” means `layout-order-exact`;
“Geo” means `geotransolver-exact`. Their v2 columns refer to the corresponding
v2 profiles. Asset keys are exact; merge them with existing validation assets.

| Asset key | Transolver | Transolver v2 | Geo | Geo v2 | DoMINO |
| --- | --- | --- | --- | --- | --- |
| `tensorrt_exact_linear_plugin` | Yes | Yes | Yes | Yes | Yes |
| `tensorrt_exact_gemm_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_token_sum_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_slice_bmm_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_layer_norm_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_softmax_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_attention_plugin` | Yes | Yes | Yes | Yes | — |
| `tensorrt_exact_gelu_plugin` | Yes | Yes | Yes | Yes | Yes |
| `tensorrt_exact_deslice_bmm_plugin` | — | Yes | — | Yes | — |
| `tensorrt_exact_weighted_blend_plugin` | — | — | Yes | Yes | — |
| `tensorrt_exact_scalar_div_plugin` | — | — | — | — | Yes |
| `tensorrt_exact_inverse_distance_blend_plugin` | — | — | — | — | Yes |

Each key maps to a library path. For example:

```json
{
  "assets": {
    "tensorrt_exact_linear_plugin": "plugins/libpnmir_tensorrt_exact_linear_plugin.so"
  }
}
```

Linux filenames follow `libpnmir_tensorrt_exact_<operator>_plugin.so`; Windows
uses `pnmir_tensorrt_exact_<operator>_plugin.dll` with the same asset keys. Copy
the libraries from the selected SDK. Build that SDK with
`PNMIR_ENABLE_TENSORRT=ON` and `PNMIR_ENABLE_TENSORRT_EXACT=ON`, then select its
runtime. The [SDK guide](../cpp-runtime/README.md) documents the pinned kernel
source/header requirements; [Windows exact builds](windows.md#exact-tensorrt-profiles)
are separate from generic `setup-env`.

The builder captures/hashes these libraries through the asset pipeline. Missing
assets fail before ONNX export; profile and library identities participate in
the project lock. Changing them requires `build --update-lock`. The SDK, plugin
binaries and TensorRT/CUDA environment must match; a generic TensorRT SDK cannot
load a package requiring exact plugins.

### Exact graph scope

Transolver v2 preserves the original deslicing BMM's physical layout and batched
cuBLAS call. It requires static FP32, batch one, more than one head and a supported
exact-attention producer. Unmatched deslicing fails export. Original Transolver
projects retain their eight plugins and numerical acceptance policy.

Both Geo profiles automatically apply `FreezeScalarSigmoidGates` before ONNX
conversion. The pass evaluates captured FP32 scalar parameter sigmoids on the
reference device, preserving their rounding. Original parameters and
input-dependent/vector sigmoids remain unchanged. WeightedBlend preserves
separate rounding of both multiplications and the addition. Geo v2 additionally
preserves deslicing after the weighted attention mixture; it requires static
FP32, batch one, more than one head and two supported exact-attention producers.
Unsupported deslicing fails export. Existing Geo profiles keep their nine-plugin
contract; switching to v2 adds DesliceBmm and requires a lock update.

DoMINO substitutes supported Linear operations throughout the surface core and
their following GELUs, preserving selected scalar division and inverse-distance
blending order. Its receipt records selectors, replacement counts and plugin
identities. See the [GeoTransolver workflow](../examples/README.md#geotransolver) and
[DoMINO workflow](../examples/README.md#domino) for input scope and example asset staging.
These profiles do not promise exactness for arbitrary models or untested toolchains.

## Export hooks

An authoring adapter can select ONNX preparation through `export_options(context)`:

```python
from model_builder.export import ExportOptions
from model_builder.export.compat import NormalizeClampBounds


def export_options(context):
    if context.backend == "tensorrt":
        return ExportOptions(onnx_passes=(NormalizeClampBounds(),))
    return ExportOptions()
```

The immutable context supplies `backend` and `device`, once per backend after
original eager references have been recorded. Eager `check` does not invoke the
hook. Adapters without it keep existing behavior. An adapter named `exporter.py`
works by setting `adapter` accordingly; model/case callbacks remain unchanged.

`NormalizeClampBounds` handles FP32 clamps by converting scalar bounds into
tensors before ONNX decomposition, retaining tensor bounds and broadcasting.
It also handles mixed scalar types such as `0.5` and `5`. Non-FP32 inputs and
tensor bounds on another device are rejected. Select this pass only where needed;
it is not globally enabled.

Custom passes may live in the adapter or a captured helper such as
`export_fixes.py`; add helper files to `source`. Each callable receives a private
captured FX `GraphModule`, edits it in place and returns a non-negative rewrite
count. `onnx_passes` is a tuple; passes run in tuple order before decomposition
and must preserve computation and the input/output contract.

The builder checks the graph, records pass code hashes and rewrite counts in
`model.export-options.json` beside the ONNX graph, compiles the backend and
compares native output with the original eager references. A failing pass stops
the build. AOTI rejects ONNX passes selected for it. Hooks select graph preparation;
native CUDA implementations and TensorRT plugins remain separate runtime dependencies.

## Explicit recipes

Use an explicit recipe when maintaining adapter/callback names, tensor contracts
and backend declarations directly. Most new integrations should use an authoring
project and let the builder generate this file.

Both recipe versions declare model identity, adapter, factory/cases names,
supported/default backend and a tensor contract. Version 1 retains zero-argument
factories. Version 2 adds configuration, checkpoint and named data descriptors:

```json
{
  "config": {"path": "config.json"},
  "checkpoint": {"format": "torch-state-dict"},
  "assets": {"normalization": {"path": "normalization.json"}}
}
```

This is a fragment for a version-2 **recipe**, not a `model-build.json` project.
`config` and `checkpoint` must be objects; an omitted `path` requires the
corresponding CLI selection. Asset descriptors require default recipe-relative
paths. Descriptors may pin `sha256`. Configuration must resolve to a JSON object;
duplicate keys and nonfinite values fail.

Recipe-relative paths must remain inside the recipe directory without symlink
traversal. Selected files must be regular files, not symlinks. Explicit paths
normalize parent-directory aliases such as macOS `/tmp` and may select files
outside the recipe directory.

Override inputs with `--config FILE`, `--checkpoint FILE`, optional
`--checkpoint-sha256 DIGEST`, and repeated `--asset NAME=FILE`. Overrides replace
whole files; configuration objects are not merged. A recipe hash pins its
default file; an override receives its own recorded identity, optionally with
an explicit checkpoint pin. Unknown assets and version-1 recipe input overrides fail.

For recipe version 2, model/case callbacks each receive a separate deep copy of
the resolved configuration plus named retained asset paths. The worker builds
the model on CPU, loads the checkpoint with `weights_only=True`, checks exact
keys/shapes/dtypes and loads strictly before device transfer and eager evaluation.
Training wrappers such as `{"state_dict": ...}` need conversion to a plain tensor
state dictionary in the producer's trusted environment.

Adapters remain trusted Python code; `weights_only=True` does not sandbox them.
Callbacks must treat retained data as read-only. The worker verifies retained
source hashes after preparation and before completing a build. The retained
`model-inputs/` directory is reserved and cannot contain the adapter. Explicit
recipe source capture retains the adapter and declared data; it does not
automatically capture/import/install arbitrary Python helper modules.

### Independently shaped tensors

Recipe version 2 can declare an explicit contract for every tensor:

```json
{
  "inputs": [
    {"name": "local", "dtype": "float32", "shape": [1, 32, 6]},
    {"name": "context", "dtype": "float32", "shape": [1, 8, 128, 224]}
  ],
  "outputs": [
    {"name": "fields", "dtype": "float32", "shape": [1, 32, 4]}
  ]
}
```

Declare both arrays and omit the shared-shape shorthand fields `input_names`,
`output_names`, `dtype` and `shape`. Mixing forms is an error. Names are unique
within each array; order defines positional arguments and returned outputs.
The current contract supports only FP32 and positive static dimensions.
Shared-shape shorthand remains available in recipe versions 1 and 2, with
legacy output shapes inferred from eager references.

The worker checks inputs before eager inference and declared outputs before
export. Native metadata, package manifests and container completion must agree
with the selected contract. Authoring projects infer this contract from cases
instead of requiring the arrays in project JSON.

## Retained build inputs

Version-2 recipe builds retain exact input bytes and a portable effective recipe:

```text
source/
  recipe.json                     # Original recipe bytes
  export.py                       # Adapter at its recipe-relative path
  effective-recipe.json            # Resolved relative paths and content pins
  model-inputs/
    config.json                   # Original selected configuration bytes
    effective-config.json         # Canonical configuration identity
    checkpoint.pt                 # Plain state dictionary
    assets/<name>/<original-name> # Selected data files
```

Authoring builds generate their recipe and retain declared Python sources with
the captured inputs; source identities are verified through compilation.
Build receipts retain input origins, hashes and sizes. The deployment inventory
retains content/effective-configuration identities without requiring the original
producer paths. Container completion checks the selected identities against
returned retained files; it does not merely trust a reported success flag.

To replay retained inputs, select the effective recipe and a fresh output:

```bash
pnms-model-builder build \
  --recipe /data/previous-build/source/effective-recipe.json \
  --executor local --runtime /opt/physicsnemo/bin/physicsnemo-infer \
  --output /data/replayed-build --json
```

Select compatible device/backends/toolchain settings as for the original build.
Replay repeats export and parity; it does not promise bitwise-identical compiled
binaries across toolchains. Compiled packages do not require these retained
source/checkpoint files at inference time.

## Structured results

Use `--json` for automation. Stdout contains one result with `schema_version: 1`,
`command`, `status` and `stage`; diagnostics go to stderr. `list --json` discovers
bundled recipes. Other fields depend on the command and how far it progressed.

| Result status | Meaning |
| --- | --- |
| `initialized` | `init` created the project/adapter; includes created paths and next steps. |
| `configuration-ok` | Configuration/input-identity checks passed; no model execution or compiled package. |
| `checked` | Authoring eager check passed; includes inferred tensor contract, case count and report paths. |
| `imported` | `.mdlus` conversion/reload verification passed; includes project settings and import report. |
| `complete` | Requested operation succeeded: for example a build, environment setup or listing. Check `command`/`stage`. |
| `incomplete` | Authoring fields need attention; diagnostics identify fields. |
| `failed` | Argument/configuration, execution or verification failure. |

Successful project resolution supplies `effective_config` and the selected
profile. Execution results generally add `executor`, `device`, `backends`,
`output` and report paths. Authoring `check --config-only` returns effective
settings without creating or guaranteeing an `output` field; early failures
may lack resolved settings entirely. Build results point to the deployment
bundle and retained graphs; inspect `build.json` and verification reports for
the actual backend/case evidence.

| Exit code in JSON mode | Meaning |
| --- | --- |
| `0` | Requested operation succeeded. |
| `2` | Invalid arguments/configuration, incomplete authoring settings or a project lock mismatch. |
| `1` | Execution, environment setup, import or verification failed. |

Diagnostics include a machine-readable `code` and human-readable `message`.
Common codes are `INVALID_ARGUMENT`, `INVALID_PROJECT`, `INIT_CONFLICT`,
`PROJECT_LOCK_MISMATCH`, `MODEL_CHECK_FAILED`, `BUILD_FAILED`,
`CHECKPOINT_IMPORT_INVALID`, `CHECKPOINT_IMPORT_FAILED`,
`ENVIRONMENT_CONFIGURATION` and `MODEL_BUILD_FAILED`. Incomplete authoring
settings additionally report field-specific codes such as `AUTHORING_INPUT_REQUIRED`
or `MISSING_BUILDER_IMAGE`.

Do not assume every diagnostic has an original child `exit_code`. The recipe
build executor records one when applicable and may return that code directly
without `--json`; authoring execution normalizes failures to `1`. Consult the
returned report/log paths for details. Available execution metadata describes
the command outcome, not a uniform raw-subprocess-status contract.

`setup-env --build` includes the child build result under `build`. If environment
creation succeeds but the model build fails, the environment remains usable and
the command reports `MODEL_BUILD_FAILED`. None of these statuses publishes a
model to a registry or schedules a remote job.

## Build output

A successful **build** creates a candidate deployment bundle and verification
evidence. Initialization, import and eager checks have their own outputs and
do not create compiled backend packages.

```text
<output>/
  source/                         # Retained recipe, adapter, sources and inputs
  exported/
    aoti/program.pt2               # Exported graph before compilation
    tensorrt/model.onnx            # ONNX graph and any external weights
  model/                          # Deployment bundle
    model-release.json            # Inventory; paths relative to model/
    source-check.json             # Authoring source-integrity report
    backends/
      aoti/
        model.json
        model.pt2
      tensorrt/
        model.json
        model.plan
  checks/                         # Native case metadata and tensor bytes
  logs/                           # Compiler/native process diagnostics
  build.json                      # Build status, sources and backend results
  check.json                      # Authoring model/check report
  execution.json                  # Selected executor/toolchain and outcome
  project/                        # Project commands only
    model-build.json
    effective-config.json
    model-build.lock.json         # Selected build lock, when published
  frontend.log                    # Captured stdout diagnostics in JSON mode
```

Only selected backends and applicable reports appear. `source-check.json`
verifies captured Python source integrity; it is not a numerical accuracy report.
CPU and CUDA AOTI use the same filenames, with the artifact `target` identifying
the device. `exported/aoti/program.pt2` and `model/backends/aoti/model.pt2` are
different files: exported graph versus compiled deployment payload.

Copy `model/` to retain its inventory and qualification reports. Load one
`model/backends/<backend>/` directory with `physicsnemo-infer` or the C++ API;
that directory contains `model.json` and its payload. The runtime does not execute
`model-release.json` as a workflow. A backend package can also be copied alone
while preserving its manifest-relative artifacts.

Automated consumers should resolve `variants[backend].package` from the release
inventory relative to `model/`, or from the build receipt using `package_base`:

```json
{
  "variants": {
    "aoti": {"package": "backends/aoti"},
    "tensorrt": {"package": "backends/tensorrt"}
  }
}
```

### Lower-level ONNX import

The ONNX importer is separate from the high-level builder's AOTI/TensorRT
backends. It writes `model.json` and `model.onnx` at its package root and preserves
graph-relative external data for top-level dense initializers and
sparse-initializer values. Nested-subgraph tensor data and sparse-index external
data are not covered yet.

External-data locations must not overlap reserved root entries `model.json` or
`model.onnx`, including case variants or paths beneath those names. Rename
conflicting files/directories and update graph references before import.

## Verification and failures

`build` repeats eager preparation, exports/compiles once per requested backend
and runs every required case through the actual native runtime. It checks
completion, selected backend/device, names, shapes, dtypes, byte counts, finite
values and parity against the original eager references. Skipping a separate
`check` command does not skip build validation.

The normal native limits require both maximum absolute error and relative L2
error ≤ `1e-4`. AOTI v3 and the byte-identical TensorRT profiles above impose
the stricter equality policy. Compiler options cannot bypass these gates.

A failed requested backend or required qualification gate prevents a complete
candidate inventory; available diagnostics and partial artifacts remain for
inspection. Do not treat a partial directory or input lock as build success.
Host completion validation rehashes returned files and checks evidence coverage;
it does not rerun inference on the host.

GeoTransolver's example checks eager/native core agreement on synthetic cached
features. It does not compare a full upstream workflow against the core or native
package. Receipts are execution evidence, not signatures or scientific CFD
acceptance for arbitrary geometries. Use representative inputs and separately
qualify the application workflow. GPU tests that skip are not GPU validation.

## Package compatibility

Compiled packages run without the authoring project, Python adapter, original
checkpoint or builder tools. They still need declared input tensors and a
compatible native SDK/dependency stack. Geometry-derived features and other
preprocessing remain the application's responsibility; see the model-specific
workflow guides for input scope.

Docker is optional when the deployment host supplies those dependencies. AOTI
needs matching native Torch libraries; TensorRT and ONNX Runtime need their
native libraries. Exact profiles additionally need their matching plugins or
sidecar. SDK installation does not bundle the complete GPU dependency closure.

Moving a package does not qualify another OS, compiler/standard-library ABI,
GPU, CUDA/driver or backend version. Build and test the combinations intended
for deployment. Core static/shared relocation tests do not qualify a complete
GPU distribution, and a required GPU architecture is one check rather than a
compatibility matrix. Published bundles and promotion tooling remain
[release work](releasing.md).

### Existing package layouts

Loading follows `model.json` and relative artifact paths, not a directory suffix.
Existing `.pnmir` and `.pnm-model` directories remain loadable without conversion,
including packages previously stored under
`model/variants/<backend>/stages/predictor.pnmir/` or
`model/backends/<backend>.pnm-model/`. Preserve the internal layout of each package.

New payloads sit beside `model.json`: `model.pt2`, `model.plan`, or imported
`model.onnx`. Older manifest paths such as `artifacts/aoti-cuda/model.pt2`,
`artifacts/aoti-cpu/model.pt2`, `artifacts/tensorrt/model.plan` and
`artifacts/onnx/model.onnx` still work. The manifest's `artifacts` array does not
require an `artifacts/` directory. The receipt/inventory `variants` key remains
unchanged; new directory names do not rename the JSON schema.

The native interface uses `physicsnemo::inference`, headers under
`physicsnemo/inference/`, CMake package `PhysicsNeMoInference` and executable
`physicsnemo-infer`. Python APIs live under `model_builder.build` and
`model_builder.export`. Source-name changes do not establish binary compatibility;
rebuild C++ integrations and qualify their runtime against the packages they run.
See [SDK build and usage](../cpp-runtime/README.md).
