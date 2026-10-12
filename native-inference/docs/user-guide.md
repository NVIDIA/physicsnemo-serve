# PhysicsNeMo Model Builder user guide

Turn your Python model and trained weights into a package that runs through the
**PhysicsNeMo C++ Inference SDK**. You provide the model and representative
inputs; Model Builder loads the weights, compiles the selected backend, and
compares native outputs with your Python model.

Choose [Linux setup](#linux-setup) or [Windows setup](#windows-setup), then
continue with [Create your project](#create-your-project). The [reference](reference.md)
covers configuration fields, compiler profiles, explicit recipes and file formats.
For prepared starting points, see the [model examples](../examples/README.md).

## Before you start

Have these inputs ready:

- Your model's Python implementation and a plain tensor PyTorch `state_dict`.
  For a PhysicsNeMo `.mdlus` checkpoint, use the [import step](#import-a-physicsnemo-checkpoint).
- Constructor settings and any required data files, such as normalization values.
- Representative, preprocessed validation inputs. Inputs and outputs must be
  FP32 tensors with positive, static shapes; every case uses the same signature.
- The Python dependencies needed to construct and run your model.

The example below assumes `model.py` defines `MyModel(input_dim=6, output_dim=4)`,
with one input shaped `[1, 6]` and one output shaped `[1, 4]`. Replace those
settings and the adapter import with your model's actual interface.

The launcher needs Python 3.10+. Checking a model needs its framework environment;
building also needs compiler tools and a compatible native runtime. CUDA builds
need an NVIDIA GPU, compatible driver and CUDA development toolkit. Start with
AOTInductor (`aoti`); add TensorRT after the first build works.

The setup chapters use Bash on Linux and PowerShell on Windows. Later examples
use Linux paths; Windows users should substitute their project and runtime paths.
AOTI can target CPU when your model and toolchain support it; set `device` to
`cpu`. TensorRT requires CUDA.

## Linux setup

From the repository root, make the checkout launcher available:

```bash
export PATH="$PWD/native-inference:$PATH"
pnms-model-builder --help
```

An installed `pnms-model-builder` supports the same model commands.
The setup command below uses the checkout launcher to locate SDK and builder
sources; installed-wheel users should follow the
[source-selection options](reference.md#environment-setup). There is currently
no published default builder image or prebuilt SDK bundle. Choose one of the
following environments and reuse it across model builds.

### Create a local environment

With a C++20 compiler, CMake 3.20+ and your CUDA toolkit already installed:

```bash
pnms-model-builder setup-env /data/builder-env \
  --requirements /data/my-model/requirements.txt --json
source /data/builder-env/bin/activate
```

Choose a new environment directory. Put compatible, pinned model/framework
dependencies in `requirements.txt`; omit `--requirements` if you need no extra
packages. Setup installs the builder and AOTI dependencies and builds the native
runtime. It does not install system compilers, drivers or CUDA.

Save the returned `project_settings`: you will copy `executor` and `runtime`
into the project. For this Linux example, setup creates the executable at
`/data/builder-env/.physicsnemo/runtime/physicsnemo-infer`. Use the returned path
if you choose another directory or reuse an existing runtime.

### Use an existing local environment

Activate an environment containing your model's dependencies. From the checkout:

```bash
python -m pip install './native-inference/model-builder[export]'
```

Set `executor` to `local` and `runtime` to your compatible
`physicsnemo-infer` executable. Eager `check` needs no SDK; `build` needs a
runtime built against compatible native libraries. See the
[SDK build instructions](../cpp-runtime/README.md) and
[advanced environment options](reference.md#environment-setup).

### Use Docker

On a Linux x86_64 Docker host with NVIDIA GPU access, build the development
image from the repository root:

```bash
docker build -f native-inference/model-builder/images/Dockerfile.builder \
  -t pnms-model-builder:dev .
docker image inspect pnms-model-builder:dev --format '{{.Id}}'
```

The image must contain your model's Python dependencies; add them to your team's
builder image when needed. Set `executor` to `container` and `builder_image` to
the returned `sha256:...` ID. For a registry image, use `name@sha256:...`.
Mutable tags are rejected. Container execution uses the SDK inside the image,
so omit the local `runtime` setting in this case.

`init` defaults to container execution on Linux and local execution on Windows.
Select the environment you prepared explicitly in the configuration below.

## Windows setup

Model Builder and the C++ Inference SDK have native Windows build paths for
AOTInductor and TensorRT. The standard environment setup builds the generic
TensorRT backend; exact TensorRT profiles require the manually configured SDK
described in the [Windows SDK reference](windows.md#exact-tensorrt-profiles).
Run these PowerShell commands in **Developer PowerShell for VS 2022**, targeting
x64, from the repository root. WSL and Docker are not required. `setup-env`
does not install system compilers or GPU drivers.

The native Windows GPU path is experimental. Verify it on your target Windows
GPU with the checks below; the configured CPU CI checks and Linux/H100 results
do not establish Windows GPU compatibility.

### Windows prerequisites

- Windows 11 x64 and an NVIDIA Windows driver compatible with your CUDA toolkit.
  Native Windows CUDA does not require changing the GPU into a WSL driver mode.
- Full CPython 3.12 x64, with development headers and libraries (not embedded Python).
- Visual Studio 2022 or Build Tools 2022, with **Desktop development with C++**,
  MSVC v143, and a Windows SDK. Use the x64 developer shell so `cl.exe` is available
  to AOTInductor as well as CMake.
- CMake 3.24 or newer, so the Visual Studio generator treats external SDK
  headers as system includes while retaining strict warnings for project code.
- CUDA Toolkit **12.8**, including compiler and development files.
- A **TensorRT 10.x Windows x64 CUDA 12** SDK, including `include/NvInfer.h`,
  `lib/nvinfer_10.lib`, and its DLLs in `bin` or `lib`. Point `--tensorrt-root` at
  the extracted SDK directory. A Python-only TensorRT installation is insufficient.

The checked-in [CUDA requirements](../model-builder/requirements-windows-cu128.txt)
select PyTorch 2.10.0 with CUDA 12.8 and the community `triton-windows` 3.6 series.
PyTorch publishes [Windows CUDA wheels](https://pytorch.org/get-started/previous-versions/);
the [Windows Triton project](https://github.com/woct0rdho/triton-windows) documents
the PyTorch/Triton version pairing. This requirements file is a starting profile,
not a fully qualified dependency lock. Model-specific requirements may need more
packages. NVIDIA documents the [TensorRT Windows SDK installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-zip.html).

Use a short checkout path such as `C:\src\physicsnemo-serve` to leave room for
generated compiler filenames. Confirm the system tools before installing Python
dependencies:

```powershell
python --version
cmake --version
Get-Command cl
nvcc --version
nvidia-smi
```

### Create and activate the Windows environment

Set the TensorRT directory to the SDK you extracted. The environment directory
must not already exist. This example targets an L4: use `8.9` for Torch and
`sm89` for the build check below. Change both values for a different GPU.

```powershell
$builderEnv = Join-Path $env:USERPROFILE '.venvs\physicsnemo-native'
$tensorRt = 'C:\SDKs\TensorRT-10.13.3.9'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:PYTHONUTF8 = '1'
# PyTorch 2.10 native Windows AOTI omits this CUDA runtime import library.
$cudaRuntimeLibrary = Join-Path $env:CUDA_PATH 'lib\x64\cudart.lib'
if (!(Test-Path $cudaRuntimeLibrary)) { throw 'CUDA runtime development library missing.' }
$env:LINK = ($env:LINK + ' "' + $cudaRuntimeLibrary + '"').Trim()

$setupJson = python .\native-inference\pnms-model-builder setup-env $builderEnv `
  --backend aoti --backend tensorrt `
  --tensorrt-root $tensorRt --tensorrt-cuda-major 12 `
  --requirements .\native-inference\model-builder\requirements-windows-cu128.txt `
  --json
if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed; inspect the reported setup.log.' }
$setup = $setupJson | ConvertFrom-Json
$runtime = $setup.runtime

. (Join-Path $builderEnv '.physicsnemo\activate.ps1')
```

Setup selects the Windows `Scripts` venv layout, builds an x64 Release SDK,
and installs TensorRT Python bindings matching the C++ SDK headers. Its
PowerShell activation wrapper adds Torch, TensorRT and CUDA DLL directories
to the current process's `PATH`; use that wrapper in subsequent developer
shells as well. The ordinary `Scripts/Activate.ps1` alone does not add those
native dependency paths. Activation follows your existing PowerShell policy.
Setup retains the runtime path and activation command in
`<environment>/.physicsnemo/environment.json` and leaves model project files unchanged.

In each new developer shell, restore the CUDA, `TORCH_CUDA_ARCH_LIST`, `LINK`
and `PYTHONUTF8` settings above before activating the wrapper. Read `runtime`
from the saved `environment.json` to set `$runtime` again. The process-scoped [`LINK` setting](https://learn.microsoft.com/en-us/cpp/build/reference/linking?view=msvc-170#link-environment-variables)
supplies `cudart.lib` to MSVC without modifying PyTorch. UTF-8 mode also allows
framework progress messages to reach redirected logs on Windows.

For an extracted CUDA SDK whose Visual Studio integration is not registered,
set `$env:CMAKE_GENERATOR_TOOLSET = "cuda=$env:CUDA_PATH,host=x64"` before setup.
That SDK must include `extras\visual_studio_integration\MSBuildExtensions`.
Setup forwards this toolset explicitly; it also honors
`CMAKE_GENERATOR_INSTANCE` when selecting a particular Visual Studio installation.

To prepare just one backend, pass only its `--backend`. TensorRT requires
CUDA-enabled PyTorch for eager reference/export. CUDA AOTInductor additionally
requires the matching Windows Triton package; CPU AOTInductor does not require
Triton or a GPU. AOTInductor exports and the SDK must use the same Torch version.

### Verify the Windows installation

Check actual GPU computation before compiling a model:

```powershell
python -c "import torch, triton, tensorrt; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, triton.__version__, tensorrt.__version__); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print((torch.ones(4, device='cuda') * 2 + 1).cpu())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency check failed.' }

$candidate = Join-Path (Get-Location) ('out\inference\l4-' + [guid]::NewGuid().ToString('N'))
pnms-model-builder build affine `
  --executor local --runtime $runtime `
  --backend aoti --backend tensorrt `
  --device cuda --required-gpu-arch sm89 --output $candidate --json
if ($LASTEXITCODE -ne 0) { throw 'Native backend parity failed; inspect candidate logs.' }

& $runtime run "$candidate\model\backends\aoti" `
  --backend aoti --device cuda --values '0,1,-1,4'
if ($LASTEXITCODE -ne 0) { throw 'AOTInductor inference failed.' }
& $runtime run "$candidate\model\backends\tensorrt" `
  --backend tensorrt --device cuda --values '0,1,-1,4'
if ($LASTEXITCODE -ne 0) { throw 'TensorRT inference failed.' }
```

Both CLI invocations should print `output: 1 3 -1 9`. The build itself compares
all three affine cases with independent Python eager references for **both**
requested backends. Preserve `build.json`, `execution.json`, `checks/` and `logs/`
with the checkout commit and dirty diff when reporting results. Each build needs
a new output directory. Build packages on Windows for the target GPU; Linux
compiled packages and H100 engines are not Windows/L4 qualification evidence.

Continue with [Create your project](#create-your-project) using a Windows project
path, for example `pnms-model-builder init C:\models\my-model`, then
`Set-Location C:\models\my-model`. The adapter and tensor contracts are the same
on both platforms. Replace the Linux paths in the example configuration with
your own paths and the `runtime` returned by setup; JSON paths can use forward
slashes (`C:/...`) or escaped backslashes (`C:\\...`). Set `executor` to `local`
and choose the backends you prepared. Run `check` before `build`.

The later multi-line shell examples use Bash continuations (`\`); PowerShell
uses a backtick (`` ` ``). To invoke the native executable from its variable, use
`& $runtime`. For SDK contributor tests or exact TensorRT profiles, continue in
the [Windows SDK reference](windows.md).

## Create your project

Initialize the directory containing your model and weights:

```bash
pnms-model-builder init /data/my-model
cd /data/my-model
```

`init` creates `model-build.json` and `build_adapter.py`. It does not load the
model or weights and refuses to overwrite either file. Your other files stay
in place. Edit the configuration; this complete example uses the local Linux
environment above. On Windows, use the `runtime` returned by your setup:

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
  "input_names": ["features"],
  "output_names": ["prediction"],
  "backends": ["aoti"],
  "executor": "local",
  "runtime": "/data/builder-env/.physicsnemo/runtime/physicsnemo-infer",
  "device": "cuda",
  "output_root": "builds"
}
```

Paths in this file are relative to the project, unless absolute. `config` can
also be a JSON file path. List the local Python files or packages imported by
your adapter in `source`; directories contribute Python files, and the adapter
is always included. Declare non-Python data files under `assets`. Dependencies
outside your project must be installed in the selected environment.

For Docker, change `executor` to `container`, remove `runtime`, and add the
immutable `builder_image` you obtained above. For more fields and named build
profiles, see [project settings](reference.md#project-settings).

## Connect your model and inputs

Replace the placeholders in `build_adapter.py` with two functions:

```python
def create_model(config, assets):
    from model import MyModel

    return MyModel(**config).eval()


def create_cases(config, assets):
    import torch

    return torch.load(
        assets["validation_inputs"], map_location="cpu", weights_only=True
    )
```

Construct the architecture on CPU. The builder strictly loads your checkpoint
and moves the model and cases to the selected device. You do not load weights
in the adapter or supply expected outputs.

`create_cases` returns a nonempty list of positional input tuples. For
`model(features, context)`, each case is `(features, context)`. All cases must
have the same tensor shapes and dtypes. Use your existing validation loader,
or save preprocessed inputs from that code for the example above:

```python
import torch

# Run in your validation code, where these real input tensors already exist.
torch.save(
    [(first_input.detach().cpu(),), (second_input.detach().cpu(),)],
    "/data/my-model/validation-inputs.pt",
)
```

Use meaningful cases for your deployment. Native/Python agreement verifies
these cases; it does not establish scientific accuracy on unseen CFD inputs.
Advanced models can select [compiler profiles](reference.md#aoti-profiles-and-options)
and [export hooks](reference.md#export-hooks).

## Import a PhysicsNeMo checkpoint

Skip this step for a plain PyTorch state dictionary. For a trusted `.mdlus`
archive, run from the project in an environment containing compatible
PhysicsNeMo and model dependencies:

```bash
pnms-model-builder import-checkpoint weights/model.mdlus \
  --output imports/my-model --json
```

The importer uses your configured environment and requires a new output
directory. It writes `checkpoint.pt`, `config.json` and conversion reports.
Copy the returned `project_settings` values into your project's `checkpoint`
and `config` fields, and connect the reported model class in `create_model`.
The command does not edit your project. Model validation and compilation happen
in the next step. See [checkpoint import](reference.md#checkpoint-import) for
hash pins, supported constructors and the conversion checks.

## Check and build

From your configured project directory:

```bash
pnms-model-builder check . --json
pnms-model-builder build . --output builds/first --json
```

| Command | What it verifies |
| --- | --- |
| `check . --config-only` | Configuration and input identities, without running the model or creating output. |
| `check .` | Imports, strict checkpoint loading, eager model execution and inferred tensor contracts. Writes a check report. |
| `build .` | Repeats eager checks, exports/compiles each selected backend, runs native inference and compares every case with Python. |

`builds/first` must not already exist. `--json` returns a structured result with
status and report paths; treat a nonzero exit code as failure. A successful
`check` does not guarantee exportability. Every requested backend must pass
compilation and native parity for a build to succeed.

These commands describe an authoring project created by `init`. Explicit
recipes use `check --config-only` or `build`; see [project formats](reference.md#project-formats).

## Use the compiled model

A successful build produces:

```text
builds/first/
├── model/                         # Deploy this bundle
│   ├── model-release.json
│   ├── source-check.json
│   └── backends/
│       └── aoti/                   # Load this backend package
│           ├── model.json
│           └── model.pt2
├── checks/                        # Native comparison reports and tensors
├── logs/                          # Compiler/runtime diagnostics
└── ...                            # Retained inputs, graphs and build records
```

Copy `model/` to retain the bundle inventory and reports, or copy an individual
backend directory with all its files. The runtime loads
`model/backends/aoti/`, not the parent `model/`. A TensorRT package uses
`model/backends/tensorrt/` with `model.plan` beside `model.json`.

For the single-input example, save a validation tensor as raw FP32 bytes from
your project directory:

```python
import torch

features = torch.load("validation-inputs.pt", map_location="cpu", weights_only=True)[0][0]
features.contiguous().numpy().astype("<f4").tofile("features.f32")
```

Inspect and run the local AOTI package with the matching SDK:

```bash
RUNTIME=/data/builder-env/.physicsnemo/runtime/physicsnemo-infer
PACKAGE=builds/first/model/backends/aoti
"$RUNTIME" inspect "$PACKAGE"
"$RUNTIME" run "$PACKAGE" --backend aoti --device cuda \
  --input-file features=features.f32 \
  --output-file prediction=prediction.f32 --output-metadata prediction.json
```

Input names, shapes and dtypes must match `model.json`; supply one `--input-file`
per input for models with multiple tensors. For a Docker build, run inference
in the compatible image or on a host with the matching SDK and native libraries.
The package no longer needs your Python model, adapter or original checkpoint.
It still needs compatible native Torch/TensorRT libraries and the intended
device/toolchain. See [SDK usage](../cpp-runtime/README.md) and
[package compatibility](reference.md#package-compatibility).

## Rebuild after changes

For another build with the same selected inputs, let the builder choose a fresh
output directory:

```bash
pnms-model-builder build . --json
```

The first build records input and environment identities in
`model-build.lock.json`. After intentionally changing code, weights, settings,
assets or the selected toolchain, acknowledge that change:

```bash
pnms-model-builder build . --update-lock --json
```

Changing only the output directory does not require a lock update. Neither
check mode updates the lock or accepts `--update-lock`; use `build` to enforce
the saved build selection. See [lock rules](reference.md#project-locks).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Incomplete project or missing checkpoint | Fill in `checkpoint`, select the executor/runtime or image, and replace both adapter placeholders. |
| Model import fails | Add local Python modules to `source`; install third-party dependencies in the selected environment. |
| Checkpoint keys, shapes or dtypes do not match | Verify the model constructor/configuration and use its plain tensor `state_dict`, not an optimizer/training wrapper. |
| CUDA or compiler dependency unavailable | Check the selected environment, driver, CUDA toolkit and native SDK; configuration-only checks do not exercise them. |
| `PROJECT_LOCK_MISMATCH` | Review the changed selection. Use `build --update-lock` only when that change is intentional. |
| Output directory already exists | Choose a new `--output`, or omit it to use a fresh directory under `output_root`. |
| Eager check passes but export or parity fails | Read the result's diagnostics and retained `logs/`/`checks/`. Confirm backend support, representative inputs and the matching toolchain; consult [profiles](reference.md#tensorrt-profiles-and-assets) when your model needs them. |

For complete fields, command results and artifact paths, use the
[technical reference](reference.md). Builds retain diagnostics after failure;
partial artifacts are not a successful deployable build.
