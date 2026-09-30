# Native Windows 11 development

Model Builder and the C++ Inference SDK have native Windows build paths for
AOTInductor and TensorRT. The standard environment setup builds the generic
TensorRT backend; exact TensorRT profiles require the manually configured SDK
described below. Run the commands in **Developer
PowerShell for VS 2022**, targeting x64, from the repository root. WSL and Docker
are not required. `setup-env` does not install system compilers or GPU drivers.

Configured Windows CI jobs exercise the core SDK, static/shared installation,
and CPU AOTInductor. They must pass on the branch before claiming Windows validation.
CUDA qualification must run on the target Windows GPU: a successful Linux/H100
run or a skipped CUDA test does not qualify a Windows/L4 deployment. The native
Windows GPU path is experimental until the checks below pass on your machine.

## Prerequisites

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

## Create one environment for both backends

Set the TensorRT directory to the SDK you extracted. The environment directory
must not already exist. For an L4, the CUDA architecture is 8.9.

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

$setupJson = python .\native-inference\physicsnemo-model-builder setup-env $builderEnv `
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

Keep the CUDA, `LINK`, and `PYTHONUTF8` settings in each developer shell used for
model builds and CTest. The process-scoped [`LINK` setting](https://learn.microsoft.com/en-us/cpp/build/reference/linking?view=msvc-170#link-environment-variables)
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

## Verify CUDA export and native inference

Check actual GPU computation before compiling a model:

```powershell
python -c "import torch, triton, tensorrt; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, triton.__version__, tensorrt.__version__); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print((torch.ones(4, device='cuda') * 2 + 1).cpu())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency check failed.' }

$candidate = Join-Path (Get-Location) ('out\inference\l4-' + [guid]::NewGuid().ToString('N'))
physicsnemo-model-builder build affine `
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

For a custom format-2 model project, run `init` using Python or the installed
command, connect the adapter/checkpoint, and set `backends` to `["aoti", "tensorrt"]`
plus the `executor`/`runtime` settings returned by setup. New projects initialized
on Windows default to local execution. Run `check` before `build`.

## C++ development tests

The environment's runtime is reusable but is built without tests. Build a
separate test tree from the same activated developer shell:

```powershell
$testBuild = Join-Path (Get-Location) 'out\inference\windows-tests'
$generatorArgs = @()
if ($env:CMAKE_GENERATOR_TOOLSET) {
  $generatorArgs += @('-T', $env:CMAKE_GENERATOR_TOOLSET)
}
if ($env:CMAKE_GENERATOR_INSTANCE) {
  $generatorArgs += "-DCMAKE_GENERATOR_INSTANCE=$env:CMAKE_GENERATOR_INSTANCE"
}
cmake -S native-inference/cpp-runtime -B $testBuild `
  -G 'Visual Studio 17 2022' -A x64 @generatorArgs `
  -DPNMIR_BUILD_TESTS=ON -DPNMIR_ENABLE_AOTI=ON -DPNMIR_ENABLE_TENSORRT=ON `
  "-DPython3_EXECUTABLE=$builderEnv\Scripts\python.exe" `
  "-DPNMIR_TENSORRT_ROOT=$tensorRt" "-DCUDAToolkit_ROOT=$env:CUDA_PATH"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
cmake --build $testBuild --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'SDK build failed.' }
ctest --test-dir $testBuild -C Release --output-on-failure --no-tests=error
if ($LASTEXITCODE -ne 0) { throw 'SDK tests failed.' }
```

Require actual passes for `pnmir_aoti_cuda_integration`, `pnmir_aoti_device_input`,
and the TensorRT integration/device-input tests. CTest can report an overall zero
exit status while optional GPU tests are skipped; inspect its results.

For installation, pass `--config Release` to `cmake --install` too. DLLs go into
the SDK's `bin` directory; add it and the external dependency DLL directories to
`PATH` for a separate C++ application. The SDK's native dependencies are not bundled.

## Exact TensorRT profiles

`setup-env` does not enable exact TensorRT plugins. For a supported Transolver
graph, build a separate SDK with `PNMIR_ENABLE_TENSORRT_EXACT=ON`, then select
`layout-order-exact-v2` and its nine plugin assets in the model project. Leaving
`tensorrt_profile` at `baseline` uses ordinary TensorRT operators, even when the
runtime contains exact plugins. Missing exact-profile assets fail the build;
there is no automatic fallback.

The exact attention plugin requires a PyTorch source checkout matching the
installed wheel's `torch.version.git_version`, that checkout's pinned CUTLASS
headers, and the installed Torch include tree. These are build-time dependencies.
Use the same CUDA/TensorRT installations for compilation, Python export and
native execution. Build for the target GPU; use `89` for L4 rather than the
SDK's default `90`. Use CMake 3.24 or newer with the Visual Studio generator
for external-header handling and CUDA warning settings.

From the same activated developer shell, set the source paths and reuse the
`$generatorArgs` from the C++ test setup:

```powershell
$torchSource = 'C:\src\pytorch'  # checkout matching torch.version.git_version
$cutlassInclude = Join-Path $torchSource 'third_party\cutlass\include'
$torchInclude = & "$builderEnv\Scripts\python.exe" -c "from torch.utils.cpp_extension import include_paths; print(include_paths()[0])"
if ($LASTEXITCODE -ne 0) { throw 'Torch header discovery failed.' }
$exactBuild = Join-Path (Get-Location) 'out\inference\windows-exact-build'
$exactSdk = Join-Path (Get-Location) 'out\inference\windows-exact-sdk'

cmake -S native-inference/cpp-runtime -B $exactBuild `
  -G 'Visual Studio 17 2022' -A x64 @generatorArgs `
  -DPNMIR_BUILD_TESTS=ON -DPNMIR_ENABLE_AOTI=ON `
  -DPNMIR_ENABLE_TENSORRT=ON -DPNMIR_ENABLE_TENSORRT_EXACT=ON `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  "-DPython3_EXECUTABLE=$builderEnv\Scripts\python.exe" `
  "-DPNMIR_TENSORRT_ROOT=$tensorRt" "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
  "-DPNMIR_PYTORCH_SOURCE_ROOT=$torchSource" `
  "-DPNMIR_CUTLASS_INCLUDE_DIR=$cutlassInclude" `
  "-DPNMIR_TORCH_INCLUDE_DIR=$torchInclude" `
  "-DCMAKE_INSTALL_PREFIX=$exactSdk"
if ($LASTEXITCODE -ne 0) { throw 'Exact SDK configuration failed.' }
cmake --build $exactBuild --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Exact SDK build failed.' }
ctest --test-dir $exactBuild -C Release --output-on-failure --no-tests=error
if ($LASTEXITCODE -ne 0) { throw 'Exact SDK tests failed.' }
cmake --install $exactBuild --config Release
if ($LASTEXITCODE -ne 0) { throw 'Exact SDK installation failed.' }

$exactRuntime = Join-Path $exactSdk 'bin\physicsnemo-infer.exe'
$env:PATH = "$exactSdk\bin;$env:PATH"
```

Keep the entire installed `bin` directory available: the runtime registers all
twelve compiled plugin DLLs, while Transolver's v2 profile uses nine. Set the
project's `runtime` to `$exactRuntime` and declare the assets listed in
[TensorRT profiles](add-model.md#tensorrt-profiles). On Windows their filenames
are `pnmir_tensorrt_exact_<operator>_plugin.dll`, without the Linux `lib` prefix.
Copy the selected DLLs into the project's declared asset paths so Model Builder
captures and hashes them. Updating a project's profile or assets requires
updating its lock with `build --update-lock`. The older `layout-order-exact`
profile retains its eight-plugin contract; use v2 to require byte equality and
preserve the deslicing BMM layout as well.

For the GeoTransolver cached surface core, select `geotransolver-exact-v2`
and its ten DLL assets, including WeightedBlend and DesliceBmm. This profile
retains scalar sigmoid gate freezing and preserves the original BMM layout
after self/cross-attention mixing. The legacy `geotransolver-exact` keeps
its nine-plugin contract; migrate explicitly with the additional DLL and
`build --update-lock`. Both profiles reject native/reference byte differences.
Geometry preprocessing remains outside the native core package.

For the DoMINO surface core, add `-DPNMIR_BUILD_DOMINO_EXACT_OPS=ON` to the
exact SDK configuration above. Select `aten-boundary-exact-v3` for AOTI and
`domino-surface-exact` for TensorRT. Copy the sidecar `pnmir_domino_exact_ops.dll`
and the four TensorRT DLLs for Linear, GELU, ScalarDiv and InverseDistanceBlend
into the [DoMINO project's declared assets](../examples/domino-surface-core/model-build.json),
using Windows filenames in place of its `.so` paths. Keep the SDK `bin`
directory, including the sidecar DLL, on `PATH` for native execution. Both
backends require byte-identical core outputs; the package consumes prepared
surface neighborhoods and does not include the full geometry preprocessing.

Require actual passes for the exact dimension, DesliceBmm and WeightedBlend tests as well
as the enabled backend integration tests; an optional-test skip is insufficient.
Then rerun Model Builder and independent Python/native comparisons on the target
Windows GPU with the original validation cases and tolerances. Compiling the SDK
does not establish byte equality. The prior H100 result and a successful baseline
affine test do not qualify a Windows Transolver exact-profile package.
