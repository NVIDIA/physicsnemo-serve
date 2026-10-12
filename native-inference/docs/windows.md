# Windows SDK reference

For prerequisites, environment setup, activation and your first native model
build, follow the user manual's [Windows setup chapter](user-guide.md#windows-setup).
This reference covers C++ SDK tests and exact TensorRT plugin builds.

Run the commands below from the repository root in **Developer PowerShell for
VS 2022**, targeting x64, after completing that chapter. Reuse its `$builderEnv`,
`$tensorRt`, `$env:CUDA_PATH`, `$env:TORCH_CUDA_ARCH_LIST`, `$env:LINK` and
`$env:PYTHONUTF8` settings, and activate
`<environment>/.physicsnemo/activate.ps1`. The ordinary venv activation script
alone does not add the native DLL paths.

Configured Windows CI jobs exercise the core SDK, static/shared installation
and CPU AOTInductor. CUDA and exact-profile validation must run on the target
Windows GPU; Linux/H100 results do not qualify a Windows deployment.

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
[TensorRT profiles](reference.md#tensorrt-profiles-and-assets). On Windows their filenames
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
