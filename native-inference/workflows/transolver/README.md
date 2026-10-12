# Transolver: raw mesh to physical fields from the CLI

This example runs preprocessing, model inference, and postprocessing in one
C++ process using the installed **PhysicsNeMo C++ Inference SDK**. It is adapted
from `gpu_programming` commit `87b78bf6cec3030cfbaea10f14e4c64cbc836407`,
`examples/physicsnemo_cfd_transolver_workflow`. The mesh reader, Warp SDF,
normalization, seeded batching, and physical decoding retain the original
implementation. Model execution uses the current SDK package API.

```text
VTP surface cells / VTU volume points + vehicle STL
    -> C++ geometry loading and normalization (Warp SDF for volume)
    -> seeded point batches -> SDK AOTI / TensorRT sessions
    -> restore source order -> decode physical units -> FP32 fields + JSON
```

Python is needed to import/export a checkpoint and qualify results. The
`physicsnemo-transolver` executable uses native LibTorch, CUDA, VTK, and Warp
libraries; it does not invoke Python or exchange intermediate tensor files.
It does not require PhysicsNeMo Serve, Redis, or a server.

## Build the SDK and example

Use Linux with CUDA, a matching CUDA-enabled Torch development installation,
VTK development libraries, CMake, a C++20 compiler, and Warp **1.15.0**. The
exporter, SDK and workflow must use the same Torch/CUDA installation and ABI.
The real-checkpoint example also needs PhysicsNeMo 2.1.1 and Model Builder's
dependencies in that environment. See [builder setup](../../docs/reference.md#environment-setup).
The exact TensorRT attention plugin requires matching PyTorch source, CUTLASS
and generated headers. The paths below use the pinned NGC environment; see
[exact SDK dependencies](../../cpp-runtime/README.md) for other installations.
Keep generated files outside the checkout. Run from the repository root:

```bash
DEMO_ROOT=/tmp/physicsnemo-transolver-demo  # choose a fresh directory
SDK_ROOT="$DEMO_ROOT/sdk"
TORCH_ROOT="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
WARP_ROOT="$(python -c 'import pathlib, warp; print(pathlib.Path(warp.__file__).parent)')"

cmake -S native-inference/cpp-runtime -B "$DEMO_ROOT/sdk-build" \
  -DPNMIR_ENABLE_AOTI=ON -DPNMIR_BUILD_TESTS=OFF \
  -DPNMIR_ENABLE_TENSORRT=ON -DPNMIR_ENABLE_TENSORRT_EXACT=ON \
  -DPNMIR_PYTORCH_SOURCE_ROOT=/opt/pytorch/pytorch \
  -DPNMIR_CUTLASS_INCLUDE_DIR=/opt/pytorch/pytorch/third_party/cutlass/include \
  -DPNMIR_TORCH_INCLUDE_DIR="$TORCH_ROOT/include" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SDK_ROOT"
cmake --build "$DEMO_ROOT/sdk-build" --parallel 4
cmake --install "$DEMO_ROOT/sdk-build"

cmake -S native-inference/workflows/transolver -B "$DEMO_ROOT/workflow-build" \
  -DPNMIR_ENABLE_AOTI=ON -DPNMIR_ENABLE_TENSORRT=ON \
  -DCMAKE_PREFIX_PATH="$SDK_ROOT;$TORCH_ROOT" \
  -DPNMIR_WARP_ROOT="$WARP_ROOT" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEMO_ROOT/workflow-build" --parallel 4
export LD_LIBRARY_PATH="$TORCH_ROOT/lib:$WARP_ROOT/bin:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$DEMO_ROOT/workflow-build" --output-on-failure
"$DEMO_ROOT/workflow-build/physicsnemo-transolver" --help
```

`90` targets the H100; select the architecture of your build/deployment GPU.
The example is a standalone CMake consumer of
`PhysicsNeMoInference::runtime`, `PhysicsNeMoInference::aoti` and
`PhysicsNeMoInference::tensorrt`.

## Import and build a surface checkpoint

Obtain the trusted Transolver surface `Transolver.0.501.mdlus` and its
`global_stats.json` from `nvidia/transolver_drivaerml` on Hugging Face
(qualified revision `96477aeb86d24c26ccf0797bca1b3851268017d0`). Use
DrivAerML `run_1/boundary_1.vtp` and `run_1/drivaer_1.stl` from
`neashton/drivaerml`. These large assets are separate from the example.
Select paths to the downloaded files:

```bash
CHECKPOINT=/data/models/transolver_drivaerml_surface_checkpoint/Transolver.0.501.mdlus
STATS=/data/models/transolver_drivaerml_surface_checkpoint/global_stats.json
CASE_ROOT=/data/drivaerml/run_1
DEMO_PROJECT="$DEMO_ROOT/model-project"

cp -R native-inference/examples/transolver-surface "$DEMO_PROJECT"
mkdir -p "$DEMO_PROJECT/assets/tensorrt"
for plugin in linear gemm token_sum slice_bmm layer_norm softmax attention gelu deslice_bmm; do
  cp "$SDK_ROOT/lib/libpnmir_tensorrt_exact_${plugin}_plugin.so" "$DEMO_PROJECT/assets/tensorrt/"
done
./native-inference/pnms-model-builder import-checkpoint "$CHECKPOINT" \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
./native-inference/pnms-model-builder check "$DEMO_PROJECT" --json
./native-inference/pnms-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_ROOT/bin/physicsnemo-infer" \
  --output "$DEMO_ROOT/model-build" --json
```

The template exports the complete learned surface model with static inputs
`fx [1,75,2]` and `embedding [1,75,6]`, and output `[1,75,4]`. It uses the
`aten-boundary-exact-v2` AOTI profile, `layout-order-exact-v2` TensorRT profile
and three deterministic parity cases. TensorRT publication requires byte-identical
native/Python results. The nine copied plugin libraries are declared, hashed
build dependencies. The packages go under `model-build/model/backends/aoti/`
and `model-build/model/backends/tensorrt/`; the original checkpoint is not
needed by the C++ executable. Keep `global_stats.json` with
your deployment inputs because physical-unit decoding needs it.

## Run the complete workflow

```bash
"$DEMO_ROOT/workflow-build/physicsnemo-transolver" \
  --backend aoti \
  --package "$DEMO_ROOT/model-build/model/backends/aoti" \
  --mesh "$CASE_ROOT/boundary_1.vtp" --stl "$CASE_ROOT/drivaer_1.stl" \
  --stats "$STATS" --domain surface \
  --point-limit 75 --block-size 75 --seed 0 \
  --air-density 1.205 --stream-velocity 30.0 \
  --standardized-output "$DEMO_ROOT/results/standardized.f32" \
  --physical-output "$DEMO_ROOT/results/physical.f32" \
  --metadata "$DEMO_ROOT/results/metadata.json"
```

This processes the first 75 surface cells and writes 75 rows of four
little-endian FP32 values: pressure, WSS-x, WSS-y, WSS-z. Each file is 1,200
bytes. Outputs are restored to source-cell order. The JSON sidecar records
shape, package paths, block plan, seed, library versions and timings. Add
`--dump-input-dir "$DEMO_ROOT/results/inputs"` to diagnose prepared inputs.
The standardized, physical, and metadata output paths must identify distinct
files and must not contain dangling symlinks; the CLI checks these paths before
preprocessing or writing outputs.

Use matching normalization statistics and flow conditions for your checkpoint.
Model/native numerical parity does not establish scientific CFD accuracy.
Run the same command with `--backend tensorrt` and
`--package "$DEMO_ROOT/model-build/model/backends/tensorrt"` to execute the
TensorRT package. Use separate result paths to preserve both sets of outputs.

## Other shapes, volume, and backends

To export a different fixed block size, change `POINT_COUNT` in a fresh copied
project, build into a new output directory, and use the same `--block-size`.
If reusing an already-built project, pass `build --update-lock` to acknowledge
the intentional source change as well as selecting a fresh `--output`.
For a final partial block, build a second package with that exact point count
and pass another `--package`. The CLI selects a persistent session by point
count. A qualified dynamic package can cover multiple block sizes through
the same CLI; older `.pnmir` packages remain supported.

Volume uses a compatible **volume** model package, its statistics,
`--domain volume`, `--mesh volume_419.vtu`, and `--stl drivaer_419.stl`.
The volume contract is `fx [1,N,2]`, `embedding [1,N,7]` and output `[1,N,5]`:
velocity-x/y/z, pressure, turbulent viscosity. The bundled builder template
is for the surface checkpoint. The native volume path is retained for
existing compatible volume packages.

The template builds both AOTI and TensorRT packages. Select one backend per
workflow invocation with `--backend aoti` or `--backend tensorrt`, and pass
the corresponding backend package paths for both full and tail shapes.
TensorRT uses the exact-enabled SDK built above; historical packages may
require other operators not present in that SDK.

The retained reader accepts the DrivAerML little-endian, uncompressed,
inline-base64 Float32 VTU point format with exactly one `Piece`. Multipiece
files are rejected before inference. The reader scans the grid's XML in
fixed-size blocks to validate this restriction, including when a point limit
is set. A volume `--point-limit N` bounds coordinate decoding/allocation;
the surface VTP reader still loads the mesh before selecting cells.
`--point-limit 0` selects all locations and can
require substantial memory. Every inference block must contain at least two
points. Keep batch size and permutation seed fixed when comparing results:
Transolver attention depends on which points share a batch. Raw-geometry Python
reference checks must also use the same VTK version as the C++ build. The large
surface cases were qualified with VTK 9.1.0 and Python PyVista 0.43.10; VTK 9.6.2 changes
some polygon centers and normals on this case.

## Validation

The C++ tests cover normalization channel order, physical decoding, model
contracts, bounded UInt32/UInt64 VTU decoding, and multipiece rejection.
The CLI tests check `--help`
and output-path collisions without loading a model. The model-project tests live in
[`test_transolver.py`](../../tests/examples/test_transolver.py).
Raw-workflow qualification compares prepared features and both output spaces
against the preserved implementation on real surface and volume inputs.

After building the surface package above, run the native output-path
regression in the same CUDA environment:

```bash
python native-inference/workflows/transolver/tests/test_cli_output_paths.py \
  --executable "$DEMO_ROOT/workflow-build/physicsnemo-transolver" \
  --package "$DEMO_ROOT/model-build/model/backends/aoti" \
  --mesh "$CASE_ROOT/boundary_1.vtp" --stl "$CASE_ROOT/drivaer_1.stl" \
  --stats "$STATS"
```

This checks relative output paths with default and custom metadata locations,
output shapes and file sizes, and omitted optional outputs. Use
`--point-count` for a package built with a different static point count.

On 2026-09-17, the current SDK/example passed on H100 80GB, driver 570.195.03,
CUDA 13.1, Torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, VTK 9.1.0 and
Warp 1.15.0:

| Check | Result |
| --- | --- |
| Fresh checkpoint import → check → AOTI build | Three model cases passed native parity |
| Fresh package, real `run_1` surface, 75 cells in one block | Inputs, standardized and physical outputs byte-identical to fresh eager Python |
| Migrated vs original CLI, real surface and volume, 75 locations in 32/32/11 blocks | All eight tensor comparisons byte-identical |
| Larger 2,048-point package | Three model cases passed native parity |
| Real `run_1`, 65,536 and 131,072 cells in 2,048-cell blocks | Both cases: inputs, standardized and physical outputs byte-identical to eager Python with matching VTK 9.1.0 |
| Complete `run_1` car coarsened from STL to 65,536 triangles; all cells in 2,048-cell blocks | Prepared inputs, standardized and physical outputs byte-identical to eager Python |
| Workflow CTest | 2/2 passed |

These are bounded real-data and coarsened-geometry checks. Original
full-resolution CFD-case execution and TensorRT were not requalified for this
port. Generated models, logs and parity evidence stay
outside the source tree.
