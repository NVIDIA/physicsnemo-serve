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
dependencies in that environment. See [builder setup](../../model-builder/README.md).
Keep generated files outside the checkout. Run from the repository root:

```bash
DEMO_ROOT=/tmp/physicsnemo-transolver-demo  # choose a fresh directory
SDK_ROOT="$DEMO_ROOT/sdk"
TORCH_ROOT="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
WARP_ROOT="$(python -c 'import pathlib, warp; print(pathlib.Path(warp.__file__).parent)')"

cmake -S native-inference/cpp-runtime -B "$DEMO_ROOT/sdk-build" \
  -DPNMIR_ENABLE_AOTI=ON -DPNMIR_BUILD_TESTS=OFF \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SDK_ROOT"
cmake --build "$DEMO_ROOT/sdk-build" --parallel 4
cmake --install "$DEMO_ROOT/sdk-build"

cmake -S native-inference/workflows/transolver -B "$DEMO_ROOT/workflow-build" \
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
`PhysicsNeMoInference::runtime` and `PhysicsNeMoInference::aoti`.

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
./native-inference/physicsnemo-model-builder import-checkpoint "$CHECKPOINT" \
  --project "$DEMO_PROJECT" --output "$DEMO_PROJECT/weights" --json
./native-inference/physicsnemo-model-builder check "$DEMO_PROJECT" --json
./native-inference/physicsnemo-model-builder build "$DEMO_PROJECT" \
  --runtime "$SDK_ROOT/bin/physicsnemo-infer" \
  --output "$DEMO_ROOT/model-build" --json
```

The template exports the complete learned surface model with static inputs
`fx [1,75,2]` and `embedding [1,75,6]`, and output `[1,75,4]`. It uses the
`aten-boundary-exact-v2` AOTI profile and three deterministic parity cases.
The model package goes under `model-build/model/backends/aoti/`; the original
checkpoint is not needed by the C++ executable. Keep `global_stats.json` with
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

Use matching normalization statistics and flow conditions for your checkpoint.
Model/native numerical parity does not establish scientific CFD accuracy.

## Show the result in a GUI

The customer demo can finish with this native workflow and an interactive
browser view. From the Mac checkout:

```bash
# Full walkthrough: 23 existing steps, then native workflow and browser results.
bash reports/customer-demo-20260916/run-demo.sh --split --e2e

# Rehearse just the final native run and GUI.
bash reports/customer-demo-20260916/run-e2e-demo.sh
```

The launcher stages fresh scripts on the prepared H100, runs all 65,536 cells
of a coarsened full-car surface in 2,048-cell batches, exports the GUI, retrieves
the results, and opens the HTML locally. The demonstration mesh is prepared
from the complete `run_1` vehicle STL before the live run; it is separate from
the original full-resolution CFD boundary mesh. `--point-limit 0` selects every
cell of this coarsened VTP. The static model package and viewer dependencies
are also prepared in advance. In split mode, press Enter to close the panes after step 25;
the browser then opens. The finale uses a separately qualified 2,048-point build of the raw-workflow
model above. The earlier tensor-only demo has a different input contract.

The viewer shows pressure or wall-shear stress across the full car: every
displayed surface triangle has its own native prediction, with no field
interpolation. It includes a neutral original-geometry overview, per-cell
values, a field chart, timings and CSV download. Its coverage label identifies
the coarsened demonstration mesh, and run details record the mesh processing
method and hash. This is numerical agreement with the Python model on a
coarsened mesh, not full-resolution CFD validation. The viewer requires no web
server or network access after export. Existing prefix results keep their
patch view and sample-location marker; cells outside their selected prefix
have no displayed predictions.

To export another existing surface run on a machine with NumPy, Python VTK,
and Plotly 6.3.1:

```bash
python native-inference/workflows/transolver/demo/export_viewer.py \
  --metadata "$DEMO_ROOT/results/metadata.json" \
  --output "$DEMO_ROOT/results/result.html"
```

For a prepared demonstration mesh, add
`--mesh-provenance "$DEMO_ROOT/mesh-provenance.json"`. This optional JSON records
`label`, `method`, `source_cell_count`, `output_cell_count` and `mesh_sha256`.
The exporter verifies the VTP hash and its cell count before displaying the
provenance. The full-car launcher supplies this file automatically.

Both standardized and physical output files must be present. To show a
reference-comparison badge, also supply `--reference-report` pointing to a
compatible saved eager-Python parity report, and produce input dumps with
`--dump-input-dir "$DEMO_ROOT/results/inputs"` when running the CLI. The
exporter compares all four current tensor hashes and the case settings against
that report; a mismatch fails export. Without a report the GUI says
"Not compared". Python is used only for result presentation after the C++
workflow completes.

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

For TensorRT packages, build the SDK with the needed TensorRT plugins and
configure this example with `-DPNMIR_ENABLE_TENSORRT=ON`, then pass
`--backend tensorrt`. Both full and tail shapes must have matching packages.
The example template and the validation described here use AOTI. TensorRT
routing does not imply that every historical package's operators are present
in the current SDK.

The retained reader accepts the DrivAerML little-endian, uncompressed,
inline-base64 Float32 VTU point format. A volume `--point-limit N` bounds
coordinate decoding/allocation; the surface VTP reader still loads the mesh
before selecting cells. `--point-limit 0` selects all locations and can
require substantial memory. Every inference block must contain at least two
points. Keep batch size and permutation seed fixed when comparing results:
Transolver attention depends on which points share a batch. Raw-geometry Python
reference checks must also use the same VTK version as the C++ build. The large
demo was qualified with VTK 9.1.0 and Python PyVista 0.43.10; VTK 9.6.2 changes
some polygon centers and normals on this case.

## Validation

The C++ tests cover normalization channel order, physical decoding, model
contracts, and bounded UInt32/UInt64 VTU decoding. The CLI test checks `--help`
without loading a model. The model-project tests live in
[`tests/examples/transolver-surface`](../../tests/examples/transolver-surface).
Raw-workflow qualification compares prepared features and both output spaces
against the preserved implementation on real surface and volume inputs.

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
| Complete `run_1` car coarsened from STL to 65,536 triangles; all cells in 2,048-cell blocks | Prepared inputs, standardized and physical outputs byte-identical to eager Python; interactive GUI ready in 42.5 seconds in the measured finale run |
| Workflow CTest | 2/2 passed |

These are bounded real-data and coarsened-geometry checks. Original
full-resolution CFD-case execution and TensorRT were not requalified for this
port. Generated models, logs and parity evidence stay
outside the source tree.
