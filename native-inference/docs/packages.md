# Model packages and build evidence

A successful Model Builder invocation creates a **candidate**, with compiled
backend packages and the evidence required by its recipe/workflow.
`model-release.json` is an inventory; writing it does not publish a model or establish complete
scientific CFD qualification.

## Generic build output

```text
<output>/
  source/                           # captured recipe, adapter and declared inputs
    effective-recipe.json           # format 2: replay recipe with retained paths
    model-inputs/                   # format 2: config, checkpoint and data assets
  exported/
    aoti/program.pt2                 # exported graph, before compilation
    tensorrt/model.onnx              # ONNX graph and any external weights
  model/
    model-release.json              # candidate inventory, paths relative to model/
    backends/
      aoti/
        model.json
        model.pt2
      tensorrt/
        model.json
        model.plan
  checks/                           # backend/case metadata and input/reference/output bytes
  logs/                             # compiler and native subprocess diagnostics
  build.json                        # build status, sources, runtime and environment
  execution.json                    # executor and selected toolchain identities
  project/                          # present for project-based commands
    model-build.json
    effective-config.json
    model-build.lock.json
  frontend.log                      # captured stdout diagnostics in JSON mode
```

Only requested backend directories appear. The compiled payload sits beside
`model.json`: `model.pt2` for AOTI or `model.plan` for TensorRT. CPU and CUDA AOTI
use the same filenames; the manifest's artifact `target` records `cpu` or `cuda`,
so the directory name does not identify the execution target.

The raw graph at `exported/aoti/program.pt2` and the compiled payload at
`model/backends/aoti/model.pt2` are separate files even though both use a `.pt2`
extension. Each backend directory is a complete, directly loadable package.
Loading uses `model.json` and its relative artifact paths, not a directory name
or suffix. The manifest schema and compiled artifact formats are unchanged.

The lower-level ONNX importer writes `model.json` and `model.onnx` at its
package root. It preserves graph-relative external data for top-level dense
initializers and sparse-initializer values; external data in nested-subgraph
tensors and sparse indices is not covered yet. External-data locations must
not overlap the reserved root entries `model.json` or `model.onnx`, including
case variants and paths beneath those names. Rename conflicting files/directories
and update the graph's references before import. This importer is separate from
the high-level builder's AOTI/TensorRT recipe backends.

The external [GeoTransolver example](geotransolver-workflow.md#build-checks-and-outputs)
imports configuration and weights into its project's `weights/` directory,
then uses the generic build layout above. Its adapter supplies synthetic
feature inputs for cached-core native parity.
See the [input contract](model-inputs.md#retained-build-inputs) for format-2
source details and the [project guide](model-build-projects.md) for lock rules.

## Package layout migration

New builds write `model/backends/<backend>/`, such as `backends/aoti/` and
`backends/tensorrt/`. Earlier builds used
`model/variants/<backend>/stages/predictor.pnmir/`, followed by the intermediate
`model/backends/<backend>.pnm-model/` layout. Both existing `.pnmir` and
`.pnm-model` package directories remain loadable without conversion because
loading uses `model.json`, not a suffix. The manifest fields and compiled
artifact formats have not changed. Frozen regression fixtures retain their
historical paths and bytes.

The JSON `variants` keys in `build.json` and `model-release.json` remain
unchanged for compatibility. Default package and payload paths change; no schema
keys are renamed. For example,
the release inventory's package-selection excerpt is now:

```json
{
  "variants": {
    "aoti": {"package": "backends/aoti"},
    "tensorrt": {"package": "backends/tensorrt"}
  }
}
```

New payloads also sit directly beside `model.json`, without an `artifacts/`
layer. AOTI records `model.pt2`, TensorRT records `model.plan`, and ONNX import
records `model.onnx` in the manifest's artifact `path`. The JSON `artifacts`
array remains part of the manifest schema; its name does not require an
`artifacts/` directory.

Older manifests referencing nested paths such as `artifacts/aoti-cuda/model.pt2`,
`artifacts/aoti-cpu/model.pt2`, `artifacts/tensorrt/model.plan` or
`artifacts/onnx/model.onnx` still load without rearranging their files. The
runtime follows the manifest path, so preserve the layout declared by an
existing package rather than moving its payload alone.

Consumers should resolve the selected `package` path from the receipt or
release inventory, then resolve its manifest's artifact paths within that
package. Avoid hardcoding either package directories or payload locations.
Release package paths are relative to `model/`; build receipts retain their
explicit `package_base`. These layout changes do not qualify a different
runtime/toolchain combination.

## Qualification and failures

The builder exports/compiles once per requested backend and runs every required
case through the actual native runtime. It checks completion, selected backend
and device, output names/dtypes/shapes/byte counts, finite values and numerical
parity against independently computed Python references. Current native limits
require both maximum absolute error and relative L2 error to be at most `1e-4`.

A failed requested backend or required qualification gate prevents a complete
candidate manifest and retains available diagnostics. Partial artifacts are
not a successful candidate. The GeoTransolver project checks core eager/native
agreement on synthetic feature inputs. It does not perform full-upstream/core
or full-upstream/native comparisons.

The standard-library host completion checks rehash returned files and validate
evidence coverage. They do not rerun inference on the host. Receipts are
execution evidence, not signatures; generic parity and the current Geo cases
do not establish physical CFD acceptance on arbitrary geometries.

## Deployment and compatibility

Copy the complete `model/` subtree to retain its release inventory and any
qualification reports. Each `backends/<backend>/` directory can also
be copied and loaded independently, preserving its internal artifact paths.
The `physicsnemo-infer` CLI and C++ API load that package directory directly;
there is no intervening `stages/` directory. The SDK does not interpret
`model-release.json` as an executable multi-stage workflow. See
[SDK usage](../cpp-runtime/README.md).

The compiled package runs without the authoring project, Python adapter, original
checkpoint or builder tools. It still requires its declared input tensors and
a compatible native SDK/dependency stack. GeoTransolver's stage consumes
cached-feature tensors. The example's synthetic inputs verify the core
deployment path; applications need to supply meaningful geometry-derived
features separately.

Docker is optional on a GPU server supplying those dependencies. The current
SDK installation does not bundle the complete GPU dependency closure. AOTI
needs compatible native Torch libraries; TensorRT and ONNX Runtime need their
matching native libraries. Core static/shared relocation checks do not qualify
a complete GPU distribution.

Moving a package does not qualify another OS, compiler/standard-library ABI,
GPU, CUDA/driver or backend version. Build and test variants for the deployment
combinations you intend to support. Exact GPU capability checks are one gate,
not a compatibility matrix. Published SDK/model bundles, a tested compatibility
catalog and promotion tooling remain [release work](releasing.md).

The native source/public interface now uses `physicsnemo::inference`, headers
under `physicsnemo/inference/`, CMake package `PhysicsNeMoInference`, and the
`physicsnemo-infer` executable. Existing `.pnmir` packages retain their loading
contract; Python modules keep the names `pnmir_build`/`pnmir_export`.
A source/API rename alone does not establish binary compatibility; rebuild
customer C++ integrations and qualify the resulting runtime against the packages
they will execute. See [SDK build and usage](../cpp-runtime/README.md).
