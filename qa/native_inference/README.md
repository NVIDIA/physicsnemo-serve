# Native inference QA on Lepton

This suite qualifies the public Model Builder and standalone C++ Inference SDK.
It runs a finite Lepton job with persistent evidence. Serve, Redis, HTTP endpoints,
and the service Docker publishing checks are not involved.

| Profile | Required coverage |
| --- | --- |
| `smoke` | Configured-affine format-2 project, AOTI and TensorRT CUDA builds, checkpoint/config/asset handling, independent analytical output checks, relocated packages, installed C++ SDK consumer, and failure checks. |
| `full` | All smoke checks plus real Transolver surface builds and C++ inference through both AOTI and TensorRT, raw VTP/STL geometry, C++ preprocessing and physical decoding, and an independent eager Python reference. |

All required stages must run and pass. A missing GPU, backend, dependency, or
full-profile asset fails the job. Numerical parity qualifies the tested software
and fixtures; it does not establish scientific CFD accuracy or another GPU's
compatibility. Timings are evidence rather than performance pass/fail thresholds.

The builder stage verifies its saved numerical output bytes and tensor metadata,
then copies the completed packages into `consumer/`. Before consumption, the job
moves its original project/checkpoint/build directories into `producer/`, retaining
diagnostics while making the original producer paths unavailable. The installed
C++ CLI runs held-out cases from the relocated package. A separate SDK executable
reuses one executor for three distinct requests and checks retained output storage.
Both stages share the image's native libraries; this is not a minimal-runtime-image
qualification. Full-profile eager references read the separately mounted original
checkpoint and geometry.

## Build the dedicated QA image

Use the pinned Model Builder image recipe and an explicit target. Its regular
default still builds Model Builder. Build from a clean committed checkout so the
recorded source SHA describes the copied SDK, builder, workflow, and QA sources:

```bash
git diff --exit-code
git diff --cached --exit-code
SOURCE_SHA="$(git rev-parse HEAD)"
QA_IMAGE="YOUR_REGISTRY/YOUR_NAMESPACE/physicsnemo-native-qa:${SOURCE_SHA}"

docker build --platform linux/amd64 \
  --file native-inference/model-builder/images/Dockerfile.builder \
  --target native-qa \
  --build-arg SOURCE_SHA="$SOURCE_SHA" \
  --build-arg PNMIR_ENABLE_TENSORRT=ON \
  --build-arg CMAKE_CUDA_ARCHITECTURES=90 \
  --tag "$QA_IMAGE" .
```

The architecture build argument is required. The QA configuration targets H100
with `90`; use `80` for A100 or `80;90` to include both. Select architectures
supported by the image's CUDA toolchain and the matching Lepton resource shape. No GPU is needed to
construct the image. This suite runs CUDA qualification in the Lepton GPU job.

The QA target builds and installs both native backends from the same checkout,
including the native exact TensorRT plugins required by Transolver,
compiles a separate downstream C++ consumer against that installation, and builds
`physicsnemo-transolver`. It installs native and Python VTK from the same distro
package family and checks their versions and binding origin before producing
`/opt/physicsnemo-qa/source.json`. It verifies the pinned Torch/CUDA environment
again after these dependencies are installed. The full source SHA and selected
CUDA architectures are recorded in the image; the SHA is a build input, not an
independent source attestation.

The NVIDIA initialization entrypoint is retained. Its default command is
`python3 /opt/physicsnemo-qa/run_job.py`. Lepton's command override can bypass
the image entrypoint, so the launcher explicitly runs
`exec /opt/nvidia/nvidia_entrypoint.sh python3 /opt/physicsnemo-qa/run_job.py`
with the job arguments. This preserves NVIDIA driver/CUDA initialization and
forwards container signals to the QA process. Model checkpoints and geometry are mounted separately, never baked
into this image. Keep untracked output/data out of the build context; the
Dockerfile's dedicated ignore file excludes common model and geometry formats.

After publishing the image through your normal authorized registry workflow,
use its immutable `repository@sha256:<64 hex digits>` reference below. Mutable
tags are rejected by the launcher. The Lepton pull secret must have access to
that repository.

## Run a smoke job

Install the Lepton CLI (`uv tool install leptonai`) and configure the workspace
token through your existing secret mechanism. Supply these settings without
putting tokens on the command line:

| Setting | Purpose |
| --- | --- |
| `LEPTON_WORKSPACE_ID`, `LEPTON_WORKSPACE_TOKEN` | Workspace identity and API access. |
| `LEPTON_NODE_GROUP` | Authorized GPU node group. |
| `LEPTON_PULL_SECRET` | Existing private-registry pull-secret name. |
| `NFS_MOUNT_BASE` | Shared filesystem mount base used by existing Lepton QA. |
| `QA_LUSTRE_DIR` | Directory under that base mounted at `/outputs`; default is `qa`. |

An existing `lep login` session can supply authentication when the token environment
variable is absent; pass `--workspace-id` or set `LEPTON_WORKSPACE_ID` to select it.
Node group, pull secret, and NFS base can also come from `deploy/config.yaml`, using
the same conventions as the existing Lepton QA runners. No physical node is pinned.

The launcher and manual workflow default to one H100 with `gpu.h100-sxm`.
Select a node group containing H100 GPUs. Override the shape with
`LEPTON_RESOURCE_SHAPE` or `--resource-shape` (CLI takes precedence), and build
the image for the matching CUDA architecture when changing GPU types.

```bash
python3 qa/scripts/run_lepton_native_inference_qa.py \
  --image 'YOUR_REGISTRY/YOUR_NAMESPACE/physicsnemo-native-qa@sha256:YOUR_DIGEST' \
  --expected-source-sha "$SOURCE_SHA" \
  --profile smoke \
  --resource-shape gpu.h100-sxm \
  --artifact-dir qa/artifacts/native-inference \
  --job-timeout 3600 \
  --dry-run
```

Remove `--dry-run` to submit and wait for the finite job. Use `--run-id` to
provide a distinct run identifier when required. Job evidence is written under
`/outputs/native-inference/<run-id>/`; the launcher collects its summary and
diagnostics under `qa/artifacts/native-inference/<run-id>/`. Evidence persists
after the job is removed. `--keep-job` retains the job for diagnosis. A remote
failure, timeout, missing evidence, wrong source SHA, or failed required result
makes the launcher exit nonzero.

The remote run retains `summary.json`, `junit.xml`, `handoff.json`,
`environment.json`, `logs/`, `producer/`, and `consumer/`. The controller validates
the exact expected case inventory, source SHA, image digest and both stage results;
a successful Lepton job state alone is insufficient. Full tensors and JUnit remain
on the shared filesystem; the local artifact upload contains the collected summary
and job diagnostics, not an implicit download of every remote file.

## Add the real Transolver qualification

Run the same launcher with `--profile full`. In the same Lepton job, the
`transolver.assets` check downloads the four pinned public assets, verifies their
expected sizes and SHA-256 hashes, and generates an asset manifest. It then runs
the build and C++ consumer stages. No manual download, Hugging Face token, or
second job is required.

| Asset | Download size |
| --- | ---: |
| Surface `Transolver.0.501.mdlus` | 39,188,235 bytes (37.4 MiB) |
| Matching `global_stats.json` | 1,375 bytes |
| `run_1/boundary_1.vtp` | 659,606,189 bytes (629.0 MiB) |
| `run_1/drivaer_1.stl` | 142,385,186 bytes (135.8 MiB) |

The first run downloads about 841 MB (802 MiB). The model and statistics come
from `nvidia/transolver_drivaerml` at revision
`96477aeb86d24c26ccf0797bca1b3851268017d0`; geometry comes from
`neashton/drivaerml` at revision `5d448b209bf654503c64ce7261c34fa125f46392`.
The source pins and expected hashes are recorded in [assets.py](assets.py).
Downloads are streamed into temporary files and published only after validation.
The versioned cache lives under `/outputs/assets/transolver/` (or the selected
mount target), outside each run's build directories. Subsequent runs recheck
cached bytes and reuse them without network access; missing or corrupt files are
downloaded again. A download or integrity failure fails `transolver.assets`.

To use an existing asset set without automatic downloads, pass
`--assets /outputs/assets/transolver/manifest.json`. Each SHA-256 must identify
the exact file bytes you intend to qualify; replace every illustrative value:

```json
{
  "format_version": 1,
  "revision": "YOUR_PINNED_ASSET_REVISION",
  "checkpoint": {
    "path": "Transolver.0.501.mdlus",
    "sha256": "REPLACE_WITH_64_HEX_DIGITS"
  },
  "stats": {
    "path": "global_stats.json",
    "sha256": "REPLACE_WITH_64_HEX_DIGITS"
  },
  "vtp": {
    "path": "boundary_1.vtp",
    "sha256": "REPLACE_WITH_64_HEX_DIGITS"
  },
  "stl": {
    "path": "drivaer_1.stl",
    "sha256": "REPLACE_WITH_64_HEX_DIGITS"
  }
}
```

Relative paths resolve against the manifest directory inside the container.
The `revision` label is descriptive; the file hashes enforce the identities.
Allow a larger `--job-timeout`, such as 7200 seconds, for the first full run.
The bounded cases use 75 and 161 surface cells,
75-point blocks, and seed 0. Each static shape (75 and 11 points) is built for
both AOTI and TensorRT, producing four packages from the same pretrained weights.
AOTI uses `aten-boundary-exact-v2`; TensorRT uses `layout-order-exact-v2` with
nine exact plugins copied from the installed SDK into the project assets. The
builder verifies both backends against Python before package handoff.

The C++ workflow then runs each backend on 75 cells and on 161 cells split into
`75 + 75 + 11`. Each backend must also reject the 161-cell case when its 11-point
tail package is absent. Backend-specific evidence directories and metadata checks
ensure one backend cannot satisfy the other's cases. The independent reference
reads the original checkpoint and raw geometry, then compares prepared tensors,
standardized output, and physical pressure/shear output against both C++ runs.
Native and Python VTK versions must match. The full profile requires all 33 cases
to pass; neither backend is optional. This is bounded surface qualification,
not whole-mesh, volume-model, or scientific CFD validation.

## Manual GitHub Actions entry point

Run **Native Inference Lepton QA** from
[lepton_native_qa.yml](../../.github/workflows/lepton_native_qa.yml), supplying
the pushed image digest, full source SHA, profile, resource shape, and optional
full-profile asset manifest path. The workflow checks out that source revision,
invokes the same launcher, and uploads local evidence even when qualification
fails. It does not build or publish an image and is not tied to the service
image's checks.

Configure repository secrets `LEPTON_WORKSPACE_TOKEN`, `LEPTON_WORKSPACE_ID`,
`LEPTON_NODE_GROUP`, and `PULL_SECRET`, plus variables `NFS_MOUNT_BASE` and
`QA_LUSTRE_DIR`. There is no implicit nightly schedule or pull-request trigger.

## Local harness verification

These tests exercise orchestration and failure handling without a GPU or cloud
resource creation:

```bash
uv run pytest tests/test_native_qa_image.py tests/test_native_qa_workflow.py
uv run pytest tests/test_run_lepton_native_inference_qa.py \
  tests/test_native_inference_qa_contract.py tests/test_native_inference_qa_job.py
```

Run `tests/test_native_inference_transolver_qa.py` in a CPU environment with Torch,
NumPy, and VTK to check independent geometry and batching references. To exercise
the downstream SDK consumer locally, build/install the core SDK, configure
`qa/native_inference/cpp` with `-DPNMIR_QA_WITH_CUDA=OFF` and that SDK prefix, then
set `NATIVE_QA_CONSUMER` to the resulting executable when running
`tests/test_native_inference_qa_consumer.py`. That local test uses the mock backend;
the Lepton suite always requires the real AOTI and TensorRT CUDA backends.

A passing local harness test does not count as CUDA or image qualification.
