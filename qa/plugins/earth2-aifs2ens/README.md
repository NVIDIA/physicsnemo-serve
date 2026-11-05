# AIFS2ENS single-GPU H100 qualification

Qualify the initial plugin by comparing a **complete ensemble on one GPU** with
standalone, unmodified Earth2Studio `run.ensemble()`. Both paths use one common
frozen input, `Zero()` initial perturbation, `model.seed=None`, and one initial
`torch.manual_seed(seed_base)` call. Use the same batch size, forecast length,
member count and output fields. Per-member independently seeded references do
not qualify this shared-stream workflow.

The current `earth2-gpu` release, source `4b91b64806d9`, passed **412 selected
tests: 405 portable and all seven H100 cases**, with zero skips. The Linux image
build also passed the 405 portable cases, with two unrelated FCN GPU cases
deselected. Eight candidate rollouts passed 7,440 member/step/field comparisons,
including both Zarr backends, serial and uneven batches, cached A/B/A reuse and
the single-task REST path.

Three separate forecasts through the private Lepton HTTPS endpoint passed
2,880 comparisons, with full A, serial B and repeated A using one cached GPU
worker and one task/release per request. Eight field hashes from a downloaded
`t2m` Zarr ZIP matched the output store. The nine standalone Earth2Studio
reference processes were originally run under image `b58eeaf61c78` / source
`53da50741e74`; the current image reused those arrays after scientific/runtime
identity checks, 30 array-hash checks and 5,280 holdout comparisons. No new
reference generation, recalibration or limit widening is claimed for the rerun.

The [execution record](../../../docs/plans/aifs2ens-parity-and-reference-workflows.md#execution-evidence)
binds the release, input, source, references and collected reports. The earlier
366-test/seven-H100-case baseline and fan-out runs remain separate evidence;
their case counts are not added to the current endpoint validation. Parity is
within native CUDA variability for the tested fixture and layouts.

The qualified image includes the asset-warmup, batch-admission and PR-readiness
changes. A QA-only bounded wait for structured result finalization corrected
the first image's REST polling race; it polls the same accepted run ID and
changes no forecast, RNG or model code. That failed qualification remains in the
execution record. The image fingerprint binds the qualification snapshot.
The subsequent PR-review changes use the SDK cancellation exception and resolve
relative paths in standalone input capture. CPU regressions cover both fixes;
these changes and later documentation updates are outside the qualified image.
Forecast arithmetic, upstream ensemble orchestration and RNG calls are unchanged.

## Runtime

Run all actual CUDA/model execution on the configured remote H100. Keep model
checkpoints, weather snapshots, environments, stores and generated reports
outside versioned source. The independent capture and reference scripts import
Earth2Studio directly and never import the plugin.

The probe runtime uses Python 3.12, NVIDIA Torch
`2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA 13.1 and FlashAttention
`2.7.4.post1+nv26.1.42222806`. The
[h100-runtime-additions.lock](h100-runtime-additions.lock) overlay pins
Earth2Studio `f8af4030cfd43ec79bf4c0cf6c23463a87bfef97` and Anemoi Models 0.11.2
without replacing the host Torch/CUDA/FlashAttention stack. Use a task-specific
venv with `--system-site-packages` when reproducing outside the QA image.
Record the resolved environment, base image and GPU/driver identity.

The current manifest and scheduler profile use `earth2-gpu`. Configure the
AIFS-enabled deployment's `python_runtime_envs["earth2-gpu"]` with this pinned
runtime and the shared artifact environment. Set
`launch.tags: [earth2, aifs2ens, gpu]`. The stock Earth2Studio 0.13 image remains
insufficient. The task still reserves one GPU and configured
`AIFS2ENS_GPU_MEMORY_MB`; preparation runs as a service hook and adds no CPU
execute worker. The published release below has been validated through the
renamed route on the deployed H100 endpoint. The archived baseline image in the
execution record retains its original executor class.

In the current source, `AIFS2ENS_MAX_BATCH_SIZE` is a positive deployment
setting with default `2`. Admission checks the normalized effective batch
before weather downloads and again before GPU execution. Larger ensembles
remain possible with explicit batches of 1 or 2. Raising the limit requires
new resource and parity qualification with a matching GPU-memory reservation;
it does not automatically adjust the request's batch layout. Asset verification
remains in model loading, once per cached initialization, without a duplicate
module-level warmup hook.

Build both Python extensions and service binaries from the same source used by
the tests. The QA Dockerfile performs these builds and provides Redis for the
localhost REST case. A manual configured-host build uses:

```sh
export PATH="/root/.cargo/bin:/usr/local/cuda/bin:$PATH"
uv pip install --python "$task_root/venv/bin/python" --no-deps \
  -r "$task_root/source/qa/plugins/earth2-aifs2ens/h100-runtime-additions.lock"
uv pip install --python "$task_root/venv/bin/python" \
  "$task_root/source/crates/scicomp-rq" "$task_root/source/crates/e2s_zarr_io"
cargo build --locked -p inference_server -p worker-runtime
```

The portable suite uses an isolated environment without optional model packages;
the actual H100 suite uses the pinned AIFS2ENS runtime. The two FCN spherical
Gaussian GPU tests belong to their separate runtime and are explicitly
deselected from this plugin's regression job. Selected AIFS2ENS tests must not
be skipped or silently omitted.

Install the existing QA extra for the portable tests. It includes Xarray/Zarr
and `pip`, which the shared native-module bootstrap invokes in fresh uv
environments. The Python CI job uses the same frozen extra:

```sh
uv run --frozen --extra qa pytest -q \
  -k 'not spherical_gaussian_skip_ahead_matches' \
  tests/test_aifs2ens_*.py tests/test_earth2_plugins.py \
  tests/test_plugin_dev.py tests/test_runtime_env_launcher.py
```

## Frozen inputs and source identity

Use a task-owned directory below `/mnt/my_lfs_data/Projects`, inspect the name
before creating it, and preserve other tasks' files. Checkpoint
`aifs-ens-crps-2.0.ckpt` comes from `ecmwf/aifs-ens-2.0` at revision
`29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf`; required SHA-256 is
`e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb`.

The artifact setup must include:

- A model-derived `qualified-input-contract.json` with exact variable/grid order
  and checkpoint, invariant and both interpolation-matrix hashes.
- A frozen float32 `assets/invariants.npy` with shape `[5,721,1440]` in order
  `lsm`, `sdor`, `slor`, `z`, `wmb`.
- The pinned checkpoint directory and configured model/weather caches.
- A common two-time weather capture for an available six-hour UTC cycle.
  `ifs_member_id` optionally chooses one IFS ENS source for the entire ensemble.

The qualified model interface uses `preload_invariants=False`, with 112 input
and 120 output fields. The independent capture helper freezes unchanged
Earth2Studio weather data at initialization minus six hours and initialization,
combines the fixed invariant inputs, and records the SHA-256 identity. It adds
no ECMWF notebook ocean mask. Use the scripts' `--help` for their exact artifact
arguments; `run_qualification.py` creates a fresh fixture and invokes capture
before generating the new ensemble references. Never regenerate old fixture
bytes under their historical identity.

The image build context contains `qa-source-inventory.json`: base commit plus
relative `path`, `bytes` and `sha256` records for each included source file.
`SOURCE_CONTENT_SHA256` hashes the canonical
`json.dumps(files, sort_keys=True, separators=(",", ":"))` bytes. Both image build
and job startup verify the inventory.

From the repository root, prepare a reviewed working tree into a **new directory
outside the repository**. The checked-in helper includes modified tracked files
and nonignored untracked source, so the initial plugin need not be committed to
build it. It excludes local credentials (`.env*`, private keys), caches, model
assets, local deployment configuration and old inventories; it rejects symlinks
and source files larger than 10 MiB. It refuses an existing output path.

```sh
PREPARED_SOURCE_CONTEXT=/absolute/task-owned/path/source-context-v1
SOURCE_CONTENT_SHA256="$(python qa/plugins/earth2-aifs2ens/prepare_source.py \
  --output "$PREPARED_SOURCE_CONTEXT")"
```

The helper prints only the fingerprint used by the image build argument. Keep
the complete prepared context, inventory, exact build command/log and immutable
image digest with the evidence. When building on another machine, transfer the
prepared context including its inventory and set `PREPARED_SOURCE_CONTEXT` to
that directory on the builder; retain the printed fingerprint unchanged.

## Linux build and Lepton execution

Build the [QA image](Dockerfile) on the authorized Linux builder. GPU execution
runs in the Lepton H100 job, not in the CPU image build:

```sh
docker buildx build --platform linux/amd64 --load \
  --file "$PREPARED_SOURCE_CONTEXT/qa/plugins/earth2-aifs2ens/Dockerfile" \
  --target release \
  --build-arg SOURCE_CONTENT_SHA256="$SOURCE_CONTENT_SHA256" \
  --tag "$QA_IMAGE_TAG" "$PREPARED_SOURCE_CONTEXT"
```

`release` is the Dockerfile's default final stage. It exposes port 8001 and
starts `python qa/plugins/earth2-aifs2ens/serve.py` by default. Set
`AIFS2ENS_SERVICE_ROOT` to a writable mounted directory and provide the artifact
environment from the [plugin guide](../../../plugins/earth2-aifs2ens/README.md#runtime-and-artifacts).
`PORT` defaults to 8001. The supervisor verifies the image source inventory,
creates a new service session with separate logs and observation snapshots,
allows 600 seconds for warmup or recovery, publishes readiness atomically, and
terminates owned process groups on shutdown or failure.

The supervisor passes `INFERENCE_READINESS_FILE` to the HTTP server. Where the
orchestrator supports HTTP readiness, configure `GET /readyz` on the serving
port. It returns 503 until the GPU worker is available and during recovery;
Redis connectivity is also required. Missing, malformed or unreadable readiness
JSON fails closed. Keep `/healthz` or TCP for liveness, so warmup does not trigger
restarts. The deployed gate passed a direct HTTP test: 200, then 503 while the
ready record was held, then 200 after restoration; liveness stayed 200.
Lepton's current API provides TCP liveness and readiness approval only. The
validated deployment has no native platform HTTP readiness probe.

Worker observation snapshots contain only the latest 100 release entries in
stream order. Polling and snapshot size stay bounded as the service handles
requests; the underlying Redis release stream is not trimmed by this observer.

The validated release tag is
`nvcr.io/dycvht5ows21/physicsnemo-serve:aifs2ens-release-20261105-4b91b64806d9`;
deploy its immutable digest
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:273bf30c623a160cf857191a041b01ddd210cbd5b6c7540eca7b734c6d364f92`.
The private [Lepton endpoint](https://r0whe339-aifs2ens-1105-4b91b64806.xenon.lepton.run)
uses workspace-token authentication. For an authenticated probe, keep the URL
and token in environment variables:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${LEPTON_WORKSPACE_TOKEN}" \
  "${AIFS2ENS_ENDPOINT_URL}/healthz"
```

For a separate complete qualification job, submit the pushed immutable digest
to one H100 in the authorized workspace,
mounting the frozen assets and a new task-owned output directory. Use zero job
retries and record the resolved job ID before monitoring. Run through the NVIDIA
entrypoint:

```sh
/opt/nvidia/nvidia_entrypoint.sh python \
  /opt/physicsnemo-serve/qa/plugins/earth2-aifs2ens/run_qualification.py \
  --assets "$FROZEN_ASSETS" \
  --output "$NEW_QUALIFICATION_OUTPUT" \
  --image "$QA_IMAGE_DIGEST" \
  --source-content-sha256 "$SOURCE_CONTENT_SHA256"
```

The prepare hook's weather download can take minutes. Its configured timeout is
7200 seconds; the dev kit maps this to
`roles.prepare.config.hook_timeout_secs`. Static GPU launch profiles prevent
plan construction from invoking the downloading hook. The REST case must show
one scheduled ensemble execution, without fan-out child IDs or a member merge.
The test owns a private localhost Redis, service and workers and must terminate
those process groups after the request.

## Reference calibration and acceptance

The standalone reference calls stock Earth2Studio `run.ensemble()` with the
pinned native model, frozen common datasource and identity initial perturbation.
It seeds the shared stream once after model loading and leaves `model.seed=None`.
The plugin uses the same numerical path inside an RNG-isolation context. No
per-member seed offsets, per-step forced reseeding, CPU noise projection or
custom math flags are allowed in either path.

A reference configuration binds its common input, invariant/checkpoint bytes,
seed, `nensemble`, `batch_size`, `nsteps`, output fields and runtime settings.
Different batch layouts require separate references. A complete mini-batch runs
all its steps before the next mini-batch starts; the seed alone is not enough to
identify the random draw/member mapping.

Before any candidate executes, create four independent calibration references
and two held-out references per tested configuration. Per-member/per-step/per-field
maximum absolute error and RMSE limits are twice the largest observed pairwise
reference values; the factor is fixed in source before execution. Zero observed
variation retains zero limits. A predefined reference-only fallback may expand
to eight calibration processes and two new holdouts. Failed final holdouts stop
qualification. Freeze and hash the passing calibration before candidate tests;
never relax it after a candidate failure.

Compare every emitted member, forecast step and model field. Reference output
may include the initial yield; align only +6h through +6×`nsteps` for the
plugin's forecast-only store. Coordinates, complete member coverage/order,
float32 dtype and NaN/Inf masks remain exact. Both maximum absolute error and
RMSE must pass the frozen limits. Relative errors are diagnostic and include
the recorded denominator floor. Passing means agreement within measured native
CUDA variability for that fixture and runtime, not a bitwise or operational
ECMWF equivalence claim.

Required coverage includes full two-member batches over four steps, sequential
mini-batches with a matching reference, both Zarr backends, whole-ensemble
cached-model A/B/A reuse and the one-task REST route. Include an uneven final
mini-batch where resources permit and explicitly record any unexecuted scope.
Only a passing complete JUnit case inventory, reference gate and summary qualifies
the revision. Full forecast arrays may use ephemeral scratch; their hashes,
reference reports, calibration, per-field comparisons and JUnit must persist.

`summary.json` must identify the image, source fingerprint, GPU, input contract,
shared seed policy, case coverage and passing/failing/skipped status. Preserve
all failed runs beside the final reports. A completed cloud job alone is not a
pass. Copy and verify the evidence archive before claiming qualification.

Further ECMWF/Anemoi preparation comparisons, additional dates, larger batches,
60-step horizons and decoded/accumulated GRIB products remain separate gates in
the [reference plan](../../../docs/plans/aifs2ens-parity-and-reference-workflows.md).
