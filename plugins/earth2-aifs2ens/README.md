# Earth2Studio AIFS ENS v2

Run a complete AIFS2ENS ensemble on **one GPU execute worker** through stock
`earth2studio.run.ensemble()`. The prepare hook freezes one common weather input;
the GPU worker generates the ensemble with a shared seeded RNG stream and writes
one registered Zarr dataset. Fan-out is deferred to a separate future version.

The current `earth2-gpu` release, source `4b91b64806d9`, passed **412 selected
tests: 405 portable and all seven H100 cases**, with zero skips. Eight candidate
rollouts passed 7,440 member/step/field comparisons. Three separate forecasts
through the private Lepton HTTPS endpoint passed another 2,880 comparisons:
full-batch A, serial-batch B, then A again used one cached GPU worker, with one
task and release per request. All eight downloaded `t2m` field hashes matched
the output store. These results establish parity within measured native CUDA
variability for the tested fixture and layouts, not bitwise equality.

The release includes the asset-warmup, batch-admission and PR-readiness fixes.
Its QA client also waits for structured result finalization on the same accepted
run ID. Nine standalone Earth2Studio references originally generated under
source `53da50741e74` were reused after runtime/scientific-source identity
checks, 30 array-hash checks and 5,280 holdout comparisons; no comparison limits
were recalibrated. The [parity record](../../docs/plans/aifs2ens-parity-and-reference-workflows.md)
binds both image provenances and preserves earlier releases, including their
failed runs, separately. The qualified fingerprint binds the image snapshot.

The later PR-review follow-up uses the SDK cancellation exception so cooperative
aborts retain `cancelled` status, and resolves relative asset paths in the
standalone capture helper. CPU regressions cover both fixes. These changes and
the documentation updates are outside the qualified image; forecast arithmetic,
the upstream ensemble call and RNG calls are unchanged.

## Request

See [examples/default_request.json](examples/default_request.json). A representative
request is:

```json
{
  "start_time": "2026-11-04T12:00:00Z",
  "nsteps": 4,
  "nensemble": 2,
  "batch_size": 2,
  "seed_base": 0,
  "ifs_member_id": 1,
  "output_variables": ["t2m", "msl", "u10m", "v10m"]
}
```

Use an available six-hour UTC initialization cycle. Both initialization minus
six hours and initialization must be available. `ifs_member_id` optionally
selects **one** IFS ENS source member whose input is shared by the whole AIFS2ENS
ensemble. Omit it, or use `null`, to fetch the common input through `IFS`.
It does not assign a different IFS input to each output ensemble member.

`nensemble` accepts 1–16 members and `nsteps` accepts 1–60 six-hour forecast steps.
`batch_size` controls how many members Earth2Studio processes together on the
same GPU. Omission defaults to `nensemble`; values above `nensemble` are clamped
like upstream. The current source rejects an effective batch above
`AIFS2ENS_MAX_BATCH_SIZE` (default `2`) before downloading inputs and rechecks
the limit before GPU execution. For larger ensembles, specify `batch_size: 1`
or `2`; the deployment limit never silently changes the requested layout.
Raising the limit requires qualification of that batch size, its GPU memory
reservation and parity. Seed zero is valid. There is no `max_in_flight` or
per-member seed argument.

The default output budget is 4 GiB and is checked before weather downloads.
Output selection happens after model inference. Forecast output contains
+6-hour increments and excludes the iterator's initial, lead-zero yield.

## Shared RNG and native model behavior

The RNG policy is `torch-seed-once-native-v1`. Inside a request-scoped RNG context,
the worker sets `torch.manual_seed(seed_base)` once immediately before calling
Earth2Studio's ensemble workflow. It temporarily sets `model.seed=None`, so the
AIFS2ENS wrapper does not reset the RNG before each forward call. The stream
advances through the forecast steps and ensemble mini-batches in upstream order.
Reference and plugin must use the same complete request, including `batch_size`
and `nsteps`, because that order determines which draws each member receives.

Earth2Studio receives `Zero()` as its initial-condition perturbation. Diversity
comes from AIFS2ENS's native internal stochastic conditioning on CUDA; the plugin
does not add FCN-style Gaussian input noise. BF16 autocast, sparse projection,
model-grid interpolation and forcings remain upstream operations. The numerical
policy is `earth2studio-native-v1`; model layers and math flags are unchanged.
The previous model seed and caller CPU/CUDA RNG states are restored after success
or failure, so a cached worker can run another ensemble without inheriting the
previous request's RNG state.

## Runtime and artifacts

Use the pinned Earth2Studio revision
`f8af4030cfd43ec79bf4c0cf6c23463a87bfef97`, Anemoi Models 0.11.2 and the qualified
Linux Torch/CUDA/FlashAttention stack. The repository's general Earth2Studio
0.13 runtime does not provide this model. The [H100 guide](../../qa/plugins/earth2-aifs2ens/README.md)
describes the pinned runtime overlay and image build.

The plugin uses the existing `earth2-gpu` executor class for execution and phase
runtime selection. In an AIFS-enabled deployment, configure
`python_runtime_envs["earth2-gpu"]` with the qualified interpreter, artifact
environment and `launch.tags: [earth2, aifs2ens, gpu]`. Keep the `aifs2ens`
capability tag so older `earth2-gpu` workers cannot accept this task. The class
name selects a configured runtime; it does not install the AIFS dependencies or
assets. Workers already warmed for another workflow are excluded by the
scheduler, even when they use the same executor class.

| Variable | Artifact or setting |
| --- | --- |
| `AIFS2ENS_INPUT_CONTRACT` | JSON binding model variable/grid order and checkpoint, invariant and interpolation hashes |
| `AIFS2ENS_INVARIANTS` | Frozen float32 array `[5, 721, 1440]`: `lsm`, `sdor`, `slor`, `z`, `wmb` |
| `AIFS2ENS_CHECKPOINT_DIR` | Directory containing the pinned `aifs-ens-crps-2.0.ckpt` |
| `AIFS2ENS_GPU_MEMORY_MB` | GPU admission reservation for the complete ensemble task and its chosen batch size |
| `AIFS2ENS_MAX_BATCH_SIZE` | Optional positive qualified GPU batch limit; default `2` |
| `EARTH2STUDIO_MODEL_CACHE` | Shared model/interpolation cache |
| `EARTH2STUDIO_DATA_CACHE` | Shared weather-data cache |
| `AIFS2ENS_SERVICE_ROOT` | Writable root for release-service sessions, readiness records and logs |

The release supervisor connects its GPU-worker readiness record to HTTP
`/readyz` through `INFERENCE_READINESS_FILE`. Where the orchestrator supports
HTTP readiness, use `GET /readyz`, with `/healthz` or TCP for liveness. The
deployed gate was tested directly: readiness returned 200, then 503 while its
record was held, then 200 after restoration; liveness stayed 200. Lepton's
current API supplies TCP liveness and readiness approval, without a native HTTP
readiness probe. See the
[QA guide](../../qa/plugins/earth2-aifs2ens/README.md#linux-build-and-lepton-execution)
for source preparation and release-service configuration.

Optional settings are `AIFS2ENS_SOURCE` (`ecmwf`, `aws`, `google`, `azure`) and
`AIFS2ENS_MAX_OUTPUT_BYTES` (default 4294967296). Configure CUDA compiler tools
and headers when needed, including `TRITON_PTXAS_PATH` and the compiler include
path. Service hooks and the GPU worker need the same absolute artifact paths.

The prepare hook fetches and validates the common input, combines the frozen
invariants under `preload_invariants=False`, and saves a NumPy input, input
contract and SHA-256 manifest under the run directory. The worker verifies those
identities and reuses its resident CUDA model. The default Zarr backend is the
shared Rust extension; `PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python` selects stock
Earth2Studio Zarr. Storage and registration complete only after validation of
all requested members, steps, fields and coordinates.

Model loading verifies the checkpoint, pinned runtime and interpolation hashes
before construction. The current source performs asset preparation once per
cached model initialization; cached warmup does not repeat it. Invalid assets
still prevent model construction.

The final `forecast-ensemble-<uuid>.zarr` contains one float32 array per requested
field with dimensions `(ensemble, time, lead_time, lat, lon)`. Ensemble values
are zero-based output member indices. `forecast_metadata.json` records the
request, common IFS source member, seed, batch size and input/model identities.
The primary artifact is `forecast_dataset`, with media type `application/x-zarr`.

## Local checks and service

```sh
python scripts/plugin_dev.py validate plugins/earth2-aifs2ens
python scripts/plugin_dev.py check-env plugins/earth2-aifs2ens
python scripts/plugin_dev.py run-local plugins/earth2-aifs2ens --dry-run
```

After configuring assets and an available cycle, exercise input preparation:

```sh
python scripts/plugin_dev.py check plugins/earth2-aifs2ens --through-phase prepare
```

Launch the service on the configured GPU host with:

```sh
python scripts/plugin_dev.py run-local plugins/earth2-aifs2ens --bootstrap-python
```

The manifest declares only GPU execute workers. Preparation is a standard CPU
service hook; there is one scheduled ensemble execution and no child collection
or merge stage. The prepare timeout is 7200 seconds because cold downloads can
take minutes. Static launch profiles let plan generation avoid the downloading
hook. A loaded model is reused across requests on its assigned device.

The [release image](../../qa/plugins/earth2-aifs2ens/README.md#linux-build-and-lepton-execution)
exposes port 8001 and starts `qa/plugins/earth2-aifs2ens/serve.py` by default.
Set `AIFS2ENS_SERVICE_ROOT` and the artifact environment above, and mount the
shared artifacts and writable service storage. The supervisor allows 600
seconds for worker warmup or recovery, writes atomic readiness observations in
a new session directory, and cleans up its process groups on exit. The private
Lepton endpoint requires workspace-token authentication; keep credentials in
environment variables.

Run portable request, staging, RNG/lifecycle and storage checks with:

```sh
uv run --extra qa pytest -q tests/test_aifs2ens_*.py
```

Run all actual model/CUDA tests on the configured remote H100. A complete-ensemble
reference must match the candidate's seed, input setup and batch size. Earlier
per-member reference runs do not establish this parity.

## Scientific scope

Input policy `earth2studio-latlon-frozen-invariants-v1` preserves Earth2Studio's
weather fields and uses its supported frozen-invariant input mode. It adds no
ECMWF notebook ocean mask. Native CUDA reductions can vary numerically; QA
measures reference variation and freezes comparison limits before candidates.

The output contains raw model fields. ECMWF notebook/Anemoi input equivalence,
wave-direction decoding, forecast-start accumulation and operational GRIB
products remain separate scientific gates. Model and Open Data attribution and
terms remain applicable; see the [ECMWF model card](https://huggingface.co/ecmwf/aifs-ens-2.0).
