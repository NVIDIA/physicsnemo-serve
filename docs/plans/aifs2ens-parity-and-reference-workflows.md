# AIFS2ENS single-GPU reference workflow and parity

The initial implementation compares **one complete single-GPU ensemble** with
stock Earth2Studio `run.ensemble()`. Reference and plugin use the same common
initial input, checkpoint, frozen invariants, initial seed, batch size, forecast
length and native numerical settings. This is the baseline for any future
fan-out implementation.

The current numerical policy is `earth2studio-native-v1`, input policy is
`earth2studio-latlon-frozen-invariants-v1`, and RNG policy is
`torch-seed-once-native-v1`. The current `earth2-gpu` release, source
`4b91b64806d9`, passed **412 selected tests: 405 portable and all seven H100
cases**, with zero skips, on 2026-11-05. Eight candidate rollouts passed 7,440
field comparisons. Three separate HTTPS forecasts passed 2,880 comparisons
against standalone Earth2Studio references under the prior immutable calibration. The
[execution record](#current-single-gpu-version) defines the tested scope and
preserves the earlier 366-test/seven-H100-case baseline separately. Parity is
within measured native CUDA variability; earlier fan-out/per-member passes do
not qualify this workflow.

The qualified release includes the asset-warmup, batch-admission and
PR-readiness changes, plus a QA-only fix for structured result finalization.
Its nine reference processes retain their original image `b58eeaf61c78` /
source `53da50741e74` provenance. They were reused after scientific/runtime
identity checks, rehashing 30 arrays and revalidating 5,280 holdout comparisons;
no new reference generation or recalibration is claimed for the final image.

## Reference contract

Pin Earth2Studio to `f8af4030cfd43ec79bf4c0cf6c23463a87bfef97`, Anemoi Models to
0.11.2, and `ecmwf/aifs-ens-2.0` to repository revision
`29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf`. The checkpoint SHA-256 is
`e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb`.
Record the actual Torch/CUDA/FlashAttention environment and GPU/driver identity.
A matching package name alone does not bind the model bytes or numerical stack.

Capture one unchanged Earth2Studio weather input at initialization minus six
hours and initialization. The optional `ifs_member_id` selects one IFS ENS
source member shared by all AIFS output members; omission uses `IFS`. Freeze the
five invariant fields independently and supply them with
`preload_invariants=False`. The qualified input is float32
`[1, 2, 112, 721, 1440]`; derive variable order from the actual loaded checkpoint
and bind coordinates, checkpoint, invariant and interpolation identities in the
input contract. Do not silently substitute another source or initialization.

The reference script imports Earth2Studio directly, without plugin helpers. It
loads AIFS2ENS with `model.seed=None`, seeds Torch **once** with `seed_base`, and
calls stock `run.ensemble()` using the frozen common datasource, `Zero()` and
request batch size. The plugin does the same inside a request-scoped RNG context
and restores caller RNG/model state afterward. Neither path assigns member seed
offsets or adds Gaussian initial-condition noise. The model's native internal
conditioning samples different member draws from the shared CUDA stream.

`run.ensemble()` finishes one mini-batch's rollout before starting the next.
Thus the complete request, especially `batch_size` and `nsteps`, is part of the
reference identity. Compare identical layouts; do not infer that changing
batch size preserves a member's random draws. Default `batch_size=None` resolves
to `nensemble`, with larger provided values clamped to `nensemble` like upstream.

The source follow-up applies `AIFS2ENS_MAX_BATCH_SIZE` after this normalization,
with default `2`. It rejects oversized batches before downloading and rechecks
them before GPU execution, without silently changing the RNG draw/member
mapping. Ensembles larger than two require an explicit batch of 1 or 2 under
the default limit. Any higher deployment limit needs GPU-memory and parity
qualification with its matching memory reservation.

The independent reference must retain all output member indices and fields.
Compare forecast lead times +6h through +6×`nsteps`; exclude the initial lead-zero
yield when aligning the plugin's forecast-only output. Stored fields are raw
Earth2Studio model fields, with exact coordinates, float32 dtype and nonfinite
masks. An initial-state output is not a forecast-parity observation.

## Native variability gate

Native CUDA sparse reductions can vary across identical runs. Preserve the
upstream path and quantify that variability before evaluating a candidate.
For each exact ensemble configuration, run independent reference processes for
calibration and separate held-out processes. The declared protocol begins with
four calibration runs and two holdouts; a reference-only fallback can expand to
eight calibration runs and two new holdouts before any candidate executes.

For every member, forecast step and field, compute the largest pairwise maximum
absolute error and RMSE among calibration runs. Limits are twice those observed
maxima, with the factor fixed in source before execution. Fields with zero
observed variation retain zero limits. All held-out references must pass these
limits. Freeze and SHA-256-bind the passing calibration before plugin tests.
Never widen a limit after observing a candidate failure.

Acceptance requires both maximum absolute error and RMSE within those frozen
limits, plus exact input identity, member coverage/order, coordinates, dtype,
variable selection and NaN/Inf masks. Record relative errors with their
near-zero denominator convention as diagnostics. A numerical envelope cannot
excuse a structural mismatch, missing member, changed seed policy, or changed
batching. Passing establishes agreement within the measured native variability
for that fixture/runtime; it does not assert bitwise reproducibility.

## Qualification cases

| Case | Complete-ensemble comparison |
| --- | --- |
| Shared-stream reference gate | Repeated stock Earth2Studio ensemble runs and held-out validation before candidate execution |
| Full batch | Two members, batch size two, four steps and all 120 model output fields |
| Sequential mini-batches | Two members, batch size one, compared with a matching batch-one reference |
| Uneven final batch | A short three-member, batch-two request, where resources permit; record execution or remaining scope explicitly |
| Storage | Python and Rust Zarr output compared with the same independent ensemble reference |
| Cached model lifecycle | Whole-ensemble requests A, B, then A; preserve reference results and restore caller RNG/model seed |
| REST | One scheduled GPU ensemble execution, successful registered Zarr artifact, and no child fan-out IDs or collection |

The service's small default field selection is not sufficient full-model
qualification coverage. Cancellation, incomplete writes and malformed inputs
also require portable lifecycle/contract regressions. Longer horizons,
additional initialization dates and larger GPU batches require their own
qualification and resource measurements.

Each result must bind the source inventory, immutable image, checkpoint and
input hashes, seed, batch layout, model seed mode, numerical/RNG policies,
reference/candidate commands, GPU/driver/math settings and field metrics.
Preserve failed references and candidates alongside passing reports. Data
acquisition and end-to-end service duration are separate timing boundaries from
model inference.

## Execution evidence

### Current single-GPU version

The current release tag is
`nvcr.io/dycvht5ows21/physicsnemo-serve:aifs2ens-release-20261105-4b91b64806d9`;
its immutable image is
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:273bf30c623a160cf857191a041b01ddd210cbd5b6c7540eca7b734c6d364f92`.
Source-content SHA-256 is
`4b91b64806d97fee88bc829c08efbf89f413f5825e4030a35bcbfef1ba77778e`, on base
commit `bc980628c03e43638dcbc0bb0b8572951252ef4d`. The Linux build on
`nvdi863k9ph16b0` used
`/home/zhiweis/Projects/physicsnemo-serve-aifs2ens-fresh-20261105-v2/source`.
Its 405 portable tests passed with zero skips and two unrelated FCN GPU cases
deselected. Source inventory, service command, image label and registry digest
were verified. Later edits to this documentation are outside that inventory;
the qualified runtime and scientific source remain unchanged.

Lepton job `aifs2ens-fullqa-4b91b64806-db52` completed at
2026-11-05T21:49:42Z with **412 passes: 405 portable and all seven H100 cases**,
zero skips and zero errors. Coverage includes both Zarr backends, serial and
uneven mini-batches, complete-ensemble cached A/B/A reuse and the one-task REST
path. Eight actual candidate rollouts passed 7,440 member/step/field comparisons:
6,720 had zero error and 720 had nonzero error; the largest error-to-limit
fraction was 0.5. Peak Torch allocation was 45.273 GiB and reserved memory was
51.215 GiB, below the configured 64 GiB task reservation for the tested layouts.
The GPU was H100 80GB HBM3, UUID `GPU-f4f3639a-f90c-26c2-c6ce-85f59f5c0df1`,
with driver 570.195.03. The seven-case count is not combined with earlier runs.

The nine standalone stock Earth2Studio reference processes were generated in
the preceding image
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:b58eeaf61c78fe5baa8e9e40927e8219aec02569dc300115c0b1627014e643a9`,
source `53da50741e749dbe9820bcca103dbd745e7048c6ac5b94b3c62af10586f34eb5`.
The final rerun reused those same reference arrays after verifying all 30 hashes
and all 5,280 holdout comparisons under calibration SHA-256
`00730662fac03c1d63ba74d6e19a523a7375240a372a2b2bc35e825d66715226`.
Only `run_qualification.py`, `test_h100_parity.py` and their portable
qualification tests changed between the two image snapshots. Scientific source,
runtime, dependencies and native model/RNG settings were byte-identical. No
reference was relabeled as generated by the final image, and no calibration or
comparison limits were changed.

Private deployment `aifs2ens-1105-4b91b64806` serves
[the current HTTPS endpoint](https://r0whe339-aifs2ens-1105-4b91b64806.xenon.lepton.run).
API readback verified the immutable image, one H100, `earth2-gpu` with the
`aifs2ens` capability tag, 65,536 MiB GPU reservation and
`AIFS2ENS_MAX_BATCH_SIZE=2`. The three authenticated requests all used
initialization `2026-11-04T12:00:00Z`, seed 0, two ensemble members, common IFS ENS
source member 1, four six-hour steps and all 120 fields. Common input SHA-256
was `631d9fed3102204074a39451159fd641108f98df1eef855e7d1e3b5982012b1e`.

| HTTPS request | Batch size | Parent run ID | Comparisons |
| --- | ---: | --- | ---: |
| A | 2 | `b65b56a9-83db-42f6-9598-b77a3192d250` | 960 |
| B | 1 | `c634fa3b-2009-4326-8d74-27d55cd8f328` | 960 |
| A repeat | 2 | `ecd85a28-8ae0-4ca5-afc6-c89fe5093a1e` | 960 |

Comparison job `aifs2ens-ep-cmp-4b91b64806-5vw9` completed successfully: all
2,880 endpoint comparisons passed with zero numerical failures and zero
observed error. This sample does not establish general bitwise reproducibility;
the separate full H100 suite above includes nonzero native variation. Worker
PID 672 on GPU `GPU-ed7654fc-8309-5a43-2bbf-a38197b758e9` started once, had no
restart and reported four cache uses including warmup. Exactly three parent GPU
releases were observed, with no child fan-out. A real HTTPS `t2m` Zarr ZIP
download returned HTTP 200, 24,996,193 bytes and SHA-256
`a243fa9d38010c6b53c3572261fee9953114a7b7335b67cd896f6a624d529c54`;
all eight member/step field hashes matched the output store.

The deployed readiness-file gate passed direct authenticated probes:
`/readyz` returned 200, then 503 while the ready record was held, then 200 after
the original record was restored. `/healthz` stayed 200. This exercised the
gate without restarting the worker or mutating model assets. Lepton's current
API provides TCP liveness and readiness approval only; no native platform HTTP
readiness probe is configured. Use `GET /readyz` where an orchestrator supports
HTTP readiness.

An over-limit request with three members and batch size 3 was accepted into the
pipeline as run `69ec38d7-8cbe-4a6f-893d-9a2f9c07f560`, then rejected immediately
by prepare before weather staging or GPU execution. The shared retry policy
moved it to failed/DLQ on attempt 5, about 241 seconds after submission. Owned
logs, absent input staging and unchanged GPU/cache observations verified that
the rejection caused no GPU allocation release or model use.

Evidence is under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/fresh-qualification`.
`full-qa-rerun/final-audit.json` binds the 412-pass inventory, eight rollouts and
reference provenance; its collected report archive SHA-256 is
`54100d2dacba5a2288dc5d578393fc04e8497cf8aa257baedbd64c6fcb79a44f`.
`endpoint-v2/comparison-summary.json` binds HTTPS comparisons, downloaded hashes
and worker continuity. The verified 20-file endpoint evidence archive SHA-256 is
`4c38265f897c740d661efd2bf2858d9f2cdb5c1aa00f365a5f48b3659cb0991e`.
The current one-H100 endpoint remains running for client testing; the superseded
`53da50741e74` endpoint was scaled to zero. The original `4ff4684ceb44` endpoint
was left untouched.

### Intermediate fresh image — source `53da50741e74`

The preceding `b58eeaf61c78` image passed 398 portable tests and its separate
2,880-comparison HTTPS A/B/A check, but its full H100 qualification failed:
six GPU cases passed and the REST case read an intermediate HTTP 200 response
before the results role had finalized its structured payload. That failed run
is preserved under `fresh-qualification/full-qa`; it is not a seven-case pass.
The QA-only correction waits within a deadline for structured result
finalization while polling the same accepted run ID. It changes no model,
forecast, RNG, scientific preparation or runtime code. The nine reference
processes generated during this attempt retain that provenance, as described
above, and are not counted as newly generated for the final rerun.

### Prior earth2-gpu endpoint — source `4ff4684ceb44`

The following historical record is preserved from the preceding endpoint
qualification. Its source-only follow-up notes and deployment state describe
the state at that audit; its counts are not added to the current release.

The source-only follow-up removes the redundant module-level asset-preparation
hook: model loading still verifies the checkpoint, runtime and interpolation
hashes once before construction, and cached warmup reuses the verified model.
It also adds the positive `AIFS2ENS_MAX_BATCH_SIZE` deployment guard, default
`2`, before downloads and again before GPU execution. These changes have not
been included in a new published image or deployed endpoint qualification.
The image, source fingerprint and results below remain the deployed release's
record.
The follow-up passed 385 portable tests, with zero skips and two unrelated FCN
GPU cases deselected. Lint, formatting, manifest validation and independent
review also passed. Follow-up evidence is under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/asset-batch-followup`.

The later PR-readiness changes install the existing QA dependency extra in the
Python CI job, add a checked-in source-context/inventory generator, and bound
the release observer to the latest 100 entries. The release supervisor now
sets `INFERENCE_READINESS_FILE` for the HTTP server; `/readyz` requires both
Redis and an atomically published GPU-ready record. Configure the deployment
to probe `GET /readyz` for readiness, with `/healthz` or TCP for liveness. These
changes preserve the model and RNG path. They remain separate from the deployed
release evidence below.

The PR-readiness follow-up passed the full Python CI command in a fresh
isolated environment with `--frozen --extra qa`: 847 passed and nine optional
cases skipped, including all 189 AIFS cases passing without skips. The selected
portable subset contains 398 passing cases; its two separate FCN GPU cases are
excluded from this plugin's portable command. The Rust inference-server suite
passed 263 cases with three existing tests ignored. Repository lint, formatting,
manifest validation, lock validation and a fresh independent review passed.
Evidence is under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/pr-readiness-fixes`.

Private Lepton deployment `aifs2ens-1105-4ff4684ceb` passed real H100 endpoint
validation on 2026-11-05 at
[the HTTPS endpoint](https://r0whe339-aifs2ens-1105-4ff4684ceb.xenon.lepton.run).
The release uses `earth2-gpu`, the `aifs2ens` capability tag, one GPU and a
65,536 MiB task reservation. Preparation remains a service hook; no CPU execute
worker is added. The deployment's actual configuration was verified through
API readback and again at HTTP-test startup. Authentication uses a private
workspace-token reference; credentials are not stored in the source or examples.

Release tag is
`nvcr.io/dycvht5ows21/physicsnemo-serve:aifs2ens-release-20261105-4ff4684ceb44`;
immutable image is
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:38408796953d7fb373b94a3c4533ac2de7f3bb2e89270218b66651d0a0980b01`.
Source-content SHA-256 is
`4ff4684ceb445cd7d51bedbb9e788486508ee75ff31e3fd559b58cfc1343133c`, on base
commit `bc980628c03e43638dcbc0bb0b8572951252ef4d`. It was built on Linux host
`nvdi863k9ph16b0` from
`/home/zhiweis/Projects/physicsnemo-serve-aifs2ens-single-gpu-20261105/source-endpoint-v1`.
All **368 portable tests passed** locally and in the image build, with zero skips
and two unrelated FCN GPU cases deselected. This includes nine service lifecycle
and recovery tests beyond the earlier 359-test portable suite.

The Dockerfile's default final `release` stage exposes port 8001 and starts
`python qa/plugins/earth2-aifs2ens/serve.py`. The supervisor requires writable
`AIFS2ENS_SERVICE_ROOT` and the artifact environment, verifies the source
inventory at startup, creates a new session for logs and observation snapshots,
allows 600 seconds for warmup or recovery, publishes readiness atomically, and
cleans up owned process groups on exit. Health, readiness, workflow readiness
and schema probes returned HTTP 200. Forecast submissions returned HTTP 202;
status and result retrieval returned HTTP 200 with successful registered Zarr
outputs.

All requests used initialization `2026-11-04T12:00:00Z`, seed 0, common IFS ENS
input member 1, four six-hour steps and all 120 output fields. Common input
SHA-256 remained
`631d9fed3102204074a39451159fd641108f98df1eef855e7d1e3b5982012b1e`.

| HTTPS request | Members | Batch size | Parent run ID | Candidate comparisons |
| --- | ---: | ---: | --- | ---: |
| A | 2 | 2 | `9ed13712-9784-4065-9117-3b4f953e7f8b` | 960 |
| B | 2 | 1 | `acfc9eb6-ac7c-4f17-80e2-6e2112f37571` | 960 |
| A repeat | 2 | 2 | `b73d3845-3de9-41d2-880a-eeca9aa97e46` | 960 |

One H100 80GB HBM3, UUID `GPU-f8273a5e-2007-705e-9cc0-946fcb7d4b27`, served
all three requests through `earth2-gpu` worker PID 669. The worker started once,
had no restart, and reported four cache uses including warmup. Each parent had
one execution and one matching GPU release, with no fan-out child runs.

Fresh reference job `aifs2ens-ep-ref-4ff4684ceb-f86v` passed six standalone
Earth2Studio processes: one fresh anchor and two fresh holdouts for each full
and serial layout. The runner verified 24 persisted forecast-array hashes and
all **3,840 held-out comparisons** against the prior calibration SHA-256
`00730662fac03c1d63ba74d6e19a523a7375240a372a2b2bc35e825d66715226`.
No recalibration or limit widening occurred. Reference arrays persist on the
task-owned shared NFS.

Comparison job `aifs2ens-ep-cmp-4ff4684ceb-wxxv` passed **2,880 unique
member/step/field comparisons**, with zero numerical failures. Of those, 2,640
had zero maximum absolute error and 240 had nonzero errors. The largest
candidate error-to-limit ratio was 0.5; no zero limit was exceeded. This
establishes parity within measured native CUDA variability for the specified
fixture and layouts, not bitwise reproducibility or ECMWF operational product
equivalence. Larger batches, additional dates/seeds and longer horizons remain
separate qualification work. The prior seven-case H100 inventory below was not
rerun as a complete suite for this release and is not added to its test totals.

A real HTTPS download of the A request's `t2m` Zarr ZIP returned HTTP 200,
24,996,193 bytes and SHA-256
`2ba1ee92361a6d473d4ed1cbeab49dd680dd1479999fba15d043585643de2ed8`.
All eight canonical member/step field hashes matched the NFS output store.

Evidence is under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/earth2-gpu-endpoint`.
The verified 61-file report/log archive SHA-256 is
`0c7ec1196148fe89b5cb706894cd41936720ff60a7b7726489a2d7797ff7257a`.
`final-endpoint-evidence-audit.json` records the successful independent review,
source/image bindings, numerical limits, worker continuity and download checks.
The final authenticated health checks passed and the one-H100 endpoint remains
running for client testing. That audit preceded the source-only follow-up
described above; the deployed image is unchanged.
Shared NFS task root is
`/mnt/my_lfs_data/Projects/gpu_programming-aifs2ens-endpoint-20261105`.
The [H100 guide](../../qa/plugins/earth2-aifs2ens/README.md) describes the release
entrypoint, authenticated access and reproduction.

### Prior earth2-gpu routing checks

The routing-only follow-up passed 359 portable
regressions, with zero skips and the same two unrelated FCN GPU tests deselected.
Three routing checks failed against the old class and passed after the change;
they cover the expanded manifest, prepare profile and run-local worker launch.
Manifest validation, lint, formatting and independent review also passed. Logs
and JUnit are under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/earth2-gpu-routing-__uu0ny8`.
These checks preceded the release endpoint validation above.

### Prior single-GPU baseline — original executor class

The image and results below qualify the preceding single-GPU baseline under its
original `earth2-aifs2ens-gpu` executor class.

Lepton job `aifs2ens-sg-1105-e80052dc63-556v` completed successfully on 2026-11-05.
All **366 selected tests passed**: 359 portable regressions and all seven H100
cases, with zero skips. Two unrelated FCN spherical-Gaussian GPU cases were
explicitly deselected. Both Zarr backends, serial and uneven mini-batches,
whole-ensemble cached A/B/A reuse and the REST path passed. The REST scheduler
dispatched one GPU ensemble task with no child run, and exactly one parent GPU
allocation release was observed.

The image source-content SHA-256 is
`e80052dc63058530307250e41ad34ed3597ed40bd12f2fe713be4951775a1523`, on base
commit `bc980628c03e43638dcbc0bb0b8572951252ef4d`. The immutable image is
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:ad5716e9d7360b1ca3f6c5bbe1f752443ede61b63a0e8f4788d9ed164d28a532`.
Linux build source was
`/home/zhiweis/Projects/physicsnemo-serve-aifs2ens-single-gpu-20261105/source-v1`
on `nvdi863k9ph16b0`; the build also passed the same 359 portable tests. The
executing GPU was H100 80GB HBM3, UUID
`GPU-f4f3639a-f90c-26c2-c6ce-85f59f5c0df1`, with kernel driver 570.195.03 and
CUDA forward compatibility driver 590.48.01. Actual worker records matched the
standalone references: Torch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA 13.1,
FlashAttention `2.7.4.post1+nv26.1.42222806`, and Anemoi Models 0.11.2. Native
TF32 and reduced-precision BF16 reductions were enabled; deterministic
algorithms were disabled. These settings were recorded, not modified by the
plugin. Model seed remained `None` and native noise projection stayed on CUDA.

The common initialization was `2026-11-04T12:00:00Z`, with IFS ENS input member
1 and initial seed 0. Independently captured input SHA-256 was
`631d9fed3102204074a39451159fd641108f98df1eef855e7d1e3b5982012b1e`; invariant
SHA-256 was `9d525850ae33b1118a49c82c30fee65f18e6190c9b1e484d87aaaa15cee585b8`.
All candidates used the same verified common input and all 120 output fields.

| Reference layout | Members | Batch size | Six-hour steps | Candidate comparisons |
| --- | ---: | ---: | ---: | ---: |
| Full | 2 | 2 | 4 | 4,800 across Python, Rust, cached A, repeated A and REST |
| Serial | 2 | 1 | 4 | 1,920 across the direct serial request and cached B |
| Uneven tail | 3 | 2 | 2 | 720 |

The initial four-calibration/two-holdout gate failed before any candidate ran:
serial and uneven layouts each had 240 failing holdout comparisons against
zero observed bounds. That failed calibration is preserved. The predeclared
reference-only fallback expanded each layout to eight calibration processes
and two new holdouts, retaining the same factor-two margin and native math.
All 5,280 final held-out comparisons passed. The runner verified 100 forecast
array hashes from 30 independent reference processes before candidate execution.
Passing calibration SHA-256 is
`00730662fac03c1d63ba74d6e19a523a7375240a372a2b2bc35e825d66715226`.
Candidate limits were frozen at that point and never widened.

All eight candidate reports passed **7,440 unique member/step/field comparisons**,
with exact coordinates, dtype, member coverage and NaN/Inf masks. Of those,
6,720 had zero maximum absolute error. The remaining 720 had nonzero errors;
the largest candidate error-to-limit ratio was 0.5. The following diagnostic
maxima are across all candidates; a field's maximum absolute error and maximum
RMSE may occur at different members or forecast steps:

| Field | Maximum absolute error | Maximum RMSE |
| --- | ---: | ---: |
| `t2m` | 3.19449 K | 0.0529305 K |
| `msl` | 371.5 Pa | 4.47049 Pa |
| `u10m` | 3.91742 m/s | 0.0738179 m/s |
| `v10m` | 2.59396 m/s | 0.0782503 m/s |

This establishes agreement within measured native CUDA variability for the
specified fixture and layouts, not bitwise reproducibility. A cached A repeat
can differ numerically from A while both pass the independent native-reference
gate. The largest measured direct-candidate CUDA allocation peak was
48,611,098,624 bytes (45.27 GiB), and the reserved peak was 54,991,519,744 bytes
(51.21 GiB). REST subprocess peak memory was not separately measured. These
results do not qualify larger batches, additional initializations or seeds,
60-step horizons or ECMWF operational product equivalence.

Collected evidence and the successful `final-single-gpu-evidence-audit.json`
are under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-single-gpu-20261105/run-v1`.
The verified 175-file report/log/JUnit archive SHA-256 is
`dffd618143a562987fc9653303a51fe9174f7f426628ee0370701a1068e7a307`.
Remote output remains at
`/mnt/my_lfs_data/Projects/gpu_programming-aifs2ens-single-gpu-20261105/qualification/lepton-e80052dc6305-01`.
Raw forecast arrays used ephemeral scratch; their reports and runner-verified
hash inventory persist, without claiming local rehashing of absent arrays.
This baseline image remains immutable and retains its original executor class;
the current `earth2-gpu` release is recorded above. The
[H100 guide](../../qa/plugins/earth2-aifs2ens/README.md) describes reproduction and
artifact binding. All actual CUDA execution ran on the configured remote H100.

### Historical native fan-out/per-member run — not the current baseline

Lepton job `aifs2ens-qa-1105-ac88d93d3d-857b` passed 327 selected tests
(321 portable regressions and six H100 cases), with zero skips and two unrelated
FCN GPU cases deselected. It ran on one H100, but evaluated separate member
trajectories using IFS members 1 and 2 and seeds 0 and 1024 under the earlier
fan-out architecture. Its 2,640 passing field-step comparisons do **not** prove
complete-ensemble shared-stream parity and do not qualify the new version.

Its source-content SHA-256 is
`ac88d93d3decbb03c0ef8abda6ffe922b154f3a7f750090f8ad9d46faf5fc94a`, on base
commit `bc980628c03e43638dcbc0bb0b8572951252ef4d`. The image is
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:2cfc4b6ffe0ca75ed3500c8510153249e6915374d6cf2433ee56eefe00969aa8`.
Linux build source was
`/home/zhiweis/Projects/physicsnemo-serve-aifs2ens-qa-20261105/source-v9` on
`nvdi863k9ph16b0`. The GPU was H100 80GB HBM3, UUID
`GPU-455de210-dcc8-6f5e-fe87-a17d4e397f8b`, with kernel driver 570.195.03 and
CUDA forward compatibility driver 590.48.01. Torch was
`2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA 13.1 and FlashAttention
`2.7.4.post1+nv26.1.42222806`.

The frozen cycle was `2026-11-04T12:00:00Z`. Historical unmasked input hashes
were `631d9fed3102204074a39451159fd641108f98df1eef855e7d1e3b5982012b1e` and
`08eaf5ad821828c36bdeb7401a415c68a606da47ecc54232193e2c79afcca4b7`.
The invariant hash was
`9d525850ae33b1118a49c82c30fee65f18e6190c9b1e484d87aaaa15cee585b8`.
Twelve reference processes verified 48 forecast-array hashes; calibration hash
was `955123aa05d6febf3edacdb7aba72ca215dd552eba05180e3514553192ff91e6`.
It measured a 26,477,719,040-byte candidate allocation peak and
29,276,241,920-byte reserved peak for its separately executed member requests;
those measurements do not qualify larger batched ensembles.

Full historical evidence remains under
`/Users/zhiweis/Projects/physicsnemo-serve-evidence/aifs2ens-lepton-20261105/run-v9`
and remote output
`/mnt/my_lfs_data/Projects/gpu_programming-aifs2ens-tdd-20261104/qualification/lepton-ac88d93d3dec-01`.
The 116-file collected archive SHA-256 is
`4c2a390681afc91a405710dc454ae3c2c1f44cfacd1bb2687dc0e762afb14530`.
Commands, JUnit, per-field metrics, inventories and
`final-native-evidence-audit.json` are preserved there. Anemoi's sparse projector
moves its unregistered matrix to the input device inside the first native
forward; the QA device check must run after prediction rather than immediately
after loading.

### Historical CPU-adapter fan-out run — not the current baseline

Lepton job `aifs2ens-qa-1105-22b11366cb-xcfz` passed 318 selected tests
(312 portable and six H100 cases), with zero skips and two unrelated FCN GPU
cases deselected. It used a custom CPU noise projection, forced math settings,
extra ocean masking and the earlier staging/fan-out design. Its 2,640 exact
field-step comparisons qualify only that disclosed historical policy.

Its source-content SHA-256 is
`22b11366cb816453bc34ed171888571e8e5e18a561369cd61813d7b773ea7ccb` and image is
`nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:9fbb76e9c3298f74a6a1408a721431257185d08d3788f545c29a65fbf4efadaf`.
Linux source was `source-v4` below the same historical build root. Reports remain
under the local `aifs2ens-lepton-20261105/run-v4` evidence directory and remote
`qualification/lepton-22b11366cb81-01`. The 74-file archive SHA-256 is
`b89633f1c61f1fdc9ba4f7382e4ba4e6393b16c31e43a63a0d8439d2ed440636`.
The masked input hashes, original commands, failed sparse-projection probes and
strict comparisons remain in that evidence. Neither source archives nor failed
logs should be rewritten to describe the new shared-stream workflow.

## External scientific references

| Reference | Purpose and remaining scope |
| --- | --- |
| [Pinned Earth2Studio workflow](https://github.com/NVIDIA/earth2studio/blob/f8af4030cfd43ec79bf4c0cf6c23463a87bfef97/earth2studio/run.py) | Immediate single-GPU ensemble reference: repeat the common input, apply the chosen perturbation object, and iterate each mini-batch |
| [Pinned AIFS2ENS wrapper](https://github.com/NVIDIA/earth2studio/blob/f8af4030cfd43ec79bf4c0cf6c23463a87bfef97/earth2studio/models/px/aifs2ens.py) | Model loading, invariant modes, regridding, forcings, BF16 prediction and optional per-step reseeding; this initial workflow leaves `model.seed=None` |
| [Earth2Studio AIFS2ENS tests](https://github.com/NVIDIA/earth2studio/blob/f8af4030cfd43ec79bf4c0cf6c23463a87bfef97/test/models/px/test_aifs2ens.py) | Coordinate/iterator/package smoke coverage; synthetic inputs and possible skips do not establish real-weather numerical parity |
| [ECMWF AIFS ENS v2 notebook](https://huggingface.co/ecmwf/aifs-ens-2.0/blob/29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf/run_AIFS_ENS_v2.0.ipynb) | Independent scientific preparation/inference target; not the immediate Earth2Studio ensemble golden |
| [ECMWF inference configuration](https://huggingface.co/ecmwf/aifs-ens-2.0/blob/29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf/inference.yaml) | Wave-direction decoding, accumulation and GRIB product semantics require separate checks |
| [Anemoi Inference 0.8.3 runner](https://github.com/ecmwf/anemoi-inference/blob/291e21c56f7610ee30689a53b201e1bc339f9c14/src/anemoi/inference/runner.py) | Further cross-library diagnostics; this pinned runner has no `seed` constructor option |

ECMWF notebook preprocessing can differ from Earth2Studio. The notebook forms
wave-direction sine/cosine features after regridding and applies an ocean mask
to snow/soil fields; Earth2Studio forms those wave features in its datasource,
and the plugin preserves that route without adding the notebook mask. A
common-input model comparison does not establish independent preparation or
operational product equivalence. Keep source provenance, unit conventions,
nonfinite masks and regrid identities explicit. Do not use downloaded
operational forecasts as exact numerical goldens.

Independent ECMWF/Anemoi input preparation, additional initialization dates,
60-step forecasts, larger batches, decoded/accumulated products and operational
GRIB equivalence remain unexecuted gates unless separately recorded. Future
fan-out qualification must compare against this initial single-GPU baseline
with the same complete input setup and shared RNG draw/member mapping.
