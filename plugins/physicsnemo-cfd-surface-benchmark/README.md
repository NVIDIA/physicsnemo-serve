# PhysicsNeMo-CFD Surface Benchmark

Runs the pinned PhysicsNeMo-CFD v0.0.2 DrivAerML surface benchmark once on
one scheduled GPU. The request may select only the models, metrics, and bounded
controls declared in `plugin.yaml`; checkpoints, packages, Python targets,
devices, output paths, Hydra overrides, environment logging, and HTTP headers
are deployment-owned.

The upstream v0.0.2 runner does not apply a public benchmark batch-size
override, so this plugin exposes no `batch_size` request field and pins
`run.batch_size` to `1` in its immutable preset.

Each case must be named `run_<number>` (at most 64 characters) and provide
allowlisted HTTPS URIs, SHA-256 digests, and exact byte sizes for both the
boundary VTP and the companion DrivAer STL geometry. Prefetch verifies every
object before execution. The adapter materializes the files as
`run_<number>/boundary_<number>.vtp` and `run_<number>/drivaer_<number>.stl`,
invokes the fixed upstream module, and publishes `benchmark_results.json` as
the primary artifact.

Every execution uses a new run-scoped attempt directory. Meshes are copied
from no-follow cache file descriptors while rechecking their exact size and
SHA-256, then fsynced and atomically installed. Reports from an earlier retry
can never satisfy a later attempt. Cancellation terminates the complete child
process group while retaining `resolved_config.json`,
`benchmark_diagnostics.json`, and the bounded `benchmark.log` as audit
artifacts.

The structured result records the provider tuple, canonical preset SHA-256,
selected metrics, case digests, and the deterministic registered-artifact
inventory.

`workflow.py` is the thin manifest entrypoint. The plugin-local
`surface_benchmark_impl.py` owns the surface request contracts, immutable
preset validation, DrivAerML layout, and report policy. It delegates bounded
subprocess supervision, safe file operations, and root-contained artifact
primitives to the import-light `physicsnemo_cfd_runtime` package.

Verified mesh prefetch is fail-closed until the operator sets
`E2S_PREFETCH_ALLOWED_HTTPS_HOSTS` to the exact permitted DNS hosts. Operators
may lower the 64 GiB defaults with `E2S_PREFETCH_MAX_OBJECT_BYTES` and
`E2S_PREFETCH_MAX_REQUEST_BYTES`. These are deployment settings and are not
request fields or plugin-manifest overrides.

`examples/default_request.json` uses a documentation hostname and exists for
fast schema/prepare checks. `examples/public_run_1_request.json` is the live
E2E input: one public DrivAerML `run_1` surface mesh and companion STL pinned
to a Hugging Face dataset commit. The VTP is 659,606,189 bytes with SHA-256
`01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40`; the STL
is 142,385,186 bytes with SHA-256
`411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2`. It runs
`domino_surface` with `l2_pressure` and does not request prediction meshes or
visuals.

`examples/prefetch_smoke_request.json` is a separate 5,297-byte transport
fixture pinned to a Kitware `vtk-examples` commit and intentionally reuses that
same tiny object for the geometry slot so REST/prepare/prefetch checks remain
cheap. It is valid VTK PolyData and is intentionally only a transport smoke
input; it is not a DrivAerML case and must not be sent to the GPU benchmark
runner.

The canonical Hugging Face URL redirects to a provider-generated signed URL.
For that fixture, the worker deployment needs:

```bash
E2S_PREFETCH_ALLOWED_HTTPS_HOSTS=huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co
E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS=us.aws.cdn.hf.co,cas-bridge.xethub.hf.co
E2S_DOWNLOAD_TIMEOUT_SECS=1800
```

The signed-query exception applies only to redirects and only when the target
is in both exact host lists. A client-provided signed URL is still rejected.
Hugging Face controls the CDN hostname and may change it; inspect the canonical
redirect and explicitly update both deployment allowlists instead of using a
wildcard.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/physicsnemo-cfd-surface-benchmark --through-phase prepare
python scripts/plugin_dev.py check-env plugins/physicsnemo-cfd-surface-benchmark
python scripts/plugin_dev.py run-local plugins/physicsnemo-cfd-surface-benchmark --dry-run
```

`check-env` and real execution require the dedicated `physicsnemo-cfd-gpu`
environment. Provider versions are locked in `provider.lock.json`; the full
immutable benchmark contract is under `configuration` in `plugin.yaml`.
The CFD runtime is packaged but not launched by the default Earth2 worker set;
a CFD deployment must explicitly set
`PHYSICSNEMO_SERVE_EXECUTOR_CLASSES=physicsnemo-cfd-gpu`.

## REST and prefetch acceptance without a GPU

The opt-in local acceptance test starts only Redis, the real Rust REST server,
and the prepare and prefetch workers. It verifies schema rejection, submits
`examples/prefetch_smoke_request.json`, checks the exact prepare-generated VTP
and STL integrity plan, proves a checksum mismatch fails before scheduling and
leaves no partial cache file, downloads over allowlisted HTTPS, validates
SHA-256 and size, checks the checksum-addressed file plus metadata sidecar, and
confirms the Redis handoff to `schedule`. It then submits the same request again
and requires a verified cache hit. No scheduler or execution worker is started.

```bash
QA_CFD_PREFETCH_E2E_ENABLED=1 \
pytest -q -s tests/test_physicsnemo_cfd_rest_prefetch_e2e.py
```

Set `QA_CFD_PREFETCH_E2E_SKIP_BUILD=1` to reuse already-built debug binaries.
The test uses a readiness-only module stub because the local test interpreter
does not contain the dedicated CFD provider. That stub is never executed; the
real provider import, model package, DrivAerML semantics, and benchmark outputs
remain covered only by the live GPU E2E.

## Live GPU E2E

The opt-in QA suite deploys only this plugin and only the
`physicsnemo-cfd-gpu` executor class, submits the public request exactly once,
waits up to 6.5 hours, downloads every required artifact, verifies the primary
artifact alias, and validates the benchmark report's model/case/metric shape.

```bash
python qa/scripts/run_qa.py \
  --service rust \
  --image-tag <already-built-and-pushed-image-tag> \
  --suite cfd_e2e \
  --num-proc 1
```

This command requires the normal Lepton QA credentials and an 80 GiB-class GPU
deployment. It writes request, status, result, report, CSV, HTML, diagnostics,
and log evidence below the configured QA artifact directory.

Against an existing correctly configured endpoint, run:

```bash
cd qa/inference
QA_CFD_E2E_ENABLED=1 \
QA_CFD_E2E_TIMEOUT_SECS=23400 \
pytest -m cfd_e2e --service rust --urls https://<endpoint> --token <token> -v
```

Do not use the shared retrying QA session for submission: the live test uses a
separate no-retry POST because the API has no idempotency key.

## Example

- `examples/default_request.json`
- `examples/prefetch_smoke_request.json` is the pinned 5.3 KB prefetch-only request.
- `examples/public_run_1_request.json` is the pinned live E2E request (~660 MB input).
- `fixtures/tiny_surface.vtp` is a small layout/test fixture, not a model-quality benchmark case.
