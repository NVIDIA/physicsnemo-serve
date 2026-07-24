# Inference API Tests

> **Note:** This guide references deployment values from `deploy/config.yaml`.
> Copy `deploy/config.example.yaml` to `deploy/config.yaml` and fill in your
> environment-specific values before following these instructions.

Pytest test suite for Earth2Studio inference server workflows.

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LEPTON_ENDPOINT_TOKEN` | Bearer token for API authentication (required) |
| `BASE_URLS` | Comma-separated list of inference server URLs |
| `SERVICE_TYPE` | `python` or `rust` (default: `python`) |

## Running Tests

```bash
# Single server
export LEPTON_ENDPOINT_TOKEN="your-token"
export BASE_URLS="https://your-server.xenon.lepton.run"
pytest

# Multiple servers in parallel
export BASE_URLS="https://server1.xenon.lepton.run,https://server2.xenon.lepton.run"
pytest -n 2
```

## Test Files

- `test_smoke.py` - Fast sanity checks (@pytest.mark.smoke)
- `test_cicd.py` - CI/CD pipeline tests (@pytest.mark.cicd)
- `test_stress.py` - Sustained concurrency/load tests (@pytest.mark.stress)
- `test_output_publication.py` - Live object-store sync tests (@pytest.mark.publication)
- `test_physicsnemo_cfd_surface_e2e.py` - Opt-in live CFD GPU test (@pytest.mark.cfd_e2e)
- `test_basic.py` - Core workflow tests (health, list workflows, run workflows)
- `test_negative.py` - Invalid parameter tests
- `test_experiments.py` - Experimental/exploratory tests

## Workflows Tested

- `deterministic_fcn_workflow`
- `deterministic_workflow`
- `diagnostic_workflow`

## PhysicsNeMo-CFD E2E

The Rust-only `cfd_e2e` suite is explicit opt-in and excluded from normal QA.
It downloads the pinned public DrivAerML VTP and STL inputs, runs one
`domino_surface` inference with `l2_pressure`, and validates the full
API-to-GPU-to-artifact path. Submission uses a no-retry HTTP session so a
transient response cannot duplicate the long-running GPU job.

The deploy runner selects only `physicsnemo-cfd-gpu`, configures the exact
Hugging Face source/CDN host policy, and increases the verified download
timeout:

```bash
python -u qa/scripts/run_qa.py \
  --service rust \
  --image-tag <already-pushed-tag> \
  --suite cfd_e2e \
  --num-proc 1
```

For an existing correctly configured endpoint:

```bash
QA_CFD_E2E_ENABLED=1 \
QA_CFD_E2E_TIMEOUT_SECS=23400 \
pytest -m cfd_e2e \
  --service rust \
  --urls https://<endpoint> \
  --token <endpoint-token> \
  -v
```

Evidence is written under `QA_CFD_E2E_ARTIFACT_DIR` (default
`artifacts/cfd-e2e`). The suite requires one compatible 80 GiB-class GPU,
writable persistent `/outputs`, and outbound Hugging Face access.

### Direct-provider parity

The opt-in parity orchestrator first runs `cfd_e2e` through REST and persists a
versioned handoff containing mount-relative input and report paths plus their
digests. After the endpoint is torn down, it starts one same-image Lepton batch
job. The job re-verifies the staged inputs, builds the checked-in provider
configuration independently of the plugin's resolved config, invokes
PhysicsNeMo-CFD directly, and compares the report structures and finite metric
values symmetrically.

```bash
python -u qa/scripts/run_lepton_cfd_parity.py \
  --image-tag <already-pushed-tag>
```

The endpoint and batch job run sequentially, so peak allocation remains one
H100. Local evidence is under `qa/artifacts/cfd-parity/<run-id>` and remote
evidence is under `/outputs/cfd-parity/<run-id>`. The default profile is
`cfd_parity_surface_run1.json`; future surface or volume coverage is added by
supplying another profile with its REST suite, direct runner module, input
layout, provider config, and per-metric tolerances.

Use `--profile qa/inference/cfd_parity_surface_run1_full_matrix.json` for the
full `run_1` matrix: five surface models by five metrics, producing 25 unique
model/case/metric selections. PhysicsNeMo-CFD expands vector and force metrics
into 55 scalar report values; parity checks both per-case and summary scopes
for 110 scalar comparisons.

Add the pinned `run_11` case without duplicating the model profile:

```bash
python -u qa/scripts/run_lepton_cfd_parity.py \
  --image-tag <already-pushed-tag> \
  --profile qa/inference/cfd_parity_surface_run1_full_matrix.json \
  --rest-request-path plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_11_full_matrix_request.json
```

This selects 50 model/case/metric combinations and compares 110 per-case plus
55 summary scalar values.

Pass `--rest-evidence-dir <completed-cfd-e2e-run>` to reuse a completed REST
run and launch only the direct comparison job. The handoff stores no tokens and
uses only mount-relative paths; the job rechecks report and input digests before
execution.

## Multi-GPU CI/CD Check

The `cicd` suite includes a Rust-only `multigpu` test. It reads visible GPU IDs
from `/metrics`, skips on single-GPU endpoints, and on multi-GPU endpoints
submits one configured workflow request per GPU concurrently. The test passes
only if completed result envelopes report `execution.gpu_stream` coverage across
every visible GPU. FCN3 presets also require the reported execution intervals to
share a common overlap window.

Useful environment variables:

- `QA_MULTIGPU_GPU_COUNT`: expected visible GPU count. When set, the test fails if
  `/metrics` reports a different count.
- `QA_MULTIGPU_WORKFLOWS`: comma-separated workflow presets. Valid presets are
  `earth2-deterministic`, `deterministic-fcn3`, `stormcast-fcn3`, and
  `e2s-foundry-fcn3`. The default is `earth2-deterministic`.
- `QA_POST_HEALTH_WAIT_SECS`: optional deploy-runner grace period after `/health`
  passes and before pytest starts. This only applies to Rust `cicd`/`full` runs
  when a multi-GPU selector such as `QA_MULTIGPU_GPU_COUNT` or
  `QA_MULTIGPU_WORKFLOWS` is set; it does not delay normal `cicd`, `smoke`, or
  `stress`. Use this when GPU execute workers register after the HTTP server
  becomes healthy.
- `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID`: Rust-only single-plugin mode. `run_qa.py`
  forwards this into the Lepton container, and pytest skips workflow tests that
  resolve to any other plugin ID. Health and workflow-list checks still run.

```bash
QA_MULTIGPU_GPU_COUNT=8 \
QA_MULTIGPU_WORKFLOWS=earth2-deterministic \
QA_POST_HEALTH_WAIT_SECS=180 \
PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=earth2-deterministic \
LEPTON_RESOURCE_SHAPE=gpu.8xh100-sxm \
python -u qa/scripts/run_qa.py \
  --service rust \
  --image-tag v0.1.0 \
  --suite cicd \
  --lustre-dir cicd
```

## Output Publication Suite

The `publication` suite is Rust-only and verifies that the local result served by
PhysicsNeMo-Serve matches the artifact uploaded to one configured object-storage target.
It is intentionally not part of `smoke` or `cicd` because it requires live cloud
credentials and writes remote objects.

The QA runner passes publication settings as JSON environment overrides by default.
Set `QA_PUBLICATION_LOCAL_MOUNT_PATH` only when the deployed NFS path is also mounted
locally; the runner then writes a runtime config through that explicit local mapping
and passes the corresponding container path as both `WORKER_RUNTIME_CONFIG` and
`PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG`. The effective config contains the normal
worker role settings plus this top-level publication block:

```json
{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "s3",
      "bucket": "forecast-bucket",
      "prefix": "outputs"
    }
  }
}
```

Useful environment variables:

- `QA_PUBLICATION_STORAGE_TYPE`: `s3` or `azure`.
- `QA_PUBLICATION_PREFIX`: remote prefix, default `outputs`.
- `QA_PUBLICATION_WORKFLOW`: workflow to run, default `earth2-deterministic`.
- `QA_PUBLICATION_REQUEST_JSON`: optional JSON object with request parameters.
- `QA_PUBLICATION_TIMEOUT_SECS`: upload wait timeout, default `900`.
- `QA_PUBLICATION_S3_BUCKET`: S3 bucket for S3 publication.
- `QA_PUBLICATION_S3_REGION`: optional S3 region. Falls back to
  `AWS_DEFAULT_REGION` or `AWS_REGION`.
- `QA_PUBLICATION_S3_ENDPOINT`: optional S3 or S3-compatible endpoint. Falls
  back to `S3_ENDPOINT_URL`.
- `QA_PUBLICATION_AZURE_ENDPOINT`: Azure Blob endpoint, for example
  `https://<account>.blob.core.windows.net`.
- `QA_PUBLICATION_AZURE_CONTAINER`: Azure container name.
- `QA_PUBLICATION_LOCAL_MOUNT_PATH`: optional local path that maps to the deployed
  NFS root mounted at `/outputs`; without it, JSON environment overrides are used.
- `QA_PUBLICATION_COMPARE_IMAGE`: optional full image reference for the compare job.
- `QA_PUBLICATION_NODE_GROUP`, `QA_PUBLICATION_PULL_SECRET`,
  `QA_PUBLICATION_RESOURCE_SHAPE`, and `QA_PUBLICATION_LUSTRE_STORAGE`: optional
  compare-job overrides.

Compare-job settings use QA-specific environment variables first, then the matching
`LEPTON_*` environment variable, then `deploy/config.yaml`, then built-in defaults.
For the image, `QA_PUBLICATION_COMPARE_IMAGE` wins; otherwise a fully qualified
`--image-tag` is used directly, or the configured `docker_registry` and `image_name`
are combined with the tag.

Provider credentials are forwarded from environment variables only; do not put
secrets in request JSON or committed config files. S3 uses the usual AWS env
chain. Azure uses `AZURE_STORAGE_SAS_TOKEN`, `AZURE_STORAGE_ACCOUNT_KEY`,
`AZURE_STORAGE_ACCESS_KEY`, or default Azure credentials.

S3 example:

```bash
QA_PUBLICATION_STORAGE_TYPE=s3 \
QA_PUBLICATION_S3_BUCKET=<bucket> \
QA_PUBLICATION_PREFIX=outputs \
python -u qa/scripts/run_qa.py \
  --service rust \
  --image-tag <tag> \
  --suite publication \
  --lustre-dir publication
```

Azure example:

```bash
QA_PUBLICATION_STORAGE_TYPE=azure \
QA_PUBLICATION_AZURE_ENDPOINT=https://<account>.blob.core.windows.net \
QA_PUBLICATION_AZURE_CONTAINER=<container> \
AZURE_STORAGE_ACCOUNT_NAME=<account> \
AZURE_STORAGE_ACCOUNT_KEY=<storage-account-key> \
QA_PUBLICATION_PREFIX=outputs \
python -u qa/scripts/run_qa.py \
  --service rust \
  --image-tag <tag> \
  --suite publication \
  --lustre-dir publication
```

Concrete S3-compatible data upload QA command:

```bash
export LEPTON_WORKSPACE_ID="<workspace-id>"
export LEPTON_WORKSPACE_TOKEN="<workspace-token>"
export LEPTON_ENDPOINT_TOKEN="<endpoint-token>"

export AWS_ACCESS_KEY_ID="<access-key>"
export AWS_SECRET_ACCESS_KEY="<secret-key>"
export AWS_REGION="us-ashburn-1"
export S3_ENDPOINT_URL="https://<namespace>.compat.objectstorage.<region>.oraclecloud.com"

export QA_PUBLICATION_STORAGE_TYPE=s3
export QA_PUBLICATION_S3_BUCKET="<bucket>"
export QA_PUBLICATION_S3_REGION="$AWS_REGION"
export QA_PUBLICATION_S3_ENDPOINT="$S3_ENDPOINT_URL"
export QA_PUBLICATION_PREFIX="outputs/s3-fcn-rerun"
export QA_PUBLICATION_WORKFLOWS="deterministic_fcn_workflow"
export QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES=96
export PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID="e2s-deterministic-fcn"

python3 qa/scripts/run_qa.py \
  --service rust \
  --image-tag <tag> \
  --suite publication \
  --lustre-dir s3-fcn-rerun \
  --endpoint-name s3-fcn-upload \
  --endpoint-log-interval 60
```

Concrete Azure Blob data upload QA command:

```bash
export LEPTON_WORKSPACE_ID="<workspace-id>"
export LEPTON_WORKSPACE_TOKEN="<workspace-token>"
export LEPTON_ENDPOINT_TOKEN="<endpoint-token>"

export AZURE_STORAGE_ACCOUNT_NAME="<account>"
export AZURE_STORAGE_ACCOUNT="$AZURE_STORAGE_ACCOUNT_NAME"
export AZURE_STORAGE_ACCOUNT_KEY="<storage-account-key>"
export AZURE_STORAGE_ACCESS_KEY="$AZURE_STORAGE_ACCOUNT_KEY"

export QA_PUBLICATION_STORAGE_TYPE=azure
export QA_PUBLICATION_AZURE_ENDPOINT="https://<account>.blob.core.windows.net"
export QA_PUBLICATION_AZURE_CONTAINER="<container>"
export QA_PUBLICATION_PREFIX="outputs/azure-fcn-rerun"
export QA_PUBLICATION_WORKFLOWS="deterministic_fcn_workflow"
export QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES=96
export PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID="e2s-deterministic-fcn"

python3 qa/scripts/run_qa.py \
  --service rust \
  --image-tag <tag> \
  --suite publication \
  --lustre-dir azure-fcn-rerun \
  --endpoint-name azure-fcn-upload \
  --endpoint-log-interval 60
```

For deployed services, object-store destinations are configured in the runtime
config's top-level `output_publication.storage` block:

```json
{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "azure",
      "endpoint": "https://<account>.blob.core.windows.net",
      "container": "<container>",
      "prefix": "outputs"
    }
  }
}
```

Upload performance is configured separately on `roles.publish.config`, for
example:

```json
{
  "roles": {
    "publish": {
      "config": {
        "max_concurrent_files": 96,
        "client_options": {
          "timeout_secs": 300,
          "connect_timeout_secs": 10,
          "pool_max_idle_per_host": 192
        },
        "retry": {
          "max_retries": 10,
          "timeout_secs": 300
        }
      }
    }
  }
}
```

## Stress Suite

The `stress` suite is separate from `cicd`. It is intended for load and
stability validation, so it is longer-running and costlier than the functional
checks. The sustained concurrency test keeps `QA_STRESS_CONCURRENCY_PER_GPU`
requests in flight per visible GPU. For example, an 8 GPU endpoint with
`QA_STRESS_CONCURRENCY_PER_GPU=2` targets 16 concurrent requests. As requests
finish, the test submits replacements until `QA_STRESS_DURATION_SECS` expires,
then stops submitting and drains outstanding requests.

Useful environment variables:

- `QA_STRESS_WORKFLOW`: workflow to run. The default is `earth2-deterministic`.
- `QA_STRESS_GPU_COUNT`: expected visible GPU count. When set, the test waits
  for `/metrics` to report exactly this count.
- `QA_STRESS_CONCURRENCY_PER_GPU`: target in-flight requests per GPU. The
  default is `2`.
- `QA_STRESS_DURATION_SECS`: active stress duration. The default is `600`.
- `QA_STRESS_DRAIN_TIMEOUT_SECS`: maximum time to wait for outstanding requests
  after the active interval. The default is `1800`.
- `QA_STRESS_REQUEST_JSON` or `QA_STRESS_REQUEST_FILE`: request parameters for
  workflows without a built-in stress preset.
- `QA_STRESS_SUMMARY_PATH`: optional path for the JSON summary. By default, the
  test writes `qa/inference/reports/stress_summary_<timestamp>.json`.

The stress test first runs a readiness probe for the configured workflow so it
does not start filling concurrency until the endpoint can actually schedule and
complete that workflow. At the end it prints and writes a JSON summary with
submitted/completed/failed counts, throughput, p50/p90/p99 latencies, per-GPU
completion counts, in-flight concurrency, and example errors.

```bash
QA_STRESS_GPU_COUNT=8 \
QA_STRESS_WORKFLOW=earth2-deterministic \
QA_STRESS_CONCURRENCY_PER_GPU=2 \
QA_STRESS_DURATION_SECS=600 \
QA_STRESS_DRAIN_TIMEOUT_SECS=1800 \
PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=earth2-deterministic \
LEPTON_RESOURCE_SHAPE=gpu.8xh100-sxm \
python -u qa/scripts/run_qa.py \
  --service rust \
  --image-tag v0.1.0 \
  --suite stress \
  --lustre-dir stress
```

## Lepton CRPS Comparison

The dual-endpoint comparison runner deploys an Earth2Studio Python baseline and
a PhysicsNeMo Serve Rust candidate, submits matched seeded requests to both workflows,
extracts their Lustre-backed forecast Zarr paths, and launches Lepton batch jobs
that run `compare_crps.py`. By default, the Python baseline is compared with two
PhysicsNeMo Serve candidate materialization modes: `scheduled_gpu` and `prepare_cpu`.
The default baseline and candidate requests are separate payloads. Both are
seeded (`seed_base: 1000`) and use Gaussian perturbation so the two ensemble
outputs are statistically comparable.
Generated endpoint names use the comparison role and image family, for example
`crps-e2s-python-base-<run-id>` and `crps-e2s-rust-ff-<run-id>`.
Both endpoints receive `DEFAULT_OUTPUT_DIR=/outputs` and
`RESULTS_ZIP_DIR=/outputs` so the Python baseline and Rust candidate write
results to the shared Lustre mount.

```bash
export LEPTON_WORKSPACE_ID=<WORKSPACE_ID>
export LEPTON_WORKSPACE_TOKEN=nvapi-...  # token only; do not include "<WORKSPACE_ID>:"

python -u qa/scripts/run_lepton_crps_compare.py \
  --workspace-id "$LEPTON_WORKSPACE_ID" \
  --baseline-image-tag <DOCKER_REGISTRY>/<PYTHON_SERVICE_IMAGE>:<TAG> \
  --candidate-image-tag <DOCKER_REGISTRY>/<IMAGE_NAME>:<TAG> \
  --baseline-workflow ensemble_workflow \
  --candidate-workflow earth2-ensemble-fanout \
  --baseline-request-json qa/inference/requests/crps_baseline_request.json \
  --candidate-request-json qa/inference/requests/crps_candidate_fanout_request.json \
  --candidate-materialization-modes scheduled_gpu,prepare_cpu \
  --threshold 0.01 \
  --run-timeout 10800 \
  --run-poll-interval 120 \
  --artifact-dir qa/artifacts
```

For a smaller first pass, swap in the smoke request payloads:

```bash
python -u qa/scripts/run_lepton_crps_compare.py \
  --workspace-id "$LEPTON_WORKSPACE_ID" \
  --baseline-request-json qa/inference/requests/crps_baseline_smoke_request.json \
  --candidate-request-json qa/inference/requests/crps_candidate_fanout_smoke_request.json \
  --artifact-dir qa/artifacts
```

Required environment:

- `LEPTON_WORKSPACE_TOKEN`: token used by `lep login`. Use the token value only,
  for example `nvapi-...`. Do not include the workspace id prefix; the scripts
  construct `<workspace-id>:<token>` internally.
- `LEPTON_ENDPOINT_TOKEN`: optional; generated for the run when omitted.

Useful flags:

- `--dry-run`: print endpoint and CRPS job commands without submitting requests.
- `--skip-teardown`: leave endpoints running for debugging.
- `--variables`: comma-separated CRPS variables passed to `compare_crps.py`.
- `--lustre-dir`: subdirectory under `<NFS_MOUNT_BASE>/`; defaults to
  `crps_tests_<YYYYMMDD>`.
- `--candidate-resource-shape`: Lepton shape for the PhysicsNeMo Serve candidate;
  defaults to `gpu.4xh100-sxm`.
- `--candidate-materialization-modes`: comma-separated PhysicsNeMo Serve fanout modes to
  compare against the Python baseline. Defaults to `scheduled_gpu,prepare_cpu`.
  Pass one mode, such as `scheduled_gpu`, for a single candidate run.
- `--comparison-image-tag`: image used for the CRPS batch job; defaults to the
  Earth2Studio baseline image because that image contains `compare_crps.py`.
  The default script path is
  `/workspace/earth2studio-project/serve/server/scripts/compare_crps.py`.
- `--run-timeout`: end-to-end workflow timeout in seconds. The full 512-member
  Python baseline can take more than one hour, so the CI default is `10800`.

Artifacts are written under `qa/artifacts/crps-compare/<run-id>/`, with endpoint
logs under `qa/artifacts/endpoint-logs/` and CRPS job logs under
`qa/artifacts/crps-jobs/`. The final result summary is:

```bash
qa/artifacts/crps-compare/<run-id>/summary.json
```

`summary.json` includes the endpoint names, baseline run id and Zarr path, one
entry under `comparisons` per candidate materialization mode, CRPS job exit
codes, final result, and parsed CRPS report fields such as
`comparison_report.result`, `comparison_report.max_relative_diff_percent`, and
`comparison_report.threshold_percent`.

## Lepton Rust I/O Benchmark

The Rust I/O benchmark runner submits a Lepton batch job that compares the
optimized Rust Zarr backend with Earth2Studio's Python async and synchronous Zarr
backends. The job streams benchmark output to Lepton logs while also teeing the
same output into `job-output.log`, and it writes both machine-readable JSON and a
markdown performance comparison report.

The `benchmark-report` preset uses report-style per-model Rust pool sizing and
runs the models that are valid on the current image:

- `fcn`
- `dlwp`
- `sfno`
- `stormcast`
- `fcn3`

```bash
export LEPTON_WORKSPACE_ID=<WORKSPACE_ID>
export LEPTON_WORKSPACE_TOKEN=nvapi-...  # token only; do not include "<WORKSPACE_ID>:"

uv run python qa/scripts/run_lepton_rust_io_benchmark.py \
  --image-tag v0.1.20260529.0 \
  --run-id rio-doc2 \
  --preset benchmark-report \
  --device cuda:0 \
  --lustre-dir rust_io_tests_20260528 \
  --resource-shape gpu.h100-sxm \
  --job-poll-interval 60 \
  --job-timeout 21600
```

For a quick smoke run:

```bash
uv run python qa/scripts/run_lepton_rust_io_benchmark.py \
  --image-tag v0.1.20260529.0 \
  --run-id rio-smoke \
  --models fcn \
  --nsteps 1 \
  --device cuda:0 \
  --lustre-dir rust_io_tests_20260528 \
  --resource-shape gpu.h100-sxm
```

Useful flags:

- `--dry-run`: print the Lepton job command without submitting it.
- `--preset benchmark-report`: use the report-style model list and per-model
  Rust pool sizing.
- `--models`: comma-separated model override, for example `fcn,sfno,stormcast`.
- `--backends`: comma-separated backend list; defaults to
  `rust,py_async,zarr_sync`.
- `--job-timeout`: Lepton job timeout in seconds. Report-style runs can take tens
  of minutes.
- `--keep-job`: leave the Lepton batch job after completion for debugging.

Artifacts are written under `qa/artifacts/rust-io-benchmark/<run-id>/`:

```bash
qa/artifacts/rust-io-benchmark/<run-id>/summary.json
qa/artifacts/rust-io-benchmark/<run-id>/benchmark-summary.json
qa/artifacts/rust-io-benchmark/<run-id>/perf_compare_report.md
```

The markdown report contains only current test results: model status, total I/O,
I/O write time, wall time, compute time, and current Rust speedups. It does not
compare against `docs/benchmark_report_e2s_zarr_io.md`.

Lepton job names include the Rust I/O context, preset, model scope, and run id,
for example `ff-rio-report-rio-doc2` or `ff-rio-custom-fcn-rio-smoke`. Job logs
are stored under:

```bash
qa/artifacts/rust-io-jobs/<lepton-job-name>.log
```
