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
- `test_basic.py` - Core workflow tests (health, list workflows, run workflows)
- `test_negative.py` - Invalid parameter tests
- `test_experiments.py` - Experimental/exploratory tests

## Workflows Tested

- `deterministic_fcn_workflow`
- `deterministic_workflow`
- `diagnostic_workflow`

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
  defaults to `gpu.8xh100-sxm`.
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
