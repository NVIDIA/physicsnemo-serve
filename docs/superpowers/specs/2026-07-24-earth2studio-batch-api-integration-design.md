# Earth2Studio Batch API Integration Design

## Goal

Update the existing `earth2-deterministic-batch` plugin to delegate each
scheduler-formed request batch to Earth2Studio's public deterministic batch API.
Package the unreleased API reproducibly, publish an AMD64 service image to NVCR,
and verify the complete path with four concurrent forecasts on an isolated
Lepton H100 endpoint.

## Architecture

PhysicsNeMo Serve remains responsible for request validation, compatibility
grouping, batch timing, GPU memory fit, reservation, dispatch, per-run output
registration, and result persistence. The plugin is a thin boundary adapter:

1. `prepare()` preserves the current resource and batch profiles.
2. `run_batch()` registers one `forecast.zarr` output for every scheduler item.
3. Each item becomes an Earth2Studio `DeterministicBatchRequest`.
4. The plugin calls `run_deterministic_batch()` once with a process-cached
   `DeterministicBatchRuntime`.
5. Ordered Earth2Studio responses become typed plugin outputs or per-item
   failures.
6. Plugin cleanup closes the Earth2Studio runtime and releases Python/Torch
   resources.

The public plugin request and result schemas do not change. No fallback to the
legacy direct `earth2studio.run.deterministic()` path is retained.

## Cold-Start Model Loading

Lepton starts one execute process per GPU. Those processes share the container's
Earth2Studio cache and can receive their first DLWP requests concurrently.
`DLWP.load_default_package()` materializes multiple files in that shared cache;
without coordination, one process can try to open a file while another is still
writing it.

The plugin supplies `DeterministicBatchRuntime` with a DLWP `model_loader`
callback. The callback:

1. Opens a process-shared lock file under `/tmp`.
2. Acquires an exclusive advisory `flock`.
3. Calls `DLWP.load_default_package()` and `DLWP.load_model()`.
4. Releases the lock after the model is fully constructed, including on errors.

The lock covers only each process's first model load. Earth2Studio continues to
cache the model in each runtime, so forecasts do not serialize after
initialization. The existing `prepare_model_cache()` hook uses the same loader
to avoid introducing a second uncoordinated download path.

`fcntl.flock` is appropriate here because the production container is Linux and
all execute processes share its `/tmp` directory and Earth2Studio cache. This
change deliberately remains in the plugin rather than modifying Earth2Studio's
general package downloader.

## Runtime Worker Topology

The scheduler owns compatibility grouping, the 200 ms batch window, and release
of work to execute processes. There is no standalone `batch` stream or
`worker-runtime` role in `worker_runtime_config.json`.

The service entrypoint therefore does not create or supervise a
`worker-runtime-batch` process. `WORKERS=all` starts the configured orchestration
roles and execute launcher only. Explicit legacy `batch` entries in `WORKERS`
are ignored rather than launching a role that the runtime rejects.

The plugin batch profile remains unchanged at `max_batch_size=4` and
`max_wait_ms=200`.

## Dependency Contract

The service image installs Earth2Studio from the public fork at immutable commit
`c7e3b772e3d28fb0a5b5c1d5b9669533ee392daf`, with the existing model and data
extras. The plugin readiness contract explicitly requires `obstore` and
`earth2studio.batched_workflows`.

Those current extras require `nvidia-physicsnemo==2.1.1`,
`warp-lang==1.14.0`, and SciPy 1.15.2 or newer. PhysicsNeMo's non-PyTorch
dependencies are installed explicitly because its final-release PyTorch metadata
would otherwise replace the CUDA 13 base image's ABI-compatible NVIDIA
2.10/0.25 builds with cu128 wheels. Earth2Studio is likewise installed without
transitive resolution after its selected extras have been installed explicitly.
The image pins the base-compatible fsspec/gcsfs/s3fs 2025.10 family and SciPy
1.16.3 instead of retaining its incompatible 1.13.1 downgrade.

The image build verifies that the installed module exposes:

- `DeterministicBatchRequest`
- `DeterministicBatchRuntime`
- `run_deterministic_batch`

## Error Handling and Lifecycle

Earth2Studio owns forecast staging, atomic output replacement, resource reuse,
and conversion of forecast exceptions into per-item failed responses. The
plugin preserves response order, forwards failed responses through
`BatchItemResult.failed`, and logs batch-level start/completion summaries without
credentials.

The workflow owns one runtime instance for its process cache lifetime. Cleanup
calls `close()`, clears the runtime reference, and then runs generic Python/Torch
cleanup. A batch-wide exception remains a framework execution failure; an item
failure does not discard successful sibling items.

## Verification and Delivery

Focused Earth2Studio, plugin, scheduler, and Redis queue tests must pass. The
resulting `linux/amd64` image is pushed under a unique
`batch-api-YYYYMMDDTHHMMSSZ` tag in
`nvcr.io/dycvht5ows21/scicomp-ferroflux`, and its remote digest is recorded.

Update the existing `kkarnam-test-ferroflux-cache` endpoint in workspace
`r0whe339` to the corrected image while preserving its existing resource shape,
pull secret, storage mounts, and authentication configuration.

After a fresh rollout, submit concurrent DLWP requests before the cache is warm.
Acceptance requires every request to succeed with a distinct output path and no
partial-cache/xarray backend error. Verify that endpoint logs contain no
`role 'batch' not found in config` restart loop. Repeat the requests against the
warm cache to confirm runtime reuse. Batch cardinality is observed but is not an
acceptance requirement because the approved correction preserves the 200 ms
window. Retain the running endpoint and collect only sanitized evidence; never
record its bearer token.
