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

## Dependency Contract

The service image installs Earth2Studio from the public fork at immutable commit
`c7e3b772e3d28fb0a5b5c1d5b9669533ee392daf`, with the existing model and data
extras. The plugin readiness contract explicitly requires `obstore` and
`earth2studio.batched_workflows`.

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

Deploy `kkarnam-pnserve-batch-api` to workspace `r0whe339` on
`gcp-iad-lepton-002-vnbwicri` with one `gpu.h100-sxm`, pull secret
`scicomp-dev`, and
`/PhysicsNeMo/platform/kkarnam/rust_runs/batch-api` mounted from
`node-nfs:lustre` at `/outputs`.

Submit four concurrent DLWP requests. Acceptance requires four successful run
IDs, one shared batch ID, `batch_size=4`, `flush_reason=max_batch_size`, four
distinct output paths, and one Earth2Studio batch start/completion log pair.
Collect sanitized evidence and stop the endpoint after verification while
retaining the image, endpoint definition, and artifacts.
