# Earth2 Multi-Model Deterministic Batch Design

## Goal

Generalize `earth2-deterministic-batch` from a DLWP-only workflow into one
deterministic batching plugin that supports DLWP, FCN, and FCN3 without changing
the Earth2 Studio API or the plugin's request and response shapes.

## Architecture

The plugin owns an explicit immutable model registry. Each entry supplies the
canonical request name, a lazy Earth2 model loader, a GFS data factory, a
display/cache name, and its scheduling memory policy. Initial entries are
`dlwp`, `fcn`, and `fcn3`.

The process-cached workflow retains at most one
`DeterministicBatchRuntime`. Requests for the same model reuse it. A request for
a different model closes the old runtime, releases Python and CUDA resources,
and lazily constructs the requested runtime. Model cache population is
serialized across worker processes with a distinct file lock for each model.

Preparation normalizes the model name, derives scheduling and batch profiles
from the registry, and uses the canonical model as the batch key. Execution
rejects mixed-model batches before allocating outputs, registers one Zarr
output per item, and calls Earth2 Studio's `run_deterministic_batch()` once.

## Public Contract

The plugin manifest version becomes `1.1.0`. The `model` enum accepts `dlwp`,
`fcn`, and `fcn3`; all other request and result fields remain unchanged. All
three models use GFS, one `earth2-gpu` executor, a maximum batch size of four,
and a 200 ms batch window.

| Model | Single request | Batch shared | Per item | Four-item batch |
|---|---:|---:|---:|---:|
| DLWP | 2,048 MiB | 8,192 MiB | 1,024 MiB | 12,288 MiB |
| FCN | 6,144 MiB | 8,192 MiB | 1,024 MiB | 12,288 MiB |
| FCN3 | 65,536 MiB | 65,536 MiB | 1,024 MiB | 69,632 MiB |

Model loading is lazy. Worker readiness imports all supported classes and the
Earth2 batch API but does not download weights.

## Failure and Cleanup Semantics

A model-load failure is attempted once for the affected batch, leaves no active
runtime, and may be retried by the next batch. Forecast failures remain
per-item and do not evict a successfully loaded runtime. Runtime cleanup is
idempotent and is used for model switches, failed construction, and workflow
shutdown.

## Verification and Rollout

Unit tests cover registry resolution, profiles, homogeneous batching, locks,
same-model reuse, model switching, load failure recovery, ordered results,
per-item failures, and cleanup. Repository checks include plugin validation,
Earth2 plugin tests, entrypoint tests, lint, shell syntax, and image imports.

The image is built for `linux/amd64`, tagged as
`nvcr.io/dycvht5ows21/scicomp-ferroflux:batch-multimodel-<UTC timestamp>`, and
pushed to NVCR. The existing Lepton deployment is updated in place while
preserving its token, two-H100 allocation, model-cache mount, registry secret,
and batching configuration. Acceptance requires five consecutive authenticated
health responses, cold and warm requests for all three models, successful
switch-back to DLWP, distinct readable outputs, and clean model lifecycle logs.
The previous immutable image is the rollback target.
