# Native worker and serving integration

The native worker connects the [C++ workflow host](../native-inference/workflows/runtime/README.md)
to the existing Rust API, scheduler and results roles. One deployment serves one
workflow with one selected compiled backend. The native pipeline is
`schedule → execute(native_workflow) → results`; the worker keeps its C++ host
and model sessions resident across requests.

This integration consumes prepared workflow configuration and compiled model
packages. Application authoring, release assembly and publication are separate
work. See the [worker reference](../crates/native-worker/README.md) for the worker
contract and [REST guide](inference-service-user-guide.md) for request, status,
result and artifact routes.

## Build and link the worker

The default Rust feature set supports CPU unit and Redis component checks without
the C++ SDK, CUDA or model packages. A serving binary requires the `cpp` feature
and the installed headers and shared libraries of the matching workflow host.
For a generic tensor or custom workflow:

```sh
export NATIVE_WORKFLOW_ONLY=1
export NATIVE_WORKFLOW_INCLUDE=/path/to/workflow-host/include
export NATIVE_WORKFLOW_LIB_DIR=/path/to/workflow-host/lib
export LD_LIBRARY_PATH=/path/to/workflow-host/lib:/path/to/inference-sdk/lib:${LD_LIBRARY_PATH:-}
cargo build --locked --release -p native-worker --features cpp
```

`NATIVE_WORKFLOW_ONLY=1` omits Transolver headers and linkage. To include the
schema-1 Transolver adapter, omit that flag and also supply
`NATIVE_RUNNER_INCLUDE` and `NATIVE_RUNNER_LIB_DIR` for the installed Transolver
runner. The workflow include/library settings default to the runner settings
when their explicit values are absent. Keep dependent provider libraries on the
loader path and build all linked components with compatible C++ ABI settings.

Host construction, startup checks, requests and destruction run on one dedicated
OS thread with a 16 MiB stack. A single execution gate remains held through
completion. Shutdown and cancellation can drain work between native operations;
they cannot preempt a running CUDA kernel.

## Select the runtime configuration

The [native runtime configuration](../native-inference/deploy/worker_runtime_config.json)
enables the existing scheduler and results roles, disables batching and declares
no Python runtime environments. Set `WORKER_RUNTIME_CONFIG` to this configuration
when using `scripts/entrypoint.sh`. The API's
`PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG` must identify the same configuration; the
entrypoint replaces its baked-in default when a different
`WORKER_RUNTIME_CONFIG` is selected. Keep the Redis URL and stream prefix
consistent across the API, scheduler, results role and native worker.

Set `NATIVE_DEPLOYMENT_CONFIG` to the deployment JSON and enable
`WORKERS=server,scheduler,results,execute`. The entrypoint then supervises
`NATIVE_WORKER_BIN` directly for execution. A missing native binary fails startup.
The worker also accepts the deployment JSON as its first command-line argument:

```sh
target/release/native-worker /path/to/deployment.json
```

Schema-2 deployment configuration selects `tensor.v1` or `custom.v1`, declares
the model aliases and per-alias compilation sources, request/response schemas,
hashed model assets, optional release inputs and multipart input rules, startup
cases, artifact/cache roots and deadlines. Schema-1 Transolver deployments
retain their fixed full/tail packages and case contract. Device admission checks
the declared product, compute capability, MIG policy, available memory and
driver bounds before the worker registers. A `build-product-v1` target records
exact build-product compatibility; it does not establish GPU qualification.

The worker verifies assets, registers as `warming`, constructs the host and runs
model startup checks before enabling execution. Named schema-2 startup cases
replay authored requests through the workflow and shared output validators.
The default schema-2 startup deadline is 300 seconds. Failed, cancelled or
uncertain startup leaves admission closed.

## Requests, outputs and provenance

Native plugins require explicit schemas and the native pipeline. Requests use
the normal API validation and scheduler envelope. Rust passes canonical validated
parameters to C++; unvalidated raw fields cannot replace them. Declared multipart
files are checked against their media type, size, digest and run ownership, then
copied into a directory owned by that execution attempt. C++ receives admitted
paths and metadata rather than request-supplied filesystem paths or URLs.

Each execution writes into its own artifact directory. Rust checks returned
artifact names, containment, regular-file status, sizes and hashes, and validates
the response schema before publishing results through the existing results role.
The generic adapter permits shapes declared by its workflow; it does not impose
the Transolver field layout.

`SOURCE_SHA` identifies the worker/runtime source. Set exactly one immutable
execution identity:

| Environment variable | Identity | Value |
| --- | --- | --- |
| `RELEASE_IMAGE_DIGEST` | `registry-image` | `registry/repository@sha256:<64 hex>` |
| `PNMS_IMAGE_CONFIG_DIGEST` | `image-config` | `sha256:<64 hex>` Docker configuration ID |
| `PNMS_BUNDLE_DIGEST` | `bundle` | `<64 hex>` bundle inventory digest |

Registration, results and readiness carry the typed `runtime_identity`. Only a
registry image carries `release_image`. Schema-2 `model_build_sources` records
the compilation source for every model alias; aliases can retain different
sources. Legacy schema-1 configurations without that map require
`MODEL_BUILD_SOURCE_SHA`. Reusing a model does not replace its compilation
source with the current worker source.

## Readiness and worker restart

`/readyz` reads the live native registry row and owner lease in one
Redis execution. It checks the deployment configuration hash, workflow, backend,
execution identity, current owner and successful startup proof. Declared workflow
startup cases and custom initialization must also have passed. A healthy busy
worker can remain ready; warming, stale, expired or mismatched ownership returns
unready. `scripts/health_stub.py` uses this readiness result for native services.

Set stable `NATIVE_DEPLOYMENT_ID` and `NATIVE_WORKER_SLOT` values; their defaults
are the workflow ID and `0`. The worker owns a 15-second slot lease and renews it
once per second. A supervised replacement uses the same queue and consumer group
after the previous lease expires. Retirement retains a `warming` logical slot,
clears its loaded-model cache and releases only the live process lease. The
replacement changes ownership in place. Readiness remains false during this gap.
The unchanged scheduler already waits for warming workers; the native runtime
configuration refreshes its inventory once per second.

Before starting the scheduler, the native entrypoint runs
`native-worker wait-registration <deployment.json>`. This waits for the exact
deployment configuration, source, execution identity and live slot owner, then
starts the existing scheduler. A timeout stops the launcher instead of starting
with an empty registry. Manual launchers must apply the same gate. Initial Redis
and NVML discovery still use the existing scheduler's error handling.

Redis fences each attempt by the source delivery, consumer and attempt token.
Completion atomically publishes the outcome, releases the original reservation
and acknowledges the delivery. Durable markers resolve lost completion replies
without executing again. Uncertain claims or commits retire the worker and leave
the delivery pending for recovery. Deterministically malformed dispatch metadata
retains the engine's quarantine behavior.

The supported restart boundary keeps Redis, API, scheduler and artifact storage
available while one native worker restarts in the same slot. Redis must preserve
pending messages and completion markers with a non-evicting policy; the native
entrypoint selects `noeviction` for embedded Redis. A hard crash may leave an
available cached row: requests can queue on its durable stream until the
replacement starts, and readiness rejects its expired lease. A stopped supervisor
can leave this work pending, so automatic worker restart is required. Whole-container or
Redis loss, scheduler reservation loss and active-active GPU failover require a
broader deployment design.

## CPU component validation

With `redis-server`, Python and the normal Rust build dependencies available:

```sh
cargo test --locked -p native-worker -p worker-runtime -p inference_server -p scicomp-rq
cargo clippy --locked -p native-worker -p worker-runtime -p inference_server -p scicomp-rq --all-targets -- -D warnings
PYTHONPATH=.:python/e2s_tools uv run pytest tests/test_entrypoint.py tests/test_native_health_stub.py tests/test_plugin_dev.py -q \
  -k 'not test_plugin_dev_run_local_postprocess_stack_serves_structured_results'
```

These checks cover worker lifecycle and request contracts, Redis completion and
recovery, scheduler placement, live API readiness, native manifests and entrypoint
selection. Default Rust tests use fake native hosts and do not compile the C++
bridge. The Linux CPU CI job additionally installs the mock workflow host and
tests the `cpp` feature with `NATIVE_WORKFLOW_ONLY=1`. Its real C++ configuration
and startup rejection checks exercise linkage without model inference.

CPU fixtures require the explicit `cpu-test-only` target and
`NATIVE_ALLOW_CPU_TEST_ONLY=1`; their registration records CPU-only scope. CPU
passes are component evidence. GPU acceptance requires a separate run against
the exact source and image on the configured remote H100 resources. Historical
Lepton receipts keep their original source, image and test inventory and do not
qualify this integration branch. This integration makes no performance claim.
