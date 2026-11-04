# Native workflow worker

This worker serves one compiled workflow per deployment on one selected GPU. The
original schema-v1 Transolver surface deployment remains supported. Schema-v2
deployments select the linked `tensor.v1` adapter or a source-linked `custom.v1`
factory through the reusable C++ workflow host.
The Rust API, scheduler and results roles use their existing Redis contracts.
Its explicit pipeline is `schedule → native execute → results`; no Python hook
or per-request C++ subprocess executes inference.

`HostClient` creates, runs and destroys the C++ runner on a dedicated OS thread.
This thread explicitly reserves a 16 MiB stack. CUDA driver initialization on the
qualified H100 image overflowed Rust's default 2 MiB stack before registration;
the 16 MiB size passed startup and graceful shutdown on that same image. This
setting is independent of `RUST_MIN_STACK` and Tokio worker-thread settings.
The runner keeps model sessions and normalization statistics resident. A single
execution gate remains held until the outcome is committed. A dropped handler,
lost lease or uncertain completion retires admission; native allocations remain
owned by the native thread until execution drains. Cancellation checks between
blocks cannot preempt a running CUDA kernel.

The worker verifies deployment model hashes at startup and checks the exact
qualified H100 product/sm90, disabled MIG, available-memory requirement and
configured driver range. It registers as `warming` before constructing the host.
User bundles use the separate `build-product-v1` target with
`hardware_compatibility=build-product-only`: it admits the recorded exact GPU
product, compute capability and CUDA architecture, disabled MIG, available
memory and driver bounds. This admission does not claim Lepton QA qualification
or portability to a different GPU product. Existing qualified H100 profiles
retain their original restrictions. Admission resolves the configured CUDA logical
ordinal through the driver, then selects that exact physical GPU in NVML by UUID.
This honors `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` without treating a CUDA
ordinal as an NVML index. Optional `PNMS_DEVICE_UUID=GPU-<uuid>` asserts the resolved
identity; a mismatched assertion fails startup instead of selecting another GPU.
New v1 CFD releases also include this full `target` policy. Legacy v1 configs
without it retain their original exact H100/sm90 and disabled-MIG checks; they
do not acquire a driver or available-memory qualification retroactively.
Host construction and model-executing self-tests run on the same native thread;
only success publishes `available` and enables execute polling. Generic models
use authored numerical input/output goldens. New Transolver releases provide a
hashed `startup_fixture_path` covering both full and tail sessions; old v1 configs
without it report the weaker `finite-static-inputs-v1` check explicitly.
Readiness requires the policy selected by the deployment; fixture-backed and
schema-v2/v3 releases cannot become ready using the legacy check.
Startup has a 300-second default deadline, supports shutdown cancellation, and
leaves admission closed after a failure or abandoned startup future.

Transolver request inputs are fixed in the deployment inventory and verified by
size and SHA-256 before use. Attempt-specific artifact directories prevent stale
attempts from overwriting accepted results. Valid execution failures atomically
publish a failure and release the exact original reservation. Malformed dispatch
metadata is left to the engine's quarantine policy; the worker invents no release.
Source stream, group, message and run identifiers are validated before claiming
an attempt, so deterministic malformed metadata keeps that quarantine policy.
Completion retries retain the same token and output bytes for three attempts,
each bounded to ten seconds. Durable Redis markers resolve lost responses.
If the claim or completion remains uncertain, the worker retires and leaves the
original delivery pending for fenced recovery. Infrastructure uncertainty does
not enter the generic dead-letter path or acknowledge an unreleased reservation.
Retired or shutting-down hosts also defer deliveries, including retirement during
submission and loss of the native reply channel, so replacement can recover them
without publishing an infrastructure interruption as a terminal model failure.
Admitted multipart copies move with the native command: cancellation during
submission discards unqueued copies; cancellation during C++ execution retains
those bytes until the real call returns. Normal completion, rejected commands and
native-thread unwind reclaim the owned copies before releasing the execution gate.
API staging inputs remain under the API's own retention policy.

Schema-v2 passes schema-validated canonical parameters to its linked host.
JSON-array FP32 inputs support bounded arbitrary values. Raw FP32 input artifacts
initially refer to release-declared logical names in `input_artifacts`, backed by
hashed local files or pinned HTTPS objects; request-supplied paths or URLs never
become admitted C++ paths. Returned artifacts must be relative regular files
inside the unique attempt directory. Rust checks names, containment, sizes and
hashes, and optionally validates the response against `response_schema` before
publishing. Generic outputs do not inherit the Transolver `[N,4]` constraint.

The Transolver request is:

```json
{"case_id":"run_1","seed":42,"capture_inputs":false}
```

It returns source-order FP32 pressure/WSS fields, standardized fields and native
provenance. Full `run_1` contains 8,828,095 cells: 147 blocks of 60,000 and one
8,095-cell tail. Both compiled packages must be present. Scientific metric
evaluation runs separately using the pinned Python evaluator; these fields are
not the Python benchmark's JSON/CSV/HTML report response.

Build and install the [C++ Inference SDK](../../native-inference/README.md) and
[workflow host](../../native-inference/workflows/runtime/README.md), then link
the Rust worker to their installed headers and libraries. A generic tensor or
custom workflow does not require the Transolver runner:

```sh
export NATIVE_WORKFLOW_ONLY=1
export NATIVE_WORKFLOW_INCLUDE=/opt/physicsnemo-workflow/include
export NATIVE_WORKFLOW_LIB_DIR=/opt/physicsnemo-workflow/lib
cargo build --release --locked -p native-worker --features cpp
```

Supply an already compiled model inventory through the
[deployment configuration](src/deployment.rs). Start Redis, the API and results
roles with the same stream prefix and
[native runtime configuration](../../native-inference/deploy/worker_runtime_config.json),
then start the worker and gate the existing scheduler on its registration:

```sh
export SOURCE_SHA='FULL_GIT_REVISION'
export RELEASE_IMAGE_DIGEST='registry/repository@sha256:SHA256_DIGEST'
export REDIS_URL=redis://127.0.0.1:6379
./target/release/native-worker /absolute/path/to/deployment.json &
./target/release/native-worker wait-registration /absolute/path/to/deployment.json &&
./target/release/worker-runtime --role scheduler --config-path native-inference/deploy/worker_runtime_config.json
```

The native container entrypoint applies this gate automatically. It waits for the
exact configuration, source, runtime identity and live owner before starting the
scheduler, including after a scheduler restart. Timeout fails closed. The gate
admits a matching warming worker; model self-tests still control availability.

Startup never builds models. Deployment requires `SOURCE_SHA` and exactly one
execution identity:

| Execution source | Environment variable | Value |
| --- | --- | --- |
| Published registry image | `RELEASE_IMAGE_DIGEST` | `registry/repository@sha256:<64 hex>` |
| Locally built Docker image | `PNMS_IMAGE_CONFIG_DIGEST` | `sha256:<64 hex>` Docker configuration ID |
| Host native bundle | `PNMS_BUNDLE_DIGEST` | `<64 hex>` bundle inventory digest |

Worker registration, result provenance and HTTP readiness carry the typed
`runtime_identity`. Only a registry image carries `release_image`; local image
configuration IDs and bundle digests are never presented as registry manifest
digests. Managed fresh-storage claims continue to require a registry image.
HTTP readiness compares the exact typed source and live worker ownership.

Schema-v2 requires `model_build_sources`, mapping every declared model alias to
its full compilation Git revision; these sources can differ when qualified
compiled packages are reused. Schema-v1 also accepts this optional map, covering
each compiled package with a simple alias. A nonempty map is authoritative and
requires no global environment SHA. A common SHA is retained in
`model_build_source_sha` only when every alias agrees; mixed sources omit it.
Legacy v1 without the map retains `MODEL_BUILD_SOURCE_SHA`. Requalification
source revisions do not replace original model compilation sources.
Python deployments keep their current build path. Native Redis must
retain completion markers and pending messages; use durable persistence and a
non-evicting policy. The supported recovery topology keeps Redis, API and scheduler
alive while one supervised native worker restarts in the same slot. Set stable
`NATIVE_DEPLOYMENT_ID` and `NATIVE_WORKER_SLOT` values (defaults: workflow ID and
`0`); pod names do not change the queue. An atomic 15-second per-slot lease prevents
simultaneous ownership, one-second heartbeats renew it during startup and work,
and a replacement acquires the slot only after the old lease expires or is
released. Retirement preserves a `warming` logical row and clears loaded-model
entries while atomically releasing the process lease. A retired owner cannot
become available again or overwrite its replacement. The unchanged scheduler
waits for warming capacity and observes readiness through its periodic discovery
(one second in the native configuration). Pending messages are reclaimed on the
same consumer group. Heartbeats do not add expiry
semantics to scheduler registry discovery: a stopped supervisor can leave a stale
row, so automatic worker restart is required. Redis loss, scheduler reservation
loss and active-active GPU failover are outside this topology.
An upgrade from the pilot's pod-named queues must drain those queues before
switching to stable slots; this worker does not migrate pending legacy streams.

CPU tests require no SDK, Torch, CUDA or GPU:

```sh
cargo test -p native-worker
cargo clippy -p native-worker --all-targets -- -D warnings
```

The `cpp` feature explicitly enables the C++ bridge. Its build requires installed
runner headers/libraries via `NATIVE_RUNNER_INCLUDE` and `NATIVE_RUNNER_LIB_DIR`.
The reusable host is linked through `NATIVE_WORKFLOW_INCLUDE` and
`NATIVE_WORKFLOW_LIB_DIR`, defaulting to those same runner directories.
For generic tensor/custom images, `NATIVE_WORKFLOW_ONLY=1` uses only those
workflow include/library variables and omits Transolver headers and linkage.
CPU-only fixture deployments use the explicit `cpu-test-only` target and require
`NATIVE_ALLOW_CPU_TEST_ONLY=1`; they are not production GPU qualification.
They emulate a logical GPU scheduler slot while registry metadata preserves
`physical_device_kind=cpu` and `qualification=cpu-test-only`. The existing
`SCHEDULER_DISCOVERY_JSON` fixture override supplies synthetic capacity without
calling NVML. Both the target profile and opt-in flag are required.
Its CPU exception test calls the real shared C++ runner through the dedicated
thread and rejects an invalid block size before CUDA initialization. The default
CPU suite also probes several MiB of live native-thread stack in a subprocess,
so stack exhaustion reports a test failure without terminating the test suite.
GPU acceptance for a serving image built from this source remains pending.
Run H100 validation on the configured remote H100 resources. CPU component
checks do not establish GPU acceptance.
