# Scheduler-Owned Fanout

## Status

Draft for design review. Agreed direction: Option A (expand on arrival) with
sibling-only scheduler batching.

This ships as a new deployment. There is no backward compatibility or migration
requirement for existing manifests, streams, or in-flight runs.

## Decision Summary

Remove the dedicated `fanout` stage and worker role. The scheduler takes over
scatter: any request that reaches a `schedule` stage with non-empty
`fanout_items` is expanded into child runs by the scheduler. Gather stays in the
existing `collect` role.

With this change every plugin gets scatter/gather natively. A plugin only returns
`fanout_items` (and optionally `fanout_profile`) from `prepare` or from an
execute stage's `_pipeline_updates`. Manifests no longer declare `fanout` or
`collect` stages; gather is always wired in by the scheduler.

Fanout children reuse the scheduler's existing request batching, limited to
siblings of the same parent.

## Goals

- Scatter/gather available to all plugins without manifest stages.
- Keep the change small: move existing code rather than redesign it.
- Keep today's guarantees: atomic child creation, idempotent collection,
  single finalizer, `max_in_flight` per parent, dropping children of terminal
  parents, `fanout_progress` status.
- Let fanout children use scheduler batching.

## Non-Goals

- Batching children from different parents into one dispatch.
- Batching or dispatching the fan-out parent itself (parents are always expanded).
- Multiple fanout rounds per run or nested fanout (see
  [Current Constraints](#current-constraints-kept)).
- Moving `collect` into the scheduler.
- Changing Earth2 ensemble RNG semantics.

## Current Design (Summary)

```
prepare/execute ──(fanout_items)──▶ fanout role ──forward_many──▶ schedule ──▶ execute ──▶ collect ──▶ postprocess/publish/results
```

- `crates/worker-runtime/src/roles/fanout.rs` validates the envelope, calls
  `CollectStore::init_group`, builds one child per item
  (`run_id = "{parent}:item:{i}"`, `parent_run_id`, `fanout_item`, item overrides for
  `operation` / `parameters` / `resource_profile`), and emits all children plus the
  parent ack atomically via `forward_many` (`LUA_FORWARD_MANY`).
- The scheduler enforces `fanout_profile.max_in_flight` with Redis parent slots
  (`scheduler/parent_slots.rs`), drops children of terminal parents, and
  excludes all fanout traffic from batching (`scheduler_batch_excluded` in
  `scheduler/batch.rs`).
- `collect.rs` appends child results idempotently, persists `fanout_*` progress,
  claims and commits parent finalization, and marks the parent terminal.
- Pipelines must declare `fanout` and `collect` stages (`ensemble` profile or
  explicit stages, as in `plugins/earth2-ensemble-fanout/plugin.yaml`).

## Alternatives Considered

| | A: expand on arrival | B: lazy expansion in queue | C: scatter + gather in scheduler |
|---|---|---|---|
| Roles removed | fanout | fanout | fanout, collect |
| New durable state | none | per-parent dispatch cursor | none |
| Crash safety | same as today | needs new work | same as today |
| Change size | small | medium to large | medium |
| Child batching | yes (via existing batching) | yes | yes |

- **B** keeps the parent as one queued entry and dispatches items directly to
  GPU streams. It avoids N intermediate messages but needs a durable per-parent
  cursor and long-lived pending parent messages. Rejected for complexity.
- **C** also folds `collect` into the scheduler. It puts gather work on the
  scheduling loop and couples result collection to scheduler health. Can be
  revisited later on top of A.

**Chosen: A.**

## Design

### 1. Trigger

A schedule message is a **fan-out parent** when its payload has a non-empty
`fanout_items` array. Plugins produce this exactly as today:

- `prepare` returns `fanout_items`, `fanout_profile`, and routes to a schedule
  stage (default next stage or `next_stage_id`).
- An execute stage returns `_pipeline_updates` with `fanout_items` /
  `fanout_profile`, and its next stage is a schedule stage.

No plugin SDK change.

**Validation.** Returning `fanout_items` when the next stage cannot expand them
fails the request with a clear error instead of silently dropping the items:

- `prepare` role (`prepare.rs`): if the hook returns non-empty `fanout_items`,
  the target stage must be `schedule`, or `prefetch` whose `next` is `schedule`.
- Execute handoff (`inference_worker.py`): if a successful result carries
  `_pipeline_updates.fanout_items`, the next stage must be `schedule`.

Error text names the stage that was found and states that fanout requires a
`schedule` stage (for example, the `simple` profile cannot fan out).

### Plugin author contract

A plugin opts in per request; nothing is declared in the manifest.

1. **Pipeline:** any profile or stage list with a `schedule` stage after the
   point where items are returned (`batch`, `postprocess`, `prefetch`,
   `ensemble`, or explicit stages). Add `postprocess` to aggregate child outputs.
2. **Produce items**, either:
   - from `prepare`: `PrepareResult(inputs=..., fanout_profile={...}, fanout_items=[...])`, or
   - from an execute stage whose `next` is `schedule`:
     `{"status": "succeeded", "_pipeline_updates": {"fanout_profile": {...}, "fanout_items": [...]}}`.
3. **Item shape:** `item_index` plus optional per-child overrides for
   `operation`, `parameters`, and `resource_profile`.
   `fanout_profile` fields: `item_count` (defaults to `len(fanout_items)`),
   `max_in_flight`, and `failure_policy` (`collect_all` default, or `fail_fast`).
4. **Execute:** each child runs the normal `execute` / `run` with its item's
   parameters; the raw item is available as `ExecutionContext.fanout_item`.
5. **Aggregate (optional):** `postprocess` receives the parent result with
   `child_results[]` (`item_index`, `child_run_id`, `fanout_item`, `result`,
   sorted by `item_index`) and `aggregation_summary`.
6. **Knobs:** `batch_profile.enabled = false` for one child per dispatch; parent
   status exposes `fanout_progress`.

### 2. Scatter in the scheduler

`SchedulerRole::handle` on the schedule stream:

```
decode payload
if payload has fanout_items:
    expand_fanout_parent(msg, payload, sink)     # new, ported from fanout.rs
    return Ok(())                                # parent acked by forward_many
else:
    enqueue_request(...)                         # unchanged
    return Err(scheduler_deferred_error())
```

`expand_fanout_parent` (new module `scheduler/fanout.rs`, code moved from
`roles/fanout.rs`):

1. Validate `fanout_items` non-empty and `fanout_profile.item_count` (if set)
   equal to `len(fanout_items)`.
2. Resolve the child's execute stage: the `next` of the current schedule stage.
3. Resolve the gather stage (see below) and build the collect-side parent payload
   (drop `fanout_items` and `result`, set `stage_context` to the gather stage).
4. `collect_store.init_group(parent_run_id, CollectGroup { ... })`.
5. Build child payloads exactly as `build_child_output` does today, except:
   - the child's `stage_context` stays at the **current schedule stage**, and
   - the output stream is the scheduler's own schedule stream.
6. `sink.forward_many(msg, &children)`. On error the group is kept, not discarded:
   the script may have committed before the client saw the error, and a retry
   reuses the group because `init_group` never overwrites an existing one.

Children then re-enter the scheduler as normal requests and pick up
`max_in_flight`, terminal-parent drops, reservations, and batching.

The scheduler builds a `RedisCollectStore` from its existing `QueueManager`
(the scheduler already requires one).

### 3. Gather routing (hidden `_gather` stage)

Children must reach `collect` after execute without a manifest stage. When
expanding, the scheduler always splices a hidden stage into each child's
`stage_context.pipeline` (and the stored parent payload):

```
{ "id": "_gather", "phase": "collect", "queue": "collect", "next": <execute.next> }
execute.next = "_gather"
```

`_gather` is a reserved stage id; a fanout parent whose pipeline already uses it
is rejected. A non-array `fanout_items` value is also rejected rather than
scheduled as a normal request.

`collect` finalization then works unchanged: the stored parent payload points at
`_gather`, and `next_stage("collect")` continues the parent pipeline at the
execute stage's original `next`. Existing collect restrictions still apply: that
stage must be `postprocess`, `publish`, or `results`; the scheduler rejects the
parent otherwise.

### 4. Sibling-only batching

Changes in `scheduler/batch.rs`:

- Delete `scheduler_batch_excluded` and its only helper
  `pipeline_contains_phase`, plus the now-unused `fanout_gate` import in
  `batch.rs`. Every check in it is fanout-specific, and fan-out parents never
  reach the queue, so nothing needs excluding.
- `batch_key`: append `parent_run_id` when present
  (`workflow::batch_key::executor_class::parent_run_id`). Only siblings batch
  together; children never batch with plain requests or with other parents.
- `build_request_batch_payload`: copy `parent_run_id` and `fanout_profile` from
  the head so `fanout_gate`, the terminal-parent drop, and dispatch see them.

Parent slots: **one slot per dispatch**, whether single or batch. The execute
worker already sends one release per dispatch carrying `parent_run_id` when all
batch items share a parent (`inference_worker.py` batch path), so
`parent_slots` is unchanged. `max_in_flight` is documented as "concurrent GPU
dispatches per parent". A batch of size `k` with `max_in_flight = m` can run
up to `k * m` children concurrently.

The worker side already supports this: per-item `parent_run_id`, per-item
terminal-parent skips, and per-item handoff to the next stage under the item's
`run_id` (`_build_batch_primary_outputs`). All siblings share one pipeline, so
using the batch head's `stage_context` for items is correct.

Whether a workflow batches at all is still governed by the existing batching
rules (`batching_enabled`, `batch_profile`, single-GPU resource profile).

### 5. Child failures reaching collect

Gap (exists today, more visible with batching): if a child dispatch, single or
batch, exhausts scheduler retries and moves to the DLQ, `mark_request_failed`
marks the child run failed but nothing reaches `collect`, so the parent never
finalizes.

Fix: when the scheduler moves a fanout child (identified by the `_gather` stage in
its pipeline) to the DLQ, it forwards the source message to both the DLQ and
`collect` (with `result = { "status": "failed", "error": ... }`) in one atomic
`forward_many`. Either both records exist or neither does and the child is retried. Scheduler retries and DLQ work per
queued message, so a failed batch attempt leaves each sibling to retry and DLQ on
its own; no batch-level handling is needed. Collect dedupes by child run id, so a
repeated report is safe, and the normal `collect_all` / `fail_fast` logic
finalizes the parent.

### 6. Execute worker handoff

`scripts/inference_worker.py` treats `execute → fanout` with `_pipeline_updates` as
an internal handoff (`_should_persist_run_status_after_execute`). Change the
condition to "next phase is `schedule` and `_pipeline_updates.fanout_items` is
present" so the parent isn't marked finished after a materialize-style execute
stage.

### 7. Removals

- `crates/worker-runtime/src/roles/fanout.rs` and its registration in
  `roles/mod.rs`; `"fanout"` in `role_requires_queue_manager` (`main.rs`).
- `fanout` and `collect` phase/handlers in `plugin_registry.rs`
  (`stage_definition`, `is_supported_stage_handler`, `ensemble` profile) and the
  Python mirror in `scripts/plugin_runtime.py`. Manifests can no longer declare
  either stage; the existing unsupported phase/handler error covers them. The
  `collect` worker role stays and is fed only by the hidden `_gather` stage.
- `ensemble` profile becomes
  `prepare [-> prefetch] -> schedule -> execute [-> postprocess] -> results`, with
  gather spliced at runtime.
- `fanout` stream and role in `scripts/worker_runtime_config.json`,
  `crates/worker-runtime/examples/runtime_config.json`, and
  `scripts/entrypoint.sh`.
- Local dev simulation of the fanout stage in `scripts/plugin_dev.py`.

### 8. Plugin update: `earth2-ensemble-fanout`

`plugin.yaml` changes `materialize_perturbations.next` from `fanout` to
`schedule`, removes the `fanout` and `collect` stages, and sets
`execute.next = postprocess`. No Python changes beyond what already emits
`_pipeline_updates`. The plugin stays as the end-to-end fanout reference and
the basis for the fanout QA and CRPS checks.

Scheduler batching only groups items that are already perturbed. Per-member noise
is still determined by the plugin's own `batch_size` and `seed_base`.

## Current Constraints Kept

- One fanout round per run. Collect can only hand off to
  `postprocess` / `publish` / `results`, and finalization marks the parent terminal.
- Children always go through the scheduler.
- `fail_fast` / `collect_all` semantics, `fanout_progress` status fields, and the
  child run id format `{parent}:item:{i}` are unchanged.

## Risks

- **Scheduler handles more work per message.** Expansion is one Redis
  round trip plus one `forward_many`, the same cost the fanout role pays today.
  Very large `fanout_items` arrays grow a single Lua call; same as today.
- **`max_in_flight` counts dispatches, not children**, once batching applies.
  Documented. Plugins that need a strict child cap can set
  `batch_profile.enabled = false`.

## Implementation Plan

This is a new deployment, so there is no migration or compatibility work. Tasks
are still ordered so each can merge on its own with tests passing: the new
scheduler path lands first, then the old stage is deleted.

### Task 1: Scheduler expands fan-out parents

- Add `scheduler/fanout.rs` with `expand_fanout_parent`, ported from
  `roles/fanout.rs` (`decode_fanout_payload`, `build_parent_payload_for_collect`,
  `build_child_output`).
- Build a `RedisCollectStore` in `SchedulerRole::from_env`; in-memory store for
  tests.
- Branch in `SchedulerRole::handle` on non-empty `fanout_items`.
- Children keep `stage_context` at the current schedule stage and target the
  schedule stream.
- `prepare.rs`: reject non-empty `fanout_items` unless the target stage is
  `schedule` or `prefetch → schedule`. Confirm the prefetch role forwards
  `fanout_items` / `fanout_profile` unchanged.
- **Tests:** port `fanout_role_expands_parent_request_into_child_schedule_messages`;
  add expansion rollback (`discard_group` on `forward_many` failure);
  `item_count` mismatch rejected; non-fanout requests still queue normally;
  prepare rejects `fanout_items` with a `simple` pipeline and accepts
  `prefetch → schedule`.

### Task 2: Hidden `_gather` stage

- Resolve execute stage from the current schedule stage and splice `_gather`
  after it.
- Apply the rewritten pipeline to every child and to the stored parent payload.
- Reject parents whose post-execute stage is not `postprocess` / `publish` /
  `results`.
- **Tests:** splice produces the expected pipeline; invalid post-execute stage
  rejected; end-to-end in-memory flow `schedule → execute (fake) → collect →
  results` finalizes the parent.

### Task 3: Execute worker handoff for `execute → schedule`

- Update `_should_persist_run_status_after_execute` (and any sibling checks for
  `next_phase == "fanout"`) in `scripts/inference_worker.py` to treat
  "next phase `schedule` + `_pipeline_updates.fanout_items`" as internal.
- Fail the run with a clear error when a successful result carries
  `_pipeline_updates.fanout_items` but the next stage is not `schedule`.
- **Tests:** Python unit tests for status persistence, for
  `_merge_pipeline_updates` landing `fanout_items` on the schedule handoff, and
  for the error when the next stage is not `schedule`.

### Task 4: Sibling-only batching for children

- Delete `scheduler_batch_excluded`, `pipeline_contains_phase`, and the
  `fanout_gate` import in `batch.rs`.
- `batch_key`: append `parent_run_id`.
- `build_request_batch_payload`: carry `parent_run_id` and `fanout_profile`.
- Update the `batching_enabled` doc comment in `config.rs`.
- **Tests:** siblings batch together; children of different parents don't;
  children don't batch with plain requests; a batch acquires exactly one parent
  slot and one release frees it; batch dropped when parent is terminal. Replace
  `scheduler_bypasses_batching_for_fanout_requests`.

### Task 5: Child failures on scheduler DLQ reach collect

- In the scheduler DLQ branch, when the DLQ'd request is a fanout child, enqueue a
  failed child envelope to `collect`.
- **Tests:** a DLQ'd child is reported to `collect` with a failed result; plain
  requests are not. Collect's existing tests cover `collect_all`, `fail_fast`, and
  dedupe.

### Task 6: Update `earth2-ensemble-fanout` and local dev tooling

- Update `plugins/earth2-ensemble-fanout/plugin.yaml` (drop `fanout` and
  `collect` stages, `materialize_perturbations.next = schedule`,
  `execute.next = postprocess`).
- Update `scripts/plugin_dev.py` / `scripts/plugin_direct_runner.py` local
  simulation to expand `fanout_items` at the schedule step.
- **Tests:** `tests/test_earth2_plugins.py`, `tests/test_plugin_dev.py`,
  `tests/test_plugin_direct_runner.py`; manifest loads with the new registry.

### Task 7: Remove the fanout stage and role

- Delete `roles/fanout.rs`; remove from `roles/mod.rs` and `main.rs`.
- Remove `fanout` and `collect` phase/handlers and the `ensemble` profile
  entries in `plugin_registry.rs` and `scripts/plugin_runtime.py`.
- Remove `fanout` stream/role from `scripts/worker_runtime_config.json`,
  `crates/worker-runtime/examples/runtime_config.json`, and `scripts/entrypoint.sh`
  (supervisor program, wrapper script, `WORKERS` value list and examples).
- Delete code paths left unused by the removal:
  - `"fanout"` branches in `scripts/plugin_dev.py` role/stage wiring (anything
    not already replaced in task 6).
  - Any remaining `next_phase == "fanout"` checks in `scripts/inference_worker.py`.
  - `_should_handoff_to_collect` in `scripts/inference_worker.py` (already has no
    callers today).
  - `fanout` test fixtures: pipelines containing a `fanout` stage in
    `roles/collect.rs`, `roles/prepare.rs`, `roles/scheduler/mod.rs` tests, and
    `plugin_registry/tests.rs`. Rewrite them to the schedule-expanded shape.
- Keep: `fanout_profile` / `fanout_items` / `fanout_item` payload fields, the
  `fanout_*` progress hash fields and `fanout_progress` status, `forward_many`
  in `scicomp-rq` (used by the scheduler), and the `-fanout` workflow-name
  profile fallback in `scheduler/profile.rs` (names, not stages).
- Verify with `git grep -niE 'fan.?out'` that every remaining hit is on the
  keep list or in historical benchmark reports under `docs/`.
- **Tests:** registry tests for the new `ensemble` expansion and rejection of
  `fanout` / `collect` stages; `example_config_smoke.rs`; `multi_worker_pipeline.rs`.

### Task 8: Docs and end-to-end validation

- Update `docs/plugin-authoring-guide.md`: profiles without `fanout`/`collect`,
  replace the `fanout/collect` section with the
  [plugin author contract](#plugin-author-contract) (prepare and execute
  examples, item shape, `ctx.fanout_item`, `child_results` in postprocess,
  the `schedule` requirement), and the batching note; `docs/inference-service-user-guide.md` (worker path, `max_in_flight`
  meaning), `README.md` role list, and the fanout worker row in
  `observability/readme_observability.md`.
- Run `qa/inference/test_cicd.py::test_earth2_ensemble_fanout` and the Lepton
  CRPS comparison with the same `seed_base` / `batch_size` as the baseline.
  Confirm identical per-member output and a finalized `fanout_progress`.

### Dependency Order

```
1 ─▶ 2 ─▶ 3 ─▶ 6 ─▶ 7 ─▶ 8
      └─▶ 4
      └─▶ 5
```

Tasks 4 and 5 depend only on task 2 and can proceed in parallel with 3 and 6.
