# Inference Service User Guide

This guide describes the current plugin-based inference service.

For the shortest entry-point view, start with [onboarding.md](./onboarding.md).

## Current Model

The service discovers manifest-driven plugins from `PLUGIN_DIR`.
Set `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID` to a manifest `metadata.id` when a deployment
should expose exactly one plugin from those search roots. Leave it unset or empty
to expose every discovered plugin.

The recommended local authoring flow is:

1. `python scripts/plugin_dev.py init ...`
2. `python scripts/plugin_dev.py check ...`
3. `python scripts/plugin_dev.py check-env ...`
4. `python scripts/plugin_dev.py run-local ...`

## API Flow

Use the same pattern for every plugin:

1. List workflows:

   ```bash
   curl http://HOST:8080/v1/infer/workflows
   ```

2. Inspect schema and readiness:

   ```bash
   curl http://HOST:8080/v1/infer/<workflow_id>/schema
   curl http://HOST:8080/v1/infer/<workflow_id>/readiness
   ```

3. Submit a run:

   ```bash
   curl -X POST http://HOST:8080/v1/infer/<workflow_id>/run ...
   ```

4. Poll status:

   ```bash
   curl http://HOST:8080/v1/infer/<workflow_id>/<run_id>/status
   ```

5. Fetch the structured result envelope or stream artifacts:

   ```bash
   curl http://HOST:8080/v1/infer/<workflow_id>/<run_id>/results
   curl -OJ "http://HOST:8080/v1/infer/<workflow_id>/<run_id>/results?artifact=primary"
   ```

## Request Path

1. The server scans `PLUGIN_DIR` for `plugin.yaml`.
2. If `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID` is set, only the manifest whose `metadata.id`
   matches that value is registered; disabled workflow ids return 404 from the REST API.
3. Compact manifest profiles are expanded into a concrete pipeline and runtime contract.
4. `GET /v1/infer/workflows`, `GET /v1/infer/<workflow_id>/schema`, and `GET /v1/infer/<workflow_id>/readiness` are served from the in-memory plugin registry.
5. `POST /run` validates readiness first.
6. JSON or multipart inputs are parsed and validated.
7. The server builds a run envelope with:
   - `workflow_id`
   - `operation`
   - `parameters`
   - `resource_profile`
   - `stage_context.pipeline`
   - `runtime`
8. The envelope is enqueued to the first pipeline stage.

## Worker Path

1. `prepare`
   - Rust stage
   - invokes Python `prepare(request, ctx) -> PrepareResult` when present
2. `prefetch`
   - Rust stage
   - materializes `prefetch_plan`
3. `batch`
   - Rust stage
   - groups compatible requests by `batch_profile`
4. `fanout`
   - Rust stage
   - expands one parent run into many child runs from `fanout_items`
5. `schedule`
   - Rust stage
   - matches `resource_profile` to worker capabilities
6. `execute`
   - Python worker
   - runs `execute(ctx)` for low-level hooks, or the typed SDK `run(inputs, ctx)` / `run_batch(items, ctx)` paths
   - older plugins may still provide `execute_batch(items, ctx)` directly
   - cacheable workflows may reuse one workflow instance per Python worker process
7. `collect`
   - Rust stage
   - recombines child results for a parent run
8. `postprocess`
   - Rust stage
   - invokes Python `postprocess(ctx)` when present
   - applies built-in `result_ops`
9. `results`
   - Rust terminal persistence stage

## Model Warmup And Cache

Execute-time model caching is process-local to each Python inference worker. It
does not share model objects through Redis, Rust, pods, GPUs, or worker restarts.

When `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID` is set, an execute worker warms that enabled
workflow before it starts polling:

1. The worker registers in `gpu:registry` with `status: "warming"`.
2. The worker resolves the enabled workflow, imports its entrypoint, and creates
   the workflow instance.
3. If the workflow opts in with `cache_scope = "process"` or `cache_models = True`,
   the worker stores that workflow instance in memory using `workflow_id` as the cache key.
4. The worker calls `workflow.warmup(ctx)` when present.
5. On success, the worker updates `gpu:registry` to `status: "available"`.

If `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID` is unset, startup warmup is skipped and the
worker registers as available immediately. A cacheable workflow can still be
cached lazily on its first request.

The `gpu:registry` `model_cache` block is observability metadata only. It reports
entries, model names, timestamps, and hit counts for the worker process. It is
not a scheduler routing contract, and it does not contain model objects.

Because cacheable workflows reuse a single Python object across requests, request
status and request data must remain request-scoped. Run status is tracked by
`run_id` in service state, not on the workflow instance.

## Scheduler Behavior

The scheduler is responsible for:

- routing by `executor_class`
- matching `device_kind`
- respecting `gpus_required`, memory, and tags
- fairness and requeue behavior
- `fanout_profile.max_in_flight` limits for child runs

FIFO is only a queue-ingestion default. The scheduler should avoid letting one parent run or one workload shape monopolize capacity.

## Fanout Status

For fanout parent runs, `GET /v1/infer/<workflow_id>/<run_id>/status` can include:

```json
{
  "status": "running",
  "stage": "executing",
  "fanout_progress": {
    "expected_count": 20,
    "collected_count": 7,
    "remaining_count": 13,
    "succeeded_count": 6,
    "failed_count": 1,
    "cancelled_count": 0,
    "child_run_ids": [
      "parent-run:item:0",
      "parent-run:item:1"
    ]
  }
}
```

## Runtime Environments

Execute workers are started per runtime environment. A plugin does not switch Python interpreters at request time.

The runtime registry is keyed by executor class, for example:

```json
{
  "python_runtime_envs": {
    "python.gpu.biology": {
      "python_executable": "/opt/physicsnemo-serve/envs/python.gpu.biology/bin/python",
      "env": {
        "BIOLOGY_MODEL_CACHE": "/models/biology"
      }
    }
  }
}
```

`prepare`, `postprocess`, and readiness can use separate runtime selectors from execute.

## Result Serving

`GET /v1/infer/<workflow_id>/<run_id>/results` returns a structured result
envelope by default.

Top-level sections:

- `request`
  - normalized request metadata captured for the run
- `execution`
  - platform-owned execution metadata such as `run_id`, `status`, `completed_at`, `execution_time_seconds`, `output_path`, and named `outputs`
- `payload`
  - plugin-defined result payload

When a named output is registered, clients may also request:

- `?artifact=<name>`
  - stream the named artifact directly
- `?artifact=<name>&format=netcdf`
  - generate an on-demand NetCDF export for a dataset artifact
- `?artifact=<name>&format=zarr_zip`
  - generate an on-demand zipped Zarr export for a dataset artifact
- `?artifact=<name>&format=netcdf&vars=var1,var2`
  - restrict an on-demand dataset export to selected variables

Dataset export or publish side effects requested by plugins should still be
expressed through built-in `result_ops`, not custom logic inside `results`.
