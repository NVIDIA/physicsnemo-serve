# Earth2 Deterministic Batching

This note explains how batching works for the
[earth2-deterministic-batch](../plugins/earth2-deterministic-batch)
plugin.

## What Batch Means Here

This plugin uses framework-level batching.

That means:

1. multiple compatible requests are grouped by the `batch` stage
2. the execute worker receives them together
3. the plugin runs one `run_batch(items, ctx)` call for the whole group

It does **not** mean one fused Earth2Studio forecast over multiple user requests in a
single model forward pass.

The current implementation batches:

- model loading
- worker invocation
- request scheduling

But it still writes one `forecast.zarr` per original request.

## Pipeline Flow

For this plugin, `pipeline.profile: batch` expands to:

`prepare -> batch -> schedule -> execute -> results`

Stage ownership:

- `prepare`
  - plugin hook
- `batch`
  - framework stage
- `schedule`
  - framework stage
- `execute`
  - plugin hook
- `results`
  - framework stage

## What `prepare()` Returns

In
[workflow.py](../plugins/earth2-deterministic-batch/workflow.py),
`prepare()` does three things:

1. normalizes request fields
2. sets `resource_profile`
3. sets `batch_profile`

The important part for batching is `batch_profile`.

Current shape:

```python
{
    "enabled": True,
    "batch_key": f"{parameters['model']}:{device_kind}",
    "max_batch_size": 4,
    "max_wait_ms": 200,
    "shared_memory_mb": 4096 if device_kind == "cpu" else 8192,
    "incremental_memory_mb": 512 if device_kind == "cpu" else 1024,
}
```

Meaning:

- `batch_key`
  - defines which requests may batch together
  - here: same `model` and same resolved `device_kind`
- `max_batch_size`
  - hard cap on grouped requests
- `max_wait_ms`
  - flush partial groups after this wait
- `shared_memory_mb`
  - one-time memory cost for the whole batch
- `incremental_memory_mb`
  - extra memory per item in the batch

## What The Framework Does

The framework `batch` stage buffers runs by:

- `workflow_id`
- `batch_key`

When the group is ready, it emits one batch payload containing:

- `batch_id`
- `batch_info`
- `items`

Each `item` still corresponds to one original run.

## What `run_batch()` Does

The plugin implements
[run_batch()](../plugins/earth2-deterministic-batch/workflow.py)
to process the grouped requests together.

Current behavior:

1. receive each item as a typed `BatchItem[DeterministicBatchInput]`
2. select CPU or GPU once for the batch
3. load the DLWP model once
4. construct the `GFS()` data source once
5. loop over items and run deterministic inference for each item
6. write one `forecast.zarr` per run ID
7. return one typed result per item, with per-item failures surfaced through the batch result wrapper

So batching here is:

- shared setup
- shared model load
- one worker invocation

not:

- one multi-request Earth2Studio tensor execution

## What Results Look Like

Each original run still gets its own result payload and its own artifact:

- `forecast_dataset`
- `application/x-zarr`
- one `forecast.zarr` per run

The result also includes `batch_info`, for example:

```json
{
  "batch_id": "6c18dad6-7735-451e-b76e-d4f51added09",
  "batch_size": 2,
  "flush_reason": "max_wait_ms",
  "waited_ms": 252
}
```

So two separate run IDs can:

- share the same `batch_id`
- share the same `batch_size`
- still have different `output_path` values

## What Was Verified

The plugin was checked in two ways.

Direct regression tests in
[test_earth2_plugins.py](../tests/test_earth2_plugins.py):

- single-request batch plugin output matches the non-batch deterministic plugin
- `run_batch()` output matches individual deterministic baselines
- identical requests in one batch produce identical outputs

Live local-stack verification also confirmed:

- two batched requests shared the same `batch_id`
- `batch_size = 2`
- batched outputs matched individual outputs exactly

## Current Limitation

This is not yet a true fused numerical batch inside Earth2Studio.

If later you want:

- multiple requests combined into one model forward pass
- explicit tensor batching inside the ML runtime

then `run_batch()` would need a different implementation.

The current plugin is still useful because it removes repeated setup work while
preserving per-request outputs.
