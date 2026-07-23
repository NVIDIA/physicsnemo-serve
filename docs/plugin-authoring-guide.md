# Plugin Authoring Guide

Use this guide when adding or iterating on a manifest-driven inference plugin.

For the shortest entry-point view, start with [onboarding.md](./onboarding.md).

## Recommended Layout

```text
plugins/my-plugin/
  plugin.yaml
  workflow.py
  workflow_impl.py        # optional
  examples/
    default_request.json  # optional
    expected_result.json  # optional
  README.md               # optional
```

`workflow.py` should be the primary authoring file for new plugins.

## Repo Touch Points

- `plugins/`
  - plugin manifests, workflow code, and request examples
- `scripts/plugin_dev.py`
  - scaffolding, validation, env checks, and local stack commands
- `scripts/plugin_runtime.py`
  - pipeline/runtime profile expansion and SDK wiring
- `python/e2s_tools/`
  - shared Python helpers used by plugins and tests
- `tests/`
  - repo-level plugin and runtime contract checks

## Smallest Manifest

```yaml
metadata:
  id: my-plugin
  display_name: My Plugin
  version: 1.0.0
  description: My first plugin

pipeline:
  profile: simple

runtime:
  profile: python-test
```

## Pipeline Profiles

`pipeline.profile` selects the workload shape:

- `simple`
  - `prepare -> execute -> results`
- `default`
  - alias for `prefetch`
  - `prepare -> prefetch -> schedule -> execute -> results`
- `prefetch`
  - `prepare -> prefetch -> schedule -> execute -> results`
- `postprocess`
  - `prepare -> schedule -> execute -> postprocess -> results`
- `batch`
  - `prepare -> batch -> schedule -> execute -> results`
- `ensemble`
  - `prepare -> fanout -> schedule -> execute -> collect -> results`

`pipeline.options` only enables a small set of extra stages:

- `postprocess: true`
  - append `postprocess` before `results`
- `prefetch: parent`
  - only for `ensemble`
  - insert `prefetch` before `fanout`

Example:

```yaml
pipeline:
  profile: ensemble
  options:
    prefetch: parent
    postprocess: true
```

That expands to:

`prepare -> prefetch -> fanout -> schedule -> execute -> collect -> postprocess -> results`

## Runtime Profiles

`runtime.profile` selects a runtime family. The default is:

- `python-test`

Custom runtime families can be expressed with explicit executor classes:

```yaml
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.gpu.biology
  prepare_executor_class: python.cpu.biology
  postprocess_executor_class: python.cpu.biology
  readiness_executor_class: python.cpu.biology
```

If a runtime family becomes reusable, it can later be promoted into a compact `runtime.profile`.

## Input And Output Models

For normal JSON plugins, define Python models in `workflow.py` and let PhysicsNeMo Serve derive schemas automatically.

```python
from dataclasses import dataclass

from plugin_sdk import PluginWorkflow


@dataclass
class DemoInput:
    value: int


@dataclass
class DemoOutput:
    value: int
    doubled: int


class DemoWorkflow(PluginWorkflow):
    input_model = DemoInput
    output_model = DemoOutput

    def run(self, inputs: DemoInput, ctx):
        return DemoOutput(value=inputs.value, doubled=inputs.value * 2)
```

Use explicit manifest schemas when:

- the plugin is multipart
- the plugin is not model-driven
- the schema must differ from the Python model contract

## User Hooks

Plugins provide Python hooks. The framework decides where they run.

- `prepare(request, ctx) -> PrepareResult`
  - preferred prepare hook for new plugins
  - read raw ingress from `request.raw_fields` and `request.input_artifacts`
  - normalize execution inputs into `PrepareResult.inputs`
  - optionally return `resource_profile`, `prefetch_plan`, `batch_profile`, `fanout_profile`, and `fanout_items`
  - use `ctx.run_id`, `ctx.run_dir`, and `ctx.default_resource_profile` for prepare-time context
- `execute(ctx)`
  - low-level compatibility hook used by older plugins
- `run(inputs, ctx)`
  - preferred typed SDK entrypoint for normal single-item execution
  - return only the payload matching `output_model`
  - register named file or dataset outputs through `ctx.outputs.create(...)` or `ctx.outputs.register(...)` when the workflow produces artifacts
- `run_batch(items, ctx)`
  - preferred typed batch hook for shared setup across compatible items
  - batch-capable workflows may implement this without also defining `run(inputs, ctx)`
  - single-item execution can still route through this hook with a one-item batch
- `execute_batch(items, ctx)`
  - legacy batch hook retained for compatibility; new SDK plugins should prefer `run_batch(items, ctx)`
- `postprocess(result, ctx) -> PostprocessOutcome`
  - optional final shaping, aggregation, or publication request stage
  - return `PostprocessOutcome(payload=..., status=..., result_ops=[...])` when overriding final status or requesting built-in side effects

## Cacheable Workflow Lifecycle

By default, class-based workflows should assume a fresh instance per request.
Only opt in to process-local reuse when the workflow can safely share one Python
object across requests.

There are two caches involved:

- shared disk cache
  - model packages, checkpoints, and weights downloaded by Earth2Studio or a
    model hub
  - shared by multiple worker processes when they use the same filesystem
  - populated under an interprocess lock
- worker process cache
  - Python workflow object and loaded model objects held in one
    `inference_worker.py` process
  - not shared across workers, GPUs, pods, or restarts
  - reused only by later requests handled by that same worker

Use these hooks consistently:

- `prepare_model_cache(ctx)`
  - module-level function, not a workflow method
  - runs first during startup warmup, under a file lock keyed by `workflow_id`
  - should force shared package/checkpoint downloads to complete safely
  - may call model loader APIs if those APIs are what populate weight files
  - should not retain the returned model object for serving
- `warmup(ctx)`
  - workflow method
  - runs after `prepare_model_cache()`
  - should create the workflow-scoped model object and move it to the worker device
  - stores reusable objects on `self`
- `cleanup_request()`
  - workflow method
  - runs after each request on cached workflows
  - clears request-scoped state while preserving model/package fields
- `cleanup()`
  - workflow method
  - final cleanup when the worker retires the cached workflow, usually shutdown
  - releases model/package/data objects

```python
def prepare_model_cache(ctx):
    package = Model.load_default_package()
    Model.load_model(package)  # force weight files into the shared disk cache
    return {"model_names": ["my-model"]}


class MyWorkflow(PluginWorkflow):
    cache_scope = "process"
    model_cache_names = ["my-model"]

    def warmup(self, ctx):
        package = Model.load_default_package()
        self.model = Model.load_model(package).to(ctx.get("device", "cuda"))
        self.model.eval()
        return {"model_names": ["my-model"]}

    def cleanup_request(self):
        self.request_temp = None

    def cleanup(self):
        self.model = None
```

Lifecycle rules:

- If a workflow declares `model_cache_names`, it should provide both
  `prepare_model_cache(ctx)` and `warmup(ctx)`.
- `warmup(ctx)` runs at execute-worker startup when `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID`
  selects the workflow, or the workflow can be cached lazily on first request.
- `cleanup_request()` runs after each request for cached workflows. Use it to
  release request-scoped state while preserving reusable model state.
- `cleanup()` is final cleanup for cached workflows. It runs when the worker
  retires the cached workflow, usually during worker shutdown.
- The cache key is `workflow_id`, so a deployment should not depend on request
  parameters to select different cached model objects for the same workflow.

Keep on the workflow instance:

- loaded model objects
- package handles required by the model object
- fixed packages or handles that are safe to reuse for all requests handled by
  that workflow id

Do not keep request-scoped state on the shared workflow instance:

- run status, `run_id`, parent run ids, or errors
- request inputs or user payloads
- output directories, staged output paths, temporary datasets, or data sources
  unless `cleanup_request()` clears them

Framework-owned stages:

- `prefetch`
- `batch`
- `fanout`
- `schedule`
- `collect`
- `results`

## Prepare Output Contract

`prepare()` returns `PrepareResult`.

The runtime maps `PrepareResult.inputs` into the internal run envelope and
passes the remaining fields to later framework stages.

Common fields:

- `inputs`
  - normalized execution inputs that later feed `run(inputs, ctx)` or `run_batch(items, ctx)`
- `resource_profile`
  - scheduler-facing execution requirements
- `prefetch_plan`
  - consumed by `prefetch`
- `batch_profile`
  - consumed by `schedule` as a hint for grouping compatible requests
- `fanout_profile`
  - consumed by `fanout`, `schedule`, and `collect`
- `fanout_items`
  - child inputs consumed by `fanout`

If `prepare()` creates temporary files, write them under `ctx.run_dir` and pass
their paths through `inputs` or `fanout_items`.

The scheduler considers every non-fanout request for batching. `batch_profile`
overrides the scheduler defaults for compatible grouping, maximum size, maximum
wait, and memory scaling. It does not force authors to implement a special hook:
plugins may keep using `run(inputs, ctx)` and let the default adapter execute
items one by one inside the batch, or implement `run_batch(items, ctx)` when
shared setup is valuable. When a workflow only implements `run_batch(items, ctx)`,
normal single-item execution still routes through that hook with a one-item batch.

## Output Registration

Use `ctx.outputs` whenever a hook creates a file or dataset that should show up
as a named artifact in the final result envelope.

```python
dataset_path = ctx.outputs.create(
    "forecast_dataset",
    filename="forecast.zarr",
    media_type="application/x-zarr",
    primary=True,
)
```

Notes:

- `create()` allocates a run-scoped path under `ctx.run_dir` and registers it immediately
- `register()` is useful when the plugin writes a file first and then attaches it as an output
- simple JSON-only plugins can just return payload data; explicit output registration is mainly for additional files and datasets

## Fanout And Collect

Use `fanout/collect` when one logical request expands into many independent child runs and later recombines.

Good fits:

- ensemble forecasting
- Monte Carlo
- parameter sweeps
- per-tile inference
- per-region processing

This contract is generic:

```json
{
  "fanout_profile": {
    "item_count": 20,
    "max_in_flight": 4,
    "failure_policy": "collect_all"
  },
  "fanout_items": [
    {
      "item_index": 0,
      "parameters": {"seed": 1000}
    }
  ]
}
```

## Scheduling

Pipelines that include `schedule` need a usable `resource_profile` by the time they reach that stage.

That can come from:

- manifest defaults
- or a dynamic override from `prepare()`

Typical fields:

- `executor_class`
- `device_kind`
- `gpus_required`
- `memory_mb`
- `tags`

## Result Ops

`results` is framework-owned. `run()` should return payload only.

`postprocess()` is the place to request built-in side effects by returning
`PostprocessOutcome(..., result_ops=[...])`.

Typical op families:

- `object_store_publish`
- `dataset_export_netcdf`

## New Runtime Family Example

For a new family such as `biology-demo`:

1. Build one or more Python envs, for example:
   - `python.cpu.biology`
   - `python.gpu.biology`
2. Register them in `python_runtime_envs`
3. Point the plugin at explicit executor classes first
4. Add readiness checks for required modules, env vars, and paths
5. Promote the family into a reusable `runtime.profile` only after the first plugin is proven

## Dev Kit Flow

```bash
python scripts/plugin_dev.py init plugins/my-plugin
python scripts/plugin_dev.py validate plugins/my-plugin
python scripts/plugin_dev.py check plugins/my-plugin
python scripts/plugin_dev.py check-env plugins/my-plugin
python scripts/plugin_dev.py run-local plugins/my-plugin --dry-run
```
