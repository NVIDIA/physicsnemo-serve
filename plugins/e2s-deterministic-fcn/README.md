# e2s-deterministic-fcn

Scaffolded PhysicsNeMo Serve plugin.

Request schema is generated from the input model in `workflow.py`. Non-simple pipeline scaffolds use explicit `prepare()` / `run()` hooks so you can control resources and artifacts.

## Model Cache

This workflow opts into process-local model caching with `cache_scope = "process"`.
Each Python execute worker loads the FCN package/model once, keeps it on the
workflow instance across requests, and releases it during final worker cleanup.

When `PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=e2s-deterministic-fcn`, execute workers warm
this workflow before polling for requests. During warmup the worker is registered
as `warming`; after the FCN model is loaded, it becomes `available`.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-deterministic-fcn
python scripts/plugin_dev.py check-env plugins/e2s-deterministic-fcn
python scripts/plugin_dev.py run-local plugins/e2s-deterministic-fcn --dry-run
```

## Authoring

- implement the workflow logic in `workflow.py`
- keep a small happy-path request in `examples/default_request.json`
- `examples/default_request.json` is optional for simple JSON plugins because the dev kit can generate one from `workflow.py`

## Examples

- `examples/default_request.json`
