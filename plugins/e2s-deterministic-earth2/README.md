# e2s-deterministic-earth2

Scaffolded PhysicsNeMo Serve plugin.

Request schema is generated from the input model in `workflow.py`. Non-simple pipeline scaffolds use explicit `prepare()` / `run()` hooks so you can control resources and artifacts.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-deterministic-earth2
python scripts/plugin_dev.py check-env plugins/e2s-deterministic-earth2
python scripts/plugin_dev.py run-local plugins/e2s-deterministic-earth2 --dry-run
```

## Authoring

- implement the workflow logic in `workflow.py`
- keep a small happy-path request in `examples/default_request.json`
- `examples/default_request.json` is optional for simple JSON plugins because the dev kit can generate one from `workflow.py`

## Examples

- `examples/default_request.json`
