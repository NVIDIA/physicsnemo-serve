# e2s-example-user

Scaffolded PhysicsNeMo Serve plugin.

Request and result schemas are generated from the input/output models in `workflow.py`.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-example-user
python scripts/plugin_dev.py check-env plugins/e2s-example-user
python scripts/plugin_dev.py run-local plugins/e2s-example-user --dry-run
```

## Authoring

- implement the workflow logic in `workflow.py`
- keep a small happy-path request in `examples/default_request.json`
- `examples/default_request.json` is optional for simple JSON plugins because the dev kit can generate one from `workflow.py`

## Examples

- `examples/default_request.json`
