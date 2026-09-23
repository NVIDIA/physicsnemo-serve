# e2s-ensemble

FCN ensemble forecasting through scheduler-owned scatter/gather. `batch_size`
controls the scientific perturbation grouping and therefore prediction results;
`max_in_flight` only limits how many prepared child groups may run concurrently.

Request schema is generated from the input model in `workflow.py`. Non-simple pipeline scaffolds use explicit `prepare()` / `run()` hooks so you can control resources and artifacts.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-ensemble
python scripts/plugin_dev.py check-env plugins/e2s-ensemble
python scripts/plugin_dev.py run-local plugins/e2s-ensemble --dry-run
```

## Authoring

- implement the workflow logic in `workflow.py`
- keep a small happy-path request in `examples/default_request.json`
- `examples/default_request.json` is optional for simple JSON plugins because the dev kit can generate one from `workflow.py`

## Examples

- `examples/default_request.json`
