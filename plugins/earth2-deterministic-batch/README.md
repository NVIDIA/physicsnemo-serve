# earth2-deterministic-batch

Batched deterministic Earth2Studio DLWP plugin for PhysicsNeMo Serve.

## Request

```json
{
  "model": "dlwp",
  "start_time": "2026-01-01T00:00:00Z",
  "nsteps": 4
}
```

## Notes

- `run-example` exercises the single-request execution path.
- True batching happens in the framework `batch` stage and calls the workflow's `run_batch()` hook.
- Requests batch together by `model`.
- See [earth2-deterministic-batching.md](../../docs/earth2-deterministic-batching.md) for the full flow.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/earth2-deterministic-batch
python scripts/plugin_dev.py check-env plugins/earth2-deterministic-batch
python scripts/plugin_dev.py run-local plugins/earth2-deterministic-batch --dry-run
```
