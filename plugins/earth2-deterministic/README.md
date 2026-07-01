# earth2-deterministic

Deterministic Earth2Studio DLWP plugin for PhysicsNeMo Serve.

## Request

```json
{
  "model": "dlwp",
  "start_time": "2026-01-01T00:00:00Z",
  "nsteps": 4
}
```

CPU vs GPU scheduling is driven by the merged manifest / run envelope `resource_profile` passed into prepare (`PrepareContext.default_resource_profile`), not by a request field.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/earth2-deterministic
python scripts/plugin_dev.py check-env plugins/earth2-deterministic
python scripts/plugin_dev.py run-local plugins/earth2-deterministic --dry-run
```
