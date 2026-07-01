# earth2-ensemble

Direct Earth2Studio ensemble forecast plugin using DLWP.

This plugin calls `earth2studio.run.ensemble()` directly, so it keeps the stock Earth2Studio ensemble semantics.

## Request

```json
{
  "model": "dlwp",
  "start_time": "2026-01-01T00:00:00Z",
  "nsteps": 1,
  "nensemble": 2,
  "batch_size": 2,
  "perturbation": "gaussian",
  "noise_amplitude": 0.05,
  "seed_base": 1000
}
```

## Local checks

```bash
python scripts/plugin_dev.py check plugins/earth2-ensemble
python scripts/plugin_dev.py check-env plugins/earth2-ensemble
python scripts/plugin_dev.py run-local plugins/earth2-ensemble --dry-run
```
