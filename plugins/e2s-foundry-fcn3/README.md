# e2s-foundry-fcn3

PhysicsNeMo Serve plugin for the Earth2Studio `foundry_fcn3.py` example workflow.

The first integration pass supports local Zarr output only. Azure container,
GeoCatalog, and NetCDF publication parameters are intentionally left out until
the core FCN3 path is verified.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-foundry-fcn3
python scripts/plugin_dev.py check-env plugins/e2s-foundry-fcn3
python scripts/plugin_dev.py run-local plugins/e2s-foundry-fcn3 --dry-run
```

## Example

- `examples/default_request.json`
