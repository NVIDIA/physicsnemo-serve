# e2s-foundry-fcn3-stormscope-goes

PhysicsNeMo Serve plugin for the Earth2Studio `foundry_fcn3_stormscope_goes.py`
example workflow.

The first integration pass follows `e2s-foundry-fcn3` and supports local Zarr
output. Azure container, GeoCatalog, and NetCDF publication parameters are left
out until the core FCN3 plus StormScope GOES path is verified.

## Local checks

```bash
python scripts/plugin_dev.py check plugins/e2s-foundry-fcn3-stormscope-goes
python scripts/plugin_dev.py check-env plugins/e2s-foundry-fcn3-stormscope-goes
python scripts/plugin_dev.py run-local plugins/e2s-foundry-fcn3-stormscope-goes --dry-run
```

## Example

- `examples/default_request.json`
