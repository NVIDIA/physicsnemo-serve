# earth2-ensemble-fanout

Earth2Studio FCN ensemble forecast plugin that materializes perturbed initial
condition batches on a GPU, fans out one prepared batch per child run, and
aggregates the batch datasets in postprocess.

The prepare hook returns a `materialize_perturbations` operation with a GPU
resource profile. That operation loads FCN, fetches GFS initial conditions on the
selected device, seeds Torch with `seed_base`, applies stock `SphericalGaussian`
perturbations for each `batch_size` group, and writes prepared batch state files.
The framework then fans out one child run per prepared state; each child loads its
assigned state and rolls out the forecast batch. Direct child-side perturbation
replay remains a fallback when no `prepared_state_path` is provided.

By default, perturbation materialization uses `scheduled_gpu`, which routes the
materialization step through the scheduler and runs it on `execute.earth2-gpu`.
Set `perturbation_materialization_mode` to `prepare_cpu` to materialize prepared
states inside the CPU prepare hook and skip directly to fanout. CPU prepare mode
avoids GPU scheduling for perturbation, but it makes prepare a heavier hook and
does not use scheduler resource accounting for that work.

## Request

```json
{
  "model": "fcn",
  "start_time": "2026-01-01T00:00:00Z",
  "nsteps": 1,
  "nensemble": 2,
  "batch_size": 2,
  "max_in_flight": 1,
  "perturbation": "spherical_gaussian",
  "noise_amplitude": 0.05,
  "seed_base": 1000,
  "perturbation_materialization_mode": "scheduled_gpu"
}
```

## Local checks

```bash
python scripts/plugin_dev.py check plugins/earth2-ensemble-fanout
python scripts/plugin_dev.py check-env plugins/earth2-ensemble-fanout
python scripts/plugin_dev.py run-local plugins/earth2-ensemble-fanout --dry-run
```

`run-example` only exercises the plugin hooks directly. Use `run-local` for the
full `materialize_perturbations -> fanout -> collect -> postprocess` path.

## Zarr backend

Fanout child datasets use the shared Earth2Studio Zarr selector. By default, or when `PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=rust`, child runs write through `e2s_zarr_io`. Set `PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python` to use `earth2studio.io.ZarrBackend` for comparison or debugging.

The Rust default requires the runtime image or local environment to include the `e2s_zarr_io` Python extension.
