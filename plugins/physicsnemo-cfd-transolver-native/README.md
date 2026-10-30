# Native CFD Transolver

This workflow serves the pinned DrivAerML `run_1` surface case through the
persistent [native worker](../../crates/native-worker/README.md). A deployment
must provide the compiled full and tail packages, verified normalization
statistics, and prepared inputs described by its deployment configuration.

Validate the public contract and inspect the configured environment:

```sh
python scripts/plugin_dev.py validate plugins/physicsnemo-cfd-transolver-native
python scripts/plugin_dev.py check-env plugins/physicsnemo-cfd-transolver-native
python scripts/plugin_dev.py check plugins/physicsnemo-cfd-transolver-native
```

These checks report contract and environment readiness. Submit inference through
the running HTTP service. Set `NATIVE_DEPLOYMENT_CONFIG` to the installed
deployment JSON, configure the native worker binary with `NATIVE_WORKER_BIN`,
and supply the immutable execution identity documented in the worker README.
The worker verifies model hashes, target compatibility, and startup self-tests
before readiness succeeds. `/readyz` checks the live worker ownership lease.

The pipeline is `schedule → native execute → results`, with one active request
per selected GPU and scheduler batching disabled. The deployment uses the
[scheduler/results configuration](../../native-inference/deploy/worker_runtime_config.json).

Submit the [default request](examples/default_request.json) to
`POST /v1/infer/physicsnemo-cfd-transolver-native/run`:

```json
{"parameters":{"case_id":"run_1","seed":42,"capture_inputs":false}}
```

Retrieve the run status, results, and artifacts through the existing workflow
API. Outputs include physical fields, standardized fields, and native metadata.
Physical fields are little-endian FP32 `[N,4]` in source-cell order: pressure,
then WSS x/y/z. `capture_inputs: true` also registers prepared tensors and the
permutation, adding substantial artifact I/O. Scientific report evaluation uses
the external reference evaluator.
