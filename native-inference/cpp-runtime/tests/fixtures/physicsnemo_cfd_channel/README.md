# Frozen PhysicsNeMo channel-flow fixture

This directory retains a small PhysicsNeMo model, its reproduction source,
and a frozen `.pnmir` package used by the C++ ONNX Runtime integration test:

```text
physicsnemo-channel-flow-v0.1.0-onnxruntime-cpu-fp32.pnmir/
  model.json
  artifacts/onnx/model.onnx
```

The 2,307-parameter `physicsnemo.models.mlp.FullyConnected` network was trained
against the analytic steady plane-Poiseuille solution. Its static contract is
`coordinates [9,2] -> flow [9,3]`, containing velocity and pressure on a 3x3 grid.
The frozen ONNX graph contains 11,139 bytes. Its original manifest and graph
bytes are preserved; this is a regression fixture, not a released CFD surrogate
or an official pretrained DrivAerML checkpoint.

## Run the frozen package

From the repository root, with an installed native ONNX Runtime distribution:

```bash
cmake -S native-inference/cpp-runtime -B out/inference/runtime-ort \
  -DPNMIR_ENABLE_ONNXRUNTIME=ON \
  -DPNMIR_ONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build out/inference/runtime-ort --parallel
ctest --test-dir out/inference/runtime-ort \
  -R pnmir_physicsnemo_channel_package --output-on-failure
```

Enabling the integration suite requires Python `onnx` during configuration.
The individual package test itself needs only a Python interpreter and the
built native executable:

```bash
python3 -S native-inference/cpp-runtime/tests/physicsnemo_channel_package.py \
  --pnmir out/inference/runtime-ort/physicsnemo-infer \
  --package native-inference/cpp-runtime/tests/fixtures/physicsnemo_cfd_channel/physicsnemo-channel-flow-v0.1.0-onnxruntime-cpu-fp32.pnmir
```

The test preserves its `0.01` maximum absolute error gate against the analytic
solution. Its internal `--pnmir` driver option selects the executable path;
the public executable is `physicsnemo-infer`.

## Reproduce into a separate output directory

Use an environment with compatible PhysicsNeMo, Torch and ONNX dependencies.
Build a Model Builder wheel using the
[scratch packaging instructions](../../../../docs/releasing.md#build-and-check-the-wheel),
then install it without replacing that environment's dependencies:

```bash
python3 -m pip install --no-deps /path/to/physicsnemo_model_builder-0.1.0-py3-none-any.whl
PYTHONPATH=native-inference/cpp-runtime/tests \
  python3 -m fixtures.physicsnemo_cfd_channel.prepare \
  --output /tmp/physicsnemo-channel-reproduced.pnmir
```

Training uses a fixed seed and a generated 21x21 grid. Preparation rejects a
model whose nine-point diagnostic exceeds the same `0.01` error bound, then
exports a static CPU ONNX Runtime package. Regeneration is separate from normal
regression testing; it must not overwrite the frozen fixture by default.

The optional diagnostic workflow composes a local data source, native runtime,
and CSV output using Torch:

```bash
PYTHONPATH=native-inference/cpp-runtime/tests \
  python3 -m fixtures.physicsnemo_cfd_channel.workflow \
  --pnmir out/inference/runtime-ort/physicsnemo-infer \
  --package native-inference/cpp-runtime/tests/fixtures/physicsnemo_cfd_channel/physicsnemo-channel-flow-v0.1.0-onnxruntime-cpu-fp32.pnmir \
  --output /tmp/physicsnemo-channel-flow.csv
```

The frozen package was generated with PhysicsNeMo 2.1.0a0, Torch 2.6.0 and
ONNX 1.20.1. Its ONNX SHA-256 is:

```text
a4b45cdb10be6fa0fd37b78e1dfe7eecfcd8cf49d937038ad3ff40ac9bb7a025
```

Original producer names remain in the frozen manifest and reproduction source
for provenance.
