# Affine model recipe

The bundled `affine` recipe computes `output = 2 * input + 1` to exercise graph
export, compilation, packaging and C++ inference. It is a deterministic pipeline
fixture, not a CFD surrogate or a scientific qualification suite.

The model is version `0.1.0`. Its default backend is `aoti`; `tensorrt` is also
supported with a compatible GPU builder. See [builder setup](../../README.md)
for the image and native runtime prerequisites.

From the repository root, with a compatible local producer environment and
installed native runtime:

```bash
./native-inference/physicsnemo-model-builder build affine \
  --backend aoti --backend tensorrt \
  --executor local --device cuda \
  --runtime /path/to/sdk/bin/physicsnemo-infer \
  --output /data/builds/affine-first
```

Choose a fresh output directory. TensorRT requires CUDA; AOTI can also target
CPU when the supplied runtime and producer support it. Container execution is
the default when `--executor` is omitted; select an immutable builder image.
To store these choices in a file, use a [model project](../../../docs/model-build-projects.md)
with `"model": "affine"`.

## Recipe contract

[`recipe.json`](recipe.json) uses format 1 with one input named `input` and one
output named `output`. Its input is statically shaped `[4]` with dtype float32.
[`export.py`](export.py) provides two zero-argument callbacks:

- `create_model()` returns a `torch.nn.Module`; the worker sets evaluation mode
  and moves it to the requested device.
- `create_cases()` returns three tuples of input tensors, ordered by the
  recipe's `input_names`.

| Case | Input |
| --- | --- |
| 0 | `[0.0, 1.0, -1.0, 4.0]` |
| 1 | `[2.5, -8.0, 0.125, 13.0]` |
| 2 | `[-0.5, 100.0, -32.0, 0.0001]` |

This model has no learned parameters or checkpoint; its constants live in the
adapter, so an empty model-state digest is expected. It uses the legacy
shared-shape tensor shorthand. Custom recipes can declare independent input and
output shapes through the [explicit tensor contract](../../../docs/model-inputs.md#tensors-with-different-shapes).
Model outputs must match the recipe's ordered output names.

## Verification and packages

Each selected backend produces a directly loadable package, for example
`model/backends/aoti/` or `model/backends/tensorrt/`.

For each selected backend, the worker compiles using the first case and runs
all three cases through the C++ runtime. It validates actual tensor metadata,
completion, device/backend and finite numerical outputs against independent
Python eager references. A requested backend failure fails the whole build and
preserves available diagnostics. See the [package and evidence contract](../../../docs/packages.md)
for numerical gates, output locations, compiled/graph distinctions and deployment
compatibility.

## Author a model with weights

For a trained model, use format 2 and declare its configuration, plain state
dictionary and optional data assets. The builder captures those declared files
and loads weights strictly; the [configured-affine example](../../../examples/README.md#configured-affine)
demonstrates two checkpoints with analytical reference outputs.

The declared Python adapter remains a single retained file. Imports from
libraries installed in the producer environment are supported; arbitrary
sibling Python helper modules or undeclared assets are not captured automatically.
The adapter is not imported as a package. Follow the
[custom input contract](../../../docs/model-inputs.md) for callbacks, file
selection, checkpoint formats and replay.
