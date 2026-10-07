# C++ runtime regression fixtures

These fixtures are deliberately kept with the C++ source so it can be copied
and tested independently. Generated test packages belong in CMake build output,
not here.

| Path | Purpose |
| --- | --- |
| `identity/` | Tiny mock package used by C++ API, CLI and installed-consumer tests. |
| `affine.py` | Tiny `2*x+1` Python factory used by AOTI and multi-backend integrations. |
| `physicsnemo_cfd_channel/` | Frozen CPU ONNX package and source for reproducing its small analytic channel-flow regression. |

The ONNX fixture was moved without regenerating it. Its original model identity,
manifest and bytes are retained. See its [provenance and reproduction guide](physicsnemo_cfd_channel/README.md).
