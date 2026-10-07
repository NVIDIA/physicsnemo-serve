# PhysicsNeMo Model Builder

Model Builder prepares a Python model, exports and compiles its selected
backends, and verifies the resulting packages with native inference.

For model users, start with the [user guide](../docs/user-guide.md).
Use the [technical reference](../docs/reference.md) for project settings,
compiler profiles, explicit recipes and output formats. Setup instructions for
[Linux and Docker](../docs/user-guide.md#linux-setup) and
[Windows](../docs/user-guide.md#windows-setup) are in the user guide, with advanced
options in [environment setup](../docs/reference.md#environment-setup).

## Install the frontend

From a checkout, use `./native-inference/pnms-model-builder` at the
repository root. To install a frontend wheel, first follow the
[scratch packaging instructions](../docs/releasing.md#build-and-check-the-wheel),
then install the resulting artifact:

```bash
python3 -m pip install /path/to/pnms_model_builder-0.1.0-py3-none-any.whl
pnms-model-builder --help
```

The frontend requires Python 3.10+ and has no mandatory ML dependencies.
Discovery, configuration-only checks and container orchestration do not import
Torch on the host. Local eager checks and builds need a compatible model and
framework environment; builds also need the native SDK and compiler dependencies.

The wheel includes `model_builder.build`, `model_builder.export` and the affine
recipe. Customer projects and adapters remain in [examples](../examples/README.md)
and are not installed as builder resources. Both the checkout launcher and
installed command use the same implementation.

## Source ownership

| Directory/file | Responsibility |
| --- | --- |
| `src/model_builder/build/` | Projects, recipes, input capture, execution, native checks and results; `completion.py` validates receipts and retained artifacts |
| `src/model_builder/export/` | AOTI, ONNX and TensorRT export/compilation primitives |
| `models/affine/` | Bundled recipe and adapter |
| `images/` | Builder Dockerfile, dependency locks and environment guards |
| `tests/` | Frontend, export and harness tests |
| `tests/installation/` | Installed-wheel contract tests |
| `tests/images/` | Image environment guard tests |
| `pyproject.toml` | Package metadata, version and installed resources |
| `toolchain.lock.json` | Default toolchain selection, currently unreleased |

The C++ runtime lives in `../cpp-runtime/`. Shared example preparation tools
live in `../tools/examples/`, with tests in `../tests/examples/`. Keep generated
checkpoints, wheels, graphs, packages and test evidence outside versioned source.

## Development checks

From the repository root:

```bash
python3 -m unittest discover -s native-inference/model-builder/tests -v
python3 -m unittest discover -s native-inference/model-builder/tests/images -v
```

ML-dependent tests need their declared frameworks; report skips separately.
CPU substitutes verify orchestration and failure handling. GPU builds and
native execution establish backend parity. Follow repository instructions for
behavioral tests and source-move characterization.

See [packaging and verification](../docs/releasing.md) for wheel and image
checks, and the [image environment contract](images/README.md) for dependencies.
