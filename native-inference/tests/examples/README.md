# Example validation

Customer examples contain `model-build.json` and `adapter.py`. Tests live here.
The affine and AOTI tests also exercise the fixture preparation helpers under
`tools/examples/`.

From the repository root, with Torch installed:

```bash
python -m unittest discover -s native-inference/tests/examples -p 'test_*.py' -v
```

This runs the configured affine, AOTI profiles, GeoTransolver, DoMINO and
Transolver tests. CI runs the same suite in the Windows CPU AOTI job, which
already installs Torch.

`test_geotransolver.py` exercises the current two-file
GeoTransolver example, which loads an imported configuration and checkpoint and
generates small synthetic cached-core inputs. The test module includes a tiny
upstream model double and raw/cached tensors for structural equality checks.
The tests also exercise the generic checkpoint importer and captured authoring
worker, including source isolation and rejection of changed captured inputs.

The former test-only preparation/qualification workflow has been removed.
Its still-applicable checks are covered at their current API boundaries:

| Contract | Current coverage |
| --- | --- |
| Exact profile/assets, constructor/state keys, deterministic cases, cached-core equality | `test_geotransolver.py` |
| Imported constructor, tensor preservation, source digest and artifact identities | `test_geotransolver.py` and `model-builder/tests/test_checkpoint_import_{worker,cli}.py` |
| Captured source isolation, changed inputs, strict weights and configuration | `test_geotransolver.py` and `model-builder/tests/test_{authoring_sources,authoring_worker,model_input_worker,project_binding}.py` |
| Every native case/backend, failed builds, output preservation, target selection | `model-builder/tests/test_{worker,project_cli,project_targets}.py` |
| Completion evidence, qualification-before-release and exact output bytes | `model-builder/tests/test_{container,pre_release_check,geotransolver_exact_profile}.py` |

Paths in the table are relative to `native-inference/`, except the example test
paths, which are relative to this directory. Historical `workflow.json`,
`preparation.json`, frozen fixture/config digests and portable full-model reference
manifests belonged to the deleted test-only implementation. The current example
has no preparation command or full-model/native qualification gate; tests of those
retired formats are not retained. Generic native parity and source/receipt
validation remain covered by the builder tests above.

`test_domino.py` verifies the DoMINO surface project's backend and asset contract,
as well as repeatable valid inputs with positive surface areas.

`test_transolver.py` verifies the Transolver surface imported constructor, checkpoint,
static tensor contract and deterministic cases using a tiny upstream model
double. Native raw-mesh workflow tests are separate under
[`workflows/transolver`](../../workflows/transolver/README.md).

The GeoTransolver tests use a tiny upstream model double; they do not establish
real-checkpoint GPU or scientific CFD accuracy.

`test_aoti_profile_examples.py` verifies the AOTI example profiles and deterministic
fixture preparation.

AOTI option and export tests remain in `model-builder/tests/test_aoti_options*.py`
and `model-builder/tests/test_aoti_option_export.py`.
