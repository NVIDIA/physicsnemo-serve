# Example validation

Customer examples contain `model-build.json` and `adapter.py`. Tests live here.
The affine and AOTI tests also exercise the fixture preparation helpers under
`tools/examples/`.

From the repository root, with Torch installed:

```bash
python -m unittest discover -s native-inference/tests/examples/configured-affine -p 'test_*.py' -v
python -m unittest discover -s native-inference/tests/examples/aoti-profiles -p 'test_*.py' -v
python -m unittest discover -s native-inference/tests/examples/geotransolver-surface-core -p 'test_*.py' -v
```

`geotransolver-surface-core/test_minimal.py` exercises the current two-file
GeoTransolver example, which loads an imported configuration and checkpoint and
generates small synthetic cached-core inputs.

The other GeoTransolver tests preserve constructor, checkpoint, cached-input,
reference, source-identity, target, and failure checks from the former preparation
workflow. Its adapter, project template, and preparation code live under the
test-only `geotransolver-surface-core/legacy/` directory. `qualification.py`
retains its reference-validation support. These historical fixtures are neither
installed with Model Builder nor needed by the current customer example.

The GeoTransolver tests use a tiny upstream model double; they do not establish
real-checkpoint GPU or scientific CFD accuracy.

AOTI option and export tests remain in `model-builder/tests/test_aoti_options*.py`
and `model-builder/tests/test_aoti_option_export.py`.
