# e2s_zarr_io Parity Framework

This directory contains parity utilities and suites for validating logical data consistency
across backends (`py_sync`, `py_async`, `rust`) after read/decompression.

## Locked policy

- Tier0 baseline refresh is manual.
- Tier1+ CI compares Rust manifests against committed truth manifests.
- Physical chunk/compression differences are non-gating.
- CI parity gate executes `pytest parity/suites` (including truth-catalog integrity checks).

## Committed Tier0 case

- Catalog: `parity/ground_truth/catalog.yaml`
- Case: `deterministic_io_small_v1`
- Truth: `parity/ground_truth/truth/deterministic_io_small_v1.truth_manifest.json.zst`
- Baseline report: `parity/ground_truth/reports/deterministic_io_small_v1.baseline_report.json`

## Quick commands

Record manual Tier0 blessed truth:

```bash
python -m parity.tools.record_truth \
  --case-spec parity/ground_truth/cases/example.case_spec.json \
  --py-sync-dataset <temporary-directory>/example_py_sync.zarr \
  --py-async-dataset <temporary-directory>/example_py_async.zarr \
  --output parity/ground_truth/truth/example.truth_manifest.json \
  --baseline-report-output parity/ground_truth/reports/example.baseline_report.json \
  --earth2studio-commit <commit> \
  --python-version <version> \
  --zarr-python-version <version> \
  --case-spec-sha256 <sha256>
```

Or generate datasets from built-in workflow adapters (runner mode):

```bash
python -m parity.tools.record_truth \
  --case-spec parity/ground_truth/cases/deterministic_io_small_v1.case_spec.json \
  --generated-datasets-dir <temporary-directory>/e2s_parity_datasets \
  --output parity/ground_truth/truth/deterministic_io_small_v1.truth_manifest.json.zst \
  --baseline-report-output parity/ground_truth/reports/deterministic_io_small_v1.baseline_report.json \
  --runner-output-map parity/ground_truth/reports/deterministic_io_small_v1.runner_paths.json \
  --earth2studio-commit <commit> \
  --python-version <version> \
  --zarr-python-version <version> \
  --case-spec-sha256 <sha256>
```

Verify candidate parity (Tier1+):

```bash
python -m parity.tools.verify_parity \
  --truth-manifest parity/ground_truth/truth/example.truth_manifest.json \
  --case-spec parity/ground_truth/cases/example.case_spec.json \
  --candidate-dataset <temporary-directory>/example_rust.zarr \
  --generated-by-backend rust
```

Or run candidate backend directly from case spec (runner mode):

```bash
python -m parity.tools.verify_parity \
  --truth-manifest parity/ground_truth/truth/deterministic_io_small_v1.truth_manifest.json.zst \
  --case-spec parity/ground_truth/cases/deterministic_io_small_v1.case_spec.json \
  --candidate-backend rust \
  --generated-datasets-dir <temporary-directory>/e2s_parity_datasets \
  --runner-output-map parity/ground_truth/reports/deterministic_io_small_v1.verify_paths.json
```

