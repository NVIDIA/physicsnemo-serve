# PhysicsNeMo Serve direct inference CLI

`physicsnemo-serve` runs one external manifest plugin without Redis,
`inference_server`, workers, or a scheduler.

```text
prepare -> optional prefetch -> execute -> optional postprocess -> result
```

The executable accepts JSON plugins using `simple`, `prefetch`/`default`,
`postprocess`, or single-item `batch` profiles. Ensemble fanout, collect,
publication, multipart ingress, and custom framework stages are rejected
explicitly.

## Build a self-contained executable

The launcher packages a filesystem CPython runtime as a compressed payload
appended to its ELF. Native Python extensions remain real files after a
content-addressed first-run extraction.

```bash
uv python install 3.12
PYTHON_BIN="$(uv python find 3.12)"
PYTHON_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"

uv run python packaging/physicsnemo-serve-cmd/assemble_runtime.py \
  --python-prefix "$PYTHON_PREFIX" \
  --requirements packaging/physicsnemo-serve-cmd/runtime-base.lock \
  --output build/inference-cli-runtime

cargo build --release -p physicsnemo-serve-cmd --bin physicsnemo-serve
target/release/physicsnemo-serve package \
  --runtime-dir build/inference-cli-runtime \
  --output dist/physicsnemo-serve
```

`runtime-base.lock` is the minimal fixed dependency allowlist. Generate a
different hashed lock and pass it to `assemble_runtime.py` when the supported
external plugins require additional packages. Repo-local packages can be
included with repeated `--extra-python-package PATH` arguments. Dependencies
are never installed while running inference.

## Run

```bash
./dist/physicsnemo-serve infer \
  --plugin /path/to/plugin \
  --request request.json \
  --output-dir outputs \
  --device 0
```

The result envelope is written to stdout and diagnostics go to stderr. Model
weights and downloaded inputs use persistent caches; they are not part of the
executable. GPU plugins still require a compatible NVIDIA GPU, kernel driver,
and writable cache/output storage.
