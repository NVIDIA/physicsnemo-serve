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

## Use a plugin-specific external runtime

A thin CLI binary can use a customer-managed Python environment instead of an
appended runtime. The external runtime owns Python, PyTorch, Earth2Studio, and
all plugin-specific packages:

```bash
make build-serve-cmd

packaging/physicsnemo-serve-cmd/external-runtime/setup.sh \
  "$HOME/.local/share/physicsnemo-serve/runtimes/earth2studio"

export PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python
./dist/physicsnemo-serve infer \
  --runtime-dir "$HOME/.local/share/physicsnemo-serve/runtimes/earth2studio" \
  --plugin plugins/e2s-deterministic-earth2 \
  --request request.json \
  --output-dir outputs \
  --device 0
```

On an Apple Silicon Mac, build a thin Linux x86_64 executable with:

```bash
brew install zig
cargo install cargo-zigbuild --locked
rustup target add x86_64-unknown-linux-gnu
make build-serve-cmd-linux-amd64
```

The sample launcher defaults to the runtime path above and a binary installed
at `$HOME/physicsnemo-serve`:

```bash
packaging/physicsnemo-serve-cmd/external-runtime/run.sh \
  --plugin plugins/e2s-deterministic-earth2 \
  --request request.json \
  --output-dir outputs \
  --device 0
```

`--runtime-dir` takes precedence over `PHYSICSNEMO_SERVE_RUNTIME_DIR`. If
neither is set, the CLI falls back to its appended runtime. Runtime creation is
an explicit provisioning step; dependencies are never installed during
inference.

The provided external runtime does not include the repo-local `e2s_zarr_io`
extension, so its launcher selects the Python Zarr backend by default. Set the
variable to `rust` only when that extension is installed in the runtime.

### Install a runtime for one plugin

Build the separate installer binary:

```bash
make build-serve-installer
```

The installer uses `uv`, creates the virtual environment in a temporary
directory, installs the base runner dependencies and plugin requirements,
copies the embedded runner/SDK modules, verifies the imports listed under
`developer.readiness.python_modules`, and only then publishes the requested
runtime directory. If `uv` is not already on `PATH`, the installer bootstraps a
pinned release under the user's local PhysicsNeMo Serve data directory without
requiring `sudo` or editing shell startup files:

```bash
./dist/physicsnemo-serve-install \
  --plugin plugins/e2s-deterministic \
  --requirements packaging/physicsnemo-serve-cmd/external-runtime/requirements-earth2studio.txt \
  --runtime-dir "$HOME/.local/share/physicsnemo-serve/runtimes/e2s-deterministic" \
  --python 3.12 \
  --torch-backend cu128
```

`--plugin` may be repeated to create one shared environment and verify the
readiness imports declared by every included plugin. `--requirements` may also
be repeated. When it is omitted, the installer uses each
`<plugin>/requirements.txt` that exists. Import names in
`plugin.yaml` are used only for verification because a Python import name does
not reliably identify its installable package. Pass `--uv PATH` to use a
preinstalled copy of the installer's pinned `uv` version in an offline or
controlled environment. Use
`--skip-import-checks` only when imports require unavailable runtime resources
such as model assets.

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

Packaging uses zstd level 5 by default. Use `--compression-level LEVEL` (from
`-7` through `22`) to trade packaging time for executable size.

`runtime-base.lock` is the minimal fixed dependency allowlist. Generate a
different hashed lock and pass it to `assemble_runtime.py` when the supported
external plugins require additional packages. Repo-local packages can be
included with repeated `--extra-python-package PATH` arguments. Dependencies
are never installed while running inference.

`--python-prefix` accepts only a standalone uv-managed CPython installation
containing its `BUILD` marker. Venvs are rejected because their interpreter
symlinks and standard-library configuration commonly refer to the host Python
installation and are not portable. Absolute or prefix-escaping symlinks are
also rejected before the runtime is copied.

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

Verified HTTPS prefetch is fail-closed. For the bundled CFD examples, set the
exact source and signed-redirect hosts documented in
`plugins/physicsnemo-cfd-surface-benchmark/README.md`; the CLI Docker image
already provides those allowlists for its bundled examples. Customer plugins
must provide their own `E2S_PREFETCH_ALLOWED_HTTPS_HOSTS` deployment setting.

`--device` accepts CUDA device ordinals, UUIDs, and MIG identifiers and is
forwarded as `CUDA_VISIBLE_DEVICES`. The CLI executes plugin Python code with
the invoking user's permissions, so plugin directories must be treated as
trusted code.

The binary also contains a hidden `__prefetch` subcommand used for internal
JSON-over-stdin communication. It is not a supported end-user interface.
