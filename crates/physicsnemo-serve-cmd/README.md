# PhysicsNeMo Serve CLI

`physicsnemo-serve` runs a single manifest plugin directly — no Redis, no
`inference_server`, no scheduler. It accepts plugins using `simple`,
`prefetch`/`default`, `postprocess`, or single-item `batch` profiles.

```text
prepare → optional prefetch → execute → optional postprocess → result
```

There are three binaries in this crate:

| Binary | Description |
| --- | --- |
| `physicsnemo-serve` (self-contained) | Portable executable with a CPython runtime appended; no external Python needed |
| `physicsnemo-serve` (thin) | Same binary without the appended runtime; uses `--runtime-dir` to point at an external Python environment |
| `physicsnemo-serve-install` | Installer that creates a plugin-specific Python runtime from a lock file |

---

## Build prerequisites

The Rust workspace uses Edition 2024 and pins Rust and Cargo 1.94.1 for the
CLI build. Edition 2024 is included in this toolchain; it is not a separate
Cargo feature or component.

On Ubuntu or Debian Linux, install the native build dependencies, rustup, and
the pinned toolchain with:

```bash
# A minimal machine needs make before it can invoke the installer target.
sudo apt-get update
sudo apt-get install --yes make

make install-serve-cmd-builders
source "$HOME/.cargo/env"
```

The target installs `build-essential`, `ca-certificates`, `cmake`, `curl`, and
`pkg-config` through `apt-get`. It then installs rustup when necessary and sets
Rust 1.94.1 as the default toolchain. It may prompt for a sudo password when
not run as root.

Verify the installation before building:

```bash
command -v rustup
command -v cargo
rustup --version
rustc --version
cargo --version
```

`cargo` should resolve under `$HOME/.cargo/bin`, and both `rustc` and `cargo`
should report version 1.94.1.

---

## 1. Self-contained executable

The binary can carry a compressed CPython runtime as a payload appended to the
native executable (ELF on Linux or Mach-O on macOS). On first run it extracts
to a content-addressed cache under
`~/.cache/physicsnemo-serve/inference-cli/`. Native Python extensions remain
real files after extraction.

### How the Python loader works

Python is not statically linked into the Rust program. The self-contained
artifact is a normal native Rust executable followed by a compressed,
relocatable Python filesystem payload and a small footer:

```text
+-------------------------+
| Rust executable         |
+-------------------------+
| zstd-compressed tar     |  CPython, dependencies, and runner scripts
+-------------------------+
| bundle footer           |  magic, payload offset, length, and SHA-256
+-------------------------+
```

The artifact is assembled in three stages:

1. `assemble_runtime.py` copies a complete uv-managed standalone CPython
   prefix into a staging directory. It installs the hashed requirements and
   adds the PhysicsNeMo Serve Python modules and runner scripts.
2. Cargo compiles the thin Rust `physicsnemo-serve` executable.
3. `physicsnemo-serve package` creates a deterministic tar archive of the
   runtime, compresses it with zstd, appends it to the executable, and writes a
   footer containing the payload location and checksum.

When no `--runtime-dir` or `PHYSICSNEMO_SERVE_RUNTIME_DIR` override is present,
the Rust loader reads the footer from its own executable. It verifies the
payload checksum and extracts the runtime into a SHA-256-addressed directory
under `~/.cache/physicsnemo-serve/inference-cli/`. An exclusive cache lock
makes concurrent first runs safe, and later runs reuse the completed cache.
The CLI then starts the extracted `bin/python` with
`scripts/plugin_direct_runner.py`.

Extraction is intentional: CPython native extensions and shared libraries
need real filesystem paths. The result is therefore one file for distribution,
but it expands into the user cache when it runs. It still depends on a
compatible host OS and architecture, plus any system GPU driver required by
CUDA packages.

The `build-serve-cmd` Make target produces only the thin Rust executable, while
`build-serve-cmd-self-contained` performs the complete runtime assembly and packaging
workflow. Likewise,
`Dockerfile.physicsnemo-serve-cmd` places the Rust executable and its Python
runtimes in separate paths inside the container; it does not append those
runtimes to the executable.

### Build self-contained executable

The self-contained Make target performs all required steps and writes
`dist/physicsnemo-serve-self-contained`:

```bash
make build-serve-cmd-self-contained
```

The Python version, requirements lock, compression level, and output are
configurable. For example, to package the Earth2Studio runtime:

```bash
make build-serve-cmd-self-contained \
    SERVE_CMD_RUNTIME_REQUIREMENTS=packaging/physicsnemo-serve-cmd/external-runtime/requirements-e2s.lock \
    SERVE_CMD_COMPRESSION_LEVEL=12
```

The equivalent manual workflow is:

```bash
# Install a standalone CPython (not a venv — a uv-managed standalone install)
uv python install 3.12
PYTHON_BIN="$(uv python find 3.12)"
PYTHON_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"

# Assemble the runtime directory from the base lock file
uv run python packaging/physicsnemo-serve-cmd/assemble_runtime.py \
    --python-prefix "$PYTHON_PREFIX" \
    --requirements packaging/physicsnemo-serve-cmd/runtime-base.lock \
    --output build/inference-cli-runtime

# Compile the binary
cargo build --locked --release \
    --package physicsnemo-serve-cmd \
    --bin physicsnemo-serve

# Append the runtime payload to produce the self-contained executable
target/release/physicsnemo-serve package \
    --runtime-dir build/inference-cli-runtime \
    --output dist/physicsnemo-serve-self-contained
```

Pass `--compression-level LEVEL` (from `-7` through `22`) to trade packaging
time for executable size. The default is level 5.

`runtime-base.lock` is the minimal fixed dependency allowlist. To include
additional packages, generate a different hashed lock and pass it to
`assemble_runtime.py`. Repo-local packages can be added with repeated
`--extra-python-package PATH` arguments.

`--python-prefix` requires a standalone uv-managed CPython containing its
`BUILD` marker. Venvs are rejected because their interpreter symlinks commonly
refer to the host installation and are not portable.

### Run self-contained executable

```bash
dist/physicsnemo-serve-self-contained infer \
    --plugin /path/to/plugin \
    --request request.json \
    --output-dir outputs \
    --device 0
```

The result envelope is written to stdout and diagnostics go to stderr.

---

## 2. Thin binary (external runtime)

The same compiled binary, without the appended runtime. At startup the CLI
resolves the runtime via:

1. `--runtime-dir DIR` flag
2. `PHYSICSNEMO_SERVE_RUNTIME_DIR` environment variable
3. Appended runtime payload (falls back if present — thin binary has none)

### Build thin binary

```bash
# Native build (runs on the host platform)
make build-serve-cmd
# Output: dist/physicsnemo-serve

# Cross-compile to Linux x86_64 from macOS (requires zig and cargo-zigbuild)
brew install zig cmake
cargo install cargo-zigbuild --locked
rustup toolchain install 1.94.1 --profile minimal
rustup target add --toolchain 1.94.1 x86_64-unknown-linux-gnu
make build-serve-cmd-linux-amd64
# Output: dist/physicsnemo-serve-linux-amd64
```

### Create the external runtime

The fastest path uses the included shell script, which creates a relocatable
uv venv pinned to the Earth2Studio cu130 requirements:

```bash
packaging/physicsnemo-serve-cmd/external-runtime/setup.sh \
    "$HOME/.local/share/physicsnemo-serve/runtimes/earth2studio"
```

For the CFD runtime or other lock files, use `physicsnemo-serve-install`
instead (see section 3 below).

### Run thin binary

```bash
export PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python

dist/physicsnemo-serve infer \
    --runtime-dir "$HOME/.local/share/physicsnemo-serve/runtimes/earth2studio" \
    --plugin plugins/e2s-deterministic \
    --request request.json \
    --output-dir outputs \
    --device 0
```

The sample launcher `packaging/physicsnemo-serve-cmd/external-runtime/run.sh`
defaults to the runtime path above and a binary at `$HOME/physicsnemo-serve`:

```bash
packaging/physicsnemo-serve-cmd/external-runtime/run.sh \
    --plugin plugins/e2s-deterministic \
    --request request.json \
    --output-dir outputs \
    --device 0
```

### Runtimes in the Docker image

The CLI Dockerfile exposes two final image targets. The default
`ubuntu-runtime` target uses the minimal Jammy base and keeps independent CUDA
stacks:

```text
/opt/physicsnemo-serve/runtimes/e2s   # consolidated E2S requirements, cu130
/opt/physicsnemo-serve/runtimes/cfd   # CFD requirements, cu128
```

Build it with `make serve-cmd-image`, or directly with Docker:

```bash
docker build \
    --target ubuntu-runtime \
    --file Dockerfile.physicsnemo-serve-cmd \
    --tag physicsnemo-serve-cmd:latest \
    .
```

The `pytorch-runtime` target is based on `nvcr.io/nvidia/pytorch:26.01-py3`,
matching the service's PyTorch/CUDA toolkit foundation. Both isolated plugin
runtimes use cu130 wheels, including the CFD runtime generated from
`requirements-cfd-cu130.lock`:

```bash
make serve-cmd-pytorch-image

# Equivalent direct build:
docker build \
    --target pytorch-runtime \
    --file Dockerfile.physicsnemo-serve-cmd \
    --tag physicsnemo-serve-cmd:pytorch \
    .
```

The image defaults to the E2S runtime via `PHYSICSNEMO_SERVE_RUNTIME_DIR`. To
use the CFD runtime:

```bash
physicsnemo-serve infer \
    --runtime-dir "$PHYSICSNEMO_SERVE_CFD_RUNTIME_DIR" \
    --plugin /opt/physicsnemo-serve/plugins/physicsnemo-cfd-surface-benchmark \
    --request request.json \
    --output-dir outputs
```

---

## 3. Installer binary

`physicsnemo-serve-install` creates a plugin-specific Python runtime from a
requirements lock file. It handles the full provisioning sequence: creating a
venv in a staging directory, installing the base runner dependencies and plugin
requirements, copying the embedded runner/SDK modules, verifying the imports
listed under `developer.readiness.python_modules` in each `plugin.yaml`, and
then atomically publishing the runtime directory. Dependencies are never
installed during inference.

If `uv` is not already on `PATH`, the installer bootstraps a pinned release
under the user's local PhysicsNeMo Serve data directory without requiring
`sudo` or modifying shell startup files.

### Build installer binary

```bash
# Native build
make build-serve-installer
# Output: dist/physicsnemo-serve-install

# Cross-compile to Linux x86_64 from macOS
make build-serve-installer-linux-amd64
# Output: dist/physicsnemo-serve-install-linux-amd64
```

### Install a runtime

```bash
dist/physicsnemo-serve-install \
    --plugin plugins/e2s-deterministic \
    --requirements packaging/physicsnemo-serve-cmd/external-runtime/requirements-e2s.lock \
    --runtime-dir "$HOME/.local/share/physicsnemo-serve/runtimes/e2s" \
    --python 3.12 \
    --torch-backend cu130
```

`--plugin` may be repeated to create one shared environment covering multiple
plugins. `--requirements` may also be repeated. When `--requirements` is
omitted the installer uses each `<plugin>/requirements.txt` that exists.

Pass `--uv PATH` to supply a preinstalled copy of the installer's pinned `uv`
version for offline or controlled environments. Use `--skip-import-checks` only
when readiness imports require unavailable runtime resources such as model
weights.

Once the runtime is ready, use the thin binary against it:

```bash
dist/physicsnemo-serve infer \
    --runtime-dir "$HOME/.local/share/physicsnemo-serve/runtimes/e2s" \
    --plugin plugins/e2s-deterministic \
    --request request.json \
    --output-dir outputs \
    --device 0
```

---

## Runtime notes

`--device` accepts CUDA device ordinals, UUIDs, and MIG identifiers; it is
forwarded as `CUDA_VISIBLE_DEVICES`. GPU plugins require a compatible NVIDIA
GPU, kernel driver, and writable cache/output storage. Model weights and
downloaded inputs use persistent caches and are not part of the binary.

The external runtime does not include the repo-local `e2s_zarr_io` Rust
extension. Set `PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python` (the default in
`run.sh` and the Docker image) unless that extension is explicitly installed in
the runtime.

Verified HTTPS prefetch is fail-closed. For the bundled CFD examples see
`plugins/physicsnemo-cfd-surface-benchmark/README.md` for the required host
allowlists; the Docker image already configures them. Customer plugins must
supply their own `E2S_PREFETCH_ALLOWED_HTTPS_HOSTS`.

The binary contains a hidden `__prefetch` subcommand used for internal
JSON-over-stdin communication between the host process and the Python worker.
It is not a supported end-user interface.
