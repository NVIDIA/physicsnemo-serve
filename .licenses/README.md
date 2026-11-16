# Updating third-party notices

`licenses.txt` combines Rust notices from `Cargo.lock` and Python notices from
the root `uv.lock`. It covers the optional QA dependencies and all Python
versions/platforms represented in that lockfile, including multiple versions
of a package when the lockfile contains them. Reviewed entries in
[`ml-native.json`](ml-native.json) extend the named inventory to the ML/native
components below. They supplement the root lockfile collectors; they are not a
complete transitive inventory of every GPU runtime, NGC base image, model,
checkpoint, dataset, or native system package.

| Component | Applicable version/source | License | Distribution role |
| --- | --- | --- | --- |
| [PyTorch / LibTorch](https://github.com/pytorch/pytorch) | `2.10.0+cu130` (E2S), `2.10.0+cu128` (CFD/local native profile), `2.10.0a0+a36e1d39eb.nv26.1.42222806` (NGC native builder) | BSD-3-Clause, with retained ATen/CUTLASS notices | Installed Python/runtime libraries; LibTorch linked by AOTI/workflow outputs; ATen templates compiled into the exact-attention plugin. |
| [CUTLASS](https://github.com/NVIDIA/cutlass/tree/f88806b1e31dfa579842638740216dd41fc6c588) | `4.3.1`, commit `f88806b1e31dfa579842638740216dd41fc6c588`, pinned by both PyTorch source revisions | BSD-3-Clause for the C++ headers used here | Templates compiled into the exact-attention plugin; not a separately linked runtime. |
| [Earth2Studio](https://github.com/NVIDIA/earth2studio) | `0.13.0` | Apache-2.0 | Separately installed forecast/model/data package. |
| [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) | `2.1.1` (`nvidia-physicsnemo`) | Apache-2.0 | Separately installed ML/runtime/export package; retain applicable notices if implementation is copied or incorporated into generated payloads. |
| [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd) | `0.0.2`, commit `921f14dc2ac14c04aabffaba3290deb792379dd8` | Apache-2.0 | Separately installed CFD workflow package. |
| [ONNX](https://github.com/onnx/onnx) | `1.18.0` (NGC builder), `1.20.1` (retained fixture's producer), `1.22.0` (E2S/CFD locks) | Apache-2.0; bundled notices retained | Installed export/import tooling. An ONNX model file does not by itself redistribute the ONNX implementation. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | User-selected through `PNMIR_ONNXRUNTIME_ROOT`; no repository pin | MIT | Optional native linked runtime. Inventory the selected release and its bundled third-party notices when distributing it. |
| [Warp](https://github.com/NVIDIA/warp) | `1.15.0` | Apache-2.0 | Installed `warp-lang`; native workflow also uses its headers and links `warp.so`. |

The manifest records exact repository version evidence, upstream notice links,
copyright attributions, distribution roles, and SHA-256-checked local license
texts. The Earth2Studio notice was also verified against the SHA-256-verified
0.13.0 wheel. Identical license texts shared by multiple versions are stored
once and rendered in each applicable component section. These entries retain
the NVIDIA, Meta/PyTorch/Caffe2 contributor, ONNX Project Contributor, and
Microsoft notices; ONNX 1.22.0's upstream `NOTICE` also retains its bundled
Abseil, protobuf and nanobind attributions.

Installing a dependency independently is distinct from copying its
implementation or compiling its headers/templates into a delivered binary.
Shipping an environment, wheel, SDK or image containing an installed dependency
redistributes that dependency. Generated artifacts must carry the notices for
implementation code they include; build-time use alone does not establish that
an entire dependency is in the artifact. Preserve the exact distribution's
bundled notices as well as the named project license.

In particular, the exact-attention plugin includes
`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h` and
instantiates `PyTorchMemEffAttention::AttentionKernel` with CUTLASS types.
PyTorch/ATen and CUTLASS notices must accompany its distributed shared library,
including SDKs, images and model packages containing that library, even when
LibTorch and Python are absent. The native SDK install copies the actual
PyTorch source `LICENSE`, CUTLASS `LICENSE.txt`, and retained ATen header notices
to `share/licenses/physicsnemo-inference/`; keep this directory with extracted
plugins. See the [native notice instructions](../native-inference/cpp-runtime/third_party/README.md)
for alternate source revisions and redistribution details.

The runtime's imported implementation and the Transolver workflow retain their
own component notices, mapped in the [root README](../README.md#license).
CUDA, TensorRT and NGC image components also retain their vendor-supplied terms;
this OSS inventory does not replace those distribution notices.

Install the tested collectors in an isolated tooling environment:

```sh
cargo install cargo-license --version 0.7.0 --locked
export PATH="$HOME/.cargo/bin:$PATH"
uv venv --python 3.12 /tmp/physicsnemo-license-tools
uv pip install --python /tmp/physicsnemo-license-tools/bin/python pip-licenses==5.5.5
```

From the repository root, fetch locked Rust sources and generate a candidate:

```sh
cargo fetch --locked
/tmp/physicsnemo-license-tools/bin/python scripts/update_license_notices.py \
  --output /tmp/physicsnemo-serve-license-candidate.txt \
  --report /tmp/physicsnemo-serve-license-provenance.json
```

Omit `--output` to update `.licenses/licenses.txt` directly after reviewing the
candidate. The default baseline is the current `.licenses/licenses.txt`;
`--baseline` can select another file. A second run against the generated
inventory should leave it unchanged. Generation finishes before the output is
atomically replaced; missing texts, incomplete inventories, and bad hashes fail
without replacing it.

By default, a matching component/version keeps its existing section verbatim,
including its display name, homepage, license label, text, position, and number.
Collector metadata differences alone do not replace a same-version notice.
Legacy project names and equivalent Python version spellings (such as `26`
and `26.0`) are matched to avoid unnecessary delete/add pairs. This is an
incremental dependency update, not a re-audit of existing license declarations.
Use `--refresh-existing` only when intentionally reviewing regenerated notices
for already-listed versions.

The merger uses:

1. `cargo-license --json --all-features` for Rust metadata. The merger filters
   its output against `cargo tree --workspace --all-features --target all` so
   unused optional transitive dependencies are excluded. Cargo runs offline
   after a locked metadata check. License expressions are reported as declared;
   all available source license/notice files are collected, rather than
   selecting one alternative under dual licensing.
2. Existing matching notices when a Rust archive has no license files. These
   fallbacks are printed and recorded in the optional provenance report. A new
   crate without license text fails; supply a reviewed notice before rerunning.
   Local workspace crates can use the root license when their declared license
   matches the workspace license.
   `--reuse-notices` can supply additional texts when generating against an older
   baseline; it defaults to the current inventory and does not affect which
   baseline sections are retained.
3. SHA-256-verified wheels from root `uv.lock`, including platform-specific and
   multiple locked versions. Universal wheels are preferred, then Linux x86_64
   wheels (CPython 3.12 when available). Only `.dist-info` metadata is staged;
   wheel package code is neither installed nor executed.
4. `pip-licenses --format=json --with-urls --with-license-file
   --with-notice-file --no-license-path` against that staged metadata, using
   the tooling interpreter. A supplemental scan collects all shipped license,
   notice, copyright, and author files, including nested bundled-library
   notices. This avoids pip-licenses' first-file-only limitation. Every locked
   Python name/version must appear in its output.
5. Reviewed ML/native entries from `ml-native.json`. The generator verifies the
   stored upstream license files against their recorded hashes and includes
   distribution roles and version evidence in the component sections. These
   entries survive ordinary root-lockfile regeneration. When a component's
   version, source revision, or role changes, review its notice sources and
   update the manifest and snapshots together. ONNX Runtime remains explicitly
   unpinned until a release selects an actual runtime distribution.

Python wheels are cached by hash in
`~/.cache/physicsnemo-serve/license-wheels`. Use `--cache-dir` to select another
directory and `--offline` to require a complete cache. The tooling environment
does not need the project's Python dependencies installed. Unknown license
labels remain `See license files below` when full texts are available.

Review the candidate's package versions, license texts, and copyright notices,
then combine the Rust and Python entries under the existing NVIDIA header in
`licenses.txt`. Preserve the original format: one numbered list,
76-character dash separators, a component/version line,
a homepage line, a license label, and the full license texts beneath each
component. Repeat applicable Rust texts under every component and concatenate
all applicable Python license/notice texts under that component.
Preserve existing display names when they can be matched unambiguously by
name/alias or homepage and version, but do not reuse a display name for a
different component when that would create ambiguous entries. Do not add ecosystem section headings
or archive/hash fields to the combined notice file.

Keep matching same-version sections byte-for-byte, in place with their original
numbers. Remove a superseded version or obsolete component as a complete block
rather than changing its contents inline. Append updated-version and new
component sections at the end,
sorted by display name/version, with numbers above the previous highest entry.
Keep gaps left by removed sections so unrelated entries are not renumbered.

Verify every locked Python name/version and every component in the all-feature,
all-target Rust dependency tree is represented. Do not replace the combined
inventory with the Rust-only candidate or collect notices from an unrelated
local environment.

The provenance report records the resolved component counts, canonical identities and
source files separately from the human-readable notice format.
It describes the collected metadata, which may differ from deliberately
preserved same-version sections in the notice file.
Additional release environments require their own inventories before claiming
coverage for those distributions.
