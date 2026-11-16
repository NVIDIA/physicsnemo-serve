# Updating third-party notices

`licenses.txt` records Rust notices from `Cargo.lock` and Python notices from
the root `uv.lock`. It covers the optional QA dependencies and all Python
versions/platforms represented in that lockfile, including multiple versions
of a package when the lockfile contains them. Reviewed entries in
[`ml-native.json`](ml-native.json) extend the named inventory to the ML/native
components below. They supplement the root lockfile inventory; they are not a
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
copyright attributions, distribution roles, and component/version references to
the full texts in [`licenses.txt`](licenses.txt). Each `notice_reference` names
the inventory path and the exact component and version in its numbered heading;
the referenced text runs until the next numbered component section or end of
file. Section numbers are not identifiers and may change. The Earth2Studio
notice was also verified against the SHA-256-verified 0.13.0 wheel. There are no
separate license snapshots or snapshot hashes to maintain. These entries retain
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
LibTorch and Python are absent. The native SDK install copies `licenses.txt`,
including the PyTorch/CUTLASS licenses and retained ATen header notices,
to `share/licenses/physicsnemo-inference/`; keep this directory with extracted
plugins. See the [native notice instructions](../native-inference/cpp-runtime/README.md#shipping-exact-plugin-notices)
for alternate source revisions and redistribution details.

The runtime's imported implementation and the Transolver workflow retain their
own component notices, mapped in the [root README](../README.md#license).
CUDA, TensorRT and NGC image components also retain their vendor-supplied terms;
this OSS inventory does not replace those distribution notices.

## Maintaining the inventory

Maintain `licenses.txt` directly when dependency versions or distribution roles
change. Use `Cargo.lock`, `uv.lock`, the separate ML runtime locks and native
builder configuration to identify the applicable versions. `ml-native.json`
retains source/version evidence and inventory references for the reviewed ML and
native entries; it is a provenance reference, not a regeneration command.

For each new or changed component, obtain the full license and copyright notices
from its exact upstream release or distribution. Retain bundled third-party
notices and distinguish separately installed dependencies from copied source,
linked libraries and template code compiled into distributed outputs. Update
relevant version evidence, source links and embedded notice sections together.

Preserve unchanged component sections verbatim, including their names, numbers,
license labels and full texts. Remove superseded entries as complete sections
and append new entries with numbers above the existing maximum. Keep the NVIDIA
header, 76-character separators and per-component format. Review the resulting
diff for missing notices, stale versions and unintended changes to other entries.

Before distributing an SDK, plugin, image or model package, verify that it carries
the notices applicable to the code and libraries it actually contains. The named
inventory does not replace an inventory of the exact release environment.
