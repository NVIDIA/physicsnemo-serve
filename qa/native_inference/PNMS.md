# Unified application QA

The public [`pnms` CLI](../../application-cli/README.md) runs on the user's own
machine. Lepton is used by repository acceptance QA, not by the user's local
application workflow. CPU unit/component checks may run locally or on Windows;
GPU and service acceptance run as finite Lepton jobs.

## Source-bound development profiles

Capture a clean revision with `scripts/prepare_pnms_qa_context.py`. Its archive,
inventory, file modes and hashes travel together. Build `Dockerfile.pnms` on a
Linux Docker host with an explicit compatible immutable SDK/toolchain base.
The image replaces inherited source and QA helpers with the captured revision.
Do not qualify a new checkout by reusing an older development image's source.

The manually dispatched `lepton_pnms_qa.yml` workflow consumes that development
image and the same full source SHA. It runs these mandatory profiles:

| Profile | Controller | Required cases |
| --- | --- | ---: |
| Stage source and pinned requirements | `run_lepton_pnms_stage_qa.py` | 1 |
| Fresh venv: AOTI tensor and TensorRT custom builds | `run_lepton_pnms_venv_qa.py` | 18 |
| HTTP on those exact venv builds and interpreter | `run_lepton_pnms_http_qa.py --mode host` | 17 |
| Public Python and five native local workflows | `run_lepton_pnms_qa.py` | 47 |
| HTTP on the same local bundles, including Python | `run_lepton_pnms_http_qa.py --mode host --include-python` | 49 |

The profiles cover new and existing Python plugins; both native backends with
tensor/custom workflows; raw multipart input; independent held-out requests;
model/workflow cache reuse; relocation; environment drift and lock refresh;
readiness, invalid requests, overlap, worker recovery and owned cleanup.
The 132-case workflow must pass without changing its required case inventory.

Configure the existing CI workspace, node group, pull-secret name and shared
storage settings through CI secrets/variables. The development-image digest and
source SHA are explicit workflow inputs. The pinned controller version is
`leptonai==0.27.2`. The optional venv wheelhouse is an installation input, not a
substitute for source or package identity checks.

## Image packaging acceptance

Image qualification is separate from the 132-case profile workflow. Use the
carried QA-only `run_lepton_pnms_release_build.py` controller to execute a public
build and retrieve a compact verified handoff after the GPU job is removed.
On a CPU Docker host, package it through the public command:

```bash
pnms build --from-build /shared/build/build.json --image REGISTRY/APP:TAG --push
```

Verify the source inventory and unchanged compiled package before packaging.
Then use `run_lepton_pnms_http_qa.py --mode image` with the resulting immutable
digest. The `tensor-files-aoti` and `tensor-files-tensorrt` projects provide three
independent raw-upload fixtures and byte-exact output goldens. Each backend has
three build/export gates and sixteen image HTTP/fault gates. Actual Python-image
qualification uses `Dockerfile.pnms-python-base` built from the current
development image, public `pnms build --image`, and the nine-case Python image
HTTP profile. These 47 gates are run separately from the profile workflow.
Qualify custom workflow images explicitly when reporting that coverage.

Record the exact application, bundle, model compilation source, SDK/library,
source, image and fixture identities. A cached compiled model retains its
original compilation provenance. Image construction is not itself a GPU or
HTTP qualification result. Preserve original failed attempts separately from
later passing runs, and verify every required result and owned cleanup receipt.

No nested Docker is required: GPU jobs run model compilation and inference in
ordinary containers, while image assembly uses a Linux Docker host. The real
host Docker-launcher E2E test remains explicitly waived; its command/path
contracts are covered by component tests. That waiver does not replace the
ordinary-container, fresh-venv or immutable release-image acceptance profiles.
