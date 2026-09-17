# C++ runtime organization and public names

The pre-release C++ runtime now lives in `native-inference/cpp-runtime`. Public names
are:

| Previous | Current |
| --- | --- |
| `pnm-ir` | `physicsnemo-infer` |
| `pnm-ir-device-runner` | `physicsnemo-infer-device-runner` |
| `pnmir/...` headers | `physicsnemo/inference/...` headers |
| `pnmir` C++ namespace | `physicsnemo::inference` |
| `find_package(PNMIR CONFIG)` | `find_package(PhysicsNeMoInference CONFIG)` |
| `pnmir::pnmir` CMake target | `PhysicsNeMoInference::runtime` |
| `pnmir::aoti`, `pnmir::onnxruntime`, `pnmir::tensorrt` | `PhysicsNeMoInference::aoti`, `PhysicsNeMoInference::onnxruntime`, `PhysicsNeMoInference::tensorrt` |
| `libpnmir` core library | `libphysicsnemo_inference` |

Update application includes, namespace references, CMake package/target names
and executable paths together. No legacy aliases are provided in this
pre-release rename. Existing `PNMIR_*` build configuration flags remain valid.
The `.pnmir` format, manifest fields, producer metadata, numerical gates and
runtime behavior are unchanged.

Tiny factories and frozen packages moved from `sdk/examples` to
`cpp-runtime/tests/fixtures`. The frozen ONNX package was not regenerated.
Fixture reproduction sources retain their original identities. Optional backend
tests use the sibling `model-builder/src` exporter or an installed builder package.

Before and after characterization includes all six core CTest entries,
static/shared installed consumers after source/build removal, fixture hashes,
and the affine/channel analytic fixture outputs. Evidence is kept outside source
under `/Users/zhiweis/Projects/physicsnemo-serve-evidence/inference-organization/sdk`.
The historical import ledger below retains the names and validation claims that
applied at that earlier stage.

## Original native SDK import

Source: `physicsnemo-inference-runtime`, commit
`f6bf26090d7cd0484781ed4e1af5a47ea5469413` (2026-09-10 migration).
The source worktree was `cfd-lepton-jobs`. Its Apache-2.0 license is retained in
this directory and installed with the SDK.

## Mechanical migration

Copied 50 source/fixture files: `include/pnmir`, native `src` and `app`, native
and backend integration tests, `CMakeLists.txt`, the license, mock/AOTI examples
and the small reviewed ONNX channel-flow fixture with its reproduction source.
The namespace, model manifest, backend selection, tensor ownership and native
backend implementations are preserved. Lepton controllers, TDD enforcement
tools and generated GPU packages were not copied into the SDK.

The original core was configured, built and tested before migration. All three
CTest entries passed: `pnmir_tests`, `pnmir_v1_api_tests` and
`pnmir_cli_dynamic_values`. Their migrated tests remain in the regression suite.
Optional Python backend tests use the one sibling `builder/pnmir_export`
implementation through a test-only CTest environment; no exporter source is
duplicated into the SDK.

## Behavioral additions

1. Installed `PNMIRConfig.cmake`, version metadata and exported core/optional
   backend targets, transitive C++20 requirements, relative installation library
   lookup and installed license. The new install test first failed its
   observable assertion that the installed CMake entry point existed. The
   unchanged test then installed and relocated the SDK, removed its copied
   source/build directories, and built/ran an external C++ consumer and CLI.
   Static and shared core installs have separate required test entries.
2. CLI `--output-metadata` reports actual output tensor metadata, selected
   backend, accepted execution device and synchronous completion. Two new tests
   first failed because the CLI lacked the requested option; the unchanged
   tests now check dynamic dimensions and integer outputs with escaped tensor
   names. A regression checks that failed inference produces no fresh report.

No SDK install or metadata change establishes full scientific model parity.
Core installed-consumer tests require no Python model package. Optional GPU
backends still require a separately supplied compatible native dependency stack;
this migration does not claim a redistributable complete GPU dependency bundle.

## Evidence

Logs and source/test/file identities are retained outside versioned source at:

```text
/Users/zhiweis/Projects/physicsnemo-serve-evidence/model-builder-bootstrap/sdk/
```

`migration-inventory.json` records the original file hashes. `red-source/` and
`red-identities.json` freeze the pre-implementation SDK and new tests.
`characterization-*.log`, `install-red.log`, `install-green.log`,
`metadata-red.log`, `metadata-green.log` and `regression-green.log` record the
native characterization and red/green outcomes. `verification.json` binds the
final source/test/log identities and commands. GPU validation, when performed,
is recorded separately by the parent build campaign with its exact remote
source, native environment and commands.

Subsequent Ruff formatting preserved Python behavior. The final formatted
metadata/install tests were replayed against the recorded pre-feature CLI/SDK
and the completed implementation; `formatted-verification.json` binds their
new byte identities and `*-formatted-*.log` records the intended red/green
outcomes. The original historical test-first evidence is retained unchanged.
