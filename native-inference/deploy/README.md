# Native release construction

The [native authoring guide](../../docs/native-workflow-user-guide.md) describes
project generation, release planning, the CPU build host and Lepton GPU gates.
The [serving guide](../../docs/native-worker-serving-integration.md) defines the
worker, readiness and same-slot recovery contracts consumed by a release.

`native_release.py` coordinates immutable compilation identities, verified model
caches, staged bundles, Docker image construction, exact-image inspection and
qualification. `native_release_artifacts.py` seals and retrieves inventories;
the controller rejects changed or unexpected files during extraction.

`native_release_job.py` runs the installed Model Builder inside an ordinary GPU
container. `native_release_http_job.py` qualifies the completed image through its
HTTP API. CFD has separate compile/reference and HTTP drivers. The Lepton adapter
owns finite deadlines, its own jobs and private artifact handoff. It does not
require nested Docker or make Lepton a dependency of the deployed worker.

`Dockerfile.stack` prepares a matched installed SDK from an immutable toolchain
base. `Dockerfile.workflow` builds the C++ workflow host and Rust serving binaries,
then stages the sealed compiled bundle into the runtime image. `Dockerfile.runner`
and `Dockerfile.native` support the lower-level CFD release construction path.
The selected builder/server images and backend must be compatible; their
immutable identities are recorded and rechecked.

Release payloads live under `/opt/physicsnemo-release`. Compiled custom workflow
code lives in the installed host under `/opt/physicsnemo-workflow/lib`. Model
compilation and artifact relocation are construction steps, not request-time
operations. Existing standalone Model Builder commands and Python plugin builds
remain unchanged.
