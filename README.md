# PhysicsNeMo-Serve

PhysicsNeMo-Serve is a Rust-based serving package for MLOps engineers that turns supported PhysicsNeMo inference pipelines into production-ready GPU services, delivering a shorter checkpoint-to-endpoint path with backend-aware deployment support.

It implements a manifest-driven inference service for Python plugins, with Rust handling orchestration, scheduling, artifact serving, and result persistence.

It reduces the friction between trained model checkpoints and production inference by giving teams a structured path for export, backend selection, and deployment, especially in NVIDIA GPU environments. 

## Core Functionality

- Write Python workflows with YAML configuration plugins that execute via a fast Rust orchestration framework.
- Workflows are exposed as REST APIs allowing the user to submit inference requests, check the status, and upload results to object storage.
- Deploy inference pipelines using Lepton.AI or other frameworks.
- Support multi-GPU, distributed inference, and multi-instance deployment options.
- Provide Rust Zarr I/O optimized backend options to speed up inference.

## Quick Start

Build the service container:

```bash
docker build -f Dockerfile.Earth2Studio.runtime-base -t physicsnemo-serve-runtime-base .
docker build -f Dockerfile.Earth2Studio.scicomp-rust-slim -t physicsnemo-serve .
```

Then choose your path:

- **Use a deployed service** — see [onboarding.md](docs/onboarding.md) for REST API examples (list workflows, submit runs, fetch results)
- **Author a new plugin** — see [plugin-authoring-guide.md](docs/plugin-authoring-guide.md) for layout, hooks, and local validation
- **Understand the full service** — see [inference-service-user-guide.md](docs/inference-service-user-guide.md) for architecture, configuration, and deployment (including Lepton.AI)

## Core Pieces

- `crates/inference_server`
  - REST API for workflow discovery, schema/readiness inspection, run submission, status, and results
- `crates/worker-runtime`
  - Rust role workers for `prepare`, `prefetch`, `batch`, `fanout`, `schedule`, `collect`, `postprocess`, and `results`
- `crates/e2s_zarr_io`
  - Rust-backed Zarr IO backend for Earth2Studio
- `crates/scicomp-rq`
  - Redis Streams–based distributed task queue for scientific computing pipelines
- `scripts/inference_worker.py`
  - Python execute worker for plugin hooks
- `scripts/plugin_dev.py`
  - scaffold, validate, local check, and local stack tooling for plugins

## Contributors

This project is currently not accepting contributions.

## License

PhysicsNeMo-Serve is provided under the Apache License 2.0, refer to the [LICENSE file](LICENSE) for full license text.

