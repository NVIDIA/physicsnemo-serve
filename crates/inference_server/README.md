# Inference Server

Rust REST API for plugin discovery, request validation, run submission, status lookup, and result serving.

Start with:

- [../../docs/onboarding.md](../../docs/onboarding.md)
- [../../docs/inference-service-user-guide.md](../../docs/inference-service-user-guide.md)
- [../../docs/plugin-authoring-guide.md](../../docs/plugin-authoring-guide.md)

## Features

- discovers manifest-driven plugins from `PLUGIN_DIR`
- exposes workflow list, schema, readiness, run, status, and results endpoints
- validates JSON or multipart requests against plugin contracts
- persists run state and result payloads in Redis

## Quick Start

```bash
cargo run -p inference_server
```

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/v1/infer/workflows
```

## Docker

The repository still keeps `Dockerfile.Earth2Studio.scicomp-rust-slim` for deployments that need it.
