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
- `crates/physicsnemo-serve-cmd`
  - Optional one-shot manifest plugin runner using a packaged or external Python runtime
- `scripts/inference_worker.py`
  - Python execute worker for plugin hooks
- `scripts/plugin_dev.py`
  - scaffold, validate, local check, and local stack tooling for plugins

## Standalone direct inference

The `physicsnemo-serve` command runs one compatible external manifest plugin
without Redis, the REST inference server, worker processes, or a scheduler. Its
packaged form appends a locked CPython runtime to the executable and extracts
that runtime into a content-addressed cache on first use. A thin binary can
instead use a customer-managed, plugin-specific Python environment selected
with `--runtime-dir`.

```bash
./dist/physicsnemo-serve infer \
  --runtime-dir /opt/physicsnemo-runtimes/customer-plugin \
  --plugin /path/to/plugin \
  --request request.json \
  --output-dir outputs \
  --device 0
```

The direct runner supports JSON plugins using `simple`,
`prefetch`/`default`, `postprocess`, and single-item `batch` profiles.
Fanout/collect, publication, multipart ingress, and custom framework stages are
rejected explicitly. See
[`crates/physicsnemo-serve-cmd/README.md`](crates/physicsnemo-serve-cmd/README.md)
for runtime assembly and packaging instructions.

The existing `inference_server`, distributed `worker-runtime`, Redis
configuration, Dockerfiles, and Python inference worker remain supported for
the full service deployment.

## Contributors

This project is currently not accepting contributions.

## License

PhysicsNeMo-Serve is provided under the Apache License 2.0, refer to the [LICENSE file](LICENSE) for full license text.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. In particular, note that NVIDIA does not release docker images or host this service. If you choose to do either, also inspect the default Dockerfile provided (and any changes you make) and ensure you comply with any additional licensing terms.

## Docs

- [onboarding.md](docs/onboarding.md)
- [plugin-authoring-guide.md](docs/plugin-authoring-guide.md)
- [inference-service-user-guide.md](docs/inference-service-user-guide.md)

## Cloud Output Publication

PhysicsNeMo-Serve can publish a workflow's primary output artifact to S3-compatible
object storage or Azure Blob Storage. Publication is disabled by default and is
enabled from the runtime config referenced by
`PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG`. The container entrypoint defaults that
variable from `WORKER_RUNTIME_CONFIG`; a caller-only `WORKER_RUNTIME_CONFIG`
override also replaces the image's baked-in default server config path. An
explicit non-default `PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG` remains the
server-specific override.

S3 or S3-compatible storage:

```json
{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "s3",
      "bucket": "forecast-bucket",
      "prefix": "outputs",
      "region": "us-east-1",
      "endpoint": "https://s3.us-east-1.amazonaws.com"
    }
  }
}
```

Azure Blob Storage:

```json
{
  "output_publication": {
    "enabled": true,
    "storage": {
      "type": "azure",
      "container": "forecast-results",
      "prefix": "outputs",
      "endpoint": "https://account.blob.core.windows.net"
    }
  }
}
```

Keep credentials out of config files. Provide them through deployment
environment variables or secret managers:

- S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional
  `AWS_SESSION_TOKEN`, and optional `AWS_REGION`/`AWS_DEFAULT_REGION`.
- S3-compatible endpoints may also use `S3_ENDPOINT_URL` if `endpoint` is not
  set in config.
- Azure: `AZURE_STORAGE_ACCOUNT` or `AZURE_STORAGE_ACCOUNT_NAME`, plus
  `AZURE_STORAGE_ACCOUNT_KEY` or `AZURE_STORAGE_ACCESS_KEY`. SAS tokens and
  default Azure credentials are also supported where available.

Upload performance is configured separately on the `publish` role:

```json
{
  "roles": {
    "publish": {
      "config": {
        "max_concurrent_files": 96,
        "multipart_threshold_bytes": 67108864,
        "multipart_part_size_bytes": 16777216,
        "multipart_max_concurrency": 4,
        "client_options": {
          "timeout_secs": 300,
          "connect_timeout_secs": 10,
          "pool_max_idle_per_host": 192
        },
        "retry": {
          "max_retries": 10,
          "timeout_secs": 300
        }
      }
    }
  }
}
```

`max_concurrent_files` controls parallel uploads for directory artifacts such as
Zarr stores. The multipart settings apply to large single-file artifacts such as
NetCDF or HDF5.
