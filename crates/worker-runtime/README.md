# worker-runtime

Config-driven Redis Streams worker runtime for manifest-driven plugins.

## Roles

- `prefetch`: reads `prefetch`, generates download plans from workflow metadata, materializes inputs, and hands off to `schedule`.
- `scheduler`: reads `schedule` and `release`, discovers GPU streams dynamically, and routes jobs to GPU-specific streams.
- `results`: terminal consumer on `results`; normalizes payloads and persists run/result records to Redis.

Pipeline shape:

```text
prefetch -> schedule -> execute.* -> results
```

## Runtime config

Use `examples/runtime_config.json` as the baseline config.

Top-level fields:
- `stream_prefix`: prefix for physical Redis stream keys.
- `streams`: allowed logical stream inventory.
- `max_retries`: retry count before moving failed messages to DLQ.
- `shared_dlq_stream`: DLQ stream name.
- `roles`: per-role inputs/outputs and optional role-specific `config`.
- `python_runtime_envs`: Python interpreter registry keyed by `executor_class`. This is used by
  `prepare`/`postprocess` hook execution and by the runtime-env execute launcher.

Example:

```json
{
  "python_runtime_envs": {
    "python.test": {
      "python_executable": "python3",
      "env": {},
      "launch": {
        "enabled": true,
        "device_kind": "cpu",
        "replicas": 1,
        "tags": ["demo", "cpu"]
      }
    }
  }
}
```

Use `launch.replicas` for CPU worker counts and `launch.workers_per_device` for GPU pools.

## Required env vars

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis connection URL (used by `QueueManager::from_env()`) |
| `WORKER_PIPELINE_CONFIG` | Path to runtime config JSON (unless `--config-path` is passed) |
| `WORKER_ROLE` | Role to run: `prefetch`, `scheduler`, or `results` (unless `--role` is passed) |

Common optional vars:
- `HEALTH_PORT`: enables TCP health endpoint; scheduler also serves Prometheus
  metrics at `/metrics` on this port.
- `SCHEDULER_DISCOVERY_JSON`: scheduler test/override worker inventory.
- `E2S_DOWNLOAD_CONCURRENCY`, `E2S_DOWNLOAD_TIMEOUT_SECS`, `E2S_EXT_CACHE`: prefetch tuning.

Pipelines that include `schedule` must provide a usable `resource_profile` before the
payload reaches the scheduler. The normal source is the plugin manifest defaults, optionally
overridden by `prepare()`.

## Run

From `platform/inference_rust`:

```bash
cargo run -p worker-runtime -- --role prefetch --config-path crates/worker-runtime/examples/runtime_config.json
```

Environment-driven startup:

```bash
export REDIS_URL="redis://127.0.0.1:6379"
export WORKER_PIPELINE_CONFIG="crates/worker-runtime/examples/runtime_config.json"
export WORKER_ROLE="scheduler"
cargo run -p worker-runtime
```

Run one process per role in production.

For execute workers, start one runtime-env launcher process per deployment and let it spawn
`inference_worker.py` instances from `python_runtime_envs.*.launch`:

```bash
export WORKER_RUNTIME_CONFIG="crates/worker-runtime/examples/runtime_config.json"
export PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG="$WORKER_RUNTIME_CONFIG"
python scripts/runtime_env_launcher.py
```

## Test

```bash
cd platform/inference_rust
cargo test -p worker-runtime
```
