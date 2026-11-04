"""Compose existing Python server launch configuration without model execution.

This file is also copied into Python release images as a standalone script.
Only prepare imports the existing image's manifest/configuration helpers; start
uses the standard library and executes the existing server entrypoint.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys


def configure(manifest, base, *, python, legacy):
    runtime = manifest["runtime"]
    if runtime.get("kind") != "python":
        raise ValueError("Python release requires a Python plugin")
    defaults = manifest.get("resources", {}).get("defaults", {})
    kind = defaults.get("device_kind") or (
        "gpu" if defaults.get("gpus_required", 0) else "cpu"
    )
    if kind not in ("cpu", "gpu"):
        raise ValueError("Python release requires declared CPU or GPU resources")
    profile = {
        "executor_class": runtime["executor_class"],
        "device_kind": kind,
        "memory_mb": defaults.get("memory_mb", 4096),
        "tags": defaults.get("tags", []),
    }
    generated = legacy._build_run_local_python_runtime_envs(manifest, [profile])
    existing = base.get("python_runtime_envs", {})
    environments = {}
    for executor, selection in generated.items():
        value = deepcopy(existing.get(executor, selection))
        if executor not in existing:
            value["python_executable"] = python
        if executor == profile["executor_class"]:
            if value.get("launch"):
                if value["launch"].get("device_kind") != kind:
                    raise ValueError(
                        f"declared device kind differs from the base runtime for {executor}"
                    )
                value["launch"]["enabled"] = True
                value["launch"]["tags"] = list(
                    dict.fromkeys([*value["launch"].get("tags", []), *profile["tags"]])
                )
            else:
                value["launch"] = {**selection["launch"], "enabled": True}
            if kind == "cpu":
                value["launch"].pop("workers_per_device", None)
            else:
                value["launch"].pop("replicas", None)
        environments[executor] = value
    stages = manifest["pipeline"]["stages"]
    config = legacy._build_runtime_config(
        stages,
        python_runtime_envs=environments,
        include_publish=bool(base.get("output_publication", {}).get("enabled")),
        output_publication=base.get("output_publication"),
        publish_role_config=base.get("roles", {}).get("publish", {}).get("config"),
    )
    for key in ("stream_prefix", "max_retries", "shared_dlq_stream"):
        if key in base:
            config[key] = deepcopy(base[key])
    for role, selected in config["roles"].items():
        if "config" in base.get("roles", {}).get(role, {}):
            selected["config"] = deepcopy(base["roles"][role]["config"])
    prefix = config.get("stream_prefix", "")
    config_path = "/opt/pnms/worker_runtime_config.json"
    return {
        "workflow_id": manifest["metadata"]["id"],
        "runtime_config": config,
        "first_queue": stages[0]["queue"],
        "environment": {
            "PLUGIN_DIR": "/opt/pnms/plugin",
            "WORKFLOW_DIR": "/opt/pnms/plugin",
            "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": manifest["metadata"]["id"],
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG": config_path,
            "WORKER_RUNTIME_CONFIG": config_path,
            "QUEUE_CONFIG": config_path,
            "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE": python,
            "PHYSICSNEMO_SERVE_EXECUTOR_CLASSES": ",".join(
                sorted(
                    name
                    for name, value in environments.items()
                    if value.get("launch", {}).get("enabled")
                )
            ),
            "REDIS_STREAM_PREFIX": prefix,
            "INFERENCE_STREAM": prefix + stages[0]["queue"],
            "PREFETCH_STREAM": prefix + "prefetch",
            "WORKERS": ",".join(dict.fromkeys(["server", *config["roles"], "execute"])),
        },
    }


def prepare(
    *,
    root=Path("/opt/pnms"),
    scripts=Path("/app/scripts"),
    base_config=None,
    python=None,
):
    root = Path(root)
    sys.path.insert(0, str(scripts))
    import plugin_dev
    from plugin_runtime import load_plugin_manifest

    plugin = root / "plugin"
    manifest = load_plugin_manifest(plugin / "plugin.yaml")
    config_path = base_config or os.environ.get(
        "WORKER_RUNTIME_CONFIG", "/app/scripts/worker_runtime_config.json"
    )
    base = json.loads(Path(config_path).read_text())
    python = python or os.environ.get(
        "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", sys.executable
    )
    result = configure(manifest, base, python=python, legacy=plugin_dev)
    (root / "worker_runtime_config.json").write_text(
        json.dumps(result["runtime_config"], indent=2) + "\n"
    )
    (root / "python-launch.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def start():
    selected = json.loads(Path("/opt/pnms/python-launch.json").read_text())
    environment = {**os.environ, **selected["environment"]}
    os.execvpe(
        "/app/scripts/entrypoint.sh", ["/app/scripts/entrypoint.sh"], environment
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["start"]:
        start()
    else:
        raise SystemExit("usage: python_release.py prepare|start")
