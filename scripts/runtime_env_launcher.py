#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def resolve_namespace() -> str:
    return os.environ.get("POD_NAMESPACE") or os.environ.get("NAMESPACE") or "default"


def resolve_pod_name() -> str:
    return os.environ.get("POD_NAME") or "unknown"


def resolve_stream_prefix() -> str:
    return (
        os.environ.get("STREAM_PREFIX") or os.environ.get("REDIS_STREAM_PREFIX") or ""
    )


def load_runtime_env_registry(config_path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    runtime_envs = config.get("python_runtime_envs")
    if not isinstance(runtime_envs, dict):
        raise ValueError(
            "runtime config must define a top-level python_runtime_envs object"
        )
    return runtime_envs


def detect_gpus() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    gpus: list[dict[str, Any]] = []
    for device_index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_index)
        gpus.append(
            {
                "device_index": device_index,
                "device_name": props.name,
                "device_uuid": (
                    f"GPU-{props.uuid}"
                    if hasattr(props, "uuid")
                    else f"GPU-{device_index}"
                ),
                "memory_mb": props.total_memory // (1024 * 1024),
            }
        )
    return gpus


def _normalize_tags(raw_tags: Any) -> list[str]:
    if not raw_tags:
        return []
    if not isinstance(raw_tags, list):
        raise ValueError("launch.tags must be a list of strings")
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _int_value(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _build_cpu_stream_name(executor_class: str, stream_prefix: str) -> str:
    return f"{stream_prefix}execute.{executor_class}"


def _build_gpu_stream_name(
    executor_class: str,
    namespace: str,
    pod_name: str,
    device_index: int,
    stream_prefix: str,
) -> str:
    return f"{stream_prefix}execute.{executor_class}:gpu:{namespace}:{pod_name}:{device_index}"


def _safe_path_segment(value: object) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip(".-")
    return segment or "unknown"


def _resolve_earth2studio_data_cache_root(extra_env: dict[str, Any]) -> str:
    for name in (
        "PHYSICSNEMO_SERVE_EARTH2STUDIO_DATA_CACHE_ROOT",
        "EARTH2STUDIO_DATA_CACHE",
        "EARTH2STUDIO_CACHE",
    ):
        value = str(extra_env.get(name) or os.environ.get(name) or "").strip()
        if value:
            return value
    return os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")


def _build_earth2studio_data_cache_path(
    root: str,
    *,
    namespace: str,
    pod_name: str,
    executor_class: str,
    device_kind: str,
    device_index: int | str,
    worker_index: int,
) -> str:
    return str(
        Path(root).expanduser()
        / "worker-data-cache"
        / _safe_path_segment(namespace)
        / _safe_path_segment(pod_name)
        / _safe_path_segment(executor_class)
        / f"{_safe_path_segment(device_kind)}-{_safe_path_segment(device_index)}"
        / f"worker-{worker_index}"
    )


def build_worker_launch_specs(
    runtime_envs: dict[str, Any],
    *,
    namespace: str,
    pod_name: str,
    enabled_executor_classes: set[str] | None,
    gpu_inventory: list[dict[str, Any]] | None,
    worker_script: str,
    stream_prefix: str = "",
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    gpu_inventory = gpu_inventory or []

    for executor_class in sorted(runtime_envs):
        if enabled_executor_classes and executor_class not in enabled_executor_classes:
            continue

        runtime_env = runtime_envs.get(executor_class) or {}
        launch = runtime_env.get("launch") or {}
        if not isinstance(launch, dict) or not launch.get("enabled"):
            continue

        python_executable = str(runtime_env.get("python_executable") or "").strip()
        if not python_executable:
            raise ValueError(
                f"python_runtime_envs['{executor_class}'].python_executable must be non-empty"
            )

        extra_env = dict(runtime_env.get("env") or {})
        data_cache_root = _resolve_earth2studio_data_cache_root(extra_env)
        device_kind = str(launch.get("device_kind") or "cpu").strip().lower()
        tags = _normalize_tags(launch.get("tags"))

        if device_kind == "cpu":
            if launch.get("workers_per_device") is not None:
                raise ValueError(
                    f"python_runtime_envs['{executor_class}'].launch.workers_per_device is GPU-only; use replicas for cpu launch configs"
                )
            replicas = _int_value(launch.get("replicas"), 1)
            memory_mb = _int_value(launch.get("memory_mb"), 4096)
            stream_name = _build_cpu_stream_name(executor_class, stream_prefix)
            for worker_index in range(replicas):
                specs.append(
                    {
                        "executor_class": executor_class,
                        "python_executable": python_executable,
                        "argv": [python_executable, worker_script],
                        "env": {
                            **extra_env,
                            "GPU_STREAM_NAME": stream_name,
                            "GPU_EXECUTOR_CLASS": executor_class,
                            "GPU_DEVICE_KIND": "cpu",
                            "GPU_MEMORY_MB": str(memory_mb),
                            "GPU_DEVICE_INDEX": "0",
                            "GPU_DEVICE_NAME": "cpu",
                            "GPU_DEVICE_UUID": f"{executor_class}-cpu",
                            "GPU_WORKER_INDEX": str(worker_index),
                            "GPU_TAGS": ",".join(tags),
                            "EARTH2STUDIO_DATA_CACHE": (
                                _build_earth2studio_data_cache_path(
                                    data_cache_root,
                                    namespace=namespace,
                                    pod_name=pod_name,
                                    executor_class=executor_class,
                                    device_kind="cpu",
                                    device_index=0,
                                    worker_index=worker_index,
                                )
                            ),
                        },
                    }
                )
            continue

        if device_kind != "gpu":
            raise ValueError(
                f"python_runtime_envs['{executor_class}'].launch.device_kind must be cpu or gpu"
            )

        workers_per_device = _int_value(
            launch.get("workers_per_device"),
            _int_value(os.environ.get("WORKERS_PER_GPU"), 1),
        )
        for gpu in gpu_inventory:
            stream_name = _build_gpu_stream_name(
                executor_class,
                namespace,
                pod_name,
                int(gpu["device_index"]),
                stream_prefix,
            )
            for worker_index in range(workers_per_device):
                specs.append(
                    {
                        "executor_class": executor_class,
                        "python_executable": python_executable,
                        "argv": [python_executable, worker_script],
                        "env": {
                            **extra_env,
                            "PYTORCH_CUDA_ALLOC_CONF": extra_env.get(
                                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                            ),
                            "CUDA_VISIBLE_DEVICES": str(gpu["device_index"]),
                            "GPU_STREAM_NAME": stream_name,
                            "GPU_EXECUTOR_CLASS": executor_class,
                            "GPU_DEVICE_KIND": "gpu",
                            "GPU_MEMORY_MB": str(gpu["memory_mb"]),
                            "GPU_DEVICE_INDEX": str(gpu["device_index"]),
                            "GPU_DEVICE_NAME": str(gpu["device_name"]),
                            "GPU_DEVICE_UUID": str(gpu["device_uuid"]),
                            "GPU_WORKER_INDEX": str(worker_index),
                            "GPU_TAGS": ",".join(tags),
                            "EARTH2STUDIO_DATA_CACHE": (
                                _build_earth2studio_data_cache_path(
                                    data_cache_root,
                                    namespace=namespace,
                                    pod_name=pod_name,
                                    executor_class=executor_class,
                                    device_kind="gpu",
                                    device_index=gpu["device_index"],
                                    worker_index=worker_index,
                                )
                            ),
                        },
                    }
                )

    return specs


def _decode_redis_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _matching_registry_fields(
    registry: dict[object, object], stream_name: str, worker_index: int | None
) -> list[str]:
    fields: list[str] = []
    for raw_field, raw_metadata in registry.items():
        field = _decode_redis_value(raw_field)
        metadata_str = _decode_redis_value(raw_metadata)
        try:
            metadata = json.loads(metadata_str)
        except Exception:
            continue
        if metadata.get("stream") != stream_name:
            continue
        if worker_index is not None:
            try:
                metadata_worker_index = int(metadata.get("worker_index", -1))
            except (TypeError, ValueError):
                continue
            if metadata_worker_index != worker_index:
                continue
        fields.append(field)
    return fields


def cleanup_registration(stream_name: str, worker_index: int | None = None) -> None:
    try:
        import redis
    except ImportError:
        return

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(redis_url)
        registry = client.hgetall("gpu:registry")
        fields = _matching_registry_fields(registry, stream_name, worker_index)
        if fields:
            client.hdel("gpu:registry", *fields)
            return
        if worker_index is None:
            client.hdel("gpu:registry", stream_name)
        else:
            client.hdel("gpu:registry", f"{stream_name}:worker:{worker_index}")
    except Exception:
        return


def _resolve_enabled_executor_classes() -> set[str] | None:
    raw = os.environ.get("PHYSICSNEMO_SERVE_EXECUTOR_CLASSES") or os.environ.get(
        "EXECUTOR_CLASSES"
    )
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _load_launch_specs() -> list[dict[str, Any]]:
    config_path = os.environ.get(
        "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"
    ) or os.environ.get("WORKER_RUNTIME_CONFIG")
    if not config_path:
        raise ValueError(
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG or WORKER_RUNTIME_CONFIG is required"
        )

    runtime_envs = load_runtime_env_registry(config_path)
    enabled = _resolve_enabled_executor_classes()
    worker_script = os.environ.get("WORKER_SCRIPT", "/app/scripts/inference_worker.py")

    gpu_needed = any(
        isinstance(runtime_env.get("launch"), dict)
        and runtime_env.get("launch", {}).get("enabled")
        and str(runtime_env.get("launch", {}).get("device_kind") or "cpu")
        .strip()
        .lower()
        == "gpu"
        and (enabled is None or executor_class in enabled)
        for executor_class, runtime_env in runtime_envs.items()
    )
    gpu_inventory = detect_gpus() if gpu_needed else []

    return build_worker_launch_specs(
        runtime_envs,
        namespace=resolve_namespace(),
        pod_name=resolve_pod_name(),
        enabled_executor_classes=enabled,
        gpu_inventory=gpu_inventory,
        worker_script=worker_script,
        stream_prefix=resolve_stream_prefix(),
    )


def spawn_worker(spec: dict[str, Any]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in spec["env"].items()})
    if data_cache := env.get("EARTH2STUDIO_DATA_CACHE"):
        Path(data_cache).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        spec["argv"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> None:
    specs = _load_launch_specs()
    if not specs:
        print("[runtime-env-launcher] ERROR: no enabled runtime env launch specs found")
        sys.exit(1)

    workers: list[tuple[dict[str, Any], subprocess.Popen]] = []
    for spec in specs:
        print(
            "[runtime-env-launcher] Spawning",
            spec["executor_class"],
            spec["env"]["GPU_DEVICE_KIND"],
            spec["env"]["GPU_STREAM_NAME"],
            f"worker_index={spec['env']['GPU_WORKER_INDEX']}",
            f"earth2studio_data_cache={spec['env'].get('EARTH2STUDIO_DATA_CACHE', '')}",
        )
        workers.append((spec, spawn_worker(spec)))

    shutdown_requested = False

    def handle_signal(signum: int, _frame: object | None) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True
        print(
            f"[runtime-env-launcher] Received signal {signum}, shutting down workers..."
        )
        for _spec, proc in workers:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not shutdown_requested:
        for index, (spec, proc) in enumerate(list(workers)):
            exit_code = proc.poll()
            if exit_code is None:
                continue

            stream_name = spec["env"]["GPU_STREAM_NAME"]
            worker_index = int(spec["env"]["GPU_WORKER_INDEX"])
            cleanup_registration(stream_name, worker_index)

            if shutdown_requested:
                print(
                    "[runtime-env-launcher] Worker exited during shutdown:",
                    spec["executor_class"],
                    stream_name,
                )
                continue

            print(
                "[runtime-env-launcher] Worker exited, restarting:",
                spec["executor_class"],
                stream_name,
                f"exit_code={exit_code}",
            )
            time.sleep(2)
            workers[index] = (spec, spawn_worker(spec))

        time.sleep(1)

    for spec, proc in workers:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        cleanup_registration(
            spec["env"]["GPU_STREAM_NAME"],
            int(spec["env"]["GPU_WORKER_INDEX"]),
        )


if __name__ == "__main__":
    main()
