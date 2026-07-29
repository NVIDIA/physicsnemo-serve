# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_runtime_env_launcher_module():
    script_path = repo_root() / "scripts" / "runtime_env_launcher.py"
    spec = importlib.util.spec_from_file_location("runtime_env_launcher", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_worker_launch_specs_supports_cpu_and_gpu_runtime_envs(
    monkeypatch,
) -> None:
    module = load_runtime_env_launcher_module()
    monkeypatch.setenv("PHYSICSNEMO_SERVE_EARTH2STUDIO_DATA_CACHE_ROOT", "/cache/root")

    runtime_envs = {
        "python.cpu.demo": {
            "python_executable": "/envs/cpu/bin/python",
            "env": {"PYTHONPATH": "/opt/demo-cpu"},
            "launch": {
                "enabled": True,
                "device_kind": "cpu",
                "replicas": 2,
                "memory_mb": 8192,
                "tags": ["demo", "cpu"],
            },
        },
        "python.gpu.demo": {
            "python_executable": "/envs/gpu/bin/python",
            "env": {"LD_LIBRARY_PATH": "/opt/cuda"},
            "launch": {
                "enabled": True,
                "device_kind": "gpu",
                "workers_per_device": 2,
                "tags": ["demo", "gpu"],
            },
        },
        "python.cpu.prepare": {
            "python_executable": "/envs/prepare/bin/python",
            "env": {},
        },
    }

    gpu_inventory = [
        {
            "device_index": 0,
            "device_name": "RTX 6000",
            "device_uuid": "GPU-0000",
            "memory_mb": 24576,
        }
    ]

    specs = module.build_worker_launch_specs(
        runtime_envs,
        namespace="ns-a",
        pod_name="pod-b",
        enabled_executor_classes=None,
        gpu_inventory=gpu_inventory,
        worker_script="/repo/scripts/inference_worker.py",
    )

    assert len(specs) == 4

    cpu_specs = [spec for spec in specs if spec["executor_class"] == "python.cpu.demo"]
    assert len(cpu_specs) == 2
    assert {spec["python_executable"] for spec in cpu_specs} == {"/envs/cpu/bin/python"}
    assert {spec["env"]["GPU_STREAM_NAME"] for spec in cpu_specs} == {
        "execute.python.cpu.demo"
    }
    assert {spec["env"]["GPU_WORKER_INDEX"] for spec in cpu_specs} == {"0", "1"}
    assert {spec["env"]["GPU_DEVICE_KIND"] for spec in cpu_specs} == {"cpu"}
    assert {spec["env"]["GPU_MEMORY_MB"] for spec in cpu_specs} == {"8192"}
    assert {spec["env"]["GPU_TAGS"] for spec in cpu_specs} == {"demo,cpu"}
    assert {spec["env"]["PYTHONPATH"] for spec in cpu_specs} == {"/opt/demo-cpu"}
    assert {spec["env"]["EARTH2STUDIO_DATA_CACHE"] for spec in cpu_specs} == {
        "/cache/root/worker-data-cache/ns-a/pod-b/python.cpu.demo/cpu-0/worker-0",
        "/cache/root/worker-data-cache/ns-a/pod-b/python.cpu.demo/cpu-0/worker-1",
    }

    gpu_specs = [spec for spec in specs if spec["executor_class"] == "python.gpu.demo"]
    assert len(gpu_specs) == 2
    assert {spec["python_executable"] for spec in gpu_specs} == {"/envs/gpu/bin/python"}
    assert {spec["env"]["GPU_STREAM_NAME"] for spec in gpu_specs} == {
        "execute.python.gpu.demo:gpu:ns-a:pod-b:0"
    }
    assert {spec["env"]["GPU_WORKER_INDEX"] for spec in gpu_specs} == {"0", "1"}
    assert {spec["env"]["GPU_DEVICE_KIND"] for spec in gpu_specs} == {"gpu"}
    assert {spec["env"]["CUDA_VISIBLE_DEVICES"] for spec in gpu_specs} == {"0"}
    assert {spec["env"]["PYTORCH_CUDA_ALLOC_CONF"] for spec in gpu_specs} == {
        "expandable_segments:True"
    }
    assert {spec["env"]["GPU_MEMORY_MB"] for spec in gpu_specs} == {"24576"}
    assert {spec["env"]["GPU_TAGS"] for spec in gpu_specs} == {"demo,gpu"}
    assert {spec["env"]["LD_LIBRARY_PATH"] for spec in gpu_specs} == {"/opt/cuda"}
    assert {spec["env"]["EARTH2STUDIO_DATA_CACHE"] for spec in gpu_specs} == {
        "/cache/root/worker-data-cache/ns-a/pod-b/python.gpu.demo/gpu-0/worker-0",
        "/cache/root/worker-data-cache/ns-a/pod-b/python.gpu.demo/gpu-0/worker-1",
    }
    assert all("E2S_EXT_CACHE" not in spec["env"] for spec in specs)


def test_worker_data_cache_uses_earth2studio_data_cache_without_changing_ext_cache(
    monkeypatch,
) -> None:
    module = load_runtime_env_launcher_module()
    monkeypatch.delenv("PHYSICSNEMO_SERVE_EARTH2STUDIO_DATA_CACHE_ROOT", raising=False)
    monkeypatch.setenv("EARTH2STUDIO_CACHE", "/general/cache")
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", "/data/cache")

    specs = module.build_worker_launch_specs(
        {
            "earth2-gpu": {
                "python_executable": "/envs/gpu/bin/python",
                "env": {"E2S_EXT_CACHE": "/shared/e2s-ext"},
                "launch": {
                    "enabled": True,
                    "device_kind": "gpu",
                    "workers_per_device": 1,
                },
            },
        },
        namespace="ns",
        pod_name="pod",
        enabled_executor_classes=None,
        gpu_inventory=[
            {
                "device_index": 0,
                "device_name": "H100",
                "device_uuid": "GPU-0000",
                "memory_mb": 81920,
            },
            {
                "device_index": 1,
                "device_name": "H100",
                "device_uuid": "GPU-1111",
                "memory_mb": 81920,
            },
        ],
        worker_script="/repo/scripts/inference_worker.py",
    )

    assert {spec["env"]["EARTH2STUDIO_DATA_CACHE"] for spec in specs} == {
        "/data/cache/worker-data-cache/ns/pod/earth2-gpu/gpu-0/worker-0",
        "/data/cache/worker-data-cache/ns/pod/earth2-gpu/gpu-1/worker-0",
    }
    assert {spec["env"]["E2S_EXT_CACHE"] for spec in specs} == {"/shared/e2s-ext"}


def test_spawn_worker_creates_earth2studio_data_cache(tmp_path, monkeypatch) -> None:
    module = load_runtime_env_launcher_module()
    data_cache = tmp_path / "worker-cache"
    popen_calls = []

    class FakePopen:
        def __init__(self, argv, *, env, stdout, stderr):
            popen_calls.append(
                {
                    "argv": argv,
                    "env": env,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)

    process = module.spawn_worker(
        {
            "argv": ["/env/bin/python", "/repo/scripts/inference_worker.py"],
            "env": {"EARTH2STUDIO_DATA_CACHE": str(data_cache)},
        }
    )

    assert isinstance(process, FakePopen)
    assert data_cache.is_dir()
    assert popen_calls[0]["env"]["EARTH2STUDIO_DATA_CACHE"] == str(data_cache)


def test_build_worker_launch_specs_respects_executor_class_filter() -> None:
    module = load_runtime_env_launcher_module()

    runtime_envs = {
        "python.cpu.demo": {
            "python_executable": "/envs/cpu/bin/python",
            "env": {},
            "launch": {
                "enabled": True,
                "device_kind": "cpu",
                "replicas": 1,
                "memory_mb": 4096,
            },
        },
        "python.gpu.demo": {
            "python_executable": "/envs/gpu/bin/python",
            "env": {},
            "launch": {
                "enabled": True,
                "device_kind": "gpu",
                "workers_per_device": 1,
            },
        },
    }

    specs = module.build_worker_launch_specs(
        runtime_envs,
        namespace="default",
        pod_name="pod-0",
        enabled_executor_classes={"python.cpu.demo"},
        gpu_inventory=[
            {
                "device_index": 0,
                "device_name": "A100",
                "device_uuid": "GPU-AAAA",
                "memory_mb": 81920,
            }
        ],
        worker_script="/repo/scripts/inference_worker.py",
    )

    assert len(specs) == 1
    assert specs[0]["executor_class"] == "python.cpu.demo"
    assert specs[0]["env"]["GPU_DEVICE_KIND"] == "cpu"


def test_explicit_executor_class_filter_opts_into_disabled_runtime() -> None:
    module = load_runtime_env_launcher_module()
    runtime_envs = {
        "earth2-gpu": {
            "python_executable": "/envs/earth2/bin/python",
            "launch": {"enabled": True, "device_kind": "gpu"},
        },
        "physicsnemo-cfd-gpu": {
            "python_executable": "/envs/cfd/bin/python",
            "launch": {"enabled": False, "device_kind": "gpu"},
        },
    }
    gpu = {
        "device_index": 0,
        "device_name": "H100",
        "device_uuid": "GPU-0000",
        "memory_mb": 81920,
    }

    default_specs = module.build_worker_launch_specs(
        runtime_envs,
        namespace="default",
        pod_name="pod-0",
        enabled_executor_classes=None,
        gpu_inventory=[gpu],
        worker_script="/repo/scripts/inference_worker.py",
    )
    assert [spec["executor_class"] for spec in default_specs] == ["earth2-gpu"]

    selected_specs = module.build_worker_launch_specs(
        runtime_envs,
        namespace="default",
        pod_name="pod-0",
        enabled_executor_classes={"physicsnemo-cfd-gpu"},
        gpu_inventory=[gpu],
        worker_script="/repo/scripts/inference_worker.py",
    )
    assert [spec["executor_class"] for spec in selected_specs] == [
        "physicsnemo-cfd-gpu"
    ]


def test_launcher_rejects_two_executor_classes_for_same_gpu() -> None:
    module = load_runtime_env_launcher_module()
    specs = [
        {
            "executor_class": "earth2-gpu",
            "env": {
                "GPU_DEVICE_KIND": "gpu",
                "GPU_DEVICE_UUID": "GPU-0000",
            },
        },
        {
            "executor_class": "physicsnemo-cfd-gpu",
            "env": {
                "GPU_DEVICE_KIND": "gpu",
                "GPU_DEVICE_UUID": "GPU-0000",
            },
        },
    ]

    try:
        module._validate_one_gpu_executor_class_per_device(specs)
    except ValueError as exc:
        assert "multiple GPU executor classes" in str(exc)
    else:
        raise AssertionError("co-resident GPU executor classes must fail closed")


def test_load_launch_specs_detects_gpu_for_explicitly_selected_disabled_runtime(
    monkeypatch, tmp_path
) -> None:
    module = load_runtime_env_launcher_module()
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "python_runtime_envs": {
                    "physicsnemo-cfd-gpu": {
                        "python_executable": "/envs/cfd/bin/python",
                        "launch": {"enabled": False, "device_kind": "gpu"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detected = []

    def fake_detect_gpus():
        detected.append(True)
        return [
            {
                "device_index": 0,
                "device_name": "H100",
                "device_uuid": "GPU-0000",
                "memory_mb": 81920,
            }
        ]

    monkeypatch.setattr(module, "detect_gpus", fake_detect_gpus)
    monkeypatch.setenv("PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG", str(config_path))
    monkeypatch.setenv("PHYSICSNEMO_SERVE_EXECUTOR_CLASSES", "physicsnemo-cfd-gpu")

    specs = module._load_launch_specs()

    assert detected == [True]
    assert len(specs) == 1
    assert specs[0]["executor_class"] == "physicsnemo-cfd-gpu"
    assert specs[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_build_worker_launch_specs_rejects_workers_per_device_for_cpu() -> None:
    module = load_runtime_env_launcher_module()

    runtime_envs = {
        "python.cpu.demo": {
            "python_executable": "/envs/cpu/bin/python",
            "env": {},
            "launch": {
                "enabled": True,
                "device_kind": "cpu",
                "workers_per_device": 3,
                "memory_mb": 6144,
            },
        },
    }

    try:
        module.build_worker_launch_specs(
            runtime_envs,
            namespace="default",
            pod_name="pod-0",
            enabled_executor_classes=None,
            gpu_inventory=[],
            worker_script="/repo/scripts/inference_worker.py",
        )
    except ValueError as exc:
        assert "use replicas for cpu launch configs" in str(exc)
    else:
        raise AssertionError("CPU launch configs must reject workers_per_device")


def test_load_runtime_env_registry_reads_python_runtime_envs() -> None:
    module = load_runtime_env_launcher_module()
    config_path = repo_root() / "target" / "test-runtime-env-launcher-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "python_runtime_envs": {
                    "python.cpu.demo": {
                        "python_executable": "/envs/cpu/bin/python",
                        "env": {"PYTHONPATH": "/opt/demo"},
                        "launch": {"enabled": True, "device_kind": "cpu"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        runtime_envs = module.load_runtime_env_registry(config_path)
    finally:
        config_path.unlink(missing_ok=True)

    assert (
        runtime_envs["python.cpu.demo"]["python_executable"] == "/envs/cpu/bin/python"
    )
    assert runtime_envs["python.cpu.demo"]["launch"]["enabled"] is True
