# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = REPO_ROOT / "scripts" / "worker_runtime_config.json"
DOCKERFILE = REPO_ROOT / "Dockerfile.Earth2Studio.scicomp-rust-slim"
CFD_PYTHON = "/opt/physicsnemo-cfd-venv/bin/python"
CFD_MODEL_CACHE = "/outputs/.cache/physicsnemo-cfd/models"


def _load_runtime_env_launcher():
    script = REPO_ROOT / "scripts" / "runtime_env_launcher.py"
    spec = importlib.util.spec_from_file_location(
        "physicsnemo_cfd_runtime_env_launcher", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cfd_image_uses_pinned_isolated_runtime_and_packages_plugins() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "ARG PHYSICSNEMO_CFD_COMMIT="
        "921f14dc2ac14c04aabffaba3290deb792379dd8" in dockerfile
    )
    assert "ARG PHYSICSNEMO_CFD_VERSION=0.0.2" in dockerfile
    assert "ARG PHYSICSNEMO_VERSION=2.1.1" in dockerfile
    assert "ARG PHYSICSNEMO_CFD_TORCH_VERSION=2.10.0+cu130" in dockerfile
    assert 'python3.12 -m venv "$PHYSICSNEMO_CFD_VENV"' in dockerfile
    assert "python3.12 -m venv --system-site-packages" not in dockerfile
    assert "base image's 2.10 alpha build" in dockerfile
    assert 'env -u PIP_CONSTRAINT "$PHYSICSNEMO_CFD_PYTHON_EXECUTABLE"' in dockerfile
    assert "--index-url https://download.pytorch.org/whl/cu130" in dockerfile
    assert '"nvidia-physicsnemo[cu13]==${PHYSICSNEMO_VERSION}"' in dockerfile
    assert "nvidia-physicsnemo-cfd[gpu,evaluation-hf] @ git+https://" in dockerfile
    assert '"$PHYSICSNEMO_CFD_PYTHON_EXECUTABLE" -m pip check' in dockerfile
    assert 'direct_url["vcs_info"]["commit_id"] == expected_commit' in dockerfile
    assert "assert torch.__version__ == expected_torch" in dockerfile
    assert (
        'importlib.metadata.version("torchvision") == expected_torchvision'
        in dockerfile
    )
    assert 'assert torch.version.cuda == "13.0"' in dockerfile
    assert "/tmp/physicsnemo-cfd-constraints.txt" in dockerfile
    assert '"torch==${PHYSICSNEMO_CFD_TORCH_VERSION}"' in dockerfile
    assert '"torchvision==${PHYSICSNEMO_CFD_TORCHVISION_VERSION}"' in dockerfile
    assert (
        'Version(importlib.metadata.version("cupy-cuda13x")) < Version("14.0.0")'
        in dockerfile
    )
    assert (
        '"$PHYSICSNEMO_CFD_PYTHON_EXECUTABLE" -m pip install '
        "--no-cache-dir /tmp/wheels/*.whl" in dockerfile
    )

    expected_copies = (
        "python/physicsnemo_cfd_runtime /app/python/physicsnemo_cfd_runtime",
        "plugins/physicsnemo-cfd-surface-benchmark "
        "/app/plugins/physicsnemo-cfd-surface-benchmark",
    )
    for copy_spec in expected_copies:
        assert f"COPY {copy_spec}" in dockerfile
    assert "COPY python/physicsnemo_cfd_plugins" not in dockerfile
    assert "COPY plugins/physicsnemo-cfd-volume-benchmark" not in dockerfile
    assert "COPY plugins/physicsnemo-cfd-domino-nim" not in dockerfile


def test_cfd_gpu_runtime_shares_cache_without_changing_earth2_runtime() -> None:
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    runtime_envs = config["python_runtime_envs"]

    assert runtime_envs["earth2-cpu"]["python_executable"] == (
        "/opt/physicsnemo-serve-venv/bin/python"
    )
    assert runtime_envs["earth2-gpu"]["launch"]["workers_per_device"] == 1

    gpu = runtime_envs["physicsnemo-cfd-gpu"]
    assert "physicsnemo-cfd-cpu" not in runtime_envs
    assert gpu["python_executable"] == CFD_PYTHON
    assert gpu["env"]["PHYSICSNEMO_CFD_MODEL_CACHE"] == CFD_MODEL_CACHE
    assert gpu["env"]["HF_HOME"] == "/outputs/.cache/physicsnemo-cfd/huggingface"
    assert gpu["env"]["PYTHONPATH"] == "/app/python"
    assert gpu["launch"] == {
        "enabled": False,
        "device_kind": "gpu",
        "workers_per_device": 1,
        "tags": ["physicsnemo-cfd", "gpu"],
    }


def test_cfd_runtime_config_launches_one_gpu_worker_per_visible_device() -> None:
    launcher = _load_runtime_env_launcher()
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    runtime_envs = {
        name: value
        for name, value in config["python_runtime_envs"].items()
        if name.startswith("physicsnemo-cfd-")
    }
    gpu_inventory = [
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
    ]

    specs = launcher.build_worker_launch_specs(
        runtime_envs,
        namespace="production",
        pod_name="cfd-0",
        enabled_executor_classes={"physicsnemo-cfd-gpu"},
        gpu_inventory=gpu_inventory,
        worker_script="/app/scripts/inference_worker.py",
        stream_prefix="physicsnemo:",
    )

    gpu_specs = [
        spec for spec in specs if spec["executor_class"] == "physicsnemo-cfd-gpu"
    ]
    assert specs == gpu_specs
    assert len(gpu_specs) == len(gpu_inventory)
    assert {spec["env"]["CUDA_VISIBLE_DEVICES"] for spec in gpu_specs} == {"0", "1"}
    assert {spec["env"]["GPU_WORKER_INDEX"] for spec in gpu_specs} == {"0"}
    assert {spec["python_executable"] for spec in specs} == {CFD_PYTHON}
    assert {spec["env"]["PHYSICSNEMO_CFD_MODEL_CACHE"] for spec in specs} == {
        CFD_MODEL_CACHE
    }
