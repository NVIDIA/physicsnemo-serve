# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deploy configuration loader.

Usage from any Python script in this repo::

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "deploy"))
    from config import load_deploy_config

    cfg = load_deploy_config()
    registry = cfg.get("docker_registry", "")

Resolution order (highest wins):
    1. Environment variable (key uppercased, e.g. ``nfs_mount_base`` → ``NFS_MOUNT_BASE``)
    2. ``deploy/config.yaml`` (local, git-ignored)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _DEPLOY_DIR / "config.yaml"

_KNOWN_KEYS = (
    "docker_registry",
    "image_name",
    "runtime_base_image",
    "python_service_image",
    "lepton_workspace_id",
    "lepton_node_group",
    "lepton_dashboard_url",
    "pull_secret",
    "nfs_mount_base",
    "repo_url",
)


def load_deploy_config() -> dict[str, str]:
    """Read deploy configuration from config.yaml and environment variables.

    Values from environment variables (uppercased key names) take
    precedence over the YAML file.  If the YAML file is missing, only
    environment variables are used (with a warning to stderr).
    """
    cfg: dict[str, str] = {}
    try:
        for line in _CONFIG_PATH.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            cfg[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        print(
            f"WARNING: deploy config not found at {_CONFIG_PATH}. "
            "Copy deploy/config.example.yaml to deploy/config.yaml and "
            "fill in your environment-specific values.",
            file=sys.stderr,
        )

    for key in _KNOWN_KEYS:
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            cfg[key] = env_val

    return cfg
