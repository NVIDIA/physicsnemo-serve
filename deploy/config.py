# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deploy configuration loader.

Usage from any Python script in this repo::

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "deploy"))
    from config import load_deploy_config

    cfg = load_deploy_config()
    registry = cfg.get("docker_registry", "")
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _DEPLOY_DIR / "config.yaml"


def load_deploy_config() -> dict[str, str]:
    """Read flat key-value pairs from deploy/config.yaml.

    If the file is missing, a warning is printed to stderr and an empty
    dict is returned so that callers can still fall back to their own
    defaults or environment variables.
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
    return cfg
