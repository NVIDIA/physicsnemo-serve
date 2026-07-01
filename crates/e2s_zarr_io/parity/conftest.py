# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for parity suites."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))


def _ensure_crate_root_on_path() -> None:
    crate_root = Path(__file__).resolve().parent.parent
    crate_root_str = str(crate_root)
    if crate_root_str not in sys.path:
        sys.path.insert(0, crate_root_str)


_ensure_crate_root_on_path()


def _crate_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _ensure_e2s_zarr_io_bindings() -> ModuleType:
    try:
        module = importlib.import_module("e2s_zarr_io")
    except ImportError:
        crate_root = _crate_root()
        repo_root = crate_root.parent.parent
        env = os.environ.copy()
        env.pop("CONDA_PREFIX", None)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(crate_root)],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to build/install e2s_zarr_io Python bindings "
                "for parity tests.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            ) from None
        importlib.invalidate_caches()
        sys.modules.pop("e2s_zarr_io", None)
        module = importlib.import_module("e2s_zarr_io")

    if not hasattr(module, "E2sZarrIoBackend"):
        raise RuntimeError(
            "e2s_zarr_io Python bindings are installed but "
            "E2sZarrIoBackend is not exposed."
        )
    return module


@pytest.fixture(scope="session")
def e2s_zarr_io_module() -> ModuleType:
    return _ensure_e2s_zarr_io_bindings()
