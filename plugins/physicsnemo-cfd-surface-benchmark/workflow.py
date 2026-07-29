# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_PATH = PLUGIN_ROOT / "surface_benchmark_impl.py"


def _load_surface_implementation():
    module_name = f"{__name__}__surface_benchmark_impl"
    spec = importlib.util.spec_from_file_location(module_name, IMPLEMENTATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load surface benchmark implementation: {IMPLEMENTATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_SURFACE_IMPLEMENTATION = _load_surface_implementation()
SurfaceBenchmarkWorkflow = _SURFACE_IMPLEMENTATION.SurfaceBenchmarkWorkflow

MANIFEST_PATH = PLUGIN_ROOT / "plugin.yaml"


class PhysicsNeMoCfdSurfaceBenchmarkWorkflow(SurfaceBenchmarkWorkflow):
    manifest_path = MANIFEST_PATH

    def __init__(self) -> None:
        super().__init__(self.manifest_path)


WORKFLOW = PhysicsNeMoCfdSurfaceBenchmarkWorkflow
