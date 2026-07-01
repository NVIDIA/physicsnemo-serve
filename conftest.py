# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

collect_ignore = []


def _can_import(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


if not _can_import("python_bridge"):
    collect_ignore.append("python/e2s_tools/test_python_bridge.py")

if not _can_import("scripts.compare_deterministic_rust_vs_py_async"):
    collect_ignore.append("tests/test_compare_deterministic_rust_vs_py_async.py")

if not _can_import("scripts.run_deterministic_cpu_profile"):
    collect_ignore.append("tests/test_run_deterministic_cpu_profile.py")
