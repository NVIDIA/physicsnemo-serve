# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report helpers for parity/baseline diagnostics."""

from __future__ import annotations

from typing import Any

from .manifest_compare import compare_semantic_manifests


def build_baseline_report(
    *,
    case_id: str,
    py_sync_manifest: dict[str, Any],
    py_async_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact baseline report for py_sync vs py_async comparison."""
    diffs = compare_semantic_manifests(py_sync_manifest, py_async_manifest)
    return {
        "case_id": case_id,
        "py_baseline_equal": len(diffs) == 0,
        "diff_count": len(diffs),
        "diff_preview": diffs[:20],
    }
