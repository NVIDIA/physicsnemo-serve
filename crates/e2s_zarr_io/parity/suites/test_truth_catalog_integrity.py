# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Committed truth catalog integrity checks for CI parity gating."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def test_catalog_cases_have_committed_truth_artifacts() -> None:
    parity_root = Path(__file__).resolve().parent.parent
    catalog_path = parity_root / "ground_truth" / "catalog.yaml"
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    assert isinstance(cases, list) and cases, (
        "ground-truth catalog must define at least one case"
    )

    for case in cases:
        case_id = str(case.get("case_id", "<unknown>"))
        status = str(case.get("status", ""))
        assert status != "pending_truth_refresh", (
            f"{case_id}: catalog status is still pending_truth_refresh; "
            "record and commit a truth manifest before enabling CI parity gate"
        )

        for key in ("case_spec", "truth_manifest", "baseline_report"):
            relative = case.get(key)
            assert isinstance(relative, str) and relative, (
                f"{case_id}: missing catalog field '{key}'"
            )
            artifact_path = parity_root.parent / relative
            assert artifact_path.exists(), (
                f"{case_id}: catalog artifact does not exist: {artifact_path}"
            )
