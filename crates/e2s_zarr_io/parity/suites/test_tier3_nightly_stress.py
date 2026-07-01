# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier3 nightly stress placeholder suite."""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Tier3 stress cases are added after Tier1/Tier2 parity lanes are stable."
)
def test_tier3_nightly_placeholder() -> None:
    """Placeholder to reserve Tier3 suite path in CI."""
