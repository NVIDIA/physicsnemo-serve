# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA lane placeholder suite."""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="CUDA lane parity cases require GPU runner setup and are added separately."
)
def test_cuda_lane_placeholder() -> None:
    """Placeholder to reserve CUDA lane suite path in CI."""
