# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-light execution primitives shared by PhysicsNeMo-CFD plugins."""

from .supervisor import SupervisedProcessResult, run_supervised_process

__all__ = ["SupervisedProcessResult", "run_supervised_process"]
