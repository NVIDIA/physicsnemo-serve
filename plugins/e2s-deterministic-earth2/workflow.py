# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from e2s_workflow import Earth2Workflow
from plugin_sdk import _close_cached_fsspec_sessions

if TYPE_CHECKING:
    from earth2studio.io import IOBackend


def prepare_model_cache(_ctx: dict) -> dict[str, list[str]]:
    from earth2studio.models.px import FCN

    package = FCN.load_default_package()
    FCN.load_model(package)
    return {"model_names": ["FCN"]}


class DeterministicEarth2Workflow(Earth2Workflow):
    """
    Deterministic workflow with auto-registration
    """

    name = "deterministic_earth2_workflow"
    description = "Deterministic workflow with auto-registration"
    cache_scope = "process"
    model_cache_names = ["FCN"]
    cache_preserve_attributes = ("package", "model", "data")

    def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
        super().__init__()

        from earth2studio.data import GFS
        from earth2studio.models.px import DLWP, FCN

        if model_type == "fcn":
            package = FCN.load_default_package()
            self.model = FCN.load_model(package)
        elif model_type == "dlwp":
            package = DLWP.load_default_package()
            self.model = DLWP.load_model(package)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.package = package
        _close_cached_fsspec_sessions()
        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 6,
    ) -> None:
        """Run the deterministic workflow pipeline"""
        from earth2studio import run

        run.deterministic(start_time, num_steps, self.model, self.data, io)

    def cleanup(self) -> None:
        # Package HTTP sessions may be shared by Earth2 package caches.
        # Drop package refs before generic cleanup to avoid closing shared clients.
        self._clear_attributes("package")
        try:
            super().cleanup()
        finally:
            self._clear_attributes("model", "data")
            self._cleanup_torch_runtime()


WORKFLOW = DeterministicEarth2Workflow
