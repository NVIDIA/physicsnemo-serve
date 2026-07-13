# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from plugin_sdk import (
    ExecutionContext,
    PluginWorkflow,
    PrepareContext,
    PrepareResult,
    RawRequest,
)


@dataclass
class ExampleUserInput:
    task_name: str = "example_task"
    num_iterations: int = 5
    delay_seconds: float = 0.5
    generate_output: bool = True


@dataclass
class ExampleUserOutput:
    status: str
    task_name: str
    num_iterations: int


class ExampleUserWorkflow(PluginWorkflow):
    cache_scope = "process"
    input_model = ExampleUserInput
    output_model = ExampleUserOutput

    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        params = dict(request.raw_fields)

        task_name = str(params.get("task_name", "example_task")).strip()
        num_iterations = int(params.get("num_iterations", 5))
        delay_seconds = float(params.get("delay_seconds", 0.5))
        generate_output = bool(params.get("generate_output", True))

        return PrepareResult(
            inputs=ExampleUserInput(
                task_name=task_name,
                num_iterations=num_iterations,
                delay_seconds=delay_seconds,
                generate_output=generate_output,
            ),
        )

    def run(self, inputs: ExampleUserInput, ctx: ExecutionContext) -> ExampleUserOutput:
        # Iterate with delay
        for i in range(inputs.num_iterations):
            if inputs.delay_seconds > 0:
                time.sleep(inputs.delay_seconds)

        # Optionally generate output files
        if inputs.generate_output:
            run_dir = Path(ctx.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "task_name": inputs.task_name,
                "num_iterations": inputs.num_iterations,
                "status": "success",
            }
            results_path = run_dir / "results.json"
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            ctx.outputs.register(
                "results",
                results_path,
                media_type="application/json",
                primary=True,
            )

            summary_path = run_dir / "summary.txt"
            summary_path.write_text(
                f"Task: {inputs.task_name}\n"
                f"Iterations: {inputs.num_iterations}\n"
                f"Status: success\n",
                encoding="utf-8",
            )

        return ExampleUserOutput(
            status="success",
            task_name=inputs.task_name,
            num_iterations=inputs.num_iterations,
        )


WORKFLOW = ExampleUserWorkflow
