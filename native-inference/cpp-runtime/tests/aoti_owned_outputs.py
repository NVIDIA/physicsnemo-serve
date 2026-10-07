from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from model_builder.export import export_package


class AliasedOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("constant", torch.arange(20.0, 24.0))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        intermediate = value + 10
        return (
            value,
            value[1:3],
            value[::2],
            intermediate[1:3],
            self.constant[1:3],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.target == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; skipping AOTI owned output test")
        return 77

    export_package(
        AliasedOutputs(),
        (torch.arange(4, dtype=torch.float32),),
        args.output,
        model_name="aoti-owned-outputs",
        model_version="0.1.0",
        input_names=("input",),
        output_names=(
            "identity",
            "offset_view",
            "strided_view",
            "intermediate",
            "constant",
        ),
        target=args.target,
        force=True,
    )
    return subprocess.run(
        [str(args.runner), str(args.output), args.target], check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
