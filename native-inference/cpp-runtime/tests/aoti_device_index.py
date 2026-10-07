from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from model_builder.export import export_package


class WeightedAffine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))
        self.bias = torch.nn.Parameter(torch.tensor([1.0, -1.0, 1.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight + self.bias


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable; skipping AOTI device index test")
        return 77

    export_package(
        WeightedAffine(),
        (torch.tensor([1.0, 2.0, 3.0]),),
        args.output,
        model_name="aoti-device-index",
        model_version="0.1.0",
        input_names=("input",),
        output_names=("output",),
        target="cuda",
        force=True,
    )
    return subprocess.run([str(args.runner), str(args.output)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
