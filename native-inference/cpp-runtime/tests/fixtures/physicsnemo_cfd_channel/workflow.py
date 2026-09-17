from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from .problem import analytic_solution, channel_coordinates


class ChannelFlowSource:
    """Local CFD data source, analogous to an Earth2Studio DataSource."""

    def __call__(self) -> torch.Tensor:
        return channel_coordinates()


@dataclass(frozen=True)
class PNMIRRunner:
    """Packaged model component backed by the PNM-IR C++ executable."""

    executable: Path
    package: Path

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = ",".join(format(value, ".9g") for value in coordinates.flatten())
        result = subprocess.run(
            [
                str(self.executable),
                "run",
                str(self.package),
                "--values",
                values,
                "--backend",
                "onnxruntime",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PNM-IR failed with exit {result.returncode}: {result.stderr.strip()}"
            )

        name, separator, output_values = result.stdout.strip().partition(":")
        if separator != ":" or name != "flow":
            raise RuntimeError(f"unexpected PNM-IR output: {result.stdout!r}")
        flow = torch.tensor(
            [float(value) for value in output_values.split()], dtype=torch.float32
        )
        return flow.reshape(coordinates.shape[0], 3)


@dataclass(frozen=True)
class CsvBackend:
    """Output component, analogous to an Earth2Studio IOBackend."""

    path: Path

    def write(self, coordinates: torch.Tensor, flow: torch.Tensor) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(("x", "y", "u", "v", "p"))
            for coordinate, prediction in zip(coordinates, flow, strict=True):
                writer.writerow((*coordinate.tolist(), *prediction.tolist()))


def run_diagnostic(
    source: ChannelFlowSource, model: PNMIRRunner, io: CsvBackend
) -> torch.Tensor:
    """Compose data, packaged inference, and output components."""
    coordinates = source()
    prediction = model(coordinates)
    io.write(coordinates, prediction)
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Earth2Studio-style PhysicsNeMo CFD workflow"
    )
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = ChannelFlowSource()
    prediction = run_diagnostic(
        source,
        PNMIRRunner(args.pnmir, args.package),
        CsvBackend(args.output),
    )
    max_error = (prediction - analytic_solution(source())).abs().max().item()
    if max_error > 0.01:
        raise RuntimeError(f"channel-flow maximum error is too large: {max_error}")
    print(f"wrote {prediction.shape[0]} points to {args.output}")
    print(f"maximum analytic error: {max_error:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
