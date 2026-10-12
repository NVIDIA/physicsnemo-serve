"""Reproduce the frozen channel-flow fixture or run its optional CSV diagnostic."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


def channel_coordinates() -> torch.Tensor:
    """Return the fixed 3x3 channel grid used by the static-shape MVP."""
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def analytic_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Steady plane-Poiseuille solution for viscosity 0.5 and unit density."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    u = 1.0 - y.square()
    v = torch.zeros_like(x)
    pressure = 1.0 - x
    return torch.stack((u, v, pressure), dim=-1)


def create_model() -> torch.nn.Module:
    """Train the tiny PhysicsNeMo surrogate used for the export smoke test."""
    from physicsnemo.models.mlp.fully_connected import FullyConnected

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = FullyConnected(
                in_features=2,
                out_features=3,
                num_layers=3,
                layer_size=32,
            )

            axis_x = torch.linspace(0.0, 1.0, 21)
            axis_y = torch.linspace(-1.0, 1.0, 21)
            xx, yy = torch.meshgrid(axis_x, axis_y, indexing="ij")
            coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
            targets = analytic_solution(coordinates)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(1200):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(coordinates), targets)
                loss.backward()
                optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)

    model.eval()
    with torch.inference_mode():
        error = model(channel_coordinates()) - analytic_solution(channel_coordinates())
        if error.abs().max().item() > 0.01:
            raise RuntimeError("PhysicsNeMo channel-flow surrogate did not converge")
    return model


def example_inputs() -> tuple[torch.Tensor, ...]:
    return (channel_coordinates(),)


def prepare_package(output: Path, *, force: bool = False) -> Path:
    from model_builder.export import export_onnxruntime_package

    return export_onnxruntime_package(
        create_model(),
        example_inputs(),
        output,
        model_name="physicsnemo-channel-flow",
        model_version="0.1.0",
        input_names=("coordinates",),
        output_names=("flow",),
        target="cpu",
        force=force,
        dynamo=False,
    )


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
        description="Reproduce or diagnose the frozen PhysicsNeMo channel-flow fixture"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Replace a reproduced package"
    )
    parser.add_argument(
        "--diagnostic", action="store_true", help="Run native inference and write CSV"
    )
    parser.add_argument(
        "--pnmir", type=Path, help="Native runtime executable for --diagnostic"
    )
    parser.add_argument("--package", type=Path, help="Model package for --diagnostic")
    args = parser.parse_args()

    if args.diagnostic:
        if args.pnmir is None or args.package is None:
            parser.error("--diagnostic requires --pnmir and --package")
        if args.force:
            parser.error("--force is only supported when reproducing a package")
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
    else:
        if args.pnmir is not None or args.package is not None:
            parser.error("--pnmir and --package require --diagnostic")
        print(prepare_package(args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
