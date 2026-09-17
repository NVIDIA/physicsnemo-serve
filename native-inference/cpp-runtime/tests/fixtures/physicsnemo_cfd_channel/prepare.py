from __future__ import annotations

import argparse
from pathlib import Path

from pnmir_export import export_onnxruntime_package

from .model import create_model, example_inputs


def prepare_package(output: Path, *, force: bool = False) -> Path:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and export the small PhysicsNeMo CFD example"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(prepare_package(args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
