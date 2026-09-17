from __future__ import annotations

import argparse
from pathlib import Path

from .tensorrt_builder import build_tensorrt_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a static TensorRT PNM-IR package from ONNX"
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--workspace-size", default=1 << 30, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = build_tensorrt_package(
        args.onnx,
        args.output,
        model_name=args.model_name,
        model_version=args.model_version,
        device=args.device,
        workspace_size=args.workspace_size,
        force=args.force,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
