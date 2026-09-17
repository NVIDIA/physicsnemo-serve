from __future__ import annotations

import argparse
from pathlib import Path

from .onnx_importer import import_onnx_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an ONNX graph as a PNM-IR ONNX Runtime package"
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--runtime-version", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = import_onnx_package(
        args.onnx,
        args.output,
        model_name=args.model_name,
        model_version=args.model_version,
        target=args.target,
        precision=args.precision,
        runtime_version=args.runtime_version,
        force=args.force,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
