from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .exporter import export_package


def _load_callable(reference: str) -> Callable[[], Any]:
    module_name, separator, symbol_name = reference.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"expected MODULE:SYMBOL, got {reference!r}")
    value = getattr(importlib.import_module(module_name), symbol_name)
    if not callable(value):
        raise TypeError(f"{reference!r} is not callable")
    return value


def _names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one name is required")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trusted PyTorch model to a PNM-IR AOTInductor package"
    )
    parser.add_argument("--factory", required=True, help="zero-argument MODULE:SYMBOL")
    parser.add_argument(
        "--example-inputs", required=True, help="zero-argument MODULE:SYMBOL"
    )
    parser.add_argument("--input-names", required=True, type=_names)
    parser.add_argument("--output-names", required=True, type=_names)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--target", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = _load_callable(args.factory)()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("--factory must return torch.nn.Module")

    example_inputs = _load_callable(args.example_inputs)()
    if not isinstance(example_inputs, tuple):
        raise TypeError("--example-inputs must return a tuple of tensors")

    output = export_package(
        model,
        example_inputs,
        args.output,
        model_name=args.model_name,
        model_version=args.model_version,
        input_names=args.input_names,
        output_names=args.output_names,
        target=args.target,
        force=args.force,
    )
    print(output)
    return 0
