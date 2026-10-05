"""Compile symbolic AOTI outputs and validate their contracts through the SDK CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch

from pnmir_export import export_package


class DynamicOutputs(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            value * 2,
            value.T,
            torch.cat((value, value), dim=0),
            value.sum(dim=0),
            value.sum(dim=1),
            value.reshape(-1),
            value[:1],
            value.sum(),
        )


OUTPUT_NAMES = (
    "doubled", "transposed", "concatenated", "columns", "rows", "flat", "first", "sum"
)
OUTPUT_SHAPES = ([-1, 4], [4, -1], [-1, 4], [4], [-1], [-1], [1, 4], [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.target == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; skipping AOTI dynamic output test")
        return 77

    executable = args.pnmir.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"PhysicsNeMo Inference executable not found: {executable}")
    package = args.output.resolve()
    export_package(
        DynamicOutputs(),
        (torch.arange(8, dtype=torch.float32).reshape(2, 4),),
        package,
        model_name="aoti-dynamic-outputs",
        model_version="0.1.0",
        input_names=("input",),
        output_names=OUTPUT_NAMES,
        target=args.target,
        dynamic_shapes=({0: torch.export.Dim("batch", min=2, max=8)},),
        force=True,
    )
    manifest = json.loads((package / "model.json").read_text(encoding="utf-8"))
    assert manifest["inputs"] == [
        {"name": "input", "dtype": "float32", "shape": [-1, 4]}
    ], manifest["inputs"]
    assert manifest["outputs"] == [
        {"name": name, "dtype": "float32", "shape": shape}
        for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES, strict=True)
    ], manifest["outputs"]

    with tempfile.TemporaryDirectory(
        prefix=f"{package.name}-validation-", dir=package.parent
    ) as temporary:
        root = Path(temporary)
        for batch in (3, 5):
            inputs = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) - 5
            expected_outputs = DynamicOutputs()(inputs)
            input_file = root / f"input-{batch}.f32"
            input_file.write_bytes(inputs.numpy().tobytes())
            metadata_file = root / f"metadata-{batch}.json"
            output_files = [root / f"{name}-{batch}.f32" for name in OUTPUT_NAMES]
            command = [
                str(executable),
                "run",
                str(package),
                "--backend",
                "aoti",
                "--device",
                args.target,
                "--input-file",
                f"input={input_file}",
                "--input-shape",
                f"input={batch},4",
                "--output-metadata",
                str(metadata_file),
                "--warmup",
                "1",
                "--iterations",
                "2",
            ]
            for name, path in zip(OUTPUT_NAMES, output_files, strict=True):
                command.extend(("--output-file", f"{name}={path}"))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f"AOTI dynamic batch {batch} on {args.target} failed with "
                    f"exit {result.returncode}\nstdout:\n{result.stdout}"
                    f"\nstderr:\n{result.stderr}"
                )
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            assert metadata["completed"] is True, metadata
            assert metadata["backend"] == "aoti", metadata
            assert metadata["execution_device"]["type"] == args.target, metadata
            assert len(metadata["outputs"]) == len(OUTPUT_NAMES), metadata
            for name, path, expected, observed in zip(
                OUTPUT_NAMES, output_files, expected_outputs, metadata["outputs"], strict=True
            ):
                assert observed["name"] == name, observed
                assert observed["dtype"] == "float32", observed
                assert observed["shape"] == list(expected.shape), observed
                assert observed["byte_size"] == expected.numel() * expected.element_size(), observed
                storage = bytearray(path.read_bytes())
                actual = torch.frombuffer(storage, dtype=torch.float32).reshape(expected.shape)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            print(f"AOTI {args.target} batch {batch}: all {len(OUTPUT_NAMES)} output shapes and values match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
