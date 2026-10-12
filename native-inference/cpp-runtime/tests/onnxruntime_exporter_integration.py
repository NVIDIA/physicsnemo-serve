from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


from model_builder.export import (
    export_onnxruntime_package,
    validate_onnxruntime_package,
)


class Affine(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 2.0 * value + 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping ONNX Runtime exporter test")
        return 77

    model = Affine().eval()
    inputs = (torch.tensor([1.0, 2.0, 3.0]),)
    with torch.inference_mode():
        expected = model(*inputs)
    package = export_onnxruntime_package(
        model,
        inputs,
        args.output,
        model_name="onnxruntime-exporter-affine",
        model_version="0.1.0",
        input_names=("input",),
        output_names=("output",),
        target="cuda",
        force=True,
    )
    manifest = json.loads((package / "model.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == [
        {
            "backend": "onnxruntime",
            "target": "cuda",
            "precision": "fp32",
            "path": "model.onnx",
        }
    ]
    assert (package / "model.onnx").is_file()
    assert not (package / "artifacts").exists()
    metrics = validate_onnxruntime_package(
        args.pnmir,
        package,
        inputs,
        expected,
        max_abs_limit=1.0e-6,
        relative_l2_limit=1.0e-6,
    )
    assert metrics[0].max_abs == 0.0
    assert metrics[0].relative_l2 == 0.0
    print("ONNX Runtime PyTorch exporter and strict CUDA parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
