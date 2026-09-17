from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from .onnx_exporter import export_onnx_model
from .tensorrt_builder import build_tensorrt_package


@contextmanager
def tensorrt_ieee_fp32() -> Iterator[None]:
    """Match TensorRT's TF32-disabled build policy and restore PyTorch state."""
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)


def export_tensorrt_package(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    device: int = 0,
    workspace_size: int = 1 << 30,
    force: bool = False,
    dynamo: bool = True,
) -> Path:
    """Export a static PyTorch CUDA model to ONNX and build a TensorRT package."""
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires an available CUDA device")
    if device < 0 or device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {device}")

    target_device = torch.device("cuda", device)
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    try:
        with tensorrt_ieee_fp32():
            with tempfile.TemporaryDirectory(
                prefix="pnm-ir-tensorrt-export-", dir=output.parent
            ) as temporary:
                onnx_path = export_onnx_model(
                    model,
                    example_inputs,
                    Path(temporary) / "model.onnx",
                    input_names=input_names,
                    output_names=output_names,
                    device=target_device,
                    dynamo=dynamo,
                )
                return build_tensorrt_package(
                    onnx_path,
                    output,
                    model_name=model_name,
                    model_version=model_version,
                    device=device,
                    workspace_size=workspace_size,
                    force=force,
                )
    finally:
        torch.cuda.set_device(previous_device)
