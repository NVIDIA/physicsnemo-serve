from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .exporter import _tensor_outputs, _validate_names
from .onnx_importer import import_onnx_package
from .options import ExportOptions


def export_onnx_model(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    onnx_path: str | Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    device: torch.device,
    dynamo: bool = True,
    options: ExportOptions | None = None,
) -> Path:
    """Export a static PyTorch model to a checked, named ONNX graph."""
    options = ExportOptions() if options is None else options
    if not isinstance(options, ExportOptions):
        raise TypeError("options must be ExportOptions")
    if options.onnx_passes and not dynamo:
        raise ValueError("ONNX graph passes require dynamo=True")
    if not example_inputs or not all(
        isinstance(value, torch.Tensor) for value in example_inputs
    ):
        raise TypeError("example_inputs must be a non-empty tuple of tensors")
    _validate_names(input_names, len(example_inputs), "input")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported ONNX export device: {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA ONNX export requires an available CUDA device")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {device.index}")

    prepared_inputs = tuple(
        value.detach().to(device).contiguous() for value in example_inputs
    )
    model = model.eval().to(device)
    with torch.inference_mode():
        # Eager execution and either capture path may mutate their inputs.
        outputs = _tensor_outputs(model(*(value.clone() for value in prepared_inputs)))
    _validate_names(output_names, len(outputs), "output")
    del outputs

    destination = Path(onnx_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if options.onnx_passes:
            from .graph_passes import prepare_onnx_program

            model = prepare_onnx_program(
                model,
                tuple(value.clone() for value in prepared_inputs),
                options,
                destination.with_suffix(".export-options.json"),
            )
        program = torch.onnx.export(
            model,
            tuple(value.clone() for value in prepared_inputs),
            None if dynamo else destination,
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamo=dynamo,
            external_data=True,
            do_constant_folding=True,
        )
    if dynamo:
        program.save(destination, external_data=True)
    return destination


def export_onnxruntime_package(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    target: str = "cuda",
    device: int = 0,
    force: bool = False,
    dynamo: bool = True,
) -> Path:
    """Export PyTorch to ONNX and publish an ONNX Runtime package."""
    if target not in {"cpu", "cuda"}:
        raise ValueError("ONNX Runtime target must be cpu or cuda")
    if device < 0:
        raise ValueError("device must be non-negative")

    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_device = (
        torch.device("cuda", device) if target == "cuda" else torch.device("cpu")
    )
    with tempfile.TemporaryDirectory(
        prefix="pnm-ir-onnxruntime-export-", dir=output.parent
    ) as temporary:
        onnx_path = export_onnx_model(
            model,
            example_inputs,
            Path(temporary) / "model.onnx",
            input_names=input_names,
            output_names=output_names,
            device=export_device,
            dynamo=dynamo,
        )
        return import_onnx_package(
            onnx_path,
            output,
            model_name=model_name,
            model_version=model_version,
            target=target,
            force=force,
        )
