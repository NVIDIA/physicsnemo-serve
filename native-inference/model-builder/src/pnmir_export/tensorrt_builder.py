from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .exporter import _prepare_output_directory, _publish_output_directory
from .onnx_importer import _graph_contract, _onnx_module
from .tensorrt_profiles import (
    load_exact_plugins,
    plugin_names,
    prepare_exact_graph,
    profile_version,
    required_operators,
    resolve_plugin_libraries,
    selection_metadata,
)


def _tensorrt_module() -> Any:
    try:
        import tensorrt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT compilation requires the optional 'tensorrt' package; "
            "install physicsnemo-model-builder[tensorrt]"
        ) from error
    return tensorrt


def _require_static_contract(
    inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> None:
    supported_input_dtypes = {"float32", "int32", "int64"}
    for tensor in inputs:
        if tensor["dtype"] not in supported_input_dtypes:
            raise ValueError(
                "the TensorRT MVP requires float32 or integer index inputs: "
                f"{tensor['name']}"
            )
    for tensor in outputs:
        if tensor["dtype"] != "float32":
            raise ValueError(
                f"the TensorRT MVP requires float32 output tensors: {tensor['name']}"
            )
    for tensor in (*inputs, *outputs):
        if any(dimension == -1 for dimension in tensor["shape"]):
            raise ValueError(
                f"the TensorRT MVP requires static tensor shapes: {tensor['name']}"
            )


def build_tensorrt_package(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    device: int = 0,
    workspace_size: int = 1 << 30,
    force: bool = False,
    profile: str = "baseline",
    plugin_libraries: dict[str, str | Path] | None = None,
) -> Path:
    """Build a static IEEE-FP32 TensorRT engine and publish a PNM-IR package."""
    libraries = resolve_plugin_libraries(profile, plugin_libraries)
    if not model_name or not model_version:
        raise ValueError("model name and version are required")
    if device < 0:
        raise ValueError("device must be non-negative")
    if workspace_size <= 0:
        raise ValueError("workspace_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT compilation requires an available CUDA device")
    if device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {device}")

    source = Path(onnx_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {source}")

    onnx = _onnx_module()
    model = onnx.load_model(source, load_external_data=bool(libraries))
    onnx.checker.check_model(str(source))
    inputs, outputs = _graph_contract(model)
    _require_static_contract(inputs, outputs)

    trt = _tensorrt_module()
    torch.cuda.set_device(device)
    logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("TensorRT plugin registration failed")

    plugin_handles = []
    correctness_profile = None
    if libraries:
        plugin_handles, library_records = load_exact_plugins(trt, libraries, profile)
        counts = prepare_exact_graph(onnx, model, profile)
        correctness_profile = {
            "name": profile,
            "version": profile_version(profile),
            "plugins": list(plugin_names(profile)),
            "plugin_libraries": library_records,
            **selection_metadata(profile),
            "replacement_counts": counts,
            "dependency_scope": "matching exact plugin libraries installed with the C++ SDK",
        }

    target, work = _prepare_output_directory(Path(output_dir), force)
    try:
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        parsed = (
            parser.parse(model.SerializeToString())
            if libraries else parser.parse_from_file(str(source))
        )
        if not parsed:
            errors = "\n".join(
                str(parser.get_error(index)) for index in range(parser.num_errors)
            )
            raise RuntimeError(f"TensorRT could not parse ONNX model:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        config.clear_flag(trt.BuilderFlag.TF32)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        artifact_path = work / "model.plan"
        artifact_path.write_bytes(bytes(serialized))

        major, minor = torch.cuda.get_device_capability(device)
        runtime_version = trt.__version__
        manifest = {
            "format_version": 1,
            "model": {"name": model_name, "version": model_version},
            "producer": {
                "name": "pnm-ir-tensorrt-build",
                "version": "0.1.0",
                "onnx_version": onnx.__version__,
                "tensorrt_version": runtime_version,
                "cuda_device": torch.cuda.get_device_name(device),
                "compute_capability": f"{major}.{minor}",
                "tf32": False,
                "workspace_size": workspace_size,
            },
            "inputs": inputs,
            "outputs": outputs,
            "artifacts": [
                {
                    "backend": "tensorrt",
                    "target": "cuda",
                    "precision": "fp32",
                    "runtime_version": runtime_version,
                    "path": "model.plan",
                }
            ],
        }
        if correctness_profile is not None:
            manifest["artifacts"][0].update(
                correctness_profile=correctness_profile,
                required_operators=required_operators(profile),
            )
        (work / "model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_output_directory(target, work)
        # Keep registration handles reachable through engine serialization.
        del plugin_handles
        return target
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
