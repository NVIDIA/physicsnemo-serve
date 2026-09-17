from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .exporter import _prepare_output_directory, _publish_output_directory


_ONNX_DTYPES = {
    1: "float32",
    2: "uint8",
    6: "int32",
    7: "int64",
    10: "float16",
    16: "bfloat16",
}


def _onnx_module() -> Any:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX import requires the optional 'onnx' package; "
            "run python -m pip install onnx"
        ) from error
    return onnx


def _dtype_name(element_type: int, tensor_name: str) -> str:
    try:
        return _ONNX_DTYPES[element_type]
    except KeyError as error:
        raise ValueError(
            f"unsupported ONNX tensor dtype {element_type} for: {tensor_name}"
        ) from error


def _tensor_schema(value_info: Any) -> dict[str, Any]:
    if not value_info.name:
        raise ValueError("ONNX graph tensor names must be non-empty")
    if not value_info.type.HasField("tensor_type"):
        raise ValueError(f"ONNX graph value is not a tensor: {value_info.name}")

    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"ONNX graph tensor has no shape: {value_info.name}")

    shape: list[int] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            if dimension.dim_value <= 0:
                raise ValueError(
                    f"ONNX graph tensor has a non-positive static dimension: "
                    f"{value_info.name}"
                )
            shape.append(dimension.dim_value)
        else:
            shape.append(-1)
    return {
        "name": value_info.name,
        "dtype": _dtype_name(tensor_type.elem_type, value_info.name),
        "shape": shape,
    }


def _graph_contract(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    initializer_names.update(
        initializer.values.name for initializer in model.graph.sparse_initializer
    )
    inputs = [
        _tensor_schema(value)
        for value in model.graph.input
        if value.name not in initializer_names
    ]
    outputs = [_tensor_schema(value) for value in model.graph.output]
    if not inputs or not outputs:
        raise ValueError("ONNX graph must have at least one input and output tensor")
    return inputs, outputs


def _external_data_locations(model: Any, source: Path) -> list[tuple[Path, Path]]:
    onnx = _onnx_module()
    locations: set[str] = set()
    for tensor in (*model.graph.initializer, *model.graph.sparse_initializer):
        values = tensor.values if hasattr(tensor, "values") else tensor
        if values.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in values.external_data}
        location = metadata.get("location", "")
        relative = Path(location)
        if not location or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe ONNX external-data location: {location!r}")
        locations.add(location)

    source_root = source.parent.resolve()
    result: list[tuple[Path, Path]] = []
    for location in sorted(locations):
        source_file = (source_root / location).resolve()
        try:
            source_file.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"ONNX external data escapes the model directory: {location}"
            ) from error
        if not source_file.is_file():
            raise FileNotFoundError(f"ONNX external data does not exist: {source_file}")
        result.append((source_file, Path(location)))
    return result


def import_onnx_package(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    target: str = "cpu",
    precision: str = "fp32",
    runtime_version: str = "",
    force: bool = False,
) -> Path:
    """Validate an ONNX graph and atomically publish a PNM-IR package."""
    if not model_name or not model_version:
        raise ValueError("model name and version are required")
    if target not in {"cpu", "cuda"}:
        raise ValueError("ONNX Runtime target must be cpu or cuda")
    if not precision:
        raise ValueError("precision is required")

    source = Path(onnx_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {source}")

    onnx = _onnx_module()
    model = onnx.load_model(source, load_external_data=False)
    onnx.checker.check_model(str(source))
    inputs, outputs = _graph_contract(model)
    external_data = _external_data_locations(model, source)
    for _, relative_path in external_data:
        if relative_path.parts[0].casefold() in {"model.json", "model.onnx"}:
            raise ValueError(
                f"ONNX external data conflicts with package file: {relative_path}; "
                "rename the external-data location before importing"
            )

    target_path, work = _prepare_output_directory(Path(output_dir), force)
    try:
        artifact_path = work / "model.onnx"
        shutil.copy2(source, artifact_path)
        for external_source, relative_path in external_data:
            external_target = work / relative_path
            external_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(external_source, external_target)

        artifact: dict[str, Any] = {
            "backend": "onnxruntime",
            "target": target,
            "precision": precision,
            "path": "model.onnx",
        }
        if runtime_version:
            artifact["runtime_version"] = runtime_version
        manifest = {
            "format_version": 1,
            "model": {"name": model_name, "version": model_version},
            "producer": {
                "name": "pnm-ir-onnx-import",
                "version": "0.1.0",
                "onnx_version": onnx.__version__,
            },
            "inputs": inputs,
            "outputs": outputs,
            "artifacts": [artifact],
        }
        (work / "model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_output_directory(target_path, work)
        return target_path
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
