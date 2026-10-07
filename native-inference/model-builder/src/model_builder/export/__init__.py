"""Export APIs; optional framework dependencies load when their API is accessed."""

from importlib import import_module


_EXPORT_MODULES = {
    "ExportContext": "options",
    "ExportOptions": "options",
    "ParityMetrics": "parity",
    "assert_tensor_parity": "parity",
    "build_tensorrt_package": "tensorrt_builder",
    "export_onnx_model": "onnx_exporter",
    "export_onnxruntime_package": "onnx_exporter",
    "export_package": "exporter",
    "export_tensorrt_package": "tensorrt_exporter",
    "import_onnx_package": "onnx_importer",
    "tensorrt_ieee_fp32": "tensorrt_exporter",
    "validate_pnmir_package": "validation",
    "validate_onnxruntime_package": "validation",
    "validate_tensorrt_package": "validation",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
