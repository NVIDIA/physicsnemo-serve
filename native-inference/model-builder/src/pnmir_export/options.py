"""Small, framework-independent adapter interface for ONNX export preparation."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ExportContext:
    """Builder-owned metadata; never the live model or its checkpoint."""

    backend: str
    device: str


@dataclass(frozen=True)
class ExportOptions:
    """Ordered ONNX preparation passes, applied before ONNX decomposition.

    Each callable receives a private captured FX GraphModule, edits its graph
    in place and returns the number of rewritten nodes. It must preserve the
    input/output contract and computation. Compilation and validation stay in
    the builder. Custom CUDA kernels and TensorRT plugins are separate assets.
    """

    onnx_passes: tuple[Callable, ...] = ()

    def __post_init__(self):
        if not isinstance(self.onnx_passes, tuple) or not all(
            callable(value) for value in self.onnx_passes
        ):
            raise ValueError("onnx_passes must be a tuple of graph pass callables")
