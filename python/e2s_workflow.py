# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Earth2Workflow base class that bridges earth2studio's workflow pattern to PhysicsNeMo Serve plugins.

Plugin authors subclass Earth2Workflow and implement __call__(self, io, **kwargs).
The base class handles output artifact creation, IO backend setup, and PhysicsNeMo Serve integration.

Example:
    class MyWorkflow(Earth2Workflow):
        def __init__(self, model_type="dlwp"):
            super().__init__()
            self.model = DLWP.load_model(DLWP.load_default_package())
            self.data = GFS()

        def __call__(self, io, start_time=["2024-01-01T00:00:00"], num_steps=6):
            run.deterministic(start_time, num_steps, self.model, self.data, io)
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from plugin_sdk import (
    ExecutionContext,
    PluginWorkflow,
    _cleanup_fsspec_client as _cleanup_plugin_sdk_fsspec_client,
    _close_cached_fsspec_sessions,
)

# Supported IO backend types and their file extensions / media types.
_IO_REGISTRY = {
    "zarr": {
        "filename": "forecast.zarr",
        "media_type": "application/x-zarr",
        "factory": lambda path, **kw: create_zarr_backend(path, **kw),
    },
    "netcdf4": {
        "filename": "forecast.nc",
        "media_type": "application/x-netcdf4",
        "factory": lambda path, **kw: _make_netcdf4_backend(path, **kw),
    },
    "xarray": {
        "filename": "forecast.nc",
        "media_type": "application/x-netcdf4",
        "factory": lambda path, **kw: _make_xarray_backend(**kw),
    },
}

logger = logging.getLogger(__name__)

_ZARR_BACKEND_ENV = "PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND"
_DEFAULT_RUST_PARALLEL_CHUNKS = {
    "ensemble": 1,
    "time": 1,
    "lead_time": 1,
    "variable": 1,
}
_SUPPORTED_RUST_ZARR_KWARGS = {
    "parallel_coords",
    "default_parallel_coord_names",
    "zarr_format",
    "chunk_key_encoding",
    "chunk_key_separator",
    "buffer_sizing_policy",
    "model_profile_hint",
    "min_hot_slab_buffers",
    "max_warm_to_hot_ratio",
    "hot_slab_buffers",
    "warm_slab_buffers",
    "hot_slab_ready_timeout_seconds",
    "warm_slab_warmup_policy",
    "warm_slab_failure_policy",
    "max_pool_buffers",
    "max_pool_bytes",
    "pool_buffer_bytes",
    "pool_alignment",
    "pin_pooled_slabs",
    "cuda_register_pool_if_available",
    "cuda_register_each_slab_once",
    "max_transient_buffer_bytes",
    "max_inflight_transient_bytes",
    "close_lease_timeout_seconds",
    "queue_capacity",
    "num_threads",
    "require_host_array_interface",
    "fsync_policy",
}


def _bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _selected_zarr_backend() -> str:
    selected = os.environ.get(_ZARR_BACKEND_ENV, "rust").strip().lower()
    if selected in ("", "rust"):
        return "rust"
    if selected == "python":
        return "python"
    raise ValueError(
        f"Unsupported {_ZARR_BACKEND_ENV}={selected!r}; expected 'rust' or 'python'"
    )


def _ensure_earth2studio_zarr_v3(value: Any) -> None:
    if value == 3:
        return
    if isinstance(value, str) and value.strip().lower() in {"3", "v3"}:
        return
    raise ValueError("Earth2Studio workflows only support Zarr v3")


def _make_python_zarr_backend(path: str, **kwargs):
    from earth2studio.io import ZarrBackend

    backend_kwargs = dict(kwargs.pop("backend_kwargs", {}) or {})
    zarr_format = kwargs.pop("zarr_format", None)
    format_candidates = [
        candidate
        for candidate in (
            zarr_format,
            backend_kwargs.get("zarr_format"),
            backend_kwargs.get("zarr_version"),
        )
        if candidate is not None
    ]
    for candidate in format_candidates:
        _ensure_earth2studio_zarr_v3(candidate)
    backend_kwargs.pop("zarr_version", None)
    backend_kwargs["zarr_format"] = 3
    backend_kwargs.setdefault("overwrite", True)
    return ZarrBackend(path, backend_kwargs=backend_kwargs, **kwargs)


def create_zarr_backend(path: str, **kwargs):
    """Create a Zarr backend using PhysicsNeMo Serve's Rust/Python selector."""
    if _selected_zarr_backend() == "python":
        return _make_python_zarr_backend(path, **kwargs)
    return _RustZarrBackendAdapter(path, **kwargs)


def _make_zarr_backend(path: str, **kwargs):
    return create_zarr_backend(path, **kwargs)


def _make_netcdf4_backend(path: str, **kwargs):
    from earth2studio.io import NetCDF4Backend

    return NetCDF4Backend(path, **kwargs)


def _make_xarray_backend(**kwargs):
    from earth2studio.io import XarrayBackend

    return XarrayBackend(**kwargs)


def _cleanup_fsspec_client(candidate: Any) -> bool:
    """Close an owned fsspec-style async client when the object exposes one."""
    return _cleanup_plugin_sdk_fsspec_client(getattr(candidate, "fs", candidate))


def _normalize_coord_values_for_rust(value: Any) -> Any:
    import numpy as np

    if isinstance(value, datetime | timedelta):
        value = [value]

    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]")
    if arr.dtype.kind == "m":
        return arr.astype("timedelta64[ns]")
    if arr.dtype.kind == "O" and arr.size > 0:
        first = arr.flat[0]
        if isinstance(first, datetime):
            return arr.astype("datetime64[ns]")
        if isinstance(first, timedelta):
            return arr.astype("timedelta64[ns]")
    if isinstance(value, np.ndarray) and arr.dtype.kind in "iuf":
        return arr
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_coord_map_for_rust(coords: Any) -> dict[str, Any]:
    return {
        str(key): _normalize_coord_values_for_rust(value)
        for key, value in coords.items()
    }


def _normalize_array_names_for_rust(array_names: Any) -> list[str]:
    if isinstance(array_names, str):
        return [array_names]
    if hasattr(array_names, "tolist"):
        array_names = array_names.tolist()
    if isinstance(array_names, tuple):
        array_names = list(array_names)
    if isinstance(array_names, list):
        return [str(name) for name in array_names]
    return [str(array_names)]


def _normalize_write_arrays_for_rust(arrays: Any) -> list[Any]:
    import numpy as np

    if isinstance(arrays, tuple):
        arrays = list(arrays)
    if not isinstance(arrays, list):
        arrays = [arrays]

    normalized = []
    for arr in arrays:
        if hasattr(arr, "detach"):
            tensor = arr.detach()
            if bool(getattr(tensor, "is_cuda", False)):
                if bool(getattr(tensor, "is_contiguous", lambda: True)()) is False:
                    tensor = tensor.contiguous()
                normalized.append(tensor)
            else:
                normalized.append(np.ascontiguousarray(tensor.cpu().numpy()))
        else:
            normalized.append(np.ascontiguousarray(arr))
    return normalized


class _RustZarrBackendAdapter:
    """Earth2Studio-compatible wrapper around the e2s_zarr_io Python extension."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        try:
            import e2s_zarr_io
        except ImportError as exc:
            raise RuntimeError(
                "PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND is set to Rust or unset, but "
                "the e2s_zarr_io Python extension is not importable. Install "
                "e2s_zarr_io or set PHYSICSNEMO_SERVE_E2S_ZARR_BACKEND=python."
            ) from exc

        backend_cls = getattr(e2s_zarr_io, "E2sZarrIoBackend", None)
        if backend_cls is None:
            raise RuntimeError(
                "e2s_zarr_io is importable but E2sZarrIoBackend is not exposed"
            )

        self.file_name = str(path)
        self._backend_cls = backend_cls
        self._backend: Any | None = None
        self._closed = False
        self._root: Any | None = None

        backend_kwargs = dict(kwargs.pop("backend_kwargs", {}) or {})
        overwrite = backend_kwargs.pop("overwrite", False)
        backend_zarr_format = backend_kwargs.pop("zarr_format", None)
        backend_zarr_version = backend_kwargs.pop("zarr_version", None)
        if not isinstance(overwrite, bool):
            backend_kwargs["overwrite"] = overwrite
        unsupported_backend_kwargs = dict(backend_kwargs)
        if unsupported_backend_kwargs:
            raise ValueError(
                "Rust Zarr backend does not support earth2studio backend_kwargs "
                f"{sorted(unsupported_backend_kwargs)}"
            )
        if overwrite is True:
            path = Path(self.file_name)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

        chunks = kwargs.pop("chunks", None)
        self._chunks_explicit = chunks is not None
        self._chunks = dict(chunks or _DEFAULT_RUST_PARALLEL_CHUNKS)
        unsupported_kwargs = sorted(set(kwargs) - _SUPPORTED_RUST_ZARR_KWARGS)
        if unsupported_kwargs:
            raise ValueError(
                f"Rust Zarr backend does not support ZarrBackend kwargs: {unsupported_kwargs}"
            )
        format_candidates = [
            candidate
            for candidate in (
                kwargs.get("zarr_format"),
                backend_zarr_format,
                backend_zarr_version,
            )
            if candidate is not None
        ]
        for candidate in format_candidates:
            _ensure_earth2studio_zarr_v3(candidate)
        kwargs["zarr_format"] = "v3"
        self._explicit_parallel_coords = kwargs.pop("parallel_coords", None)
        self._rust_kwargs = dict(kwargs)
        self._rust_kwargs.setdefault("fsync_policy", "never")

    @staticmethod
    def _axis_len(value: Any) -> int:
        try:
            return len(value)
        except TypeError:
            return 1

    def _parallel_coords_for(self, coords: dict[str, Any]) -> dict[str, Any] | None:
        if self._explicit_parallel_coords is not None:
            return _normalize_coord_map_for_rust(self._explicit_parallel_coords)

        unsupported_chunks = {}
        parallel_coords = {
            name: coords[name]
            for name, chunk_size in self._chunks.items()
            if chunk_size == 1 and name in coords
        }
        if self._chunks_explicit:
            for name, chunk_size in self._chunks.items():
                if name not in coords or chunk_size == 1:
                    continue
                axis_len = self._axis_len(coords[name])
                if int(chunk_size) != axis_len:
                    unsupported_chunks[name] = {
                        "chunk_size": int(chunk_size),
                        "axis_len": axis_len,
                    }
            if unsupported_chunks:
                raise ValueError(
                    "Rust Zarr backend supports explicit chunks only as 1 or "
                    f"the full registered axis length; unsupported chunks: {unsupported_chunks}"
                )
            if not parallel_coords and any(name in coords for name in self._chunks):
                raise ValueError(
                    "Rust Zarr backend explicit chunks must include at least one "
                    "registered dimension with chunk size 1"
                )
        return parallel_coords or None

    def _ensure_backend(self, coords: dict[str, Any]) -> Any:
        if self._backend is not None:
            return self._backend

        init_kwargs = {"file_name": self.file_name, **self._rust_kwargs}
        parallel_coords = self._parallel_coords_for(coords)
        if parallel_coords:
            init_kwargs["parallel_coords"] = parallel_coords
            init_kwargs.setdefault(
                "default_parallel_coord_names", list(parallel_coords)
            )
        self._backend = self._backend_cls(**init_kwargs)
        return self._backend

    def add_array(self, coords: Any, array_name: Any, data: Any = None) -> None:
        normalized_coords = _normalize_coord_map_for_rust(coords)
        normalized_names = _normalize_array_names_for_rust(array_name)
        backend = self._ensure_backend(normalized_coords)
        if data is None:
            backend.add_array(normalized_coords, normalized_names)
        else:
            backend.add_array(
                normalized_coords,
                normalized_names,
                data=_normalize_write_arrays_for_rust(data),
            )

    def write(self, x: Any, coords: Any, array_name: Any) -> None:
        normalized_coords = _normalize_coord_map_for_rust(coords)
        normalized_names = _normalize_array_names_for_rust(array_name)
        backend = self._ensure_backend(normalized_coords)
        backend.write(
            _normalize_write_arrays_for_rust(x),
            normalized_coords,
            normalized_names,
        )

    def close(self, timeout_seconds: float | None = None) -> Any:
        if self._backend is None:
            self._closed = True
            Path(self.file_name).mkdir(parents=True, exist_ok=True)
            return None
        if not self.is_closed():
            result = self._backend.close(timeout_seconds)
            self._closed = True
            return result
        self._closed = True
        return None

    def finalize(self) -> "_RustZarrBackendAdapter":
        self.close()
        return self

    def is_closed(self) -> bool:
        if self._backend is None:
            return self._closed
        is_closed = getattr(self._backend, "is_closed", None)
        return bool(is_closed()) if callable(is_closed) else self._closed

    def last_write_timing(self) -> Any:
        if self._backend is None:
            return None
        getter = getattr(self._backend, "last_write_timing", None)
        return getter() if callable(getter) else None

    def _open_root(self) -> Any:
        if self._root is not None:
            return self._root

        self.close()
        import zarr

        storage = getattr(zarr, "storage", None)
        local_store_cls = getattr(storage, "LocalStore", None)
        if local_store_cls is not None:
            self._root = zarr.open_group(
                store=local_store_cls(str(self.file_name)),
                mode="r",
            )
        else:
            self._root = zarr.open_group(str(self.file_name), mode="r")
        return self._root

    def __getitem__(self, key: str) -> Any:
        return self._open_root()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._open_root()

    def __iter__(self):
        return iter(self._open_root())

    def keys(self):
        return self._open_root().keys()

    def __len__(self) -> int:
        return len(self._open_root())

    def __fspath__(self) -> str:
        return self.file_name

    def __str__(self) -> str:
        return self.file_name


class Earth2Workflow(PluginWorkflow):
    """Base class for earth2studio workflow plugins.

    Subclasses implement ``__call__(self, io, **kwargs)`` following the same
    pattern as earth2studio's ``Earth2Workflow``.  The base class:

    1. Creates the output artifact and IO backend (zarr, netcdf4, or xarray).
    2. Extracts ``__call__`` parameters (excluding ``io``) as the ``input_model``.
    3. Calls ``self(io, **inputs)`` with the coerced user inputs.
    4. Returns ``{"dataset_path": "<path>"}`` as the result.

    Class attributes:
        io_backend: IO backend type — "zarr" (default), "netcdf4", or "xarray".
        io_kwargs: Extra keyword arguments passed to the IO backend factory.
        artifact_name: Name for the registered output artifact.
    """

    io_backend: str = "zarr"
    io_kwargs: dict[str, Any] | None = None
    artifact_name: str = "forecast_dataset"
    cache_preserve_attributes: tuple[str, ...] = ()

    @staticmethod
    def _staged_dataset_path(dataset_path: Path) -> Path:
        return dataset_path.with_name(f".{dataset_path.name}.tmp-{uuid.uuid4().hex}")

    @staticmethod
    def _remove_output_path(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def _promote_output_path(self, active_path: Path, final_path: Path) -> None:
        if active_path == final_path:
            return

        if final_path.exists():
            logger.info(
                "Final Earth2 output already exists at %s; discarding staged output %s",
                final_path,
                active_path,
            )
            self._remove_output_path(active_path)
            return

        try:
            active_path.replace(final_path)
        except OSError:
            if final_path.exists():
                logger.info(
                    "Final Earth2 output won a promotion race at %s; discarding staged output %s",
                    final_path,
                    active_path,
                )
                self._remove_output_path(active_path)
                return
            raise

    @classmethod
    def _build_input_model(cls):
        """Auto-generate a dataclass input_model from __call__ signature."""
        import dataclasses
        from typing import get_type_hints

        sig = inspect.signature(cls.__call__)
        # Resolve type hints individually — skip unresolvable ones (e.g., IOBackend)
        try:
            hints = get_type_hints(cls.__call__)
        except Exception:
            hints = {}
            globalns = getattr(sys.modules.get(cls.__module__, None), "__dict__", {})
            for name, param in sig.parameters.items():
                if (
                    name in ("self", "io")
                    or param.annotation is inspect.Parameter.empty
                ):
                    continue
                try:
                    hints[name] = (
                        eval(param.annotation, globalns)
                        if isinstance(param.annotation, str)
                        else param.annotation
                    )
                except Exception:
                    pass
        fields = []
        for name, param in sig.parameters.items():
            if name in ("self", "io"):
                continue
            annotation = hints.get(name, Any)
            if param.default is inspect.Parameter.empty:
                fields.append((name, annotation))
            elif isinstance(param.default, (list, dict, set)):
                default_value = param.default
                fields.append(
                    (
                        name,
                        annotation,
                        dataclasses.field(
                            default_factory=lambda d=default_value: (
                                list(d) if isinstance(d, list) else type(d)(d)
                            )
                        ),
                    )
                )
            else:
                fields.append(
                    (name, annotation, dataclasses.field(default=param.default))
                )

        if not fields:
            return None

        return dataclasses.make_dataclass(f"{cls.__name__}Input", fields)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-generate input_model from __call__ signature if not explicitly set
        if "input_model" not in cls.__dict__ and "__call__" in cls.__dict__:
            cls.input_model = cls._build_input_model()

    def create_io(self, dataset_path: str):
        """Create the IO backend. Override for fully custom IO setup."""
        entry = _IO_REGISTRY.get(self.io_backend)
        if entry is None:
            raise ValueError(
                f"Unknown io_backend '{self.io_backend}'. "
                f"Supported: {list(_IO_REGISTRY.keys())}"
            )
        return entry["factory"](dataset_path, **(self.io_kwargs or {}))

    def create_zarr_io(self, dataset_path: str, **kwargs: Any):
        """Create a Zarr IO backend using the configured PhysicsNeMo Serve backend selector."""
        return create_zarr_backend(dataset_path, **kwargs)

    def finalize_zarr_output(self, io: Any) -> Any:
        """Flush/consolidate Zarr output before promotion or plot reads."""
        finalizer = getattr(io, "finalize", None)
        if callable(finalizer):
            finalized_io = finalizer()
        else:
            import zarr

            zarr.consolidate_metadata(str(self.output_dataset_path))
            finalized_io = io
        self._zarr_output_finalized = True
        return finalized_io

    @staticmethod
    def _device_is_cuda(device: str | None = None) -> bool:
        if device is not None and not str(device).strip().lower().startswith("cuda"):
            return False

        try:
            import torch
        except ImportError:
            return False

        cuda = getattr(torch, "cuda", None)
        return bool(
            cuda is not None and hasattr(cuda, "is_available") and cuda.is_available()
        )

    @staticmethod
    def _offload_module(module: Any) -> None:
        """Move a torch.nn.Module to the meta device to release all GPU tensors.

        Unlike ``module.cpu()`` this avoids allocating real CPU memory for
        large checkpoint tensors -- the meta device creates zero-storage
        placeholders so the original CUDA tensors become unreferenced
        immediately.  Falls back to ``module.cpu()`` when the meta device
        is unavailable (PyTorch < 2.0).
        """
        import torch

        if not isinstance(module, torch.nn.Module):
            return
        try:
            module.to(device="meta")
        except (RuntimeError, NotImplementedError):
            try:
                module.cpu()
            except Exception:
                pass

    @classmethod
    def _release_models(cls, *models: Any) -> None:
        """Offload modules from GPU and delete references.

        Call this on every model / nn.Module created during a request
        *before* ``gc.collect`` + ``empty_cache`` so the CUDA tensors
        are actually unreferenced when the allocator tries to reclaim
        them.
        """
        import torch

        torch_nn = getattr(torch, "nn", None)
        module_type = getattr(torch_nn, "Module", None)
        if module_type is None:
            return

        for model in models:
            if model is None:
                continue
            if isinstance(model, module_type):
                cls._offload_module(model)
            inner = getattr(model, "px_model", None)
            if inner is not None and isinstance(inner, module_type):
                cls._offload_module(inner)

    def _log_cuda_memory_snapshot(
        self, stage: str, *, device: str | None = None
    ) -> None:
        if not self._device_is_cuda(device=device):
            return

        try:
            import torch
        except ImportError:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return

        try:
            allocated_bytes = int(cuda.memory_allocated())
            reserved_bytes = int(cuda.memory_reserved())
            peak_allocated_bytes = int(cuda.max_memory_allocated())
        except Exception:
            logger.debug(
                "Failed to read Earth2 CUDA memory snapshot for workflow %s stage %s",
                type(self).__name__,
                stage,
                exc_info=True,
            )
            return

        logger.info(
            "Earth2 CUDA memory snapshot: workflow=%s stage=%s "
            "allocated=%.1fMiB reserved=%.1fMiB peak_allocated=%.1fMiB",
            type(self).__name__,
            stage,
            _bytes_to_mib(allocated_bytes),
            _bytes_to_mib(reserved_bytes),
            _bytes_to_mib(peak_allocated_bytes),
        )

    def _cleanup_torch_runtime(self, device: str | None = None) -> None:
        for _ in range(3):
            gc.collect()

        if not self._device_is_cuda(device=device):
            return

        import torch

        synchronize = getattr(torch.cuda, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize()
            except Exception:
                pass

        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()

        torch.cuda.empty_cache()

    def _clear_attributes(self, *attribute_names: str) -> None:
        """Drop workflow-owned references so Python can reclaim them sooner."""
        for attribute_name in attribute_names:
            if attribute_name in self.__dict__:
                self.__dict__[attribute_name] = None

    def run(self, inputs: Any, ctx: ExecutionContext) -> dict[str, Any]:
        entry = _IO_REGISTRY.get(self.io_backend, _IO_REGISTRY["zarr"])
        filename = entry["filename"]
        media_type = entry["media_type"]

        dataset_path = Path(
            ctx.outputs.create(
                self.artifact_name,
                filename=filename,
                media_type=media_type,
                primary=True,
            )
        )
        active_dataset_path = (
            self._staged_dataset_path(dataset_path)
            if self.io_backend == "zarr"
            else dataset_path
        )
        self._staged_output_path = (
            active_dataset_path if active_dataset_path != dataset_path else None
        )
        self._zarr_output_finalized = False
        self.output_dataset_path = active_dataset_path
        self.final_output_dataset_path = dataset_path

        io = self.create_io(str(active_dataset_path))

        # Expose output_dir so __call__ can write metadata/plots alongside the dataset
        self.output_dir = ctx.run_dir

        # Extract input fields as kwargs for __call__
        if hasattr(inputs, "__dataclass_fields__"):
            kwargs = {f: getattr(inputs, f) for f in inputs.__dataclass_fields__}
        elif isinstance(inputs, dict):
            kwargs = dict(inputs)
        else:
            kwargs = {}

        self(io, **kwargs)
        if self.io_backend == "zarr" and not self._zarr_output_finalized:
            self.finalize_zarr_output(io)
        self._promote_output_path(active_dataset_path, dataset_path)
        self._staged_output_path = None
        self.output_dataset_path = dataset_path

        return {"dataset_path": str(dataset_path)}

    def cleanup(self) -> None:
        staged_output_path = getattr(self, "_staged_output_path", None)
        if staged_output_path is not None:
            try:
                self._remove_output_path(staged_output_path)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception(
                    "Failed to remove staged Earth2 output '%s' on %s",
                    staged_output_path,
                    type(self).__name__,
                )
            finally:
                self._staged_output_path = None

        cleaned_ids: set[int] = set()
        for attribute_name, candidate in self.__dict__.items():
            candidate_id = id(candidate)
            if candidate_id in cleaned_ids:
                continue

            try:
                if _cleanup_fsspec_client(candidate):
                    cleaned_ids.add(candidate_id)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception(
                    "Failed to clean up Earth2 workflow resource '%s' on %s",
                    attribute_name,
                    type(self).__name__,
                )

        _close_cached_fsspec_sessions()

    def cleanup_request(self) -> None:
        preserved = {
            name: self.__dict__[name]
            for name in self.cache_preserve_attributes
            if name in self.__dict__
        }
        for name in preserved:
            self.__dict__[name] = None
        try:
            self.cleanup()
        finally:
            for name, value in preserved.items():
                self.__dict__[name] = value
