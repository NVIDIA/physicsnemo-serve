# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow registry helpers for parity tooling."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

WorkflowRunner = Callable[[dict[str, Any], Path], None]
DEFAULT_WORKFLOW_ID = "deterministic_io_small_v1"
DEFAULT_PARALLEL_COORD_NAMES = ("time", "lead_time", "ensemble")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_step_delta(step_delta: str) -> Any:
    import numpy as np

    _require(
        isinstance(step_delta, str) and len(step_delta) >= 2,
        "step_delta must be like '1h' or '30m'",
    )
    unit_map = {"s": "s", "m": "m", "h": "h", "d": "D"}
    unit = step_delta[-1]
    _require(unit in unit_map, f"unsupported step_delta unit '{unit}'")
    try:
        value = int(step_delta[:-1])
    except ValueError as exc:
        raise ValueError(
            "step_delta value must be integer-prefixed, e.g. '6h'"
        ) from exc
    _require(value > 0, "step_delta integer prefix must be > 0")
    return np.timedelta64(value, unit_map[unit])


def _parse_start_time_to_datetime64_ns(raw_value: Any) -> Any:
    import numpy as np

    if isinstance(raw_value, np.datetime64):
        return raw_value.astype("datetime64[ns]")
    _require(
        isinstance(raw_value, str) and raw_value.strip(),
        "start_times entries must be non-empty strings",
    )
    start_time_str = raw_value.strip()
    iso_input = (
        start_time_str[:-1] + "+00:00"
        if start_time_str.endswith("Z")
        else start_time_str
    )
    try:
        parsed = datetime.fromisoformat(iso_input)
    except ValueError:
        return np.datetime64(start_time_str, "ns")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(parsed.isoformat(), "ns")


def _discover_earth2studio_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "earth2studio" / "__init__.py").exists() and (
            parent / "pyproject.toml"
        ).exists():
            return parent
    raise RuntimeError(
        "could not discover earth2studio repository root from current file location"
    )


def _ensure_earth2studio_import_path() -> None:
    root = _discover_earth2studio_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _resolve_parallel_coords(
    case_spec: dict[str, Any], total_coords: OrderedDict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    explicit = case_spec.get("parallel_coords")
    if isinstance(explicit, dict) and explicit:
        return {key: np.asarray(value) for key, value in explicit.items()}
    resolved: dict[str, Any] = {}
    for name in DEFAULT_PARALLEL_COORD_NAMES:
        if name in total_coords:
            resolved[name] = total_coords[name]
    return resolved


def _build_workflow_inputs(
    case_spec: dict[str, Any],
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    import numpy as np

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        torch = None

    seed = int(case_spec["deterministic_seed"])
    start_times = case_spec["start_times"]
    _require(
        isinstance(start_times, list) and start_times,
        "start_times must be a non-empty list",
    )
    time_values = np.asarray(
        [_parse_start_time_to_datetime64_ns(value) for value in start_times],
        dtype="datetime64[ns]",
    )
    num_steps = int(case_spec["num_steps"])
    _require(num_steps > 0, "num_steps must be > 0")
    step_delta = _parse_step_delta(str(case_spec["step_delta"])).astype(
        "timedelta64[ns]"
    )
    lead_time_values = np.asarray(
        [i * step_delta for i in range(num_steps)], dtype="timedelta64[ns]"
    )
    # Use float64 for coordinate axes so Python and Rust runners serialize identical coord payloads.
    lat_values = np.linspace(-90, 90, 8, dtype=np.float64)
    lon_values = np.linspace(0, 360, 16, endpoint=False, dtype=np.float64)
    total_coords: OrderedDict[str, Any] = OrderedDict(
        {
            "time": time_values,
            "lead_time": lead_time_values,
            "lat": lat_values,
            "lon": lon_values,
        }
    )
    output_arrays = case_spec["output_array_names"]
    _require(
        isinstance(output_arrays, list) and output_arrays,
        "output_array_names must be a non-empty list",
    )
    rng = np.random.default_rng(seed)
    shape = (len(time_values), len(lead_time_values), len(lat_values), len(lon_values))
    tensors: dict[str, Any] = {}
    for index, array_name in enumerate(output_arrays):
        _require(
            isinstance(array_name, str) and array_name,
            "output_array_names must contain non-empty strings",
        )
        data = rng.standard_normal(shape, dtype=np.float32) + (index * 0.01)
        if torch is None:
            tensors[array_name] = data.copy()
        else:
            tensors[array_name] = torch.from_numpy(data.copy())
    return total_coords, tensors


def _import_earth2studio_backends() -> tuple[Any, Any]:
    _ensure_earth2studio_import_path()
    from earth2studio.io.async_zarr import AsyncZarrBackend
    from earth2studio.io.zarr import ZarrBackend

    return ZarrBackend, AsyncZarrBackend


def _run_with_py_sync_backend(case_spec: dict[str, Any], dataset_path: Path) -> None:
    import numpy as np

    ZarrBackend, _ = _import_earth2studio_backends()
    total_coords, tensors = _build_workflow_inputs(case_spec)
    chunks = OrderedDict(
        {
            "time": 1,
            "lead_time": 1,
            "lat": len(total_coords["lat"]),
            "lon": len(total_coords["lon"]),
        }
    )
    backend = ZarrBackend(file_name=str(dataset_path), chunks=chunks)
    array_names = list(tensors.keys())
    backend.add_array(total_coords, array_names)
    lead_time_axis = total_coords["lead_time"]
    lead_time_dtype = getattr(lead_time_axis, "dtype", None)
    for step_index, lead_time in enumerate(total_coords["lead_time"]):
        step_coords = total_coords.copy()
        if lead_time_dtype is None:
            step_coords["lead_time"] = [lead_time]
        else:
            step_coords["lead_time"] = np.asarray([lead_time], dtype=lead_time_dtype)
        step_tensors = [
            tensor[:, step_index : step_index + 1, :, :] for tensor in tensors.values()
        ]
        backend.write(step_tensors, step_coords, array_names)
    backend.close()


def _run_with_py_async_backend(case_spec: dict[str, Any], dataset_path: Path) -> None:
    import numpy as np

    _, AsyncZarrBackend = _import_earth2studio_backends()
    total_coords, tensors = _build_workflow_inputs(case_spec)
    parallel_coords = _resolve_parallel_coords(case_spec, total_coords)
    backend = AsyncZarrBackend(
        str(dataset_path),
        parallel_coords=parallel_coords,
        blocking=True,
    )
    array_names = list(tensors.keys())
    lead_time_axis = total_coords["lead_time"]
    lead_time_dtype = getattr(lead_time_axis, "dtype", None)
    for step_index, lead_time in enumerate(total_coords["lead_time"]):
        step_coords = total_coords.copy()
        if lead_time_dtype is None:
            step_coords["lead_time"] = [lead_time]
        else:
            step_coords["lead_time"] = np.asarray([lead_time], dtype=lead_time_dtype)
        step_tensors = [
            tensor[:, step_index : step_index + 1, :, :] for tensor in tensors.values()
        ]
        backend.write(step_tensors, step_coords, array_names)
    backend.close()


def _run_with_rust_backend(case_spec: dict[str, Any], dataset_path: Path) -> None:
    _ensure_earth2studio_import_path()
    try:
        import e2s_zarr_io  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Rust backend runner requires the e2s_zarr_io Python extension module. "
            "Build and install python bindings before running rust parity lane."
        ) from exc

    if not hasattr(e2s_zarr_io, "E2sZarrIoBackend"):
        raise RuntimeError(
            "e2s_zarr_io Python bindings are installed but E2sZarrIoBackend is not exposed."
        )

    import numpy as np

    total_coords, tensors = _build_workflow_inputs(case_spec)
    array_names = list(tensors.keys())
    init_kwargs: dict[str, Any] = {"file_name": str(dataset_path)}
    parallel_coords = _resolve_parallel_coords(case_spec, total_coords)
    if parallel_coords:
        init_kwargs["parallel_coords"] = parallel_coords
    for key in ("zarr_format", "chunk_key_encoding", "chunk_key_separator"):
        value = case_spec.get(key)
        if isinstance(value, str) and value:
            init_kwargs[key] = value

    backend = e2s_zarr_io.E2sZarrIoBackend(**init_kwargs)
    backend.add_array(total_coords, array_names)
    lead_time_axis = total_coords["lead_time"]
    lead_time_dtype = getattr(lead_time_axis, "dtype", None)
    for step_index, lead_time in enumerate(total_coords["lead_time"]):
        step_coords = total_coords.copy()
        if lead_time_dtype is None:
            step_coords["lead_time"] = [lead_time]
        else:
            step_coords["lead_time"] = np.asarray([lead_time], dtype=lead_time_dtype)
        step_arrays = []
        for tensor in tensors.values():
            step_tensor = tensor[:, step_index : step_index + 1, :, :]
            if hasattr(step_tensor, "detach"):
                step_np = step_tensor.detach().cpu().numpy()
            else:
                step_np = np.asarray(step_tensor)
            # Rust host-array ingestion requires C-contiguous array_interface strides.
            step_arrays.append(np.ascontiguousarray(step_np))
        backend.write(step_arrays, step_coords, array_names)
    backend.close()


def run_default_workflow(case_spec: dict[str, Any], dataset_path: Path) -> None:
    """Run default deterministic workflow using backend kind from case_spec."""
    backend_kind = case_spec.get("backend_kind")
    _require(
        isinstance(backend_kind, str) and backend_kind,
        "case_spec.backend_kind must be set by runner",
    )
    if backend_kind == "py_sync":
        _run_with_py_sync_backend(case_spec, dataset_path)
        return
    if backend_kind == "py_async":
        _run_with_py_async_backend(case_spec, dataset_path)
        return
    if backend_kind == "rust":
        _run_with_rust_backend(case_spec, dataset_path)
        return
    raise ValueError(f"unsupported backend_kind in workflow: {backend_kind}")


def create_default_workflow_catalog() -> "WorkflowCatalog":
    """Create workflow catalog with default deterministic workflow adapter."""
    catalog = WorkflowCatalog()
    catalog.register(DEFAULT_WORKFLOW_ID, run_default_workflow)
    return catalog


class WorkflowCatalog:
    """In-memory workflow runner catalog keyed by workflow_id."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowRunner] = {}

    def register(self, workflow_id: str, runner: WorkflowRunner) -> None:
        """Register workflow runner."""
        if not workflow_id:
            raise ValueError("workflow_id must be non-empty")
        self._workflows[workflow_id] = runner

    def resolve(self, workflow_id: str) -> WorkflowRunner:
        """Resolve workflow runner by id."""
        if workflow_id not in self._workflows:
            raise KeyError(f"unknown workflow_id: {workflow_id}")
        return self._workflows[workflow_id]

    def run(self, case_spec: dict[str, Any], dataset_path: Path) -> None:
        """Run workflow based on case_spec.workflow_id."""
        workflow_id = case_spec.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise ValueError("case_spec.workflow_id must be a non-empty string")
        runner = self.resolve(workflow_id)
        runner(case_spec, dataset_path)

    def run_with_backend(
        self,
        backend_kind: str,
        case_spec: dict[str, Any],
        dataset_path: Path,
    ) -> None:
        """Run workflow after injecting backend kind into case spec."""
        case_spec_with_backend = dict(case_spec)
        case_spec_with_backend["backend_kind"] = backend_kind
        self.run(case_spec_with_backend, dataset_path)
