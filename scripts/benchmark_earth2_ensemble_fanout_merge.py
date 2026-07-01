# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_PATH = (
    REPO_ROOT
    / "plugins"
    / "earth2-ensemble-fanout"
    / "earth2_ensemble_fanout_support.py"
)


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "earth2_ensemble_fanout_support_benchmark",
        SUPPORT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def store_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def write_child_store(
    path: Path,
    *,
    member_ids: list[int],
    nsteps: int,
    variable_count: int,
    lat_count: int,
    lon_count: int,
) -> None:
    coords = {
        "ensemble": np.asarray(member_ids, dtype=np.int64),
        "time": np.asarray(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
        "lead_time": np.arange(nsteps + 1, dtype=np.int64).astype("timedelta64[h]"),
        "lat": np.linspace(-90.0, 90.0, lat_count, dtype=np.float32),
        "lon": np.linspace(0.0, 360.0, lon_count, endpoint=False, dtype=np.float32),
    }
    data_vars = {}
    shape = (len(member_ids), 1, nsteps + 1, lat_count, lon_count)
    for variable_index in range(variable_count):
        data = np.empty(shape, dtype=np.float32)
        for local_index, member_id in enumerate(member_ids):
            data[local_index, ...] = variable_index * 1000 + member_id
        data_vars[f"var{variable_index}"] = (
            ("ensemble", "time", "lead_time", "lat", "lon"),
            data,
        )
    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_zarr(
        path,
        mode="w",
        encoding={
            name: {"chunks": (1, 1, 1, lat_count, lon_count)} for name in data_vars
        },
    )


def build_child_stores(
    root: Path,
    *,
    nensemble: int,
    batch_size: int,
    nsteps: int,
    variable_count: int,
    lat_count: int,
    lon_count: int,
) -> list[tuple[Path, dict[str, Any]]]:
    children: list[tuple[Path, dict[str, Any]]] = []
    for batch_index, start in enumerate(range(0, nensemble, batch_size)):
        member_ids = list(range(start, min(start + batch_size, nensemble)))
        path = root / f"child-{batch_index:04d}.zarr"
        write_child_store(
            path,
            member_ids=member_ids,
            nsteps=nsteps,
            variable_count=variable_count,
            lat_count=lat_count,
            lon_count=lon_count,
        )
        children.append((path, {"batch_member_ids": member_ids}))
    return children


def benchmark_xarray_concat(
    children: list[tuple[Path, dict[str, Any]]], output: Path
) -> float:
    datasets = [xr.open_zarr(path, consolidated=False) for path, _ in children]
    start = time.perf_counter()
    try:
        combined = xr.concat(datasets, dim="ensemble").sortby("ensemble")
        combined.to_zarr(output, mode="w")
    finally:
        for dataset in datasets:
            dataset.close()
    return time.perf_counter() - start


def benchmark_lightweight_merge(
    children: list[tuple[Path, dict[str, Any]]],
    output: Path,
    *,
    nensemble: int,
) -> float:
    module = load_support_module()
    start = time.perf_counter()
    module._merge_zarr_child_stores(children, output, nensemble=nensemble)
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare earth2-ensemble-fanout aggregation strategies.",
    )
    parser.add_argument("--nensemble", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--nsteps", type=int, default=4)
    parser.add_argument("--variables", type=int, default=2)
    parser.add_argument("--lat", type=int, default=32)
    parser.add_argument("--lon", type=int, default=64)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nensemble < 1 or args.batch_size < 1:
        raise ValueError("--nensemble and --batch-size must be >= 1")

    temp_root = None
    root = args.work_dir
    if root is None:
        temp_root = tempfile.TemporaryDirectory(prefix="fanout-merge-bench-")
        root = Path(temp_root.name)
    root.mkdir(parents=True, exist_ok=True)

    try:
        children_root = root / "children"
        if children_root.exists():
            shutil.rmtree(children_root)
        children_root.mkdir()
        children = build_child_stores(
            children_root,
            nensemble=args.nensemble,
            batch_size=args.batch_size,
            nsteps=args.nsteps,
            variable_count=args.variables,
            lat_count=args.lat,
            lon_count=args.lon,
        )

        xarray_output = root / "xarray-merged.zarr"
        lightweight_output = root / "lightweight-merged.zarr"
        for output in (xarray_output, lightweight_output):
            if output.exists():
                shutil.rmtree(output)

        xarray_seconds = benchmark_xarray_concat(children, xarray_output)
        lightweight_seconds = benchmark_lightweight_merge(
            children,
            lightweight_output,
            nensemble=args.nensemble,
        )

        print(
            json.dumps(
                {
                    "nensemble": args.nensemble,
                    "batch_size": args.batch_size,
                    "nsteps": args.nsteps,
                    "variables": args.variables,
                    "lat": args.lat,
                    "lon": args.lon,
                    "child_store_count": len(children),
                    "xarray_concat_seconds": xarray_seconds,
                    "lightweight_merge_seconds": lightweight_seconds,
                    "speedup": xarray_seconds / lightweight_seconds
                    if lightweight_seconds > 0
                    else None,
                    "xarray_output_bytes": store_size_bytes(xarray_output),
                    "lightweight_output_bytes": store_size_bytes(lightweight_output),
                    "work_dir": str(root),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if temp_root is not None and not args.keep:
            temp_root.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
