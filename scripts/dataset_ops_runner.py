#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _variables(payload: dict[str, object]) -> list[str]:
    raw = payload.get("variables")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _open_dataset(source_path: Path):
    import xarray as xr

    if source_path.suffix.lower() == ".nc":
        return xr.open_dataset(str(source_path))
    return xr.open_zarr(str(source_path))


def export_netcdf(payload: dict[str, object]) -> dict[str, object]:
    source_path = Path(str(payload["source_path"])).expanduser()
    destination_path = Path(str(payload["destination_path"])).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    variables = _variables(payload)

    if source_path.suffix.lower() == ".nc" and not variables:
        shutil.copy2(source_path, destination_path)
        return {"output_path": str(destination_path)}

    dataset = _open_dataset(source_path)
    try:
        if variables:
            dataset = dataset[variables]
        dataset.to_netcdf(str(destination_path))
    finally:
        dataset.close()

    return {"output_path": str(destination_path)}


def export_zarr_zip(payload: dict[str, object]) -> dict[str, object]:
    source_path = Path(str(payload["source_path"])).expanduser()
    destination_path = Path(str(payload["destination_path"])).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    variables = _variables(payload)

    if variables or source_path.suffix.lower() == ".nc":
        dataset = _open_dataset(source_path)
        with tempfile.TemporaryDirectory(
            prefix="physicsnemo-serve-zarr-export-"
        ) as tmp_dir:
            subset_path = Path(tmp_dir) / "subset.zarr"
            try:
                if variables:
                    dataset = dataset[variables]
                dataset.to_zarr(str(subset_path), mode="w")
            finally:
                dataset.close()
            zip_directory(subset_path, destination_path)
        return {"output_path": str(destination_path)}

    zip_directory(source_path, destination_path)
    return {"output_path": str(destination_path)}


def zip_directory(source_dir: Path, destination_zip: Path) -> None:
    with zipfile.ZipFile(
        destination_zip, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(source_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run built-in dataset export operations for PhysicsNeMo Serve."
    )
    parser.add_argument(
        "--op", required=True, choices=["export_netcdf", "export_zarr_zip"]
    )
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("dataset op runner payload must be a JSON object")

        if args.op == "export_netcdf":
            result = export_netcdf(payload)
        elif args.op == "export_zarr_zip":
            result = export_zarr_zip(payload)
        else:
            raise ValueError(f"unsupported dataset op: {args.op}")

        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
