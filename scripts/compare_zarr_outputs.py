#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two Zarr forecast datasets and report numerical differences."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayComparison:
    name: str
    status: str
    dims_a: list[str] | None = None
    dims_b: list[str] | None = None
    shape_a: list[int] | None = None
    shape_b: list[int] | None = None
    dtype_a: str | None = None
    dtype_b: str | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    rmse: float | None = None
    max_rel_diff: float | None = None
    nonzero_diff_count: int | None = None
    finite_mismatch_count: int | None = None
    nan_count_a: int | None = None
    nan_count_b: int | None = None
    allclose: bool | None = None
    message: str | None = None


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _load_dataset(path: Path):
    import xarray as xr

    return xr.open_zarr(path, consolidated=False)


def _is_temporal_dtype(dtype: np.dtype) -> bool:
    return dtype.kind in {"M", "m"}


def _selected_names(
    available_a: set[str], available_b: set[str], requested: list[str] | None
) -> list[str]:
    if requested:
        return sorted(set(requested))
    return sorted(available_a | available_b)


def _compare_numeric_arrays(
    name: str,
    array_a,
    array_b,
    *,
    rtol: float,
    atol: float,
) -> ArrayComparison:
    dims_a = list(getattr(array_a, "dims", []))
    dims_b = list(getattr(array_b, "dims", []))
    values_a = np.asarray(array_a.values)
    values_b = np.asarray(array_b.values)

    base = {
        "name": name,
        "dims_a": dims_a,
        "dims_b": dims_b,
        "shape_a": list(values_a.shape),
        "shape_b": list(values_b.shape),
        "dtype_a": str(values_a.dtype),
        "dtype_b": str(values_b.dtype),
    }

    if dims_a != dims_b:
        return ArrayComparison(
            **base,
            status="schema_mismatch",
            message=f"dimension order differs: {dims_a} != {dims_b}",
        )

    if values_a.shape != values_b.shape:
        return ArrayComparison(
            **base,
            status="schema_mismatch",
            message=f"shape differs: {values_a.shape} != {values_b.shape}",
        )

    if _is_temporal_dtype(values_a.dtype) or _is_temporal_dtype(values_b.dtype):
        equal = bool(np.array_equal(values_a, values_b))
        return ArrayComparison(
            **base,
            status="equal" if equal else "different",
            allclose=equal,
            message=None if equal else "temporal arrays differ",
        )

    if not (
        np.issubdtype(values_a.dtype, np.number)
        and np.issubdtype(values_b.dtype, np.number)
    ):
        equal = bool(np.array_equal(values_a, values_b))
        return ArrayComparison(
            **base,
            status="equal" if equal else "different",
            allclose=equal,
            message=None if equal else "non-numeric arrays differ",
        )

    finite_a = np.isfinite(values_a)
    finite_b = np.isfinite(values_b)
    finite_mismatch = finite_a != finite_b
    shared_finite = finite_a & finite_b
    shared_nonfinite = ~finite_a & ~finite_b

    if np.any(shared_finite):
        diff = values_b[shared_finite].astype(np.float64) - values_a[
            shared_finite
        ].astype(np.float64)
        abs_diff = np.abs(diff)
        denom = np.maximum(np.abs(values_a[shared_finite].astype(np.float64)), atol)
        rel_diff = abs_diff / denom
        max_abs_diff = _float_or_none(abs_diff.max())
        mean_abs_diff = _float_or_none(abs_diff.mean())
        rmse = _float_or_none(np.sqrt(np.mean(diff * diff)))
        max_rel_diff = _float_or_none(rel_diff.max())
        nonzero_diff_count = int(np.count_nonzero(abs_diff))
    else:
        max_abs_diff = None
        mean_abs_diff = None
        rmse = None
        max_rel_diff = None
        nonzero_diff_count = 0

    allclose = bool(
        np.array_equal(finite_a, finite_b)
        and np.allclose(
            values_a[shared_finite], values_b[shared_finite], rtol=rtol, atol=atol
        )
        and np.array_equal(
            values_a[shared_nonfinite],
            values_b[shared_nonfinite],
            equal_nan=True,
        )
    )
    return ArrayComparison(
        **base,
        status="equal" if allclose else "different",
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rmse=rmse,
        max_rel_diff=max_rel_diff,
        nonzero_diff_count=nonzero_diff_count,
        finite_mismatch_count=int(np.count_nonzero(finite_mismatch)),
        nan_count_a=int(np.count_nonzero(np.isnan(values_a))),
        nan_count_b=int(np.count_nonzero(np.isnan(values_b))),
        allclose=allclose,
    )


def _compare_dataset(
    path_a: Path,
    path_b: Path,
    *,
    variables: list[str] | None,
    include_coords: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    ds_a = _load_dataset(path_a)
    ds_b = _load_dataset(path_b)
    try:
        vars_a = set(ds_a.data_vars)
        vars_b = set(ds_b.data_vars)
        names = _selected_names(vars_a, vars_b, variables)
        comparisons: list[ArrayComparison] = []

        for name in names:
            if name not in vars_a:
                comparisons.append(
                    ArrayComparison(
                        name=name,
                        status="missing_in_a",
                        message="missing from first dataset",
                    )
                )
                continue
            if name not in vars_b:
                comparisons.append(
                    ArrayComparison(
                        name=name,
                        status="missing_in_b",
                        message="missing from second dataset",
                    )
                )
                continue
            comparisons.append(
                _compare_numeric_arrays(
                    name, ds_a[name], ds_b[name], rtol=rtol, atol=atol
                )
            )

        coord_comparisons: list[ArrayComparison] = []
        if include_coords:
            coord_names = sorted(set(ds_a.coords) | set(ds_b.coords))
            for name in coord_names:
                if name not in ds_a.coords:
                    coord_comparisons.append(
                        ArrayComparison(
                            name=name,
                            status="missing_in_a",
                            message="coordinate missing from first dataset",
                        )
                    )
                    continue
                if name not in ds_b.coords:
                    coord_comparisons.append(
                        ArrayComparison(
                            name=name,
                            status="missing_in_b",
                            message="coordinate missing from second dataset",
                        )
                    )
                    continue
                coord_comparisons.append(
                    _compare_numeric_arrays(
                        name, ds_a.coords[name], ds_b.coords[name], rtol=rtol, atol=atol
                    )
                )

        failures = [
            item
            for item in [*comparisons, *coord_comparisons]
            if item.status not in {"equal"}
        ]
        return {
            "status": "passed" if not failures else "failed",
            "path_a": str(path_a),
            "path_b": str(path_b),
            "rtol": rtol,
            "atol": atol,
            "data_variables": [asdict(item) for item in comparisons],
            "coordinates": [asdict(item) for item in coord_comparisons],
            "summary": {
                "variables_compared": len(comparisons),
                "coordinates_compared": len(coord_comparisons),
                "failure_count": len(failures),
                "missing_in_a": sum(item.status == "missing_in_a" for item in failures),
                "missing_in_b": sum(item.status == "missing_in_b" for item in failures),
                "schema_mismatch": sum(
                    item.status == "schema_mismatch" for item in failures
                ),
                "different": sum(item.status == "different" for item in failures),
            },
        }
    finally:
        ds_a.close()
        ds_b.close()


def _print_text_report(result: dict[str, Any]) -> None:
    print(f"Status: {result['status']}")
    print(f"Dataset A: {result['path_a']}")
    print(f"Dataset B: {result['path_b']}")
    print(f"Tolerance: rtol={result['rtol']} atol={result['atol']}")
    print(f"Summary: {json.dumps(result['summary'], sort_keys=True)}")

    for section_name, key in [
        ("Data Variables", "data_variables"),
        ("Coordinates", "coordinates"),
    ]:
        items = result.get(key) or []
        if not items:
            continue
        print(f"\n{section_name}:")
        for item in items:
            status = item["status"]
            name = item["name"]
            if status == "equal":
                print(
                    f"  {name}: equal max_abs={item['max_abs_diff']} "
                    f"mean_abs={item['mean_abs_diff']} rmse={item['rmse']}"
                )
            else:
                print(f"  {name}: {status} {item.get('message') or ''}".rstrip())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Zarr v3 datasets, e.g. e2s-ensemble vs earth2-ensemble-fanout outputs."
    )
    parser.add_argument("dataset_a", type=_path_arg, help="First Zarr dataset path")
    parser.add_argument("dataset_b", type=_path_arg, help="Second Zarr dataset path")
    parser.add_argument(
        "--variables",
        nargs="+",
        help="Optional variable names to compare. Defaults to union of data variables.",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--include-coords",
        action="store_true",
        help="Also compare coordinate arrays.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write the full JSON comparison report.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON report to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = _compare_dataset(
        args.dataset_a,
        args.dataset_b,
        variables=args.variables,
        include_coords=bool(args.include_coords),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

    if args.json_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_report(result)

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
