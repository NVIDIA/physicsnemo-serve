"""Field-level contract for the pinned CFD-provider Transolver reference.

This contract is separate from the bounded native SDK example and from Python
REST/direct report parity. Raw predictions do not imply report equivalence.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qa.scripts.cfd_parity_contract import (
    ParityContractError,
    read_json_object,
    normalize_report,
    sha256_file,
    validate_profile,
    validate_request,
)


PROVIDER_COMMIT = "921f14dc2ac14c04aabffaba3290deb792379dd8"
MODEL_REVISION = "96477aeb86d24c26ccf0797bca1b3851268017d0"
PACKAGE = f"hf://nvidia/transolver_drivaerml@{MODEL_REVISION}"
ASSET_PINS = {
    "checkpoint": {
        "filename": "Transolver.0.501.mdlus",
        "sha256": "eb98f399a050a8f8a24919335c61642e4a835bd4044f7e21abec231aa31fd82c",
        "size_bytes": 39188235,
    },
    "statistics": {
        "filename": "global_stats.json",
        "sha256": "c4b19eaca6158219570232ce094e6ef48b8705eb727b335441a310c7479b6b25",
        "size_bytes": 1375,
    },
}
CASE_PINS = {
    "run_1": {
        "sha256": "01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40",
        "size_bytes": 659606189,
        "geometry_sha256": "411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2",
        "geometry_size_bytes": 142385186,
    }
}
CASE_POINT_COUNTS = {"run_1": 8828095}
ARRAY_WIDTHS = {"fx": 2, "embedding": 6, "standardized": 4, "physical": 4}
ARRAY_NAMES = (*ARRAY_WIDTHS, "permutation", "source_cell_ids")
MATH_KEYS = (
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "deterministic_algorithms",
    "float32_matmul_precision",
    "preferred_blas_library",
)
METRIC_OUTPUTS = {
    "l2_pressure": {"l2_pressure"},
    "l2_shear_stress": {
        f"l2_shear_stress_wall_shear_stress_true_{axis}_l2_error"
        for axis in ("x", "y", "z")
    },
    "l2_pressure_area_weighted": {"l2_pressure_area_weighted"},
    "drag": {"drag_error", "drag_true", "drag_pred"},
    "lift": {"lift_error", "lift_true", "lift_pred"},
}


class FieldContractError(ParityContractError):
    """Reference or candidate evidence does not satisfy the frozen profile."""


def inference_seed(run_seed: int, case_id: str) -> int:
    payload = str(int(run_seed)).encode("ascii") + b"\0" + case_id.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def block_plan(point_count: int, block_size: int = 60000) -> list[int]:
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 1
    ):
        raise FieldContractError("point_count must be a positive integer")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise FieldContractError("block_size must be a positive integer")
    return [
        min(block_size, point_count - begin)
        for begin in range(0, point_count, block_size)
    ]


def validate_reference_profile(profile: Mapping[str, Any]) -> None:
    validate_profile(profile)
    expected_provider = {
        "commit": PROVIDER_COMMIT,
        "version": "0.0.2",
        "physicsnemo_version": "2.1.1",
        "python_version": "3.12",
    }
    for key, expected in expected_provider.items():
        _equal(profile["provider"].get(key), expected, f"provider.{key}")
    _equal(profile["request"]["models"], ["transolver_surface"], "request.models")
    _equal(profile["request"]["seed"], 42, "request.seed")
    _equal(
        profile["request"]["metrics"],
        [
            "l2_pressure",
            "l2_shear_stress",
            "l2_pressure_area_weighted",
            "drag",
            "lift",
        ],
        "request.metrics",
    )
    model = profile["config"]["models"][0]
    _equal(model.get("package"), PACKAGE, "model.package")
    kwargs = model["kwargs"]
    _equal(
        kwargs,
        {
            "batch_resolution": 60000,
            "geometry_sampling": 300000,
            "cuda_bf16_autocast": False,
        },
        "model.kwargs",
    )
    run = profile["config"]["run"]
    _equal(run.get("metrics_cache", {}).get("enabled"), False, "metrics_cache.enabled")
    _equal(run.get("fail_on_any_metric_nan"), True, "fail_on_any_metric_nan")
    reference = profile.get("native_reference", {})
    for key, expected in {
        "schema_version": 1,
        "case_ids": ["run_1"],
        "batch_resolution": 60000,
        "resolution": None,
        "include_geometry": False,
        "output_location": "cell",
        "source_order": True,
        "model_revision": MODEL_REVISION,
        "point_count": 8828095,
        "package_point_dimensions": [60000, 8095],
    }.items():
        if key not in reference:
            raise FieldContractError(f"native_reference.{key} is required")
        _equal(reference[key], expected, f"native_reference.{key}")
    for key, expected in ASSET_PINS.items():
        _equal(reference.get(key), expected, f"native_reference.{key}")
    _equal(set(reference.get("fields", {})), set(ARRAY_WIDTHS), "field tolerances")
    for name, tolerance in reference["fields"].items():
        for key in ("rtol", "atol"):
            value = tolerance.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise FieldContractError(f"{name}.{key} must be finite and nonnegative")


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FieldContractError(f"{label} mismatch: {actual!r} != {expected!r}")


def validate_reference_request(
    profile: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    validate_reference_profile(profile)
    validate_request(profile, request)
    _equal(
        [case["case_id"] for case in request["cases"]],
        list(CASE_PINS),
        "case inventory",
    )
    for case in request["cases"]:
        for key, expected in CASE_PINS[case["case_id"]].items():
            _equal(case.get(key), expected, f"{case['case_id']}.{key}")


def validate_report(profile: Mapping[str, Any], report: Any) -> None:
    structure, _ = normalize_report(report)
    _equal(len(structure), 1, "report model/dataset inventory")
    row = structure[0]
    _equal(
        (row["model"], row["dataset"]),
        ("transolver_surface", "drivaerml"),
        "report model/dataset",
    )
    _equal(
        row["cases"], profile["native_reference"]["case_ids"], "report case inventory"
    )
    expected_metrics = set().union(
        *(METRIC_OUTPUTS[name] for name in profile["request"]["metrics"])
    )
    for label, keys in [
        ("summary", row["summary_metric_keys"]),
        *[(case["case_id"], case["metric_keys"]) for case in row["per_case"]],
    ]:
        _equal(set(keys), expected_metrics, f"{label} metric inventory")


def verify_file(path: str | Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FieldContractError(f"missing artifact: {path}")
    _equal(path.stat().st_size, pin["size_bytes"], f"{path.name} size")
    digest = sha256_file(path)
    _equal(digest, pin["sha256"], f"{path.name} SHA-256")
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def _artifact_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise FieldContractError(f"unsafe artifact path: {relative!r}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise FieldContractError(f"artifact escapes evidence directory: {relative}")
    return path


def array_descriptor(
    path: Path, root: Path, array: np.ndarray, *, encoding: str = "npy"
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "encoding": encoding,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_array(
    root: Path, descriptor: Mapping[str, Any], *, shape: list[int], dtype: str
) -> np.ndarray:
    _equal(descriptor.get("shape"), shape, "array shape metadata")
    _equal(descriptor.get("dtype"), dtype, "array dtype metadata")
    path = _artifact_path(root, descriptor.get("path"))
    verify_file(path, descriptor)
    encoding = descriptor.get("encoding")
    if encoding == "npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif encoding == "raw":
        expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        _equal(path.stat().st_size, expected_bytes, "raw array byte count")
        array = np.memmap(path, mode="r", dtype=dtype, shape=tuple(shape))
    else:
        raise FieldContractError(f"unsupported array encoding: {encoding!r}")
    _equal(list(array.shape), shape, "array shape")
    _equal(array.dtype.str, dtype, "array dtype")
    flat = array.reshape(-1)
    for begin in range(0, flat.size, 1024 * 1024):
        if not np.isfinite(flat[begin : begin + 1024 * 1024]).all():
            raise FieldContractError(f"nonfinite array: {descriptor['path']}")
    return array


def validate_evidence(
    profile: Mapping[str, Any], evidence: Mapping[str, Any], root: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Verify provenance and bytes before any numerical comparison; no GPU needed."""
    validate_reference_profile(profile)
    _equal(evidence.get("schema_version"), 1, "evidence schema_version")
    _equal(evidence.get("profile_id"), profile["profile_id"], "profile_id")
    _equal(evidence.get("model"), "transolver_surface", "model")
    _equal(evidence.get("model_revision"), MODEL_REVISION, "model_revision")
    if evidence.get("runtime") not in (
        "python-provider",
        "native-aoti",
        "native-tensorrt",
    ):
        raise FieldContractError(
            "runtime must identify python-provider, native-aoti or native-tensorrt"
        )
    for key, pin in ASSET_PINS.items():
        for field in ("sha256", "size_bytes"):
            _equal(evidence.get(key, {}).get(field), pin[field], f"{key}.{field}")
    provider = evidence.get(
        "provider"
        if evidence["runtime"] == "python-provider"
        else "reference_provider",
        {},
    )
    _equal(provider.get("commit"), PROVIDER_COMMIT, "reference provider commit")
    _equal(
        provider.get("physicsnemo_version"), "2.1.1", "reference physicsnemo version"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", str(evidence.get("source_sha", ""))):
        raise FieldContractError("source_sha must be a full Git SHA")
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", str(evidence.get("image", ""))):
        raise FieldContractError("image must use an immutable digest")
    environment = evidence.get("environment", {})
    for key in ("gpu_name", "compute_capability", "torch", "cuda", "vtk", "math"):
        if key not in environment:
            raise FieldContractError(f"environment.{key} is required")
    for key in MATH_KEYS:
        if key not in environment["math"]:
            raise FieldContractError(f"environment.math.{key} is required")
    cases = evidence.get("cases", [])
    _equal(
        [case.get("case_id") for case in cases],
        profile["native_reference"]["case_ids"],
        "evidence case inventory",
    )
    loaded = {}
    for case in cases:
        case_id = case["case_id"]
        for key, expected in CASE_PINS[case_id].items():
            _equal(case.get("inputs", {}).get(key), expected, f"{case_id} input {key}")
        _equal(case.get("run_seed"), profile["request"]["seed"], "run_seed")
        _equal(
            case.get("inference_seed"),
            inference_seed(case["run_seed"], case_id),
            "inference_seed",
        )
        count = case.get("point_count")
        plan = block_plan(count)
        _equal(count, CASE_POINT_COUNTS[case_id], "full source cell count")
        if count < 60000:
            raise FieldContractError(
                "full run_1 qualification requires at least one 60000-point block"
            )
        _equal(case.get("block_points"), plan, "block_points")
        _equal(case.get("extent"), "all-source-cells", "inference extent")
        _equal(case.get("batch_resolution"), 60000, "batch_resolution")
        _equal(case.get("output_location"), "cell", "output_location")
        _equal(case.get("source_order"), True, "source_order")
        _equal(set(case.get("arrays", {})), set(ARRAY_NAMES), "array inventory")
        arrays = {}
        for name in ARRAY_NAMES:
            shape = (
                [1, count, ARRAY_WIDTHS[name]]
                if name in ("fx", "embedding")
                else [count, ARRAY_WIDTHS[name]]
                if name in ARRAY_WIDTHS
                else [count]
            )
            arrays[name] = load_array(
                root,
                case["arrays"][name],
                shape=shape,
                dtype="<f4" if name in ARRAY_WIDTHS else "<i8",
            )
        _equal(
            np.array_equal(arrays["source_cell_ids"], np.arange(count, dtype=np.int64)),
            True,
            "source cell identities",
        )
        _equal(
            np.array_equal(
                np.sort(arrays["permutation"]), np.arange(count, dtype=np.int64)
            ),
            True,
            "permutation bijection",
        )
        loaded[case_id] = arrays
    return loaded


def compare_evidence(
    profile: Mapping[str, Any], reference_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Compare field evidence; scientific report comparison remains an additional gate."""
    reference, candidate = (
        read_json_object(reference_path),
        read_json_object(candidate_path),
    )
    _equal(reference.get("runtime"), "python-provider", "reference runtime")
    expected = validate_evidence(profile, reference, reference_path.parent)
    actual = validate_evidence(profile, candidate, candidate_path.parent)
    _equal(
        candidate["environment"]["math"],
        reference["environment"]["math"],
        "effective math settings",
    )
    _equal(
        candidate["environment"]["compute_capability"],
        reference["environment"]["compute_capability"],
        "GPU compute capability",
    )
    results = []
    for case_id, arrays in expected.items():
        for name, reference_array in arrays.items():
            observed = actual[case_id][name]
            _equal(
                list(observed.shape),
                list(reference_array.shape),
                f"{case_id}/{name} shape",
            )
            if name not in ARRAY_WIDTHS:
                _equal(
                    np.array_equal(observed, reference_array), True, f"{case_id}/{name}"
                )
                continue
            tolerance = profile["native_reference"]["fields"][name]
            flat_ref, flat_actual = reference_array.reshape(-1), observed.reshape(-1)
            mismatches, maximum = 0, 0.0
            for begin in range(0, flat_ref.size, 1024 * 1024):
                ref = np.asarray(
                    flat_ref[begin : begin + 1024 * 1024], dtype=np.float64
                )
                value = np.asarray(
                    flat_actual[begin : begin + 1024 * 1024], dtype=np.float64
                )
                delta = np.abs(value - ref)
                maximum = max(maximum, float(delta.max(initial=0)))
                mismatches += int(
                    np.count_nonzero(
                        delta > tolerance["atol"] + tolerance["rtol"] * np.abs(ref)
                    )
                )
            results.append(
                {
                    "case_id": case_id,
                    "array": name,
                    "mismatches": mismatches,
                    "max_abs_error": maximum,
                    **tolerance,
                }
            )
    return {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "passed": all(result["mismatches"] == 0 for result in results),
        "reference_image": reference["image"],
        "candidate_image": candidate["image"],
        "fields": results,
        "scientific_reports_compared": False,
    }
