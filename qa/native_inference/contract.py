"""Independent evidence checks for native inference QA (standard library only)."""

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

SUMMARY_BEGIN = "NATIVE_INFERENCE_QA_SUMMARY_BEGIN"
SUMMARY_END = "NATIVE_INFERENCE_QA_SUMMARY_END"


def expected_cases(profile):
    if profile not in ("smoke", "full"):
        raise ValueError(f"unknown QA profile: {profile}")
    return (
        [
            "environment",
            "affine.config",
            "affine.check",
            "affine.build_a",
            "affine.lock_rejection",
            "affine.build_b",
            "handoff",
        ]
        + [
            f"affine.{checkpoint}.{backend}.{operation}"
            for checkpoint in ("a", "b")
            for backend in ("aoti", "tensorrt")
            for operation in ("cli", "sdk", "missing_payload", "truncated_input")
        ]
        + (
            [
                "transolver.assets",
                "transolver.import",
                "transolver.build75",
                "transolver.build11",
            ]
            + [
                f"transolver.{backend}.{operation}"
                for backend in ("aoti", "tensorrt")
                for operation in ("surface75", "surface161", "missing_tail")
            ]
            if profile == "full"
            else []
        )
    )


def validate_summary(summary, *, run_id, source_sha, image_digest, profile):
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    if type(summary.get("schema_version")) is not int:
        raise ValueError("summary schema_version must be an integer")
    identity = dict(
        schema_version=1,
        run_id=run_id,
        source_sha=source_sha,
        image_digest=image_digest,
        profile=profile,
        status="passed",
    )
    for key, value in identity.items():
        if summary.get(key) != value:
            raise ValueError(f"summary {key} mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source SHA must be a full Git revision")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("image must be pinned by digest")
    stages = summary.get("stages")
    if not isinstance(stages, dict) or set(stages) != {"build", "consumer"}:
        raise ValueError("summary must describe both stages")
    if any(
        not isinstance(v, dict) or v.get("status") != "passed" for v in stages.values()
    ):
        raise ValueError("both stages must pass")
    required = expected_cases(profile)
    declared = summary.get("expected_cases")
    cases = summary.get("cases")
    if not isinstance(declared, list) or sorted(declared) != sorted(required):
        raise ValueError("declared case inventory does not match the profile")
    if not isinstance(cases, list) or any(not isinstance(c, dict) for c in cases):
        raise ValueError("missing case results")
    if sorted(c.get("name", "") for c in cases) != sorted(required):
        raise ValueError("executed case inventory does not match the profile")
    if any(c.get("status") != "passed" for c in cases):
        raise ValueError("every required case must pass")


def compare_f32(actual, expected, *, max_abs_limit=1e-4, relative_l2_limit=1e-4):
    if len(actual) != len(expected) * 4 or not expected:
        raise ValueError("output byte count does not match the FP32 reference")
    values = struct.unpack(f"<{len(expected)}f", actual)
    if not all(math.isfinite(v) for v in (*values, *expected)):
        raise ValueError("nonfinite native output or reference")
    differences = [a - b for a, b in zip(values, expected, strict=True)]
    maximum = max(abs(v) for v in differences)
    relative = math.sqrt(math.fsum(v * v for v in differences)) / max(
        math.sqrt(math.fsum(v * v for v in expected)), sys.float_info.epsilon
    )
    if maximum > max_abs_limit or relative > relative_l2_limit:
        raise ValueError(
            f"native parity failed: max_abs={maximum}, relative_l2={relative}"
        )
    return {"max_abs": maximum, "relative_l2": relative}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"artifact path escapes its root: {relative}")
    path = root / value
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"artifact path escapes its root: {relative}")
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"artifact symlink is unsupported: {relative}")
        cursor = cursor.parent
    return path


def verify_file(root, record):
    path = contained(root, record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256(path) != record["sha256"]
    ):
        raise ValueError(f"artifact identity mismatch: {record['path']}")
    return path


def verify_model(model_dir, backends):
    root = Path(model_dir).resolve()
    release = json.loads((root / "model-release.json").read_text())
    variants = release.get("variants", {})
    if release.get("format_version") != 1 or set(variants) != set(backends):
        raise ValueError("release does not contain exactly the requested backends")
    packages = {}
    for backend in backends:
        variant = variants[backend]
        package = contained(root, variant["package"])
        records = variant.get("files", [])
        if not records:
            raise ValueError("package inventory is empty")
        files = [verify_file(root, record) for record in records]
        if len(set(files)) != len(files) or any(
            not p.is_relative_to(package) for p in files
        ):
            raise ValueError("package inventory contains duplicate or foreign files")
        actual = set(package.rglob("*"))
        if any(p.is_symlink() for p in actual):
            raise ValueError("package contains symlinks")
        if set(files) != {p for p in actual if p.is_file()}:
            raise ValueError("package inventory is incomplete")
        manifest_path = package / "model.json"
        if manifest_path not in files:
            raise ValueError("package manifest is not inventoried")
        manifest = json.loads(manifest_path.read_text())
        artifacts = manifest.get("artifacts", [])
        if not artifacts:
            raise ValueError("package has no compiled payload")
        for artifact in artifacts:
            payload = contained(package, artifact["path"])
            if (
                artifact["backend"] != backend
                or artifact["target"] != "cuda"
                or payload not in files
            ):
                raise ValueError("package backend/target/payload mismatch")
        packages[backend] = str(package)
    return packages


def verify_build(output, backends, *, case_count=3):
    output = Path(output)
    report = json.loads((output / "build.json").read_text())
    if (
        report.get("status") != "complete"
        or report.get("case_count") != case_count
        or set(report.get("variants", {})) != set(backends)
    ):
        raise ValueError("build receipt has incomplete backend/case coverage")
    verify_file(output, report["release"])
    packages = verify_model(output / "model", backends)
    for backend, variant in report["variants"].items():
        checks = json.loads(verify_file(output, variant["checks"]).read_text())
        cases = checks.get("cases", [])
        if (
            variant.get("status") != "complete"
            or checks.get("passed") is not True
            or checks.get("backend") != backend
            or len(cases) != case_count
        ):
            raise ValueError("native build checks are incomplete")
        manifest = json.loads((Path(packages[backend]) / "model.json").read_text())
        for index, case in enumerate(cases):
            metadata = case.get("metadata", {})
            if (
                case.get("passed") is not True
                or metadata.get("completed") is not True
                or metadata.get("backend") != backend
                or metadata.get("execution_device") != {"type": "cuda", "index": 0}
            ):
                raise ValueError(
                    "native build check did not execute the requested backend/device"
                )
            outputs = case.get("outputs", [])
            if len(outputs) != len(manifest["outputs"]):
                raise ValueError("native build check output coverage mismatch")
            observed = metadata.get("outputs", [])
            if len(observed) != len(outputs):
                raise ValueError("native build output metadata is missing")
            case_dir = output / "checks" / backend / f"case-{index}"
            for number, tensor in enumerate(outputs):
                for prefix, key in (
                    ("output", "actual_sha256"),
                    ("reference", "reference_sha256"),
                ):
                    if sha256(case_dir / f"{prefix}-{number}.bin") != tensor[key]:
                        raise ValueError("native check tensor identity mismatch")
                spec = manifest["outputs"][number]
                count = math.prod(spec["shape"])
                expected_metadata = {
                    "name": spec["name"],
                    "shape": spec["shape"],
                    "dtype": "float32",
                    "byte_size": count * 4,
                    "device": {"type": "cpu", "index": 0},
                }
                if spec["dtype"] != "float32" or observed[number] != expected_metadata:
                    raise ValueError("native build output tensor contract mismatch")
                actual = (case_dir / f"output-{number}.bin").read_bytes()
                reference = (case_dir / f"reference-{number}.bin").read_bytes()
                if len(reference) != count * 4:
                    raise ValueError(
                        "reference byte count does not match its tensor shape"
                    )
                limit = 0.0 if checks.get("require_byte_identical") is True else 1e-4
                compare_f32(
                    actual,
                    struct.unpack(f"<{count}f", reference),
                    max_abs_limit=limit,
                    relative_l2_limit=limit,
                )
                if limit == 0 and actual != reference:
                    raise ValueError(
                        "exact profile requires byte-identical native outputs"
                    )
    return packages
