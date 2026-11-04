"""Mandatory H100 provider-policy evidence, independent of test pass flags."""

import hashlib
import json
import math
from pathlib import Path
import struct
import sys


PLUGINS = ("linear", "gemm", "slice_bmm", "deslice_bmm")
MATH_POLICY = {
    "name": "provider-default-v1",
    "compute_capability": "9.0",
    "float32_matmul_precision": "high",
    "allow_tf32_matmul": True,
    "allow_tf32_cudnn": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "deterministic_algorithms": False,
    "preferred_blas_library": "cublas",
    "cublas_workspace_bytes": 33554432,
    "cublaslt_workspace_bytes": 1048576,
    "native_tensorrt_tf32": False,
}


def verify_package(package, plugin_root):
    if __package__:
        from .transolver import TENSORRT_PLUGINS
    else:
        from transolver import TENSORRT_PLUGINS
    manifest = json.loads((Path(package) / "model.json").read_text())
    (artifact,) = manifest["artifacts"]
    profile = artifact.get("correctness_profile", {})
    operators = artifact.get("required_operators", [])
    expected = {
        "pnmir.tensorrt-"
        + ("provider-default-" if name.removeprefix("exact_") in PLUGINS else "")
        + name.replace("_", "-")
        for name in TENSORRT_PLUGINS
    }
    if (
        artifact.get("backend") != "tensorrt"
        or artifact.get("target") != "cuda"
        or profile.get("name") != "provider-default-exact-v1"
        or profile.get("version") != 1
        or profile.get("math_policy") != MATH_POLICY
        or set(profile.get("plugins", [])) != set(TENSORRT_PLUGINS)
        or len(operators) != 9
        or {item.get("id") for item in operators} != expected
        or any(item.get("abi") != "1" for item in operators)
        or manifest.get("producer", {}).get("compute_capability") != "9.0"
        or manifest.get("producer", {}).get("tf32") is not False
    ):
        raise ValueError("provider package math/operator policy mismatch")
    libraries = profile.get("plugin_libraries", {})
    if set(libraries) != set(TENSORRT_PLUGINS):
        raise ValueError("provider package requires all nine plugin identities")
    for name, record in libraries.items():
        filename = f"libpnmir_tensorrt_{name}_plugin.so"
        if (
            record.get("filename") != filename
            or record.get("sha256")
            != hashlib.sha256((Path(plugin_root) / filename).read_bytes()).hexdigest()
        ):
            raise ValueError("provider plugin identity differs from installed SDK")
    return profile


def verify_math(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text())
    cases = report.get("cases", [])
    expected = {(name, version) for name in PLUGINS for version in ("1", "2")}
    if (
        report.get("status") != "passed"
        or len(cases) != 8
        or {(c.get("plugin"), c.get("creator_version")) for c in cases} != expected
    ):
        raise ValueError("provider math requires all eight completed policy cases")
    for case in cases:
        stem = f"{case['plugin']}-v{case['creator_version']}"
        reference = (directory / f"{stem}-reference.bin").read_bytes()
        if (
            not reference
            or len(reference) % 4
            or case.get("byte_identical") is not True
            or case.get("repeats") != 2
            or hashlib.sha256(reference).hexdigest() != case.get("sha256")
        ):
            raise ValueError(f"{stem}: invalid provider math evidence")
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", reference)):
            raise ValueError(f"{stem}: nonfinite reference")
        for repeat in range(2):
            if (directory / f"{stem}-actual-{repeat}.bin").read_bytes() != reference:
                raise ValueError(f"{stem}: byte parity failed")
    return report


def qualify_plugins(ctx):
    tests = ctx.native_root / "cpp-runtime/tests"
    plugins = Path(ctx.runtime).resolve().parent.parent / "lib"

    def policy():
        result = ctx.run(
            "provider-policy",
            [
                sys.executable,
                tests / "tensorrt_provider_policy.py",
                "--plugin-root",
                plugins,
            ],
        )
        report = json.loads(result.stdout)
        if report != {"provider_creators": 4, "invalid_policy_rejected": True}:
            raise ValueError("provider creator-policy evidence is incomplete")
        return report

    def math_check():
        output = ctx.root / "provider-plugins"
        ctx.run(
            "provider-math",
            [
                sys.executable,
                tests / "tensorrt_provider_math.py",
                "--plugin-root",
                plugins,
                "--output",
                output,
            ],
        )
        return verify_math(output)

    return {
        "policy": ctx.case("provider.policy", policy),
        "math": ctx.case("provider.math", math_check),
    }
