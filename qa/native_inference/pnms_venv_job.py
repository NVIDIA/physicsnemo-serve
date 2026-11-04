#!/usr/bin/env python3
"""Build native applications in a fresh isolated venv inside a Lepton GPU job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

if __package__:
    from . import pnms_local_job as local
    from .contract import sha256
else:
    import pnms_local_job as local
    from contract import sha256

SUMMARY_BEGIN = local.SUMMARY_BEGIN
SUMMARY_END = local.SUMMARY_END
extract_summary = local.extract_summary
NATIVE_TARGETS = (("aoti", "tensor"), ("tensorrt", "custom"))
ISOLATION_PROBE = """
import json, sys
print(json.dumps(dict(python=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
                     sys_path=sys.path, no_user_site=sys.flags.no_user_site)))
"""
STACK_PROBE = """
import importlib, importlib.metadata as metadata, json, sys
modules = {name: importlib.import_module(name) for name in
           ('torch', 'tensorrt', 'numpy', 'onnx', 'onnxscript', 'model_builder', 'pnms_app')}
torch = modules['torch']
assert torch.cuda.is_available(), 'CUDA unavailable in selected venv'
assert torch.cuda.device_count() == 1, 'exactly one allocated GPU is required'
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(python=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
    sys_path=sys.path, no_user_site=sys.flags.no_user_site,
    origins={name: module.__file__ for name,module in modules.items()},
    torch=torch.__version__, cuda=torch.version.cuda, tensorrt=modules['tensorrt'].__version__,
    cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), visible_gpu_count=1,
    gpu=dict(name=p.name, uuid=str(getattr(p,'uuid','')), major=p.major, minor=p.minor),
    distributions=sorted((d.metadata['Name'],d.version) for d in metadata.distributions()
                         if d.metadata.get('Name')))))
"""


def expected_cases(profile="fresh-venv"):
    if profile != "fresh-venv":
        raise ValueError("Fresh venv QA requires the fresh-venv profile")
    return (
        ["environment", "venv.setup", "venv.verify"]
        + [
            f"native.{backend}.{template}.{operation}"
            for backend, template in NATIVE_TARGETS
            for operation in (
                "init",
                "check",
                "build",
                "held_out",
                "cache_reuse",
                "relocation",
                "reject_invalid",
            )
        ]
        + ["environment_lock.refresh"]
    )


def isolated_environment(original):
    result = {
        key: value
        for key, value in original.items()
        if key
        not in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CMAKE_PREFIX_PATH",
            "Torch_DIR",
            "Caffe2_DIR",
        )
        and not key.startswith("PIP_")
    }
    result.update(
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Keep CUDA forward-compatibility paths, while removing baked Python/SDK
    # libraries that could hide dependencies missing from the fresh venv.
    if result.get("LD_LIBRARY_PATH"):
        result["LD_LIBRARY_PATH"] = os.pathsep.join(
            value
            for value in result["LD_LIBRARY_PATH"].split(os.pathsep)
            if value
            and not any(
                part in value
                for part in (
                    "site-packages",
                    "dist-packages",
                    "/opt/physicsnemo-inference",
                )
            )
        )
    return result


def configure_cuda_environment(environment):
    nvcc = shutil.which("nvcc", path=environment.get("PATH", ""))
    if nvcc:
        executable = Path(nvcc).resolve()
        toolkit = str(executable.parent.parent)
        # Torch's legacy FindCUDA does not resolve our tool-only PATH symlink
        # before inferring the prefix, and does not read CUDA_HOME. Supply the
        # variables it recognizes as well as those used by modern CMake.
        environment.update(
            CUDA_HOME=toolkit,
            CUDAToolkit_ROOT=toolkit,
            CUDA_TOOLKIT_ROOT=toolkit,
            CUDA_PATH=toolkit,
            CUDA_BIN_PATH=toolkit,
            CUDACXX=str(executable),
        )


def requirement_inventory(paths, digests):
    if not paths or len(paths) != len(digests):
        raise ValueError(
            "Each explicit requirements file needs one expected SHA-256 digest"
        )
    records = []
    for path, digest in zip(paths, digests):
        path = Path(path).absolute()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(f"Requirements file digest mismatch: {path}")
        records.append({"path": str(path), "sha256": digest})
    return records


def verify_isolation(details, venv):
    venv = Path(venv).absolute()
    if (
        Path(details.get("prefix", "")) != venv
        or details.get("prefix") == details.get("base_prefix")
        or not Path(details.get("python", "")).is_relative_to(venv)
        or details.get("no_user_site") != 1
    ):
        raise ValueError("Selected Python is not the fresh isolated venv")
    for value in details.get("sys_path", []):
        path = Path(value)
        if any(
            part in ("site-packages", "dist-packages") for part in path.parts
        ) and not path.is_relative_to(venv):
            raise ValueError(
                f"Global site-packages/dist-packages leaked into venv: {path}"
            )


def setup_command(args, destination):
    # Import the unchanged source frontend explicitly without PYTHONPATH leaking
    # into the venv or CMake child processes.
    launcher = "import sys; sys.path.insert(0,sys.argv.pop(1)); from model_builder.build.cli import main; raise SystemExit(main())"
    command = [
        args.python,
        "-c",
        launcher,
        args.repository / "native-inference/model-builder/src",
        "setup-env",
        destination,
        "--python",
        args.python,
        "--builder-package",
        args.repository / "native-inference/model-builder",
        "--sdk-source",
        args.repository / "native-inference/cpp-runtime",
        "--backend",
        "aoti",
        "--backend",
        "tensorrt",
        "--tensorrt-root",
        args.tensorrt_root,
        "--tensorrt-cuda-major",
        args.tensorrt_cuda_major,
    ]
    for path in args.requirements:
        command.extend(["--requirements", path])
    return [*map(str, command), "--json"]


def validate_tensorrt_versions(python_version, cpp_version):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", cpp_version) or not re.fullmatch(
        re.escape(cpp_version) + r"(?:\.post[0-9]+)?", python_version
    ):
        raise ValueError(
            "TensorRT Python bindings do not match all four C++ SDK version components"
        )
    return {"python": python_version, "cpp": cpp_version, "compatible": True}


class Context(local.Context):
    def __init__(self, args):
        super().__init__(args)
        self.environment = isolated_environment(self.environment)
        configure_cuda_environment(self.environment)
        self.report.update(
            profile="fresh-venv",
            qa_scope="fresh-venv-workflow",
            expected_cases=expected_cases(),
        )
        self.venv = self.root / "venv"


def preflight(ctx):
    metadata = json.loads(ctx.args.source_metadata.read_text())
    if metadata.get("source_sha") != ctx.args.expected_source_sha:
        raise ValueError("Prepared development image source SHA differs")
    ctx.environment["PNMS_SOURCE_SHA"] = metadata["source_sha"]
    for relative in (
        "pnms",
        "native-inference/model-builder/pyproject.toml",
        "native-inference/cpp-runtime/CMakeLists.txt",
    ):
        if not (ctx.repository / relative).is_file():
            raise ValueError(f"Repository input is missing: {relative}")
    for tool in ("cmake", "c++", "nvcc", "nvidia-smi"):
        if not shutil.which(tool, path=ctx.environment["PATH"]):
            raise ValueError(f"Development container lacks {tool}")
    records = requirement_inventory(ctx.args.requirements, ctx.args.requirements_sha256)
    if ctx.args.wheelhouse:
        if not ctx.args.wheelhouse.is_dir():
            raise ValueError("Explicit wheelhouse directory is missing")
        ctx.environment.update(
            PIP_NO_INDEX="1", PIP_FIND_LINKS=str(ctx.args.wheelhouse)
        )
    return {
        "requirements": records,
        "runner_sha256": sha256(Path(__file__)),
        "pnms_sha256": sha256(ctx.repository / "pnms"),
        "docker": "absent",
        "lepton_cli": "absent",
        "wheelhouse": str(ctx.args.wheelhouse) if ctx.args.wheelhouse else None,
    }


def setup(ctx):
    if ctx.venv.exists():
        raise ValueError("Fresh venv destination already exists")
    records = requirement_inventory(ctx.args.requirements, ctx.args.requirements_sha256)
    # The configured CUDA target comes from the allocated GPU, not the baked SDK.
    ctx.command(
        "gpu.target",
        [
            ctx.args.python,
            "-c",
            "import ctypes,json; c=ctypes.CDLL('libcuda.so.1'); assert c.cuInit(0)==0; n=ctypes.c_int(); assert c.cuDeviceGetCount(ctypes.byref(n))==0 and n.value==1; a=ctypes.c_int(); b=ctypes.c_int(); assert c.cuDeviceGetAttribute(ctypes.byref(a),75,0)==0; assert c.cuDeviceGetAttribute(ctypes.byref(b),76,0)==0; print(json.dumps({'architecture':str(a.value)+str(b.value)}))",
        ],
    )
    target = json.loads((ctx.root / "logs/gpu.target/result.json").read_text())[
        "architecture"
    ]
    ctx.environment["CMAKE_CUDA_ARCHITECTURES"] = target
    ctx.environment["CUDAARCHS"] = target
    ctx.environment["TORCH_CUDA_ARCH_LIST"] = target[:-1] + "." + target[-1]
    result = ctx.command("venv.setup", setup_command(ctx.args, ctx.venv))
    if (
        result.get("status") != "complete"
        or result.get("stage") != "environment"
        or result.get("backends") != ["aoti", "tensorrt"]
    ):
        raise ValueError("Model Builder setup-env did not complete")
    interpreter = ctx.venv / "bin/python"
    if result.get("python") != str(interpreter) or not Path(
        result.get("runtime", "")
    ).is_relative_to(ctx.venv):
        raise ValueError(
            "setup-env returned an interpreter/runtime outside the new venv"
        )
    config = (ctx.venv / "pyvenv.cfg").read_text().lower()
    if not re.search(r"^include-system-site-packages\s*=\s*false\s*$", config, re.M):
        raise ValueError("setup-env unexpectedly enabled global site packages")
    ctx.environment["VIRTUAL_ENV"] = str(ctx.venv)
    ctx.environment["PATH"] = (
        str(ctx.venv / "bin") + os.pathsep + ctx.environment["PATH"]
    )
    ctx.environment["TENSORRT_ROOT"] = str(ctx.args.tensorrt_root)
    application_source = ctx.root / "application-source"
    shutil.copytree(
        ctx.repository / "application-cli",
        application_source,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    ctx.command(
        "venv.application",
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            application_source,
        ],
        structured=False,
    )
    # Independently verify the stack selected by the public dual-backend
    # setup command, including the complete native TensorRT version.
    trt = ctx.command(
        "venv.tensorrt",
        [
            interpreter,
            "-c",
            "import json,pathlib,sys,tensorrt; "
            "from model_builder.build.environment import _tensorrt_header_version; "
            "r=pathlib.Path(sys.argv[1]); "
            "h=next(p/'NvInferVersion.h' for p in (r/'include',r/'include/x86_64-linux-gnu') if (p/'NvInferVersion.h').is_file()); "
            "print(json.dumps({'python':tensorrt.__version__,'cpp':_tensorrt_header_version(h.read_text())}))",
            ctx.args.tensorrt_root,
        ],
    )
    trt_compatibility = validate_tensorrt_versions(trt["python"], trt["cpp"])
    ctx.command(
        "venv.tensorrt-dependencies",
        [
            interpreter,
            "-c",
            "from model_builder.build.environment import _DEPENDENCY_PROBE; exec(_DEPENDENCY_PROBE)",
            ctx.args.tensorrt_cuda_major,
            trt["cpp"],
            "aoti",
        ],
        structured=False,
    )
    # setup-env deliberately produces a runtime executable. Install that same
    # freshly built SDK for the public C++ workflow consumer as well.
    runtime_build = ctx.venv / ".physicsnemo/runtime"
    prefix = ctx.venv / ".physicsnemo/sdk"
    ctx.command(
        "venv.sdk-build",
        ["cmake", "--build", runtime_build, "--parallel", "2"],
        structured=False,
    )
    ctx.command(
        "venv.sdk-install",
        ["cmake", "--install", runtime_build, "--prefix", prefix],
        structured=False,
    )
    ctx.args.python = interpreter
    ctx.args.sdk_prefix = prefix
    return {
        "fresh": True,
        "system_site_packages": False,
        "sdk_built": True,
        "setup_backends": result["backends"],
        "sdk_backends": ["aoti", "tensorrt"],
        "tensorrt_compatibility": trt_compatibility,
        "requirements": records,
        "setup_receipt": str(ctx.venv / ".physicsnemo/environment.json"),
        "sdk_prefix": str(prefix),
        "architecture": target,
    }


def verify(ctx):
    details = ctx.command("venv.stack", [ctx.args.python, "-c", STACK_PROBE])
    verify_isolation(details, ctx.venv)
    for name, origin in details.get("origins", {}).items():
        if not origin or not Path(origin).resolve().is_relative_to(ctx.venv.resolve()):
            raise ValueError(
                f"Selected dependency {name} came from outside the venv: {origin}"
            )
    if len(details.get("origins", {})) != 7 or details.get("visible_gpu_count") != 1:
        raise ValueError("Fresh venv did not probe every required dependency/GPU")
    ctx.command(
        "venv.pip-check", [ctx.args.python, "-m", "pip", "check"], structured=False
    )
    ctx.command(
        "venv.freeze",
        [ctx.args.python, "-m", "pip", "freeze", "--all"],
        structured=False,
    )
    cmake = (ctx.venv / ".physicsnemo/runtime/CMakeCache.txt").read_text()
    if (
        str(ctx.args.python) not in cmake
        or str(Path(details["origins"]["torch"]).parent) not in cmake
    ):
        raise ValueError(
            "Fresh SDK CMake provider paths do not select the venv Python/Torch"
        )
    return {
        **details,
        "isolated": True,
        "module_origins_verified": True,
        "sdk_provider_paths_verified": True,
        "pip_check": "passed",
        "pip_freeze_sha256": sha256(ctx.root / "logs/venv.freeze/stdout.log"),
        "runtime_sha256": sha256(ctx.args.sdk_prefix / "bin/physicsnemo-infer"),
    }


def validate_summary(summary, *, run_id, source_sha, image_digest, profile):
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "profile": profile,
        "status": "passed",
        "qa_scope": "fresh-venv-workflow",
        "http_qa": "not_run",
    }
    if (
        not isinstance(summary, dict)
        or profile != "fresh-venv"
        or any(summary.get(k) != v for k, v in identity.items())
    ):
        raise ValueError("Fresh venv QA summary identity/status/scope differs")
    names = expected_cases(profile)
    cases = summary.get("cases", [])
    if (
        summary.get("expected_cases") != names
        or sorted(case.get("name", "") for case in cases) != sorted(names)
        or any(case.get("status") != "passed" for case in cases)
    ):
        raise ValueError("Every required fresh venv QA case must pass exactly once")
    details = {case["name"]: case.get("details", {}) for case in cases}
    setup = details["venv.setup"]
    check = details["venv.verify"]
    if (
        setup.get("fresh") is not True
        or setup.get("system_site_packages") is not False
        or setup.get("sdk_built") is not True
        or not setup.get("requirements")
    ):
        raise ValueError("Fresh isolated venv/SDK setup evidence is missing")
    if (
        any(
            check.get(key) is not True
            for key in (
                "isolated",
                "module_origins_verified",
                "sdk_provider_paths_verified",
            )
        )
        or check.get("pip_check") != "passed"
        or check.get("visible_gpu_count") != 1
    ):
        raise ValueError("Fresh venv dependency isolation verification is missing")
    for backend, template in NATIVE_TARGETS:
        prefix = f"native.{backend}.{template}"
        build = details[prefix + ".build"]
        held = details[prefix + ".held_out"]
        repeated = details[prefix + ".cache_reuse"]
        for key, length in (
            ("build_id", 32),
            ("bundle_digest", 64),
            ("model_identity", 64),
        ):
            if not re.fullmatch(f"[0-9a-f]{{{length}}}", str(build.get(key, ""))):
                raise ValueError("Native build identity is missing")
        if (
            build.get("native_checks") != {"verified": True, "case_count": 3}
            or held.get("case_count") != 3
            or held.get("parity") != "passed"
        ):
            raise ValueError("Native independent numerical parity is missing")
        if (
            repeated.get("model_reused") is not True
            or repeated.get("workflow_reused") is not True
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(repeated.get("environment_lock_digest", ""))
            )
        ):
            raise ValueError("Native model/workflow cache or lock evidence is missing")
    refresh = details["environment_lock.refresh"]
    if any(
        refresh.get(key) is not True
        for key in (
            "drift_rejected",
            "prior_success_preserved",
            "refresh_passed",
            "model_reused",
            "workflow_reused",
        )
    ) or not re.fullmatch(
        r"[0-9a-f]{64}", str(refresh.get("environment_lock_digest", ""))
    ):
        raise ValueError("Lock drift/refresh evidence is missing")
    candidates = summary.get("release_candidates", [])
    if len(candidates) != len(NATIVE_TARGETS) or {
        local.candidate_key(item) for item in candidates
    } != {(b, t, "application/json") for b, t in NATIVE_TARGETS}:
        raise ValueError("Fresh venv release candidate inventory differs")
    for item in candidates:
        latest = (
            refresh
            if item["backend"] == "aoti"
            else details[local.candidate_name(item) + ".cache_reuse"]
        )
        if any(
            item.get(key) != latest.get(key) for key in ("build_id", "bundle_digest")
        ):
            raise ValueError(
                "Release candidate identity differs from latest verified build"
            )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=Path("/workspace"))
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument("--requirements", type=Path, action="append", required=True)
    result.add_argument("--requirements-sha256", action="append", required=True)
    result.add_argument("--wheelhouse", type=Path)
    result.add_argument("--tensorrt-root", type=Path, default=Path("/usr"))
    result.add_argument("--tensorrt-cuda-major", choices=("12", "13"), default="13")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-source-sha", required=True)
    result.add_argument("--image-digest", required=True)
    result.add_argument("--profile", choices=("fresh-venv",), default="fresh-venv")
    result.add_argument("--command-timeout", type=float, default=2400)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    ctx = Context(args)
    code = 1
    try:
        ctx.case("environment", lambda: preflight(ctx))
        ctx.case("venv.setup", lambda: setup(ctx))
        ctx.case("venv.verify", lambda: verify(ctx))
        for backend, template in NATIVE_TARGETS:
            local.native_cases(ctx, backend, template)
        ctx.report["status"] = "passed"
        validate_summary(
            ctx.report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            profile=args.profile,
        )
        code = 0
    except BaseException as error:
        ctx.report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        ctx.save()
        print(SUMMARY_BEGIN, flush=True)
        print(json.dumps(ctx.report, sort_keys=True), flush=True)
        print(SUMMARY_END, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
