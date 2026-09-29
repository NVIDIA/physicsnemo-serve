"""One-time local environment bootstrap; model builds reuse the native SDK."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from . import authoring_config, cli


# Execute inside the newly installed environment, never in the frontend Python.
_DEPENDENCY_PROBE = """
import importlib.metadata as metadata
import sys
import pnmir_build.cli
import torch
import numpy
print('Python:', sys.version.split()[0], 'Torch:', torch.__version__, 'NumPy:', numpy.__version__)
if sys.platform == 'win32' and sys.argv[3] and torch.version.cuda is not None:
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError('CUDA AOTInductor on Windows requires a triton-windows '
                           'version matching PyTorch; add it to --requirements.') from exc
if sys.argv[1]:
    if sys.platform == 'win32' and torch.version.cuda is None:
        raise RuntimeError('Native Windows TensorRT requires CUDA-enabled PyTorch; '
                           'pass --requirements with a CUDA PyTorch wheel and its official wheel index.')
    variants = []
    for major in ('12', '13'):
        try:
            metadata.version('tensorrt-cu' + major)
            variants.append(major)
        except metadata.PackageNotFoundError:
            pass
    if variants != [sys.argv[1]]:
        raise RuntimeError('Conflicting TensorRT CUDA packages: ' + str(variants) +
                           '; remove generic tensorrt or the other CUDA variant from your requirements.')
    import onnx
    import onnxscript
    import tensorrt
    if sys.argv[2] and tensorrt.__version__.split('.')[:3] != sys.argv[2].split('.')[:3]:
        raise RuntimeError('TensorRT Python version ' + tensorrt.__version__ +
                           ' does not match the C++ SDK ' + sys.argv[2])
    print('TensorRT:', tensorrt.__version__, 'CUDA package:', sys.argv[1])
"""


def _executable(path, label):
    # Do not resolve a Python venv symlink: its invocation path selects the env.
    path = Path(path).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{label} must be an executable file: {path}")
    return path


def _tensorrt_header_version(header):
    # SDK headers use either decimal literals or aliases to enterprise macros.
    # Resolve only those forms; do not evaluate C preprocessor expressions.
    header = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S)
    definitions = dict(
        re.findall(
            r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+([^\r\n]*)",
            header,
            re.M,
        )
    )
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        value = f"NV_TENSORRT_{part}"
        visited = set()
        while value in definitions and value not in visited:
            visited.add(value)
            value = definitions[value].strip()
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("TensorRT SDK headers must identify TensorRT 10 or newer.")
        parts.append(value)
    if int(parts[0]) < 10:
        raise ValueError("TensorRT SDK headers must identify TensorRT 10 or newer.")
    return ".".join(parts)


def _resolve(args):
    if sys.platform not in ("linux", "darwin", "win32"):
        raise ValueError("setup-env supports Linux, macOS, and Windows.")
    destination = args.directory.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            "Environment already exists; choose a fresh directory. Use build --executor local for an existing environment."
        )
    python = shutil.which(str(args.python))
    if python is None:
        raise ValueError(f"Python executable was not found: {args.python}")
    python = _executable(python, "--python")
    project = authoring_config.load(args.build) if args.build else None
    backends = list(
        dict.fromkeys(
            args.backend or (project["effective"]["backends"] if project else ["aoti"])
        )
    )
    requirements = [path.expanduser().absolute() for path in args.requirements]
    for path in requirements:
        if not path.is_file():
            raise ValueError(f"Requirements file does not exist: {path}")
    runtime = _executable(args.runtime, "--runtime") if args.runtime else None
    checkout = Path(__file__).resolve().parents[2]
    sdk = (args.sdk_source or checkout.parent / "cpp-runtime").expanduser().absolute()
    cmake = None
    trt_version = None
    if runtime is None:
        if not (sdk / "CMakeLists.txt").is_file():
            raise ValueError(
                "Set --sdk-source to the checkout's native-inference/cpp-runtime directory, or supply a compatible --runtime."
            )
        cmake = shutil.which("cmake")
        if cmake is None:
            raise ValueError(
                "SDK bootstrap requires CMake >=3.20 and a C++20 compiler; install them or supply --runtime."
            )
        if "tensorrt" in backends:
            if args.tensorrt_root is None:
                raise ValueError(
                    "TensorRT SDK bootstrap requires --tensorrt-root with C++ headers and libraries; pip TensorRT does not include the headers."
                )
            trt_root = args.tensorrt_root.expanduser().absolute()
            includes = (trt_root / "include", trt_root / "include/x86_64-linux-gnu")
            include = next(
                (
                    path
                    for path in includes
                    if (path / "NvInfer.h").is_file()
                    and (path / "NvInferVersion.h").is_file()
                ),
                None,
            )
            if include is None:
                raise ValueError(
                    f"TensorRT C++ headers NvInfer.h and NvInferVersion.h are missing under {trt_root}."
                )
            header = (include / "NvInferVersion.h").read_text()
            trt_version = _tensorrt_header_version(header)
    package = args.builder_package
    if package is None:
        package = next(
            (
                path
                for path in (checkout, sdk.parent / "model-builder")
                if (path / "pyproject.toml").is_file()
            ),
            None,
        )
    if package is None:
        raise ValueError(
            "Set --builder-package to the Model Builder source directory or wheel; no published default package is assumed."
        )
    package = package.expanduser().absolute()
    if not (
        (package.is_dir() and (package / "pyproject.toml").is_file())
        or (package.is_file() and package.suffix == ".whl")
    ):
        raise ValueError(
            "--builder-package must be a source directory with pyproject.toml or an existing wheel."
        )
    if destination.resolve().is_relative_to(package.resolve()):
        raise ValueError(
            "Create the environment outside the Model Builder package source directory."
        )
    return (
        destination,
        python,
        project,
        backends,
        requirements,
        runtime,
        sdk,
        cmake,
        trt_version,
        package,
    )


def command(args, result):
    try:
        (
            destination,
            python,
            project,
            backends,
            requirements,
            runtime,
            sdk,
            cmake,
            trt_version,
            package,
        ) = _resolve(args)
    except (ValueError, OSError) as exc:
        raise cli.UsageError(str(exc), code="ENVIRONMENT_CONFIGURATION") from exc
    # Reserve the destination without overwriting an existing environment.
    destination.mkdir(parents=True, exist_ok=False)
    state = destination / ".physicsnemo"
    state.mkdir()
    result.update(environment=str(destination), stage="environment", backends=backends)
    process_env = os.environ.copy()
    windows = sys.platform == "win32"
    path_separator = ";" if windows else os.pathsep
    dll_paths = []
    if windows:
        dll_paths.append(destination / "Lib/site-packages/torch/lib")
        if args.tensorrt_root:
            trt_root = args.tensorrt_root.expanduser().absolute()
            dll_paths.extend(
                path for path in (trt_root / "bin", trt_root / "lib") if path.is_dir()
            )
        if process_env.get("CUDA_PATH"):
            cuda_bin = Path(process_env["CUDA_PATH"]) / "bin"
            if cuda_bin.is_dir():
                dll_paths.append(cuda_bin)
    # A caller's import/venv settings must not redirect the new interpreter.
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        process_env.pop(name, None)
    log_path = state / "setup.log"
    result["log"] = str(log_path)
    with log_path.open("w") as log:

        def run(arguments):
            print(shlex.join(map(str, arguments)), file=log, flush=True)
            print(f"setup-env: {result['stage']} (log: {log_path})", file=sys.stderr)
            completed = subprocess.run(
                list(map(str, arguments)),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=process_env,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Environment {result['stage']} failed (exit {completed.returncode}); see {log_path}. The partial environment is retained; retry with a fresh directory."
                )

        run([python, "-m", "venv", destination])
        scripts = destination / ("Scripts" if windows else "bin")
        venv_python = scripts / ("python.exe" if windows else "python")
        process_env["VIRTUAL_ENV"] = str(destination)
        process_env["PATH"] = path_separator.join(
            [str(scripts), *map(str, dll_paths), process_env.get("PATH", "")]
        )
        result.update(python=str(venv_python), stage="dependencies")
        # pip's source build writes egg-info in its input directory. Work on a
        # private copy so setup never modifies the source checkout.
        if package.is_dir():
            source = state / "builder-source"
            shutil.copytree(
                package,
                source,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    "*.egg-info",
                    "build",
                    "dist",
                    ".venv",
                    ".pytest_cache",
                    ".ruff_cache",
                ),
            )
            package = source
        install = [venv_python, "-m", "pip", "install", f"{package}[export]"]
        if "tensorrt" in backends:
            # The generic TensorRT extra selects upstream's default CUDA major.
            # Select a variant here so a CUDA 12 SDK never silently gets cu13.
            version = f"=={trt_version}" if trt_version else ">=10"
            install.extend(
                [
                    "onnx>=1.16",
                    "onnxscript>=0.3",
                    f"tensorrt-cu{args.tensorrt_cuda_major}{version}",
                ]
            )
        for path in requirements:
            install.extend(["-r", path])
        run(install)
        run([venv_python, "-m", "pip", "check"])
        _executable(
            scripts
            / (
                "physicsnemo-model-builder.exe"
                if windows
                else "physicsnemo-model-builder"
            ),
            "Installed Model Builder",
        )
        run(
            [
                venv_python,
                "-c",
                _DEPENDENCY_PROBE,
                args.tensorrt_cuda_major if "tensorrt" in backends else "",
                trt_version or "",
                "aoti" if "aoti" in backends else "",
            ]
        )
        if runtime is None:
            result["stage"] = "runtime"
            build = state / "runtime"
            configure = [
                cmake,
                "-S",
                sdk,
                "-B",
                build,
                "-G",
                "Visual Studio 17 2022" if windows else "Unix Makefiles",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DPNMIR_BUILD_TESTS=OFF",
                f"-DPython3_EXECUTABLE={venv_python}",
            ]
            if windows:
                configure.extend(["-A", "x64"])
                # Explicit -G bypasses CMake's generator environment defaults.
                toolset = os.environ.get("CMAKE_GENERATOR_TOOLSET")
                if toolset:
                    configure.extend(["-T", toolset])
                instance = os.environ.get("CMAKE_GENERATOR_INSTANCE")
                if instance:
                    configure.append(f"-DCMAKE_GENERATOR_INSTANCE={instance}")
            configure.extend(
                f"-DPNMIR_ENABLE_{backend.upper()}={'ON' if backend in backends else 'OFF'}"
                for backend in ("aoti", "tensorrt")
            )
            if args.tensorrt_root:
                configure.append(
                    f"-DPNMIR_TENSORRT_ROOT={args.tensorrt_root.expanduser().absolute()}"
                )
            run(configure)
            run(
                [
                    cmake,
                    "--build",
                    build,
                    "--config",
                    "Release",
                    "--target",
                    "pnmir_cli",
                    "--parallel",
                    "2",
                ]
            )
            runtime = build / (
                "Release/physicsnemo-infer.exe" if windows else "physicsnemo-infer"
            )
        runtime = _executable(runtime, "Native runtime")
        build_command = (
            [str(venv_python), "-m", "pnmir_build.cli"]
            if windows
            else [str(venv_python), str(scripts / "physicsnemo-model-builder")]
        ) + [
            "build",
            str(args.build.expanduser().absolute()) if project else ".",
            "--executor",
            "local",
            "--runtime",
            str(runtime),
        ]
        for backend in backends:
            build_command.extend(["--backend", backend])
        build_command.append("--json")
        if windows:
            # Keep dependency DLLs available to the standalone C++ process
            # after setup exits. Venv activation still owns PATH restoration.
            def quote_powershell(value):
                return "'" + str(value).replace("'", "''") + "'"

            activation = state / "activate.ps1"
            content = f". {quote_powershell(scripts / 'Activate.ps1')}\n"
            if dll_paths:
                prefix = path_separator.join(map(str, dll_paths)) + path_separator
                content += f"$env:PATH = {quote_powershell(prefix)} + $env:PATH\n"
            activation.write_text(content, encoding="utf-8-sig")
            activate_command = f". {quote_powershell(activation)}"
        else:
            activate_command = f"source {shlex.quote(str(scripts / 'activate'))}"
        result.update(
            runtime=str(runtime),
            project_settings={"executor": "local", "runtime": str(runtime)},
            activate=activate_command,
            build_command=build_command,
            status="complete",
            stage="environment",
        )
        (state / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
        if project:
            result.update(status="failed", stage="build")
            print(shlex.join(build_command), file=log, flush=True)
            completed = subprocess.run(
                build_command,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                env=process_env,
            )
            try:
                child = json.loads(completed.stdout)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Model build did not return JSON; see {log_path}."
                ) from exc
            result["build"] = child
            if completed.returncode or child.get("status") != "complete":
                result["diagnostics"] = [
                    {
                        "code": "MODEL_BUILD_FAILED",
                        "message": f"The environment is ready, but the model build failed; see build diagnostics and {log_path}.",
                    }
                ]
                if not args.json:
                    print(json.dumps(result, indent=2))
                return 1
            result["status"] = "complete"
    if not args.json:
        print(json.dumps(result, indent=2))
    return 0
