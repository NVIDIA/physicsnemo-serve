"""Compose the unchanged Model Builder CLI without importing ML dependencies.

Every invocation owns its input capture and its locks. Original projects are
never passed to a command that can publish a model-build.lock.json.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from . import workflow_cache
from .common import read_json
from .compatibility import PROBE as _PROVIDER_PROBE
from .workflow_cache import _BUILD_VARIABLES, _command_tokens


class ModelError(ValueError):
    """A model preparation or delegated execution failed; evidence is retained."""


_PROBE = """
import importlib.metadata as metadata
import hashlib, json, platform, re, sys
from pathlib import Path
import torch
file_hashes = {}
def fingerprint(paths):
    records = {}
    for label, path in sorted(paths, key=lambda item: item[0]):
        if not path.is_file():
            continue
        target = path.resolve()
        if target not in file_hashes:
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            file_hashes[target] = dict(sha256=digest.hexdigest(), size_bytes=target.stat().st_size)
        records[label] = file_hashes[target]
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), file_count=len(records))

torch_root = Path(torch.__file__).resolve().parent
torch_files = [(path.relative_to(torch_root).as_posix(), path)
               for name in ('lib', 'include') for path in (torch_root / name).rglob('*')]
distributions = list(metadata.distributions())
versions = sorted((re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower(), dist.version)
                  for dist in distributions if dist.metadata.get('Name'))
trt_files = []
for dist in distributions:
    name = re.sub(r'[-_.]+', '-', dist.metadata.get('Name', '')).lower()
    if name.startswith('tensorrt'):
        for relative in dist.files or []:
            if '.so' in relative.name or relative.suffix in ('.dll', '.dylib', '.pyd', '.h', '.hpp'):
                trt_files.append((name + '/' + str(relative), Path(dist.locate_file(relative))))
if len(sys.argv) > 2 and sys.argv[2]:
    trt_root = Path(sys.argv[2])
    for directory, patterns in [('include', ('Nv*.h', 'Nv*.hpp')),
                                ('lib', ('libnvinfer*', 'libnvonnxparser*', 'libnvparsers*')),
                                ('lib64', ('libnvinfer*', 'libnvonnxparser*', 'libnvparsers*'))]:
        for pattern in patterns:
            for path in (trt_root / directory).rglob(pattern):
                trt_files.append(('sdk/' + path.relative_to(trt_root).as_posix(), path))
dependencies = dict(distributions=versions, torch=fingerprint(torch_files),
                    tensorrt=fingerprint(trt_files))
trt = None
if len(sys.argv) > 1 and sys.argv[1] == 'tensorrt':
    import tensorrt
    trt = tensorrt.__version__
else:
    for distribution in ('tensorrt', 'tensorrt-cu12', 'tensorrt-cu13'):
        try:
            trt = metadata.version(distribution)
            break
        except metadata.PackageNotFoundError:
            pass
gpus = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    gpus.append(dict(name=p.name, uuid=str(getattr(p, 'uuid', '')),
                     major=p.major, minor=p.minor, total_memory=p.total_memory))
print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
    cuda=torch.version.cuda, tensorrt=trt, torch_root=str(torch_root),
    libc=platform.libc_ver(), machine=platform.machine(), system=platform.system(),
    cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), dependencies=dependencies, gpus=gpus)))
"""
_SKIP = {"build", "builds", "dist", "env", "venv", "node_modules", "__pycache__"}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _source_revision(repository, selected=None):
    source = selected or os.environ.get("PNMS_SOURCE_SHA")
    if source is None:
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    if not isinstance(source, str) or not re.fullmatch(r"[a-f0-9]{40}", source):
        raise ModelError("model compilation source requires a full Git revision")
    return source


def _compilation_source(output):
    receipt = json.loads((output / "compilation-source.json").read_text())
    source = receipt.get("source_sha") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or not isinstance(source, str)
        or not re.fullmatch(r"[a-f0-9]{40}", source)
    ):
        raise ModelError(
            "model compilation source receipt is invalid; rebuild the model"
        )
    return source


def _inventory(root, *, sdk=False):
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            if (
                not sdk
                or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ModelError(f"Unexpected symlink in model output: {path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = {
                "sha256": _hash(path),
                "size_bytes": path.stat().st_size,
            }
            if path.is_symlink():
                records[path.relative_to(root).as_posix()]["symlink"] = os.readlink(
                    path
                )
    return records


def _executable(value, label):
    # resolve() would turn a venv python symlink into the global interpreter.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ModelError(f"{label} must be an executable file: {path}")
    return path


def _python(value):
    return _executable(value or sys.executable, "Selected Python")


def _run(command, evidence, *, runner, environment=None, cwd=None):
    evidence.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in command]
    _write(
        evidence / "command.json",
        {"command": command, "cwd": str(cwd) if cwd else None},
    )
    try:
        completed = runner(
            command,
            cwd=cwd,
            env=environment or dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        (evidence / "stderr.log").write_text(str(exc) + "\n")
        raise ModelError(
            f"Could not launch {command[0]}: {exc}; evidence: {evidence}"
        ) from exc
    (evidence / "stdout.log").write_text(completed.stdout or "")
    (evidence / "stderr.log").write_text(completed.stderr or "")
    return completed


def _json_command(command, evidence, *, statuses, runner, environment=None, cwd=None):
    completed = _run(command, evidence, runner=runner, environment=environment, cwd=cwd)
    try:
        result = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"Builder returned invalid JSON (exit {completed.returncode}); evidence: {evidence}"
        ) from exc
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ModelError(f"Unsupported Model Builder JSON result; evidence: {evidence}")
    _write(evidence / "result.json", result)
    if completed.returncode or result.get("status") not in statuses:
        messages = "; ".join(
            str(item.get("message", item)) for item in result.get("diagnostics", [])
        )
        raise ModelError(
            f"Model Builder {result.get('stage', 'execution')} failed: "
            f"{messages or result.get('status')}; evidence: {evidence}"
        )
    return result


def _builder(repository, python):
    launcher = repository / "native-inference/pnms-model-builder"
    if not launcher.is_file():
        raise ModelError(f"Model Builder checkout launcher is missing: {launcher}")
    return [str(python), str(launcher)]


def _selected_gpu(gpus, requested, environment):
    # Torch exposes the UUID payload; CUDA_VISIBLE_DEVICES and nvidia-smi need
    # NVIDIA's GPU- prefix. Preserve full GPU-/MIG- identifiers verbatim.
    gpus = [dict(item) for item in gpus]
    for item in gpus:
        if re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            item.get("uuid", ""),
        ):
            item["uuid"] = "GPU-" + item["uuid"]
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    allocation = (
        [part.strip() for part in visible.split(",")] if visible is not None else None
    )
    if not gpus:
        raise ModelError(
            "No CUDA GPU is visible in the selected Python environment; check CUDA/driver and allocation."
        )
    if requested is None:
        if len(gpus) != 1:
            raise ModelError(
                "Several GPUs are visible; select one with --gpu (logical index or allocated UUID)."
            )
        index = 0
    else:
        selector = str(requested).removeprefix("cuda:")
        if selector.isdecimal():
            index = int(selector)
        else:
            matches = [
                i
                for i, item in enumerate(gpus)
                if item.get("uuid") == selector
                or (allocation and i < len(allocation) and allocation[i] == selector)
            ]
            if len(matches) != 1:
                raise ModelError("--gpu must select one of the allocated visible GPUs.")
            index = matches[0]
        if index >= len(gpus):
            raise ModelError("--gpu index is outside the allocated visible GPUs.")
    record = dict(gpus[index])
    binding = (
        allocation[index]
        if allocation and index < len(allocation)
        else record.get("uuid") or str(index)
    )
    if binding in ("", "-1"):
        raise ModelError("Selected CUDA allocation is disabled.")
    record.update(
        binding=binding,
        logical_index=index,
        architecture=f"sm{record['major']}{record['minor']}",
    )
    return record


def _sdk_configuration(prefix):
    matches = list(
        prefix.glob("lib*/cmake/PhysicsNeMoInference/PhysicsNeMoInferenceConfig.cmake")
    )
    if len(matches) != 1:
        raise ModelError(
            f"Installed SDK development files are missing under {prefix}; "
            "a physicsnemo-infer executable alone is not a development SDK."
        )
    if not (prefix / "include/physicsnemo/inference/runtime.hpp").is_file() or not any(
        path.is_file() for path in prefix.glob("lib*/libphysicsnemo_inference.*")
    ):
        raise ModelError(f"Installed SDK headers/libraries are missing under {prefix}.")
    return matches[0]


def _valid_sdk_cache(cache):
    try:
        seal = json.loads((cache / "seal.json").read_text())
        return _inventory(cache / "sdk", sdk=True) == seal["inventory"]
    except (OSError, ValueError, KeyError):
        return False


def _sdk_source_identity(repository):
    source = repository / "native-inference/cpp-runtime"
    paths = [source / "CMakeLists.txt"]
    for directory in ("src", "app", "include", "cmake"):
        paths += [path for path in (source / directory).rglob("*") if path.is_file()]
    return {path.relative_to(source).as_posix(): _hash(path) for path in sorted(paths)}


def _sdk_compiler_inputs(environment, *, unresolved_key=None):
    """Fingerprint known file operands; unknown options must prevent SDK reuse."""
    files = {}

    def referenced(name, *, response=False):
        path = Path(name)
        # Relative inputs use compiler search paths/build working directories.
        # If that resolution or transitive inputs are unknown, rebuild safely.
        if not path.is_absolute() or not path.is_file():
            raise ValueError("unresolved compiler input")
        key = str(path)
        if key in files:
            return
        files[key] = _hash(path)
        text = path.read_text().replace("\\\n", "")
        if response:
            collect(_command_tokens(text))
        elif re.search(
            r"\b(include|include_next|import|embed|__has_include)\b", text
        ) or any(Path(str(path) + suffix).exists() for suffix in (".gch", ".pch")):
            raise ValueError("undiscovered compiler input dependencies")

    def collect(tokens):
        tokens = iter(tokens)
        forwarded = []
        for token in tokens:
            if token.startswith("-Wp,"):
                forwarded.extend(token[4:].split(","))
            elif token == "-Xpreprocessor":
                forwarded.append(next(tokens))
            else:
                forwarded.append(token)
        tokens = iter(forwarded)
        for token in tokens:
            if token.startswith("@"):
                referenced(token[1:], response=True)
            elif token in ("-D", "-U", "/D", "/U"):
                next(tokens)
            elif re.fullmatch(
                r"[-/][DU].+|-O(?:[0-3sgz]|fast)?|-g[0-3]?|-std=[\w+]+"
                r"|-f(?:PIC|pic|PIE|pie)|-pthread",
                token,
            ):
                continue
            else:
                for option in ("-include", "-imacros", "/FI"):
                    if token == option:
                        referenced(next(tokens))
                        break
                    if token.startswith(option):
                        referenced(token[len(option) :])
                        break
                else:
                    # Includes, plugins, modules and linker inputs have many
                    # spellings. Only explicitly understood options are reusable.
                    raise ValueError("unknown compiler input option")

    try:
        for name in (
            "CC",
            "CXX",
            "CUDACXX",
            "CUDAHOSTCXX",
            "CFLAGS",
            "CXXFLAGS",
            "CPPFLAGS",
            "CUDAFLAGS",
            "LDFLAGS",
        ):
            tokens = _command_tokens(environment.get(name, ""))
            collect(
                tokens[1:]
                if name in ("CC", "CXX", "CUDACXX", "CUDAHOSTCXX")
                else tokens
            )
        # Search directories can supply arbitrary/transitive headers, symlinks,
        # and generated files. Their path strings cannot prove reusable inputs.
        # Empty GCC search entries can also select the build working directory.
        if any(
            name in environment
            for name in (
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "C_INCLUDE_PATH",
                "INCLUDE",
                "SDKROOT",
            )
        ):
            raise ValueError("undiscovered compiler search-path inputs")
    except (OSError, UnicodeError, ValueError, StopIteration):
        return files | {"unresolved": unresolved_key or uuid.uuid4().hex}
    return files


def _builder_source_identity(repository):
    source = repository / "native-inference"
    builder = source / "model-builder"
    paths = [
        source / "pnms-model-builder",
        builder / "pyproject.toml",
        builder / "toolchain.lock.json",
    ]
    paths += [
        path
        for path in (builder / "src").rglob("*")
        if "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
    ]
    return {
        path.relative_to(source).as_posix(): _hash(path)
        for path in sorted(paths)
        if path.is_file()
    }


def _driver(selected, output, runner, environment):
    if selected.get("uuid", "").startswith("MIG-") or " MIG " in selected.get(
        "name", ""
    ):
        raise ModelError(
            "MIG compilation targets are not supported yet; select a full allocated GPU."
        )
    executable = shutil.which("nvidia-smi")
    if not executable:
        selected.update(driver_version=None, mig_mode="unobserved")
        return
    command = [
        executable,
        "--query-gpu=driver_version,mig.mode.current",
        "--format=csv,noheader",
        "--id=" + (selected.get("uuid") or selected["binding"]),
    ]
    completed = _run(command, output / "driver", runner=runner, environment=environment)
    if completed.returncode:
        raise ModelError(
            f"Could not query selected GPU driver; evidence: {output / 'driver'}"
        )
    fields = completed.stdout.strip().split(",")
    if len(fields) != 2 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", fields[0].strip()):
        raise ModelError(f"Invalid GPU driver probe; evidence: {output / 'driver'}")
    selected.update(driver_version=fields[0].strip(), mig_mode=fields[1].strip())


def _cuda_environment(environment, settings):
    candidates = []
    if environment.get("CUDACXX"):
        candidates.append(Path(environment["CUDACXX"]))
    for value in (
        settings.get("cuda_home"),
        environment.get("CUDA_HOME"),
        environment.get("CUDAToolkit_ROOT"),
        environment.get("CUDA_PATH"),
    ):
        if value:
            candidates.append(Path(value) / "bin/nvcc")
    found = shutil.which("nvcc")
    if found:
        candidates.append(Path(found))
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    compiler = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if environment.get("CUDACXX") and compiler != candidates[0]:
        raise ModelError(f"Configured CUDACXX is not executable: {candidates[0]}")
    if compiler:
        # A PATH tool directory may expose nvcc through a symlink. CUDA's
        # headers/libraries belong beside the real compiler, not that directory.
        # Torch/CMake also derives its toolkit prefix from CUDACXX itself.
        compiler = compiler.resolve()
        environment["CUDACXX"] = str(compiler)
        toolkit = (
            environment.get("CUDAToolkit_ROOT")
            or environment.get("CUDA_HOME")
            or environment.get("CUDA_PATH")
            or str(compiler.parent.parent)
        )
        environment.setdefault("CUDA_HOME", toolkit)
        environment.setdefault("CUDAToolkit_ROOT", toolkit)
        # Torch's legacy FindCUDA ignores the modern toolkit environment hints.
        environment.setdefault("CUDA_BIN_PATH", toolkit)


def prepare_environment(
    project,
    output,
    *,
    repository,
    python=None,
    runtime=None,
    sdk_prefix=None,
    gpu=None,
    runner=subprocess.run,
    before_compile=None,
):
    """Validate a selected environment, optionally prepare a managed venv/SDK."""
    output, repository = Path(output).absolute(), Path(repository).absolute()
    settings = project.get("build", {})
    interpreter = _python(python or settings.get("python"))
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    _cuda_environment(environment, settings)
    managed = settings.get("venv")
    if managed:
        destination = Path(managed).expanduser().absolute()
        candidate = destination / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        if not destination.exists():
            command = _builder(repository, interpreter) + [
                "setup-env",
                str(destination),
                "--backend",
                project["backend"],
                "--json",
            ]
            for requirement in settings.get("requirements", []):
                command += ["--requirements", str(Path(project["root"]) / requirement)]
            for field in ("tensorrt_root", "tensorrt_cuda_major"):
                if settings.get(field):
                    command += ["--" + field.replace("_", "-"), str(settings[field])]
            if runtime or settings.get("runtime"):
                command += ["--runtime", str(runtime or settings["runtime"])]
            _json_command(
                command,
                output / "setup",
                statuses={"complete"},
                runner=runner,
                environment=environment,
            )
        interpreter = _executable(candidate, "Managed venv Python")
    completed = _run(
        [
            interpreter,
            "-c",
            _PROBE,
            project["backend"],
            settings.get("tensorrt_root") or environment.get("TENSORRT_ROOT", ""),
        ],
        output / "probe",
        runner=runner,
        environment=environment,
    )
    if completed.returncode:
        raise ModelError(
            "Selected Python environment cannot import the model dependencies. "
            "Install Model Builder export requirements into a venv or set build.venv "
            f"to prepare one; evidence: {output / 'probe'}"
        )
    try:
        stack = json.loads(completed.stdout)
        selected = _selected_gpu(stack.pop("gpus"), gpu, environment)
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError(
            f"Invalid native environment probe; evidence: {output / 'probe'}"
        ) from exc
    if project["backend"] == "tensorrt" and not stack.get("tensorrt"):
        raise ModelError(
            "TensorRT Python bindings are missing in the selected environment."
        )
    environment["CUDA_VISIBLE_DEVICES"] = selected["binding"]
    _driver(selected, output, runner, environment)
    stack["driver_version"] = selected.get("driver_version")
    if environment.get("CUDACXX"):
        completed = _run(
            [environment["CUDACXX"], "--version"],
            output / "tools/nvcc",
            runner=runner,
            environment=environment,
        )
        if completed.returncode:
            raise ModelError(
                f"CUDA compiler is not usable; evidence: {output / 'tools/nvcc'}"
            )
        stack["cuda_compiler_version"] = completed.stdout.strip()
    if before_compile is not None:
        before_compile({"kind": "local", "stack": stack, "gpu": selected})
    prefix = (
        Path(sdk_prefix or settings.get("sdk_prefix") or output / "sdk")
        .expanduser()
        .absolute()
    )
    if not (sdk_prefix or settings.get("sdk_prefix")):
        toolchain = {}
        compiler = environment.get("CXX", "c++")
        compiler_command = (
            [compiler] if shutil.which(compiler) else _command_tokens(compiler)
        )
        toolchain_commands = (
            ("cmake", ["cmake", "--version"]),
            ("cxx", [*compiler_command, "--version"]),
        )
        for name, command in toolchain_commands:
            completed = _run(
                command, output / "tools" / name, runner=runner, environment=environment
            )
            if completed.returncode:
                raise ModelError(
                    f"Native SDK requires a working {name}; evidence: {output / 'tools' / name}"
                )
            toolchain[name] = completed.stdout.strip()
        stack["compiler"] = toolchain
        toolchain_file = environment.get("CMAKE_TOOLCHAIN_FILE")
        toolchain_paths = []
        if toolchain_file:
            path = Path(toolchain_file).expanduser()
            # CMake searches relative toolchain paths in the build, then source.
            toolchain_paths = (
                [path]
                if path.is_absolute()
                else [
                    output / "sdk-build" / path,
                    repository / "native-inference/cpp-runtime" / path,
                ]
            )
        unresolved_key = uuid.uuid4().hex
        cache_root = (
            Path(project.get("_state_root", project["root"]))
            / ".pnms/cache/environments"
        )
        prefix = cache_root / (".pending-" + uuid.uuid4().hex) / "sdk"
        build = output / "sdk-build"
        command = [
            "cmake",
            "-S",
            repository / "native-inference/cpp-runtime",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPNMIR_BUILD_TESTS=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DPython3_EXECUTABLE={interpreter}",
            f"-DPNMIR_ENABLE_AOTI={'ON' if project['backend'] == 'aoti' else 'OFF'}",
            f"-DPNMIR_ENABLE_TENSORRT={'ON' if project['backend'] == 'tensorrt' else 'OFF'}",
            f"-DCMAKE_CUDA_ARCHITECTURES={selected['major']}{selected['minor']}",
        ]
        if project["backend"] == "tensorrt":
            trt_root = settings.get("tensorrt_root") or environment.get("TENSORRT_ROOT")
            if not trt_root:
                raise ModelError(
                    "TensorRT needs full C++ headers/libraries; set build.tensorrt_root."
                )
            command += [f"-DPNMIR_TENSORRT_ROOT={trt_root}"]
        # Package selection can change in CMake's standard search directories,
        # even with an unchanged environment. Discover before looking up an SDK.
        workflow_cache.prepare_configuration(build)
        result = _run(
            command, output / "configure", runner=runner, environment=environment
        )
        if result.returncode:
            raise ModelError(
                f"Native SDK configure failed; evidence: {output / 'configure'}"
            )
        configured = None
        if not toolchain_file and "unresolved" not in _sdk_compiler_inputs(
            environment, unresolved_key=unresolved_key
        ):
            try:
                configured = workflow_cache.configured_inputs(build)
            except (OSError, ValueError, KeyError, TypeError):
                pass

        def sdk_inputs(*, snapshot=None):
            executables = {}
            for name, command in toolchain_commands:
                executable = shutil.which(command[0])
                if executable:
                    path = Path(executable).resolve()
                    executables[name] = {"path": str(path), "sha256": _hash(path)}
            compiler_inputs = _sdk_compiler_inputs(
                environment, unresolved_key=unresolved_key
            )
            if toolchain_file:
                # Arbitrary CMake can select additional file inputs; its top-level
                # hash alone cannot identify those dependencies for reuse.
                compiler_inputs["unresolved"] = unresolved_key
            discovered = {"unresolved": unresolved_key}
            if configured is not None and "unresolved" not in compiler_inputs:
                try:
                    discovered = workflow_cache._normalize(
                        snapshot
                        if snapshot is not None
                        else workflow_cache.configured_inputs(
                            build, baseline=configured
                        ),
                        [
                            (str(build.resolve()), "<build>"),
                            (str(prefix.resolve()), "<sdk>"),
                        ],
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            return {
                "source": _sdk_source_identity(repository),
                "compiler_executables": executables,
                "compiler_inputs": compiler_inputs,
                "configured": discovered,
                "toolchain_file": next(
                    (_hash(path) for path in toolchain_paths if path.is_file()), None
                ),
            }

        inputs = sdk_inputs(snapshot=configured)
        cache_key = _digest(
            {
                **inputs,
                "stack": stack,
                "backend": project["backend"],
                "tensorrt_root": settings.get("tensorrt_root"),
                "architecture": selected["architecture"],
                "compiler_environment": {
                    name: environment[name]
                    for name in _BUILD_VARIABLES
                    if name in environment
                },
            }
        )
        cache = cache_root / cache_key
        valid = _valid_sdk_cache(cache)
        if valid:
            prefix = cache / "sdk"
        else:
            prefix.parent.mkdir(parents=True)
        if not valid:
            for stage, command in (
                (
                    "compile",
                    [
                        "cmake",
                        "--build",
                        build,
                        "--config",
                        "Release",
                        "--parallel",
                        "2",
                    ],
                ),
                ("install", ["cmake", "--install", build, "--config", "Release"]),
            ):
                result = _run(
                    command, output / stage, runner=runner, environment=environment
                )
                if result.returncode:
                    raise ModelError(
                        f"Native SDK {stage} failed; evidence: {output / stage}"
                    )
            _sdk_configuration(prefix)
            inventory = _inventory(prefix, sdk=True)
            from .build import publication_lock

            with publication_lock(project):
                try:
                    stable = sdk_inputs() == inputs
                except OSError:
                    stable = False
                if not stable:
                    raise ModelError(
                        f"Native SDK inputs changed during compilation; retry preparation; evidence: {output}"
                    )
                _write(prefix.parent / "seal.json", {"inventory": inventory})
                # A concurrent build may have published a valid SDK while we compiled.
                if _valid_sdk_cache(cache):
                    shutil.rmtree(prefix.parent)
                else:
                    # Corrupt SDKs are quarantined, never repaired in place.
                    if cache.exists():
                        cache.rename(
                            cache.with_name(cache.name + ".invalid." + uuid.uuid4().hex)
                        )
                    prefix.parent.rename(cache)
            prefix = cache / "sdk"
    config = _sdk_configuration(prefix)
    executable = _executable(
        runtime
        or settings.get("runtime")
        or prefix
        / "bin"
        / ("physicsnemo-infer.exe" if sys.platform == "win32" else "physicsnemo-infer"),
        "Native runtime",
    )
    loader = [prefix / "lib", prefix / "lib64", Path(stack["torch_root"]) / "lib"]
    if sys.platform == "win32":
        loader.insert(0, prefix / "bin")
    if settings.get("tensorrt_root"):
        loader += [
            Path(settings["tensorrt_root"]) / "lib",
            Path(settings["tensorrt_root"]) / "lib64",
        ]
    loader_key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    environment[loader_key] = os.pathsep.join(map(str, loader)) + (
        os.pathsep + environment[loader_key] if environment.get(loader_key) else ""
    )
    stack["runtime_sha256"] = _hash(executable)
    stack["sdk_configuration_sha256"] = _hash(config)
    stack["sdk_inventory"] = _inventory(prefix, sdk=True)
    # A sealed compiled package is reusable only against the providers it was
    # verified with. Version strings alone do not identify rebuilt native libs.
    completed = _run(
        [interpreter, "-c", _PROVIDER_PROBE, executable],
        output / "providers",
        runner=runner,
        environment=environment,
    )
    if completed.returncode:
        raise ModelError(
            f"Cannot fingerprint native runtime providers; evidence: {output / 'providers'}"
        )
    try:
        providers = json.loads(completed.stdout)
        if providers.get("schema_version") != 1 or not isinstance(
            providers.get("system_libraries"), dict
        ):
            raise ValueError("invalid provider fingerprint")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelError(
            f"Invalid native runtime provider fingerprint; evidence: {output / 'providers'}"
        ) from exc
    stack["native_providers"] = providers
    record = {
        "kind": "local",
        "python": str(interpreter),
        "runtime": str(executable),
        "sdk_prefix": str(prefix),
        "gpu": selected,
        "stack": stack,
        "native_environment": {
            key: environment[key]
            for key in (
                "CUDACXX",
                "CUDA_HOME",
                "CUDAToolkit_ROOT",
                "CUDA_BIN_PATH",
                loader_key,
            )
            if key in environment
        },
    }
    _write(output / "environment.json", record)
    return {**record, "process_env": environment}


def _regular(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ModelError(f"Model input must be a regular file: {path}")
    return path


def _capture(path, destination, backend, profile=None):
    """Capture path-bearing authoring inputs, retaining original configuration."""
    path = Path(path)
    if path.is_dir():
        path /= "model-build.json"
    path = _regular(path)
    try:
        original_bytes = path.read_bytes()
        document = read_json(path)
    except (OSError, ValueError) as exc:
        raise ModelError(f"Cannot read model project {path}: {exc}") from exc
    profile = profile or document.get("default_profile")
    profiles = document.get("profiles", {})
    if profile is not None and profile not in profiles:
        raise ModelError(f"Unknown model profile: {profile}")
    selected = profiles.get(profile, {})
    selected_backends = selected.get("backends")
    if "backends" in selected and (
        not isinstance(selected_backends, list)
        or any(value not in ("aoti", "tensorrt") for value in selected_backends)
        or len(selected_backends) != len(set(selected_backends))
        or backend not in selected_backends
    ):
        raise ModelError(
            f"Application backend {backend} conflicts with model profile {profile}.backends"
        )
    backends = selected.get("backends", document.get("backends", ["aoti"]))
    if backend not in backends:
        raise ModelError(
            f"Application backend {backend} is not declared by model project {path}"
        )
    if document.get("format_version") == 1 and "recipe" in document:
        recipe_source = Path(document["recipe"])
        if not recipe_source.is_absolute():
            recipe_source = path.parent / recipe_source
        try:
            recipe_source = _regular(recipe_source)
            recipe_bytes = recipe_source.read_bytes()
            recipe = read_json(recipe_source)
        except (OSError, ValueError) as exc:
            raise ModelError(
                f"Cannot read model recipe {recipe_source}: {exc}"
            ) from exc
    destination.mkdir(parents=True)
    _write(destination / "original-model-build.json", document)
    original_lock = path.parent / "model-build.lock.json"
    if original_lock.exists():
        shutil.copyfile(
            _regular(original_lock), destination / "original-model-build.lock.json"
        )
    captured = json.loads(json.dumps(document))
    if profile is not None:
        captured["profiles"][profile]["backends"] = [backend]
    identities = {}

    def copy(source, target):
        source = _regular(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = _hash(source)
        shutil.copyfile(source, target)
        if _hash(target) != before or _hash(source) != before:
            raise ModelError(f"Model input changed during capture: {source}")
        identities[str(source)] = {
            "sha256": before,
            "size_bytes": target.stat().st_size,
        }
        return str(target)

    def external(value, label):
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        return copy(source, destination / "_inputs" / label / source.name)

    for label, section in [
        ("base", captured),
        *[
            ("profiles/" + name, value)
            for name, value in captured.get("profiles", {}).items()
        ],
    ]:
        if label != "base" and not re.fullmatch(
            r"profiles/[A-Za-z0-9][A-Za-z0-9_.-]*", label
        ):
            raise ModelError(f"Invalid model profile name: {label}")
        for field in ("checkpoint", "toolchain_lock"):
            if section.get(field):
                section[field] = external(section[field], f"{label}/{field}")
        if isinstance(section.get("config"), str):
            section["config"] = external(section["config"], f"{label}/config")
        if isinstance(section.get("assets"), dict):
            for name in section["assets"]:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                    raise ModelError(f"Invalid model asset name: {name}")
            section["assets"] = {
                name: external(value, f"{label}/assets/{name}")
                for name, value in section["assets"].items()
            }
        if section.get("runtime"):
            runtime = Path(section["runtime"]).expanduser()
            section["runtime"] = str(
                runtime if runtime.is_absolute() else path.parent / runtime
            )
    if document.get("format_version") == 2:
        roots = [document.get("adapter"), *document.get("source", [])]
        for selection in roots:
            if not isinstance(selection, str):
                raise ModelError(
                    "Model adapter/source paths must be relative project paths."
                )
            relative = Path(selection)
            if relative.is_absolute() or ".." in relative.parts:
                raise ModelError(
                    f"Model adapter/source must remain inside project: {selection}"
                )
            source = path.parent / relative
            cursor = path.parent
            for component in relative.parts:
                cursor /= component
                if cursor.is_symlink():
                    raise ModelError(
                        f"Model source must not include a symlink: {cursor}"
                    )
            if source.is_dir():
                (destination / relative).mkdir(parents=True, exist_ok=True)
                for directory, dirs, files in os.walk(source, followlinks=False):
                    dirs[:] = [
                        name
                        for name in dirs
                        if not name.startswith(".") and name not in _SKIP
                    ]
                    for name in dirs:
                        if (Path(directory) / name).is_symlink():
                            raise ModelError(
                                f"Model source contains a symlink: {Path(directory) / name}"
                            )
                    for name in files:
                        if name.startswith(".") or name in _SKIP:
                            continue
                        if (Path(directory) / name).is_symlink():
                            raise ModelError(
                                f"Model source contains a symlink: {Path(directory) / name}"
                            )
                        if name.endswith(".py"):
                            candidate = Path(directory) / name
                            copy(
                                candidate,
                                destination / candidate.relative_to(path.parent),
                            )
            else:
                copy(source, destination / relative)
    elif document.get("format_version") == 1 and "recipe" in document:
        recipe_hash = hashlib.sha256(recipe_bytes).hexdigest()
        identities[str(recipe_source)] = {
            "sha256": recipe_hash,
            "size_bytes": len(recipe_bytes),
        }
        for name in recipe.get("assets", {}):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ModelError(f"Invalid model recipe asset name: {name}")
        recipe_root = destination / "recipe"
        for source in recipe_source.parent.rglob("*.py"):
            if not any(
                part.startswith(".") or part in _SKIP
                for part in source.relative_to(recipe_source.parent).parts
            ):
                copy(source, recipe_root / source.relative_to(recipe_source.parent))
        for label, item in [
            ("config", recipe.get("config")),
            ("checkpoint", recipe.get("checkpoint")),
            *[
                ("asset-" + name, value)
                for name, value in recipe.get("assets", {}).items()
            ],
        ]:
            if isinstance(item, dict) and item.get("path"):
                source = Path(item["path"])
                if not source.is_absolute():
                    source = recipe_source.parent / source
                item["path"] = copy(
                    source, recipe_root / "_inputs" / label / source.name
                )
        if _hash(recipe_source) != recipe_hash:
            raise ModelError(f"Model recipe changed during capture: {recipe_source}")
        _write(recipe_root / recipe_source.name, recipe)
        captured["recipe"] = str(recipe_root / recipe_source.name)
    elif document.get("format_version") != 1:
        raise ModelError("Model project requires format_version 1 or 2.")
    _write(destination / "model-build.json", captured)
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    if _hash(path) != original_hash:
        raise ModelError(f"Model configuration changed during capture: {path}")
    identities[str(path)] = {"sha256": original_hash, "size_bytes": len(original_bytes)}
    _write(
        destination / "capture.json",
        {"source": str(path), "profile": profile, "inputs": identities},
    )
    return destination / "model-build.json", profile, identities


def build_models(
    project,
    output,
    *,
    repository,
    environment="local",
    python=None,
    runtime=None,
    sdk_prefix=None,
    gpu=None,
    runner=subprocess.run,
    cache_dir=None,
    environment_check=None,
    environment_preflight=None,
    source_sha=None,
):
    return _models(
        project,
        output,
        repository=repository,
        operation="build",
        environment=environment,
        python=python,
        runtime=runtime,
        sdk_prefix=sdk_prefix,
        gpu=gpu,
        runner=runner,
        cache_dir=cache_dir,
        environment_check=environment_check,
        environment_preflight=environment_preflight,
        source_sha=source_sha,
    )


def check_models(
    project,
    output,
    *,
    repository,
    config_only=False,
    environment="local",
    python=None,
    runtime=None,
    sdk_prefix=None,
    gpu=None,
    runner=subprocess.run,
):
    return _models(
        project,
        output,
        repository=repository,
        operation="check",
        config_only=config_only,
        environment=environment,
        python=python,
        runtime=runtime,
        sdk_prefix=sdk_prefix,
        gpu=gpu,
        runner=runner,
    )


def _selected_package(release_root, backend):
    try:
        release = json.loads((release_root / "model-release.json").read_text())
        relative = Path(release["variants"][backend]["package"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelError(
            f"Model Builder release does not identify {backend}: {release_root}"
        ) from exc
    package = release_root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or package.is_symlink()
        or not package.resolve().is_relative_to(release_root.resolve())
        or not (package / "model.json").is_file()
    ):
        raise ModelError(
            f"Model Builder {backend} package is missing or outside its release: {package}"
        )
    return package


def _model_cache_seal(cache, identity):
    try:
        seal = json.loads((cache / "seal.json").read_text())
        if (
            isinstance(seal, dict)
            and seal.get("schema_version") == 2
            and seal["identity"] == identity
            and "result" in seal
            and _inventory(cache / "output") == seal["inventory"]
            and seal.get("compilation_source_sha")
            == _compilation_source(cache / "output")
        ):
            return seal
    except (OSError, ValueError, KeyError):
        pass
    return None


def _models(
    project,
    output,
    *,
    repository,
    operation,
    environment,
    python,
    runtime,
    sdk_prefix,
    gpu,
    runner,
    config_only=False,
    cache_dir=None,
    environment_check=None,
    environment_preflight=None,
    source_sha=None,
):
    output, repository = Path(output).absolute(), Path(repository).absolute()
    compilation_source = (
        _source_revision(repository, source_sha) if operation == "build" else None
    )
    # Resolve profile/backend errors before any dependency probing or setup.
    captures = {}
    for alias, selection in project["models"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
            raise ModelError(f"Invalid model alias: {alias}")
        captures[alias] = _capture(
            selection["project"],
            output / "captures" / alias,
            project["backend"],
            selection.get("profile"),
        )
    if environment not in ("local", "container"):
        raise ModelError("Build environment must be local or container.")
    if environment == "container" and not config_only:
        return _container_models(
            project,
            output,
            captures,
            repository=repository,
            operation=operation,
            config_only=config_only,
            gpu=gpu,
            runner=runner,
            source_sha=compilation_source,
        )
    if config_only:
        context = {
            "kind": "local",
            "python": str(_python(python or project.get("build", {}).get("python"))),
            "process_env": dict(os.environ),
        }
        if runtime:
            context["runtime"] = str(runtime)
    else:
        context = prepare_environment(
            project,
            output / "environment",
            repository=repository,
            python=python,
            runtime=runtime,
            sdk_prefix=sdk_prefix,
            gpu=gpu,
            runner=runner,
            **(
                {"before_compile": environment_preflight}
                if environment_preflight is not None
                else {}
            ),
        )
    if environment_check is not None:
        environment_check(
            {key: value for key, value in context.items() if key != "process_env"}
        )
    results = {}
    builder_identity = _builder_source_identity(repository)
    for alias, (captured, profile, identities) in captures.items():
        destination = output / "model-builds" / alias
        evidence = output / "logs" / alias
        command = _builder(repository, context["python"]) + [
            operation,
            str(captured),
            "--json",
            "--executor",
            "local",
            "--backend",
            project["backend"],
        ]
        if profile:
            command += ["--profile", profile]
        if context.get("runtime"):
            command += ["--runtime", context["runtime"]]
        if not config_only:
            command += [
                "--device",
                "cuda:0",
                "--required-gpu-arch",
                context["gpu"]["architecture"],
                "--output",
                str(destination),
            ]
        else:
            command += ["--config-only"]
        if operation == "build":
            # Only the invocation-owned staged lock may be refreshed.
            command += ["--update-lock"]
        identity = {
            "inputs": identities,
            "builder": builder_identity,
            "backend": project["backend"],
            "profile": profile,
            "stack": context.get("stack"),
            "gpu": {
                key: value
                for key, value in context.get("gpu", {}).items()
                if key not in ("uuid", "binding", "logical_index")
            },
        }
        key = _digest(identity)
        cache_root = (
            Path(cache_dir or project["cache_dir"])
            if cache_dir or project.get("cache_dir")
            else None
        )
        cache = cache_root / key if cache_root else None
        reused = False
        result = None
        if operation == "build" and cache:
            try:
                seal = _model_cache_seal(cache, identity)
                if seal:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(cache / "output", destination)
                    result = seal["result"]
                    compilation_source_for_alias = seal["compilation_source_sha"]
                    reused = True
            except (OSError, ValueError, KeyError):
                pass
        if result is None:
            result = _json_command(
                command,
                evidence,
                statuses={"configuration-ok"}
                if config_only
                else {"complete"}
                if operation == "build"
                else {"checked"},
                runner=runner,
                environment=context["process_env"],
                cwd=captured.parent,
            )
            if operation == "build":
                compilation_source_for_alias = compilation_source
                _write(
                    destination / "compilation-source.json",
                    {"schema_version": 1, "source_sha": compilation_source_for_alias},
                )
        record = {
            "output_root": str(destination),
            "backend": project["backend"],
            "result": result,
            "project_capture": str(captured),
            "identity": key,
            "reused": reused,
        }
        if operation == "build":
            record["compilation_source_sha"] = compilation_source_for_alias
            package = Path(
                result.get("artifacts", {}).get("package", destination / "model")
            )
            if reused:
                package = destination / "model"
            if not package.is_dir() or not package.resolve().is_relative_to(
                destination.resolve()
            ):
                raise ModelError(
                    f"Model Builder did not produce a complete package under {destination}"
                )
            record["release_root"] = str(package)
            record["package"] = str(_selected_package(package, project["backend"]))
            if cache and not reused:
                cache.parent.mkdir(parents=True, exist_ok=True)
                staging = cache.with_name(cache.name + "." + uuid.uuid4().hex)
                staging.mkdir()
                shutil.copytree(destination, staging / "output")
                _write(
                    staging / "seal.json",
                    {
                        "schema_version": 2,
                        "identity": identity,
                        "compilation_source_sha": compilation_source_for_alias,
                        "result": result,
                        "inventory": _inventory(staging / "output"),
                    },
                )
                from .build import publication_lock

                # Explicit cache directories may be shared by multiple projects.
                with publication_lock({"root": cache.parent}):
                    # Another builder may have published while we compiled.
                    if _model_cache_seal(cache, identity):
                        shutil.rmtree(staging)
                    else:
                        if cache.exists():
                            cache.rename(
                                cache.with_name(
                                    cache.name + ".invalid." + uuid.uuid4().hex
                                )
                            )
                        try:
                            staging.rename(cache)
                        except OSError as error:
                            if (
                                error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                                and not isinstance(error, PermissionError)
                            ) or not _model_cache_seal(cache, identity):
                                raise
                            shutil.rmtree(staging)
        results[alias] = record
    return {
        "models": results,
        "environment": {k: v for k, v in context.items() if k != "process_env"},
    }


def _container_models(
    project,
    output,
    captures,
    *,
    repository,
    operation,
    config_only,
    gpu,
    runner,
    source_sha,
):
    """An outer launcher limits allocation; Builder remains in local mode."""
    settings = project.get("build", {})
    image = settings.get("builder_image")
    if not isinstance(image, str) or not (
        re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        or re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image)
    ):
        raise ModelError(
            "Container builds require immutable build.builder_image (digest or image ID)."
        )
    allocation = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is None and (not allocation or "," in allocation):
        raise ModelError(
            "Select one allocated GPU with --gpu or CUDA_VISIBLE_DEVICES for container builds."
        )
    bindings = allocation.split(",") if allocation else None
    if gpu is None:
        binding = allocation
    elif str(gpu).isdecimal() and bindings:
        try:
            binding = bindings[int(gpu)]
        except IndexError as exc:
            raise ModelError(
                "--gpu index is outside the allocated visible GPUs."
            ) from exc
    elif bindings and str(gpu) not in bindings:
        raise ModelError("--gpu must select one of the allocated visible GPUs.")
    else:
        binding = str(gpu)
    if not binding or binding in ("all", "-1") or "," in binding:
        raise ModelError("Container compilation requires one allocated GPU.")
    runtime = str(
        settings.get("runtime") or "/opt/physicsnemo-inference/bin/physicsnemo-infer"
    )
    command_prefix = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={binding}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={output},dst={output}",
        "--mount",
        f"type=bind,src={repository},dst={repository},readonly",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        image,
        "python3",
        str(repository / "native-inference/pnms-model-builder"),
    ]
    results = {}
    for alias, (captured, profile, _) in captures.items():
        destination = output / "model-builds" / alias
        command = command_prefix + [
            operation,
            str(captured),
            "--executor",
            "local",
            "--runtime",
            runtime,
            "--backend",
            project["backend"],
            "--device",
            "cuda:0",
            "--json",
        ]
        if profile:
            command += ["--profile", profile]
        if config_only:
            command += ["--config-only"]
        else:
            command += ["--output", str(destination)]
        if operation == "build":
            command += ["--update-lock"]
        result = _json_command(
            command,
            output / "logs" / alias,
            runner=runner,
            statuses={"configuration-ok"}
            if config_only
            else {"complete"}
            if operation == "build"
            else {"checked"},
        )
        record = {
            "output_root": str(destination),
            "backend": project["backend"],
            "result": result,
            "project_capture": str(captured),
            "reused": False,
        }
        if operation == "build":
            record["compilation_source_sha"] = source_sha
            package = Path(
                result.get("artifacts", {}).get("package", destination / "model")
            )
            if not package.is_dir() or not package.is_relative_to(destination):
                raise ModelError(
                    f"Container Model Builder package is missing under {destination}"
                )
            record["release_root"] = str(package)
            record["package"] = str(_selected_package(package, project["backend"]))
            _write(
                destination / "compilation-source.json",
                {"schema_version": 1, "source_sha": source_sha},
            )
        results[alias] = record
    return {
        "models": results,
        "environment": {
            "kind": "container",
            "image": image,
            "gpu": {"binding": binding},
            "python": "python3",
            "runtime": runtime,
            "sdk_prefix": str(
                settings.get("sdk_prefix") or "/opt/physicsnemo-inference"
            ),
        },
    }
