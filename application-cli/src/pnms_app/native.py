"""Compile and execute shared native workflows using an installed SDK.

This adapter only orchestrates the existing SDK and WorkflowHost. It does not
import the model builder or require Rust, Redis, Docker or Lepton.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from .common import read_json


def _validate_schema(bundle: Path, value, name: str):
    path = bundle / "configuration" / f"{name}.schema.json"
    if not path.is_file():
        return
    from jsonschema import ValidationError, SchemaError, validate

    try:
        validate(value, read_json(path))
    except (ValidationError, SchemaError) as error:
        raise ValueError(
            f"native {name} schema validation failed: {error.message}"
        ) from error


def loader_environment(
    python: Path | None = None,
    sdk_prefix: Path | None = None,
    *,
    extra: Sequence[Path] = (),
) -> dict[str, str]:
    """Bind native dependencies from the explicitly selected environment."""
    directories = []
    if sdk_prefix is not None:
        directories.extend(Path(sdk_prefix) / name for name in ("lib", "lib64", "bin"))
    if python is not None:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.util,json,pathlib; "
                "s=importlib.util.find_spec('torch'); "
                "print(json.dumps(str(pathlib.Path(s.origin).parent/'lib') if s and s.origin else None))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        torch_library = json.loads(result.stdout)
        if torch_library is not None:
            directories.append(Path(torch_library))
    directories.extend(Path(path) for path in extra)
    selected = list(
        dict.fromkeys(str(path.resolve()) for path in directories if path.is_dir())
    )
    environment = os.environ.copy()
    key = (
        "PATH"
        if sys.platform == "win32"
        else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    # On Windows the system PATH remains necessary to start normal executables.
    # Unix loader search paths come only from the selected native environment;
    # do not silently prefer an unrelated global CUDA/Torch/SDK installation.
    if key == "PATH" and environment.get(key):
        selected.append(environment[key])
    environment[key] = os.pathsep.join(selected)
    return environment


def _project_configuration(project: Mapping | Path) -> tuple[dict, Path]:
    if isinstance(project, Mapping):
        return dict(project), Path(project.get("root", ".")).resolve()
    import yaml

    path = Path(project).resolve()
    descriptor = path / "pnms.yaml" if path.is_dir() else path
    configuration = yaml.safe_load(descriptor.read_text())
    if not isinstance(configuration, dict):
        raise ValueError("application project must contain an object")
    return configuration, descriptor.parent


def _execute(
    arguments: list[str], *, log: Path, environment: Mapping[str, str]
) -> None:
    with log.open("a") as stream:
        stream.write(json.dumps({"command": arguments}) + "\n")
        stream.flush()
        completed = subprocess.run(
            arguments, stdout=stream, stderr=subprocess.STDOUT, env=dict(environment)
        )
    if completed.returncode:
        raise RuntimeError(
            f"native workflow command failed with exit {completed.returncode}; diagnostics: {log}"
        )


def build_workflow(
    project: Mapping | Path,
    output: Path,
    *,
    repository: Path,
    sdk_prefix: Path,
    python: Path,
    backend: str,
    custom_source: Path | None = None,
    custom_target: str | None = None,
    tensorrt_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
    build_environment: Mapping | None = None,
) -> dict:
    """Install a shared host and standalone runner; the SDK is already built."""
    if backend not in {"aoti", "tensorrt", "onnxruntime", "mock"}:
        raise ValueError(f"unsupported native workflow backend: {backend}")
    configuration, project_root = _project_configuration(project)
    configuration["root"] = project_root
    workflow = configuration.get("workflow", {})
    template = workflow.get("template", "tensor")
    if template not in {"tensor", "custom"}:
        raise ValueError(f"unsupported native workflow template: {template}")
    if custom_source is None and workflow.get("source_dir"):
        custom_source = project_root / workflow["source_dir"]
    custom_target = custom_target or workflow.get("target")
    if template == "custom" and (custom_source is None or not custom_target):
        raise ValueError("custom workflow requires source_dir and target")
    if bool(custom_source) != bool(custom_target):
        raise ValueError("set both custom_source and custom_target")
    if (
        custom_source is not None
        and not (Path(custom_source) / "CMakeLists.txt").is_file()
    ):
        raise ValueError(f"custom workflow CMakeLists.txt is missing: {custom_source}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    install, build, log = output / "install", output / "build", output / "build.log"
    prefixes = [str(Path(sdk_prefix).resolve())]
    extra_libraries = (
        [
            Path(tensorrt_root) / suffix
            for suffix in ("lib", "lib64", "lib/x86_64-linux-gnu")
        ]
        if tensorrt_root
        else []
    )
    selected_environment = loader_environment(python, sdk_prefix, extra=extra_libraries)
    if environment is not None:
        selected_environment.update(environment)
    if backend == "aoti":
        result = subprocess.run(
            [str(python), "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
            check=True,
            capture_output=True,
            text=True,
            env=selected_environment,
        )
        prefixes.append(result.stdout.strip())
    arguments = [
        "cmake",
        "-S",
        str(Path(repository) / "native-inference/workflows/runtime"),
        "-B",
        str(build),
        f"-DCMAKE_PREFIX_PATH={';'.join(prefixes)}",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        f"-DPython3_EXECUTABLE={python}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        f"-DPNMS_ENABLE_MOCK_BACKEND={'ON' if backend == 'mock' else 'OFF'}",
    ]
    toolkit = selected_environment.get("CUDAToolkit_ROOT") or selected_environment.get(
        "CUDA_HOME"
    )
    if toolkit:
        # Torch still uses legacy FindCUDA, which ignores these environment
        # variables and otherwise infers the prefix from a PATH nvcc symlink.
        arguments.append(
            f"-DCUDA_TOOLKIT_ROOT_DIR={Path(toolkit).expanduser().absolute()}"
        )
    for candidate in ("aoti", "tensorrt", "onnxruntime"):
        arguments.append(
            f"-DPNMS_ENABLE_{candidate.upper()}={'ON' if candidate == backend else 'OFF'}"
        )
    if custom_source is not None:
        arguments += [
            f"-DPNMS_CUSTOM_WORKFLOW_SOURCE_DIR={Path(custom_source).resolve()}",
            f"-DPNMS_CUSTOM_WORKFLOW_TARGET={custom_target}",
        ]
    if tensorrt_root is not None:
        arguments.append(f"-DPNMIR_TENSORRT_ROOT={Path(tensorrt_root).resolve()}")
    from . import workflow_cache

    workflow_cache.prepare_configuration(build)
    _execute(arguments, log=log, environment=selected_environment)
    configured_snapshot = workflow_cache.configured_inputs(build)
    cache_root = (
        Path(configuration.get("_state_root", project_root)) / ".pnms/cache/workflows"
    )
    identity_arguments = dict(
        repository=repository,
        sdk_prefix=sdk_prefix,
        python=python,
        backend=backend,
        custom_source=custom_source,
        custom_target=custom_target,
        tensorrt_root=tensorrt_root,
        environment=selected_environment,
        build_environment=build_environment,
        configured=build,
        environment_overrides=environment,
    )
    identity = workflow_cache.fingerprint(
        configuration, configured_snapshot=configured_snapshot, **identity_arguments
    )
    cached = workflow_cache.restore(cache_root, identity, install)
    if cached is None:
        _execute(
            ["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"],
            log=log,
            environment=selected_environment,
        )
        _execute(
            ["cmake", "--install", str(build), "--config", "Release"],
            log=log,
            environment=selected_environment,
        )
    else:
        with log.open("a") as stream:
            stream.write(
                json.dumps({"workflow_cache": "reused", "identity": identity["key"]})
                + "\n"
            )
    runner = (
        install
        / "bin"
        / ("pnms-native-run.exe" if sys.platform == "win32" else "pnms-native-run")
    )
    library_name = (
        "physicsnemo_workflow.dll"
        if sys.platform == "win32"
        else "libphysicsnemo_workflow.dylib"
        if sys.platform == "darwin"
        else "libphysicsnemo_workflow.so"
    )
    libraries = list(install.rglob(library_name))
    if not runner.is_file() or len(libraries) != 1:
        raise RuntimeError(
            f"native install omitted runner or shared workflow library; diagnostics: {log}"
        )
    from .compatibility import capture

    compatibility = capture(
        python,
        libraries[0],
        environment=selected_environment,
        output=output / "compatibility.json",
    )
    if cached is not None and compatibility != cached["compatibility"]:
        raise ValueError(
            "workflow cache providers changed; declare external inputs in workflow.dependencies and rebuild with the matched toolchain"
        )
    if cached is None:
        current_snapshot = workflow_cache.configured_inputs(
            build, baseline=configured_snapshot
        )
        if (
            workflow_cache.fingerprint(
                configuration,
                configured_snapshot=current_snapshot,
                **identity_arguments,
            )
            != identity
        ):
            raise ValueError(
                "workflow source or toolchain changed during compilation; retry pnms build"
            )
        workflow_cache.publish(cache_root, identity, install, compatibility)
    return {
        "install_root": str(install),
        "runner": str(runner),
        "library": str(libraries[0]),
        "identity": identity["key"],
        "reused": cached is not None,
        "cache": {
            "root": str(cache_root),
            "entry": str(cache_root / identity["key"]),
            "seal": str(cache_root / identity["key"] / "seal.json"),
        },
        "sdk_prefix": str(Path(sdk_prefix).resolve()),
        "python": str(python),
        "loader_environment": {
            key: selected_environment[key]
            for key in (
                ("PATH",)
                if sys.platform == "win32"
                else ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
            )
            if key in selected_environment
        },
        "build_log": str(log),
        "system": platform.system(),
        "architecture": platform.machine(),
        "compatibility": compatibility,
    }


def _bundle_path(bundle: Path, relative: str) -> Path:
    value = Path(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {".", ".."} for part in value.parts)
    ):
        raise ValueError(f"bundle path must be relative and contained: {relative}")
    path = bundle
    for part in value.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"bundle symlinks are not admitted: {relative}")
    if not path.exists():
        raise ValueError(f"bundle path does not exist: {relative}")
    return path


def run_bundle(
    bundle: Path,
    input: Path,
    output: Path,
    *,
    inputs: Mapping[str, Path] | None = None,
    input_metadata: Mapping[str, Mapping] | None = None,
    environment: Mapping[str, str] | None = None,
    request_id: str | None = None,
) -> dict:
    """Execute plain user parameters with a trusted, attempt-owned envelope."""
    bundle = Path(bundle).resolve()
    output = Path(output).absolute()
    # Canonicalize before creating parents so aliases cannot write into the
    # sealed bundle or leak symlinks into the WorkflowHost's paths.
    output = output.parent.resolve() / output.name
    if output.resolve().is_relative_to(bundle):
        raise ValueError(
            f"attempt output directory must be outside the bundle: {output}"
        )
    descriptor = json.loads((bundle / "bundle.json").read_text())
    if descriptor.get("adapter") == "transolver_surface":
        from .cfd import run_bundle as run_cfd

        return run_cfd(
            bundle,
            input,
            output,
            inputs=inputs,
            environment=environment,
            request_id=request_id,
        )
    workflow = descriptor["workflow"]
    configuration = json.loads(_bundle_path(bundle, workflow["config"]).read_text())
    runner = _bundle_path(bundle, workflow["runner"])
    for model in configuration["models"].values():
        model["package_path"] = str(_bundle_path(bundle, model["package_path"]))
    for name, relative in configuration.get("assets", {}).items():
        configuration["assets"][name] = str(_bundle_path(bundle, relative))
    parameters = read_json(input)
    if not isinstance(parameters, dict):
        raise ValueError("native request parameters must be a JSON object")
    _validate_schema(bundle, parameters, "request")
    if output.exists():
        raise ValueError(f"attempt output directory already exists: {output}")
    supplied = inputs or {}
    captured = input_metadata or {}
    if set(captured) - set(supplied):
        raise ValueError("native input metadata must name supplied files")
    rules = descriptor.get("input_files", {})
    if set(supplied) - set(rules):
        raise ValueError("undeclared native input file")
    for alias, rule in rules.items():
        if rule["required"] and alias not in supplied:
            raise ValueError(f"missing required native input file: {alias}")
    admitted, metadata = {}, {}
    for alias, raw in supplied.items():
        path = Path(raw).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"input must be a regular non-symlink file: {alias}")
        path = path.resolve()
        size = path.stat().st_size
        identity = captured.get(alias, {})
        media = (
            identity.get("media_type")
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        if media not in rules[alias]["media_types"]:
            raise ValueError(f"native input file media type is not admitted: {alias}")
        if size > rules[alias]["max_size_bytes"]:
            raise ValueError(f"native input file exceeds declared size limit: {alias}")
        admitted[alias] = str(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if alias in captured and (
            identity.get("size_bytes") != size
            or identity.get("sha256") != digest.hexdigest()
        ):
            raise ValueError(
                f"native input file differs from captured identity: {alias}"
            )
        metadata[alias] = {
            "artifact_id": alias,
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "media_type": media,
            "source": "request",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    retained = output / ".pnms-run"
    retained.mkdir()
    config_path, request_path = retained / "workflow.json", retained / "request.json"
    config_path.write_text(json.dumps(configuration))
    request_path.write_text(
        json.dumps(
            {
                "request_id": request_id or str(uuid.uuid4()),
                "parameters": parameters,
                "output_dir": str(output),
                "inputs": admitted,
                "input_metadata": metadata,
            }
        )
    )
    result_path, startup_path = output / "result.json", retained / "startup.json"
    log = retained / "native.log"
    directories = [bundle / "workflow/lib", bundle / "workflow/lib64"]
    if sys.platform == "win32":
        directories.insert(0, bundle / "workflow/bin")
    selected_environment = loader_environment(extra=directories)
    if environment is not None:
        selected_environment.update(environment)
    loader = (
        "PATH"
        if sys.platform == "win32"
        else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    # The selected stack supplies dependencies; the sealed application always
    # supplies the workflow library, including its source-linked custom factory.
    selected_environment[loader] = os.pathsep.join(
        filter(
            None,
            [
                *map(str, directories),
                selected_environment.get(loader),
            ],
        )
    )
    _execute(
        [
            str(runner),
            "--config",
            str(config_path),
            "--request",
            str(request_path),
            "--output",
            str(result_path),
            "--startup-report",
            str(startup_path),
        ],
        log=log,
        environment=selected_environment,
    )
    result = read_json(result_path)
    _validate_schema(bundle, result.get("payload"), "result")
    return {
        "output_dir": str(output),
        "result": result,
        "result_path": str(result_path),
        "startup_path": str(startup_path),
        "log": str(log),
    }
