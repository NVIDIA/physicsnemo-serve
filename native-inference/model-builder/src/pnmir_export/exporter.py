from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from pnmir_export.aoti_profiles import compiler_profile, validate_aoti_profile
from pnmir_export.aoti_options import SUPPORTED_AOTI_OPTIONS, validate_aoti_options

_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
}
_STANDARD_OPERATOR_NAMESPACES = frozenset(
    ("aten", "higher_order", "prims", "quantized_decomposed")
)
_RANDOM_OPERATOR_NAMES = frozenset(
    (
        "aten::bernoulli",
        "aten::cauchy",
        "aten::exponential",
        "aten::geometric",
        "aten::log_normal",
        "aten::multinomial",
        "aten::normal",
        "aten::poisson",
        "aten::rand",
        "aten::rand_like",
        "aten::randint",
        "aten::randint_like",
        "aten::randn",
        "aten::randn_like",
        "aten::random",
        "aten::randperm",
        "aten::uniform",
    )
)


def _compiler_options_metadata(options: dict[str, bool]) -> dict | None:
    if not options:
        return None
    from torch._inductor import config

    unavailable = [
        key
        for key in options
        if not hasattr(config, key) or type(getattr(config, key)) is not bool
    ]
    if unavailable:
        raise ValueError(
            f"The installed Torch {torch.__version__} does not support boolean AOTI options: "
            + ", ".join(sorted(unavailable))
        )
    effective = {
        key: getattr(config, key)
        for key in sorted(SUPPORTED_AOTI_OPTIONS)
        if hasattr(config, key) and type(getattr(config, key)) is bool
    }
    effective.update(options)
    return {
        "requested": dict(options),
        "applied": dict(options),
        "effective": effective,
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
    }


def _configure_cuda_export_environment() -> None:
    """Point Triton/AOTInductor at the active CUDA toolkit."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        return
    cuda_home = Path(CUDA_HOME)
    ptxas = cuda_home / "bin" / ("ptxas.exe" if sys.platform == "win32" else "ptxas")
    if ptxas.is_file():
        os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas))
    cuda_include = cuda_home / "include"
    if (cuda_include / "cuda.h").is_file():
        include_paths = os.environ.get("CPATH", "").split(os.pathsep)
        if str(cuda_include) not in include_paths:
            os.environ["CPATH"] = os.pathsep.join(
                path for path in (str(cuda_include), *include_paths) if path
            )
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def _tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        return tuple(value)
    raise TypeError("model outputs must be a tensor or a flat tuple/list of tensors")


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported tensor dtype: {dtype}") from error


def _validate_names(names: tuple[str, ...], expected: int, kind: str) -> None:
    if len(names) != expected:
        raise ValueError(f"expected {expected} {kind} names, got {len(names)}")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{kind} names must be non-empty and unique")


def _strict_export(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: tuple[dict[int, Any] | None, ...] | None,
) -> torch.export.ExportedProgram:
    return torch.export.export(
        model,
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def _validate_exported_program(
    exported: torch.export.ExportedProgram,
    required_operators: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    custom_namespaces: set[str] = set()
    random_operators: set[str] = set()
    for node in exported.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        namespace = getattr(target, "namespace", None)
        if namespace and namespace not in _STANDARD_OPERATOR_NAMESPACES:
            custom_namespaces.add(namespace)
        schema = getattr(target, "_schema", None)
        if (
            schema is not None
            and schema.name.removesuffix("_") in _RANDOM_OPERATOR_NAMES
        ):
            random_operators.add(str(target))

    if random_operators:
        operators = ", ".join(sorted(random_operators))
        raise ValueError(
            "exported graph contains random operators; materialize randomness as "
            f"explicit tensor inputs: {operators}"
        )
    if custom_namespaces and not required_operators:
        namespaces = ", ".join(sorted(custom_namespaces))
        raise ValueError(
            "exported graph contains undeclared custom operator namespaces: "
            f"{namespaces}"
        )
    return tuple(sorted(custom_namespaces))


def _run_isolated_aoti_package(
    artifact_path: Path,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    validation_dir: Path,
) -> tuple[torch.Tensor, ...]:
    validation_dir.mkdir()
    input_path = validation_dir / "inputs.pt"
    output_path = validation_dir / "outputs.pt"
    torch.save(tuple(value.detach().cpu() for value in inputs), input_path)
    runner = Path(__file__).with_name("_isolated_aoti_runner.py")
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--package",
            str(artifact_path),
            "--inputs",
            str(input_path),
            "--outputs",
            str(output_path),
            "--target",
            str(device),
        ],
        capture_output=True,
        text=True,
    )
    try:
        if result.returncode != 0:
            details = "\n".join(
                part for part in (result.stdout.strip(), result.stderr.strip()) if part
            )
            raise RuntimeError(
                "isolated AOTI package validation failed with exit "
                f"{result.returncode}" + (f"\n{details}" if details else "")
            )
        outputs = torch.load(output_path, map_location="cpu", weights_only=True)
        return _tensor_outputs(outputs)
    finally:
        shutil.rmtree(validation_dir, ignore_errors=True)


def _tensor_schema(
    name: str, tensor: torch.Tensor, dynamic_dimensions: set[int] | None = None
) -> dict[str, Any]:
    dynamic_dimensions = dynamic_dimensions or set()
    return {
        "name": name,
        "dtype": _dtype_name(tensor.dtype),
        "shape": [
            -1 if dimension in dynamic_dimensions else size
            for dimension, size in enumerate(tensor.shape)
        ],
    }


def _prepare_output_directory(target: Path, force: bool) -> tuple[Path, Path]:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not force:
            raise FileExistsError(f"output package already exists: {target}")
        if not target.is_dir() or not (target / "model.json").is_file():
            raise ValueError(f"refusing to replace an unrecognized package: {target}")
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    return target, work


def _publish_output_directory(target: Path, work: Path) -> None:
    if not target.exists():
        work.replace(target)
        return

    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent)
    )
    backup = backup_root / "package"
    target.replace(backup)
    try:
        work.replace(target)
    except BaseException:
        backup.replace(target)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def export_package(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    target: str = "cpu",
    force: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    dynamic_shapes: tuple[dict[int, Any] | None, ...] | None = None,
    required_operators: tuple[tuple[str, str], ...] = (),
    package_weights_as_binary_blob: bool = False,
    exported_program_path: str | Path | None = None,
    aoti_profile: str = "baseline",
    aoti_options: dict[str, bool] | None = None,
) -> Path:
    """Export, compile, validate, and atomically publish an AOTI package."""
    validate_aoti_profile(aoti_profile)
    selected_options = validate_aoti_options(
        {} if aoti_options is None else aoti_options, aoti_profile
    )
    options_metadata = _compiler_options_metadata(selected_options)
    if not model_name or not model_version:
        raise ValueError("model name and version are required")
    if not example_inputs or not all(
        isinstance(value, torch.Tensor) for value in example_inputs
    ):
        raise TypeError("example_inputs must be a non-empty tuple of tensors")
    _validate_names(input_names, len(example_inputs), "input")
    if dynamic_shapes is not None and len(dynamic_shapes) != len(example_inputs):
        raise ValueError("dynamic_shapes must have one entry per input")
    operator_ids = [operator_id for operator_id, _ in required_operators]
    if any(not operator_id or not abi for operator_id, abi in required_operators):
        raise ValueError("required operator id and ABI cannot be empty")
    if len(set(operator_ids)) != len(operator_ids):
        raise ValueError("required operator ids must be unique")

    device = torch.device(target)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("target must be cpu, cuda, or cuda:<index>")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is not available")
    if device.type == "cuda":
        _configure_cuda_export_environment()

    prepared_inputs = tuple(
        value.detach().to(device).contiguous() for value in example_inputs
    )
    model = model.eval().to(device)

    target, work = _prepare_output_directory(Path(output_dir), force)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if device.type == "cuda":
        # Match the frozen references and native runtime for both matmuls and
        # cuDNN convolutions; cuDNN has an independent TF32 policy.
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    try:
        artifact_path = work / "model.pt2"

        with torch.inference_mode(), compiler_profile(aoti_profile) as profile_metadata:
            eager_outputs = tuple(
                output.detach().cpu()
                for output in _tensor_outputs(model(*prepared_inputs))
            )
            _validate_names(output_names, len(eager_outputs), "output")
            if device.type == "cuda":
                # Eager reference execution may leave large temporary buffers
                # cached. They are not inputs to export or compilation.
                gc.collect()
                torch.cuda.empty_cache()

            exported = _strict_export(model, prepared_inputs, dynamic_shapes)
            from pnmir_export.domino_exact import REQUIRED_OPERATORS as domino_operators

            if any(operator in required_operators for operator in domino_operators):
                from pnmir_export.compat import DoMINOExactBoundaryPass

                count = DoMINOExactBoundaryPass()(exported.graph_module)
                exported.validate()
                profile_metadata["tensor_only_boundaries"] = {
                    "operator": dict(zip(("id", "abi"), domino_operators[0])),
                    "rewritten_nodes": count,
                }
            custom_operator_namespaces = _validate_exported_program(
                exported, required_operators
            )
            output_dynamic_dimensions = (None,) * len(eager_outputs)
            if dynamic_shapes is not None:
                # User outputs exclude mutation bookkeeping and retain the
                # model's output order, including repeated tensor outputs.
                graph_nodes = {node.name: node for node in exported.graph.nodes}
                output_dynamic_dimensions = tuple(
                    {
                        dimension
                        for dimension, size in enumerate(
                            graph_nodes[name].meta["val"].shape
                        )
                        if isinstance(size, torch.SymInt) and size.node.is_symbolic()
                    }
                    for name in exported.graph_signature.user_outputs
                )
                del graph_nodes
            if exported_program_path is not None:
                graph_path = Path(exported_program_path)
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                if graph_path.exists():
                    raise FileExistsError(
                        f"exported graph already exists: {graph_path}"
                    )
                torch.export.save(exported, graph_path)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if package_weights_as_binary_blob:
                from torch._inductor import config

                package_config = config.patch(
                    {"aot_inductor.package_constants_on_disk_format": ("binary_blob")}
                )
            else:
                package_config = nullcontext()
            with package_config:
                compilation = (
                    {"inductor_configs": dict(selected_options)}
                    if selected_options
                    else {}
                )
                torch._inductor.aoti_compile_and_package(
                    exported, package_path=str(artifact_path), **compilation
                )
            # AOTI package loading creates a second model instance. Release the
            # build-time CUDA graph and move eager weights off the GPU first so
            # large models can still be validated on their target device.
            del exported
            if device.type == "cuda":
                model.to("cpu")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if required_operators:
                # Declared runtime sidecars are registered in the build process.
                compiled = torch._inductor.aoti_load_package(str(artifact_path))
                compiled_outputs = tuple(
                    output.detach().cpu()
                    for output in _tensor_outputs(compiled(*prepared_inputs))
                )
                validation_mode = "in_process_declared_operators"
            else:
                compiled_outputs = _run_isolated_aoti_package(
                    artifact_path,
                    prepared_inputs,
                    device,
                    work / ".aoti-validation",
                )
                validation_mode = "isolated_process"

        if len(compiled_outputs) != len(eager_outputs):
            raise RuntimeError("compiled output count differs from eager output count")
        for name, eager, actual in zip(
            output_names, eager_outputs, compiled_outputs, strict=True
        ):
            try:
                torch.testing.assert_close(actual, eager, rtol=rtol, atol=atol)
            except AssertionError as error:
                difference = (actual.to(torch.float64) - eager.to(torch.float64)).abs()
                max_abs = float(difference.max().item()) if difference.numel() else 0.0
                denominator = torch.linalg.vector_norm(eager.to(torch.float64))
                relative_l2 = float(
                    (
                        torch.linalg.vector_norm(difference) / denominator
                        if denominator > 0
                        else torch.linalg.vector_norm(difference)
                    ).item()
                )
                raise AssertionError(
                    f"compiled output {name!r} failed parity: "
                    f"max_abs={max_abs:.8g}, relative_l2={relative_l2:.8g}; "
                    f"torch assert_close limits were rtol={rtol:.1e}, atol={atol:.1e}"
                ) from error

        torch_version = torch.__version__.split("+", maxsplit=1)[0]
        manifest = {
            "format_version": 1,
            "model": {"name": model_name, "version": model_version},
            "producer": {
                "name": "pnm-ir-export",
                "version": "0.1.0",
                "torch_version": torch_version,
                "torch_export_strict": True,
                "aoti_validation": validation_mode,
            },
            "inputs": [
                _tensor_schema(
                    name,
                    tensor,
                    set(dynamic_shapes[index])
                    if dynamic_shapes is not None and dynamic_shapes[index]
                    else None,
                )
                for index, (name, tensor) in enumerate(
                    zip(input_names, prepared_inputs, strict=True)
                )
            ],
            "outputs": [
                _tensor_schema(name, tensor, dynamic_dimensions)
                for name, tensor, dynamic_dimensions in zip(
                    output_names, eager_outputs, output_dynamic_dimensions, strict=True
                )
            ],
            "artifacts": [
                {
                    "backend": "aoti",
                    "target": device.type,
                    "precision": "fp32",
                    "runtime_version": torch_version,
                    "path": "model.pt2",
                    "torch_operator_namespaces": list(custom_operator_namespaces),
                    **(
                        {"correctness_profile": profile_metadata}
                        if aoti_profile != "baseline"
                        else {}
                    ),
                    "required_operators": [
                        {"id": operator_id, "abi": abi}
                        for operator_id, abi in required_operators
                    ],
                }
            ],
        }
        if options_metadata is not None:
            manifest["artifacts"][0]["compiler_options"] = options_metadata
        (work / "model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_output_directory(target, work)
        return target
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)
