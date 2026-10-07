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

from model_builder.export.aoti_profiles import compiler_profile, validate_aoti_profile
from model_builder.export.aoti_options import (
    SUPPORTED_AOTI_OPTIONS,
    validate_aoti_options,
)

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


def _validate_installed_aoti_options(options: dict[str, bool]) -> None:
    if not options:
        return
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


def _compiler_options_metadata(options: dict[str, bool]) -> dict | None:
    if not options:
        return None
    from torch._inductor import config

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


def _has_active_randomness(node: torch.fx.Node) -> bool:
    target = node.target
    schema = getattr(target, "_schema", None)
    name = schema.name.removesuffix("_") if schema is not None else ""
    # Keep the compatibility guard for operators without randomness tags.
    if name in _RANDOM_OPERATOR_NAMES:
        return True
    if torch.Tag.nondeterministic_seeded not in getattr(target, "tags", ()):
        return False
    if schema is None or getattr(target, "namespace", None) != "aten":
        return True

    arguments = {
        argument.name: (
            node.args[index]
            if index < len(node.args)
            else node.kwargs.get(argument.name, argument.default_value)
        )
        for index, argument in enumerate(schema.arguments)
    }
    # ATen tags describe every mode, including inactive dropout/RReLU and
    # attention/recurrent operators with zero dropout. Only literal arguments
    # can establish that randomness is disabled for every execution.
    if arguments.get("train") is False or arguments.get("training") is False:
        return False
    for parameter in ("dropout", "dropout_p"):
        probability = arguments.get(parameter)
        if type(probability) in (int, float) and probability == 0:
            return False
    if name in (
        "aten::dropout",
        "aten::native_dropout",
        "aten::feature_dropout",
        "aten::alpha_dropout",
        "aten::feature_alpha_dropout",
    ):
        probability = arguments.get("p")
        if type(probability) in (int, float) and probability in (0, 1):
            return False
    if name == "aten::scaled_dot_product_attention":
        probability = arguments.get("dropout_p")
        if type(probability) in (int, float) and probability == 1:
            return False
    if name == "aten::_scaled_dot_product_attention_math":
        if arguments.get("dropout_mask") is not None:
            return False
    if name == "aten::rrelu":
        lower, upper = arguments.get("lower"), arguments.get("upper")
        if (
            type(lower) in (int, float)
            and type(upper) in (int, float)
            and lower == upper
        ):
            return False
    return True


def _validate_exported_program(
    exported: torch.export.ExportedProgram,
    required_operators: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    custom_namespaces: set[str] = set()
    random_operators: set[str] = set()
    graphs = [exported.graph]
    graph_module = getattr(exported, "graph_module", None)
    if graph_module is not None and isinstance(graph_module, torch.fx.GraphModule):
        graphs.extend(
            module.graph
            for module in graph_module.modules()
            if module is not graph_module and isinstance(module, torch.fx.GraphModule)
        )
    for graph in graphs:
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            namespace = getattr(target, "namespace", None)
            if namespace and namespace not in _STANDARD_OPERATOR_NAMESPACES:
                custom_namespaces.add(namespace)
            if _has_active_randomness(node):
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


def _publish_output_directory(target: Path, work: Path, *, force: bool) -> None:
    if not target.exists():
        work.replace(target)
        return
    # A competing build can publish after output preparation, while this build
    # is compiling. Recheck replacement eligibility before moving its package.
    if not force:
        raise FileExistsError(f"output package already exists: {target}")
    if not target.is_dir() or not (target / "model.json").is_file():
        raise ValueError(f"refusing to replace an unrecognized package: {target}")

    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent)
    )
    backup = backup_root / "package"
    target.replace(backup)
    try:
        work.replace(target)
    except BaseException:
        # Keep the original package recoverable if restoring it also fails.
        backup.replace(target)
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
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
    _validate_installed_aoti_options(selected_options)
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
            # Each execution may mutate its inputs or return aliases of them.
            # Keep the prepared values pristine and freeze reference storage.
            eager_outputs = tuple(
                output.detach().cpu().clone()
                for output in _tensor_outputs(
                    model(*(value.clone() for value in prepared_inputs))
                )
            )
            _validate_names(output_names, len(eager_outputs), "output")
            if device.type == "cuda":
                # Eager reference execution may leave large temporary buffers
                # cached. They are not inputs to export or compilation.
                gc.collect()
                torch.cuda.empty_cache()

            exported = _strict_export(
                model, tuple(value.clone() for value in prepared_inputs), dynamic_shapes
            )
            from model_builder.export.domino_exact import (
                REQUIRED_OPERATORS as domino_operators,
            )

            if any(operator in required_operators for operator in domino_operators):
                from model_builder.export.compat import DoMINOExactBoundaryPass

                count = DoMINOExactBoundaryPass()(exported.graph_module)
                exported.validate()
                profile_metadata["tensor_only_boundaries"] = {
                    "operator": dict(zip(("id", "abi"), domino_operators[0])),
                    "rewritten_nodes": count,
                }
            custom_operator_namespaces = _validate_exported_program(
                exported, required_operators
            )
            input_dynamic_dimensions = (None,) * len(prepared_inputs)
            output_dynamic_dimensions = (None,) * len(eager_outputs)
            if dynamic_shapes is not None:
                # The captured graph decides which declared axes are symbolic.
                # User signatures exclude parameters, buffers and mutations,
                # preserving input/output order and repeated tensor outputs.
                graph_nodes = {node.name: node for node in exported.graph.nodes}
                dynamic_dimensions = {
                    name: {
                        dimension
                        for dimension, size in enumerate(
                            graph_nodes[name].meta["val"].shape
                        )
                        if isinstance(size, torch.SymInt) and size.node.is_symbolic()
                    }
                    for name in (
                        *exported.graph_signature.user_inputs,
                        *exported.graph_signature.user_outputs,
                    )
                }
                input_dynamic_dimensions = tuple(
                    dynamic_dimensions[name]
                    for name in exported.graph_signature.user_inputs
                )
                output_dynamic_dimensions = tuple(
                    dynamic_dimensions[name]
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
                options_metadata = _compiler_options_metadata(selected_options)
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
                    output.detach().cpu().clone()
                    for output in _tensor_outputs(
                        compiled(*(value.clone() for value in prepared_inputs))
                    )
                )
                validation_mode = "in_process_declared_operators"
            else:
                compiled_outputs = _run_isolated_aoti_package(
                    artifact_path,
                    tuple(value.clone() for value in prepared_inputs),
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
                _tensor_schema(name, tensor, dynamic_dimensions)
                for name, tensor, dynamic_dimensions in zip(
                    input_names, prepared_inputs, input_dynamic_dimensions, strict=True
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
        _publish_output_directory(target, work, force=force)
        return target
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)
