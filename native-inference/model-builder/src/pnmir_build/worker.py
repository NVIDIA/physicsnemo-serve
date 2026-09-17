"""Recipe-driven graph production and frozen native verification.

The current recipe contract is deliberately static, float32, and single-stage.
Torch and backend packages are loaded only when a build executes. The native
runtime is a supplied executable; this module never builds or substitutes it.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from typing import Any


PARITY_LIMITS = {"max_abs": 1.0e-4, "relative_l2": 1.0e-4}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, root: Path | None = None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    return {
        "path": str(path.relative_to(root)) if root else str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _inventory(directory: Path, root: Path) -> list[dict]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact symlinks are unsupported: {path}")
        if path.is_file():
            records.append(_file_identity(path, root))
    if not records:
        raise ValueError(f"artifact directory is empty: {directory}")
    return records


def _verify_retained_sources(records: list[dict], root: Path) -> None:
    for record in records:
        if _file_identity(root / record["path"], root) != record:
            raise ValueError("retained source changed during build")


def _device_spec(device: str) -> dict:
    if device == "cpu":
        return {"type": "cpu", "index": 0}
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if match:
        return {"type": "cuda", "index": int(match.group(1) or 0)}
    raise ValueError("device must be cpu, cuda, or cuda:<index>")


def _read_recipe(recipe_path: Path, backends: list[str]) -> tuple[dict, Path]:
    from pnmir_build.inputs import validate_input_spec
    from pnmir_build.tensors import tensor_contracts
    from pnmir_export.aoti_profiles import validate_aoti_profile
    from pnmir_export.aoti_options import validate_aoti_options
    from pnmir_export.tensorrt_profiles import validate_tensorrt_profile

    recipe = json.loads(recipe_path.read_text())
    if not isinstance(recipe, dict):
        raise ValueError("unsupported recipe format_version")
    validate_input_spec(recipe)
    supported = recipe.get("supported_backends", [])
    if (
        not backends
        or len(set(backends)) != len(backends)
        or any(
            value not in {"aoti", "tensorrt"} or value not in supported
            for value in backends
        )
    ):
        raise ValueError("requested backends must be unique, supported, and non-empty")
    for key in ("name", "version", "factory", "cases"):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            raise ValueError(f"recipe requires {key}")
    tensor_contracts(recipe)
    validate_aoti_profile(recipe.get("aoti_profile", "baseline"))
    validate_aoti_options(
        recipe.get("aoti_options", {}), recipe.get("aoti_profile", "baseline")
    )
    validate_tensorrt_profile(recipe.get("tensorrt_profile", "baseline"))
    adapter_value = recipe.get("adapter")
    if not isinstance(adapter_value, str):
        raise ValueError("recipe adapter must be a relative file")
    relative = Path(adapter_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("recipe adapter must stay inside its recipe directory")
    adapter = recipe_path.parent / relative
    if (
        adapter.is_symlink()
        or not adapter.is_file()
        or not adapter.resolve().is_relative_to(recipe_path.parent.resolve())
    ):
        raise ValueError("recipe adapter must be a contained regular file")
    return recipe, adapter


def _tensor_record(name: str, tensor: Any) -> dict:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise ValueError("recipe bootstrap requires float32 inputs and outputs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("reference tensor contains nonfinite values")
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(value.shape),
        "data": value.numpy().tobytes(),
    }


def _validate_tensor_contract(tensor: Any, spec: dict, kind: str) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{kind} {spec['name']} must be a tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{kind} {spec['name']} dtype must be float32")
    if spec["shape"] is not None and list(tensor.shape) != spec["shape"]:
        raise ValueError(
            f"{kind} {spec['name']} shape {list(tensor.shape)} "
            f"does not match declared shape {spec['shape']}"
        )


def _infer_tensor_specs(values: tuple, names: Any, kind: str) -> list[dict]:
    from pnmir_build.tensors import _descriptors, _names

    if not values:
        raise ValueError(f"inferred {kind}s must be non-empty")
    if names is None:
        names = [f"{kind}_{index}" for index in range(len(values))]
    _names(names, f"{kind}_names")
    if len(names) != len(values):
        raise ValueError(f"{kind} names must match the actual tensor count")
    specs = []
    for name, value in zip(names, values, strict=True):
        _validate_tensor_contract(value, {"name": name, "shape": None}, kind)
        specs.append({"name": name, "dtype": "float32", "shape": list(value.shape)})
    return _descriptors(specs, f"{kind}s")


def _prepare_model(
    recipe: dict,
    recipe_path: Path,
    device: str,
    *,
    model_inputs=None,
    infer_contract: bool = False,
) -> dict:
    import torch
    from pnmir_export.exporter import _tensor_outputs
    from pnmir_export.tensorrt_exporter import tensorrt_ieee_fp32
    from pnmir_build.tensors import tensor_contracts

    input_specs, output_specs = (
        (None, None) if infer_contract else tensor_contracts(recipe)
    )
    adapter = recipe_path.parent / recipe["adapter"]
    module_name = "_pnmir_recipe_" + _sha256(adapter)
    spec = importlib.util.spec_from_file_location(module_name, adapter)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load recipe adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assets = {}
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, recipe["factory"])
        case_factory = getattr(module, recipe["cases"])
        if recipe["format_version"] == 2:
            if model_inputs is None:
                raise ValueError(
                    "format-2 model inputs must be resolved before preparation"
                )
            assets = {
                name: Path(value["path"])
                for name, value in model_inputs["assets"].items()
            }
            with torch.device("cpu"):
                model = factory(
                    copy.deepcopy(model_inputs["config_data"]), dict(assets)
                )
            if not isinstance(model, torch.nn.Module):
                raise ValueError("recipe factory must return a torch.nn.Module")
            model.cpu()
            state = torch.load(
                model_inputs["checkpoint"]["path"],
                weights_only=True,
                map_location="cpu",
            )
            if not isinstance(state, dict) or any(
                not isinstance(name, str) or not isinstance(value, torch.Tensor)
                for name, value in state.items()
            ):
                raise ValueError("checkpoint must be a plain tensor state dictionary")
            expected = model.state_dict()
            if set(state) != set(expected):
                raise ValueError(
                    "checkpoint keys do not match model: "
                    f"missing={sorted(set(expected) - set(state))}, "
                    f"unexpected={sorted(set(state) - set(expected))}"
                )
            for name, value in state.items():
                target = expected[name]
                if not isinstance(target, torch.Tensor):
                    raise ValueError(
                        "model state must be a plain tensor state dictionary"
                    )
                if value.shape != target.shape:
                    raise ValueError(f"checkpoint shape mismatch for {name}")
                if value.dtype != target.dtype:
                    raise ValueError(f"checkpoint dtype mismatch for {name}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"checkpoint contains nonfinite weights for {name}"
                    )
            model.load_state_dict(state, strict=True)
            model.eval().to(device)
            raw_cases = case_factory(
                copy.deepcopy(model_inputs["config_data"]), dict(assets)
            )
        else:
            if model_inputs is not None:
                raise ValueError("format-1 recipes do not accept model inputs")
            model = factory().eval().to(device)
            raw_cases = case_factory()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(raw_cases, (list, tuple)) or not raw_cases:
        raise ValueError("recipe must produce at least one case")
    cases, inputs, references = [], [], []
    with tensorrt_ieee_fp32(), torch.inference_mode():
        for raw in raw_cases:
            if infer_contract and input_specs is None:
                if not isinstance(raw, tuple) or not raw:
                    raise ValueError("recipe cases must be non-empty tuples")
                input_specs = _infer_tensor_specs(
                    raw, recipe.get("input_names"), "input"
                )
            if not isinstance(raw, tuple) or len(raw) != len(input_specs):
                raise ValueError(
                    "recipe cases must be tuples with the declared input count"
                )
            for value, spec in zip(raw, input_specs, strict=True):
                _validate_tensor_contract(value, spec, "input")
            case = tuple(value.detach().to(device).contiguous() for value in raw)
            eager = _tensor_outputs(model(*case))
            if infer_contract and output_specs is None:
                output_specs = _infer_tensor_specs(
                    eager, recipe.get("output_names"), "output"
                )
            if len(eager) != len(output_specs):
                raise ValueError("reference output count does not match recipe")
            for value, spec in zip(eager, output_specs, strict=True):
                _validate_tensor_contract(value, spec, "output")
            cases.append(case)
            inputs.append(
                tuple(
                    _tensor_record(spec["name"], value)
                    for spec, value in zip(input_specs, case, strict=True)
                )
            )
            references.append(
                tuple(
                    _tensor_record(spec["name"], value)
                    for spec, value in zip(output_specs, eager, strict=True)
                )
            )
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (metadata, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    environment = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    if _device_spec(device)["type"] == "cuda":
        index = _device_spec(device)["index"]
        environment.update(
            gpu_name=torch.cuda.get_device_name(index),
            compute_capability=list(torch.cuda.get_device_capability(index)),
        )
    prepared = {
        "model": model,
        "cases": cases,
        "inputs": inputs,
        "references": references,
        "weights": {
            "kind": "model_state",
            "state_sha256": digest.hexdigest(),
            "source": (
                "retained torch-state-dict checkpoint loaded strictly on CPU"
                if model_inputs is not None
                else "recipe factory; no external checkpoint override in this bootstrap"
            ),
        },
        "environment": environment,
        "assets": assets,
    }
    # Resolve customization only after recording the original eager references.
    hook = getattr(module, "export_options", None)
    captured_hook = getattr(module, "_export_options_from_source", None)
    if captured_hook is not None:
        from functools import partial

        hook = partial(
            captured_hook, copy.deepcopy(model_inputs["config_data"]), dict(assets)
        )
    if hook is not None and not callable(hook):
        raise ValueError("adapter export_options must be callable")
    prepared["export_options"] = hook
    if infer_contract:
        prepared["tensor_contract"] = {"inputs": input_specs, "outputs": output_specs}
    return prepared


def _build_backend(
    backend: str,
    prepared: dict,
    recipe: dict,
    device: str,
    package: Path,
    exported: Path,
) -> dict:
    import torch
    from pnmir_export.exporter import export_package
    from pnmir_export.onnx_exporter import export_onnx_model
    from pnmir_export.tensorrt_builder import build_tensorrt_package
    from pnmir_export.tensorrt_exporter import tensorrt_ieee_fp32
    from pnmir_build.tensors import tensor_names
    from pnmir_export.options import ExportContext, ExportOptions
    from pnmir_export.tensorrt_profiles import (
        plugin_names,
        validate_tensorrt_profile,
    )

    exported.mkdir(parents=True)
    names = {
        "input_names": tuple(tensor_names(recipe, "inputs")),
        "output_names": tuple(tensor_names(recipe, "outputs")),
    }
    identity = {"model_name": recipe["name"], "model_version": recipe["version"]}
    model, example = prepared["model"], prepared["cases"][0]
    hook = prepared.get("export_options")
    options = (
        hook(ExportContext(backend=backend, device=device)) if hook else ExportOptions()
    )
    if not isinstance(options, ExportOptions):
        raise ValueError("adapter export_options(context) must return ExportOptions")
    if backend == "aoti":
        if options.onnx_passes:
            raise ValueError(
                "ONNX graph passes require a TensorRT export; select them by context.backend"
            )
        graph = exported / "program.pt2"
        profile = recipe.get("aoti_profile", "baseline")
        compilation = {} if profile == "baseline" else {"aoti_profile": profile}
        if "aoti_options" in recipe:
            compilation["aoti_options"] = recipe["aoti_options"]
        sidecar = prepared.get("assets", {}).get("domino_exact_ops")
        if sidecar is not None:
            from pnmir_export.domino_exact import load_exact_ops, REQUIRED_OPERATORS

            if profile != "aten-boundary-exact-v3":
                raise ValueError("domino_exact_ops requires aoti_profile=aten-boundary-exact-v3")
            load_exact_ops(Path(sidecar))
            compilation["required_operators"] = REQUIRED_OPERATORS
        export_package(
            model,
            example,
            package,
            target=device,
            exported_program_path=graph,
            **compilation,
            **names,
            **identity,
        )
        return {"format": "torch.export.ExportedProgram", "entrypoint": graph.name}
    target = _device_spec(device)
    if target["type"] != "cuda":
        raise ValueError("TensorRT requires a CUDA device")
    profile = validate_tensorrt_profile(recipe.get("tensorrt_profile", "baseline"))
    if profile == "geotransolver-exact":
        from pnmir_export.compat import FreezeScalarSigmoidGates

        options = ExportOptions(
            onnx_passes=(*options.onnx_passes, FreezeScalarSigmoidGates())
        )
    compilation = {}
    if profile != "baseline":
        names_for_profile = plugin_names(profile)
        assets = prepared.get("assets", {})
        missing = [
            f"tensorrt_{name}_plugin"
            for name in names_for_profile
            if f"tensorrt_{name}_plugin" not in assets
        ]
        if missing:
            raise ValueError(
                f"TensorRT profile {profile} requires captured plugin assets: "
                + ", ".join(missing)
            )
        compilation = {
            "profile": profile,
            "plugin_libraries": {
                name: assets[f"tensorrt_{name}_plugin"] for name in names_for_profile
            },
        }
    previous = torch.cuda.current_device()
    torch.cuda.set_device(target["index"])
    try:
        with tensorrt_ieee_fp32():
            graph = export_onnx_model(
                model,
                example,
                exported / "model.onnx",
                device=torch.device(device),
                options=options,
                **names,
            )
            build_tensorrt_package(
                graph, package, device=target["index"], **identity, **compilation
            )
    finally:
        torch.cuda.set_device(previous)
    return {"format": "onnx", "entrypoint": graph.name}


def _compare_output(
    actual: bytes, expected: dict, metadata: dict, *, require_byte_identical=False
) -> dict:
    for key in ("name", "dtype", "shape"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"native output metadata {key} mismatch for {expected['name']}"
            )
    expected_size = len(expected["data"])
    if (
        type(metadata.get("byte_size")) is not int
        or metadata["byte_size"] != expected_size
        or len(actual) != expected_size
    ):
        raise ValueError("native output byte count mismatch")
    if metadata.get("device") != {"type": "cpu", "index": 0}:
        raise ValueError("native output metadata must describe CPU result storage")
    count = math.prod(expected["shape"])
    if expected_size != count * 4:
        raise ValueError("reference byte count does not match float32 shape")
    values, reference = (
        struct.unpack(f"={count}f", actual),
        struct.unpack(f"={count}f", expected["data"]),
    )
    if not all(math.isfinite(value) for value in (*values, *reference)):
        raise ValueError("native or reference output contains nonfinite values")
    differences = [a - b for a, b in zip(values, reference, strict=True)]
    maximum = max((abs(value) for value in differences), default=0.0)
    relative = math.sqrt(sum(value * value for value in differences)) / max(
        math.sqrt(sum(value * value for value in reference)), sys.float_info.epsilon
    )
    if require_byte_identical and (
        actual != expected["data"] or maximum != 0.0 or relative != 0.0
    ):
        raise ValueError(
            "native outputs must be byte-identical for the selected exact profile: "
            f"max_abs={maximum}, relative_l2={relative}"
        )
    if maximum > PARITY_LIMITS["max_abs"] or relative > PARITY_LIMITS["relative_l2"]:
        raise ValueError(
            f"native parity failed: max_abs={maximum}, relative_l2={relative}"
        )
    return {
        "name": expected["name"],
        "dtype": expected["dtype"],
        "shape": expected["shape"],
        "max_abs": maximum,
        "relative_l2": relative,
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "reference_sha256": hashlib.sha256(expected["data"]).hexdigest(),
    }


def _native_case(
    runtime: Path,
    package: Path,
    backend: str,
    device: str,
    inputs: tuple[dict, ...],
    references: tuple[dict, ...],
    case_dir: Path,
    log: Path,
    *,
    require_byte_identical=False,
) -> dict:
    case_dir.mkdir(parents=True)
    command = [
        str(runtime),
        "run",
        str(package),
        "--backend",
        backend,
        "--device",
        device,
    ]
    for index, tensor in enumerate(inputs):
        path = case_dir / f"input-{index}.bin"
        path.write_bytes(tensor["data"])
        command.extend(("--input-file", f"{tensor['name']}={path}"))
    output_paths = []
    for index, tensor in enumerate(references):
        (case_dir / f"reference-{index}.bin").write_bytes(tensor["data"])
        path = case_dir / f"output-{index}.bin"
        output_paths.append(path)
        command.extend(("--output-file", f"{tensor['name']}={path}"))
    metadata_path = case_dir / "native-metadata.json"
    command.extend(("--output-metadata", str(metadata_path)))
    with log.open("w") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, timeout=300
        )
    if result.returncode:
        raise RuntimeError(
            f"native {backend} inference failed with exit {result.returncode}; see {log}"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1 or metadata.get("completed") is not True:
        raise ValueError("native output metadata does not confirm completed inference")
    if metadata.get("backend") != backend:
        raise ValueError("native metadata backend does not match requested backend")
    if metadata.get("execution_device") != _device_spec(device):
        raise ValueError("native metadata execution device does not match request")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(references):
        raise ValueError("native metadata output count does not match reference")
    metrics = [
        _compare_output(
            path.read_bytes(), expected, actual,
            require_byte_identical=require_byte_identical,
        )
        for path, expected, actual in zip(
            output_paths, references, outputs, strict=True
        )
    ]
    return {
        "passed": True,
        "command": command,
        "metadata": metadata,
        "outputs": metrics,
        "inputs": [
            {key: value for key, value in tensor.items() if key != "data"}
            | {"sha256": hashlib.sha256(tensor["data"]).hexdigest()}
            for tensor in inputs
        ],
    }


def verify_expected_identity(receipt, expected_identity):
    """Validate the selected project identity against captured build inputs."""
    if expected_identity is None:
        return
    allowed = {"recipe", "adapter", "runtime", "model_inputs"}
    if type(expected_identity) is not dict or set(expected_identity) - allowed:
        raise ValueError("unsupported expected project identity fields")
    for key, expected in expected_identity.items():
        if key == "model_inputs":
            actual = receipt.get(key)
        else:
            if (
                type(expected) is not dict
                or set(expected) != {"sha256", "size_bytes"}
                or not isinstance(expected["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
                or type(expected["size_bytes"]) is not int
                or expected["size_bytes"] < 0
            ):
                raise ValueError(f"invalid expected project {key} identity")
            record = (
                receipt.get("source", {}).get("adapter")
                if key == "adapter"
                else receipt.get(key)
            )
            actual = (
                {field: record.get(field) for field in ("sha256", "size_bytes")}
                if isinstance(record, dict)
                else None
            )
        try:
            matches = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
                expected, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid expected project {key} identity") from error
        if not matches:
            raise ValueError(
                f"project {key} identity differs from selected build inputs"
            )


def _verify_expected_runtime(runtime, expected_identity):
    if expected_identity is not None and "runtime" in expected_identity:
        verify_expected_identity(
            {"runtime": _file_identity(runtime)},
            {"runtime": expected_identity["runtime"]},
        )


def _verify_target_environment(environment, required_gpu_arch):
    if required_gpu_arch is None:
        return
    capability = (
        environment.get("compute_capability") if isinstance(environment, dict) else None
    )
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
        or f"sm{capability[0]}{capability[1]}" != required_gpu_arch
    ):
        raise ValueError(
            "prepared model environment differs from required GPU architecture"
        )


def execute_build(
    recipe_path: Path,
    output: Path,
    backends: list[str],
    device: str,
    runtime: Path,
    *,
    model_inputs=None,
    pre_release_check=None,
    expected_identity=None,
    required_gpu_arch=None,
) -> dict:
    from pnmir_build.inputs import (
        effective_recipe,
        input_identities,
        resolve_inputs,
        stage_inputs,
    )
    from pnmir_build.targets import validate_target
    from pnmir_export.tensorrt_profiles import requires_byte_identical

    recipe_path = recipe_path.expanduser().resolve(strict=True)
    recipe, adapter = _read_recipe(recipe_path, backends)
    if recipe["format_version"] == 1 and model_inputs is not None:
        raise ValueError("format-1 recipes do not accept model inputs")
    if model_inputs is None:
        model_inputs = resolve_inputs(recipe, recipe_path)
    validate_target(device, required_gpu_arch)
    runtime = runtime.expanduser().resolve(strict=True)
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise ValueError("native runtime must be an executable file")
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "checks", "model"):
        (output / name).mkdir()
    receipt = {
        "format_version": 1,
        "status": "building",
        "requested_backends": backends,
        "device": device,
        "recipe": _file_identity(recipe_path),
        "source": {"adapter": _file_identity(adapter)},
        "runtime": _file_identity(runtime),
        "variants": {},
    }
    if model_inputs is not None:
        receipt["model_inputs"] = input_identities(model_inputs)
        receipt["model_input_sources"] = {
            key: copy.deepcopy(model_inputs[key])
            for key in ("config", "checkpoint", "assets")
        }
    _write_json(output / "build.json", receipt)
    active = None
    try:
        verify_expected_identity(receipt, expected_identity)
        source_root = output / "source"
        source_root.mkdir()
        retained_recipe = source_root / "recipe.json"
        retained_adapter = source_root / recipe["adapter"]
        retained_adapter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe_path, retained_recipe)
        shutil.copyfile(adapter, retained_adapter)
        if (
            _sha256(retained_recipe) != receipt["recipe"]["sha256"]
            or _sha256(retained_adapter) != receipt["source"]["adapter"]["sha256"]
        ):
            raise ValueError("recipe source changed while capturing build inputs")
        recipe, _ = _read_recipe(retained_recipe, backends)
        retained_inputs = stage_inputs(model_inputs, source_root)
        if retained_inputs is not None:
            retained_recipe = source_root / "effective-recipe.json"
            _write_json(
                retained_recipe, effective_recipe(recipe, retained_inputs, source_root)
            )
            recipe, _ = _read_recipe(retained_recipe, backends)
        receipt["source"]["files"] = _inventory(source_root, output)
        if retained_inputs is None:
            prepared = _prepare_model(recipe, retained_recipe, device)
        else:
            prepared = _prepare_model(
                recipe, retained_recipe, device, model_inputs=retained_inputs
            )
        _verify_retained_sources(receipt["source"]["files"], output)
        count = len(prepared["cases"])
        if (
            count < 1
            or len(prepared["inputs"]) != count
            or len(prepared["references"]) != count
        ):
            raise ValueError(
                "recipe must provide matching non-empty cases and references"
            )
        receipt.update(
            weights=prepared["weights"],
            environment=prepared["environment"],
            case_count=count,
        )
        _verify_target_environment(receipt["environment"], required_gpu_arch)
        for backend in backends:
            active = backend
            package = output / "model" / "backends" / backend
            exported = output / "exported" / backend
            variant = {
                "status": "building",
                "package": str(package.relative_to(output / "model")),
                "package_base": "model",
                "evidence_base": "build",
            }
            receipt["variants"][backend] = variant
            _write_json(output / "build.json", receipt)
            with (
                (output / "logs" / f"{backend}-build.log").open("w") as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                variant["graph"] = _build_backend(
                    backend, prepared, recipe, device, package, exported
                )
            variant["graphs"] = _inventory(exported, output)
            variant["files"] = _inventory(package, output / "model")
            byte_identical = (
                backend == "tensorrt"
                and requires_byte_identical(recipe.get("tensorrt_profile", "baseline"))
            ) or (
                backend == "aoti"
                and recipe.get("aoti_profile") == "aten-boundary-exact-v3"
            )
            check = {
                "format_version": 1,
                "backend": backend,
                "passed": False,
                "runtime": receipt["runtime"],
                "limits": (
                    {"max_abs": 0.0, "relative_l2": 0.0}
                    if byte_identical else PARITY_LIMITS
                ),
                "cases": [],
            }
            if byte_identical:
                check["require_byte_identical"] = True
            check_path = output / "checks" / f"{backend}.json"
            _write_json(check_path, check)
            try:
                for index, (inputs, references) in enumerate(
                    zip(prepared["inputs"], prepared["references"], strict=True)
                ):
                    _verify_expected_runtime(runtime, expected_identity)
                    check["cases"].append(
                        _native_case(
                            runtime,
                            package,
                            backend,
                            device,
                            inputs,
                            references,
                            output / "checks" / backend / f"case-{index}",
                            output / "logs" / f"{backend}-case-{index}.log",
                            **({"require_byte_identical": True} if byte_identical else {}),
                        )
                    )
                    _write_json(check_path, check)
                check["passed"] = True
            except BaseException as error:
                check["error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                _write_json(check_path, check)
            variant.update(status="complete", checks=_file_identity(check_path, output))
            _write_json(output / "build.json", receipt)
        _verify_retained_sources(receipt["source"]["files"], output)
        qualification = None
        if pre_release_check is not None:
            qualification = pre_release_check(output, receipt)
            if (
                not isinstance(qualification, dict)
                or qualification.get("passed") is not True
            ):
                raise ValueError("pre-release qualification must report passed=True")
            report = qualification.get("report")
            if not isinstance(report, dict) or not isinstance(report.get("path"), str):
                raise ValueError("pre-release qualification requires a report identity")
            relative = Path(report["path"])
            model_root = output / "model"
            if (
                not report["path"]
                or relative.is_absolute()
                or ".." in relative.parts
                or not (model_root / relative)
                .resolve()
                .is_relative_to(model_root.resolve())
                or any(
                    (model_root / Path(*relative.parts[:index])).is_symlink()
                    for index in range(1, len(relative.parts) + 1)
                )
                or _file_identity(model_root / relative, model_root) != report
            ):
                raise ValueError("pre-release qualification report identity is invalid")
            receipt["qualification"] = qualification
            _verify_retained_sources(receipt["source"]["files"], output)
            for variant in receipt["variants"].values():
                _verify_retained_sources(variant["files"], output / "model")
                _verify_retained_sources(variant["graphs"], output)
                _verify_retained_sources([variant["checks"]], output)
        verify_expected_identity(receipt, expected_identity)
        _verify_expected_runtime(runtime, expected_identity)
        _verify_target_environment(receipt["environment"], required_gpu_arch)
        release = {
            "format_version": 1,
            "model": {"name": recipe["name"], "version": recipe["version"]},
            "verification": "native parity on recipe fixtures; not production CFD qualification",
            "runtime": {
                "sha256": receipt["runtime"]["sha256"],
                "size_bytes": receipt["runtime"]["size_bytes"],
                "dependency_scope": "native executable only; external backend dependencies must match the recorded build environment",
            },
            "variants": {
                backend: {key: variant[key] for key in ("package", "files")}
                for backend, variant in receipt["variants"].items()
            },
            "weights": receipt["weights"],
            "environment": receipt["environment"],
        }
        if model_inputs is not None:
            release["model_inputs"] = receipt["model_inputs"]
        if qualification is not None:
            release["qualification"] = qualification
        _write_json(output / "model" / "model-release.json", release)
        receipt["status"] = "complete"
        receipt["release"] = _file_identity(
            output / "model" / "model-release.json", output
        )
        _write_json(output / "build.json", receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        if active is not None:
            receipt["variants"][active]["status"] = "failed"
        _write_json(output / "build.json", receipt)
        raise
