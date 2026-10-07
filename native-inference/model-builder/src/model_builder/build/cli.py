"""Framework-free command orchestration for PhysicsNeMo Model Builder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from . import completion, inputs
from .completion import (
    _completion_file as _completion_file,
    _completion_inventory as _completion_inventory,
    _verify_completion_file as _verify_completion_file,
    _verify_target_completion as _verify_target_completion,
)
from .tensors import tensor_contracts
from model_builder.export.aoti_profiles import validate_aoti_profile
from model_builder.export.aoti_options import validate_aoti_options
from model_builder.export.tensorrt_profiles import validate_tensorrt_profile


class UsageError(ValueError):
    def __init__(self, message, *, code="INVALID_ARGUMENT"):
        super().__init__(message)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, allow_abbrev=False, **kwargs)

    def error(self, message):
        raise UsageError(message)


def assets_root() -> Path:
    module = Path(__file__).resolve()
    checkout = module.parents[3]
    if (
        module.parents[2].name == "src"
        and (checkout / "models").is_dir()
        and (checkout / "toolchain.lock.json").is_file()
    ):
        return checkout
    try:
        distribution = metadata.distribution("pnms-model-builder")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for entry in distribution.files or ():
            # Installed RECORD entries may use native Windows separators.
            if Path(entry).parts[-3:] != (
                "share",
                "pnms-model-builder",
                "toolchain.lock.json",
            ):
                continue
            lock = Path(distribution.locate_file(entry)).resolve()
            if lock.is_file() and (lock.parent / "models").is_dir():
                return lock.parent
    raise UsageError(
        "Installed model recipes are missing; reinstall pnms-model-builder."
    )


def _adapter_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise UsageError(
            "Recipe adapter must be a relative path within its recipe directory, without .. components."
        )
    return relative


def read_recipe(path: Path) -> dict:
    try:
        recipe = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise UsageError(f"Cannot read recipe {path}: {exc}") from exc
    if (
        not isinstance(recipe, dict)
        or type(recipe.get("format_version")) is not int
        or recipe["format_version"] not in (1, 2)
    ):
        raise UsageError("Unsupported recipe format_version; expected 1 or 2.")
    for field in ("name", "version", "adapter", "factory", "cases"):
        if not isinstance(recipe.get(field), str) or not recipe[field]:
            raise UsageError(f"Recipe requires {field}.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", recipe["name"]):
        raise UsageError("Recipe name must be a simple model identifier.")
    adapter = (path.parent / _adapter_relative_path(recipe["adapter"])).resolve()
    if not adapter.is_relative_to(path.parent.resolve()):
        raise UsageError("Recipe adapter must remain within its recipe directory.")
    if not adapter.is_file() or adapter.suffix != ".py":
        raise UsageError(f"Recipe adapter does not exist: {adapter}")
    supported = recipe.get("supported_backends")
    if (
        not isinstance(supported, list)
        or not supported
        or any(x not in ("aoti", "tensorrt") for x in supported)
    ):
        raise UsageError(
            "Recipe must declare supported_backends from aoti and tensorrt."
        )
    if recipe.get("default_backend") not in supported:
        raise UsageError("Recipe default_backend must be supported.")
    try:
        inputs.validate_input_spec(recipe)
        validate_aoti_profile(recipe.get("aoti_profile", "baseline"))
        validate_aoti_options(
            recipe.get("aoti_options", {}), recipe.get("aoti_profile", "baseline")
        )
        validate_tensorrt_profile(recipe.get("tensorrt_profile", "baseline"))
        tensor_contracts(recipe)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return recipe


def parser() -> argparse.ArgumentParser:
    result = _Parser(
        description="PhysicsNeMo Model Builder: export, compile and verify native model packages."
    )
    commands = result.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="List bundled model recipes without importing ML frameworks."
    )
    listing.add_argument("--json", action="store_true")
    initialize = commands.add_parser(
        "init",
        help="Create an editable model project; checkpoint selection is optional.",
    )
    initialize.add_argument("directory", nargs="?", type=Path, default=Path("."))
    initialize.add_argument("--checkpoint", type=Path)
    initialize.add_argument("--source", action="append")
    initialize.add_argument("--json", action="store_true")
    setup = commands.add_parser(
        "setup-env",
        help="Create a local Python environment and matching native runtime; optionally build a model.",
    )
    setup.add_argument(
        "directory", type=Path, help="A new virtual environment directory."
    )
    setup.add_argument(
        "--python", default=sys.executable, help="Python 3.10+ used to create the venv."
    )
    setup.add_argument(
        "--backend",
        action="append",
        choices=("aoti", "tensorrt"),
        help="Backend to prepare; repeat for both (default: project backends or aoti).",
    )
    setup.add_argument("--requirements", action="append", type=Path, default=[])
    setup.add_argument(
        "--sdk-source",
        type=Path,
        help="C++ SDK source directory containing CMakeLists.txt.",
    )
    setup.add_argument(
        "--builder-package",
        type=Path,
        help="Model Builder source directory or wheel; defaults to this checkout.",
    )
    setup.add_argument(
        "--runtime",
        type=Path,
        help="Reuse an existing compatible physicsnemo-infer instead of compiling the SDK.",
    )
    setup.add_argument(
        "--tensorrt-root",
        type=Path,
        help="TensorRT C++ SDK directory (required when compiling its backend).",
    )
    setup.add_argument(
        "--tensorrt-cuda-major",
        choices=("12", "13"),
        default="13",
        help="TensorRT Python package CUDA major; match the SDK/toolkit (default: 13).",
    )
    setup.add_argument(
        "--build",
        type=Path,
        help="Build this authoring project in the new environment after setup.",
    )
    setup.add_argument("--json", action="store_true")
    importing = commands.add_parser(
        "import-checkpoint",
        help="Convert a trusted PhysicsNeMo .mdlus checkpoint to model weights and config.",
    )
    importing.add_argument("checkpoint", type=Path)
    importing.add_argument(
        "--project",
        type=Path,
        help="Read execution settings from this project (defaults to the current project).",
    )
    importing.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Write converted files to a new directory.",
    )
    importing.add_argument("--executor", choices=("container", "local"))
    importing.add_argument(
        "--builder-image", help="Immutable builder image ID or digest."
    )
    importing.add_argument(
        "--checkpoint-sha256", help="Require the source checkpoint SHA-256."
    )
    importing.add_argument("--json", action="store_true")
    for name in ("build", "check"):
        cmd = commands.add_parser(
            name,
            help={
                "build": "Build a model package.",
                "check": "Check model execution, or validate configuration with --config-only.",
            }[name],
        )
        cmd.add_argument("model", nargs="?", default=None)
        if name == "check":
            cmd.add_argument(
                "--config-only",
                action="store_true",
                help="Validate project or recipe configuration and input identities without executing the model.",
            )
        cmd.add_argument("--profile", help="Select a profile in model-build.json.")
        cmd.add_argument(
            "--update-lock",
            action="store_true",
            help="Explicitly refresh the selected project lock entry before building.",
        )
        cmd.add_argument(
            "--json",
            action="store_true",
            help="Write one versioned JSON result; send diagnostics to stderr.",
        )
        cmd.add_argument(
            "--required-gpu-arch",
            help="Require the selected GPU architecture, for example sm90.",
        )
        cmd.add_argument("--recipe", type=Path)
        cmd.add_argument(
            "--config", type=Path, help="Override the recipe JSON config file."
        )
        cmd.add_argument(
            "--checkpoint",
            type=Path,
            help="Select a recipe tensor state dictionary.",
        )
        cmd.add_argument(
            "--checkpoint-sha256", help="Require this checkpoint SHA-256 digest."
        )
        cmd.add_argument(
            "--asset",
            action="append",
            default=[],
            metavar="NAME=FILE",
            help="Override a declared asset file; repeat for different names.",
        )
        cmd.add_argument("--backend", action="append", default=[])
        cmd.add_argument("--executor", choices=("container", "local"), default=None)
        cmd.add_argument("--device", default=None)
        cmd.add_argument(
            "--runtime",
            type=Path,
            help="Prebuilt physicsnemo-infer executable for explicit local execution.",
        )
        cmd.add_argument(
            "--builder-image",
            help="Explicit image digest for an unreleased/development builder.",
        )
        cmd.add_argument("--lock", type=Path)
        cmd.add_argument("--output", type=Path)
        cmd.add_argument("--expected-inputs", type=Path, help=argparse.SUPPRESS)
    return result


def resolve(args) -> dict:
    from . import project_run, projects, targets

    try:
        project = projects.apply_project(args)
    except (OSError, ValueError) as exc:
        raise UsageError(str(exc), code="INVALID_PROJECT") from exc
    args.executor = args.executor or "container"
    args.device = args.device or "cuda"
    if getattr(args, "update_lock", False) and (
        project is None or args.command != "build"
    ):
        raise UsageError("--update-lock requires a project build command.")
    try:
        targets.validate_target(args.device, getattr(args, "required_gpu_arch", None))
        if project is not None:
            project_run.resolve_checkpoint(args, project)
    except (OSError, ValueError) as exc:
        raise UsageError(
            str(exc), code="INVALID_PROJECT" if project else "INVALID_ARGUMENT"
        ) from exc
    plan = _resolve_selection(args)
    if getattr(args, "required_gpu_arch", None):
        plan["required_gpu_arch"] = args.required_gpu_arch
    if project is not None:
        try:
            project_run.prepare(plan, project, update=args.update_lock)
        except (OSError, ValueError) as exc:
            raise UsageError(str(exc), code="PROJECT_LOCK_MISMATCH") from exc
    if getattr(args, "expected_inputs", None):
        if project is not None:
            raise UsageError(
                "Project identities cannot be overridden by --expected-inputs."
            )
        try:
            expected, _ = inputs._configuration(
                inputs._regular_file(args.expected_inputs, "expected inputs")
            )
        except (OSError, ValueError) as exc:
            raise UsageError(str(exc)) from exc
        plan["expected_identity"] = expected
    return plan


def _resolve_selection(args) -> dict:
    root = assets_root()
    if args.recipe and args.model:
        raise UsageError("Choose either a model name or --recipe, not both.")
    model = args.model or "affine"
    if args.recipe:
        recipe_path = args.recipe.expanduser().resolve()
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model):
            raise UsageError("Unknown model name; use list to see supported recipes.")
        recipe_path = root / "models" / model / "recipe.json"
    recipe = read_recipe(recipe_path)
    assets = {}
    for value in args.asset:
        name, separator, file = value.partition("=")
        if not separator or not name or not file:
            raise UsageError(
                "Asset overrides must use NAME=FILE with a nonempty name and file."
            )
        if name in assets:
            raise UsageError(f"duplicate asset override: {name}")
        if name not in recipe.get("assets", {}):
            raise UsageError(f"unknown asset override: {name}")
        assets[name] = file
    try:
        model_inputs = inputs.resolve_inputs(
            recipe,
            recipe_path,
            config=args.config,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            assets=assets,
        )
    except (OSError, ValueError) as exc:
        raise UsageError(str(exc)) from exc
    backends = list(dict.fromkeys(args.backend or [recipe["default_backend"]]))
    unsupported = [x for x in backends if x not in recipe["supported_backends"]]
    if unsupported:
        raise UsageError(
            f"unsupported backend for {recipe['name']}: {', '.join(unsupported)}"
        )
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        raise UsageError("Device must be cpu, cuda, or cuda:<index>.")
    if "tensorrt" in backends and args.device == "cpu":
        raise UsageError("TensorRT requires a CUDA device.")
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output = (
        (args.output or Path.cwd() / "out" / "inference" / recipe["name"] / stamp)
        .expanduser()
        .resolve()
    )
    if output.exists():
        raise UsageError(
            f"Output already exists; choose a new build directory: {output}"
        )
    plan = dict(
        recipe_path=recipe_path,
        recipe=recipe,
        model_inputs=model_inputs,
        backends=backends,
        output=output,
        device=args.device,
        executor=args.executor,
    )
    return resolve_executor(args, root, plan)


def resolve_executor(args, root: Path, plan: dict, *, require_runtime=True) -> dict:
    lock_path = (args.lock or root / "toolchain.lock.json").expanduser().resolve()
    try:
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        if not isinstance(lock, dict) or lock.get("format_version") != 1:
            raise ValueError("expected format_version 1")
    except (OSError, ValueError) as exc:
        raise UsageError(f"Cannot read toolchain lock {lock_path}: {exc}") from exc
    plan["toolchain_lock"] = {
        "path": str(lock_path),
        "sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }
    if args.executor == "local":
        if not require_runtime:
            plan["selection_source"] = "local-preparation"
            return plan
        runtime = args.runtime or lock.get("runtime_path")
        if not runtime:
            raise UsageError(
                "Local execution requires --runtime pointing to a prebuilt physicsnemo-infer executable; SDK compilation is a separate bootstrap step."
            )
        runtime = Path(runtime).expanduser().resolve()
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise UsageError(f"Native --runtime is not an executable file: {runtime}")
        plan["runtime"] = runtime
        plan["selection_source"] = "argument" if args.runtime else "lock"
    else:
        image = args.builder_image or lock.get("builder_image")
        if not image:
            raise UsageError(
                "No released builder image is configured. Bootstrap the documented builder image and select --builder-image <digest>, or explicitly use --executor local --runtime <installed-physicsnemo-infer>."
            )
        if not isinstance(image, str) or not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", image
        ):
            raise UsageError(
                "Builder image must be an immutable image digest (name@sha256:... or local sha256:...); mutable tags are not accepted."
            )
        if sys.platform == "win32":
            raise UsageError(
                "Native Windows builds require --executor local; run the launcher "
                "inside WSL for Linux container execution."
            )
        if not shutil.which("docker"):
            raise UsageError(
                "Docker is unavailable; install/configure GPU container access or explicitly select local execution."
            )
        plan["image"] = image
        plan["selection_source"] = "argument" if args.builder_image else "lock"
    return plan


def _validate_container_completion(plan: dict) -> None:
    completion.validate_container_completion(plan, read_recipe=read_recipe)


def container_build(plan: dict) -> int:
    output = plan["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pnmir-input-") as directory:
        staged = Path(directory)
        recipe = plan["recipe"]
        relative = _adapter_relative_path(recipe["adapter"])
        adapter = (plan["recipe_path"].parent / relative).resolve()
        target = (staged / relative).resolve()
        if not target.is_relative_to(staged.resolve()):
            raise UsageError(
                "Recipe adapter staging destination must remain within the staging directory."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter, target)
        shutil.copy2(plan["recipe_path"], staged / "recipe.json")
        staged_inputs = inputs.stage_inputs(plan.get("model_inputs"), staged)
        command = ["docker", "run", "--rm"]
        if plan["device"].startswith("cuda"):
            command += ["--gpus", "all"]
        command += [
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={staged},dst=/inputs,readonly",
            "--mount",
            f"type=bind,src={output.parent},dst=/outputs",
            "-e",
            "USER=pnmir-builder",
            "-e",
            "LOGNAME=pnmir-builder",
            "-e",
            "TRITON_CACHE_DIR=/tmp/pnmir-triton",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/pnmir-inductor",
            "-e",
            "TORCH_EXTENSIONS_DIR=/tmp/pnmir-extensions",
            "-e",
            "XDG_CACHE_HOME=/tmp/pnmir-cache",
            plan["image"],
            "build",
            "--recipe",
            "/inputs/recipe.json",
            "--executor",
            "local",
            "--runtime",
            "/opt/physicsnemo-inference/bin/physicsnemo-infer",
            "--device",
            plan["device"],
            "--output",
            "/outputs/" + output.name,
        ]
        for backend in plan["backends"]:
            command += ["--backend", backend]
        if staged_inputs is not None:

            def container_path(descriptor):
                relative = Path(descriptor["path"]).relative_to(staged.resolve())
                return (Path("/inputs") / relative).as_posix()

            command += [
                "--config",
                container_path(staged_inputs["config"]),
                "--checkpoint",
                container_path(staged_inputs["checkpoint"]),
                "--checkpoint-sha256",
                staged_inputs["checkpoint"]["sha256"],
            ]
            for name, descriptor in staged_inputs["assets"].items():
                command += ["--asset", name + "=" + container_path(descriptor)]
        if plan.get("required_gpu_arch"):
            command += ["--required-gpu-arch", plan["required_gpu_arch"]]
        if plan.get("expected_identity") is not None:
            (staged / "expected-inputs.json").write_text(
                json.dumps(plan["expected_identity"], allow_nan=False)
            )
            command += ["--expected-inputs", "/inputs/expected-inputs.json"]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            _validate_container_completion(plan)
            if plan.get("expected_identity") is not None:
                from .worker import verify_expected_identity

                receipt = json.loads(_completion_file(output, "build.json").read_text())
                verify_expected_identity(receipt, plan["expected_identity"])
        return result.returncode


def _record_execution(plan: dict, exit_code: int) -> None:
    output = plan["output"]
    if not output.is_dir() or output.is_symlink():
        return
    receipt = {
        "format_version": 1,
        "executor": plan["executor"],
        "builder_image": plan.get("image"),
        "toolchain_lock": plan["toolchain_lock"],
        "selection_source": plan["selection_source"],
        "exit_code": exit_code,
    }
    if plan.get("required_gpu_arch"):
        receipt["target_check"] = plan.get("target_check")
    if plan.get("project"):
        from .project_run import retain

        receipt["project"] = retain(plan)
    if plan.get("model_inputs") is not None:
        receipt["model_input_sources"] = {
            field: plan["model_inputs"][field]
            for field in ("config", "checkpoint", "assets", "effective_config")
        }
    temporary = output / (".execution-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "execution.json")


def _command(args, result) -> int:
    if args.command == "setup-env":
        from .environment import command

        return command(args, result)
    if args.command == "import-checkpoint":
        from .checkpoint_import import command

        return command(args, result)
    if args.command == "init":
        from .scaffold import initialize

        try:
            initialized = initialize(
                args.directory, checkpoint=args.checkpoint, source=args.source
            )
        except (OSError, ValueError) as exc:
            raise UsageError(str(exc), code="INIT_CONFLICT") from exc
        result.update(initialized, status="initialized", stage="initialization")
        if not args.json:
            print(json.dumps(result, indent=2))
        return 0
    if args.command in ("build", "check"):
        from . import authoring

        if args.recipe is None and authoring.is_project(args.model or "."):
            return authoring.command(args, result)
        if args.command == "check" and not args.config_only:
            raise UsageError(
                "Model execution with check requires an authoring project created by init; "
                "use check --config-only to validate recipe configuration or build for native verification."
            )
    if args.command == "list":
        recipes = [
            read_recipe(p)
            for p in sorted((assets_root() / "models").glob("*/recipe.json"))
        ]
        items = [
            {
                k: r[k]
                for k in (
                    "name",
                    "version",
                    "supported_backends",
                    "default_backend",
                )
            }
            for r in recipes
        ]
        if args.json:
            result.update(models=items, status="complete")
        else:
            for item in items:
                print(
                    f"{item['name']}: {', '.join(item['supported_backends'])} (default: {item['default_backend']})"
                )
        return 0
    plan = resolve(args)
    plan["json"] = args.json
    result.update(
        output=str(plan["output"]),
        executor=plan["executor"],
        backends=plan["backends"],
        device=plan["device"],
    )
    if plan.get("project"):
        result.update(
            effective_config=plan["project"]["effective"],
            profile=plan["project"]["profile"],
        )
    if args.command == "check" and args.config_only:
        result.update(
            **{
                "status": "configuration-ok",
                "executor": plan["executor"],
                "backends": plan["backends"],
                "device": plan["device"],
                "note": "Configuration checks only; runtime GPU/dependency checks occur inside the selected builder.",
            }
        )
        if not args.json:
            print(json.dumps(result, indent=2))
        return 0
    from . import project_run, targets

    try:
        project_run.publish(plan)
    except (OSError, ValueError) as exc:
        raise UsageError(str(exc), code="PROJECT_LOCK_MISMATCH") from exc
    result["stage"] = "build"
    exit_code = 1
    try:
        if plan["executor"] == "local":
            plan["target_check"] = targets.check_target(
                plan["device"], plan.get("required_gpu_arch")
            )
        if plan["executor"] == "container":
            exit_code = container_build(plan)
        else:
            from .worker import execute_build

            receipt = execute_build(
                plan["recipe_path"],
                plan["output"],
                plan["backends"],
                plan["device"],
                plan["runtime"],
                **(
                    {"model_inputs": plan["model_inputs"]}
                    if plan.get("model_inputs") is not None
                    else {}
                ),
                **(
                    {"expected_identity": plan["expected_identity"]}
                    if plan.get("expected_identity") is not None
                    else {}
                ),
                **(
                    {"required_gpu_arch": plan["required_gpu_arch"]}
                    if plan.get("required_gpu_arch")
                    else {}
                ),
            )
            if not args.json:
                print(json.dumps(receipt, indent=2))
            exit_code = 0
        if exit_code == 0:
            _verify_target_completion(plan)
            result["status"] = "complete"
            result["reports"] = {"execution": str(plan["output"] / "execution.json")}
            build_root = plan["output"]
            if args.command == "build":
                result["artifacts"] = {
                    "package": str(build_root / "model"),
                    "graphs": str(build_root / "exported"),
                }
                result["reports"]["build"] = str(build_root / "build.json")
                result["qualification"] = {
                    "scope": "native parity on required recipe cases",
                    "scientific_cfd": "not_evaluated",
                }
        else:
            result.update(
                status="failed",
                diagnostics=[
                    {
                        "code": "BUILD_FAILED",
                        "message": f"Builder process exited with code {exit_code}.",
                        "exit_code": exit_code,
                    }
                ],
            )
        return int(exit_code != 0) if args.json else exit_code
    except BaseException:
        exit_code = 1
        raise
    finally:
        _record_execution(plan, exit_code)


def main(argv=None) -> int:
    from .results import capture_stdout

    arguments = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in arguments
    result = {
        "schema_version": 1,
        "command": arguments[0] if arguments else None,
        "status": "failed",
        "stage": "configuration",
    }
    parsed = False
    try:
        with capture_stdout(json_mode, result):
            args = parser().parse_args(arguments)
            parsed = True
            exit_code = _command(args, result)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        result.update(
            status="failed", diagnostics=[{"code": exc.code, "message": str(exc)}]
        )
        exit_code = 2
    except SystemExit as exc:
        if not parsed:
            exit_code = exc.code
            result["status"] = "complete" if exit_code == 0 else "failed"
        else:
            exit_code = 1
            message = (
                f"Builder execution raised SystemExit({exc.code!r}) before completion."
            )
            print(f"build failed: {message}", file=sys.stderr)
            result.update(
                status="failed",
                diagnostics=[
                    {"code": "BUILD_FAILED", "message": message, "type": "SystemExit"}
                ],
            )
    except Exception as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        result.update(
            status="failed",
            diagnostics=[
                {
                    "code": "BUILD_FAILED",
                    "message": str(exc),
                    "type": type(exc).__name__,
                }
            ],
        )
        exit_code = 1
    if json_mode:
        print(json.dumps(result, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    # Workers import this module by its package name; use the same UsageError
    # class for module execution and the installed console entry point.
    from model_builder.build.cli import main as entry_main

    raise SystemExit(entry_main())
