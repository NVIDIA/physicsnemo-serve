"""Customer projects with generated recipes and explicitly captured Python code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from . import inputs, project_lock, targets


WRAPPER = '''"""Generated adapter; customer callbacks are captured source assets."""
from pnmir_build.authoring_sources import create_model, create_cases
from pnmir_build.authoring_sources import export_options as _export_options_from_source
'''


def is_project(value):
    path = Path(value).expanduser()
    if path.is_dir():
        path /= "model-build.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value, _ = inputs._configuration(path)
    except (OSError, ValueError):
        return False
    return value.get("format_version") == 2


def _selected(project):
    from . import authoring_sources

    effective = project["effective"]
    root = Path(project["path"]).parent.resolve()
    source = authoring_sources.capture(
        root,
        effective.get("source", []),
        Path(effective["adapter"]).relative_to(root).as_posix(),
        exclude=[effective["output_root"]],
    )
    if any(
        name.startswith("pnm-source-") or name in ("checkpoint", "config")
        for name in effective.get("assets", {})
    ):
        raise ValueError(
            "Asset names checkpoint, config and pnm-source-* are reserved."
        )
    selected = {}
    for name, value in {
        "checkpoint": effective["checkpoint"],
        **effective.get("assets", {}),
    }.items():
        path = inputs._regular_file(value, name)
        selected[name] = {"path": str(path), **inputs._identity(path)}
    pin = effective.get("checkpoint_sha256")
    if pin and selected["checkpoint"]["sha256"] != pin:
        raise ValueError(
            "checkpoint SHA-256 mismatch; check checkpoint_sha256 in model-build.json."
        )
    config = effective.get("config", {})
    if isinstance(config, str):
        path = inputs._regular_file(config, "config")
        configuration, _ = inputs._configuration(path)
        selected["config"] = {"path": str(path), **inputs._identity(path)}
    else:
        configuration = config
    return source, selected, configuration


def _identity(project, source, selected, configuration, plan):
    effective = project["effective"]
    value = {
        "format_version": 2,
        "name": effective["name"],
        "version": effective["version"],
        "adapter": Path(effective["adapter"])
        .relative_to(Path(project["path"]).parent)
        .as_posix(),
        "source": {
            name: {k: item[k] for k in ("sha256", "size_bytes")}
            for name, item in source.items()
        },
        "inputs": {
            name: {k: item[k] for k in ("sha256", "size_bytes")}
            for name, item in selected.items()
        },
        "configuration_sha256": hashlib.sha256(
            inputs._canonical(configuration)
        ).hexdigest(),
        "backends": plan["backends"],
        "device": plan["device"],
        "executor": plan["executor"],
        "required_gpu_arch": effective.get("required_gpu_arch"),
        "input_names": effective.get("input_names"),
        "output_names": effective.get("output_names"),
        "toolchain_lock": plan["toolchain_lock"]["sha256"],
    }
    if plan.get("image"):
        value["builder_image"] = plan["image"]
    if plan.get("runtime"):
        value["runtime"] = inputs._identity(plan["runtime"])
    for field in ("aoti_profile", "aoti_options", "tensorrt_profile"):
        if field in effective:
            value[field] = effective[field]
    return value


def _snapshot(destination, project, source, selected, configuration, identity):
    effective = project["effective"]
    destination.mkdir()

    def copy(record, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(record["path"]).open("rb") as original, target.open("xb") as output:
            shutil.copyfileobj(original, output, length=1024 * 1024)
        if inputs._identity(target) != {k: record[k] for k in ("sha256", "size_bytes")}:
            raise ValueError("Project input changed while capturing: " + record["path"])
        return {"path": relative, "sha256": record["sha256"]}

    assets = {}
    source_files = {}
    for index, (relative, record) in enumerate(sorted(source.items())):
        name = f"pnm-source-{index:04d}"
        assets[name] = copy(record, f"assets/{name}/{Path(relative).name}")
        source_files[relative] = name
    for name in effective.get("assets", {}):
        if name.startswith("pnm-source-") or name in ("checkpoint", "config"):
            raise ValueError(f"Reserved asset name: {name}")
        assets[name] = copy(
            selected[name], f"assets/{name}/{Path(selected[name]['path']).name}"
        )
    checkpoint = copy(selected["checkpoint"], "checkpoint.pt")
    config = {
        "_source_files": source_files,
        "_adapter": identity["adapter"],
        "_model_config": configuration,
        "_user_assets": list(effective.get("assets", {})),
    }
    (destination / "config.json").write_bytes(inputs._canonical(config))
    (destination / "export.py").write_text(WRAPPER)
    recipe = {
        "format_version": 2,
        "name": effective["name"],
        "version": effective["version"],
        "adapter": "export.py",
        "factory": "create_model",
        "cases": "create_cases",
        "config": {"path": "config.json"},
        "checkpoint": {"format": "torch-state-dict", **checkpoint},
        "assets": assets,
        "supported_backends": effective["backends"],
        "default_backend": effective["backends"][0],
    }
    for key in (
        "input_names",
        "output_names",
        "aoti_profile",
        "aoti_options",
        "tensorrt_profile",
    ):
        if key in effective:
            recipe[key] = effective[key]
    (destination / "recipe.json").write_bytes(inputs._canonical(recipe))
    resolved = inputs.resolve_inputs(recipe, destination / "recipe.json")
    files = [
        {"path": p.relative_to(destination).as_posix(), **inputs._identity(p)}
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "files": files,
        "input_identity": identity,
        "model_inputs": inputs.input_identities(resolved),
    }
    if "runtime" in identity:
        manifest["runtime"] = identity["runtime"]
    (destination / "snapshot.json").write_bytes(inputs._canonical(manifest))
    return recipe, resolved


def _execution_command(plan, snapshot, operation):
    output = plan["output"]
    args = ["--operation", operation, "--device", plan["device"]]
    if plan.get("required_gpu_arch"):
        args += ["--required-gpu-arch", plan["required_gpu_arch"]]
    if plan["executor"] == "local":
        command = [
            sys.executable,
            "-m",
            "pnmir_build.authoring_worker",
            "--input",
            str(snapshot),
            "--output",
            str(output),
            *args,
        ]
        if plan.get("runtime"):
            command += ["--runtime", str(plan["runtime"])]
        return command
    command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
    if plan["device"].startswith("cuda"):
        command += ["--gpus", "all"]
    command += [
        "--mount",
        f"type=bind,src={snapshot},dst=/inputs,readonly",
        "--mount",
        f"type=bind,src={output.parent},dst=/outputs",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "USER=physicsnemo-builder",
        "--env",
        "LOGNAME=physicsnemo-builder",
        "--env",
        "TRITON_CACHE_DIR=/tmp/physicsnemo-triton",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/physicsnemo-inductor",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/physicsnemo-extensions",
        "--env",
        "XDG_CACHE_HOME=/tmp/physicsnemo-cache",
        "--entrypoint",
        "/opt/nvidia/nvidia_entrypoint.sh",
        plan["image"],
        "python3",
        "-m",
        "pnmir_build.authoring_worker",
        "--input",
        "/inputs",
        "--output",
        f"/outputs/{output.name}",
        *args,
    ]
    return command


def _run(plan, snapshot, operation):
    command = _execution_command(plan, snapshot, operation)
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = (
        source_root + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        completed = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        log.seek(0)
        shutil.copyfileobj(log, sys.stderr)
        if not plan["output"].exists():
            plan["output"].mkdir(parents=True, exist_ok=False)
        if plan["output"].is_dir():
            log.seek(0)
            with (plan["output"] / "execution.log").open("x") as target:
                shutil.copyfileobj(log, target)
    return completed.returncode


def _verify_report(plan, report, recipe, resolved, snapshot, identity, operation):
    from . import cli, worker
    from .tensors import tensor_contracts

    output = plan["output"]
    if (
        type(report.get("format_version")) is not int
        or report["format_version"] != 1
        or report.get("operation") != operation
        or report.get("status") != ("checked" if operation == "check" else "complete")
        or report.get("input_identity") != identity
        or report.get("device") != plan["device"]
        or type(report.get("case_count")) is not int
        or report["case_count"] < 1
    ):
        raise ValueError(
            "Model check receipt does not match the requested inputs or operation."
        )
    inventory = cli._completion_inventory(output, report["source_files"])
    contract = {"format_version": 2, **report["tensor_contract"]}
    tensor_contracts(contract)
    path = cli._completion_file(output, "source/effective-recipe.json")
    captured_recipe = cli.read_recipe(path)
    captured_inputs = inputs.resolve_inputs(captured_recipe, path)
    if inputs.input_identities(captured_inputs) != inputs.input_identities(resolved):
        raise ValueError("Retained model inputs differ from the selected project.")
    if tensor_contracts(captured_recipe) != tensor_contracts(contract):
        raise ValueError("Retained recipe differs from the checked tensor contract.")
    for field in (
        "name",
        "version",
        "adapter",
        "factory",
        "cases",
        "supported_backends",
        "default_backend",
        "aoti_profile",
        "aoti_options",
        "tensorrt_profile",
    ):
        if captured_recipe.get(field) != recipe.get(field):
            raise ValueError(f"Retained recipe changed the requested {field}.")
    for field, kind in (("input_names", "inputs"), ("output_names", "outputs")):
        if field in recipe and recipe[field] != [
            spec["name"] for spec in report["tensor_contract"][kind]
        ]:
            raise ValueError(f"Checked tensors changed the selected {field}.")
    for descriptor in (
        captured_inputs["config"],
        captured_inputs["checkpoint"],
        *captured_inputs["assets"].values(),
    ):
        relative = Path(descriptor["path"]).relative_to(output.resolve()).as_posix()
        if relative not in inventory:
            raise ValueError(
                "A retained model input is missing from the source inventory."
            )
    for relative in ("source/export.py", "source/effective-recipe.json"):
        if relative not in inventory:
            raise ValueError(
                "The generated adapter or recipe is missing from the source inventory."
            )
    if inputs._identity(
        cli._completion_file(output, "source/export.py")
    ) != inputs._identity(snapshot / "export.py"):
        raise ValueError(
            "The retained generated adapter differs from the selected builder."
        )
    if plan.get("required_gpu_arch"):
        expected = {
            "device": plan["device"],
            "actual_gpu_arch": plan["required_gpu_arch"],
            "required_gpu_arch": plan["required_gpu_arch"],
        }
        checked = report.get("target_check") or {}
        if any(checked.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Model check did not confirm the requested GPU device and architecture."
            )
        worker._verify_target_environment(
            report["environment"], plan["required_gpu_arch"]
        )
    if operation == "build":
        qualified_recipe = {
            key: value
            for key, value in recipe.items()
            if key not in ("input_names", "output_names")
        }
        qualified_recipe.update(report["tensor_contract"])
        cli._validate_container_completion(
            {**plan, "recipe": qualified_recipe, "model_inputs": resolved}
        )
        built, _ = inputs._configuration(cli._completion_file(output, "build.json"))
        for field in ("case_count", "weights", "environment"):
            if built.get(field) != report.get(field):
                raise ValueError(
                    f"Build summary differs from native verification: {field}."
                )
        if (
            identity.get("runtime")
            and {key: built["runtime"][key] for key in ("sha256", "size_bytes")}
            != identity["runtime"]
        ):
            raise ValueError(
                "Build used a different native runtime from the selected SDK."
            )
        release, _ = inputs._configuration(
            cli._completion_file(output, "model/model-release.json")
        )
        qualification = built.get("qualification")
        if (
            not isinstance(qualification, dict)
            or qualification.get("passed") is not True
            or release.get("qualification") != qualification
        ):
            raise ValueError(
                "Build is missing matching source integrity qualification."
            )
        record = qualification.get("report")
        if not isinstance(record, dict) or record.get("path") != "source-check.json":
            raise ValueError("Build source integrity report path is invalid.")
        proof, _ = inputs._configuration(
            cli._verify_completion_file(output / "model", record)
        )
        if proof.get("passed") is not True or proof.get("source") != identity["source"]:
            raise ValueError(
                "Build source integrity qualification differs from the selected source."
            )


def command(args, result):
    from . import authoring_config, cli

    if args.update_lock and args.command != "build":
        raise cli.UsageError(
            "--update-lock is supported with build; check and doctor do not publish locks."
        )
    if args.recipe or args.expected_inputs is not None:
        raise cli.UsageError(
            "Custom model projects do not accept recipe or expected-input overrides."
        )
    try:
        project = authoring_config.load(args.model or ".", args)
        effective = project["effective"]
        problems = authoring_config.missing(
            effective,
            Path(project["path"]).parent,
            require_runtime=args.command == "build",
        )
        if problems:
            result.update(status="incomplete", diagnostics=problems)
            if not args.json:
                for problem in problems:
                    print(f"{problem['field']}: {problem['message']}", file=sys.stderr)
            return 2
        targets.validate_target(effective["device"], effective.get("required_gpu_arch"))
        if effective["device"] == "cpu" and "tensorrt" in effective["backends"]:
            raise ValueError("TensorRT requires a CUDA device.")
        source, selected, configuration = _selected(project)
        execution_args = argparse.Namespace(
            executor=effective["executor"],
            runtime=Path(effective["runtime"]) if effective.get("runtime") else None,
            lock=Path(effective["toolchain_lock"])
            if effective.get("toolchain_lock")
            else None,
            builder_image=effective.get("builder_image"),
        )
        plan = cli.resolve_executor(
            execution_args,
            cli.assets_root(),
            {
                "device": effective["device"],
                "executor": effective["executor"],
                "backends": effective["backends"],
                "required_gpu_arch": effective.get("required_gpu_arch"),
            },
            require_runtime=args.command == "build",
        )
        identity = _identity(project, source, selected, configuration, plan)
    except (OSError, ValueError) as exc:
        raise cli.UsageError(str(exc), code="INVALID_PROJECT") from exc
    result.update(effective_config=effective, profile=project.get("profile"))
    if args.command == "doctor":
        result.update(
            status="configuration-ok",
            note="Configuration and input identities only; use check to execute the model.",
        )
        if not args.json:
            print(json.dumps(result, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output).expanduser().absolute()
        if args.output
        else Path(effective["output_root"]) / f"{stamp}-{uuid.uuid4().hex[:12]}"
    )
    if output.exists() or output.is_symlink():
        raise cli.UsageError(f"Output already exists; choose a new directory: {output}")
    output = output.parent.resolve() / output.name
    if any(
        Path(item["path"]).is_relative_to(output)
        for item in [*source.values(), *selected.values()]
    ):
        raise cli.UsageError(
            "The output directory cannot contain selected model inputs."
        )
    plan["output"] = output
    lock = None
    lock_path = Path(project["path"]).parent / "model-build.lock.json"
    if args.command == "build":
        key = f"build:{project['profile']}" if project.get("profile") else "build"
        try:
            lock = project_lock.inspect_lock(
                lock_path, key, identity, update=args.update_lock
            )
        except (OSError, ValueError) as exc:
            raise cli.UsageError(str(exc), code="PROJECT_LOCK_MISMATCH") from exc
    exit_code = 1
    result.update(
        output=str(output),
        stage=args.command,
        executor=plan["executor"],
        device=plan["device"],
        backends=plan["backends"],
    )
    published = None
    try:
        with tempfile.TemporaryDirectory(prefix="physicsnemo-project-") as temporary:
            snapshot = Path(temporary) / "inputs"
            recipe, resolved = _snapshot(
                snapshot, project, source, selected, configuration, identity
            )
            if inputs._identity(Path(project["path"])) != project["source_identity"]:
                raise ValueError("Project configuration changed before execution.")
            if lock is not None:
                published = project_lock.publish_lock(lock_path, lock)
            output.parent.mkdir(parents=True, exist_ok=True)
            code = _run(plan, snapshot, args.command)
            if code:
                message = f"Model {args.command} failed. See execution.log in {output}."
                report_path = output / "check.json"
                if report_path.is_file():
                    report, _ = inputs._configuration(report_path)
                    message = report.get("error", {}).get("message", message)
                result.update(
                    status="failed",
                    diagnostics=[
                        {
                            "code": "MODEL_CHECK_FAILED"
                            if args.command == "check"
                            else "BUILD_FAILED",
                            "message": message,
                        }
                    ],
                )
                return 1
            report, _ = inputs._configuration(
                cli._completion_file(output, "check.json")
            )
            _verify_report(
                plan, report, recipe, resolved, snapshot, identity, args.command
            )
            result.update(
                status="checked" if args.command == "check" else "complete",
                tensor_contract=report["tensor_contract"],
                case_count=report["case_count"],
                reports={
                    "check": str(output / "check.json"),
                    "execution": str(output / "execution.json"),
                },
            )
            if args.command == "build":
                result["artifacts"] = {
                    "package": str(output / "model"),
                    "graphs": str(output / "exported"),
                }
                result["reports"]["build"] = str(output / "build.json")
            exit_code = 0
            return 0
    finally:
        if output.is_dir() and not output.is_symlink():
            from .worker import _write_json

            retained = output / "project"
            retained.mkdir(exist_ok=False)
            _write_json(retained / "model-build.json", project["document"])
            _write_json(retained / "effective-config.json", effective)
            if published:
                (retained / "model-build.lock.json").write_text(
                    published["serialized_text"]
                )
            _write_json(
                output / "execution.json",
                {
                    "format_version": 1,
                    "operation": args.command,
                    "executor": plan["executor"],
                    "device": plan["device"],
                    "builder_image": plan.get("image"),
                    "toolchain_lock": plan["toolchain_lock"],
                    "exit_code": exit_code,
                    "input_identity": identity,
                },
            )
        if not args.json:
            print(json.dumps(result, indent=2))
