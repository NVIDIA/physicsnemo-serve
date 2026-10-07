"""Framework-free orchestration for importing PhysicsNeMo model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from . import authoring_config, cli, inputs


def _resolve(args):
    checkpoint = inputs._regular_file(
        args.checkpoint.expanduser().absolute(), "checkpoint"
    )
    if checkpoint.suffix != ".mdlus":
        raise ValueError("import-checkpoint requires a PhysicsNeMo .mdlus checkpoint.")
    identity = inputs._identity(checkpoint)
    if args.checkpoint_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
            raise ValueError(
                "checkpoint SHA-256 must be 64 lowercase hexadecimal characters."
            )
        if identity["sha256"] != args.checkpoint_sha256:
            raise ValueError(
                "Checkpoint SHA-256 mismatch; the selected file differs from the expected checkpoint."
            )
    output = args.output.expanduser().absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError("Import output must not contain a symlink.")
    if output.exists():
        raise ValueError("Import output already exists; choose a fresh directory.")
    project = None
    if args.project is not None or (Path.cwd() / "model-build.json").exists():
        project = authoring_config.load(args.project or Path.cwd())
    settings = project["effective"] if project else {}
    executor = args.executor or settings.get("executor", "container")
    image = args.builder_image or settings.get("builder_image")
    lock = settings.get("toolchain_lock")
    if executor == "container" and not image and not lock:
        raise ValueError(
            "Set builder_image in model-build.json or pass --builder-image with "
            "an immutable image; alternatively use --executor local in a Python "
            "environment containing Torch, PhysicsNeMo and your model dependencies."
        )
    selection = argparse.Namespace(
        executor=executor,
        builder_image=image,
        lock=Path(lock) if lock else None,
        runtime=None,
    )
    plan = cli.resolve_executor(
        selection, cli.assets_root(), {"executor": executor}, require_runtime=False
    )
    return {
        **plan,
        "checkpoint": checkpoint,
        "identity": identity,
        "output": output,
        "project": project,
    }


def _execution_command(plan, snapshot, destination):
    worker = snapshot / "checkpoint_import_worker.py"
    checkpoint = snapshot / "checkpoint.mdlus"
    if plan["executor"] == "local":
        return [
            sys.executable,
            str(worker),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(destination),
            "--checkpoint-sha256",
            plan["identity"]["sha256"],
        ]
    # Ship the small importer with the frontend. An existing dependency image
    # can import checkpoints without rebuilding its installed Model Builder.
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={snapshot},dst=/inputs,readonly",
        "--mount",
        f"type=bind,src={destination.parent},dst=/outputs",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "LOCAL_CACHE=/tmp/physicsnemo-cache",
        "--env",
        "XDG_CACHE_HOME=/tmp/physicsnemo-cache",
        "--env",
        "USER=physicsnemo-builder",
        "--env",
        "LOGNAME=physicsnemo-builder",
        "--entrypoint",
        "/opt/nvidia/nvidia_entrypoint.sh",
        plan["image"],
        "python3",
        "/inputs/checkpoint_import_worker.py",
        "--checkpoint",
        "/inputs/checkpoint.mdlus",
        "--output",
        f"/outputs/{destination.name}",
        "--checkpoint-sha256",
        plan["identity"]["sha256"],
    ]


def _verify(plan, directory):
    report, _ = inputs._configuration(cli._completion_file(directory, "import.json"))
    if (
        type(report.get("format_version")) is not int
        or report["format_version"] != 1
        or report.get("status") != "imported"
        or report.get("checkpoint") != plan["identity"]
        or report.get("verification", {}).get("strict_reload") is not True
        or report.get("verification", {}).get("tensor_equality") is not True
        or report.get("verification", {}).get("source_tensors_unchanged") is not True
    ):
        raise RuntimeError(
            "Checkpoint import receipt does not verify the selected checkpoint."
        )
    for key, filename in (("checkpoint", "checkpoint.pt"), ("config", "config.json")):
        record = report.get("artifacts", {}).get(key)
        if not isinstance(record, dict) or record.get("path") != filename:
            raise RuntimeError(f"Checkpoint import receipt is missing {filename}.")
        cli._verify_completion_file(directory, record)
    inputs._configuration(directory / "config.json")
    if inputs._identity(plan["checkpoint"]) != plan["identity"]:
        raise RuntimeError("Source checkpoint changed during import.")
    return report


def _execute(plan):
    output = plan["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-import-", dir=output.parent
    ) as temporary:
        scratch = Path(temporary)
        snapshot = scratch / "inputs"
        snapshot.mkdir()
        shutil.copyfile(plan["checkpoint"], snapshot / "checkpoint.mdlus")
        if inputs._identity(snapshot / "checkpoint.mdlus") != plan["identity"]:
            raise RuntimeError("Source checkpoint changed while staging the import.")
        worker = Path(__file__).with_name("checkpoint_import_worker.py")
        shutil.copyfile(worker, snapshot / worker.name)
        worker_identity = inputs._identity(snapshot / worker.name)
        outputs = scratch / "outputs"
        outputs.mkdir()
        destination = outputs / "converted"
        command = _execution_command(plan, snapshot, destination)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as log:
            completed = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            log.seek(0)
            shutil.copyfileobj(log, sys.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Checkpoint importer exited with code {completed.returncode}; see diagnostics above."
                )
            report = _verify(plan, destination)
            # Reserve a fresh directory and publish only verified files. No
            # project configuration, checkpoint or existing output is replaced.
            output.mkdir()
            for filename in ("checkpoint.pt", "config.json"):
                with (
                    (destination / filename).open("rb") as source,
                    (output / filename).open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
            log.seek(0)
            with (output / "execution.log").open("x") as target:
                shutil.copyfileobj(log, target)
            execution = {
                "format_version": 1,
                "executor": plan["executor"],
                "builder_image": plan.get("image"),
                "worker": worker_identity,
                "source_checkpoint": str(plan["checkpoint"]),
                "exit_code": 0,
            }
            with (output / "execution.json").open("x") as target:
                json.dump(execution, target, indent=2)
            with (output / "import.json").open("x") as target:
                json.dump(report, target, indent=2, allow_nan=False)
    return report


def command(args, result):
    try:
        plan = _resolve(args)
    except (OSError, ValueError) as exc:
        raise cli.UsageError(str(exc), code="CHECKPOINT_IMPORT_INVALID") from exc
    result.update(stage="checkpoint-import", executor=plan["executor"])
    try:
        report = _execute(plan)
    except (OSError, ValueError, RuntimeError) as exc:
        result.update(
            status="failed",
            diagnostics=[{"code": "CHECKPOINT_IMPORT_FAILED", "message": str(exc)}],
        )
        if not args.json:
            print(json.dumps(result, indent=2))
        return 1
    root = Path(plan["project"]["path"]).parent if plan["project"] else Path.cwd()

    def selected(filename):
        path = plan["output"] / filename
        return (
            path.relative_to(root).as_posix()
            if path.is_relative_to(root)
            else str(path)
        )

    result.update(
        status="imported",
        output=str(plan["output"]),
        model=report["model"],
        verification=report["verification"],
        project_settings={
            "checkpoint": selected("checkpoint.pt"),
            "config": selected("config.json"),
        },
        reports={"import": str(plan["output"] / "import.json")},
        next_steps=[
            "Set checkpoint and config in model-build.json to project_settings.",
            "Connect the model constructor and representative inputs in build_adapter.py, then run check.",
        ],
    )
    if not args.json:
        print(json.dumps(result, indent=2))
    return 0
