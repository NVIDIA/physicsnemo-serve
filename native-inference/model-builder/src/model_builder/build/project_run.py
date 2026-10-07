"""Bind a resolved customer project to the existing build input contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import inputs, project_lock


def resolve_checkpoint(args, project):
    """Hash project-selected checkpoints without importing or loading the model."""
    if args.checkpoint is not None:
        checkpoint = inputs._regular_file(args.checkpoint, "checkpoint")
        if args.checkpoint_sha256 is None:
            args.checkpoint_sha256 = inputs._identity(checkpoint)["sha256"]
            project["effective"]["checkpoint_sha256"] = args.checkpoint_sha256


def identity(plan):
    result = {
        "backends": plan["backends"],
        "device": plan["device"],
        "executor": plan["executor"],
        "required_gpu_arch": plan.get("required_gpu_arch"),
        "toolchain_lock": {"sha256": plan["toolchain_lock"]["sha256"]},
    }
    if plan["executor"] == "container":
        result["builder_image"] = plan["image"]
    elif "runtime" in plan:
        result["runtime"] = inputs._identity(
            inputs._regular_file(plan["runtime"], "runtime")
        )
    result.update(
        model={
            "name": plan["recipe"]["name"],
            "version": plan["recipe"]["version"],
        },
        recipe=inputs._identity(plan["recipe_path"]),
        adapter=inputs._identity(
            plan["recipe_path"].parent / plan["recipe"]["adapter"]
        ),
        model_inputs=inputs.input_identities(plan.get("model_inputs")),
    )
    return result


def prepare(plan, project, *, update=False):
    path = Path(project["path"])
    source = path.read_bytes()
    if (
        inputs._identity(path) != project["source_identity"]
        or hashlib.sha256(source).hexdigest() != project["source_identity"]["sha256"]
    ):
        raise ValueError("Project configuration changed during resolution.")
    key = f"build:{project['profile']}" if project["profile"] else "build"
    selected = identity(plan)
    effective = project["effective"]
    effective.update(
        backends=plan["backends"],
        executor=plan["executor"],
        device=plan["device"],
        toolchain_lock=plan["toolchain_lock"]["path"],
    )
    if plan["executor"] == "container":
        effective["builder_image"] = plan["image"]
        effective.pop("runtime", None)
    else:
        effective.pop("builder_image", None)
        if "runtime" in plan:
            effective["runtime"] = str(plan["runtime"])
    if "model" not in effective:
        effective["recipe"] = str(plan["recipe_path"])
    model_inputs = plan.get("model_inputs")
    if model_inputs:
        effective.update(
            config=model_inputs["config"]["path"],
            checkpoint=model_inputs["checkpoint"]["path"],
            checkpoint_sha256=model_inputs["checkpoint"]["sha256"],
            assets={
                name: item["path"] for name, item in model_inputs["assets"].items()
            },
        )
    expected = {
        field: selected[field] for field in ("recipe", "adapter", "model_inputs")
    }
    if "runtime" in selected:
        expected["runtime"] = selected["runtime"]
    plan["expected_identity"] = expected
    lock_path = path.parent / "model-build.lock.json"
    inspection = project_lock.inspect_lock(lock_path, key, selected, update=update)
    plan["project"] = dict(
        project,
        source_text=source.decode("utf-8"),
        identity=selected,
        lock_inspection=inspection,
    )


def publish(plan):
    project = plan.get("project")
    if project is None:
        return
    if inputs._identity(Path(project["path"])) != project["source_identity"]:
        raise ValueError("Project configuration changed before execution.")
    if identity(plan) != project["identity"]:
        raise ValueError("Project inputs changed before execution.")
    project["published_lock"] = project_lock.publish_lock(
        Path(project["path"]).parent / "model-build.lock.json",
        project["lock_inspection"],
    )


def retain(plan):
    """Store project provenance alongside successful or failed build evidence."""
    project = plan.get("project")
    output = plan["output"]
    if project is None or not output.is_dir() or output.is_symlink():
        return None
    directory = output / "project"
    directory.mkdir(exist_ok=False)
    published = project["published_lock"]
    lock_text = published["serialized_text"]
    documents = {
        "model-build.json": project["source_text"],
        "effective-config.json": json.dumps(
            project["effective"], indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        "model-build.lock.json": lock_text,
    }
    for name, value in documents.items():
        (directory / name).write_bytes(value.encode("utf-8"))
    return {
        "source": {
            "path": "project/model-build.json",
            **inputs._identity(directory / "model-build.json"),
        },
        "effective_config": {
            "path": "project/effective-config.json",
            **inputs._identity(directory / "effective-config.json"),
        },
        "profile": project["profile"],
        "lock": {
            "path": "project/model-build.lock.json",
            **inputs._identity(directory / "model-build.lock.json"),
        },
    }
