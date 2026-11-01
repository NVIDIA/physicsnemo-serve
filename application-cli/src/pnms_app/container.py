"""Optional Docker execution of the complete application build/run."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import read_json, require, write_json


def result_document(stdout):
    """Read one final pnms document after a forwarding entrypoint's banner."""
    decoder = json.JSONDecoder()
    candidates, offset = [], 0
    for line in stdout.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        start = offset + len(line) - len(stripped)
        if stripped.startswith("{"):
            try:
                value, end = decoder.raw_decode(stdout, start)
            except ValueError:
                pass
            else:
                candidates.append((start, end, value))
        offset += len(line)
    # Pretty-printed nested objects are parts of their enclosing document,
    # rather than additional result documents.
    documents = [
        item
        for item in candidates
        if not any(other[0] < item[0] and item[1] <= other[1] for other in candidates)
    ]
    valid = len(documents) == 1 and not stdout[documents[0][1] :].strip()
    if not valid and stdout:
        print(stdout, file=sys.stderr, end="" if stdout.endswith("\n") else "\n")
    require(
        valid,
        "container did not return one unambiguous final pnms JSON result; "
        "build.builder_image must use an ENTRYPOINT that forwards the supplied command",
    )
    start, _, value = documents[0]
    if start:
        print(stdout[:start], file=sys.stderr, end="")
    return value


def _model_source_mounts(project, root, mounts):
    """Expose the path-bearing inputs captured by model._capture in the image."""

    def bind(value, base):
        if not isinstance(value, str):
            return
        path = (base / Path(value).expanduser()).resolve()
        if path.exists() and not path.is_relative_to(root):
            mounts.setdefault(path if path.is_dir() else path.parent, True)

    for selection in project.get("models", {}).values():
        path = Path(selection["project"]).resolve()
        bind(str(path), root)
        document = read_json(path)
        # Capture validates and copies every declared profile, so all profiles
        # need the same original path base inside the builder container.
        for section in [document, *document.get("profiles", {}).values()]:
            for key in ("adapter", "checkpoint", "config", "runtime", "toolchain_lock"):
                bind(section.get(key), path.parent)
            for value in section.get("source", []):
                bind(value, path.parent)
            for value in section.get("assets", {}).values():
                bind(value, path.parent)
        if document.get("format_version") == 1 and document.get("recipe"):
            recipe_path = (path.parent / document["recipe"]).resolve()
            bind(str(recipe_path), path.parent)
            recipe = read_json(recipe_path)
            for item in [
                recipe.get("config"),
                recipe.get("checkpoint"),
                *recipe.get("assets", {}).values(),
            ]:
                if isinstance(item, dict):
                    bind(item.get("path"), recipe_path.parent)


def command(
    project,
    *,
    repository,
    arguments,
    environment=None,
    source_sha=None,
    extra_mounts=None,
):
    environment = dict(os.environ) if environment is None else environment
    settings = project.get("build", {})
    image = settings.get("builder_image")
    require(
        isinstance(image, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image),
        "container execution needs a pinned immutable build.builder_image",
    )
    allocation = environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    allocation = [value.strip() for value in allocation if value.strip()]
    gpu = settings.get("gpu")
    if gpu is not None:
        if str(gpu).isdigit() and allocation:
            require(int(gpu) < len(allocation), "--gpu is outside allocated devices")
            gpu = allocation[int(gpu)]
        elif allocation:
            require(gpu in allocation, "--gpu is outside allocated devices")
    elif len(allocation) == 1:
        gpu = allocation[0]
    else:
        require(
            False,
            "select one allocated GPU with CUDA_VISIBLE_DEVICES or --gpu before container execution",
        )
    require(gpu not in ("", "-1", "all"), "select one enabled GPU")
    root, repository = Path(project["root"]).resolve(), Path(repository).resolve()
    mounts = {repository: True, root: False}
    # Inference consumes the selected compiled receipt and must not depend on
    # original authoring projects, checkpoints or profile files still existing.
    if arguments and arguments[0] in ("build", "check"):
        source_dir = project.get("workflow", {}).get("source_dir")
        if source_dir:
            source = (root / source_dir).resolve()
            if not source.is_relative_to(root) and not source.is_relative_to(
                repository
            ):
                mounts[source] = True
        _model_source_mounts(project, root, mounts)
    for path, readonly in (extra_mounts or {}).items():
        path = Path(path).resolve()
        # Writable application/output bindings must not be downgraded by a
        # duplicate read-only input binding.
        mounts[path] = mounts.get(path, True) and readonly
    args = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=" + str(gpu),
        "--env",
        "CUDA_VISIBLE_DEVICES=0",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if source_sha:
        args += ["--env", "PNMS_SOURCE_SHA=" + source_sha]
    for path, readonly in mounts.items():
        require("," not in str(path), "Docker bind paths must not contain commas")
        args += [
            "--mount",
            f"type=bind,src={path},dst={path}" + (",readonly" if readonly else ""),
        ]
    return args + [
        "--workdir",
        str(root),
        image,
        "python3",
        str(repository / "pnms"),
        *map(str, arguments),
    ]


def build(
    project,
    *,
    repository,
    image=None,
    push=False,
    publish_result=True,
    refresh_lock=False,
):
    from . import environment_lock

    previous_lock = environment_lock.capture(project)
    sha = (
        os.environ.get("PNMS_SOURCE_SHA")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    )
    args = [
        "build",
        str(project["root"]),
        "--environment",
        "local",
        "--gpu",
        "0",
        "--container-process",
        "--no-publish",
        "--repository",
        str(repository),
    ]
    if refresh_lock:
        args.append("--refresh-lock")
    result = subprocess.run(
        command(project, repository=repository, arguments=args, source_sha=sha),
        text=True,
        capture_output=True,
    )
    require(
        result.returncode == 0,
        f"container build failed: {result.stderr or result.stdout}",
    )
    built = result_document(result.stdout)
    require(
        built.get("status") == "complete"
        and isinstance(built.get("environment"), dict)
        and isinstance(built.get("root"), str),
        "container did not forward the pnms build command to a complete build result",
    )
    built["environment"].update(
        kind="container", builder_image=project["build"]["builder_image"]
    )
    root = Path(built["root"])
    if image:
        try:
            from .images import build_image

            built["image"] = build_image(
                project, built, root=root, repository=repository, image=image, push=push
            )
        except BaseException as error:
            built.update(status="failed", error=str(error))
            write_json(
                root / "build.json",
                {key: value for key, value in built.items() if key != "root"},
            )
            raise
    saved = {key: value for key, value in built.items() if key != "root"}
    write_json(root / "build.json", saved)
    from .build import publish_success

    if publish_result:
        publish_success(project, root, built, previous_lock=previous_lock)
    return built
