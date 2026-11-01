"""Immutable application build receipts and automatic stage handoff."""

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import subprocess
import uuid

from .common import (
    contained,
    file_identity,
    inventory,
    snapshot,
    read_json,
    require,
    write_json,
)


def last_build(project, *, image=False):
    state = Path(project["root"]) / ".pnms"
    pointer = state / (
        "last-image.json"
        if image and (state / "last-image.json").is_file()
        else "last-build.json"
    )
    require(pointer.is_file(), "No successful application build; run pnms build first")
    selected = read_json(pointer)
    path = (
        Path(selected["build_receipt"])
        if "build_receipt" in selected
        else contained(pointer.parent, selected["receipt"])
    )
    require(
        file_identity(path) == selected["identity"], "successful build receipt changed"
    )
    result = read_json(path)
    require(result["status"] == "complete", "build is incomplete; run pnms build")
    require(
        result["runtime"] == project["runtime"] and result["name"] == project["name"],
        "selected build belongs to a different application/runtime; run pnms build",
    )
    if "image" in selected:
        result["image"] = selected["image"]
    root = path.parent
    if result["runtime"] == "cpp":
        from .bundle import verify

        verify(root / result["bundle"], result["bundle_digest"])
    else:
        from .python import verify_plugin_snapshot

        verify_plugin_snapshot(
            root / result["python"]["plugin_root"], result["python"]["files"]
        )
    return {**result, "root": str(root)}


def build_application(
    project,
    *,
    repository,
    image=None,
    push=False,
    publish_result=True,
    refresh_lock=False,
):
    if (
        project["runtime"] == "cpp"
        and project.get("workflow", {}).get("template") == "transolver_surface"
    ):
        from .cfd import build_application as build_cfd

        return build_cfd(
            project,
            repository=repository,
            image=image,
            push=push,
            publish_result=publish_result,
            refresh_lock=refresh_lock,
        )
    if (
        project["runtime"] == "cpp"
        and project.get("build", {}).get("environment") == "container"
    ):
        from .container import build

        return build(
            project,
            repository=repository,
            image=image,
            push=push,
            publish_result=publish_result,
            refresh_lock=refresh_lock,
        )
    from . import bundle, environment_lock

    previous_lock = environment_lock.capture(project)
    build_id = uuid.uuid4().hex
    root = Path(project["root"]) / ".pnms/builds" / build_id
    root.mkdir(parents=True, exist_ok=False)
    sha = os.environ.get("PNMS_SOURCE_SHA")
    if not sha:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    require(
        len(sha) == 40 and all(ch in "0123456789abcdef" for ch in sha),
        "source identity requires a full Git revision (PNMS_SOURCE_SHA for an exported checkout)",
    )
    result = {
        "schema_version": 1,
        "build_id": build_id,
        "status": "building",
        "runtime": project["runtime"],
        "name": project["name"],
        "source_sha": sha,
        "application": copy.deepcopy(project.get("document", {})),
        "checks": {"model": "not_run", "application": "not_run", "http": "not_run"},
        "inputs": inventory(project["root"], source=True),
    }
    try:
        if project["runtime"] == "python":
            from .python import build_project

            selected_python = environment_lock.python_identity(project)
            result["environment_lock"] = environment_lock.check(
                project, {}, refresh=refresh_lock, python=selected_python
            )
            built = build_project(project, root / "python", repository=repository)
            plugin = Path(built["plugin_root"])
            result["python"] = {
                **built,
                "plugin_root": plugin.relative_to(root).as_posix(),
                "files": inventory(plugin),
            }
            result["environment"] = built.get("environment", {})
            result["checks"]["application"] = "configuration_passed"
        else:
            from .model import build_models
            from .native import build_workflow, run_bundle

            authored_root = Path(project["root"])
            source_dir = project.get("workflow", {}).get("source_dir")
            custom_source = Path(source_dir) if source_dir else None
            if custom_source is not None:
                custom_source = (authored_root / custom_source).resolve()
                # Keep project sources frozen and external sibling paths intact.
                if custom_source.is_relative_to(authored_root.resolve()):
                    custom_source = (
                        root
                        / "inputs"
                        / custom_source.relative_to(authored_root.resolve())
                    )
            snapshot(authored_root, root / "inputs")
            project = {**project, "root": root / "inputs", "_state_root": authored_root}
            settings = project.get("build", {})

            def check_environment(environment):
                result["environment_lock"] = environment_lock.check(
                    project, environment, refresh=refresh_lock
                )

            models = build_models(
                project,
                root,
                repository=repository,
                source_sha=sha,
                cache_dir=authored_root / ".pnms/cache/models",
                environment_check=check_environment,
                environment_preflight=lambda environment: environment_lock.preflight(
                    project, environment, refresh=refresh_lock
                ),
                **{
                    key: settings[key]
                    for key in ("python", "runtime", "sdk_prefix", "gpu", "environment")
                    if key in settings
                },
            )
            result["environment"] = models["environment"]
            if "environment_lock" not in result:
                check_environment(models["environment"])
            result["models"] = {
                alias: {
                    **info,
                    "output_root": Path(info["output_root"])
                    .relative_to(root)
                    .as_posix(),
                    "package": Path(info["package"]).relative_to(root).as_posix(),
                }
                for alias, info in models["models"].items()
            }
            result["checks"]["model"] = "passed"
            write_json(root / "build.json", result)
            env = models["environment"]
            workflow = build_workflow(
                project,
                root / "workflow-build",
                repository=repository,
                sdk_prefix=Path(env["sdk_prefix"]),
                python=Path(env["python"]),
                backend=project["backend"],
                custom_source=custom_source,
                tensorrt_root=settings.get("tensorrt_root"),
                environment=env.get("native_environment"),
                build_environment=env,
            )
            result["workflow"] = {
                key: value
                for key, value in workflow.items()
                if key not in ("loader_environment",)
            }
            result["loader_environment"] = workflow.get("loader_environment", {})
            seal = bundle.assemble(
                project,
                models["models"],
                workflow,
                root / "bundle",
                repository=repository,
                environment=env,
                source_sha=sha,
            )
            result.update(bundle="bundle", bundle_digest=seal["bundle_digest"])
            fixtures = read_json(root / "bundle/fixtures/cases.json")
            for index, case in enumerate(fixtures["cases"]):
                request = root / "checks" / f"case-{index}.json"
                write_json(request, case["request"])
                executed = run_bundle(
                    root / "bundle",
                    request,
                    root / "checks" / f"case-{index}",
                    inputs={
                        alias: contained(root / "bundle", relative)
                        for alias, relative in case.get("inputs", {}).items()
                    },
                    input_metadata=case.get("files", {}),
                    environment={
                        **workflow.get("loader_environment", {}),
                        "CUDA_VISIBLE_DEVICES": env["gpu"].get("binding", "0"),
                    },
                )
                bundle.compare(
                    executed["result"]["payload"],
                    case["expected"],
                    rtol=case.get("rtol", 1e-5),
                    atol=case.get("atol", 1e-6),
                    where=f"fixture {case['name']}",
                )
                bundle.compare_artifacts(executed, case)
            result["checks"]["application"] = "passed"
        if image:
            from .images import build_image

            result["image"] = build_image(
                project,
                result,
                root=root,
                repository=repository,
                image=image,
                push=push,
            )
        result["status"] = "complete"
        write_json(root / "build.json", result)
        if publish_result:
            publish_success(
                {**project, "root": project.get("_state_root", project["root"])},
                root,
                result,
                previous_lock=previous_lock,
            )
        return {**result, "root": str(root)}
    except BaseException as error:
        result.update(status="failed", error=str(error))
        write_json(root / "build.json", result)
        raise


@contextmanager
def publication_lock(project):
    """Serialize publication across processes; keep this lock inode in place."""
    state = Path(project.get("_state_root", project["root"])).resolve() / ".pnms"
    state.mkdir(parents=True, exist_ok=True)
    with (state / "publication.lock").open("a+b") as handle:
        if os.name == "nt":
            import errno
            import msvcrt
            import time

            # CRT byte locks may extend beyond EOF; every publisher locks byte 0.
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno != errno.EACCES:
                        raise
                    # LK_LOCK gives up after ten attempts; publication must wait.
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def publish_success(project, root, result, *, previous_lock):
    """Publish the lock and success pointers together, restoring them on failure."""
    from . import environment_lock

    with publication_lock(project):
        state = Path(project["root"]) / ".pnms"
        paths = [
            Path(project["root"]) / "pnms.lock",
            state / "last-build.json",
            state / "last-image.json",
        ]
        saved = {path: path.read_bytes() if path.exists() else None for path in paths}
        try:
            if "environment_lock" in result:
                environment_lock.commit(
                    project, result["environment_lock"], previous=previous_lock
                )
            publish(project, root)
            if "image" in result:
                publish_image(project, root / "build.json", result["image"])
        except BaseException:
            for path, contents in saved.items():
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    temporary = path.with_name(
                        path.name + "." + uuid.uuid4().hex + ".tmp"
                    )
                    temporary.write_bytes(contents)
                    os.replace(temporary, path)
            raise


def publish(project, root):
    state = Path(project["root"]) / ".pnms"
    write_json(
        state / "last-build.json",
        {
            "schema_version": 1,
            "receipt": (Path("builds") / root.name / "build.json").as_posix(),
            "identity": file_identity(root / "build.json"),
        },
    )


def publish_image(project, receipt, image):
    write_json(
        Path(project["root"]) / ".pnms/last-image.json",
        {
            "schema_version": 1,
            "build_receipt": str(Path(receipt).resolve()),
            "identity": file_identity(receipt),
            "image": image,
        },
    )


def package_build(path, *, project, repository, image, push=False):
    from .images import build_image

    path = Path(path).resolve()
    result = read_json(path)
    require(
        result["status"] == "complete"
        and result["name"] == project["name"]
        and result["runtime"] == project["runtime"],
        "--from-build must name this application's completed build.json",
    )
    if result["runtime"] == "cpp":
        from .bundle import verify

        verify(path.parent / result["bundle"], result["bundle_digest"])
    packaged = build_image(
        project, result, root=path.parent, repository=repository, image=image, push=push
    )
    # Packaging evidence is separate from the immutable original build receipt.
    write_json(
        path.parent / "images" / (uuid.uuid4().hex + ".json"),
        {"build_id": result["build_id"], "image": packaged},
    )
    with publication_lock(project):
        publish_image(project, path, packaged)
    return {
        "schema_version": 1,
        "status": "complete",
        "build_id": result["build_id"],
        "image": packaged,
        "root": str(path.parent),
    }
