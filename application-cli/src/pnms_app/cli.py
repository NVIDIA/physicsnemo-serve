"""Application commands; runtime tooling is imported only after dispatch."""

import argparse
import json
from pathlib import Path
import signal
import sys
import subprocess
import uuid

from .common import repository_root, require
from .config import load_project, plan


def parser():
    root = argparse.ArgumentParser(
        description="PhysicsNeMo Serve application development and release"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("init", "check", "build", "run", "serve"):
        cmd = commands.add_parser(name)
        cmd.add_argument("project", nargs="?", type=Path, default=Path("."))
        cmd.add_argument("--repository", type=Path)
        cmd.add_argument(
            "--json", action="store_true", help="Structured output (also the default)"
        )
        if name == "init":
            cmd.add_argument("--runtime", choices=("python", "cpp"), default="python")
            cmd.add_argument("--backend", choices=("aoti", "tensorrt"))
            cmd.add_argument("--template", choices=("tensor", "custom"))
            cmd.add_argument("--model-project", type=Path)
            cmd.add_argument("--pipeline", default="simple")
        if name in ("build", "check"):
            cmd.add_argument("--environment", choices=("local", "container"))
            cmd.add_argument("--python", type=Path)
            cmd.add_argument("--runtime", dest="native_runtime", type=Path)
            cmd.add_argument("--sdk-prefix", type=Path)
            cmd.add_argument("--gpu")
        if name == "build":
            cmd.add_argument("--plan", action="store_true")
            cmd.add_argument("--image")
            cmd.add_argument("--push", action="store_true")
            cmd.add_argument("--from-build", type=Path)
            cmd.add_argument(
                "--refresh-lock",
                action="store_true",
                help="Accept the selected stack in pnms.lock after a successful build",
            )
            cmd.add_argument(
                "--container-process", action="store_true", help=argparse.SUPPRESS
            )
            cmd.add_argument(
                "--no-publish", action="store_true", help=argparse.SUPPRESS
            )
        if name == "check":
            cmd.add_argument("--config-only", action="store_true")
            cmd.add_argument("--through-phase")
        if name in ("run", "serve"):
            cmd.add_argument("--dev", action="store_true")
            cmd.add_argument("--output", type=Path)
            cmd.add_argument("--gpu")
            cmd.add_argument(
                "--host-process", action="store_true", help=argparse.SUPPRESS
            )
        if name == "run":
            cmd.add_argument("--input", required=True, type=Path)
            cmd.add_argument(
                "--input-file", action="append", default=[], metavar="ALIAS=PATH"
            )
            cmd.add_argument("--through-phase")
        if name == "serve":
            cmd.add_argument("--port", type=int)
            cmd.add_argument("--skip-build", action="store_true")
            cmd.add_argument(
                "--env",
                action="append",
                default=[],
                metavar="NAME",
                help="Forward an exported host variable by name to the image (repeatable)",
            )
    return root


def _model_options(project):
    build = project.get("build", {})
    return {
        key: build[key]
        for key in ("python", "runtime", "sdk_prefix", "gpu", "environment")
        if key in build
    }


def execute(args):
    env_names = ()
    if args.command == "serve":
        from .service import image_environment_names

        names = getattr(args, "env", ())
        require(
            not (args.dev and names),
            "--env is image-serving-only; --dev already inherits exported host variables",
        )
        env_names = image_environment_names(names)
    repository = repository_root(args.repository)
    if args.command == "init":
        if args.runtime == "cpp":
            from .scaffold import init_native

            return init_native(
                args.project,
                repository=repository,
                backend=args.backend or "aoti",
                template=args.template or "tensor",
                model_project=args.model_project,
            )
        require(
            not (args.backend or args.template or args.model_project),
            "native options require --runtime cpp",
        )
        from .python import init_project

        return init_project(
            args.project,
            args.project.name,
            repository=repository,
            pipeline_profile=args.pipeline,
        )
    project = load_project(
        args.project,
        repository,
        require_sources=(
            args.dev
            if hasattr(args, "dev")
            else args.command != "build" or not args.from_build
        ),
    )
    if getattr(args, "container_process", False):
        project["_container_image"] = project.get("build", {}).get("builder_image")
        settings = project.setdefault("build", {})
        for key in ("python", "runtime", "sdk_prefix", "venv"):
            settings.pop(key, None)
        settings["python"] = sys.executable
        sdk = settings.get("container_sdk_prefix") or "/opt/physicsnemo-inference"
        if Path(sdk).is_dir():
            settings["sdk_prefix"] = sdk
    if project["runtime"] == "python":
        require(
            getattr(args, "environment", None) != "container",
            "Python source builds use the local environment; --environment container is native-only. Use --image for Python Docker release packaging",
        )
        unsupported = [
            key
            for key in ("native_runtime", "sdk_prefix", "gpu")
            if getattr(args, key, None)
        ]
        require(
            not unsupported,
            "python project cannot use native options: " + ", ".join(unsupported),
        )
        require(
            not getattr(args, "input_file", None),
            "Python multipart input is described in --input JSON; --input-file is native-only",
        )
    for arg, key in (
        ("environment", "environment"),
        ("python", "python"),
        ("native_runtime", "runtime"),
        ("sdk_prefix", "sdk_prefix"),
        ("gpu", "gpu"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            project.setdefault("build", {})[key] = str(value)
    if args.command == "build":
        require(not args.push or args.image, "--push requires --image")
        if args.plan:
            return {
                **plan(project),
                "image": args.image,
                "publication": "requested" if args.push else "not_requested",
            }
        from .build import build_application, package_build

        if args.from_build:
            require(args.image, "--from-build requires --image")
            require(
                not args.refresh_lock,
                "--refresh-lock requires a new build; it cannot be combined with --from-build",
            )
            return package_build(
                args.from_build,
                project=project,
                repository=repository,
                image=args.image,
                push=args.push,
            )
        return build_application(
            project,
            repository=repository,
            image=args.image,
            push=args.push,
            publish_result=not args.no_publish,
            refresh_lock=args.refresh_lock,
        )
    if args.command == "check":
        if project["runtime"] == "python":
            from .python import check_project

            return check_project(
                project,
                repository=repository,
                config_only=args.config_only,
                through_phase=args.through_phase,
            )
        if project.get("workflow", {}).get("template") == "transolver_surface":
            from .cfd import check_project

            return check_project(
                project,
                repository=repository,
                config_only=args.config_only,
                **_model_options(project),
            )
        from .model import check_models

        return check_models(
            project,
            project["root"] / ".pnms/checks" / uuid.uuid4().hex,
            repository=repository,
            config_only=args.config_only,
            **_model_options(project),
        )
    from .build import last_build

    built = None
    if not args.dev or project["runtime"] == "cpp":
        built = last_build(project, image=args.command == "serve" and not args.dev)
    output = (
        args.output or project["root"] / ".pnms/runs" / uuid.uuid4().hex
    ).resolve()
    if built:
        artifact = (
            built["python"]["plugin_root"]
            if project["runtime"] == "python"
            else built["bundle"]
        )
        require(
            not output.is_relative_to((Path(built["root"]) / artifact).resolve()),
            f"{args.command} output directory must be outside the selected build artifact: {output}",
        )
    if args.command == "run":
        if project["runtime"] == "python":
            from .python import run_project

            effective = dict(project)
            if built:
                effective["plugin"] = (
                    Path(built["root"]) / built["python"]["plugin_root"]
                )
                effective["python_executable"] = built["environment"][
                    "python_executable"
                ]
            return run_project(
                effective,
                args.input,
                output,
                repository=repository,
                dev=args.dev,
                through_phase=args.through_phase,
            )
        require(
            not args.dev,
            "C++ source changes require pnms build; run uses the compiled bundle",
        )
        from .native import run_bundle

        inputs = {}
        for item in args.input_file:
            alias, sep, path = item.partition("=")
            require(
                sep and alias and alias not in inputs,
                "--input-file requires unique ALIAS=PATH",
            )
            inputs[alias] = str(Path(path).resolve())
        if built["environment"].get("kind") == "container" and not args.host_process:
            from .container import command, result_document

            effective = {
                **project,
                "build": {
                    **project.get("build", {}),
                    "builder_image": built["environment"]["builder_image"],
                },
            }
            arguments = [
                "run",
                str(project["root"]),
                "--input",
                str(args.input.resolve()),
                "--output",
                str(output),
                "--host-process",
                "--gpu",
                "0",
            ]
            for alias, path in inputs.items():
                arguments += ["--input-file", alias + "=" + path]
            output.parent.mkdir(parents=True, exist_ok=True)
            mounts = {args.input.resolve(): True}
            mounts.update({Path(path): True for path in inputs.values()})
            mounts[output.parent] = False
            result = subprocess.run(
                command(
                    effective,
                    repository=repository,
                    arguments=arguments,
                    source_sha=built["source_sha"],
                    extra_mounts=mounts,
                ),
                capture_output=True,
                text=True,
            )
            require(
                result.returncode == 0,
                f"container inference failed: {result.stderr or result.stdout}",
            )
            return result_document(result.stdout)
        from .admission import local_environment, verify_providers

        environment = local_environment(project, built)
        verify_providers(
            built,
            environment=environment,
            output=project["root"]
            / ".pnms/checks"
            / uuid.uuid4().hex
            / "providers.json",
        )
        return run_bundle(
            Path(built["root"]) / built["bundle"],
            args.input,
            output,
            inputs=inputs,
            environment=environment,
        )
    port = (
        args.port
        if args.port is not None
        else project.get("serve", {}).get("port", 8080)
    )
    require(0 < port < 65536, "port must be 1..65535")
    if args.dev and project["runtime"] == "python":
        from .python import serve_project

        return serve_project(
            project,
            repository=repository,
            dev=True,
            port=port,
            workspace=output,
            skip_build=args.skip_build,
        )
    from .service import serve

    return serve(
        project,
        built,
        repository=repository,
        output=output,
        port=port,
        dev=args.dev,
        skip_build=args.skip_build,
        env_names=env_names,
    )


def _terminate(_signal, _frame):
    raise KeyboardInterrupt()


def main(argv=None):
    args = parser().parse_args(argv)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        result = execute(args)
        status = result.pop("_exit_code", 0) if isinstance(result, dict) else 0
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
        return status
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"pnms: {error}", file=sys.stderr)
        return getattr(error, "exit_code", 2)
    finally:
        signal.signal(signal.SIGTERM, previous)
