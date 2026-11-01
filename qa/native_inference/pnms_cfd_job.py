"""Finite Lepton QA for common pnms CFD imports and matched HTTP sessions.

The immutable prepared image mirrors the two previously compiled model payloads.
This job does not recompile models, invoke Docker, or submit other Lepton jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "application-cli/src"))
from pnms_app.common import (  # noqa: E402
    file_identity,
    read_json,
    require,
    write_json,
)

BEGIN = "PNMS_CFD_QA_SUMMARY_BEGIN"
END = "PNMS_CFD_QA_SUMMARY_END"
IMPORT_ORIGIN = "c85aed8f5e354cc210339cb73d72080cbd493992"
IMPORT_IMAGES = {
    "aoti": "nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:ca6b0dad6380316e09ce063de67f768b6284f2192c76bed86d647a11c6689195",
    "tensorrt": "nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:5c02cba080871d4db26a5909e055b86997cc4fdaf3dc90ea063937f1f665c4d1",
}
ORDER = ("python", "aoti", "tensorrt", "tensorrt", "aoti", "python")
CASES = [
    "preflight",
    "import.aoti",
    "import.tensorrt",
    "build.aoti",
    "build.tensorrt",
    "run.aoti",
    "run.tensorrt",
    "build.python",
    "prepare.server",
    *[f"session.{i}.{runtime}" for i, runtime in enumerate(ORDER)],
    "compare",
]


def relocate_import(deployment, payload, output, *, backend):
    """Verify the old inventory before rebinding its exact mirrored file paths."""
    from pnms_app import cfd

    deployment, payload = Path(deployment).resolve(), Path(payload).resolve()
    require(backend in IMPORT_IMAGES, "unknown CFD import backend")
    require(
        deployment.is_file() and not deployment.is_symlink(),
        "missing original CFD deployment",
    )
    original = read_json(deployment)
    require(original.get("backend") == backend, "mirrored CFD backend differs")
    require(
        original.get("model_build_sources")
        == {"full": IMPORT_ORIGIN, "tail": IMPORT_ORIGIN},
        "mirrored CFD compilation origin differs from the qualified payload",
    )
    config = copy.deepcopy(original)
    prefix = Path("/opt/physicsnemo-native")

    def relocate(raw):
        selected = Path(raw)
        require(
            selected.is_absolute() and selected.is_relative_to(prefix),
            "CFD import path is outside its original payload",
        )
        relative = selected.relative_to(prefix)
        require(
            relative.parts and ".." not in relative.parts,
            "CFD import path escapes its payload",
        )
        path = payload
        for part in relative.parts:
            path /= part
            require(not path.is_symlink(), "mirrored CFD payload contains a symlink")
        require(path.exists(), "missing mirrored CFD payload: " + str(relative))
        return str(path)

    config["packages"] = [relocate(raw) for raw in config["packages"]]
    for key in ("stats_path", "startup_fixture_path"):
        config[key] = relocate(config[key])
    for item in config["model_assets"]:
        item["path"] = relocate(item["path"])
        require(
            Path(item["path"]).is_file()
            and file_identity(item["path"])
            == {key: item[key] for key in ("sha256", "size_bytes")},
            "mirrored CFD bytes differ from the original inventory",
        )
    output = Path(output)
    write_json(output, config)
    selected = cfd.import_deployment(
        {
            "root": output.parent,
            "backend": backend,
            "workflow": {"deployment": str(output)},
        }
    )
    record = {
        "schema_version": 1,
        "backend": backend,
        "source_image": IMPORT_IMAGES[backend],
        "original_deployment": {"path": str(deployment), **file_identity(deployment)},
        "relocated_deployment": {"path": str(output), **file_identity(output)},
        "model_build_sources": selected["model_build_sources"],
        "model_assets": config["model_assets"],
        "recompiled": False,
    }
    write_json(output.with_name("import.json"), record)
    return record


def descendants(pid, *, proc=Path("/proc")):
    parents = {}
    for path in proc.glob("[0-9]*/stat"):
        try:
            parents[int(path.parent.name)] = int(
                path.read_text().rsplit(")", 1)[1].split()[1]
            )
        except (OSError, ValueError, IndexError):
            pass
    selected, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        for child, owner in parents.items():
            if owner == parent and child not in selected:
                selected.add(child)
                pending.append(child)
    return selected


def process_token(pid, *, proc=Path("/proc")):
    """Linux start time prevents signalling an unrelated reused descendant PID."""
    try:
        return (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def execute(argv, *, output, environment, cwd, timeout):
    """Keep real exits and retire all nested service processes on timeout."""
    require(math.isfinite(timeout) and timeout > 0, "CFD QA deadline expired")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".command.json"), argv)
    with output.open("wb") as stream:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            children = descendants(process.pid)
            tokens = {pid: process_token(pid) for pid in children}
            for pid in [*sorted(children, reverse=True), process.pid]:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for pid in children:
                if tokens[pid] is None or process_token(pid) != tokens[pid]:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            raise
    require(code == 0, f"CFD QA command exited {code}; diagnostics: {output}")
    return output


def completed_build(project, repository):
    from pnms_app import build
    from pnms_app.config import load_project

    return build.last_build(load_project(project, repository, require_sources=False))


def native_project(repository, root, backend):
    shutil.copytree(repository / "plugins/physicsnemo-cfd-transolver-native", root)
    return root / ("release.yaml" if backend == "aoti" else "release-tensorrt.yaml")


def session_project(
    root, *, runtime, receipt, artifact_root, python_plugin=None, python=None
):
    """Select the completed build and fresh service storage through public config."""
    document = {
        "schema_version": 1,
        "name": "physicsnemo-cfd-transolver-native"
        if runtime != "python"
        else "physicsnemo-cfd-surface-benchmark",
        "runtime": "cpp" if runtime != "python" else "python",
        "build": {"python": str(python)},
    }
    if runtime == "python":
        document["plugin"] = str(python_plugin)
    else:
        document.update(
            workflow={"template": "transolver_surface"},
            serve={"artifact_root": str(artifact_root)},
        )
        document["build"].update(backend=runtime, environment="local")
        write_json(
            root / ".pnms/last-build.json",
            {"build_receipt": str(receipt), "identity": file_identity(receipt)},
        )
    write_json(root / "pnms.yaml", document)
    return root


def service_inventory(repository, binaries, built):
    files = {str(path): file_identity(path) for path in binaries}
    for selected in (
        repository / "pnms",
        *sorted((repository / "application-cli/src/pnms_app").glob("*.py")),
        *sorted((repository / "scripts").glob("*.py")),
    ):
        files[str(selected)] = file_identity(selected)
    if built["runtime"] == "cpp":
        root = Path(built["root"]) / built["bundle"]
    else:
        root = Path(built["root"]) / built["python"]["plugin_root"]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.resolve())] = file_identity(path)
    return files


def session_argv(args, runtime, ordinal, project, built, service_files):
    directory = args.output / "sessions" / f"{ordinal}-{runtime}"
    workspace = args.output / "services" / f"{ordinal}-{runtime}"
    service = [
        str(args.python),
        str(args.repository / "pnms"),
        "serve",
        str(project),
        "--repository",
        str(args.repository),
        "--dev",
        "--skip-build",
        "--port",
        "8080",
        "--output",
        str(workspace),
    ]
    command = args.output / "commands" / f"service-{ordinal}.json"
    write_json(command, service)
    argv = [
        str(args.python),
        str(args.repository / "qa/scripts/benchmark_pnms_cfd.py"),
        "session",
        "--runtime",
        runtime,
        "--ordinal",
        str(ordinal),
        "--service-command-json",
        str(command),
        "--service-files-json",
        str(service_files),
        "--expected-image",
        args.image,
        "--expected-source-sha",
        args.source_sha,
        "--profile",
        str(args.profile),
        "--request",
        str(args.request),
        "--reference-fields",
        str(args.reference_fields),
        "--input-root",
        str(args.input_root),
        "--checkpoint",
        str(args.checkpoint),
        "--stats",
        str(args.stats),
        "--environment-python",
        str(args.python),
        "--output",
        str(directory),
        "--run-id",
        args.run_id,
        "--service-cwd",
        str(args.repository),
    ]
    if runtime != "python":
        argv += ["--native-bundle", str(Path(built["root"]) / built["bundle"])]
    else:
        argv += ["--service-artifact-root", str(workspace / "outputs")]
    return argv, directory / "session.json"


def python_readiness(python, repository, output, *, environment, timeout):
    """Check the Python plugin and serving prerequisites before compilation."""
    report_path = Path(output) / "python-readiness.json"
    execute(
        [
            str(python),
            "-c",
            """import json, sys
from pathlib import Path
from plugin_dev import check_env_plugin, ensure_run_local_prerequisites
report = check_env_plugin(Path(sys.argv[1]))
check = {"name": "serving_prerequisites", "required": True, "ok": True,
         "detail": "Python queue binding, Redis and build tools are ready"}
try:
    ensure_run_local_prerequisites(skip_build=False)
except ValueError as exc:
    check.update(ok=False, detail=str(exc))
    report["status"] = "not_ready"
report.setdefault("checks", []).append(check)
Path(sys.argv[2]).write_text(json.dumps(report, sort_keys=True))
""",
            str(Path(repository) / "plugins/physicsnemo-cfd-surface-benchmark"),
            str(report_path),
        ],
        output=report_path.with_suffix(".log"),
        environment=environment,
        cwd=repository,
        timeout=min(180, timeout),
    )
    report = read_json(report_path)
    failed = [
        check
        for check in report.get("checks", [])
        if check.get("required") and not check.get("ok")
    ]
    require(
        report.get("workflow_id") == "physicsnemo-cfd-surface-benchmark"
        and report.get("status") == "ready"
        and not failed,
        "CFD Python workflow is not ready: "
        + "; ".join(f"{check.get('name')}: {check.get('detail')}" for check in failed)
        + f"; readiness report: {report_path}",
    )
    return {"path": str(report_path), **file_identity(report_path)}


def run(args):
    args.repository, args.output = args.repository.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PNMS_IMAGE_CONFIG_DIGEST", None)
    environment.pop("PNMS_SERVICE_BIN_DIR", None)
    environment.pop("NATIVE_DEPLOYMENT_CONFIG", None)
    environment.update(
        PNMS_SOURCE_SHA=args.source_sha,
        SOURCE_SHA=args.source_sha,
        RELEASE_IMAGE_DIGEST=args.image,
        PHYSICSNEMO_CFD_PYTHON_EXECUTABLE=str(args.python),
    )
    deadline = time.monotonic() + args.timeout
    summary = {
        "schema_version": 1,
        "qualification": "pnms-cfd-common-cli-http-session6-v1",
        "status": "running",
        "image_digest": args.image,
        "source_sha": args.source_sha,
        "run_id": args.run_id,
        "recompiled": False,
        "expected_cases": CASES if not args.prepare_only else CASES[:3],
        "cases": [],
        "driver": file_identity(Path(__file__)),
        "imports": {},
        "builds": {},
        "sessions": [],
    }

    def gate(name, action):
        started = time.monotonic()
        try:
            result = action()
        except BaseException as error:
            summary["cases"].append(
                {
                    "name": name,
                    "status": "failed",
                    "seconds": time.monotonic() - started,
                    "error": str(error),
                }
            )
            raise
        summary["cases"].append(
            {"name": name, "status": "passed", "seconds": time.monotonic() - started}
        )
        write_json(args.output / "summary.json", summary)
        return result

    def command(name, argv, overrides=None):
        return execute(
            argv,
            output=args.output / "logs" / (name + ".log"),
            environment={**environment, **(overrides or {})},
            cwd=args.repository,
            timeout=deadline - time.monotonic(),
        )

    try:

        def preflight():
            require(
                re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", args.image),
                "CFD QA requires an immutable image",
            )
            require(
                re.fullmatch(r"[a-f0-9]{40}", args.source_sha)
                and os.environ.get("PNMS_SOURCE_SHA", os.environ.get("SOURCE_SHA"))
                == args.source_sha,
                "CFD QA source differs from its prepared image",
            )
            require(
                math.isfinite(args.timeout) and args.timeout > 0,
                "finite CFD QA timeout required",
            )
            require(
                args.python.is_file() and os.access(args.python, os.X_OK),
                "selected CFD Python is unavailable",
            )
            require(
                file_identity(Path(__file__))["sha256"] == args.expected_runner_sha256,
                "CFD QA driver differs from the selected version",
            )
            require(
                file_identity(args.source_inventory)["sha256"]
                == args.expected_source_inventory_sha256,
                "CFD prepared source inventory differs from the selected version",
            )
            from qa.native_inference.pnms_stage_job import stage

            original = args.repository
            captured_inventory = args.output / "verified-source-inventory.json"
            shutil.copyfile(args.source_inventory, captured_inventory)
            require(
                file_identity(captured_inventory)["sha256"]
                == args.expected_source_inventory_sha256,
                "CFD source inventory changed during capture",
            )
            staged = stage(
                original,
                args.source_metadata,
                captured_inventory,
                args.output / "source",
                args.source_sha,
            )
            args.repository = Path(staged["source_root"])
            for key in ("profile", "request"):
                selected = getattr(args, key)
                if selected.is_relative_to(original):
                    setattr(args, key, args.repository / selected.relative_to(original))
            summary["source_inventory"] = {
                "path": str(captured_inventory),
                **file_identity(captured_inventory),
            }
            environment["PYTHONPATH"] = os.pathsep.join(
                str(args.repository / relative)
                for relative in ("application-cli/src", "scripts", "python", ".")
            )
            summary["python_readiness"] = python_readiness(
                args.python,
                args.repository,
                args.output / "logs",
                environment=environment,
                timeout=deadline - time.monotonic(),
            )
            from qa.native_inference.cfd_contract import ASSET_PINS, verify_file

            for name, path in (
                ("checkpoint", args.checkpoint),
                ("statistics", args.stats),
            ):
                verify_file(path, ASSET_PINS[name])
            from qa.scripts.run_native_cfd_reference import _verify_inputs

            _verify_inputs(read_json(args.request), args.input_root)
            from qa.native_inference.cfd_benchmark_reference import qualified_reference

            qualified_reference(args.reference_fields)
            write_json(
                args.output / "inputs.json",
                {
                    "reference_fields": {
                        "path": str(args.reference_fields),
                        **file_identity(args.reference_fields),
                    },
                    "checkpoint": {
                        "path": str(args.checkpoint),
                        **file_identity(args.checkpoint),
                    },
                    "stats": {"path": str(args.stats), **file_identity(args.stats)},
                    "profile": file_identity(args.profile),
                    "request": file_identity(args.request),
                },
            )

        gate("preflight", preflight)
        deployments, projects = {}, {}
        for backend in IMPORT_IMAGES:
            root = args.import_root / backend
            output = args.output / "imports" / backend / "deployment.json"
            summary["imports"][backend] = gate(
                "import." + backend,
                lambda backend=backend, root=root, output=output: relocate_import(
                    root / "deployment.json", root, output, backend=backend
                ),
            )
            deployments[backend] = output
        if args.prepare_only:
            summary["status"] = "prepared"
            return summary
        builds = {}
        for backend in IMPORT_IMAGES:
            project = native_project(
                args.repository, args.output / "applications" / backend, backend
            )
            projects[backend] = project
            gate(
                "build." + backend,
                lambda backend=backend, project=project: command(
                    "build-" + backend,
                    [
                        str(args.python),
                        str(args.repository / "pnms"),
                        "build",
                        str(project),
                        "--repository",
                        str(args.repository),
                        "--python",
                        str(args.python),
                        "--sdk-prefix",
                        str(args.sdk_prefix),
                        "--gpu",
                        "0",
                        "--json",
                    ],
                    {"NATIVE_DEPLOYMENT_CONFIG": str(deployments[backend])},
                ),
            )
            built = builds[backend] = completed_build(project, args.repository)
            require(
                built.get("recompiled") is False
                and built.get("model_build_sources")
                == {"full": IMPORT_ORIGIN, "tail": IMPORT_ORIGIN},
                "common CFD build altered imported model provenance",
            )
            summary["builds"][backend] = {
                "root": built["root"],
                "bundle_digest": built["bundle_digest"],
                "source_sha": built["source_sha"],
                "model_build_sources": built["model_build_sources"],
                "recompiled": False,
            }
        native_request = args.output / "native-request.json"
        write_json(
            native_request, {"case_id": "run_1", "seed": 42, "capture_inputs": False}
        )
        for backend in IMPORT_IMAGES:
            gate(
                "run." + backend,
                lambda backend=backend: command(
                    "run-" + backend,
                    [
                        str(args.python),
                        str(args.repository / "pnms"),
                        "run",
                        str(projects[backend].parent),
                        "--repository",
                        str(args.repository),
                        "--input",
                        str(native_request),
                        "--input-file",
                        "mesh=" + str(args.input_root / "run_1/boundary_1.vtp"),
                        "--input-file",
                        "geometry=" + str(args.input_root / "run_1/drivaer_1.stl"),
                        "--output",
                        str(args.output / "direct" / backend),
                        "--gpu",
                        "0",
                    ],
                ),
            )
        python_project = args.output / "applications/python"
        shutil.copytree(
            args.repository / "plugins/physicsnemo-cfd-surface-benchmark",
            python_project,
        )
        write_json(
            python_project / "pnms.yaml",
            {
                "schema_version": 1,
                "name": "physicsnemo-cfd-surface-benchmark",
                "runtime": "python",
                "plugin": ".",
                "build": {"python": str(args.python)},
            },
        )
        gate(
            "build.python",
            lambda: command(
                "build-python",
                [
                    str(args.python),
                    str(args.repository / "pnms"),
                    "build",
                    str(python_project),
                    "--repository",
                    str(args.repository),
                    "--json",
                ],
            ),
        )
        builds["python"] = completed_build(python_project, args.repository)
        native = builds["aoti"]
        location = Path(native["root"]) / native["bundle"]
        abi = native["environment"]["stack"]["cxx11_abi"]
        server_env = {
            **native.get("loader_environment", {}),
            "CARGO_TARGET_DIR": str(args.repository / "target"),
            "PYO3_PYTHON": str(args.python),
            "NATIVE_WORKFLOW_INCLUDE": str(location / "workflow/include"),
            "NATIVE_WORKFLOW_LIB_DIR": str(location / "workflow/lib"),
            "NATIVE_RUNNER_INCLUDE": str(location / "workflow/include"),
            "NATIVE_RUNNER_LIB_DIR": str(location / "workflow/lib"),
            "CXXFLAGS": environment.get("CXXFLAGS", "")
            + f" -D_GLIBCXX_USE_CXX11_ABI={int(abi)}",
        }
        environment.pop("NATIVE_WORKFLOW_ONLY", None)
        gate(
            "prepare.server",
            lambda: command(
                "prepare-server",
                [
                    "cargo",
                    "build",
                    "--locked",
                    "--manifest-path",
                    str(args.repository / "Cargo.toml"),
                    "-p",
                    "inference_server",
                    "-p",
                    "worker-runtime",
                    "-p",
                    "native-worker",
                    "--features",
                    "native-worker/cpp",
                ],
                server_env,
            ),
        )
        binaries = [
            args.repository / "target/debug" / name
            for name in ("inference_server", "worker-runtime", "native-worker")
        ]
        require(
            all(path.is_file() and os.access(path, os.X_OK) for path in binaries),
            "service preparation omitted current binaries",
        )
        environment["PNMS_SERVICE_BIN_DIR"] = str(binaries[0].parent)
        inventories = {}
        for runtime, built in builds.items():
            path = args.output / "service-files" / (runtime + ".json")
            write_json(path, service_inventory(args.repository, binaries, built))
            inventories[runtime] = path
        for ordinal, runtime in enumerate(ORDER):
            directory = args.output / "sessions" / f"{ordinal}-{runtime}"
            built = builds[runtime]
            project = session_project(
                args.output / "session-projects" / str(ordinal),
                runtime=runtime,
                receipt=Path(built["root"]) / "build.json",
                artifact_root=directory / "service-artifacts",
                python_plugin=Path(built["root"])
                / built.get("python", {}).get("plugin_root", "."),
                python=args.python,
            )
            argv, receipt = session_argv(
                args, runtime, ordinal, project, built, inventories[runtime]
            )
            gate(
                f"session.{ordinal}.{runtime}",
                lambda ordinal=ordinal, argv=argv: command(f"session-{ordinal}", argv),
            )
            record = read_json(receipt)
            require(
                record.get("status") == "passed"
                and record.get("runtime") == runtime
                and record.get("ordinal") == ordinal,
                "HTTP CFD session did not pass its selected runtime/order",
            )
            summary["sessions"].append(str(receipt))
        comparison = args.output / "comparison.json"
        gate(
            "compare",
            lambda: command(
                "compare",
                [
                    str(args.python),
                    str(args.repository / "qa/scripts/benchmark_pnms_cfd.py"),
                    "compare",
                    "--sessions",
                    *summary["sessions"],
                    "--profile",
                    str(args.profile),
                    "--reference-fields",
                    str(args.reference_fields),
                    "--output",
                    str(comparison),
                ],
            ),
        )
        require(
            read_json(comparison).get("status") == "passed",
            "matched session comparison failed",
        )
        require(
            [item["name"] for item in summary["cases"]] == CASES
            and all(item["status"] == "passed" for item in summary["cases"]),
            "CFD QA gate coverage differs",
        )
        summary.update(
            status="passed",
            comparison={"path": str(comparison), **file_identity(comparison)},
        )
        return summary
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "summary.json", summary)
        print(
            BEGIN + "\n" + json.dumps(summary, sort_keys=True) + "\n" + END, flush=True
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=REPOSITORY)
    result.add_argument(
        "--import-root", type=Path, default=Path("/opt/pnms-qa/cfd-imports")
    )
    result.add_argument(
        "--sdk-prefix", type=Path, default=Path("/opt/physicsnemo-inference")
    )
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument(
        "--source-inventory",
        type=Path,
        default=Path("/opt/physicsnemo-qa/source-inventory.json"),
    )
    for name in (
        "image",
        "source-sha",
        "run-id",
        "expected-source-inventory-sha256",
        "expected-runner-sha256",
    ):
        result.add_argument("--" + name, required=True)
    for name in ("output", "reference-fields", "input-root"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/outputs/assets/transolver/15e1c59a166037e2/Transolver.0.501.mdlus"
        ),
    )
    result.add_argument(
        "--stats",
        type=Path,
        default=Path("/outputs/assets/transolver/15e1c59a166037e2/global_stats.json"),
    )
    result.add_argument(
        "--profile",
        type=Path,
        default=REPOSITORY / "qa/inference/cfd_parity_surface_run1_transolver.json",
    )
    result.add_argument(
        "--request",
        type=Path,
        default=REPOSITORY
        / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json",
    )
    result.add_argument("--timeout", type=float, default=21600)
    result.add_argument("--prepare-only", action="store_true")
    return result


def main(argv=None):
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
