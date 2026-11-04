"""Docker release packaging consumes built models and Python plugin snapshots."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import uuid

from .common import (
    digest,
    read_json,
    require,
    write_json,
)
from .compatibility import PROBE as STACK_PROBE

# This verifier deliberately uses only the image's Python standard library. It
# reads the sealed bytes without launching models, services, Redis or CUDA.
CONTENT_PROBE = r"""
# PNMS_IMAGE_CONTENT_CHECK_V1
import hashlib, importlib.util, json, os, pathlib, subprocess, sys
def require(value, message):
    if not value:
        raise RuntimeError(message)
root = pathlib.Path(sys.argv[1]).resolve()
expected_digest, expected_source = sys.argv[2:4]
seal = json.loads((root / "inventory.json").read_text())
canonical = json.dumps(seal["files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
require(seal["schema_version"] == 1, "unsupported inventory schema")
require(hashlib.sha256(canonical).hexdigest() == seal["bundle_digest"] == expected_digest, "bundle content identity differs")
files = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if relative == "inventory.json":
        continue
    if path.is_symlink():
        require(path.resolve().is_relative_to(root) and path.resolve().is_file(), "unsafe bundle symlink")
        files.append(dict(path=relative, symlink=os.readlink(path)))
    elif path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(dict(path=relative, sha256=digest.hexdigest(), size_bytes=path.stat().st_size))
    else:
        require(path.is_dir(), "unsupported bundle file type")
require(files == seal["files"], "image bundle content differs from inventory")
descriptor = json.loads((root / "bundle.json").read_text())
require(descriptor["source_sha"] == expected_source, "image source identity differs")
require(descriptor["runtime"] == "cpp", "image omitted native application contract")
runner = root / descriptor["workflow"]["runner"]
require(runner.resolve().is_relative_to(root) and runner.is_file(), "native runner is not in the bundle")
environment = os.environ.copy()
torch = importlib.util.find_spec("torch")
if torch is not None and torch.origin:
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(pathlib.Path(torch.origin).parent / "lib"), str(root / "workflow/lib"),
        environment.get("LD_LIBRARY_PATH", "")])
loaded = subprocess.run([str(runner), "--help"], env=environment, capture_output=True, text=True)
require(loaded.returncode == 0 and runner.name in loaded.stdout, "native loader check failed: " + loaded.stderr)
print(json.dumps(dict(status="passed", bundle_digest=seal["bundle_digest"],
                      source_sha=descriptor["source_sha"], native_load="passed", files=len(files))))
"""


PYTHON_BASE_PROBE = r"""
# PNMS_PYTHON_BASE_V1
import importlib.util, json, os, pathlib
programs = ["/app/bin/inference-server", "/app/bin/worker-runtime", "/usr/bin/dumb-init", "/app/scripts/entrypoint.sh"]
scripts = ["plugin_dev.py", "plugin_runtime.py", "runtime_env_launcher.py", "inference_worker.py", "plugin_hook_runner.py"]
missing = [path for path in programs if not pathlib.Path(path).is_file() or not os.access(path, os.X_OK)]
missing += ["/app/scripts/" + name for name in scripts if not pathlib.Path("/app/scripts", name).is_file()]
missing += ["python:" + name for name in ("yaml", "jsonschema", "redis", "scicomp_rq") if importlib.util.find_spec(name) is None]
config = pathlib.Path(os.environ.get("WORKER_RUNTIME_CONFIG", "/app/scripts/worker_runtime_config.json"))
if not config.is_file():
    missing.append("worker runtime configuration")
elif not isinstance(json.loads(config.read_text()).get("python_runtime_envs"), dict):
    missing.append("python_runtime_envs")
print(json.dumps(dict(status="failed" if missing else "passed", capability="python-server-v1", missing=missing)))
"""

PYTHON_CONTENT_PROBE = r"""
# PNMS_PYTHON_CONTENT_V1
import hashlib, json, pathlib, sys
sys.path.insert(0, "/app/scripts")
from plugin_runtime import load_plugin_manifest
def require(value, message):
    if not value:
        raise RuntimeError(message)
root = pathlib.Path("/opt/pnms/plugin")
files = []
for path in sorted(root.rglob("*")):
    require(not path.is_symlink(), "Python plugin image contains a symlink")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append(dict(path=path.relative_to(root).as_posix(), sha256=digest.hexdigest(), size_bytes=path.stat().st_size))
actual = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
require(actual == sys.argv[1], "Python plugin image inventory differs from build")
manifest = load_plugin_manifest(root / "plugin.yaml")
launcher = json.loads(pathlib.Path("/opt/pnms/python-launch.json").read_text())
config = json.loads(pathlib.Path("/opt/pnms/worker_runtime_config.json").read_text())
require(config == launcher["runtime_config"], "Python image launch configuration differs")
require(launcher["workflow_id"] == manifest["metadata"]["id"], "Python image workflow identity differs")
require(launcher["environment"]["PLUGIN_DIR"] == str(root), "Python image plugin is not isolated")
require(launcher["environment"]["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"] == manifest["metadata"]["id"], "Python enabled workflow differs")
for field in ("executor_class", "prepare_executor_class", "postprocess_executor_class", "readiness_executor_class"):
    executor = manifest["runtime"].get(field)
    if executor:
        require(executor in config["python_runtime_envs"], "declared Python executor has no runtime environment: " + executor)
executor = manifest["runtime"]["executor_class"]
require(config["python_runtime_envs"][executor].get("launch", {}).get("enabled"), "Python execution worker is not enabled")
print(json.dumps(dict(status="passed", plugin_digest=actual, workflow_id=manifest["metadata"]["id"], files=len(files))))
"""


def _python_probe(reference, probe, arguments=(), *, runner, log):
    return json.loads(
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "CUDA_VISIBLE_DEVICES=-1",
                "--entrypoint",
                "/bin/sh",
                reference,
                "-c",
                'pnms_probe=$1; shift; exec "${PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE:-python3}" -c "$pnms_probe" "$@"',
                "pnms-python-probe",
                probe,
                *arguments,
            ],
            runner=runner,
            log=log,
        )
    )


def _run(command, *, runner, log, timeout=None):
    command = [str(part) for part in command]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            **({"timeout": timeout} if timeout is not None else {}),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        with log.open("a") as stream:
            stream.write(json.dumps({"command": command, "error": str(error)}) + "\n")
        raise ValueError(
            f"Docker operation could not finish: {error}; diagnostics: {log}"
        ) from error
    with log.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "command": [str(part) for part in command],
                    "returncode": result.returncode,
                }
            )
            + "\n"
        )
        stream.write(result.stdout or "")
        stream.write(result.stderr or "")
    require(result.returncode == 0, f"Docker operation failed; diagnostics: {log}")
    return result.stdout


def _inspect(reference, *, runner, log):
    value = json.loads(
        _run(["docker", "image", "inspect", reference], runner=runner, log=log)
    )
    require(
        isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict),
        "Docker image configuration inspection is invalid",
    )
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value[0].get("Id", "")),
        "Docker image identity is invalid",
    )
    return value[0]


def _identity(inspected):
    """Keep Docker's runnable ID distinct from the OCI configuration digest."""
    image_id = inspected["Id"]
    descriptor = inspected.get("Descriptor")
    config_digest, kind = image_id, "image-config"
    if descriptor is not None:
        # The containerd store exposes the manifest digest as Id. BuildKit's
        # descriptor annotation identifies its configuration without exporting
        # model/image layers. The classic store's Id already is this digest.
        require(
            isinstance(descriptor, dict)
            and descriptor.get("digest") == image_id
            and descriptor.get("mediaType")
            in (
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            ),
            "loaded release must have a single-platform image manifest identity",
        )
        annotations = descriptor.get("annotations", {})
        config_digest = (
            annotations.get("config.digest") if isinstance(annotations, dict) else None
        )
        kind = "oci-manifest"
    require(
        isinstance(config_digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest),
        "Docker did not expose the built image configuration digest; "
        "rebuild with a Docker/BuildKit version that preserves config.digest metadata",
    )
    return {
        "image_config_digest": config_digest,
        "runtime_identity": {"kind": "image-config", "value": config_digest},
        "docker_identity": {"kind": kind, "value": image_id},
    }


def _base_startup(reference, *, runner, log):
    """Require base initialization to forward the native service command."""
    inspected = _inspect(reference, runner=runner, log=log)
    entrypoint = inspected.get("Config", {}).get("Entrypoint") or []
    require(
        isinstance(entrypoint, list)
        and all(isinstance(part, str) and part for part in entrypoint),
        "runtime base entrypoint is invalid",
    )
    name = "pnms-base-init-" + uuid.uuid4().hex
    marker = "PNMS_BASE_COMMAND_FORWARD_V1"
    try:
        output = _run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--platform",
                "linux/amd64",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "CUDA_VISIBLE_DEVICES=-1",
                reference,
                "python3",
                "-c",
                f"print({marker!r})",
            ],
            runner=runner,
            log=log,
            timeout=30,
        )
    except BaseException:
        # A base that ignores CMD may start its own long-lived server. Stop
        # only this named probe if the Docker client times out or is interrupted.
        try:
            runner(
                ["docker", "rm", "--force", name],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            pass
        raise
    require(
        output.splitlines().count(marker) == 1,
        "runtime base initialization does not forward the supplied command",
    )
    return {
        "entrypoint": entrypoint,
        "command": ["python3", "-m", "pnms_app.image_entrypoint"],
    }


def _stack(
    reference,
    expected,
    *,
    bundle=None,
    library="libphysicsnemo_workflow.so",
    runner,
    log,
):
    mounted = (
        [
            "--mount",
            f"type=bind,src={Path(bundle).resolve()},dst=/opt/physicsnemo-release,readonly",
        ]
        if bundle
        else []
    )
    observed = json.loads(
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "CUDA_VISIBLE_DEVICES=-1",
                "--entrypoint",
                "python3",
                *mounted,
                reference,
                "-c",
                STACK_PROBE,
                "/opt/physicsnemo-release/workflow/lib/" + library,
            ],
            runner=runner,
            log=log,
        )
    )
    for key in (
        "system",
        "machine",
        "libc",
        "torch",
        "cuda",
        "cxx11_abi",
        "tensorrt",
        "torch_libraries",
        "system_libraries",
        "driver_libraries",
        "workflow_binary",
    ):
        if expected.get(key) != observed.get(key):
            details = ""
            if isinstance(expected.get(key), dict) and isinstance(
                observed.get(key), dict
            ):
                differences = sorted(
                    name
                    for name in set(expected[key]) | set(observed[key])
                    if expected[key].get(name) != observed[key].get(name)
                )
                details = " (" + ", ".join(differences[:5]) + ")"
            require(
                False,
                f"runtime image stack is incompatible at {key}{details}; "
                f"select a matching image or explicitly rebuild in it; diagnostics: {log}",
            )
    return observed


def _repository(reference):
    value = reference.split("@", 1)[0]
    if ":" in value.rsplit("/", 1)[-1]:
        value = value.rsplit(":", 1)[0]
    parts = value.split("/")
    if len(parts) == 1:
        parts = ["docker.io", "library", parts[0]]
    elif "." not in parts[0] and ":" not in parts[0] and parts[0] != "localhost":
        parts = ["docker.io", *parts]
    if parts[0] == "index.docker.io":
        parts[0] = "docker.io"
    return "/".join(parts)


def source_context(repository, source_sha, output, *, native, metadata=None):
    """Freeze committed packaging inputs, or inputs bound to an exported receipt.

    Exported QA sources carry source.json and source-inventory.json together;
    PNMS_SOURCE_METADATA explicitly selects that source.json for CLI packaging.
    """
    repository, output = Path(repository).resolve(), Path(output)
    inputs = (
        (
            "Cargo.toml",
            "Cargo.lock",
            ".cargo",
            "crates",
            "application-cli",
            "scripts",
            "python",
        )
        if native
        else (
            "application-cli/docker/Dockerfile.python",
            "application-cli/src/pnms_app/python_release.py",
        )
    )
    guidance = (
        "For --from-build, use the exact clean source revision from the build receipt; "
        "for exported sources set PNMS_SOURCE_METADATA to the verified source.json "
        "beside source-inventory.json."
    )
    require(
        isinstance(source_sha, str) and re.fullmatch(r"[0-9a-f]{40}", source_sha),
        "image source revision is missing or invalid. " + guidance,
    )

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *arguments], stderr=subprocess.PIPE
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                "image source Git verification failed. " + guidance
            ) from error

    committed = (repository / ".git").exists()
    if committed:
        require(
            git("rev-parse", "HEAD").decode().strip() == source_sha,
            "image source revision differs from the build receipt. " + guidance,
        )
        # Archive the pinned commit, not the live directory later read by Docker.
        # Compare each copied file against that commit even when Git's index has
        # assume-unchanged/skip-worktree flags or staged changes.
        archive = tarfile.open(
            fileobj=io.BytesIO(
                git("archive", "--format=tar", source_sha, "--", *inputs)
            )
        )
        records = {}
        with archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                require(
                    member.isfile(), "image source contains a symlink or special file"
                )
                with archive.extractfile(member) as stream:
                    data = stream.read()
                records[member.name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "mode": 0o755 if member.mode & 0o111 else 0o644,
                    "symlink": None,
                }
        provenance = {"kind": "git", "source_sha": source_sha}
    else:
        require(metadata, "image source metadata is required without Git. " + guidance)
        metadata = Path(metadata)
        identity = read_json(metadata)
        inventory_path = metadata.with_name("source-inventory.json")
        require(
            inventory_path.is_file(), "image source inventory is missing. " + guidance
        )
        inventory_bytes = inventory_path.read_bytes()
        require(
            identity.get("source_sha") == source_sha
            and identity.get("source_dirty") is False
            and identity.get("source_inventory_sha256")
            == hashlib.sha256(inventory_bytes).hexdigest(),
            "image source metadata/inventory differs or records dirty source. "
            + guidance,
        )
        records = json.loads(inventory_bytes)
        require(
            isinstance(records, dict) and records, "image source inventory is empty"
        )
        records = {
            name: record
            for name, record in records.items()
            if any(name == item or name.startswith(item + "/") for item in inputs)
        }
        provenance = {
            "kind": "source-inventory",
            "source_sha": source_sha,
            "source_inventory_sha256": identity["source_inventory_sha256"],
        }
    required = (
        {"Cargo.toml", "Cargo.lock", "application-cli/docker/Dockerfile.native"}
        if native
        else set(inputs)
    )
    require(
        required <= records.keys(),
        "image source inventory omits required packaging inputs",
    )
    output.mkdir(parents=True, exist_ok=False)
    for relative, record in records.items():
        path = Path(relative)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == relative
            and "\\" not in relative,
            "unsafe image source inventory path",
        )
        origin = repository / path
        require(
            record.get("symlink") is None
            and not any(
                parent.is_symlink()
                for parent in (origin, *origin.parents)
                if parent != repository and parent.is_relative_to(repository)
            )
            and origin.is_file(),
            "image source changed or contains a symlink: " + relative + ". " + guidance,
        )
        data = origin.read_bytes()
        mode = origin.stat().st_mode & 0o777
        require(
            len(data) == record["size_bytes"]
            and hashlib.sha256(data).hexdigest() == record["sha256"]
            and (
                os.name == "nt"
                or (
                    bool(mode & 0o111) == bool(record["mode"] & 0o111)
                    if committed
                    else mode == record["mode"]
                )
            ),
            "image source changed from its verified revision: "
            + relative
            + ". "
            + guidance,
        )
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy the bytes just verified, rather than reopening the live source.
        target.write_bytes(data)
        target.chmod(record["mode"])
    if committed:
        require(
            git("rev-parse", "HEAD").decode().strip() == source_sha,
            "image source revision changed during capture. " + guidance,
        )
    return {**provenance, "context_digest": digest(records), "files": len(records)}


def build_image(project, built, *, root, repository, image, push=False, runner=None):
    runner = runner or subprocess.run
    root, repository = Path(root), Path(repository)
    native = built["runtime"] == "cpp"
    if not native:
        from .python import verify_plugin_snapshot

        verify_plugin_snapshot(
            root / built["python"]["plugin_root"], built["python"]["files"]
        )
    base = project.get("serve", {}).get("image") or project.get("build", {}).get(
        "builder_image"
    )
    require(
        isinstance(base, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", base),
        "Set serve.image to an immutable matched runtime image (registry/repository@sha256:...) before --image packaging",
    )
    require(
        isinstance(image, str)
        and image
        and "@" not in image
        and not image.startswith("-")
        and not any(ch.isspace() for ch in image),
        "invalid release image name",
    )
    attempt = root / "images" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    log = attempt / "docker.log"
    result = {
        "status": "building",
        "local_tag": image,
        "bundle_digest": built.get("bundle_digest"),
        "runtime_image": base,
        "checks": {
            "stack": "not_run" if native else "not_required",
            "contents": "not_run",
            "native_load": "not_run" if native else "not_required",
            "gpu_startup": "not_run",
            "http": "not_run",
        },
        "diagnostics": str(log),
    }
    try:
        context = attempt / "source"
        result["packaging_source"] = source_context(
            repository,
            built.get("source_sha"),
            context,
            native=native,
            metadata=os.environ.get("PNMS_SOURCE_METADATA"),
        )
        if native:
            from .bundle import verify

            verify(root / built["bundle"], built.get("bundle_digest"))
            expected = built["environment"]["stack"]
            compatibility = built.get("workflow", {}).get("compatibility")
            require(
                isinstance(compatibility, dict)
                and compatibility.get("schema_version") == 1
                and compatibility.get("torch_libraries")
                and compatibility.get("system_libraries")
                and compatibility.get("workflow_binary"),
                "build receipt lacks native provider hashes; rebuild the workflow before image packaging",
            )
            for key in (
                "system",
                "machine",
                "libc",
                "torch",
                "cuda",
                "cxx11_abi",
                "tensorrt",
            ):
                require(
                    expected.get(key) == compatibility.get(key),
                    f"native provider fingerprint differs from compiled model stack at {key}",
                )
            expected = compatibility
            library = built.get("workflow", {}).get(
                "library_name", "libphysicsnemo_workflow.so"
            )
            require(
                library in ("libphysicsnemo_workflow.so", "libtransolver_workflow.so"),
                "unsupported release workflow library",
            )
            _stack(
                base,
                expected,
                bundle=root / built["bundle"],
                library=library,
                runner=runner,
                log=log,
            )
            startup = _base_startup(base, runner=runner, log=log)
        else:
            python_base = _python_probe(base, PYTHON_BASE_PROBE, runner=runner, log=log)
            require(
                python_base.get("status") == "passed"
                and python_base.get("capability") == "python-server-v1",
                f"Python release base must be a full existing server image with its entrypoint and runtime environments; missing={python_base.get('missing', [])}; diagnostics: {log}",
            )
            result["checks"]["python_base"] = "passed"
        iid = attempt / "image-id.txt"
        contexts = (
            ["--build-context", "application-bundle=" + str(root / built["bundle"])]
            if native
            else [
                "--build-context",
                "application-plugin=" + str(root / built["python"]["plugin_root"]),
            ]
        )
        dockerfile = (
            context
            / "application-cli/docker"
            / ("Dockerfile.native" if native else "Dockerfile.python")
        )
        command = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "--file",
            str(dockerfile),
            "--build-arg",
            "RUNTIME_IMAGE=" + base,
            "--build-arg",
            "SOURCE_SHA=" + built["source_sha"],
            "--build-arg",
            "WORKFLOW_ID=" + project["name"],
            "--build-arg",
            "BUNDLE_DIGEST=" + built.get("bundle_digest", ""),
            "--tag",
            image,
            "--iidfile",
            str(iid),
            "--load",
            *contexts,
            str(context),
        ]
        if native:
            command[-1:-1] = [
                "--build-arg",
                "PNMS_NATIVE_ADAPTER=" + built.get("adapter", "tensor"),
            ]
        _run(command, runner=runner, log=log)
        image_id = iid.read_text().strip()
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", image_id),
            "Docker did not produce a valid immutable local image ID",
        )
        result["image_id"] = image_id
        loaded = _inspect(image, runner=runner, log=log)
        require(
            loaded["Id"] == image_id,
            "loaded release tag identity differs from build output",
        )
        result.update(_identity(loaded))
        if native:
            configuration = loaded.get("Config", {})
            require(
                (configuration.get("Entrypoint") or []) == startup["entrypoint"]
                and configuration.get("Cmd") == startup["command"],
                "native image entrypoint must preserve runtime base initialization and forward the service command",
            )
            result["checks"]["entrypoint"] = "passed"
            result["startup"] = startup
            # Recheck after apt/pip stages too: the final image must retain the
            # matched provider closure rather than silently upgrading it.
            result["runtime_stack"] = _stack(
                image_id, expected, library=library, runner=runner, log=log
            )
            result["checks"]["stack"] = "passed"
            contents = json.loads(
                _run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--platform",
                        "linux/amd64",
                        "--env",
                        "NVIDIA_VISIBLE_DEVICES=void",
                        "--env",
                        "CUDA_VISIBLE_DEVICES=-1",
                        "--entrypoint",
                        "python3",
                        image_id,
                        "-c",
                        CONTENT_PROBE,
                        "/opt/physicsnemo-release",
                        built["bundle_digest"],
                        built["source_sha"],
                    ],
                    runner=runner,
                    log=log,
                )
            )
            require(
                contents.get("status") == "passed"
                and contents.get("bundle_digest") == built["bundle_digest"]
                and contents.get("source_sha") == built["source_sha"]
                and contents.get("native_load") == "passed"
                and type(contents.get("files")) is int
                and contents["files"] > 0,
                "built image content or native loader verification did not pass",
            )
            result["checks"].update(contents="passed", native_load="passed")
            result["contents"] = contents
        else:
            plugin_digest = digest(built["python"]["files"])
            contents = _python_probe(
                image_id, PYTHON_CONTENT_PROBE, [plugin_digest], runner=runner, log=log
            )
            require(
                contents.get("status") == "passed"
                and contents.get("plugin_digest") == plugin_digest
                and isinstance(contents.get("workflow_id"), str)
                and type(contents.get("files")) is int
                and contents["files"] > 0,
                "Python image content or launch configuration verification did not pass",
            )
            result["checks"]["contents"] = "passed"
            result["contents"] = contents
        if push:
            require(
                _inspect(image, runner=runner, log=log)["Id"] == image_id,
                "release tag identity binding changed before publication",
            )
            _run(["docker", "push", image], runner=runner, log=log)
            inspected = _inspect(image_id, runner=runner, log=log)
            references = [
                value
                for value in inspected.get("RepoDigests", [])
                if isinstance(value, str)
                and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value)
                and _repository(value) == _repository(image)
            ]
            require(
                len(references) == 1,
                "published registry manifest binding is unavailable",
            )
            published = json.loads(
                _run(
                    [
                        "docker",
                        "buildx",
                        "imagetools",
                        "inspect",
                        "--raw",
                        references[0],
                    ],
                    runner=runner,
                    log=log,
                )
            )
            require(
                published.get("schemaVersion") == 2
                and published.get("config", {}).get("digest")
                == result["image_config_digest"],
                "published registry manifest configuration identity differs from built image",
            )
            result["registry_digest"] = references[0]
        result["status"] = "complete"
        write_json(attempt / "image.json", result)
        return result
    except BaseException as error:
        result.update(status="failed", error=str(error))
        write_json(attempt / "image.json", result)
        raise
