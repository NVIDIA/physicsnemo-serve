"""Owned local HTTP service processes and immutable-image launches."""

from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager, suppress
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import uuid

from . import bundle
from .common import read_json, require, write_json


_IMAGE_ENV_RESERVED = {
    "SOURCE_SHA",
    "RELEASE_IMAGE_DIGEST",
    "PORT",
    "SERVER_PORT",
    "HEALTH_PORT",
    "HEALTH_STUB_ENABLED",
    "HEALTH_STUB_PORT",
    "SCHEDULER_METRICS_PORT",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "REDIS_URL",
    "REDIS_STREAM_PREFIX",
    "STREAM_PREFIX",
    "INFERENCE_STREAM",
    "PREFETCH_STREAM",
    "USE_PREFETCH",
    "QUEUE_CONFIG",
    "WORKER_RUNTIME_CONFIG",
    "WORKER_PIPELINE_CONFIG",
    "WORKER_ROLE",
    "WORKERS",
    "WORKERS_PER_GPU",
    "WORKER_SCRIPT",
    "PLUGIN_DIR",
    "WORKFLOW_DIR",
    "ARTIFACT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "INFERENCE_SERVER_BIN",
    "WORKER_RUNTIME_BIN",
    "RUNTIME_ENV_LAUNCHER_SCRIPT",
    "GPU_LAUNCHER_SCRIPT",
    "POD_NAMESPACE",
    "POD_NAME",
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "TENSORRT_ROOT",
}


def image_environment_names(names):
    """Validate opt-in Docker environment names without reading their values."""
    selected = []
    for name in names:
        require(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name),
            "--env accepts exported variable names only; use --env NAME",
        )
        protected_name = name.upper()
        require(
            protected_name not in _IMAGE_ENV_RESERVED
            and not protected_name.startswith(
                ("PNMS_", "NATIVE_", "PHYSICSNEMO_SERVE_")
            ),
            "--env cannot override launcher-owned identity or runtime variables",
        )
        require(name in os.environ, f"--env variable is not exported: {name}")
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _gpu_binding(project, built):
    requested = project.get("gpu", project.get("build", {}).get("gpu"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allocated = [part.strip() for part in visible.split(",")]
        require(
            all(part and part != "-1" for part in allocated),
            "CUDA GPU allocation is disabled",
        )
        if requested is None:
            recorded = built.get("environment", {}).get("gpu", {}).get("binding")
            require(
                len(allocated) == 1 or recorded in allocated,
                "Select one allocated GPU with --gpu",
            )
            binding = allocated[0] if len(allocated) == 1 else recorded
        elif str(requested).removeprefix("cuda:").isdecimal():
            index = int(str(requested).removeprefix("cuda:"))
            require(index < len(allocated), "GPU index is outside current allocation")
            binding = allocated[index]
        else:
            require(
                str(requested) in allocated,
                "GPU selector is outside current allocation",
            )
            binding = str(requested)
    elif requested is None:
        observed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        devices = observed.stdout.strip().splitlines()
        require(
            observed.returncode == 0 and len(devices) == 1,
            "Select one currently available GPU with --gpu or CUDA_VISIBLE_DEVICES",
        )
        binding = devices[0].strip()
    else:
        binding = str(requested).removeprefix("cuda:")
    require(
        re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", binding),
        "Select one GPU by allocated index or UUID",
    )
    return binding


def _gpu_uuid(binding, built):
    if binding.startswith("GPU-"):
        return binding
    result = subprocess.run(
        ["nvidia-smi", "--id=" + binding, "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    require(
        result.returncode == 0 and re.fullmatch(r"GPU-[A-Za-z0-9-]+", value),
        "Could not resolve selected GPU UUID for host admission",
    )
    return value


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _available(port):
    require(
        type(port) is int and 0 < port < 65536, "HTTP port must be between 1 and 65535"
    )
    # Winsock reuse can share a live listener; exclusive binding rejects it.
    # POSIX reuse allows restarts while closed connections remain in TIME_WAIT.
    option = getattr(socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR)
    # Check both addresses used by the host and container service modes.
    for host in ("127.0.0.1", "0.0.0.0"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, option, 1)
            try:
                listener.bind((host, port))
            except OSError as error:
                raise ValueError(
                    f"Port {port} is already in use; select another port"
                ) from error


def _outside_artifact(project, built, path, name):
    path = Path(path).resolve()
    artifact = (
        built["python"]["plugin_root"]
        if project["runtime"] == "python"
        else built["bundle"]
    )
    require(
        not path.is_relative_to((Path(built["root"]) / artifact).resolve()),
        f"{name} must be outside the selected build artifact: {path}",
    )
    return path


def _workspace(output):
    output = Path(output).resolve()
    require(
        not output.exists() or (output.is_dir() and not any(output.iterdir())),
        "Service requires a fresh empty workspace",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "artifacts").mkdir()
    (output / "logs").mkdir()
    return output


def _identity(built):
    source = built["source_sha"]
    require(
        re.fullmatch(r"[a-fA-F0-9]{40}", source),
        "Build source must identify a full Git revision",
    )
    in_image = os.environ.get("PNMS_SERVICE_BIN_DIR") or os.environ.get(
        "PNMS_IMAGE_CONFIG_DIGEST"
    )
    identity = {
        "SOURCE_SHA": source,
        "NATIVE_DEPLOYMENT_ID": os.environ.get(
            "NATIVE_DEPLOYMENT_ID", "pnms-" + uuid.uuid4().hex
        )
        if in_image
        else "pnms-" + uuid.uuid4().hex,
        "NATIVE_WORKER_SLOT": os.environ.get("NATIVE_WORKER_SLOT", "0")
        if in_image
        else "0",
    }
    image_config = os.environ.get("PNMS_IMAGE_CONFIG_DIGEST")
    if image_config:
        require(
            re.fullmatch(r"sha256:[a-f0-9]{64}", image_config),
            "Invalid Docker image configuration identity",
        )
        identity["PNMS_IMAGE_CONFIG_DIGEST"] = image_config
    elif in_image and os.environ.get("RELEASE_IMAGE_DIGEST"):
        image = os.environ["RELEASE_IMAGE_DIGEST"]
        require(
            re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image),
            "Invalid registry release identity",
        )
        identity["RELEASE_IMAGE_DIGEST"] = image
    return identity


def _process_environment(overrides):
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "RELEASE_IMAGE_DIGEST",
            "PNMS_BUNDLE_DIGEST",
            "PNMS_IMAGE_CONFIG_DIGEST",
            "NATIVE_DEPLOYMENT_OPERATION_ID",
            "NATIVE_WORKFLOW_ONLY",
        }
    }
    return {**inherited, **overrides}


def plan_image(project, built, output, *, port=8080, env_names=()):
    env_names = image_environment_names(env_names)
    require(
        built.get("image"),
        "Run pnms build --image IMAGE before image serving, or use pnms serve --dev",
    )
    image = built["image"]["image_id"]
    require(
        re.fullmatch(r"sha256:[a-f0-9]{64}", image),
        "Image serving requires an immutable Docker image ID",
    )
    config_digest = built["image"].get("image_config_digest", "")
    require(
        re.fullmatch(r"sha256:[a-f0-9]{64}", config_digest),
        "Image serving requires the recorded Docker configuration digest; rebuild the image",
    )
    output = _outside_artifact(project, built, output, "Service output directory")
    resources = built.get("python", {}).get("resources", {}).get("defaults", {})
    needs_gpu = (
        project["runtime"] == "cpp"
        or resources.get("device_kind") == "gpu"
        or resources.get("gpus_required", 0) > 0
    )
    selected = _gpu_binding(project, built) if needs_gpu else None
    _available(port)
    output = _workspace(output)
    identity = _identity(built)
    identity.update(
        PNMS_IMAGE_CONFIG_DIGEST=config_digest,
        PORT="8080",
        SERVER_PORT="8080",
        HEALTH_STUB_ENABLED="false",
    )
    if selected is not None:
        identity["CUDA_VISIBLE_DEVICES"] = "0"
    name = "pnms-" + uuid.uuid4().hex
    require(
        "," not in str(output), "Docker service workspace paths cannot contain commas"
    )
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--cidfile",
        str(output / "container.id"),
        "--publish",
        f"127.0.0.1:{port}:8080",
        "--mount",
        f"type=bind,source={output / 'artifacts'},target=/outputs",
    ]
    if selected is not None:
        argv.extend(["--gpus", "device=" + selected])
    for key, value in identity.items():
        argv.extend(["--env", f"{key}={value}"])
    for env_name in env_names:
        argv.extend(["--env", env_name])
    argv.append(image)
    return {
        "kind": "image",
        "runtime": project["runtime"],
        "output": str(output),
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "argv": argv,
        "container_name": name,
        "deployment_id": identity["NATIVE_DEPLOYMENT_ID"],
        "bundle_digest": built.get("bundle_digest"),
        "startup_timeout": project.get("serve", {}).get("startup_timeout_seconds", 300),
        "shutdown_timeout": 2
        * project.get("serve", {}).get("drain_timeout_seconds", 60)
        + 90,
    }


def plan_host(project, built, output, *, repository, port=8080, skip_build=False):
    require(
        project["runtime"] == "cpp",
        "Python host serving uses the existing plugin development adapter",
    )
    output = _outside_artifact(project, built, output, "Service output directory")
    artifact_root = _outside_artifact(
        project,
        built,
        project.get("serve", {}).get("artifact_root", output / "artifacts"),
        "Service artifact root",
    )
    binding = _gpu_binding(project, built)
    device_uuid = _gpu_uuid(binding, built)
    _available(port)
    output = _workspace(output)
    location = Path(built["root"]) / built["bundle"]
    descriptor = read_json(location / "bundle.json")
    artifact_root.mkdir(parents=True, exist_ok=True)
    deployment = bundle.materialize_deployment(
        location, output, artifact_root=artifact_root, input_cache=output / "cache"
    )
    config_path = output / "deployment.json"
    runtime_path = location / "configuration/worker_runtime_config.json"
    runtime = read_json(runtime_path)
    redis_port = _free_port()
    prefix = runtime.get("stream_prefix", "physicsnemo:")
    identity = _identity(built)
    if not ({"PNMS_IMAGE_CONFIG_DIGEST", "RELEASE_IMAGE_DIGEST"} & identity.keys()):
        identity["PNMS_BUNDLE_DIGEST"] = built["bundle_digest"]
    loader_settings = built.get("loader_environment", {})
    in_image = bool(
        os.environ.get("PNMS_SERVICE_BIN_DIR") or "PNMS_IMAGE_CONFIG_DIGEST" in identity
    )
    if in_image:
        from .native import loader_environment

        explicit = [
            Path(item)
            for item in os.environ.get("PNMS_NATIVE_LIBRARY_DIRS", "").split(os.pathsep)
            if item
        ]
        # The image's selected runtime stack may need UCX/HPC or SDK libraries.
        # Host mode uses its recorded selected loader environment instead.
        explicit.extend(
            Path(item)
            for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
            for item in os.environ.get(key, "").split(os.pathsep)
            if item
        )
        tensorrt = os.environ.get("TENSORRT_ROOT") or os.environ.get(
            "PNMS_TENSORRT_ROOT"
        )
        if tensorrt:
            explicit.extend(Path(tensorrt) / suffix for suffix in ("lib", "lib64"))
        loader_settings = {
            key: value
            for key, value in loader_environment(
                python=Path(sys.executable),
                extra=(*explicit, location / "workflow/lib"),
            ).items()
            if key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
        }
    environment = {
        **loader_settings,
        **identity,
        "CUDA_VISIBLE_DEVICES": binding,
        "PNMS_DEVICE_UUID": device_uuid,
        "PORT": str(port),
        "SERVER_PORT": str(port),
        "PHYSICSNEMO_SERVE_BIND_HOST": "0.0.0.0" if in_image else "127.0.0.1",
        "REDIS_URL": f"redis://127.0.0.1:{redis_port}/0",
        "REDIS_STREAM_PREFIX": prefix,
        "INFERENCE_STREAM": prefix + "schedule",
        "NATIVE_DEPLOYMENT_CONFIG": str(config_path),
        "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG": str(runtime_path),
        "WORKER_RUNTIME_CONFIG": str(runtime_path),
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": descriptor["name"],
        "PLUGIN_DIR": str(location / "configuration/plugin"),
        "ARTIFACT_DIR": str(artifact_root),
        "DEFAULT_OUTPUT_DIR": str(artifact_root),
        "RUST_LOG": os.environ.get("RUST_LOG", "info"),
        "USE_PREFETCH": "false",
    }
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    environment[loader] = os.pathsep.join(
        filter(
            None,
            [
                str(location / "workflow/lib"),
                str(location / "workflow/lib64"),
                environment.get(loader),
            ],
        )
    )
    target = Path(project["root"]) / ".pnms/cache/server"
    binary_directory = Path(os.environ.get("PNMS_SERVICE_BIN_DIR", target / "debug"))
    binaries = {
        name: binary_directory / name
        for name in ("inference_server", "worker-runtime", "native-worker")
    }
    build_environment = {
        **environment,
        "CARGO_TARGET_DIR": str(target),
        "NATIVE_WORKFLOW_INCLUDE": str(location / "workflow/include"),
        "NATIVE_WORKFLOW_LIB_DIR": str(location / "workflow/lib"),
        "PYO3_PYTHON": descriptor["environment"]["python"],
    }
    if descriptor.get("adapter") == "transolver_surface":
        build_environment.update(
            NATIVE_RUNNER_INCLUDE=str(location / "workflow/include"),
            NATIVE_RUNNER_LIB_DIR=str(location / "workflow/lib"),
        )
    else:
        build_environment["NATIVE_WORKFLOW_ONLY"] = "1"
    abi = descriptor["environment"].get("stack", {}).get("cxx11_abi")
    if type(abi) is bool:
        flags = os.environ.get("CXXFLAGS", "")
        explicit_abi = re.findall(r"_GLIBCXX_USE_CXX11_ABI=([01])", flags)
        require(
            not explicit_abi or all(value == str(int(abi)) for value in explicit_abi),
            "CXXFLAGS C++ ABI differs from the compiled workflow",
        )
        build_environment["CXXFLAGS"] = (
            flags + f" -D_GLIBCXX_USE_CXX11_ABI={int(abi)}"
        ).strip()
    processes = [
        {
            "name": "redis",
            "argv": [
                "redis-server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(redis_port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--maxmemory-policy",
                "noeviction",
            ],
            "env": {},
        },
        {
            "name": "native-worker",
            "argv": [str(binaries["native-worker"]), str(config_path)],
            "env": environment,
        },
    ]
    for role in ("scheduler", "results"):
        processes.append(
            {
                "name": role,
                "argv": [
                    str(binaries["worker-runtime"]),
                    "--role",
                    role,
                    "--config-path",
                    str(runtime_path),
                ],
                "env": {**environment, "HEALTH_PORT": str(_free_port())},
            }
        )
    processes.append(
        {
            "name": "inference_server",
            "argv": [str(binaries["inference_server"])],
            "env": environment,
        }
    )
    return {
        "kind": "host",
        "in_image": in_image,
        "runtime": "cpp",
        "output": str(output),
        "repository": str(repository),
        "port": port,
        "redis_port": redis_port,
        "url": f"http://127.0.0.1:{port}",
        "environment": environment,
        "processes": processes,
        "build_environment": build_environment,
        "build_argv": [
            "cargo",
            "build",
            "--manifest-path",
            str(Path(repository) / "Cargo.toml"),
            "-p",
            "inference_server",
            "-p",
            "worker-runtime",
            "-p",
            "native-worker",
            "--features",
            "native-worker/cpp",
        ],
        "binaries": [str(path) for path in binaries.values()],
        "skip_build": skip_build,
        "deployment_id": identity["NATIVE_DEPLOYMENT_ID"],
        "bundle_digest": built["bundle_digest"],
        "startup_timeout": deployment.get("startup_timeout_seconds", 300),
        "shutdown_timeout": 2 * deployment.get("drain_timeout_seconds", 60) + 30,
    }


def _check_children(processes):
    for name, child in processes:
        require(
            child.poll() is None,
            f"Service process {name} exited before shutdown; inspect its workspace log",
        )


def _stop(child, timeout):
    if child.poll() is not None:
        return
    try:
        child.terminate()
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            child.kill()
        child.wait(timeout=5)


@contextmanager
def _cleanup_signals():
    """Finish owned cleanup even if another termination signal arrives."""
    previous = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, signal.SIG_IGN)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _supervise(plan):
    processes = []
    streams = []
    try:
        logs = Path(plan["output"]) / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        def start(process, stream):
            return subprocess.Popen(
                process["argv"],
                env=_process_environment(process.get("env", {})),
                cwd=plan.get("repository"),
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        worker_process = None
        worker_stream = None
        for process in plan["processes"]:
            stream = (logs / (process["name"] + ".log")).open("a", encoding="utf-8")
            streams.append(stream)
            child = start(process, stream)
            processes.append((process["name"], child))
            if process["name"] == "redis":
                _wait_redis(plan, child)
            elif process["name"] == "native-worker":
                worker_process, worker_stream = process, stream
                _wait_native_worker(plan, child)
            elif process["name"] == "scheduler":
                _wait_scheduler(plan, child, process["env"]["HEALTH_PORT"])
        _wait_ready(plan, processes)
        print(f"Service ready: {plan['url']}\nLogs: {logs}", flush=True)
        restarts = 0
        while True:
            peers = [
                (name, child) for name, child in processes if name != "native-worker"
            ]
            _check_children(peers)
            if plan.get("runtime") == "cpp" and worker_process is not None:
                index = next(
                    index
                    for index, (name, _) in enumerate(processes)
                    if name == "native-worker"
                )
                worker = processes[index][1]
                if worker.poll() is not None:
                    # Only a reaped child can release its owned lease. Close stale
                    # readiness before waiting or starting its replacement.
                    _retire_native_worker(plan, worker)
                    _stop(worker, plan.get("shutdown_timeout", 30))
                    require(
                        restarts < 3,
                        "Native worker restart budget exhausted; inspect logs/native-worker.log",
                    )
                    time.sleep(0.25 * 2**restarts)
                    _check_children(peers)
                    replacement = start(worker_process, worker_stream)
                    processes[index] = ("native-worker", replacement)
                    restarts += 1
                    _wait_native_worker(plan, replacement, peers)
                    _wait_ready(plan, processes)
                    print(
                        f"Native worker recovered ({restarts}/3); service ready: {plan['url']}",
                        flush=True,
                    )
            _check_children(processes)
            time.sleep(0.25)
    except KeyboardInterrupt:
        return {"status": "stopped", "workspace": plan["output"]}
    finally:
        with _cleanup_signals():
            for name, child in reversed(processes):
                try:
                    _stop(child, plan.get("shutdown_timeout", 30))
                except (OSError, subprocess.TimeoutExpired) as error:
                    print(
                        f"Could not finish cleanup for owned service process {name}: {error}",
                        file=sys.stderr,
                    )
            for stream in streams:
                stream.close()


def _wait_ready(plan, processes):
    deadline = time.monotonic() + plan.get("startup_timeout", 300)
    while time.monotonic() < deadline:
        _check_children(processes)
        if _http_ready(plan):
            return
        time.sleep(0.1)
    raise ValueError(
        f"Service did not become ready; inspect {Path(plan['output']) / 'logs'}"
    )


def _http_ready(plan):
    try:
        with urlopen(plan["url"] + "/readyz", timeout=1) as response:
            data = json.load(response)
        if plan["runtime"] == "cpp":
            native = data.get("native", {})
            return (
                data.get("ready") is True
                and native.get("ready") is True
                and native.get("bundle_digest") == plan["bundle_digest"]
                and native.get("deployment_id") == plan["deployment_id"]
            )
        return data.get("status") == "ready"
    except (OSError, HTTPError, URLError, ValueError):
        return False


def _wait_redis(plan, child):
    import redis

    client = redis.Redis.from_url(
        plan["environment"]["REDIS_URL"], socket_connect_timeout=0.5, socket_timeout=0.5
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            require(child.poll() is None, "Owned Redis process exited during startup")
            try:
                if client.info("server").get("process_id") == child.pid:
                    return
            except redis.RedisError:
                pass
            time.sleep(0.1)
        raise ValueError("Owned Redis process did not become ready")
    finally:
        client.close()


def _retire_native_worker(plan, child):
    """Fence a dead owned process; retain its warming slot and request streams."""
    import redis

    require(child.poll() is not None, "Cannot retire a live native worker")
    client = redis.Redis.from_url(
        plan["environment"]["REDIS_URL"], socket_connect_timeout=0.5, socket_timeout=0.5
    )
    try:
        candidates = []
        slot = plan["environment"].get("NATIVE_WORKER_SLOT", "0")
        for field, raw in client.hgetall("gpu:registry").items():
            try:
                record = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(record, dict)
                and record.get("deployment_id") == plan["deployment_id"]
                and record.get("worker_slot") == slot
            ):
                candidates.append((field, raw, record))
        require(
            len(candidates) <= 1, "Native worker slot has ambiguous registry ownership"
        )
        for field, raw, record in candidates:
            owner = record.get("worker_instance_id")
            require(
                record.get("pid") == child.pid
                and record.get("bundle_digest") == plan["bundle_digest"]
                and isinstance(owner, str)
                and owner,
                "Native worker registry belongs to a different process or bundle",
            )
            encoded = field if isinstance(field, bytes) else field.encode()
            lease = "native:worker-lease:" + hashlib.sha256(encoded).hexdigest()
            record["status"] = "warming"
            record["model_cache"] = {
                "schema_version": 1,
                "scope": "process",
                "entries": [],
                "total_entries": 0,
                "warmup": {
                    "workflow_id": record.get("workflow_id"),
                    "status": "warming",
                    "reason": "restart_pending",
                },
            }
            accepted = client.eval(
                "local current=redis.call('HGET',KEYS[1],ARGV[1]); "
                "local owner=redis.call('GET',KEYS[2]); "
                "if current~=ARGV[2] or (owner and owner~=ARGV[3]) then return 0 end; "
                "redis.call('HSET',KEYS[1],ARGV[1],ARGV[4]); "
                "if owner==ARGV[3] then redis.call('DEL',KEYS[2]) end; return 1",
                2,
                "gpu:registry",
                lease,
                field,
                raw,
                owner,
                json.dumps(record),
            )
            require(accepted == 1, "Native worker ownership changed during retirement")
    finally:
        client.close()


def _wait_native_worker(plan, child, companions=()):
    import redis

    client = redis.Redis.from_url(
        plan["environment"]["REDIS_URL"], socket_connect_timeout=0.5, socket_timeout=0.5
    )
    try:
        deadline = time.monotonic() + plan.get("startup_timeout", 300)
        while time.monotonic() < deadline:
            _check_children(companions)
            status = child.poll()
            detail = f"return code {status}"
            if status is not None and status < 0:
                detail += f", signal {-status}"
            require(
                status is None,
                f"Native worker exited during startup ({detail}); inspect logs/native-worker.log",
            )
            try:
                records = client.hgetall("gpu:registry").values()
                for raw in records:
                    try:
                        record = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if (
                        isinstance(record, dict)
                        and record.get("status") == "available"
                        and record.get("deployment_id") == plan["deployment_id"]
                        and record.get("bundle_digest") == plan["bundle_digest"]
                        and record.get("pid") == child.pid
                    ):
                        return
            except redis.RedisError:
                pass
            time.sleep(0.1)
        raise ValueError(
            "Native worker did not register as available; inspect logs/native-worker.log"
        )
    finally:
        client.close()


def _wait_scheduler(plan, child, port):
    deadline = time.monotonic() + plan.get("startup_timeout", 300)
    while time.monotonic() < deadline:
        require(
            child.poll() is None,
            "Scheduler exited during startup; inspect logs/scheduler.log",
        )
        try:
            with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1) as response:
                metrics = response.read()
            if isinstance(metrics, bytes):
                metrics = metrics.decode("utf-8")
            discovered = re.search(
                r"^physicsnemo_serve_scheduler_discovered_workers\s+(\S+)\s*$",
                metrics,
                re.MULTILINE,
            )
            if discovered and float(discovered.group(1)) >= 1:
                return
        except (OSError, HTTPError, URLError, ValueError):
            pass
        time.sleep(0.1)
    raise ValueError(
        "Scheduler did not discover the native worker; inspect logs/scheduler.log"
    )


def serve(
    project,
    built,
    *,
    repository,
    output,
    port=8080,
    dev=False,
    skip_build=False,
    dry_run=False,
    env_names=(),
):
    require(
        not (dev and env_names),
        "--env is image-serving-only; --dev already inherits exported host variables",
    )
    plan = (
        plan_host(
            project,
            built,
            output,
            repository=repository,
            port=port,
            skip_build=skip_build,
        )
        if dev
        else plan_image(project, built, output, port=port, env_names=env_names)
    )
    write_json(Path(plan["output"]) / "launch.json", plan)
    if dry_run:
        return {"status": "planned", **plan}
    if plan["kind"] == "image":
        return _serve_image(plan)
    if not plan.get("in_image"):
        from .admission import verify_providers

        verify_providers(
            built,
            environment=_process_environment(plan["environment"]),
            output=Path(plan["output"]) / "checks/providers.json",
        )
    require(
        shutil.which("redis-server"),
        "Host serving requires redis-server; install it before pnms serve --dev",
    )
    if not skip_build:
        require(
            shutil.which("cargo"),
            "Host serving requires Cargo for the service binaries",
        )
        with (Path(plan["output"]) / "logs/build.log").open("w") as stream:
            result = subprocess.run(
                plan["build_argv"],
                cwd=repository,
                env=_process_environment(plan["build_environment"]),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        require(
            result.returncode == 0,
            f"Service build failed; inspect {plan['output']}/logs/build.log",
        )
    require(
        all(os.access(path, os.X_OK) for path in plan["binaries"]),
        "Missing service binaries; rerun without --skip-build",
    )
    return _supervise(plan)


def _serve_image(plan):
    output = Path(plan["output"])
    container = None
    try:
        result = subprocess.run(
            plan["argv"], capture_output=True, text=True, check=False
        )
        (output / "logs/docker-run.log").write_text(result.stdout + result.stderr)
        cidfile = output / "container.id"
        if cidfile.is_file() and not cidfile.is_symlink():
            value = cidfile.read_text().strip()
            require(
                re.fullmatch(r"[a-f0-9]{64}", value), "Invalid owned container identity"
            )
            container = value
        require(
            result.returncode == 0 and container is not None,
            f"Docker service launch failed; inspect {output}/logs/docker-run.log",
        )
        child = _Container(container)
        _wait_ready(plan, [("container", child)])
        print(f"Service ready: {plan['url']}\nWorkspace: {output}", flush=True)
        while True:
            _check_children([("container", child)])
            time.sleep(0.5)
    except KeyboardInterrupt:
        return {"status": "stopped", "workspace": str(output)}
    finally:
        with _cleanup_signals():
            if container is None:
                # Docker may have created the container before Ctrl-C interrupted run.
                cidfile = output / "container.id"
                if cidfile.is_file() and not cidfile.is_symlink():
                    value = cidfile.read_text().strip()
                    if re.fullmatch(r"[a-f0-9]{64}", value):
                        container = value
            if container is not None:
                with suppress(OSError, subprocess.TimeoutExpired):
                    subprocess.run(
                        [
                            "docker",
                            "stop",
                            "--time",
                            str(plan["shutdown_timeout"]),
                            container,
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=plan["shutdown_timeout"] + 10,
                    )
                with suppress(OSError, subprocess.TimeoutExpired):
                    with (output / "logs/container.log").open("w") as stream:
                        subprocess.run(
                            ["docker", "logs", container],
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                            timeout=10,
                        )
                with suppress(OSError, subprocess.TimeoutExpired):
                    subprocess.run(
                        ["docker", "rm", "--force", container],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )


class _Container:
    def __init__(self, identity):
        self.identity = identity

    def poll(self):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", self.identity],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode:
            return 1
        state = json.loads(result.stdout)
        return None if state["Running"] else state["ExitCode"]
