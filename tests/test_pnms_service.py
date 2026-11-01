"""CPU component checks for service launch configuration and owned cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "application-cli/src"))
from pnms_app import bundle, service  # noqa: E402


def _await_file(path, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        assert process.poll() is None, f"launcher exited with {process.returncode}"
        time.sleep(0.02)
    pytest.fail(f"Timed out waiting for {path}")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process and signal ownership")
@pytest.mark.parametrize("mode", ["host", "image", "python"])
def test_cli_sigterm_cleans_owned_services_despite_repeated_signals(tmp_path, mode):
    """Use real child processes; the image variant substitutes only Docker itself."""
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "root, name = Path(sys.argv[1]), sys.argv[2]\n"
        "def stop(*_):\n"
        "    (root / 'stopping').touch()\n"
        "    time.sleep(0.5)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "if len(sys.argv) > 3:\n"
        "    subprocess.Popen([sys.executable, __file__, str(root), sys.argv[3]])\n"
        "    while not (root / (sys.argv[3] + '.pid')).is_file(): time.sleep(0.01)\n"
        "(root / (name + '.pid')).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"root = Path({str(tmp_path)!r})\n"
        "command = sys.argv[1]\n"
        "if command == 'run':\n"
        f"    subprocess.Popen([{sys.executable!r}, {str(sleeper)!r}, str(root), 'container'], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    (root / 'container.id').write_text('a' * 64)\n"
        "elif command == 'inspect':\n"
        "    print(json.dumps({'Running': True, 'ExitCode': 0}))\n"
        "elif command == 'stop':\n"
        "    (root / 'stopping').touch()\n"
        "    time.sleep(0.5)\n"
        "    os.kill(int((root / 'container.pid').read_text()), signal.SIGTERM)\n"
        "    time.sleep(0.7)\n"
        "elif command == 'rm':\n"
        "    (root / 'removed').touch()\n"
    )
    docker.chmod(0o755)
    plan = {
        "output": str(tmp_path),
        "url": "unused-no-http",
        "shutdown_timeout": 3,
        "environment": {},
        "processes": [
            {"name": name, "argv": [sys.executable, str(sleeper), str(tmp_path), name]}
            for name in ("first", "second")
        ],
        "argv": [str(docker), "run"],
    }
    if mode == "python":
        plan["processes"][0]["argv"].append("descendant")
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        plan.update(
            workflow_id="test",
            workspace=str(tmp_path),
            server_url="unused-no-http",
            example_request={"command": "unused"},
        )
        (scripts / "plugin_dev.py").write_text(
            "import builtins, importlib.util\n"
            "from pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('legacy_plugin', {str(REPOSITORY / 'scripts/plugin_dev.py')!r})\n"
            "legacy = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(legacy)\n"
            f"plan = {plan!r}\n"
            "def output(*args, **kwargs):\n"
            "    builtins.print(*args, **kwargs)\n"
            "    if args and args[0] == 'Press Ctrl-C to stop.':\n"
            "        (Path(plan['output']) / 'ready').touch()\n"
            "legacy.print = output\n"
            "def run_local_plugin(*args, **kwargs):\n"
            "    legacy.run_local_stack(plan)\n"
            "    return {'status': 'completed'}\n"
        )
        (tmp_path / "plugin").mkdir()
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(REPOSITORY / 'application-cli/src')!r})\n"
        "from pnms_app import cli, service\n"
        f"plan = {plan!r}\n"
        "root = Path(plan['output'])\n"
        "(root / 'logs').mkdir()\n"
        "def ready(*_):\n"
        f"    expected = {['container.pid'] if mode == 'image' else ['first.pid', 'second.pid']!r}\n"
        "    while not all((root / name).is_file() for name in expected): time.sleep(0.01)\n"
        "    (root / 'ready').touch()\n"
        "service._wait_ready = ready\n"
        + (
            "from pnms_app import python as adapter\n"
            "cli.execute = lambda args: adapter.serve_project(\n"
            "    {'root': root / 'plugin', 'plugin': root / 'plugin'},\n"
            "    repository=root, workspace=root / 'workspace')\n"
            if mode == "python"
            else f"cli.execute = lambda args: service.{'_supervise' if mode == 'host' else '_serve_image'}(plan)\n"
        )
        + "raise SystemExit(cli.main(['serve']))\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(launcher)],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _await_file(tmp_path / "ready", process)
        process.send_signal(signal.SIGTERM)
        _await_file(tmp_path / "stopping", process)
        process.send_signal(signal.SIGTERM)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == (130 if mode == "python" else 0), (stdout, stderr)
        for path in tmp_path.glob("*.pid"):
            pid = int(path.read_text())
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
            ).stdout.strip()
            assert not state or state.startswith("Z"), f"owned process {pid} survived"
        if mode == "image":
            assert (tmp_path / "removed").is_file()
    finally:
        if process.poll() is None:
            process.kill()
        for path in tmp_path.glob("*.pid"):
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate(timeout=5)


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("import sys; sys.exit(23)", "return code 23"),
        (
            "import signal; signal.raise_signal(signal.SIGTERM)",
            "return code 3"
            if sys.platform == "win32"
            else "return code -15, signal 15",
        ),
    ],
)
def test_startup_failure_reports_actual_native_process_exit(command, message):
    child = subprocess.Popen([sys.executable, "-c", command])
    try:
        child.wait(timeout=5)
        with pytest.raises(ValueError, match=message):
            service._wait_native_worker(
                {"environment": {"REDIS_URL": "redis://127.0.0.1:1"}}, child
            )
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_port_admission_allows_reusable_closed_server_connections():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    # A real reusable server can bind after the peer exchange closes. The
    # admission probe must allow the same restart even while TCP state remains.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as replacement:
        replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        replacement.bind(("0.0.0.0", port))
        replacement.listen()
    service._available(port)


def test_port_admission_still_rejects_a_live_reusable_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        with pytest.raises(ValueError, match="already in use"):
            service._available(listener.getsockname()[1])


def test_port_admission_still_rejects_a_live_loopback_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(ValueError, match="already in use"):
            service._available(listener.getsockname()[1])


@pytest.mark.parametrize("live_host", ["127.0.0.1", "0.0.0.0"])
def test_port_admission_rejects_winsock_reusable_listener(monkeypatch, live_host):
    exclusive_option = -5

    class WinsockSocket:
        exclusive = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, level, option, value):
            if option == exclusive_option:
                self.exclusive = bool(value)

        def bind(self, address):
            # Winsock permits SO_REUSEADDR to share an existing listener.
            host, _ = address
            if self.exclusive and (
                live_host == "0.0.0.0" or host in (live_host, "0.0.0.0")
            ):
                raise OSError("address already in use")

    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", exclusive_option, raising=False)
    monkeypatch.setattr(socket, "socket", lambda *args: WinsockSocket())
    with pytest.raises(ValueError, match="already in use"):
        service._available(18082)


@pytest.fixture
def native(tmp_path, monkeypatch):
    root = tmp_path / "build"
    location = root / "bundle"
    (location / "workflow/lib").mkdir(parents=True)
    (location / "workflow/include").mkdir()
    (location / "configuration").mkdir()
    descriptor = {
        "name": "demo",
        "executor_class": "native.demo",
        "environment": {
            "python": sys.executable,
            "gpu": {"binding": "GPU-allocated", "uuid": "GPU-allocated"},
        },
    }
    (location / "bundle.json").write_text(json.dumps(descriptor))
    (location / "configuration/worker_runtime_config.json").write_text(
        '{"roles":{"scheduler":{},"results":{}}}'
    )

    def materialize(_bundle, output, *, artifact_root, input_cache):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "workflow_id": "demo",
            "executor_class": "native.demo",
            "device_index": 0,
            "drain_timeout_seconds": 1,
            "startup_timeout_seconds": 1,
            "artifact_root": str(artifact_root),
            "input_cache": str(input_cache),
        }
        (output / "deployment.json").write_text(json.dumps(config))
        return config

    monkeypatch.setattr(bundle, "materialize_deployment", materialize)
    project = {"root": tmp_path / "app", "name": "demo", "runtime": "cpp"}
    built = {
        "root": str(root),
        "bundle": "bundle",
        "source_sha": "a" * 40,
        "bundle_digest": "b" * 64,
        "environment": descriptor["environment"],
        "loader_environment": {"LD_LIBRARY_PATH": "/selected/sdk/lib"},
        "image": {
            "image_id": "sha256:" + "c" * 64,
            "image_config_digest": "sha256:" + "d" * 64,
            "local_tag": "mutable:latest",
        },
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated")
    monkeypatch.delenv("PNMS_IMAGE_CONFIG_DIGEST", raising=False)
    monkeypatch.setattr(service, "_available", lambda port: None)
    return project, built


@pytest.mark.parametrize(
    "runtime,dev", [("python", True), ("cpp", True), ("python", False), ("cpp", False)]
)
@pytest.mark.parametrize("port,expected", [(None, 9000), (18082, 18082), (0, None)])
def test_cli_serve_port_selection_and_validation(
    native, monkeypatch, capsys, runtime, dev, port, expected
):
    from pnms_app import build, cli, python

    project, built = native
    project["runtime"] = runtime
    if runtime == "python" and not dev:
        built["python"] = {"plugin_root": "plugin"}
    project["serve"] = {"port": 9000}
    monkeypatch.setattr(cli, "load_project", lambda *args, **kwargs: project)
    monkeypatch.setattr(build, "last_build", lambda *args, **kwargs: built)
    ports = []

    def serve(*args, **kwargs):
        ports.append(kwargs["port"])
        return {}

    monkeypatch.setattr(python, "serve_project", serve)
    monkeypatch.setattr(service, "serve", serve)
    arguments = ["serve", str(project["root"])]
    if dev:
        arguments.append("--dev")
    if port is not None:
        arguments += ["--port", str(port)]

    status = cli.main(arguments)

    if expected is None:
        assert status == 2
        assert "port must be 1..65535" in capsys.readouterr().err
        assert ports == []
    else:
        assert status == 0
        assert ports == [expected]


def test_image_launch_uses_immutable_id_and_one_allocated_gpu(
    native, tmp_path, monkeypatch
):
    project, built = native
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-current-allocation")
    plan = service.plan_image(project, built, tmp_path / "serve", port=18082)
    argv = plan["argv"]
    assert argv[-1] == built["image"]["image_id"]
    assert argv[argv.index("--gpus") + 1] == "device=GPU-current-allocation"
    assert argv[argv.index("--publish") + 1] == "127.0.0.1:18082:8080"
    assert "mutable:latest" not in argv
    assert f"PNMS_IMAGE_CONFIG_DIGEST={built['image']['image_config_digest']}" in argv


def test_host_plan_uses_workspace_paths_and_bundle_identity(native, tmp_path):
    project, built = native
    output = tmp_path / "serve"
    plan = service.plan_host(project, built, output, repository=REPOSITORY, port=18082)
    env = plan["environment"]
    assert env["PNMS_BUNDLE_DIGEST"] == built["bundle_digest"]
    assert "RELEASE_IMAGE_DIGEST" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-allocated"
    assert env["PNMS_DEVICE_UUID"] == "GPU-allocated"
    assert env["ARTIFACT_DIR"] == str(output / "artifacts")
    assert env["PHYSICSNEMO_SERVE_BIND_HOST"] == "127.0.0.1"
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    assert env[loader].split(os.pathsep)[0] == str(
        Path(built["root"]) / "bundle/workflow/lib"
    )
    assert {item["name"] for item in plan["processes"]} == {
        "redis",
        "scheduler",
        "results",
        "native-worker",
        "inference_server",
    }
    names = [item["name"] for item in plan["processes"]]
    assert names.index("native-worker") < names.index("scheduler")
    assert '"/app/' not in json.dumps(plan)
    assert plan["build_environment"]["NATIVE_WORKFLOW_ONLY"] == "1"
    assert plan["environment"]["REDIS_URL"].startswith("redis://127.0.0.1:")


@pytest.mark.parametrize(
    "runtime,dev,alias",
    [("cpp", True, False), ("cpp", False, True), ("python", False, True)],
)
def test_service_planning_keeps_writes_outside_selected_artifact(
    native, tmp_path, monkeypatch, runtime, dev, alias
):
    from pnms_app.common import inventory, verify_inventory

    project, built = native
    project["runtime"] = runtime
    if runtime == "python":
        built["python"] = {"plugin_root": "plugin"}
        selected = Path(built["root"]) / "plugin"
        selected.mkdir()
        (selected / "plugin.yaml").write_text("name: frozen\n")
    else:
        selected = Path(built["root"]) / built["bundle"]
    before = inventory(selected)
    monkeypatch.chdir(tmp_path)
    output = selected.relative_to(tmp_path) / "new/../service"
    if alias:
        link = tmp_path / "selected-artifact"
        link.symlink_to(selected, target_is_directory=True)
        output = link / "new/../service"
    probes = []
    monkeypatch.setattr(
        service, "_gpu_binding", lambda *args: probes.append("gpu") or "GPU-allocated"
    )
    monkeypatch.setattr(service, "_available", lambda *args: probes.append("port"))

    def plan(output):
        if dev:
            return service.plan_host(project, built, output, repository=REPOSITORY)
        return service.plan_image(project, built, output)

    with pytest.raises(ValueError, match="output.*outside"):
        plan(output)
    with pytest.raises(ValueError, match="output.*outside"):
        service.serve(
            project, built, repository=REPOSITORY, output=output, dev=dev, dry_run=True
        )
    assert probes == []
    assert not (selected / "service").exists()
    verify_inventory(selected, before)

    # A sibling workspace may retain launch files without changing the artifact.
    outside = selected.parent / "outside-service"
    result = service.serve(
        project, built, repository=REPOSITORY, output=outside, dev=dev, dry_run=True
    )
    assert result["status"] == "planned"
    assert (outside / "launch.json").is_file()
    verify_inventory(selected, before)


def test_host_plan_keeps_configured_artifact_root_outside_bundle(native, tmp_path):
    from pnms_app.common import inventory, verify_inventory

    project, built = native
    selected = Path(built["root"]) / built["bundle"]
    before = inventory(selected)
    alias = tmp_path / "selected-artifact"
    alias.symlink_to(selected, target_is_directory=True)
    project["serve"] = {"artifact_root": str(alias / "new/../artifacts")}
    output = tmp_path / "serve"

    with pytest.raises(ValueError, match="artifact root.*outside"):
        service.plan_host(project, built, output, repository=REPOSITORY)
    assert not output.exists()
    assert not (selected / "artifacts").exists()
    verify_inventory(selected, before)

    artifact_root = selected.parent / "outside-artifacts"
    project["serve"]["artifact_root"] = str(artifact_root)
    plan = service.plan_host(project, built, output, repository=REPOSITORY)
    assert artifact_root.is_dir()
    assert plan["environment"]["ARTIFACT_DIR"] == str(artifact_root)
    verify_inventory(selected, before)


def test_cfd_host_plan_builds_runner_adapter_with_shared_workflow_bridge(
    native, tmp_path, monkeypatch
):
    project, built = native
    monkeypatch.setenv("NATIVE_WORKFLOW_ONLY", "1")
    location = Path(built["root"]) / built["bundle"]
    descriptor = json.loads((location / "bundle.json").read_text())
    descriptor["adapter"] = "transolver_surface"
    (location / "bundle.json").write_text(json.dumps(descriptor))

    plan = service.plan_host(project, built, tmp_path / "serve", repository=REPOSITORY)

    environment = plan["build_environment"]
    assert "NATIVE_WORKFLOW_ONLY" not in environment
    assert environment["NATIVE_RUNNER_INCLUDE"] == str(location / "workflow/include")
    assert environment["NATIVE_RUNNER_LIB_DIR"] == str(location / "workflow/lib")
    assert environment["NATIVE_WORKFLOW_INCLUDE"] == str(location / "workflow/include")
    assert environment["NATIVE_WORKFLOW_LIB_DIR"] == str(location / "workflow/lib")
    assert "NATIVE_WORKFLOW_ONLY" not in service._process_environment(environment)


def test_supervisor_gates_scheduler_and_http_on_worker_discovery(tmp_path, monkeypatch):
    events = []

    class Child:
        pid = 123456789

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def start(argv, **kwargs):
        events.append(argv[0])
        return Child()

    monkeypatch.setattr(service.subprocess, "Popen", start)
    monkeypatch.setattr(
        service,
        "_wait_native_worker",
        lambda *args: events.append("available"),
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "_wait_scheduler",
        lambda *args: events.append("discovered"),
        raising=False,
    )

    def stop_after_start(*args):
        raise ValueError("finished startup check")

    monkeypatch.setattr(service, "_wait_ready", stop_after_start)
    plan = {
        "output": str(tmp_path),
        "processes": [
            {"name": name, "argv": [name], "env": {"HEALTH_PORT": "18083"}}
            for name in ("native-worker", "scheduler", "inference_server")
        ],
        "environment": {},
        "shutdown_timeout": 1,
    }
    with pytest.raises(ValueError, match="finished startup check"):
        service._supervise(plan)
    assert events == [
        "native-worker",
        "available",
        "scheduler",
        "discovered",
        "inference_server",
    ]


def test_worker_startup_requires_own_available_bundle_record(monkeypatch):
    import redis

    expected = {
        "status": "available",
        "deployment_id": "ours",
        "bundle_digest": "b" * 64,
        "pid": 123,
    }
    records = [
        {**expected, "status": "warming"},
        {**expected, "deployment_id": "another"},
        {**expected, "bundle_digest": "other"},
        {**expected, "pid": 456},
        expected,
    ]
    observed = []

    class Client:
        def hgetall(self, key):
            assert key == "gpu:registry"
            observed.append(records.pop(0))
            return {b"worker": json.dumps(observed[-1]).encode()}

        def close(self):
            observed.append("closed")

    class Child:
        pid = 123

        def poll(self):
            return None

    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: Client())
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    service._wait_native_worker(
        {
            "environment": {"REDIS_URL": "redis://owned"},
            "deployment_id": "ours",
            "bundle_digest": "b" * 64,
            "startup_timeout": 1,
        },
        Child(),
    )
    assert len(observed) == 6
    assert observed[-1] == "closed"


def test_scheduler_startup_waits_for_discovered_worker_metric(monkeypatch):
    import io

    counts = iter([0, 0, 1])
    observed = []

    class Child:
        def poll(self):
            return None

    def metrics(url, **kwargs):
        observed.append(url)
        return io.StringIO(
            "# TYPE physicsnemo_serve_scheduler_discovered_workers gauge\n"
            f"physicsnemo_serve_scheduler_discovered_workers {next(counts)}\n"
        )

    monkeypatch.setattr(service, "urlopen", metrics)
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    service._wait_scheduler({"startup_timeout": 1}, Child(), "18083")
    assert observed == ["http://127.0.0.1:18083/metrics"] * 3


@pytest.mark.parametrize("failed", ["native-worker", "scheduler"])
def test_startup_gate_failure_stops_owned_children_before_http(
    tmp_path, monkeypatch, failed
):
    started = []
    stopped = []

    class Child:
        def __init__(self, name):
            self.name = name

        def poll(self):
            return None

    def start(argv, **kwargs):
        started.append(argv[0])
        return Child(argv[0])

    def gate(plan, child, *args):
        if child.name == failed:
            raise ValueError("startup gate failed")

    monkeypatch.setattr(service.subprocess, "Popen", start)
    monkeypatch.setattr(service, "_wait_native_worker", gate)
    monkeypatch.setattr(service, "_wait_scheduler", gate)
    monkeypatch.setattr(
        service, "_stop", lambda child, timeout: stopped.append(child.name)
    )
    plan = {
        "output": str(tmp_path),
        "processes": [
            {"name": name, "argv": [name], "env": {"HEALTH_PORT": "18083"}}
            for name in ("native-worker", "scheduler", "inference_server")
        ],
    }
    with pytest.raises(ValueError, match="startup gate failed"):
        service._supervise(plan)
    assert "inference_server" not in started
    assert stopped == list(reversed(started))


def test_scheduler_discovery_timeout_rejects_empty_inventory(monkeypatch):
    import io

    times = iter([0, 0.1, 2])

    class Child:
        def poll(self):
            return None

    monkeypatch.setattr(service.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        service,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            "physicsnemo_serve_scheduler_discovered_workers 0\n"
        ),
    )
    with pytest.raises(ValueError, match="Scheduler did not discover"):
        service._wait_scheduler({"startup_timeout": 1}, Child(), "18083")


def test_host_cleanup_on_readiness_failure_only_stops_started_children(
    tmp_path, monkeypatch
):
    events = []

    class Child:
        pid = 123456789

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append("wait")
            return 0

    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: Child())

    def not_ready(*args):
        raise ValueError("readiness failed")

    monkeypatch.setattr(service, "_wait_ready", not_ready)
    plan = {
        "output": str(tmp_path),
        "processes": [{"name": "owned", "argv": ["unused"]}],
        "environment": {},
        "shutdown_timeout": 1,
    }
    with pytest.raises(ValueError, match="readiness failed"):
        service._supervise(plan)
    assert events == ["terminate", "wait"]


def test_image_gpu_selector_does_not_escape_current_allocation(
    native, tmp_path, monkeypatch
):
    project, built = native
    project["gpu"] = "GPU-not-allocated"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated")
    with pytest.raises(ValueError, match="allocat"):
        service.plan_image(project, built, tmp_path / "serve")


def test_image_readiness_failure_removes_only_owned_container(
    native, tmp_path, monkeypatch
):
    import subprocess

    project, built = native
    plan = service.plan_image(project, built, tmp_path / "serve")
    events = []
    container = "d" * 64

    def run(argv, **kwargs):
        events.append(argv)
        if argv[1] == "run":
            Path(argv[argv.index("--cidfile") + 1]).write_text(container)
        return subprocess.CompletedProcess(argv, 0, stdout=container, stderr="")

    def not_ready(*args):
        raise ValueError("readiness failed")

    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setattr(service, "_wait_ready", not_ready)
    with pytest.raises(ValueError, match="readiness failed"):
        service._serve_image(plan)
    cleanup = [argv for argv in events if argv[1] in {"stop", "rm"}]
    assert [argv[1] for argv in cleanup] == ["stop", "rm"]
    assert all(argv[-1] == container for argv in cleanup)
    assert not any(argv[-1] == plan["container_name"] for argv in cleanup)


def test_ready_probe_rejects_other_native_deployment_or_bundle(monkeypatch):
    import io

    plan = {
        "runtime": "cpp",
        "url": "http://127.0.0.1:18082",
        "bundle_digest": "b" * 64,
        "deployment_id": "ours",
    }
    report = {
        "ready": True,
        "native": {
            "ready": True,
            "bundle_digest": "b" * 64,
            "deployment_id": "someone-else",
        },
    }
    monkeypatch.setattr(
        service, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps(report))
    )
    assert service._http_ready(plan) is False
    report["native"]["deployment_id"] = "ours"
    assert service._http_ready(plan) is True
    report["native"]["bundle_digest"] = "other"
    assert service._http_ready(plan) is False


def test_container_interrupt_during_start_still_cleans_cidfile_owner(
    native, tmp_path, monkeypatch
):
    import subprocess

    project, built = native
    plan = service.plan_image(project, built, tmp_path / "serve")
    container = "e" * 64
    events = []

    def run(argv, **kwargs):
        events.append(argv)
        if argv[1] == "run":
            Path(argv[argv.index("--cidfile") + 1]).write_text(container)
            raise KeyboardInterrupt()
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(service.subprocess, "run", run)
    assert service._serve_image(plan)["status"] == "stopped"
    assert any(argv[1] == "stop" and argv[-1] == container for argv in events)


def test_image_host_uses_installed_binaries_and_one_truthful_image_identity(
    native, tmp_path, monkeypatch
):
    from pnms_app import native as native_module

    project, built = native
    image = "sha256:" + "f" * 64
    monkeypatch.setenv("PNMS_IMAGE_CONFIG_DIGEST", image)
    monkeypatch.setenv("PNMS_BUNDLE_DIGEST", "stale-bundle")
    monkeypatch.setenv("RELEASE_IMAGE_DIGEST", "stale-registry-image")
    monkeypatch.setenv("NATIVE_DEPLOYMENT_ID", "outer-owned-container")
    monkeypatch.setenv("PNMS_SERVICE_BIN_DIR", "/opt/pnms/bin")
    monkeypatch.setattr(
        native_module,
        "loader_environment",
        lambda **kwargs: {"LD_LIBRARY_PATH": "/current/torch/lib"},
    )
    plan = service.plan_host(
        project, built, tmp_path / "serve", repository=REPOSITORY, skip_build=True
    )
    assert set(plan["binaries"]) == {
        str(Path("/opt/pnms/bin") / name)
        for name in ("inference_server", "worker-runtime", "native-worker")
    }
    env = service._process_environment(plan["environment"])
    assert env["PNMS_IMAGE_CONFIG_DIGEST"] == image
    assert env["NATIVE_DEPLOYMENT_ID"] == "outer-owned-container"
    assert env["PHYSICSNEMO_SERVE_BIND_HOST"] == "0.0.0.0"
    assert "RELEASE_IMAGE_DIGEST" not in env
    assert "PNMS_BUNDLE_DIGEST" not in env
    assert "/selected/sdk" not in env["LD_LIBRARY_PATH"]


def test_python_cpu_image_launch_never_selects_gpu(native, tmp_path, monkeypatch):
    project, built = native
    project["runtime"] = "python"
    built["python"] = {"plugin_root": "plugin"}

    def unexpected(*args):
        raise AssertionError("Python image must not probe GPUs")

    monkeypatch.setattr(service, "_gpu_binding", unexpected)
    plan = service.plan_image(project, built, tmp_path / "serve")
    assert "--gpus" not in plan["argv"]


@pytest.mark.parametrize("runtime", ["cpp", "python"])
@pytest.mark.parametrize(
    "env_names",
    [(), ("APP_RUNTIME_TOKEN",), ("APP_RUNTIME_TOKEN", "APP_RUNTIME_ENDPOINT")],
)
def test_image_runtime_env_preserves_container_name(
    native, tmp_path, monkeypatch, runtime, env_names
):
    project, built = native
    project["runtime"] = runtime
    built["python"] = {"plugin_root": "plugin"}
    for name in env_names:
        monkeypatch.setenv(name, "application-value")
    output = tmp_path / "serve"
    report = service.serve(
        project,
        built,
        output=output,
        repository=REPOSITORY,
        env_names=env_names,
        dry_run=True,
    )
    record = json.loads((output / "launch.json").read_text())
    argv = report["argv"]
    container_name = argv[argv.index("--name") + 1]
    assert container_name.startswith("pnms-")
    assert report["container_name"] == container_name
    assert record["container_name"] == container_name


@pytest.mark.parametrize("runtime", ["cpp", "python"])
@pytest.mark.parametrize("selected", [False, True])
def test_image_runtime_env_is_opt_in_and_records_names_only(
    native, tmp_path, monkeypatch, runtime, selected
):
    project, built = native
    project["runtime"] = runtime
    built["python"] = {"plugin_root": "plugin"}
    values = {
        "APP_RUNTIME_TOKEN": "credential-fixture-value\nwith $special 'characters'",
        "APP_RUNTIME_ENDPOINT": "endpoint-fixture-value",
        "APP_UNSELECTED_TOKEN": "unselected-fixture-value",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "serve"
    options = {}
    if selected:
        options["env_names"] = [
            "APP_RUNTIME_TOKEN",
            "APP_RUNTIME_ENDPOINT",
            "APP_RUNTIME_TOKEN",
        ]
    report = service.serve(
        project,
        built,
        output=output,
        repository=REPOSITORY,
        dry_run=True,
        **options,
    )
    record = (output / "launch.json").read_text()
    assert json.loads(record)["argv"] == report["argv"]
    assert "environment" not in report
    argv = report["argv"]
    forwarded = [
        argv[index + 1]
        for index, arg in enumerate(argv)
        if arg == "--env" and "=" not in argv[index + 1]
    ]
    assert forwarded == (
        ["APP_RUNTIME_TOKEN", "APP_RUNTIME_ENDPOINT"] if selected else []
    )
    assert "APP_UNSELECTED_TOKEN" not in json.dumps(report) + record
    for value in values.values():
        assert value not in json.dumps(report) + record
        assert value.splitlines()[0] not in json.dumps(report) + record


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("APP_RUNTIME_TOKEN=credential-fixture-value", "names only"),
        ("INVALID-NAME", "names only"),
        ("1INVALID", "names only"),
        ("", "names only"),
        ("APP_MISSING_TOKEN", "not exported"),
    ],
)
def test_image_runtime_env_validation_precedes_workspace_and_gpu_selection(
    native, tmp_path, monkeypatch, selection, expected
):
    project, built = native
    monkeypatch.delenv("APP_MISSING_TOKEN", raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid environment selection must fail before launch setup")

    monkeypatch.setattr(service, "_gpu_binding", unexpected)
    monkeypatch.setattr(service, "_available", unexpected)
    output = tmp_path / "serve"
    with pytest.raises(ValueError, match=expected) as raised:
        service.plan_image(project, built, output, env_names=[selection])
    assert "credential-fixture-value" not in str(raised.value)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "NGC_API_KEY",
        "HF_TOKEN",
        "NVIDIA_API_KEY",
        "RUST_LOG",
        "app_token",
        "_APP_TOKEN",
    ],
)
def test_image_runtime_env_accepts_application_names_and_empty_exported_values(
    native, tmp_path, monkeypatch, name
):
    project, built = native
    monkeypatch.setenv(name, "")
    plan = service.plan_image(project, built, tmp_path / "serve", env_names=[name])
    index = plan["argv"].index(name)
    assert plan["argv"][index - 1] == "--env"
    assert plan["argv"][-1] == built["image"]["image_id"]


@pytest.mark.parametrize(
    "name",
    [
        "SOURCE_SHA",
        "PNMS_SOURCE_SHA",
        "PNMS_BUNDLE_DIGEST",
        "PNMS_IMAGE_CONFIG_DIGEST",
        "RELEASE_IMAGE_DIGEST",
        "NATIVE_DEPLOYMENT_ID",
        "NATIVE_WORKER_SLOT",
        "PORT",
        "SERVER_PORT",
        "HEALTH_STUB_ENABLED",
        "HEALTH_PORT",
        "HEALTH_STUB_PORT",
        "SCHEDULER_METRICS_PORT",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PNMS_SERVICE_BIN_DIR",
        "PNMS_OUTPUT_ROOT",
        "PHYSICSNEMO_SERVE_BIND_HOST",
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
        "path",
        "pnms_source_sha",
        "native_worker_slot",
        "physicsnemo_serve_bind_host",
    ],
)
def test_image_runtime_env_cannot_override_launcher_owned_settings(
    native, tmp_path, monkeypatch, name
):
    project, built = native
    monkeypatch.setenv(name, "credential-fixture-value")
    output = tmp_path / "serve"
    with pytest.raises(ValueError, match="launcher-owned") as raised:
        service.plan_image(project, built, output, env_names=[name])
    assert "credential-fixture-value" not in str(raised.value)
    assert not output.exists()


def test_image_runtime_env_rejects_dev_service_before_host_setup(
    native, tmp_path, monkeypatch
):
    project, built = native
    monkeypatch.setenv("APP_RUNTIME_TOKEN", "credential-fixture-value")

    def unexpected(*args, **kwargs):
        pytest.fail("Development environment selection must fail before host setup")

    monkeypatch.setattr(service, "plan_host", unexpected)
    output = tmp_path / "serve"
    with pytest.raises(ValueError, match="--dev"):
        service.serve(
            project,
            built,
            output=output,
            repository=REPOSITORY,
            dev=True,
            env_names=["APP_RUNTIME_TOKEN"],
        )
    assert not output.exists()


def test_image_runtime_env_values_remain_inherited_by_docker_only(
    native, tmp_path, monkeypatch
):
    project, built = native
    token = "credential-fixture-value"
    monkeypatch.setenv("APP_RUNTIME_TOKEN", token)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert token not in repr(argv) + repr(kwargs)
        assert "env" not in kwargs
        if argv[1] == "run":
            assert os.environ["APP_RUNTIME_TOKEN"] == token
            assert argv[argv.index("APP_RUNTIME_TOKEN") - 1] == "--env"
            Path(argv[argv.index("--cidfile") + 1]).write_text("e" * 64)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setattr(service, "_wait_ready", interrupt)
    output = tmp_path / "serve"
    report = service.serve(
        project,
        built,
        output=output,
        repository=REPOSITORY,
        env_names=["APP_RUNTIME_TOKEN"],
    )
    assert report["status"] == "stopped"
    assert [argv[1] for argv in calls] == ["run", "stop", "logs", "rm"]
    assert token not in (output / "launch.json").read_text() + json.dumps(report)


def test_python_gpu_image_uses_declared_resources_and_current_allocation(
    native, tmp_path, monkeypatch
):
    project, built = native
    project["runtime"] = "python"
    built["python"] = {
        "plugin_root": "plugin",
        "resources": {"defaults": {"gpus_required": 1, "device_kind": "gpu"}},
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-current")
    plan = service.plan_image(project, built, tmp_path / "serve")
    assert "--gpus" in plan["argv"]
    assert plan["argv"][plan["argv"].index("--gpus") + 1] == "device=GPU-current"


def test_host_cleanup_continues_after_one_child_disappears(tmp_path, monkeypatch):
    stopped = []

    class Child:
        def __init__(self, name):
            self.name = name

        def poll(self):
            return None

        def terminate(self):
            stopped.append(self.name)
            if self.name == "second":
                raise ProcessLookupError("already gone")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        service.subprocess, "Popen", lambda argv, **kwargs: Child(argv[0])
    )

    def not_ready(*args):
        raise ValueError("readiness failed")

    monkeypatch.setattr(service, "_wait_ready", not_ready)
    plan = {
        "output": str(tmp_path),
        "processes": [{"name": name, "argv": [name]} for name in ("first", "second")],
        "environment": {},
        "shutdown_timeout": 1,
    }
    with pytest.raises(ValueError, match="readiness failed"):
        service._supervise(plan)
    assert stopped == ["second", "first"]


def test_image_entrypoint_reuses_sealed_bytes_and_current_python(
    native, tmp_path, monkeypatch
):
    from pnms_app import image_entrypoint

    project, built = native
    location = Path(built["root"]) / built["bundle"]
    descriptor = json.loads((location / "bundle.json").read_text())
    descriptor["source_sha"] = built["source_sha"]
    (location / "bundle.json").write_text(json.dumps(descriptor))
    monkeypatch.setattr(
        bundle,
        "verify",
        lambda *args, **kwargs: {"bundle_digest": built["bundle_digest"]},
    )
    monkeypatch.setenv("PORT", "18083")
    captured = {}

    def serve(project, built, **options):
        captured.update(project=project, built=built, options=options)
        return {"status": "stopped"}

    monkeypatch.setattr(service, "serve", serve)
    assert (
        image_entrypoint.launch(
            bundle_root=location, artifact_root=tmp_path / "outputs"
        )["status"]
        == "stopped"
    )
    assert captured["options"]["dev"] is True
    assert captured["options"]["skip_build"] is True
    assert captured["options"]["port"] == 18083
    assert captured["built"]["environment"]["python"] == sys.executable
    assert captured["project"]["serve"]["artifact_root"] == str(tmp_path / "outputs")
    assert Path(captured["built"]["root"]) / captured["built"]["bundle"] == location


def test_service_bridge_uses_recorded_cpp_abi(native, tmp_path):
    project, built = native
    location = Path(built["root"]) / built["bundle"]
    descriptor = json.loads((location / "bundle.json").read_text())
    descriptor["environment"]["stack"] = {"cxx11_abi": False}
    (location / "bundle.json").write_text(json.dumps(descriptor))
    plan = service.plan_host(project, built, tmp_path / "serve", repository=REPOSITORY)
    assert "-D_GLIBCXX_USE_CXX11_ABI=0" in plan["build_environment"].get("CXXFLAGS", "")


def test_image_loader_keeps_selected_runtime_stack_native_dependencies(
    native, tmp_path, monkeypatch
):
    from pnms_app import native as native_module

    project, built = native
    monkeypatch.setenv("PNMS_SERVICE_BIN_DIR", "/opt/pnms/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/selected/image/hpc/lib")
    monkeypatch.setenv("PNMS_NATIVE_LIBRARY_DIRS", "/selected/image/operators/lib")
    monkeypatch.setenv("TENSORRT_ROOT", "/selected/image/TensorRT")
    captured = {}

    def loader(**options):
        captured.update(options)
        return {}

    monkeypatch.setattr(native_module, "loader_environment", loader)
    service.plan_host(
        project, built, tmp_path / "serve", repository=REPOSITORY, skip_build=True
    )
    assert Path("/selected/image/hpc/lib") in captured["extra"]
    assert Path("/selected/image/operators/lib") in captured["extra"]
    assert Path("/selected/image/TensorRT/lib") in captured["extra"]


def test_image_launch_rejects_untyped_legacy_image_identity(native, tmp_path):
    project, built = native
    del built["image"]["image_config_digest"]
    with pytest.raises(ValueError, match="configuration digest"):
        service.plan_image(project, built, tmp_path / "serve")


@pytest.fixture
def supervised_native(tmp_path, monkeypatch):
    from types import SimpleNamespace

    state = SimpleNamespace(
        started=[], stopped=[], retired=[], gates=[], ready=[], sleep=None
    )

    class Child:
        def __init__(self, name, environment):
            self.name = name
            self.pid = 100 + len(state.started)
            self.environment = environment
            self.code = None

        def poll(self):
            return self.code

    def start(argv, **options):
        child = Child(argv[0], options["env"])
        state.started.append(child)
        return child

    def retire(plan, child):
        assert child.poll() is not None
        state.retired.append(child)

    def worker_gate(plan, child, *companions):
        if len(state.workers()) > 1:
            assert state.retired[-1] is state.workers()[-2]
            assert all(peer.poll() is None for peer in state.peers())
        state.gates.append(child)

    def ready(plan, processes):
        state.ready.append([child.pid for _, child in processes])

    def sleep(seconds):
        if state.sleep is not None:
            state.sleep(seconds)
        elif len(state.workers()) == 1:
            state.workers()[0].code = -9
        else:
            raise KeyboardInterrupt

    state.workers = lambda: [
        child for child in state.started if child.name == "native-worker"
    ]
    state.peers = lambda: [
        child for child in state.started if child.name != "native-worker"
    ]
    identity = {"NATIVE_DEPLOYMENT_ID": "owned-deployment", "NATIVE_WORKER_SLOT": "0"}
    state.plan = {
        "runtime": "cpp",
        "output": str(tmp_path),
        "url": "http://owned",
        "deployment_id": "owned-deployment",
        "bundle_digest": "b" * 64,
        "environment": {**identity, "REDIS_URL": "redis://owned"},
        "processes": [
            {"name": name, "argv": [name], "env": {**identity, "HEALTH_PORT": "18083"}}
            for name in (
                "redis",
                "native-worker",
                "scheduler",
                "results",
                "inference_server",
            )
        ],
        "shutdown_timeout": 1,
    }
    monkeypatch.setattr(service.subprocess, "Popen", start)
    monkeypatch.setattr(
        service, "_stop", lambda child, timeout: state.stopped.append(child)
    )
    state.retire_worker = service._retire_native_worker
    monkeypatch.setattr(service, "_retire_native_worker", retire, raising=False)
    monkeypatch.setattr(service, "_wait_redis", lambda *args: None)
    monkeypatch.setattr(service, "_wait_native_worker", worker_gate)
    monkeypatch.setattr(service, "_wait_scheduler", lambda *args: None)
    monkeypatch.setattr(service, "_wait_ready", ready)
    monkeypatch.setattr(service.time, "sleep", sleep)
    return state


def test_native_worker_death_restarts_only_worker_with_stable_owned_identity(
    supervised_native,
):
    state = supervised_native
    result = service._supervise(state.plan)
    assert result["status"] == "stopped"
    assert len(state.workers()) == 2
    original, replacement = state.workers()
    assert replacement.pid != original.pid
    assert replacement.environment["NATIVE_DEPLOYMENT_ID"] == "owned-deployment"
    assert replacement.environment["NATIVE_WORKER_SLOT"] == "0"
    assert len(state.peers()) == 4, (
        "Redis/API/scheduler/results must survive worker death"
    )
    assert all(child.poll() is None for child in state.peers())
    assert state.retired == [original]
    assert state.gates == [original, replacement]
    assert len(state.ready) == 2, (
        "replacement must pass native and HTTP readiness again"
    )
    assert {child.pid for child in state.stopped} == {
        child.pid for child in state.started
    }


def test_repeated_native_crashes_exhaust_a_bounded_restart_budget(supervised_native):
    state = supervised_native
    state.sleep = lambda seconds: setattr(state.workers()[-1], "code", -9)
    with pytest.raises(ValueError, match="restart budget"):
        service._supervise(state.plan)
    assert len(state.workers()) == 4, "one original plus at most three replacements"
    assert len(state.peers()) == 4


@pytest.mark.parametrize("name", ["redis", "scheduler", "results", "inference_server"])
def test_essential_peer_exit_fails_closed_without_worker_respawn(
    supervised_native, name
):
    state = supervised_native
    state.sleep = lambda seconds: setattr(
        next(child for child in state.peers() if child.name == name), "code", 1
    )
    with pytest.raises(ValueError, match=name):
        service._supervise(state.plan)
    assert len(state.workers()) == 1 and state.retired == []


def test_replacement_startup_failure_fails_closed_without_another_retry(
    supervised_native, monkeypatch
):
    state = supervised_native

    def gate(plan, child, *companions):
        if len(state.workers()) > 1:
            raise ValueError("replacement startup failed")

    monkeypatch.setattr(service, "_wait_native_worker", gate)
    with pytest.raises(ValueError, match="replacement startup failed"):
        service._supervise(state.plan)
    assert len(state.workers()) == 2 and len(state.peers()) == 4
    assert {child.pid for child in state.stopped} == {
        child.pid for child in state.started
    }


def test_shutdown_during_restart_backoff_never_respawns(supervised_native):
    state = supervised_native

    def sleep(seconds):
        if state.workers()[0].poll() is None:
            state.workers()[0].code = -9
        else:
            raise KeyboardInterrupt

    state.sleep = sleep
    assert service._supervise(state.plan)["status"] == "stopped"
    assert len(state.workers()) == 1
    assert len(state.ready) == 1


@pytest.fixture
def retirement_registry(monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import redis

    field = b"native:execute.native.demo.ours.0.0"
    record = {
        "deployment_id": "owned-deployment",
        "worker_slot": "0",
        "bundle_digest": "b" * 64,
        "workflow_id": "demo",
        "executor_class": "native.demo",
        "stream": "execute.native.demo.ours.0",
        "pid": 123,
        "worker_instance_id": "old-instance",
        "status": "available",
        "model_cache": {
            "schema_version": 1,
            "scope": "process",
            "entries": [{"workflow_id": "demo", "status": "loaded"}],
            "total_entries": 1,
            "warmup": {"status": "succeeded", "self_test": {"status": "passed"}},
        },
    }
    lease = "native:worker-lease:" + hashlib.sha256(field).hexdigest()
    state = SimpleNamespace(
        field=field,
        record=record,
        lease=lease,
        values={field: json.dumps(record).encode(), b"unrelated": b"{}"},
        leases={lease: "old-instance"},
        streams={record["stream"]: [b"queued request"]},
        before_eval=None,
        evaluations=0,
        closed=0,
        plan={
            "environment": {"REDIS_URL": "redis://owned", "NATIVE_WORKER_SLOT": "0"},
            "deployment_id": "owned-deployment",
            "bundle_digest": "b" * 64,
        },
    )

    class Client:
        def hgetall(self, name):
            assert name == "gpu:registry"
            return state.values.copy()

        def eval(self, script, count, *arguments):
            assert count == 2
            assert arguments[:3] == ("gpu:registry", lease, field)
            state.evaluations += 1
            if state.before_eval is not None:
                state.before_eval()
            _, _, _, expected, owner, *replacement = arguments
            current_owner = state.leases.get(lease)
            if state.values.get(field) != expected or current_owner not in (
                None,
                owner,
            ):
                return 0
            # Model the registry mutation at the Redis boundary, including the
            # old deletion behavior so the retained-slot regression detects it.
            if "'HDEL'" in script:
                del state.values[field]
            else:
                assert "redis.call('HSET',KEYS[1],ARGV[1],ARGV[4])" in script
                (value,) = replacement
                state.values[field] = value.encode()
            if current_owner == owner:
                del state.leases[lease]
            return 1

        def close(self):
            state.closed += 1

    class Child:
        pid = 123

        def poll(self):
            return -9

    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: Client())
    state.child = Child()
    return state


@pytest.mark.parametrize("lease_present", [True, False])
@pytest.mark.parametrize("already_warming", [True, False])
def test_worker_retirement_retains_exact_slot_without_stale_readiness(
    retirement_registry, lease_present, already_warming
):
    state = retirement_registry
    if not lease_present:
        state.leases.clear()
    if already_warming:
        state.record["status"] = "warming"
        state.record["model_cache"]["warmup"]["reason"] = "restart_pending"
        state.values[state.field] = json.dumps(state.record).encode()
    service._retire_native_worker(state.plan, state.child)
    retired = json.loads(state.values[state.field])
    assert retired == {
        **state.record,
        "status": "warming",
        "model_cache": {
            "schema_version": 1,
            "scope": "process",
            "entries": [],
            "total_entries": 0,
            "warmup": {
                "workflow_id": "demo",
                "status": "warming",
                "reason": "restart_pending",
            },
        },
    }
    assert state.values[b"unrelated"] == b"{}"
    assert not state.leases
    assert state.streams == {state.record["stream"]: [b"queued request"]}
    assert state.evaluations == state.closed == 1


def test_durable_worker_slot_remains_warming_through_every_supervisor_backoff(
    supervised_native, retirement_registry, monkeypatch
):
    state, registry = supervised_native, retirement_registry
    backoffs = []

    def gate(plan, child, *companions):
        registry.record["pid"] = child.pid
        registry.record["worker_instance_id"] = str(child.pid)
        registry.values[registry.field] = json.dumps(registry.record).encode()
        registry.leases[registry.lease] = str(child.pid)

    def sleep(seconds):
        worker = state.workers()[-1]
        if worker.poll() is None:
            if len(state.workers()) == 4:
                raise KeyboardInterrupt
            worker.code = -9
            return
        retired = json.loads(registry.values[registry.field])
        assert retired["status"] == "warming"
        assert retired["model_cache"]["entries"] == []
        assert retired["model_cache"]["warmup"]["reason"] == "restart_pending"
        assert retired["stream"] == registry.record["stream"]
        assert retired["worker_instance_id"] == str(worker.pid)
        assert registry.lease not in registry.leases
        assert registry.streams == {registry.record["stream"]: [b"queued request"]}
        backoffs.append(seconds)

    monkeypatch.setattr(service, "_retire_native_worker", state.retire_worker)
    monkeypatch.setattr(service, "_wait_native_worker", gate)
    state.sleep = sleep
    assert service._supervise(state.plan)["status"] == "stopped"
    assert backoffs == [0.25, 0.5, 1.0]
    assert len(state.workers()) == 4 and len(state.peers()) == 4


@pytest.mark.parametrize("changed", ["pid", "bundle_digest", "row-race", "owner-race"])
def test_worker_retirement_refuses_foreign_or_changed_registry(
    retirement_registry, changed
):
    state = retirement_registry
    if changed in ("pid", "bundle_digest"):
        state.record[changed] = 456 if changed == "pid" else "another"
        state.values[state.field] = json.dumps(state.record).encode()
    elif changed == "row-race":
        replacement = {**state.record, "worker_instance_id": "new-instance", "pid": 456}
        state.before_eval = lambda: state.values.update(
            {state.field: json.dumps(replacement).encode()}
        )
    else:
        state.before_eval = lambda: state.leases.update({state.lease: "new-instance"})
    with pytest.raises(ValueError, match="different process|ownership changed"):
        service._retire_native_worker(state.plan, state.child)
    retained = json.loads(state.values[state.field])
    assert retained["status"] == "available"
    assert retained["model_cache"]["total_entries"] == 1
    if changed == "row-race":
        assert (
            retained["worker_instance_id"] == "new-instance" and retained["pid"] == 456
        )
    assert state.leases[state.lease] == (
        "new-instance" if changed == "owner-race" else "old-instance"
    )
    assert state.closed == 1
    assert state.evaluations == (changed in ("row-race", "owner-race"))


def test_replacement_warmup_fails_if_an_essential_peer_dies(monkeypatch):
    import redis

    class Client:
        def hgetall(self, name):
            raise AssertionError(
                "must detect failed Redis/API peer before registry reads"
            )

        def close(self):
            pass

    class Worker:
        pid = 123

        def poll(self):
            return None

    class Peer:
        def poll(self):
            return 1

    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: Client())
    with pytest.raises(ValueError, match="inference_server"):
        service._wait_native_worker(
            {"environment": {"REDIS_URL": "redis://owned"}},
            Worker(),
            [("inference_server", Peer())],
        )
