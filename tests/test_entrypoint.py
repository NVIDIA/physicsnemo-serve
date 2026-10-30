from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint.sh"


def _entrypoint_runtime_config_env(overrides: dict[str, str]) -> tuple[str, str]:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    configuration = source.split("# Helper Functions", 1)[0]
    script = configuration + (
        '\nprintf "%s\\n%s\\n" "$WORKER_RUNTIME_CONFIG" '
        '"$PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"\n'
    )
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, **overrides},
    )
    worker_config, server_config = result.stdout.splitlines()
    return worker_config, server_config


def test_entrypoint_worker_runtime_override_replaces_baked_in_server_default():
    worker_config, server_config = _entrypoint_runtime_config_env(
        {
            "WORKER_RUNTIME_CONFIG": "/outputs/custom/worker-runtime.json",
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG": (
                "/app/scripts/worker_runtime_config.json"
            ),
        }
    )

    assert worker_config == "/outputs/custom/worker-runtime.json"
    assert server_config == worker_config


def _generate_execute_config(tmp_path: Path, native: bool, *, binary: bool = True):
    launcher = tmp_path / "runtime_env_launcher.py"
    launcher.write_text(
        "raise RuntimeError('must not execute during config generation')\n"
    )
    native_binary = tmp_path / "native-worker"
    if binary:
        native_binary.write_text("#!/bin/bash\nexit 0\n")
        native_binary.chmod(0o755)
    deployment = tmp_path / "deployment.json"
    deployment.write_text('{"drain_timeout_seconds": 7}\n')
    config_path = tmp_path / "supervisord.conf"
    source = ENTRYPOINT.read_text(encoding="utf-8").rsplit('\nmain "$@"', 1)[0]
    result = subprocess.run(
        ["bash"],
        input=source + '\nSUPERVISORD_CONF="$TEST_CONF"\ngenerate_supervisord_config\n',
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "WORKERS": "execute",
            "TEST_CONF": str(config_path),
            "RUNTIME_ENV_LAUNCHER_SCRIPT": str(launcher),
            "NATIVE_WORKER_BIN": str(native_binary),
            "WORKERS_PER_GPU": "1",
            "NATIVE_DEPLOYMENT_CONFIG": str(deployment) if native else "",
        },
    )
    return result, config_path.read_text() if config_path.exists() else ""


def test_entrypoint_native_deployment_supervises_rust_worker_directly(tmp_path):
    result, config = _generate_execute_config(tmp_path, native=True)
    assert result.returncode == 0, result.stderr
    assert "[program:native-worker]" in config
    assert "runtime-env-launcher" not in config
    assert "inference_worker.py" not in config
    wrapper = Path("/tmp/native-worker-wrapper.sh").read_text()
    assert 'exec "' + str(tmp_path / "native-worker") + '" "' in wrapper
    # Allow the request drain, completion retries, and final native destructor.
    assert "stopwaitsecs=104\n" in config


def test_entrypoint_native_deployment_fails_closed_when_binary_missing(tmp_path):
    result, config = _generate_execute_config(tmp_path, native=True, binary=False)
    assert result.returncode != 0, "native deployment must not fall back to Python"
    assert "runtime-env-launcher" not in config


def test_entrypoint_python_deployment_preserves_existing_launcher(tmp_path):
    result, config = _generate_execute_config(tmp_path, native=False)
    assert result.returncode == 0, result.stderr
    assert "[program:runtime-env-launcher]" in config
    assert "[program:native-worker]" not in config
    assert 'WORKER_SCRIPT="/app/scripts/inference_worker.py"' in config


def test_python_image_cargo_planner_can_parse_native_workspace_member(tmp_path):
    import re
    import shutil

    dockerfile = (
        REPO_ROOT / "Dockerfile.physicsnemo-serve.scicomp-rust-slim"
    ).read_text()
    planner = dockerfile.split("FROM chef AS planner", 1)[1].split(
        "FROM chef AS builder", 1
    )[0]
    for line in planner.splitlines():
        if line.startswith("COPY "):
            *sources, destination = line.split()[1:]
            target = tmp_path / destination
            target.mkdir(parents=True, exist_ok=True)
            for source in sources:
                shutil.copy2(REPO_ROOT / source, target / Path(source).name)
    stub = re.search(r"RUN mkdir .*?(?=\n\n)", planner, re.DOTALL)
    assert stub
    subprocess.run(["bash", "-ec", stub.group()[4:]], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--offline", "--format-version=1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_entrypoint_native_redis_retains_completion_markers_without_changing_python_policy(
    tmp_path,
):
    source = ENTRYPOINT.read_text().rsplit('\nmain "$@"', 1)[0]
    script = (
        source
        + """
redis-server() { printf '%s\\n' "$@" > "$REDIS_ARGUMENTS"; }
redis-cli() { echo PONG; }
start_redis
"""
    )
    for deployment, expected_policy in [
        ("", "allkeys-lru"),
        ("/opt/native/deployment.json", "noeviction"),
    ]:
        args_path = tmp_path / "redis-arguments"
        subprocess.run(
            ["bash"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
            env={
                **os.environ,
                "NATIVE_DEPLOYMENT_CONFIG": deployment,
                "REDIS_ARGUMENTS": str(args_path),
            },
        )
        arguments = args_path.read_text().splitlines()
        assert arguments[arguments.index("--maxmemory-policy") + 1] == expected_policy


def _shutdown_events(tmp_path: Path, *, native: bool) -> list[str]:
    source = ENTRYPOINT.read_text().rsplit('\nmain "$@"', 1)[0]
    pid_path = tmp_path / "supervisord.pid"
    pid_path.write_text("4242\n")
    source = source.replace("/tmp/supervisord.pid", str(pid_path))
    event_path = tmp_path / "shutdown-events"
    script = (
        source
        + """
SUPERVISORD_PID=4242
fake_ticks=0
kill() {
    if [[ "$1" == "-0" ]]; then
        [[ "$fake_ticks" -lt 65 ]]
    else
        printf 'supervisor-term\\n' >> "$SHUTDOWN_EVENTS"
    fi
}
sleep() { fake_ticks=$((fake_ticks + 1)); }
wait() {
    fake_ticks=65
    printf 'supervisor-reaped\\n' >> "$SHUTDOWN_EVENTS"
}
redis-cli() {
    printf 'redis-stop:%s\\n' "$fake_ticks" >> "$SHUTDOWN_EVENTS"
}
shutdown_handler
"""
    )
    subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
        env={
            **os.environ,
            "SHUTDOWN_EVENTS": str(event_path),
            "NATIVE_DEPLOYMENT_CONFIG": "/opt/native/deployment.json" if native else "",
        },
    )
    return event_path.read_text().splitlines()


def test_native_shutdown_reaps_supervisor_before_stopping_redis(tmp_path):
    # A supervisor taking 65 ticks must outlive the previous shell's 60s limit.
    assert _shutdown_events(tmp_path, native=True) == [
        "supervisor-term",
        "supervisor-reaped",
        "redis-stop:65",
    ]


def test_python_shutdown_preserves_existing_sixty_second_wait(tmp_path):
    assert _shutdown_events(tmp_path, native=False) == [
        "supervisor-term",
        "redis-stop:60",
    ]


def test_native_shutdown_reaps_an_actual_child_without_hanging(tmp_path):
    source = ENTRYPOINT.read_text().rsplit('\nmain "$@"', 1)[0]
    ready = tmp_path / "ready"
    events = tmp_path / "events"
    child = """
import os
import signal
from pathlib import Path

def stop(*_):
    Path(os.environ["SHUTDOWN_EVENTS"]).write_text("supervisor-exit\\n")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.alarm(2)
Path(os.environ["SUPERVISOR_READY"]).touch()
signal.pause()
"""
    script = (
        source
        + f"""
{shlex.quote(sys.executable)} -c {shlex.quote(child)} &
SUPERVISORD_PID=$!
while [[ ! -f "$SUPERVISOR_READY" ]]; do sleep 0.01; done
redis-cli() {{
    if kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
        printf 'redis-before-reap\\n' >> "$SHUTDOWN_EVENTS"
    else
        printf 'redis-after-reap\\n' >> "$SHUTDOWN_EVENTS"
    fi
}}
shutdown_handler
"""
    )
    subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
        env={
            **os.environ,
            "NATIVE_DEPLOYMENT_CONFIG": "/opt/native/deployment.json",
            "SUPERVISOR_READY": str(ready),
            "SHUTDOWN_EVENTS": str(events),
        },
    )
    assert events.read_text().splitlines() == ["supervisor-exit", "redis-after-reap"]
