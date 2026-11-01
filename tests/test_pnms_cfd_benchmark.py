"""CPU contract checks; these do not substitute for Lepton timing evidence."""

from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from qa.native_inference.cfd_contract import FieldContractError, METRIC_OUTPUTS
from qa.scripts import benchmark_pnms_cfd as benchmark


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("python", "aoti", "tensorrt", "tensorrt", "aoti", "python")


@pytest.fixture
def service_tree(tmp_path):
    """Real CLI -> supervisor -> listener trees, with bounded failure cleanup."""
    processes, streams, descendants = [], [], []

    def start(
        *,
        detached=False,
        detached_supervisor=False,
        scrub_environment=False,
        double_fork=False,
        stubborn=False,
        exit_leader=False,
        exit_code=0,
        graceful_leader=False,
        runtime="python",
    ):
        output = tmp_path / f"service-{len(processes)}"
        output.mkdir()
        ready = output / "listener.json"
        child_pid = output / "child.pid"
        leader_pid = output / "leader.pid"
        listener = f"""
import json, os, signal, socket, time
from pathlib import Path
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen()
if {stubborn!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready = Path({str(ready)!r})
temporary = ready.with_suffix('.tmp')
temporary.write_text(json.dumps(dict(pid=os.getpid(), port=listener.getsockname()[1])))
temporary.replace(ready)
while True:
    time.sleep(0.1)
"""
        child = f"""
import os, signal, subprocess, sys, time
from pathlib import Path
if {double_fork!r}:
    if os.fork():
        os._exit(0)
    os.setsid()
Path({str(child_pid)!r}).write_text(str(os.getpid()))
listener = subprocess.Popen([sys.executable, '-c', {listener!r}], start_new_session={detached!r})
def stop(signum, frame):
    listener.terminate()
    listener.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.1)
"""
        leader = f"""
import os, signal, subprocess, sys, time
__import__('pathlib').Path({str(leader_pid)!r}).write_text(str(os.getpid()))
child = subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session={detached_supervisor!r}, env={{}} if {scrub_environment!r} else None)
if {graceful_leader!r}:
    def stop(signum, frame):
        time.sleep(0.15)
        if child.poll() is not None:
            sys.exit(2)
        child.terminate()
        child.wait(timeout=2)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
while not __import__('pathlib').Path({str(ready)!r}).exists():
    time.sleep(0.01)
if not {exit_leader!r}:
    while True:
        time.sleep(0.1)
sys.exit({exit_code!r})
"""
        args = SimpleNamespace(
            runtime=runtime,
            expected_image="registry/image@sha256:" + "f" * 64,
            output=output,
            service_cwd=ROOT,
        )
        process, stream = benchmark.start_service(
            args,
            {
                "command": [sys.executable, "-c", leader],
                "artifact_root": output / "artifacts",
                "workflow": benchmark.PYTHON_WORKFLOW,
            },
        )
        processes.append(process)
        streams.append(stream)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                assert process.poll() is None, (output / "service.log").read_text()
                time.sleep(0.01)
            assert ready.exists(), (output / "service.log").read_text()
            data = json.loads(ready.read_text())
            pids = [int(child_pid.read_text()), data["pid"]]
            descendants.extend((pid, benchmark._process_token(pid)) for pid in pids)
            if exit_leader:
                assert process.wait(timeout=5) == exit_code
            return SimpleNamespace(
                process=process,
                pids=pids,
                port=data["port"],
                launcher_pid=int(leader_pid.read_text()),
            )
        except BaseException as error:
            try:
                benchmark.stop_service(process, 2)
            except FieldContractError as cleanup_error:
                if str(cleanup_error) != "shipped service required forced shutdown":
                    raise error from cleanup_error
            raise

    yield start
    for process in processes:
        guard = getattr(process, "_pnms_service_guard", None)
        try:
            if guard is None or guard.poll() is None:
                try:
                    benchmark.stop_service(process, 0.2)
                except FieldContractError as error:
                    if str(error) != "shipped service required forced shutdown":
                        raise
        finally:
            if guard is not None:
                if guard.poll() is None:
                    guard.terminate()
                guard.wait(timeout=20)
    for pid, token in reversed(descendants):
        if token is None or benchmark._process_token(pid) != token:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        if hasattr(process, "_pnms_service_guard"):
            continue
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    for stream in streams:
        stream.close()


def assert_listener_retired(tree):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", tree.port))
    for pid in tree.pids:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "state="], capture_output=True, text=True
        )
        assert not result.stdout.strip() or result.stdout.lstrip().startswith("Z"), (
            f"owned service descendant {pid} survived shutdown: {result.stdout}"
        )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process ownership")
def test_service_tree_startup_failure_retires_owned_descendants(
    service_tree, tmp_path, monkeypatch
):
    original_start, original_loads = benchmark.start_service, json.loads
    identities, processes, observed = {}, [], {}
    failure = json.JSONDecodeError("forced readiness failure", "", 0)

    def record_start(*args, **kwargs):
        process, stream = original_start(*args, **kwargs)
        processes.append(process)
        identities[process.pid] = benchmark._process_token(process.pid)
        return process, stream

    def fail_readiness(value, *args, **kwargs):
        data = original_loads(value, *args, **kwargs)
        if isinstance(data, dict) and set(data) == {"pid", "port"}:
            pids = [int((tmp_path / "service-0/child.pid").read_text()), data["pid"]]
            observed.update(pids=pids, port=data["port"])
            identities.update((pid, benchmark._process_token(pid)) for pid in pids)
            assert all(identities.values()), "the owned tree must exist before failure"
            with socket.create_connection(("127.0.0.1", data["port"]), timeout=2):
                pass
            raise failure
        return data

    monkeypatch.setattr(benchmark, "start_service", record_start)
    monkeypatch.setattr(json, "loads", fail_readiness)
    try:
        with pytest.raises(json.JSONDecodeError) as caught:
            service_tree(detached=True, stubborn=True)
        assert caught.value is failure, "cleanup must preserve the original setup error"
        assert_listener_retired(SimpleNamespace(**observed))
        assert processes[0].poll() is not None
    finally:
        for pid, token in reversed(identities.items()):
            if token is not None and benchmark._process_token(pid) == token:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            process.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process ownership")
def test_service_identity_setup_failure_retires_already_launched_tree(
    service_tree, tmp_path, monkeypatch
):
    observed, identities, processes = {}, {}, []
    failure = RuntimeError("process identity setup failed")
    original_token = benchmark._process_token

    def fail_identity(pid, *args):
        monkeypatch.setattr(benchmark, "_process_token", original_token)
        identities[pid] = original_token(pid)
        ready = tmp_path / "service-0/listener.json"
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        data = json.loads(ready.read_text())
        pids = [int((ready.parent / "child.pid").read_text()), data["pid"]]
        identities.update((pid, original_token(pid)) for pid in pids)
        observed.update(pids=pids, port=data["port"])
        raise failure

    original_popen = benchmark.subprocess.Popen

    def record_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        if kwargs.get("start_new_session"):
            processes.append(process)
        return process

    monkeypatch.setattr(benchmark, "_process_token", fail_identity)
    monkeypatch.setattr(benchmark.subprocess, "Popen", record_process)
    try:
        with pytest.raises(RuntimeError) as caught:
            service_tree(detached=True, detached_supervisor=True)
        assert caught.value is failure
        assert processes[0].poll() is not None
        assert_listener_retired(SimpleNamespace(**observed))
    finally:
        for pid, token in reversed(identities.items()):
            if token is not None and benchmark._process_token(pid) == token:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            process.wait(timeout=5)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_shutdown_rejects_invalid_timeout_before_signalling_process(timeout):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        with pytest.raises(FieldContractError, match="finite and positive"):
            benchmark.stop_service(process, timeout)
        assert process.poll() is None, "invalid shutdown arguments must not signal work"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_linux_service_rejects_invalid_guard_deadline_before_allocating_ownership(
    timeout, monkeypatch
):
    def unexpected_resource(*args, **kwargs):
        raise AssertionError("invalid guard deadline must not allocate or spawn")

    monkeypatch.setattr(benchmark.socket, "socketpair", unexpected_resource)
    monkeypatch.setattr(benchmark.subprocess, "Popen", unexpected_resource)
    with pytest.raises(FieldContractError, match="finite and positive"):
        benchmark._start_linux_service([], None, {}, None, timeout)


@pytest.mark.parametrize("runtime", ["python", "aoti"])
@pytest.mark.parametrize("detached", [False, True])
def test_shutdown_retires_real_service_descendants_and_releases_listener(
    service_tree, runtime, detached
):
    tree = service_tree(runtime=runtime, detached=detached)
    benchmark.stop_service(tree.process, 2)
    assert tree.process.poll() is not None
    assert_listener_retired(tree)


def test_shutdown_retires_owned_group_after_cli_leader_exits(service_tree):
    tree = service_tree(exit_leader=True)
    benchmark.stop_service(tree.process, 2)
    assert_listener_retired(tree)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process ownership")
@pytest.mark.parametrize("ready_before_failure", [False, True])
def test_shutdown_retires_detached_tree_after_launcher_dies_before_cleanup(
    service_tree, ready_before_failure
):
    tree = service_tree(detached=True, detached_supervisor=True)
    if ready_before_failure:

        def readiness(path):
            assert path == f"/v1/infer/{benchmark.PYTHON_WORKFLOW}/readiness"
            with socket.create_connection(("127.0.0.1", tree.port), timeout=2):
                pass
            return {"readiness": {"ready": True}}

        benchmark.wait_ready(
            SimpleNamespace(startup_timeout=2),
            {"workflow": benchmark.PYTHON_WORKFLOW},
            SimpleNamespace(json=readiness),
            tree.process,
        )
    # Its children have their own sessions, so the launcher's process group
    # cannot find them. No shutdown observation happens before the launcher dies.
    tree.process.kill()
    tree.process.wait(timeout=5)
    rows = benchmark._process_snapshot()
    assert rows[tree.pids[0]][0] != tree.process.pid
    assert all(rows[pid][1] != tree.process.pid for pid in tree.pids)
    with socket.create_connection(("127.0.0.1", tree.port), timeout=2):
        pass
    benchmark.stop_service(tree.process, 2)
    assert_listener_retired(tree)


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
@pytest.mark.parametrize("double_fork", [False, True])
def test_linux_guard_retires_empty_environment_orphans_before_any_observation(
    service_tree, double_fork
):
    tree = service_tree(
        detached=True,
        detached_supervisor=True,
        scrub_environment=True,
        double_fork=double_fork,
    )
    guard = getattr(tree.process, "_pnms_service_guard", None)
    assert tree.launcher_pid == tree.process.pid
    tree.process.kill()
    assert tree.process.wait(timeout=5) == -signal.SIGKILL
    with socket.create_connection(("127.0.0.1", tree.port), timeout=2):
        pass
    if guard is not None:
        assert benchmark._process_snapshot()[tree.pids[0]][0] == guard.pid
        assert guard.poll() is None
    benchmark.stop_service(tree.process, 2)
    assert_listener_retired(tree)
    assert guard is not None and guard.returncode == 0
    assert not Path(f"/proc/{guard.pid}").exists(), "the benchmark must reap its guard"
    assert all(not Path(f"/proc/{pid}").exists() for pid in tree.pids)


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
@pytest.mark.parametrize("exit_code", [0, 17])
def test_linux_service_handle_preserves_public_cli_pid_and_exit_status(
    service_tree, exit_code
):
    tree = service_tree(exit_leader=True, exit_code=exit_code)
    guard = tree.process._pnms_service_guard
    assert tree.process.pid == tree.launcher_pid and tree.process.pid != guard.pid
    assert tree.process.poll() == exit_code
    assert tree.process.wait(timeout=0) == exit_code
    assert guard.poll() is None, "CLI wait must not wait for service cleanup"
    benchmark.stop_service(tree.process, 2)
    assert tree.process.returncode == exit_code and guard.returncode == 0
    assert_listener_retired(tree)


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
def test_linux_guard_control_eof_reaps_orphans_after_public_cli_exit(service_tree):
    tree = service_tree(
        detached=True,
        detached_supervisor=True,
        scrub_environment=True,
        double_fork=True,
    )
    guard = tree.process._pnms_service_guard
    tree.process.kill()
    assert tree.process.wait(timeout=5) == -signal.SIGKILL
    tree.process._control.shutdown(socket.SHUT_RDWR)
    assert guard.wait(timeout=5) == 0
    tree.process._control.close()
    assert_listener_retired(tree)
    assert all(not Path(f"/proc/{pid}").exists() for pid in tree.pids)


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
def test_linux_failed_guard_without_cleanup_ack_is_reaped_and_channel_closed(tmp_path):
    args = SimpleNamespace(
        runtime="python",
        output=tmp_path,
        service_cwd=ROOT,
        expected_image="registry/image@sha256:" + "f" * 64,
    )
    process, stream = benchmark.start_service(
        args,
        {
            "command": [sys.executable, "-c", "pass"],
            "artifact_root": tmp_path / "artifacts",
            "workflow": benchmark.PYTHON_WORKFLOW,
        },
    )
    guard = process._pnms_service_guard
    try:
        assert process.wait(timeout=5) == 0
        guard.kill()
        assert select.select([process._control], [], [], 5)[0]
        assert guard.returncode is None, "stop must reap the dead guard itself"
        with pytest.raises(FieldContractError, match="guard exited before cleanup"):
            benchmark.stop_service(process, 1)
        assert guard.returncode == -signal.SIGKILL
        assert process._control.fileno() == -1
        assert not Path(f"/proc/{guard.pid}").exists()
    finally:
        if guard.poll() is None:
            guard.kill()
        guard.wait(timeout=5)
        process._control.close()
        stream.close()


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_linux_guard_signals_clean_live_cli_and_preserve_unrelated_sibling(
    service_tree, signum
):
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        tree = service_tree(
            detached=True, detached_supervisor=True, scrub_environment=True
        )
        guard = tree.process._pnms_service_guard
        guard.send_signal(signum)
        assert tree.process.wait(timeout=5) == -signal.SIGTERM
        assert guard.wait(timeout=5) == 0
        assert_listener_retired(tree)
        assert sentinel.poll() is None
        tree.process._control.close()
    finally:
        sentinel.kill()
        sentinel.wait(timeout=5)


def test_shutdown_allows_supervisor_to_drain_and_reap_its_own_children(service_tree):
    tree = service_tree(detached=True, graceful_leader=True, runtime="aoti")
    benchmark.stop_service(tree.process, 2)
    assert tree.process.returncode == 0, "the supervisor must finish its graceful drain"
    assert_listener_retired(tree)


@pytest.mark.skipif(sys.platform != "linux", reason="requires private Linux subreaper")
def test_linux_guard_handshake_failure_reaps_already_launched_empty_env_tree(
    service_tree, tmp_path, monkeypatch
):
    guards, identities, observed = [], {}, {}
    failure = RuntimeError("forced failure before launcher identity receipt")
    original_init = benchmark._LinuxService.__init__

    def record_init(process, guard, *args):
        guards.append(guard)
        original_init(process, guard, *args)

    def fail_receipt(process, timeout=0):
        ready = tmp_path / "service-0/listener.json"
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        data = json.loads(ready.read_text())
        pids = [int((ready.parent / "child.pid").read_text()), data["pid"]]
        identities.update((pid, benchmark._process_token(pid)) for pid in pids)
        observed.update(pids=pids, port=data["port"])
        raise failure

    monkeypatch.setattr(benchmark._LinuxService, "__init__", record_init)
    monkeypatch.setattr(benchmark._LinuxService, "_receive", fail_receipt)
    try:
        with pytest.raises(RuntimeError) as caught:
            service_tree(
                detached=True,
                detached_supervisor=True,
                scrub_environment=True,
                double_fork=True,
            )
        assert caught.value is failure
        assert guards[0].returncode == 0
        assert_listener_retired(SimpleNamespace(**observed))
        assert all(not Path(f"/proc/{pid}").exists() for pid in identities)
    finally:
        benchmark._signal_owned_processes(identities, signal.SIGKILL)
        for guard in guards:
            if guard.poll() is None:
                guard.terminate()
            guard.wait(timeout=20)


def test_shutdown_forces_stubborn_descendant_and_fails_closed_at_deadline(service_tree):
    tree = service_tree(detached=True, stubborn=True)
    started = time.monotonic()
    with pytest.raises(FieldContractError, match="forced shutdown"):
        benchmark.stop_service(tree.process, 0.2)
    assert time.monotonic() - started < 5
    assert_listener_retired(tree)


def test_shutdown_preserves_unrelated_service_processes(service_tree):
    owned, unrelated = service_tree(detached=True), service_tree(detached=True)
    benchmark.stop_service(owned.process, 2)
    assert_listener_retired(owned)
    assert unrelated.process.poll() is None
    for pid in unrelated.pids:
        os.kill(pid, 0)
    with socket.socket() as listener, pytest.raises(OSError):
        listener.bind(("127.0.0.1", unrelated.port))
    benchmark.stop_service(unrelated.process, 2)
    assert_listener_retired(unrelated)


def test_shutdown_rejects_reused_leader_pid_without_signalling_it(service_tree):
    tree = service_tree()
    original = tree.process._pnms_service_start
    tree.process._pnms_service_start = "a different process start"
    try:
        with pytest.raises(FieldContractError, match="identity changed"):
            benchmark.stop_service(
                SimpleNamespace(
                    pid=tree.process.pid,
                    _pnms_service_start=tree.process._pnms_service_start,
                ),
                0.2,
            )
        assert tree.process.poll() is None
        for pid in tree.pids:
            os.kill(pid, 0)
    finally:
        tree.process._pnms_service_start = original
        benchmark.stop_service(tree.process, 2)
    assert_listener_retired(tree)


def test_linux_guard_cleanup_survives_reuse_of_the_public_cli_pid(monkeypatch):
    process = object.__new__(benchmark._LinuxService)
    process.pid, process._pnms_service_start = 123, "original CLI start"
    requested = []
    process._stop = requested.append
    monkeypatch.setattr(
        benchmark, "_process_snapshot", lambda: {123: (1, 123, "unrelated reused PID")}
    )

    def unexpected_signal(*args):
        raise AssertionError("an unrelated reused CLI PID must never be signalled")

    monkeypatch.setattr(benchmark.os, "kill", unexpected_signal)
    benchmark.stop_service(process, 0.2)
    assert requested == [0.2]


@pytest.mark.parametrize("wait_fails", [False, True])
def test_linux_service_stop_error_still_reaps_guard_and_closes_channel(wait_fails):
    control, peer = socket.socketpair()
    waits = []

    def wait(timeout):
        waits.append(timeout)
        if wait_fails:
            raise RuntimeError("fixture guard wait failed")

    guard = SimpleNamespace(wait=wait)
    process = benchmark._LinuxService(guard, control, ["already-finished-cli"])
    peer.close()
    try:
        with pytest.raises(FieldContractError, match="exited before cleanup"):
            process._stop(0.2)
        assert waits, "stop failure must still reap its known private guard"
        assert control.fileno() == -1
    finally:
        control.close()


def test_shutdown_rechecks_descendant_start_identity_before_signalling(service_tree):
    tree = service_tree(detached=True)
    benchmark._signal_owned_processes({tree.pids[-1]: "old start"}, signal.SIGKILL)
    for pid in tree.pids:
        os.kill(pid, 0)
    with socket.socket() as listener, pytest.raises(OSError):
        listener.bind(("127.0.0.1", tree.port))
    benchmark.stop_service(tree.process, 2)
    assert_listener_retired(tree)


def test_service_ownership_retains_orphans_and_tracks_replacements(monkeypatch):
    process = SimpleNamespace(pid=100, _pnms_service_start="launcher")
    rows = {
        100: (1, 100, "launcher"),
        200: (100, 200, "first worker"),
        201: (200, 201, "detached listener"),
        900: (1, 900, "unrelated"),
    }
    monkeypatch.setattr(benchmark, "_process_snapshot", lambda: rows)
    owned = {}
    assert benchmark._observe_service_processes(process, owned) == {
        pid: rows[pid][2] for pid in (100, 200, 201)
    }
    # A replacement has a new session and PID. The original detached listener
    # remains owned after its parent dies.
    del rows[200]
    rows[201] = (1, 201, "detached listener")
    rows[300] = (100, 300, "replacement worker")
    assert benchmark._observe_service_processes(process, owned) == {
        pid: rows[pid][2] for pid in (100, 201, 300)
    }
    del rows[100]
    rows[300] = (1, 300, "replacement worker")
    rows[200] = (1, 200, "reused unrelated PID")
    rows[202] = (200, 202, "unrelated child")
    assert benchmark._observe_service_processes(process, owned) == {
        pid: rows[pid][2] for pid in (201, 300)
    }


def test_linux_launch_marker_matches_whole_environment_entry_and_current_start(
    tmp_path, monkeypatch
):
    proc = tmp_path / "proc"
    marker = f"{benchmark.SERVICE_OWNER_ENV}=owned"
    rows = {}
    for pid, environment, observed_start in (
        (200, marker, "2000"),
        (201, "OTHER=" + marker, "2010"),
        (202, marker, "2020"),
        (203, marker + "-other-session", "2030"),
    ):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        fields = ["S", "1", str(pid), *("0" for _ in range(16)), str(pid * 10)]
        (directory / "stat").write_text(f"{pid} (detached worker) {' '.join(fields)}")
        (directory / "environ").write_bytes(f"OTHER=value\0{environment}\0".encode())
        rows[pid] = (1, pid, observed_start)
    rows[204] = (1, 204, "disappeared")
    original_token = benchmark._process_token
    observations = []

    def token(pid, proc):
        if pid == 202:
            observations.append(pid)
            if len(observations) == 2:
                return "reused while reading environment"
        return original_token(pid, proc)

    monkeypatch.setattr(benchmark, "_process_token", token)
    assert benchmark._marked_service_processes(rows, "owned", proc) == {200: "2000"}
    assert observations == [202, 202]


def test_posix_launch_marker_fallback_rechecks_identity_and_preserves_other_sessions(
    tmp_path, monkeypatch
):
    marker = f"{benchmark.SERVICE_OWNER_ENV}=owned"
    rows = {pid: (1, pid, str(pid)) for pid in (200, 201, 202, 203)}
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"200 worker {marker}\n201 worker OTHER={marker}\n"
            f"202 worker {marker}\n203 worker {marker}-other-session\n",
        ),
    )
    monkeypatch.setattr(
        benchmark, "_process_token", lambda pid, proc: "200" if pid == 200 else "new"
    )
    assert benchmark._marked_service_processes(rows, "owned", tmp_path / "no-proc") == {
        200: "200"
    }


def test_posix_launch_marker_failure_discards_environment_output(tmp_path, monkeypatch):
    environment = "SIMULATED_PRIVATE_ENVIRONMENT=fixture-only-value"

    def failed_ps(command, **kwargs):
        result = subprocess.CompletedProcess(
            command, 1, stdout=environment, stderr=environment
        )
        if kwargs.get("check"):
            result.check_returncode()
        return result

    monkeypatch.setattr(benchmark.subprocess, "run", failed_ps)
    with pytest.raises(
        FieldContractError, match="could not inspect service ownership"
    ) as caught:
        benchmark._marked_service_processes({}, "owned", tmp_path / "no-proc")
    error = caught.value
    assert str(error) == "could not inspect service ownership"
    assert error.__cause__ is None and error.__context__ is None
    for attribute in ("output", "stdout", "stderr"):
        assert getattr(error, attribute, None) is None


def test_linux_process_identity_survives_parentheses_and_rejects_zombies(tmp_path):
    proc = tmp_path / "proc"
    for pid, state in ((200, "S"), (201, "Z")):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        fields = [state, "100", "100", *("0" for _ in range(16)), str(pid * 10)]
        (directory / "stat").write_text(f"{pid} (worker (child)) {' '.join(fields)}")
    assert benchmark._process_snapshot(proc) == {200: (100, 100, "2000")}
    assert benchmark._process_token(200, proc) == "2000"
    assert benchmark._process_token(201, proc) is None
    assert benchmark._process_token(202, proc) is None


@pytest.mark.parametrize("inherited_hosts", [None, "unrelated.example"])
def test_python_service_can_prefetch_pinned_inputs_with_bounded_host_policy(
    tmp_path, monkeypatch, inherited_hosts
):
    from urllib.parse import urlsplit

    names = (
        "E2S_PREFETCH_ALLOWED_HTTPS_HOSTS",
        "E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS",
    )
    for name in names:
        if inherited_hosts is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, inherited_hosts)
    request = json.loads(
        (
            ROOT
            / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
        ).read_text()
    )
    input_hosts = {
        urlsplit(case[key]).hostname
        for case in request["cases"]
        for key in ("mesh_uri", "geometry_uri")
    }
    output = tmp_path / "session"
    output.mkdir()
    observed = output / "prefetch-policy.json"
    code = (
        "import os,json;from pathlib import Path;"
        f"Path({str(observed)!r}).write_text(json.dumps("
        f"{{name:os.environ.get(name,'').split(',') for name in {names!r}}}))"
    )
    args = SimpleNamespace(
        runtime="python",
        expected_image="registry/image@sha256:" + "f" * 64,
        output=output,
        service_cwd=ROOT,
    )
    process, stream = benchmark.start_service(
        args,
        {
            "command": [sys.executable, "-c", code],
            "artifact_root": output / "artifacts",
            "workflow": benchmark.PYTHON_WORKFLOW,
        },
    )
    try:
        assert process.wait(timeout=10) == 0, (output / "service.log").read_text()
    finally:
        benchmark.stop_service(process, 2)
        stream.close()
    policy = json.loads(observed.read_text())
    allowed, signed = (set(policy[name]) for name in names)
    redirects = {"us.aws.cdn.hf.co", "cas-bridge.xethub.hf.co"}
    assert input_hosts <= allowed, "pinned verified inputs must be admitted"
    assert redirects <= allowed & signed, "pinned input redirects must be admitted"
    assert allowed <= input_hosts | redirects, "ambient hosts must not broaden QA"
    assert signed <= redirects, "source hosts must not accept signed queries"


@pytest.mark.parametrize("inherited_prefix", ["", "outer-deployment:"])
def test_python_benchmark_launcher_stream_matches_local_registration_plan(
    tmp_path, monkeypatch, inherited_prefix
):
    from scripts import plugin_dev

    plugin = tmp_path / "plugin"
    plugin_dev.init_plugin(plugin)
    plan = plugin_dev.build_run_local_plan(
        plugin,
        workspace=tmp_path / "service",
        port=0,
        redis_port=0,
    )
    launcher = next(
        row for row in plan["processes"] if row["name"] == "runtime_env_launcher"
    )
    for name, value in launcher["env"].items():
        monkeypatch.setenv(name, value)
    for name in ("STREAM_PREFIX", "REDIS_STREAM_PREFIX"):
        monkeypatch.setenv(name, inherited_prefix)
    output = tmp_path / "session"
    output.mkdir()
    observed = output / "actual-worker-streams.json"
    code = (
        f"import sys,json;sys.path.insert(0,{str(ROOT / 'scripts')!r});"
        "from runtime_env_launcher import _load_launch_specs;"
        "from pathlib import Path;"
        f"Path({str(observed)!r}).write_text(json.dumps("
        "[row['env']['GPU_STREAM_NAME'] for row in _load_launch_specs()]))"
    )
    args = SimpleNamespace(
        runtime="python",
        expected_image="registry/image@sha256:" + "f" * 64,
        output=output,
        service_cwd=ROOT,
    )
    process, stream = benchmark.start_service(
        args,
        {
            "command": [sys.executable, "-c", code],
            "artifact_root": output / "artifacts",
            "workflow": "plugin",
        },
    )
    try:
        assert process.wait(timeout=10) == 0, (output / "service.log").read_text()
    finally:
        benchmark.stop_service(process, 2)
        stream.close()
    assert json.loads(observed.read_text()) == plan["execute_registration_streams"]


@pytest.fixture
def profile():
    return json.loads(
        (ROOT / "qa/inference/cfd_parity_surface_run1_transolver.json").read_text()
    )


@pytest.fixture
def report():
    metrics = {
        name: float(index + 1)
        for index, name in enumerate(sorted(set().union(*METRIC_OUTPUTS.values())))
    }
    return [
        {
            "model": "transolver_surface",
            "dataset": "drivaerml",
            "cases": ["run_1"],
            "metrics": metrics,
            "per_case": [
                {"case_id": "run_1", "metric_dtype": "float64", "metrics": metrics}
            ],
            "wall_seconds": 999,
        }
    ]


def session(runtime, *, ordinal=0):
    math_flags = {
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "float32_matmul_precision": "highest",
        "preferred_blas_library": "default",
    }
    common = {
        "schema_version": 1,
        "status": "passed",
        "runtime": runtime,
        "image_digest": "registry.test/" + runtime + "@sha256:" + "a" * 64,
        "source_sha": "b" * 40,
        "boundary": "client-http-to-full-canonical-report-v1",
        "ordinal": ordinal,
        "workload": {
            "profile_sha256": "p",
            "request_sha256": "r",
            "reference_fields_sha256": "f",
            "reference_report_sha256": "s",
            "model_revision": "m",
            "checkpoint": "weights",
            "statistics": "stats",
            "case": "run_1",
            "point_count": 8828095,
            "seed": 42,
            "block_size": 60000,
            "precision": "float32",
            "provider": {"commit": "provider"},
        },
        "hardware": {
            "uuid": "GPU-same",
            "name": "NVIDIA H100 80GB HBM3",
            "driver_version": "580.1",
            "compute_capability": [9, 0],
            "cpu_model": "same",
            "cpu_affinity_count": 8,
            "memory_limit_bytes": 64000000000,
        },
        "environment": {
            "python": "3.12.3",
            "torch": "2.10+nv",
            "cuda": "13.1",
            "vtk": "9.1.0",
            "math": math_flags,
        },
        "evaluator_environment": {
            "python": "3.12.3",
            "torch": "2.10+nv",
            "cuda": "13.1",
            "vtk": "9.1.0",
            "math": math_flags,
        },
        "cold_startup_seconds": 7.0,
        "numerical_libraries": {
            "libtorch_cpu.so": {"sha256": "e" * 64, "size_bytes": 8},
            "libtorch_cuda.so": {"sha256": "f" * 64, "size_bytes": 16},
        },
        "prerequisite": {
            "parity": "passed",
            "canonical_report_sha256": "c" * 64,
            "artifacts_verified": True,
        },
        "postrun": {
            "parity": "passed",
            "canonical_report_sha256": "c" * 64,
            "artifacts_verified": True,
        },
        "samples": [
            {
                "label": label,
                "seconds": seconds,
                "parity": "passed",
                "canonical_report_sha256": "c" * 64,
                "artifacts_verified": True,
            }
            for label, seconds in zip(
                ("first", "warm-1", "warm-2", "warm-3"),
                (20.0, 10.0, 12.0, 11.0),
                strict=True,
            )
        ],
    }
    return common


def test_canonical_output_preserves_all_22_metrics_but_excludes_incidental_timing(
    profile, report
):
    changed = copy.deepcopy(report)
    changed[0]["wall_seconds"] = 1
    first = benchmark.canonical_report(profile, report)
    assert len(first["metrics"]) == 22
    assert first == benchmark.canonical_report(profile, changed)


def test_matched_comparison_uses_only_three_warm_samples_and_keeps_cold_separate():
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    for row in (rows[1], rows[4]):
        for sample, seconds in zip(row["samples"], (999, 5.0, 6.0, 5.5), strict=True):
            sample["seconds"] = seconds
    summary = benchmark.compare_sessions(rows)
    assert summary["status"] == "passed"
    assert summary["warm_speedup_vs_python"]["aoti"] == 2.0
    assert summary["runtime_statistics"]["aoti"]["first_seconds"]["median"] == 999
    assert summary["runtime_statistics"]["aoti"]["cold_startup_seconds"]["median"] == 7
    assert summary["execution_order"] == list(ORDER)
    assert summary["runtime_statistics"]["aoti"]["warm_seconds"]["count"] == 6
    assert summary["kernel_throughput_measured"] is False


def test_shuffled_session_arguments_keep_receipts_in_execution_order(
    tmp_path, monkeypatch
):
    from qa.native_inference import cfd_benchmark_reference

    profile = tmp_path / "profile.json"
    reference_fields = tmp_path / "reference-fields.json"
    profile.write_text("{}")
    reference_fields.write_text("{}")
    report_sha256 = "c" * 64
    paths = []
    for ordinal, runtime in enumerate(ORDER):
        row = session(runtime, ordinal=ordinal)
        row["workload"].update(
            profile_sha256=benchmark.sha256_file(profile),
            reference_fields_sha256=benchmark.sha256_file(reference_fields),
            reference_report_sha256=report_sha256,
        )
        path = tmp_path / f"{ordinal}-{runtime}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    monkeypatch.setattr(
        cfd_benchmark_reference,
        "qualified_reference",
        lambda path: ([], {"report_sha256": report_sha256}),
    )
    monkeypatch.setattr(benchmark, "validate_evidence", lambda *args: {"run_1": {}})
    monkeypatch.setattr(
        benchmark, "load_session", lambda path, *args: benchmark.read_json_object(path)
    )
    output = tmp_path / "comparison.json"
    args = benchmark.parser().parse_args(
        [
            "compare",
            "--profile",
            str(profile),
            "--reference-fields",
            str(reference_fields),
            "--sessions",
            *(str(paths[index]) for index in (4, 1, 5, 2, 0, 3)),
            "--output",
            str(output),
        ]
    )

    assert benchmark.compare(args) == 0

    summary = benchmark.read_json_object(output)
    assert summary["status"] == "passed"
    assert summary["execution_order"] == list(ORDER)
    assert summary["session_receipts"] == [
        {"path": str(path.resolve()), **benchmark.file_identity(path)} for path in paths
    ]


@pytest.mark.parametrize("order", [ORDER[:3], ORDER[:3] * 2, ORDER + ("python",)])
def test_comparison_rejects_missing_reversed_or_unbalanced_sessions(order):
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(order)]
    with pytest.raises(FieldContractError, match="six|counterbalanced|order"):
        benchmark.compare_sessions(rows)


@pytest.mark.parametrize(
    "section,key",
    [
        ("hardware", "uuid"),
        ("hardware", "driver_version"),
        ("hardware", "cpu_affinity_count"),
        ("environment", "torch"),
        ("environment", "math"),
        ("evaluator_environment", "vtk"),
        ("workload", "request_sha256"),
        ("workload", "checkpoint"),
    ],
)
def test_unmatched_work_hardware_math_or_stack_cannot_emit_speedups(section, key):
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    rows[1][section][key] = "different"
    with pytest.raises(FieldContractError, match="match"):
        benchmark.compare_sessions(rows)


@pytest.mark.parametrize("gate", ["prerequisite", "postrun"])
def test_parity_before_and_after_timing_is_mandatory(gate):
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    rows[1][gate]["parity"] = "failed"
    with pytest.raises(FieldContractError, match="parity"):
        benchmark.compare_sessions(rows)


def test_changed_output_hash_after_timing_cannot_emit_speedup():
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    rows[1]["postrun"]["canonical_report_sha256"] = "d" * 64
    with pytest.raises(FieldContractError, match="output"):
        benchmark.compare_sessions(rows)


@pytest.mark.parametrize(
    "section,key",
    [
        ("environment", "math"),
        ("environment", "cuda"),
        ("evaluator_environment", "vtk"),
        ("hardware", "driver_version"),
    ],
)
def test_equally_missing_observations_do_not_count_as_a_match(section, key):
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    for row in rows:
        del row[section][key]
    with pytest.raises(FieldContractError, match="observation"):
        benchmark.compare_sessions(rows)


@pytest.fixture
def artifact_server():
    state = SimpleNamespace(files={}, content_lengths={}, requested=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = parse_qs(urlsplit(self.path).query)["artifact"][0]
            state.requested.append(name)
            data = state.files[name]
            self.send_response(200)
            length = state.content_lengths.get(name, len(data))
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.http = benchmark.HTTP(f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def python_response(profile, report):
    config = {
        "benchmark": {"models": copy.deepcopy(profile["config"]["models"])},
        "run": {"seed": 42, "save_inference_mesh": False},
        "metrics": profile["request"]["metrics"],
    }
    diagnostics = {
        "returncode": 0,
        "timed_out": False,
        "cancelled": False,
        "duration_seconds": 12.0,
        "argv": ["python", "-m", "physicsnemo.cfd.evaluation.benchmarks.run"],
    }
    files = {
        "benchmark_results.json": json.dumps(report).encode(),
        "resolved_config.json": json.dumps(config).encode(),
        "benchmark_diagnostics.json": json.dumps(diagnostics).encode(),
        "benchmark.log": b"completed provider inference",
        "benchmark_results.csv": b"all metric rows",
        "benchmark_results.html": b"<html>all metric rows</html>",
        "benchmark_artifacts.json": b'{"outputs":["benchmark_results.json"]}\n',
    }
    downloaded = []

    class HTTP:
        def download_python(self, url, destination):
            name = destination.name
            downloaded.append(name)
            destination.write_bytes(files[name])
            return benchmark.identity_bytes(files[name])

    request = {
        "models": ["transolver_surface"],
        "metrics": profile["request"]["metrics"],
        "cases": [],
    }
    context = {
        "runtime": "python",
        "workflow": benchmark.PYTHON_WORKFLOW,
        "profile": profile,
        "request": request,
        "environment": session("python")["environment"],
        "interpreter": "python",
    }
    result = {
        "execution": {
            "run_id": "run",
            "workflow": benchmark.PYTHON_WORKFLOW,
            "status": "succeeded",
            "outputs": [
                {
                    "name": name,
                    "media_type": {
                        ".json": "application/json",
                        ".csv": "text/csv",
                        ".html": "text/html",
                        ".log": "text/plain",
                    }[Path(name).suffix],
                    "storage_path": "/outputs/provider-attempt/" + name,
                    **({"primary": True} if name == "benchmark_results.json" else {}),
                }
                for name in files
            ],
        },
        "payload": {
            "model_names": ["transolver_surface"],
            "case_ids": ["run_1"],
            "selected_metrics": profile["request"]["metrics"],
            "provider": {
                key: profile["provider"][key]
                for key in (
                    "repository",
                    "tag",
                    "version",
                    "commit",
                    "physicsnemo_version",
                    "python_version",
                )
            },
            "case_digests": [],
        },
    }
    return SimpleNamespace(
        http=HTTP(), files=files, downloaded=downloaded, context=context, result=result
    )


def test_python_accepts_the_unchanged_provider_default_fp32_configuration(
    python_response, tmp_path
):
    state = python_response
    config = json.loads(state.files["resolved_config.json"])
    del config["benchmark"]["models"][0]["kwargs"]["cuda_bf16_autocast"]
    state.files["resolved_config.json"] = json.dumps(config).encode()
    report, *_ = benchmark.collect_python(
        state.http, state.context, state.result, "run", tmp_path
    )
    assert report[0]["cases"] == ["run_1"]


def test_python_full_report_boundary_downloads_every_registered_report_and_audit_artifact(
    python_response, tmp_path
):
    state = python_response
    benchmark.collect_python(state.http, state.context, state.result, "run", tmp_path)
    assert set(state.downloaded) == set(state.files), (
        "CSV/HTML and audit outputs must not be omitted from Python report wall time"
    )


def test_python_public_artifact_descriptors_download_through_real_http(
    python_response, artifact_server, tmp_path
):
    state = python_response
    artifact_server.files.update(state.files)
    report, report_path, *_ = benchmark.collect_python(
        artifact_server.http, state.context, state.result, "run", tmp_path
    )
    assert report == json.loads(state.files["benchmark_results.json"])
    assert report_path == tmp_path / "benchmark_results.json"
    assert set(artifact_server.requested) == set(state.files)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == state.files


def test_python_artifact_download_hashes_actual_bytes(artifact_server, tmp_path):
    data = b"legacy provider audit bytes\n"
    artifact_server.files["audit"] = data
    observed = artifact_server.http.download_python(
        "/results?artifact=audit", tmp_path / "audit"
    )
    assert observed == {
        "sha256": "32cf099346a3bd42d54add05c9c1ae68f52feb0d1bb51b8841684d6c0670ebee",
        "size_bytes": 28,
    }
    assert (tmp_path / "audit").read_bytes() == data


@pytest.mark.parametrize("advertise_length", [True, False])
def test_python_artifact_download_rejects_bytes_over_the_audit_bound(
    artifact_server, tmp_path, advertise_length
):
    artifact_server.files["audit"] = b"x" * (16 * 1024 * 1024 + 1)
    if not advertise_length:
        artifact_server.content_lengths["audit"] = None
    with pytest.raises(FieldContractError, match="size bound"):
        artifact_server.http.download_python(
            "/results?artifact=audit", tmp_path / "audit"
        )


def test_python_artifact_download_rejects_truncated_body(artifact_server, tmp_path):
    artifact_server.files["audit"] = b"incomplete audit"
    artifact_server.content_lengths["audit"] = 50
    with pytest.raises(FieldContractError, match="Content-Length"):
        artifact_server.http.download_python(
            "/results?artifact=audit", tmp_path / "audit"
        )


def test_python_artifact_download_preserves_an_existing_file(artifact_server, tmp_path):
    artifact_server.files["audit"] = b"new audit"
    destination = tmp_path / "audit"
    destination.write_bytes(b"previous audit")
    with pytest.raises(FileExistsError):
        artifact_server.http.download_python("/results?artifact=audit", destination)
    assert destination.read_bytes() == b"previous audit"


def test_native_artifact_download_still_rejects_wrong_digest(artifact_server, tmp_path):
    data = b"native artifact with an incorrect advertised digest"
    artifact_server.files["native"] = data
    with pytest.raises(FieldContractError, match="advertised digest/size"):
        artifact_server.http.download(
            "/results?artifact=native",
            tmp_path / "native",
            {"sha256": "0" * 64, "size_bytes": len(data)},
        )


@pytest.fixture
def preflight(profile, report, tmp_path, monkeypatch):
    from qa.native_inference import cfd_reference
    from qa.scripts import run_native_cfd_reference
    from qa.native_inference import cfd_benchmark_reference

    args = SimpleNamespace(
        expected_image="registry.test/python@sha256:" + "a" * 64,
        expected_source_sha="b" * 40,
        poll_interval=0.1,
        profile=tmp_path / "profile.json",
        request=tmp_path / "request.json",
        reference_fields=tmp_path / "fields.json",
        input_root=tmp_path,
        checkpoint=tmp_path / "checkpoint",
        stats=tmp_path / "stats",
        service_command_json=tmp_path / "command.json",
        service_files_json=tmp_path / "sources.json",
        output=tmp_path,
        environment_python=Path("/selected/python"),
        runtime="python",
        request_timeout=300,
    )
    args.profile.write_text(json.dumps(profile))
    args.request.write_text("{}")
    args.reference_fields.write_text(
        json.dumps({"environment": session("python")["environment"]})
    )
    args.checkpoint.write_bytes(b"weights")
    args.stats.write_bytes(b"stats")
    args.service_command_json.write_text(json.dumps(["/shipped/entrypoint"]))
    args.service_files_json.write_text(
        json.dumps({str(args.checkpoint): benchmark.file_identity(args.checkpoint)})
    )
    monkeypatch.setattr(benchmark, "validate_reference_request", lambda *args: None)
    monkeypatch.setattr(benchmark, "validate_evidence", lambda *args: {"run_1": {}})
    monkeypatch.setattr(benchmark, "verify_file", lambda *args: None)
    monkeypatch.setattr(run_native_cfd_reference, "_verify_inputs", lambda *args: None)
    monkeypatch.setattr(
        cfd_benchmark_reference, "qualified_reference", lambda *args: (report, {})
    )

    def foreign_assets(*args):
        raise AssertionError(
            "explicit local pins must not reopen reference producer paths"
        )

    monkeypatch.setattr(
        cfd_reference, "reference_model_asset_arguments", foreign_assets
    )
    monkeypatch.setattr(benchmark.shutil, "which", lambda value: "/shipped/entrypoint")

    selected = {
        "environment": session("python")["environment"],
        "provider": profile["provider"],
        "interpreter": "/selected/python",
        "torch_libraries": session("python")["numerical_libraries"],
    }
    actual = copy.deepcopy(selected)

    def probe(argv, **kwargs):
        (tmp_path / "environment.json").write_text(json.dumps(selected))

    monkeypatch.setattr(benchmark.subprocess, "run", probe)
    monkeypatch.setattr(benchmark, "hardware", lambda: session("python")["hardware"])
    monkeypatch.setattr(
        benchmark, "environment_observation", lambda profile: actual, raising=False
    )
    return SimpleNamespace(args=args, selected=selected, actual=actual)


def test_preflight_uses_explicit_assets_without_reopening_other_jobs_reference_paths(
    preflight,
):
    assert benchmark.prepare(preflight.args)["checkpoint"] == preflight.args.checkpoint


@pytest.mark.parametrize("cache_exists", [False, True])
def test_configured_common_cli_output_root_cannot_hide_python_metric_cache(
    preflight, tmp_path, cache_exists
):
    preflight.args.service_artifact_root = tmp_path / "owned-workspace/outputs"
    context = benchmark.prepare(preflight.args)
    output = preflight.args.service_artifact_root / "attempt/benchmark-output"
    output.mkdir(parents=True)
    cache = output / "metrics_cache"
    if cache_exists:
        cache.mkdir()
    config = output.parent / "resolved_config.json"
    config.write_text(
        json.dumps(
            {
                "run": {
                    "output_dir": str(output),
                    "metrics_cache": {"path": str(cache), "enabled": True},
                }
            }
        )
    )
    if cache_exists:
        with pytest.raises(FieldContractError, match="not initially absent"):
            benchmark.observe_python_cache(context)
    else:
        benchmark.observe_python_cache(context)
        assert context["cache_configs"] == {config}
        assert context["artifact_root"] == preflight.args.service_artifact_root


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")),
    reason="the shipped secure JSON writer requires O_NOFOLLOW/O_DIRECTORY",
)
@pytest.mark.parametrize("write_stage", ["created", "partial"])
def test_python_cache_observer_retries_the_in_progress_shipped_config_writer(
    tmp_path, monkeypatch, write_stage
):
    from physicsnemo_cfd_runtime import safe_files

    output = tmp_path / "attempt/benchmark-output"
    output.mkdir(parents=True)
    config = output.parent / "resolved_config.json"
    payload = {
        "run": {
            "output_dir": str(output),
            "metrics_cache": {"path": str(output / "metrics_cache"), "enabled": True},
        }
    }
    context = {"runtime": "python", "artifact_root": tmp_path, "cache_configs": set()}
    visible, finish = Event(), Event()
    errors = []
    original_open, original_dump = safe_files.open_exclusive_file, json.dump

    def pause():
        visible.set()
        assert finish.wait(5), "the observer did not release the config writer"

    def open_file(path):
        descriptor = original_open(path)
        if write_stage == "created":
            try:
                pause()
            except BaseException:
                os.close(descriptor)
                raise
        return descriptor

    def dump(payload, handle, **kwargs):
        class PartialWriter:
            def write(self, chunk):
                count = handle.write(chunk)
                if not visible.is_set():
                    handle.flush()
                    pause()
                return count

        return original_dump(payload, PartialWriter(), **kwargs)

    def write():
        try:
            safe_files.write_json_exclusive(config, payload)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(safe_files, "open_exclusive_file", open_file)
    if write_stage == "partial":
        monkeypatch.setattr(safe_files.json, "dump", dump)
    writer = Thread(target=write)
    writer.start()
    try:
        assert visible.wait(5), "the shipped writer did not expose the config path"
        assert config.read_text() == ("" if write_stage == "created" else "{")
        benchmark.observe_python_cache(context)
        assert context["cache_configs"] == set()
    finally:
        finish.set()
        writer.join(timeout=5)
    assert not writer.is_alive(), "the shipped config writer did not terminate"
    assert not errors
    benchmark.observe_python_cache(context)
    assert context["cache_configs"] == {config}


@pytest.fixture
def python_cache_request(python_response, report, tmp_path):
    state = python_response
    output = tmp_path / "artifacts/attempt/benchmark-output"
    output.mkdir(parents=True)
    state.config = output.parent / "resolved_config.json"
    state.cache_config = {
        "run": {
            "output_dir": str(output),
            "metrics_cache": {"path": str(output / "metrics_cache"), "enabled": True},
        }
    }
    state.context.update(
        artifact_root=tmp_path / "artifacts",
        cache_configs=set(),
        timeout=10,
        poll_interval=0,
        reference_report=report,
        reference_fields={},
    )
    state.events = []

    def collect(http, context, result, run_id, output):
        state.events.append("collect")
        path = output / "benchmark_results.json"
        path.write_text(json.dumps(report))
        return report, path, {}, context["environment"], {}

    state.collect = collect
    return state


def test_python_request_rechecks_config_completed_with_terminal_results(
    python_cache_request, tmp_path
):
    state = python_cache_request
    state.config.write_text("")

    class HTTP:
        def json(self, path, *, body=None, expected=200):
            if body is not None:
                return {"run_id": "run"}
            if path.endswith("status"):
                assert state.context["cache_configs"] == set()
                return {"status": "succeeded"}
            state.config.write_text(json.dumps(state.cache_config))
            return state.result

    sample = benchmark.request_sample(
        HTTP(), state.context, tmp_path / "sample", collect=state.collect
    )
    assert sample["parity"] == "passed"
    assert state.context["cache_configs"] == {state.config}
    assert state.events == ["collect"]


@pytest.mark.parametrize("contents", ["", '{"run":', "corrupt"])
@pytest.mark.parametrize("terminal", [False, True])
def test_python_request_rejects_unfinished_config_at_success_or_deadline(
    python_cache_request, tmp_path, monkeypatch, contents, terminal
):
    state = python_cache_request
    state.config.write_text(contents)

    class HTTP:
        def json(self, path, *, body=None, expected=200):
            if body is not None:
                state.events.append("submit")
                return {"run_id": "run"}
            if path.endswith("status"):
                state.events.append("status")
                return {"status": "succeeded" if terminal else "running"}
            state.events.append("results")
            return state.result

    ticks = iter((0, 0, 11))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    error = json.JSONDecodeError if terminal else FieldContractError
    with pytest.raises(error, match=None if terminal else "deadline"):
        benchmark.request_sample(
            HTTP(), state.context, tmp_path / "sample", collect=state.collect
        )
    assert state.events == ["submit", "status", *(["results"] if terminal else [])]
    assert state.context["cache_configs"] == set()
    assert not (tmp_path / "sample/http-results.json").exists()


def test_configured_service_observation_root_must_be_initially_absent(
    preflight, tmp_path
):
    preflight.args.service_artifact_root = tmp_path / "previous-service"
    preflight.args.service_artifact_root.mkdir()
    (preflight.args.service_artifact_root / "resolved_config.json").write_text("{}")
    with pytest.raises(FieldContractError, match="fresh|absent"):
        benchmark.prepare(preflight.args)


@pytest.mark.parametrize(
    "mismatch", ["interpreter", "provider", "environment", "libraries"]
)
def test_preflight_rejects_driver_evaluator_that_differs_from_selected_stack(
    preflight, mismatch
):
    actual = preflight.actual
    if mismatch == "interpreter":
        actual["interpreter"] = "/foreign/python"
    elif mismatch == "provider":
        actual["provider"]["commit"] = "f" * 40
    elif mismatch == "environment":
        actual["environment"]["math"]["allow_tf32_matmul"] = True
    else:
        actual["torch_libraries"]["libtorch_cuda.so"]["sha256"] = "0" * 64
    with pytest.raises(FieldContractError, match="evaluator"):
        benchmark.prepare(preflight.args)


@pytest.fixture
def native_response(profile, report, monkeypatch):
    import numpy as np
    from qa.native_inference.cfd_contract import CASE_PINS, inference_seed

    monkeypatch.setitem(benchmark.CASE_POINT_COUNTS, "run_1", 2)
    environment = session("aoti")["environment"]
    metadata = {
        "point_count": 2,
        "point_limit": 0,
        "block_size": 60000,
        "output_dtype": "float32",
        "math_profile": "provider-default-v1",
        "permutation": "torch.cuda.randperm",
        "permutation_seed": inference_seed(42, "run_1"),
        "block_plan": [{"begin": 0, "points": 2}],
        "deployment": {
            "compute_capability": [9, 0],
            "torch_version": environment["torch"],
            "cuda_runtime_version": 13010,
        },
        "vtk_version": environment["vtk"],
        "math_flags": environment["math"],
    }
    physical = np.arange(8, dtype="<f4").reshape(2, 4)
    files = {
        "native_metadata": json.dumps(metadata).encode(),
        "physical_fields": physical.tobytes(),
        "standardized_fields": physical.tobytes(),
    }
    context = {
        "runtime": "aoti",
        "workflow": benchmark.NATIVE_WORKFLOW,
        "profile": profile,
        "request": {},
        "environment": environment,
        "source_sha": "b" * 40,
        "image": session("aoti")["image_digest"],
        "model_build_sources": {"full": "c" * 40, "tail": "c" * 40},
        "reference_report": report,
        "reference_fields": {"physical": physical, "standardized": physical},
        "timeout": 10,
        "poll_interval": 0.01,
        "cache_configs": set(),
    }
    native = {
        "runtime": "cpp",
        "backend": "aoti",
        "case_id": "run_1",
        "run_seed": 42,
        "permutation_seed": inference_seed(42, "run_1"),
        "source_sha": context["source_sha"],
        "mesh_sha256": CASE_PINS["run_1"]["sha256"],
        "geometry_sha256": CASE_PINS["run_1"]["geometry_sha256"],
        "release_image": context["image"],
        "model_build_sources": context["model_build_sources"],
        "metadata": metadata,
    }
    result = {
        "execution": {
            "run_id": "run",
            "workflow": context["workflow"],
            "status": "succeeded",
            "outputs": [
                {"name": name, **benchmark.identity_bytes(data)}
                for name, data in files.items()
            ],
        },
        "payload": {"native": native},
    }
    state = SimpleNamespace(
        context=context, result=result, files=files, elapsed=0.0, events=[], posts=0
    )

    class HTTP:
        def json(self, path, *, body=None, expected=200):
            if body is not None:
                state.posts += 1
                state.events.append("submit")
                state.elapsed += 1
                return {"run_id": "run"}
            if path.endswith("status"):
                state.elapsed += 2
                return {"status": "succeeded"}
            state.events.append("results")
            return result

        def download(self, url, destination, identity):
            state.elapsed += 3
            state.events.append("download:" + destination.name)
            destination.write_bytes(files[destination.name])
            assert benchmark.file_identity(destination) == {
                key: identity[key] for key in ("sha256", "size_bytes")
            }

    def evaluate(context, fields, output):
        assert np.array_equal(fields, physical)
        assert (
            len([event for event in state.events if event.startswith("download:")]) == 3
        )
        state.events.append("shared-evaluator")
        state.elapsed += 5
        output.mkdir()
        path = output / "benchmark_results.json"
        path.write_text(json.dumps(report))
        return report, path

    monkeypatch.setattr(benchmark, "evaluate_native", evaluate)
    state.http = HTTP()
    return state


def test_native_accepts_the_shared_registry_image_identity_without_legacy_alias(
    native_response,
):
    state = native_response
    native = state.result["payload"]["native"]
    del native["release_image"]
    native["runtime_identity"] = {
        "kind": "registry-image",
        "value": state.context["image"],
    }
    assert (
        benchmark.native_metadata(state.result, state.context, "run")[0]["point_count"]
        == 2
    )


def test_native_wall_includes_actual_downloads_and_shared_evaluator_before_report_hash(
    native_response, tmp_path
):
    state = native_response

    def clock():
        state.events.append("clock")
        return state.elapsed

    sample = benchmark.request_sample(
        state.http, state.context, tmp_path / "first", clock=clock
    )
    assert sample["seconds"] == 17
    assert state.events[-1] == "clock"
    assert state.events.index("shared-evaluator") > state.events.index(
        "download:physical_fields"
    )
    assert set(sample["files"]) >= {
        "physical_fields",
        "standardized_fields",
        "native_metadata",
        "shared-evaluator/benchmark_results.json",
        "canonical-report.json",
        "http-results.json",
    }
    assert sample["parity"] == "passed" and state.posts == 1


def test_uncertain_request_is_not_resubmitted(native_response, tmp_path, monkeypatch):
    state = native_response

    class HTTP:
        def json(self, path, *, body=None, expected=200):
            if body is not None:
                state.posts += 1
                return {"run_id": "run"}
            return {"status": "running"}

    ticks = iter((0, 0, 11))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    with pytest.raises(FieldContractError, match="deadline"):
        benchmark.request_sample(HTTP(), state.context, tmp_path / "first")
    assert state.posts == 1


def test_actual_output_tampering_blocks_reloaded_session(native_response, tmp_path):
    state = native_response
    sample = benchmark.request_sample(
        state.http, state.context, tmp_path / "prerequisite", clock=lambda: 1
    )
    sample["directory"] = "prerequisite"
    saved = {
        **session("aoti"),
        "prerequisite": sample,
        "samples": [],
        "postrun": sample,
    }
    receipt = tmp_path / "session.json"
    receipt.write_text(json.dumps(saved))
    (tmp_path / "prerequisite/physical_fields").write_bytes(b"tampered fields")
    with pytest.raises(FieldContractError, match="size|SHA-256"):
        benchmark.load_session(
            receipt,
            state.context["profile"],
            state.context["reference_report"],
            state.context["reference_fields"],
        )


def test_matching_versions_without_torch_binary_observations_cannot_emit_speedup():
    rows = [session(runtime, ordinal=index) for index, runtime in enumerate(ORDER)]
    for row in rows:
        del row["numerical_libraries"]
    with pytest.raises(FieldContractError, match="librar|observation"):
        benchmark.compare_sessions(rows)


def test_reload_rejects_changed_runtime_environment_even_when_report_bytes_match(
    native_response, tmp_path
):
    state = native_response
    sample = benchmark.request_sample(
        state.http, state.context, tmp_path / "prerequisite", clock=lambda: 1
    )
    sample.update(directory="prerequisite", environment={"torch": "unmatched"})
    receipt = tmp_path / "session.json"
    receipt.write_text(
        json.dumps(
            {
                **session("aoti"),
                "prerequisite": sample,
                "samples": [],
                "postrun": sample,
            }
        )
    )
    with pytest.raises(FieldContractError, match="environment"):
        benchmark.load_session(
            receipt,
            state.context["profile"],
            state.context["reference_report"],
            state.context["reference_fields"],
        )


@pytest.fixture
def fake_session(tmp_path, profile, report, monkeypatch):
    import io

    args = SimpleNamespace(
        output=tmp_path / "session",
        runtime="python",
        expected_image=session("python")["image_digest"],
        expected_source_sha="b" * 40,
        run_id="cpu-component",
        ordinal=0,
        environment_python=Path("/selected/python"),
        shutdown_timeout=1,
        profile=tmp_path / "profile.json",
        request=tmp_path / "request.json",
        reference_fields=tmp_path / "fields.json",
        url="http://owned",
    )
    for path in (args.profile, args.request, args.reference_fields):
        path.write_text("{}")
    context = {
        "runtime": "python",
        "hardware": session("python")["hardware"],
        "environment": session("python")["environment"],
        "interpreter": "/selected/python",
        "torch_libraries": {
            name: {"path": "/selected/" + name, **identity}
            for name, identity in session("python")["numerical_libraries"].items()
        },
        "service_files": {"/shipped/entrypoint": {"sha256": "a" * 64, "size_bytes": 1}},
        "model_build_sources": None,
        "bundle_digest": None,
        "qualification": {"report_sha256": "c" * 64},
        "command": ["/shipped/entrypoint"],
        "profile": profile,
        "artifact_root": args.output / "service-artifacts",
    }
    state = SimpleNamespace(
        args=args,
        context=context,
        labels=[],
        stopped=[],
        fail_post=False,
        fail_stop=False,
    )

    class Process:
        pid = 123

        def poll(self):
            return None

    state.process = Process()
    monkeypatch.setattr(benchmark, "prepare", lambda args: context)
    monkeypatch.setattr(
        benchmark, "start_service", lambda *args: (state.process, io.BytesIO())
    )
    monkeypatch.setattr(
        benchmark, "wait_ready", lambda *args: {"readiness": {"ready": True}}
    )
    monkeypatch.setattr(
        benchmark,
        "verify_native_stack",
        lambda *args: {"pid": 123, "start": "1", "libraries": {}},
    )

    def sample(http, context, output):
        state.labels.append(output.name)
        if output.name == "postrun" and state.fail_post:
            raise FieldContractError("postrun metric parity failed")
        return {
            "seconds": 1.0,
            "environment": context["environment"],
            "parity": "passed",
            "artifacts_verified": True,
            "canonical_report_sha256": "d" * 64,
        }

    monkeypatch.setattr(benchmark, "request_sample", sample)

    def stop(process, timeout):
        state.stopped.append(process)
        if state.fail_stop:
            raise FieldContractError("forced shutdown")

    monkeypatch.setattr(benchmark, "stop_service", stop)
    return state


@pytest.mark.parametrize("runtime", ["python", "aoti", "tensorrt"])
def test_session_qualifies_before_and_after_first_plus_three_warm_requests(
    fake_session, runtime
):
    state = fake_session
    state.args.runtime = runtime
    assert benchmark.run_session(state.args) == 0
    saved = json.loads((state.args.output / "session.json").read_text())
    assert saved["status"] == "passed"
    assert state.labels == [
        "prerequisite",
        "first",
        "warm-1",
        "warm-2",
        "warm-3",
        "postrun",
    ]
    assert state.stopped == [state.process]
    assert saved["cold_startup_seconds"] >= 0
    assert "not cold-model" in saved["methodology"]["first"]


def test_session_retains_ownership_at_readiness_outside_the_request_loop(fake_session):
    state = fake_session
    state.process._pnms_service_owned = {}
    observed = []

    def observe(process, owned):
        assert process is state.process
        assert owned is state.process._pnms_service_owned
        observed.append(len(state.labels))

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark, "_observe_service_processes", observe)
        assert benchmark.run_session(state.args) == 0
    assert observed == [0, 6]


@pytest.mark.parametrize("failure", ["fail_post", "fail_stop"])
def test_failed_postqualification_or_shutdown_cannot_leave_passed_receipt(
    fake_session, failure
):
    state = fake_session
    setattr(state, failure, True)
    with pytest.raises(FieldContractError, match="parity|shutdown"):
        benchmark.run_session(state.args)
    saved = json.loads((state.args.output / "session.json").read_text())
    assert saved["status"] == "failed"
    assert state.stopped == [state.process]


def test_session_receipt_records_actual_service_output_observation_root(fake_session):
    state = fake_session
    assert benchmark.run_session(state.args) == 0
    saved = json.loads((state.args.output / "session.json").read_text())
    assert saved["service_artifact_root"] == str(state.context["artifact_root"])


def test_native_library_proof_hashes_each_mapped_file_once_and_rejects_changed_bytes(
    tmp_path, monkeypatch
):
    from qa.native_inference import pnms_http_job

    proc = tmp_path / "proc"
    worker = proc / "321"
    worker.mkdir(parents=True)
    executable = tmp_path / "native-worker"
    executable.write_bytes(b"executable")
    (worker / "exe").symlink_to(executable)
    libraries = {}
    lines = []
    for name, content in (
        ("libtorch_cpu.so", b"cpu-stack"),
        ("libtorch_cuda.so", b"cuda-stack"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        libraries[name] = {"path": str(path), **benchmark.file_identity(path)}
        lines.extend([f"100-200 r-xp 0 00:00 0 {path}"] * 3)
    (worker / "maps").write_text("\n".join(lines))
    monkeypatch.setattr(pnms_http_job, "_descendants", lambda *args: {321: "start"})
    original = benchmark.file_identity
    calls = []

    def identity(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(benchmark, "file_identity", identity)
    process = SimpleNamespace(pid=123)
    proof = benchmark.verify_native_stack(
        process, {"torch_libraries": libraries}, proc_root=proc
    )
    assert proof["pid"] == 321
    assert sorted(calls) == sorted(libraries), (
        "shared memory segments must not trigger repeated multi-GB hashes"
    )
    (tmp_path / "libtorch_cuda.so").write_bytes(b"edited-library")
    with pytest.raises(FieldContractError, match="library bytes"):
        benchmark.verify_native_stack(
            process, {"torch_libraries": libraries}, proc_root=proc
        )
