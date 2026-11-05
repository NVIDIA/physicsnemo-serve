# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable lifecycle checks for the externally exposed AIFS service."""

import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def serving():
    path = ROOT / "qa/plugins/earth2-aifs2ens/serve.py"
    spec = importlib.util.spec_from_file_location("aifs2ens_service", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Process:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.code = 0
        return 0


def fake_service(serving, monkeypatch, tmp_path, *, available=True, failure=None):
    stream = "execute.earth2-gpu:gpu:local:earth2-aifs2ens-run-local:0"
    plan = {
        "workspace": tmp_path,
        "port": 8001,
        "redis_port": 6379,
        "redis_url": "redis://127.0.0.1:6379/0",
        "execute_registration_streams": [stream],
        "processes": [
            {"name": name, "argv": [name], "env": {}}
            for name in ("redis", "inference_server", "runtime_env_launcher")
        ],
    }
    (tmp_path / "logs").mkdir()
    processes, killed, spawned, ports = [], [], [], []
    stop = threading.Event()
    clock = SimpleNamespace(value=0, reads=0, ready=None)

    def popen(argv, **kwargs):
        if failure == "spawn" and len(processes) == 1:
            raise OSError("failed to start server")
        proc = Process(100 + len(processes), 7 if failure == "child" else None)
        processes.append(proc)
        spawned.append(kwargs)
        return proc

    class Redis:
        def hgetall(self, key):
            assert key == "gpu:registry"
            clock.reads += 1
            return {
                "worker": json.dumps(
                    {
                        "stream": stream,
                        "executor_class": "earth2-gpu",
                        "device_kind": "gpu",
                        "status": "available"
                        if available and clock.reads > 1
                        else "warming",
                    }
                )
            }

        def xrevrange(self, key, *, count):
            assert key == "release"
            assert count == 100
            return [] if clock.reads < 3 else [("1-0", {"run_id": "parent"})]

        def close(self):
            pass

    def sleep(seconds):
        clock.value += seconds
        ready = tmp_path / "service-ready.json"
        if ready.exists():
            clock.ready = json.loads(ready.read_text())
        if available and clock.reads >= 3:
            stop.set()

    monkeypatch.setattr(serving.subprocess, "Popen", popen)
    monkeypatch.setattr(
        serving.os, "killpg", lambda pid, signum: killed.append((pid, signum))
    )
    monkeypatch.setattr(
        serving.plugin_dev, "_wait_for_tcp_port", lambda *a, **kw: ports.append((a, kw))
    )
    monkeypatch.setattr(serving.redis.Redis, "from_url", lambda *a, **kw: Redis())
    monkeypatch.setattr(serving.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(serving.time, "sleep", sleep)
    return plan, stop, clock, processes, killed, spawned, ports


def test_waits_for_available_worker_then_persists_releases_and_cleans_groups(
    serving, monkeypatch, tmp_path
):
    plan, stop, clock, processes, killed, spawned, ports = fake_service(
        serving, monkeypatch, tmp_path
    )
    serving.run_service(plan, stop_event=stop, startup_timeout=600)
    assert clock.reads == 3
    assert clock.ready["executor_class"] == "earth2-gpu"
    assert clock.ready["workspace"] == str(tmp_path)
    assert clock.ready["registry_path"] == str(tmp_path / "worker-observations.json")
    observations = json.loads((tmp_path / "worker-observations.json").read_text())
    assert observations["registry"]["worker"]["status"] == "available"
    assert observations["releases"] == [["1-0", {"run_id": "parent"}]]
    assert (
        json.loads((tmp_path / "service-state.json").read_text())["status"] == "stopped"
    )
    assert not (tmp_path / "service-ready.json").exists()
    assert all(item["start_new_session"] is True for item in spawned)
    assert len(ports) == 2
    assert all(
        (proc.pid, signal.SIGTERM) in killed and (proc.pid, signal.SIGKILL) in killed
        for proc in processes
    )


def test_release_observations_keep_only_recent_history_in_stream_order(serving):
    entries = [(f"{index}-0", {"run_id": str(index)}) for index in range(1000)]
    calls = []

    class Redis:
        def hgetall(self, key):
            assert key == "gpu:registry"
            return {}

        def xrevrange(self, key, *, count):
            calls.append((key, count))
            return list(reversed(entries))[:count]

    result = serving._observations(Redis())
    assert calls == [("release", 100)]
    assert result["releases"] == entries[-100:]


def test_gpu_readiness_file_is_passed_only_to_the_http_server(
    serving, monkeypatch, tmp_path
):
    monkeypatch.delenv("INFERENCE_READINESS_FILE", raising=False)
    plan, stop, _, _, _, spawned, _ = fake_service(serving, monkeypatch, tmp_path)
    serving.run_service(plan, stop_event=stop)
    assert spawned[1]["env"]["INFERENCE_READINESS_FILE"] == str(
        tmp_path / "service-ready.json"
    )
    assert "INFERENCE_READINESS_FILE" not in spawned[0]["env"]
    assert "INFERENCE_READINESS_FILE" not in spawned[2]["env"]


def test_stale_readiness_is_removed_before_starting_processes(
    serving, monkeypatch, tmp_path
):
    plan, stop, _, _, _, _, _ = fake_service(serving, monkeypatch, tmp_path)
    ready = tmp_path / "service-ready.json"
    ready.write_text('{"status": "ready"}')
    popen = serving.subprocess.Popen

    def spawn(*args, **kwargs):
        assert not ready.exists(), "stale GPU readiness leaked into startup"
        return popen(*args, **kwargs)

    monkeypatch.setattr(serving.subprocess, "Popen", spawn)
    serving.run_service(plan, stop_event=stop)


@pytest.mark.parametrize("failure", ["timeout", "child", "spawn"])
def test_startup_failure_never_signals_ready_and_cleans_started_groups(
    serving, monkeypatch, tmp_path, failure
):
    plan, stop, _, processes, killed, _, _ = fake_service(
        serving, monkeypatch, tmp_path, available=False, failure=failure
    )
    error = {"timeout": TimeoutError, "child": RuntimeError, "spawn": OSError}[failure]
    with pytest.raises(error):
        serving.run_service(plan, stop_event=stop, startup_timeout=2)
    assert not (tmp_path / "service-ready.json").exists()
    assert (
        json.loads((tmp_path / "service-state.json").read_text())["status"] == "failed"
    )
    assert all(
        (proc.pid, signal.SIGTERM) in killed and (proc.pid, signal.SIGKILL) in killed
        for proc in processes
    )


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_main_verifies_source_and_handles_signals_in_unique_sessions(
    serving, monkeypatch, tmp_path, signum
):
    calls, handlers, sessions = [], {}, []
    monkeypatch.setenv("AIFS2ENS_SOURCE_CONTENT_SHA256", "a" * 64)
    monkeypatch.setattr(serving, "verify_source", lambda value: calls.append(value))

    def install(signum, handler):
        handlers[signum] = handler

    monkeypatch.setattr(serving.signal, "signal", install)

    def plan(plugin, *, workspace, port, redis_port):
        calls.append("plan")
        assert port == 8001 and redis_port == 6379
        sessions.append(workspace)
        return {"workspace": workspace}

    def run(plan, *, stop_event, startup_timeout):
        assert startup_timeout == 600
        handlers[signum](signum, None)
        assert stop_event.is_set()

    monkeypatch.setattr(serving.plugin_dev, "build_run_local_plan", plan)
    monkeypatch.setattr(serving, "run_service", run)
    assert serving.main(["--workspace", str(tmp_path)]) == 0
    assert serving.main(["--workspace", str(tmp_path)]) == 0
    assert calls == ["a" * 64, "plan", "a" * 64, "plan"]
    assert sessions[0] != sessions[1]
    pointer = json.loads((tmp_path / "current-service.json").read_text())
    assert pointer["workspace"] == str(sessions[1])
    assert pointer["source_content_sha256"] == "a" * 64


def test_source_mismatch_blocks_service_creation(serving, monkeypatch, tmp_path):
    monkeypatch.setenv("AIFS2ENS_SOURCE_CONTENT_SHA256", "a" * 64)

    def reject(_):
        raise ValueError("image source bytes differ")

    monkeypatch.setattr(serving, "verify_source", reject)
    with pytest.raises(ValueError, match="source bytes"):
        serving.main(["--workspace", str(tmp_path)])
    assert not (tmp_path / "current-service.json").exists()


def test_readiness_tracks_worker_restarts_and_preserves_healthy_busy_workers(
    serving, monkeypatch, tmp_path
):
    plan, stop, clock, _, _, _, _ = fake_service(serving, monkeypatch, tmp_path)
    statuses = ["available", "busy", "warming", None, "failed", "busy", "available"]
    observed = []

    def read(_):
        status = statuses[clock.reads]
        clock.reads += 1
        return {
            "registry": {}
            if status is None
            else {
                "worker": {
                    "stream": plan["execute_registration_streams"][0],
                    "executor_class": "earth2-gpu",
                    "device_kind": "gpu",
                    "status": status,
                }
            },
            "releases": [],
        }

    def sleep(seconds):
        clock.value += seconds
        observed.append(
            (
                (tmp_path / "service-ready.json").exists(),
                json.loads((tmp_path / "service-state.json").read_text())["status"],
            )
        )
        if clock.reads == len(statuses):
            stop.set()

    monkeypatch.setattr(serving, "_observations", read)
    monkeypatch.setattr(serving.time, "sleep", sleep)
    serving.run_service(plan, stop_event=stop, startup_timeout=600)
    assert observed == [
        (True, "ready"),
        (True, "ready"),
        (False, "recovering"),
        (False, "recovering"),
        (False, "recovering"),
        (True, "ready"),
        (True, "ready"),
    ]


def test_worker_recovery_has_a_bounded_deadline(serving, monkeypatch, tmp_path):
    plan, stop, clock, _, _, _, _ = fake_service(serving, monkeypatch, tmp_path)

    def read(_):
        clock.reads += 1
        return {
            "registry": {
                "worker": {
                    "stream": plan["execute_registration_streams"][0],
                    "executor_class": "earth2-gpu",
                    "device_kind": "gpu",
                    "status": "available",
                }
            }
            if clock.reads == 1
            else {},
            "releases": [],
        }

    monkeypatch.setattr(serving, "_observations", read)

    def sleep(seconds):
        clock.value += seconds
        assert clock.value <= 4, "recovery deadline was never enforced"

    monkeypatch.setattr(serving.time, "sleep", sleep)
    with pytest.raises(TimeoutError):
        serving.run_service(plan, stop_event=stop, startup_timeout=2)
    assert clock.value == 3
    assert not (tmp_path / "service-ready.json").exists()
