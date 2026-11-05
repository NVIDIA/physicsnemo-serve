# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose the qualified single-GPU service through a deployed HTTP endpoint."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plugin_dev  # noqa: E402
import redis  # noqa: E402
from run_qualification import verify_source, write_json  # noqa: E402

RELEASE_OBSERVATION_LIMIT = 100


def _check_processes(processes):
    for name, process in processes:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Service process {name!r} exited with code {code}")


def _stop_processes(processes, timeout=30):
    # Launchers and their execute workers share the owned process group.
    for _, process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for _, process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # Also remove descendants whose group leader has already exited.
    for _, process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _observations(client):
    return {
        "registry": {
            key: json.loads(value)
            for key, value in client.hgetall("gpu:registry").items()
        },
        "releases": list(
            reversed(client.xrevrange("release", count=RELEASE_OBSERVATION_LIMIT))
        ),
    }


def _workers_available(observations, streams, *, allow_busy=False):
    statuses = {"available", "busy"} if allow_busy else {"available"}
    return bool(streams) and all(
        any(
            entry.get("stream") == stream
            and entry.get("executor_class") == "earth2-gpu"
            and entry.get("device_kind") == "gpu"
            and entry.get("status") in statuses
            for entry in observations["registry"].values()
        )
        for stream in streams
    )


def run_service(plan, *, stop_event, startup_timeout=600):
    workspace = Path(plan["workspace"])
    ready_path = workspace / "service-ready.json"
    state_path = workspace / "service-state.json"
    observation_path = workspace / "worker-observations.json"
    ready_path.unlink(missing_ok=True)
    write_json(state_path, {"status": "starting"})
    processes, logs, client = [], [], None
    status, error = "stopped", None
    try:
        for item in plan["processes"]:
            if stop_event.is_set():
                return
            log = (workspace / "logs" / f"{item['name']}.log").open("w")
            logs.append(log)
            env = {**os.environ, **item.get("env", {})}
            if item["name"] == "inference_server":
                env["INFERENCE_READINESS_FILE"] = str(ready_path)
            process = subprocess.Popen(
                item["argv"],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((item["name"], process))
            if item["name"] in {"redis", "inference_server"}:
                port = plan["redis_port"] if item["name"] == "redis" else plan["port"]
                plugin_dev._wait_for_tcp_port("127.0.0.1", port, timeout_secs=30)
            _check_processes(processes)
        client = redis.Redis.from_url(
            plan["redis_url"],
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        deadline = time.monotonic() + startup_timeout
        ready, previously_ready, previous = False, False, None
        while not stop_event.is_set():
            _check_processes(processes)
            observations = _observations(client)
            if observations != previous:
                write_json(observation_path, observations)
                previous = observations
            available = _workers_available(
                observations,
                plan["execute_registration_streams"],
                allow_busy=previously_ready,
            )
            if ready and not available:
                ready = False
                ready_path.unlink(missing_ok=True)
                write_json(state_path, {"status": "recovering"})
                deadline = time.monotonic() + startup_timeout
            if not ready:
                if available:
                    write_json(
                        ready_path,
                        {
                            "status": "ready",
                            "executor_class": "earth2-gpu",
                            "workspace": str(workspace),
                            "registry_path": str(observation_path),
                            "release_path": str(observation_path),
                            "registration_streams": plan[
                                "execute_registration_streams"
                            ],
                        },
                    )
                    write_json(state_path, {"status": "ready"})
                    print(f"AIFS2ENS_SERVICE_READY {workspace}", flush=True)
                    ready = previously_ready = True
                elif time.monotonic() >= deadline:
                    raise TimeoutError(
                        "AIFS2ENS earth2-gpu worker did not become available"
                    )
            time.sleep(1)
    except BaseException as exc:
        status, error = "failed", {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        ready_path.unlink(missing_ok=True)
        try:
            _stop_processes(processes)
        finally:
            if client is not None:
                client.close()
            for log in logs:
                log.close()
            write_json(state_path, {"status": status, "error": error})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=os.environ.get("AIFS2ENS_SERVICE_ROOT")
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    args = parser.parse_args(argv)
    if args.workspace is None:
        parser.error("--workspace or AIFS2ENS_SERVICE_ROOT is required")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    source_sha = os.environ["AIFS2ENS_SOURCE_CONTENT_SHA256"]
    verify_source(source_sha)
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="service-", dir=root))
    write_json(
        root / "current-service.json",
        {
            "workspace": str(workspace),
            "pid": os.getpid(),
            "source_content_sha256": source_sha,
        },
    )
    plan = plugin_dev.build_run_local_plan(
        REPO / "plugins/earth2-aifs2ens",
        workspace=workspace,
        port=args.port,
        redis_port=6379,
    )
    stop_event = threading.Event()
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for signum in previous:
            signal.signal(signum, lambda *_: stop_event.set())
        run_service(plan, stop_event=stop_event, startup_timeout=600)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
