"""Local service fault probes; GPU scheduling belongs to the outer controller.

The supported recovery topology keeps Redis/API/scheduler alive and restarts one
native worker in a stable slot. A claimed request is not proof of an active CUDA
kernel. These probes retain that distinction in their evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(value):
    return (
        value
        if isinstance(value, dict)
        else dict(zip(value[::2], value[1::2], strict=True))
    )


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def stable_worker_identity(worker):
    keys = (
        "pid",
        "worker_instance_id",
        "registered_at",
        "stream",
        "workflow_id",
        "executor_class",
        "device_index",
        "device_uuid",
    )
    require(
        all(key in worker for key in keys[:4]), "native worker identity is incomplete"
    )
    return {key: copy.deepcopy(worker[key]) for key in keys if key in worker}


def assert_same_worker_registry(before, after):
    if set(before) != set(after) or any(
        stable_worker_identity(before[key]) != stable_worker_identity(after[key])
        for key in before
    ):
        raise ValueError("native worker identity changed")


def _available(worker):
    proof = worker.get("model_cache", {}).get("warmup", {}).get("self_test", {})
    return worker.get("status") == "available" and proof.get("status") == "passed"


def _kill_native(pid):
    require(type(pid) is int and pid > 1, "invalid native process PID")
    executable = os.readlink(f"/proc/{pid}/exe")
    require(
        Path(executable).name == "native-worker",
        "refusing to kill a different executable",
    )
    os.kill(pid, signal.SIGKILL)


def wait_replacement(
    job, http, redis, worker, *, clock=time.monotonic, sleep=time.sleep
):
    deadline = clock() + job.args.startup_timeout
    observed = []
    while clock() < deadline:
        require(
            job.stack.poll() is None, "service supervisor exited during worker recovery"
        )
        require(
            "entered FATAL state" not in job.stack_tail()[-12000:],
            "supervised native replacement failed",
        )
        registry = redis.registry()
        require(
            len(registry) <= 1, "multiple workers appeared in the single-slot topology"
        )
        if registry:
            replacement = next(iter(registry.values()))
            require(
                replacement.get("stream") == worker["stream"],
                "replacement changed the durable execute stream",
            )
            observed.append(
                {
                    key: replacement.get(key)
                    for key in ("pid", "worker_instance_id", "status", "heartbeat_at")
                }
            )
            if replacement.get("worker_instance_id") != worker[
                "worker_instance_id"
            ] and _available(replacement):
                require(
                    replacement.get("pid") != worker["pid"],
                    "replacement did not have a new process identity",
                )
                for key in (
                    "workflow_id",
                    "device_uuid",
                    "device_index",
                    "executor_class",
                ):
                    require(
                        replacement.get(key) == worker.get(key),
                        "replacement changed stable " + key,
                    )
                health = http.json("/readyz")
                readiness = http.json(f"/v1/infer/{job.workflow}/readiness")
                if (
                    health.get("status") == "ready"
                    and readiness.get("readiness", {}).get("ready") is True
                ):
                    return replacement, observed
        sleep(0.1)
    raise TimeoutError("supervised native replacement/readmission deadline exceeded")


def run_idle_crash(job, http, redis, case, worker, *, kill=_kill_native):
    evidence = {
        "status": "running",
        "before": worker,
        "active_request_kill": False,
        "scope": "single supervised worker restart with persistent Redis/API/scheduler",
    }
    path = job.root / "restart.json"
    save(path, evidence)
    try:
        kill(worker["pid"])
        replacement, observed = wait_replacement(job, http, redis, worker)
        evidence.update(after=replacement, observed_registry_states=observed)
        run_id, _ = job.request(http, "after-restart", case)
        evidence.update(status="passed", post_restart_run_id=run_id)
        return replacement
    except BaseException as error:
        evidence.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        save(path, evidence)


def marker_key(stream, group, message):
    # Rust length prefixes count UTF-8 bytes.
    return f"native-attempt:{len(stream.encode())}:{stream}:{len(group.encode())}:{group}:{len(message.encode())}:{message}"


def _events(redis, stream, run_id):
    return [
        (event, json.loads(fields["payload"]))
        for event, raw in redis.call("XRANGE", stream, "-", "+")
        if (fields := pairs(raw)).get("run_id") == run_id
    ]


def active_delivery(redis, stream, run_id, *, prefix="physicsnemo:"):
    deliveries = [
        (message, pairs(raw))
        for message, raw in redis.call("XRANGE", stream, "-", "+")
        if pairs(raw).get("run_id") == run_id
    ]
    require(len(deliveries) <= 1, "scheduler dispatched the same run more than once")
    if not deliveries:
        return None
    groups = [pairs(row) for row in redis.call("XINFO", "GROUPS", stream)]
    require(
        len(groups) == 1, "native execute stream must have exactly one consumer group"
    )
    message, fields = deliveries[0]
    group = groups[0]["name"]
    pending = redis.call("XPENDING", stream, group, message, message, 1)
    marker = pairs(redis.call("HGETALL", marker_key(stream, group, message)))
    if len(pending) != 1 or marker.get("state") != "active":
        return None
    require(
        marker.get("consumer") == pending[0][1] and marker.get("token"),
        "BEGIN marker does not own the pending delivery",
    )
    require(
        not _events(redis, prefix + "release", run_id)
        and not _events(redis, prefix + "results", run_id),
        "request completed before the active fault",
    )
    original = json.loads(fields["payload"])
    require(
        "resource_id" in original and "memory_mb" in original,
        "original scheduler reservation is missing",
    )
    return {
        "stream": stream,
        "group": group,
        "message_id": message,
        "marker": marker,
        "original_reservation": {
            key: original[key] for key in ("resource_id", "memory_mb")
        },
        "run_id": run_id,
    }


def assert_recovered_completion(redis, claim, *, prefix="physicsnemo:"):
    run_id, stream, group, message = (
        claim[key] for key in ("run_id", "stream", "group", "message_id")
    )
    releases, results = (
        _events(redis, prefix + name, run_id) for name in ("release", "results")
    )
    require(
        len(releases) == len(results) == 1,
        "recovered request must have exactly one terminal result and release",
    )
    released = releases[0][1]
    require(
        all(
            released.get(key) == value
            for key, value in claim["original_reservation"].items()
        ),
        "recovery did not release the exact original reservation",
    )
    marker = pairs(redis.call("HGETALL", marker_key(stream, group, message)))
    require(
        marker.get("state") == "committed"
        and json.loads(marker.get("ids", "[]")) == [releases[0][0], results[0][0]],
        "recovered durable marker differs from terminal events",
    )
    require(
        marker.get("consumer") != claim["marker"]["consumer"]
        and marker.get("token") != claim["marker"]["token"],
        "request was not reclaimed by a new consumer and attempt",
    )
    require(
        not redis.call("XPENDING", stream, group, message, message, 1),
        "recovered execute delivery remains pending",
    )
    deliveries = [
        entry
        for entry in redis.call("XRANGE", stream, "-", "+")
        if pairs(entry[1]).get("run_id") == run_id
    ]
    require(
        len(deliveries) == 1 and deliveries[0][0] == message,
        "recovery redispatched instead of reclaiming the original message",
    )
    return {
        "run_id": run_id,
        "source_message_id": message,
        "release_event_id": releases[0][0],
        "results_event_id": results[0][0],
        "original_reservation": claim["original_reservation"],
        "marker": marker,
        "pending": 0,
    }


def run_active_crash(
    job, http, redis, case, worker, *, kill=_kill_native, prefix="physicsnemo:"
):
    """Kill after BEGIN owns the PEL, then await the original single POST only."""
    label = "active-crash"
    evidence = {
        "status": "running",
        "before": worker,
        "active_request_kill": True,
        "scope": "claimed request after durable BEGIN; not evidence of an active CUDA kernel; persistent Redis/API/scheduler",
    }
    path = job.root / "active-restart.json"
    save(path, evidence)
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=1)
    request = pool.submit(job.request, http, label, case)
    try:
        claim = None
        while time.monotonic() < started + job.args.request_timeout:
            require(
                not request.done(),
                "request completed before active-worker fault could be injected",
            )
            submitted = job.root / label / "submit.json"
            if submitted.is_file():
                try:
                    run_id = json.loads(submitted.read_text())["run_id"]
                except json.JSONDecodeError:
                    # A driver may use a plain write rather than atomic rename.
                    # No POST is repeated while its submission file is pending.
                    pass
                else:
                    claim = active_delivery(
                        redis, worker["stream"], run_id, prefix=prefix
                    )
                    if claim:
                        break
            time.sleep(0.02)
        require(
            claim is not None,
            "request did not acquire a durable native BEGIN before fault deadline",
        )
        evidence["claim_before_kill"] = claim
        save(path, evidence)
        kill(worker["pid"])
        replacement, observed = wait_replacement(job, http, redis, worker)
        evidence.update(after=replacement, observed_registry_states=observed)
        run_id, _ = request.result(
            timeout=max(
                0.001, started + job.args.request_timeout + 60 - time.monotonic()
            )
        )
        require(
            run_id == claim["run_id"],
            "observer returned a different request after recovery",
        )
        evidence["completion"] = assert_recovered_completion(
            redis, claim, prefix=prefix
        )
        evidence["status"] = "passed"
        return replacement
    except BaseException as error:
        evidence.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        save(path, evidence)


def reject_startup(job, redis, deployment, *, kind):
    require(kind in {"gpu", "model"}, "unknown negative startup probe")
    before = redis.registry()
    changed = copy.deepcopy(deployment)
    if kind == "gpu":
        if isinstance(changed.get("target"), dict):
            changed["target"]["compilation"]["gpu_product"] = (
                "unsupported-other-product-same-sm"
            )
        else:
            changed["compute_capability"] = [8, 9]
        expected = ("product", "gpu", "device", "h100", "sm90")
    else:
        changed["model_assets"][0]["sha256"] = "0" * 64
        expected = ("sha", "digest", "integrity", "hash")
    path = job.root / (
        "wrong-gpu-config.json" if kind == "gpu" else "tampered-config.json"
    )
    save(path, changed)
    result = subprocess.run(
        ["/app/bin/native-worker", str(path)],
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "RELEASE_IMAGE_DIGEST": job.args.image},
    )
    evidence = {
        "exit_code": result.returncode,
        "stderr": result.stderr[-8000:],
        "registry_before": before,
        "registry_after": redis.registry(),
    }
    save(
        job.root / ("admission-rejection.json" if kind == "gpu" else "tamper.json"),
        evidence,
    )
    require(
        result.returncode != 0
        and any(word in result.stderr.lower() for word in expected),
        "invalid native release was not rejected before readiness",
    )
    assert_same_worker_registry(before, evidence["registry_after"])
    return evidence
