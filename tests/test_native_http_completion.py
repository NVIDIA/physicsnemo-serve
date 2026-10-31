"""Real Redis regressions for independently consumed results and releases."""

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release_http_job as http_job
import test_native_release_faults as fault_tests

redis = fault_tests.redis


def completed_job(redis, tmp_path):
    stream, group, message = fault_tests.claim_request(redis)
    claim = {"stream": stream, "group": group, "message_id": message}
    fault_tests.complete_request(redis, claim)
    # ResultsRole can publish the API result before Scheduler consumes release.
    redis.call("HSET", "run:run-one", "status", "succeeded")
    redis.call("SET", "result:run-one", json.dumps({"status": "succeeded"}))
    redis.call("HSET", "scheduler:active_reserved_memory_mb", "0", "12345")
    job = object.__new__(http_job.ServiceJob)
    job.root, job.runs = tmp_path, ["run-one"]
    return job, claim


def test_completion_waits_for_scheduler_to_consume_published_release(redis, tmp_path):
    job, _ = completed_job(redis, tmp_path)
    released = threading.Event()

    def release_reservation():
        redis.call("HSET", "scheduler:active_reserved_memory_mb", "0", "0")
        released.set()

    consumer = threading.Timer(0.2, release_reservation)
    consumer.start()
    try:
        job.completion(redis)
        assert released.is_set(), "completion returned before reservation release"
        assert json.loads((tmp_path / "completion.json").read_text())["pending"] == 0
    finally:
        consumer.join(timeout=3)
        assert not consumer.is_alive()


def virtual_time(monkeypatch):
    state = {"elapsed": 0.0, "sleeps": []}

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["elapsed"] += seconds * 100

    monkeypatch.setattr(
        http_job,
        "time",
        SimpleNamespace(monotonic=lambda: state["elapsed"], sleep=sleep),
    )
    return state


def test_completion_rejects_permanent_reservation_at_finite_deadline(
    redis, tmp_path, monkeypatch
):
    job, _ = completed_job(redis, tmp_path)
    clock = virtual_time(monkeypatch)
    with pytest.raises(ValueError, match="reservation"):
        job.completion(redis)
    assert clock["elapsed"] == pytest.approx(30), "reservation settle deadline differs"
    assert clock["sleeps"] and all(0 < delay <= 0.1 for delay in clock["sleeps"])
    assert redis.call("HGET", "scheduler:active_reserved_memory_mb", "0") == "12345"
    assert not (tmp_path / "completion.json").exists()


@pytest.mark.parametrize("failure", ["duplicate", "marker"])
def test_completion_durable_failures_are_not_retried(
    redis, tmp_path, monkeypatch, failure
):
    job, claim = completed_job(redis, tmp_path)
    if failure == "duplicate":
        redis.call(
            "XADD", "physicsnemo:release", "*", "run_id", "run-one", "payload", "{}"
        )
        expected = "duplicate terminal"
    else:
        stream, group, message = (
            claim[key] for key in ("stream", "group", "message_id")
        )
        key = f"native-attempt:{len(stream)}:{stream}:{len(group)}:{group}:{len(message)}:{message}"
        redis.call("HSET", key, "ids", "[]")
        expected = "guarded completion marker"
    clock = virtual_time(monkeypatch)
    with pytest.raises(ValueError, match=expected):
        job.completion(redis)
    assert clock["sleeps"] == [], "durable completion failures must fail immediately"
