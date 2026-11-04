"""CPU contract tests for the observers used by actual Lepton fault probes."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).parents[1] / "native-inference/deploy/native_release_faults.py"
SPEC = importlib.util.spec_from_file_location("native_release_faults", MODULE)
faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(faults)


def worker(pid=123, owner="first"):
    return {
        "pid": pid,
        "worker_instance_id": owner,
        "registered_at": "start",
        "stream": "physicsnemo:execute.tensor.slot.0",
        "workflow_id": "tensor",
        "device_index": 0,
        "device_uuid": "GPU-same",
        "status": "available",
        "heartbeat_at": "one",
        "updated_at": "one",
        "model_cache": {"warmup": {"self_test": {"status": "passed"}}},
    }


def test_heartbeat_updates_do_not_change_worker_identity():
    before = {"native:slot": worker()}
    after = copy.deepcopy(before)
    after["native:slot"].update(heartbeat_at="two", updated_at="two")
    faults.assert_same_worker_registry(before, after)
    after["native:slot"]["worker_instance_id"] = "replacement"
    with pytest.raises(ValueError, match="identity"):
        faults.assert_same_worker_registry(before, after)


@pytest.fixture
def redis(tmp_path):
    if not shutil.which("redis-server") or not shutil.which("redis-cli"):
        pytest.skip("real Redis commands unavailable")
    short_root = tempfile.TemporaryDirectory(prefix="pn-redis-", dir="/tmp")
    socket = Path(short_root.name) / "redis.sock"
    process = subprocess.Popen(
        [
            "redis-server",
            "--port",
            "0",
            "--unixsocket",
            str(socket),
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while (
        not socket.exists() and process.poll() is None and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert socket.exists(), "ephemeral Redis did not start"

    class Redis:
        row = worker()

        def call(self, *args):
            output = subprocess.run(
                ["redis-cli", "--json", "-s", str(socket), *map(str, args)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            result = json.loads(output.stdout)
            if isinstance(result, dict) and "error" in result:
                raise ValueError(result["error"])
            return result

        def registry(self):
            return {"native:slot": copy.deepcopy(self.row)}

    yield Redis()
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    short_root.cleanup()


def claim_request(redis, *, begin=True):
    stream = worker()["stream"]
    original = {"resource_id": 0, "memory_mb": 12345}
    message = redis.call(
        "XADD", stream, "*", "run_id", "run-one", "payload", json.dumps(original)
    )
    group = "native:grp"
    redis.call("XGROUP", "CREATE", stream, group, "0")
    redis.call(
        "XREADGROUP", "GROUP", group, "old-consumer", "COUNT", 1, "STREAMS", stream, ">"
    )
    if begin:
        redis.call(
            "HSET",
            faults.marker_key(stream, group, message),
            "state",
            "active",
            "consumer",
            "old-consumer",
            "token",
            "attempt-one",
        )
    return stream, group, message


def complete_request(
    redis, claim, *, amount=12345, consumer="new-consumer", token="attempt-two"
):
    stream, group, message = (claim[key] for key in ("stream", "group", "message_id"))
    redis.call("XCLAIM", stream, group, consumer, 0, message)
    release = redis.call(
        "XADD",
        "physicsnemo:release",
        "*",
        "run_id",
        "run-one",
        "payload",
        json.dumps({"resource_id": 0, "memory_mb": amount}),
    )
    result = redis.call(
        "XADD", "physicsnemo:results", "*", "run_id", "run-one", "payload", "{}"
    )
    redis.call("XACK", stream, group, message)
    redis.call(
        "HSET",
        faults.marker_key(stream, group, message),
        "state",
        "committed",
        "consumer",
        consumer,
        "token",
        token,
        "ids",
        json.dumps([release, result]),
    )


def test_active_probe_requires_begin_owned_by_actual_pending_consumer(redis):
    stream, group, message = claim_request(redis, begin=False)
    assert faults.active_delivery(redis, stream, "run-one") is None
    redis.call(
        "HSET",
        faults.marker_key(stream, group, message),
        "state",
        "active",
        "consumer",
        "wrong-consumer",
        "token",
        "attempt-one",
    )
    with pytest.raises(ValueError, match="does not own"):
        faults.active_delivery(redis, stream, "run-one")
    redis.call(
        "HSET", faults.marker_key(stream, group, message), "consumer", "old-consumer"
    )
    claim = faults.active_delivery(redis, stream, "run-one")
    assert claim["original_reservation"] == {"resource_id": 0, "memory_mb": 12345}


@pytest.mark.parametrize(
    "fault", [None, "amount", "duplicate", "marker", "owner", "redispatch"]
)
def test_recovery_oracle_checks_original_reservation_unique_commit_and_reclaim(
    redis, fault
):
    stream, group, message = claim_request(redis)
    claim = faults.active_delivery(redis, stream, "run-one")
    complete_request(
        redis,
        claim,
        amount=12344 if fault == "amount" else 12345,
        consumer="old-consumer" if fault == "owner" else "new-consumer",
    )
    if fault == "duplicate":
        redis.call(
            "XADD", "physicsnemo:release", "*", "run_id", "run-one", "payload", "{}"
        )
    if fault == "marker":
        redis.call("HSET", faults.marker_key(stream, group, message), "ids", "[]")
    if fault == "redispatch":
        redis.call("XADD", stream, "*", "run_id", "run-one", "payload", "{}")
    if fault:
        with pytest.raises(ValueError):
            faults.assert_recovered_completion(redis, claim)
    else:
        result = faults.assert_recovered_completion(redis, claim)
        assert result["source_message_id"] == message
        assert result["original_reservation"]["memory_mb"] == 12345


class HTTP:
    def json(self, path):
        return (
            {"status": "ready"} if path == "/readyz" else {"readiness": {"ready": True}}
        )


def job(tmp_path):
    return SimpleNamespace(
        root=tmp_path,
        workflow="tensor",
        stack=SimpleNamespace(poll=lambda: None),
        stack_tail=lambda: "",
        args=SimpleNamespace(
            startup_timeout=1,
            request_timeout=2,
            image="registry/release@sha256:" + "1" * 64,
        ),
    )


def test_readmission_waits_past_stale_available_and_requires_self_test(tmp_path):
    before = worker()
    warming = worker(456, "second")
    warming["status"] = "warming"
    ready = worker(456, "second")
    rows = iter([before, warming, ready])
    redis = SimpleNamespace(registry=lambda: {"native:slot": next(rows)})
    after, observed = faults.wait_replacement(
        job(tmp_path), HTTP(), redis, before, sleep=lambda _: None
    )
    assert after["pid"] == 456
    assert [row["status"] for row in observed] == ["available", "warming", "available"]


def test_readmission_rejects_new_pod_queue_and_has_original_deadline(tmp_path):
    changed = worker(456, "second")
    changed["stream"] = "physicsnemo:execute.new-pod"
    with pytest.raises(ValueError, match="durable execute"):
        faults.wait_replacement(
            job(tmp_path),
            HTTP(),
            SimpleNamespace(registry=lambda: {"native:slot": changed}),
            worker(),
        )
    ticks = iter([0, 0, 2])
    with pytest.raises(TimeoutError, match="readmission"):
        faults.wait_replacement(
            job(tmp_path),
            HTTP(),
            SimpleNamespace(registry=lambda: {"native:slot": worker()}),
            worker(),
            clock=lambda: next(ticks),
            sleep=lambda _: None,
        )


def test_active_crash_uses_one_post_and_preserves_same_delivery(redis, tmp_path):
    service = job(tmp_path)
    finished, calls = threading.Event(), []

    def request(http, label, case):
        calls.append(case)
        root = tmp_path / label
        root.mkdir()
        faults.save(root / "submit.json", {"run_id": "run-one"})
        claim_request(redis)
        assert finished.wait(3), "test observer failed to kill claimed worker"
        return "run-one", {"payload": {"y": [1]}}

    service.request = request

    def kill(pid):
        assert pid == 123
        claim = faults.active_delivery(redis, worker()["stream"], "run-one")
        complete_request(redis, claim)
        redis.row = worker(456, "second")
        finished.set()

    after = faults.run_active_crash(
        service, HTTP(), redis, {"x": [1]}, worker(), kill=kill
    )
    assert after["pid"] == 456 and len(calls) == 1
    evidence = json.loads((tmp_path / "active-restart.json").read_text())
    assert evidence["status"] == "passed"
    assert evidence["completion"]["original_reservation"]["memory_mb"] == 12345
    assert evidence["claim_before_kill"]["marker"]["token"] == "attempt-one"


def test_startup_integrity_rejection_cannot_replace_available_worker(
    monkeypatch, tmp_path
):
    service = job(tmp_path)
    row = worker()
    redis = SimpleNamespace(registry=lambda: {"native:slot": copy.deepcopy(row)})
    invoked = []

    def run(command, **kwargs):
        kwargs.clear()
        invoked.append(command)
        assert (
            json.loads(Path(command[-1]).read_text())["model_assets"][0]["sha256"]
            == "0" * 64
        )
        row["heartbeat_at"] = "renewed"
        return SimpleNamespace(returncode=1, stderr="asset SHA256 mismatch")

    monkeypatch.setattr(faults.subprocess, "run", run)
    evidence = faults.reject_startup(
        service,
        redis,
        {"schema_version": 2, "model_assets": [{"sha256": "a" * 64}]},
        kind="model",
    )
    assert evidence["exit_code"] == 1 and len(invoked) == 1
    row["worker_instance_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        faults.assert_same_worker_registry(
            evidence["registry_before"], redis.registry()
        )


@pytest.mark.parametrize("schema", [1, 2])
def test_same_sm_different_sku_probe_preserves_compute_capability(
    monkeypatch, tmp_path, schema
):
    service = job(tmp_path)
    redis = SimpleNamespace(registry=lambda: {"native:slot": worker()})
    config = {
        "schema_version": schema,
        "compute_capability": [9, 0],
        "target": {
            "compilation": {
                "gpu_product": "NVIDIA H100 80GB HBM3",
                "compute_capability": [9, 0],
            }
        },
    }

    def run(command, **kwargs):
        kwargs.clear()  # Never retain inherited environment in assertion locals.
        changed = json.loads(Path(command[-1]).read_text())
        assert (
            changed["target"]["compilation"]["gpu_product"]
            == "unsupported-other-product-same-sm"
        )
        assert changed["target"]["compilation"]["compute_capability"] == [9, 0]
        assert changed["compute_capability"] == [9, 0]
        return SimpleNamespace(returncode=1, stderr="GPU product differs")

    monkeypatch.setattr(faults.subprocess, "run", run)
    faults.reject_startup(service, redis, config, kind="gpu")
