from __future__ import annotations

import copy
import hashlib
import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from qa.scripts import run_native_cfd_http_job as job


SOURCE = "a" * 40
IMAGE = "registry.test/native@sha256:" + "b" * 64


def result(root, run_id="run-first", capture=True):
    attempt = "12345678-1234-1234-1234-123456789abc"
    directory = root / f"native-{hashlib.sha256(run_id.encode()).hexdigest()}" / attempt
    names = job.ARTIFACTS if capture else dict(list(job.ARTIFACTS.items())[:3])
    size = {
        "physical_fields": 32,
        "standardized_fields": 32,
        "fx.f32": 16,
        "embedding.f32": 48,
        "permutation.i64": 16,
        "source_cell_ids.i64": 16,
        "native_metadata": 100,
    }
    native = {
        "attempt": attempt,
        "runtime": "cpp",
        "backend": "aoti",
        "case_id": "run_1",
        "run_seed": 42,
        "permutation_seed": 1615895419752914611,
        "source_sha": SOURCE,
        "release_image": IMAGE,
        "mesh_sha256": job.CASE_PINS["run_1"]["sha256"],
        "geometry_sha256": job.CASE_PINS["run_1"]["geometry_sha256"],
        "metadata": {
            "point_count": 2,
            "initialization_ms": 1.0,
            "session_load_count": 2,
        },
    }
    return {
        "execution": {
            "run_id": run_id,
            "workflow": job.WORKFLOW,
            "status": "succeeded",
            "output_path": str(directory / "physical.bin"),
            "outputs": [
                {
                    "name": name,
                    "storage_path": str(directory / file),
                    "sha256": "0" * 64,
                    "size_bytes": size[name],
                }
                for name, file in names.items()
            ],
        },
        "payload": {"native": native},
    }


@pytest.fixture(autouse=True)
def small_fixture(monkeypatch):
    monkeypatch.setitem(job.CASE_POINT_COUNTS, "run_1", 2)


def validate(value, root, capture=True):
    return job.assert_result(
        value,
        run_id="run-first",
        source_sha=SOURCE,
        image=IMAGE,
        artifact_root=root,
        capture_inputs=capture,
    )


def test_real_api_result_shape_preserves_attempt_provenance_and_full_capture_inventory(
    tmp_path,
):
    directory, outputs, metadata = validate(result(tmp_path), tmp_path)
    assert directory.name == "12345678-1234-1234-1234-123456789abc"
    assert set(outputs) == set(job.ARTIFACTS)
    assert metadata["session_load_count"] == 2
    assert (
        len(validate(result(tmp_path, capture=False), tmp_path, capture=False)[1]) == 3
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["payload"]["native"].update(source_sha="c" * 40),
        lambda value: value["payload"]["native"].update(release_image="mutable:latest"),
        lambda value: value["execution"]["outputs"].pop(),
        lambda value: value["execution"]["outputs"][0].update(
            storage_path="/tmp/outside.bin"
        ),
        lambda value: value["execution"]["outputs"][0].update(size_bytes=16),
        lambda value: value["execution"].update(status="failed"),
    ],
)
def test_native_http_result_rejects_stale_partial_or_escaped_artifacts(
    tmp_path, mutation
):
    value = result(tmp_path)
    mutation(value)
    with pytest.raises(job.FieldContractError):
        validate(value, tmp_path)


def test_post_transport_failure_is_never_retried(monkeypatch):
    calls = []

    def failed(request, **kwargs):
        calls.append(request)
        raise urllib.error.URLError("response lost after potential acceptance")

    monkeypatch.setattr(job.urllib.request, "urlopen", failed)
    with pytest.raises(urllib.error.URLError):
        job.HTTP("http://127.0.0.1:8080").json(
            "/run", body={"parameters": {"case_id": "run_1"}}, expected=202
        )
    assert len(calls) == 1
    assert calls[0].method == "POST"


def test_http_artifact_download_checks_streamed_bytes_without_loading_entire_field(
    monkeypatch, tmp_path
):
    class Response(io.BytesIO):
        def read(self, size=-1):
            assert size == 1024 * 1024
            return super().read(min(size, 3))

    data = b"native-field-bytes"
    monkeypatch.setattr(
        job.urllib.request, "urlopen", lambda *args, **kwargs: Response(data)
    )
    descriptor = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    observed = job.HTTP("http://localhost").download(
        "/artifact", tmp_path / "fields.bin", descriptor
    )
    assert observed == descriptor
    with pytest.raises(job.FieldContractError, match="digest/size"):
        job.HTTP("http://localhost").download(
            "/artifact", tmp_path / "tampered.bin", {**descriptor, "sha256": "0" * 64}
        )


class Redis:
    def __init__(self):
        self.prefix = "test:"
        self.stream = "test:execute.native.transolver.aoti.default.pod.0"
        self.group = "native:grp"
        self.releases = [
            [
                f"{index}-0",
                [
                    "run_id",
                    run_id,
                    "payload",
                    json.dumps(
                        {
                            "run_id": run_id,
                            "resource_id": 0,
                            "memory_mb": 60000,
                            "status": "completed",
                        }
                    ),
                ],
            ]
            for index, run_id in enumerate(("r1", "r2"), 1)
        ]
        self.results = [
            [f"{index}-1", ["run_id", run_id, "payload", "{}"]]
            for index, run_id in enumerate(("r1", "r2"), 1)
        ]
        self.pending, self.active, self.bad_marker = 0, None, False

    def call(self, *args):
        if args[:2] == ("XRANGE", "test:release"):
            return self.releases
        if args[:2] == ("XRANGE", "test:results"):
            return self.results
        if args[0] == "HGET":
            return self.active
        if args[0] == "XLEN":
            return 0
        if args[:2] == ("XINFO", "GROUPS"):
            return [["name", self.group, "pending", self.pending]]
        if args[:2] == ("XRANGE", self.stream):
            return [
                [f"{i}-2", ["run_id", run_id]]
                for i, run_id in enumerate(("r1", "r2"), 1)
            ]
        if args[0] == "HGETALL":
            index = 1 if args[1].endswith("3:1-2") else 2
            values = (
                [f"{index}-0", f"{index}-1"]
                if not self.bad_marker
                else ["wrong", "ids"]
            )
            return ["state", "committed", "ids", json.dumps(values)]
        raise AssertionError(args)


def redis_check(redis):
    return job.assert_redis_completion(
        redis,
        prefix=redis.prefix,
        runs=["r1", "r2"],
        registry={"worker": {"stream": redis.stream, "pid": 123}},
    )


def test_http_qa_checks_exact_release_ack_and_marker_events():
    result = redis_check(Redis())
    assert result == {
        "release_events": 2,
        "result_events": 2,
        "native_pending": 0,
        "active_reserved_memory_mb": 0,
        "worker_pid": 123,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda redis: redis.releases.append(copy.deepcopy(redis.releases[0])),
        lambda redis: setattr(redis, "pending", 1),
        lambda redis: setattr(redis, "active", "60000"),
        lambda redis: setattr(redis, "bad_marker", True),
        lambda redis: redis.releases[0][1].__setitem__(
            3, json.dumps({"resource_id": 0, "memory_mb": 1, "status": "completed"})
        ),
    ],
)
def test_http_qa_rejects_duplicate_release_unacked_delivery_or_leaked_reservation(
    mutation,
):
    redis = Redis()
    mutation(redis)
    with pytest.raises(job.FieldContractError):
        redis_check(redis)


def test_redis_cli_pair_decoding_handles_resp2_and_resp3():
    assert job.pairs(["name", "native", "pending", 0]) == {
        "name": "native",
        "pending": 0,
    }
    assert job.pairs({"pending": 0}) == {"pending": 0}
    with pytest.raises(job.FieldContractError):
        job.pairs(["unpaired"])


def readiness_job(tmp_path, *, exit_code=None):
    runner = object.__new__(job.HTTPJob)
    runner.root = tmp_path
    runner.args = SimpleNamespace(startup_timeout=10)
    runner.stack = SimpleNamespace(poll=lambda: exit_code)
    return runner


def test_readiness_fails_immediately_when_supervisor_exhausts_worker_restarts(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "stack.log").write_text(
        "native-worker exited: SIGSEGV\n"
        "gave up: native-worker entered FATAL state, too many start retries too quickly\n"
    )
    monkeypatch.setattr(
        job.time, "sleep", lambda _: pytest.fail("must not wait after worker FATAL")
    )
    http = SimpleNamespace(json=lambda _: pytest.fail("must detect worker FATAL first"))
    with pytest.raises(job.FieldContractError, match="native-worker.*FATAL"):
        readiness_job(tmp_path).ready(http, None)
    output = capsys.readouterr().out
    assert "NATIVE_CFD_STACK_FAILURE" in output
    assert "SIGSEGV" in output


def test_readiness_failure_prints_only_bounded_tail_when_entrypoint_exits(
    tmp_path, capsys
):
    (tmp_path / "stack.log").write_bytes(b"old-noisy-log\n" * 100000 + b"final error\n")
    with pytest.raises(job.FieldContractError, match="native stack exited"):
        readiness_job(tmp_path, exit_code=1).ready(None, None)
    output = capsys.readouterr().out
    assert "final error" in output
    assert len(output) < 20000


def test_readiness_allows_successful_worker_restart(tmp_path):
    (tmp_path / "stack.log").write_text("exited: native-worker (exit status 1)\n")
    http = SimpleNamespace(
        json=lambda path: (
            {"status": "ready"} if path == "/readyz" else {"readiness": {"ready": True}}
        )
    )
    registry = {
        "worker": {"executor_class": "native.transolver.aoti", "status": "available"}
    }
    redis = SimpleNamespace(registry=lambda: registry)
    assert readiness_job(tmp_path).ready(http, redis) == registry


def test_final_release_and_compilation_provenance_are_separate(tmp_path):
    value = result(tmp_path)
    value["payload"]["native"]["model_build_source_sha"] = "c" * 40
    job.assert_result(
        value,
        run_id="run-first",
        source_sha=SOURCE,
        image=IMAGE,
        artifact_root=tmp_path,
        capture_inputs=True,
        compilation_source_sha="c" * 40,
    )
    with pytest.raises(job.FieldContractError, match="compilation source"):
        job.assert_result(
            value,
            run_id="run-first",
            source_sha=SOURCE,
            image=IMAGE,
            artifact_root=tmp_path,
            capture_inputs=True,
            compilation_source_sha="d" * 40,
        )


def test_http_qa_observes_persistent_native_thread_without_python_subprocess(tmp_path):
    process = tmp_path / "100"
    (process / "task/100").mkdir(parents=True)
    (process / "task/101").mkdir()
    (process / "task/102").mkdir()
    (process / "exe").symlink_to("/app/bin/native-worker")
    (process / "task/100/comm").write_text("native-worker\n")
    (process / "task/101/comm").write_text("native-workflow\n")
    # CUDA/LibTorch helpers can inherit their creator's Linux comm value.
    (process / "task/102/comm").write_text("native-workflow\n")
    (process / "task/100/children").write_text("")
    assert job.native_process_snapshot(100, proc_root=tmp_path) == {
        "pid": 100,
        "executable": "/app/bin/native-worker",
        "native_thread_ids": [101, 102],
    }
    (process / "task/100/children").write_text("102\n")
    with pytest.raises(job.FieldContractError, match="child process"):
        job.native_process_snapshot(100, proc_root=tmp_path)


def test_persistent_native_process_allows_helper_thread_churn():
    initial = {
        "pid": 100,
        "executable": "/app/bin/native-worker",
        "native_thread_ids": [101, 102],
    }
    current = {**initial, "native_thread_ids": [101, 103]}
    assert job.assert_persistent_native_process(initial, current) == [101]


@pytest.mark.parametrize(
    "change",
    [
        {"pid": 200},
        {"executable": "/app/bin/python"},
        {"native_thread_ids": [103, 104]},
    ],
)
def test_persistent_native_process_rejects_restart_or_complete_thread_replacement(
    change,
):
    initial = {
        "pid": 100,
        "executable": "/app/bin/native-worker",
        "native_thread_ids": [101, 102],
    }
    with pytest.raises(job.FieldContractError, match="changed"):
        job.assert_persistent_native_process(initial, {**initial, **change})


def test_registry_heartbeat_updates_preserve_same_resident_worker():
    import copy
    from qa.scripts.run_native_cfd_http_job import assert_same_worker_registry

    before = {
        "slot": {
            "pid": 100,
            "stream": "physicsnemo:execute.slot",
            "worker_instance_id": "original",
            "workflow_id": "workflow",
            "executor_class": "native.transolver.aoti",
            "device_index": 0,
            "heartbeat_at": "before",
            "model_cache": {
                "warmup": {
                    "self_test": {
                        "status": "passed",
                        "models": [{"point_count": 60000}],
                    }
                }
            },
        }
    }
    after = copy.deepcopy(before)
    after["slot"].update(heartbeat_at="after", updated_at="after")
    assert_same_worker_registry(before, after)


@pytest.mark.parametrize("field", ["pid", "stream", "worker_instance_id"])
def test_registry_heartbeat_comparison_still_rejects_worker_replacement(field):
    import copy
    from qa.scripts.run_native_cfd_http_job import assert_same_worker_registry

    before = {
        "slot": {
            "pid": 100,
            "stream": "physicsnemo:execute.slot",
            "worker_instance_id": "original",
        }
    }
    after = copy.deepcopy(before)
    after["slot"][field] = "replacement"
    with pytest.raises(ValueError, match="restarted"):
        assert_same_worker_registry(before, after)


def test_mixed_compilation_sources_must_match_exact_alias_provenance(tmp_path):
    value = result(tmp_path)
    value["payload"]["native"]["model_build_sources"] = {
        "full": "c" * 40,
        "tail": "d" * 40,
    }
    value["payload"]["native"].pop("model_build_source_sha", None)
    with pytest.raises(job.FieldContractError, match="alias compilation"):
        job.assert_result(
            value,
            run_id="run-first",
            source_sha=SOURCE,
            image=IMAGE,
            artifact_root=tmp_path,
            capture_inputs=True,
            model_build_sources={"full": "d" * 40, "tail": "c" * 40},
        )


def test_mixed_compilation_sources_have_no_invented_common_source(tmp_path):
    value = result(tmp_path)
    sources = {"full": "c" * 40, "tail": "d" * 40}
    value["payload"]["native"]["model_build_sources"] = sources
    value["payload"]["native"].pop("model_build_source_sha", None)
    job.assert_result(
        value,
        run_id="run-first",
        source_sha=SOURCE,
        image=IMAGE,
        artifact_root=tmp_path,
        capture_inputs=True,
        model_build_sources=sources,
    )
    value["payload"]["native"]["model_build_source_sha"] = "c" * 40
    with pytest.raises(job.FieldContractError, match="alias compilation"):
        job.assert_result(
            value,
            run_id="run-first",
            source_sha=SOURCE,
            image=IMAGE,
            artifact_root=tmp_path,
            capture_inputs=True,
            model_build_sources=sources,
        )


def test_http_result_binds_selected_tensorrt_backend(tmp_path):
    value = result(tmp_path)
    value["payload"]["native"]["backend"] = "tensorrt"
    kwargs = dict(
        run_id="run-first",
        source_sha=SOURCE,
        image=IMAGE,
        artifact_root=tmp_path,
        capture_inputs=True,
    )
    assert (
        job.assert_result(value, backend="tensorrt", **kwargs)[2]["session_load_count"]
        == 2
    )
    with pytest.raises(job.FieldContractError, match="backend mismatch"):
        job.assert_result(value, backend="aoti", **kwargs)
