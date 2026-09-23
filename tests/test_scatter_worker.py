# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import dataclass

import pytest

import inference_worker
from inference_worker import (
    _MessageLeaseGuard,
    _build_primary_completion,
    _build_release_envelope,
    _plugin_error_retryable,
    _should_persist_run_status_after_execute,
)
from plugin_sdk import PluginRetryableError


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_child_outcomes_go_to_internal_gather_without_public_collect_step(status):
    payload = {
        "run_id": "child",
        "parent_run_id": "parent",
        "round_context": {
            "round_id": "round",
            "child_index": 1,
            "attempt": 2,
            "attempt_token": "token",
        },
        "stage_context": {
            "current_stage_id": "execute",
            "current_phase": "execute",
            "pipeline": [
                {
                    "id": "execute",
                    "phase": "execute",
                    "queue": "execute",
                    "next": "results",
                },
                {"id": "results", "phase": "results", "queue": "results", "next": None},
            ],
        },
    }
    result = {"status": status, "retryable": True}
    stream, forwarded, stage = _build_primary_completion("execute", payload, result)
    assert (stream, stage) == ("collect", "collect")
    assert forwarded["round_context"] == payload["round_context"]
    assert forwarded["result"] == result
    assert "result" not in payload
    assert not _should_persist_run_status_after_execute(payload, result)


@pytest.mark.parametrize(
    "error,retryable",
    [
        (PluginRetryableError("service unavailable"), True),
        (TimeoutError("timeout"), True),
        (ConnectionError("connection lost"), True),
        (ValueError("timeout in input file"), False),
        (FileNotFoundError("immutable input missing"), False),
        (MemoryError("OOM"), False),
        (RuntimeError("connection timeout"), False),
    ],
)
def test_retry_classification_uses_types_not_error_strings(error, retryable):
    assert _plugin_error_retryable(error) is retryable


def test_round_child_release_preserves_scheduler_allocation_identity():
    release = _build_release_envelope(
        "child",
        3,
        4096,
        "succeeded",
        allocation_id="attempt-allocation",
    )

    assert release["allocation_id"] == "attempt-allocation"
    assert "parent_run_id" not in release


def test_message_lease_renewal_checks_the_current_owner():
    class Redis:
        def __init__(self):
            self.responses = iter((1, 0))

        def eval(self, *args):
            assert args[2:] == ("execute", "workers", "worker-a", "1-0")
            return next(self.responses)

    guard = _MessageLeaseGuard(Redis(), "execute", "workers", "worker-a", "1-0")

    assert guard._renew_once()
    assert not guard._renew_once()


def test_lost_message_owner_suppresses_completion(monkeypatch):
    class LostGuard:
        def __init__(self, *_args):
            self.lost = threading.Event()

        def start(self):
            self.lost.set()

        def stop(self):
            pass

    completed = []
    monkeypatch.setattr(inference_worker, "_MessageLeaseGuard", LostGuard)
    monkeypatch.setattr(
        inference_worker,
        "process_job",
        lambda *_args: {"run_id": "child", "status": "succeeded"},
    )
    monkeypatch.setattr(
        inference_worker, "_run_after_request_cleanup", lambda **_kw: None
    )
    monkeypatch.setattr(
        inference_worker, "_log_cuda_memory_snapshot", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        inference_worker, "_reset_torch_cuda_peak_memory_stats", lambda **_kw: None
    )
    monkeypatch.setattr(
        inference_worker, "complete_job", lambda *_args: completed.append(True)
    )

    inference_worker.process_message(
        object(),
        object(),
        "execute",
        {"run_id": "child", "msg_id": "lost-1", "payload": "{}"},
        "worker-a",
    )

    assert completed == []


def test_reclaimed_owner_emits_one_release(monkeypatch):
    class OwnedGuard:
        def __init__(self, *_args):
            self.lost = threading.Event()

        def start(self):
            pass

        def stop(self):
            pass

    class Redis:
        def __init__(self):
            self.added = []
            self.acked = []

        def xadd(self, stream, fields):
            self.added.append((stream, fields))
            return f"{len(self.added)}-0"

        def xack(self, stream, group, message_id):
            self.acked.append((stream, group, message_id))
            return 1

    redis = Redis()
    monkeypatch.setattr(inference_worker, "_MessageLeaseGuard", OwnedGuard)
    monkeypatch.setattr(
        inference_worker,
        "process_job",
        lambda *_args: {"run_id": "child", "status": "succeeded"},
    )
    monkeypatch.setattr(
        inference_worker, "_run_after_request_cleanup", lambda **_kw: None
    )
    monkeypatch.setattr(
        inference_worker, "_log_cuda_memory_snapshot", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        inference_worker, "_reset_torch_cuda_peak_memory_stats", lambda **_kw: None
    )

    inference_worker.process_message(
        object(),
        redis,
        "execute",
        {
            "run_id": "child",
            "msg_id": "reclaimed-1",
            "_reclaimed": True,
            "payload": json.dumps(
                {
                    "resource_id": 0,
                    "memory_mb": 4096,
                    "parent_run_id": "parent",
                    "allocation_id": "allocation-1",
                }
            ),
        },
        "worker-b",
    )

    release_entries = [entry for stream, entry in redis.added if stream == "release"]
    assert len(release_entries) == 1
    release = json.loads(release_entries[0]["payload"])
    assert release["allocation_id"] == "allocation-1"
    assert redis.acked == [("execute", "workers", "reclaimed-1")]


@pytest.mark.asyncio
async def test_async_scatter_completion_uses_normalized_payload(monkeypatch):
    @dataclass
    class Output:
        stream: str
        payload: str
        stage: str

    @dataclass
    class Message:
        id: str = "1-0"
        run_id: str = "parent"
        stream: str = "execute"
        payload: str = "{}"

    class Queue:
        async def forward_many(self, _msg, outputs):
            self.outputs = outputs
            return ["2-0", "2-1"]

    monkeypatch.setattr(inference_worker, "Output", Output)
    queue = Queue()
    normalized_payload = {
        "stage_invocation_id": "invocation-1",
        "resource_id": 0,
        "memory_mb": 4096,
        "stage_context": {
            "pipeline": [
                {
                    "id": "scheduler",
                    "phase": "schedule",
                    "queue": "custom-schedule",
                }
            ]
        },
    }
    scatter = {
        "kind": "scatter",
        "children": [{"operation": "run", "parameters": {}}],
        "child_stage_id": "child",
        "continuation_stage_id": "finish",
        "max_in_flight": 1,
        "run_id": "parent",
        "status": "succeeded",
    }

    await inference_worker.complete_job_async(
        queue, Message(), scatter, normalized_payload
    )

    scheduled = json.loads(queue.outputs[0].payload)
    assert queue.outputs[0].stream == "custom-schedule"
    assert scheduled["stage_invocation_id"] == "invocation-1"
