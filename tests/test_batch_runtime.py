# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import pytest

from batch_runtime import (
    RUN_ITEM,
    BatchExecutionCoordinator,
    max_parallel_items_from_env,
)


def test_parallel_item_configuration_defaults_to_rollout_safe_one(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS", raising=False)
    assert max_parallel_items_from_env() == 1


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_parallel_item_configuration_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS", value)
    with pytest.raises(ValueError, match="positive integer"):
        max_parallel_items_from_env()


def test_items_overlap_and_results_keep_input_order():
    coordinator = BatchExecutionCoordinator(max_parallel_items=2)
    barrier = threading.Barrier(2)

    def execute(item: int) -> str:
        barrier.wait(timeout=2)
        if item == 1:
            time.sleep(0.03)
        return f"result-{item}"

    try:
        assert coordinator.execute([1, 2], execute) == ["result-1", "result-2"]
    finally:
        coordinator.close()


def test_preflight_results_do_not_consume_executor_slots():
    coordinator = BatchExecutionCoordinator(max_parallel_items=1)
    executed: list[int] = []

    def preflight(item: int):
        if item == 0:
            return "replayed"
        return RUN_ITEM

    def execute(item: int) -> str:
        executed.append(item)
        return f"result-{item}"

    try:
        results = coordinator.execute([0, 1], execute, preflight=preflight)
        assert results == ["replayed", "result-1"]
        assert executed == [1]
    finally:
        coordinator.close()


def test_single_item_execution_stays_on_the_calling_thread():
    coordinator = BatchExecutionCoordinator(max_parallel_items=1)
    calling_thread = threading.current_thread()

    try:
        result = coordinator.execute(
            [1],
            lambda _item: threading.current_thread(),
        )
    finally:
        coordinator.close()

    assert result == [calling_thread]


def test_one_item_exception_does_not_affect_siblings():
    coordinator = BatchExecutionCoordinator(max_parallel_items=2)
    barrier = threading.Barrier(2)

    def execute(item: int) -> str:
        barrier.wait(timeout=2)
        if item == 0:
            raise ValueError("item 0 failed")
        return f"result-{item}"

    def handle_exception(item: int, exc: Exception) -> str:
        return f"failed-{item}"

    try:
        results = coordinator.execute([0, 1], execute, handle_exception=handle_exception)
        assert results == ["failed-0", "result-1"]
    finally:
        coordinator.close()


def test_fatal_exception_is_delayed_until_siblings_settle():
    coordinator = BatchExecutionCoordinator(max_parallel_items=2)
    barrier = threading.Barrier(2)
    sibling_settled = threading.Event()

    def execute(item: int) -> str:
        barrier.wait(timeout=2)
        if item == 0:
            raise KeyboardInterrupt("fatal")
        time.sleep(0.05)
        sibling_settled.set()
        return f"result-{item}"

    try:
        with pytest.raises(KeyboardInterrupt):
            coordinator.execute([0, 1], execute)
        assert sibling_settled.is_set()
    finally:
        coordinator.close()


def test_close_rejects_new_submissions():
    coordinator = BatchExecutionCoordinator(max_parallel_items=1)
    coordinator.close()
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.execute([1], lambda x: x)


def test_empty_items_returns_empty_list():
    coordinator = BatchExecutionCoordinator(max_parallel_items=1)
    try:
        assert coordinator.execute([], lambda x: x) == []
    finally:
        coordinator.close()
