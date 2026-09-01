# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve-owned coordination for ordinary plugin item execution."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, TypeVar, cast

__all__ = [
    "BatchExecutionCoordinator",
    "RUN_ITEM",
    "max_parallel_items_from_env",
]

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


class _RunItem:
    pass


RUN_ITEM: Final = _RunItem()
"""Sentinel returned by preflight to indicate the item needs executor time."""


def max_parallel_items_from_env(default: int = 1) -> int:
    """Return the configured worker-wide item concurrency limit.

    Defaults to 1 (sequential). Override with
    PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS.
    """
    raw = os.environ.get("PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS")
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS must be a positive integer"
        ) from exc
    if value < 1:
        raise ValueError(
            "PHYSICSNEMO_SERVE_MAX_PARALLEL_ITEMS must be a positive integer"
        )
    return value


class BatchExecutionCoordinator:
    """Execute items through one bounded worker-wide executor.

    execute() returns results in input order. A preflight callback may return
    an immediate result (replay/cancellation) or RUN_ITEM. Ordinary Exception
    instances are isolated per item via handle_exception. Fatal BaseException
    instances are held until all siblings settle, then re-raised.
    """

    def __init__(self, max_parallel_items: int | None = None) -> None:
        if max_parallel_items is None:
            max_parallel_items = max_parallel_items_from_env()
        if isinstance(max_parallel_items, bool) or not isinstance(
            max_parallel_items, int
        ):
            raise ValueError("max_parallel_items must be a positive integer")
        if max_parallel_items < 1:
            raise ValueError("max_parallel_items must be a positive integer")

        self.max_parallel_items = max_parallel_items
        self._lifecycle_lock = threading.Lock()
        self._accepting = True
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max_parallel_items,
                thread_name_prefix="plugin-item",
            )
            if max_parallel_items > 1
            else None
        )

    def execute(
        self,
        items: Sequence[ItemT],
        execute_item: Callable[[ItemT], ResultT],
        *,
        preflight: Callable[[ItemT], ResultT | _RunItem] | None = None,
        handle_exception: Callable[[ItemT, Exception], ResultT] | None = None,
    ) -> list[ResultT]:
        """Execute runnable items and return all outcomes in input order."""
        if not items:
            return []

        missing = object()
        results: list[ResultT | object] = [missing] * len(items)
        submitted: list[tuple[int, ItemT, Future[ResultT]]] = []
        first_fatal: BaseException | None = None
        first_error: Exception | None = None

        with self._lifecycle_lock:
            if not self._accepting:
                raise RuntimeError("batch execution coordinator is closed")

            for index, item in enumerate(items):
                decision: ResultT | _RunItem = RUN_ITEM
                if preflight is not None:
                    try:
                        decision = preflight(item)
                    except Exception as exc:
                        if handle_exception is None:
                            if first_error is None:
                                first_error = exc
                        else:
                            results[index] = handle_exception(item, exc)
                        continue
                    except BaseException as exc:
                        if first_fatal is None:
                            first_fatal = exc
                        continue
                if decision is not RUN_ITEM:
                    results[index] = cast(ResultT, decision)
                    continue

                if self._executor is None:
                    try:
                        results[index] = execute_item(item)
                    except Exception as exc:
                        if handle_exception is None:
                            if first_error is None:
                                first_error = exc
                        else:
                            results[index] = handle_exception(item, exc)
                    except BaseException as exc:
                        if first_fatal is None:
                            first_fatal = exc
                    continue

                future = self._executor.submit(execute_item, item)
                submitted.append((index, item, future))

        for index, item, future in submitted:
            try:
                results[index] = future.result()
            except Exception as exc:
                if handle_exception is None:
                    if first_error is None:
                        first_error = exc
                else:
                    results[index] = handle_exception(item, exc)
            except BaseException as exc:
                if first_fatal is None:
                    first_fatal = exc

        if first_fatal is not None:
            raise first_fatal
        if first_error is not None:
            raise first_error

        if any(r is missing for r in results):  # pragma: no cover
            raise RuntimeError("coordinator failed to produce a result for every item")
        return cast(list[ResultT], results)

    def close(self) -> None:
        """Stop submissions and drain all accepted items before returning."""
        with self._lifecycle_lock:
            if not self._accepting:
                return
            self._accepting = False
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
