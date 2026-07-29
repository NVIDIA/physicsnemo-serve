# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess supervision shared by PhysicsNeMo-CFD plugins."""

from __future__ import annotations

import os
import queue
import selectors
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .safe_files import open_exclusive_file, validated_directory

ABORT_PROBE_WORKER_COUNT = 4
_ABORT_PROBE_INTERVAL_SECONDS = 0.05
_ABORT_PROBE_MAX_AGE_SECONDS = 0.25


@dataclass
class _AbortCallbackRequest:
    callback: Callable[[], bool]
    submitted_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: bool = False
    error: BaseException | None = None


_ABORT_PROBE_QUEUE: queue.Queue[_AbortCallbackRequest] = queue.Queue(
    maxsize=ABORT_PROBE_WORKER_COUNT
)
_ABORT_PROBE_START_LOCK = threading.Lock()
_ABORT_PROBE_ACTIVITY_LOCK = threading.Lock()
_ABORT_PROBE_WORKERS_STARTED = False
_ABORT_PROBE_ACTIVE_WORKERS = 0


def _abort_probe_worker() -> None:
    global _ABORT_PROBE_ACTIVE_WORKERS
    while True:
        request = _ABORT_PROBE_QUEUE.get()
        try:
            if not request.cancelled.is_set():
                with _ABORT_PROBE_ACTIVITY_LOCK:
                    _ABORT_PROBE_ACTIVE_WORKERS += 1
                try:
                    try:
                        request.result = bool(request.callback())
                    except BaseException as exc:
                        request.error = exc
                finally:
                    with _ABORT_PROBE_ACTIVITY_LOCK:
                        _ABORT_PROBE_ACTIVE_WORKERS -= 1
        finally:
            request.done.set()
            _ABORT_PROBE_QUEUE.task_done()
            request = None


def _ensure_abort_probe_workers() -> None:
    global _ABORT_PROBE_WORKERS_STARTED
    with _ABORT_PROBE_START_LOCK:
        if _ABORT_PROBE_WORKERS_STARTED:
            return
        for index in range(ABORT_PROBE_WORKER_COUNT):
            threading.Thread(
                target=_abort_probe_worker,
                name=f"cfd-abort-callback-worker-{index}",
                daemon=True,
            ).start()
        _ABORT_PROBE_WORKERS_STARTED = True


def _submit_abort_callback(
    callback: Callable[[], bool],
) -> _AbortCallbackRequest | None:
    _ensure_abort_probe_workers()
    with _ABORT_PROBE_ACTIVITY_LOCK:
        if _ABORT_PROBE_ACTIVE_WORKERS >= ABORT_PROBE_WORKER_COUNT:
            return None
    request = _AbortCallbackRequest(callback=callback)
    try:
        _ABORT_PROBE_QUEUE.put_nowait(request)
    except queue.Full:
        return None
    return request


class AbortProbe:
    def __init__(self, callback: Callable[[], bool], *, label: str) -> None:
        self._callback = callback
        self._label = label
        self._request: _AbortCallbackRequest | None = None
        self._next_submit_at = 0.0
        self._closed = False

    def check(self) -> bool:
        if self._closed:
            return False
        request = self._request
        if request is not None:
            if not request.done.is_set():
                if (
                    time.monotonic() - request.submitted_at
                    >= _ABORT_PROBE_MAX_AGE_SECONDS
                ):
                    request.cancelled.set()
                    self._request = None
                    return True
                return False
            self._request = None
            if request.error is not None:
                raise RuntimeError(f"{self._label} abort callback failed") from (
                    request.error
                )
            if request.result:
                return True
            self._next_submit_at = time.monotonic() + _ABORT_PROBE_INTERVAL_SECONDS

        now = time.monotonic()
        if now >= self._next_submit_at:
            self._request = _submit_abort_callback(self._callback)
            self._next_submit_at = now + _ABORT_PROBE_INTERVAL_SECONDS
            if self._request is None:
                return True
        return False

    def close(self) -> None:
        self._closed = True
        if self._request is not None:
            self._request.cancelled.set()


@dataclass(frozen=True)
class SupervisedProcessResult:
    returncode: int
    duration_seconds: float
    log_bytes: int
    log_truncated: bool
    timed_out: bool = False
    cancelled: bool = False


def run_supervised_process(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    timeout_seconds: float,
    termination_grace_seconds: float,
    max_log_bytes: int,
    abort_requested: Callable[[], bool],
    env: Mapping[str, str] | None = None,
) -> SupervisedProcessResult:
    """Run one process group, draining output while enforcing bounds and aborts."""
    if timeout_seconds <= 0 or max_log_bytes <= 0:
        raise ValueError("timeout_seconds and max_log_bytes must be positive")
    log = Path(log_path)
    validated_directory(log.parent, label="benchmark log directory")
    log_fd = open_exclusive_file(log)
    started = time.monotonic()
    deadline = started + timeout_seconds
    process: subprocess.Popen[bytes] | None = None
    output_selector: selectors.BaseSelector | None = None
    output_fd = -1
    log_bytes = 0
    log_truncated = False
    drain_error: BaseException | None = None
    abort_probe = AbortProbe(abort_requested, label="benchmark")
    signal_cancelled = threading.Event()
    old_handlers: dict[signal.Signals, Any] = {}

    def handle_cancel_signal(signum: int, frame: Any) -> None:
        signal_cancelled.set()
        previous = old_handlers.get(signal.Signals(signum))
        if (
            callable(previous)
            and previous is not handle_cancel_signal
            and previous is not signal.default_int_handler
        ):
            previous(signum, frame)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_cancel_signal)

    def drain_available_output() -> bool:
        """Drain a bounded amount without waiting; return whether EOF was seen."""
        nonlocal log_bytes, log_truncated
        for _ in range(16):
            try:
                chunk = os.read(output_fd, 64 * 1024)
            except BlockingIOError:
                return False
            except InterruptedError:
                continue
            if not chunk:
                return True
            remaining = max_log_bytes - log_bytes
            if remaining > 0:
                written = chunk[:remaining]
                view = memoryview(written)
                while view:
                    count = os.write(log_fd, view)
                    view = view[count:]
                log_bytes += len(written)
            if len(chunk) > max(remaining, 0):
                log_truncated = True
        return False

    timed_out = False
    cancelled = False
    returncode = -1
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=0,
        )
        assert process.stdout is not None
        output_fd = process.stdout.fileno()
        os.set_blocking(output_fd, False)
        output_selector = selectors.DefaultSelector()
        output_selector.register(output_fd, selectors.EVENT_READ)

        output_open = True
        while True:
            try:
                if output_open and drain_available_output():
                    output_selector.unregister(output_fd)
                    output_open = False
            except BaseException as exc:
                drain_error = exc
                break
            if abort_probe.check() or signal_cancelled.is_set():
                cancelled = True
                break
            if process.poll() is not None:
                if output_open:
                    try:
                        drain_available_output()
                    except BaseException as exc:
                        drain_error = exc
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            wait_seconds = min(0.05, remaining)
            if output_open:
                output_selector.select(wait_seconds)
            else:
                time.sleep(wait_seconds)
    finally:
        abort_probe.close()
        try:
            if process is not None:
                _terminate_process_group(process, termination_grace_seconds)
                returncode = process.wait()
        finally:
            try:
                if output_selector is not None:
                    output_selector.close()
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                os.close(log_fd)
            finally:
                for signum, old_handler in old_handlers.items():
                    signal.signal(signum, old_handler)

    if drain_error is not None:
        raise RuntimeError("benchmark log drain failed") from drain_error

    return SupervisedProcessResult(
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        log_bytes=log_bytes,
        log_truncated=log_truncated,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def supervisor_failure_diagnostics(
    argv: Sequence[str],
    error: Exception,
    *,
    started: float,
    log_path: Path,
    max_log_bytes: int,
) -> dict[str, Any]:
    try:
        log_stat = os.lstat(log_path)
        log_bytes = log_stat.st_size if stat.S_ISREG(log_stat.st_mode) else 0
    except OSError:
        log_bytes = 0
    return {
        "argv": [_bounded_diagnostic_text(token, 2048) for token in argv],
        "returncode": None,
        "duration_seconds": max(0.0, time.monotonic() - started),
        "log_bytes": min(log_bytes, max_log_bytes),
        "log_truncated": log_bytes > max_log_bytes,
        "timed_out": False,
        "cancelled": False,
        "supervisor_error": {
            "type": _bounded_diagnostic_text(type(error).__name__, 128),
            "message": _bounded_diagnostic_text(error, 2048),
        },
    }


def _bounded_diagnostic_text(value: Any, max_characters: int) -> str:
    try:
        text = str(value)[:max_characters]
    except Exception:
        text = "<unavailable>"
    return "".join(
        character
        if character in {"\n", "\t"} or character.isprintable()
        else "\N{REPLACEMENT CHARACTER}"
        for character in text
    )


def _terminate_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
