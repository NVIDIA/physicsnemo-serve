# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import gc
import inspect
import logging
import os
import traceback
import weakref
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

PayloadT = TypeVar("PayloadT")
InputT = TypeVar("InputT")
FinalStatus = Literal["succeeded", "failed", "cancelled"]
logger = logging.getLogger(__name__)
_HTTP_SESSION_TRACE_ENV = "PHYSICSNEMO_SERVE_TRACE_HTTP_SESSIONS"
_HTTP_SESSION_TRACING_INSTALLED = False


class PluginCancelledError(RuntimeError):
    """Signal that a plugin stopped cooperatively and the run is cancelled."""


def _debug_identity(value: Any) -> str:
    if value is None:
        return "None"
    return f"{type(value).__module__}.{type(value).__qualname__}@0x{id(value):x}"


def _safe_getattr_chain(root: Any, *attrs: str) -> Any | None:
    current = root
    for attr in attrs:
        if current is None:
            return None
        try:
            current = getattr(current, attr)
        except Exception:
            return None
    return current


def _tracked_http_session_labels(root: Any, *, label: str) -> list[str]:
    if root is None:
        return []

    details: list[str] = []
    seen_ids: set[int] = set()
    for suffix, candidate in (
        ("", root),
        (
            "._endpoint.http_session",
            _safe_getattr_chain(root, "_endpoint", "http_session"),
        ),
        (
            "._s3._endpoint.http_session",
            _safe_getattr_chain(root, "_s3", "_endpoint", "http_session"),
        ),
        (
            "._client._endpoint.http_session",
            _safe_getattr_chain(root, "_client", "_endpoint", "http_session"),
        ),
    ):
        if candidate is None:
            continue

        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)

        sessions = getattr(candidate, "_sessions", None)
        if isinstance(sessions, dict):
            session_labels = [
                f"{key!r}:{_debug_identity(session)}"
                for key, session in sessions.items()
            ]
            details.append(
                f"{label}{suffix}={_debug_identity(candidate)} "
                f"sessions=[{', '.join(session_labels)}]"
            )
            continue

        if type(candidate).__module__.startswith("aiohttp"):
            details.append(f"{label}{suffix}={_debug_identity(candidate)}")

    return details


def _tracked_http_session_summary(candidate: Any, session_owner: Any) -> list[str]:
    return [
        *_tracked_http_session_labels(candidate, label="candidate"),
        *_tracked_http_session_labels(session_owner, label="session_owner"),
    ]


def _http_session_tracing_requested() -> bool:
    raw = os.environ.get(_HTTP_SESSION_TRACE_ENV)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _trace_created_stack() -> str:
    stack = traceback.format_stack(limit=16)
    return "".join(stack[:-1]).strip()


def _enable_http_session_tracing() -> None:
    global _HTTP_SESSION_TRACING_INSTALLED
    if _HTTP_SESSION_TRACING_INSTALLED or not _http_session_tracing_requested():
        return
    _HTTP_SESSION_TRACING_INSTALLED = True

    try:
        import aiohttp
    except ImportError:
        logger.debug("HTTP session tracing is enabled, but aiohttp is not installed")
        return

    session_cls = getattr(aiohttp, "ClientSession", None)
    if session_cls is None:
        logger.debug(
            "HTTP session tracing is enabled, but aiohttp.ClientSession is unavailable"
        )
        return

    if not getattr(session_cls, "__physicsnemo_serve_trace_init__", False):
        original_init = session_cls.__init__

        @functools.wraps(original_init)
        def traced_init(self, *args: Any, **kwargs: Any) -> Any:
            result = original_init(self, *args, **kwargs)
            try:
                self._physicsnemo_serve_trace_created_stack = _trace_created_stack()
            except Exception:
                self._physicsnemo_serve_trace_created_stack = "<stack unavailable>"
            logger.debug(
                "aiohttp trace: created session=%s connector=%s owner=%s",
                _debug_identity(self),
                _debug_identity(getattr(self, "connector", None)),
                getattr(self, "_physicsnemo_serve_trace_owner", "unknown"),
            )
            return result

        session_cls.__init__ = traced_init
        session_cls.__physicsnemo_serve_trace_init__ = True

    if not getattr(session_cls, "__physicsnemo_serve_trace_close__", False):
        original_close = session_cls.close

        @functools.wraps(original_close)
        def traced_close(self, *args: Any, **kwargs: Any) -> Any:
            owner = getattr(self, "_physicsnemo_serve_trace_owner", "unknown")
            logger.debug(
                "aiohttp trace: closing session=%s owner=%s connector=%s",
                _debug_identity(self),
                owner,
                _debug_identity(getattr(self, "connector", None)),
            )
            result = original_close(self, *args, **kwargs)
            if inspect.isawaitable(result):

                async def _await_close() -> Any:
                    try:
                        return await result
                    finally:
                        logger.debug(
                            "aiohttp trace: closed session=%s owner=%s closed=%s",
                            _debug_identity(self),
                            owner,
                            getattr(self, "closed", None),
                        )

                return _await_close()

            logger.debug(
                "aiohttp trace: closed session=%s owner=%s closed=%s",
                _debug_identity(self),
                owner,
                getattr(self, "closed", None),
            )
            return result

        session_cls.close = traced_close
        session_cls.__physicsnemo_serve_trace_close__ = True

    original_del = getattr(session_cls, "__del__", None)
    if callable(original_del) and not getattr(
        session_cls, "__physicsnemo_serve_trace_del__", False
    ):

        @functools.wraps(original_del)
        def traced_del(self) -> Any:
            try:
                closed = bool(getattr(self, "closed", True))
            except Exception:
                closed = False

            if not closed:
                owner = getattr(self, "_physicsnemo_serve_trace_owner", "unknown")
                recovered = False
                owner_ref = getattr(self, "_physicsnemo_serve_trace_owner_ref", None)
                owner_obj = None
                if callable(owner_ref):
                    try:
                        owner_obj = owner_ref()
                    except Exception:
                        owner_obj = None
                if owner_obj is not None:
                    try:
                        recovered = _close_aiobotocore_http_session(owner_obj) > 0
                    except Exception:
                        recovered = False
                    if not recovered:
                        try:
                            recovered = bool(getattr(self, "closed", False))
                        except Exception:
                            recovered = False
                    if recovered:
                        try:
                            logger.debug(
                                "aiohttp trace: recovered gc session via owner close "
                                "session=%s owner=%s",
                                _debug_identity(self),
                                owner,
                            )
                        except Exception:
                            pass
                if not recovered:
                    try:
                        recovered = _close_aiohttp_session(self)
                    except Exception:
                        recovered = False
                    if recovered:
                        try:
                            logger.debug(
                                "aiohttp trace: emergency-closed connector for gc session=%s",
                                _debug_identity(self),
                            )
                        except Exception:
                            pass
                if not recovered:
                    try:
                        logger.warning(
                            "aiohttp trace: session garbage collected without explicit close "
                            "session=%s owner=%s created_stack=\n%s",
                            _debug_identity(self),
                            owner,
                            getattr(
                                self,
                                "_physicsnemo_serve_trace_created_stack",
                                "<missing stack>",
                            ),
                        )
                    except Exception:
                        pass
            return original_del(self)

        session_cls.__del__ = traced_del
        session_cls.__physicsnemo_serve_trace_del__ = True

    try:
        from aiobotocore.httpsession import AIOHTTPSession
    except ImportError:
        logger.debug(
            "HTTP session tracing is enabled; aiobotocore.httpsession not available"
        )
        return

    if not getattr(AIOHTTPSession, "__physicsnemo_serve_trace_get_session__", False):
        original_get_session = AIOHTTPSession._get_session

        @functools.wraps(original_get_session)
        async def traced_get_session(self, proxy_url: Any) -> Any:
            session = await original_get_session(self, proxy_url)
            owner = f"{_debug_identity(self)} proxy={proxy_url!r}"
            try:
                session._physicsnemo_serve_trace_owner = owner
                session._physicsnemo_serve_trace_owner_ref = weakref.ref(self)
            except Exception:
                pass
            logger.debug(
                "aiobotocore trace: get_session owner=%s session=%s tracked=%s",
                owner,
                _debug_identity(session),
                _tracked_http_session_labels(self, label="httpsession"),
            )
            return session

        AIOHTTPSession._get_session = traced_get_session
        AIOHTTPSession.__physicsnemo_serve_trace_get_session__ = True

    original_http_close = getattr(AIOHTTPSession, "close", None)
    if callable(original_http_close) and not getattr(
        AIOHTTPSession, "__physicsnemo_serve_trace_close__", False
    ):

        @functools.wraps(original_http_close)
        def traced_http_close(self, *args: Any, **kwargs: Any) -> Any:
            logger.debug(
                "aiobotocore trace: closing httpsession=%s tracked_before=%s",
                _debug_identity(self),
                _tracked_http_session_labels(self, label="httpsession"),
            )
            result = original_http_close(self, *args, **kwargs)
            if inspect.isawaitable(result):

                async def _await_close() -> Any:
                    try:
                        return await result
                    finally:
                        logger.debug(
                            "aiobotocore trace: closed httpsession=%s tracked_after=%s",
                            _debug_identity(self),
                            _tracked_http_session_labels(self, label="httpsession"),
                        )

                return _await_close()

            logger.debug(
                "aiobotocore trace: closed httpsession=%s tracked_after=%s",
                _debug_identity(self),
                _tracked_http_session_labels(self, label="httpsession"),
            )
            return result

        AIOHTTPSession.close = traced_http_close
        AIOHTTPSession.__physicsnemo_serve_trace_close__ = True

    original_http_del = getattr(AIOHTTPSession, "__del__", None)
    if not getattr(AIOHTTPSession, "__physicsnemo_serve_trace_del__", False):

        def traced_http_del(self) -> Any:
            try:
                closed = _close_aiobotocore_http_session(self)
            except Exception:
                closed = 0
            else:
                if closed:
                    try:
                        logger.debug(
                            "aiobotocore trace: recovered gc httpsession=%s closed_sessions=%d",
                            _debug_identity(self),
                            closed,
                        )
                    except Exception:
                        pass
            if callable(original_http_del):
                return original_http_del(self)
            return None

        AIOHTTPSession.__del__ = traced_http_del
        AIOHTTPSession.__physicsnemo_serve_trace_del__ = True


_enable_http_session_tracing()


def _abort_not_requested() -> bool:
    return False


def _close_fsspec_session(
    close_session: Callable[..., Any], *, loop: Any, session_owner: Any
) -> bool:
    try:
        raw_close_session = getattr(close_session, "__func__", close_session)
        signature = inspect.signature(raw_close_session)
    except (TypeError, ValueError):
        raw_close_session = close_session
        signature = None

    if signature is not None:
        call_target = raw_close_session
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if (
            getattr(close_session, "__self__", None) is not None
            and parameters
            and parameters[0].name == "self"
        ):
            call_target = close_session
            parameters = parameters[1:]

        parameter_names = [parameter.name for parameter in parameters]
        if len(parameter_names) >= 2:
            if session_owner is None:
                return False
            call_target(loop, session_owner)
            return True
        if len(parameter_names) == 1:
            if parameter_names[0] in {"session", "s3creator", "s3"}:
                if session_owner is None:
                    return False
                call_target(session_owner)
                return True
            call_target(loop)
            return True
        if not parameter_names:
            call_target()
            return True
        return False

    if session_owner is not None:
        close_session(loop, session_owner)
        return True
    close_session(loop)
    return True


def _run_awaitable_synchronously(awaitable: Any) -> bool:
    if not inspect.isawaitable(awaitable):
        return False

    loop = None
    created_loop = False
    try:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None

        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            created_loop = True

        loop.run_until_complete(awaitable)
        return True
    except Exception:
        close = getattr(awaitable, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        return False
    finally:
        if created_loop and loop is not None:
            loop.close()


def _close_aiohttp_connector(session: Any) -> bool:
    """Synchronously close an aiohttp.ClientSession's connector.

    ``aiohttp.ClientSession.close()`` is a coroutine which cannot be awaited
    from a synchronous context.  Closing the underlying connector directly
    is sufficient: it releases TCP connections and marks the session as
    closed, preventing the ``Unclosed client session`` warning in
    ``__del__``.
    """
    if session is None:
        return False
    try:
        if getattr(session, "closed", True):
            return False
    except Exception:
        return False
    connector = getattr(session, "connector", None)
    if connector is None:
        return False
    try:
        if not getattr(connector, "closed", False):
            result = connector.close()
            if inspect.isawaitable(result):
                _run_awaitable_synchronously(result)
    except Exception:
        pass
    try:
        if getattr(connector, "closed", False):
            if getattr(session, "_connector", None) is connector:
                session._connector = None
            return True
    except Exception:
        pass
    return False


def _close_aiohttp_session(session: Any) -> bool:
    if session is None:
        return False

    try:
        if getattr(session, "closed", True):
            return False
    except Exception:
        return False

    close = getattr(session, "close", None)
    if callable(close):
        try:
            result = close()
        except Exception:
            result = None
        else:
            if inspect.isawaitable(result):
                _run_awaitable_synchronously(result)
            try:
                if getattr(session, "closed", False):
                    return True
            except Exception:
                pass

    return _close_aiohttp_connector(session)


def _iter_aiobotocore_http_sessions(root: Any) -> list[Any]:
    seen_ids: set[int] = set()
    http_sessions: list[Any] = []
    for candidate in (
        root,
        _safe_getattr_chain(root, "_endpoint", "http_session"),
        _safe_getattr_chain(root, "_s3", "_endpoint", "http_session"),
        _safe_getattr_chain(root, "_client", "_endpoint", "http_session"),
    ):
        if candidate is None:
            continue
        if not callable(getattr(candidate, "close", None)):
            continue
        if not (
            hasattr(candidate, "_sessions")
            or hasattr(candidate, "_session")
            or type(candidate).__module__ == "aiobotocore.httpsession"
        ):
            continue

        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        http_sessions.append(candidate)

    return http_sessions


def _close_aiobotocore_http_session(http_session: Any) -> int:
    closed = 0
    closed_ids: set[int] = set()

    def close_tracked_session(session: Any) -> None:
        nonlocal closed
        if session is None:
            return
        session_id = id(session)
        if session_id in closed_ids:
            return
        closed_ids.add(session_id)
        if _close_aiohttp_session(session):
            closed += 1

    direct_session = getattr(http_session, "_session", None)
    if direct_session is not None:
        close_tracked_session(direct_session)

    sessions_map = getattr(http_session, "_sessions", None)
    if isinstance(sessions_map, dict):
        for session in list(sessions_map.values()):
            close_tracked_session(session)

    close = getattr(http_session, "close", None)
    if callable(close):
        try:
            result = close()
        except Exception:
            result = None
        else:
            if inspect.isawaitable(result):
                _run_awaitable_synchronously(result)

    direct_session = getattr(http_session, "_session", None)
    if direct_session is not None:
        close_tracked_session(direct_session)
        try:
            http_session._session = None
        except Exception:
            pass

    sessions_map = getattr(http_session, "_sessions", None)
    if isinstance(sessions_map, dict):
        for session in list(sessions_map.values()):
            close_tracked_session(session)
        try:
            sessions_map.clear()
        except Exception:
            pass

    return closed


def _close_live_aiobotocore_http_sessions() -> int:
    try:
        from aiobotocore.httpsession import AIOHTTPSession
    except ImportError:
        return 0

    closed = 0
    seen_ids: set[int] = set()
    for candidate in gc.get_objects():
        try:
            if not isinstance(candidate, AIOHTTPSession):
                continue
        except Exception:
            continue

        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        closed += _close_aiobotocore_http_session(candidate)

    return closed


def _close_aiobotocore_http_sessions(candidate: Any) -> int:
    """Close aiohttp sessions held by aiobotocore's http_session layer.

    ``s3fs.close_session()`` closes the ``ClientCreatorContext`` but does NOT
    close the ``AIOHTTPSession`` that owns the actual ``aiohttp.ClientSession``.
    Walk the reachable ``AIOHTTPSession`` objects and explicitly close both the
    high-level http_session and any tracked ``aiohttp.ClientSession`` objects.
    """
    closed = 0
    for http_session in _iter_aiobotocore_http_sessions(candidate):
        closed += _close_aiobotocore_http_session(http_session)
    return closed


def _cleanup_fsspec_client(candidate: Any) -> bool:
    close_session = getattr(candidate, "close_session", None)
    if callable(close_session):
        loop = None
        session_owner = None
        for session_source in (
            candidate,
            getattr(close_session, "__self__", None),
        ):
            if session_source is None:
                continue
            if loop is None:
                loop = getattr(session_source, "loop", None)
            # Avoid public lazy properties like `s3`, which may create or reopen
            # clients during teardown.
            for attribute_name in ("_s3creator", "_session", "_s3"):
                session_owner = getattr(session_source, attribute_name, None)
                if session_owner is not None:
                    break
            if session_owner is not None:
                break
        if _close_fsspec_session(close_session, loop=loop, session_owner=session_owner):
            _close_aiobotocore_http_sessions(candidate)
            return True

    close = getattr(candidate, "close", None)
    if callable(close):
        close()
        _close_aiobotocore_http_sessions(candidate)
        return True

    return False


def _device_requests_cuda(device: Any | None) -> bool | None:
    if device is None:
        return None

    normalized = str(getattr(device, "type", device) or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"gpu", "cuda"} or normalized.startswith("cuda"):
        return True
    return False


def cleanup_python_and_torch_runtime(*, device: Any | None = None) -> None:
    """Best-effort Python and Torch runtime cleanup after request-scoped work.

    Runs multiple GC passes to break nested reference cycles, waits for
    in-flight CUDA kernels to finish, then releases the allocator cache.
    """
    _close_cached_fsspec_sessions()
    _close_live_aiobotocore_http_sessions()
    for _ in range(3):
        gc.collect()

    requested_cuda = _device_requests_cuda(device)
    if requested_cuda is False:
        return

    try:
        import torch
    except ImportError:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available") or not cuda.is_available():
        return

    try:
        cuda.synchronize()
    except Exception:
        pass

    ipc_collect = getattr(cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()

    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _close_cached_fsspec_sessions() -> int:
    """Close async sessions on cached fsspec filesystem instances.

    earth2studio model loaders (e.g. ``DLWP.load_default_package()``) create
    ``gcsfs.GCSFileSystem`` instances internally that get cached by fsspec.
    Their ``aiohttp`` sessions are never explicitly closed, causing
    ``Unclosed client session`` errors at GC time.

    This function walks the fsspec instance cache, closes any open async
    sessions, and evicts the closed entries so subsequent requests get fresh
    instances.
    """
    try:
        from fsspec.spec import AbstractFileSystem
    except ImportError:
        return 0

    cache = getattr(AbstractFileSystem, "_cache", None)
    if not isinstance(cache, dict) or not cache:
        return 0

    closed = 0
    stale_keys: list[Any] = []
    for key in list(cache):
        try:
            instance = cache[key]
        except (KeyError, ReferenceError):
            continue

        try:
            logger.debug(
                "earth2 cleanup: inspecting cached fsspec instance key=%r instance=%s tracked=%s",
                key,
                _debug_identity(instance),
                _tracked_http_session_summary(instance, None),
            )
            if _cleanup_fsspec_client(instance):
                closed += 1
                stale_keys.append(key)
        except Exception:
            pass

    for key in stale_keys:
        try:
            del cache[key]
        except (KeyError, ReferenceError):
            pass

    if closed:
        logger.debug("Closed %d cached fsspec session(s)", closed)

    return closed


def cleanup_earth2_runtime_resources(*resources: Any) -> None:
    """Best-effort cleanup for Earth2 objects that own fsspec-style clients."""
    _enable_http_session_tracing()
    cleaned_ids: set[int] = set()
    for resource in resources:
        if resource is None:
            continue

        candidate = getattr(resource, "fs", resource)
        candidate_id = id(candidate)
        if candidate_id in cleaned_ids:
            continue
        cleaned_ids.add(candidate_id)

        try:
            _cleanup_fsspec_client(candidate)
        except Exception:
            logger.exception(
                "Failed to clean up Earth2 runtime resource %s",
                type(resource).__name__,
            )

    _close_cached_fsspec_sessions()
    _close_live_aiobotocore_http_sessions()


@dataclass(frozen=True)
class InputArtifact:
    name: str
    media_type: str
    storage_path: str
    original_filename: str | None = None


@dataclass(frozen=True)
class RawRequest:
    content_type: str
    operation: str
    raw_fields: dict[str, Any]
    input_artifacts: list[InputArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class ResourceProfile:
    executor_class: str
    gpus_required: int
    memory_mb: int
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrepareContext:
    run_id: str
    workflow_id: str
    run_dir: Path
    parent_run_id: str | None = None
    batch_id: str | None = None
    default_resource_profile: ResourceProfile | dict[str, Any] | None = None
    services: Mapping[str, Any] = field(default_factory=dict)
    stage_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareResult:
    inputs: Any
    resource_profile: dict[str, Any] | ResourceProfile | None = None
    prefetch_plan: list[dict[str, Any]] = field(default_factory=list)
    batch_profile: dict[str, Any] | None = None
    fanout_profile: dict[str, Any] | None = None
    fanout_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class OutputRef:
    name: str
    media_type: str
    path: str
    primary: bool = False


class OutputRegistry:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self._outputs: list[OutputRef] = []

    def create(
        self,
        name: str,
        *,
        filename: str,
        media_type: str,
        primary: bool = False,
    ) -> Path:
        target = self.run_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        self.register(name, target, media_type=media_type, primary=primary)
        return target

    def register(
        self,
        name: str,
        path: str | Path,
        *,
        media_type: str,
        primary: bool = False,
    ) -> None:
        normalized_path = Path(path)
        if primary and any(output.primary for output in self._outputs):
            raise ValueError("Only one primary output may be registered per run")
        self._outputs.append(
            OutputRef(
                name=name,
                media_type=media_type,
                path=str(normalized_path),
                primary=primary,
            )
        )

    def registered_outputs(self) -> list[OutputRef]:
        return list(self._outputs)

    def primary_output(self) -> OutputRef | None:
        return next((output for output in self._outputs if output.primary), None)


@dataclass(frozen=True)
class ExecutionInfo:
    run_id: str
    status: FinalStatus
    outputs: list[OutputRef]
    primary_output: OutputRef | None = None
    execution_time_seconds: float | None = None
    published_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PriorResult(Generic[PayloadT]):
    payload: PayloadT
    execution: ExecutionInfo


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    run_dir: Path
    outputs: OutputRegistry
    resource_profile: dict[str, Any] | ResourceProfile | None = None
    batch_info: Mapping[str, Any] | None = None
    fanout_item: Mapping[str, Any] | None = None
    services: Mapping[str, Any] = field(default_factory=dict)
    abort_requested: Callable[[], bool] = _abort_not_requested


@dataclass(frozen=True)
class BatchExecutionContext:
    batch_id: str
    run_dir: Path
    batch_info: Mapping[str, Any] = field(default_factory=dict)
    resource_profile: dict[str, Any] | ResourceProfile | None = None
    services: Mapping[str, Any] = field(default_factory=dict)
    abort_requested: Callable[[], bool] = _abort_not_requested


@dataclass(frozen=True)
class BatchItem(Generic[InputT]):
    index: int
    inputs: InputT
    context: ExecutionContext


@dataclass(frozen=True)
class BatchItemResult(Generic[PayloadT]):
    payload: PayloadT | None = None
    status: FinalStatus = "succeeded"
    error: str | None = None

    @classmethod
    def failed(cls, error: str) -> "BatchItemResult[PayloadT]":
        return cls(status="failed", error=error)

    @classmethod
    def cancelled(cls, error: str | None = None) -> "BatchItemResult[PayloadT]":
        return cls(status="cancelled", error=error)


@dataclass(frozen=True)
class ObjectStorePublishOp:
    output: str = "primary"
    destination_uri: str = ""


@dataclass(frozen=True)
class DatasetExportNetcdfOp:
    output: str = "primary"
    target_output_name: str | None = None
    filename: str | None = None


ResultOp = Union[ObjectStorePublishOp, DatasetExportNetcdfOp]


@dataclass(frozen=True)
class PostprocessContext:
    run_id: str
    run_dir: Path
    outputs: OutputRegistry
    request: RawRequest
    resource_profile: dict[str, Any] | ResourceProfile | None = None
    services: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PostprocessOutcome(Generic[PayloadT]):
    payload: PayloadT
    status: FinalStatus = "succeeded"
    result_ops: list[ResultOp] = field(default_factory=list)


class PluginWorkflow:
    """Base class for class-based PhysicsNeMo Serve plugins.

    New simple plugins can define:
    - `input_model`
    - `form_model` for multipart form fields
    - `output_model`
    - `run(inputs, ctx)`

    and inherit the default `prepare()` / `execute()` behavior. Existing plugins
    may still override `prepare()`, `execute()`, and `postprocess()` directly.
    """

    input_model: Any = None
    form_model: Any = None
    output_model: Any = None
    primary_artifact_name = "primary"
    primary_artifact_media_type = "application/json"
    output_filename = "result.json"
    cache_scope: str | None = None

    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        if self.input_model is None:
            return PrepareResult(inputs=model_to_jsonable(request.raw_fields))

        instance = coerce_model(self.input_model, request.raw_fields, label="input")
        return PrepareResult(inputs=model_to_jsonable(instance))

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs: Any = ctx.get("parameters", {})
        if self.input_model is not None:
            inputs = coerce_model(self.input_model, inputs, label="input")

        if type(self).run is not PluginWorkflow.run:
            exec_ctx = build_execution_context(ctx)
            result = self.run(inputs, exec_ctx)
            _register_companion_outputs(exec_ctx)
            return self._normalize_run_result(result, ctx)

        if type(self).run_batch is not PluginWorkflow.run_batch:
            item_ctx = dict(ctx)
            batch_info = item_ctx.get("batch_info")
            if not isinstance(batch_info, dict):
                batch_info = {
                    "batch_id": str(ctx.get("run_id") or ""),
                    "batch_size": 1,
                }
            else:
                batch_info = dict(batch_info)
                batch_info.setdefault("batch_id", str(ctx.get("run_id") or ""))
                batch_info.setdefault("batch_size", 1)
            item_ctx["batch_info"] = batch_info
            exec_ctx = build_execution_context(item_ctx)
            results = self._normalize_run_batch_results(
                self.run_batch(
                    [
                        BatchItem(
                            index=0,
                            inputs=inputs,
                            context=exec_ctx,
                        )
                    ],
                    build_batch_execution_context(
                        {
                            **ctx,
                            "batch_id": batch_info.get("batch_id"),
                            "batch_info": batch_info,
                        }
                    ),
                ),
                [item_ctx],
            )
            _register_companion_outputs(exec_ctx)
            return results[0]

        raise NotImplementedError(
            "PluginWorkflow.execute(ctx) must be implemented or the workflow "
            "must override run(inputs, ctx) or run_batch(items, ctx)"
        )

    def execute_batch(
        self, items: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if type(self).run_batch is not PluginWorkflow.run_batch:
            batch_items = self._build_batch_items(items)
            results = self._normalize_run_batch_results(
                self.run_batch(
                    batch_items,
                    build_batch_execution_context(ctx),
                ),
                items,
            )
            for bi in batch_items:
                _register_companion_outputs(bi.context)
            return results
        return [self.execute(item_ctx) for item_ctx in items]

    def run(self, inputs: Any, ctx: ExecutionContext) -> Any:
        raise NotImplementedError(
            "PluginWorkflow.run(inputs, ctx) must be implemented when using the "
            "model-driven workflow path"
        )

    def run_batch(
        self, items: list[BatchItem[Any]], ctx: BatchExecutionContext
    ) -> list[Any]:
        raise NotImplementedError(
            "PluginWorkflow.run_batch(items, ctx) may be implemented for typed "
            "batch execution, or plugins can keep using run(inputs, ctx) and let "
            "the default batch adapter call it per item"
        )

    def postprocess(
        self,
        result: PriorResult[Any],
        ctx: PostprocessContext,
    ) -> PostprocessOutcome[Any]:
        status = "succeeded"
        payload: Any = result
        if isinstance(result, PriorResult):
            status = result.execution.status
            payload = result.payload
        return PostprocessOutcome(payload=payload, status=status)

    def warmup(self, ctx: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Optionally load reusable model resources before serving requests."""
        return None

    def cleanup_request(self) -> None:
        """Release request-scoped resources while keeping cached model state."""
        return None

    def cleanup(self) -> None:
        """Release optional per-workflow resources when the workflow is retired."""
        return None

    def _normalize_run_result(self, result: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        if self.output_model is not None:
            result = coerce_model(self.output_model, result, label="output")

        if is_model_instance(result):
            return model_to_jsonable(result)
        elif isinstance(result, dict):
            return model_to_jsonable(result)
        else:
            raise TypeError(
                f"PluginWorkflow.run(inputs, ctx) returned {type(result).__name__}, "
                "expected a payload dataclass, a supported model instance, or a dict"
            )

    def _build_batch_items(
        self, items: list[Mapping[str, Any]]
    ) -> list[BatchItem[Any]]:
        batch_items: list[BatchItem[Any]] = []
        for index, item_ctx in enumerate(items):
            inputs: Any = item_ctx.get("parameters", {})
            if self.input_model is not None:
                inputs = coerce_model(
                    self.input_model, inputs, label=f"items[{index}].input"
                )
            batch_items.append(
                BatchItem(
                    index=index,
                    inputs=inputs,
                    context=build_execution_context(item_ctx),
                )
            )
        return batch_items

    def _normalize_run_batch_results(
        self,
        results: Any,
        items: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(results, list):
            raise TypeError(
                f"PluginWorkflow.run_batch(items, ctx) returned {type(results).__name__}, "
                "expected a list"
            )
        if len(results) != len(items):
            raise ValueError(
                f"PluginWorkflow.run_batch(items, ctx) returned {len(results)} results "
                f"for {len(items)} items"
            )
        return [
            self._normalize_run_batch_result_item(result, item_ctx)
            for result, item_ctx in zip(results, items)
        ]

    def _normalize_run_batch_result_item(
        self,
        result: Any,
        ctx: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, BatchItemResult):
            return self._normalize_run_result(result, dict(ctx))

        if result.payload is None and result.status == "succeeded":
            raise TypeError(
                "BatchItemResult.payload is required when status='succeeded'"
            )

        normalized: dict[str, Any] = {}
        if result.payload is not None:
            normalized = self._normalize_run_result(result.payload, dict(ctx))
        normalized["status"] = result.status
        if result.error is not None:
            normalized["error"] = result.error
        return normalized


def workflow_request_schema(workflow: Any) -> dict[str, Any] | None:
    model = getattr(workflow, "input_model", None)
    if model is None:
        return None
    return schema_for_model(model)


def workflow_form_schema(workflow: Any) -> dict[str, Any] | None:
    model = getattr(workflow, "form_model", None)
    if model is None:
        return None
    return schema_for_model(model)


def workflow_result_schema(workflow: Any) -> dict[str, Any] | None:
    model = getattr(workflow, "output_model", None)
    if model is None:
        return None
    return schema_for_model(model)


def default_run_dir(run_id: Any) -> Path:
    output_root = Path(
        os.environ.get("DEFAULT_OUTPUT_DIR", "/tmp/physicsnemo-serve-plugin-output")
    )
    return output_root / str(run_id)


def build_execution_context(
    ctx: Mapping[str, Any] | ExecutionContext,
) -> ExecutionContext:
    if isinstance(ctx, ExecutionContext):
        return ctx

    outputs = ctx.get("outputs")
    explicit_run_dir = ctx.get("run_dir")
    if explicit_run_dir is not None:
        run_dir = Path(explicit_run_dir)
    elif isinstance(outputs, OutputRegistry):
        run_dir = outputs.run_dir
    else:
        run_dir = default_run_dir(ctx.get("run_id"))
    if not isinstance(outputs, OutputRegistry):
        outputs = OutputRegistry(run_dir)

    services = ctx.get("services", {})
    if not isinstance(services, dict):
        services = {}

    abort_requested = ctx.get("abort_requested")
    if not callable(abort_requested):
        abort_requested = _abort_not_requested

    batch_info = ctx.get("batch_info")
    if not isinstance(batch_info, dict):
        batch_info = None

    fanout_item = ctx.get("fanout_item")
    if not isinstance(fanout_item, dict):
        fanout_item = None

    return ExecutionContext(
        run_id=str(ctx.get("run_id") or ""),
        run_dir=run_dir,
        outputs=outputs,
        resource_profile=ctx.get("resource_profile"),
        batch_info=batch_info,
        fanout_item=fanout_item,
        services=services,
        abort_requested=abort_requested,
    )


def _register_companion_outputs(ctx: ExecutionContext) -> None:
    """Auto-register companion files (metadata JSON, plot PNGs) as artifacts."""
    metadata_path = ctx.run_dir / "forecast_metadata.json"
    if metadata_path.exists():
        ctx.outputs.register(
            "forecast_metadata.json",
            metadata_path,
            media_type="application/json",
        )
    for plot_file in sorted(ctx.run_dir.glob("forecast_plot_*.png")):
        ctx.outputs.register(
            plot_file.name,
            plot_file,
            media_type="image/png",
        )


def build_batch_execution_context(
    ctx: Mapping[str, Any] | BatchExecutionContext,
) -> BatchExecutionContext:
    if isinstance(ctx, BatchExecutionContext):
        return ctx

    explicit_run_dir = ctx.get("run_dir")
    if explicit_run_dir is not None:
        run_dir = Path(explicit_run_dir)
    else:
        run_dir = default_run_dir(ctx.get("batch_id") or ctx.get("run_id"))

    services = ctx.get("services", {})
    if not isinstance(services, dict):
        services = {}

    abort_requested = ctx.get("abort_requested")
    if not callable(abort_requested):
        abort_requested = _abort_not_requested

    batch_info = ctx.get("batch_info")
    if not isinstance(batch_info, dict):
        batch_info = {}

    return BatchExecutionContext(
        batch_id=str(ctx.get("batch_id") or ctx.get("run_id") or ""),
        run_dir=run_dir,
        batch_info=batch_info,
        resource_profile=ctx.get("resource_profile"),
        services=services,
        abort_requested=abort_requested,
    )


def is_model_instance(value: Any) -> bool:
    return is_dataclass(value) or _is_pydantic_model_instance(value)


def model_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return model_to_jsonable(asdict(value))
    if _is_pydantic_model_instance(value):
        return model_to_jsonable(_pydantic_dump(value))
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def coerce_model(model_type: Any, value: Any, *, label: str) -> Any:
    if model_type is Any or model_type is object or model_type is None:
        return value

    if _is_pydantic_model_class(model_type):
        return _coerce_pydantic_model(model_type, value, label=label)

    if is_dataclass(model_type):
        if isinstance(value, model_type):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                f"{label} must be an object for dataclass model '{model_type.__name__}'"
            )

        kwargs: dict[str, Any] = {}
        type_hints = get_type_hints(model_type)
        for field in fields(model_type):
            field_type = type_hints.get(field.name, field.type)
            if field.name in value:
                kwargs[field.name] = coerce_model(
                    field_type,
                    value[field.name],
                    label=f"{label}.{field.name}",
                )
                continue
            if field.default is not MISSING:
                kwargs[field.name] = field.default
                continue
            if field.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[field.name] = field.default_factory()  # type: ignore[misc]
                continue
            raise TypeError(f"{label}.{field.name} is required")

        return model_type(**kwargs)

    origin = get_origin(model_type)
    args = get_args(model_type)

    if origin in {list, tuple}:
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a list")
        item_type = args[0] if args else Any
        return [coerce_model(item_type, item, label=f"{label}[]") for item in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be an object")
        value_type = args[1] if len(args) > 1 else Any
        return {
            key: coerce_model(value_type, item, label=f"{label}.{key}")
            for key, item in value.items()
        }

    if origin is not None and str(origin).endswith("Literal"):
        if value not in args:
            raise TypeError(f"{label} must be one of {list(args)}")
        return value

    if origin is not None and _is_union_origin(origin):
        if value is None and type(None) in args:
            return None
        last_error: Exception | None = None
        for candidate in [item for item in args if item is not type(None)]:
            try:
                return coerce_model(candidate, value, label=label)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise last_error
        return value

    if model_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise TypeError(f"{label} must be a boolean")

    if model_type is int:
        return int(value)
    if model_type is float:
        return float(value)
    if model_type is str:
        return str(value)
    if model_type is Path:
        return Path(value)
    if model_type in {datetime, date}:
        return model_type.fromisoformat(str(value))

    return value


def schema_for_model(model_type: Any) -> dict[str, Any]:
    if _is_pydantic_model_class(model_type):
        schema = _pydantic_schema(model_type)
        if isinstance(schema, dict):
            return schema
        raise TypeError(
            f"Pydantic model schema must be a dict, got {type(schema).__name__}"
        )

    if is_dataclass(model_type):
        required: list[str] = []
        properties: dict[str, Any] = {}
        type_hints = get_type_hints(model_type)
        for field in fields(model_type):
            properties[field.name] = schema_for_type(
                type_hints.get(field.name, field.type)
            )
            if field.default is MISSING and field.default_factory is MISSING:  # type: ignore[attr-defined]
                required.append(field.name)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    raise TypeError(
        f"Unsupported model type for schema generation: {getattr(model_type, '__name__', model_type)}"
    )


def schema_for_type(annotation: Any) -> dict[str, Any]:
    if annotation is Any or annotation is object:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if _is_union_origin(origin):
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1 and len(non_null) != len(args):
            return {"anyOf": [schema_for_type(non_null[0]), {"type": "null"}]}
        return {"anyOf": [schema_for_type(item) for item in args]}

    if origin in {list, tuple}:
        return {"type": "array", "items": schema_for_type(args[0] if args else Any)}

    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": schema_for_type(value_type),
        }

    if origin is not None and str(origin).endswith("Literal"):
        enum_values = list(args)
        schema: dict[str, Any] = {"enum": enum_values}
        if enum_values and all(isinstance(item, str) for item in enum_values):
            schema["type"] = "string"
        elif enum_values and all(isinstance(item, bool) for item in enum_values):
            schema["type"] = "boolean"
        elif enum_values and all(
            isinstance(item, int) and not isinstance(item, bool) for item in enum_values
        ):
            schema["type"] = "integer"
        elif enum_values and all(
            isinstance(item, (int, float)) for item in enum_values
        ):
            schema["type"] = "number"
        return schema

    if _is_pydantic_model_class(annotation):
        return schema_for_model(annotation)
    if is_dataclass(annotation):
        return schema_for_model(annotation)
    if annotation is str or annotation is Path:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is datetime:
        return {"type": "string", "format": "date-time"}
    if annotation is date:
        return {"type": "string", "format": "date"}
    return {}


def _is_union_origin(origin: Any) -> bool:
    try:
        from types import UnionType
    except ImportError:  # pragma: no cover
        UnionType = None  # type: ignore[assignment]
    return origin is not None and (str(origin) == "typing.Union" or origin is UnionType)


def _is_pydantic_model_class(candidate: Any) -> bool:
    return hasattr(candidate, "model_json_schema") or hasattr(candidate, "schema")


def _is_pydantic_model_instance(candidate: Any) -> bool:
    return hasattr(candidate, "model_dump") or hasattr(candidate, "dict")


def _coerce_pydantic_model(model_type: Any, value: Any, *, label: str) -> Any:
    try:
        if hasattr(model_type, "model_validate"):
            return model_type.model_validate(value)
        if hasattr(model_type, "parse_obj"):
            return model_type.parse_obj(value)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"{label} does not conform to model '{model_type.__name__}': {exc}"
        ) from exc
    raise TypeError(f"Unsupported pydantic model type: {model_type}")


def _pydantic_dump(instance: Any) -> Any:
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    if hasattr(instance, "dict"):
        return instance.dict()
    raise TypeError(f"Unsupported pydantic model instance: {type(instance).__name__}")


def _pydantic_schema(model_type: Any) -> Any:
    if hasattr(model_type, "model_json_schema"):
        return model_type.model_json_schema()
    if hasattr(model_type, "schema"):
        return model_type.schema()
    raise TypeError(f"Unsupported pydantic model type: {model_type}")
