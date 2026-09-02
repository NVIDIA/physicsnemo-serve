#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference Worker - Processes plugin jobs from a single execute stream.

This worker:
1. Listens to a Redis stream assigned by the runtime env launcher
2. Loads manifest-driven plugin implementations
3. Executes plugin hooks with the specified parameters
4. Reports results back to Redis streams

Environment variables (set by the runtime env launcher or local dev kit):
- GPU_STREAM_NAME: Redis stream to read from (for example, execute.python.test)
- GPU_DEVICE_INDEX: CUDA device index (for logging)
- GPU_DEVICE_NAME: GPU model name
- GPU_MEMORY_MB: GPU memory in MB
- CUDA_VISIBLE_DEVICES: Set by launcher to isolate GPU
- RECLAIM_IDLE_MS: Threshold for reclaiming stale messages (default: 900000)
- RECLAIM_INTERVAL_SECS: How often to run reclaim (default: 30)
- PLUGIN_DIR: Comma or colon-separated list of plugin directories
- DEFAULT_OUTPUT_DIR: Base directory for workflow outputs

Lifecycle:
1. Register stream in gpu:registry
2. Load plugin entrypoints
3. Start background reclaimer thread (XAUTOCLAIM for stale messages)
4. Poll stream for jobs (XREADGROUP)
5. Execute workflow for each job
6. Send results to 'results' stream
7. Send release notification to 'release' stream
8. On shutdown, deregister from gpu:registry
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import faulthandler
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from plugin_runtime import (  # noqa: E402
    build_context as build_plugin_context,
    ensure_workflow_enabled,
    get_workflow_instance,
    load_plugin_module as runtime_load_plugin_module,
    resolve_phase_hook,
    resolve_workflow_hook,
    workflow_is_cacheable,
)
from plugin_sdk import PluginCancelledError, cleanup_python_and_torch_runtime  # noqa: E402
from batch_runtime import BatchExecutionCoordinator, RUN_ITEM  # noqa: E402

if TYPE_CHECKING:
    import redis as redis_lib
    from scicomp_rq import QueueManager, Message, Output

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    faulthandler.enable(file=sys.__stderr__, all_threads=True)
except Exception as exc:  # pragma: no cover - best-effort diagnostics
    logger.warning("Failed to enable faulthandler: %s", exc)


# =============================================================================
# Configuration
# =============================================================================

# Reclaim configuration
RECLAIM_IDLE_MS = int(os.environ.get("RECLAIM_IDLE_MS", "900000"))  # 15 minutes
RECLAIM_INTERVAL_SECS = int(os.environ.get("RECLAIM_INTERVAL_SECS", "30"))

# Track in-flight stream messages so the reclaimer cannot re-enter the same
# message on this worker while the original handler is still running.
_IN_FLIGHT_MESSAGE_IDS: set[str] = set()
_IN_FLIGHT_MESSAGE_IDS_LOCK = threading.Lock()

# Output directory
DEFAULT_OUTPUT_DIR = os.environ.get("DEFAULT_OUTPUT_DIR", "/outputs")
REDIS_STREAM_PREFIX = os.environ.get("REDIS_STREAM_PREFIX", "")
DEFAULT_PLUGIN_MANIFEST_NAME = "plugin.yaml"
PARENT_TERMINAL_PREFIX = os.environ.get(
    "PHYSICSNEMO_SERVE_PARENT_TERMINAL_PREFIX", "parent_terminal"
)
ITEM_RUNNER_PATH = SCRIPT_DIR / "inference_item_runner.py"
MAX_CHILD_ERROR_CHARS = 16 * 1024


def _execute_plugin_item_subprocess(
    workflow_name: str,
    run_id: str,
    parameters: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute one ordinary plugin request in its own Python interpreter."""
    request = {
        "workflow_name": workflow_name,
        "run_id": run_id,
        "parameters": parameters,
        "payload": payload,
    }
    child_env = os.environ.copy()
    child_env["PHYSICSNEMO_SERVE_MAX_BATCH_PARALLEL_ITEMS"] = "1"
    completed = subprocess.run(
        [sys.executable, str(ITEM_RUNNER_PATH)],
        input=json.dumps(request),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        check=False,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()
    if completed.returncode != 0:
        message = (
            f"Plugin item process for run {run_id} exited with status "
            f"{completed.returncode}"
        )
        child_error = completed.stderr.strip()
        if child_error:
            message = f"{message}:\n{child_error[-MAX_CHILD_ERROR_CHARS:]}"
        raise RuntimeError(message)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Plugin item process for run {run_id} returned invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise TypeError(
            f"Plugin item process for run {run_id} returned "
            f"{type(result).__name__}, expected object"
        )
    return result


def _torch_cuda_runtime() -> Any | None:
    try:
        torch = __import__("torch")
    except ImportError:
        return None

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available") or not cuda.is_available():
        return None
    return cuda


def _bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _reset_torch_cuda_peak_memory_stats(*, run_id: str, workflow_id: str) -> None:
    cuda = _torch_cuda_runtime()
    if cuda is None:
        return

    reset_peak = getattr(cuda, "reset_peak_memory_stats", None)
    if not callable(reset_peak):
        return

    try:
        reset_peak()
    except Exception as exc:
        logger.debug(
            "Failed to reset CUDA peak memory stats for run %s workflow %s: %s",
            run_id,
            workflow_id,
            exc,
        )
        logger.debug(traceback.format_exc())


def _log_cuda_memory_snapshot(
    stage: str,
    *,
    run_id: str,
    workflow_id: str,
) -> None:
    cuda = _torch_cuda_runtime()
    if cuda is None:
        return

    try:
        allocated_bytes = int(cuda.memory_allocated())
        reserved_bytes = int(cuda.memory_reserved())
        peak_allocated_bytes = int(cuda.max_memory_allocated())
    except Exception as exc:
        logger.debug(
            "Failed to read CUDA memory snapshot for run %s workflow %s stage %s: %s",
            run_id,
            workflow_id,
            stage,
            exc,
        )
        logger.debug(traceback.format_exc())
        return

    logger.info(
        "CUDA memory snapshot: run_id=%s workflow_id=%s stage=%s "
        "allocated=%.1fMiB reserved=%.1fMiB peak_allocated=%.1fMiB %s",
        run_id,
        workflow_id,
        stage,
        _bytes_to_mib(allocated_bytes),
        _bytes_to_mib(reserved_bytes),
        _bytes_to_mib(peak_allocated_bytes),
        worker_identity_tag(),
    )


def _run_after_request_cleanup(*, run_id: str, workflow_id: str) -> None:
    try:
        cleanup_python_and_torch_runtime(device=os.environ.get("GPU_DEVICE_KIND"))
    except Exception as exc:
        logger.debug(
            "Failed to run worker cleanup for run %s workflow %s: %s",
            run_id,
            workflow_id,
            exc,
        )
        logger.debug(traceback.format_exc())


def resolve_output_stream(logical_stream: str) -> str:
    """Return the Redis stream key for an output logical stream name."""
    prefix = REDIS_STREAM_PREFIX.strip()
    if not prefix:
        return logical_stream
    if not prefix.endswith(":"):
        prefix = f"{prefix}:"
    if logical_stream.startswith(prefix):
        return logical_stream
    return f"{prefix}{logical_stream}"


def worker_identity_tag() -> str:
    """Return worker identity for request-level logging."""
    return (
        f"worker_index={os.environ.get('GPU_WORKER_INDEX', 'unknown')} "
        f"pid={os.getpid()}"
    )


def log_fatal_base_exception(
    context: str,
    exc: BaseException,
    *,
    run_id: str | None = None,
) -> None:
    run_suffix = f" run_id={run_id}" if run_id else ""
    logger.critical(
        "%s%s raised fatal %s: %s %s",
        context,
        run_suffix,
        type(exc).__name__,
        exc,
        worker_identity_tag(),
    )
    logger.critical("Fatal traceback follows:\n%s", traceback.format_exc())


def parse_worker_tags(raw_tags: str | None) -> list[str]:
    """Parse and deduplicate worker capability tags from env configuration."""
    if not raw_tags:
        return []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags.split(","):
        tag = raw_tag.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def parent_run_terminal(
    redis_client: "redis_lib.Redis | None", parent_run_id: str | None
) -> bool:
    """Return True when the parent run has already been marked terminal."""
    parent = (parent_run_id or "").strip()
    if not parent or redis_client is None or not hasattr(redis_client, "exists"):
        return False

    terminal_key = f"{PARENT_TERMINAL_PREFIX}:{parent}"
    try:
        return bool(redis_client.exists(terminal_key))
    except Exception as exc:
        logger.debug(
            "Failed to query parent terminal state for %s: %s",
            parent,
            exc,
        )
        return False


def _batch_item_parent_run_id(
    raw_item: dict[str, Any], item_payload: dict[str, Any]
) -> str | None:
    for candidate in (raw_item.get("parent_run_id"), item_payload.get("parent_run_id")):
        parent_run_id = str(candidate or "").strip()
        if parent_run_id:
            return parent_run_id
    return None


def _normalize_batch_outcome_status(raw_status: Any) -> str:
    status = str(raw_status or "succeeded").strip().lower()
    if status in {"success", "succeeded", "completed"}:
        return "succeeded"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    return "failed"


def _batch_result_status(batch_results: list[dict[str, Any]]) -> str:
    statuses = [
        _normalize_batch_outcome_status(entry.get("result", {}).get("status"))
        for entry in batch_results
        if isinstance(entry, dict) and isinstance(entry.get("result"), dict)
    ]
    if not statuses:
        return "failed"
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "cancelled" for status in statuses):
        return "cancelled"
    return "succeeded"


def _cancelled_batch_item_result(
    *,
    run_id: str,
    parent_run_id: str | None,
    execution_time: float,
    batch_info: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "run_id": run_id,
        "status": "cancelled",
        "artifacts": [],
        "output_path": None,
        "execution_time_seconds": execution_time,
        "batch_info": batch_info,
        "skipped_reason": "parent_run_terminal",
    }
    if parent_run_id:
        result["parent_run_id"] = parent_run_id
    return result


def _registered_output_refs(ctx: dict[str, Any]) -> list[Any]:
    outputs = ctx.get("outputs")
    if outputs is None or not hasattr(outputs, "registered_outputs"):
        return []
    return list(outputs.registered_outputs())


def _primary_registered_output_path(ctx: dict[str, Any]) -> str | None:
    outputs = ctx.get("outputs")
    if outputs is None:
        return None

    primary_output = None
    if hasattr(outputs, "primary_output"):
        primary_output = outputs.primary_output()
    if primary_output is not None:
        return str(primary_output.path)

    registered = _registered_output_refs(ctx)
    if not registered:
        return None
    return str(registered[0].path)


def _legacy_artifacts_from_context(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for output in _registered_output_refs(ctx):
        artifact = {
            "name": str(output.name),
            "media_type": str(output.media_type),
            "storage_path": str(output.path),
        }
        if bool(getattr(output, "primary", False)):
            artifact["primary"] = True
        artifacts.append(artifact)
    return artifacts


def _normalize_legacy_execute_result(
    result: dict[str, Any],
    ctx: dict[str, Any],
    execution_time: float,
) -> dict[str, Any]:
    normalized_result = dict(result)

    if not isinstance(normalized_result.get("artifacts"), list):
        normalized_result["artifacts"] = _legacy_artifacts_from_context(ctx)
    elif not normalized_result["artifacts"]:
        normalized_result["artifacts"] = _legacy_artifacts_from_context(ctx)

    if normalized_result.get("output_path") is None:
        normalized_result["output_path"] = _primary_registered_output_path(ctx)

    normalized_result.setdefault("status", "succeeded")
    normalized_result.setdefault("execution_time_seconds", execution_time)
    normalized_result.setdefault("artifacts", [])
    normalized_result.setdefault("output_path", None)
    return normalized_result


def build_worker_metadata(
    *,
    stream_name: str,
    device_index: int,
    device_name: str,
    device_uuid: str,
    memory_mb: int,
    worker_index: int,
    worker_pid: int,
    registry_field: str,
) -> dict[str, Any]:
    """Build the worker registry payload used by the scheduler."""
    executor_class = os.environ.get("GPU_EXECUTOR_CLASS", "").strip() or None
    device_kind = os.environ.get("GPU_DEVICE_KIND", "gpu").strip() or "gpu"
    enabled_workflow_id = os.environ.get(
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", ""
    ).strip()

    return {
        "stream": stream_name,
        "pod": os.environ.get("POD_NAME", "unknown"),
        "namespace": os.environ.get("POD_NAMESPACE", "default"),
        "device_index": device_index,
        "device_name": device_name,
        "device_uuid": device_uuid,
        "memory_mb": memory_mb,
        "device_kind": device_kind,
        "executor_class": executor_class,
        "tags": parse_worker_tags(os.environ.get("GPU_TAGS")),
        "registered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": worker_pid,
        "worker_index": worker_index,
        "registry_field": registry_field,
        "status": "warming" if enabled_workflow_id else "available",
        "model_cache": {
            "schema_version": 1,
            "scope": "process",
            "entries": [],
            "total_entries": 0,
            "warmup": {
                "workflow_id": enabled_workflow_id or None,
                "status": "not_started" if enabled_workflow_id else "skipped",
            },
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


# =============================================================================
# Workflow Executor - Executes manifest-driven plugins
# =============================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelCacheWarmupLock:
    """File lock used to serialize model downloads across worker processes."""

    def __init__(self, lock_name: str) -> None:
        safe_name = "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in lock_name
        )
        lock_dir = Path(
            os.environ.get(
                "PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_DIR",
                "/tmp/physicsnemo-serve-model-cache-locks",
            )
        )
        self.lock_path = lock_dir / f"{safe_name}.lock"
        self.marker_path = lock_dir / f"{safe_name}.prepared"
        self.timeout_secs = float(
            os.environ.get("PHYSICSNEMO_SERVE_MODEL_CACHE_LOCK_TIMEOUT_SECS", "900")
        )
        self._handle: Any | None = None

    def __enter__(self) -> "ModelCacheWarmupLock":
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_secs
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(f"pid={os.getpid()} acquired_at={_utc_now_iso()}\n")
                self._handle.flush()
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for model cache lock {self.lock_path}"
                    )
                time.sleep(0.25)

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        import fcntl

        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
        return False

    def prepared(self) -> bool:
        return self.marker_path.is_file()

    def mark_prepared(self) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "prepared_at": _utc_now_iso(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class WorkerRegistryPublisher:
    """Publishes this worker's mutable cache state back to gpu:registry."""

    def __init__(
        self,
        redis_client: "redis_lib.Redis",
        *,
        stream_name: str,
        registry_field: str,
        metadata: dict[str, Any],
    ) -> None:
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.registry_field = registry_field
        self.metadata = copy.deepcopy(metadata)

    def publish(self) -> None:
        self.metadata.setdefault("model_cache", {})
        self.metadata["model_cache"]["updated_at"] = _utc_now_iso()
        self.redis_client.hset(
            "gpu:registry",
            self.registry_field,
            json.dumps(self.metadata),
        )

    def set_status(self, status: str, *, error: str | None = None) -> None:
        self.metadata["status"] = status
        if error:
            self.metadata["last_error"] = error
        else:
            self.metadata.pop("last_error", None)
        self.publish()

    def set_warmup_status(
        self,
        status: str,
        *,
        workflow_id: str | None = None,
        error: str | None = None,
    ) -> None:
        model_cache = self.metadata.setdefault("model_cache", {})
        warmup = model_cache.setdefault("warmup", {})
        if workflow_id is not None:
            warmup["workflow_id"] = workflow_id
        warmup["status"] = status
        if error:
            warmup["error"] = error
        else:
            warmup.pop("error", None)
        self.publish()

    def update_model_cache(self, entries: list[dict[str, Any]]) -> None:
        model_cache = self.metadata.setdefault("model_cache", {})
        model_cache["entries"] = entries
        model_cache["total_entries"] = len(entries)
        self.publish()


class WorkflowExecutor:
    """
    Executes manifest-driven plugins.

    Discovers plugin entrypoints from configured plugin directories and caches
    loaded modules for reuse.
    """

    def __init__(
        self,
        redis_client: "redis_lib.Redis",
        registry_publisher: WorkerRegistryPublisher | None = None,
        worker_shutdown_event: threading.Event | None = None,
    ):
        self.redis_client = redis_client
        self.registry_publisher = registry_publisher
        self.worker_shutdown_event = (
            worker_shutdown_event
            if worker_shutdown_event is not None
            else threading.Event()
        )
        self._plugin_modules: dict[str, Any] = {}
        self._workflow_cache: dict[str, Any] = {}
        self._workflow_cache_entries: dict[str, dict[str, Any]] = {}
        self._workflow_cache_lock = threading.RLock()
        self._batch_coordinator = BatchExecutionCoordinator()
        self._legacy_batch_warnings: set[str] = set()

    def execute(
        self,
        workflow_name: str,
        run_id: str,
        parameters: dict[str, Any],
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a plugin workflow with the given parameters.

        Args:
            workflow_name: Plugin workflow id
            run_id: Unique execution identifier
            parameters: Workflow parameters

        Returns:
            Dictionary with execution results including status and output_path

        Raises:
            ValueError: If the payload is not a manifest-driven plugin envelope
            Exception: If plugin execution fails
        """
        if (
            payload
            and payload.get("workflow_id")
            and isinstance(payload.get("items"), list)
        ):
            return self._execute_plugin_batch_workflow(
                workflow_name,
                run_id,
                payload,
            )

        if payload and payload.get("workflow_id"):
            return self._execute_plugin_workflow(
                workflow_name,
                run_id,
                parameters,
                payload,
            )
        raise ValueError(
            "Plugin-only worker received a non-plugin payload. "
            "Expected workflow_id in the message envelope."
        )

    def _execute_plugin_workflow(
        self,
        workflow_name: str,
        run_id: str,
        parameters: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a manifest-driven plugin workflow."""
        workflow_id = payload.get("workflow_id", workflow_name)
        start_time = time.time()
        _initialize_execution_state(
            self.redis_client,
            workflow_id,
            run_id,
            parameters,
        )

        try:
            plugin_root, manifest = _resolve_plugin_manifest(workflow_id)
            runtime = manifest.get("runtime", {})
            entrypoint_name = runtime.get("entrypoint") or payload.get(
                "runtime", {}
            ).get("entrypoint")
            if not entrypoint_name:
                raise ValueError(
                    f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
                )

            module = self._load_plugin_module(
                workflow_id, plugin_root / entrypoint_name
            )
            execute_hook = self._resolve_execute_phase_hook(
                module,
                workflow_id=workflow_id,
                manifest=manifest,
                runtime=runtime,
                payload=payload,
                entrypoint_name=entrypoint_name,
                phase="execute",
            )

            operation = payload.get("operation")
            if operation is None:
                operation = (
                    manifest.get("ingress", {})
                    .get("operations", {})
                    .get("default", "run")
                )

            logger.info(
                "Executing plugin workflow: %s (run_id=%s)", workflow_id, run_id
            )
            logger.debug("Plugin parameters: %s", json.dumps(parameters, indent=2))

            payload_for_context = dict(payload)
            services = payload_for_context.get("services", {})
            if not isinstance(services, dict):
                services = {}
            services.setdefault("redis_url", os.environ.get("REDIS_URL"))
            services.setdefault(
                "default_output_dir", os.environ.get("DEFAULT_OUTPUT_DIR")
            )

            service_objects = payload_for_context.get("service_objects", {})
            if not isinstance(service_objects, dict):
                service_objects = {}
            service_objects.setdefault("redis_client", self.redis_client)
            service_objects["worker_shutdown_event"] = self.worker_shutdown_event

            payload_for_context.update(
                {
                    "run_id": run_id,
                    "parent_run_id": payload.get("parent_run_id"),
                    "workflow_id": workflow_id,
                    "operation": operation,
                    "parameters": parameters,
                    "fanout_profile": payload.get("fanout_profile"),
                    "fanout_item": payload.get("fanout_item"),
                    "runtime": runtime,
                    "services": services,
                    "service_objects": service_objects,
                }
            )
            ctx = build_plugin_context(payload_for_context)

            parent_run_id = payload.get("parent_run_id")
            if parent_run_terminal(self.redis_client, parent_run_id):
                execution_time = time.time() - start_time
                _update_execution_state(
                    self.redis_client,
                    workflow_id,
                    run_id,
                    "cancelled",
                    execution_time,
                    output_path=None,
                    error_message="Cancelled because parent run is already terminal",
                )
                return {
                    "status": "cancelled",
                    "workflow": workflow_id,
                    "operation": operation,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "artifacts": [],
                    "output_path": None,
                    "execution_time_seconds": execution_time,
                    "skipped_reason": "parent_run_terminal",
                }

            result = execute_hook(ctx)
            if not isinstance(result, dict):
                raise TypeError(
                    f"Plugin workflow '{workflow_id}' returned {type(result).__name__}, expected dict"
                )

            execution_time = time.time() - start_time
            normalized_result = _normalize_legacy_execute_result(
                result, ctx, execution_time
            )
            output_path = normalized_result.get("output_path")
            normalized_status = normalized_result.get("status")
            execution_state = {
                "succeeded": "completed",
                "cancelled": "cancelled",
            }.get(normalized_status, "failed")
            _update_execution_state(
                self.redis_client,
                workflow_id,
                run_id,
                execution_state,
                execution_time,
                output_path=output_path,
                error_message=normalized_result.get("error"),
            )
            return normalized_result
        except PluginCancelledError as exc:
            execution_time = time.time() - start_time
            logger.info("Plugin workflow %s was cancelled: %s", workflow_id, exc)
            artifacts = (
                _legacy_artifacts_from_context(ctx)
                if "ctx" in locals() and isinstance(ctx, dict)
                else []
            )
            _update_execution_state(
                self.redis_client,
                workflow_id,
                run_id,
                "cancelled",
                execution_time,
                output_path=None,
                error_message=str(exc),
            )
            return {
                "status": "cancelled",
                "output_path": None,
                "execution_time_seconds": execution_time,
                "error": str(exc),
                "artifacts": artifacts,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Plugin workflow %s failed: %s", workflow_id, e)
            logger.debug(traceback.format_exc())
            artifacts = (
                _legacy_artifacts_from_context(ctx)
                if "ctx" in locals() and isinstance(ctx, dict)
                else []
            )
            _update_execution_state(
                self.redis_client,
                workflow_id,
                run_id,
                "failed",
                execution_time,
                output_path=None,
                error_message=str(e),
            )
            return {
                "status": "failed",
                "output_path": None,
                "execution_time_seconds": execution_time,
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "artifacts": artifacts,
            }
        except BaseException as exc:
            log_fatal_base_exception(
                f"Plugin workflow {workflow_id}",
                exc,
                run_id=run_id,
            )
            raise

    def _uses_legacy_batch_only_api(self, module: Any) -> bool:
        """Return True when the module has batch hooks but no run() method."""
        from plugin_sdk import PluginWorkflow

        # Do not invoke build_workflow() solely to inspect its hooks: construction
        # may load expensive models. Legacy batch-only hooks returned exclusively
        # by that factory are therefore not detected here; no current in-repo
        # batch-only plugin relies on that export form.
        workflow_cls = getattr(module, "WORKFLOW", None)
        if workflow_cls is not None:
            if not isinstance(workflow_cls, type):
                # Factory or callable WORKFLOW — cannot inspect without
                # instantiation. Route conservatively through the legacy path.
                return True
            cls = workflow_cls
            # Walk the MRO down to (but not including) PluginWorkflow so that
            # hooks defined in plugin-specific base classes are recognized.
            plugin_mro = [
                c
                for c in cls.__mro__
                if c is not PluginWorkflow
                and c is not object
                and not issubclass(PluginWorkflow, c)
            ]
            has_run_batch = any("run_batch" in c.__dict__ for c in plugin_mro)
            has_execute_batch = any(
                "execute_batch" in c.__dict__ for c in plugin_mro
            ) or callable(getattr(module, "execute_batch", None))
            has_run = any("run" in c.__dict__ for c in plugin_mro)
            has_execute = callable(getattr(module, "execute", None))
            return (has_run_batch or has_execute_batch) and not (has_run or has_execute)
        has_run_batch = callable(getattr(module, "run_batch", None))
        has_execute_batch = callable(getattr(module, "execute_batch", None))
        has_run = callable(getattr(module, "run", None))
        has_execute = callable(getattr(module, "execute", None))
        return (has_run_batch or has_execute_batch) and not (has_run or has_execute)

    def _execute_plugin_batch_workflow(
        self,
        workflow_name: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        workflow_id = payload.get("workflow_id", workflow_name)
        plugin_root, manifest = _resolve_plugin_manifest(workflow_id)
        runtime = manifest.get("runtime", {})
        entrypoint_name = runtime.get("entrypoint") or payload.get("runtime", {}).get(
            "entrypoint"
        )
        if not entrypoint_name:
            raise ValueError(
                f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
            )

        module = self._load_plugin_module(workflow_id, plugin_root / entrypoint_name)

        if self._uses_legacy_batch_only_api(module):
            if workflow_id not in self._legacy_batch_warnings:
                logger.warning(
                    "Plugin workflow %s uses a batch-only API; run_batch()/"
                    "execute_batch() support is deprecated. Implement run() for one item.",
                    workflow_id,
                )
                self._legacy_batch_warnings.add(workflow_id)
            return self._execute_legacy_plugin_batch_workflow(
                workflow_name, batch_id, payload
            )

        operation = payload.get("operation") or (
            manifest.get("ingress", {}).get("operations", {}).get("default", "run")
        )
        services = dict(payload.get("services") or {})
        services.setdefault("redis_url", os.environ.get("REDIS_URL"))
        services.setdefault("default_output_dir", os.environ.get("DEFAULT_OUTPUT_DIR"))
        service_objects = dict(payload.get("service_objects") or {})
        service_objects.setdefault("redis_client", self.redis_client)

        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"Plugin batch workflow '{workflow_id}' is missing items")
        batch_info = _normalize_batch_info(batch_id, payload, len(raw_items))

        items: list[dict[str, Any]] = []
        raw_payloads: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise TypeError(
                    f"Plugin batch workflow '{workflow_id}' items must be objects"
                )
            raw_payload = raw_item.get("payload")
            if not isinstance(raw_payload, dict):
                raise TypeError(
                    f"Plugin batch workflow '{workflow_id}' item payload must be an object"
                )
            item_payload = dict(raw_payload)
            # Batching is a worker implementation detail. The child receives the
            # same envelope it would receive for a single request.
            item_payload.pop("batch_id", None)
            item_payload.pop("batch_info", None)
            item_payload.pop("items", None)
            item_payload.pop("service_objects", None)
            item_run_id = str(
                raw_item.get("run_id") or item_payload.get("run_id") or ""
            ).strip()
            if not item_run_id:
                raise ValueError(
                    f"Plugin batch workflow '{workflow_id}' item is missing run_id"
                )
            # Extract parameters from the original payload before batch metadata
            # is added, so the fallback (using payload itself as parameters) does
            # not pick up framework fields like batch_id or batch_info.
            item_parameters = raw_payload.get(
                "parameters", raw_payload.get("inputs", raw_payload)
            )
            if not isinstance(item_parameters, dict):
                item_parameters = {}
            parent_run_id = _batch_item_parent_run_id(raw_item, item_payload)
            if parent_run_id:
                item_payload["parent_run_id"] = parent_run_id
            item_payload.setdefault("operation", operation)
            item_payload.setdefault("resource_id", payload.get("resource_id"))
            item_payload.setdefault("memory_mb", payload.get("memory_mb"))
            if "services" not in item_payload and isinstance(
                payload.get("services"), dict
            ):
                item_payload["services"] = payload["services"]
            if "resource_profile" not in item_payload and isinstance(
                payload.get("resource_profile"), dict
            ):
                item_payload["resource_profile"] = payload["resource_profile"]
            item_payload.setdefault("workflow_id", workflow_id)
            items.append(
                {
                    "run_id": item_run_id,
                    "parameters": item_parameters,
                    "payload": item_payload,
                    "parent_run_id": parent_run_id,
                    "batch_info": batch_info,
                }
            )
            raw_payloads.append(raw_payload)

        start_time = time.time()

        def preflight(item: dict[str, Any]) -> Any:
            item_run_id = item["run_id"]
            item_parameters = item["parameters"]
            parent_run_id = item["parent_run_id"]
            _initialize_execution_state(
                self.redis_client, workflow_id, item_run_id, item_parameters
            )
            if parent_run_terminal(self.redis_client, parent_run_id):
                elapsed = time.time() - start_time
                _update_execution_state(
                    self.redis_client,
                    workflow_id,
                    item_run_id,
                    "cancelled",
                    elapsed,
                    None,
                    "Cancelled because parent run is already terminal",
                )
                result = _cancelled_batch_item_result(
                    run_id=item_run_id,
                    parent_run_id=parent_run_id,
                    execution_time=elapsed,
                    batch_info=item["batch_info"],
                )
                result["batch_info"] = item["batch_info"]
                return result
            return RUN_ITEM

        def execute_item(item: dict[str, Any]) -> dict[str, Any]:
            item_run_id = item["run_id"]
            item_parameters = item["parameters"]
            item_payload = item["payload"]
            if self._batch_coordinator.max_parallel_items == 1:
                normalized = self._execute_plugin_workflow(
                    workflow_name, item_run_id, item_parameters, item_payload
                )
            else:
                normalized = _execute_plugin_item_subprocess(
                    workflow_name, item_run_id, item_parameters, item_payload
                )
            normalized.setdefault("run_id", item_run_id)
            normalized.setdefault("batch_info", item["batch_info"])
            return normalized

        def handle_exception(item: dict[str, Any], exc: Exception) -> dict[str, Any]:
            elapsed = time.time() - start_time
            item_run_id = item["run_id"]
            _update_execution_state(
                self.redis_client,
                workflow_id,
                item_run_id,
                "failed",
                elapsed,
                None,
                str(exc),
            )
            return {
                "run_id": item_run_id,
                "status": "failed",
                "output_path": None,
                "execution_time_seconds": elapsed,
                "error": str(exc),
                "error_traceback": traceback.format_exc(),
                "batch_info": item["batch_info"],
            }

        item_results = self._batch_coordinator.execute(
            items,
            execute_item,
            preflight=preflight,
            handle_exception=handle_exception,
        )

        batch_results = [
            {
                "run_id": item["run_id"],
                "batch_info": batch_info,
                "payload": raw_payload,
                "result": result,
            }
            for item, raw_payload, result in zip(items, raw_payloads, item_results)
        ]
        return {
            "run_id": batch_id,
            "status": _batch_result_status(batch_results),
            "execution_time_seconds": time.time() - start_time,
            "batch_results": batch_results,
        }

    def _execute_legacy_plugin_batch_workflow(
        self,
        workflow_name: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        workflow_id = payload.get("workflow_id", workflow_name)
        plugin_root, manifest = _resolve_plugin_manifest(workflow_id)
        runtime = manifest.get("runtime", {})
        entrypoint_name = runtime.get("entrypoint") or payload.get("runtime", {}).get(
            "entrypoint"
        )
        if not entrypoint_name:
            raise ValueError(
                f"Plugin workflow '{workflow_id}' is missing runtime.entrypoint"
            )

        module = self._load_plugin_module(workflow_id, plugin_root / entrypoint_name)
        try:
            batch_hook = self._resolve_execute_phase_hook(
                module,
                workflow_id=workflow_id,
                manifest=manifest,
                runtime=runtime,
                payload=payload,
                entrypoint_name=entrypoint_name,
                phase="execute_batch",
            )
        except ValueError:
            batch_hook = None
        operation = payload.get("operation")
        if operation is None:
            operation = (
                manifest.get("ingress", {}).get("operations", {}).get("default", "run")
            )

        services = payload.get("services", {})
        if not isinstance(services, dict):
            services = {}
        services.setdefault("redis_url", os.environ.get("REDIS_URL"))
        services.setdefault("default_output_dir", os.environ.get("DEFAULT_OUTPUT_DIR"))

        service_objects = payload.get("service_objects", {})
        if not isinstance(service_objects, dict):
            service_objects = {}
        service_objects.setdefault("redis_client", self.redis_client)
        service_objects["worker_shutdown_event"] = self.worker_shutdown_event

        batch_payload_for_context = dict(payload)
        batch_payload_for_context.update(
            {
                "run_id": batch_id,
                "batch_id": batch_id,
                "workflow_id": workflow_id,
                "operation": operation,
                "runtime": runtime,
                "services": services,
                "service_objects": service_objects,
            }
        )

        item_contexts: list[dict[str, Any]] = []
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"Plugin batch workflow '{workflow_id}' is missing items")
        batch_info = _normalize_batch_info(batch_id, payload, len(raw_items))
        batch_payload_for_context["batch_info"] = batch_info
        start_time = time.time()
        batch_results_by_index: list[dict[str, Any] | None] = [None] * len(raw_items)
        active_items: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        all_parent_run_ids: set[str] = set()

        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                raise TypeError(
                    f"Plugin batch workflow '{workflow_id}' items must be objects"
                )
            item_payload = raw_item.get("payload")
            if not isinstance(item_payload, dict):
                raise TypeError(
                    f"Plugin batch workflow '{workflow_id}' item payload must be an object"
                )
            item_run_id = str(
                raw_item.get("run_id") or item_payload.get("run_id") or ""
            ).strip()
            if not item_run_id:
                raise ValueError(
                    f"Plugin batch workflow '{workflow_id}' item is missing run_id"
                )
            item_operation = item_payload.get("operation", operation)
            item_parameters = item_payload.get(
                "parameters", item_payload.get("inputs", item_payload)
            )
            item_parent_run_id = _batch_item_parent_run_id(raw_item, item_payload)
            if item_parent_run_id:
                all_parent_run_ids.add(item_parent_run_id)
            _initialize_execution_state(
                self.redis_client,
                workflow_id,
                item_run_id,
                item_parameters if isinstance(item_parameters, dict) else {},
            )

            item_payload_for_context = dict(item_payload)
            item_payload_for_context.update(
                {
                    "run_id": item_run_id,
                    "batch_id": batch_id,
                    "batch_info": batch_info,
                    "workflow_id": workflow_id,
                    "operation": item_operation,
                    "runtime": runtime,
                    "services": services,
                    "service_objects": service_objects,
                }
            )
            if item_parent_run_id:
                item_payload_for_context.setdefault("parent_run_id", item_parent_run_id)
            item_ctx = build_plugin_context(item_payload_for_context)
            if parent_run_terminal(self.redis_client, item_parent_run_id):
                execution_time = time.time() - start_time
                _update_execution_state(
                    self.redis_client,
                    workflow_id,
                    item_run_id,
                    "cancelled",
                    execution_time,
                    output_path=None,
                    error_message="Cancelled because parent run is already terminal",
                )
                batch_results_by_index[index] = {
                    "run_id": item_run_id,
                    "batch_info": batch_info,
                    "payload": raw_item.get("payload"),
                    "result": _cancelled_batch_item_result(
                        run_id=item_run_id,
                        parent_run_id=item_parent_run_id,
                        execution_time=execution_time,
                        batch_info=batch_info,
                    ),
                }
                continue
            item_contexts.append(item_ctx)
            active_items.append((index, item_ctx, raw_item))

        payload_parent_run_id = str(payload.get("parent_run_id") or "").strip()
        if payload_parent_run_id:
            batch_payload_for_context["parent_run_id"] = payload_parent_run_id
        elif len(all_parent_run_ids) == 1:
            batch_payload_for_context["parent_run_id"] = next(iter(all_parent_run_ids))

        if not active_items:
            execution_time = time.time() - start_time
            batch_results = [
                entry for entry in batch_results_by_index if isinstance(entry, dict)
            ]
            response = {
                "run_id": batch_id,
                "status": "cancelled",
                "execution_time_seconds": execution_time,
                "batch_results": batch_results,
                "skipped_reason": "parent_run_terminal",
            }
            if len(all_parent_run_ids) == 1:
                response["parent_run_id"] = next(iter(all_parent_run_ids))
            return response

        if batch_hook is None:
            execution_time = time.time() - start_time
            for index, item_ctx, raw_item in active_items:
                item_payload = dict(raw_item.get("payload") or {})
                item_payload.setdefault("batch_id", batch_id)
                item_payload.setdefault("batch_info", batch_info)
                item_payload.setdefault("resource_id", payload.get("resource_id"))
                item_payload.setdefault("memory_mb", payload.get("memory_mb"))
                if "resource_profile" not in item_payload and isinstance(
                    payload.get("resource_profile"), dict
                ):
                    item_payload["resource_profile"] = payload["resource_profile"]
                item_parameters = item_payload.get(
                    "parameters", item_payload.get("inputs", item_payload)
                )
                if not isinstance(item_parameters, dict):
                    item_parameters = {}
                normalized_result = self._execute_plugin_workflow(
                    workflow_name,
                    item_ctx["run_id"],
                    item_parameters,
                    item_payload,
                )
                normalized_result.setdefault("run_id", item_ctx["run_id"])
                normalized_result.setdefault("batch_info", batch_info)
                batch_results_by_index[index] = {
                    "run_id": item_ctx["run_id"],
                    "batch_info": batch_info,
                    "payload": raw_item.get("payload"),
                    "result": normalized_result,
                }

            batch_results = [
                entry for entry in batch_results_by_index if isinstance(entry, dict)
            ]
            return {
                "run_id": batch_id,
                "status": _batch_result_status(batch_results),
                "execution_time_seconds": execution_time,
                "batch_results": batch_results,
            }

        batch_ctx = build_plugin_context(batch_payload_for_context)
        batch_ctx["items"] = item_contexts

        try:
            try:
                param_count = len(inspect.signature(batch_hook).parameters)
            except (TypeError, ValueError):
                param_count = 2
            raw_results = (
                batch_hook(item_contexts, batch_ctx)
                if param_count >= 2
                else batch_hook(batch_ctx)
            )
            if not isinstance(raw_results, list):
                raise TypeError(
                    f"Plugin batch workflow '{workflow_id}' returned {type(raw_results).__name__}, expected list"
                )
            if len(raw_results) != len(item_contexts):
                raise ValueError(
                    f"Plugin batch workflow '{workflow_id}' returned {len(raw_results)} results for {len(item_contexts)} items"
                )

            execution_time = time.time() - start_time
            for (index, item_ctx, raw_item), item_result in zip(
                active_items, raw_results
            ):
                if not isinstance(item_result, dict):
                    raise TypeError(
                        f"Plugin batch workflow '{workflow_id}' returned non-dict item result"
                    )
                normalized_result = _normalize_legacy_execute_result(
                    item_result,
                    item_ctx,
                    execution_time,
                )
                normalized_result.setdefault("run_id", item_ctx["run_id"])
                normalized_result.setdefault("batch_info", batch_info)
                outcome_status = _normalize_batch_outcome_status(
                    normalized_result.get("status")
                )
                _update_execution_state(
                    self.redis_client,
                    workflow_id,
                    item_ctx["run_id"],
                    "completed" if outcome_status == "succeeded" else outcome_status,
                    execution_time,
                    normalized_result.get("output_path"),
                    normalized_result.get("error"),
                )
                batch_results_by_index[index] = {
                    "run_id": item_ctx["run_id"],
                    "batch_info": batch_info,
                    "payload": raw_item.get("payload"),
                    "result": normalized_result,
                }

            batch_results = [
                entry for entry in batch_results_by_index if isinstance(entry, dict)
            ]
            return {
                "run_id": batch_id,
                "status": _batch_result_status(batch_results),
                "execution_time_seconds": execution_time,
                "batch_results": batch_results,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Plugin batch workflow %s failed: %s", workflow_id, e)
            logger.debug(traceback.format_exc())
            for index, item_ctx, raw_item in active_items:
                failed_result = {
                    "run_id": item_ctx["run_id"],
                    "status": "failed",
                    "output_path": None,
                    "execution_time_seconds": execution_time,
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "batch_info": batch_info,
                }
                _update_execution_state(
                    self.redis_client,
                    workflow_id,
                    item_ctx["run_id"],
                    "failed",
                    execution_time,
                    None,
                    str(e),
                )
                batch_results_by_index[index] = {
                    "run_id": item_ctx["run_id"],
                    "batch_info": batch_info,
                    "payload": raw_item.get("payload"),
                    "result": failed_result,
                }
            batch_results = [
                entry for entry in batch_results_by_index if isinstance(entry, dict)
            ]
            return {
                "run_id": batch_id,
                "status": _batch_result_status(batch_results),
                "execution_time_seconds": execution_time,
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "batch_results": batch_results,
            }
        except BaseException as exc:
            log_fatal_base_exception(
                f"Plugin batch workflow {workflow_id}",
                exc,
                run_id=batch_id,
            )
            raise

    def _load_plugin_module(self, workflow_id: str, entrypoint_path: Path) -> Any:
        """Load and cache a plugin module from its runtime entrypoint."""
        cached = self._plugin_modules.get(workflow_id)
        if cached is not None:
            return cached

        module = runtime_load_plugin_module(
            workflow_id, entrypoint_path, module_prefix="physicsnemo_serve_plugin"
        )
        self._plugin_modules[workflow_id] = module
        return module

    def _workflow_cache_key(
        self,
        *,
        workflow_id: str,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
        payload: dict[str, Any],
        entrypoint_name: str,
    ) -> str:
        _ = manifest, runtime, payload, entrypoint_name
        return workflow_id

    def _cache_entry(
        self,
        *,
        cache_key: str,
        workflow: Any,
        workflow_id: str,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
        entrypoint_name: str,
    ) -> dict[str, Any]:
        loaded_at = _utc_now_iso()
        model_names = list(getattr(workflow, "model_cache_names", []) or [])
        return {
            "cache_key": cache_key,
            "workflow_id": workflow_id,
            "manifest_version": str(manifest.get("metadata", {}).get("version") or ""),
            "entrypoint": entrypoint_name,
            "executor_class": str(runtime.get("executor_class") or ""),
            "device_index": os.environ.get("GPU_DEVICE_INDEX"),
            "device_uuid": os.environ.get("GPU_DEVICE_UUID"),
            "model_names": [str(name) for name in model_names],
            "loaded_at": loaded_at,
            "last_used_at": loaded_at,
            "hit_count": 0,
        }

    def _publish_model_cache(self) -> None:
        if self.registry_publisher is None:
            return
        self.registry_publisher.update_model_cache(
            [dict(entry) for entry in self._workflow_cache_entries.values()]
        )

    def _get_or_create_cached_workflow(
        self,
        module: Any,
        *,
        workflow_id: str,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
        payload: dict[str, Any],
        entrypoint_name: str,
    ) -> tuple[str, Any] | None:
        cache_key = self._workflow_cache_key(
            workflow_id=workflow_id,
            manifest=manifest,
            runtime=runtime,
            payload=payload,
            entrypoint_name=entrypoint_name,
        )
        cached = self._workflow_cache.get(cache_key)
        if cached is not None:
            return cache_key, cached

        workflow = get_workflow_instance(module, workflow_id)
        if not workflow_is_cacheable(workflow):
            return None

        self._workflow_cache[cache_key] = workflow
        self._workflow_cache_entries[cache_key] = self._cache_entry(
            cache_key=cache_key,
            workflow=workflow,
            workflow_id=workflow_id,
            manifest=manifest,
            runtime=runtime,
            entrypoint_name=entrypoint_name,
        )
        self._publish_model_cache()
        return cache_key, workflow

    @staticmethod
    def _module_declares_cacheable_workflow(module: Any) -> bool:
        workflow_obj = getattr(module, "WORKFLOW", None)
        if workflow_obj is not None:
            cache_scope = (
                str(
                    getattr(workflow_obj, "cache_scope", "")
                    or getattr(type(workflow_obj), "cache_scope", "")
                    or ""
                )
                .strip()
                .lower()
            )
            return cache_scope == "process" or bool(
                getattr(workflow_obj, "cache_models", False)
                or getattr(type(workflow_obj), "cache_models", False)
            )
        return False

    def _record_cache_use(
        self,
        cache_key: str,
        *,
        model_names: list[str] | None = None,
    ) -> None:
        entry = self._workflow_cache_entries.get(cache_key)
        if entry is None:
            return
        entry["hit_count"] = int(entry.get("hit_count") or 0) + 1
        entry["last_used_at"] = _utc_now_iso()
        if model_names is not None:
            entry["model_names"] = [str(name) for name in model_names]
        self._publish_model_cache()

    def _resolve_execute_phase_hook(
        self,
        module: Any,
        *,
        workflow_id: str,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
        payload: dict[str, Any],
        entrypoint_name: str,
        phase: str,
    ) -> Callable[..., Any]:
        direct = getattr(module, phase, None)
        if callable(direct):
            return direct

        if not self._module_declares_cacheable_workflow(module):
            return resolve_phase_hook(module, workflow_id, phase)

        with self._workflow_cache_lock:
            cached = self._get_or_create_cached_workflow(
                module,
                workflow_id=workflow_id,
                manifest=manifest,
                runtime=runtime,
                payload=payload,
                entrypoint_name=entrypoint_name,
            )
            if cached is None:
                return resolve_phase_hook(module, workflow_id, phase)

            cache_key, workflow = cached
            hook = resolve_workflow_hook(
                workflow,
                workflow_id,
                phase,
                cleanup_method="cleanup_request",
            )

        def invoke_cached(*args: Any, **kwargs: Any) -> Any:
            # The cached workflow instance is shared and not thread-safe, so
            # requests for that workflow are intentionally serialized here.
            with self._workflow_cache_lock:
                try:
                    return hook(*args, **kwargs)
                finally:
                    self._record_cache_use(cache_key)

        return invoke_cached

    def warm_enabled_workflow(self) -> dict[str, Any]:
        workflow_id = os.environ.get("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", "").strip()
        if not workflow_id:
            return {"status": "skipped", "reason": "no_enabled_workflow"}

        if self.registry_publisher is not None:
            self.registry_publisher.set_warmup_status(
                "warming", workflow_id=workflow_id
            )
            self.registry_publisher.set_status("warming")

        try:
            plugin_root, manifest = _resolve_plugin_manifest(workflow_id)
            runtime = manifest.get("runtime", {})
            workflow_executor_class = str(runtime.get("executor_class") or "").strip()
            worker_executor_class = str(
                os.environ.get("GPU_EXECUTOR_CLASS") or ""
            ).strip()
            if (
                workflow_executor_class
                and worker_executor_class
                and workflow_executor_class != worker_executor_class
            ):
                if self.registry_publisher is not None:
                    self.registry_publisher.set_warmup_status(
                        "skipped", workflow_id=workflow_id
                    )
                    self.registry_publisher.set_status("available")
                return {
                    "status": "skipped",
                    "reason": "executor_class_mismatch",
                    "workflow_id": workflow_id,
                }
            entrypoint_name = runtime.get("entrypoint") or "workflow.py"
            payload = {
                "workflow_id": workflow_id,
                "manifest_version": manifest.get("metadata", {}).get("version"),
                "runtime": runtime,
                "resource_profile": {
                    "executor_class": runtime.get("executor_class"),
                    **(manifest.get("resources", {}).get("defaults", {}) or {}),
                },
            }
            module = self._load_plugin_module(
                workflow_id, plugin_root / entrypoint_name
            )
            warmup_context = self._build_warmup_context(workflow_id, runtime, payload)
            with ModelCacheWarmupLock(workflow_id):
                prepare_model_cache = getattr(module, "prepare_model_cache", None)
                if callable(prepare_model_cache):
                    prepare_model_cache(warmup_context)
                result = self._warm_cached_workflow(
                    module,
                    workflow_id=workflow_id,
                    manifest=manifest,
                    runtime=runtime,
                    payload=payload,
                    entrypoint_name=entrypoint_name,
                    warmup_context=warmup_context,
                )

            if self.registry_publisher is not None:
                self.registry_publisher.set_warmup_status(
                    "succeeded", workflow_id=workflow_id
                )
                self.registry_publisher.set_status("available")
            return result
        except Exception as exc:
            if self.registry_publisher is not None:
                error = str(exc)
                self.registry_publisher.set_warmup_status(
                    "failed", workflow_id=workflow_id, error=error
                )
                self.registry_publisher.set_status("failed", error=error)
            self.close()
            raise

    def _warm_cached_workflow(
        self,
        module: Any,
        *,
        workflow_id: str,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
        payload: dict[str, Any],
        entrypoint_name: str,
        warmup_context: dict[str, Any],
    ) -> dict[str, Any]:
        with self._workflow_cache_lock:
            cached = self._get_or_create_cached_workflow(
                module,
                workflow_id=workflow_id,
                manifest=manifest,
                runtime=runtime,
                payload=payload,
                entrypoint_name=entrypoint_name,
            )
            if cached is None:
                return {"status": "skipped", "reason": "workflow_not_cacheable"}

            cache_key, workflow = cached
            warmup = getattr(workflow, "warmup", None)
            warmup_result = warmup(warmup_context) if callable(warmup) else None
            model_names = self._model_names_from_warmup(workflow, warmup_result)
            self._record_cache_use(cache_key, model_names=model_names)
            return {"status": "warmed", "workflow_id": workflow_id}

    def _build_warmup_context(
        self,
        workflow_id: str,
        runtime: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        services = {
            "redis_url": os.environ.get("REDIS_URL"),
            "default_output_dir": os.environ.get("DEFAULT_OUTPUT_DIR"),
        }
        service_objects = {
            "redis_client": self.redis_client,
            "worker_shutdown_event": self.worker_shutdown_event,
        }
        return {
            "workflow_id": workflow_id,
            "runtime": runtime,
            "resource_profile": payload.get("resource_profile"),
            "services": services,
            "service_objects": service_objects,
            "device_index": os.environ.get("GPU_DEVICE_INDEX"),
            "device_uuid": os.environ.get("GPU_DEVICE_UUID"),
            "device": "cuda"
            if str(os.environ.get("GPU_DEVICE_KIND") or "gpu").lower() == "gpu"
            else "cpu",
        }

    @staticmethod
    def _model_names_from_warmup(
        workflow: Any,
        warmup_result: Any,
    ) -> list[str] | None:
        if isinstance(warmup_result, dict) and isinstance(
            warmup_result.get("model_names"), list
        ):
            return [str(name) for name in warmup_result["model_names"]]
        model_names = getattr(workflow, "model_cache_names", None)
        if isinstance(model_names, list):
            return [str(name) for name in model_names]
        return None

    def close(self) -> None:
        self._batch_coordinator.close()

        with self._workflow_cache_lock:
            had_entries = bool(self._workflow_cache_entries)
            workflows = list(self._workflow_cache.values())
            self._workflow_cache.clear()
            self._workflow_cache_entries.clear()
            if had_entries:
                self._publish_model_cache()

        for workflow in workflows:
            cleanup = getattr(workflow, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning(
                        "Cached workflow cleanup failed for %s: %s",
                        type(workflow).__name__,
                        exc,
                    )
                    logger.debug("Cached workflow cleanup traceback", exc_info=True)


def _plugin_dirs() -> list[Path]:
    """Resolve plugin search directories from environment."""
    raw = os.environ.get("PLUGIN_DIR") or ""
    dirs: list[Path] = []
    for item in raw.replace(",", os.pathsep).split(os.pathsep):
        candidate = item.strip()
        if candidate:
            dirs.append(Path(candidate).expanduser().resolve())
    return dirs


def _load_plugin_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a plugin manifest from disk."""
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ValueError(f"Plugin manifest not found: {manifest_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse plugin manifest: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Plugin manifest must be a mapping: {manifest_path}")
    return manifest


def _resolve_plugin_manifest(workflow_id: str) -> tuple[Path, dict[str, Any]]:
    """Resolve a workflow id to a plugin directory and loaded manifest."""
    ensure_workflow_enabled(workflow_id)

    for plugin_dir in _plugin_dirs():
        manifest_path = plugin_dir / workflow_id / DEFAULT_PLUGIN_MANIFEST_NAME
        if manifest_path.is_file():
            manifest = _load_plugin_manifest(manifest_path)
            manifest_id = manifest.get("metadata", {}).get("id")
            if manifest_id and manifest_id != workflow_id:
                raise ValueError(
                    f"Plugin manifest id mismatch for '{workflow_id}': found '{manifest_id}'"
                )
            return manifest_path.parent, manifest

        direct_manifest_path = plugin_dir / DEFAULT_PLUGIN_MANIFEST_NAME
        if direct_manifest_path.is_file():
            manifest = _load_plugin_manifest(direct_manifest_path)
            manifest_id = manifest.get("metadata", {}).get("id")
            if manifest_id == workflow_id:
                return plugin_dir, manifest

    raise ValueError(
        f"Plugin workflow '{workflow_id}' not found in plugin directories: {_plugin_dirs()}"
    )


def _decode_payload_object(payload_raw: Any) -> dict[str, Any]:
    """Decode a payload JSON string or mapping into a mutable dict."""
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    if isinstance(payload_raw, dict):
        return payload_raw

    return {}


def _next_plugin_stage(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the next manifest stage from the current stage context."""
    if not payload.get("workflow_id"):
        return None

    stage_context = payload.get("stage_context")
    if not isinstance(stage_context, dict):
        return None

    pipeline = stage_context.get("pipeline")
    current_stage_id = stage_context.get("current_stage_id")
    current_phase = stage_context.get("current_phase")
    if not isinstance(pipeline, list) or not isinstance(current_stage_id, str):
        return None

    current_stage = next(
        (
            stage
            for stage in pipeline
            if isinstance(stage, dict) and stage.get("id") == current_stage_id
        ),
        None,
    )
    if not isinstance(current_stage, dict):
        return None

    if isinstance(current_phase, str) and current_stage.get("phase") != current_phase:
        return None

    next_stage_id = current_stage.get("next")
    if not isinstance(next_stage_id, str) or not next_stage_id:
        return None

    return next(
        (
            stage
            for stage in pipeline
            if isinstance(stage, dict) and stage.get("id") == next_stage_id
        ),
        None,
    )


def _should_handoff_to_postprocess(
    payload: dict[str, Any], result: dict[str, Any]
) -> bool:
    """Return True when a successful plugin execute result should go to postprocess."""
    if not _is_success_status(result.get("status", "succeeded")):
        return False

    next_stage = _next_plugin_stage(payload)
    return isinstance(next_stage, dict) and next_stage.get("phase") == "postprocess"


def _should_handoff_to_collect(payload: dict[str, Any]) -> bool:
    """Return True when the next plugin stage is collect."""
    next_stage = _next_plugin_stage(payload)
    return isinstance(next_stage, dict) and next_stage.get("phase") == "collect"


def _should_persist_run_status_after_execute(
    payload: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    """Skip status persistence for successful execute handoffs that are still internal."""
    if not _is_success_status(result.get("status", "succeeded")):
        return True

    next_stage = _next_plugin_stage(payload)
    if not isinstance(next_stage, dict):
        return True

    next_phase = next_stage.get("phase")
    if next_phase in {"postprocess", "publish"}:
        return False

    return not (
        next_phase == "fanout" and isinstance(result.get("_pipeline_updates"), dict)
    )


def _should_mark_publication_skipped_after_execute_failure(
    payload: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if _is_success_status(result.get("status", "succeeded")):
        return False
    if not isinstance(payload.get("output_publication"), dict):
        return False

    next_stage = _next_plugin_stage(payload)
    return not (isinstance(next_stage, dict) and next_stage.get("phase") == "publish")


_PIPELINE_UPDATE_KEYS = {
    "operation",
    "parameters",
    "request",
    "resource_profile",
    "batch_profile",
    "prefetch_plan",
    "fanout_profile",
    "fanout_items",
}


def _result_without_private_pipeline_updates(result: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(result)
    sanitized.pop("_pipeline_updates", None)
    return sanitized


def _is_success_status(raw_status: Any) -> bool:
    return str(raw_status or "succeeded").strip().lower() in {
        "success",
        "succeeded",
        "completed",
    }


def _merge_pipeline_updates(
    payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    updates = result.get("_pipeline_updates")
    if not isinstance(updates, dict):
        return

    for key in _PIPELINE_UPDATE_KEYS:
        if key in updates:
            payload[key] = copy.deepcopy(updates[key])


def _move_execution_field(
    payload: dict[str, Any],
    execution: dict[str, Any],
    source_key: str,
    target_key: str,
) -> None:
    if source_key not in payload:
        return
    value = payload.pop(source_key)
    execution.setdefault(target_key, value)


def _derive_primary_output_path(outputs: Any) -> str | None:
    if not isinstance(outputs, list) or not outputs:
        return None

    primary = next(
        (
            entry
            for entry in outputs
            if isinstance(entry, dict) and bool(entry.get("primary"))
        ),
        None,
    )
    if not isinstance(primary, dict):
        primary = outputs[0] if isinstance(outputs[0], dict) else None
    if not isinstance(primary, dict):
        return None

    for key in ("storage_path", "path", "output_path"):
        value = primary.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _build_request_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request")
    request_envelope = dict(request) if isinstance(request, dict) else {}

    operation = payload.get("operation")
    if "operation" not in request_envelope and isinstance(operation, str) and operation:
        request_envelope["operation"] = operation

    parameters = payload.get("parameters")
    if "parameters" not in request_envelope and isinstance(parameters, dict):
        request_envelope["parameters"] = parameters

    return request_envelope


def _build_results_envelope(
    stream_name: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    workflow_name = payload.get("workflow_id") or payload.get("workflow")
    completed_at = datetime.now(timezone.utc).isoformat()
    plugin_payload = dict(result)
    run_id = str(plugin_payload.pop("run_id", payload.get("run_id") or "")).strip()
    status = str(plugin_payload.pop("status", "succeeded"))

    execution: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "gpu_stream": stream_name,
        "completed_at": completed_at,
    }
    if isinstance(workflow_name, str) and workflow_name:
        execution["workflow"] = workflow_name

    _move_execution_field(plugin_payload, execution, "outputs", "outputs")
    _move_execution_field(plugin_payload, execution, "artifacts", "outputs")
    _move_execution_field(
        plugin_payload, execution, "published_outputs", "published_outputs"
    )
    _move_execution_field(
        plugin_payload, execution, "published_artifacts", "published_artifacts"
    )
    _move_execution_field(plugin_payload, execution, "output_path", "output_path")
    _move_execution_field(plugin_payload, execution, "output_archive", "output_archive")
    _move_execution_field(plugin_payload, execution, "error", "error")
    _move_execution_field(
        plugin_payload,
        execution,
        "execution_time_seconds",
        "execution_time_seconds",
    )
    _move_execution_field(plugin_payload, execution, "batch_info", "batch_info")

    if "output_path" not in execution:
        derived_path = _derive_primary_output_path(execution.get("outputs"))
        if derived_path is not None:
            execution["output_path"] = derived_path

    results_envelope: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "request": _build_request_envelope(payload),
        "execution": execution,
        "payload": plugin_payload,
        "completed_at": completed_at,
        "gpu_stream": stream_name,
    }
    if isinstance(workflow_name, str) and workflow_name:
        results_envelope["workflow"] = workflow_name

    return results_envelope


def _build_primary_completion(
    stream_name: str,
    payload_raw: Any,
    result: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    """Build the primary downstream message for a completed execute step."""
    payload = _decode_payload_object(payload_raw)
    next_stage = _next_plugin_stage(payload)
    result_for_handoff = _result_without_private_pipeline_updates(result)
    result_succeeded = _is_success_status(result.get("status", "succeeded"))
    next_phase = next_stage.get("phase") if isinstance(next_stage, dict) else None
    should_handoff = bool(
        isinstance(next_stage, dict)
        and (
            result_succeeded
            or next_phase == "collect"
            or next_phase == "publish"
            or _should_handoff_to_postprocess(payload, result)
        )
        and next_phase != "results"
    )
    if should_handoff and isinstance(next_stage, dict):
        handoff_payload = copy.deepcopy(payload)
        handoff_payload["result"] = result_for_handoff
        if result_succeeded:
            _merge_pipeline_updates(handoff_payload, result)
        stage_context = handoff_payload.get("stage_context")
        if isinstance(stage_context, dict):
            stage_context["current_stage_id"] = next_stage.get("id")
            stage_context["current_phase"] = next_stage.get("phase")
        return (
            str(next_stage.get("queue", next_phase or "results")),
            handoff_payload,
            str(next_phase or "results"),
        )

    return (
        "results",
        _build_results_envelope(stream_name, payload, result_for_handoff),
        "results",
    )


def _build_batch_primary_outputs(
    stream_name: str,
    batch_payload: dict[str, Any],
    batch_result: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str, str]]:
    outputs: list[tuple[str, dict[str, Any], str, str]] = []
    for entry in batch_result.get("batch_results", []):
        if not isinstance(entry, dict):
            continue
        item_result = entry.get("result")
        item_payload = entry.get("payload")
        if not isinstance(item_result, dict) or not isinstance(item_payload, dict):
            continue
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            continue
        item_payload_for_handoff = _batch_item_completion_payload(
            batch_payload, item_payload
        )
        item_result_for_handoff = dict(item_result)
        item_result_for_handoff.pop("batch_info", None)
        item_result_for_handoff["run_id"] = run_id
        stream_name_out, payload_out, stage_out = _build_primary_completion(
            stream_name,
            item_payload_for_handoff,
            item_result_for_handoff,
        )
        outputs.append((stream_name_out, payload_out, stage_out, run_id))
    return outputs


def _batch_item_completion_payload(
    batch_payload: dict[str, Any], item_payload: dict[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(item_payload)
    run_id = str(
        payload.get("run_id")
        or batch_payload.get("run_id")
        or batch_payload.get("batch_id")
        or ""
    ).strip()
    stage_context = batch_payload.get("stage_context")
    if isinstance(stage_context, dict):
        payload["stage_context"] = copy.deepcopy(stage_context)
    for key in ("batch_id", "batch_info", "output_publication"):
        if key in batch_payload and key not in payload:
            payload[key] = copy.deepcopy(batch_payload[key])
    if not isinstance(payload.get("output_publication"), dict):
        workflow_id = str(
            payload.get("workflow_id") or batch_payload.get("workflow_id") or ""
        ).strip()
        output_publication = _output_publication_from_env(workflow_id, run_id)
        if output_publication is not None:
            payload["output_publication"] = output_publication
    if isinstance(payload.get("output_publication"), dict):
        _ensure_publish_stage_after_execute(payload)
    return payload


def _output_publication_from_env(
    workflow_id: str, run_id: str
) -> dict[str, Any] | None:
    if not workflow_id or not run_id:
        return None
    raw = os.environ.get("PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON", "").strip()
    if not raw:
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Invalid output publication config JSON in environment")
        return None
    if not isinstance(config, dict) or not config.get("enabled"):
        return None
    storage = config.get("storage")
    if not isinstance(storage, dict):
        return None
    provider = str(storage.get("type") or "").strip().lower()
    prefix = "/".join(
        part.strip("/")
        for part in (str(storage.get("prefix") or "outputs"), workflow_id, run_id)
        if part and part.strip("/")
    )
    if provider == "s3":
        bucket = str(storage.get("bucket") or "").strip()
        if not bucket:
            return None
        resolved_storage = {
            "type": "s3",
            "bucket": bucket,
            "prefix": prefix,
        }
        for key in ("region", "endpoint"):
            value = storage.get(key)
            if isinstance(value, str) and value.strip():
                resolved_storage[key] = value.strip()
    elif provider == "azure":
        container = str(storage.get("container") or "").strip().strip("/")
        endpoint = str(storage.get("endpoint") or "").strip()
        if not container or not endpoint:
            return None
        resolved_storage = {
            "type": "azure",
            "container": container,
            "prefix": prefix,
            "endpoint": endpoint.rstrip("/"),
        }
    else:
        return None
    return {
        "target": {
            "artifact": "primary",
            "provider": provider,
            "storage": resolved_storage,
        }
    }


def _ensure_publish_stage_after_execute(payload: dict[str, Any]) -> None:
    stage_context = payload.get("stage_context")
    if not isinstance(stage_context, dict):
        return
    pipeline = stage_context.get("pipeline")
    if not isinstance(pipeline, list):
        return
    current_stage_id = stage_context.get("current_stage_id")
    if not isinstance(current_stage_id, str):
        return
    current_stage = next(
        (
            stage
            for stage in pipeline
            if isinstance(stage, dict) and stage.get("id") == current_stage_id
        ),
        None,
    )
    if not isinstance(current_stage, dict) or current_stage.get("phase") != "execute":
        return
    if any(
        isinstance(stage, dict) and stage.get("phase") == "publish"
        for stage in pipeline
    ):
        return
    next_stage_id = current_stage.get("next")
    publish_stage = {
        "id": "publish",
        "phase": "publish",
        "handler": "publish_outputs",
        "queue": "publish",
        "next": next_stage_id,
    }
    try:
        insert_at = pipeline.index(current_stage) + 1
    except ValueError:
        insert_at = len(pipeline)
    pipeline.insert(insert_at, publish_stage)
    current_stage["next"] = "publish"


def _normalize_batch_info(
    batch_id: str,
    payload: dict[str, Any],
    item_count: int,
) -> dict[str, Any]:
    raw = payload.get("batch_info")
    batch_info = dict(raw) if isinstance(raw, dict) else {}
    batch_info.setdefault("batch_id", batch_id)
    batch_info.setdefault("batch_size", item_count)
    return batch_info


def _build_release_envelope(
    run_id: str,
    resource_id: int,
    memory_mb: int,
    status: str,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """Build a release payload for the scheduler release stream."""
    payload = {
        "run_id": run_id,
        "resource_id": resource_id,
        "memory_mb": memory_mb,
        "status": status,
        "released_at": datetime.now(timezone.utc).isoformat(),
    }
    if parent_run_id:
        payload["parent_run_id"] = parent_run_id
    return payload


def _stream_entry(run_id: str, payload: dict[str, Any], stage: str) -> dict[str, Any]:
    """Encode a queue-compatible stream entry with run_id, payload, and stage."""
    return {
        "run_id": run_id,
        "payload": json.dumps(payload),
        "stage": stage,
    }


def _initialize_execution_state(
    redis_client: "redis_lib.Redis | None",
    workflow_name: str,
    run_id: str,
    parameters: dict[str, Any],
) -> None:
    """Initialize workflow execution state for status tracking."""
    if redis_client is None:
        return

    try:
        from datetime import datetime, timezone

        execution_key = f"workflow_execution:{workflow_name}:{run_id}"
        payload = {
            "workflow_name": workflow_name,
            "execution_id": run_id,
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {"parameters": parameters},
        }
        redis_client.setex(execution_key, 86400, json.dumps(payload, default=str))
    except Exception as e:
        logger.warning("Failed to initialize execution state: %s", e)


def _update_execution_state(
    redis_client: "redis_lib.Redis | None",
    workflow_name: str,
    run_id: str,
    status: str,
    execution_time_seconds: float,
    output_path: str | None,
    error_message: str | None,
) -> None:
    """Update workflow execution state for status tracking."""
    if redis_client is None:
        return

    try:
        from datetime import datetime, timezone

        execution_key = f"workflow_execution:{workflow_name}:{run_id}"
        payload = {
            "workflow_name": workflow_name,
            "execution_id": run_id,
            "status": status,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": execution_time_seconds,
            "error_message": error_message,
            "metadata": {"output_path": output_path},
        }
        redis_client.setex(execution_key, 86400, json.dumps(payload, default=str))
    except Exception as e:
        logger.warning("Failed to update execution state: %s", e)


def _persist_run_status_and_result(
    redis_client: "redis_lib.Redis | None",
    workflow_name: str,
    run_id: str,
    result: dict[str, Any],
    *,
    publication_skipped: bool = False,
) -> None:
    """Persist run status and result payload for REST endpoints."""
    if redis_client is None:
        return

    try:
        now = str(int(time.time()))
        status = result.get("status", "succeeded")
        output_path = result.get("output_path")
        error = result.get("error")
        execution_time = result.get("execution_time_seconds")

        run_key = f"run:{run_id}"
        run_mapping = {
            "status": status,
            "stage": "completed" if status == "succeeded" else "failed",
            "updated_at": now,
            "inference_completed_at": now,
        }
        if publication_skipped:
            run_mapping.update(
                {
                    "output_location": "local_and_cloud",
                    "output_publication_status": "skipped",
                    "publish_completed_at": now,
                    "published_artifact_count": "0",
                }
            )
        redis_client.hset(run_key, mapping=run_mapping)
        if publication_skipped and hasattr(redis_client, "hdel"):
            redis_client.hdel(run_key, "publish_error")

        result_key = f"result:{run_id}"
        result_payload: dict[str, Any] = {
            "run_id": run_id,
            "workflow": workflow_name,
            "status": status,
            "output_path": output_path,
            "error": error,
            "execution_time_seconds": execution_time,
        }
        artifacts = result.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            result_payload["execution"] = {"outputs": artifacts}
        redis_client.setex(result_key, 86400, json.dumps(result_payload, default=str))
    except Exception as e:
        logger.warning("Failed to persist run result: %s", e)


# =============================================================================
# Message Reclaimer (Background Thread)
# =============================================================================


class MessageReclaimer:
    """
    Periodically reclaims stale messages from the worker's input stream.

    Uses XAUTOCLAIM to take ownership of messages that have been pending
    longer than idle_ms (worker probably crashed).
    """

    def __init__(
        self,
        redis_client: "redis_lib.Redis",
        stream_name: str,
        group_name: str,
        consumer_name: str,
        idle_ms: int = RECLAIM_IDLE_MS,
        interval_secs: int = RECLAIM_INTERVAL_SECS,
    ):
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.idle_ms = idle_ms
        self.interval_secs = interval_secs
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[Callable[[dict], None]] = None

    def start(self, handler: Callable[[dict], None]) -> None:
        """Start the reclaimer background thread."""
        self._handler = handler
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Reclaimer started (idle_ms={self.idle_ms}, interval={self.interval_secs}s)"
        )

    def stop(self) -> None:
        """Stop the reclaimer thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Reclaimer stopped")

    def _run_loop(self) -> None:
        """Background loop that periodically reclaims stale messages."""
        while not self._stop_event.is_set():
            try:
                claimed = self._reclaim_once()

                for msg in claimed:
                    if self._handler:
                        try:
                            self._handler(msg)
                        except Exception as e:
                            logger.error(f"Reclaimer handler error: {e}")

            except Exception as e:
                logger.error(f"Reclaim cycle error: {e}")

            # Wait for next cycle
            self._stop_event.wait(timeout=self.interval_secs)

    def _reclaim_once(self) -> list[dict]:
        """Run one reclaim cycle using XAUTOCLAIM."""
        try:
            # XAUTOCLAIM stream group consumer min-idle-time start [COUNT count]
            result = self.redis_client.xautoclaim(
                self.stream_name,
                self.group_name,
                self.consumer_name,
                min_idle_time=self.idle_ms,
                start_id="0-0",
                count=10,
            )

            # result = (next_start_id, [(msg_id, fields), ...], [deleted_ids])
            if not result or len(result) < 2:
                return []

            messages = result[1]
            if not messages:
                return []

            logger.info(
                f"Reclaimer claimed {len(messages)} stale messages from {self.stream_name}"
            )

            # Convert to job dicts
            claimed = []
            for msg_id, fields in messages:
                job = {
                    k.decode() if isinstance(k, bytes) else k: (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in fields.items()
                }
                job["msg_id"] = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                job["_reclaimed"] = True  # Mark as reclaimed for logging
                claimed.append(job)

            return claimed

        except Exception as e:
            if "XAUTOCLAIM" in str(e) and "unknown command" in str(e):
                return self._reclaim_with_xclaim()
            if "NOGROUP" in str(e):
                # Group doesn't exist yet, ignore
                return []
            raise

    def _reclaim_with_xclaim(self) -> list[dict]:
        """Fallback reclaim cycle using XPENDING + XCLAIM (Redis < 6.2)."""
        pending = self.redis_client.xpending_range(
            self.stream_name,
            self.group_name,
            "-",
            "+",
            count=10,
            idle=self.idle_ms,
        )
        if not pending:
            return []

        message_ids: list[str] = []
        for entry in pending:
            if isinstance(entry, dict) and "message_id" in entry:
                message_ids.append(entry["message_id"])
            elif isinstance(entry, (list, tuple)) and entry:
                message_ids.append(entry[0])

        if not message_ids:
            return []

        messages = self.redis_client.xclaim(
            self.stream_name,
            self.group_name,
            self.consumer_name,
            min_idle_time=self.idle_ms,
            message_ids=message_ids,
        )
        if not messages:
            return []

        logger.info(
            f"Reclaimer claimed {len(messages)} stale messages from {self.stream_name} using XCLAIM"
        )

        claimed = []
        for msg_id, fields in messages:
            job = {
                k.decode() if isinstance(k, bytes) else k: (
                    v.decode() if isinstance(v, bytes) else v
                )
                for k, v in fields.items()
            }
            job["msg_id"] = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
            job["_reclaimed"] = True
            claimed.append(job)

        return claimed


# =============================================================================
# Stream Registration
# =============================================================================


def build_registry_field(
    stream_name: str, worker_index: int | str, pid: int | None = None
) -> str:
    """Build a unique registry field key for this worker process."""
    worker_pid = os.getpid() if pid is None else pid
    return f"{stream_name}:worker:{worker_index}:pid:{worker_pid}"


def register_stream(
    redis_client: "redis_lib.Redis",
    stream_name: str,
    metadata: dict,
    registry_field: str | None = None,
) -> None:
    """Register this worker's stream in gpu:registry."""
    # Create stream with consumer group
    try:
        redis_client.xgroup_create(stream_name, "workers", id="$", mkstream=True)
        logger.info(f"Created stream {stream_name}")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise
        logger.info(f"Stream {stream_name} already exists")

    # Register in gpu:registry using a per-process field key so workers
    # sharing a stream do not overwrite each other.
    field = registry_field or stream_name
    redis_client.hset("gpu:registry", field, json.dumps(metadata))
    logger.info("Registered in gpu:registry: field=%s stream=%s", field, stream_name)


def deregister_stream(
    redis_client: "redis_lib.Redis", stream_name: str, registry_field: str | None = None
) -> None:
    """Deregister this worker's stream."""
    field = registry_field or stream_name
    redis_client.hdel("gpu:registry", field)
    logger.info(
        "Deregistered from gpu:registry: field=%s stream=%s", field, stream_name
    )


# =============================================================================
# Async Stream Registration (scicomp-rq QueueManager)
# =============================================================================


async def register_stream_async(
    qm: "QueueManager",
    stream_name: str,
    metadata: dict,
    registry_field: str | None = None,
) -> None:
    """Register this worker's stream using QueueManager (async).

    Args:
        qm: QueueManager instance from scicomp-rq
        stream_name: Stream name used for worker registration and scheduling
        metadata: Worker metadata dictionary
        registry_field: Optional per-process registry field key
    """
    # Create stream with consumer group
    try:
        await qm.create_consumer_group(
            stream_name, "workers", start_id="$", create_stream=True
        )
        logger.info(f"Created stream {stream_name}")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise
        logger.info(f"Stream {stream_name} already exists")

    # Register in gpu:registry using a per-process field key so workers
    # sharing a stream do not overwrite each other.
    field = registry_field or stream_name
    await qm.hset("gpu:registry", field, json.dumps(metadata))
    logger.info("Registered in gpu:registry: field=%s stream=%s", field, stream_name)


async def deregister_stream_async(
    qm: "QueueManager", stream_name: str, registry_field: str | None = None
) -> None:
    """Deregister this worker's stream using QueueManager (async).

    Args:
        qm: QueueManager instance from scicomp-rq
        stream_name: Stream name to deregister
        registry_field: Optional per-process registry field key
    """
    field = registry_field or stream_name
    await qm.hdel("gpu:registry", field)
    logger.info(
        "Deregistered from gpu:registry: field=%s stream=%s", field, stream_name
    )


# =============================================================================
# Job Processing
# =============================================================================


def parse_job_payload(job: dict) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """
    Parse job payload from Redis stream message.

    Args:
        job: Raw job dictionary from Redis

    Returns:
        Tuple of (workflow_name, run_id, parameters, payload)
    """
    run_id = job.get("run_id", "unknown")

    # Parse payload (may be JSON string or already a dict)
    payload_raw = job.get("payload", "{}")
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse payload JSON: {payload_raw}")
            payload = {}
    else:
        payload = payload_raw

    workflow = (
        job.get("workflow")
        or payload.get("workflow")
        or payload.get("workflow_id")
        or "deterministic_workflow"
    )

    # Extract parameters from payload (may be nested under 'parameters' or 'inputs')
    parameters = payload.get("parameters", payload.get("inputs", payload))

    return workflow, run_id, parameters, payload


def _job_workflow_name(job: dict[str, Any]) -> str:
    workflow_name, _, _, _ = parse_job_payload(job)
    return workflow_name


def _is_source_not_pending_error(exc: BaseException) -> bool:
    return "SOURCE_NOT_PENDING" in str(exc)


def _begin_in_flight_message(msg_id: str) -> bool:
    if not msg_id:
        return True

    with _IN_FLIGHT_MESSAGE_IDS_LOCK:
        if msg_id in _IN_FLIGHT_MESSAGE_IDS:
            return False
        _IN_FLIGHT_MESSAGE_IDS.add(msg_id)
    return True


def _finish_in_flight_message(msg_id: str) -> None:
    if not msg_id:
        return

    with _IN_FLIGHT_MESSAGE_IDS_LOCK:
        _IN_FLIGHT_MESSAGE_IDS.discard(msg_id)


def process_job(
    executor: WorkflowExecutor,
    job: dict,
) -> dict[str, Any]:
    """
    Process a job by executing the specified workflow.

    Args:
        executor: WorkflowExecutor instance
        job: Job dictionary from Redis stream

    Returns:
        Result dictionary with status, output_path, and optional error
    """
    workflow_name, run_id, parameters, payload = parse_job_payload(job)

    logger.info(
        "Processing job: run_id=%s workflow=%s %s",
        run_id,
        workflow_name,
        worker_identity_tag(),
    )
    logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")

    try:
        logger.info(
            "Loading workflow model and running inference: run_id=%s workflow=%s %s",
            run_id,
            workflow_name,
            worker_identity_tag(),
        )
        result = executor.execute(workflow_name, run_id, parameters, payload=payload)
        if isinstance(result.get("batch_results"), list):
            for entry in result["batch_results"]:
                if not isinstance(entry, dict):
                    continue
                item_result = entry.get("result")
                item_payload = entry.get("payload")
                item_run_id = str(entry.get("run_id") or "").strip()
                if (
                    item_run_id
                    and isinstance(item_result, dict)
                    and isinstance(item_payload, dict)
                ):
                    item_payload_for_status = _batch_item_completion_payload(
                        payload, item_payload
                    )
                    if not _should_persist_run_status_after_execute(
                        item_payload_for_status, item_result
                    ):
                        continue
                    _persist_run_status_and_result(
                        executor.redis_client,
                        workflow_name,
                        item_run_id,
                        item_result,
                        publication_skipped=_should_mark_publication_skipped_after_execute_failure(
                            item_payload_for_status, item_result
                        ),
                    )
        elif _should_persist_run_status_after_execute(payload, result):
            _persist_run_status_and_result(
                executor.redis_client,
                workflow_name,
                run_id,
                result,
                publication_skipped=_should_mark_publication_skipped_after_execute_failure(
                    payload, result
                ),
            )
        response = {
            "run_id": run_id,
            "status": result.get("status", "succeeded"),
            "output_path": result.get("output_path"),
            "error": result.get("error"),
            "execution_time_seconds": result.get("execution_time_seconds"),
        }
        for key, value in result.items():
            response.setdefault(key, value)
        return response
    except ValueError as e:
        # Workflow not found
        logger.error(f"Workflow error for {run_id}: {e}")
        return {
            "run_id": run_id,
            "status": "failed",
            "output_path": None,
            "error": str(e),
        }
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error for {run_id}: {e}")
        logger.debug(traceback.format_exc())
        return {
            "run_id": run_id,
            "status": "failed",
            "output_path": None,
            "error": f"Worker error: {e}",
        }
    except BaseException as exc:
        log_fatal_base_exception("process_job", exc, run_id=run_id)
        raise


def extract_resource_info(payload_raw: Any) -> tuple[int, int]:
    """
    Extract resource_id and memory_mb from payload JSON; fall back to env.

    The scheduler uses these fields for memory-aware release accounting.
    """
    payload_obj: dict[str, Any]
    if isinstance(payload_raw, str):
        try:
            payload_obj = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload_obj = {}
    elif isinstance(payload_raw, dict):
        payload_obj = payload_raw
    else:
        payload_obj = {}

    resource_id = payload_obj.get("resource_id")
    memory_mb = payload_obj.get("memory_mb")
    if resource_id is None:
        resource_id = int(os.environ.get("GPU_DEVICE_INDEX", "-1"))
    if memory_mb is None:
        memory_mb = int(os.environ.get("GPU_MEMORY_MB", "0"))
    return int(resource_id), int(memory_mb)


def complete_job(
    redis_client: "redis_lib.Redis",
    stream_name: str,
    job: dict,
    result: dict,
) -> None:
    """
    Complete a job by sending to the next primary stage and the release stream.

    The primary destination is either:
    1. `postprocess`, when the plugin pipeline declares it after execute
    2. `results`, otherwise

    The release stream is always emitted so the scheduler can free the GPU.
    """
    run_id = result["run_id"]
    status = result["status"]
    resource_id, memory_mb = extract_resource_info(job.get("payload", "{}"))
    payload_obj = _decode_payload_object(job.get("payload", "{}"))
    parent_run_id = result.get("parent_run_id") or payload_obj.get("parent_run_id")
    release_stream = resolve_output_stream("release")
    if isinstance(result.get("batch_results"), list):
        primary_outputs = _build_batch_primary_outputs(stream_name, payload_obj, result)
        for (
            primary_stream_name,
            primary_payload,
            primary_stage,
            item_run_id,
        ) in primary_outputs:
            primary_stream = resolve_output_stream(primary_stream_name)
            primary_msg_id = redis_client.xadd(
                primary_stream,
                _stream_entry(item_run_id, primary_payload, primary_stage),
            )
            logger.info(
                "Sent to '%s' stream: %s (status=%s) %s",
                primary_stream,
                primary_msg_id,
                status,
                worker_identity_tag(),
            )
    else:
        primary_stream_name, primary_payload, primary_stage = _build_primary_completion(
            stream_name,
            job.get("payload", "{}"),
            result,
        )
        primary_stream = resolve_output_stream(primary_stream_name)
        primary_msg_id = redis_client.xadd(
            primary_stream,
            _stream_entry(run_id, primary_payload, primary_stage),
        )
        logger.info(
            "Sent to '%s' stream: %s (status=%s) %s",
            primary_stream,
            primary_msg_id,
            status,
            worker_identity_tag(),
        )

    # Step 2: Send to release stream to free GPU for scheduler
    release_msg_id = redis_client.xadd(
        release_stream,
        _stream_entry(
            run_id,
            _build_release_envelope(
                run_id,
                resource_id,
                memory_mb,
                status,
                parent_run_id=parent_run_id,
            ),
            "release",
        ),
    )
    logger.info(
        "Sent to '%s' stream: %s (resource_id=%s) %s",
        release_stream,
        release_msg_id,
        resource_id,
        worker_identity_tag(),
    )

    # Step 3: ACK the original message (remove from pending)
    msg_id = job.get("msg_id")
    if msg_id:
        redis_client.xack(stream_name, "workers", msg_id)
        logger.info("ACKed message %s %s", msg_id, worker_identity_tag())


async def complete_job_async(
    qm: "QueueManager",
    msg: "Message",
    result: dict[str, Any],
) -> None:
    """Complete a job using atomic QueueManager fan-out operations."""
    if Output is None:
        raise RuntimeError(
            "scicomp_rq.Output is required for async fan-out. "
            "Install a compatible scicomp_rq package."
        )

    run_id = result["run_id"]
    status = result["status"]
    resource_id, memory_mb = extract_resource_info(msg.payload)
    payload_obj = _decode_payload_object(msg.payload)
    parent_run_id = result.get("parent_run_id") or payload_obj.get("parent_run_id")
    release_stream = resolve_output_stream("release")
    output_targets = []
    if isinstance(result.get("batch_results"), list):
        for (
            primary_stream_name,
            primary_payload,
            primary_stage,
            item_run_id,
        ) in _build_batch_primary_outputs(msg.stream, payload_obj, result):
            output_targets.append(
                Output(
                    resolve_output_stream(primary_stream_name),
                    json.dumps(primary_payload),
                    stage=primary_stage,
                    run_id=item_run_id,
                )
            )
    else:
        primary_stream_name, primary_payload, primary_stage = _build_primary_completion(
            msg.stream,
            msg.payload,
            result,
        )
        output_targets.append(
            Output(
                resolve_output_stream(primary_stream_name),
                json.dumps(primary_payload),
                stage=primary_stage,
            )
        )
    output_targets.append(
        Output(
            release_stream,
            json.dumps(
                _build_release_envelope(
                    run_id,
                    resource_id,
                    memory_mb,
                    status,
                    parent_run_id=parent_run_id,
                )
            ),
            stage="release",
        )
    )
    message_ids = await qm.forward_many(msg, output_targets)

    logger.info(
        "Fanned out primary outputs and release: run_id=%s resource_id=%s ids=%s %s",
        run_id,
        resource_id,
        message_ids,
        worker_identity_tag(),
    )


def process_message(
    executor: WorkflowExecutor,
    redis_client: "redis_lib.Redis",
    stream_name: str,
    job: dict,
) -> None:
    """
    Process a single message (job). Used by both main loop and reclaimer.
    """
    run_id = job.get("run_id", "unknown")
    is_reclaimed = job.get("_reclaimed", False)
    workflow_name = _job_workflow_name(job)
    msg_id = str(job.get("msg_id") or "").strip()
    result: dict[str, Any] | None = None
    log_completion_status = False

    if is_reclaimed:
        logger.info("Processing RECLAIMED job: %s %s", run_id, worker_identity_tag())

    if not _begin_in_flight_message(msg_id):
        logger.info(
            "Skipping duplicate in-flight job: run_id=%s msg_id=%s reclaimed=%s %s",
            run_id,
            msg_id,
            is_reclaimed,
            worker_identity_tag(),
        )
        return

    try:
        try:
            _reset_torch_cuda_peak_memory_stats(
                run_id=run_id, workflow_id=workflow_name
            )
            _log_cuda_memory_snapshot(
                "before_execute", run_id=run_id, workflow_id=workflow_name
            )

            try:
                result = process_job(executor, job)
            finally:
                _log_cuda_memory_snapshot(
                    "after_execute", run_id=run_id, workflow_id=workflow_name
                )
            log_completion_status = True
        except Exception as e:
            # Unexpected error (e.g., Redis connection lost)
            logger.error(
                "CRITICAL ERROR for %s: %s %s", run_id, e, worker_identity_tag()
            )
            logger.debug(traceback.format_exc())

            result = {
                "run_id": run_id,
                "status": "failed",
                "output_path": None,
                "error": f"Worker error: {e}",
            }
        except BaseException as exc:
            log_fatal_base_exception("process_message", exc, run_id=run_id)
            raise
        finally:
            _run_after_request_cleanup(run_id=run_id, workflow_id=workflow_name)
            _log_cuda_memory_snapshot(
                "after_request_cleanup",
                run_id=run_id,
                workflow_id=workflow_name,
            )

        if result is None:
            return

        try:
            complete_job(redis_client, stream_name, job, result)
        except Exception as completion_error:
            logger.error(
                "Failed to send completion for %s: %s %s",
                run_id,
                completion_error,
                worker_identity_tag(),
            )
            logger.debug(traceback.format_exc())
            # Message will be reclaimed after timeout.
            return

        if log_completion_status:
            status_icon = "✓" if result["status"] == "succeeded" else "✗"
            reclaim_tag = " (reclaimed)" if is_reclaimed else ""
            exec_time = result.get("execution_time_seconds")
            time_tag = f" [{exec_time:.1f}s]" if exec_time else ""
            logger.info(
                "%s Job %s completed%s: %s%s %s",
                status_icon,
                run_id,
                reclaim_tag,
                result["status"],
                time_tag,
                worker_identity_tag(),
            )
    finally:
        _finish_in_flight_message(msg_id)


async def process_message_async(
    executor: WorkflowExecutor,
    qm: "QueueManager",
    msg: "Message",
) -> None:
    """Process a message using QueueManager for stream operations."""
    job = {
        "run_id": msg.run_id,
        "payload": msg.payload,
        "msg_id": msg.id,
    }
    workflow_name = _job_workflow_name(job)
    run_id = msg.run_id
    msg_id = str(msg.id or "").strip()
    result: dict[str, Any] | None = None
    log_completion_status = False

    if not _begin_in_flight_message(msg_id):
        logger.info(
            "Skipping duplicate in-flight job: run_id=%s msg_id=%s reclaimed=%s %s",
            run_id,
            msg_id,
            True,
            worker_identity_tag(),
        )
        return

    try:
        try:
            _reset_torch_cuda_peak_memory_stats(
                run_id=run_id, workflow_id=workflow_name
            )
            _log_cuda_memory_snapshot(
                "before_execute", run_id=run_id, workflow_id=workflow_name
            )

            try:
                result = process_job(executor, job)
            finally:
                _log_cuda_memory_snapshot(
                    "after_execute", run_id=run_id, workflow_id=workflow_name
                )
            log_completion_status = True
        except Exception as e:
            logger.error(
                "CRITICAL ERROR for %s: %s %s",
                msg.run_id,
                e,
                worker_identity_tag(),
            )
            logger.debug(traceback.format_exc())

            result = {
                "run_id": run_id,
                "status": "failed",
                "output_path": None,
                "error": f"Worker error: {e}",
            }
        except BaseException as exc:
            log_fatal_base_exception("process_message_async", exc, run_id=msg.run_id)
            raise
        finally:
            _run_after_request_cleanup(run_id=run_id, workflow_id=workflow_name)
            _log_cuda_memory_snapshot(
                "after_request_cleanup",
                run_id=run_id,
                workflow_id=workflow_name,
            )

        if result is None:
            return

        try:
            await complete_job_async(qm, msg, result)
        except Exception as completion_error:
            if _is_source_not_pending_error(completion_error):
                logger.info(
                    "Ignoring duplicate async completion for %s: %s %s",
                    run_id,
                    completion_error,
                    worker_identity_tag(),
                )
                return
            logger.error(
                "Failed to send completion for %s: %s %s",
                msg.run_id,
                completion_error,
                worker_identity_tag(),
            )
            logger.debug(traceback.format_exc())
            return

        if log_completion_status:
            status_icon = "✓" if result["status"] == "succeeded" else "✗"
            exec_time = result.get("execution_time_seconds")
            time_tag = f" [{exec_time:.1f}s]" if exec_time else ""
            logger.info(
                "%s Job %s completed%s %s",
                status_icon,
                msg.run_id,
                time_tag,
                worker_identity_tag(),
            )
    finally:
        _finish_in_flight_message(msg_id)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the inference worker."""
    # Import redis here to allow testing without redis installed
    try:
        import redis
    except ImportError:
        logger.error("redis package not installed. Install with: pip install redis")
        raise SystemExit(1)

    # Get configuration from environment
    stream_name = os.environ.get("GPU_STREAM_NAME")
    if not stream_name:
        logger.error("GPU_STREAM_NAME environment variable is required")
        raise SystemExit(1)

    device_index = int(os.environ.get("GPU_DEVICE_INDEX", "0"))
    device_name = os.environ.get("GPU_DEVICE_NAME", "unknown")
    memory_mb = int(os.environ.get("GPU_MEMORY_MB", "0"))
    device_uuid = os.environ.get("GPU_DEVICE_UUID", f"GPU-{device_index}")

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    worker_index = int(os.environ.get("GPU_WORKER_INDEX", "0"))
    worker_pid = os.getpid()
    registry_field = build_registry_field(stream_name, worker_index, worker_pid)

    logger.info("=" * 70)
    logger.info("PhysicsNeMo Serve Inference Worker")
    logger.info("=" * 70)
    logger.info(f"Stream: {stream_name}")
    logger.info(f"GPU Device: {device_index} ({device_name}, {memory_mb}MB)")
    logger.info(f"GPU UUID: {device_uuid}")
    logger.info(f"Redis URL: {redis_url}")
    logger.info(f"Output Directory: {DEFAULT_OUTPUT_DIR}")
    logger.info(
        "Worker Identity: %s registry_field=%s", worker_identity_tag(), registry_field
    )
    logger.info("=" * 70)

    # Connect to Redis
    r = redis.from_url(redis_url)
    consumer_name = f"{stream_name}-{worker_pid}"

    # Prepare metadata
    metadata = build_worker_metadata(
        stream_name=stream_name,
        device_index=device_index,
        device_name=device_name,
        device_uuid=device_uuid,
        memory_mb=memory_mb,
        worker_index=worker_index,
        worker_pid=worker_pid,
        registry_field=registry_field,
    )

    # Register stream
    register_stream(r, stream_name, metadata, registry_field=registry_field)

    worker_shutdown_event = threading.Event()
    registry_publisher = WorkerRegistryPublisher(
        r,
        stream_name=stream_name,
        registry_field=registry_field,
        metadata=metadata,
    )
    executor = WorkflowExecutor(
        r,
        registry_publisher=registry_publisher,
        worker_shutdown_event=worker_shutdown_event,
    )
    executor.warm_enabled_workflow()

    # Start background reclaimer
    reclaimer = MessageReclaimer(
        redis_client=r,
        stream_name=stream_name,
        group_name="workers",
        consumer_name=consumer_name,
        idle_ms=RECLAIM_IDLE_MS,
        interval_secs=RECLAIM_INTERVAL_SECS,
    )

    # Handler for reclaimed messages
    def handle_reclaimed(job: dict) -> None:
        process_message(executor, r, stream_name, job)

    reclaimer.start(handle_reclaimed)

    # Shutdown handling
    shutdown = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal shutdown
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown = True
        worker_shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info(
        "Listening on %s (GPU %s: %s) %s",
        stream_name,
        device_index,
        device_name,
        worker_identity_tag(),
    )

    # Main loop: Poll for new messages
    try:
        while not shutdown:
            # Block-read from stream (only new messages with ">")
            messages = r.xreadgroup(
                groupname="workers",
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=1,
                block=5000,  # 5 second timeout
            )

            if not messages:
                continue

            for stream, entries in messages:
                for msg_id, fields in entries:
                    # Decode fields
                    job = {
                        k.decode() if isinstance(k, bytes) else k: (
                            v.decode() if isinstance(v, bytes) else v
                        )
                        for k, v in fields.items()
                    }
                    job["msg_id"] = (
                        msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    )

                    # Process the message
                    process_message(executor, r, stream_name, job)

    finally:
        # Cleanup
        logger.info("Stopping reclaimer...")
        reclaimer.stop()

        executor.close()
        deregister_stream(r, stream_name, registry_field=registry_field)
        logger.info("Shutdown complete")


# =============================================================================
# Async Main Entry Point (scicomp-rq QueueManager)
# =============================================================================


# Import QueueManager at module level for patching in tests
try:
    from scicomp_rq import Output, QueueManager
except ImportError:
    QueueManager = None  # type: ignore
    Output = None  # type: ignore


def _register_async_shutdown_handlers(
    shutdown_event: asyncio.Event,
    worker_shutdown_event: threading.Event,
) -> list[tuple[int, Any]]:
    """Register immediate signal handlers that trigger async worker shutdown."""
    loop = asyncio.get_running_loop()
    registered_handlers: list[tuple[int, Any]] = []

    def request_shutdown(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        if worker_shutdown_event.is_set():
            logger.info(
                "Received %s while shutdown is already in progress", signal_name
            )
            return
        logger.info("Received %s, shutting down...", signal_name)
        worker_shutdown_event.set()
        loop.call_soon_threadsafe(shutdown_event.set)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handler = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
        except (OSError, RuntimeError, ValueError):
            continue
        registered_handlers.append((signum, previous_handler))

    return registered_handlers


async def _cancel_task(task: asyncio.Future[Any]) -> None:
    """Cancel an asyncio task or future and wait for it to finish."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _close_redis_client(redis_client: object) -> None:
    """Close a sync or async Redis client if it exposes a close method."""
    close = getattr(redis_client, "close", None)
    if not callable(close):
        return

    result = close()
    if inspect.isawaitable(result):
        await result


async def _deregister_stream_async_safely(
    qm: "QueueManager",
    stream_name: str,
    registry_field: str | None = None,
) -> None:
    """Best-effort async worker deregistration during shutdown."""
    try:
        await deregister_stream_async(qm, stream_name, registry_field=registry_field)
    except RuntimeError as exc:
        if "broken pipe" in str(exc).lower():
            logger.warning(
                "Failed to deregister stream '%s' during shutdown: %s",
                stream_name,
                exc,
            )
            return
        logger.exception(
            "Failed to deregister stream '%s' during shutdown", stream_name
        )
    except Exception:
        logger.exception(
            "Failed to deregister stream '%s' during shutdown", stream_name
        )


async def main_async() -> None:
    """Async main entry point using scicomp-rq QueueManager.

    This is the new async version of main() that uses QueueManager
    for Redis stream operations.
    """
    # Check if scicomp_rq is available
    if QueueManager is None:
        logger.error("scicomp_rq not installed. Install with: pip install scicomp_rq")
        raise SystemExit(1)

    # Import redis here to allow testing without redis installed
    try:
        import redis
    except ImportError:
        logger.error("redis package not installed. Install with: pip install redis")
        raise SystemExit(1)

    # Get configuration from environment
    stream_name = os.environ.get("GPU_STREAM_NAME")
    if not stream_name:
        logger.error("GPU_STREAM_NAME environment variable is required")
        raise SystemExit(1)

    device_index = int(os.environ.get("GPU_DEVICE_INDEX", "0"))
    device_name = os.environ.get("GPU_DEVICE_NAME", "unknown")
    memory_mb = int(os.environ.get("GPU_MEMORY_MB", "0"))
    device_uuid = os.environ.get("GPU_DEVICE_UUID", f"GPU-{device_index}")
    worker_index = int(os.environ.get("GPU_WORKER_INDEX", "0"))
    worker_pid = os.getpid()
    registry_field = build_registry_field(stream_name, worker_index, worker_pid)

    logger.info("=" * 70)
    logger.info("PhysicsNeMo Serve Inference Worker (Async)")
    logger.info("=" * 70)
    logger.info(f"Stream: {stream_name}")
    logger.info(f"GPU Device: {device_index} ({device_name}, {memory_mb}MB)")
    logger.info(f"GPU UUID: {device_uuid}")
    logger.info(
        "Worker Identity: %s registry_field=%s", worker_identity_tag(), registry_field
    )
    logger.info("=" * 70)

    # Create QueueManager from environment
    qm = await QueueManager.from_env()

    # Prepare metadata
    metadata = build_worker_metadata(
        stream_name=stream_name,
        device_index=device_index,
        device_name=device_name,
        device_uuid=device_uuid,
        memory_mb=memory_mb,
        worker_index=worker_index,
        worker_pid=worker_pid,
        registry_field=registry_field,
    )

    # Register stream using async version
    await register_stream_async(
        qm, stream_name, metadata, registry_field=registry_field
    )

    # Connect to Redis for workflow executor
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    registry_publisher = WorkerRegistryPublisher(
        redis_client,
        stream_name=stream_name,
        registry_field=registry_field,
        metadata=metadata,
    )
    worker_shutdown_event = threading.Event()
    executor = WorkflowExecutor(
        redis_client,
        registry_publisher=registry_publisher,
        worker_shutdown_event=worker_shutdown_event,
    )
    executor.warm_enabled_workflow()

    consumer_name = f"{stream_name}-{worker_pid}"
    shutdown_event = asyncio.Event()
    registered_handlers = _register_async_shutdown_handlers(
        shutdown_event, worker_shutdown_event
    )

    async def reclaim_loop() -> None:
        while not shutdown_event.is_set():
            try:
                _, claimed = await qm.claim_idle_messages(
                    stream_name,
                    "workers",
                    consumer_name,
                    RECLAIM_IDLE_MS,
                    "0-0",
                    10,
                )
                for msg in claimed:
                    await process_message_async(executor, qm, msg)
            except Exception as e:
                logger.error(f"Reclaim cycle error: {e}")
            await asyncio.sleep(RECLAIM_INTERVAL_SECS)

    reclaimer_task = asyncio.create_task(reclaim_loop())

    try:
        logger.info(
            "Listening on %s (GPU %s: %s) %s",
            stream_name,
            device_index,
            device_name,
            worker_identity_tag(),
        )
        while not shutdown_event.is_set():
            read_task = asyncio.ensure_future(
                qm.read_messages(
                    stream_name,
                    "workers",
                    consumer_name,
                    count=1,
                    block_ms=5000,
                )
            )
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, _pending = await asyncio.wait(
                {read_task, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_task in done:
                await _cancel_task(read_task)
                break

            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task

            messages = read_task.result()
            if not messages:
                continue
            for msg in messages:
                await process_message_async(executor, qm, msg)
    except asyncio.CancelledError:
        logger.info("Shutdown requested...")
    finally:
        worker_shutdown_event.set()
        shutdown_event.set()
        await _cancel_task(reclaimer_task)
        try:
            executor.close()
            await _deregister_stream_async_safely(
                qm,
                stream_name,
                registry_field=registry_field,
            )
        finally:
            await _close_redis_client(redis_client)
            for signum, previous_handler in registered_handlers:
                try:
                    signal.signal(signum, previous_handler)
                except (OSError, RuntimeError, ValueError):
                    continue
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except BaseException as exc:
        log_fatal_base_exception("worker main", exc)
        raise
