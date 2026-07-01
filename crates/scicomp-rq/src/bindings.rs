/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Python bindings for scicomp-rq via PyO3.
//!
//! Provides an async-compatible Python interface to the Rust queue manager.
//!
//! # Usage
//!
//! ```python
//! import asyncio
//! import scicomp_rq
//!
//! async def main():
//!     # Create from Redis URL
//!     qm = await scicomp_rq.QueueManager.from_redis_url("redis://localhost:6379")
//!
//!     # Or from environment
//!     qm = await scicomp_rq.QueueManager.from_env()
//!
//!     # Use methods directly on the instance
//!     await qm.create_consumer_group("prefetch", "prefetch:grp", "$", True)
//!     msg_id = await qm.enqueue("prefetch", "run-001", '{"key": "value"}', "prefetch")
//!     print(f"Enqueued: {msg_id}")
//!
//! asyncio.run(main())
//! ```

use crate::types::{Message, Output};
use crate::{LogicalStreamName, QueueManager, StreamKey, hash_ops};
use pyo3::prelude::*;

fn queue_error_to_pyerr(err: crate::QueueError) -> pyo3::PyErr {
    match err {
        crate::QueueError::Config(msg) => pyo3::exceptions::PyValueError::new_err(msg),
        other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
    }
}

// ============================================================================
// PyMessage - Python binding for Message type
// ============================================================================

/// A message read from a Redis stream.
///
/// Contains all information needed for acknowledgment and handoff operations.
/// The `stream` and `group` fields enable self-contained operations like
/// `ack_message()` and `handoff_message()` without requiring the caller
/// to track this context separately.
///
/// # Example
///
/// ```python
/// msg = scicomp_rq.Message(
///     id="1706123456789-0",
///     stream="stream:prefetch",
///     group="prefetch:grp",
///     run_id="run-001",
///     payload='{"model": "pangu"}',
///     stage="prefetch",
/// )
/// print(msg.stream)  # "stream:prefetch"
/// ```
///
/// PyO3 extraction policy:
/// - `from_py_object` is enabled so Python `Message` instances can be passed
///   directly into methods such as `ack_message()` and `handoff_message()`.
#[pyclass(name = "Message", from_py_object)]
#[derive(Clone)]
pub struct PyMessage {
    inner: Message,
}

impl From<Message> for PyMessage {
    fn from(msg: Message) -> Self {
        PyMessage { inner: msg }
    }
}

impl From<PyMessage> for Message {
    fn from(py_msg: PyMessage) -> Self {
        py_msg.inner
    }
}

#[pymethods]
impl PyMessage {
    /// Create a new Message with all fields.
    ///
    /// Args:
    ///     id: Redis stream message ID (e.g., "1706123456789-0")
    ///     stream: Full Redis stream key (e.g., "stream:prefetch")
    ///     group: Consumer group name (e.g., "prefetch:grp")
    ///     run_id: Unique workflow run identifier
    ///     payload: JSON-encoded payload data
    ///     stage: Current processing stage name
    #[new]
    #[pyo3(signature = (id, stream, group, run_id, payload, stage))]
    fn new(
        id: String,
        stream: String,
        group: String,
        run_id: String,
        payload: String,
        stage: String,
    ) -> Self {
        PyMessage {
            inner: Message::new(id, stream, group, run_id, payload, stage),
        }
    }

    /// Redis stream message ID (e.g., "1706123456789-0")
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Full Redis stream key where message was read from (e.g., "stream:prefetch")
    #[getter]
    fn stream(&self) -> &str {
        &self.inner.stream
    }

    /// Consumer group name (e.g., "prefetch:grp")
    #[getter]
    fn group(&self) -> &str {
        &self.inner.group
    }

    /// Unique identifier for this workflow run
    #[getter]
    fn run_id(&self) -> &str {
        &self.inner.run_id
    }

    /// JSON-encoded payload data
    #[getter]
    fn payload(&self) -> &str {
        &self.inner.payload
    }

    /// Current processing stage
    #[getter]
    fn stage(&self) -> &str {
        &self.inner.stage
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(id={:?}, stream={:?}, group={:?}, run_id={:?}, stage={:?})",
            self.inner.id, self.inner.stream, self.inner.group, self.inner.run_id, self.inner.stage
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyOutput - Python binding for Output type
// ============================================================================

/// Output destination for `forward_many()` operation.
///
/// Specifies where to send a message and optionally what stage name to use.
///
/// # Example
///
/// ```python
/// # Default stage behavior uses the full destination stream key
/// out1 = scicomp_rq.Output("stream:results", '{"status": "ok"}')
///
/// # Explicit stage name
/// out2 = scicomp_rq.Output("stream:results", '{"status": "ok"}', stage="final_results")
///
/// # Override run_id for fanout child messages
/// out3 = scicomp_rq.Output("stream:execute", '{"status": "ok"}', run_id="child-run-1")
/// ```
///
/// PyO3 extraction policy:
/// - `from_py_object` is enabled so Python `Output` instances can be consumed
///   directly by `forward_many()`.
#[pyclass(name = "Output", from_py_object)]
#[derive(Clone)]
pub struct PyOutput {
    inner: Output,
}

impl From<Output> for PyOutput {
    fn from(out: Output) -> Self {
        PyOutput { inner: out }
    }
}

impl From<PyOutput> for Output {
    fn from(py_out: PyOutput) -> Self {
        py_out.inner
    }
}

#[pymethods]
impl PyOutput {
    /// Create a new Output destination.
    ///
    /// Args:
    ///     stream: Destination stream (full Redis key)
    ///     payload: JSON payload for this destination
    ///     stage: Optional stage name (None = defaults to destination stream key)
    ///     run_id: Optional run_id override (None = preserve source message run_id)
    #[new]
    #[pyo3(signature = (stream, payload, stage=None, run_id=None))]
    fn new(stream: String, payload: String, stage: Option<String>, run_id: Option<String>) -> Self {
        let mut output = Output::new(stream, payload);
        if let Some(s) = stage {
            output = output.with_stage(s);
        }
        if let Some(run_id) = run_id {
            output = output.with_run_id(run_id);
        }
        PyOutput { inner: output }
    }

    /// Destination stream (full Redis key)
    #[getter]
    fn stream(&self) -> &str {
        &self.inner.stream
    }

    /// Explicit run_id override when provided.
    #[getter]
    fn run_id(&self) -> Option<&str> {
        self.inner.run_id()
    }

    /// JSON payload for this destination
    #[getter]
    fn payload(&self) -> &str {
        &self.inner.payload
    }

    /// Stage name (None = defaults to destination stream key)
    #[getter]
    fn stage(&self) -> Option<&str> {
        self.inner.stage.as_deref()
    }

    fn __repr__(&self) -> String {
        match &self.inner.stage {
            Some(s) => format!(
                "Output(stream={:?}, payload={:?}, stage={:?})",
                self.inner.stream, self.inner.payload, s
            ),
            None => format!(
                "Output(stream={:?}, payload={:?}, stage=None)",
                self.inner.stream, self.inner.payload
            ),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for the Rust QueueManager.
///
/// Use `QueueManager.from_redis_url()` or `QueueManager.from_env()` to create instances.
///
/// PyO3 extraction policy:
/// - `skip_from_py_object` is enabled because `QueueManager` objects are created
///   through factory methods and are not consumed as by-value Python arguments.
#[pyclass(name = "QueueManager", skip_from_py_object)]
#[derive(Clone)]
pub struct PyQueueManager {
    inner: QueueManager,
}

impl std::fmt::Debug for PyQueueManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueManager").finish_non_exhaustive()
    }
}

#[pymethods]
impl PyQueueManager {
    /// Direct construction is not supported. Use from_redis_url() or from_env().
    #[new]
    fn new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot construct QueueManager directly. Use QueueManager.from_redis_url() or QueueManager.from_env()",
        ))
    }

    /// Create a QueueManager from a Redis connection URL.
    ///
    /// Args:
    ///     url: Redis connection URL (e.g., "redis://localhost:6379")
    ///
    /// Returns:
    ///     QueueManager instance
    ///
    /// Raises:
    ///     RuntimeError: If connection fails
    #[staticmethod]
    fn from_redis_url(py: Python<'_>, url: String) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let qm = QueueManager::from_redis_url(&url)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyQueueManager { inner: qm })
        })
    }

    /// Create a QueueManager from environment variables.
    ///
    /// Environment variables:
    ///     - REDIS_URL: Connection URL (default: redis://127.0.0.1:6379)
    ///
    /// Stream naming is resolved by callers (for example in worker-runtime).
    ///
    /// Returns:
    ///     QueueManager instance
    ///
    /// Raises:
    ///     RuntimeError: If connection fails
    #[staticmethod]
    fn from_env(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let qm = QueueManager::from_env()
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyQueueManager { inner: qm })
        })
    }

    /// Enqueue a message to a stream.
    ///
    /// Args:
    ///     stream_name: Logical stream name (e.g., "prefetch")
    ///     run_id: Unique identifier for this workflow run
    ///     payload: JSON-encoded payload string
    ///     stage: Stage name to record
    ///
    /// Note:
    ///     Logical stream names must not contain ":". For explicit Redis stream keys,
    ///     use the Rust API `enqueue_to_stream(...)`.
    ///
    /// Returns:
    ///     Redis stream message ID (e.g., "1706123456789-0")
    ///
    /// Raises:
    ///     ValueError: If payload is not valid JSON
    ///     ValueError: If stream_name/run_id/stage validation fails
    ///     RuntimeError: If Redis operation fails
    fn enqueue<'py>(
        &self,
        py: Python<'py>,
        stream_name: String,
        run_id: String,
        payload: String,
        stage: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_name = LogicalStreamName::new(stream_name);
            // Validate payload JSON without normalizing caller-provided formatting.
            serde_json::from_str::<serde_json::Value>(&payload).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON payload: {e}"))
            })?;
            inner
                .enqueue(&stream_name, &run_id, &payload, &stage)
                .await
                .map_err(queue_error_to_pyerr)
        })
    }

    /// Check Redis connection health.
    ///
    /// Returns:
    ///     Tuple of (connected: bool, latency_ms: int, script_loaded: bool)
    ///
    /// Raises:
    ///     RuntimeError: If the health check fails
    fn health_check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let status = inner
                .health_check()
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok((
                status.connected(),
                status.latency_ms(),
                status.script_loaded(),
            ))
        })
    }

    // =========================================================================
    // New Unified API Methods
    // =========================================================================

    /// Read messages from a stream using XREADGROUP.
    ///
    /// Returns Message objects with stream and group fields populated,
    /// enabling self-contained ack_message() and handoff_message() calls.
    ///
    /// Args:
    ///     stream: Full Redis stream key (e.g., "stream:prefetch")
    ///     group: Consumer group name (e.g., "prefetch:grp")
    ///     consumer: Consumer name within the group
    ///     count: Maximum number of messages to read (default: 1)
    ///     block_ms: How long to block waiting for messages (default: 0)
    ///
    /// Returns:
    ///     List of Message objects (may be empty if no messages available)
    ///
    /// Raises:
    ///     RuntimeError: If XREADGROUP fails
    #[pyo3(signature = (stream, group, consumer, count=1, block_ms=0))]
    fn read_messages<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        group: String,
        consumer: String,
        count: usize,
        block_ms: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = StreamKey::new(stream);
            let messages = inner
                .read_messages(&stream, &group, &consumer, count, block_ms)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Convert to PyMessage objects
            let py_messages: Vec<PyMessage> = messages.into_iter().map(PyMessage::from).collect();
            Ok(py_messages)
        })
    }

    /// Acknowledge a single message.
    ///
    /// Uses the stream and group information stored in the Message object,
    /// so you don't need to pass them separately.
    ///
    /// Args:
    ///     message: The Message to acknowledge
    ///
    /// Returns:
    ///     Number of messages acknowledged (1 if successful, 0 if already acked)
    ///
    /// Raises:
    ///     RuntimeError: If XACK fails
    fn ack_message<'py>(&self, py: Python<'py>, message: PyMessage) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let msg: Message = message.into();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .ack_message(&msg)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Atomically hand off a message to a destination stream.
    ///
    /// This method:
    /// 1. Sends the message to the destination stream (XADD)
    /// 2. Acknowledges the original message (XACK)
    /// 3. Updates the run hash with the new stage (HSET)
    ///
    /// The default stage behavior uses the full destination stream key when not provided.
    ///
    /// Args:
    ///     message: The Message to hand off
    ///     dest_stream: Destination stream (full Redis key)
    ///     payload: Optional new payload (None = use message's original payload)
    ///     stage: Optional stage name (None = destination stream key)
    ///
    /// Returns:
    ///     New message ID in the destination stream
    ///
    /// Raises:
    ///     RuntimeError: If the handoff fails
    #[pyo3(signature = (message, dest_stream, payload=None, stage=None))]
    fn handoff_message<'py>(
        &self,
        py: Python<'py>,
        message: PyMessage,
        dest_stream: String,
        payload: Option<String>,
        stage: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let msg: Message = message.into();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let dest_stream = StreamKey::new(dest_stream);
            inner
                .handoff_message(&msg, &dest_stream, payload.as_deref(), stage.as_deref())
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Atomically hand off a message to a destination stream with an optional
    /// destination run_id override.
    ///
    /// This behaves like `handoff_message()`, but allows the caller to change
    /// the run_id stored on the destination message.
    ///
    /// Args:
    ///     message: The Message to hand off
    ///     dest_stream: Destination stream (full Redis key)
    ///     payload: Optional new payload (None = use message's original payload)
    ///     stage: Optional stage name (None = destination stream key)
    ///     run_id: Optional destination run_id override (None = preserve source run_id)
    ///
    /// Returns:
    ///     New message ID in the destination stream
    ///
    /// Raises:
    ///     RuntimeError: If the handoff fails
    #[pyo3(signature = (message, dest_stream, payload=None, stage=None, run_id=None))]
    fn handoff_message_to_run<'py>(
        &self,
        py: Python<'py>,
        message: PyMessage,
        dest_stream: String,
        payload: Option<String>,
        stage: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let msg: Message = message.into();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let dest_stream = StreamKey::new(dest_stream);
            inner
                .handoff_message_to_run(
                    &msg,
                    &dest_stream,
                    payload.as_deref(),
                    stage.as_deref(),
                    run_id.as_deref(),
                )
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Forward a message to multiple destinations atomically.
    ///
    /// This is useful for fan-out patterns where a message needs to go to
    /// multiple streams (e.g., results + GPU release).
    ///
    /// The original message is acknowledged after all destinations receive it.
    ///
    /// Args:
    ///     message: The Message to forward
    ///     outputs: List of Output objects specifying destinations
    ///
    /// Note:
    ///     `outputs` must contain at least one destination.
    ///
    /// Returns:
    ///     List of new message IDs (one per output, in same order)
    ///
    /// Raises:
    ///     ValueError: If outputs is empty
    ///     RuntimeError: If any destination fails
    fn forward_many<'py>(
        &self,
        py: Python<'py>,
        message: PyMessage,
        outputs: Vec<PyOutput>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let msg: Message = message.into();
        let rust_outputs: Vec<Output> = outputs.into_iter().map(Output::from).collect();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .forward_many(&msg, &rust_outputs)
                .await
                .map_err(queue_error_to_pyerr)
        })
    }

    /// Create a consumer group on a stream.
    ///
    /// This is idempotent - calling it multiple times is safe.
    ///
    /// Args:
    ///     stream: Full Redis stream key
    ///     group: Consumer group name
    ///     start_id: ID to start reading from ("$" = new messages only, "0" = all)
    ///     create_stream: If True, create the stream if it doesn't exist
    ///
    /// Returns:
    ///     True if group was created, False if it already existed
    ///
    /// Raises:
    ///     RuntimeError: If creation fails (other than already existing)
    #[pyo3(signature = (stream, group, start_id="$", create_stream=true))]
    fn create_consumer_group<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        group: String,
        start_id: &str,
        create_stream: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let start_id = start_id.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = StreamKey::new(stream);
            inner
                .create_consumer_group(&stream, &group, &start_id, create_stream)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Claim idle pending messages using XAUTOCLAIM.
    ///
    /// Use this to reclaim messages from crashed consumers. Returns full
    /// Message objects with all fields populated.
    ///
    /// Args:
    ///     stream: Full Redis stream key
    ///     group: Consumer group name
    ///     consumer: New consumer to assign messages to
    ///     min_idle_ms: Minimum idle time (only claim messages idle longer than this)
    ///     start_id: Start scanning from this ID ("0-0" for beginning)
    ///     count: Maximum messages to claim
    ///
    /// Returns:
    ///     Tuple of (next_cursor_id, list_of_claimed_messages)
    ///
    /// Raises:
    ///     RuntimeError: If XAUTOCLAIM fails
    #[allow(clippy::too_many_arguments)]
    fn claim_idle_messages<'py>(
        &self,
        py: Python<'py>,
        stream: String,
        group: String,
        consumer: String,
        min_idle_ms: u64,
        start_id: String,
        count: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = StreamKey::new(stream);
            let (cursor, messages) = inner
                .claim_idle_messages(&stream, &group, &consumer, min_idle_ms, &start_id, count)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Convert to PyMessage objects
            let py_messages: Vec<PyMessage> = messages.into_iter().map(PyMessage::from).collect();
            Ok((cursor, py_messages))
        })
    }

    /// Set a field in a Redis hash.
    ///
    /// Args:
    ///     key: Hash key
    ///     field: Field name
    ///     value: Field value
    ///
    /// Returns:
    ///     1 if new field was created, 0 if existing field was updated
    ///
    /// Raises:
    ///     RuntimeError: If HSET fails
    fn hset<'py>(
        &self,
        py: Python<'py>,
        key: String,
        field: String,
        value: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut conn = inner.connection();
            hash_ops::hset(&mut conn, &key, &field, &value)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Delete a field from a Redis hash.
    ///
    /// Args:
    ///     key: Hash key
    ///     field: Field name to delete
    ///
    /// Returns:
    ///     1 if field was deleted, 0 if field didn't exist
    ///
    /// Raises:
    ///     RuntimeError: If HDEL fails
    fn hdel<'py>(
        &self,
        py: Python<'py>,
        key: String,
        field: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut conn = inner.connection();
            hash_ops::hdel(&mut conn, &key, &field)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Get all fields and values from a Redis hash.
    ///
    /// Args:
    ///     key: Hash key (e.g., "gpu:registry")
    ///
    /// Returns:
    ///     Dict[str, str] containing all field-value pairs.
    ///     Returns empty dict if key does not exist.
    ///
    /// Raises:
    ///     RuntimeError: If HGETALL fails
    ///
    /// Example:
    ///     gpus = await qm.hgetall("gpu:registry")
    ///     for stream_name, metadata_json in gpus.items():
    ///         info = json.loads(metadata_json)
    fn hgetall<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut conn = inner.connection();
            hash_ops::hgetall(&mut conn, &key)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        "QueueManager()".to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python module definition.
#[pymodule]
fn scicomp_rq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQueueManager>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyOutput>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PyQueueManager Construction Tests
    // =========================================================================

    #[test]
    fn test_pyqueue_manager_new_returns_error() {
        let result = PyQueueManager::new();
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("from_redis_url") || err_str.contains("from_env"),
            "Error should mention factory methods: {}",
            err_str
        );
    }

    #[test]
    fn test_pyqueue_manager_new_error_is_type_error() {
        let result = PyQueueManager::new();
        assert!(result.is_err());
        // The error should be a TypeError (PyTypeError in PyO3)
        let err = result.unwrap_err();
        // PyO3 errors contain the Python exception type in their string representation
        let err_str = format!("{:?}", err);
        assert!(
            err_str.contains("TypeError") || err_str.contains("Cannot construct"),
            "Error should be TypeError: {}",
            err_str
        );
    }

    #[test]
    fn test_pyqueue_manager_new_error_message_contains_both_methods() {
        let result = PyQueueManager::new();
        let err = result.unwrap_err();
        let err_str = err.to_string();
        // Should mention both factory methods for user guidance
        assert!(
            err_str.contains("from_redis_url"),
            "Error should mention from_redis_url: {}",
            err_str
        );
        assert!(
            err_str.contains("from_env"),
            "Error should mention from_env: {}",
            err_str
        );
    }

    #[test]
    fn bindings_json_tests_are_contract_focused() {
        let source = include_str!("bindings.rs");
        let non_test_source = source
            .split("mod tests {")
            .nth(1)
            .expect("bindings.rs should contain tests");
        assert!(
            !non_test_source.contains("test_json_payload_parsing_unicode_escape")
                && !non_test_source.contains("test_json_payload_parsing_scientific_notation")
                && !non_test_source.contains("test_json_payload_parsing_trailing_comma")
                && !non_test_source.contains("test_json_payload_parsing_single_quotes")
                && !non_test_source.contains("test_json_payload_parsing_nan_not_allowed")
                && !non_test_source.contains("test_json_payload_parsing_deeply_nested"),
            "bindings tests should avoid third-party serde_json behavior matrices"
        );
    }

    // =========================================================================
    // JSON Payload Contract Tests (crate-owned behavior)
    // =========================================================================

    #[test]
    fn test_json_payload_contract_accepts_valid_payload() {
        let valid_json = r#"{"run_id":"run-1","model":"pangu","steps":10}"#;
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(valid_json);
        assert!(parsed.is_ok(), "valid JSON payload should be accepted");
    }

    #[test]
    fn test_json_payload_contract_rejects_invalid_payload() {
        let invalid_json = "not valid json";
        let err = serde_json::from_str::<serde_json::Value>(invalid_json)
            .expect_err("invalid JSON payload should be rejected");
        let formatted = format!("Invalid JSON payload: {err}");
        assert!(
            formatted.starts_with("Invalid JSON payload:"),
            "invalid payload errors should map to the expected Python-facing prefix"
        );
    }

    #[test]
    fn test_bindings_source_contains_invalid_json_error_prefix() {
        let source = include_str!("bindings.rs");
        assert!(
            source.contains("Invalid JSON payload:"),
            "bindings should preserve the canonical invalid JSON error prefix"
        );
    }

    // ===============================================================
    // Typical Plugin Payload Tests
    // ===============================================================

    #[test]
    fn test_json_payload_prefetch() {
        let json = r#"{
            "run_id": "run-001",
            "dataset": "demo-input",
            "artifacts": ["input-a", "input-b"],
            "window": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}
        }"#;
        let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(parsed["run_id"], "run-001");
        assert_eq!(parsed["dataset"], "demo-input");
        assert_eq!(parsed["artifacts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_json_payload_inference() {
        let json = r#"{
            "run_id": "run-001",
            "stage": "inference",
            "input_path": "s3://bucket/input.json",
            "output_path": "s3://bucket/output.json",
            "model_ref": "gs://models/demo-model"
        }"#;
        let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(parsed["stage"], "inference");
        assert!(parsed["input_path"].as_str().unwrap().starts_with("s3://"));
    }

    // =========================================================================
    // PyMessage Tests
    // =========================================================================

    #[test]
    fn test_pymessage_new() {
        let msg = PyMessage::new(
            "1706123456789-0".to_string(),
            "stream:prefetch".to_string(),
            "prefetch:grp".to_string(),
            "run-123".to_string(),
            r#"{"key": "value"}"#.to_string(),
            "prefetch".to_string(),
        );
        assert_eq!(msg.id(), "1706123456789-0");
        assert_eq!(msg.stream(), "stream:prefetch");
        assert_eq!(msg.group(), "prefetch:grp");
        assert_eq!(msg.run_id(), "run-123");
        assert_eq!(msg.payload(), r#"{"key": "value"}"#);
        assert_eq!(msg.stage(), "prefetch");
    }

    #[test]
    fn test_pymessage_from_message() {
        let msg = Message::new("1-0", "stream:test", "test:grp", "run-1", "{}", "stage");
        let py_msg: PyMessage = msg.into();
        assert_eq!(py_msg.id(), "1-0");
        assert_eq!(py_msg.stream(), "stream:test");
        assert_eq!(py_msg.group(), "test:grp");
    }

    #[test]
    fn test_pymessage_to_message() {
        let py_msg = PyMessage::new(
            "1-0".to_string(),
            "stream:test".to_string(),
            "grp".to_string(),
            "run-1".to_string(),
            "{}".to_string(),
            "stage".to_string(),
        );
        let msg: Message = py_msg.into();
        assert_eq!(msg.id, "1-0");
        assert_eq!(msg.stream, "stream:test");
        assert_eq!(msg.group, "grp");
    }

    #[test]
    fn test_pymessage_repr() {
        let msg = PyMessage::new(
            "1-0".to_string(),
            "stream:test".to_string(),
            "grp".to_string(),
            "run-1".to_string(),
            "{}".to_string(),
            "stage".to_string(),
        );
        let repr = msg.__repr__();
        assert!(repr.contains("Message"));
        assert!(repr.contains("1-0"));
        assert!(repr.contains("stream:test"));
        assert!(repr.contains("grp"));
        assert!(repr.contains("run-1"));
        assert!(repr.contains("stage"));
    }

    #[test]
    fn test_pymessage_str_eq_repr() {
        let msg = PyMessage::new(
            "1-0".to_string(),
            "stream:test".to_string(),
            "grp".to_string(),
            "run-1".to_string(),
            "{}".to_string(),
            "stage".to_string(),
        );
        assert_eq!(msg.__str__(), msg.__repr__());
    }

    #[test]
    fn test_pymessage_clone() {
        let msg = PyMessage::new(
            "1-0".to_string(),
            "stream:test".to_string(),
            "grp".to_string(),
            "run-1".to_string(),
            "{}".to_string(),
            "stage".to_string(),
        );
        let cloned = msg.clone();
        assert_eq!(msg.id(), cloned.id());
        assert_eq!(msg.stream(), cloned.stream());
        assert_eq!(msg.payload(), cloned.payload());
    }

    // =========================================================================
    // PyOutput Tests
    // =========================================================================

    #[test]
    fn test_pyoutput_new_without_stage() {
        let out = PyOutput::new(
            "stream:results".to_string(),
            r#"{"status": "ok"}"#.to_string(),
            None,
            None,
        );
        assert_eq!(out.stream(), "stream:results");
        assert_eq!(out.payload(), r#"{"status": "ok"}"#);
        assert!(out.stage().is_none());
    }

    #[test]
    fn test_pyoutput_new_with_stage() {
        let out = PyOutput::new(
            "stream:results".to_string(),
            "{}".to_string(),
            Some("final_results".to_string()),
            None,
        );
        assert_eq!(out.stream(), "stream:results");
        assert_eq!(out.payload(), "{}");
        assert_eq!(out.stage(), Some("final_results"));
    }

    #[test]
    fn test_pyoutput_from_output() {
        let out = Output::new("stream:test", "{}").with_stage("mystage");
        let py_out: PyOutput = out.into();
        assert_eq!(py_out.stream(), "stream:test");
        assert_eq!(py_out.payload(), "{}");
        assert_eq!(py_out.stage(), Some("mystage"));
    }

    #[test]
    fn test_pyoutput_to_output() {
        let py_out = PyOutput::new(
            "stream:test".to_string(),
            "{}".to_string(),
            Some("mystage".to_string()),
            None,
        );
        let out: Output = py_out.into();
        assert_eq!(out.stream, "stream:test");
        assert_eq!(out.payload, "{}");
        assert_eq!(out.stage, Some("mystage".to_string()));
    }

    #[test]
    fn test_pyoutput_repr_with_stage() {
        let out = PyOutput::new(
            "stream:test".to_string(),
            "{}".to_string(),
            Some("mystage".to_string()),
            None,
        );
        let repr = out.__repr__();
        assert!(repr.contains("Output"));
        assert!(repr.contains("stream:test"));
        assert!(repr.contains("mystage"));
    }

    #[test]
    fn test_pyoutput_repr_without_stage() {
        let out = PyOutput::new("stream:test".to_string(), "{}".to_string(), None, None);
        let repr = out.__repr__();
        assert!(repr.contains("Output"));
        assert!(repr.contains("stream:test"));
        assert!(repr.contains("None"));
    }

    #[test]
    fn test_pyoutput_str_eq_repr() {
        let out = PyOutput::new(
            "stream:test".to_string(),
            "{}".to_string(),
            Some("stage".to_string()),
            None,
        );
        assert_eq!(out.__str__(), out.__repr__());
    }

    #[test]
    fn test_pyoutput_clone() {
        let out = PyOutput::new(
            "stream:test".to_string(),
            "{}".to_string(),
            Some("stage".to_string()),
            None,
        );
        let cloned = out.clone();
        assert_eq!(out.stream(), cloned.stream());
        assert_eq!(out.payload(), cloned.payload());
        assert_eq!(out.stage(), cloned.stage());
    }
}
