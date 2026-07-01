# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for scicomp_rq - Redis Streams Queue Manager.

This file provides type hints for IDE support (VS Code, PyCharm, etc.).
"""

from typing import Dict, List, Optional, Tuple  # noqa: F401, I001, UP035

class Message:
    """
    A message read from a Redis stream.

    Contains all information needed for acknowledgment and handoff operations.
    The `stream` and `group` fields enable self-contained operations like
    `ack_message()` and `handoff_message()` without requiring the caller
    to track this context separately.

    Example:
        >>> msg = Message(
        ...     id="1706123456789-0",
        ...     stream="stream:prefetch",
        ...     group="prefetch:grp",
        ...     run_id="run-001",
        ...     payload='{"model": "pangu"}',
        ...     stage="prefetch",
        ... )
        >>> print(msg.stream)  # "stream:prefetch"
    """

    def __init__(
        self,
        id: str,
        stream: str,
        group: str,
        run_id: str,
        payload: str,
        stage: str,
    ) -> None:
        """
        Create a new Message with all fields.

        Args:
            id: Redis stream message ID (e.g., "1706123456789-0")
            stream: Full Redis stream key (e.g., "stream:prefetch")
            group: Consumer group name (e.g., "prefetch:grp")
            run_id: Unique workflow run identifier
            payload: JSON-encoded payload data
            stage: Current processing stage name
        """
        ...

    @property
    def id(self) -> str:
        """Redis stream message ID (e.g., "1706123456789-0")."""
        ...

    @property
    def stream(self) -> str:
        """Full Redis stream key where message was read from (e.g., "stream:prefetch")."""
        ...

    @property
    def group(self) -> str:
        """Consumer group name (e.g., "prefetch:grp")."""
        ...

    @property
    def run_id(self) -> str:
        """Unique identifier for this workflow run."""
        ...

    @property
    def payload(self) -> str:
        """JSON-encoded payload data."""
        ...

    @property
    def stage(self) -> str:
        """Current processing stage."""
        ...

    def __repr__(self) -> str:
        """Return string representation of the Message."""
        ...

    def __str__(self) -> str:
        """Return string representation of the Message."""
        ...

class Output:
    """
    Output destination for `forward_many()` operation.

    Specifies where to send a message and optionally what stage name to use.

    Example:
        >>> # Default stage behavior uses the full destination stream key
        >>> out1 = Output("stream:results", '{"status": "ok"}')
        >>>
        >>> # Explicit stage name
        >>> out2 = Output("stream:results", '{"status": "ok"}', stage="final_results")
        >>> out3 = Output("stream:execute", '{"status": "ok"}', run_id="child-run-1")
    """

    def __init__(
        self,
        stream: str,
        payload: str,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Create a new Output destination.

        Args:
            stream: Destination stream (full Redis key)
            payload: JSON payload for this destination
            stage: Optional stage name (None = destination stream key)
            run_id: Optional run_id override (None = preserve source message run_id)
        """
        ...

    @property
    def stream(self) -> str:
        """Destination stream (full Redis key)."""

    @property
    def run_id(self) -> str | None:
        """Explicit run_id override when provided."""
        ...

    @property
    def payload(self) -> str:
        """JSON payload for this destination."""
        ...

    @property
    def stage(self) -> str | None:
        """Stage name (None = destination stream key)."""
        ...

    def __repr__(self) -> str:
        """Return string representation of the Output."""
        ...

    def __str__(self) -> str:
        """Return string representation of the Output."""
        ...

class QueueManager:
    """
    Redis Streams queue manager for scientific computing pipelines.

    Use `QueueManager.from_redis_url()` or `QueueManager.from_env()` to create instances.
    Direct construction via `QueueManager()` is not allowed and will raise TypeError.

    Example:
        >>> qm = await QueueManager.from_redis_url("redis://localhost:6379")
        >>> await qm.create_consumer_group("prefetch", "prefetch:grp", "$", True)
        >>> msg_id = await qm.enqueue("prefetch", "run-001", '{"key": "value"}', "prefetch")
    """

    @staticmethod
    async def from_redis_url(url: str) -> QueueManager:
        """
        Create a QueueManager from a Redis connection URL.

        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379")
        Returns:
            Configured QueueManager instance

        Raises:
            RuntimeError: If connection fails
        """
        ...

    @staticmethod
    async def from_env() -> QueueManager:
        """
        Create a QueueManager from environment variables.

        Environment variables:
            REDIS_URL: Connection URL (default: redis://127.0.0.1:6379)

        Returns:
            Configured QueueManager instance

        Raises:
            RuntimeError: If connection fails
        """
        ...

    async def enqueue(
        self,
        stream_name: str,
        run_id: str,
        payload: str,
        stage: str,
    ) -> str:
        """
        Enqueue a message to a stream.

        Args:
            stream_name: Logical stream name (e.g., "prefetch")
            run_id: Unique identifier for this workflow run
            payload: JSON-encoded payload string
            stage: Stage name to record

        Note:
            Logical stream names must not contain ":".

        Returns:
            Redis stream message ID (e.g., "1706123456789-0")

        Raises:
            ValueError: If payload is not valid JSON
            ValueError: If stream_name/run_id/stage validation fails
            RuntimeError: If Redis operation fails
        """
        ...

    async def health_check(self) -> Tuple[bool, int, bool]:  # noqa: UP006
        """
        Check Redis connection health.

        Returns:
            Tuple of (connected, latency_ms, script_loaded):
                - connected: Whether Redis responded to PING
                - latency_ms: Round-trip latency in milliseconds
                - script_loaded: Whether Lua handoff script is cached

        Raises:
            RuntimeError: If the health check fails
        """
        ...

    async def read_messages(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block_ms: int = 0,
    ) -> list[Message]:
        """
        Read messages from a stream using XREADGROUP.

        Returns Message objects with stream and group fields populated,
        enabling self-contained ack_message() and handoff_message() calls.

        Args:
            stream: Full Redis stream key (e.g., "stream:prefetch")
            group: Consumer group name (e.g., "prefetch:grp")
            consumer: Consumer name within the group
            count: Maximum number of messages to read (default: 1)
            block_ms: How long to block waiting for messages (default: 0)

        Returns:
            List of Message objects (may be empty if no messages available)

        Raises:
            RuntimeError: If XREADGROUP fails
        """
        ...

    async def ack_message(self, message: Message) -> int:
        """
        Acknowledge a single message.

        Uses the stream and group information stored in the Message object,
        so you don't need to pass them separately.

        Args:
            message: The Message to acknowledge

        Returns:
            Number of messages acknowledged (1 if successful, 0 if already acked)

        Raises:
            RuntimeError: If XACK fails
        """
        ...

    async def handoff_message(
        self,
        message: Message,
        dest_stream: str,
        payload: str | None = None,
        stage: str | None = None,
    ) -> str:
        """
        Atomically hand off a message to a destination stream.

        This method:
        1. Sends the message to the destination stream (XADD)
        2. Acknowledges the original message (XACK)
        3. Updates the run hash with the new stage (HSET)

        When `stage` is omitted, the full destination stream key is used.

        Args:
            message: The Message to hand off
            dest_stream: Destination stream (full Redis key)
            payload: Optional new payload (None = use message's original payload)
            stage: Optional stage name (None = destination stream key)

        Returns:
            New message ID in the destination stream

        Raises:
            RuntimeError: If the handoff fails
        """
        ...

    async def handoff_message_to_run(
        self,
        message: Message,
        dest_stream: str,
        payload: str | None = None,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        Atomically hand off a message to a destination stream with an optional
        destination run_id override.

        This behaves like `handoff_message()`, but allows the caller to change
        the run_id stored on the destination message.

        Args:
            message: The Message to hand off
            dest_stream: Destination stream (full Redis key)
            payload: Optional new payload (None = use message's original payload)
            stage: Optional stage name (None = destination stream key)
            run_id: Optional destination run_id override (None = preserve source run_id)

        Returns:
            New message ID in the destination stream

        Raises:
            RuntimeError: If the handoff fails
        """
        ...

    async def forward_many(
        self,
        message: Message,
        outputs: list[Output],
    ) -> list[str]:
        """
        Forward a message to multiple destinations atomically.

        This is useful for fan-out patterns where a message needs to go to
        multiple streams (e.g., results + GPU release).

        The original message is acknowledged after all destinations receive it.

        Args:
            message: The Message to forward
            outputs: List of Output objects specifying destinations
                Must contain at least one output.

        Returns:
            List of new message IDs (one per output, in same order)

        Raises:
            ValueError: If outputs is empty
            RuntimeError: If any destination fails
        """
        ...

    async def create_consumer_group(
        self,
        stream: str,
        group: str,
        start_id: str = "$",
        create_stream: bool = True,
    ) -> bool:
        """
        Create a consumer group on a stream.

        This is idempotent - calling it multiple times is safe.

        Args:
            stream: Full Redis stream key
            group: Consumer group name
            start_id: ID to start reading from ("$" = new messages only, "0" = all)
            create_stream: If True, create the stream if it doesn't exist

        Returns:
            True if group was created, False if it already existed

        Raises:
            RuntimeError: If creation fails (other than already existing)
        """
        ...

    async def claim_idle_messages(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int,
        start_id: str,
        count: int,
    ) -> tuple[str, list[Message]]:
        """
        Claim idle pending messages using XAUTOCLAIM.

        Use this to reclaim messages from crashed consumers. Returns full
        Message objects with stream/group context for self-contained operations.

        Args:
            stream: Full Redis stream key
            group: Consumer group name
            consumer: New consumer to assign messages to
            min_idle_ms: Minimum idle time (only claim messages idle longer)
            start_id: Start scanning from this ID ("0-0" for beginning)
            count: Maximum messages to claim

        Returns:
            Tuple of (next_cursor_id, list_of_claimed_messages)

        Raises:
            RuntimeError: If XAUTOCLAIM fails
        """
        ...

    async def hset(self, key: str, field: str, value: str) -> int:
        """
        Set a field in a Redis hash.

        Args:
            key: Hash key
            field: Field name
            value: Field value

        Returns:
            1 if new field was created, 0 if existing field was updated

        Raises:
            RuntimeError: If HSET fails
        """
        ...

    async def hdel(self, key: str, field: str) -> int:
        """
        Delete a field from a Redis hash.

        Args:
            key: Hash key
            field: Field name to delete

        Returns:
            1 if field was deleted, 0 if field didn't exist

        Raises:
            RuntimeError: If HDEL fails
        """
        ...

    async def hgetall(self, key: str) -> Dict[str, str]:  # noqa: UP006
        """
        Get all fields and values from a Redis hash.

        Args:
            key: Hash key

        Returns:
            Dict of field-value pairs. Empty dict if key doesn't exist.

        Raises:
            RuntimeError: If HGETALL fails
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the QueueManager."""
        ...

    def __str__(self) -> str:
        """Return string representation of the QueueManager."""
        ...
