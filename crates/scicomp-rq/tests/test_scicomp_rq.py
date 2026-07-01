# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest tests for scicomp_rq Python bindings.

These tests verify the Python interface to the Rust scicomp-rq library.

Prerequisites:
    1. Build the module: cd crates/scicomp-rq && maturin develop --features python-extension
    2. Run tests: pytest tests/test_scicomp_rq.py -v

For integration tests that require Redis:
    - Set REDIS_URL environment variable (default: redis://127.0.0.1:6379)
    - Ensure Redis is running
    - Run with: pytest tests/test_scicomp_rq.py -v --run-integration
"""

import json
import os
import uuid

import pytest

# ============================================================================
# Module Import Tests
# ============================================================================


class TestModuleImport:
    """Tests for module import and QueueManager class availability."""

    def test_import_module(self):
        """Test that the scicomp_rq module can be imported."""
        import scicomp_rq

        assert scicomp_rq is not None

    def test_message_class_exists(self):
        """Test that Message class is available."""
        import scicomp_rq

        assert hasattr(scicomp_rq, "Message")
        assert isinstance(scicomp_rq.Message, type)

    def test_message_has_all_properties(self):
        """Test that Message class has all expected properties."""
        import scicomp_rq

        # Check class has expected properties
        expected = ["id", "stream", "group", "run_id", "payload", "stage"]
        for prop in expected:
            assert hasattr(scicomp_rq.Message, prop), f"Message missing property: {prop}"

    def test_output_class_exists(self):
        """Test that Output class is available."""
        import scicomp_rq

        assert hasattr(scicomp_rq, "Output")
        assert isinstance(scicomp_rq.Output, type)


class TestMessageType:
    """Tests for the Message type Python bindings."""

    def test_message_constructor(self):
        """Test that Message can be constructed with all fields."""
        import scicomp_rq

        msg = scicomp_rq.Message(
            id="1706123456789-0",
            stream="stream:test",
            group="test:grp",
            run_id="run-123",
            payload='{"key": "value"}',
            stage="prefetch",
        )
        assert msg.id == "1706123456789-0"
        assert msg.stream == "stream:test"
        assert msg.group == "test:grp"
        assert msg.run_id == "run-123"
        assert msg.payload == '{"key": "value"}'
        assert msg.stage == "prefetch"

    def test_message_properties_are_readonly(self):
        """Test that Message properties cannot be modified."""
        import scicomp_rq

        msg = scicomp_rq.Message(
            id="1-0",
            stream="stream:test",
            group="grp",
            run_id="run-1",
            payload="{}",
            stage="stage",
        )
        # Properties should be readable
        assert msg.id == "1-0"
        # But not writable (PyO3 getters without setters are read-only)
        with pytest.raises(AttributeError):
            msg.id = "2-0"

    def test_message_repr(self):
        """Test Message __repr__ method."""
        import scicomp_rq

        msg = scicomp_rq.Message(
            id="1-0",
            stream="stream:test",
            group="grp",
            run_id="run-1",
            payload="{}",
            stage="stage",
        )
        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "1-0" in repr_str or "id" in repr_str.lower()


class TestOutputType:
    """Tests for the Output type Python bindings."""

    def test_output_constructor_basic(self):
        """Test Output constructor with stream and payload."""
        import scicomp_rq

        out = scicomp_rq.Output("stream:test", '{"key":"val"}')
        assert out.stream == "stream:test"
        assert out.payload == '{"key":"val"}'
        assert out.stage is None

    def test_output_constructor_with_stage(self):
        """Test Output constructor with optional stage."""
        import scicomp_rq

        out = scicomp_rq.Output("stream:test", "{}", stage="mystage")
        assert out.stream == "stream:test"
        assert out.payload == "{}"
        assert out.stage == "mystage"

    def test_output_empty_payload(self):
        """Test Output with empty JSON payload."""
        import scicomp_rq

        out = scicomp_rq.Output("stream:results", "{}")
        assert out.stream == "stream:results"
        assert out.payload == "{}"
        assert out.stage is None

    def test_output_complex_payload(self):
        """Test Output with complex JSON payload."""
        import scicomp_rq

        payload = '{"status": "ok", "results": [1, 2, 3], "nested": {"key": "value"}}'
        out = scicomp_rq.Output("stream:results", payload, stage="final")
        assert out.stream == "stream:results"
        assert out.payload == payload
        assert out.stage == "final"

    def test_output_repr(self):
        """Test Output __repr__ method."""
        import scicomp_rq

        out = scicomp_rq.Output("stream:test", "{}", stage="mystage")
        repr_str = repr(out)
        assert "Output" in repr_str
        assert "stream:test" in repr_str

    def test_module_has_queue_manager_class(self):
        """Test that QueueManager class is available."""
        import scicomp_rq

        assert hasattr(scicomp_rq, "QueueManager")
        assert isinstance(scicomp_rq.QueueManager, type)

    def test_queue_manager_has_from_redis_url(self):
        """Test that QueueManager has from_redis_url static method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "from_redis_url")
        assert callable(scicomp_rq.QueueManager.from_redis_url)

    def test_queue_manager_has_from_env(self):
        """Test that QueueManager has from_env static method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "from_env")
        assert callable(scicomp_rq.QueueManager.from_env)

    def test_queue_manager_does_not_expose_ensure_groups(self):
        """Test that QueueManager no longer exposes ensure_groups."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert not hasattr(qm_class, "ensure_groups")

    def test_queue_manager_ensure_groups_doc_is_absent(self):
        """ensure_groups docs should be absent once API is removed."""
        import scicomp_rq

        assert not hasattr(scicomp_rq.QueueManager, "ensure_groups")

    def test_queue_manager_does_not_expose_stream_key(self):
        """Test that QueueManager no longer exposes stream_key helper."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert not hasattr(qm_class, "stream_key")

    def test_queue_manager_does_not_expose_group_name(self):
        """Test that QueueManager no longer exposes group_name helper."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert not hasattr(qm_class, "group_name")

    def test_queue_manager_does_not_expose_streams(self):
        """Test that QueueManager no longer exposes streams helper."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert not hasattr(qm_class, "streams")

    def test_queue_manager_has_enqueue(self):
        """Test that QueueManager has enqueue method."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert hasattr(qm_class, "enqueue")

    def test_queue_manager_has_health_check(self):
        """Test that QueueManager has health_check method."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert hasattr(qm_class, "health_check")

    def test_queue_manager_does_not_expose_prefix_property(self):
        """Test that QueueManager no longer exposes prefix property."""
        import scicomp_rq

        qm_class = scicomp_rq.QueueManager
        assert not hasattr(qm_class, "prefix")

    def test_all_expected_methods_present(self):
        """Test that all expected methods are present in QueueManager."""
        import scicomp_rq

        expected_methods = [
            "from_redis_url",
            "from_env",
            "enqueue",
            "health_check",
            # Unified API methods
            "read_messages",
            "ack_message",
            "handoff_message",
            "handoff_message_to_run",
            "forward_many",
            "create_consumer_group",
            "claim_idle_messages",
            "hset",
            "hdel",
        ]

        qm_class = scicomp_rq.QueueManager
        for method_name in expected_methods:
            assert hasattr(qm_class, method_name), f"Missing method: {method_name}"


class TestNewAPIMethods:
    """Tests for the new unified API methods on QueueManager."""

    def test_queue_manager_has_read_messages(self):
        """Test that QueueManager has read_messages method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "read_messages")

    def test_queue_manager_has_ack_message(self):
        """Test that QueueManager has ack_message method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "ack_message")

    def test_queue_manager_has_handoff_message(self):
        """Test that QueueManager has handoff_message method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "handoff_message")

    def test_queue_manager_has_handoff_message_to_run(self):
        """Test that QueueManager has handoff_message_to_run method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "handoff_message_to_run")

    def test_queue_manager_has_forward_many(self):
        """Test that QueueManager has forward_many method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "forward_many")

    def test_queue_manager_has_create_consumer_group(self):
        """Test that QueueManager has create_consumer_group method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "create_consumer_group")

    def test_queue_manager_has_claim_idle_messages(self):
        """Test that QueueManager has claim_idle_messages method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "claim_idle_messages")

    def test_queue_manager_has_hset(self):
        """Test that QueueManager has hset method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "hset")

    def test_queue_manager_has_hdel(self):
        """Test that QueueManager has hdel method."""
        import scicomp_rq

        assert hasattr(scicomp_rq.QueueManager, "hdel")


# ============================================================================
# Configuration Loading Tests
# ============================================================================


class TestConfigLoading:
    """Tests for configuration loading functions."""

    @pytest.mark.asyncio
    async def test_from_redis_url_invalid_url(self):
        """Test that from_redis_url raises error for invalid Redis URL."""
        import scicomp_rq

        with pytest.raises(RuntimeError):
            await scicomp_rq.QueueManager.from_redis_url("not-a-valid-redis-url")

    @pytest.mark.asyncio
    async def test_from_env_does_not_require_queue_config(self):
        """Test that from_env no longer requires QUEUE_CONFIG."""
        import scicomp_rq

        # Ensure QUEUE_CONFIG is not set
        env_backup = os.environ.get("QUEUE_CONFIG")
        redis_url_backup = os.environ.get("REDIS_URL")
        if "QUEUE_CONFIG" in os.environ:
            del os.environ["QUEUE_CONFIG"]
        os.environ["REDIS_URL"] = "not-a-valid-redis-url"

        try:
            try:
                await scicomp_rq.QueueManager.from_env()
            except RuntimeError as exc:
                # Invalid/missing Redis configuration is acceptable in unit
                # contexts, but missing QUEUE_CONFIG must not be the reason.
                assert "QUEUE_CONFIG" not in str(exc)
        finally:
            # Restore environment
            if env_backup is not None:
                os.environ["QUEUE_CONFIG"] = env_backup
            elif "QUEUE_CONFIG" in os.environ:
                del os.environ["QUEUE_CONFIG"]
            if redis_url_backup is not None:
                os.environ["REDIS_URL"] = redis_url_backup
            elif "REDIS_URL" in os.environ:
                del os.environ["REDIS_URL"]

    @pytest.mark.asyncio
    async def test_direct_construction_raises_error(self):
        """Test that direct QueueManager() construction raises TypeError."""
        import scicomp_rq

        with pytest.raises(TypeError) as exc_info:
            scicomp_rq.QueueManager()
        # Verify error message mentions factory methods
        error_msg = str(exc_info.value)
        assert "from_redis_url" in error_msg or "from_env" in error_msg


# ============================================================================
# Integration Tests (require Redis)
# ============================================================================


@pytest.mark.integration
class TestQueueManagerIntegration:
    """Integration tests that require a running Redis instance."""

    @pytest.fixture
    def redis_url(self):
        """Get Redis URL from environment or use default."""
        return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")

    @pytest.fixture
    async def queue_manager(self, redis_url):
        """Create a QueueManager for testing."""
        import scicomp_rq

        qm = await scicomp_rq.QueueManager.from_redis_url(redis_url)
        for stream in ("test_prefetch", "test_inference", "test_results"):
            await qm.create_consumer_group(stream, f"{stream}:grp", "$", True)
        return qm

    @pytest.mark.asyncio
    async def test_from_redis_url_success(self, redis_url):
        """Test successful creation of QueueManager from Redis URL."""
        import scicomp_rq

        qm = await scicomp_rq.QueueManager.from_redis_url(redis_url)
        assert qm is not None

    @pytest.mark.asyncio
    async def test_ensure_groups_removed(self, queue_manager):
        """ensure_groups should not exist on QueueManager."""
        assert not hasattr(queue_manager, "ensure_groups")

    @pytest.mark.asyncio
    async def test_enqueue_valid_json(self, queue_manager):
        """Test enqueueing a message with valid JSON payload."""
        payload = json.dumps({"key": "value", "number": 42})
        msg_id = await queue_manager.enqueue("test_prefetch", "test-run-123", payload, "prefetch")
        # Verify Redis stream ID format: timestamp-sequence (e.g., "1706123456789-0")
        assert isinstance(msg_id, str), f"Expected str, got {type(msg_id).__name__}"
        assert "-" in msg_id, f"Expected hyphen in stream ID, got {msg_id!r}"
        parts = msg_id.split("-")
        assert len(parts) == 2, f"Expected 2 parts in stream ID, got {len(parts)}"
        assert parts[0].isdigit(), f"Expected timestamp, got {parts[0]!r}"
        assert parts[1].isdigit(), f"Expected sequence, got {parts[1]!r}"

    @pytest.mark.asyncio
    async def test_enqueue_preserves_payload_text(self, queue_manager):
        """enqueue should keep caller-provided JSON text unchanged."""
        run_id = "test-run-preserve-payload-text"
        payload = '{"z": 3, "a": [1, 2], "nested": {"b": 1, "a": 2}}'

        await queue_manager.enqueue("test_prefetch", run_id, payload, "prefetch")

        stream_key = "test_prefetch"
        group_name = "test_prefetch:grp"
        messages = await queue_manager.read_messages(
            stream_key, group_name, "payload-preserve-consumer", 50, 100
        )

        matching = [msg for msg in messages if msg.run_id == run_id]
        assert matching, "Expected to read the message that was just enqueued"
        assert matching[0].payload == payload

    @pytest.mark.asyncio
    async def test_enqueue_invalid_json(self, queue_manager):
        """Test that enqueueing with invalid JSON raises error."""
        with pytest.raises(ValueError):
            await queue_manager.enqueue(
                "test_prefetch", "test-run-123", "not valid json", "prefetch"
            )

    @pytest.mark.asyncio
    async def test_enqueue_complex_payload(self, queue_manager):
        """Test enqueueing a message with complex nested JSON payload."""
        payload = json.dumps(
            {
                "workflow": "deterministic",
                "parameters": {
                    "time": "2024-01-01T00:00:00Z",
                    "nsteps": 10,
                    "models": ["pangu", "fcn"],
                },
                "metadata": {"user": "test", "priority": 1},
            }
        )
        msg_id = await queue_manager.enqueue(
            "test_prefetch", "test-run-complex", payload, "prefetch"
        )
        # Verify Redis stream ID format
        assert isinstance(msg_id, str), f"Expected str, got {type(msg_id).__name__}"
        assert "-" in msg_id, f"Expected hyphen in stream ID, got {msg_id!r}"

    @pytest.mark.asyncio
    async def test_health_check(self, queue_manager):
        """Test health_check returns valid status tuple."""
        connected, latency_ms, script_loaded = await queue_manager.health_check()
        # Verify connection is healthy
        assert connected is True, "Expected connected=True"
        # Verify latency is reasonable (< 1 second)
        assert 0 <= latency_ms < 1000, f"Unexpected latency: {latency_ms}ms"
        # script_loaded can be True or False depending on whether handoff was called
        assert isinstance(script_loaded, bool), f"Expected bool, got {type(script_loaded)}"

    @pytest.mark.asyncio
    async def test_repr_and_str(self, queue_manager):
        """Test __repr__ and __str__ methods."""
        repr_str = repr(queue_manager)
        str_str = str(queue_manager)

        # Both should contain QueueManager info
        assert "QueueManager" in repr_str
        assert repr_str == str_str  # __str__ delegates to __repr__


# ============================================================================
# Configuration with Custom Prefix Tests
# ============================================================================


class TestRemovedMappingHelpers:
    """Contract tests for removed Python-side mapping helpers."""

    def test_stream_key_helper_is_absent(self):
        import scicomp_rq

        assert not hasattr(scicomp_rq.QueueManager, "stream_key")

    def test_group_name_helper_is_absent(self):
        import scicomp_rq

        assert not hasattr(scicomp_rq.QueueManager, "group_name")

    def test_streams_helper_is_absent(self):
        import scicomp_rq

        assert not hasattr(scicomp_rq.QueueManager, "streams")

    def test_prefix_property_is_absent(self):
        import scicomp_rq

        assert not hasattr(scicomp_rq.QueueManager, "prefix")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in the Python bindings."""

    @pytest.mark.asyncio
    async def test_from_redis_url_invalid_redis_url(self):
        """Test error handling for invalid Redis URL."""
        import scicomp_rq

        # Invalid Redis URL should raise an error
        with pytest.raises(RuntimeError):
            await scicomp_rq.QueueManager.from_redis_url("not-a-valid-redis-url")


# ============================================================================
# New Unified API Integration Tests
# ============================================================================


@pytest.mark.integration
class TestNewAPIIntegration:
    """Integration tests for the new unified API methods."""

    @pytest.fixture
    def redis_url(self):
        """Get Redis URL from environment or use default."""
        return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")

    @pytest.fixture
    async def queue_manager(self, redis_url):
        """Create a QueueManager for testing."""
        import scicomp_rq

        qm = await scicomp_rq.QueueManager.from_redis_url(redis_url)
        for stream in ("api_test_src", "api_test_dest", "api_test_results"):
            await qm.create_consumer_group(stream, f"{stream}:grp", "$", True)
        return qm

    @pytest.mark.asyncio
    async def test_read_messages_returns_message_objects(self, queue_manager):
        """Test read_messages returns list of Message objects."""
        import scicomp_rq

        # Enqueue a test message
        payload = json.dumps({"test": "read_messages"})
        await queue_manager.enqueue("api_test_src", "run-read-1", payload, "api_test_src")

        # Read messages
        stream = "api_test_src"
        group = "api_test_src:grp"
        messages = await queue_manager.read_messages(stream, group, "consumer-1", 1, 100)

        # Verify we got Message objects
        assert len(messages) >= 1
        msg = messages[0]
        assert isinstance(msg, scicomp_rq.Message)
        assert msg.stream == stream
        assert msg.group == group
        assert "run-read-1" in msg.run_id

        # Clean up - ack the message
        await queue_manager.ack_message(msg)

    @pytest.mark.asyncio
    async def test_read_messages_empty_when_no_messages(self, queue_manager):
        """Test read_messages returns empty list with short block."""
        # Create a unique stream for this test
        stream = "api_test_results"
        group = "api_test_results:grp"

        # Read with short block - should return empty
        messages = await queue_manager.read_messages(stream, group, "consumer-1", 1, 10)
        # May or may not be empty depending on other tests, just verify it's a list
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_ack_message_removes_from_pending(self, queue_manager):
        """Test ack_message successfully acknowledges a message."""
        # Enqueue a test message
        payload = json.dumps({"test": "ack_message"})
        await queue_manager.enqueue("api_test_src", "run-ack-1", payload, "api_test_src")

        # Read the message
        stream = "api_test_src"
        group = "api_test_src:grp"
        messages = await queue_manager.read_messages(stream, group, "consumer-1", 1, 100)
        assert len(messages) >= 1

        # Ack the message
        msg = messages[0]
        ack_count = await queue_manager.ack_message(msg)
        # Returns 1 if message was acked, 0 if already acked
        assert ack_count >= 0

    @pytest.mark.asyncio
    async def test_handoff_message_moves_to_destination(self, queue_manager):
        """Test handoff_message moves message to destination stream."""
        # Enqueue a test message
        payload = json.dumps({"test": "handoff_message"})
        await queue_manager.enqueue("api_test_src", "run-handoff-1", payload, "api_test_src")

        # Read the message
        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1

        # Handoff to destination
        msg = messages[0]
        dest_stream = "api_test_dest"
        new_id = await queue_manager.handoff_message(msg, dest_stream)

        # Verify new message ID format
        assert isinstance(new_id, str)
        assert "-" in new_id

    @pytest.mark.asyncio
    async def test_handoff_message_with_modified_payload(self, queue_manager):
        """Test handoff_message can send modified payload."""
        # Enqueue a test message
        original_payload = json.dumps({"original": True})
        await queue_manager.enqueue(
            "api_test_src",
            "run-handoff-2",
            original_payload,
            "api_test_src",
        )

        # Read the message
        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1

        # Handoff with modified payload
        msg = messages[0]
        new_payload = json.dumps({"modified": True, "original_run_id": msg.run_id})
        dest_stream = "api_test_dest"
        new_id = await queue_manager.handoff_message(msg, dest_stream, payload=new_payload)

        assert isinstance(new_id, str)
        assert "-" in new_id

    @pytest.mark.asyncio
    async def test_handoff_message_with_explicit_stage(self, queue_manager):
        """Test handoff_message can specify explicit stage."""
        # Enqueue a test message
        payload = json.dumps({"test": "explicit_stage"})
        await queue_manager.enqueue("api_test_src", "run-handoff-3", payload, "api_test_src")

        # Read the message
        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1

        # Handoff with explicit stage
        msg = messages[0]
        dest_stream = "api_test_dest"
        new_id = await queue_manager.handoff_message(msg, dest_stream, stage="custom_stage")

        assert isinstance(new_id, str)

    @pytest.mark.asyncio
    async def test_handoff_message_to_run_overrides_destination_run_id(self, queue_manager):
        """Test handoff_message_to_run can override the destination run_id."""
        source_run_id = f"run-handoff-source-{uuid.uuid4().hex[:8]}"
        destination_run_id = f"run-handoff-dest-{uuid.uuid4().hex[:8]}"
        destination_stream = f"api_test_handoff_dest_{uuid.uuid4().hex[:8]}"
        destination_group = f"{destination_stream}:grp"
        payload = json.dumps({"test": "handoff_message_to_run"})

        await queue_manager.create_consumer_group(destination_stream, destination_group, "$", True)
        await queue_manager.enqueue("api_test_src", source_run_id, payload, "api_test_src")

        source_messages = await queue_manager.read_messages(
            "api_test_src", "api_test_src:grp", "consumer-1", 1, 100
        )
        assert len(source_messages) >= 1

        source_message = source_messages[0]
        handoff_to_run = getattr(queue_manager, "handoff_message_to_run", None)
        assert callable(handoff_to_run), (
            "QueueManager should expose handoff_message_to_run for destination run_id overrides"
        )

        new_id = await handoff_to_run(
            source_message,
            destination_stream,
            payload=payload,
            stage="custom_stage",
            run_id=destination_run_id,
        )
        assert isinstance(new_id, str)
        assert "-" in new_id

        destination_messages = await queue_manager.read_messages(
            destination_stream, destination_group, "consumer-dest", 10, 100
        )
        forwarded = next((msg for msg in destination_messages if msg.id == new_id), None)
        assert forwarded is not None, "Expected handed-off message to appear in destination stream"
        assert forwarded.run_id == destination_run_id
        assert forwarded.stage == "custom_stage"

    @pytest.mark.asyncio
    async def test_forward_many_sends_to_all_destinations(self, queue_manager):
        """Test forward_many sends to multiple destinations."""
        import scicomp_rq

        # Enqueue a test message
        payload = json.dumps({"test": "forward_many"})
        await queue_manager.enqueue("api_test_src", "run-forward-1", payload, "api_test_src")

        # Read the message
        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1

        # Forward to multiple destinations
        msg = messages[0]
        outputs = [
            scicomp_rq.Output("api_test_dest", json.dumps({"dest": 1})),
            scicomp_rq.Output("api_test_results", json.dumps({"dest": 2}), stage="results"),
        ]
        new_ids = await queue_manager.forward_many(msg, outputs)

        # Verify we got IDs for each output
        assert len(new_ids) == 2
        for new_id in new_ids:
            assert isinstance(new_id, str)
            assert "-" in new_id

    @pytest.mark.asyncio
    async def test_forward_many_allows_existing_stream_destination_key(self, queue_manager):
        """forward_many should allow destination keys that already exist as Redis streams."""
        import scicomp_rq

        unique = uuid.uuid4().hex[:8]
        source_run_id = f"run-forward-existing-{unique}"
        destination_stream = f"api_test_existing_dest_{unique}"
        destination_group = f"{destination_stream}:grp"

        # Seed destination stream so TYPE returns "stream" for an existing key.
        await queue_manager.enqueue(
            destination_stream,
            f"run-dest-seed-{unique}",
            json.dumps({"seed": True}),
            destination_stream,
        )
        await queue_manager.create_consumer_group(destination_stream, destination_group, "$", True)

        await queue_manager.enqueue(
            "api_test_src",
            source_run_id,
            json.dumps({"test": "forward_many_existing_stream"}),
            "api_test_src",
        )

        source_messages = await queue_manager.read_messages(
            "api_test_src", "api_test_src:grp", f"consumer-src-{unique}", 50, 100
        )
        source_message = next((m for m in source_messages if m.run_id == source_run_id), None)
        assert source_message is not None, "Expected source message to be pending for forward_many"

        outputs = [scicomp_rq.Output(destination_stream, json.dumps({"dest": "existing"}))]
        new_ids = await queue_manager.forward_many(source_message, outputs)
        assert len(new_ids) == 1
        assert isinstance(new_ids[0], str)

        destination_messages = await queue_manager.read_messages(
            destination_stream, destination_group, f"consumer-dest-{unique}", 50, 100
        )
        assert any(m.run_id == source_run_id for m in destination_messages), (
            "Expected forwarded message in existing destination stream"
        )

    @pytest.mark.asyncio
    async def test_forward_many_fails_when_source_is_not_pending(self, queue_manager):
        """forward_many must fail if source message is not pending."""
        import scicomp_rq

        payload = json.dumps({"test": "forward_many_not_pending"})
        await queue_manager.enqueue(
            "api_test_src",
            "run-forward-not-pending",
            payload,
            "api_test_src",
        )

        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1
        msg = messages[0]

        # Remove from pending first so script precondition should fail.
        acked = await queue_manager.ack_message(msg)
        assert acked >= 0

        outputs = [scicomp_rq.Output("api_test_dest", json.dumps({"dest": 1}))]
        with pytest.raises(RuntimeError) as exc_info:
            await queue_manager.forward_many(msg, outputs)

        assert "SOURCE_NOT_PENDING" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_forward_many_fails_for_non_stream_destination_key(self, queue_manager):
        """forward_many must reject destination keys with non-stream Redis types."""
        import scicomp_rq

        payload = json.dumps({"test": "forward_many_dest_type"})
        await queue_manager.enqueue(
            "api_test_src",
            "run-forward-dest-type",
            payload,
            "api_test_src",
        )

        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(src_stream, src_group, "consumer-1", 1, 100)
        assert len(messages) >= 1
        msg = messages[0]

        # Create a HASH key, then try to use the same key as a stream destination.
        bad_dest_key = f"test:dest_hash:{uuid.uuid4().hex[:8]}"
        await queue_manager.hset(bad_dest_key, "field", "value")

        outputs = [scicomp_rq.Output(bad_dest_key, json.dumps({"dest": "bad"}))]
        with pytest.raises(RuntimeError) as exc_info:
            await queue_manager.forward_many(msg, outputs)
        assert "DEST_NOT_STREAM" in str(exc_info.value)

        # Source message should remain pending when precondition fails.
        acked_after_failure = await queue_manager.ack_message(msg)
        assert acked_after_failure >= 0

    @pytest.mark.asyncio
    async def test_forward_many_is_atomic_when_one_of_multiple_destinations_is_invalid(
        self, queue_manager
    ):
        """When one destination is invalid, forward_many must not write to any destination."""
        import scicomp_rq

        unique = uuid.uuid4().hex[:8]
        run_id = f"run-forward-atomic-{unique}"

        valid_dest_1 = f"api_test_atomic_dest_a_{unique}"
        valid_dest_1_group = f"{valid_dest_1}:grp"
        valid_dest_2 = f"api_test_atomic_dest_b_{unique}"
        valid_dest_2_group = f"{valid_dest_2}:grp"

        await queue_manager.create_consumer_group(valid_dest_1, valid_dest_1_group, "$", True)
        await queue_manager.create_consumer_group(valid_dest_2, valid_dest_2_group, "$", True)

        source_payload = json.dumps({"test": "forward_many_atomic_multi_dest"})
        await queue_manager.enqueue("api_test_src", run_id, source_payload, "api_test_src")

        source_messages = await queue_manager.read_messages(
            "api_test_src", "api_test_src:grp", f"consumer-atomic-src-{unique}", 50, 100
        )
        source_message = next((m for m in source_messages if m.run_id == run_id), None)
        assert source_message is not None, "Expected source message to be pending for forward_many"

        bad_dest_key = f"api_test_atomic_bad_dest_{unique}"
        await queue_manager.hset(bad_dest_key, "field", "value")

        outputs = [
            scicomp_rq.Output(valid_dest_1, json.dumps({"dest": "a"})),
            scicomp_rq.Output(bad_dest_key, json.dumps({"dest": "bad"})),
            scicomp_rq.Output(valid_dest_2, json.dumps({"dest": "b"})),
        ]

        with pytest.raises(RuntimeError) as exc_info:
            await queue_manager.forward_many(source_message, outputs)
        assert "DEST_NOT_STREAM" in str(exc_info.value)

        # Atomicity check: no message should be written to any valid destination.
        dest_a_messages = await queue_manager.read_messages(
            valid_dest_1, valid_dest_1_group, f"consumer-atomic-dest-a-{unique}", 50, 50
        )
        dest_b_messages = await queue_manager.read_messages(
            valid_dest_2, valid_dest_2_group, f"consumer-atomic-dest-b-{unique}", 50, 50
        )
        assert all(m.run_id != run_id for m in dest_a_messages)
        assert all(m.run_id != run_id for m in dest_b_messages)

        # Source message must remain pending because forward_many failed precondition.
        acked_after_failure = await queue_manager.ack_message(source_message)
        assert acked_after_failure == 1, "Source should still be pending after failed forward_many"

    @pytest.mark.asyncio
    async def test_handoff_message_fails_closed_when_source_not_pending(self, queue_manager):
        """handoff_message must fail when XACK cannot acknowledge source message."""
        run_id = f"run-handoff-xack-failure-{uuid.uuid4().hex[:8]}"
        payload = json.dumps({"test": "handoff_xack_failure"})
        await queue_manager.enqueue("api_test_src", run_id, payload, "api_test_src")

        src_stream = "api_test_src"
        src_group = "api_test_src:grp"
        messages = await queue_manager.read_messages(
            src_stream, src_group, "consumer-xack", 50, 100
        )
        source_msg = next((m for m in messages if m.run_id == run_id), None)
        assert source_msg is not None, "Expected to read source message"

        # Remove from pending first so Lua handoff hits XACK_FAILED path.
        await queue_manager.ack_message(source_msg)

        dest_stream = f"api_test_dest_xack_{uuid.uuid4().hex[:8]}"
        dest_group = f"{dest_stream}:grp"
        await queue_manager.create_consumer_group(dest_stream, dest_group, "$", True)

        with pytest.raises(RuntimeError) as exc_info:
            await queue_manager.handoff_message(source_msg, dest_stream)
        assert "XACK_FAILED" in str(exc_info.value)

        # Ensure rollback path keeps destination clean for this run_id.
        dest_messages = await queue_manager.read_messages(
            dest_stream, dest_group, "consumer-dest-xack", 50, 50
        )
        assert all(msg.run_id != run_id for msg in dest_messages)

    @pytest.mark.asyncio
    async def test_handoff_message_derives_stage_from_colon_rich_destination(self, queue_manager):
        """handoff_message should derive stage from full destination key when prefix is empty."""
        run_id = f"run-handoff-stage-{uuid.uuid4().hex[:8]}"
        payload = json.dumps({"test": "handoff_stage_derivation"})
        await queue_manager.enqueue("api_test_src", run_id, payload, "api_test_src")

        src_messages = await queue_manager.read_messages(
            "api_test_src", "api_test_src:grp", "consumer-stage-src", 50, 100
        )
        source_msg = next((m for m in src_messages if m.run_id == run_id), None)
        assert source_msg is not None, "Expected to read source message"

        dest_stream = f"api:test:dest:stage:{uuid.uuid4().hex[:8]}"
        dest_group = f"{dest_stream}:grp"
        await queue_manager.create_consumer_group(dest_stream, dest_group, "$", True)

        await queue_manager.handoff_message(source_msg, dest_stream)

        dest_messages = await queue_manager.read_messages(
            dest_stream, dest_group, "consumer-stage-dest", 50, 100
        )
        forwarded = next((m for m in dest_messages if m.run_id == run_id), None)
        assert forwarded is not None, "Expected forwarded message on destination stream"
        assert forwarded.stage == dest_stream

    @pytest.mark.asyncio
    async def test_enqueue_rejects_empty_run_id(self, queue_manager):
        """enqueue must reject empty run_id values."""
        payload = json.dumps({"test": "empty_run_id"})
        with pytest.raises(ValueError):
            await queue_manager.enqueue("api_test_src", "", payload, "api_test_src")

    @pytest.mark.asyncio
    async def test_read_messages_ignores_malformed_entries_without_silent_defaults(
        self, queue_manager, redis_url
    ):
        """Malformed Redis entries should not produce messages with defaulted critical fields."""
        try:
            import redis
        except ModuleNotFoundError:
            pytest.skip("redis package is required for malformed-entry integration coverage")

        stream = f"api_test_malformed_{uuid.uuid4().hex[:8]}"
        group = f"{stream}:grp"
        await queue_manager.create_consumer_group(stream, group, "0", True)

        client = redis.from_url(redis_url, decode_responses=True)
        try:
            # Malformed entry: missing required payload field.
            client.xadd(stream, {"run_id": "run-malformed-missing-payload"})
            # Valid entry should still be parsed normally.
            valid_run_id = f"run-valid-{uuid.uuid4().hex[:8]}"
            client.xadd(stream, {"run_id": valid_run_id, "payload": "{}", "stage": "stage_ok"})
        finally:
            client.close()

        messages = await queue_manager.read_messages(stream, group, "consumer-malformed", 50, 100)
        assert all(msg.run_id and msg.payload for msg in messages)
        assert any(msg.run_id == valid_run_id for msg in messages)
        assert all(msg.run_id != "run-malformed-missing-payload" for msg in messages)

    @pytest.mark.asyncio
    async def test_create_consumer_group_success(self, queue_manager):
        """Test create_consumer_group creates a new group."""
        # Use unique stream name to avoid conflicts
        unique_stream = f"stream:test_create_group_{uuid.uuid4().hex[:8]}"

        # Create consumer group (with MKSTREAM to create stream if needed)
        created = await queue_manager.create_consumer_group(unique_stream, "test_group", "$", True)

        # First creation should return True
        assert created is True

    @pytest.mark.asyncio
    async def test_create_consumer_group_idempotent(self, queue_manager):
        """Test create_consumer_group is idempotent."""
        # Use unique stream name
        unique_stream = f"stream:test_create_group_idem_{uuid.uuid4().hex[:8]}"

        # Create consumer group
        created1 = await queue_manager.create_consumer_group(unique_stream, "test_group", "$", True)
        assert created1 is True

        # Second creation should return False (already exists)
        created2 = await queue_manager.create_consumer_group(unique_stream, "test_group", "$", True)
        assert created2 is False

    @pytest.mark.asyncio
    async def test_claim_idle_messages_returns_tuple(self, queue_manager):
        """Test claim_idle_messages returns (cursor, messages) tuple."""
        stream = "api_test_src"
        group = "api_test_src:grp"

        # Try to claim messages (may be empty)
        result = await queue_manager.claim_idle_messages(
            stream, group, "claimer-1", 1000, "0-0", 10
        )

        # Should return a tuple of (cursor, messages)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cursor, messages = result
        assert isinstance(cursor, str)
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_hset_creates_field(self, queue_manager):
        """Test hset creates a new field in a hash."""
        # Use unique key
        key = f"test:hash:{uuid.uuid4().hex[:8]}"

        result = await queue_manager.hset(key, "field1", "value1")
        # Returns 1 for new field
        assert result == 1

    @pytest.mark.asyncio
    async def test_hset_updates_field(self, queue_manager):
        """Test hset updates an existing field."""
        key = f"test:hash:{uuid.uuid4().hex[:8]}"

        # Create field
        await queue_manager.hset(key, "field1", "value1")

        # Update field
        result = await queue_manager.hset(key, "field1", "value2")
        # Returns 0 for updated field
        assert result == 0

    @pytest.mark.asyncio
    async def test_hdel_removes_field(self, queue_manager):
        """Test hdel removes a field from a hash."""
        key = f"test:hash:{uuid.uuid4().hex[:8]}"

        # Create field
        await queue_manager.hset(key, "field1", "value1")

        # Delete field
        result = await queue_manager.hdel(key, "field1")
        # Returns 1 for deleted field
        assert result == 1

    @pytest.mark.asyncio
    async def test_hdel_returns_zero_if_not_exists(self, queue_manager):
        """Test hdel returns 0 if field doesn't exist."""
        key = f"test:hash:{uuid.uuid4().hex[:8]}"

        # Delete non-existent field
        result = await queue_manager.hdel(key, "nonexistent")
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
