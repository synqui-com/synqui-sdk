"""Test suite for the CognitionFlow SDK."""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from queue import Queue

from cognitionflow.sdk import CognitionFlowSDK
from cognitionflow.config import SDKConfig
from cognitionflow.models import TraceData, SpanStatus
from cognitionflow.context import get_current_span, set_current_span


class TestSDKConfig:
    """Test SDK configuration."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project"
        )
        assert config.api_key == "test-key"
        assert config.project_id == "test-project"
        assert config.enabled is True

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="api_key is required"):
            SDKConfig(api_key="", project_id="test-project")

    def test_missing_project_id(self):
        """Test error when project ID is missing."""
        with pytest.raises(ValueError, match="project_id is required"):
            SDKConfig(api_key="test-key", project_id="")

    def test_disabled_config(self):
        """Test configuration when SDK is disabled."""
        config = SDKConfig(
            api_key="",
            project_id="",
            enabled=False
        )
        # Should not raise error when disabled
        assert config.enabled is False

    def test_invalid_batch_size(self):
        """Test error with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SDKConfig(
                api_key="test-key",
                project_id="test-project",
                batch_size=0
            )

    def test_invalid_flush_interval(self):
        """Test error with invalid flush interval."""
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            SDKConfig(
                api_key="test-key",
                project_id="test-project",
                flush_interval=-1.0
            )


class TestCognitionFlowSDK:
    """Test suite for CognitionFlow SDK."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SDKConfig(
            api_key="test-key",
            project_id="test-project",
            endpoint="http://test-endpoint",
            enabled=True,
            batch_size=10,
            flush_interval=1.0
        )

    @pytest.fixture
    def disabled_config(self):
        """Disabled test configuration."""
        return SDKConfig(
            api_key="test-key",
            project_id="test-project",
            enabled=False
        )

    @pytest.fixture
    def sdk(self, config):
        """Test SDK instance."""
        sdk = CognitionFlowSDK(config)
        yield sdk
        sdk.shutdown()

    @pytest.fixture
    def disabled_sdk(self, disabled_config):
        """Disabled test SDK instance."""
        return CognitionFlowSDK(disabled_config)

    def test_sdk_initialization(self, config):
        """Test SDK initialization."""
        sdk = CognitionFlowSDK(config)

        assert sdk.config == config
        assert sdk.is_enabled() is True
        assert sdk._batch_processor is not None

        sdk.shutdown()

    def test_disabled_sdk_initialization(self, disabled_config):
        """Test disabled SDK initialization."""
        sdk = CognitionFlowSDK(disabled_config)

        assert sdk.is_enabled() is False
        assert sdk._batch_processor is None

    def test_sync_trace_decorator_success(self, sdk):
        """Test successful synchronous function tracing."""
        @sdk.trace("test_agent")
        def test_function(x: int, y: int) -> int:
            return x + y

        result = test_function(2, 3)
        assert result == 5

        # Verify trace was queued
        assert sdk.get_queue_size() == 1

    def test_sync_trace_decorator_with_tags(self, sdk):
        """Test synchronous tracing with tags."""
        @sdk.trace("test_agent", tags={"environment": "test"})
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"
        assert sdk.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_async_trace_decorator_success(self, sdk):
        """Test successful asynchronous function tracing."""
        @sdk.trace("test_agent")
        async def test_function(x: int, y: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y

        result = await test_function(2, 3)
        assert result == 5

        # Verify trace was queued
        assert sdk.get_queue_size() == 1

    def test_sync_trace_decorator_error(self, sdk):
        """Test error handling in synchronous traced functions."""
        @sdk.trace("test_agent")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Verify error trace was queued
        assert sdk.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_async_trace_decorator_error(self, sdk):
        """Test error handling in asynchronous traced functions."""
        @sdk.trace("test_agent")
        async def failing_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_function()

        # Verify error trace was queued
        assert sdk.get_queue_size() == 1

    def test_span_context_manager(self, sdk):
        """Test span context manager."""
        with sdk.span("test_operation") as span:
            assert isinstance(span, TraceData)
            span.set_attribute("test_key", "test_value")
            # Simulate some work
            time.sleep(0.01)

        # Verify trace was queued
        assert sdk.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_async_span_context_manager(self, sdk):
        """Test async span context manager."""
        async with sdk.async_span("test_operation") as span:
            assert isinstance(span, TraceData)
            span.set_attribute("test_key", "test_value")
            await asyncio.sleep(0.01)

        # Verify trace was queued
        assert sdk.get_queue_size() == 1

    def test_span_context_manager_error(self, sdk):
        """Test span context manager with error."""
        with pytest.raises(ValueError, match="Test error"):
            with sdk.span("test_operation") as span:
                span.set_attribute("test_key", "test_value")
                raise ValueError("Test error")

        # Verify error trace was queued
        assert sdk.get_queue_size() == 1

    def test_nested_spans(self, sdk):
        """Test nested span creation."""
        @sdk.trace("outer_agent")
        def outer_function():
            with sdk.span("inner_operation") as inner_span:
                inner_span.set_attribute("nested", True)
                return "result"

        result = outer_function()
        assert result == "result"

        # Should have two traces (outer and inner)
        assert sdk.get_queue_size() == 2

    def test_input_output_capture(self, sdk):
        """Test input and output capture."""
        @sdk.trace("test_agent")
        def test_function(data: dict, multiplier: int = 2) -> dict:
            return {"processed": data["input"] * multiplier}

        result = test_function({"input": 5}, multiplier=3)
        assert result == {"processed": 15}

        # Verify trace was captured
        assert sdk.get_queue_size() == 1

    def test_disabled_sdk_no_tracing(self, disabled_sdk):
        """Test that disabled SDK doesn't perform tracing."""
        @disabled_sdk.trace("test_agent")
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

        # No traces should be queued
        assert disabled_sdk.get_queue_size() == 0

    def test_disabled_sdk_span_context(self, disabled_sdk):
        """Test disabled SDK span context manager."""
        with disabled_sdk.span("test_operation") as span:
            # Should still work but not capture data
            span.set_attribute("test_key", "test_value")

        # No traces should be queued
        assert disabled_sdk.get_queue_size() == 0

    def test_global_tags(self, config):
        """Test global tags from configuration."""
        config.tags = {"global_tag": "global_value"}
        sdk = CognitionFlowSDK(config)

        @sdk.trace("test_agent")
        def test_function():
            return "result"

        test_function()

        # Verify trace was queued
        assert sdk.get_queue_size() == 1
        sdk.shutdown()

    def test_flush(self, sdk):
        """Test manual flush operation."""
        @sdk.trace("test_agent")
        def test_function():
            return "result"

        test_function()
        assert sdk.get_queue_size() == 1

        # Flush should not raise error (exact behavior depends on batch processor)
        sdk.flush()

    def test_shutdown(self, config):
        """Test SDK shutdown."""
        sdk = CognitionFlowSDK(config)

        @sdk.trace("test_agent")
        def test_function():
            return "result"

        test_function()

        # Shutdown should complete without error
        sdk.shutdown()

        # After shutdown, batch processor should be None
        assert sdk._batch_processor is None


class TestTraceData:
    """Test TraceData model."""

    def test_trace_data_creation(self):
        """Test trace data creation."""
        trace = TraceData(
            agent_name="test_agent",
            function_name="test_function"
        )

        assert trace.agent_name == "test_agent"
        assert trace.function_name == "test_function"
        assert trace.status == SpanStatus.RUNNING
        assert trace.trace_id is not None
        assert trace.span_id is not None

    def test_set_attribute(self):
        """Test setting attributes."""
        trace = TraceData()
        trace.set_attribute("key", "value")

        assert trace.attributes["key"] == "value"

    def test_set_tag(self):
        """Test setting tags."""
        trace = TraceData()
        trace.set_tag("environment", "test")

        assert trace.tags["environment"] == "test"

    def test_set_error(self):
        """Test setting error information."""
        trace = TraceData()
        error = ValueError("Test error")
        trace.set_error(error)

        assert trace.status == SpanStatus.FAILED
        assert trace.error is not None
        assert trace.error["type"] == "ValueError"
        assert trace.error["message"] == "Test error"

    def test_finish(self):
        """Test finishing a trace."""
        trace = TraceData()
        time.sleep(0.01)  # Small delay
        trace.finish()

        assert trace.status == SpanStatus.COMPLETED
        assert trace.end_time is not None
        assert trace.duration_ms is not None
        assert trace.duration_ms > 0

    def test_to_dict(self):
        """Test converting trace to dictionary."""
        trace = TraceData(
            agent_name="test_agent",
            function_name="test_function"
        )
        trace.set_attribute("key", "value")
        trace.finish()

        trace_dict = trace.to_dict()

        assert trace_dict["agent_name"] == "test_agent"
        assert trace_dict["function_name"] == "test_function"
        assert trace_dict["status"] == "completed"
        assert trace_dict["attributes"]["key"] == "value"


class TestContextManagement:
    """Test context management."""

    def test_context_isolation(self):
        """Test that contexts are properly isolated between threads."""
        results = {}

        def thread_function(thread_id):
            trace = TraceData(span_id=f"span_{thread_id}")
            set_current_span(trace)

            # Small delay to allow other threads to set their contexts
            time.sleep(0.01)

            current = get_current_span()
            results[thread_id] = current.span_id if current else None

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Each thread should have its own context
        for i in range(5):
            assert results[i] == f"span_{i}"

    def test_nested_context(self):
        """Test nested context management."""
        outer_trace = TraceData(span_id="outer")
        inner_trace = TraceData(span_id="inner")

        set_current_span(outer_trace)
        assert get_current_span().span_id == "outer"

        # Nested context
        from cognitionflow.context import span_context
        with span_context(inner_trace):
            assert get_current_span().span_id == "inner"

        # Should restore outer context
        assert get_current_span().span_id == "outer"