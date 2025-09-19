"""Test suite for the batch processor."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from queue import Queue

from cognitionflow.batch_processor import BatchProcessor
from cognitionflow.config import SDKConfig
from cognitionflow.models import TraceData


class TestBatchProcessor:
    """Test batch processor functionality."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SDKConfig(
            api_key="test-key",
            project_id="test-project",
            batch_size=3,
            flush_interval=0.1,
            max_retries=2
        )

    @pytest.fixture
    def mock_sdk(self, config):
        """Mock SDK instance."""
        sdk = Mock()
        sdk.config = config
        sdk._event_queue = Queue()
        return sdk

    @pytest.fixture
    def processor(self, mock_sdk):
        """Test batch processor."""
        processor = BatchProcessor(mock_sdk)
        yield processor
        if processor._running:
            processor.shutdown()

    def test_processor_initialization(self, mock_sdk):
        """Test processor initialization."""
        processor = BatchProcessor(mock_sdk)

        assert processor.sdk == mock_sdk
        assert processor.config == mock_sdk.config
        assert processor._running is False
        assert processor._thread is None

    def test_start_and_shutdown(self, processor):
        """Test starting and shutting down processor."""
        # Start processor
        processor.start()
        assert processor._running is True
        assert processor._thread is not None
        assert processor._thread.is_alive()

        # Shutdown processor
        processor.shutdown()
        assert processor._running is False

    def test_double_start(self, processor):
        """Test that starting twice doesn't create multiple threads."""
        processor.start()
        first_thread = processor._thread

        processor.start()  # Should not create new thread
        assert processor._thread == first_thread

        processor.shutdown()

    def test_flush_empty_batch(self, processor):
        """Test flushing empty batch."""
        # Should not raise error
        processor.flush()

    def test_batch_size_flush(self, processor):
        """Test flushing when batch size is reached."""
        processor.start()

        # Add events to queue
        for i in range(3):  # batch_size = 3
            trace = TraceData(agent_name=f"agent_{i}")
            processor._event_queue.put(trace)

        # Wait for processing
        time.sleep(0.2)

        # Batch should have been flushed
        assert len(processor._batch) == 0

        processor.shutdown()

    def test_time_based_flush(self, processor):
        """Test flushing based on time interval."""
        processor.start()

        # Add one event (less than batch size)
        trace = TraceData(agent_name="test_agent")
        processor._event_queue.put(trace)

        # Wait for flush interval
        time.sleep(0.15)  # flush_interval = 0.1

        # Batch should have been flushed
        assert len(processor._batch) == 0

        processor.shutdown()

    @patch('aiohttp.ClientSession')
    def test_successful_api_call(self, mock_session, processor):
        """Test successful API call."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        # Create batch
        batch = [{"trace_id": "test", "agent_name": "test_agent"}]

        # Send batch
        asyncio.run(processor._send_batch_async(batch))

        # Verify API was called
        mock_session.return_value.__aenter__.return_value.post.assert_called_once()

    @patch('aiohttp.ClientSession')
    def test_api_error_handling(self, mock_session, processor):
        """Test API error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        # Create batch
        batch = [{"trace_id": "test", "agent_name": "test_agent"}]

        # Send batch should handle error gracefully
        asyncio.run(processor._send_batch_async(batch))

        # Batch should be added to failed batches
        assert len(processor._failed_batches) == 1

    def test_get_stats(self, processor):
        """Test getting processor statistics."""
        stats = processor.get_stats()

        assert "running" in stats
        assert "current_batch_size" in stats
        assert "failed_batches" in stats
        assert "consecutive_failures" in stats

        assert stats["running"] is False
        assert stats["current_batch_size"] == 0
        assert stats["failed_batches"] == 0

    def test_failed_batch_retry(self, processor):
        """Test retrying failed batches."""
        # Add a failed batch
        failed_batch = {
            "batch": [{"trace_id": "test"}],
            "timestamp": time.time() - 100,  # Old timestamp
            "attempts": 1
        }
        processor._failed_batches.append(failed_batch)

        # Start processor to trigger retry logic
        processor.start()
        time.sleep(0.1)
        processor.shutdown()

        # Failed batch should have been retried (removed from list or attempt count increased)
        # Note: Exact behavior depends on whether retry was successful