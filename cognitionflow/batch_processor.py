"""Batch processor for handling asynchronous data transmission."""

import asyncio
import json
import logging
import time
import threading
from dataclasses import asdict
from queue import Queue, Empty
from threading import Thread
from typing import List, Dict, Any, Optional

from .models import TraceData, BatchPayload
from .serialization import serialize_for_api

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batching and sending of trace events.

    This class runs in a background thread and periodically collects
    trace events from a queue, batches them, and sends them to the
    CognitionFlow API using asynchronous HTTP requests.

    Features:
    - Automatic batching based on size and time intervals
    - Exponential backoff retry logic
    - Graceful degradation on failures
    - Memory-efficient queue management
    """

    def __init__(self, sdk: 'CognitionFlowSDK'):
        """Initialize the batch processor.

        Args:
            sdk: Reference to the main SDK instance
        """
        self.sdk = sdk
        self.config = sdk.config
        self._event_queue = sdk._event_queue
        self._batch: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[Thread] = None
        self._last_flush = time.time()
        self._failed_batches: List[Dict[str, Any]] = []
        self._consecutive_failures = 0

    def start(self):
        """Start the batch processor thread."""
        if self._running:
            logger.warning("Batch processor already running")
            return

        self._running = True
        self._thread = Thread(target=self._process_loop, daemon=True, name="CognitionFlow-BatchProcessor")
        self._thread.start()
        logger.info("Batch processor started")

    def shutdown(self, timeout: float = 5.0):
        """Shutdown the batch processor and flush remaining events.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return

        logger.info("Shutting down batch processor...")
        self._running = False

        # Process remaining events in the queue
        self._process_remaining_events()

        # Flush any remaining batch
        if self._batch:
            self._flush_batch()

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        logger.info("Batch processor shutdown completed")

    def flush(self):
        """Manually flush the current batch."""
        if self._batch:
            self._flush_batch()
            logger.debug("Manual flush completed")

    def _process_loop(self):
        """Main processing loop that runs in the background thread."""
        logger.debug("Batch processor loop started")

        while self._running:
            try:
                # Process events from queue
                self._process_queue()

                # Retry failed batches
                self._retry_failed_batches()

                # Check if we need to flush based on time or size
                if self._should_flush():
                    self._flush_batch()

                # Small delay to prevent busy waiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                # Increase delay on errors to prevent error storms
                time.sleep(min(1.0, 0.1 * (self._consecutive_failures + 1)))

    def _process_queue(self):
        """Process events from the queue and add them to the current batch."""
        events_processed = 0
        max_events_per_cycle = 50  # Limit to prevent blocking too long

        while len(self._batch) < self.config.batch_size and events_processed < max_events_per_cycle:
            try:
                # Get event with short timeout to avoid blocking
                event = self._event_queue.get(timeout=0.1)

                # Convert TraceData to dictionary
                if isinstance(event, TraceData):
                    event_dict = event.to_dict()
                else:
                    event_dict = event

                self._batch.append(event_dict)
                events_processed += 1

            except Empty:
                # No more events in queue
                break
            except Exception as e:
                logger.error(f"Error processing queue item: {e}")
                break

        if events_processed > 0:
            logger.debug(f"Processed {events_processed} events, batch size: {len(self._batch)}")

    def _process_remaining_events(self):
        """Process any remaining events in the queue during shutdown."""
        remaining_events = []

        try:
            while True:
                try:
                    event = self._event_queue.get(timeout=0.1)
                    if isinstance(event, TraceData):
                        remaining_events.append(event.to_dict())
                    else:
                        remaining_events.append(event)
                except Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing remaining events: {e}")

        if remaining_events:
            self._batch.extend(remaining_events)
            logger.info(f"Added {len(remaining_events)} remaining events to final batch")

    def _should_flush(self) -> bool:
        """Check if the current batch should be flushed.

        Returns:
            True if batch should be flushed, False otherwise
        """
        if not self._batch:
            return False

        # Flush if batch is full
        if len(self._batch) >= self.config.batch_size:
            return True

        # Flush if enough time has passed
        time_since_last_flush = time.time() - self._last_flush
        if time_since_last_flush >= self.config.flush_interval:
            return True

        return False

    def _flush_batch(self):
        """Send the current batch to the API."""
        if not self._batch:
            return

        batch_to_send = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()

        logger.debug(f"Flushing batch with {len(batch_to_send)} events")

        # Send batch in a separate thread to avoid blocking
        send_thread = Thread(
            target=self._send_batch_sync,
            args=(batch_to_send,),
            daemon=True,
            name="CognitionFlow-BatchSender"
        )
        send_thread.start()

    def _send_batch_sync(self, batch: List[Dict[str, Any]]):
        """Send batch synchronously in a separate thread.

        Args:
            batch: List of events to send
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async send operation
            loop.run_until_complete(self._send_batch_async(batch))

        except Exception as e:
            logger.error(f"Error in batch sender thread: {e}")
            # Add batch to failed batches for retry
            self._failed_batches.append({
                "batch": batch,
                "timestamp": time.time(),
                "attempts": 1
            })
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _send_batch_async(self, batch: List[Dict[str, Any]]):
        """Send batch to the API asynchronously.

        Args:
            batch: List of events to send
        """
        if not batch:
            return

        # Create batch payload
        payload = BatchPayload(
            project_id=self.config.project_id,
            events=batch,
            environment=self.config.environment
        )

        for attempt in range(self.config.max_retries):
            try:
                # Import aiohttp here to avoid import errors if not installed
                try:
                    import aiohttp
                except ImportError:
                    logger.error("aiohttp is required for SDK functionality. Install it with: pip install aiohttp")
                    return

                # Prepare request
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"CognitionFlow-Python-SDK/0.1.0"
                }

                url = f"{self.config.endpoint.rstrip('/')}/api/v1/traces"
                payload_json = serialize_for_api(payload.to_dict())

                # Send request
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, data=payload_json, headers=headers) as response:
                        if response.status < 400:
                            logger.debug(f"Successfully sent batch of {len(batch)} events")
                            self._consecutive_failures = 0
                            return
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")

            except Exception as e:
                self._consecutive_failures += 1

                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to send batch after {self.config.max_retries} attempts: {e}")
                    # Add to failed batches for later retry
                    self._failed_batches.append({
                        "batch": batch,
                        "timestamp": time.time(),
                        "attempts": self.config.max_retries
                    })
                else:
                    # Exponential backoff
                    delay = min(60.0, 2 ** attempt + (time.time() % 1))  # Add jitter
                    logger.warning(f"Batch send attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)

    def _retry_failed_batches(self):
        """Retry sending failed batches."""
        if not self._failed_batches:
            return

        current_time = time.time()
        batches_to_retry = []

        # Check which batches are ready for retry
        for failed_batch in self._failed_batches[:]:
            time_since_failure = current_time - failed_batch["timestamp"]

            # Retry after exponential backoff delay
            retry_delay = min(300.0, 30 * (2 ** (failed_batch["attempts"] - 1)))

            if time_since_failure >= retry_delay:
                # Don't retry too many times
                if failed_batch["attempts"] < 10:
                    batches_to_retry.append(failed_batch)
                    self._failed_batches.remove(failed_batch)
                else:
                    # Give up on this batch
                    logger.error(f"Giving up on batch after {failed_batch['attempts']} attempts")
                    self._failed_batches.remove(failed_batch)

        # Retry batches
        for failed_batch in batches_to_retry:
            logger.info(f"Retrying failed batch (attempt {failed_batch['attempts'] + 1})")
            failed_batch["attempts"] += 1
            failed_batch["timestamp"] = current_time

            # Send in separate thread
            send_thread = Thread(
                target=self._send_batch_sync,
                args=(failed_batch["batch"],),
                daemon=True,
                name="CognitionFlow-BatchRetry"
            )
            send_thread.start()

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics.

        Returns:
            Dictionary containing processor statistics
        """
        return {
            "running": self._running,
            "current_batch_size": len(self._batch),
            "failed_batches": len(self._failed_batches),
            "consecutive_failures": self._consecutive_failures,
            "queue_size": self._event_queue.qsize() if hasattr(self._event_queue, 'qsize') else 0,
            "last_flush": self._last_flush,
        }