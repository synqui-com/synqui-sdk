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
import uuid

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
        print(f"🔍 Batch processor: Started batch processor thread")

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
        # Ensure we pull any pending events from the queue
        try:
            self._process_queue()
        except Exception as e:
            logger.error(f"Error processing queue before flush: {e}")

        print(f"🔍 Batch processor: Flush called, batch size: {len(self._batch)}")
        if not self._batch:
            print("🔍 Batch processor: No batch to flush")
            return

        # Perform a synchronous send so the caller waits until delivery
        batch_to_send = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()
        logger.debug(f"Flushing batch with {len(batch_to_send)} events")
        print(f"🔍 Batch processor: Manual flush with {len(batch_to_send)} events")
        self._send_batch_sync(batch_to_send)
        logger.debug("Manual flush completed")
        print("🔍 Batch processor: Manual flush completed")

    def _process_loop(self):
        """Main processing loop that runs in the background thread."""
        logger.debug("Batch processor loop started")
        print("🔍 Batch processor: Background thread started")

        while self._running:
            try:
                # Process events from queue
                self._process_queue()

                # Retry failed batches
                self._retry_failed_batches()

                # Check if we need to flush based on time or size
                if self._should_flush():
                    print("🔍 Batch processor: Background thread flushing batch")
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
                print(f"🔍 Batch processor: Got event from queue: {type(event)}")

                # Convert TraceData to dictionary
                if isinstance(event, TraceData):
                    event_dict = event.to_dict()
                    print(f"🔍 Batch processor: Converted TraceData to dict: {event_dict.get('agent_name', 'no_agent_name')}")
                else:
                    event_dict = event
                    print(f"🔍 Batch processor: Using event as dict: {event_dict}")

                self._batch.append(event_dict)
                events_processed += 1
                print(f"🔍 Batch processor: Added event to batch, size: {len(self._batch)}")

            except Empty:
                # No more events in queue
                break
            except Exception as e:
                logger.error(f"Error processing queue item: {e}")
                break

        if events_processed > 0:
            logger.debug(f"Processed {events_processed} events, batch size: {len(self._batch)}")
            print(f"🔍 Batch processor: Processed {events_processed} events, batch size: {len(self._batch)}")

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
            print(f"🔍 Batch processor: Should flush - batch is full ({len(self._batch)} >= {self.config.batch_size})")
            return True

        # Flush if enough time has passed
        time_since_last_flush = time.time() - self._last_flush
        if time_since_last_flush >= self.config.flush_interval:
            print(f"🔍 Batch processor: Should flush - time interval ({time_since_last_flush:.1f}s >= {self.config.flush_interval}s)")
            return True

        return False

    def _flush_batch(self):
        """Send the current batch to the API."""
        print(f"🔍 Batch processor: _flush_batch called, batch size: {len(self._batch)}")
        if not self._batch:
            print("🔍 Batch processor: _flush_batch - no batch to flush")
            return

        batch_to_send = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()

        logger.debug(f"Flushing batch with {len(batch_to_send)} events")
        print(f"🔍 Batch processor: Flushing batch with {len(batch_to_send)} events")

        # Send batch in a separate thread to avoid blocking
        send_thread = Thread(
            target=self._send_batch_sync,
            args=(batch_to_send,),
            daemon=True,
            name="CognitionFlow-BatchSender"
        )
        send_thread.start()

    def _send_batch_sync(self, batch: List[Dict[str, Any]]):
        """Send batch synchronously using requests (no event loop required)."""
        if not batch:
            return

        # Normalize events to include required fields (TraceCreate schema)
        normalized: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for ev in batch:
            item = dict(ev)
            # Each span gets its own unique trace_id, but child spans link to parent via parent_trace_id
            if item.get("parent_span_id"):
                # This is a child span - use span_id as trace_id and parent_span_id as parent_trace_id
                item["trace_id"] = item.get("span_id")
                item["parent_trace_id"] = item.get("parent_span_id")
            else:
                # This is a root span - use trace_id as is
                if not item.get("trace_id"):
                    item["trace_id"] = item.get("span_id") or str(uuid.uuid4())
            if not item.get("start_time"):
                item["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # Ensure integer duration per API schema
            if "duration_ms" in item and item["duration_ms"] is not None:
                try:
                    item["duration_ms"] = int(item["duration_ms"])
                except Exception:
                    item["duration_ms"] = None
            # Ensure unique trace_id per event to satisfy backend unique constraint
            tid = item.get("trace_id")
            if tid in seen_ids:
                # Prefer span_id if present, else generate a new UUID
                new_tid = item.get("span_id") or str(uuid.uuid4())
                item["trace_id"] = new_tid
                tid = new_tid
            seen_ids.add(tid)
            status = item.get("status")
            if status not in {"running", "completed", "failed", "cancelled"}:
                item["status"] = "completed"
            normalized.append(item)

        url = f"{self.config.endpoint.rstrip('/')}/api/v1/traces/batch"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CognitionFlow-Python-SDK/0.1.0"
        }
        # Extract agent information from traces
        agents = []
        seen_agents = set()
        
        for trace in normalized:
            agent_name = trace.get("agent_name")
            if agent_name and agent_name not in seen_agents:
                seen_agents.add(agent_name)
                # Create agent data from trace information
                agent_data = {
                    "trace_id": trace.get("trace_id"),
                    "agent_id": trace.get("span_id", trace.get("trace_id")),
                    "name": agent_name,
                    "type": trace.get("metadata", {}).get("agent_type", "generic"),
                    "description": trace.get("metadata", {}).get("description"),
                    "tags": trace.get("tags", {}),
                    "start_time": trace.get("start_time"),
                    "end_time": trace.get("end_time"),
                    "duration_ms": trace.get("duration_ms"),
                    "input_tokens": trace.get("input_tokens", 0),
                    "output_tokens": trace.get("output_tokens", 0),
                    "total_tokens": trace.get("total_tokens", 0),
                    "cost": trace.get("cost", 0.0),
                    "status": trace.get("status", "completed"),
                    "input_data": trace.get("inputs"),
                    "output_data": trace.get("outputs"),
                    "error_message": trace.get("error", {}).get("message") if trace.get("error") else None,
                    "error_type": trace.get("error", {}).get("type") if trace.get("error") else None,
                    "error_stack_trace": trace.get("error", {}).get("traceback") if trace.get("error") else None,
                    "llm_model_name": trace.get("metadata", {}).get("llm_model_name"),
                    "llm_model_provider": trace.get("metadata", {}).get("llm_model_provider"),
                    "llm_model_parameters": trace.get("metadata", {}).get("llm_model_parameters")
                }
                agents.append(agent_data)
        
        payload = {"traces": normalized, "agents": agents, "dependencies": []}

        # Lazy import requests to avoid hard dependency at import time
        try:
            import requests
        except ImportError:
            logger.error("'requests' is required to send batches. Install with: pip install requests")
            return

        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                if resp.status_code < 400:
                    logger.debug(f"Successfully sent batch of {len(batch)} events")
                    print(f"🔍 Batch processor: Successfully sent batch of {len(batch)} events")
                    self._consecutive_failures = 0
                    return
                else:
                    raise Exception(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                self._consecutive_failures += 1
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to send batch after {self.config.max_retries} attempts: {e}")
                    self._failed_batches.append({
                        "batch": batch,
                        "timestamp": time.time(),
                        "attempts": self.config.max_retries
                    })
                else:
                    delay = min(60.0, 2 ** attempt + (time.time() % 1))
                    logger.warning(f"Batch send attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)

    # Removed async variant to avoid event loop conflicts

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