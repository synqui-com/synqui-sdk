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
        # Buffer events per trace_id until a root span is available
        self._pending_by_trace: Dict[str, List[Dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()

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
        # Attempt to drain any per-trace buffered groups that are now ready (e.g., root just queued)
        try:
            self._send_batch_sync([])
        except Exception:
            pass

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
        """Send batch synchronously using requests (no event loop required).
        This buffers events per trace_id and only sends groups that include a root span.
        """
        if not batch:
            return

        # Normalize events to include required fields (TraceCreate schema)
        normalized: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for ev in batch:
            item = dict(ev)
            if not item.get("trace_id"):
                item["trace_id"] = item.get("span_id") or str(uuid.uuid4())
            # Ensure parent_trace_id rules
            if item.get("parent_span_id"):
                item["parent_trace_id"] = None
            else:
                if "parent_trace_id" not in item:
                    item["parent_trace_id"] = None
            if not item.get("start_time"):
                from datetime import datetime
                item["start_time"] = datetime.utcnow().isoformat() + "Z"
            if "duration_ms" in item and item["duration_ms"] is not None:
                try:
                    item["duration_ms"] = int(item["duration_ms"])
                except Exception:
                    item["duration_ms"] = None
            # Ensure only one unique root per trace id in this normalization pass
            if not item.get("parent_span_id"):
                tid = item.get("trace_id")
                if tid in seen_ids:
                    # If we somehow see multiple roots for same trace id, keep first, demote others to agents by faking a parent
                    item["parent_span_id"] = item.get("parent_span_id") or "__root_duplicate__"
                else:
                    seen_ids.add(tid)
            status = item.get("status")
            if status not in {"running", "completed", "failed", "cancelled"}:
                item["status"] = "completed"
            normalized.append(item)

        # Buffer by trace_id
        with self._pending_lock:
            for item in normalized:
                tid = item.get("trace_id")
                if not tid:
                    continue

                # Debug logging for workflow spans
                if item.get("name") == "linear_workflow" or item.get("agent_name") == "linear_workflow":
                    print(f"🔍 Workflow span: trace_id={tid}, parent_span_id={item.get('parent_span_id')}, span_id={item.get('span_id')}")
                    print(f"🔍 Workflow span details: {item}")
                elif item.get("name"):
                    print(f"🔍 Regular span: {item.get('name')}")
                elif item.get("agent_name"):
                    print(f"🔍 Agent span: {item.get('agent_name')}")

                self._pending_by_trace.setdefault(tid, []).append(item)

            # Build sendable groups: those with a root span present
            ready_trace_ids = []
            for tid, items in self._pending_by_trace.items():
                # Check if we have a root span (no parent_span_id) OR a main workflow span
                has_root_span = any(not it.get("parent_span_id") for it in items)
                has_workflow_span = any(
                    it.get("name") in ["linear_workflow", "main_workflow", "workflow"] or
                    it.get("agent_name") in ["linear_workflow", "main_workflow", "workflow"]
                    for it in items
                )

                if has_root_span or has_workflow_span:
                    ready_trace_ids.append(tid)

            # If nothing is ready, return and keep buffering
            if not ready_trace_ids:
                logger.debug("No ready trace groups with root span present; buffering")
                return

            # Collect payload parts from all ready groups
            traces: List[Dict[str, Any]] = []
            agents: List[Dict[str, Any]] = []
            dependencies: List[Dict[str, Any]] = []

            # Build a comprehensive agent_id_map across all ready trace groups
            all_agent_id_map = {}  # span_id -> agent_data mapping for dependency creation

            # Track agent IDs within each trace to ensure uniqueness
            agent_id_counters = {}  # trace_id -> {agent_name -> counter}

            # Track which span_ids have already been processed to avoid duplicates
            processed_span_ids = set()  # Global set to track all processed span_ids

            for tid in ready_trace_ids:
                items = self._pending_by_trace.pop(tid, [])
                print(f"🔍 Processing trace {tid} with {len(items)} items")

                # Split into one trace (root) and agents (children)
                # Root can be either: no parent_span_id OR a main workflow span
                # IMPORTANT: Make this logic mutually exclusive to avoid duplicates
                root_items = [
                    it for it in items
                    if not it.get("parent_span_id") or
                    it.get("name") in ["linear_workflow", "main_workflow", "workflow"] or
                    it.get("agent_name") in ["linear_workflow", "main_workflow", "workflow"]
                ]
                child_items = [
                    it for it in items
                    if it.get("parent_span_id") and not (
                        it.get("name") in ["linear_workflow", "main_workflow", "workflow"] or
                        it.get("agent_name") in ["linear_workflow", "main_workflow", "workflow"]
                    )
                ]

                print(f"🔍 Trace {tid}: {len(root_items)} root items, {len(child_items)} child items")

                # Debug: Show what items we have
                for i, item in enumerate(items):
                    span_id = item.get("span_id")
                    name = item.get("agent_name") or item.get("function_name") or item.get("name")
                    parent_span_id = item.get("parent_span_id")
                    print(f"   Item {i}: span_id={span_id}, name={name}, parent_span_id={parent_span_id}")

                # Debug: Check for duplicates before processing
                span_ids = [item.get("span_id") for item in items]
                duplicates = [x for x in span_ids if span_ids.count(x) > 1]
                if duplicates:
                    print(f"🔍 DUPLICATES FOUND in items for trace {tid}: {set(duplicates)}")

                # First, process child items and mark their span_ids as processed
                for child in child_items:
                    span_id = child.get("span_id")
                    if span_id not in processed_span_ids:
                        processed_span_ids.add(span_id)
                        print(f"🔍 Processing child item: span_id={span_id}")
                    else:
                        print(f"🔍 SKIPPING child item (already processed): span_id={span_id}")

                # Then, process root items and mark their span_ids as processed
                for item in root_items[1:]:  # Skip first one, it's the trace
                    span_id = item.get("span_id")
                    if span_id not in processed_span_ids:
                        processed_span_ids.add(span_id)
                        print(f"🔍 Processing root item: span_id={span_id}")
                    else:
                        print(f"🔍 SKIPPING root item (already processed): span_id={span_id}")

                # Check for duplicates and remove them
                seen_span_ids = set()
                deduplicated_items = []
                for item in items:
                    span_id = item.get("span_id")
                    if span_id not in seen_span_ids:
                        seen_span_ids.add(span_id)
                        deduplicated_items.append(item)

                # Log if duplicates were found
                if len(deduplicated_items) < len(items):
                    print(f"🔍 DEBUG: Removed {len(items) - len(deduplicated_items)} duplicate spans for trace_id {tid}")
                    for item in items:
                        if item.get("span_id") in seen_span_ids:
                            seen_span_ids.remove(item.get("span_id"))
                        else:
                            print(f"    - DUPLICATE: {item.get('name') or item.get('agent_name')} (span_id: {item.get('span_id')})")

                # Use deduplicated items
                items = deduplicated_items

                # Re-split after deduplication
                # Workflow spans should ONLY be treated as roots, never as agents
                workflow_names = ["linear_workflow", "main_workflow", "workflow"]
                root_items = [
                    it for it in items
                    if not it.get("parent_span_id") or
                    it.get("name") in workflow_names or
                    it.get("agent_name") in workflow_names
                ]
                child_items = [
                    it for it in items
                    if it.get("parent_span_id") and not (
                        it.get("name") in workflow_names or
                        it.get("agent_name") in workflow_names
                    )
                ]

                # Process root items (traces)
                # Filter out workflow spans from being treated as traces since they should only be agents
                non_workflow_root_items = [
                    item for item in root_items
                    if item.get("name") not in workflow_names and
                    item.get("agent_name") not in workflow_names
                ]

                if non_workflow_root_items:
                    # Use the first root item as the trace
                    item = non_workflow_root_items[0]
                    trace_data = {
                        "trace_id": item.get("trace_id"),
                        "parent_trace_id": item.get("parent_trace_id"),
                        "session_id": item.get("session_id"),
                        "name": item.get("name"),
                        "description": item.get("description"),
                        "tags": item.get("tags", {}),
                        "status": item.get("status", "completed"),
                        "start_time": item.get("start_time"),
                        "end_time": item.get("end_time"),
                        "duration_ms": item.get("duration_ms"),
                        "total_tokens": item.get("total_tokens", 0),
                        "total_cost": item.get("cost", 0.0),
                        "error_count": 1 if item.get("error") else 0,
                        "git_commit_sha": item.get("git_commit_sha"),
                        "git_branch": item.get("git_branch"),
                        "git_repository": item.get("git_repository"),
                        "environment": item.get("environment"),
                        "hostname": item.get("hostname"),
                        "sdk_version": item.get("sdk_version"),
                        "raw_data": {
                            "inputs": item.get("inputs"),
                            "outputs": item.get("outputs"),
                            "error": item.get("error"),
                            "metadata": item.get("metadata"),
                            "attributes": item.get("attributes"),
                            "input_tokens": item.get("input_tokens"),
                            "output_tokens": item.get("output_tokens"),
                            "model_name": item.get("model_name"),
                            "model_provider": item.get("model_provider"),
                            "system_prompt": item.get("system_prompt"),
                            "prompt_name": item.get("prompt_name"),
                            "prompt_version": item.get("prompt_version"),
                            "prompt_parameters": item.get("prompt_parameters"),
                            "prompt_hash": item.get("prompt_hash"),
                        }
                    }
                    trace_data = {k: v for k, v in trace_data.items() if v is not None}
                    traces.append(trace_data)

                # Process workflow items as agents if they exist
                # Note: Don't add workflow items to child_items to avoid duplication
                # The workflow span will be handled separately as a root item
                workflow_items = [
                    item for item in root_items
                    if item.get("name") in workflow_names or
                    item.get("agent_name") in workflow_names
                ]
                # Don't extend child_items with workflow items to prevent duplication
                print(f"🔍 Found {len(workflow_items)} workflow items (will be processed as root items only)")

                # Create agents from child items with unique agent IDs
                for child in child_items:
                    span_id = child.get("span_id")

                    # Skip if this span_id has already been processed (prevents duplicates)
                    if span_id in processed_span_ids:
                        print(f"🔍 DEBUG: Skipping duplicate span_id {span_id} in child_items")
                        continue

                    processed_span_ids.add(span_id)
                    trace_id = child.get("trace_id")
                    agent_name = child.get("agent_name") or child.get("function_name") or child.get("name")

                    # Initialize counter for this trace if not exists
                    if trace_id not in agent_id_counters:
                        agent_id_counters[trace_id] = {}

                    # Ensure unique agent ID within this trace
                    if agent_name in agent_id_counters[trace_id]:
                        agent_id_counters[trace_id][agent_name] += 1
                        # Create unique agent ID by appending counter
                        agent_id = f"{span_id}_{agent_id_counters[trace_id][agent_name]}"
                        print(f"🔍 DEBUG: Created unique agent_id {agent_id} for duplicate {agent_name} in trace {trace_id}")
                    else:
                        agent_id_counters[trace_id][agent_name] = 0
                        agent_id = span_id

                    agent_data = {
                        "trace_id": trace_id,
                        "agent_id": agent_id,  # Use unique agent_id
                        "name": agent_name,
                        "type": child.get("agent_name"),
                        "description": child.get("description"),
                        "tags": child.get("tags", {}),
                        "start_time": child.get("start_time"),
                        "end_time": child.get("end_time"),
                        "duration_ms": child.get("duration_ms"),
                        "input_tokens": child.get("input_tokens", 0),
                        "output_tokens": child.get("output_tokens", 0),
                        "total_tokens": child.get("total_tokens", 0),
                        "cost": child.get("cost", 0.0),
                        "status": child.get("status", "completed"),
                        "input_data": child.get("inputs"),
                        "output_data": child.get("outputs"),
                        "error_message": child.get("error", {}).get("message") if child.get("error") else None,
                        "error_type": child.get("error", {}).get("type") if child.get("error") else None,
                        "error_stack_trace": child.get("error", {}).get("traceback") if child.get("error") else None,
                        "llm_model_name": child.get("model_name"),
                        "llm_model_provider": child.get("model_provider"),
                        "llm_model_parameters": child.get("model_parameters"),
                        "system_prompt": child.get("system_prompt"),
                        "prompt_name": child.get("prompt_name"),
                        "prompt_version": child.get("prompt_version"),
                        "prompt_parameters": child.get("prompt_parameters"),
                        "prompt_hash": child.get("prompt_hash")
                    }
                    agents.append(agent_data)

                    # Build agent ID mapping for dependency creation
                    all_agent_id_map[agent_id] = agent_data  # Use the unique agent_id

                    # Create dependency if this child has a parent that's also an agent
                    parent_span_id = child.get("parent_span_id")
                    if parent_span_id and parent_span_id in all_agent_id_map:
                        # Find the parent agent data
                        parent_agent = None
                        for agent in agents:
                            if agent["agent_id"] == parent_span_id:
                                parent_agent = agent
                                break

                        if parent_agent:
                            dependency = {
                                "trace_id": tid,
                                "parent_agent_id": parent_span_id,  # Use span_id directly for parent
                                "child_agent_id": agent_id,        # Use unique agent_id for child
                                "dependency_type": "calls"
                            }
                            dependencies.append(dependency)
                            print(f"🔗 Created dependency: {parent_agent['name']} → {agent_data['name']}")

                    # Also add remaining root items as agents (for nested root spans)
                    # Skip workflow items as they should only be processed once as root items
                    for item in root_items[1:]:  # Skip first one, it's the trace
                        span_id = item.get("span_id")

                        # Skip if this span_id has already been processed
                        if span_id in processed_span_ids:
                            print(f"🔍 DEBUG: Skipping duplicate span_id {span_id} in root_items")
                            continue

                        # Skip workflow items to prevent duplication
                        item_name = item.get("agent_name") or item.get("function_name") or item.get("name")
                        if item_name in workflow_names:
                            print(f"🔍 DEBUG: Skipping workflow item {span_id} in root_items (already handled)")
                            continue

                        processed_span_ids.add(span_id)
                        trace_id = item.get("trace_id")
                        agent_name = item_name

                        # Ensure unique agent ID for root items too
                        if trace_id not in agent_id_counters:
                            agent_id_counters[trace_id] = {}

                        if agent_name in agent_id_counters[trace_id]:
                            agent_id_counters[trace_id][agent_name] += 1
                            agent_id = f"{span_id}_{agent_id_counters[trace_id][agent_name]}"
                            print(f"🔍 DEBUG: Created unique agent_id {agent_id} for root item {agent_name} in trace {trace_id}")
                        else:
                            agent_id_counters[trace_id][agent_name] = 0
                            agent_id = span_id

                        agent_data = {
                            "trace_id": trace_id,
                            "agent_id": agent_id,  # Use unique agent_id
                            "name": agent_name,
                            "type": item.get("agent_name"),
                            "description": item.get("description"),
                            "tags": item.get("tags", {}),
                            "start_time": item.get("start_time"),
                            "end_time": item.get("end_time"),
                            "duration_ms": item.get("duration_ms"),
                            "input_tokens": item.get("input_tokens", 0),
                            "output_tokens": item.get("output_tokens", 0),
                            "total_tokens": item.get("total_tokens", 0),
                            "cost": item.get("cost", 0.0),
                            "status": item.get("status", "completed"),
                            "input_data": item.get("inputs"),
                            "output_data": item.get("outputs"),
                            "error_message": item.get("error", {}).get("message") if item.get("error") else None,
                            "error_type": item.get("error", {}).get("type") if item.get("error") else None,
                            "error_stack_trace": item.get("error", {}).get("traceback") if item.get("error") else None,
                            "llm_model_name": item.get("model_name"),
                            "llm_model_provider": item.get("model_provider"),
                            "llm_model_parameters": item.get("model_parameters"),
                            "system_prompt": item.get("system_prompt"),
                            "prompt_name": item.get("prompt_name"),
                            "prompt_version": item.get("prompt_version"),
                            "prompt_parameters": item.get("prompt_parameters"),
                            "prompt_hash": item.get("prompt_hash")
                        }
                        agents.append(agent_data)

                        # Build agent ID mapping for dependency creation
                        all_agent_id_map[agent_id] = agent_data  # Use the unique agent_id

                # Debug: print what we found for this trace group
                workflow_roots = [it for it in root_items if it.get("name") == "linear_workflow" or it.get("agent_name") == "linear_workflow"]
                if workflow_roots:
                    print(f"🔍 Found workflow root: {workflow_roots[0].get('name')} with parent_span_id={workflow_roots[0].get('parent_span_id')}")
                else:
                    print(f"🔍 No workflow root found in trace group {tid}, using first root item")

                if not root_items:
                    # Should not happen due to ready selection, but guard
                    self._pending_by_trace[tid] = items
                    continue

                # Use the first root item as the trace
                item = root_items[0]
                trace_data = {
                    "trace_id": item.get("trace_id"),
                    "parent_trace_id": item.get("parent_trace_id"),
                    "session_id": item.get("session_id"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "tags": item.get("tags", {}),
                    "status": item.get("status", "completed"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                    "duration_ms": item.get("duration_ms"),
                    "total_tokens": item.get("total_tokens", 0),
                    "total_cost": item.get("cost", 0.0),
                    "error_count": 1 if item.get("error") else 0,
                    "git_commit_sha": item.get("git_commit_sha"),
                    "git_branch": item.get("git_branch"),
                    "git_repository": item.get("git_repository"),
                    "environment": item.get("environment"),
                    "hostname": item.get("hostname"),
                    "sdk_version": item.get("sdk_version"),
                    "raw_data": {
                        "inputs": item.get("inputs"),
                        "outputs": item.get("outputs"),
                        "error": item.get("error"),
                        "metadata": item.get("metadata"),
                        "attributes": item.get("attributes"),
                        "input_tokens": item.get("input_tokens"),
                        "output_tokens": item.get("output_tokens"),
                        "model_name": item.get("model_name"),
                        "model_provider": item.get("model_provider"),
                        "system_prompt": item.get("system_prompt"),
                        "prompt_name": item.get("prompt_name"),
                        "prompt_version": item.get("prompt_version"),
                        "prompt_parameters": item.get("prompt_parameters"),
                        "prompt_hash": item.get("prompt_hash"),
                    }
                }
                trace_data = {k: v for k, v in trace_data.items() if v is not None}
                traces.append(trace_data)

                for child in child_items:
                    agent_data = {
                        "trace_id": child.get("trace_id"),
                        "agent_id": child.get("span_id"),
                        "name": child.get("agent_name") or child.get("function_name") or child.get("name"),
                        "type": child.get("agent_name"),
                        "description": child.get("description"),
                        "tags": child.get("tags", {}),
                        "start_time": child.get("start_time"),
                        "end_time": child.get("end_time"),
                        "duration_ms": child.get("duration_ms"),
                        "input_tokens": child.get("input_tokens", 0),
                        "output_tokens": child.get("output_tokens", 0),
                        "total_tokens": child.get("total_tokens", 0),
                        "cost": child.get("cost", 0.0),
                        "status": child.get("status", "completed"),
                        "input_data": child.get("inputs"),
                        "output_data": child.get("outputs"),
                        "error_message": child.get("error", {}).get("message") if child.get("error") else None,
                        "error_type": child.get("error", {}).get("type") if child.get("error") else None,
                        "error_stack_trace": child.get("error", {}).get("traceback") if child.get("error") else None,
                        "llm_model_name": child.get("model_name"),
                        "llm_model_provider": child.get("model_provider"),
                        "llm_model_parameters": child.get("model_parameters"),
                        "system_prompt": child.get("system_prompt"),
                        "prompt_name": child.get("prompt_name"),
                        "prompt_version": child.get("prompt_version"),
                        "prompt_parameters": child.get("prompt_parameters"),
                        "prompt_hash": child.get("prompt_hash")
                    }
                    agents.append(agent_data)

        # If nothing ready to send, return
        if not traces and not agents:
            return

        url = f"{self.config.endpoint.rstrip('/')}/api/v1/traces/batch"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CognitionFlow-Python-SDK/0.1.0"
        }

        # Convert datetime strings to datetime objects for backend compatibility
        from datetime import datetime as dt
        for trace in traces:
            if isinstance(trace.get("start_time"), str):
                try:
                    trace["start_time"] = dt.fromisoformat(trace["start_time"].replace("Z", "+00:00"))
                except ValueError:
                    pass  # Keep as string if parsing fails
            if isinstance(trace.get("end_time"), str):
                try:
                    trace["end_time"] = dt.fromisoformat(trace["end_time"].replace("Z", "+00:00"))
                except ValueError:
                    pass

        for agent in agents:
            if isinstance(agent.get("start_time"), str):
                try:
                    agent["start_time"] = dt.fromisoformat(agent["start_time"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            if isinstance(agent.get("end_time"), str):
                try:
                    agent["end_time"] = dt.fromisoformat(agent["end_time"].replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Convert datetime objects back to ISO strings for JSON serialization
        from datetime import datetime as dt_obj
        for trace in traces:
            if isinstance(trace.get("start_time"), dt_obj):
                trace["start_time"] = trace["start_time"].isoformat() + "Z"
            if isinstance(trace.get("end_time"), dt_obj):
                trace["end_time"] = trace["end_time"].isoformat() + "Z"

        for agent in agents:
            if isinstance(agent.get("start_time"), dt_obj):
                agent["start_time"] = agent["start_time"].isoformat() + "Z"
            if isinstance(agent.get("end_time"), dt_obj):
                agent["end_time"] = agent["end_time"].isoformat() + "Z"

        payload = {"traces": traces, "agents": agents, "dependencies": dependencies}

        # Debug: Log the actual payload being sent
        logger.debug(f"🔍 Batch processor: Sending payload with {len(traces)} traces, {len(agents)} agents, and {len(dependencies)} dependencies")

        # Log agent ID uniqueness check
        if agents:
            trace_agent_counts = {}
            for agent in agents:
                trace_id = agent["trace_id"]
                agent_id = agent["agent_id"]
                if trace_id not in trace_agent_counts:
                    trace_agent_counts[trace_id] = {}
                trace_agent_counts[trace_id][agent_id] = trace_agent_counts[trace_id].get(agent_id, 0) + 1

            # Check for duplicates (should be none with our fix)
            duplicates_found = []
            for trace_id, agent_counts in trace_agent_counts.items():
                for agent_id, count in agent_counts.items():
                    if count > 1:
                        duplicates_found.append(f"trace {trace_id}: agent {agent_id} appears {count} times")

            if duplicates_found:
                logger.warning(f"🔍 DUPLICATE AGENT IDs FOUND: {duplicates_found}")
                print(f"🔍 WARNING: Duplicate agent IDs found: {duplicates_found}")
            else:
                logger.debug("✅ All agent IDs are unique within traces")
                print("✅ Agent ID uniqueness check passed - no duplicates found")

            # Log agent ID distribution summary
            print(f"🔍 AGENT ID SUMMARY:")
            for trace_id, agent_counts in trace_agent_counts.items():
                print(f"   Trace {trace_id}: {len(agent_counts)} unique agents")
                for agent_id, count in agent_counts.items():
                    print(f"     - {agent_id}: {count}")
            print(f"🔍 Total agents: {len(agents)}")

        if traces:
            logger.debug(f"🔍 First trace keys: {list(traces[0].keys())}")
            logger.debug(f"🔍 First trace raw_data keys: {list(traces[0].get('raw_data', {}).keys()) if traces[0].get('raw_data') else 'No raw_data'}")
        if agents:
            logger.debug(f"🔍 First agent keys: {list(agents[0].keys())}")
            # Print the payload to stderr for debugging
            import sys
            print(f"🔍 DEBUG: Payload being sent: {payload}", file=sys.stderr)
        logger.debug(f"🔍 Batch processor: First trace keys: {list(traces[0].keys()) if traces else 'No traces'}")
        if traces:
            logger.debug(f"🔍 Batch processor: First trace start_time: {traces[0].get('start_time')}")
        if agents:
            logger.debug(f"🔍 Batch processor: First agent keys: {list(agents[0].keys())}")

        try:
            import requests
        except ImportError:
            logger.error("'requests' is required to send batches. Install with: pip install requests")
            return

        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                if resp.status_code < 400:
                    logger.debug(f"Successfully sent grouped payload with {len(traces)} traces and {len(agents)} agents")
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