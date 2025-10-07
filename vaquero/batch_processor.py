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

# Framework processors for hierarchical storage
try:
    import sys
    import os
    # Add backend to path if running from SDK
    backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from app.services.framework_processors import trace_processor
except ImportError:
    # Fallback for when running from SDK directly without backend
    trace_processor = None

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batching and sending of trace events.

    This class runs in a background thread and periodically collects
    trace events from a queue, batches them, and sends them to the
    Vaquero API using asynchronous HTTP requests.

    Features:
    - Automatic batching based on size and time intervals
    - Exponential backoff retry logic
    - Graceful degradation on failures
    - Memory-efficient queue management
    """

    def __init__(self, sdk: 'VaqueroSDK'):
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
        self._thread = Thread(target=self._process_loop, daemon=True, name="Vaquero-BatchProcessor")
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

        # Flush any remaining batch synchronously to avoid daemon thread issues
        if self._batch:
            try:
                batch_to_send = self._batch.copy()
                self._batch.clear()
                self._last_flush = time.time()
                logger.debug(f"Flushing final batch with {len(batch_to_send)} events")
                self._send_batch_sync(batch_to_send)
            except Exception as e:
                logger.error(f"Error flushing final batch: {e}")

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        logger.info("Batch processor shutdown completed")

    def _detect_framework(self, span: Dict[str, Any]) -> Optional[str]:
        """Detect the framework from span metadata or name patterns."""
        # Check metadata for framework information
        metadata = span.get("tags", {})
        if isinstance(metadata, dict):
            # Check for LangChain metadata
            if "langchain.metadata" in metadata:
                return "langchain"

            # Check for OpenAI metadata
            if "openai.metadata" in metadata or "openai" in str(metadata).lower():
                return "openai"

            # Check for Anthropic metadata
            if "anthropic.metadata" in metadata or "anthropic" in str(metadata).lower():
                return "anthropic"

        # Check span name patterns
        name = span.get("name", "").lower()
        if "langchain" in name:
            return "langchain"
        elif "openai" in name:
            return "openai"
        elif "anthropic" in name or "claude" in name:
            return "anthropic"

        return None

    def flush(self):
        """Manually flush the current batch."""
        # Ensure we pull any pending events from the queue
        try:
            self._process_queue()
        except Exception as e:
            logger.error(f"Error processing queue before flush: {e}")

        if not self._batch:
            return

        # Perform a synchronous send so the caller waits until delivery
        batch_to_send = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()
        logger.debug(f"Flushing batch with {len(batch_to_send)} events")
        self._send_batch_sync(batch_to_send)
        logger.debug("Manual flush completed")
        # Attempt to drain any per-trace buffered groups that are now ready (e.g., root just queued)
        try:
            self._send_batch_sync([])
        except Exception:
            pass

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
            name="Vaquero-BatchSender"
        )
        send_thread.start()

    def _send_batch_sync(self, batch: List[Dict[str, Any]]):
        """Send batch synchronously using requests (no event loop required).
        This buffers events per trace_id and only sends groups that include a root span.
        """
        if not batch:
            logger.debug("Batch processor: Empty batch, nothing to send")
            return

        logger.info(f"Batch processor: === SENDING BATCH TO API ===")
        logger.info(f"Batch processor: Batch size: {len(batch)} events")

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
                if self.config.debug:
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

            # Store items before popping them in case send fails
            self._current_items_by_trace = {}
            for tid in ready_trace_ids:
                self._current_items_by_trace[tid] = self._pending_by_trace.pop(tid, [])
                if self.config.debug:
                    print(f"🔍 Processing trace {tid} with {len(self._current_items_by_trace[tid])} items")

                # Split into one trace (root) and agents (children)
                # Logic: Root items have no parent_span_id. Everything else is a child.
                # This ensures mutual exclusivity and prevents duplicate trace creation.
                items = self._current_items_by_trace[tid]
                root_items = [it for it in items if not it.get("parent_span_id")]
                child_items = [it for it in items if it.get("parent_span_id")]

                if self.config.debug:
                    print(f"🔍 Trace {tid}: {len(root_items)} root items, {len(child_items)} child items")

                # Debug: Show what items we have
                if self.config.debug:
                    for i, item in enumerate(items):
                        span_id = item.get("span_id")
                        name = item.get("agent_name") or item.get("function_name") or item.get("name")
                        parent_span_id = item.get("parent_span_id")
                        print(f"   Item {i}: span_id={span_id}, name={name}, parent_span_id={parent_span_id}")

                # Debug: Check for duplicates before processing
                span_ids = [item.get("span_id") for item in items]
                duplicates = [x for x in span_ids if span_ids.count(x) > 1]
                if duplicates and self.config.debug:
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
                if len(deduplicated_items) < len(items) and self.config.debug:
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
                # Group root items by trace_id to ensure one trace per trace_id
                # Filter out workflow spans from being treated as traces since they should only be agents
                non_workflow_root_items = [
                    item for item in root_items
                    if item.get("name") not in workflow_names and
                    item.get("agent_name") not in workflow_names
                ]

                # Group by trace_id to avoid duplicates
                traces_by_id = {}
                for item in non_workflow_root_items:
                    trace_id = item.get("trace_id")
                    if trace_id not in traces_by_id:
                        # First time seeing this trace_id - create the trace entry
                        traces_by_id[trace_id] = {
                            "trace_id": trace_id,
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

                # Add all traces to the batch
                for trace_data in traces_by_id.values():
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
                if self.config.debug:
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
                        if self.config.debug:
                            print(f"🔍 DEBUG: Created unique agent_id {agent_id} for duplicate {agent_name} in trace {trace_id}")
                    else:
                        agent_id_counters[trace_id][agent_name] = 0
                        agent_id = span_id

                    # Get metadata from the child span (which should contain source code)
                    child_metadata = child.get("metadata", {})
                    logger.info(f"Batch processor: Child span metadata keys: {list(child_metadata.keys())}")

                    # Check if source code is in the child metadata
                    source_code = child_metadata.get("source_code")
                    if source_code:
                        logger.info(f"Batch processor: CHILD SPAN CONTAINS SOURCE CODE (length: {len(source_code)})")
                    else:
                        logger.warning(f"Batch processor: CHILD SPAN DOES NOT CONTAIN SOURCE CODE")

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
                        "prompt_hash": child.get("prompt_hash"),
                        "raw_data": {
                            # Keep non-source-code metadata for backward compatibility
                            **{k: v for k, v in child_metadata.items() if k not in ['source_code', 'docstring', 'function_signature', 'module_name', 'file_path', 'function_name']},
                            "inputs": child.get("inputs"),
                            "outputs": child.get("outputs")
                        }
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
                            if self.config.debug:
                                print(f"🔗 Created dependency: {parent_agent['name']} → {agent_data['name']}")

                    # Also add remaining root items as agents (for nested root spans)
                    # Skip workflow items as they should only be processed once as root items
                    for item in root_items[1:]:  # Skip first one, it's the trace
                        span_id = item.get("span_id")

                        # Skip if this span_id has already been processed
                        if span_id in processed_span_ids:
                            if self.config.debug:
                                print(f"🔍 DEBUG: Skipping duplicate span_id {span_id} in root_items")
                            continue

                        # Skip workflow items to prevent duplication
                        item_name = item.get("agent_name") or item.get("function_name") or item.get("name")
                        if item_name in workflow_names:
                            if self.config.debug:
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
                            if self.config.debug:
                                print(f"🔍 DEBUG: Created unique agent_id {agent_id} for root item {agent_name} in trace {trace_id}")
                        else:
                            agent_id_counters[trace_id][agent_name] = 0
                            agent_id = span_id

                        # Get metadata from the item (which should contain source code)
                        item_metadata = item.get("metadata", {})
                        logger.info(f"Batch processor: Item metadata keys: {list(item_metadata.keys())}")

                        # Check if source code is in the item metadata
                        source_code = item_metadata.get("source_code")
                        if source_code:
                            logger.info(f"Batch processor: ITEM CONTAINS SOURCE CODE (length: {len(source_code)})")
                        else:
                            logger.warning(f"Batch processor: ITEM DOES NOT CONTAIN SOURCE CODE")

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
                            "prompt_hash": item.get("prompt_hash"),
                            "raw_data": {
                                # Keep non-source-code metadata for backward compatibility
                                **{k: v for k, v in item_metadata.items() if k not in ['source_code', 'docstring', 'function_signature', 'module_name', 'file_path', 'function_name']},
                                "inputs": item.get("inputs"),
                                "outputs": item.get("outputs")
                            }
                        }
                        agents.append(agent_data)

                        # Build agent ID mapping for dependency creation
                        all_agent_id_map[agent_id] = agent_data  # Use the unique agent_id

                # Debug: print what we found for this trace group
                workflow_roots = [it for it in root_items if it.get("name") == "linear_workflow" or it.get("agent_name") == "linear_workflow"]
                if self.config.debug:
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

                # Get metadata from the item (which should contain source code)
                item_metadata = item.get("metadata", {})

                # Also collect metadata from child spans (which contain source_code)
                # This ensures source_code from individual agents is included in the trace
                for child in child_items:
                    child_metadata = child.get("metadata", {})
                    if child_metadata and "source_code" in child_metadata:
                        # Merge child metadata into item metadata to include source_code
                        item_metadata.update(child_metadata)
                        logger.info(f"Batch processor: Merged source_code from child span into trace metadata")
                        break  # Only need one source_code, they should be the same for all agents in a workflow

                logger.info(f"Batch processor: Trace metadata keys: {list(item_metadata.keys())}")

                # Check if source code is in the trace metadata
                source_code = item_metadata.get("source_code")
                if source_code:
                    logger.info(f"Batch processor: TRACE CONTAINS SOURCE CODE (length: {len(source_code)})")
                else:
                    logger.warning(f"Batch processor: TRACE DOES NOT CONTAIN SOURCE CODE")

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
                        "metadata": item_metadata,  # Include the full metadata with source code
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
                # Check if source_code is in the trace_data before sending
                if trace_data.get("raw_data") and trace_data["raw_data"].get("metadata"):
                    metadata = trace_data["raw_data"]["metadata"]
                    if "source_code" in metadata:
                        logger.info(f"Batch processor: TRACE DATA CONTAINS SOURCE CODE (length: {len(metadata['source_code'])})")
                    else:
                        logger.warning(f"Batch processor: TRACE DATA DOES NOT CONTAIN SOURCE CODE")

                trace_data = {k: v for k, v in trace_data.items() if v is not None}
                traces.append(trace_data)

                for child in child_items:
                    # Get metadata from the child span (which should contain source code)
                    child_metadata = child.get("metadata", {})
                    logger.info(f"Batch processor: Child span metadata keys: {list(child_metadata.keys())}")

                    # Check if source code is in the child metadata
                    source_code = child_metadata.get("source_code")
                    if source_code:
                        logger.info(f"Batch processor: CHILD SPAN CONTAINS SOURCE CODE (length: {len(source_code)})")
                    else:
                        logger.warning(f"Batch processor: CHILD SPAN DOES NOT CONTAIN SOURCE CODE")

                    # Apply framework-specific processing for hierarchical storage
                    processed_child = child
                    if trace_processor:
                        # Detect framework from span metadata or name patterns
                        framework = self._detect_framework(child)
                        if framework:
                            try:
                                # Apply framework-specific processing
                                # Note: This is a simplified approach - in production we'd group by trace first
                                logger.debug(f"🔍 Processing child span with framework: {framework}")
                            except Exception as e:
                                logger.warning(f"Error processing framework {framework}: {e}")
                                # Continue with original child data

                    agent_data = {
                        "trace_id": processed_child.get("trace_id"),
                        "agent_id": processed_child.get("span_id"),
                        "name": processed_child.get("agent_name") or processed_child.get("function_name") or processed_child.get("name"),
                        "type": processed_child.get("agent_name"),
                        "description": processed_child.get("description"),
                        "tags": processed_child.get("tags", {}),
                        "start_time": processed_child.get("start_time"),
                        "end_time": processed_child.get("end_time"),
                        "duration_ms": processed_child.get("duration_ms"),
                        "input_tokens": processed_child.get("input_tokens", 0),
                        "output_tokens": processed_child.get("output_tokens", 0),
                        "total_tokens": processed_child.get("total_tokens", 0),
                        "cost": processed_child.get("cost", 0.0),
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
                        "prompt_hash": child.get("prompt_hash"),
                        "raw_data": {
                            **child_metadata,  # Flatten metadata so source_code is directly accessible
                            "inputs": child.get("inputs"),
                            "outputs": child.get("outputs")
                        }
                    }
                    agents.append(agent_data)

        # If nothing ready to send, return
        if not traces and not agents:
            return

        url = f"{self.config.endpoint.rstrip('/')}/api/v1/traces/batch"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Vaquero-Python-SDK/0.1.0"
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
                if self.config.debug:
                    print("✅ Agent ID uniqueness check passed - no duplicates found")

            # Log agent ID distribution summary
            if self.config.debug:
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
            if self.config.debug:
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
                    if self.config.debug:
                        print(f"🔍 Batch processor: Successfully sent batch of {len(batch)} events")
                    self._consecutive_failures = 0
                    return
                else:
                    raise Exception(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                self._consecutive_failures += 1
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to send batch after {self.config.max_retries} attempts: {e}")
                    # Put items back into pending queue for retry
                    for tid, items in self._current_items_by_trace.items():
                        if items:  # Only put back if there are items
                            self._pending_by_trace[tid] = items
                            logger.info(f"Put {len(items)} items back into pending queue for trace {tid}")
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
                name="Vaquero-BatchRetry"
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