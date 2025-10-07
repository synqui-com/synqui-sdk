"""
Simple trace collector for in-memory trace storage.

This replaces the complex batch processor with a simple approach:
1. Collect all spans in memory during trace execution
2. Send complete trace when context exits
3. No complex retry logic or async queues
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TraceCollector:
    """Simple in-memory trace collector."""
    
    def __init__(self, sdk: 'VaqueroSDK'):
        """Initialize the trace collector.
        
        Args:
            sdk: Reference to the main SDK instance
        """
        self.sdk = sdk
        self.config = sdk.config
        self._traces: Dict[str, Dict[str, Any]] = {}  # trace_id -> trace data
        self._spans: Dict[str, List[Dict[str, Any]]] = {}  # trace_id -> list of spans
        self._current_trace_id: Optional[str] = None
        self._trace_stack: List[str] = []  # Stack of trace IDs for nested traces
    
    def start_trace(self, trace_id: str, span_data: Dict[str, Any]) -> None:
        """Start a new trace.
        
        Args:
            trace_id: Unique identifier for the trace
            span_data: Initial span data for the trace
        """
        self._current_trace_id = trace_id
        self._trace_stack.append(trace_id)
        
        # Store trace data
        self._traces[trace_id] = {
            "trace_id": trace_id,
            "parent_trace_id": span_data.get("parent_trace_id"),
            "session_id": span_data.get("session_id"),
            "name": span_data.get("name"),
            "description": span_data.get("description"),
            "tags": span_data.get("tags", {}),
            "status": span_data.get("status", "running"),
            "start_time": span_data.get("start_time"),
            "end_time": None,
            "duration_ms": None,
            "total_tokens": 0,
            "total_cost": 0.0,
            "error_count": 0,
            "git_commit_sha": span_data.get("git_commit_sha"),
            "git_branch": span_data.get("git_branch"),
            "git_repository": span_data.get("git_repository"),
            "environment": span_data.get("environment"),
            "hostname": span_data.get("hostname"),
            "sdk_version": span_data.get("sdk_version"),
            "raw_data": {
                "inputs": span_data.get("inputs"),
                "outputs": span_data.get("outputs"),
                "error": span_data.get("error"),
                "metadata": span_data.get("metadata", {}),
                "attributes": span_data.get("attributes"),
            }
        }
        
        # Initialize spans list
        self._spans[trace_id] = []
        
        logger.debug(f"Started trace {trace_id}")
    
    def add_span(self, span_data: Dict[str, Any]) -> None:
        """Add a span to the current trace.
        
        Args:
            span_data: Span data to add
        """
        if not self._current_trace_id:
            logger.warning("No active trace to add span to")
            return
        
        trace_id = self._current_trace_id
        
        # Add span to the trace
        self._spans[trace_id].append(span_data)
        
        # Update trace metrics
        self._update_trace_metrics(trace_id, span_data)
        
        logger.debug(f"Added span to trace {trace_id}: {span_data.get('name', 'unnamed')}")
    
    def end_trace(self, trace_id: str, end_data: Dict[str, Any]) -> None:
        """End a trace and send it to the API.
        
        Args:
            trace_id: Trace ID to end
            end_data: Final trace data
        """
        if trace_id not in self._traces:
            logger.warning(f"Trace {trace_id} not found")
            return
        
        # Update trace with end data
        trace = self._traces[trace_id]
        trace.update({
            "status": end_data.get("status", "completed"),
            "end_time": end_data.get("end_time"),
            "duration_ms": int(round(end_data.get("duration_ms", 0))),
            "error_count": end_data.get("error_count", 0)
        })
        
        # Remove from stack
        if trace_id in self._trace_stack:
            self._trace_stack.remove(trace_id)
        
        # Update current trace
        if self._trace_stack:
            self._current_trace_id = self._trace_stack[-1]
        else:
            self._current_trace_id = None
        
        # Send complete trace to API
        self._send_trace(trace_id)
        
        logger.debug(f"Ended trace {trace_id}")
    
    def _update_trace_metrics(self, trace_id: str, span_data: Dict[str, Any]) -> None:
        """Update trace metrics with span data.
        
        Args:
            trace_id: Trace ID to update
            span_data: Span data to aggregate
        """
        trace = self._traces[trace_id]
        
        # Aggregate tokens and cost
        trace["total_tokens"] += span_data.get("total_tokens", 0)
        trace["total_cost"] += span_data.get("cost", 0.0)
        
        # Count errors
        if span_data.get("error"):
            trace["error_count"] += 1
    
    def _send_trace(self, trace_id: str) -> None:
        """Send complete trace to the API.
        
        Args:
            trace_id: Trace ID to send
        """
        if trace_id not in self._traces:
            return
        
        trace = self._traces[trace_id]
        spans = self._spans.get(trace_id, [])
        
        # Convert spans to agents format
        agents = []
        dependencies = []
        
        for span in spans:
            agent_data = {
                "trace_id": trace_id,
                "agent_id": span.get("span_id"),
                "name": span.get("agent_name") or span.get("function_name") or span.get("name"),
                "type": span.get("agent_name"),
                "description": span.get("description"),
                "tags": span.get("tags", {}),
                "start_time": span.get("start_time"),
                "end_time": span.get("end_time"),
                "duration_ms": int(round(span.get("duration_ms", 0))),
                "input_tokens": span.get("input_tokens", 0),
                "output_tokens": span.get("output_tokens", 0),
                "total_tokens": span.get("total_tokens", 0),
                "cost": span.get("cost", 0.0),
                "status": span.get("status", "completed"),
                "input_data": span.get("inputs"),
                "output_data": span.get("outputs"),
                "error_message": span.get("error", {}).get("message") if span.get("error") else None,
                "error_type": span.get("error", {}).get("type") if span.get("error") else None,
                "error_stack_trace": span.get("error", {}).get("traceback") if span.get("error") else None,
                "llm_model_name": span.get("model_name"),
                "llm_model_provider": span.get("model_provider"),
                "llm_model_parameters": span.get("model_parameters"),
                "system_prompt": span.get("system_prompt"),
                "prompt_name": span.get("prompt_name"),
                "prompt_version": span.get("prompt_version"),
                "prompt_parameters": span.get("prompt_parameters"),
                "prompt_hash": span.get("prompt_hash"),
                "raw_data": {
                    "inputs": span.get("inputs"),
                    "outputs": span.get("outputs"),
                    "metadata": span.get("metadata", {}),
                }
            }
            agents.append(agent_data)
            
            # Create dependency if this span has a parent
            parent_span_id = span.get("parent_span_id")
            if parent_span_id:
                dependency = {
                    "trace_id": trace_id,
                    "parent_agent_id": parent_span_id,
                    "child_agent_id": span.get("span_id"),
                    "dependency_type": "calls"
                }
                dependencies.append(dependency)
        
        # Prepare payload
        payload = {
            "traces": [trace],
            "agents": agents,
            "dependencies": dependencies
        }
        
        # Send to API
        self._send_to_api(payload)
        
        # Clean up
        del self._traces[trace_id]
        del self._spans[trace_id]
    
    def _send_to_api(self, payload: Dict[str, Any]) -> None:
        """Send payload to the API.
        
        Args:
            payload: Data to send
        """
        url = f"{self.config.endpoint.rstrip('/')}/api/v1/traces/batch"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Vaquero-Python-SDK/0.1.0"
        }
        
        try:
            import requests
        except ImportError:
            logger.error("'requests' is required to send traces. Install with: pip install requests")
            return
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
            if response.status_code < 400:
                logger.debug(f"Successfully sent trace with {len(payload.get('agents', []))} agents")
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send trace: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics.
        
        Returns:
            Dictionary containing collector statistics
        """
        return {
            "active_traces": len(self._traces),
            "current_trace_id": self._current_trace_id,
            "trace_stack_depth": len(self._trace_stack),
        }
