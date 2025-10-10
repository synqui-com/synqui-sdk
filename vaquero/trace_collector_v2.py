"""
Simple trace collector for in-memory trace storage - Version 2.

This version handles multiple traces properly by grouping spans by their parent trace.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TraceCollectorV2:
    """Simple in-memory trace collector that handles multiple traces."""
    
    def __init__(self, sdk: 'VaqueroSDK'):
        """Initialize the trace collector.
        
        Args:
            sdk: Reference to the main SDK instance
        """
        self.sdk = sdk
        self.config = sdk.config
        self._traces: Dict[str, Dict[str, Any]] = {}  # trace_id -> trace data
        self._spans: Dict[str, List[Dict[str, Any]]] = {}  # trace_id -> list of spans
        self._root_traces: Dict[str, str] = {}  # span_id -> root_trace_id mapping
    
    def process_span(self, span_data: Dict[str, Any]) -> None:
        """Process a span and determine which trace it belongs to.

        Args:
            span_data: Span data to process
        """
        span_id = span_data.get('span_id')
        parent_span_id = span_data.get('parent_span_id')
        session_id = span_data.get('session_id')
        metadata = span_data.get('metadata', {})
        agent_name = span_data.get('agent_name', 'unknown')

        # Debug logging for trace grouping
        logger.debug(f"Processing span: {agent_name} (span_id: {span_id})")
        logger.debug(f"  Session ID from span_data: {session_id}")
        logger.debug(f"  Session ID from metadata: {metadata.get('session_id')}")
        logger.debug(f"  Parent span ID: {parent_span_id}")
        logger.debug(f"  Metadata keys: {list(metadata.keys())}")

        if not span_id:
            logger.warning("Span has no ID, skipping")
            return

        # Enhanced trace grouping logic - get session_id from metadata if not in span_data
        if not session_id and metadata:
            session_id = metadata.get('session_id')

        logger.debug(f"  Final session ID: {session_id}")

        # Find the root trace for this span
        root_trace_id = self._find_root_trace(span_id, parent_span_id, session_id)

        if not root_trace_id:
            # This is a new root trace - check if it should be grouped with existing session
            root_trace_id = self._determine_trace_group(span_id, span_data)
            if root_trace_id:
                # Group with existing trace
                self._root_traces[span_id] = root_trace_id
                self._add_span_to_trace(root_trace_id, span_data)
            else:
                # This is a truly new root trace
                root_trace_id = span_id
                self._root_traces[span_id] = span_id
                self._start_trace(root_trace_id, span_data)
        else:
            # This span belongs to an existing trace
            self._add_span_to_trace(root_trace_id, span_data)

        # Update the mapping for this span
        self._root_traces[span_id] = root_trace_id
    
    def _find_root_trace(self, span_id: str, parent_span_id: Optional[str], session_id: Optional[str] = None) -> Optional[str]:
        """Find the root trace ID for a span.

        Args:
            span_id: Current span ID
            parent_span_id: Parent span ID
            session_id: Session ID for grouping related spans

        Returns:
            Root trace ID if found, None otherwise
        """
        # First try to find via parent relationship
        if parent_span_id and parent_span_id in self._root_traces:
            return self._root_traces[parent_span_id]

        # Check if this span is already mapped
        if span_id in self._root_traces:
            return self._root_traces[span_id]

        # If we have a session_id, look for other traces with the same session
        if session_id:
            logger.debug(f"  Looking for existing traces with session_id: {session_id}")
            for trace_id, trace_data in self._traces.items():
                existing_session = trace_data.get("session_id")
                logger.debug(f"    Checking trace {trace_id}: session_id={existing_session}, status={trace_data.get('status')}")
                if existing_session == session_id:
                    # Group with any trace that has the same session_id
                    # The idea is that all spans with the same session_id belong together
                    logger.debug(f"    Found matching trace: {trace_id}")
                    return trace_id
            logger.debug(f"  No matching traces found for session_id: {session_id}")

        return None

    def _determine_trace_group(self, span_id: str, span_data: Dict[str, Any]) -> Optional[str]:
        """Determine which trace group a span should belong to.

        Args:
            span_id: Current span ID
            span_data: Span data

        Returns:
            Trace ID to group with, or None for new trace
        """
        session_id = span_data.get('session_id')
        metadata = span_data.get('metadata', {})
        agent_name = span_data.get('agent_name', '')
        
        # Get session_id from metadata if not in span_data
        if not session_id and metadata:
            session_id = metadata.get('session_id')

        # Check if this is a main workflow span - if so, it should be the root of a new trace
        # but only if there's no existing trace with the same session_id
        if 'workflow' in agent_name.lower() or self._is_workflow_span(agent_name):
            # If we have a session_id, check if there's already a trace for this session
            if session_id:
                for trace_id, trace_data in self._traces.items():
                    if trace_data.get("session_id") == session_id:
                        # There's already a trace for this session, group with it
                        return trace_id
            # No existing trace for this session, so this workflow span should start a new trace
            return None

        # If we have a session_id, look for any trace in the same session
        if session_id:
            for trace_id, trace_data in self._traces.items():
                if trace_data.get("session_id") == session_id:
                    return trace_id

        # If no session grouping works, check for recent traces with similar characteristics
        # This is a fallback for cases where session_id might not be set properly
        current_time = span_data.get('start_time')
        if current_time:
            for trace_id, trace_data in self._traces.items():
                trace_start = trace_data.get("start_time")
                if (trace_start and trace_data.get("status") == "running"):
                    # If spans are within 30 seconds of each other, group them
                    try:
                        from datetime import datetime
                        trace_time = datetime.fromisoformat(trace_start.replace('Z', '+00:00'))
                        span_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                        if abs((span_time - trace_time).total_seconds()) < 30:
                            return trace_id
                    except:
                        pass

        return None

    def _determine_agent_hierarchy(self, span: Dict[str, Any], agent_name: str) -> tuple[int, str]:
        """Determine the hierarchy level and component type for an agent.

        Args:
            span: Span data
            agent_name: Name of the agent

        Returns:
            Tuple of (level, component_type)
        """
        # Logical agents - these represent business logic steps
        # Use pattern-based detection instead of hardcoded names
        logical_agents = set()

        # Check for workflow patterns - generic detection
        if self._is_workflow_span(agent_name):
            logical_agents.add(agent_name)
        # Check for agent patterns (ends with _agent)
        elif agent_name.endswith('_agent'):
            logical_agents.add(agent_name)
        # Check for stage-based agents (created from metadata stages)
        elif metadata.get('stage'):
            logical_agents.add(f"{metadata['stage']}_agent")
        # Check for spans with significant metadata that suggest they're logical agents
        elif (metadata.get('session_id') or
              metadata.get('stage') or
              span_data.get('duration_ms', 0) > 100):
            logical_agents.add(agent_name)

        # Internal components - these are LangChain implementation details
        # Use pattern-based detection instead of hardcoded names
        internal_components = set()

        # Check for known internal LangChain patterns
        internal_patterns = [
            "langchain:", "llm:", "tool:", "chain", "prompt_template", "output_parser",
            "chat_prompt_template", "str_output_parser"
        ]

        for pattern in internal_patterns:
            if pattern in agent_name.lower():
                internal_components.add(agent_name)
                break

        # Check if this is a logical agent
        if (agent_name in logical_agents or
            any(logical in agent_name.lower() for logical in ["validation", "analysis", "report"])):

            # Special handling for workflow spans
            if "workflow" in agent_name.lower():
                return 0, "workflow"  # Root level for workflows

            return 1, "logical_agent"

        # Check if this is an internal component
        if (agent_name in internal_components or
            agent_name.startswith("langchain:") or
            agent_name.startswith("llm:") or
            agent_name.startswith("tool:")):

            # Determine specific component type
            if agent_name.startswith("langchain:"):
                component_type = "langchain_component"
            elif agent_name.startswith("llm:"):
                component_type = "llm_component"
            elif agent_name.startswith("tool:"):
                component_type = "tool_component"
            else:
                component_type = "internal_component"

            return 2, component_type

        # Default case - treat as logical agent
        return 1, "agent"

    def _start_trace(self, trace_id: str, span_data: Dict[str, Any]) -> None:
        """Start a new trace.

        Args:
            trace_id: Unique identifier for the trace
            span_data: Initial span data for the trace
        """
        # Get session_id from metadata if not in span_data
        session_id = span_data.get("session_id")
        if not session_id and span_data.get("metadata"):
            session_id = span_data["metadata"].get("session_id")

        # Determine trace name - prefer workflow names over agent names
        agent_name = span_data.get("agent_name", "unnamed")
        if self._is_workflow_span(agent_name):
            trace_name = agent_name
        else:
            trace_name = f"workflow_{agent_name}"  # Prefix with workflow for non-workflow root spans

        # Store trace data
        self._traces[trace_id] = {
            "trace_id": trace_id,
            "parent_trace_id": span_data.get("parent_trace_id"),
            "session_id": session_id,
            "name": trace_name,
            "status": span_data.get("status", "running"),
            "start_time": span_data.get("start_time"),
            "end_time": span_data.get("end_time"),
            "duration_ms": int(round(span_data.get("duration_ms", 0))),
            "total_tokens": span_data.get("total_tokens", 0),
            "total_cost": span_data.get("cost", 0.0),
            "error_count": span_data.get("error_count", 0),
            "tags": span_data.get("tags", {}),
            "metadata": span_data.get("metadata", {}),
            "environment": self.config.environment,
            "hostname": span_data.get("hostname"),
            "sdk_version": "0.1.0"
        }
        
        # Initialize spans list
        self._spans[trace_id] = [span_data]
        
        logger.debug(f"Started trace {trace_id}")
    
    def _add_span_to_trace(self, trace_id: str, span_data: Dict[str, Any]) -> None:
        """Add a span to an existing trace.

        Args:
            trace_id: Trace ID to add span to
            span_data: Span data to add
        """
        if trace_id not in self._spans:
            self._spans[trace_id] = []

        self._spans[trace_id].append(span_data)

        # Update trace metrics
        self._update_trace_metrics(trace_id, span_data)

        # Update trace name if this is a workflow span and current name is not a workflow
        agent_name = span_data.get('agent_name', '')
        current_trace_name = self._traces[trace_id].get('name', '')

        logger.debug(f"  Checking if should update trace name: agent='{agent_name}', current='{current_trace_name}'")

        # Check if this is a workflow span (various naming patterns)
        is_workflow_span = (
            "workflow" in agent_name.lower() or
            "langchain" in agent_name.lower() or
            self._is_workflow_span(agent_name)
        )

        # Check if current trace name is already a workflow name
        current_is_workflow = (
            "workflow" in current_trace_name.lower() or
            "langchain" in current_trace_name.lower() or
            self._is_workflow_span(current_trace_name)
        )

        logger.debug(f"  is_workflow_span={is_workflow_span}, current_is_workflow={current_is_workflow}")

        if is_workflow_span and not current_is_workflow:
            logger.debug(f"Updating trace name from '{current_trace_name}' to '{agent_name}'")
            self._traces[trace_id]["name"] = agent_name
        else:
            logger.debug(f"Not updating trace name")

        logger.debug(f"Added span to trace {trace_id}: {span_data.get('agent_name', 'unnamed')}")
    
    def _update_trace_metrics(self, trace_id: str, span_data: Dict[str, Any]) -> None:
        """Update trace metrics with span data.
        
        Args:
            trace_id: Trace ID to update
            span_data: Span data to use for metrics
        """
        if trace_id not in self._traces:
            return
        
        trace = self._traces[trace_id]
        
        # Update tokens and cost
        trace["total_tokens"] += span_data.get("total_tokens", 0)
        trace["total_cost"] += span_data.get("cost", 0.0)
        
        # Update error count
        if span_data.get("status") == "failed":
            trace["error_count"] += 1
        
        # Update end time if this span is later
        span_end_time = span_data.get("end_time")
        if span_end_time:
            trace_end_time = trace.get("end_time")
            if not trace_end_time or span_end_time > trace_end_time:
                trace["end_time"] = span_end_time
                trace["duration_ms"] = int(round(span_data.get("duration_ms", 0)))
    
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
        
        # Send trace to API
        self._send_trace_to_api(trace_id)
        
        # Clean up
        del self._traces[trace_id]
        del self._spans[trace_id]
        
        # Remove from root traces mapping
        for span_id, root_id in list(self._root_traces.items()):
            if root_id == trace_id:
                del self._root_traces[span_id]
        
        logger.debug(f"Ended trace {trace_id}")
    
    def _send_trace_to_api(self, trace_id: str) -> None:
        """Send a complete trace to the API.
        
        Args:
            trace_id: Trace ID to send
        """
        if trace_id not in self._traces or trace_id not in self._spans:
            logger.warning(f"Trace {trace_id} not found for sending")
            return
        
        trace = self._traces[trace_id]
        spans = self._spans[trace_id]
        
        # Prepare trace data for API - fix field names and data types
        processed_spans = []
        for span in spans:
            agent_name = span.get("agent_name", span.get("name", ""))

            # Determine hierarchy level and component type
            level, component_type = self._determine_agent_hierarchy(span, agent_name)

            processed_span = {
                "trace_id": trace_id,  # Add trace_id field
                "agent_id": span.get("span_id", span.get("agent_id", "")),
                "name": agent_name,
                "type": span.get("type", ""),
                "description": span.get("description", ""),
                "status": span.get("status", "completed"),
                "start_time": span.get("start_time"),
                "end_time": span.get("end_time"),
                "duration_ms": int(round(span.get("duration_ms", 0))),
                "input_tokens": span.get("input_tokens", 0),
                "output_tokens": span.get("output_tokens", 0),
                "total_tokens": span.get("total_tokens", 0),
                "cost": span.get("cost", 0.0),
                "tags": span.get("tags", {}),
                "input_data": span.get("inputs", {}),
                "output_data": span.get("outputs", {}),
                "metadata": span.get("metadata", {}),
                "parent_agent_id": span.get("parent_span_id"),
                "level": level,
                "framework": "langchain",
                "component_type": component_type,
                "framework_metadata": span.get("tags", {})
            }
            processed_spans.append(processed_span)
        
        # Prepare trace data for API
        trace_data = {
            "traces": [trace],
            "agents": processed_spans
        }
        
        try:
            import requests
            
            # Build URL
            url = f"{self.config.endpoint}/api/v1/traces/batch"
            
            # Build headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Send request
            response = requests.post(url, headers=headers, json=trace_data, timeout=self.config.timeout)
            
            if response.status_code == 202:
                logger.debug(f"Successfully sent trace with {len(spans)} agents")
            else:
                logger.error(f"Failed to send trace: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending trace to API: {e}")

    def _is_workflow_span(self, agent_name: str) -> bool:
        """Check if an agent name represents a workflow span."""
        workflow_patterns = [
            'workflow', 'langchain', 'main_workflow', 'root_workflow',
            'orchestrator', 'coordinator', 'pipeline', 'process'
        ]

        return any(pattern in agent_name.lower() for pattern in workflow_patterns)

    def shutdown(self) -> None:
        """Shutdown the trace collector and send any remaining traces."""
        logger.info("Shutting down trace collector")
        
        # Send all remaining traces
        for trace_id in list(self._traces.keys()):
            self._send_trace_to_api(trace_id)
        
        # Clear all data
        self._traces.clear()
        self._spans.clear()
        self._root_traces.clear()
        
        logger.info("Trace collector shutdown complete")
