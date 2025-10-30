"""
LangGraph integration for Vaquero SDK.

This module provides specialized integration for LangGraph applications,
handling the unique timing and execution flow of LangGraph agent swarms.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager

from .sdk import VaqueroSDK, get_global_instance
from .chat_session import ChatSession
from .cost_calculator import calculate_cost

logger = logging.getLogger(__name__)

# Global registry for active handlers
_active_handlers = set()

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # Fallback for type hints


class VaqueroLangGraphHandler(BaseCallbackHandler):
    """
    Specialized Vaquero handler for LangGraph applications.
    
    This handler addresses the timing issues specific to LangGraph's
    agent swarm architecture by ensuring proper trace finalization
    and sending before SDK shutdown.
    
    Key Features:
    - Captures individual node executions (agents) within LangGraph workflows
    - Handles LangGraph-specific callbacks for comprehensive tracing
    - Maintains proper hierarchical structure (Session → Message/Orchestration → Agents → Components)
    - Supports chat session management for conversational applications
    """
    
    def __init__(self, session: Optional[ChatSession] = None, sdk: Optional[VaqueroSDK] = None):
        """Initialize the LangGraph handler with optional chat session.

        Args:
            session: Optional ChatSession for chat-based applications
            sdk: Vaquero SDK instance to use. If None, uses the global instance.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: "
                "pip install langchain langchain-openai langchain-community"
            )
        
        super().__init__()
        self.sdk = sdk or get_global_instance()
        self.session = session
        self._active_traces: Dict[str, Dict[str, Any]] = {}
        self._active_nodes: Dict[str, Dict[str, Any]] = {}  # Track individual node executions
        self._trace_finalization_pending = False
        self._message_sequence = 0  # Track message sequence for chat sessions
        self._current_orchestration_id = None  # Track current agent orchestration
        self._session_trace_id = None  # Single trace ID for the entire session
        self._session_monitoring_active = False  # Track if session monitoring is active
        self._last_activity_check = None  # Track last activity check time
        
        # Set up automatic session monitoring if session is provided
        if self.session:
            self._setup_automatic_session_monitoring()
        
        # Register this handler for shutdown
        _active_handlers.add(self)

    def _setup_automatic_session_monitoring(self) -> None:
        """Set up automatic session monitoring for timeout detection."""
        if not self.session:
            logger.debug("No session available for monitoring setup")
            return
            
        logger.debug(f"Setting up automatic session monitoring for session {self.session.session_id}")
        self._session_monitoring_active = True
        self._last_activity_check = datetime.utcnow()
        
        # Register session timeout callback
        self.session._register_timeout_callback(self._on_session_timeout)

    def _on_session_timeout(self, session: 'ChatSession') -> None:
        """Handle session timeout by automatically finalizing traces."""
        logger.info(f"Session {session.session_id} timed out - automatically finalizing traces")
        
        # Finalize traces automatically
        self.finalize_langgraph_traces()
        
        # Mark session monitoring as inactive
        self._session_monitoring_active = False

    def force_finalize_traces(self) -> None:
        """Force finalization of traces for testing purposes."""
        logger.debug("Force finalizing traces for testing")
        self.finalize_langgraph_traces()

    def _check_session_timeout(self) -> None:
        """Check if session has timed out and handle accordingly."""
        if not self.session or not self._session_monitoring_active:
            logger.debug(f"Session monitoring check - session: {self.session is not None}, active: {self._session_monitoring_active}")
            return
            
        # Check if session should end
        if self.session.should_end_session():
            logger.info(f"Session {self.session.session_id} should end - finalizing traces")
            self._on_session_timeout(self.session)
        else:
            logger.debug(f"Session {self.session.session_id} is still active")

    def handle_user_message(self, message_content: str, message_id: Optional[str] = None) -> None:
        """Handle a user message in a chat session.

        Args:
            message_content: The user's message content
            message_id: Optional message ID (generated if not provided)
        """
        if not self.session:
            logger.debug("No session available for user message")
            return

        # Initialize session trace ID if not set
        if self._session_trace_id is None:
            self._session_trace_id = str(uuid.uuid4())
            logger.debug(f"Initialized session trace ID: {self._session_trace_id}")

        # Generate message ID if not provided
        if not message_id:
            message_id = str(uuid.uuid4())

        # Increment message sequence for next response
        self._message_sequence += 1

        # Update session activity and metrics
        self.session.update_activity()
        self.session.add_message(tokens=0, cost=0.0)  # Tokens/cost will be calculated from agents
        
        # Check for session timeout
        self._check_session_timeout()

        # Create user message span (not a complete trace)
        user_span_data = {
            'trace_id': self._session_trace_id,  # Use session trace ID
            'agent_name': 'user_message',
            'function_name': 'user_input',
            'start_time': datetime.utcnow().isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'status': 'completed',
            'duration': 0,
            'inputs': {'content': message_content},
            'outputs': {'content': message_content},
            'session_id': self.session.session_id,
            'session_type': self.session.session_type,
            'chat_session_id': self.session.session_id,
            'message_type': 'user_message',
            'message_content': message_content,
            'message_sequence': self._message_sequence,
            'user_message_id': message_id,
            'framework': 'langgraph'
        }

        # Send user message span
        if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
            self.sdk._trace_collector.process_span(user_span_data)
            logger.debug(f"Sent user message span for session trace: {self._session_trace_id}")
        else:
            logger.warning("No trace collector available for user message")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start for LangGraph applications."""
        try:
            # DEBUG: Log the essential debugging info
            logger.info(f"🔧 LANGGRAPH CHAIN START - Run ID: {run_id}, Parent: {parent_run_id}, Serialized: {type(serialized).__name__}")
            logger.info(f"🔧 LANGGRAPH CHAIN START - Inputs: {inputs}")
            logger.info(f"🔧 LANGGRAPH CHAIN START - Tags: {tags}")
            logger.info(f"🔧 LANGGRAPH CHAIN START - Metadata: {metadata}")
            
            # SIMPLIFIED APPROACH: In LangGraph context, ALL node executions are agents
            # Extract agent name from metadata/tags (LangGraph nodes don't always have serialized data)
            agent_name = self._extract_node_name_from_metadata(metadata, tags)

            logger.info(f"🔧 LANGGRAPH NODE EXECUTION - Agent: '{agent_name}', Run ID: {run_id}, Parent: {parent_run_id}")

            # Set orchestration ID for real agents (not system nodes like __start__)
            # All agents responding to the same user message share the same orchestration ID
            if agent_name not in ['__start__', 'unknown'] and agent_name:
                self._current_orchestration_id = f"orchestration_{self._message_sequence}"
                logger.info(f"🔧 SET ORCHESTRATION ID - Using message-based orchestration: {self._current_orchestration_id} for agent: {agent_name} (message #{self._message_sequence})")
            else:
                logger.debug(f"🔧 SKIP ORCHESTRATION ID - System node or unknown agent: {agent_name}")

            # Create agent span for this LangGraph node execution
            self._create_agent_span(agent_name, run_id, parent_run_id, inputs, serialized, metadata)

        except Exception as e:
            logger.warning(f"Error in LangGraph chain start callback: {e}")
    
    def _extract_node_name_from_metadata(self, metadata: Optional[Dict[str, Any]], tags: Optional[List[str]]) -> str:
        """Extract LangGraph node name from metadata or tags when serialized is None.
        
        Args:
            metadata: LangGraph metadata containing node information
            tags: LangGraph tags
            
        Returns:
            str: Extracted node name or "unknown" if not found
        """
        if not metadata:
            return "unknown"
        
        logger.debug(f"🔧   Extracting node name from metadata: {metadata}")
        logger.debug(f"🔧   Available tags: {tags}")

        # Pattern 1: Check for explicit node name in metadata
        if 'node' in metadata:
            node_name = metadata['node']
            logger.info(f"🔧   Extracted agent name from node field: {node_name}")
            return node_name

        # Pattern 2: langgraph_node in metadata
        if metadata.get('langgraph_node') == 'agent':
            # Extract from langgraph_checkpoint_ns
            checkpoint_ns = metadata.get('langgraph_checkpoint_ns', '')
            logger.debug(f"🔧   Checkpoint NS: {checkpoint_ns}")
            if '|' in checkpoint_ns:
                # Format: "explainer:uuid|agent:uuid"
                node_part = checkpoint_ns.split('|')[0]
                if ':' in node_part:
                    node_name = node_part.split(':')[0]
                    logger.info(f"🔧   Extracted agent name from checkpoint_ns: {node_name}")
                    return node_name
        
        # Pattern 3: langgraph_path in metadata
        if 'langgraph_path' in metadata:
            path = metadata['langgraph_path']
            logger.debug(f"🔧   LangGraph path: {path}")
            if isinstance(path, tuple) and len(path) > 1:
                # Format: ('__pregel_pull', 'agent')
                node_name = path[1]
                logger.info(f"🔧   Extracted agent name from path: {node_name}")
                return node_name

        # Pattern 4: Check tags for known agent names
        if tags:
            known_agents = ["explainer", "developer", "summarizer", "analogy_creator", "vulnerability_expert"]
            for tag in tags:
                if tag in known_agents:
                    logger.info(f"🔧   Extracted agent name from tags: {tag}")
                    return tag

        # Pattern 5: Check for other metadata patterns
        if 'langgraph_triggers' in metadata:
            triggers = metadata['langgraph_triggers']
            logger.debug(f"🔧   LangGraph triggers: {triggers}")
            if isinstance(triggers, tuple) and len(triggers) > 0:
                trigger = triggers[0]
                if ':' in trigger:
                    # Format: "branch:to:agent"
                    parts = trigger.split(':')
                    if len(parts) > 2:
                        node_name = parts[2]
                        logger.info(f"🔧   Extracted agent name from triggers: {node_name}")
                        return node_name

        # Pattern 6: Check for serialized data in metadata
        if 'serialized' in metadata and isinstance(metadata['serialized'], dict):
            serialized = metadata['serialized']
            if 'name' in serialized:
                node_name = serialized['name']
                logger.info(f"🔧   Extracted agent name from serialized: {node_name}")
                return node_name
        
        logger.info(f"🔧   No node name found in metadata/tags, returning 'unknown'")
        return "unknown"
    
    def _extract_comprehensive_state(
        self, 
        inputs: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]], 
        serialized: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract comprehensive state information from LangGraph inputs and metadata.
        
        Args:
            inputs: The inputs to the LangGraph node
            metadata: LangGraph metadata
            serialized: Serialized node information
            
        Returns:
            Dict containing comprehensive state information
        """
        state_info = {
            'agent_state': {},
            'messages': [],
            'node_type': 'unknown',
            'langgraph_step': None,
            'langgraph_path': None,
            'checkpoint_ns': None
        }
        
        try:
            # Extract LangGraph specific metadata
            if metadata:
                state_info.update({
                    'langgraph_step': metadata.get('langgraph_step'),
                    'langgraph_path': metadata.get('langgraph_path'),
                    'checkpoint_ns': metadata.get('checkpoint_ns'),
                    'langgraph_node': metadata.get('langgraph_node'),
                    'langgraph_triggers': metadata.get('langgraph_triggers')
                })
            
            # Extract messages from inputs (common in LangGraph)
            if 'messages' in inputs:
                state_info['messages'] = inputs['messages']
                state_info['node_type'] = 'message_processing'
            elif 'input' in inputs:
                # Handle single input case
                state_info['agent_state'] = {'input': inputs['input']}
                state_info['node_type'] = 'single_input'
            else:
                # Handle multiple inputs
                state_info['agent_state'] = inputs
                state_info['node_type'] = 'multi_input'
            
            # Extract additional context from serialized data
            if serialized and isinstance(serialized, dict):
                state_info['serialized_info'] = {
                    'name': serialized.get('name'),
                    'id': serialized.get('id'),
                    'kwargs': serialized.get('kwargs', {})
                }
                
        except Exception as e:
            logger.warning(f"Error extracting comprehensive state: {e}")

        return state_info

    def _capture_state_snapshot(self, inputs: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Capture a complete state snapshot for checkpointing and diff analysis.

        Args:
            inputs: The current inputs to the node
            metadata: LangGraph metadata containing state information

        Returns:
            Dict containing state snapshot with timestamp and metadata
        """
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'state_hash': None,
            'state_size': 0,
            'checkpoint_id': None,
            'parent_checkpoint_id': None,
            'state_changes': {},
            'metadata': {}
        }

        try:
            # Extract state from inputs
            state_data = {}
            if 'messages' in inputs:
                state_data['messages'] = inputs['messages']
            elif 'input' in inputs:
                state_data['input'] = inputs['input']
            else:
                state_data = inputs

            # Calculate state size and hash for change detection
            import json
            state_str = json.dumps(state_data, sort_keys=True, default=str)
            snapshot['state_size'] = len(state_str)

            # Simple hash for state comparison (can be enhanced with proper hashing)
            snapshot['state_hash'] = hash(state_str)

            # Extract checkpoint information from metadata
            if metadata:
                snapshot.update({
                    'checkpoint_id': metadata.get('checkpoint_id'),
                    'parent_checkpoint_id': metadata.get('parent_checkpoint_id'),
                    'checkpoint_ns': metadata.get('checkpoint_ns'),
                    'checkpoint_ts': metadata.get('checkpoint_ts')
                })

                # Extract additional metadata
                snapshot['metadata'] = {
                    'langgraph_step': metadata.get('langgraph_step'),
                    'langgraph_path': metadata.get('langgraph_path'),
                    'langgraph_node': metadata.get('langgraph_node'),
                    'langgraph_triggers': metadata.get('langgraph_triggers'),
                    'configurable': metadata.get('configurable', {})
                }

            logger.debug(f"🔍 Captured state snapshot: hash={snapshot['state_hash']}, size={snapshot['state_size']}")

        except Exception as e:
            logger.warning(f"Error capturing state snapshot: {e}")

        return snapshot

    def _calculate_state_diff(self, previous_state: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the difference between two state snapshots.

        Args:
            previous_state: The previous state snapshot
            current_state: The current state snapshot

        Returns:
            Dict containing state differences and change metrics
        """
        diff = {
            'has_changes': False,
            'changes_detected': [],
            'added_keys': [],
            'removed_keys': [],
            'modified_keys': [],
            'state_size_change': 0,
            'timestamp_diff_ms': 0
        }

        try:
            # Compare state hashes
            prev_hash = previous_state.get('state_hash')
            curr_hash = current_state.get('state_hash')

            if prev_hash != curr_hash:
                diff['has_changes'] = True
                diff['changes_detected'].append('state_hash_changed')

            # Compare state sizes
            prev_size = previous_state.get('state_size', 0)
            curr_size = current_state.get('state_size', 0)
            diff['state_size_change'] = curr_size - prev_size

            # Calculate time difference
            if previous_state.get('timestamp') and current_state.get('timestamp'):
                from datetime import datetime
                prev_time = datetime.fromisoformat(previous_state['timestamp'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(current_state['timestamp'].replace('Z', '+00:00'))
                diff['timestamp_diff_ms'] = int((curr_time - prev_time).total_seconds() * 1000)

            # Compare metadata changes
            prev_metadata = previous_state.get('metadata', {})
            curr_metadata = current_state.get('metadata', {})

            prev_keys = set(prev_metadata.keys())
            curr_keys = set(curr_metadata.keys())

            diff['added_keys'] = list(curr_keys - prev_keys)
            diff['removed_keys'] = list(prev_keys - curr_keys)

            # Check for modified values
            common_keys = prev_keys & curr_keys
            for key in common_keys:
                if prev_metadata[key] != curr_metadata[key]:
                    diff['modified_keys'].append(key)

            if diff['added_keys'] or diff['removed_keys'] or diff['modified_keys']:
                diff['has_changes'] = True
                diff['changes_detected'].extend(['metadata_keys_changed'])

            logger.debug(f"🔍 Calculated state diff: changes={diff['has_changes']}, size_change={diff['state_size_change']}")

        except Exception as e:
            logger.warning(f"Error calculating state diff: {e}")

        return diff

    def _extract_checkpoint_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive checkpoint metadata from LangGraph.

        Args:
            metadata: LangGraph metadata containing checkpoint information

        Returns:
            Dict containing checkpoint metadata and state information
        """
        checkpoint_info = {
            'checkpoint_id': None,
            'parent_checkpoint_id': None,
            'checkpoint_ns': None,
            'checkpoint_ts': None,
            'checkpoint_config': {},
            'thread_id': None,
            'thread_ts': None,
            'state_channel_values': {},
            'state_channel_versions': {},
            'pending_writes': [],
            'pending_sends': []
        }

        try:
            if not metadata:
                return checkpoint_info

            # Extract checkpoint identifiers
            checkpoint_info.update({
                'checkpoint_id': metadata.get('checkpoint_id'),
                'parent_checkpoint_id': metadata.get('parent_checkpoint_id'),
                'checkpoint_ns': metadata.get('checkpoint_ns'),
                'checkpoint_ts': metadata.get('checkpoint_ts'),
                'thread_id': metadata.get('thread_id'),
                'thread_ts': metadata.get('thread_ts')
            })

            # Extract checkpoint configuration
            if 'checkpoint' in metadata:
                checkpoint = metadata['checkpoint']
                if isinstance(checkpoint, dict):
                    checkpoint_info['checkpoint_config'] = {
                        'configurable': checkpoint.get('configurable', {}),
                        'metadata': checkpoint.get('metadata', {}),
                        'created_at': checkpoint.get('created_at'),
                        'parent_checkpoint_id': checkpoint.get('parent_checkpoint_id')
                    }

            # Extract state channel information
            if 'channel_values' in metadata:
                checkpoint_info['state_channel_values'] = metadata['channel_values']

            if 'channel_versions' in metadata:
                checkpoint_info['state_channel_versions'] = metadata['channel_versions']

            # Extract pending operations
            if 'pending_writes' in metadata:
                checkpoint_info['pending_writes'] = metadata['pending_writes']

            if 'pending_sends' in metadata:
                checkpoint_info['pending_sends'] = metadata['pending_sends']

            logger.debug(f"🔍 Extracted checkpoint metadata: id={checkpoint_info['checkpoint_id']}")

        except Exception as e:
            logger.warning(f"Error extracting checkpoint metadata: {e}")

        return checkpoint_info

    def _extract_performance_metrics(self, start_time: datetime, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract performance metrics for the current operation.

        Args:
            start_time: When the operation started
            metadata: Additional metadata that might contain performance info

        Returns:
            Dict containing performance metrics
        """
        metrics = {
            'start_time': start_time.isoformat(),
            'current_time': datetime.utcnow().isoformat(),
            'duration_ms': 0,
            'memory_usage_mb': None,
            'cpu_usage_percent': None,
            'thread_count': None,
            'active_threads': None
        }

        try:
            # Calculate duration
            end_time = datetime.utcnow()
            metrics['duration_ms'] = int((end_time - start_time).total_seconds() * 1000)

            # Try to get memory usage (if psutil is available)
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024  # Convert to MB

                # Get CPU usage
                metrics['cpu_usage_percent'] = process.cpu_percent(interval=0.1)

                # Get thread information
                metrics['thread_count'] = len(process.threads())
                metrics['active_threads'] = len([t for t in process.threads() if t.user_time > 0])

            except ImportError:
                logger.debug("psutil not available for performance metrics")
            except Exception as e:
                logger.debug(f"Could not collect performance metrics: {e}")

            # Extract any performance metadata from LangGraph
            if metadata:
                if 'performance' in metadata:
                    perf_meta = metadata['performance']
                    if isinstance(perf_meta, dict):
                        metrics.update({
                            'graph_execution_time': perf_meta.get('execution_time'),
                            'node_count': perf_meta.get('node_count'),
                            'edge_count': perf_meta.get('edge_count'),
                            'checkpoint_count': perf_meta.get('checkpoint_count')
                        })

            logger.debug(f"🔍 Extracted performance metrics: duration={metrics['duration_ms']}ms, memory={metrics['memory_usage_mb']}MB")

        except Exception as e:
            logger.warning(f"Error extracting performance metrics: {e}")

        return metrics

    def _log_callback_event(self, event_type: str, run_id: str, details: Dict[str, Any], level: str = "info") -> None:
        """Log callback events with structured information for testing and debugging.

        Args:
            event_type: Type of callback event (e.g., 'chain_start', 'llm_end')
            run_id: The run ID for the event
            details: Dictionary of event details to log
            level: Logging level ('debug', 'info', 'warning', 'error')
        """
        try:
            # Create structured log message
            log_data = {
                'event_type': event_type,
                'run_id': run_id,
                'session_id': getattr(self.session, 'session_id', None) if self.session else None,
                'trace_id': self._session_trace_id,
                'timestamp': datetime.utcnow().isoformat(),
                **details
            }

            # Format log message
            log_message = f"🔧 LANGGRAPH {event_type.upper()} - {run_id}"
            if details.get('agent_name'):
                log_message += f" [{details['agent_name']}]"
            if details.get('duration'):
                log_message += f" ({details['duration']}ms)"
            if details.get('status'):
                log_message += f" [{details['status']}]"

            # Log with appropriate level
            if level == "debug":
                logger.debug(log_message)
            elif level == "info":
                logger.info(log_message)
            elif level == "warning":
                logger.warning(log_message)
            elif level == "error":
                logger.error(log_message)

            # Log structured data for detailed debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔧 Structured event data: {log_data}")

        except Exception as e:
            logger.warning(f"Error in callback event logging: {e}")

    def _log_execution_summary(self, run_id: str, execution_type: str, summary: Dict[str, Any]) -> None:
        """Log execution summary with key metrics and statistics.

        Args:
            run_id: The run ID for the execution
            execution_type: Type of execution ('graph', 'node', 'chain', 'llm', 'tool', 'retriever')
            summary: Dictionary containing execution summary data
        """
        try:
            # Extract key metrics
            duration = summary.get('duration', 0)
            status = summary.get('status', 'unknown')
            tokens = summary.get('total_tokens', 0)
            cost = summary.get('cost', 0.0)

            # Create summary message
            summary_msg = f"🔧 EXECUTION SUMMARY - {execution_type.upper()} {run_id}: "
            summary_msg += f"status={status}, duration={duration}ms"

            if tokens > 0:
                summary_msg += f", tokens={tokens}"
            if cost > 0:
                summary_msg += f", cost=${cost:.4f}"

            # Add component-specific metrics
            if execution_type == 'graph':
                node_count = summary.get('node_count', 0)
                checkpoint_count = summary.get('checkpoint_count', 0)
                if node_count > 0:
                    summary_msg += f", nodes={node_count}"
                if checkpoint_count > 0:
                    summary_msg += f", checkpoints={checkpoint_count}"
            elif execution_type == 'retriever':
                doc_count = summary.get('document_count', 0)
                if doc_count > 0:
                    summary_msg += f", docs={doc_count}"
            elif execution_type == 'llm':
                model = summary.get('model_name', 'unknown')
                provider = summary.get('model_provider', 'unknown')
                summary_msg += f", model={model}@{provider}"

            logger.info(summary_msg)

            # Log performance warnings
            if duration > 30000:  # 30 seconds
                logger.warning(f"🔧 PERFORMANCE WARNING - {execution_type} {run_id} took {duration}ms (>30s)")
            if cost > 1.0:  # $1.00
                logger.warning(f"🔧 COST WARNING - {execution_type} {run_id} cost ${cost:.4f} (>$1.00)")

        except Exception as e:
            logger.warning(f"Error logging execution summary: {e}")

    def _log_error_with_context(self, error: Exception, context: Dict[str, Any], run_id: str) -> None:
        """Log errors with comprehensive context information for debugging.

        Args:
            error: The exception that occurred
            context: Dictionary containing context information
            run_id: The run ID where the error occurred
        """
        try:
            import traceback

            # Get full stack trace
            stack_trace = traceback.format_exc()

            # Create error context
            error_context = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'run_id': run_id,
                'session_id': getattr(self.session, 'session_id', None) if self.session else None,
                'trace_id': self._session_trace_id,
                'timestamp': datetime.utcnow().isoformat(),
                'stack_trace': stack_trace,
                **context
            }

            # Log error with context
            logger.error(f"🔧 ERROR in LangGraph callback - {type(error).__name__}: {error}")
            logger.error(f"🔧 ERROR CONTEXT - Run ID: {run_id}, Session: {error_context['session_id']}")
            logger.debug(f"🔧 FULL ERROR CONTEXT: {error_context}")

            # Log to trace collector if available
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                error_span = {
                    'trace_id': self._session_trace_id,
                    'agent_name': f"error:{type(error).__name__}",
                    'function_name': f"error_handler",
                    'start_time': datetime.utcnow().isoformat(),
                    'end_time': datetime.utcnow().isoformat(),
                    'status': 'error',
                    'duration': 0,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'error_context': error_context,
                    'stack_trace': stack_trace,
                    'framework': 'langgraph',
                    'component_type': 'error'
                }

                # Add session information if available
                if self.session:
                    error_span.update({
                        'session_id': self.session.session_id,
                        'session_type': self.session.session_type,
                        'chat_session_id': self.session.session_id,
                        'message_type': 'error_occurred'
                    })

                self.sdk._trace_collector.process_span(error_span)
                logger.debug("Sent error span to trace collector")

        except Exception as log_error:
            # Last resort error logging
            logger.critical(f"CRITICAL: Failed to log error properly: {log_error}")
            logger.critical(f"Original error was: {error}")

    def _log_state_transition(self, from_state: Dict[str, Any], to_state: Dict[str, Any], transition_type: str) -> None:
        """Log state transitions with diff information.

        Args:
            from_state: The previous state
            to_state: The current state
            transition_type: Type of transition ('node_entry', 'node_exit', 'checkpoint', etc.)
        """
        try:
            # Calculate state diff
            state_diff = self._calculate_state_diff(from_state, to_state)

            # Log transition if there were changes
            if state_diff.get('has_changes', False):
                logger.info(f"🔄 STATE TRANSITION - {transition_type}: changes={len(state_diff.get('changes_detected', []))}")
                logger.debug(f"🔄 State diff details: {state_diff}")
            else:
                logger.debug(f"🔄 STATE TRANSITION - {transition_type}: no changes detected")

        except Exception as e:
            logger.warning(f"Error logging state transition: {e}")

    def _create_execution_report(self, run_id: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive execution report for the given run.

        Args:
            run_id: The run ID to create a report for
            execution_data: Dictionary containing all execution data

        Returns:
            Dict containing the comprehensive execution report
        """
        report = {
            'run_id': run_id,
            'session_id': getattr(self.session, 'session_id', None) if self.session else None,
            'trace_id': self._session_trace_id,
            'timestamp': datetime.utcnow().isoformat(),
            'execution_summary': {},
            'performance_metrics': {},
            'state_changes': [],
            'errors': [],
            'warnings': []
        }

        try:
            # Aggregate execution summary
            spans = execution_data.get('spans', [])
            total_duration = sum(span.get('duration', 0) for span in spans)
            total_tokens = sum(span.get('total_tokens', 0) for span in spans)
            total_cost = sum(span.get('cost', 0) for span in spans)

            report['execution_summary'] = {
                'total_spans': len(spans),
                'total_duration_ms': total_duration,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'avg_span_duration': total_duration / len(spans) if spans else 0,
                'span_types': {}
            }

            # Count span types
            for span in spans:
                span_type = span.get('component_type', 'unknown')
                report['execution_summary']['span_types'][span_type] = report['execution_summary']['span_types'].get(span_type, 0) + 1

            # Extract performance metrics
            report['performance_metrics'] = execution_data.get('performance_metrics', {})

            # Extract state changes
            report['state_changes'] = execution_data.get('state_changes', [])

            # Extract errors and warnings
            for span in spans:
                if span.get('status') == 'failed':
                    report['errors'].append({
                        'span_id': span.get('agent_name'),
                        'error': span.get('error'),
                        'timestamp': span.get('end_time')
                    })

            logger.info(f"🔧 Generated execution report for {run_id}: {len(spans)} spans, {total_duration}ms total, ${total_cost:.4f} cost")

        except Exception as e:
            logger.warning(f"Error creating execution report: {e}")

        return report

    def _capture_debugging_context(self, run_id: str, context_type: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Capture comprehensive debugging context for troubleshooting.

        Args:
            run_id: The run ID to capture context for
            context_type: Type of context ('execution', 'error', 'state', 'performance')
            additional_context: Additional context information

        Returns:
            Dict containing comprehensive debugging context
        """
        debug_context = {
            'run_id': run_id,
            'context_type': context_type,
            'timestamp': datetime.utcnow().isoformat(),
            'session_info': {},
            'trace_info': {},
            'system_info': {},
            'active_traces': {},
            'performance_snapshot': {}
        }

        try:
            # Session information
            if self.session:
                debug_context['session_info'] = {
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'session_start_time': getattr(self.session, '_start_time', None),
                    'message_count': getattr(self.session, '_message_count', 0),
                    'total_tokens': getattr(self.session, '_total_tokens', 0),
                    'total_cost': getattr(self.session, '_total_cost', 0.0)
                }

            # Trace information
            debug_context['trace_info'] = {
                'session_trace_id': self._session_trace_id,
                'active_trace_count': len(self._active_traces),
                'active_node_count': len(self._active_nodes),
                'trace_finalization_pending': self._trace_finalization_pending,
                'current_orchestration_id': self._current_orchestration_id
            }

            # System information
            import platform
            import sys
            debug_context['system_info'] = {
                'platform': platform.platform(),
                'python_version': sys.version,
                'process_id': None,
                'thread_id': None,
                'memory_usage': None
            }

            # Try to get process information
            try:
                import os
                debug_context['system_info']['process_id'] = os.getpid()
                debug_context['system_info']['thread_id'] = str(id(self))  # Object ID as thread proxy

                # Memory usage if psutil available
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    debug_context['system_info']['memory_usage'] = {
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024 if hasattr(memory_info, 'vms') else None
                    }
                except ImportError:
                    pass
            except Exception as e:
                logger.debug(f"Could not capture system info: {e}")

            # Active traces snapshot
            debug_context['active_traces'] = {
                trace_id: {
                    'agent_name': trace_data.get('span_data', {}).get('agent_name'),
                    'component_type': trace_data.get('span_data', {}).get('component_type'),
                    'start_time': trace_data.get('start_time'),
                    'status': trace_data.get('span_data', {}).get('status')
                }
                for trace_id, trace_data in self._active_traces.items()
            }

            # Performance snapshot
            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                start_time = trace_data.get('start_time')
                if start_time:
                    debug_context['performance_snapshot'] = self._extract_performance_metrics(start_time)

            # Add additional context
            if additional_context:
                debug_context['additional_context'] = additional_context

            logger.debug(f"🔧 Captured debugging context for {run_id}: type={context_type}")

        except Exception as e:
            logger.warning(f"Error capturing debugging context: {e}")

        return debug_context

    def _create_error_diagnostic_report(self, error: Exception, run_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive diagnostic report for errors.

        Args:
            error: The exception that occurred
            run_id: The run ID where the error occurred
            context: Context information about the error

        Returns:
            Dict containing comprehensive error diagnostic information
        """
        diagnostic = {
            'error_summary': {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'run_id': run_id,
                'timestamp': datetime.utcnow().isoformat()
            },
            'execution_context': {},
            'system_state': {},
            'trace_history': [],
            'recommendations': []
        }

        try:
            import traceback

            # Error details
            diagnostic['error_summary'].update({
                'stack_trace': traceback.format_exc(),
                'error_location': traceback.extract_tb(error.__traceback__)[-1] if error.__traceback__ else None
            })

            # Execution context
            diagnostic['execution_context'] = self._capture_debugging_context(run_id, 'error', context)

            # System state at time of error
            diagnostic['system_state'] = {
                'active_handlers': len(_active_handlers),
                'session_monitoring_active': self._session_monitoring_active,
                'trace_finalization_pending': self._trace_finalization_pending,
                'last_activity_check': self._last_activity_check.isoformat() if self._last_activity_check else None
            }

            # Recent trace history (last 10 spans)
            if hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                # This would need to be implemented in the trace collector
                # For now, we'll use the active traces
                diagnostic['trace_history'] = list(self._active_traces.keys())[-10:]

            # Generate recommendations based on error type
            error_type = type(error).__name__
            if 'timeout' in error_type.lower():
                diagnostic['recommendations'].append("Consider increasing timeout values for long-running operations")
                diagnostic['recommendations'].append("Check network connectivity and API rate limits")
            elif 'authentication' in error_type.lower() or 'auth' in error_type.lower():
                diagnostic['recommendations'].append("Verify API keys and authentication credentials")
                diagnostic['recommendations'].append("Check token expiration and refresh mechanisms")
            elif 'memory' in error_type.lower():
                diagnostic['recommendations'].append("Monitor memory usage and consider optimizing data structures")
                diagnostic['recommendations'].append("Consider processing data in smaller batches")
            elif 'network' in str(error).lower():
                diagnostic['recommendations'].append("Check network connectivity and retry logic")
                diagnostic['recommendations'].append("Consider implementing exponential backoff")
            else:
                diagnostic['recommendations'].append("Review error details and consider adding additional error handling")
                diagnostic['recommendations'].append("Check LangGraph configuration and input validation")

            logger.error(f"🔧 Generated error diagnostic report for {run_id}: {error_type}")

        except Exception as e:
            logger.warning(f"Error creating diagnostic report: {e}")
            diagnostic['diagnostic_error'] = str(e)

        return diagnostic

    def _validate_callback_chain_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the callback chain and detect potential issues.

        Returns:
            Dict containing validation results and any issues found
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        try:
            # Check for orphaned traces
            orphaned_traces = []
            current_time = datetime.utcnow()

            for trace_id, trace_data in self._active_traces.items():
                start_time = trace_data.get('start_time')
                if start_time and (current_time - start_time).total_seconds() > 300:  # 5 minutes
                    orphaned_traces.append({
                        'trace_id': trace_id,
                        'agent_name': trace_data.get('span_data', {}).get('agent_name'),
                        'start_time': start_time.isoformat(),
                        'age_seconds': (current_time - start_time).total_seconds()
                    })

            if orphaned_traces:
                validation['issues'].append(f"Found {len(orphaned_traces)} orphaned traces (>5min old)")
                validation['orphaned_traces'] = orphaned_traces

            # Check session monitoring
            if self.session and not self._session_monitoring_active:
                validation['warnings'].append("Session monitoring is not active despite having a session")

            # Check for excessive active traces
            if len(self._active_traces) > 50:
                validation['warnings'].append(f"High number of active traces: {len(self._active_traces)}")

            # Calculate metrics
            validation['metrics'] = {
                'active_traces': len(self._active_traces),
                'active_nodes': len(self._active_nodes),
                'session_trace_id': self._session_trace_id,
                'session_monitoring_active': self._session_monitoring_active,
                'trace_finalization_pending': self._trace_finalization_pending
            }

            # Overall validation status
            validation['is_valid'] = len(validation['issues']) == 0

            if not validation['is_valid']:
                logger.warning(f"🔧 Callback chain validation failed: {len(validation['issues'])} issues found")
            else:
                logger.debug("🔧 Callback chain validation passed")

        except Exception as e:
            logger.warning(f"Error validating callback chain: {e}")
            validation['validation_error'] = str(e)
            validation['is_valid'] = False

        return validation
    
    def _extract_comprehensive_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive output information from LangGraph outputs.
        
        Args:
            outputs: The outputs from the LangGraph node
            
        Returns:
            Dict containing comprehensive output information
        """
        output_info = {
            'raw_outputs': outputs,
            'messages': [],
            'agent_response': None,
            'tool_calls': [],
            'output_type': 'unknown'
        }
        
        try:
            # Extract messages from outputs (common in LangGraph)
            if 'messages' in outputs:
                output_info['messages'] = outputs['messages']
                output_info['output_type'] = 'message_processing'
                
                # Extract the last message as agent response
                if output_info['messages']:
                    last_message = output_info['messages'][-1]
                    if hasattr(last_message, 'content'):
                        output_info['agent_response'] = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        output_info['agent_response'] = last_message['content']
            
            # Extract tool calls if present
            if 'tool_calls' in outputs:
                output_info['tool_calls'] = outputs['tool_calls']
                output_info['output_type'] = 'tool_execution'
            elif 'actions' in outputs:
                output_info['tool_calls'] = outputs['actions']
                output_info['output_type'] = 'action_execution'
            
            # Handle single output case
            if 'output' in outputs:
                output_info['agent_response'] = outputs['output']
                output_info['output_type'] = 'single_output'
            elif 'result' in outputs:
                output_info['agent_response'] = outputs['result']
                output_info['output_type'] = 'result_output'
            
            # Extract additional metadata
            if 'metadata' in outputs:
                output_info['metadata'] = outputs['metadata']
                
        except Exception as e:
            logger.warning(f"Error extracting comprehensive outputs: {e}")
            
        return output_info
    
    def _force_token_extraction(self, response: Any, run_id: str, span_data: Dict[str, Any]) -> None:
        """Force token extraction from LLM response, even if callback chain is broken.
        
        Args:
            response: The LLM response object
            run_id: The run ID for this LLM call
            span_data: The span data to update
        """
        try:
            logger.info(f"🔧 FORCE TOKEN EXTRACTION - Run ID: {run_id}")
            logger.info(f"🔧 FORCE TOKEN EXTRACTION - Response type: {type(response)}")
            
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            # Extract tokens using the same logic as the main on_llm_end method
            if hasattr(response, 'generations') and response.generations:
                logger.info(f"🔧 FORCE TOKEN EXTRACTION - Processing {len(response.generations)} generations")
                
                for i, generation in enumerate(response.generations):
                    logger.info(f"🔧 FORCE TOKEN EXTRACTION - Generation {i}: {type(generation)}")
                    
                    if hasattr(generation, 'message') and hasattr(generation.message, 'usage_metadata'):
                        usage = generation.message.usage_metadata
                        logger.info(f"🔧 FORCE TOKEN EXTRACTION - Found usage_metadata: {usage}")
                        
                        input_tokens += usage.get('input_tokens', 0)
                        output_tokens += usage.get('output_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        
                        logger.info(f"🔧 FORCE TOKEN EXTRACTION - Extracted tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}")
                        break
            
            # Update span data with extracted tokens
            if total_tokens > 0:
                span_data.update({
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                })
                
                logger.info(f"🔧 FORCE TOKEN EXTRACTION - Updated span data with tokens: {total_tokens}")
                
                # Associate tokens with parent agent
                self._associate_tokens_with_parent_agent(run_id, input_tokens, output_tokens, total_tokens, 0.0)
            else:
                logger.warning(f"🔧 FORCE TOKEN EXTRACTION - No tokens extracted from response")
                
        except Exception as e:
            logger.warning(f"Error in force token extraction: {e}")
    
    def _setup_llm_completion_fallback(self, run_id: str, parent_run_id: Optional[str]) -> None:
        """Set up a fallback mechanism to capture LLM completion if on_llm_end is not triggered.
        
        Args:
            run_id: The LLM run ID
            parent_run_id: The parent run ID
        """
        try:
            import threading
            import time
            
            def check_llm_completion():
                """Check if LLM has completed and extract tokens if needed."""
                max_wait_time = 30  # Maximum wait time in seconds
                check_interval = 0.5  # Check every 500ms
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    
                    # Check if LLM is still running
                    if run_id in self._active_traces:
                        trace_data = self._active_traces[run_id]
                        span_data = trace_data['span_data']
                        
                        # If LLM is still running after 5 seconds, try to extract tokens
                        if elapsed_time >= 5.0 and span_data.get('status') == 'running':
                            logger.info(f"🔧 LLM COMPLETION FALLBACK - Attempting token extraction for run_id: {run_id}")
                            
                            # Try to find the response in the active traces
                            # This is a fallback mechanism for when on_llm_end is not called
                            if 'response' in trace_data:
                                response = trace_data['response']
                                self._force_token_extraction(response, run_id, span_data)
                                break
                    
                    # If LLM completed normally, break
                    if run_id not in self._active_traces:
                        break
            
            # Start the fallback check in a separate thread
            fallback_thread = threading.Thread(target=check_llm_completion, daemon=True)
            fallback_thread.start()
            
            logger.debug(f"🔧 LLM COMPLETION FALLBACK - Set up for run_id: {run_id}")
            
        except Exception as e:
            logger.warning(f"Error setting up LLM completion fallback: {e}")
    
    def _associate_tokens_with_parent_agent(self, llm_run_id: str, input_tokens: int, output_tokens: int, total_tokens: int, cost: float) -> None:
        """Associate LLM tokens with the parent agent span.
        
        Args:
            llm_run_id: The run_id of the LLM call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total number of tokens
            cost: Calculated cost
        """
        try:
            if llm_run_id not in self._active_traces:
                logger.warning(f"No active trace found for LLM run_id: {llm_run_id}")
                return

            llm_trace_data = self._active_traces[llm_run_id]
            llm_span_data = llm_trace_data['span_data']
            parent_run_id = llm_trace_data.get('parent_run_id')
            
            logger.info(f"🔧 TOKEN ASSOCIATION - LLM run_id: {llm_run_id}, parent_run_id: {parent_run_id}, tokens: {total_tokens}")

            # Find the parent agent span and update it with token information
            parent_agent_found = False
            
            if parent_run_id and parent_run_id in self._active_traces:
                parent_trace_data = self._active_traces[parent_run_id]
                parent_span_data = parent_trace_data['span_data']
                parent_agent_name = parent_span_data.get('agent_name', 'unknown')

                logger.info(f"🔧 Found parent agent: {parent_agent_name}")

                # Update parent agent with token information
                current_input_tokens = parent_span_data.get('input_tokens', 0)
                current_output_tokens = parent_span_data.get('output_tokens', 0)
                current_total_tokens = parent_span_data.get('total_tokens', 0)
                current_cost = parent_span_data.get('cost', 0.0)

                # Add tokens to existing counts (in case there are multiple LLM calls)
                new_input_tokens = current_input_tokens + input_tokens
                new_output_tokens = current_output_tokens + output_tokens
                new_total_tokens = current_total_tokens + total_tokens
                new_cost = current_cost + cost

                parent_span_data.update({
                    'input_tokens': new_input_tokens,
                    'output_tokens': new_output_tokens,
                    'total_tokens': new_total_tokens,
                    'cost': new_cost
                })

                logger.info(f"🔧 UPDATED parent agent {parent_agent_name} - tokens: {new_total_tokens} (was {current_total_tokens}), cost: {new_cost}")

                # Also update the trace data for consistency
                parent_trace_data['span_data'] = parent_span_data
                parent_agent_found = True
                
            else:
                # Fallback: Find the most recent LangGraph agent in active traces
                logger.warning(f"🔧 No direct parent found for LLM run_id: {llm_run_id}, searching for recent LangGraph agent")
                
                # Look for the most recent LangGraph agent (component_type == 'agent')
                most_recent_agent = None
                most_recent_time = None
                
                for trace_id, trace_data in self._active_traces.items():
                    span_data = trace_data.get('span_data', {})
                    if span_data.get('component_type') == 'agent':
                        start_time = trace_data.get('start_time')
                        if start_time and (most_recent_time is None or start_time > most_recent_time):
                            most_recent_agent = trace_data
                            most_recent_time = start_time
                
                if most_recent_agent:
                    parent_span_data = most_recent_agent['span_data']
                    parent_agent_name = parent_span_data.get('agent_name', 'unknown')
                    
                    logger.info(f"🔧 Found most recent LangGraph agent: {parent_agent_name}")

                    # Update parent agent with token information
                    current_input_tokens = parent_span_data.get('input_tokens', 0)
                    current_output_tokens = parent_span_data.get('output_tokens', 0)
                    current_total_tokens = parent_span_data.get('total_tokens', 0)
                    current_cost = parent_span_data.get('cost', 0.0)

                    # Add tokens to existing counts (in case there are multiple LLM calls)
                    new_input_tokens = current_input_tokens + input_tokens
                    new_output_tokens = current_output_tokens + output_tokens
                    new_total_tokens = current_total_tokens + total_tokens
                    new_cost = current_cost + cost

                    parent_span_data.update({
                        'input_tokens': new_input_tokens,
                        'output_tokens': new_output_tokens,
                        'total_tokens': new_total_tokens,
                        'cost': new_cost
                    })

                    logger.info(f"🔧 UPDATED most recent agent {parent_agent_name} - tokens: {new_total_tokens} (was {current_total_tokens}), cost: {new_cost}")
                    parent_agent_found = True
                else:
                    logger.warning(f"🔧 No LangGraph agent found to associate tokens with")

            if not parent_agent_found:
                logger.warning(f"🔧 No parent agent found for LLM run_id: {llm_run_id}")

        except Exception as e:
            logger.warning(f"Error associating tokens with parent agent: {e}")
    
    def _create_agent_span(
        self,
        agent_name: str,
        run_id: str,
        parent_run_id: Optional[str],
        inputs: Dict[str, Any],
        serialized: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Create an agent span for LangGraph agent execution."""
        try:
            logger.info(f"🔧 CREATING LANGGRAPH AGENT SPAN:")
            logger.info(f"🔧   Agent name: {agent_name}")
            logger.info(f"🔧   Run ID: {run_id}")
            logger.info(f"🔧   Parent Run ID: {parent_run_id}")
            logger.info(f"🔧   Inputs: {inputs}")
            logger.info(f"🔧   Metadata: {metadata}")
            
            # Extract comprehensive state information
            state_info = self._extract_comprehensive_state(inputs, metadata, serialized)
            logger.info(f"🔧   Extracted state info: {state_info}")
            
            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.info(f"🔧   Initialized session trace ID: {self._session_trace_id}")
            else:
                logger.info(f"🔧   Using existing session trace ID: {self._session_trace_id}")
            
            # Check for session timeout
            self._check_session_timeout()

            # Create agent span data with LangGraph trace ID and metadata
            agent_span_data = {
                'trace_id': self._session_trace_id,  # Use LangGraph trace ID
                'agent_name': agent_name,  # Use just the agent name, not prefixed
                'function_name': agent_name,
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {
                    'raw_inputs': inputs, 
                    'serialized': serialized,
                    'state_info': state_info
                },
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'agent',
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'agent_state': state_info.get('agent_state', {}),
                'messages': state_info.get('messages', []),
                'node_type': state_info.get('node_type', 'unknown'),
                'langgraph_metadata': {
                    'langgraph_step': state_info.get('langgraph_step'),
                    'langgraph_path': state_info.get('langgraph_path'),
                    'checkpoint_ns': state_info.get('checkpoint_ns'),
                    'langgraph_node': state_info.get('langgraph_node'),
                    'langgraph_triggers': state_info.get('langgraph_triggers')
                }
            }

            # Add session information if available
            if self.session:
                agent_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'agent_execution',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })

            # Store span data for on_agent_end
            self._active_traces[run_id] = {
                'span_data': agent_span_data,
                'start_time': datetime.utcnow(),
                'agent_name': agent_name
            }

            # Track individual node execution
            self._active_nodes[run_id] = {
                'start_time': datetime.utcnow(),
                'parent_run_id': parent_run_id,
                'agent_name': agent_name,
                'inputs': inputs,
                'metadata': metadata or {}
            }

            logger.debug(f"LangGraph agent start: {agent_name} (run_id: {run_id}, parent: {parent_run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph agent span creation: {e}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH CHAIN END - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH CHAIN END - Outputs: {outputs}")
            
            if run_id not in self._active_traces:
                logger.warning(f"🔧 LANGGRAPH CHAIN END - No active trace found for chain run_id: {run_id}")
                return

            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']

            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Extract comprehensive output information
            output_info = self._extract_comprehensive_outputs(outputs)
            logger.info(f"🔧 LANGGRAPH CHAIN END - Extracted output info: {output_info}")

            # Determine if this is an agent or chain span
            component_type = span_data.get('component_type', 'chain')
            if component_type == 'agent':
                agent_name = trace_data.get('agent_name', 'unknown')
                logger.debug(f"LangGraph agent end - {agent_name} (run_id: {run_id}, duration: {duration_ms}ms)")
            else:
                chain_name = trace_data.get('chain_name', 'unknown')
                logger.debug(f"LangGraph chain end - {chain_name} (run_id: {run_id}, duration: {duration_ms}ms)")

            # Update span data with comprehensive output information
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {
                    'raw_outputs': outputs,
                    'output_info': output_info
                },
                'agent_response': output_info.get('agent_response'),
                'output_messages': output_info.get('messages', []),
                'tool_calls': output_info.get('tool_calls', []),
                'output_type': output_info.get('output_type', 'unknown')
            })

            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.debug(f"Sent LangGraph span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for LangGraph span")

            # Clean up
            del self._active_traces[run_id]
            if run_id in self._active_nodes:
                del self._active_nodes[run_id]
            
            logger.debug(f"LangGraph chain end: {run_id}")

        except Exception as e:
            logger.warning(f"Error in LangGraph chain end callback: {e}")
    
    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error for LangGraph."""
        try:
            super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LangGraph chain error callback: {e}")
        
        # Mark trace as errored
        if run_id in self._active_traces:
            self._active_traces[run_id]['error'] = str(error)
            self._active_traces[run_id]['end_time'] = datetime.utcnow()

    def on_graph_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle graph start for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH GRAPH START - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH GRAPH START - Inputs: {inputs}")
            logger.info(f"🔧 LANGGRAPH GRAPH START - Tags: {tags}")
            logger.info(f"🔧 LANGGRAPH GRAPH START - Metadata: {metadata}")

            # Extract graph name from serialized data or metadata
            graph_name = "langgraph_workflow"
            if isinstance(serialized, dict):
                graph_name = serialized.get('name', graph_name)
            elif metadata and 'graph_name' in metadata:
                graph_name = metadata['graph_name']

            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.info(f"🔧 Initialized session trace ID for graph: {self._session_trace_id}")

            # Check for session timeout
            self._check_session_timeout()

            # Create graph span data with comprehensive metadata
            graph_span_data = {
                'trace_id': self._session_trace_id,
                'agent_name': f"graph:{graph_name}",
                'function_name': f"graph:{graph_name}",
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {
                    'raw_inputs': inputs,
                    'serialized': serialized,
                    'graph_config': {
                        'tags': tags or [],
                        'metadata': metadata or {},
                        'run_id': run_id,
                        'parent_run_id': parent_run_id
                    }
                },
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'graph',
                'graph_name': graph_name,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'graph_metadata': {
                    'graph_start_time': datetime.utcnow().isoformat(),
                    'graph_execution_context': {
                        'run_id': run_id,
                        'parent_run_id': parent_run_id,
                        'tags': tags or [],
                        'metadata': metadata or {}
                    }
                }
            }

            # Add session information if available
            if self.session:
                graph_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'graph_execution',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })

            # Store graph span data for on_graph_end
            self._active_traces[run_id] = {
                'span_data': graph_span_data,
                'start_time': datetime.utcnow(),
                'graph_name': graph_name,
                'parent_run_id': parent_run_id
            }

            logger.info(f"🔧 LANGGRAPH GRAPH START - Created graph span: {graph_name} (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph graph start callback: {e}")

    def on_graph_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle graph end for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH GRAPH END - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH GRAPH END - Outputs: {outputs}")

            if run_id not in self._active_traces:
                logger.warning(f"🔧 LANGGRAPH GRAPH END - No active trace found for graph run_id: {run_id}")
                return

            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']

            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Extract comprehensive output information
            output_info = self._extract_graph_outputs(outputs)
            logger.info(f"🔧 LANGGRAPH GRAPH END - Extracted output info: {output_info}")

            # Update span data with comprehensive output information
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {
                    'raw_outputs': outputs,
                    'output_info': output_info
                },
                'graph_result': output_info.get('graph_result'),
                'execution_summary': output_info.get('execution_summary', {}),
                'graph_metadata': {
                    **span_data.get('graph_metadata', {}),
                    'graph_end_time': end_time.isoformat(),
                    'total_execution_time_ms': duration_ms,
                    'execution_status': 'completed'
                }
            })

            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.info(f"🔧 LANGGRAPH GRAPH END - Sent graph span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for LangGraph graph span")

            # Clean up
            del self._active_traces[run_id]
            logger.info(f"🔧 LANGGRAPH GRAPH END - Completed graph execution (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph graph end callback: {e}")

    def on_graph_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle graph error for LangGraph."""
        try:
            logger.error(f"🔧 LANGGRAPH GRAPH ERROR - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.error(f"🔧 LANGGRAPH GRAPH ERROR - Error: {error}")

            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                span_data = trace_data['span_data']
                start_time = trace_data['start_time']

                # Calculate duration
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Update span data for error
                span_data.update({
                    'end_time': end_time.isoformat(),
                    'status': 'failed',
                    'duration': duration_ms,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'graph_metadata': {
                        **span_data.get('graph_metadata', {}),
                        'graph_end_time': end_time.isoformat(),
                        'total_execution_time_ms': duration_ms,
                        'execution_status': 'failed',
                        'failure_reason': str(error)
                    }
                })

                # Send error span to trace collector
                if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                    self.sdk._trace_collector.process_span(span_data)
                    logger.error(f"🔧 LANGGRAPH GRAPH ERROR - Sent error span for graph run_id: {run_id}")

                # Clean up
                del self._active_traces[run_id]

            logger.error(f"🔧 LANGGRAPH GRAPH ERROR - Graph execution failed (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph graph error callback: {e}")

    def _extract_graph_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive graph output information."""
        output_info = {
            'raw_outputs': outputs,
            'graph_result': None,
            'execution_summary': {},
            'output_type': 'unknown'
        }

        try:
            # Extract graph result
            if 'output' in outputs:
                output_info['graph_result'] = outputs['output']
                output_info['output_type'] = 'single_output'
            elif 'result' in outputs:
                output_info['graph_result'] = outputs['result']
                output_info['output_type'] = 'result_output'
            else:
                # Handle multiple outputs
                output_info['graph_result'] = outputs
                output_info['output_type'] = 'multi_output'

            # Extract execution summary information
            if 'metadata' in outputs:
                metadata = outputs['metadata']
                output_info['execution_summary'].update({
                    'node_count': metadata.get('node_count'),
                    'edge_count': metadata.get('edge_count'),
                    'execution_time': metadata.get('execution_time'),
                    'checkpoint_count': metadata.get('checkpoint_count')
                })

            # Extract final state information
            if 'state' in outputs:
                output_info['final_state'] = outputs['state']
                output_info['execution_summary']['has_final_state'] = True

        except Exception as e:
            logger.warning(f"Error extracting graph outputs: {e}")

        return output_info

    def on_agent_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent start for LangGraph nodes."""
        try:
            super().on_agent_start(
                serialized, inputs, run_id=run_id, parent_run_id=parent_run_id,
                tags=tags, metadata=metadata, **kwargs
            )
            
            # Extract agent name from serialized data
            agent_name = serialized.get('name', 'unknown_agent') if serialized else 'unknown_agent'
            
            # Track individual node execution
            self._active_nodes[run_id] = {
                'start_time': datetime.utcnow(),
                'parent_run_id': parent_run_id,
                'agent_name': agent_name,
                'inputs': inputs,
                'metadata': metadata or {}
            }
            
            logger.debug(f"Started LangGraph node: {agent_name} (run_id: {run_id})")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph agent start callback: {e}")

    def on_agent_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent end for LangGraph nodes."""
        try:
            super().on_agent_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LangGraph agent end callback: {e}")
        
        # Update node info and send trace
        if run_id in self._active_nodes:
            node_info = self._active_nodes[run_id]
            node_info['end_time'] = datetime.utcnow()
            node_info['outputs'] = outputs
            
            # Calculate duration
            duration = (node_info['end_time'] - node_info['start_time']).total_seconds()
            
            # Create agent orchestration ID if this is the first agent in a new orchestration
            if not self._current_orchestration_id:
                self._current_orchestration_id = str(uuid.uuid4())
            
            # Send individual agent span
            self._send_agent_span(node_info, duration)
            
            # Clean up
            del self._active_nodes[run_id]
            
            logger.debug(f"Completed LangGraph node: {node_info['agent_name']} (run_id: {run_id})")

    def on_agent_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent error for LangGraph nodes."""
        try:
            super().on_agent_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LangGraph agent error callback: {e}")
        
        # Mark node as errored
        if run_id in self._active_nodes:
            self._active_nodes[run_id]['error'] = str(error)
            self._active_nodes[run_id]['end_time'] = datetime.utcnow()
            
            # Send error span
            node_info = self._active_nodes[run_id]
            duration = (node_info['end_time'] - node_info['start_time']).total_seconds()
            self._send_agent_span(node_info, duration, is_error=True)
            
            # Clean up
            del self._active_nodes[run_id]

    def on_node_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle node start for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH NODE START - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH NODE START - Inputs: {inputs}")
            logger.info(f"🔧 LANGGRAPH NODE START - Tags: {tags}")
            logger.info(f"🔧 LANGGRAPH NODE START - Metadata: {metadata}")

            # Extract node name from metadata or serialized data
            node_name = self._extract_node_name_from_metadata(metadata, tags)

            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.info(f"🔧 Initialized session trace ID for node: {self._session_trace_id}")

            # Check for session timeout
            self._check_session_timeout()

            # Extract comprehensive state information
            state_info = self._extract_comprehensive_state(inputs, metadata, serialized)

            # Create node span data with enhanced metadata
            node_span_data = {
                'trace_id': self._session_trace_id,
                'agent_name': f"node:{node_name}",
                'function_name': f"node:{node_name}",
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {
                    'raw_inputs': inputs,
                    'serialized': serialized,
                    'state_info': state_info
                },
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'node',
                'node_name': node_name,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'node_metadata': {
                    'node_start_time': datetime.utcnow().isoformat(),
                    'node_execution_context': {
                        'run_id': run_id,
                        'parent_run_id': parent_run_id,
                        'tags': tags or [],
                        'metadata': metadata or {}
                    },
                    'node_type': state_info.get('node_type', 'unknown'),
                    'checkpoint_ns': state_info.get('checkpoint_ns'),
                    'langgraph_step': state_info.get('langgraph_step'),
                    'langgraph_path': state_info.get('langgraph_path'),
                    'langgraph_triggers': state_info.get('langgraph_triggers')
                }
            }

            # Add session information if available
            if self.session:
                node_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'node_execution',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })

            # Store node span data for on_node_end
            self._active_traces[run_id] = {
                'span_data': node_span_data,
                'start_time': datetime.utcnow(),
                'node_name': node_name,
                'parent_run_id': parent_run_id
            }

            logger.info(f"🔧 LANGGRAPH NODE START - Created node span: {node_name} (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph node start callback: {e}")

    def on_node_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle node end for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH NODE END - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH NODE END - Outputs: {outputs}")

            if run_id not in self._active_traces:
                logger.warning(f"🔧 LANGGRAPH NODE END - No active trace found for node run_id: {run_id}")
                return

            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']

            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Extract comprehensive output information
            output_info = self._extract_comprehensive_outputs(outputs)
            logger.info(f"🔧 LANGGRAPH NODE END - Extracted output info: {output_info}")

            # Update span data with comprehensive output information
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {
                    'raw_outputs': outputs,
                    'output_info': output_info
                },
                'node_result': output_info.get('agent_response'),
                'output_messages': output_info.get('messages', []),
                'tool_calls': output_info.get('tool_calls', []),
                'output_type': output_info.get('output_type', 'unknown'),
                'node_metadata': {
                    **span_data.get('node_metadata', {}),
                    'node_end_time': end_time.isoformat(),
                    'total_execution_time_ms': duration_ms,
                    'execution_status': 'completed'
                }
            })

            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.info(f"🔧 LANGGRAPH NODE END - Sent node span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for LangGraph node span")

            # Clean up
            del self._active_traces[run_id]
            logger.info(f"🔧 LANGGRAPH NODE END - Completed node execution (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph node end callback: {e}")

    def on_node_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle node error for LangGraph."""
        try:
            logger.error(f"🔧 LANGGRAPH NODE ERROR - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.error(f"🔧 LANGGRAPH NODE ERROR - Error: {error}")

            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                span_data = trace_data['span_data']
                start_time = trace_data['start_time']

                # Calculate duration
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Update span data for error
                span_data.update({
                    'end_time': end_time.isoformat(),
                    'status': 'failed',
                    'duration': duration_ms,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'node_metadata': {
                        **span_data.get('node_metadata', {}),
                        'node_end_time': end_time.isoformat(),
                        'total_execution_time_ms': duration_ms,
                        'execution_status': 'failed',
                        'failure_reason': str(error)
                    }
                })

                # Send error span to trace collector
                if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                    self.sdk._trace_collector.process_span(span_data)
                    logger.error(f"🔧 LANGGRAPH NODE ERROR - Sent error span for node run_id: {run_id}")

                # Clean up
                del self._active_traces[run_id]

            logger.error(f"🔧 LANGGRAPH NODE ERROR - Node execution failed (run_id: {run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph node error callback: {e}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH LLM START - Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH LLM START - Prompts: {prompts}")
            logger.info(f"🔧 LANGGRAPH LLM START - Serialized: {serialized}")
            logger.info(f"🔧 LANGGRAPH LLM START - Tags: {tags}")
            logger.info(f"🔧 LANGGRAPH LLM START - Metadata: {metadata}")
            
            # Extract model information
            model_name = "unknown"
            model_provider = "unknown"
            
            if isinstance(serialized, dict):
                if serialized.get("kwargs", {}).get("model"):
                    model_name = serialized["kwargs"]["model"]
                elif serialized.get("model"):
                    model_name = serialized["model"]
                elif serialized.get("model_name"):
                    model_name = serialized["model_name"]
                elif serialized.get("name"):
                    model_name = serialized["name"]
                
                # Determine provider
                if "google" in model_name.lower() or "gemini" in model_name.lower():
                    model_provider = "google"
                elif "openai" in model_name.lower():
                    model_provider = "openai"
                elif "anthropic" in model_name.lower():
                    model_provider = "anthropic"
                
                logger.debug(f"LangGraph extracted model name: {model_name}, provider: {model_provider}")
            
            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.debug(f"Initialized session trace ID in LLM start: {self._session_trace_id}")
            
            # Check for session timeout
            self._check_session_timeout()
            
            # Create LLM span data with LangGraph trace ID and metadata
            llm_span_data = {
                'trace_id': self._session_trace_id,  # Use LangGraph trace ID
                'agent_name': f"llm:{model_name}",
                'function_name': f"llm:{model_name}",
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {'prompts': prompts},
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'llm',
                'model_name': model_name,
                'model_provider': model_provider,
                'model_parameters': serialized.get('kwargs', {}),
                'input_tokens': 0,  # Will be updated in on_llm_end
                'output_tokens': 0,  # Will be updated in on_llm_end
                'cost': 0.0  # Will be calculated in on_llm_end
            }
            
            # Add session information if available
            if self.session:
                llm_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'llm_call',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })
            
            # Add parent context for better span attribution
            if parent_run_id and parent_run_id in self._active_traces:
                parent_trace = self._active_traces[parent_run_id]
                parent_span_data = parent_trace.get('span_data', {})
                parent_agent_name = parent_span_data.get('agent_name', '')
                parent_component_type = parent_span_data.get('component_type', '')
                
                # If parent is a LangGraph agent, use it as parent
                if parent_component_type == 'agent':
                    llm_span_data['parent_agent_name'] = parent_agent_name
                    llm_span_data['parent_span_id'] = parent_run_id
                    logger.debug(f"LLM span attributed to parent agent: {parent_agent_name} (type: {parent_component_type})")
            
            # Store span data for on_llm_end
            self._active_traces[run_id] = {
                'span_data': llm_span_data,
                'start_time': datetime.utcnow(),
                'model_name': model_name,
                'model_provider': model_provider,
                'parent_run_id': parent_run_id  # FIXED: Store parent_run_id for token association
            }
            
            # CRITICAL FIX: Set up a fallback mechanism to capture LLM completion
            # This ensures we capture tokens even if on_llm_end callback is not triggered
            self._setup_llm_completion_fallback(run_id, parent_run_id)
            
            logger.debug(f"LangGraph LLM start: {run_id} (parent: {parent_run_id})")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph LLM start callback: {e}")

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH LLM END - Run ID: {run_id}")
            logger.info(f"🔧 LANGGRAPH LLM END - Response type: {type(response)}")
            logger.info(f"🔧 LANGGRAPH LLM END - Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            
            if run_id not in self._active_traces:
                logger.warning(f"🔧 LANGGRAPH LLM END - No active trace found for LLM run_id: {run_id}")
                return
            
            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']
            
            logger.info(f"🔧 Span data before token extraction: {span_data}")

            # Debug logging for token extraction
            logger.info(f"🔧 LANGGRAPH LLM END - Processing LLM response with {len(response.generations) if response.generations else 0} generations")
            
            # CRITICAL FIX: Force token extraction even if callback chain is broken
            self._force_token_extraction(response, run_id, span_data)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Extract token usage from response - FIXED for Google Gemini
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            reasoning_tokens = 0
            reasoning_text = None

            if isinstance(response, LLMResult):
                logger.info(f"🔧 Processing LLMResult with {len(response.generations) if response.generations else 0} generations")
                
                # PRIORITY 1: Check usage_metadata in generations (Google Gemini format)
                if response.generations:
                    for i, generation in enumerate(response.generations):
                        logger.info(f"🔧 Processing generation {i}: {type(generation)}")
                        logger.info(f"🔧 Generation attributes: {dir(generation)}")
                        
                        # Check if generation has message attribute
                        if hasattr(generation, 'message'):
                            message = generation.message
                            logger.info(f"🔧 Message type: {type(message)}")
                            logger.info(f"🔧 Message attributes: {dir(message)}")
                            
                            # Check for usage_metadata in the message
                            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                                usage = message.usage_metadata
                                logger.info(f"🔧 Found usage_metadata in message: {usage}")
                                input_tokens += usage.get('input_tokens', 0)
                                output_tokens += usage.get('output_tokens', 0)
                                total_tokens += usage.get('total_tokens', 0)
                                reasoning_tokens += usage.get('reasoning_tokens', 0)
                                logger.info(f"🔧 Extracted tokens from message usage_metadata: input={input_tokens}, output={output_tokens}, total={total_tokens}")
                            else:
                                logger.info(f"🔧 No usage_metadata found in message")
                                
                                # Check if message has additional_metadata
                                if hasattr(message, 'additional_metadata') and message.additional_metadata:
                                    additional = message.additional_metadata
                                    logger.info(f"🔧 Found additional_metadata: {additional}")
                                    if 'usage_metadata' in additional:
                                        usage = additional['usage_metadata']
                                        input_tokens += usage.get('input_tokens', 0)
                                        output_tokens += usage.get('output_tokens', 0)
                                        total_tokens += usage.get('total_tokens', 0)
                                        reasoning_tokens += usage.get('reasoning_tokens', 0)
                                        logger.info(f"🔧 Extracted tokens from additional_metadata: input={input_tokens}, output={output_tokens}, total={total_tokens}")

                                # Try to extract reasoning text from common locations
                                try:
                                    # OpenAI/Anthropic often include provider-specific fields on message
                                    if hasattr(message, 'additional_kwargs') and isinstance(message.additional_kwargs, dict):
                                        if 'reasoning' in message.additional_kwargs:
                                            reasoning_text = message.additional_kwargs.get('reasoning')
                                        elif 'thinking' in message.additional_kwargs:
                                            reasoning_text = message.additional_kwargs.get('thinking')
                                    # LangChain may surface reasoning in response_metadata
                                    if hasattr(message, 'response_metadata') and isinstance(message.response_metadata, dict):
                                        if not reasoning_text:
                                            reasoning_text = message.response_metadata.get('reasoning') or message.response_metadata.get('thinking')
                                except Exception as rex:
                                    logger.debug(f"Reasoning extraction from message failed: {rex}")
                        else:
                            logger.info(f"🔧 Generation has no message attribute")

                # PRIORITY 2: Check llm_output token_usage (OpenAI format)
                if total_tokens == 0 and response.llm_output:
                    logger.info(f"🔧 Checking llm_output: {response.llm_output}")
                    if 'token_usage' in response.llm_output:
                        token_usage = response.llm_output['token_usage']
                        input_tokens = token_usage.get('prompt_tokens', 0)
                        output_tokens = token_usage.get('completion_tokens', 0)
                        total_tokens = token_usage.get('total_tokens', 0)
                        reasoning_tokens = token_usage.get('reasoning_tokens', reasoning_tokens)
                        logger.info(f"🔧 Found tokens in llm_output token_usage: input={input_tokens}, output={output_tokens}, total={total_tokens}")

                # PRIORITY 3: Try alternative extraction methods
                if total_tokens == 0 and response.llm_output:
                    for key in ['token_usage', 'usage_metadata', 'usage']:
                        if key in response.llm_output:
                            usage_data = response.llm_output[key]
                            if isinstance(usage_data, dict):
                                input_tokens = usage_data.get('prompt_tokens', usage_data.get('input_tokens', 0))
                                output_tokens = usage_data.get('completion_tokens', usage_data.get('output_tokens', 0))
                                total_tokens = usage_data.get('total_tokens', input_tokens + output_tokens)
                                reasoning_tokens = usage_data.get('reasoning_tokens', reasoning_tokens)
                                if total_tokens > 0:
                                    logger.info(f"🔧 Found tokens in llm_output.{key}: input={input_tokens}, output={output_tokens}, total={total_tokens}")
                                    break
                
                # PRIORITY 4: Check response metadata
                if total_tokens == 0 and hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if isinstance(metadata, dict):
                        input_tokens = metadata.get('prompt_tokens', metadata.get('input_tokens', 0))
                        output_tokens = metadata.get('completion_tokens', metadata.get('output_tokens', 0))
                        total_tokens = metadata.get('total_tokens', input_tokens + output_tokens)
                        reasoning_tokens = metadata.get('reasoning_tokens', reasoning_tokens)
                        # Also try to capture reasoning text if present
                        if not reasoning_text:
                            reasoning_text = metadata.get('reasoning') or metadata.get('thinking')
                        if total_tokens > 0:
                            logger.info(f"🔧 Found tokens in response_metadata: input={input_tokens}, output={output_tokens}, total={total_tokens}")

            logger.info(f"🔧 Extracted tokens - input: {input_tokens}, output: {output_tokens}, total: {total_tokens}")
            
            # Calculate cost
            cost = 0.0
            if input_tokens > 0 or output_tokens > 0:
                cost = calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=span_data['model_name'],
                    provider=span_data['model_provider']
                )
                logger.info(f"🔧 Calculated cost: {cost}")
            else:
                logger.warning(f"🔧 No tokens found - input: {input_tokens}, output: {output_tokens}")
            
            # Update span data
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {'response': str(response)},
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'reasoning_tokens': reasoning_tokens,
                'reasoning': reasoning_text,
                'cost': cost
            })
            
            logger.info(f"🔧 Final span data: {span_data}")
            
            # FIXED: Associate tokens with parent agent spans
            self._associate_tokens_with_parent_agent(run_id, input_tokens, output_tokens, total_tokens, cost)
            
            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.debug(f"Sent LangGraph LLM span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for LLM span")
            
            # Clean up
            del self._active_traces[run_id]
            logger.debug(f"LangGraph LLM end: {run_id}")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph LLM end callback: {e}")

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle LLM error for LangGraph applications."""
        try:
            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                span_data = trace_data['span_data']
                start_time = trace_data['start_time']
                
                # Calculate duration
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Update span data for error
                span_data.update({
                    'end_time': end_time.isoformat(),
                    'status': 'failed',
                    'duration': duration_ms,
                    'error': str(error)
                })
                
                # Send error span to trace collector
                if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                    self.sdk._trace_collector.process_span(span_data)
                
                # Clean up
                del self._active_traces[run_id]
            
            logger.debug(f"LangGraph LLM error: {run_id} - {error}")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph LLM error callback: {e}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool start for LangGraph applications."""
        try:
            # Extract tool name
            tool_name = "unknown"
            if isinstance(serialized, dict):
                tool_name = serialized.get("name", "unknown")
            
            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.debug(f"Initialized session trace ID in tool start: {self._session_trace_id}")
            
            # Check for session timeout
            self._check_session_timeout()
            
            # Find the parent agent for this tool call
            parent_agent_name = "unknown"
            if parent_run_id and parent_run_id in self._active_traces:
                parent_trace_data = self._active_traces[parent_run_id]
                parent_span_data = parent_trace_data.get('span_data', {})
                parent_agent_name = parent_span_data.get('agent_name', 'unknown')
            else:
                # Fallback: Find the most recent LangGraph agent
                most_recent_agent = None
                most_recent_time = None
                
                for trace_id, trace_data in self._active_traces.items():
                    span_data = trace_data.get('span_data', {})
                    if span_data.get('component_type') == 'agent':
                        start_time = trace_data.get('start_time')
                        if start_time and (most_recent_time is None or start_time > most_recent_time):
                            most_recent_agent = trace_data
                            most_recent_time = start_time
                
                if most_recent_agent:
                    parent_span_data = most_recent_agent['span_data']
                    parent_agent_name = parent_span_data.get('agent_name', 'unknown')
            
            logger.info(f"🔧 TOOL CALL - Tool: {tool_name}, Parent Agent: {parent_agent_name}")
            
            # Create tool span data with LangGraph trace ID and metadata
            tool_span_data = {
                'trace_id': self._session_trace_id,  # Use LangGraph trace ID
                'agent_name': f"tool:{tool_name}",
                'function_name': f"tool:{tool_name}",
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {'input_str': input_str, 'serialized': serialized},
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'tool',
                'tool_name': tool_name,
                'parent_agent_name': parent_agent_name,
                'parent_run_id': parent_run_id,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0
            }
            
            # Add session information if available
            if self.session:
                tool_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'tool_call',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })
            
            # Store span data for on_tool_end
            self._active_traces[run_id] = {
                'span_data': tool_span_data,
                'start_time': datetime.utcnow(),
                'tool_name': tool_name,
                'parent_run_id': parent_run_id,
                'parent_agent_name': parent_agent_name
            }
            
            logger.debug(f"LangGraph tool start: {run_id} (parent: {parent_run_id})")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph tool start callback: {e}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle tool end for LangGraph applications."""
        try:
            if run_id not in self._active_traces:
                logger.warning(f"No active trace found for tool run_id: {run_id}")
                return
            
            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update span data
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {'output': str(output)}
            })
            
            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.debug(f"Sent LangGraph tool span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for tool span")
            
            # Clean up
            del self._active_traces[run_id]
            logger.debug(f"LangGraph tool end: {run_id}")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph tool end callback: {e}")

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle tool error for LangGraph applications."""
        try:
            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                span_data = trace_data['span_data']
                start_time = trace_data['start_time']
                
                # Calculate duration
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Update span data for error
                span_data.update({
                    'end_time': end_time.isoformat(),
                    'status': 'failed',
                    'duration': duration_ms,
                    'error': str(error)
                })
                
                # Send error span to trace collector
                if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                    self.sdk._trace_collector.process_span(span_data)
                
                # Clean up
                del self._active_traces[run_id]
            
            logger.debug(f"LangGraph tool error: {run_id} - {error}")
            
        except Exception as e:
            logger.warning(f"Error in LangGraph tool error callback: {e}")

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start for LangGraph applications."""
        try:
            # Extract retriever name
            retriever_name = "unknown"
            if isinstance(serialized, dict):
                retriever_name = serialized.get("name", retriever_name)

            logger.info(f"🔧 LANGGRAPH RETRIEVER START - Retriever: {retriever_name}, Run ID: {run_id}, Parent: {parent_run_id}")
            logger.info(f"🔧 LANGGRAPH RETRIEVER START - Query: {query}")
            logger.info(f"🔧 LANGGRAPH RETRIEVER START - Tags: {tags}")
            logger.info(f"🔧 LANGGRAPH RETRIEVER START - Metadata: {metadata}")

            # Initialize session trace ID if not set
            if self._session_trace_id is None:
                self._session_trace_id = str(uuid.uuid4())
                logger.debug(f"Initialized session trace ID for retriever: {self._session_trace_id}")

            # Check for session timeout
            self._check_session_timeout()

            # Find the parent agent for this retriever call
            parent_agent_name = "unknown"
            if parent_run_id and parent_run_id in self._active_traces:
                parent_trace_data = self._active_traces[parent_run_id]
                parent_span_data = parent_trace_data.get('span_data', {})
                parent_agent_name = parent_span_data.get('agent_name', 'unknown')
            else:
                # Fallback: Find the most recent LangGraph agent
                most_recent_agent = None
                most_recent_time = None

                for trace_id, trace_data in self._active_traces.items():
                    span_data = trace_data.get('span_data', {})
                    if span_data.get('component_type') == 'agent':
                        start_time = trace_data.get('start_time')
                        if start_time and (most_recent_time is None or start_time > most_recent_time):
                            most_recent_agent = trace_data
                            most_recent_time = start_time

                if most_recent_agent:
                    parent_span_data = most_recent_agent['span_data']
                    parent_agent_name = parent_span_data.get('agent_name', 'unknown')

            # Create retriever span data
            retriever_span_data = {
                'trace_id': self._session_trace_id,
                'agent_name': f"retriever:{retriever_name}",
                'function_name': f"retriever:{retriever_name}",
                'start_time': datetime.utcnow().isoformat(),
                'status': 'running',
                'duration': 0,
                'inputs': {
                    'query': query,
                    'serialized': serialized,
                    'metadata': metadata or {}
                },
                'outputs': {},
                'framework': 'langgraph',
                'component_type': 'retriever',
                'retriever_name': retriever_name,
                'parent_agent_name': parent_agent_name,
                'parent_run_id': parent_run_id,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'retriever_metadata': {
                    'retriever_start_time': datetime.utcnow().isoformat(),
                    'retriever_execution_context': {
                        'run_id': run_id,
                        'parent_run_id': parent_run_id,
                        'tags': tags or [],
                        'metadata': metadata or {}
                    }
                }
            }

            # Add session information if available
            if self.session:
                retriever_span_data.update({
                    'session_id': self.session.session_id,
                    'session_type': self.session.session_type,
                    'chat_session_id': self.session.session_id,
                    'message_type': 'retriever_call',
                    'message_sequence': self._message_sequence,
                    'agent_orchestration_id': self._current_orchestration_id
                })

            # Store span data for on_retriever_end
            self._active_traces[run_id] = {
                'span_data': retriever_span_data,
                'start_time': datetime.utcnow(),
                'retriever_name': retriever_name,
                'parent_run_id': parent_run_id,
                'parent_agent_name': parent_agent_name
            }

            logger.debug(f"LangGraph retriever start: {run_id} (parent: {parent_run_id})")

        except Exception as e:
            logger.warning(f"Error in LangGraph retriever start callback: {e}")

    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle retriever end for LangGraph applications."""
        try:
            logger.info(f"🔧 LANGGRAPH RETRIEVER END - Run ID: {run_id}")
            logger.info(f"🔧 LANGGRAPH RETRIEVER END - Documents count: {len(documents) if documents else 0}")

            if run_id not in self._active_traces:
                logger.warning(f"No active trace found for retriever run_id: {run_id}")
                return

            trace_data = self._active_traces[run_id]
            span_data = trace_data['span_data']
            start_time = trace_data['start_time']

            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Extract document information
            doc_info = self._extract_retriever_documents(documents)
            logger.info(f"🔧 LANGGRAPH RETRIEVER END - Extracted document info: {doc_info}")

            # Update span data
            span_data.update({
                'end_time': end_time.isoformat(),
                'status': 'completed',
                'duration': duration_ms,
                'outputs': {
                    'documents': doc_info.get('documents', []),
                    'document_count': doc_info.get('document_count', 0),
                    'total_characters': doc_info.get('total_characters', 0)
                },
                'retriever_result_count': doc_info.get('document_count', 0),
                'retriever_metadata': {
                    **span_data.get('retriever_metadata', {}),
                    'retriever_end_time': end_time.isoformat(),
                    'total_execution_time_ms': duration_ms,
                    'execution_status': 'completed'
                }
            })

            # Send span to trace collector
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                self.sdk._trace_collector.process_span(span_data)
                logger.debug(f"Sent LangGraph retriever span: {span_data['agent_name']} for session trace: {self._session_trace_id}")
            else:
                logger.warning("No trace collector available for LangGraph retriever span")

            # Clean up
            del self._active_traces[run_id]
            logger.debug(f"LangGraph retriever end: {run_id}")

        except Exception as e:
            logger.warning(f"Error in LangGraph retriever end callback: {e}")

    def on_retriever_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle retriever error for LangGraph applications."""
        try:
            logger.error(f"🔧 LANGGRAPH RETRIEVER ERROR - Run ID: {run_id}")
            logger.error(f"🔧 LANGGRAPH RETRIEVER ERROR - Error: {error}")

            if run_id in self._active_traces:
                trace_data = self._active_traces[run_id]
                span_data = trace_data['span_data']
                start_time = trace_data['start_time']

                # Calculate duration
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Update span data for error
                span_data.update({
                    'end_time': end_time.isoformat(),
                    'status': 'failed',
                    'duration': duration_ms,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'retriever_metadata': {
                        **span_data.get('retriever_metadata', {}),
                        'retriever_end_time': end_time.isoformat(),
                        'total_execution_time_ms': duration_ms,
                        'execution_status': 'failed',
                        'failure_reason': str(error)
                    }
                })

                # Send error span to trace collector
                if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                    self.sdk._trace_collector.process_span(span_data)
                    logger.error(f"Sent error span for retriever run_id: {run_id}")

                # Clean up
                del self._active_traces[run_id]

            logger.error(f"LangGraph retriever error: {run_id} - {error}")

        except Exception as e:
            logger.warning(f"Error in LangGraph retriever error callback: {e}")

    def _extract_retriever_documents(self, documents: List[Any]) -> Dict[str, Any]:
        """Extract information from retrieved documents."""
        doc_info = {
            'documents': [],
            'document_count': 0,
            'total_characters': 0,
            'document_types': set(),
            'has_metadata': False,
            'has_content': False
        }

        try:
            if not documents:
                return doc_info

            doc_info['document_count'] = len(documents)

            for i, doc in enumerate(documents):
                doc_summary = {}

                # Extract basic document information
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    doc_summary['content_length'] = len(content)
                    doc_summary['has_content'] = True
                    doc_info['has_content'] = True
                    doc_info['total_characters'] += len(content)
                else:
                    doc_summary['content_length'] = 0

                # Extract metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_summary['metadata'] = doc.metadata
                    doc_summary['has_metadata'] = True
                    doc_info['has_metadata'] = True

                    # Track document types
                    doc_type = doc.metadata.get('source', doc.metadata.get('type', 'unknown'))
                    doc_info['document_types'].add(doc_type)

                doc_info['documents'].append(doc_summary)

            # Convert set to list for JSON serialization
            doc_info['document_types'] = list(doc_info['document_types'])

        except Exception as e:
            logger.warning(f"Error extracting retriever documents: {e}")

        return doc_info

    def _send_agent_span(self, node_info: Dict[str, Any], duration: float, is_error: bool = False) -> None:
        """Send individual agent span to the unified trace collector."""
        try:
            if self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
                # Initialize session trace ID if not set
                if self._session_trace_id is None:
                    self._session_trace_id = str(uuid.uuid4())
                    logger.debug(f"Initialized session trace ID in agent span: {self._session_trace_id}")

                # Enhanced logging for debugging
                logger.debug(f"🔍 Creating agent span for: {node_info['agent_name']}")
                logger.debug(f"🔍 Node info keys: {list(node_info.keys())}")
                logger.debug(f"🔍 Duration: {duration}")
                logger.debug(f"🔍 Start time: {node_info['start_time']}")
                logger.debug(f"🔍 End time: {node_info['end_time']}")
                logger.debug(f"🔍 Inputs: {node_info.get('inputs', {})}")
                logger.debug(f"🔍 Outputs: {node_info.get('outputs', {})}")

                # Create agent span data (not a complete trace)
                agent_span_data = {
                    'trace_id': self._session_trace_id,  # Use session trace ID
                    'agent_name': node_info['agent_name'],
                    'function_name': node_info['agent_name'],
                    'start_time': node_info['start_time'].isoformat(),
                    'end_time': node_info['end_time'].isoformat(),
                    'status': 'failed' if is_error else 'completed',
                    'duration': duration,
                    'inputs': node_info['inputs'],
                    'outputs': node_info.get('outputs', {}),
                    'error': node_info.get('error'),
                    'framework': 'langgraph',
                    'component_type': 'agent'
                }
                
                logger.debug(f"🔍 Created agent span data: {agent_span_data}")

                # Add session information if available
                if self.session:
                    agent_span_data.update({
                        'session_id': self.session.session_id,
                        'session_type': self.session.session_type,
                        'chat_session_id': self.session.session_id,
                        'message_type': 'agent_response',
                        'message_sequence': self._message_sequence,
                        'agent_orchestration_id': self._current_orchestration_id,
                        'session_metadata': self.session.to_dict()
                    })

                # Send span to trace collector (will be collected for hierarchical processing)
                self.sdk._trace_collector.process_span(agent_span_data)
                logger.debug(f"Sent LangGraph agent span: {node_info['agent_name']} for session trace: {self._session_trace_id}")
                
        except Exception as e:
            logger.error(f"Failed to send LangGraph agent span: {e}")

    def finalize_langgraph_traces(self):
        """Finalize LangGraph session by triggering hierarchical trace processing."""
        logger.debug("Finalizing LangGraph session traces...")

        # If we have a session trace ID, finalize the hierarchical trace
        if self._session_trace_id and self.sdk and hasattr(self.sdk, '_trace_collector') and self.sdk._trace_collector:
            try:
                logger.debug(f"Finalizing hierarchical trace for session: {self._session_trace_id}")
                self.sdk._trace_collector.finalize_trace(self._session_trace_id)
                logger.debug(f"Successfully finalized hierarchical trace: {self._session_trace_id}")
            except Exception as e:
                logger.error(f"Failed to finalize hierarchical trace {self._session_trace_id}: {e}")
        else:
            logger.debug("No session trace ID available for hierarchical finalization")

        # Clear active traces and reset state
        self._active_traces.clear()
        self._active_nodes.clear()
        self._trace_finalization_pending = False
        self._current_orchestration_id = None
        
        # Note: We don't reset _session_trace_id here as it should persist for the session
        # It will be reset when a new session is created

        logger.debug("LangGraph session traces finalized")
    
    def shutdown_session_monitoring(self) -> None:
        """Shutdown automatic session monitoring."""
        if self._session_monitoring_active:
            logger.debug(f"Shutting down session monitoring for session {self.session.session_id if self.session else 'unknown'}")
            self._session_monitoring_active = False
            
            # Unregister timeout callback
            if self.session:
                self.session._unregister_timeout_callback(self._on_session_timeout)

    def __del__(self):
        """Destructor to ensure session monitoring is cleaned up."""
        self.shutdown_session_monitoring()

    def shutdown(self) -> None:
        """Shutdown handler and finalize any pending traces."""
        logger.info(f"Shutting down LangGraph handler for session {self.session.session_id if self.session else 'unknown'}")
        
        # Finalize any pending traces before shutdown
        if self._session_trace_id:
            logger.info(f"Finalizing pending traces before shutdown: {self._session_trace_id}")
            self.finalize_langgraph_traces()
        
        # Shutdown session monitoring
        self.shutdown_session_monitoring()
        
        # Unregister from global registry
        _active_handlers.discard(self)
        
        logger.info("LangGraph handler shutdown complete")


@asynccontextmanager
async def langgraph_trace_context(handler: VaqueroLangGraphHandler):
    """
    Context manager for LangGraph tracing that ensures proper cleanup.
    
    This context manager ensures that traces are finalized and sent
    before the context exits, addressing the timing issues with
    LangGraph's execution flow.
    """
    try:
        yield handler
    finally:
        # Ensure traces are finalized before context exit
        handler.finalize_langgraph_traces()
        
        # Give a moment for traces to be sent
        await asyncio.sleep(0.1)


def get_vaquero_langgraph_handler(**kwargs) -> VaqueroLangGraphHandler:
    """
    Get a Vaquero handler specifically configured for LangGraph.
    
    Args:
        **kwargs: Additional arguments for the handler
        
    Returns:
        VaqueroLangGraphHandler: Configured handler for LangGraph
    """
    return VaqueroLangGraphHandler(**kwargs)


def create_langgraph_config(handler: VaqueroLangGraphHandler) -> Dict[str, Any]:
    """
    Create a configuration dict for LangGraph with the Vaquero handler.
    
    Args:
        handler: VaqueroLangGraphHandler instance
        
    Returns:
        Dict containing the configuration for LangGraph
    """
    return {
        'callbacks': [handler],
        'configurable': {
            'vaquero_handler': handler
        }
    }


def shutdown_all_langgraph_handlers() -> None:
    """Shutdown all active LangGraph handlers and finalize their traces."""
    logger.info(f"Shutting down {len(_active_handlers)} active LangGraph handlers")
    
    for handler in list(_active_handlers):  # Create a copy to avoid modification during iteration
        try:
            handler.shutdown()
        except Exception as e:
            logger.error(f"Failed to shutdown LangGraph handler: {e}")
    
    logger.info("All LangGraph handlers shutdown complete")