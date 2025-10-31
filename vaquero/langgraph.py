"""
LangGraph integration for Vaquero SDK.

This module provides a thin callback handler that emits normalized spans
for LangGraph applications, delegating all aggregation to the processor.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid

from .sdk import VaqueroSDK, get_global_instance
from .chat_session import ChatSession

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # Fallback for type hints


class VaqueroLangGraphHandler(BaseCallbackHandler):
    """
    Thin Vaquero handler for LangGraph applications.
    
    This handler emits normalized spans for LangGraph workflows,
    delegating all aggregation and hierarchy building to the processor.
    
    Key Features:
    - Minimal state: only tracks active runs with basic metadata
    - Emits normalized spans with proper parentage
    - Extracts names from metadata/tags without complex logic
    - Preserves raw inputs/outputs for processor analysis
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

        # Minimal run registry: run_id -> {span_id, parent_span_id, start_time, name, component_type}
        self._runs: Dict[str, Dict[str, Any]] = {}

        # Session context for chat workflows
        self._session_context = {
            'chat_session_id': session.session_id if session else None,
            'message_sequence': 0
        }

        # Stable trace identifier: use session id when available, else per-handler UUID
        self._trace_id = session.session_id if session else str(uuid.uuid4())

    def _extract_component_name(self, serialized: Optional[Dict], metadata: Optional[Dict], tags: Optional[List[str]]) -> str:
        """Extract component name from metadata/tags for LangGraph spans."""
        # Continue to check tags even if metadata is None
        
        # For LangGraph nodes, extract from metadata
        if metadata and 'node' in metadata:
            return metadata['node']

        # For agents, try to extract from langgraph_path or triggers
        if metadata and 'langgraph_path' in metadata:
            path = metadata['langgraph_path']
            if isinstance(path, tuple) and len(path) > 1:
                return str(path[1])

        if metadata and 'langgraph_triggers' in metadata:
            triggers = metadata['langgraph_triggers']
            if isinstance(triggers, tuple) and len(triggers) > 0:
                trigger = triggers[0]
                if ':' in trigger:
                    parts = trigger.split(':')
                    if len(parts) >= 3:
                        return parts[2]  # For "branch:to:agent", take "agent"

        # Fallback to tags or serialized name
        if tags:
            for tag in tags:
                if tag in ['explainer', 'developer', 'analogy_creator', 'vulnerability_expert', 'agent']:
                    return tag

        if serialized and 'name' in serialized:
            return serialized['name']

        return "unknown"
    
    def _emit_span(self, span_data: Dict[str, Any]) -> None:
        """Emit a normalized span to the trace collector."""
        if self.sdk and getattr(self.sdk, '_trace_collector', None):
            self.sdk._trace_collector.process_span(span_data)
            logger.debug(f"Emitted LangGraph span: {span_data.get('component_type')} {span_data.get('name')}")
        else:
            logger.warning("No trace collector available for LangGraph span")

    # ===== MINIMAL CALLBACK IMPLEMENTATIONS =====

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
        logger.debug(f"LangGraph graph start: {run_id}")

        # Track the graph run
        span_id = str(uuid.uuid4())
        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': None,  # Graph is root
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'langgraph_workflow') if serialized else 'langgraph_workflow',
            'component_type': 'graph'
        }

        # Emit graph span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': span_id,
            'parent_span_id': None,
            'component_type': 'graph',
            'name': self._runs[run_id]['name'],
            'start_time': self._runs[run_id]['start_time'].isoformat(),
            'status': 'running',
            'inputs': inputs,
            'outputs': {},
            'metadata': metadata or {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)

    def on_graph_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle graph end for LangGraph applications."""
        if run_id not in self._runs:
            logger.warning(f"No graph run found for {run_id}")
            return

        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Emit completed graph span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'graph',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {},  # Inputs were sent in start
            'outputs': outputs,
            'metadata': {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]

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
        logger.debug(f"LangGraph node start: {run_id}")

        # Extract node name
        name = self._extract_component_name(serialized, metadata, tags)

        # Track the node run
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': name,
            'component_type': 'node'
        }

        # Emit node span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'component_type': 'node',
            'name': name,
            'start_time': self._runs[run_id]['start_time'].isoformat(),
            'status': 'running',
            'inputs': inputs,
            'outputs': {},
            'metadata': metadata or {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)

    def on_node_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle node end for LangGraph applications."""
        if run_id not in self._runs:
            logger.warning(f"No node run found for {run_id}")
            return

        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Emit completed node span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'node',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {},  # Inputs were sent in start
            'outputs': outputs,
            'metadata': {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]

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
        logger.debug(f"LangGraph LLM start: {run_id}")

        # Track the LLM run
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'llm') if serialized else 'llm',
            'component_type': 'llm'
        }

        # Emit LLM span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'component_type': 'llm',
            'name': self._runs[run_id]['name'],
            'start_time': self._runs[run_id]['start_time'].isoformat(),
            'status': 'running',
            'inputs': {'prompts': prompts},
            'outputs': {},
            'metadata': metadata or {}
        }

        # Extract system prompt if present
        if prompts and len(prompts) > 0:
            span_data['system_prompt'] = prompts[0]

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle new LLM token (optional for reasoning capture)."""
        # For now, we don't capture individual tokens to keep the handler thin
        # Reasoning can be captured from final outputs
        pass

    def on_llm_end(
        self,
        response: Any,  # Use Any to avoid LLMResult import issues
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end for LangGraph applications."""
        if run_id not in self._runs:
            logger.warning(f"No LLM run found for {run_id}")
            return
            
        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Extract token usage and other LLM metadata
        llm_metadata = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            llm_metadata.update(response.llm_output)

        # Extract token counts if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            llm_metadata.update({
                'input_tokens': getattr(usage, 'input_tokens', 0),
                'output_tokens': getattr(usage, 'output_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            })

        # Emit completed LLM span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'llm',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {},  # Inputs were sent in start
            'outputs': {'response': getattr(response, 'generations', []) if hasattr(response, 'generations') else []},
            'metadata': llm_metadata
        }

        # Add token counts to span data
        if 'input_tokens' in llm_metadata:
            span_data['input_tokens'] = llm_metadata['input_tokens']
        if 'output_tokens' in llm_metadata:
            span_data['output_tokens'] = llm_metadata['output_tokens']
        if 'total_tokens' in llm_metadata:
            span_data['total_tokens'] = llm_metadata['total_tokens']

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]

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
        logger.debug(f"LangGraph tool start: {run_id}")

        # Track the tool run
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'tool') if serialized else 'tool',
            'component_type': 'tool'
        }

        # Emit tool span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'component_type': 'tool',
            'name': self._runs[run_id]['name'],
            'start_time': self._runs[run_id]['start_time'].isoformat(),
            'status': 'running',
            'inputs': {'input_str': input_str},
            'outputs': {},
            'metadata': metadata or {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle tool end for LangGraph applications."""
        if run_id not in self._runs:
            logger.warning(f"No tool run found for {run_id}")
            return
            
        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Emit completed tool span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'tool',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {},  # Inputs were sent in start
            'outputs': {'output': output},
            'metadata': {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]

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
        logger.debug(f"LangGraph chain start: {run_id}")

        # Track the chain run
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'chain') if serialized else 'chain',
            'component_type': 'node'
        }

        # Emit chain span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'component_type': 'node',
            'name': self._runs[run_id]['name'],
            'start_time': self._runs[run_id]['start_time'].isoformat(),
            'status': 'running',
            'inputs': inputs,
            'outputs': {},
            'metadata': metadata or {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle chain end for LangGraph applications."""
        if run_id not in self._runs:
            logger.warning(f"No chain run found for {run_id}")
            return

        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Emit completed chain span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'node',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {},  # Inputs were sent in start
            'outputs': outputs,
            'metadata': {}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]

    # ===== ERROR HANDLERS =====

    def on_graph_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle graph error."""
        self._handle_error(run_id, error, 'graph')

    def on_node_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle node error."""
        self._handle_error(run_id, error, 'node')

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle LLM error."""
        self._handle_error(run_id, error, 'llm')

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle tool error."""
        self._handle_error(run_id, error, 'tool')

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Handle chain error."""
        self._handle_error(run_id, error, 'chain')

    def _handle_error(self, run_id: str, error: Exception, component_type: str) -> None:
        """Handle errors for any component type."""
        if run_id not in self._runs:
            logger.warning(f"No {component_type} run found for {run_id}")
            return

        run_info = self._runs[run_id]
        end_time = datetime.utcnow()

        # Emit error span
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': component_type,
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'error',
            'inputs': {},
            'outputs': {'error': str(error)},
            'metadata': {'error_type': type(error).__name__}
        }

        if self._session_context['chat_session_id']:
            span_data['chat_session_id'] = self._session_context['chat_session_id']

        self._emit_span(span_data)
        del self._runs[run_id]
