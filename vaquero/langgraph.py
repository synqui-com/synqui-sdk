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
        logger.info(f"🔍 NAME EXTRACTION: Starting extraction")
        logger.info(f"🔍 NAME EXTRACTION: serialized keys = {list(serialized.keys()) if serialized else None}")
        logger.info(f"🔍 NAME EXTRACTION: serialized.get('name') = {serialized.get('name') if serialized else None}")
        logger.info(f"🔍 NAME EXTRACTION: metadata keys = {list(metadata.keys()) if metadata else None}")
        logger.info(f"🔍 NAME EXTRACTION: metadata = {metadata}")
        logger.info(f"🔍 NAME EXTRACTION: tags = {tags}")
        
        # Priority 1: Direct node name from metadata
        if metadata and 'node' in metadata:
            logger.info(f"🔍 NAME EXTRACTION: Found via Priority 1 (metadata['node']) = {metadata['node']}")
            return metadata['node']

        # Priority 2: LangGraph path extraction
        if metadata and 'langgraph_path' in metadata:
            path = metadata['langgraph_path']
            logger.info(f"🔍 NAME EXTRACTION: Checking Priority 2 (langgraph_path) = {path}, type = {type(path)}")
            if isinstance(path, tuple) and len(path) > 1:
                result = str(path[1])
                logger.info(f"🔍 NAME EXTRACTION: Found via Priority 2 = {result}")
                return result

        # Priority 3: LangGraph triggers extraction
        if metadata and 'langgraph_triggers' in metadata:
            triggers = metadata['langgraph_triggers']
            logger.info(f"🔍 NAME EXTRACTION: Checking Priority 3 (langgraph_triggers) = {triggers}, type = {type(triggers)}")
            if isinstance(triggers, tuple) and len(triggers) > 0:
                trigger = triggers[0]
                if ':' in trigger:
                    parts = trigger.split(':')
                    if len(parts) >= 3:
                        result = parts[2]  # "branch:to:agent" -> "agent"
                        logger.info(f"🔍 NAME EXTRACTION: Found via Priority 3 = {result}")
                        return result

        # Priority 4: Tag-based extraction for known agent types
        if tags:
            logger.info(f"🔍 NAME EXTRACTION: Checking Priority 4 (tags) = {tags}")
            for tag in tags:
                if tag in ['explainer', 'developer', 'analogy_creator', 'vulnerability_expert']:
                    logger.info(f"🔍 NAME EXTRACTION: Found via Priority 4 (known tag) = {tag}")
                    return tag
                # Also check for general 'agent' tag as fallback
                if tag == 'agent':
                    logger.info(f"🔍 NAME EXTRACTION: Found via Priority 4 (generic agent tag) = {tag}")
                    return tag

        # Priority 5: Serialized name (fallback) - but avoid using "chain"
        if serialized and 'name' in serialized:
            name = serialized['name']
            logger.info(f"🔍 NAME EXTRACTION: Checking Priority 5 (serialized['name']) = {name}")
            if name and name != 'chain':  # Avoid using "chain" as agent name
                logger.info(f"🔍 NAME EXTRACTION: Found via Priority 5 = {name}")
                return name

        # Priority 6: Additional metadata fields that might contain agent names
        if metadata:
            logger.info(f"🔍 NAME EXTRACTION: Checking Priority 6 (metadata fields)")
            # Check for any field that might contain the agent name
            for key in ['agent_name', 'node_name', 'component_name']:
                if key in metadata and metadata[key]:
                    result = str(metadata[key])
                    logger.info(f"🔍 NAME EXTRACTION: Found via Priority 6 (metadata['{key}']) = {result}")
                    return result

        # Return "unknown_node" for nodes to distinguish from other component types
        logger.info(f"🔍 NAME EXTRACTION: No match found, returning 'unknown_node'")
        return "unknown_node"
    
    def _is_node_execution(
        self,
        metadata: Optional[Dict[str, Any]],
        parent_run_id: Optional[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if this chain callback represents a logical agent node execution.
        
        A chain callback represents a node execution if:
        1. metadata contains 'langgraph_node' with a logical agent name (not 'agent')
        2. The parent (if exists) does NOT have the same langgraph_node (not nested)
        
        Args:
            metadata: Callback metadata dictionary
            parent_run_id: ID of parent run if exists
        
        Returns:
            (is_node, node_name): (True, 'explainer') if it's a node, (False, None) otherwise
        """
        if not metadata:
            return (False, None)
        
        langgraph_node = metadata.get('langgraph_node')
        
        # Must have langgraph_node and it must not be 'agent'
        if not langgraph_node or langgraph_node == 'agent':
            return (False, None)
        
        # Check if parent has the same langgraph_node (means we're nested within the node)
        is_nested = False
        if parent_run_id and parent_run_id in self._runs:
            parent_metadata = self._runs[parent_run_id].get('metadata', {})
            parent_node = parent_metadata.get('langgraph_node')
            if parent_node == langgraph_node:
                # Parent is the same node → this is an internal chain within the node
                logger.info(f"🔍 NODE DETECTION: Parent has same langgraph_node='{langgraph_node}' → nested chain")
                is_nested = True
        
        if is_nested:
            return (False, None)  # Internal chain, not the node itself
        
        # This is a top-level node execution
        logger.info(f"🔍 NODE DETECTION: Detected as NODE execution: {langgraph_node}")
        return (True, langgraph_node)
    
    def _emit_span(self, span_data: Dict[str, Any]) -> None:
        """Emit a normalized span to the trace collector."""
        # Always add session context if available
        if self._session_context.get('chat_session_id'):
            span_data['chat_session_id'] = self._session_context['chat_session_id']
        span_data['message_sequence'] = self._session_context.get('message_sequence', 0)

        logger.info(f"📤 EMIT SPAN: component_type={span_data.get('component_type')}, name={span_data.get('name')}, "
                   f"span_id={span_data.get('span_id')}, message_sequence={span_data.get('message_sequence')}")

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

        # Increment message sequence per graph invocation (user turn)
        self._session_context['message_sequence'] = (self._session_context.get('message_sequence') or 0) + 1

        # Track the graph run - store data for end callback
        span_id = str(uuid.uuid4())
        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': None,  # Graph is root
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'langgraph_workflow') if serialized else 'langgraph_workflow',
            'component_type': 'graph',
            'inputs': inputs,
            'metadata': metadata or {}
        }

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

        # Emit completed graph span with stored inputs
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'graph',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': run_info.get('inputs', {}),
            'outputs': outputs,
            'metadata': run_info.get('metadata', {})
        }

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
        logger.info(f"🚀 NODE START CALLBACK: run_id={run_id}, parent_run_id={parent_run_id}")
        logger.info(f"🚀 NODE START: serialized keys={list(serialized.keys()) if serialized else None}")
        logger.info(f"🚀 NODE START: serialized={serialized}")
        logger.info(f"🚀 NODE START: metadata={metadata}")
        logger.info(f"🚀 NODE START: tags={tags}")

        # Extract node name
        name = self._extract_component_name(serialized, metadata, tags)
        logger.info(f"🚀 NODE START: Extracted name = {name}")

        # Track the node run - store data for end callback
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': name,
            'component_type': 'node',
            'inputs': inputs,
            'metadata': metadata or {}
        }

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

        # Emit completed node span with stored inputs
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'node',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': run_info.get('inputs', {}),
            'outputs': outputs,
            'metadata': run_info.get('metadata', {})
        }

        logger.info(f"🚀 NODE END: Emitting span - component_type='node', name='{run_info['name']}', span_id={run_info['span_id']}")
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

        # Track the LLM run - store data for end callback
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'llm') if serialized else 'llm',
            'component_type': 'llm',
            'prompts': prompts,
            'metadata': metadata or {},
            'system_prompt': prompts[0] if prompts and len(prompts) > 0 else None
        }

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

        # Merge stored metadata with LLM metadata
        stored_metadata = run_info.get('metadata', {})
        stored_metadata.update(llm_metadata)

        # Emit completed LLM span with stored inputs and merged metadata
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'llm',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {'prompts': run_info.get('prompts', [])},
            'outputs': {'response': getattr(response, 'generations', []) if hasattr(response, 'generations') else []},
            'metadata': stored_metadata
        }

        # Add system prompt if stored
        if run_info.get('system_prompt'):
            span_data['system_prompt'] = run_info['system_prompt']

        # Add token counts to span data
        if 'input_tokens' in llm_metadata:
            span_data['input_tokens'] = llm_metadata['input_tokens']
        if 'output_tokens' in llm_metadata:
            span_data['output_tokens'] = llm_metadata['output_tokens']
        if 'total_tokens' in llm_metadata:
            span_data['total_tokens'] = llm_metadata['total_tokens']

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

        # Track the tool run - store data for end callback
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'tool') if serialized else 'tool',
            'component_type': 'tool',
            'input_str': input_str,
            'metadata': metadata or {}
        }

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

        # Emit completed tool span with stored inputs
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': 'tool',
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': {'input_str': run_info.get('input_str', '')},
            'outputs': {'output': output},
            'metadata': run_info.get('metadata', {})
        }

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
        logger.info(f"⛓️ CHAIN START CALLBACK: run_id={run_id}, parent_run_id={parent_run_id}")
        logger.info(f"⛓️ CHAIN START: serialized keys={list(serialized.keys()) if serialized else None}")
        logger.info(f"⛓️ CHAIN START: serialized={serialized}")
        logger.info(f"⛓️ CHAIN START: metadata={metadata}")
        logger.info(f"⛓️ CHAIN START: tags={tags}")

        # Detect if this is a node execution or internal chain
        is_node, node_name = self._is_node_execution(metadata, parent_run_id)
        
        if is_node:
            # This is a logical agent node execution
            component_type = 'node'
            name = node_name  # Use langgraph_node directly
            logger.info(f"⛓️ CHAIN START: Detected as NODE execution: {name}")
        else:
            # This is an internal chain or component
            component_type = 'chain'
            name = self._extract_component_name(serialized, metadata, tags)
            logger.info(f"⛓️ CHAIN START: Detected as CHAIN component: {name}")

        # Track the chain run - store data for end callback
        span_id = str(uuid.uuid4())
        parent_span_id = self._runs.get(parent_run_id, {}).get('span_id') if parent_run_id else None

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': name,
            'component_type': component_type,  # Dynamic: 'node' for logical agents, 'chain' for components
            'inputs': inputs,
            'metadata': metadata or {}
        }
        logger.info(f"⛓️ CHAIN START: Stored in _runs with component_type='{component_type}', name='{name}'")

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

        # Emit completed span with stored component_type (dynamic, not hardcoded)
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': run_info['component_type'],  # Use stored value (can be 'node' or 'chain')
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
            'inputs': run_info.get('inputs', {}),
            'outputs': outputs,
            'metadata': run_info.get('metadata', {})
        }

        logger.info(f"⛓️ CHAIN END: Emitting span - component_type='{run_info['component_type']}', "
                   f"name='{run_info['name']}', span_id={run_info['span_id']}")
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

        # Merge stored metadata with error info
        metadata = run_info.get('metadata', {})
        metadata['error_type'] = type(error).__name__

        # Get inputs based on component type (LLM uses 'prompts', tool uses 'input_str')
        if component_type == 'llm':
            inputs = {'prompts': run_info.get('prompts', [])}
        elif component_type == 'tool':
            inputs = {'input_str': run_info.get('input_str', '')}
        else:
            inputs = run_info.get('inputs', {})

        # Emit error span with stored inputs and merged metadata
        span_data = {
            'trace_id': self._trace_id,
            'span_id': run_info['span_id'],
            'parent_span_id': run_info['parent_span_id'],
            'component_type': component_type,
            'name': run_info['name'],
            'start_time': run_info['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'status': 'error',
            'inputs': inputs,
            'outputs': {'error': str(error)},
            'metadata': metadata
        }

        self._emit_span(span_data)
        del self._runs[run_id]

