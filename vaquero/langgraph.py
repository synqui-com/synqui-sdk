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
from .serialization import safe_serialize

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
        
        # Graph architecture cache - can be set via set_graph_architecture() or extracted from config
        self._graph_architecture: Optional[Dict[str, Any]] = None

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
    
    def _detect_model_provider(self, model_name: str) -> str:
        """
        Detect model provider from model name.
        
        Args:
            model_name: Model name (e.g., "ChatGoogleGenerativeAI", "ChatOpenAI", "gpt-4")
            
        Returns:
            Provider name (google, openai, anthropic, llama, or unknown)
        """
        if not model_name:
            return "unknown"
        
        model_lower = model_name.lower()
        
        # Check for Google/Gemini models
        if ("google" in model_lower or "gemini" in model_lower or 
            "ChatGoogleGenerativeAI" in model_name or "ChatGoogleGenerative" in model_name):
            return "google"
        
        # Check for OpenAI models
        elif ("gpt" in model_lower or "openai" in model_lower or 
              "ChatOpenAI" in model_name or "davinci" in model_lower or 
              "curie" in model_lower):
            return "openai"
        
        # Check for Anthropic models
        elif ("claude" in model_lower or "anthropic" in model_lower or 
              "ChatAnthropic" in model_name):
            return "anthropic"
        
        # Check for Llama models
        elif "llama" in model_lower:
            return "llama"
        
        else:
            return "unknown"
    
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
    
    def set_graph_architecture(self, graph: Any, graph_name: Optional[str] = None) -> None:
        """Set the complete graph architecture for this handler.
        
        This allows capturing the full graph structure (all nodes and edges) 
        even if they weren't executed during this trace.
        
        Args:
            graph: LangGraph graph object (from app.get_graph() or workflow.compile())
            graph_name: Optional graph name/identifier
        """
        try:
            # Extract nodes from graph
            graph_nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
            
            # Extract edges from graph
            graph_edges = []
            if hasattr(graph, 'edges'):
                for edge in graph.edges:
                    graph_edges.append({
                        'source': edge.source if hasattr(edge, 'source') else str(edge).split('->')[0].strip(),
                        'target': edge.target if hasattr(edge, 'target') else str(edge).split('->')[1].strip(),
                        'conditional': getattr(edge, 'conditional', False)
                    })
            
            # Find entry point (target of __start__ edge)
            entry_point = None
            for edge_dict in graph_edges:
                if edge_dict.get('source') == '__start__':
                    entry_point = edge_dict.get('target')
                    break
            
            self._graph_architecture = {
                'nodes': graph_nodes,
                'edges': graph_edges,
                'entry_point': entry_point,
                'graph_name': graph_name or 'langgraph_workflow'
            }
            logger.info(f"📊 GRAPH ARCH: Graph architecture set: {len(graph_nodes)} nodes, {len(graph_edges)} edges, entry={entry_point}")
        except Exception as e:
            logger.warning(f"Failed to extract graph architecture: {e}")
            self._graph_architecture = None
    
    def _emit_span(self, span_data: Dict[str, Any]) -> None:
        """Emit a normalized span to the trace collector."""
        # Always add session context if available
        if self._session_context.get('chat_session_id'):
            span_data['chat_session_id'] = self._session_context['chat_session_id']
        span_data['message_sequence'] = self._session_context.get('message_sequence', 0)

        # Ensure metadata exists
        if 'metadata' not in span_data:
            span_data['metadata'] = {}
        
        # Add graph architecture to metadata if available (for cases where on_graph_start isn't called)
        # This ensures the processor can extract it even if there are no graph spans
        if self._graph_architecture and 'graph_architecture' not in span_data.get('metadata', {}):
            span_data['metadata']['graph_architecture'] = self._graph_architecture
            logger.info(f"📊 GRAPH ARCH: Added graph architecture to span metadata (fallback): {len(self._graph_architecture.get('nodes', []))} nodes")

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

        # Try to extract graph architecture from config if available
        # LangGraph may pass config through kwargs in some cases, or we can check configurable
        graph_architecture = self._graph_architecture
        
        # Check if graph architecture was passed via configurable (if handler was in config)
        if not graph_architecture and 'config' in kwargs:
            config = kwargs.get('config', {})
            configurable = config.get('configurable', {})
            if 'graph_architecture' in configurable:
                graph_architecture = configurable['graph_architecture']
                logger.debug("Extracted graph architecture from config.configurable")
        
        # Increment message sequence per graph invocation (user turn)
        self._session_context['message_sequence'] = (self._session_context.get('message_sequence') or 0) + 1

        # Track the graph run - store data for end callback
        span_id = str(uuid.uuid4())
        graph_metadata = metadata or {}
        
        # Add graph architecture to metadata if we have it
        if graph_architecture:
            graph_metadata['graph_architecture'] = graph_architecture
            logger.info(f"📊 GRAPH ARCH: Added graph architecture to graph span metadata: {len(graph_architecture.get('nodes', []))} nodes, {len(graph_architecture.get('edges', []))} edges")
        
        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': None,  # Graph is root
            'start_time': datetime.utcnow(),
            'name': serialized.get('name', 'langgraph_workflow') if serialized else 'langgraph_workflow',
            'component_type': 'graph',
            'inputs': inputs,
            'metadata': graph_metadata
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

        # Extract model name and provider
        model_name = serialized.get('name', 'llm') if serialized else 'llm'
        model_provider = self._detect_model_provider(model_name)
        
        # Extract model parameters from serialized kwargs
        model_parameters = {}
        if serialized and isinstance(serialized, dict):
            kwargs = serialized.get('kwargs', {})
            if kwargs:
                # Extract common LLM parameters
                for param in ['temperature', 'max_tokens', 'top_p', 'top_k', 
                             'frequency_penalty', 'presence_penalty', 'n', 'stream']:
                    if param in kwargs:
                        model_parameters[param] = kwargs[param]
                
                # Also capture model name from kwargs if different
                if 'model' in kwargs and kwargs['model'] != model_name:
                    model_parameters['model'] = kwargs['model']

        self._runs[run_id] = {
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'start_time': datetime.utcnow(),
            'name': model_name,
            'component_type': 'llm',
            'prompts': prompts,
            'metadata': metadata or {},
            'system_prompt': prompts[0] if prompts and len(prompts) > 0 else None,
            'llm_model_provider': model_provider,
            'llm_model_parameters': model_parameters if model_parameters else None
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

        # INFO-level logging to inspect LLMResult structure
        logger.info(f"🔍 LLM RESULT INSPECTION: response type={type(response).__name__}")
        logger.info(f"🔍 LLM RESULT INSPECTION: response class={type(response)}")
        
        # Check all attributes of response
        response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
        logger.info(f"🔍 LLM RESULT INSPECTION: response attributes={response_attrs}")
        
        # Check if it's LLMResult type
        if 'LLMResult' in str(type(response)):
            logger.info(f"🔍 LLM RESULT INSPECTION: This is an LLMResult object")
            
            # Inspect llm_output
            if hasattr(response, 'llm_output'):
                llm_out = response.llm_output
                logger.info(f"🔍 LLM RESULT INSPECTION: has llm_output={llm_out is not None}")
                if llm_out:
                    if isinstance(llm_out, dict):
                        logger.info(f"🔍 LLM RESULT INSPECTION: llm_output keys={list(llm_out.keys())}")
                        logger.info(f"🔍 LLM RESULT INSPECTION: llm_output={llm_out}")
                        if 'token_usage' in llm_out:
                            logger.info(f"🔍 LLM RESULT INSPECTION: llm_output['token_usage']={llm_out['token_usage']}")
                    else:
                        logger.info(f"🔍 LLM RESULT INSPECTION: llm_output type={type(llm_out)}, value={llm_out}")
            
            # Inspect generations
            if hasattr(response, 'generations'):
                gens = response.generations
                logger.info(f"🔍 LLM RESULT INSPECTION: has generations={gens is not None}")
                if gens and isinstance(gens, list) and len(gens) > 0:
                    logger.info(f"🔍 LLM RESULT INSPECTION: generations count={len(gens)}")
                    if len(gens[0]) > 0:
                        first_gen = gens[0][0]
                        logger.info(f"🔍 LLM RESULT INSPECTION: first generation type={type(first_gen).__name__}")
                        logger.info(f"🔍 LLM RESULT INSPECTION: first generation attributes={[attr for attr in dir(first_gen) if not attr.startswith('_')]}")
                        
                        # Check if generation has message with response_metadata
                        if hasattr(first_gen, 'message'):
                            msg = first_gen.message
                            logger.info(f"🔍 LLM RESULT INSPECTION: message type={type(msg).__name__}")
                            if hasattr(msg, 'response_metadata'):
                                meta = msg.response_metadata
                                logger.info(f"🔍 LLM RESULT INSPECTION: response_metadata type={type(meta)}")
                                logger.info(f"🔍 LLM RESULT INSPECTION: response_metadata={meta}")
                                if isinstance(meta, dict) and 'token_usage' in meta:
                                    logger.info(f"🔍 LLM RESULT INSPECTION: response_metadata['token_usage']={meta['token_usage']}")
                                
                                # Also check usage_metadata in message
                                if hasattr(msg, 'usage_metadata'):
                                    usage = msg.usage_metadata
                                    logger.info(f"🔍 LLM RESULT INSPECTION: message.usage_metadata type={type(usage)}")
                                    logger.info(f"🔍 LLM RESULT INSPECTION: message.usage_metadata={usage}")
        else:
            # Standard response object inspection
            logger.info(f"🔍 LLM RESULT INSPECTION: has usage_metadata={hasattr(response, 'usage_metadata')}")
            logger.info(f"🔍 LLM RESULT INSPECTION: has llm_output={hasattr(response, 'llm_output')}")
            if hasattr(response, 'llm_output') and response.llm_output:
                logger.info(f"🔍 LLM RESULT INSPECTION: llm_output keys={list(response.llm_output.keys()) if isinstance(response.llm_output, dict) else 'not_dict'}")
                logger.info(f"🔍 LLM RESULT INSPECTION: llm_output={response.llm_output}")
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                logger.info(f"🔍 LLM RESULT INSPECTION: usage_metadata type={type(usage)}")
                logger.info(f"🔍 LLM RESULT INSPECTION: usage_metadata attributes={[attr for attr in dir(usage) if not attr.startswith('_')]}")
                if hasattr(usage, 'input_tokens'):
                    logger.info(f"🔍 LLM RESULT INSPECTION: usage.input_tokens={getattr(usage, 'input_tokens', None)}")
                if hasattr(usage, 'output_tokens'):
                    logger.info(f"🔍 LLM RESULT INSPECTION: usage.output_tokens={getattr(usage, 'output_tokens', None)}")
                if hasattr(usage, 'total_tokens'):
                    logger.info(f"🔍 LLM RESULT INSPECTION: usage.total_tokens={getattr(usage, 'total_tokens', None)}")

        # Extract token counts if available
        # Try usage_metadata first (LangChain standard)
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'input_tokens', None)
            output_tokens = getattr(usage, 'output_tokens', None)
            total_tokens = getattr(usage, 'total_tokens', None)
            
            if input_tokens is not None or output_tokens is not None or total_tokens is not None:
                llm_metadata.update({
                    'input_tokens': input_tokens or 0,
                    'output_tokens': output_tokens or 0,
                    'total_tokens': total_tokens or ((input_tokens or 0) + (output_tokens or 0))
                })
                logger.info(f"🔢 TOKEN EXTRACTION (usage_metadata): input={input_tokens}, output={output_tokens}, total={total_tokens}")
        
        # Also check llm_output for token usage (fallback for different response formats)
        if hasattr(response, 'llm_output') and response.llm_output:
            if isinstance(response.llm_output, dict):
                if 'token_usage' in response.llm_output:
                    token_usage = response.llm_output['token_usage']
                    logger.debug(f"🔍 LLM RESPONSE DEBUG: token_usage={token_usage}")
                    if isinstance(token_usage, dict):
                        # Only update if usage_metadata didn't set them or set them to 0
                        if 'input_tokens' not in llm_metadata or llm_metadata.get('input_tokens', 0) == 0:
                            llm_metadata['input_tokens'] = token_usage.get('prompt_tokens', 0) or token_usage.get('input_tokens', 0) or 0
                            logger.info(f"🔢 TOKEN EXTRACTION (llm_output.token_usage): input={llm_metadata['input_tokens']}")
                        if 'output_tokens' not in llm_metadata or llm_metadata.get('output_tokens', 0) == 0:
                            llm_metadata['output_tokens'] = token_usage.get('completion_tokens', 0) or token_usage.get('output_tokens', 0) or 0
                            logger.info(f"🔢 TOKEN EXTRACTION (llm_output.token_usage): output={llm_metadata['output_tokens']}")
                        if 'total_tokens' not in llm_metadata or llm_metadata.get('total_tokens', 0) == 0:
                            llm_metadata['total_tokens'] = token_usage.get('total_tokens', 0) or (llm_metadata.get('input_tokens', 0) + llm_metadata.get('output_tokens', 0))
                            logger.info(f"🔢 TOKEN EXTRACTION (llm_output.token_usage): total={llm_metadata['total_tokens']}")
                
                # Also check for direct token fields in llm_output
                if 'input_tokens' not in llm_metadata and 'input_token_count' in response.llm_output:
                    llm_metadata['input_tokens'] = response.llm_output.get('input_token_count', 0)
                    logger.info(f"🔢 TOKEN EXTRACTION (llm_output.input_token_count): input={llm_metadata['input_tokens']}")
                if 'output_tokens' not in llm_metadata and 'output_token_count' in response.llm_output:
                    llm_metadata['output_tokens'] = response.llm_output.get('output_token_count', 0)
                    logger.info(f"🔢 TOKEN EXTRACTION (llm_output.output_token_count): output={llm_metadata['output_tokens']}")
        
        # Check generations for token usage - PRIMARY PATH for LLMResult objects
        if hasattr(response, 'generations') and response.generations:
            for gen_list in response.generations:
                if isinstance(gen_list, list) and len(gen_list) > 0:
                    gen = gen_list[0]
                    if hasattr(gen, 'message'):
                        msg = gen.message
                        
                        # PRIORITY 1: Check message.usage_metadata (LangChain standard for LLMResult)
                        if hasattr(msg, 'usage_metadata'):
                            usage_meta = msg.usage_metadata
                            if isinstance(usage_meta, dict):
                                input_toks = usage_meta.get('input_tokens') or usage_meta.get('prompt_tokens', 0)
                                output_toks = usage_meta.get('output_tokens') or usage_meta.get('completion_tokens', 0)
                                total_toks = usage_meta.get('total_tokens', 0) or (input_toks + output_toks)
                                
                                if input_toks or output_toks:
                                    llm_metadata['input_tokens'] = input_toks
                                    llm_metadata['output_tokens'] = output_toks
                                    llm_metadata['total_tokens'] = total_toks
                                    logger.info(f"🔢 TOKEN EXTRACTION (generations[].message.usage_metadata): input={input_toks}, output={output_toks}, total={total_toks}")
                                    break  # Found tokens, use this
                        
                        # PRIORITY 2: Check message.response_metadata.token_usage (alternative format)
                        if hasattr(msg, 'response_metadata'):
                            metadata = msg.response_metadata
                            if isinstance(metadata, dict) and 'token_usage' in metadata:
                                token_usage = metadata['token_usage']
                                if isinstance(token_usage, dict):
                                    if 'input_tokens' not in llm_metadata or llm_metadata.get('input_tokens', 0) == 0:
                                        llm_metadata['input_tokens'] = token_usage.get('prompt_tokens', 0) or token_usage.get('input_tokens', 0) or 0
                                        logger.info(f"🔢 TOKEN EXTRACTION (generations[].message.response_metadata.token_usage): input={llm_metadata['input_tokens']}")
                                    if 'output_tokens' not in llm_metadata or llm_metadata.get('output_tokens', 0) == 0:
                                        llm_metadata['output_tokens'] = token_usage.get('completion_tokens', 0) or token_usage.get('output_tokens', 0) or 0
                                        logger.info(f"🔢 TOKEN EXTRACTION (generations[].message.response_metadata.token_usage): output={llm_metadata['output_tokens']}")
                                    if 'total_tokens' not in llm_metadata or llm_metadata.get('total_tokens', 0) == 0:
                                        llm_metadata['total_tokens'] = token_usage.get('total_tokens', 0) or (llm_metadata.get('input_tokens', 0) + llm_metadata.get('output_tokens', 0))
                                        logger.info(f"🔢 TOKEN EXTRACTION (generations[].message.response_metadata.token_usage): total={llm_metadata['total_tokens']}")
                                    break  # Found tokens, use this
        
        # Final check: ensure we have at least zeros
        if 'input_tokens' not in llm_metadata:
            llm_metadata['input_tokens'] = 0
        if 'output_tokens' not in llm_metadata:
            llm_metadata['output_tokens'] = 0
        if 'total_tokens' not in llm_metadata:
            llm_metadata['total_tokens'] = llm_metadata.get('input_tokens', 0) + llm_metadata.get('output_tokens', 0)
        
        # Log final token extraction result
        if llm_metadata.get('input_tokens') or llm_metadata.get('output_tokens'):
            logger.info(f"🔢 TOKEN EXTRACTION FINAL: input={llm_metadata.get('input_tokens', 0)}, "
                       f"output={llm_metadata.get('output_tokens', 0)}, "
                       f"total={llm_metadata.get('total_tokens', 0)}")
        else:
            logger.warning(f"⚠️ TOKEN EXTRACTION FAILED: No tokens found in response. response type={type(response)}, "
                          f"has usage_metadata={hasattr(response, 'usage_metadata')}, "
                          f"has llm_output={hasattr(response, 'llm_output')}")

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
            'outputs': {'response': safe_serialize(getattr(response, 'generations', []) if hasattr(response, 'generations') else [])},
            'metadata': stored_metadata
        }

        # Add system prompt if stored
        if run_info.get('system_prompt'):
            span_data['system_prompt'] = run_info['system_prompt']
            
            # Generate prompt hash
            import hashlib
            span_data['prompt_hash'] = hashlib.sha256(
                run_info['system_prompt'].encode('utf-8')
            ).hexdigest()
            
            # Extract prompt name/version from metadata if available
            metadata = run_info.get('metadata', {})
            if metadata.get('prompt_name'):
                span_data['prompt_name'] = metadata['prompt_name']
            if metadata.get('prompt_version'):
                span_data['prompt_version'] = metadata['prompt_version']

        # Add model provider if stored
        if run_info.get('llm_model_provider'):
            span_data['llm_model_provider'] = run_info['llm_model_provider']
        
        # Add model parameters if stored
        if run_info.get('llm_model_parameters'):
            span_data['llm_model_parameters'] = run_info['llm_model_parameters']

        # Add token counts to span data
        if 'input_tokens' in llm_metadata:
            span_data['input_tokens'] = llm_metadata['input_tokens']
        if 'output_tokens' in llm_metadata:
            span_data['output_tokens'] = llm_metadata['output_tokens']
        if 'total_tokens' in llm_metadata:
            span_data['total_tokens'] = llm_metadata['total_tokens']

        # Calculate cost if we have tokens and model info
        input_tokens = llm_metadata.get('input_tokens', 0)
        output_tokens = llm_metadata.get('output_tokens', 0)
        if (input_tokens > 0 or output_tokens > 0):
            try:
                from .cost_calculator import calculate_cost
                model_name = run_info['name']
                model_provider = run_info.get('llm_model_provider', 'unknown')
                
                cost = calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=model_name,
                    provider=model_provider
                )
                span_data['cost'] = cost
            except Exception as e:
                logger.debug(f"Failed to calculate cost: {e}")
                span_data['cost'] = 0.0
        else:
            span_data['cost'] = 0.0

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

