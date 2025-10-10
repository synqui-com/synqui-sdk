"""LangChain integration for Vaquero SDK.

This module provides a callback handler that integrates LangChain operations
with Vaquero tracing, allowing seamless observability of LCEL chains, agents,
tools, and retrievers.
"""

from typing import Any, Dict, List, Optional
import uuid
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # Fallback for type hints

from .sdk import VaqueroSDK, get_global_instance
from .context import (
    get_current_span as _vaq_get_current_span,
    create_child_span as _vaq_create_child_span,
    span_context as _vaq_span_context,
)


class VaqueroCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that sends traces to Vaquero.

    This handler implements LangChain's BaseCallbackHandler interface to
    automatically trace chain, LLM, tool, and retriever operations.

    Example:
        from vaquero.langchain import VaqueroCallbackHandler

        handler = VaqueroCallbackHandler()
        chain.invoke(input, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        sdk: Optional[VaqueroSDK] = None,
        redact_prompts: bool = True,
        redact_outputs: bool = True,
        parent_context: Optional[Dict[str, Any]] = None
    ):
        """Initialize the callback handler.

        Args:
            sdk: Vaquero SDK instance to use. If None, uses the global instance.
            redact_prompts: Whether to redact prompt content in traces
            redact_outputs: Whether to redact output content in traces
            parent_context: Parent span context to inherit (session_id, parent_span_id, etc.)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: "
                "pip install langchain langchain-openai langchain-community"
            )

        self.sdk = sdk or get_global_instance()
        self.redact_prompts = redact_prompts
        self.redact_outputs = redact_outputs
        self.parent_context = parent_context or {}
        # Track active spans and their context managers so we can close on *_end
        # Structure: run_id -> {"span": TraceData, "cm": context_manager}
        self._spans = {}
        self._trace_id = str(uuid.uuid4())
        # Capture root context (if handler is created inside a vaquero.span)
        try:
            _root = _vaq_get_current_span()
            self._root_trace_id = _root.trace_id if _root else None
            self._root_span_id = _root.span_id if _root else None
        except Exception:
            self._root_trace_id = None
            self._root_span_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a chain starts."""
        span_name = serialized.get("name", "chain") if isinstance(serialized, dict) else "chain"

        # Check if we have a current span context from parent workflow (global context)
        current_span = _vaq_get_current_span()
        if current_span:
            # Inherit trace_id from current span context
            span = _vaq_create_child_span(
                agent_name=f"langchain:{span_name}",
                function_name=f"langchain:{span_name}",
                metadata=self.parent_context,
                tags=self.parent_context.get("tags", {})
            )
            # Set up context manager manually for this span
            cm = _vaq_span_context(span)
            span = cm.__enter__()
            self._spans[run_id] = {"span": span, "cm": cm}
        else:
            # Create new context manager if no parent context exists
            cm = self.sdk._span_context_manager(
                f"langchain:{span_name}",
                metadata=self.parent_context
            )
            span = cm.__enter__()
            # Force grouping under the root trace when available
            if self._root_trace_id:
                span.trace_id = self._root_trace_id
                if self._root_span_id:
                    span.parent_span_id = self._root_span_id
                    span.inputs = span.inputs or {}
                    span.inputs.setdefault("parent_span_id", self._root_span_id)
                    span.metadata.setdefault("parent_span_id", self._root_span_id)
            self._spans[run_id] = {"span": span, "cm": cm}

        # Add metadata
        if metadata:
            span.set_tag("langchain.metadata", metadata)
        if tags:
            span.set_tag("langchain.tags", tags)

        # Add inputs (also map to canonical inputs when not redacted)
        if inputs and not self.redact_prompts:
            try:
                # Serialize LangChain objects for storage
                serialized_inputs = self._serialize_langchain_data(inputs)
                span.set_tag("langchain.inputs", str(serialized_inputs)[:1000])  # Truncate for safety
                span.inputs = {"inputs": serialized_inputs}
            except Exception:
                span.set_tag("langchain.inputs", str(inputs)[:1000])  # Truncate for safety

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Called when a chain ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            # Add outputs (redacted if configured) and canonical outputs
            if outputs and not self.redact_outputs:
                try:
                    # Serialize LangChain objects for storage
                    serialized_outputs = self._serialize_langchain_data(outputs)
                    span.set_tag("langchain.outputs", str(serialized_outputs)[:1000])  # Truncate for safety
                    span.outputs = {"outputs": serialized_outputs}
                except Exception:
                    span.set_tag("langchain.outputs", str(outputs)[:1000])  # Truncate for safety
            # Close the span
            cm.__exit__(None, None, None)
            del self._spans[run_id]

    def on_chain_error(self, error, *, run_id, **kwargs):
        """Called when a chain errors."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            # Close the span with error
            try:
                cm.__exit__(type(error), error, None)
            finally:
                del self._spans[run_id]

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when an LLM starts."""
        model_name = serialized.get("kwargs", {}).get("model", "unknown") if isinstance(serialized, dict) else "unknown"
        span_name = f"llm:{model_name}"

        # Check if we have a current span context from parent workflow (global context)
        current_span = _vaq_get_current_span()
        if current_span:
            # Inherit trace_id from current span context
            span = _vaq_create_child_span(
                agent_name=span_name,
                function_name=span_name,
                metadata=self.parent_context,
                tags=self.parent_context.get("tags", {})
            )
            # Set up context manager manually for this span
            cm = _vaq_span_context(span)
            span = cm.__enter__()
            self._spans[run_id] = {"span": span, "cm": cm}
        else:
            # Keep LLM span open with parent context
            cm = self.sdk._span_context_manager(span_name, metadata=self.parent_context)
            span = cm.__enter__()
            # Force grouping under the root trace when available
            if self._root_trace_id:
                span.trace_id = self._root_trace_id
                if self._root_span_id:
                    span.parent_span_id = self._root_span_id
                    span.inputs = span.inputs or {}
                    span.inputs.setdefault("parent_span_id", self._root_span_id)
                    span.metadata.setdefault("parent_span_id", self._root_span_id)
            self._spans[run_id] = {"span": span, "cm": cm}

        # Add metadata
        if metadata:
            span.set_tag("langchain.metadata", metadata)
        if tags:
            span.set_tag("langchain.tags", tags)

        # Add prompt info (redacted if configured) and canonical inputs
        if prompts and not self.redact_prompts:
            try:
                # Serialize LangChain objects for storage
                serialized_prompts = self._serialize_langchain_data(prompts)
                span.set_tag("langchain.prompts", str(serialized_prompts)[:1000])
                span.set_tag("langchain.prompt_count", len(prompts))
                span.inputs = {"prompts": serialized_prompts}
            except Exception:
                span.set_tag("langchain.prompts", str(prompts)[:1000])
                span.set_tag("langchain.prompt_count", len(prompts))

    def on_llm_end(self, response, *, run_id, **kwargs):
        """Called when an LLM ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]

            # Extract token usage if available and map to canonical fields
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage', {})
                if usage:
                    span.set_tag("langchain.token_usage", usage)
                    try:
                        span.input_tokens = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
                        span.output_tokens = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
                        span.total_tokens = int(usage.get('total_tokens') or (span.input_tokens + span.output_tokens))
                    except Exception:
                        pass

            # Add response info (redacted if configured) and canonical outputs
            if hasattr(response, 'generations') and not self.redact_outputs:
                generations = response.generations
                try:
                    # Serialize LangChain objects for storage
                    serialized_generations = self._serialize_langchain_data(generations)
                    span.set_tag("langchain.generations", str(serialized_generations)[:1000])
                    span.outputs = {"generations": serialized_generations}
                except Exception:
                    span.set_tag("langchain.generations", str(generations)[:1000])

            # Close the span
            cm.__exit__(None, None, None)
            del self._spans[run_id]

    def on_llm_error(self, error, *, run_id, **kwargs):
        """Called when an LLM errors."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            try:
                cm.__exit__(type(error), error, None)
            finally:
                del self._spans[run_id]

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a tool starts."""
        tool_name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
        span_name = f"tool:{tool_name}"

        # Check if we have a current span context from parent workflow (global context)
        current_span = _vaq_get_current_span()
        if current_span:
            # Inherit trace_id from current span context
            span = _vaq_create_child_span(
                agent_name=span_name,
                function_name=span_name,
                metadata=self.parent_context,
                tags=self.parent_context.get("tags", {})
            )
            # Set up context manager manually for this span
            cm = _vaq_span_context(span)
            span = cm.__enter__()
            self._spans[run_id] = {"span": span, "cm": cm}
        else:
            cm = self.sdk._span_context_manager(span_name, metadata=self.parent_context)
            span = cm.__enter__()
            # Force grouping under the root trace when available
            if self._root_trace_id:
                span.trace_id = self._root_trace_id
                if self._root_span_id:
                    span.parent_span_id = self._root_span_id
                    span.inputs = span.inputs or {}
                    span.inputs.setdefault("parent_span_id", self._root_span_id)
                    span.metadata.setdefault("parent_span_id", self._root_span_id)
            self._spans[run_id] = {"span": span, "cm": cm}

        # Add metadata
        if metadata:
            span.set_tag("langchain.metadata", metadata)
        if tags:
            span.set_tag("langchain.tags", tags)

        # Add input (redacted if configured) and canonical inputs
        if input_str and not self.redact_prompts:
            try:
                # Serialize LangChain objects for storage
                serialized_input = self._serialize_langchain_data(input_str)
                span.set_tag("langchain.tool_input", str(serialized_input)[:1000])
                span.inputs = {"tool_input": serialized_input}
            except Exception:
                span.set_tag("langchain.tool_input", str(input_str)[:1000])

    def on_tool_end(self, output, *, run_id, **kwargs):
        """Called when a tool ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            # Add output (redacted if configured) and canonical outputs
            if output and not self.redact_outputs:
                try:
                    # Serialize LangChain objects for storage
                    serialized_output = self._serialize_langchain_data(output)
                    span.set_tag("langchain.tool_output", str(serialized_output)[:1000])
                    span.outputs = {"tool_output": serialized_output}
                except Exception:
                    span.set_tag("langchain.tool_output", str(output)[:1000])
            cm.__exit__(None, None, None)
            del self._spans[run_id]

    def on_tool_error(self, error, *, run_id, **kwargs):
        """Called when a tool errors."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            try:
                cm.__exit__(type(error), error, None)
            finally:
                del self._spans[run_id]

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a retriever starts."""
        retriever_name = serialized.get("name", "retriever") if isinstance(serialized, dict) else "retriever"
        span_name = f"retriever:{retriever_name}"

        # Check if we have a current span context from parent workflow (global context)
        current_span = _vaq_get_current_span()
        if current_span:
            # Inherit trace_id from current span context
            span = _vaq_create_child_span(
                agent_name=span_name,
                function_name=span_name,
                metadata=self.parent_context,
                tags=self.parent_context.get("tags", {})
            )
            # Set up context manager manually for this span
            cm = _vaq_span_context(span)
            span = cm.__enter__()
            self._spans[run_id] = {"span": span, "cm": cm}
        else:
            cm = self.sdk._span_context_manager(span_name, metadata=self.parent_context)
            span = cm.__enter__()
            # Force grouping under the root trace when available
            if self._root_trace_id:
                span.trace_id = self._root_trace_id
                if self._root_span_id:
                    span.parent_span_id = self._root_span_id
                    span.inputs = span.inputs or {}
                    span.inputs.setdefault("parent_span_id", self._root_span_id)
                    span.metadata.setdefault("parent_span_id", self._root_span_id)
            self._spans[run_id] = {"span": span, "cm": cm}

        # Add metadata
        if metadata:
            span.set_tag("langchain.metadata", metadata)
        if tags:
            span.set_tag("langchain.tags", tags)

        # Add query (redacted if configured) and canonical inputs
        if query and not self.redact_prompts:
            try:
                # Serialize LangChain objects for storage
                serialized_query = self._serialize_langchain_data(query)
                span.set_tag("langchain.query", str(serialized_query)[:1000])
                span.inputs = {"retriever_query": serialized_query}
            except Exception:
                span.set_tag("langchain.query", str(query)[:1000])

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        """Called when a retriever ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            # Add document count and canonical outputs summary
            if documents:
                span.set_tag("langchain.document_count", len(documents))
                try:
                    span.outputs = {"retriever_document_count": len(documents)}
                except Exception:
                    pass
            cm.__exit__(None, None, None)
            del self._spans[run_id]

    def on_retriever_error(self, error, *, run_id, **kwargs):
        """Called when a retriever errors."""
        if run_id in self._spans:
            span = self._spans[run_id]["span"]
            cm = self._spans[run_id]["cm"]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            try:
                cm.__exit__(type(error), error, None)
            finally:
                del self._spans[run_id]

    def _serialize_langchain_data(self, data):
        """Serialize LangChain objects to JSON-serializable format.
        
        Args:
            data: LangChain object or data to serialize
            
        Returns:
            JSON-serializable representation of the data
        """
        try:
            # Handle None first
            if data is None:
                return None
            
            # Handle specific LangChain objects first
            if hasattr(data, 'tool') and hasattr(data, 'tool_input'):
                # ToolAgentAction
                return {
                    "type": "ToolAgentAction",
                    "tool": data.tool,
                    "tool_input": self._serialize_langchain_data(data.tool_input),
                    "log": getattr(data, 'log', ''),
                    "message_log": [self._serialize_langchain_data(msg) for msg in getattr(data, 'message_log', [])],
                    "tool_call_id": getattr(data, 'tool_call_id', None)
                }
            elif hasattr(data, '__class__') and 'ToolAgentAction' in str(data.__class__):
                # Additional check for ToolAgentAction objects
                return {
                    "type": "ToolAgentAction",
                    "tool": getattr(data, 'tool', ''),
                    "tool_input": self._serialize_langchain_data(getattr(data, 'tool_input', {})),
                    "log": getattr(data, 'log', ''),
                    "message_log": [self._serialize_langchain_data(msg) for msg in getattr(data, 'message_log', [])],
                    "tool_call_id": getattr(data, 'tool_call_id', None)
                }
            elif hasattr(data, 'content'):
                # Message objects (AIMessage, HumanMessage, SystemMessage, etc.)
                return {
                    "type": type(data).__name__,
                    "content": data.content,
                    "additional_kwargs": getattr(data, 'additional_kwargs', {}),
                    "response_metadata": getattr(data, 'response_metadata', {}),
                    "id": getattr(data, 'id', None)
                }
            elif hasattr(data, 'messages'):
                # ChatPromptValue
                return {
                    "type": type(data).__name__,
                    "messages": [self._serialize_langchain_data(msg) for msg in data.messages]
                }
            elif isinstance(data, (str, int, float, bool)):
                # Primitive types
                return data
            elif isinstance(data, list):
                # List of objects
                return [self._serialize_langchain_data(item) for item in data]
            elif isinstance(data, dict):
                # Dictionary
                return {k: self._serialize_langchain_data(v) for k, v in data.items()}
            elif hasattr(data, '__dict__'):
                # Generic object with attributes - be more selective about what we include
                serializable_attrs = {}
                for k, v in data.__dict__.items():
                    if not k.startswith('_'):
                        try:
                            serializable_attrs[k] = self._serialize_langchain_data(v)
                        except Exception:
                            serializable_attrs[k] = str(v)
                
                return {
                    "type": type(data).__name__,
                    "data": serializable_attrs
                }
            else:
                # Fallback to string representation
                return str(data)
        except Exception as e:
            # Fallback to string representation with error info
            return f"<SerializationError: {type(data).__name__} - {str(e)}>"


def get_vaquero_handler(
    sdk: Optional[VaqueroSDK] = None,
    redact_prompts: bool = True,
    redact_outputs: bool = True,
    parent_context: Optional[Dict[str, Any]] = None
) -> VaqueroCallbackHandler:
    """Get a configured Vaquero callback handler for LangChain.

    This is a convenience function for creating a VaqueroCallbackHandler
    with sensible defaults.

    Args:
        sdk: Vaquero SDK instance to use. If None, uses the global instance.
        redact_prompts: Whether to redact prompt content in traces
        redact_outputs: Whether to redact output content in traces
        parent_context: Parent span context to inherit (session_id, parent_span_id, etc.)

    Returns:
        Configured VaqueroCallbackHandler instance

    Example:
        from vaquero.langchain import get_vaquero_handler

        handler = get_vaquero_handler()
        chain.invoke(input, config={"callbacks": [handler]})
    """
    return VaqueroCallbackHandler(sdk=sdk, redact_prompts=redact_prompts, redact_outputs=redact_outputs, parent_context=parent_context)
