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
        redact_outputs: bool = True
    ):
        """Initialize the callback handler.

        Args:
            sdk: Vaquero SDK instance to use. If None, uses the global instance.
            redact_prompts: Whether to redact prompt content in traces
            redact_outputs: Whether to redact output content in traces
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: "
                "pip install langchain langchain-openai langchain-community"
            )

        self.sdk = sdk or get_global_instance()
        self.redact_prompts = redact_prompts
        self.redact_outputs = redact_outputs
        self._spans = {}
        self._trace_id = str(uuid.uuid4())

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a chain starts."""
        span_name = serialized.get("name", "chain") if isinstance(serialized, dict) else "chain"

        # Create a span for this chain
        with self.sdk.span(f"langchain:{span_name}") as span:
            self._spans[run_id] = span

            # Add metadata
            if metadata:
                span.set_tag("langchain.metadata", metadata)
            if tags:
                span.set_tag("langchain.tags", tags)

            # Add inputs (redacted if configured)
            if inputs and not self.redact_prompts:
                span.set_tag("langchain.inputs", str(inputs)[:1000])  # Truncate for safety

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Called when a chain ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]
            # Add outputs (redacted if configured)
            if outputs and not self.redact_outputs:
                span.set_tag("langchain.outputs", str(outputs)[:1000])  # Truncate for safety
            del self._spans[run_id]

    def on_chain_error(self, error, *, run_id, **kwargs):
        """Called when a chain errors."""
        if run_id in self._spans:
            span = self._spans[run_id]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            del self._spans[run_id]

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when an LLM starts."""
        model_name = serialized.get("kwargs", {}).get("model", "unknown") if isinstance(serialized, dict) else "unknown"
        span_name = f"llm:{model_name}"

        with self.sdk.span(span_name) as span:
            self._spans[run_id] = span

            # Add metadata
            if metadata:
                span.set_tag("langchain.metadata", metadata)
            if tags:
                span.set_tag("langchain.tags", tags)

            # Add prompt info (redacted if configured)
            if prompts and not self.redact_prompts:
                span.set_tag("langchain.prompts", str(prompts)[:1000])
                span.set_tag("langchain.prompt_count", len(prompts))

    def on_llm_end(self, response, *, run_id, **kwargs):
        """Called when an LLM ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]

            # Extract token usage if available
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage', {})
                if usage:
                    span.set_tag("langchain.token_usage", usage)

            # Add response info (redacted if configured)
            if hasattr(response, 'generations') and not self.redact_outputs:
                generations = response.generations
                span.set_tag("langchain.generations", str(generations)[:1000])

            del self._spans[run_id]

    def on_llm_error(self, error, *, run_id, **kwargs):
        """Called when an LLM errors."""
        if run_id in self._spans:
            span = self._spans[run_id]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            del self._spans[run_id]

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a tool starts."""
        tool_name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
        span_name = f"tool:{tool_name}"

        with self.sdk.span(span_name) as span:
            self._spans[run_id] = span

            # Add metadata
            if metadata:
                span.set_tag("langchain.metadata", metadata)
            if tags:
                span.set_tag("langchain.tags", tags)

            # Add input (redacted if configured)
            if input_str and not self.redact_prompts:
                span.set_tag("langchain.tool_input", str(input_str)[:1000])

    def on_tool_end(self, output, *, run_id, **kwargs):
        """Called when a tool ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]
            # Add output (redacted if configured)
            if output and not self.redact_outputs:
                span.set_tag("langchain.tool_output", str(output)[:1000])
            del self._spans[run_id]

    def on_tool_error(self, error, *, run_id, **kwargs):
        """Called when a tool errors."""
        if run_id in self._spans:
            span = self._spans[run_id]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            del self._spans[run_id]

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        """Called when a retriever starts."""
        retriever_name = serialized.get("name", "retriever") if isinstance(serialized, dict) else "retriever"
        span_name = f"retriever:{retriever_name}"

        with self.sdk.span(span_name) as span:
            self._spans[run_id] = span

            # Add metadata
            if metadata:
                span.set_tag("langchain.metadata", metadata)
            if tags:
                span.set_tag("langchain.tags", tags)

            # Add query (redacted if configured)
            if query and not self.redact_prompts:
                span.set_tag("langchain.query", str(query)[:1000])

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        """Called when a retriever ends successfully."""
        if run_id in self._spans:
            span = self._spans[run_id]
            # Add document count
            if documents:
                span.set_tag("langchain.document_count", len(documents))
                # Don't capture full documents for privacy/security
            del self._spans[run_id]

    def on_retriever_error(self, error, *, run_id, **kwargs):
        """Called when a retriever errors."""
        if run_id in self._spans:
            span = self._spans[run_id]
            span.set_tag("langchain.error", str(error))
            span.set_tag("langchain.error_type", type(error).__name__)
            del self._spans[run_id]


def get_vaquero_handler(
    sdk: Optional[VaqueroSDK] = None,
    redact_prompts: bool = True,
    redact_outputs: bool = True
) -> VaqueroCallbackHandler:
    """Get a configured Vaquero callback handler for LangChain.

    This is a convenience function for creating a VaqueroCallbackHandler
    with sensible defaults.

    Args:
        sdk: Vaquero SDK instance to use. If None, uses the global instance.
        redact_prompts: Whether to redact prompt content in traces
        redact_outputs: Whether to redact output content in traces

    Returns:
        Configured VaqueroCallbackHandler instance

    Example:
        from vaquero.langchain import get_vaquero_handler

        handler = get_vaquero_handler()
        chain.invoke(input, config={"callbacks": [handler]})
    """
    return VaqueroCallbackHandler(sdk=sdk, redact_prompts=redact_prompts, redact_outputs=redact_outputs)
