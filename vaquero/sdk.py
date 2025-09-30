"""Main Vaquero SDK implementation."""

import asyncio
import functools
import logging
import time
import atexit
from contextlib import asynccontextmanager, contextmanager
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional, Union, Generator
from datetime import datetime

from .config import SDKConfig
from .models import TraceData, SpanStatus
from .context import span_context, create_child_span
from .serialization import safe_serialize
from .batch_processor import BatchProcessor
from .token_counter import count_function_tokens, extract_tokens_from_llm_response
from .auto_instrumentation import AutoInstrumentationEngine

logger = logging.getLogger(__name__)


class VaqueroSDK:
    """Main SDK class for Vaquero instrumentation.

    This class provides the core functionality for tracing function calls
    and sending trace data to the Vaquero platform.

    Example:
        config = SDKConfig(api_key="your-key", project_id="your-project")
        sdk = VaqueroSDK(config)

        @sdk.trace("my_agent")
        def my_function():
            return "result"
    """

    def __init__(self, config: SDKConfig):
        """Initialize the SDK with configuration.

        Args:
            config: SDK configuration instance
        """
        self.config = config
        self._event_queue: Queue = Queue()
        self._batch_processor: Optional[BatchProcessor] = None
        self._enabled = config.enabled
        self._auto_instrumentation: Optional[AutoInstrumentationEngine] = None

        # Set up logging
        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Start batch processor if enabled
        if self._enabled:
            self._start_batch_processor()

            # Initialize auto-instrumentation if enabled
            if config.auto_instrument_llm:
                self._start_auto_instrumentation()

        # Register automatic shutdown to ensure traces are flushed on program exit
        if self._enabled:
            atexit.register(self.shutdown)

        logger.info(f"Vaquero SDK initialized (enabled={self._enabled})")

    def _start_batch_processor(self):
        """Start the background batch processor."""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(self)
            self._batch_processor.start()
            logger.debug("Batch processor started")
    
    def _start_auto_instrumentation(self):
        """Start automatic LLM instrumentation."""
        if self._auto_instrumentation is None:
            self._auto_instrumentation = AutoInstrumentationEngine(self)
            self._auto_instrumentation.instrument_all()
            logger.debug("Auto-instrumentation started")
    
    def _stop_auto_instrumentation(self):
        """Stop automatic LLM instrumentation."""
        if self._auto_instrumentation:
            self._auto_instrumentation.restore_original_methods()
            self._auto_instrumentation = None
            logger.debug("Auto-instrumentation stopped")

    def trace(self, agent_name: str, **kwargs) -> Callable:
        """Decorator for tracing function calls.

        This decorator can be used on both synchronous and asynchronous functions.
        It automatically captures timing, inputs, outputs, and errors.

        Args:
            agent_name: Name of the agent/component being traced
            **kwargs: Additional options (tags, metadata, etc.)

        Returns:
            Decorated function

        Example:
            @sdk.trace("data_processor")
            def process_data(data):
                return {"processed": data}

            @sdk.trace("api_client")
            async def fetch_data(url):
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    return response.json()
        """

        def decorator(func: Callable) -> Callable:
            if not self._enabled:
                return func

            if asyncio.iscoroutinefunction(func):
                return self._async_trace_decorator(func, agent_name, **kwargs)
            else:
                return self._sync_trace_decorator(func, agent_name, **kwargs)

        return decorator

    def _sync_trace_decorator(self, func: Callable, agent_name: str, **kwargs) -> Callable:
        """Synchronous trace decorator implementation."""

        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            if not self._enabled:
                return func(*args, **func_kwargs)

            # Create trace data
            trace_data = create_child_span(
                agent_name=agent_name,
                function_name=func.__name__,
                tags=kwargs.get("tags", {}),
                metadata=kwargs.get("metadata", {})
            )

            # Set the name field for workflow traces to match agent_name
            # This ensures the batch processor can identify workflow spans
            trace_data.name = agent_name

            # Set prompt fields if provided
            self._set_prompt_fields(trace_data, kwargs)

            # Add global tags from config
            trace_data.tags.update(self.config.tags)

            with span_context(trace_data):
                try:
                    # Capture inputs
                    if self.config.capture_inputs:
                        trace_data.inputs = self._capture_inputs(args, func_kwargs)

                    # Execute function
                    result = func(*args, **func_kwargs)

                    # Capture outputs
                    if self.config.capture_outputs:
                        trace_data.outputs = self._capture_outputs(result)

                    # Count tokens if enabled
                    if self.config.capture_tokens:
                        self._count_tokens(trace_data, args, func_kwargs, result)

                    # Mark as completed
                    trace_data.finish(SpanStatus.COMPLETED)

                    return result

                except Exception as e:
                    # Capture error
                    if self.config.capture_errors:
                        trace_data.set_error(e)
                    else:
                        trace_data.finish(SpanStatus.FAILED)

                    raise

                finally:
                    # Ensure parent reference is persisted
                    if trace_data.parent_span_id:
                        trace_data.inputs = trace_data.inputs or {}
                        trace_data.inputs.setdefault("parent_span_id", trace_data.parent_span_id)
                        trace_data.metadata.setdefault("parent_span_id", trace_data.parent_span_id)

                    # Send trace data
                    self._send_trace(trace_data)

        return wrapper

    def _async_trace_decorator(self, func: Callable, agent_name: str, **kwargs) -> Callable:
        """Asynchronous trace decorator implementation."""

        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs):
            if not self._enabled:
                return await func(*args, **func_kwargs)

            # Create trace data
            trace_data = create_child_span(
                agent_name=agent_name,
                function_name=func.__name__,
                tags=kwargs.get("tags", {}),
                metadata=kwargs.get("metadata", {})
            )

            # Set the name field for workflow traces to match agent_name
            # This ensures the batch processor can identify workflow spans
            trace_data.name = agent_name

            # Set prompt fields if provided
            self._set_prompt_fields(trace_data, kwargs)

            # Add global tags from config
            trace_data.tags.update(self.config.tags)

            with span_context(trace_data):
                try:
                    # Capture inputs
                    if self.config.capture_inputs:
                        trace_data.inputs = self._capture_inputs(args, func_kwargs)

                    # Execute function
                    result = await func(*args, **func_kwargs)

                    # Capture outputs
                    if self.config.capture_outputs:
                        trace_data.outputs = self._capture_outputs(result)

                    # Count tokens if enabled
                    if self.config.capture_tokens:
                        self._count_tokens(trace_data, args, func_kwargs, result)

                    # Mark as completed
                    trace_data.finish(SpanStatus.COMPLETED)

                    return result

                except Exception as e:
                    # Capture error
                    if self.config.capture_errors:
                        trace_data.set_error(e)
                    else:
                        trace_data.finish(SpanStatus.FAILED)

                    raise

                finally:
                    # Ensure parent reference is persisted
                    if trace_data.parent_span_id:
                        trace_data.inputs = trace_data.inputs or {}
                        trace_data.inputs.setdefault("parent_span_id", trace_data.parent_span_id)
                        trace_data.metadata.setdefault("parent_span_id", trace_data.parent_span_id)

                    # Send trace data
                    self._send_trace(trace_data)

        return wrapper

    @contextmanager
    def span(self, operation_name: str, **kwargs) -> Generator[TraceData, None, None]:
        """Context manager for manual span creation.

        This allows for fine-grained control over span creation and management.

        Args:
            operation_name: Name of the operation
            **kwargs: Additional options (tags, metadata, etc.)

        Yields:
            TraceData instance for the span

        Example:
            with sdk.span("custom_operation") as span:
                span.set_attribute("batch_size", 100)
                # Your code here
        """
        if not self._enabled:
            # Create a dummy span that does nothing
            dummy_span = TraceData(agent_name=operation_name)
            yield dummy_span
            return

        # Create trace data
        trace_data = create_child_span(
            agent_name=operation_name,
            function_name=operation_name,
            tags=kwargs.get("tags", {}),
            metadata=kwargs.get("metadata", {})
        )

        # Set the name field for workflow spans to match agent_name
        # This ensures the batch processor can identify workflow spans
        trace_data.name = operation_name

        # Add global tags from config
        trace_data.tags.update(self.config.tags)

        with span_context(trace_data):
            try:
                yield trace_data
            except Exception as e:
                if self.config.capture_errors:
                    trace_data.set_error(e)
                else:
                    trace_data.finish(SpanStatus.FAILED)
                raise
            finally:
                # Finish span if not already finished
                if trace_data.status == SpanStatus.RUNNING:
                    trace_data.finish(SpanStatus.COMPLETED)

                # Send trace data
                self._send_trace(trace_data)

    @asynccontextmanager
    async def async_span(self, operation_name: str, **kwargs):
        """Async context manager for manual span creation.

        Args:
            operation_name: Name of the operation
            **kwargs: Additional options (tags, metadata, etc.)

        Yields:
            TraceData instance for the span
        """
        if not self._enabled:
            # Create a dummy span that does nothing
            dummy_span = TraceData(agent_name=operation_name)
            yield dummy_span
            return

        # Create trace data
        trace_data = create_child_span(
            agent_name=operation_name,
            function_name=operation_name,
            tags=kwargs.get("tags", {}),
            metadata=kwargs.get("metadata", {})
        )

        # Set the name field for workflow spans to match agent_name
        # This ensures the batch processor can identify workflow spans
        trace_data.name = operation_name

        # Add global tags from config
        trace_data.tags.update(self.config.tags)

        with span_context(trace_data):
            try:
                yield trace_data
            except Exception as e:
                if self.config.capture_errors:
                    trace_data.set_error(e)
                else:
                    trace_data.finish(SpanStatus.FAILED)
                raise
            finally:
                # Finish span if not already finished
                if trace_data.status == SpanStatus.RUNNING:
                    trace_data.finish(SpanStatus.COMPLETED)

                # Send trace data
                self._send_trace(trace_data)

    def _capture_inputs(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Safely capture function inputs.

        Args:
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Dictionary containing serialized inputs
        """
        try:
            return {
                "args": [safe_serialize(arg) for arg in args],
                "kwargs": {k: safe_serialize(v) for k, v in kwargs.items()}
            }
        except Exception as e:
            logger.debug(f"Failed to capture inputs: {e}")
            return {"error": "Failed to capture inputs"}

    def _capture_outputs(self, result: Any) -> Dict[str, Any]:
        """Safely capture function outputs.

        Args:
            result: Function return value

        Returns:
            Dictionary containing serialized outputs
        """
        try:
            return {"result": safe_serialize(result)}
        except Exception as e:
            logger.debug(f"Failed to capture outputs: {e}")
            return {"error": "Failed to capture outputs"}

    def _count_tokens(self, trace_data: TraceData, args: tuple, kwargs: dict, result: Any) -> None:
        """Count tokens for function inputs and outputs.

        Args:
            trace_data: Trace data to update with token counts
            args: Function arguments
            kwargs: Function keyword arguments
            result: Function result
        """
        try:
            # Try to extract tokens from LLM response first
            if hasattr(result, 'usage') or isinstance(result, dict) and 'usage' in result:
                token_count = extract_tokens_from_llm_response(result)
                if token_count.total_tokens > 0:
                    trace_data.input_tokens = token_count.input_tokens
                    trace_data.output_tokens = token_count.output_tokens
                    trace_data.total_tokens = token_count.total_tokens
                    trace_data.model_name = token_count.model
                    trace_data.model_provider = token_count.provider
                    return
            
            # Fallback to counting tokens from inputs and outputs
            token_count = count_function_tokens(args, result)
            trace_data.input_tokens = token_count.input_tokens
            trace_data.output_tokens = token_count.output_tokens
            trace_data.total_tokens = token_count.total_tokens
            
        except Exception as e:
            logger.debug(f"Failed to count tokens: {e}")
            # Set default values
            trace_data.input_tokens = 0
            trace_data.output_tokens = 0
            trace_data.total_tokens = 0

    def _send_trace(self, trace_data: TraceData):
        """Send trace data to the batch processor.

        Args:
            trace_data: TraceData instance to send
        """
        if not self._enabled:
            return

        try:
            # Add environment and mode information
            trace_data.metadata["environment"] = self.config.environment
            trace_data.metadata["mode"] = self.config.mode
            trace_data.metadata["sdk_version"] = "0.1.0"

            # Queue the trace data
            self._event_queue.put(trace_data, timeout=1.0)
            logger.debug(f"Queued trace: {trace_data.agent_name} ({trace_data.span_id})")

        except Exception as e:
            logger.warning(f"Failed to queue trace data: {e}")

    def _set_prompt_fields(self, trace_data: TraceData, kwargs: Dict[str, Any]) -> None:
        """Populate explicit prompt fields on the trace if provided.

        Supported kwargs:
            - system_prompt: str
            - prompt_name: str
            - prompt_version: str
            - prompt_parameters: dict
        """
        try:
            system_prompt = kwargs.get("system_prompt")
            if isinstance(system_prompt, str):
                trace_data.system_prompt = system_prompt
                # Compute a stable hash for dedup/version hint
                import hashlib
                trace_data.prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

            prompt_name = kwargs.get("prompt_name")
            if isinstance(prompt_name, str):
                trace_data.prompt_name = prompt_name

            prompt_version = kwargs.get("prompt_version")
            if isinstance(prompt_version, str):
                trace_data.prompt_version = prompt_version

            prompt_parameters = kwargs.get("prompt_parameters")
            if isinstance(prompt_parameters, dict):
                trace_data.prompt_parameters = prompt_parameters
        except Exception as e:
            logger.debug(f"Failed to set prompt fields: {e}")

    def flush(self, timeout: Optional[float] = None):
        """Manually flush pending traces.

        Args:
            timeout: Maximum time to wait for flush to complete
        """
        if self._batch_processor:
            self._batch_processor.flush()
            logger.debug("Flushed pending traces")

    def shutdown(self, timeout: Optional[float] = None):
        """Shutdown the SDK and flush remaining traces.

        Args:
            timeout: Maximum time to wait for shutdown to complete
        """
        # Stop auto-instrumentation
        self._stop_auto_instrumentation()
        
        if self._batch_processor:
            self._batch_processor.shutdown()
            self._batch_processor = None
            logger.info("SDK shutdown completed")

    def is_enabled(self) -> bool:
        """Check if the SDK is enabled.

        Returns:
            True if SDK is enabled, False otherwise
        """
        return self._enabled

    def get_queue_size(self) -> int:
        """Get the current size of the event queue.

        Returns:
            Number of events in the queue
        """
        return self._event_queue.qsize()


# Global SDK instance
_sdk_instance: Optional[VaqueroSDK] = None


def get_current_sdk() -> Optional[VaqueroSDK]:
    """Get the current global SDK instance.
    
    Returns:
        Current SDK instance or None if not initialized
    """
    return _sdk_instance


def initialize(config: SDKConfig) -> VaqueroSDK:
    """Initialize the global SDK instance.
    
    Args:
        config: SDK configuration
        
    Returns:
        Initialized SDK instance
    """
    global _sdk_instance
    _sdk_instance = VaqueroSDK(config)
    return _sdk_instance