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
from .trace_collector_unified import UnifiedTraceCollector
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
        self._trace_collector: Optional[UnifiedTraceCollector] = None
        self._enabled = config.enabled
        self._auto_instrumentation: Optional[AutoInstrumentationEngine] = None

        # Set up logging
        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Start trace collector if enabled
        if self._enabled:
            self._start_trace_collector()

            # Initialize auto-instrumentation if enabled
            if config.auto_instrument_llm:
                self._start_auto_instrumentation()

        # Register automatic shutdown to ensure traces are flushed on program exit
        if self._enabled:
            atexit.register(self.shutdown)

        logger.info(f"Vaquero SDK initialized (enabled={self._enabled})")

    def _start_trace_collector(self):
        """Start the unified trace collector."""
        if self._trace_collector is None:
            self._trace_collector = UnifiedTraceCollector(self)
            logger.debug("Unified trace collector started")
    
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

    def trace(self, agent_name: str, capture_code: bool = True, **kwargs) -> Callable:
        """Decorator for tracing function calls.

        This decorator can be used on both synchronous and asynchronous functions.
        It automatically captures timing, inputs, outputs, and errors.

        Args:
            agent_name: Name of the agent/component being traced
            capture_code: Whether to capture source code and docstring for analysis
            **kwargs: Additional options (tags, metadata, etc.)

        Returns:
            Decorated function

        Example:
            @sdk.trace("data_processor", capture_code=True)
            def process_data(data):
                \"\"\"
                Process data with expected performance of 1-5 seconds.
                \"\"\"
                return {"processed": data}

            @sdk.trace("api_client")
            async def fetch_data(url):
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    return response.json()
        """
        logger.info(f"SDK: === CREATING TRACE DECORATOR ===")
        logger.info(f"SDK: Agent name: {agent_name}")
        logger.info(f"SDK: Capture code: {capture_code}")
        logger.info(f"SDK: Additional kwargs: {kwargs}")

        def decorator(func: Callable) -> Callable:
            logger.info(f"SDK: === APPLYING DECORATOR TO FUNCTION ===")
            logger.info(f"SDK: Function name: {func.__name__}")
            logger.info(f"SDK: Function module: {func.__module__}")
            logger.info(f"SDK: Function qualname: {func.__qualname__}")
            logger.info(f"SDK: Is coroutine function: {asyncio.iscoroutinefunction(func)}")

            if not self._enabled:
                logger.warning(f"SDK: SDK is disabled, returning original function {func.__name__}")
                return func

            # Capture code context if enabled
            logger.info(f"SDK: Capture code enabled: {capture_code}")
            code_context = {}
            if capture_code:
                logger.info(f"SDK: Calling _capture_code_context for {func.__name__}")
                code_context = self._capture_code_context(func)
                logger.info(f"SDK: Code context captured: {len(code_context)} keys")
            else:
                logger.warning(f"SDK: Capture code disabled for {func.__name__}")

            if asyncio.iscoroutinefunction(func):
                logger.info(f"SDK: Creating async trace decorator for {func.__name__}")
                return self._async_trace_decorator(func, agent_name, code_context, **kwargs)
            else:
                logger.info(f"SDK: Creating sync trace decorator for {func.__name__}")
                return self._sync_trace_decorator(func, agent_name, code_context, **kwargs)

        return decorator

    def _capture_code_context(self, func: Callable) -> Dict[str, Any]:
        """Capture code context for analysis."""
        logger.info(f"SDK: === CAPTURING CODE CONTEXT FOR {func.__name__} ===")
        logger.info(f"SDK: Function object: {func}")
        logger.info(f"SDK: Function name: {func.__name__}")
        logger.info(f"SDK: Function qualname: {func.__qualname__}")
        logger.info(f"SDK: Function module: {func.__module__}")

        try:
            import inspect

            logger.info(f"SDK: Starting inspect.getsource() for {func.__name__}")
            # Extract source code
            source_code = inspect.getsource(func)
            logger.info(f"SDK: Successfully extracted source code for {func.__name__} (length: {len(source_code)})")
            logger.debug(f"SDK: source_code preview: {source_code[:200]}...")

            # Extract docstring
            docstring = func.__doc__
            docstring_length = len(docstring) if docstring else 0
            logger.info(f"SDK: Extracted docstring for {func.__name__} (length: {docstring_length})")
            logger.debug(f"SDK: docstring: {docstring or 'None'}")

            # Extract function signature
            signature = str(inspect.signature(func))
            logger.info(f"SDK: Extracted function signature: {signature}")

            # Extract module and file info
            module = inspect.getmodule(func)
            module_name = module.__name__ if module else None
            file_path = module.__file__ if module else None
            logger.info(f"SDK: Module info - name: {module_name}, file: {file_path}")

            code_context = {
                'source_code': source_code,
                'docstring': docstring,
                'function_signature': signature,
                'module_name': module_name,
                'file_path': file_path,
                'function_name': func.__name__
            }

            logger.info(f"SDK: Successfully captured complete code context for {func.__name__}")
            logger.debug(f"SDK: Code context keys: {list(code_context.keys())}")

            return code_context
        except Exception as e:
            logger.error(f"SDK: FAILED to capture code context for {func.__name__}: {e}", exc_info=True)
            logger.error(f"SDK: Exception type: {type(e).__name__}")
            logger.error(f"SDK: Exception args: {e.args}")
            logger.warning(f"SDK: Returning empty code context for {func.__name__}")
            return {}

    def _sync_trace_decorator(self, func: Callable, agent_name: str, code_context: Dict[str, Any], **kwargs) -> Callable:
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

            # Add code context to metadata
            if code_context:
                logger.info(f"SDK: === ATTACHING CODE CONTEXT TO TRACE '{agent_name}' ===")
                logger.info(f"SDK: Function: {func.__name__}")
                logger.info(f"SDK: Source code length: {len(code_context.get('source_code', ''))}")
                logger.info(f"SDK: Docstring length: {len(code_context.get('docstring') or '')}")
                logger.info(f"SDK: File path: {code_context.get('file_path')}")

                source_code = code_context.get('source_code', '')
                docstring = code_context.get('docstring', '')

                logger.debug(f"SDK: Attaching source code (first 100 chars): {source_code[:100]}...")
                logger.debug(f"SDK: Attaching docstring: {docstring or 'None'}")

                trace_data.metadata.update({
                    'source_code': source_code,
                    'docstring': docstring,
                    'function_signature': code_context.get('function_signature', ''),
                    'module_name': code_context.get('module_name', ''),
                    'file_path': code_context.get('file_path', ''),
                    'function_name': code_context.get('function_name', '')
                })

                logger.info(f"SDK: Successfully attached code context to trace '{agent_name}'")
            else:
                logger.warning(f"SDK: NO CODE CONTEXT to attach to trace '{agent_name}' (function: {func.__name__})")

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

    def _async_trace_decorator(self, func: Callable, agent_name: str, code_context: Dict[str, Any], **kwargs) -> Callable:
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

            # Add code context to metadata
            if code_context:
                logger.info(f"SDK: === ATTACHING CODE CONTEXT TO TRACE '{agent_name}' ===")
                logger.info(f"SDK: Function: {func.__name__}")
                logger.info(f"SDK: Source code length: {len(code_context.get('source_code', ''))}")
                logger.info(f"SDK: Docstring length: {len(code_context.get('docstring') or '')}")
                logger.info(f"SDK: File path: {code_context.get('file_path')}")

                source_code = code_context.get('source_code', '')
                docstring = code_context.get('docstring', '')

                logger.debug(f"SDK: Attaching source code (first 100 chars): {source_code[:100]}...")
                logger.debug(f"SDK: Attaching docstring: {docstring or 'None'}")

                trace_data.metadata.update({
                    'source_code': source_code,
                    'docstring': docstring,
                    'function_signature': code_context.get('function_signature', ''),
                    'module_name': code_context.get('module_name', ''),
                    'file_path': code_context.get('file_path', ''),
                    'function_name': code_context.get('function_name', '')
                })

                logger.info(f"SDK: Successfully attached code context to trace '{agent_name}'")
            else:
                logger.warning(f"SDK: NO CODE CONTEXT to attach to trace '{agent_name}' (function: {func.__name__})")

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

    def span(self, operation_name: str, **kwargs):
        """Context manager for manual span creation.

        This is used as a context manager for code blocks that need manual span control.
        For function decoration, use @vaquero.trace() instead.

        Args:
            operation_name: Name of the operation/agent
            **kwargs: Additional options (tags, metadata, etc.)

        Returns:
            Context manager yielding TraceData instance

        Examples:
            # As context manager (for code blocks)
            with sdk.span("custom_operation") as span:
                span.set_attribute("batch_size", 100)
                # Your code here
        """
        logger.info(f"SDK: === SPAN CONTEXT MANAGER CALLED ===")
        logger.info(f"SDK: Operation name: {operation_name}")
        logger.info(f"SDK: Additional kwargs: {kwargs}")

        return self._span_context_manager(operation_name, **kwargs)
    
    @contextmanager
    def _span_context_manager(self, operation_name: str, **kwargs) -> Generator[TraceData, None, None]:
        """Internal context manager implementation for span."""
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
    
    def _sync_span_decorator(self, func: Callable, operation_name: str, code_context: Dict[str, Any], **kwargs) -> Callable:
        """Synchronous span decorator implementation."""
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            # Create trace data as a child span
            trace_data = create_child_span(
                agent_name=operation_name,
                function_name=func.__name__,
                tags=kwargs.get("tags", {}),
                metadata=kwargs.get("metadata", {})
            )
            
            # Set the name field to match agent_name
            trace_data.name = operation_name
            
            # Add code context to metadata
            if code_context:
                trace_data.metadata.update({
                    'source_code': code_context.get('source_code', ''),
                    'docstring': code_context.get('docstring', ''),
                    'function_signature': code_context.get('function_signature', ''),
                    'module_name': code_context.get('module_name', ''),
                    'file_path': code_context.get('file_path', ''),
                    'function_name': code_context.get('function_name', '')
                })
            
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
                    # Send trace data
                    self._send_trace(trace_data)
        
        return wrapper
    
    def _async_span_decorator(self, func: Callable, operation_name: str, code_context: Dict[str, Any], **kwargs) -> Callable:
        """Asynchronous span decorator implementation."""
        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs):
            # Create trace data as a child span
            trace_data = create_child_span(
                agent_name=operation_name,
                function_name=func.__name__,
                tags=kwargs.get("tags", {}),
                metadata=kwargs.get("metadata", {})
            )
            
            # Set the name field to match agent_name
            trace_data.name = operation_name
            
            # Add code context to metadata
            if code_context:
                trace_data.metadata.update({
                    'source_code': code_context.get('source_code', ''),
                    'docstring': code_context.get('docstring', ''),
                    'function_signature': code_context.get('function_signature', ''),
                    'module_name': code_context.get('module_name', ''),
                    'file_path': code_context.get('file_path', ''),
                    'function_name': code_context.get('function_name', '')
                })
            
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
                    # Send trace data
                    self._send_trace(trace_data)
        
        return wrapper

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
        """Send trace data to the trace collector.

        Args:
            trace_data: TraceData instance to send
        """
        if not self._enabled:
            logger.debug(f"SDK: Trace sending disabled for {trace_data.agent_name}")
            return

        logger.info(f"SDK: === SENDING TRACE TO TRACE COLLECTOR ===")
        logger.info(f"SDK: Agent name: {trace_data.agent_name}")
        logger.info(f"SDK: Trace ID: {trace_data.trace_id}")
        logger.info(f"SDK: Span ID: {trace_data.span_id}")
        logger.info(f"SDK: Status: {trace_data.status}")
        logger.info(f"SDK: Duration: {trace_data.duration_ms}ms")

        try:
            # Add environment and mode information
            trace_data.metadata["environment"] = self.config.environment
            trace_data.metadata["mode"] = self.config.mode
            trace_data.metadata["sdk_version"] = "0.1.0"

            # Check for source code in metadata
            source_code = trace_data.metadata.get("source_code")
            if source_code:
                logger.info(f"SDK: TRACE CONTAINS SOURCE CODE (length: {len(source_code)})")
                logger.debug(f"SDK: Source code preview: {source_code[:100]}...")
            else:
                logger.warning(f"SDK: TRACE DOES NOT CONTAIN SOURCE CODE")

            docstring = trace_data.metadata.get("docstring")
            if docstring:
                logger.info(f"SDK: TRACE CONTAINS DOCSTRING (length: {len(docstring)})")
                logger.debug(f"SDK: Docstring preview: {docstring[:100]}...")
            else:
                logger.info(f"SDK: TRACE DOES NOT CONTAIN DOCSTRING")

            # Convert TraceData to dictionary for trace collector
            span_data = trace_data.to_dict()
            
            # Process the span with the trace collector
            self._trace_collector.process_span(span_data)
            
            # NOTE: We do NOT automatically end traces here.
            # Traces are ended explicitly by:
            # 1. The span context manager (for @vaquero.trace decorators)
            # 2. The LangChain callback handler (when the workflow completes)
            # 3. Manual calls to vaquero.end_trace() by the user

        except Exception as e:
            logger.error(f"SDK: FAILED to send trace data for {trace_data.agent_name}: {e}", exc_info=True)

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
        if self._trace_collector:
            # TraceCollectorV2 handles flushing automatically
            logger.debug("Trace collector handles flushing automatically")

    def shutdown(self, timeout: Optional[float] = None):
        """Shutdown the SDK and flush remaining traces.

        Args:
            timeout: Maximum time to wait for shutdown to complete
        """
        # Stop auto-instrumentation
        self._stop_auto_instrumentation()
        
        # Shutdown trace collector
        if self._trace_collector:
            self._trace_collector.shutdown()
            self._trace_collector = None
            
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


def get_global_instance() -> VaqueroSDK:
    """Get the global SDK instance.

    Returns:
        Global SDK instance

    Raises:
        RuntimeError: If no global instance has been created
    """
    if _sdk_instance is None:
        raise RuntimeError(
            "No global SDK instance available. "
            "Call vaquero.init() or VaqueroSDK.initialize() first."
        )
    return _sdk_instance