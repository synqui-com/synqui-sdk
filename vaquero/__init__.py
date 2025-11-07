"""Vaquero Python SDK for observability and tracing.

This SDK provides easy instrumentation for Python applications to capture
trace data and send it to the Vaquero platform.

Basic Usage (Recommended):
    import vaquero

    # Simple initialization with sensible defaults
    vaquero.init(api_key="your-api-key")

    # Trace a function
    @vaquero.trace(agent_name="data_processor")
    def process_data(data):
        return {"processed": data}

Advanced Usage (Legacy):
    # Advanced configuration (still supported)
    vaquero.configure(
        api_key="your-api-key",
        project_id="your-project-id",
        capture_inputs=True
    )

Manual Tracing:
    # Manual span creation
    with vaquero.span("custom_operation") as span:
        span.set_attribute("key", "value")
        # Your code here
"""

__version__ = "0.1.0"
__author__ = "Vaquero Team"
__email__ = "team@vaquero.com"

from typing import Optional

from .sdk import VaqueroSDK, get_global_instance
from .config import SDKConfig, configure, configure_from_env, init
from .context import get_current_span
from .decorators import trace as _trace_decorator
from .workflow import workflow, Workflow
from .chat_session import ChatSession, ChatSessionManager, create_chat_session, get_session_manager
from .utils import json_loads

# Optional integrations (only import if dependencies are available)
try:
    from .langchain import VaqueroCallbackHandler, get_vaquero_handler
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

try:
    from .langgraph import VaqueroLangGraphHandler, get_vaquero_langgraph_handler, create_langgraph_config
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

# Default SDK instance
_default_sdk: Optional[VaqueroSDK] = None


def get_default_sdk() -> VaqueroSDK:
    """Get the default SDK instance."""
    global _default_sdk
    if _default_sdk is None:
        raise RuntimeError("SDK not configured. Call vaquero.configure() first.")
    return _default_sdk


def set_default_sdk(sdk: VaqueroSDK):
    """Set the default SDK instance."""
    global _default_sdk
    _default_sdk = sdk


# Convenience functions that delegate to the default SDK
def trace(agent_name: str, **kwargs):
    """Trace decorator using the default SDK instance."""
    return get_default_sdk().trace(agent_name, **kwargs)


def span(operation_name: str, **kwargs):
    """Context manager for manual span creation using the default SDK instance.

    This is used for code blocks that need manual span control.
    For function decoration, use @vaquero.trace() instead.

    Args:
        operation_name: Name of the operation/agent
        **kwargs: Additional options (tags, metadata, etc.)

    Returns:
        Context manager yielding TraceData instance

    Example:
        # As context manager (for code blocks)
        with vaquero.span("custom_operation") as span:
            span.set_attribute("batch_size", 100)
            # Your code here
    """
    return get_default_sdk().span(operation_name, **kwargs)




def flush():
    """Flush pending traces using the default SDK instance."""
    get_default_sdk().flush()


def shutdown():
    """Shutdown the default SDK instance."""
    if _default_sdk:
        _default_sdk.shutdown()


def start_chat_session(
    name: Optional[str] = None,
    session_type: str = "chat",
    timeout_minutes: int = 30,
    max_duration_minutes: int = 240,
    metadata: Optional[dict] = None
) -> ChatSession:
    """Start a new chat session.

    Args:
        name: Human-readable name for the session
        session_type: Type of session ('chat', 'pipeline', 'workflow')
        timeout_minutes: Minutes of inactivity before session timeout
        max_duration_minutes: Maximum session duration in minutes
        metadata: Additional session metadata

    Returns:
        The created ChatSession instance

    Example:
        import vaquero

        # Initialize SDK first
        vaquero.init(api_key="your-api-key")

        # Create a chat session for conversational AI
        session = vaquero.start_chat_session(
            name="pdf_chat_assistant",
            session_type="chat",
            timeout_minutes=30
        )

        # Use the session with LangGraph handler
        handler = vaquero.get_vaquero_langgraph_handler_with_session(session)
    """
    return create_chat_session(
        name=name,
        session_type=session_type,
        timeout_minutes=timeout_minutes,
        max_duration_minutes=max_duration_minutes,
        metadata=metadata
    )


# Build __all__ list dynamically to include optional integrations
__all__ = [
    "init",
    "configure",
    "configure_from_env",
    "trace",
    "span",
    "workflow",
    "flush",
    "shutdown",
    "get_current_span",
    "get_global_instance",
    "SDKConfig",
    "VaqueroSDK",
    "Workflow",
    "ChatSession",
    "ChatSessionManager",
    "create_chat_session",
    "start_chat_session",
    "get_session_manager",
    "get_vaquero_langgraph_handler_with_session",
    "json_loads",
    "__version__",
]

# Add LangChain integration if available
if _LANGCHAIN_AVAILABLE:
    __all__.extend(["VaqueroCallbackHandler", "get_vaquero_handler"])

# Add LangGraph integration if available
if _LANGGRAPH_AVAILABLE:
    __all__.extend(["VaqueroLangGraphHandler", "get_vaquero_langgraph_handler", "create_langgraph_config"])


# Add session-aware handler function
def get_vaquero_langgraph_handler_with_session(session: ChatSession):
    """Get a VaqueroLangGraphHandler configured for a specific chat session.

    Args:
        session: The ChatSession instance to associate with the handler

    Returns:
        VaqueroLangGraphHandler configured for the session

    Example:
        import vaquero

        # Create session
        session = vaquero.start_chat_session("my_chat")

        # Get session-aware handler
        handler = vaquero.get_vaquero_langgraph_handler_with_session(session)
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph integration not available. Install langgraph to use this feature.")

    from .langgraph import VaqueroLangGraphHandler
    return VaqueroLangGraphHandler(session=session)

# Register automatic shutdown for the global SDK instance
import atexit
atexit.register(lambda: shutdown() if _default_sdk else None)