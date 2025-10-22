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
        capture_inputs=True,
        debug=True
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
    "__version__",
]

# Add LangChain integration if available
if _LANGCHAIN_AVAILABLE:
    __all__.extend(["VaqueroCallbackHandler", "get_vaquero_handler"])

# Add LangGraph integration if available
if _LANGGRAPH_AVAILABLE:
    __all__.extend(["VaqueroLangGraphHandler", "get_vaquero_langgraph_handler", "create_langgraph_config"])

# Register automatic shutdown for the global SDK instance
import atexit
atexit.register(lambda: shutdown() if _default_sdk else None)