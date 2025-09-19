"""CognitionFlow Python SDK for observability and tracing.

This SDK provides easy instrumentation for Python applications to capture
trace data and send it to the CognitionFlow platform.

Basic Usage:
    import cognitionflow

    # Configure the SDK
    cognitionflow.configure(
        api_key="your-api-key",
        project_id="your-project-id"
    )

    # Trace a function
    @cognitionflow.trace(agent_name="data_processor")
    def process_data(data):
        return {"processed": data}

Advanced Usage:
    # Manual span creation
    with cognitionflow.span("custom_operation") as span:
        span.set_attribute("key", "value")
        # Your code here
"""

__version__ = "0.1.0"
__author__ = "CognitionFlow Team"
__email__ = "team@cognitionflow.com"

from typing import Optional

from .sdk import CognitionFlowSDK
from .config import SDKConfig, configure, configure_from_env
from .context import get_current_span
from .decorators import trace as _trace_decorator

# Default SDK instance
_default_sdk: Optional[CognitionFlowSDK] = None


def get_default_sdk() -> CognitionFlowSDK:
    """Get the default SDK instance."""
    global _default_sdk
    if _default_sdk is None:
        raise RuntimeError("SDK not configured. Call cognitionflow.configure() first.")
    return _default_sdk


def set_default_sdk(sdk: CognitionFlowSDK):
    """Set the default SDK instance."""
    global _default_sdk
    _default_sdk = sdk


# Convenience functions that delegate to the default SDK
def trace(agent_name: str, **kwargs):
    """Trace decorator using the default SDK instance."""
    return get_default_sdk().trace(agent_name, **kwargs)


def span(operation_name: str, **kwargs):
    """Span context manager using the default SDK instance."""
    return get_default_sdk().span(operation_name, **kwargs)


def flush():
    """Flush pending traces using the default SDK instance."""
    get_default_sdk().flush()


def shutdown():
    """Shutdown the default SDK instance."""
    if _default_sdk:
        _default_sdk.shutdown()


__all__ = [
    "configure",
    "configure_from_env",
    "trace",
    "span",
    "flush",
    "shutdown",
    "get_current_span",
    "SDKConfig",
    "CognitionFlowSDK",
    "__version__",
]