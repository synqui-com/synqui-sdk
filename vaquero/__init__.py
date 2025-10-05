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

from .sdk import VaqueroSDK
from .config import SDKConfig, configure, configure_from_env, init
from .context import get_current_span
from .decorators import trace as _trace_decorator
from .workflow import workflow, Workflow

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
    """Span context manager using the default SDK instance."""
    return get_default_sdk().span(operation_name, **kwargs)


def async_span(operation_name: str, **kwargs):
    """Async span context manager using the default SDK instance."""
    return get_default_sdk().async_span(operation_name, **kwargs)


def flush():
    """Flush pending traces using the default SDK instance."""
    get_default_sdk().flush()


def shutdown():
    """Shutdown the default SDK instance."""
    if _default_sdk:
        _default_sdk.shutdown()


__all__ = [
    "init",
    "configure",
    "configure_from_env",
    "trace",
    "span",
    "async_span",
    "workflow",
    "flush",
    "shutdown",
    "get_current_span",
    "SDKConfig",
    "VaqueroSDK",
    "Workflow",
    "__version__",
]

# Register automatic shutdown for the global SDK instance
import atexit
atexit.register(lambda: shutdown() if _default_sdk else None)