"""Convenience decorators for the Vaquero SDK."""

from typing import Callable


def trace(agent_name: str, capture_code: bool = True, **kwargs) -> Callable:
    """Convenience decorator that uses the default SDK instance.

    This is a shorthand for sdk.trace() that uses the globally
    configured SDK instance.

    Args:
        agent_name: Name of the agent/component being traced
        capture_code: Whether to capture source code and docstring for analysis
        **kwargs: Additional options (tags, metadata, etc.)

    Returns:
        Decorated function

    Example:
        import vaquero

        vaquero.configure(api_key="...", project_id="...")

        @vaquero.trace("data_processor", capture_code=True)
        def process_data(data):
            \"\"\"
            Process data with expected performance of 1-5 seconds.
            \"\"\"
            return {"processed": data}
    """
    # Import here to avoid circular import
    from . import get_default_sdk
    return get_default_sdk().trace(agent_name, capture_code=capture_code, **kwargs)