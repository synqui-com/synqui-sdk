"""Utility functions for Vaquero SDK.

This module provides convenient wrappers around common operations that
automatically capture errors to trace when there's an active trace context.
"""

import json
from typing import Any, Optional

from .context import get_current_span, create_child_span
from .sdk import get_global_instance
from .models import SpanStatus


def json_loads(s: str, **kwargs) -> Any:
    """Wrapper around json.loads that automatically captures errors to trace.
    
    This is a drop-in replacement for json.loads() that will automatically
    create error spans if JSON parsing fails and there's an active trace context.
    
    Args:
        s: JSON string to parse
        **kwargs: Additional arguments passed to json.loads() (e.g., strict=False)
        
    Returns:
        Parsed JSON object
        
    Raises:
        JSONDecodeError: If JSON parsing fails (same as json.loads)
        
    Example:
        import vaquero
        
        # Instead of: result = json.loads(data)
        result = vaquero.json_loads(data)
        
        # With options
        result = vaquero.json_loads(data, strict=False)
    """
    try:
        return json.loads(s, **kwargs)
    except json.JSONDecodeError as e:
        # Check if we have an active trace context
        current_span = get_current_span()
        if current_span:
            # Create error span as child of current span
            error_span = create_child_span(
                agent_name="json_parse_error",
                function_name="json_loads",
                metadata={
                    "error_context": "json_parsing",
                    "error_type": "JSONDecodeError"
                }
            )
            
            # Set error details
            error_span.set_error(e)
            
            # Add input preview (truncated for safety)
            input_preview = str(s)[:500]
            error_span.inputs = {
                "json_string_preview": input_preview,
                "json_string_length": len(s)
            }
            
            # Finish the error span
            error_span.finish(SpanStatus.FAILED)
            
            # Send to trace collector
            sdk = get_global_instance()
            if sdk:
                sdk._send_trace(error_span)
        
        # Re-raise the exception so calling code can handle it
        raise

