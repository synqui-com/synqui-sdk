"""Safe serialization utilities for the CognitionFlow SDK."""

import json
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# Maximum depth for nested object serialization
MAX_DEPTH = 10

# Maximum number of items in collections
MAX_COLLECTION_SIZE = 100

# Maximum string length
MAX_STRING_LENGTH = 10000


def safe_serialize(obj: Any, depth: int = 0) -> Any:
    """Safely serialize an object for transmission.

    This function handles various Python types and converts them to
    JSON-serializable formats while preventing infinite recursion
    and limiting the size of serialized data.

    Args:
        obj: Object to serialize
        depth: Current recursion depth

    Returns:
        JSON-serializable representation of the object
    """
    # Prevent infinite recursion
    if depth > MAX_DEPTH:
        return f"<max_depth_exceeded:{type(obj).__name__}>"

    # Handle None
    if obj is None:
        return None

    # Handle primitive types
    if isinstance(obj, (bool, int, float)):
        return obj

    # Handle strings with length limit
    if isinstance(obj, str):
        if len(obj) > MAX_STRING_LENGTH:
            return obj[:MAX_STRING_LENGTH] + "..."
        return obj

    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, date):
        return obj.isoformat()

    # Handle Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return f"<bytes:{len(obj)}>"

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [
            safe_serialize(item, depth + 1)
            for item in obj[:MAX_COLLECTION_SIZE]
        ]

    # Handle sets
    if isinstance(obj, set):
        return [
            safe_serialize(item, depth + 1)
            for item in list(obj)[:MAX_COLLECTION_SIZE]
        ]

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for i, (key, value) in enumerate(obj.items()):
            if i >= MAX_COLLECTION_SIZE:
                result["..."] = f"<truncated:{len(obj) - MAX_COLLECTION_SIZE}_more_items>"
                break

            # Convert key to string if necessary
            str_key = str(key) if not isinstance(key, str) else key
            if len(str_key) > 100:  # Limit key length
                str_key = str_key[:100] + "..."

            result[str_key] = safe_serialize(value, depth + 1)

        return result

    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        try:
            return safe_serialize(obj.__dict__, depth + 1)
        except Exception:
            pass

    # Handle objects with common attributes
    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__
        module_name = getattr(obj.__class__, '__module__', 'unknown')

        # Try to get a string representation
        try:
            str_repr = str(obj)
            if len(str_repr) > 200:
                str_repr = str_repr[:200] + "..."
            return f"<{module_name}.{class_name}:{str_repr}>"
        except Exception:
            return f"<{module_name}.{class_name}>"

    # Fallback for unknown types
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def serialize_for_api(data: Dict[str, Any]) -> str:
    """Serialize data for API transmission.

    Args:
        data: Dictionary to serialize

    Returns:
        JSON string representation

    Raises:
        ValueError: If serialization fails
    """
    try:
        # First pass: safe serialize all values
        safe_data = safe_serialize(data)

        # Second pass: JSON encode
        return json.dumps(safe_data, ensure_ascii=False, separators=(',', ':'))

    except Exception as e:
        logger.error(f"Failed to serialize data for API: {e}")
        raise ValueError(f"Serialization failed: {e}")


def truncate_large_objects(obj: Any, max_size_bytes: int = 1024 * 1024) -> Any:
    """Truncate objects that are too large for transmission.

    Args:
        obj: Object to check and potentially truncate
        max_size_bytes: Maximum size in bytes

    Returns:
        Original object or truncated version
    """
    try:
        # Quick size check using JSON serialization
        serialized = json.dumps(safe_serialize(obj))
        size_bytes = len(serialized.encode('utf-8'))

        if size_bytes <= max_size_bytes:
            return obj

        # Object is too large, truncate it
        logger.warning(f"Object too large ({size_bytes} bytes), truncating")

        # If it's a collection, truncate the collection
        if isinstance(obj, (list, tuple)):
            # Binary search to find the right size
            left, right = 0, len(obj)
            best_size = 0

            while left <= right:
                mid = (left + right) // 2
                truncated = obj[:mid]
                test_size = len(json.dumps(safe_serialize(truncated)).encode('utf-8'))

                if test_size <= max_size_bytes:
                    best_size = mid
                    left = mid + 1
                else:
                    right = mid - 1

            return obj[:best_size]

        elif isinstance(obj, dict):
            # Truncate dictionary by removing items
            items = list(obj.items())
            left, right = 0, len(items)
            best_size = 0

            while left <= right:
                mid = (left + right) // 2
                truncated = dict(items[:mid])
                test_size = len(json.dumps(safe_serialize(truncated)).encode('utf-8'))

                if test_size <= max_size_bytes:
                    best_size = mid
                    left = mid + 1
                else:
                    right = mid - 1

            return dict(items[:best_size])

        elif isinstance(obj, str):
            # Truncate string to fit within size limit
            target_chars = max_size_bytes // 4  # Rough estimate for UTF-8
            return obj[:target_chars] + "..."

        else:
            # For other types, return a placeholder
            return f"<truncated:{type(obj).__name__}>"

    except Exception as e:
        logger.error(f"Failed to truncate object: {e}")
        return f"<truncation_failed:{type(obj).__name__}>"