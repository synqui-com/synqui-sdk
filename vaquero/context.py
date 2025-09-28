"""Context management for nested traces and spans."""

import threading
from contextlib import contextmanager
from typing import Optional, Generator
from .models import TraceData


class TraceContext:
    """Thread-local context for managing trace hierarchy.

    This class manages the current trace context using thread-local storage,
    allowing for proper nesting of spans and traces.
    """

    def __init__(self):
        self._local = threading.local()

    @property
    def current_span(self) -> Optional[TraceData]:
        """Get the current active span."""
        return getattr(self._local, "current_span", None)

    @property
    def current_trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        span = self.current_span
        return span.trace_id if span else None

    @property
    def current_span_id(self) -> Optional[str]:
        """Get the current span ID."""
        span = self.current_span
        return span.span_id if span else None

    def set_current_span(self, span: Optional[TraceData]):
        """Set the current active span."""
        self._local.current_span = span

    @contextmanager
    def span_context(self, span: TraceData) -> Generator[TraceData, None, None]:
        """Context manager for setting the current span.

        Args:
            span: The span to set as current

        Yields:
            The span instance
        """
        previous_span = self.current_span
        self.set_current_span(span)
        try:
            yield span
        finally:
            self.set_current_span(previous_span)

    def create_child_span(
        self,
        agent_name: str,
        function_name: str = "",
        **kwargs
    ) -> TraceData:
        """Create a child span of the current span.

        Args:
            agent_name: Name of the agent for the new span
            function_name: Name of the function for the new span
            **kwargs: Additional arguments for the span

        Returns:
            New TraceData instance with proper parent relationship
        """
        current = self.current_span
        trace_id = current.trace_id if current else None
        parent_span_id = current.span_id if current else None

        span = TraceData(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            function_name=function_name,
            **kwargs
        )

        if parent_span_id:
            span.inputs = span.inputs or {}
            span.inputs.setdefault("parent_span_id", parent_span_id)
            span.metadata.setdefault("parent_span_id", parent_span_id)

        # If this is the first span in a trace, it becomes the trace ID
        if not trace_id:
            span.trace_id = span.span_id

        return span


# Global trace context instance
_trace_context = TraceContext()


def get_current_span() -> Optional[TraceData]:
    """Get the current active span.

    Returns:
        Current TraceData instance or None if no span is active
    """
    return _trace_context.current_span


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID.

    Returns:
        Current trace ID or None if no trace is active
    """
    return _trace_context.current_trace_id


def get_current_span_id() -> Optional[str]:
    """Get the current span ID.

    Returns:
        Current span ID or None if no span is active
    """
    return _trace_context.current_span_id


def set_current_span(span: Optional[TraceData]):
    """Set the current active span.

    Args:
        span: The span to set as current, or None to clear
    """
    _trace_context.set_current_span(span)


@contextmanager
def span_context(span: TraceData) -> Generator[TraceData, None, None]:
    """Context manager for setting the current span.

    Args:
        span: The span to set as current

    Yields:
        The span instance
    """
    with _trace_context.span_context(span) as s:
        yield s


def create_child_span(
    agent_name: str,
    function_name: str = "",
    **kwargs
) -> TraceData:
    """Create a child span of the current span.

    Args:
        agent_name: Name of the agent for the new span
        function_name: Name of the function for the new span
        **kwargs: Additional arguments for the span

    Returns:
        New TraceData instance with proper parent relationship
    """
    return _trace_context.create_child_span(agent_name, function_name, **kwargs)