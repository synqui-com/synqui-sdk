"""Data models for the CognitionFlow SDK."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class SpanStatus(Enum):
    """Span status enumeration."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TraceData:
    """Trace data structure.

    This represents a single trace event that will be sent to the CognitionFlow API.
    It contains all the information about a function call or operation.

    Attributes:
        trace_id: Unique identifier for the trace
        span_id: Unique identifier for this span
        parent_span_id: Parent span ID for nested calls
        agent_name: Name of the agent/component being traced
        function_name: Name of the function being traced
        start_time: When the operation started
        end_time: When the operation ended
        duration_ms: Duration in milliseconds
        status: Current status of the span
        inputs: Function inputs (if captured)
        outputs: Function outputs (if captured)
        error: Error information (if any)
        tags: Key-value tags for filtering and grouping
        metadata: Additional metadata
        attributes: Custom attributes set during execution
        input_tokens: Number of input tokens (for LLM calls)
        output_tokens: Number of output tokens (for LLM calls)
        total_tokens: Total number of tokens
        cost: Estimated cost of the operation
        model_name: Name of the model used (for LLM calls)
        model_provider: Provider of the model (openai, anthropic, etc.)
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    agent_name: str = ""
    function_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.RUNNING
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Token counting fields
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model_name: Optional[str] = None
    model_provider: Optional[str] = None

    # Prompt capture fields (explicit, provider-agnostic)
    system_prompt: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_parameters: Optional[Dict[str, Any]] = None
    prompt_hash: Optional[str] = None  # hash of system_prompt for dedup/lookup

    def set_attribute(self, key: str, value: Any):
        """Set a custom attribute on the span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def set_tag(self, key: str, value: str):
        """Set a tag on the span.

        Args:
            key: Tag key
            value: Tag value
        """
        self.tags[key] = value

    def set_error(self, error: Exception):
        """Set error information on the span.

        Args:
            error: Exception that occurred
        """
        import traceback
        self.status = SpanStatus.FAILED
        self.error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }

    def finish(self, status: Optional[SpanStatus] = None):
        """Mark the span as finished.

        Args:
            status: Final status (defaults to COMPLETED if no error)
        """
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if status:
            self.status = status
        elif self.status == SpanStatus.RUNNING:
            self.status = SpanStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace data to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "agent_name": self.agent_name,
            "function_name": self.function_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value if isinstance(self.status, SpanStatus) else self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "tags": self.tags,
            "metadata": self.metadata,
            "attributes": self.attributes,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            # Prompt fields
            "system_prompt": self.system_prompt,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_parameters": self.prompt_parameters,
            "prompt_hash": self.prompt_hash,
        }


@dataclass
class BatchPayload:
    """Payload for batched trace events.

    This represents a batch of trace events that will be sent to the API
    in a single request.

    Attributes:
        project_id: Project ID for all traces in the batch
        events: List of trace events
        timestamp: When the batch was created
        sdk_version: Version of the SDK that created the batch
        environment: Environment name
    """

    project_id: str
    events: list[Dict[str, Any]]
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    sdk_version: str = "0.1.0"
    environment: str = "development"

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch payload to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "project_id": self.project_id,
            "events": self.events,
            "timestamp": self.timestamp,
            "sdk_version": self.sdk_version,
            "environment": self.environment,
        }